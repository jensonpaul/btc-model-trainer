//! Kraken historical trades fetcher.
//!
//! # Endpoint
//!
//! ```text
//! GET https://api.kraken.com/0/public/Trades?pair=XBTUSD&since={since_ns}
//! ```
//!
//! # Response shape
//!
//! ```json
//! {
//!   "error": [],
//!   "result": {
//!     "XXBTZUSD": [
//!       [price, volume, time, "b"|"s", "m"|"l", misc, trade_id],
//!       ...
//!     ],
//!     "last": "1234567890123456789"
//!   }
//! }
//! ```
//!
//! Fields per trade (all strings in the wire format except `time`):
//!
//! | Index | Name   | Notes                                          |
//! |-------|--------|------------------------------------------------|
//! | 0     | price  | f64 string, USD                                |
//! | 1     | volume | f64 string, BTC                                |
//! | 2     | time   | f64 seconds since epoch, microsecond precision |
//! | 3     | side   | "b" (buy) or "s" (sell)                        |
//! | 4     | type   | "m" (market) or "l" (limit) — ignored          |
//! | 5     | misc   | ignored                                         |
//! | 6     | trade_id | ignored                                       |
//!
//! Reference: https://docs.kraken.com/api/docs/rest-api/get-recent-trades
//!
//! # Why not the OHLC endpoint
//!
//! Kraken's `OHLC` endpoint returns only the most recent ~720 candles
//! regardless of the `since` parameter, so it cannot be used to backfill
//! history. The `Trades` endpoint supports full-history pagination via a
//! cursor (`since` / `last`), which we use here. As a bonus this gives us
//! genuine per-trade rows (including taker `side`) instead of OHLC-derived
//! synthetic rows.
//!
//! # Pagination
//!
//! `since` is a nanosecond-precision timestamp. The `last` field in the
//! result is the cursor to pass as `since` on the next request. The API
//! returns up to 1000 trades per call.

use std::path::PathBuf;

use anyhow::{Context, Result};
use async_trait::async_trait;
use chrono::{Datelike, TimeZone, Utc};
use serde_json::Value;
use tracing::{info, warn};

use crate::historical::{
    cursor::{self, Cursor as FetchCursor},
    exchange_floor,
    schema::TradeRow,
    store::DataStore,
};
use super::{build_client, rate_limit_sleep, HistoricalFetcher};

const TRADES_URL: &str = "https://api.kraken.com/0/public/Trades";

pub struct KrakenFetcher {
    data_root: PathBuf,
    client:    reqwest::Client,
}

impl KrakenFetcher {
    pub fn new(data_root: PathBuf) -> Self {
        Self { data_root, client: build_client() }
    }
}

#[async_trait]
impl HistoricalFetcher for KrakenFetcher {
    fn exchange(&self) -> &'static str { "kraken" }

    /// Pages are accumulated **in memory** for the calendar month currently
    /// being fetched. Each shard is written exactly once — either when the
    /// data crosses into the next month, or at the end of the run (for the
    /// in-progress month). This avoids the previous per-page
    /// read-existing-shard → merge → dedup → resort → rewrite cycle, which
    /// dominated disk I/O for active months.
    ///
    /// The cursor is advanced in memory on every page (so progress is never
    /// lost if a later page errors out) but is only persisted to disk at the
    /// same points the shard is written, i.e. at most once per month per run.
    async fn fetch_range(&self, resume_from_micros: i64, up_to_micros: i64) -> Result<u64> {
        let store       = DataStore::new(&self.data_root, "kraken")?;
        let cursor_path = cursor::cursor_path(&self.data_root, "kraken");
        let mut cur     = FetchCursor::load(&cursor_path)?;

        let floor = exchange_floor("kraken").timestamp_micros();
        let start_micros = resume_from_micros.max(floor);

        // Kraken `since` for the Trades endpoint is in **nanoseconds**.
        let mut since_ns: i64 = start_micros * 1_000;
        let up_to_micros_i64  = up_to_micros;

        let mut total_rows: u64 = 0;

        // In-memory accumulator for the month currently being filled. If the
        // cursor's last write left a *partial* shard for the in-progress
        // month, seed the accumulator from it now — a single read at the
        // start of the run — so the eventual flush (which overwrites the
        // file) doesn't lose that month's previously-written rows.
        let mut month_key: Option<(i32, u32)> = None;
        let mut month_buf: Vec<TradeRow> = Vec::new();

        if let Some(c) = cur.last_ts_micros {
            let dt = Utc.timestamp_micros(c).single().context("invalid cursor timestamp")?;
            let (y, m) = (dt.year(), dt.month());
            let path = store.shard_path(y, m)?;
            if path.exists() {
                let existing = store.read_shard_for_seed(&path)
                    .with_context(|| format!("seeding from existing shard {y}-{m:02}"))?;
                if !existing.is_empty() {
                    month_key = Some((y, m));
                    month_buf = existing;
                    info!(exchange = "kraken", year = y, month = m, rows = month_buf.len(),
                          "seeded in-progress month from existing shard");
                }
            }
        }

        // Flush the accumulated month to disk (one write_shard call) and
        // persist the cursor up to the last row in the buffer.
        let flush = |month_buf: &mut Vec<TradeRow>,
                      cur: &mut FetchCursor,
                      total_rows: &mut u64| -> Result<()> {
            if month_buf.is_empty() { return Ok(()); }
            let n = month_buf.len() as u64;
            let last_ts = month_buf.last().unwrap().ts_micros;
            store.write_batch(month_buf)
                .context("writing Kraken shard")?;
            cur.advance(last_ts, n);
            cur.save(&cursor_path)?;
            *total_rows += n;
            info!(exchange = "kraken", rows = n, last_ts, "month shard written");
            month_buf.clear();
            Ok(())
        };

        loop {
            let url = format!("{TRADES_URL}?pair=XBTUSD&since={since_ns}");
            info!(exchange = "kraken", since_ns, "fetching trades page");

            let resp = self.client.get(&url).send().await;
            let json: Value = match resp {
                Err(e) => {
                    warn!(exchange = "kraken", "request error: {e}");
                    rate_limit_sleep("kraken").await;
                    continue;
                }
                Ok(r) if !r.status().is_success() => {
                    warn!(exchange = "kraken", "HTTP {}", r.status());
                    rate_limit_sleep("kraken").await;
                    continue;
                }
                Ok(r) => r.json().await.context("parsing Kraken JSON")?,
            };

            // Check for API-level errors.
            if let Some(errors) = json["error"].as_array() {
                if !errors.is_empty() {
                    warn!(exchange = "kraken", "API error: {:?}", errors);
                    break;
                }
            }

            let result = match json["result"].as_object() {
                Some(r) => r,
                None    => {
                    warn!(exchange = "kraken", "unexpected result shape");
                    break;
                }
            };

            // The trades array is keyed by Kraken's internal pair name
            // (e.g. "XXBTZUSD"), which we don't hardcode - find the first
            // non-"last" array value.
            let trades = match result
                .iter()
                .find_map(|(k, v)| if k != "last" { v.as_array() } else { None })
            {
                Some(t) => t,
                None    => {
                    warn!(exchange = "kraken", "unexpected result shape (no trades array)");
                    break;
                }
            };

            let last_str = result.get("last").and_then(|v| v.as_str());
            let next_since: i64 = match last_str.and_then(|s| s.parse().ok()) {
                Some(n) => n,
                None    => {
                    warn!(exchange = "kraken", "missing/invalid 'last' cursor");
                    break;
                }
            };

            if trades.is_empty() { break; }

            let mut reached_end = false;
            for trade in trades {
                let arr = match trade.as_array() { Some(a) => a, None => continue };
                if arr.len() < 4 { continue; }

                let time_secs: f64 = arr[2].as_f64().unwrap_or(0.0);
                let ts_micros = (time_secs * 1_000_000.0).round() as i64;

                if ts_micros >= up_to_micros_i64 {
                    reached_end = true;
                    break;
                }

                // Skip rows already in cursor.
                if let Some(c) = cur.last_ts_micros {
                    if ts_micros <= c { continue; }
                }

                let price: f64 = arr[0].as_str().unwrap_or_default().parse().unwrap_or(0.0);
                let volume: f64 = arr[1].as_str().unwrap_or_default().parse().unwrap_or(0.0);
                let side_raw = arr[3].as_str().unwrap_or_default();
                let side = match side_raw {
                    "b" => Some("buy"),
                    "s" => Some("sell"),
                    _   => None,
                };

                if price <= 0.0 || volume <= 0.0 { continue; }

                let row = TradeRow {
                    ts_micros,
                    price,
                    quantity: volume,
                    side,
                    exchange: "kraken",
                };

                let dt = Utc.timestamp_micros(row.ts_micros).single().unwrap();
                let key = (dt.year(), dt.month());
                match month_key {
                    None => month_key = Some(key),
                    Some(k) if k != key => {
                        // Crossed a month boundary: flush the completed
                        // month before starting the new one.
                        flush(&mut month_buf, &mut cur, &mut total_rows)?;
                        month_key = Some(key);
                    }
                    _ => {}
                }
                month_buf.push(row);
            }

            if reached_end { break; }

            // Advance pagination cursor.
            if next_since <= since_ns {
                // No progress — we've reached the end of available data.
                break;
            }
            since_ns = next_since;

            rate_limit_sleep("kraken").await;
        }

        // Flush whatever remains for the in-progress month.
        flush(&mut month_buf, &mut cur, &mut total_rows)?;

        Ok(total_rows)
    }
}

