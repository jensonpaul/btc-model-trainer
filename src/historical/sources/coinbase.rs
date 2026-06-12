//! Coinbase Advanced Trade candles fetcher.
//!
//! # Endpoint
//!
//! ```text
//! GET https://api.coinbase.com/api/v3/brokerage/market/products/BTC-USD/candles
//!     ?start={unix_secs}
//!     &end={unix_secs}
//!     &granularity=ONE_MINUTE
//! ```
//!
//! # Response shape
//!
//! ```json
//! {
//!   "candles": [
//!     {
//!       "start":  "1700000000",
//!       "low":    "33950.00",
//!       "high":   "34050.00",
//!       "open":   "34000.00",
//!       "close":  "34010.00",
//!       "volume": "12.345"
//!     },
//!     ...
//!   ]
//! }
//! ```
//!
//! `start` is a UNIX timestamp **string** in seconds.  Candles are returned
//! in **descending** order.
//!
//! Reference: https://docs.cdp.coinbase.com/coinbase-app/advanced-trade-apis/rest/market-data/get-candles
//!
//! # Pagination
//!
//! The endpoint accepts a time window via `start` + `end` and returns at most
//! **300 candles** per call.  At `ONE_MINUTE` granularity that is 300 minutes
//! = 5 hours.  We paginate by sliding the window backward 300 minutes per
//! request until the window start is before our cursor.
//!
//! # Side information
//!
//! Candles are aggregated bars; individual taker side is unavailable.
//! `side` is written as null.
//!
//! # Authentication
//!
//! The public market candles endpoint does not require authentication.
//! (The JWT logic used by the live feed's `coinbase.rs` is not needed here.)

use std::path::PathBuf;

use anyhow::{Context, Result};
use async_trait::async_trait;
use chrono::{Datelike, TimeZone, Utc};
use serde::Deserialize;
use tracing::{info, warn};

use crate::historical::{
    cursor::{self, Cursor as FetchCursor},
    exchange_floor,
    schema::TradeRow,
    store::DataStore,
};
use super::{build_client, rate_limit_sleep, HistoricalFetcher};

const CANDLES_URL: &str =
    "https://api.coinbase.com/api/v3/brokerage/market/products/BTC-USD/candles";

/// Max candles per API call at ONE_MINUTE granularity.
const PAGE_CANDLES: i64 = 300;
/// Window width in seconds for a full page (300 min).
const PAGE_SECS: i64 = PAGE_CANDLES * 60;

#[derive(Debug, Deserialize)]
struct CbResponse {
    candles: Vec<CbCandle>,
}

#[derive(Debug, Deserialize)]
struct CbCandle {
    start:  String,   // UNIX seconds string
    close:  String,   // USD price
    volume: String,   // BTC volume
    // open, high, low fields are present but not used
}

pub struct CoinbaseFetcher {
    data_root: PathBuf,
    client:    reqwest::Client,
}

impl CoinbaseFetcher {
    pub fn new(data_root: PathBuf) -> Self {
        Self { data_root, client: build_client() }
    }
}

#[async_trait]
impl HistoricalFetcher for CoinbaseFetcher {
    fn exchange(&self) -> &'static str { "coinbase" }

    /// Rows are accumulated **in memory** for the calendar month currently
    /// being fetched and written with a single `write_batch` call per month
    /// — either when a window's rows cross into the next month, or at the
    /// end of the run for the in-progress month. The cursor advances on
    /// every window in memory but is only persisted at those flush points.
    async fn fetch_range(&self, resume_from_micros: i64, up_to_micros: i64) -> Result<u64> {
        let store       = DataStore::new(&self.data_root, "coinbase")?;
        let cursor_path = cursor::cursor_path(&self.data_root, "coinbase");
        let mut cur     = FetchCursor::load(&cursor_path)?;

        let floor = exchange_floor("coinbase").timestamp_micros();
        let start_micros = resume_from_micros.max(floor);

        let start_secs: i64 = start_micros  / 1_000_000;
        let end_secs:   i64 = up_to_micros  / 1_000_000;

        let mut total_rows: u64 = 0;

        // In-memory accumulator for the month currently being filled. Seed
        // it from an existing partial shard for the in-progress month, if
        // one is present — a single read at the start of the run.
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
                    info!(exchange = "coinbase", year = y, month = m, rows = month_buf.len(),
                          "seeded in-progress month from existing shard");
                }
            }
        }

        let flush = |month_buf: &mut Vec<TradeRow>,
                      cur: &mut FetchCursor,
                      total_rows: &mut u64| -> Result<()> {
            if month_buf.is_empty() { return Ok(()); }
            let n = month_buf.len() as u64;
            let last_ts = month_buf.last().unwrap().ts_micros;
            store.write_batch(month_buf)
                .context("writing Coinbase shard")?;
            cur.advance(last_ts, n);
            cur.save(&cursor_path)?;
            *total_rows += n;
            info!(exchange = "coinbase", rows = n, last_ts, "month shard written");
            month_buf.clear();
            Ok(())
        };

        // Walk forward in PAGE_SECS windows.
        let mut window_start = start_secs;

        while window_start < end_secs {
            let window_end = (window_start + PAGE_SECS).min(end_secs);

            info!(
                exchange = "coinbase",
                window_start, window_end,
                "fetching candles"
            );

            let resp = self.client
                .get(CANDLES_URL)
                .query(&[
                    ("start",       window_start.to_string()),
                    ("end",         window_end.to_string()),
                    ("granularity", "ONE_MINUTE".to_string()),
                ])
                .send()
                .await;

            let cb: CbResponse = match resp {
                Err(e) => {
                    warn!(exchange = "coinbase", "request error: {e}");
                    rate_limit_sleep("coinbase").await;
                    continue;
                }
                Ok(r) if !r.status().is_success() => {
                    warn!(exchange = "coinbase", "HTTP {}", r.status());
                    rate_limit_sleep("coinbase").await;
                    continue;
                }
                Ok(r) => r.json().await.context("parsing Coinbase JSON")?,
            };

            let mut rows: Vec<TradeRow> = Vec::new();
            for candle in &cb.candles {
                let ts_secs: i64 = candle.start.parse().unwrap_or(0);
                let ts_micros    = ts_secs * 1_000_000;

                // Skip rows already committed.
                if let Some(c) = cur.last_ts_micros {
                    if ts_micros <= c { continue; }
                }

                let price:    f64 = candle.close.parse().unwrap_or(0.0);
                let quantity: f64 = candle.volume.parse().unwrap_or(0.0);
                if price <= 0.0 || quantity <= 0.0 { continue; }

                rows.push(TradeRow {
                    ts_micros,
                    price,
                    quantity,
                    side:     None,  // candles have no per-trade side
                    exchange: "coinbase",
                });
            }

            // Candles come back newest-first; sort ascending.
            rows.sort_unstable_by_key(|r| r.ts_micros);

            if !rows.is_empty() {
                for row in rows {
                    let dt  = Utc.timestamp_micros(row.ts_micros).single().unwrap();
                    let key = (dt.year(), dt.month());
                    match month_key {
                        None => month_key = Some(key),
                        Some(k) if k != key => {
                            flush(&mut month_buf, &mut cur, &mut total_rows)?;
                            month_key = Some(key);
                        }
                        _ => {}
                    }
                    month_buf.push(row);
                }
            } else {
                info!(exchange = "coinbase", window_start, "no new candles");
            }

            window_start = window_end;
            rate_limit_sleep("coinbase").await;
        }

        // Flush whatever remains for the in-progress month.
        flush(&mut month_buf, &mut cur, &mut total_rows)?;

        Ok(total_rows)
    }
}
