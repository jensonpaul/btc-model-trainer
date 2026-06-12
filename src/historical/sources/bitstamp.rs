//! Bitstamp historical OHLC fetcher.
//!
//! # Endpoint
//!
//! ```text
//! GET https://www.bitstamp.net/api/v2/ohlc/btcusd/
//!     ?step=60            — 1-minute candles
//!     &limit=1000         — max candles per page
//!     &start={unix_secs}  — inclusive range start
//! ```
//!
//! # Why not `/transactions/`
//!
//! The previous implementation used
//! `GET /api/v2/transactions/btcusd/?time=day`, intending `time=day` as a
//! historical date selector. It is not: per Bitstamp's docs, `time` only
//! accepts `minute|hour|day` as a window **relative to now** (i.e. "trades
//! from the last 24 hours"), and the endpoint has no `start`/`end`/`since`
//! parameter at all. Iterating historical calendar days against this
//! endpoint therefore always returns recent trades that fall outside the
//! requested historical day's filter window, producing zero rows for every
//! historical day — exactly the "no data for this request" symptom. There
//! is no tick-level historical endpoint on Bitstamp's public API.
//!
//! # Response shape
//!
//! ```json
//! {
//!   "data": {
//!     "ohlc": [
//!       {
//!         "timestamp": "1700000000",
//!         "open": "34000.00", "high": "34050.00",
//!         "low": "33950.00", "close": "34010.00",
//!         "volume": "12.345"
//!       },
//!       ...
//!     ],
//!     "pair": "BTC/USD"
//!   }
//! }
//! ```
//!
//! Reference: https://www.bitstamp.net/api/#tag/Market-info/paths/~1api~1v2~1ohlc~1{currency_pair}~1/get
//!
//! # Pagination
//!
//! Up to 1000 1-minute candles per request (~16.7 hours). We page forward by
//! setting `start` to `last_returned_timestamp + step` until we reach
//! `up_to_micros` or the API stops returning new candles (caught up to now).
//!
//! # Side information
//!
//! OHLC candles have no per-trade side; `side` is written as `None`, same
//! as the Kraken OHLC-derived rows.

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

const OHLC_URL: &str = "https://www.bitstamp.net/api/v2/ohlc/btcusd/";
const STEP_SECS: i64 = 60;
const LIMIT: i64 = 1000;

#[derive(Debug, Deserialize)]
struct OhlcResponse {
    data: OhlcData,
}

#[derive(Debug, Deserialize)]
struct OhlcData {
    ohlc: Vec<OhlcEntry>,
}

#[derive(Debug, Deserialize)]
struct OhlcEntry {
    timestamp: String,
    close:     String,
    volume:    String,
}

pub struct BitstampFetcher {
    data_root: PathBuf,
    client:    reqwest::Client,
}

impl BitstampFetcher {
    pub fn new(data_root: PathBuf) -> Self {
        Self { data_root, client: build_client() }
    }
}

#[async_trait]
impl HistoricalFetcher for BitstampFetcher {
    fn exchange(&self) -> &'static str { "bitstamp" }

    /// Rows are accumulated **in memory** for the calendar month currently
    /// being fetched and written with a single `write_batch` call per month
    /// — either when a page's rows cross into the next month, or at the end
    /// of the run for the in-progress month. The cursor advances on every
    /// page in memory but is only persisted at those flush points.
    async fn fetch_range(&self, resume_from_micros: i64, up_to_micros: i64) -> Result<u64> {
        let store       = DataStore::new(&self.data_root, "bitstamp")?;
        let cursor_path = cursor::cursor_path(&self.data_root, "bitstamp");
        let mut cur     = FetchCursor::load(&cursor_path)?;

        let floor = exchange_floor("bitstamp").timestamp_micros();
        let start_micros = resume_from_micros.max(floor);

        // Bitstamp `start` is in seconds. Resume one step past the cursor so
        // we don't re-request (and re-filter-out) the last committed candle.
        let mut cursor_secs: i64 = start_micros / 1_000_000;
        let up_to_secs: i64 = up_to_micros / 1_000_000;

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
                    info!(exchange = "bitstamp", year = y, month = m, rows = month_buf.len(),
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
                .context("writing Bitstamp shard")?;
            cur.advance(last_ts, n);
            cur.save(&cursor_path)?;
            *total_rows += n;
            info!(exchange = "bitstamp", rows = n, last_ts, "month shard written");
            month_buf.clear();
            Ok(())
        };

        loop {
            if cursor_secs >= up_to_secs { break; }

            let url = format!("{OHLC_URL}?step={STEP_SECS}&limit={LIMIT}&start={cursor_secs}");
            info!(exchange = "bitstamp", start = cursor_secs, "fetching OHLC page");

            let resp = self.client.get(&url).send().await;
            let parsed: OhlcResponse = match resp {
                Err(e) => {
                    warn!(exchange = "bitstamp", "request error: {e}");
                    rate_limit_sleep("bitstamp").await;
                    continue;
                }
                Ok(r) if !r.status().is_success() => {
                    warn!(exchange = "bitstamp", "HTTP {}", r.status());
                    rate_limit_sleep("bitstamp").await;
                    continue;
                }
                Ok(r) => r.json().await.context("parsing Bitstamp JSON")?,
            };

            if parsed.data.ohlc.is_empty() { break; }

            let mut max_secs = cursor_secs;

            for entry in &parsed.data.ohlc {
                let ts_secs: i64 = entry.timestamp.parse().unwrap_or(0);
                if ts_secs < cursor_secs || ts_secs >= up_to_secs { continue; }

                let ts_micros = ts_secs * 1_000_000;

                // Skip rows already committed.
                if let Some(c) = cur.last_ts_micros {
                    if ts_micros <= c {
                        max_secs = max_secs.max(ts_secs);
                        continue;
                    }
                }

                let price:    f64 = entry.close.parse().unwrap_or(0.0);
                let quantity: f64 = entry.volume.parse().unwrap_or(0.0);
                if price <= 0.0 { max_secs = max_secs.max(ts_secs); continue; }

                let row = TradeRow {
                    ts_micros,
                    price,
                    quantity,
                    side:     None, // OHLC candles have no per-trade side
                    exchange: "bitstamp",
                };

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
                max_secs = max_secs.max(ts_secs);
            }

            if max_secs <= cursor_secs {
                // No forward progress (e.g. caught up to the present); stop.
                break;
            }
            cursor_secs = max_secs + STEP_SECS;

            rate_limit_sleep("bitstamp").await;
        }

        // Flush whatever remains for the in-progress month.
        flush(&mut month_buf, &mut cur, &mut total_rows)?;

        Ok(total_rows)
    }
}

