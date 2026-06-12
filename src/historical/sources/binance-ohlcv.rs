//! Binance historical klines fetcher — data.binance.vision bulk downloads.
//!
//! # Strategy
//!
//! Binance publishes pre-generated monthly ZIP files of 1-minute `klines` on
//! <https://data.binance.vision>.  Each ZIP contains a single headerless CSV:
//!
//! ```text
//! https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1m/
//!     BTCUSDT-1m-{YYYY}-{MM}.zip
//! ```
//!
//! # CSV columns (inside the ZIP)
//!
//! | # | Name                        | Type    | Notes                                      |
//! |---|-----------------------------|---------|--------------------------------------------|
//! | 0 | open_time                   | i64     | µs since epoch (ms before 2025-01-01)      |
//! | 1 | open                        | f64     | USD                                        |
//! | 2 | high                        | f64     | USD                                        |
//! | 3 | low                         | f64     | USD                                        |
//! | 4 | close                       | f64     | USD — used as the representative price     |
//! | 5 | volume                      | f64     | BTC — base asset volume for the candle     |
//! | 6 | close_time                  | i64     | ignored                                    |
//! | 7 | quote_asset_volume          | f64     | ignored                                    |
//! | 8 | number_of_trades            | i64     | ignored                                    |
//! | 9 | taker_buy_base_asset_volume | f64     | ignored                                    |
//! |10 | taker_buy_quote_asset_volume| f64     | ignored                                    |
//! |11 | ignore                      | f64     | ignored                                    |
//!
//! Reference: https://github.com/binance/binance-public-data/#klines
//!
//! # Timestamp normalisation
//!
//! Binance changed the SPOT klines timestamp precision on 2025-01-01:
//! - Before 2025-01-01: `open_time` is in **milliseconds** (13-digit).
//! - From  2025-01-01:  `open_time` is in **microseconds** (16-digit).
//!
//! We normalise both to microseconds using the magnitude of the raw value.
//!
//! # Row representation
//!
//! Each 1-minute kline is stored as one `TradeRow` using:
//! - `ts_micros` = open_time normalised to microseconds
//! - `price`     = close price
//! - `quantity`  = base asset (BTC) volume
//! - `side`      = `None` (klines are aggregated; individual side is unavailable)
//!
//! # Incremental behaviour
//!
//! The cursor stores the last written `ts_micros`.  On each run only months
//! where `month_end > cursor` are downloaded.  Already-complete months
//! (both start and end are before the cursor) are skipped entirely.

use std::io::{Cursor, Read};
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use chrono::{Datelike, TimeZone, Utc};
use tracing::{info, warn};
use zip::ZipArchive;

use crate::historical::{
    exchange_floor,
    schema::TradeRow,
    store::DataStore,
};
use super::{build_client, rate_limit_sleep, HistoricalFetcher};

/// Interval used for klines bulk downloads.  1-minute provides the finest
/// granularity available on data.binance.vision.
const KLINE_INTERVAL: &str = "1m";

const BASE_URL: &str =
    "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT";

pub struct BinanceFetcher {
    data_root: PathBuf,
    client:    reqwest::Client,
}

impl BinanceFetcher {
    pub fn new(data_root: PathBuf) -> Self {
        Self { data_root, client: build_client() }
    }
}

#[async_trait]
impl HistoricalFetcher for BinanceFetcher {
    fn exchange(&self) -> &'static str { "binance" }

    /// Binance publishes one immutable ZIP per *completed* calendar month.
    /// There is therefore no need for a read-merge-dedup cycle:
    /// a month's shard either exists on disk (already fully fetched) or it
    /// doesn't (fetch the whole ZIP and write the shard in one shot).  The
    /// current/in-progress month's ZIP simply isn't published yet (404) and
    /// is naturally retried on the next run.
    async fn fetch_range(&self, resume_from_micros: i64, up_to_micros: i64) -> Result<u64> {
        let store = DataStore::new(&self.data_root, "binance")?;

        let floor = exchange_floor("binance").timestamp_micros();
        let start = resume_from_micros.max(floor);

        let start_dt = Utc.timestamp_micros(start).single()
            .context("invalid start timestamp")?;
        let end_dt   = Utc.timestamp_micros(up_to_micros).single()
            .context("invalid end timestamp")?;

        let mut total_rows: u64 = 0;

        // Iterate month by month from start to end.
        let mut year  = start_dt.year();
        let mut month = start_dt.month();

        loop {
            let month_start = Utc.with_ymd_and_hms(year, month, 1, 0, 0, 0).unwrap();
            if month_start >= end_dt { break; }

            // Completed-month check: if the shard already exists on disk it
            // was written in full by a previous run — skip without touching it.
            let shard_path = store.shard_path(year, month)?;
            if shard_path.exists() {
                info!(exchange = "binance", year, month, "shard already exists, skipping");
                advance_month(&mut year, &mut month);
                continue;
            }

            let url = format!(
                "{BASE_URL}/{KLINE_INTERVAL}/BTCUSDT-{KLINE_INTERVAL}-{year}-{month:02}.zip"
            );
            info!(exchange = "binance", year, month, %url, "fetching klines");

            let resp = self.client.get(&url).send().await;
            match resp {
                Err(e) => {
                    warn!(exchange = "binance", year, month, error = %e, "request error");
                }
                Ok(r) if !r.status().is_success() => {
                    // 404 is normal for the current/future month — the monthly
                    // ZIP is published on the first Monday after month end.
                    if r.status().as_u16() == 404 {
                        info!(exchange = "binance", year, month, "not found (404), skipping");
                    } else {
                        warn!(exchange = "binance", year, month, status = %r.status(), "unexpected HTTP status");
                    }
                }
                Ok(r) => {
                    let bytes = r.bytes().await.context("reading Binance klines ZIP")?;
                    let rows  = parse_klines_zip(&bytes, year, month)
                        .with_context(|| format!("parsing Binance klines ZIP {year}-{month:02}"))?;

                    if !rows.is_empty() {
                        let n = rows.len() as u64;
                        store.write_batch(&rows)
                            .with_context(|| format!("writing shard {year}-{month:02}"))?;
                        total_rows += n;
                        info!(exchange = "binance", year, month, rows = n, "written");
                    } else {
                        info!(exchange = "binance", year, month, "ZIP contained no valid rows");
                    }
                }
            }

            rate_limit_sleep("binance").await;
            advance_month(&mut year, &mut month);
        }

        Ok(total_rows)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Advance `(year, month)` by one calendar month in place.
#[inline]
fn advance_month(year: &mut i32, month: &mut u32) {
    if *month == 12 {
        *year  += 1;
        *month  = 1;
    } else {
        *month += 1;
    }
}

/// Normalise a raw Binance klines timestamp to **microseconds**.
///
/// Binance changed SPOT klines precision on 2025-01-01:
/// - 16-digit value → already microseconds.
/// - 13-digit value → milliseconds; multiply by 1 000.
///
/// Any other magnitude is treated as milliseconds for forward-compatibility.
#[inline]
fn to_micros(raw: i64) -> i64 {
    // 1e15 is the smallest 16-digit positive integer.
    if raw >= 1_000_000_000_000_000 {
        raw           // already µs
    } else {
        raw * 1_000   // ms → µs
    }
}

/// Parse a monthly klines ZIP into a sorted `Vec<TradeRow>`.
///
/// The ZIP contains a single headerless CSV whose columns follow the
/// `/api/v3/klines` spec documented at
/// <https://github.com/binance/binance-public-data/#klines>.
fn parse_klines_zip(bytes: &[u8], year: i32, month: u32) -> Result<Vec<TradeRow>> {
    let cursor  = Cursor::new(bytes);
    let mut archive = ZipArchive::new(cursor).context("opening klines ZIP")?;

    if archive.len() == 0 {
        bail!("ZIP is empty for {year}-{month:02}");
    }

    let mut csv_bytes = Vec::new();
    {
        let mut entry = archive.by_index(0).context("reading ZIP entry")?;
        entry.read_to_end(&mut csv_bytes).context("reading CSV bytes from ZIP")?;
    }

    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)   // Binance bulk files have no header row
        .from_reader(csv_bytes.as_slice());

    let mut rows = Vec::new();

    for result in rdr.records() {
        let rec = result.context("parsing CSV record")?;

        // Column 0: open_time (ms before 2025-01-01, µs from 2025-01-01 onwards)
        let raw_open_time: i64 = rec.get(0).unwrap_or("").parse().unwrap_or(0);
        // Column 4: close price — representative price for the candle
        let price:    f64 = rec.get(4).unwrap_or("").parse().unwrap_or(0.0);
        // Column 5: base asset volume (BTC)
        let quantity: f64 = rec.get(5).unwrap_or("").parse().unwrap_or(0.0);

        if raw_open_time <= 0 || price <= 0.0 || quantity <= 0.0 {
            continue;
        }

        let ts_micros = to_micros(raw_open_time);

        rows.push(TradeRow {
            ts_micros,
            price,
            quantity,
            // Klines are aggregated bars — individual taker side is unavailable.
            side:     None,
            exchange: "binance",
        });
    }

    rows.sort_unstable_by_key(|r| r.ts_micros);
    Ok(rows)
}
