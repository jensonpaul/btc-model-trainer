//! Binance historical tick fetcher — data.binance.vision bulk downloads.
//!
//! # Strategy
//!
//! Binance publishes pre-generated monthly ZIP files of `aggTrades` on
//! <https://data.binance.vision>.  Each ZIP contains a single CSV:
//!
//! ```text
//! https://data.binance.vision/data/spot/monthly/aggTrades/BTCUSDT/
//!     BTCUSDT-aggTrades-{YYYY}-{MM}.zip
//! ```
//!
//! The file listing index is available at the same base URL as an XML
//! document (S3-style bucket listing).
//!
//! # CSV columns (inside the ZIP)
//!
//! | # | Name                   | Type    | Notes                         |
//! |---|------------------------|---------|-------------------------------|
//! | 0 | agg_trade_id           | i64     | ignored                       |
//! | 1 | price                  | f64     | USD                           |
//! | 2 | quantity               | f64     | BTC                           |
//! | 3 | first_trade_id         | i64     | ignored                       |
//! | 4 | last_trade_id          | i64     | ignored                       |
//! | 5 | transact_time          | i64     | **milliseconds** since epoch  |
//! | 6 | is_buyer_maker         | bool    | true → taker is seller        |
//! | 7 | is_best_match          | bool    | ignored                       |
//!
//! Reference: https://github.com/binance/binance-public-data/#aggTrades
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

const BASE_URL: &str =
    "https://data.binance.vision/data/spot/monthly/aggTrades/BTCUSDT";

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
    /// There is therefore no need for a cursor or a read-merge-dedup cycle:
    /// a month's shard either exists on disk (already fully fetched and
    /// written in one shot) or it doesn't (fetch the whole ZIP and write
    /// the shard once). The current/in-progress month's ZIP simply isn't
    /// published yet (404) and is naturally retried on the next run.
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

            // Completed-month check: if the shard already exists, it was
            // written in full by a previous run — skip without touching it.
            let shard_path = store.shard_path(year, month)?;
            if shard_path.exists() {
                info!(exchange = "binance", year, month, "shard already exists, skipping");
                if month == 12 { year += 1; month = 1; } else { month += 1; }
                continue;
            }

            let url = format!("{BASE_URL}/BTCUSDT-aggTrades-{year}-{month:02}.zip");
            info!(exchange = "binance", year, month, "fetching {}", url);

            let resp = self.client.get(&url).send().await;
            match resp {
                Err(e) => {
                    warn!(exchange = "binance", year, month, "request error: {e}");
                }
                Ok(r) if !r.status().is_success() => {
                    // 404 is normal for the current/future month — the
                    // monthly ZIP is published a few days after month end.
                    if r.status().as_u16() == 404 {
                        info!(exchange = "binance", year, month, "not found (404), skipping");
                    } else {
                        warn!(exchange = "binance", year, month,
                              "HTTP {}", r.status());
                    }
                }
                Ok(r) => {
                    let bytes = r.bytes().await.context("reading Binance ZIP bytes")?;
                    let rows  = parse_zip(&bytes, year, month)
                        .with_context(|| format!("parsing Binance ZIP {year}-{month:02}"))?;

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

            // Advance to next month.
            if month == 12 { year += 1; month = 1; } else { month += 1; }
        }

        Ok(total_rows)
    }
}

/// Parse the monthly aggTrades ZIP into a sorted Vec<TradeRow>.
fn parse_zip(bytes: &[u8], year: i32, month: u32) -> Result<Vec<TradeRow>> {
    let cursor = Cursor::new(bytes);
    let mut archive = ZipArchive::new(cursor).context("opening ZIP")?;

    if archive.len() == 0 {
        bail!("ZIP is empty for {year}-{month:02}");
    }

    let mut csv_bytes = Vec::new();
    {
        let mut entry = archive.by_index(0).context("reading ZIP entry")?;
        entry.read_to_end(&mut csv_bytes).context("reading CSV from ZIP")?;
    }

    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)  // Binance bulk files have no header row
        .from_reader(csv_bytes.as_slice());

    let mut rows = Vec::new();
    for result in rdr.records() {
        let rec = result.context("parsing CSV record")?;
        // Column indices per Binance aggTrades spec (0-indexed).
        let price:     f64  = rec.get(1).unwrap_or("").parse().unwrap_or(0.0);
        let quantity:  f64  = rec.get(2).unwrap_or("").parse().unwrap_or(0.0);
        let ts_ms:     i64  = rec.get(5).unwrap_or("").parse().unwrap_or(0);
        //let is_maker:  bool = rec.get(6).unwrap_or("false") == "true";
        let is_maker: bool =
            rec.get(6)
               .map(|s| s.eq_ignore_ascii_case("true"))
               .unwrap_or(false);

        let raw_ts:    i64 =  rec.get(5).unwrap_or("").parse().unwrap_or(0);
        let ts_micros = match raw_ts {
            x if x > 10_000_000_000_000_000 => x / 1_000,      // ns -> µs
            x if x > 10_000_000_000_000     => x,              // already µs
            _                               => raw_ts * 1_000 // ms -> µs
        };

        if price <= 0.0 || quantity <= 0.0 || ts_ms <= 0 { continue; }

        rows.push(TradeRow {
            ts_micros,
            price,
            quantity,
            // is_buyer_maker = true  → buyer placed the resting order → taker is SELLER
            // is_buyer_maker = false → seller placed the resting order → taker is BUYER
            side:     Some(if is_maker { "sell" } else { "buy" }),
            exchange: "binance",
        });
    }

    rows.sort_unstable_by_key(|r| r.ts_micros);
    Ok(rows)
}
