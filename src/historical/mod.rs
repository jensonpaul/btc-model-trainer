//! Synchronized historical tick data collection for model training.
//!
//! # Architecture
//!
//! Each exchange has a dedicated fetcher that downloads trades/candles from
//! public REST APIs and persists them as **Parquet files** on disk:
//!
//! ```text
//! data/
//!   binance/
//!     2017/01.parquet  ..  2025/12.parquet
//!     cursor.json           ← last fetched timestamp (µs)
//!   kraken/
//!     2017/01.parquet  ..
//!     cursor.json
//!   bitstamp/
//!     ...
//!   coinbase/
//!     ...
//! ```
//!
//! Running `fetch-historical-data` is **idempotent**: it reads each exchange's
//! `cursor.json`, resumes from that point, and writes only new months.
//! Re-running after an interruption costs nothing already fetched.
//!
//! # Origin date rationale
//!
//! **2017-01-01 UTC** is chosen as the global start:
//!
//! * Covers all major BTC market regimes: the 2017 bubble, the 2018–2019
//!   bear market, the 2020 COVID crash and recovery, the 2021 double-peak
//!   cycle, the 2022 FTX/LUNA crash, and the 2023-2025 recovery/ETF era.
//! * Gives the model ≥8 years of diverse volatility states and
//!   correlation regimes — critical for robust `Broad` (24 h) scale
//!   predictions without overfitting to a single cycle.
//! * Pre-2017 data is structurally different (illiquid, no derivatives
//!   market, single-digit-billion USD daily volume) and would introduce
//!   regime noise rather than signal.
//! * All four exchanges (Binance launched Jul 2017, Kraken since 2013,
//!   Bitstamp since 2011, Coinbase since 2015) have full liquidity from
//!   2017 onward; months before each exchange's launch are simply skipped.
//!
//! # Row schema (all sources normalised before writing)
//!
//! | Column        | Arrow type         | Description                        |
//! |---------------|--------------------|------------------------------------|
//! | `ts_micros`   | `Int64`            | µs since UNIX epoch, ascending     |
//! | `price`       | `Float64`          | USD                                |
//! | `quantity`    | `Float64`          | BTC                                |
//! | `side`        | `Utf8` (nullable)  | `"buy"` \| `"sell"` \| null        |
//! | `exchange`    | `Utf8`             | `"binance"` \| …                   |

pub mod schema;
pub mod store;
pub mod cursor;
pub mod sources;

pub use schema::{TradeRow, SCHEMA};
pub use store::DataStore;
pub use cursor::Cursor;

use chrono::{DateTime, NaiveDate, TimeZone, Utc};

/// The earliest date from which we collect data for all exchanges.
///
/// See module-level docs for rationale.
pub const ORIGIN_DATE: NaiveDate = {
    // NaiveDate::from_ymd_opt is not const-stable, so we use a workaround.
    // This evaluates at compile time; the panic is unreachable.
    match NaiveDate::from_ymd_opt(2017, 1, 1) {
        Some(d) => d,
        None    => panic!("invalid origin date"),
    }
};

/// `ORIGIN_DATE` as a UTC timestamp in microseconds.
pub fn origin_micros() -> i64 {
    Utc.from_utc_datetime(&ORIGIN_DATE.and_hms_opt(0, 0, 0).unwrap())
        .timestamp_micros()
}

/// Earliest date each exchange has reliable BTC/USD public data.
///
/// Months before this floor are skipped even if they fall after
/// [`ORIGIN_DATE`].  Binance launched its spot market in August 2017.
pub fn exchange_floor(exchange: &str) -> DateTime<Utc> {
    let (y, m) = match exchange {
        "binance"  => (2017, 8),  // Binance spot launched 2017-08
        "kraken"   => (2017, 1),  // Kraken BTC/USD liquid from 2017
        "bitstamp" => (2017, 1),  // Bitstamp has full data from 2017
        "coinbase" => (2017, 1),  // Coinbase Pro (GDAX) from 2016, use 2017
        _          => (2017, 1),
    };
    Utc.with_ymd_and_hms(y, m, 1, 0, 0, 0).unwrap()
}
