//! `fetch-historical-data` — synchronised historical BTC/USD tick collection.
//!
//! Downloads trade data from all four exchanges into a monthly Parquet store,
//! resuming from each exchange's on-disk cursor so re-runs are fully
//! incremental.
//!
//! ## Usage
//!
//! ```bash
//! # Full initial backfill (2017-01-01 → today):
//! cargo run --release --bin fetch-historical-data -- \
//!     --data-dir ./data
//!
//! # Fetch only specific exchanges:
//! cargo run --release --bin fetch-historical-data -- \
//!     --data-dir ./data \
//!     --exchanges binance kraken
//!
//! # Fetch up to a specific date (e.g. for a training cut-off):
//! cargo run --release --bin fetch-historical-data -- \
//!     --data-dir ./data \
//!     --until 2025-01-01
//!
//! # After fetching, emit a merged CSV for the feature pipeline:
//! cargo run --release --bin fetch-historical-data -- \
//!     --data-dir ./data \
//!     --emit-csv ./data/merged.csv
//! ```
//!
//! ## Disk layout after a full run
//!
//! ```text
//! data/
//!   binance/
//!     cursor.json
//!     2017/08.parquet .. 2025/12.parquet
//!   kraken/
//!     cursor.json
//!     2017/01.parquet .. 2025/12.parquet
//!   bitstamp/
//!     cursor.json
//!     2017/01.parquet .. 2025/12.parquet
//!   coinbase/
//!     cursor.json
//!     2017/01.parquet .. 2025/12.parquet
//! ```
//!
//! ## Feeding the training pipeline
//!
//! Once data is collected, pass `--emit-csv` to produce the merged CSV, then
//! feed it directly to `collect-training-data`:
//!
//! ```bash
//! cargo run --release --bin collect-training-data -- \
//!     --input ./data/merged.csv \
//!     --output ./data/training.csv \
//!     --label-window-secs 300
//!
//! cd python && python train_direction_model.py \
//!     --input ../data/training.csv \
//!     --output model/direction_model.onnx \
//!     --scale short
//! ```

use std::path::PathBuf;

use anyhow::{Context, Result};
use chrono::{NaiveDate, TimeZone, Utc};
use clap::Parser;
use tracing::{error, info};
use tracing_subscriber::EnvFilter;

use btc_model_trainer::historical::{
    cursor::cursor_path,
    cursor::Cursor,
    origin_micros,
    sources::{
        binance::BinanceFetcher,
        bitstamp::BitstampFetcher,
        coinbase::CoinbaseFetcher,
        kraken::KrakenFetcher,
        HistoricalFetcher,
    },
    store::emit_csv,
};

// ── CLI ────────────────────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(
    name  = "fetch-historical-data",
    about = "Download and persist historical BTC/USD trades from all exchanges"
)]
struct Args {
    /// Root directory for the Parquet data store.
    #[arg(long, default_value = "data")]
    data_dir: PathBuf,

    /// Exchanges to fetch.  Defaults to all four.
    #[arg(long, value_delimiter = ' ', num_args = 1.., default_values_t = [
        "binance".to_string(),
        "kraken".to_string(),
        "bitstamp".to_string(),
        "coinbase".to_string(),
    ])]
    exchanges: Vec<String>,

    /// Fetch data starting from this date (YYYY-MM-DD).
    /// Defaults to 2017-01-01 (the global ORIGIN_DATE).
    /// Per-exchange launch floors are always respected regardless of this
    /// value.
    #[arg(long)]
    from: Option<NaiveDate>,

    /// Stop fetching at this date (YYYY-MM-DD, exclusive).
    /// Defaults to today (UTC).
    #[arg(long)]
    until: Option<NaiveDate>,

    /// After fetching, emit a merged multi-exchange CSV to this path.
    /// The CSV is exactly the format consumed by `collect-training-data`.
    /// If omitted, only the Parquet shards are written.
    #[arg(long)]
    emit_csv: Option<PathBuf>,

    /// Only emit the merged CSV from existing Parquet shards — do not fetch
    /// any new data.  Useful when you have up-to-date shards and just need
    /// to regenerate the merged CSV.
    #[arg(long, default_value_t = false)]
    csv_only: bool,
}

// ── Entry point ────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env()
            .add_directive("fetch_historical_data=info".parse().unwrap())
            .add_directive("btc_model_trainer=info".parse().unwrap()))
        .init();

    let args = Args::parse();

    std::fs::create_dir_all(&args.data_dir)
        .with_context(|| format!("creating data dir {}", args.data_dir.display()))?;

    // ── Resolve time range ─────────────────────────────────────────────────────
    let from_micros: i64 = args.from
        .map(|d| Utc.from_utc_datetime(&d.and_hms_opt(0, 0, 0).unwrap()).timestamp_micros())
        .unwrap_or_else(origin_micros);

    let until_micros: i64 = args.until
        .map(|d| Utc.from_utc_datetime(&d.and_hms_opt(0, 0, 0).unwrap()).timestamp_micros())
        .unwrap_or_else(|| Utc::now().timestamp_micros());

    info!(
        from  = %Utc.timestamp_micros(from_micros).single().unwrap(),
        until = %Utc.timestamp_micros(until_micros).single().unwrap(),
        exchanges = ?args.exchanges,
        data_dir  = %args.data_dir.display(),
        "starting historical data collection",
    );

    // ── Fetch ──────────────────────────────────────────────────────────────────
    if !args.csv_only {
        // Each exchange has its own cursor file and its own Parquet
        // sub-tree, so the fetches are fully independent and safe to run
        // concurrently. This turns 4 sequential network-bound fetches into
        // 4 parallel ones — typically a 3-4x wall-clock improvement during
        // a historical backfill, with each task still writing/advancing
        // its own cursor incrementally as before.
        let mut tasks = Vec::new();

        for exchange_name in args.exchanges.clone() {
            let data_dir = args.data_dir.clone();

            tasks.push(tokio::spawn(async move {
                let cursor_p     = cursor_path(&data_dir, &exchange_name);
                let saved_cursor = Cursor::load(&cursor_p)?;
                let resume_from  = match saved_cursor.last_ts_micros {
                    Some(c) => c.max(from_micros),
                    None    => from_micros,
                };

                info!(
                    exchange = %exchange_name,
                    resume_from = %Utc.timestamp_micros(resume_from).single().unwrap(),
                    "fetching",
                );

                let fetcher: Box<dyn HistoricalFetcher> = match exchange_name.as_str() {
                    "binance"  => Box::new(BinanceFetcher::new(data_dir.clone())),
                    "kraken"   => Box::new(KrakenFetcher::new(data_dir.clone())),
                    "bitstamp" => Box::new(BitstampFetcher::new(data_dir.clone())),
                    "coinbase" => Box::new(CoinbaseFetcher::new(data_dir.clone())),
                    other => {
                        error!("unknown exchange: {other}");
                        return Ok::<(), anyhow::Error>(());
                    }
                };

                match fetcher.fetch_range(resume_from, until_micros).await {
                    Ok(n)  => info!(exchange = %exchange_name, rows = n, "fetch complete"),
                    Err(e) => error!(exchange = %exchange_name, "fetch failed: {e:#}"),
                }
                Ok(())
            }));
        }

        for task in tasks {
            // A panic in one exchange's task must not prevent the others
            // from being awaited / reported.
            if let Err(e) = task.await {
                error!("exchange fetch task panicked: {e:#}");
            }
        }
    }

    // ── Emit CSV ───────────────────────────────────────────────────────────────
    if let Some(csv_path) = &args.emit_csv {
        info!(csv = %csv_path.display(), "emitting merged CSV");

        let exchange_refs: Vec<&str> = args.exchanges.iter().map(|s| s.as_str()).collect();

        let file = std::fs::File::create(csv_path)
            .with_context(|| format!("creating CSV {}", csv_path.display()))?;
        let writer = std::io::BufWriter::new(file);

        let rows = emit_csv(&args.data_dir, &exchange_refs, writer)
            .context("emitting merged CSV")?;

        info!(rows, csv = %csv_path.display(), "CSV written");
    }

    info!("done");
    Ok(())
}

