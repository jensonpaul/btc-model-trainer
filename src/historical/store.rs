//! Monthly Parquet shard writer and CSV reader.
//!
//! # Shard layout
//!
//! ```text
//! {data_root}/{exchange}/{YYYY}/{MM}.parquet
//! ```
//!
//! Each shard contains all trades for one calendar month, sorted ascending
//! by `ts_micros`.  A shard is written once and never modified; only the
//! current month's shard is appended to (by writing a new file that replaces
//! the previous partial write).
//!
//! # Read path
//!
//! [`DataStore::emit_csv`] streams all shards for all exchanges in timestamp
//! order into a [`csv::Writer`], producing the exact format that
//! `src/main.rs` expects:
//!
//! ```text
//! ts_micros,price,quantity,side,exchange
//! ```

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use arrow::array::{
    Array, Float64Array, Int64Array, StringArray, StringBuilder, Float64Builder,
    Int64Builder,
};
use arrow::record_batch::RecordBatch;
use chrono::{Datelike, TimeZone, Utc};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use tracing::{debug, info};

use crate::historical::schema::{TradeRow, SCHEMA};

/// Manages reading and writing the monthly Parquet shards for one exchange.
pub struct DataStore {
    data_root: PathBuf,
    exchange:  &'static str,
}

impl DataStore {
    pub fn new(data_root: &Path, exchange: &'static str) -> Result<Self> {
        let dir = data_root.join(exchange);
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("creating data dir {}", dir.display()))?;
        Ok(Self { data_root: data_root.to_path_buf(), exchange })
    }

    /// Path for a given year/month shard (creates the year directory if needed).
    pub fn shard_path(&self, year: i32, month: u32) -> Result<PathBuf> {
        let dir = self.data_root.join(self.exchange).join(year.to_string());
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("creating shard dir {}", dir.display()))?;
        Ok(dir.join(format!("{:02}.parquet", month)))
    }

    /// Write a batch of rows into the shard for their calendar month.
    ///
    /// Rows **must** be sorted ascending by `ts_micros` before calling this.
    /// Multiple month-boundaries within one batch are handled — the batch is
    /// split and each month-slice written to its own shard file.
    ///
    /// If a shard for a given month already exists it is **replaced** (the
    /// new data must already contain everything the old shard did, since the
    /// fetcher only calls this for months where the cursor is behind).
    pub fn write_batch(&self, rows: &[TradeRow]) -> Result<()> {
        if rows.is_empty() {
            return Ok(());
        }

        // Group rows by (year, month).
        let mut by_month: BTreeMap<(i32, u32), Vec<&TradeRow>> = BTreeMap::new();
        for row in rows {
            let dt = Utc
                .timestamp_micros(row.ts_micros)
                .single()
                .unwrap_or(DateTime::<Utc>::MIN_UTC);
            by_month.entry((dt.year(), dt.month())).or_default().push(row);
        }

        for ((year, month), chunk) in &by_month {
            let path = self.shard_path(*year, *month)?;
            self.write_shard(&path, chunk)
                .with_context(|| format!("writing shard {}", path.display()))?;
            info!(
                exchange = self.exchange,
                year, month,
                rows = chunk.len(),
                path = %path.display(),
                "shard written",
            );
        }
        Ok(())
    }

    /// Read an existing shard back into owned [`TradeRow`]s, for seeding an
    /// in-memory accumulator before a single overwrite write later in the
    /// run (the "current month" case). This is the *only* read of an
    /// existing shard a fetcher should need per run.
    pub fn read_shard_for_seed(&self, path: &Path) -> Result<Vec<TradeRow>> {
        let owned = self.read_shard_owned(path)?;
        Ok(owned
            .into_iter()
            .map(|r| TradeRow {
                ts_micros: r.ts_micros,
                price:     r.price,
                quantity:  r.quantity,
                side:      r.side.as_deref().map(side_to_static),
                exchange:  exchange_to_static(&r.exchange),
            })
            .collect())
    }

    // ── Internal helpers ───────────────────────────────────────────────────

    fn write_shard(&self, path: &Path, rows: &[&TradeRow]) -> Result<()> {
        // Larger row groups reduce per-group metadata/compression overhead
        // for the large monthly shards produced during a full backfill
        // (millions of rows). Default row groups (~1M rows already, but
        // capped lower in some parquet-rs versions) leave CPU on the table
        // for big batches; cap at 2M to keep per-group memory reasonable.
        let row_group_size = rows.len().min(2_000_000).max(1);
        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .set_max_row_group_size(row_group_size)
            .build();

        let file = std::fs::File::create(path)
            .with_context(|| format!("creating {}", path.display()))?;

        let mut writer = ArrowWriter::try_new(file, SCHEMA.clone(), Some(props))
            .context("creating ArrowWriter")?;

        // Build Arrow arrays column-by-column.
        let n = rows.len();
        let mut ts_b   = Int64Builder::with_capacity(n);
        let mut px_b   = Float64Builder::with_capacity(n);
        let mut qty_b  = Float64Builder::with_capacity(n);
        let mut side_b = StringBuilder::new();
        let mut exch_b = StringBuilder::new();

        for r in rows {
            ts_b.append_value(r.ts_micros);
            px_b.append_value(r.price);
            qty_b.append_value(r.quantity);
            match r.side {
                Some(s) => side_b.append_value(s),
                None    => side_b.append_null(),
            }
            exch_b.append_value(r.exchange);
        }

        let batch = RecordBatch::try_new(
            SCHEMA.clone(),
            vec![
                Arc::new(ts_b.finish()),
                Arc::new(px_b.finish()),
                Arc::new(qty_b.finish()),
                Arc::new(side_b.finish()),
                Arc::new(exch_b.finish()),
            ],
        )
        .context("building RecordBatch")?;

        writer.write(&batch).context("writing RecordBatch")?;
        writer.close().context("closing ArrowWriter")?;
        Ok(())
    }

    fn read_shard_owned(&self, path: &Path) -> Result<Vec<OwnedRow>> {
        let file = std::fs::File::open(path)
            .with_context(|| format!("opening {}", path.display()))?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .context("ParquetRecordBatchReaderBuilder")?;
        let mut reader = builder.build().context("building parquet reader")?;

        let mut out = Vec::new();
        while let Some(batch) = reader.next() {
            let batch = batch.context("reading batch")?;
            let ts   = batch.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
            let px   = batch.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
            let qty  = batch.column(2).as_any().downcast_ref::<Float64Array>().unwrap();
            let side = batch.column(3).as_any().downcast_ref::<StringArray>().unwrap();
            let exch = batch.column(4).as_any().downcast_ref::<StringArray>().unwrap();
            for i in 0..batch.num_rows() {
                out.push(OwnedRow {
                    ts_micros: ts.value(i),
                    price:     px.value(i),
                    quantity:  qty.value(i),
                    side:      if side.is_null(i) { None } else { Some(side.value(i).to_owned()) },
                    exchange:  exch.value(i).to_owned(),
                });
            }
        }
        Ok(out)
    }
}

// ── Month iterator (preferred low-memory path) ───────────────────────────────

/// Returns a sorted list of all `(year, month)` keys present across all
/// exchanges under `data_root`.
fn collect_month_keys(data_root: &Path, exchanges: &[&str]) -> Result<Vec<(i32, u32)>> {
    let mut month_keys: std::collections::BTreeSet<(i32, u32)> = Default::default();
    for exchange in exchanges {
        let base = data_root.join(exchange);
        if !base.exists() { continue; }
        for year_entry in std::fs::read_dir(&base)? {
            let year_entry = year_entry?;
            if !year_entry.file_type()?.is_dir() { continue; }
            let year: i32 = match year_entry.file_name().to_str().and_then(|s| s.parse().ok()) {
                Some(y) => y,
                None    => continue,
            };
            for shard_entry in std::fs::read_dir(year_entry.path())? {
                let shard_entry = shard_entry?;
                let name = shard_entry.file_name();
                let stem = Path::new(name.to_str().unwrap_or("")).file_stem()
                    .and_then(|s| s.to_str())
                    .and_then(|s| s.parse::<u32>().ok());
                if let Some(month) = stem {
                    month_keys.insert((year, month));
                }
            }
        }
    }
    Ok(month_keys.into_iter().collect())
}

/// Lazy iterator that yields one month's merged-and-sorted [`OwnedRow`]s at a
/// time.  Each `Vec` is dropped by the caller before the next one is produced,
/// keeping peak RAM bounded to a single month of trade rows plus the pipeline
/// ring buffer (label_buckets + 1 feature entries).
///
/// Prefer this over [`emit_csv`] whenever the downstream consumer works
/// row-by-row (e.g. the `collect-training-data` feature pipeline).
pub fn iter_months_sorted<'a>(
    data_root: &'a Path,
    exchanges: &'a [&'a str],
) -> Result<impl Iterator<Item = Result<Vec<OwnedRow>>> + 'a> {
    let month_keys = collect_month_keys(data_root, exchanges)?;
    let data_root  = data_root.to_path_buf();

    let iter = month_keys.into_iter().map(move |(year, month)| {
        // Allocate fresh each iteration; the caller drops the previous Vec
        // before this closure runs again, so memory never accumulates.
        // Read each exchange's shard for this month in parallel — Parquet
        // decoding is CPU-bound and these reads are fully independent.
        use rayon::prelude::*;
        let chunks: Result<Vec<Vec<OwnedRow>>> = exchanges
            .par_iter()
            .map(|exchange| -> Result<Vec<OwnedRow>> {
                let path = data_root
                    .join(exchange)
                    .join(year.to_string())
                    .join(format!("{:02}.parquet", month));
                if !path.exists() { return Ok(Vec::new()); }

                let store = DataStore::new(&data_root, exchange_to_static(exchange))?;
                store.read_shard_owned(&path)
            })
            .collect();

        let mut month_rows: Vec<OwnedRow> = chunks?.into_iter().flatten().collect();
        month_rows.par_sort_unstable_by_key(|r| r.ts_micros);
        debug!(year, month, rows = month_rows.len(), "loaded month shard");
        // Ownership transferred to caller; freed at end of caller's loop body.
        Ok(month_rows)
    });

    Ok(iter)
}

// ── CSV emission (kept for fetch-historical-data --emit-csv) ─────────────────

/// Streams all Parquet shards for all exchanges under `data_root` into a
/// single CSV writer, sorted globally by `ts_micros`, one month at a time.
///
/// This produces the exact input format consumed by `collect-training-data`
/// when using `--input` (the legacy CSV workflow):
///
/// ```text
/// ts_micros,price,quantity,side,exchange
/// ```
pub fn emit_csv<W: Write>(data_root: &Path, exchanges: &[&str], writer: W) -> Result<u64> {
    let mut wtr = csv::Writer::from_writer(writer);
    wtr.write_record(["ts_micros", "price", "quantity", "side", "exchange"])?;

    let mut total_rows: u64 = 0;

    for month_result in iter_months_sorted(data_root, exchanges)? {
        let month_rows = month_result?;
        for r in &month_rows {
            wtr.write_record(&[
                r.ts_micros.to_string(),
                format!("{:.8}", r.price),
                format!("{:.8}", r.quantity),
                r.side.as_deref().unwrap_or("").to_owned(),
                r.exchange.clone(),
            ])?;
        }
        total_rows += month_rows.len() as u64;
        // month_rows dropped here — memory freed before next month loads.
    }

    wtr.flush()?;
    Ok(total_rows)
}

// ── Owned row type (shared between store internals and callers) ───────────────

/// An owned, heap-allocated trade row used for in-process merge-sorting.
/// Exposed publicly so that callers of [`iter_months_sorted`] can iterate
/// rows directly without going through an intermediate CSV serialisation step.
#[derive(Debug, Clone)]
pub struct OwnedRow {
    pub ts_micros: i64,
    pub price:     f64,
    pub quantity:  f64,
    pub side:      Option<String>,
    pub exchange:  String,
}

use chrono::DateTime;

fn side_to_static(s: &str) -> &'static str {
    match s { "buy" => "buy", "sell" => "sell", _ => "buy" }
}

fn exchange_to_static(s: &str) -> &'static str {
    match s {
        "binance"  => "binance",
        "kraken"   => "kraken",
        "bitstamp" => "bitstamp",
        "coinbase" => "coinbase",
        _          => "unknown",
    }
}
