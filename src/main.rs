//! `collect-training-data` — offline historical → per-month labelled feature shards.
//!
//! Reads trade ticks either from a **pre-merged CSV** (`--input`) or directly
//! from the **Parquet data store** written by `fetch-historical-data`
//! (`--data-dir`), buckets them into 100 ms `FusedTick`s, replays them through
//! `btc_prediction_engine::features::FeatureState`, and emits one labelled row
//! per bucket.
//!
//! ## Output layout
//!
//! ```text
//! <output-dir>/
//!   2017-01.csv      ← one file per calendar month
//!   2017-02.csv
//!   ...
//!   2024-12.csv
//!   manifest.json    ← per-shard stats + global totals (no need to open CSVs)
//! ```
//!
//! ### Why shards instead of one file?
//!
//! * **Time-aware splits** — hard chronological train/val/test cuts are
//!   trivial: just pick which shard files go in each set.
//! * **Parallel DataLoader I/O** — PyTorch / Burn workers can each own a
//!   separate file handle; a single 10 GB file serialises all reads.
//! * **Incremental regeneration** — change a feature or label threshold for a
//!   specific period and only regenerate those months.
//! * **Bounded memory** — each shard is a few MB; the training process can
//!   load one at a time and shuffle within it.
//!
//! ## Important: feature state is continuous across shard boundaries
//!
//! The `FeatureState` (RSI, VWAP, EWMA, etc.) and the label ring buffer are
//! **not reset** when a new monthly shard is opened.  Only the CSV writer is
//! swapped.  This ensures that features at the start of February are correctly
//! conditioned on January's history, and that label-window rows that span a
//! month boundary (e.g. the last 5 minutes of January labelled by early
//! February prices) are written to the correct shard.
//!
//! ## manifest.json schema
//!
//! ```json
//! {
//!   "generated_at": "2024-06-01T12:00:00Z",
//!   "label_window_secs": 300,
//!   "label_threshold": 0.001,
//!   "bucket_ms": 100,
//!   "features": ["rsi_14", ...],
//!   "total": { "rows": 12345678, "bearish": 4000000, "sideways": 4345678, "bullish": 4000000 },
//!   "shards": [
//!     {
//!       "file": "2017-01.csv",
//!       "year": 2017, "month": 1,
//!       "rows": 123456,
//!       "bearish": 40000, "sideways": 43456, "bullish": 40000,
//!       "first_ts_micros": 1483228800000000,
//!       "last_ts_micros":  1485907199000000
//!     },
//!     ...
//!   ]
//! }
//! ```
//!
//! ## Usage
//!
//! ```bash
//! # From Parquet store (recommended):
//! cargo run --release --bin collect-training-data -- \
//!     --data-dir ./data \
//!     --output-dir ./data/training \
//!     --label-window-secs 300
//!
//! # From pre-merged CSV (legacy):
//! cargo run --release --bin collect-training-data -- \
//!     --input data/merged.csv \
//!     --output-dir ./data/training
//! ```

use btc_model_trainer::historical;

use std::collections::{HashMap, VecDeque};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Write};
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use chrono::{Datelike, TimeZone, Utc};
use clap::Parser;
use serde::{Deserialize, Serialize};

use btc_prediction_engine::features::{FeatureState, FeatureVector};
use btc_prediction_engine::price_fusion::FusedTick;
use btc_prediction_engine::types::{Exchange, Symbol, TradeSide};

// ── Constants ─────────────────────────────────────────────────────────────────

const FEATURE_NAMES: &[&str] = &[
    "rsi_14", "vwap_dev", "mom_micro", "mom_short", "ewma_vol",
    "tick_vel", "ofi_30s", "ofi_300s", "autocorr", "rvol_30s",
    "xchg_spread", "price_norm", "ewma_var",
    "book_imb5", "book_imb_full", "book_wmid", "book_spread",
];
const N_FEATURES: usize = 17;

// ── CLI ───────────────────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(about = "Replay historical BTC/USD trades through the feature \
                    pipeline and emit per-month labelled training shards")]
struct Args {
    /// [Mode A] Pre-merged CSV of historical trade ticks.
    /// Format: ts_micros,price,quantity,side,exchange
    /// Mutually exclusive with --data-dir.
    #[arg(long, conflicts_with = "data_dir")]
    input: Option<PathBuf>,

    /// [Mode B] Root of the Parquet store written by `fetch-historical-data`.
    /// All exchanges are merged and processed one month at a time.
    /// Mutually exclusive with --input.
    #[arg(long, conflicts_with = "input")]
    data_dir: Option<PathBuf>,

    /// Exchanges to include (--data-dir mode only).
    #[arg(
        long,
        value_delimiter = ' ',
        num_args = 1..,
        default_values_t = [
            "binance".to_string(),
            "kraken".to_string(),
            "bitstamp".to_string(),
            "coinbase".to_string(),
        ]
    )]
    exchanges: Vec<String>,

    /// Directory where per-month CSVs and manifest.json are written.
    /// Created if it does not exist.
    #[arg(long)]
    output_dir: PathBuf,

    /// Bucket width in milliseconds (must match the live engine's config).
    #[arg(long, default_value_t = 100)]
    bucket_ms: i64,

    /// Forward-looking label window in seconds (e.g. 300 = 5 minutes).
    #[arg(long, default_value_t = 300)]
    label_window_secs: i64,

    /// Return threshold separating Bullish/Bearish from Sideways (e.g. 0.001 = 0.1%).
    #[arg(long, default_value_t = 0.001)]
    label_threshold: f64,
}

// ── Trade row (input) ─────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct TradeRow {
    ts_micros: i64,
    price:     f64,
    quantity:  f64,
    #[serde(default)]
    side:      String,
    exchange:  String,
}

impl TradeRow {
    fn parse_side(&self) -> Option<TradeSide> {
        match self.side.to_ascii_lowercase().as_str() {
            "buy" | "b"  => Some(TradeSide::Buy),
            "sell" | "s" => Some(TradeSide::Sell),
            _            => None,
        }
    }

    fn parse_exchange(&self) -> Result<Exchange> {
        match self.exchange.to_ascii_lowercase().as_str() {
            "binance"  => Ok(Exchange::Binance),
            "coinbase" => Ok(Exchange::Coinbase),
            "kraken"   => Ok(Exchange::Kraken),
            "bitstamp" => Ok(Exchange::Bitstamp),
            other      => bail!("unknown exchange: {other:?}"),
        }
    }

    fn year_month(&self) -> Result<(i32, u32)> {
        let dt = Utc.timestamp_micros(self.ts_micros)
            .single()
            .context("invalid ts_micros")?;
        Ok((dt.year(), dt.month()))
    }
}

// ── Bucket accumulator ────────────────────────────────────────────────────────

#[derive(Default)]
struct BucketAccumulator {
    pv:            f64,
    volume:        f64,
    notional:      f64,
    buy_vol:       f64,
    sell_vol:      f64,
    has_side_info: bool,
    exchanges:     HashMap<Exchange, f64>,
    tick_count:    u32,
}

impl BucketAccumulator {
    fn add(&mut self, row: &TradeRow, exchange: Exchange) {
        self.pv       += row.price * row.quantity;
        self.volume   += row.quantity;
        self.notional += row.price * row.quantity;
        self.tick_count += 1;
        self.exchanges.insert(exchange, row.price);

        match row.parse_side() {
            Some(TradeSide::Buy)  => { self.buy_vol  += row.quantity; self.has_side_info = true; }
            Some(TradeSide::Sell) => { self.sell_vol += row.quantity; self.has_side_info = true; }
            None => {}
        }
    }

    fn flush(&self, bucket_id: i64, bucket_width_micros: i64) -> Option<FusedTick> {
        if self.volume <= 0.0 { return None; }
        let price     = self.pv / self.volume;
        let buy_ratio = if self.has_side_info {
            let total = self.buy_vol + self.sell_vol;
            if total > 0.0 { Some(self.buy_vol / total) } else { None }
        } else {
            None
        };
        let cross_exchange_spread = if self.exchanges.len() >= 2 {
            let hi = self.exchanges.values().cloned().fold(f64::MIN, f64::max);
            let lo = self.exchanges.values().cloned().fold(f64::MAX, f64::min);
            hi - lo
        } else {
            0.0
        };
        Some(FusedTick {
            ts_micros: bucket_id * bucket_width_micros + bucket_width_micros / 2,
            price,
            volume:                self.volume,
            notional:              self.notional,
            buy_ratio,
            exchange_count:        self.exchanges.len().max(1) as u8,
            tick_count:            self.tick_count,
            exchange_prices:       self.exchanges.clone(),
            cross_exchange_spread,
            symbol:                Symbol::BtcUsd,
        })
    }
}

// ── Feature normalisation ─────────────────────────────────────────────────────

fn feature_array(f: &FeatureVector) -> [f64; N_FEATURES] {
    [
        f.rsi_14.unwrap_or(50.0) / 100.0,
        f.vwap_deviation.unwrap_or(0.0),
        f.momentum_micro.unwrap_or(0.0),
        f.momentum_short.unwrap_or(0.0),
        f.ewma_vol_tick.unwrap_or(0.001),
        f.tick_velocity / 20.0,
        f.ofi_30s,
        f.ofi_300s,
        f.autocorr_lag1.unwrap_or(0.0),
        f.realised_vol_30s.unwrap_or(0.001),
        f.inter_exchange_spread / 100.0,
        (f.price - 30_000.0) / 70_000.0,
        f.ewma_variance,
        f.book_imbalance_top5.unwrap_or(0.0),
        f.book_imbalance_full.unwrap_or(0.0),
        f.book_weighted_mid.map(|m| (m - 30_000.0) / 70_000.0).unwrap_or(0.0),
        f.book_spread_usd.map(|s| s / 100.0).unwrap_or(0.0),
    ]
}

// ── Per-shard stats (feeds manifest.json) ─────────────────────────────────────

#[derive(Debug, Default, Clone, Serialize)]
struct ShardStats {
    file:           String,
    year:           i32,
    month:          u32,
    rows:           u64,
    bearish:        u64,
    sideways:       u64,
    bullish:        u64,
    first_ts_micros: Option<i64>,
    last_ts_micros:  Option<i64>,
}

impl ShardStats {
    fn record_label(&mut self, label: u8, ts: i64) {
        match label {
            0 => self.bearish  += 1,
            1 => self.sideways += 1,
            2 => self.bullish  += 1,
            _ => {}
        }
        self.rows += 1;
        if self.first_ts_micros.is_none() { self.first_ts_micros = Some(ts); }
        self.last_ts_micros = Some(ts);
    }
}

// ── Manifest ──────────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct ManifestTotal {
    rows:     u64,
    bearish:  u64,
    sideways: u64,
    bullish:  u64,
}

#[derive(Serialize)]
struct Manifest {
    generated_at:      String,
    label_window_secs: i64,
    label_threshold:   f64,
    bucket_ms:         i64,
    features:          Vec<String>,
    total:             ManifestTotal,
    shards:            Vec<ShardStats>,
}

// ── Shard writer ──────────────────────────────────────────────────────────────

/// Owns the currently-open shard CSV writer plus its accumulated stats.
/// Call `roll(year, month)` to close the current shard and open a new one.
struct ShardWriter {
    output_dir:   PathBuf,
    current:      Option<(i32, u32)>,           // (year, month) of open shard
    wtr:          Option<csv::Writer<BufWriter<File>>>,
    stats:        ShardStats,
    all_stats:    Vec<ShardStats>,              // one entry per closed shard
}

impl ShardWriter {
    fn new(output_dir: PathBuf) -> Self {
        Self {
            output_dir,
            current:   None,
            wtr:       None,
            stats:     ShardStats::default(),
            all_stats: Vec::new(),
        }
    }

    /// Open a new monthly shard CSV, writing the header row.
    fn open_shard(&mut self, year: i32, month: u32) -> Result<()> {
        let filename = format!("{year}-{month:02}.csv");
        let path     = self.output_dir.join(&filename);
        let file     = File::create(&path)
            .with_context(|| format!("creating shard {filename}"))?;
        let mut wtr  = csv::Writer::from_writer(BufWriter::new(file));

        // Header: all feature names + "label"
        let mut header: Vec<&str> = FEATURE_NAMES.to_vec();
        header.push("label");
        wtr.write_record(&header)?;

        self.wtr     = Some(wtr);
        self.current = Some((year, month));
        self.stats   = ShardStats { file: filename, year, month, ..Default::default() };
        Ok(())
    }

    /// If (year, month) differs from the currently open shard, flush and
    /// close the current one, then open a fresh shard.
    ///
    /// The feature state and ring buffer in `Pipeline` are intentionally
    /// NOT reset here — only the writer is swapped.  This preserves indicator
    /// history and in-flight label-window entries across month boundaries.
    fn roll(&mut self, year: i32, month: u32) -> Result<()> {
        match self.current {
            Some((y, m)) if y == year && m == month => return Ok(()),
            Some(_) => self.close_current()?,
            None    => {}
        }
        self.open_shard(year, month)
    }

    /// Close the currently-open shard, flushing the writer and archiving stats.
    fn close_current(&mut self) -> Result<()> {
        if let Some(mut wtr) = self.wtr.take() {
            wtr.flush()?;
        }
        if self.stats.rows > 0 {
            let stats = std::mem::take(&mut self.stats);
            eprintln!(
                "  Closed shard {} — {} rows  \
                 (bear={} side={} bull={})",
                stats.file, stats.rows,
                stats.bearish, stats.sideways, stats.bullish,
            );
            self.all_stats.push(stats);
        }
        self.current = None;
        Ok(())
    }

    /// Write one labelled feature row.  `ts_micros` determines which shard
    /// the row belongs to (labels are written to the shard of their *past*
    /// price timestamp, not the future one).
    fn write_row(
        &mut self,
        ts_micros: i64,
        feats: &[f64; N_FEATURES],
        label: u8,
    ) -> Result<()> {
        // Determine calendar month from the row's own timestamp.
        let dt = Utc.timestamp_micros(ts_micros)
            .single()
            .context("invalid ts_micros in labelled row")?;
        self.roll(dt.year(), dt.month())?;

        let wtr = self.wtr.as_mut().expect("shard writer open after roll()");
        let mut record: Vec<String> = feats.iter().map(|v| format!("{v:.8}")).collect();
        record.push(label.to_string());
        wtr.write_record(&record)?;

        self.stats.record_label(label, ts_micros);
        Ok(())
    }

    /// Flush and close the last open shard; return all accumulated shard stats.
    fn finish(mut self) -> Result<Vec<ShardStats>> {
        self.close_current()?;
        Ok(self.all_stats)
    }
}

// ── Pipeline ──────────────────────────────────────────────────────────────────

/// Stateful feature pipeline.  The writer is now a `ShardWriter` that
/// transparently rolls to a new CSV file on each calendar-month boundary.
///
/// Critical invariant: `FeatureState` and `ring` are continuous across shard
/// rolls — they are never reset mid-stream.
struct Pipeline {
    state:               FeatureState,
    acc:                 BucketAccumulator,
    current_bucket:      Option<i64>,
    last_ts:             Option<i64>,
    bucket_width_micros: i64,
    label_threshold:     f64,

    /// Fixed-size ring: (ts_micros_of_past_row, past_price, past_features).
    /// Capacity = label_buckets + 1.
    ring:          VecDeque<(i64, f64, [f64; N_FEATURES])>,
    label_buckets: usize,

    shard_wtr:    ShardWriter,
    row_count:    u64,
    bucket_count: u64,
}

impl Pipeline {
    fn new(
        output_dir: PathBuf,
        bucket_width_micros: i64,
        label_buckets: usize,
        label_threshold: f64,
    ) -> Self {
        Self {
            state: FeatureState::new(),
            acc:   BucketAccumulator::default(),
            current_bucket: None,
            last_ts:        None,
            bucket_width_micros,
            label_threshold,
            ring:          VecDeque::with_capacity(label_buckets + 1),
            label_buckets,
            shard_wtr:    ShardWriter::new(output_dir),
            row_count:    0,
            bucket_count: 0,
        }
    }

    fn flush_bucket(&mut self, bucket_id: i64) -> Result<()> {
        if let Some(fused) = self.acc.flush(bucket_id, self.bucket_width_micros) {
            let fv    = self.state.update_from_fused(&fused);
            let entry = (fused.ts_micros, fv.price, feature_array(&fv));
            self.ring.push_back(entry);
            self.bucket_count += 1;

            if self.ring.len() > self.label_buckets {
                let (past_ts, past_price, past_feats) = self.ring[0];
                let (_,       future_price, _)        = *self.ring.back().unwrap();

                let ret   = (future_price - past_price) / past_price;
                let label: u8 = if ret > self.label_threshold {
                    2 // Bullish
                } else if ret < -self.label_threshold {
                    0 // Bearish
                } else {
                    1 // Sideways
                };

                // Write to whichever monthly shard owns past_ts.
                self.shard_wtr.write_row(past_ts, &past_feats, label)?;
                self.ring.pop_front();
            }
        }
        self.acc = BucketAccumulator::default();
        Ok(())
    }

    fn process_row(&mut self, row: TradeRow) -> Result<()> {
        self.row_count += 1;

        if let Some(prev) = self.last_ts {
            anyhow::ensure!(
                row.ts_micros >= prev,
                "input not sorted at row {} ({} < {prev}) — sort first",
                self.row_count, row.ts_micros,
            );
        }
        self.last_ts = Some(row.ts_micros);

        let exchange  = row.parse_exchange()?;
        let bucket_id = row.ts_micros.div_euclid(self.bucket_width_micros);

        match self.current_bucket {
            None => self.current_bucket = Some(bucket_id),
            Some(cur) if bucket_id != cur => {
                for b in cur..bucket_id {
                    self.flush_bucket(b)?;
                }
                self.current_bucket = Some(bucket_id);
            }
            _ => {}
        }

        self.acc.add(&row, exchange);
        Ok(())
    }

    fn finish(mut self) -> Result<Vec<ShardStats>> {
        if let Some(cur) = self.current_bucket {
            self.flush_bucket(cur)?;
        }
        self.shard_wtr.finish()
    }
}

// ── Manifest writer ───────────────────────────────────────────────────────────

fn write_manifest(
    output_dir: &std::path::Path,
    shards: &[ShardStats],
    args: &Args,
) -> Result<()> {
    let total = ManifestTotal {
        rows:     shards.iter().map(|s| s.rows).sum(),
        bearish:  shards.iter().map(|s| s.bearish).sum(),
        sideways: shards.iter().map(|s| s.sideways).sum(),
        bullish:  shards.iter().map(|s| s.bullish).sum(),
    };
    let manifest = Manifest {
        generated_at:      Utc::now().to_rfc3339(),
        label_window_secs: args.label_window_secs,
        label_threshold:   args.label_threshold,
        bucket_ms:         args.bucket_ms,
        features:          FEATURE_NAMES.iter().map(|s| s.to_string()).collect(),
        total,
        shards:            shards.to_vec(),
    };
    let path = output_dir.join("manifest.json");
    // Write to a temp file first, then rename atomically so a partial write
    // never leaves a corrupt manifest.
    let tmp_path = output_dir.join("manifest.json.tmp");
    {
        let f = File::create(&tmp_path)
            .context("creating manifest.json.tmp")?;
        serde_json::to_writer_pretty(BufWriter::new(f), &manifest)
            .context("serialising manifest")?;
    }
    fs::rename(&tmp_path, &path).context("renaming manifest into place")?;
    Ok(())
}

// ── main ──────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let args = Args::parse();

    if args.input.is_none() && args.data_dir.is_none() {
        bail!("one of --input <csv> or --data-dir <path> is required");
    }

    fs::create_dir_all(&args.output_dir)
        .with_context(|| format!("creating output dir {}", args.output_dir.display()))?;

    let bucket_width_micros = args.bucket_ms * 1_000;
    let label_window_micros = args.label_window_secs * 1_000_000;
    let label_buckets       = (label_window_micros / bucket_width_micros).max(1) as usize;

    eprintln!(
        "Config — bucket={}ms  label_window={}s  threshold={}  ring_depth={}",
        args.bucket_ms, args.label_window_secs, args.label_threshold, label_buckets,
    );

    let mut pipeline = Pipeline::new(
        args.output_dir.clone(),
        bucket_width_micros,
        label_buckets,
        args.label_threshold,
    );

    // ── Mode A: pre-merged CSV ────────────────────────────────────────────────
    if let Some(input_path) = &args.input {
        eprintln!("Reading {}", input_path.display());
        let rdr = BufReader::new(
            File::open(input_path)
                .with_context(|| format!("opening {}", input_path.display()))?,
        );
        let mut csv_rdr = csv::Reader::from_reader(rdr);
        for result in csv_rdr.deserialize() {
            let row: TradeRow = result.context("parsing trade row")?;
            pipeline.process_row(row)?;
        }
    }
    // ── Mode B: Parquet store — one month at a time ───────────────────────────
    else if let Some(data_dir) = &args.data_dir {
        let exchange_refs: Vec<&str> = args.exchanges.iter().map(|s| s.as_str()).collect();
        eprintln!(
            "Reading Parquet store at {} (exchanges: {})",
            data_dir.display(),
            exchange_refs.join(", "),
        );

        for month_result in historical::store::iter_months_sorted(data_dir, &exchange_refs)
            .context("opening Parquet store")?
        {
            let month_rows = month_result.context("reading month shard")?;
            if month_rows.is_empty() { continue; }

            // Log which month we are processing using the first row's timestamp.
            if let Some(first) = month_rows.first() {
                if let Some(dt) = Utc.timestamp_micros(first.ts_micros).single() {
                    eprintln!("  Processing {}-{:02} ({} rows) …",
                        dt.year(), dt.month(), month_rows.len());
                }
            }

            for store_row in month_rows {
                pipeline.process_row(TradeRow {
                    ts_micros: store_row.ts_micros,
                    price:     store_row.price,
                    quantity:  store_row.quantity,
                    side:      store_row.side.unwrap_or_default(),
                    exchange:  store_row.exchange,
                })?;
            }
            // Vec<OwnedRow> is dropped here — memory freed before next month loads.
        }
    }

    let shards = pipeline.finish()?;

    let total_rows: u64    = shards.iter().map(|s| s.rows).sum();
    let total_bearish: u64 = shards.iter().map(|s| s.bearish).sum();
    let total_side: u64    = shards.iter().map(|s| s.sideways).sum();
    let total_bull: u64    = shards.iter().map(|s| s.bullish).sum();

    anyhow::ensure!(
        total_rows > 0,
        "no labelled rows produced — check that there is at least \
         {} seconds of data ({} buckets) in the input",
        args.label_window_secs, label_buckets,
    );

    write_manifest(&args.output_dir, &shards, &args)
        .context("writing manifest.json")?;

    eprintln!();
    eprintln!("Output dir   : {}", args.output_dir.display());
    eprintln!("Shards written: {}", shards.len());
    eprintln!("Total rows   : {total_rows}");
    eprintln!(
        "Label split  : Bearish {total_bearish} ({:.1}%)  \
                        Sideways {total_side} ({:.1}%)  \
                        Bullish {total_bull} ({:.1}%)",
        100.0 * total_bearish as f64 / total_rows as f64,
        100.0 * total_side    as f64 / total_rows as f64,
        100.0 * total_bull    as f64 / total_rows as f64,
    );
    if total_side as f64 / total_rows as f64 > 0.9 {
        eprintln!(
            "WARNING: Sideways >90% — consider lowering --label-threshold \
             or the model will trivially predict Sideways."
        );
    }

    Ok(())
}
