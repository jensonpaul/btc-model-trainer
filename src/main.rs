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
//! ## Labelling: volatility-adaptive z-score (triple-barrier style)
//!
//! Instead of a fixed return threshold, each label is assigned by comparing
//! the realised future return against a horizon-scaled volatility estimate:
//!
//! ```text
//!   sigma_window  = std(log_returns over [past_ts - vol_window, past_ts])
//!   sigma_h       = sigma_window * sqrt(label_buckets / vol_buckets)
//!   z             = future_log_return / max(sigma_h, MIN_VOL_FLOOR)
//!
//!   Bullish   if z  >  k   (--label-vol-multiple, default 0.5)
//!   Bearish   if z  < -k
//!   Sideways  otherwise
//! ```
//!
//! The volatility estimate is **strictly causal**: only bucket log-returns with
//! `ts ≤ past_ts` are used; future prices never bleed into the label.
//!
//! ## manifest.json schema
//!
//! ```json
//! {
//!   "generated_at": "2024-06-01T12:00:00Z",
//!   "label_window_secs": 300,
//!   "label_vol_window_secs": 1800,
//!   "label_vol_multiple": 0.5,
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
//!     --label-window-secs 300 \
//!     --label-vol-window-secs 1800 \
//!     --label-vol-multiple 0.5
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

use ryu::Buffer as RyuBuffer;

use anyhow::{bail, Context, Result};
use chrono::{Datelike, TimeZone, Utc};
use clap::Parser;
use serde::{Deserialize, Serialize};

use btc_prediction_engine::features::{FeatureState, FeatureVector};
use btc_prediction_engine::price_fusion::FusedTick;
use btc_prediction_engine::types::{Exchange, Symbol, TradeSide};

// ── Constants ─────────────────────────────────────────────────────────────────

const FEATURE_NAMES: &[&str] = &[
    // Returns — time-horizon aligned, all log-space.
    "return_5s",
    "return_30s",
    "return_300s",
    // Volatility — same horizon hierarchy as returns.
    "vol_30s",
    "vol_300s",
    "vol_1800s",
    "vol_ratio",          // vol_30s / vol_1800s
    // Order flow imbalance.
    "ofi_5s",
    "ofi_30s",
    "ofi_300s",
    "ofi_delta_30s",      // ofi_5s − ofi_30s
    "buy_ratio_30s",
    "buy_ratio_300s",
    // VWAP deviation — z-scored by vol_1800s.
    "vwap_dev_30s",
    "vwap_dev_300s",
    // Volume activity.
    "volume_ratio",       // volume_30s / volume_300s
    // Tick activity.
    "tick_velocity",
    "activity_regime",    // tick_rate_30s / tick_rate_1800s
    // Cross-exchange spread (percentage, scale-normalised).
    "spread_pct",
    // Order book.
    "book_imb5",
    "book_imb_full",
    "book_spread_pct",
    "book_pressure",
    // Regime features.
    "trend_strength",     // abs(return_300s) / vol_1800s
    "vol_regime",         // vol_300s / vol_1800s
    // z-scored returns — the key cross-regime generalisation features.
    "zreturn_30s",
    "zreturn_300s",
];
//const N_FEATURES: usize = 28;
const N_FEATURES: usize = FEATURE_NAMES.len();

/// Minimum annualised-equivalent per-bucket vol floor (10 bps expressed as a
/// fraction). Prevents degenerate z-scores during dead markets or data gaps.
const MIN_VOL_FLOOR: f64 = 0.001;

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

    /// Trailing window in seconds used to estimate realised volatility at the
    /// time of labelling.  Longer = more stable but slower to adapt.
    /// Must be ≥ label_window_secs for meaningful scaling.
    #[arg(long, default_value_t = 1800)]
    label_vol_window_secs: i64,

    /// Number of standard deviations (horizon-scaled) required to call a move
    /// Bullish or Bearish.  Lower → more directional labels, higher → more
    /// Sideways.  Tune to target class balance for your loss function.
    #[arg(long, default_value_t = 0.5)]
    label_vol_multiple: f64,
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
    // Parallel fixed arrays: slot i corresponds to EXCHANGE_SLOTS[i].
    exch_price:    [f64; 4],
    exch_seen:     u8, // bitmask over the 4 slots
    tick_count:    u32,
}

/// Fixed slot order — independent of Exchange's internal discriminants.
const EXCHANGE_SLOTS: [Exchange; 4] = [
    Exchange::Binance,
    Exchange::Coinbase,
    Exchange::Kraken,
    Exchange::Bitstamp,
];

fn exchange_slot(e: Exchange) -> usize {
    match e {
        Exchange::Binance  => 0,
        Exchange::Coinbase => 1,
        Exchange::Kraken   => 2,
        Exchange::Bitstamp => 3,
        _ => unreachable!("unsupported exchange variant: {e:?} — update EXCHANGE_SLOTS/exchange_slot"),
    }
}

impl BucketAccumulator {
    fn add(&mut self, row: &TradeRow, exchange: Exchange) {
        self.pv       += row.price * row.quantity;
        self.volume   += row.quantity;
        self.notional += row.price * row.quantity;
        self.tick_count += 1;

        let idx = exchange_slot(exchange);
        self.exch_price[idx] = row.price;
        self.exch_seen |= 1 << idx;

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

        let exchange_count = self.exch_seen.count_ones() as u8;
        let mut exchange_prices = HashMap::with_capacity(exchange_count as usize);
        let mut hi = f64::MIN;
        let mut lo = f64::MAX;
        for i in 0..4 {
            if self.exch_seen & (1 << i) != 0 {
                let p = self.exch_price[i];
                exchange_prices.insert(EXCHANGE_SLOTS[i], p);
                hi = hi.max(p);
                lo = lo.min(p);
            }
        }
        let cross_exchange_spread = if exchange_count >= 2 { hi - lo } else { 0.0 };

        Some(FusedTick {
            ts_micros: bucket_id * bucket_width_micros + bucket_width_micros / 2,
            price,
            volume: self.volume,
            notional: self.notional,
            buy_ratio,
            exchange_count: exchange_count.max(1),
            tick_count: self.tick_count,
            exchange_prices,
            cross_exchange_spread,
            symbol: Symbol::BtcUsd,
        })
    }
}

// ── Feature normalisation ─────────────────────────────────────────────────────

fn feature_array(f: &FeatureVector) -> [f32; N_FEATURES] {
    // Defaults for seeding period:
    //   · Signed features (returns, OFI, imbalance, VWAP dev) → 0.0
    //   · Volatility features → small positive floor (avoids divide-by-zero
    //     in downstream z-score computations during replay warm-up)
    //   · Ratio features → 1.0 (neutral: short == long)
    //   · Buy ratio → 0.5 (balanced)
    //   · Trend/regime strength → 0.0 (no trend assumed)
    const VOL_FLOOR: f32 = 0.001;

    [
        // Returns.
        f.return_5s.unwrap_or(0.0)   as f32,
        f.return_30s.unwrap_or(0.0)  as f32,
        f.return_300s.unwrap_or(0.0) as f32,

        // Volatility.
        f.vol_30s.unwrap_or(VOL_FLOOR as f64)   as f32,
        f.vol_300s.unwrap_or(VOL_FLOOR as f64)  as f32,
        f.vol_1800s.unwrap_or(VOL_FLOOR as f64) as f32,
        f.vol_ratio.unwrap_or(1.0)               as f32,

        // Order flow imbalance.
        f.ofi_5s          as f32,
        f.ofi_30s         as f32,
        f.ofi_300s        as f32,
        f.ofi_delta_30s   as f32,
        f.buy_ratio_30s   as f32,
        f.buy_ratio_300s  as f32,

        // VWAP deviation (z-scored).
        f.vwap_dev_30s.unwrap_or(0.0)  as f32,
        f.vwap_dev_300s.unwrap_or(0.0) as f32,

        // Volume activity ratio.
        f.volume_ratio.unwrap_or(1.0) as f32,

        // Tick activity — no normalisation needed (ratio is already dimensionless;
        // raw rate is already in ticks/s and trees handle the scale).
        (f.tick_velocity / 20.0)  as f32,   // soft-clip: /20 puts typical 5–15 t/s into 0.25–0.75
        f.activity_regime         as f32,

        // Cross-exchange spread (already percentage).
        f.spread_pct as f32,

        // Order book.
        f.book_imbalance_5.unwrap_or(0.0)    as f32,
        f.book_imbalance_full.unwrap_or(0.0) as f32,
        f.book_spread_pct.unwrap_or(0.0)     as f32,
        f.book_pressure.unwrap_or(0.0)       as f32,

        // Regime.
        f.trend_strength.unwrap_or(0.0) as f32,
        f.vol_regime.unwrap_or(1.0)     as f32,

        // z-scored returns — the primary cross-regime generalisation signal.
        f.zreturn_30s.unwrap_or(0.0)  as f32,
        f.zreturn_300s.unwrap_or(0.0) as f32,
    ]
}

// ── Volatility tracker ────────────────────────────────────────────────────────

/// Causal trailing realised-vol estimator for label z-scoring.
///
/// Maintains a deque of `(ts_micros, log_return)` pairs for consecutive
/// bucket mid-prices.  When queried at `past_ts`, it expels entries older
/// than `past_ts - vol_window_micros` and returns the sample std of the
/// remaining returns — using **only data that was available at `past_ts`**.
///
/// Horizon scaling converts per-bucket vol to expected std over the full
/// `label_buckets`-wide window under a random-walk assumption:
///
/// ```
///   sigma_h = sigma_bucket * sqrt(label_buckets)
/// ```
///
/// The caller holds onto this struct across the entire stream; it is never
/// reset across shard boundaries.
/// Causal O(1) trailing realised-vol estimator for label z-scoring.
///
/// Maintains running `sum` and `sum_sq` of log-returns in the trailing
/// window. Push/evict are O(1); `sigma_at` is O(1) amortized (eviction
/// only walks past entries once, ever).
struct LabelVolTracker {
    /// (ts_micros, log_return), sorted ascending.
    returns: VecDeque<(i64, f64)>,
    sum:     f64,
    sum_sq:  f64,
    vol_window_micros: i64,
    label_buckets:     usize,
    prev_price:        Option<f64>,
}

impl LabelVolTracker {
    fn new(vol_window_micros: i64, label_buckets: usize) -> Self {
        Self {
            returns: VecDeque::new(),
            sum: 0.0,
            sum_sq: 0.0,
            vol_window_micros,
            label_buckets,
            prev_price: None,
        }
    }

    fn push_bucket(&mut self, ts_micros: i64, price: f64) {
        if let Some(prev) = self.prev_price {
            if prev > 0.0 && price > 0.0 {
                let log_ret = (price / prev).ln();
                self.returns.push_back((ts_micros, log_ret));
                self.sum    += log_ret;
                self.sum_sq += log_ret * log_ret;
            }
        }
        self.prev_price = Some(price);
    }

    /// Evict entries older than `cutoff_ts - vol_window_micros`, updating
    /// the running sums incrementally. O(1) amortized: each entry is
    /// evicted at most once across the whole stream.
    fn evict(&mut self, cutoff_ts: i64) {
        let oldest_allowed = cutoff_ts - self.vol_window_micros;
        while let Some(&(ts, r)) = self.returns.front() {
            if ts < oldest_allowed {
                self.sum    -= r;
                self.sum_sq -= r * r;
                self.returns.pop_front();
            } else {
                break;
            }
        }
    }

    /// Horizon-scaled vol using only data with `ts_micros <= cutoff_ts`.
    fn sigma_at(&mut self, cutoff_ts: i64) -> f64 {
        self.evict(cutoff_ts);

        let n = self.returns.len();
        if n < 2 {
            return MIN_VOL_FLOOR;
        }
        let n_f = n as f64;

        // Bessel-corrected variance from running sums:
        // var = (sum_sq - sum^2/n) / (n-1)
        let mut variance = (self.sum_sq - self.sum * self.sum / n_f) / (n_f - 1.0);
        // Guard against tiny negative values from floating-point cancellation.
        if variance < 0.0 {
            variance = 0.0;
        }

        let sigma_bucket = variance.sqrt();
        let sigma_h = sigma_bucket * (self.label_buckets as f64).sqrt();
        sigma_h.max(MIN_VOL_FLOOR)
    }
}

// ── Per-shard stats (feeds manifest.json) ─────────────────────────────────────

#[derive(Debug, Default, Clone, Serialize)]
struct ShardStats {
    features_file:   String,
    labels_file:     String,
    year:            i32,
    month:           u32,
    rows:            u64,
    bearish:         u64,
    sideways:        u64,
    bullish:         u64,
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
    generated_at:          String,
    label_window_secs:     i64,
    label_vol_window_secs: i64,
    label_vol_multiple:    f64,
    bucket_ms:             i64,
    n_features:            usize,
    feature_dtype:         &'static str, // "f32"
    label_dtype:           &'static str, // "u8"
    features:              Vec<String>,
    total:                 ManifestTotal,
    shards:                Vec<ShardStats>,
}

// ── Shard writer ──────────────────────────────────────────────────────────────

/// Owns the currently-open shard CSV writer plus its accumulated stats.
/// Call `roll(year, month)` to close the current shard and open a new one.
struct ShardWriter {
    output_dir: PathBuf,
    current:    Option<(i32, u32)>,
    feat_wtr:   Option<BufWriter<File>>,
    label_wtr:  Option<BufWriter<File>>,
    stats:      ShardStats,
    all_stats:  Vec<ShardStats>,
}

impl ShardWriter {
    fn new(output_dir: PathBuf) -> Self {
        Self {
            output_dir,
            current:   None,
            feat_wtr:  None,
            label_wtr: None,
            stats:     ShardStats::default(),
            all_stats: Vec::new(),
        }
    }

    fn open_shard(&mut self, year: i32, month: u32) -> Result<()> {
        let feat_name  = format!("{year}-{month:02}.features.f32");
        let label_name = format!("{year}-{month:02}.labels.u8");

        let feat_file  = File::create(self.output_dir.join(&feat_name))
            .with_context(|| format!("creating {feat_name}"))?;
        let label_file = File::create(self.output_dir.join(&label_name))
            .with_context(|| format!("creating {label_name}"))?;

        self.feat_wtr  = Some(BufWriter::new(feat_file));
        self.label_wtr = Some(BufWriter::new(label_file));
        self.current   = Some((year, month));
        self.stats     = ShardStats {
            features_file: feat_name,
            labels_file:   label_name,
            year, month,
            ..Default::default()
        };
        Ok(())
    }

    fn roll(&mut self, year: i32, month: u32) -> Result<()> {
        match self.current {
            Some((y, m)) if y == year && m == month => return Ok(()),
            Some(_) => self.close_current()?,
            None    => {}
        }
        self.open_shard(year, month)
    }

    fn close_current(&mut self) -> Result<()> {
        if let Some(mut w) = self.feat_wtr.take()  { w.flush()?; }
        if let Some(mut w) = self.label_wtr.take() { w.flush()?; }
        if self.stats.rows > 0 {
            let stats = std::mem::take(&mut self.stats);
            eprintln!(
                "  Closed shard {} — {} rows  (bear={} side={} bull={})",
                stats.features_file, stats.rows,
                stats.bearish, stats.sideways, stats.bullish,
            );
            self.all_stats.push(stats);
        }
        self.current = None;
        Ok(())
    }

    fn write_row(
        &mut self,
        ts_micros: i64,
        feats: &[f32; N_FEATURES],
        label: u8,
    ) -> Result<()> {
        let dt = Utc.timestamp_micros(ts_micros).single()
            .context("invalid ts_micros in labelled row")?;
        self.roll(dt.year(), dt.month())?;

        // SAFETY: [f32; N] is Plain Old Data — no padding, no invalid
        // bitpatterns matter for f32 (NaN/inf are valid f32 values and
        // round-trip fine through raw bytes). bytemuck enforces this at
        // compile time via the Pod/NoUninit bounds.
        let bytes: &[u8] = bytemuck::bytes_of(feats);
        self.feat_wtr.as_mut().unwrap().write_all(bytes)?;
        self.label_wtr.as_mut().unwrap().write_all(&[label])?;

        self.stats.record_label(label, ts_micros);
        Ok(())
    }

    fn finish(mut self) -> Result<Vec<ShardStats>> {
        self.close_current()?;
        Ok(self.all_stats)
    }
}

// ── Pipeline ──────────────────────────────────────────────────────────────────

/// Stateful feature + labelling pipeline.
///
/// Three independent state machines run in lockstep:
///
/// 1. `FeatureState` — indicator state (RSI, VWAP, EWMA, …).  Never reset.
/// 2. `ring`         — fixed-depth deque holding `(ts, price, features)` for
///                     the `label_buckets` most recent buckets; used to
///                     compute the future return when the head is popped.
/// 3. `vol_tracker`  — causal trailing-vol estimator; queried at `past_ts`
///                     (the timestamp of the row being labelled) to obtain a
///                     lookahead-free volatility estimate.
///
/// Critical invariant: all three are continuous across shard rolls.
struct Pipeline {
    state:               FeatureState,
    acc:                 BucketAccumulator,
    current_bucket:      Option<i64>,
    last_ts:             Option<i64>,
    bucket_width_micros: i64,

    /// Volatility-adaptive threshold multiplier.
    label_vol_multiple: f64,

    /// Ring: (ts_micros, mid_price, features) for each emitted bucket.
    /// Capacity = label_buckets + 1.
    ring: VecDeque<(i64, f64, [f32; N_FEATURES])>,
    label_buckets: usize,

    /// Causal vol tracker — separate from ring so it accumulates every
    /// bucket price whether or not a label has been popped yet.
    vol_tracker: LabelVolTracker,

    shard_wtr:    ShardWriter,
    row_count:    u64,
    bucket_count: u64,
}

impl Pipeline {
    fn new(
        output_dir:            PathBuf,
        bucket_width_micros:   i64,
        label_buckets:         usize,
        vol_window_micros:     i64,
        label_vol_multiple:    f64,
    ) -> Self {
        Self {
            state:               FeatureState::new(),
            acc:                 BucketAccumulator::default(),
            current_bucket:      None,
            last_ts:             None,
            bucket_width_micros,
            label_vol_multiple,
            ring:                VecDeque::with_capacity(label_buckets + 1),
            label_buckets,
            vol_tracker:         LabelVolTracker::new(vol_window_micros, label_buckets),
            shard_wtr:           ShardWriter::new(output_dir),
            row_count:           0,
            bucket_count:        0,
        }
    }

    fn flush_bucket(&mut self, bucket_id: i64) -> Result<()> {
        if let Some(fused) = self.acc.flush(bucket_id, self.bucket_width_micros) {
            let fv       = self.state.update_from_fused(&fused);
            let mid_ts   = fused.ts_micros;
            let mid_price = fv.price;

            // Update the vol tracker with this bucket's price *before*
            // deciding whether to pop a label — the label uses data up to
            // past_ts, which is strictly earlier than mid_ts.
            self.vol_tracker.push_bucket(mid_ts, mid_price);

            self.ring.push_back((mid_ts, mid_price, feature_array(&fv)));
            self.bucket_count += 1;

            if self.ring.len() > self.label_buckets {
                let (past_ts, past_price, past_feats) = self.ring[0];
                let (_,       future_price, _)        = *self.ring.back().unwrap();

                // --- Volatility-adaptive z-score label ---
                //
                // sigma_h is the expected std of the log-return over the
                // label horizon, computed causally at past_ts.
                let sigma_h = self.vol_tracker.sigma_at(past_ts);

                // Use log-return: more symmetric, additive across time steps.
                let log_return = if past_price > 0.0 && future_price > 0.0 {
                    (future_price / past_price).ln()
                } else {
                    0.0
                };

                let z = log_return / sigma_h;

                let label: u8 = if z > self.label_vol_multiple {
                    2 // Bullish
                } else if z < -self.label_vol_multiple {
                    0 // Bearish
                } else {
                    1 // Sideways
                };

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
    output_dir:  &std::path::Path,
    shards:      &[ShardStats],
    args:        &Args,
) -> Result<()> {
    let total = ManifestTotal {
        rows:     shards.iter().map(|s| s.rows).sum(),
        bearish:  shards.iter().map(|s| s.bearish).sum(),
        sideways: shards.iter().map(|s| s.sideways).sum(),
        bullish:  shards.iter().map(|s| s.bullish).sum(),
    };
let manifest = Manifest {
    generated_at:          Utc::now().to_rfc3339(),
    label_window_secs:     args.label_window_secs,
    label_vol_window_secs: args.label_vol_window_secs,
    label_vol_multiple:    args.label_vol_multiple,
    bucket_ms:             args.bucket_ms,
    n_features:            N_FEATURES,
    feature_dtype:         "f32",
    label_dtype:           "u8",
    features:              FEATURE_NAMES.iter().map(|s| s.to_string()).collect(),
    total,
    shards:                shards.to_vec(),
};
    let path     = output_dir.join("manifest.json");
    let tmp_path = output_dir.join("manifest.json.tmp");
    {
        let f = File::create(&tmp_path).context("creating manifest.json.tmp")?;
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
    if args.label_vol_multiple <= 0.0 {
        bail!("--label-vol-multiple must be positive");
    }
    if args.label_vol_window_secs < args.label_window_secs {
        eprintln!(
            "WARNING: --label-vol-window-secs ({}) < --label-window-secs ({}). \
             Vol estimate will be noisier than the label horizon; \
             consider setting label-vol-window-secs ≥ label-window-secs.",
            args.label_vol_window_secs, args.label_window_secs,
        );
    }

    fs::create_dir_all(&args.output_dir)
        .with_context(|| format!("creating output dir {}", args.output_dir.display()))?;

    let bucket_width_micros  = args.bucket_ms * 1_000;
    let label_window_micros  = args.label_window_secs * 1_000_000;
    let vol_window_micros    = args.label_vol_window_secs * 1_000_000;
    let label_buckets        = (label_window_micros / bucket_width_micros).max(1) as usize;

    eprintln!(
        "Config — bucket={}ms  label_window={}s  \
         vol_window={}s  vol_multiple={:.3}  ring_depth={}  vol_floor={:.4}",
        args.bucket_ms,
        args.label_window_secs,
        args.label_vol_window_secs,
        args.label_vol_multiple,
        label_buckets,
        MIN_VOL_FLOOR,
    );

    let mut pipeline = Pipeline::new(
        args.output_dir.clone(),
        bucket_width_micros,
        label_buckets,
        vol_window_micros,
        args.label_vol_multiple,
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
    // ── Mode B: Parquet store ─────────────────────────────────────────────────
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

            if let Some(first) = month_rows.first() {
                if let Some(dt) = Utc.timestamp_micros(first.ts_micros).single() {
                    eprintln!(
                        "  Processing {}-{:02} ({} rows) …",
                        dt.year(), dt.month(), month_rows.len()
                    );
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
        }
    }

    let shards = pipeline.finish()?;

    let total_rows:    u64 = shards.iter().map(|s| s.rows).sum();
    let total_bearish: u64 = shards.iter().map(|s| s.bearish).sum();
    let total_side:    u64 = shards.iter().map(|s| s.sideways).sum();
    let total_bull:    u64 = shards.iter().map(|s| s.bullish).sum();

    anyhow::ensure!(
        total_rows > 0,
        "no labelled rows produced — check that there is at least \
         {} seconds of data ({} buckets) in the input",
        args.label_window_secs, label_buckets,
    );

    write_manifest(&args.output_dir, &shards, &args)
        .context("writing manifest.json")?;

    eprintln!();
    eprintln!("Output dir    : {}", args.output_dir.display());
    eprintln!("Shards written: {}", shards.len());
    eprintln!("Total rows    : {total_rows}");
    eprintln!(
        "Label split   : Bearish {total_bearish} ({:.1}%)  \
                         Sideways {total_side} ({:.1}%)  \
                         Bullish {total_bull} ({:.1}%)",
        100.0 * total_bearish as f64 / total_rows as f64,
        100.0 * total_side    as f64 / total_rows as f64,
        100.0 * total_bull    as f64 / total_rows as f64,
    );

    // Actionable warnings for common mis-configurations.
    let sideways_frac = total_side as f64 / total_rows as f64;
    let directional_frac = 1.0 - sideways_frac;
    if sideways_frac > 0.85 {
        eprintln!(
            "WARNING: Sideways {:.1}% > 85% — try lowering --label-vol-multiple \
             (currently {:.2}) or shortening --label-window-secs.",
            sideways_frac * 100.0, args.label_vol_multiple,
        );
    } else if directional_frac > 0.70 {
        eprintln!(
            "WARNING: Directional labels {:.1}% > 70% — vol estimate may be \
             too low. Try raising --label-vol-multiple or --label-vol-window-secs.",
            directional_frac * 100.0,
        );
    }

    let bull_frac = total_bull as f64 / total_rows as f64;
    let bear_frac = total_bearish as f64 / total_rows as f64;
    if (bull_frac - bear_frac).abs() / bull_frac.max(bear_frac).max(1e-9) > 0.3 {
        eprintln!(
            "WARNING: Bull/bear asymmetry detected ({:.1}% / {:.1}%). \
             Consider class-weighted loss or oversampling the minority class.",
            bull_frac * 100.0, bear_frac * 100.0,
        );
    }

    Ok(())
}
