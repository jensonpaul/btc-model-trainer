//! `collect-training-data` — offline historical → labelled feature CSV.
//!
//! Reads a CSV of historical trade ticks (one row per executed trade),
//! buckets them into 100 ms `FusedTick`s exactly like the live fusion stage,
//! replays them through `btc_prediction_engine::features::FeatureState`
//! (the *same* incremental feature code the live engine uses), and emits one
//! row per bucket: the 17-element normalised feature array plus a forward-
//! return direction label.
//!
//! ## Input CSV format
//!
//! ```text
//! ts_micros,price,quantity,side,exchange
//! 1700000000000000,43250.12,0.014,buy,binance
//! 1700000000010000,43250.50,0.002,sell,binance
//! ...
//! ```
//!
//! - `ts_micros`: i64, microseconds since UNIX epoch, **ascending order**.
//! - `price`:     f64, USD.
//! - `quantity`:  f64, BTC.
//! - `side`:      "buy" | "sell" | "" (empty = unknown).
//! - `exchange`:  "binance" | "coinbase" | "kraken" | "bitstamp".
//!
//! Multi-exchange input is supported and recommended: `inter_exchange_spread`
//! and `cross_exchange_spread` are only meaningful with ≥ 2 exchanges feeding
//! the same time buckets. A single-exchange file works too — those features
//! will simply be ~0.
//!
//! ## Where to get historical trade data
//!
//! - Binance: <https://data.binance.vision/> (monthly `aggTrades` ZIP/CSV dumps,
//!   free, no API key).
//! - Kraken:  <https://support.kraken.com/articles/360047124832> (historical
//!   trade data export).
//! - Bitstamp / Coinbase: paid historical-data vendors (e.g. Tardis.dev,
//!   Kaiko) if cross-exchange spread features matter to you.
//!
//! Convert whatever format you obtain into the CSV schema above with a small
//! script; column order doesn't matter (this reads by header name).
//!
//! ## Usage
//!
//! ```bash
//! cargo run --release -- \
//!     --input data/binance_2025_q1_trades.csv \
//!     --output data/training_2025_q1.csv \
//!     --label-window-secs 300 \
//!     --label-threshold 0.001
//! ```
//!
//! ## Output CSV format
//!
//! 17 normalised feature columns (matching `OnnxTrendModel::feature_array`
//! exactly) + `label` (0 = Bearish, 1 = Sideways, 2 = Bullish).

use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use serde::Deserialize;

use btc_prediction_engine::features::{FeatureState, FeatureVector};
use btc_prediction_engine::price_fusion::FusedTick;
use btc_prediction_engine::types::{Exchange, Symbol, TradeSide};

/// CSV header for the output training file.
const OUTPUT_HEADER: &[&str] = &[
    "rsi_14", "vwap_dev", "mom_micro", "mom_short", "ewma_vol",
    "tick_vel", "ofi_30s", "ofi_300s", "autocorr", "rvol_30s",
    "xchg_spread", "price_norm", "ewma_var",
    "book_imb5", "book_imb_full", "book_wmid", "book_spread",
    "label",
];

const N_FEATURES: usize = 17;

#[derive(Parser, Debug)]
#[command(about = "Replay historical BTC/USD trades through the engine's \
                    feature pipeline and emit a labelled ONNX-training CSV")]
struct Args {
    /// Input CSV of historical trade ticks (ts_micros,price,quantity,side,exchange).
    #[arg(long)]
    input: PathBuf,

    /// Output CSV of (feature_array[17], label) rows.
    #[arg(long)]
    output: PathBuf,

    /// Bucket width in milliseconds. Must match `FusionConfig::bucket_width_micros`
    /// used by the live engine (default 100 ms) for the data to be consistent.
    #[arg(long, default_value_t = 100)]
    bucket_ms: i64,

    /// Forward-looking label window in seconds.
    /// e.g. 300 = label each row by the price 5 minutes later.
    #[arg(long, default_value_t = 300)]
    label_window_secs: i64,

    /// Return threshold for Bullish / Bearish vs. Sideways.
    /// e.g. 0.001 = 0.1 %.
    #[arg(long, default_value_t = 0.001)]
    label_threshold: f64,
}

/// One row of the input CSV.
#[derive(Debug, Deserialize)]
struct TradeRow {
    ts_micros: i64,
    price: f64,
    quantity: f64,
    #[serde(default)]
    side: String,
    exchange: String,
}

impl TradeRow {
    fn parse_side(&self) -> Option<TradeSide> {
        match self.side.to_ascii_lowercase().as_str() {
            "buy" | "b" => Some(TradeSide::Buy),
            "sell" | "s" => Some(TradeSide::Sell),
            _ => None,
        }
    }

    fn parse_exchange(&self) -> Result<Exchange> {
        match self.exchange.to_ascii_lowercase().as_str() {
            "binance" => Ok(Exchange::Binance),
            "coinbase" => Ok(Exchange::Coinbase),
            "kraken" => Ok(Exchange::Kraken),
            "bitstamp" => Ok(Exchange::Bitstamp),
            other => anyhow::bail!("unknown exchange: {other:?}"),
        }
    }
}

/// Accumulator for one open fusion bucket. Mirrors the logic in
/// `btc_prediction_engine::price_fusion::run_fuser` closely enough to produce
/// equivalent `FusedTick`s for offline replay (NTP correction is skipped
/// since historical timestamps from a single source are already aligned).
#[derive(Default)]
struct BucketAccumulator {
    bucket_id: Option<i64>,
    pv: f64,           // price * quantity, summed
    volume: f64,
    notional: f64,
    buy_vol: f64,
    sell_vol: f64,
    has_side_info: bool,
    exchanges: HashMap<Exchange, f64>, // last price per exchange in this bucket
    tick_count: u32,
}

impl BucketAccumulator {
    fn add(&mut self, row: &TradeRow, exchange: Exchange) {
        self.pv += row.price * row.quantity;
        self.volume += row.quantity;
        self.notional += row.price * row.quantity;
        self.tick_count += 1;
        self.exchanges.insert(exchange, row.price);

        match row.parse_side() {
            Some(TradeSide::Buy) => {
                self.buy_vol += row.quantity;
                self.has_side_info = true;
            }
            Some(TradeSide::Sell) => {
                self.sell_vol += row.quantity;
                self.has_side_info = true;
            }
            None => {}
        }
    }

    fn flush(&self, bucket_id: i64, bucket_width_micros: i64) -> Option<FusedTick> {
        if self.volume <= 0.0 {
            return None;
        }
        let price = self.pv / self.volume;
        let buy_ratio = if self.has_side_info {
            let total = self.buy_vol + self.sell_vol;
            if total > 0.0 { Some(self.buy_vol / total) } else { None }
        } else {
            None
        };

        let cross_exchange_spread = if self.exchanges.len() >= 2 {
            let max = self.exchanges.values().cloned().fold(f64::MIN, f64::max);
            let min = self.exchanges.values().cloned().fold(f64::MAX, f64::min);
            max - min
        } else {
            0.0
        };

        Some(FusedTick {
            ts_micros: bucket_id * bucket_width_micros + bucket_width_micros / 2,
            price,
            volume: self.volume,
            notional: self.notional,
            buy_ratio,
            exchange_count: self.exchanges.len().max(1) as u8,
            tick_count: self.tick_count,
            exchange_prices: self.exchanges.clone(),
            cross_exchange_spread,
            symbol: Symbol::BtcUsd,
        })
    }
}

/// Mirror of `OnnxTrendModel::feature_array` in
/// `btc-prediction-engine/docs/onnx_trend_model_implementation.md`.
/// **Keep this in sync** with the Rust inference-side normalisation —
/// a mismatch here silently degrades model accuracy.
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

fn main() -> Result<()> {
    let args = Args::parse();
    let bucket_width_micros = args.bucket_ms * 1_000;
    let label_window_micros = args.label_window_secs * 1_000_000;

    eprintln!("Reading {}", args.input.display());
    let reader = BufReader::new(
        File::open(&args.input).with_context(|| format!("opening {:?}", args.input))?,
    );
    let mut rdr = csv::Reader::from_reader(reader);

    // ── Pass 1: bucket raw trades into FusedTicks, replay through FeatureState ──
    let mut state = FeatureState::new();
    let mut feature_rows: Vec<(f64, [f64; N_FEATURES])> = Vec::new(); // (price, features)

    let mut acc = BucketAccumulator::default();
    let mut current_bucket: Option<i64> = None;
    let mut last_ts: Option<i64> = None;
    let mut row_count: u64 = 0;

    for result in rdr.deserialize() {
        let row: TradeRow = result.context("parsing trade row")?;
        row_count += 1;

        if let Some(prev) = last_ts {
            anyhow::ensure!(
                row.ts_micros >= prev,
                "input is not sorted ascending by ts_micros at row {row_count} \
                 ({} < {prev}) — sort the file first",
                row.ts_micros
            );
        }
        last_ts = Some(row.ts_micros);

        let exchange = row.parse_exchange()?;
        let bucket_id = row.ts_micros.div_euclid(bucket_width_micros);

        match current_bucket {
            None => current_bucket = Some(bucket_id),
            Some(cur) if bucket_id != cur => {
                // Flush every bucket strictly between `cur` and `bucket_id`
                // (inclusive of `cur`), preserving real wall-clock spacing so
                // the incremental TickVelocity / OFI / RealisedVol windows
                // see correctly-spaced timestamps even across gaps.
                for b in cur..bucket_id {
                    if let Some(fused) = acc.flush(b, bucket_width_micros) {
                        let fv = state.update_from_fused(&fused);
                        feature_rows.push((fv.price, feature_array(&fv)));
                    }
                    acc = BucketAccumulator::default();
                }
                current_bucket = Some(bucket_id);
            }
            _ => {}
        }

        acc.add(&row, exchange);
    }
    // Flush the final open bucket.
    if let Some(cur) = current_bucket {
        if let Some(fused) = acc.flush(cur, bucket_width_micros) {
            let fv = state.update_from_fused(&fused);
            feature_rows.push((fv.price, feature_array(&fv)));
        }
    }

    eprintln!(
        "Processed {row_count} trades into {} feature buckets",
        feature_rows.len()
    );

    // ── Pass 2: forward-return labelling ────────────────────────────────────
    let label_buckets = (label_window_micros / bucket_width_micros).max(1) as usize;
    anyhow::ensure!(
        feature_rows.len() > label_buckets,
        "not enough data: {} buckets but label window needs {label_buckets} \
         buckets ahead — provide more history or shorten --label-window-secs",
        feature_rows.len()
    );

    let writer = BufWriter::new(
        File::create(&args.output).with_context(|| format!("creating {:?}", args.output))?,
    );
    let mut wtr = csv::Writer::from_writer(writer);
    wtr.write_record(OUTPUT_HEADER)?;

    let mut window: VecDeque<f64> = VecDeque::with_capacity(label_buckets + 1);
    let mut counts = [0u64; 3];

    for (i, (price, feats)) in feature_rows.iter().enumerate() {
        window.push_back(*price);
        if i >= label_buckets {
            let past_price = feature_rows[i - label_buckets].0;
            let future_price = *price;
            let ret = (future_price - past_price) / past_price;

            let label: u8 = if ret > args.label_threshold {
                2 // Bullish
            } else if ret < -args.label_threshold {
                0 // Bearish
            } else {
                1 // Sideways
            };
            counts[label as usize] += 1;

            let past_feats = feature_rows[i - label_buckets].1;
            let mut record: Vec<String> =
                past_feats.iter().map(|v| format!("{v:.8}")).collect();
            record.push(label.to_string());
            wtr.write_record(&record)?;
        }
        let _ = window.pop_front();
    }
    wtr.flush()?;

    let total: u64 = counts.iter().sum();
    eprintln!("Wrote {total} labelled rows to {}", args.output.display());
    eprintln!(
        "Label distribution — Bearish: {} ({:.1}%), Sideways: {} ({:.1}%), Bullish: {} ({:.1}%)",
        counts[0], 100.0 * counts[0] as f64 / total as f64,
        counts[1], 100.0 * counts[1] as f64 / total as f64,
        counts[2], 100.0 * counts[2] as f64 / total as f64,
    );
    if counts[1] as f64 / total as f64 > 0.9 {
        eprintln!(
            "WARNING: Sideways class is >90% of the data. Consider lowering \
             --label-threshold or the model will trivially predict 'Sideways' \
             for everything."
        );
    }

    Ok(())
}
