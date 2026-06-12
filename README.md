# btc-model-trainer

Standalone, offline data pipeline for Phase 1 ("Collect") of training a
replacement for `HeuristicDirectionClassifier` behind `TrendModelExt`.

This crate is **completely external** to `btc-prediction-engine` — it depends
on it as a normal library dependency (no fork, no patch) and reuses the exact
same `FeatureState` / `FeatureVector` / `FusedTick` types the live engine
uses. This guarantees the offline training features are bit-for-bit identical
to what the model will see in production.

## Why a separate crate?

- Zero risk to the live trading system — nothing in `btc-prediction-engine`
  or `polymarket-trading-terminal` is touched.
- Runs once (or periodically, offline) against historical data — no need to
  run live feeds for weeks to accumulate a dataset.
- Easy to re-run with different `--label-window-secs` / `--label-threshold`
  values to experiment with labelling strategies without re-collecting data.

## 1. Get historical trade data

Download historical trade/tick data and convert it to this CSV schema:

```text
ts_micros,price,quantity,side,exchange
1700000000000000,43250.12,0.014,buy,binance
1700000000010000,43250.50,0.002,sell,binance
```

- **Binance**: <https://data.binance.vision/> — free monthly `aggTrades`
  dumps, no API key needed. `aggTrades` columns map directly:
  `transact_time → ts_micros` (×1000 if ms), `price`, `quantity`,
  `is_buyer_maker` (invert for `side`).
- **Kraken**: historical trade export via Kraken support.
- For multi-exchange spread features, merge multiple exchanges' files and
  sort the combined result by `ts_micros` — the tool handles interleaved
  exchanges natively.

**The input file must be sorted ascending by `ts_micros`.** The tool errors
out if it isn't (rather than silently producing garbage features).

## 2. Run the collector

```bash
cargo run --release -- \
    --input data/binance_2025_q1_trades.csv \
    --output data/training_2025_q1.csv \
    --bucket-ms 100 \
    --label-window-secs 300 \
    --label-threshold 0.001
```

- `--bucket-ms` must match the live engine's `FusionConfig::bucket_width_micros`
  (default 100 ms) — this controls how raw trades are aggregated into the
  `FusedTick`s that feed `FeatureState`.
- `--label-window-secs` is the forward-looking horizon for the label
  (e.g. 300 = "where is price 5 minutes from now"). Match this to the
  `TimeScale` you're training for (`Short` ≈ 300s).
- `--label-threshold` is the return magnitude that separates Bullish/Bearish
  from Sideways (e.g. `0.001` = 0.1%).

The tool prints the label distribution at the end. If "Sideways" dominates
(>90%), lower `--label-threshold` — otherwise the model will trivially learn
to always predict Sideways.

## 3. Output

`training_2025_q1.csv` has 18 columns: the 17 normalised features (identical
to `OnnxTrendModel::feature_array` in the implementation guide) plus `label`
(0=Bearish, 1=Sideways, 2=Bullish). This feeds directly into
`train_direction_model.py` from Phase 2 — no further transformation needed.

## 4. Phase 2 — Train and export to ONNX

```bash
cd python
pip install -r requirements.txt
python train_direction_model.py \
    --input ../data/training_2025_q1.csv \
    --output model/direction_model.onnx \
    --scale short
```

This:

- Splits the data **chronologically** (last 20% as validation, no shuffling —
  shuffling time-series data causes lookahead leakage and inflated accuracy).
- Trains a `LGBMClassifier` with class balancing (Sideways usually dominates).
- Prints a classification report, confusion matrix, feature importances, and
  compares against a majority-class baseline so you can tell if the model is
  actually learning anything.
- Reports **directional accuracy** (Bullish vs. Bearish only, ignoring
  Sideways) — often the metric that matters most for trading decisions.
- Exports to ONNX (`opset 17`, `zipmap=False` so output is a raw probability
  tensor) and verifies the exported model's predictions match sklearn's on a
  sample before declaring success.

Run this once per `TimeScale` if training per-scale models — pass the
matching `training_*.csv` (produced with the corresponding
`--label-window-secs` in Phase 1) and a different `--output` / `--scale` each
time.

`model/direction_model.onnx` is the artifact Phase 3 (`OnnxTrendModel::load`)
consumes.

## Multiple time scales

Run the collector multiple times with different `--label-window-secs` (e.g.
30, 300, 3600, 86400) against the same input file to produce one training set
per `TimeScale`, if you're training per-scale models as recommended in
Phase 3.

## Important caveats

- **`book_imbalance_*` features will be `None`/0** unless you also have
  historical order-book snapshots and feed them through
  `FeatureState::update_book` (not currently wired into this tool — add a
  second input file + interleaving pass if your historical data includes
  L2 book snapshots).
- **NTP correction is skipped** — historical timestamps from a single vendor
  are assumed already aligned. If combining feeds from multiple raw sources
  with clock skew, pre-correct timestamps before feeding this tool.
- Keep `feature_array()` in `src/main.rs` in sync with
  `OnnxTrendModel::feature_array()` in the trading terminal. A mismatch here
  is a silent train/serve skew bug — the model will be trained on different
  numbers than it sees at inference time.
