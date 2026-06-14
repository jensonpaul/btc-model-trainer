# train_direction_model.py

Phase 2 of the BTC directional classifier pipeline. Trains a LightGBM
multiclass model on per-month CSV shards produced by Phase 1
(`collect-training-data`), evaluates it with walk-forward validation, retrains
on the full dataset, and exports to ONNX for inference in Rust.

---

## Table of contents

1. [Quick start](#quick-start)
2. [Prerequisites](#prerequisites)
3. [Input format](#input-format)
4. [Memory architecture](#memory-architecture)
5. [Walk-forward validation](#walk-forward-validation)
6. [CLI reference](#cli-reference)
7. [Choosing subsample rate](#choosing-subsample-rate)
8. [Interpreting output](#interpreting-output)
9. [Accuracy baseline and signal validation](#accuracy-baseline-and-signal-validation)
10. [Tuning guide](#tuning-guide)
11. [ONNX export and verification](#onnx-export-and-verification)
12. [Troubleshooting](#troubleshooting)
13. [Design decisions](#design-decisions)

---

## Quick start

```bash
# Minimal invocation — 5% subsample, sensible defaults
python train_direction_model.py \
    --training-dir ../data/training \
    --output       model/direction_model.onnx \
    --scale        short \
    --subsample    0.05

# With explicit held-out test set
python train_direction_model.py \
    --training-dir ../data/training \
    --output       model/direction_model.onnx \
    --test-from    2024-01 \
    --subsample    0.05
```

Expected runtime on 114 monthly shards at `--subsample 0.05`: roughly
15–30 minutes depending on CPU core count and disk speed.

---

## Prerequisites

```
Python  >= 3.10
lightgbm >= 4.0
scikit-learn
pandas
numpy
onnxmltools
onnxruntime          # optional, for post-export verification
```

Install:

```bash
pip install lightgbm scikit-learn pandas numpy onnxmltools onnxruntime
```

---

## Input format

The script expects the directory layout written by `collect-training-data`:

```
data/training/
    manifest.json
    2017-01.csv
    2017-02.csv
    ...
    2026-06.csv
```

### manifest.json

```json
{
  "label_window_secs": 300,
  "label_threshold":   0.001,
  "bucket_ms":         100,
  "features":          ["rsi_14", "vwap_dev", ...],
  "total": { "rows": 971363104 },
  "shards": [
    {
      "file":    "2017-01.csv",
      "year":    2017,
      "month":   1,
      "rows":    1234567,
      "bearish": 380000,
      "sideways": 470000,
      "bullish": 384567,
      "first_ts_micros": 1483228800000000,
      "last_ts_micros":  1485907199000000
    },
    ...
  ]
}
```

The manifest's `features` array must exactly match the `FEATURES` constant at
the top of this script (and `FEATURE_NAMES` in `btc-model-trainer/src/main.rs`).
The script exits with a clear error message if they diverge.

### Per-month CSV shards

Each CSV has one row per 100 ms bucket and must contain:

| Column         | Type    | Description                        |
|----------------|---------|------------------------------------|
| `rsi_14`       | float32 | RSI(14)                            |
| `vwap_dev`     | float32 | Deviation from VWAP                |
| `mom_micro`    | float32 | Micro-scale momentum               |
| `mom_short`    | float32 | Short-scale momentum               |
| `ewma_vol`     | float32 | EWMA volatility                    |
| `tick_vel`     | float32 | Tick velocity                      |
| `ofi_30s`      | float32 | Order flow imbalance, 30s          |
| `ofi_300s`     | float32 | Order flow imbalance, 300s         |
| `autocorr`     | float32 | Return autocorrelation             |
| `rvol_30s`     | float32 | Realised vol, 30s                  |
| `xchg_spread`  | float32 | Cross-exchange spread              |
| `price_norm`   | float32 | Normalised price                   |
| `ewma_var`     | float32 | EWMA variance                      |
| `book_imb5`    | float32 | Book imbalance, top 5 levels       |
| `book_imb_full`| float32 | Book imbalance, full depth         |
| `book_wmid`    | float32 | Weighted mid-price                 |
| `book_spread`  | float32 | Book spread                        |
| `label`        | int8    | 0 = Bearish, 1 = Sideways, 2 = Bullish |

Rows with `NaN` in any feature or the label are dropped at load time with a
warning.

---

## Memory architecture

The original script pre-allocated the entire training window as a contiguous
NumPy array before passing it to LightGBM, producing ~38 GB of heap
allocation for fold 3 alone (502M rows × 17 features × 4 bytes). Combined
with LightGBM's internal histogram structures (~15 GB) this triggered an OOM
kill at ~62 GB RSS.

This version uses four complementary techniques:

### 1. Row subsampling (`--subsample`)

Rows are sampled with a fixed seed inside `stream_shard` before being written
to the memmap. At `--subsample 0.05` the effective training set for fold 3
drops from 502M rows to ~25M rows, reducing the memmap file from ~34 GB to
~1.7 GB.

LightGBM histogram bins saturate at ~5–20M rows for most feature
distributions. Training on 500M rows provides negligible additional
information once the bins are full.

### 2. Memmap-backed dataset construction

Instead of allocating a Python heap array for the entire fold, shards are
written one at a time into a `numpy.memmap` file on disk. The OS pages
blocks in and out of RAM as needed; the Python heap never holds more than
one shard at a time.

The memmap file is deleted immediately after `lgb.Dataset.construct()`
completes, so LightGBM's own binned representation is the only survivor.

Peak RSS formula:

```
max(shard_bytes × subsample)            # one shard in heap at a time
+ memmap working set (OS-managed)       # cold pages evicted automatically
+ LightGBM histogram structures         # num_leaves × max_bin × n_features × 4
```

With `--subsample 0.05`, `--num-leaves 63`, `--max-bin 63`:
LightGBM histogram ≈ 63 × 63 × 17 × 4 ≈ 270 KB — negligible.

### 3. Reduced histogram bins (`--max-bin 63`)

Default is 255. Reducing to 63 cuts LightGBM's histogram memory by ~4× with
minimal impact on split quality — the model still finds essentially the same
decision boundaries at 63 bins as at 255 for financial time-series features.

### 4. Shard-by-shard validation evaluation

The original code concatenated all val shards into a single array before
predicting, adding ~12 GB for fold 3's 172M-row val set. The rewrite streams
shards individually, predicts each one, accumulates probabilities as float32
(not float64), and concatenates only the compact result arrays.

---

## Walk-forward validation

WFV is the only statistically honest evaluation methodology for time-series
classifiers. Cross-validation with shuffling would allow the model to train
on future data and validate on past data, inflating metrics to levels that
do not exist in production.

### Fold structure

Each fold uses an **expanding training window**:

```
fold 1:  [====train 45m====|val 11m|                              ]
fold 2:  [====train 56m=====|val 11m|                             ]
fold 3:  [====train 67m======|val 11m|                            ]
fold 4:  [====train 78m=======|val 11m|                           ]
fold 5:  [====train 89m========|val 11m|                          ]
```

Earlier training data is never discarded — this replicates the information
available to a model that is retrained periodically in production.

### Fold parameters

| Parameter             | Default | Meaning                                   |
|-----------------------|---------|-------------------------------------------|
| `--n-folds`           | 5       | Number of WFV folds                       |
| `--min-train-months`  | 40% of available | Months in the first training window |
| `--val-months`        | 10% of available | Months per validation window         |

For 114 shards with no `--test-from`:
- `min_train_months` = 45
- `val_months` = 11

### Distribution shift detection

If accuracy drops by more than 0.05 between the first two and last two folds,
the script prints a warning. This indicates the market regime has changed and
the model trained on older data is less effective on recent data.

Remedies: reduce `--min-train-months` (gives recent data more weight),
retrain more frequently, or add regime-detection features.

---

## CLI reference

### Required

| Flag              | Description                                         |
|-------------------|-----------------------------------------------------|
| `--training-dir`  | Directory containing `manifest.json` and CSV shards |
| `--output`        | Destination `.onnx` file path                       |

### Optional — general

| Flag          | Default | Description                                         |
|---------------|---------|-----------------------------------------------------|
| `--scale`     | `short` | Label for logging only (short/medium/broad/micro)   |
| `--cache-dir` | system temp | Directory for memmap temp files. Use a fast local disk. |

### Optional — split boundaries

| Flag           | Example      | Description                              |
|----------------|--------------|------------------------------------------|
| `--test-from`  | `2024-01`    | First month of the held-out test set     |
| `--val-from`   | `2023-01`    | First month of the final val window (requires `--test-from`) |

### Optional — walk-forward

| Flag                  | Default        | Description                        |
|-----------------------|----------------|------------------------------------|
| `--n-folds`           | `5`            | Number of WFV folds                |
| `--min-train-months`  | 40% of shards  | Months in first training window    |
| `--val-months`        | 10% of shards  | Months per validation window       |

### Optional — sampling

| Flag               | Default | Description                                        |
|--------------------|---------|----------------------------------------------------|
| `--subsample`      | `0.05`  | Fraction of rows to keep per shard (0 < x ≤ 1.0)  |
| `--subsample-seed` | `42`    | Random seed for reproducible sampling              |

### Optional — LightGBM hyper-parameters

| Flag                      | Default | Description                              |
|---------------------------|---------|------------------------------------------|
| `--n-estimators`          | `300`   | Maximum number of boosting rounds        |
| `--learning-rate`         | `0.05`  | Step size shrinkage                      |
| `--num-leaves`            | `63`    | Maximum leaves per tree                  |
| `--min-child-samples`     | `100`   | Minimum samples required to split a leaf |
| `--max-bin`               | `63`    | Histogram bin count                      |
| `--early-stopping-rounds` | `30`    | Stop if val loss doesn't improve for N rounds |

---

## Choosing subsample rate

| Rate   | Effective rows (fold 3) | Peak RSS estimate | Recommended when              |
|--------|-------------------------|-------------------|-------------------------------|
| `1.00` | 502M                    | ~60+ GB           | Never — OOMs on most hardware |
| `0.20` | 100M                    | ~8 GB             | High-RAM server (32+ GB)      |
| `0.10` | 50M                     | ~4 GB             | 16 GB RAM, validating signal  |
| `0.05` | 25M                     | ~2 GB             | **Default. Recommended.**     |
| `0.02` | 10M                     | ~800 MB           | Quick iteration / debugging   |
| `0.01` | 5M                      | ~400 MB           | Rapid prototyping only        |

LightGBM histogram bins (default `max_bin=63`) saturate at roughly 5–20M
rows for most feature distributions. Going beyond 50M rows adds training
time without meaningfully changing split boundaries or model quality.

The subsample is applied with a fixed seed per shard, so results are
reproducible across runs. The same seed is used for all shards in all folds,
ensuring that the sampled subset is consistent when comparing experiments.

---

## Interpreting output

### Per-fold metrics

```
Fold  Val period          Train rows    Val rows  Accuracy  Dir.Acc  LogLoss  BestIter
   1  2020-10 → 2021-08   10,092,069   8,327,511     0.401    0.512   1.0412        47
   2  2021-09 → 2022-07   18,419,580   6,718,975     0.389    0.401   1.0587        38
```

| Metric       | Meaning                                                          |
|--------------|------------------------------------------------------------------|
| `Accuracy`   | Overall fraction of correct predictions across all 3 classes     |
| `Dir.Acc`    | Accuracy on non-Sideways rows only (Bearish or Bullish)          |
| `LogLoss`    | Cross-entropy loss; lower is better; random = ln(3) ≈ 1.0986    |
| `BestIter`   | Round at which early stopping fired; used for final retraining   |

### Baselines to beat

| Metric    | Trivial baseline             | Notes                               |
|-----------|------------------------------|-------------------------------------|
| Accuracy  | Sideways class fraction      | ~38.5% in the example dataset       |
| Dir.Acc   | 50%                          | Random binary classifier            |
| LogLoss   | ln(3) ≈ 1.0986               | Uniform probability across 3 classes |

A model that beats accuracy baseline but has `Dir.Acc < 0.50` is predicting
directional classes but getting the direction wrong more often than not.

### Distribution shift warning

```
WARNING: accuracy dropped 0.072 from early folds (0.401) to late folds (0.329).
```

This means the model trained on 2017–2021 data generalises poorly to 2023–2025
data. See the Tuning guide for remedies.

---

## Accuracy baseline and signal validation

With 971M rows across 114 months, the training data is large, but size does not
imply learnability. Before investing compute in full training runs, validate
that the features contain predictive signal:

### Step 1 — Quick signal check at low subsample

```bash
python train_direction_model.py \
    --training-dir ../data/training \
    --output /tmp/test.onnx \
    --subsample 0.01 \
    --n-folds 2 \
    --n-estimators 50
```

If accuracy is at or below the Sideways baseline (≈ 38.5%) across both folds,
the features do not contain learnable signal at this label definition and
timescale. More data will not fix this.

### Step 2 — Check directional accuracy specifically

Overall accuracy can be misleading when `class_weight="balanced"` suppresses
Sideways predictions. Focus on `Dir.Acc`:

- `Dir.Acc > 0.55` — meaningful directional signal present
- `Dir.Acc ~ 0.50` — noise; model is guessing direction
- `Dir.Acc < 0.50` — model is systematically predicting the wrong direction

### Step 3 — If signal is absent

Consider:

- Increasing `--label-threshold` in Phase 1 to filter out ambiguous labels
- Increasing `--label-window-secs` to label on a longer horizon
- Adding regime features (e.g. rolling volatility z-score) to help the model
  identify which market state it is in
- Removing `class_weight="balanced"` and using a confidence threshold at
  inference time instead of predicting all three classes equally

---

## Tuning guide

### Memory vs accuracy trade-off

| Change                          | Memory impact    | Accuracy impact         |
|---------------------------------|------------------|-------------------------|
| `--subsample 0.05` (default)    | −95% heap usage  | Negligible (bins saturate) |
| `--max-bin 63` (default)        | −75% histogram   | Negligible for this task  |
| `--num-leaves 31`               | −50% histogram   | Small reduction           |
| `--min-child-samples 100`       | Regularisation   | Reduces overfit on noise  |

### Improving accuracy

| Symptom                         | Likely cause               | Remedy                                 |
|---------------------------------|----------------------------|----------------------------------------|
| Accuracy ≤ Sideways baseline    | No learnable signal        | Review label generation in Phase 1     |
| Dir.Acc < 0.50                  | Directional noise          | Increase label threshold/window        |
| Strong early folds, weak late   | Distribution shift         | Reduce `--min-train-months`            |
| All predictions are Sideways    | Class imbalance dominates  | Remove `class_weight=balanced`, tune threshold |
| Very low `BestIter` (< 20)      | Underfitting / weak signal | Lower `--learning-rate`, add features  |
| `BestIter` = `--n-estimators`   | Training didn't converge   | Increase `--n-estimators`, lower `--learning-rate` |

### Recommended starting experiment

```bash
python train_direction_model.py \
    --training-dir ../data/training \
    --output       model/direction_model.onnx \
    --test-from    2024-01 \
    --subsample    0.05 \
    --n-folds      5 \
    --n-estimators 300 \
    --learning-rate 0.05 \
    --num-leaves   63 \
    --min-child-samples 100 \
    --max-bin      63 \
    --early-stopping-rounds 30
```

---

## ONNX export and verification

The final model is exported via `onnxmltools.convert_lightgbm` with
`zipmap=False`, producing a model that outputs:

- Output 0: predicted class labels (int64), shape `[N]`
- Output 1: class probabilities (float32), shape `[N, 3]`

Column order: `[Bearish, Sideways, Bullish]` (matches `CLASS_NAMES`).

The Rust inference side (`OnnxTrendModel`) expects input named `float_input`
with shape `[N, 17]` in the same feature order as `FEATURES`.

### Verification

If `onnxruntime` is installed, the script automatically loads the exported
model and runs 500 sample rows through both LightGBM and ONNX Runtime,
comparing predictions. A mismatch count of 0 is expected; any mismatch
indicates a conversion issue and the model should not be deployed.

### Manual verification

```python
import onnxruntime as rt
import numpy as np

sess  = rt.InferenceSession("model/direction_model.onnx")
X     = np.random.rand(10, 17).astype(np.float32)
label, probs = sess.run(None, {"float_input": X})
print(label.shape, probs.shape)  # (10,), (10, 3)
```

---

## Troubleshooting

### OOM / Killed

The script has been designed to stay well under 8 GB RSS at `--subsample 0.05`.
If you still OOM:

1. Verify `--subsample` is being passed (default is 0.05 in this version).
2. Check that `--cache-dir` points to a disk with sufficient free space
   (`total_sampled_rows × 17 × 5 bytes` per fold — roughly 2 GB at default settings).
3. Reduce `--num-leaves` to 31 or lower.
4. Reduce `--subsample` to 0.02.

### `manifest.json` not found

Run Phase 1 first:

```bash
cd btc-model-trainer
cargo run --release -- collect-training-data \
    --output-dir ../data/training \
    --scale short
```

### Feature mismatch error

```
ERROR: feature mismatch between manifest.json and this script.
missing from manifest: ['book_wmid']
```

The manifest was generated by an older version of `collect-training-data`
that did not emit all features. Re-run Phase 1 to regenerate the shards.

### `BestIter` is always 1–5

Early stopping fires almost immediately, meaning the model is not learning.
Likely causes:

- Features are constant or near-constant (check for all-zero shards)
- Label is all the same class in the training window
- `--learning-rate` is too high (try 0.01)

### ONNX verification mismatch

If the post-export verification reports mismatched predictions:

- Check `onnxmltools` version matches what was used to train
- Try re-exporting with a fresh virtual environment
- File an issue with the LightGBM booster `.txt` dump and `onnxmltools` version

---

## Design decisions

### Why memmap instead of LightGBM binary cache?

LightGBM's `.bin` format is not portable across LightGBM versions and
requires the reference dataset (train) to be present when loading the val
binary. This makes fold-level caching complex. Memmap files are plain NumPy
format, trivially deleted, and the OS manages their memory footprint
automatically.

### Why not push_rows / streaming Dataset?

LightGBM's `push_rows` API requires pre-declaring the dataset size and has
limited support for labels in older versions. The memmap approach gives
identical results with simpler code and better OS-level memory management.

### Why int8 labels instead of int64?

Labels only take values 0, 1, 2. int8 is sufficient and reduces the label
array from 8 bytes/row to 1 byte/row — a saving of ~4 GB per fold at
full-data scale, and proportionally less after subsampling. The int8 arrays
are cast to int32 immediately before passing to LightGBM (which requires
int32 or float32 labels).

### Why float32 probabilities in evaluation?

sklearn's `log_loss` and `classification_report` accept float32. The original
code cast probabilities to float64 before computing metrics, doubling the
memory of the accumulated val probability array. float32 is sufficient for
the precision of these metrics.

### Why `min_child_samples=100`?

With subsampling at 5%, the smallest folds contain ~5–10M rows. A leaf
containing only 30 rows (the original default) represents `30 / 5_000_000
= 0.0006%` of the data — effectively a noise fit. Raising to 100 provides
meaningful regularisation without sacrificing split granularity.
