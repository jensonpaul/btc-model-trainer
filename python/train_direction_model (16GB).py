"""
Phase 2 — Train a LightGBM directional classifier on per-month shards,
using walk-forward validation, then retrain on the full dataset and export
to ONNX.

## Shard-aware design

Phase 1 (collect-training-data) writes one CSV per calendar month plus a
manifest.json that describes every shard:

    data/training/
        manifest.json
        2017-01.csv
        2017-02.csv
        ...
        2024-12.csv

This script reads the manifest to discover shards and their row counts, then
streams them **one shard at a time**.  A full dataset is never loaded into
memory at once.

## Walk-forward validation (WFV)

WFV is the only statistically honest way to evaluate a time-series classifier.
Folds are aligned to **calendar months** so boundaries are interpretable and
reproducible:

    fold 1:  train [2017-01 .. 2022-06)  →  val [2022-06 .. 2022-12)
    fold 2:  train [2017-01 .. 2022-12)  →  val [2022-12 .. 2023-06)
    ...

Each fold's training window *expands* (earlier data is never discarded).
Shuffling is never applied across time — doing so would allow the model to
peek at future prices, producing metrics that collapse to baseline in
production.

## Memory model (actual)

Peak RSS per fold is bounded to approximately:

    max(largest_shard_bytes) × subsample
    + memmap file (OS-managed, paged on demand)
    + LightGBM histogram structures (~num_leaves × max_bin × n_features × 4 bytes)
    + one shard read-buffer at a time

With --subsample 0.05 and the default LightGBM params, fold 3 (502M manifest
rows) uses roughly 1.7 GB of Python heap instead of ~38 GB.

## Usage

    python train_direction_model.py \\
        --training-dir ../data/training \\
        --output       model/direction_model.onnx \\
        --scale        short \\
        --subsample    0.05

    # Explicit train / val / test split by year:
    python train_direction_model.py \\
        --training-dir ../data/training \\
        --output       model/direction_model.onnx \\
        --test-from    2024-01 \\
        --val-from     2023-01 \\
        --subsample    0.05

The --scale flag is a label for filenames/logging only; it does not change
training logic.  Run once per TimeScale with the matching --training-dir.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import classification_report, confusion_matrix, log_loss
from onnxmltools import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType

# ── Column contract ────────────────────────────────────────────────────────────
# Must match FEATURE_NAMES in btc-model-trainer/src/main.rs and
# OnnxTrendModel::feature_array in btc-onnx-trend-model.  Do NOT reorder.

FEATURES: list[str] = [
    "rsi_14", "vwap_dev", "mom_micro", "mom_short", "ewma_vol",
    "tick_vel", "ofi_30s", "ofi_300s", "autocorr", "rvol_30s",
    "xchg_spread", "price_norm", "ewma_var",
    "book_imb5", "book_imb_full", "book_wmid", "book_spread",
]
CLASS_NAMES: list[str] = ["Bearish", "Sideways", "Bullish"]
N_CLASSES = len(CLASS_NAMES)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--training-dir", required=True,
        help="Directory containing manifest.json and per-month CSV shards.",
    )
    p.add_argument("--output", required=True, help="Destination .onnx path.")
    p.add_argument(
        "--scale", default="short",
        help="TimeScale label for logging (short|medium|broad|micro).",
    )
    p.add_argument(
        "--cache-dir", default=None,
        help="Directory for temporary memmap files during dataset construction. "
             "Defaults to the system temp directory.  Use a path on a fast "
             "local disk (e.g. /tmp or an NVMe mount) for best performance.",
    )

    # Chronological split boundaries (inclusive month strings YYYY-MM).
    split = p.add_argument_group("Explicit split boundaries (optional)")
    split.add_argument(
        "--test-from", default=None, metavar="YYYY-MM",
        help="First month of the held-out test set.",
    )
    split.add_argument(
        "--val-from", default=None, metavar="YYYY-MM",
        help="First month of the validation window in the final evaluation pass. "
             "Only used when --test-from is also given.",
    )

    # Walk-forward parameters
    wfv = p.add_argument_group("Walk-forward validation")
    wfv.add_argument(
        "--n-folds", type=int, default=5,
        help="Number of WFV folds (default 5).",
    )
    wfv.add_argument(
        "--min-train-months", type=int, default=None,
        help="Minimum number of months in the first training window. "
             "Defaults to 40%% of available training months.",
    )
    wfv.add_argument(
        "--val-months", type=int, default=None,
        help="Number of months per validation window. "
             "Defaults to 10%% of available training months, minimum 1.",
    )

    # Sampling
    samp = p.add_argument_group("Sampling")
    samp.add_argument(
        "--subsample", type=float, default=0.05,
        help="Fraction of rows to keep per shard (0 < x ≤ 1.0).  "
             "Default 0.05 (5%%).  LightGBM histogram bins saturate well "
             "below 50M rows; values above 0.20 rarely improve accuracy "
             "while multiplying memory usage.",
    )
    samp.add_argument(
        "--subsample-seed", type=int, default=42,
        help="Random seed for row sampling (default 42).",
    )

    # Model hyper-parameters
    hp = p.add_argument_group("LightGBM hyper-parameters")
    hp.add_argument("--n-estimators",          type=int,   default=300)
    hp.add_argument("--learning-rate",         type=float, default=0.05)
    hp.add_argument("--num-leaves",            type=int,   default=63)
    hp.add_argument("--min-child-samples",     type=int,   default=100)
    hp.add_argument("--max-bin",               type=int,   default=63,
        help="LightGBM histogram bin count.  Lower values reduce memory "
             "and training time with minimal accuracy loss (default 63).")
    hp.add_argument("--early-stopping-rounds", type=int,   default=30)

    return p.parse_args()


# ── Manifest ──────────────────────────────────────────────────────────────────

@dataclass
class ShardInfo:
    file:            str
    year:            int
    month:           int
    rows:            int
    bearish:         int
    sideways:        int
    bullish:         int
    first_ts_micros: int | None
    last_ts_micros:  int | None

    @property
    def ym(self) -> str:
        return f"{self.year}-{self.month:02d}"


@dataclass
class Manifest:
    training_dir:      Path
    label_window_secs: int
    label_threshold:   float
    bucket_ms:         int
    features:          list[str]
    total_rows:        int
    shards:            list[ShardInfo]

    @classmethod
    def load(cls, training_dir: Path) -> "Manifest":
        path = training_dir / "manifest.json"
        if not path.exists():
            sys.exit(
                f"ERROR: manifest.json not found in {training_dir}.\n"
                f"Run collect-training-data --output-dir {training_dir} first."
            )
        raw = json.loads(path.read_text())

        manifest_features = raw.get("features", [])
        if manifest_features != FEATURES:
            missing = [f for f in FEATURES if f not in manifest_features]
            extra   = [f for f in manifest_features if f not in FEATURES]
            msgs = []
            if missing: msgs.append(f"missing from manifest: {missing}")
            if extra:   msgs.append(f"extra in manifest: {extra}")
            sys.exit(
                "ERROR: feature mismatch between manifest.json and this script.\n"
                + "\n".join(msgs)
                + "\nEnsure FEATURES list matches FEATURE_NAMES in src/main.rs."
            )

        shards = [
            ShardInfo(
                file=s["file"], year=s["year"], month=s["month"],
                rows=s["rows"],
                bearish=s["bearish"], sideways=s["sideways"], bullish=s["bullish"],
                first_ts_micros=s.get("first_ts_micros"),
                last_ts_micros=s.get("last_ts_micros"),
            )
            for s in raw["shards"]
        ]
        shards.sort(key=lambda s: (s.year, s.month))

        return cls(
            training_dir      = training_dir,
            label_window_secs = raw.get("label_window_secs", 300),
            label_threshold   = raw.get("label_threshold", 0.001),
            bucket_ms         = raw.get("bucket_ms", 100),
            features          = manifest_features,
            total_rows        = raw["total"]["rows"],
            shards            = shards,
        )

    def shard_path(self, s: ShardInfo) -> Path:
        return self.training_dir / s.file


# ── Shard streaming ────────────────────────────────────────────────────────────

def stream_shard(
    path: Path,
    subsample: float = 1.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Read one monthly CSV shard and return (X float32, y int8).

    Labels are stored as int8 (values 0–2) rather than int64 to save ~8×
    label-array memory across large folds.

    When subsample < 1.0, rows are sampled with a fixed random seed so
    results are reproducible across runs.
    """
    df = pd.read_csv(path, dtype={f: np.float32 for f in FEATURES})

    missing = [c for c in FEATURES + ["label"] if c not in df.columns]
    if missing:
        sys.exit(
            f"ERROR: shard {path.name} is missing columns: {missing}\n"
            f"Re-run collect-training-data to regenerate the shards."
        )

    before = len(df)
    df.dropna(subset=FEATURES + ["label"], inplace=True)
    if len(df) < before:
        print(f"    [warn] dropped {before - len(df)} NaN rows in {path.name}")

    if subsample < 1.0 and len(df) > 0:
        df = df.sample(frac=subsample, random_state=seed)

    X = df[FEATURES].values.astype(np.float32)
    # int8 is sufficient for labels 0–2 and saves ~8× vs int64.
    y = df["label"].values.astype(np.int8)

    del df

    bad = set(np.unique(y)) - {0, 1, 2}
    if bad:
        sys.exit(
            f"ERROR: {path.name} contains label values outside {{0,1,2}}: {bad}"
        )

    return X, y


# ── Dataset construction ───────────────────────────────────────────────────────

def build_lgb_dataset(
    shards: list[ShardInfo],
    manifest: Manifest,
    args: argparse.Namespace,
    reference: lgb.Dataset | None = None,
    label: str = "",
) -> lgb.Dataset:
    """
    Stream shards into a lgb.Dataset using a memmap-backed array so that
    peak Python heap usage is bounded to one shard at a time rather than
    the sum of all shards.

    Memory model
    ------------
    The memmap file lives on disk; the OS pages individual blocks into RAM
    on demand and evicts them under memory pressure.  Python never holds
    more than one shard's worth of data in heap at once.

    Peak RSS ≈ max(shard_bytes × subsample) + LightGBM internal structures.

    The memmap files are deleted immediately after lgb.Dataset.construct()
    so that LightGBM's own binned representation is the only copy that
    survives.

    Parameters
    ----------
    shards:    ordered list of ShardInfo objects to include.
    manifest:  Manifest providing shard paths and metadata.
    args:      parsed CLI namespace (subsample, subsample_seed).
    reference: existing lgb.Dataset whose bin boundaries should be reused
               (pass the training dataset when building the val dataset).
    label:     string used in temp-file names and log messages.
    """
    # Use manifest row counts as a capacity hint; actual rows after NaN-drop
    # and subsampling will be equal or smaller.
    total_rows_hint = max(1, int(sum(s.rows for s in shards) * args.subsample))
    n_features      = len(FEATURES)

    if total_rows_hint == 0:
        sys.exit(f"ERROR: shard list for '{label}' is empty after subsampling.")

    cache_dir = Path(args.cache_dir) if args.cache_dir else Path(tempfile.gettempdir())
    cache_dir.mkdir(parents=True, exist_ok=True)

    x_path = cache_dir / f"lgb_{label}_X.npy"
    y_path = cache_dir / f"lgb_{label}_y.npy"

    try:
        # Open memory-mapped files at the hinted capacity.  The OS will not
        # commit physical RAM for pages that are never touched.
        X_mm: np.memmap = np.lib.format.open_memmap(
            str(x_path), mode="w+",
            dtype=np.float32, shape=(total_rows_hint, n_features),
        )
        y_mm: np.memmap = np.lib.format.open_memmap(
            str(y_path), mode="w+",
            dtype=np.int8, shape=(total_rows_hint,),
        )

        cursor = 0
        for s in shards:
            X, y = stream_shard(
                manifest.shard_path(s),
                subsample=args.subsample,
                seed=args.subsample_seed,
            )
            n = len(X)
            if n == 0:
                del X, y
                continue

            if cursor + n > len(X_mm):
                # Manifest hint was stale; grow the memmap by remapping.
                new_cap = cursor + n + int(total_rows_hint * 0.05)
                X_mm = np.lib.format.open_memmap(
                    str(x_path), mode="r+",
                    dtype=np.float32, shape=(new_cap, n_features),
                )
                y_mm = np.lib.format.open_memmap(
                    str(y_path), mode="r+",
                    dtype=np.int8, shape=(new_cap,),
                )

            X_mm[cursor : cursor + n] = X
            y_mm[cursor : cursor + n] = y
            cursor += n

            # Release shard arrays immediately; only the memmap slice persists.
            del X, y
            gc.collect()

        if cursor == 0:
            sys.exit(f"ERROR: all shards for '{label}' were empty after NaN drop.")

        # Trim to actual size.
        X_view = X_mm[:cursor]
        y_view = y_mm[:cursor]

        # Flush before handing to LightGBM.
        X_mm.flush()
        y_mm.flush()

        # LightGBM reads from the memmap view; it does not copy it back into
        # the Python heap.  free_raw_data=True tells LightGBM to release its
        # reference to the raw array once binning is complete.
        dataset = lgb.Dataset(
            X_view,
            label=y_view.astype(np.int32),  # LightGBM label must be int32/float32
            feature_name=FEATURES,
            free_raw_data=True,
            reference=reference,
            params={
                "max_bin": args.max_bin,
            },
        )
        dataset.construct()

    finally:
        # Remove temp files regardless of success or failure.
        for p in (x_path, y_path):
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass

    del X_mm, y_mm
    gc.collect()
    return dataset


# ── Walk-forward folds ─────────────────────────────────────────────────────────

@dataclass
class Fold:
    index:        int
    train_shards: list[ShardInfo]
    val_shards:   list[ShardInfo]


def walk_forward_folds(
    shards: list[ShardInfo],
    n_folds: int,
    min_train_months: int,
    val_months: int,
) -> list[Fold]:
    """
    Generate expanding-window walk-forward folds aligned to calendar months.

    Timeline view (n_folds=5, min_train=24 months, val_window=6 months):

        fold 1:  [=====train 24m=====|val 6m|                        ]
        fold 2:  [=====train 30m======|val 6m|                       ]
        fold 3:  [=====train 36m=======|val 6m|                      ]
        fold 4:  [=====train 42m========|val 6m|                     ]
        fold 5:  [=====train 48m=========|val 6m|                    ]
    """
    n = len(shards)
    folds: list[Fold] = []

    for k in range(n_folds):
        train_end = min_train_months + k * val_months
        val_start = train_end
        val_end   = val_start + val_months

        if val_end > n:
            if k == 0:
                raise ValueError(
                    f"Not enough months for even one fold.  "
                    f"Available: {n}, need: {train_end + val_months}.  "
                    f"Reduce --min-train-months or --val-months."
                )
            warnings.warn(
                f"Only {k} fold(s) fit in the available {n} months "
                f"(need {train_end + val_months}).  Stopping early.",
                stacklevel=2,
            )
            break

        folds.append(Fold(
            index        = k + 1,
            train_shards = shards[:train_end],
            val_shards   = shards[val_start:val_end],
        ))

    return folds


# ── Single-fold trainer ────────────────────────────────────────────────────────

@dataclass
class FoldResult:
    fold_index:      int
    train_months:    int
    val_months:      int
    train_rows:      int
    val_rows:        int
    val_period:      str
    accuracy:        float
    directional_acc: float | None
    logloss:         float
    best_iteration:  int
    label_report:    str
    confusion:       np.ndarray


def _evaluate_on_shards(
    booster: lgb.Booster,
    shards: list[ShardInfo],
    manifest: Manifest,
    args: argparse.Namespace,
    best_iter: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Stream val shards one at a time, predict each independently, accumulate
    results.  Avoids materialising the full val set in memory simultaneously.

    Probabilities are kept as float32 throughout — sklearn metrics accept
    float32 and this halves the accumulation cost vs the original float64 cast.
    """
    y_true_parts:  list[np.ndarray] = []
    y_proba_parts: list[np.ndarray] = []

    for s in shards:
        X, y = stream_shard(
            manifest.shard_path(s),
            subsample=args.subsample,
            seed=args.subsample_seed,
        )
        if len(X) == 0:
            del X, y
            continue

        proba = booster.predict(X, num_iteration=best_iter).astype(np.float32)
        y_true_parts.append(y)
        y_proba_parts.append(proba)

        del X, y, proba
        gc.collect()

    if not y_true_parts:
        sys.exit("ERROR: all val shards were empty after subsampling.")

    y_true  = np.concatenate(y_true_parts).astype(np.int64)
    y_proba = np.concatenate(y_proba_parts)          # float32, shape (n, 3)
    return y_true, y_proba


def train_fold(
    fold: Fold,
    manifest: Manifest,
    args: argparse.Namespace,
) -> FoldResult:
    print(
        f"    Building train dataset ({len(fold.train_shards)} shards, "
        f"~{int(sum(s.rows for s in fold.train_shards) * args.subsample):,} "
        f"sampled rows) …",
        flush=True,
    )
    train_ds = build_lgb_dataset(
        fold.train_shards, manifest, args, label=f"fold{fold.index}_train"
    )

    print(
        f"    Building val dataset ({len(fold.val_shards)} shards, "
        f"~{int(sum(s.rows for s in fold.val_shards) * args.subsample):,} "
        f"sampled rows) …",
        flush=True,
    )
    # Pass train_ds as reference so val uses the same bin boundaries.
    val_ds = build_lgb_dataset(
        fold.val_shards, manifest, args,
        reference=train_ds, label=f"fold{fold.index}_val",
    )

    params = _lgb_params(args)
    callbacks = [
        lgb.early_stopping(args.early_stopping_rounds, verbose=False),
        lgb.log_evaluation(period=-1),
    ]
    booster = lgb.train(
        params,
        train_ds,
        num_boost_round = args.n_estimators,
        valid_sets      = [val_ds],
        valid_names     = ["val"],
        callbacks       = callbacks,
    )
    best_iter = booster.best_iteration or args.n_estimators

    # Free datasets before evaluation — this is the largest memory release.
    del train_ds, val_ds
    gc.collect()

    # ── Shard-by-shard evaluation ─────────────────────────────────────────────
    # We re-stream the val shards rather than materialising the full val set,
    # eliminating the ~12 GB spike from the original np.concatenate approach.
    y_val, y_proba = _evaluate_on_shards(
        booster, fold.val_shards, manifest, args, best_iter
    )

    y_pred  = y_proba.argmax(axis=1).astype(np.int64)
    acc     = float((y_pred == y_val).mean())
    # float32 is accepted by sklearn; no float64 cast needed.
    loss    = float(log_loss(y_val, y_proba, labels=[0, 1, 2]))
    report  = classification_report(
        y_val, y_pred, target_names=CLASS_NAMES, digits=3, zero_division=0,
    )
    cm      = confusion_matrix(y_val, y_pred, labels=[0, 1, 2])

    dir_mask = y_val != 1
    dir_acc: float | None = None
    if dir_mask.sum() > 0:
        dir_acc = float((y_pred[dir_mask] == y_val[dir_mask]).mean())

    val_period = (
        f"{fold.val_shards[0].ym} → {fold.val_shards[-1].ym}"
        if fold.val_shards else "—"
    )
    train_rows = int(sum(s.rows for s in fold.train_shards) * args.subsample)
    val_rows   = int(y_val.shape[0])

    del y_val, y_proba, booster
    gc.collect()

    return FoldResult(
        fold_index      = fold.index,
        train_months    = len(fold.train_shards),
        val_months      = len(fold.val_shards),
        train_rows      = train_rows,
        val_rows        = val_rows,
        val_period      = val_period,
        accuracy        = acc,
        directional_acc = dir_acc,
        logloss         = loss,
        best_iteration  = best_iter,
        label_report    = report,
        confusion       = cm,
    )


# ── WFV summary ───────────────────────────────────────────────────────────────

def print_wfv_summary(results: list[FoldResult]) -> int:
    """Print per-fold and aggregate metrics.  Returns median best_iteration."""
    print("\n" + "=" * 78)
    print("WALK-FORWARD VALIDATION SUMMARY")
    print("=" * 78)
    print(
        f"{'Fold':>4}  {'Val period':>18}  {'Train rows':>10}  {'Val rows':>8}  "
        f"{'Accuracy':>8}  {'Dir.Acc':>7}  {'LogLoss':>7}  {'BestIter':>8}"
    )
    print("-" * 78)

    accs, dir_accs, losses, iters = [], [], [], []
    for r in results:
        da = f"{r.directional_acc:.3f}" if r.directional_acc is not None else "    N/A"
        print(
            f"{r.fold_index:>4}  {r.val_period:>18}  {r.train_rows:>10,}  "
            f"{r.val_rows:>8,}  {r.accuracy:>8.3f}  {da:>7}  "
            f"{r.logloss:>7.4f}  {r.best_iteration:>8}"
        )
        accs.append(r.accuracy)
        if r.directional_acc is not None:
            dir_accs.append(r.directional_acc)
        losses.append(r.logloss)
        iters.append(r.best_iteration)

    print("-" * 78)
    da_str = (
        f"{np.mean(dir_accs):.3f} ± {np.std(dir_accs):.3f}"
        if dir_accs else "N/A"
    )
    print(
        f"{'Mean':>4}  {'':>18}  {'':>10}  {'':>8}  "
        f"{np.mean(accs):>8.3f}  {da_str:>7}  {np.mean(losses):>7.4f}"
    )
    print(
        f"{'Std':>4}  {'':>18}  {'':>10}  {'':>8}  "
        f"{np.std(accs):>8.3f}  {'':>7}  {np.std(losses):>7.4f}"
    )
    print("=" * 78)

    if len(results) >= 4:
        early_acc = np.mean([r.accuracy for r in results[:2]])
        late_acc  = np.mean([r.accuracy for r in results[-2:]])
        if early_acc - late_acc > 0.05:
            print(
                f"\nWARNING: accuracy dropped {early_acc - late_acc:.3f} from "
                f"early folds ({early_acc:.3f}) to late folds ({late_acc:.3f}).\n"
                f"This suggests distribution shift — recent data differs from "
                f"older data.  Consider:\n"
                f"  • Reducing --min-train-months to weight recent shards more.\n"
                f"  • Retraining more frequently.\n"
                f"  • Adding regime features (e.g. realised-vol z-score)."
            )

    print("\nPer-fold classification reports:\n")
    for r in results:
        print(
            f"── Fold {r.fold_index}  val={r.val_period}  "
            f"(train≈{r.train_rows:,} rows, val={r.val_rows:,} rows) ──"
        )
        print(r.label_report)
        cm_df = pd.DataFrame(r.confusion, index=CLASS_NAMES, columns=CLASS_NAMES)
        print("Confusion matrix (rows=true, cols=pred):")
        print(cm_df.to_string())
        print()

    median_iter = int(np.median(iters))
    print(
        f"Median best_iteration across folds: {median_iter}  "
        f"(used for final retraining on full dataset)"
    )
    return median_iter


# ── Final model ───────────────────────────────────────────────────────────────

def train_final(
    train_shards: list[ShardInfo],
    manifest: Manifest,
    args: argparse.Namespace,
    n_estimators_final: int,
) -> lgb.Booster:
    total_rows_est = int(sum(s.rows for s in train_shards) * args.subsample)
    print(
        f"\nRetraining on full training set "
        f"({len(train_shards)} shards, ~{total_rows_est:,} sampled rows, "
        f"n_estimators={n_estimators_final}) …"
    )
    train_ds = build_lgb_dataset(
        train_shards, manifest, args, label="final-train"
    )
    params  = _lgb_params(args)
    booster = lgb.train(
        params,
        train_ds,
        num_boost_round = n_estimators_final,
        callbacks       = [lgb.log_evaluation(50)],
    )
    del train_ds
    gc.collect()
    return booster


def _lgb_params(args: argparse.Namespace) -> dict:
    return {
        "objective":         "multiclass",
        "num_class":         N_CLASSES,
        "learning_rate":     args.learning_rate,
        "num_leaves":        args.num_leaves,
        "min_child_samples": args.min_child_samples,
        # "max_bin":           args.max_bin,
        "class_weight":      "balanced",
        "random_state":      42,
        "n_jobs":            -1,
        "verbose":           -1,
    }


# ── Feature importance ────────────────────────────────────────────────────────

def print_feature_importance(booster: lgb.Booster) -> None:
    imp = pd.Series(
        booster.feature_importance(importance_type="gain"),
        index=FEATURES,
    ).sort_values(ascending=False)
    print("\n=== Feature importance (gain, final model) ===")
    print(imp.to_string())
    low = imp[imp < imp.mean() * 0.1]
    if not low.empty:
        print(
            f"\nNote: {len(low)} feature(s) have very low importance "
            f"({', '.join(low.index.tolist())}).  "
            f"Consider removing them to reduce overfitting."
        )


# ── ONNX export ───────────────────────────────────────────────────────────────

def export_onnx(booster: lgb.Booster, output_path: str) -> None:
    initial_types = [
        ("float_input", FloatTensorType([None, len(FEATURES)]))
    ]
    onnx_model = convert_lightgbm(
        booster,
        initial_types=initial_types,
        zipmap=False,
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"Exported ONNX model → {output_path}")


def verify_onnx(
    booster: lgb.Booster,
    output_path: str,
    sample_shards: list[ShardInfo],
    manifest: Manifest,
    args: argparse.Namespace,
) -> None:
    try:
        import onnxruntime as rt
    except ImportError:
        print("onnxruntime not available — skipping ONNX verification.")
        return

    if not sample_shards:
        print("No shards available for ONNX verification — skipping.")
        return

    X_sample, _ = stream_shard(
        manifest.shard_path(sample_shards[0]),
        subsample=min(args.subsample, 0.1),
        seed=args.subsample_seed,
    )
    X_sample = X_sample[:min(500, len(X_sample))]

    sess        = rt.InferenceSession(output_path)
    onnx_probs  = sess.run(None, {"float_input": X_sample})[1]
    onnx_labels = onnx_probs.argmax(axis=1)
    lgbm_labels = booster.predict(X_sample).argmax(axis=1)

    mismatch = int((onnx_labels != lgbm_labels).sum())
    if mismatch:
        print(
            f"WARNING: {mismatch}/{len(X_sample)} ONNX predictions differ "
            f"from LightGBM on the sample — investigate before deploying."
        )
    else:
        print(
            f"ONNX export verified: all {len(X_sample)} sample predictions "
            f"match LightGBM."
        )
    print(f"ONNX output shape: probabilities={onnx_probs.shape}")
    del X_sample
    gc.collect()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if not (0.0 < args.subsample <= 1.0):
        sys.exit("ERROR: --subsample must be in (0, 1].")

    training_dir = Path(args.training_dir)

    # ── Load manifest ─────────────────────────────────────────────────────────
    print(f"Loading manifest from {training_dir}  (scale={args.scale})")
    manifest = Manifest.load(training_dir)
    n_shards  = len(manifest.shards)

    print(
        f"Manifest: {n_shards} monthly shards  |  "
        f"{manifest.shards[0].ym} → {manifest.shards[-1].ym}  |  "
        f"{manifest.total_rows:,} total rows"
    )
    t = manifest.total_rows
    b = sum(s.bearish  for s in manifest.shards)
    s = sum(s.sideways for s in manifest.shards)
    u = sum(s.bullish  for s in manifest.shards)
    print(
        f"Label distribution: "
        f"Bearish={b:,} ({100*b/t:.1f}%)  "
        f"Sideways={s:,} ({100*s/t:.1f}%)  "
        f"Bullish={u:,} ({100*u/t:.1f}%)"
    )
    sampled_est = int(manifest.total_rows * args.subsample)
    print(
        f"Subsample rate: {args.subsample:.0%}  →  "
        f"~{sampled_est:,} rows effective across all shards"
    )

    if s / t > 0.9:
        print(
            "WARNING: Sideways >90% — consider lowering --label-threshold "
            "in Phase 1 or the model will trivially predict Sideways."
        )
    if n_shards < 6:
        print(
            f"WARNING: only {n_shards} monthly shards available.  "
            f"Results may be unreliable."
        )

    # ── Partition shards into train / test sets ───────────────────────────────
    if args.test_from:
        test_shards  = [sh for sh in manifest.shards if sh.ym >= args.test_from]
        train_shards = [sh for sh in manifest.shards if sh.ym <  args.test_from]
        if not test_shards:
            sys.exit(
                f"ERROR: --test-from {args.test_from} is after the last shard "
                f"({manifest.shards[-1].ym}).  No test shards."
            )
        if not train_shards:
            sys.exit(
                f"ERROR: --test-from {args.test_from} is before the first shard "
                f"({manifest.shards[0].ym}).  No training shards."
            )
        print(
            f"\nSplit: train {manifest.shards[0].ym}→{train_shards[-1].ym}  "
            f"| test {test_shards[0].ym}→{test_shards[-1].ym}"
        )
    else:
        train_shards = manifest.shards
        test_shards  = []
        print("\nNo --test-from supplied — using all shards for training and WFV.")

    # ── Walk-forward parameters ───────────────────────────────────────────────
    n_train          = len(train_shards)
    val_months       = args.val_months       or max(1, int(n_train * 0.10))
    min_train_months = args.min_train_months or max(1, int(n_train * 0.40))

    print(
        f"\nWalk-forward config: {args.n_folds} folds  |  "
        f"min_train={min_train_months} months  |  val_window={val_months} months"
    )

    # ── Build folds ───────────────────────────────────────────────────────────
    folds = walk_forward_folds(
        shards           = train_shards,
        n_folds          = args.n_folds,
        min_train_months = min_train_months,
        val_months       = val_months,
    )
    print(f"  → {len(folds)} fold(s) constructed")
    for f in folds:
        print(
            f"     fold {f.index}: train {f.train_shards[0].ym}→"
            f"{f.train_shards[-1].ym} ({len(f.train_shards)}m)  |  "
            f"val {f.val_shards[0].ym}→{f.val_shards[-1].ym} "
            f"({len(f.val_shards)}m)"
        )

    # ── Walk-forward training loop ────────────────────────────────────────────
    print(f"\nTraining {len(folds)} fold(s) …\n")
    results: list[FoldResult] = []
    for fold in folds:
        print(
            f"  Fold {fold.index}/{len(folds)}  "
            f"val={fold.val_shards[0].ym}→{fold.val_shards[-1].ym} …",
            flush=True,
        )
        result = train_fold(fold, manifest, args)
        results.append(result)
        da = f"{result.directional_acc:.3f}" if result.directional_acc is not None else "N/A"
        print(
            f"  → acc={result.accuracy:.3f}  dir_acc={da}  "
            f"logloss={result.logloss:.4f}  best_iter={result.best_iteration}"
        )
        gc.collect()

    # ── Summary & median iteration count ─────────────────────────────────────
    median_iter = print_wfv_summary(results)

    # ── Final retraining on all training shards ───────────────────────────────
    booster_final = train_final(
        train_shards, manifest, args, n_estimators_final=median_iter,
    )

    # ── Feature importance ────────────────────────────────────────────────────
    print_feature_importance(booster_final)

    # ── Optional: evaluate on held-out test set ───────────────────────────────
    if test_shards:
        print(
            f"\nEvaluating on test set "
            f"({test_shards[0].ym}→{test_shards[-1].ym}, "
            f"~{int(sum(s.rows for s in test_shards) * args.subsample):,} "
            f"sampled rows) …"
        )
        y_test, y_proba_test = _evaluate_on_shards(
            booster_final, test_shards, manifest, args,
            best_iter=median_iter,
        )
        y_pred_test  = y_proba_test.argmax(axis=1).astype(np.int64)
        test_acc     = float((y_pred_test == y_test).mean())
        test_loss    = float(log_loss(y_test, y_proba_test, labels=[0, 1, 2]))
        print(f"Test accuracy: {test_acc:.4f}  log-loss: {test_loss:.4f}")
        print(classification_report(
            y_test, y_pred_test, target_names=CLASS_NAMES,
            digits=3, zero_division=0,
        ))
        cm = confusion_matrix(y_test, y_pred_test, labels=[0, 1, 2])
        print("Confusion matrix (rows=true, cols=pred):")
        print(pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_string())
        del y_test, y_proba_test
        gc.collect()

    # ── Export & verify ───────────────────────────────────────────────────────
    print(f"\nExporting to {args.output} …")
    export_onnx(booster_final, args.output)
    verify_onnx(booster_final, args.output, manifest.shards[:1], manifest, args)

    print(f"\nDone.  Model ready for deployment (scale={args.scale}).")


if __name__ == "__main__":
    main()
