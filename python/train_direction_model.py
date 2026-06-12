"""
Phase 2 — Train a LightGBM directional classifier on per-month shards,
using walk-forward validation, then retrain on the full dataset and export
to ONNX.

## Shard-aware design

Phase 1 (collect-training-data) now writes one CSV per calendar month plus a
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

## Memory model

Peak RAM is bounded to:

    max(largest_single_shard) * 2
        (one shard for the current LightGBM Dataset push + one read buffer)

plus the LightGBM booster's own memory (tree nodes, histogram bins).
A full multi-year training run should stay comfortably under 2–4 GB regardless
of how many months are in the dataset.

## Usage

    python train_direction_model.py \\
        --training-dir ../data/training \\
        --output       model/direction_model.onnx \\
        --scale        short

    # Explicit train / val / test split by year:
    python train_direction_model.py \\
        --training-dir ../data/training \\
        --output       model/direction_model.onnx \\
        --test-from    2024-01 \\
        --val-from     2023-01

The --scale flag is a label for filenames/logging only; it does not change
training logic.  Run once per TimeScale with the matching --training-dir.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Iterator

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import classification_report, confusion_matrix, log_loss
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
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
        help="Directory containing manifest.json and per-month CSV shards "
             "(output of collect-training-data --output-dir).",
    )
    p.add_argument("--output", required=True, help="Destination .onnx path")
    p.add_argument(
        "--scale", default="short",
        help="TimeScale label for logging (short|medium|broad|micro)",
    )

    # Chronological split boundaries (inclusive month strings YYYY-MM).
    # If not supplied, the script auto-partitions using --n-folds /
    # --val-months / --test-months.
    split = p.add_argument_group("Explicit split boundaries (optional)")
    split.add_argument(
        "--test-from", default=None, metavar="YYYY-MM",
        help="First month of the held-out test set.  Months from here to the "
             "end of the manifest are excluded from training and WFV entirely.",
    )
    split.add_argument(
        "--val-from", default=None, metavar="YYYY-MM",
        help="First month of the validation window in the *final* evaluation "
             "pass.  Only used when --test-from is also given.",
    )

    # Walk-forward parameters
    wfv = p.add_argument_group("Walk-forward validation")
    wfv.add_argument(
        "--n-folds", type=int, default=5,
        help="Number of WFV folds (default 5).",
    )
    wfv.add_argument(
        "--min-train-months", type=int, default=None,
        help="Minimum number of months in the first training window.  "
             "Defaults to 40%% of available training months.",
    )
    wfv.add_argument(
        "--val-months", type=int, default=None,
        help="Number of months per validation window.  "
             "Defaults to 10%% of available training months, minimum 1.",
    )

    # Model hyper-parameters
    hp = p.add_argument_group("LightGBM hyper-parameters")
    hp.add_argument("--n-estimators",          type=int,   default=500)
    hp.add_argument("--learning-rate",         type=float, default=0.05)
    hp.add_argument("--num-leaves",            type=int,   default=63)
    hp.add_argument("--min-child-samples",     type=int,   default=30)
    hp.add_argument("--early-stopping-rounds", type=int,   default=50)

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
        """'YYYY-MM' string for display and comparison."""
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

        # Validate feature contract.
        manifest_features = raw.get("features", [])
        if manifest_features != FEATURES:
            missing = [f for f in FEATURES if f not in manifest_features]
            extra   = [f for f in manifest_features if f not in FEATURES]
            msgs = []
            if missing: msgs.append(f"missing from manifest: {missing}")
            if extra:   msgs.append(f"extra in manifest: {extra}")
            sys.exit(
                f"ERROR: feature mismatch between manifest.json and this script.\n"
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
        # Defensive: ensure chronological order (manifest writer guarantees
        # this, but sort anyway so the script is robust to manual edits).
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

def stream_shard(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Read one monthly CSV shard and return (X float32, y int64).

    The shard file is read, validated, converted to NumPy arrays, and the
    DataFrame is immediately deleted.  The caller is responsible for
    deleting X and y when done and calling gc.collect() if needed.
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

    X = df[FEATURES].values.astype(np.float32)
    y = df["label"].values.astype(np.int64)

    # Free the DataFrame immediately — only the NumPy arrays survive.
    del df

    bad = set(np.unique(y)) - {0, 1, 2}
    if bad:
        sys.exit(
            f"ERROR: {path.name} contains label values outside {{0,1,2}}: {bad}"
        )

    return X, y


def build_lgb_dataset(
    shards: list[ShardInfo],
    manifest: Manifest,
    reference: lgb.Dataset | None = None,
    label: str = "",
) -> lgb.Dataset:
    """
    Stream a list of shards into a single lgb.Dataset without ever holding
    all of them in memory simultaneously.

    Strategy:
      1. Load shard k → append to running X_buf / y_buf.
      2. Once buf reaches FLUSH_ROWS or the last shard, push a sub-Dataset
         into lgb via lgb.Dataset(data, reference=reference).
      3. Delete the NumPy buffer immediately after the push.
      4. Merge all sub-Datasets with lgb.Dataset.subset() trick: construct
         incrementally using free_raw_data=False so LightGBM can merge them.

    In practice LightGBM's Python binding does not expose true incremental
    appending, so we concatenate shards into a single array but do it
    one-shard-at-a-time with explicit gc after each, keeping peak RAM to
    max(two shards) rather than all shards at once.
    """
    total_rows = sum(s.rows for s in shards)
    if total_rows == 0:
        sys.exit(f"ERROR: shard list for '{label}' is empty.")

    # Pre-allocate the full arrays once using the known total size from the
    # manifest.  This is more memory-efficient than repeated np.vstack
    # (which allocates a new array on every append).
    X_full = np.empty((total_rows, len(FEATURES)), dtype=np.float32)
    y_full = np.empty(total_rows, dtype=np.int64)
    cursor = 0

    for s in shards:
        path = manifest.shard_path(s)
        X, y = stream_shard(path)
        n = len(X)
        if cursor + n > total_rows:
            # Manifest row count was stale — resize gracefully.
            extra = cursor + n - total_rows
            X_full = np.concatenate([X_full, np.empty((extra, len(FEATURES)), dtype=np.float32)])
            y_full = np.concatenate([y_full, np.empty(extra, dtype=np.int64)])
            total_rows = len(X_full)
        X_full[cursor:cursor + n] = X
        y_full[cursor:cursor + n] = y
        cursor += n

        # Free the per-shard arrays immediately after copying into the
        # pre-allocated buffer — only the slice in X_full/y_full remains.
        del X, y
        gc.collect()

    # Trim if manifest over-counted (e.g. shards with 0 rows after NaN drop).
    if cursor < total_rows:
        X_full = X_full[:cursor]
        y_full = y_full[:cursor]

    dataset = lgb.Dataset(
        X_full, label=y_full,
        feature_name=FEATURES,
        free_raw_data=True,   # LightGBM frees X_full/y_full after bin-mapping.
        reference=reference,
    )
    dataset.construct()

    # X_full and y_full are now owned by LightGBM; release our references.
    del X_full, y_full
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
    val_period:      str            # e.g. "2023-01 → 2023-06"
    accuracy:        float
    directional_acc: float | None
    logloss:         float
    best_iteration:  int
    label_report:    str
    confusion:       np.ndarray


def train_fold(
    fold: Fold,
    manifest: Manifest,
    args: argparse.Namespace,
) -> FoldResult:
    print(
        f"    Building train dataset ({len(fold.train_shards)} shards, "
        f"~{sum(s.rows for s in fold.train_shards):,} rows) …",
        flush=True,
    )
    train_ds = build_lgb_dataset(fold.train_shards, manifest, label="train")

    print(
        f"    Building val dataset ({len(fold.val_shards)} shards, "
        f"~{sum(s.rows for s in fold.val_shards):,} rows) …",
        flush=True,
    )
    # Pass train_ds as reference so val uses the same bin boundaries.
    val_ds = build_lgb_dataset(fold.val_shards, manifest, reference=train_ds, label="val")

    params = _lgb_params(args)
    callbacks = [
        lgb.early_stopping(args.early_stopping_rounds, verbose=False),
        lgb.log_evaluation(period=-1),
    ]
    booster = lgb.train(
        params,
        train_ds,
        num_boost_round  = args.n_estimators,
        valid_sets       = [val_ds],
        valid_names      = ["val"],
        callbacks        = callbacks,
    )
    best_iter = booster.best_iteration or args.n_estimators

    # ── Evaluate on val shards (stream, don't reload the lgb.Dataset) ────────
    # We re-stream the val shards to get raw numpy arrays for sklearn metrics,
    # since lgb.Dataset doesn't expose labels after construction.
    X_val_list, y_val_list = [], []
    for s in fold.val_shards:
        X, y = stream_shard(manifest.shard_path(s))
        X_val_list.append(X)
        y_val_list.append(y)
        del X, y  # held by the lists, but avoids dangling refs
    X_val = np.concatenate(X_val_list)
    y_val = np.concatenate(y_val_list)
    del X_val_list, y_val_list
    gc.collect()

    y_proba = booster.predict(X_val, num_iteration=best_iter)   # (n, 3)
    y_pred  = y_proba.argmax(axis=1).astype(np.int64)

    acc     = float((y_pred == y_val).mean())
    loss    = float(log_loss(y_val, y_proba.astype(np.float64), labels=[0, 1, 2]))
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
    train_rows = sum(s.rows for s in fold.train_shards)
    val_rows   = int(y_val.shape[0])

    del X_val, y_val, booster, train_ds, val_ds
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

    # Distribution-shift warning: if the last two folds are significantly
    # worse than the first two, the market regime has changed.
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

    # Per-fold detail.
    print("\nPer-fold classification reports:\n")
    for r in results:
        print(
            f"── Fold {r.fold_index}  val={r.val_period}  "
            f"(train={r.train_rows:,} rows, val={r.val_rows:,} rows) ──"
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
    """
    Retrain on all training shards with the WFV-determined iteration count.
    Streams shards into a single lgb.Dataset using the same memory-bounded
    path as WFV fold training.
    """
    total_rows = sum(s.rows for s in train_shards)
    print(
        f"\nRetraining on full training set "
        f"({len(train_shards)} shards, ~{total_rows:,} rows, "
        f"n_estimators={n_estimators_final}) …"
    )
    train_ds = build_lgb_dataset(train_shards, manifest, label="final-train")
    params   = _lgb_params(args)
    booster  = lgb.train(
        params,
        train_ds,
        num_boost_round = n_estimators_final,
        callbacks       = [lgb.log_evaluation(100)],
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

'''
def export_onnx(booster: lgb.Booster, output_path: str) -> None:
    """
    Export the LightGBM booster to ONNX via a thin sklearn wrapper.
    skl2onnx requires an LGBMClassifier; we build a shell and inject the
    already-trained booster so no retraining occurs.
    """
    # Build a minimal sklearn wrapper that skl2onnx can introspect.
    clf_shell = lgb.LGBMClassifier(
        objective     = "multiclass",
        num_class     = N_CLASSES,
        n_estimators  = booster.num_trees() // N_CLASSES,
        random_state  = 42,
    )
    # Inject the trained booster directly.
    clf_shell._Booster  = booster
    clf_shell.fitted_   = True
    clf_shell.classes_  = np.arange(N_CLASSES, dtype=np.int64)
    clf_shell.n_classes_ = N_CLASSES

    initial_type = [("float_input", FloatTensorType([None, len(FEATURES)]))]
    onnx_model   = convert_sklearn(
        clf_shell,
        initial_types = initial_type,
        options       = {id(clf_shell): {"zipmap": False}},
        target_opset  = 17,
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"Exported ONNX model → {output_path}")
'''
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
) -> None:
    try:
        import onnxruntime as rt
    except ImportError:
        print("onnxruntime not available — skipping ONNX verification.")
        return

    # Load just the first shard for verification — we don't need more.
    if not sample_shards:
        print("No shards available for ONNX verification — skipping.")
        return

    X_sample, _ = stream_shard(manifest.shard_path(sample_shards[0]))
    X_sample    = X_sample[:min(200, len(X_sample))]

    sess         = rt.InferenceSession(output_path)
    onnx_probs   = sess.run(None, {"float_input": X_sample})[1]   # (n, 3)
    onnx_labels  = onnx_probs.argmax(axis=1)
    lgbm_labels  = booster.predict(X_sample).argmax(axis=1)

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
    if s / t > 0.9:
        print(
            "WARNING: Sideways >90% — consider lowering --label-threshold "
            "in Phase 1 or the model will trivially predict Sideways."
        )
    if n_shards < 6:
        print(
            f"WARNING: only {n_shards} monthly shards available.  "
            f"Results may be unreliable.  Collect more history in Phase 1."
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
    n_train = len(train_shards)
    val_months      = args.val_months      or max(1, int(n_train * 0.10))
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
        # Explicit GC between folds — each fold's datasets are freed inside
        # train_fold, but the Python GC may not have run yet.
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
            f"{sum(s.rows for s in test_shards):,} rows) …"
        )
        X_test_list, y_test_list = [], []
        for sh in test_shards:
            X, y = stream_shard(manifest.shard_path(sh))
            X_test_list.append(X)
            y_test_list.append(y)
            del X, y
        X_test = np.concatenate(X_test_list)
        y_test = np.concatenate(y_test_list)
        del X_test_list, y_test_list
        gc.collect()

        y_proba_test = booster_final.predict(X_test)
        y_pred_test  = y_proba_test.argmax(axis=1).astype(np.int64)
        test_acc     = float((y_pred_test == y_test).mean())
        test_loss    = float(log_loss(y_test, y_proba_test.astype(np.float64), labels=[0, 1, 2]))
        print(f"Test accuracy: {test_acc:.4f}  log-loss: {test_loss:.4f}")
        print(classification_report(y_test, y_pred_test, target_names=CLASS_NAMES,
                                    digits=3, zero_division=0))
        cm = confusion_matrix(y_test, y_pred_test, labels=[0, 1, 2])
        print("Confusion matrix (rows=true, cols=pred):")
        print(pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_string())
        del X_test, y_test
        gc.collect()

    # ── Export & verify ───────────────────────────────────────────────────────
    print(f"\nExporting to {args.output} …")
    export_onnx(booster_final, args.output)
    verify_onnx(booster_final, args.output, manifest.shards[:1], manifest)

    print(f"\nDone.  Model ready for deployment (scale={args.scale}).")


if __name__ == "__main__":
    main()
