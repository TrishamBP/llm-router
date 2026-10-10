"""Shared pieces of the XGBoost / MLP complexity trainers.

Trainers load only the train and validation splits; the test split is read exclusively by
`evaluate_classifiers` after every configuration choice has been made.
"""

from __future__ import annotations

import json
import os
import platform
import random
import shutil
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from llm_router.config import LABELS, ComplexityConfig
from llm_router.data.datasets import assert_disjoint_ids
from llm_router.embeddings.generate import resolve_embeddings_dir
from llm_router.features.complexity_features import (
    FeatureSet,
    SplitFeatures,
    decode_labels,
    load_split_features,
)
from llm_router.training.train_complexity import calibration_metrics, classification_metrics

TRAINING_SPLITS = ("train", "validation")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def stratified_subsample(y: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Deterministic, label-stratified row indices (smoke runs only)."""
    if n >= len(y):
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    picked = []
    for label in range(len(LABELS)):
        rows = np.flatnonzero(y == label)
        take = max(1, round(n * len(rows) / len(y))) if len(rows) else 0
        picked.append(rng.choice(rows, size=min(take, len(rows)), replace=False))
    return np.sort(np.concatenate(picked))


def prepare_training_data(
    cfg: ComplexityConfig,
    feature_set: FeatureSet,
    *,
    embeddings_dir: Path | None = None,
    smoke: bool = False,
    smoke_rows: int = 2000,
) -> tuple[SplitFeatures, SplitFeatures, dict[str, Any], Path]:
    emb_dir = resolve_embeddings_dir(cfg, embeddings_dir)
    processed = cfg.paths.processed_dir
    train, meta = load_split_features(processed, emb_dir, "train", feature_set)
    val, _ = load_split_features(processed, emb_dir, "validation", feature_set)
    assert_disjoint_ids({"train": train.example_ids, "validation": val.example_ids})
    if smoke:
        seed = cfg.resources.seed
        train = train.subset(stratified_subsample(train.y, smoke_rows, seed))
        val = val.subset(stratified_subsample(val.y, max(smoke_rows // 4, 30), seed))
    return train, val, meta, emb_dir


def class_distribution(y: np.ndarray) -> dict[str, Any]:
    counts = np.bincount(y, minlength=len(LABELS))
    total = int(counts.sum())
    nonzero = counts[counts > 0]
    return {
        "counts": {label: int(c) for label, c in zip(LABELS, counts, strict=True)},
        "fractions": {
            label: round(float(c) / total, 4) if total else None
            for label, c in zip(LABELS, counts, strict=True)
        },
        "max_min_ratio": round(float(nonzero.max() / nonzero.min()), 3) if nonzero.size else None,
    }


def class_weight_plan(
    y_train: np.ndarray, configured: Sequence[str], threshold: float
) -> tuple[list[str], dict[str, Any]]:
    """Which class-weight settings to try. `balanced` only if the imbalance justifies it."""
    dist = class_distribution(y_train)
    ratio = dist["max_min_ratio"] or 1.0
    options = [w for w in configured if w == "none" or ratio >= threshold]
    if not options:
        options = ["none"]
    return options, {
        "configured": list(configured),
        "evaluated": options,
        "train_max_min_ratio": ratio,
        "threshold": threshold,
        "rule": "balanced is evaluated only when max/min training class count >= threshold; "
        "weights are computed from training counts only",
    }


def balanced_class_weights(y_train: np.ndarray) -> np.ndarray:
    """n / (k * count_c), as in scikit-learn's class_weight='balanced'. float32, LABELS order."""
    counts = np.bincount(y_train, minlength=len(LABELS)).astype(np.float64)
    weights = len(y_train) / (len(LABELS) * np.maximum(counts, 1))
    return weights.astype(np.float32)


def evaluate_proba(y_idx: np.ndarray, proba: np.ndarray) -> dict[str, Any]:
    if proba.shape != (len(y_idx), len(LABELS)):
        raise ValueError(f"probabilities have shape {proba.shape}")
    pred = proba.argmax(axis=1)
    metrics = classification_metrics(decode_labels(y_idx), decode_labels(pred))
    metrics["calibration"] = calibration_metrics(y_idx, proba)
    metrics["log_loss"] = metrics["calibration"]["log_loss"]
    return metrics


def selection_key(metrics: dict[str, Any]) -> tuple[float, float]:
    """Validation macro-F1, ties broken by hard recall (same rule as the baseline)."""
    return (metrics["macro_f1"], metrics["hard_recall"])


def brief(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = ("accuracy", "macro_f1", "weighted_f1", "hard_recall", "hard_predicted_easy", "log_loss")
    return {k: metrics[k] for k in keys if k in metrics}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def environment_info(extra: dict[str, str] | None = None) -> dict[str, Any]:
    import sklearn
    import xgboost

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "device": "cpu",
        "cpu_count": os.cpu_count(),
        **(extra or {}),
    }


def provenance(emb_meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "embedding_model": emb_meta["embedding_model"],
        "embedding_dimension": int(emb_meta["dimension"]),
        "embedding_cache_key": emb_meta["cache_key"],
        "processed_fingerprint": emb_meta["processed_fingerprint"],
        "preprocessing_version": emb_meta["preprocessing_version"],
    }


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def replace_directory(staging: Path, target: Path) -> None:
    """Swap a fully written staging directory into place (old artifacts removed only then)."""
    if target.exists():
        shutil.rmtree(target)
    os.replace(staging, target)


def staging_dir(target: Path) -> Path:
    staging = target.with_name(target.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    return staging
