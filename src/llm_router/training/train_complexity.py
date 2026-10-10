"""Train the complexity classifier on cached embeddings (CPU only).

Usage:
    uv run python -m llm_router.training.train_complexity [--models logreg svm]

Fits every grid point on the training split only, selects by validation macro-F1 (ties broken
by validation hard-class recall), and saves the selected model per classifier family to
`paths.models_dir/<name>/`. The test split is never loaded here.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC

from llm_router.config import LABELS, ComplexityConfig, TrainingConfig, load_config
from llm_router.data.datasets import EmbeddedSplit, assert_disjoint, load_embedded_split
from llm_router.embeddings.generate import resolve_embeddings_dir
from llm_router.logging import configure_logging, get_logger

log = get_logger(__name__)

MODEL_DIRS = {"logreg": "logistic_regression", "svm": "linear_svm"}


def classification_metrics(
    y_true: Sequence[str] | np.ndarray, y_pred: Sequence[str] | np.ndarray
) -> dict[str, Any]:
    labels = list(LABELS)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    hard, easy = labels.index("hard"), labels.index("easy")
    hard_total = int(matrix[hard].sum())
    return {
        "n": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
        "per_class": {
            label: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(labels)
        },
        "confusion_matrix": {"labels": labels, "rows_true_cols_pred": matrix.tolist()},
        "hard_recall": float(recall[hard]),
        "hard_predicted_easy": {
            "count": int(matrix[hard, easy]),
            "rate_of_true_hard": float(matrix[hard, easy] / hard_total) if hard_total else None,
        },
    }


def calibration_metrics(
    y_true_idx: np.ndarray, proba: np.ndarray, bins: int = 10
) -> dict[str, float]:
    """Log loss, multi-class Brier score, and expected calibration error of the top class."""
    onehot = np.eye(proba.shape[1])[y_true_idx]
    confidence = proba.max(axis=1)
    correct = proba.argmax(axis=1) == y_true_idx
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (confidence > lo) & (confidence <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return {
        "log_loss": float(log_loss(y_true_idx, proba, labels=list(range(proba.shape[1])))),
        "brier": float(np.mean(np.sum((proba - onehot) ** 2, axis=1))),
        "ece_top_class": float(ece),
    }


def grid(kind: str, tcfg: TrainingConfig) -> list[dict[str, Any]]:
    cs = tcfg.logreg_c if kind == "logreg" else tcfg.svm_c
    return [{"C": c, "class_weight": w} for c in cs for w in tcfg.class_weights]


def make_estimator(kind: str, params: dict[str, Any], tcfg: TrainingConfig) -> Pipeline:
    class_weight = None if params["class_weight"] == "none" else params["class_weight"]
    if kind == "logreg":
        clf: Any = LogisticRegression(
            C=params["C"], class_weight=class_weight, max_iter=tcfg.max_iter, random_state=tcfg.seed
        )
    elif kind == "svm":
        clf = LinearSVC(
            C=params["C"], class_weight=class_weight, max_iter=tcfg.max_iter, random_state=tcfg.seed
        )
    else:
        raise ValueError(f"unknown classifier '{kind}'")
    return Pipeline([("scale", StandardScaler()), ("clf", clf)])


def train(
    cfg: ComplexityConfig,
    *,
    embeddings_dir: Path | None = None,
    kinds: Sequence[str] = ("logreg",),
    output_suffix: str = "",
) -> dict[str, Any]:
    emb_dir = resolve_embeddings_dir(cfg, embeddings_dir)
    train_split = load_embedded_split(emb_dir, "train")
    val_split = load_embedded_split(emb_dir, "validation")
    assert_disjoint(train_split, val_split)
    encoder = LabelEncoder().fit(list(LABELS))
    y_train = encoder.transform(train_split.y)
    y_val_idx = encoder.transform(val_split.y)

    summary: dict[str, Any] = {"embeddings_dir": str(emb_dir), "classifiers": {}}
    for kind in kinds:
        results = []
        best: tuple[tuple[float, float], Pipeline, dict[str, Any], dict[str, Any], float] | None = (
            None
        )
        for params in grid(kind, cfg.training):
            estimator = make_estimator(kind, params, cfg.training)
            started = time.perf_counter()
            estimator.fit(train_split.X, y_train)
            fit_s = time.perf_counter() - started
            pred = encoder.inverse_transform(estimator.predict(val_split.X))
            metrics = classification_metrics(val_split.y, pred)
            results.append(
                {"params": params, "fit_seconds": round(fit_s, 2), "validation": _brief(metrics)}
            )
            log.info(
                "fitted",
                classifier=kind,
                **params,
                val_macro_f1=round(metrics["macro_f1"], 4),
                seconds=round(fit_s, 1),
            )
            score = (metrics["macro_f1"], metrics["hard_recall"])
            if best is None or score > best[0]:
                best = (score, estimator, params, metrics, fit_s)
        assert best is not None
        _, estimator, params, val_metrics, fit_s = best
        calibration = (
            calibration_metrics(y_val_idx, estimator.predict_proba(val_split.X))
            if hasattr(estimator, "predict_proba")
            else None
        )
        out_dir = cfg.paths.models_dir / (MODEL_DIRS[kind] + output_suffix)
        metadata = _save(
            out_dir,
            estimator,
            encoder,
            kind,
            params,
            val_metrics,
            calibration,
            fit_s,
            results,
            train_split,
            cfg,
        )
        summary["classifiers"][kind] = {
            "model_dir": str(out_dir),
            **{
                k: metadata[k]
                for k in ("hyperparameters", "validation_metrics", "validation_calibration")
            },
        }
    reports = cfg.paths.reports_dir
    reports.mkdir(parents=True, exist_ok=True)
    (reports / f"training_summary{output_suffix}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def _brief(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        k: metrics[k]
        for k in ("accuracy", "macro_f1", "weighted_f1", "hard_recall", "hard_predicted_easy")
    }


def _save(
    out_dir: Path,
    estimator: Pipeline,
    encoder: LabelEncoder,
    kind: str,
    params: dict[str, Any],
    val_metrics: dict[str, Any],
    calibration: dict[str, float] | None,
    fit_s: float,
    results: list[dict[str, Any]],
    train_split: EmbeddedSplit,
    cfg: ComplexityConfig,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(estimator, out_dir / "model.joblib")
    joblib.dump(encoder, out_dir / "label_encoder.joblib")
    emb = train_split.metadata
    metadata = {
        "classifier": MODEL_DIRS[kind],
        "classes": encoder.classes_.tolist(),
        "supports_probabilities": hasattr(estimator, "predict_proba"),
        "hyperparameters": params,
        "selection": "max validation macro-F1, tie-break validation hard recall",
        "validation_metrics": val_metrics,
        "validation_calibration": calibration,
        "probability_note": "Probabilities are not guaranteed to be calibrated.",
        "grid_results": results,
        "train_rows": int(train_split.X.shape[0]),
        "training_seconds": round(fit_s, 2),
        "embedding_model": emb["embedding_model"],
        "embedding_dimension": emb["dimension"],
        "embedding_cache_key": emb["cache_key"],
        "processed_fingerprint": emb["processed_fingerprint"],
        "preprocessing_version": emb["preprocessing_version"],
        "seed": cfg.training.seed,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "environment": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "device": "cpu",
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/complexity.yaml")
    parser.add_argument("--embeddings-dir", default=None)
    parser.add_argument("--models", nargs="+", default=["logreg"], choices=sorted(MODEL_DIRS))
    args = parser.parse_args(argv)
    configure_logging()
    embeddings_dir = Path(args.embeddings_dir) if args.embeddings_dir else None
    summary = train(load_config(args.config), embeddings_dir=embeddings_dir, kinds=args.models)
    for kind, info in summary["classifiers"].items():
        print(
            kind,
            json.dumps(info["hyperparameters"]),
            json.dumps(_brief(info["validation_metrics"])),
        )


if __name__ == "__main__":
    main()
