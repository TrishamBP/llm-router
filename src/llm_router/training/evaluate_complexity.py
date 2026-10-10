"""Final, one-time evaluation of a trained complexity classifier on the held-out test split.

Usage: uv run python -m llm_router.training.evaluate_complexity [--model-dir ...]

Writes to `paths.reports_dir`: metrics.json, classification_report.txt,
confusion_matrix.png, evaluation_report.md. Uses cached test embeddings only (no API calls).
Latency figures cover local classification only; embedding latency is reported separately
from the embedding job's metadata.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import classification_report

from llm_router.config import LABELS, ComplexityConfig, load_config
from llm_router.data.datasets import load_embedded_split
from llm_router.inference.classifier import ComplexityRouter
from llm_router.logging import configure_logging, get_logger
from llm_router.training.train_complexity import classification_metrics

log = get_logger(__name__)


def _quantiles(values: np.ndarray) -> dict[str, float] | None:
    if values.size == 0:
        return None
    q = np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        **{k: float(v) for k, v in zip(("p05", "p25", "median", "p75", "p95"), q, strict=True)},
    }


def evaluate(
    cfg: ComplexityConfig, model_dir: Path, *, latency_samples: int = 500
) -> dict[str, Any]:
    router = ComplexityRouter.from_pretrained(model_dir, offline=True)
    meta = router.metadata
    emb_dir = cfg.paths.embeddings_dir / meta["embedding_cache_key"]
    test = load_embedded_split(emb_dir, "test")
    if test.metadata["embedding_model"] != router.embedding_model:
        raise RuntimeError("test embeddings come from a different embedding model")

    predictions = router.predict_embeddings(test.X)
    y_pred = [p.label for p in predictions]
    metrics = classification_metrics(test.y, y_pred)

    confidence: dict[str, Any] | None = None
    if predictions and predictions[0].probabilities is not None:
        top = np.array([max(p.probabilities.values()) for p in predictions if p.probabilities])
        correct = np.array(y_pred) == test.y
        confidence = {
            "top_probability_all": _quantiles(top),
            "top_probability_correct": _quantiles(top[correct]),
            "top_probability_incorrect": _quantiles(top[~correct]),
            "by_predicted_label": {
                label: _quantiles(top[np.array(y_pred) == label]) for label in LABELS
            },
            "note": "Logistic-regression probabilities are not guaranteed to be calibrated.",
        }

    n = min(latency_samples, len(test.X))
    timings = []
    for row in test.X[:n]:
        started = time.perf_counter()
        router.predict_embeddings(row)
        timings.append((time.perf_counter() - started) * 1000)
    started = time.perf_counter()
    router.predict_embeddings(test.X)
    batch_ms = (time.perf_counter() - started) * 1000

    emb_meta = test.metadata
    result = {
        "model_dir": str(model_dir),
        "classifier": router.classifier,
        "hyperparameters": meta["hyperparameters"],
        "embedding_model": router.embedding_model,
        "embedding_dimension": router.dimension,
        "test": metrics,
        "confidence": confidence,
        "validation_calibration": meta.get("validation_calibration"),
        "durations": {
            "training_seconds_selected_model": meta["training_seconds"],
            "embedding_generation_seconds_total": emb_meta.get("embedding_seconds"),
        },
        "local_inference_latency_ms": {
            "single_query_samples": n,
            "median": statistics.median(timings) if timings else None,
            "p95": float(np.quantile(timings, 0.95)) if len(timings) >= 20 else None,
            "batch_all_test_rows_ms": batch_ms,
            "excludes": "embedding API latency",
        },
        "embedding_usage": emb_meta.get("usage"),
        "embedding_estimated_cost_usd": emb_meta.get("estimated_cost_usd"),
    }
    write_reports(cfg.paths.reports_dir, result, test.y, y_pred)
    return result


def write_reports(out: Path, result: dict[str, Any], y_true: Any, y_pred: list[str]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    report = classification_report(y_true, y_pred, labels=list(LABELS), digits=4, zero_division=0)
    (out / "classification_report.txt").write_text(report, encoding="utf-8")

    matrix = np.array(result["test"]["confusion_matrix"]["rows_true_cols_pred"])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(3), LABELS)
    ax.set_yticks(range(3), LABELS)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(3):
        for j in range(3):
            ax.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
                color="white" if matrix[i, j] > matrix.max() / 2 else "black",
            )
    ax.set_title(f"Test confusion matrix ({result['classifier']})")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    t = result["test"]
    lat = result["local_inference_latency_ms"]
    lines = [
        f"# Complexity classifier: test evaluation ({result['classifier']})",
        "",
        f"- Embedding model: `{result['embedding_model']}` ({result['embedding_dimension']} dims)",
        f"- Hyperparameters: `{json.dumps(result['hyperparameters'])}`",
        f"- Test rows: {t['n']}",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Accuracy | {t['accuracy']:.4f} |",
        f"| Macro F1 | {t['macro_f1']:.4f} |",
        f"| Weighted F1 | {t['weighted_f1']:.4f} |",
        f"| Hard recall | {t['hard_recall']:.4f} |",
        f"| Hard predicted as easy | {t['hard_predicted_easy']['count']} "
        f"({(t['hard_predicted_easy']['rate_of_true_hard'] or 0):.2%} of true hard) |",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "|---|---|---|---|---|",
        *[
            f"| {c} | {v['precision']:.4f} | {v['recall']:.4f} | {v['f1']:.4f} | {v['support']} |"
            for c, v in t["per_class"].items()
        ],
        "",
        "![confusion matrix](confusion_matrix.png)",
        "",
        f"Local classification latency (excludes embedding API): median {lat['median']:.3f} ms, "
        f"p95 {lat['p95'] if lat['p95'] is None else round(lat['p95'], 3)} ms over "
        f"{lat['single_query_samples']} single-query calls.",
        "",
        "Probabilities from logistic regression are not guaranteed to be calibrated; see "
        "`validation_calibration` in metrics.json.",
    ]
    (out / "evaluation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/complexity.yaml")
    parser.add_argument("--model-dir", default=None)
    args = parser.parse_args(argv)
    configure_logging()
    cfg = load_config(args.config)
    model_dir = (
        Path(args.model_dir) if args.model_dir else cfg.paths.models_dir / "logistic_regression"
    )
    result = evaluate(cfg, model_dir)
    t = result["test"]
    print(
        json.dumps(
            {
                k: t[k]
                for k in (
                    "n",
                    "accuracy",
                    "macro_f1",
                    "weighted_f1",
                    "hard_recall",
                    "hard_predicted_easy",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
