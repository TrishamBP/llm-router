"""Held-out evaluation of trained complexity classifiers (any type) on cached features.

Usage:
    uv run python -m llm_router.training.evaluate_classifiers \
        --model-dirs artifacts/models/xgboost artifacts/models/mlp [--split test]

Run this only after every configuration choice has been made on validation data. It uses the
same split files and the same `ComplexityRouter` feature path as inference, writes one JSON
per model as soon as it is evaluated (`<out-dir>/<model-dir-name>.json`), and never calls an
API. Latency covers local feature construction + classification only, not embedding calls.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np

from llm_router.config import LABELS, ComplexityConfig, load_config
from llm_router.data.datasets import load_embedded_split
from llm_router.features.complexity_features import encode_labels, load_aligned_queries
from llm_router.inference.classifier import ComplexityRouter
from llm_router.logging import configure_logging, get_logger
from llm_router.training import complexity_common as common
from llm_router.training.resources import (
    directory_size_bytes,
    limit_threads,
    peak_rss_bytes,
    to_gib,
)
from llm_router.training.train_complexity import calibration_metrics, classification_metrics

log = get_logger(__name__)


def _metrics(
    classes: list[str], y_true: np.ndarray, indices: np.ndarray, proba: np.ndarray | None
) -> dict[str, Any]:
    y_pred = np.asarray(classes)[indices]
    metrics = classification_metrics(y_true, y_pred)
    if proba is not None:
        ordered = proba[:, [classes.index(label) for label in LABELS]]
        metrics["calibration"] = calibration_metrics(encode_labels(y_true), ordered)
        metrics["log_loss"] = metrics["calibration"]["log_loss"]
    else:
        metrics["calibration"] = None
        metrics["log_loss"] = None
    return metrics


def evaluate_model_dir(
    cfg: ComplexityConfig,
    model_dir: Path,
    *,
    split: str = "test",
    latency_samples: int = 500,
    batch_rows: int = 4096,
    exclude_ids: set[str] | None = None,
) -> dict[str, Any]:
    router = ComplexityRouter.from_pretrained(
        model_dir, offline=True, threads=cfg.resources.threads
    )
    meta = router.metadata
    emb_dir = cfg.paths.embeddings_dir / meta["embedding_cache_key"]
    data = load_embedded_split(emb_dir, split, mmap=True)
    if data.metadata["embedding_model"] != router.embedding_model:
        raise RuntimeError(f"{split} embeddings come from a different embedding model")
    texts: list[str] | None = None
    if router.needs_texts:
        ids, _, texts = load_aligned_queries(cfg.paths.processed_dir, emb_dir, split)
        if tuple(ids) != data.example_ids:
            raise RuntimeError("processed texts and embedding rows are not aligned")

    n = data.X.shape[0]
    indices_parts, proba_parts = [], []
    started = time.perf_counter()
    for start in range(0, n, batch_rows):
        stop = min(start + batch_rows, n)
        idx, proba = router.predict_arrays(data.X[start:stop], texts[start:stop] if texts else None)
        indices_parts.append(idx)
        if proba is not None:
            proba_parts.append(proba)
    batch_seconds = time.perf_counter() - started
    indices = np.concatenate(indices_parts).astype(np.int64)
    proba_all = np.concatenate(proba_parts) if proba_parts else None
    classes = router.classes

    timings = []
    for row in range(min(latency_samples, n)):
        t0 = time.perf_counter()
        router.predict_arrays(data.X[row : row + 1], texts[row : row + 1] if texts else None)
        timings.append((time.perf_counter() - t0) * 1000)

    result: dict[str, Any] = {
        "model_dir": str(model_dir),
        "classifier": router.classifier,
        "feature_set": router.feature_set,
        "split": split,
        "hyperparameters": meta.get("hyperparameters"),
        "embedding_model": router.embedding_model,
        "embedding_dimension": router.dimension,
        split: _metrics(classes, data.y, indices, proba_all),
        "validation_metrics": common.brief(meta.get("validation_metrics") or {}),
        "training_seconds": meta.get("training_seconds"),
        "training_resources": meta.get("resources"),
        "artifact_bytes": directory_size_bytes(model_dir),
        "local_inference_latency_ms": {
            "single_query_samples": len(timings),
            "median": statistics.median(timings) if timings else None,
            "p95": float(np.quantile(timings, 0.95)) if len(timings) >= 20 else None,
            "batch_ms_per_row": batch_seconds * 1000 / max(n, 1),
            "excludes": "embedding API latency",
        },
    }
    if exclude_ids:
        keep = np.array([eid not in exclude_ids for eid in data.example_ids])
        result[f"{split}_excluding_near_duplicates"] = {
            "excluded_rows": int((~keep).sum()),
            **common.brief(
                _metrics(
                    classes,
                    data.y[keep],
                    indices[keep],
                    None if proba_all is None else proba_all[keep],
                )
            ),
        }
    result["evaluation_peak_rss_gib"] = to_gib(peak_rss_bytes())
    return result


def evaluate_many(
    cfg: ComplexityConfig,
    model_dirs: list[Path],
    out_dir: Path,
    *,
    split: str = "test",
    exclude_ids: set[str] | None = None,
    latency_samples: int = 500,
) -> dict[str, Any]:
    limit_threads(cfg.resources.threads)
    results: dict[str, Any] = {}
    for model_dir in model_dirs:
        name = model_dir.name
        try:
            results[name] = evaluate_model_dir(
                cfg,
                model_dir,
                split=split,
                exclude_ids=exclude_ids,
                latency_samples=latency_samples,
            )
        except Exception as exc:  # record and continue with the remaining models
            log.error("evaluation failed", model=name, error=f"{type(exc).__name__}: {exc}")
            results[name] = {"model_dir": str(model_dir), "error": f"{type(exc).__name__}: {exc}"}
        common.write_json(out_dir / f"{name}.json", results[name])
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/complexity.yaml")
    parser.add_argument("--model-dirs", nargs="+", required=True)
    parser.add_argument("--split", default="test", choices=["validation", "test"])
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--exclude-ids-file", default=None, help="JSON list of example IDs")
    parser.add_argument("--latency-samples", type=int, default=500)
    parser.add_argument("--threads", type=int, default=None)
    args = parser.parse_args(argv)
    configure_logging()
    cfg = load_config(args.config)
    if args.threads:
        cfg = cfg.model_copy(
            update={"resources": cfg.resources.model_copy(update={"threads": args.threads})}
        )
    out_dir = Path(args.out_dir) if args.out_dir else cfg.paths.reports_dir / "evaluation"
    exclude = (
        set(json.loads(Path(args.exclude_ids_file).read_text(encoding="utf-8")))
        if args.exclude_ids_file
        else None
    )
    results = evaluate_many(
        cfg,
        [Path(p) for p in args.model_dirs],
        out_dir,
        split=args.split,
        exclude_ids=exclude,
        latency_samples=args.latency_samples,
    )
    brief = {name: r.get("error") or common.brief(r[args.split]) for name, r in results.items()}
    print(json.dumps(brief, indent=2))


if __name__ == "__main__":
    main()
