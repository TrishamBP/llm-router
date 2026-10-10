"""Verify the router dataset's processed splits and embedding cache end to end.

Usage:
    uv run python -m llm_router.data.validate_router_artifacts [--config CONFIG]

Runs every check in `load_router_dataset` (fingerprints, label mapping, row alignment,
prompt hashes, dimensions, finiteness, split isolation), then an embedding-space
near-duplicate audit (cosine similarity of each validation/test row to its nearest training
row; report only, splits are never changed). Writes
`paths.reports_dir/router_artifacts_validation.{json,md}`. No API calls.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from typing import Any

import numpy as np

from llm_router.data.leakage import nearest_train_similarity
from llm_router.data.router_dataset import (
    DEFAULT_CONFIG,
    RouterDatasetConfig,
    RouterDatasetError,
    load_router_config,
    load_router_dataset,
)
from llm_router.logging import configure_logging, get_logger
from llm_router.training.resources import peak_rss_bytes, to_gib

log = get_logger(__name__)

THRESHOLDS = (0.95, 0.98, 0.99)
FLAG = 0.98


def validate(cfg: RouterDatasetConfig) -> dict[str, Any]:
    started = time.perf_counter()
    data = load_router_dataset(config=cfg, mmap=True)
    names = data.label_names
    distribution = {}
    for split, y in (
        ("train", data.y_train),
        ("validation", data.y_validation),
        ("test", data.y_test),
    ):
        counts = Counter(names[int(i)] for i in y)
        distribution[split] = {
            "rows": int(len(y)),
            "counts": dict(sorted(counts.items())),
            "proportions": {k: round(v / len(y), 4) for k, v in sorted(counts.items())},
        }
    near: dict[str, Any] = {"thresholds": list(THRESHOLDS), "flag_threshold": FLAG}
    for split, X, y in (
        ("validation", data.X_validation, data.y_validation),
        ("test", data.X_test, data.y_test),
    ):
        best, row = nearest_train_similarity(data.X_train, X)
        flagged = np.flatnonzero(best >= FLAG)
        near[split] = {
            "rows": int(len(best)),
            "max_similarity_quantiles": {
                q: round(float(np.quantile(best, float(q))), 4) for q in ("0.5", "0.9", "0.99")
            },
            "rows_at_or_above": {str(t): int((best >= t).sum()) for t in THRESHOLDS},
            "flagged_same_label": int(sum(y[i] == data.y_train[row[i]] for i in flagged)),
            "flagged": [
                {
                    "example_id": data.example_ids[split][i],
                    "nearest_train_id": data.example_ids["train"][int(row[i])],
                    "cosine": round(float(best[i]), 5),
                }
                for i in flagged
            ],
        }
    meta = data.embedding_metadata
    return {
        "valid": True,
        "checks": data.checks,
        "embeddings_dir": str(data.embeddings_dir),
        "embedding_model": meta["embedding_model"],
        "reported_models": meta.get("reported_models"),
        "dimension": meta["dimension"],
        "dtype": str(data.X_train.dtype),
        "label_mapping": data.label_mapping["label_to_id"],
        "class_distribution": distribution,
        "class_imbalance_ratio_train": data.split_metadata.get("class_imbalance_ratio_train"),
        "near_duplicates": near,
        "usage": meta.get("usage"),
        "estimated_cost_usd": meta.get("estimated_cost_usd"),
        "processed_fingerprint": meta["processed_fingerprint"],
        "resources": {
            "elapsed_seconds": round(time.perf_counter() - started, 2),
            "peak_rss_gib": to_gib(peak_rss_bytes()),
        },
    }


def to_markdown(r: dict[str, Any]) -> str:
    lines = [
        "# Router dataset artifact validation",
        "",
        f"- Valid: **{r['valid']}**; model `{r['embedding_model']}`, "
        f"{r['dimension']} dims, {r['dtype']}",
        f"- Cache: `{r['embeddings_dir']}`; processed fingerprint `{r['processed_fingerprint']}`",
        f"- Label mapping: {r['label_mapping']}; "
        f"train imbalance ratio {r['class_imbalance_ratio_train']}",
        f"- Checks: {r['checks']['alignment']}; {r['checks']['split_isolation']}",
        "",
        "| Split | Rows | Counts | Proportions |",
        "|---|---|---|---|",
        *[
            f"| {s} | {d['rows']} | {d['counts']} | {d['proportions']} |"
            for s, d in r["class_distribution"].items()
        ],
        "",
        "Embedding near-duplicates (max cosine to any training row):",
        *[
            f"- {s}: {r['near_duplicates'][s]['rows_at_or_above']} rows at or above thresholds; "
            f"quantiles {r['near_duplicates'][s]['max_similarity_quantiles']}"
            for s in ("validation", "test")
        ],
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    configure_logging()
    cfg = load_router_config(args.config)
    out = cfg.paths.reports_dir
    out.mkdir(parents=True, exist_ok=True)
    try:
        report = validate(cfg)
    except (RouterDatasetError, FileNotFoundError) as exc:
        (out / "router_artifacts_validation.json").write_text(
            json.dumps({"valid": False, "error": str(exc)}, indent=2), encoding="utf-8"
        )
        sys.exit(f"router artifacts invalid: {exc}")
    (out / "router_artifacts_validation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (out / "router_artifacts_validation.md").write_text(to_markdown(report), encoding="utf-8")
    # IDs to exclude in a sensitivity evaluation (splits themselves are never changed).
    exclude = {
        split: [f["example_id"] for f in report["near_duplicates"][split]["flagged"]]
        for split in ("validation", "test")
    }
    exclude["threshold"] = FLAG  # type: ignore[assignment]
    (out / "router_near_duplicate_ids.json").write_text(
        json.dumps(exclude, indent=2), encoding="utf-8"
    )
    keys = (
        "valid",
        "checks",
        "dimension",
        "dtype",
        "label_mapping",
        "class_distribution",
        "resources",
    )
    print(json.dumps({k: report[k] for k in keys}, indent=2))
    print(
        json.dumps(
            {s: report["near_duplicates"][s]["rows_at_or_above"] for s in ("validation", "test")}
        )
    )


if __name__ == "__main__":
    main()
