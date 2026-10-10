"""Build the merged complexity training set: complexity data + label-mapped router data.

Usage:
    uv run python -m llm_router.data.build_merged_dataset [--merged-config CONFIG]

Label mapping (a user decision, not a property of the source data):
    large_llm -> hard, small_llm -> medium      (no router rows become `easy`)

Composition:
- train      = complexity train + router *train* rows (mapped), minus router rows that
               (a) share a normalised prompt with ANY complexity row (complexity labels win),
               or (b) have cosine >= 0.98 to any complexity validation/test row
               (near-paraphrase leakage into evaluation);
- validation = complexity validation, unchanged;
- test       = complexity test, unchanged (so results compare with the complexity-only run).
Router validation/test rows are not used here (kept for a separate, secondary evaluation).

Outputs follow the shared layout, so every trainer and the comparison runner work unchanged:
`paths.processed_dir/{train,validation,test}.jsonl + manifest.json` and
`paths.embeddings_dir/<key>/{split}.npy + row mappings + metadata.json` (+ latest.json).
The training matrix is streamed into a memory-mapped file; ends with
`verify_embedding_artifacts` on the result.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from llm_router.config import ComplexityConfig, load_config, sha256_json
from llm_router.data.datasets import load_embedded_split
from llm_router.data.leakage import nearest_train_similarity, verify_embedding_artifacts
from llm_router.data.preprocess import SPLITS, load_processed
from llm_router.data.router_dataset import (
    RouterDatasetConfig,
    load_router_config,
    load_router_dataset,
)
from llm_router.data.splitting import group_key
from llm_router.embeddings.generate import resolve_embeddings_dir
from llm_router.logging import configure_logging, get_logger
from llm_router.training.resources import peak_rss_bytes, to_gib

log = get_logger(__name__)

LABEL_MAP = {"large_llm": "hard", "small_llm": "medium"}
NEAR_DUPLICATE_THRESHOLD = 0.98
MERGE_VERSION = "1"
CHUNK = 4096


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def build(
    merged_cfg: ComplexityConfig,
    *,
    complexity_config: str | Path | ComplexityConfig = "configs/complexity.yaml",
    router_config: str | Path | RouterDatasetConfig = "configs/router_dataset.yaml",
    threshold: float = NEAR_DUPLICATE_THRESHOLD,
) -> dict[str, Any]:
    started = time.perf_counter()
    base_cfg = (
        complexity_config
        if isinstance(complexity_config, ComplexityConfig)
        else load_config(complexity_config)
    )
    r_cfg = (
        router_config
        if isinstance(router_config, RouterDatasetConfig)
        else load_router_config(router_config)
    )
    base_emb = resolve_embeddings_dir(base_cfg)
    base_examples, base_manifest = load_processed(base_cfg.paths.processed_dir)
    router = load_router_dataset(config=r_cfg)  # fully validated
    base_meta = json.loads((base_emb / "metadata.json").read_text(encoding="utf-8"))
    r_meta = router.embedding_metadata
    for key in ("embedding_model", "dimension", "dimensions_parameter"):
        if base_meta.get(key) != r_meta.get(key):
            raise ValueError(
                f"embedding caches are incompatible: {key} "
                f"{base_meta.get(key)!r} != {r_meta.get(key)!r}"
            )
    if not base_meta.get("complete"):
        raise ValueError("complexity embedding cache is incomplete")
    dim = int(base_meta["dimension"])

    # Router train records in embedding-row order.
    r_proc = r_cfg.paths.processed_dir
    r_records = {r["example_id"]: r for r in _read_jsonl(r_proc / "train.jsonl")}
    r_rows = [r_records[i] for i in router.example_ids["train"]]

    # (a) exact normalised-prompt overlap with any complexity split.
    base_groups = {e["group"]: s for s in SPLITS for e in base_examples[s]}
    dropped: dict[str, str] = {}
    for r in r_rows:
        hit = base_groups.get(group_key(r["prompt"]))
        if hit is not None:
            dropped[r["example_id"]] = f"same normalised prompt as complexity {hit} row"

    # (b) near-paraphrases of complexity validation/test rows (cosine >= threshold).
    near_counts: dict[str, int] = {}
    for split in ("validation", "test"):
        eval_split = load_embedded_split(base_emb, split, mmap=True)
        best, _ = nearest_train_similarity(eval_split.X, router.X_train)
        hits = np.flatnonzero(best >= threshold)
        near_counts[split] = int(len(hits))
        for i in hits:
            dropped.setdefault(
                r_rows[int(i)]["example_id"], f"cosine >= {threshold} to complexity {split}"
            )
    keep_idx = [i for i, r in enumerate(r_rows) if r["example_id"] not in dropped]

    # Processed records.
    out_proc = merged_cfg.paths.processed_dir
    out_proc.mkdir(parents=True, exist_ok=True)
    router_train = [
        {
            "example_id": f"router-{r_rows[i]['example_id']}",
            "query": r_rows[i]["prompt"],
            "label": LABEL_MAP[r_rows[i]["label"]],
            "original_label": r_rows[i]["label"],
            "source_dataset": "DevQuasar/llm_router_dataset-synth",
            "source_id": r_rows[i]["source_id"],
            "source_split": r_rows[i]["source_split"],
            "source_index": r_rows[i]["source_row_index"],
            "group": group_key(r_rows[i]["prompt"]),
            "router_group_id": r_rows[i]["group_id"],
            "split": "train",
        }
        for i in keep_idx
    ]
    merged_examples = {
        "train": [
            dict(e, source_dataset="regolo/brick-complexity-extractor")
            for e in base_examples["train"]
        ]
        + router_train,
        "validation": base_examples["validation"],
        "test": base_examples["test"],
    }
    seen: dict[str, str] = {}
    for split in SPLITS:  # isolation: example IDs and normalised-prompt groups
        for e in merged_examples[split]:
            for kind, value in (("id", e["example_id"]), ("group", e["group"])):
                tag = f"{kind}:{value}"
                if seen.setdefault(tag, split) != split:
                    raise AssertionError(f"{kind} {value[:16]} spans {seen[tag]} and {split}")
    for split in SPLITS:
        _write_jsonl(out_proc / f"{split}.jsonl", merged_examples[split])

    fingerprint = sha256_json(
        {
            "complexity": base_manifest["processed_fingerprint"],
            "router": router.split_metadata["processed_fingerprint"],
            "label_map": LABEL_MAP,
            "threshold": threshold,
            "kept": [r["example_id"] for r in router_train],
            "version": MERGE_VERSION,
        }
    )[:16]
    label_counts = {
        s: dict(sorted(Counter(e["label"] for e in merged_examples[s]).items())) for s in SPLITS
    }
    manifest = {
        "preprocessing_version": f"merged-{MERGE_VERSION}",
        "processed_fingerprint": fingerprint,
        "raw_content_fingerprint": sha256_json(
            [
                base_manifest.get("raw_content_fingerprint"),
                router.split_metadata.get("raw_content_fingerprint"),
            ]
        )[:16],
        "sources": {
            "complexity": {
                "processed_fingerprint": base_manifest["processed_fingerprint"],
                "embeddings": str(base_emb),
            },
            "router": {
                "processed_fingerprint": router.split_metadata["processed_fingerprint"],
                "embeddings": str(router.embeddings_dir),
            },
        },
        "label_map": LABEL_MAP,
        "label_map_note": "user decision; router labels are routing labels, "
        "mapped for augmentation",
        "router_train_rows": len(r_rows),
        "router_train_kept": len(router_train),
        "router_train_dropped": dict(
            sorted(Counter(v.split(" complexity ")[0] for v in dropped.values()).items())
        ),
        "router_near_duplicates_of_eval": near_counts,
        "near_duplicate_threshold": threshold,
        "splits": {
            s: {
                "rows": len(merged_examples[s]),
                "label_distribution": label_counts[s],
                "source_datasets": dict(
                    Counter(
                        e.get("source_dataset", "regolo/brick-complexity-extractor")
                        for e in merged_examples[s]
                    )
                ),
            }
            for s in SPLITS
        },
        "evaluation_note": "validation/test are the unchanged complexity splits",
        "split_isolation": {
            "verified": True,
            "checked": "example IDs and normalised-prompt groups",
        },
    }
    (out_proc / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out_proc / "dropped_router_rows.json").write_text(
        json.dumps(dropped, indent=2), encoding="utf-8"
    )

    # Embeddings: stream train (complexity rows, then kept router rows); copy validation/test.
    key = sha256_json(
        {
            "processed": fingerprint,
            "model": base_meta["embedding_model"],
            "dims": base_meta.get("dimensions_parameter"),
        }
    )[:16]
    out_emb = merged_cfg.paths.embeddings_dir / key
    out_emb.mkdir(parents=True, exist_ok=True)
    base_train = np.load(base_emb / "train.npy", mmap_mode="r")
    n_total = base_train.shape[0] + len(keep_idx)
    tmp = out_emb / "train.tmp.npy"
    out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(n_total, dim))
    offset = base_train.shape[0]
    for start in range(0, offset, CHUNK):
        stop = min(start + CHUNK, offset)  # never spill into the router rows' region
        out[start:stop] = base_train[start:stop]
    keep = np.asarray(keep_idx, dtype=np.int64)
    for start in range(0, len(keep), CHUNK):
        block = keep[start : start + CHUNK]
        out[offset + start : offset + start + len(block)] = router.X_train[block]
    out.flush()
    del out, base_train
    os.replace(tmp, out_emb / "train.npy")
    base_map = _read_jsonl(base_emb / "train.row_mapping.jsonl")
    r_map = _read_jsonl(router.embeddings_dir / "train.row_mapping.jsonl")
    train_map = base_map + [
        {
            "row": offset + n,
            "example_id": f"router-{r_map[i]['example_id']}",
            "label": LABEL_MAP[r_map[i]["label"]],
            "query_sha256": r_map[i]["query_sha256"],
        }
        for n, i in enumerate(keep_idx)
    ]
    _write_jsonl(out_emb / "train.row_mapping.jsonl", train_map)
    for split in ("validation", "test"):
        shutil.copyfile(base_emb / f"{split}.npy", out_emb / f"{split}.npy")
        shutil.copyfile(
            base_emb / f"{split}.row_mapping.jsonl", out_emb / f"{split}.row_mapping.jsonl"
        )

    metadata = {
        "cache_key": key,
        "embedding_model": base_meta["embedding_model"],
        "reported_models": sorted(
            set(base_meta.get("reported_models", [])) | set(r_meta.get("reported_models", []))
        ),
        "dimension": dim,
        "dimensions_parameter": base_meta.get("dimensions_parameter"),
        "processed_fingerprint": fingerprint,
        "raw_content_fingerprint": manifest["raw_content_fingerprint"],
        "preprocessing_version": manifest["preprocessing_version"],
        "complete": True,
        "assembled_splits": list(SPLITS),
        "rows": {s: len(merged_examples[s]) for s in SPLITS},
        "failed_examples": {s: {} for s in SPLITS},
        "sources": {
            "complexity_cache": base_meta["cache_key"],
            "router_cache": r_meta["cache_key"],
        },
        "usage": {"note": "no new API calls; vectors reused from the two source caches"},
        "embedding_seconds": round(
            float(base_meta.get("embedding_seconds", 0))
            + float(r_meta.get("embedding_seconds", 0)),
            2,
        ),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out_emb / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (merged_cfg.paths.embeddings_dir / "latest.json").write_text(
        json.dumps(
            {"cache_key": key, "embedding_model": metadata["embedding_model"], "dimension": dim}
        ),
        encoding="utf-8",
    )

    verification = verify_embedding_artifacts(out_proc, out_emb)  # the trainers' own checks
    summary = {
        "processed_dir": str(out_proc),
        "embeddings_dir": str(out_emb),
        "label_map": LABEL_MAP,
        "router_train_rows": len(r_rows),
        "router_train_kept": len(router_train),
        "router_train_dropped": manifest["router_train_dropped"],
        "router_near_duplicates_of_eval": near_counts,
        "splits": verification["splits"],
        "train_imbalance_max_min": verification["train_imbalance_max_min"],
        "verified": True,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "peak_rss_gib": to_gib(peak_rss_bytes()),
    }
    merged_cfg.paths.reports_dir.mkdir(parents=True, exist_ok=True)
    (merged_cfg.paths.reports_dir / "merged_dataset.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--merged-config", default="configs/complexity_merged.yaml")
    parser.add_argument("--complexity-config", default="configs/complexity.yaml")
    parser.add_argument("--router-config", default="configs/router_dataset.yaml")
    args = parser.parse_args(argv)
    configure_logging()
    summary = build(
        load_config(args.merged_config),
        complexity_config=args.complexity_config,
        router_config=args.router_config,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
