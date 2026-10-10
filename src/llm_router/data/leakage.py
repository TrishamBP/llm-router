"""Pre-training integrity checks on the cached embeddings and processed splits.

`verify_embedding_artifacts` checks that the embedding cache is complete and matches the
processed dataset (model ID, dimension, fingerprints, row alignment, finite float32 values),
that example IDs and duplicate-group keys never span splits, and reports class balance.

`near_duplicate_audit` goes beyond the normalised-text grouping used for splitting: it finds
validation/test rows whose embedding has cosine similarity >= a threshold with some training
row (paraphrases, near-identical prompts). It only reports; splits are never changed after
the fact, but flagged IDs can be excluded in a sensitivity evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from llm_router.config import LABELS
from llm_router.data.datasets import AlignmentError, assert_disjoint_ids
from llm_router.data.preprocess import SPLITS
from llm_router.features.complexity_features import load_split_features


def verify_embedding_artifacts(processed_dir: Path, embeddings_dir: Path) -> dict[str, Any]:
    metadata = json.loads((embeddings_dir / "metadata.json").read_text(encoding="utf-8"))
    manifest = json.loads((processed_dir / "manifest.json").read_text(encoding="utf-8"))
    problems = []
    if not metadata.get("complete"):
        problems.append(
            "embedding cache is incomplete (metadata.complete is false); finish "
            "`python -m llm_router.embeddings.generate --yes` first"
        )
    missing = [s for s in SPLITS if not (embeddings_dir / f"{s}.npy").exists()]
    if missing:
        problems.append(f"missing assembled splits: {missing}")
    if not manifest.get("split_isolation", {}).get("verified"):
        problems.append("processed manifest does not record verified split isolation")
    if metadata.get("processed_fingerprint") != manifest.get("processed_fingerprint"):
        problems.append("embeddings were generated from a different processed dataset")
    connectivity = embeddings_dir.parent / "connectivity" / "metadata.json"
    if connectivity.exists():
        conn = json.loads(connectivity.read_text(encoding="utf-8"))
        if conn.get("model") == metadata.get("embedding_model") and int(conn["dimension"]) != int(
            metadata["dimension"]
        ):
            problems.append("connectivity test and cache disagree on the embedding dimension")
    if problems:
        raise AlignmentError("; ".join(problems))

    ids: dict[str, tuple[str, ...]] = {}
    splits: dict[str, Any] = {}
    for split in SPLITS:
        # Loading checks dtype, finiteness (chunked), dimension, IDs, labels, and query hashes.
        features, _ = load_split_features(processed_dir, embeddings_dir, split, "embeddings")
        ids[split] = features.example_ids
        counts = np.bincount(features.y, minlength=len(LABELS))
        splits[split] = {
            "rows": len(features.example_ids),
            "label_counts": {label: int(c) for label, c in zip(LABELS, counts, strict=True)},
            "failed_examples": len(metadata.get("failed_examples", {}).get(split, {})),
        }
    assert_disjoint_ids(ids)

    group_split: dict[str, str] = {}
    for split in SPLITS:
        with (processed_dir / f"{split}.jsonl").open(encoding="utf-8") as fh:
            for line in fh:
                group = json.loads(line)["group"]
                if group_split.setdefault(group, split) != split:
                    raise AlignmentError(
                        f"duplicate group {group[:12]} spans {group_split[group]} and {split}"
                    )
    train_counts = np.array(list(splits["train"]["label_counts"].values()))
    return {
        "embeddings_dir": str(embeddings_dir),
        "embedding_model": metadata["embedding_model"],
        "reported_models": metadata.get("reported_models"),
        "dimension": int(metadata["dimension"]),
        "processed_fingerprint": metadata["processed_fingerprint"],
        "raw_content_fingerprint": metadata.get("raw_content_fingerprint"),
        "splits": splits,
        "total_rows": sum(s["rows"] for s in splits.values()),
        "train_imbalance_max_min": round(float(train_counts.max() / max(train_counts.min(), 1)), 3),
        "example_ids_disjoint": True,
        "duplicate_groups_disjoint": True,
        "groups_checked": len(group_split),
    }


def _normalise(block: np.ndarray) -> np.ndarray:
    block = np.asarray(block, dtype=np.float32)
    norms = np.linalg.norm(block, axis=1, keepdims=True)
    out: np.ndarray = block / np.maximum(norms, 1e-12)
    return out


def nearest_train_similarity(
    train_X: np.ndarray, target_X: np.ndarray, *, chunk_rows: int = 4096
) -> tuple[np.ndarray, np.ndarray]:
    """For each target row: max cosine similarity to any training row, and that row's index.

    Training rows are streamed in chunks (works on memory-mapped arrays), so peak memory is
    about chunk_rows x n_target similarities plus one normalised chunk.
    """
    target = _normalise(target_X)
    best = np.full(target.shape[0], -1.0, dtype=np.float32)
    best_row = np.zeros(target.shape[0], dtype=np.int64)
    for start in range(0, train_X.shape[0], chunk_rows):
        sims = _normalise(train_X[start : start + chunk_rows]) @ target.T
        arg = sims.argmax(axis=0)
        top = sims[arg, np.arange(sims.shape[1])]
        better = top > best
        best[better] = top[better]
        best_row[better] = arg[better] + start
        del sims
    return best, best_row


def near_duplicate_audit(
    embeddings_dir: Path,
    *,
    splits: tuple[str, ...] = ("validation", "test"),
    thresholds: tuple[float, ...] = (0.95, 0.98, 0.99),
    flag_threshold: float = 0.98,
    chunk_rows: int = 4096,
) -> dict[str, Any]:
    """Max cosine similarity of each validation/test row to any training row (chunked)."""
    from llm_router.data.datasets import load_embedded_split

    train = load_embedded_split(embeddings_dir, "train", mmap=True)
    report: dict[str, Any] = {"thresholds": list(thresholds), "flag_threshold": flag_threshold}
    for split in splits:
        other = load_embedded_split(embeddings_dir, split, mmap=True)
        best, best_row = nearest_train_similarity(train.X, other.X, chunk_rows=chunk_rows)
        flagged = np.flatnonzero(best >= flag_threshold)
        report[split] = {
            "rows": int(best.shape[0]),
            "max_similarity_quantiles": {
                q: float(np.quantile(best, float(q))) for q in ("0.5", "0.9", "0.99")
            }
            if best.size
            else None,
            "rows_at_or_above": {str(t): int((best >= t).sum()) for t in thresholds},
            "flagged": [
                {
                    "example_id": other.example_ids[i],
                    "nearest_train_id": train.example_ids[int(best_row[i])],
                    "cosine": round(float(best[i]), 5),
                    "same_label": bool(other.y[i] == train.y[int(best_row[i])]),
                }
                for i in flagged
            ],
        }
    return report
