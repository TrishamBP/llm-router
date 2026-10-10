"""Load cached embedding matrices with strictly validated row alignment."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from llm_router.config import LABELS


class AlignmentError(ValueError):
    pass


@dataclass(frozen=True)
class EmbeddedSplit:
    name: str
    X: np.ndarray  # (n, d) float32
    y: np.ndarray  # (n,) str labels
    example_ids: tuple[str, ...]
    metadata: dict[str, Any]


def load_embedding_metadata(embeddings_dir: Path) -> dict[str, Any]:
    path = embeddings_dir / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; generate embeddings first")
    metadata: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return metadata


def all_finite(X: np.ndarray, chunk_rows: int = 4096) -> bool:
    """Finiteness check in row chunks, so a memory-mapped matrix is never copied whole."""
    return all(np.isfinite(X[i : i + chunk_rows]).all() for i in range(0, X.shape[0], chunk_rows))


def load_embedded_split(embeddings_dir: Path, split: str, *, mmap: bool = False) -> EmbeddedSplit:
    """Load one split. With `mmap=True` the matrix is a read-only memory map (no RAM copy)."""
    metadata = load_embedding_metadata(embeddings_dir)
    X = np.load(embeddings_dir / f"{split}.npy", mmap_mode="r" if mmap else None)
    with (embeddings_dir / f"{split}.row_mapping.jsonl").open(encoding="utf-8") as fh:
        mapping = [json.loads(line) for line in fh if line.strip()]
    if [m["row"] for m in mapping] != list(range(len(mapping))):
        raise AlignmentError(f"{split}: row mapping is not 0..n-1")
    if X.ndim != 2 or X.shape[0] != len(mapping):
        raise AlignmentError(f"{split}: {X.shape[0]} vectors but {len(mapping)} mapping rows")
    if X.shape[1] != metadata["dimension"]:
        raise AlignmentError(f"{split}: dimension {X.shape[1]} != metadata {metadata['dimension']}")
    if X.dtype != np.float32:
        raise AlignmentError(f"{split}: embeddings are {X.dtype}, expected float32")
    if not all_finite(X):
        raise AlignmentError(f"{split}: non-finite values in embeddings")
    ids = tuple(m["example_id"] for m in mapping)
    if len(set(ids)) != len(ids):
        raise AlignmentError(f"{split}: duplicate example IDs")
    y = np.asarray([m["label"] for m in mapping])
    bad = sorted(set(y) - set(LABELS))
    if bad:
        raise AlignmentError(f"{split}: unexpected labels {bad}")
    return EmbeddedSplit(split, X, y, ids, metadata)


def assert_disjoint(*splits: EmbeddedSplit) -> None:
    assert_disjoint_ids({split.name: split.example_ids for split in splits})


def assert_disjoint_ids(splits: Mapping[str, Sequence[str]]) -> None:
    seen: dict[str, str] = {}
    for name, example_ids in splits.items():
        for example_id in example_ids:
            if example_id in seen:
                raise AlignmentError(f"{example_id} appears in both {seen[example_id]} and {name}")
            seen[example_id] = name
