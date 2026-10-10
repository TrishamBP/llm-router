"""Feature sets, schema, and aligned split loading for the XGBoost / MLP complexity classifiers.

Feature groups (kept separate so their contribution can be measured):
  - `embedding`: the cached Fireworks vector (`emb_0000` ... `emb_<d-1>`), unchanged;
  - `engineered`: `text_features` counts after a fixed log1p.

Feature sets: `embeddings`, `engineered`, `embeddings+engineered` (columns in that order).

Row alignment is verified on every load: each embedding row's example_id, label, and query hash
must match the processed split file. Embeddings stay memory-mapped; `FeatureMatrix` builds
float32 row blocks on demand, so the combined feature set never needs a second full copy.
Labels are encoded in `LABELS` order (easy=0, medium=1, hard=2).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from llm_router.config import LABELS
from llm_router.data.datasets import AlignmentError, load_embedded_split
from llm_router.features import text_features

FeatureSet = Literal["embeddings", "engineered", "embeddings+engineered"]
FEATURE_SETS: tuple[FeatureSet, ...] = ("embeddings", "engineered", "embeddings+engineered")
SCHEMA_VERSION = "1"
LABEL_TO_INDEX = {label: i for i, label in enumerate(LABELS)}


class FeatureSchemaError(ValueError):
    pass


def encode_labels(labels: Sequence[str] | np.ndarray) -> np.ndarray:
    try:
        return np.fromiter((LABEL_TO_INDEX[str(x)] for x in labels), dtype=np.int32)
    except KeyError as exc:
        raise AlignmentError(f"unexpected label {exc.args[0]!r}") from None


def decode_labels(indices: np.ndarray) -> np.ndarray:
    return np.asarray(LABELS)[np.asarray(indices, dtype=np.int64)]


def uses_embeddings(feature_set: str) -> bool:
    return feature_set in ("embeddings", "embeddings+engineered")


def uses_engineered(feature_set: str) -> bool:
    return feature_set in ("engineered", "embeddings+engineered")


def build_schema(feature_set: FeatureSet, embedding_metadata: dict[str, Any]) -> dict[str, Any]:
    if feature_set not in FEATURE_SETS:
        raise FeatureSchemaError(f"unknown feature set '{feature_set}'")
    dim = int(embedding_metadata["dimension"])
    names: list[str] = []
    groups: dict[str, Any] = {}
    if uses_embeddings(feature_set):
        groups["embedding"] = {"start": 0, "stop": dim, "transform": "none"}
        names += [f"emb_{i:04d}" for i in range(dim)]
    engineered = text_features.schema() if uses_engineered(feature_set) else None
    if engineered is not None:
        start = len(names)
        groups["engineered"] = {"start": start, "stop": start + len(engineered["names"])}
        names += engineered["names"]
    return {
        "schema_version": SCHEMA_VERSION,
        "feature_set": feature_set,
        "n_features": len(names),
        "feature_names": names,
        "groups": groups,
        "labels": list(LABELS),
        "text_normalisation": "llm_router.data.splitting.normalize_text",
        # Embedding identity is recorded even for `engineered`: the router still embeds
        # nothing in that case, but the training data provenance stays traceable.
        "embedding_model": embedding_metadata["embedding_model"],
        "embedding_dimension": dim,
        "embedding_cache_key": embedding_metadata["cache_key"],
        "engineered": engineered,
    }


def check_schema(schema: dict[str, Any]) -> None:
    """Refuse a saved schema the current code cannot reproduce exactly."""
    if schema.get("schema_version") != SCHEMA_VERSION:
        raise FeatureSchemaError(f"feature schema version {schema.get('schema_version')!r}")
    if schema.get("labels") != list(LABELS):
        raise FeatureSchemaError(f"label order {schema.get('labels')} != {list(LABELS)}")
    feature_set = schema.get("feature_set")
    if feature_set not in FEATURE_SETS:
        raise FeatureSchemaError(f"unknown feature set {feature_set!r}")
    saved = schema.get("engineered")
    if uses_engineered(feature_set):
        current = text_features.schema()
        if saved is None or saved.get("definition_hash") != current["definition_hash"]:
            raise FeatureSchemaError(
                "engineered feature definitions differ from the ones the model was trained "
                f"with (saved {saved and saved.get('definition_hash')}, "
                f"current {current['definition_hash']})"
            )
    expected = (int(schema["embedding_dimension"]) if uses_embeddings(feature_set) else 0) + (
        len(saved["names"]) if saved else 0
    )
    if schema.get("n_features") != expected or len(schema.get("feature_names", [])) != expected:
        raise FeatureSchemaError("feature schema is internally inconsistent")


def build_features(
    schema: dict[str, Any], embeddings: np.ndarray | None, texts: Sequence[str] | None
) -> np.ndarray:
    """Model input matrix for rows of embeddings and/or (already normalised) texts."""
    feature_set = schema["feature_set"]
    blocks = []
    if uses_embeddings(feature_set):
        if embeddings is None:
            raise FeatureSchemaError("this feature set needs embeddings")
        emb = np.asarray(embeddings, dtype=np.float32)
        if emb.ndim != 2 or emb.shape[1] != schema["embedding_dimension"]:
            raise FeatureSchemaError(
                f"embedding dimension {emb.shape[-1]} != schema {schema['embedding_dimension']}"
            )
        blocks.append(emb)
    if uses_engineered(feature_set):
        if texts is None:
            raise FeatureSchemaError("this feature set needs the query texts")
        blocks.append(text_features.extract(texts))
    if len(blocks) == 2 and blocks[0].shape[0] != blocks[1].shape[0]:
        raise FeatureSchemaError("embeddings and texts have different row counts")
    out = blocks[0] if len(blocks) == 1 else np.hstack(blocks)
    return np.ascontiguousarray(out, dtype=np.float32)


class FeatureMatrix:
    """Lazy row access over (memory-mapped embeddings, precomputed engineered features)."""

    def __init__(self, embeddings: np.ndarray | None, engineered: np.ndarray | None) -> None:
        if embeddings is None and engineered is None:
            raise ValueError("FeatureMatrix needs at least one feature group")
        if (
            embeddings is not None
            and engineered is not None
            and embeddings.shape[0] != engineered.shape[0]
        ):
            raise AlignmentError("feature groups have different row counts")
        self.embeddings = embeddings
        self.engineered = engineered
        first = embeddings if embeddings is not None else engineered
        assert first is not None
        self.n_rows = int(first.shape[0])
        self.n_features = sum(int(a.shape[1]) for a in (embeddings, engineered) if a is not None)

    def _select(self, index: slice | np.ndarray) -> np.ndarray:
        parts = [a[index] for a in (self.embeddings, self.engineered) if a is not None]
        out = parts[0] if len(parts) == 1 else np.hstack(parts)
        return np.ascontiguousarray(out, dtype=np.float32)

    def rows(self, start: int, stop: int) -> np.ndarray:
        return self._select(slice(start, stop))

    def take(self, indices: np.ndarray) -> np.ndarray:
        # Sorted access is much faster on a memory map; restore the requested order after.
        order = np.argsort(indices, kind="stable")
        block = self._select(np.asarray(indices)[order])
        out = np.empty_like(block)
        out[order] = block
        return out

    def iter_batches(self, batch_rows: int) -> Iterator[np.ndarray]:
        for start in range(0, self.n_rows, batch_rows):
            yield self.rows(start, min(start + batch_rows, self.n_rows))

    def subset(self, indices: np.ndarray) -> FeatureMatrix:
        """In-memory copy of selected rows (used for smoke runs and small splits)."""
        idx = np.sort(np.asarray(indices))
        emb = None if self.embeddings is None else np.asarray(self.embeddings[idx], np.float32)
        eng = None if self.engineered is None else self.engineered[idx]
        return FeatureMatrix(emb, eng)


@dataclass(frozen=True)
class SplitFeatures:
    name: str
    matrix: FeatureMatrix
    y: np.ndarray  # int32 indices into LABELS
    example_ids: tuple[str, ...]
    texts: tuple[str, ...] | None  # normalised queries, kept only if engineered features used

    def subset(self, indices: np.ndarray) -> SplitFeatures:
        idx = np.sort(np.asarray(indices))
        return SplitFeatures(
            self.name,
            self.matrix.subset(idx),
            self.y[idx],
            tuple(self.example_ids[i] for i in idx),
            None if self.texts is None else tuple(self.texts[i] for i in idx),
        )


def query_hash(query: str) -> str:
    return hashlib.sha256(query.encode()).hexdigest()[:16]


def load_aligned_queries(
    processed_dir: Path, embeddings_dir: Path, split: str
) -> tuple[list[str], list[str], list[str]]:
    """(example_ids, labels, queries) in embedding-row order, verified against processed data.

    Streams the processed split file and keeps only that split's queries.
    """
    with (embeddings_dir / f"{split}.row_mapping.jsonl").open(encoding="utf-8") as fh:
        mapping = [json.loads(line) for line in fh if line.strip()]
    wanted = {m["example_id"]: m for m in mapping}
    found: dict[str, dict[str, Any]] = {}
    with (processed_dir / f"{split}.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row["example_id"] in wanted:
                found[row["example_id"]] = {"label": row["label"], "query": row["query"]}
    queries, labels = [], []
    for m in mapping:
        row = found.get(m["example_id"])
        if row is None:
            raise AlignmentError(f"{split}: {m['example_id']} missing from processed data")
        if row["label"] != m["label"]:
            raise AlignmentError(f"{split}: label mismatch for {m['example_id']}")
        if "query_sha256" in m and query_hash(row["query"]) != m["query_sha256"]:
            raise AlignmentError(f"{split}: query hash mismatch for {m['example_id']}")
        queries.append(row["query"])
        labels.append(row["label"])
    return [m["example_id"] for m in mapping], labels, queries


def load_split_features(
    processed_dir: Path,
    embeddings_dir: Path,
    split: str,
    feature_set: FeatureSet,
    *,
    mmap: bool = True,
) -> tuple[SplitFeatures, dict[str, Any]]:
    """Aligned features for one split, plus the embedding cache metadata."""
    if feature_set not in FEATURE_SETS:
        raise FeatureSchemaError(f"unknown feature set '{feature_set}'")
    emb = load_embedded_split(embeddings_dir, split, mmap=mmap)
    manifest = json.loads((processed_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["processed_fingerprint"] != emb.metadata["processed_fingerprint"]:
        raise AlignmentError("embeddings were generated from a different processed dataset")
    ids, labels, queries = load_aligned_queries(processed_dir, embeddings_dir, split)
    if tuple(ids) != emb.example_ids or list(emb.y) != labels:
        raise AlignmentError(f"{split}: row mapping and processed data disagree")
    engineered = text_features.extract(queries) if uses_engineered(feature_set) else None
    matrix = FeatureMatrix(emb.X if uses_embeddings(feature_set) else None, engineered)
    texts = tuple(queries) if engineered is not None else None
    return SplitFeatures(split, matrix, encode_labels(labels), emb.example_ids, texts), emb.metadata


class Standardizer:
    """Per-feature mean / std, fitted in row chunks on the training split only."""

    def __init__(self, mean: np.ndarray, scale: np.ndarray) -> None:
        self.mean = np.asarray(mean, dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)

    @classmethod
    def fit(cls, matrix: FeatureMatrix, chunk_rows: int = 8192) -> Standardizer:
        total = np.zeros(matrix.n_features, dtype=np.float64)
        sq = np.zeros(matrix.n_features, dtype=np.float64)
        for block in matrix.iter_batches(chunk_rows):
            b = block.astype(np.float64)
            total += b.sum(axis=0)
            sq += np.square(b).sum(axis=0)
        n = max(matrix.n_rows, 1)
        mean = total / n
        std = np.sqrt(np.maximum(sq / n - mean**2, 0.0))
        return cls(mean, np.where(std < 1e-6, 1.0, std))

    def transform(self, X: np.ndarray) -> np.ndarray:
        if X.shape[1] != self.mean.shape[0]:
            raise FeatureSchemaError(f"standardizer expects {self.mean.shape[0]} features")
        return ((X - self.mean) / self.scale).astype(np.float32, copy=False)

    def save(self, path: Path) -> None:
        np.savez(path, mean=self.mean, scale=self.scale)

    @classmethod
    def load(cls, path: Path) -> Standardizer:
        with np.load(path) as data:
            return cls(data["mean"], data["scale"])
