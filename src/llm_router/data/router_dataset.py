"""DevQuasar/llm_router_dataset-synth: config, label mapping, and a validated data loader.

    from llm_router.data.router_dataset import load_router_dataset

    data = load_router_dataset()          # latest embedding cache for configs/router_dataset.yaml
    data.X_train, data.y_train            # memory-mapped float32 (n, d), int64 class IDs
    data.label_names                      # class ID -> original label string

The source labels are routing labels (`small_llm` / `large_llm`, as declared by the dataset
card). The card does not document how they were assigned; they are not complexity scores.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from llm_router.config import (
    ComplexityConfig,
    DatasetConfig,
    EmbeddingConfig,
    PathsConfig,
    sha256_json,
)

SPLITS = ("train", "validation", "test")
DEFAULT_CONFIG = "configs/router_dataset.yaml"
MANIFEST_NAME = "split_metadata.json"
TEXT_FIELD = "prompt"


class RouterDatasetError(ValueError):
    pass


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RouterSourceConfig(_Section):
    repo_id: str
    revision: str = Field(min_length=7)
    data_files: dict[str, str]  # official split -> file in the dataset repository
    card_file: str = "README.md"
    id_field: str = "id"
    text_field: str = "prompt"
    label_field: str = "label"
    license: str


class RouterPreparationConfig(_Section):
    version: str = "1"
    split_ratios: dict[str, float] = {"train": 0.8, "validation": 0.1, "test": 0.1}
    seed: int = 42
    group_by_source_id: bool = True  # rows sharing a source ID always stay together
    duplicate_policy: Literal["keep_first"] = "keep_first"
    conflict_policy: Literal["drop"] = "drop"

    @field_validator("split_ratios")
    @classmethod
    def _ratios(cls, value: dict[str, float]) -> dict[str, float]:
        if set(value) != set(SPLITS):
            raise ValueError("split_ratios needs exactly train, validation, test")
        if any(r <= 0 for r in value.values()) or abs(sum(value.values()) - 1) > 1e-9:
            raise ValueError("split ratios must be positive and sum to 1")
        return value


class RouterPathsConfig(_Section):
    raw_dir: Path
    processed_dir: Path
    embeddings_dir: Path
    reports_dir: Path


class RouterDatasetConfig(_Section):
    dataset: RouterSourceConfig
    preparation: RouterPreparationConfig = RouterPreparationConfig()
    embeddings: EmbeddingConfig = EmbeddingConfig()
    paths: RouterPathsConfig

    def config_hash(self) -> str:
        payload = {
            "dataset": self.dataset.model_dump(mode="json"),
            "preparation": self.preparation.model_dump(mode="json"),
        }
        return sha256_json(payload)[:16]

    def pipeline_config(self) -> ComplexityConfig:
        """Adapter onto the shared pipeline config used by download/loading/EmbeddingJob."""
        return ComplexityConfig(
            dataset=DatasetConfig(
                repo_id=self.dataset.repo_id,
                revision=self.dataset.revision,
                data_files=self.dataset.data_files,
                text_field=self.dataset.text_field,
                label_field=self.dataset.label_field,
                confidence_field=None,
                license=self.dataset.license,
            ),
            embeddings=self.embeddings,
            paths=PathsConfig(
                raw_dir=self.paths.raw_dir,
                processed_dir=self.paths.processed_dir,
                embeddings_dir=self.paths.embeddings_dir,
                models_dir=self.paths.embeddings_dir.parent.parent / "models" / "router_dataset",
                reports_dir=self.paths.reports_dir,
            ),
        )


def load_router_config(path: str | Path = DEFAULT_CONFIG) -> RouterDatasetConfig:
    path = Path(path).resolve()
    data: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = path.parent.parent
    data["paths"] = {k: str((root / v).resolve()) for k, v in (data.get("paths") or {}).items()}
    return RouterDatasetConfig.model_validate(data)


# ---------------------------------------------------------------- labels


def parse_card_label_names(card_text: str, label_field: str = "label") -> dict[int, str]:
    """Class-label names declared in the dataset card's YAML front matter ({int: name})."""
    parts = card_text.split("---")
    if len(parts) < 3:
        raise RouterDatasetError("dataset card has no YAML front matter")
    front = yaml.safe_load(parts[1]) or {}
    features = (front.get("dataset_info") or {}).get("features") or []
    for feature in features:
        if feature.get("name") == label_field:
            names = ((feature.get("dtype") or {}).get("class_label") or {}).get("names")
            if not isinstance(names, dict) or not names:
                break
            return {int(k): str(v) for k, v in names.items()}
    raise RouterDatasetError(f"dataset card declares no class-label names for '{label_field}'")


def build_label_mapping(card_names: dict[int, str], observed: set[Any]) -> dict[str, Any]:
    """Stable mapping that keeps the source integer IDs; every observed label must be declared."""
    bad = sorted(repr(v) for v in observed if isinstance(v, bool) or not isinstance(v, int))
    if bad:
        raise RouterDatasetError(f"labels must be integers, found {bad[:5]}")
    undeclared = sorted(v for v in observed if v not in card_names)
    if undeclared:
        raise RouterDatasetError(f"labels {undeclared} are not declared in the dataset card")
    unused = sorted(set(card_names) - observed)
    names = [card_names[i] for i in sorted(card_names)]
    if len(set(names)) != len(names):
        raise RouterDatasetError("dataset card declares duplicate label names")
    mapping = {
        "label_to_id": {card_names[i]: i for i in sorted(card_names)},
        "id_to_label": {str(i): card_names[i] for i in sorted(card_names)},
        "observed_ids": sorted(observed),
        "declared_but_unobserved": unused,
        "source": "dataset card dataset_info.features[label].dtype.class_label.names",
        "semantics": "routing labels as published; assignment method undocumented by the "
        "source; not verified complexity scores",
    }
    mapping["mapping_hash"] = sha256_json(mapping["label_to_id"])[:16]
    return mapping


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- loader


@dataclass(frozen=True)
class RouterData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_validation: np.ndarray
    y_validation: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    example_ids: dict[str, tuple[str, ...]]
    label_names: tuple[str, ...]  # index = class ID
    label_mapping: dict[str, Any]
    embedding_metadata: dict[str, Any]
    split_metadata: dict[str, Any]
    embeddings_dir: Path
    checks: dict[str, Any] = field(default_factory=dict)

    @property
    def n_classes(self) -> int:
        return len(self.label_names)


def resolve_router_embeddings_dir(cfg: RouterDatasetConfig, root: str | Path | None) -> Path:
    """`root` may be a cache directory or the directory holding `latest.json`."""
    base = Path(root) if root is not None else cfg.paths.embeddings_dir
    if (base / "metadata.json").exists():
        return base
    pointer = base / "latest.json"
    if not pointer.exists():
        raise FileNotFoundError(f"no embedding cache at {base}; generate router embeddings first")
    return base / str(json.loads(pointer.read_text(encoding="utf-8"))["cache_key"])


def _finite(X: np.ndarray, chunk_rows: int = 4096) -> bool:
    return all(bool(np.isfinite(X[i : i + chunk_rows]).all()) for i in range(0, len(X), chunk_rows))


def load_router_dataset(
    root: str | Path | None = None,
    *,
    config: str | Path | RouterDatasetConfig = DEFAULT_CONFIG,
    mmap: bool = True,
) -> RouterData:
    """Load embeddings + labels for all splits after validating every artifact link."""
    cfg = config if isinstance(config, RouterDatasetConfig) else load_router_config(config)
    emb_dir = resolve_router_embeddings_dir(cfg, root)
    processed = cfg.paths.processed_dir
    meta = json.loads((emb_dir / "metadata.json").read_text(encoding="utf-8"))
    split_meta = json.loads((processed / MANIFEST_NAME).read_text(encoding="utf-8"))
    label_mapping = json.loads((processed / "label_mapping.json").read_text(encoding="utf-8"))
    problems: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            problems.append(message)

    check(bool(meta.get("complete")), "embedding cache is incomplete")
    check(
        meta.get("processed_fingerprint") == split_meta.get("processed_fingerprint"),
        "embeddings were generated for a different processed dataset (fingerprint mismatch)",
    )
    check(
        label_mapping.get("mapping_hash") == split_meta.get("label_mapping_hash"),
        "label_mapping.json does not match the split metadata",
    )
    check(
        bool(split_meta.get("split_isolation", {}).get("verified")), "split isolation not verified"
    )
    if problems:
        raise RouterDatasetError("; ".join(problems))

    label_to_id: dict[str, int] = label_mapping["label_to_id"]
    label_names = tuple(label_mapping["id_to_label"][str(i)] for i in range(len(label_to_id)))
    dim = int(meta["dimension"])
    arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    ids: dict[str, tuple[str, ...]] = {}
    groups: dict[str, set[str]] = {}
    prompts: dict[str, set[str]] = {}
    for split in SPLITS:
        X = np.load(emb_dir / f"{split}.npy", mmap_mode="r" if mmap else None)
        with (emb_dir / f"{split}.row_mapping.jsonl").open(encoding="utf-8") as fh:
            mapping = [json.loads(line) for line in fh if line.strip()]
        with (processed / f"{split}.jsonl").open(encoding="utf-8") as fh:
            records = [json.loads(line) for line in fh if line.strip()]
        failed = set(meta.get("failed_examples", {}).get(split, {}))
        expected = [r for r in records if r["example_id"] not in failed]
        if X.dtype != np.float32 or X.ndim != 2 or X.shape != (len(mapping), dim):
            raise RouterDatasetError(
                f"{split}: array {X.shape} {X.dtype} != ({len(mapping)}, {dim}) float32"
            )
        if [m["row"] for m in mapping] != list(range(len(mapping))):
            raise RouterDatasetError(f"{split}: row mapping is not 0..n-1")
        if [m["example_id"] for m in mapping] != [r["example_id"] for r in expected]:
            raise RouterDatasetError(f"{split}: embedding rows do not align with processed records")
        for m, r in zip(mapping, expected, strict=True):
            if m["label"] != r["label"] or label_to_id.get(r["label"]) != r["label_id"]:
                raise RouterDatasetError(f"{split}: label mismatch for {r['example_id']}")
            if m["query_sha256"] != prompt_hash(r["prompt"])[:16]:
                raise RouterDatasetError(f"{split}: prompt hash mismatch for {r['example_id']}")
        if not _finite(X):
            raise RouterDatasetError(f"{split}: non-finite embedding values")
        y = np.asarray([r["label_id"] for r in expected], dtype=np.int64)
        arrays[split] = (X, y)
        ids[split] = tuple(r["example_id"] for r in expected)
        groups[split] = {r["group_id"] for r in records}
        prompts[split] = {r["prompt_key"] for r in records}

    for a_index, a in enumerate(SPLITS):
        for b in SPLITS[a_index + 1 :]:
            for name, sets in (
                ("example IDs", {k: set(v) for k, v in ids.items()}),
                ("group IDs", groups),
                ("normalised prompts", prompts),
            ):
                overlap = sets[a] & sets[b]
                if overlap:
                    raise RouterDatasetError(f"{len(overlap)} {name} shared by {a} and {b}")

    return RouterData(
        X_train=arrays["train"][0],
        y_train=arrays["train"][1],
        X_validation=arrays["validation"][0],
        y_validation=arrays["validation"][1],
        X_test=arrays["test"][0],
        y_test=arrays["test"][1],
        example_ids=ids,
        label_names=label_names,
        label_mapping=label_mapping,
        embedding_metadata=meta,
        split_metadata=split_meta,
        embeddings_dir=emb_dir,
        checks={
            "dimension": dim,
            "rows": {s: int(arrays[s][0].shape[0]) for s in SPLITS},
            "fingerprint": meta["processed_fingerprint"],
            "embedding_model": meta["embedding_model"],
            "split_isolation": "example IDs, group IDs, normalised prompts disjoint",
            "alignment": "row order, labels, label IDs, and prompt hashes verified",
            "finite": True,
        },
    )
