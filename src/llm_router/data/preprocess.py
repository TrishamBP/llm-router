"""Deterministic cleaning, deduplication, and leakage-free 80/10/10 splitting.

Usage: uv run python -m llm_router.data.preprocess [--config configs/complexity.yaml]

Pipeline (all official splits are pooled and re-split):
1. Drop rows with missing/empty text or labels outside {easy, medium, hard}.
2. Validate confidence when the column exists (invalid values are reported and set to null).
3. Group rows by a normalised key (case, whitespace, edge punctuation). Groups with different
   labels are conflicts (dropped by default); same-label duplicates keep one representative.
4. Optional confidence filtering (disabled by default).
5. Stratified group split; a group can only ever appear in one split. Isolation is verified
   and recorded in the manifest, which embedding generation checks before running.

Outputs in `paths.processed_dir`: {train,validation,test}.jsonl, removed.jsonl, manifest.json.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from llm_router.config import LABELS, ComplexityConfig, load_config, sha256_json
from llm_router.data.download import read_provenance
from llm_router.data.loading import load_raw_records
from llm_router.data.splitting import group_key, normalize_text, stratified_group_split
from llm_router.logging import configure_logging, get_logger

log = get_logger(__name__)

SPLITS = ("train", "validation", "test")
PREPROCESSING_CODE_VERSION = "1"


@dataclass
class Example:
    example_id: str  # "<official split>-<row index>"; stable across runs
    query: str  # normalised text used as model input
    label: str
    original_label: str
    confidence: float | None
    source_split: str
    source_index: int
    group: str
    duplicate_ids: list[str] = field(default_factory=list)
    split: str = ""


@dataclass(frozen=True)
class Removal:
    example_id: str
    reason: str
    detail: str = ""


@dataclass
class PreprocessResult:
    splits: dict[str, list[Example]]
    removed: list[Removal]
    manifest: dict[str, Any]


def preprocess_records(
    records: Mapping[str, Sequence[Mapping[str, Any]]], cfg: ComplexityConfig
) -> PreprocessResult:
    ds, pp = cfg.dataset, cfg.preprocessing
    removed: list[Removal] = []
    kept: list[Example] = []
    invalid_confidence = 0
    has_confidence = False

    for source_split in sorted(records):
        for index, row in enumerate(records[source_split]):
            example_id = f"{source_split}-{index:06d}"
            text = row.get(ds.text_field)
            label = row.get(ds.label_field)
            if not isinstance(text, str):
                removed.append(Removal(example_id, "missing_query"))
                continue
            query = normalize_text(text)
            if not query:
                removed.append(Removal(example_id, "empty_query"))
                continue
            if label not in LABELS:
                removed.append(Removal(example_id, "invalid_label", repr(label)[:40]))
                continue
            confidence = None
            if ds.confidence_field and ds.confidence_field in row:
                has_confidence = True
                value = row[ds.confidence_field]
                if isinstance(value, int | float) and math.isfinite(value) and 0 <= value <= 1:
                    confidence = float(value)
                else:
                    invalid_confidence += 1
            kept.append(
                Example(
                    example_id=example_id,
                    query=query,
                    label=str(label),
                    original_label=str(label),
                    confidence=confidence,
                    source_split=source_split,
                    source_index=index,
                    group=group_key(query),
                )
            )

    # Duplicates and conflicts, resolved per normalised group.
    groups: dict[str, list[Example]] = defaultdict(list)
    for example in kept:
        groups[example.group].append(example)
    resolved: list[Example] = []
    conflict_groups = 0
    for members in groups.values():
        labels = Counter(m.label for m in members)
        if len(labels) > 1:
            conflict_groups += 1
            (top, top_n), *rest = labels.most_common()
            if pp.conflict_policy == "majority" and top_n > rest[0][1]:
                for m in members:
                    m.label = top
            else:
                for m in members:
                    removed.append(
                        Removal(m.example_id, "conflicting_labels", dict(labels).__repr__())
                    )
                continue
        if pp.duplicate_policy == "keep_first" and len(members) > 1:
            first, *others = sorted(members, key=lambda m: m.example_id)
            first.duplicate_ids = [o.example_id for o in others]
            removed.extend(
                Removal(o.example_id, "duplicate", f"of {first.example_id}") for o in others
            )
            resolved.append(first)
        else:
            resolved.extend(members)

    if pp.min_confidence is not None:
        if not has_confidence:
            raise ValueError("min_confidence is set but the dataset has no confidence column")
        filtered = []
        for m in resolved:
            if m.confidence is None or m.confidence < pp.min_confidence:
                removed.append(Removal(m.example_id, "below_min_confidence", str(m.confidence)))
            else:
                filtered.append(m)
        resolved = filtered

    resolved.sort(key=lambda m: m.example_id)
    assignment = stratified_group_split(
        [m.label for m in resolved], [m.group for m in resolved], pp.split_ratios, pp.seed
    )
    splits: dict[str, list[Example]] = {name: [] for name in SPLITS}
    for example, name in zip(resolved, assignment, strict=True):
        example.split = name
        splits[name].append(example)

    isolation = verify_split_isolation(splits)
    if not isolation["verified"]:
        raise RuntimeError(f"split isolation failed: {isolation}")

    provenance = read_provenance(cfg) if (cfg.paths.raw_dir / ds.revision).exists() else {}
    reasons = Counter(r.reason for r in removed)
    total_in = sum(len(rows) for rows in records.values())
    manifest = {
        "preprocessing_version": pp.version,
        "code_version": PREPROCESSING_CODE_VERSION,
        "config_hash": cfg.preprocessing_hash(),
        "raw_content_fingerprint": provenance.get("content_fingerprint"),
        "dataset": {"repo_id": ds.repo_id, "revision": ds.revision, "license": ds.license},
        "rows_in": total_in,
        "rows_out": sum(len(v) for v in splits.values()),
        "removed": dict(sorted(reasons.items())),
        "conflict_groups": conflict_groups,
        "confidence": {"present": has_confidence, "invalid": invalid_confidence},
        "split_ratios": pp.split_ratios,
        "splits": {
            name: {
                "rows": len(rows),
                "label_distribution": dict(sorted(Counter(m.label for m in rows).items())),
                "source_splits": dict(sorted(Counter(m.source_split for m in rows).items())),
            }
            for name, rows in splits.items()
        },
        "split_isolation": isolation,
    }
    manifest["processed_fingerprint"] = sha256_json(
        {
            "config": manifest["config_hash"],
            "code": PREPROCESSING_CODE_VERSION,
            "raw": manifest["raw_content_fingerprint"],
            "ids": {n: [m.example_id for m in rows] for n, rows in splits.items()},
        }
    )[:16]
    return PreprocessResult(splits, removed, manifest)


def verify_split_isolation(splits: Mapping[str, Sequence[Example]]) -> dict[str, Any]:
    seen_ids: dict[str, str] = {}
    seen_groups: dict[str, str] = {}
    id_overlaps = group_overlaps = 0
    for name, rows in splits.items():
        for m in rows:
            for key, seen in ((m.example_id, seen_ids), (m.group, seen_groups)):
                if key in seen and seen[key] != name:
                    if seen is seen_ids:
                        id_overlaps += 1
                    else:
                        group_overlaps += 1
                seen.setdefault(key, name)
    return {
        "verified": id_overlaps == 0 and group_overlaps == 0,
        "overlapping_example_ids": id_overlaps,
        "overlapping_groups": group_overlaps,
        "method": "normalised-text group key (NFKC, casefold, whitespace, edge punctuation)",
    }


def write_outputs(result: PreprocessResult, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in result.splits.items():
        with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
            for m in rows:
                fh.write(json.dumps(asdict(m), ensure_ascii=False) + "\n")
    with (out_dir / "removed.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for r in result.removed:
            fh.write(json.dumps(asdict(r)) + "\n")
    (out_dir / "manifest.json").write_text(json.dumps(result.manifest, indent=2), encoding="utf-8")


def load_processed(
    processed_dir: Path, manifest_name: str = "manifest.json"
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    manifest_path = processed_dir / manifest_name
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} not found; run `python -m llm_router.data.preprocess`"
        )
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    splits = {}
    for name in SPLITS:
        with (processed_dir / f"{name}.jsonl").open(encoding="utf-8") as fh:
            splits[name] = [json.loads(line) for line in fh]
    return splits, manifest


def run(cfg: ComplexityConfig) -> PreprocessResult:
    result = preprocess_records(load_raw_records(cfg), cfg)
    write_outputs(result, cfg.paths.processed_dir)
    log.info(
        "preprocessed",
        rows_in=result.manifest["rows_in"],
        rows_out=result.manifest["rows_out"],
        fingerprint=result.manifest["processed_fingerprint"],
    )
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/complexity.yaml")
    args = parser.parse_args(argv)
    configure_logging()
    manifest = run(load_config(args.config)).manifest
    print(
        json.dumps(
            {
                k: manifest[k]
                for k in (
                    "rows_in",
                    "rows_out",
                    "removed",
                    "splits",
                    "split_isolation",
                    "processed_fingerprint",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
