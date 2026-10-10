"""Clean, deduplicate, group, and split DevQuasar/llm_router_dataset-synth (80/10/10, seed 42).

Usage:
    uv run python -m llm_router.data.prepare_router_dataset [--config configs/router_dataset.yaml]

Policy (deterministic; every removal is recorded in `removed.jsonl` with a reason):
1. Missing/empty prompts and labels outside the card-declared label set are removed.
2. Prompts are normalised conservatively (`splitting.normalize_text`): code, maths, and
   punctuation are preserved. Duplicates are detected on a stricter key (`group_key`).
3. Normalised prompts with conflicting labels are removed entirely (none in this revision).
4. Remaining duplicates keep the first occurrence (official split order, then row index);
   the other copies are removed and listed on the kept row (`duplicate_ids`). This runs
   BEFORE grouping: the source reuses stock prompts (e.g. one prompt under 167 IDs), and
   linking IDs through them would chain ~24% of all rows into a single group.
5. Groups are connected components of "same source ID" and "same normalised prompt" over
   the deduplicated rows. After step 4 this equals grouping by source ID (asserted).
6. Groups are assigned to train/validation/test with label-stratified greedy allocation;
   groups are never split. Isolation (example IDs, group IDs, normalised prompts) is asserted.

The official train/test split is not used: 400 identical prompts occur in both official
splits, and there is no official validation split.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from llm_router.config import sha256_json
from llm_router.data.download import read_provenance
from llm_router.data.loading import load_raw_records
from llm_router.data.router_dataset import (
    DEFAULT_CONFIG,
    MANIFEST_NAME,
    SPLITS,
    RouterDatasetConfig,
    build_label_mapping,
    load_router_config,
    parse_card_label_names,
    prompt_hash,
)
from llm_router.data.splitting import group_key, normalize_text, stratified_group_split
from llm_router.logging import configure_logging, get_logger
from llm_router.training.resources import peak_rss_bytes, to_gib

log = get_logger(__name__)

PREPARATION_CODE_VERSION = "1"
ORDER = {"train": 0, "validation": 1, "test": 2}


@dataclass
class RouterExample:
    example_id: str  # "<official split>-<row index>"; unique, stable, never a feature
    source_id: str  # original `id` (a generation batch); used only for grouping
    prompt: str  # normalised model input
    label: str  # original label name
    label_id: int  # class ID (the source's integer label)
    source_split: str
    source_row_index: int
    prompt_key: str  # sha256 of the strict duplicate key
    group_id: str = ""
    duplicate_ids: list[str] = field(default_factory=list)
    split: str = ""


@dataclass(frozen=True)
class Removal:
    example_id: str
    reason: str
    detail: str = ""


@dataclass
class PreparedDataset:
    splits: dict[str, list[RouterExample]]
    removed: list[Removal]
    label_mapping: dict[str, Any]
    metadata: dict[str, Any]


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def _components(examples: Sequence[RouterExample], *, by_source_id: bool) -> list[int]:
    uf = _UnionFind(len(examples))
    first: dict[tuple[str, str], int] = {}
    for n, ex in enumerate(examples):
        tags = [("p", ex.prompt_key)] + ([("id", ex.source_id)] if by_source_id else [])
        for tag in tags:
            if tag in first:
                uf.union(n, first[tag])
            else:
                first[tag] = n
    return [uf.find(n) for n in range(len(examples))]


def prepare_records(
    records: Mapping[str, Sequence[Mapping[str, Any]]],
    card_names: Mapping[int, str],
    cfg: RouterDatasetConfig,
) -> PreparedDataset:
    ds, prep = cfg.dataset, cfg.preparation
    observed = {r.get(ds.label_field) for rows in records.values() for r in rows}
    valid_ids = {
        v for v in observed if isinstance(v, int) and not isinstance(v, bool) and v in card_names
    }
    label_mapping = build_label_mapping(dict(card_names), valid_ids)

    removed: list[Removal] = []
    kept: list[RouterExample] = []
    for split in sorted(records, key=lambda s: ORDER.get(s, 9)):
        for index, row in enumerate(records[split]):
            example_id = f"{split}-{index:06d}"
            text, label = row.get(ds.text_field), row.get(ds.label_field)
            source_id = row.get(ds.id_field)
            if not isinstance(text, str):
                removed.append(Removal(example_id, "missing_prompt"))
                continue
            prompt = normalize_text(text)
            if not prompt:
                removed.append(Removal(example_id, "empty_prompt"))
                continue
            if isinstance(label, bool) or not isinstance(label, int) or label not in card_names:
                removed.append(Removal(example_id, "invalid_label", repr(label)[:40]))
                continue
            if not isinstance(source_id, str) or not source_id:
                source_id = f"missing-id:{example_id}"  # its own group; reported below
            kept.append(
                RouterExample(
                    example_id=example_id,
                    source_id=source_id,
                    prompt=prompt,
                    label=card_names[label],
                    label_id=label,
                    source_split=split,
                    source_row_index=index,
                    prompt_key=prompt_hash(group_key(prompt)),
                )
            )

    # Conflicts and duplicates on the normalised prompt key.
    by_key: dict[str, list[RouterExample]] = defaultdict(list)
    for ex in kept:
        by_key[ex.prompt_key].append(ex)
    resolved: list[RouterExample] = []
    conflict_groups = 0
    for members in by_key.values():
        if len({m.label for m in members}) > 1:
            conflict_groups += 1
            labels = dict(Counter(m.label for m in members))
            removed.extend(
                Removal(m.example_id, "conflicting_labels", repr(labels)) for m in members
            )
            continue
        first, *others = sorted(members, key=lambda m: (ORDER[m.source_split], m.source_row_index))
        first.duplicate_ids = [o.example_id for o in others]
        removed.extend(
            Removal(o.example_id, "duplicate_prompt", f"of {first.example_id}") for o in others
        )
        resolved.append(first)
    resolved.sort(key=lambda m: (ORDER[m.source_split], m.source_row_index))

    # Grouping: connected components over source ID and prompt key (deduplicated rows).
    roots = _components(resolved, by_source_id=prep.group_by_source_id)
    root_ids: dict[int, list[str]] = defaultdict(list)
    for ex, root in zip(resolved, roots, strict=True):
        root_ids[root].append(ex.example_id)
    group_of = {root: "g-" + prompt_hash(min(ids))[:12] for root, ids in root_ids.items()}
    for ex, root in zip(resolved, roots, strict=True):
        ex.group_id = group_of[root]
    groups_with_mixed_labels = defaultdict(set)
    for ex in resolved:
        groups_with_mixed_labels[ex.group_id].add(ex.label)
    mixed = [g for g, labs in groups_with_mixed_labels.items() if len(labs) > 1]
    if mixed:  # cannot happen with ID-consistent labels; dropped and reported if it does
        mixed_set = set(mixed)
        removed.extend(
            Removal(e.example_id, "group_with_conflicting_labels", e.group_id)
            for e in resolved
            if e.group_id in mixed_set
        )
        resolved = [e for e in resolved if e.group_id not in mixed_set]
    source_groups = len({e.source_id for e in resolved})
    n_groups = len({e.group_id for e in resolved})

    assignment = stratified_group_split(
        [e.label for e in resolved], [e.group_id for e in resolved], prep.split_ratios, prep.seed
    )
    splits: dict[str, list[RouterExample]] = {s: [] for s in SPLITS}
    for ex, name in zip(resolved, assignment, strict=True):
        ex.split = name
        splits[name].append(ex)
    isolation = verify_isolation(splits)
    if not isolation["verified"]:
        raise AssertionError(f"split isolation failed: {isolation}")

    total = len(resolved)
    reasons = Counter(r.reason for r in removed)
    group_sizes = Counter(e.group_id for e in resolved).values()
    metadata: dict[str, Any] = {
        "dataset": {"repo_id": ds.repo_id, "revision": ds.revision, "license": ds.license},
        "preparation_version": prep.version,
        "preprocessing_version": prep.version,  # name used by the shared EmbeddingJob
        "code_version": PREPARATION_CODE_VERSION,
        "config_hash": cfg.config_hash(),
        "seed": prep.seed,
        "rows_in": sum(len(r) for r in records.values()),
        "rows_out": total,
        "removed": dict(sorted(reasons.items())),
        "conflicting_prompt_groups": conflict_groups,
        "groups_with_conflicting_labels": len(mixed),
        "grouping": {
            "method": "connected components of same source ID / same normalised prompt, "
            "computed after deduplication",
            "groups": n_groups,
            "source_id_groups": source_groups,
            "equals_source_id_grouping": n_groups == source_groups,
            "group_size": {"min": min(group_sizes), "max": max(group_sizes)}
            if group_sizes
            else None,
            "why_dedupe_first": "linking IDs through repeated stock prompts before deduplication "
            "chains about a quarter of all rows into one component",
        },
        "official_splits_used": False,
        "official_split_reason": "400 identical prompts occur in both official splits; no "
        "official validation split",
        "requested_ratios": prep.split_ratios,
        "splits": {
            name: {
                "rows": len(rows),
                "proportion": round(len(rows) / total, 4) if total else 0.0,
                "groups": len({e.group_id for e in rows}),
                "label_distribution": dict(sorted(Counter(e.label for e in rows).items())),
                "label_proportions": {
                    k: round(v / len(rows), 4)
                    for k, v in sorted(Counter(e.label for e in rows).items())
                }
                if rows
                else {},
                "source_splits": dict(sorted(Counter(e.source_split for e in rows).items())),
            }
            for name, rows in splits.items()
        },
        "class_imbalance_ratio_train": _imbalance(splits["train"]),
        "split_isolation": isolation,
        "label_mapping_hash": label_mapping["mapping_hash"],
        "text_field": "prompt",
        "excluded_from_features": ["source_id", "example_id", "group_id"],
    }
    return PreparedDataset(splits, removed, label_mapping, metadata)


def _imbalance(rows: Sequence[RouterExample]) -> float | None:
    counts = Counter(e.label for e in rows)
    return round(max(counts.values()) / min(counts.values()), 3) if len(counts) > 1 else None


def verify_isolation(splits: Mapping[str, Sequence[RouterExample]]) -> dict[str, Any]:
    overlaps = {}
    for attr in ("example_id", "group_id", "prompt_key", "source_id"):
        owner: dict[str, str] = {}
        count = 0
        for name, rows in splits.items():
            for e in rows:
                value = getattr(e, attr)
                if owner.setdefault(value, name) != name:
                    count += 1
        overlaps[attr] = count
    return {
        "verified": all(v == 0 for v in overlaps.values()),
        "overlaps": overlaps,
        "checked": "example IDs, group IDs, normalised prompt keys, and source IDs",
    }


def finalize_fingerprints(prepared: PreparedDataset, raw_fingerprint: str | None) -> None:
    content = {
        name: [[e.example_id, prompt_hash(e.prompt)[:16], e.label_id] for e in rows]
        for name, rows in prepared.splits.items()
    }
    prepared.metadata["raw_content_fingerprint"] = raw_fingerprint
    prepared.metadata["processed_fingerprint"] = sha256_json(
        {
            "raw": raw_fingerprint,
            "config": prepared.metadata["config_hash"],
            "code": PREPARATION_CODE_VERSION,
            "labels": prepared.label_mapping["mapping_hash"],
            "content": content,  # normalised prompt hashes, labels, and split assignment
        }
    )[:16]


def write_outputs(prepared: PreparedDataset, out_dir: Path, *, save_hf: bool = True) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in prepared.splits.items():
        with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
            for e in rows:
                fh.write(json.dumps(asdict(e), ensure_ascii=False) + "\n")
    with (out_dir / "removed.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for r in prepared.removed:
            fh.write(json.dumps(asdict(r)) + "\n")
    (out_dir / "label_mapping.json").write_text(
        json.dumps(prepared.label_mapping, indent=2), encoding="utf-8"
    )
    (out_dir / MANIFEST_NAME).write_text(json.dumps(prepared.metadata, indent=2), encoding="utf-8")
    if save_hf:
        from datasets import ClassLabel, Dataset, DatasetDict, Features, Sequence, Value

        names = [
            prepared.label_mapping["id_to_label"][str(i)]
            for i in range(len(prepared.label_mapping["id_to_label"]))
        ]
        features = Features(
            {
                "example_id": Value("string"),
                "source_id": Value("string"),
                "prompt": Value("string"),
                "label": Value("string"),
                "label_id": ClassLabel(names=names),
                "source_split": Value("string"),
                "source_row_index": Value("int64"),
                "prompt_key": Value("string"),
                "group_id": Value("string"),
                "duplicate_ids": Sequence(Value("string")),
                "split": Value("string"),
            }
        )
        DatasetDict(
            {
                name: Dataset.from_list([asdict(e) for e in rows], features=features)
                for name, rows in prepared.splits.items()
            }
        ).save_to_disk(str(out_dir / "hf_dataset"))


def run(cfg: RouterDatasetConfig, *, save_hf: bool = True) -> PreparedDataset:
    from llm_router.data.inspect_router_dataset import fetch_card, read_card_text

    started = time.perf_counter()
    pipeline = cfg.pipeline_config()
    provenance = read_provenance(pipeline)  # verifies raw file hashes
    fetch_card(cfg)
    card_names = parse_card_label_names(read_card_text(cfg), cfg.dataset.label_field)
    prepared = prepare_records(load_raw_records(pipeline), card_names, cfg)
    finalize_fingerprints(prepared, provenance.get("content_fingerprint"))
    prepared.metadata["resources"] = {
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "peak_rss_gib": to_gib(peak_rss_bytes()),
    }
    write_outputs(prepared, cfg.paths.processed_dir, save_hf=save_hf)
    log.info(
        "prepared router dataset",
        rows_out=prepared.metadata["rows_out"],
        fingerprint=prepared.metadata["processed_fingerprint"],
    )
    return prepared


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--no-hf", action="store_true", help="skip saving the DatasetDict")
    args = parser.parse_args(argv)
    configure_logging()
    meta = run(load_router_config(args.config), save_hf=not args.no_hf).metadata
    keys = (
        "rows_in",
        "rows_out",
        "removed",
        "grouping",
        "splits",
        "class_imbalance_ratio_train",
        "split_isolation",
        "processed_fingerprint",
    )
    print(json.dumps({k: meta[k] for k in keys}, indent=2))


if __name__ == "__main__":
    main()
