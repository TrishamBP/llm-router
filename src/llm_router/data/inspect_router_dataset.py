"""Inspect DevQuasar/llm_router_dataset-synth and validate its schema and labels.

Usage:
    uv run python -m llm_router.data.inspect_router_dataset [--config configs/router_dataset.yaml]

Downloads the pinned raw files and dataset card (byte-for-byte, hash-recorded) if missing,
then writes `paths.reports_dir/router_dataset_inspection.{json,md}`. Exits non-zero when the
schema or labels fail validation; preparation and embedding refuse to run in that case.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from llm_router.data.download import download, file_sha256, raw_snapshot_dir
from llm_router.data.inspect import length_stats
from llm_router.data.loading import load_raw_records
from llm_router.data.router_dataset import (
    DEFAULT_CONFIG,
    RouterDatasetConfig,
    RouterDatasetError,
    build_label_mapping,
    load_router_config,
    parse_card_label_names,
)
from llm_router.data.splitting import group_key, normalize_text
from llm_router.logging import configure_logging, get_logger

log = get_logger(__name__)

REQUIRED_FIELDS = ("id", "prompt", "label")


def fetch_card(cfg: RouterDatasetConfig) -> dict[str, Any]:
    """Copy the dataset card at the pinned revision next to the raw files; return its info."""
    target = raw_snapshot_dir(cfg.pipeline_config()) / cfg.dataset.card_file
    if not target.exists():
        from huggingface_hub import hf_hub_download

        cached = hf_hub_download(
            repo_id=cfg.dataset.repo_id,
            filename=cfg.dataset.card_file,
            repo_type="dataset",
            revision=cfg.dataset.revision,
        )
        shutil.copyfile(cached, target)
    return {"path": str(target), "sha256": file_sha256(target)}


def read_card_text(cfg: RouterDatasetConfig) -> str:
    return (raw_snapshot_dir(cfg.pipeline_config()) / cfg.dataset.card_file).read_text(
        encoding="utf-8"
    )


def inspect_records(
    records: Mapping[str, Sequence[Mapping[str, Any]]],
    card_names: Mapping[int, str],
    *,
    id_field: str = "id",
    text_field: str = "prompt",
    label_field: str = "label",
) -> dict[str, Any]:
    splits: dict[str, Any] = {}
    columns: set[str] = set()
    by_key: dict[str, list[tuple[str, int, Any, Any]]] = defaultdict(list)
    exact: dict[str, list[tuple[str, int]]] = defaultdict(list)
    ids_by_split: dict[str, set[str]] = defaultdict(set)
    labels_by_id: dict[str, set[Any]] = defaultdict(set)
    rows_by_id: Counter[str] = Counter()
    observed_labels: set[Any] = set()
    chars: list[int] = []
    words: list[int] = []

    for split, rows in records.items():
        labels: Counter[Any] = Counter()
        missing = empty = missing_id = 0
        types: dict[str, Counter[str]] = defaultdict(Counter)
        for index, row in enumerate(rows):
            columns.update(row)
            for key, value in row.items():
                types[key][type(value).__name__] += 1
            label = row.get(label_field)
            labels[label] += 1
            observed_labels.add(label)
            source_id = row.get(id_field)
            if not isinstance(source_id, str) or not source_id:
                missing_id += 1
            else:
                ids_by_split[split].add(source_id)
                labels_by_id[source_id].add(label)
                rows_by_id[source_id] += 1
            text = row.get(text_field)
            if not isinstance(text, str):
                missing += 1
                continue
            cleaned = normalize_text(text)
            if not cleaned:
                empty += 1
                continue
            exact[text].append((split, index))
            by_key[group_key(cleaned)].append((split, index, label, source_id))
            chars.append(len(cleaned))
            words.append(len(cleaned.split()))
        n = len(rows)
        splits[split] = {
            "rows": n,
            "column_types": {k: dict(v) for k, v in sorted(types.items())},
            "label_distribution": {
                str(k): {
                    "count": v,
                    "proportion": round(v / n, 4) if n else 0.0,
                    "name": card_names.get(k) if isinstance(k, int) else None,
                }
                for k, v in sorted(labels.items(), key=lambda kv: str(kv[0]))
            },
            "missing_prompt": missing,
            "empty_prompt": empty,
            "missing_id": missing_id,
            "unique_ids": len(ids_by_split[split]),
        }

    total = sum(s["rows"] for s in splits.values())
    overall = Counter(r.get(label_field) for rows in records.values() for r in rows)
    exact_dupes = {t: o for t, o in exact.items() if len(o) > 1}
    near = {k: o for k, o in by_key.items() if len(o) > 1}
    conflicts = {k: o for k, o in near.items() if len({x[2] for x in o}) > 1}
    cross_exact = sum(1 for o in exact_dupes.values() if len({s for s, _ in o}) > 1)
    cross_norm = sum(1 for o in near.values() if len({x[0] for x in o}) > 1)
    id_split_overlap = {
        f"{a}&{b}": len(ids_by_split[a] & ids_by_split[b])
        for i, a in enumerate(sorted(ids_by_split))
        for b in sorted(ids_by_split)[i + 1 :]
    }
    ids_multi_label = sorted(i for i, labs in labels_by_id.items() if len(labs) > 1)
    rows_per_id = list(rows_by_id.values())

    schema_problems = [f"missing field '{f}'" for f in REQUIRED_FIELDS if f not in columns]
    try:
        mapping = build_label_mapping(dict(card_names), observed_labels)
    except RouterDatasetError as exc:
        mapping = None
        schema_problems.append(str(exc))

    return {
        "schema_valid": not schema_problems,
        "schema_problems": schema_problems,
        "columns": sorted(columns),
        "card_label_names": {str(k): v for k, v in sorted(card_names.items())},
        "label_mapping": mapping,
        "total_rows": total,
        "splits": splits,
        "label_distribution": {
            str(k): {
                "count": v,
                "proportion": round(v / total, 4),
                "name": card_names.get(k) if isinstance(k, int) else None,
            }
            for k, v in sorted(overall.items(), key=lambda kv: str(kv[0]))
        },
        "class_imbalance_ratio": round(max(overall.values()) / min(overall.values()), 3)
        if len(overall) > 1
        else None,
        "ids": {
            "unique": len(rows_by_id),
            "rows_per_id": {
                "min": min(rows_per_id),
                "median": statistics.median(rows_per_id),
                "max": max(rows_per_id),
            }
            if rows_per_id
            else None,
            "ids_repeated": sum(1 for v in rows_per_id if v > 1),
            "ids_in_multiple_official_splits": id_split_overlap,
            "ids_with_multiple_labels": len(ids_multi_label),
            "ids_with_multiple_labels_examples": ids_multi_label[:10],
        },
        "exact_duplicate_prompts": {
            "groups": len(exact_dupes),
            "extra_rows": sum(len(o) - 1 for o in exact_dupes.values()),
            "groups_spanning_official_splits": cross_exact,
        },
        "normalized_duplicate_prompts": {
            "groups": len(near),
            "extra_rows": sum(len(o) - 1 for o in near.values()),
            "groups_spanning_official_splits": cross_norm,
            "groups_spanning_source_ids": sum(
                1 for o in near.values() if len({x[3] for x in o}) > 1
            ),
            "method": "NFKC + casefold + collapsed whitespace + stripped edge punctuation",
        },
        "conflicting_labels": {
            "groups": len(conflicts),
            "rows": sum(len(o) for o in conflicts.values()),
            "examples": [
                [{"split": s, "row": i, "label": lab, "id": sid} for s, i, lab, sid in o][:6]
                for o in list(conflicts.values())[:5]
            ],
        },
        "prompt_length_chars": length_stats(chars),
        "prompt_length_words": length_stats(words),
        "near_duplicate_note": "Embedding-space near-duplicates are audited after embedding "
        "(validate_router_artifacts); text-level near-duplicates are the normalised groups above.",
    }


def to_markdown(report: Mapping[str, Any], provenance: Mapping[str, Any]) -> str:
    ids, ex, nd, cf = (
        report["ids"],
        report["exact_duplicate_prompts"],
        report["normalized_duplicate_prompts"],
        report["conflicting_labels"],
    )
    lines = [
        f"# Inspection: {provenance['repo_id']}",
        "",
        f"- Revision `{provenance['revision']}`; "
        f"content fingerprint `{provenance['content_fingerprint']}`",
        f"- License: {provenance['license']} (source: {provenance['url']})",
        f"- Schema valid: **{report['schema_valid']}** {report['schema_problems'] or ''}",
        f"- Columns: {', '.join(report['columns'])}",
        f"- Card label names: {report['card_label_names']} (semantics undocumented by the card)",
        f"- Total rows: {report['total_rows']}; "
        f"class imbalance ratio {report['class_imbalance_ratio']}",
        "",
        "| Split | Rows | Labels (count, proportion) | Missing | Empty | Unique IDs |",
        "|---|---|---|---|---|---|",
    ]
    for split, s in report["splits"].items():
        labs = ", ".join(
            f"{v['name'] or k}: {v['count']} ({v['proportion']:.1%})"
            for k, v in s["label_distribution"].items()
        )
        lines.append(
            f"| {split} | {s['rows']} | {labs} | {s['missing_prompt']} | "
            f"{s['empty_prompt']} | {s['unique_ids']} |"
        )
    lines += [
        "",
        f"- IDs: {ids['unique']} unique; rows per ID {ids['rows_per_id']}; "
        f"IDs in several official splits {ids['ids_in_multiple_official_splits']}; "
        f"IDs with several labels: {ids['ids_with_multiple_labels']}",
        f"- Exact duplicate prompts: {ex['groups']} groups, {ex['extra_rows']} extra rows, "
        f"{ex['groups_spanning_official_splits']} groups spanning official splits",
        f"- Normalised duplicates: {nd['groups']} groups, {nd['extra_rows']} extra rows, "
        f"{nd['groups_spanning_official_splits']} spanning official splits, "
        f"{nd['groups_spanning_source_ids']} spanning several IDs",
        f"- Conflicting labels: {cf['groups']} groups covering {cf['rows']} rows",
        f"- Prompt length (chars): {report['prompt_length_chars']}",
        f"- Prompt length (words): {report['prompt_length_words']}",
        "",
    ]
    return "\n".join(lines)


def run(cfg: RouterDatasetConfig) -> dict[str, Any]:
    pipeline = cfg.pipeline_config()
    provenance = download(pipeline)
    card = fetch_card(cfg)
    card_names = parse_card_label_names(read_card_text(cfg), cfg.dataset.label_field)
    records = load_raw_records(pipeline)
    report = inspect_records(
        records,
        card_names,
        id_field=cfg.dataset.id_field,
        text_field=cfg.dataset.text_field,
        label_field=cfg.dataset.label_field,
    )
    report["provenance"] = {**provenance, "card": card}
    out = cfg.paths.reports_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "router_dataset_inspection.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (out / "router_dataset_inspection.md").write_text(
        to_markdown(report, provenance), encoding="utf-8"
    )
    log.info("wrote inspection report", schema_valid=report["schema_valid"])
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    configure_logging()
    report = run(load_router_config(args.config))
    keys = (
        "schema_valid",
        "schema_problems",
        "total_rows",
        "label_distribution",
        "ids",
        "exact_duplicate_prompts",
        "normalized_duplicate_prompts",
    )
    print(json.dumps({k: report[k] for k in keys}, indent=2))
    print(json.dumps({"conflicting_label_groups": report["conflicting_labels"]["groups"]}))
    if not report["schema_valid"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
