"""Inspect the raw dataset and write a data-quality report.

Usage: uv run python -m llm_router.data.inspect [--config configs/complexity.yaml]

Downloads the pinned snapshot first if it is missing. Writes
`paths.reports_dir/data_inspection.{json,md}`. Never modifies the raw files.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from llm_router.config import LABELS, ComplexityConfig, load_config
from llm_router.data.download import LICENSE_NOTE, download
from llm_router.data.loading import load_raw_records
from llm_router.data.splitting import group_key, normalize_text
from llm_router.logging import configure_logging, get_logger

log = get_logger(__name__)


def length_stats(values: Sequence[int]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)

    def pct(q: float) -> float:
        pos = (len(ordered) - 1) * q
        lo, hi = math.floor(pos), math.ceil(pos)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)

    return {
        "min": ordered[0],
        "p05": pct(0.05),
        "median": pct(0.5),
        "mean": round(statistics.fmean(ordered), 2),
        "p95": pct(0.95),
        "p99": pct(0.99),
        "max": ordered[-1],
    }


def inspect_records(
    records: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    text_field: str,
    label_field: str,
    confidence_field: str | None,
) -> dict[str, Any]:
    splits: dict[str, Any] = {}
    exact: dict[str, list[tuple[str, int]]] = defaultdict(list)
    grouped: dict[str, list[tuple[str, int, Any]]] = defaultdict(list)
    all_columns: set[str] = set()
    overall_labels: Counter[str] = Counter()
    char_lengths: list[int] = []
    word_lengths: list[int] = []

    for split, rows in records.items():
        labels: Counter[str] = Counter()
        missing = empty = 0
        confidences: list[float] = []
        invalid_confidence = 0
        columns: set[str] = set()
        for index, row in enumerate(rows):
            columns.update(row)
            label = row.get(label_field)
            labels[str(label)] += 1
            text = row.get(text_field)
            if not isinstance(text, str):
                missing += 1
                continue
            cleaned = normalize_text(text)
            if not cleaned:
                empty += 1
                continue
            exact[text].append((split, index))
            grouped[group_key(cleaned)].append((split, index, label))
            char_lengths.append(len(cleaned))
            word_lengths.append(len(cleaned.split()))
            if confidence_field and confidence_field in row:
                value = row[confidence_field]
                if isinstance(value, int | float) and math.isfinite(value) and 0 <= value <= 1:
                    confidences.append(float(value))
                else:
                    invalid_confidence += 1
        all_columns |= columns
        overall_labels.update(labels)
        splits[split] = {
            "rows": len(rows),
            "columns": sorted(columns),
            "label_distribution": dict(sorted(labels.items())),
            "missing_text": missing,
            "empty_text": empty,
            "confidence": (
                {
                    "present": True,
                    "distribution": dict(sorted(Counter(confidences).items())),
                    "invalid": invalid_confidence,
                }
                if confidence_field and confidence_field in columns
                else {"present": False}
            ),
        }

    exact_dupes = {t: occ for t, occ in exact.items() if len(occ) > 1}
    near_dupes = {k: occ for k, occ in grouped.items() if len(occ) > 1}
    conflicts = {k: occ for k, occ in near_dupes.items() if len({o[2] for o in occ}) > 1}
    cross_split = {k: occ for k, occ in near_dupes.items() if len({o[0] for o in occ}) > 1}
    unexpected = sorted(set(overall_labels) - set(LABELS))

    return {
        "splits": splits,
        "total_rows": sum(s["rows"] for s in splits.values()),
        "columns": sorted(all_columns),
        "expected_fields_missing": sorted(
            {text_field, label_field} - all_columns
            | ({confidence_field} - all_columns if confidence_field else set())
        ),
        "label_distribution": dict(sorted(overall_labels.items())),
        "unexpected_labels": unexpected,
        "exact_duplicates": {
            "groups": len(exact_dupes),
            "extra_rows": sum(len(o) - 1 for o in exact_dupes.values()),
        },
        "normalized_duplicates": {
            "groups": len(near_dupes),
            "extra_rows": sum(len(o) - 1 for o in near_dupes.values()),
            "groups_spanning_official_splits": len(cross_split),
        },
        "conflicting_labels": {
            "groups": len(conflicts),
            "rows": sum(len(o) for o in conflicts.values()),
            "examples": [
                [{"split": s, "index": i, "label": lab} for s, i, lab in occ]
                for occ in list(conflicts.values())[:5]
            ],
        },
        "length_chars": length_stats(char_lengths),
        "length_words": length_stats(word_lengths),
    }


def to_markdown(report: Mapping[str, Any], provenance: Mapping[str, Any]) -> str:
    lines = [
        "# Data inspection: " + provenance["repo_id"],
        "",
        f"- Revision: `{provenance['revision']}`",
        f"- Content fingerprint: `{provenance['content_fingerprint']}`",
        f"- Downloaded: {provenance['downloaded_at']}",
        f"- License: {provenance['license']}. {LICENSE_NOTE}",
        f"- Columns found: {', '.join(report['columns'])}",
        f"- Expected fields missing: {', '.join(report['expected_fields_missing']) or 'none'}",
        "",
        "| Split | Rows | easy | medium | hard | Missing | Empty | Confidence |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for split, s in report["splits"].items():
        d = s["label_distribution"]
        conf = "present" if s["confidence"]["present"] else "absent"
        lines.append(
            f"| {split} | {s['rows']} | {d.get('easy', 0)} | {d.get('medium', 0)} | "
            f"{d.get('hard', 0)} | {s['missing_text']} | {s['empty_text']} | {conf} |"
        )
    ex, nd, cf = (
        report["exact_duplicates"],
        report["normalized_duplicates"],
        report["conflicting_labels"],
    )
    lines += [
        "",
        f"- Unexpected labels: {report['unexpected_labels'] or 'none'}",
        f"- Exact duplicates: {ex['groups']} groups, {ex['extra_rows']} extra rows",
        f"- Normalised duplicates: {nd['groups']} groups, {nd['extra_rows']} extra rows, "
        f"{nd['groups_spanning_official_splits']} groups spanning official splits",
        f"- Conflicting labels: {cf['groups']} groups covering {cf['rows']} rows",
        f"- Length (chars): {report['length_chars']}",
        f"- Length (words): {report['length_words']}",
        "",
    ]
    return "\n".join(lines)


def run(cfg: ComplexityConfig) -> dict[str, Any]:
    provenance = download(cfg)
    records = load_raw_records(cfg)
    report = inspect_records(
        records,
        text_field=cfg.dataset.text_field,
        label_field=cfg.dataset.label_field,
        confidence_field=cfg.dataset.confidence_field,
    )
    report["provenance"] = provenance
    out = Path(cfg.paths.reports_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "data_inspection.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "data_inspection.md").write_text(to_markdown(report, provenance), encoding="utf-8")
    log.info("wrote inspection report", path=str(out / "data_inspection.md"))
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/complexity.yaml")
    args = parser.parse_args(argv)
    configure_logging()
    report = run(load_config(args.config))
    summary_keys = ("total_rows", "label_distribution", "expected_fields_missing")
    print(json.dumps({k: report[k] for k in summary_keys}, indent=2))
    print(
        json.dumps({k: report[k] for k in ("exact_duplicates", "normalized_duplicates")}, indent=2)
    )
    print(json.dumps({"conflicting_groups": report["conflicting_labels"]["groups"]}))


if __name__ == "__main__":
    main()
