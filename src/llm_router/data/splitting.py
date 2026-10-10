"""Text normalisation, grouping keys, and leakage-free stratified group splitting."""

from __future__ import annotations

import hashlib
import random
import re
import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b-\u200d\u2060\ufeff]")
_TRAILING_SPACE = re.compile(r"[ \t]+$", re.MULTILINE)
_BLANK_RUNS = re.compile(r"\n{3,}")
_WHITESPACE = re.compile(r"\s+")
_EDGE_PUNCT = re.compile(r"^[\s\"'`.,;:!?]+|[\s\"'`.,;:!?]+$")


def normalize_text(text: str) -> str:
    """Conservative cleaning applied to model inputs (training and inference alike).

    Keeps internal spacing, indentation, tabs, and all punctuation/symbols intact so code and
    mathematical notation survive. Only: NFC normalisation, CRLF -> LF, removal of control and
    zero-width characters, trailing spaces per line, 3+ blank lines -> 2, outer whitespace.
    """
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL.sub("", text)
    text = _TRAILING_SPACE.sub("", text)
    text = _BLANK_RUNS.sub("\n\n", text)
    return text.strip()


def group_key(text: str) -> str:
    """Aggressive key for duplicate detection only (never used as a model input).

    Case-folded NFKC text with all whitespace collapsed and leading/trailing quotes and
    punctuation removed, so "What is X?" and "what is  x" fall in the same group.
    """
    key = unicodedata.normalize("NFKC", text).casefold()
    key = _WHITESPACE.sub(" ", key)
    key = _EDGE_PUNCT.sub("", key)
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def stratified_group_split(
    labels: Sequence[str],
    groups: Sequence[str],
    ratios: Mapping[str, float],
    seed: int,
) -> list[str]:
    """Assign each row to a split so that every group lands in exactly one split.

    Groups must be label-consistent (resolve conflicts first). Within each label, groups are
    shuffled deterministically and allocated greedily so row counts follow `ratios`.
    """
    if len(labels) != len(groups):
        raise ValueError("labels and groups must have the same length")
    group_rows: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        group_rows[group].append(index)
    by_label: dict[str, list[str]] = defaultdict(list)
    for group, rows in group_rows.items():
        group_labels = {labels[i] for i in rows}
        if len(group_labels) != 1:
            raise ValueError(f"group {group[:12]} has conflicting labels {sorted(group_labels)}")
        by_label[group_labels.pop()].append(group)

    names = list(ratios)
    assignment = [""] * len(labels)
    for label in sorted(by_label):
        label_groups = sorted(by_label[label])  # canonical order before seeded shuffle
        random.Random(f"{seed}:{label}").shuffle(label_groups)
        total = sum(len(group_rows[g]) for g in label_groups)
        targets = {name: ratios[name] * total for name in names}
        filled = dict.fromkeys(names, 0)
        for group in label_groups:
            # Put the group where the remaining deficit relative to target is largest.
            name = max(names, key=lambda n: (targets[n] - filled[n], -names.index(n)))
            for row in group_rows[group]:
                assignment[row] = name
            filled[name] += len(group_rows[group])
    return assignment
