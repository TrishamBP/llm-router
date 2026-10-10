"""Deterministic engineered text features for the complexity classifier.

Every value is a count computed from the normalised query text (`normalize_text`), with
regular expressions only: no learned parameters, no generative model, no network, and nothing
derived from labels. The same function runs at training and inference time.

`transform` applies a fixed log1p to the raw counts (heavy-tailed lengths). Any learned scaling
(e.g. the MLP's standardisation) is fitted later, on the training split only.

Counts are heuristics. `n_tokens` uses a regex tokenizer (`\\w+|[^\\w\\s]`), not a model
tokenizer, so it approximates but does not equal the embedding model's token count. Longer or
more symbol-heavy prompts are not assumed to be harder; the classifier learns any relation.

Change `FEATURE_VERSION` when a definition changes. The schema also carries a hash of every
pattern, so a silently edited definition is detected when a saved model is loaded.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from llm_router.config import sha256_json

FEATURE_VERSION = "1"
TOKENIZER = r"regex:\w+|[^\w\s]"

_TOKEN = re.compile(r"\w+|[^\w\s]")
_CODE_FENCE = re.compile(r"^[ \t]*(?:```|~~~)", re.MULTILINE)
_INLINE_CODE = re.compile(r"(?<!`)`[^`\n]+`(?!`)")
_CODE_KEYWORD = re.compile(
    r"\b(?:def|class|return|import|lambda|elif|async|await|function|const|let|var|void|"
    r"struct|public|private|static|printf|println|SELECT|INSERT|UPDATE|WHERE|JOIN)\b"
    r"|#include\b|console\.log|System\.out"
)
_CODE_OPERATOR = re.compile(r"==|!=|<=|>=|=>|->|::|&&|\|\||\+\+|\+=|-=")
_CODE_CALL = re.compile(r"\b[A-Za-z_]\w*\([^()\n]{0,80}\)")
_CODE_LINE_END = re.compile(r"[;{}][ \t]*$", re.MULTILINE)
_MATH_SYMBOLS = re.compile(r"[∑∏∫∮∂∇√∞≈≠≡≤≥±∓×÷∈∉⊂⊆⊃∪∩∀∃∧∨¬→⇒⇔αβγδεθλμπσφωΔΣΩ^]")
_LATEX_COMMAND = re.compile(r"\\[A-Za-z]+")
_DOLLAR_MATH = re.compile(r"\$[^$\n]+\$")
_ARITHMETIC = re.compile(r"\d\s*[-+*/^=<>]\s*\(?\s*[\d.a-zA-Z]")
_INSTRUCTION_VERB = re.compile(
    r"\b(?:explain|describe|write|implement|design|build|create|develop|prove|derive|compute|"
    r"calculate|solve|compare|analy[sz]e|evaluate|list|summari[sz]e|translate|classify|"
    r"generate|optimi[sz]e|refactor|debug|fix|convert|identify|determine|find|give|provide|"
    r"outline|draft|rewrite|estimate|justify|discuss)\b",
    re.IGNORECASE,
)
_LIST_ITEM = re.compile(r"^[ \t]*(?:\d+[.)]|[-*•])[ \t]+\S", re.MULTILINE)
_JSON_KEY = re.compile(r"\"[^\"\n]{1,64}\"\s*:")


def _table_rows(text: str) -> int:
    return sum(1 for line in text.split("\n") if line.count("|") >= 2)


def _code_tokens(text: str) -> int:
    return sum(
        len(p.findall(text)) for p in (_CODE_KEYWORD, _CODE_OPERATOR, _CODE_CALL, _CODE_LINE_END)
    )


_FEATURES: dict[str, Callable[[str], int]] = {
    "n_chars": len,
    "n_words": lambda t: len(t.split()),
    "n_tokens": lambda t: len(_TOKEN.findall(t)),
    "n_lines": lambda t: t.count("\n") + 1 if t else 0,
    "n_code_fences": lambda t: len(_CODE_FENCE.findall(t)),
    "n_inline_code": lambda t: len(_INLINE_CODE.findall(t)),
    "n_code_tokens": _code_tokens,
    "n_math_symbols": lambda t: len(_MATH_SYMBOLS.findall(t)),
    "n_latex_commands": lambda t: len(_LATEX_COMMAND.findall(t)),
    "n_dollar_math": lambda t: len(_DOLLAR_MATH.findall(t)),
    "n_arithmetic_expressions": lambda t: len(_ARITHMETIC.findall(t)),
    "n_question_marks": lambda t: t.count("?"),
    "n_instruction_verbs": lambda t: len(_INSTRUCTION_VERB.findall(t)),
    "n_list_items": lambda t: len(_LIST_ITEM.findall(t)),
    "n_json_keys": lambda t: len(_JSON_KEY.findall(t)),
    "n_table_rows": _table_rows,
}
FEATURE_NAMES: tuple[str, ...] = tuple(_FEATURES)

_PATTERNS = [
    p.pattern
    for p in (
        _TOKEN,
        _CODE_FENCE,
        _INLINE_CODE,
        _CODE_KEYWORD,
        _CODE_OPERATOR,
        _CODE_CALL,
        _CODE_LINE_END,
        _MATH_SYMBOLS,
        _LATEX_COMMAND,
        _DOLLAR_MATH,
        _ARITHMETIC,
        _INSTRUCTION_VERB,
        _LIST_ITEM,
        _JSON_KEY,
    )
]


def extract_raw(texts: Sequence[str]) -> np.ndarray:
    """Raw counts, shape (n, len(FEATURE_NAMES)), float32. Texts must already be normalised."""
    out = np.zeros((len(texts), len(FEATURE_NAMES)), dtype=np.float32)
    for row, text in enumerate(texts):
        for col, fn in enumerate(_FEATURES.values()):
            out[row, col] = fn(text)
    return out


def transform(raw: np.ndarray) -> np.ndarray:
    """Fixed (not learned) log1p of the non-negative counts."""
    return np.log1p(raw, dtype=np.float32)


def extract(texts: Sequence[str]) -> np.ndarray:
    return transform(extract_raw(texts))


def schema() -> dict[str, Any]:
    payload = {
        "version": FEATURE_VERSION,
        "names": list(FEATURE_NAMES),
        "tokenizer": TOKENIZER,
        "transform": "log1p",
        "patterns": _PATTERNS,
        "table_row_rule": "line with >= 2 '|'",
    }
    return {
        "version": FEATURE_VERSION,
        "names": list(FEATURE_NAMES),
        "tokenizer": TOKENIZER,
        "transform": "log1p",
        "definition_hash": sha256_json(payload)[:16],
    }
