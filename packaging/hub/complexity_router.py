# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26", "openai>=1.40"]
# ///
"""LLM complexity router: classify a prompt as easy / medium / hard.

The prompt is embedded with Fireworks `fireworks/qwen3-embedding-8b` (your FIREWORKS_API_KEY),
then a small classifier runs locally in NumPy (no pickles, no PyTorch, no GPU).

    uv run --script complexity_router.py classify "Design a lock-free queue"     # JSON result
    uv run --script complexity_router.py classify --model mlp "..."              # other classifier
    uv run --script complexity_router.py hook  < hook-input.json                 # Claude Code hook

Python API (from this folder):
    from complexity_router import ComplexityRouter
    ComplexityRouter.load().classify(["What is 2 + 2?"])

Environment:
    FIREWORKS_API_KEY        required for classification (never logged)
    FIREWORKS_BASE_URL       default https://api.fireworks.ai/inference/v1
    COMPLEXITY_ROUTER_MODEL  logistic_regression (default) | mlp
    COMPLEXITY_ROUTER_HOOK   set to "off" to disable the Claude Code prompt hook

Licence: weights CC BY-NC 4.0 (trained on regolo/brick-complexity-extractor). Non-commercial use.
"""

from __future__ import annotations

import json
import os
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
LABELS = ("easy", "medium", "hard")
DEFAULT_BASE_URL = "https://api.fireworks.ai/inference/v1"

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b-\u200d\u2060\ufeff]")
_TRAILING_SPACE = re.compile(r"[ \t]+$", re.MULTILINE)
_BLANK_RUNS = re.compile(r"\n{3,}")


class RouterError(Exception):
    """Configuration, credential, API, or artifact problem. Messages never contain the key."""


def normalize_text(text: str) -> str:
    """Same conservative cleaning as training: keeps code, maths, and punctuation intact."""
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL.sub("", text)
    text = _TRAILING_SPACE.sub("", text)
    text = _BLANK_RUNS.sub("\n\n", text)
    return text.strip()


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    out: np.ndarray = e / e.sum(axis=1, keepdims=True)
    return out


class _Classifier:
    """NumPy forward pass over exported weights; probabilities in LABELS order."""

    def __init__(self, path: Path) -> None:
        if not path.exists():
            raise RouterError(f"classifier weights not found: {path}")
        with np.load(path, allow_pickle=False) as z:
            self.kind = str(z["kind"])
            self.arrays = {k: z[k] for k in z.files if k not in ("kind", "labels")}
            labels = [str(x) for x in z["labels"]]
        if sorted(labels) != sorted(LABELS):
            raise RouterError(f"unexpected labels in {path.name}: {labels}")
        self.order = [labels.index(lab) for lab in LABELS]  # model column -> LABELS order
        self.dim = int(self.arrays["mean"].shape[0])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        a = self.arrays
        x = (np.asarray(X, dtype=np.float64) - a["mean"]) / a["scale"]
        if self.kind == "logistic_regression":
            logits = x @ a["coef"].T + a["intercept"]
        elif self.kind == "mlp":
            h = np.maximum(x @ a["w0"].T + a["b0"], 0.0)
            h = np.maximum(h @ a["w1"].T + a["b1"], 0.0)
            logits = h @ a["w2"].T + a["b2"]
        else:
            raise RouterError(f"unknown classifier kind {self.kind!r}")
        return _softmax(logits)[:, self.order]


@dataclass(frozen=True)
class Prediction:
    label: str
    probabilities: dict[str, float]
    recommended_model: str
    classifier: str
    embedding_model: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "probabilities": self.probabilities,
            "recommended_model": self.recommended_model,
            "classifier": self.classifier,
            "embedding_model": self.embedding_model,
        }


class ComplexityRouter:
    def __init__(
        self, config: dict[str, Any], classifier_name: str, *, timeout: float = 20.0
    ) -> None:
        self.config = config
        self.classifier_name = classifier_name
        self.embedding_model: str = config["embedding_model"]
        self.dimension = int(config["embedding_dimension"])
        self.tiers: dict[str, str] = config["recommended_models"]
        self.timeout = timeout
        files = config["classifiers"]
        if classifier_name not in files:
            raise RouterError(
                f"unknown classifier {classifier_name!r}; choose from {sorted(files)}"
            )
        self.classifier = _Classifier(ROOT / files[classifier_name])
        if self.classifier.dim != self.dimension:
            raise RouterError("classifier and embedding dimensions disagree")
        self._client: Any = None

    @classmethod
    def load(cls, classifier: str | None = None, *, timeout: float = 20.0) -> ComplexityRouter:
        config = json.loads((ROOT / "router_config.json").read_text(encoding="utf-8"))
        name = (
            classifier or os.environ.get("COMPLEXITY_ROUTER_MODEL") or config["default_classifier"]
        )
        return cls(config, name, timeout=timeout)

    def _embed(self, texts: list[str]) -> np.ndarray:
        if self._client is None:
            key = os.environ.get("FIREWORKS_API_KEY", "").strip()
            if not key:
                raise RouterError("FIREWORKS_API_KEY is not set")
            configured = os.environ.get("FIREWORKS_EMBEDDING_MODEL", "").strip()
            if configured and configured != self.embedding_model:
                raise RouterError(
                    f"FIREWORKS_EMBEDDING_MODEL={configured!r}, but this classifier needs "
                    f"{self.embedding_model!r} embeddings; unset it or set it to that model"
                )
            from openai import OpenAI

            base_url = os.environ.get("FIREWORKS_BASE_URL", "").strip() or DEFAULT_BASE_URL
            self._client = OpenAI(
                api_key=key, base_url=base_url, timeout=self.timeout, max_retries=1
            )
            self._secret = key
        try:
            response = self._client.embeddings.create(
                model=self.embedding_model, input=texts, encoding_format="float"
            )
        except Exception as exc:  # report the class and status only; never the key
            status = getattr(exc, "status_code", "")
            raise RouterError(
                f"Fireworks embeddings request failed: {type(exc).__name__} {status}".strip()
            ) from None
        rows = sorted(response.data, key=lambda d: d.index)
        X = np.asarray([r.embedding for r in rows], dtype=np.float32)
        if X.shape != (len(texts), self.dimension) or not np.isfinite(X).all():
            raise RouterError(
                f"unexpected embedding shape {X.shape} (want ({len(texts)}, {self.dimension}))"
            )
        return X

    def classify(self, texts: list[str]) -> list[Prediction]:
        cleaned = [normalize_text(t) for t in texts]
        if not cleaned or not all(cleaned):
            raise RouterError("prompts must be non-empty")
        proba = self.classifier.predict_proba(self._embed(cleaned))
        out = []
        for row in proba:
            label = LABELS[int(row.argmax())]
            out.append(
                Prediction(
                    label=label,
                    probabilities={
                        lab: round(float(p), 4) for lab, p in zip(LABELS, row, strict=True)
                    },
                    recommended_model=self.tiers[label],
                    classifier=self.classifier_name,
                    embedding_model=self.embedding_model,
                )
            )
        return out


def hook_main() -> int:
    """Claude Code UserPromptSubmit hook. Always exits 0: it never blocks or alters a prompt."""
    try:
        if os.environ.get("COMPLEXITY_ROUTER_HOOK", "").lower() in ("off", "0", "false"):
            return 0
        event = json.loads(sys.stdin.read() or "{}")
        prompt = str(event.get("prompt") or "")
        if (
            event.get("source", "user") != "user"
            or len(prompt.strip()) < 3
            or prompt.lstrip().startswith("/")
        ):
            return 0
        router = ComplexityRouter.load(timeout=8.0)
        p = router.classify([prompt[:8000]])[0]
        probs = ", ".join(f"{k} {v:.2f}" for k, v in p.probabilities.items())
        context = (
            f"[llm-complexity-router] Estimated prompt complexity: {p.label.upper()} ({probs}; "
            f"{p.classifier}). Suggested Claude tier: {p.recommended_model}. This is an automated, "
            "approximate estimate: use it as a hint for how much effort the task needs and, for "
            "self-contained subtasks, whether to delegate to the haiku-worker, sonnet-worker, or "
            "opus-worker subagent. Do not mention this note unless asked."
        )
        print(
            json.dumps(
                {
                    "hookSpecificOutput": {
                        "hookEventName": "UserPromptSubmit",
                        "additionalContext": context,
                    }
                }
            )
        )
    except Exception as exc:  # fail open: never break the user's session
        print(f"llm-complexity-router hook skipped: {type(exc).__name__}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="LLM complexity router (easy / medium / hard)")
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("classify", help="classify one or more prompts; prints JSON")
    c.add_argument("prompts", nargs="+")
    c.add_argument("--model", default=None, help="logistic_regression (default) or mlp")
    sub.add_parser("hook", help="Claude Code UserPromptSubmit hook (reads JSON on stdin)")
    args = parser.parse_args(argv)
    if args.command == "hook":
        return hook_main()
    try:
        results = ComplexityRouter.load(args.model).classify(args.prompts)
    except RouterError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps([r.to_dict() for r in results], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
