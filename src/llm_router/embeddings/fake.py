"""Deterministic offline embedding client for tests and smoke runs. Never calls an API.

Vectors are hashed bag-of-words features: the same text always gives the same vector, and
texts sharing words get similar vectors, so a classifier can learn from them. They carry no
semantic meaning and must never be mixed with real embeddings (the model name differs).
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

import numpy as np

from llm_router.embeddings.fireworks_client import EmbeddingBatch

_TOKEN = re.compile(r"\w+")


class FakeEmbeddingClient:
    def __init__(self, dimension: int = 64, model: str = "fake/hashed-bow-64") -> None:
        self.model = model
        self.dimension = dimension
        self.calls: list[list[str]] = []

    def embed(self, texts: Sequence[str]) -> EmbeddingBatch:
        self.calls.append(list(texts))
        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in _TOKEN.findall(text.lower()):
                digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
                index = int.from_bytes(digest[:4], "little") % self.dimension
                vectors[row, index] += 1.0 if digest[4] % 2 else -1.0
            norm = np.linalg.norm(vectors[row])
            if norm > 0:
                vectors[row] /= norm
        tokens = sum(len(_TOKEN.findall(t)) for t in texts)
        return EmbeddingBatch(vectors, tokens, tokens, self.model, attempts=1, elapsed_s=0.0)
