"""Deterministic in-process embedders for testing and evals.

Two stubs are exposed:

* :class:`BagOfWordsEmbeddingModel` — hashes each token into a fixed
  number of buckets, L2-normalises. Identical text → identical
  vector; texts sharing tokens → cosine similarity > 0. Good enough
  for any case that exercises ``find_similar`` *ranking* without
  needing real semantic embeddings.
* :class:`OnesEmbeddingModel` — every input → an all-ones unit vector.
  Cheap to import (no hashlib). Useful when a test only cares that
  the embedding plumbing fires, not what it returns.

Wiring: pass ``"test:bag-of-words"`` (or ``"test:ones"``) as the
``embedding_model`` in a wiki's ``config.yaml`` and
:func:`outmem.semantic.embeddings.build_embedder` will return a
matching :class:`EmbedderHandle` without going through PydanticAI.
No API key, no network, no cost.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass

from outmem.semantic.embeddings import EmbedderHandle

DEFAULT_DIMENSIONS = 64


@dataclass
class _StubUsage:
    """Minimal stand-in for ``pydantic_ai.usage.RequestUsage`` — just the
    one field outmem reads for embedding cost."""

    input_tokens: int = 0


@dataclass
class _StubResult:
    """Shape-compatible with :class:`pydantic_ai.embeddings.EmbeddingResult`."""

    embeddings: list[list[float]]
    model_name: str
    usage: _StubUsage | None = None


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenise(text: str) -> list[str]:
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]


class BagOfWordsEmbeddingModel:
    """Hash-bucket bag-of-words embedder.

    Per-text vector: for each token, increment the bucket
    ``int(sha1(token)[:4]) % dimensions``; L2-normalise. Stable
    across runs, derivable on paper, and ranks paraphrased text
    above unrelated text — sufficient for testing ``find_similar``
    behaviour without paying for real embeddings.
    """

    def __init__(self, *, dimensions: int = DEFAULT_DIMENSIONS) -> None:
        self.dimensions = dimensions
        self.model_name = "test:bag-of-words"

    async def embed_documents(self, texts: list[str]) -> _StubResult:
        # Fake a token count (~1 per whitespace word) so cost-tracking
        # paths are exercisable in tests without a real provider.
        tokens = sum(len(t.split()) for t in texts)
        return _StubResult(
            embeddings=[self._embed(t) for t in texts],
            model_name=self.model_name,
            usage=_StubUsage(input_tokens=tokens),
        )

    async def embed_query(self, text: str) -> _StubResult:
        return _StubResult(embeddings=[self._embed(text)], model_name=self.model_name)

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dimensions
        for token in _tokenise(text):
            digest = hashlib.sha1(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dimensions
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0:
            # Empty input — deterministic non-zero vector so cosine
            # similarity stays well-defined for the caller.
            vec[0] = 1.0
            return vec
        return [v / norm for v in vec]


class OnesEmbeddingModel:
    """Trivial embedder: every input → ``[1.0, 1.0, …]``.

    Useful when a test only needs the plumbing to fire (e.g.
    "did the agent call ``find_similar`` at all?") and doesn't care
    about ranking quality. Don't use this for cases that need
    *similarity* to be meaningful — everything will look identical.
    """

    def __init__(self, *, dimensions: int = DEFAULT_DIMENSIONS) -> None:
        self.dimensions = dimensions
        self.model_name = "test:ones"

    async def embed_documents(self, texts: list[str]) -> _StubResult:
        vec = [1.0] * self.dimensions
        return _StubResult(
            embeddings=[list(vec) for _ in texts],
            model_name=self.model_name,
        )

    async def embed_query(self, text: str) -> _StubResult:
        return _StubResult(
            embeddings=[[1.0] * self.dimensions],
            model_name=self.model_name,
        )


def make_bag_of_words_handle(*, dimensions: int = DEFAULT_DIMENSIONS) -> EmbedderHandle:
    """Construct an :class:`EmbedderHandle` around :class:`BagOfWordsEmbeddingModel`."""
    model = BagOfWordsEmbeddingModel(dimensions=dimensions)
    return EmbedderHandle(
        embedder=model,
        model_name=model.model_name,
        dimensions=dimensions,
    )


def make_ones_handle(*, dimensions: int = DEFAULT_DIMENSIONS) -> EmbedderHandle:
    """Construct an :class:`EmbedderHandle` around :class:`OnesEmbeddingModel`."""
    model = OnesEmbeddingModel(dimensions=dimensions)
    return EmbedderHandle(
        embedder=model,
        model_name=model.model_name,
        dimensions=dimensions,
    )


# Mapping consulted by ``outmem.semantic.embeddings.build_embedder`` when
# the model id starts with ``test:``. Kept here (rather than in
# ``embeddings.py``) so the production module doesn't import test code.
STUB_BUILDERS = {
    "test:bag-of-words": make_bag_of_words_handle,
    "test:ones": make_ones_handle,
}
