"""Embedder wrapper around :class:`pydantic_ai.Embedder`.

Why a wrapper:

- We need a single place to memoise the embedder (constructing it
  inspects the provider config and validates API keys; we don't want
  that on every ``write_page`` call).
- We need a way to inject a deterministic test embedder during pytest
  without setting ``OPENAI_API_KEY``.
- Embedding dimensions get persisted to ``.vectors.db.meta`` for
  migration detection; we expose them as a property here.

The default provider is ``openai:text-embedding-3-small`` (1536 dims,
$0.02/M tokens). Any model id PydanticAI accepts works.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EmbedderHandle:
    """Cached embedder + the dimensions it produces.

    Construct via :func:`build_embedder` rather than directly.
    """

    embedder: Any  # pydantic_ai.Embedder, kept untyped so import is lazy
    model_name: str
    dimensions: int
    # Running total of input tokens billed across this handle's lifetime
    # (embeddings bill on input tokens; output is always 0). Lets reindex
    # report cost without re-instrumenting every call site.
    total_tokens: int = field(default=0)
    # Query-embedding cache. The optimizer re-asks the SAME bank questions
    # on every eval (semantic, hybrid, every config), so without this each
    # question hits the network dozens of times — the real cause of the
    # "semantic/hybrid stalls" behaviour. Keyed by exact query string;
    # guarded because retrieval runs across a thread pool.
    _query_cache: dict[str, list[float]] = field(default_factory=dict, repr=False)
    _query_lock: Any = field(default_factory=threading.Lock, repr=False)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of documents (synchronous wrapper).

        PydanticAI's API is async; the rest of outmem is sync, so we
        run the coroutine to completion here. Callers in an async
        context should call ``embed_documents_async`` directly.
        """
        return asyncio.run(self.embed_documents_async(texts))

    async def embed_documents_async(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        result = await self.embedder.embed_documents(list(texts))
        self._accrue(result)
        return [list(vec) for vec in result.embeddings]

    def _accrue(self, result: Any) -> None:
        """Add this call's billed input tokens to the running total.
        Best-effort — stub embedders and older results may lack usage."""
        usage = getattr(result, "usage", None)
        tokens = getattr(usage, "input_tokens", None) if usage is not None else None
        if isinstance(tokens, int):
            self.total_tokens += tokens

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string (sync wrapper), cached by text.

        Cache hit → no network call, no event-loop spin. This is what keeps
        the optimizer fast: it scores many configs against the same bank, so
        each distinct question is embedded once, not once per eval."""
        with self._query_lock:
            hit = self._query_cache.get(text)
        if hit is not None:
            return hit
        vec = asyncio.run(self.embed_query_async(text))
        with self._query_lock:
            self._query_cache[text] = vec
        return vec

    async def embed_query_async(self, text: str) -> list[float]:
        result = await self.embedder.embed_query(text)
        return list(result.embeddings[0])


def build_embedder(model: str | Any = "openai:text-embedding-3-small") -> EmbedderHandle:
    """Construct an :class:`EmbedderHandle` for the given model.

    ``model`` is either:

    * A real-provider string id (``"openai:text-embedding-3-small"``,
      ``"voyage:voyage-2"``, etc.) — routed through PydanticAI.
    * A pre-constructed PydanticAI
      :class:`pydantic_ai.embeddings.base.EmbeddingModel`.
    * A ``"test:…"`` stub id (``"test:bag-of-words"``, ``"test:ones"``)
      — returns a deterministic in-process :class:`EmbedderHandle`
      from :mod:`outmem.semantic.testing`. No network, no API key,
      no cost. Use these in evals and unit tests.

    Dimensions are auto-detected by calling the model on a tiny probe
    string. This adds one extra embed call at construction time but
    avoids hard-coding per-model dim tables. Stub embedders short-
    circuit the probe.
    """
    if isinstance(model, str) and model.startswith("test:"):
        from outmem.semantic.testing import STUB_BUILDERS

        try:
            builder = STUB_BUILDERS[model]
        except KeyError as exc:
            known = sorted(STUB_BUILDERS)
            raise ValueError(
                f"unknown stub embedder {model!r}; known: {known}"
            ) from exc
        return builder()

    from pydantic_ai import Embedder

    embedder = Embedder(model)
    probe = asyncio.run(embedder.embed_query("probe"))
    dimensions = len(probe.embeddings[0])
    model_name = probe.model_name
    return EmbedderHandle(embedder=embedder, model_name=model_name, dimensions=dimensions)
