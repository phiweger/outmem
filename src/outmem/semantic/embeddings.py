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
from collections.abc import Coroutine, Sequence
from dataclasses import dataclass, field
from typing import Any


# A single, persistent background loop for every sync->async hop in this
# module. Why this exists: provider clients (OpenAI's httpx-based AsyncOpenAI,
# Voyage, Cohere, …) cache TCP connections + pool state that's bound to the
# event loop they were created on. `asyncio.run` creates a FRESH loop per
# call and CLOSES it on exit — so the cached client now points at a closed
# loop, and the next call raises `RuntimeError: Event loop is closed`
# (bubbled up as `APIConnectionError: Connection error`). One stable loop
# fixes it. The loop runs on a daemon thread so it never blocks shutdown.
_loop_lock = threading.Lock()
_loop: asyncio.AbstractEventLoop | None = None


def _shared_loop() -> asyncio.AbstractEventLoop:
    global _loop
    with _loop_lock:
        if _loop is None:
            loop = asyncio.new_event_loop()
            threading.Thread(
                target=loop.run_forever, name="outmem-embed-loop", daemon=True
            ).start()
            _loop = loop
        return _loop


def _run_sync[T](coro: Coroutine[Any, Any, T]) -> T:
    """Run ``coro`` to completion on the shared loop and return its result.

    Safe to call from any thread; raises whatever the coroutine raised.
    Replaces ``asyncio.run`` for embedder calls — provider async clients
    cache pool state tied to the loop, so we keep ONE loop alive for the
    process instead of creating-and-closing one per call.
    """
    return asyncio.run_coroutine_threadsafe(coro, _shared_loop()).result()


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
    # Per-key locks for in-flight first-time embeds (prevents thundering
    # herd: 8 concurrent first-time misses on the same text share one
    # embed call). Entries are evicted once the value lands in the cache.
    _query_pending: dict[str, Any] = field(default_factory=dict, repr=False)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of documents (synchronous wrapper).

        PydanticAI's API is async; the rest of outmem is sync, so we
        run the coroutine to completion here. Callers in an async
        context should call ``embed_documents_async`` directly.
        """
        return _run_sync(self.embed_documents_async(texts))

    async def embed_documents_async(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        # Call the EmbeddingModel.embed() interface (NOT Embedder.embed_documents)
        # — only embed() goes through InstrumentedEmbeddingModel and so emits a
        # Logfire span with model/tokens. Embedder.embed_documents falls through
        # __getattr__ to the bare model and bypasses instrumentation entirely.
        result = await self.embedder.embed(list(texts), input_type="document")
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

        Cache hit → no network call, no event-loop spin. Concurrent first-
        time misses on the SAME text share one embed call via a per-key
        lock (prevents a thundering-herd Nx over-bill on the first eval).
        """
        # Fast path: already cached → no lock contention on the hot path.
        with self._query_lock:
            hit = self._query_cache.get(text)
            if hit is not None:
                return hit
            # Get-or-create a per-text lock so 8 workers asking the same
            # question wait on each other, not the global cache lock.
            key_lock = self._query_pending.setdefault(text, threading.Lock())
        with key_lock:
            # Re-check under the per-key lock: the first arrival writes,
            # everyone else now hits the cache.
            with self._query_lock:
                hit = self._query_cache.get(text)
            if hit is not None:
                return hit
            vec = _run_sync(self.embed_query_async(text))
            with self._query_lock:
                self._query_cache[text] = vec
                # Drop the per-key lock — keeping per-text locks around
                # forever would leak one Lock per distinct query.
                self._query_pending.pop(text, None)
            return vec

    async def embed_query_async(self, text: str) -> list[float]:
        # See embed_documents_async — the EmbeddingModel.embed() entry point
        # is the one that's instrumented.
        result = await self.embedder.embed(text, input_type="query")
        self._accrue(result)  # queries are billed too — count them
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

    # Use the EmbeddingModel directly (not the Embedder convenience wrapper)
    # because the InstrumentedEmbeddingModel wrapper only intercepts embed();
    # Embedder.embed_query/embed_documents fall through __getattr__ to the
    # bare model and bypass instrumentation. EmbedderHandle therefore calls
    # `embedder.embed(...)` exclusively.
    from pydantic_ai import Embedder

    embedding_model = Embedder(model)._model
    embedding_model = _maybe_instrument(embedding_model)
    probe = _run_sync(embedding_model.embed("probe", input_type="query"))
    dimensions = len(probe.embeddings[0])
    model_name = probe.model_name
    return EmbedderHandle(
        embedder=embedding_model, model_name=model_name, dimensions=dimensions
    )


def _maybe_instrument(embedding_model: Any) -> Any:
    """Wrap ``embedding_model`` with pydantic_ai's
    :class:`InstrumentedEmbeddingModel` so every ``embed()`` call emits a
    Logfire span (model, prompt count, ``gen_ai.usage.input_tokens``) — the
    embeddings analogue of ``logfire.instrument_pydantic_ai`` which only
    covers agent/chat. Best-effort: returns the bare model if pydantic_ai
    is too old to expose the wrapper, so reindex never breaks on an
    unsupported install.
    """
    try:
        from pydantic_ai.embeddings import instrument_embedding_model
    except ImportError:
        return embedding_model
    try:
        return instrument_embedding_model(embedding_model, True)
    except Exception:  # any wrapping failure → fall back to bare model
        return embedding_model
