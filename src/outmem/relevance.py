"""The LLM relevance gate used by the ``rerank`` retrieval strategy.

:func:`judge_relevance` takes a query and a list of ``(slug, excerpt)``
candidates and asks a cheap model (e.g. Haiku) a single yes/no per
candidate — "is this page relevant to the query?" — keeping only the
yes's. It's a **filter, not a ranker**: no score, no ordering claim.

The load-bearing invariant — **no LLM emits wiki content here.** The
model *consumes* deterministic excerpts (assembled by the caller from
disk reads) and *emits only decisions*: ``{slug, one-line reason}``.

Reliability: the model may return only slugs from the candidate list (an
invented slug is dropped); it may return empty (a weak match is not
laundered into a false positive); and any model error/timeout/malformed
output falls back to the input slug order, so retrieval never gets worse
because of a gate failure.

The pydantic_ai import is lazy so the core library has no hard dependency
on the optional ``agent`` extra.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from outmem.config import ANTHROPIC_CACHE_SETTINGS

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RelevantPage:
    """One page the gate judged relevant to the query."""

    slug: str
    reason: str  # one-line why-relevant (the only model-generated text)


# Haiku-friendly settings. The Anthropic cache keys are no-ops on other
# providers (silently ignored); on Anthropic they cache the system prompt
# across calls. Output is a short structured list, so a small max_tokens
# is plenty.
_RELEVANCE_MODEL_SETTINGS: dict[str, Any] = {
    **ANTHROPIC_CACHE_SETTINGS,  # no tools (structured output) → no tool-def cache
    "max_tokens": 2048,
}

_RELEVANCE_SYSTEM_PROMPT = (
    "You are a relevance filter sitting between a keyword search and an "
    "expensive downstream agent. You are given a QUERY and a list of "
    "CANDIDATE wiki pages, each with its slug and a verbatim excerpt.\n\n"
    "For each candidate decide a single yes/no question: is this page "
    "relevant to the QUERY? Return ONLY the relevant ones.\n\n"
    "Rules:\n"
    "- Use slugs EXACTLY as given. Never invent, alter, or merge slugs.\n"
    "- If NOTHING is relevant, return an empty list. A weak keyword match "
    "is not relevance — do not pass through false positives.\n"
    "- `reason` is ONE short line (≤ ~12 words) naming why the page bears "
    "on the query. Describe the page; do NOT answer the query yourself, do "
    "NOT quote more than a few words, do NOT invent content.\n"
    "- Judge relevance to THIS query, not general page quality."
)


@dataclass(frozen=True)
class _Candidate:
    slug: str
    excerpt: str  # deterministic file read — what the gate model sees


@dataclass
class _Selection:
    """Structured-output row from the gate model."""

    slug: str
    reason: str


@dataclass
class _FilterResult:
    """Wrapper output type — more portable across pydantic_ai versions
    than a bare ``list[...]`` output."""

    relevant: list[_Selection] = field(default_factory=list)


_MODEL_CACHE = threading.local()


def infer_model_cached(model: Any) -> Any:
    """Infer a pydantic_ai ``Model`` from ``model``, memoised per thread.

    Passing a model-id *string* to ``Agent(...)`` makes pydantic_ai build a
    fresh provider + ``httpx.AsyncClient`` (own SSL context + sockets) on
    every construction. The optimizer evaluates rerank/hyde strategies
    across a thread pool (``evaluate(max_concurrency=...)``), calling the
    model once per bank question — so per-call construction leaks file
    descriptors until the process hits its limit
    (``OSError: [Errno 24] Too many open files``).

    Caching the inferred ``Model`` per *thread* reuses one client across
    that thread's sequential calls, bounding live clients to the worker
    count. Per-thread (not global) is deliberate: an ``httpx.AsyncClient``
    binds to the event loop that first drives it, and ``run_sync`` uses one
    persistent loop per thread — sharing a client across threads/loops is
    unsafe. A ``Model`` instance (e.g. a test ``FunctionModel``) is
    returned unchanged, so callers that already pass an object are
    unaffected.
    """
    from pydantic_ai.models import Model, infer_model

    if isinstance(model, Model):  # already concrete (e.g. tests) — no client
        return model
    cache: dict[Any, Any] | None = getattr(_MODEL_CACHE, "by_key", None)
    if cache is None:
        cache = {}
        _MODEL_CACHE.by_key = cache
    cached = cache.get(model)
    if cached is None:
        cached = infer_model(model)
        cache[model] = cached
    return cached


def _run_filter(
    model: Any,
    query: str,
    candidates: list[_Candidate],
    max_relevant: int,
) -> list[RelevantPage]:
    """One structured model call; keep the model's chosen candidate slugs."""
    from pydantic_ai import Agent

    # `model_settings` carries provider-specific Anthropic keys
    # (anthropic_cache*) that aren't in PydanticAI's ModelSettings
    # TypedDict; splat as **kwargs so mypy doesn't try to narrow the dict.
    agent_kwargs: dict[str, Any] = {"model_settings": _RELEVANCE_MODEL_SETTINGS}
    agent: Agent[None, _FilterResult] = Agent(
        infer_model_cached(model),
        output_type=_FilterResult,
        system_prompt=_RELEVANCE_SYSTEM_PROMPT,
        **agent_kwargs,
    )
    run = agent.run_sync(_format_prompt(query, candidates))

    by_slug = {c.slug for c in candidates}
    kept: list[RelevantPage] = []
    seen: set[str] = set()
    for sel in run.output.relevant:
        slug = sel.slug.strip()
        if slug not in by_slug or slug in seen:  # invent-guard + dedup
            continue
        seen.add(slug)
        kept.append(RelevantPage(slug=slug, reason=sel.reason.strip()))
        if len(kept) >= max_relevant:
            break
    return kept


def judge_relevance(
    *,
    model: Any,
    query: str,
    candidates: Sequence[tuple[str, str]],
    max_relevant: int,
) -> tuple[tuple[str, ...], str | None]:
    """LLM yes/no relevance gate over pre-built ``(slug, excerpt)`` pairs.

    Decoupled from store/search so retrieval blocks can feed candidates
    from any source (lexical, bm25, semantic, ...). Returns
    ``(kept_slugs, error)`` where ``error`` is ``None`` on success or a
    one-line reason on fallback; on any model failure the kept set is the
    input slug order so retrieval never gets WORSE on a gate failure.
    """
    if not candidates:
        return (), None
    cand_objs = [_Candidate(slug=slug, excerpt=excerpt) for slug, excerpt in candidates]
    try:
        kept = _run_filter(model, query, cand_objs, max_relevant)
        return tuple(p.slug for p in kept), None
    except Exception as exc:  # model/timeout/validation failure → source order
        reason = _brief_error(exc)
        log.warning(
            "judge_relevance failed (%s); falling back to source order", reason
        )
        return tuple(slug for slug, _ in candidates)[:max_relevant], reason


def _format_prompt(query: str, candidates: list[_Candidate]) -> str:
    parts = [f"QUERY: {query}", "", "CANDIDATES:"]
    for c in candidates:
        parts.append(f"\n[slug: {c.slug}]\n{c.excerpt}")
    return "\n".join(parts)


def _brief_error(exc: Exception, *, limit: int = 160) -> str:
    """One-line, length-capped error summary. Model failures (e.g. an
    Anthropic content-filter refusal) can carry a multi-KB JSON body;
    this keeps the per-question fallback log to a single readable line."""
    msg = " ".join(str(exc).split())
    if len(msg) > limit:
        msg = msg[:limit] + "…"
    return f"{type(exc).__name__}: {msg}" if msg else type(exc).__name__
