"""Relevance filter — a cheap-model gate between lexical retrieval and
the expensive downstream agent.

The problem: ``search_wiki`` returns raw ripgrep lines in traversal
order, byte-capped. The expensive (outer) agent then triages them
in-context. This module moves that triage to a small model (e.g.
Haiku) and out of the expensive context, and lets the candidate net be
*wider* than what the outer agent should ever see.

It is a **filter, not a ranker**: for each candidate page the model
answers "is this relevant to the query — yes/no", and we keep the
yes's. There is no score and no ordering claim.

The load-bearing invariant — **no LLM emits wiki content here.** The
filter model *consumes* deterministic file reads (candidate excerpts
assembled by :func:`relevance_filter` reading disk) and *emits only
decisions*: ``{slug, one-line reason}``. The supporting lines handed
back are the real :class:`~outmem.search.SearchHit` rows from ripgrep;
the full page text the outer agent reads comes from ``read_page``
reading disk. The only model-generated string anywhere in the data
path is the one-line ``reason``.

Reliability guarantees (mirrors ``search_wiki``'s "(search failed: …)"
and ``find_similar``'s "unavailable" precedents):

* **Select-only** — the model may return only slugs from the candidate
  list; an invented slug is dropped.
* **May return empty** — if nothing is relevant the kept set is empty;
  a weak keyword match is not laundered into a false positive.
* **Fallback** — any model error, timeout, or malformed output yields
  ``fell_back=True`` and the candidate hits in lexical order, *trimmed
  back to the normal agent byte budget* (not the wide net). Retrieval
  never gets worse than today because of a filter failure.

The pydantic_ai import is lazy (like :func:`build_consult_wiki`) so the
core library has no hard dependency on the optional ``agent`` extra.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from outmem.config import (
    ANTHROPIC_CACHE_SETTINGS,
    DEFAULT_RELEVANCE_CANDIDATE_MAX_BYTES,
    DEFAULT_RELEVANCE_CONTEXT,
    DEFAULT_RELEVANCE_CONTEXT_CHARS,
    DEFAULT_RELEVANCE_MAX_CANDIDATES,
    DEFAULT_RELEVANCE_MAX_RELEVANT,
)
from outmem.search import DEFAULT_RESULT_BYTES, SearchHit
from outmem.slug import relpath_to_slug

if TYPE_CHECKING:
    from outmem.store import WikiStore

log = logging.getLogger(__name__)

# Defaults for RelevanceConfig live in config.py (the single home for every
# DEFAULT_* constant); imported above so the dataclass defaults and
# config.RelevanceSettings can never drift.


@dataclass(frozen=True)
class RelevantPage:
    """One page the filter judged relevant to the query."""

    slug: str
    reason: str  # one-line why-relevant (the only model-generated text)
    lines: tuple[SearchHit, ...]  # supporting lexical hits, verbatim from rg


@dataclass(frozen=True)
class FilterOutcome:
    """Result of one :func:`relevance_filter` call."""

    query: str
    kept: tuple[RelevantPage, ...]  # relevant subset, filter order (NOT ranked)
    candidates_considered: int  # distinct slugs ripgrep surfaced (pre-cap)
    model: str
    fell_back: bool  # True ⇒ filter failed; kept == bounded lexical candidates
    usage: object | None  # pydantic_ai usage if available


@dataclass(frozen=True)
class RelevanceConfig:
    """Consumer-facing config — the seam a downstream consumer wires to.

    ``model`` is required (the triage model). ``on_filter`` is the
    observability hook: it receives every :class:`FilterOutcome` so a
    consumer can record the inner triage as a first-class trace event.
    outmem stays ignorant of the consumer's tracing — it just fires the
    callback (guarded; a raising callback never breaks retrieval).
    """

    model: Any
    max_relevant: int = DEFAULT_RELEVANCE_MAX_RELEVANT
    max_candidates: int = DEFAULT_RELEVANCE_MAX_CANDIDATES
    candidate_max_bytes: int = DEFAULT_RELEVANCE_CANDIDATE_MAX_BYTES
    context: str = DEFAULT_RELEVANCE_CONTEXT  # "page" | "lines"
    context_chars_per_page: int = DEFAULT_RELEVANCE_CONTEXT_CHARS
    case_insensitive: bool = False
    on_filter: Callable[[FilterOutcome], None] | None = None


# Haiku-friendly settings. The Anthropic cache keys are no-ops on other
# providers (silently ignored); on Anthropic they cache the system
# prompt + tool defs across calls. Output is a short structured list, so
# a small max_tokens is plenty.
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
    lines: tuple[SearchHit, ...]
    excerpt: str  # deterministic file read — what the filter model sees


@dataclass
class _Selection:
    """Structured-output row from the filter model."""

    slug: str
    reason: str


@dataclass
class _FilterResult:
    """Wrapper output type — more portable across pydantic_ai versions
    than a bare ``list[...]`` output."""

    relevant: list[_Selection] = field(default_factory=list)


def relevance_filter(
    store: WikiStore,
    *,
    query: str,
    scope: str = "wiki",
    model: Any,
    max_relevant: int = DEFAULT_RELEVANCE_MAX_RELEVANT,
    max_candidates: int = DEFAULT_RELEVANCE_MAX_CANDIDATES,
    candidate_max_bytes: int = DEFAULT_RELEVANCE_CANDIDATE_MAX_BYTES,
    context: str = DEFAULT_RELEVANCE_CONTEXT,
    context_chars_per_page: int = DEFAULT_RELEVANCE_CONTEXT_CHARS,
    case_insensitive: bool = False,
) -> FilterOutcome:
    """Filter a wide lexical search down to the relevant pages.

    Phases: **gather** a wide ripgrep net (``candidate_max_bytes``,
    default 64 KiB vs the 8 KiB agent default), dedup by slug, keep the
    top ``max_candidates`` by hit frequency; **contextualise** each
    candidate by reading its page body (``context="page"``, capped at
    ``context_chars_per_page``) or its matched lines (``context="lines"``)
    — always a deterministic disk read; **filter** with one structured
    model call, mapping the kept slugs back onto the candidate hits.

    Designed for ``scope="wiki"`` (the slug-bearing scope). Other scopes
    work but key candidates by path and force ``context="lines"`` (no
    ``read_page`` for raw/log material).

    The ``store.search`` (gather) step may raise :class:`OutmemError`
    (e.g. ripgrep missing) — that propagates, exactly as ``search``
    does today. Only the model (filter) step is wrapped in the lexical
    fallback.
    """
    result = store.search(
        query,
        scope=scope,
        case_insensitive=case_insensitive,
        max_bytes=candidate_max_bytes,
    )
    candidates, distinct = _build_candidates(
        store,
        result.hits,
        scope=scope,
        max_candidates=max_candidates,
        context=context,
        context_chars_per_page=context_chars_per_page,
    )
    model_name = _model_name(model)
    if not candidates:
        return FilterOutcome(
            query=query,
            kept=(),
            candidates_considered=distinct,
            model=model_name,
            fell_back=False,
            usage=None,
        )

    try:
        kept, usage = _run_filter(model, query, candidates, max_relevant)
        return FilterOutcome(
            query=query,
            kept=tuple(kept),
            candidates_considered=distinct,
            model=model_name,
            fell_back=False,
            usage=usage,
        )
    except Exception as exc:  # any model/timeout/validation failure
        log.warning(
            "relevance filter failed (%s); falling back to lexical order", exc
        )
        return FilterOutcome(
            query=query,
            kept=tuple(_lexical_fallback(candidates)),
            candidates_considered=distinct,
            model=model_name,
            fell_back=True,
            usage=None,
        )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _candidate_key(scope: str, path: str) -> str:
    """Slug for wiki-scope hits (``abx/penicillin.md`` → ``abx:penicillin``),
    raw path otherwise."""
    if scope == "wiki":
        return relpath_to_slug(Path(path))
    return path


def _build_candidates(
    store: WikiStore,
    hits: tuple[SearchHit, ...],
    *,
    scope: str,
    max_candidates: int,
    context: str,
    context_chars_per_page: int,
) -> tuple[list[_Candidate], int]:
    """Group hits by slug, keep the top ``max_candidates`` by hit
    frequency, and attach a deterministic excerpt to each.

    Returns ``(candidates, distinct_count)`` where ``distinct_count`` is
    the true number of slugs ripgrep surfaced — the recall signal a
    curator reads off ``on_filter`` even when the net is capped.
    """
    grouped: dict[str, list[SearchHit]] = {}
    order: list[str] = []
    for hit in hits:
        key = _candidate_key(scope, hit.path)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(hit)

    distinct = len(order)
    # Top N by hit frequency, ties broken by first-seen order.
    chosen = sorted(order, key=lambda k: (-len(grouped[k]), order.index(k)))
    chosen = chosen[:max_candidates]
    # Present in slug order for determinism.
    chosen.sort()

    candidates: list[_Candidate] = []
    for key in chosen:
        page_hits = tuple(grouped[key])
        excerpt = _excerpt(
            store,
            scope=scope,
            key=key,
            hits=page_hits,
            context=context,
            context_chars_per_page=context_chars_per_page,
        )
        candidates.append(_Candidate(slug=key, lines=page_hits, excerpt=excerpt))
    return candidates, distinct


def _excerpt(
    store: WikiStore,
    *,
    scope: str,
    key: str,
    hits: tuple[SearchHit, ...],
    context: str,
    context_chars_per_page: int,
) -> str:
    """The deterministic text the filter model sees for one candidate.

    ``context="page"`` reads the page body from disk (capped); anything
    else — or a non-wiki scope, or a read failure — falls back to the
    matched lines. Never model-generated.
    """
    if context == "page" and scope == "wiki":
        try:
            body = store.read(key).body
        except Exception:  # unreadable page (e.g. non-UTF-8) → fall back to lines
            body = ""
        if body:
            return body[:context_chars_per_page]
    joined = "\n".join(h.text for h in hits)
    return joined[:context_chars_per_page]


def _run_filter(
    model: Any,
    query: str,
    candidates: list[_Candidate],
    max_relevant: int,
) -> tuple[list[RelevantPage], object | None]:
    """One structured model call; map kept slugs back onto candidate hits."""
    from pydantic_ai import Agent

    # `model_settings` carries provider-specific Anthropic keys
    # (anthropic_cache*) that aren't in PydanticAI's ModelSettings
    # TypedDict; splat as **kwargs so mypy doesn't try to narrow the
    # dict (mirrors build_consult_wiki).
    agent_kwargs: dict[str, Any] = {"model_settings": _RELEVANCE_MODEL_SETTINGS}
    agent: Agent[None, _FilterResult] = Agent(
        model,
        output_type=_FilterResult,
        system_prompt=_RELEVANCE_SYSTEM_PROMPT,
        **agent_kwargs,
    )
    run = agent.run_sync(_format_prompt(query, candidates))

    # `run.usage` is a property on current pydantic_ai (calling it is
    # deprecated). Capture it as-is for the consumer's token accounting.
    usage: object | None = getattr(run, "usage", None)

    by_slug = {c.slug: c for c in candidates}
    kept: list[RelevantPage] = []
    seen: set[str] = set()
    for sel in run.output.relevant:
        slug = sel.slug.strip()
        if slug not in by_slug or slug in seen:  # invent-guard + dedup
            continue
        seen.add(slug)
        kept.append(
            RelevantPage(
                slug=slug,
                reason=sel.reason.strip(),
                lines=by_slug[slug].lines,
            )
        )
        if len(kept) >= max_relevant:
            break
    return kept, usage


def _lexical_fallback(candidates: list[_Candidate]) -> list[RelevantPage]:
    """Filter-failure path: candidate hits in slug order, trimmed back to
    the normal agent byte budget so a failure never floods the expensive
    context with the wide net. ``reason`` is empty (no model ran)."""
    out: list[RelevantPage] = []
    consumed = 0
    for cand in sorted(candidates, key=lambda c: c.slug):
        size = sum(len(h.text.encode("utf-8")) for h in cand.lines)
        if out and consumed + size > DEFAULT_RESULT_BYTES:
            break
        out.append(RelevantPage(slug=cand.slug, reason="", lines=cand.lines))
        consumed += size
    return out


def _format_prompt(query: str, candidates: list[_Candidate]) -> str:
    parts = [f"QUERY: {query}", "", "CANDIDATES:"]
    for c in candidates:
        parts.append(f"\n[slug: {c.slug}]\n{c.excerpt}")
    return "\n".join(parts)


def _model_name(model: Any) -> str:
    if isinstance(model, str):
        return model
    name = getattr(model, "model_name", None)
    return str(name) if name else type(model).__name__
