"""Retrieval "lego blocks" — composable strategies behind one interface.

A :class:`Retriever` turns a natural-language question into a *ranked
list of page slugs*. That single shape lets us compare heterogeneous
strategies (keyword, keyword+rerank, semantic, future BM25/hybrid) on
one metric, and lets a config describe a composition without code.

Two invariants the metric leans on:

* **Ranked slugs, best first** — so ``Hit@k`` is well defined.
* **Empty == abstain** — a retriever that finds nothing returns no
  slugs, which is the *correct* behaviour for an unanswerable query
  (the abstention half of the metric, see :mod:`outmem.optimize.bench`).

Query formulation is itself part of a block: keyword-based blocks turn
the NL question into a ripgrep pattern via :func:`_keywords` (mirroring
how an agent would call ``search_wiki`` with terms, not a sentence). A
dedicated formulator could later become its own block; today it's a
shared helper.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from outmem.exceptions import OutmemError
from outmem.relevance import DEFAULT_RELEVANCE_MODEL, relevance_filter
from outmem.slug import PAGES_DIR, relpath_to_slug

if TYPE_CHECKING:
    from outmem.store import WikiStore


@dataclass(frozen=True)
class RetrievalResult:
    """A retriever's answer: page slugs, most-relevant first. Empty == abstain."""

    slugs: tuple[str, ...]


@runtime_checkable
class Retriever(Protocol):
    """The lego-block contract. ``name`` labels it in scorecards/traces."""

    name: str

    def retrieve(self, question: str, *, k: int) -> RetrievalResult: ...


# --- the (small, honest) search space ------------------------------------

_STRATEGIES = ("lexical", "rerank", "semantic")


@dataclass(frozen=True)
class RetrievalConfig:
    """One point in the search space — a lego composition the agent proposes.

    ``strategy`` picks the pipeline; the rest are knobs. Kept flat and
    JSON-round-trippable so the optimizer agent can emit/ingest it as a
    plain dict.
    """

    strategy: str = "lexical"  # "lexical" | "rerank" | "semantic"
    case_insensitive: bool = True
    max_candidates: int = 30  # width of the keyword net before rerank
    rerank_model: str = DEFAULT_RELEVANCE_MODEL
    max_relevant: int = 8
    semantic_top_k: int = 8  # used by the (stub) semantic block

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "case_insensitive": self.case_insensitive,
            "max_candidates": self.max_candidates,
            "rerank_model": self.rerank_model,
            "max_relevant": self.max_relevant,
            "semantic_top_k": self.semantic_top_k,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RetrievalConfig:
        """Lenient parse — unknown keys ignored, missing keys defaulted,
        bad ``strategy`` rejected with a clear error the agent can read."""
        cfg = cls()
        if "strategy" in data:
            strat = str(data["strategy"]).strip().lower()
            if strat not in _STRATEGIES:
                raise OutmemError(
                    f"unknown strategy {strat!r}; choose one of {_STRATEGIES}"
                )
            cfg = replace(cfg, strategy=strat)
        if "case_insensitive" in data:
            cfg = replace(cfg, case_insensitive=bool(data["case_insensitive"]))
        if "max_candidates" in data:
            cfg = replace(cfg, max_candidates=int(data["max_candidates"]))
        if "max_relevant" in data:
            cfg = replace(cfg, max_relevant=int(data["max_relevant"]))
        if "semantic_top_k" in data:
            cfg = replace(cfg, semantic_top_k=int(data["semantic_top_k"]))
        if "rerank_model" in data:
            cfg = replace(cfg, rerank_model=str(data["rerank_model"]))
        return cfg


def build_retriever(
    store: WikiStore, config: RetrievalConfig, *, model: Any = None
) -> Retriever:
    """Compose the lego blocks named by ``config`` into a live retriever.

    ``model`` overrides the rerank model object (e.g. a ``FunctionModel``
    in tests); when ``None`` the string ``config.rerank_model`` is used.
    """
    if config.strategy == "lexical":
        return LexicalRetriever(store, case_insensitive=config.case_insensitive)
    if config.strategy == "rerank":
        return RerankRetriever(
            store,
            model=model if model is not None else config.rerank_model,
            max_candidates=config.max_candidates,
            max_relevant=config.max_relevant,
            case_insensitive=config.case_insensitive,
        )
    if config.strategy == "semantic":
        return SemanticRetriever(store, top_k=config.semantic_top_k)
    raise OutmemError(f"unknown strategy {config.strategy!r}")


# --- concrete blocks (wrapping outmem's existing retrieval) ----------------


class LexicalRetriever:
    """Keyword ripgrep over the wiki, slugs ranked by hit frequency.

    The BM25-lite baseline: formulate keywords from the question, search,
    rank pages by how many lines matched. No model call.
    """

    name = "lexical"

    def __init__(self, store: WikiStore, *, case_insensitive: bool = True) -> None:
        self._store = store
        self._ci = case_insensitive

    def retrieve(self, question: str, *, k: int) -> RetrievalResult:
        pattern = _keywords(question)
        if not pattern:
            return RetrievalResult(())
        try:
            result = self._store.search(
                pattern, scope="wiki", case_insensitive=self._ci
            )
        except OutmemError:
            return RetrievalResult(())
        return RetrievalResult(_rank_by_frequency(result.hits)[:k])


class RerankRetriever:
    """Wide keyword net → cheap-model relevance gate (the relevance filter
    as a retrieval block). Returns the kept slugs in filter order; empty
    when the model judges nothing relevant (a real abstention)."""

    name = "rerank"

    def __init__(
        self,
        store: WikiStore,
        *,
        model: Any,
        max_candidates: int = 30,
        max_relevant: int = 8,
        case_insensitive: bool = True,
    ) -> None:
        self._store = store
        self._model = model
        self._max_candidates = max_candidates
        self._max_relevant = max_relevant
        self._ci = case_insensitive

    def retrieve(self, question: str, *, k: int) -> RetrievalResult:
        pattern = _keywords(question)
        if not pattern:
            return RetrievalResult(())
        outcome = relevance_filter(
            self._store,
            query=pattern,
            model=self._model,
            max_relevant=max(k, self._max_relevant),
            max_candidates=self._max_candidates,
            case_insensitive=self._ci,
        )
        return RetrievalResult(tuple(p.slug for p in outcome.kept)[:k])


class SemanticRetriever:
    """Vector-similarity block — wraps the wiki's semantic index.

    Embeds the question (via ``store.semantic_find_similar``) and returns
    the wiki *pages* whose chunks are most similar, deduped to one entry
    per page (best chunk wins) with source chunks filtered out. Empty
    when nothing clears the config similarity threshold — a real
    abstention. This is the recall tier: it surfaces pages that share no
    keywords with the query, which lexical/rerank cannot.

    Requires ``semantic.enabled`` + a built index (``outmem[semantic]``);
    raises :class:`OutmemError` otherwise, so the optimizer marks the
    config unavailable instead of crashing the loop.
    """

    name = "semantic"

    def __init__(self, store: WikiStore, *, top_k: int = 8) -> None:
        self._store = store
        self._top_k = top_k

    def retrieve(self, question: str, *, k: int) -> RetrievalResult:
        if not self._store.semantic_enabled():
            raise OutmemError(
                "semantic block needs `semantic.enabled: true` + a built "
                "index (outmem[semantic]); run `outmem reindex`"
            )
        # Chunks → pages: over-fetch chunks so dedup-to-pages still yields k.
        chunk_k = max(self._top_k, k) * 4
        try:
            matches = self._store.semantic_find_similar(question, top_k=chunk_k)
        except OutmemError:
            raise
        except Exception as exc:  # missing extra / embedder / db error
            raise OutmemError(f"semantic retrieval failed: {exc}") from exc

        prefix = f"{self._store.config.wiki_dir}/{PAGES_DIR}/"
        slugs: list[str] = []
        for match in matches:  # similarity-descending
            rel = match.rel_path
            if not rel.startswith(prefix):  # source chunk → no page slug
                continue
            slug = relpath_to_slug(Path(rel[len(prefix):]))
            if slug not in slugs:
                slugs.append(slug)
            if len(slugs) >= k:
                break
        return RetrievalResult(tuple(slugs))


# --- shared helpers --------------------------------------------------------

# Tiny stopword set — enough to stop the keyword net from being dominated
# by function words. Not linguistics; just the 80-20.
_STOP = frozenset({
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "at",
    "by", "from", "is", "are", "was", "were", "be", "been", "being", "this",
    "that", "these", "those", "it", "its", "as", "how", "what", "when", "why",
    "who", "which", "does", "do", "did", "can", "could", "should", "would",
    "about", "into", "out", "over", "under",
})


def _keywords(question: str, *, max_terms: int = 12) -> str:
    """NL question → a ripgrep alternation pattern (``term1|term2|…``).

    Lowercase, split on non-alphanumerics, drop short/stopword tokens,
    dedup, cap. Tokens are alphanumeric so they need no regex escaping.
    """
    seen: list[str] = []
    for tok in re.split(r"[^a-zA-Z0-9]+", question.lower()):
        if len(tok) >= 3 and tok not in _STOP and tok not in seen:
            seen.append(tok)
    return "|".join(seen[:max_terms])


def _rank_by_frequency(hits: tuple[Any, ...]) -> tuple[str, ...]:
    """Slugs ordered by number of matching lines (desc), ties by first-seen."""
    freq: dict[str, int] = {}
    order: list[str] = []
    for hit in hits:
        slug = relpath_to_slug(Path(hit.path))
        if slug not in freq:
            freq[slug] = 0
            order.append(slug)
        freq[slug] += 1
    ranked = sorted(order, key=lambda s: (-freq[s], order.index(s)))
    return tuple(ranked)
