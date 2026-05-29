"""Retrieval "lego blocks" — composable strategies behind one interface.

A :class:`Retriever` turns a natural-language question into a *ranked
list of page slugs*. That single shape lets us compare heterogeneous
strategies (keyword, BM25, keyword+rerank, semantic, hybrid) on one
metric, and lets a config describe a composition without code.

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
import sqlite3
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from outmem.config import (
    DEFAULT_OPTIMIZE_MAX_CANDIDATES,
    DEFAULT_OPTIMIZE_MAX_RELEVANT,
    DEFAULT_OPTIMIZE_RRF_K,
    DEFAULT_OPTIMIZE_SEMANTIC_TOP_K,
    DEFAULT_OPTIMIZE_STRATEGY,
    DEFAULT_RELEVANCE_MODEL,
)
from outmem.exceptions import OutmemError
from outmem.relevance import relevance_filter
from outmem.slug import PAGES_DIR, relpath_to_slug

if TYPE_CHECKING:
    from outmem.store import WikiStore


@dataclass(frozen=True)
class RetrievalResult:
    """A retriever's answer: page slugs, most-relevant first. Empty == abstain."""

    slugs: tuple[str, ...]
    note: str | None = None  # optional diagnostic, e.g. a rerank fallback reason


@runtime_checkable
class Retriever(Protocol):
    """The lego-block contract. ``name`` labels it in scorecards/traces."""

    name: str

    def retrieve(self, question: str, *, k: int) -> RetrievalResult: ...


# --- the (small, honest) search space ------------------------------------

_STRATEGIES = ("lexical", "bm25", "rerank", "semantic", "hybrid")


@dataclass(frozen=True)
class RetrievalConfig:
    """One point in the search space — a lego composition the agent proposes.

    ``strategy`` picks the pipeline; the rest are knobs. Kept flat and
    JSON-round-trippable so the optimizer agent can emit/ingest it as a
    plain dict.
    """

    strategy: str = DEFAULT_OPTIMIZE_STRATEGY  # lexical | rerank | semantic | hybrid
    case_insensitive: bool = True
    max_candidates: int = DEFAULT_OPTIMIZE_MAX_CANDIDATES  # keyword net width
    rerank_model: str = DEFAULT_RELEVANCE_MODEL
    max_relevant: int = DEFAULT_OPTIMIZE_MAX_RELEVANT
    semantic_top_k: int = DEFAULT_OPTIMIZE_SEMANTIC_TOP_K  # semantic / hybrid blocks
    rrf_k: int = DEFAULT_OPTIMIZE_RRF_K  # Reciprocal Rank Fusion (hybrid block)

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "case_insensitive": self.case_insensitive,
            "max_candidates": self.max_candidates,
            "rerank_model": self.rerank_model,
            "max_relevant": self.max_relevant,
            "semantic_top_k": self.semantic_top_k,
            "rrf_k": self.rrf_k,
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
            cfg = replace(cfg, case_insensitive=_as_bool(data["case_insensitive"]))
        if "max_candidates" in data:
            cfg = replace(cfg, max_candidates=_as_int(data["max_candidates"], "max_candidates"))
        if "max_relevant" in data:
            cfg = replace(cfg, max_relevant=_as_int(data["max_relevant"], "max_relevant"))
        if "semantic_top_k" in data:
            cfg = replace(cfg, semantic_top_k=_as_int(data["semantic_top_k"], "semantic_top_k"))
        if "rrf_k" in data:
            cfg = replace(cfg, rrf_k=_as_int(data["rrf_k"], "rrf_k"))
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
    if config.strategy == "bm25":
        return BM25Retriever(store)
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
    if config.strategy == "hybrid":
        return HybridRetriever(
            store,
            top_k=config.semantic_top_k,
            case_insensitive=config.case_insensitive,
            rrf_k=config.rrf_k,
        )
    raise OutmemError(f"unknown strategy {config.strategy!r}")


# --- concrete blocks (wrapping outmem's existing retrieval) ----------------


class LexicalRetriever:
    """Keyword ripgrep over the wiki, slugs ranked by hit frequency.

    The cheapest baseline: formulate keywords from the question, search,
    rank pages by how many lines matched. No model call, no index. (For
    proper IDF-weighted term ranking, see :class:`BM25Retriever`.)
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


class BM25Retriever:
    """Proper BM25 ranking over page bodies via SQLite FTS5.

    Builds an in-memory FTS5 table (slug + body) from every wiki page on
    construction and ranks matches with SQLite's built-in ``bm25()``
    function (IDF-weighted term scoring — better on jargon-heavy corpora
    than the frequency-rank ``lexical`` baseline). No extra dependency:
    FTS5 ships with standard SQLite; no embedding model, no API call, no
    on-disk index. The table is ephemeral — rebuilt per retriever, which
    is fine for a one-shot tuning run over a fixed corpus.

    The NL question becomes an ``OR`` of its keyword terms (the same
    extraction the lexical block uses), so a partial match still scores.
    """

    name = "bm25"

    def __init__(self, store: WikiStore) -> None:
        # Build the index once into a plain in-memory list of (slug, body).
        # The FTS5 table is built per-call on a thread-local connection,
        # because evaluate() queries retrievers across a thread pool and a
        # single sqlite3.Connection cannot be shared across threads.
        self._rows = _read_page_rows(store)
        self._local = threading.local()

    def _conn(self) -> sqlite3.Connection:
        con = getattr(self._local, "con", None)
        if con is None:
            con = _fts5_from_rows(self._rows)
            self._local.con = con
        return con

    def retrieve(self, question: str, *, k: int) -> RetrievalResult:
        terms = [t for t in _keywords(question).split("|") if t]
        if not terms:
            return RetrievalResult(())
        # FTS5 MATCH query: quote each term (so it can't be read as syntax)
        # and OR them. bm25() returns a score where MORE negative = better,
        # so ascending order is most-relevant-first.
        match = " OR ".join(f'"{t}"' for t in terms)
        rows = self._conn().execute(
            "SELECT slug FROM pages WHERE pages MATCH ? ORDER BY bm25(pages) LIMIT ?",
            (match, k),
        ).fetchall()
        return RetrievalResult(tuple(r[0] for r in rows))


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
        max_candidates: int = DEFAULT_OPTIMIZE_MAX_CANDIDATES,
        max_relevant: int = DEFAULT_OPTIMIZE_MAX_RELEVANT,
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
        note = f"rerank fell back to lexical: {outcome.error}" if outcome.fell_back else None
        return RetrievalResult(tuple(p.slug for p in outcome.kept)[:k], note=note)


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

    def __init__(
        self, store: WikiStore, *, top_k: int = DEFAULT_OPTIMIZE_SEMANTIC_TOP_K
    ) -> None:
        self._store = store
        self._top_k = top_k

    def retrieve(self, question: str, *, k: int) -> RetrievalResult:
        if not self._store.semantic_enabled():
            raise OutmemError(
                "semantic retrieval needs `semantic.enabled: true` in "
                "config.yaml (+ `pip install outmem[semantic]`)"
            )
        # Enabled but never indexed → fail loud rather than silently return
        # nothing (which would look like a useless retriever, not a setup gap).
        try:
            empty = self._store.semantic_index_is_empty()
        except OutmemError:
            raise
        except Exception as exc:  # missing extra / db open error
            raise OutmemError(f"semantic index unavailable: {exc}") from exc
        if empty:
            raise OutmemError(
                "semantic index is empty — run `outmem reindex` to build it "
                "before tuning with the semantic/hybrid strategies"
            )
        # Chunks → pages: over-fetch chunks so dedup-to-pages still yields k.
        chunk_k = max(self._top_k, k) * 4
        try:
            matches = self._store.semantic_find_similar(question, top_k=chunk_k)
        except OutmemError:
            raise
        except Exception as exc:  # embedder / query error
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


class HybridRetriever:
    """Reciprocal Rank Fusion of the ``lexical`` and ``semantic`` blocks.

    Runs both, then fuses their ranked lists by RRF — each page scores
    ``sum 1 / (rrf_k + rank)`` across the lists it appears in, so a page
    ranked highly by *either* signal surfaces, and one ranked by *both*
    surfaces strongest. Often the best single strategy: keyword precision
    plus semantic recall.

    Requires the semantic block to be available (``semantic.enabled`` +
    a built index); otherwise :meth:`retrieve` raises :class:`OutmemError`
    so the optimizer marks the config unavailable rather than scoring a
    lexical-only result under a misleading ``hybrid`` label.
    """

    name = "hybrid"

    def __init__(
        self,
        store: WikiStore,
        *,
        top_k: int = DEFAULT_OPTIMIZE_SEMANTIC_TOP_K,
        case_insensitive: bool = True,
        rrf_k: int = DEFAULT_OPTIMIZE_RRF_K,
    ) -> None:
        self._lexical = LexicalRetriever(store, case_insensitive=case_insensitive)
        self._semantic = SemanticRetriever(store, top_k=top_k)
        self._rrf_k = rrf_k

    def retrieve(self, question: str, *, k: int) -> RetrievalResult:
        depth = max(k, 10)  # fuse a little deeper than the cutoff
        # Semantic must be available — let its OutmemError propagate so the
        # optimizer skips hybrid rather than scoring it as lexical-only.
        semantic = self._semantic.retrieve(question, k=depth).slugs
        lexical = self._lexical.retrieve(question, k=depth).slugs
        return RetrievalResult(
            _reciprocal_rank_fusion([lexical, semantic], self._rrf_k)[:k]
        )


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


def _read_page_rows(store: WikiStore) -> list[tuple[str, str]]:
    """``(slug, body)`` for every readable wiki page. Built once; the FTS5
    table is created from it per thread (see :class:`BM25Retriever`)."""
    rows: list[tuple[str, str]] = []
    for slug in store.list_slugs():
        try:
            rows.append((slug, store.read(slug).body))
        except OutmemError:
            continue  # skip an unreadable page rather than abort the index
    return rows


def _fts5_from_rows(rows: list[tuple[str, str]]) -> sqlite3.Connection:
    """A fresh in-memory FTS5 table populated from ``rows``.

    ``slug`` is UNINDEXED (stored, not searched) so MATCH/bm25 score only
    the body. Raises :class:`OutmemError` if this SQLite build lacks FTS5.
    One connection per caller thread — sqlite3 connections aren't sharable
    across threads, and ``evaluate`` queries across a thread pool.
    """
    con = sqlite3.connect(":memory:")
    try:
        con.execute("CREATE VIRTUAL TABLE pages USING fts5(slug UNINDEXED, body)")
    except sqlite3.OperationalError as exc:  # FTS5 not compiled in
        con.close()
        raise OutmemError(
            f"bm25 block needs SQLite FTS5, unavailable in this build: {exc}"
        ) from exc
    con.executemany("INSERT INTO pages (slug, body) VALUES (?, ?)", rows)
    con.commit()
    return con


def _as_bool(value: Any) -> bool:
    """Lenient bool coercion. ``bool("false")`` is ``True`` in Python — a
    footgun for hand-authored/JSON configs — so treat strings explicitly."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "on"}
    return bool(value)


def _as_int(value: Any, field: str) -> int:
    """Int coercion that fails as :class:`OutmemError` (honouring the
    'lenient parse' contract) instead of a bare ``ValueError`` traceback."""
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise OutmemError(f"{field} must be an integer, got {value!r}") from exc


def _reciprocal_rank_fusion(
    ranked_lists: list[tuple[str, ...]], rrf_k: int
) -> tuple[str, ...]:
    """Fuse ranked slug lists by RRF; ties keep first-contributed order."""
    scores: dict[str, float] = {}
    for slugs in ranked_lists:
        for rank, slug in enumerate(slugs):
            scores[slug] = scores.get(slug, 0.0) + 1.0 / (rrf_k + rank + 1)
    # sorted() is stable, so equal-score slugs retain dict insertion order.
    return tuple(sorted(scores, key=lambda s: -scores[s]))


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
