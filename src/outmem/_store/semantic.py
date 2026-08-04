"""Semantic-index operations for :class:`outmem.store.WikiStore`.

Split out of ``store.py`` for file-size hygiene. Public access is via
the ``WikiStore.semantic_*`` methods, which forward here.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from outmem.config import (
    DEFAULT_SEMANTIC_REINDEX_CONCURRENCY,
    SEMANTIC_INDEX_PAGES,
    SEMANTIC_UNAVAILABLE_HELP,
)
from outmem.exceptions import FrontmatterError, OutmemError
from outmem.index import RESERVED_WIKI_FILES, editorial_pages, load_page_text
from outmem.slug import PAGES_DIR, slug_to_relpath
from outmem.sources import REGISTRY_FILENAME, SOURCES_DIR, SOURCES_LOCAL_DIR

if TYPE_CHECKING:
    from outmem.semantic import Match, ReindexResult, VectorStore
    from outmem.store import WikiStore

log = logging.getLogger(__name__)

WikiContentKind = Literal["wiki", "source"]


def available(store: WikiStore) -> bool:
    """True if the semantic index has been built (its db file exists).

    Semantic has no on/off config flag: a wiki "has semantic" once
    ``outmem reindex`` has created the index. A cheap path check (no DB
    open / embedder probe), so it's safe to call at tool-build time to
    decide whether to expose ``find_similar``, etc."""
    return db_path(store).exists()


def index_is_empty(store: WikiStore) -> bool:
    """True if the semantic index has no indexed files yet.

    Returns ``True`` when the index hasn't been built at all, so a caller
    that probes emptiness without first checking :func:`available` does
    not accidentally *create* an empty db via ``vector_store_or_open``
    (which would flip :func:`available` permanently True with no content).
    Once the db exists, opens the vector store — the first call pays the
    one-time ``build_embedder`` probe (a tiny embed request to detect
    dimensions); the handle is then cached on the store.
    """
    if not available(store):  # don't materialise an empty db just to probe
        return True
    return len(vector_store_or_open(store).list_indexed_files()) == 0


def db_path(store: WikiStore) -> Path:
    return store.root / store.config.outmem.semantic.db_filename


def vector_store_or_open(store: WikiStore) -> VectorStore:
    """Lazy open of the :class:`VectorStore` (creating the db if missing).

    Callers that must *not* create an empty index on a read (e.g. the
    retrieval strategies, ``find_similar``) gate on :func:`available`
    first and fail loud with :data:`SEMANTIC_UNAVAILABLE_HELP`. The
    ``build_embedder`` probe is real (one API call) so we cache the
    handle.
    """
    if store._vector_store is not None:
        return store._vector_store
    # Double-checked lock: concurrent callers (the optimize thread pool)
    # must not each build an embedder + open a connection, orphaning all
    # but the last. The probe/open happens once.
    with store._vector_store_lock:
        if store._vector_store is not None:
            return store._vector_store
        # Lazy import so the optional extra is only required when used.
        from outmem.semantic import VectorStore, build_embedder

        settings = store.config.outmem.semantic
        try:
            embedder = build_embedder(settings.embedding_model)
        except OutmemError:
            raise
        except Exception as exc:
            # Surface a clean message instead of a raw provider traceback —
            # the usual cause is a missing API key for the embedding model.
            raise OutmemError(
                f"could not initialise the embedding model "
                f"{settings.embedding_model!r}: {exc}. Check the provider API "
                f"key (e.g. OPENAI_API_KEY) is set in your environment / .env."
            ) from exc
        store._vector_store = VectorStore.open(db_path(store), embedder=embedder)
        return store._vector_store


def find_similar(
    store: WikiStore,
    text: str,
    *,
    top_k: int | None = None,
    threshold: float | None = None,
    exclude_slug: str | None = None,
) -> list[Match]:
    if not available(store):  # fail loud, don't auto-create an empty index
        raise OutmemError(SEMANTIC_UNAVAILABLE_HELP)
    settings = store.config.outmem.semantic
    if top_k is None:
        top_k = settings.top_k
    if threshold is None:
        threshold = settings.similarity_threshold
    vs = vector_store_or_open(store)
    exclude_rel = (
        f"{store.config.wiki_dir}/{PAGES_DIR}/{slug_to_relpath(exclude_slug).as_posix()}"
        if exclude_slug
        else None
    )
    return vs.find_similar(
        text,
        top_k=top_k,
        threshold=threshold,
        exclude_rel_path=exclude_rel,
    )


def reindex_path(store: WikiStore, rel_path: str) -> ReindexResult | None:
    if not available(store):  # don't build an index on a write — opt in via reindex
        return None
    load = load_for_index(store, rel_path)
    if load is None:
        return None
    body, kind, header = load
    vs = vector_store_or_open(store)
    settings = store.config.outmem.semantic
    return vs.reindex_file(
        rel_path,
        body=body,
        kind=kind,
        header=header,
        chunk_size=settings.chunk_size,
        chunk_max=settings.chunk_max,
        overlap_paragraphs=settings.overlap_paragraphs,
    )


def remove_path(store: WikiStore, rel_path: str) -> int:
    if not available(store):
        return 0
    vs = vector_store_or_open(store)
    return vs.remove_file(rel_path)


def reindex_all(
    store: WikiStore,
    *,
    force: bool = False,
    max_concurrency: int = DEFAULT_SEMANTIC_REINDEX_CONCURRENCY,
    on_progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Resync the whole index with disk — the opt-in that *builds* the
    index (creating its db). Embeds files concurrently (the network
    bottleneck), at most ``max_concurrency`` in flight; writes stay serial.
    ``on_progress(done, total)`` fires as each file completes. Raises if
    the ``outmem[semantic]`` extra isn't installed.

    The returned summary carries ``dropped_paths``: wiki pages that exist
    on disk but did not make it into the index. That is the reconciliation
    between what was discovered and what was indexed — without it a page
    can silently vanish from retrieval."""
    vs = vector_store_or_open(store)

    on_disk = indexable_files_on_disk(store)
    # Load bodies (skipping non-text/reserved files). force=True drops the
    # existing entry first so reindex_files re-embeds even on a hash match.
    batch: list[tuple[str, str, WikiContentKind, str]] = []
    dropped_paths: list[str] = []
    for rel_path in on_disk:
        if force:
            vs.remove_file(rel_path)
        loaded = load_for_index(store, rel_path)
        if loaded is None:
            # Sources/reserved files skip legitimately and quietly; a *page*
            # that fails to load is data loss and load_for_index has already
            # logged why. Surface it in the summary so the caller can act.
            if store.is_page_path(rel_path):
                dropped_paths.append(rel_path)
            continue
        body, kind, header = loaded
        batch.append((rel_path, body, kind, header))

    settings = store.config.outmem.semantic
    tokens_before = getattr(vs.embedder, "total_tokens", 0)
    # One parent span so reindex shows up in the Logfire UI with its cost
    # (embeddings aren't agent calls, so instrument_pydantic_ai doesn't
    # cover them — we record the billed input tokens explicitly).
    from outmem._logfire import span as _span

    with _span("outmem.reindex", files=len(batch), force=force) as sp:
        results = vs.reindex_files(
            batch,
            chunk_size=settings.chunk_size,
            chunk_max=settings.chunk_max,
            overlap_paragraphs=settings.overlap_paragraphs,
            max_concurrency=max_concurrency,
            on_progress=on_progress,
        )
        embed_tokens = getattr(vs.embedder, "total_tokens", 0) - tokens_before
        reindexed = sum(1 for r in results if not r.skipped)
        added_chunks = sum(r.chunks_added for r in results)
        sp.set_attribute("reindexed", reindexed)
        sp.set_attribute("chunks_added", added_chunks)
        sp.set_attribute("embed_tokens", embed_tokens)
    # A file that FAILED (embed error) is not "unchanged" — it needed
    # indexing and isn't there. Counting it as skipped reported a clean run
    # while the page was unreachable, which is the failure this whole
    # reconciliation exists to catch.
    failed = [r for r in results if r.error]
    skipped = sum(1 for r in results if r.skipped and not r.error)
    for r in failed:
        log.warning("semantic: %s NOT indexed — %s", r.rel_path, r.error)
        if store.is_page_path(r.rel_path):
            dropped_paths.append(r.rel_path)

    removed = 0
    # Orphan sweep: anything indexed that is no longer indexable on disk.
    # Dropped pages count as orphans even though the file still exists —
    # their indexed chunks are from a previous, now-unreadable revision, and
    # leaving them in means `search_wiki` answers from stale content while
    # the caller is told the page is unreachable. Purging makes the report
    # true and matches what `--force` already does.
    stale = set(dropped_paths)
    on_disk_set = set(on_disk) - stale
    for rel_path, _, _ in vs.list_indexed_files():
        if rel_path not in on_disk_set:
            vs.remove_file(rel_path)
            removed += 1

    # Per-page reasons were already logged (by load_for_index, or by the
    # failed loop above) — no aggregate restatement here; the CLI prints
    # the actionable list and sets the exit code.
    return {
        "reindexed": reindexed,
        "skipped": skipped,
        "removed": removed,
        "chunks_added": added_chunks,
        "embed_tokens": embed_tokens,
        "dropped_paths": dropped_paths,
    }


def frontmatter_header(frontmatter: Any) -> str:
    """The ``"<title> — <tags>"`` line prepended to every chunk of a page.

    Answers the retrieval problem that a page's own title and tags are not
    in its body: ``parse_wiki_page`` splits them off, so without this the
    entire tag vocabulary is invisible to the embedder and a page whose
    body never repeats its own title is unretrievable by that title.
    Opt-in via ``semantic.embed_frontmatter``.
    """
    title = (frontmatter.title or "").strip()
    tags = [t.strip() for t in (frontmatter.tags or []) if t and t.strip()]
    if title and tags:
        return f"{title} — {', '.join(tags)}"
    return title or (", ".join(tags) if tags else "")


def load_for_index(
    store: WikiStore, rel_path: str
) -> tuple[str, WikiContentKind, str] | None:
    """Return ``(body, kind, header)`` for an indexable file, or ``None``.

    ``header`` is the per-chunk frontmatter line for wiki pages (empty
    when ``semantic.embed_frontmatter`` is off, and always empty for
    sources, which have no frontmatter).

    Skips, all of which are legitimate and quiet:

    - ``wiki/index.md`` (auto-generated, indexing it is just noise)
    - ``wiki/AGENTS.md`` (agent-conventions doc, not content)
    - ``wiki/sources/.sources.db`` (registry, not content)
    - binary or undecodable source files (logged at INFO)
    - anything outside ``wiki/pages/`` or ``wiki/sources/``

    A wiki page that fails to load is NOT quiet: it is logged at WARNING
    and counted by :func:`reindex_all` as ``dropped``. Losing a page from
    the index means losing it from retrieval entirely, so it must never
    pass unnoticed.
    """
    wiki_prefix = f"{store.config.wiki_dir}/"
    pages_prefix = store.pages_prefix()
    sources_prefix = f"{wiki_prefix}{SOURCES_DIR}/"
    sources_local_prefix = f"{wiki_prefix}{SOURCES_LOCAL_DIR}/"

    if any(rel_path == f"{wiki_prefix}{name}" for name in RESERVED_WIKI_FILES):
        return None
    if rel_path == f"{sources_prefix}{REGISTRY_FILENAME}":
        return None
    if rel_path.startswith(sources_local_prefix):
        # NEVER indexed, under any `semantic.index` setting. The vector
        # DB stores each chunk's *verbatim text* alongside its embedding
        # and is staged into the same commit as the write that triggered
        # it — so indexing the local tree would push the exact bytes the
        # tree exists to withhold straight into git, via a path nobody
        # would think to audit.
        #
        # Deliberately unconditional rather than "skip when the DB is
        # tracked". A safety property has to be checkable in one
        # sentence; making it depend on the current .gitignore means a
        # user can silently turn it off by editing an unrelated file.
        # The cost is that `find_similar` cannot see local material —
        # documented in docs/configuration.md, and the derived pages
        # (which is what recall actually wants) are indexed as normal.
        return None

    abs_path = store.root / rel_path
    if not abs_path.is_file():
        if store.is_page_path(rel_path):
            # Reachable via a broken symlink, a directory named `*.md`, or a
            # page deleted between the tree walk and this load. Counted as
            # dropped, so it must say why — otherwise the run exits non-zero
            # pointing at `outmem lint`, which has nothing to report.
            log.warning(
                "semantic: page %s NOT indexed — not a readable file "
                "(broken symlink, or removed during the scan)",
                rel_path,
            )
        return None

    if rel_path.startswith(sources_prefix):
        if store.config.outmem.semantic.index == SEMANTIC_INDEX_PAGES:
            # Honour the scope here too, not just in the batch walk: every
            # incremental path (write-time reindex, `reindex --path`, the
            # `--staged` pre-commit hook) comes through this function. Gate
            # it only in indexable_files_on_disk and a `pages`-scoped wiki
            # re-adds each source on the next commit, pays an embedding
            # round-trip inside the git path for chunks the next reindex
            # deletes again, and churns the tracked .vectors.db forever.
            return None
        try:
            text = abs_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            log.info("semantic: skipping non-text source %s", rel_path)
            return None
        except OSError:
            return None
        return text, "source", ""

    if rel_path.startswith(pages_prefix) and rel_path.endswith(".md"):
        try:
            raw = abs_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            log.warning(
                "semantic: page %s NOT indexed — unreadable: %s", rel_path, exc
            )
            return None
        try:
            # Shared loader (same one `WikiStore.read` policy uses), so a
            # page that is readable, greppable and bm25-indexable because
            # its frontmatter self-heals is also indexed here. Reimplementing
            # a stricter policy left such a page permanently absent from the
            # index and permanently failing `outmem reindex`.
            frontmatter, body, _ = load_page_text(raw)
        except FrontmatterError as exc:
            # Loud on purpose. This used to be a bare `except Exception:
            # return None`, which silently deleted the page from the index
            # (and therefore from retrieval) for something as small as an
            # unquoted year in `tags:`.
            log.warning(
                "semantic: page %s NOT indexed — bad frontmatter: %s. "
                "Fix the page or run `outmem lint`.",
                rel_path,
                exc,
            )
            return None
        header = (
            frontmatter_header(frontmatter)
            if store.config.outmem.semantic.embed_frontmatter
            else ""
        )
        return body, "wiki", header

    return None


def indexable_files_on_disk(store: WikiStore) -> list[str]:
    """Every repo-relative path that would normally be indexed.

    Honours ``semantic.index`` (``"pages"`` default, or ``"pages+sources"``).
    Scoping to ``pages`` keeps ingested material out of the vector
    store: sources are near-duplicates of the pages distilled from them,
    and because the vector search takes a fixed-``k`` KNN before anything
    can filter by kind, they crowd curated pages out of the candidate
    window.

    ``wiki/sources-local/`` is excluded unconditionally — see
    :func:`load_for_index` for why. ``"pages+sources"`` means the
    *tracked* sources tree only.

    Restricting here makes ``reindex_all``'s orphan sweep prune
    already-indexed source chunks on the next run rather than stranding
    them; :func:`load_for_index` enforces the same setting on the
    incremental paths.

    Iterates the on-disk tree without materialising an intermediate
    sorted list of every path under ``wiki/sources/`` — for a corpus
    with thousands of sources, that saved a non-trivial transient
    allocation per ``reindex_all``.
    """
    scope = store.config.outmem.semantic.index
    rels: list[str] = []
    if store.pages_path.is_dir():
        for path in editorial_pages(store.pages_path):
            rel = path.relative_to(store.pages_path).as_posix()
            rels.append(f"{store.config.wiki_dir}/{PAGES_DIR}/{rel}")
    if scope == SEMANTIC_INDEX_PAGES:
        return rels
    if store.sources_path.is_dir():
        for path in store.sources_path.rglob("*"):
            if not path.is_file():
                continue
            if path.parent == store.sources_path and path.name == REGISTRY_FILENAME:
                continue
            rels.append(path.relative_to(store.root).as_posix())
    return rels


def maybe_reindex_commit_paths(
    store: WikiStore, paths: Sequence[str]
) -> str | None:
    """Reindex any indexable file in ``paths`` and return the DB rel-path.

    Called from :meth:`WikiStore._commit_paths` so the vector DB lands
    in the same commit as the page write. Returns ``None`` when
    semantic indexing is off *or* nothing indexable was in ``paths``.
    Errors during reindex are logged and swallowed — they must never
    block a writeback.
    """
    if not available(store):  # don't build an index on a write — opt in via reindex
        return None
    try:
        vs = vector_store_or_open(store)
    except OutmemError:
        raise
    except Exception as exc:
        log.warning("semantic indexing unavailable: %s", exc)
        return None

    did_any = False
    for rel_path in paths:
        abs_path = store.root / rel_path
        if not abs_path.exists():
            try:
                removed = vs.remove_file(rel_path)
            except Exception as exc:
                log.warning("semantic remove %s failed: %s", rel_path, exc)
                continue
            if removed:
                did_any = True
            continue
        load = load_for_index(store, rel_path)
        if load is None:
            continue
        body, kind, header = load
        try:
            settings = store.config.outmem.semantic
            result = vs.reindex_file(
                rel_path,
                body=body,
                kind=kind,
                header=header,
                chunk_size=settings.chunk_size,
                chunk_max=settings.chunk_max,
                overlap_paragraphs=settings.overlap_paragraphs,
            )
        except Exception as exc:
            log.warning("semantic reindex %s failed: %s", rel_path, exc)
            continue
        if not result.skipped:
            did_any = True
    return store.config.outmem.semantic.db_filename if did_any else None
