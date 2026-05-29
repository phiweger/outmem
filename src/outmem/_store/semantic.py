"""Semantic-index operations for :class:`outmem.store.WikiStore`.

Split out of ``store.py`` for file-size hygiene. Public access is via
the ``WikiStore.semantic_*`` methods, which forward here.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from outmem.config import SEMANTIC_DISABLED_HELP
from outmem.exceptions import OutmemError
from outmem.frontmatter import parse_wiki_page
from outmem.index import RESERVED_WIKI_FILES, editorial_pages
from outmem.slug import PAGES_DIR, slug_to_relpath
from outmem.sources import REGISTRY_FILENAME, SOURCES_DIR

if TYPE_CHECKING:
    from outmem.semantic import Match, ReindexResult, VectorStore
    from outmem.store import WikiStore

log = logging.getLogger(__name__)

WikiContentKind = Literal["wiki", "source"]


def enabled(store: WikiStore) -> bool:
    return store.config.outmem.semantic.enabled


def index_is_empty(store: WikiStore) -> bool:
    """True if the semantic index has no indexed files yet.

    Used to fail loud when ``semantic.enabled`` is true but ``outmem
    reindex`` was never run — otherwise queries return nothing and look
    like a useless retriever. Opens the vector store, so the first call
    pays the one-time ``build_embedder`` probe (a tiny embed request to
    detect dimensions); the handle is then cached on the store.
    """
    return len(vector_store_or_open(store).list_indexed_files()) == 0


def db_path(store: WikiStore) -> Path:
    return store.root / store.config.outmem.semantic.db_filename


def vector_store_or_open(store: WikiStore) -> VectorStore:
    """Lazy open of the :class:`VectorStore`.

    Raises :class:`OutmemError` if semantic indexing is disabled or
    the extras aren't installed. The build_embedder probe is real
    (one API call) so we cache the handle.
    """
    if not enabled(store):
        raise OutmemError(SEMANTIC_DISABLED_HELP)
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
        embedder = build_embedder(settings.embedding_model)
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
    if not enabled(store):
        return None
    load = load_for_index(store, rel_path)
    if load is None:
        return None
    body, kind = load
    vs = vector_store_or_open(store)
    settings = store.config.outmem.semantic
    return vs.reindex_file(
        rel_path,
        body=body,
        kind=kind,
        chunk_size=settings.chunk_size,
        chunk_max=settings.chunk_max,
        overlap_paragraphs=settings.overlap_paragraphs,
    )


def remove_path(store: WikiStore, rel_path: str) -> int:
    if not enabled(store):
        return 0
    vs = vector_store_or_open(store)
    return vs.remove_file(rel_path)


def reindex_all(store: WikiStore, *, force: bool = False) -> dict[str, Any]:
    if not enabled(store):
        raise OutmemError(SEMANTIC_DISABLED_HELP)
    vs = vector_store_or_open(store)

    on_disk = indexable_files_on_disk(store)
    reindexed = 0
    skipped = 0
    added_chunks = 0
    for rel_path in on_disk:
        if force:
            vs.remove_file(rel_path)
        result = reindex_path(store, rel_path)
        if result is None:
            continue
        if result.skipped:
            skipped += 1
        else:
            reindexed += 1
            added_chunks += result.chunks_added

    removed = 0
    on_disk_set = set(on_disk)
    for rel_path, _, _ in vs.list_indexed_files():
        if rel_path not in on_disk_set:
            vs.remove_file(rel_path)
            removed += 1

    return {
        "reindexed": reindexed,
        "skipped": skipped,
        "removed": removed,
        "chunks_added": added_chunks,
    }


def load_for_index(store: WikiStore, rel_path: str) -> tuple[str, WikiContentKind] | None:
    """Return ``(body, kind)`` for an indexable file, or ``None`` to skip.

    Skips:

    - ``wiki/index.md`` (auto-generated, indexing it is just noise)
    - ``wiki/AGENTS.md`` (agent-conventions doc, not content)
    - ``wiki/sources/.sources.db`` (registry, not content)
    - binary or undecodable source files (logged at INFO)
    - anything outside ``wiki/pages/`` or ``wiki/sources/``
    """
    wiki_prefix = f"{store.config.wiki_dir}/"
    pages_prefix = f"{wiki_prefix}{PAGES_DIR}/"
    sources_prefix = f"{wiki_prefix}{SOURCES_DIR}/"

    if any(rel_path == f"{wiki_prefix}{name}" for name in RESERVED_WIKI_FILES):
        return None
    if rel_path == f"{sources_prefix}{REGISTRY_FILENAME}":
        return None

    abs_path = store.root / rel_path
    if not abs_path.is_file():
        return None

    if rel_path.startswith(sources_prefix):
        try:
            text = abs_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            log.info("semantic: skipping non-text source %s", rel_path)
            return None
        except OSError:
            return None
        return text, "source"

    if rel_path.startswith(pages_prefix) and rel_path.endswith(".md"):
        try:
            raw = abs_path.read_text(encoding="utf-8")
        except OSError:
            return None
        try:
            _, body = parse_wiki_page(raw)
        except Exception:
            # Malformed frontmatter isn't indexed — lint surfaces it.
            return None
        return body, "wiki"

    return None


def indexable_files_on_disk(store: WikiStore) -> list[str]:
    """Every repo-relative path that would normally be indexed.

    Iterates the on-disk tree without materialising an intermediate
    sorted list of every path under ``wiki/sources/`` — for a corpus
    with thousands of sources, that saved a non-trivial transient
    allocation per ``reindex_all``.
    """
    rels: list[str] = []
    if store.pages_path.is_dir():
        for path in editorial_pages(store.pages_path):
            rel = path.relative_to(store.pages_path).as_posix()
            rels.append(f"{store.config.wiki_dir}/{PAGES_DIR}/{rel}")
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
    if not enabled(store):
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
        body, kind = load
        try:
            settings = store.config.outmem.semantic
            result = vs.reindex_file(
                rel_path,
                body=body,
                kind=kind,
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
