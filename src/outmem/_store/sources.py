"""Source-registry operations for :class:`outmem.store.WikiStore`.

Split out of ``store.py`` for file-size hygiene. These are
implementation helpers — public access is via the
``WikiStore.{add_source,list_sources,get_source,read_source,record_ingestion}``
methods, which forward here.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from outmem.sources import (
    REGISTRY_FILENAME,
    SOURCES_DIR,
    IngestionRecord,
    SourceEntry,
    SourceRegistry,
    compute_sha256,
    copy_source,
    read_source_text,
)

if TYPE_CHECKING:
    from outmem.store import WikiStore


def get_registry(store: WikiStore) -> SourceRegistry:
    """Lazy-open + cache the registry for the store's lifetime."""
    if store._source_registry is None:
        store._source_registry = SourceRegistry.load(store.sources_path)
    return store._source_registry


def add_source(
    store: WikiStore,
    source: str | Path,
    *,
    into_subdir: str | None = None,
    rename: str | None = None,
    commit: bool = True,
) -> SourceEntry:
    source_path = Path(source).expanduser()
    registry = get_registry(store)
    dest, rel_path = copy_source(
        source_path,
        store.sources_path,
        into_subdir=into_subdir,
        rename=rename,
    )
    sha = compute_sha256(dest)
    existing = registry.entries.get(rel_path)
    if existing and existing.sha256 == sha:
        return existing
    entry = registry.register(rel_path, sha256=sha, size_bytes=dest.stat().st_size)
    if commit:
        store._commit_paths(
            [
                f"{store.config.wiki_dir}/{SOURCES_DIR}/{rel_path}",
                f"{store.config.wiki_dir}/{SOURCES_DIR}/{REGISTRY_FILENAME}",
            ],
            subject=f"source: {rel_path}",
        )
    return entry


def list_sources(store: WikiStore) -> list[SourceEntry]:
    registry = get_registry(store)
    return sorted(registry.entries.values(), key=lambda e: e.rel_path)


def get_source(store: WikiStore, rel_path: str) -> SourceEntry | None:
    return get_registry(store).get(rel_path)


def read_source(store: WikiStore, rel_path: str, *, max_chars: int | None = None) -> str:
    cap = max_chars if max_chars is not None else store.config.outmem.sources.max_chars
    return read_source_text(store.sources_path, rel_path, max_chars=cap)


def record_ingestion(
    store: WikiStore,
    rel_path: str,
    *,
    prompt: str | None,
    pages_touched: Sequence[str],
    commit: bool = True,
    when: datetime | None = None,
) -> IngestionRecord:
    registry = get_registry(store)
    record = registry.record_ingestion(
        rel_path,
        prompt=prompt,
        pages_touched=pages_touched,
        when=when,
    )
    if commit:
        store._commit_paths(
            [f"{store.config.wiki_dir}/{SOURCES_DIR}/{REGISTRY_FILENAME}"],
            subject=f"ingest: {rel_path}",
        )
    return record
