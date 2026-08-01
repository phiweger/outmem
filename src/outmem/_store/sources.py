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

from outmem.exceptions import OutmemError
from outmem.sources import (
    REGISTRY_FILENAME,
    SOURCES_DIR,
    IngestionRecord,
    SourceEntry,
    SourceRegistry,
    compute_sha256,
    copy_source,
    derive_document_key,
    distinguishing_segment,
    normalize_document_key,
    read_source_text,
)

if TYPE_CHECKING:
    from outmem.store import WikiStore


def get_registry(store: WikiStore) -> SourceRegistry:
    """Lazy-open + cache the registry for the store's lifetime."""
    if store._source_registry is None:
        store._source_registry = SourceRegistry.load(store.sources_path)
    return store._source_registry


def _ambiguous_identity_error(
    candidate: str, clashing: SourceEntry, origin: str
) -> OutmemError:
    """Refuse an ambiguous identity, with the evidence and both answers.

    The two readings — "a new version of that document" and "a different
    document that happens to share a filename" — are indistinguishable
    from the wiki path, because copying into ``<into>/<sha>/<filename>``
    already discarded what told them apart. The ingest origins still have
    it, so quote both and propose a name from the first place they
    diverge: the operator reads an answer instead of inventing one.
    """
    lines = [f"{candidate!r} is already the identity of {clashing.rel_path!r}."]
    if clashing.origin_path:
        lines += [
            f"    that row came from    {clashing.origin_path}",
            f"    this file comes from  {origin}",
        ]
    distinct = distinguishing_segment(origin, clashing.origin_path)
    namespace = candidate.rsplit("/", 1)[0] if "/" in candidate else ""
    example = f"{namespace}/{distinct}" if distinct and namespace else distinct
    lines += [
        f"If this file is a new version of that document, pass `--as {candidate}`.",
        "If it is a different document that happens to share a filename, pass "
        "`--as` with a name that distinguishes it"
        + (f", e.g. `--as {example}`." if example else " (e.g. `--as fachinfo/amikacin`)."),
    ]
    return OutmemError("\n".join(lines))


def add_source(
    store: WikiStore,
    source: str | Path,
    *,
    into_subdir: str | None = None,
    rename: str | None = None,
    as_key: str | None = None,
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

    if as_key is not None:
        document_key = normalize_document_key(as_key)
    else:
        candidate = derive_document_key(rel_path, sha) or normalize_document_key(rel_path)
        # Refuse rather than guess. A candidate already claimed by a
        # DIFFERENT document means we cannot tell "new version of that" from
        # "different document, same filename" — and guessing wrong writes a
        # supersession edge that would later drive a recheck of one drug's
        # page against another's.
        clashing = [
            e
            for e in registry.entries.values()
            if e.document_key == candidate and e.rel_path != rel_path
        ]
        if clashing:
            raise _ambiguous_identity_error(candidate, clashing[0], str(source_path))
        document_key = candidate

    entry = registry.register(
        rel_path,
        sha256=sha,
        size_bytes=dest.stat().st_size,
        document_key=document_key,
        origin_path=str(source_path),
    )
    if commit:
        store._commit_paths(
            [
                f"{store.config.wiki_dir}/{SOURCES_DIR}/{rel_path}",
                f"{store.config.wiki_dir}/{SOURCES_DIR}/{REGISTRY_FILENAME}",
            ],
            subject=f"source: {rel_path}",
        )
    return entry


def list_sources(store: WikiStore, *, include_missing: bool = False) -> list[SourceEntry]:
    """Registered sources, newest-path-order, excluding rows whose file is gone.

    The registry is not self-cleaning: a source re-ingested after its
    content changed lands at a new sha-addressed path and the old row
    survives, and nothing removes a row when a directory is deleted
    out-of-band. This function never stat'd the filesystem, so those rows
    were handed to the agent as readable sources — it would spend a call
    on ``read_source`` and get "no such source" back.

    Filtering here rather than at the call sites keeps the agent's view
    honest by default. ``include_missing=True`` returns the raw rows for
    maintenance tooling (``outmem sources gc``).
    """
    registry = get_registry(store)
    entries = sorted(registry.entries.values(), key=lambda e: e.rel_path)
    if include_missing:
        return entries
    return [e for e in entries if (store.sources_path / e.rel_path).is_file()]


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
