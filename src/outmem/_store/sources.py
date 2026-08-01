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
from outmem.slug import extract_slug_references
from outmem.sources import (
    REGISTRY_FILENAME,
    SOURCES_DIR,
    DocumentKeyConflict,
    IngestionRecord,
    SourceEntry,
    SourceRef,
    SourceRegistry,
    candidate_document_key,
    copy_source,
    distinguishing_segment,
    normalize_document_key,
    plan_source_copy,
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
    # Plan before copying: a refused ingest must not leave an
    # unregistered orphan under wiki/sources/ that lint then flags and gc
    # refuses to delete.
    placement = plan_source_copy(
        source_path, store.sources_path, into_subdir=into_subdir, rename=rename
    )
    rel_path, sha = placement.rel_path, placement.sha256
    # Absolute: a relative origin is meaningless once recorded, and a
    # relative/absolute mix makes `distinguishing_segment` diverge at the
    # root and propose a name that distinguishes nothing.
    origin = str(source_path.resolve())

    existing = registry.entries.get(rel_path)
    if existing and existing.sha256 == sha:
        # Identical content is the same row, not a new version — but an
        # explicit `--as` still has to land, because "re-ingest with
        # `--as <name>`" is exactly what `sources backfill` tells the
        # operator to do about an ambiguous group.
        if as_key is None:
            return existing
        return _adopt_or_refuse(
            registry, existing, normalize_document_key(as_key), origin
        )

    document_key = (
        normalize_document_key(as_key)
        if as_key is not None
        else candidate_document_key(rel_path, sha)
    )
    dest, rel_path = copy_source(
        source_path, store.sources_path, into_subdir=into_subdir, rename=rename
    )
    try:
        entry = registry.register(
            rel_path,
            sha256=sha,
            size_bytes=dest.stat().st_size,
            document_key=document_key,
            origin_path=origin,
            # A *derived* key that is already taken is unresolvable; a
            # declared one means "supersede that".
            derived_key=as_key is None,
        )
    except DocumentKeyConflict as conflict:
        _unlink_orphan(dest, store.sources_path)
        raise _ambiguous_identity_error(
            conflict.document_key, conflict.claimant, origin
        ) from None
    record_source_refs(store, rel_path)
    if commit:
        store._commit_paths(
            [
                f"{store.config.wiki_dir}/{SOURCES_DIR}/{rel_path}",
                f"{store.config.wiki_dir}/{SOURCES_DIR}/{REGISTRY_FILENAME}",
            ],
            subject=f"source: {rel_path}",
        )
    return entry


def record_source_refs(store: WikiStore, rel_path: str) -> list[SourceRef]:
    """Resolve the page slugs a source names and record the mapping.

    Runs at ingest because that is the only moment the tokens are known
    to be correct: the slugs are live, the wiki is in a state the author
    was looking at, and the file is about to become immutable. Afterwards
    only the registry can carry this — the bytes cannot be touched.

    An **exact** ``[[link]]`` is recorded whenever it resolves. A bare
    prose token is a guess, so it is recorded only if it resolves *and*
    its namespace exists — the same gate lint uses to keep ``12:30`` and
    ``3:1`` out. Anything unresolvable is skipped rather than stored
    wrong; it was already dead when it arrived.
    """
    path = store.sources_path / rel_path
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    known = set(store.list_slugs())
    namespaces = {
        ":".join(s.split(":")[:i])
        for s in known
        for i in range(1, len(s.split(":")))
    }
    resolved: list[SourceRef] = []
    for ref in extract_slug_references(text):
        if not ref.exact and ref.slug.rsplit(":", 1)[0] not in namespaces:
            continue
        try:
            canonical = store.resolve_slug(ref.slug)
        except OutmemError:
            continue
        if canonical not in known:
            continue
        resolved.append(
            SourceRef(
                rel_path=rel_path,
                token=ref.slug,
                page_slug=canonical,
                exact=ref.exact,
            )
        )
    return get_registry(store).record_refs(rel_path, resolved)


def _adopt_or_refuse(
    registry: SourceRegistry, existing: SourceEntry, key: str, origin: str
) -> SourceEntry:
    try:
        return registry.adopt_document_key(existing.rel_path, key)
    except DocumentKeyConflict as conflict:
        raise _ambiguous_identity_error(
            conflict.document_key, conflict.claimant, origin
        ) from None


def _unlink_orphan(dest: Path, sources_path: Path) -> None:
    """Undo a copy whose registration was refused.

    Only ever removes what this call created: the file, and its hash
    directory if that left it empty. A hash directory is content
    addressed, so an empty one has no other claimant.
    """
    dest.unlink(missing_ok=True)
    parent = dest.parent
    if parent != sources_path and parent.is_dir() and not any(parent.iterdir()):
        parent.rmdir()


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
