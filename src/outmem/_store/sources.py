"""Source-registry operations for :class:`outmem.store.WikiStore`.

Split out of ``store.py`` for file-size hygiene. These are
implementation helpers — public access is via the
``WikiStore.{add_source,list_sources,get_source,read_source,record_ingestion}``
methods, which forward here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from outmem.exceptions import OutmemError
from outmem.slug import extract_slug_references
from outmem.sources import (
    REGISTRY_FILENAME,
    SOURCES_DIR,
    SOURCES_LOCAL_DIR,
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


@dataclass(frozen=True)
class SourceTree:
    """One source tree: where it lives, and whether git may see it.

    Two trees exist — ``sources/`` (tracked) and ``sources-local/``
    (never tracked). They are otherwise identical: same layout, same
    registry schema, same read path. Passing this object around instead
    of a ``local: bool`` keeps the three facts that must agree — the
    directory, the registry to write, and whether to commit — bound
    together at the point they are decided rather than re-derived at
    each call site.
    """

    name: str
    path: Path
    tracked: bool

    @property
    def registry_relpath(self) -> str:
        """Registry path relative to ``wiki/`` (for ``git add``)."""
        return f"{self.name}/{REGISTRY_FILENAME}"

    def entry_relpath(self, rel_path: str) -> str:
        """Source path relative to ``wiki/`` (for ``git add`` / citation)."""
        return f"{self.name}/{rel_path}"


def tracked_tree(store: WikiStore) -> SourceTree:
    """The git-tracked source tree — the default home for ingests."""
    return SourceTree(name=SOURCES_DIR, path=store.sources_path, tracked=True)


def local_tree(store: WikiStore) -> SourceTree:
    """The untracked source tree, for material that must not ship."""
    return SourceTree(
        name=SOURCES_LOCAL_DIR, path=store.sources_local_path, tracked=False
    )


def existing_trees(store: WikiStore) -> list[SourceTree]:
    """Both trees, minus any that isn't on disk yet.

    ``sources-local/`` is created on first local ingest, so a wiki that
    never used it has only one tree. Readers iterate this rather than
    hardcoding the pair, so "the local tree is absent" never needs a
    special case at the call site.
    """
    return [t for t in (tracked_tree(store), local_tree(store)) if t.path.is_dir()]


def get_registry(store: WikiStore, tree: SourceTree | None = None) -> SourceRegistry:
    """Lazy-open + cache a tree's registry for the store's lifetime.

    Each tree carries its own ``.sources.db``. That separation is the
    load-bearing part of the split, not an implementation detail: the
    registry records ``rel_path`` (which embeds the filename),
    ``sha256``, and ``origin_path`` (an absolute path on the ingesting
    machine). A shared registry would commit all three for every local
    source, leaking precisely what the local tree exists to withhold.

    A read-only store never materialises a missing registry.
    :meth:`SourceRegistry.load` creates the directory and the DB file as
    a side effect of opening, which on a wiki with no registered sources
    means merely *listing* them would write into a tree the caller was
    promised would not be touched.
    """
    if tree is None or tree.tracked:
        if store._source_registry is None:
            store._source_registry = _load_registry(store, store.sources_path)
        return store._source_registry
    if store._source_registry_local is None:
        store._source_registry_local = _load_registry(store, tree.path)
    return store._source_registry_local


def _load_registry(store: WikiStore, path: Path) -> SourceRegistry:
    """Open a tree's registry, or an empty stand-in for a read-only store."""
    if store.config.read_only and not (path / REGISTRY_FILENAME).is_file():
        return SourceRegistry(sources_dir=path)
    return SourceRegistry.load(path)


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
    local: bool = False,
    commit: bool = True,
) -> SourceEntry:
    source_path = Path(source).expanduser()
    if local:
        # Creates the directory AND its .gitignore entry, in that order.
        store.ensure_sources_local()
    tree = local_tree(store) if local else tracked_tree(store)
    registry = get_registry(store, tree)
    # Plan before copying: a refused ingest must not leave an
    # unregistered orphan under the tree that lint then flags and gc
    # refuses to delete.
    placement = plan_source_copy(
        source_path, tree.path, into_subdir=into_subdir, rename=rename
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
            return replace(existing, local=not tree.tracked)
        return replace(
            _adopt_or_refuse(registry, existing, normalize_document_key(as_key), origin),
            local=not tree.tracked,
        )

    document_key = (
        normalize_document_key(as_key)
        if as_key is not None
        else candidate_document_key(rel_path, sha)
    )
    dest, rel_path = copy_source(
        source_path, tree.path, into_subdir=into_subdir, rename=rename
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
        _unlink_orphan(dest, tree.path)
        raise _ambiguous_identity_error(
            conflict.document_key, conflict.claimant, origin
        ) from None
    # The row itself carries no tree column (see SourceEntry.local); tag
    # the returned copy so the caller's `citation_path` is right without
    # a second lookup.
    entry = replace(entry, local=not tree.tracked)
    record_source_refs(store, rel_path, tree)
    # A local ingest has nothing to commit: both the file and its
    # registry live inside the gitignored tree. Committing here would be
    # a no-op at best and, if the ignore rule were ever missing, exactly
    # the leak the split exists to prevent.
    if commit and tree.tracked:
        store._commit_paths(
            [
                f"{store.config.wiki_dir}/{tree.entry_relpath(rel_path)}",
                f"{store.config.wiki_dir}/{tree.registry_relpath}",
            ],
            subject=f"source: {rel_path}",
        )
    return entry


def record_source_refs(
    store: WikiStore, rel_path: str, tree: SourceTree | None = None
) -> list[SourceRef]:
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
    tree = tree if tree is not None else tracked_tree(store)
    path = tree.path / rel_path
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
    return get_registry(store, tree).record_refs(rel_path, resolved)


def split_tree_prefix(store: WikiStore, rel_path: str) -> tuple[SourceTree | None, str]:
    """Strip a leading ``sources/`` / ``sources-local/`` off ``rel_path``.

    Both spellings reach these functions in practice and both must work.
    The registry keys on the tree-relative form (``<sha>/file.md``), but
    what a caller has in hand is usually the prefixed one: it is what
    ``grep_wiki`` prints, and what pages carry in ``provenance:``. An
    agent that greps and then reads passes the string it just saw.

    Returns ``(tree, remainder)``; ``tree`` is ``None`` when no prefix
    was present, leaving the choice to :func:`resolve_source`.
    """
    for tree in (tracked_tree(store), local_tree(store)):
        prefix = f"{tree.name}/"
        if rel_path.startswith(prefix):
            return tree, rel_path[len(prefix) :]
    return None, rel_path


def resolve_source(store: WikiStore, rel_path: str) -> tuple[SourceTree, str] | None:
    """Locate the tree holding ``rel_path``, honouring an explicit prefix.

    Without a prefix the trees are tried tracked-first. A collision needs
    identical content on both sides (the path embeds the content hash),
    so the two candidates are the same bytes and preferring the tracked
    one is both deterministic and harmless.

    Falls back to a registry lookup so a row whose file was deleted
    out-of-band still resolves — callers can then report "the source is
    gone" instead of "no such source", which are different problems.
    """
    hinted, remainder = split_tree_prefix(store, rel_path)
    candidates = [hinted] if hinted is not None else existing_trees(store)
    for tree in candidates:
        if (tree.path / remainder).is_file():
            return tree, remainder
    for tree in candidates:
        if tree.path.is_dir() and get_registry(store, tree).get(remainder) is not None:
            return tree, remainder
    return None


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

    Spans both trees. ``rel_path`` stays the registry key (tree-relative)
    because every internal consumer keys on it; which tree a row came
    from rides along as :attr:`SourceEntry.local`, and the qualified form
    is available as :attr:`SourceEntry.citation_path` for anything
    user-facing.
    """
    out: list[SourceEntry] = []
    for tree in existing_trees(store):
        for entry in get_registry(store, tree).entries.values():
            if not include_missing and not (tree.path / entry.rel_path).is_file():
                continue
            out.append(replace(entry, local=not tree.tracked))
    return sorted(out, key=lambda e: e.citation_path)


def get_source(store: WikiStore, rel_path: str) -> SourceEntry | None:
    found = resolve_source(store, rel_path)
    if found is None:
        return None
    tree, remainder = found
    entry = get_registry(store, tree).get(remainder)
    if entry is None:
        return None
    return replace(entry, local=not tree.tracked)


def read_source(store: WikiStore, rel_path: str, *, max_chars: int | None = None) -> str:
    cap = max_chars if max_chars is not None else store.config.outmem.sources.max_chars
    found = resolve_source(store, rel_path)
    if found is None:
        # Preserve the not-found message shape the tracked-only path
        # produced, so callers keying on it keep working.
        return read_source_text(store.sources_path, rel_path, max_chars=cap)
    tree, remainder = found
    return read_source_text(tree.path, remainder, max_chars=cap)


def record_ingestion(
    store: WikiStore,
    rel_path: str,
    *,
    prompt: str | None,
    pages_touched: Sequence[str],
    commit: bool = True,
    when: datetime | None = None,
) -> IngestionRecord:
    found = resolve_source(store, rel_path)
    tree, key = found if found is not None else (tracked_tree(store), rel_path)
    record = get_registry(store, tree).record_ingestion(
        key,
        prompt=prompt,
        pages_touched=pages_touched,
        when=when,
    )
    # The local registry lives inside the gitignored tree; there is
    # nothing for git to record.
    if commit and tree.tracked:
        store._commit_paths(
            [f"{store.config.wiki_dir}/{tree.registry_relpath}"],
            subject=f"ingest: {key}",
        )
    return record
