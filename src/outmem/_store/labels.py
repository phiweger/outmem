"""The label index — what every page and source is labelled with.

One walk of the wiki, cached on the store, invalidated by HEAD. Every
visibility decision consults it, so the cost of resolving labels is
paid once per commit rather than once per item examined.

Resolution never raises. A page whose frontmatter will not parse, a
registry column that will not decode, a label nobody declared — each
resolves to :data:`~outmem.restricted.DENY_SET`, which is invisible in
every mode. That is what lets the callers be a single subset test with
no error handling of their own: there is no failure mode in which an
item ends up *more* visible than it should be.

Three sources of labels for a page (spec §5.1), unioned:

1. **Explicit** — ``restricted: [hr]`` in its frontmatter.
2. **Path rules** — ``restricted.paths`` in config. The safety net that
   cannot be forgotten: a new page under ``hr:`` is restricted whether
   or not anyone edited its frontmatter.
3. **Inherited** — the labels of every source it cites (§3.4). The
   highest-leverage rule in the design, because it means restricting a
   document at ingest restricts every page ever compiled from it. It
   also does double duty for closure: a source's ``rel_path`` embeds
   its original filename, so ``sources/policy/9b3d0d/severance-plan.md``
   sitting in an open page's ``provenance:`` is disclosure even when the
   file itself is unreadable.
"""

from __future__ import annotations

import contextlib
import logging
import posixpath
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from outmem.restricted import DENY_SET, RestrictedSettings
from outmem.sources import SOURCES_DIR, SOURCES_LOCAL_DIR

if TYPE_CHECKING:  # pragma: no cover
    from outmem.store import WikiStore

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class LabelIndex:
    """Resolved labels for everything in the wiki, as of one commit."""

    head: tuple[object, ...] | None
    """What HEAD was when this index was built. Half of its validity token.

    Almost every outmem write produces a commit, so a moved HEAD is
    usually the invalidation signal — but see :attr:`registries` for the
    writes that produce none.

    ``None`` means the repo had no commits when this was built, so there
    is no token and the index must not be cached. That state is rare (a
    wiki with nothing committed) and rebuilding is merely slower, where
    caching against a token that cannot change would be wrong.
    """

    registries: tuple[tuple[int, int], ...] = ()
    """``(mtime_ns, size)`` per source registry — the other half.

    Source labels live in ``.sources.db``, and two writes to it produce
    no commit: an ingest into the untracked local tree, and re-ingesting
    identical bytes with ``--restricted`` (which is the documented way
    to correct a mislabelled source). HEAD does not move, so a process
    that is not the writer would keep serving a source it no longer may.

    Invalidating in-process covers the writer. This covers everyone
    else, which is the deployment the design assumes — several workers
    against one repository — and it is two ``stat`` calls.
    """

    pages: dict[str, frozenset[str]] = field(default_factory=dict)
    """slug → labels, for every page including aliases and unparseable ones."""

    sources: dict[str, frozenset[str]] = field(default_factory=dict)
    """Source labels, keyed by the bare registry ``rel_path``. Callers
    name a source three different ways, so lookups normalise (see
    :meth:`for_source`) rather than the map carrying every spelling."""

    settings: RestrictedSettings = field(default_factory=RestrictedSettings)
    """The config this index was built from, kept for lookups of names
    that are not in it — see :meth:`for_page`."""

    wiki_dir: str = "wiki"

    def for_page(self, slug: str) -> frozenset[str]:
        """Labels for ``slug``, including path rules for names that are
        not live pages.

        ``pages`` is built from the ``.md`` files under ``wiki/pages/``,
        so anything else in that tree — a ``.txt`` a search can still
        reach, a slug named in ``AGENTS.md`` before the page exists —
        was missing from it and read as open, right through a namespace
        the config had restricted. The path rule is the safety net that
        is supposed to cover exactly those cases, so it has to apply at
        lookup as well as at build.
        """
        known = self.pages.get(slug)
        if known is not None:
            return known
        return self.settings.labels_for_slug(slug)

    def for_source(self, key: str) -> frozenset[str]:
        """Labels for a source, under any spelling of its path.

        ``resolve_source`` resolves a caller's string against the
        *filesystem*, so ``sources/./x``, ``sources/../sources/x`` and
        ``wiki/sources/x`` all reach the same file. A map of literal
        keys cannot keep up with that, and a miss reads as open — which
        made ``read_source("sources/./<rel>")`` return a restricted
        file's bytes. Normalise the query instead of enumerating
        spellings.
        """
        for candidate in self._source_keys(key):
            found = self.sources.get(candidate)
            if found is not None:
                return found
        return frozenset()

    def knows_source(self, key: str) -> bool:
        """Whether any spelling of ``key`` is in this index.

        Distinct from :meth:`for_source` returning the empty set, which
        is also what an unknown key gives — and the caller needs to tell
        "this source is open" from "this index has never heard of it".
        """
        return any(c in self.sources for c in self._source_keys(key))

    def source_keys(self, key: str) -> list[str]:
        """``key`` reduced toward the bare registry ``rel_path``.

        Public because the store needs the same normalisation to look a
        candidate up on disk.
        """
        return self._source_keys(key)

    def _source_keys(self, key: str) -> list[str]:
        """``key`` reduced toward the bare registry ``rel_path``."""
        cleaned = posixpath.normpath(key.replace("\\", "/")).lstrip("/")
        out = [cleaned]
        for prefix in (
            f"{self.wiki_dir}/{SOURCES_DIR}/",
            f"{self.wiki_dir}/{SOURCES_LOCAL_DIR}/",
            f"{SOURCES_DIR}/",
            f"{SOURCES_LOCAL_DIR}/",
        ):
            if cleaned.startswith(prefix):
                out.append(cleaned[len(prefix):])
        return out


# How long a label index built without a reachable HEAD stays valid.
# Only reached by a wiki whose `.git` is absent, which outmem cannot
# write to anyway; see `LabelCache._without_head`.
_HEADLESS_TTL_SECONDS = 5.0

_warned_headless: set[str] = set()


def _warn_headless_once(store: WikiStore) -> None:
    """Say so, once per wiki per process.

    The failure this replaces was silent: access control worked
    perfectly and every tool call took most of a second, with nothing
    anywhere saying why.
    """
    key = str(store.root)
    if key in _warned_headless:
        return
    _warned_headless.add(key)
    log.warning(
        "%s has restriction labels but no reachable git HEAD, so the label "
        "index cannot be cached against a commit. Falling back to a %.0fs "
        "cache; visibility checks will be slower and a change made outside "
        "this process can take that long to be seen. Keep the wiki's .git "
        "directory (a depth-1 clone is enough) to remove both.",
        store.root,
        _HEADLESS_TTL_SECONDS,
    )


def _release_registries(store: WikiStore) -> None:
    """Close and drop both registry handles.

    Closing matters: dropping the reference alone orphans a live SQLite
    connection that ``WikiStore.close`` can then never reach, so a
    long-lived worker leaks one per cross-process registry write.
    """
    for attr in ("_source_registry", "_source_registry_local"):
        handle = getattr(store, attr)
        if handle is not None:
            # Suppressed: the handle is on its way out either way, and a
            # close that fails must not stop the caller getting a fresh
            # one.
            with contextlib.suppress(Exception):
                handle.close()
            setattr(store, attr, None)


#: An index for a wiki that declares no labels. Everything is open, and
#: every lookup is a dict miss on an empty dict.
EMPTY = LabelIndex(head=())


def registry_stamp(store: WikiStore) -> tuple[tuple[int, int], ...]:
    """A cheap fingerprint of the source registries, for the token."""
    from outmem.sources import REGISTRY_FILENAME

    out: list[tuple[int, int]] = []
    for directory in (store.sources_path, store.sources_local_path):
        out.append(_stamp(directory / REGISTRY_FILENAME))
    return tuple(out)


def _stamp(path: Path) -> tuple[int, int]:
    try:
        stat = path.stat()
    except OSError:
        return (0, 0)
    return (stat.st_mtime_ns, stat.st_size)


def head_stamp(store: WikiStore) -> tuple[object, ...]:
    """What HEAD is, read from the filesystem rather than from git.

    ``git rev-parse HEAD`` is a subprocess — about 2 ms — and every
    visibility check on a view was paying for one. The label index needs
    to know only whether HEAD *moved*, and ``.git/HEAD`` plus the ref it
    names answer that from two ``stat`` calls.

    Falls back to the subprocess for anything unusual (a worktree, a
    packed ref this cannot see), so the token is never weaker than it
    was — only cheaper in the common case.
    """
    git_dir = store.root / ".git"
    if not git_dir.exists():
        # No repository at all — a depth-1 export, a read-only mount.
        # Returning early matters: the fallback below shells out to `git
        # rev-parse`, which on a directory with no `.git` *fails*, once
        # per visibility check. A subprocess per call to learn something
        # a single `exists()` already told us.
        return (None,)
    head_file = git_dir / "HEAD"
    try:
        raw = head_file.read_text(encoding="utf-8").strip()
    except OSError:
        return (store.head(),)
    if not raw.startswith("ref: "):
        return ("detached", raw)  # already the sha
    ref = git_dir / raw[len("ref: ") :]
    if not ref.exists():
        # Packed refs, or a branch with no commits yet. Ask git; the
        # answer is what it always was.
        return (store.head(),)
    return ("ref", raw, _stamp(ref))


def build(store: WikiStore, head: tuple[object, ...] | None = None) -> LabelIndex:
    """Walk the wiki and resolve every item's labels.

    Sources first: a page's labels include those of the sources it
    cites, so the source map has to be complete before any page is
    resolved.
    """
    settings = store.restrictions
    sources = _source_labels(store, settings)
    index = LabelIndex(
        head=head,
        registries=registry_stamp(store),
        sources=sources,
        settings=settings,
        wiki_dir=store.config.wiki_dir,
    )
    index.pages.update(_page_labels(store, settings, index))
    return index


def _source_labels(
    store: WikiStore, settings: RestrictedSettings
) -> dict[str, frozenset[str]]:
    from outmem._store.sources import list_sources

    out: dict[str, frozenset[str]] = {}
    # The unfiltered implementation, deliberately, not the WikiStore
    # method: `store` here may be a view, and building the index from a
    # filtered listing would resolve labels from a corpus the filter had
    # already trimmed — an index that agrees with itself and with
    # nothing else. Every enforcement point filters in the WikiStore
    # method and leaves these primitives whole for exactly this reason.
    for entry in list_sources(store, include_missing=True):
        labels = settings.resolve(entry.restricted) | settings.labels_for_source(
            entry.rel_path
        )
        # One key, the bare rel_path, unioned across trees: the same
        # path can exist in both `sources/` and `sources-local/`, and
        # letting the second row overwrite the first would resolve an
        # ambiguous lookup to whichever tree was walked last — a
        # coin-flip that fails open half the time. Every other spelling
        # a caller might use is normalised down to this key by
        # `LabelIndex.for_source`.
        out[entry.rel_path] = out.get(entry.rel_path, frozenset()) | labels
    return out


def _page_labels(
    store: WikiStore,
    settings: RestrictedSettings,
    index: LabelIndex,
) -> dict[str, frozenset[str]]:
    from outmem.index import load_editorial_pages
    from outmem.lint import provenance_ref

    out: dict[str, frozenset[str]] = {}
    pages, failures = load_editorial_pages(store.pages_path)

    for failure in failures:
        # Its explicit labels cannot be read, so they cannot be trusted
        # to be empty. Denied to every mode, and reported by lint.
        out[failure.slug] = DENY_SET

    for page in pages:
        labels = settings.resolve(page.frontmatter.restricted)
        labels |= settings.labels_for_slug(page.slug)
        for entry in page.frontmatter.provenance:
            ref = provenance_ref(entry)
            if ref is None:
                continue
            # Resolved against the filesystem, not matched as text: a
            # page may cite a source under any spelling that reaches the
            # file, and a miss leaves the page open while printing the
            # restricted source's filename in its own provenance.
            labels |= _source_labels_for_ref(store, index, ref)
        out[page.slug] = labels

    # Aliases inherit the labels of the page they resolve to; otherwise
    # `resolve_slug` would follow an alias straight past the filter.
    live = {page.slug for page in pages}
    for page in pages:
        labels = out.get(page.slug, frozenset())
        if not labels:
            continue
        for alias in page.frontmatter.aliases:
            # Only for a name no live page occupies. A live page always
            # beats an alias claiming its name (`resolve_slug` checks the
            # file first), so an alias must not change that page's labels
            # in EITHER direction.
            #
            # Raising them is the dangerous one, and it was open: a
            # session in mode {hr} could write an HR page whose
            # `aliases:` named an open page, relabel that page to {hr}
            # without touching it, and then legally edit it — a
            # three-call write-down through a metadata field, ending
            # with HR text in a file whose own frontmatter carries no
            # label at all.
            if alias in live:
                continue
            out[alias] = out.get(alias, frozenset()) | labels
    return out


def _source_labels_for_ref(
    store: WikiStore, index: LabelIndex, ref: str
) -> frozenset[str]:
    """Labels of the source a provenance entry cites, under any spelling."""
    from outmem._store.sources import resolve_source

    found = resolve_source(store, ref)
    if found is not None:
        tree, canonical = found
        return index.for_source(f"{tree.name}/{canonical}")
    # Cites nothing that exists — a dangling provenance entry, which
    # `outmem lint` reports and which discloses nothing.
    return index.for_source(ref)


class LabelCache:
    """Holds the current :class:`LabelIndex` for one store, keyed by HEAD.

    Lives on the bare store and is shared by every view derived from it
    — views are per-request, the wiki is not, and rebuilding the index
    per request would make each one an O(corpus) walk.
    """

    def __init__(self) -> None:
        self._index: LabelIndex | None = None
        self._lock = threading.Lock()
        self._handle_stamp: tuple[tuple[int, int], ...] | None = None
        self._headless_built_at = float("-inf")

    def get(self, store: WikiStore) -> LabelIndex:
        # Deliberately NOT short-circuited on `restrictions.enabled`.
        #
        # Emptying or deleting the `restricted:` block would otherwise
        # publish everything already labelled — one deleted line and the
        # whole HR corpus is open to every view. Building the index
        # anyway means an undeclared label collapses to DENY (the same
        # rule `RestrictedSettings.resolve` applies to a typo), so
        # withdrawing a declaration *hides* its content rather than
        # releasing it, and the operator, who holds the bare store and
        # is not filtered, can still see and fix it.
        #
        # The cost lands only on callers that took a view: every
        # enforcement point tests `_mode is None` first, so a wiki with
        # no access control never reaches this method at all.
        head = head_stamp(store)
        stamp = registry_stamp(store)
        if head == (None,):
            return self._without_head(store, stamp)
        cached = self._index
        if cached is not None and (cached.head, cached.registries) == (head, stamp):
            return cached
        with self._lock:
            # Re-check: another thread may have rebuilt it while we waited.
            cached = self._index
            if cached is not None and (cached.head, cached.registries) == (
                head,
                stamp,
            ):
                return cached
            if self._handle_stamp != stamp:
                # A registry changed since these handles were opened, and
                # `SourceRegistry` is an in-memory snapshot taken at load.
                # Rebuilding from the handle we hold would read the
                # labels this process saw last time — the stale value the
                # stamp just told us not to trust.
                #
                # Keyed on when the HANDLES were loaded, not on the
                # cached index: a first build, or one triggered by HEAD
                # alone, would otherwise reuse a snapshot that predates
                # another process's registry write.
                _release_registries(store)
                self._handle_stamp = stamp
            built = build(store, head)
            self._index = built
            return built

    def _without_head(
        self, store: WikiStore, stamp: tuple[tuple[int, int], ...]
    ) -> LabelIndex:
        """The index for a wiki with no reachable HEAD.

        A deployment that strips ``.git`` — a depth-1 export, a
        read-only mount — has no commit to key the cache on, and
        rebuilding on every visibility check is an O(corpus) walk per
        store call. Measured on 1200 pages: 648 ms against 0.10 ms with
        a repo, which is not a slow path, it is a cliff, and a silent
        one.

        Such a wiki cannot be written through outmem at all — every
        write path commits, and committing needs git — so the corpus
        only changes when something outside the process replaces it,
        which in practice means a redeploy and a restart. Caching is
        therefore correct for that shape. The TTL is the concession to
        the shape it is *not* correct for: somebody editing files under
        a non-git directory in a long-lived process sees their change
        within a bounded window rather than never.
        """
        import time

        now = time.monotonic()
        cached = self._index
        if (
            cached is not None
            and cached.registries == stamp
            and now - self._headless_built_at < _HEADLESS_TTL_SECONDS
        ):
            return cached
        with self._lock:
            cached = self._index
            if (
                cached is not None
                and cached.registries == stamp
                and now - self._headless_built_at < _HEADLESS_TTL_SECONDS
            ):
                return cached
            _warn_headless_once(store)
            built = build(store, None)
            self._index = built
            self._headless_built_at = time.monotonic()
            return built

    def note_registry_write(self, store: WikiStore) -> None:
        """Record that this process just wrote a registry itself.

        Without it the next rebuild would see a moved stamp and throw
        away a handle that is not stale at all — the one this process
        used to make the change.
        """
        with self._lock:
            self._handle_stamp = registry_stamp(store)

    def invalidate(self) -> None:
        """Drop the cached index for this process.

        The registry stamp in the token already catches an uncommitted
        registry write from *another* process. This is the same-process
        shortcut: it takes effect immediately rather than on the next
        differing stat, which matters when a write and the read that
        must see it happen in the same call.
        """
        with self._lock:
            self._index = None
