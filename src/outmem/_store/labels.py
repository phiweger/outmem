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

import posixpath
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from outmem.restricted import DENY_SET, RestrictedSettings
from outmem.sources import SOURCES_DIR, SOURCES_LOCAL_DIR

if TYPE_CHECKING:  # pragma: no cover
    from outmem.store import WikiStore


@dataclass(frozen=True)
class LabelIndex:
    """Resolved labels for everything in the wiki, as of one commit."""

    head: str | None
    """The HEAD sha this index was built from. Half of its validity token.

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


#: An index for a wiki that declares no labels. Everything is open, and
#: every lookup is a dict miss on an empty dict.
EMPTY = LabelIndex(head="")


def registry_stamp(store: WikiStore) -> tuple[tuple[int, int], ...]:
    """A cheap fingerprint of the source registries, for the token."""
    from outmem.sources import REGISTRY_FILENAME

    out: list[tuple[int, int]] = []
    for directory in (store.sources_path, store.sources_local_path):
        try:
            stat = (directory / REGISTRY_FILENAME).stat()
        except OSError:
            out.append((0, 0))
        else:
            out.append((stat.st_mtime_ns, stat.st_size))
    return tuple(out)


def build(store: WikiStore, head: str | None = None) -> LabelIndex:
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
        # Both keys: `provenance:` cites the tree-qualified form while
        # the registry is keyed on the bare rel_path, and a lookup miss
        # would silently read as "this source is open".
        #
        # The bare key takes the UNION across trees. The same rel_path
        # can exist in both `sources/` and `sources-local/`, and letting
        # the second row overwrite the first would resolve an ambiguous
        # lookup to whichever tree happened to be walked last — a
        # coin-flip that fails open half the time.
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
            # Through the index, not the raw map: a page may cite a
            # source under any spelling, and a raw miss would leave the
            # page open while printing the restricted source's filename.
            labels |= index.for_source(ref)
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


class LabelCache:
    """Holds the current :class:`LabelIndex` for one store, keyed by HEAD.

    Lives on the bare store and is shared by every view derived from it
    — views are per-request, the wiki is not, and rebuilding the index
    per request would make each one an O(corpus) walk.
    """

    def __init__(self) -> None:
        self._index: LabelIndex | None = None
        self._lock = threading.Lock()

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
        head = store.head()
        stamp = registry_stamp(store)
        if head is None:
            # No commit to key on. Rebuilding every time is correct and
            # merely slow; caching against a token that never changes
            # would be wrong.
            return build(store, None)
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
            if cached is not None and cached.registries != stamp:
                # A registry changed under us, and `SourceRegistry` is an
                # in-memory snapshot taken at load. Rebuilding the index
                # from the handle we already hold would read the labels
                # this process saw last time, which is the stale value
                # the stamp just told us not to trust. Drop the handles
                # so `build` re-reads. Existing references stay valid;
                # they are merely old.
                store._source_registry = None
                store._source_registry_local = None
            built = build(store, head)
            self._index = built
            return built

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
