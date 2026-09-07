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

import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from outmem.restricted import DENY_SET, RestrictedSettings

if TYPE_CHECKING:  # pragma: no cover
    from outmem.store import WikiStore


@dataclass(frozen=True)
class LabelIndex:
    """Resolved labels for everything in the wiki, as of one commit."""

    head: str | None
    """The HEAD sha this index was built from — its validity token.

    Every outmem write produces a commit, so a moved HEAD is exactly the
    invalidation signal. Without it a multi-worker deployment can serve
    a stale index after another worker restricts something, which is the
    one caching bug in this feature that fails *open*.

    ``None`` means the repo had no commits when this was built, so there
    is no token and the index must not be cached. That state is rare (a
    wiki with nothing committed) and rebuilding is merely slower, where
    caching against a token that cannot change would be wrong.
    """

    pages: dict[str, frozenset[str]] = field(default_factory=dict)
    """slug → labels, for every page including aliases and unparseable ones."""

    sources: dict[str, frozenset[str]] = field(default_factory=dict)
    """Source key → labels, keyed by BOTH the registry ``rel_path`` and the
    tree-qualified ``citation_path``, because callers hold one or the other
    and a miss would read as "open"."""

    def for_page(self, slug: str) -> frozenset[str]:
        """Labels for ``slug``. An unknown slug is open — it does not exist,
        and inventing labels for it would make ``exists`` disagree with
        ``read`` about a page that is simply absent."""
        return self.pages.get(slug, frozenset())

    def for_source(self, key: str) -> frozenset[str]:
        """Labels for a source, by registry key or citation path."""
        return self.sources.get(key, frozenset())


#: An index for a wiki that declares no labels. Everything is open, and
#: every lookup is a dict miss on an empty dict.
EMPTY = LabelIndex(head="")


def build(store: WikiStore, head: str | None = None) -> LabelIndex:
    """Walk the wiki and resolve every item's labels.

    Sources first: a page's labels include those of the sources it
    cites, so the source map has to be complete before any page is
    resolved.
    """
    settings = store.restrictions
    sources = _source_labels(store, settings)
    pages = _page_labels(store, settings, sources)
    return LabelIndex(head=head, pages=pages, sources=sources)


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
        out[entry.rel_path] = out.get(entry.rel_path, frozenset()) | labels
        out[entry.citation_path] = labels
    return out


def _page_labels(
    store: WikiStore,
    settings: RestrictedSettings,
    sources: dict[str, frozenset[str]],
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
            labels |= sources.get(ref, frozenset())
        out[page.slug] = labels

    # Aliases inherit the labels of the page they resolve to; otherwise
    # `resolve_slug` would follow an alias straight past the filter.
    for page in pages:
        labels = out.get(page.slug, frozenset())
        if not labels:
            continue
        for alias in page.frontmatter.aliases:
            # A live page always beats an alias claiming its name, so an
            # alias must never *lower* a real page's labels.
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
        if not store.restrictions.enabled:
            return EMPTY
        head = store.head()
        if head is None:
            # No commit to key on. Rebuilding every time is correct and
            # merely slow; caching against a token that never changes
            # would be wrong.
            return build(store, None)
        cached = self._index
        if cached is not None and cached.head == head:
            return cached
        with self._lock:
            # Re-check: another thread may have rebuilt it while we waited.
            cached = self._index
            if cached is not None and cached.head == head:
                return cached
            built = build(store, head)
            self._index = built
            return built

    def invalidate(self) -> None:
        """Drop the cached index.

        HEAD alone is not always enough: ``restricted`` lives in the
        source registry, and a registry write that is not committed (a
        local-tree ingest) moves no HEAD. Write paths that change labels
        call this directly.
        """
        with self._lock:
            self._index = None
