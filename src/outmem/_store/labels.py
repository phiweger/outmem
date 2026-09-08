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
import time
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

    ``None`` means there was no reachable HEAD — a wiki with no commits
    yet, or a deployment that stripped ``.git``. There is no token to
    compare, so :meth:`LabelCache._without_head` falls back to a short
    time-based cache instead; rebuilding on every check turned out to
    cost 648 ms per call on 1200 pages.
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

def _headless_warning(store: WikiStore) -> str:
    """What to say, which depends on WHY there is no HEAD.

    Telling somebody to keep their `.git` while `.git` is sitting right
    there is worse than saying nothing — they go looking for a missing
    directory instead of the empty branch that is actually the cause.
    """
    common = (
        f"{store.root} has restriction labels but no reachable git HEAD, so "
        f"the label index cannot be keyed on a commit. Falling back to a "
        f"{_HEADLESS_TTL_SECONDS:.0f}s cache: a label change made outside "
        f"this process can take that long to be seen."
    )
    if _git_dir(store) is None:
        return (
            f"{common} Keep the wiki's .git directory (a depth-1 clone is "
            "enough) and the index is invalidated by the commit instead."
        )
    return (
        f"{common} The repository is there but has no commit on the current "
        "branch yet; this resolves itself on the first one."
    )


def corpus_stamp(store: WikiStore) -> tuple[int, int, int]:
    """A cheap fingerprint of ``wiki/pages/`` — no file is opened.

    ``(file count, newest mtime, total size)``. Used only on the
    headless path, where there is no commit to ask instead: it turns
    "the clock says this may be stale" into "the disk says whether it
    is", at a stat per file rather than a parse per file.
    """
    count = 0
    newest = 0
    total = 0
    for path in store.pages_path.rglob("*.md"):
        try:
            stat = path.stat()
        except OSError:
            continue
        count += 1
        newest = max(newest, stat.st_mtime_ns)
        total += stat.st_size
    return (count, newest, total)


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

    Returns ``(None,)`` when there is no repository at all, without
    asking git — the fallback below spawns a process, and on a
    directory with no ``.git`` that process *fails*, once per
    visibility check.

    Falls back to the subprocess for anything unusual that does have a
    repository (a worktree whose ``.git`` is a file, a packed ref this
    cannot see), so the token is never weaker than it was — only
    cheaper in the common case.
    """
    git_dir = _git_dir(store)
    if git_dir is None:
        # No repository at all — a depth-1 export, a read-only mount.
        # Returning early matters: the fallback below shells out to `git
        # rev-parse`, which on a directory with no `.git` *fails*, once
        # per visibility check. A subprocess per call to learn something
        # a single `exists()` already told us.
        return (None,)
    try:
        raw = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
    except OSError:
        return (store.head(),)
    if not raw.startswith("ref: "):
        return ("detached", raw)  # already the sha
    ref = git_dir / raw[len("ref: ") :]
    if ref.exists():
        return ("ref", raw, _stamp(ref))
    packed = git_dir / "packed-refs"
    if packed.exists():
        # `git gc --auto` packs refs on any long-lived server-side repo,
        # which used to move it silently onto the subprocess path — 28x
        # slower per check on a corpus this size. A commit writes the
        # ref back out loose, so the token changes shape and rebuilds.
        return ("packed", raw, _stamp(packed))
    # A branch with no commits yet. Nothing to key on, but there IS a
    # repository, which is what `_warn_headless_once` needs to know.
    return (None,)


def _git_dir(store: WikiStore) -> Path | None:
    """The directory holding ``HEAD``, or ``None`` when there is no repo.

    ``.git`` is a *file* in a worktree or a submodule, holding
    ``gitdir: <path>``. Following it keeps those shapes on the cheap
    path instead of a subprocess per visibility check.
    """
    candidate = store.root / ".git"
    if candidate.is_dir():
        return candidate
    if not candidate.is_file():
        return None
    try:
        text = candidate.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not text.startswith("gitdir: "):
        return None
    target = Path(text[len("gitdir: ") :])
    if not target.is_absolute():
        target = store.root / target
    return target if target.is_dir() else None


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
        self._corpus_stamp: tuple[int, int, int] | None = None
        # Per cache, not a module-level set keyed on the path: that set
        # grew without bound, and it let any earlier consumer of the
        # same wiki silence the warning for the next one — including
        # across tests, which is why one had to reach in and clear it.
        self._warned_headless = False

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
            return self._rebuild(store, head, stamp)

    def _rebuild(
        self,
        store: WikiStore,
        head: tuple[object, ...] | None,
        stamp: tuple[tuple[int, int], ...],
    ) -> LabelIndex:
        """Build and store the index. The caller must hold the lock.

        The single place a rebuild happens, because there used to be
        two and they drifted: the headless path was a copy that had
        forgotten the handle release below, so a source restricted by
        another process was re-read from this process's stale snapshot
        and cached under the *new* stamp — fail-open, and permanent,
        since every later rebuild repeated it. One function means the
        next step added here cannot be forgotten by half the callers.
        """
        if self._handle_stamp != stamp:
            # A registry changed since these handles were opened, and
            # `SourceRegistry` is an in-memory snapshot taken at load.
            # Rebuilding from the handle we hold would read the labels
            # this process saw last time — the stale value the stamp
            # just told us not to trust.
            #
            # Keyed on when the HANDLES were loaded, not on the cached
            # index: a first build, or one triggered by HEAD alone,
            # would otherwise reuse a snapshot that predates another
            # process's registry write.
            _release_registries(store)
            self._handle_stamp = stamp
        built = build(store, head)
        self._index = built
        self._headless_built_at = time.monotonic()
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

        Every *page* write commits, and committing needs git, so page
        labels cannot change under such a wiki at all — they move only
        when something outside the process replaces the corpus, which
        in practice means a redeploy and a restart.

        Source labels are the exception, and worth naming because the
        first version of this got it wrong: ``add_source(local=True)``,
        ``add_source(commit=False)`` and ``restrict_source(commit=False)``
        all work without a repository, and all change what a source is
        labelled. Those are caught by the registry fingerprint rather
        than by the clock, so they take effect at once.

        The TTL covers what neither of those does: somebody editing page
        files under a non-git directory in a long-lived process, who
        sees their change within a bounded window rather than never.
        """
        fresh = self._fresh_headless(stamp)
        if fresh is not None:
            return fresh
        with self._lock:
            # Re-checked inside the lock against a FRESH clock read: a
            # thread that waited here may have waited past the TTL, and
            # deciding on the reading it took before the wait would let
            # it serve an index it had just established was too old.
            fresh = self._fresh_headless(stamp)
            if fresh is not None:
                return fresh
            if not self._warned_headless:
                self._warned_headless = True
                log.warning("%s", _headless_warning(store))
            cached = self._index
            # The clock expiring says the index MAY be stale, not that it
            # is. Ask the disk before parsing it: a stat per file against
            # a full YAML parse per file, and on the shape this path
            # exists for — a deployed copy nothing writes to — the answer
            # is always "unchanged", so the steady state is one cheap
            # walk per window rather than one expensive one forever.
            #
            # Only when the REGISTRY is also unchanged. `corpus_stamp`
            # covers `wiki/pages/` and nothing else, so a source whose
            # labels moved would sail past it — which is the same
            # fail-open the two rebuild paths drifting apart produced.
            if cached is not None and cached.registries == stamp:
                corpus = corpus_stamp(store)
                if corpus == self._corpus_stamp:
                    self._headless_built_at = time.monotonic()
                    return cached
                self._corpus_stamp = corpus
            else:
                self._corpus_stamp = corpus_stamp(store)
            return self._rebuild(store, None, stamp)

    def _fresh_headless(
        self, stamp: tuple[tuple[int, int], ...]
    ) -> LabelIndex | None:
        """The cached index if it is still good, else ``None``.

        Returns the value it validated rather than a boolean. Answering
        "yes" and letting the caller re-read ``self._index`` was a race:
        the fast path holds no lock, so `invalidate()` could null the
        field in between and the caller returned ``None`` into every
        visibility check.
        """
        cached = self._index
        if (
            cached is not None
            and cached.registries == stamp
            and time.monotonic() - self._headless_built_at < _HEADLESS_TTL_SECONDS
        ):
            return cached
        return None

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
