"""``WikiStore`` — the public API a consumer reaches for.

The store wires the lower-level modules (:mod:`outmem.git_ops`,
:mod:`outmem.search`, :mod:`outmem.history`, :mod:`outmem.backlinks`,
:mod:`outmem.frontmatter`, :mod:`outmem.identity`, :mod:`outmem.state`)
into a single object scoped to one wiki directory. Downstream
consumers — the bundled CLI, your own FastAPI app, a notebook —
never have to touch the lower-level modules directly.

Mandatory writeback (spec v0.5 §9) is *not* enforced by the store; that
is the agent-runtime's job (phase E). The store exposes ``write_page``,
``extend_page``, and ``append_log`` as primitives — each one commits
exactly once and returns the new HEAD SHA — and the runtime sequences
``pull → think → write → push`` around them.
"""

from __future__ import annotations

import copy
import hashlib
import logging
import threading
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

from outmem._store import import_vault as _import
from outmem._store import semantic as _semantic
from outmem._store import sources as _sources
from outmem._time import ensure_utc, utc_now

if TYPE_CHECKING:
    from outmem.index import PageLoadFailure
    from outmem.lint import ProvenanceAnnotation
    from outmem.semantic import Match, ReindexResult, VectorStore
    from outmem.sources import KeyCandidate, RegistryAudit, RekeyResult, StaleCitation

from outmem._store.labels import LabelCache, LabelIndex
from outmem.backlinks import BacklinkCache
from outmem.completeness import (
    TOOL_SENTINEL_OPEN,
    find_elision_markers,
    find_tool_sentinels,
)
from outmem.config import (
    CONFIG_FILENAME,
    DEFAULT_AGENT_EMAIL,
    DEFAULT_AGENT_NAME,
    DEFAULT_BRANCH,
    DEFAULT_REMOTE,
    DEFAULT_SEMANTIC_REINDEX_CONCURRENCY,
    OutmemConfig,
    load_dotenv_if_present,
    load_yaml_config,
    starter_agents_md,
    starter_yaml,
)
from outmem.exceptions import (
    FrontmatterError,
    IncompleteBodyError,
    OutmemError,
    RestrictionError,
    SlugError,
)
from outmem.frontmatter import (
    ProvenanceEntry,
    WikiFrontmatter,
    parse_wiki_page,
    repair_wiki_page,
    serialize_wiki_page,
    touch_updated,
)
from outmem.git_ops import (
    CommitInfo,
    add,
    clear_stale_index_lock,
    commit_as,
    current_head,
    git_available,
    head_or_none,
    init_repo,
    is_git_repo,
    log_since,
    path_is_dirty,
)
from outmem.git_ops import (
    pull_rebase as _git_pull_rebase,
)
from outmem.git_ops import (
    push as _git_push,
)
from outmem.history import page_history, topic_evolution
from outmem.hooks import ensure_hook
from outmem.identity import Contributors, load_contributors
from outmem.index import (
    AGENTS_FILENAME,
    INDEX_FILENAME,
    INDEX_SLUG,
    IndexLevel,
    editorial_pages,
    index_page_text,
    navigate_index,
)
from outmem.restricted import (
    UNRESTRICTED_GRANTS,
    Grants,
    LabelError,
    RestrictedSettings,
    mode_dirname,
    normalise_labels,
    visible,
    writable,
)
from outmem.search import DEFAULT_RESULT_BYTES, SearchResult, rg_available, search
from outmem.slug import PAGES_DIR, relpath_to_slug, slug_to_relpath, validate_slug
from outmem.sources import (
    REGISTRY_FILENAME,
    SOURCES_DIR,
    SOURCES_LOCAL_DIR,
    IngestionRecord,
    SourceEntry,
    SourceRef,
    SourceRegistry,
    normalize_document_key,
)
from outmem.state import LastRun, OutmemState

log = logging.getLogger(__name__)


def _acknowledgement(
    annotation: ProvenanceAnnotation | None, head: SourceEntry | None
) -> str | None:
    """Why this stale citation is deliberate — if the ack still applies.

    An acknowledgement is **scoped to the version it was made against**,
    exactly as ``finding:`` is. "We deliberately cite the 2024 edition
    while the 2026 one exists" is a statement about those two editions;
    it says nothing about the 2027 one, and reading it as permanent
    would restore the silent staleness the whole feature exists to
    break — one level up, and harder to see, because now a human has
    signed it.

    So it holds only while the acknowledged head is still the head: the
    ack must be dated on or after the day the current version was
    registered. A newer edition moves that date past the ack and the row
    is reported again. Compared as dates rather than instants so an ack
    written the same day a version landed still counts, whatever hour
    each happened.

    Returns None when there is nothing to compare against — a missing
    date (``outmem lint`` names it) or a head that is no longer
    registered. Both are cases where suppressing would be a guess, and a
    guess that hides a stale clinical page is the wrong way to be wrong.
    """
    if annotation is None or annotation.superseded_ok is None:
        return None
    if annotation.date is None or head is None:
        return None
    if annotation.date >= head.registered_at.date():
        return annotation.superseded_ok
    return None


@dataclass(frozen=True)
class WikiPage:
    """A loaded wiki page — frontmatter + body."""

    slug: str
    frontmatter: WikiFrontmatter
    body: str
    path: Path  # absolute path on disk

    @property
    def title(self) -> str:
        return self.frontmatter.title


@dataclass(frozen=True)
class AgentIdentity:
    """The author identity outmem uses for its own commits."""

    name: str = DEFAULT_AGENT_NAME
    email: str = DEFAULT_AGENT_EMAIL


@dataclass
class WikiStoreConfig:
    """Operational config for a :class:`WikiStore` instance.

    Composes the file-loaded :class:`OutmemConfig` (``outmem``) with
    the per-store runtime values that aren't in ``config.yaml`` —
    ``root``, ``agent_identity``, and the resolved ``remote`` /
    ``branch`` after CLI overrides. The directory-layout fields are
    customisable but rarely changed.

    File-loaded settings live under ``store.config.outmem.*`` —
    e.g. ``store.config.outmem.semantic.embedding_model``,
    ``store.config.outmem.git.remove_stale_lock``,
    ``store.config.outmem.model``.
    """

    root: Path
    outmem: OutmemConfig = field(default_factory=OutmemConfig)
    agent_identity: AgentIdentity = field(default_factory=AgentIdentity)
    remote: str = DEFAULT_REMOTE
    branch: str = DEFAULT_BRANCH
    wiki_dir: str = "wiki"
    log_dir: str = "log"
    contributors_file: str = "CONTRIBUTORS.md"
    # When True, every commit-producing entry point on :class:`WikiStore`
    # refuses via a single guard in :meth:`WikiStore._commit_paths`. Used
    # by downstream consumers that want to attach a curated wiki to an
    # external agent system as a read-only tool (see
    # :func:`outmem.adapters.pydantic_ai.build_consult_wiki`).
    read_only: bool = False


def _require_external_binaries() -> None:
    """Raise :class:`OutmemError` if ``git`` or ``rg`` is missing.

    Both are runtime requirements for any wiki operation — every commit
    needs ``git``, every search needs ``rg``. Checked at ``init`` so the
    user gets a clear actionable error before any directories are
    created, rather than a cryptic subprocess failure later.
    """
    missing: list[str] = []
    if not git_available():
        missing.append("git")
    if not rg_available():
        missing.append("rg (ripgrep)")
    if missing:
        names = " and ".join(missing)
        raise OutmemError(
            f"outmem requires {names} on PATH. Install with your OS package "
            "manager (e.g. `brew install git ripgrep` or `apt install git ripgrep`) "
            "and retry."
        )


def _seed_config_files(root: Path, *, agent_identity: AgentIdentity) -> None:
    """Drop a starter ``config.yaml`` at the wiki root.

    Idempotent — does not overwrite an existing file. ``.env`` is
    *not* seeded here; it lives at the user's project root (CWD), and
    :func:`load_dotenv_if_present` walks upward from CWD to find it.
    """
    yaml_path = root / CONFIG_FILENAME
    if yaml_path.exists():
        return
    yaml_path.write_text(
        starter_yaml(
            agent_name=agent_identity.name,
            agent_email=agent_identity.email,
        ),
        encoding="utf-8",
    )


def _config_from_yaml(
    root: Path,
    *,
    agent_identity: AgentIdentity | None,
    remote: str | None,
    branch: str | None,
    read_only: bool = False,
) -> WikiStoreConfig:
    """Resolve a :class:`WikiStoreConfig` from ``config.yaml`` + overrides.

    Explicit constructor args win; otherwise values come from the
    YAML; otherwise the built-in defaults.

    ``load_dotenv()`` is fired here with no path argument — it walks
    upward from CWD looking for ``.env`` (the standard
    :mod:`python-dotenv` behaviour). That puts ``.env`` at the user's
    project root, not the wiki root, which is the typical layout:
    the wiki holds data, the project holds secrets and code.
    """
    load_dotenv_if_present()
    yaml_cfg: OutmemConfig = load_yaml_config(root)

    if agent_identity is None:
        agent_identity = AgentIdentity(
            name=yaml_cfg.agent.name,
            email=yaml_cfg.agent.email,
        )

    return WikiStoreConfig(
        root=root,
        outmem=yaml_cfg,
        agent_identity=agent_identity,
        remote=remote or yaml_cfg.remote.name,
        branch=branch or yaml_cfg.remote.branch,
        read_only=read_only,
    )


# ---------------------------------------------------------------------------
# Access-control classification of the public surface
#
# Every public member of ``WikiStore`` belongs to exactly one of these
# four sets, and ``tests/test_restricted_reads.py`` fails until a newly
# added one is placed. That failure is the mechanism: coverage of a
# boundary is a property of the whole surface, not of the paths someone
# remembered to test, and a method added later with no filtering is
# precisely how a boundary like this rots.
#
# Adding a member means answering one question — can this return item
# content, identity, or existence to a viewer? — and the four answers
# are: filter it, gate it on the write rules, refuse it to views
# outright, or nothing to do.
# ---------------------------------------------------------------------------

# How far semantic retrieval widens its fetch when filtering has left it
# short of k, and where it stops. The ceiling exists so a query whose
# whole neighbourhood is restricted cannot walk the entire index.
_OVERFETCH_FACTOR = 4
_OVERFETCH_CEILING = 200

def _write_refusal(
    mode: frozenset[str], labels: frozenset[str], *, new: bool
) -> str:
    """The message for a refused write.

    Says which way the mismatch runs, because the two directions call
    for opposite fixes and "denied" leaves the caller guessing. Names
    only labels — never the item, and never anything about what else
    lives in the compartment.
    """
    shown_mode = ", ".join(sorted(mode)) or "open"
    shown_labels = ", ".join(sorted(labels)) or "open"
    if not labels <= mode:
        return (
            f"refusing to write: the target is restricted to [{shown_labels}] "
            f"but this session is scoped to [{shown_mode}]. Start a session in "
            "that compartment."
        )
    return (
        f"refusing to write: this session is scoped to [{shown_mode}] and the "
        f"{'new page would be' if new else 'target is'} [{shown_labels}]. "
        "Writing down from a restricted session into less restricted content "
        "is not allowed; work in an open session for open content."
    )


# Commit-subject grammar, as produced by the `_commit_paths` call sites.
# `steering` recovers the item a commit is about from its subject, so
# these have to stay in step with what gets committed; a verb missing
# here falls through to "shown", and the pathspec narrowing in
# `_steering_paths` is what covers the ones whose subject names no item.
_SLUG_SUBJECT_VERBS = frozenset({"compact", "extend", "append", "rename", "restrict"})
_SOURCE_SUBJECT_VERBS = frozenset({"source", "ingest"})

_VISIBILITY_ENFORCED = frozenset({
    "backlinks",
    "compartment_hint",
    "exists",
    "get_source",
    "index_tree",
    "list_slugs",
    "list_sources",
    "provenance_annotations",
    "provenance_findings",
    "read",
    "read_agents_md",
    "read_source",
    "resolve_slug",
    "search",
    "semantic_find_similar",
    "source_citations",
    "source_refs",
    "steering",
    "unreadable",
})
"""Returns content, identity or existence — filtered for the viewer.

Each has a test in ``tests/test_restricted_reads.py`` showing a
restricted item absent from its result.
"""

_WRITE_ENFORCED = frozenset({
    "append_log",
    "append_page",
    "extend_page",
    "record_ingestion",
    "write_page",
})
"""Produces a commit — subject to the write rule (§3.2) and closure."""

_OPERATOR_ONLY = frozenset({
    "add_source",
    "rename_page",
    "restrict_page",
    "restrict_source",
    "assign_document_keys",
    "commit_registry",
    "ensure_sources_local",
    "evolution",
    "history",
    "import_vault",
    "propose_document_keys",
    "rebuild_index",
    "record_source_refs",
    "rekey_document",
    "repair_pages",
    "semantic_reindex_all",
    "semantic_reindex_path",
    "semantic_remove_path",
    "sources_gc",
    "stale_pages",
})
"""Refused to a view entirely — the cheapest kind of safety.

Two reasons appear here. The history readers (``history``,
``evolution``) are answered by git, which knows nothing about labels,
while the label index describes only the current commit: a page
restricted today was open last month and its old bodies are still in
the diff stream. Everything else is bulk maintenance that walks the
whole wiki by design and that a served request has no business
reaching.
"""

_NO_CONTENT = frozenset({
    "allow_elision_body",
    "as_viewer",
    "corpus_token",
    "close",
    "contributors",
    "enforces_visibility",
    "grants",
    "head",
    "init",
    "is_page_path",
    "last_run",
    "mode",
    "open",
    "pages_prefix",
    "pull",
    "push",
    "record_run",
    "restrictions",
    "semantic_available",
    "semantic_index_is_empty",
})
"""Returns no item content, identity, or existence. Nothing to filter."""


@dataclass
class _LazyResources:
    """The store's lazily-opened, reassignable slots, in one place.

    Shared by reference across every view (see
    :meth:`WikiStore.as_viewer`). Not a cache in the invalidation sense
    — these are handles, and a view holding its own copy of one is both
    a resource leak and, for the registries, a correctness bug.
    """

    source_registry: SourceRegistry | None = None
    source_registry_local: SourceRegistry | None = None
    """Separate handle: each source tree carries its own registry, so the
    tracked one never records a local source's filename, hash, or origin
    path."""
    vector_store: VectorStore | None = None
    contributors: Contributors | None = None
    alias_map: dict[str, str] | None = None


class WikiStore:
    """Filesystem-backed wiki — the unit downstream code interacts with."""

    # Property pairs over `_lazy`, so every existing `store._x` /
    # `store._x = y` call site keeps working while the value itself
    # lives on the shared object.
    @property
    def _source_registry(self) -> SourceRegistry | None:
        return self._lazy.source_registry

    @_source_registry.setter
    def _source_registry(self, value: SourceRegistry | None) -> None:
        self._lazy.source_registry = value

    @property
    def _source_registry_local(self) -> SourceRegistry | None:
        return self._lazy.source_registry_local

    @_source_registry_local.setter
    def _source_registry_local(self, value: SourceRegistry | None) -> None:
        self._lazy.source_registry_local = value

    @property
    def _vector_store(self) -> VectorStore | None:
        return self._lazy.vector_store

    @_vector_store.setter
    def _vector_store(self, value: VectorStore | None) -> None:
        self._lazy.vector_store = value

    @property
    def _contributors(self) -> Contributors | None:
        return self._lazy.contributors

    @_contributors.setter
    def _contributors(self, value: Contributors | None) -> None:
        self._lazy.contributors = value

    @property
    def _alias_map(self) -> dict[str, str] | None:
        return self._lazy.alias_map

    @_alias_map.setter
    def _alias_map(self, value: dict[str, str] | None) -> None:
        self._lazy.alias_map = value

    def __init__(self, config: WikiStoreConfig) -> None:
        self.config = config
        self.root = Path(config.root)
        self.wiki_path = self.root / config.wiki_dir
        self.pages_path = self.wiki_path / PAGES_DIR
        self.log_path = self.root / config.log_dir
        self.sources_path = self.wiki_path / SOURCES_DIR
        # Untracked sibling of ``sources/`` for material that may be read
        # but not redistributed (licensed / copyrighted / embargoed). Its
        # own registry lives inside it, so the tracked registry never
        # records a local source's filename, hash, or origin path.
        self.sources_local_path = self.wiki_path / SOURCES_LOCAL_DIR
        self.contributors_path = self.root / config.contributors_file
        self.agents_path = self.wiki_path / AGENTS_FILENAME
        self.state = OutmemState(self.root)
        self.backlinks_cache = BacklinkCache(
            state=self.state,
            wiki_dir=self.wiki_path,
            pages_dir=self.pages_path,
            read_only=config.read_only,
        )
        # Slugs already warned-about by read()'s frontmatter self-heal, so
        # repeated reads of one broken page log once, not per read.
        self._healed_slugs: set[str] = set()
        # Every lazily-opened resource lives behind one shared object.
        #
        # `as_viewer` is a shallow copy, which gives the view its own
        # `__dict__` — so a slot the view later *assigns* diverges from
        # the store's, and both then hold their own SQLite connections.
        # For the source registries that is fail-open: a source
        # registered after the view first touched sources is absent from
        # the view's snapshot, so the label index never sees it and
        # `for_source` reads it as unlabelled. Holding the mutable slots
        # in one object means the copy shares the reference and every
        # view sees the same registry, the same vector store, and the
        # same connections.
        self._lazy = _LazyResources()
        # Guards lazy VectorStore open — the optimize tool queries
        # concurrently across a thread pool, so the check-then-open must be
        # atomic or 8 threads each build an embedder + orphan 7 connections.
        self._vector_store_lock = threading.Lock()
        # Serialises every read-modify-write-commit on the wiki. Page
        # writes are not atomic on their own: each reads the current
        # body, rewrites the file, regenerates the index, and commits.
        # PydanticAI runs a response's tool calls concurrently, and the
        # write guidance is explicitly "one `append_page` per section",
        # so this is the ordinary path, not an exotic one. Unguarded,
        # four parallel appends lose sections outright and raise
        # `cannot lock ref 'HEAD'` / half-read-file FrontmatterError —
        # silent content loss, which is the exact failure the
        # completeness work exists to prevent.
        #
        # Re-entrant because these methods legitimately nest (a write
        # path calling another guarded helper must not deadlock).
        self._write_lock = threading.RLock()
        # Body texts pre-authorised past the elision guard. The guard is
        # a positional heuristic and therefore fallible, so every caller
        # needs a way to say "I looked, this text is right": the tool
        # wrapper adds a body the model re-sent unchanged, the approval
        # gate adds one a human reviewer edited and approved, and the CLI
        # has `--allow-elision`. Keyed by the text itself, not by slug —
        # an adjudication is about the words, and an alias must not make
        # the same decision be taken twice.
        self._elision_allowed: set[str] = set()
        # Access control. ``None`` means no view has been taken: this is
        # the bare server-side store and nothing is filtered. A view
        # created by :meth:`as_viewer` carries a real (possibly empty)
        # set, and every enforced path checks against it. The
        # distinction matters — an empty mode is a *restricted* session
        # that sees open content only, which is not the same thing as an
        # unrestricted operator.
        self._mode: frozenset[str] | None = None
        self._grants: Grants = UNRESTRICTED_GRANTS
        # Shared by every view derived from this store: views are
        # per-request, the corpus walk that resolves labels is not.
        self._label_cache = LabelCache()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def open(
        cls,
        path: str | Path,
        *,
        agent_identity: AgentIdentity | None = None,
        remote: str | None = None,
        branch: str | None = None,
        read_only: bool = False,
    ) -> WikiStore:
        """Open an existing wiki at ``path``.

        Reads ``config.yaml`` and ``.env`` from the wiki root for
        defaults (model, agent identity, git resilience settings,
        remote). Explicit kwargs override the YAML; the YAML overrides
        the built-in defaults. ``.env`` is loaded into ``os.environ``
        without overriding pre-existing values.

        Creates the subdirectories (``wiki/pages/``, ``wiki/sources/``,
        ``log/``, ``.outmem/``) if they don't yet exist. Does not
        initialise a git repo — :meth:`init` is the explicit
        constructor for that.
        If a stale ``.git/index.lock`` is present and the user's
        ``config.yaml`` enables ``git.remove_stale_lock``, it gets
        cleaned up here.

        ``read_only=True`` flips the store into a refusing-to-mutate
        mode:

        * Every commit-producing entry point (``write_page``,
          ``extend_page``, ``append_log``, ``add_source``,
          ``record_ingestion``, ``rebuild_index``, ``import_vault``)
          raises :class:`OutmemError` via a single guard in
          :meth:`_commit_paths`.
        * ``pull()`` is also refused — ``git pull --rebase`` would
          mutate the working tree.
        * The directory-creating layout step is skipped, the stale
          ``.git/index.lock`` cleanup is skipped, and
          :class:`~outmem.backlinks.BacklinkCache` runs memo-only
          (no writes to ``.outmem/``). The wiki's filesystem state
          is left exactly as the caller found it, which makes the
          mode safe to use on a literally read-only mount.

        Use this when handing a curated wiki to an external agentic
        system that should only consult it. See
        :func:`outmem.adapters.pydantic_ai.build_consult_wiki` for
        the ergonomic one-call factory.
        """
        root = Path(path).expanduser()
        if not root.exists():
            raise OutmemError(f"Wiki root does not exist: {root}")
        config = _config_from_yaml(
            root,
            agent_identity=agent_identity,
            remote=remote,
            branch=branch,
            read_only=read_only,
        )
        store = cls(config)
        if not read_only:
            store._ensure_layout()
            store._maybe_clear_stale_lock()
            store._maybe_auto_install_hook()
        return store

    @classmethod
    def init(
        cls,
        path: str | Path,
        *,
        agent_identity: AgentIdentity | None = None,
        remote: str | None = None,
        branch: str | None = None,
    ) -> WikiStore:
        """Create a new wiki at ``path``.

        Creates the directory, initialises a git repo on ``branch``,
        writes a starter ``CONTRIBUTORS.md`` if one does not exist,
        scaffolds ``wiki/pages/``, ``wiki/sources/``, ``log/``,
        ``.outmem/``, seeds
        ``config.yaml`` (machine config) and ``wiki/AGENTS.md`` (the
        user-editable wiki-conventions doc that gets loaded into the
        agent's system prompt every turn). ``.env`` is gitignored by
        default.

        Pre-flight: requires ``git`` and ``rg`` (ripgrep) on PATH.
        Both are runtime dependencies of every wiki operation; catching
        their absence here gives a clear error before any directories
        get created.
        """
        _require_external_binaries()
        root = Path(path).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        init_repo(root, initial_branch=branch or DEFAULT_BRANCH)
        # Seed config before resolving it so the yaml exists for read.
        _seed_config_files(root, agent_identity=agent_identity or AgentIdentity())
        config = _config_from_yaml(
            root, agent_identity=agent_identity, remote=remote, branch=branch
        )
        store = cls(config)
        store._ensure_layout()
        store._seed_contributors()
        store._seed_agents_md()
        store._maybe_ignore_dotenv()
        store._maybe_auto_install_hook()
        return store

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def read(self, slug: str) -> WikiPage:
        """Load the wiki page for ``slug`` into a :class:`WikiPage`.

        The on-disk path is ``wiki/pages/<slug-as-relpath>.md`` (see
        :func:`outmem.slug.slug_to_relpath`). The auto-generated index
        lives at ``wiki/index.md`` and is fetched via the special
        ``index`` slug.

        Self-heals frontmatter that won't parse but is mechanically
        fixable — the imported-data case where a title contains an
        unquoted ``: `` (see :func:`outmem.frontmatter.repair_wiki_page`).
        The repair is applied **in memory** and logged at WARNING (naming
        the page), so callers like ``generate_bank`` and the agent's
        retrieval tools get usable content instead of silently dropping
        the page — without ``read`` taking on a surprise disk write. The
        on-disk file is persisted by the pre-commit hook (next commit) or
        :meth:`repair_pages`.

        Raises :class:`OutmemError` if the page does not exist;
        :class:`outmem.exceptions.FrontmatterError` if frontmatter is
        missing or malformed in a way the repair doesn't cover.
        """
        slug = self.resolve_slug(slug)
        path = self._page_path(slug)
        if not path.exists():
            raise OutmemError(f"No such wiki page: {slug}")
        # After the existence check and before any content is read, so a
        # hidden page and an absent one are indistinguishable from here.
        if not self._page_visible(slug):
            self._no_such_page(slug)
        if slug == INDEX_SLUG and self.enforces_visibility:
            # wiki/index.md on disk catalogues every page. Rendered live
            # from what this view can see instead — a stored catalogue is
            # a list of slugs the filter never gets to touch.
            return self._visible_index_page(path)
        text = path.read_text(encoding="utf-8")
        try:
            frontmatter, body = parse_wiki_page(text, fallback_slug=slug)
        except FrontmatterError:
            repaired = repair_wiki_page(text)
            if repaired is None:
                raise  # not a shape we can mend — surface it loudly
            # Warn ONCE per slug per process: an eval reads a candidate page
            # for many questions, so logging every read floods the console
            # with the same line dozens of times. One warning is enough to
            # tell the user to persist the fix.
            if slug not in self._healed_slugs:
                self._healed_slugs.add(slug)
                log.warning(
                    "self-healed unparseable frontmatter in %r (in memory; "
                    "persist with `store.repair_pages(dry_run=False)` or the "
                    "pre-commit hook)",
                    slug,
                )
            frontmatter, body = parse_wiki_page(repaired, fallback_slug=slug)
        return WikiPage(slug=slug, frontmatter=frontmatter, body=body, path=path)

    def exists(self, slug: str) -> bool:
        """Whether a page lives at ``slug``. ``False`` for a hidden one:
        an honest ``True`` here is an existence oracle that costs the
        caller nothing to query."""
        try:
            resolved = self.resolve_slug(slug)
            if not self._page_path(resolved).exists():
                return False
        except SlugError:
            return False
        return self._page_visible(resolved)

    def list_slugs(self) -> list[str]:
        """Every editorial slug under ``wiki/pages/``, alphabetically.

        The auto-generated ``index`` slug is hidden — it's structural,
        not content. Consumers who need to read it can still call
        ``read("index")`` directly.

        Slugs that :meth:`read` would reject are skipped, so the catalogue
        never advertises a page it can't open — a filename with uppercase
        or an underscore derives a grammatically invalid slug, and handing
        that to an agent costs it a tool call to find out. Use
        :meth:`unreadable` to see what was skipped and why.
        """
        if not self.pages_path.is_dir():
            return []
        from outmem.slug import relpath_to_slug

        out: list[str] = []
        for path in editorial_pages(self.pages_path):
            slug = relpath_to_slug(path.relative_to(self.pages_path))
            try:
                validate_slug(slug)
            except SlugError:
                continue
            out.append(slug)
        if self.enforces_visibility:
            index = self._labels()
            out = [s for s in out if self._page_visible(s, index)]
        return sorted(out)

    def _alias_index(self) -> dict[str, str]:
        """Alias → canonical slug, built lazily and cached.

        Only consulted on a resolution *miss* (see :meth:`resolve_slug`),
        so the hot paths — every loop over ``list_slugs()`` calling
        ``read`` — never pay for the corpus walk that builds it.
        """
        if self._alias_map is None:
            from outmem.index import alias_index

            self._alias_map = alias_index(self.pages_path)
        return self._alias_map

    def resolve_slug(self, slug: str) -> str:
        """The canonical slug for ``slug``, following an alias if needed.

        Returns ``slug`` unchanged when a page lives there — checking the
        file **first** is what guarantees a stale alias can never shadow a
        live page, and is also what keeps the alias map off the hot path.

        Resolution happens here, at the API boundary, rather than inside
        the path builders: ``_page_relpath``, ``history._slug_relpath``
        and the semantic ``exclude_slug`` all derive paths without going
        through ``_page_path``, so hooking that one function would leave
        ``extend_page`` reading the aliased page, writing the canonical
        file, and then staging a path that doesn't exist.
        """
        if slug == INDEX_SLUG:
            return slug
        try:
            validate_slug(slug)
        except SlugError:
            return slug  # let the caller raise its own error
        if (self.pages_path / slug_to_relpath(slug)).exists():
            return slug
        canonical = self._alias_index().get(slug, slug)
        if canonical != slug and not self._page_visible(canonical):
            # Following the alias would hand back the canonical slug of a
            # page this view cannot see — the name itself is the leak.
            # Returning the input unchanged is what happens for an alias
            # that does not exist.
            return slug
        return canonical

    def unreadable(self) -> list[tuple[str, str]]:
        """Every page under ``wiki/pages/`` that isn't cleanly addressable.

        Returns ``(slug, reason)`` — the load-time audit a consumer needs
        to know its wiki is sound, without attempting :meth:`read` on
        every slug (O(n) file reads to learn something one walk can tell
        you). Covers three defects:

        - a filename whose derived slug fails the slug grammar, so it is
          omitted from :meth:`list_slugs` and unopenable by that name
        - a page whose frontmatter won't parse at all
        - a page whose declared ``slug:`` disagrees with its path — the
          silent one. It reads fine by path, but the same page then has
          two names and the declared one resolves to nothing. Reported,
          never fatal: the page stays available.

        A clean wiki returns ``[]``, which is the assertion a downstream
        consumer wants in its own test suite.
        """
        if not self.pages_path.is_dir():
            return []
        from outmem.index import load_editorial_pages
        from outmem.slug import relpath_to_slug

        out: list[tuple[str, str]] = []
        index = self._labels()
        for path in editorial_pages(self.pages_path):
            slug = relpath_to_slug(path.relative_to(self.pages_path))
            try:
                validate_slug(slug)
            except SlugError as exc:
                out.append((slug, f"not addressable: {exc}"))
        pages, failures = load_editorial_pages(self.pages_path)
        for failure in failures:
            out.append((failure.slug, f"does not parse: {failure.error}"))
        for page in pages:
            if page.slug_mismatch:
                out.append(
                    (
                        page.slug,
                        f"declares slug {page.declared_slug!r} but lives at "
                        f"{page.slug!r} — the declared name resolves to nothing "
                        "(usually a `git mv` that didn't update the frontmatter)",
                    )
                )
        if self.enforces_visibility:
            # An unparseable page resolves to DENY, so it drops out here
            # for every mode — which is the point: its labels could not be
            # read, so it cannot be shown to anyone.
            out = [row for row in out if self._page_visible(row[0], index)]
        return sorted(set(out))

    def index_tree(self, prefix: str = "", *, titles: bool = False) -> IndexLevel:
        """Navigate the slug index (the TOC) one namespace level at a time.

        Groups :meth:`list_slugs` by the ``:`` namespace separator via
        :func:`outmem.index.navigate_index`. ``prefix=""`` returns the
        root level; pass a namespace from ``IndexLevel.namespaces`` back
        as ``prefix`` to drill in.

        ``titles=True`` fills :attr:`IndexLevel.titles` with the
        frontmatter title of each page at this level. Opt-in because it
        costs a parse per page, where the rest is a directory walk — but
        worth reaching for, since a browsing surface that has to fetch
        titles some other way ends up walking ``wiki/pages/`` itself and
        building its own slug map, which is how a consumer's addressing
        silently falls behind the library's (:mod:`outmem.testing`).

        Only the pages *at this level* are parsed, not the whole wiki —
        drilling into a namespace is a per-click operation, and paying
        for every page on each click makes the cost scale with the wiki
        rather than with what is on screen. A page whose frontmatter will
        not parse is absent from ``titles`` rather than blocking the
        level; :meth:`unreadable` says which and why.
        """
        level = navigate_index(self.list_slugs(), prefix)
        if not titles or not level.pages:
            return level
        import dataclasses

        from outmem.index import load_page_text

        found: dict[str, str] = {}
        for slug in level.pages:
            path = self.pages_path / slug_to_relpath(slug)
            try:
                frontmatter, _body, _repaired = load_page_text(
                    path.read_text(encoding="utf-8"), fallback_slug=slug
                )
            except (OSError, UnicodeDecodeError, FrontmatterError):
                continue
            found[slug] = frontmatter.title
        # Replace rather than mutate: IndexLevel is frozen, and filling a
        # dict in place would make that promise a lie to anyone holding
        # the earlier reference.
        return dataclasses.replace(level, titles=found)

    def repair_pages(
        self, *, dry_run: bool = True, commit_subject: str | None = None
    ) -> list[tuple[str, str]]:
        """Walk every wiki page; repair the ones whose frontmatter won't parse.

        Targets the imported-data failure mode where a top-level scalar
        value contains an unquoted ``: `` (colon-space) and YAML reads it
        as a malformed nested mapping. See
        :func:`outmem.frontmatter.repair_wiki_page` for the exact shape
        repaired. Returns ``[(slug, summary), …]`` for every page touched
        (or that would be touched, with ``dry_run=True``); pages already
        parsing — or broken in a way the repair doesn't address — are
        silently skipped.

        ``dry_run=True`` (default) reports only — call again with
        ``dry_run=False`` to write the fixes back and commit them as one
        ``fix: repair frontmatter…`` commit (``commit_subject`` overrides
        the subject). Read-only stores refuse the write step.
        """
        self._require_operator('repairing pages')
        from outmem.slug import relpath_to_slug

        repaired: list[tuple[str, str]] = []
        if not self.pages_path.is_dir():
            return repaired
        for path in editorial_pages(self.pages_path):
            slug = relpath_to_slug(path.relative_to(self.pages_path))
            text = path.read_text(encoding="utf-8")
            fixed = repair_wiki_page(text)
            if fixed is None:
                continue
            summary = "quoted scalar values containing ': '"
            if not dry_run:
                path.write_text(fixed, encoding="utf-8")
            repaired.append((slug, summary))
        if not dry_run and repaired:
            rels = [str(self._page_path(s).relative_to(self.root)) for s, _ in repaired]
            subject = commit_subject or (
                f"fix: repair frontmatter on {len(repaired)} page(s)"
            )
            self._commit_paths(rels, subject=subject)
        return repaired

    def search(
        self,
        pattern: str,
        *,
        scope: str = "wiki",
        case_insensitive: bool = False,
        fixed_strings: bool = False,
        context: int = 0,
        max_bytes: int = DEFAULT_RESULT_BYTES,
        max_hits: int | None = None,
    ) -> SearchResult:
        """Run a ripgrep search anchored at the store.

        ``scope`` is one of ``"wiki"``, ``"sources"``, ``"log"``, or
        ``"all"``. Tier 1 of the agent's retrieval palette is
        ``scope="wiki"``; Tier 2 falls through to ``"sources"``, which
        spans both the tracked ``wiki/sources/`` tree and the untracked
        ``wiki/sources-local/`` one.

        ``context`` requests N lines either side of each match (``rg
        -C``). Those rows arrive as :class:`SearchHit` with
        ``is_match=False``.
        """
        path, paths = self._resolve_scope(scope)
        result = search(
            pattern,
            root=path,
            paths=paths,
            case_insensitive=case_insensitive,
            fixed_strings=fixed_strings,
            context=context,
            max_bytes=max_bytes,
            max_hits=max_hits,
        )
        return self._filter_hits(result, scope)

    def backlinks(self, slug: str) -> tuple[str, ...]:
        """Slugs of pages that link to ``slug`` at the current HEAD.

        Filtered for the viewer, and empty for a target the viewer
        cannot see: the referrer list is a list of slugs, so an
        unfiltered answer would name hidden pages, and answering at all
        about a hidden target would confirm it exists.
        """
        slug = self.resolve_slug(slug)
        validate_slug(slug)
        if not self._page_visible(slug):
            return ()
        refs = self.backlinks_cache.referrers(slug, head_or_none(self.root))
        if not self.enforces_visibility:
            return refs
        index = self._labels()
        return tuple(r for r in refs if self._page_visible(r, index))

    def history(self, slug: str) -> list[CommitInfo]:
        """Per-page commit history (newest first), tracking renames.

        Operator-only. History is answered by git, which knows nothing
        about labels, and the label index only describes the current
        commit — a page restricted today was open last month, and its
        old commits are still there. Filtering it correctly would mean
        resolving labels at every commit in the range.
        """
        self._require_operator("page history")
        slug = self.resolve_slug(slug)
        validate_slug(slug)
        return page_history(self.root, slug, wiki_dir=self.config.wiki_dir)

    def evolution(
        self,
        slugs: Sequence[str],
        *,
        include_log: bool = True,
    ) -> str:
        """Raw ``git log -p`` stream — the EXPANSION-pattern helper.

        Operator-only, for the same reason as :meth:`history` and with
        more at stake: this returns diff *bodies*, so a page restricted
        today would hand over the text it had while it was open.
        """
        self._require_operator("topic evolution")
        slugs = [self.resolve_slug(s) for s in slugs]
        return topic_evolution(
            self.root,
            slugs,
            wiki_dir=self.config.wiki_dir,
            include_log=include_log,
            log_dir=self.config.log_dir,
        )

    def steering(
        self,
        *,
        since: datetime | str | None = None,
        include_log: bool = True,
        default_window: str = "30 days ago",
    ) -> list[CommitInfo]:
        """Phase-1 steering signal: human commits since ``since``.

        Excludes the agent's own commits via :func:`git_ops.log_since`'s
        ``exclude_author`` filter. If ``since`` is ``None`` the last-run
        marker is consulted; when no marker exists the lookback is
        bounded by ``default_window`` (a string ``git log --since``
        understands) so the first run doesn't dump every non-agent
        commit ever made into the agent's context.

        Filtered for the viewer by commit subject. This matters more
        than it looks: the steering signal is rendered into the *system
        prompt*, so an unfiltered ``compact: hr:severance-policy``
        reaches every request regardless of who is asking — the one leak
        in the read surface that no tool call is needed to trigger.
        Filtering on the mode rather than the user keeps the system
        prompt identical for everyone in one compartment, which is what
        prompt caching needs.
        """
        if head_or_none(self.root) is None:
            # No commits yet; nothing to steer on.
            return []
        if since is None:
            marker = self.state.last_run()
            since = marker.timestamp if marker else default_window
        paths = self._steering_paths(include_log=include_log)
        if not paths:
            return []
        commits = log_since(
            self.root,
            since=since,
            paths=paths,
            exclude_author=self.config.agent_identity.email,
        )
        if not self.enforces_visibility:
            return commits
        index = self._labels()
        return [c for c in commits if self._subject_visible(c.subject, index)]

    def _subject_visible(self, subject: str, index: LabelIndex) -> bool:
        """Whether a commit subject may be shown to this viewer.

        outmem's own subjects are ``<verb>: <slug-or-path>``, so the
        slug is recoverable without parsing the diff. A subject in any
        other shape is a human's own commit message and is shown as-is —
        it names no item this mechanism can resolve, and lint's
        ``restricted-slug-mentioned`` check is what covers prose.
        """
        verb, sep, rest = subject.partition(": ")
        if not sep:
            # Not outmem's grammar at all — a human's own commit message.
            # It names no item this mechanism can resolve, and lint's
            # `restricted-slug-mentioned` check is what covers prose.
            return True
        target = rest.strip()
        if verb in _SLUG_SUBJECT_VERBS:
            # `rename: old -> new` names two slugs; either being hidden
            # hides the commit.
            return all(
                self._can_see(index.for_page(part.strip()))
                for part in target.replace("->", " ").split()
            )
        if verb in _SOURCE_SUBJECT_VERBS:
            return self._can_see(index.for_source(target))
        # Everything else — `log:`, `fix:`, `import:`, `index:` — carries
        # free text somebody wrote rather than a name this can resolve.
        # Those are handled by narrowing what git walks (see
        # `_steering_paths`) rather than by parsing the subject, because
        # a subject cannot be trusted to say which item it is about.
        return True

    def _steering_paths(self, *, include_log: bool) -> list[str]:
        """Which trees ``steering`` lets git walk for this viewer.

        Page and source commits are filtered afterwards by subject,
        because their subjects name the item. Log commits cannot be: the
        rest of a `log:` subject is a topic somebody typed, so
        `log: severance cap` written in a restricted session would
        otherwise reach the system prompt of every open one. Narrow the
        pathspec instead — a partition the viewer cannot see is simply
        not walked, so no subject from it exists to filter.
        """
        paths = [self.config.wiki_dir]
        if not include_log:
            return paths
        if not self.enforces_visibility:
            paths.append(self.config.log_dir)
            return paths
        # The open partition is `log/<date>.md`, one level down; the
        # restricted ones are `log/<label-set>/<date>.md`.
        paths.extend(
            str(path.relative_to(self.root).as_posix())
            for path in sorted(self.log_path.glob("*.md"))
        )
        for directory in sorted(p for p in self.log_path.glob("*") if p.is_dir()):
            labels = self.restrictions.mode_from_log_dirname(directory.name)
            if labels is not None and self._can_see(labels):
                paths.append(str(directory.relative_to(self.root).as_posix()))
        return paths

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write_page(
        self,
        slug: str,
        *,
        title: str,
        body: str,
        provenance: Sequence[ProvenanceEntry] | None = None,
        tags: Sequence[str] | None = None,
        created: datetime | None = None,
        extra: dict[str, Any] | None = None,
        commit_subject: str | None = None,
        allow_elision: bool = False,
    ) -> str:
        """Create a new wiki page (under ``wiki/pages/``) and commit it.

        The on-disk path is derived from the slug by
        :func:`outmem.slug.slug_to_relpath` (``:`` → ``/``,
        appending ``.md``). Frontmatter is built per spec v0.5 §4.
        The commit message defaults to ``compact: <slug>`` (TARS Retained
        depends on the prefix grammar — see spec §9).
        ``wiki/index.md`` is regenerated and staged in the same commit.
        """
        with self._write_lock:
            if slug == INDEX_SLUG:
                raise OutmemError(
                    "Cannot write to the reserved 'index' slug — `wiki/index.md` "
                    "is auto-maintained by outmem on every page write."
                )
            page_path = self._page_path(slug)
            if page_path.exists():
                raise OutmemError(f"Page already exists: {slug}. Use extend_page() to edit it.")
            owner = self._alias_index().get(slug)
            if owner is not None:
                # Writing here would succeed (no file at that path) and then win
                # resolution file-first, silently retargeting every [[slug]] in
                # the corpus from `owner` to this new stub. Lint would report it
                # afterwards, by which point the links have changed meaning.
                if not self._page_visible(owner):
                    # Naming the owner would hand a hidden slug to
                    # whoever guessed this one, and `rename_page` leaves
                    # an alias behind on every move — so guessing an old
                    # name would return the new, hidden one. The bounded
                    # oracle §8.1 accepts is "something blocks this
                    # slug", not "and here is what".
                    raise OutmemError(
                        f"{slug!r} cannot be used: another page already answers "
                        "to that name. Choose another slug."
                    )
                raise OutmemError(
                    f"{slug!r} is an alias of {owner!r}; writing a page here would "
                    f"silently retarget every [[{slug}]] link. Remove the alias from "
                    f"{owner!r} first, or choose another slug."
                )
            extra_fields = dict(extra or {})
            declared = extra_fields.pop("restricted", None)
            labels = self._check_write(
                slug,
                new=True,
                body=body,
                provenance=provenance,
                declared=declared,
            )
            now = utc_now()
            frontmatter = WikiFrontmatter(
                title=title,
                slug=slug,
                provenance=list(provenance or []),
                created=(created or now).replace(microsecond=0),
                updated=now,
                tags=list(tags or []),
                restricted=sorted(labels)
                if self.enforces_visibility
                else sorted(normalise_labels(declared)),
                extra=extra_fields,
            )
            if not allow_elision:
                _reject_incomplete_body(
                body, tool="write_page", allowed=self._elision_allowed
            )
            page_text = serialize_wiki_page(frontmatter, body)
            page_path.parent.mkdir(parents=True, exist_ok=True)
            page_path.write_text(page_text, encoding="utf-8")
            self._regenerate_index()
            return self._commit_paths(
                [
                    self._page_relpath(slug),
                    f"{self.config.wiki_dir}/{INDEX_FILENAME}",
                ],
                subject=commit_subject or f"compact: {slug}",
            )

    def rename_page(
        self,
        old_slug: str,
        new_slug: str,
        *,
        alias: bool = True,
        rewrite_links: bool = True,
        commit_subject: str | None = None,
    ) -> str:
        """Move a page to a new slug, rewriting inbound links. One commit.

        Reorganising a namespace by hand means moving the file, editing
        ``slug:``, and finding every inbound ``[[link]]`` — one production
        wiki rewrote 583 of them via a throwaway script that shipped two
        bugs. Doing it here means that work is written once, with tests.

        **Operator-only.** Renaming rewrites inbound ``[[links]]`` across
        the whole corpus, which means writing into files chosen by the
        link graph rather than by the caller — open pages, pages in
        other compartments, pages this session cannot see. There is no
        useful way to label-check a write whose targets are discovered
        rather than named, and the content that lands in them is the new
        slug, which the caller chooses. That is a write-down with
        attacker-supplied text, and the only clean answer is that a view
        does not get to reorganise the namespace. Same reasoning as
        ``import_vault`` and ``repair_pages``.

        ``alias=True`` (default) records ``old_slug`` in the moved page's
        ``aliases:``, so the old name keeps resolving. That matters even
        with a perfect link rewrite: references *outside* the wiki —
        tickets, configs, a shipped answer citing a slug — are ones
        outmem cannot reach. The alias is the safety net under the
        rewrite, not a replacement for it.

        Returns the new HEAD SHA.
        """
        self._require_operator("renaming a page")
        with self._write_lock:
            old_slug = self.resolve_slug(old_slug)
            validate_slug(old_slug)
            validate_slug(new_slug)
            if old_slug == new_slug:
                raise OutmemError(f"Cannot rename {old_slug!r} to itself.")
            if INDEX_SLUG in (old_slug, new_slug):
                raise OutmemError("The reserved 'index' slug cannot be renamed.")
            old_path = self._page_path(old_slug)
            if not old_path.exists():
                raise OutmemError(f"No such wiki page: {old_slug}")
            new_path = self._page_path(new_slug)
            if new_path.exists():
                raise OutmemError(f"Page already exists: {new_slug}")
            owner = self._alias_index().get(new_slug)
            if owner is not None and owner != old_slug:
                raise OutmemError(
                    f"{new_slug!r} is an alias of {owner!r}; renaming here would "
                    f"silently retarget every [[{new_slug}]] link."
                )

            frontmatter, body = parse_wiki_page(
                old_path.read_text(encoding="utf-8"), fallback_slug=old_slug
            )
            frontmatter.slug = new_slug
            if alias and old_slug not in frontmatter.aliases:
                frontmatter.aliases = [*frontmatter.aliases, old_slug]
            touch_updated(frontmatter)

            new_path.parent.mkdir(parents=True, exist_ok=True)
            new_path.write_text(serialize_wiki_page(frontmatter, body), encoding="utf-8")
            old_path.unlink()
            touched = [self._page_relpath(old_slug), self._page_relpath(new_slug)]

            if rewrite_links:
                touched.extend(self._rewrite_links_to(old_slug, new_slug))

            # A frozen source naming this page cannot be rewritten — that is
            # what content addressing means — but the mapping recorded at
            # ingest can be, and that is the point of recording it. Do this
            # even when rewrite_links is off: the caller declined to touch
            # *page* text, not to corrupt the registry.
            #
            # BOTH registries. A local source's refs go stale on rename for
            # exactly the same reason a tracked one's do, and the local tree
            # is the one whose drift nobody can spot in a diff. Only the
            # tracked registry is staged — the local one lives inside the
            # gitignored tree, so there is nothing for git to record.
            for tree in _sources.existing_trees(self):
                repointed = _sources.get_registry(self, tree).repoint_refs(
                    old_slug, new_slug
                )
                if repointed and tree.tracked:
                    touched.append(tree.repo_registry_relpath)

            self._alias_map = None  # the page moved; any cached map is stale
            self._regenerate_index()
            touched.append(f"{self.config.wiki_dir}/{INDEX_FILENAME}")
            return self._commit_paths(
                touched,
                subject=commit_subject or f"rename: {old_slug} -> {new_slug}",
            )

    def commit_registry(self, subject: str) -> str | None:
        """Commit ``.sources.db`` alone, for registry-only mutations."""
        self._require_operator("committing the registry")
        return self._commit_paths(
            [f"{self.config.wiki_dir}/{SOURCES_DIR}/{REGISTRY_FILENAME}"],
            subject=subject,
        )

    def record_source_refs(self, rel_path: str) -> list[SourceRef]:
        """Resolve and record the page slugs one source names.

        Runs automatically at ingest. Exposed because a source registered
        before the mapping existed has none, and what is still resolvable
        *today* is worth capturing before the next reorganisation makes it
        unresolvable — see ``outmem sources backfill``.
        """
        self._require_operator("recording source references")
        return _sources.record_source_refs(self, rel_path)

    def source_refs(self, rel_path: str | None = None) -> list[SourceRef]:
        """Recorded source→page references, kept current across renames.

        The reverse of ``provenance:``: which *pages* a frozen source
        names, rather than which sources a page was compacted from.
        Recorded at ingest because that is the only moment the tokens are
        known correct, and re-pointed by :meth:`rename_page`.

        Spans both source trees. With ``rel_path`` given, the tree that
        actually holds it is consulted — asking the tracked registry
        about a local source returns an empty list, which reads as "this
        source names no pages" when the truth is "wrong registry".

        Filtered at both ends for a viewer: a ref pairs a source key
        with a page slug, so it discloses whichever of the two is
        hidden.
        """
        if rel_path is None:
            refs = [
                ref
                for tree in _sources.existing_trees(self)
                for ref in _sources.get_registry(self, tree).refs(None)
            ]
        else:
            if not self._source_visible(rel_path):
                return []
            found = _sources.resolve_source(self, rel_path)
            tree, key = (
                found if found is not None else (_sources.tracked_tree(self), rel_path)
            )
            refs = _sources.get_registry(self, tree).refs(key)
        if not self.enforces_visibility:
            return refs
        index = self._labels()
        return [
            ref
            for ref in refs
            if self._source_visible(ref.rel_path, index)
            and self._page_visible(ref.page_slug, index)
        ]

    def _rewrite_links_to(self, old_slug: str, new_slug: str) -> list[str]:
        """Point every ``[[old_slug]]`` at ``new_slug``. Returns paths touched.

        Matches the whole wikilink rather than substituting the slug as
        raw text — a naive text replace also hits ``[[old:slug:child]]``
        (a different page) and any prose that happens to contain the
        slug, which is exactly the class of bug a hand-rolled rename
        script produces.
        """
        import re as _re

        from outmem.slug import _WIKILINK_RE

        touched: list[str] = []
        roots = [(self.pages_path, True), (self.log_path, False)]
        for root, is_page in roots:
            if not root.is_dir():
                continue
            for path in sorted(root.rglob("*.md")):
                try:
                    text = path.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    continue
                if old_slug not in text:
                    continue

                def _sub(match: _re.Match[str]) -> str:
                    inner = match.group(0)[2:-2]
                    target, sep, display = inner.partition("|")
                    if target.strip() != old_slug:
                        return match.group(0)
                    return f"[[{new_slug}{sep}{display}]]"

                rewritten = _WIKILINK_RE.sub(_sub, text)
                if rewritten == text:
                    continue
                path.write_text(rewritten, encoding="utf-8")
                if is_page:
                    rel = relpath_to_slug(path.relative_to(self.pages_path))
                    touched.append(self._page_relpath(rel))
                else:
                    touched.append(
                        f"{self.config.log_dir}/"
                        f"{path.relative_to(self.log_path).as_posix()}"
                    )
        return touched

    def extend_page(
        self,
        slug: str,
        *,
        body: str,
        provenance: Sequence[ProvenanceEntry] | None = None,
        commit_subject: str | None = None,
        allow_elision: bool = False,
    ) -> str:
        """Replace the body of an existing page and commit.

        Frontmatter is preserved; ``updated`` is bumped to now. The
        commit message defaults to ``extend: <slug>``. ``wiki/index.md``
        is regenerated and staged in the same commit (title or tag
        edits will surface there).

        ``provenance`` *replaces* the page's source pointers. Pass it when
        re-compacting a page against a newer version of its source —
        without it there is no way to update the field, so a page reported
        by ``outmem stale`` would keep citing the superseded version and
        keep being reported, forever. Omit it and provenance is untouched.
        """
        # Resolve BEFORE anything else: read() would follow the alias but
        # _page_relpath(slug) would not, so the commit would stage a path
        # that doesn't exist — after the page and index.md were already
        # rewritten on disk.
        with self._write_lock:
            slug = self.resolve_slug(slug)
            if slug == INDEX_SLUG:
                raise OutmemError(
                    "Cannot edit the reserved 'index' slug — `wiki/index.md` "
                    "is auto-maintained by outmem on every page write."
                )
            self._check_write(slug, new=False, body=body, provenance=provenance)
            if not allow_elision:
                _reject_incomplete_body(
                body, tool="extend_page", allowed=self._elision_allowed
            )
            page = self.read(slug)
            if provenance is not None:
                page.frontmatter.provenance = list(provenance)
            touch_updated(page.frontmatter)
            page_text = serialize_wiki_page(page.frontmatter, body)
            page.path.write_text(page_text, encoding="utf-8")
            self._regenerate_index()
            return self._commit_paths(
                [
                    self._page_relpath(slug),
                    f"{self.config.wiki_dir}/{INDEX_FILENAME}",
                ],
                subject=commit_subject or f"extend: {slug}",
            )

    def allow_elision_body(self, body: str) -> None:
        """Pre-authorise this exact body text past the elision guard.

        For callers that have already adjudicated the text: a human
        reviewer who edited and approved it, or a model that re-sent it
        unchanged after being handed the refusal. The page is still
        reported by ``outmem lint`` as ``truncated-page`` — the bargain
        is bounded damage and a visible record, not silence.
        """
        self._elision_allowed.add(_body_text_key(body))

    def append_page(
        self,
        slug: str,
        *,
        body: str,
        provenance: Sequence[ProvenanceEntry] | None = None,
        commit_subject: str | None = None,
        allow_elision: bool = False,
    ) -> str:
        """Append to an existing page's body and commit.

        The counterpart :meth:`extend_page` never had. ``extend_page``
        *replaces* the whole body, so building a long page in pieces
        means re-emitting everything written so far on every call: the
        calls grow monotonically and the last one still has to emit the
        entire page in a single turn. That is the exact output-budget
        pressure that makes a model truncate, so "write it in sections"
        was not actually a way to avoid it — and a model that reads
        ``extend`` as *append* silently destroys the earlier sections
        instead.

        Appending removes the constraint: each call carries one section,
        sized to comfortably fit, and the page's length stops being
        bounded by a single turn's output budget.

        ``body`` is separated from the existing text by a blank line, so
        callers pass a section rather than worrying about joins.
        ``provenance`` *adds* pointers the page doesn't already cite
        (dedup by path) — appended content often comes from a source the
        earlier sections didn't use, and additive is the only semantic
        that lets a section be written without restating the citations
        of every section before it. This is deliberately unlike
        :meth:`extend_page`, whose replace semantics exist so a
        re-compaction can drop a superseded source.
        """
        with self._write_lock:
            slug = self.resolve_slug(slug)
            if slug == INDEX_SLUG:
                raise OutmemError(
                    "Cannot edit the reserved 'index' slug — `wiki/index.md` "
                    "is auto-maintained by outmem on every page write."
                )
            if not body.strip():
                raise OutmemError(
                    "append_page: body is empty — nothing to append. Pass the "
                    "section text, or use `extend_page` to replace the body."
                )
            self._check_write(
                slug,
                new=False,
                body=body,
                provenance=provenance,
                provenance_additive=True,
            )
            if not allow_elision:
                _reject_incomplete_body(
                body, tool="append_page", allowed=self._elision_allowed
            )
            page = self.read(slug)
            existing = page.body.rstrip()
            merged = f"{existing}\n\n{body.strip()}\n" if existing else f"{body.strip()}\n"
            if provenance:
                page.frontmatter.provenance = _merge_provenance(
                    page.frontmatter.provenance, provenance
                )
            touch_updated(page.frontmatter)
            page.path.write_text(
                serialize_wiki_page(page.frontmatter, merged), encoding="utf-8"
            )
            self._regenerate_index()
            return self._commit_paths(
                [
                    self._page_relpath(slug),
                    f"{self.config.wiki_dir}/{INDEX_FILENAME}",
                ],
                subject=commit_subject or f"append: {slug}",
            )

    def rebuild_index(self, *, commit: bool = True) -> str | None:
        """Regenerate ``wiki/index.md`` from the current wiki state.

        Returns the commit SHA when a commit landed, ``None`` when the
        index was already in sync with the wiki tree (so the regen was
        a no-op and no commit was produced).

        Use after manual edits — ``write_page`` / ``extend_page``
        keep the index current automatically, but Obsidian / vim /
        direct-file edits don't go through them.

        With ``commit=False`` the index is rewritten but staging /
        committing is left to the caller (useful in the pre-commit
        hook, where we want the rebuilt index to land in the
        human's commit rather than a separate one).
        """
        self._require_operator("rebuilding the index")
        self._regenerate_index()
        rel = f"{self.config.wiki_dir}/{INDEX_FILENAME}"
        if not commit:
            return None
        if not path_is_dirty(self.root, rel):
            return None
        return self._commit_paths([rel], subject="index: rebuild")

    def append_log(
        self,
        *,
        topic: str,
        content: str,
        when: datetime | None = None,
        commit_subject: str | None = None,
    ) -> str:
        """Append an entry to ``log/<today>.md`` and commit.

        The file is created if missing. ``content`` is appended as-is;
        callers compose their own structure (timestamp, session ID, etc.).
        Commit message defaults to ``log: <topic>``.

        A restricted session writes to ``log/<label-set>/<date>.md``
        instead. The log is otherwise an open file, and mandatory
        writeback actively pushes an agent there when nothing else was
        warranted — so in mode ``S`` the default path is a write-down of
        whatever the session was just reading, reached by the one tool
        the runtime insists on calling. The open mode is unpartitioned,
        so a wiki with no restrictions has no new directory level and
        nothing to migrate.
        """
        with self._write_lock:
            if not topic.strip():
                raise OutmemError("append_log: topic must be non-empty.")
            self._require_write_grant()
            ts = ensure_utc(when) if when else utc_now()
            log_date = ts.date()
            partition = mode_dirname(self._mode or ())
            log_dir = self.log_path / partition if partition else self.log_path
            log_file = log_dir / f"{_format_log_filename(log_date)}.md"
            log_file.parent.mkdir(parents=True, exist_ok=True)

            existed = log_file.exists()
            existing = log_file.read_text(encoding="utf-8") if existed else ""
            prefix = "" if not existed else "\n"
            if not existed:
                existing = f"# {log_date.isoformat()}\n\n"
            log_file.write_text(existing + prefix + content.rstrip() + "\n", encoding="utf-8")

            rel = f"{self.config.log_dir}/{log_file.name}"
            if partition:
                rel = f"{self.config.log_dir}/{partition}/{log_file.name}"
            return self._commit_paths(
                [rel],
                subject=commit_subject or f"log: {topic}",
            )

    # ------------------------------------------------------------------
    # Sources — implementations live in :mod:`outmem._store.sources`
    # ------------------------------------------------------------------

    def add_source(
        self,
        source: str | Path,
        *,
        into_subdir: str | None = None,
        rename: str | None = None,
        as_key: str | None = None,
        local: bool = False,
        commit: bool = True,
        restricted: Iterable[str] | None = None,
    ) -> SourceEntry:
        """Copy a source file into a source tree and register it.

        Content-addressed: the file lands at
        ``wiki/sources/[<into>/]<sha[:12]>/<filename>``. Re-adding the
        same content is a no-op.

        ``local=True`` targets ``wiki/sources-local/`` instead — the
        untracked tree for material you may read but not redistribute
        (licensed corpora, copyrighted text, embargoed drafts). The
        source bytes stay on this machine; the pages compiled from them
        are ordinary tracked wiki pages, because the derived knowledge
        is yours to ship even when the source is not. Nothing is
        committed for a local ingest — both the file and its registry
        live inside the gitignored tree.

        Note the asymmetry this creates: a page citing a local source
        records ``sources-local/<sha>/<filename>`` in its ``provenance:``,
        and that page *is* tracked. The filename travels even though the
        bytes do not. That is the intended trade (a citation is not a
        redistribution) but it means the tree is about distribution
        rights, not secrecy — do not use it for material whose *name*
        must stay private.

        ``as_key`` declares *which document* this file is a version of. A
        revision therefore supersedes its predecessor instead of landing
        as an unrelated row. Without it, the identity is derived from the
        path — and the call is **refused** when that derivation is
        ambiguous, because "new version of that document" and "different
        document, same filename" are indistinguishable from the path
        alone, and guessing wrong writes a supersession edge that would
        later drive a recheck of one document against another.

        ``restricted`` labels the source (see
        :mod:`outmem.restricted`). Ingest is the right moment for the
        decision because it is the one point where a person is holding
        the document; every page later compiled from it inherits these
        labels automatically, so restricting here restricts the whole
        downstream. Labels from matching ``restricted.sources`` path
        rules are unioned in. Orthogonal to ``local``, which is about
        redistribution rights rather than secrecy.
        """
        self._require_operator("registering a source")
        # Under the same lock as every other commit-producing path. The
        # registry itself is already safe across processes — SQLite
        # serialises the writers — but the git half was not: two ingests
        # interleaving between `git add` and `git commit` race on the
        # index, and one of them dies. This was the only write path
        # outside the lock.
        with self._write_lock:
            return _sources.add_source(
                self,
                source,
                into_subdir=into_subdir,
                rename=rename,
                as_key=as_key,
                local=local,
                commit=commit,
                restricted=restricted,
            )

    def source_citations(
        self, *, local: bool | None = None
    ) -> tuple[dict[str, list[str]], list[PageLoadFailure]]:
        """``source rel_path -> [slug, …]`` from every page's ``provenance:``.

        Keys are tree-relative registry keys, so callers can look them up
        directly. ``local`` filters by which tree a citation names:
        ``False`` for the tracked tree, ``True`` for the local one,
        ``None`` (default) for both. A citation with no tree prefix is
        assumed tracked, which is what pages written before the split
        carry.

        The reverse of the provenance edge, built in one walk. Nothing
        stored it because nothing consumed it — it is what turns
        provenance from an audit trail into a liveness signal.

        Returns the loader's failures alongside the map, per the shared
        loader contract: a page whose frontmatter will not parse is not
        in the map, so a caller that drops the failures would report "no
        page cites a superseded source" for a wiki that does. Silence is
        the one answer this feature must never give.
        """
        from outmem.index import load_editorial_pages
        from outmem.lint import provenance_ref

        out: dict[str, list[str]] = {}
        index = self._labels() if self.enforces_visibility else None
        pages, failures = load_editorial_pages(self.pages_path)
        for page in pages:
            for entry in page.frontmatter.provenance:
                ref = provenance_ref(entry)
                if ref is None:
                    continue
                # `sources-local/x` must not be read as a tracked ref: a
                # bare removeprefix("sources/") leaves it unchanged, so it
                # would never match a registry row and the citation would
                # silently drop out of supersession reporting.
                tree, key = _sources.split_tree_prefix(self, ref)
                is_local = tree is not None and not tree.tracked
                if local is not None and is_local is not local:
                    continue
                # Both ends of the edge: the citation names a source
                # (whose rel_path embeds a filename) and a page. Either
                # being hidden hides the edge.
                if index is not None and not (
                    self._page_visible(page.slug, index)
                    and self._source_visible(key, index)
                ):
                    continue
                out.setdefault(key, []).append(page.slug)
        return out, self._visible_failures(failures, index)

    def provenance_annotations(self) -> dict[tuple[str, str], ProvenanceAnnotation]:
        """``(source key, slug) -> annotation`` for citations carrying one.

        A parallel lookup rather than a richer :meth:`source_citations`
        return, because every existing caller of that map wants "which
        pages cite this source" and would have to learn a new shape to
        keep asking it.

        Values are returned as written, unrecognised ones included —
        ``outmem lint`` is where a typo gets named, and silently dropping
        it here would make the lint warning describe something the rest
        of outmem pretends it never saw.
        """
        from outmem.index import load_editorial_pages
        from outmem.lint import provenance_annotation, provenance_ref

        out: dict[tuple[str, str], ProvenanceAnnotation] = {}
        index = self._labels() if self.enforces_visibility else None
        pages, _failures = load_editorial_pages(self.pages_path)
        for page in pages:
            for entry in page.frontmatter.provenance:
                annotation = provenance_annotation(entry)
                ref = provenance_ref(entry)
                if ref is None or not annotation:
                    continue
                _tree, key = _sources.split_tree_prefix(self, ref)
                if index is not None and not (
                    self._page_visible(page.slug, index)
                    and self._source_visible(key, index)
                ):
                    continue
                out[(key, page.slug)] = annotation
        return out

    def provenance_findings(self) -> dict[tuple[str, str], str]:
        """``(source key, slug) -> finding`` for citations recording a check.

        See :data:`outmem.sources.PROVENANCE_FINDINGS`.
        """
        return {
            key: annotation.finding
            for key, annotation in self.provenance_annotations().items()
            if annotation.finding is not None
        }

    def stale_pages(
        self, *, include_acknowledged: bool = False
    ) -> tuple[list[StaleCitation], list[PageLoadFailure]]:
        """Pages whose provenance cites a source version since superseded.

        The payoff of supersession: a source moving to v2 tells you exactly
        which pages were compacted from v1 and may no longer hold. Reports
        only — deciding whether a page still stands is a judgement call,
        and on clinical content that belongs to a human (or an explicit
        agent run over this list), not to a side effect of ingest.

        Returns the loader failures too — a page that would not parse is
        a page this check could not run on, and reporting a clean wiki
        while silently skipping it is the failure this exists to prevent.

        Covers both source trees. A licensed handbook that gets a revised
        edition supersedes exactly like a tracked source does, and the
        pages compacted from the old edition are just as stale — the
        report would be quietly half-blind if it only consulted the
        tracked registry.

        A citation carrying ``superseded_ok:`` is omitted unless
        ``include_acknowledged`` — see :func:`_acknowledgement` for why
        that suppression expires rather than being permanent.
        """
        self._require_operator('the staleness report')
        from outmem.sources import StaleCitation

        out: list[StaleCitation] = []
        failures: list[PageLoadFailure] = []
        annotations = self.provenance_annotations()
        for tree in _sources.existing_trees(self):
            registry = _sources.get_registry(self, tree)
            citations, tree_failures = self.source_citations(local=not tree.tracked)
            if tree.tracked:
                # Page-load failures are a property of the wiki, not of a
                # tree; collect them once so they aren't double-reported.
                failures = tree_failures
            for rel_path, slugs in sorted(citations.items()):
                entry = registry.entries.get(rel_path)
                if entry is None or entry.superseded_by is None:
                    continue
                current = entry.superseded_by
                # Follow the chain to the newest version, not just the next one.
                seen = {rel_path}
                while current in registry.entries and current not in seen:
                    seen.add(current)
                    nxt = registry.entries[current].superseded_by
                    if nxt is None:
                        break
                    current = nxt
                for slug in sorted(slugs):
                    annotation = annotations.get((rel_path, slug))
                    acknowledged = _acknowledgement(
                        annotation, registry.entries.get(current)
                    )
                    if acknowledged is not None and not include_acknowledged:
                        continue
                    out.append(
                        StaleCitation(
                            slug=slug,
                            cited=tree.entry_relpath(rel_path)
                            if not tree.tracked
                            else rel_path,
                            current=current,
                            document_key=entry.document_key or "",
                            finding=(
                                annotation.finding if annotation is not None else None
                            ),
                            current_exists=current in registry.entries,
                            acknowledged=acknowledged,
                        )
                    )
        return sorted(out, key=lambda c: (c.slug, c.cited)), failures

    def propose_document_keys(self) -> tuple[list[KeyCandidate], list[PageLoadFailure]]:
        """Candidate ``document_key`` groupings for rows that predate identity.

        Tracked tree only, and the citations are filtered to match:
        feeding cross-tree citations to a single-tree registry would key
        candidates on rows that registry has never heard of. The local
        tree needs no backfill by construction — it postdates document
        identity, so every row in it was registered with ``--as``
        available. Should that ever stop being true, this becomes a loop
        over ``existing_trees`` like :meth:`stale_pages`.
        """
        self._require_operator("proposing document keys")
        from outmem.sources import propose_document_keys

        citations, failures = self.source_citations(local=False)
        return propose_document_keys(_sources.get_registry(self), citations), failures

    def assign_document_keys(self, pairs: Sequence[tuple[str, str]]) -> int:
        """Set ``document_key`` on rows that have none. Commits once.

        Goes through :meth:`SourceRegistry.adopt_document_key`, which
        re-checks inside a write transaction that no live row already
        holds the key. Doing the UPDATE directly here meant backfill could
        put two live rows on one identity — the very merge ``add_source``
        refuses to perform, done silently by outmem's own migration
        command. Rows that fail that check are skipped, so the count
        returned is what was actually written.
        """
        from outmem.sources import DocumentKeyConflict

        self._require_operator("assigning document keys")
        registry = _sources.get_registry(self)
        written = 0
        for rel_path, key in pairs:
            entry = registry.entries.get(rel_path)
            if entry is None or entry.document_key is not None:
                continue
            try:
                registry.adopt_document_key(rel_path, key)
            except DocumentKeyConflict:
                continue
            written += 1
        if written:
            self._commit_paths(
                [f"{self.config.wiki_dir}/{SOURCES_DIR}/{REGISTRY_FILENAME}"],
                subject=f"sources: assign {written} document identit(ies)",
            )
        return written

    def rekey_document(
        self,
        old_key: str,
        new_key: str | None = None,
        *,
        local: bool | None = None,
        dry_run: bool = True,
    ) -> RekeyResult:
        """Move a document to another identity, rebuilding its chain.

        The repair for two editions of one document that landed under
        different *derived* keys: they hold no edge between them, so
        ``outmem stale`` never fires for either. Merging is the normal
        case — pass the key you want to keep as ``new_key`` and the rows
        are relabelled *and* chained by ``registered_at``.

        Called with no ``new_key`` it rebuilds the chain under the key it
        is given, which repairs a registry whose ``document_key`` was set
        out of band (leaving several rows live under one identity, where
        :meth:`SourceRegistry.latest_for` silently picks the newest).

        ``local`` picks the tree; ``None`` finds whichever holds the key.
        Each tree has its own registry, so a key held in both names two
        unrelated documents and must be disambiguated.
        """
        self._require_operator('rekeying a document')
        tree = self._tree_for_document(old_key, local=local)
        registry = _sources.get_registry(self, tree)
        if dry_run:
            return registry.plan_rekey(old_key, new_key)
        written = registry.rekey(old_key, new_key)
        if written.applied and tree.tracked:
            # A local rekey has nothing to commit — that registry lives
            # inside the gitignored tree, like the sources it indexes.
            self.commit_registry(
                f"sources: rekey {normalize_document_key(old_key)} "
                f"-> {written.document_key}"
                if new_key is not None
                else f"sources: rechain {written.document_key}"
            )
        return written

    def _tree_for_document(
        self, document_key: str, *, local: bool | None = None
    ) -> _sources.SourceTree:
        """Which source tree holds ``document_key``.

        Falls back to the tracked tree when nothing holds it, so the
        registry raises its own "no such document" rather than this
        method inventing a second wording for the same miss.
        """
        if local is not None:
            tree = _sources.local_tree(self) if local else _sources.tracked_tree(self)
            if not tree.path.is_dir():
                # Opening a registry creates its directory, and for the
                # local tree that would leave one without the .gitignore
                # entry `ensure_sources_local` writes alongside it.
                raise OutmemError(f"this wiki has no {tree.name}/ tree.")
            return tree
        key = normalize_document_key(document_key)
        holders = [
            tree
            for tree in _sources.existing_trees(self)
            if any(
                e.document_key == key
                for e in _sources.get_registry(self, tree).entries.values()
            )
        ]
        if len(holders) > 1:
            raise OutmemError(
                f"{key!r} is held in both {holders[0].name}/ and "
                f"{holders[1].name}/. Each tree carries its own registry, so "
                "those are two different documents — pass local=True/False "
                "(CLI: `--tree`) to say which one you mean."
            )
        return holders[0] if holders else _sources.tracked_tree(self)

    def sources_gc(self, *, dry_run: bool = True) -> RegistryAudit:
        """Reconcile ``.sources.db`` against disk; drop rows whose file is gone.

        Returns a :class:`outmem.sources.RegistryAudit`. ``dry_run=True``
        by default (the ``repair_pages`` convention) because the registry
        is a git-tracked binary — every apply writes a full blob into
        history. Files with no registry row are reported, never deleted.

        Reconciles both source trees. The local registry drifts for the
        same reasons the tracked one does, and leaving it out would mean
        the one registry a user cannot inspect through git history is
        also the one nothing cleans. Only the tracked registry produces
        a commit; the local one lives inside the gitignored tree.
        """
        self._require_operator("source garbage collection")
        from outmem.sources import RegistryAudit, gc_registry

        audit = gc_registry(self.sources_path, dry_run=dry_run)
        if not dry_run and (audit.missing_files or audit.orphan_ingestions):
            self._source_registry = None  # drop the cached handle
            self._commit_paths(
                [f"{self.config.wiki_dir}/{SOURCES_DIR}/{REGISTRY_FILENAME}"],
                subject=f"sources: gc — dropped {len(audit.missing_files)} stale row(s)",
            )

        if not self.sources_local_path.is_dir():
            return audit

        local_audit = gc_registry(self.sources_local_path, dry_run=dry_run)
        if not dry_run and (local_audit.missing_files or local_audit.orphan_ingestions):
            self._source_registry_local = None
        # Merge so a caller sees one report. Local paths are tree-qualified
        # so the two trees stay distinguishable in the output.
        return RegistryAudit(
            missing_files=[
                *audit.missing_files,
                *(f"{SOURCES_LOCAL_DIR}/{p}" for p in local_audit.missing_files),
            ],
            unregistered=[
                *audit.unregistered,
                *(f"{SOURCES_LOCAL_DIR}/{p}" for p in local_audit.unregistered),
            ],
            orphan_ingestions=audit.orphan_ingestions + local_audit.orphan_ingestions,
        )

    def list_sources(self, *, include_missing: bool = False) -> list[SourceEntry]:
        """Every registered source visible to this viewer, by relative path.

        A source's ``rel_path`` embeds its original filename, so listing
        one is disclosure even without reading it.
        """
        entries = _sources.list_sources(self, include_missing=include_missing)
        if not self.enforces_visibility:
            return entries
        index = self._labels()
        return [e for e in entries if self._source_visible(e.rel_path, index)]

    def get_source(self, rel_path: str) -> SourceEntry | None:
        """Lookup a single registered source by its relative path.

        ``None`` for a hidden source — the same answer as for one that
        was never registered.
        """
        if not self._source_visible(rel_path):
            return None
        return _sources.get_source(self, rel_path)

    def read_source(self, rel_path: str, *, max_chars: int | None = None) -> str:
        """Return the text of a source file, capped at ``max_chars``.

        A hidden source produces the byte-identical not-found text that
        an unregistered path does.
        """
        if not self._source_visible(rel_path):
            raise OutmemError(f"no such source: {rel_path}")
        return _sources.read_source(self, rel_path, max_chars=max_chars)

    def record_ingestion(
        self,
        rel_path: str,
        *,
        prompt: str | None,
        pages_touched: Sequence[str],
        commit: bool = True,
        when: datetime | None = None,
    ) -> IngestionRecord:
        """Append an ingestion entry to a registered source.

        Called after the agent has finished writing pages from a
        source. ``commit=True`` lands an ``ingest: <rel-path>`` commit.

        The only source-touching write in any tool palette, and its
        ``prompt`` is agent-written free text stored in the registry —
        so it is held to the same rule as a page write: the source's
        labels must equal the session's mode. Recording an open
        ingestion against a restricted source would put session text
        where an open reader can find it.
        """
        if self._mode is not None:
            self._require_write_grant()
            labels = self._labels().for_source(rel_path)
            if not writable(labels, self._mode, self._grants):
                raise RestrictionError(
                    _write_refusal(self._mode, labels, new=False)
                )
        return _sources.record_ingestion(
            self,
            rel_path,
            prompt=prompt,
            pages_touched=pages_touched,
            commit=commit,
            when=when,
        )

    # ------------------------------------------------------------------
    # Vault import — implementations live in :mod:`outmem._store.import_vault`
    # ------------------------------------------------------------------

    def import_vault(
        self,
        source: str | Path,
        *,
        force: bool = False,
    ) -> _import.ImportSummary:
        """Import every ``*.md`` under ``source`` into ``wiki/``.

        See :func:`outmem._store.import_vault.import_vault` for the
        full contract — flat slug namespace with collision resolution,
        wikilink rewriting, one atomic commit.
        """
        self._require_operator('vault import')
        return _import.import_vault(self, Path(source).expanduser(), force=force)

    # ------------------------------------------------------------------
    # Semantic index — implementations live in :mod:`outmem._store.semantic`
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Access control — see ``specs/restricted-content.md``
    # ------------------------------------------------------------------

    @property
    def restrictions(self) -> RestrictedSettings:
        """The wiki's ``restricted:`` config block.

        A shorthand for ``config.outmem.restricted`` — consulted from
        enough places that the long form buries the interesting part of
        every call site.
        """
        return self.config.outmem.restricted

    def as_viewer(
        self,
        *,
        mode: Iterable[str] = (),
        grants: Grants | None = None,
    ) -> WikiStore:
        """A store that sees only what ``mode`` is allowed to see.

        This is the object a served request should hold. Hand it to
        ``wiki_read_tools(view)`` and the model's whole world is what
        the view exposes — restricted content is filtered before it ever
        becomes prompt text, so there is no instruction, tool
        description, or system-prompt rule anywhere in the enforcement
        path, and nothing for a model to get wrong.

        ``mode`` is the set of compartments this session works in, and
        it is **immutable for the session's duration**. It defaults to
        the empty set: a session that does not name a compartment
        retrieves open content only, exactly as outmem behaves with no
        restrictions configured. Restricted content is opt-in, and there
        is deliberately no mechanism by which retrieving something
        widens the mode.

        ``grants`` is what the *user* is entitled to, from the calling
        application's own authentication; ``mode`` is what this session
        is scoped to within that. The two differ: a user who holds
        ``hr`` but is running in mode ``∅`` is entitled to HR content
        and simply is not asking for it, which is why retrieval may tell
        them matches exist elsewhere (see ``search``).

        The returned object is a shallow copy. Its lazily-opened
        resources — both source registries, the vector store, the alias
        map — live on one shared object rather than on the copy, so
        every view sees the same registry rows and the process opens one
        set of SQLite connections rather than one per request. The write
        lock and the label cache are shared for the same reason. It
        holds no reference to the unrestricted store, so nothing
        downstream can climb back out of the view.

        Raises:
            LabelError: if ``mode`` names an undeclared label, or one the
                user does not hold a read grant for.
        """
        resolved = normalise_labels(mode)
        entitlement = grants if grants is not None else UNRESTRICTED_GRANTS
        self.restrictions.check_declared(resolved, what="label in mode")
        if grants is not None and not entitlement.may_read(resolved):
            # The one direction that would otherwise be silent: a mode
            # wider than the grant would show the user content they are
            # not entitled to, and every downstream check trusts the mode.
            missing = ", ".join(sorted(resolved - entitlement.read))
            raise LabelError(
                f"mode requests label(s) the user cannot read: {missing}."
            )
        view = copy.copy(self)
        view._mode = resolved
        view._grants = entitlement
        return view

    @property
    def enforces_visibility(self) -> bool:
        """True when this store is a view, i.e. filtering is in effect.

        The bare store returns ``False`` and every guard below is a
        single attribute test — a wiki with no restrictions pays nothing
        for this feature beyond that.
        """
        return self._mode is not None

    @property
    def mode(self) -> frozenset[str]:
        """This view's compartments. Empty on the bare store."""
        return self._mode or frozenset()

    @property
    def grants(self) -> Grants:
        """What the viewing user is entitled to, independent of the mode."""
        return self._grants

    def _labels(self) -> LabelIndex:
        """The resolved label index, rebuilt when the corpus has moved."""
        return self._label_cache.get(self)

    def corpus_token(self) -> tuple[object, ...] | None:
        """What a cache of derived content must be keyed on, or ``None``.

        ``None`` for a wiki that declares no labels: nothing downstream
        needs re-deriving for access control, and a caller that skips
        the key skips a ``git rev-parse`` per call.

        Otherwise HEAD plus the source registries' fingerprints — the
        same pair the label index uses. HEAD alone is not enough,
        because an ingest into the untracked local tree, and a re-ingest
        that only sets labels, both write the registry and commit
        nothing.
        """
        if not self.restrictions.enabled:
            return None
        from outmem._store.labels import registry_stamp

        return (self.head(), registry_stamp(self))

    def _can_see(self, labels: Iterable[str]) -> bool:
        """§3.1 — ``labels ⊆ mode``. Always true on the bare store."""
        if self._mode is None:
            return True
        return visible(labels, self._mode)

    def _page_visible(self, slug: str, index: LabelIndex | None = None) -> bool:
        if self._mode is None:
            return True
        idx = index if index is not None else self._labels()
        if slug not in idx.pages and slug != INDEX_SLUG:
            # The index does not know this slug. Either the page does not
            # exist — in which case there is nothing to hide and the path
            # rules alone decide — or it was written after the index was
            # built, and we do not know what it is labelled.
            #
            # That second case is a real race, not a theoretical one: a
            # write puts the file on disk and commits afterwards, both
            # under the write lock, while readers hold no lock at all. In
            # the window between, HEAD has not moved, so the token says
            # the index is current when it is not — and a page created in
            # mode {hr} would be served to everyone until the commit
            # landed. Deny what we cannot classify, as everywhere else.
            try:
                if self._page_path(slug).exists():
                    return False
            except (SlugError, OutmemError):
                pass  # not addressable as a page; path rules decide
        return self._can_see(idx.for_page(slug))

    def _source_visible(self, key: str, index: LabelIndex | None = None) -> bool:
        if self._mode is None:
            return True
        idx = index if index is not None else self._labels()
        if not idx.knows_source(key) and self._source_file_exists(key, idx):
            # The mirror of the page race. `add_source` copies the file
            # into the tree and registers it afterwards, so between the
            # two there is a file on disk with no registry row — and a
            # row is where its labels live. An unknown key that names a
            # real file means the index predates it, not that it is open.
            return False
        return self._can_see(idx.for_source(key))

    def _source_file_exists(self, key: str, index: LabelIndex) -> bool:
        """Whether ``key``, in any spelling, names a file in either tree."""
        for candidate in index.source_keys(key):
            for tree in (self.sources_path, self.sources_local_path):
                path = tree / candidate
                try:
                    if path.is_file() and path.resolve().is_relative_to(
                        tree.resolve()
                    ):
                        return True
                except OSError:
                    continue
        return False

    def _repo_path_visible(self, rel_path: str, index: LabelIndex) -> bool:
        """Visibility for a repo-relative path — the form ripgrep hits and
        semantic chunks arrive in.

        Everything a search can reach falls into one of four trees, and
        each maps back onto an item whose labels are already resolved:
        a page path to its slug, a source path to its registry key, a
        log path to the mode that wrote it. A path in none of them is
        open — the alternative, denying what we cannot classify, would
        hide ``AGENTS.md`` from every view for no gain.
        """
        pages_prefix = self.pages_prefix()
        if rel_path.startswith(pages_prefix):
            from outmem.slug import relpath_to_slug

            tail = rel_path[len(pages_prefix) :]
            # Through `_page_visible`, not `index.for_page`: a ripgrep hit
            # can name a file the index has not classified yet, and that
            # is the case the fail-closed branch in there exists for.
            return self._page_visible(relpath_to_slug(Path(tail)), index)
        for tree in (SOURCES_DIR, SOURCES_LOCAL_DIR):
            prefix = f"{self.config.wiki_dir}/{tree}/"
            if rel_path.startswith(prefix):
                # Through `_source_visible` for the same reason the page
                # branch goes through `_page_visible`: one predicate
                # reached two ways is one predicate too many.
                return self._source_visible(f"{tree}/{rel_path[len(prefix):]}", index)
        log_prefix = f"{self.config.log_dir}/"
        if rel_path.startswith(log_prefix):
            parts = rel_path[len(log_prefix) :].split("/")
            if len(parts) < 2:
                return True  # log/<date>.md — the open partition
            labels = self.restrictions.mode_from_log_dirname(parts[0])
            return True if labels is None else self._can_see(labels)
        return True

    def _filter_hits(self, result: SearchResult, scope: str) -> SearchResult:
        """Drop ripgrep rows from files this viewer cannot see.

        Hit paths are relative to the scope's search root, not the repo,
        so they are re-anchored before being classified. ``truncated``
        is carried through unchanged: it describes the ripgrep run, and
        rewriting it from the filtered count would tell the caller
        something about what was removed.
        """
        if not self.enforces_visibility:
            return result
        import dataclasses

        prefix = {
            "wiki": self.pages_prefix(),
            "log": f"{self.config.log_dir}/",
        }.get(scope, "")
        index = self._labels()
        kept = tuple(
            hit
            for hit in result.hits
            if self._repo_path_visible(prefix + hit.path, index)
        )
        return dataclasses.replace(result, hits=kept)

    def _check_write(
        self,
        slug: str,
        *,
        new: bool,
        body: str | None = None,
        provenance: Sequence[ProvenanceEntry] | None = None,
        declared: Iterable[str] | None = None,
        provenance_additive: bool = False,
    ) -> frozenset[str]:
        """§3.2 — decide whether this view may write ``slug``, and with
        which labels. Returns the labels the item must carry.

        The rule is ``labels(item) == mode``, equality rather than
        containment, forced from both directions. ``labels ⊆ mode``
        because you must be able to see what you are modifying;
        ``labels ⊇ mode`` because you must not carry facts out of a more
        restricted context into a less restricted item. Together they
        mean an agent in mode ``S`` can only ever write items labelled
        exactly ``S``, which confines the damage of anything it read to
        the compartment it read from — structurally, not by trust.

        A new item defaults to the mode's labels. Without that, writing
        an HR page in mode ``{hr}`` that happened to cite only open
        sources would compute ``∅ ≠ {hr}`` and be refused unless
        somebody remembered an explicit label. Defaulting to the mode is
        fail-safe: it is always the *more* restricted option, and it can
        only be narrowed by the privileged operation.

        Returns the empty set on a bare store, where nothing is enforced.
        """
        if self._mode is None:
            return frozenset()
        mode = self._mode
        self._require_write_grant()
        index = self._labels()
        if new:
            # Everything that would end up on the item: the mode it is
            # being written from, the labels its name attracts, whatever
            # its sources carry, and anything the caller declared. The
            # equality test below then covers all four at once — a
            # declared label the mode does not cover is refused with the
            # same message as any other mismatch, rather than being
            # quietly dropped and leaving the caller believing the page
            # is restricted when it is not.
            labels = mode | self.restrictions.labels_for_slug(slug)
            labels |= normalise_labels(declared)
            labels |= self._inherited(provenance, index)
            if not writable(labels, mode, self._grants):
                raise RestrictionError(_write_refusal(mode, labels, new=True))
        else:
            labels = index.for_page(slug)
            if not writable(labels, mode, self._grants):
                raise RestrictionError(_write_refusal(mode, labels, new=False))
            # An edit may also change what the page WILL be labelled,
            # because `provenance` is part of the write and a page
            # inherits its sources' labels. Both directions matter and
            # both are refused here rather than left to compute after
            # the commit:
            #
            # Adding a restricted citation to an open page relabels it,
            # which lets an open session it should not — it silently
            # removes the page from the open corpus, and the writer can
            # no longer read what they just wrote.
            #
            # Dropping the citation a page's label was inherited FROM
            # declassifies it, through a tool that looks like an
            # ordinary edit. `restrict_page` is the verb for changing
            # what an item is; these two only change what it says.
            after = self._labels_after_edit(
                slug, provenance, index, additive=provenance_additive
            )
            if not writable(after, mode, self._grants):
                raise RestrictionError(
                    _write_refusal(mode, after, new=False)
                    + " (the citation change would relabel the page; use "
                    "`restrict_page` to change what an item is.)"
                )
        if body is not None:
            self._check_closure(body, labels, index)
        return labels

    def _inherited(
        self, provenance: Sequence[ProvenanceEntry] | None, index: LabelIndex
    ) -> frozenset[str]:
        """Labels a page picks up from the sources in ``provenance``."""
        from outmem.lint import provenance_ref

        labels: frozenset[str] = frozenset()
        for entry in provenance or ():
            ref = provenance_ref(entry)
            if ref is not None:
                labels |= index.for_source(ref)
        return labels

    def _labels_after_edit(
        self,
        slug: str,
        provenance: Sequence[ProvenanceEntry] | None,
        index: LabelIndex,
        *,
        additive: bool,
    ) -> frozenset[str]:
        """What ``slug`` would be labelled once this edit lands.

        Only reached for a page the caller may already write, so reading
        it here discloses nothing. ``provenance=None`` means "leave the
        citations alone", which is the common case and cannot change
        anything.

        ``additive`` distinguishes the two write shapes: ``append_page``
        merges into the existing citations and so can only ever add
        labels, while ``extend_page`` replaces them and can drop one.
        Treating append as a replacement would refuse an ordinary
        section that re-cites one source out of several.
        """
        if provenance is None:
            return index.for_page(slug)
        if additive:
            return index.for_page(slug) | self._inherited(provenance, index)
        page = self.read(slug)
        labels = self.restrictions.resolve(frozenset(page.frontmatter.restricted))
        labels |= self.restrictions.labels_for_slug(slug)
        return labels | self._inherited(provenance, index)

    def _check_closure(
        self, body: str, labels: frozenset[str], index: LabelIndex
    ) -> None:
        """§3.3 — refuse a body that links to something more restricted.

        Filtering the page list achieves nothing if an open page's body
        contains ``[[hr:severance-policy]]``: ``read_page`` on the open
        page hands over the slug. Because the write rule has already
        forced ``labels == mode``, this reduces to "you may only link to
        what you can see" — a link target whose labels are not a subset
        of the writer's mode is refused.

        A link to a slug that does not exist resolves to no labels and
        passes. A dangling link discloses nothing that is there.
        """
        from outmem.slug import extract_wikilinks

        for link in extract_wikilinks(body):
            target = index.for_page(link.slug)
            if not target <= labels:
                # Deliberately generic. Naming the target would confirm
                # a hidden page exists — a bounded oracle either way (the
                # writer learns "something blocks this link"), so the
                # message gives up as little as it can while still
                # telling them which link to remove.
                raise RestrictionError(
                    f"refusing to write: the link [[{link.slug}]] points outside "
                    "this session's scope. Remove it, or work in a session that "
                    "covers it."
                )

    def restrict_source(
        self,
        rel_path: str,
        *,
        labels: Iterable[str],
        commit: bool = True,
    ) -> SourceEntry:
        """Set a source's restriction labels. Operator-only.

        The other half of ``restrict_page``, and the one with more
        reach: a page inherits the labels of every source it cites, so
        restricting a document here restricts everything ever compiled
        from it, without anybody having to find those pages.

        *Removing* a label is declassification and, like the page verb,
        is available only to the operator holding the bare store —
        which is who the registry error messages have always pointed
        at.
        """
        self._require_operator("restricting a source")
        wanted = normalise_labels(labels)
        self.restrictions.check_declared(wanted)
        found = _sources.resolve_source(self, rel_path)
        if found is None:
            raise OutmemError(f"no such source: {rel_path}")
        tree, key = found
        registry = _sources.get_registry(self, tree)
        before = registry.entries[key].restricted if key in registry.entries else None
        # A `restricted.sources` rule outranks the column, so a label it
        # supplies cannot be removed here. Reporting success while the
        # rule silently reapplies it is the worst of the three possible
        # outcomes — the same refusal `restrict_page` makes.
        pinned = (before or frozenset()) - wanted
        pinned &= self.restrictions.labels_for_source(key)
        if pinned:
            raise RestrictionError(
                f"cannot remove [{', '.join(sorted(pinned))}] from {key!r}: a "
                "`restricted.sources` rule in config.yaml applies it to this "
                "path, and a path rule outranks the registry. Move the source "
                "out of that directory, or change the rule."
            )
        entry = registry.set_restricted(key, wanted, allow_narrowing=True)
        self._label_cache.invalidate()
        # Nothing to commit when the labels did not move — and asking git
        # to commit an unchanged file surfaces its "nothing added to
        # commit" message, which reads like a failure for what is in
        # fact a no-op. Note the labels may still be non-empty: a path
        # rule can supply what the caller asked to remove.
        if commit and tree.tracked and before != entry.restricted:
            self._commit_paths(
                [tree.repo_registry_relpath], subject=f"restrict: {key}"
            )
        return replace(entry, local=not tree.tracked)

    def restrict_page(
        self,
        slug: str,
        *,
        labels: Iterable[str],
        cascade: bool = False,
        commit_subject: str | None = None,
    ) -> str:
        """Set a page's explicit restriction labels. The operational verb.

        Restricting is a **graph** operation, not a field edit. A page
        that visible pages already link to cannot simply become
        restricted: the inbound link would still name it in a body its
        readers can see, which is the closure invariant (§3.3) and the
        thing that makes hiding real. So inbound references from items
        that would become non-conforming are reported, and the call is
        refused until they are resolved — or ``cascade=True`` applies
        the same labels to the referrers.

        **Operator-only**, for two reasons that both come back to the
        same thing: it writes files the caller did not name. ``cascade``
        picks its targets from the backlink graph, so they can include
        pages a view may not write or even see; and the refusal below
        has to name the referrers, which are exactly the pages a view
        might not be allowed to know about. Restricting is an
        administrative act taken from the server — ``outmem restrict``
        runs it against the bare store — and the point of it is to touch
        content the compartment's own users cannot yet reach.
        """
        self._require_operator("restricting a page")
        with self._write_lock:
            slug = self.resolve_slug(slug)
            if slug == INDEX_SLUG:
                raise OutmemError("The reserved 'index' slug cannot be restricted.")
            wanted = normalise_labels(labels)
            self.restrictions.check_declared(wanted)
            index = self._labels()
            current = index.for_page(slug)
            removed = current - wanted
            # A path rule is a safety net that frontmatter cannot
            # override, so labels it supplies cannot be removed here.
            # Silently keeping them would be the worst outcome: the
            # caller asked to declassify, got a success and a commit,
            # and nothing changed.
            pinned = removed & self.restrictions.labels_for_slug(slug)
            if pinned:
                raise RestrictionError(
                    f"cannot remove [{', '.join(sorted(pinned))}] from {slug!r}: "
                    "a `restricted.paths` rule in config.yaml applies it to this "
                    "slug, and a path rule outranks frontmatter. Rename the page "
                    "out of that namespace, or change the rule."
                )
            # Same for a source: inheritance is computed, not stored, so
            # dropping the frontmatter label would leave the page
            # restricted anyway and the call a no-op.
            inherited: frozenset[str] = frozenset()
            for entry in self.read(slug).frontmatter.provenance:
                from outmem.lint import provenance_ref

                ref = provenance_ref(entry)
                if ref is not None:
                    inherited |= index.for_source(ref)
            blocked = removed & inherited
            if blocked:
                raise RestrictionError(
                    f"cannot remove [{', '.join(sorted(blocked))}] from {slug!r}: "
                    "the page cites a source carrying that label, and a page "
                    "inherits its sources' labels. Declassify the source first, "
                    "or drop the citation."
                )
            # The bare store is the server-side operator; there is no
            # mode and no grant to check against, and gating it would
            # block the very tooling that has to fix a mislabelled page.
            if self._mode is not None and removed and not self._grants.may_declassify(removed):
                raise RestrictionError(
                    f"refusing to remove restriction label(s) "
                    f"[{', '.join(sorted(removed))}] from {slug!r}: "
                    "declassification needs a declassify grant."
                )
            targets = [slug]
            if wanted - current:
                inbound = [
                    referrer
                    for referrer in self.backlinks_cache.referrers(
                        slug, head_or_none(self.root)
                    )
                    if not (wanted <= index.for_page(referrer))
                ]
                if inbound and not cascade:
                    raise RestrictionError(
                        f"refusing to restrict {slug!r}: it is linked from "
                        f"{', '.join(sorted(inbound))}, which would still name "
                        "it in a body their readers can see. Resolve those "
                        "links, or pass cascade=True to restrict them too."
                    )
                targets += sorted(inbound) if cascade else []
            # Load every target BEFORE writing any. A cascade that
            # raises halfway leaves earlier pages rewritten on disk with
            # no commit — a dirty worktree carrying a label change
            # nobody recorded, which is worse than either outcome the
            # call could have had.
            loaded = [(target, self.read(target)) for target in targets]
            paths: list[str] = []
            for target, page in loaded:
                merged = sorted(
                    wanted if target == slug else wanted | index.for_page(target)
                )
                page.frontmatter.restricted = merged
                touch_updated(page.frontmatter)
                page.path.write_text(
                    serialize_wiki_page(page.frontmatter, page.body), encoding="utf-8"
                )
                paths.append(self._page_relpath(target))
            self._regenerate_index()
            self._label_cache.invalidate()
            return self._commit_paths(
                [*paths, f"{self.config.wiki_dir}/{INDEX_FILENAME}"],
                subject=commit_subject or f"restrict: {slug}",
            )

    def _require_write_grant(self) -> None:
        """Refuse a session that may read its compartment but not write it.

        Split out because three paths need it — page writes,
        ``append_log`` and ``record_ingestion`` — and each had grown its
        own copy of the message. The label comparison itself goes
        through :func:`~outmem.restricted.writable`, so the predicate
        the tests exercise is the predicate the store enforces.
        """
        if self._mode is None or self._grants.may_write(self._mode):
            return
        missing = ", ".join(sorted(self._mode - self._grants.write)) or "open content"
        raise RestrictionError(
            f"this session may not write: no write grant for {missing}."
        )

    def _require_operator(self, what: str) -> None:
        """Refuse a path that a served request has no business calling.

        The cheapest kind of safety: a method not reachable from a view
        is a bypass class that needs no filtering code, no test, and
        cannot regress. Used for bulk maintenance and for the two
        history readers, whose output is raw git and cannot be filtered
        by a label index that only knows the current commit.
        """
        if self._mode is not None:
            raise RestrictionError(
                f"{what} is not available to a restricted view; it runs "
                "against the whole wiki and is an operator-only path."
            )

    def _visible_index_page(self, path: Path) -> WikiPage:
        """The ``index`` slug, rendered live from what this view can see.

        ``wiki/index.md`` on disk catalogues every page in the wiki —
        one file whose entire purpose is to list slugs. Serving it to a
        view would hand over the name of everything hidden, which is the
        single largest leak in the read surface and the one that no
        amount of per-page filtering elsewhere would catch.
        """
        from outmem.index import INDEX_TITLE, render_index

        index = self._labels()
        body = render_index(
            self.pages_path, include=lambda slug: self._page_visible(slug, index)
        )
        return WikiPage(
            slug=INDEX_SLUG,
            frontmatter=WikiFrontmatter(
                title=INDEX_TITLE,
                slug=INDEX_SLUG,
                tags=["index"],
                extra={"generated": True},
            ),
            body=body,
            path=path,
        )

    def _visible_failures(
        self, failures: list[PageLoadFailure], index: LabelIndex | None
    ) -> list[PageLoadFailure]:
        """Drop load failures for pages this viewer cannot see.

        A failure names a slug and a reason, which is enough to learn
        that a page exists. Unparseable pages resolve to DENY, so under
        a view this drops all of them — that is deliberate, and it is
        why lint runs against the bare store.
        """
        if index is None:
            return failures
        return [f for f in failures if self._page_visible(f.slug, index)]

    def _no_such_page(self, slug: str) -> NoReturn:
        """Raise exactly what a nonexistent slug raises.

        Not a distinct "denied" error, and not a distinct message: the
        difference between "no such page" and "you may not see that
        page" is precisely the fact being withheld.
        """
        raise OutmemError(f"No such wiki page: {slug}")

    def semantic_available(self) -> bool:
        """Whether this wiki's semantic index has been built (its db
        exists). Semantic has no config flag — build the index with
        ``outmem reindex`` to turn it on."""
        return _semantic.available(self)

    def semantic_index_is_empty(self) -> bool:
        """True if the semantic index has no files indexed yet — including
        when no index has been built at all (``outmem reindex`` hasn't
        run). Safe to call without a prior :meth:`semantic_available`
        check: it never creates an empty index. Once the index exists it
        opens the vector store, paying the one-time embedder dimension
        probe on the first call."""
        return _semantic.index_is_empty(self)

    def semantic_find_similar(
        self,
        text: str,
        *,
        top_k: int | None = None,
        threshold: float | None = None,
        exclude_slug: str | None = None,
    ) -> list[Match]:
        """Return the top semantic matches for ``text``.

        Restricted pages and sources are indexed normally and filtered
        here, at query time, so a cleared user keeps full semantic
        recall. The price is stated in the spec rather than engineered
        away: ``.vectors.db`` holds restricted chunk text verbatim and
        is as sensitive as the most restricted item in it.
        """
        if not self.enforces_visibility:
            return _semantic.find_similar(
                self,
                text,
                top_k=top_k,
                threshold=threshold,
                exclude_slug=exclude_slug,
            )
        # Over-fetch rather than filter a fixed top-k. Trimming after the
        # cut makes the result *count* an existence oracle: eight matches
        # for a cleared user and three for an uncleared one says five
        # restricted items sit near that topic. Widen the fetch until k
        # visible rows are found or the ceiling is hit, so the count
        # depends on the corpus rather than on the viewer.
        wanted = top_k if top_k is not None else self.config.outmem.semantic.top_k
        index = self._labels()
        fetch = wanted
        seen: list[Match] = []
        while True:
            matches = _semantic.find_similar(
                self,
                text,
                top_k=fetch,
                threshold=threshold,
                exclude_slug=exclude_slug,
            )
            seen = [m for m in matches if self._repo_path_visible(m.rel_path, index)]
            if (
                len(seen) >= wanted
                or len(matches) < fetch  # the index is exhausted
                or fetch >= _OVERFETCH_CEILING
            ):
                return seen[:wanted]
            fetch = min(fetch * _OVERFETCH_FACTOR, _OVERFETCH_CEILING)

    def compartment_hint(self) -> dict[str, int]:
        """``label -> pages this session would gain by adding that label``.

        The deliberate disclosure that makes an opt-in default workable.
        Hiding is a property of **grants**, not of mode: for a user who
        does not hold ``hr``, HR content must be undetectable, but for
        one who *does* hold it and is merely running in mode ``∅``,
        being told the compartment exists and has content discloses
        nothing they are not already entitled to see. Without it a
        cleared user asking about parental leave gets nothing and never
        learns to switch compartment.

        **It takes no query, and that is the point.** The obvious
        implementation — count the items that matched *this question*
        and fell outside the mode — is a content oracle, because the
        model chooses the question. Asking for "twelve" and then
        "eleven" and comparing the two counts reads a fact out of a
        restricted page without ever retrieving it, one keyword at a
        time, and the model could then commit that fact to an open page.
        Relying on the model not to try is exactly what §1.3 forbids.

        So the number here is a property of the corpus, not of the
        query: how many additional pages a session scoped to that label
        would be able to see. It is the same answer for every question,
        so it carries no bits about any of them, and it still tells the
        user the thing they need — that the compartment is there and is
        worth switching to.

        Only labels the user holds are named; an aggregate count would
        leak the existence of compartments they do not hold. A label
        that adds nothing is omitted, so an empty compartment is not
        advertised.

        Empty for a bare store and for a user with no grants beyond the
        current mode.
        """
        if self._mode is None:
            return {}
        elsewhere = self._grants.read - self._mode
        if not elsewhere:
            return {}
        index = self._labels()
        counts: dict[str, int] = {}
        for label in elsewhere:
            widened = self._mode | {label}
            gained = sum(
                1
                for slug, labels in index.pages.items()
                if labels
                and not visible(labels, self._mode)
                and visible(labels, widened)
                and self._grants.may_read(labels)
                and slug not in self._alias_index()
            )
            if gained:
                counts[label] = gained
        return counts

    def semantic_reindex_path(self, rel_path: str) -> ReindexResult | None:
        """Reindex a single file by repo-relative path.

        Returns ``None`` for non-indexable or missing paths. The hash
        check inside :meth:`VectorStore.reindex_file` short-circuits
        unchanged content.
        """
        self._require_operator("reindexing")
        return _semantic.reindex_path(self, rel_path)

    def semantic_remove_path(self, rel_path: str) -> int:
        """Drop all chunks + vectors for ``rel_path``. Returns count removed."""
        self._require_operator("reindexing")
        return _semantic.remove_path(self, rel_path)

    def semantic_reindex_all(
        self,
        *,
        force: bool = False,
        max_concurrency: int = DEFAULT_SEMANTIC_REINDEX_CONCURRENCY,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> dict[str, Any]:
        """Walk every indexable file, sync the index, return a summary.

        Embeds files concurrently (≤ ``max_concurrency`` in flight);
        ``on_progress(done, total)`` fires per file. What gets walked
        follows ``semantic.index`` (``"pages"`` default | ``"pages+sources"``).

        The summary's ``dropped_paths`` lists wiki pages that exist on disk
        but did not make it into the index — check them, they are
        unreachable by search."""
        self._require_operator('reindexing')
        return _semantic.reindex_all(
            self,
            force=force,
            max_concurrency=max_concurrency,
            on_progress=on_progress,
        )

    def _maybe_reindex_commit_paths(self, paths: Sequence[str]) -> str | None:
        """Reindex any indexable file in ``paths`` and return the DB rel-path.

        Called from :meth:`_commit_paths` so the vector DB lands in the
        same commit as the page write. ``None`` when nothing changed.
        """
        return _semantic.maybe_reindex_commit_paths(self, paths)

    # ------------------------------------------------------------------
    # Sync
    # ------------------------------------------------------------------

    def pull(self) -> None:
        """``git pull --rebase`` from the configured remote / branch.

        Refused when the store was opened ``read_only=True`` — the
        rebase would mutate the working tree.
        """
        if self.config.read_only:
            raise OutmemError(
                f"wiki at {self.root} is opened read-only; refused to "
                "pull. Reopen with `WikiStore.open(..., read_only=False)` "
                "to sync from the remote."
            )
        _git_pull_rebase(self.root, remote=self.config.remote, branch=self.config.branch)
        # The cached backlinks key off HEAD; invalidate so the next
        # caller picks up the new state.
        self.backlinks_cache.invalidate()
        self._alias_map = None

    def push(self) -> None:
        """``git push`` to the configured remote / branch."""
        _git_push(self.root, remote=self.config.remote, branch=self.config.branch)

    def head(self) -> str | None:
        """Current HEAD SHA, or ``None`` if the repo has no commits."""
        return head_or_none(self.root)

    # ------------------------------------------------------------------
    # Identity + run marker
    # ------------------------------------------------------------------

    def contributors(self, *, refresh: bool = False) -> Contributors:
        """Parsed ``CONTRIBUTORS.md``. Cached after first read."""
        if refresh or self._contributors is None:
            self._contributors = load_contributors(self.contributors_path)
        return self._contributors

    def record_run(self, *, when: datetime | None = None) -> LastRun:
        """Record a successful run — used by the agent runtime."""
        return self.state.record_run(head=self.head(), timestamp=when)

    def last_run(self) -> LastRun | None:
        return self.state.last_run()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_scope(self, scope: str) -> tuple[Path, list[str] | None]:
        """Map ``scope`` to ``(search_root, relative_paths)`` for ripgrep.

        ``paths`` is ``None`` when the whole root is in scope, or a list
        of root-relative subpaths when the scope spans several trees
        (``sources``, which covers both the tracked and the local tree).

        ``wiki`` resolves to ``wiki/pages/`` — the editorial-page subtree
        — so ripgrep doesn't slosh through ``sources/`` or pick up
        ``index.md`` / ``AGENTS.md``.

        ``sources`` deliberately spans BOTH ``wiki/sources/`` and
        ``wiki/sources-local/``. Searching only one of them is the sharp
        edge this scope exists to remove: an agent told to "fall through
        to the sources" must not silently miss half the corpus because
        of a distribution policy it has no reason to know about.

        ``all`` enumerates the same trees explicitly rather than handing
        ripgrep the repo root. Two reasons, and the first is not
        cosmetic: **rg honours .gitignore while walking a directory**, so
        a bare-root search silently skips ``sources-local/`` — the
        "search everything" scope would be the one place the local tree
        went missing. Explicit paths are exempt from that filtering.
        Second, it keeps ``.outmem/``, ``.vectors.db`` and any stray
        repo-root files out of results the agent has to read past.
        """
        if scope == "wiki":
            return self.pages_path, None
        if scope in ("sources", "all"):
            trees: list[tuple[str, Path]] = [
                (f"{self.config.wiki_dir}/{SOURCES_DIR}", self.sources_path),
                (f"{self.config.wiki_dir}/{SOURCES_LOCAL_DIR}", self.sources_local_path),
            ]
            if scope == "all":
                trees = [
                    (self.pages_prefix().rstrip("/"), self.pages_path),
                    *trees,
                    (self.config.log_dir, self.log_path),
                ]
            # Only existing trees — sources-local/ is created lazily on
            # first local ingest, and a read-only store skips layout
            # creation entirely. An empty list means "nothing in scope",
            # which `search` renders as a clean no-hits result rather
            # than an rg failure on a missing path.
            return self.root, [rel for rel, path in trees if path.is_dir()]
        if scope == "log":
            return self.log_path, None
        hint = (
            "  ('raw' was removed in 0.10 — source documents now live in "
            "wiki/sources/ and wiki/sources-local/, both covered by "
            "scope='sources'. See docs/sources.md.)"
            if scope == "raw"
            else ""
        )
        raise OutmemError(
            f"Unknown search scope {scope!r}; expected 'wiki', 'sources', "
            f"'log', or 'all'.{hint}"
        )

    def _page_path(self, slug: str) -> Path:
        """Absolute filesystem path for ``slug``.

        Validates the slug as a side effect. ``index`` is special-cased
        to ``wiki/index.md`` (the auto-generated catalog).
        """
        if slug == INDEX_SLUG:
            return self.wiki_path / INDEX_FILENAME
        validate_slug(slug)
        return self.pages_path / slug_to_relpath(slug)

    def pages_prefix(self) -> str:
        """Repo-relative prefix every wiki page path starts with.

        One definition of "lives under ``wiki/pages/``" — the layout is
        configurable via ``wiki_dir``, so hand-rolling the f-string per
        call site is how a layout change goes half-applied.
        """
        return f"{self.config.wiki_dir}/{PAGES_DIR}/"

    def is_page_path(self, rel_path: str) -> bool:
        """True if ``rel_path`` names a file under ``wiki/pages/``."""
        return rel_path.startswith(self.pages_prefix())

    def _page_relpath(self, slug: str) -> str:
        """Repo-relative path string for ``slug`` (for ``git add`` etc)."""
        if slug == INDEX_SLUG:
            return f"{self.config.wiki_dir}/{INDEX_FILENAME}"
        return f"{self.pages_prefix()}{slug_to_relpath(slug).as_posix()}"

    def _ensure_layout(self) -> None:
        # ``sources-local/`` is intentionally absent: it is created on
        # first local ingest, together with its .gitignore entry, so a
        # wiki that never uses restricted material stays byte-identical
        # to one from before the split existed.
        for sub in (
            self.wiki_path,
            self.pages_path,
            self.log_path,
            self.sources_path,
        ):
            sub.mkdir(parents=True, exist_ok=True)
        self.state.ensure()

    def _maybe_clear_stale_lock(self) -> None:
        """Cleanup ``.git/index.lock`` if the user has opted in via config."""
        settings = self.config.outmem.git
        if not settings.remove_stale_lock:
            return
        clear_stale_index_lock(self.root, max_age_seconds=settings.stale_lock_seconds)

    def _maybe_auto_install_hook(self) -> None:
        """Ensure the pre-commit hook unless the user opted out.

        Best-effort and idempotent (see :func:`outmem.hooks.ensure_hook`):
        installs our hook when absent/stale, never clobbers a foreign one,
        never raises. Skipped for read-only stores (they must not mutate
        the repo). This is what lets manual ``git commit`` self-repair +
        reindex without the user remembering ``outmem hook install``."""
        if self.config.read_only:
            return
        if not self.config.outmem.git.auto_install_hook:
            return
        ensure_hook(self.root)

    def _ensure_gitignored(self, pattern: str, *, comment: str) -> bool:
        """Append ``pattern`` to the wiki's top-level ``.gitignore``.

        Idempotent and conservative: a pattern already present in any
        of its equivalent spellings is left alone, and an existing file
        is only ever appended to. Returns ``True`` if a line was added.

        Single funnel for every "outmem must keep this out of git" rule
        so the equivalence check (bare / leading-slash / trailing-slash)
        is written once — a second copy of it is how one caller ends up
        appending a duplicate line on every run.
        """
        gitignore = self.root / ".gitignore"
        existing = gitignore.read_text(encoding="utf-8") if gitignore.exists() else ""
        lines = {line.strip() for line in existing.splitlines() if line.strip()}
        bare = pattern.strip("/")
        if lines & {bare, f"/{bare}", f"{bare}/", f"/{bare}/"}:
            return False
        prefix = "" if not existing or existing.endswith("\n") else "\n"
        gitignore.write_text(
            existing + prefix + f"{comment}\n{pattern}\n", encoding="utf-8"
        )
        return True

    def _maybe_ignore_dotenv(self) -> None:
        """Keep ``.env`` out of git. Called once at :meth:`init`."""
        self._ensure_gitignored(".env", comment="# secrets — never committed")

    def ensure_sources_local(self) -> Path:
        """Create ``wiki/sources-local/`` and gitignore it. Returns the path.

        Called on the first local ingest rather than at ``init`` so a
        wiki that never touches restricted material keeps the exact
        layout it had before this feature existed.

        The ``.gitignore`` entry is written *before* the directory is
        populated — the ordering matters, because a source copied in
        first and ignored second is a source that a concurrent
        ``git add -A`` can still catch.
        """
        self._require_operator("creating the local source tree")
        if self.config.read_only:
            raise OutmemError(
                f"wiki at {self.root} is opened read-only; refused to create "
                f"{SOURCES_LOCAL_DIR}/."
            )
        self._ensure_gitignored(
            f"{self.config.wiki_dir}/{SOURCES_LOCAL_DIR}/",
            comment=(
                "# local-only sources — readable by the agent, never redistributed"
            ),
        )
        self.sources_local_path.mkdir(parents=True, exist_ok=True)
        return self.sources_local_path

    def _regenerate_index(self) -> None:
        """Rewrite ``wiki/index.md`` from the current wiki state.

        Called by :meth:`write_page` and :meth:`extend_page` so the
        index is always in lockstep with the page set. The caller
        stages ``wiki/index.md`` alongside the primary write so both
        land in the same commit.
        """
        text = index_page_text(self.pages_path)
        index_path = self.wiki_path / INDEX_FILENAME
        index_path.write_text(text, encoding="utf-8")

    def _seed_contributors(self) -> None:
        if self.contributors_path.exists():
            return
        identity = self.config.agent_identity
        body = (
            "# Contributors\n"
            "\n"
            "Team members known to the outmem steering loop. Each line:\n"
            "`- Name <email> [aliases: alt@x, alt2@y]`.\n"
            "\n"
            f"- {identity.name} <{identity.email}>\n"
        )
        self.contributors_path.write_text(body, encoding="utf-8")

    def _seed_agents_md(self) -> None:
        if self.agents_path.exists():
            return
        self.agents_path.write_text(starter_agents_md(), encoding="utf-8")

    def read_agents_md(self) -> str | None:
        """Return the wiki's ``AGENTS.md`` body if present, else ``None``.

        The agent-runtime injects this into the system prompt as the
        wiki-conventions section; see :func:`outmem.agent.render_system_prompt`.

        For a viewer, lines naming a hidden slug are dropped. A
        conventions file is exactly where someone writes "HR policies go
        under ``hr:``", and this text reaches the system prompt of every
        request. Line granularity is crude, but it is deterministic and
        it errs toward removing context rather than disclosing a name;
        ``outmem lint`` reports the mentions so they can be rewritten
        properly.
        """
        try:
            text = self.agents_path.read_text(encoding="utf-8").strip() or None
        except OSError:
            return None
        if text is None or not self.enforces_visibility:
            return text
        return self._drop_hidden_mentions(text)

    def _drop_hidden_mentions(self, text: str) -> str | None:
        """Remove lines naming a slug this viewer cannot see."""
        from outmem.slug import extract_slug_references

        index = self._labels()
        kept = [
            line
            for line in text.splitlines()
            if all(
                self._page_visible(ref, index)
                for ref in {r.slug for r in extract_slug_references(line)}
            )
        ]
        return "\n".join(kept).strip() or None

    def _commit_paths(self, paths: Sequence[str], *, subject: str) -> str:
        if self.config.read_only:
            raise OutmemError(
                f"wiki at {self.root} is opened read-only; refused to commit "
                f"{subject!r}. Reopen with `WikiStore.open(..., read_only=False)` "
                "to mutate it."
            )
        if not is_git_repo(self.root):
            raise OutmemError(f"{self.root} is not a git repo — call WikiStore.init() first.")
        commit_paths = list(paths)
        # Reindex first so the vector DB mutates *before* `git add` runs.
        db_rel = self._maybe_reindex_commit_paths(commit_paths)
        if db_rel is not None and (self.root / db_rel).exists():
            commit_paths.append(db_rel)
        add(self.root, commit_paths)
        sha = commit_as(
            self.root,
            message=subject,
            author_name=self.config.agent_identity.name,
            author_email=self.config.agent_identity.email,
        )
        # Backlinks are HEAD-keyed; invalidate so the next reader rebuilds.
        self.backlinks_cache.invalidate()
        self._alias_map = None
        try:
            return current_head(self.root)
        except OutmemError:
            return sha

    def close(self) -> None:
        """Release any open SQLite connections (vector DB, source registry)."""
        if self._vector_store is not None:
            self._vector_store.close()
            self._vector_store = None
        if self._source_registry is not None:
            self._source_registry.close()
            self._source_registry = None
        if self._source_registry_local is not None:
            self._source_registry_local.close()
            self._source_registry_local = None


def _format_log_filename(d: date) -> str:
    return d.isoformat()


def _body_text_key(body: str) -> str:
    """Identity of one adjudicated body text."""
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _reject_incomplete_body(
    body: str, *, tool: str, allowed: set[str] | None = None
) -> None:
    """Raise if ``body`` stops at an elision marker.

    Deliberately at the store layer rather than in the tool wrapper, so
    the refusal covers the CLI, the Python API, and any downstream app
    driving its own agent — not just outmem's own tool palette.

    The guard is a positional heuristic and therefore fallible, so every
    caller needs a way past it — but not the *same* way. A human has a
    one-step override (``allow_elision=True``, ``--allow-elision``, or a
    reviewer edit under the approval gate). A model has no argument at
    all: its only route is to submit the identical body again after
    being handed the refusal, which the tool wrapper registers via
    :meth:`WikiStore.allow_elision_body`. That asymmetry is the design.
    A flag the model could set would become a checkbox it learns to
    tick; re-sending unchanged costs a round trip, distinguishes "this
    text is right" from "I ran out of room" (a truncating model adds
    content rather than repeating itself), and leaves the page reported
    by ``outmem lint`` either way.
    """
    if allowed is not None and _body_text_key(body) in allowed:
        return

    # A tool-output marker in a page body is the same defect one step
    # earlier: outmem withheld content from a tool result and the page
    # was written from what was shown anyway. Caught here for the same
    # reason as the elision — while the source is still in context.
    sentinels = find_tool_sentinels(body)
    if sentinels:
        lines = tuple(f"line {e.line}: {e.text}" for e in sentinels[:3])
        raise IncompleteBodyError(
            f"{tool}: the body contains an outmem tool-output marker "
            f"({TOOL_SENTINEL_OPEN!r}) — that marker means outmem itself "
            f"withheld content from a tool result, so this page would be "
            f"built on material it never showed you. Read the source in "
            f"full (`read_source`, or narrow the range) and write the "
            f"passage from that. Offending: {'; '.join(lines)}",
            markers=lines,
        )

    found = find_elision_markers(body)
    if not found:
        return
    lines = tuple(f"line {e.line}: {e.text}" for e in found[:3])
    raise IncompleteBodyError(
        f"{tool}: the body stops at an elision marker "
        f"({found[0].marker!r}) — outmem pages must carry the complete "
        f"text, since nothing downstream can tell a shortened page from a "
        f"finished one. Write the full content; if it does not fit in one "
        f"call, send what fits now and add the rest with `append_page`. "
        f"If the ellipsis belongs to a quotation and the text is already "
        f"complete, submit the same body again unchanged — or pass "
        f"`--allow-elision` (CLI) / `allow_elision=True` (API). "
        f"Offending: {'; '.join(lines)}",
        markers=lines,
    )


def _merge_provenance(
    existing: Sequence[ProvenanceEntry], additions: Sequence[ProvenanceEntry]
) -> list[ProvenanceEntry]:
    """``existing`` plus any of ``additions`` it doesn't already cite.

    Compared by source path, so re-citing a source the page already
    points at is a no-op rather than a duplicate entry — an appended
    section usually draws on sources earlier sections already used, and
    the model has no cheap way to know which. Entries outmem can't read
    a path from are appended as-is: dropping an unparseable pointer
    would lose provenance an upstream ingester may rely on.
    """
    from outmem.lint import provenance_ref

    merged = list(existing)
    at: dict[str, int] = {}
    for index, entry in enumerate(merged):
        ref = provenance_ref(entry)
        if ref is not None:
            at.setdefault(ref, index)
    for entry in additions:
        ref = provenance_ref(entry)
        if ref is None:
            merged.append(entry)
            continue
        if ref in at:
            # Same source, re-cited. Take the new entry only when it is at
            # least as informative: a dict may carry an updated sha256 or
            # label, and skipping it would leave the page citing a
            # superseded version that `outmem stale` reports forever. But
            # a bare path must NEVER overwrite a dict — the appended
            # section usually re-cites what the page already has, and
            # replacing there would drop the recorded sha256 and silently
            # disable the staleness check for that page.
            current = merged[at[ref]]
            if isinstance(entry, dict) and entry != current:
                merged[at[ref]] = entry
            continue
        at[ref] = len(merged)
        merged.append(entry)
    return merged
