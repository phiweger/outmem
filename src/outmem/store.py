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

import hashlib
import logging
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from outmem._store import import_vault as _import
from outmem._store import semantic as _semantic
from outmem._store import sources as _sources
from outmem._store.locking import repo_commit_lock
from outmem._time import ensure_utc, utc_now

if TYPE_CHECKING:
    from outmem.index import PageLoadFailure
    from outmem.lint import ProvenanceAnnotation
    from outmem.semantic import Match, ReindexResult, VectorStore
    from outmem.sources import KeyCandidate, RegistryAudit, RekeyResult, StaleCitation

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
    SlugError,
    UnregisteredProvenanceError,
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
from outmem.repo import find_repo_root, qualify_subject
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
    # The git repository ``root`` commits into. Equal to ``root`` for a
    # standalone wiki; an ancestor when several wikis share one repo (see
    # :mod:`outmem.repo`). Resolved by ``open``/``init``; construct a
    # config by hand and it defaults to ``root``, which is the
    # single-wiki behaviour.
    repo: Path | None = None
    # Registry name of this wiki within its repository — ``None`` for a
    # standalone wiki, which has nothing to distinguish it from.
    wiki_name: str | None = None
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


def ensure_gitignored(root: Path, pattern: str, *, comment: str) -> bool:
    """Append ``pattern`` to ``root``'s ``.gitignore``. Returns True if added.

    Idempotent and conservative: a pattern already present in any of its
    equivalent spellings is left alone, and an existing file is only ever
    *appended to* — never rewritten. Somebody's `__pycache__/` and
    `.vectors.db` rules are not ours to drop.

    Single funnel for every "outmem must keep this out of git" rule so the
    equivalence check (bare / leading-slash / trailing-slash) is written
    once. A second copy of it is how one caller ends up appending a
    duplicate line on every run — or, worse, writing the file wholesale.
    """
    gitignore = root / ".gitignore"
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


def _repo_prefix(root: Path, repo: Path) -> str:
    """``root``'s location inside ``repo``, as a POSIX prefix or ``""``.

    ``""`` means the wiki *is* the repository, so a wiki-relative path is
    already repo-relative and nothing needs translating — the standalone
    case, and the reason this returns a string rather than a Path.
    """
    if root == repo:
        return ""
    try:
        rel = root.resolve().relative_to(repo.resolve()).as_posix()
    except (OSError, ValueError) as exc:
        raise OutmemError(
            f"wiki root {root} is not inside its repository {repo}."
        ) from exc
    return "" if rel == "." else f"{rel}/"


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

    repo, _prefix, wiki_name = find_repo_root(root)
    return WikiStoreConfig(
        root=root,
        repo=repo,
        wiki_name=wiki_name,
        outmem=yaml_cfg,
        agent_identity=agent_identity,
        remote=remote or yaml_cfg.remote.name,
        branch=branch or yaml_cfg.remote.branch,
        read_only=read_only,
    )


class WikiStore:
    """Filesystem-backed wiki — the unit downstream code interacts with."""

    def __init__(self, config: WikiStoreConfig) -> None:
        self.config = config
        self.root = Path(config.root)
        # `root` is the *content* root — pages, sources, log, config.yaml,
        # `.vectors.db`, `.outmem/` all hang off it. `repo` is the *git*
        # root, which every `git_ops` call takes. They are the same
        # directory for a standalone wiki, and differ only when several
        # wikis share one repository; `repo_prefix` is then this wiki's
        # location within it, and is what turns a wiki-relative path into
        # one git will accept.
        self.repo = Path(config.repo) if config.repo is not None else self.root
        self.repo_prefix = _repo_prefix(self.root, self.repo)
        self.wiki_name = config.wiki_name
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
        self._contributors: Contributors | None = None
        # Slugs already warned-about by read()'s frontmatter self-heal, so
        # repeated reads of one broken page log once, not per read.
        self._healed_slugs: set[str] = set()
        # Lazily-opened resources holding sqlite connections.
        self._vector_store: VectorStore | None = None
        self._source_registry: SourceRegistry | None = None
        # Separate handle: each source tree carries its own registry, so
        # the tracked one never records a local source's filename, hash,
        # or origin path.
        self._source_registry_local: SourceRegistry | None = None
        self._alias_map: dict[str, str] | None = None
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
        # A wiki placed inside an existing multi-wiki repository joins
        # that repository's history rather than starting one of its own —
        # a nested `.git` would make its commits invisible to the repo
        # that contains it.
        repo, _prefix, _name = find_repo_root(root)
        if repo == root:
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
        try:
            return self._page_path(self.resolve_slug(slug)).exists()
        except SlugError:
            return False

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
        return self._alias_index().get(slug, slug)

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
        return search(
            pattern,
            root=path,
            paths=paths,
            case_insensitive=case_insensitive,
            fixed_strings=fixed_strings,
            context=context,
            max_bytes=max_bytes,
            max_hits=max_hits,
        )

    def backlinks(self, slug: str) -> tuple[str, ...]:
        """Slugs of pages that link to ``slug`` at the current HEAD."""
        slug = self.resolve_slug(slug)
        validate_slug(slug)
        return self.backlinks_cache.referrers(slug, head_or_none(self.repo))

    def history(self, slug: str) -> list[CommitInfo]:
        """Per-page commit history (newest first), tracking renames."""
        slug = self.resolve_slug(slug)
        validate_slug(slug)
        return page_history(
            self.repo, slug, wiki_dir=self.config.wiki_dir, prefix=self.repo_prefix
        )

    def evolution(
        self,
        slugs: Sequence[str],
        *,
        include_log: bool = True,
    ) -> str:
        """Raw ``git log -p`` stream — the EXPANSION-pattern helper."""
        slugs = [self.resolve_slug(s) for s in slugs]
        return topic_evolution(
            self.repo,
            slugs,
            wiki_dir=self.config.wiki_dir,
            include_log=include_log,
            log_dir=self.config.log_dir,
            prefix=self.repo_prefix,
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
        """
        if head_or_none(self.repo) is None:
            # No commits yet; nothing to steer on.
            return []
        if since is None:
            marker = self.state.last_run()
            since = marker.timestamp if marker else default_window
        paths = [self._repo_relpath(self.config.wiki_dir)]
        if include_log:
            paths.append(self._repo_relpath(self.config.log_dir))
        return log_since(
            self.repo,
            since=since,
            paths=paths,
            exclude_author=self.config.agent_identity.email,
        )

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
        allow_unregistered_provenance: bool = False,
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
                raise OutmemError(
                    f"{slug!r} is an alias of {owner!r}; writing a page here would "
                    f"silently retarget every [[{slug}]] link. Remove the alias from "
                    f"{owner!r} first, or choose another slug."
                )
            # Before anything touches disk, so a refused write leaves no
            # half-applied state — no page file, no regenerated index, no
            # commit.
            if not allow_unregistered_provenance:
                _reject_unregistered_provenance(
                    self, list(provenance or []), tool="write_page"
                )
            now = utc_now()
            frontmatter = WikiFrontmatter(
                title=title,
                slug=slug,
                provenance=list(provenance or []),
                created=(created or now).replace(microsecond=0),
                updated=now,
                tags=list(tags or []),
                extra=dict(extra or {}),
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

        ``alias=True`` (default) records ``old_slug`` in the moved page's
        ``aliases:``, so the old name keeps resolving. That matters even
        with a perfect link rewrite: references *outside* the wiki —
        tickets, configs, a shipped answer citing a slug — are ones
        outmem cannot reach. The alias is the safety net under the
        rewrite, not a replacement for it.

        Returns the new HEAD SHA.
        """
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
        """
        if rel_path is None:
            return [
                ref
                for tree in _sources.existing_trees(self)
                for ref in _sources.get_registry(self, tree).refs(None)
            ]
        found = _sources.resolve_source(self, rel_path)
        tree, key = found if found is not None else (_sources.tracked_tree(self), rel_path)
        return _sources.get_registry(self, tree).refs(key)

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
        allow_unregistered_provenance: bool = False,
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
            if not allow_elision:
                _reject_incomplete_body(
                    body, tool="extend_page", allowed=self._elision_allowed
                )
            page = self.read(slug)
            if provenance is not None:
                # Only the entries being *set*. `provenance=None` leaves the
                # page's own untouched and is not re-validated — a page that
                # already carries a dangling ref is lint's to report, and
                # re-checking here would refuse an unrelated body edit.
                if not allow_unregistered_provenance:
                    _reject_unregistered_provenance(
                        self, list(provenance), tool="extend_page"
                    )
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
        allow_unregistered_provenance: bool = False,
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
            if not allow_elision:
                _reject_incomplete_body(
                    body, tool="append_page", allowed=self._elision_allowed
                )
            page = self.read(slug)
            existing = page.body.rstrip()
            merged = f"{existing}\n\n{body.strip()}\n" if existing else f"{body.strip()}\n"
            if provenance:
                # Validated as a whole before merging: one bad ref refuses
                # the call and nothing is appended, rather than landing the
                # good refs and the body and leaving the caller to guess
                # what took.
                if not allow_unregistered_provenance:
                    _reject_unregistered_provenance(
                        self, list(provenance), tool="append_page"
                    )
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
        self._regenerate_index()
        rel = f"{self.config.wiki_dir}/{INDEX_FILENAME}"
        if not commit:
            return None
        if not path_is_dirty(self.repo, self._repo_relpath(rel)):
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
        """
        with self._write_lock:
            if not topic.strip():
                raise OutmemError("append_log: topic must be non-empty.")
            ts = ensure_utc(when) if when else utc_now()
            log_date = ts.date()
            log_file = self.log_path / f"{_format_log_filename(log_date)}.md"
            log_file.parent.mkdir(parents=True, exist_ok=True)

            existed = log_file.exists()
            existing = log_file.read_text(encoding="utf-8") if existed else ""
            prefix = "" if not existed else "\n"
            if not existed:
                existing = f"# {log_date.isoformat()}\n\n"
            log_file.write_text(existing + prefix + content.rstrip() + "\n", encoding="utf-8")

            return self._commit_paths(
                [f"{self.config.log_dir}/{log_file.name}"],
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
        """
        return _sources.add_source(
            self,
            source,
            into_subdir=into_subdir,
            rename=rename,
            as_key=as_key,
            local=local,
            commit=commit,
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
                out.setdefault(key, []).append(page.slug)
        return out, failures

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
        pages, _failures = load_editorial_pages(self.pages_path)
        for page in pages:
            for entry in page.frontmatter.provenance:
                annotation = provenance_annotation(entry)
                ref = provenance_ref(entry)
                if ref is None or not annotation:
                    continue
                _tree, key = _sources.split_tree_prefix(self, ref)
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
        """Every registered source, ordered by relative path."""
        return _sources.list_sources(self, include_missing=include_missing)

    def get_source(self, rel_path: str) -> SourceEntry | None:
        """Lookup a single registered source by its relative path."""
        return _sources.get_source(self, rel_path)

    def read_source(self, rel_path: str, *, max_chars: int | None = None) -> str:
        """Return the text of a source file, capped at ``max_chars``."""
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
        """
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
        return _import.import_vault(self, Path(source).expanduser(), force=force)

    # ------------------------------------------------------------------
    # Semantic index — implementations live in :mod:`outmem._store.semantic`
    # ------------------------------------------------------------------

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
        """Return the top semantic matches for ``text``."""
        return _semantic.find_similar(
            self,
            text,
            top_k=top_k,
            threshold=threshold,
            exclude_slug=exclude_slug,
        )

    def semantic_reindex_path(self, rel_path: str) -> ReindexResult | None:
        """Reindex a single file by repo-relative path.

        Returns ``None`` for non-indexable or missing paths. The hash
        check inside :meth:`VectorStore.reindex_file` short-circuits
        unchanged content.
        """
        return _semantic.reindex_path(self, rel_path)

    def semantic_remove_path(self, rel_path: str) -> int:
        """Drop all chunks + vectors for ``rel_path``. Returns count removed."""
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
        _git_pull_rebase(self.repo, remote=self.config.remote, branch=self.config.branch)
        # The cached backlinks key off HEAD; invalidate so the next
        # caller picks up the new state.
        self.backlinks_cache.invalidate()
        self._alias_map = None

    def push(self) -> None:
        """``git push`` to the configured remote / branch."""
        _git_push(self.repo, remote=self.config.remote, branch=self.config.branch)

    def head(self) -> str | None:
        """Current HEAD SHA, or ``None`` if the repo has no commits."""
        return head_or_none(self.repo)

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
        clear_stale_index_lock(self.repo, max_age_seconds=settings.stale_lock_seconds)

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
        ensure_hook(self.repo)

    def _ensure_gitignored(self, pattern: str, *, comment: str) -> bool:
        """Append ``pattern`` to the wiki's top-level ``.gitignore``."""
        return ensure_gitignored(self.root, pattern, comment=comment)

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
        """
        try:
            return self.agents_path.read_text(encoding="utf-8").strip() or None
        except OSError:
            return None

    def _repo_relpath(self, rel: str) -> str:
        """A wiki-relative path as git sees it, from the repository root.

        Refuses anything that would leave the wiki. Several wikis share
        one index and one history, so a path that escapes is a store
        staging another wiki's files — and the guard sits here, at the
        one place paths cross into git, rather than at each of the eight
        callers that would each have to remember.

        A no-op in effect for a standalone wiki, where the two roots
        coincide and the prefix is empty.
        """
        path = PurePosixPath(rel)
        if path.is_absolute() or ".." in path.parts:
            raise OutmemError(
                f"refusing to stage {rel!r}: commit paths are relative to "
                f"the wiki at {self.root} and may not leave it."
            )
        return f"{self.repo_prefix}{rel}" if self.repo_prefix else rel

    def stage(self, paths: Sequence[str]) -> None:
        """``git add`` wiki-relative paths, without committing.

        The pre-commit hook needs this: the human's commit *is* the
        commit, so derived artefacts are staged into it rather than
        committed on their own.
        """
        add(self.repo, [self._repo_relpath(p) for p in paths])

    def _commit_paths(self, paths: Sequence[str], *, subject: str) -> str:
        if self.config.read_only:
            raise OutmemError(
                f"wiki at {self.root} is opened read-only; refused to commit "
                f"{subject!r}. Reopen with `WikiStore.open(..., read_only=False)` "
                "to mutate it."
            )
        if not is_git_repo(self.repo):
            # Two different situations, two different remedies. A
            # standalone wiki needs `WikiStore.init`; a wiki that found
            # its repository through a `wikis.yaml` needs that
            # *repository* initialised, and calling `init` on the wiki
            # would not help — the old message sent people the wrong way.
            if self.repo != self.root:
                raise OutmemError(
                    f"{self.repo} holds a wikis.yaml listing this wiki but is "
                    "not a git repo. Run `outmem repo init` there, or remove "
                    "the registry entry to use this wiki standalone."
                )
            raise OutmemError(f"{self.repo} is not a git repo — call WikiStore.init() first.")
        commit_paths = list(paths)
        # Reindex first so the vector DB mutates *before* `git add` runs.
        # These paths stay wiki-relative: the vector DB keys on them, and
        # a key that moved when a wiki was placed in a repo would orphan
        # every chunk already stored.
        db_rel = self._maybe_reindex_commit_paths(commit_paths)
        if db_rel is not None and (self.root / db_rel).exists():
            commit_paths.append(db_rel)
        # Translate to repo-relative here and nowhere else. Every caller
        # passes wiki-relative paths, so one wiki cannot name another's
        # files by construction rather than by each caller remembering.
        #
        # Staging and committing are one operation across the repository,
        # not two: between them the git index is shared state, and another
        # process committing in the window carries these paths into *its*
        # commit under its subject and author. `_write_lock` is per-store
        # and per-process, so it cannot cover two wikis or two processes.
        with repo_commit_lock(self.repo):
            add(self.repo, [self._repo_relpath(p) for p in commit_paths])
            sha = commit_as(
                self.repo,
                message=qualify_subject(subject, self.wiki_name),
                author_name=self.config.agent_identity.name,
                author_email=self.config.agent_identity.email,
            )
        # Backlinks are HEAD-keyed; invalidate so the next reader rebuilds.
        self.backlinks_cache.invalidate()
        self._alias_map = None
        try:
            return current_head(self.repo)
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


def _reject_unregistered_provenance(
    store: WikiStore, entries: Sequence[ProvenanceEntry], *, tool: str
) -> None:
    """Raise if any entry cites a source with no registry row.

    At the store layer, not the tool wrapper, so the refusal covers the
    CLI, the Python API and any downstream app driving its own agent —
    the same reasoning as :func:`_reject_incomplete_body`.

    **A row existing is the criterion, not a file existing.** Citing a
    superseded version is legal and ``outmem stale`` / ``superseded_ok:``
    are the machinery for it; a row whose file was later deleted is a
    lint concern (``stale-provenance``), not a write-time one. Both of
    those resolve through ``get_source``, which falls back to the
    registry when the file is gone.

    Entries carrying no reference at all — an annotation-only mapping —
    are left to lint, as before: there is nothing here to resolve.
    """
    from outmem.lint import provenance_ref

    missing = [
        ref
        for entry in entries
        if (ref := provenance_ref(entry)) is not None
        and store.get_source(ref) is None
    ]
    if not missing:
        return
    listed = ", ".join(repr(r) for r in missing)
    raise UnregisteredProvenanceError(
        f"{tool}: provenance names {'a source' if len(missing) == 1 else 'sources'} "
        f"the registry does not hold: {listed}. A citation to nothing opts the "
        "page out of staleness and supersession reporting while looking like a "
        "real one, so it is refused here rather than reported by `outmem lint` "
        "later. Register the source first (`outmem ingest`, or "
        "`WikiStore.add_source`), or cite one that is already registered "
        "(`list_sources`, `outmem sources list`). "
        "Accepted forms: '<sha>/file.md', 'sources/<sha>/file.md', "
        "'sources-local/<sha>/file.md', each optionally prefixed 'wiki/'.",
        tuple(missing),
    )


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
