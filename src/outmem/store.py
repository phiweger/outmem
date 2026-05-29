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

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from outmem._store import import_vault as _import
from outmem._store import semantic as _semantic
from outmem._store import sources as _sources
from outmem._time import ensure_utc, utc_now

if TYPE_CHECKING:
    from outmem.semantic import Match, ReindexResult, VectorStore

from outmem.backlinks import BacklinkCache
from outmem.config import (
    CONFIG_FILENAME,
    DEFAULT_AGENT_EMAIL,
    DEFAULT_AGENT_NAME,
    DEFAULT_BRANCH,
    DEFAULT_REMOTE,
    OutmemConfig,
    load_dotenv_if_present,
    load_yaml_config,
    starter_agents_md,
    starter_yaml,
)
from outmem.exceptions import OutmemError, SlugError
from outmem.frontmatter import (
    ProvenanceEntry,
    WikiFrontmatter,
    parse_wiki_page,
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
from outmem.identity import Contributors, load_contributors
from outmem.index import (
    AGENTS_FILENAME,
    INDEX_FILENAME,
    INDEX_SLUG,
    editorial_pages,
    index_page_text,
)
from outmem.search import DEFAULT_RESULT_BYTES, SearchResult, rg_available, search
from outmem.slug import PAGES_DIR, slug_to_relpath, validate_slug
from outmem.sources import (
    SOURCES_DIR,
    IngestionRecord,
    SourceEntry,
    SourceRegistry,
)
from outmem.state import LastRun, OutmemState

log = logging.getLogger(__name__)


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
    e.g. ``store.config.outmem.semantic.enabled``,
    ``store.config.outmem.git.remove_stale_lock``,
    ``store.config.outmem.model``.
    """

    root: Path
    outmem: OutmemConfig = field(default_factory=OutmemConfig)
    agent_identity: AgentIdentity = field(default_factory=AgentIdentity)
    remote: str = DEFAULT_REMOTE
    branch: str = DEFAULT_BRANCH
    wiki_dir: str = "wiki"
    raw_dir: str = "raw"
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


class WikiStore:
    """Filesystem-backed wiki — the unit downstream code interacts with."""

    def __init__(self, config: WikiStoreConfig) -> None:
        self.config = config
        self.root = Path(config.root)
        self.wiki_path = self.root / config.wiki_dir
        self.pages_path = self.wiki_path / PAGES_DIR
        self.raw_path = self.root / config.raw_dir
        self.log_path = self.root / config.log_dir
        self.sources_path = self.wiki_path / SOURCES_DIR
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
        # Lazily-opened resources holding sqlite connections.
        self._vector_store: VectorStore | None = None
        self._source_registry: SourceRegistry | None = None

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

        Creates the subdirectories (``wiki/``, ``raw/``, ``log/``,
        ``.outmem/``) if they don't yet exist. Does not initialise a
        git repo — :meth:`init` is the explicit constructor for that.
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
        scaffolds ``raw/``, ``wiki/``, ``log/``, ``.outmem/``, seeds
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

        Raises :class:`OutmemError` if the page does not exist;
        :class:`outmem.exceptions.FrontmatterError` if frontmatter is
        missing or malformed.
        """
        path = self._page_path(slug)
        if not path.exists():
            raise OutmemError(f"No such wiki page: {slug}")
        text = path.read_text(encoding="utf-8")
        frontmatter, body = parse_wiki_page(text)
        return WikiPage(slug=slug, frontmatter=frontmatter, body=body, path=path)

    def exists(self, slug: str) -> bool:
        try:
            return self._page_path(slug).exists()
        except SlugError:
            return False

    def list_slugs(self) -> list[str]:
        """Every editorial slug under ``wiki/pages/``, alphabetically.

        The auto-generated ``index`` slug is hidden — it's structural,
        not content. Consumers who need to read it can still call
        ``read("index")`` directly.
        """
        if not self.pages_path.is_dir():
            return []
        from outmem.slug import relpath_to_slug
        return sorted(
            relpath_to_slug(p.relative_to(self.pages_path))
            for p in editorial_pages(self.pages_path)
        )

    def search(
        self,
        pattern: str,
        *,
        scope: str = "wiki",
        case_insensitive: bool = False,
        fixed_strings: bool = False,
        max_bytes: int = DEFAULT_RESULT_BYTES,
        max_hits: int | None = None,
    ) -> SearchResult:
        """Run a ripgrep search anchored at the store.

        ``scope`` is one of ``"wiki"``, ``"raw"``, ``"log"``, or ``"all"``.
        Tier 1 of the agent's retrieval palette is ``scope="wiki"``;
        Tier 2 falls through to ``"raw"`` (spec v0.5 §8).
        """
        path = self._resolve_scope(scope)
        return search(
            pattern,
            root=path,
            case_insensitive=case_insensitive,
            fixed_strings=fixed_strings,
            max_bytes=max_bytes,
            max_hits=max_hits,
        )

    def backlinks(self, slug: str) -> tuple[str, ...]:
        """Slugs of pages that link to ``slug`` at the current HEAD."""
        validate_slug(slug)
        return self.backlinks_cache.referrers(slug, head_or_none(self.root))

    def history(self, slug: str) -> list[CommitInfo]:
        """Per-page commit history (newest first), tracking renames."""
        validate_slug(slug)
        return page_history(self.root, slug, wiki_dir=self.config.wiki_dir)

    def evolution(
        self,
        slugs: Sequence[str],
        *,
        include_log: bool = True,
    ) -> str:
        """Raw ``git log -p`` stream — the EXPANSION-pattern helper."""
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
        """
        if head_or_none(self.root) is None:
            # No commits yet; nothing to steer on.
            return []
        if since is None:
            marker = self.state.last_run()
            since = marker.timestamp if marker else default_window
        paths = [self.config.wiki_dir]
        if include_log:
            paths.append(self.config.log_dir)
        return log_since(
            self.root,
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
    ) -> str:
        """Create a new wiki page (under ``wiki/pages/``) and commit it.

        The on-disk path is derived from the slug by
        :func:`outmem.slug.slug_to_relpath` (``:`` → ``/``,
        appending ``.md``). Frontmatter is built per spec v0.5 §4.
        The commit message defaults to ``compact: <slug>`` (TARS Retained
        depends on the prefix grammar — see spec §9).
        ``wiki/index.md`` is regenerated and staged in the same commit.
        """
        if slug == INDEX_SLUG:
            raise OutmemError(
                "Cannot write to the reserved 'index' slug — `wiki/index.md` "
                "is auto-maintained by outmem on every page write."
            )
        page_path = self._page_path(slug)
        if page_path.exists():
            raise OutmemError(f"Page already exists: {slug}. Use extend_page() to edit it.")
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

    def extend_page(
        self,
        slug: str,
        *,
        body: str,
        commit_subject: str | None = None,
    ) -> str:
        """Replace the body of an existing page and commit.

        Frontmatter is preserved; ``updated`` is bumped to now. The
        commit message defaults to ``extend: <slug>``. ``wiki/index.md``
        is regenerated and staged in the same commit (title or tag
        edits will surface there).
        """
        if slug == INDEX_SLUG:
            raise OutmemError(
                "Cannot edit the reserved 'index' slug — `wiki/index.md` "
                "is auto-maintained by outmem on every page write."
            )
        page = self.read(slug)
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
        """
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
        commit: bool = True,
    ) -> SourceEntry:
        """Copy a source file into ``wiki/sources/`` and register it.

        Content-addressed: the file lands at
        ``wiki/sources/[<into>/]<sha[:12]>/<filename>``. Re-adding the
        same content is a no-op; different content under the same
        slug refreshes the registry row.
        """
        return _sources.add_source(
            self, source, into_subdir=into_subdir, rename=rename, commit=commit
        )

    def list_sources(self) -> list[SourceEntry]:
        """Every registered source, ordered by relative path."""
        return _sources.list_sources(self)

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

    def semantic_enabled(self) -> bool:
        """Whether ``semantic.enabled: true`` is set in ``config.yaml``."""
        return _semantic.enabled(self)

    def semantic_index_is_empty(self) -> bool:
        """True if semantic is enabled but the index has no files yet
        (i.e. ``outmem reindex`` hasn't run). Opens the vector store, so
        the first call pays the one-time embedder dimension probe."""
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

    def semantic_reindex_all(self, *, force: bool = False) -> dict[str, Any]:
        """Walk every indexable file, sync the index, return a summary."""
        return _semantic.reindex_all(self, force=force)

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

    def _resolve_scope(self, scope: str) -> Path:
        """Map ``scope`` ∈ {wiki, raw, log, all} to a search root.

        ``wiki`` scope resolves to ``wiki/pages/`` — the editorial-page
        subtree — so ripgrep doesn't slosh through ``sources/`` or pick
        up ``index.md`` / ``AGENTS.md``.
        """
        if scope == "wiki":
            return self.pages_path
        if scope == "raw":
            return self.raw_path
        if scope == "log":
            return self.log_path
        if scope == "all":
            return self.root
        raise OutmemError(
            f"Unknown search scope {scope!r}; expected 'wiki', 'raw', 'log', or 'all'."
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

    def _page_relpath(self, slug: str) -> str:
        """Repo-relative path string for ``slug`` (for ``git add`` etc)."""
        if slug == INDEX_SLUG:
            return f"{self.config.wiki_dir}/{INDEX_FILENAME}"
        return f"{self.config.wiki_dir}/{PAGES_DIR}/{slug_to_relpath(slug).as_posix()}"

    def _ensure_layout(self) -> None:
        for sub in (
            self.wiki_path,
            self.pages_path,
            self.raw_path,
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

    def _maybe_ignore_dotenv(self) -> None:
        """Ensure ``.env`` is in the wiki's top-level ``.gitignore``.

        Idempotent — only appends if the pattern isn't already present.
        Keeps secrets out of git.
        """
        gitignore = self.root / ".gitignore"
        existing = gitignore.read_text(encoding="utf-8") if gitignore.exists() else ""
        lines = {line.strip() for line in existing.splitlines() if line.strip()}
        if ".env" in lines or "/.env" in lines:
            return
        additions = ["# secrets — never committed", ".env"]
        prefix = "" if not existing or existing.endswith("\n") else "\n"
        gitignore.write_text(
            existing + prefix + "\n".join(additions) + "\n", encoding="utf-8"
        )

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


def _format_log_filename(d: date) -> str:
    return d.isoformat()
