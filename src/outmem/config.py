"""Configuration for an outmem wiki.

Two files at the wiki root are consulted on :meth:`WikiStore.open`:

* ``config.yaml`` — non-secret config (model, agent identity, git
  behaviour, remote name). Committed alongside the wiki by default
  so a team shares the same defaults; individual users can override
  via environment variables.
* ``.env`` — secrets (API keys). Loaded via :mod:`python-dotenv` so
  values land in ``os.environ`` before PydanticAI consults them.
  Gitignored by default.

Resolution order, highest priority first:

1. Explicit constructor argument (``WikiStore.open(..., remote=…)``)
2. Environment variable (``OUTMEM_MODEL``, ``ANTHROPIC_API_KEY``, …)
3. ``config.yaml`` value
4. Built-in default

The loader is *lenient*: missing files return empty config; malformed
YAML logs a warning and returns empty config; unknown keys are
preserved in ``extra`` so a forward-compatible config doesn't error
out when this code is older than the file.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import find_dotenv, load_dotenv

log = logging.getLogger(__name__)

CONFIG_FILENAME = "config.yaml"

DEFAULT_MODEL = "anthropic:claude-sonnet-4-6"
DEFAULT_AGENT_NAME = "outmem agent"
DEFAULT_AGENT_EMAIL = "agent@host"
DEFAULT_REMOTE = "origin"
DEFAULT_BRANCH = "main"

DEFAULT_REMOVE_STALE_LOCK = True
DEFAULT_STALE_LOCK_SECONDS = 60
DEFAULT_RETRY_ON_LOCK = True
DEFAULT_AUTO_INSTALL_HOOK = True
DEFAULT_RETRIEVAL_STRATEGY = "rerank(bm25)"
# Production search_wiki default: BM25 keyword shortlist, LLM-gated for
# relevance — one cheap Haiku call per query, but lifts recall on
# paraphrase-heavy questions where plain term overlap misses the right
# page. The bm25 source needs no semantic index. Run optimize_retrieval
# and `result.save(rank, store)` to swap in a measured winner
# (rerank(semantic), bm25+semantic, …).

DEFAULT_SOURCE_MAX_CHARS = 200_000  # cap on `read_source` tool returns

DEFAULT_SEMANTIC_MODEL = "openai:text-embedding-3-small"
DEFAULT_SEMANTIC_DB_FILENAME = ".vectors.db"
DEFAULT_SEMANTIC_CHUNK_SIZE = 2000
DEFAULT_SEMANTIC_CHUNK_MAX = 8000
DEFAULT_SEMANTIC_OVERLAP_PARAGRAPHS = 1
DEFAULT_SEMANTIC_SIMILARITY_THRESHOLD = 0.80
DEFAULT_SEMANTIC_TOP_K = 5
DEFAULT_SEMANTIC_REINDEX_CONCURRENCY = 8  # in-flight embed calls during reindex

# What `outmem reindex` walks. `pages` keeps raw ingested sources out of the
# vector store; `pages+sources` (the default, and the historical behaviour)
# indexes both.
SEMANTIC_INDEX_PAGES = "pages"
SEMANTIC_INDEX_PAGES_AND_SOURCES = "pages+sources"
SEMANTIC_INDEX_CHOICES = (SEMANTIC_INDEX_PAGES, SEMANTIC_INDEX_PAGES_AND_SOURCES)
DEFAULT_SEMANTIC_INDEX = SEMANTIC_INDEX_PAGES_AND_SOURCES
# Prepend "<title> — <tags>" to every chunk before embedding. Off by default:
# flipping it re-embeds every page on the next reindex.
DEFAULT_SEMANTIC_EMBED_FRONTMATTER = False

DEFAULT_APPROVAL_REQUIRED_FOR_WRITES = False

# The cheap gate model the rerank strategy uses, and the per-candidate
# excerpt cap it feeds that model.
DEFAULT_RELEVANCE_MODEL = "anthropic:claude-haiku-4-5"
DEFAULT_RELEVANCE_CONTEXT_CHARS = 2000

# Defaults for the optional retrieval-tuning tool (outmem.optimize). It is an
# API/script tool — not config.yaml-driven — but its defaults live here, the
# one defaults home, rather than inline across the optimize modules.
DEFAULT_OPTIMIZE_STRATEGY = "lexical"
DEFAULT_OPTIMIZE_MAX_CANDIDATES = 30        # keyword-net width before reranking
DEFAULT_OPTIMIZE_MAX_RELEVANT = 8           # pages the rerank block keeps
DEFAULT_OPTIMIZE_SEMANTIC_TOP_K = 8         # neighbours for semantic / hybrid
DEFAULT_OPTIMIZE_RRF_K = 60                 # Reciprocal Rank Fusion constant
DEFAULT_OPTIMIZE_PER_PAGE = 2               # generated questions per page
DEFAULT_OPTIMIZE_CONCURRENCY = 8            # in-flight model calls (gen + eval)
DEFAULT_OPTIMIZE_K = 1                      # Hit@k cutoff (k=1 stays discriminative on small wikis)
DEFAULT_OPTIMIZE_RERANK_SOURCE = "lexical"  # candidate generator that feeds the rerank LLM gate
DEFAULT_OPTIMIZE_MAX_EVALS = 12             # optimizer turn budget
DEFAULT_OPTIMIZE_MAX_FAILURES_SHOWN = 6     # failing questions shown per eval
DEFAULT_OPTIMIZE_UNANSWERABLE_LIMIT = 20    # gap-log questions harvested

DEFAULT_LOGFIRE_ENABLED = False
LOGFIRE_SERVICE_NAME = "outmem"

# Anthropic prompt-caching keys for ``model_settings`` (pydantic_ai passes
# them through; no-ops on non-Anthropic models). Caching the static system
# prompt + tool-def array across the many calls an agent or tuning loop
# makes cuts the bill ~5-10x. Spread into a per-call ``model_settings``
# dict alongside ``max_tokens``; agents that expose tools use the
# ``*_WITH_TOOLS`` variant to also cache the tool schemas.
ANTHROPIC_CACHE_SETTINGS: dict[str, bool] = {
    "anthropic_cache": True,               # top-level auto-cache breakpoint
    "anthropic_cache_instructions": True,  # cache the system-prompt block
}
ANTHROPIC_CACHE_WITH_TOOLS: dict[str, bool] = {
    **ANTHROPIC_CACHE_SETTINGS,
    "anthropic_cache_tool_definitions": True,  # cache the tool-def array
}

# Error string when a caller hits a semantic path but the index hasn't
# been built — shared across the CLI, the WikiStore facet, the optimizer,
# and the PydanticAI adapter so the user sees the same fix-it everywhere.
# Semantic is implied by use (a `semantic`/`hyde`/`*+semantic` retrieval
# strategy, or `find_similar`), not a config flag: build the index to
# turn it on.
SEMANTIC_UNAVAILABLE_HELP = (
    "semantic index not built — run `outmem reindex` "
    "(needs `pip install outmem[semantic]`)."
)

@dataclass
class SourceSettings:
    """Resilience knobs for source ingestion."""

    max_chars: int = DEFAULT_SOURCE_MAX_CHARS


@dataclass
class ApprovalSettings:
    """Human-in-the-loop gates around agent writes.

    When ``required_for_writes`` is ``True``, the agent's
    ``write_page`` / ``extend_page`` tool calls are deferred and surfaced
    to a :class:`outmem.agent.approval.Reviewer` (typically a CLI prompt)
    before the underlying git commit lands. The agent's other tools
    (``append_log``, ``read_*``, ``search_*``) are unaffected.

    Mirrors the YAML block::

        approval:
          required_for_writes: true     # default false
    """

    required_for_writes: bool = DEFAULT_APPROVAL_REQUIRED_FOR_WRITES


@dataclass
class SemanticSettings:
    """Knobs for the vector index (``outmem[semantic]``).

    There is no ``enabled`` flag: semantic is on for a wiki once its index
    exists (build it with ``outmem reindex``). These settings configure
    *how* the index is built and queried when a ``semantic``/``hyde``/
    ``*+semantic`` retrieval strategy or ``find_similar`` needs it.

    ``index`` scopes *what gets indexed*: ``pages+sources`` (default,
    everything) or ``pages`` (curated wiki pages only). Raw sources are
    near-duplicates of the pages distilled from them, and the vector
    search takes a fixed-``k`` KNN before anything can filter by kind, so
    on a source-heavy wiki they crowd curated pages out of the candidate
    window. Narrowing to ``pages`` also prunes already-indexed source
    chunks on the next ``outmem reindex``.

    ``embed_frontmatter`` prepends ``"<title> — <tags>"`` to every chunk
    before embedding. Off by default: turning it on changes what is
    embedded, so the next reindex re-embeds every page (a real API cost,
    but it is the only way for titles and tags to affect retrieval at all —
    ``parse_wiki_page`` otherwise strips them before the chunker sees them).

    Mirrors the YAML block::

        semantic:
          embedding_model: openai:text-embedding-3-small
          db_filename: .vectors.db          # relative to wiki root
          index: pages+sources              # or: pages
          embed_frontmatter: false          # prepend "<title> — <tags>"
          chunk_size: 2000
          chunk_max: 8000
          overlap_paragraphs: 1
          similarity_threshold: 0.80
          top_k: 5
    """

    embedding_model: str = DEFAULT_SEMANTIC_MODEL
    db_filename: str = DEFAULT_SEMANTIC_DB_FILENAME
    index: str = DEFAULT_SEMANTIC_INDEX
    embed_frontmatter: bool = DEFAULT_SEMANTIC_EMBED_FRONTMATTER
    chunk_size: int = DEFAULT_SEMANTIC_CHUNK_SIZE
    chunk_max: int = DEFAULT_SEMANTIC_CHUNK_MAX
    overlap_paragraphs: int = DEFAULT_SEMANTIC_OVERLAP_PARAGRAPHS
    similarity_threshold: float = DEFAULT_SEMANTIC_SIMILARITY_THRESHOLD
    top_k: int = DEFAULT_SEMANTIC_TOP_K


@dataclass
class LogfireSettings:
    """Optional Pydantic Logfire instrumentation.

    Off by default; ``enabled: true`` opts in. When on, outmem configures
    Logfire once (``service_name="outmem"``) and instruments pydantic_ai,
    so every model call — the agent runtime, the rerank gate, and
    the optimize tool — is traced. The destination project is determined
    entirely by ``$LOGFIRE_TOKEN`` (Logfire's API has no project-name
    kwarg), so there is nothing else to configure here. Requires
    ``pip install 'outmem[logfire]'``.

    Mirrors the YAML block::

        logfire:
          enabled: true     # + a LOGFIRE_TOKEN in the environment to send
    """

    enabled: bool = DEFAULT_LOGFIRE_ENABLED


@dataclass
class RetrievalSettings:
    """How the agent's wiki search picks which pages to read.

    Lives in the ``retrieval:`` block of ``config.yaml`` like every other
    setting. ``OptimizeResult.save(...)`` writes it there too (replacing
    just that block); there is no separate file.

    ``strategy`` names the pipeline via a small controlled vocabulary
    (see :mod:`outmem.optimize.dsl`):

    * ``lexical`` / ``bm25`` / ``semantic`` / ``hyde`` — atomic
    * ``rerank`` or ``rerank(<source>)`` — LLM yes/no gate over a source
    * ``a+b[+c…]`` (≥2 atomic legs) — RRF-fused hybrid

    A bad ``strategy`` warns and falls back (config.yaml's forgiving-load
    contract), it doesn't crash the open. Default is ``rerank(bm25)`` (a
    BM25 keyword shortlist gated by one cheap model call per query); set
    ``bm25`` for free, model-free keyword ranking.

    ``from_optimization`` is purely a marker: ``true`` when the block was
    written by ``OptimizeResult.save(...)``, ``false`` for a hand-edit. It
    surfaces in ``git diff`` so a teammate can see whether the current
    pipeline came from an empirical run or a guess.

    For ``rerank``, the source can be written either inline in the
    strategy (``strategy: rerank(semantic)``) or as a sibling field
    (``strategy: rerank`` + ``rerank_source: semantic``) — both resolve to
    the same pipeline.

    Mirrors the ``config.yaml`` block::

        retrieval:
          strategy: bm25
          from_optimization: false
          semantic_top_k: 8
          rrf_k: 60
    """

    strategy: str = DEFAULT_RETRIEVAL_STRATEGY
    from_optimization: bool = False
    semantic_top_k: int = DEFAULT_OPTIMIZE_SEMANTIC_TOP_K
    rrf_k: int = DEFAULT_OPTIMIZE_RRF_K
    max_candidates: int = DEFAULT_OPTIMIZE_MAX_CANDIDATES
    max_relevant: int = DEFAULT_OPTIMIZE_MAX_RELEVANT
    rerank_model: str = DEFAULT_RELEVANCE_MODEL
    hyde_model: str = DEFAULT_RELEVANCE_MODEL
    case_insensitive: bool = True


@dataclass
class GitSettings:
    """Resilience knobs for git subprocess operations."""

    remove_stale_lock: bool = DEFAULT_REMOVE_STALE_LOCK
    stale_lock_seconds: int = DEFAULT_STALE_LOCK_SECONDS
    retry_on_lock: bool = DEFAULT_RETRY_ON_LOCK
    # Auto-ensure the pre-commit hook on store init/open (idempotent, never
    # clobbers a hook you wrote). Set false to manage the hook yourself;
    # note `outmem hook uninstall` alone won't stick while this is true.
    auto_install_hook: bool = DEFAULT_AUTO_INSTALL_HOOK


@dataclass
class AgentSettings:
    """Identity outmem commits under."""

    name: str = DEFAULT_AGENT_NAME
    email: str = DEFAULT_AGENT_EMAIL


@dataclass
class RemoteSettings:
    """Default remote / branch for ``git pull`` and ``git push``."""

    name: str = DEFAULT_REMOTE
    branch: str = DEFAULT_BRANCH


@dataclass
class OutmemConfig:
    """Resolved configuration for a wiki.

    See module docstring for resolution semantics. Unknown keys from
    ``config.yaml`` land in :attr:`extra`.
    """

    model: str = DEFAULT_MODEL
    agent: AgentSettings = field(default_factory=AgentSettings)
    remote: RemoteSettings = field(default_factory=RemoteSettings)
    git: GitSettings = field(default_factory=GitSettings)
    sources: SourceSettings = field(default_factory=SourceSettings)
    semantic: SemanticSettings = field(default_factory=SemanticSettings)
    approval: ApprovalSettings = field(default_factory=ApprovalSettings)
    logfire: LogfireSettings = field(default_factory=LogfireSettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    extra: dict[str, Any] = field(default_factory=dict)


def _outmem_repo_root() -> Path | None:
    """Locate the outmem package's repo root, if any.

    Walks up from the installed ``outmem`` package directory looking
    for a ``pyproject.toml`` — the canonical project-root marker.
    Works cleanly for editable installs (the user's clone) and for
    pip-installed-into-a-project setups (finds the host project's
    pyproject). Returns ``None`` for PyPI-into-site-packages installs
    where there's no repo to find.
    """
    import outmem

    pkg_dir = Path(outmem.__file__).resolve().parent
    for candidate in (pkg_dir, *pkg_dir.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
        if candidate.parent == candidate:  # filesystem root
            break
    return None


def _outmem_repo_dotenv() -> Path | None:
    """``.env`` at the outmem repo root, if it exists."""
    root = _outmem_repo_root()
    if root is None:
        return None
    env = root / ".env"
    return env if env.is_file() else None


def _outmem_repo_defaults() -> OutmemConfig:
    """Read the outmem repo-level ``config.yaml`` (if any) as a
    per-user defaults source for ``outmem init``.

    Returns an all-built-in :class:`OutmemConfig` when no such file
    exists, so callers can always treat the result as the source of
    truth for "what does a fresh wiki get when scaffolded".
    """
    root = _outmem_repo_root()
    if root is None:
        return OutmemConfig()
    return load_yaml_config(root)


def load_dotenv_if_present(path: Path | None = None) -> bool:
    """Load ``.env`` into ``os.environ``.

    Resolution order (existing env vars are never overridden):

    1. Explicit ``path`` if supplied (loads that exact file or no-ops).
    2. CWD-upward search via :func:`find_dotenv` — finds a ``.env``
       sitting next to wherever you invoked ``outmem`` from. Lets
       per-project secrets take precedence over the global fallback.
    3. ``.env`` next to outmem's own ``pyproject.toml`` (the cloned
       repo root, for editable installs). Lets users keep one
       ``.env`` co-located with their outmem source tree and have it
       found regardless of CWD.

    Returns ``True`` if any file was loaded.
    """
    loaded = False

    if path is not None:
        if path.exists():
            load_dotenv(path, override=False)
            return True
        return False

    found = find_dotenv(usecwd=True)
    if found:
        load_dotenv(found, override=False)
        loaded = True

    repo_env = _outmem_repo_dotenv()
    if repo_env is not None:
        load_dotenv(repo_env, override=False)
        loaded = True

    return loaded


def load_yaml_config(wiki_root: Path) -> OutmemConfig:
    """Parse ``<wiki_root>/config.yaml`` into an :class:`OutmemConfig`.

    Returns the all-defaults config when ``config.yaml`` is missing or
    malformed. Logs a warning on malformed YAML so the user knows the
    file was ignored. The ``retrieval:`` block (see
    :class:`RetrievalSettings`) — what the agent's wiki search runs, and
    what ``OptimizeResult.save`` writes — lives in ``config.yaml`` like
    every other setting.
    """
    raw = _read_yaml_mapping(wiki_root / CONFIG_FILENAME)
    return _config_from_dict(raw) if raw is not None else OutmemConfig()


def _read_yaml_mapping(path: Path) -> dict[str, Any] | None:
    """Read a YAML file expected to be a top-level mapping.

    Returns ``None`` on missing / empty / malformed / non-mapping content
    (each case logged), so the caller falls back to defaults instead of
    raising — the forgiving-load contract for ``config.yaml``.
    """
    if not path.exists():
        return None
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        log.warning("Malformed %s, ignoring: %s", path, exc)
        return None
    if raw is None:
        return None
    if not isinstance(raw, dict):
        log.warning(
            "%s must be a YAML mapping, got %s; ignoring",
            path, type(raw).__name__,
        )
        return None
    return raw


def _config_from_dict(data: dict[str, Any]) -> OutmemConfig:
    """Build an :class:`OutmemConfig` from a raw dict.

    Unknown keys land in ``extra`` rather than raising — keeps the
    config schema forward-compatible.
    """
    known = {
        "model",
        "agent",
        "remote",
        "git",
        "sources",
        "semantic",
        "approval",
        "logfire",
        "retrieval",
    }
    extra = {k: v for k, v in data.items() if k not in known}

    config = OutmemConfig(extra=extra)

    if "model" in data and isinstance(data["model"], str):
        config.model = data["model"]

    agent_block = data.get("agent")
    if isinstance(agent_block, dict):
        if isinstance(agent_block.get("name"), str):
            config.agent.name = agent_block["name"]
        if isinstance(agent_block.get("email"), str):
            config.agent.email = agent_block["email"]

    remote_block = data.get("remote")
    if isinstance(remote_block, dict):
        if isinstance(remote_block.get("name"), str):
            config.remote.name = remote_block["name"]
        if isinstance(remote_block.get("branch"), str):
            config.remote.branch = remote_block["branch"]

    git_block = data.get("git")
    if isinstance(git_block, dict):
        if isinstance(git_block.get("remove_stale_lock"), bool):
            config.git.remove_stale_lock = git_block["remove_stale_lock"]
        if isinstance(git_block.get("stale_lock_seconds"), int):
            config.git.stale_lock_seconds = git_block["stale_lock_seconds"]
        if isinstance(git_block.get("retry_on_lock"), bool):
            config.git.retry_on_lock = git_block["retry_on_lock"]
        if isinstance(git_block.get("auto_install_hook"), bool):
            config.git.auto_install_hook = git_block["auto_install_hook"]

    sources_block = data.get("sources")
    if isinstance(sources_block, dict) and isinstance(sources_block.get("max_chars"), int):
        config.sources.max_chars = sources_block["max_chars"]

    semantic_block = data.get("semantic")
    if isinstance(semantic_block, dict):
        if isinstance(semantic_block.get("embedding_model"), str):
            config.semantic.embedding_model = semantic_block["embedding_model"]
        if isinstance(semantic_block.get("db_filename"), str):
            config.semantic.db_filename = semantic_block["db_filename"]
        index_scope = semantic_block.get("index")
        if isinstance(index_scope, str):
            candidate = index_scope.strip().lower()
            if candidate in SEMANTIC_INDEX_CHOICES:
                config.semantic.index = candidate
            else:
                # Forgiving like the rest of config.yaml: warn and keep the
                # default rather than bricking the wiki on a typo.
                log.warning(
                    "config: semantic.index %r is not one of %s — using %r",
                    index_scope,
                    list(SEMANTIC_INDEX_CHOICES),
                    config.semantic.index,
                )
        if "embed_frontmatter" in semantic_block:
            flag = semantic_block["embed_frontmatter"]
            if isinstance(flag, bool):
                config.semantic.embed_frontmatter = flag
            else:
                # Warn rather than ignore: a quoted `"true"` looks correct,
                # and silently keeping the default would leave the user
                # concluding the feature is broken.
                log.warning(
                    "config: semantic.embed_frontmatter must be true/false, "
                    "got %r — using %r",
                    flag,
                    config.semantic.embed_frontmatter,
                )
        if isinstance(semantic_block.get("chunk_size"), int):
            config.semantic.chunk_size = semantic_block["chunk_size"]
        if isinstance(semantic_block.get("chunk_max"), int):
            config.semantic.chunk_max = semantic_block["chunk_max"]
        if isinstance(semantic_block.get("overlap_paragraphs"), int):
            config.semantic.overlap_paragraphs = semantic_block["overlap_paragraphs"]
        # similarity_threshold accepts int (1) or float (0.8)
        threshold = semantic_block.get("similarity_threshold")
        if isinstance(threshold, (int, float)) and not isinstance(threshold, bool):
            config.semantic.similarity_threshold = float(threshold)
        if isinstance(semantic_block.get("top_k"), int):
            config.semantic.top_k = semantic_block["top_k"]

    approval_block = data.get("approval")
    if isinstance(approval_block, dict) and isinstance(
        approval_block.get("required_for_writes"), bool
    ):
        config.approval.required_for_writes = approval_block["required_for_writes"]

    logfire_block = data.get("logfire")
    if isinstance(logfire_block, dict) and isinstance(
        logfire_block.get("enabled"), bool
    ):
        config.logfire.enabled = logfire_block["enabled"]

    retrieval_block = data.get("retrieval")
    if isinstance(retrieval_block, dict):
        _apply_retrieval_block(config.retrieval, retrieval_block)

    return config


def _is_int(value: Any) -> bool:
    """True for a real int, excluding bool.

    ``bool`` is a subclass of ``int`` in Python, so a YAML ``true`` (or
    ``yes``/``on`` under YAML 1.1) would otherwise satisfy an
    ``isinstance(x, int)`` guard and silently set a numeric knob to 1/0.
    Mirrors the ``not isinstance(..., bool)`` defence already used for
    ``semantic.similarity_threshold``."""
    return isinstance(value, int) and not isinstance(value, bool)


def _apply_retrieval_block(target: RetrievalSettings, block: dict[str, Any]) -> None:
    """Mutate ``target`` with values from ``config.yaml``'s ``retrieval:`` block.

    Forgiving, like the rest of ``config.yaml``: a bad ``strategy`` DSL
    string logs a warning and leaves ``strategy`` untouched rather than
    crashing every ``WikiStore.open`` on one typo, and a bad-typed knob is
    ignored so one fat-fingered field doesn't take down the load.

    A bare ``rerank_source:`` key is folded into ``strategy`` (``strategy:
    rerank`` + ``rerank_source: semantic`` ⇒ ``rerank(semantic)``), so the
    field a user naturally reaches for from the optimizer's ``run_eval``
    surface actually takes effect instead of being silently dropped. The
    stored ``strategy`` is canonicalised through the DSL round-trip, so
    ``bm25 + semantic`` lands as ``bm25+semantic``."""
    # Lazy import avoids a config <-> optimize import cycle at load time.
    from outmem.exceptions import OutmemError
    from outmem.optimize.dsl import format_strategy, parse_strategy

    if "strategy" in block:
        spec = str(block["strategy"]).strip().lower()
        source = block.get("rerank_source")
        if spec == "rerank" and isinstance(source, str):
            spec = f"rerank({source.strip().lower()})"
        try:
            # Round-trip through the DSL: validates, and stores the
            # canonical form a save() would write.
            target.strategy = format_strategy(parse_strategy(spec))
        except OutmemError:
            log.warning(
                "Ignoring invalid retrieval.strategy %r in config.yaml; "
                "keeping %r",
                block["strategy"], target.strategy,
            )
    if isinstance(block.get("from_optimization"), bool):
        target.from_optimization = block["from_optimization"]
    if _is_int(block.get("semantic_top_k")):
        target.semantic_top_k = block["semantic_top_k"]
    if _is_int(block.get("rrf_k")):
        target.rrf_k = block["rrf_k"]
    if _is_int(block.get("max_candidates")):
        target.max_candidates = block["max_candidates"]
    if _is_int(block.get("max_relevant")):
        target.max_relevant = block["max_relevant"]
    if isinstance(block.get("rerank_model"), str):
        target.rerank_model = block["rerank_model"]
    if isinstance(block.get("hyde_model"), str):
        target.hyde_model = block["hyde_model"]
    if isinstance(block.get("case_insensitive"), bool):
        target.case_insensitive = block["case_insensitive"]


def starter_yaml(
    *,
    agent_name: str = DEFAULT_AGENT_NAME,
    agent_email: str = DEFAULT_AGENT_EMAIL,
    model: str | None = None,
) -> str:
    """Render the contents of an initial ``config.yaml`` file.

    Written by :meth:`WikiStore.init` so a fresh wiki has a visible
    config the user can tune.

    ``model`` defaults to the outmem repo-level ``config.yaml``'s
    ``model:`` field (if the user has created one at the cloned repo
    root), falling back to :data:`DEFAULT_MODEL`. That lets a user
    set a per-install default model once at
    ``<outmem-clone>/config.yaml`` and have every ``outmem init``
    pick it up.
    """
    if model is None:
        model = _outmem_repo_defaults().model
    return (
        "# config.yaml — wiki-level config for the agent runtime\n"
        "# Tracked in git; secrets live in .env (gitignored).\n"
        "#\n"
        "# Resolution: CLI args > env vars > this file > built-in defaults.\n"
        "\n"
        f"model: {model}\n"
        "\n"
        "agent:\n"
        f"  name: {agent_name}\n"
        f"  email: {agent_email}\n"
        "\n"
        "remote:\n"
        f"  name: {DEFAULT_REMOTE}\n"
        f"  branch: {DEFAULT_BRANCH}\n"
        "\n"
        "git:\n"
        f"  remove_stale_lock: {str(DEFAULT_REMOVE_STALE_LOCK).lower()}\n"
        f"  stale_lock_seconds: {DEFAULT_STALE_LOCK_SECONDS}\n"
        f"  retry_on_lock: {str(DEFAULT_RETRY_ON_LOCK).lower()}\n"
        "\n"
        "sources:\n"
        f"  max_chars: {DEFAULT_SOURCE_MAX_CHARS}    # cap on read_source returns\n"
        "\n"
        "# Semantic vector index (requires `pip install outmem[semantic]`).\n"
        "# No on/off flag: it's active once built — run `outmem reindex` to\n"
        "# index pages + sources into a local sqlite-vec DB. A semantic /\n"
        "# hyde / *+semantic retrieval strategy and `find_similar` need it;\n"
        "# the keys below configure how it's built and queried.\n"
        "semantic:\n"
        f"  embedding_model: {DEFAULT_SEMANTIC_MODEL}\n"
        f"  db_filename: {DEFAULT_SEMANTIC_DB_FILENAME}\n"
        "  # What to index: `pages+sources` (default) or `pages` to keep raw\n"
        "  # ingested sources out of the vector store (they crowd curated\n"
        "  # pages out of the candidate window on source-heavy wikis).\n"
        f"  index: {DEFAULT_SEMANTIC_INDEX}\n"
        "  # Prepend \"<title> \u2014 <tags>\" to every chunk before embedding, so\n"
        "  # titles/tags affect retrieval. Turning this on re-embeds every\n"
        "  # page on the next `outmem reindex`.\n"
        f"  embed_frontmatter: {str(DEFAULT_SEMANTIC_EMBED_FRONTMATTER).lower()}\n"
        f"  chunk_size: {DEFAULT_SEMANTIC_CHUNK_SIZE}\n"
        f"  chunk_max: {DEFAULT_SEMANTIC_CHUNK_MAX}\n"
        f"  overlap_paragraphs: {DEFAULT_SEMANTIC_OVERLAP_PARAGRAPHS}\n"
        f"  similarity_threshold: {DEFAULT_SEMANTIC_SIMILARITY_THRESHOLD}\n"
        f"  top_k: {DEFAULT_SEMANTIC_TOP_K}\n"
        "\n"
        "# Human-in-the-loop approval for agent writes. When on, every\n"
        "# `write_page` / `extend_page` is shown to a reviewer (CLI\n"
        "# prompt by default) and only commits after explicit approval.\n"
        "# `append_log` and read tools are not gated.\n"
        "approval:\n"
        f"  required_for_writes: {str(DEFAULT_APPROVAL_REQUIRED_FOR_WRITES).lower()}\n"
        "\n"
        "# Optional: send spans + LLM traces to Pydantic Logfire.\n"
        "# Requires `pip install outmem[logfire]` and $LOGFIRE_TOKEN\n"
        "# (the token determines which project the data lands in).\n"
        "# Spans are labeled service_name=outmem so they're easy to\n"
        "# filter when other tools publish to the same project.\n"
        "logfire:\n"
        "  enabled: false     # true + a $LOGFIRE_TOKEN in the env sends traces;\n"
        "                     # the token (not this file) picks the project.\n"
        "\n"
        "# Which pipeline the agent's `search_wiki` search runs. `strategy` is\n"
        "# a small DSL: lexical | bm25 | semantic | hyde | rerank(<source>)\n"
        "# | a+b[+c...] (RRF hybrid, e.g. bm25+semantic). Default is\n"
        "# rerank(bm25): BM25 keyword shortlist gated by one Haiku call per\n"
        "# query (~1.5 s/query, lifts recall on paraphrases). For free/fast\n"
        "# plain keyword ranking with no model call, set strategy: bm25.\n"
        "# semantic/hyde/<...>+semantic need a built index (`outmem\n"
        "# reindex`). Tune empirically with `outmem.optimize.optimize_retrieval`,\n"
        "# then `result.save(rank, store)` rewrites this block. See\n"
        "# docs/configuration.md.\n"
        "retrieval:\n"
        f"  strategy: {DEFAULT_RETRIEVAL_STRATEGY}\n"
        "  from_optimization: false   # true once written by an optimize run\n"
    )


def starter_agents_md() -> str:
    """Render the starter body of ``wiki/AGENTS.md``.

    Loaded into the agent's system prompt every turn (see
    :func:`outmem.agent.render_system_prompt`). The starter is sparse
    placeholders — the wiki owner populates it as they discover what
    rules their wiki needs.
    """
    return """# AGENTS.md — wiki conventions

This file is loaded into the agent's system prompt on every run. It is
the place to record domain-specific conventions and preferences that
the runtime defaults don't already cover. Keep it short — the agent
re-reads it every turn and a bloated AGENTS.md is paid for in tokens.

You and the agent co-evolve this file over time. When you notice the
agent making the same mistake twice, write the rule here so it stops
making it a third time.

## What this wiki is for

<!-- Describe the wiki's domain in one or two sentences. Examples:
- "A personal medical-knowledge wiki — drug dosing, interactions, side
  effects, drawn from product Fachinformationen."
- "Research notes for my book on X. The audience is me, six months from
  now, trying to remember what I knew."
- "Team knowledge: meeting decisions, project status, customer-call
  takeaways." -->

## Page conventions

<!-- Optional. Page-structure templates the agent should follow.
Examples:
- "Drug pages have sections: Indication, Dosing, Side effects,
  Interactions, Provenance."
- "Comparison pages always end with a one-line takeaway."
- "Tag every page with the project name in `tags`." -->

## What goes where

<!-- Optional. Guide the agent's choice between write_page, extend_page,
and append_log. Examples:
- "Single-source observations go to the log; compact into a page only
  after a second source confirms."
- "Always create a new page for any drug mentioned in a
  Fachinformation, even if only briefly." -->

## Anything else the agent should know

<!-- Free-form. Terminology preferences, source-handling quirks, things
that have bitten you twice. -->
"""


