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

DEFAULT_SOURCE_MAX_CHARS = 200_000  # cap on `read_source` tool returns

DEFAULT_SEMANTIC_ENABLED = False
DEFAULT_SEMANTIC_MODEL = "openai:text-embedding-3-small"
DEFAULT_SEMANTIC_DB_FILENAME = ".vectors.db"
DEFAULT_SEMANTIC_CHUNK_SIZE = 2000
DEFAULT_SEMANTIC_CHUNK_MAX = 8000
DEFAULT_SEMANTIC_OVERLAP_PARAGRAPHS = 1
DEFAULT_SEMANTIC_SIMILARITY_THRESHOLD = 0.80
DEFAULT_SEMANTIC_TOP_K = 5

DEFAULT_APPROVAL_REQUIRED_FOR_WRITES = False

DEFAULT_RELEVANCE_ENABLED = False
DEFAULT_RELEVANCE_MODEL = "anthropic:claude-haiku-4-5"
DEFAULT_RELEVANCE_MAX_RELEVANT = 8
DEFAULT_RELEVANCE_MAX_CANDIDATES = 20
DEFAULT_RELEVANCE_CONTEXT = "page"
DEFAULT_RELEVANCE_CONTEXT_CHARS = 2000
DEFAULT_RELEVANCE_CANDIDATE_MAX_BYTES = 64 * 1024

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
DEFAULT_OPTIMIZE_K = 5                      # Hit@k cutoff
DEFAULT_OPTIMIZE_MAX_EVALS = 12             # optimizer turn budget
DEFAULT_OPTIMIZE_MAX_FAILURES_SHOWN = 6     # failing questions shown per eval
DEFAULT_OPTIMIZE_UNANSWERABLE_LIMIT = 20    # gap-log questions harvested

DEFAULT_LOGFIRE_PROJECT: str | None = None
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

# Error string when a caller hits a semantic-only path but the wiki has
# the feature off — shared across the CLI, the WikiStore facet, and the
# PydanticAI adapter so the user sees the same fix-it advice everywhere.
SEMANTIC_DISABLED_HELP = (
    "semantic indexing is disabled — set `semantic.enabled: true` "
    "in config.yaml and `pip install outmem[semantic]`."
)

RELEVANCE_DISABLED_HELP = (
    "relevance filtering is disabled — set `relevance.enabled: true` "
    "in config.yaml (and `pip install outmem[agent]` for the triage model)."
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
    """Knobs for the optional vector index (``outmem[semantic]``).

    Mirrors the YAML block::

        semantic:
          enabled: true
          embedding_model: openai:text-embedding-3-small
          db_filename: .vectors.db          # relative to wiki root
          chunk_size: 2000
          chunk_max: 8000
          overlap_paragraphs: 1
          similarity_threshold: 0.80
          top_k: 5
    """

    enabled: bool = DEFAULT_SEMANTIC_ENABLED
    embedding_model: str = DEFAULT_SEMANTIC_MODEL
    db_filename: str = DEFAULT_SEMANTIC_DB_FILENAME
    chunk_size: int = DEFAULT_SEMANTIC_CHUNK_SIZE
    chunk_max: int = DEFAULT_SEMANTIC_CHUNK_MAX
    overlap_paragraphs: int = DEFAULT_SEMANTIC_OVERLAP_PARAGRAPHS
    similarity_threshold: float = DEFAULT_SEMANTIC_SIMILARITY_THRESHOLD
    top_k: int = DEFAULT_SEMANTIC_TOP_K


@dataclass
class RelevanceSettings:
    """Knobs for the optional relevance filter (``outmem[agent]``).

    A cheap-model gate between lexical ``search_wiki`` and the expensive
    outer agent: it keeps only the candidate pages relevant to the
    query (yes/no, no score) and lets the candidate net be wider than
    what the outer agent should see. Off by default — opt-in, like
    ``semantic``.

    This is the config-driven default; a downstream consumer that needs
    the ``on_filter`` observability hook constructs an explicit
    :class:`outmem.relevance.RelevanceConfig` and passes it to the
    adapter factories instead. When the adapter is given no explicit
    config, it falls back to these settings (disabled ⇒ today's
    ``search_wiki`` byte-for-byte).

    Mirrors the YAML block::

        relevance:
          enabled: true
          model: anthropic:claude-haiku-4-5
          max_relevant: 8
          max_candidates: 20
          candidate_max_bytes: 65536      # width of the wide ripgrep net
          context: page                 # "page" | "lines"
          context_chars_per_page: 2000
    """

    enabled: bool = DEFAULT_RELEVANCE_ENABLED
    model: str = DEFAULT_RELEVANCE_MODEL
    max_relevant: int = DEFAULT_RELEVANCE_MAX_RELEVANT
    max_candidates: int = DEFAULT_RELEVANCE_MAX_CANDIDATES
    candidate_max_bytes: int = DEFAULT_RELEVANCE_CANDIDATE_MAX_BYTES
    context: str = DEFAULT_RELEVANCE_CONTEXT
    context_chars_per_page: int = DEFAULT_RELEVANCE_CONTEXT_CHARS


@dataclass
class LogfireSettings:
    """Optional Pydantic Logfire instrumentation.

    Off by default. Any non-null ``project`` value opts in: the CLI
    configures Logfire once per invocation with ``service_name="outmem"``
    (so spans are distinguishable from other services publishing to the
    same project) and instruments pydantic_ai. The actual project is
    determined by ``$LOGFIRE_TOKEN`` (Logfire's API doesn't accept a
    project-name kwarg); the config field is therefore an opt-in marker
    and self-documentation of which project the user expects to feed.
    Requires ``pip install 'outmem[logfire]'``.

    Mirrors the YAML block::

        logfire:
          project: my-project    # null/absent = disabled
    """

    project: str | None = DEFAULT_LOGFIRE_PROJECT


@dataclass
class GitSettings:
    """Resilience knobs for git subprocess operations."""

    remove_stale_lock: bool = DEFAULT_REMOVE_STALE_LOCK
    stale_lock_seconds: int = DEFAULT_STALE_LOCK_SECONDS
    retry_on_lock: bool = DEFAULT_RETRY_ON_LOCK


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
    relevance: RelevanceSettings = field(default_factory=RelevanceSettings)
    approval: ApprovalSettings = field(default_factory=ApprovalSettings)
    logfire: LogfireSettings = field(default_factory=LogfireSettings)
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

    Returns the all-defaults config when the file is missing or
    malformed. Logs a warning on malformed YAML so the user knows
    the file was ignored.
    """
    path = wiki_root / CONFIG_FILENAME
    if not path.exists():
        return OutmemConfig()

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        log.warning("Malformed %s, ignoring: %s", path, exc)
        return OutmemConfig()

    if raw is None:
        return OutmemConfig()
    if not isinstance(raw, dict):
        log.warning("%s must be a YAML mapping, got %s; ignoring", path, type(raw).__name__)
        return OutmemConfig()

    return _config_from_dict(raw)


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
        "relevance",
        "approval",
        "logfire",
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

    sources_block = data.get("sources")
    if isinstance(sources_block, dict) and isinstance(sources_block.get("max_chars"), int):
        config.sources.max_chars = sources_block["max_chars"]

    semantic_block = data.get("semantic")
    if isinstance(semantic_block, dict):
        if isinstance(semantic_block.get("enabled"), bool):
            config.semantic.enabled = semantic_block["enabled"]
        if isinstance(semantic_block.get("embedding_model"), str):
            config.semantic.embedding_model = semantic_block["embedding_model"]
        if isinstance(semantic_block.get("db_filename"), str):
            config.semantic.db_filename = semantic_block["db_filename"]
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

    relevance_block = data.get("relevance")
    if isinstance(relevance_block, dict):
        if isinstance(relevance_block.get("enabled"), bool):
            config.relevance.enabled = relevance_block["enabled"]
        if isinstance(relevance_block.get("model"), str):
            config.relevance.model = relevance_block["model"]
        if isinstance(relevance_block.get("max_relevant"), int):
            config.relevance.max_relevant = relevance_block["max_relevant"]
        if isinstance(relevance_block.get("max_candidates"), int):
            config.relevance.max_candidates = relevance_block["max_candidates"]
        if isinstance(relevance_block.get("candidate_max_bytes"), int):
            config.relevance.candidate_max_bytes = relevance_block["candidate_max_bytes"]
        if relevance_block.get("context") in ("page", "lines"):
            config.relevance.context = relevance_block["context"]
        if isinstance(relevance_block.get("context_chars_per_page"), int):
            config.relevance.context_chars_per_page = relevance_block[
                "context_chars_per_page"
            ]

    approval_block = data.get("approval")
    if isinstance(approval_block, dict) and isinstance(
        approval_block.get("required_for_writes"), bool
    ):
        config.approval.required_for_writes = approval_block["required_for_writes"]

    logfire_block = data.get("logfire")
    if isinstance(logfire_block, dict):
        project = logfire_block.get("project")
        if project is None or isinstance(project, str):
            config.logfire.project = project

    return config


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
        "# Semantic retrieval / lint — requires `pip install outmem[semantic]`.\n"
        "# Off by default; flip `enabled: true` to index pages and sources\n"
        "# into a local sqlite-vec DB and surface near-duplicate / similar chunks.\n"
        "semantic:\n"
        f"  enabled: {str(DEFAULT_SEMANTIC_ENABLED).lower()}\n"
        f"  embedding_model: {DEFAULT_SEMANTIC_MODEL}\n"
        f"  db_filename: {DEFAULT_SEMANTIC_DB_FILENAME}\n"
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
        "  project: null      # any non-null string opts in; use your\n"
        "                     # project name for self-documentation.\n"
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


