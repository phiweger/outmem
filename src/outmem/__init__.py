"""outmem — agentic RAG memory over a git-versioned wiki.

Public API::

    from outmem import WikiStore, WikiPage, AgentIdentity

    store = WikiStore.open("/srv/agent")
    hits = store.search("pricing formula")
    page = store.read("pricing-formula")
    store.extend_page("pricing-formula", body="…")
    store.append_log(topic="pricing", content="noticed an inconsistency")

See ``docs/`` for the conceptual rationale, the v0.1 spec, and integration
patterns. See ``src/outmem/skills/notes/`` for the skills a downstream agent
loads to learn the search / evolution / write workflows.
"""

from __future__ import annotations

from outmem.exceptions import (
    ConflictError,
    FrontmatterError,
    GitOperationError,
    IdentityWarning,
    OutmemError,
    SlugError,
    WritebackError,
)
from outmem.frontmatter import ProvenanceEntry, WikiFrontmatter
from outmem.observability import setup_logfire
from outmem.relevance import (
    FilterOutcome,
    RelevanceConfig,
    RelevantPage,
    relevance_filter,
)
from outmem.search import SearchHit
from outmem.store import AgentIdentity, WikiPage, WikiStore, WikiStoreConfig

try:
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("outmem")
except Exception:  # not installed (bare source checkout) — avoid hard failure
    __version__ = "0.0.0+unknown"

__all__ = [
    "AgentIdentity",
    "ConflictError",
    "FilterOutcome",
    "FrontmatterError",
    "GitOperationError",
    "IdentityWarning",
    "OutmemError",
    "ProvenanceEntry",
    "RelevanceConfig",
    "RelevantPage",
    "SearchHit",
    "SlugError",
    "WikiFrontmatter",
    "WikiPage",
    "WikiStore",
    "WikiStoreConfig",
    "WritebackError",
    "__version__",
    "relevance_filter",
    "setup_logfire",
]
