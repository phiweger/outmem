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
from outmem.store import AgentIdentity, WikiPage, WikiStore, WikiStoreConfig

__version__ = "0.1.0.dev0"

__all__ = [
    "AgentIdentity",
    "ConflictError",
    "FrontmatterError",
    "GitOperationError",
    "IdentityWarning",
    "OutmemError",
    "ProvenanceEntry",
    "SlugError",
    "WikiFrontmatter",
    "WikiPage",
    "WikiStore",
    "WikiStoreConfig",
    "WritebackError",
    "__version__",
    "setup_logfire",
]
