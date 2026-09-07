"""outmem — agentic RAG memory over a git-versioned wiki.

Public API::

    from outmem import WikiStore, WikiPage, AgentIdentity

    store = WikiStore.open("/srv/agent")
    hits = store.search("pricing formula")
    page = store.read("pricing-formula")
    store.extend_page("pricing-formula", body="Revised: cost-plus 40%.")
    store.append_log(topic="pricing", content="noticed an inconsistency")

Serving a wiki whose content is not uniformly readable::

    from outmem import Grants, RestrictionError

    view = store.as_viewer(mode={"hr"}, grants=Grants.reader("hr"))

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
    IncompleteBodyError,
    OutmemError,
    RestrictionError,
    SlugError,
    WritebackError,
)
from outmem.frontmatter import ProvenanceEntry, WikiFrontmatter
from outmem.observability import setup_logfire
from outmem.relevance import RelevantPage, judge_relevance
from outmem.restricted import Grants, LabelError, RestrictedSettings
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
    "FrontmatterError",
    "GitOperationError",
    "Grants",
    "IdentityWarning",
    "IncompleteBodyError",
    "LabelError",
    "OutmemError",
    "ProvenanceEntry",
    "RelevantPage",
    "RestrictedSettings",
    "RestrictionError",
    "SearchHit",
    "SlugError",
    "WikiFrontmatter",
    "WikiPage",
    "WikiStore",
    "WikiStoreConfig",
    "WritebackError",
    "__version__",
    "judge_relevance",
    "setup_logfire",
]
