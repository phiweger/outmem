"""Public observability helpers — currently just Logfire setup.

Why this exists: the CLI's ``outmem ask`` auto-configures Pydantic
Logfire when ``config.yaml`` sets ``logfire.project: <name>``, but
library entry points (:func:`outmem.agent.ask_sync`,
:func:`outmem.adapters.pydantic_ai.build_consult_wiki`) and custom
integrations (:func:`outmem.adapters.pydantic_ai.wiki_tools` + your
own :class:`pydantic_ai.Agent`) previously didn't.

``ask`` and ``build_consult_wiki`` now call :func:`setup_logfire`
themselves, so wikis with ``logfire.project`` set get instrumentation
from every code path. For custom integrations, call this once at
startup::

    from outmem import WikiStore, setup_logfire

    store = WikiStore.open("/srv/wiki")
    setup_logfire(store)   # respects store.config.outmem.logfire

The actual Logfire wiring (``logfire.configure`` +
``instrument_pydantic_ai``) lives in :mod:`outmem._logfire`; this
module is the public façade.
"""

from __future__ import annotations

from outmem._logfire import setup as _setup
from outmem.config import LogfireSettings
from outmem.store import WikiStore


def setup_logfire(target: WikiStore | LogfireSettings) -> bool:
    """Configure Pydantic Logfire from a wiki's config.

    Idempotent process-wide: the first call activates instrumentation,
    later calls are no-ops (so calling from multiple library entry
    points in the same process is safe).

    Returns ``True`` if instrumentation was activated, ``False`` when
    Logfire is disabled (``logfire.enabled`` false). Raises
    :class:`OutmemError` when it's enabled but the ``logfire`` package
    isn't installed — the user asked for instrumentation, so the failure
    surfaces rather than being silently swallowed.

    Args:
        target: Either a :class:`WikiStore` (the typical case — pulls
            :class:`LogfireSettings` out of
            ``store.config.outmem.logfire``) or a raw
            :class:`LogfireSettings` if you've assembled it yourself.
    """
    if isinstance(target, WikiStore):
        return _setup(target.config.outmem.logfire)
    return _setup(target)
