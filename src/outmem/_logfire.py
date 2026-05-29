"""Optional Pydantic Logfire wiring.

:func:`setup` is a CLI-side hook — call it once per invocation with the
config's :class:`LogfireSettings`. When disabled (``enabled`` false and
no deprecated ``project``), this is a no-op and the ``logfire`` dep
doesn't have to be installed. When enabled, we configure Logfire with
``service_name="outmem"`` (so traces are tagged distinctly from other
services publishing to the same project) and instrument pydantic_ai.

The auth token comes from ``$LOGFIRE_TOKEN`` (read by
``logfire.configure`` itself), which is what routes data to a specific
project — Logfire's API doesn't accept a project-name kwarg. So
``enabled`` is purely the on-switch; the token decides the destination.

Enabled but the dep missing raises :class:`OutmemError` rather than
silently degrading — the user asked for instrumentation; tell them how
to install it.
"""

from __future__ import annotations

from outmem.config import LOGFIRE_SERVICE_NAME, LogfireSettings
from outmem.exceptions import OutmemError

_configured = False


def setup(settings: LogfireSettings) -> bool:
    """Configure Logfire if ``settings.enabled`` (or the deprecated
    ``project``) is set. Returns ``True`` when instrumentation was
    activated, ``False`` when disabled."""
    global _configured

    if not settings.enabled and settings.project is None:
        return False
    if _configured:
        return True

    try:
        import logfire
    except ImportError as exc:
        raise OutmemError(
            "Logfire is enabled in config.yaml but the `logfire` package "
            "is not installed. Run: pip install 'outmem[logfire]'"
        ) from exc

    logfire.configure(
        service_name=LOGFIRE_SERVICE_NAME,
        send_to_logfire="if-token-present",
    )
    logfire.instrument_pydantic_ai()
    _configured = True
    return True
