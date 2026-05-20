"""Optional Pydantic Logfire wiring.

:func:`setup` is a CLI-side hook — call it once per invocation with the
config's :class:`LogfireSettings`. When the ``project`` field is
``None`` (default), this is a no-op and the ``logfire`` dep doesn't
have to be installed. When it's set, we configure Logfire with
``service_name="outmem"`` (so traces are tagged distinctly from other
services publishing to the same project) and instrument pydantic_ai.

The auth token comes from ``$LOGFIRE_TOKEN`` (read by
``logfire.configure`` itself), which is also what routes data to a
specific project — Logfire's API doesn't accept a project-name kwarg.
The config's ``project`` field is therefore an opt-in marker plus
self-documentation of which project the user expects to feed; routing
is the token's job.

Missing dep with ``project`` configured raises :class:`OutmemError`
rather than silently degrading — the user asked for instrumentation;
tell them how to install it.
"""

from __future__ import annotations

from outmem.config import LOGFIRE_SERVICE_NAME, LogfireSettings
from outmem.exceptions import OutmemError

_configured = False


def setup(settings: LogfireSettings) -> bool:
    """Configure Logfire if ``settings.project`` is set. Returns ``True``
    when instrumentation was activated, ``False`` when disabled."""
    global _configured

    if settings.project is None:
        return False
    if _configured:
        return True

    try:
        import logfire
    except ImportError as exc:
        raise OutmemError(
            "logfire.project is set in config.yaml but the `logfire` "
            "package is not installed. Run: pip install 'outmem[logfire]'"
        ) from exc

    logfire.configure(
        service_name=LOGFIRE_SERVICE_NAME,
        send_to_logfire="if-token-present",
    )
    logfire.instrument_pydantic_ai()
    _configured = True
    return True
