"""Optional Pydantic Logfire wiring.

:func:`setup` is a CLI-side hook — call it once per invocation with the
config's :class:`LogfireSettings`. When disabled (``enabled`` false),
this is a no-op and the ``logfire`` dep doesn't have to be installed.
When enabled, we configure Logfire with
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

import logging
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from outmem.config import LOGFIRE_SERVICE_NAME, LogfireSettings
from outmem.exceptions import OutmemError

log = logging.getLogger(__name__)

_configured = False


def setup(settings: LogfireSettings) -> bool:
    """Configure Logfire if ``settings.enabled``. Returns ``True`` when
    instrumentation was activated, ``False`` when disabled."""
    global _configured

    if not settings.enabled:
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
        # Don't echo every span to the local terminal — spans still go to
        # the Logfire UI. outmem prints its own progress; the console
        # exporter just floods stdout (dozens of "agent run" lines).
        console=False,
    )
    logfire.instrument_pydantic_ai()
    _quiet_export_noise()
    _configured = True
    return True


def _quiet_export_noise() -> None:
    """Keep a misconfigured token from flooding the console.

    A rejected token makes Logfire emit a ``UserWarning`` per span plus a
    repeated ``Failed to export span batch ... 401`` from its exporter. We
    can't validate the token at configure-time (Logfire checks it lazily on
    the background export thread), so instead we cap the spam: collapse the
    repeated warning to one, and raise Logfire's own logger to ERROR so the
    per-batch export failures don't carpet stdout. Genuine errors still log;
    routine 401/connection retries don't.
    """
    # The token-rejected UserWarning is identical every span — show it once.
    warnings.filterwarnings(
        "once", message=".*Logfire API returned status code.*", category=UserWarning
    )
    # Logfire's exporter logs "Failed to export span batch" per attempt;
    # ERROR-floor it so transient/repeated export failures stay quiet.
    logging.getLogger("logfire").setLevel(logging.ERROR)


@contextmanager
def span(name: str, **attributes: object) -> Iterator[None]:
    """A parent span for grouping nested work, or a no-op.

    Yields a Logfire span when logfire is installed AND already configured
    (i.e. ``setup`` ran with ``logfire.enabled``); otherwise a plain no-op,
    so the core never hard-depends on the optional extra. Used to nest the
    many per-page / per-eval ``agent run`` spans under one parent in the
    trace instead of leaving them flat.
    """
    if not _configured:
        yield
        return
    try:
        import logfire
    except ImportError:
        yield
        return
    # logfire.span's typed signature doesn't accept arbitrary **kwargs as
    # attributes; call via an untyped alias so mypy doesn't narrow them.
    _span: Any = logfire.span
    with _span(name, **attributes):
        yield
