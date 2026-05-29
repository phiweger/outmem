"""Shared progress reporting — a tiny stderr counter with a callback hook.

Used by long-running, countable operations (question-bank generation,
semantic reindex). The default prints a live ``label: done/total unit``
line to stderr — ``\\r``-updated on a TTY, one line per tick otherwise
(Jupyter, redirected output, logs). pytest captures stderr, so it stays
silent in test runs. Pass an ``on_progress(done, total)`` callback to
route progress elsewhere (a bar, a logger); a raising callback is
swallowed so it can never break the underlying operation.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable

log = logging.getLogger(__name__)

ProgressFn = Callable[[int, int], None]


def report_progress(
    on_progress: ProgressFn | None,
    done: int,
    total: int,
    *,
    label: str,
    unit: str = "items",
) -> None:
    """Emit one progress tick. With ``on_progress`` set, call it (guarded);
    otherwise write ``label: done/total unit`` to stderr."""
    if on_progress is not None:
        try:
            on_progress(done, total)
        except Exception as exc:  # a progress callback must never break the op
            log.warning("on_progress raised (%s); ignoring", exc)
        return
    end = "\r" if (sys.stderr.isatty() and done < total) else "\n"
    sys.stderr.write(f"{label}: {done}/{total} {unit}{end}")
    sys.stderr.flush()
