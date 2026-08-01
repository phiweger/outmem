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

# At most this many lines when stderr is not a TTY. A `\r`-updated
# counter costs one line on a terminal however many ticks it has, but
# captured output gets one line *per tick* — so reindexing a few thousand
# pages buried everything else in the log, including the install line
# above it. Progress you cannot scroll past is not progress you can read.
_CAPTURED_TICKS = 10


def _is_milestone(done: int, total: int) -> bool:
    """Whether this tick is worth a line when output is being captured."""
    if done <= 1 or done >= total:
        return True  # always show the first and the last
    return done % max(1, total // _CAPTURED_TICKS) == 0


def report_progress(
    on_progress: ProgressFn | None,
    done: int,
    total: int,
    *,
    label: str,
    unit: str = "items",
) -> None:
    """Emit one progress tick. With ``on_progress`` set, call it (guarded);
    otherwise write ``label: done/total unit`` to stderr.

    On a TTY every tick redraws one ``\\r``-updated line. When stderr is
    captured (CI, a redirected log, a subprocess reading the tail) the
    output is throttled to ~:data:`_CAPTURED_TICKS` lines plus the first
    and last, so a long operation stays visible without flooding the log
    it shares with everything else.
    """
    if on_progress is not None:
        try:
            on_progress(done, total)
        except Exception as exc:  # a progress callback must never break the op
            log.warning("on_progress raised (%s); ignoring", exc)
        return
    tty = sys.stderr.isatty()
    if not tty and not _is_milestone(done, total):
        return
    end = "\r" if (tty and done < total) else "\n"
    sys.stderr.write(f"{label}: {done}/{total} {unit}{end}")
    sys.stderr.flush()
