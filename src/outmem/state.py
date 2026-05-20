"""Non-git-tracked state for a wiki — the ``.outmem/`` directory.

Some things outmem needs to remember between runs are intentionally
outside git: the last-run marker for the steering loop, the backlinks
cache, the parsed identity map. Persisting them in commits would
generate churn that pollutes ``git log`` and tangles with the steering
signal itself.

The state directory lives at ``<wiki_root>/.outmem/`` and is excluded
from git via a ``.gitignore`` written by :meth:`OutmemState.ensure`.
Every file inside is small JSON. Writes:

- Hold an exclusive ``fcntl.flock`` on a per-file lockfile so concurrent
  processes (CLI + dashboard + agent) serialise their updates.
- Stage to ``<name>.tmp``, ``fsync`` the bytes, then ``os.replace`` —
  the rename is atomic and the data is on disk before the swap.

On non-POSIX platforms (where :mod:`fcntl` is unavailable) the lock
step is skipped with a warning; reads still work correctly but the
multi-writer race surfaces. Outmem v0.1 targets Linux servers.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from outmem._time import ensure_utc, format_iso_z, parse_iso_z, utc_now

try:
    import fcntl as _fcntl

    _HAS_FLOCK = True
except ImportError:  # pragma: no cover — non-POSIX fallback
    _fcntl = None  # type: ignore[assignment]
    _HAS_FLOCK = False

_log = logging.getLogger(__name__)

STATE_DIR_NAME = ".outmem"
LAST_RUN_FILE = "last_run.json"
BACKLINKS_FILE = "backlinks.json"

_GITIGNORE_BODY = "# Created by outmem — non-git-tracked state lives here.\n*\n!.gitignore\n"


@dataclass(frozen=True)
class LastRun:
    """The marker recorded after each successful agent run.

    ``head`` is the HEAD SHA the agent observed when its writeback
    completed; the next run uses ``timestamp`` for ``git log --since``
    and ``head`` to detect concurrent writes that happened *after* its
    last run finished.
    """

    timestamp: datetime
    head: str | None


class OutmemState:
    """File-backed state for one wiki.

    Construct with the wiki root directory; the class manages the
    ``.outmem/`` subdirectory beneath it. Reads tolerate missing or
    malformed files (returns sensible defaults); writes are atomic.
    """

    def __init__(self, wiki_root: Path) -> None:
        self.wiki_root = Path(wiki_root)
        self.state_dir = self.wiki_root / STATE_DIR_NAME

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def ensure(self) -> None:
        """Create ``.outmem/`` and its ``.gitignore`` if missing.

        Idempotent — safe to call on every WikiStore open.
        """
        self.state_dir.mkdir(parents=True, exist_ok=True)
        gitignore = self.state_dir / ".gitignore"
        if not gitignore.exists():
            gitignore.write_text(_GITIGNORE_BODY, encoding="utf-8")

    # ------------------------------------------------------------------
    # Generic JSON helpers
    # ------------------------------------------------------------------

    def read_json(self, name: str) -> dict[str, Any] | None:
        """Read a JSON file under ``.outmem/`` or return ``None``.

        ``None`` is returned both for missing files and for malformed
        JSON, on the principle that state-cache corruption should
        gracefully fall back to a rebuild rather than hard-failing.
        """
        path = self.state_dir / name
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return data if isinstance(data, dict) else None

    def write_json(self, name: str, data: dict[str, Any]) -> None:
        """Atomically write a JSON file under ``.outmem/``.

        Holds an exclusive ``fcntl.flock`` on a per-file lockfile for
        the duration of the write so concurrent processes never
        clobber each other's updates. The lockfile is gitignored
        alongside the rest of ``.outmem/``.
        """
        self.ensure()
        target = self.state_dir / name
        tmp = target.with_suffix(target.suffix + ".tmp")
        lockfile = self.state_dir / f".{name}.lock"
        payload = json.dumps(data, indent=2, sort_keys=True)

        with open(lockfile, "a") as lock_fd:
            if _HAS_FLOCK:
                _fcntl.flock(lock_fd.fileno(), _fcntl.LOCK_EX)
            else:  # pragma: no cover — non-POSIX
                _log.warning(
                    "fcntl unavailable; OutmemState.write_json is not locked. "
                    "Concurrent writers may race."
                )
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, target)
            # Lock auto-released on lock_fd close at context exit.

    # ------------------------------------------------------------------
    # Last-run marker
    # ------------------------------------------------------------------

    def last_run(self) -> LastRun | None:
        """Return the most recently recorded run, or ``None`` if absent."""
        data = self.read_json(LAST_RUN_FILE)
        if not data:
            return None
        iso = data.get("timestamp")
        head = data.get("head")
        if not isinstance(iso, str):
            return None
        try:
            ts = parse_iso_z(iso)
        except ValueError:
            return None
        return LastRun(
            timestamp=ts,
            head=head if isinstance(head, str) and head else None,
        )

    def record_run(self, *, head: str | None, timestamp: datetime | None = None) -> LastRun:
        """Record a successful run; returns the persisted marker."""
        ts = ensure_utc(timestamp.replace(microsecond=0)) if timestamp else utc_now()
        self.write_json(
            LAST_RUN_FILE,
            {
                "timestamp": format_iso_z(ts),
                "head": head,
            },
        )
        return LastRun(timestamp=ts, head=head)
