"""Repository-wide serialisation of the stage-and-commit sequence.

A commit is not one operation. `git add` writes the index, `git commit`
reads it back and moves the ref, and between those two steps the index is
shared state that any other process in the same repository can overwrite.
Two wikis in one repository make that ordinary rather than exotic:
separate stores, separate `WikiStore._write_lock`s, separate processes —
and one `.git/index`.

The failure is not only the visible ``.git/index.lock: File exists``.
Worse is the quiet one: process A stages its page, process B stages its
own and commits, and A's paths ride along in B's commit under B's
subject and author. Nothing errors, and the history is wrong.

So the sequence is serialised across the whole repository with an
`fcntl.flock` on a lockfile beside it. A flock is held by the file
description and released when it closes — including when the process
dies — so a crash cannot leave the repository wedged, which is the
property a lockfile-as-mutex does not have.

Non-POSIX platforms have no `fcntl`; the lock degrades to a warning
there, as it does in :mod:`outmem.state`. Outmem targets Linux servers.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

try:
    import fcntl as _fcntl

    _HAS_FLOCK = True
except ImportError:  # pragma: no cover — non-POSIX fallback
    _fcntl = None  # type: ignore[assignment]
    _HAS_FLOCK = False

_log = logging.getLogger(__name__)

# Repository-level state, one directory up from any wiki. Not `.git/`,
# which is git's to manage, and not a wiki's `.outmem/`, which several
# wikis in one repository would each have their own of — defeating the
# point.
REPO_STATE_DIR = ".outmem-repo"
COMMIT_LOCK_FILENAME = "commit.lock"

# `*` with no `!.gitignore` exception, so the ignore file ignores itself
# too and the directory is invisible to `git status`. It is created on
# demand by whichever process commits first — unlike a wiki's `.outmem/`,
# whose ignore rule is meant to travel with the wiki, nothing here needs
# to be committed, and an untracked file appearing in a repository the
# moment somebody writes is its own small confusion.
_GITIGNORE_BODY = "*\n"


@contextmanager
def repo_commit_lock(repo: Path) -> Iterator[None]:
    """Hold the repository's commit lock for the duration of the block.

    Blocking and re-entrant across processes only in the sense flock is:
    a second acquisition *within one process* on a new file description
    would deadlock, so callers must not nest this. `_commit_paths` is the
    single caller for exactly that reason.

    Never fatal on its own. If the lock directory cannot be created — a
    read-only mount, a permissions problem — the commit proceeds
    unserialised rather than failing, on the principle that a wiki with
    one writer must keep working where a lock cannot be taken.
    """
    lock_path = _prepare(repo)
    if lock_path is None:
        yield
        return
    with open(lock_path, "a") as fd:
        if _HAS_FLOCK:
            _fcntl.flock(fd.fileno(), _fcntl.LOCK_EX)
        else:  # pragma: no cover — non-POSIX
            _log.warning(
                "fcntl unavailable; commits in %s are not serialised across "
                "processes. Concurrent writers may race on the git index.",
                repo,
            )
        yield
        # Released when `fd` closes, including if the block raised.


def _prepare(repo: Path) -> Path | None:
    """Create the lock directory and return the lockfile path, or ``None``."""
    state_dir = repo / REPO_STATE_DIR
    try:
        state_dir.mkdir(parents=True, exist_ok=True)
        gitignore = state_dir / ".gitignore"
        if not gitignore.exists():
            gitignore.write_text(_GITIGNORE_BODY, encoding="utf-8")
        return state_dir / COMMIT_LOCK_FILENAME
    except OSError as exc:
        _log.debug("commit lock unavailable at %s: %s", state_dir, exc)
        return None
