"""Subprocess wrappers around ``git``.

All functions take an explicit ``repo_path``, run ``git`` via ``argv``
lists (never ``shell=True``), capture stdout and stderr, and raise
:class:`GitOperationError` on non-zero exit. The wrappers are stateless —
the calling layer (typically :class:`outmem.store.WikiStore`) is
responsible for sequencing pull / commit / push.

The agent's commit identity is set per-call via ``-c user.name=…
-c user.email=…`` (spec v0.5 §3); we never mutate global git config.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from outmem.exceptions import GitOperationError

_log = logging.getLogger(__name__)

INDEX_LOCK_RELPATH = ".git/index.lock"
DEFAULT_STALE_LOCK_SECONDS = 60
DEFAULT_LOCK_RETRY_DELAY_MS = 100

# Record / field separators for ``--pretty=format``. These bytes never
# appear in real commit metadata, so splitting on them is unambiguous.
_FIELD_SEP = "\x1f"
_RECORD_SEP = "\x1e"

_LOG_FORMAT = _FIELD_SEP.join(["%H", "%an", "%ae", "%aI", "%s", "%b"]) + _RECORD_SEP


@dataclass(frozen=True)
class CommitInfo:
    """Structured view of a single commit."""

    sha: str
    author_name: str
    author_email: str
    date: datetime
    subject: str
    body: str


# ---------------------------------------------------------------------------
# Repo lifecycle
# ---------------------------------------------------------------------------


def git_available() -> bool:
    """Return True iff a ``git`` executable is on PATH."""
    return shutil.which("git") is not None


def is_git_repo(repo_path: Path) -> bool:
    """Return True iff ``repo_path`` is the working tree of a git repo."""
    git_dir = repo_path / ".git"
    return git_dir.is_dir() or git_dir.is_file()


def init_repo(repo_path: Path, *, initial_branch: str = "main") -> None:
    """Initialise a new git repo at ``repo_path`` with ``main`` as default.

    Creates the directory if it does not exist. No-op if a repo already
    lives there.
    """
    repo_path.mkdir(parents=True, exist_ok=True)
    if is_git_repo(repo_path):
        return
    _run_git(["init", "--initial-branch", initial_branch], cwd=repo_path)


def current_head(repo_path: Path) -> str:
    """Return the SHA of the current HEAD commit.

    Raises :class:`GitOperationError` if the repo has no commits yet.
    """
    return _run_git(["rev-parse", "HEAD"], cwd=repo_path).strip()


def head_or_none(repo_path: Path) -> str | None:
    """Return the SHA of HEAD, or ``None`` if the repo has no commits."""
    try:
        return current_head(repo_path)
    except GitOperationError:
        return None


# ---------------------------------------------------------------------------
# Sync
# ---------------------------------------------------------------------------


def pull_rebase(repo_path: Path, *, remote: str = "origin", branch: str = "main") -> None:
    """Run ``git pull --rebase <remote> <branch>``."""
    _run_git(["pull", "--rebase", remote, branch], cwd=repo_path)


def push(repo_path: Path, *, remote: str = "origin", branch: str = "main") -> None:
    """Run ``git push <remote> <branch>``."""
    _run_git(["push", remote, branch], cwd=repo_path)


def has_remote(repo_path: Path, *, remote: str = "origin") -> bool:
    """Return ``True`` iff the repo has a remote with the given name.

    Used by the agent runtime to detect local-only wikis (no origin
    configured) and skip the push step rather than failing the
    mandatory-writeback contract on a non-existent remote.
    """
    try:
        _run_git(["remote", "get-url", remote], cwd=repo_path)
    except GitOperationError:
        return False
    return True


# ---------------------------------------------------------------------------
# Staging and commits
# ---------------------------------------------------------------------------


def add(repo_path: Path, paths: Sequence[str | Path]) -> None:
    """Stage one or more paths via ``git add``."""
    if not paths:
        return
    _run_git(["add", "--", *[str(p) for p in paths]], cwd=repo_path)


def staged_changes(repo_path: Path) -> tuple[list[str], list[str]]:
    """Return ``(added_or_modified, deleted)`` paths in the git index.

    Walks ``git diff --cached --name-status -z`` and splits the changes
    into "exists on disk now" vs "was deleted". Renames count as a
    delete (old name) plus a modification (new name) so semantic
    indexes can drop the stale entry and add the renamed one.

    Used by the pre-commit hook (``outmem reindex --staged``) to keep
    the vector DB in lockstep with externally edited files.
    """
    raw = _run_git(
        ["diff", "--cached", "--name-status", "-z"],
        cwd=repo_path,
    )
    if not raw:
        return [], []

    # The ``-z`` form is null-terminated. Renames take two paths
    # (R<score>\0old\0new\0); everything else is one path
    # (M\0path\0).
    tokens = raw.split("\x00")
    if tokens and tokens[-1] == "":
        tokens = tokens[:-1]

    added: list[str] = []
    deleted: list[str] = []
    i = 0
    while i < len(tokens):
        status = tokens[i]
        i += 1
        if not status:
            continue
        if status.startswith(("R", "C")) and i + 1 < len(tokens):
            old_path = tokens[i]
            new_path = tokens[i + 1]
            i += 2
            if status.startswith("R"):
                deleted.append(old_path)
            added.append(new_path)
        elif i < len(tokens):
            path = tokens[i]
            i += 1
            if status.startswith("D"):
                deleted.append(path)
            else:
                added.append(path)
    return added, deleted


def path_is_dirty(repo_path: Path, rel_path: str) -> bool:
    """Return ``True`` if ``rel_path`` has changes versus its tracked
    state in git (either staged or unstaged).

    Untracked files count as dirty. Used by helpers like
    :meth:`WikiStore.rebuild_index` to decide whether a regen actually
    needs a commit — ``git commit`` of an unchanged tree fails with
    "nothing to commit", and "did the file change at all" is the
    cleanest precondition to check.
    """
    raw = _run_git(
        ["status", "--porcelain", "--", rel_path],
        cwd=repo_path,
    )
    return bool(raw.strip())


def commit_as(
    repo_path: Path,
    *,
    message: str,
    author_name: str,
    author_email: str,
    allow_empty: bool = False,
) -> str:
    """Create a commit under the supplied identity.

    Uses ``git -c user.name=… -c user.email=…`` so we never depend on or
    mutate the user's global git config (spec v0.5 §3). Returns the
    new HEAD SHA.
    """
    if not author_name.strip():
        raise GitOperationError("commit_as: author_name must be non-empty.")
    if not author_email.strip():
        raise GitOperationError("commit_as: author_email must be non-empty.")
    if not message.strip():
        raise GitOperationError("commit_as: message must be non-empty.")

    # spec v0.5 §12 explicitly defers GPG-signed agent commits to v0.2;
    # turn signing off so v0.1 commits succeed regardless of the user's
    # global ``commit.gpgsign`` setting.
    args = [
        "-c",
        f"user.name={author_name}",
        "-c",
        f"user.email={author_email}",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-m",
        message,
    ]
    if allow_empty:
        args.append("--allow-empty")
    _run_git(args, cwd=repo_path)
    return current_head(repo_path)


# ---------------------------------------------------------------------------
# Log queries
# ---------------------------------------------------------------------------


def log_since(
    repo_path: Path,
    *,
    since: datetime | str | None = None,
    paths: Sequence[str | Path] | None = None,
    author: str | None = None,
    exclude_author: str | None = None,
) -> list[CommitInfo]:
    """Run ``git log`` with the given filters and return parsed commits.

    Args:
        since: Only include commits authored at or after this point.
            Accepts a datetime or a string ``git log --since`` understands.
        paths: Restrict the log to commits touching these paths.
        author: Only include commits by this author (matched against
            ``user.name`` or ``user.email`` via ``--author``).
        exclude_author: Exclude commits by this author. Used by the
            steering loop to drop the agent's own commits.
    """
    args = ["log", f"--pretty=format:{_LOG_FORMAT}"]
    if since is not None:
        args.append(f"--since={_format_since(since)}")
    if author is not None:
        args.append(f"--author={author}")
    if exclude_author is not None:
        args.extend(["--perl-regexp", f"--author=^(?!.*{_escape_perl(exclude_author)}).*$"])
    if paths:
        args.append("--")
        args.extend(str(p) for p in paths)
    raw = _run_git(args, cwd=repo_path)
    return _parse_log(raw)


def log_range(
    repo_path: Path,
    *,
    range_spec: str,
    author: str | None = None,
    paths: Sequence[str | Path] | None = None,
) -> list[CommitInfo]:
    """Run ``git log <range_spec>`` and return parsed commits.

    ``range_spec`` is anything ``git log`` accepts as a revision range
    — ``A..B`` for "commits reachable from B but not A",
    ``A...B`` for symmetric difference, or just ``B`` for "history of B".

    Used by the agent runtime to count commits authored by the agent
    between ``head_before`` and ``head_after``: the writeback contract
    in spec v0.5 §9 requires distinguishing *the agent committed* from
    *something else moved HEAD*, which the raw ``head_before !=
    head_after`` test conflates.
    """
    args = ["log", range_spec, f"--pretty=format:{_LOG_FORMAT}"]
    if author is not None:
        args.append(f"--author={author}")
    if paths:
        args.append("--")
        args.extend(str(p) for p in paths)
    raw = _run_git(args, cwd=repo_path)
    return _parse_log(raw)


def log_for_paths(
    repo_path: Path,
    paths: Sequence[str | Path],
    *,
    follow: bool = True,
    with_patch: bool = False,
) -> str:
    """Return the raw ``git log`` output for the given paths.

    With ``with_patch=True`` and ``follow=True`` this is the temporal
    evolution pattern from spec v0.5 §8: the EXPANSION branch of the
    planning prompt walks history this way to surface how a topic has
    been treated over time.
    """
    if not paths:
        raise GitOperationError("log_for_paths: at least one path is required.")
    args = ["log"]
    if follow:
        args.append("--follow")
    if with_patch:
        args.append("-p")
    args.append("--")
    args.extend(str(p) for p in paths)
    return _run_git(args, cwd=repo_path)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def clear_stale_index_lock(
    repo_path: Path, *, max_age_seconds: int = DEFAULT_STALE_LOCK_SECONDS
) -> bool:
    """Remove ``<repo>/.git/index.lock`` if it's older than ``max_age_seconds``.

    Interrupted ``git commit`` / ``git add`` runs (Ctrl-C, killed
    terminals, crashed processes) can leave the index lock file
    behind. Subsequent git operations then fail with "Unable to create
    ``.git/index.lock``: File exists" until someone removes it
    manually. This helper is the automated equivalent: only files
    older than the threshold are removed, so a genuinely concurrent
    git process is left alone.

    Returns ``True`` if a lock file was removed.
    """
    lock = repo_path / INDEX_LOCK_RELPATH
    if not lock.exists():
        return False
    try:
        age = time.time() - lock.stat().st_mtime
    except OSError:
        return False
    if age < max_age_seconds:
        return False
    try:
        lock.unlink()
    except OSError as exc:
        _log.warning("could not remove stale %s: %s", lock, exc)
        return False
    _log.info("removed stale %s (age %.0fs)", lock, age)
    return True


def _is_index_lock_error(stderr: str) -> bool:
    """Heuristic — did this git invocation fail because of ``index.lock``?"""
    return "index.lock" in stderr and "File exists" in stderr


def _run_git(
    args: Sequence[str],
    *,
    cwd: Path,
    retry_on_lock: bool = True,
    lock_retry_delay_ms: int = DEFAULT_LOCK_RETRY_DELAY_MS,
) -> str:
    """Run ``git <args>`` and return stdout.

    Always uses argv lists (``shell=False``). Captures stdout and stderr
    so failures surface with full context. When ``retry_on_lock=True``
    (the default), a single failure due to ``.git/index.lock`` is
    retried once after a brief sleep — this absorbs both transient
    races between rapid-fire commits and stale locks left by killed
    prior runs (since the second attempt may also find the lock has
    aged past the cleanup threshold the caller applied earlier).
    """
    if not git_available():
        raise GitOperationError("git executable not found on PATH.")
    if not cwd.exists():
        raise GitOperationError(f"git working directory does not exist: {cwd}")

    env = os.environ.copy()
    # Disable any system-wide aliases / hooks-on-by-default config that
    # might lurk in the user's home dir while leaving the per-repo
    # config (the wiki repo's own .git/config) intact.
    env.setdefault("GIT_TERMINAL_PROMPT", "0")

    attempts = 2 if retry_on_lock else 1
    last_stderr = ""
    last_stdout = ""
    last_returncode = 0
    for attempt in range(attempts):
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=str(cwd),
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError as exc:
            raise GitOperationError(f"git invocation failed: {exc}") from exc

        if result.returncode == 0:
            return result.stdout

        last_stderr = result.stderr
        last_stdout = result.stdout
        last_returncode = result.returncode

        if attempt + 1 < attempts and _is_index_lock_error(result.stderr):
            _log.info(
                "git op hit index.lock; retrying after %dms",
                lock_retry_delay_ms,
            )
            time.sleep(lock_retry_delay_ms / 1000)
            # Don't auto-remove the lock here — that's :func:`clear_stale_index_lock`'s
            # job and it consults the user's configured threshold. If the
            # lock is still there after the retry delay we surface the
            # original error and let the next `outmem.open()` clean up
            # if the lock has aged past the stale threshold.
            continue
        break

    cmd = " ".join(args[:4]) + ("…" if len(args) > 4 else "")
    raise GitOperationError(
        f"git {cmd} failed (exit {last_returncode}): {last_stderr.strip() or last_stdout.strip()}"
    )


def _parse_log(raw: str) -> list[CommitInfo]:
    """Split ``git log --pretty=format`` output into :class:`CommitInfo`."""
    if not raw.strip():
        return []
    records = raw.split(_RECORD_SEP)
    out: list[CommitInfo] = []
    for record in records:
        record = record.strip("\n")
        if not record:
            continue
        fields = record.split(_FIELD_SEP)
        if len(fields) < 6:
            continue
        sha, name, email, iso, subject, body = fields[:6]
        try:
            date = datetime.fromisoformat(iso)
        except ValueError:
            continue
        out.append(
            CommitInfo(
                sha=sha,
                author_name=name,
                author_email=email,
                date=date,
                subject=subject,
                body=body,
            )
        )
    return out


def _format_since(value: datetime | str) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _escape_perl(text: str) -> str:
    """Escape regex metacharacters for a Perl-compatible ``--author`` pattern."""
    return re.escape(text)
