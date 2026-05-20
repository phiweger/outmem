"""Tests for stale-index-lock recovery and the retry-on-lock path in git_ops."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from outmem.exceptions import GitOperationError
from outmem.git_ops import (
    INDEX_LOCK_RELPATH,
    _is_index_lock_error,
    _run_git,
    clear_stale_index_lock,
)

# ---------------------------------------------------------------------------
# clear_stale_index_lock
# ---------------------------------------------------------------------------


def test_clear_returns_false_when_no_lock(git_repo: Path) -> None:
    assert clear_stale_index_lock(git_repo) is False


def test_clear_leaves_fresh_lock_alone(git_repo: Path) -> None:
    lock = git_repo / INDEX_LOCK_RELPATH
    lock.touch()
    assert clear_stale_index_lock(git_repo, max_age_seconds=60) is False
    assert lock.exists()


def test_clear_removes_stale_lock(git_repo: Path) -> None:
    lock = git_repo / INDEX_LOCK_RELPATH
    lock.touch()
    # Backdate the mtime so the lock is considered stale.
    past = time.time() - 3600  # one hour ago
    os.utime(lock, (past, past))

    assert clear_stale_index_lock(git_repo, max_age_seconds=60) is True
    assert not lock.exists()


def test_clear_zero_threshold_removes_anything(git_repo: Path) -> None:
    lock = git_repo / INDEX_LOCK_RELPATH
    lock.touch()
    assert clear_stale_index_lock(git_repo, max_age_seconds=0) is True
    assert not lock.exists()


# ---------------------------------------------------------------------------
# index.lock error detection
# ---------------------------------------------------------------------------


def test_is_index_lock_error_positive() -> None:
    msg = (
        "fatal: Unable to create '/tmp/wiki/.git/index.lock': File exists.\n"
        "Another git process seems to be running in this repository, e.g."
    )
    assert _is_index_lock_error(msg) is True


def test_is_index_lock_error_negative() -> None:
    assert _is_index_lock_error("fatal: pathspec 'x' did not match") is False
    assert _is_index_lock_error("") is False


# ---------------------------------------------------------------------------
# _run_git retry-on-lock
# ---------------------------------------------------------------------------


def test_run_git_retries_once_then_succeeds(
    git_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Simulate `git add` racing with a stale lock that clears between attempts."""
    import subprocess

    calls = {"n": 0}
    fail_then_succeed = [
        subprocess.CompletedProcess(
            args=["git"],
            returncode=128,
            stdout="",
            stderr="fatal: Unable to create '/repo/.git/index.lock': File exists.\n",
        ),
        subprocess.CompletedProcess(
            args=["git"],
            returncode=0,
            stdout="ok",
            stderr="",
        ),
    ]

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess:
        result = fail_then_succeed[min(calls["n"], len(fail_then_succeed) - 1)]
        calls["n"] += 1
        return result

    monkeypatch.setattr("outmem.git_ops.subprocess.run", fake_run)

    out = _run_git(["status"], cwd=git_repo, lock_retry_delay_ms=0)
    assert out == "ok"
    assert calls["n"] == 2


def test_run_git_raises_when_retry_also_fails(
    git_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess

    def always_fail(*args: object, **kwargs: object) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(
            args=["git"],
            returncode=128,
            stdout="",
            stderr="fatal: Unable to create '/repo/.git/index.lock': File exists.\n",
        )

    monkeypatch.setattr("outmem.git_ops.subprocess.run", always_fail)

    with pytest.raises(GitOperationError, match=r"index\.lock"):
        _run_git(["status"], cwd=git_repo, lock_retry_delay_ms=0)


def test_run_git_no_retry_when_disabled(git_repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    calls = {"n": 0}

    def always_lock(*args: object, **kwargs: object) -> subprocess.CompletedProcess:
        calls["n"] += 1
        return subprocess.CompletedProcess(
            args=["git"],
            returncode=128,
            stdout="",
            stderr="fatal: Unable to create '/repo/.git/index.lock': File exists.\n",
        )

    monkeypatch.setattr("outmem.git_ops.subprocess.run", always_lock)

    with pytest.raises(GitOperationError):
        _run_git(["status"], cwd=git_repo, retry_on_lock=False)
    assert calls["n"] == 1


def test_run_git_does_not_retry_for_non_lock_errors(
    git_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Other failures must not silently retry — only index.lock does."""
    import subprocess

    calls = {"n": 0}

    def other_failure(*args: object, **kwargs: object) -> subprocess.CompletedProcess:
        calls["n"] += 1
        return subprocess.CompletedProcess(
            args=["git"],
            returncode=128,
            stdout="",
            stderr="fatal: pathspec 'nothing' did not match any files",
        )

    monkeypatch.setattr("outmem.git_ops.subprocess.run", other_failure)

    with pytest.raises(GitOperationError, match="pathspec"):
        _run_git(["add", "nothing"], cwd=git_repo)
    assert calls["n"] == 1
