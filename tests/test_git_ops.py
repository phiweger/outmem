"""Tests for ``outmem.git_ops``."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from outmem.exceptions import GitOperationError
from outmem.git_ops import (
    add,
    commit_as,
    current_head,
    git_available,
    head_or_none,
    init_repo,
    is_git_repo,
    log_for_paths,
    log_range,
    log_since,
    pull_rebase,
    push,
)


class TestRepoLifecycle:
    def test_git_available(self) -> None:
        assert git_available()

    def test_is_git_repo_false_for_empty_dir(self, tmp_path: Path) -> None:
        assert is_git_repo(tmp_path) is False

    def test_init_creates_repo(self, tmp_path: Path) -> None:
        repo = tmp_path / "new"
        init_repo(repo)
        assert is_git_repo(repo)

    def test_init_is_idempotent(self, git_repo: Path) -> None:
        init_repo(git_repo)  # should not raise
        assert is_git_repo(git_repo)

    def test_head_before_first_commit_raises(self, git_repo: Path) -> None:
        with pytest.raises(GitOperationError):
            current_head(git_repo)

    def test_head_or_none_before_first_commit(self, git_repo: Path) -> None:
        assert head_or_none(git_repo) is None

    def test_head_after_commit(self, populated_repo: Path) -> None:
        sha = current_head(populated_repo)
        assert len(sha) == 40
        assert all(c in "0123456789abcdef" for c in sha)


class TestCommit:
    def test_commit_writes_file(self, git_repo: Path) -> None:
        (git_repo / "hello.md").write_text("hi\n")
        add(git_repo, ["hello.md"])
        sha = commit_as(
            git_repo,
            message="compact: hello",
            author_name="Test",
            author_email="test@example.com",
        )
        assert len(sha) == 40

    def test_commit_records_supplied_identity(self, git_repo: Path) -> None:
        (git_repo / "x.md").write_text("body\n")
        add(git_repo, ["x.md"])
        commit_as(
            git_repo,
            message="compact: x",
            author_name="outmem agent",
            author_email="agent@host",
        )
        log = log_since(git_repo)
        assert log[0].author_name == "outmem agent"
        assert log[0].author_email == "agent@host"

    def test_commit_without_changes_fails(self, git_repo: Path) -> None:
        with pytest.raises(GitOperationError):
            commit_as(
                git_repo,
                message="compact: nothing",
                author_name="Test",
                author_email="test@example.com",
            )

    def test_commit_empty_message_rejected(self, git_repo: Path) -> None:
        (git_repo / "x.md").write_text("body\n")
        add(git_repo, ["x.md"])
        with pytest.raises(GitOperationError, match="message"):
            commit_as(
                git_repo,
                message="   ",
                author_name="Test",
                author_email="test@example.com",
            )

    def test_commit_empty_identity_rejected(self, git_repo: Path) -> None:
        (git_repo / "x.md").write_text("body\n")
        add(git_repo, ["x.md"])
        with pytest.raises(GitOperationError, match="author_name"):
            commit_as(
                git_repo,
                message="compact: x",
                author_name="",
                author_email="test@example.com",
            )
        with pytest.raises(GitOperationError, match="author_email"):
            commit_as(
                git_repo,
                message="compact: x",
                author_name="Test",
                author_email="",
            )

    def test_add_empty_list_is_noop(self, git_repo: Path) -> None:
        add(git_repo, [])  # should not raise


class TestLogSince:
    def test_returns_all_commits_newest_first(self, populated_repo: Path) -> None:
        log = log_since(populated_repo)
        assert len(log) == 4
        subjects = [c.subject for c in log]
        assert subjects[0] == "log: pricing-inconsistency"
        assert subjects[-1] == "compact: pricing-formula"

    def test_paths_filter(self, populated_repo: Path) -> None:
        log = log_since(populated_repo, paths=["wiki/pages/pricing-formula.md"])
        assert len(log) == 2
        assert all("pricing-formula" in c.subject for c in log)

    def test_exclude_author(self, populated_repo: Path) -> None:
        log = log_since(populated_repo, exclude_author="agent@host")
        emails = {c.author_email for c in log}
        assert "agent@host" not in emails
        assert "alice@example.com" in emails
        assert "bob@example.com" in emails

    def test_author_filter(self, populated_repo: Path) -> None:
        log = log_since(populated_repo, author="agent@host")
        emails = {c.author_email for c in log}
        assert emails == {"agent@host"}

    def test_parses_iso_date(self, populated_repo: Path) -> None:
        log = log_since(populated_repo)
        assert log[0].date.tzinfo is not None  # commit dates carry an offset


class TestLogForPaths:
    def test_with_patch_includes_diff(self, populated_repo: Path) -> None:
        out = log_for_paths(
            populated_repo,
            ["wiki/pages/pricing-formula.md"],
            follow=True,
            with_patch=True,
        )
        assert "diff --git" in out
        assert "v2 with clarification" in out

    def test_without_patch_omits_diff(self, populated_repo: Path) -> None:
        out = log_for_paths(
            populated_repo,
            ["wiki/pages/pricing-formula.md"],
            follow=True,
            with_patch=False,
        )
        assert "diff --git" not in out
        assert "extend: pricing-formula" in out

    def test_empty_paths_rejected(self, populated_repo: Path) -> None:
        with pytest.raises(GitOperationError, match="at least one path"):
            log_for_paths(populated_repo, [])


class TestPushPull:
    def test_push_publishes_commits(self, populated_repo: Path, bare_remote: Path) -> None:
        subprocess.run(
            ["git", "remote", "add", "origin", str(bare_remote)],
            cwd=str(populated_repo),
            check=True,
            capture_output=True,
        )
        push(populated_repo, remote="origin", branch="main")
        # The bare remote should now have refs/heads/main pointing at our HEAD.
        result = subprocess.run(
            ["git", "rev-parse", "refs/heads/main"],
            cwd=str(bare_remote),
            check=True,
            capture_output=True,
            text=True,
        )
        assert result.stdout.strip() == current_head(populated_repo)

    def test_pull_rebase_fast_forwards(
        self,
        populated_repo: Path,
        bare_remote: Path,
        tmp_path: Path,
    ) -> None:
        # Wire repo A to the bare remote and push.
        subprocess.run(
            ["git", "remote", "add", "origin", str(bare_remote)],
            cwd=str(populated_repo),
            check=True,
            capture_output=True,
        )
        push(populated_repo, remote="origin", branch="main")

        # Clone the remote into repo B.
        clone = tmp_path / "clone"
        subprocess.run(
            ["git", "clone", str(bare_remote), str(clone)],
            check=True,
            capture_output=True,
        )

        # Repo A adds another commit and pushes.
        (populated_repo / "wiki/pages/new-page.md").write_text(
            "---\ntitle: New page\nslug: new-page\n---\n\nbody\n"
        )
        add(populated_repo, ["wiki/pages/new-page.md"])
        commit_as(
            populated_repo,
            message="compact: new-page",
            author_name="Test",
            author_email="test@example.com",
        )
        push(populated_repo, remote="origin", branch="main")

        # Repo B does a pull --rebase and should now see the new commit.
        pull_rebase(clone, remote="origin", branch="main")
        assert (clone / "wiki/pages/new-page.md").exists()


class TestLogRange:
    def test_returns_commits_in_range(self, populated_repo: Path) -> None:
        # The fixture has 4 commits; grab two boundary SHAs from log_since.
        log = log_since(populated_repo)
        head = log[0].sha
        first = log[-1].sha
        out = log_range(populated_repo, range_spec=f"{first}..{head}")
        # `first..head` excludes `first` itself, so 3 commits.
        assert len(out) == 3
        assert all(c.sha != first for c in out)

    def test_filters_by_author(self, populated_repo: Path) -> None:
        log = log_since(populated_repo)
        head = log[0].sha
        agent_only = log_range(
            populated_repo,
            range_spec=head,
            author="agent@host",
        )
        emails = {c.author_email for c in agent_only}
        assert emails == {"agent@host"}

    def test_empty_range_returns_empty(self, populated_repo: Path) -> None:
        head = current_head(populated_repo)
        assert log_range(populated_repo, range_spec=f"{head}..{head}") == []


class TestErrorPaths:
    def test_missing_cwd_raises(self, tmp_path: Path) -> None:
        with pytest.raises(GitOperationError, match="does not exist"):
            current_head(tmp_path / "nowhere")

    def test_non_repo_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(GitOperationError):
            current_head(tmp_path)
