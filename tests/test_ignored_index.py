"""A vector index the wiki's ``.gitignore`` excludes stays out of the commit.

A wiki that publishes its index separately ignores ``.vectors.db`` — tens
of megabytes of sqlite rewritten on every content push is history nobody
wants for a file ``outmem reindex`` regenerates. ``_commit_paths`` used to
append the index to every write's commit regardless and stage with
``git add`` (no ``-f``), so such a wiki failed every write — after the
reindex ran and the page and ``index.md`` were already staged, leaving the
write half-applied for the next commit to sweep up.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from outmem.store import WikiStore

from .conftest import _run_git


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=str(root), capture_output=True, text=True, check=True
    ).stdout


def _use_test_embeddings(root: Path) -> None:
    cfg = root / "config.yaml"
    cfg.write_text(
        cfg.read_text(encoding="utf-8").replace(
            "embedding_model:", "embedding_model: test:bag-of-words  #", 1
        ),
        encoding="utf-8",
    )


def _ignore(repo: Path, rule: str) -> None:
    """Append an ignore rule at the repository root and commit it."""
    gi = repo / ".gitignore"
    gi.write_text(gi.read_text(encoding="utf-8") + rule + "\n", encoding="utf-8")
    _git(repo, "add", ".gitignore")
    _git(
        repo, "-c", "user.name=T", "-c", "user.email=t@t.invalid",
        "commit", "-q", "--no-verify", "-m", "ignore the index",
    )


@pytest.fixture
def indexed(tmp_path: Path) -> WikiStore:
    """A standalone wiki with an index built and untracked."""
    root = tmp_path / "w"
    WikiStore.init(root).close()
    _use_test_embeddings(root)
    store = WikiStore.open(root)
    store.write_page("seed", title="Seed", body="Seed body.\n")
    store.semantic_reindex_all()
    assert (root / ".vectors.db").exists()
    assert ".vectors.db" not in _git(root, "ls-files")
    return store


class TestIgnoredIndexIsLeftOut:
    def test_the_write_commits_and_the_index_is_still_rebuilt(
        self, indexed: WikiStore
    ) -> None:
        root = indexed.root
        _ignore(root, ".vectors.db")
        db = root / ".vectors.db"
        before = db.stat().st_mtime_ns

        sha = indexed.write_page("abx:x", title="X", body="Body about x.\n")

        assert _git(root, "rev-parse", "HEAD").strip() == sha
        # Nothing half-applied: no staged leftovers, no untracked page.
        assert _git(root, "status", "--porcelain", "--untracked-files=no") == ""
        # The reindex still ran — that is the point of an index.
        assert db.stat().st_mtime_ns != before
        assert ".vectors.db" not in _git(root, "ls-files")

    def test_a_root_rule_reaches_a_nested_wiki(self, tmp_path: Path) -> None:
        # Multi-wiki: the rule lives in the repository's root .gitignore,
        # which the wiki's own directory never sees — the reason to ask
        # git rather than to pattern-match the wiki's file.
        repo = tmp_path / "mem"
        repo.mkdir()
        _run_git(["init", "--initial-branch", "main"], cwd=repo)
        (repo / "wikis.yaml").write_text(
            "wikis:\n  legal: {path: wikis/legal}\n", encoding="utf-8"
        )
        (repo / "wikis" / "legal").mkdir(parents=True)
        WikiStore.init(repo / "wikis" / "legal").close()
        _use_test_embeddings(repo / "wikis" / "legal")
        store = WikiStore.open(repo / "wikis" / "legal")
        store.write_page("seed", title="Seed", body="Seed body.\n")
        store.semantic_reindex_all()
        (repo / ".gitignore").write_text("wikis/*/.vectors.db\n", encoding="utf-8")
        _git(repo, "add", ".gitignore")
        _git(
            repo, "-c", "user.name=T", "-c", "user.email=t@t.invalid",
            "commit", "-q", "--no-verify", "-m", "ignore every wiki's index",
        )

        sha = store.write_page("nda", title="NDA", body="Terms.\n")

        assert _git(repo, "rev-parse", "HEAD").strip() == sha
        assert _git(repo, "status", "--porcelain", "--untracked-files=no") == ""
        assert "wikis/legal/.vectors.db" not in _git(repo, "ls-files")

    def test_a_tracked_index_keeps_being_committed(self, indexed: WikiStore) -> None:
        # Tracked beats ignored: git applies exclude rules to untracked
        # paths only, and a wiki that tracks its index keeps getting it in
        # every write's commit, rule or no rule.
        root = indexed.root
        _git(root, "add", "-f", ".vectors.db")
        _git(
            root, "-c", "user.name=T", "-c", "user.email=t@t.invalid",
            "commit", "-q", "--no-verify", "-m", "track the index",
        )
        _ignore(root, ".vectors.db")

        indexed.write_page("abx:y", title="Y", body="Body about y.\n")

        committed = _git(root, "show", "--name-only", "--format=", "HEAD").split()
        assert ".vectors.db" in committed

    def test_a_wiki_that_never_ignored_its_index_sees_no_change(
        self, indexed: WikiStore
    ) -> None:
        root = indexed.root
        indexed.write_page("abx:z", title="Z", body="Body about z.\n")
        committed = _git(root, "show", "--name-only", "--format=", "HEAD").split()
        assert ".vectors.db" in committed
