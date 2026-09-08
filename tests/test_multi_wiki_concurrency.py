"""Several processes committing into one repository.

A commit is two operations — `git add` writes the index, `git commit`
reads it back — and between them the index is shared state. Two wikis in
one repository make concurrent writers ordinary rather than exotic, and
`WikiStore._write_lock` is per-store and per-process, so it cannot cover
them.

The interesting failure is not the noisy ``index.lock: File exists``.
It is the quiet one: one process's staged paths ride into another's
commit, under that commit's subject and author. Nothing errors and the
history is simply wrong. These tests assert against that directly.
"""

from __future__ import annotations

import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest

from outmem._store.locking import COMMIT_LOCK_FILENAME, REPO_STATE_DIR
from outmem.store import WikiStore

from .conftest import _run_git

WIKIS = ("alpha", "beta", "gamma")
PAGES_PER_WIKI = 6


def _build(root: Path) -> None:
    root.mkdir(parents=True)
    _run_git(["init", "--initial-branch", "main"], cwd=root)
    (root / "wikis.yaml").write_text(
        "wikis:\n" + "".join(f"  {w}: {{path: wikis/{w}}}\n" for w in WIKIS),
        encoding="utf-8",
    )
    for name in WIKIS:
        (root / "wikis" / name).mkdir(parents=True)
        WikiStore.init(root / "wikis" / name)


def _write_pages(args: tuple[str, str, int]) -> tuple[str, int]:
    """Run in a child process: write ``n`` pages into one wiki."""
    root, wiki, n = args
    store = WikiStore.open(Path(root) / "wikis" / wiki)
    written = 0
    for i in range(n):
        try:
            store.write_page(f"p{i}", title=f"P{i}", body=f"Body {wiki} {i}.\n")
        except Exception as exc:  # reported, not swallowed
            print(f"{wiki}/p{i}: {type(exc).__name__}: {exc}", file=sys.stderr)
        else:
            written += 1
    return wiki, written


def _commits_touching_other_wikis(root: Path) -> list[tuple[str, str]]:
    """Commits whose subject names one wiki but whose paths name another."""
    raw = subprocess.run(
        ["git", "log", "--format=%s", "--name-only"],
        cwd=str(root),
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    mixed: list[tuple[str, str]] = []
    subject = ""
    for line in raw.splitlines():
        if not line:
            continue
        if line.startswith("wikis/"):
            owner = line.split("/")[1]
            if subject and not subject.startswith(f"{owner}/"):
                mixed.append((subject, line))
        else:
            subject = line
    return mixed


class TestConcurrentCommits:
    @pytest.fixture
    def hammered(self, tmp_path: Path) -> tuple[Path, dict[str, int]]:
        root = tmp_path / "mem"
        _build(root)
        with ProcessPoolExecutor(max_workers=len(WIKIS)) as pool:
            results = list(
                pool.map(
                    _write_pages,
                    [(str(root), w, PAGES_PER_WIKI) for w in WIKIS],
                )
            )
        return root, dict(results)

    def test_no_write_is_refused(
        self, hammered: tuple[Path, dict[str, int]]
    ) -> None:
        _root, written = hammered
        assert written == {w: PAGES_PER_WIKI for w in WIKIS}

    def test_no_commit_carries_another_wikis_paths(
        self, hammered: tuple[Path, dict[str, int]]
    ) -> None:
        root, _written = hammered
        assert _commits_touching_other_wikis(root) == []

    def test_every_reported_write_is_in_head(
        self, hammered: tuple[Path, dict[str, int]]
    ) -> None:
        # `write_page` returning a sha and the page not being in HEAD is
        # the silent half of the failure: the caller was told it worked.
        root, written = hammered
        tracked = set(
            subprocess.run(
                ["git", "ls-files"],
                cwd=str(root),
                check=True,
                capture_output=True,
                text=True,
            ).stdout.splitlines()
        )
        missing = [
            f"wikis/{w}/wiki/pages/p{i}.md"
            for w, n in written.items()
            for i in range(n)
            if f"wikis/{w}/wiki/pages/p{i}.md" not in tracked
        ]
        assert missing == []


class TestLockPlacement:
    def test_the_lock_lives_at_the_repository_root(self, tmp_path: Path) -> None:
        root = tmp_path / "mem"
        _build(root)
        store = WikiStore.open(root / "wikis" / "alpha")
        store.write_page("x", title="X", body="Body.\n")
        # Repository-level, not per-wiki: several wikis each holding their
        # own lock would serialise nothing.
        assert (root / REPO_STATE_DIR / COMMIT_LOCK_FILENAME).is_file()
        assert not (store.root / REPO_STATE_DIR).exists()

    def test_the_lock_directory_is_gitignored(self, tmp_path: Path) -> None:
        root = tmp_path / "mem"
        _build(root)
        WikiStore.open(root / "wikis" / "alpha").write_page(
            "x", title="X", body="Body.\n"
        )
        ignored = subprocess.run(
            ["git", "status", "--porcelain", "--ignored=no"],
            cwd=str(root),
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        assert REPO_STATE_DIR not in ignored

    def test_a_standalone_wiki_still_commits(self, tmp_path: Path) -> None:
        store = WikiStore.init(tmp_path / "solo")
        store.write_page("x", title="X", body="Body.\n")
        assert store.exists("x")
        assert (store.root / REPO_STATE_DIR / COMMIT_LOCK_FILENAME).is_file()

    def test_an_unwritable_lock_directory_does_not_block_the_commit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A read-only mount or a permissions problem must not stop a
        # single-writer wiki from working: not being able to take the lock
        # is a reason to go unserialised, not a reason to refuse.
        store = WikiStore.init(tmp_path / "solo")
        monkeypatch.setattr(
            "outmem._store.locking._prepare", lambda _repo: None
        )
        store.write_page("x", title="X", body="Body.\n")
        assert store.exists("x")


class TestLockPreparationFailure:
    def test_a_lock_directory_that_cannot_be_created_degrades_to_unlocked(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The real branch, not a stub of it: `mkdir` raising for the lock
        # directory alone. A wiki with one writer must keep working where
        # a lock cannot be taken.
        from outmem._store.locking import REPO_STATE_DIR

        store = WikiStore.init(tmp_path / "solo")
        real_mkdir = Path.mkdir

        def refuse_lock_dir(self: Path, *a: object, **k: object) -> None:
            if self.name == REPO_STATE_DIR:
                raise OSError("read-only file system")
            real_mkdir(self, *a, **k)  # type: ignore[arg-type]

        monkeypatch.setattr(Path, "mkdir", refuse_lock_dir)
        store.write_page("x", title="X", body="Body.\n")
        assert store.exists("x")
        assert not (store.root / REPO_STATE_DIR).exists()
