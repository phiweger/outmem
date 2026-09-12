"""A write holds the repository lock for its whole body, not just its commit.

What a write rewrites *before* committing — ``wiki/index.md``,
``.sources.db``, ``.vectors.db`` — is shared with every other writer of
the same wiki, and git refuses to stage a file that changes under its
hash: ``fatal: confused by unstable object source data``. With the lock
covering only stage-and-commit, two processes registering sources into
one wiki failed each other at ``git add`` — the README's "parallel
ingest is safe" was not true. The lock is re-entrant within a thread so
a method can hold it across its body and the commit funnel inside can
take it again.
"""

from __future__ import annotations

import multiprocessing as mp
import threading
import time
from pathlib import Path

from outmem._store.locking import repo_commit_lock
from outmem.store import WikiStore

PROCS = 8
PER_PROC = 6
ROUNDS = 2  # ~4% of registrations hit the window unlocked; 96 tries makes a miss rare


class TestReentrancy:
    def test_nested_acquisition_in_one_thread_does_not_deadlock(self, tmp_path: Path) -> None:
        repo = tmp_path / "r"
        repo.mkdir()
        done = threading.Event()

        def nest() -> None:
            with repo_commit_lock(repo), repo_commit_lock(repo), repo_commit_lock(repo):
                done.set()  # would have blocked on its own flock before

        # In a daemon thread with a timeout, so a regression fails in
        # seconds instead of hanging the suite on a flock nobody releases.
        worker = threading.Thread(target=nest, daemon=True)
        worker.start()
        assert done.wait(timeout=10), "nested acquisition deadlocked"

    def test_another_thread_waits_until_the_outermost_release(self, tmp_path: Path) -> None:
        repo = tmp_path / "r"
        repo.mkdir()
        entered = threading.Event()
        released = threading.Event()
        order: list[str] = []

        def holder() -> None:
            with repo_commit_lock(repo), repo_commit_lock(repo):  # nested: one hold
                entered.set()
                released.wait(timeout=10)
                order.append("holder-out")

        def contender() -> None:
            entered.wait(timeout=10)
            with repo_commit_lock(repo):
                order.append("contender-in")

        threads = [threading.Thread(target=holder), threading.Thread(target=contender)]
        for th in threads:
            th.start()
        entered.wait(timeout=10)
        time.sleep(0.2)  # the contender is now blocked on the flock
        assert order == []
        released.set()
        for th in threads:
            th.join(timeout=10)
        assert order == ["holder-out", "contender-in"]


def _register(root: str, tag: str, barrier, out) -> None:  # type: ignore[no-untyped-def]
    from outmem.store import WikiStore

    store = WikiStore.open(Path(root))
    barrier.wait()
    failures: list[str] = []
    for i in range(PER_PROC):
        doc = Path(root).parent / f"{tag}-{i}.md"
        doc.write_text(f"# {tag}-{i}\n\nbody\n", encoding="utf-8")
        try:
            store.add_source(doc)
        except Exception as exc:  # reported, never swallowed
            failures.append(f"{type(exc).__name__}: {exc}")
    out.put(failures)


def _write(root: str, tag: str, barrier, out) -> None:  # type: ignore[no-untyped-def]
    from outmem.store import WikiStore

    store = WikiStore.open(Path(root))
    barrier.wait()
    failures: list[str] = []
    for i in range(PER_PROC):
        try:
            store.write_page(f"{tag}-{i}", title=f"{tag} {i}", body=f"Body {tag} {i}.\n")
        except Exception as exc:  # reported, never swallowed
            failures.append(f"{type(exc).__name__}: {exc}")
    out.put(failures)


def _hammer(root: Path, target) -> list[str]:  # type: ignore[no-untyped-def]
    # spawn, not fork: pytest is multi-threaded by the time this runs.
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(PROCS)
    out = ctx.Queue()
    procs = [
        ctx.Process(target=target, args=(str(root), f"p{k}", barrier, out))
        for k in range(PROCS)
    ]
    for p in procs:
        p.start()
    failures = [f for _ in procs for f in out.get(timeout=300)]
    for p in procs:
        p.join()
    return failures


class TestWritersOfOneWikiAcrossProcesses:
    def test_parallel_registrations_into_one_wiki(self, tmp_path: Path) -> None:
        # The reproduction: three rounds out of three lost at least one
        # registration to "unstable object source data" before the lock
        # covered the registry write.
        for r in range(ROUNDS):
            root = tmp_path / f"w{r}"
            WikiStore.init(root).close()
            assert _hammer(root, _register) == []
            store = WikiStore.open(root, read_only=True)
            assert len(store.list_sources()) == PROCS * PER_PROC

    def test_parallel_page_writes_into_one_wiki(self, tmp_path: Path) -> None:
        # Same shape for the index regeneration: every write rewrites
        # wiki/index.md before committing it.
        for r in range(ROUNDS):
            root = tmp_path / f"w{r}"
            WikiStore.init(root).close()
            assert _hammer(root, _write) == []
            store = WikiStore.open(root, read_only=True)
            assert len(store.list_slugs()) == PROCS * PER_PROC
