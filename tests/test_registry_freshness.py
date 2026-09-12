"""A long-lived store's registry snapshot keeps itself current.

A store caches its registry for its lifetime — right for reads, but it
meant a read-only store answered ``list_sources`` from the snapshot taken
at open, and a server had to close and reopen its read stores after every
registration to avoid serving a stale listing. SQLite's
``PRAGMA data_version`` changes exactly when another connection has
committed, so the snapshot now asks before it answers.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from outmem.store import WikiStore


def _doc(tmp_path: Path, name: str = "doc.md") -> Path:
    path = tmp_path / name
    path.write_text(f"# {name}\n\nbody of {name}\n", encoding="utf-8")
    return path


class TestAnotherHandleInTheSameProcess:
    def test_list_sources_sees_the_new_row(self, tmp_path: Path) -> None:
        root = tmp_path / "w"
        WikiStore.init(root).close()
        reader = WikiStore.open(root, read_only=True)
        writer = WikiStore.open(root)
        first = writer.add_source(_doc(tmp_path, "first.md"))
        assert [e.rel_path for e in reader.list_sources()] == [first.rel_path]  # warm

        second = writer.add_source(_doc(tmp_path, "second.md"))

        assert sorted(e.rel_path for e in reader.list_sources()) == sorted(
            [first.rel_path, second.rel_path]
        )

    def test_get_source_resolves_it_too(self, tmp_path: Path) -> None:
        root = tmp_path / "w"
        WikiStore.init(root).close()
        reader = WikiStore.open(root, read_only=True)
        writer = WikiStore.open(root)
        writer.add_source(_doc(tmp_path, "first.md"))
        reader.list_sources()  # take the snapshot

        entry = writer.add_source(_doc(tmp_path, "second.md"))

        assert reader.get_source(entry.citation_path) is not None

    def test_a_registry_that_did_not_exist_at_open_is_picked_up(
        self, tmp_path: Path
    ) -> None:
        # The read-only store is handed a no-database stand-in when
        # `.sources.db` is absent at open — a curator-shipped clone with no
        # sources yet. That stand-in has no connection to ask, so the
        # upgrade is a `stat`, made only in this case.
        root = tmp_path / "w"
        WikiStore.init(root).close()
        reader = WikiStore.open(root, read_only=True)
        assert reader.list_sources() == []
        assert not (reader.sources_path / ".sources.db").exists()

        entry = WikiStore.open(root).add_source(_doc(tmp_path))

        assert [e.rel_path for e in reader.list_sources()] == [entry.rel_path]

    def test_the_stand_in_creates_nothing_while_waiting(self, tmp_path: Path) -> None:
        root = tmp_path / "w"
        WikiStore.init(root).close()
        reader = WikiStore.open(root, read_only=True)
        for _ in range(3):
            reader.list_sources()
            reader.get_source("nope/x.md")
        assert not (reader.sources_path / ".sources.db").exists()


class TestAnotherProcess:
    def test_a_registration_in_another_process_is_visible(self, tmp_path: Path) -> None:
        root = tmp_path / "w"
        WikiStore.init(root).close()
        WikiStore.open(root).add_source(_doc(tmp_path, "first.md"))
        reader = WikiStore.open(root, read_only=True)
        assert len(reader.list_sources()) == 1  # snapshot taken

        doc = _doc(tmp_path, "second.md")
        subprocess.run(
            [
                sys.executable, "-c",
                "import sys; from outmem.store import WikiStore; "
                "WikiStore.open(sys.argv[1]).add_source(sys.argv[2])",
                str(root), str(doc),
            ],
            check=True,
            capture_output=True,
        )

        assert len(reader.list_sources()) == 2


class TestOwnWritesStayInLockstep:
    def test_a_store_sees_its_own_registration_without_a_reread(
        self, tmp_path: Path
    ) -> None:
        # Own commits leave data_version alone; the mutation keeps the
        # snapshot current itself. This pins that the property does not
        # break the existing contract.
        root = tmp_path / "w"
        store = WikiStore.init(root)
        entry = store.add_source(_doc(tmp_path))
        assert store.get_source(entry.rel_path) is not None
        assert [e.rel_path for e in store.list_sources()] == [entry.rel_path]

    def test_gc_through_its_own_connection_is_reflected(self, tmp_path: Path) -> None:
        # `gc_registry` opens the database on a connection of its own, so
        # the store's cached handle used to be dropped by hand afterwards.
        # The snapshot notices the foreign commit on its own now.
        root = tmp_path / "w"
        store = WikiStore.init(root)
        entry = store.add_source(_doc(tmp_path))
        (store.sources_path / entry.rel_path).unlink()
        assert store.list_sources(include_missing=True)

        store.sources_gc(dry_run=False)

        assert store.list_sources(include_missing=True) == []


class TestUnderThreads:
    def test_concurrent_registrations_and_reads_on_one_store(self, tmp_path: Path) -> None:
        # The registry's connection is shared across threads (PydanticAI
        # dispatches tool calls that way). A refresh must not swap the
        # snapshot out from under a mutation's lockstep update — which is
        # what the registry lock is for — and nothing may raise. The
        # narrow interleaving itself is not something a test can force;
        # this pins that the mechanism holds up under real contention.
        import threading

        root = tmp_path / "w"
        store = WikiStore.init(root)
        reader = WikiStore.open(root, read_only=True)
        errors: list[BaseException] = []
        stop = threading.Event()

        def register(worker: int) -> None:
            try:
                for i in range(5):
                    store.add_source(_doc(tmp_path, f"w{worker}-{i}.md"))
            except BaseException as exc:
                errors.append(exc)

        def read() -> None:
            try:
                while not stop.is_set():
                    reader.list_sources()
                    store.list_sources()
            except BaseException as exc:
                errors.append(exc)

        readers = [threading.Thread(target=read) for _ in range(2)]
        writers = [threading.Thread(target=register, args=(w,)) for w in range(8)]
        for th in readers + writers:
            th.start()
        for th in writers:
            th.join(timeout=120)
        stop.set()
        for th in readers:
            th.join(timeout=30)

        assert errors == []
        assert len(store.list_sources()) == 40
        assert len(reader.list_sources()) == 40
