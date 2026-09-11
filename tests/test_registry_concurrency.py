"""Several processes opening one registry at the same instant.

`_migrate` decides what to ALTER by reading the schema, and the decision
is only correct if nobody else acts between the read and the ALTER. On a
fresh registry every opener finds the same columns missing, so the
window is open to every one of them at once — sixteen ingest workers
starting together is the ordinary shape of this, not an exotic one.

The interesting failure is loud (`duplicate column name`), but it lands
on whichever opener lost the race, so a caller sees their own open fail
for something another process did.
"""

from __future__ import annotations

import multiprocessing as mp
import sqlite3
from pathlib import Path

import pytest

from outmem.sources import _V2_COLUMNS, SCHEMA_VERSION, SourceRegistry

OPENERS = 16
ROUNDS = 6


def _open_when_released(sources_dir: str, barrier, out) -> None:  # type: ignore[no-untyped-def]
    """Child process: wait for every sibling, then open the registry."""
    barrier.wait()
    try:
        SourceRegistry.load(Path(sources_dir)).close()
    except Exception as exc:  # reported, never swallowed
        out.put(f"{type(exc).__name__}: {exc}")
    else:
        out.put("ok")


def _hammer(sources_dir: Path) -> list[str]:
    # spawn, not fork: pytest is multi-threaded by the time this runs, and a
    # forked child of a threaded parent is the deadlock the interpreter
    # warns about. The extra second is cheaper than a hang.
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(OPENERS)
    out = ctx.Queue()
    procs = [
        ctx.Process(target=_open_when_released, args=(str(sources_dir), barrier, out))
        for _ in range(OPENERS)
    ]
    for p in procs:
        p.start()
    results = [out.get(timeout=60) for _ in procs]
    for p in procs:
        p.join()
    return [r for r in results if r != "ok"]


class TestConcurrentOpen:
    def test_no_opener_fails_on_a_fresh_registry(self, tmp_path: Path) -> None:
        failures: list[str] = []
        for i in range(ROUNDS):
            sources = tmp_path / f"round{i}" / "sources"
            sources.mkdir(parents=True)
            failures.extend(_hammer(sources))
        assert failures == []

    def test_the_registry_ends_up_migrated_exactly_once(self, tmp_path: Path) -> None:
        sources = tmp_path / "sources"
        sources.mkdir()
        assert _hammer(sources) == []
        con = sqlite3.connect(str(sources / ".sources.db"))
        columns = [row[1] for row in con.execute("PRAGMA table_info(sources)")]
        # Every v2 column present once — a duplicate would have failed
        # the ALTER, but the assertion is on the state, not the absence
        # of a message.
        for column in _V2_COLUMNS:
            assert columns.count(column) == 1
        assert con.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION


class TestTheDecisionIsMadeUnderTheLock:
    def test_an_opener_that_arrives_mid_migration_waits_then_finds_nothing_to_do(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The deterministic form of the race. A holds the write lock while
        # it adds the columns; B opens meanwhile. B's decision about what
        # to ALTER must be made after it gets the lock — i.e. after A has
        # committed — or B acts on the pre-A schema and hits "duplicate
        # column" the moment A releases.
        import threading
        import time

        from outmem import sources as mod

        sources = tmp_path / "sources"
        sources.mkdir()
        # All tables present, no v2 columns: the shape of a v1 registry,
        # so B's CREATE TABLE IF NOT EXISTS statements have nothing to
        # write and B reaches `_migrate` without blocking earlier.
        monkeypatch.setattr(mod, "_migrate", lambda con: None)
        mod._open_registry(sources / ".sources.db").close()
        monkeypatch.undo()

        holder = sqlite3.connect(str(sources / ".sources.db"))
        holder.execute("BEGIN IMMEDIATE")  # A: holds the write lock
        outcome: list[str] = []

        def b_opens() -> None:
            try:
                SourceRegistry.load(sources).close()
            except Exception as exc:
                outcome.append(f"{type(exc).__name__}: {exc}")
            else:
                outcome.append("ok")

        thread = threading.Thread(target=b_opens)
        thread.start()
        time.sleep(0.5)  # B has looked, and is now waiting on the lock
        for column in _V2_COLUMNS:  # A migrates and releases
            holder.execute(f"ALTER TABLE sources ADD COLUMN {column} TEXT")
        holder.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        holder.commit()
        thread.join(timeout=30)
        assert outcome == ["ok"]

    def test_a_v1_registry_still_gets_its_columns(self, tmp_path: Path) -> None:
        # The lock must not turn the migration into a no-op: a genuinely
        # old registry (v1 shape, no v2 columns) is still upgraded.
        sources = tmp_path / "sources"
        sources.mkdir()
        con = sqlite3.connect(str(sources / ".sources.db"))
        con.execute(
            "CREATE TABLE sources (rel_path TEXT PRIMARY KEY, sha256 TEXT NOT NULL,"
            " size_bytes INTEGER NOT NULL, registered_at TEXT NOT NULL)"
        )
        con.execute("PRAGMA user_version = 1")
        con.commit()
        con.close()

        SourceRegistry.load(sources).close()

        con = sqlite3.connect(str(sources / ".sources.db"))
        columns = {row[1] for row in con.execute("PRAGMA table_info(sources)")}
        assert set(_V2_COLUMNS) <= columns
        assert con.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
