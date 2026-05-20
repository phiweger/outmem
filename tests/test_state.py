"""Tests for ``outmem.state``."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from outmem.state import (
    BACKLINKS_FILE,
    LAST_RUN_FILE,
    STATE_DIR_NAME,
    OutmemState,
)


def test_ensure_creates_dir_and_gitignore(tmp_path: Path) -> None:
    state = OutmemState(tmp_path)
    state.ensure()
    assert (tmp_path / STATE_DIR_NAME).is_dir()
    gitignore = tmp_path / STATE_DIR_NAME / ".gitignore"
    assert gitignore.exists()
    body = gitignore.read_text()
    assert "*" in body
    assert "!.gitignore" in body


def test_ensure_is_idempotent(tmp_path: Path) -> None:
    state = OutmemState(tmp_path)
    state.ensure()
    gitignore = tmp_path / STATE_DIR_NAME / ".gitignore"
    gitignore.write_text("custom\n")
    state.ensure()  # should not clobber
    assert gitignore.read_text() == "custom\n"


def test_read_json_missing_returns_none(tmp_path: Path) -> None:
    state = OutmemState(tmp_path)
    assert state.read_json("nonexistent.json") is None


def test_read_json_malformed_returns_none(tmp_path: Path) -> None:
    state = OutmemState(tmp_path)
    state.ensure()
    (tmp_path / STATE_DIR_NAME / "x.json").write_text("not json")
    assert state.read_json("x.json") is None


def test_write_json_roundtrips(tmp_path: Path) -> None:
    state = OutmemState(tmp_path)
    payload = {"a": 1, "nested": {"b": "two"}, "list": [1, 2, 3]}
    state.write_json("data.json", payload)
    assert state.read_json("data.json") == payload


def test_write_json_is_atomic_on_disk(tmp_path: Path) -> None:
    state = OutmemState(tmp_path)
    state.write_json("x.json", {"v": 1})
    files = list((tmp_path / STATE_DIR_NAME).iterdir())
    # The temp file should have been renamed away.
    suffixes = {p.suffix for p in files}
    assert ".tmp" not in suffixes


def test_last_run_missing_returns_none(tmp_path: Path) -> None:
    state = OutmemState(tmp_path)
    assert state.last_run() is None


def test_record_run_then_read(tmp_path: Path) -> None:
    state = OutmemState(tmp_path)
    when = datetime(2026, 5, 11, 12, 0, 0, tzinfo=UTC)
    state.record_run(head="abc1234", timestamp=when)
    marker = state.last_run()
    assert marker is not None
    assert marker.timestamp == when
    assert marker.head == "abc1234"


def test_record_run_persists_z_suffix(tmp_path: Path) -> None:
    state = OutmemState(tmp_path)
    state.record_run(head="x", timestamp=datetime(2026, 5, 11, 12, 0, 0, tzinfo=UTC))
    raw = (tmp_path / STATE_DIR_NAME / LAST_RUN_FILE).read_text()
    assert "2026-05-11T12:00:00Z" in raw


def test_record_run_naive_datetime_assumed_utc(tmp_path: Path) -> None:
    state = OutmemState(tmp_path)
    marker = state.record_run(head=None, timestamp=datetime(2026, 6, 1, 8, 0, 0))
    assert marker.timestamp.tzinfo == UTC


def test_record_run_default_timestamp_uses_now(tmp_path: Path) -> None:
    state = OutmemState(tmp_path)
    before = datetime.now(UTC).replace(microsecond=0)
    marker = state.record_run(head="head")
    after = datetime.now(UTC).replace(microsecond=0)
    assert before <= marker.timestamp <= after


def test_state_files_are_predictable_names() -> None:
    # These are public constants — pin them so consumers can shell into
    # the wiki and inspect manually.
    assert STATE_DIR_NAME == ".outmem"
    assert LAST_RUN_FILE == "last_run.json"
    assert BACKLINKS_FILE == "backlinks.json"
