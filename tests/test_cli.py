"""Smoke tests for the ``outmem`` CLI.

The CLI is a thin shell over :class:`outmem.store.WikiStore`; these
tests verify the argv → store wiring rather than re-testing the store
itself.
"""

from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path

import pytest

from outmem.cli.__main__ import main
from outmem.store import WikiStore


@pytest.fixture
def cli_root(tmp_path: Path) -> Path:
    """A fresh wiki initialised via the CLI's own init code path."""
    root = tmp_path / "w"
    assert main(["init", str(root)]) == 0
    return root


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "outmem" in captured.out


def test_init_creates_layout(tmp_path: Path) -> None:
    root = tmp_path / "fresh"
    rc = main(["init", str(root)])
    assert rc == 0
    assert (root / "wiki").is_dir()
    assert (root / "CONTRIBUTORS.md").exists()


def test_write_then_read(
    cli_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO("alpha body content\n"))
    rc = main(["--root", str(cli_root), "write", "alpha", "--title", "Alpha"])
    assert rc == 0
    sha = capsys.readouterr().out.strip()
    assert len(sha) == 40

    rc = main(["--root", str(cli_root), "read", "alpha"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "title: Alpha" in out
    assert "alpha body content" in out


def test_read_body_only(
    cli_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO("just the body\n"))
    main(["--root", str(cli_root), "write", "alpha", "--title", "Alpha"])
    capsys.readouterr()

    rc = main(["--root", str(cli_root), "read", "alpha", "--body-only"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "title" not in out
    assert "just the body" in out


def test_search_returns_matches(
    cli_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO("the unique-token-xyz lives here\n"))
    main(["--root", str(cli_root), "write", "alpha", "--title", "Alpha"])
    capsys.readouterr()

    rc = main(["--root", str(cli_root), "search", "unique-token-xyz"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "alpha.md" in out
    assert "unique-token-xyz" in out


def test_search_no_match_returns_1(
    cli_root: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = main(["--root", str(cli_root), "search", "nothing-here-token"])
    assert rc == 1


def test_write_empty_body_rejected(
    cli_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO("   \n"))
    rc = main(["--root", str(cli_root), "write", "alpha", "--title", "X"])
    assert rc == 2
    assert "empty body" in capsys.readouterr().err


def test_extend_after_write(
    cli_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO("v1\n"))
    main(["--root", str(cli_root), "write", "alpha", "--title", "Alpha"])
    capsys.readouterr()

    monkeypatch.setattr("sys.stdin", io.StringIO("v2 different\n"))
    rc = main(["--root", str(cli_root), "extend", "alpha"])
    assert rc == 0


def test_log_appends_entry(
    cli_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO("- saw a thing\n"))
    rc = main(["--root", str(cli_root), "log", "pricing"])
    assert rc == 0
    log_files = list((cli_root / "log").glob("*.md"))
    assert len(log_files) == 1
    assert "saw a thing" in log_files[0].read_text()


def test_history_after_two_commits(
    cli_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO("v1\n"))
    main(["--root", str(cli_root), "write", "alpha", "--title", "Alpha"])
    monkeypatch.setattr("sys.stdin", io.StringIO("v2\n"))
    main(["--root", str(cli_root), "extend", "alpha"])
    capsys.readouterr()

    rc = main(["--root", str(cli_root), "history", "alpha"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "extend: alpha" in out
    assert "compact: alpha" in out


def test_steering_excludes_agent(populated_repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["--root", str(populated_repo), "steering"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "agent@host" not in out
    assert "alice@example.com" in out


def test_record_run(
    cli_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO("body\n"))
    main(["--root", str(cli_root), "write", "alpha", "--title", "A"])
    capsys.readouterr()

    rc = main(["--root", str(cli_root), "record-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "recorded run" in out


def test_outmem_path_env_var(
    cli_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OUTMEM_PATH", str(cli_root))
    monkeypatch.setattr("sys.stdin", io.StringIO("body from env-path\n"))
    rc = main(["write", "from-env", "--title", "FromEnv"])
    assert rc == 0
    page = WikiStore.open(cli_root).read("from-env")
    assert "body from env-path" in page.body


def test_outmem_agent_identity_env_var(
    cli_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OUTMEM_AGENT_NAME", "CustomAgent")
    monkeypatch.setenv("OUTMEM_AGENT_EMAIL", "custom@elsewhere")
    monkeypatch.setattr("sys.stdin", io.StringIO("body\n"))
    rc = main(["--root", str(cli_root), "write", "alpha", "--title", "A"])
    assert rc == 0
    log = subprocess.run(
        ["git", "log", "-1", "--pretty=format:%an <%ae>"],
        cwd=str(cli_root),
        check=True,
        capture_output=True,
        text=True,
    )
    assert log.stdout.strip() == "CustomAgent <custom@elsewhere>"


def test_console_script_entry_point_loads() -> None:
    """The installed `outmem` shell command should exist on PATH after install."""
    rc = subprocess.run(
        [sys.executable, "-m", "outmem.cli", "--version"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert rc.returncode == 0
    assert "outmem" in rc.stdout
