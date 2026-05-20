"""Tests for ``outmem._logfire`` + the public ``setup_logfire`` façade.

The Logfire dep is intentionally optional; the no-op-when-disabled path
must work without it installed. The public façade lives in
:mod:`outmem.observability` and is auto-called by ``ask()`` and
``build_consult_wiki`` so library users get the same instrumentation as
the CLI.
"""

from __future__ import annotations

import contextlib
import importlib
import sys
from pathlib import Path

import pytest

import outmem
from outmem import WikiStore, setup_logfire
from outmem._logfire import setup
from outmem.config import LogfireSettings
from outmem.exceptions import OutmemError


def test_setup_noop_when_project_unset() -> None:
    assert setup(LogfireSettings(project=None)) is False


def test_setup_raises_friendly_error_when_dep_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If project is set but the logfire package isn't importable, point
    the user at the install command rather than a bare ImportError."""
    import outmem._logfire as logfire_setup

    monkeypatch.setattr(logfire_setup, "_configured", False)
    monkeypatch.setitem(sys.modules, "logfire", None)
    # Importlib refresh — make sure subsequent `import logfire` resolves
    # to the None sentinel above.
    importlib.invalidate_caches()

    with pytest.raises(OutmemError, match="outmem\\[logfire\\]"):
        setup(LogfireSettings(project="my-project"))


# ---------------------------------------------------------------------------
# Public façade — `outmem.setup_logfire`
# ---------------------------------------------------------------------------


def test_setup_logfire_is_exported() -> None:
    """The helper has to be top-level discoverable — that's the whole
    point of having a public façade over ``_logfire.setup``."""
    assert outmem.setup_logfire is setup_logfire
    assert "setup_logfire" in outmem.__all__


def test_setup_logfire_accepts_store(tmp_path: Path) -> None:
    store = WikiStore.init(tmp_path / "w")
    # Default config has project=None → no-op, returns False.
    assert setup_logfire(store) is False


def test_setup_logfire_accepts_settings() -> None:
    assert setup_logfire(LogfireSettings(project=None)) is False


def test_ask_invokes_setup_logfire(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``ask()`` must call ``_logfire.setup(store.config.outmem.logfire)``
    so library callers get the same auto-config the CLI does. We assert
    on the call, not on Logfire's internal state — the helper itself is
    tested separately."""
    calls: list[LogfireSettings] = []

    def fake_setup(settings: LogfireSettings) -> bool:
        calls.append(settings)
        return False

    monkeypatch.setattr("outmem._logfire.setup", fake_setup)

    store = WikiStore.init(tmp_path / "w")
    # Inject a non-default project so we can verify the right settings
    # object is forwarded.
    store.config.outmem.logfire.project = "test-project"

    # We don't need a real run — even an early error would prove the
    # setup call happened first. Easier: import ask, call it with a
    # TestModel and assert calls.
    from pydantic_ai.models.test import TestModel

    from outmem.agent import ask_sync

    # WritebackError on TestModel runs (no tools called → no commit) is
    # fine here; we only care that setup was invoked first.
    with contextlib.suppress(Exception):
        ask_sync(store, query="hi", model=TestModel(call_tools=[]), push=False, pull=False)

    assert any(s.project == "test-project" for s in calls)


def test_build_consult_wiki_invokes_setup_logfire(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same contract for the read-only factory."""
    calls: list[LogfireSettings] = []

    def fake_setup(settings: LogfireSettings) -> bool:
        calls.append(settings)
        return False

    monkeypatch.setattr("outmem._logfire.setup", fake_setup)

    seed = WikiStore.init(tmp_path / "w")
    seed.write_page("p", title="T", body="b")
    # Edit config.yaml so the re-opened (read-only) store picks the project up.
    (seed.root / "config.yaml").write_text(
        "logfire:\n  project: ro-test-project\n",
        encoding="utf-8",
    )
    seed.close()

    from pydantic_ai.models.test import TestModel

    from outmem.adapters.pydantic_ai import build_consult_wiki

    build_consult_wiki(seed.root, model=TestModel())
    assert any(s.project == "ro-test-project" for s in calls)
