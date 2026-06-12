"""Tests for `outmem.hooks` and the WikiStore auto-ensure wiring."""

from __future__ import annotations

from pathlib import Path

import pytest

from outmem.hooks import (
    HOOK_MARKER,
    HOOK_NAME,
    ensure_hook,
    install_hook,
    uninstall_hook,
)
from outmem.store import WikiStore


class TestInstallHook:
    def test_installs_when_absent(self, tmp_path: Path) -> None:
        (tmp_path / ".git" / "hooks").mkdir(parents=True)
        assert install_hook(tmp_path) == "installed"
        hook = tmp_path / ".git" / "hooks" / HOOK_NAME
        assert hook.exists()
        assert HOOK_MARKER in hook.read_text()
        assert hook.stat().st_mode & 0o111  # executable

    def test_unchanged_when_current(self, tmp_path: Path) -> None:
        (tmp_path / ".git" / "hooks").mkdir(parents=True)
        install_hook(tmp_path)
        assert install_hook(tmp_path) == "unchanged"  # idempotent

    def test_refreshes_our_stale_hook(self, tmp_path: Path) -> None:
        hooks = tmp_path / ".git" / "hooks"
        hooks.mkdir(parents=True)
        # Our marker but old body → refreshed (not "foreign").
        (hooks / HOOK_NAME).write_text(f"#!/bin/sh\n{HOOK_MARKER}\n# old\nexit 0\n")
        assert install_hook(tmp_path) == "installed"

    def test_refuses_foreign_hook(self, tmp_path: Path) -> None:
        hooks = tmp_path / ".git" / "hooks"
        hooks.mkdir(parents=True)
        foreign = hooks / HOOK_NAME
        foreign.write_text("#!/bin/sh\necho mine\n")
        assert install_hook(tmp_path) == "foreign"
        assert foreign.read_text() == "#!/bin/sh\necho mine\n"  # untouched
        # force overwrites
        assert install_hook(tmp_path, force=True) == "installed"

    def test_no_git_dir(self, tmp_path: Path) -> None:
        assert install_hook(tmp_path) == "no-git"


class TestUninstallHook:
    def test_removes_ours(self, tmp_path: Path) -> None:
        (tmp_path / ".git" / "hooks").mkdir(parents=True)
        install_hook(tmp_path)
        assert uninstall_hook(tmp_path) == "removed"
        assert not (tmp_path / ".git" / "hooks" / HOOK_NAME).exists()

    def test_absent(self, tmp_path: Path) -> None:
        (tmp_path / ".git" / "hooks").mkdir(parents=True)
        assert uninstall_hook(tmp_path) == "absent"

    def test_keeps_foreign(self, tmp_path: Path) -> None:
        hooks = tmp_path / ".git" / "hooks"
        hooks.mkdir(parents=True)
        (hooks / HOOK_NAME).write_text("#!/bin/sh\necho mine\n")
        assert uninstall_hook(tmp_path) == "foreign"
        assert (hooks / HOOK_NAME).exists()


class TestEnsureHook:
    def test_never_raises_without_git(self, tmp_path: Path) -> None:
        ensure_hook(tmp_path)  # no .git → silent no-op, must not raise

    def test_installs_idempotently(self, tmp_path: Path) -> None:
        (tmp_path / ".git" / "hooks").mkdir(parents=True)
        ensure_hook(tmp_path)
        ensure_hook(tmp_path)  # second call no-op
        assert (tmp_path / ".git" / "hooks" / HOOK_NAME).exists()

    def test_leaves_foreign_alone(self, tmp_path: Path) -> None:
        hooks = tmp_path / ".git" / "hooks"
        hooks.mkdir(parents=True)
        (hooks / HOOK_NAME).write_text("#!/bin/sh\necho mine\n")
        ensure_hook(tmp_path)
        assert (hooks / HOOK_NAME).read_text() == "#!/bin/sh\necho mine\n"


class TestAutoInstallWiring:
    """WikiStore.init/open auto-ensure the hook (the conftest autouse fixture
    that no-ops ensure_hook is overridden here with a spy)."""

    def test_init_calls_ensure_hook(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[Path] = []
        monkeypatch.setattr("outmem.store.ensure_hook", lambda root: calls.append(root))
        store = WikiStore.init(tmp_path / "w")
        assert calls == [store.root]

    def test_open_calls_ensure_hook(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = WikiStore.init(tmp_path / "w")  # autouse-noop still active here
        calls: list[Path] = []
        monkeypatch.setattr("outmem.store.ensure_hook", lambda root: calls.append(root))
        WikiStore.open(store.root)
        assert calls == [store.root]

    def test_opt_out_via_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = tmp_path / "w"
        root.mkdir()
        (root / "config.yaml").write_text(
            "git:\n  auto_install_hook: false\n", encoding="utf-8"
        )
        store = WikiStore.init(root)
        calls: list[Path] = []
        monkeypatch.setattr("outmem.store.ensure_hook", lambda root: calls.append(root))
        WikiStore.open(store.root)
        assert calls == []  # disabled → never called

    def test_read_only_skips(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = WikiStore.init(tmp_path / "w")
        calls: list[Path] = []
        monkeypatch.setattr("outmem.store.ensure_hook", lambda root: calls.append(root))
        WikiStore.open(store.root, read_only=True)
        assert calls == []  # read-only must not mutate the repo
