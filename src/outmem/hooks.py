"""Git pre-commit hook management.

The hook keeps a manually edited wiki self-consistent at commit time: it
repairs unparseable frontmatter, regenerates ``wiki/index.md``, and
updates the semantic vector DB — re-staging each into the commit. See
``_cmd_reindex_staged`` in :mod:`outmem.cli.__main__` for what runs.

Lives in its own module (not the CLI) so :class:`outmem.store.WikiStore`
can *auto-ensure* the hook on ``init`` / ``open`` without importing the
CLI. The explicit ``outmem hook install`` / ``uninstall`` commands and
the automatic path share this code, so they can't drift.

Git note: ``.git/hooks`` is per-clone and never committed/pushed (a
security property — cloning a repo must not run code). So a hook can't
"come with" the repo; every clone needs a one-time activation. The auto-
ensure path is outmem performing that activation for you as a side effect
of normal use — still per-clone, but no explicit command to remember.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

HOOK_NAME = "pre-commit"
HOOK_MARKER = "# outmem pre-commit hook"
HOOK_SCRIPT = f"""#!/bin/sh
{HOOK_MARKER}
# For externally edited wiki pages (Obsidian, etc.): repairs unparseable
# frontmatter, keeps wiki/index.md current, and keeps the semantic vector
# DB in lockstep — re-staging each into the commit. Installed by
# `outmem hook install` (or auto-ensured on store open). Safe to remove:
# `outmem hook uninstall`.
set -e
exec outmem reindex --staged
"""


def install_hook(root: Path, *, force: bool = False) -> str:
    """Install the pre-commit hook into ``<root>/.git/hooks``.

    Returns a status token describing what happened:

    * ``"installed"`` — written fresh, or refreshed our own existing hook.
    * ``"unchanged"`` — our hook was already present and current.
    * ``"foreign"`` — a non-outmem hook is in the way (kept; pass
      ``force=True`` to overwrite).
    * ``"no-git"`` — ``.git/hooks`` doesn't exist (not a git repo).

    Idempotent: re-running when our hook is current is a no-op.
    """
    hooks_dir = root / ".git" / "hooks"
    if not hooks_dir.is_dir():
        return "no-git"
    target = hooks_dir / HOOK_NAME
    if target.exists():
        existing = target.read_text(encoding="utf-8", errors="replace")
        if HOOK_MARKER not in existing and not force:
            return "foreign"
        if HOOK_MARKER in existing and existing == HOOK_SCRIPT:
            return "unchanged"
    target.write_text(HOOK_SCRIPT, encoding="utf-8")
    target.chmod(0o755)
    return "installed"


def uninstall_hook(root: Path, *, force: bool = False) -> str:
    """Remove the pre-commit hook. Returns ``"removed"``, ``"absent"``, or
    ``"foreign"`` (a non-outmem hook left in place unless ``force``)."""
    target = root / ".git" / "hooks" / HOOK_NAME
    if not target.exists():
        return "absent"
    existing = target.read_text(encoding="utf-8", errors="replace")
    if HOOK_MARKER not in existing and not force:
        return "foreign"
    target.unlink()
    return "removed"


def ensure_hook(root: Path) -> None:
    """Best-effort idempotent install for the auto-ensure path.

    Installs our hook when absent or stale, but never clobbers a hook the
    user wrote themselves, and never raises — a missing ``.git/hooks``, a
    read-only filesystem, or a permissions error must not break opening a
    store. A foreign hook is logged at debug and left untouched.
    """
    try:
        status = install_hook(root, force=False)
    except OSError as exc:  # read-only FS, perms, … — never fatal on open
        log.debug("auto-ensure pre-commit hook skipped: %s", exc)
        return
    if status == "foreign":
        log.debug(
            "a non-outmem pre-commit hook is present at %s; leaving it "
            "(run `outmem hook install --force` to replace)", root,
        )
    elif status == "installed":
        log.debug("auto-installed pre-commit hook at %s", root)
