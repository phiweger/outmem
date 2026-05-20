"""Named history queries — convergent reading of git log for the wiki.

These wrappers translate slug-level concerns ("how has the pricing
formula evolved?") into the path-level calls :mod:`outmem.git_ops`
makes against ``git log``. Two functions land in v0.1:

- :func:`page_history` — chronological list of commits touching a single
  wiki page. Backs the dashboard's per-page history view (spec v0.5 §5).
- :func:`topic_evolution` — the EXPANSION-branch helper from the
  planning prompt phase 2. Returns the raw ``git log -p --follow``
  stream so the agent can read the diff sequence as-is.

Both treat ``wiki/<slug>.md`` paths uniformly and let the caller
supplement with additional paths (typically ``log/`` for evolution).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from outmem.git_ops import CommitInfo, log_for_paths, log_since
from outmem.slug import validate_slug


def page_history(repo_path: Path, slug: str, *, wiki_dir: str = "wiki") -> list[CommitInfo]:
    """List every commit that touched ``wiki/<slug>.md``, newest first.

    Uses ``--follow`` so renames are tracked. The slug is validated for
    safety before becoming a path component.
    """
    validate_slug(slug)
    return log_since(repo_path, paths=[f"{wiki_dir}/{slug}.md"])


def topic_evolution(
    repo_path: Path,
    slugs: Sequence[str],
    *,
    wiki_dir: str = "wiki",
    include_log: bool = True,
    log_dir: str = "log",
) -> str:
    """Return the chronological diff stream across the given wiki pages.

    This is the EXPANSION branch of the planning prompt: the agent reads
    this output to understand how thinking on a topic has shifted over
    time, rather than just retrieving the current state.

    With ``include_log=True`` (the default) the ``log/`` directory is
    appended so decisions and observations recorded there join the
    timeline. Set ``include_log=False`` for a tighter "just the wiki
    page" view.
    """
    if not slugs:
        raise ValueError("topic_evolution: at least one slug is required.")
    paths: list[str] = []
    for slug in slugs:
        validate_slug(slug)
        paths.append(f"{wiki_dir}/{slug}.md")
    if include_log:
        paths.append(f"{log_dir}/")

    # ``git log --follow`` only accepts a single pathspec. When the caller
    # asks for evolution across multiple paths we drop ``--follow`` (the
    # combined log still answers "how has thinking changed" — it just
    # doesn't trace renames). Single-slug calls keep the rename tracking.
    use_follow = len(paths) == 1
    return log_for_paths(repo_path, paths, follow=use_follow, with_patch=True)
