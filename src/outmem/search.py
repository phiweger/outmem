"""Ripgrep-backed search over a wiki / log / raw directory.

The agent's primary retrieval path. ``rg --json`` emits one JSON event
per line; we parse the "match" events into :class:`SearchHit` records.
Output is hard-capped at a configurable byte budget (default 8 KiB) so
a broad query against a large directory does not blow up the agent's
context window on the third call (FAIL.md anti-pattern: unbounded
tool results).

Path arguments are validated against the repo root before invoking
``rg`` — symlinks, ``..`` segments, and absolute paths outside the
repo are rejected.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from outmem.exceptions import OutmemError

DEFAULT_RESULT_BYTES = 8 * 1024  # 8 KiB token-cap soft ceiling.


@dataclass(frozen=True)
class SearchHit:
    """A single ripgrep match — one file, one line."""

    path: str  # relative to the search root
    line_number: int
    text: str


@dataclass(frozen=True)
class SearchResult:
    """The combined output of a :func:`search` call.

    ``truncated`` is True when the result was clipped to ``max_bytes``;
    the caller should narrow the pattern or paginate before requesting
    more.
    """

    hits: tuple[SearchHit, ...]
    truncated: bool


def rg_available() -> bool:
    """Return True iff a ``rg`` executable is on PATH."""
    return shutil.which("rg") is not None


def search(
    pattern: str,
    *,
    root: Path,
    paths: Sequence[str | Path] | None = None,
    case_insensitive: bool = False,
    fixed_strings: bool = False,
    max_bytes: int = DEFAULT_RESULT_BYTES,
    max_hits: int | None = None,
    extra_args: Sequence[str] = (),
) -> SearchResult:
    """Run ``rg --json`` over ``root`` and return parsed hits.

    Args:
        pattern: The pattern to search for. Treated as a regex unless
            ``fixed_strings=True``.
        root: The directory to anchor the search at. All ``paths`` are
            resolved relative to it and confined within it.
        paths: Optional list of subdirectories or files (relative to
            ``root``) to restrict the search. ``None`` means search
            everything under ``root``.
        case_insensitive: ``rg -i``.
        fixed_strings: ``rg -F`` — treat the pattern as a literal string.
        max_bytes: Soft ceiling on the bytes of ``rg`` output we consume.
            Exceeding it sets ``SearchResult.truncated``.
        max_hits: Optional hard cap on the number of returned hits.
        extra_args: Additional ``rg`` flags appended verbatim. Use with
            care — anything that changes the JSON shape will break parsing.

    Raises:
        OutmemError: If ripgrep is not installed or a path escapes
            ``root``.
    """
    if not rg_available():
        raise OutmemError("ripgrep (`rg`) is not on PATH — install it to enable search.")

    root = root.resolve()
    if not root.is_dir():
        raise OutmemError(f"Search root does not exist or is not a directory: {root}")

    resolved_paths = _resolve_search_paths(root, paths)

    # --sort path forces a stable file order across runs. Without it
    # ripgrep parallelises the walk and returns hits in thread-scheduling
    # order — so identical inputs yielded different rankings across calls,
    # which broke optimizer score reproducibility for lexical/hybrid.
    args = ["rg", "--json", "--sort", "path"]
    if case_insensitive:
        args.append("-i")
    if fixed_strings:
        args.append("-F")
    args.extend(extra_args)
    args.append("--")
    args.append(pattern)
    args.extend(str(p) for p in resolved_paths)

    env = os.environ.copy()
    env.setdefault("RIPGREP_CONFIG_PATH", "")  # ignore the user's ~/.ripgreprc.

    try:
        result = subprocess.run(
            args,
            cwd=str(root),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise OutmemError(f"rg invocation failed: {exc}") from exc

    # rg exits 1 when there are no matches — that is a legitimate result,
    # not an error. Anything else (2+) means a real failure.
    if result.returncode > 1:
        raise OutmemError(
            f"rg failed (exit {result.returncode}): "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )

    return _parse_rg_json(
        result.stdout,
        root=root,
        max_bytes=max_bytes,
        max_hits=max_hits,
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _resolve_search_paths(
    root: Path,
    paths: Sequence[str | Path] | None,
) -> list[Path]:
    """Resolve and confine each path to ``root``.

    Symlinks are followed during resolution; the result must still live
    under ``root`` or we refuse to search there.
    """
    if not paths:
        return [root]
    resolved: list[Path] = []
    for raw in paths:
        candidate = (root / raw).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise OutmemError(f"Search path {raw!r} escapes the root {root}.") from exc
        resolved.append(candidate)
    return resolved


def _parse_rg_json(
    stdout: str,
    *,
    root: Path,
    max_bytes: int,
    max_hits: int | None,
) -> SearchResult:
    """Parse ``rg --json`` output into a :class:`SearchResult`.

    rg emits one JSON object per line. We only care about ``type:"match"``
    events; ``begin``, ``end``, ``summary``, and ``context`` are ignored.
    """
    hits: list[SearchHit] = []
    consumed = 0
    truncated = False

    for line in stdout.splitlines():
        if not line:
            continue
        # Track bytes consumed *before* parsing so we stop at the cap
        # cleanly rather than mid-record.
        consumed += len(line) + 1  # +1 for the newline
        if consumed > max_bytes:
            truncated = True
            break

        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "match":
            continue

        data = event.get("data", {})
        path_obj = data.get("path", {})
        path_text = path_obj.get("text") or path_obj.get("bytes") or ""
        if not path_text:
            continue

        # rg emits absolute paths if we ran it with an absolute search root.
        try:
            rel = str(Path(path_text).resolve().relative_to(root))
        except ValueError:
            rel = path_text

        lines = data.get("lines", {})
        text = lines.get("text") or ""
        line_number = data.get("line_number")
        if not isinstance(line_number, int):
            continue
        hits.append(SearchHit(path=rel, line_number=line_number, text=text.rstrip("\n")))

        if max_hits is not None and len(hits) >= max_hits:
            truncated = truncated or consumed < len(stdout)
            break

    return SearchResult(hits=tuple(hits), truncated=truncated)
