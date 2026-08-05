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
    """A single ripgrep row — one file, one line.

    With ``context=0`` (the default) every row is a match. Above that,
    ``rg`` also returns the surrounding lines and they arrive here with
    ``is_match=False`` — same shape, different role. Renderers keep the
    two distinguishable (ripgrep spells them ``path:line:text`` and
    ``path-line-text``); a caller that ignores the flag gets a plain
    line list, which is why it defaults to the match case.
    """

    path: str  # relative to the search root
    line_number: int
    text: str
    is_match: bool = True


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
    context: int = 0,
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
            everything under ``root``; an explicitly *empty* list means
            nothing is in scope and yields an empty result. The two are
            not interchangeable — callers that compute a path list by
            filtering (e.g. "every sources tree that exists") rely on
            the empty case staying empty rather than silently widening
            to the whole root.
        case_insensitive: ``rg -i``.
        fixed_strings: ``rg -F`` — treat the pattern as a literal string.
        context: Lines of surrounding context per match (``rg -C``).
            ``0`` returns matches only. Context rows come back as
            :class:`SearchHit` with ``is_match=False``, interleaved in
            file/line order. They count against ``max_bytes`` like any
            other output, so a wide pattern with generous context
            truncates sooner — which is the intended signal.
        max_bytes: Soft ceiling on the bytes of ``rg`` output we consume.
            Exceeding it sets ``SearchResult.truncated``.
        max_hits: Optional hard cap on the number of returned hits.
            Counts matches only; context rows ride along with the match
            they belong to rather than consuming the budget themselves.
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

    if paths is not None and not paths:
        # Explicitly empty scope. Short-circuit rather than invoking rg
        # with no path (which would search the whole root).
        return SearchResult(hits=(), truncated=False)

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
    if context > 0:
        args.extend(["-C", str(context)])
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
        context=context,
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

    ``None`` (search everything) is the only input that widens to
    ``root``; an empty list is handled by the caller and never reaches
    here.
    """
    if paths is None:
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


def _trim_dangling_context(hits: list[SearchHit], context: int) -> None:
    """Drop trailing context rows that belong to an excluded match.

    When ``max_hits`` cuts the result short, rg has usually already
    emitted the *leading* context for the next match. Those rows are
    real file content, but they orbit a match the caller will never see
    — so a renderer that groups by contiguity shows a group with nothing
    matched in it.

    The rule is exact rather than heuristic: keep trailing rows within
    ``context`` lines of the last match, drop the rest. Contiguity alone
    can't decide it, because with a large enough ``context`` the two
    matches' windows touch and the run is unbroken. Mutates in place.
    """
    last_match = next((h for h in reversed(hits) if h.is_match), None)
    if last_match is None:
        return
    while hits and not hits[-1].is_match:
        if hits[-1].line_number <= last_match.line_number + context:
            break
        hits.pop()


def _parse_rg_json(
    stdout: str,
    *,
    root: Path,
    max_bytes: int,
    max_hits: int | None,
    context: int = 0,
) -> SearchResult:
    """Parse ``rg --json`` output into a :class:`SearchResult`.

    rg emits one JSON object per line. We keep ``type:"match"`` and —
    when the caller asked for context — ``type:"context"``, which has an
    identical payload shape and differs only in role. ``begin``, ``end``
    and ``summary`` are ignored.

    ``max_hits`` counts matches only. Dropping a match's context rows
    from the budget would make the cap mean something different at each
    context width, and the caller asked for N *results*, not N lines.
    """
    hits: list[SearchHit] = []
    matches = 0
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
        event_type = event.get("type")
        if event_type not in ("match", "context"):
            continue
        # Budget spent, but rg's trailing context for the last match is
        # still arriving. Take it, then stop at the next match — a match
        # shown without the context that was explicitly asked for is the
        # half-answer this parameter exists to avoid.
        if max_hits is not None and matches >= max_hits and event_type == "match":
            truncated = truncated or consumed < len(stdout)
            _trim_dangling_context(hits, context)
            break

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
        is_match = event_type == "match"
        hits.append(
            SearchHit(
                path=rel,
                line_number=line_number,
                text=text.rstrip("\n"),
                is_match=is_match,
            )
        )
        if is_match:
            matches += 1
        # No break here: the cap is enforced at the top of the loop, on
        # the *next* match, so trailing context lands first. With
        # context=0 that is the same thing — the next event after a
        # match is the next match.

    return SearchResult(hits=tuple(hits), truncated=truncated)
