"""Is a page all there? — elision detection for wiki bodies.

Every other invariant outmem enforces is structural: a provenance SHA
matches or it doesn't, a slug is well-formed or it isn't, a wikilink
resolves or it doesn't. Completeness is the one contract that was only
ever *asked* for — ``write_page``'s docstring says "the complete
markdown body" and nothing checked it.

The failure that motivates this module is specific and quiet. When a
turn's output budget binds, the model does not necessarily fail: it can
emit a schema-valid ``write_page`` call whose body stops early and marks
the cut with an ellipsis. That call validates, commits, and passes every
other lint check, because provenance, hashes, links, and the index are
all immaculate — those are exactly what the linter examined. One wiki
carried 181 such passages across 75 pages for months, ~226k characters
of content that was sitting in registered sources the whole time.

The hard half of budget pressure was already handled: a response cut
*mid*-tool-call arrives with ``body`` missing, schema validation fires,
and the retry budget recovers it. This module is the soft half.

**The discriminator is position, not vocabulary.** A bracketed ellipsis
is also the standard elision marker in quoted material — on a wiki that
quotes guidelines verbatim, ``> "die Therapie [...] wird empfohlen"`` is
correct usage, and a checker that flags it is one a team turns off. So a
marker only counts as truncation when nothing follows it on its line:
text stopping at the marker is a cut, text continuing past it is an
elision within running prose. Blockquotes, code, and link display text
are skipped outright.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Bracketed ellipsis in either spelling, tolerating inner spaces, plus a
# bare ellipsis (caught only when it is the entire line — see below).
_BRACKETED = r"\[\s*(?:\.\s*\.\s*\.|…)\s*\]"
_BARE = r"(?:\.\s*\.\s*\.|…)"

_BRACKETED_RE = re.compile(_BRACKETED)
_BARE_LINE_RE = re.compile(rf"^\s*{_BARE}\s*$")

# An HTML comment that says content is missing. Invisible when rendered,
# which is what makes it worse than the visible markers, not better.
_COMMENT_RE = re.compile(
    r"<!--[^>]*\b(?:truncat\w*|omitted|abbreviated|continues?|gek(?:ue|ü)rzt|"
    r"fortsetzung|rest\s+folgt)\b[^>]*-->",
    re.IGNORECASE,
)

# Typographic closers, spelled as escapes: a literal U+2019 in source
# trips ruff's confusable-character check (it reads as a backtick).
# German pages close quotations with these, so they have to count as
# trailing punctuation — otherwise a cut inside a quotation reads as
# prose continuing and is never flagged.
_CLOSERS = (
    "\u00bb"      # right-pointing double angle quotation mark
    "\u201c\u201d"  # double quotation marks
    "\u2018\u2019"  # single quotation marks
)
# Only whitespace and closing punctuation may follow a marker for it to
# count as line-terminal. "Die Punkte sind: […]." is a cut wearing a full
# stop; "die Therapie [...] wird empfohlen" is a quotation.
_TRAILING_OK_RE = re.compile(rf"^[\s.;,:!?)\]\"'{_CLOSERS}]*$")

_FENCE_RE = re.compile(r"^\s*(?P<ticks>```+|~~~+)")
_BLOCKQUOTE_RE = re.compile(r"^\s*>")

# outmem's own out-of-band marker for content it withheld from a tool
# result (an oversized source, a spliced rerank excerpt). Deliberately
# unmistakable: it must never be confusable with prose, so that banning
# elision in page bodies does not also ban outmem's own vocabulary — and
# so that finding it *in* a page body is itself a detectable bug.
TOOL_SENTINEL_OPEN = "⟪ outmem:"
TOOL_SENTINEL_CLOSE = "⟫"


def tool_note(text: str) -> str:
    """Wrap ``text`` as an outmem tool-output note.

    Used wherever outmem withholds content from a tool result. Kept in
    this module, next to the checks, because the two are one decision:
    the marker outmem emits and the markers outmem rejects must never
    overlap, or the framework teaches the model the very pattern it then
    refuses.
    """
    return f"{TOOL_SENTINEL_OPEN} {text} {TOOL_SENTINEL_CLOSE}"


@dataclass(frozen=True)
class Elision:
    """One suspected truncation point in a page body.

    ``line`` is 1-based and relative to whatever text was scanned;
    callers holding the original file add their own frontmatter offset.
    ``text`` is the offending line, stripped — quoted back to the user
    because that, not the number, is what makes the spot findable.
    """

    line: int
    marker: str
    text: str


def _inline_code_spans(line: str) -> list[tuple[int, int]]:
    """Half-open ``(start, end)`` ranges of backtick spans in ``line``."""
    spans: list[tuple[int, int]] = []
    for match in re.finditer(r"`+", line):
        ticks = match.group(0)
        closer = line.find(ticks, match.end())
        if closer != -1:
            spans.append((match.start(), closer + len(ticks)))
    return spans


def _is_link_text(line: str, end: int) -> bool:
    """True when the marker ending at ``end`` is a link's display text.

    ``[…](https://example.org/…)`` and ``[…][ref]`` are both legitimate:
    the brackets are markdown syntax, not an elision.
    """
    return line[end : end + 1] in {"(", "["}


def find_elision_markers(text: str) -> list[Elision]:
    """Elision markers in ``text`` that read as truncation, not quotation.

    Flagged: a bracketed ellipsis with nothing but closing punctuation
    after it on its line; a bare ellipsis alone on a line; an HTML
    comment announcing omitted content.

    Not flagged: anything inside a fenced code block, an inline code
    span, or a blockquote; a bracketed ellipsis used as link display
    text; a bracketed ellipsis with prose continuing after it, which is
    an elision *within* a sentence and the normal way to shorten a quote.
    """
    found: list[Elision] = []
    fence: str | None = None

    for index, line in enumerate(text.splitlines(), start=1):
        fence_match = _FENCE_RE.match(line)
        if fence_match is not None:
            ticks = fence_match.group("ticks")
            if fence is None:
                fence = ticks[0]
            elif ticks[0] == fence:
                fence = None
            continue
        if fence is not None:
            continue
        # Quoted material: an ellipsis here shortens the quotation, and a
        # quotation that stops mid-sentence is the author's business.
        if _BLOCKQUOTE_RE.match(line):
            continue

        stripped = line.strip()
        code_spans = _inline_code_spans(line)

        def _in_code(pos: int, spans: list[tuple[int, int]] = code_spans) -> bool:
            return any(start <= pos < end for start, end in spans)

        comment = _COMMENT_RE.search(line)
        if comment is not None and not _in_code(comment.start()):
            found.append(
                Elision(line=index, marker=comment.group(0), text=stripped)
            )
            continue

        if _BARE_LINE_RE.match(line):
            found.append(Elision(line=index, marker=stripped, text=stripped))
            continue

        for match in _BRACKETED_RE.finditer(line):
            if _in_code(match.start()) or _is_link_text(line, match.end()):
                continue
            if _TRAILING_OK_RE.match(line[match.end() :]):
                found.append(
                    Elision(line=index, marker=match.group(0), text=stripped)
                )
                break

    return found


def find_tool_sentinels(text: str) -> list[Elision]:
    """Occurrences of outmem's own tool-output marker in ``text``.

    In a page body this is never legitimate: it means a tool result that
    outmem had deliberately shortened — an oversized source, a spliced
    search excerpt — was copied into the wiki verbatim, so the page now
    asserts content outmem itself declined to show.
    """
    return [
        Elision(line=index, marker=TOOL_SENTINEL_OPEN, text=line.strip())
        for index, line in enumerate(text.splitlines(), start=1)
        if TOOL_SENTINEL_OPEN in line
    ]
