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
    r"<!--(?:(?!-->).)*?\b(?:truncat\w*|omitted|abbreviated|continues?|"
    r"gek(?:ue|ü)rzt|fortsetzung|rest\s+folgt)\b(?:(?!-->).)*?-->",
    re.IGNORECASE | re.DOTALL,
)

# Typographic closers, spelled as escapes: a literal U+2019 in source
# trips ruff's confusable-character check (it reads as a backtick).
# German pages close quotations with these, so they have to count as
# trailing punctuation — otherwise a cut inside a quotation reads as
# prose continuing and is never flagged.
_CLOSERS = (
    # Both guillemets. German chevron style points inward and therefore
    # CLOSES with U+00AB; French closes with U+00BB. A wiki may use
    # either, and getting it backwards means a truncated quotation in
    # that style is never flagged — which is what this set exists for.
    "\u00ab\u00bb"
    "\u201c\u201d"  # double quotation marks
    "\u2018\u2019"  # single quotation marks
    "\u203a\u2039"  # single angle quotation marks
)
# Only whitespace and closing punctuation may follow a marker for it to
# count as line-terminal. "Die Punkte sind: […]." is a cut wearing a full
# stop; "die Therapie [...] wird empfohlen" is a quotation.
_TRAILING_OK_RE = re.compile(rf"^[\s.;,:!?)\]\"'{_CLOSERS}]*$")

_FENCE_RE = re.compile(r"^\s*(?P<ticks>```+|~~~+)")
_BLOCKQUOTE_RE = re.compile(r"^\s*>")
_TICKS_RE = re.compile(r"`+")
# CommonMark indented code — four spaces or a tab, with no fence to key
# on. Excludes list markers, whose continuation lines are indented the
# same way and are ordinary prose.
_INDENTED_CODE_RE = re.compile(r"^(?: {4,}|\t)(?![-*+]\s|\d+[.)]\s)")

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
    """Half-open ``(start, end)`` ranges of backtick spans in ``line``.

    Runs are consumed in pairs. Treating every run as a potential opener
    would let a *closing* run open a second span, so the prose between
    two code spans would be swallowed as code and skipped — which is how
    a marker between two inline snippets escapes the check entirely.
    """
    spans: list[tuple[int, int]] = []
    pos = 0
    while (opener := _TICKS_RE.search(line, pos)) is not None:
        ticks = opener.group(0)
        closer = line.find(ticks, opener.end())
        if closer == -1:
            break  # unpaired run — the rest of the line is not code
        end = closer + len(ticks)
        spans.append((opener.start(), end))
        pos = end
    return spans


def _fenced_line_numbers(lines: list[str]) -> frozenset[int]:
    """Indices of lines inside a *closed* code fence.

    Pairing matters: an unterminated fence — a lone ``` quoted from a
    source, a block the model forgot to close — would otherwise exempt
    every remaining line from every check, silently and with no signal
    that the scan stopped. An unpaired opener is treated as ordinary
    text instead, so the failure mode is a possible false positive
    rather than a whole page going unchecked.
    """
    fenced: set[int] = set()
    open_at: int | None = None
    marker: str | None = None
    for index, line in enumerate(lines):
        match = _FENCE_RE.match(line)
        if match is None:
            continue
        ticks = match.group("ticks")[0]
        if open_at is None:
            open_at, marker = index, ticks
        elif ticks == marker:
            fenced.update(range(open_at, index + 1))
            open_at, marker = None, None
    return frozenset(fenced)


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
    lines = text.splitlines()
    fenced = _fenced_line_numbers(lines)

    # HTML comments can span lines, so they are matched over the whole
    # text and mapped back to a line number rather than scanned per line.
    for match in _COMMENT_RE.finditer(text):
        line_no = text.count("\n", 0, match.start()) + 1
        if line_no - 1 in fenced:
            continue
        found.append(
            Elision(
                line=line_no,
                marker=" ".join(match.group(0).split()),
                text=lines[line_no - 1].strip() if line_no <= len(lines) else "",
            )
        )

    for index, line in enumerate(lines, start=1):
        if index - 1 in fenced or _INDENTED_CODE_RE.match(line):
            continue
        # Quoted material: an ellipsis here shortens the quotation, and a
        # quotation that stops mid-sentence is the author's business.
        if _BLOCKQUOTE_RE.match(line):
            continue

        stripped = line.strip()
        code_spans = _inline_code_spans(line)

        def _in_code(pos: int, spans: list[tuple[int, int]] = code_spans) -> bool:
            return any(start <= pos < end for start, end in spans)

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
