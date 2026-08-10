"""Markdown section outline — a page's shape, without its contents.

The map behind ``read_page(peek=True)`` and the addressing behind
``read_page(section=…)``.

The distinction that motivates this module: "is this page about the
right topic?" and "where in this page is the fact I need?" are
different questions, and only the first can be answered by a prefix of
the text. A caller that already has a slug has usually answered the
first — ``search_wiki`` returns pages with excerpts — so the prefix
form of a preview costs a model round-trip and hands back something
the follow-up read repeats verbatim. An outline answers the second
question, is bounded by the number of headings rather than by a
character budget, and costs the same read to produce.

ATX headings only (``## Foo``). Setext underlining is not used
anywhere in outmem's own page grammar and treating a ``---`` line as a
heading would collide with frontmatter delimiters.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Leading ``#``s, at least one space, then the text. Trailing ``#``s are
# a legal ATX closing sequence and are stripped.
_HEADING = re.compile(r"^(?P<hashes>#{1,6})\s+(?P<text>.+?)\s*#*\s*$")

# A fenced block's contents are not structure — a ``# comment`` line in a
# shell snippet is not a section, and treating it as one puts noise in
# the map and lets `section=` address something that isn't a section.
_FENCE = re.compile(r"^\s*(?P<ticks>```+|~~~+)")


@dataclass(frozen=True)
class Section:
    """One heading and the span it owns.

    ``start_line`` is the heading's own line and ``end_line`` the last
    line before the next heading of the same or shallower level, both
    1-based and inclusive. Line numbers are *file*-relative when
    :func:`parse_outline` is given a ``line_offset``, so they line up
    with what ``grep_wiki`` prints for the same page.

    ``start_char`` / ``end_char`` are the same span in characters and
    are always relative to the ``body`` that was parsed — never shifted
    by ``line_offset``. The asymmetry is deliberate: line numbers exist
    to be shown to a reader alongside grep output, so they need file
    coordinates; character offsets exist to be compared against
    :class:`~outmem.semantic.chunker.Chunk` offsets, which are body
    coordinates. Shifting both would make one of the two callers wrong.
    """

    heading: str
    level: int
    start_line: int
    end_line: int
    char_count: int
    start_char: int = 0
    end_char: int = 0

    @property
    def line_count(self) -> int:
        return self.end_line - self.start_line + 1


def parse_outline(body: str, *, line_offset: int = 0) -> list[Section]:
    """Return the ATX headings in ``body`` with the span each owns.

    ``line_offset`` is added to every line number — pass the number of
    lines the frontmatter occupies and the result is addressed in the
    same coordinates as the on-disk file.

    Text before the first heading belongs to no section and is
    reported by :func:`preamble_chars` instead; a page with no headings
    yields an empty list, which callers render as "no sections".
    """
    lines = body.splitlines()
    starts: list[tuple[int, int, str]] = []  # (index, level, heading)
    fence: str | None = None

    for index, line in enumerate(lines):
        fence_match = _FENCE.match(line)
        if fence_match is not None:
            ticks = fence_match.group("ticks")
            if fence is None:
                fence = ticks[0]  # remember the character, not the run length
            elif ticks[0] == fence:
                fence = None
            continue
        if fence is not None:
            continue
        heading_match = _HEADING.match(line)
        if heading_match is not None:
            starts.append(
                (index, len(heading_match.group("hashes")), heading_match.group("text"))
            )

    # Character offset of each line's first character, so a section can
    # report its span in the same coordinates a chunker works in.
    line_starts: list[int] = []
    cursor = 0
    for line in lines:
        line_starts.append(cursor)
        cursor += len(line) + 1  # +1 for the newline splitlines() removed

    sections: list[Section] = []
    for position, (index, level, heading) in enumerate(starts):
        # The section ends before the next heading at the same or a
        # shallower level; a deeper one is a child and stays inside.
        end_index = len(lines) - 1
        for next_index, next_level, _ in starts[position + 1 :]:
            if next_level <= level:
                end_index = next_index - 1
                break
        span = "\n".join(lines[index : end_index + 1])
        sections.append(
            Section(
                heading=heading,
                level=level,
                start_line=index + 1 + line_offset,
                end_line=end_index + 1 + line_offset,
                char_count=len(span),
                start_char=line_starts[index],
                end_char=line_starts[end_index] + len(lines[end_index]),
            )
        )
    return sections


def heading_path_at(sections: list[Section], offset: int) -> tuple[str, ...]:
    """Headings enclosing ``offset``, outermost first.

    ``("Therapie", "Dosierung")`` for a position inside an ``### Dosierung``
    nested under ``## Therapie``. Empty when the offset precedes the first
    heading (a page's preamble belongs to no section).

    Used to tell the embedder which section a chunk came from. Without it
    a chunk carries its page's title and tags but nothing about the
    heading above it, so a section whose body never repeats its own
    heading is unretrievable by that heading — the same defect
    ``embed_frontmatter`` fixes one scope up.
    """
    enclosing = [s for s in sections if s.start_char <= offset <= s.end_char]
    return tuple(s.heading for s in sorted(enclosing, key=lambda s: s.level))


def preamble_chars(body: str) -> int:
    """Characters before the first heading — content no section owns.

    Worth reporting separately: a page whose body is mostly preamble has
    a misleading outline, and the number is how a caller notices.
    """
    lines = body.splitlines()
    for index, line in enumerate(lines):
        if _HEADING.match(line):
            return len("\n".join(lines[:index]))
    return len(body)


def find_section(sections: list[Section], query: str) -> list[Section]:
    """Resolve ``query`` against heading text — exact, then substring.

    Returns every candidate rather than picking one, so the caller can
    tell "no such section" from "which of these three did you mean".
    Matching is case- and whitespace-insensitive because the query is
    typically a heading copied out of an outline by a language model,
    and a rejection over a stray space costs a whole round-trip.
    """
    normalised = " ".join(query.split()).casefold()
    exact = [s for s in sections if " ".join(s.heading.split()).casefold() == normalised]
    if exact:
        return exact
    return [s for s in sections if normalised in " ".join(s.heading.split()).casefold()]
