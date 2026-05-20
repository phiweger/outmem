"""Paragraph-aware text chunker for the semantic index.

Breaks markdown bodies on paragraph boundaries (``\\n\\n``) and groups
consecutive paragraphs into chunks targeting a configured character
count. Two properties make this design useful for incremental indexing:

1. **Chunk boundaries are stable across local edits.** A paragraph
   inserted into the middle of a document shifts at most one or two
   adjacent chunk boundaries; chunks before and after stay
   byte-identical.

2. **Paragraphs are not split mid-way.** A paragraph that exceeds the
   target ``chunk_size`` becomes its own chunk (up to a hard
   ``chunk_max`` ceiling).

Frontmatter is the *caller's* responsibility to strip before calling
this — use :func:`outmem.frontmatter.parse_wiki_page` for wiki pages.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from hashlib import sha256

# A paragraph break is one or more blank lines (which may contain
# whitespace). We preserve a single newline within a paragraph so
# markdown lists / soft line breaks survive.
_PARA_SPLIT = re.compile(r"\n[ \t]*\n+")


@dataclass(frozen=True)
class Chunk:
    """One chunk produced by :func:`chunk_text`."""

    index: int
    text: str
    start_char: int  # offset of chunk start in the source body
    end_char: int  # offset of chunk end (exclusive)

    @property
    def content_hash(self) -> str:
        return sha256(self.text.encode("utf-8")).hexdigest()


def chunk_text(
    body: str,
    *,
    chunk_size: int = 2000,
    chunk_max: int = 8000,
    overlap_paragraphs: int = 1,
) -> list[Chunk]:
    """Split ``body`` into paragraph-aware chunks.

    Algorithm:

    1. Split on ``\\n\\n`` to get paragraphs (with their original offsets).
    2. Greedily group paragraphs into chunks while their combined size
       stays below ``chunk_size``. A single paragraph larger than
       ``chunk_size`` becomes its own chunk (up to ``chunk_max``).
    3. Overlap: include the last ``overlap_paragraphs`` paragraphs of
       chunk N at the start of chunk N+1. Setting it to ``0`` disables
       overlap.

    Empty bodies return an empty list. A body with one short paragraph
    returns one chunk.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_max < chunk_size:
        raise ValueError("chunk_max must be >= chunk_size")
    if overlap_paragraphs < 0:
        raise ValueError("overlap_paragraphs must be non-negative")

    body = body.strip()
    if not body:
        return []

    paragraphs = _split_paragraphs(body)
    if not paragraphs:
        return []

    chunks: list[Chunk] = []
    last_used_idx = -1
    i = 0
    while i < len(paragraphs):
        group_indices: list[int] = []
        size = 0
        while i < len(paragraphs):
            start, end, _ = paragraphs[i]
            para_len = end - start
            # Always include at least one paragraph, even if oversized.
            if group_indices and size + para_len + 2 > chunk_size:
                break
            group_indices.append(i)
            size += para_len + 2
            i += 1
            if size >= chunk_size or size >= chunk_max:
                break

        if not group_indices:
            break

        last_idx = group_indices[-1]
        chunk_start = paragraphs[group_indices[0]][0]
        chunk_end = paragraphs[last_idx][1]
        chunk_body = "\n\n".join(paragraphs[j][2] for j in group_indices)
        chunks.append(
            Chunk(
                index=len(chunks),
                text=chunk_body,
                start_char=chunk_start,
                end_char=chunk_end,
            )
        )

        # Backtrack `overlap_paragraphs` paragraphs so the next chunk
        # starts with the tail of this one. Cap so we always make
        # forward progress (the next chunk must include at least one
        # paragraph past the previous chunk's last paragraph).
        if overlap_paragraphs > 0 and i < len(paragraphs):
            backtrack = min(overlap_paragraphs, len(group_indices) - 1)
            i = max(last_idx + 1 - backtrack, last_used_idx + 1)
        last_used_idx = last_idx

    return chunks


def _split_paragraphs(body: str) -> list[tuple[int, int, str]]:
    """Return ``[(start, end, text)]`` for each paragraph in ``body``.

    Offsets are into the *stripped* body the caller passed in. Empty
    paragraphs are skipped.
    """
    out: list[tuple[int, int, str]] = []
    cursor = 0
    for match in _PARA_SPLIT.finditer(body):
        segment = body[cursor : match.start()]
        stripped = segment.strip()
        if stripped:
            offset = segment.index(stripped) + cursor
            out.append((offset, offset + len(stripped), stripped))
        cursor = match.end()
    tail = body[cursor:]
    stripped = tail.strip()
    if stripped:
        offset = tail.index(stripped) + cursor
        out.append((offset, offset + len(stripped), stripped))
    return out


def hash_text(text: str) -> str:
    """SHA-256 hex digest of ``text`` (utf-8) — file-level content hash."""
    return sha256(text.encode("utf-8")).hexdigest()
