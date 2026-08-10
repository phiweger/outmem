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
A page's title/tags can still be fed to the embedder without polluting
the stored chunk: see :func:`with_header`, which the VectorStore applies
at embed time only.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
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
    heading_path: tuple[str, ...] = ()
    """ATX headings enclosing this chunk's own content, outermost first.

    Carried for the embedder, not for storage: like the page header it
    is applied by :func:`with_header` at embed time, so
    ``Chunk.text`` stays contractually ``body[start_char:end_char]``.

    Taken at the first paragraph this chunk does *not* share with its
    predecessor, not at ``start_char``. With the default
    ``overlap_paragraphs=1`` those differ exactly when a chunk boundary
    lands on a section boundary — the chunk then opens with the tail of
    the previous section and is otherwise entirely about the next one.
    Labelling it with the previous section is worse than labelling it
    with nothing, because it pulls the chunk toward a topic it does not
    discuss.

    A chunk can still span several sections when they are short; the
    path names the one it starts in. Naming all of them would render as
    ``A > B``, which reads as nesting when they are siblings.
    """

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

    # One parse for the whole body; each chunk then reads its enclosing
    # headings off it by offset. Imported here rather than at module
    # scope to keep the chunker importable without pulling the outline
    # parser into callers that only want plain text splitting.
    from outmem.outline import heading_path_at, parse_outline

    sections = parse_outline(body)

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
        # Which section this chunk is *about*, which is not always the one
        # it starts in: `last_used_idx` is the previous chunk's last
        # paragraph, so anything past it is content this chunk introduces.
        # Reading the heading at `chunk_start` instead would label a chunk
        # by the section its overlap paragraph trails out of.
        own_start_idx = min(max(group_indices[0], last_used_idx + 1), last_idx)
        chunks.append(
            Chunk(
                index=len(chunks),
                text=chunk_body,
                start_char=chunk_start,
                end_char=chunk_end,
                heading_path=heading_path_at(sections, paragraphs[own_start_idx][0]),
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


HEADING_SEPARATOR = " > "
"""Joins a chunk's heading path. Distinctive enough not to collide with
prose, so the embedded line reads as structure rather than a sentence."""


def with_header(
    header: str, text: str, heading_path: Sequence[str] = ()
) -> str:
    """``text`` with ``header`` prepended, for what gets EMBEDDED.

    Kept out of :class:`Chunk` on purpose. ``Chunk.text`` is persisted as
    ``chunks.content`` and is contractually ``body[start_char:end_char]``;
    baking a header into it would break that identity, store the same
    header once per chunk in a git-tracked DB, push every chunk over
    ``chunk_max``, and burn preview budget re-printing the page title next
    to the path it already appears in. Applying it at the embed call sites
    keeps the header in the vectors, where it is wanted, and nowhere else.

    ``heading_path`` adds the chunk's section trail on its own line::

        Erysipel und Phlegmone — clinical, haut
        Diagnostik > Blutkulturen

        …chunk text…

    Same rationale as the page header, one scope down: a section whose
    body never repeats its own heading is unretrievable by that heading,
    and the heading is not otherwise in the vector — the chunker splits
    on blank lines, so ``## Blutkulturen`` is just another paragraph and
    every chunk after the first in that section loses it.

    The same function feeds the content hash, so toggling
    ``semantic.embed_frontmatter`` / ``embed_headings`` (or editing a
    title, tags, or heading) invalidates exactly the files whose
    embedded text changed.
    """
    prefix = "\n".join(
        part
        for part in (header, HEADING_SEPARATOR.join(heading_path))
        if part
    )
    return f"{prefix}\n\n{text}" if prefix else text
