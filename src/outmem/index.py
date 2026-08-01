"""Auto-maintained ``wiki/index.md`` — a catalog of every wiki page.

Generated, not hand-edited. :class:`outmem.store.WikiStore` regenerates
the file as part of every ``write_page`` / ``extend_page`` commit so
``index.md`` stays in lockstep with the wiki content. The file is
never updated outside an explicit outmem write — external edits (e.g.
via Obsidian) won't trigger a rebuild; that drift is caught by
``outmem lint`` (see :mod:`outmem.lint`).

Pages live recursively under ``wiki/pages/``; the index walks that
subtree and emits one ``[[slug]]`` line per page, sorted alphabetically
by slug. Namespaced pages render as ``[[abx:penicillin]]`` — the
projection to a directory on disk is handled by :func:`slug_to_relpath`.

Format:

.. code-block:: markdown

    # Wiki index

    - [[abx:penicillin]] — Penicillin (abx, antibiotics)
    - [[acme-msa]] — Acme MSA (contracts, acme, pricing)
    - [[pricing-formula]] — Pricing formula (pricing, contracts, finance)

The index page itself lives at ``wiki/index.md`` with its own slug
(``index``) — so consumers can navigate to it via the dashboard at
``/wiki/index`` like any other page. It has no inbound wikilinks
(it's the entry point) so ``outmem lint`` knows not to flag it as
an orphan.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from outmem.exceptions import FrontmatterError
from outmem.frontmatter import parse_wiki_page, repair_wiki_page
from outmem.slug import PAGES_DIR, relpath_to_slug

if TYPE_CHECKING:
    from outmem.frontmatter import WikiFrontmatter

log = logging.getLogger(__name__)

INDEX_SLUG = "index"
INDEX_FILENAME = "index.md"
INDEX_TITLE = "Wiki index"

# Files that live at the wiki root but are not editorial pages —
# infrastructure that outmem auto-maintains or that the user customises
# globally. Editorial pages live under ``wiki/pages/`` instead, so this
# set is informational only: it documents the reserved names.
AGENTS_FILENAME = "AGENTS.md"
RESERVED_WIKI_FILES = frozenset({INDEX_FILENAME, AGENTS_FILENAME})


def editorial_pages(pages_dir: Path) -> list[Path]:
    """Every wiki page on disk, walking ``pages_dir`` recursively.

    Single source of truth for "which `*.md` files in `wiki/pages/`
    are editorial content" — used by the indexer, the linter, the
    slug listing, and the semantic indexer. Sorted for deterministic
    output.
    """
    if not pages_dir.is_dir():
        return []
    return sorted(pages_dir.rglob("*.md"))


@dataclass(frozen=True)
class LoadedPage:
    """One successfully-parsed editorial page."""

    path: Path
    slug: str  # path-derived — the address
    frontmatter: WikiFrontmatter
    body: str
    repaired: bool = False  # frontmatter was mended in memory to parse
    declared_slug: str = ""  # what the page's own `slug:` claims

    @property
    def slug_mismatch(self) -> bool:
        """The page declares a slug that isn't where it lives.

        Harmless to read (the path still addresses it) but it means the
        same page has two names, and the declared one resolves to
        nothing. Usually a ``git mv`` that didn't update the frontmatter.
        """
        return bool(self.declared_slug) and self.declared_slug != self.slug


@dataclass(frozen=True)
class PageLoadFailure:
    """An editorial page on disk that could not be parsed.

    Every consumer of :func:`load_editorial_pages` gets these; what
    differs is only how each one *reports* them (lint makes a finding,
    reindex sets an exit code, the index logs a warning). Losing a page
    silently — which every loader used to do independently — is what this
    type exists to prevent.
    """

    path: Path
    slug: str
    error: str


def load_page_text(
    text: str, *, fallback_slug: str | None = None
) -> tuple[WikiFrontmatter, str, bool]:
    """Parse one page's text, self-healing the repairable shape.

    Returns ``(frontmatter, body, repaired)``. Mirrors what
    :meth:`outmem.store.WikiStore.read` does, so "how outmem loads a
    page" has one answer: a page that ``read_page`` can serve is a page
    the index, the TOC and the backlink graph can also see. Raises
    :class:`FrontmatterError` when the break isn't mechanically fixable.

    ``fallback_slug`` (the path-derived name) stands in for a missing
    ``slug:`` — see :func:`outmem.frontmatter.parse_wiki_page`.
    """
    try:
        frontmatter, body = parse_wiki_page(text, fallback_slug=fallback_slug)
    except FrontmatterError:
        repaired_text = repair_wiki_page(text)
        if repaired_text is None:
            raise  # not a shape we can mend — surface it
        frontmatter, body = parse_wiki_page(
            repaired_text, fallback_slug=fallback_slug
        )
        return frontmatter, body, True
    return frontmatter, body, False


def load_editorial_pages(
    pages_dir: Path,
) -> tuple[list[LoadedPage], list[PageLoadFailure]]:
    """Load every editorial page under ``pages_dir``.

    The companion to :func:`editorial_pages`: that one answers *which*
    files are editorial content, this one answers *what is in them* and
    *which ones failed*. Returning both halves is the point — a loader
    that only gets the successes has no way to notice a page vanished,
    which is how the same silent-drop bug appeared independently in the
    indexer, the TOC builder and the backlink graph.
    """
    pages: list[LoadedPage] = []
    failures: list[PageLoadFailure] = []
    for path in editorial_pages(pages_dir):
        fallback_slug = relpath_to_slug(path.relative_to(pages_dir))
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            failures.append(PageLoadFailure(path, fallback_slug, f"unreadable: {exc}"))
            continue
        try:
            frontmatter, body, repaired = load_page_text(
                text, fallback_slug=fallback_slug
            )
        except FrontmatterError as exc:
            failures.append(PageLoadFailure(path, fallback_slug, str(exc)))
            continue
        pages.append(
            LoadedPage(
                path=path,
                # The PATH addresses the page — always. A declared `slug:`
                # that disagrees is a defect to report (see
                # WikiStore.unreadable), never an alternative address:
                # honouring it here would put a name in wiki/index.md that
                # `read()` cannot open.
                slug=fallback_slug,
                declared_slug=frontmatter.slug,
                frontmatter=frontmatter,
                body=body,
                repaired=repaired,
            )
        )
    return pages, failures


def alias_index(pages_dir: Path) -> dict[str, str]:
    """Map every declared alias to the slug of the page that claims it.

    File-first: an alias that collides with a live page's slug is
    dropped, so a stale alias can never shadow a real page. Collisions
    between two pages' aliases are also dropped — the winner would depend
    on directory order — and ``outmem lint`` reports both.
    """
    pages, _failures = load_editorial_pages(pages_dir)
    real = {p.slug for p in pages}
    out: dict[str, str] = {}
    clashing: set[str] = set()
    for page in pages:
        for alias in page.frontmatter.aliases:
            if alias in real or alias == INDEX_SLUG:
                continue  # a live page always wins its own name
            if alias in out and out[alias] != page.slug:
                clashing.add(alias)
                continue
            out[alias] = page.slug
    for alias in clashing:
        out.pop(alias, None)
    return out


def render_index(pages_dir: Path) -> str:
    """Build the index.md content from the current state of ``pages_dir``.

    Walks ``*.md`` files under ``pages_dir`` recursively, parses each
    frontmatter, and emits an alphabetised list keyed by slug. A page
    whose frontmatter won't parse is logged at WARNING and left out, so
    the index renders against a partially-broken wiki without crashing —
    but never drops a page without saying so.
    """
    pages, failures = load_editorial_pages(pages_dir)
    for failure in failures:
        log.warning(
            "index: %s left out of index.md — %s", failure.slug, failure.error
        )
    entries: list[tuple[str, str]] = []
    for page in pages:
        line = f"- [[{page.slug}]] — {page.frontmatter.title}"
        if page.frontmatter.tags:
            line += f" ({', '.join(page.frontmatter.tags)})"
        entries.append((page.slug, line))

    entries.sort(key=lambda e: e[0])
    body = "\n".join(line for _, line in entries) if entries else "_(no pages yet)_"
    return f"# {INDEX_TITLE}\n\n{len(entries)} page{'' if len(entries) == 1 else 's'}.\n\n{body}\n"


@dataclass(frozen=True)
class IndexLevel:
    """One navigable level of the slug index (the wiki's table of contents).

    ``prefix`` is the namespace this level sits at (``""`` is the root).
    ``namespaces`` are the child namespaces directly below it — each a
    ``(name, page_count)`` pair where ``name`` is itself a valid prefix to
    drill into and ``page_count`` is how many pages live anywhere beneath
    it. ``pages`` are the leaf-page slugs that sit *directly* at this level
    (no further namespace segment). Both lists are sorted.
    """

    prefix: str
    namespaces: list[tuple[str, int]]
    pages: list[str]


def navigate_index(slugs: Sequence[str], prefix: str = "") -> IndexLevel:
    """Group a flat slug list into one navigable level of the TOC.

    Slugs are ``:``-namespaced (``abx:penicillin``,
    ``abx:side-effects:misc``). Given ``prefix`` (``""`` = root), return
    the child namespaces directly below it (with page counts) plus the
    leaf pages sitting directly at that level — the shape a caller walks
    one step at a time, passing a returned namespace back as the next
    ``prefix``. A trailing ``:`` on ``prefix`` is tolerated.

    A slug can be both a page *and* a namespace (``abx.md`` alongside
    ``abx/penicillin.md``): it then appears in ``pages`` at its own level
    and as a namespace one level up.
    """
    prefix = prefix.strip().rstrip(":")
    seg_prefix = f"{prefix}:" if prefix else ""
    counts: dict[str, int] = {}
    pages: list[str] = []
    for slug in slugs:
        if seg_prefix:
            if not slug.startswith(seg_prefix):
                continue
            remainder = slug[len(seg_prefix) :]
        else:
            remainder = slug
        head, sep, _ = remainder.partition(":")
        if not head:
            continue
        if sep:  # more segments below → ``head`` names a child namespace
            child = f"{seg_prefix}{head}"
            counts[child] = counts.get(child, 0) + 1
        else:  # a leaf page sitting directly at this level
            pages.append(slug)
    return IndexLevel(
        prefix=prefix,
        namespaces=sorted(counts.items()),
        pages=sorted(pages),
    )


def index_page_text(pages_dir: Path) -> str:
    """Render the full ``index.md`` file with frontmatter + body."""
    from outmem.frontmatter import WikiFrontmatter, serialize_wiki_page

    fm = WikiFrontmatter(
        title=INDEX_TITLE,
        slug=INDEX_SLUG,
        tags=["index"],
        extra={"generated": True},
    )
    body = render_index(pages_dir)
    return serialize_wiki_page(fm, body)


__all__ = [
    "AGENTS_FILENAME",
    "INDEX_FILENAME",
    "INDEX_SLUG",
    "INDEX_TITLE",
    "PAGES_DIR",
    "RESERVED_WIKI_FILES",
    "IndexLevel",
    "LoadedPage",
    "PageLoadFailure",
    "alias_index",
    "editorial_pages",
    "index_page_text",
    "load_editorial_pages",
    "load_page_text",
    "navigate_index",
    "render_index",
]
