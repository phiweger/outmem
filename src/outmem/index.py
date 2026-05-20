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

from pathlib import Path

from outmem.frontmatter import parse_wiki_page
from outmem.slug import PAGES_DIR, relpath_to_slug

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


def render_index(pages_dir: Path) -> str:
    """Build the index.md content from the current state of ``pages_dir``.

    Walks ``*.md`` files under ``pages_dir`` recursively, parses each
    frontmatter, and emits an alphabetised list keyed by slug. Pages
    with malformed frontmatter are silently skipped — ``outmem lint``
    catches them separately so the index can render against a
    partially-broken wiki without crashing.
    """
    entries: list[tuple[str, str]] = []
    for path in editorial_pages(pages_dir):
        try:
            frontmatter, _ = parse_wiki_page(path.read_text(encoding="utf-8"))
        except Exception:
            # Malformed page — skip; lint will surface it.
            continue
        slug = frontmatter.slug or relpath_to_slug(path.relative_to(pages_dir))
        title = frontmatter.title
        tags = frontmatter.tags
        line = f"- [[{slug}]] — {title}"
        if tags:
            line += f" ({', '.join(tags)})"
        entries.append((slug, line))

    entries.sort(key=lambda e: e[0])
    body = "\n".join(line for _, line in entries) if entries else "_(no pages yet)_"
    return f"# {INDEX_TITLE}\n\n{len(entries)} page{'' if len(entries) == 1 else 's'}.\n\n{body}\n"


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
    "editorial_pages",
    "index_page_text",
    "render_index",
]
