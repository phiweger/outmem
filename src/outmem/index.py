"""Auto-maintained ``wiki/index.md`` — a catalog of every wiki page.

Generated, not hand-edited. :class:`outmem.store.WikiStore` regenerates
the file as part of every ``write_page`` / ``extend_page`` commit so
``index.md`` stays in lockstep with the wiki content. The file is
never updated outside an explicit outmem write — external edits (e.g.
via Obsidian) won't trigger a rebuild; that drift is caught by
``outmem lint`` (see :mod:`outmem.lint`).

Format (alphabetical, one line per page):

.. code-block:: markdown

    # Wiki index

    - [[acme-msa]] — Acme MSA (contracts, acme, pricing)
    - [[discounts]] — Discount tiers (pricing)
    - [[pricing-formula]] — Pricing formula (pricing, contracts, finance)

The index page itself is a regular wiki page with its own slug
(``index``) — so consumers can navigate to it via the dashboard at
``/wiki/index`` like any other page. It has no inbound wikilinks
(it's the entry point) so ``outmem lint`` knows not to flag it as
an orphan.
"""

from __future__ import annotations

from pathlib import Path

from outmem.frontmatter import parse_wiki_page

INDEX_SLUG = "index"
INDEX_FILENAME = "index.md"
INDEX_TITLE = "Wiki index"

# Files that live in ``wiki/`` but are not wiki pages — they don't have
# the page frontmatter, don't get an index entry, aren't crawled for
# backlinks, and don't show up in ``list_slugs``. Add to this set when
# introducing another auto-managed wiki file.
AGENTS_FILENAME = "AGENTS.md"
RESERVED_WIKI_FILES = frozenset({INDEX_FILENAME, AGENTS_FILENAME})


def editorial_pages(wiki_dir: Path) -> list[Path]:
    """Every wiki page on disk, skipping the auto-managed reserved files.

    Single source of truth for "which `*.md` files in `wiki/` are
    editorial content" — used by the indexer, the linter, the slug
    listing, and the semantic indexer. Sorted for deterministic
    output.
    """
    return sorted(
        p for p in wiki_dir.glob("*.md") if p.name not in RESERVED_WIKI_FILES
    )


def render_index(wiki_dir: Path) -> str:
    """Build the index.md content from the current state of ``wiki_dir``.

    Scans for ``*.md`` files (excluding ``index.md`` itself), parses
    each frontmatter, and emits an alphabetised list keyed by slug.
    Pages with malformed frontmatter are silently skipped — ``outmem
    lint`` catches them separately so the index can render against a
    partially-broken wiki without crashing.
    """
    entries: list[str] = []
    for path in editorial_pages(wiki_dir):
        try:
            frontmatter, _ = parse_wiki_page(path.read_text(encoding="utf-8"))
        except Exception:
            # Malformed page — skip; lint will surface it.
            continue
        slug = frontmatter.slug
        title = frontmatter.title
        tags = frontmatter.tags
        line = f"- [[{slug}]] — {title}"
        if tags:
            line += f" ({', '.join(tags)})"
        entries.append(line)

    body = "\n".join(entries) if entries else "_(no pages yet)_"
    return f"# {INDEX_TITLE}\n\n{len(entries)} page{'' if len(entries) == 1 else 's'}.\n\n{body}\n"


def index_page_text(wiki_dir: Path) -> str:
    """Render the full ``index.md`` file with frontmatter + body."""
    from outmem.frontmatter import WikiFrontmatter, serialize_wiki_page

    fm = WikiFrontmatter(
        title=INDEX_TITLE,
        slug=INDEX_SLUG,
        tags=["index"],
        extra={"generated": True},
    )
    body = render_index(wiki_dir)
    return serialize_wiki_page(fm, body)
