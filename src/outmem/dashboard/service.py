"""Rendering helpers — markdown → HTML with wikilink rewriting.

Wikilinks (``[[slug]]`` / ``[[slug|display]]``) are rewritten into
markdown link syntax *before* feeding the body to markdown-it-py, so
they round-trip through the standard markdown pipeline rather than
requiring a custom HTML post-processor (and the HTML escape pass stays
clean).

``markdown-it-py`` is invoked with ``html=False`` so any raw HTML in
the wiki body is escaped, not rendered — this is the defence-in-depth
layer alongside the slug/display validation in :mod:`outmem.slug`.
"""

from __future__ import annotations

import re

from markdown_it import MarkdownIt

from outmem.exceptions import SlugError
from outmem.slug import validate_slug

# Wikilink pattern, same as outmem.slug but rebuilt locally so this
# module can stay self-contained for the render pipeline.
_WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]*))?\]\]")


def wikilinks_to_markdown(body: str, *, base: str = "/wiki/") -> str:
    """Rewrite ``[[slug]]`` references into markdown ``[label](url)``.

    Invalid slugs are left as literal ``[[…]]`` text so a typo is
    visible to the reader rather than swallowed by the renderer.
    """

    def _sub(match: re.Match[str]) -> str:
        slug = match.group(1).strip()
        display = (match.group(2) or slug).strip()
        try:
            validate_slug(slug)
        except SlugError:
            return match.group(0)
        href = f"{base.rstrip('/')}/{slug}"
        # Escape ``]`` in the display so it doesn't terminate the
        # markdown link prematurely.
        safe_display = display.replace("]", "\\]")
        return f"[{safe_display}]({href})"

    return _WIKILINK_RE.sub(_sub, body)


def build_renderer() -> MarkdownIt:
    """A MarkdownIt instance configured for safe wiki rendering."""
    return MarkdownIt("commonmark", {"html": False, "breaks": False, "linkify": True})


def render_body(body: str, *, base: str = "/wiki/") -> str:
    """Render markdown body to HTML, rewriting wikilinks first."""
    rewritten = wikilinks_to_markdown(body, base=base)
    html: str = build_renderer().render(rewritten)
    return html
