"""Slugs and Obsidian-style wikilinks.

A *slug* is the filename-safe identifier for a wiki page: lowercase
alphanumeric characters and hyphens only, no leading or trailing hyphen.
It corresponds to ``wiki/<slug>.md`` on disk and ``/wiki/<slug>`` in the
dashboard URL space.

A *wikilink* is an Obsidian-style ``[[slug]]`` or ``[[slug|display text]]``
reference inside a markdown body. :func:`extract_wikilinks` is used by
the linter and the backlink index to compute the graph; the dashboard
has its own markdown-it-flavoured rewriter in
:mod:`outmem.dashboard.service`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from outmem.exceptions import SlugError

# Match a single wikilink. Group 1 is the slug, group 2 is the optional
# display override (empty if not present).
_WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]*))?\]\]")

# Slug rules: lowercase ASCII letters, digits, hyphens; not empty; no
# leading or trailing hyphen; no consecutive hyphens.
_SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


@dataclass(frozen=True)
class Wikilink:
    """A single ``[[slug]]`` or ``[[slug|display]]`` reference.

    ``display`` is the literal text the renderer should show. It falls
    back to ``slug`` when no ``|display`` segment is present.
    """

    slug: str
    display: str
    raw: str  # the original ``[[…]]`` substring, useful for replacement


def validate_slug(slug: str) -> None:
    """Raise :class:`SlugError` if ``slug`` is not a valid slug.

    The rules are deliberately strict: lowercase ASCII alphanumerics
    and single hyphens only, no leading or trailing hyphen.
    """
    if not isinstance(slug, str):
        raise SlugError(f"slug must be a string, got {type(slug).__name__}.")
    if not slug:
        raise SlugError("slug is empty.")
    if not _SLUG_RE.match(slug):
        raise SlugError(
            f"slug {slug!r} is not valid: lowercase ASCII alphanumerics "
            "and single hyphens only, no leading/trailing hyphen."
        )


def extract_wikilinks(body: str) -> list[Wikilink]:
    """Return every ``[[…]]`` reference in ``body``, in order of appearance.

    Slugs are returned exactly as written in the source — they are *not*
    normalised. Obsidian links to a literal file name; the resolver
    (lint / backlinks) decides what to do with invalid slugs.
    """
    out: list[Wikilink] = []
    for match in _WIKILINK_RE.finditer(body):
        slug = match.group(1).strip()
        display = (match.group(2) or slug).strip()
        out.append(Wikilink(slug=slug, display=display, raw=match.group(0)))
    return out
