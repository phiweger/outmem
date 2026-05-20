"""Slugs and Obsidian-style wikilinks.

A *slug* is the filename-safe identifier for a wiki page. Each slug is
one or more *segments* joined by ``:``; each segment is lowercase ASCII
alphanumerics with single hyphens, no leading or trailing hyphen. So
``pricing-formula`` and ``abx:penicillin`` and ``abx:side-effects:misc``
are all valid; ``abx:`` and ``-abx`` and ``abx/penicillin`` are not.

A slug maps to a path under ``wiki/pages/`` by replacing each ``:`` with
``/`` and appending ``.md``:

* ``pricing-formula``         → ``wiki/pages/pricing-formula.md``
* ``abx:penicillin``          → ``wiki/pages/abx/penicillin.md``
* ``abx:side-effects:misc``   → ``wiki/pages/abx/side-effects/misc.md``

The reverse mapping (:func:`relpath_to_slug`) is the path with
``.md`` stripped and ``/`` replaced by ``:``. Wiki infrastructure
(``index.md``, ``AGENTS.md``, ``CONTRIBUTORS.md``, ``sources/``) lives
at the wiki root, not under ``pages/``, so it can never collide with a
page slug.

A *wikilink* is an Obsidian-style ``[[slug]]`` or ``[[slug|display text]]``
reference inside a markdown body — including slugs with ``:`` segments,
e.g. ``[[abx:penicillin]]``. The extractor doesn't care about the
internal structure of the slug; resolution happens at lookup time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from outmem.exceptions import SlugError

# Match a single wikilink. Group 1 is the slug, group 2 is the optional
# display override (empty if not present). The slug character class
# excludes ``]`` and ``|`` only — ``:`` is allowed so namespaced slugs
# like ``[[abx:penicillin]]`` parse natively.
_WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]*))?\]\]")

# Slug grammar: one or more segments joined by ``:``. Each segment is
# lowercase ASCII alphanumerics with single hyphens, no leading or
# trailing hyphen, no consecutive hyphens.
_SEGMENT = r"[a-z0-9]+(?:-[a-z0-9]+)*"
_SLUG_RE = re.compile(rf"^{_SEGMENT}(?::{_SEGMENT})*$")

# Subdirectory of the wiki root where editorial pages live. Sibling to
# ``wiki/sources/`` (source documents), ``wiki/index.md`` (auto-generated
# catalog), ``wiki/AGENTS.md`` (per-wiki conventions), and
# ``wiki/CONTRIBUTORS.md`` (known identities).
PAGES_DIR = "pages"


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

    The rules are deliberately strict: one or more colon-separated
    segments, each segment lowercase ASCII alphanumerics with single
    hyphens only, no leading or trailing hyphen.
    """
    if not isinstance(slug, str):
        raise SlugError(f"slug must be a string, got {type(slug).__name__}.")
    if not slug:
        raise SlugError("slug is empty.")
    if not _SLUG_RE.match(slug):
        raise SlugError(
            f"slug {slug!r} is not valid: one or more ``:``-separated segments, "
            "each lowercase ASCII alphanumerics with single hyphens only, no "
            "leading/trailing hyphen."
        )


def slug_to_relpath(slug: str) -> Path:
    """Map a slug to its path relative to ``wiki/pages/``.

    The slug is assumed to be already validated by :func:`validate_slug`.

    >>> slug_to_relpath("pricing-formula").as_posix()
    'pricing-formula.md'
    >>> slug_to_relpath("abx:penicillin").as_posix()
    'abx/penicillin.md'
    >>> slug_to_relpath("abx:side-effects:misc").as_posix()
    'abx/side-effects/misc.md'
    """
    return Path(slug.replace(":", "/") + ".md")


def relpath_to_slug(relpath: Path) -> str:
    """Map a path (relative to ``wiki/pages/``, including ``.md``) to its slug.

    The inverse of :func:`slug_to_relpath`. Accepts both forward and
    backslash separators so Windows-style paths round-trip cleanly.

    >>> relpath_to_slug(Path("pricing-formula.md"))
    'pricing-formula'
    >>> relpath_to_slug(Path("abx/penicillin.md"))
    'abx:penicillin'
    """
    stem = relpath.with_suffix("").as_posix()
    return stem.replace("/", ":")


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
