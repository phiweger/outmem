"""``outmem import`` — bring an existing markdown vault into ``wiki/pages/``.

The vault — typically an Obsidian directory tree — is recursively scanned
for ``*.md`` files. Each note becomes ``wiki/pages/<slug-as-relpath>.md``
with generated frontmatter; wikilinks are rewritten so they target outmem
slugs; the whole import lands in one ``import: <vault-name>`` commit.

Public entry point: :func:`import_vault`. The CLI subcommand
``outmem import <dir>`` is a thin wrapper.

Design notes
------------

- **Flat slug namespace**. outmem's wiki is flat; collisions are
  resolved by prefixing with the parent directory (deterministic,
  sorted-order tiebreak).
- **Wikilink rewriting**. Obsidian's ``[[Note Name]]`` gets rewritten
  to ``[[note-slug|Note Name]]`` — preserving the display text while
  making the slug machine-resolvable. Wikilinks whose target doesn't
  resolve to any imported slug are left untouched; ``outmem lint``
  surfaces them.
- **Frontmatter**. ``title`` from the first H1 (or humanised filename
  fallback), ``slug`` from the resolved name, ``created`` / ``updated``
  from the file mtime, ``provenance`` recording the original vault-
  relative path so the audit trail stays honest about origin.
- **One commit**. The whole import is atomic; revert is a single
  ``git reset --hard HEAD^``.
- **Refuses non-empty wikis** unless ``force=True``. A re-import is
  semantically "the vault is the canonical source, clobber the wiki" —
  the agent's intermediate edits would be lost, so we require the
  caller to acknowledge.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from outmem.exceptions import OutmemError
from outmem.frontmatter import WikiFrontmatter, serialize_wiki_page
from outmem.index import editorial_pages
from outmem.slug import PAGES_DIR, slug_to_relpath, validate_slug

if TYPE_CHECKING:
    from outmem.store import WikiStore

log = logging.getLogger(__name__)

_WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(#[^|\]]*)?(?:\|([^\]]*))?\]\]")
_H1_RE = re.compile(r"^\s*#\s+(.+?)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class ImportSummary:
    """Result of one :func:`import_vault` call."""

    pages_imported: int
    slug_collisions: tuple[tuple[str, str], ...]  # (original_path, resolved_slug)
    wikilinks_rewritten: int
    wikilinks_unresolved: int


@dataclass
class _Candidate:
    source_path: Path  # absolute
    rel_source: Path  # relative to vault root
    body: str  # original markdown body
    title: str
    slug: str = ""  # filled in by collision-resolution pass
    mtime: datetime = field(default_factory=datetime.now)


def import_vault(
    store: WikiStore,
    source: Path,
    *,
    force: bool = False,
) -> ImportSummary:
    """Import every ``*.md`` under ``source`` into ``store``'s wiki.

    Raises :class:`OutmemError` if ``source`` isn't a directory or if
    the target wiki already has pages and ``force`` is False.
    """
    if not source.is_dir():
        raise OutmemError(f"import source is not a directory: {source}")

    existing = editorial_pages(store.pages_path)
    if existing and not force:
        raise OutmemError(
            f"target wiki already has {len(existing)} page(s); "
            "pass force=True to overwrite."
        )

    candidates = _collect_candidates(source)
    if not candidates:
        raise OutmemError(f"no *.md files found under {source}")

    _resolve_slugs(candidates)
    slug_by_basename = _build_link_index(candidates)

    collisions: list[tuple[str, str]] = []
    rewrites_total = 0
    unresolved_total = 0
    written: list[str] = []

    for c in candidates:
        # Defence in depth: every produced slug must pass the canonical
        # check, else outmem itself refuses to read the page back.
        validate_slug(c.slug)
        rewritten_body, rewrites, unresolved = _rewrite_wikilinks(
            c.body, slug_by_basename
        )
        rewrites_total += rewrites
        unresolved_total += unresolved

        frontmatter = WikiFrontmatter(
            title=c.title,
            slug=c.slug,
            created=c.mtime,
            updated=c.mtime,
            provenance=[{"path": str(c.rel_source), "source": "obsidian-import"}],
        )
        rendered = serialize_wiki_page(frontmatter, rewritten_body)
        dest = store.pages_path / slug_to_relpath(c.slug)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(rendered, encoding="utf-8")
        written.append(
            f"{store.config.wiki_dir}/{PAGES_DIR}/{slug_to_relpath(c.slug).as_posix()}"
        )

        if c.slug != _slugify(c.rel_source.stem):
            collisions.append((str(c.rel_source), c.slug))

    # Regenerate the index + commit everything in one go.
    store.rebuild_index(commit=False)
    written.append(f"{store.config.wiki_dir}/index.md")
    store._commit_paths(written, subject=f"import: {source.name}")

    return ImportSummary(
        pages_imported=len(candidates),
        slug_collisions=tuple(collisions),
        wikilinks_rewritten=rewrites_total,
        wikilinks_unresolved=unresolved_total,
    )


# ---------------------------------------------------------------------------
# Scanning / candidate building
# ---------------------------------------------------------------------------


def _collect_candidates(source: Path) -> list[_Candidate]:
    """Walk ``source`` for ``*.md``, ignoring hidden directories."""
    out: list[_Candidate] = []
    for path in sorted(source.rglob("*.md")):
        rel = path.relative_to(source)
        # Skip files whose path includes a hidden component
        # (.obsidian/, .trash/, .git/, etc.).
        if any(part.startswith(".") for part in rel.parts):
            continue
        try:
            body = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            log.warning("import: unreadable %s: %s", path, exc)
            continue
        title = _extract_title(body, fallback=rel.stem)
        mtime = datetime.fromtimestamp(path.stat().st_mtime).astimezone()
        out.append(
            _Candidate(
                source_path=path,
                rel_source=rel,
                body=body,
                title=title,
                mtime=mtime,
            )
        )
    return out


def _extract_title(body: str, *, fallback: str) -> str:
    """Pull the first ``# H1`` line, falling back to a humanised filename."""
    match = _H1_RE.search(body)
    if match:
        return match.group(1).strip()
    # Humanise: "my-great-note" → "My great note", "Pricing Formula" stays.
    return fallback.replace("_", " ").replace("-", " ").strip() or fallback


# ---------------------------------------------------------------------------
# Slug resolution
# ---------------------------------------------------------------------------


def _slugify(text: str) -> str:
    """Filesystem-safe slug from arbitrary text.

    Lowercases, strips diacritics, collapses whitespace / underscores to
    hyphens, drops everything not ``[a-z0-9-]``, trims edge hyphens.
    Returns an empty string for un-slugifiable input (e.g. emoji-only).
    """
    normalised = unicodedata.normalize("NFKD", text)
    ascii_only = "".join(c for c in normalised if not unicodedata.combining(c))
    lower = ascii_only.lower()
    gapped = re.sub(r"[\s_]+", "-", lower)
    filtered = re.sub(r"[^a-z0-9-]", "", gapped)
    collapsed = re.sub(r"-+", "-", filtered)
    return collapsed.strip("-")


def _resolve_slugs(candidates: list[_Candidate]) -> None:
    """Assign a unique slug to every candidate, mutating in place.

    Strategy:

    1. Compute the bare ``slugify(stem)`` for each.
    2. On collision, prefix with the parent directory slug; repeat if
       the prefixed form also collides (walking further up the path).
    3. Final fallback: suffix with ``-2``, ``-3``, … so we never throw.
    """
    taken: set[str] = set()
    for c in candidates:
        candidate_slug = _propose_slug(c)
        # Walk the path upward to disambiguate on collision.
        parts = list(c.rel_source.parts[:-1])  # parent dirs, leaf-first
        i = len(parts)
        while candidate_slug in taken or not candidate_slug:
            if i > 0:
                i -= 1
                prefix = _slugify(parts[i])
                if not prefix:
                    continue
                candidate_slug = f"{prefix}-{_propose_slug(c)}"
            else:
                # All parent dirs consumed; numeric suffix as last resort.
                n = 2
                base = candidate_slug or "page"
                while f"{base}-{n}" in taken:
                    n += 1
                candidate_slug = f"{base}-{n}"
                break
        c.slug = candidate_slug
        taken.add(candidate_slug)


def _propose_slug(c: _Candidate) -> str:
    """The bare slug from the file stem; guarded against empties."""
    proposed = _slugify(c.rel_source.stem)
    return proposed or "page"


# ---------------------------------------------------------------------------
# Wikilink rewriting
# ---------------------------------------------------------------------------


def _build_link_index(candidates: list[_Candidate]) -> dict[str, str]:
    """Map ``basename`` → ``slug`` so wikilinks resolve by file name.

    Obsidian's typical link form is ``[[Note Name]]`` (no path), which
    Obsidian resolves to whatever file ``Note Name.md`` lives in. We
    mirror that: indexed by the slugified stem.

    On collision (two notes with the same basename in different
    directories), the first one in sorted order wins the basename key;
    others are reachable only by their resolved slug.
    """
    index: dict[str, str] = {}
    for c in candidates:
        key = _slugify(c.rel_source.stem)
        index.setdefault(key, c.slug)
        # Also index by the resolved slug itself so explicit
        # ``[[parent-prefix-name]]`` references work.
        index.setdefault(c.slug, c.slug)
    return index


def _rewrite_wikilinks(
    body: str, slug_by_basename: dict[str, str]
) -> tuple[str, int, int]:
    """Rewrite ``[[X]]`` and ``[[X|display]]`` to use resolved slugs.

    Preserves display text by always emitting the ``[[slug|display]]``
    form. Header anchors (``[[X#section]]``) are kept attached to the
    rewritten slug. Unresolved links are returned unchanged; the
    counter lets the caller report how many.
    """
    rewrites = 0
    unresolved = 0

    def _sub(match: re.Match[str]) -> str:
        nonlocal rewrites, unresolved
        target = match.group(1).strip()
        anchor = match.group(2) or ""
        display = (match.group(3) or target).strip()
        key = _slugify(target)
        slug = slug_by_basename.get(key)
        if slug is None:
            unresolved += 1
            return match.group(0)
        rewrites += 1
        return f"[[{slug}{anchor}|{display}]]"

    return _WIKILINK_RE.sub(_sub, body), rewrites, unresolved


