"""Backlinks — the inverse wikilink graph, cached by HEAD SHA.

Computed once per HEAD change rather than on every dashboard render
(spec v0.5 §5). Persisted to ``.outmem/backlinks.json`` so cold starts
don't re-scan the wiki, and reads fall through to a rebuild when the
cache key doesn't match the current HEAD.

The graph maps ``slug -> [referrer_slug, …]`` — every wiki page that
contains a ``[[slug]]`` reference to the target. Self-links are
dropped (a page does not back-link to itself).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from outmem.slug import extract_wikilinks, relpath_to_slug
from outmem.state import BACKLINKS_FILE, OutmemState


@dataclass(frozen=True)
class BacklinkGraph:
    """Frozen graph snapshot tied to a specific HEAD."""

    head: str
    graph: dict[str, tuple[str, ...]]

    def referrers(self, slug: str) -> tuple[str, ...]:
        return self.graph.get(slug, ())


class BacklinkCache:
    """In-memory + on-disk cache for the wiki's backlink graph.

    When ``read_only=True`` the on-disk persist step is skipped — the
    cache becomes memo-only, valid for the lifetime of the process.
    Pair with :meth:`WikiStore.open(..., read_only=True)` to keep a
    curated wiki on a literally read-only filesystem usable.
    """

    def __init__(
        self,
        *,
        state: OutmemState,
        wiki_dir: Path,
        pages_dir: Path,
        read_only: bool = False,
    ) -> None:
        self._state = state
        self._wiki_dir = wiki_dir
        self._pages_dir = pages_dir
        self._read_only = read_only
        self._memo: BacklinkGraph | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def graph_for(self, head_sha: str | None) -> BacklinkGraph:
        """Return a graph valid for ``head_sha``.

        If the in-memory or on-disk cache was built against the same
        HEAD, return it. Otherwise rebuild by scanning ``wiki/``.

        ``head_sha=None`` (the repo has no commits yet) returns an
        empty graph without caching — there is no SHA to key on.
        """
        if head_sha is None:
            return BacklinkGraph(head="", graph={})

        if self._memo is not None and self._memo.head == head_sha:
            return self._memo

        cached = self._load_cached(head_sha)
        if cached is not None:
            self._memo = cached
            return cached

        rebuilt = self.rebuild(head_sha)
        return rebuilt

    def referrers(self, slug: str, head_sha: str | None) -> tuple[str, ...]:
        """Convenience: backlinks for a single slug at ``head_sha``."""
        return self.graph_for(head_sha).referrers(slug)

    def rebuild(self, head_sha: str) -> BacklinkGraph:
        """Walk ``wiki/pages/`` and recompute the graph from scratch.

        Writes back to ``.outmem/backlinks.json`` so the next cold
        start hits the cache. The persist is skipped when this cache
        was opened ``read_only=True`` — useful for consult-only
        consumers that must not touch the wiki's filesystem state.
        """
        graph = _build_graph(self._pages_dir)
        snapshot = BacklinkGraph(head=head_sha, graph=graph)
        self._memo = snapshot
        if not self._read_only:
            self._persist(snapshot)
        return snapshot

    def invalidate(self) -> None:
        """Drop the in-memory memo. The on-disk file is left intact."""
        self._memo = None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_cached(self, head_sha: str) -> BacklinkGraph | None:
        data = self._state.read_json(BACKLINKS_FILE)
        if not data:
            return None
        if data.get("head") != head_sha:
            return None
        raw_graph = data.get("graph")
        if not isinstance(raw_graph, dict):
            return None
        try:
            graph = {
                slug: tuple(referrers)
                for slug, referrers in raw_graph.items()
                if isinstance(slug, str)
                and isinstance(referrers, list)
                and all(isinstance(r, str) for r in referrers)
            }
        except (TypeError, ValueError):
            return None
        return BacklinkGraph(head=head_sha, graph=graph)

    def _persist(self, snapshot: BacklinkGraph) -> None:
        self._state.write_json(
            BACKLINKS_FILE,
            {
                "head": snapshot.head,
                "graph": {slug: list(refs) for slug, refs in snapshot.graph.items()},
            },
        )


def _build_graph(pages_dir: Path) -> dict[str, tuple[str, ...]]:
    """Inverse map of every wikilink referenced from ``wiki/pages/**/*.md``.

    Pages with frontmatter ``generated: true`` (currently just the
    auto-maintained ``index.md``) are ignored on the *source* side —
    their wikilinks are navigational, not editorial, and counting
    them would mean every page has at least one backlink from the
    index and no page is ever an orphan.
    """
    if not pages_dir.is_dir():
        return {}

    # Import lazily to avoid a frontmatter ↔ backlinks circular import.
    from outmem.frontmatter import parse_wiki_page

    # slug -> set of referrer slugs (set so duplicates collapse).
    inverse: dict[str, set[str]] = {}

    for page_path in sorted(pages_dir.rglob("*.md")):
        page_slug = relpath_to_slug(page_path.relative_to(pages_dir))
        try:
            text = page_path.read_text(encoding="utf-8")
        except OSError:
            continue
        try:
            frontmatter, body = parse_wiki_page(text)
        except Exception:
            # Malformed page — fall back to scanning the full file as body.
            body = text
            frontmatter = None
        if frontmatter is not None and frontmatter.extra.get("generated"):
            continue  # generated pages' wikilinks don't count as backlinks
        for link in extract_wikilinks(body):
            if link.slug == page_slug:
                continue  # self-links don't count
            inverse.setdefault(link.slug, set()).add(page_slug)

    return {slug: tuple(sorted(refs)) for slug, refs in sorted(inverse.items())}
