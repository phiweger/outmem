"""Tests for the auto-maintained ``wiki/index.md``."""

from __future__ import annotations

from pathlib import Path

import pytest

from outmem.frontmatter import parse_wiki_page
from outmem.index import (
    INDEX_FILENAME,
    INDEX_SLUG,
    index_page_text,
    load_editorial_pages,
    navigate_index,
    render_index,
)
from outmem.store import WikiStore


def _make_page(pages_dir: Path, slug: str, title: str, *, tags: list[str] | None = None) -> None:
    tags_yaml = "[]" if not tags else "[" + ", ".join(tags) + "]"
    pages_dir.mkdir(parents=True, exist_ok=True)
    (pages_dir / f"{slug}.md").write_text(
        f"---\ntitle: {title}\nslug: {slug}\ntags: {tags_yaml}\n---\n\nbody\n",
        encoding="utf-8",
    )


def test_render_index_empty(tmp_path: Path) -> None:
    pages = tmp_path / "wiki" / "pages"
    pages.mkdir(parents=True)
    out = render_index(pages)
    assert "no pages yet" in out
    assert "0 pages" in out


def test_render_index_alphabetised(tmp_path: Path) -> None:
    pages = tmp_path / "wiki" / "pages"
    _make_page(pages, "pricing-formula", "Pricing formula", tags=["pricing"])
    _make_page(pages, "acme-msa", "Acme MSA", tags=["contracts"])
    out = render_index(pages)
    # acme-msa comes before pricing-formula
    acme_pos = out.index("acme-msa")
    pricing_pos = out.index("pricing-formula")
    assert acme_pos < pricing_pos
    assert "[[acme-msa]] — Acme MSA (contracts)" in out
    assert "[[pricing-formula]] — Pricing formula (pricing)" in out
    assert "2 pages" in out


def test_render_index_excludes_itself(tmp_path: Path) -> None:
    pages = tmp_path / "wiki" / "pages"
    _make_page(pages, "alpha", "Alpha")
    # Pre-existing index.md sits at the wiki root, not under pages/, so it
    # can't accidentally appear in its own listing.
    (pages.parent / INDEX_FILENAME).write_text("stale", encoding="utf-8")
    out = render_index(pages)
    assert "[[index]]" not in out


def test_render_index_skips_malformed_pages(tmp_path: Path) -> None:
    pages = tmp_path / "wiki" / "pages"
    _make_page(pages, "alpha", "Alpha")
    (pages / "broken.md").write_text("no frontmatter here", encoding="utf-8")
    out = render_index(pages)
    assert "[[alpha]]" in out
    assert "[[broken]]" not in out


def test_index_page_text_round_trips(tmp_path: Path) -> None:
    pages = tmp_path / "wiki" / "pages"
    _make_page(pages, "alpha", "Alpha")
    text = index_page_text(pages)
    fm, body = parse_wiki_page(text)
    assert fm.slug == INDEX_SLUG
    assert fm.extra.get("generated") is True
    assert "[[alpha]]" in body


# ---------------------------------------------------------------------------
# navigate_index — one-level TOC navigation over a flat slug list
# ---------------------------------------------------------------------------


_NS_SLUGS = [
    "pricing-formula",
    "abx:penicillin",
    "abx:ceftriaxone",
    "abx:side-effects:rash",
]


def test_navigate_index_root() -> None:
    level = navigate_index(_NS_SLUGS)
    assert level.prefix == ""
    assert level.namespaces == [("abx", 3)]  # 3 pages anywhere under abx:
    assert level.pages == ["pricing-formula"]  # the only top-level leaf


def test_navigate_index_drill() -> None:
    level = navigate_index(_NS_SLUGS, "abx")
    assert level.prefix == "abx"
    assert level.namespaces == [("abx:side-effects", 1)]
    assert level.pages == ["abx:ceftriaxone", "abx:penicillin"]  # sorted


def test_navigate_index_tolerates_trailing_colon() -> None:
    level = navigate_index(_NS_SLUGS, "abx:")
    assert level.prefix == "abx"
    assert level.pages == ["abx:ceftriaxone", "abx:penicillin"]


def test_navigate_index_page_and_namespace_coexist() -> None:
    # ``abx`` is both a leaf page (abx.md) and a namespace (abx/…).
    slugs = ["abx", "abx:penicillin"]
    root = navigate_index(slugs)
    assert root.pages == ["abx"]
    assert root.namespaces == [("abx", 1)]
    drill = navigate_index(slugs, "abx")
    assert drill.pages == ["abx:penicillin"]


def test_navigate_index_empty_and_unknown_prefix() -> None:
    assert navigate_index([]).namespaces == []
    miss = navigate_index(_NS_SLUGS, "nope")
    assert miss.namespaces == [] and miss.pages == []


def test_store_index_tree_round_trips(tmp_path: Path) -> None:
    store = WikiStore.init(tmp_path / "w")
    store.write_page("pricing-formula", title="P", body="b")
    store.write_page("abx:penicillin", title="Pen", body="b")
    root = store.index_tree()
    assert root.namespaces == [("abx", 1)]
    assert root.pages == ["pricing-formula"]


# ---------------------------------------------------------------------------
# WikiStore integration
# ---------------------------------------------------------------------------


def test_init_creates_no_index_until_first_write(tmp_path: Path) -> None:
    store = WikiStore.init(tmp_path / "w")
    assert not (store.wiki_path / INDEX_FILENAME).exists()


def test_write_page_creates_index_in_same_commit(tmp_path: Path) -> None:
    store = WikiStore.init(tmp_path / "w")
    store.write_page("alpha", title="Alpha", body="body")
    index = store.wiki_path / INDEX_FILENAME
    assert index.exists()
    assert "[[alpha]]" in index.read_text()


def test_write_page_index_updated_on_each_write(tmp_path: Path) -> None:
    store = WikiStore.init(tmp_path / "w")
    store.write_page("alpha", title="Alpha", body="body")
    store.write_page("beta", title="Beta", body="body")
    index = (store.wiki_path / INDEX_FILENAME).read_text()
    assert "[[alpha]]" in index
    assert "[[beta]]" in index


def test_extend_page_refreshes_index(tmp_path: Path) -> None:
    store = WikiStore.init(tmp_path / "w")
    store.write_page("alpha", title="Original", body="body")
    store.extend_page("alpha", body="new body")
    # extend doesn't change title or tags, so the index line is the same
    # — but the commit should still touch index.md, verified by git.
    import subprocess

    out = subprocess.run(
        ["git", "log", "--name-only", "-2", "--format="],
        cwd=str(store.root),
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    files = {line.strip() for line in out.splitlines() if line.strip()}
    assert "wiki/pages/alpha.md" in files
    assert "wiki/index.md" in files


def test_write_index_slug_rejected(tmp_path: Path) -> None:
    import pytest

    from outmem.exceptions import OutmemError

    store = WikiStore.init(tmp_path / "w")
    with pytest.raises(OutmemError, match="reserved 'index' slug"):
        store.write_page(INDEX_SLUG, title="x", body="x")


def test_extend_index_slug_rejected(tmp_path: Path) -> None:
    import pytest

    from outmem.exceptions import OutmemError

    store = WikiStore.init(tmp_path / "w")
    store.write_page("alpha", title="a", body="b")  # makes the index exist
    with pytest.raises(OutmemError, match="reserved 'index' slug"):
        store.extend_page(INDEX_SLUG, body="x")


def test_list_slugs_hides_index(tmp_path: Path) -> None:
    store = WikiStore.init(tmp_path / "w")
    store.write_page("alpha", title="Alpha", body="body")
    assert store.list_slugs() == ["alpha"]
    assert "index" not in store.list_slugs()


def test_rebuild_index_after_manual_edit_picks_up_new_page(tmp_path: Path) -> None:
    """User edits wiki/ directly (Obsidian, vim, etc.). Index is stale.
    rebuild_index() picks up the manual edit and commits a single
    `index: rebuild` commit."""
    from datetime import UTC, datetime

    from outmem.frontmatter import WikiFrontmatter, serialize_wiki_page
    from outmem.git_ops import add, commit_as, log_since

    store = WikiStore.init(tmp_path / "w")

    # Hand-place a new wiki page WITHOUT going through write_page.
    now = datetime.now(UTC).replace(microsecond=0)
    fm = WikiFrontmatter(
        title="Manually added",
        slug="manual",
        provenance=[],
        created=now,
        updated=now,
        tags=[],
        extra={},
    )
    page = serialize_wiki_page(fm, "Body added by hand.\n")
    (store.pages_path / "manual.md").write_text(page, encoding="utf-8")
    # Commit the manual edit so the workspace is clean before rebuild.
    add(store.root, ["wiki/pages/manual.md"])
    commit_as(
        store.root,
        message="manual edit",
        author_name="alice",
        author_email="alice@example.com",
    )
    # index.md doesn't yet know about `manual`.
    index_path = store.wiki_path / "index.md"
    assert (not index_path.exists()) or "manual" not in index_path.read_text()

    sha = store.rebuild_index()
    assert sha is not None
    assert "manual" in index_path.read_text()
    log = log_since(store.root)
    assert log[0].subject == "index: rebuild"


def test_rebuild_index_no_op_when_in_sync(tmp_path: Path) -> None:
    """write_page already kept the index current; a follow-up rebuild
    is a no-op and produces no commit."""
    store = WikiStore.init(tmp_path / "w")
    store.write_page("alpha", title="Alpha", body="body")
    sha = store.rebuild_index()
    assert sha is None


def test_rebuild_index_no_commit_leaves_dirty_tree(tmp_path: Path) -> None:
    """commit=False writes the file but doesn't produce a commit.
    Useful for callers (pre-commit hook) that want the rebuilt index
    in the human's commit, not a separate one."""
    from datetime import UTC, datetime

    from outmem.frontmatter import WikiFrontmatter, serialize_wiki_page
    from outmem.git_ops import add, commit_as, path_is_dirty

    store = WikiStore.init(tmp_path / "w")
    now = datetime.now(UTC).replace(microsecond=0)
    fm = WikiFrontmatter(
        title="Manual",
        slug="manual",
        provenance=[],
        created=now,
        updated=now,
        tags=[],
        extra={},
    )
    (store.pages_path / "manual.md").write_text(
        serialize_wiki_page(fm, "body\n"), encoding="utf-8"
    )
    add(store.root, ["wiki/pages/manual.md"])
    commit_as(
        store.root,
        message="manual edit",
        author_name="alice",
        author_email="alice@example.com",
    )

    sha = store.rebuild_index(commit=False)
    assert sha is None
    # index.md was rewritten on disk and is now dirty vs HEAD.
    assert path_is_dirty(store.root, "wiki/index.md")


def test_index_does_not_count_as_backlink(tmp_path: Path) -> None:
    store = WikiStore.init(tmp_path / "w")
    store.write_page("alpha", title="Alpha", body="body")
    store.write_page("beta", title="Beta", body="body")
    # Each page is wikilinked from index.md, but the index is generated
    # so its links shouldn't count as backlinks.
    assert store.backlinks("alpha") == ()
    assert store.backlinks("beta") == ()


def test_editorial_backlinks_still_work(tmp_path: Path) -> None:
    store = WikiStore.init(tmp_path / "w")
    store.write_page("acme-msa", title="Acme", body="x")
    store.write_page("pricing", title="Pricing", body="See [[acme-msa]] for terms.")
    assert store.backlinks("acme-msa") == ("pricing",)


# ---------------------------------------------------------------------------
# load_editorial_pages — one discovery + failure contract for every loader
# ---------------------------------------------------------------------------


def _write(pages: Path, name: str, text: str) -> Path:
    pages.mkdir(parents=True, exist_ok=True)
    p = pages / name
    p.write_text(text, encoding="utf-8")
    return p


def test_load_editorial_pages_splits_successes_from_failures(tmp_path: Path) -> None:
    pages = tmp_path / "wiki" / "pages"
    _make_page(pages, "good", "Good")
    _write(pages, "broken.md", "---\ntitle: T\nslug: broken\ntags: [a, [n]]\n---\n\nb\n")

    loaded, failures = load_editorial_pages(pages)
    assert [p.slug for p in loaded] == ["good"]
    assert [f.slug for f in failures] == ["broken"]
    assert failures[0].error  # carries the reason, for the caller to report


def test_load_editorial_pages_self_heals_the_repairable_shape(tmp_path: Path) -> None:
    """The imported-data case: an unquoted `: ` in the title. `read_page`
    already heals it, so every other loader must see the page too."""
    pages = tmp_path / "wiki" / "pages"
    _write(
        pages,
        "flu.md",
        "---\ntitle: Influenza (Teil 1): Erkrankungen\nslug: flu\n---\n\nbody\n",
    )
    loaded, failures = load_editorial_pages(pages)
    assert failures == []
    assert len(loaded) == 1
    assert loaded[0].repaired is True
    assert loaded[0].frontmatter.title.startswith("Influenza (Teil 1)")


def test_render_index_includes_a_self_healed_page(tmp_path: Path) -> None:
    """Regression: the TOC used to silently omit a page that read_page
    could serve, so the page existed everywhere except the index."""
    pages = tmp_path / "wiki" / "pages"
    _write(
        pages,
        "flu.md",
        "---\ntitle: Influenza (Teil 1): Erkrankungen\nslug: flu\n---\n\nbody\n",
    )
    assert "[[flu]]" in render_index(pages)


def test_render_index_warns_instead_of_silently_dropping(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    pages = tmp_path / "wiki" / "pages"
    _make_page(pages, "good", "Good")
    _write(pages, "broken.md", "---\ntitle: T\nslug: broken\ntags: [a, [n]]\n---\n\nb\n")

    with caplog.at_level(logging.WARNING):
        out = render_index(pages)
    assert "[[good]]" in out
    assert "[[broken]]" not in out
    assert "broken" in caplog.text  # no longer silent


def test_every_loader_agrees_on_a_self_healed_page(tmp_path: Path) -> None:
    """The altitude guarantee: one page, five readers, same verdict.

    A page that `read_page` self-heals must also be in the TOC, the
    backlink graph, the bm25 keyword net, and the semantic index — the
    bug class this refactor exists to close was each loader deciding
    independently.
    """
    from outmem._store.semantic import load_for_index
    from outmem.backlinks import _build_graph
    from outmem.optimize.blocks import _read_page_rows

    store = WikiStore.init(tmp_path / "w")
    store.write_page("target", title="Target", body="target body\n")
    rel = f"{store.config.wiki_dir}/pages/flu.md"
    (store.root / rel).write_text(
        "---\ntitle: Influenza (Teil 1): Erkrankungen\nslug: flu\n---\n\n"
        "see [[target]]\n",
        encoding="utf-8",
    )

    # 1. read_page (the reference policy)
    assert store.read("flu").title.startswith("Influenza")
    # 2. the TOC
    assert "[[flu]]" in render_index(store.pages_path)
    # 3. the backlink graph — flu links to target
    assert "flu" in _build_graph(store.pages_path).get("target", ())
    # 4. the bm25 keyword net (backs the DEFAULT rerank(bm25) strategy)
    assert "flu" in {slug for slug, _ in _read_page_rows(store)}
    # 5. the semantic indexer
    assert load_for_index(store, rel) is not None
