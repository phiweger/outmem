"""Tests for the auto-maintained ``wiki/index.md``."""

from __future__ import annotations

from pathlib import Path

from outmem.frontmatter import parse_wiki_page
from outmem.index import INDEX_FILENAME, INDEX_SLUG, index_page_text, render_index
from outmem.store import WikiStore


def _make_page(wiki_dir: Path, slug: str, title: str, *, tags: list[str] | None = None) -> None:
    tags_yaml = "[]" if not tags else "[" + ", ".join(tags) + "]"
    wiki_dir.mkdir(parents=True, exist_ok=True)
    (wiki_dir / f"{slug}.md").write_text(
        f"---\ntitle: {title}\nslug: {slug}\ntags: {tags_yaml}\n---\n\nbody\n",
        encoding="utf-8",
    )


def test_render_index_empty(tmp_path: Path) -> None:
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    out = render_index(wiki)
    assert "no pages yet" in out
    assert "0 pages" in out


def test_render_index_alphabetised(tmp_path: Path) -> None:
    wiki = tmp_path / "wiki"
    _make_page(wiki, "pricing-formula", "Pricing formula", tags=["pricing"])
    _make_page(wiki, "acme-msa", "Acme MSA", tags=["contracts"])
    out = render_index(wiki)
    # acme-msa comes before pricing-formula
    acme_pos = out.index("acme-msa")
    pricing_pos = out.index("pricing-formula")
    assert acme_pos < pricing_pos
    assert "[[acme-msa]] — Acme MSA (contracts)" in out
    assert "[[pricing-formula]] — Pricing formula (pricing)" in out
    assert "2 pages" in out


def test_render_index_excludes_itself(tmp_path: Path) -> None:
    wiki = tmp_path / "wiki"
    _make_page(wiki, "alpha", "Alpha")
    # Pre-existing index.md should not appear in its own listing.
    (wiki / INDEX_FILENAME).write_text("stale", encoding="utf-8")
    out = render_index(wiki)
    assert "[[index]]" not in out


def test_render_index_skips_malformed_pages(tmp_path: Path) -> None:
    wiki = tmp_path / "wiki"
    _make_page(wiki, "alpha", "Alpha")
    (wiki / "broken.md").write_text("no frontmatter here", encoding="utf-8")
    out = render_index(wiki)
    assert "[[alpha]]" in out
    assert "[[broken]]" not in out


def test_index_page_text_round_trips(tmp_path: Path) -> None:
    wiki = tmp_path / "wiki"
    _make_page(wiki, "alpha", "Alpha")
    text = index_page_text(wiki)
    fm, body = parse_wiki_page(text)
    assert fm.slug == INDEX_SLUG
    assert fm.extra.get("generated") is True
    assert "[[alpha]]" in body


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
    assert "wiki/alpha.md" in files
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
    (store.wiki_path / "manual.md").write_text(page, encoding="utf-8")
    # Commit the manual edit so the workspace is clean before rebuild.
    add(store.root, ["wiki/manual.md"])
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
    (store.wiki_path / "manual.md").write_text(
        serialize_wiki_page(fm, "body\n"), encoding="utf-8"
    )
    add(store.root, ["wiki/manual.md"])
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
