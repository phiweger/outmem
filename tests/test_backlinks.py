"""Tests for ``outmem.backlinks``."""

from __future__ import annotations

from pathlib import Path

import pytest

from outmem.backlinks import BacklinkCache
from outmem.state import BACKLINKS_FILE, OutmemState


@pytest.fixture
def wiki_with_links(tmp_path: Path) -> Path:
    """Wiki containing a small backlink graph::

    pricing-formula → [[acme-msa]]
    pricing-formula → [[discounts|discount policy]]
    acme-msa        → [[pricing-formula]]
    discounts       → (no links out)
    """
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    (wiki / "pricing-formula.md").write_text(
        "---\ntitle: x\nslug: pricing-formula\n---\n\n"
        "Reference: [[acme-msa]] and [[discounts|discount policy]].\n",
        encoding="utf-8",
    )
    (wiki / "acme-msa.md").write_text(
        "---\ntitle: x\nslug: acme-msa\n---\n\nSee [[pricing-formula]].\n",
        encoding="utf-8",
    )
    (wiki / "discounts.md").write_text(
        "---\ntitle: x\nslug: discounts\n---\n\nNo outbound links.\n",
        encoding="utf-8",
    )
    return tmp_path


def _make_cache(root: Path) -> BacklinkCache:
    state = OutmemState(root)
    return BacklinkCache(state=state, wiki_dir=root / "wiki")


def test_rebuild_produces_expected_graph(wiki_with_links: Path) -> None:
    cache = _make_cache(wiki_with_links)
    snapshot = cache.rebuild("head1")
    assert snapshot.head == "head1"
    assert snapshot.graph["acme-msa"] == ("pricing-formula",)
    assert snapshot.graph["pricing-formula"] == ("acme-msa",)
    assert snapshot.graph["discounts"] == ("pricing-formula",)


def test_referrers_for_unlinked_slug(wiki_with_links: Path) -> None:
    cache = _make_cache(wiki_with_links)
    cache.rebuild("h")
    assert cache.referrers("nonexistent", "h") == ()


def test_referrers_drops_self_links(tmp_path: Path) -> None:
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    (wiki / "loop.md").write_text(
        "---\ntitle: x\nslug: loop\n---\n\nLink to [[loop]].\n",
        encoding="utf-8",
    )
    cache = _make_cache(tmp_path)
    snapshot = cache.rebuild("h")
    assert snapshot.graph == {}


def test_cache_persists_to_disk(wiki_with_links: Path) -> None:
    cache = _make_cache(wiki_with_links)
    cache.rebuild("head1")
    path = wiki_with_links / ".outmem" / BACKLINKS_FILE
    assert path.exists()
    cache2 = _make_cache(wiki_with_links)
    snapshot = cache2.graph_for("head1")
    assert snapshot.head == "head1"
    assert snapshot.graph["acme-msa"] == ("pricing-formula",)


def test_cache_rebuilds_when_head_changes(wiki_with_links: Path) -> None:
    cache = _make_cache(wiki_with_links)
    cache.rebuild("head1")
    # Mutate the wiki so a rebuild would yield a different graph, then
    # ask for a different head.
    (wiki_with_links / "wiki/discounts.md").write_text(
        "---\ntitle: x\nslug: discounts\n---\n\nNow links to [[acme-msa]].\n",
        encoding="utf-8",
    )
    snapshot = cache.graph_for("head2")
    assert "discounts" in snapshot.graph["acme-msa"] or snapshot.graph.get("acme-msa") == (
        "discounts",
        "pricing-formula",
    )


def test_cache_returns_memo_on_same_head(wiki_with_links: Path) -> None:
    cache = _make_cache(wiki_with_links)
    first = cache.rebuild("head1")
    # Delete the on-disk cache; the memo should still serve.
    (wiki_with_links / ".outmem" / BACKLINKS_FILE).unlink()
    second = cache.graph_for("head1")
    assert second is first


def test_invalidate_drops_memo(wiki_with_links: Path) -> None:
    cache = _make_cache(wiki_with_links)
    cache.rebuild("head1")
    cache.invalidate()
    # Wipe disk too — next call has nothing cached.
    (wiki_with_links / ".outmem" / BACKLINKS_FILE).unlink()
    # Mutate wiki to make the rebuild observable.
    (wiki_with_links / "wiki/new.md").write_text(
        "---\ntitle: x\nslug: new\n---\n\nlinks [[acme-msa]]\n", encoding="utf-8"
    )
    snapshot = cache.graph_for("head1")
    assert "new" in snapshot.graph["acme-msa"]


def test_empty_wiki_returns_empty_graph(tmp_path: Path) -> None:
    (tmp_path / "wiki").mkdir()
    cache = _make_cache(tmp_path)
    snapshot = cache.rebuild("head")
    assert snapshot.graph == {}


def test_missing_wiki_dir_returns_empty(tmp_path: Path) -> None:
    cache = BacklinkCache(state=OutmemState(tmp_path), wiki_dir=tmp_path / "noplace")
    snapshot = cache.rebuild("head")
    assert snapshot.graph == {}


def test_none_head_returns_empty_uncached(wiki_with_links: Path) -> None:
    cache = _make_cache(wiki_with_links)
    snapshot = cache.graph_for(None)
    assert snapshot.head == ""
    assert snapshot.graph == {}
