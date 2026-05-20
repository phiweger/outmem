"""Tests for ``outmem.search``."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from outmem.exceptions import OutmemError
from outmem.search import SearchHit, rg_available, search


@pytest.fixture
def wiki_root(tmp_path: Path) -> Path:
    """A small directory tree with two wiki pages and two raw files."""
    (tmp_path / "wiki" / "pages").mkdir(parents=True)
    (tmp_path / "raw").mkdir()
    (tmp_path / "wiki/pages/pricing-formula.md").write_text(
        "---\ntitle: Pricing formula\nslug: pricing-formula\n---\n\n"
        "The pricing formula uses cost-plus margin.\n"
        "Anchor: see [[acme-msa]] for the agreement.\n",
        encoding="utf-8",
    )
    (tmp_path / "wiki/pages/acme-msa.md").write_text(
        "---\ntitle: Acme MSA\nslug: acme-msa\n---\n\n"
        "Standard Master Service Agreement. Pricing terms in §4.\n",
        encoding="utf-8",
    )
    (tmp_path / "raw/pricing-deck.md").write_text(
        "Slide 3: pricing is cost-plus 35%.\n", encoding="utf-8"
    )
    (tmp_path / "raw/acme-msa.md").write_text(
        "Original Master Service Agreement text…\n", encoding="utf-8"
    )
    return tmp_path


def test_rg_available() -> None:
    assert rg_available()


def test_search_finds_term_in_wiki(wiki_root: Path) -> None:
    result = search("pricing", root=wiki_root / "wiki" / "pages")
    assert result.truncated is False
    assert any(hit.path == "pricing-formula.md" for hit in result.hits)


def test_search_returns_line_numbers(wiki_root: Path) -> None:
    result = search("Anchor", root=wiki_root / "wiki" / "pages")
    assert any(hit.line_number > 0 and "Anchor" in hit.text for hit in result.hits)


def test_search_case_insensitive(wiki_root: Path) -> None:
    result = search("PRICING", root=wiki_root / "wiki" / "pages", case_insensitive=True)
    assert len(result.hits) > 0


def test_search_fixed_strings(wiki_root: Path) -> None:
    result = search(
        "cost-plus",
        root=wiki_root / "wiki" / "pages",
        fixed_strings=True,
    )
    assert any("cost-plus" in hit.text for hit in result.hits)


def test_search_restricted_to_paths(wiki_root: Path) -> None:
    result = search(
        "MSA",
        root=wiki_root,
        paths=["raw"],
    )
    for hit in result.hits:
        assert hit.path.startswith("raw/")


def test_search_no_matches_is_empty(wiki_root: Path) -> None:
    result = search("nonexistent-token-xyz", root=wiki_root / "wiki" / "pages")
    assert result.hits == ()
    assert result.truncated is False


def test_search_byte_cap_truncates(wiki_root: Path) -> None:
    # Force a very small cap so the first hit triggers truncation.
    result = search("formula", root=wiki_root / "wiki" / "pages", max_bytes=10)
    assert result.truncated is True


def test_search_max_hits_caps_results(wiki_root: Path) -> None:
    result = search("the", root=wiki_root / "wiki" / "pages", max_hits=1, case_insensitive=True)
    assert len(result.hits) == 1


def test_search_rejects_path_escape(wiki_root: Path) -> None:
    with pytest.raises(OutmemError, match="escapes"):
        search("anything", root=wiki_root / "wiki" / "pages", paths=["../raw"])


def test_search_rejects_missing_root(tmp_path: Path) -> None:
    with pytest.raises(OutmemError, match="not a directory"):
        search("x", root=tmp_path / "nope")


def test_hit_dataclass_immutable() -> None:
    hit = SearchHit(path="x.md", line_number=1, text="body")
    with pytest.raises(FrozenInstanceError):
        hit.line_number = 2  # type: ignore[misc]
