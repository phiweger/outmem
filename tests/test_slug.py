"""Tests for ``outmem.slug``."""

from __future__ import annotations

import pytest

from outmem.exceptions import SlugError
from outmem.slug import Wikilink, extract_wikilinks, validate_slug


class TestValidateSlug:
    @pytest.mark.parametrize(
        "slug",
        ["x", "pricing-formula", "v2-policy", "a1-b2-c3"],
    )
    def test_valid(self, slug: str) -> None:
        validate_slug(slug)

    @pytest.mark.parametrize(
        "slug",
        ["", "Pricing", "pricing--formula", "-leading", "trailing-", "with space", "with/slash"],
    )
    def test_invalid(self, slug: str) -> None:
        with pytest.raises(SlugError):
            validate_slug(slug)

    def test_non_string_raises(self) -> None:
        with pytest.raises(SlugError, match="string"):
            validate_slug(None)  # type: ignore[arg-type]


class TestExtractWikilinks:
    def test_simple(self) -> None:
        body = "See [[acme-msa]] for terms."
        links = extract_wikilinks(body)
        assert links == [Wikilink(slug="acme-msa", display="acme-msa", raw="[[acme-msa]]")]

    def test_with_display(self) -> None:
        body = "See [[discounts|the discount policy]] for details."
        links = extract_wikilinks(body)
        assert len(links) == 1
        assert links[0].slug == "discounts"
        assert links[0].display == "the discount policy"

    def test_multiple_in_order(self) -> None:
        body = "[[a]] and [[b]] and [[c|see C]]."
        slugs = [link.slug for link in extract_wikilinks(body)]
        assert slugs == ["a", "b", "c"]

    def test_none_returns_empty(self) -> None:
        assert extract_wikilinks("plain text, no links here.") == []

    def test_trims_whitespace_inside_brackets(self) -> None:
        body = "[[ slug-x | Display Y ]]"
        link = extract_wikilinks(body)[0]
        assert link.slug == "slug-x"
        assert link.display == "Display Y"
