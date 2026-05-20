"""Tests for ``outmem.slug``."""

from __future__ import annotations

from pathlib import Path

import pytest

from outmem.exceptions import SlugError
from outmem.slug import (
    Wikilink,
    extract_wikilinks,
    relpath_to_slug,
    slug_to_relpath,
    validate_slug,
)


class TestValidateSlug:
    @pytest.mark.parametrize(
        "slug",
        [
            # Flat (single-segment) slugs — the v0.1 shape.
            "x",
            "pricing-formula",
            "v2-policy",
            "a1-b2-c3",
            # Namespaced (multi-segment) slugs introduced in v0.2.
            "abx:penicillin",
            "abx:side-effects:misc",
            "a:b:c:d",
            "infection-control:policy",
            "deep:ns:with-hyphens:leaf",
        ],
    )
    def test_valid(self, slug: str) -> None:
        validate_slug(slug)

    @pytest.mark.parametrize(
        "slug",
        [
            "",
            "Pricing",
            "pricing--formula",
            "-leading",
            "trailing-",
            "with space",
            "with/slash",
            # Namespace edge cases — these were silently dropped by
            # the v0.1 regex (which rejected ``:`` outright); after
            # v0.2 the grammar is stricter on shape.
            ":",
            "::",
            "abx:",
            ":abx",
            "abx::pen",
            "abx:-pen",
            "abx:pen-",
            "abx:Pen",
            "abx: pen",
            "abx :pen",
        ],
    )
    def test_invalid(self, slug: str) -> None:
        with pytest.raises(SlugError):
            validate_slug(slug)

    def test_non_string_raises(self) -> None:
        with pytest.raises(SlugError, match="string"):
            validate_slug(None)  # type: ignore[arg-type]


class TestSlugRelpathHelpers:
    """``slug_to_relpath`` ↔ ``relpath_to_slug`` are inverses."""

    @pytest.mark.parametrize(
        ("slug", "rel"),
        [
            ("pricing-formula", "pricing-formula.md"),
            ("abx:penicillin", "abx/penicillin.md"),
            ("abx:side-effects:misc", "abx/side-effects/misc.md"),
            ("a:b:c:d", "a/b/c/d.md"),
        ],
    )
    def test_round_trip(self, slug: str, rel: str) -> None:
        produced = slug_to_relpath(slug)
        assert produced.as_posix() == rel
        assert relpath_to_slug(produced) == slug

    def test_relpath_to_slug_strips_md_suffix(self) -> None:
        assert relpath_to_slug(Path("abx/penicillin.md")) == "abx:penicillin"

    def test_slug_to_relpath_is_idempotent_under_round_trip(self) -> None:
        # Use a deeply-nested example to exercise multi-segment handling.
        slug = "tier-one:tier-two:tier-three:leaf-page"
        assert relpath_to_slug(slug_to_relpath(slug)) == slug


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

    def test_namespaced_target(self) -> None:
        body = "Cross-link: [[abx:penicillin]] and [[abx:side-effects:misc|side-effects]]."
        links = extract_wikilinks(body)
        slugs = [link.slug for link in links]
        assert slugs == ["abx:penicillin", "abx:side-effects:misc"]
        assert links[1].display == "side-effects"

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
