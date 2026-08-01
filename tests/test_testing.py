"""Tests for ``outmem.testing`` — the resolver conformance harness.

The harness exists so a consumer that reimplements addressing fails on
upgrade instead of drifting silently (issue #9). These tests pin the two
properties that makes true: it catches the divergence that actually
happened, and its three moving parts cannot fall out of sync.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from outmem.store import WikiStore
from outmem.testing import (
    ADDRESSING_FEATURES,
    addressing_cases,
    assert_resolver_conforms,
    build_conformance_wiki,
    resolve_like_outmem,
)


@pytest.fixture
def store(tmp_path: Path) -> WikiStore:
    return build_conformance_wiki(tmp_path / "wiki")


def _delegating(store: WikiStore) -> Callable[[str], str | None]:
    """What a consumer should write — correct by construction."""
    return lambda slug: resolve_like_outmem(store, slug)


def _map_from_list_slugs(store: WikiStore) -> Callable[[str], str | None]:
    """What a consumer *did* write: a map built by walking the pages.

    The shape Fleming had before 0.8.0 — correct until aliases arrived.
    """
    known = set(store.list_slugs())
    return lambda slug: slug if slug in known else None


# ---------------------------------------------------------------------------
# The harness catches the divergence that actually happened
# ---------------------------------------------------------------------------


def test_a_delegating_resolver_conforms(store: WikiStore) -> None:
    assert_resolver_conforms(_delegating(store), store)


def test_a_pre_alias_resolver_fails_on_the_alias_case(store: WikiStore) -> None:
    """The reported symptom: `read_page(old)` worked while the consumer's
    own resolver returned None for the same page."""
    with pytest.raises(AssertionError) as exc:
        assert_resolver_conforms(_map_from_list_slugs(store), store)
    message = str(exc.value)
    assert "[aliases] 'legacy:name'" in message
    assert "expected 'current:name', got None" in message
    assert "store.resolve_slug" in message  # names the durable fix


def test_a_resolver_that_honours_the_declared_slug_fails(store: WikiStore) -> None:
    """The 0.7.0-era divergence: the path addresses the page, not the
    frontmatter. A resolver keyed on `slug:` answers a name outmem
    cannot open, and misses the one it can."""

    def by_declared_slug(slug: str) -> str | None:
        from outmem.index import load_editorial_pages

        pages, _ = load_editorial_pages(store.pages_path)
        for page in pages:
            if page.frontmatter.slug == slug:
                return slug
        return None

    with pytest.raises(AssertionError) as exc:
        assert_resolver_conforms(by_declared_slug, store)
    assert "[path-authoritative]" in str(exc.value)


def test_a_resolver_that_raises_is_reported_not_propagated(store: WikiStore) -> None:
    """A conformance failure must read as a diff, not as someone else's
    traceback."""

    def explodes(slug: str) -> str | None:
        raise RuntimeError("boom")

    with pytest.raises(AssertionError) as exc:
        assert_resolver_conforms(explodes, store)
    assert "raised RuntimeError: boom" in str(exc.value)


def test_every_failure_is_listed_not_just_the_first(store: WikiStore) -> None:
    with pytest.raises(AssertionError) as exc:
        assert_resolver_conforms(lambda _slug: None, store)
    resolvable = [c for c in addressing_cases(store) if c.expected is not None]
    assert f"{len(resolvable)} addressing case(s)" in str(exc.value)


# ---------------------------------------------------------------------------
# The maintainer contract — the three lists cannot drift apart
# ---------------------------------------------------------------------------


def test_every_feature_has_at_least_one_case(store: WikiStore) -> None:
    """Without this, adding a feature name without a case would make the
    harness *look* like it covers a behaviour it never exercises."""
    covered = {c.feature for c in addressing_cases(store)}
    assert covered == set(ADDRESSING_FEATURES), {
        "features with no case": sorted(ADDRESSING_FEATURES - covered),
        "cases with no feature": sorted(covered - ADDRESSING_FEATURES),
    }


def test_the_fixture_exercises_every_case(store: WikiStore) -> None:
    """A case whose query resolves to nothing when it should resolve —
    or vice versa — means the fixture stopped building the page it needs.
    Pinned explicitly so a fixture typo cannot quietly weaken the corpus.
    """
    expected = {c.query: c.expected for c in addressing_cases(store)}
    assert expected == {
        "flat": "flat",
        "abx:penicillin": "abx:penicillin",
        "abx:side-effects:misc": "abx:side-effects:misc",
        "legacy:name": "current:name",
        "current:name": "current:name",
        "live-name": "live-name",
        "mismatch": "mismatch",
        "somewhere-else": None,
        "noslug": "noslug",
        "SOP-Upper:x": None,
        "has_underscore:page": None,
        "no-such-page": None,
        "abx:no-such-page": None,
        "index": "index",
    }


def test_expectations_come_from_the_store_not_a_constant(tmp_path: Path) -> None:
    """The oracle is outmem itself, so the harness stays honest when a
    behaviour *changes* rather than only when one is added."""
    store = build_conformance_wiki(tmp_path / "wiki")
    before = {c.query: c.expected for c in addressing_cases(store)}
    assert before["legacy:name"] == "current:name"

    store.rename_page("current:name", "third:name")
    after = {c.query: c.expected for c in addressing_cases(store)}
    assert after["legacy:name"] == "third:name"
    assert_resolver_conforms(_delegating(store), store)


# ---------------------------------------------------------------------------
# Opting out
# ---------------------------------------------------------------------------


def test_features_can_be_narrowed(store: WikiStore) -> None:
    """A browsing surface that deliberately hides the TOC page records
    that as a decision rather than a silent divergence."""
    assert_resolver_conforms(
        _map_from_list_slugs(store),
        store,
        features=ADDRESSING_FEATURES - {"aliases", "reserved-index"},
    )


def test_an_unknown_feature_name_is_rejected(store: WikiStore) -> None:
    """A typo in an opt-out would otherwise silently skip nothing — or,
    worse, read as covering a feature that doesn't exist."""
    with pytest.raises(AssertionError, match="unknown addressing feature"):
        assert_resolver_conforms(_delegating(store), store, features={"typo"})


# ---------------------------------------------------------------------------
# Removing the reason to reimplement
# ---------------------------------------------------------------------------


def test_index_tree_can_carry_titles(store: WikiStore) -> None:
    """The cause behind the whole issue: a browsing surface needed titles,
    `index_tree` had none, so the consumer walked the directory itself —
    and building a slug map falls out of that walk for free."""
    level = store.index_tree(titles=True)
    assert level.titles["flat"] == "Flat"
    assert "abx:penicillin" not in level.titles  # not at this level

    nested = store.index_tree("abx", titles=True)
    assert nested.titles["abx:penicillin"] == "Penicillin"


def test_index_tree_titles_are_opt_in(store: WikiStore) -> None:
    """Filling them costs a parse per page; the default stays a walk."""
    assert store.index_tree().titles == {}
