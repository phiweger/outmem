"""Conformance harness for consumers that reimplement slug addressing.

Twice now an outmem release has taught the store a new way to resolve a
name, and a downstream resolver silently fell behind: 0.7.0 began
filtering ungrammatical slugs and stopped treating a missing
frontmatter ``slug:`` as fatal; 0.8.0 added ``aliases:`` and taught
:meth:`~outmem.store.WikiStore.resolve_slug` to follow them. In both
cases the *wiki* was sound — ``lint_wiki`` clean, ``unreadable()`` empty
— and the stale component was the consumer, which nothing was positioned
to notice.

``unreadable()`` answers "is my wiki well-formed". This module answers
the other question: **does your resolver still agree with ours.**

    from outmem.testing import assert_resolver_conforms, build_conformance_wiki

    def test_my_resolver(tmp_path):
        store = build_conformance_wiki(tmp_path / "wiki")
        assert_resolver_conforms(lambda s: my_resolve(store, s), store)

The corpus is built *here*, not supplied by the caller, and that is the
whole point. A consumer's own wiki may contain no aliased page, in which
case testing against it would have passed straight through the 0.8.0
regression. Because outmem owns the fixture, a behaviour added in a
future release arrives as a new case in every consumer's suite —
**their tests go red on upgrade without them having heard of the
feature**, which is precisely the unknown-unknown that bites.

A consequence worth stating: a resolver that cannot be pointed at a
different wiki cannot be conformance-tested. Parameterise it by store or
root.

**If you can delegate, delegate instead.** The harness is a safety net
for consumers who genuinely cannot — ones caching a resolution map,
rendering offline, or running without a live store. Everyone else should
write::

    canonical = store.resolve_slug(slug)
    return canonical if store.exists(canonical) else None

which is correct by construction and picks up future rules for free.

Maintainer contract
-------------------

Adding an addressing behaviour means adding a name to
:data:`ADDRESSING_FEATURES`, a page to :func:`build_conformance_wiki`,
and a case to :func:`addressing_cases`. ``tests/test_testing.py`` fails
if a feature has no case or a case has no feature, so the three cannot
drift apart — which is the mechanism that makes the guarantee above real
rather than aspirational.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from outmem.store import WikiStore

__all__ = [
    "ADDRESSING_FEATURES",
    "AddressingCase",
    "addressing_cases",
    "assert_resolver_conforms",
    "build_conformance_wiki",
    "resolve_like_outmem",
]


ADDRESSING_FEATURES = frozenset(
    {
        "canonical",
        "namespaces",
        "aliases",
        "alias-does-not-shadow",
        "path-authoritative",
        "derived-slug",
        "grammar",
        "missing",
        "reserved-index",
    }
)
"""Every way outmem decides what a name addresses.

Exported as data for consumers who want the cheap pin rather than the
full harness::

    assert outmem.testing.ADDRESSING_FEATURES == EXPECTED_AT_TIME_OF_WRITING

That is cruder — it says *something* changed without saying what — but
it is one line and still turns a silent divergence into a failed test.
"""


@dataclass(frozen=True)
class AddressingCase:
    """One name, and what outmem says it addresses."""

    feature: str
    query: str
    expected: str | None
    """The canonical slug, or None when the name addresses no page."""
    why: str


def resolve_like_outmem(store: WikiStore, slug: str) -> str | None:
    """The reference resolver — outmem's own answer, in one place.

    Also the implementation a delegating consumer should copy. It is the
    oracle every case is measured against, so it cannot drift from what
    the library actually does.
    """
    return store.resolve_slug(slug) if store.exists(slug) else None


def build_conformance_wiki(root: Path) -> WikiStore:
    """Create a wiki exercising every addressing behaviour outmem knows.

    Deliberately contains states ``outmem lint`` reports — a page whose
    declared slug disagrees with its path, a filename that derives an
    invalid slug, an alias shadowed by a live page. Those are exactly the
    cases a reimplemented resolver gets wrong, so a corpus without them
    would test only the easy half.
    """
    store = WikiStore.init(root)
    pages = store.pages_path

    # canonical + namespaces
    store.write_page("flat", title="Flat", body="a flat page\n")
    store.write_page("abx:penicillin", title="Penicillin", body="namespaced\n")
    store.write_page(
        "abx:side-effects:misc", title="Misc", body="deeply namespaced\n"
    )

    # aliases — the old name must keep resolving to the new page
    store.write_page("legacy:name", title="Renamed", body="moved since\n")
    store.rename_page("legacy:name", "current:name")

    # alias-does-not-shadow — file-first, so a live page keeps its own
    # name even when another page claims it. Written by hand because
    # `write_page` (correctly) refuses a slug that is another page's alias.
    store.write_page("live-name", title="Live", body="owns its own name\n")
    (pages / "claimant.md").write_text(
        "---\ntitle: Claimant\nslug: claimant\naliases:\n- live-name\n---\n\nb\n",
        encoding="utf-8",
    )

    # path-authoritative — the declared slug is a tripwire, not an address
    (pages / "mismatch.md").write_text(
        "---\ntitle: Mismatch\nslug: somewhere-else\n---\n\nb\n", encoding="utf-8"
    )
    # derived-slug — a missing `slug:` is not fatal; the path names it
    (pages / "noslug.md").write_text("---\ntitle: No slug\n---\n\nb\n", encoding="utf-8")

    # grammar — filenames that derive names `read()` would reject
    (pages / "SOP-Upper").mkdir(parents=True, exist_ok=True)
    (pages / "SOP-Upper" / "x.md").write_text(
        "---\ntitle: Upper\nslug: sop-upper:x\n---\n\nb\n", encoding="utf-8"
    )
    (pages / "has_underscore").mkdir(parents=True, exist_ok=True)
    (pages / "has_underscore" / "page.md").write_text(
        "---\ntitle: Underscore\nslug: page\n---\n\nb\n", encoding="utf-8"
    )

    store._alias_map = None  # hand-written files bypassed the cache
    return store


def addressing_cases(store: WikiStore) -> list[AddressingCase]:
    """Every case, with expectations taken from ``store`` itself.

    The expectation is never hard-coded: it is whatever outmem answers
    today. That keeps the harness honest when a behaviour *changes* as
    well as when one is added — the case list says which names matter,
    the library says what they mean.
    """
    probes: list[tuple[str, str, str]] = [
        ("canonical", "flat", "a page addresses itself by its own name"),
        ("namespaces", "abx:penicillin", "`:` segments map to directories"),
        (
            "namespaces",
            "abx:side-effects:misc",
            "every segment maps, not just the first",
        ),
        ("aliases", "legacy:name", "a renamed page still answers to its old name"),
        ("aliases", "current:name", "and to its new one"),
        (
            "alias-does-not-shadow",
            "live-name",
            "a live page wins its own name over another page's alias",
        ),
        (
            "path-authoritative",
            "mismatch",
            "the path names the page, whatever the frontmatter says",
        ),
        (
            "path-authoritative",
            "somewhere-else",
            "a declared slug that disagrees with the path addresses nothing",
        ),
        ("derived-slug", "noslug", "a missing `slug:` is derived from the path"),
        ("grammar", "SOP-Upper:x", "uppercase never addresses a page"),
        ("grammar", "has_underscore:page", "underscores never address a page"),
        ("missing", "no-such-page", "an unknown name addresses nothing"),
        ("missing", "abx:no-such-page", "including under a live namespace"),
        ("reserved-index", "index", "the auto-maintained TOC is addressable"),
    ]
    return [
        AddressingCase(
            feature=feature,
            query=query,
            expected=resolve_like_outmem(store, query),
            why=why,
        )
        for feature, query, why in probes
    ]


def assert_resolver_conforms(
    resolver: Callable[[str], str | None],
    store: WikiStore,
    *,
    features: Iterable[str] | None = None,
) -> None:
    """Assert ``resolver`` agrees with outmem on every addressing case.

    ``resolver`` takes a name and returns the **canonical slug** it
    addresses, or None. Raises :class:`AssertionError` naming every
    disagreement, the feature it belongs to, and why the case exists.

    ``features`` narrows the check. The honest use is opting out of a
    behaviour you deliberately diverge from — a browsing surface that
    hides the TOC page, say::

        assert_resolver_conforms(
            r, store, features=ADDRESSING_FEATURES - {"reserved-index"}
        )

    Opting out is then a decision recorded in your suite, rather than a
    divergence nobody noticed. Note this also means a *newly added*
    feature is only skipped if you name it — passing a hard-coded set
    silently skips whatever 0.9 adds, so prefer set subtraction.
    """
    wanted = ADDRESSING_FEATURES if features is None else frozenset(features)
    unknown = wanted - ADDRESSING_FEATURES
    if unknown:
        raise AssertionError(
            f"unknown addressing feature(s): {sorted(unknown)}. "
            f"Known: {sorted(ADDRESSING_FEATURES)}"
        )

    failures: list[str] = []
    for case in addressing_cases(store):
        if case.feature not in wanted:
            continue
        try:
            got = resolver(case.query)
        except Exception as exc:  # a resolver must answer, not explode
            failures.append(
                f"  [{case.feature}] {case.query!r}\n"
                f"      expected {case.expected!r}, raised "
                f"{type(exc).__name__}: {exc}\n"
                f"      {case.why}"
            )
            continue
        if got != case.expected:
            failures.append(
                f"  [{case.feature}] {case.query!r}\n"
                f"      expected {case.expected!r}, got {got!r}\n"
                f"      {case.why}"
            )
    if failures:
        raise AssertionError(
            f"resolver disagrees with outmem on {len(failures)} addressing "
            "case(s):\n"
            + "\n".join(failures)
            + "\n\nIf this appeared after an upgrade, outmem learned an "
            "addressing rule your resolver has not. The durable fix is to "
            "delegate:\n"
            "    canonical = store.resolve_slug(slug)\n"
            "    return canonical if store.exists(canonical) else None"
        )
