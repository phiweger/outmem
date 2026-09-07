"""The read boundary — what a view can and cannot see.

Two halves, and the second is the one that matters over time.

The first half tests each read path: a restricted page is absent from
listings, denied by ``read``, and invisible to search, with the denial
shaped exactly like a page that does not exist. That is a finite set of
assertions about the code as it is today.

The second half is :class:`TestEveryPublicMethodIsClassified`, which
reflects over ``WikiStore`` and fails until a newly added public method
is put in one of four buckets. Coverage of a boundary is a property of
the *whole* surface, not of the paths someone remembered to test, and a
method added next year with no filtering is exactly how a boundary like
this rots.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from outmem.exceptions import OutmemError, RestrictionError
from outmem.restricted import DENY_SET, Grants, LabelError
from outmem.store import (
    _NO_CONTENT,
    _OPERATOR_ONLY,
    _VISIBILITY_ENFORCED,
    _WRITE_ENFORCED,
    WikiStore,
)


def _wiki(tmp_path: Path, **blocks: object) -> WikiStore:
    """A wiki declaring `hr` and `legal`, with one open and one HR page."""
    import yaml

    root = tmp_path / "w"
    store = WikiStore.init(root)
    raw = yaml.safe_load((root / "config.yaml").read_text()) or {}
    raw["restricted"] = {"labels": ["hr", "legal"], **blocks}
    (root / "config.yaml").write_text(yaml.safe_dump(raw))
    store.close()

    store = WikiStore.open(root)
    store.write_page("glossary", title="Glossary", body="A shared glossary.\n")
    store.write_page(
        "hr:severance",
        title="Severance",
        body="Twelve weeks of severance pay.\n",
        extra={"restricted": ["hr"]},
    )
    return store


@pytest.fixture
def store(tmp_path: Path) -> WikiStore:
    return _wiki(tmp_path)


@pytest.fixture
def open_view(store: WikiStore) -> WikiStore:
    return store.as_viewer()


@pytest.fixture
def hr_view(store: WikiStore) -> WikiStore:
    return store.as_viewer(mode={"hr"}, grants=Grants.writer("hr"))


class TestTheBareStoreIsUnchanged:
    """A wiki with no view taken behaves exactly as it did before this
    feature existed — including one with restricted content in it, which
    is what makes the server-side operator's tooling keep working."""

    def test_it_reads_restricted_pages(self, store: WikiStore) -> None:
        assert "severance" in store.read("hr:severance").body

    def test_it_lists_them(self, store: WikiStore) -> None:
        assert "hr:severance" in store.list_slugs()

    def test_it_does_not_enforce(self, store: WikiStore) -> None:
        assert not store.enforces_visibility


class TestVisibility:
    def test_the_open_mode_cannot_read_a_restricted_page(
        self, open_view: WikiStore
    ) -> None:
        with pytest.raises(OutmemError):
            open_view.read("hr:severance")

    def test_the_open_mode_still_reads_open_pages(
        self, open_view: WikiStore
    ) -> None:
        assert "glossary" in open_view.read("glossary").body.lower()

    def test_the_matching_mode_reads_it(self, hr_view: WikiStore) -> None:
        assert "severance" in hr_view.read("hr:severance").body

    def test_a_mode_sees_open_content_too(self, hr_view: WikiStore) -> None:
        """Mode {hr} is not an HR-only session — an agent composing HR
        pages still needs the company glossary."""
        assert hr_view.read("glossary")

    def test_a_mode_the_user_does_not_hold_is_refused(
        self, store: WikiStore
    ) -> None:
        with pytest.raises(LabelError, match="cannot read"):
            store.as_viewer(mode={"hr"}, grants=Grants.reader("legal"))

    def test_an_undeclared_mode_label_is_refused(self, store: WikiStore) -> None:
        with pytest.raises(LabelError, match="unknown restriction"):
            store.as_viewer(mode={"nope"}, grants=Grants.reader("nope"))

    def test_the_view_holds_no_route_back_to_the_open_store(
        self, hr_view: WikiStore
    ) -> None:
        """A view that exposed the unrestricted store would make every
        filter below it decorative."""
        for name in dir(hr_view):
            if name.startswith("_"):
                continue
            value = getattr(hr_view, name, None)
            assert not (
                isinstance(value, WikiStore) and not value.enforces_visibility
            ), name


class TestDenialsAreIndistinguishableFromAbsence:
    """A distinguishable error is an existence oracle: learning that
    something is hidden is learning that it is there."""

    def test_read_raises_the_same_error_as_a_missing_page(
        self, open_view: WikiStore
    ) -> None:
        with pytest.raises(OutmemError) as hidden:
            open_view.read("hr:severance")
        with pytest.raises(OutmemError) as absent:
            open_view.read("hr:nonexistent")
        assert str(hidden.value).replace("severance", "nonexistent") == str(
            absent.value
        )

    def test_the_denial_is_not_a_restriction_error(
        self, open_view: WikiStore
    ) -> None:
        """RestrictionError is for writes. Raising it on a read would
        name the very fact the filter exists to withhold."""
        with pytest.raises(OutmemError) as caught:
            open_view.read("hr:severance")
        assert not isinstance(caught.value, RestrictionError)

    def test_exists_reports_false(self, open_view: WikiStore) -> None:
        assert not open_view.exists("hr:severance")
        assert open_view.exists("glossary")

    def test_read_source_denial_matches_an_unregistered_path(
        self, store: WikiStore, tmp_path: Path
    ) -> None:
        doc = tmp_path / "memo.md"
        doc.write_text("Internal memo.\n")
        entry = store.add_source(doc, restricted=["hr"])
        view = store.as_viewer()
        with pytest.raises(OutmemError) as hidden:
            view.read_source(entry.rel_path)
        with pytest.raises(OutmemError) as absent:
            view.read_source("nope/missing.md")
        assert str(hidden.value).startswith("no such source:")
        assert str(absent.value).startswith("no such source:")


class TestListingsOmitHiddenItems:
    def test_list_slugs(self, open_view: WikiStore, hr_view: WikiStore) -> None:
        assert "hr:severance" not in open_view.list_slugs()
        assert "hr:severance" in hr_view.list_slugs()

    def test_index_tree_hides_the_whole_namespace(
        self, open_view: WikiStore
    ) -> None:
        """A namespace whose every page is hidden must not appear at all
        — the namespace name is itself the disclosure."""
        level = open_view.index_tree()
        assert "hr" not in dict(level.namespaces)

    def test_the_generated_index_page_is_rendered_per_viewer(
        self, open_view: WikiStore, hr_view: WikiStore
    ) -> None:
        """wiki/index.md on disk catalogues every page in the wiki. Served
        as stored it would be the one page that hands a reader the name
        of everything hidden from them."""
        assert "hr:severance" not in open_view.read("index").body
        assert "hr:severance" in hr_view.read("index").body

    def test_the_index_still_lists_what_is_visible(
        self, open_view: WikiStore
    ) -> None:
        assert "glossary" in open_view.read("index").body

    def test_list_sources(self, store: WikiStore, tmp_path: Path) -> None:
        """A source's rel_path embeds its original filename, so listing
        one is disclosure even without reading it."""
        doc = tmp_path / "severance-plan-2026.md"
        doc.write_text("Plan.\n")
        store.add_source(doc, restricted=["hr"])
        assert store.as_viewer().list_sources() == []
        assert store.as_viewer(mode={"hr"}, grants=Grants.reader("hr")).list_sources()

    def test_get_source_returns_none(self, store: WikiStore, tmp_path: Path) -> None:
        doc = tmp_path / "memo.md"
        doc.write_text("Memo.\n")
        entry = store.add_source(doc, restricted=["hr"])
        assert store.as_viewer().get_source(entry.rel_path) is None


class TestSearchIsFiltered:
    def test_a_hidden_page_never_appears_in_hits(
        self, open_view: WikiStore, hr_view: WikiStore
    ) -> None:
        assert not open_view.search("severance").hits
        assert hr_view.search("severance").hits

    def test_open_pages_still_match(self, open_view: WikiStore) -> None:
        assert open_view.search("glossary").hits

    def test_the_all_scope_is_filtered_too(
        self, store: WikiStore, tmp_path: Path
    ) -> None:
        doc = tmp_path / "memo.md"
        doc.write_text("The redundancy multiplier is 1.5.\n")
        store.add_source(doc, restricted=["hr"])
        view = store.as_viewer()
        assert not view.search("redundancy", scope="all").hits

    def test_source_scope_is_filtered(
        self, store: WikiStore, tmp_path: Path
    ) -> None:
        doc = tmp_path / "memo.md"
        doc.write_text("The redundancy multiplier is 1.5.\n")
        store.add_source(doc, restricted=["hr"])
        assert not store.as_viewer().search("redundancy", scope="sources").hits
        assert store.as_viewer(
            mode={"hr"}, grants=Grants.reader("hr")
        ).search("redundancy", scope="sources").hits


class TestGraphEdgesAreFiltered:
    def test_backlinks_omit_hidden_referrers(self, store: WikiStore) -> None:
        store.write_page(
            "hr:policy",
            title="Policy",
            body="See [[glossary]].\n",
            extra={"restricted": ["hr"]},
        )
        assert "hr:policy" not in store.as_viewer().backlinks("glossary")
        assert "hr:policy" in store.as_viewer(
            mode={"hr"}, grants=Grants.reader("hr")
        ).backlinks("glossary")

    def test_backlinks_of_a_hidden_target_are_empty(
        self, open_view: WikiStore
    ) -> None:
        assert open_view.backlinks("hr:severance") == ()

    def test_an_alias_does_not_resolve_to_a_hidden_page(
        self, store: WikiStore
    ) -> None:
        """Resolution would hand back the canonical slug of a page the
        viewer cannot see — the name itself is the leak."""
        store.write_page(
            "hr:pay",
            title="Pay",
            body="Bands.\n",
            extra={"restricted": ["hr"], "aliases": ["old-pay-page"]},
        )
        assert store.as_viewer().resolve_slug("old-pay-page") == "old-pay-page"
        assert (
            store.as_viewer(mode={"hr"}, grants=Grants.reader("hr")).resolve_slug(
                "old-pay-page"
            )
            == "hr:pay"
        )


class TestSteeringIsFilteredByMode:
    """The steering signal is rendered into the *system prompt*, so an
    unfiltered `compact: hr:severance-policy` reaches every request
    regardless of who is asking — the one leak needing no tool call."""

    @pytest.fixture
    def human(self, store: WikiStore) -> WikiStore:
        """Steering excludes the agent's own commits, so the fixture's
        writes only register as a signal once the store is looking for a
        different author."""
        import dataclasses

        store.config.agent_identity = dataclasses.replace(
            store.config.agent_identity, email="somebody-else@host"
        )
        return store

    def test_a_restricted_commit_subject_is_dropped(
        self, human: WikiStore
    ) -> None:
        subjects = [c.subject for c in human.as_viewer().steering()]
        assert not any("hr:severance" in s for s in subjects)

    def test_the_cleared_mode_keeps_it(self, human: WikiStore) -> None:
        view = human.as_viewer(mode={"hr"}, grants=Grants.reader("hr"))
        subjects = [c.subject for c in view.steering()]
        assert any("hr:severance" in s for s in subjects)

    def test_open_commits_survive(self, human: WikiStore) -> None:
        subjects = [c.subject for c in human.as_viewer().steering()]
        assert any("glossary" in s for s in subjects)

    def test_filtering_is_on_the_mode_not_the_user(
        self, human: WikiStore
    ) -> None:
        """Two users in one compartment must get the same system prompt,
        or prompt caching stops working across them."""
        a = human.as_viewer(mode={"hr"}, grants=Grants.reader("hr"))
        b = human.as_viewer(mode={"hr"}, grants=Grants.writer("hr"))
        assert [c.subject for c in a.steering()] == [c.subject for c in b.steering()]


class TestAgentsMd:
    def test_a_line_naming_a_hidden_slug_is_dropped(
        self, store: WikiStore
    ) -> None:
        store.agents_path.write_text(
            "# Conventions\n\nKeep terms in [[glossary]].\n"
            "Severance rules live in [[hr:severance]].\n"
        )
        text = store.as_viewer().read_agents_md()
        assert "glossary" in text
        assert "severance" not in text

    def test_the_cleared_mode_gets_the_whole_file(self, store: WikiStore) -> None:
        store.agents_path.write_text("Rules in [[hr:severance]].\n")
        view = store.as_viewer(mode={"hr"}, grants=Grants.reader("hr"))
        assert "hr:severance" in view.read_agents_md()


class TestFailClosed:
    def test_an_unparseable_page_is_hidden_from_every_mode(
        self, store: WikiStore
    ) -> None:
        """Its explicit labels cannot be read, so they cannot be trusted
        to be empty. Resolving to "no labels" would be fail-open."""
        (store.pages_path / "broken.md").write_text("no frontmatter here\n")
        for view in (
            store.as_viewer(),
            store.as_viewer(mode={"hr"}, grants=Grants.reader("hr")),
        ):
            assert "broken" not in view.list_slugs()
            assert not view.exists("broken")

    def test_an_undeclared_label_hides_the_page(self, store: WikiStore) -> None:
        store.write_page(
            "mystery", title="M", body="Text.\n", extra={"restricted": ["hr"]}
        )
        (store.pages_path / "mystery.md").write_text(
            (store.pages_path / "mystery.md")
            .read_text()
            .replace("- hr", "- notdeclared")
        )
        for view in (
            store.as_viewer(),
            store.as_viewer(mode={"hr"}, grants=Grants.reader("hr")),
        ):
            assert "mystery" not in view.list_slugs()

    def test_a_page_inherits_its_sources_labels(
        self, store: WikiStore, tmp_path: Path
    ) -> None:
        """The highest-leverage rule: restricting a document at ingest
        restricts every page ever compiled from it, with nobody having to
        remember a second flag."""
        doc = tmp_path / "memo.md"
        doc.write_text("Memo.\n")
        entry = store.add_source(doc, restricted=["hr"])
        store.write_page(
            "derived",
            title="Derived",
            body="A fact from the memo.\n",
            provenance=[entry.citation_path],
        )
        assert "derived" not in store.as_viewer().list_slugs()
        assert "derived" in store.as_viewer(
            mode={"hr"}, grants=Grants.reader("hr")
        ).list_slugs()

    def test_a_path_rule_restricts_without_frontmatter(
        self, tmp_path: Path
    ) -> None:
        store = _wiki(tmp_path, paths={"hr:*": ["hr"]})
        store.write_page("hr:handbook", title="Handbook", body="Text.\n")
        assert "hr:handbook" not in store.as_viewer().list_slugs()

    def test_a_path_rule_covers_the_namespace_root_page(
        self, tmp_path: Path
    ) -> None:
        store = _wiki(tmp_path, paths={"hr:*": ["hr"]})
        store.write_page("hr", title="HR", body="Index of HR.\n")
        assert "hr" not in store.as_viewer().list_slugs()


class TestOperatorOnlyPathsAreRefused:
    """Shrink before filtering: a method a view cannot reach is a bypass
    class that needs no filtering code and cannot regress."""

    def test_history_is_refused(self, hr_view: WikiStore) -> None:
        with pytest.raises(RestrictionError):
            hr_view.history("glossary")

    def test_evolution_is_refused(self, hr_view: WikiStore) -> None:
        """It returns diff bodies, so a page restricted today would hand
        over the text it had while it was open."""
        with pytest.raises(RestrictionError):
            hr_view.evolution(["glossary"])

    def test_the_bare_store_still_has_them(self, store: WikiStore) -> None:
        assert store.history("glossary")
        assert store.evolution(["glossary"])

    def test_the_refusal_names_no_hidden_item(self, hr_view: WikiStore) -> None:
        with pytest.raises(RestrictionError) as caught:
            hr_view.history("glossary")
        assert "severance" not in str(caught.value)

    @pytest.mark.parametrize("name", sorted(_OPERATOR_ONLY))
    def test_every_operator_path_refuses_a_view(
        self, hr_view: WikiStore, name: str
    ) -> None:
        """Called with whatever arguments the signature demands: the
        guard runs first, so the values never matter."""
        method = getattr(hr_view, name)
        sig = inspect.signature(method)
        args = [
            _dummy(p)
            for p in sig.parameters.values()
            if p.default is inspect.Parameter.empty
            and p.kind
            in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
        ]
        with pytest.raises(RestrictionError):
            method(*args)


def _dummy(param: inspect.Parameter) -> object:
    annotation = str(param.annotation)
    if "Sequence" in annotation or "list" in annotation:
        return []
    return "x"


class TestLabelIndexCaching:
    def test_the_index_is_rebuilt_when_head_moves(self, store: WikiStore) -> None:
        """HEAD is the validity token: every outmem write produces a
        commit, so a moved HEAD is exactly the invalidation signal. A
        stale index is the one caching bug here that fails open."""
        view = store.as_viewer()
        assert "later" not in view.list_slugs()
        store.write_page(
            "later", title="Later", body="Written after the view.\n"
        )
        assert "later" in view.list_slugs()

    def test_restricting_a_page_takes_effect_for_an_existing_view(
        self, store: WikiStore
    ) -> None:
        view = store.as_viewer()
        assert "glossary" in view.list_slugs()
        path = store.pages_path / "glossary.md"
        path.write_text(path.read_text().replace("slug:", "restricted: [hr]\nslug:"))
        store._commit_paths(
            [f"{store.config.wiki_dir}/pages/glossary.md"], subject="restrict: glossary"
        )
        assert "glossary" not in view.list_slugs()

    def test_views_share_one_index(self, store: WikiStore) -> None:
        """Views are per-request and the corpus walk is not; rebuilding
        per request would make each one O(corpus)."""
        a, b = store.as_viewer(), store.as_viewer()
        a.list_slugs()
        assert b._label_cache is a._label_cache is store._label_cache

    def test_a_wiki_with_no_labels_pays_nothing(self, tmp_path: Path) -> None:
        plain = WikiStore.init(tmp_path / "plain")
        plain.write_page("p", title="P", body="Text.\n")
        view = plain.as_viewer()
        assert view.list_slugs() == ["p"]
        assert view._labels().head == ""  # the shared empty index


class TestEveryPublicMethodIsClassified:
    """The mechanism that keeps this boundary complete as the code grows.

    Adding a public method to ``WikiStore`` fails this suite until
    somebody decides which bucket it belongs in. That decision is the
    whole point: it is the moment a person asks "can this return
    something a viewer should not see?", which is a question no amount
    of testing the paths we already thought of will ever prompt.
    """

    def _public(self) -> set[str]:
        return {
            name
            for name, _ in inspect.getmembers(WikiStore)
            if not name.startswith("_")
        }

    def test_the_four_buckets_are_disjoint(self) -> None:
        buckets = [
            _VISIBILITY_ENFORCED,
            _WRITE_ENFORCED,
            _OPERATOR_ONLY,
            _NO_CONTENT,
        ]
        for i, first in enumerate(buckets):
            for second in buckets[i + 1 :]:
                assert not first & second, sorted(first & second)

    def test_every_public_method_is_in_exactly_one(self) -> None:
        classified = (
            _VISIBILITY_ENFORCED | _WRITE_ENFORCED | _OPERATOR_ONLY | _NO_CONTENT
        )
        missing = self._public() - classified
        assert not missing, (
            f"unclassified WikiStore members: {sorted(missing)}. Decide whether "
            "each can return item content, identity, or existence to a viewer, "
            "and add it to the matching set in outmem.store — with a test in "
            "this file if it needs filtering."
        )

    def test_no_bucket_names_a_method_that_no_longer_exists(self) -> None:
        classified = (
            _VISIBILITY_ENFORCED | _WRITE_ENFORCED | _OPERATOR_ONLY | _NO_CONTENT
        )
        stale = classified - self._public()
        assert not stale, f"classified but gone: {sorted(stale)}"

    def test_operator_only_methods_actually_guard(self) -> None:
        """Membership of the set is a claim; this is the check that the
        claim is backed by a call."""
        for name in _OPERATOR_ONLY:
            source = inspect.getsource(getattr(WikiStore, name))
            assert "_require_operator(" in source, name


class TestGrantsVersusMode:
    def test_a_grant_alone_shows_nothing(self, store: WikiStore) -> None:
        """Grants are entitlement; mode is scope. Holding `hr` while
        running in mode ∅ retrieves open content only — restricted
        content is opt-in, and nothing a session reads widens it."""
        view = store.as_viewer(grants=Grants.reader("hr"))
        assert "hr:severance" not in view.list_slugs()

    def test_the_mode_is_not_settable_after_construction(
        self, hr_view: WikiStore
    ) -> None:
        """It is bound at view construction and immutable for the
        session. A mode the model could change would let it read under
        one and write under another."""
        assert hr_view.mode == frozenset({"hr"})
        assert not any(
            "mode" in name and not name.startswith("_")
            for name in dir(hr_view)
            if callable(getattr(hr_view, name, None))
        )


def test_deny_set_is_not_reachable_as_a_mode(store: WikiStore) -> None:
    """Everything fail-closed rests on DENY being unrepresentable in a
    mode. If it could be requested, an unparseable page would become
    readable by asking for it."""
    with pytest.raises(LabelError):
        store.as_viewer(mode=DENY_SET, grants=Grants())
