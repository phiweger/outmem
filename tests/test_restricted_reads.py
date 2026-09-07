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
            and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        kwargs = {
            p.name: _dummy(p)
            for p in sig.parameters.values()
            if p.default is inspect.Parameter.empty and p.kind is p.KEYWORD_ONLY
        }
        with pytest.raises(RestrictionError):
            method(*args, **kwargs)


def _dummy(param: inspect.Parameter) -> object:
    annotation = str(param.annotation)
    if any(t in annotation for t in ("Sequence", "list", "Iterable")):
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

    def test_a_wiki_with_no_labels_is_unaffected(self, tmp_path: Path) -> None:
        """No content carries a label, so the index is empty and every
        subset test passes. The saving for such a wiki is that nothing
        ever takes a view, not that a view is free."""
        plain = WikiStore.init(tmp_path / "plain")
        plain.write_page("p", title="P", body="Text.\n")
        view = plain.as_viewer()
        assert view.list_slugs() == ["p"]
        assert view.read("p")

    def test_an_unrestricted_store_never_builds_the_index(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The zero-cost claim, made structural: every enforcement point
        tests `_mode is None` first, so a wiki with no access control
        does not reach the corpus walk at all."""
        from outmem._store import labels as labels_mod

        plain = WikiStore.init(tmp_path / "plain")
        plain.write_page("p", title="P", body="Text.\n")
        monkeypatch.setattr(
            labels_mod, "build", lambda *a, **k: pytest.fail("index was built")
        )
        plain.list_slugs()
        plain.read("p")
        plain.search("Text")
        plain.exists("p")
        plain.write_page("q", title="Q", body="More.\n")


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

    def test_no_content_members_really_return_no_content(
        self, hr_view: WikiStore, store: WikiStore
    ) -> None:
        """The other half of the same claim, and the one that was easier
        to leave as an assertion nobody checked. Each member is called on
        a view over a wiki that HAS restricted content, and its result
        must not contain any of it.

        Members needing arguments, doing git I/O, or mutating state are
        skipped by name — with the skip list kept short and explicit, so
        adding a member to `_NO_CONTENT` to dodge this check is a
        visible act rather than a quiet one.
        """
        secret = "Twelve weeks of severance"
        store.extend_page("hr:severance", body=f"{secret} at full pay.\n")

        # Constructors, git I/O, and state mutation — nothing here reads
        # a page, and calling them would touch a remote or a marker file.
        skip = {
            "init", "open", "close", "pull", "push", "record_run",
            "as_viewer", "allow_elision_body", "is_page_path",
        }
        checked: list[str] = []
        for name in sorted(_NO_CONTENT - skip):
            member = getattr(type(hr_view), name, None)
            value = (
                getattr(hr_view, name)
                if isinstance(member, property)
                else getattr(hr_view, name)()
            )
            checked.append(name)
            assert secret not in repr(value), name
            assert "hr:severance" not in repr(value), name
        # Most of the bucket is actually exercised, so growing the skip
        # list to dodge the check shows up here rather than passing
        # quietly.
        assert len(checked) > len(skip), sorted(skip)


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
        # `mode` is a read-only property, so there is no setter to reach
        # even from inside the process — let alone from a tool argument.
        with pytest.raises(AttributeError):
            hr_view.mode = frozenset()  # type: ignore[misc]
        assert hr_view.mode == frozenset({"hr"})


def test_deny_set_is_not_reachable_as_a_mode(store: WikiStore) -> None:
    """Everything fail-closed rests on DENY being unrepresentable in a
    mode. If it could be requested, an unparseable page would become
    readable by asking for it."""
    with pytest.raises(LabelError):
        store.as_viewer(mode=DENY_SET, grants=Grants())


class TestViewsShareTheStoresResources:
    """A view is a shallow copy, so anything it *assigns* diverges — and
    the source registries are assigned lazily.

    A view that had touched sources held a registry snapshot from that
    moment and never saw a row added afterwards. The label index then
    had no entry for that source, and a missing entry reads as open. The
    same divergence gave every view its own SQLite connections, which
    only that view's `close()` would release.
    """

    def test_a_view_sees_a_source_registered_after_it_was_built(
        self, store: WikiStore, tmp_path: Path
    ) -> None:
        view = store.as_viewer()
        view.list_sources()  # force the view to open a registry

        doc = tmp_path / "severance-framework.md"
        doc.write_text("SECRET twelve weeks.\n")
        store.add_source(doc, restricted=["hr"])

        assert view.list_sources() == []
        assert not view.search("SECRET", scope="all").hits

    def test_the_registries_are_the_same_objects(
        self, store: WikiStore
    ) -> None:
        view = store.as_viewer()
        view.list_sources()
        assert view._source_registry is store._source_registry

    def test_so_are_the_lock_and_the_label_index(self, store: WikiStore) -> None:
        view = store.as_viewer()
        assert view._write_lock is store._write_lock
        assert view._label_cache is store._label_cache

    def test_two_views_share_them_too(self, store: WikiStore) -> None:
        a, b = store.as_viewer(), store.as_viewer(grants=Grants.reader("hr"))
        a.list_sources()
        assert a._source_registry is b._source_registry


class TestSourcePathSpellings:
    """`resolve_source` resolves against the filesystem, so a caller can
    name one file many ways. A guard that compares strings cannot keep
    up, and a miss reads as open."""

    @pytest.fixture
    def entry(self, store: WikiStore, tmp_path: Path):  # type: ignore[no-untyped-def]
        doc = tmp_path / "severance-plan-2026.md"
        doc.write_text("SECRET: twelve weeks.\n")
        return store.add_source(doc, restricted=["hr"])

    @pytest.mark.parametrize(
        "shape",
        [
            "{rel}",
            "sources/{rel}",
            "wiki/sources/{rel}",
            "sources/./{rel}",
            "sources/../sources/{rel}",
            "wiki/sources/../sources/{rel}",
            "./sources/{rel}",
        ],
    )
    def test_every_spelling_is_refused(
        self, store: WikiStore, entry, shape: str
    ) -> None:
        key = shape.format(rel=entry.rel_path)
        with pytest.raises(OutmemError, match="no such source"):
            store.as_viewer().read_source(key)

    def test_the_cleared_mode_reads_it_by_any_spelling(
        self, store: WikiStore, entry
    ) -> None:
        view = store.as_viewer(mode={"hr"}, grants=Grants.reader("hr"))
        assert "twelve weeks" in view.read_source(f"sources/./{entry.rel_path}")

    def test_a_page_citing_an_odd_spelling_still_inherits(
        self, store: WikiStore, entry
    ) -> None:
        """Otherwise the page stays open while printing the restricted
        source's filename in its own provenance."""
        store.write_page(
            "derived",
            title="D",
            body="A fact.\n",
            provenance=[f"sources/./{entry.rel_path}"],
        )
        assert "derived" not in store.as_viewer().list_slugs()


class TestPathRulesCoverNamesThatAreNotLivePages:
    """The page map holds only the `.md` files under `wiki/pages/`, so
    everything else in that tree was missing from it — and a miss reads
    as open, right through a namespace the config had restricted."""

    @pytest.fixture
    def store(self, tmp_path: Path) -> WikiStore:
        return _wiki(tmp_path, paths={"hr:*": ["hr"]})

    def test_a_non_markdown_file_under_a_restricted_namespace(
        self, store: WikiStore
    ) -> None:
        (store.pages_path / "hr").mkdir(parents=True, exist_ok=True)
        (store.pages_path / "hr" / "table.txt").write_text("SECRETPAYROLL band 4\n")
        view = store.as_viewer()
        assert not view.search("SECRETPAYROLL", scope="wiki").hits
        assert not view.search("SECRETPAYROLL", scope="all").hits

    def test_agents_md_naming_a_slug_before_its_page_exists(
        self, store: WikiStore
    ) -> None:
        """AGENTS.md reaches every system prompt, and it is exactly where
        somebody writes "payroll bands belong in hr:payroll-bands" for a
        page that has not been written yet."""
        store.agents_path.write_text(
            "Keep terms in [[glossary]].\n"
            "Payroll bands belong in hr:payroll-bands.\n"
        )
        text = store.as_viewer().read_agents_md()
        assert "glossary" in text
        assert "payroll-bands" not in text

    def test_the_cleared_mode_keeps_both(self, store: WikiStore) -> None:
        (store.pages_path / "hr").mkdir(parents=True, exist_ok=True)
        (store.pages_path / "hr" / "table.txt").write_text("SECRETPAYROLL band 4\n")
        view = store.as_viewer(mode={"hr"}, grants=Grants.reader("hr"))
        assert view.search("SECRETPAYROLL", scope="wiki").hits


class TestLogPartitionsAreCompartments:
    """`log/` subdirectories are compartments. One sentence, because the
    ambiguity underneath it has no better answer.

    outmem creates only `log/<date>.md` and `log/<label-set>/<date>.md`,
    so a label-shaped subdirectory is a partition. Once a label is
    withdrawn from `restricted.labels` its old partition still holds
    that compartment's entries, and reading an undeclared name as open
    would let one deleted config line publish them — the exact inversion
    the rest of the design refuses.

    The cost is that a hand-made `log/archive/` is hidden from views.
    That is the safe side of an ambiguity nothing on disk can settle,
    and the operator still sees it.
    """

    def test_a_declared_partition_is_hidden_from_the_open_mode(
        self, store: WikiStore
    ) -> None:
        hr = store.as_viewer(mode={"hr"}, grants=Grants.writer("hr"))
        hr.append_log(topic="t", content="A severance discussion.\n")
        assert not store.as_viewer().search("severance", scope="log").hits
        assert hr.search("severance", scope="log").hits

    def test_withdrawing_the_label_does_not_publish_its_partition(
        self, store: WikiStore
    ) -> None:
        """The case that decides the ambiguity."""
        import yaml

        hr = store.as_viewer(mode={"hr"}, grants=Grants.writer("hr"))
        hr.append_log(topic="t", content="A severance discussion.\n")
        root = store.root
        store.close()

        raw = yaml.safe_load((root / "config.yaml").read_text())
        raw["restricted"] = {"labels": ["legal"]}  # hr withdrawn
        (root / "config.yaml").write_text(yaml.safe_dump(raw))

        reopened = WikiStore.open(root)
        assert not reopened.as_viewer().search("severance", scope="log").hits
        assert reopened.search("severance", scope="log").hits  # operator sees it

    def test_a_label_shaped_directory_is_treated_as_one(
        self, store: WikiStore
    ) -> None:
        """`archive` is a valid label name and nothing on disk says it
        was not written as a compartment, so it is hidden rather than
        guessed at."""
        (store.log_path / "archive").mkdir(parents=True, exist_ok=True)
        (store.log_path / "archive" / "2024-01-01.md").write_text("Old note.\n")
        assert not store.as_viewer().search("Old note", scope="log").hits
        assert store.search("Old note", scope="log").hits

    def test_a_name_that_could_not_be_a_label_set_stays_open(
        self, store: WikiStore
    ) -> None:
        """The other side of the line: outmem could not have written
        this, so hiding it would break a wiki that never used
        compartments."""
        (store.log_path / "2024 backup").mkdir(parents=True, exist_ok=True)
        (store.log_path / "2024 backup" / "d.md").write_text("Kept note.\n")
        assert store.as_viewer().search("Kept note", scope="log").hits

    def test_the_open_partition_is_unaffected(self, store: WikiStore) -> None:
        store.as_viewer(grants=Grants.none())
        store.append_log(topic="t", content="An open note.\n")
        assert store.as_viewer().search("An open note", scope="log").hits

class TestSteeringDoesNotCarryLogTopics:
    """A `log:` subject is a topic somebody typed, not a name this can
    resolve — so it cannot be filtered by parsing it. The pathspec is
    narrowed instead, and a partition the viewer cannot see is never
    walked."""

    def test_a_restricted_log_subject_never_reaches_an_open_view(
        self, store: WikiStore
    ) -> None:
        import dataclasses

        store.config.agent_identity = dataclasses.replace(
            store.config.agent_identity, email="somebody-else@host"
        )
        hr = store.as_viewer(mode={"hr"}, grants=Grants.writer("hr"))
        hr.append_log(topic="severance cap review", content="Notes.\n")

        subjects = [c.subject for c in store.as_viewer().steering()]
        assert not any("severance" in s for s in subjects)
        assert any(
            "severance" in c.subject
            for c in store.as_viewer(
                mode={"hr"}, grants=Grants.reader("hr")
            ).steering()
        )


class TestTheIndexTokenCoversWritesThatMakeNoCommit:
    """HEAD is not enough on its own.

    Source labels live in `.sources.db`, and two writes to it produce no
    commit: an ingest into the untracked local tree, and re-ingesting
    identical bytes with `--restricted`, which is the documented way to
    correct a mislabelled source. HEAD does not move, so a process that
    is not the writer kept serving a source it no longer may.

    That is the deployment the design assumes — several workers against
    one repository — so the token carries the registries' fingerprints
    alongside HEAD, and a changed fingerprint drops the registry handle
    too. `SourceRegistry` is an in-memory snapshot taken at load, so
    rebuilding the index from the handle already held would have read
    back the very value the fingerprint said not to trust.
    """

    def test_another_process_sees_a_local_ingest_restriction(
        self, store: WikiStore, tmp_path: Path
    ) -> None:
        other = WikiStore.open(store.root)
        doc = tmp_path / "handbook.md"
        doc.write_text("LOCAL SECRET handbook.\n")

        entry = other.add_source(doc, into_subdir="policy", local=True)
        view = store.as_viewer()
        assert entry.citation_path in [e.citation_path for e in view.list_sources()]

        head_before = store.head()
        other.add_source(doc, into_subdir="policy", local=True, restricted=["hr"])
        assert store.head() == head_before, "the premise is that no commit lands"

        assert entry.citation_path not in [
            e.citation_path for e in view.list_sources()
        ]
        with pytest.raises(OutmemError):
            view.read_source(entry.citation_path)

    def test_the_writers_own_view_sees_it_immediately(
        self, store: WikiStore, tmp_path: Path
    ) -> None:
        doc = tmp_path / "memo.md"
        doc.write_text("SECRET memo.\n")
        entry = store.add_source(doc, restricted=["hr"], commit=False)
        assert store.as_viewer().list_sources() == []
        assert entry.restricted == frozenset({"hr"})

    def test_derived_caches_are_keyed_on_the_same_token(
        self, store: WikiStore
    ) -> None:
        """Anything holding a snapshot of page bodies has to move when
        the labels do — a BM25 net built before a restriction would keep
        answering from the text it had."""
        assert store.corpus_token() is not None
        before = store.corpus_token()
        store.write_page("later", title="L", body="Text.\n")
        assert store.corpus_token() != before

    def test_a_wiki_with_no_labels_gets_no_token(self, tmp_path: Path) -> None:
        """`None` means "nothing to re-derive", so callers skip a `git
        rev-parse` per query rather than paying for a feature that is
        not switched on."""
        plain = WikiStore.init(tmp_path / "plain")
        assert plain.corpus_token() is None


class TestAPageTheIndexHasNotClassifiedIsDenied:
    """The fail-closed rule applied to time rather than to content.

    A write puts the file on disk and commits afterwards, both under the
    write lock — while readers hold no lock at all. In the window
    between, HEAD has not moved, so the index's token says it is current
    when it is not, and the new page is absent from its page map. Read
    as "no labels", a page created in mode {hr} would be served to
    everyone until the commit landed.

    Found by hammering the cache with concurrent readers and writers,
    which is what the deployment does by construction: one process, many
    requests. The window is milliseconds and the state converges — which
    is exactly why nothing else would have caught it.
    """

    def test_an_uncommitted_page_is_hidden_rather_than_open(
        self, store: WikiStore
    ) -> None:
        view = store.as_viewer()
        view.list_slugs()  # warm the index

        # Reproduce the window: the file exists, HEAD has not moved.
        head = store.head()
        (store.pages_path / "hr").mkdir(parents=True, exist_ok=True)
        (store.pages_path / "hr" / "draft.md").write_text(
            "---\ntitle: Draft\nslug: hr:draft\nrestricted: [hr]\n---\n\nSECRETDRAFT\n"
        )
        assert store.head() == head, "the premise is that no commit landed"

        assert "hr:draft" not in view.list_slugs()
        assert not view.exists("hr:draft")
        assert not view.search("SECRETDRAFT", scope="wiki").hits
        assert not view.search("SECRETDRAFT", scope="all").hits
        with pytest.raises(OutmemError):
            view.read("hr:draft")

    def test_an_uncommitted_open_page_is_hidden_too(
        self, store: WikiStore
    ) -> None:
        """Denying only the ones that turn out to be restricted would
        need us to already know the labels, which is the thing we do not
        have. Availability yields to confidentiality for the width of
        one commit."""
        view = store.as_viewer()
        view.list_slugs()
        (store.pages_path / "plain.md").write_text(
            "---\ntitle: Plain\nslug: plain\n---\n\nOrdinary.\n"
        )
        assert "plain" not in view.list_slugs()

    def test_and_becomes_visible_once_the_commit_lands(
        self, store: WikiStore
    ) -> None:
        """It is a window, not a wall — the state converges."""
        view = store.as_viewer()
        view.list_slugs()
        store.write_page("plain", title="Plain", body="Ordinary.\n")
        assert "plain" in view.list_slugs()

    def test_the_operator_is_unaffected(self, store: WikiStore) -> None:
        (store.pages_path / "plain.md").write_text(
            "---\ntitle: Plain\nslug: plain\n---\n\nOrdinary.\n"
        )
        assert "plain" in store.list_slugs()

    def test_a_slug_with_no_file_still_resolves_by_path_rule(
        self, tmp_path: Path
    ) -> None:
        """The other branch: an unknown slug that names no file is not a
        race, it is simply absent, and the path rules alone decide."""
        store = _wiki(tmp_path, paths={"hr:*": ["hr"]})
        view = store.as_viewer()
        assert not view.exists("hr:never-written")
        assert view.as_viewer()._page_visible("nothing-here")

    def test_the_generated_index_slug_is_not_caught_by_the_rule(
        self, store: WikiStore
    ) -> None:
        """`wiki/index.md` exists and is deliberately absent from the page
        map, so the rule has to exempt it or the catalogue disappears."""
        assert store.as_viewer().read("index")
