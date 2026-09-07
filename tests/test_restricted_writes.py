"""The write boundary — no write-down, closure, and declassification.

The read boundary stops restricted content reaching a model. This half
stops a session that legitimately read some from putting it somewhere
less restricted, and it does so structurally rather than by trust: in
mode ``S`` an agent may only write items labelled exactly ``S``, which
confines the damage of anything it read to the compartment it read
from.

Equality is the rule, not containment, and it is forced from both
directions — ``labels ⊆ mode`` because you must be able to see what you
are modifying, ``labels ⊇ mode`` because you must not carry facts out of
a restricted context into a less restricted item.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from outmem.exceptions import OutmemError, RestrictionError
from outmem.restricted import Grants, LabelError
from outmem.store import WikiStore


def _wiki(tmp_path: Path, **blocks: object) -> WikiStore:
    import yaml

    root = tmp_path / "w"
    store = WikiStore.init(root)
    raw = yaml.safe_load((root / "config.yaml").read_text()) or {}
    raw["restricted"] = {"labels": ["hr", "legal"], **blocks}
    (root / "config.yaml").write_text(yaml.safe_dump(raw))
    store.close()

    store = WikiStore.open(root)
    store.write_page("glossary", title="Glossary", body="Shared terms.\n")
    store.write_page(
        "hr:severance",
        title="Severance",
        body="Twelve weeks.\n",
        extra={"restricted": ["hr"]},
    )
    return store


@pytest.fixture
def store(tmp_path: Path) -> WikiStore:
    return _wiki(tmp_path)


@pytest.fixture
def hr(store: WikiStore) -> WikiStore:
    return store.as_viewer(mode={"hr"}, grants=Grants.writer("hr"))


@pytest.fixture
def open_writer(store: WikiStore) -> WikiStore:
    return store.as_viewer(grants=Grants.writer("hr"))


class TestNoWriteDown:
    def test_a_restricted_session_cannot_edit_an_open_page(
        self, hr: WikiStore
    ) -> None:
        """The path that carries restricted facts into open content."""
        with pytest.raises(RestrictionError, match="scoped to"):
            hr.extend_page("glossary", body="A fact learned in the HR session.\n")

    def test_nor_append_to_one(self, hr: WikiStore) -> None:
        with pytest.raises(RestrictionError):
            hr.append_page("glossary", body="## More\n\nText.\n")

    def test_nor_create_one(self, hr: WikiStore) -> None:
        """A new page defaults to the mode's labels, so a page created in
        mode {hr} is an HR page — there is no way to make an open one."""
        hr.write_page("notes", title="Notes", body="Text.\n")
        assert "notes" not in store_of(hr).as_viewer().list_slugs()

    def test_the_message_says_which_way_the_mismatch_runs(
        self, hr: WikiStore
    ) -> None:
        """The two directions call for opposite fixes; "denied" leaves the
        caller guessing."""
        with pytest.raises(RestrictionError) as caught:
            hr.extend_page("glossary", body="Text.\n")
        assert "Writing down" in str(caught.value)

    def test_the_message_names_no_item(self, hr: WikiStore) -> None:
        with pytest.raises(RestrictionError) as caught:
            hr.extend_page("glossary", body="Text.\n")
        assert "severance" not in str(caught.value).lower()


class TestNoWriteUp:
    def test_a_write_grant_alone_does_not_reach_a_restricted_page(
        self, open_writer: WikiStore
    ) -> None:
        """Holding `write hr` does not let you edit an HR page from an
        open session; you must start a session in mode {hr}."""
        with pytest.raises(RestrictionError, match="restricted to"):
            open_writer.extend_page("hr:severance", body="Sixteen weeks.\n")

    def test_the_matching_mode_succeeds(self, hr: WikiStore) -> None:
        hr.extend_page("hr:severance", body="Sixteen weeks.\n")
        assert "Sixteen" in hr.read("hr:severance").body

    def test_a_read_grant_is_not_enough_to_write(self, store: WikiStore) -> None:
        view = store.as_viewer(mode={"hr"}, grants=Grants.reader("hr"))
        with pytest.raises(RestrictionError, match="no write grant"):
            view.extend_page("hr:severance", body="Text.\n")

    def test_every_label_in_the_mode_needs_the_grant(
        self, store: WikiStore
    ) -> None:
        grants = Grants(
            read=frozenset({"hr", "legal"}), write=frozenset({"hr"})
        )
        view = store.as_viewer(mode={"hr", "legal"}, grants=grants)
        with pytest.raises(RestrictionError, match="no write grant"):
            view.write_page("x", title="X", body="Text.\n")


class TestNewItemsDefaultToTheMode:
    def test_a_page_written_in_a_mode_carries_its_labels(
        self, hr: WikiStore
    ) -> None:
        """Without this, an HR page citing only open sources would compute
        ∅ ≠ {hr} and be refused unless somebody remembered a label.
        Defaulting to the mode is always the more restricted option."""
        hr.write_page("hr:notes", title="Notes", body="Text.\n")
        assert hr.read("hr:notes").frontmatter.restricted == ["hr"]

    def test_and_is_invisible_to_the_open_mode(self, hr: WikiStore) -> None:
        hr.write_page("hr:notes", title="Notes", body="Text.\n")
        assert "hr:notes" not in store_of(hr).as_viewer().list_slugs()

    def test_an_open_session_writes_open_pages(
        self, open_writer: WikiStore
    ) -> None:
        open_writer.write_page("plain", title="Plain", body="Text.\n")
        assert open_writer.read("plain").frontmatter.restricted == []

    def test_the_extra_field_cannot_override_the_mode(
        self, open_writer: WikiStore
    ) -> None:
        """Otherwise `extra={"restricted": [...]}` is the one way to write
        an item labelled something other than the mode."""
        with pytest.raises(RestrictionError):
            open_writer.write_page(
                "sneaky",
                title="S",
                body="Text.\n",
                extra={"restricted": ["hr"]},
            )

    def test_the_bare_store_still_honours_an_explicit_label(
        self, store: WikiStore
    ) -> None:
        """The operator path: no mode, so the declared value stands."""
        store.write_page(
            "op", title="Op", body="Text.\n", extra={"restricted": ["hr"]}
        )
        assert store.read("op").frontmatter.restricted == ["hr"]


class TestInheritanceBlocksFactLaundering:
    def test_citing_a_restricted_source_from_an_open_session_is_refused(
        self, store: WikiStore, tmp_path: Path
    ) -> None:
        """No separate mechanism needed: the page's computed labels become
        {hr}, which is ≠ ∅, so the write is refused by the same rule."""
        doc = tmp_path / "memo.md"
        doc.write_text("Internal.\n")
        entry = store.add_source(doc, restricted=["hr"])
        view = store.as_viewer(grants=Grants.writer("hr"))
        with pytest.raises(RestrictionError):
            view.write_page(
                "laundered",
                title="L",
                body="A fact from the memo.\n",
                provenance=[entry.citation_path],
            )

    def test_the_same_citation_is_fine_in_the_matching_mode(
        self, store: WikiStore, tmp_path: Path
    ) -> None:
        doc = tmp_path / "memo.md"
        doc.write_text("Internal.\n")
        entry = store.add_source(doc, restricted=["hr"])
        view = store.as_viewer(mode={"hr"}, grants=Grants.writer("hr"))
        view.write_page(
            "hr:derived",
            title="D",
            body="A fact.\n",
            provenance=[entry.citation_path],
        )
        assert view.read("hr:derived")


class TestCitationsCannotRelabelThroughAnEdit:
    """`provenance` is part of a write, and a page inherits its sources'
    labels — so an ordinary-looking edit can change what an item *is*.
    Both directions are refused; `restrict_page` is the verb for that."""

    @pytest.fixture
    def memo(self, store: WikiStore, tmp_path: Path) -> str:
        doc = tmp_path / "memo.md"
        doc.write_text("Confidential.\n")
        return store.add_source(doc, restricted=["hr"]).citation_path

    def test_extend_cannot_attach_a_restricted_source_to_an_open_page(
        self, open_writer: WikiStore, memo: str
    ) -> None:
        """It relabels the page: an open session silently removes a page
        from the open corpus, and cannot read back what it just wrote."""
        with pytest.raises(RestrictionError, match="relabel"):
            open_writer.extend_page(
                "glossary", body="A fact.\n", provenance=[memo]
            )

    def test_append_cannot_either(
        self, open_writer: WikiStore, memo: str
    ) -> None:
        with pytest.raises(RestrictionError, match="relabel"):
            open_writer.append_page(
                "glossary", body="## More\n\nA fact.\n", provenance=[memo]
            )

    def test_nothing_is_written_when_refused(
        self, store: WikiStore, open_writer: WikiStore, memo: str
    ) -> None:
        head = store.head()
        with pytest.raises(RestrictionError):
            open_writer.extend_page("glossary", body="x\n", provenance=[memo])
        assert store.head() == head
        assert store.read("glossary").frontmatter.provenance == []

    def test_dropping_the_citation_a_label_came_from_is_refused(
        self, store: WikiStore, memo: str
    ) -> None:
        """Declassification through a tool that looks like an ordinary
        edit. The page's only label is inherited, so replacing the
        citations with none would publish it."""
        store.write_page("legacy", title="L", body="x\n", provenance=[memo])
        assert "legacy" not in store.as_viewer().list_slugs()
        view = store.as_viewer(mode={"hr"}, grants=Grants.writer("hr"))
        with pytest.raises(RestrictionError):
            view.extend_page("legacy", body="y\n", provenance=[])
        assert "legacy" not in store.as_viewer().list_slugs()

    def test_an_explicit_label_survives_a_citation_change(
        self, store: WikiStore, memo: str
    ) -> None:
        """A page written in a mode carries the label explicitly, so
        re-citing is an ordinary edit rather than a relabelling."""
        view = store.as_viewer(mode={"hr"}, grants=Grants.writer("hr"))
        view.write_page("hr:d", title="D", body="x\n", provenance=[memo])
        view.extend_page("hr:d", body="y\n", provenance=[])
        assert store.read("hr:d").frontmatter.restricted == ["hr"]

    def test_re_citing_within_the_compartment_still_works(
        self, store: WikiStore, memo: str
    ) -> None:
        """The check must not refuse the ordinary path: a section that
        cites one source out of several."""
        view = store.as_viewer(mode={"hr"}, grants=Grants.writer("hr"))
        view.write_page("hr:d", title="D", body="x\n", provenance=[memo])
        view.append_page("hr:d", body="## More\n\ny\n", provenance=[memo])
        view.extend_page("hr:d", body="z\n")  # provenance untouched
        assert "z" in view.read("hr:d").body


class TestClosure:
    def test_a_link_to_a_more_restricted_page_is_refused(
        self, open_writer: WikiStore
    ) -> None:
        """Filtering the page list achieves nothing if an open page's body
        contains [[hr:severance]] — read_page hands over the slug."""
        with pytest.raises(RestrictionError, match="outside this session"):
            open_writer.write_page(
                "leak", title="Leak", body="See [[hr:severance]].\n"
            )

    def test_a_link_within_the_mode_is_fine(self, hr: WikiStore) -> None:
        hr.write_page("hr:notes", title="N", body="See [[hr:severance]].\n")
        assert hr.read("hr:notes")

    def test_linking_up_is_allowed(self, hr: WikiStore) -> None:
        """An HR page may link to the glossary: everyone who can see the
        HR page can already see the glossary."""
        hr.write_page("hr:notes", title="N", body="See [[glossary]].\n")
        assert hr.read("hr:notes")

    def test_a_dangling_link_passes(self, open_writer: WikiStore) -> None:
        """It discloses nothing that is there."""
        open_writer.write_page("x", title="X", body="See [[no:such:page]].\n")

    def test_extend_is_checked_too(self, open_writer: WikiStore) -> None:
        with pytest.raises(RestrictionError):
            open_writer.extend_page("glossary", body="See [[hr:severance]].\n")

    def test_append_is_checked_too(self, open_writer: WikiStore) -> None:
        with pytest.raises(RestrictionError):
            open_writer.append_page("glossary", body="See [[hr:severance]].\n")

    def test_the_refusal_does_not_confirm_the_target_exists(
        self, open_writer: WikiStore
    ) -> None:
        """A bounded oracle either way — the writer typed the name — so
        the message says only which link to remove."""
        with pytest.raises(RestrictionError) as caught:
            open_writer.write_page("leak", title="L", body="[[hr:severance]]\n")
        message = str(caught.value)
        assert "exists" not in message
        assert "hr:severance" in message  # the link they wrote, nothing more


class TestPerModeLogs:
    def test_the_open_mode_writes_the_unpartitioned_path(
        self, open_writer: WikiStore
    ) -> None:
        """A wiki with no restrictions gets no new directory level and
        has nothing to migrate."""
        open_writer.append_log(topic="t", content="Open note.\n")
        assert list(open_writer.log_path.glob("*.md"))

    def test_a_restricted_session_writes_its_own_partition(
        self, hr: WikiStore
    ) -> None:
        """append_log writes an open file, and mandatory writeback pushes
        an agent there when nothing else was warranted — so in mode {hr}
        the default path is a write-down through the one tool the runtime
        insists on calling."""
        hr.append_log(topic="t", content="HR note.\n")
        assert list((hr.log_path / "hr").glob("*.md"))
        assert not list(hr.log_path.glob("*.md"))

    def test_the_partition_is_hidden_from_the_open_mode(
        self, store: WikiStore
    ) -> None:
        hr = store.as_viewer(mode={"hr"}, grants=Grants.writer("hr"))
        hr.append_log(topic="t", content="A severance discussion.\n")
        assert not store.as_viewer().search("severance", scope="log").hits
        assert hr.search("severance", scope="log").hits

    def test_a_read_only_session_cannot_log(self, store: WikiStore) -> None:
        view = store.as_viewer(mode={"hr"}, grants=Grants.reader("hr"))
        with pytest.raises(RestrictionError, match="no write grant"):
            view.append_log(topic="t", content="Text.\n")


class TestRecordIngestion:
    def test_an_open_session_cannot_record_against_a_restricted_source(
        self, store: WikiStore, tmp_path: Path
    ) -> None:
        """Its `prompt` is agent-written free text stored in the registry;
        an open ingestion against a restricted source puts session text
        where an open reader can find it."""
        doc = tmp_path / "memo.md"
        doc.write_text("Internal.\n")
        entry = store.add_source(doc, restricted=["hr"])
        view = store.as_viewer(grants=Grants.writer("hr"))
        with pytest.raises(RestrictionError):
            view.record_ingestion(entry.rel_path, prompt="x", pages_touched=[])

    def test_the_matching_mode_records(
        self, store: WikiStore, tmp_path: Path
    ) -> None:
        doc = tmp_path / "memo.md"
        doc.write_text("Internal.\n")
        entry = store.add_source(doc, restricted=["hr"])
        view = store.as_viewer(mode={"hr"}, grants=Grants.writer("hr"))
        view.record_ingestion(entry.rel_path, prompt="x", pages_touched=[])
        assert store.get_source(entry.rel_path).ingestions


class TestRenameAndRestrictAreOperatorOnly:
    """Both write files the caller did not name, so neither is
    label-checkable and neither belongs to a view.

    ``rename_page`` rewrites inbound ``[[links]]`` across the corpus:
    the targets come from the link graph rather than from the caller,
    and the text that lands in them is the new slug, which the caller
    chooses. That is a write-down with attacker-supplied content into
    open pages, into other compartments, and into pages the session
    cannot see — and there is no way to label-check a write whose
    targets are discovered rather than named.

    ``restrict_page --cascade`` picks its targets the same way, and its
    refusal has to name the referrers, which are exactly the pages a
    view might not be allowed to know about.
    """

    def test_rename_is_refused_to_a_view(self, tmp_path: Path) -> None:
        store = _wiki(tmp_path, paths={"hr:*": ["hr"]})
        store.write_page("hr:bands", title="Bands", body="Pay bands.\n")
        view = store.as_viewer(mode={"hr"}, grants=Grants.writer("hr"))
        with pytest.raises(RestrictionError, match="operator-only"):
            view.rename_page("hr:bands", "notes:bands")

    def test_restrict_is_refused_to_a_view(self, store: WikiStore) -> None:
        view = store.as_viewer(mode={"hr"}, grants=Grants.writer("hr"))
        with pytest.raises(RestrictionError, match="operator-only"):
            view.restrict_page("hr:severance", labels=[])

    def test_a_declassify_grant_does_not_buy_a_view_in(
        self, store: WikiStore
    ) -> None:
        """The grant exists for an application that holds the bare store;
        it is not a way to reach these paths through a view."""
        grants = Grants(
            read=frozenset({"hr"}),
            write=frozenset({"hr"}),
            declassify=frozenset({"hr"}),
        )
        view = store.as_viewer(mode={"hr"}, grants=grants)
        with pytest.raises(RestrictionError, match="operator-only"):
            view.restrict_page("hr:severance", labels=[])

    def test_the_refusal_names_no_item(self, store: WikiStore) -> None:
        view = store.as_viewer(mode={"hr"}, grants=Grants.writer("hr"))
        with pytest.raises(RestrictionError) as caught:
            view.rename_page("hr:severance", "notes:x")
        assert "severance" not in str(caught.value)

    def test_the_operator_still_has_both(self, tmp_path: Path) -> None:
        store = _wiki(tmp_path, paths={"hr:*": ["hr"]})
        store.write_page("hr:bands", title="Bands", body="Pay bands.\n")
        store.rename_page("hr:bands", "hr:pay-bands")
        assert "hr:pay-bands" in store.list_slugs()

    def test_neither_is_in_any_model_facing_palette(
        self, store: WikiStore
    ) -> None:
        from outmem.adapters.pydantic_ai import wiki_tools

        names = {t.__name__ for t in wiki_tools(store.as_viewer())}
        assert "rename_page" not in names
        assert "restrict_page" not in names


class TestAliasesCannotCaptureAnotherPage:
    """The three-call write-down that a metadata field opened.

    An alias inherits the labels of the page it resolves to, so that
    `resolve_slug` cannot follow one past the filter. Applying that to a
    name a LIVE page already occupies inverted it: a session in mode
    {hr} could write an HR page whose `aliases:` named an open page,
    relabel that page to {hr} without touching it, and then legally edit
    it — ending with HR text in a file whose own frontmatter carries no
    label at all.

    A live page always beats an alias claiming its name, so an alias
    must not change that page's labels in either direction.
    """

    def test_an_alias_does_not_relabel_a_live_page(
        self, store: WikiStore, hr: WikiStore
    ) -> None:
        hr.write_page(
            "hr:note", title="N", body="x\n", extra={"aliases": ["glossary"]}
        )
        assert store._labels().for_page("glossary") == frozenset()

    def test_and_the_capture_does_not_open_the_page_to_editing(
        self, hr: WikiStore
    ) -> None:
        hr.write_page(
            "hr:note", title="N", body="x\n", extra={"aliases": ["glossary"]}
        )
        with pytest.raises(RestrictionError):
            hr.extend_page("glossary", body="Twelve weeks of severance.\n")

    def test_the_open_page_stays_visible_to_open_sessions(
        self, store: WikiStore, hr: WikiStore
    ) -> None:
        hr.write_page(
            "hr:note", title="N", body="x\n", extra={"aliases": ["glossary"]}
        )
        assert "glossary" in store.as_viewer().list_slugs()

    def test_an_alias_on_a_free_name_still_inherits(
        self, store: WikiStore, hr: WikiStore
    ) -> None:
        """The behaviour the rule exists for is unchanged: an alias no
        live page occupies must not resolve past the filter."""
        hr.write_page(
            "hr:pay", title="Pay", body="Bands.\n", extra={"aliases": ["old-pay"]}
        )
        assert store.as_viewer().resolve_slug("old-pay") == "old-pay"
        assert hr.resolve_slug("old-pay") == "hr:pay"


class TestSourceKeysAreCheckedInEverySpelling:
    """``resolve_source`` accepts a bare rel_path, a tree-qualified one,
    and a repo-relative one. The label index knew only two, and a miss
    reads as open — so the third spelling returned a restricted file's
    bytes."""

    @pytest.fixture
    def source(self, store: WikiStore, tmp_path: Path):  # type: ignore[no-untyped-def]
        doc = tmp_path / "severance-plan-2026.md"
        doc.write_text("Enhanced severance is 1.5 weeks per year.\n")
        return store.add_source(doc, restricted=["hr"])

    def _spellings(self, store: WikiStore, entry) -> list[str]:  # type: ignore[no-untyped-def]
        return [
            entry.rel_path,
            entry.citation_path,
            f"{store.config.wiki_dir}/{entry.citation_path}",
        ]

    def test_read_source_refuses_every_spelling(
        self, store: WikiStore, source
    ) -> None:
        view = store.as_viewer()
        for key in self._spellings(store, source):
            with pytest.raises(OutmemError, match="no such source"):
                view.read_source(key)

    def test_get_source_returns_none_for_every_spelling(
        self, store: WikiStore, source
    ) -> None:
        view = store.as_viewer()
        for key in self._spellings(store, source):
            assert view.get_source(key) is None, key

    def test_record_ingestion_is_refused_for_every_spelling(
        self, store: WikiStore, source
    ) -> None:
        """The write half of the same hole: an open session could land
        agent-written free text in a restricted source's registry row."""
        view = store.as_viewer(grants=Grants.writer("hr"))
        for key in self._spellings(store, source):
            with pytest.raises(RestrictionError):
                view.record_ingestion(key, prompt="x", pages_touched=[])

    def test_the_cleared_mode_reads_it_by_every_spelling(
        self, store: WikiStore, source
    ) -> None:
        view = store.as_viewer(mode={"hr"}, grants=Grants.reader("hr"))
        for key in self._spellings(store, source):
            assert "1.5 weeks" in view.read_source(key), key

class TestRestrictPage:
    def test_it_labels_the_page(self, store: WikiStore) -> None:
        store.restrict_page("glossary", labels=["hr"])
        assert "glossary" not in store.as_viewer().list_slugs()

    def test_an_inbound_link_from_a_visible_page_refuses(
        self, store: WikiStore
    ) -> None:
        """Restricting is a graph operation, not a field edit. The
        inbound link would still name the page in a body its readers can
        see, which is the closure invariant."""
        store.write_page("public", title="Public", body="See [[glossary]].\n")
        with pytest.raises(RestrictionError, match="linked from"):
            store.restrict_page("glossary", labels=["hr"])

    def test_the_refusal_names_the_referrers(self, store: WikiStore) -> None:
        """They are pages the caller can already see, and naming them is
        the difference between an actionable refusal and a wall."""
        store.write_page("public", title="Public", body="See [[glossary]].\n")
        with pytest.raises(RestrictionError) as caught:
            store.restrict_page("glossary", labels=["hr"])
        assert "public" in str(caught.value)

    def test_cascade_restricts_the_referrers_too(self, store: WikiStore) -> None:
        store.write_page("public", title="Public", body="See [[glossary]].\n")
        store.restrict_page("glossary", labels=["hr"], cascade=True)
        visible = store.as_viewer().list_slugs()
        assert "glossary" not in visible
        assert "public" not in visible

    def test_an_undeclared_label_is_refused(self, store: WikiStore) -> None:
        with pytest.raises(LabelError, match="unknown restriction"):
            store.restrict_page("glossary", labels=["nope"])

    def test_the_operator_may_declassify(self, store: WikiStore) -> None:
        """The bare store holds no mode, so the server-side operator's
        own tooling is not blocked by a grant it never had."""
        store.restrict_page("hr:severance", labels=[])
        assert "hr:severance" in store.as_viewer().list_slugs()

    def test_the_index_is_invalidated_immediately(
        self, store: WikiStore
    ) -> None:
        view = store.as_viewer()
        assert "glossary" in view.list_slugs()
        store.restrict_page("glossary", labels=["hr"])
        assert "glossary" not in view.list_slugs()


class TestCli:
    def test_the_restrict_verb_labels_a_page(
        self, store: WikiStore, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from outmem.cli.__main__ import main

        root = store.root
        store.close()
        rc = main(["restrict", "glossary", "--label", "hr", "--root", str(root)])
        assert rc == 0
        assert "restricted to: hr" in capsys.readouterr().out
        reopened = WikiStore.open(root)
        assert "glossary" not in reopened.as_viewer().list_slugs()

    def test_it_reports_inbound_links_and_exits_nonzero(
        self, store: WikiStore, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from outmem.cli.__main__ import main

        store.write_page("public", title="Public", body="See [[glossary]].\n")
        root = store.root
        store.close()
        rc = main(["restrict", "glossary", "--label", "hr", "--root", str(root)])
        assert rc == 1
        assert "linked from" in capsys.readouterr().err


class TestTheBareStoreIsUnaffected:
    def test_writes_are_not_gated(self, store: WikiStore) -> None:
        store.extend_page("glossary", body="Edited by the operator.\n")
        store.extend_page("hr:severance", body="Also edited.\n")

    def test_a_wiki_with_no_labels_behaves_as_before(
        self, tmp_path: Path
    ) -> None:
        plain = WikiStore.init(tmp_path / "plain")
        plain.write_page("a", title="A", body="See [[b]].\n")
        view = plain.as_viewer()
        view.write_page("b", title="B", body="Text.\n")
        view.extend_page("a", body="Edited.\n")
        view.append_log(topic="t", content="Note.\n")
        assert view.read("a").frontmatter.restricted == []


def store_of(view: WikiStore) -> WikiStore:
    """A bare store over the same wiki — the operator's own handle.

    Tests need to check what a *different* viewer sees; reopening is how
    they get one without the view having to expose a route back.
    """
    return WikiStore.open(view.root)


def test_a_write_refusal_leaves_nothing_behind(store: WikiStore) -> None:
    """The guard runs before any disk write, so a refusal cannot leave a
    half-written page or a stray commit."""
    head = store.head()
    view = store.as_viewer(grants=Grants.writer("hr"))
    with pytest.raises(RestrictionError):
        view.write_page("leak", title="L", body="[[hr:severance]]\n")
    assert not (store.pages_path / "leak.md").exists()
    assert store.head() == head


def test_a_hidden_page_blocks_its_slug_generically(store: WikiStore) -> None:
    """A bounded oracle, documented rather than eliminated: the writer
    learns something blocks the slug, not what."""
    view = store.as_viewer(grants=Grants.writer("hr"))
    with pytest.raises(OutmemError) as caught:
        view.write_page("hr:severance", title="Mine", body="Text.\n")
    assert "Twelve weeks" not in str(caught.value)
