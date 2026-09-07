"""The whole feature, exercised the way a deployment actually uses it.

Every other restricted-* test isolates one rule. This one walks the
path a company would: declare labels, ingest a confidential document,
let an agent compile pages from it, then serve two different users and
check that each sees exactly what they are entitled to and nothing
else.

It exists because the rules compose, and composition is where a design
like this fails. Each individual check can pass while the arrangement
of them still leaks — a page correctly labelled but reachable through
an unfiltered listing, a source correctly hidden but named in a
citation, an agent correctly scoped but writing into the wrong log.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from outmem.adapters.pydantic_ai import wiki_read_tools, wiki_tools
from outmem.exceptions import OutmemError, RestrictionError
from outmem.lint import lint_wiki
from outmem.restricted import Grants
from outmem.store import WikiStore

CONFIG = {
    "labels": ["hr", "legal"],
    "paths": {"hr:*": ["hr"]},
    "sources": {"hr/*": ["hr"]},
}


@pytest.fixture
def company(tmp_path: Path) -> WikiStore:
    """A wiki as an operator would set one up, with content in place."""
    root = tmp_path / "company"
    store = WikiStore.init(root)
    raw = yaml.safe_load((root / "config.yaml").read_text()) or {}
    raw["restricted"] = CONFIG
    raw["retrieval"] = {"strategy": "bm25"}
    (root / "config.yaml").write_text(yaml.safe_dump(raw))
    store.close()

    store = WikiStore.open(root)

    # Open material anyone may see.
    store.write_page(
        "glossary",
        title="Glossary",
        body="**FTE** — full-time equivalent. **Notice period** — see contract.\n",
    )
    store.write_page(
        "benefits:cycling",
        title="Cycle to work",
        body="The cycle-to-work scheme covers bicycles up to 1000 pounds.\n",
    )

    # A confidential document, labelled at the moment someone holds it.
    memo = tmp_path / "severance-framework-2026.md"
    memo.write_text(
        "Enhanced severance is 1.5 weeks per year of service, capped at 52.\n"
    )
    store.add_source(memo, into_subdir="hr", restricted=["hr"])

    # A page compiled from it, written by an HR-scoped session.
    entry = store.list_sources()[0]
    hr = store.as_viewer(mode={"hr"}, grants=Grants.writer("hr"))
    hr.write_page(
        "hr:severance",
        title="Severance",
        body=(
            "Enhanced severance is 1.5 weeks per year, capped at 52.\n"
            "See [[glossary]] for terms.\n"
        ),
        provenance=[entry.citation_path],
    )
    return store


@pytest.fixture
def uncleared(company: WikiStore) -> WikiStore:
    """An ordinary employee: no labels at all."""
    return company.as_viewer(grants=Grants.none())


@pytest.fixture
def cleared(company: WikiStore) -> WikiStore:
    """Somebody in HR, working in the HR compartment."""
    return company.as_viewer(mode={"hr"}, grants=Grants.writer("hr"))


def _tools(store: WikiStore) -> dict[str, object]:
    return {t.__name__: t for t in wiki_read_tools(store)}


class TestTheUnclearedEmployeeSeesNothing:
    """Every surface at once. Any one of these leaking makes the rest
    pointless, which is why they are asserted together rather than
    spread across the files that own each mechanism."""

    def test_no_surface_volunteers_the_page_the_source_or_the_content(
        self, uncleared: WikiStore
    ) -> None:
        """Nothing the user did not already name comes back to them."""
        tools = _tools(uncleared)
        surfaces = {
            "search_wiki": tools["search_wiki"](question="severance pay"),  # type: ignore[operator]
            "grep_wiki": tools["grep_wiki"](pattern="severance"),  # type: ignore[operator]
            "list_pages": tools["list_pages"](),  # type: ignore[operator]
            "list_sources": tools["list_sources"](),  # type: ignore[operator]
            "read_page(index)": tools["read_page"](slug="index"),  # type: ignore[operator]
            "find_backlinks": tools["find_backlinks"](slug="glossary"),  # type: ignore[operator]
            "search_index": tools["search_index"](prefix=""),  # type: ignore[operator]
        }
        for name, text in surfaces.items():
            assert "hr:severance" not in text, name
            assert "1.5 weeks" not in text, name
            assert "severance-framework-2026" not in text, name

    def test_the_generated_index_is_the_sharpest_case(
        self, uncleared: WikiStore, cleared: WikiStore
    ) -> None:
        """`wiki/index.md` on disk is one file whose whole purpose is to
        list every slug. Serving it as stored would hand over the name of
        everything hidden, and no amount of per-page filtering elsewhere
        would catch it — the tool used to re-read the file after the
        store had already rendered a filtered page for the viewer."""
        tools = _tools(uncleared)
        assert "benefits:cycling" in tools["read_page"](slug="index")  # type: ignore[operator]
        assert "hr:severance" not in tools["read_page"](slug="index")  # type: ignore[operator]
        assert "hr:severance" in _tools(cleared)["read_page"](slug="index")  # type: ignore[operator]

    def test_reading_the_hidden_page_answers_exactly_as_for_a_missing_one(
        self, uncleared: WikiStore
    ) -> None:
        """The message echoes the slug the caller typed, which tells them
        nothing they did not already know. What matters is that the two
        answers are the same answer."""
        read = _tools(uncleared)["read_page"]
        hidden = read(slug="hr:severance")  # type: ignore[operator]
        absent = read(slug="hr:no-such-page")  # type: ignore[operator]
        assert hidden.replace("severance", "no-such-page") == absent
        assert "1.5 weeks" not in hidden

    def test_the_source_filename_never_appears(
        self, uncleared: WikiStore
    ) -> None:
        """A source's path embeds its original filename, which is often
        the most disclosing thing about it."""
        assert not uncleared.list_sources()
        assert not uncleared.search("severance", scope="all").hits

    def test_they_still_get_the_open_wiki_in_full(
        self, uncleared: WikiStore
    ) -> None:
        """A boundary that also breaks ordinary use is not a boundary
        anyone will keep switched on."""
        assert set(uncleared.list_slugs()) == {"glossary", "benefits:cycling"}
        assert "cycle-to-work" in uncleared.read("benefits:cycling").body
        out = _tools(uncleared)["search_wiki"](question="cycle to work")  # type: ignore[operator]
        assert "benefits:cycling" in out

    def test_no_hint_is_offered(self, uncleared: WikiStore) -> None:
        """Hiding is a property of grants: for someone who does not hold
        the label, even the count must not exist."""
        out = _tools(uncleared)["search_wiki"](question="severance")  # type: ignore[operator]
        assert "outside this session's scope" not in out


class TestTheClearedEmployeeSeesEverything:
    def test_the_restricted_page_and_its_source(self, cleared: WikiStore) -> None:
        assert "1.5 weeks" in cleared.read("hr:severance").body
        assert cleared.list_sources()

    def test_and_the_open_wiki_too(self, cleared: WikiStore) -> None:
        """Mode {hr} is not an HR-only session — an agent composing HR
        pages still needs the glossary it links to."""
        assert "FTE" in cleared.read("glossary").body

    def test_retrieval_finds_the_restricted_page(
        self, cleared: WikiStore
    ) -> None:
        out = _tools(cleared)["search_wiki"](question="severance pay")  # type: ignore[operator]
        assert "hr:severance" in out


class TestTheClearedEmployeeInAnOpenSession:
    """Grants are entitlement; mode is scope. Somebody who holds `hr`
    but has not opted into it retrieves open content only — and is told
    a count, which is what makes opting in something they can discover."""

    @pytest.fixture
    def view(self, company: WikiStore) -> WikiStore:
        return company.as_viewer(grants=Grants.writer("hr"))

    def test_the_content_is_still_hidden(self, view: WikiStore) -> None:
        assert "hr:severance" not in view.list_slugs()
        with pytest.raises(OutmemError):
            view.read("hr:severance")

    def test_but_they_are_told_the_compartment_is_there(
        self, view: WikiStore
    ) -> None:
        """A count of what the compartment holds, not of what this
        question matched — the count is the same for every question, so
        it carries no bits about any of them."""
        out = _tools(view)["search_wiki"](question="zzzznonexistent")  # type: ignore[operator]
        assert "you also have access to: hr" in out
        assert "hr:severance" not in out
        assert "1.5 weeks" not in out

    def test_and_cannot_write_the_restricted_page_from_here(
        self, view: WikiStore
    ) -> None:
        with pytest.raises(RestrictionError):
            view.extend_page("hr:severance", body="Edited.\n")


class TestAnAgentInTheHrCompartment:
    def test_it_can_extend_hr_pages(self, cleared: WikiStore) -> None:
        tool = next(t for t in wiki_tools(cleared) if t.__name__ == "append_page")
        tool(slug="hr:severance", body="## Notice\n\nTwelve weeks.\n")
        assert "Twelve weeks" in cleared.read("hr:severance").body

    def test_it_cannot_write_down_into_the_open_wiki(
        self, cleared: WikiStore
    ) -> None:
        """The path the whole write rule exists to close: an agent reads
        restricted material and compiles an open page from context.

        Note the shape of the closure. The call is not refused — the
        agent writes wherever it likes — but the page it produces is
        labelled with the session's mode, so the *content* never leaves
        the compartment. Refusing instead would only teach the model to
        try a different slug.
        """
        tool = next(t for t in wiki_tools(cleared) if t.__name__ == "write_page")
        tool(slug="notes:pay", title="Pay", body="1.5 weeks per year.\n")
        assert cleared.read("notes:pay").frontmatter.restricted == ["hr"]

    def test_and_an_open_session_never_sees_what_it_wrote(
        self, cleared: WikiStore, uncleared: WikiStore
    ) -> None:
        tool = next(t for t in wiki_tools(cleared) if t.__name__ == "write_page")
        tool(slug="notes:pay", title="Pay", body="1.5 weeks per year.\n")
        assert "notes:pay" not in uncleared.list_slugs()
        assert not uncleared.search("1.5 weeks", scope="all").hits

    def test_editing_an_existing_open_page_is_refused_outright(
        self, cleared: WikiStore
    ) -> None:
        """Where defaulting cannot save it: the target already exists and
        already has fewer labels."""
        tool = next(t for t in wiki_tools(cleared) if t.__name__ == "extend_page")
        out = tool(slug="glossary", body="Severance is 1.5 weeks per year.\n")
        assert "failed" in out.lower()
        assert "1.5 weeks" not in cleared.read("glossary").body

    def test_new_pages_it_writes_are_restricted_by_default(
        self, cleared: WikiStore, uncleared: WikiStore
    ) -> None:
        tool = next(t for t in wiki_tools(cleared) if t.__name__ == "write_page")
        tool(slug="hr:redundancy", title="Redundancy", body="Process.\n")
        assert "hr:redundancy" in cleared.list_slugs()
        assert "hr:redundancy" not in uncleared.list_slugs()

    def test_its_log_entries_land_in_the_hr_partition(
        self, cleared: WikiStore, uncleared: WikiStore
    ) -> None:
        """Mandatory writeback pushes an agent to append_log when nothing
        else was warranted, so the open log is a write-down reached
        through the one tool the runtime insists on calling."""
        tool = next(t for t in wiki_tools(cleared) if t.__name__ == "append_log")
        tool(topic="gap", content="No page covers the severance cap.\n")
        assert not uncleared.search("severance cap", scope="log").hits
        assert cleared.search("severance cap", scope="log").hits

    def test_it_is_not_offered_the_history_tools(
        self, cleared: WikiStore
    ) -> None:
        names = {t.__name__ for t in wiki_tools(cleared)}
        assert "topic_evolution" not in names
        assert "page_history" not in names


class TestThePathRuleSafetyNet:
    def test_a_page_under_hr_is_restricted_without_frontmatter(
        self, company: WikiStore, uncleared: WikiStore
    ) -> None:
        """Somebody writes into the namespace and forgets the label."""
        company.write_page("hr:handbook", title="Handbook", body="Rules.\n")
        assert "hr:handbook" not in uncleared.list_slugs()

    def test_a_source_ingested_into_hr_is_restricted_without_the_flag(
        self, company: WikiStore, tmp_path: Path, uncleared: WikiStore
    ) -> None:
        doc = tmp_path / "board-minutes.md"
        doc.write_text("Minutes.\n")
        company.add_source(doc, into_subdir="hr")
        assert not any(
            "board-minutes" in e.rel_path for e in uncleared.list_sources()
        )


class TestTheCorpusPassesItsOwnChecks:
    def test_lint_is_clean(self, company: WikiStore) -> None:
        """The arrangement built above satisfies every invariant — so a
        finding in a later test means the change broke something, not
        that the fixture was always non-conforming."""
        report = lint_wiki(
            company.wiki_path,
            log_dir=company.log_path,
            sources_dir=company.sources_path,
            sources_local_dir=company.sources_local_path,
            repo_root=company.root,
            restricted=company.restrictions,
        )
        restricted_findings = [
            f for f in report.findings if f.kind.startswith("restricted-")
        ]
        assert not restricted_findings, [f.message for f in restricted_findings]

    def test_lint_catches_a_leak_introduced_by_hand(
        self, company: WikiStore
    ) -> None:
        """The case write-time enforcement cannot reach: the operator
        edits an open page directly and names a restricted slug."""
        company.extend_page(
            "glossary", body="**Severance** — see [[hr:severance]].\n"
        )
        report = lint_wiki(
            company.wiki_path,
            sources_dir=company.sources_path,
            restricted=company.restrictions,
        )
        assert any(
            f.kind == "restricted-link-violation" for f in report.findings
        )


class TestTheOperatorRetainsFullControl:
    def test_the_bare_store_sees_and_edits_everything(
        self, company: WikiStore
    ) -> None:
        assert "hr:severance" in company.list_slugs()
        company.extend_page("hr:severance", body="Revised.\n")
        assert company.history("hr:severance")

    def test_it_can_declassify(
        self, company: WikiStore, uncleared: WikiStore
    ) -> None:
        """Somebody has to be able to fix a mislabelled page, and it is
        not going to be the compartment's own users."""
        company.write_page(
            "notes:draft", title="Draft", body="Text.\n", extra={"restricted": ["hr"]}
        )
        company.restrict_page("notes:draft", labels=[])
        assert "notes:draft" in uncleared.list_slugs()

    def test_a_path_rule_cannot_be_undone_by_declassifying(
        self, company: WikiStore
    ) -> None:
        """The failure this refusal replaces is the bad one: the caller
        asked to declassify, got a success and a commit, and nothing
        changed because the path rule reapplied the label."""
        with pytest.raises(RestrictionError, match=r"restricted\.paths"):
            company.restrict_page("hr:severance", labels=[])

    def test_nor_by_ignoring_an_inherited_source_label(
        self, company: WikiStore, tmp_path: Path
    ) -> None:
        doc = tmp_path / "board-memo.md"
        doc.write_text("Confidential.\n")
        entry = company.add_source(doc, restricted=["legal"])
        company.write_page(
            "notes:summary",
            title="Summary",
            body="A fact.\n",
            provenance=[entry.citation_path],
            extra={"restricted": ["legal"]},
        )
        with pytest.raises(RestrictionError, match="cites a source"):
            company.restrict_page("notes:summary", labels=[])

    def test_restricting_an_already_linked_page_is_refused(
        self, company: WikiStore
    ) -> None:
        company.extend_page("glossary", body="See [[benefits:cycling]].\n")
        with pytest.raises(RestrictionError, match="linked from"):
            company.restrict_page("benefits:cycling", labels=["hr"])


def test_deleting_the_config_block_hides_labelled_content_rather_than_publishing_it(
    company: WikiStore,
) -> None:
    """The single most dangerous edit anyone can make to this feature.

    Removing the `restricted:` block looks like "turn it off", and the
    obvious implementation — skip the label index when nothing is
    declared — makes one deleted line publish the whole HR corpus to
    every view. The safe reading is the one an undeclared label already
    gets: a label nobody declares is a label nobody can hold, so its
    content is hidden rather than released.

    The operator is unaffected, which is what makes this recoverable:
    they hold the bare store, they can still read the page, and they can
    put the declaration back.
    """
    raw = yaml.safe_load((company.root / "config.yaml").read_text())
    del raw["restricted"]
    (company.root / "config.yaml").write_text(yaml.safe_dump(raw))
    company.close()

    plain = WikiStore.open(company.root)
    assert "hr:severance" not in plain.as_viewer().list_slugs()
    assert "hr:severance" in plain.list_slugs()  # the operator still sees it


def test_withdrawing_one_label_hides_only_that_compartment(
    company: WikiStore,
) -> None:
    """Rules naming the withdrawn label have to go too — the config
    refuses to open otherwise, which is the loud half of the same
    protection."""
    raw = yaml.safe_load((company.root / "config.yaml").read_text())
    raw["restricted"] = {"labels": ["legal"]}
    (company.root / "config.yaml").write_text(yaml.safe_dump(raw))
    company.close()

    plain = WikiStore.open(company.root)
    visible = plain.as_viewer().list_slugs()
    assert "hr:severance" not in visible
    assert "glossary" in visible


def test_an_orphaned_path_rule_refuses_to_open(company: WikiStore) -> None:
    """Withdrawing a label while a rule still assigns it would restrict a
    whole namespace to a compartment nobody can hold. Loud, not silent."""
    from outmem.restricted import LabelError

    raw = yaml.safe_load((company.root / "config.yaml").read_text())
    raw["restricted"]["labels"] = ["legal"]  # rule "hr:*" still names hr
    (company.root / "config.yaml").write_text(yaml.safe_dump(raw))
    company.close()

    with pytest.raises(LabelError, match="unknown restriction"):
        WikiStore.open(company.root)
