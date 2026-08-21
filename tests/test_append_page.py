"""``append_page`` — writing a long page without one turn having to hold it.

The gap this closes: ``extend_page`` *replaces* the body, so "write it in
sections" meant re-emitting everything written so far on every call. The
calls grew monotonically and the last one still had to emit the whole
page in a single turn — which is the output-budget pressure that makes a
model truncate in the first place. Sectioned writing only becomes a real
remedy once appending exists.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from outmem.exceptions import OutmemError
from outmem.store import WikiStore


@pytest.fixture
def store(tmp_path: Path) -> WikiStore:
    s = WikiStore.init(tmp_path / "w")
    s.write_page(
        "clinical:sepsis",
        title="Sepsis",
        body="## Erreger\n\nGramnegative Erreger dominieren.",
        provenance=["sources/a1b2c3d4e5f6/leitlinie.md"],
    )
    return s


class TestAppendSemantics:
    def test_existing_content_is_kept(self, store: WikiStore) -> None:
        store.append_page("clinical:sepsis", body="## Diagnostik\n\nBlutkulturen.")
        body = store.read("clinical:sepsis").body
        assert "Gramnegative Erreger dominieren." in body
        assert "Blutkulturen." in body

    def test_appended_after_not_before(self, store: WikiStore) -> None:
        store.append_page("clinical:sepsis", body="## Diagnostik\n\nBlutkulturen.")
        body = store.read("clinical:sepsis").body
        assert body.index("Erreger") < body.index("Diagnostik")

    def test_blank_line_separates_sections(self, store: WikiStore) -> None:
        """Markdown needs the blank line: without it the appended heading
        is swallowed into the preceding paragraph."""
        store.append_page("clinical:sepsis", body="## Diagnostik\n\nBlutkulturen.")
        assert "dominieren.\n\n## Diagnostik" in store.read("clinical:sepsis").body

    def test_repeated_appends_accumulate(self, store: WikiStore) -> None:
        """The point of the tool: a page grows across turns, and no single
        turn ever has to emit the whole thing."""
        for section in ("Diagnostik", "Therapie", "Prävention"):
            store.append_page("clinical:sepsis", body=f"## {section}\n\nText.")
        body = store.read("clinical:sepsis").body
        for section in ("Erreger", "Diagnostik", "Therapie", "Prävention"):
            assert f"## {section}" in body

    def test_frontmatter_is_preserved(self, store: WikiStore) -> None:
        store.append_page("clinical:sepsis", body="## Diagnostik\n\nText.")
        page = store.read("clinical:sepsis")
        assert page.frontmatter.title == "Sepsis"
        assert page.frontmatter.slug == "clinical:sepsis"

    def test_updated_is_bumped(self, store: WikiStore) -> None:
        before = store.read("clinical:sepsis").frontmatter.updated
        store.append_page("clinical:sepsis", body="## Diagnostik\n\nText.")
        assert store.read("clinical:sepsis").frontmatter.updated >= before

    def test_returns_a_commit_sha(self, store: WikiStore) -> None:
        sha = store.append_page("clinical:sepsis", body="## Diagnostik\n\nText.")
        assert len(sha) >= 7

    def test_appending_to_an_empty_body(self, tmp_path: Path) -> None:
        """No leading blank lines when there is nothing to separate from."""
        s = WikiStore.init(tmp_path / "empty")
        s.write_page("stub", title="Stub", body="")
        s.append_page("stub", body="First real content.")
        assert s.read("stub").body.startswith("First real content.")


class TestProvenanceIsAdditive:
    """Unlike ``extend_page``, which replaces — an appended section can
    cite a new source without restating the page's existing citations."""

    def test_new_pointer_is_added(self, store: WikiStore) -> None:
        store.append_page(
            "clinical:sepsis",
            body="## Therapie\n\nText.",
            provenance=["sources/999888777666/therapie.md"],
        )
        refs = store.read("clinical:sepsis").frontmatter.provenance
        assert "sources/a1b2c3d4e5f6/leitlinie.md" in refs
        assert "sources/999888777666/therapie.md" in refs

    def test_recited_pointer_is_not_duplicated(self, store: WikiStore) -> None:
        store.append_page(
            "clinical:sepsis",
            body="## Therapie\n\nText.",
            provenance=["sources/a1b2c3d4e5f6/leitlinie.md"],
        )
        refs = store.read("clinical:sepsis").frontmatter.provenance
        assert refs.count("sources/a1b2c3d4e5f6/leitlinie.md") == 1

    def test_dict_entry_dedups_against_string_entry(self, store: WikiStore) -> None:
        """Provenance entries are strings or dicts; the same source in
        either shape is the same citation."""
        store.append_page(
            "clinical:sepsis",
            body="## Therapie\n\nText.",
            provenance=[{"path": "sources/a1b2c3d4e5f6/leitlinie.md", "label": "LL"}],
        )
        assert len(store.read("clinical:sepsis").frontmatter.provenance) == 1

    def test_omitting_provenance_leaves_it_untouched(self, store: WikiStore) -> None:
        store.append_page("clinical:sepsis", body="## Therapie\n\nText.")
        assert store.read("clinical:sepsis").frontmatter.provenance == [
            "sources/a1b2c3d4e5f6/leitlinie.md"
        ]


class TestRefusals:
    def test_empty_body_is_refused(self, store: WikiStore) -> None:
        """An empty append is a no-op commit — noise in the history."""
        with pytest.raises(OutmemError, match="empty"):
            store.append_page("clinical:sepsis", body="   \n")

    def test_missing_page_is_refused(self, store: WikiStore) -> None:
        with pytest.raises(OutmemError):
            store.append_page("no:such:page", body="Text.")

    def test_index_slug_is_refused(self, store: WikiStore) -> None:
        with pytest.raises(OutmemError, match="index"):
            store.append_page("index", body="Text.")


class TestAdapterTool:
    def test_tool_is_exposed_and_gated(self, store: WikiStore) -> None:
        from outmem.adapters.pydantic_ai import wiki_read_tools, wiki_tools
        from outmem.agent.runtime import _APPROVAL_GATED_TOOLS

        assert "append_page" in {t.__name__ for t in wiki_tools(store)}
        # Commit-producing, so it must be absent from the read-only palette
        # and present in the approval gate — a write tool that slips either
        # is a write path with no human in front of it.
        assert "append_page" not in {t.__name__ for t in wiki_read_tools(store)}
        assert "append_page" in _APPROVAL_GATED_TOOLS

    def test_tool_appends_through_the_store(self, store: WikiStore) -> None:
        from outmem.adapters.pydantic_ai import wiki_tools

        tool = next(t for t in wiki_tools(store) if t.__name__ == "append_page")
        tool(slug="clinical:sepsis", body="## Diagnostik\n\nBlutkulturen.")
        assert "Blutkulturen." in store.read("clinical:sepsis").body

    def test_tool_reports_a_missing_page_without_raising(
        self, store: WikiStore
    ) -> None:
        from outmem.adapters.pydantic_ai import wiki_tools

        tool = next(t for t in wiki_tools(store) if t.__name__ == "append_page")
        assert "append_page failed" in tool(slug="no:such:page", body="Text.")
