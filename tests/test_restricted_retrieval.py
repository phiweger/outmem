"""Retrieval under a view — filtering, over-fetch, hints, and the palette.

Retrieval is where the boundary is easiest to get subtly wrong. Three
things are pinned here that are not obvious from the read-boundary
tests:

- **Filtering happens at candidate generation, not on the gate's
  output.** The rerank gate ships excerpts to an LLM provider;
  restricted text that reaches it has already left the deployment,
  whatever the gate then returns.
- **Over-fetch is a security requirement, not a quality one.** Trimming
  a fixed top-k after filtering makes the result *count* an existence
  oracle.
- **The compartment hint is a deliberate disclosure with a precise
  shape.** Counts only, per label, and only for labels the viewer
  actually holds.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from outmem.adapters.pydantic_ai import wiki_read_tools, wiki_tools
from outmem.restricted import Grants
from outmem.store import WikiStore


def _wiki(tmp_path: Path) -> WikiStore:
    import yaml

    root = tmp_path / "w"
    store = WikiStore.init(root)
    raw = yaml.safe_load((root / "config.yaml").read_text()) or {}
    raw["restricted"] = {"labels": ["hr", "legal"]}
    raw["retrieval"] = {"strategy": "bm25"}
    (root / "config.yaml").write_text(yaml.safe_dump(raw))
    store.close()

    store = WikiStore.open(root)
    store.write_page(
        "benefits:cycling",
        title="Cycle to work",
        body="The cycle-to-work scheme covers bicycles up to 1000 pounds.\n",
    )
    store.write_page(
        "hr:parental-leave",
        title="Parental leave",
        body="Parental leave is 26 weeks at full pay.\n",
        extra={"restricted": ["hr"]},
    )
    store.write_page(
        "legal:settlements",
        title="Settlements",
        body="Settlement agreements follow the parental leave schedule.\n",
        extra={"restricted": ["legal"]},
    )
    return store


@pytest.fixture
def store(tmp_path: Path) -> WikiStore:
    return _wiki(tmp_path)


def _tool(store: WikiStore, name: str):  # type: ignore[no-untyped-def]
    return next(t for t in wiki_read_tools(store) if t.__name__ == name)


class TestSearchToolIsFiltered:
    def test_an_open_session_gets_no_restricted_page(
        self, store: WikiStore
    ) -> None:
        out = _tool(store.as_viewer(), "search_wiki")(question="parental leave")
        assert "hr:parental-leave" not in out

    def test_the_cleared_mode_gets_it(self, store: WikiStore) -> None:
        view = store.as_viewer(mode={"hr"}, grants=Grants.reader("hr"))
        out = _tool(view, "search_wiki")(question="parental leave")
        assert "hr:parental-leave" in out

    def test_open_pages_are_unaffected(self, store: WikiStore) -> None:
        out = _tool(store.as_viewer(), "search_wiki")(question="cycle to work")
        assert "benefits:cycling" in out

    def test_grep_is_filtered_too(self, store: WikiStore) -> None:
        assert "parental" not in _tool(store.as_viewer(), "grep_wiki")(
            pattern="Parental leave"
        )

    def test_read_page_is_filtered(self, store: WikiStore) -> None:
        out = _tool(store.as_viewer(), "read_page")(slug="hr:parental-leave")
        assert "26 weeks" not in out

    def test_list_pages_is_filtered(self, store: WikiStore) -> None:
        assert "hr:parental-leave" not in _tool(store.as_viewer(), "list_pages")()


class TestFilteringPrecedesTheRerankGate:
    """The gate ships candidate excerpts to an LLM provider. Restricted
    text that reaches it has already left the deployment, whatever the
    gate decides afterwards — so the filter has to be at candidate
    generation."""

    def test_the_gate_never_sees_a_hidden_page(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = _wiki(tmp_path)
        import yaml

        raw = yaml.safe_load((store.root / "config.yaml").read_text())
        raw["retrieval"] = {"strategy": "rerank(bm25)"}
        (store.root / "config.yaml").write_text(yaml.safe_dump(raw))
        store.close()
        store = WikiStore.open(tmp_path / "w")

        seen: list[list[tuple[str, str]]] = []

        def _spy(*, model, query, candidates, max_relevant):  # type: ignore[no-untyped-def]
            seen.append(list(candidates))
            return [slug for slug, _ in candidates], None

        monkeypatch.setattr("outmem.optimize.blocks.judge_relevance", _spy)
        _tool(store.as_viewer(), "search_wiki")(question="parental leave")

        shipped = {slug for batch in seen for slug, _ in batch}
        assert "hr:parental-leave" not in shipped
        assert "legal:settlements" not in shipped

    def test_the_cleared_mode_does_ship_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = _wiki(tmp_path)
        import yaml

        raw = yaml.safe_load((store.root / "config.yaml").read_text())
        raw["retrieval"] = {"strategy": "rerank(bm25)"}
        (store.root / "config.yaml").write_text(yaml.safe_dump(raw))
        store.close()
        store = WikiStore.open(tmp_path / "w")

        seen: list[list[tuple[str, str]]] = []

        def _spy(*, model, query, candidates, max_relevant):  # type: ignore[no-untyped-def]
            seen.append(list(candidates))
            return [slug for slug, _ in candidates], None

        monkeypatch.setattr("outmem.optimize.blocks.judge_relevance", _spy)
        view = store.as_viewer(mode={"hr"}, grants=Grants.reader("hr"))
        _tool(view, "search_wiki")(question="parental leave")
        assert "hr:parental-leave" in {slug for batch in seen for slug, _ in batch}


class TestRetrieverCacheIsKeyedOnHead:
    def test_restricting_a_page_takes_effect_on_the_next_query(
        self, store: WikiStore
    ) -> None:
        """BM25 snapshots every page body at construction. A retriever
        built before a page was restricted would keep answering from the
        text it had — a cache that fails open."""
        tool = _tool(store.as_viewer(), "search_wiki")
        assert "benefits:cycling" in tool(question="cycle to work")

        path = store.pages_path / "benefits" / "cycling.md"
        path.write_text(path.read_text().replace("slug:", "restricted: [hr]\nslug:"))
        store._commit_paths(
            [f"{store.config.wiki_dir}/pages/benefits/cycling.md"],
            subject="restrict: benefits:cycling",
        )
        assert "benefits:cycling" not in tool(question="cycle to work")


class TestCompartmentHint:
    """Counts only, per label, grant-gated — and independent of the
    question. This is what makes the opt-in default workable: without it
    a cleared user asking about parental leave gets nothing and never
    learns to switch."""

    def test_a_holder_in_the_open_mode_is_told_what_they_could_reach(
        self, store: WikiStore
    ) -> None:
        view = store.as_viewer(grants=Grants.reader("hr"))
        assert view.compartment_hint() == {"hr": 1}

    def test_a_user_without_the_grant_is_told_nothing(
        self, store: WikiStore
    ) -> None:
        assert store.as_viewer().compartment_hint() == {}

    def test_only_labels_the_user_holds_are_named(self, store: WikiStore) -> None:
        """An aggregate count would leak the existence of compartments the
        user does not hold."""
        counts = store.as_viewer(grants=Grants.reader("hr")).compartment_hint()
        assert "legal" not in counts

    def test_a_multi_label_item_the_user_cannot_fully_hold_is_not_counted(
        self, store: WikiStore
    ) -> None:
        """No mode this user could choose would show it, so its existence
        is not theirs to learn."""
        store.write_page(
            "joint:case",
            title="Case",
            body="A dispute.\n",
            extra={"restricted": ["hr", "legal"]},
        )
        view = store.as_viewer(grants=Grants.reader("hr"))
        assert view.compartment_hint() == {"hr": 1}

    def test_nothing_is_offered_once_already_in_the_compartment(
        self, store: WikiStore
    ) -> None:
        view = store.as_viewer(mode={"hr"}, grants=Grants.reader("hr"))
        assert view.compartment_hint() == {}

    def test_an_empty_compartment_is_not_advertised(
        self, tmp_path: Path
    ) -> None:
        store = _wiki(tmp_path)
        store.restrict_page("hr:parental-leave", labels=[])
        view = store.as_viewer(grants=Grants.reader("hr"))
        assert "hr" not in view.compartment_hint()

    def test_the_bare_store_offers_no_hint(self, store: WikiStore) -> None:
        assert store.compartment_hint() == {}


class TestTheHintIsNotAContentOracle:
    """The sharpest failure mode this feature has, and the reason the
    hint takes no query at all.

    The obvious implementation counts the items that matched *this
    question* and fell outside the mode. The model writes the question,
    so asking "twelve" and then "eleven" and comparing the two counts
    reads a fact out of a restricted page without ever retrieving it —
    and the model could then commit that fact to an open page. Relying
    on the model not to try is exactly what the design forbids.
    """

    def test_the_method_takes_no_question(self, store: WikiStore) -> None:
        """Structural, not behavioural: there is no argument to probe
        with. A future refactor that reintroduces one fails here."""
        import inspect

        params = inspect.signature(WikiStore.compartment_hint).parameters
        assert list(params) == ["self"]

    def test_the_answer_does_not_vary_with_the_query(
        self, store: WikiStore
    ) -> None:
        """`hr:parental-leave` says "26 weeks at full pay". A
        query-sensitive hint would answer differently for a term the page
        contains than for one it does not."""
        view = store.as_viewer(grants=Grants.reader("hr"))
        tool = _tool(view, "search_wiki")
        hit = tool(question="26 weeks full pay parental leave")
        miss = tool(question="31 weeks half pay parental leave")
        assert _note(hit) == _note(miss)

    def test_and_not_with_a_term_present_only_in_restricted_text(
        self, store: WikiStore
    ) -> None:
        view = store.as_viewer(grants=Grants.reader("hr"))
        tool = _tool(view, "search_wiki")
        assert _note(tool(question="parental")) == _note(
            tool(question="zzzznonexistent")
        )


def _note(output: str) -> str:
    """The compartment note from a search_wiki result, or ""."""
    for line in output.splitlines():
        if line.startswith("(this session is scoped"):
            return line
    return ""


class TestTheHintIsRendered:
    def test_it_appears_on_an_empty_result(self, store: WikiStore) -> None:
        """The case it exists for: an empty answer is exactly when a
        cleared user needs telling the material is elsewhere."""
        view = store.as_viewer(grants=Grants.reader("hr"))
        out = _tool(view, "search_wiki")(question="zzzznonexistent")
        assert "you also have access to: hr" in out

    def test_it_carries_no_slug_title_or_excerpt(
        self, store: WikiStore
    ) -> None:
        view = store.as_viewer(grants=Grants.reader("hr"))
        out = _tool(view, "search_wiki")(question="zzzznonexistent")
        assert "parental-leave" not in out
        assert "26 weeks" not in out

    def test_an_uncleared_user_sees_no_note_at_all(
        self, store: WikiStore
    ) -> None:
        out = _tool(store.as_viewer(), "search_wiki")(question="zzzznonexistent")
        assert "you also have access to" not in out

    def test_a_nonempty_result_is_not_cluttered_with_it(
        self, store: WikiStore
    ) -> None:
        """Attaching it to every search would be noise on the common
        path; the empty result is where it changes what the user does."""
        view = store.as_viewer(grants=Grants.reader("hr"))
        out = _tool(view, "search_wiki")(question="cycle to work")
        assert "benefits:cycling" in out
        assert "you also have access to" not in out


class TestARefusedSlugIsNeverEchoed:
    def test_a_hidden_slug_from_the_retriever_is_dropped(
        self, store: WikiStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fail-open by construction otherwise: any retriever that
        surfaces a hidden slug — a stale index, a cache built before the
        page was restricted — turns into a disclosure of the slug
        itself. The read is refused, so the row must go too."""
        from outmem.optimize.blocks import RetrievalResult

        view = store.as_viewer()
        monkeypatch.setattr(
            "outmem.optimize.blocks.BM25Retriever.retrieve",
            lambda self, question, *, k: RetrievalResult(
                ("hr:parental-leave", "benefits:cycling")
            ),
        )
        out = _tool(view, "search_wiki")(question="anything")
        assert "hr:parental-leave" not in out
        assert "benefits:cycling" in out

    def test_an_all_hidden_result_reads_as_no_match(
        self, store: WikiStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from outmem.optimize.blocks import RetrievalResult

        view = store.as_viewer()
        monkeypatch.setattr(
            "outmem.optimize.blocks.BM25Retriever.retrieve",
            lambda self, question, *, k: RetrievalResult(("hr:parental-leave",)),
        )
        assert "no pages matched" in _tool(view, "search_wiki")(question="x")


class TestPaletteReduction:
    """Shrink before filtering: a tool not exposed is a bypass class
    that needs no code, no test, and cannot regress."""

    def test_the_history_pair_is_absent_from_a_view(
        self, store: WikiStore
    ) -> None:
        names = {t.__name__ for t in wiki_read_tools(store.as_viewer())}
        assert "page_history" not in names
        assert "topic_evolution" not in names

    def test_they_survive_on_an_unfiltered_store(self, store: WikiStore) -> None:
        names = {t.__name__ for t in wiki_read_tools(store)}
        assert {"page_history", "topic_evolution"} <= names

    def test_the_write_palette_is_reduced_the_same_way(
        self, store: WikiStore
    ) -> None:
        names = {t.__name__ for t in wiki_tools(store.as_viewer())}
        assert "topic_evolution" not in names
        assert "write_page" in names

    def test_the_rest_of_the_read_palette_is_unchanged(
        self, store: WikiStore
    ) -> None:
        names = {t.__name__ for t in wiki_read_tools(store.as_viewer())}
        assert names == {
            "search_wiki",
            "grep_wiki",
            "read_page",
            "list_pages",
            "search_index",
            "find_backlinks",
            "list_sources",
            "read_source",
        }


class TestToolLoggingIsRedacted:
    """`_log_call` attaches its kwargs to every LogRecord and Logfire is
    a handler, so an unredacted body exports restricted page text to an
    observability backend outside the deployment."""

    def _records(self, caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
        return [r for r in caplog.records if hasattr(r, "tool_call")]

    def test_a_page_body_does_not_reach_the_structured_payload(
        self, store: WikiStore, caplog: pytest.LogCaptureFixture
    ) -> None:
        secret = "Twelve weeks of severance at full pay."
        with caplog.at_level(logging.INFO, logger="outmem.agent.tool"):
            tool = next(
                t for t in wiki_tools(store) if t.__name__ == "write_page"
            )
            tool(slug="hr:sev", title="Sev", body=secret)
        payloads = [r.tool_call for r in self._records(caplog)]
        assert payloads
        assert not any(secret in str(p) for p in payloads)

    def test_the_redaction_keeps_the_length(
        self, store: WikiStore, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A trace that says nothing about size is a worse trace; the
        length is not the content."""
        with caplog.at_level(logging.INFO, logger="outmem.agent.tool"):
            tool = next(
                t for t in wiki_tools(store) if t.__name__ == "write_page"
            )
            tool(slug="hr:sev", title="Sev", body="x" * 40)
        payload = self._records(caplog)[0].tool_call[1]
        assert payload["body"] == "(40 chars, redacted)"

    def test_references_are_logged_on_a_wiki_with_nothing_to_protect(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Redaction must not blind the trace by default: slugs, paths
        and tags are what make a tool log useful, and none of them is
        content. Every wiki that existed before this feature is here."""
        plain = WikiStore.init(tmp_path / "plain")
        plain.write_page("benefits:cycling", title="C", body="Text.\n")
        with caplog.at_level(logging.INFO, logger="outmem.agent.tool"):
            _tool(plain, "read_page")(slug="benefits:cycling")
        assert self._records(caplog)[0].tool_call[1]["slug"] == "benefits:cycling"

    def test_references_are_masked_once_the_wiki_declares_labels(
        self, store: WikiStore, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A slug can be as disclosing as a filename (`hr:alice-severance`),
        and a source path embeds its original name. This record reaches
        every handler, Logfire among them, which sends it outside the
        deployment — so the wikis with something to protect get the
        safer trace without having to ask for it."""
        with caplog.at_level(logging.INFO, logger="outmem.agent.tool"):
            _tool(store, "read_page")(slug="hr:parental-leave")
        record = self._records(caplog)[0]
        assert "parental-leave" not in str(record.tool_call)
        assert "parental-leave" not in record.getMessage()
        assert record.tool_call[0] == "read_page"  # the verb still shows


class TestEveryLoggedArgumentIsClassified:
    """The redaction set was written from a list of tool signatures, and
    the list was wrong — `content` (a whole log entry), `title`, `topic`
    and `section` all reached the LogRecord verbatim while the set
    claimed to cover content.

    Enumerating the call sites is the only version of this check that
    cannot drift: adding a tool with a new content argument fails here
    until somebody decides which side of the line it is on.
    """

    def _logged_kwargs(self) -> set[str]:
        import ast
        import inspect

        import outmem.adapters.pydantic_ai as mod

        tree = ast.parse(inspect.getsource(mod))
        return {
            kw.arg
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_log"
            for kw in node.keywords
            if kw.arg
        }

    def test_no_logged_argument_is_unclassified(self) -> None:
        from outmem.adapters.pydantic_ai import (
            _CONTENT_ARGS,
            _QUERY_ARGS,
            _REFERENCE_ARGS,
        )

        unclassified = (
            self._logged_kwargs() - _CONTENT_ARGS - _REFERENCE_ARGS - _QUERY_ARGS
        )
        assert not unclassified, (
            f"tool arguments not classified for logging: {sorted(unclassified)}. "
            "Decide whether each carries page or source text (add to "
            "_CONTENT_ARGS) or names something (add to _REFERENCE_ARGS / "
            "_QUERY_ARGS)."
        )

    def test_the_sets_are_disjoint(self) -> None:
        from outmem.adapters.pydantic_ai import (
            _CONTENT_ARGS,
            _QUERY_ARGS,
            _REFERENCE_ARGS,
        )

        assert not _CONTENT_ARGS & _REFERENCE_ARGS
        assert not _CONTENT_ARGS & _QUERY_ARGS

    def test_a_log_topic_is_redacted_in_both_halves(
        self, store: WikiStore, caplog: pytest.LogCaptureFixture
    ) -> None:
        """`_summarise` only collapses strings over 60 characters, so a
        topic — short by nature — appeared verbatim in the formatted
        message even while the structured payload was clean."""
        with caplog.at_level(logging.INFO, logger="outmem.agent.tool"):
            tool = next(
                t for t in wiki_tools(store) if t.__name__ == "append_log"
            )
            tool(topic="severance cap", content="Twelve weeks.\n")
        record = next(r for r in caplog.records if hasattr(r, "tool_call"))
        assert "severance cap" not in record.getMessage()
        assert "severance cap" not in str(record.tool_call)

    def test_an_unset_default_is_not_dressed_up_as_withheld_content(
        self, store: WikiStore, caplog: pytest.LogCaptureFixture
    ) -> None:
        """`read_page(section="")` means "no section asked for". Logging
        that as "(0 chars, redacted)" implies something was taken away
        and makes the line harder to read than the truth."""
        with caplog.at_level(logging.INFO, logger="outmem.agent.tool"):
            _tool(store, "read_page")(slug="benefits:cycling")
        payload = next(
            r for r in caplog.records if hasattr(r, "tool_call")
        ).tool_call[1]
        assert payload["section"] == ""

    def test_a_page_title_is_redacted(
        self, store: WikiStore, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO, logger="outmem.agent.tool"):
            tool = next(t for t in wiki_tools(store) if t.__name__ == "write_page")
            tool(slug="x", title="Q4 layoff list", body="Text.\n")
        record = next(r for r in caplog.records if hasattr(r, "tool_call"))
        assert "layoff" not in record.getMessage()
        assert "layoff" not in str(record.tool_call)


class TestConsultWikiThreadsTheView:
    def test_a_store_argument_is_used_as_given(
        self, store: WikiStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Re-opening from a path would discard the caller's mode and
        grants and serve the whole wiki — a complete bypass reached by
        passing the obvious argument."""
        from outmem.adapters.pydantic_ai import build_consult_wiki

        captured: dict[str, object] = {}

        class _FakeAgent:
            def __init__(self, model, *, tools, system_prompt, **kwargs):  # type: ignore[no-untyped-def]
                captured["tools"] = tools

        monkeypatch.setattr("pydantic_ai.Agent", _FakeAgent)
        view = store.as_viewer(mode={"hr"}, grants=Grants.reader("hr"))
        build_consult_wiki(view)
        names = {t.__name__ for t in captured["tools"]}  # type: ignore[union-attr]
        assert "page_history" not in names  # the view's reduced palette

    def test_a_path_still_works(
        self, store: WikiStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from outmem.adapters.pydantic_ai import build_consult_wiki

        class _FakeAgent:
            def __init__(self, model, *, tools, system_prompt, **kwargs):  # type: ignore[no-untyped-def]
                pass

        monkeypatch.setattr("pydantic_ai.Agent", _FakeAgent)
        build_consult_wiki(store.root)


class TestSemanticOverFetch:
    def test_the_visible_count_does_not_depend_on_the_viewer(
        self, store: WikiStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Filtering a fixed top-k makes the result count an existence
        oracle: three results where a cleared user gets eight says five
        restricted items sit near that topic."""
        from outmem.semantic.store import Match

        pages = f"{store.config.wiki_dir}/pages"
        rows = [
            Match(f"{pages}/hr/x{i}.md", 0, 0.9, "restricted", 0, 1)
            for i in range(6)
        ] + [
            Match(f"{pages}/benefits/cycling.md", 0, 0.8, "open", 0, 1),
            Match(f"{pages}/open2.md", 0, 0.7, "open", 0, 1),
            Match(f"{pages}/open3.md", 0, 0.6, "open", 0, 1),
        ]
        for i in range(6):
            store.write_page(
                f"hr:x{i}", title=f"X{i}", body="Text.\n", extra={"restricted": ["hr"]}
            )
        store.write_page("open2", title="O2", body="Text.\n")
        store.write_page("open3", title="O3", body="Text.\n")

        def _fake(store_, text, *, top_k, threshold, exclude_slug):  # type: ignore[no-untyped-def]
            return rows[:top_k]

        monkeypatch.setattr("outmem._store.semantic.find_similar", _fake)
        uncleared = store.as_viewer().semantic_find_similar("anything", top_k=3)
        cleared = store.as_viewer(
            mode={"hr"}, grants=Grants.reader("hr")
        ).semantic_find_similar("anything", top_k=3)
        # The count is what leaks. Both viewers get k; only the contents
        # differ, so the size of the answer says nothing about how much
        # was filtered out of it.
        assert len(uncleared) == len(cleared) == 3
        assert all("hr/" not in m.rel_path for m in uncleared)
        assert any("hr/" in m.rel_path for m in cleared)

    def test_it_stops_when_the_index_runs_out(
        self, store: WikiStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A query whose whole neighbourhood is restricted must return
        short, not spin to the ceiling on every call."""
        from outmem.semantic.store import Match

        pages = f"{store.config.wiki_dir}/pages"
        rows = [Match(f"{pages}/hr/only.md", 0, 0.9, "x", 0, 1)]
        store.write_page(
            "hr:only", title="Only", body="Text.\n", extra={"restricted": ["hr"]}
        )
        calls = {"n": 0}

        def _fake(store_, text, *, top_k, threshold, exclude_slug):  # type: ignore[no-untyped-def]
            calls["n"] += 1
            return rows[:top_k]

        monkeypatch.setattr("outmem._store.semantic.find_similar", _fake)
        assert store.as_viewer().semantic_find_similar("x", top_k=5) == []
        # Bounded, not zero: the loop must notice the index is exhausted
        # rather than widening to the ceiling on every such query.
        assert calls["n"] <= 2
