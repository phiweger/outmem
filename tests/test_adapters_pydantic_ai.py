"""Tests for ``outmem.adapters.pydantic_ai``.

We exercise the tool functions both directly (verifying they wrap the
store correctly) and through PydanticAI's :class:`TestModel`
(verifying the adapter is genuinely attachable to an agent).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from outmem.adapters.pydantic_ai import (
    build_consult_wiki,
    skill_text,
    wiki_read_tools,
    wiki_tools,
)
from outmem.store import WikiStore

# ---------------------------------------------------------------------------
# Direct-invocation tests — the tools as plain callables
# ---------------------------------------------------------------------------


@pytest.fixture
def seeded_store(tmp_path: Path) -> WikiStore:
    store = WikiStore.init(tmp_path / "wiki")
    store.write_page(
        "pricing-formula",
        title="Pricing formula",
        body="The pricing formula is cost-plus 35%.\n",
        provenance=["sources/deck.md"],
        tags=["pricing"],
    )
    store.write_page(
        "acme-msa",
        title="Acme MSA",
        body="See [[pricing-formula]] for the standard rate.\n",
    )
    return store


def _by_name(tools: list, name: str):
    for tool in tools:
        if tool.__name__ == name:
            return tool
    raise AssertionError(f"missing tool: {name}")


def test_wiki_tools_returns_expected_set(seeded_store: WikiStore) -> None:
    names = [t.__name__ for t in wiki_tools(seeded_store)]
    assert set(names) == {
        "search_wiki",
        "grep_wiki",
        "read_page",
        "list_pages",
        "search_index",
        "find_backlinks",
        "page_history",
        "topic_evolution",
        "write_page",
        "extend_page",
        "append_log",
        # Source / ingestion tools (added in the ingestion PR).
        "list_sources",
        "read_source",
        "record_ingestion",
    }


def test_grep_wiki_returns_slug_keyed_rows(seeded_store: WikiStore) -> None:
    """``scope="wiki"`` rows lead with the slug, not the on-disk path,
    so the agent can pass the leading token straight to ``read_page``."""
    tools = wiki_tools(seeded_store)
    out = _by_name(tools, "grep_wiki")(pattern="cost-plus")
    # No `.md` and no `/` in the slug-shaped leading token.
    first_line = out.splitlines()[0]
    leading = first_line.split(":")[0]
    assert leading == "pricing-formula"
    assert ".md" not in first_line.split(":", 2)[0]


def test_grep_wiki_emits_namespaced_slug(tmp_path: Path) -> None:
    """Hits in a nested page (``wiki/pages/abx/penicillin.md``) come back
    as ``abx:penicillin:line:text`` — slug, not path."""
    from outmem.store import WikiStore

    store = WikiStore.init(tmp_path / "w")
    store.write_page("abx:penicillin", title="P", body="A beta-lactam.")
    out = _by_name(wiki_tools(store), "grep_wiki")(pattern="beta-lactam")
    first = out.splitlines()[0]
    assert first.startswith("abx:penicillin:")
    # The middle is the line number, then ``:`` separator, then content.
    parts = first.split(":", 3)
    assert parts[0] == "abx"
    assert parts[1] == "penicillin"
    assert parts[2].isdigit()  # line number


def test_grep_wiki_raw_scope_keeps_paths(tmp_path: Path) -> None:
    """Non-wiki scopes keep path-shaped output; only ``scope="wiki"``
    converts to slugs."""
    from outmem.store import WikiStore

    store = WikiStore.init(tmp_path / "w")
    (store.sources_path / "deck.md").write_text(
        "Slide 3: cost-plus 35%.\n", encoding="utf-8"
    )
    out = _by_name(wiki_tools(store), "grep_wiki")(pattern="cost-plus", scope="sources")
    assert "deck.md" in out  # path preserved


def test_grep_wiki_no_match(seeded_store: WikiStore) -> None:
    out = _by_name(wiki_tools(seeded_store), "grep_wiki")(pattern="absent-token")
    assert out == "(no matches)"


def test_read_page_returns_full_file(seeded_store: WikiStore) -> None:
    out = _by_name(wiki_tools(seeded_store), "read_page")(slug="pricing-formula")
    assert "title: Pricing formula" in out
    assert "cost-plus 35%" in out


def test_list_pages(seeded_store: WikiStore) -> None:
    out = _by_name(wiki_tools(seeded_store), "list_pages")()
    assert out.split("\n") == ["acme-msa", "pricing-formula"]


def test_read_page_peek_outlines_a_sectioned_page(seeded_store: WikiStore) -> None:
    """A peek is a map, not a prefix: it must name every section however
    long the page, including ones far past any character budget."""
    seeded_store.write_page(
        "long-page",
        title="Long",
        body=("## First\n" + "x" * 3000 + "\n\n## Buried\ndeep content\n"),
    )
    out = _by_name(wiki_tools(seeded_store), "read_page")(slug="long-page", peek=True)
    assert out.startswith("# Long")
    assert "First" in out
    assert "Buried" in out          # 3000 chars past a prefix window
    assert "x" * 100 not in out     # the body itself is not in the map


def test_read_page_peek_on_a_headingless_page_returns_it(
    seeded_store: WikiStore,
) -> None:
    """An empty outline would be useless. A short unsectioned page is
    handed over rather than making the caller ask a second time — which
    is the round-trip this whole change is about."""
    out = _by_name(wiki_tools(seeded_store), "read_page")(
        slug="pricing-formula", peek=True
    )
    assert "no sections" in out
    assert "cost-plus 35%" in out


@pytest.fixture
def namespaced_store(tmp_path: Path) -> WikiStore:
    store = WikiStore.init(tmp_path / "wiki")
    store.write_page("pricing-formula", title="Pricing", body="b\n")
    store.write_page("abx:penicillin", title="Pen", body="b\n")
    store.write_page("abx:ceftriaxone", title="Cef", body="b\n")
    store.write_page("abx:side-effects:rash", title="Rash", body="b\n")
    return store


def test_search_index_root_shows_namespaces_and_top_pages(
    namespaced_store: WikiStore,
) -> None:
    out = _by_name(wiki_tools(namespaced_store), "search_index")()
    assert "index (root):" in out
    assert "abx:  (3 pages)" in out  # penicillin + ceftriaxone + side-effects:rash
    assert "pricing-formula" in out  # a top-level leaf page
    assert "penicillin" not in out  # hidden until you drill into abx:


def test_search_index_drills_into_namespace(namespaced_store: WikiStore) -> None:
    out = _by_name(wiki_tools(namespaced_store), "search_index")(prefix="abx")
    assert "index of abx:" in out
    assert "abx:side-effects:  (1 page)" in out  # singular, one page below
    assert "abx:penicillin" in out
    assert "abx:ceftriaxone" in out


def test_search_index_unknown_prefix(namespaced_store: WikiStore) -> None:
    out = _by_name(wiki_tools(namespaced_store), "search_index")(prefix="nope")
    assert out == "(nothing under 'nope')"


def test_find_backlinks(seeded_store: WikiStore) -> None:
    out = _by_name(wiki_tools(seeded_store), "find_backlinks")(slug="pricing-formula")
    assert out == "acme-msa"


def test_find_backlinks_empty(seeded_store: WikiStore) -> None:
    out = _by_name(wiki_tools(seeded_store), "find_backlinks")(slug="acme-msa")
    assert out == "(no backlinks)"


def test_page_history(seeded_store: WikiStore) -> None:
    out = _by_name(wiki_tools(seeded_store), "page_history")(slug="pricing-formula")
    assert "compact: pricing-formula" in out


def test_topic_evolution_returns_diff(seeded_store: WikiStore) -> None:
    seeded_store.extend_page("pricing-formula", body="updated formula\n")
    out = _by_name(wiki_tools(seeded_store), "topic_evolution")(slugs=["pricing-formula"])
    assert "diff --git" in out
    assert "updated formula" in out


def test_topic_evolution_requires_slug(seeded_store: WikiStore) -> None:
    out = _by_name(wiki_tools(seeded_store), "topic_evolution")(slugs=[])
    assert "requires at least one slug" in out


def test_write_page_creates_new(seeded_store: WikiStore) -> None:
    sha = _by_name(wiki_tools(seeded_store), "write_page")(
        slug="discounts",
        title="Discounts",
        body="Standard discount tiers.\n",
        provenance=["sources/discount-table.md"],
        tags=["pricing"],
    )
    assert len(sha) == 40
    page = seeded_store.read("discounts")
    assert page.frontmatter.title == "Discounts"
    assert page.frontmatter.provenance == ["sources/discount-table.md"]


def test_extend_page_replaces_body(seeded_store: WikiStore) -> None:
    sha = _by_name(wiki_tools(seeded_store), "extend_page")(
        slug="pricing-formula",
        body="Revised: cost-plus 40%.\n",
    )
    assert len(sha) == 40
    page = seeded_store.read("pricing-formula")
    assert "40%" in page.body


def test_append_log_creates_entry(seeded_store: WikiStore) -> None:
    sha = _by_name(wiki_tools(seeded_store), "append_log")(
        topic="pricing-inconsistency",
        content="- noticed pricing mismatch between deck and msa.\n",
    )
    assert len(sha) == 40
    log_files = list(seeded_store.log_path.glob("*.md"))
    assert len(log_files) == 1
    assert "pricing mismatch" in log_files[0].read_text()


# ---------------------------------------------------------------------------
# Attach to a PydanticAI Agent and verify schema extraction works
# ---------------------------------------------------------------------------


def test_attach_to_agent_with_test_model(seeded_store: WikiStore) -> None:
    """The tools must be valid PydanticAI tools — schema extracts cleanly
    and the agent can be constructed without errors. We don't run the
    agent here: TestModel's default behaviour is to fire every tool with
    placeholder arguments, which would call e.g. ``read_page(slug='a')``
    and surface a domain error rather than a schema error."""
    tools = wiki_tools(seeded_store)
    model = TestModel(call_tools=[])
    agent = Agent(model, tools=tools)
    result = agent.run_sync("Find anything about pricing.")
    assert isinstance(result.output, str)


def test_docstrings_lead_with_required_args() -> None:
    """AGENTS.md §"Tool docstrings" — multi-required-arg tools must
    flag their required-count loudly so models don't drop arguments."""
    # Get a dummy store just to instantiate the closures.
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        store = WikiStore.init(Path(tmp) / "w")
        tools = wiki_tools(store)

    multi_required = {
        "write_page": "REQUIRES ALL THREE",
        "extend_page": "REQUIRES BOTH",
        "append_log": "REQUIRES BOTH",
    }
    for tool in tools:
        prefix = multi_required.get(tool.__name__)
        if prefix is None:
            continue
        assert prefix in (tool.__doc__ or ""), (
            f"{tool.__name__} docstring missing required-args prefix"
        )


# ---------------------------------------------------------------------------
# wiki_read_tools — read-only subset for consult subagents
# ---------------------------------------------------------------------------


def test_wiki_read_tools_drops_write_tools(seeded_store: WikiStore) -> None:
    names = {t.__name__ for t in wiki_read_tools(seeded_store)}
    # Read paths survive.
    assert {
        "search_wiki",
        "read_page",
        "list_pages",
        "search_index",
        "find_backlinks",
        "page_history",
        "topic_evolution",
        "list_sources",
        "read_source",
    } <= names
    # Every commit-producing tool is dropped.
    assert names.isdisjoint({"write_page", "extend_page", "append_log", "record_ingestion"})


def test_wiki_read_tools_still_attachable_to_agent(seeded_store: WikiStore) -> None:
    tools = wiki_read_tools(seeded_store)
    model = TestModel(call_tools=[])
    agent = Agent(model, tools=tools)
    result = agent.run_sync("Find anything about pricing.")
    assert isinstance(result.output, str)


def test_wiki_read_tools_with_read_only_store_cannot_commit(
    tmp_path: Path,
) -> None:
    """A read-only store + wiki_read_tools: there is no exposed path
    that produces a commit, regardless of what the model tries."""
    seed = WikiStore.init(tmp_path / "w")
    seed.write_page("pricing", title="Pricing", body="Cost-plus 35%.\n")
    head_before = seed.head()
    seed.close()

    ro = WikiStore.open(seed.root, read_only=True)
    tools = wiki_read_tools(ro)
    # Every survivor can be called without raising — they are all pure
    # retrieval and do not flow through `_commit_paths`.
    _by_name(tools, "grep_wiki")(pattern="cost-plus")
    _by_name(tools, "read_page")(slug="pricing")
    _by_name(tools, "list_pages")()
    assert ro.head() == head_before


def test_build_consult_wiki_returns_callable(tmp_path: Path) -> None:
    """The factory returns a single-arg ``consult_wiki(question)`` function
    with a docstring describing the WHEN of using it (not the HOW of the
    wiki). We pass a TestModel so construction doesn't require an
    Anthropic API key."""
    seed = WikiStore.init(tmp_path / "w")
    seed.write_page("pricing", title="Pricing", body="Cost-plus 35%.\n")
    seed.close()

    consult = build_consult_wiki(seed.root, model=TestModel(call_tools=[]))
    assert callable(consult)
    assert consult.__name__ == "consult_wiki"
    doc = consult.__doc__ or ""
    assert "knowledge base" in doc.lower()
    # No outmem-internal tool names leak into the docstring the outer
    # agent will see — the encapsulation is the whole point.
    for term in ("write_page", "extend_page", "append_log", "search_wiki"):
        assert term not in doc, f"outmem internal {term!r} leaked into consult_wiki docstring"


def test_build_consult_wiki_inner_run_does_not_mutate_wiki(tmp_path: Path) -> None:
    """End-to-end with TestModel calling every available tool: invoking
    consult_wiki must not produce a commit on the underlying wiki. Uses
    the default ``call_tools='all'`` so TestModel actually exercises
    every read tool with dummy arguments — anything that flows through
    ``_commit_paths`` would surface as a HEAD change."""
    seed = WikiStore.init(tmp_path / "w")
    seed.write_page("pricing", title="Pricing", body="Cost-plus 35%.\n")
    head_before = seed.head()
    seed.close()

    consult = build_consult_wiki(seed.root, model=TestModel())
    answer = consult("What's our pricing?")
    assert isinstance(answer, str)
    # Verify nothing was committed by reopening as a writable store and
    # comparing HEAD.
    ro = WikiStore.open(seed.root)
    assert ro.head() == head_before


def test_build_consult_wiki_inner_settings_match_runtime(tmp_path: Path) -> None:
    """The inner agent must carry the same Anthropic prompt-caching +
    max_tokens settings as the full ``outmem ask`` runtime. Without
    these, multi-page reads truncate (the 4096-token default eats tool
    JSON) and each call re-bills the system prompt and tool defs.
    """
    seed = WikiStore.init(tmp_path / "w")
    seed.close()
    consult = build_consult_wiki(seed.root, model=TestModel())
    inner = consult.__closure__[0].cell_contents  # type: ignore[index]
    settings = inner.model_settings or {}
    assert settings.get("max_tokens") == 16384
    assert settings.get("anthropic_cache") is True
    assert settings.get("anthropic_cache_instructions") is True
    assert settings.get("anthropic_cache_tool_definitions") is True


def test_build_consult_wiki_missing_path_raises(tmp_path: Path) -> None:
    """A clear OutmemError when the wiki path doesn't exist — the
    failure happens at factory time, not when consult_wiki is called."""
    from outmem.exceptions import OutmemError

    with pytest.raises(OutmemError, match="does not exist"):
        build_consult_wiki(tmp_path / "nope", model=TestModel())


def test_search_wiki_uses_configured_strategy(seeded_store: WikiStore) -> None:
    """``search_wiki`` reads ``RetrievalSettings`` and ranks via the
    configured pipeline. Returns slugs the agent can then ``read_page``
    on. Regression for the optimizer→production seam: the optim's picked
    config must actually change what the agent's search tool returns.

    Pinned to ``bm25`` so the test is model-free and deterministic — the
    out-of-box default is ``rerank(bm25)``, which would make a Haiku call
    per query."""
    seeded_store.config.outmem.retrieval.strategy = "bm25"
    tools = wiki_read_tools(seeded_store)
    search_wiki = _by_name(tools, "search_wiki")
    out = search_wiki(question="cost-plus pricing formula", k=3)
    # Returns slug-citation lines; the seeded fixture has a pricing page.
    assert "[[" in out and "]]" in out


def test_wiki_read_tools_includes_find_similar_when_index_built(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Positive case of the semantic gating — when the semantic index is
    available (built), ``find_similar`` makes it into the read-tool list.
    Pair with the implicit negative case (find_similar absent when there's
    no index — every other read-tool test demonstrates that)."""
    store = WikiStore.init(tmp_path / "w")
    monkeypatch.setattr(store, "semantic_available", lambda: True)
    names = {t.__name__ for t in wiki_read_tools(store)}
    assert "find_similar" in names


# ---------------------------------------------------------------------------
# skill_text
# ---------------------------------------------------------------------------


def test_skill_text_loads_from_supplied_dir(tmp_path: Path) -> None:
    skills = tmp_path / "skills"
    (skills / "notes" / "demo").mkdir(parents=True)
    (skills / "notes" / "demo" / "SKILL.md").write_text(
        "---\nname: demo\ndescription: x\n---\n\ndemo body\n",
        encoding="utf-8",
    )
    out = skill_text("demo", skills_dir=skills)
    assert "demo body" in out


def test_skill_text_unknown_raises(tmp_path: Path) -> None:
    from outskilled import UnknownSkillError

    skills = tmp_path / "skills"
    (skills / "notes").mkdir(parents=True)
    with pytest.raises(UnknownSkillError, match="Unknown skill"):
        skill_text("missing", skills_dir=skills)


# ---------------------------------------------------------------------------
# grep_wiki context — issue: a one-line fact should not cost a read_page
# ---------------------------------------------------------------------------


class TestGrepWikiContext:
    @pytest.fixture
    def ctx_store(self, tmp_path: Path) -> WikiStore:
        store = WikiStore.init(tmp_path / "w")
        store.write_page(
            "ifsg",
            title="IfSG",
            body=(
                "## Labormeldepflicht\n"
                "namentlich zu melden ist der direkte NACHWEIS\n"
                "unverzueglich, spaetestens 24 Stunden\n"
                "\n"
                "## Fristen\n"
                "unrelated\n"
            ),
        )
        return store

    def _grep(self, store: WikiStore):
        return _by_name(wiki_tools(store), "grep_wiki")

    def test_context_returns_surrounding_lines(self, ctx_store: WikiStore) -> None:
        out = self._grep(ctx_store)(pattern="NACHWEIS", context=2)
        assert "Labormeldepflicht" in out          # the line before
        assert "24 Stunden" in out                 # the line after

    def test_match_and_context_rows_are_distinguishable(
        self, ctx_store: WikiStore
    ) -> None:
        """ripgrep's convention: ':' for matches, '-' for context."""
        out = self._grep(ctx_store)(pattern="NACHWEIS", context=1)
        match_rows = [ln for ln in out.splitlines() if "NACHWEIS" in ln]
        context_rows = [
            ln for ln in out.splitlines() if ln and "NACHWEIS" not in ln
        ]
        assert all(":" in ln.split("ifsg", 1)[1][:1] for ln in match_rows)
        assert all(ln.startswith("ifsg-") for ln in context_rows)

    def test_context_rows_keep_the_slug_leading_token(
        self, ctx_store: WikiStore
    ) -> None:
        """A slug read off a context row must still work with read_page —
        otherwise the caller has to reconstruct it."""
        out = self._grep(ctx_store)(pattern="NACHWEIS", context=2)
        for line in out.splitlines():
            if not line:
                continue
            assert line.split(":")[0].split("-")[0] == "ifsg"
        read = _by_name(wiki_tools(ctx_store), "read_page")
        assert "IfSG" in read(slug="ifsg")

    def test_context_zero_is_unchanged(self, ctx_store: WikiStore) -> None:
        """The acceptance criterion: existing callers see byte-identical
        output. In particular, no blank-line grouping appears."""
        grep = self._grep(ctx_store)
        ctx_store.write_page("other", title="Other", body="another NACHWEIS here\n")
        out = grep(pattern="NACHWEIS")
        assert "" not in out.splitlines()  # no group separators
        assert out == grep(pattern="NACHWEIS", context=0)

    def test_groups_are_blank_line_separated(self, ctx_store: WikiStore) -> None:
        ctx_store.write_page("other", title="Other", body="another NACHWEIS here\n")
        out = self._grep(ctx_store)(pattern="NACHWEIS", context=1)
        assert "" in out.splitlines()

    def test_context_is_clamped_not_rejected(self, ctx_store: WikiStore) -> None:
        """An out-of-range value is a slip, not a question worth a turn."""
        grep = self._grep(ctx_store)
        assert grep(pattern="NACHWEIS", context=999) == grep(
            pattern="NACHWEIS", context=10
        )
        assert grep(pattern="NACHWEIS", context=-5) == grep(
            pattern="NACHWEIS", context=0
        )

    def test_truncation_still_fires_with_context(self, tmp_path: Path) -> None:
        """A wide pattern plus context is exactly when the caller needs
        to be told to narrow it."""
        store = WikiStore.init(tmp_path / "wide")
        for i in range(60):
            store.write_page(
                f"p{i}", title=f"P{i}", body=("filler NACHWEIS line\n" * 40)
            )
        out = _by_name(wiki_tools(store), "grep_wiki")(
            pattern="NACHWEIS", context=5
        )
        assert "(truncated — narrow the pattern)" in out


# ---------------------------------------------------------------------------
# read_page peek/section — issue: triage should not cost a full read
# ---------------------------------------------------------------------------


class TestReadPageOutline:
    @pytest.fixture
    def ifsg(self, tmp_path: Path) -> WikiStore:
        store = WikiStore.init(tmp_path / "w")
        store.write_page(
            "meldewesen:ifsg",
            title="Meldepflicht nach IfSG",
            tags=["ifsg"],
            body=(
                "Vorbemerkung.\n\n"
                "## Arztmeldepflicht\n" + "arzt\n" * 8 + "\n"
                "## Labormeldepflicht\n"
                "lead-in\n\n"
                "### Abs. 1 namentlich\n" + "UNIQUEMARKER hier\n" + "n\n" * 6 + "\n"
                "### Abs. 3 nichtnamentlich\n" + "x\n" * 6 + "\n"
                "## Fristen\n" + "f\n" * 5
            ),
        )
        return store

    def _read(self, store: WikiStore):
        return _by_name(wiki_tools(store), "read_page")

    def test_outline_names_every_section_with_span_and_size(
        self, ifsg: WikiStore
    ) -> None:
        out = self._read(ifsg)(slug="meldewesen:ifsg", peek=True)
        for heading in (
            "Arztmeldepflicht",
            "Labormeldepflicht",
            "Abs. 1 namentlich",
            "Abs. 3 nichtnamentlich",
            "Fristen",
        ):
            assert heading in out
        assert "L" in out and "-" in out          # line spans
        assert "B" in out or "kB" in out          # sizes

    def test_outline_line_numbers_agree_with_grep_wiki(self, ifsg: WikiStore) -> None:
        """The map is only actionable if it describes the same coordinate
        space the other retrieval tool reports in."""
        tools = wiki_tools(ifsg)
        grep_out = _by_name(tools, "grep_wiki")(pattern="UNIQUEMARKER")
        grep_line = int(grep_out.split(":")[2])

        peek = self._read(ifsg)(slug="meldewesen:ifsg", peek=True)
        row = next(ln for ln in peek.splitlines() if "Abs. 1 namentlich" in ln)
        span = row.split("L")[-1].split()[0]
        start, end = (int(n) for n in span.split("-"))
        assert start <= grep_line <= end

    def test_section_returns_only_that_section(self, ifsg: WikiStore) -> None:
        out = self._read(ifsg)(slug="meldewesen:ifsg", section="Abs. 1 namentlich")
        assert "UNIQUEMARKER" in out
        assert "arzt" not in out          # sibling section absent
        assert "nichtnamentlich" not in out

    def test_section_match_is_forgiving(self, ifsg: WikiStore) -> None:
        """A rejection over casing or a stray space costs a round-trip."""
        read = self._read(ifsg)
        for query in ("abs. 1 namentlich", "  Abs. 1   namentlich ", "Abs. 1"):
            assert "UNIQUEMARKER" in read(slug="meldewesen:ifsg", section=query), query

    def test_ambiguous_section_reports_candidates(self, ifsg: WikiStore) -> None:
        """Picking one would look like an answer; the caller could not
        tell it was a coin flip."""
        out = self._read(ifsg)(slug="meldewesen:ifsg", section="meldepflicht")
        assert "matches 2 sections" in out
        assert "Arztmeldepflicht" in out
        assert "Labormeldepflicht" in out

    def test_unknown_section_lists_what_exists(self, ifsg: WikiStore) -> None:
        out = self._read(ifsg)(slug="meldewesen:ifsg", section="Nonexistent")
        assert "no section matching" in out
        assert "Fristen" in out

    def test_one_long_heading_does_not_pad_every_row(
        self, tmp_path: Path
    ) -> None:
        """The outline exists to save context; letting the longest
        heading set the column spends it back across every other row."""
        store = WikiStore.init(tmp_path / "wide")
        store.write_page(
            "p",
            title="P",
            body="## Short\na\n## " + "L" * 120 + "\nb\n## Another\nc\n",
        )
        out = _by_name(wiki_tools(store), "read_page")(slug="p", peek=True)
        assert max(len(line) for line in out.splitlines()) < 100

    def test_an_elided_heading_is_still_addressable(self, tmp_path: Path) -> None:
        """Truncation must not strand a section: section= matches on
        substrings, so the surviving prefix has to be enough."""
        store = WikiStore.init(tmp_path / "wide")
        store.write_page(
            "p", title="P", body="## " + "L" * 120 + "\nBURIED\n"
        )
        read = _by_name(wiki_tools(store), "read_page")
        peek = read(slug="p", peek=True)
        prefix = next(
            ln for ln in peek.splitlines() if "LLLL" in ln
        ).strip().split("  ")[0].rstrip("…")
        assert "BURIED" in read(slug="p", section=prefix)

    def test_section_wins_over_peek(self, ifsg: WikiStore) -> None:
        """Naming a section is the more specific request."""
        out = self._read(ifsg)(
            slug="meldewesen:ifsg", peek=True, section="Abs. 1 namentlich"
        )
        assert "UNIQUEMARKER" in out

    def test_full_read_is_unchanged(self, ifsg: WikiStore) -> None:
        """Neither new mode may alter the default: still the whole file,
        frontmatter included."""
        out = self._read(ifsg)(slug="meldewesen:ifsg")
        assert out.startswith("---")
        assert "title: Meldepflicht nach IfSG" in out
        assert "UNIQUEMARKER" in out
        assert "Fristen" in out
