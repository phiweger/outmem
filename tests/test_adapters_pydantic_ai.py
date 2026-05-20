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
        provenance=["raw/deck.md"],
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
        "read_page",
        "list_pages",
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


def test_search_wiki_returns_rg_format(seeded_store: WikiStore) -> None:
    tools = wiki_tools(seeded_store)
    out = _by_name(tools, "search_wiki")(pattern="cost-plus")
    assert "pricing-formula.md" in out
    assert ":" in out  # path:line:text


def test_search_wiki_no_match(seeded_store: WikiStore) -> None:
    out = _by_name(wiki_tools(seeded_store), "search_wiki")(pattern="absent-token")
    assert out == "(no matches)"


def test_read_page_returns_full_file(seeded_store: WikiStore) -> None:
    out = _by_name(wiki_tools(seeded_store), "read_page")(slug="pricing-formula")
    assert "title: Pricing formula" in out
    assert "cost-plus 35%" in out


def test_list_pages(seeded_store: WikiStore) -> None:
    out = _by_name(wiki_tools(seeded_store), "list_pages")()
    assert out.split("\n") == ["acme-msa", "pricing-formula"]


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
        provenance=["raw/discount-table.md"],
        tags=["pricing"],
    )
    assert len(sha) == 40
    page = seeded_store.read("discounts")
    assert page.frontmatter.title == "Discounts"
    assert page.frontmatter.provenance == ["raw/discount-table.md"]


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
    _by_name(tools, "search_wiki")(pattern="cost-plus")
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


def test_wiki_read_tools_includes_find_similar_when_semantic_enabled(
    tmp_path: Path,
) -> None:
    """Positive case of the semantic gating — when ``semantic.enabled``
    is True, ``find_similar`` makes it into the read-tool list. Pair
    with the implicit negative case (find_similar absent when semantic
    is off — every other read-tool test demonstrates that)."""
    root = tmp_path / "w"
    root.mkdir()
    (root / "config.yaml").write_text(
        "semantic:\n  enabled: true\n  embedding_model: bag-of-words:stub\n",
        encoding="utf-8",
    )
    # Init after writing config so the WikiStore picks it up.
    store = WikiStore.init(root)
    assert store.semantic_enabled()
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
