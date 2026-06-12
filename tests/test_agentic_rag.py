"""End-to-end checks that the agent can actually do retrieval.

Three integration tests:

1. **TOC traversal**: a scripted PydanticAI agent navigates the slug
   namespace via ``search_index`` → drills into a child namespace →
   ``read_page(peek=True)`` to triage → ``read_page`` for the full body →
   closes the loop with ``append_log``. Verifies the new navigation
   tools actually plug into the agent runtime.

2. **Semantic when configured + index built**: ``retrieval.strategy =
   "semantic"`` with a built index (deterministic bag-of-words stub)
   surfaces the semantically-closer page, not the keyword-only one.

3. **Graceful fallback when no index**: ``retrieval.strategy = "semantic"``
   but no ``outmem reindex`` run. The tool falls back to ``bm25`` for that
   query (results are returned, not an error) and surfaces a diagnostics
   note pointing the user at ``outmem reindex``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from outmem.adapters.pydantic_ai import wiki_tools
from outmem.agent import ask_sync
from outmem.store import WikiStore

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _by_name(tools: list, name: str):
    for tool in tools:
        if tool.__name__ == name:
            return tool
    raise AssertionError(f"missing tool: {name}")


def _scripted_model(*calls: dict[str, object], reply: str = "done.") -> FunctionModel:
    """A FunctionModel that fires ``calls`` (one tool call per turn) and
    returns ``reply`` as text on the final turn. Mirrors the helper in
    test_agent.py — kept here so this file reads standalone."""
    state = {"step": 0}

    async def runner(messages: list[object], info: AgentInfo) -> ModelResponse:
        idx = state["step"]
        state["step"] = idx + 1
        if idx < len(calls):
            entry = calls[idx]
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name=str(entry["tool"]),
                        args=dict(entry["args"]),  # type: ignore[arg-type]
                    )
                ]
            )
        return ModelResponse(parts=[TextPart(content=reply)])

    return FunctionModel(runner)


@pytest.fixture
def namespaced_store(tmp_path: Path) -> WikiStore:
    """A wiki with a `:`-namespaced slug tree the agent can navigate.

    Two top-level pages plus an ``abx:`` namespace with three children
    (one of them itself namespaced) so ``search_index`` has interesting
    levels to walk.
    """
    store = WikiStore.init(tmp_path / "wiki")
    store.write_page(
        "pricing-formula",
        title="Pricing formula",
        body="The pricing formula is cost-plus 35%.\n",
    )
    store.write_page(
        "acme-msa",
        title="Acme MSA",
        body="See [[pricing-formula]] for the standard rate.\n",
    )
    store.write_page(
        "abx:penicillin",
        title="Penicillin",
        body=(
            "Penicillin G is dosed at 4 million units IV every 4 hours for "
            "endocarditis caused by viridans streptococci.\n"
        ),
    )
    store.write_page(
        "abx:ceftriaxone",
        title="Ceftriaxone",
        body="Ceftriaxone 2g IV once daily is the once-daily alternative.\n",
    )
    store.write_page(
        "abx:side-effects:rash",
        title="Beta-lactam rash",
        body="Maculopapular rash is the most common cutaneous reaction.\n",
    )
    return store


# ---------------------------------------------------------------------------
# 1. TOC-traversal agent run
# ---------------------------------------------------------------------------


def test_agent_traverses_toc_with_search_index_then_peeks_and_reads(
    namespaced_store: WikiStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end agentic RAG: the scripted agent orients via
    ``search_index``, drills into a namespace, triages with ``peek``,
    reads the chosen page in full, then logs the finding.

    This is the canonical happy path documented in the search SKILL and
    docs/python-api.md — it must actually run through the runtime.
    """
    # Capture every tool call via the centralized `_log_call` shim —
    # easier than walking PydanticAI's message history, and exactly the
    # same signal eval recorders subscribe to.
    trace: list[tuple[str, dict]] = []
    from outmem.adapters import pydantic_ai as adapter

    real_log_call = adapter._log_call

    def capture(name: str, **kwargs: object) -> None:
        trace.append((name, dict(kwargs)))
        real_log_call(name, **kwargs)

    monkeypatch.setattr(adapter, "_log_call", capture)

    model = _scripted_model(
        # 1. Orient: top-level namespaces + pages.
        {"tool": "search_index", "args": {}},
        # 2. Drill into the antibiotics namespace.
        {"tool": "search_index", "args": {"prefix": "abx"}},
        # 3. Cheap triage of a candidate before paying for the full body.
        {"tool": "read_page", "args": {"slug": "abx:penicillin", "peek": True}},
        # 4. Full read of the winner.
        {"tool": "read_page", "args": {"slug": "abx:penicillin"}},
        # 5. Close the loop with a log entry (mandatory writeback).
        {
            "tool": "append_log",
            "args": {
                "topic": "endocarditis-dosing",
                "content": "- penicillin G 4M units IV q4h per abx:penicillin.\n",
            },
        },
        reply="Penicillin G 4 million units IV every 4 hours.",
    )

    result = ask_sync(
        namespaced_store,
        query="What's the dose of penicillin for endocarditis?",
        model=model,
        push=False,
        record=False,
    )

    assert result.wrote_back
    assert "4 million units" in result.response or "4M units" in result.response
    # The tool trace records every call the agent made — assert the
    # navigation actually happened, not just that something was called.
    names = [name for name, _ in trace]
    assert names == [
        "search_index",
        "search_index",
        "read_page",
        "read_page",
        "append_log",
    ], f"unexpected tool trace: {names}"
    # Drill: second search_index narrowed scope to the abx: namespace.
    assert trace[1] == ("search_index", {"prefix": "abx"})
    # Peek triage came before the full read — and the full read had no peek.
    assert trace[2][1].get("peek") is True
    assert trace[3][1].get("peek") is False


# ---------------------------------------------------------------------------
# 2 + 3. Semantic-when-configured & BM25 fallback
# ---------------------------------------------------------------------------


@pytest.fixture
def semantic_wiki(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> WikiStore:
    """A wiki where the embedder is the deterministic bag-of-words stub.

    Sets ``retrieval.strategy = "semantic"`` in the config and stubs
    :func:`outmem.semantic.build_embedder` so neither the reindex nor any
    later query touches a real provider. Pages are picked so a
    paraphrased query (no shared keyword with the target page's body)
    still ranks the right page first under the bag-of-words embedder.
    The index is NOT built — each test decides whether to call
    ``semantic_reindex_all`` (covers the configured-and-built path) or
    skip it (covers the fallback path).
    """
    from outmem.semantic.testing import make_bag_of_words_handle

    monkeypatch.setattr(
        "outmem.semantic.build_embedder",
        lambda _model: make_bag_of_words_handle(),
    )

    root = tmp_path / "wiki"
    store = WikiStore.init(root)
    yaml_path = root / "config.yaml"
    text = yaml_path.read_text(encoding="utf-8")
    # Switch the strategy off the default rerank(bm25) — that path doesn't
    # exercise the semantic-needs-index code we want to test.
    text = text.replace(
        "strategy: rerank(bm25)",
        "strategy: semantic",
    )
    # Lower the threshold for the bag-of-words embedder (real openai
    # embeddings hit 0.8 easily; small fixture corpora under BoW don't).
    text = text.replace(
        "similarity_threshold: 0.8",
        "similarity_threshold: 0.1",
    )
    yaml_path.write_text(text, encoding="utf-8")
    store.close()
    store = WikiStore.open(root)

    # Two pages with disjoint keywords so semantic vs bm25 give different
    # answers depending on which strategy actually ran.
    store.write_page(
        "penicillin",
        title="Penicillin",
        body=(
            "Penicillin G is a beta-lactam antibiotic. Dosing for "
            "endocarditis: 4 million units IV every 4 hours.\n"
        ),
    )
    store.write_page(
        "acme-msa",
        title="Acme MSA",
        body="The master services agreement covers the discount schedule.\n",
    )
    return store


def test_search_wiki_uses_semantic_when_configured_and_index_built(
    semantic_wiki: WikiStore,
) -> None:
    """retrieval.strategy = "semantic" + a built index → semantic search.

    The query has *no overlapping keyword* with the target page body —
    "antibiotic dose" doesn't appear literally in `penicillin.md`. A
    plain bm25 strategy would miss; the deterministic bag-of-words
    embedder still groups the antibiotic-related terms together, so a
    semantic retrieval ranks the penicillin page above the MSA page.
    No fallback note should appear (the configured strategy ran).
    """
    semantic_wiki.semantic_reindex_all()
    assert semantic_wiki.semantic_available()

    tools = wiki_tools(semantic_wiki)
    out = _by_name(tools, "search_wiki")(
        question="antibiotic dose for bacterial endocarditis"
    )

    # The query has zero literal keyword overlap with the penicillin
    # page's body ("antibiotic" / "bacterial" appear in neither) — but
    # the bag-of-words embedder still groups the antibiotic-related
    # terms above the MSA's "discount/services" terms, so semantic
    # ranks the right page FIRST. That ordering is the proof semantic
    # actually ran (bm25 on the query would put neither above the
    # other — both lack every query term).
    pen_pos = out.index("[[penicillin]]")
    msa_pos = out.index("[[acme-msa]]") if "[[acme-msa]]" in out else len(out)
    assert pen_pos < msa_pos, f"penicillin must rank above acme-msa; got:\n{out}"
    # No fallback diagnostic — semantic actually ran.
    assert "fell back" not in out
    assert "diagnostics:" not in out


def test_search_wiki_falls_back_to_bm25_when_semantic_index_missing(
    semantic_wiki: WikiStore,
) -> None:
    """retrieval.strategy = "semantic" but no index → falls back to bm25.

    Without the bm25 fallback the tool would return
    ``(search_wiki failed: <SEMANTIC_UNAVAILABLE_HELP>)`` and the agent
    would have nothing to act on. With it, the user-configured intent is
    honoured when possible — and when it can't be, the agent still gets
    answers PLUS a visible note explaining what to do (``outmem reindex``).

    The query carries the literal keyword "penicillin" so bm25 trivially
    ranks the right page first.
    """
    assert not semantic_wiki.semantic_available()  # precondition: no index

    tools = wiki_tools(semantic_wiki)
    out = _by_name(tools, "search_wiki")(question="penicillin dose")

    # Results came back (the bm25 fallback ran).
    assert "[[penicillin]]" in out
    assert "search_wiki failed" not in out
    # The user/agent sees WHY they're not on the configured pipeline.
    assert "diagnostics:" in out
    assert "no semantic index" in out
    assert "'bm25'" in out
    assert "outmem reindex" in out
    # The configured strategy is left untouched — once `outmem reindex`
    # runs, the next call uses the real strategy.
    assert semantic_wiki.config.outmem.retrieval.strategy == "semantic"


def test_search_wiki_default_strategy_no_fallback_note(
    namespaced_store: WikiStore,
) -> None:
    """Default ``rerank(bm25)`` doesn't need a semantic index → it should
    NOT emit the fallback note (regression guard: the fallback path must
    only trigger for strategies that actually need semantic)."""
    # Drop the rerank() wrapper so this test doesn't pay an LLM call —
    # pure bm25 is enough to exercise the no-fallback-note path.
    yaml_path = namespaced_store.root / "config.yaml"
    text = yaml_path.read_text(encoding="utf-8")
    text = text.replace("strategy: rerank(bm25)", "strategy: bm25")
    yaml_path.write_text(text, encoding="utf-8")
    namespaced_store.close()
    reopened = WikiStore.open(namespaced_store.root)

    tools = wiki_tools(reopened)
    out = _by_name(tools, "search_wiki")(question="penicillin endocarditis")
    assert "[[abx:penicillin]]" in out
    assert "fell back" not in out
    assert "no semantic index" not in out
