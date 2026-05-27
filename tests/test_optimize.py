"""Tests for ``outmem.optimize`` — retrieval lego blocks, the benchmark,
test-data generation, and the agent-driven config optimizer.

LLM paths are scripted with ``pydantic_ai.models.function.FunctionModel``
(no real model). The semantic block is tested by stubbing the store's
``semantic_find_similar`` so we exercise *our* chunk→slug wiring without
needing an embedding model or a built index.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from outmem.exceptions import OutmemError
from outmem.optimize import (
    Question,
    QuestionBank,
    RetrievalConfig,
    build_retriever,
    evaluate,
    generate_bank,
    optimize_retrieval,
)
from outmem.optimize.blocks import LexicalRetriever, SemanticRetriever
from outmem.store import WikiStore


@pytest.fixture
def store(tmp_path: Path) -> WikiStore:
    s = WikiStore.init(tmp_path / "w")
    s.write_page(
        "abx:penicillin",
        title="Penicillin",
        body="IV penicillin G 18-24 MU/day in divided doses for infective endocarditis.",
    )
    s.write_page(
        "abx:ceftriaxone",
        title="Ceftriaxone",
        body="Ceftriaxone 2g IV q24h; a once-daily cephalosporin alternative.",
    )
    s.write_page(
        "ops:pricing",
        title="Pricing",
        body="Internal cost-plus 35% margin applied to antibiotic sales.",
    )
    return s


@pytest.fixture
def bank() -> QuestionBank:
    return QuestionBank(
        answerable=[
            Question("IV penicillin G dose for endocarditis", ("abx:penicillin",)),
            Question("once-daily cephalosporin option", ("abx:ceftriaxone",)),
        ],
        unanswerable=[Question("What is the capital of France?", ())],
    )


def _questions_model(questions: list[str]) -> FunctionModel:
    def respond(messages: object, info: AgentInfo) -> ModelResponse:
        name = info.output_tools[0].name
        return ModelResponse(
            parts=[ToolCallPart(tool_name=name, args={"questions": questions})]
        )

    return FunctionModel(respond)


def _rerank_model(selections: list[dict[str, str]]) -> FunctionModel:
    def respond(messages: object, info: AgentInfo) -> ModelResponse:
        name = info.output_tools[0].name
        return ModelResponse(
            parts=[ToolCallPart(tool_name=name, args={"relevant": selections})]
        )

    return FunctionModel(respond)


# --- blocks + metric -------------------------------------------------------


class TestLexicalAndMetric:
    def test_hit_and_abstain(self, store: WikiStore, bank: QuestionBank) -> None:
        card = evaluate(LexicalRetriever(store), bank, k=3)
        assert 0.0 <= card.score <= 1.0
        assert card.n_answerable == 2
        assert card.n_unanswerable == 1
        # An off-topic query yields no keyword hits → correct abstention.
        assert card.abstention == 1.0

    def test_empty_query_returns_nothing(self, store: WikiStore) -> None:
        # A question of only stopwords formulates an empty pattern.
        assert LexicalRetriever(store).retrieve("what is the?", k=5).slugs == ()

    def test_failures_exposed(self, store: WikiStore) -> None:
        bad = QuestionBank(
            answerable=[Question("zzz nonexistent topic", ("abx:penicillin",))]
        )
        card = evaluate(LexicalRetriever(store), bad, k=3)
        assert card.hit_at_k == 0.0
        assert len(card.failures) == 1
        assert card.failures[0].gold_slugs == ("abx:penicillin",)


class TestRetrievalConfig:
    def test_round_trip(self) -> None:
        cfg = RetrievalConfig(strategy="rerank", max_candidates=42)
        assert RetrievalConfig.from_dict(cfg.to_dict()) == cfg

    def test_lenient_parse_and_defaults(self) -> None:
        cfg = RetrievalConfig.from_dict({"strategy": "LEXICAL", "unknown": 1})
        assert cfg.strategy == "lexical"
        assert cfg.case_insensitive is True  # default preserved

    def test_bad_strategy_raises(self) -> None:
        with pytest.raises(OutmemError):
            RetrievalConfig.from_dict({"strategy": "bm25-typo"})


# --- dataset ---------------------------------------------------------------


class TestDataset:
    def test_json_round_trip(self, tmp_path: Path, bank: QuestionBank) -> None:
        p = tmp_path / "bank.json"
        bank.save(p)
        loaded = QuestionBank.load(p)
        assert len(loaded.answerable) == 2
        assert len(loaded.unanswerable) == 1
        assert loaded.answerable[0].gold_slugs == ("abx:penicillin",)

    def test_generate_bank(self, store: WikiStore) -> None:
        model = _questions_model(["How is it dosed?", "What is it for?"])
        gb = generate_bank(
            store, model=model, per_page=2, max_pages=2, include_unanswerable=False
        )
        assert len(gb.answerable) == 4  # 2 pages x 2 questions
        assert all(len(q.gold_slugs) == 1 for q in gb.answerable)


# --- rerank block (relevance filter as a retriever) ------------------------


def test_rerank_block_returns_kept_slugs(store: WikiStore) -> None:
    model = _rerank_model([{"slug": "abx:penicillin", "reason": "dosing"}])
    retriever = build_retriever(
        store, RetrievalConfig(strategy="rerank"), model=model
    )
    assert retriever.retrieve("penicillin dose", k=3).slugs == ("abx:penicillin",)


# --- semantic block (wiring tested with a stubbed index) -------------------


class TestSemanticBlock:
    def test_disabled_raises(self, store: WikiStore) -> None:
        # A fresh wiki has semantic disabled → the block must raise (so the
        # optimizer marks the config unavailable rather than crashing).
        retriever = build_retriever(store, RetrievalConfig(strategy="semantic"))
        with pytest.raises(OutmemError):
            retriever.retrieve("anything", k=3)

    def test_chunk_to_slug_mapping(
        self, store: WikiStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        prefix = f"{store.config.wiki_dir}/pages/"

        def fake_find(text: str, *, top_k: int = 0, **_: Any) -> list[Any]:
            return [
                SimpleNamespace(rel_path=f"{prefix}abx/penicillin.md", chunk_index=0,
                                similarity=0.91, content="…"),
                SimpleNamespace(rel_path=f"{store.config.wiki_dir}/sources/x/deck.md",
                                chunk_index=0, similarity=0.88, content="…"),  # source → skip
                SimpleNamespace(rel_path=f"{prefix}abx/penicillin.md", chunk_index=2,
                                similarity=0.80, content="…"),  # dup page → dedup
                SimpleNamespace(rel_path=f"{prefix}abx/ceftriaxone.md", chunk_index=0,
                                similarity=0.75, content="…"),
            ]

        monkeypatch.setattr(store, "semantic_enabled", lambda: True)
        monkeypatch.setattr(store, "semantic_find_similar", fake_find)

        result = SemanticRetriever(store, top_k=8).retrieve("penicillin", k=5)
        # Source chunk filtered out, page dedup preserves best-first order.
        assert result.slugs == ("abx:penicillin", "abx:ceftriaxone")

    def test_respects_k(
        self, store: WikiStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        prefix = f"{store.config.wiki_dir}/pages/"

        def fake_find(text: str, *, top_k: int = 0, **_: Any) -> list[Any]:
            return [
                SimpleNamespace(rel_path=f"{prefix}abx/penicillin.md", chunk_index=0,
                                similarity=0.9, content="…"),
                SimpleNamespace(rel_path=f"{prefix}abx/ceftriaxone.md", chunk_index=0,
                                similarity=0.8, content="…"),
                SimpleNamespace(rel_path=f"{prefix}ops/pricing.md", chunk_index=0,
                                similarity=0.7, content="…"),
            ]

        monkeypatch.setattr(store, "semantic_enabled", lambda: True)
        monkeypatch.setattr(store, "semantic_find_similar", fake_find)

        assert len(SemanticRetriever(store).retrieve("x", k=2).slugs) == 2


# --- the agent-driven optimizer --------------------------------------------


def test_optimize_returns_best_seen(store: WikiStore, bank: QuestionBank) -> None:
    """The FunctionModel agent evaluates lexical, peeks at a page, then
    finishes. The result is the best-SCORING config it measured, not its
    closing words."""
    state = {"n": 0}

    def optimizer(messages: object, info: AgentInfo) -> ModelResponse:
        state["n"] += 1
        if state["n"] == 1:
            return ModelResponse(
                parts=[ToolCallPart(tool_name="run_eval", args={"strategy": "lexical"})]
            )
        if state["n"] == 2:
            return ModelResponse(
                parts=[ToolCallPart(tool_name="read_page", args={"slug": "abx:penicillin"})]
            )
        return ModelResponse(parts=[TextPart("lexical baseline was best in budget")])

    result = optimize_retrieval(
        store, bank, optimizer_model=FunctionModel(optimizer), k=3, max_evals=5
    )
    assert result.best_config.strategy == "lexical"
    assert len(result.trace) == 1
    # best_score must equal a direct evaluation of the same config.
    direct = evaluate(build_retriever(store, result.best_config), bank, k=3)
    assert result.best_score == direct.score
    assert "best" in result.notes.lower()


def test_optimize_falls_back_when_agent_never_evals(
    store: WikiStore, bank: QuestionBank
) -> None:
    """If the agent finishes without a single scorable config, we still
    return a real scored baseline rather than nothing."""

    def lazy(messages: object, info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart("I did nothing useful")])

    result = optimize_retrieval(
        store, bank, optimizer_model=FunctionModel(lazy), k=3, max_evals=5
    )
    assert result.trace == []
    assert result.best_config.strategy == "lexical"  # the default baseline
    assert 0.0 <= result.best_score <= 1.0
