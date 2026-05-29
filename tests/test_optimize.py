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
    EvalEvent,
    Question,
    QuestionBank,
    RetrievalConfig,
    build_retriever,
    evaluate,
    generate_bank,
    optimize_retrieval,
)
from outmem.optimize.blocks import (
    HybridRetriever,
    LexicalRetriever,
    RetrievalResult,
    SemanticRetriever,
)
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

    def test_sample_caps_answerable_only(self) -> None:
        class _Counting:
            name = "count"

            def __init__(self) -> None:
                self.n = 0

            def retrieve(self, question: str, *, k: int) -> RetrievalResult:
                self.n += 1
                return RetrievalResult(())

        bank = QuestionBank(
            answerable=[Question(f"q{i}?", ("s",)) for i in range(10)],
            unanswerable=[Question("u?", ())],
        )
        r = _Counting()
        card = evaluate(r, bank, sample=3, max_concurrency=1)
        assert r.n == 3 + 1  # 3 sampled answerable + all (1) unanswerable
        assert card.n_answerable == 3
        assert card.n_unanswerable == 1

    def test_concurrency_matches_sequential(
        self, store: WikiStore, bank: QuestionBank
    ) -> None:
        seq = evaluate(LexicalRetriever(store), bank, k=3, max_concurrency=1)
        par = evaluate(LexicalRetriever(store), bank, k=3, max_concurrency=8)
        assert (seq.score, seq.hit_at_k, seq.abstention) == (
            par.score,
            par.hit_at_k,
            par.abstention,
        )


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

    def test_bad_int_raises_outmemerror(self) -> None:
        # Lenient parser must fail as OutmemError, not a bare ValueError.
        with pytest.raises(OutmemError):
            RetrievalConfig.from_dict({"max_candidates": "abc"})

    def test_lenient_bool_strings(self) -> None:
        # bool("false") is True in Python — the parser must not fall for it.
        assert RetrievalConfig.from_dict({"case_insensitive": "false"}).case_insensitive is False
        assert RetrievalConfig.from_dict({"case_insensitive": "true"}).case_insensitive is True
        assert RetrievalConfig.from_dict({"case_insensitive": False}).case_insensitive is False


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

    def test_generate_bank_raises_on_total_failure(self, store: WikiStore) -> None:
        # A bad API key makes every page's generation raise → swallowed to
        # []. The bank must refuse to come back silently empty.
        def boom(messages: object, info: AgentInfo) -> ModelResponse:
            raise RuntimeError("invalid api key")

        with pytest.raises(OutmemError):
            generate_bank(store, model=FunctionModel(boom), per_page=2)

    def test_generate_bank_skips_unreadable_page(self, store: WikiStore) -> None:
        # One malformed page (no frontmatter) must not abort the whole bank.
        (store.pages_path / "malformed.md").write_text("no frontmatter", encoding="utf-8")
        gb = generate_bank(
            store, model=_questions_model(["q?"]), per_page=1, include_unanswerable=False
        )
        assert gb.answerable  # the good fixture pages still generated
        assert all("malformed" not in s for q in gb.answerable for s in q.gold_slugs)

    def test_first_source_handles_str_and_dict_provenance(self) -> None:
        from outmem.optimize.dataset import _first_source

        # dict-shaped provenance (ingested source) → the path, not a stringified dict
        assert (
            _first_source(SimpleNamespace(provenance=[{"path": "sources/x/doc.md"}]))
            == "sources/x/doc.md"
        )
        assert _first_source(SimpleNamespace(provenance=["raw/deck.md"])) == "raw/deck.md"
        assert _first_source(SimpleNamespace(provenance=[])) is None

    def test_generate_bank_reports_progress(self, store: WikiStore) -> None:
        calls: list[tuple[int, int]] = []
        generate_bank(
            store,
            model=_questions_model(["q?"]),
            per_page=1,
            include_unanswerable=False,
            on_progress=lambda done, total: calls.append((done, total)),
        )
        n = len(store.list_slugs())
        assert len(calls) == n                              # one tick per page
        assert [done for done, _ in calls] == list(range(1, n + 1))  # monotonic
        assert calls[-1] == (n, n)                          # ends at total

    def test_generate_bank_max_concurrency_one(self, store: WikiStore) -> None:
        # Serialised generation must produce the same count as parallel.
        gb = generate_bank(
            store,
            model=_questions_model(["a?", "b?"]),
            per_page=2,
            include_unanswerable=False,
            max_concurrency=1,
        )
        assert len(gb.answerable) == 2 * len(store.list_slugs())


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

    def test_enabled_but_empty_index_fails_loud(
        self, store: WikiStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Enabled but never reindexed → a clear "run outmem reindex" error,
        # not a silent empty result that looks like a useless retriever.
        monkeypatch.setattr(store, "semantic_enabled", lambda: True)
        monkeypatch.setattr(store, "semantic_index_is_empty", lambda: True)
        with pytest.raises(OutmemError, match="reindex"):
            SemanticRetriever(store).retrieve("anything", k=3)

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
        monkeypatch.setattr(store, "semantic_index_is_empty", lambda: False)
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
        monkeypatch.setattr(store, "semantic_index_is_empty", lambda: False)
        monkeypatch.setattr(store, "semantic_find_similar", fake_find)

        assert len(SemanticRetriever(store).retrieve("x", k=2).slugs) == 2


# --- hybrid block (RRF of lexical + semantic) ------------------------------


class TestHybridBlock:
    def test_fuses_both_signals(
        self, store: WikiStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        prefix = f"{store.config.wiki_dir}/pages/"

        # Semantic surfaces ceftriaxone first, then penicillin — a different
        # order than lexical (which keys on the word "penicillin").
        def fake_find(text: str, *, top_k: int = 0, **_: Any) -> list[Any]:
            return [
                SimpleNamespace(rel_path=f"{prefix}abx/ceftriaxone.md", chunk_index=0,
                                similarity=0.9, content="…"),
                SimpleNamespace(rel_path=f"{prefix}abx/penicillin.md", chunk_index=0,
                                similarity=0.7, content="…"),
            ]

        monkeypatch.setattr(store, "semantic_enabled", lambda: True)
        monkeypatch.setattr(store, "semantic_index_is_empty", lambda: False)
        monkeypatch.setattr(store, "semantic_find_similar", fake_find)

        fused = HybridRetriever(store).retrieve("penicillin", k=5).slugs
        # penicillin appears in BOTH lists → fuses to the top.
        assert fused[0] == "abx:penicillin"
        # ceftriaxone (semantic-only) is still pulled in.
        assert "abx:ceftriaxone" in fused

    def test_raises_when_semantic_off(self, store: WikiStore) -> None:
        # Fresh wiki: semantic disabled. Hybrid must RAISE (not silently run
        # lexical-only under a "hybrid" label), so the optimizer skips it.
        with pytest.raises(OutmemError):
            HybridRetriever(store).retrieve("penicillin endocarditis", k=3)

    def test_build_retriever_hybrid(self, store: WikiStore) -> None:
        r = build_retriever(store, RetrievalConfig(strategy="hybrid", rrf_k=30))
        assert r.name == "hybrid"


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


def test_optimize_survives_bad_strategy_from_agent(
    store: WikiStore, bank: QuestionBank
) -> None:
    """An agent proposing an out-of-enum strategy must be told "unavailable",
    not crash the whole run (regression: from_dict was outside the try)."""
    state = {"n": 0}

    def optimizer(messages: object, info: AgentInfo) -> ModelResponse:
        state["n"] += 1
        if state["n"] == 1:
            return ModelResponse(
                parts=[ToolCallPart(tool_name="run_eval", args={"strategy": "bm25"})]
            )
        return ModelResponse(parts=[TextPart("done")])

    result = optimize_retrieval(
        store, bank, optimizer_model=FunctionModel(optimizer), k=3, max_evals=5
    )
    # The bad config was rejected (not recorded), and we still return a
    # real scored baseline rather than crashing.
    assert result.trace == []
    assert result.best_config.strategy == "lexical"
    # …and the unavailable config is captured on result.log.
    assert any("bm25" in line for line in result.log)


def _exploding_model() -> FunctionModel:
    def respond(messages: object, info: AgentInfo) -> ModelResponse:
        raise RuntimeError("model is down")

    return FunctionModel(respond)


def test_evaluate_aggregates_retriever_notes(store: WikiStore) -> None:
    class _Noting:
        name = "noting"

        def retrieve(self, question: str, *, k: int) -> RetrievalResult:
            return RetrievalResult((), note="rerank fell back: refusal")

    bank = QuestionBank(answerable=[Question(f"q{i}?", ("s",)) for i in range(4)])
    card = evaluate(_Noting(), bank, max_concurrency=1)
    assert card.notes == ("rerank fell back: refusal (x4)",)  # deduped + counted


def test_optimize_log_records_rerank_fallback(
    store: WikiStore, bank: QuestionBank
) -> None:
    state = {"n": 0}

    def optimizer(messages: object, info: AgentInfo) -> ModelResponse:
        state["n"] += 1
        if state["n"] == 1:
            return ModelResponse(
                parts=[ToolCallPart(tool_name="run_eval", args={"strategy": "rerank"})]
            )
        return ModelResponse(parts=[TextPart("done")])

    result = optimize_retrieval(
        store,
        bank,
        optimizer_model=FunctionModel(optimizer),
        rerank_model=_exploding_model(),  # every per-question rerank call fails
        k=3,
        max_evals=3,
    )
    assert any("rerank" in line and "fell back" in line for line in result.log)


def test_optimize_reports_epochs(store: WikiStore, bank: QuestionBank) -> None:
    """on_eval fires once per scored eval — an epoch with index/max_evals,
    the config tried, its metrics, and best-so-far."""
    events: list[EvalEvent] = []
    state = {"n": 0}

    def optimizer(messages: object, info: AgentInfo) -> ModelResponse:
        state["n"] += 1
        if state["n"] <= 2:  # two scored evals, then finish
            return ModelResponse(
                parts=[ToolCallPart(tool_name="run_eval", args={"strategy": "lexical"})]
            )
        return ModelResponse(parts=[TextPart("done")])

    optimize_retrieval(
        store,
        bank,
        optimizer_model=FunctionModel(optimizer),
        k=3,
        max_evals=5,
        on_eval=events.append,
    )
    assert [e.index for e in events] == [1, 2]          # one epoch per scored eval
    assert all(e.max_evals == 5 for e in events)        # the turn budget is carried
    assert events[1].best_score >= events[0].best_score  # best is non-decreasing
    assert events[-1].config.strategy == "lexical"
    assert 0.0 <= events[-1].scorecard.score <= 1.0


def test_optimize_eval_sample_rescores_winner_on_full_bank(
    store: WikiStore, bank: QuestionBank
) -> None:
    """With eval_sample, configs are tuned on a subset but the winner is
    re-scored on the FULL bank, so the reported scorecard covers all of it."""
    state = {"n": 0}

    def optimizer(messages: object, info: AgentInfo) -> ModelResponse:
        state["n"] += 1
        if state["n"] == 1:
            return ModelResponse(
                parts=[ToolCallPart(tool_name="run_eval", args={"strategy": "lexical"})]
            )
        return ModelResponse(parts=[TextPart("done")])

    result = optimize_retrieval(
        store,
        bank,
        optimizer_model=FunctionModel(optimizer),
        k=3,
        max_evals=5,
        eval_sample=1,  # tune on 1 answerable question…
    )
    # …but the returned scorecard reflects the whole bank (re-scored).
    assert result.scorecard.n_answerable == len(bank.answerable)


def test_generate_bank_invokes_logfire_setup(
    store: WikiStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    import outmem._logfire as lf

    seen: list[object] = []
    monkeypatch.setattr(lf, "setup", lambda s: bool(seen.append(s)))
    generate_bank(
        store, model=_questions_model(["q?"]), per_page=1, include_unanswerable=False
    )
    assert len(seen) == 1 and seen[0] is store.config.outmem.logfire


def test_optimize_invokes_logfire_setup(
    store: WikiStore, bank: QuestionBank, monkeypatch: pytest.MonkeyPatch
) -> None:
    import outmem._logfire as lf

    seen: list[object] = []
    monkeypatch.setattr(lf, "setup", lambda s: bool(seen.append(s)))

    def opt(messages: object, info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart("done")])

    optimize_retrieval(store, bank, optimizer_model=FunctionModel(opt), k=3, max_evals=2)
    assert len(seen) == 1 and seen[0] is store.config.outmem.logfire
