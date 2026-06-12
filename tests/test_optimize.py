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
    BM25Retriever,
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

    def test_latency_recorded(self, store: WikiStore, bank: QuestionBank) -> None:
        card = evaluate(LexicalRetriever(store), bank, k=3, max_concurrency=1)
        # Per-search and aggregate latency are populated and non-negative.
        assert all(r.latency_ms >= 0.0 for r in card.results)
        assert card.mean_latency_ms >= 0.0
        assert card.p95_latency_ms >= card.mean_latency_ms or len(card.results) <= 1

    def test_latency_stats_helper(self) -> None:
        from outmem.optimize.bench import _latency_stats

        assert _latency_stats([]) == (0.0, 0.0)
        mean, p95 = _latency_stats([10.0, 20.0, 30.0, 40.0])
        assert mean == 25.0
        assert p95 == 40.0  # ceil(0.95*4)=4 → rank 4 → the max
        # n=1 / n=2 must not IndexError, and p95 is deterministic (no
        # banker's-rounding drift): ceil(0.95*20)=19 → rank 19 → 190.
        assert _latency_stats([5.0])[1] == 5.0
        assert _latency_stats([float(i) for i in range(10, 201, 10)])[1] == 190.0


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

    def test_rerank_source_rejects_rerank(self) -> None:
        # No recursive rerank-over-rerank — the source must be atomic.
        with pytest.raises(OutmemError):
            RetrievalConfig.from_dict({"rerank_source": "rerank"})

    def test_rerank_source_accepts_atomic(self) -> None:
        cfg = RetrievalConfig.from_dict({"rerank_source": "SEMANTIC"})  # case-fold
        assert cfg.rerank_source == "semantic"


class TestBM25Block:
    def test_ranks_by_relevance(self, store: WikiStore) -> None:
        # "penicillin" appears in the penicillin page (and is mentioned on
        # ceftriaxone's) → the penicillin page should rank first.
        out = BM25Retriever(store).retrieve("penicillin endocarditis", k=3)
        assert out.slugs  # non-empty
        assert out.slugs[0] == "abx:penicillin"

    def test_respects_k(self, store: WikiStore) -> None:
        out = BM25Retriever(store).retrieve("antibiotic", k=1)
        assert len(out.slugs) <= 1

    def test_empty_query_returns_nothing(self, store: WikiStore) -> None:
        # Only stopwords → no terms → empty (abstain), no crash.
        assert BM25Retriever(store).retrieve("what is the?", k=5).slugs == ()

    def test_no_match_abstains(self, store: WikiStore) -> None:
        assert BM25Retriever(store).retrieve("zzzznonexistent", k=5).slugs == ()

    def test_build_retriever_bm25(self, store: WikiStore) -> None:
        r = build_retriever(store, RetrievalConfig(strategy="bm25"))
        assert r.name == "bm25"

    def test_concurrent_scoring_matches_sequential(self, store: WikiStore) -> None:
        # Regression: bm25 shares no sqlite connection across threads. A bank
        # big enough to spin the 8-worker pool must score the SAME as mc=1
        # (the bug returned 0.0 under concurrency via a cross-thread error).
        bank = QuestionBank(
            answerable=[Question(f"penicillin endocarditis {i}", ("abx:penicillin",))
                        for i in range(12)]
        )
        r = build_retriever(store, RetrievalConfig(strategy="bm25"))
        seq = evaluate(r, bank, k=3, max_concurrency=1)
        par = evaluate(build_retriever(store, RetrievalConfig(strategy="bm25")),
                       bank, k=3, max_concurrency=8)
        assert seq.hit_at_k == par.hit_at_k
        assert par.hit_at_k > 0.0  # actually retrieved under concurrency

    def test_bm25_in_strategies(self) -> None:
        # The optimizer agent can select it.
        assert RetrievalConfig.from_dict({"strategy": "bm25"}).strategy == "bm25"


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


def test_rerank_block_uses_configured_source(
    store: WikiStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """rerank_source="semantic" routes candidates through SemanticRetriever
    rather than the keyword net, so the LLM judge sees pages the lexical
    net would never have surfaced."""
    prefix = f"{store.config.wiki_dir}/pages/"

    def fake_find(text: str, *, top_k: int = 0, **_: Any) -> list[Any]:
        return [
            SimpleNamespace(rel_path=f"{prefix}abx/ceftriaxone.md", chunk_index=0,
                            similarity=0.92, content="…"),
            SimpleNamespace(rel_path=f"{prefix}abx/penicillin.md", chunk_index=0,
                            similarity=0.81, content="…"),
        ]

    monkeypatch.setattr(store, "semantic_enabled", lambda: True)
    monkeypatch.setattr(store, "semantic_index_is_empty", lambda: False)
    monkeypatch.setattr(store, "semantic_find_similar", fake_find)

    model = _rerank_model([{"slug": "abx:penicillin", "reason": "exact match"}])
    retriever = build_retriever(
        store,
        RetrievalConfig(strategy="rerank", rerank_source="semantic"),
        model=model,
    )
    # The query has NO keyword overlap with either page — lexical-source
    # rerank would return empty, but semantic-source surfaces both pages
    # and the LLM judge keeps the relevant one.
    assert retriever.retrieve("how do I treat a bug bite?", k=3).slugs == (
        "abx:penicillin",
    )


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

        # Default hybrid fuses lexical + semantic.
        r = build_retriever(store, RetrievalConfig(strategy="hybrid"))
        fused = r.retrieve("penicillin", k=5).slugs
        # penicillin appears in BOTH lists → fuses to the top.
        assert fused[0] == "abx:penicillin"
        # ceftriaxone (semantic-only) is still pulled in.
        assert "abx:ceftriaxone" in fused

    def test_fuse_bm25_semantic(
        self, store: WikiStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        prefix = f"{store.config.wiki_dir}/pages/"
        monkeypatch.setattr(store, "semantic_enabled", lambda: True)
        monkeypatch.setattr(store, "semantic_index_is_empty", lambda: False)
        monkeypatch.setattr(
            store, "semantic_find_similar",
            lambda text, *, top_k=0, **_: [
                SimpleNamespace(rel_path=f"{prefix}abx/penicillin.md", chunk_index=0,
                                similarity=0.9, content="x")
            ],
        )
        r = build_retriever(store, RetrievalConfig(strategy="hybrid", fuse=("bm25", "semantic")))
        assert "abx:penicillin" in r.retrieve("penicillin endocarditis", k=5).slugs

    def test_raises_when_semantic_off(self, store: WikiStore) -> None:
        # Fresh wiki: semantic disabled. A hybrid with a semantic leg must
        # RAISE (not silently fuse what's available), so the optimizer skips it.
        r = build_retriever(store, RetrievalConfig(strategy="hybrid"))
        with pytest.raises(OutmemError):
            r.retrieve("penicillin endocarditis", k=3)

    def test_build_retriever_hybrid(self, store: WikiStore) -> None:
        r = build_retriever(store, RetrievalConfig(strategy="hybrid", rrf_k=30))
        assert r.name == "hybrid"

    def test_fuse_validation(self) -> None:
        # A fuse leg must be an atomic strategy (not nested hybrid), 2+ legs.
        with pytest.raises(OutmemError):
            RetrievalConfig.from_dict({"fuse": ["lexical", "hybrid"]})
        with pytest.raises(OutmemError):
            RetrievalConfig.from_dict({"fuse": ["lexical"]})

    def test_build_retriever_guards_direct_short_fuse(self, store: WikiStore) -> None:
        # A directly-constructed (from_dict-bypassing) zero/one-leg hybrid
        # must still be rejected by build_retriever, not silently abstain.
        with pytest.raises(OutmemError):
            build_retriever(store, RetrievalConfig(strategy="hybrid", fuse=()))
        with pytest.raises(OutmemError):
            build_retriever(store, RetrievalConfig(strategy="hybrid", fuse=("lexical",)))


class TestHydeBlock:
    def _store_with_fake_semantic(
        self, store: WikiStore, monkeypatch: pytest.MonkeyPatch
    ) -> str:
        prefix = f"{store.config.wiki_dir}/pages/"
        monkeypatch.setattr(store, "semantic_enabled", lambda: True)
        monkeypatch.setattr(store, "semantic_index_is_empty", lambda: False)
        monkeypatch.setattr(
            store, "semantic_find_similar",
            lambda text, *, top_k=0, **_: [
                SimpleNamespace(rel_path=f"{prefix}abx/penicillin.md", chunk_index=0,
                                similarity=0.9, content="x")
            ],
        )
        return prefix

    def test_generates_then_searches(
        self, store: WikiStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._store_with_fake_semantic(store, monkeypatch)

        def hyde_model(messages: object, info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart("Penicillin G is dosed IV for endocarditis.")])

        r = build_retriever(
            store, RetrievalConfig(strategy="hyde"), model=FunctionModel(hyde_model)
        )
        out = r.retrieve("how much penicillin?", k=3)
        assert out.slugs == ("abx:penicillin",)
        assert out.note is None  # generation succeeded

    def test_generation_failure_falls_back(
        self, store: WikiStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._store_with_fake_semantic(store, monkeypatch)

        def boom(messages: object, info: AgentInfo) -> ModelResponse:
            raise RuntimeError("model down")

        r = build_retriever(
            store, RetrievalConfig(strategy="hyde"), model=FunctionModel(boom)
        )
        out = r.retrieve("penicillin", k=3)
        assert out.slugs == ("abx:penicillin",)  # fell back to raw question
        assert out.note and "failed" in out.note

    def test_raises_when_semantic_off(self, store: WikiStore) -> None:
        def hyde_model(messages: object, info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart("x")])

        r = build_retriever(
            store, RetrievalConfig(strategy="hyde"), model=FunctionModel(hyde_model)
        )
        with pytest.raises(OutmemError):  # no index
            r.retrieve("penicillin", k=3)


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


def test_optimize_run_eval_accepts_hyde_model_id(
    store: WikiStore, bank: QuestionBank
) -> None:
    """The agent can set the hyde generation model via run_eval — the
    param must exist on the tool and flow into the config (regression: it
    was missing, so hyde_model was unreachable from the agent surface)."""
    seen: list[dict[str, Any]] = []

    def optimizer(messages: object, info: AgentInfo) -> ModelResponse:
        if not seen:
            # lexical so no model is actually needed; we only assert the
            # tool accepts hyde_model_id without error.
            seen.append({"called": True})
            return ModelResponse(
                parts=[ToolCallPart(
                    tool_name="run_eval",
                    args={"strategy": "lexical", "hyde_model_id": "anthropic:claude-haiku-4-5"},
                )]
            )
        return ModelResponse(parts=[TextPart("done")])

    result = optimize_retrieval(
        store, bank, optimizer_model=FunctionModel(optimizer), k=3, max_evals=5
    )
    # The eval ran (hyde_model_id was accepted, not rejected as unknown kwarg).
    assert len(result.trace) == 1


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
                parts=[ToolCallPart(tool_name="run_eval", args={"strategy": "tfidf"})]
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
    assert any("tfidf" in line for line in result.log)


def test_allowed_strategies_bounces_disabled_config(
    store: WikiStore, bank: QuestionBank
) -> None:
    """A run restricted to {bm25} must reject a semantic config without
    burning an eval, while still scoring an allowed one."""
    state = {"n": 0}

    def optimizer(messages: object, info: AgentInfo) -> ModelResponse:
        state["n"] += 1
        if state["n"] == 1:  # disallowed → bounced, no eval consumed
            return ModelResponse(parts=[ToolCallPart(
                tool_name="run_eval", args={"strategy": "semantic"})])
        if state["n"] == 2:  # allowed → scored
            return ModelResponse(parts=[ToolCallPart(
                tool_name="run_eval", args={"strategy": "bm25"})])
        return ModelResponse(parts=[TextPart("done")])

    result = optimize_retrieval(
        store, bank, optimizer_model=FunctionModel(optimizer),
        k=1, max_evals=5, allowed_strategies=["bm25"],
    )
    # Only the bm25 eval was recorded; semantic was bounced.
    assert [cfg["strategy"] for cfg, _ in result.trace] == ["bm25"]


def test_allowed_strategies_rejects_unknown_name(
    store: WikiStore, bank: QuestionBank
) -> None:
    def lazy(messages: object, info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart("done")])

    with pytest.raises(OutmemError, match="unknown"):
        optimize_retrieval(
            store, bank, optimizer_model=FunctionModel(lazy),
            allowed_strategies=["bm25", "tfidf"],  # tfidf is not a strategy
        )


def test_allowed_strategies_gates_rerank_source(
    store: WikiStore, bank: QuestionBank
) -> None:
    """Regression for 'why does it try lexical?': allowed=['rerank','bm25']
    must BOUNCE rerank(lexical) (lexical not allowed) but ACCEPT
    rerank(bm25). (bm25 source needs no semantic index.)"""
    state = {"n": 0}

    def optimizer(messages: object, info: AgentInfo) -> ModelResponse:
        state["n"] += 1
        if state["n"] == 1:  # rerank over lexical → bounced (lexical not allowed)
            return ModelResponse(parts=[ToolCallPart(tool_name="run_eval",
                args={"strategy": "rerank", "rerank_source": "lexical"})])
        if state["n"] == 2:  # rerank over bm25 → allowed
            return ModelResponse(parts=[ToolCallPart(tool_name="run_eval",
                args={"strategy": "rerank", "rerank_source": "bm25"})])
        return ModelResponse(parts=[TextPart("done")])

    result = optimize_retrieval(
        store, bank, optimizer_model=FunctionModel(optimizer),
        k=1, max_evals=5, rerank_model=_rerank_model([]),
        allowed_strategies=["rerank", "bm25"],
    )
    sources = [cfg["rerank_source"] for cfg, _ in result.trace]
    assert sources == ["bm25"]  # lexical bounced, bm25 scored


def test_disallowed_blocks() -> None:
    from outmem.optimize.optimizer import _disallowed_blocks

    allowed = frozenset({"rerank", "semantic"})
    rerank_lex = RetrievalConfig(strategy="rerank", rerank_source="lexical")
    rerank_sem = RetrievalConfig(strategy="rerank", rerank_source="semantic")
    assert _disallowed_blocks(rerank_lex, allowed) == {"lexical"}
    assert _disallowed_blocks(rerank_sem, allowed) == set()
    # hybrid legs are gated too
    hyb = RetrievalConfig(strategy="hybrid", fuse=("bm25", "semantic"))
    assert _disallowed_blocks(hyb, frozenset({"hybrid", "semantic"})) == {"bm25"}


def test_normalise_allowed_strategies() -> None:
    from outmem.optimize.optimizer import _normalise_allowed_strategies

    assert _normalise_allowed_strategies(None) is None
    assert _normalise_allowed_strategies(["BM25", " Semantic "]) == {"bm25", "semantic"}
    with pytest.raises(OutmemError):
        _normalise_allowed_strategies([])  # empty → nothing to evaluate


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
        # Two distinct configs so neither is a dedupe no-op — the harness's
        # cache short-circuits identical evals, which would (correctly) drop
        # the second on_eval below if we sent the same config twice.
        if state["n"] == 1:
            return ModelResponse(parts=[ToolCallPart(
                tool_name="run_eval",
                args={"strategy": "lexical", "case_insensitive": True},
            )])
        if state["n"] == 2:
            return ModelResponse(parts=[ToolCallPart(
                tool_name="run_eval",
                args={"strategy": "lexical", "case_insensitive": False},
            )])
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


def test_optimize_dedupes_repeat_configs(store: WikiStore, bank: QuestionBank) -> None:
    """run_eval called with an already-tried config returns the cached
    scorecard, does NOT consume an eval slot, and does NOT fire on_eval —
    so the agent can't burn its 12-turn budget asking for the same thing."""
    events: list[EvalEvent] = []
    state = {"n": 0}

    def optimizer(messages: object, info: AgentInfo) -> ModelResponse:
        state["n"] += 1
        # Same config, twice in a row — second call should be a no-op.
        if state["n"] <= 2:
            return ModelResponse(parts=[ToolCallPart(
                tool_name="run_eval",
                args={"strategy": "lexical", "case_insensitive": True},
            )])
        return ModelResponse(parts=[TextPart("done")])

    optimize_retrieval(
        store,
        bank,
        optimizer_model=FunctionModel(optimizer),
        k=3,
        max_evals=5,
        on_eval=events.append,
    )
    assert [e.index for e in events] == [1]  # only the first call scored


# --- strategy DSL ----------------------------------------------------------


class TestStrategyDsl:
    def test_atomic(self) -> None:
        from outmem.optimize.dsl import parse_strategy
        for name in ("lexical", "bm25", "semantic", "hyde"):
            assert parse_strategy(name) == {"strategy": name}

    def test_rerank_default_source(self) -> None:
        from outmem.optimize.dsl import parse_strategy
        assert parse_strategy("rerank") == {
            "strategy": "rerank", "rerank_source": "lexical",
        }

    def test_rerank_with_source(self) -> None:
        from outmem.optimize.dsl import parse_strategy
        assert parse_strategy("rerank(semantic)") == {
            "strategy": "rerank", "rerank_source": "semantic",
        }

    def test_hybrid_two_legs(self) -> None:
        from outmem.optimize.dsl import parse_strategy
        assert parse_strategy("bm25+semantic") == {
            "strategy": "hybrid", "fuse": ["bm25", "semantic"],
        }

    def test_hybrid_three_legs(self) -> None:
        from outmem.optimize.dsl import parse_strategy
        assert parse_strategy("lexical+bm25+semantic") == {
            "strategy": "hybrid", "fuse": ["lexical", "bm25", "semantic"],
        }

    def test_case_and_whitespace_tolerant(self) -> None:
        from outmem.optimize.dsl import parse_strategy
        assert parse_strategy("  BM25  ") == {"strategy": "bm25"}
        assert parse_strategy("BM25+Semantic") == {
            "strategy": "hybrid", "fuse": ["bm25", "semantic"],
        }

    @pytest.mark.parametrize("bad", [
        "foo",                # not in vocabulary
        "bm25+rerank",        # rerank is not a fuse leg
        "rerank(bogus)",      # bogus source
        "rerank(rerank)",     # no recursion
        "",                   # empty
        "+",                  # legs both empty
        "bm25+bm25",          # duplicate legs
        "bm25+",              # trailing leg empty
    ])
    def test_rejects_garbage(self, bad: str) -> None:
        from outmem.optimize.dsl import parse_strategy
        with pytest.raises(OutmemError):
            parse_strategy(bad)

    def test_format_roundtrip(self) -> None:
        from outmem.optimize.dsl import format_strategy, parse_strategy
        for spec in (
            "bm25", "semantic", "rerank(semantic)", "bm25+semantic",
            "lexical+bm25+semantic",
        ):
            parsed = parse_strategy(spec)
            base = RetrievalConfig().to_dict()
            assert format_strategy({**base, **parsed}) == spec


# --- summary table + pick + save ------------------------------------------


def test_optimize_result_summary_table_and_save(
    store: WikiStore, bank: QuestionBank, tmp_path: Path
) -> None:
    """End-to-end of the post-run UX: the result carries one EvalRow per
    scored eval, summary_table renders them, pick(rank) returns the
    config at that rank, save(rank, store) writes config.yaml's retrieval
    block that a subsequent load_yaml_config picks up."""
    from outmem.config import CONFIG_FILENAME, load_yaml_config

    state = {"n": 0}

    def optimizer(messages: object, info: AgentInfo) -> ModelResponse:
        state["n"] += 1
        if state["n"] == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name="run_eval",
                args={"strategy": "lexical"})])
        if state["n"] == 2:
            return ModelResponse(parts=[ToolCallPart(tool_name="run_eval",
                args={"strategy": "bm25"})])
        return ModelResponse(parts=[TextPart("done")])

    result = optimize_retrieval(
        store, bank, optimizer_model=FunctionModel(optimizer),
        k=1, max_evals=3,
    )

    assert len(result.summary) == 2
    table = result.summary_table()
    assert "score" in table and "hit@k" in table
    # bm25 + lexical both score; whichever wins is rank 1; the other is rank 2.
    top = result.pick(1)
    second = result.pick(2)
    assert {top.strategy, second.strategy} == {"lexical", "bm25"}

    written = result.save(2, store)
    assert written.name == CONFIG_FILENAME
    assert "from_optimization: true" in written.read_text()

    # The loader reads config.yaml's retrieval block; settings now carry the
    # picked strategy as the DSL string.
    fresh = load_yaml_config(store.root).retrieval
    assert fresh.from_optimization is True
    assert fresh.strategy == second.strategy


def test_optimize_result_fallback_is_pickable(
    store: WikiStore, bank: QuestionBank
) -> None:
    """When the agent never scores a config, the fallback default eval is
    recorded as a real summary row — so best_config and pick/save agree,
    and pick(1) returns the (pickable) fallback rather than raising. An
    out-of-range rank still raises."""
    def optimizer(messages: object, info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart("done")])

    result = optimize_retrieval(
        store, bank, optimizer_model=FunctionModel(optimizer),
        k=1, max_evals=2,
    )
    assert len(result.summary) == 1
    assert result.pick(1) == result.best_config
    with pytest.raises(OutmemError):
        result.pick(2)  # out of range


def test_optimize_pick_matches_printed_leaderboard_rank() -> None:
    """Regression for the rank-inversion bug: pick(rank) must resolve the
    SAME row the leaderboard prints at that rank, even though the optimizer
    appends rows in eval-arrival order. Built directly (deterministic, no
    fixture-scoring dependence): a low-scoring config is appended FIRST,
    yet pick(1) must return the high-scoring one."""
    from outmem.optimize.bench import Scorecard
    from outmem.optimize.optimizer import EvalRow, OptimizeResult

    loser = RetrievalConfig(strategy="lexical")
    winner = RetrievalConfig(strategy="bm25")
    rows = [  # appended in eval order: loser first
        EvalRow(loser, score=0.40, hit_at_k=0.40, abstention=0.0,
                mean_latency_ms=5.0, p95_latency_ms=6.0),
        EvalRow(winner, score=0.90, hit_at_k=0.90, abstention=0.0,
                mean_latency_ms=3.0, p95_latency_ms=4.0),
    ]
    card = Scorecard(
        score=0.90, hit_at_k=0.90, abstention=0.0, k=1,
        n_answerable=1, n_unanswerable=0, results=(),
    )
    result = OptimizeResult(
        best_config=winner, best_score=0.90, scorecard=card,
        trace=[], notes="", summary=list(rows),
    )
    # __post_init__ sorted the summary score-desc; rank 1 is the winner.
    assert result.pick(1) is winner
    assert result.pick(2) is loser
    # And the printed table leads with the winner, not the first-evaluated.
    first_body_line = result.summary_table().splitlines()[2]
    assert "bm25" in first_body_line


def _row(strategy: str, *, score: float, n_unans: int) -> Any:
    from outmem.optimize.optimizer import EvalRow

    return EvalRow(
        RetrievalConfig(strategy=strategy), score=score, hit_at_k=score,
        abstention=0.0, mean_latency_ms=1.0, p95_latency_ms=1.0,
        n_unanswerable=n_unans,
    )


def test_summary_table_hides_abst_when_no_unanswerables() -> None:
    """All-answerable bank → the structurally-zero abstention column is
    dropped and a banner explains score = hit@k."""
    from outmem.optimize.optimizer import _format_summary_table

    table = _format_summary_table([_row("bm25", score=0.6, n_unans=0)])
    header = table.splitlines()[0]
    assert "abst" not in header  # column gone (the banner may mention it)
    assert "score = hit@k" in table


def test_summary_table_keeps_abst_when_unanswerables_present() -> None:
    """A bank with unanswerables keeps the abstention column and drops the
    banner — the metric is informative."""
    from outmem.optimize.optimizer import _format_summary_table

    table = _format_summary_table([_row("bm25", score=0.6, n_unans=5)])
    assert "abst" in table.splitlines()[0]  # column header present
    assert "score = hit@k" not in table


def test_epoch_line_omits_abstain_without_unanswerables() -> None:
    """The live progress line drops `abstain=` when there's nothing to
    abstain on, matching the table."""
    from outmem.optimize.bench import Scorecard
    from outmem.optimize.optimizer import EvalEvent, _format_epoch

    def card(n_unans: int) -> Scorecard:
        return Scorecard(score=0.6, hit_at_k=0.6, abstention=0.0, k=1,
                         n_answerable=5, n_unanswerable=n_unans, results=())

    cfg = RetrievalConfig(strategy="bm25")
    line_no = _format_epoch(EvalEvent(index=1, max_evals=12, config=cfg,
                                      scorecard=card(0), best_score=0.6))
    line_yes = _format_epoch(EvalEvent(index=1, max_evals=12, config=cfg,
                                       scorecard=card(5), best_score=0.6))
    assert "abstain=" not in line_no
    assert "abstain=" in line_yes


def test_retrieval_block_read_from_config_yaml(tmp_path: Path) -> None:
    """The ``retrieval:`` block in config.yaml drives the settings."""
    from outmem.config import load_yaml_config

    (tmp_path / "config.yaml").write_text(
        "retrieval:\n  strategy: bm25+semantic\n  from_optimization: true\n",
        encoding="utf-8",
    )
    settings = load_yaml_config(tmp_path).retrieval
    assert settings.strategy == "bm25+semantic"
    assert settings.from_optimization is True


def test_config_yaml_bad_strategy_is_lenient(tmp_path: Path) -> None:
    """A bad strategy in config.yaml's retrieval block follows the
    forgiving-load contract: warn + keep the default, NOT crash the open."""
    from outmem.config import load_yaml_config

    (tmp_path / "config.yaml").write_text(
        "retrieval:\n  strategy: not-a-strategy\n", encoding="utf-8",
    )
    settings = load_yaml_config(tmp_path).retrieval  # must not raise
    assert settings.strategy == "bm25"  # default preserved


def test_retrieval_numeric_knobs_reject_bool(tmp_path: Path) -> None:
    """`semantic_top_k: true` must NOT slip through as 1 (bool is an int
    subclass). The knob keeps its default."""
    from outmem.config import load_yaml_config

    (tmp_path / "config.yaml").write_text(
        "retrieval:\n  strategy: bm25\n  semantic_top_k: true\n"
        "  max_candidates: false\n",
        encoding="utf-8",
    )
    settings = load_yaml_config(tmp_path).retrieval
    assert settings.semantic_top_k != 1  # default, not coerced True→1
    assert settings.max_candidates != 0  # default, not coerced False→0


def test_retrieval_rerank_source_field_folds_into_strategy(tmp_path: Path) -> None:
    """`strategy: rerank` + `rerank_source: semantic` resolves to the same
    pipeline as `strategy: rerank(semantic)` instead of being dropped."""
    from outmem.config import load_yaml_config

    (tmp_path / "config.yaml").write_text(
        "retrieval:\n  strategy: rerank\n  rerank_source: semantic\n",
        encoding="utf-8",
    )
    settings = load_yaml_config(tmp_path).retrieval
    assert settings.strategy == "rerank(semantic)"


def test_retrieval_strategy_canonicalised_on_load(tmp_path: Path) -> None:
    """A hand-written `bm25 + semantic` (spaces) is stored canonical."""
    from outmem.config import load_yaml_config

    (tmp_path / "config.yaml").write_text(
        "retrieval:\n  strategy: BM25 + Semantic\n", encoding="utf-8",
    )
    assert load_yaml_config(tmp_path).retrieval.strategy == "bm25+semantic"


def test_save_refreshes_in_memory_store(
    store: WikiStore, bank: QuestionBank
) -> None:
    """save(rank, store) updates store.config.outmem.retrieval in place, so
    an agent built from the same store sees the picked strategy without a
    reopen."""
    state = {"n": 0}

    def optimizer(messages: object, info: AgentInfo) -> ModelResponse:
        state["n"] += 1
        if state["n"] == 1:
            return ModelResponse(parts=[ToolCallPart(
                tool_name="run_eval", args={"strategy": "lexical"})])
        return ModelResponse(parts=[TextPart("done")])

    result = optimize_retrieval(
        store, bank, optimizer_model=FunctionModel(optimizer),
        k=1, max_evals=2,
    )
    assert store.config.outmem.retrieval.from_optimization is False
    result.save(1, store)
    assert store.config.outmem.retrieval.from_optimization is True
    assert store.config.outmem.retrieval.strategy == result.pick(1).strategy


def test_save_preserves_other_config_and_comments(
    store: WikiStore, bank: QuestionBank, tmp_path: Path
) -> None:
    """save() replaces ONLY the retrieval: block in config.yaml, leaving
    other settings and comments byte-intact; creates parent + no .tmp."""
    dest = tmp_path / "nested" / "config.yaml"
    dest.parent.mkdir(parents=True)
    dest.write_text(
        "model: anthropic:claude-sonnet-4-6\n\n"
        "# my notes\n"
        "retrieval:\n  strategy: lexical\n\n"
        "logfire:\n  enabled: true\n",
        encoding="utf-8",
    )

    def optimizer(messages: object, info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart("done")])

    result = optimize_retrieval(
        store, bank, optimizer_model=FunctionModel(optimizer),
        k=1, max_evals=1,
    )
    written = result.save(1, store, path=dest)
    assert written == dest
    text = dest.read_text(encoding="utf-8")
    assert "model: anthropic:claude-sonnet-4-6" in text  # untouched
    assert "# my notes" in text  # comment preserved
    assert "logfire:\n  enabled: true" in text  # later block intact
    assert "from_optimization: true" in text  # new block written
    assert text.count("retrieval:") == 1
    assert not (dest.parent / f"{dest.name}.tmp").exists()


def test_save_rejects_unparseable_roundtrip_via_format_guard() -> None:
    """format_strategy refuses to render a hybrid whose legs aren't DSL
    atomics (e.g. a 'rerank' leg) — so save() can never write a
    retrieval block that the next load would reject."""
    from outmem.optimize.dsl import format_strategy

    with pytest.raises(OutmemError):
        format_strategy({"strategy": "hybrid", "fuse": ["rerank", "semantic"]})


def test_fuse_rejects_rerank_leg() -> None:
    """RetrievalConfig.from_dict rejects 'rerank' as a fuse leg at the
    source, so the optimizer can never score/pick a bricking config."""
    with pytest.raises(OutmemError):
        RetrievalConfig.from_dict({"strategy": "hybrid", "fuse": ["rerank", "bm25"]})


def test_prewarm_query_cache_warms_every_question() -> None:
    """Pre-warm embeds each answerable + unanswerable question once, so the
    first semantic eval isn't a cold-cache latency outlier."""
    from outmem.optimize.optimizer import _prewarm_query_cache

    seen: list[str] = []
    store = SimpleNamespace(
        semantic_enabled=lambda: True,
        semantic_index_is_empty=lambda: False,
        semantic_find_similar=lambda text, top_k, threshold: seen.append(text),
    )
    bank = QuestionBank(
        answerable=[Question(question="q1"), Question(question="q2")],
        unanswerable=[Question(question="u1", gold_slugs=())],
    )
    _prewarm_query_cache(store, bank, max_concurrency=4)  # type: ignore[arg-type]
    assert sorted(seen) == ["q1", "q2", "u1"]


def test_prewarm_query_cache_noop_when_semantic_off() -> None:
    """No semantic index → warming is a no-op (and never raises)."""
    from outmem.optimize.optimizer import _prewarm_query_cache

    called: list[int] = []
    store = SimpleNamespace(
        semantic_enabled=lambda: False,
        semantic_find_similar=lambda *a, **k: called.append(1),
    )
    bank = QuestionBank(answerable=[Question(question="q1")])
    _prewarm_query_cache(store, bank, max_concurrency=2)  # type: ignore[arg-type]
    assert called == []


def test_dsl_vocab_no_drift() -> None:
    """Single source of truth: the DSL atomic vocabulary is exactly the
    retriever atomics minus the rerank gate. If someone adds an atomic to
    one tuple and not the other, this fails (preventing optimizer-picks
    the loader can't parse)."""
    from outmem.optimize.blocks import _ATOMIC_STRATEGIES
    from outmem.optimize.dsl import _DSL_ATOMICS

    assert set(_DSL_ATOMICS) == set(_ATOMIC_STRATEGIES) - {"rerank"}


def test_describe_config_distinguishes_tuned_knobs() -> None:
    """rerank trials that differ only in numeric knobs must render as
    DISTINCT labels, so tuning a family doesn't look like repeated runs
    of the same config in the epoch lines / leaderboard."""
    from outmem.optimize.optimizer import _describe_config

    a = RetrievalConfig(strategy="rerank", rerank_source="bm25", max_candidates=30,
                        max_relevant=8)
    b = RetrievalConfig(strategy="rerank", rerank_source="bm25", max_candidates=50,
                        max_relevant=8)
    assert _describe_config(a) != _describe_config(b)
    assert "cand=30" in _describe_config(a) and "cand=50" in _describe_config(b)


def test_optimize_surfaces_unavailable_strategy_on_stderr(
    store: WikiStore, bank: QuestionBank, capsys: pytest.CaptureFixture[str]
) -> None:
    """When a strategy is refused (e.g. semantic with no index), the skip is
    printed to stderr — not just buried in result.log — so a whole family
    vanishing is visible rather than looking like the agent ignored it."""
    state = {"n": 0}

    def optimizer(messages: object, info: AgentInfo) -> ModelResponse:
        state["n"] += 1
        if state["n"] == 1:
            return ModelResponse(parts=[ToolCallPart(
                tool_name="run_eval", args={"strategy": "semantic"})])
        return ModelResponse(parts=[TextPart("done")])

    result = optimize_retrieval(
        store, bank, optimizer_model=FunctionModel(optimizer), k=1, max_evals=2,
    )
    err = capsys.readouterr().err
    assert "semantic unavailable" in err  # loud on stderr
    assert any("unavailable" in line for line in result.log)  # and in the log


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
