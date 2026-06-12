"""Tests for ``outmem.relevance`` — the LLM relevance gate (``judge_relevance``)
the rerank retrieval strategy uses, plus its per-thread model cache.

Uses ``pydantic_ai.models.function.FunctionModel`` to script the gate
model's structured output deterministically (no real LLM).
"""

from __future__ import annotations

from typing import ClassVar

import pytest
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from outmem.relevance import judge_relevance


def _model_returning(selections: list[dict[str, str]]) -> FunctionModel:
    """A FunctionModel that emits the structured ``_FilterResult`` once."""

    def respond(messages: object, info: AgentInfo) -> ModelResponse:
        name = info.output_tools[0].name
        return ModelResponse(
            parts=[ToolCallPart(tool_name=name, args={"relevant": selections})]
        )

    return FunctionModel(respond)


def _exploding_model() -> FunctionModel:
    def respond(messages: object, info: AgentInfo) -> ModelResponse:
        raise RuntimeError("gate model is down")

    return FunctionModel(respond)


class TestJudgeRelevance:
    """The gate over pre-built (slug, excerpt) candidates."""

    _cands: ClassVar = [
        ("abx:penicillin", "IV penicillin G dosing for endocarditis"),
        ("abx:ceftriaxone", "ceftriaxone once-daily alternative"),
        ("pricing-formula", "cost-plus 35% on product sales"),
    ]

    def test_keeps_only_selected(self) -> None:
        model = _model_returning([{"slug": "abx:penicillin", "reason": "dosing"}])
        kept, err = judge_relevance(
            model=model, query="penicillin", candidates=self._cands, max_relevant=5
        )
        assert kept == ("abx:penicillin",)
        assert err is None

    def test_invented_slug_dropped(self) -> None:
        model = _model_returning(
            [{"slug": "made-up", "reason": "x"},
             {"slug": "abx:ceftriaxone", "reason": "alt"}]
        )
        kept, _ = judge_relevance(
            model=model, query="alternative", candidates=self._cands, max_relevant=5
        )
        assert kept == ("abx:ceftriaxone",)  # invented slug filtered out

    def test_empty_candidates_returns_empty(self) -> None:
        kept, err = judge_relevance(
            model=_model_returning([]), query="x", candidates=[], max_relevant=5
        )
        assert kept == () and err is None

    def test_model_error_falls_back_to_source_order(self) -> None:
        kept, err = judge_relevance(
            model=_exploding_model(), query="penicillin",
            candidates=self._cands, max_relevant=2,
        )
        # source order, capped — retrieval never gets worse on a gate failure
        assert kept == ("abx:penicillin", "abx:ceftriaxone")
        assert err is not None and "RuntimeError" in err

    def test_brief_error_is_concise(self) -> None:
        # A content-filter refusal carries a multi-KB JSON body; the fallback
        # log must collapse it to one capped line, not dump the whole thing.
        from outmem.relevance import _brief_error

        exc = RuntimeError("Content filter triggered.\n" + "x" * 5000)
        out = _brief_error(exc)
        assert "\n" not in out
        assert len(out) <= 200
        assert out.startswith("RuntimeError: Content filter triggered.")


class TestInferModelCached:
    """The per-thread model cache that stops the optimizer's threaded
    rerank/hyde eval from opening a fresh httpx client (and FDs) per call —
    the `OSError: Too many open files` regression."""

    def test_string_inferred_once_per_thread(self) -> None:
        from outmem.relevance import infer_model_cached

        a = infer_model_cached("test")  # pydantic_ai builds a TestModel
        b = infer_model_cached("test")
        assert a is b  # memoised, not rebuilt

    def test_distinct_strings_distinct_models(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from outmem.relevance import infer_model_cached

        # Dummy key: provider construction reads it but makes no network call
        # at inference time, so two ids resolve to two distinct cached models.
        monkeypatch.setenv("ANTHROPIC_API_KEY", "x-not-real")
        a = infer_model_cached("anthropic:claude-haiku-4-5")
        b = infer_model_cached("anthropic:claude-sonnet-4-6")
        assert a is not b
        assert infer_model_cached("anthropic:claude-haiku-4-5") is a  # still cached

    def test_model_instance_passes_through(self) -> None:
        from pydantic_ai.models.test import TestModel

        from outmem.relevance import infer_model_cached

        m = TestModel()
        assert infer_model_cached(m) is m  # concrete model returned as-is

    def test_cache_is_per_thread(self) -> None:
        import threading

        from outmem.relevance import infer_model_cached

        main = infer_model_cached("test")
        other: list[object] = []
        t = threading.Thread(target=lambda: other.append(infer_model_cached("test")))
        t.start()
        t.join()
        # Different thread → its own client-bearing model instance (httpx
        # clients are loop/thread-bound, so they must not be shared).
        assert other and other[0] is not main

    def test_judge_relevance_infers_model_once_per_thread(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The leak's mechanism, pinned: many judge_relevance calls in one
        thread (the rerank eval inner loop) must build the model — and thus
        the httpx client — exactly once, not once per call."""
        from pydantic_ai.models import Model

        from outmem.relevance import judge_relevance

        calls = {"n": 0}
        stub = _model_returning([])  # a stub model, no network

        def counting_infer(model: object) -> Model:
            # Mirror the real infer_model: a concrete Model passes through
            # untouched (this is what Agent.__init__ calls internally). Only
            # a *string* triggers a build — that's the call we're counting.
            if isinstance(model, Model):
                return model
            calls["n"] += 1
            return stub

        monkeypatch.setattr("pydantic_ai.models.infer_model", counting_infer)
        for _ in range(5):
            judge_relevance(
                model="anthropic:leak-probe",  # unique string → fresh cache slot
                query="penicillin dose",
                candidates=[("abx:penicillin", "IV penicillin G dosing")],
                max_relevant=3,
            )
        assert calls["n"] == 1  # 5 calls, one inference (one client) in this thread
