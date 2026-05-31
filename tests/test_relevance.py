"""Tests for ``outmem.relevance`` — the cheap-model relevance filter.

Uses ``pydantic_ai.models.function.FunctionModel`` to script the
filter model's structured output deterministically (no real LLM): we
return a specific ``_FilterResult`` (or raise) and assert the gate's
contract — select-only, empty allowed, fallback-on-error, and the
no-LLM-content invariant.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from outmem.relevance import FilterOutcome, relevance_filter
from outmem.store import WikiStore


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
        raise RuntimeError("triage model is down")

    return FunctionModel(respond)


@pytest.fixture
def store(tmp_path: Path) -> WikiStore:
    s = WikiStore.init(tmp_path / "w")
    s.write_page(
        "abx:penicillin",
        title="Penicillin",
        body="IV penicillin G 18-24 MU/day in divided doses for endocarditis.",
    )
    s.write_page(
        "abx:ceftriaxone",
        title="Ceftriaxone",
        body="ceftriaxone 2g IV q24h; a penicillin alternative for beta-lactam allergy.",
    )
    s.write_page(
        "pricing-formula",
        title="Pricing",
        body="cost-plus 35% applied to penicillin product sales.",
    )
    return s


class TestRelevanceFilter:
    def test_keeps_only_selected_subset(self, store: WikiStore) -> None:
        model = _model_returning(
            [{"slug": "abx:penicillin", "reason": "IV penicillin dosing"}]
        )
        out = relevance_filter(store, query="penicillin", model=model)
        assert [p.slug for p in out.kept] == ["abx:penicillin"]
        assert out.kept[0].reason == "IV penicillin dosing"
        assert not out.fell_back
        # The candidate net saw all three pages that mention "penicillin".
        assert out.candidates_considered == 3

    def test_supporting_lines_are_real_hits(self, store: WikiStore) -> None:
        model = _model_returning([{"slug": "abx:penicillin", "reason": "dosing"}])
        out = relevance_filter(store, query="penicillin", model=model)
        # Lines are verbatim ripgrep hits (deterministic), not model text.
        assert out.kept[0].lines
        assert any("penicillin" in h.text.lower() for h in out.kept[0].lines)

    def test_invented_slug_is_dropped(self, store: WikiStore) -> None:
        model = _model_returning(
            [
                {"slug": "abx:penicillin", "reason": "ok"},
                {"slug": "hallucinated-slug", "reason": "should be dropped"},
            ]
        )
        out = relevance_filter(store, query="penicillin", model=model)
        assert [p.slug for p in out.kept] == ["abx:penicillin"]

    def test_empty_selection_allowed(self, store: WikiStore) -> None:
        out = relevance_filter(store, query="penicillin", model=_model_returning([]))
        assert out.kept == ()
        assert not out.fell_back  # empty ≠ failure

    def test_no_candidates_returns_empty_not_fallback(self, store: WikiStore) -> None:
        out = relevance_filter(
            store, query="nonexistent-token-xyz", model=_model_returning([])
        )
        assert out.kept == ()
        assert out.candidates_considered == 0
        assert not out.fell_back

    def test_model_error_falls_back_to_lexical(self, store: WikiStore) -> None:
        out = relevance_filter(store, query="penicillin", model=_exploding_model())
        assert out.fell_back is True
        # Lexical fallback keeps the candidate hits, in slug order, no reason.
        assert {p.slug for p in out.kept} >= {"abx:penicillin"}
        assert all(p.reason == "" for p in out.kept)
        # The fallback records WHY (the brief reason) on the outcome.
        assert out.error and "RuntimeError" in out.error

    def test_max_relevant_caps_kept(self, store: WikiStore) -> None:
        model = _model_returning(
            [
                {"slug": "abx:penicillin", "reason": "a"},
                {"slug": "abx:ceftriaxone", "reason": "b"},
                {"slug": "pricing-formula", "reason": "c"},
            ]
        )
        out = relevance_filter(store, query="penicillin", model=model, max_relevant=2)
        assert len(out.kept) == 2

    def test_lines_context_does_not_read_pages(self, store: WikiStore) -> None:
        # context="lines" must still work and keep the selected subset.
        model = _model_returning([{"slug": "abx:penicillin", "reason": "dosing"}])
        out = relevance_filter(
            store, query="penicillin", model=model, context="lines"
        )
        assert [p.slug for p in out.kept] == ["abx:penicillin"]

    def test_fallback_log_is_concise(self) -> None:
        # A content-filter refusal carries a multi-KB JSON body; the fallback
        # log must collapse it to one capped line, not dump the whole thing.
        from outmem.relevance import _brief_error

        exc = RuntimeError("Content filter triggered.\n" + "x" * 5000)
        out = _brief_error(exc)
        assert "\n" not in out
        assert len(out) <= 200
        assert out.startswith("RuntimeError: Content filter triggered.")

    def test_survives_non_utf8_page(self, store: WikiStore) -> None:
        """A non-UTF-8 page must not crash the filter — plain search
        tolerates it, so the filtered variant must too (regression:
        _excerpt only caught OutmemError, letting UnicodeDecodeError escape)."""
        (store.pages_path / "badpage.md").write_bytes(b"penicillin \xff\xfe bytes\n")
        out = relevance_filter(store, query="penicillin", model=_model_returning([]))
        assert isinstance(out, FilterOutcome)  # did not raise


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
