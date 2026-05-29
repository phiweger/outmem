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
