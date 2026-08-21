"""Detecting a write made by a turn that ran out of output room.

The elision guard catches a body the model *marked* as cut. This is the
case it did not mark: the response ends on ``length`` — Anthropic's
``max_tokens``, or a blown context window — and that same response called
a content-write tool. Whatever it wrote is suspect however complete it
looks, and no other signal downstream can tell that page from a finished
one.

Conservative by construction: it flags for verification rather than
failing the run, because by the time the turn is over the commit has
landed and the honest report is "check this page", not a rollback.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from outmem.agent.service import _budget_truncated_writes, ask_sync
from outmem.store import WikiStore


@pytest.fixture
def store(tmp_path: Path) -> WikiStore:
    return WikiStore.init(tmp_path / "w")


class _Run:
    """Minimal stand-in for a PydanticAI run result."""

    def __init__(self, messages: list[object]) -> None:
        self._messages = messages

    def all_messages(self) -> list[object]:
        return self._messages


def _write_call(slug: str = "clinical:sepsis") -> ToolCallPart:
    return ToolCallPart(
        tool_name="write_page",
        args={"slug": slug, "title": "S", "body": "text"},
    )


class TestDetection:
    def test_length_finish_with_a_write_is_flagged(self) -> None:
        run = _Run([ModelResponse(parts=[_write_call()], finish_reason="length")])
        assert _budget_truncated_writes(run) == ("write_page(clinical:sepsis)",)

    def test_normal_finish_is_not_flagged(self) -> None:
        run = _Run([ModelResponse(parts=[_write_call()], finish_reason="tool_call")])
        assert _budget_truncated_writes(run) == ()

    def test_length_finish_without_a_write_is_not_flagged(self) -> None:
        """A long *answer* that hit the cap is a truncated reply, not a
        truncated page — noisy to report as a wiki-integrity problem."""
        run = _Run([ModelResponse(parts=[TextPart(content="…")], finish_reason="length")])
        assert _budget_truncated_writes(run) == ()

    def test_every_content_write_tool_counts(self) -> None:
        for tool in ("write_page", "extend_page", "append_page"):
            run = _Run(
                [
                    ModelResponse(
                        parts=[ToolCallPart(tool_name=tool, args={"slug": "a:b"})],
                        finish_reason="length",
                    )
                ]
            )
            assert _budget_truncated_writes(run) == (f"{tool}(a:b)",)

    def test_non_content_tools_do_not_count(self) -> None:
        """append_log writes a log line, not a page — a short log entry is
        not a wiki-completeness problem."""
        run = _Run(
            [
                ModelResponse(
                    parts=[ToolCallPart(tool_name="append_log", args={"topic": "t"})],
                    finish_reason="length",
                )
            ]
        )
        assert _budget_truncated_writes(run) == ()

    def test_duplicates_are_collapsed(self) -> None:
        parts = [_write_call(), _write_call()]
        run = _Run([ModelResponse(parts=parts, finish_reason="length")])
        assert _budget_truncated_writes(run) == ("write_page(clinical:sepsis)",)

    def test_missing_finish_reason_flags_nothing(self) -> None:
        """outmem supports pydantic-ai versions predating finish_reason;
        there the signal is simply unavailable, not wrong."""

        class _Old:
            parts: ClassVar = [_write_call()]

        assert _budget_truncated_writes(_Run([_Old()])) == ()

    def test_a_result_without_messages_is_safe(self) -> None:
        assert _budget_truncated_writes(object()) == ()


class TestEndToEnd:
    """Through a real agent run, so the wiring is covered rather than the
    helper alone."""

    def _model(self, *, finish_reason: str) -> FunctionModel:
        calls = {"n": 0}

        def respond(messages: object, info: AgentInfo) -> ModelResponse:
            calls["n"] += 1
            if calls["n"] == 1:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="write_page",
                            args={
                                "slug": "clinical:sepsis",
                                "title": "Sepsis",
                                "body": "## Erreger\n\nGramnegative Erreger.\n",
                            },
                        )
                    ],
                    finish_reason=finish_reason,
                )
            return ModelResponse(parts=[TextPart(content="done")])

        return FunctionModel(respond)

    def test_flagged_on_the_result(self, store: WikiStore) -> None:
        result = ask_sync(
            store,
            query="compact sepsis",
            model=self._model(finish_reason="length"),
            push=False,
            pull=False,
        )
        assert result.budget_truncated_writes == ("write_page(clinical:sepsis)",)
        # The commit still landed — the flag is a request to verify, not a
        # rollback.
        assert result.wrote_back

    def test_clean_run_flags_nothing(self, store: WikiStore) -> None:
        result = ask_sync(
            store,
            query="compact sepsis",
            model=self._model(finish_reason="tool_call"),
            push=False,
            pull=False,
        )
        assert result.budget_truncated_writes == ()
