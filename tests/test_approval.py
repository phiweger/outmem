"""Tests for the HITL approval gate around agent writes.

Approval is opt-in via ``approval.required_for_writes: true`` in
``config.yaml``. When on, the agent's ``write_page`` / ``extend_page``
tool calls are deferred — the underlying git commit only happens after
a :class:`outmem.agent.approval.Reviewer` returns a verdict. The
tests below use a deterministic :class:`FunctionModel` to script the
agent's tool-call sequence and a :class:`RecordingReviewer` to script
the human verdicts, so the round-trip is end-to-end exercised without
a real LLM or a real terminal.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai.messages import (
    ModelResponse,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import ToolApproved, ToolDenied

from outmem.agent import (
    AutoApproveReviewer,
    AutoDenyReviewer,
    CliReviewer,
    RecordingReviewer,
    ask_sync,
)
from outmem.exceptions import OutmemError, WritebackError
from outmem.store import WikiStore

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def wiki_approval_on(tmp_path: Path) -> WikiStore:
    """A fresh wiki with ``approval.required_for_writes: true``."""
    root = tmp_path / "wiki"
    WikiStore.init(root)
    yaml_path = root / "config.yaml"
    text = yaml_path.read_text(encoding="utf-8").replace(
        "approval:\n  required_for_writes: false",
        "approval:\n  required_for_writes: true",
    )
    yaml_path.write_text(text, encoding="utf-8")
    return WikiStore.open(root)


@pytest.fixture
def wiki_approval_off(tmp_path: Path) -> WikiStore:
    """Approval flag explicitly off — confirms the non-gated path still works."""
    root = tmp_path / "wiki-off"
    WikiStore.init(root)
    return WikiStore.open(root)


def _script(*calls: dict[str, Any], reply: str = "done.") -> FunctionModel:
    """Build a FunctionModel that emits ``calls`` (one ToolCallPart per
    model turn) then a final ``TextPart(reply)``. Same shape as the
    helper in ``test_agent.py`` but reproduced here so the tests are
    independent."""
    state = {"step": 0}

    async def _runner(messages: list[object], info: AgentInfo) -> ModelResponse:
        idx = state["step"]
        state["step"] = idx + 1
        if idx < len(calls):
            entry = calls[idx]
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name=str(entry["tool"]),
                        args=dict(entry["args"]),
                    )
                ]
            )
        return ModelResponse(parts=[TextPart(content=reply)])

    return FunctionModel(_runner)


def _committed_paths_for_head(repo: Path) -> set[str]:
    return set(
        subprocess.run(
            ["git", "show", "--name-only", "--pretty=format:", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
    )


# ---------------------------------------------------------------------------
# Config plumbing
# ---------------------------------------------------------------------------


class TestApprovalConfig:
    def test_default_off(self, wiki_approval_off: WikiStore) -> None:
        assert wiki_approval_off.config.outmem.approval.required_for_writes is False

    def test_yaml_flag_flips_on(self, wiki_approval_on: WikiStore) -> None:
        assert wiki_approval_on.config.outmem.approval.required_for_writes is True


# ---------------------------------------------------------------------------
# Deferred-tool round trip via ask_sync
# ---------------------------------------------------------------------------


class TestAskApprovalGate:
    def test_approve_commits_the_write(self, wiki_approval_on: WikiStore) -> None:
        model = _script(
            {
                "tool": "write_page",
                "args": {
                    "slug": "pricing-formula",
                    "title": "Pricing formula",
                    "body": "cost-plus 35%.\n",
                },
            },
            reply="Wrote pricing-formula.",
        )
        reviewer = RecordingReviewer({"write_page": [True]})

        result = ask_sync(
            wiki_approval_on,
            query="what is our pricing formula?",
            model=model,
            push=False,
            pull=False,
            record=False,
            reviewer=reviewer,
            include_steering=False,
        )

        assert result.response == "Wrote pricing-formula."
        assert any("compact: pricing-formula" in c.subject for c in result.commits)
        assert "wiki/pages/pricing-formula.md" in _committed_paths_for_head(wiki_approval_on.root)
        # The reviewer saw the proposed call.
        assert reviewer.calls and reviewer.calls[0][0] == "write_page"

    def test_approve_with_override_args_commits_revised_body(
        self, wiki_approval_on: WikiStore
    ) -> None:
        model = _script(
            {
                "tool": "write_page",
                "args": {
                    "slug": "pricing-formula",
                    "title": "Pricing formula",
                    "body": "cost-plus 30%.\n",  # the model's proposal
                },
            },
            reply="Done.",
        )
        # The reviewer corrects the figure before approving.
        reviewer = RecordingReviewer(
            {
                "write_page": [
                    ToolApproved(
                        override_args={
                            "slug": "pricing-formula",
                            "title": "Pricing formula",
                            "body": "cost-plus 35% (corrected by reviewer).\n",
                        }
                    )
                ]
            }
        )

        ask_sync(
            wiki_approval_on,
            query="x?",
            model=model,
            push=False,
            pull=False,
            record=False,
            reviewer=reviewer,
            include_steering=False,
        )

        body = (wiki_approval_on.pages_path / "pricing-formula.md").read_text(
            encoding="utf-8"
        )
        assert "cost-plus 35% (corrected by reviewer)." in body
        assert "cost-plus 30%" not in body

    def test_deny_does_not_commit_and_agent_falls_back_to_log(
        self, wiki_approval_on: WikiStore
    ) -> None:
        # Turn 1: model proposes write_page (deferred → denied).
        # Turn 2: model recovers with append_log to satisfy writeback.
        # Turn 3: model emits the final text reply.
        model = _script(
            {
                "tool": "write_page",
                "args": {
                    "slug": "pricing-formula",
                    "title": "Pricing formula",
                    "body": "speculative.\n",
                },
            },
            {
                "tool": "append_log",
                "args": {
                    "topic": "denied-write",
                    "content": "- reviewer denied my write_page proposal.\n",
                },
            },
            reply="OK, logged.",
        )
        reviewer = RecordingReviewer(
            {"write_page": [ToolDenied(message="Don't speculate.")]}
        )

        result = ask_sync(
            wiki_approval_on,
            query="x?",
            model=model,
            push=False,
            pull=False,
            record=False,
            reviewer=reviewer,
            include_steering=False,
        )

        # Page was NOT created.
        assert not (wiki_approval_on.wiki_path / "pricing-formula.md").exists()
        # But the agent did make a writeback (via append_log) so the run
        # is considered successful.
        assert any("log: denied-write" in c.subject for c in result.commits)
        assert result.response == "OK, logged."

    def test_required_without_reviewer_raises(
        self, wiki_approval_on: WikiStore
    ) -> None:
        # No model call needed — ask should fail at the precheck.
        with pytest.raises(OutmemError, match="reviewer"):
            ask_sync(
                wiki_approval_on,
                query="x?",
                model=_script(reply="x"),
                push=False,
                pull=False,
                record=False,
                reviewer=None,
                include_steering=False,
            )

    def test_off_path_unaffected_by_reviewer_arg(
        self, wiki_approval_off: WikiStore
    ) -> None:
        """When approval is off, the agent uses the plain tools= path
        and a reviewer argument is harmless (the agent never produces a
        DeferredToolRequests, so the reviewer is never consulted)."""
        model = _script(
            {
                "tool": "append_log",
                "args": {
                    "topic": "smoke",
                    "content": "- still works without approval.\n",
                },
            },
            reply="done.",
        )
        reviewer = RecordingReviewer({})  # empty — never consulted

        result = ask_sync(
            wiki_approval_off,
            query="x?",
            model=model,
            push=False,
            pull=False,
            record=False,
            reviewer=reviewer,
            include_steering=False,
        )
        assert result.response == "done."
        assert any("log: smoke" in c.subject for c in result.commits)
        assert reviewer.calls == []


# ---------------------------------------------------------------------------
# Reviewer implementations
# ---------------------------------------------------------------------------


class TestCliReviewer:
    """The CLI reviewer is interactive; drive it with a fake input stream."""

    def _fake_call(self, name: str, args: dict[str, Any]) -> Any:
        # Build a minimal stand-in for ToolCallPart that exposes the
        # attributes the reviewer touches.
        class _Call:
            def __init__(self) -> None:
                self.tool_name = name
                self.args = args
                self.tool_call_id = "test-id"

            def args_as_dict(self) -> dict[str, Any]:
                return dict(args)

        return _Call()

    def test_approve(self) -> None:
        inputs = iter(["a"])
        out: list[str] = []

        class _Stream:
            def write(self, s: str) -> None:
                out.append(s)

            def flush(self) -> None:
                pass

        reviewer = CliReviewer(
            stream=_Stream(),
            input_fn=lambda _prompt: next(inputs),
            edit_fn=lambda _body: _body,
        )
        verdict = reviewer.review(
            self._fake_call("write_page", {"slug": "x", "body": "hi"})
        )
        assert isinstance(verdict, ToolApproved)
        assert verdict.override_args is None
        # The proposal was rendered.
        assert any("write_page" in line for line in out)

    def test_deny_with_reason(self) -> None:
        inputs = iter(["d", "speculative"])
        reviewer = CliReviewer(
            stream=type("S", (), {"write": lambda *a: None, "flush": lambda *a: None})(),
            input_fn=lambda _prompt: next(inputs),
            edit_fn=lambda _body: _body,
        )
        verdict = reviewer.review(
            self._fake_call("extend_page", {"slug": "x", "body": "guess"})
        )
        assert isinstance(verdict, ToolDenied)
        assert "speculative" in verdict.message

    def test_edit_then_approve(self) -> None:
        inputs = iter(["e"])
        reviewer = CliReviewer(
            stream=type("S", (), {"write": lambda *a: None, "flush": lambda *a: None})(),
            input_fn=lambda _prompt: next(inputs),
            edit_fn=lambda body: body.replace("30%", "35%"),
        )
        verdict = reviewer.review(
            self._fake_call(
                "write_page",
                {"slug": "x", "title": "T", "body": "cost-plus 30%."},
            )
        )
        assert isinstance(verdict, ToolApproved)
        assert verdict.override_args is not None
        assert verdict.override_args["body"] == "cost-plus 35%."
        # Other args preserved.
        assert verdict.override_args["slug"] == "x"

    def test_unrecognised_then_approve(self) -> None:
        inputs = iter(["zzz", "a"])
        reviewer = CliReviewer(
            stream=type("S", (), {"write": lambda *a: None, "flush": lambda *a: None})(),
            input_fn=lambda _prompt: next(inputs),
            edit_fn=lambda _body: _body,
        )
        verdict = reviewer.review(
            self._fake_call("write_page", {"slug": "x", "body": "ok"})
        )
        assert isinstance(verdict, ToolApproved)


class TestStaticReviewers:
    def test_auto_approve(self) -> None:
        class _Call:
            tool_name = "write_page"
            tool_call_id = "id"

            def __init__(self) -> None:
                self.args: dict[str, Any] = {}

        verdict = AutoApproveReviewer().review(_Call())
        assert isinstance(verdict, ToolApproved)

    def test_auto_deny(self) -> None:
        class _Call:
            tool_name = "write_page"
            tool_call_id = "id"

            def __init__(self) -> None:
                self.args: dict[str, Any] = {}

        verdict = AutoDenyReviewer().review(_Call())
        assert isinstance(verdict, ToolDenied)
        assert "non-interactive" in verdict.message or "no interactive" in verdict.message.lower()


# ---------------------------------------------------------------------------
# require_interactive_reviewer
# ---------------------------------------------------------------------------


class TestRequireInteractiveReviewer:
    def test_off_returns_none(self) -> None:
        from outmem.agent import require_interactive_reviewer

        assert require_interactive_reviewer(False) is None

    def test_required_no_tty_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from outmem.agent import require_interactive_reviewer

        # Pytest captures stdin so isatty is False in this environment.
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        with pytest.raises(OutmemError, match="not a tty"):
            require_interactive_reviewer(True)

    def test_required_tty_returns_cli_reviewer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from outmem.agent import require_interactive_reviewer

        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        reviewer = require_interactive_reviewer(True)
        assert isinstance(reviewer, CliReviewer)


# ---------------------------------------------------------------------------
# Writeback discipline under denial
# ---------------------------------------------------------------------------


def test_deny_without_fallback_raises_writeback_error(
    wiki_approval_on: WikiStore,
) -> None:
    """Mandatory writeback (spec §9) still applies. If the agent's only
    proposal is denied and it doesn't fall back to ``append_log``,
    ``ask_sync`` must raise ``WritebackError`` — the gate is not an
    escape hatch from the writeback contract."""

    # Turn 1: write_page (denied). Turn 2: the model gives up and
    # produces a text reply with no append_log.
    model = _script(
        {
            "tool": "write_page",
            "args": {"slug": "x", "title": "T", "body": "guess.\n"},
        },
        reply="I'll stop here.",
    )
    reviewer = RecordingReviewer(
        {"write_page": [ToolDenied(message="No.")]}
    )

    with pytest.raises(WritebackError):
        ask_sync(
            wiki_approval_on,
            query="x?",
            model=model,
            push=False,
            pull=False,
            record=False,
            reviewer=reviewer,
            include_steering=False,
        )
