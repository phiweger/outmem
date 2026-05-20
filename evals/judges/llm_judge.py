"""Structured LLM judge.

Each :meth:`EvalRun.judge` call sends one (criterion, response) pair
to a separate :class:`pydantic_ai.Agent` configured with
``output_type=JudgeVerdict``. The judge defaults to
``anthropic:claude-sonnet-4-6``; override via ``--judge-model`` or
:func:`set_judge_model`.

The judge agent is held in a module-level cache so repeated calls
within a single eval run reuse the same client / context.
"""

from __future__ import annotations

import asyncio
from typing import Any

from pydantic import BaseModel, Field

DEFAULT_JUDGE_MODEL = "anthropic:claude-sonnet-4-6"


def _resolve_default() -> str:
    """Pick the initial judge model from the repo-level config, falling
    back to :data:`DEFAULT_JUDGE_MODEL`. Called once at module import.

    CLI flags / explicit :func:`set_judge_model` calls still win — they
    overwrite the cached value after this initial resolution.
    """
    from evals.config import load_repo_config

    cfg = load_repo_config()
    return cfg.judge_model or DEFAULT_JUDGE_MODEL


_JUDGE_MODEL = _resolve_default()
_JUDGE_AGENT: Any | None = None


class JudgeVerdict(BaseModel):
    """One graded criterion."""

    passed: bool = Field(
        description="True if and only if the criterion is clearly satisfied."
    )
    reasoning: str = Field(
        description=(
            "One short sentence (≤ 200 chars) explaining the verdict. "
            "If passed=False, the sentence must say what's missing or wrong."
        )
    )


_SYSTEM_PROMPT = (
    "You are evaluating whether an agent's response satisfies a specific "
    "criterion. Be strict — return passed=True only when the criterion is "
    "clearly and unambiguously met. Surface superficial wording that "
    "doesn't actually answer the criterion as a failure. Always provide a "
    "one-sentence reasoning."
)


def set_judge_model(model: str) -> None:
    """Override the judge model. Clears the cached agent."""
    global _JUDGE_MODEL, _JUDGE_AGENT
    _JUDGE_MODEL = model
    _JUDGE_AGENT = None


def _get_agent() -> Any:
    global _JUDGE_AGENT
    if _JUDGE_AGENT is None:
        from pydantic_ai import Agent

        _JUDGE_AGENT = Agent(
            _JUDGE_MODEL,
            output_type=JudgeVerdict,
            system_prompt=_SYSTEM_PROMPT,
        )
    return _JUDGE_AGENT


def grade(*, criterion: str, response: str) -> JudgeVerdict:
    """Synchronous wrapper: grade ``response`` against ``criterion``.

    The agent is async; we run it to completion via :func:`asyncio.run`
    so case bodies stay synchronous (matching :func:`outmem.agent.ask_sync`).
    """
    agent = _get_agent()
    prompt = (
        f"Criterion to evaluate:\n{criterion}\n\n"
        f"Agent's response:\n---\n{response}\n---\n"
    )
    result = asyncio.run(agent.run(prompt))
    return result.output  # type: ignore[no-any-return]
