"""Standalone agent runtime for outmem (``outmem[agent]`` extra).

The runtime wires :class:`outmem.store.WikiStore` to a PydanticAI
agent, drives the orient → retrieve → compact loop (spec v0.5 §6,
planning prompt §3), and enforces the mandatory-writeback contract.

Public surface::

    from outmem.agent import build_agent, ask, ask_sync, AskResult

    answer = ask_sync(store, query="what did we decide about pricing?")

The agent is model-agnostic: the consumer sets ``OUTMEM_MODEL`` (e.g.
``anthropic:claude-sonnet-4-6``, ``openai:gpt-5``) and provides the
appropriate API key in the environment. PydanticAI handles model
resolution.
"""

from __future__ import annotations

from outmem.agent.approval import (
    AutoApproveReviewer,
    AutoDenyReviewer,
    CliReviewer,
    RecordingReviewer,
    Reviewer,
    require_interactive_reviewer,
)
from outmem.agent.runtime import build_agent, render_system_prompt
from outmem.agent.service import AskResult, TokenUsage, ask, ask_sync

__all__ = [
    "AskResult",
    "AutoApproveReviewer",
    "AutoDenyReviewer",
    "CliReviewer",
    "RecordingReviewer",
    "Reviewer",
    "TokenUsage",
    "ask",
    "ask_sync",
    "build_agent",
    "render_system_prompt",
    "require_interactive_reviewer",
]
