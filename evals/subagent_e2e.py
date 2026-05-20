"""End-to-end eval: outer agent delegates to a wiki SUBAGENT, via tool.

**Not** a pytest test — costs money (real Anthropic calls, ~$0.10-0.20
per run) and is run intentionally, not as part of the regular test
suite. Invoke with::

    ANTHROPIC_API_KEY=... python -m evals.subagent_e2e

The flow:

1. Builds a temp wiki and seeds it with one page (Acme's pricing).
2. Constructs an outer PydanticAI agent whose only tool is
   ``consult_wiki`` — a thin wrapper around
   :func:`outmem.agent.ask_sync`. The outer agent's system prompt
   describes WHEN to use the tool but never how the wiki works.
3. Runs two scenarios:
   - *In-scope question* the wiki can answer. Verifies the outer
     agent delegates, the wiki commits (mandatory writeback), the
     response reflects the wiki content, and no outmem-internal
     vocabulary leaks through the tool boundary.
   - *No-answer question* outside the wiki's coverage. Verifies
     the subagent still commits (typically an ``append_log``) and
     the response signals absence rather than hallucinating.
4. Cleans up the temp wiki regardless of outcome.

Exits 0 on full pass, 1 on any failure. Prints per-scenario status
to stderr so output redirects don't lose the human signal.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

# Outmem-internal vocabulary that should NEVER appear in the outer
# agent's surface. If any of these leak through, the wiki agent has
# talked about its tooling — violating the encapsulation under test.
_OUTMEM_INTERNAL_TERMS = (
    "write_page",
    "extend_page",
    "append_log",
    "search_wiki",
    "list_pages",
    "find_backlinks",
    "page_history",
    "topic_evolution",
    "provenance",
    "AGENTS.md",
    "wiki/index.md",
)

_OUTER_SYSTEM_PROMPT = (
    "You're a helpful general assistant. When the user asks a question "
    "that might be answered by the team's documented knowledge, call "
    "the `consult_wiki` tool with their question. Use the tool's reply "
    "as the basis of your answer. For questions clearly outside that "
    "scope (general world knowledge, math, etc.), answer directly."
)


def _seed_wiki(root: Path):
    """Init a tmp wiki and write one seed page about Acme's pricing."""
    from outmem.store import WikiStore

    store = WikiStore.init(root)
    store.write_page(
        "acme-pricing",
        title="Acme pricing",
        body=(
            "Acme has a custom pricing arrangement: cost-plus 28%, "
            "lower than the standard 35% we use for other clients.\n"
        ),
        provenance=["raw/acme-msa.md"],
        tags=["pricing", "acme"],
    )
    return store


def _make_consult_wiki(store) -> Callable[[str], str]:
    """Tool factory: wraps :func:`ask_sync` as a PydanticAI-attachable function.

    The docstring is what the OUTER agent sees when deciding whether
    to use the tool. Deliberately neutral — describes WHEN, not HOW.
    """
    from outmem.agent import ask_sync

    def consult_wiki(question: str) -> str:
        """Ask the team's documented knowledge base about a question.

        Use this when the user asks something that might be in our
        internal docs — policies, pricing, customer history, decisions
        made in past meetings, etc. Returns the knowledge base's
        synthesised answer or a clear "no record" if it has nothing
        on the topic.
        """
        result = ask_sync(store, query=question, push=False, pull=False)
        return result.response

    return consult_wiki


def _build_outer_agent(store):
    from pydantic_ai import Agent

    return Agent(
        "anthropic:claude-sonnet-4-6",
        tools=[_make_consult_wiki(store)],
        system_prompt=_OUTER_SYSTEM_PROMPT,
    )


def _consult_wiki_was_called(result: Any) -> bool:
    from pydantic_ai.messages import ToolCallPart

    for msg in result.all_messages():
        for part in getattr(msg, "parts", []):
            if isinstance(part, ToolCallPart) and part.tool_name == "consult_wiki":
                return True
    return False


def _check_no_leakage(text: str) -> list[str]:
    """Return any outmem-internal terms that leaked into ``text``."""
    return [t for t in _OUTMEM_INTERNAL_TERMS if t in text]


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def scenario_in_scope_question(store) -> list[str]:
    """In-scope question → outer delegates → wiki answers + commits.

    Returns a list of failure strings (empty = pass).
    """
    failures: list[str] = []
    head_before = store.head()

    outer = _build_outer_agent(store)
    result = outer.run_sync("What do we know about Acme's pricing?")

    if not _consult_wiki_was_called(result):
        failures.append("outer agent did not delegate to consult_wiki")

    if store.head() == head_before:
        failures.append(
            "wiki subagent did not commit — mandatory writeback violated"
        )

    leaked = _check_no_leakage(result.output)
    if leaked:
        failures.append(f"outmem internal terms leaked into response: {leaked}")

    response_lower = result.output.lower()
    if "28%" not in result.output and "cost-plus" not in response_lower:
        failures.append(
            f"outer response doesn't reflect wiki content: {result.output[:200]}"
        )

    return failures


def scenario_no_answer_in_wiki(store) -> list[str]:
    """Out-of-scope question → subagent still commits, response signals absence.

    Returns a list of failure strings (empty = pass).
    """
    failures: list[str] = []
    head_before = store.head()

    outer = _build_outer_agent(store)
    result = outer.run_sync(
        "What's our pricing arrangement with BlueSky Corp?"
    )

    if store.head() == head_before:
        failures.append(
            "wiki subagent did not commit — mandatory writeback violated "
            "(typically an append_log for an unanswered query)"
        )

    leaked = _check_no_leakage(result.output)
    if leaked:
        failures.append(f"outmem internal terms leaked into response: {leaked}")

    absence_signals = (
        "no information",
        "not aware",
        "don't have",
        "no record",
        "not found",
        "no entry",
        "no documented",
        "doesn't appear",
        "unable to find",
        "no data",
    )
    response_lower = result.output.lower()
    if not any(s in response_lower for s in absence_signals):
        failures.append(
            f"expected absence signal in response, got: {result.output[:300]}"
        )

    return failures


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _run_scenario(name: str, fn: Callable[[Any], list[str]], store) -> bool:
    """Run one scenario, print status to stderr, return True iff passed."""
    print(f"[ ... ] {name}", file=sys.stderr, flush=True)
    try:
        failures = fn(store)
    except Exception as exc:
        print(f"[ERROR] {name}: {exc}", file=sys.stderr, flush=True)
        return False
    if failures:
        print(f"[FAIL] {name}", file=sys.stderr, flush=True)
        for f in failures:
            print(f"         {f}", file=sys.stderr, flush=True)
        return False
    print(f"[ PASS ] {name}", file=sys.stderr, flush=True)
    return True


def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "ANTHROPIC_API_KEY is not set; this eval calls Anthropic.",
            file=sys.stderr,
        )
        return 2

    workspace = Path(tempfile.mkdtemp(prefix="outmem-subagent-eval-"))
    try:
        store = _seed_wiki(workspace / "wiki")
        passed = 0
        failed = 0
        for name, fn in (
            ("in-scope question reaches wiki + commits", scenario_in_scope_question),
            ("no-answer question still commits + signals absence", scenario_no_answer_in_wiki),
        ):
            if _run_scenario(name, fn, store):
                passed += 1
            else:
                failed += 1
        print(
            f"\n{passed} passed, {failed} failed (workspace: {workspace})",
            file=sys.stderr,
        )
        return 0 if failed == 0 else 1
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
