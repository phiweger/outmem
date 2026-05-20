"""EXPANSION: agent walks history via topic_evolution."""

from __future__ import annotations

from evals import EvalRun, eval_case


@eval_case(
    wiki="temporal-evolution",
    query=(
        "Walk through how our pricing formula has evolved over time — what "
        "did it used to be and what is it now? Show your work from git history."
    ),
    description="Open-ended history walk: agent must reach for topic_evolution "
    "and read the diff stream, not just the current page.",
)
def case_pricing_history(r: EvalRun) -> None:
    r.expect_tool_called("topic_evolution")
    r.expect_commit(subject_matches=r"^(extend|log):")

    r.judge("answer says the pricing formula started at cost-plus 30%")
    r.judge("answer says the formula was later raised to cost-plus 35%")
    r.judge(
        "answer attributes the timing using the log entries "
        "(2025 set, 2026-Q1 raise) rather than guessing"
    )
