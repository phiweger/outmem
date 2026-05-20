"""raw/ contradicts wiki/: surface the disagreement, don't average."""

from __future__ import annotations

from evals import EvalRun, eval_case


@eval_case(
    wiki="raw-contradicts-wiki",
    query=(
        "What is the maximum volume discount we offer to top-tier customers? "
        "Make sure to check raw/ as well, not just the wiki page."
    ),
    description="wiki/discounts.md says 15%, raw/sales-policy-2025.md says 20%. "
    "Agent must surface the contradiction in its answer.",
)
def case_raw_contradicts_wiki(r: EvalRun) -> None:
    # The agent should read both.
    r.expect_tool_called("read_page", slug="discounts")
    # The raw scan is what surfaces the discrepancy.
    r.expect_tool_called("search_wiki", scope="raw")
    r.expect_commit()

    r.judge(
        "answer explicitly mentions BOTH 15% (wiki) and 20% (raw policy) figures"
    )
    r.judge(
        "answer makes the wiki-vs-raw disagreement VISIBLE to the reader. "
        "Any of these framings count: "
        "(a) flags it as a contradiction; "
        "(b) calls the wiki stale or out of date; "
        "(c) describes the agent's action of updating the wiki to match "
        "the raw policy (the update + the disagreement are both in the "
        "response — silent absorption would be a failure, but explicit "
        "resolution is fine). "
        "The point is: the reader must be able to see that wiki and raw "
        "disagreed. Pure silent acceptance of one figure is the failure "
        "mode this criterion is testing for."
    )
