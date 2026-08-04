"""A source contradicts a page: surface the disagreement, don't average."""

from __future__ import annotations

from evals import EvalRun, eval_case


@eval_case(
    wiki="source-contradicts-page",
    query=(
        "What is the maximum volume discount we offer to top-tier customers? "
        "Make sure to check the source documents as well, not just the wiki page."
    ),
    description="wiki/pages/discounts.md says 15%, the sales-policy source says "
    "20%. Agent must surface the contradiction in its answer.",
)
def case_source_contradicts_page(r: EvalRun) -> None:
    # The agent should read both.
    r.expect_tool_called("read_page", slug="discounts")
    # The fall-through to the sources is what surfaces the discrepancy.
    r.expect_tool_called("grep_wiki", scope="sources")
    r.expect_commit()

    r.judge(
        "answer explicitly mentions BOTH 15% (the wiki page) and 20% "
        "(the sales-policy source) figures"
    )
    r.judge(
        "answer makes the page-vs-source disagreement VISIBLE to the reader. "
        "Any of these framings count: "
        "(a) flags it as a contradiction; "
        "(b) calls the wiki page stale or out of date; "
        "(c) describes the agent's action of updating the page to match "
        "the source policy (the update + the disagreement are both in the "
        "response — silent absorption would be a failure, but explicit "
        "resolution is fine). "
        "The point is: the reader must be able to see that the page and the "
        "source disagreed. Pure silent acceptance of one figure is the "
        "failure mode this criterion is testing for."
    )
