"""Ingestion-shape case: a stale wikilink to a slug that doesn't exist.

Tests the agent's behaviour when the wiki graph is broken: it must not
crash and should surface the dangling link in its answer rather than
silently pretending the target page exists.
"""

from __future__ import annotations

from evals import EvalRun, eval_case


@eval_case(
    wiki="stale-wikilink",
    query=(
        "Read the pricing-formula page and tell me what it links to. Is "
        "every link valid? If not, flag the broken ones."
    ),
    description="pricing-formula links [[discounts]] (no such page) and "
    "[[acme-msa]] (exists). Agent should flag the dangling link.",
)
def case_stale_wikilink(r: EvalRun) -> None:
    r.expect_tool_called("read_page", slug="pricing-formula")
    r.expect_commit()

    r.judge("answer notes that [[discounts]] points to a page that does NOT exist")
    r.judge("answer notes that [[acme-msa]] is a valid link OR that acme-msa exists")
    r.judge(
        "answer does not invent contents for the missing discounts page"
    )
