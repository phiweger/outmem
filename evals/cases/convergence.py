"""CONVERGENCE: agent finds a fact in the wiki and cites it."""

from __future__ import annotations

from evals import EvalRun, eval_case


@eval_case(
    wiki="pricing-cost-plus",
    query="What is our standard pricing formula and where does it come from?",
    description="Tier-1 lookup: agent should search the wiki, read the page, "
    "answer with cost-plus 35% citing the Q1 2026 deck.",
)
def case_pricing_lookup(r: EvalRun) -> None:
    # The compiled page exists in wiki/ — agent should converge there.
    r.expect_tool_called("search_wiki", pattern__contains="pricing")
    r.expect_tool_called("read_page", slug="pricing-formula")
    r.expect_commit(subject_matches=r"^(extend|log):")

    r.judge("answer states the pricing formula is cost-plus 35%")
    r.judge(
        "answer attributes the formula to the Q1 2026 pricing deck. "
        "Any ONE of these identifiers counts as attribution because all "
        "three name the SAME source document: "
        "(a) the raw path 'raw/pricing-deck-2026-Q1.md', "
        "(b) the drive path '/shared/Sales/2026-Q1-pricing-deck.pdf', "
        "(c) the phrase 'Q1 2026 pricing deck' (with or without 'slide 2'). "
        "The drive path and the raw path point to the same file — listed "
        "together under `provenance:` in the wiki frontmatter — so "
        "mentioning the drive path alone IS valid source attribution."
    )
    # NB: deliberately not requiring the agent to mention the Acme
    # exception — the query asks for the standard formula + source, not
    # for a full picture. A tight answer is a correct answer.
