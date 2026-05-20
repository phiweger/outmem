"""Two humans pushed contradicting framings; agent must name both
authors and describe the divergence rather than silently merging."""

from __future__ import annotations

from evals import EvalRun, eval_case


@eval_case(
    wiki="multi-author-divergence",
    query=(
        "What's the current state of the pricing formula? I heard there's "
        "disagreement about a Q3 bump — what's going on?"
    ),
    description="Alice committed the formula at 35%; Bob's log proposes 40%. "
    "Agent should surface both authors and the unresolved disagreement.",
    # Steering rendering is the whole point of this case — without it
    # the agent has to derive authorship from `page_history` /
    # `topic_evolution` instead of reading recent human commits
    # straight from PHASE-1 of its system prompt (spec §6).
    include_steering=True,
)
def case_multi_author_divergence(r: EvalRun) -> None:
    # Don't pin a specific tool — either `read_page` (then `search_wiki`
    # for the log) or `topic_evolution` is a legitimate path. The
    # judges grade the *outcome*: did both authors get named?
    r.expect_commit()

    r.judge(
        "answer names Alice in connection with the cost-plus 35% formula. "
        "ANY of these framings count as 'naming Alice': "
        "(a) Alice is the committer/author of the pricing-formula page, "
        "(b) Alice committed the 35% rate / wants to keep the 35% rate, "
        "(c) Alice wants to hold at 35% / wants to wait, "
        "(d) any other attribution that connects Alice to the current "
        "35% position. The point of the criterion is 'did the agent "
        "surface Alice as a participant in this disagreement?' — yes "
        "if any of the above is present."
    )
    r.judge("answer names Bob as the author proposing the Q3 40% bump")
    r.judge(
        "answer characterises the situation as an unresolved disagreement "
        "between the two authors, not a settled decision"
    )
