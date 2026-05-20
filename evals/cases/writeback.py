"""Mandatory writeback: a query with no answer must still produce a
commit — typically an ``append_log`` with topic ``no-new-compaction``."""

from __future__ import annotations

from evals import EvalRun, eval_case


@eval_case(
    wiki="no-match-query",
    query="What is our deployment policy for the inference cluster?",
    description="No relevant material exists. Agent must NOT fabricate; "
    "it must satisfy mandatory writeback via append_log.",
)
def case_no_match_falls_back_to_log(r: EvalRun) -> None:
    # The agent should at least try to look first.
    r.expect_tool_called("search_wiki")
    # Mandatory writeback: at least one commit; should NOT be a wiki write.
    r.expect_commit(subject_matches=r"^log:")
    r.expect_tool_called("append_log")

    r.judge(
        "answer says the agent could not find information about the "
        "deployment policy in the wiki or raw material"
    )
    r.judge(
        "answer does NOT invent a deployment policy or fabricate "
        "specific deployment details that aren't in the source"
    )
