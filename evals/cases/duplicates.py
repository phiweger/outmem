"""semantic on: agent should recognise a near-duplicate via find_similar
and choose ``extend_page`` over ``write_page``."""

from __future__ import annotations

from evals import EvalRun, eval_case


@eval_case(
    wiki="duplicate-trap",
    query=(
        "Please add a new wiki page summarising standard ivermectin and "
        "amoxicillin dosing for cats, drawing from "
        "`wiki/sources/veterinary/drugs.md`."
    ),
    semantic=True,
    description="cat-drug-dosages already exists with the same content. "
    "Agent should call find_similar and extend the existing page "
    "rather than creating a parallel slug.",
)
def case_duplicate_trap(r: EvalRun) -> None:
    # Originally this case required `find_similar` in the trace. In
    # practice, with a 1-page wiki the agent can solve the duplicate
    # problem just as well via `list_pages` + `read_page` — which is a
    # legitimate path, not a failure mode. The OUTCOME (no parallel
    # slug created) is what matters; let the judge handle it. We
    # still check that `write_page` was NOT called and that the
    # writeback was either an extend on the canonical slug or a log
    # entry.
    r.expect_no_tool_called("write_page")
    r.expect_commit(subject_matches=r"^(extend: cat-drug-dosages|log:)")

    r.judge(
        "answer indicates the agent reused or extended an existing wiki "
        "page (cat-drug-dosages) rather than creating a new one"
    )
