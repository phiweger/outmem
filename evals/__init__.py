"""End-to-end behavioural evaluations for outmem.

Separate from ``tests/`` — these call **real** LLMs to verify that the
agent does the right thing on realistic wiki scenarios. Two judging
modes:

* **Trace assertions** (deterministic, free given the run already
  happened): the agent must have called certain tools and produced
  certain commit subjects. See :mod:`evals.judges.trace`.
* **LLM judge** (probabilistic, cents per call): a separate
  :class:`pydantic_ai.Agent` graded with structured output decides
  whether the agent's final response satisfies a natural-language
  criterion. See :mod:`evals.judges.llm_judge`.

Default judge: ``anthropic:claude-sonnet-4-6``. Override per-run with
``--judge-model``.

Run::

    python -m evals.run                       # all cases, trace + judge
    python -m evals.run --no-judge            # trace-only (still calls the agent's LLM)
    python -m evals.run --case duplicate-trap # single case
    python -m evals.run --json out.json       # machine-readable report

Costs roughly $0.30-1.00 per full run with the defaults — gate behind
nightly / release-tag CI, not per-PR. Requires ``ANTHROPIC_API_KEY``
(or whichever provider ``OUTMEM_MODEL`` selects) in the environment.
"""

from __future__ import annotations

from evals.harness import EvalCase, EvalRun, eval_case, run_case

__all__ = ["EvalCase", "EvalRun", "eval_case", "run_case"]
