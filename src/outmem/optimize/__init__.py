"""Retrieval tuning — lego blocks, a benchmark, and an agent optimizer.

Two loops, one of them here:

* **Config-space (this package, user-facing, safe):** compose shipped
  retrieval blocks via :class:`RetrievalConfig`, score them on a
  provenance-labelled :class:`QuestionBank` with :func:`evaluate`, and
  let :func:`optimize_retrieval` drive an agent to find the best config
  for *your* wiki. No code is modified.
* **Code-space (maintainer-side, not here):** an agent writes *new*
  blocks, gated by tests + the benchmark, opening PRs. See ``improve.md``
  and ``.github/workflows/autoresearch.yml``.

Generation/optimization use ``pydantic_ai`` via lazy imports, so the
deterministic core (blocks + bench + the bank JSON contract) works
without the ``agent`` extra.
"""

from outmem.optimize.bench import QuestionResult, Scorecard, evaluate
from outmem.optimize.blocks import (
    RetrievalConfig,
    RetrievalResult,
    Retriever,
    build_retriever,
)
from outmem.optimize.dataset import (
    Question,
    QuestionBank,
    generate_bank,
    harvest_unanswerable,
)
from outmem.optimize.optimizer import OptimizeResult, optimize_retrieval

__all__ = [
    "OptimizeResult",
    "Question",
    "QuestionBank",
    "QuestionResult",
    "RetrievalConfig",
    "RetrievalResult",
    "Retriever",
    "Scorecard",
    "build_retriever",
    "evaluate",
    "generate_bank",
    "harvest_unanswerable",
    "optimize_retrieval",
]
