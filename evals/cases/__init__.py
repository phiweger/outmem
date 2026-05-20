"""Eval cases. Importing this package registers every case via the
``@eval_case`` decorator side-effects in each module below.

Each module exports one or more ``case_*`` functions. The naming
convention (``case_<slug>``) maps to the case's stable identifier
(``--case <slug>``) — see :func:`evals.harness.eval_case`.
"""

from __future__ import annotations

from evals.cases import (
    approval,
    contradictions,
    convergence,
    duplicates,
    expansion,
    ingestion,
    multi_author,
    writeback,
)

__all__ = [
    "approval",
    "contradictions",
    "convergence",
    "duplicates",
    "expansion",
    "ingestion",
    "multi_author",
    "writeback",
]
