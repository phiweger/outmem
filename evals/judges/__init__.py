"""Trace + LLM judges used by :mod:`evals.harness`.

* :mod:`.trace` — deterministic helpers re-exported through
  :class:`evals.EvalRun`. Kept here as a module for future expansion
  (e.g. graph assertions about tool call sequences).
* :mod:`.llm_judge` — the structured PydanticAI agent that grades
  individual criteria against the agent's final response.
"""

from __future__ import annotations
