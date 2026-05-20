"""Trace assertions are implemented as methods on :class:`evals.EvalRun`.

This module exists as a documentation seam for now — extend it with
graph-style assertions (e.g. "search_wiki MUST come before read_page")
once we have a real need.
"""

from __future__ import annotations
