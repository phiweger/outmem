"""Loader for the top-level repo ``config.yaml`` (eval-suite config).

This is the **repo-level** config — distinct from a wiki's own
``config.yaml``. Lives at the repo root (sibling of ``pyproject.toml``
and the ``evals/`` directory).

Reads the ``evals:`` block:

.. code-block:: yaml

    evals:
      agent_model: anthropic:claude-sonnet-4-6   # optional
      judge_model: anthropic:claude-sonnet-4-6 # optional

Resolution chain (highest priority first), enforced in
:mod:`evals.run`:

1. CLI flag (``--model``, ``--judge-model``)
2. Environment variable (``OUTMEM_MODEL`` for the agent)
3. Top-level ``config.yaml`` (this loader)
4. Built-in default (e.g.
   :data:`evals.judges.llm_judge.DEFAULT_JUDGE_MODEL`)

Missing file → empty :class:`EvalsConfig`. Malformed YAML logs a
warning and returns empty config — matches the lenient posture of
:mod:`outmem.config`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

log = logging.getLogger(__name__)

REPO_CONFIG_FILENAME = "config.yaml"


@dataclass(frozen=True)
class EvalsConfig:
    """Resolved eval-suite settings from the top-level ``config.yaml``."""

    agent_model: str | None = None
    judge_model: str | None = None


def repo_root() -> Path:
    """Best-effort path to the repo root.

    The evals package lives at ``<repo>/evals/``, so the parent of this
    file's directory is the repo root. Honoured even when the package
    is installed editable (``pip install -e .``) because the source
    tree is the install location in that case.
    """
    return Path(__file__).resolve().parent.parent


def load_repo_config(*, repo: Path | None = None) -> EvalsConfig:
    """Parse the top-level ``config.yaml``; return defaults on miss."""
    path = (repo or repo_root()) / REPO_CONFIG_FILENAME
    if not path.exists():
        return EvalsConfig()
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        log.warning("Malformed %s, ignoring: %s", path, exc)
        return EvalsConfig()
    if not isinstance(raw, dict):
        return EvalsConfig()
    block = raw.get("evals")
    if not isinstance(block, dict):
        return EvalsConfig()
    return EvalsConfig(
        agent_model=_str_or_none(block.get("agent_model")),
        judge_model=_str_or_none(block.get("judge_model")),
    )


def _str_or_none(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None
