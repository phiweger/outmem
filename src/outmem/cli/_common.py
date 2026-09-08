"""Helpers shared by the command modules under :mod:`outmem.cli`.

Small on purpose: what the top-level command file and the command
groups split out of it both need, and nothing else. Anything a single
command uses stays with that command.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

from outmem.store import AgentIdentity

TIMESTAMP_FMT = "%H:%M:%S"


def agent_identity() -> AgentIdentity:
    """Resolve the agent identity from env, falling back to defaults.

    ``OUTMEM_AGENT_NAME`` and ``OUTMEM_AGENT_EMAIL`` override the
    defaults so the same install can serve multiple wikis with different
    agent identities (one process per wiki).
    """
    name = os.environ.get("OUTMEM_AGENT_NAME")
    email = os.environ.get("OUTMEM_AGENT_EMAIL")
    if name and email:
        return AgentIdentity(name=name, email=email)
    return AgentIdentity()


def status(msg: str) -> None:
    """Print a status / progress line to stdout, prefixed with a local
    ``[HH:MM:SS]`` timestamp. Reserve for human-readable progress messages;
    keep raw :func:`print` for structured output (commit SHAs, grep hits,
    paths) that downstream tools might pipe."""
    print(f"[{datetime.now().strftime(TIMESTAMP_FMT)}] {msg}")


def base_root(args: argparse.Namespace) -> Path:
    """The directory a command starts from: ``--root``, else
    ``$OUTMEM_PATH``, else the current directory.

    ``--root`` uses ``argparse.SUPPRESS`` as its default so a nested
    subparser cannot clobber a value given at the outer level, which
    means it is *absent* from ``args`` rather than ``None`` when unset —
    hence ``getattr``.

    This is the one place that order is written down. Whether the result
    is then a wiki or a multi-wiki repository is the caller's business.
    """
    root = getattr(args, "root", None)
    if root:
        return Path(root).expanduser()
    env = os.environ.get("OUTMEM_PATH")
    if env:
        return Path(env).expanduser()
    return Path.cwd()
