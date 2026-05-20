"""Internal SQLite helpers — single source for the connection idiom.

outmem has two SQLite-backed stores (``.sources.db`` registry,
``.vectors.db`` semantic index). Both want the same boilerplate:
ensure the parent dir exists, connect with ``check_same_thread=False``
(PydanticAI dispatches async work across threads), and use
:class:`sqlite3.Row` so call sites can use dict access. PRAGMAs and
schema initialisation stay with each caller — they differ enough that
abstracting them would add more friction than it saves.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


def connect(db_path: Path) -> sqlite3.Connection:
    """Open / create a SQLite DB at ``db_path`` with outmem's defaults."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path), check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con
