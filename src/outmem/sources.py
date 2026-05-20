"""Source management — ``wiki/sources/`` directory + ``.sources.db`` registry.

Sources are the raw material the agent ingests into wiki pages. They
live under ``wiki/sources/`` (tracked in git, alongside the compiled
pages they produce) so the audit trail is self-contained: every
page's ``provenance:`` field can cite ``sources/<rel-path>`` + a
sha256, and ``outmem lint`` can verify those references resolve.

Supported source types are flat non-binary text: ``.md``, ``.txt``,
``.csv``, ``.json``, ``.mmd`` (mermaid), ``.yaml`` / ``.yml``. The
LLM reads them as plain text and decides how to interpret structure.

Registry format
---------------

``.sources.db`` is a SQLite file with two tables:

.. code-block:: sql

    CREATE TABLE sources (
        rel_path      TEXT PRIMARY KEY,
        sha256        TEXT NOT NULL,
        size_bytes    INTEGER NOT NULL,
        registered_at TEXT NOT NULL
    );

    CREATE TABLE ingestions (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        rel_path       TEXT NOT NULL REFERENCES sources(rel_path) ON DELETE CASCADE,
        timestamp      TEXT NOT NULL,
        prompt         TEXT,
        pages_touched  TEXT NOT NULL  -- JSON array
    );

SQLite (rollback-journal mode + ``busy_timeout``) makes two concurrent
``outmem ingest`` runs against the same wiki safe — writers serialise
at the OS file lock level instead of racing on a JSON read-modify-write.
The rollback-journal choice (not WAL) means the main ``.sources.db``
file always reflects committed state, so it's a normal git-tracked
binary with no ``-wal`` / ``-shm`` companion files.

Layout: every source lives at
``<sources_dir>/[<into>/]<sha256[:12]>/<filename>``. The hash
directory makes the layout collision-free and dedupes
identical-content re-ingests.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from outmem._sqlite import connect as _sqlite_connect
from outmem._time import format_iso_z, parse_iso_z, utc_now
from outmem.exceptions import OutmemError

SOURCES_DIR = "sources"
REGISTRY_FILENAME = ".sources.db"

ALLOWED_EXTENSIONS = frozenset({".md", ".txt", ".csv", ".json", ".mmd", ".yaml", ".yml"})

# 12 hex chars = 48 bits, plenty of headroom against accidental
# collision across realistic source corpora (millions of files).
SHA_PREFIX_LEN = 12


@dataclass(frozen=True)
class IngestionRecord:
    """One ingestion of a source with its associated prompt + outputs."""

    timestamp: datetime
    prompt: str | None
    pages_touched: tuple[str, ...]


@dataclass
class SourceEntry:
    """One row in the registry — a single registered source file."""

    rel_path: str  # relative to wiki/sources/, e.g. "veterinary/<sha>/drugs.md"
    sha256: str
    registered_at: datetime
    size_bytes: int
    ingestions: list[IngestionRecord] = field(default_factory=list)


@dataclass
class SourceRegistry:
    """SQLite-backed view of the ``wiki/sources/.sources.db`` registry.

    Construct via :meth:`load`. Mutations through :meth:`register` /
    :meth:`record_ingestion` commit immediately and keep
    :attr:`entries` (the in-memory snapshot) in lockstep.
    """

    sources_dir: Path
    entries: dict[str, SourceEntry] = field(default_factory=dict)
    _con: sqlite3.Connection | None = field(default=None, repr=False)

    @classmethod
    def load(cls, sources_dir: Path) -> SourceRegistry:
        """Open / create the registry DB and return an in-memory snapshot."""
        sources_dir.mkdir(parents=True, exist_ok=True)
        con = _open_registry(sources_dir / REGISTRY_FILENAME)
        entries = _read_all_entries(con)
        return cls(sources_dir=sources_dir, entries=entries, _con=con)

    def close(self) -> None:
        """Close the underlying SQLite connection. Idempotent."""
        if self._con is not None:
            self._con.close()
            self._con = None

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def register(
        self,
        rel_path: str,
        *,
        sha256: str,
        size_bytes: int,
        when: datetime | None = None,
    ) -> SourceEntry:
        """Add or refresh an entry. Returns the canonical entry.

        Re-registering with the same hash returns the existing entry
        unchanged. A new hash refreshes the row and clears its
        ingestion chain — old entries belonged to the old content.
        """
        existing = self.entries.get(rel_path)
        if existing and existing.sha256 == sha256:
            return existing

        ts = when.replace(microsecond=0) if when else utc_now()
        entry = SourceEntry(
            rel_path=rel_path,
            sha256=sha256,
            registered_at=ts,
            size_bytes=size_bytes,
            ingestions=[],
        )
        con = self._connection()
        with con:
            con.execute(
                "INSERT OR REPLACE INTO sources "
                "(rel_path, sha256, size_bytes, registered_at) "
                "VALUES (?, ?, ?, ?)",
                (rel_path, sha256, size_bytes, format_iso_z(ts)),
            )
            # INSERT OR REPLACE preserves the row, so FK ON DELETE
            # CASCADE doesn't fire — clear ingestions explicitly when
            # the sha rolled over.
            if existing and existing.sha256 != sha256:
                con.execute("DELETE FROM ingestions WHERE rel_path = ?", (rel_path,))
        self.entries[rel_path] = entry
        return entry

    def record_ingestion(
        self,
        rel_path: str,
        *,
        prompt: str | None,
        pages_touched: Iterable[str],
        when: datetime | None = None,
    ) -> IngestionRecord:
        if rel_path not in self.entries:
            raise OutmemError(
                f"cannot record ingestion: {rel_path!r} not registered. "
                "Call register() first."
            )
        record = IngestionRecord(
            timestamp=when.replace(microsecond=0) if when else utc_now(),
            prompt=prompt,
            pages_touched=tuple(pages_touched),
        )
        con = self._connection()
        with con:
            con.execute(
                "INSERT INTO ingestions (rel_path, timestamp, prompt, pages_touched) "
                "VALUES (?, ?, ?, ?)",
                (
                    rel_path,
                    format_iso_z(record.timestamp),
                    record.prompt,
                    json.dumps(list(record.pages_touched)),
                ),
            )
        self.entries[rel_path].ingestions.append(record)
        return record

    # ------------------------------------------------------------------
    # Read-only helpers
    # ------------------------------------------------------------------

    def get(self, rel_path: str) -> SourceEntry | None:
        return self.entries.get(rel_path)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _connection(self) -> sqlite3.Connection:
        if self._con is None:
            self._con = _open_registry(self.sources_dir / REGISTRY_FILENAME)
        return self._con


# ---------------------------------------------------------------------------
# File-level helpers
# ---------------------------------------------------------------------------


def compute_sha256(path: Path, *, chunk: int = 65536) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            data = f.read(chunk)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def is_allowed_source(path: Path) -> bool:
    return path.suffix.lower() in ALLOWED_EXTENSIONS


def copy_source(
    source: Path,
    sources_dir: Path,
    *,
    into_subdir: str | None = None,
    rename: str | None = None,
) -> tuple[Path, str]:
    """Copy ``source`` into a content-addressed layout under ``sources_dir``.

    The destination is::

        <sources_dir> / [<into_subdir> /] <sha256[:12]> / <filename>

    where ``filename`` is the source's basename (or ``rename`` if
    supplied). The short-sha directory makes the layout
    collision-free: two source files with the same name but
    different content land under different hash dirs, and the same
    file ingested twice deduplicates to the same dir.

    Returns ``(destination_path, rel_path)`` where ``rel_path`` is
    relative to ``sources_dir`` and suitable for the registry key /
    ``provenance:`` citations
    (e.g. ``"veterinary/d72224543518/drugs.md"``).

    Raises :class:`OutmemError` for binary / disallowed file types or
    unsafe path components.
    """
    if not source.exists() or not source.is_file():
        raise OutmemError(f"source not found: {source}")
    if not is_allowed_source(source):
        raise OutmemError(
            f"source has disallowed extension {source.suffix!r}; "
            f"allowed: {sorted(ALLOWED_EXTENSIONS)}"
        )

    filename = rename or source.name
    if "/" in filename or ".." in filename:
        raise OutmemError(f"unsafe destination filename: {filename!r}")

    if into_subdir and (
        into_subdir.startswith("/") or ".." in into_subdir.split("/")
    ):
        raise OutmemError(f"unsafe into_subdir: {into_subdir!r}")

    sha = compute_sha256(source)
    short = sha[:SHA_PREFIX_LEN]

    parent = sources_dir / into_subdir if into_subdir else sources_dir
    hash_dir = parent / short
    hash_dir.mkdir(parents=True, exist_ok=True)

    dest = hash_dir / filename
    rel_path = str(dest.relative_to(sources_dir))

    # Same content → same hash dir → idempotent.
    if not dest.exists():
        shutil.copy2(source, dest)
    return dest, rel_path


def read_source_text(
    sources_dir: Path,
    rel_path: str,
    *,
    max_chars: int,
) -> str:
    """Read a source file as text, capped at ``max_chars``.

    The cap exists so an oversize source doesn't blow up the agent's
    context when returned via the ``read_source`` PydanticAI tool.
    Configurable via ``config.yaml``'s ``sources.max_chars``.
    """
    path = sources_dir / rel_path
    try:
        path.resolve().relative_to(sources_dir.resolve())
    except ValueError as exc:
        raise OutmemError(f"source path escapes sources dir: {rel_path!r}") from exc
    if not path.exists():
        raise OutmemError(f"no such source: {rel_path}")
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) > max_chars:
        return text[:max_chars] + f"\n\n[truncated — file is {len(text)} chars, cap {max_chars}]"
    return text


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _open_registry(db_path: Path) -> sqlite3.Connection:
    """Open / create the registry SQLite DB with the right PRAGMAs.

    Rollback-journal mode (the default) keeps the main DB file in
    sync with committed state on every transaction — the file is a
    plain git-trackable binary, no ``-wal`` / ``-shm`` companions.
    ``busy_timeout`` lets concurrent writers block-and-retry instead
    of erroring on contention; ``foreign_keys=ON`` enforces the FK
    from ``ingestions`` to ``sources``.
    """
    con = _sqlite_connect(db_path)
    con.execute("PRAGMA busy_timeout=5000")
    con.execute("PRAGMA foreign_keys=ON")
    _init_schema(con)
    return con


def _init_schema(con: sqlite3.Connection) -> None:
    cur = con.cursor()
    cur.execute(
        "CREATE TABLE IF NOT EXISTS sources ("
        " rel_path TEXT PRIMARY KEY,"
        " sha256 TEXT NOT NULL,"
        " size_bytes INTEGER NOT NULL,"
        " registered_at TEXT NOT NULL)"
    )
    cur.execute(
        "CREATE TABLE IF NOT EXISTS ingestions ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " rel_path TEXT NOT NULL"
        " REFERENCES sources(rel_path) ON DELETE CASCADE,"
        " timestamp TEXT NOT NULL,"
        " prompt TEXT,"
        " pages_touched TEXT NOT NULL)"
    )
    cur.execute("CREATE INDEX IF NOT EXISTS ingestions_rel_path ON ingestions(rel_path)")
    con.commit()


def _read_all_entries(con: sqlite3.Connection) -> dict[str, SourceEntry]:
    cur = con.cursor()
    entries: dict[str, SourceEntry] = {}
    for row in cur.execute(
        "SELECT rel_path, sha256, size_bytes, registered_at FROM sources "
        "ORDER BY rel_path"
    ).fetchall():
        entries[row["rel_path"]] = SourceEntry(
            rel_path=row["rel_path"],
            sha256=row["sha256"],
            registered_at=parse_iso_z(row["registered_at"]),
            size_bytes=int(row["size_bytes"]),
            ingestions=[],
        )
    for row in cur.execute(
        "SELECT rel_path, timestamp, prompt, pages_touched FROM ingestions "
        "ORDER BY rel_path, id"
    ).fetchall():
        if row["rel_path"] not in entries:
            continue
        entries[row["rel_path"]].ingestions.append(
            IngestionRecord(
                timestamp=parse_iso_z(row["timestamp"]),
                prompt=row["prompt"],
                pages_touched=tuple(json.loads(row["pages_touched"]) or ()),
            )
        )
    return entries
