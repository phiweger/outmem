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
        registered_at TEXT NOT NULL,
        document_key  TEXT,   -- which document this file is a version of
        superseded_by TEXT,   -- rel_path of the version that replaced it
        origin_path   TEXT    -- where it was ingested from
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

The last three columns arrived with schema v2 and are added in place by
:func:`_migrate` on open, so an existing registry never needs rebuilding.

Layout: every source lives at
``<sources_dir>/[<into>/]<sha256[:12]>/<filename>``. The hash
directory makes the layout collision-free and dedupes
identical-content re-ingests — and is exactly why ``document_key``
exists, since it also makes a *revised* document look unrelated to the
one it replaces.
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

# Bumped only when the sources/ingestions schema changes shape.
SCHEMA_VERSION = 2

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
    document_key: str | None = None
    """Which *document* this file is a version of.

    ``rel_path`` embeds the content hash, so a revised document lands at a
    new path and looks like an unrelated row. ``document_key`` is the
    identity that survives a revision — set explicitly at ingest
    (``--as``), because no derivation can tell "same document, new
    version" from "different document, same filename" in general.
    """
    superseded_by: str | None = None  # rel_path of the version that replaced this
    origin_path: str | None = None
    """Where the file was ingested from.

    Recorded because it is the field that would have disambiguated the
    historical rows and was previously discarded — a pipeline emitting
    ``.../<drug>/output/<hash>/document.md`` puts the distinguishing part
    here and nowhere else.
    """


def normalize_document_key(raw: str) -> str:
    """Canonical form of a document identity.

    A document key names a *document*, not a file, so the two properties
    that change without the document changing are dropped: case, and the
    source file's extension. A pipeline that starts exporting ``.txt``
    where it used to export ``.md`` keeps the same identity — the raw
    path would silently start a new one, and a *silent* break is the one
    failure mode supersession exists to remove.

    Only extensions outmem accepts as sources are stripped, so an
    external identifier survives intact: ``doi/10.1001-jama-2026`` keeps
    its dot because ``.1001-jama-2026`` is not a source type.

    Note the direction this moves in — dropping the extension makes
    *collisions* more likely, and a collision is a refusal, never a
    merge. ``report.md`` and ``report.csv`` under one ``--into`` now stop
    and ask instead of quietly becoming unrelated documents.
    """
    key = "/".join(part for part in raw.strip().lower().split("/") if part)
    if not key:
        raise OutmemError(f"{raw!r} is not a usable document identity (it is empty).")
    head, dot, ext = key.rpartition(".")
    if dot and head and f".{ext}" in ALLOWED_EXTENSIONS:
        key = head
    return key


def derive_document_key(rel_path: str, sha256: str) -> str | None:
    """The identity a content-addressed path implies, or None.

    The sha segment is *verified* against the row's own hash rather than
    pattern-matched, so a directory legitimately named like a hash can't
    be mistaken for one. Returns None when the path has no verifiable sha
    segment (a pre-hash-dir flat row — whose ``rel_path`` already is the
    candidate).

    This is a **candidate**, never an identity. The derivation reads the
    wiki path, which has already discarded the distinguishing part: a
    pipeline emitting ``document.md`` per drug produces the same
    candidate for every drug — the collision the hash directory exists to
    survive. ``origin_path`` is where that part survives, which is why
    the refusal quotes it.
    """
    parts = rel_path.split("/")
    if len(parts) >= 2 and parts[-2] == sha256[:SHA_PREFIX_LEN]:
        return normalize_document_key("/".join([*parts[:-2], parts[-1]]))
    return None


def distinguishing_segment(origin: str | None, other: str | None) -> str | None:
    """The first path segment where two ingest origins diverge.

    When two files derive the same document key, this is the part of
    their provenance that tells them apart — for a
    ``parsed/fachinfo/<drug>/output/<hash>/document.md`` pipeline it is
    the drug, which is exactly the name the operator wants to pass to
    ``--as``. The *first* divergence rather than the last, because that
    is the most general distinguishing segment; deeper ones tend to be
    hashes.

    Returns None when either origin is unknown (a historical row), when
    the paths never diverge before the filename, or when the divergence
    is itself a hash — proposing ``--as fachinfo/a1b2c3d4`` would be
    worse than proposing nothing.
    """
    if not origin or not other:
        return None
    # Different-length origins are the norm; the common prefix is all we
    # can compare, so strict=False is the intent, not an oversight.
    for mine, theirs in zip(
        Path(origin).parts[:-1], Path(other).parts[:-1], strict=False
    ):
        if mine == theirs:
            continue
        stripped = mine.lower().lstrip("0123456789abcdef")
        if not stripped and len(mine) >= 8:
            return None  # a hash directory is not a name
        return mine
    return None


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

    def latest_for(self, document_key: str) -> SourceEntry | None:
        """The current (un-superseded) version of ``document_key``, if any."""
        candidates = [
            e
            for e in self.entries.values()
            if e.document_key == document_key and e.superseded_by is None
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda e: e.registered_at)

    def register(
        self,
        rel_path: str,
        *,
        sha256: str,
        size_bytes: int,
        when: datetime | None = None,
        document_key: str | None = None,
        origin_path: str | None = None,
    ) -> SourceEntry:
        """Add or refresh an entry. Returns the canonical entry.

        Re-registering with the same hash returns the existing entry
        unchanged.

        When ``document_key`` is given and a live version of that document
        already exists, this registration **supersedes** it: the older row
        keeps its file, its sha and its ingestion history, and gains a
        ``superseded_by`` pointer to this one. Nothing is deleted —
        knowing that v1 existed and what was compacted from it is the
        point, and it is what lets ``outmem stale`` find the pages that
        still cite it.

        Note the sha is part of ``rel_path``, so a revised document is
        always a *new row*; the pre-existing ``!= sha256`` refresh branch
        below is only reachable for callers that build paths themselves.
        """
        existing = self.entries.get(rel_path)
        if existing and existing.sha256 == sha256:
            return existing

        ts = when.replace(microsecond=0) if when else utc_now()
        predecessor = self.latest_for(document_key) if document_key else None
        if predecessor is not None and predecessor.rel_path == rel_path:
            predecessor = None  # re-registering the same path, not a new version

        entry = SourceEntry(
            rel_path=rel_path,
            sha256=sha256,
            registered_at=ts,
            size_bytes=size_bytes,
            ingestions=[],
            document_key=document_key,
            origin_path=origin_path,
        )
        con = self._connection()
        with con:
            con.execute(
                "INSERT OR REPLACE INTO sources "
                "(rel_path, sha256, size_bytes, registered_at, document_key, "
                "superseded_by, origin_path) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (rel_path, sha256, size_bytes, format_iso_z(ts), document_key,
                 None, origin_path),
            )
            # INSERT OR REPLACE preserves the row, so FK ON DELETE
            # CASCADE doesn't fire — clear ingestions explicitly when
            # the sha rolled over.
            if existing and existing.sha256 != sha256:
                con.execute("DELETE FROM ingestions WHERE rel_path = ?", (rel_path,))
            if predecessor is not None:
                con.execute(
                    "UPDATE sources SET superseded_by = ? WHERE rel_path = ?",
                    (rel_path, predecessor.rel_path),
                )
        if predecessor is not None:
            self.entries[predecessor.rel_path].superseded_by = rel_path
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
    # A fresh file is at user_version 0, so it takes the same path as an
    # existing one — the migration *is* how the v2 columns get added, and
    # running it on both shapes is what keeps them identical.
    _migrate(con)
    con.commit()


def _migrate(con: sqlite3.Connection) -> None:
    """Bring an existing registry up to :data:`SCHEMA_VERSION`.

    ``ALTER TABLE ADD COLUMN`` is the one cheap, non-destructive SQLite
    DDL: existing rows get NULL, which is exactly right here — a row
    registered before identity existed genuinely has no known identity,
    and inventing one by parsing ``rel_path`` would merge documents that
    only share a filename (see ``outmem sources backfill``).
    """
    version = con.execute("PRAGMA user_version").fetchone()[0]
    existing = {row[1] for row in con.execute("PRAGMA table_info(sources)")}
    wanted = ("document_key", "superseded_by", "origin_path")
    missing = [c for c in wanted if c not in existing]
    # The version is the fast path; the columns are the actual
    # precondition. Checking both means a registry stamped v2 by an
    # earlier build of this change — when the identity column was still
    # called `logical_key` — repairs itself on open instead of failing
    # every read with "no such column".
    if version >= SCHEMA_VERSION and not missing:
        return
    with con:
        for column in missing:
            con.execute(f"ALTER TABLE sources ADD COLUMN {column} TEXT")
        con.execute(
            "CREATE INDEX IF NOT EXISTS sources_document_key "
            "ON sources(document_key)"
        )
        con.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")


def _read_all_entries(con: sqlite3.Connection) -> dict[str, SourceEntry]:
    cur = con.cursor()
    entries: dict[str, SourceEntry] = {}
    for row in cur.execute(
        "SELECT rel_path, sha256, size_bytes, registered_at, document_key, "
        "superseded_by, origin_path FROM sources ORDER BY rel_path"
    ).fetchall():
        entries[row["rel_path"]] = SourceEntry(
            rel_path=row["rel_path"],
            sha256=row["sha256"],
            registered_at=parse_iso_z(row["registered_at"]),
            document_key=row["document_key"],
            superseded_by=row["superseded_by"],
            origin_path=row["origin_path"],
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


@dataclass(frozen=True)
class RegistryAudit:
    """What ``gc`` found. Every list is repo-relative to ``sources_dir``."""

    missing_files: list[str]  # registered, but the file is gone
    unregistered: list[str]  # on disk, but no registry row
    orphan_ingestions: int  # ingestion rows whose parent row is gone

    @property
    def is_clean(self) -> bool:
        return not (self.missing_files or self.unregistered or self.orphan_ingestions)


def audit_registry(sources_dir: Path) -> RegistryAudit:
    """Reconcile ``.sources.db`` against what is actually on disk.

    Nothing did this before, which is how a registry reaches double-digit
    percentages of junk unnoticed: ``list_sources`` never stats the
    filesystem, so orphaned rows are handed to the agent as readable
    sources it then fails to open.
    """
    registry = SourceRegistry.load(sources_dir)
    registered = set(registry.entries)
    missing = sorted(r for r in registered if not (sources_dir / r).is_file())
    on_disk = {
        p.relative_to(sources_dir).as_posix()
        for p in sources_dir.rglob("*")
        if p.is_file() and p.name != REGISTRY_FILENAME
    }
    con = registry._connection()
    orphans = con.execute(
        "SELECT COUNT(*) FROM ingestions WHERE rel_path NOT IN (SELECT rel_path FROM sources)"
    ).fetchone()[0]
    return RegistryAudit(
        missing_files=missing,
        unregistered=sorted(on_disk - registered),
        orphan_ingestions=int(orphans),
    )


def gc_registry(sources_dir: Path, *, dry_run: bool = True) -> RegistryAudit:
    """Drop registry rows whose file is gone, plus orphaned ingestions.

    Returns the audit describing what was found (and, unless ``dry_run``,
    removed). Deliberately conservative in two ways:

    - **Files with no row are reported, never deleted.** Deleting a user's
      data to satisfy a registry is backwards; that direction is theirs to
      resolve (re-register or remove).
    - **``dry_run=True`` by default**, matching ``repair_pages``.
      ``.sources.db`` is a git-tracked binary, so every apply writes a
      full blob into history.

    Hard delete rather than tombstoning: ``git show HEAD~1:<db>`` already
    recovers the exact pre-gc state, so an in-file tombstone would
    duplicate git at the cost of a schema change and a
    did-I-filter-tombstones bug class. No ``VACUUM`` either — it would
    rewrite the whole file and guarantee a maximal diff.
    """
    audit = audit_registry(sources_dir)
    if dry_run or (not audit.missing_files and not audit.orphan_ingestions):
        return audit
    registry = SourceRegistry.load(sources_dir)
    con = registry._connection()
    with con:
        for rel_path in audit.missing_files:
            # FK ON DELETE CASCADE takes the ingestion chain with it — an
            # ingestion of content nobody can read has no recoverable meaning.
            con.execute("DELETE FROM sources WHERE rel_path = ?", (rel_path,))
        con.execute(
            "DELETE FROM ingestions WHERE rel_path NOT IN (SELECT rel_path FROM sources)"
        )
    for rel_path in audit.missing_files:
        registry.entries.pop(rel_path, None)
    return audit


@dataclass(frozen=True)
class StaleCitation:
    """A page still citing a source version that has been superseded."""

    slug: str
    cited: str  # rel_path the page's provenance names
    current: str  # rel_path of the version that replaced it
    document_key: str


@dataclass(frozen=True)
class KeyCandidate:
    """A proposed ``document_key`` for rows that predate identity."""

    document_key: str
    rows: list[str]  # rel_paths sharing this candidate
    citing_pages: dict[str, list[str]]  # rel_path -> slugs whose provenance cites it
    origins: dict[str, str | None] = field(default_factory=dict)
    """rel_path -> where that file was ingested from, when known.

    The second half of the evidence: two rows from
    ``parsed/fachinfo/amikacin/…`` and ``parsed/fachinfo/aztreonam/…``
    are visibly different documents even when nothing cites either yet.
    """

    @property
    def is_ambiguous(self) -> bool:
        """More than one row claims this candidate.

        Cannot be resolved mechanically: the rows are either versions of
        one document or different documents that share a filename, and
        nothing outmem *derives* distinguishes them. ``citing_pages`` and
        ``origins`` are the evidence a human needs — two rows cited by
        *different* pages are different documents; two cited by the same
        page are versions.
        """
        return len(self.rows) > 1


def propose_document_keys(
    registry: SourceRegistry,
    citations: dict[str, list[str]] | None = None,
) -> list[KeyCandidate]:
    """Group un-keyed rows by the identity their path implies.

    Proposes; never assigns. See :class:`KeyCandidate.is_ambiguous` for
    why an ambiguous group cannot be resolved without a human.
    """
    citations = citations or {}
    groups: dict[str, list[str]] = {}
    origins: dict[str, str | None] = {}
    for entry in sorted(registry.entries.values(), key=lambda e: e.rel_path):
        if entry.document_key is not None:
            continue
        # A pre-hash-dir flat row has no sha segment to strip; its own
        # rel_path is already the candidate.
        candidate = derive_document_key(entry.rel_path, entry.sha256)
        if candidate is None:
            candidate = normalize_document_key(entry.rel_path)
        groups.setdefault(candidate, []).append(entry.rel_path)
        origins[entry.rel_path] = entry.origin_path
    return [
        KeyCandidate(
            document_key=key,
            origins={r: origins.get(r) for r in rows},
            rows=rows,
            citing_pages={r: citations.get(r, []) for r in rows},
        )
        for key, rows in sorted(groups.items())
    ]
