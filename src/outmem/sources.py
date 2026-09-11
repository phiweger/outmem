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
        origin_path   TEXT,   -- where it was ingested from
        refs_scanned_at TEXT  -- when it was scanned for page references
    );

    CREATE TABLE source_refs (
        rel_path  TEXT NOT NULL REFERENCES sources(rel_path) ON DELETE CASCADE,
        token     TEXT NOT NULL,  -- the slug as written in the frozen bytes
        page_slug TEXT NOT NULL,  -- what it meant; kept current by rename
        exact     INTEGER NOT NULL,  -- 1 when written as [[token]]
        PRIMARY KEY (rel_path, token)
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

``source_refs`` is what uncouples a frozen file from mutable page slugs:
the bytes keep naming ``clinical:sepsis``, but the registry remembers
which page that meant at ingest, and ``rename_page`` re-points it. New
columns and tables are added in place by :func:`_migrate` on open, so an
existing registry never needs rebuilding.

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
import re
import shutil
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from datetime import datetime
from itertools import pairwise
from pathlib import Path

from outmem._sqlite import connect as _sqlite_connect
from outmem._time import format_iso_z, parse_iso_z, utc_now
from outmem.completeness import tool_note
from outmem.exceptions import OutmemError

SOURCES_DIR = "sources"

# Untracked sibling of ``sources/``. Same layout, same registry schema,
# same tooling — the only difference is that git never sees it, so it is
# where material you may *read* but not *redistribute* belongs
# (licensed corpora, copyrighted PDFs-converted-to-text, embargoed
# drafts). Pages compiled from it are ordinary tracked wiki pages: the
# derived knowledge ships, the source bytes do not.
#
# Each tree carries its OWN ``.sources.db``. That is deliberate rather
# than incidental: the registry records ``rel_path`` (which embeds the
# filename), ``sha256``, and ``origin_path`` (the absolute path on the
# ingesting machine). A single shared registry would commit all three
# for local sources, leaking exactly what the split exists to protect.
SOURCES_LOCAL_DIR = "sources-local"

REGISTRY_FILENAME = ".sources.db"

# Bumped only when the sources/ingestions schema changes shape.
SCHEMA_VERSION = 3

ALLOWED_EXTENSIONS = frozenset({".md", ".txt", ".csv", ".json", ".mmd", ".yaml", ".yml"})

# What a `finding:` on a provenance entry may say. A citation normally
# means "this page draws on that source"; `finding:` records the outcome
# of a check that produced no such claim, which a reference wiki needs to
# state as a fact rather than leave as an absence of prose:
#
#     provenance:
#       - path: sources/7b6cb641/awmf-s3-hwi-2024.md
#         finding: silent
#         scope: "Therapiedauer der Pyelonephritis in der Schwangerschaft"
#         note: "S3 fuehrt unter 12.2 ausdruecklich 'Keine Empfehlungen'."
#         date: 2026-08-10
#
# Without it "we checked and there is nothing" is indistinguishable from
# "nobody looked", and a reader — or an agent — fills the gap from its own
# knowledge. Only `finding` is a controlled vocabulary; `scope`, `note`
# and `date` are free-form and round-trip verbatim like any other key.
PROVENANCE_FINDINGS = frozenset({"silent", "contradicts", "out-of-scope"})

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

    rel_path: str  # relative to the OWNING TREE, e.g. "veterinary/<sha>/drugs.md"
    """Registry key — always relative to the tree that holds it.

    Deliberately *not* prefixed with ``sources/`` / ``sources-local/``:
    it is the primary key of the row, and every internal consumer
    (``source_refs``, ``backfill``, ``gc``, supersession edges) keys on
    it. The tree-qualified form belongs to presentation and is built on
    demand via :attr:`citation_path`.
    """
    sha256: str
    registered_at: datetime
    size_bytes: int
    ingestions: list[IngestionRecord] = field(default_factory=list)
    local: bool = False
    """Whether this row lives in the untracked ``sources-local/`` tree.

    Derived from which registry the row was read out of, not stored in
    the DB — each registry is wholly inside one tree, so the column
    would be a constant per file and a lie waiting to happen if a tree
    were ever moved.
    """
    document_key: str | None = None
    """Which *document* this file is a version of.

    ``rel_path`` embeds the content hash, so a revised document lands at a
    new path and looks like an unrelated row. ``document_key`` is the
    identity that survives a revision — set explicitly at ingest
    (``--as``), because no derivation can tell "same document, new
    version" from "different document, same filename" in general.
    """
    superseded_by: str | None = None  # rel_path of the version that replaced this
    refs_scanned_at: str | None = None
    """When this source was scanned for page references, if ever.

    Distinct from "has no references": a source that names no pages is
    scanned and empty, while one ingested before the scan existed is
    unknown. Without this column ``sources backfill`` cannot tell them
    apart and re-reports the empty ones on every run.
    """
    origin_path: str | None = None
    """Where the file was ingested from.

    Recorded because it is the field that would have disambiguated the
    historical rows and was previously discarded — a pipeline emitting
    ``.../<drug>/output/<hash>/document.md`` puts the distinguishing part
    here and nowhere else.
    """

    @property
    def citation_path(self) -> str:
        """Tree-qualified path, as a page's ``provenance:`` should cite it.

        ``sources/<rel>`` or ``sources-local/<rel>``. Use this whenever
        the string leaves the registry — display, citations, tool output
        — and :attr:`rel_path` whenever it is used as a key. Collapsing
        the two is how a lookup starts failing on exactly the rows that
        live in the other tree.
        """
        return f"{SOURCES_LOCAL_DIR if self.local else SOURCES_DIR}/{self.rel_path}"


class DocumentKeyConflict(OutmemError):
    """A document identity is already held by a different live row.

    Carries the claimant so the caller can build a message with the
    evidence in it; the bare string is a usable fallback.
    """

    def __init__(self, document_key: str, claimant: SourceEntry) -> None:
        self.document_key = document_key
        self.claimant = claimant
        super().__init__(
            f"{document_key!r} is already the identity of {claimant.rel_path!r}."
        )


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

    Stripping repeats until stable, which makes the function
    **idempotent** — ``report.csv.md`` and ``report.csv`` land on the
    same key. Idempotence is not cosmetic here: the refusal message
    prints a key and tells the operator to pass it back as ``--as``, so a
    key that normalises to something else would silently fail to link the
    two versions it was suggested to link.

    Note the direction this moves in — dropping the extension makes
    *collisions* more likely, and a collision is a refusal, never a
    merge. ``report.md`` and ``report.csv`` under one ``--into`` now stop
    and ask instead of quietly becoming unrelated documents.
    """

    def collapse(value: str) -> str:
        return "/".join(part for part in value.strip().lower().split("/") if part)

    key = collapse(raw)
    while True:
        head, dot, ext = key.rpartition(".")
        if not (dot and f".{ext}" in ALLOWED_EXTENSIONS):
            break
        # Stripping down to nothing means the key was only ever a file
        # type — `--as .md` names no document. Refuse rather than store a
        # key that still carries an extension, which would break the
        # "identity survives a format change" guarantee for that row.
        key = collapse(head)
    if not key:
        raise OutmemError(
            f"{raw!r} is not a usable document identity — a key names a "
            "document, not a file type."
        )
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


NORMALIZED_EXTENSIONS = ALLOWED_EXTENSIONS | frozenset(
    {
        ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
        ".htm", ".html", ".rtf", ".odt", ".epub",
    }
)
"""Extensions :func:`sibling_form` ignores when comparing two keys.

Deliberately wider than :data:`ALLOWED_EXTENSIONS`, which says what
outmem will *ingest*. A guideline corpus is mostly PDFs converted to
text, so ``handbook.pdf`` and ``handbook.docx`` are one document that
changed format — but only ingestable extensions are stripped from a real
key, so the two hold different identities.

Widening the *derivation* instead was the obvious fix and is the wrong
one: dropping an extension makes collisions more likely, and a collision
at ingest is a refusal. A corpus that renders one edition as both
spreadsheet and PDF would start having its ingests rejected. Comparing
in this form and letting `outmem sources rekey` merge keeps the failure
where it belongs — a report, not a blocked pipeline.
"""

_DIGIT_RUN = re.compile(r"(\d+)")


def version_order(document_key: str | None) -> tuple[object, ...]:
    """Sort key for an identity, with digit runs compared as numbers.

    The tie-break when ``registered_at`` cannot order two versions —
    which a bulk ingest guarantees, since the column is stored to the
    second. The edition marker is in the key (``…-2024`` / ``…-2026``),
    so reading it is a far better guess than falling through to
    ``rel_path``, whose leading segment is a content hash and therefore
    orders editions at random.

    Numeric rather than lexicographic because ``v10`` follows ``v9``.
    ``re.split`` on a captured group alternates literal/number starting
    with a literal, so the same tuple position always holds the same
    type and the comparison never mixes ``int`` with ``str``.
    """
    if not document_key:
        return ()
    return tuple(
        int(part) if index % 2 else part
        for index, part in enumerate(_DIGIT_RUN.split(document_key))
    )


def sibling_form(document_key: str) -> str:
    """A comparison form for spotting two keys that name one document.

    Digit runs collapse to ``#`` and a document extension is dropped, so
    ``guidelines/eucast-2024`` and ``guidelines/eucast-2026`` land on one
    form, as do ``handbook.pdf`` and ``handbook.docx``. Two editions
    ingested without ``--as`` derive exactly that pair of keys and hold
    no supersession edge between them.

    Never stored, never an identity, and only ever compared against
    another key's form — a form is lossy on purpose, and writing one to
    the registry would merge documents rather than report them.

    Only meaningful for keys outmem *derived* from a path, which is the
    filter every caller applies. A declared key is a statement, and two
    statements that happen to differ in a digit are two documents: the
    reason ``doi/10.1001-jama-2026`` and ``doi/10.1001-jama-2027`` are
    not reported as versions of each other.
    """
    key = document_key
    while True:
        head, dot, ext = key.rpartition(".")
        if not (head and dot and f".{ext.lower()}" in NORMALIZED_EXTENSIONS):
            break
        key = head
    return _DIGIT_RUN.sub("#", key)


def candidate_key_or_none(rel_path: str, sha256: str) -> str | None:
    """:func:`candidate_document_key`, for callers sweeping every row.

    A file named only for its type (``.md.md``, ``" .md"``) has an
    allowed suffix, so it ingests fine with an explicit ``--as`` — and
    its path then implies no usable identity, which
    :func:`candidate_document_key` reports by raising. That is right for
    a single ingest, where the operator is standing there. It is wrong
    for a pass over the whole registry: one such row took down the
    entire ``outmem lint`` run, not just the check that met it, on a
    wiki that had done nothing wrong.
    """
    try:
        return candidate_document_key(rel_path, sha256)
    except OutmemError:
        return None


def derives_its_own_key(entry: SourceEntry) -> bool:
    """Whether ``entry`` holds exactly the identity its path implies.

    The filter for anything that second-guesses an identity: a declared
    key (``--as``) is a statement about what a document is, where a
    derived one is a guess outmem made from a filename.
    """
    return entry.document_key == candidate_key_or_none(entry.rel_path, entry.sha256)


def candidate_document_key(rel_path: str, sha256: str) -> str:
    """The identity a row's path implies — the rule, in one place.

    Both readers of that rule need to agree: the ingest-time refusal in
    ``add_source`` and the grouping in :func:`propose_document_keys`. If
    they ever disagree, ``outmem sources backfill`` proposes an identity
    that a re-ingest would not derive, and the two halves of one
    invariant drift apart silently.
    """
    return derive_document_key(rel_path, sha256) or normalize_document_key(rel_path)


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


@dataclass(frozen=True)
class SourceRef:
    """A page a frozen source names, and what that name resolved to."""

    rel_path: str  # the source
    token: str  # the slug as written in the frozen bytes
    page_slug: str  # the page it meant; kept current across renames
    exact: bool  # written as [[token]] rather than guessed from prose


@dataclass
class SourceRegistry:
    """SQLite-backed view of the ``wiki/sources/.sources.db`` registry.

    Construct via :meth:`load`, or :meth:`empty` for the deliberate
    no-database case. Mutations through :meth:`register` /
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

    @classmethod
    def empty(cls, sources_dir: Path) -> SourceRegistry:
        """A registry with no rows and no database behind it.

        For a tree whose ``.sources.db`` does not exist and must not be
        created: :meth:`load` creates both the directory and the file, so
        a read-only store asked to list sources would write the tracked
        tree it was opened forbidden to write. Reads answer "nothing
        registered"; a mutation raises, having no connection to write
        through — which is the right failure for a caller that reached
        this constructor by accident.
        """
        return cls(sources_dir=sources_dir)

    def close(self) -> None:
        """Close the underlying SQLite connection. Idempotent."""
        if self._con is not None:
            self._con.close()
            self._con = None

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def latest_for(self, document_key: str) -> SourceEntry | None:
        """The current (un-superseded) version of ``document_key``, if any.

        Reads the in-memory snapshot, so it is a *view*, not a decision
        procedure. Anything that writes an identity resolves it inside
        the write transaction instead — see :meth:`register`.
        """
        candidates = [
            e
            for e in self.entries.values()
            if e.document_key == document_key and e.superseded_by is None
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda e: e.registered_at)

    def _live_claimants(
        self, con: sqlite3.Connection, document_key: str, *, excluding: str
    ) -> list[SourceEntry]:
        """Un-superseded rows holding ``document_key``, newest first.

        Reads the DB rather than :attr:`entries` because it runs inside a
        write transaction: the snapshot was taken when this process
        opened the registry, and a concurrent ``outmem ingest`` — which
        the docs bless via ``xargs -P`` — may have claimed the key since.
        """
        rows = con.execute(
            f"SELECT {_ENTRY_COLUMNS} FROM sources "
            "WHERE document_key = ? AND superseded_by IS NULL AND rel_path != ? "
            "ORDER BY registered_at DESC, rel_path",
            (document_key, excluding),
        ).fetchall()
        return [_entry_from_row(r) for r in rows]

    def _rows_with_key(
        self, con: sqlite3.Connection, document_key: str
    ) -> list[SourceEntry]:
        """Every row holding ``document_key`` — superseded ones included.

        The counterpart to :meth:`_live_claimants` for :meth:`rekey`,
        which moves a *document* rather than a version and so must see
        the whole chain. Reading only the live head would relabel it and
        strand its predecessors under the old key: two documents where
        there was one, which is the corruption rekey exists to repair.
        """
        rows = con.execute(
            f"SELECT {_ENTRY_COLUMNS} FROM sources WHERE document_key = ? "
            "ORDER BY registered_at, rel_path",
            (document_key,),
        ).fetchall()
        return [_entry_from_row(r) for r in rows]

    def register(
        self,
        rel_path: str,
        *,
        sha256: str,
        size_bytes: int,
        when: datetime | None = None,
        document_key: str | None = None,
        origin_path: str | None = None,
        derived_key: bool = False,
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

        ``derived_key=True`` says the key was *inferred* from the path
        rather than declared, which inverts the meaning of an existing
        claimant: a declared key means "supersede that", an inferred one
        means "I cannot tell whether this is the next version or a
        different document" — so it raises
        :class:`DocumentKeyConflict` instead of linking.

        The whole identity decision happens inside one ``BEGIN
        IMMEDIATE`` transaction, reading the DB rather than the in-memory
        snapshot. Deciding it from the snapshot let two concurrent
        ingests both see v1 as the live head, each supersede it, and
        leave two live rows for one document — a page compacted from the
        losing head would then never appear in ``outmem stale``.

        Note the sha is part of ``rel_path``, so a revised document is
        always a *new row*; the ``!= sha256`` refresh branch below is only
        reachable for callers that build paths themselves.
        """
        existing = self.entries.get(rel_path)
        if existing and existing.sha256 == sha256:
            return existing

        ts = when.replace(microsecond=0) if when else utc_now()
        con = self._connection()
        # IMMEDIATE, not the implicit deferred transaction: the claimant
        # lookup below is a read that a later write depends on, and two
        # deferred transactions upgrading to a write is the one case
        # SQLite resolves by returning BUSY rather than waiting.
        con.execute("BEGIN IMMEDIATE")
        try:
            predecessor = None
            if document_key is not None:
                claimants = self._live_claimants(con, document_key, excluding=rel_path)
                if claimants and derived_key:
                    raise DocumentKeyConflict(document_key, claimants[0])
                predecessor = claimants[0] if claimants else None
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
        except BaseException:
            con.rollback()
            raise
        con.commit()

        entry = SourceEntry(
            rel_path=rel_path,
            sha256=sha256,
            registered_at=ts,
            size_bytes=size_bytes,
            ingestions=[],
            document_key=document_key,
            origin_path=origin_path,
        )
        if predecessor is not None and predecessor.rel_path in self.entries:
            self.entries[predecessor.rel_path].superseded_by = rel_path
        self.entries[rel_path] = entry
        return entry

    def adopt_document_key(self, rel_path: str, document_key: str) -> SourceEntry:
        """Declare the identity of a row that already exists.

        The path for ``--as`` on content already registered, which is
        exactly what ``outmem sources backfill`` tells the operator to do
        to resolve an ambiguous group. Refuses when a *different* live row
        already holds the key (the same rule as ``register``), and when
        the row already holds a different key — changing an established
        identity would strand the supersession edges pointing at it, so
        that is a deliberate operation, not a side effect of re-ingest.
        """
        entry = self.entries.get(rel_path)
        if entry is None:
            raise OutmemError(f"cannot set identity: {rel_path!r} is not registered.")
        if entry.document_key == document_key:
            return entry
        if entry.document_key is not None:
            raise OutmemError(
                f"{rel_path!r} is already the identity {entry.document_key!r}. "
                f"Refusing to silently rename it to {document_key!r} — supersession "
                "edges point at the old identity. Use `outmem sources rekey "
                f"{entry.document_key} --to {document_key}`, which moves the whole "
                "chain instead of stranding it."
            )
        con = self._connection()
        con.execute("BEGIN IMMEDIATE")
        try:
            claimants = self._live_claimants(con, document_key, excluding=rel_path)
            if claimants:
                raise DocumentKeyConflict(document_key, claimants[0])
            con.execute(
                "UPDATE sources SET document_key = ? WHERE rel_path = ?",
                (document_key, rel_path),
            )
        except BaseException:
            con.rollback()
            raise
        con.commit()
        entry.document_key = document_key
        return entry

    def _snapshot_rows_with_key(self, document_key: str) -> list[SourceEntry]:
        """:meth:`_rows_with_key` against the snapshot, for the dry run."""
        return sorted(
            (e for e in self.entries.values() if e.document_key == document_key),
            key=lambda e: (e.registered_at, e.rel_path),
        )

    def plan_rekey(self, old_key: str, new_key: str | None = None) -> RekeyResult:
        """What :meth:`rekey` would do, read off the snapshot. Writes nothing."""
        old_key = normalize_document_key(old_key)
        new_key = old_key if new_key is None else normalize_document_key(new_key)
        old_rows = self._snapshot_rows_with_key(old_key)
        if not old_rows:
            raise _no_such_document(old_key)
        new_rows = [] if new_key == old_key else self._snapshot_rows_with_key(new_key)
        return _plan_rekey(old_rows, new_rows, new_key)

    def rekey(self, old_key: str, new_key: str | None = None) -> RekeyResult:
        """Move a *document* to another identity, and rebuild its chain.

        The deliberate operation :meth:`adopt_document_key` refuses to
        perform as a side effect. Two things make it safe where a bare
        ``UPDATE sources SET document_key`` is not:

        **It moves every row holding the key**, so a chain is never split
        across two identities.

        **It rewrites the supersession edges** over the resulting set,
        oldest to newest. That is not bookkeeping — it is the whole point
        when ``new_key`` is already held, which is the common case: two
        editions of one document that landed under different *derived*
        keys (``…-2024`` and ``…-2026``) have no edge between them, and
        relabelling alone leaves two live heads under one identity.
        :meth:`latest_for` then silently returns the newer of them and
        ``outmem stale`` reports nothing — the exact silence supersession
        exists to break, reintroduced by the repair meant to fix it.

        Called with no ``new_key`` it rebuilds the chain in place, which
        is the repair for a registry already in that state.

        Nothing is deleted, so ingestion history and recorded page
        references survive. Returns the same :class:`RekeyResult` shape
        :meth:`plan_rekey` previews, with ``applied`` saying whether
        anything was actually written — a rekey that would change
        nothing writes nothing, because the registry is a git-tracked
        binary and an idempotent repair would otherwise add a full blob
        to history on every run.

        That decision is made in here, against the rows this transaction
        read, rather than by the caller against its snapshot: the
        snapshot was taken when the process opened the registry, and a
        concurrent ``outmem ingest`` since then would make "already
        correct" a stale answer and silently skip a write that was
        needed.
        """
        old_key = normalize_document_key(old_key)
        new_key = old_key if new_key is None else normalize_document_key(new_key)
        con = self._connection()
        con.execute("BEGIN IMMEDIATE")
        try:
            old_rows = self._rows_with_key(con, old_key)
            if not old_rows:
                raise _no_such_document(old_key)
            new_rows = [] if new_key == old_key else self._rows_with_key(con, new_key)
            plan = _plan_rekey(old_rows, new_rows, new_key)
            self._refuse_foreign_edges(con, plan.chain)
            if _already_chained(plan, [*old_rows, *new_rows]):
                con.rollback()
                return plan
            if new_key != old_key:
                con.execute(
                    "UPDATE sources SET document_key = ? WHERE document_key = ?",
                    (new_key, old_key),
                )
            for older, newer in plan.edges:
                con.execute(
                    "UPDATE sources SET superseded_by = ? WHERE rel_path = ?",
                    (newer, older),
                )
            con.execute(
                "UPDATE sources SET superseded_by = NULL WHERE rel_path = ?",
                (plan.chain[-1],),
            )
        except BaseException:
            con.rollback()
            raise
        con.commit()
        for rel_path in plan.chain:
            entry = self.entries.get(rel_path)
            if entry is not None:
                entry.document_key = new_key
                entry.superseded_by = None
        for older, newer in plan.edges:
            if older in self.entries:
                self.entries[older].superseded_by = newer
        return replace(plan, applied=True)

    def _refuse_foreign_edges(
        self, con: sqlite3.Connection, chain: tuple[str, ...]
    ) -> None:
        """Refuse if a row outside ``chain`` supersedes one inside it.

        Supersession edges stay within a ``document_key`` by
        construction, so this cannot fire on a registry outmem wrote
        alone. It can on one edited out of band — and rewriting the chain
        under such an edge would leave the outside row pointing into a
        document it is not a version of, silently.
        """
        marks = ",".join("?" * len(chain))
        rows = con.execute(
            f"SELECT rel_path, superseded_by FROM sources "
            f"WHERE superseded_by IN ({marks}) AND rel_path NOT IN ({marks})",
            (*chain, *chain),
        ).fetchall()
        if rows:
            pointers = ", ".join(f"{r['rel_path']} -> {r['superseded_by']}" for r in rows)
            raise OutmemError(
                "refusing to rekey: supersession edge(s) from outside this "
                f"document point into it ({pointers}). The registry was edited "
                "out of band; resolve those rows first."
            )

    def record_refs(self, rel_path: str, refs: Iterable[SourceRef]) -> list[SourceRef]:
        """Record which pages a frozen source names, resolved at ingest.

        This is what uncouples a content-addressed file from mutable page
        slugs. The bytes keep saying ``clinical:sepsis`` — they must, a
        source is a faithful copy — but the registry remembers that *at
        the moment of ingest* that token meant a specific page, and
        :meth:`repoint_refs` keeps that current across renames. The
        reference is then held by identity rather than by a string, which
        is the same move ``origin_path`` makes for document names.

        Only resolvable references are worth storing; the caller filters.
        """
        rows = list(refs)
        stamp = format_iso_z(utc_now())
        con = self._connection()
        with con:
            if rows:
                con.executemany(
                    "INSERT OR REPLACE INTO source_refs "
                    "(rel_path, token, page_slug, exact) VALUES (?, ?, ?, ?)",
                    [(rel_path, r.token, r.page_slug, int(r.exact)) for r in rows],
                )
            # Stamped even when nothing was found, so "scanned and empty"
            # is distinguishable from "never scanned".
            con.execute(
                "UPDATE sources SET refs_scanned_at = ? WHERE rel_path = ?",
                (stamp, rel_path),
            )
        entry = self.entries.get(rel_path)
        if entry is not None:
            entry.refs_scanned_at = stamp
        return rows

    def refs(self, rel_path: str | None = None) -> list[SourceRef]:
        """Recorded source→page references, for one source or all of them."""
        con = self._connection()
        if rel_path is None:
            rows = con.execute(
                "SELECT rel_path, token, page_slug, exact FROM source_refs "
                "ORDER BY rel_path, token"
            ).fetchall()
        else:
            rows = con.execute(
                "SELECT rel_path, token, page_slug, exact FROM source_refs "
                "WHERE rel_path = ? ORDER BY token",
                (rel_path,),
            ).fetchall()
        return [
            SourceRef(
                rel_path=r["rel_path"],
                token=r["token"],
                page_slug=r["page_slug"],
                exact=bool(r["exact"]),
            )
            for r in rows
        ]

    def repoint_refs(self, old_slug: str, new_slug: str) -> int:
        """Follow a rename. Returns the number of references re-pointed.

        ``rename_page`` rewrites ``[[links]]`` in pages and ``log/`` but
        physically cannot rewrite a source — that is what content
        addressing means. It can rewrite the *mapping*, which is why the
        mapping exists.
        """
        con = self._connection()
        with con:
            cur = con.execute(
                "UPDATE source_refs SET page_slug = ? WHERE page_slug = ?",
                (new_slug, old_slug),
            )
        return int(cur.rowcount or 0)

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


@dataclass(frozen=True)
class SourcePlacement:
    """Where a source *would* land, computed without touching the disk."""

    dest: Path
    rel_path: str
    sha256: str


def plan_source_copy(
    source: Path,
    sources_dir: Path,
    *,
    into_subdir: str | None = None,
    rename: str | None = None,
) -> SourcePlacement:
    """Validate ``source`` and compute its destination — no writes.

    Split out of :func:`copy_source` so a caller can decide whether to
    accept an ingest *before* anything lands on disk. Copying first meant
    every refused ingest left an unregistered orphan under
    ``wiki/sources/`` that ``outmem lint`` then flagged and
    ``outmem sources gc`` deliberately refuses to delete — routine, since
    a ``<drug>/output/<hash>/document.md`` pipeline is refused on its
    second document by design.
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

    if into_subdir and (into_subdir.startswith("/") or ".." in into_subdir.split("/")):
        raise OutmemError(f"unsafe into_subdir: {into_subdir!r}")

    sha = compute_sha256(source)
    parent = sources_dir / into_subdir if into_subdir else sources_dir
    dest = parent / sha[:SHA_PREFIX_LEN] / filename
    return SourcePlacement(
        dest=dest, rel_path=str(dest.relative_to(sources_dir)), sha256=sha
    )


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
    placement = plan_source_copy(
        source, sources_dir, into_subdir=into_subdir, rename=rename
    )
    placement.dest.parent.mkdir(parents=True, exist_ok=True)
    # Same content → same hash dir → idempotent.
    if not placement.dest.exists():
        shutil.copy2(source, placement.dest)
    return placement.dest, placement.rel_path


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
        # An out-of-band sentinel, NOT a bracketed ellipsis. This lands
        # directly in the writing agent's context at the moment it is
        # reading source material to compact into a page, so the marker
        # outmem emits here is a marker it teaches — and page bodies
        # ending in an elision are refused (see outmem.completeness).
        note = tool_note(
            f"source truncated — {max_chars} of {len(text)} chars shown "
            f"(sources.max_chars)"
        )
        return f"{text[:max_chars]}\n\n{note}"
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
    cur.execute(
        "CREATE TABLE IF NOT EXISTS source_refs ("
        " rel_path TEXT NOT NULL"
        " REFERENCES sources(rel_path) ON DELETE CASCADE,"
        " token TEXT NOT NULL,"      # as written in the frozen file
        " page_slug TEXT NOT NULL,"  # what it meant; kept current by rename
        " exact INTEGER NOT NULL,"   # 1 when written as [[token]]
        " PRIMARY KEY (rel_path, token))"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS source_refs_page ON source_refs(page_slug)"
    )
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
    # v3 added source_refs, created unconditionally above by CREATE TABLE
    # IF NOT EXISTS — so the migration only has to move user_version.
    # The version is the fast path; the columns are the actual
    # precondition. Checking both means a registry stamped v2 by an
    # earlier build of this change — when the identity column was still
    # called `logical_key` — repairs itself on open instead of failing
    # every read with "no such column".
    #
    # This first look is unlocked, and that is safe only because it can
    # go stale in one direction: columns are added, never removed, so
    # "nothing to do" stays true once it is true.
    if _schema_current(con):
        return
    with con:
        # The decision has to be made *under* the write lock. Sixteen
        # processes opening a fresh registry together all read "column
        # missing" from the unlocked check above, and every one of them
        # then issues the ALTER — the second fails with "duplicate
        # column". BEGIN IMMEDIATE takes the lock now rather than at the
        # first write (busy_timeout makes the others wait instead of
        # erroring), and reading the schema again inside it sees what the
        # first opener did.
        con.execute("BEGIN IMMEDIATE")
        existing = {row[1] for row in con.execute("PRAGMA table_info(sources)")}
        for column in _V2_COLUMNS:
            if column not in existing:
                con.execute(f"ALTER TABLE sources ADD COLUMN {column} TEXT")
        con.execute(
            "CREATE INDEX IF NOT EXISTS sources_document_key "
            "ON sources(document_key)"
        )
        con.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")


_V2_COLUMNS = ("document_key", "superseded_by", "origin_path", "refs_scanned_at")


def _schema_current(con: sqlite3.Connection) -> bool:
    """Whether ``sources`` is stamped :data:`SCHEMA_VERSION` *and* carries
    every v2 column — both, for the `logical_key` repair described above."""
    version = con.execute("PRAGMA user_version").fetchone()[0]
    existing = {row[1] for row in con.execute("PRAGMA table_info(sources)")}
    return version >= SCHEMA_VERSION and all(c in existing for c in _V2_COLUMNS)


_ENTRY_COLUMNS = (
    "rel_path, sha256, size_bytes, registered_at, document_key, "
    "superseded_by, origin_path, refs_scanned_at"
)
"""Every column :func:`_entry_from_row` reads, in one place.

Shared by the snapshot load and the in-transaction reads. A query that
selected a subset built entries whose missing fields silently defaulted
— which is how ``_live_claimants`` used to hand back rows reporting
``refs_scanned_at=None`` for sources that had in fact been scanned.
"""


def _entry_from_row(row: sqlite3.Row) -> SourceEntry:
    """Build a :class:`SourceEntry` from a row selected as ``_ENTRY_COLUMNS``."""
    return SourceEntry(
        rel_path=row["rel_path"],
        sha256=row["sha256"],
        registered_at=parse_iso_z(row["registered_at"]),
        document_key=row["document_key"],
        superseded_by=row["superseded_by"],
        origin_path=row["origin_path"],
        refs_scanned_at=row["refs_scanned_at"],
        size_bytes=int(row["size_bytes"]),
        ingestions=[],
    )


def _read_all_entries(con: sqlite3.Connection) -> dict[str, SourceEntry]:
    cur = con.cursor()
    entries: dict[str, SourceEntry] = {}
    for row in cur.execute(
        f"SELECT {_ENTRY_COLUMNS} FROM sources ORDER BY rel_path"
    ).fetchall():
        entries[row["rel_path"]] = _entry_from_row(row)
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
    # An audit is a read. `load` creates the directory and the database as
    # a side effect of opening, so auditing a tree with no registry used to
    # *write* one — a dry run of `sources gc` on a read-only store, or on a
    # curator-shipped clone, left a fresh `.sources.db` in the tracked tree.
    # No file means no rows and no ingestions; everything on disk is
    # unregistered, and there is nothing to open.
    has_db = (sources_dir / REGISTRY_FILENAME).is_file()
    registry = SourceRegistry.load(sources_dir) if has_db else SourceRegistry.empty(sources_dir)
    registered = set(registry.entries)
    missing = sorted(r for r in registered if not (sources_dir / r).is_file())
    on_disk = {
        p.relative_to(sources_dir).as_posix()
        for p in sources_dir.rglob("*")
        if p.is_file() and p.name != REGISTRY_FILENAME
    }
    orphans = 0
    if has_db:
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
            # Splice the row out of its version chain before deleting it.
            # Leaving the pointer dangling makes `outmem stale` tell the
            # reader to diff against a path that is in neither the registry
            # nor the filesystem.
            successor = registry.entries[rel_path].superseded_by
            con.execute(
                "UPDATE sources SET superseded_by = ? WHERE superseded_by = ?",
                (successor, rel_path),
            )
            # FK ON DELETE CASCADE takes the ingestion chain with it — an
            # ingestion of content nobody can read has no recoverable meaning.
            con.execute("DELETE FROM sources WHERE rel_path = ?", (rel_path,))
        con.execute(
            "DELETE FROM ingestions WHERE rel_path NOT IN (SELECT rel_path FROM sources)"
        )
    for rel_path in audit.missing_files:
        successor = registry.entries[rel_path].superseded_by
        for entry in registry.entries.values():
            if entry.superseded_by == rel_path:
                entry.superseded_by = successor
        registry.entries.pop(rel_path, None)
    return audit


@dataclass(frozen=True)
class UnchainedVersions:
    """Live rows that belong to one document without being chained to it.

    "Live" is the whole test. Two editions properly chained leave one
    un-superseded row and are not reported; two that hold no edge between
    them leave two, and ``outmem stale`` is silent about both.
    """

    entries: list[SourceEntry]

    @property
    def keys(self) -> list[str]:
        """The distinct identities in this group, oldest row first."""
        seen: list[str] = []
        for entry in self.entries:
            if entry.document_key and entry.document_key not in seen:
                seen.append(entry.document_key)
        return seen

    @property
    def shares_one_key(self) -> bool:
        """All rows already hold the same identity — only the edges are missing.

        A different defect from two keys that merely *resemble* each
        other: the registry is asserting these are one document while
        leaving several of them current, which its own write paths refuse
        to do. It needs no judgement to fix — just the chain.
        """
        return len(self.keys) == 1


def find_unchained_versions(registry: SourceRegistry) -> list[UnchainedVersions]:
    """Live rows that should be one chain and are not.

    Two defects, found in one pass because both are cured by
    :meth:`SourceRegistry.rekey` and neither is visible any other way.

    **One identity, several live rows.** A broken invariant — ``register``
    and ``adopt_document_key`` both refuse to create it — so this half is
    exact, needs no heuristic, and applies to *declared* identities as
    much as derived ones. Reachable by editing ``.sources.db`` out of
    band, which is exactly what an operator does when outmem's own API
    refuses the relabel they want.

    **Identities that differ only in a number.** A guess, and restricted
    to keys outmem *derived* from a path. A declared key (``--as``) is a
    statement about what a document is; second-guessing it would report
    every deliberately-numbered pair — every DOI, every register number —
    forever, and a check that cannot reach zero gets silenced wholesale.
    """
    live = [
        entry
        for entry in registry.entries.values()
        if entry.superseded_by is None and entry.document_key
    ]
    by_key: dict[str, list[SourceEntry]] = {}
    for entry in live:
        by_key.setdefault(entry.document_key or "", []).append(entry)
    groups = [
        UnchainedVersions(sorted(rows, key=_version_sort))
        for _key, rows in sorted(by_key.items())
        if len(rows) > 1
    ]

    by_form: dict[str, list[SourceEntry]] = {}
    for entry in live:
        # Already reported above as a broken invariant. That comes first;
        # a re-lint then surfaces any sibling it was hiding.
        if len(by_key[entry.document_key or ""]) > 1:
            continue
        if not derives_its_own_key(entry):
            continue
        by_form.setdefault(sibling_form(entry.document_key or ""), []).append(entry)
    groups += [
        UnchainedVersions(sorted(rows, key=_version_sort))
        for _form, rows in sorted(by_form.items())
        if len({r.document_key for r in rows}) > 1
    ]
    return groups


@dataclass(frozen=True)
class RekeyResult:
    """What a :meth:`SourceRegistry.rekey` did, or would do."""

    document_key: str
    """The identity every row in :attr:`chain` ends up holding."""
    moved: tuple[str, ...]
    """rel_paths read out of the source identity — every row holding it.

    Not "rows whose key changes": a chain repair (no target given) moves
    a document onto itself, and the rows still count as the ones being
    operated on. What changes is :attr:`chain`.
    """
    merged_with: tuple[str, ...]
    """rel_paths that already held the target identity.

    Non-empty means this is a *merge* — two derived keys turning out to
    name one document, which is the usual reason to rekey at all.
    """
    chain: tuple[str, ...]
    """The document's versions, oldest first, as the chain now reads.

    Every element but the last gains a ``superseded_by`` pointing at its
    successor; the last is the live head.
    """
    applied: bool = False
    """Whether this result came from a write rather than a preview.

    False from :meth:`SourceRegistry.plan_rekey`, and also from a
    :meth:`WikiStore.rekey_document` that found nothing to do — so a
    caller can tell "already correct" from "written", which the chain
    alone cannot say.
    """
    tied_order: tuple[str, ...] = ()
    """Chain members registered in the same second as their predecessor.

    ``registered_at`` is stored to the second, so a bulk ingest can
    register two editions with identical timestamps — and then nothing
    in the registry *records* which is newer. The chain falls back to
    :func:`version_order`, which reads the edition marker out of the key
    and is usually right; ``rel_path`` breaks any remaining tie so the
    preview and the write agree. Right or not, it decides which version
    ``outmem stale`` calls current, so it is named here rather than
    presented as a fact.
    """

    @property
    def edges(self) -> tuple[tuple[str, str], ...]:
        """``(superseded, successor)`` pairs the chain implies."""
        return tuple(pairwise(self.chain))


def _already_chained(plan: RekeyResult, rows: list[SourceEntry]) -> bool:
    """Whether ``rows`` already read exactly as ``plan`` would leave them."""
    expected = dict(plan.edges)
    return all(
        entry.document_key == plan.document_key
        and entry.superseded_by == expected.get(entry.rel_path)
        for entry in rows
    )


def _version_sort(entry: SourceEntry) -> tuple[object, ...]:
    """Oldest version first: ingest time, then the key, then the path.

    ``rel_path`` last and only as a determinism backstop — it starts with
    a content hash, so letting it decide would order editions at random.
    """
    return (entry.registered_at, version_order(entry.document_key), entry.rel_path)


def _no_such_document(document_key: str) -> OutmemError:
    return OutmemError(
        f"no source holds the identity {document_key!r}. "
        "`outmem sources list` shows what is registered; note a key is the "
        "*document* identity, not a file path."
    )


def _plan_rekey(
    old_rows: list[SourceEntry], new_rows: list[SourceEntry], new_key: str
) -> RekeyResult:
    """Order a document's versions and say which rows move.

    Split out because the dry run reads the in-memory snapshot while the
    write re-reads the DB inside its transaction — the same rule has to
    produce both, or ``--apply`` writes a chain the preview never showed.

    Ordering is by ``registered_at``, which is the only ordering outmem
    has and the one :meth:`SourceRegistry.latest_for` already resolves
    ties with. A pre-existing chain that disagrees with it was written by
    that same rule, so re-deriving rather than preserving edges is not a
    loss — and it is what lets a merge interleave two chains instead of
    concatenating them.
    """
    union = sorted([*old_rows, *new_rows], key=_version_sort)
    return RekeyResult(
        document_key=new_key,
        moved=tuple(e.rel_path for e in old_rows),
        merged_with=tuple(e.rel_path for e in new_rows),
        chain=tuple(e.rel_path for e in union),
        tied_order=tuple(
            later.rel_path
            for earlier, later in pairwise(union)
            if earlier.registered_at == later.registered_at
        ),
    )


@dataclass(frozen=True)
class StaleCitation:
    """A page still citing a source version that has been superseded."""

    slug: str
    cited: str  # rel_path the page's provenance names
    current: str  # rel_path of the version that replaced it
    document_key: str
    finding: str | None = None
    """The citation's ``finding:``, when it recorded a checked absence.

    Raises the priority of the row rather than decorating it. "The page
    says X, citing v1" going stale means the claim may have changed.
    "We checked v1 and it was **silent** on X" going stale means the new
    version may finally answer — the recorded fact is about what a
    specific version did *not* contain, so it expires with that version
    in a way a positive citation does not.
    """
    acknowledged: str | None = None
    """Why citing the superseded version is deliberate, if it is.

    From the citation's ``superseded_ok:``, and only when that
    acknowledgement still applies to the *current* head — see
    :func:`outmem.store._acknowledgement`. Rows carrying it are left out
    of the default report; a caller that asked for them can tell the two
    apart by this field.
    """
    current_exists: bool = True
    """Whether ``current`` is still a registered row.

    A supersession pointer can outlive its target — ``outmem sources gc``
    splices chains it prunes, but a registry edited out-of-band can still
    leave one dangling. Telling the reader to diff against a path that is
    in neither the registry nor the filesystem is worse than saying so.
    """


@dataclass(frozen=True)
class KeyCandidate:
    """A proposed ``document_key`` for rows that predate identity."""

    document_key: str
    rows: list[str]  # un-keyed rel_paths sharing this candidate
    citing_pages: dict[str, list[str]]  # rel_path -> slugs whose provenance cites it
    origins: dict[str, str | None] = field(default_factory=dict)
    """rel_path -> where that file was ingested from, when known.

    The second half of the evidence: two rows from
    ``parsed/fachinfo/amikacin/…`` and ``parsed/fachinfo/aztreonam/…``
    are visibly different documents even when nothing cites either yet.
    """
    held_by: list[str] = field(default_factory=list)
    """rel_paths that *already* hold this identity, if any.

    A candidate whose name is taken is not assignable, however few
    un-keyed rows share it: writing it would put two live rows on one
    identity, which is precisely the merge ``add_source`` refuses to
    perform — done silently, by outmem's own migration command.
    """

    @property
    def is_ambiguous(self) -> bool:
        """This candidate cannot be assigned without a human.

        Either several un-keyed rows derive it — they are versions of one
        document or different documents sharing a filename, and nothing
        outmem *derives* distinguishes them — or the name is already
        held. ``citing_pages`` and ``origins`` are the evidence: two rows
        cited by *different* pages are different documents; two cited by
        the same page are versions.
        """
        return len(self.rows) > 1 or bool(self.held_by)


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
    claimed: dict[str, list[str]] = {}
    for entry in sorted(registry.entries.values(), key=lambda e: e.rel_path):
        origins[entry.rel_path] = entry.origin_path
        if entry.document_key is not None:
            claimed.setdefault(entry.document_key, []).append(entry.rel_path)
            continue
        # A row whose path implies no usable identity has nothing to
        # propose; skipping leaves it keyless, which is what it already
        # is. Raising here took `outmem sources backfill` down with it.
        candidate = candidate_key_or_none(entry.rel_path, entry.sha256)
        if candidate is None:
            continue
        groups.setdefault(candidate, []).append(entry.rel_path)
    return [
        KeyCandidate(
            document_key=key,
            origins={r: origins.get(r) for r in [*rows, *claimed.get(key, [])]},
            rows=rows,
            citing_pages={
                r: citations.get(r, []) for r in [*rows, *claimed.get(key, [])]
            },
            held_by=claimed.get(key, []),
        )
        for key, rows in sorted(groups.items())
    ]
