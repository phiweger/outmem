"""``sqlite-vec`` wrapper — the persisted vector index.

Schema (versioned in the ``meta`` table):

.. code-block:: sql

    CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
    -- rows: schema_version, embedding_model, embedding_dims

    CREATE TABLE files (
        rel_path TEXT PRIMARY KEY,        -- "wiki/pages/pricing-formula.md"
        content_hash TEXT NOT NULL,        -- sha256 of the chunked body
        indexed_at TEXT NOT NULL,
        chunks_count INTEGER NOT NULL,
        kind TEXT NOT NULL                 -- 'wiki' | 'source'
    );

    CREATE TABLE chunks (
        id INTEGER PRIMARY KEY,
        rel_path TEXT NOT NULL REFERENCES files(rel_path),
        chunk_index INTEGER NOT NULL,
        start_char INTEGER NOT NULL,
        end_char INTEGER NOT NULL,
        content TEXT NOT NULL,
        content_hash TEXT NOT NULL
    );
    CREATE INDEX chunks_rel_path ON chunks(rel_path);

    CREATE VIRTUAL TABLE chunks_vec USING vec0(
        embedding float[<dims>] distance_metric=cosine
    );

The dim count in ``chunks_vec`` is fixed at table creation, so
changing the embedding model means rebuilding the table —
:func:`VectorStore.open` detects this and surfaces a clear error.

The update strategy is *dumb-and-correct*: when a file changes,
delete every chunk + vector for that ``rel_path``, re-chunk,
re-embed, insert. This dodges the chunk-frame-shift problem (an
insert in the middle of a doc invalidates every subsequent hash
anyway) and keeps the code simple. Files whose ``content_hash`` is
unchanged are skipped entirely — :meth:`VectorStore.reindex_file`
returns ``ReindexResult(skipped=True)``.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import struct
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from outmem._sqlite import connect as _sqlite_connect
from outmem._time import format_iso_z, utc_now
from outmem.config import DEFAULT_SEMANTIC_REINDEX_CONCURRENCY
from outmem.exceptions import OutmemError
from outmem.semantic.chunker import Chunk, chunk_text, hash_text

log = logging.getLogger(__name__)

DEFAULT_DB_FILENAME = ".vectors.db"
SCHEMA_VERSION = "1"


@dataclass(frozen=True)
class Match:
    """One result row from :func:`VectorStore.find_similar`."""

    rel_path: str
    chunk_index: int
    similarity: float  # 1.0 - cosine_distance, in [0, 1]
    content: str
    start_char: int
    end_char: int


@dataclass(frozen=True)
class ReindexResult:
    """Outcome of a single ``reindex_file`` call."""

    rel_path: str
    skipped: bool  # True if content_hash matched and nothing was done
    chunks_removed: int
    chunks_added: int
    embeddings_called: int  # number of texts sent to the embedder


class VectorStore:
    """Persisted vector index over wiki pages and sources.

    Construct via :func:`VectorStore.open` so the schema is initialised
    and the embedding model is reconciled with what's on disk.
    """

    def __init__(
        self,
        *,
        db_path: Path,
        connection: sqlite3.Connection,
        embedder: Any,  # EmbedderHandle, kept Any for lazy import
    ) -> None:
        self.db_path = db_path
        self.con = connection
        self.embedder = embedder
        # Serialize every connection touch. The connection is shared across
        # worker threads (evaluate runs retrievers in a ThreadPoolExecutor;
        # the optimizer drives many concurrent semantic queries). sqlite3
        # allows check_same_thread=False, but the caller must serialize
        # access — cursors are not concurrent-safe and racing reads have
        # been observed to corrupt row-factory state (rows come back as
        # plain tuples → IndexError on dict-style access).
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @classmethod
    def open(
        cls,
        db_path: Path,
        *,
        embedder: Any,
    ) -> VectorStore:
        """Open or create the vector DB at ``db_path``.

        Loads the ``sqlite-vec`` extension, applies the schema if the
        file is fresh, and validates the embedding model + dims match
        the supplied ``embedder``. Mismatch raises :class:`OutmemError`
        with instructions to run ``outmem reindex --force``.
        """
        # Lazy import so the core package doesn't depend on sqlite-vec.
        try:
            import sqlite_vec  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover — extra not installed
            raise OutmemError(
                "semantic features require `outmem[semantic]` "
                "(pip install 'outmem[semantic]')"
            ) from exc

        # See _sqlite.connect for the check_same_thread / row_factory
        # rationale. We additionally load the sqlite-vec extension
        # here, which has to happen on the live connection before any
        # query touches the vec0 virtual table.
        con = _sqlite_connect(db_path)
        con.enable_load_extension(True)
        sqlite_vec.load(con)
        con.enable_load_extension(False)

        cls._init_schema(con, dimensions=embedder.dimensions)
        cls._reconcile_meta(con, embedder=embedder)

        return cls(db_path=db_path, connection=con, embedder=embedder)

    def close(self) -> None:
        # Acquire the lock so we don't pull the connection out from under a
        # concurrent reader mid-query (every other public method takes it).
        with self._lock:
            self.con.close()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    @staticmethod
    def _init_schema(con: sqlite3.Connection, *, dimensions: int) -> None:
        cur = con.cursor()
        cur.execute(
            "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)"
        )
        cur.execute(
            "CREATE TABLE IF NOT EXISTS files ("
            " rel_path TEXT PRIMARY KEY,"
            " content_hash TEXT NOT NULL,"
            " indexed_at TEXT NOT NULL,"
            " chunks_count INTEGER NOT NULL,"
            " kind TEXT NOT NULL)"
        )
        cur.execute(
            "CREATE TABLE IF NOT EXISTS chunks ("
            " id INTEGER PRIMARY KEY,"
            " rel_path TEXT NOT NULL,"
            " chunk_index INTEGER NOT NULL,"
            " start_char INTEGER NOT NULL,"
            " end_char INTEGER NOT NULL,"
            " content TEXT NOT NULL,"
            " content_hash TEXT NOT NULL)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS chunks_rel_path ON chunks(rel_path)"
        )

        # The vec0 virtual table can only be created once per file
        # because the column type bakes the dimensions in. Check first.
        existing = cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='chunks_vec'"
        ).fetchone()
        if existing is None:
            cur.execute(
                f"CREATE VIRTUAL TABLE chunks_vec USING vec0("
                f"embedding float[{dimensions}] distance_metric=cosine)"
            )

        # Pin schema version on first init.
        cur.execute(
            "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', ?)",
            (SCHEMA_VERSION,),
        )
        con.commit()

    @staticmethod
    def _reconcile_meta(con: sqlite3.Connection, *, embedder: Any) -> None:
        cur = con.cursor()
        stored_model = cur.execute(
            "SELECT value FROM meta WHERE key='embedding_model'"
        ).fetchone()
        stored_dims = cur.execute(
            "SELECT value FROM meta WHERE key='embedding_dims'"
        ).fetchone()

        if stored_model is None:
            cur.execute(
                "INSERT INTO meta (key, value) VALUES ('embedding_model', ?)",
                (embedder.model_name,),
            )
            cur.execute(
                "INSERT INTO meta (key, value) VALUES ('embedding_dims', ?)",
                (str(embedder.dimensions),),
            )
            con.commit()
            return

        if stored_model["value"] != embedder.model_name:
            raise OutmemError(
                f"vector DB was built with embedding model "
                f"{stored_model['value']!r} but the configured model is "
                f"{embedder.model_name!r}. Run `outmem reindex --force` "
                f"to rebuild against the new model (or revert the config)."
            )
        if int(stored_dims["value"]) != embedder.dimensions:
            raise OutmemError(
                f"vector DB embedding dims {stored_dims['value']} do not "
                f"match the embedder's {embedder.dimensions}. Run "
                f"`outmem reindex --force` to rebuild."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reindex_file(
        self,
        rel_path: str,
        *,
        body: str,
        kind: Literal["wiki", "source"],
        chunk_size: int = 2000,
        chunk_max: int = 8000,
        overlap_paragraphs: int = 1,
    ) -> ReindexResult:
        """Re-index a single file.

        If ``hash_text(body)`` matches the stored ``content_hash`` for
        ``rel_path``, returns ``ReindexResult(skipped=True)`` without
        touching the DB. Otherwise deletes every chunk for the file,
        re-chunks, re-embeds, and inserts the new chunks.

        Embedding happens *before* any DB mutation so a network failure
        leaves the index untouched. DB writes are then wrapped in a
        single transaction that rolls back on failure — without the
        rollback, the connection's open transaction would be implicitly
        committed by the next caller and silently merge half-written
        state across files.
        """
        prepared = self._prepare(rel_path, body, chunk_size, chunk_max, overlap_paragraphs)
        if prepared is None:  # content_hash matched → nothing to do
            return ReindexResult(rel_path, skipped=True, chunks_removed=0,
                                 chunks_added=0, embeddings_called=0)
        content_hash, chunks = prepared
        # Embed BEFORE any DB writes — if this raises, no transaction is
        # open and the index is unchanged.
        vectors = self.embedder.embed_documents([c.text for c in chunks]) if chunks else []
        return self._commit_file(rel_path, content_hash, kind, chunks, vectors)

    def _prepare(
        self, rel_path: str, body: str, chunk_size: int, chunk_max: int,
        overlap_paragraphs: int,
    ) -> tuple[str, list[Chunk]] | None:
        """Hash-check + chunk. Returns ``(content_hash, chunks)`` to embed,
        or ``None`` when the stored hash matches (skip). Read-only on the DB."""
        content_hash = hash_text(body)
        with self._lock:
            existing = self.con.execute(
                "SELECT content_hash FROM files WHERE rel_path = ?", (rel_path,)
            ).fetchone()
        if existing is not None and existing["content_hash"] == content_hash:
            return None
        chunks = chunk_text(
            body, chunk_size=chunk_size, chunk_max=chunk_max,
            overlap_paragraphs=overlap_paragraphs,
        )
        return content_hash, chunks

    def _commit_file(
        self, rel_path: str, content_hash: str, kind: Literal["wiki", "source"],
        chunks: list[Chunk], vectors: list[list[float]],
    ) -> ReindexResult:
        """Serial transactional write of one file's chunks+vectors. The
        single shared connection means this MUST stay serial across files
        — interleaving open transactions corrupts the index."""
        with self._lock:
            cur = self.con.cursor()
            try:
                removed = self._delete_file_locked(cur, rel_path)
                added = 0
                if chunks:
                    self._insert_chunks(cur, rel_path=rel_path, chunks=chunks, vectors=vectors)
                    added = len(chunks)
                cur.execute(
                    "INSERT OR REPLACE INTO files "
                    "(rel_path, content_hash, indexed_at, chunks_count, kind) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (rel_path, content_hash, _now_iso(), added, kind),
                )
                self.con.commit()
            except Exception:
                self.con.rollback()
                raise
            return ReindexResult(
                rel_path=rel_path, skipped=False, chunks_removed=removed,
                chunks_added=added, embeddings_called=len(chunks),
            )

    def reindex_files(
        self,
        files: list[tuple[str, str, Literal["wiki", "source"]]],
        *,
        chunk_size: int = 2000,
        chunk_max: int = 8000,
        overlap_paragraphs: int = 1,
        max_concurrency: int = DEFAULT_SEMANTIC_REINDEX_CONCURRENCY,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> list[ReindexResult]:
        """Re-index a batch of ``(rel_path, body, kind)`` files.

        Three phases: serial hash-check + chunk (read-only), then EMBED in
        parallel (the network bottleneck — at most ``max_concurrency`` in
        flight), then SERIAL transactional writes (the single shared sqlite
        connection forbids concurrent writers). ``on_progress(done, total)``
        fires as each file's write completes. Skipped (hash-match) files do
        no embedding and commit nothing.
        """
        total = len(files)
        # Phase 1 — prepare (serial, read-only): drop hash-matches up front
        # so we never embed unchanged files.
        prepared: list[tuple[str, str, Literal["wiki", "source"], list[Chunk]]] = []
        results: list[ReindexResult] = []
        done = 0
        for rel_path, body, kind in files:
            p = self._prepare(rel_path, body, chunk_size, chunk_max, overlap_paragraphs)
            if p is None:
                results.append(ReindexResult(rel_path, skipped=True, chunks_removed=0,
                                             chunks_added=0, embeddings_called=0))
                done += 1
                if on_progress:
                    on_progress(done, total)
            else:
                content_hash, chunks = p
                prepared.append((rel_path, content_hash, kind, chunks))

        if prepared:
            vectors_by_path = asyncio.run(self._embed_batch(prepared, max_concurrency))
            # Phase 3 — serial writes. Per-file embed failures (from phase 2)
            # arrive as exceptions in vectors_by_path; record an error result
            # and skip the commit so other files still land.
            for rel_path, content_hash, kind, chunks in prepared:
                vectors = vectors_by_path[rel_path]
                if isinstance(vectors, BaseException):
                    log.warning(
                        "reindex: embed failed for %s (%s); skipping commit",
                        rel_path, vectors,
                    )
                    results.append(ReindexResult(
                        rel_path, skipped=True, chunks_removed=0,
                        chunks_added=0, embeddings_called=0,
                    ))
                else:
                    results.append(
                        self._commit_file(rel_path, content_hash, kind, chunks, vectors)
                    )
                done += 1
                if on_progress:
                    on_progress(done, total)
        return results

    async def _embed_batch(
        self,
        prepared: list[tuple[str, str, Literal["wiki", "source"], list[Chunk]]],
        max_concurrency: int,
    ) -> dict[str, list[list[float]] | BaseException]:
        """Embed every prepared file's chunks concurrently (≤ ``max_concurrency``
        in flight), keyed by rel_path. No DB access — pure network.

        Per-file errors are CAUGHT and returned as the value (instead of the
        vectors), so one transient 429/timeout doesn't abort the batch and
        discard work already done by other files. Callers inspect the value
        type to distinguish success from failure."""
        sem = asyncio.Semaphore(max(1, max_concurrency))
        out: dict[str, list[list[float]] | BaseException] = {}

        async def _one(rel_path: str, chunks: list[Chunk]) -> None:
            if not chunks:
                out[rel_path] = []
                return
            async with sem:
                try:
                    out[rel_path] = await self.embedder.embed_documents_async(
                        [c.text for c in chunks]
                    )
                except BaseException as exc:
                    out[rel_path] = exc

        await asyncio.gather(*(_one(rp, ch) for rp, _, _, ch in prepared))
        return out

    def remove_file(self, rel_path: str) -> int:
        """Drop every chunk + vector for ``rel_path``. Returns count removed."""
        with self._lock:
            cur = self.con.cursor()
            try:
                removed = self._delete_file_locked(cur, rel_path)
                cur.execute("DELETE FROM files WHERE rel_path = ?", (rel_path,))
                self.con.commit()
            except Exception:
                self.con.rollback()
                raise
            return removed

    def find_similar(
        self,
        text: str,
        *,
        top_k: int = 5,
        threshold: float = 0.0,
        exclude_rel_path: str | None = None,
    ) -> list[Match]:
        """Return the top-``top_k`` matches above ``threshold`` similarity.

        Similarity is ``1.0 - cosine_distance``, so a perfect match
        scores ``1.0``. ``exclude_rel_path`` is honoured so the agent
        can ask "what's similar to this page *other than this page
        itself*" during a write.
        """
        if not text.strip():
            return []
        # Embed OUTSIDE the lock — embed_query may take a network round-trip;
        # holding the DB lock across it would serialize the whole batch.
        vector = self.embedder.embed_query(text)
        with self._lock:
            cur = self.con.cursor()
            # vec0 KNN returns rowid + distance; join to chunks for content.
            rows = cur.execute(
                "SELECT chunks_vec.rowid AS id, chunks_vec.distance AS distance, "
                "chunks.rel_path, chunks.chunk_index, chunks.content, "
                "chunks.start_char, chunks.end_char "
                "FROM chunks_vec "
                "JOIN chunks ON chunks.id = chunks_vec.rowid "
                "WHERE chunks_vec.embedding MATCH ? AND k = ? "
                "ORDER BY chunks_vec.distance",
                # Over-fetch by 3x so post-filtering (exclude_rel_path,
                # threshold) still leaves us with up to top_k results when
                # the excluded page or low-similarity rows dominate the
                # head. Pathological case: the excluded page has >2*top_k
                # near-identical chunks — we'd return fewer than top_k,
                # which the caller can detect by checking len(matches).
                (_pack(vector), top_k * 3),
            ).fetchall()

        out: list[Match] = []
        for row in rows:
            if exclude_rel_path is not None and row["rel_path"] == exclude_rel_path:
                continue
            similarity = 1.0 - float(row["distance"])
            if similarity < threshold:
                continue
            out.append(
                Match(
                    rel_path=row["rel_path"],
                    chunk_index=row["chunk_index"],
                    similarity=similarity,
                    content=row["content"],
                    start_char=row["start_char"],
                    end_char=row["end_char"],
                )
            )
            if len(out) >= top_k:
                break
        return out

    def list_indexed_files(self) -> list[tuple[str, str, str]]:
        """Return ``[(rel_path, content_hash, kind)]`` for every indexed file."""
        with self._lock:
            cur = self.con.cursor()
            rows = cur.execute(
                "SELECT rel_path, content_hash, kind FROM files ORDER BY rel_path"
            ).fetchall()
            return [(r["rel_path"], r["content_hash"], r["kind"]) for r in rows]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _delete_file_locked(self, cur: sqlite3.Cursor, rel_path: str) -> int:
        """Remove the chunks + vectors for ``rel_path``. Caller commits."""
        ids = [
            row["id"]
            for row in cur.execute(
                "SELECT id FROM chunks WHERE rel_path = ?", (rel_path,)
            ).fetchall()
        ]
        if not ids:
            return 0
        placeholders = ",".join("?" * len(ids))
        cur.execute(f"DELETE FROM chunks_vec WHERE rowid IN ({placeholders})", ids)
        cur.execute(f"DELETE FROM chunks WHERE id IN ({placeholders})", ids)
        return len(ids)

    def _insert_chunks(
        self,
        cur: sqlite3.Cursor,
        *,
        rel_path: str,
        chunks: Iterable[Chunk],
        vectors: Iterable[list[float]],
    ) -> None:
        for chunk, vector in zip(chunks, vectors, strict=True):
            cur.execute(
                "INSERT INTO chunks "
                "(rel_path, chunk_index, start_char, end_char, content, content_hash) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    rel_path,
                    chunk.index,
                    chunk.start_char,
                    chunk.end_char,
                    chunk.text,
                    chunk.content_hash,
                ),
            )
            chunk_id = cur.lastrowid
            cur.execute(
                "INSERT INTO chunks_vec (rowid, embedding) VALUES (?, ?)",
                (chunk_id, _pack(vector)),
            )


def _pack(vector: list[float]) -> bytes:
    """Pack a Python float list into sqlite-vec's expected float32 bytes."""
    return struct.pack(f"{len(vector)}f", *vector)


def _now_iso() -> str:
    return format_iso_z(utc_now())
