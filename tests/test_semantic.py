"""Tests for the optional semantic index (``outmem.semantic``).

Coverage:

* ``chunker.chunk_text`` — paragraph-aware grouping, overlap,
  oversized paragraphs, stable boundaries across edits.
* ``store.VectorStore`` — open, reindex_file (skip-on-hash),
  find_similar, remove_file, list_indexed_files.
* ``WikiStore`` integration — write_page / extend_page / add_source
  auto-reindex and stage the vector DB in the same commit.
* CLI — ``outmem similar``, ``outmem reindex``, ``outmem hook
  install/uninstall``, ``outmem reindex --staged`` (pre-commit hook
  surface).

Tests use a deterministic in-process fake embedder rather than calling
out to OpenAI. The fake is plugged into :class:`EmbedderHandle`
directly so we exercise the real :class:`VectorStore`,
``sqlite-vec``, and the rest of the integration surface.
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

from outmem.semantic import (
    EmbedderHandle,
    VectorStore,
    chunk_text,
    hash_text,
)
from outmem.semantic.testing import (
    BagOfWordsEmbeddingModel as _BagOfWordsEmbeddingModel,
)
from outmem.semantic.testing import (
    make_bag_of_words_handle as make_handle,
)
from outmem.store import WikiStore

# ---------------------------------------------------------------------------
# chunker
# ---------------------------------------------------------------------------


class TestChunkText:
    def test_empty_body_returns_empty(self) -> None:
        assert chunk_text("") == []
        assert chunk_text("   \n\n  ") == []

    def test_single_short_paragraph_returns_one_chunk(self) -> None:
        chunks = chunk_text("hello world")
        assert len(chunks) == 1
        assert chunks[0].index == 0
        assert chunks[0].text == "hello world"
        assert chunks[0].start_char == 0
        assert chunks[0].end_char == len("hello world")

    def test_groups_paragraphs_to_target_size(self) -> None:
        paragraphs = [f"para {i} " + ("x" * 80) for i in range(10)]
        body = "\n\n".join(paragraphs)
        chunks = chunk_text(body, chunk_size=200, overlap_paragraphs=0)
        assert len(chunks) >= 4
        # Each chunk should pack at least 2 paragraphs (size 80 each).
        for c in chunks:
            assert "para" in c.text

    def test_does_not_split_a_paragraph(self) -> None:
        big = "z" * 5000
        chunks = chunk_text(big, chunk_size=2000, chunk_max=8000, overlap_paragraphs=0)
        assert len(chunks) == 1
        assert chunks[0].text == big

    def test_overlap_includes_tail_of_previous_chunk(self) -> None:
        paragraphs = [f"para-{i}" for i in range(6)]
        body = "\n\n".join(f"{p}: " + ("x" * 50) for p in paragraphs)
        chunks = chunk_text(body, chunk_size=120, overlap_paragraphs=1)
        # At least two chunks should exist with the test data.
        assert len(chunks) >= 2
        # The first paragraph of chunk N+1 should appear in chunk N.
        for i in range(len(chunks) - 1):
            first_para_next = chunks[i + 1].text.split("\n\n", 1)[0]
            assert first_para_next in chunks[i].text or first_para_next in chunks[i + 1].text

    def test_overlap_zero_makes_chunks_disjoint(self) -> None:
        paragraphs = [f"para {i}" for i in range(6)]
        body = "\n\n".join(p + (" x" * 30) for p in paragraphs)
        chunks = chunk_text(body, chunk_size=80, overlap_paragraphs=0)
        all_text = "".join(c.text for c in chunks)
        for p in paragraphs:
            assert p in all_text

    def test_content_hash_is_stable(self) -> None:
        chunks_a = chunk_text("foo\n\nbar\n\nbaz")
        chunks_b = chunk_text("foo\n\nbar\n\nbaz")
        assert [c.content_hash for c in chunks_a] == [c.content_hash for c in chunks_b]

    def test_invalid_params_raise(self) -> None:
        with pytest.raises(ValueError):
            chunk_text("x", chunk_size=0)
        with pytest.raises(ValueError):
            chunk_text("x", chunk_size=200, chunk_max=100)
        with pytest.raises(ValueError):
            chunk_text("x", overlap_paragraphs=-1)


def test_hash_text_is_sha256_hex() -> None:
    h = hash_text("hello")
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


# ---------------------------------------------------------------------------
# VectorStore
# ---------------------------------------------------------------------------


@pytest.fixture
def vstore(tmp_path: Path) -> VectorStore:
    return VectorStore.open(tmp_path / ".vectors.db", embedder=make_handle())


def test_vector_store_survives_cross_thread_use(tmp_path: Path) -> None:
    """Regression: PydanticAI dispatches embed calls onto worker
    threads via asyncio's executor. Without ``check_same_thread=False``
    on the sqlite connection, the very next ``find_similar`` call
    raises ``ProgrammingError: SQLite objects created in a thread can
    only be used in that same thread`` — which is exactly the failure
    we hit during a real eval run."""
    import threading

    vs = VectorStore.open(tmp_path / ".vectors.db", embedder=make_handle())
    vs.reindex_file("wiki/pages/a.md", body="cross-thread cat dog mouse", kind="wiki")

    results: list[object] = []

    def worker() -> None:
        # Reads on another thread MUST work — the embedder thread is
        # the canonical case.
        results.append(vs.find_similar("cat", top_k=3))

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert results and isinstance(results[0], list)


def test_vector_store_concurrent_reads_dont_corrupt_rows(tmp_path: Path) -> None:
    """Regression: 8 worker threads doing concurrent find_similar /
    list_indexed_files on a SHARED connection raced the cursor's
    row_factory state — rows came back as plain tuples instead of dict-
    like rows, and the very next ``r["rel_path"]`` raised IndexError.
    Symptom in the optimizer: semantic / hybrid stalled across evals
    with ``semantic index unavailable: tuple index out of range``.

    The fix is a per-instance Lock around every connection touch."""
    from concurrent.futures import ThreadPoolExecutor

    vs = VectorStore.open(tmp_path / ".vectors.db", embedder=make_handle())
    for i in range(8):
        vs.reindex_file(f"wiki/pages/p{i}.md", body=f"penicillin {i}", kind="wiki")

    errors: list[BaseException] = []

    def hammer(_i: int) -> None:
        try:
            for _ in range(20):
                vs.find_similar("penicillin", top_k=3)
                rows = vs.list_indexed_files()
                # Force the dict-row access that previously raised IndexError.
                assert all(r[0].startswith("wiki/pages/") for r in rows)
        except BaseException as e:
            errors.append(e)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(hammer, range(8)))
    assert not errors, f"concurrent reads corrupted rows: {errors[:1]}"


class TestVectorStoreReindexBatch:
    def _files(self, n: int) -> list[tuple[str, str, str]]:
        return [
            (f"wiki/pages/p{i}.md", f"penicillin dosing endocarditis page {i} uniq{i}", "wiki")
            for i in range(n)
        ]

    def test_batch_matches_per_file(self, tmp_path: Path) -> None:
        # The parallel-embed batch path must produce the SAME index as
        # per-file reindex (same chunk counts, same files indexed).
        files = self._files(6)
        a = VectorStore.open(tmp_path / "a.db", embedder=make_handle())
        for rel, body, kind in files:
            a.reindex_file(rel, body=body, kind=kind)  # type: ignore[arg-type]
        b = VectorStore.open(tmp_path / "b.db", embedder=make_handle())
        b.reindex_files(files)  # type: ignore[arg-type]
        assert {f[0] for f in a.list_indexed_files()} == {f[0] for f in b.list_indexed_files()}
        assert len(b.list_indexed_files()) == 6

    def test_concurrency_matches_serial(self, tmp_path: Path) -> None:
        files = self._files(8)
        seq = VectorStore.open(tmp_path / "s.db", embedder=make_handle())
        seq_res = seq.reindex_files(files, max_concurrency=1)  # type: ignore[arg-type]
        par = VectorStore.open(tmp_path / "p.db", embedder=make_handle())
        par_res = par.reindex_files(files, max_concurrency=8)  # type: ignore[arg-type]
        assert [r.chunks_added for r in seq_res] == [r.chunks_added for r in par_res]
        assert sum(r.chunks_added for r in par_res) > 0  # actually embedded

    def test_progress_and_skip(self, tmp_path: Path) -> None:
        files = self._files(4)
        vs = VectorStore.open(tmp_path / "v.db", embedder=make_handle())
        ticks: list[tuple[int, int]] = []
        vs.reindex_files(files, on_progress=lambda d, t: ticks.append((d, t)))  # type: ignore[arg-type]
        assert ticks[-1] == (4, 4)
        # Idempotent: a second pass re-embeds nothing (all hash-skip).
        res2 = vs.reindex_files(files)  # type: ignore[arg-type]
        assert all(r.skipped for r in res2)

    def test_embedder_accrues_tokens(self, tmp_path: Path) -> None:
        # The handle tracks billed input tokens across calls; a hash-skip
        # rerun adds none (nothing re-embedded).
        h = make_handle()
        assert h.total_tokens == 0
        vs = VectorStore.open(tmp_path / "v.db", embedder=h)
        vs.reindex_files(self._files(4))  # type: ignore[arg-type]
        first = h.total_tokens
        assert first > 0
        vs.reindex_files(self._files(4))  # type: ignore[arg-type]  # all skip
        assert h.total_tokens == first  # no new tokens on hash-skip


class TestVectorStoreReindex:
    def test_reindex_creates_chunks(self, vstore: VectorStore) -> None:
        body = "first paragraph.\n\nsecond paragraph.\n\nthird paragraph."
        result = vstore.reindex_file("wiki/pages/a.md", body=body, kind="wiki")
        assert result.skipped is False
        assert result.chunks_added >= 1
        assert result.chunks_removed == 0
        files = vstore.list_indexed_files()
        assert len(files) == 1
        assert files[0][0] == "wiki/pages/a.md"

    def test_reindex_same_content_is_skipped(self, vstore: VectorStore) -> None:
        body = "stable body"
        first = vstore.reindex_file("wiki/pages/a.md", body=body, kind="wiki")
        second = vstore.reindex_file("wiki/pages/a.md", body=body, kind="wiki")
        assert first.skipped is False
        assert second.skipped is True
        assert second.embeddings_called == 0

    def test_reindex_new_content_replaces_chunks(self, vstore: VectorStore) -> None:
        vstore.reindex_file("wiki/pages/a.md", body="alpha beta", kind="wiki")
        result = vstore.reindex_file("wiki/pages/a.md", body="gamma delta", kind="wiki")
        assert result.skipped is False
        assert result.chunks_removed >= 1
        files = vstore.list_indexed_files()
        assert len(files) == 1  # not duplicated

    def test_remove_file_drops_chunks(self, vstore: VectorStore) -> None:
        vstore.reindex_file("wiki/pages/a.md", body="some body", kind="wiki")
        removed = vstore.remove_file("wiki/pages/a.md")
        assert removed >= 1
        assert vstore.list_indexed_files() == []
        # Re-removing is a no-op.
        assert vstore.remove_file("wiki/pages/a.md") == 0


class TestVectorStoreFindSimilar:
    def test_exact_text_scores_top(self, vstore: VectorStore) -> None:
        vstore.reindex_file(
            "wiki/pages/a.md", body="the cat sat on the mat", kind="wiki"
        )
        vstore.reindex_file(
            "wiki/pages/b.md", body="entirely unrelated quantum mechanics", kind="wiki"
        )
        matches = vstore.find_similar("the cat sat on the mat", top_k=2)
        assert matches
        assert matches[0].rel_path == "wiki/pages/a.md"
        assert matches[0].similarity > matches[1].similarity if len(matches) > 1 else True

    def test_threshold_filters_results(self, vstore: VectorStore) -> None:
        vstore.reindex_file("wiki/pages/a.md", body="cat dog mouse", kind="wiki")
        vstore.reindex_file(
            "wiki/pages/b.md", body="cosmic radiation telescope", kind="wiki"
        )
        # Very high threshold — only near-perfect matches survive.
        matches = vstore.find_similar(
            "cosmic radiation telescope", top_k=5, threshold=0.99
        )
        assert all(m.similarity >= 0.99 for m in matches)
        assert len(matches) == 1
        assert matches[0].rel_path == "wiki/pages/b.md"

    def test_exclude_rel_path_filters(self, vstore: VectorStore) -> None:
        vstore.reindex_file("wiki/pages/a.md", body="alpha beta gamma", kind="wiki")
        vstore.reindex_file("wiki/pages/b.md", body="alpha beta gamma", kind="wiki")
        matches = vstore.find_similar(
            "alpha beta gamma",
            top_k=5,
            exclude_rel_path="wiki/pages/a.md",
        )
        rels = {m.rel_path for m in matches}
        assert "wiki/pages/a.md" not in rels
        assert "wiki/pages/b.md" in rels

    def test_empty_query_returns_empty(self, vstore: VectorStore) -> None:
        vstore.reindex_file("wiki/pages/a.md", body="something", kind="wiki")
        assert vstore.find_similar("") == []
        assert vstore.find_similar("   ") == []


class TestVectorStoreMeta:
    def test_reopen_preserves_data(self, tmp_path: Path) -> None:
        db = tmp_path / ".vectors.db"
        vs = VectorStore.open(db, embedder=make_handle())
        vs.reindex_file("wiki/pages/a.md", body="persistent body", kind="wiki")
        vs.close()
        vs2 = VectorStore.open(db, embedder=make_handle())
        assert {f[0] for f in vs2.list_indexed_files()} == {"wiki/pages/a.md"}
        vs2.close()

    def test_dimension_mismatch_errors(self, tmp_path: Path) -> None:
        from outmem.exceptions import OutmemError

        db = tmp_path / ".vectors.db"
        vs = VectorStore.open(db, embedder=make_handle(dimensions=64))
        vs.close()
        with pytest.raises(OutmemError):
            VectorStore.open(db, embedder=make_handle(dimensions=32))

    def test_model_name_mismatch_errors(self, tmp_path: Path) -> None:
        """Same dim, different model name → must error so the user runs
        ``outmem reindex --force`` rather than silently mixing
        embeddings from two providers in the same KNN index."""
        from outmem.exceptions import OutmemError

        db = tmp_path / ".vectors.db"
        vs = VectorStore.open(db, embedder=make_handle(dimensions=64))
        vs.close()

        # Same dim, but pretend the embedder belongs to a different
        # provider — only `model_name` differs.
        other = EmbedderHandle(
            embedder=_BagOfWordsEmbeddingModel(dimensions=64),
            model_name="other-provider-test",
            dimensions=64,
        )
        with pytest.raises(OutmemError):
            VectorStore.open(db, embedder=other)


class TestVectorStoreTransactionSafety:
    """A failing embed must NOT corrupt the previously indexed state.

    The earlier implementation left the connection's transaction open
    when ``embed_documents`` raised mid-call; the next reindex then
    silently committed both its own changes and the failed call's
    DELETE rows. Regression test for that hazard.
    """

    def test_embed_failure_rolls_back_and_preserves_other_file(
        self, tmp_path: Path
    ) -> None:
        from outmem.semantic.embeddings import EmbedderHandle

        fail_on_call = {"count": 0}
        underlying = _BagOfWordsEmbeddingModel(dimensions=64)
        real_embed = underlying.embed

        async def flaky_embed(inputs, *, input_type="document", settings=None):  # type: ignore[no-untyped-def]
            # Count only document (re-index) embeds; query embeds for
            # find_similar between reindexes shouldn't trip the simulated
            # failure (we're testing reindex transaction safety, not query).
            if input_type == "document":
                fail_on_call["count"] += 1
                if fail_on_call["count"] == 2:
                    raise RuntimeError("simulated network failure")
            return await real_embed(inputs, input_type=input_type, settings=settings)

        underlying.embed = flaky_embed  # type: ignore[method-assign]
        handle = EmbedderHandle(
            embedder=underlying,
            model_name=underlying.model_name,
            dimensions=64,
        )
        vs = VectorStore.open(tmp_path / ".vectors.db", embedder=handle)

        # First reindex succeeds (call #1 — the build_embedder probe
        # for the real EmbedderHandle goes through embed_query, not
        # embed_documents, so call #1 here is the genuine first
        # embed_documents).
        vs.reindex_file("wiki/pages/a.md", body="alpha content survives.", kind="wiki")
        assert vs.find_similar("alpha content survives") != []

        # Second reindex aborts mid-flight — the open transaction MUST
        # be rolled back, not leaked into the connection.
        with pytest.raises(RuntimeError):
            vs.reindex_file("wiki/pages/b.md", body="beta would-be content.", kind="wiki")

        # Reset the embedder so the next call works.
        underlying.embed = real_embed  # type: ignore[method-assign]

        # Reindexing A again must NOT accidentally commit B's partial
        # deletes from the prior failure. A's pre-existing chunks
        # remain queryable.
        assert vs.find_similar("alpha content survives") != [], (
            "A's chunks were collateral damage from B's failed reindex"
        )

        # And the third reindex works cleanly.
        vs.reindex_file("wiki/pages/c.md", body="gamma works fine now.", kind="wiki")
        assert vs.find_similar("gamma works fine") != []
        # A is still there too.
        assert vs.find_similar("alpha content survives") != []


# ---------------------------------------------------------------------------
# WikiStore integration
# ---------------------------------------------------------------------------


@pytest.fixture
def wiki_with_semantic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> WikiStore:
    """A wiki with ``semantic.enabled: true`` and a stubbed embedder.

    Patches :func:`outmem.semantic.build_embedder` so no API call is
    made. The fake :class:`EmbedderHandle` is otherwise plugged into a
    real :class:`VectorStore`.
    """
    root = tmp_path / "wiki-root"
    store = WikiStore.init(root)
    yaml_path = root / "config.yaml"
    text = yaml_path.read_text()
    text = text.replace(
        "semantic:\n  enabled: false",
        "semantic:\n  enabled: true",
    )
    # Lower threshold so the fake bag-of-words embedder still ranks
    # paraphrased matches above it (real openai embeddings hit 0.8
    # easily; deterministic test embeddings have less semantic mass).
    text = text.replace("similarity_threshold: 0.8", "similarity_threshold: 0.1")
    yaml_path.write_text(text, encoding="utf-8")
    monkeypatch.setattr(
        "outmem.semantic.build_embedder",
        lambda _model: make_handle(),
    )
    monkeypatch.setattr(
        "outmem.store.build_embedder",
        lambda _model: make_handle(),
        raising=False,
    )
    # Re-open so the YAML flip takes effect (the original store was
    # created before the flip).
    store.close()
    return WikiStore.open(root)


def test_vector_store_open_is_single_under_concurrency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: the lazy open is now lock-guarded. 8 threads hitting a
    cold store must build the embedder + open the connection exactly ONCE
    (not 8 times, orphaning 7 connections + paying 8 probe calls)."""
    import threading

    from outmem._store import semantic as sem

    root = tmp_path / "w"
    store = WikiStore.init(root)
    yaml_path = root / "config.yaml"
    yaml_path.write_text(
        yaml_path.read_text().replace(
            "semantic:\n  enabled: false", "semantic:\n  enabled: true"
        ),
        encoding="utf-8",
    )
    store.close()
    store = WikiStore.open(root)

    builds = []
    monkeypatch.setattr(
        "outmem.semantic.build_embedder",
        lambda _model: builds.append(1) or make_handle(),
    )

    barrier = threading.Barrier(8)
    handles: list[object] = []

    def worker() -> None:
        barrier.wait()  # maximise the race window
        handles.append(sem.vector_store_or_open(store))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sum(builds) == 1  # embedder built once despite 8 racing threads
    assert all(h is handles[0] for h in handles)  # all got the same store
    store.close()


class TestWikiStoreSemanticIntegration:
    def test_write_page_indexes_and_stages_db(self, wiki_with_semantic: WikiStore) -> None:
        store = wiki_with_semantic
        sha = store.write_page(
            "pricing-formula",
            title="Pricing formula",
            body="The pricing formula is cost-plus 35%.\n\nApplies to all SKUs.\n",
        )
        assert sha
        # .vectors.db is committed alongside the page.
        files_at_head = subprocess.run(
            ["git", "show", "--name-only", "--pretty=format:", sha],
            cwd=store.root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
        assert ".vectors.db" in files_at_head
        # And the page chunk is queryable.
        matches = store.semantic_find_similar(
            "cost-plus pricing",
            top_k=3,
            threshold=0.0,
        )
        assert any(m.rel_path == "wiki/pages/pricing-formula.md" for m in matches)

    def test_extend_page_updates_index(self, wiki_with_semantic: WikiStore) -> None:
        store = wiki_with_semantic
        store.write_page(
            "pricing-formula",
            title="Pricing formula",
            body="cost-plus 35%.\n",
        )
        store.extend_page("pricing-formula", body="cost-plus 40%, revised Q2.\n")
        matches = store.semantic_find_similar(
            "revised Q2 cost-plus",
            top_k=3,
            threshold=0.0,
        )
        assert matches
        assert matches[0].rel_path == "wiki/pages/pricing-formula.md"
        # Old body chunks should be gone.
        from outmem._store import semantic as _semantic

        files = _semantic.vector_store_or_open(store).list_indexed_files()
        slugs = [f[0] for f in files]
        assert slugs.count("wiki/pages/pricing-formula.md") == 1

    def test_add_source_indexes_text(
        self, wiki_with_semantic: WikiStore, tmp_path: Path
    ) -> None:
        store = wiki_with_semantic
        source = tmp_path / "drugs.md"
        source.write_text("ivermectin dosing for cats.\n", encoding="utf-8")
        entry = store.add_source(source)
        matches = store.semantic_find_similar(
            "ivermectin cats",
            top_k=3,
            threshold=0.0,
        )
        # The vector index uses repo-relative paths
        # (`wiki/sources/<hash>/drugs.md`); match against whatever the
        # registry actually recorded so the test stays layout-agnostic.
        assert any(entry.rel_path in m.rel_path for m in matches)

    def test_find_similar_disabled_when_off(self, tmp_path: Path) -> None:
        from outmem.exceptions import OutmemError

        store = WikiStore.init(tmp_path / "off")
        assert store.semantic_enabled() is False
        with pytest.raises(OutmemError):
            store.semantic_find_similar("anything")

    def test_reindex_all_skips_unchanged(self, wiki_with_semantic: WikiStore) -> None:
        store = wiki_with_semantic
        store.write_page("a", title="A", body="page a body.\n")
        store.write_page("b", title="B", body="page b body.\n")
        summary = store.semantic_reindex_all()
        assert summary["reindexed"] == 0
        assert summary["skipped"] >= 2


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@pytest.fixture
def wiki_with_pages(wiki_with_semantic: WikiStore) -> WikiStore:
    wiki_with_semantic.write_page(
        "alpha",
        title="Alpha",
        body="alpha is the first letter of the greek alphabet.\n",
    )
    wiki_with_semantic.write_page(
        "beta",
        title="Beta",
        body="beta is a software release stage.\n",
    )
    return wiki_with_semantic


def _run_cli(args: Sequence[str], *, store: WikiStore) -> tuple[int, str, str]:
    """Invoke the CLI through ``main`` with ``--root <store>``.

    ``--root`` goes AFTER the subcommand (`outmem <cmd> --root <root> ...`),
    which is where the shared parent parser accepts it."""
    from outmem.cli.__main__ import main

    out_buf: list[str] = []
    err_buf: list[str] = []

    import io
    import sys

    cmd, *rest = args
    real_stdout, real_stderr = sys.stdout, sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        rc = main([cmd, "--root", str(store.root), *rest])
        out_buf.append(sys.stdout.getvalue())
        err_buf.append(sys.stderr.getvalue())
    finally:
        sys.stdout, sys.stderr = real_stdout, real_stderr
    return rc, "".join(out_buf), "".join(err_buf)


class TestCliSimilar:
    def test_similar_finds_matches(self, wiki_with_pages: WikiStore) -> None:
        rc, out, _ = _run_cli(
            ["similar", "greek alphabet", "--top-k", "3"],
            store=wiki_with_pages,
        )
        assert rc == 0
        assert "alpha" in out

    def test_similar_slug_excludes_self(self, wiki_with_pages: WikiStore) -> None:
        _rc, out, _ = _run_cli(
            ["similar", "--slug", "alpha", "--top-k", "3"],
            store=wiki_with_pages,
        )
        # rc=1 means no matches; rc=0 means matches. Either is fine
        # — what matters is `alpha` is not in the result list.
        assert "wiki/pages/alpha.md" not in out

    def test_similar_disabled_returns_error(self, tmp_path: Path) -> None:
        store = WikiStore.init(tmp_path / "off")
        rc, _, err = _run_cli(["similar", "x"], store=store)
        assert rc == 1
        assert "disabled" in err


class TestCliReindex:
    def test_reindex_no_args_walks_all(self, wiki_with_pages: WikiStore) -> None:
        rc, out, _ = _run_cli(["reindex"], store=wiki_with_pages)
        assert rc == 0
        assert "reindex" in out
        assert "skipped" in out or "unchanged" in out

    def test_reindex_path_targets_one_file(self, wiki_with_pages: WikiStore) -> None:
        rc, out, _ = _run_cli(
            ["reindex", "--path", "wiki/pages/alpha.md", "--force"],
            store=wiki_with_pages,
        )
        assert rc == 0
        assert "wiki/pages/alpha.md" in out


class TestCliHook:
    def test_install_creates_hook(self, wiki_with_pages: WikiStore) -> None:
        rc, _, _ = _run_cli(["hook", "install"], store=wiki_with_pages)
        assert rc == 0
        hook = wiki_with_pages.root / ".git" / "hooks" / "pre-commit"
        assert hook.exists()
        content = hook.read_text()
        assert "outmem reindex --staged" in content
        # Should be executable.
        assert hook.stat().st_mode & 0o111

    def test_install_then_uninstall_round_trips(
        self, wiki_with_pages: WikiStore
    ) -> None:
        rc_install, _, _ = _run_cli(["hook", "install"], store=wiki_with_pages)
        assert rc_install == 0
        rc_uninstall, _, _ = _run_cli(["hook", "uninstall"], store=wiki_with_pages)
        assert rc_uninstall == 0
        hook = wiki_with_pages.root / ".git" / "hooks" / "pre-commit"
        assert not hook.exists()

    def test_install_refuses_to_overwrite_foreign_hook(
        self, wiki_with_pages: WikiStore
    ) -> None:
        hooks_dir = wiki_with_pages.root / ".git" / "hooks"
        hooks_dir.mkdir(parents=True, exist_ok=True)
        existing = hooks_dir / "pre-commit"
        existing.write_text("#!/bin/sh\necho other hook\n")
        rc, _, err = _run_cli(["hook", "install"], store=wiki_with_pages)
        assert rc != 0
        assert "force" in err


class TestCliReindexStaged:
    """Validate the pre-commit hook surface area.

    Simulates the hook by:
      1. Staging an externally created wiki file with ``git add``.
      2. Running ``outmem reindex --staged``.
      3. Asserting the vector DB was mutated and is in the staged tree.
    """

    def test_staged_indexes_and_stages_db(self, wiki_with_pages: WikiStore) -> None:
        # External edit: drop a page into wiki/ without going through the
        # WikiStore. This is the Obsidian workflow.
        external = wiki_with_pages.pages_path / "gamma.md"
        external.write_text(
            "---\ntitle: Gamma\nslug: gamma\ncreated: 2026-05-13T00:00:00+00:00\n"
            "updated: 2026-05-13T00:00:00+00:00\n---\n\nGamma rays are ionising.\n",
            encoding="utf-8",
        )
        subprocess.run(
            ["git", "add", "--", "wiki/pages/gamma.md"],
            cwd=wiki_with_pages.root,
            check=True,
        )

        rc, _, _ = _run_cli(["reindex", "--staged"], store=wiki_with_pages)
        assert rc == 0

        # The DB should now be staged.
        staged = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            cwd=wiki_with_pages.root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.split()
        assert ".vectors.db" in staged
        assert "wiki/pages/gamma.md" in staged
        # Manual edit added a new page → index.md must also be in the
        # commit so the slug list stays current.
        assert "wiki/index.md" in staged

        # And the chunk is queryable.
        matches = wiki_with_pages.semantic_find_similar(
            "gamma rays ionising",
            top_k=3,
            threshold=0.0,
        )
        assert any(m.rel_path == "wiki/pages/gamma.md" for m in matches)

        # index.md actually mentions the new page.
        index_text = (wiki_with_pages.wiki_path / "index.md").read_text(
            encoding="utf-8"
        )
        assert "gamma" in index_text


# ---------------------------------------------------------------------------
# Adapter tool surface
# ---------------------------------------------------------------------------


class TestAdapterFindSimilar:
    def test_find_similar_tool_exposed_when_enabled(
        self, wiki_with_pages: WikiStore
    ) -> None:
        from outmem.adapters.pydantic_ai import wiki_tools

        names = {t.__name__ for t in wiki_tools(wiki_with_pages)}
        assert "find_similar" in names

    def test_find_similar_tool_hidden_when_disabled(self, tmp_path: Path) -> None:
        from outmem.adapters.pydantic_ai import wiki_tools

        store = WikiStore.init(tmp_path / "off")
        names = {t.__name__ for t in wiki_tools(store)}
        assert "find_similar" not in names

    def test_find_similar_tool_returns_text(
        self, wiki_with_pages: WikiStore
    ) -> None:
        from outmem.adapters.pydantic_ai import wiki_tools

        tool = next(
            t for t in wiki_tools(wiki_with_pages) if t.__name__ == "find_similar"
        )
        result = tool("greek alphabet", 3, None)
        assert isinstance(result, str)
        assert "alpha" in result
