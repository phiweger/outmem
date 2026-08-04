"""Tests for ``outmem.sources`` — source registry, hashing, ingestion records."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from outmem.exceptions import OutmemError
from outmem.sources import (
    REGISTRY_FILENAME,
    SourceRegistry,
    compute_sha256,
    copy_source,
    is_allowed_source,
    read_source_text,
)
from outmem.store import WikiStore

# ---------------------------------------------------------------------------
# compute_sha256
# ---------------------------------------------------------------------------


def test_sha256_deterministic(tmp_path: Path) -> None:
    f = tmp_path / "x.md"
    f.write_text("hello\n", encoding="utf-8")
    h1 = compute_sha256(f)
    h2 = compute_sha256(f)
    assert h1 == h2
    assert len(h1) == 64


def test_sha256_changes_on_content_change(tmp_path: Path) -> None:
    f = tmp_path / "x.md"
    f.write_text("hello\n", encoding="utf-8")
    h_before = compute_sha256(f)
    f.write_text("changed\n", encoding="utf-8")
    h_after = compute_sha256(f)
    assert h_before != h_after


# ---------------------------------------------------------------------------
# is_allowed_source
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["a.md", "b.txt", "c.csv", "d.json", "e.mmd", "f.yaml", "g.yml", "H.MD"],
)
def test_allowed_extensions(name: str, tmp_path: Path) -> None:
    f = tmp_path / name
    f.touch()
    assert is_allowed_source(f)


@pytest.mark.parametrize("name", ["a.pdf", "b.png", "c.docx", "d.bin"])
def test_disallowed_extensions(name: str, tmp_path: Path) -> None:
    f = tmp_path / name
    f.touch()
    assert not is_allowed_source(f)


# ---------------------------------------------------------------------------
# copy_source
# ---------------------------------------------------------------------------


def test_copy_source_uses_hash_dir(tmp_path: Path) -> None:
    """Layout: ``<sources>/<sha[:12]>/<filename>``."""
    from outmem.sources import SHA_PREFIX_LEN, compute_sha256

    src = tmp_path / "doc.md"
    src.write_text("body\n", encoding="utf-8")
    sources = tmp_path / "sources"
    dest, rel = copy_source(src, sources)
    sha = compute_sha256(src)
    assert dest == sources / sha[:SHA_PREFIX_LEN] / "doc.md"
    assert dest.exists()
    assert rel == f"{sha[:SHA_PREFIX_LEN]}/doc.md"


def test_copy_source_into_subdir_under_hash(tmp_path: Path) -> None:
    """``--into`` lives above the hash dir, not below."""
    from outmem.sources import SHA_PREFIX_LEN, compute_sha256

    src = tmp_path / "doc.md"
    src.write_text("body\n", encoding="utf-8")
    sources = tmp_path / "sources"
    dest, rel = copy_source(src, sources, into_subdir="veterinary")
    sha = compute_sha256(src)
    assert dest == sources / "veterinary" / sha[:SHA_PREFIX_LEN] / "doc.md"
    assert rel == f"veterinary/{sha[:SHA_PREFIX_LEN]}/doc.md"


def test_copy_source_with_rename_under_hash(tmp_path: Path) -> None:
    """``rename`` controls the filename leaf; hash dir is still inserted."""
    from outmem.sources import SHA_PREFIX_LEN, compute_sha256

    src = tmp_path / "doc.md"
    src.write_text("body\n", encoding="utf-8")
    sources = tmp_path / "sources"
    dest, rel = copy_source(src, sources, rename="renamed.md")
    sha = compute_sha256(src)
    assert dest == sources / sha[:SHA_PREFIX_LEN] / "renamed.md"
    assert rel == f"{sha[:SHA_PREFIX_LEN]}/renamed.md"


def test_copy_source_rejects_binary_extension(tmp_path: Path) -> None:
    src = tmp_path / "doc.pdf"
    src.write_bytes(b"%PDF-1.4 ...")
    with pytest.raises(OutmemError, match="disallowed extension"):
        copy_source(src, tmp_path / "sources")


def test_copy_source_rejects_unsafe_subdir(tmp_path: Path) -> None:
    src = tmp_path / "doc.md"
    src.write_text("body\n", encoding="utf-8")
    with pytest.raises(OutmemError, match="unsafe"):
        copy_source(src, tmp_path / "sources", into_subdir="../escape")


def test_copy_source_rejects_unsafe_rename(tmp_path: Path) -> None:
    src = tmp_path / "doc.md"
    src.write_text("body\n", encoding="utf-8")
    with pytest.raises(OutmemError, match="unsafe"):
        copy_source(src, tmp_path / "sources", rename="../escape.md")


def test_copy_source_idempotent_same_content(tmp_path: Path) -> None:
    """Identical content → identical hash dir → no-op."""
    src = tmp_path / "doc.md"
    src.write_text("body\n", encoding="utf-8")
    sources = tmp_path / "sources"
    dest1, rel1 = copy_source(src, sources)
    dest2, rel2 = copy_source(src, sources)
    assert dest1 == dest2
    assert rel1 == rel2


def test_copy_source_same_name_different_content_no_collision(tmp_path: Path) -> None:
    """Two ``document.md`` files with different bodies → two distinct
    hash dirs. This is the bug the layout change was made to fix
    (the user's amikacin / aztreonam Fachinfo collision)."""
    sources = tmp_path / "sources"
    src1 = tmp_path / "a-document.md"
    src1.write_text("amikacin content\n", encoding="utf-8")
    dest1, rel1 = copy_source(src1, sources, rename="document.md")

    src2 = tmp_path / "b-document.md"
    src2.write_text("aztreonam content\n", encoding="utf-8")
    dest2, rel2 = copy_source(src2, sources, rename="document.md")

    assert dest1 != dest2
    assert rel1 != rel2
    assert dest1.exists() and dest2.exists()
    assert dest1.read_text() == "amikacin content\n"
    assert dest2.read_text() == "aztreonam content\n"


# ---------------------------------------------------------------------------
# SourceRegistry
# ---------------------------------------------------------------------------


def test_registry_empty_on_missing_file(tmp_path: Path) -> None:
    sources = tmp_path / "sources"
    sources.mkdir()
    reg = SourceRegistry.load(sources)
    assert reg.entries == {}


def test_registry_register_then_load(tmp_path: Path) -> None:
    sources = tmp_path / "sources"
    sources.mkdir()
    reg = SourceRegistry.load(sources)
    entry = reg.register(
        "veterinary/drugs.md",
        sha256="abc" * 21 + "1",  # 64 chars
        size_bytes=42,
        when=datetime(2026, 5, 12, 10, 0, 0, tzinfo=UTC),
    )
    assert (sources / REGISTRY_FILENAME).exists()

    reloaded = SourceRegistry.load(sources)
    assert "veterinary/drugs.md" in reloaded.entries
    assert reloaded.entries["veterinary/drugs.md"].sha256 == entry.sha256


def test_register_same_hash_returns_existing(tmp_path: Path) -> None:
    sources = tmp_path / "sources"
    sources.mkdir()
    reg = SourceRegistry.load(sources)
    first = reg.register("a.md", sha256="x" * 64, size_bytes=10)
    again = reg.register("a.md", sha256="x" * 64, size_bytes=10)
    assert first is again
    assert first.registered_at == again.registered_at


def test_register_new_hash_refreshes_entry(tmp_path: Path) -> None:
    """When a source's content changes (new sha), the registry refreshes
    the row and drops the old ingestion chain — those entries belonged
    to the old content."""
    sources = tmp_path / "sources"
    sources.mkdir()
    reg = SourceRegistry.load(sources)
    reg.register("a.md", sha256="x" * 64, size_bytes=10)
    reg.record_ingestion("a.md", prompt="first", pages_touched=["p1"])
    assert reg.entries["a.md"].ingestions  # pre-condition: one ingestion logged

    refreshed = reg.register("a.md", sha256="y" * 64, size_bytes=20)
    assert refreshed.sha256 == "y" * 64
    assert refreshed.ingestions == []  # fresh chain


def test_record_ingestion(tmp_path: Path) -> None:
    sources = tmp_path / "sources"
    sources.mkdir()
    reg = SourceRegistry.load(sources)
    reg.register("a.md", sha256="x" * 64, size_bytes=10)
    record = reg.record_ingestion(
        "a.md",
        prompt="extract X",
        pages_touched=["page-x", "page-y"],
    )
    assert record.prompt == "extract X"
    assert record.pages_touched == ("page-x", "page-y")
    assert len(reg.entries["a.md"].ingestions) == 1


def test_record_ingestion_unknown_source_raises(tmp_path: Path) -> None:
    sources = tmp_path / "sources"
    sources.mkdir()
    reg = SourceRegistry.load(sources)
    with pytest.raises(OutmemError, match="not registered"):
        reg.record_ingestion("never.md", prompt=None, pages_touched=[])


def test_multiple_ingestions_appended(tmp_path: Path) -> None:
    """Same source, different prompts — both recorded."""
    sources = tmp_path / "sources"
    sources.mkdir()
    reg = SourceRegistry.load(sources)
    reg.register("a.md", sha256="x" * 64, size_bytes=10)
    reg.record_ingestion("a.md", prompt="cats", pages_touched=["cat-doses"])
    reg.record_ingestion("a.md", prompt="dogs", pages_touched=["dog-doses"])

    reloaded = SourceRegistry.load(sources)
    ingestions = reloaded.entries["a.md"].ingestions
    assert [i.prompt for i in ingestions] == ["cats", "dogs"]
    assert ingestions[0].pages_touched == ("cat-doses",)
    assert ingestions[1].pages_touched == ("dog-doses",)


def test_load_creates_empty_db_when_no_registry_present(tmp_path: Path) -> None:
    sources = tmp_path / "sources"
    sources.mkdir()
    reg = SourceRegistry.load(sources)
    assert reg.entries == {}
    # The DB file should now exist on disk.
    assert (sources / REGISTRY_FILENAME).exists()


def test_parallel_register_does_not_lose_entries(tmp_path: Path) -> None:
    """Two threads each registering 25 distinct sources must end up with
    50 rows on disk. The pre-SQLite JSON registry would have lost
    entries here because two read-modify-write cycles can interleave."""
    import threading

    sources = tmp_path / "sources"
    sources.mkdir()
    # Materialise the DB once so the threads only contend on writes.
    SourceRegistry.load(sources)

    errors: list[BaseException] = []

    def worker(prefix: str) -> None:
        try:
            reg = SourceRegistry.load(sources)
            for i in range(25):
                reg.register(
                    f"{prefix}-{i}.md",
                    sha256=f"{prefix}{i:062d}",
                    size_bytes=i,
                )
        except BaseException as exc:
            errors.append(exc)

    t1 = threading.Thread(target=worker, args=("a",))
    t2 = threading.Thread(target=worker, args=("b",))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert not errors, errors
    reloaded = SourceRegistry.load(sources)
    assert len(reloaded.entries) == 50


# ---------------------------------------------------------------------------
# read_source_text
# ---------------------------------------------------------------------------


def test_read_source_text_returns_content(tmp_path: Path) -> None:
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "x.md").write_text("hello there\n", encoding="utf-8")
    out = read_source_text(sources, "x.md", max_chars=1024)
    assert "hello there" in out


def test_read_source_text_truncates(tmp_path: Path) -> None:
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "x.md").write_text("a" * 500, encoding="utf-8")
    out = read_source_text(sources, "x.md", max_chars=100)
    assert "truncated" in out
    assert len(out) < 500 + 100  # capped plus footer


def test_read_source_text_rejects_path_escape(tmp_path: Path) -> None:
    sources = tmp_path / "sources"
    sources.mkdir()
    outside = tmp_path / "secret.md"
    outside.write_text("not yours\n", encoding="utf-8")
    with pytest.raises(OutmemError, match="escapes"):
        read_source_text(sources, "../secret.md", max_chars=1024)


def test_read_source_text_missing(tmp_path: Path) -> None:
    sources = tmp_path / "sources"
    sources.mkdir()
    with pytest.raises(OutmemError, match="no such source"):
        read_source_text(sources, "ghost.md", max_chars=1024)


# ---------------------------------------------------------------------------
# Registry gc — nothing reconciled .sources.db against disk, so a registry
# could drift to double-digit percent junk unnoticed.
# ---------------------------------------------------------------------------


class TestRegistryGc:
    def _wiki_with_drift(self, tmp_path: Path) -> WikiStore:
        from outmem.sources import SourceRegistry

        store = WikiStore.init(tmp_path / "w")
        store.sources_path.mkdir(parents=True, exist_ok=True)
        real = store.sources_path / "kept.md"
        real.write_text("kept\n", encoding="utf-8")
        reg = SourceRegistry.load(store.sources_path)
        reg.register("kept.md", sha256="k" * 64, size_bytes=6)
        # a row whose directory was deleted out-of-band
        reg.register("gone/abcdef123456/x.md", sha256="g" * 64, size_bytes=9)
        reg.record_ingestion("gone/abcdef123456/x.md", prompt="p", pages_touched=["a"])
        # a file nobody registered
        (store.sources_path / "stray.md").write_text("stray\n", encoding="utf-8")
        return store

    def test_dry_run_reports_and_changes_nothing(self, tmp_path: Path) -> None:
        from outmem.sources import SourceRegistry

        store = self._wiki_with_drift(tmp_path)
        audit = store.sources_gc()  # dry_run default
        assert audit.missing_files == ["gone/abcdef123456/x.md"]
        assert audit.unregistered == ["stray.md"]
        assert not audit.is_clean
        # untouched
        reg = SourceRegistry.load(store.sources_path)
        assert "gone/abcdef123456/x.md" in reg.entries

    def test_apply_removes_rows_and_cascades_ingestions(self, tmp_path: Path) -> None:
        from outmem.sources import SourceRegistry, audit_registry

        store = self._wiki_with_drift(tmp_path)
        store.sources_gc(dry_run=False)
        reg = SourceRegistry.load(store.sources_path)
        assert "gone/abcdef123456/x.md" not in reg.entries
        assert "kept.md" in reg.entries  # a live row is never touched
        assert audit_registry(store.sources_path).orphan_ingestions == 0

    def test_apply_never_deletes_an_unregistered_file(self, tmp_path: Path) -> None:
        """Deleting a user's data to satisfy a registry is backwards."""
        store = self._wiki_with_drift(tmp_path)
        store.sources_gc(dry_run=False)
        assert (store.sources_path / "stray.md").is_file()

    def test_list_sources_hides_rows_whose_file_is_gone(self, tmp_path: Path) -> None:
        """These were handed to the agent as readable sources — it would
        burn a call on read_source and get 'no such source' back."""
        store = self._wiki_with_drift(tmp_path)
        listed = {e.rel_path for e in store.list_sources()}
        assert listed == {"kept.md"}
        raw = {e.rel_path for e in store.list_sources(include_missing=True)}
        assert "gone/abcdef123456/x.md" in raw

    def test_schema_version_is_stamped(self, tmp_path: Path) -> None:
        from outmem.sources import SCHEMA_VERSION, SourceRegistry

        sources = tmp_path / "s"
        sources.mkdir()
        reg = SourceRegistry.load(sources)
        assert reg._connection().execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION

    def test_clean_registry_is_clean(self, tmp_path: Path) -> None:
        from outmem.sources import audit_registry

        store = WikiStore.init(tmp_path / "w")
        store.sources_path.mkdir(parents=True, exist_ok=True)
        assert audit_registry(store.sources_path).is_clean


# ---------------------------------------------------------------------------
# The tracked / local split
# ---------------------------------------------------------------------------


class TestSourcesLocal:
    """``wiki/sources-local/`` — readable by the agent, never redistributed.

    The invariant under test throughout: source *bytes* stay on the
    machine while the pages compiled from them ship normally. Every
    check here is one of the ways that could silently stop being true.
    """

    def _doc(self, tmp_path: Path, name: str = "licensed.md") -> Path:
        src = tmp_path / name
        src.write_text("Chapter 1. All rights reserved.\n", encoding="utf-8")
        return src

    def test_local_ingest_lands_outside_the_tracked_tree(self, tmp_path: Path) -> None:
        store = WikiStore.init(tmp_path / "w")
        entry = store.add_source(self._doc(tmp_path), local=True)
        assert (store.sources_local_path / entry.rel_path).is_file()
        assert not (store.sources_path / entry.rel_path).exists()

    def test_local_tree_is_created_lazily(self, tmp_path: Path) -> None:
        store = WikiStore.init(tmp_path / "w")
        assert not store.sources_local_path.exists()
        store.add_source(self._doc(tmp_path), local=True)
        assert store.sources_local_path.is_dir()

    def test_local_tree_is_gitignored(self, tmp_path: Path) -> None:
        store = WikiStore.init(tmp_path / "w")
        store.add_source(self._doc(tmp_path), local=True)
        ignored = (store.root / ".gitignore").read_text(encoding="utf-8")
        assert "wiki/sources-local/" in ignored

    def test_git_never_sees_local_bytes(self, tmp_path: Path) -> None:
        """The whole point. If this fails the feature is worthless."""
        import subprocess

        store = WikiStore.init(tmp_path / "w")
        store.add_source(self._doc(tmp_path), local=True)
        # Simulate the careless-but-common `git add -A` before a commit.
        subprocess.run(["git", "add", "-A"], cwd=store.root, check=True, capture_output=True)
        staged = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            cwd=store.root, check=True, capture_output=True, text=True,
        ).stdout
        assert "sources-local" not in staged

    def test_local_ingest_produces_no_commit(self, tmp_path: Path) -> None:
        store = WikiStore.init(tmp_path / "w")
        store.write_page("seed", title="Seed", body="body")
        head_before = store.head()
        store.add_source(self._doc(tmp_path), local=True)
        assert store.head() == head_before

    def test_registries_are_separate(self, tmp_path: Path) -> None:
        """A shared registry would put the local file's name, hash, and
        the absolute path it was ingested from into a tracked DB —
        exactly what the split exists to withhold."""
        store = WikiStore.init(tmp_path / "w")
        entry = store.add_source(self._doc(tmp_path), local=True)
        assert (store.sources_local_path / ".sources.db").is_file()
        tracked_db = store.sources_path / ".sources.db"
        blob = tracked_db.read_bytes() if tracked_db.is_file() else b""
        assert entry.rel_path.encode() not in blob
        assert b"licensed.md" not in blob

    def test_local_source_is_readable(self, tmp_path: Path) -> None:
        store = WikiStore.init(tmp_path / "w")
        entry = store.add_source(self._doc(tmp_path), local=True)
        assert "All rights reserved" in store.read_source(entry.citation_path)
        # …and by the bare registry key too, since both spellings circulate.
        assert "All rights reserved" in store.read_source(entry.rel_path)

    def test_listing_marks_which_tree_a_row_came_from(self, tmp_path: Path) -> None:
        store = WikiStore.init(tmp_path / "w")
        store.add_source(self._doc(tmp_path, "open.md"))
        store.add_source(self._doc(tmp_path, "closed.md"), local=True)
        by_local = {e.local: e for e in store.list_sources()}
        assert set(by_local) == {True, False}
        assert by_local[True].citation_path.startswith("sources-local/")
        assert by_local[False].citation_path.startswith("sources/")

    def test_rel_path_stays_the_registry_key(self, tmp_path: Path) -> None:
        """Prefixing rel_path broke backfill once; pin the invariant so it
        cannot regress. rel_path is a key, citation_path is presentation."""
        store = WikiStore.init(tmp_path / "w")
        entry = store.add_source(self._doc(tmp_path), local=True)
        assert not entry.rel_path.startswith("sources")
        assert store.get_source(entry.citation_path) is not None

    def test_local_source_never_enters_the_semantic_index(
        self, tmp_path: Path
    ) -> None:
        """The subtle leak: the vector DB stores chunk text verbatim and
        is committed, so indexing local material ships it anyway."""
        from outmem._store.semantic import load_for_index

        store = WikiStore.init(tmp_path / "w")
        entry = store.add_source(self._doc(tmp_path), local=True)
        rel = f"wiki/sources-local/{entry.rel_path}"
        # Even under the most permissive index setting.
        store.config.outmem.semantic.index = "pages+sources"
        assert load_for_index(store, rel) is None
