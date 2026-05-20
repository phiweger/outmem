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
