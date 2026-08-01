"""Source supersession + provenance-driven staleness (issue #7).

``rel_path`` embeds the content hash, so a revised document lands at a
*new row* and looks unrelated to the one it replaces. These tests pin the
identity that survives a revision (``logical_key``), the edge it creates
(``superseded_by``), and the thing that edge is for: finding the pages
that were compacted from a version no longer current.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from outmem.cli.__main__ import main
from outmem.exceptions import OutmemError
from outmem.sources import REGISTRY_FILENAME, SourceRegistry
from outmem.store import WikiStore

# ---------------------------------------------------------------------------
# Schema migration — an existing wiki must upgrade in place, not be rebuilt.
# ---------------------------------------------------------------------------


class TestMigration:
    def _v1_registry(self, sources: Path) -> None:
        """A ``.sources.db`` in exactly the shape v0.6 shipped."""
        sources.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(sources / REGISTRY_FILENAME)
        with con:
            con.execute(
                "CREATE TABLE sources ("
                " rel_path TEXT PRIMARY KEY,"
                " sha256 TEXT NOT NULL,"
                " size_bytes INTEGER NOT NULL,"
                " registered_at TEXT NOT NULL)"
            )
            con.execute(
                "CREATE TABLE ingestions ("
                " id INTEGER PRIMARY KEY AUTOINCREMENT,"
                " rel_path TEXT NOT NULL"
                " REFERENCES sources(rel_path) ON DELETE CASCADE,"
                " timestamp TEXT NOT NULL,"
                " prompt TEXT,"
                " pages_touched TEXT NOT NULL)"
            )
            con.execute(
                "INSERT INTO sources VALUES (?, ?, ?, ?)",
                ("abx/aaaaaaaaaaaa/document.md", "a" * 64, 12, "2026-01-01T00:00:00Z"),
            )
            con.execute(
                "INSERT INTO ingestions (rel_path, timestamp, prompt, pages_touched) "
                "VALUES (?, ?, ?, ?)",
                (
                    "abx/aaaaaaaaaaaa/document.md",
                    "2026-01-01T00:00:00Z",
                    "p",
                    '["abx:x"]',
                ),
            )
            con.execute("PRAGMA user_version = 1")
        con.close()

    def test_v1_registry_upgrades_in_place(self, tmp_path: Path) -> None:
        from outmem.sources import SCHEMA_VERSION

        sources = tmp_path / "sources"
        self._v1_registry(sources)
        reg = SourceRegistry.load(sources)

        entry = reg.entries["abx/aaaaaaaaaaaa/document.md"]
        assert entry.sha256 == "a" * 64
        assert entry.logical_key is None  # honestly unknown, not guessed
        assert entry.superseded_by is None
        assert entry.origin_path is None
        # Ingestion history survives the migration.
        assert [i.prompt for i in entry.ingestions] == ["p"]
        con = reg._connection()
        assert con.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION

    def test_migration_is_idempotent(self, tmp_path: Path) -> None:
        """Re-opening must not re-run ALTER TABLE — that would raise."""
        sources = tmp_path / "sources"
        self._v1_registry(sources)
        SourceRegistry.load(sources)
        reg = SourceRegistry.load(sources)
        assert "abx/aaaaaaaaaaaa/document.md" in reg.entries

    def test_migrated_registry_accepts_supersession(self, tmp_path: Path) -> None:
        """The point of migrating rather than rebuilding: a wiki built on
        v0.6 can start superseding without re-ingesting everything."""
        sources = tmp_path / "sources"
        self._v1_registry(sources)
        reg = SourceRegistry.load(sources)
        reg.register(
            "abx/aaaaaaaaaaaa/document.md",
            sha256="a" * 64,
            size_bytes=12,
            logical_key="abx/amikacin",
        )
        # Same sha => returns existing, which has no key yet; assign, then
        # a genuinely new version supersedes it.
        con = reg._connection()
        with con:
            con.execute(
                "UPDATE sources SET logical_key = ? WHERE rel_path = ?",
                ("abx/amikacin", "abx/aaaaaaaaaaaa/document.md"),
            )
        reg = SourceRegistry.load(sources)
        reg.register(
            "abx/bbbbbbbbbbbb/document.md",
            sha256="b" * 64,
            size_bytes=14,
            logical_key="abx/amikacin",
        )
        reloaded = SourceRegistry.load(sources)
        old = reloaded.entries["abx/aaaaaaaaaaaa/document.md"]
        assert old.superseded_by == "abx/bbbbbbbbbbbb/document.md"


# ---------------------------------------------------------------------------
# source_citations — the reverse provenance edge
# ---------------------------------------------------------------------------


class TestSourceCitations:
    def test_collects_string_and_mapping_provenance(self, tmp_path: Path) -> None:
        """Both documented provenance shapes point at the same source."""
        store = WikiStore.init(tmp_path / "w")
        store.write_page(
            "abx:amikacin",
            title="Amikacin",
            body="dosing\n",
            provenance=["sources/abx/aaaaaaaaaaaa/document.md"],
        )
        store.write_page(
            "abx:dosing",
            title="Dosing",
            body="overview\n",
            provenance=[
                {"path": "sources/abx/aaaaaaaaaaaa/document.md", "source": "ingest"}
            ],
        )
        citations = store.source_citations()
        assert citations["abx/aaaaaaaaaaaa/document.md"] == [
            "abx:amikacin",
            "abx:dosing",
        ]

    def test_page_without_provenance_contributes_nothing(self, tmp_path: Path) -> None:
        store = WikiStore.init(tmp_path / "w")
        store.write_page("solo", title="Solo", body="no source\n")
        assert store.source_citations() == {}


# ---------------------------------------------------------------------------
# stale_pages / `outmem stale`
# ---------------------------------------------------------------------------


def _wiki_with_a_superseded_source(tmp_path: Path) -> tuple[WikiStore, str, str]:
    """A page compacted from v1 of a document that has since moved to v2."""
    store = WikiStore.init(tmp_path / "w")
    v1 = tmp_path / "v1.md"
    v1.write_text("amikacin dosing, 2024 edition\n", encoding="utf-8")
    e1 = store.add_source(v1, into_subdir="abx", rename="document.md",
                          as_key="abx/amikacin")
    store.write_page(
        "abx:amikacin",
        title="Amikacin",
        body="15 mg/kg once daily.\n",
        provenance=[f"sources/{e1.rel_path}"],
    )
    v2 = tmp_path / "v2.md"
    v2.write_text("amikacin dosing, 2026 edition\n", encoding="utf-8")
    e2 = store.add_source(v2, into_subdir="abx", rename="document.md",
                          as_key="abx/amikacin")
    return store, e1.rel_path, e2.rel_path


class TestStalePages:
    def test_reports_the_page_that_cites_the_old_version(self, tmp_path: Path) -> None:
        store, old, new = _wiki_with_a_superseded_source(tmp_path)
        stale = store.stale_pages()
        assert len(stale) == 1
        assert stale[0].slug == "abx:amikacin"
        assert stale[0].cited == old
        assert stale[0].current == new
        assert stale[0].logical_key == "abx/amikacin"

    def test_citing_the_current_version_is_not_stale(self, tmp_path: Path) -> None:
        store, _old, new = _wiki_with_a_superseded_source(tmp_path)
        store.write_page(
            "abx:amikacin-current",
            title="Current",
            body="up to date\n",
            provenance=[f"sources/{new}"],
        )
        stale = {c.slug for c in store.stale_pages()}
        assert "abx:amikacin-current" not in stale

    def test_follows_the_chain_to_the_newest_version(self, tmp_path: Path) -> None:
        """A page citing v1 must be pointed at v3, not at v2 — otherwise
        the reader diffs against a version that is itself superseded."""
        store, old, mid = _wiki_with_a_superseded_source(tmp_path)
        v3 = tmp_path / "v3.md"
        v3.write_text("amikacin dosing, 2027 edition\n", encoding="utf-8")
        e3 = store.add_source(v3, into_subdir="abx", rename="document.md",
                              as_key="abx/amikacin")
        (stale,) = [c for c in store.stale_pages() if c.cited == old]
        assert stale.current == e3.rel_path
        assert stale.current != mid

    def test_clean_wiki_reports_nothing(self, tmp_path: Path) -> None:
        store = WikiStore.init(tmp_path / "w")
        src = tmp_path / "s.md"
        src.write_text("only version\n", encoding="utf-8")
        entry = store.add_source(src, as_key="s")
        store.write_page(
            "p", title="P", body="b\n", provenance=[f"sources/{entry.rel_path}"]
        )
        assert store.stale_pages() == []


class TestStaleCommand:
    def test_exits_nonzero_and_names_both_versions(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        store, old, new = _wiki_with_a_superseded_source(tmp_path)
        rc = main(["stale", "--root", str(store.root)])
        assert rc == 1
        out = capsys.readouterr().out
        assert "[[abx:amikacin]]" in out
        assert old in out
        assert new in out

    def test_clean_wiki_exits_zero(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        store = WikiStore.init(tmp_path / "w")
        store.write_page("p", title="P", body="b\n")
        assert main(["stale", "--root", str(store.root)]) == 0
        assert "no page cites a superseded source" in capsys.readouterr().out


class TestIngestAsFlag:
    """The refusal message names `--as`, so `--as` has to exist."""

    def test_as_flag_sets_the_identity(self, tmp_path: Path) -> None:
        store = WikiStore.init(tmp_path / "w")
        src = tmp_path / "v1.md"
        src.write_text("v1\n", encoding="utf-8")
        rc = main(
            [
                "ingest",
                "--root",
                str(store.root),
                str(src),
                "--into",
                "abx",
                "--rename",
                "document.md",
                "--as",
                "fachinfo/amikacin",
                "--register-only",
            ]
        )
        assert rc == 0
        reg = SourceRegistry.load(store.sources_path)
        (entry,) = reg.entries.values()
        assert entry.logical_key == "fachinfo/amikacin"

    def test_as_flag_supersedes_across_two_cli_runs(self, tmp_path: Path) -> None:
        store = WikiStore.init(tmp_path / "w")
        argv = ["ingest", "--root", str(store.root)]
        for name, text in (("v1.md", "v1\n"), ("v2.md", "v2 revised\n")):
            src = tmp_path / name
            src.write_text(text, encoding="utf-8")
            rc = main(
                [*argv, str(src), "--rename", "ratgeber.md",
                 "--as", "rki/ratgeber", "--register-only"]
            )
            assert rc == 0
        reg = SourceRegistry.load(store.sources_path)
        superseded = [e for e in reg.entries.values() if e.superseded_by]
        assert len(superseded) == 1

    def test_ambiguous_ingest_exits_one_with_actionable_guidance(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        store = WikiStore.init(tmp_path / "w")
        for name, text in (("a.md", "amikacin\n"), ("b.md", "aztreonam\n")):
            (tmp_path / name).write_text(text, encoding="utf-8")
        base = ["ingest", "--root", str(store.root)]
        opts = ["--into", "abx", "--rename", "document.md", "--register-only"]
        assert main([*base, str(tmp_path / "a.md"), *opts]) == 0
        assert main([*base, str(tmp_path / "b.md"), *opts]) == 1
        err = capsys.readouterr().err
        assert "already the identity of" in err
        assert "--as" in err


# ---------------------------------------------------------------------------
# sources backfill — propose an identity for rows that predate one
# ---------------------------------------------------------------------------


def _wiki_needing_backfill(tmp_path: Path) -> WikiStore:
    """One unambiguous row, plus two rows sharing a filename cited by
    different pages — the collision the hash directory exists to survive.
    """
    store = WikiStore.init(tmp_path / "w")
    store.sources_path.mkdir(parents=True, exist_ok=True)
    reg = SourceRegistry.load(store.sources_path)
    reg.register("rki/aaaaaaaaaaaa/ratgeber.md", sha256="a" * 64, size_bytes=1)
    reg.register("abx/bbbbbbbbbbbb/document.md", sha256="b" * 64, size_bytes=2)
    reg.register("abx/cccccccccccc/document.md", sha256="c" * 64, size_bytes=3)
    store._source_registry = None  # drop the cached pre-registration view
    store.write_page(
        "abx:amikacin",
        title="Amikacin",
        body="a\n",
        provenance=["sources/abx/bbbbbbbbbbbb/document.md"],
    )
    store.write_page(
        "abx:aztreonam",
        title="Aztreonam",
        body="b\n",
        provenance=["sources/abx/cccccccccccc/document.md"],
    )
    return store


class TestProposeSourceKeys:
    def test_hash_dir_row_proposes_the_path_without_the_hash(
        self, tmp_path: Path
    ) -> None:
        store = _wiki_needing_backfill(tmp_path)
        (clear,) = [c for c in store.propose_source_keys() if not c.is_ambiguous]
        assert clear.logical_key == "rki/ratgeber.md"
        assert clear.rows == ["rki/aaaaaaaaaaaa/ratgeber.md"]

    def test_shared_filename_is_ambiguous_with_citing_pages_as_evidence(
        self, tmp_path: Path
    ) -> None:
        """Two drugs' ``document.md`` derive the same candidate. Merging
        them would declare one the successor of the other and later drive
        a recheck of amikacin's page against aztreonam's source."""
        store = _wiki_needing_backfill(tmp_path)
        (amb,) = [c for c in store.propose_source_keys() if c.is_ambiguous]
        assert amb.logical_key == "abx/document.md"
        assert set(amb.rows) == {
            "abx/bbbbbbbbbbbb/document.md",
            "abx/cccccccccccc/document.md",
        }
        assert amb.citing_pages["abx/bbbbbbbbbbbb/document.md"] == ["abx:amikacin"]
        assert amb.citing_pages["abx/cccccccccccc/document.md"] == ["abx:aztreonam"]

    def test_a_directory_named_like_a_hash_is_not_mistaken_for_one(
        self, tmp_path: Path
    ) -> None:
        """The sha segment is verified against the row's own hash, so a
        legitimately hex-named directory keeps its full path as the key."""
        store = WikiStore.init(tmp_path / "w")
        store.sources_path.mkdir(parents=True, exist_ok=True)
        reg = SourceRegistry.load(store.sources_path)
        reg.register("deadbeefcafe/notes.md", sha256="f" * 64, size_bytes=1)
        store._source_registry = None
        (cand,) = store.propose_source_keys()
        assert cand.logical_key == "deadbeefcafe/notes.md"

    def test_rows_that_already_have_an_identity_are_skipped(
        self, tmp_path: Path
    ) -> None:
        store = WikiStore.init(tmp_path / "w")
        src = tmp_path / "s.md"
        src.write_text("x\n", encoding="utf-8")
        store.add_source(src, as_key="explicit")
        assert store.propose_source_keys() == []


class TestAssignSourceKeys:
    def test_writes_the_key_and_persists_it(self, tmp_path: Path) -> None:
        store = _wiki_needing_backfill(tmp_path)
        written = store.assign_source_keys(
            [("rki/aaaaaaaaaaaa/ratgeber.md", "rki/ratgeber.md")]
        )
        assert written == 1
        reg = SourceRegistry.load(store.sources_path)
        assert reg.entries["rki/aaaaaaaaaaaa/ratgeber.md"].logical_key == (
            "rki/ratgeber.md"
        )

    def test_never_overwrites_an_explicit_identity(self, tmp_path: Path) -> None:
        """`--as` always wins, so re-running backfill is safe."""
        store = WikiStore.init(tmp_path / "w")
        src = tmp_path / "s.md"
        src.write_text("x\n", encoding="utf-8")
        entry = store.add_source(src, as_key="explicit")
        assert store.assign_source_keys([(entry.rel_path, "guessed")]) == 0
        reg = SourceRegistry.load(store.sources_path)
        assert reg.entries[entry.rel_path].logical_key == "explicit"


class TestBackfillCommand:
    def test_dry_run_reports_and_writes_nothing(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        store = _wiki_needing_backfill(tmp_path)
        rc = main(["sources", "backfill", "--root", str(store.root)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "would assign" in out
        assert "rki/ratgeber.md" in out
        assert "dry run" in out
        reg = SourceRegistry.load(store.sources_path)
        assert reg.entries["rki/aaaaaaaaaaaa/ratgeber.md"].logical_key is None

    def test_apply_writes_only_the_unambiguous_rows(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        store = _wiki_needing_backfill(tmp_path)
        assert main(["sources", "backfill", "--root", str(store.root), "--apply"]) == 0
        capsys.readouterr()
        reg = SourceRegistry.load(store.sources_path)
        assert reg.entries["rki/aaaaaaaaaaaa/ratgeber.md"].logical_key == (
            "rki/ratgeber.md"
        )
        # The colliding pair is left alone — outmem cannot tell versions of
        # one document from two documents that share a filename.
        assert reg.entries["abx/bbbbbbbbbbbb/document.md"].logical_key is None
        assert reg.entries["abx/cccccccccccc/document.md"].logical_key is None

    def test_ambiguous_groups_print_their_citing_pages(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        store = _wiki_needing_backfill(tmp_path)
        main(["sources", "backfill", "--root", str(store.root)])
        out = capsys.readouterr().out
        assert "abx/document.md" in out
        assert "[[abx:amikacin]]" in out
        assert "[[abx:aztreonam]]" in out
        assert "--as" in out

    def test_nothing_to_do_exits_zero(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        store = WikiStore.init(tmp_path / "w")
        src = tmp_path / "s.md"
        src.write_text("x\n", encoding="utf-8")
        store.add_source(src, as_key="explicit")
        assert main(["sources", "backfill", "--root", str(store.root)]) == 0
        assert "already has an identity" in capsys.readouterr().out

    def test_apply_is_idempotent(self, tmp_path: Path) -> None:
        store = _wiki_needing_backfill(tmp_path)
        main(["sources", "backfill", "--root", str(store.root), "--apply"])
        # Second run finds only the ambiguous pair, which it never resolves.
        assert main(["sources", "backfill", "--root", str(store.root), "--apply"]) == 0
        reg = SourceRegistry.load(store.sources_path)
        assert reg.entries["rki/aaaaaaaaaaaa/ratgeber.md"].logical_key == (
            "rki/ratgeber.md"
        )


# ---------------------------------------------------------------------------
# The whole point, end to end
# ---------------------------------------------------------------------------


def test_v06_wiki_backfills_then_supersedes_then_surfaces_the_page(
    tmp_path: Path,
) -> None:
    """The whole journey for a wiki that predates identity.

    v0.6 rows have no ``logical_key``, so a re-ingest lands as an
    unrelated row and nothing notices. Backfill gives the old row an
    identity; the re-ingest is then *refused* rather than guessed at, and
    once resolved it supersedes — surfacing the page compacted from the
    version that is no longer current.
    """
    store = WikiStore.init(tmp_path / "w")
    v1 = tmp_path / "v1.md"
    v1.write_text("guideline, 2024\n", encoding="utf-8")
    e1 = store.add_source(v1, into_subdir="rki", rename="ratgeber.md")
    # Simulate a v0.6 row: registered before logical_key existed.
    con = SourceRegistry.load(store.sources_path)._connection()
    with con:
        con.execute("UPDATE sources SET logical_key = NULL")
    store._source_registry = None
    store.write_page(
        "rki:ratgeber",
        title="Ratgeber",
        body="summary\n",
        provenance=[f"sources/{e1.rel_path}"],
    )
    assert store.stale_pages() == []  # no identity, so no supersession

    (cand,) = store.propose_source_keys()
    assert not cand.is_ambiguous
    store.assign_source_keys([(cand.rows[0], cand.logical_key)])

    v2 = tmp_path / "v2.md"
    v2.write_text("guideline, 2026 revision\n", encoding="utf-8")
    # The key is now claimed, and a bare path cannot say whether this is the
    # next version or a different document — so outmem asks instead of
    # guessing, and names the exact flag that answers it.
    with pytest.raises(OutmemError, match=r"rki/ratgeber\.md"):
        store.add_source(v2, into_subdir="rki", rename="ratgeber.md")

    e2 = store.add_source(
        v2, into_subdir="rki", rename="ratgeber.md", as_key="rki/ratgeber.md"
    )
    (stale,) = store.stale_pages()
    assert stale.slug == "rki:ratgeber"
    assert stale.cited == e1.rel_path
    assert stale.current == e2.rel_path


def test_reingesting_identical_content_is_still_a_no_op(tmp_path: Path) -> None:
    """Identity checks must not turn the idempotent case into a refusal —
    same content is the same row, not a competing claim on the key."""
    store = WikiStore.init(tmp_path / "w")
    src = tmp_path / "s.md"
    src.write_text("unchanged\n", encoding="utf-8")
    first = store.add_source(src, into_subdir="rki", rename="ratgeber.md")
    again = store.add_source(src, into_subdir="rki", rename="ratgeber.md")
    assert again.rel_path == first.rel_path
    assert again.superseded_by is None
