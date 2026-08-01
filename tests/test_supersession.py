"""Source supersession + provenance-driven staleness (issue #7).

``rel_path`` embeds the content hash, so a revised document lands at a
*new row* and looks unrelated to the one it replaces. These tests pin the
identity that survives a revision (``document_key``), the edge it creates
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
        assert entry.document_key is None  # honestly unknown, not guessed
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
            document_key="abx/amikacin",
        )
        # Same sha => returns existing, which has no key yet; assign, then
        # a genuinely new version supersedes it.
        con = reg._connection()
        with con:
            con.execute(
                "UPDATE sources SET document_key = ? WHERE rel_path = ?",
                ("abx/amikacin", "abx/aaaaaaaaaaaa/document.md"),
            )
        reg = SourceRegistry.load(sources)
        reg.register(
            "abx/bbbbbbbbbbbb/document.md",
            sha256="b" * 64,
            size_bytes=14,
            document_key="abx/amikacin",
        )
        reloaded = SourceRegistry.load(sources)
        old = reloaded.entries["abx/aaaaaaaaaaaa/document.md"]
        assert old.superseded_by == "abx/bbbbbbbbbbbb/document.md"


class TestMigrationRepairsARenamedColumn:
    def test_registry_stamped_v2_without_the_column_repairs_itself(
        self, tmp_path: Path
    ) -> None:
        """An earlier build of this change called the identity column
        `logical_key` and stamped the same schema version. Gating the
        migration on the version alone would leave every read failing with
        "no such column: document_key"."""
        sources = tmp_path / "sources"
        sources.mkdir()
        con = sqlite3.connect(sources / REGISTRY_FILENAME)
        with con:
            con.execute(
                "CREATE TABLE sources ("
                " rel_path TEXT PRIMARY KEY, sha256 TEXT NOT NULL,"
                " size_bytes INTEGER NOT NULL, registered_at TEXT NOT NULL,"
                " logical_key TEXT, superseded_by TEXT, origin_path TEXT)"
            )
            con.execute(
                "CREATE TABLE ingestions ("
                " id INTEGER PRIMARY KEY AUTOINCREMENT, rel_path TEXT NOT NULL"
                " REFERENCES sources(rel_path) ON DELETE CASCADE,"
                " timestamp TEXT NOT NULL, prompt TEXT, pages_touched TEXT NOT NULL)"
            )
            con.execute(
                "INSERT INTO sources (rel_path, sha256, size_bytes, registered_at, "
                "logical_key) VALUES (?, ?, ?, ?, ?)",
                ("x/aaaaaaaaaaaa/d.md", "a" * 64, 1, "2026-01-01T00:00:00Z", "x/d"),
            )
            con.execute("PRAGMA user_version = 2")
        con.close()

        reg = SourceRegistry.load(sources)
        assert reg.entries["x/aaaaaaaaaaaa/d.md"].document_key is None


# ---------------------------------------------------------------------------
# Document keys name a document, not a file
# ---------------------------------------------------------------------------


class TestDocumentKeyShape:
    def test_derived_key_drops_the_source_extension(self, tmp_path: Path) -> None:
        """An extension is a property of the file, not of the document.
        Keeping it means a pipeline that switches from .md to .txt starts a
        new identity *silently* — supersession's whole job is removing the
        silent break."""
        store = WikiStore.init(tmp_path / "w")
        md = tmp_path / "guideline.md"
        md.write_text("2024 edition\n", encoding="utf-8")
        entry = store.add_source(md, into_subdir="rki")
        assert entry.document_key == "rki/guideline"

    def test_format_change_lands_on_the_same_identity(self, tmp_path: Path) -> None:
        """The payoff: the .txt re-export is refused as ambiguous rather
        than silently accepted as an unrelated document."""
        store = WikiStore.init(tmp_path / "w")
        md = tmp_path / "guideline.md"
        md.write_text("2024 edition\n", encoding="utf-8")
        store.add_source(md, into_subdir="rki")
        txt = tmp_path / "guideline.txt"
        txt.write_text("2026 edition\n", encoding="utf-8")
        with pytest.raises(OutmemError, match=r"--as rki/guideline\b"):
            store.add_source(txt, into_subdir="rki")

    def test_explicit_key_is_normalised_the_same_way(self, tmp_path: Path) -> None:
        """`--as` and the derived form must agree, or the flag the refusal
        message suggests would fail to link."""
        store = WikiStore.init(tmp_path / "w")
        for name, text, key in (
            ("v1.md", "one\n", None),
            ("v2.md", "two\n", "/RKI/Guideline.md"),
        ):
            src = tmp_path / name
            src.write_text(text, encoding="utf-8")
            store.add_source(src, into_subdir="rki", rename="guideline.md", as_key=key)
        reg = SourceRegistry.load(store.sources_path)
        assert {e.document_key for e in reg.entries.values()} == {"rki/guideline"}
        assert len([e for e in reg.entries.values() if e.superseded_by]) == 1

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("fachinfo/document.md", "fachinfo/document"),
            ("Fachinfo/Document.MD", "fachinfo/document"),
            ("/fachinfo//document.md/", "fachinfo/document"),
            # An external identifier is not a filename — its dot survives,
            # because `.1001-jama-2026` is not a source type.
            ("doi/10.1001-jama-2026", "doi/10.1001-jama-2026"),
            ("awmf/113-001", "awmf/113-001"),
        ],
    )
    def test_normalisation(self, raw: str, expected: str) -> None:
        from outmem.sources import normalize_document_key

        assert normalize_document_key(raw) == expected

    def test_empty_key_is_rejected(self) -> None:
        from outmem.sources import normalize_document_key

        with pytest.raises(OutmemError, match="not a usable document identity"):
            normalize_document_key("  //  ")


class TestRefusalQuotesTheOrigins:
    def test_proposes_the_segment_where_the_two_origins_diverge(
        self, tmp_path: Path
    ) -> None:
        """The wiki path discarded what told these apart; the ingest
        origins still have it. Read the answer instead of inventing it."""
        store = WikiStore.init(tmp_path / "w")
        first = tmp_path / "parsed" / "fachinfo" / "aztreonam" / "out" / "document.md"
        first.parent.mkdir(parents=True)
        first.write_text("aztreonam\n", encoding="utf-8")
        store.add_source(first, into_subdir="fachinfo")

        second = tmp_path / "parsed" / "fachinfo" / "amikacin" / "out" / "document.md"
        second.parent.mkdir(parents=True)
        second.write_text("amikacin\n", encoding="utf-8")
        with pytest.raises(OutmemError) as exc:
            store.add_source(second, into_subdir="fachinfo")

        msg = str(exc.value)
        assert "aztreonam" in msg and "amikacin" in msg  # both origins quoted
        assert "--as fachinfo/document" in msg  # the "new version" answer
        assert "--as fachinfo/amikacin" in msg  # the "different document" answer

    def test_no_proposal_when_the_divergence_is_a_hash(self, tmp_path: Path) -> None:
        """`--as fachinfo/9b3d0d4e1a35` would be worse than no suggestion."""
        from outmem.sources import distinguishing_segment

        assert (
            distinguishing_segment(
                "parsed/9b3d0d4e1a35/document.md", "parsed/a1b2c3d4e5f6/document.md"
            )
            is None
        )

    def test_no_proposal_when_the_older_origin_is_unknown(
        self, tmp_path: Path
    ) -> None:
        """Historical rows have no origin — refuse just as firmly, but
        don't invent evidence."""
        store = WikiStore.init(tmp_path / "w")
        store.sources_path.mkdir(parents=True, exist_ok=True)
        reg = SourceRegistry.load(store.sources_path)
        reg.register(
            "rki/aaaaaaaaaaaa/guideline.md",
            sha256="a" * 64,
            size_bytes=1,
            document_key="rki/guideline",
        )
        store._source_registry = None
        src = tmp_path / "guideline.md"
        src.write_text("new\n", encoding="utf-8")
        with pytest.raises(OutmemError) as exc:
            store.add_source(src, into_subdir="rki")
        assert "came from" not in str(exc.value)


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
        citations, _ = store.source_citations()
        assert citations["abx/aaaaaaaaaaaa/document.md"] == [
            "abx:amikacin",
            "abx:dosing",
        ]

    def test_page_without_provenance_contributes_nothing(self, tmp_path: Path) -> None:
        store = WikiStore.init(tmp_path / "w")
        store.write_page("solo", title="Solo", body="no source\n")
        assert store.source_citations() == ({}, [])


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
        stale, _ = store.stale_pages()
        assert len(stale) == 1
        assert stale[0].slug == "abx:amikacin"
        assert stale[0].cited == old
        assert stale[0].current == new
        assert stale[0].document_key == "abx/amikacin"

    def test_citing_the_current_version_is_not_stale(self, tmp_path: Path) -> None:
        store, _old, new = _wiki_with_a_superseded_source(tmp_path)
        store.write_page(
            "abx:amikacin-current",
            title="Current",
            body="up to date\n",
            provenance=[f"sources/{new}"],
        )
        stale = {c.slug for c in store.stale_pages()[0]}
        assert "abx:amikacin-current" not in stale

    def test_follows_the_chain_to_the_newest_version(self, tmp_path: Path) -> None:
        """A page citing v1 must be pointed at v3, not at v2 — otherwise
        the reader diffs against a version that is itself superseded."""
        store, old, mid = _wiki_with_a_superseded_source(tmp_path)
        v3 = tmp_path / "v3.md"
        v3.write_text("amikacin dosing, 2027 edition\n", encoding="utf-8")
        e3 = store.add_source(v3, into_subdir="abx", rename="document.md",
                              as_key="abx/amikacin")
        (stale,) = [c for c in store.stale_pages()[0] if c.cited == old]
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
        assert store.stale_pages() == ([], [])


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
        assert entry.document_key == "fachinfo/amikacin"

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
        (clear,) = [c for c in store.propose_document_keys()[0] if not c.is_ambiguous]
        assert clear.document_key == "rki/ratgeber"
        assert clear.rows == ["rki/aaaaaaaaaaaa/ratgeber.md"]

    def test_shared_filename_is_ambiguous_with_citing_pages_as_evidence(
        self, tmp_path: Path
    ) -> None:
        """Two drugs' ``document.md`` derive the same candidate. Merging
        them would declare one the successor of the other and later drive
        a recheck of amikacin's page against aztreonam's source."""
        store = _wiki_needing_backfill(tmp_path)
        (amb,) = [c for c in store.propose_document_keys()[0] if c.is_ambiguous]
        assert amb.document_key == "abx/document"
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
        (cand,), _ = store.propose_document_keys()
        assert cand.document_key == "deadbeefcafe/notes"

    def test_rows_that_already_have_an_identity_are_skipped(
        self, tmp_path: Path
    ) -> None:
        store = WikiStore.init(tmp_path / "w")
        src = tmp_path / "s.md"
        src.write_text("x\n", encoding="utf-8")
        store.add_source(src, as_key="explicit")
        assert store.propose_document_keys() == ([], [])


class TestAssignSourceKeys:
    def test_writes_the_key_and_persists_it(self, tmp_path: Path) -> None:
        store = _wiki_needing_backfill(tmp_path)
        written = store.assign_document_keys(
            [("rki/aaaaaaaaaaaa/ratgeber.md", "rki/ratgeber")]
        )
        assert written == 1
        reg = SourceRegistry.load(store.sources_path)
        assert reg.entries["rki/aaaaaaaaaaaa/ratgeber.md"].document_key == (
            "rki/ratgeber"
        )

    def test_never_overwrites_an_explicit_identity(self, tmp_path: Path) -> None:
        """`--as` always wins, so re-running backfill is safe."""
        store = WikiStore.init(tmp_path / "w")
        src = tmp_path / "s.md"
        src.write_text("x\n", encoding="utf-8")
        entry = store.add_source(src, as_key="explicit")
        assert store.assign_document_keys([(entry.rel_path, "guessed")]) == 0
        reg = SourceRegistry.load(store.sources_path)
        assert reg.entries[entry.rel_path].document_key == "explicit"


class TestBackfillCommand:
    def test_dry_run_reports_and_writes_nothing(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        store = _wiki_needing_backfill(tmp_path)
        rc = main(["sources", "backfill", "--root", str(store.root)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "can be assigned an identity" in out
        assert "rki/ratgeber" in out
        assert "dry run" in out
        reg = SourceRegistry.load(store.sources_path)
        assert reg.entries["rki/aaaaaaaaaaaa/ratgeber.md"].document_key is None

    def test_apply_writes_only_the_unambiguous_rows(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        store = _wiki_needing_backfill(tmp_path)
        assert main(["sources", "backfill", "--root", str(store.root), "--apply"]) == 0
        capsys.readouterr()
        reg = SourceRegistry.load(store.sources_path)
        assert reg.entries["rki/aaaaaaaaaaaa/ratgeber.md"].document_key == (
            "rki/ratgeber"
        )
        # The colliding pair is left alone — outmem cannot tell versions of
        # one document from two documents that share a filename.
        assert reg.entries["abx/bbbbbbbbbbbb/document.md"].document_key is None
        assert reg.entries["abx/cccccccccccc/document.md"].document_key is None

    def test_ambiguous_groups_print_their_citing_pages(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        store = _wiki_needing_backfill(tmp_path)
        main(["sources", "backfill", "--root", str(store.root)])
        out = capsys.readouterr().out
        assert "abx/document" in out
        assert "[[abx:amikacin]]" in out
        assert "[[abx:aztreonam]]" in out
        assert "--as" in out

    def test_ambiguous_groups_print_known_origins(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Origins are evidence even when nothing cites either row yet."""
        store = WikiStore.init(tmp_path / "w")
        for drug in ("amikacin", "aztreonam"):
            src = tmp_path / "parsed" / drug / "document.md"
            src.parent.mkdir(parents=True)
            src.write_text(f"{drug}\n", encoding="utf-8")
            store.add_source(src, into_subdir="abx", as_key=f"tmp/{drug}")
        # Strip the identities to recreate the pre-0.7 shape, keeping origins.
        con = SourceRegistry.load(store.sources_path)._connection()
        with con:
            con.execute("UPDATE sources SET document_key = NULL")
        store._source_registry = None

        main(["sources", "backfill", "--root", str(store.root)])
        out = capsys.readouterr().out
        assert "abx/document" in out
        assert "parsed/amikacin/document.md" in out
        assert "parsed/aztreonam/document.md" in out

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
        assert reg.entries["rki/aaaaaaaaaaaa/ratgeber.md"].document_key == (
            "rki/ratgeber"
        )


# ---------------------------------------------------------------------------
# One live row per identity — the invariant, from every angle that writes one
# ---------------------------------------------------------------------------


class TestOneLiveRowPerIdentity:
    def test_backfill_never_assigns_a_name_another_row_holds(
        self, tmp_path: Path
    ) -> None:
        """Two live rows on one identity is the merge add_source refuses to
        perform. Backfill must not do it silently instead: latest_for()
        would pick one head and the other would stay live forever, so a
        page citing it never appears in `outmem stale`."""
        store = WikiStore.init(tmp_path / "w")
        store.sources_path.mkdir(parents=True, exist_ok=True)
        reg = SourceRegistry.load(store.sources_path)
        reg.register("abx/aaaaaaaaaaaa/document.md", sha256="a" * 64, size_bytes=1)
        reg.register(
            "abx/bbbbbbbbbbbb/document.md",
            sha256="b" * 64,
            size_bytes=2,
            document_key="abx/document",  # the name the refusal itself suggests
        )
        store._source_registry = None

        (cand,), _ = store.propose_document_keys()
        assert cand.is_ambiguous
        assert cand.held_by == ["abx/bbbbbbbbbbbb/document.md"]
        assert main(["sources", "backfill", "--root", str(store.root), "--apply"]) == 0
        reloaded = SourceRegistry.load(store.sources_path)
        assert reloaded.entries["abx/aaaaaaaaaaaa/document.md"].document_key is None

    def test_assign_refuses_a_claimed_key_even_when_asked_directly(
        self, tmp_path: Path
    ) -> None:
        store = WikiStore.init(tmp_path / "w")
        store.sources_path.mkdir(parents=True, exist_ok=True)
        reg = SourceRegistry.load(store.sources_path)
        reg.register("a/aaaaaaaaaaaa/d.md", sha256="a" * 64, size_bytes=1)
        reg.register(
            "a/bbbbbbbbbbbb/d.md", sha256="b" * 64, size_bytes=2, document_key="a/d"
        )
        store._source_registry = None
        assert store.assign_document_keys([("a/aaaaaaaaaaaa/d.md", "a/d")]) == 0

    def test_identity_is_decided_against_the_db_not_a_stale_snapshot(
        self, tmp_path: Path
    ) -> None:
        """Two processes each hold a snapshot from before the other wrote.
        Deciding supersession from `self.entries` let both see v1 as the
        live head and both supersede it, leaving two live rows."""
        store = WikiStore.init(tmp_path / "w")
        store.sources_path.mkdir(parents=True, exist_ok=True)
        first = SourceRegistry.load(store.sources_path)
        first.register(
            "d/aaaaaaaaaaaa/x.md", sha256="a" * 64, size_bytes=1, document_key="d/x"
        )
        # Two independent handles, both snapshotted while v1 is the only row.
        proc_a = SourceRegistry.load(store.sources_path)
        proc_b = SourceRegistry.load(store.sources_path)
        proc_a.register(
            "d/bbbbbbbbbbbb/x.md", sha256="b" * 64, size_bytes=2, document_key="d/x"
        )
        proc_b.register(
            "d/cccccccccccc/x.md", sha256="c" * 64, size_bytes=3, document_key="d/x"
        )

        reloaded = SourceRegistry.load(store.sources_path)
        live = [
            e
            for e in reloaded.entries.values()
            if e.document_key == "d/x" and e.superseded_by is None
        ]
        assert len(live) == 1, [e.rel_path for e in live]

    def test_a_derived_key_still_refuses_across_snapshots(
        self, tmp_path: Path
    ) -> None:
        """The same staleness let two different documents both pass the
        clash check and land on one identity — the outcome the refusal
        exists to prevent."""
        store = WikiStore.init(tmp_path / "w")
        a = tmp_path / "a.md"
        a.write_text("amikacin\n", encoding="utf-8")
        store.add_source(a, into_subdir="abx", rename="document.md")
        stale_store = WikiStore.open(store.root)
        stale_store.list_sources()  # take a snapshot
        store.add_source(  # a third party claims nothing new, but re-reads
            a, into_subdir="abx", rename="document.md"
        )
        b = tmp_path / "b.md"
        b.write_text("aztreonam\n", encoding="utf-8")
        with pytest.raises(OutmemError, match="already the identity of"):
            stale_store.add_source(b, into_subdir="abx", rename="document.md")


class TestAsKeyOnAlreadyRegisteredContent:
    def test_as_key_lands_when_the_content_is_unchanged(self, tmp_path: Path) -> None:
        """`sources backfill` tells the operator to 're-ingest each with
        `--as <name>`'. The bytes are already registered, so returning
        early on the sha match made that instruction a no-op that
        reported success."""
        store = WikiStore.init(tmp_path / "w")
        src = tmp_path / "document.md"
        src.write_text("amikacin\n", encoding="utf-8")
        first = store.add_source(src, into_subdir="abx")
        con = SourceRegistry.load(store.sources_path)._connection()
        with con:  # simulate the pre-0.7 row backfill can't resolve
            con.execute("UPDATE sources SET document_key = NULL")
        store._source_registry = None

        again = store.add_source(src, into_subdir="abx", as_key="fachinfo/amikacin")
        assert again.rel_path == first.rel_path
        assert again.document_key == "fachinfo/amikacin"
        assert SourceRegistry.load(store.sources_path).entries[
            first.rel_path
        ].document_key == "fachinfo/amikacin"

    def test_repeating_the_same_key_is_a_no_op(self, tmp_path: Path) -> None:
        store = WikiStore.init(tmp_path / "w")
        src = tmp_path / "d.md"
        src.write_text("x\n", encoding="utf-8")
        store.add_source(src, as_key="a/b")
        again = store.add_source(src, as_key="a/b")
        assert again.document_key == "a/b"
        assert again.superseded_by is None

    def test_changing_an_established_identity_is_refused(self, tmp_path: Path) -> None:
        """Supersession edges point at the old identity; silently renaming
        it would strand them."""
        store = WikiStore.init(tmp_path / "w")
        src = tmp_path / "d.md"
        src.write_text("x\n", encoding="utf-8")
        store.add_source(src, as_key="a/one")
        with pytest.raises(OutmemError, match="already the identity"):
            store.add_source(src, as_key="a/two")

    def test_as_key_colliding_with_a_live_row_is_refused(self, tmp_path: Path) -> None:
        store = WikiStore.init(tmp_path / "w")
        other = tmp_path / "other.md"
        other.write_text("aztreonam\n", encoding="utf-8")
        store.add_source(other, as_key="abx/taken")
        src = tmp_path / "d.md"
        src.write_text("amikacin\n", encoding="utf-8")
        entry = store.add_source(src)
        con = SourceRegistry.load(store.sources_path)._connection()
        with con:  # an un-keyed row, as backfill would find it
            con.execute(
                "UPDATE sources SET document_key = NULL WHERE rel_path = ?",
                (entry.rel_path,),
            )
        store._source_registry = None
        with pytest.raises(OutmemError, match="already the identity of"):
            store.add_source(src, as_key="abx/taken")


class TestRefusalLeavesNoOrphan:
    def test_a_refused_ingest_does_not_copy_the_file(self, tmp_path: Path) -> None:
        """Copying first meant every refusal left an unregistered file that
        lint flags and `sources gc` refuses to delete — routine, since a
        <drug>/output/<hash>/document.md pipeline is refused by design on
        its second document."""
        store = WikiStore.init(tmp_path / "w")
        a = tmp_path / "a.md"
        a.write_text("amikacin\n", encoding="utf-8")
        store.add_source(a, into_subdir="abx", rename="document.md")
        before = sorted(p.name for p in store.sources_path.rglob("*") if p.is_file())

        b = tmp_path / "b.md"
        b.write_text("aztreonam\n", encoding="utf-8")
        with pytest.raises(OutmemError):
            store.add_source(b, into_subdir="abx", rename="document.md")

        after = sorted(p.name for p in store.sources_path.rglob("*") if p.is_file())
        assert after == before
        assert store.sources_gc().unregistered == []

    def test_an_unusable_as_key_leaves_nothing_behind(self, tmp_path: Path) -> None:
        store = WikiStore.init(tmp_path / "w")
        src = tmp_path / "d.md"
        src.write_text("x\n", encoding="utf-8")
        with pytest.raises(OutmemError, match="not a usable document identity"):
            store.add_source(src, as_key="  //  ")
        assert store.sources_gc().unregistered == []


class TestGcRepairsVersionChains:
    def test_deleting_a_row_splices_the_chain_rather_than_dangling(
        self, tmp_path: Path
    ) -> None:
        """`outmem stale` would otherwise point the reader at a path that is
        in neither the registry nor the filesystem."""
        store = WikiStore.init(tmp_path / "w")
        rels = []
        for i, text in enumerate(("v1\n", "v2\n", "v3\n")):
            src = tmp_path / f"v{i}.md"
            src.write_text(text, encoding="utf-8")
            rels.append(
                store.add_source(src, rename="d.md", as_key="a/d").rel_path
            )
        store.write_page(
            "p", title="P", body="b\n", provenance=[f"sources/{rels[0]}"]
        )
        # v2's file disappears out-of-band; gc drops the row.
        (store.sources_path / rels[1]).unlink()
        store.sources_gc(dry_run=False)

        (stale,), _ = store.stale_pages()
        assert stale.current == rels[2]
        assert stale.current_exists

    def test_a_dangling_pointer_is_reported_as_such(self, tmp_path: Path) -> None:
        store = WikiStore.init(tmp_path / "w")
        store.sources_path.mkdir(parents=True, exist_ok=True)
        reg = SourceRegistry.load(store.sources_path)
        reg.register("a/aaaaaaaaaaaa/d.md", sha256="a" * 64, size_bytes=1)
        con = reg._connection()
        with con:  # a registry edited out-of-band
            con.execute(
                "UPDATE sources SET superseded_by = 'a/gone/d.md' WHERE rel_path = ?",
                ("a/aaaaaaaaaaaa/d.md",),
            )
        store._source_registry = None
        store.write_page(
            "p", title="P", body="b\n", provenance=["sources/a/aaaaaaaaaaaa/d.md"]
        )
        (stale,), _ = store.stale_pages()
        assert not stale.current_exists
        main(["stale", "--root", str(store.root)])


class TestStaleSeesEveryProvenanceShape:
    def test_source_and_file_keys_are_read_too(self, tmp_path: Path) -> None:
        """lint resolves and sha-checks these shapes; a narrower extractor
        here meant a silent miss of the exact failure mode `outmem stale`
        exists to catch."""
        store, old, _new = _wiki_with_a_superseded_source(tmp_path)
        store.write_page(
            "via-source", title="A", body="a\n", provenance=[{"source": f"sources/{old}"}]
        )
        store.write_page(
            "via-file", title="B", body="b\n", provenance=[{"file": f"sources/{old}"}]
        )
        stale, _ = store.stale_pages()
        assert {"via-source", "via-file"} <= {c.slug for c in stale}

    def test_an_unparseable_page_is_reported_not_swallowed(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A page this check could not run on must not read as a clean
        wiki — that is the silence the shared loader contract exists to
        prevent."""
        store = WikiStore.init(tmp_path / "w")
        broken = store.pages_path / "broken.md"
        broken.parent.mkdir(parents=True, exist_ok=True)
        broken.write_text("---\ntitle: [unclosed\n---\n\nbody\n", encoding="utf-8")

        _stale, failures = store.stale_pages()
        assert [f.path.name for f in failures] == ["broken.md"]
        assert main(["stale", "--root", str(store.root)]) == 2
        assert "broken.md" in capsys.readouterr().err


class TestExtendCanUpdateProvenance:
    def test_recompacting_clears_the_stale_report(self, tmp_path: Path) -> None:
        """Without this there is no exit from `outmem stale`: extend_page
        preserved frontmatter verbatim and write_page refuses an existing
        page, so a reported page stayed reported forever."""
        store, _old, new = _wiki_with_a_superseded_source(tmp_path)
        assert store.stale_pages()[0]  # pre-condition

        store.extend_page(
            "abx:amikacin",
            body="15 mg/kg once daily, per the 2026 edition.\n",
            provenance=[f"sources/{new}"],
        )
        assert store.stale_pages() == ([], [])
        assert store.read("abx:amikacin").frontmatter.provenance == [f"sources/{new}"]

    def test_omitting_provenance_leaves_it_untouched(self, tmp_path: Path) -> None:
        store, old, _new = _wiki_with_a_superseded_source(tmp_path)
        store.extend_page("abx:amikacin", body="revised wording only\n")
        assert store.read("abx:amikacin").frontmatter.provenance == [f"sources/{old}"]

    def test_cli_extend_accepts_provenance(self, tmp_path: Path) -> None:
        import io

        store, _old, new = _wiki_with_a_superseded_source(tmp_path)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("sys.stdin", io.StringIO("re-compacted\n"))
            rc = main(
                [
                    "extend",
                    "--root",
                    str(store.root),
                    "abx:amikacin",
                    "--provenance",
                    f"sources/{new}",
                ]
            )
        assert rc == 0
        assert store.stale_pages() == ([], [])


# ---------------------------------------------------------------------------
# The whole point, end to end
# ---------------------------------------------------------------------------


def test_v06_wiki_backfills_then_supersedes_then_surfaces_the_page(
    tmp_path: Path,
) -> None:
    """The whole journey for a wiki that predates identity.

    v0.6 rows have no ``document_key``, so a re-ingest lands as an
    unrelated row and nothing notices. Backfill gives the old row an
    identity; the re-ingest is then *refused* rather than guessed at, and
    once resolved it supersedes — surfacing the page compacted from the
    version that is no longer current.
    """
    store = WikiStore.init(tmp_path / "w")
    v1 = tmp_path / "v1.md"
    v1.write_text("guideline, 2024\n", encoding="utf-8")
    e1 = store.add_source(v1, into_subdir="rki", rename="ratgeber.md")
    # Simulate a v0.6 row: registered before document_key existed.
    con = SourceRegistry.load(store.sources_path)._connection()
    with con:
        con.execute("UPDATE sources SET document_key = NULL")
    store._source_registry = None
    store.write_page(
        "rki:ratgeber",
        title="Ratgeber",
        body="summary\n",
        provenance=[f"sources/{e1.rel_path}"],
    )
    assert store.stale_pages() == ([], [])  # no identity, so no supersession

    (cand,), _ = store.propose_document_keys()
    assert not cand.is_ambiguous
    store.assign_document_keys([(cand.rows[0], cand.document_key)])

    v2 = tmp_path / "v2.md"
    v2.write_text("guideline, 2026 revision\n", encoding="utf-8")
    # The key is now claimed, and a bare path cannot say whether this is the
    # next version or a different document — so outmem asks instead of
    # guessing, and names the exact flag that answers it.
    with pytest.raises(OutmemError, match=r"--as rki/ratgeber\b"):
        store.add_source(v2, into_subdir="rki", rename="ratgeber.md")

    e2 = store.add_source(
        v2, into_subdir="rki", rename="ratgeber.md", as_key="rki/ratgeber"
    )
    (stale,), _ = store.stale_pages()
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


# ---------------------------------------------------------------------------
# Uncoupling frozen sources from mutable page slugs
# ---------------------------------------------------------------------------


class TestSourceRefsSurviveRenames:
    def test_a_rename_repoints_the_recorded_reference(self, tmp_path: Path) -> None:
        """The uncoupling. `rename_page` cannot rewrite a frozen source —
        that is what content addressing means — but it can rewrite the
        mapping recorded at ingest, which is why the mapping exists."""
        store = WikiStore.init(tmp_path / "w")
        store.write_page("clinical:sepsis", title="Sepsis", body="s")
        src = tmp_path / "sop.md"
        src.write_text("Vgl. clinical:sepsis fuer das Protokoll.\n", encoding="utf-8")
        entry = store.add_source(src, into_subdir="sop")

        (ref,) = store.source_refs(entry.rel_path)
        assert ref.token == "clinical:sepsis"
        assert ref.page_slug == "clinical:sepsis"
        assert not ref.exact

        store.rename_page("clinical:sepsis", "clinical:infektion:sepsis")

        (ref,) = store.source_refs(entry.rel_path)
        assert ref.token == "clinical:sepsis"  # the frozen bytes never change
        assert ref.page_slug == "clinical:infektion:sepsis"  # the mapping follows

    def test_lint_is_quiet_once_the_mapping_holds_the_reference(
        self, tmp_path: Path
    ) -> None:
        """Held by identity, not by a string — so it survives even after
        the alias is gone, which the alias-only fix could not."""
        from outmem.lint import lint_wiki

        store = WikiStore.init(tmp_path / "w")
        store.write_page("clinical:sepsis", title="Sepsis", body="s")
        src = tmp_path / "sop.md"
        src.write_text("Vgl. clinical:sepsis\n", encoding="utf-8")
        store.add_source(src, into_subdir="sop")
        store.rename_page("clinical:sepsis", "clinical:infektion:sepsis", alias=False)

        report = lint_wiki(
            store.wiki_path,
            log_dir=store.log_path,
            raw_dir=store.raw_path,
            sources_dir=store.sources_path,
        )
        assert [
            f for f in report.findings if f.kind == "source-references-dead-slug"
        ] == []

    def test_an_exact_wikilink_needs_no_heuristics(self, tmp_path: Path) -> None:
        """A single-segment slug is just a word — `_SLUG_MENTION_RE`
        requires a colon, so prose can never detect `glossary`. Written as
        a link it is unambiguous, which is why markup is the better input."""
        store = WikiStore.init(tmp_path / "w")
        store.write_page("glossary", title="Glossary", body="g")
        src = tmp_path / "sop.md"
        src.write_text("See [[glossary]] and also glossary.\n", encoding="utf-8")
        entry = store.add_source(src, into_subdir="sop")

        (ref,) = store.source_refs(entry.rel_path)
        assert ref.token == "glossary"
        assert ref.exact

    def test_unresolvable_tokens_are_not_recorded(self, tmp_path: Path) -> None:
        """A slug already dead on arrival stays dead — recording a guess
        would be worse than recording nothing."""
        store = WikiStore.init(tmp_path / "w")
        store.write_page("clinical:sepsis", title="Sepsis", body="s")
        src = tmp_path / "sop.md"
        src.write_text("Vgl. clinical:ghost, Abnahme 12:30, ratio 3:1\n", encoding="utf-8")
        entry = store.add_source(src, into_subdir="sop")
        assert store.source_refs(entry.rel_path) == []

    def test_a_deleted_page_still_reports(self, tmp_path: Path) -> None:
        """Uncoupling protects against rename, not deletion. The mapping
        makes the break precise; it cannot repair it."""
        from outmem.lint import lint_wiki

        store = WikiStore.init(tmp_path / "w")
        store.write_page("clinical:sepsis", title="Sepsis", body="s")
        src = tmp_path / "sop.md"
        src.write_text("Vgl. clinical:sepsis\n", encoding="utf-8")
        store.add_source(src, into_subdir="sop")
        store._page_path("clinical:sepsis").unlink()

        report = lint_wiki(
            store.wiki_path,
            log_dir=store.log_path,
            raw_dir=store.raw_path,
            sources_dir=store.sources_path,
        )
        assert [f for f in report.findings if f.kind == "source-references-dead-slug"]


class TestLoadBearingAliases:
    def test_an_alias_a_source_needs_is_not_nudged_for_retirement(
        self, tmp_path: Path
    ) -> None:
        """Both alias nudges said "so the alias can eventually go". For an
        alias a content-addressed file depends on, that advice is wrong:
        the source cannot be edited to stop needing it."""
        from outmem.lint import lint_wiki

        store = WikiStore.init(tmp_path / "w")
        store.write_page("clinical:sepsis", title="Sepsis", body="s")
        store.write_page("clinical:other", title="Other", body="Vgl. clinical:sepsis\n")
        src = tmp_path / "sop.md"
        src.write_text("Vgl. clinical:sepsis\n", encoding="utf-8")
        store.add_source(src, into_subdir="sop")
        store.rename_page("clinical:sepsis", "clinical:infektion:sepsis")

        report = lint_wiki(
            store.wiki_path,
            log_dir=store.log_path,
            raw_dir=store.raw_path,
            sources_dir=store.sources_path,
        )
        (via,) = [f for f in report.findings if f.kind == "slug-mention-via-alias"]
        assert "keep the alias" in via.message
        assert "can eventually go" not in via.message

    def test_an_unpinned_alias_still_gets_the_retirement_nudge(
        self, tmp_path: Path
    ) -> None:
        """The guarantee this must not weaken: aliases with no frozen
        dependant are still debt with cleanup pressure."""
        from outmem.lint import lint_wiki

        store = WikiStore.init(tmp_path / "w")
        store.write_page("clinical:sepsis", title="Sepsis", body="s")
        store.write_page("clinical:other", title="Other", body="Vgl. clinical:sepsis\n")
        store.rename_page("clinical:sepsis", "clinical:infektion:sepsis")

        report = lint_wiki(store.wiki_path, log_dir=store.log_path, raw_dir=store.raw_path)
        (via,) = [f for f in report.findings if f.kind == "slug-mention-via-alias"]
        assert "can eventually go" in via.message


class TestSourceRefsBackfill:
    def test_scanned_and_empty_is_not_reported_again(self, tmp_path: Path) -> None:
        """"No references" and "never scanned" are different states. Without
        `refs_scanned_at` backfill re-reports every source that names no
        pages, forever."""
        store = WikiStore.init(tmp_path / "w")
        src = tmp_path / "plain.md"
        src.write_text("No slugs here at all.\n", encoding="utf-8")
        store.add_source(src)
        assert main(["sources", "backfill", "--root", str(store.root)]) == 0

    def test_apply_records_refs_for_a_pre_existing_source(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        store = WikiStore.init(tmp_path / "w")
        store.write_page("clinical:sepsis", title="Sepsis", body="s")
        src = tmp_path / "sop.md"
        src.write_text("Vgl. clinical:sepsis\n", encoding="utf-8")
        entry = store.add_source(src, into_subdir="sop")
        # Simulate a source registered before the mapping existed.
        con = SourceRegistry.load(store.sources_path)._connection()
        with con:
            con.execute("DELETE FROM source_refs")
            con.execute("UPDATE sources SET refs_scanned_at = NULL")
        store._source_registry = None

        assert main(["sources", "backfill", "--root", str(store.root)]) == 0
        assert "no recorded page references" in capsys.readouterr().out
        assert store.source_refs(entry.rel_path) == []  # dry run wrote nothing

        assert main(["sources", "backfill", "--root", str(store.root), "--apply"]) == 0
        store._source_registry = None
        (ref,) = store.source_refs(entry.rel_path)
        assert ref.page_slug == "clinical:sepsis"
