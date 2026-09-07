"""Restriction labels on ingested sources — the registry half.

Labelling happens at ingest because that is the one moment a person is
holding the document and knows what it is. Everything downstream —
which pages inherit the label, what retrieval hides — follows from the
value written here, so this is where the fail-safe directions are
pinned: omitting the flag never declassifies, a new version never
drops its predecessor's labels, and a value the column cannot parse
hides the source rather than publishing it.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from outmem._store.sources import get_registry
from outmem.exceptions import OutmemError
from outmem.restricted import DENY_SET, LabelError
from outmem.sources import REGISTRY_FILENAME, SCHEMA_VERSION, SourceRegistry
from outmem.store import WikiStore


def _wiki(tmp_path: Path, labels: list[str] | None = None, **blocks: object) -> WikiStore:
    root = tmp_path / "w"
    store = WikiStore.init(root)
    if labels is not None:
        import yaml

        raw = yaml.safe_load((root / "config.yaml").read_text()) or {}
        raw["restricted"] = {"labels": labels, **blocks}
        (root / "config.yaml").write_text(yaml.safe_dump(raw))
        store.close()
        store = WikiStore.open(root)
    return store


def _registry(store: WikiStore) -> SourceRegistry:
    return get_registry(store, None)


def _doc(tmp_path: Path, name: str, text: str = "Body.\n") -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


class TestIngestLabelling:
    def test_a_source_ingests_open_by_default(self, tmp_path: Path) -> None:
        store = _wiki(tmp_path, ["hr"])
        entry = store.add_source(_doc(tmp_path, "a.md"))
        assert entry.restricted == frozenset()

    def test_the_flag_labels_it(self, tmp_path: Path) -> None:
        store = _wiki(tmp_path, ["hr"])
        entry = store.add_source(_doc(tmp_path, "a.md"), restricted=["hr"])
        assert entry.restricted == frozenset({"hr"})

    def test_the_label_survives_a_reopen(self, tmp_path: Path) -> None:
        store = _wiki(tmp_path, ["hr"])
        entry = store.add_source(_doc(tmp_path, "a.md"), restricted=["hr"])
        store.close()
        reopened = WikiStore.open(tmp_path / "w")
        assert reopened.get_source(entry.rel_path).restricted == frozenset({"hr"})

    def test_an_undeclared_label_is_refused_at_ingest(self, tmp_path: Path) -> None:
        """The operator is holding the document and can fix a typo now.
        Reads fail closed and silent; this entry point does not."""
        store = _wiki(tmp_path, ["hr"])
        with pytest.raises(LabelError, match="unknown restriction"):
            store.add_source(_doc(tmp_path, "a.md"), restricted=["legel"])

    def test_a_path_rule_labels_it_without_the_flag(self, tmp_path: Path) -> None:
        """The safety net: someone drops a file into hr/ and forgets."""
        store = _wiki(tmp_path, ["hr"], sources={"hr/*": ["hr"]})
        entry = store.add_source(_doc(tmp_path, "a.md"), into_subdir="hr")
        assert entry.restricted == frozenset({"hr"})

    def test_flag_and_path_rule_union(self, tmp_path: Path) -> None:
        store = _wiki(tmp_path, ["hr", "legal"], sources={"hr/*": ["hr"]})
        entry = store.add_source(
            _doc(tmp_path, "a.md"), into_subdir="hr", restricted=["legal"]
        )
        assert entry.restricted == frozenset({"hr", "legal"})

    def test_an_unmatched_path_stays_open(self, tmp_path: Path) -> None:
        store = _wiki(tmp_path, ["hr"], sources={"hr/*": ["hr"]})
        entry = store.add_source(_doc(tmp_path, "a.md"), into_subdir="policy")
        assert entry.restricted == frozenset()

    def test_local_and_restricted_are_orthogonal(self, tmp_path: Path) -> None:
        """The local tree is about redistribution rights; restriction is
        about secrecy. A source can be either, both, or neither."""
        store = _wiki(tmp_path, ["hr"])
        entry = store.add_source(
            _doc(tmp_path, "a.md"), local=True, restricted=["hr"]
        )
        assert entry.local and entry.restricted == frozenset({"hr"})


class TestOmittingTheFlagNeverDeclassifies:
    def test_re_ingesting_the_same_bytes_keeps_the_label(self, tmp_path: Path) -> None:
        store = _wiki(tmp_path, ["hr"])
        doc = _doc(tmp_path, "a.md")
        first = store.add_source(doc, restricted=["hr"])
        again = store.add_source(doc)  # no flag
        assert again.rel_path == first.rel_path
        assert store.get_source(first.rel_path).restricted == frozenset({"hr"})

    def test_re_ingesting_with_the_flag_widens(self, tmp_path: Path) -> None:
        """How an operator corrects a source they should have labelled."""
        store = _wiki(tmp_path, ["hr"])
        doc = _doc(tmp_path, "a.md")
        entry = store.add_source(doc)
        store.add_source(doc, restricted=["hr"])
        assert store.get_source(entry.rel_path).restricted == frozenset({"hr"})

    def test_a_new_version_inherits_its_predecessors_labels(
        self, tmp_path: Path
    ) -> None:
        """Silent declassification, closed at the cause rather than only
        reported by lint: re-ingesting v2 without the flag would
        otherwise leave the document key with an OPEN current version,
        which `outmem stale` then quietly points pages at."""
        store = _wiki(tmp_path, ["hr"])
        v1 = store.add_source(
            _doc(tmp_path, "v1.md", "One.\n"), as_key="policy/sev", restricted=["hr"]
        )
        v2 = store.add_source(
            _doc(tmp_path, "v2.md", "Two.\n"), as_key="policy/sev"
        )
        assert store.get_source(v1.rel_path).superseded_by == v2.rel_path
        assert v2.restricted == frozenset({"hr"})

    def test_a_new_version_may_still_add_labels(self, tmp_path: Path) -> None:
        store = _wiki(tmp_path, ["hr", "legal"])
        store.add_source(
            _doc(tmp_path, "v1.md", "One.\n"), as_key="policy/sev", restricted=["hr"]
        )
        v2 = store.add_source(
            _doc(tmp_path, "v2.md", "Two.\n"),
            as_key="policy/sev",
            restricted=["legal"],
        )
        assert v2.restricted == frozenset({"hr", "legal"})


class TestSetRestricted:
    def test_widening_is_allowed(self, tmp_path: Path) -> None:
        store = _wiki(tmp_path, ["hr", "legal"])
        entry = store.add_source(_doc(tmp_path, "a.md"), restricted=["hr"])
        registry = _registry(store)
        registry.set_restricted(entry.rel_path, {"hr", "legal"})
        assert registry.entries[entry.rel_path].restricted == {"hr", "legal"}

    def test_narrowing_is_refused_by_default(self, tmp_path: Path) -> None:
        store = _wiki(tmp_path, ["hr", "legal"])
        entry = store.add_source(
            _doc(tmp_path, "a.md"), restricted=["hr", "legal"]
        )
        with pytest.raises(LabelError, match="privileged"):
            _registry(store).set_restricted(entry.rel_path, {"hr"})

    def test_narrowing_goes_through_when_declared(self, tmp_path: Path) -> None:
        store = _wiki(tmp_path, ["hr", "legal"])
        entry = store.add_source(
            _doc(tmp_path, "a.md"), restricted=["hr", "legal"]
        )
        registry = _registry(store)
        registry.set_restricted(entry.rel_path, {"hr"}, allow_narrowing=True)
        assert registry.entries[entry.rel_path].restricted == {"hr"}

    def test_an_unregistered_path_is_refused(self, tmp_path: Path) -> None:
        store = _wiki(tmp_path, ["hr"])
        with pytest.raises(OutmemError, match="not registered"):
            _registry(store).set_restricted("nope/x.md", {"hr"})


class TestColumnEncoding:
    def _column(self, store: WikiStore, rel_path: str) -> object:
        con = sqlite3.connect(store.sources_path / REGISTRY_FILENAME)
        try:
            row = con.execute(
                "SELECT restricted FROM sources WHERE rel_path = ?", (rel_path,)
            ).fetchone()
        finally:
            con.close()
        return row[0]

    def test_open_sources_store_null_not_an_empty_array(
        self, tmp_path: Path
    ) -> None:
        """One representation of "open", so a query can rely on it."""
        store = _wiki(tmp_path, ["hr"])
        entry = store.add_source(_doc(tmp_path, "a.md"))
        assert self._column(store, entry.rel_path) is None

    def test_labels_are_stored_as_a_sorted_json_array(self, tmp_path: Path) -> None:
        store = _wiki(tmp_path, ["hr", "legal"])
        entry = store.add_source(
            _doc(tmp_path, "a.md"), restricted=["legal", "hr"]
        )
        assert json.loads(self._column(store, entry.rel_path)) == ["hr", "legal"]

    @pytest.mark.parametrize("junk", ["{", "null", '"hr"', "[1, 2]", '["HR"]', "[]x"])
    def test_an_unreadable_value_hides_the_source(
        self, tmp_path: Path, junk: str
    ) -> None:
        """The same choice frontmatter makes for an unparseable page: if
        the labels cannot be read, the item is denied to everyone. The
        other reading — "no labels, so open" — is fail-open."""
        store = _wiki(tmp_path, ["hr"])
        entry = store.add_source(_doc(tmp_path, "a.md"), restricted=["hr"])
        db = store.sources_path / REGISTRY_FILENAME
        store.close()
        con = sqlite3.connect(db)
        with con:
            con.execute(
                "UPDATE sources SET restricted = ? WHERE rel_path = ?",
                (junk, entry.rel_path),
            )
        con.close()
        registry = SourceRegistry.load(store.sources_path)
        assert registry.entries[entry.rel_path].restricted == DENY_SET


class TestMigration:
    def test_the_schema_version_moved(self) -> None:
        assert SCHEMA_VERSION == 4

    def test_a_v3_registry_gains_the_column_on_open(self, tmp_path: Path) -> None:
        """An existing registry must never need rebuilding."""
        store = _wiki(tmp_path)
        entry = store.add_source(_doc(tmp_path, "a.md"))
        db = store.sources_path / REGISTRY_FILENAME
        store.close()
        con = sqlite3.connect(db)
        with con:
            # Reproduce a pre-v4 file: no column, stamped v3.
            con.execute("ALTER TABLE sources DROP COLUMN restricted")
            con.execute("PRAGMA user_version = 3")
        con.close()

        registry = SourceRegistry.load(store.sources_path)
        assert registry.entries[entry.rel_path].restricted == frozenset()
        con = sqlite3.connect(db)
        try:
            assert con.execute("PRAGMA user_version").fetchone()[0] == 4
        finally:
            con.close()

    def test_pre_existing_rows_read_as_open(self, tmp_path: Path) -> None:
        """NULL means open — the honest answer for a registry filled
        before restrictions existed, and the same answer the wiki gave
        before the column did. Restricting an already-ingested source is
        a deliberate act, not a migration default."""
        store = _wiki(tmp_path)
        entry = store.add_source(_doc(tmp_path, "a.md"))
        assert store.get_source(entry.rel_path).restricted == frozenset()


class TestCli:
    def test_the_flag_labels_and_reports(self, tmp_path: Path, capsys) -> None:
        from outmem.cli.__main__ import main

        store = _wiki(tmp_path, ["hr"])
        root = store.root
        store.close()
        doc = _doc(tmp_path, "a.md")
        rc = main(
            ["ingest", str(doc), "--restricted", "hr", "--register-only",
             "--root", str(root)]
        )
        assert rc == 0
        assert "restricted to: hr" in capsys.readouterr().out
        assert WikiStore.open(root).list_sources()[0].restricted == frozenset({"hr"})

    def test_an_undeclared_label_exits_nonzero(
        self, tmp_path: Path, capsys
    ) -> None:
        from outmem.cli.__main__ import main

        store = _wiki(tmp_path, ["hr"])
        root = store.root
        store.close()
        rc = main(
            ["ingest", str(_doc(tmp_path, "a.md")), "--restricted", "nope",
             "--register-only", "--root", str(root)]
        )
        assert rc == 1
        assert "unknown restriction" in capsys.readouterr().err

    def test_repeating_the_flag_gives_several_labels(
        self, tmp_path: Path
    ) -> None:
        from outmem.cli.__main__ import main

        store = _wiki(tmp_path, ["hr", "legal"])
        root = store.root
        store.close()
        main(
            ["ingest", str(_doc(tmp_path, "a.md")), "--restricted", "hr",
             "--restricted", "legal", "--register-only", "--root", str(root)]
        )
        assert WikiStore.open(root).list_sources()[0].restricted == {"hr", "legal"}
