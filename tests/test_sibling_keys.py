"""Finding two source rows that are really one document.

``document_key`` links a revision to what it replaces, but a source
ingested without ``--as`` has its identity *derived* from its filename —
and the edition marker is usually in the filename. So the 2024 and 2026
editions of one guideline become two documents, no supersession edge is
written, and ``outmem stale`` never reports the page compacted from the
older one. Nothing in the registry is wrong; the failure is an absence,
which is why it takes a check of its own to see.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest

from outmem.lint import Severity, lint_wiki
from outmem.sources import (
    SourceRegistry,
    find_unchained_versions,
    sibling_form,
    version_order,
)
from outmem.store import WikiStore


class TestSiblingForm:
    @pytest.mark.parametrize(
        ("left", "right"),
        [
            # The reported case.
            ("guidelines/eucast-2024", "guidelines/eucast-2026"),
            # An AWMF register number is itself digits, so the year is not
            # the only run that collapses — the rest of the name still
            # has to match for the two to group.
            (
                "guidelines/awmf-043-044-s3-hwi-erwachsene-2024",
                "guidelines/awmf-043-044-s3-hwi-erwachsene-2026",
            ),
            # Year *and* month: a bare-year rule misses this one.
            ("epi/stiko-2024-01", "epi/stiko-2025-01"),
            # A format change, which is what the wider extension set is
            # for — neither .pdf nor .docx is stripped from a real key.
            ("ref/handbook.pdf", "ref/handbook.docx"),
        ],
    )
    def test_editions_of_one_document_share_a_form(
        self, left: str, right: str
    ) -> None:
        assert sibling_form(left) == sibling_form(right)

    @pytest.mark.parametrize(
        ("left", "right"),
        [
            ("fachinfo/amikacin", "fachinfo/aztreonam"),
            ("guidelines/eucast-2024", "guidelines/eucast-tables-2024"),
            ("a/report", "b/report"),
        ],
    )
    def test_different_documents_do_not(self, left: str, right: str) -> None:
        assert sibling_form(left) != sibling_form(right)

    def test_a_form_is_never_an_identity(self) -> None:
        """Lossy on purpose. Storing one would merge documents rather
        than report them, which is the opposite of the intent."""
        assert sibling_form("guidelines/eucast-2024") == "guidelines/eucast-#"

    def test_it_is_idempotent(self) -> None:
        """Forms are compared for equality, so a form that re-forms into
        something else would group two keys on one pass and not the next."""
        for key in ("x/report.csv.pdf", "x/a.pdf", "x/a", "doi/10.1001-x", ""):
            assert sibling_form(sibling_form(key)) == sibling_form(key), key


class TestVersionOrder:
    """The tie-break when `registered_at` cannot order two versions —
    which a bulk ingest guarantees, since the column stores seconds."""

    def test_numbers_compare_as_numbers(self) -> None:
        keys = ["g/eucast-v9", "g/eucast-v10", "g/eucast-v2"]
        assert sorted(keys, key=version_order) == [
            "g/eucast-v2",
            "g/eucast-v9",
            "g/eucast-v10",
        ]
        # Lexicographic order gets this wrong, which is why the tie-break
        # is not just `sorted(keys)`.
        assert sorted(keys) != sorted(keys, key=version_order)

    def test_no_shape_pair_mixes_int_with_str(self) -> None:
        """A key alternates literal/number, so the same tuple position
        always holds the same type. If that ever stopped being true the
        comparison would raise TypeError mid-sort, on someone's corpus."""
        shapes = ["a", "a1", "ab", "1", "", "a-1-b", "a-1b", "doi/10.1001-x", "z/9"]
        for left, right in itertools.combinations(shapes, 2):
            # Comparing at all is the assertion — a mixed pair raises.
            assert isinstance(version_order(left) < version_order(right), bool)


def _wiki(tmp_path: Path) -> WikiStore:
    return WikiStore.init(tmp_path / "w")


def _ingest(
    store: WikiStore, tmp_path: Path, name: str, *, into: str = "guidelines", **kw: object
) -> str:
    src = tmp_path / name
    src.write_text(f"contents of {name}\n", encoding="utf-8")
    return store.add_source(src, into_subdir=into, **kw).rel_path  # type: ignore[arg-type]


class TestFindUnchainedVersions:
    def test_two_derived_editions_are_reported(self, tmp_path: Path) -> None:
        store = _wiki(tmp_path)
        _ingest(store, tmp_path, "eucast-2024.md")
        _ingest(store, tmp_path, "eucast-2026.md")
        (group,) = find_unchained_versions(SourceRegistry.load(store.sources_path))
        assert group.keys == ["guidelines/eucast-2024", "guidelines/eucast-2026"]
        assert not group.shares_one_key

    def test_a_chained_document_is_not_reported(self, tmp_path: Path) -> None:
        """The point of the live-rows-only filter: once `rekey` has
        chained them there is one current row and nothing to say."""
        store = _wiki(tmp_path)
        _ingest(store, tmp_path, "eucast-2024.md")
        _ingest(store, tmp_path, "eucast-2026.md")
        store.rekey_document(
            "guidelines/eucast-2024", "guidelines/eucast-2026", dry_run=False
        )
        assert find_unchained_versions(SourceRegistry.load(store.sources_path)) == []

    def test_properly_superseded_versions_are_not_reported(
        self, tmp_path: Path
    ) -> None:
        """Ingested with `--as`, which is the whole point of `--as`."""
        store = _wiki(tmp_path)
        _ingest(store, tmp_path, "eucast-2024.md", as_key="guidelines/eucast")
        _ingest(store, tmp_path, "eucast-2026.md", as_key="guidelines/eucast")
        assert find_unchained_versions(SourceRegistry.load(store.sources_path)) == []

    def test_declared_identities_are_never_second_guessed(
        self, tmp_path: Path
    ) -> None:
        """Two DOIs differ only in digits and share a form. They are
        different articles, and saying so is what `--as` is for — so the
        check ignores any key it did not derive itself."""
        store = _wiki(tmp_path)
        _ingest(store, tmp_path, "a.md", as_key="doi/10.1001-jama-2026")
        _ingest(store, tmp_path, "b.md", as_key="doi/10.1001-jama-2027")
        assert sibling_form("doi/10.1001-jama-2026") == sibling_form(
            "doi/10.1001-jama-2027"
        )
        assert find_unchained_versions(SourceRegistry.load(store.sources_path)) == []

    def test_unrelated_sources_are_quiet(self, tmp_path: Path) -> None:
        store = _wiki(tmp_path)
        _ingest(store, tmp_path, "amikacin.md", into="fachinfo")
        _ingest(store, tmp_path, "aztreonam.md", into="fachinfo")
        assert find_unchained_versions(SourceRegistry.load(store.sources_path)) == []

    def test_one_key_with_several_live_rows_is_its_own_finding(
        self, tmp_path: Path
    ) -> None:
        """The state a hand-written `UPDATE sources SET document_key`
        leaves: the registry asserts one document while leaving both rows
        current, which its own write paths refuse to do."""
        import sqlite3

        from outmem.sources import REGISTRY_FILENAME

        store = _wiki(tmp_path)
        _ingest(store, tmp_path, "eucast-2024.md")
        _ingest(store, tmp_path, "eucast-2026.md")
        con = sqlite3.connect(store.sources_path / REGISTRY_FILENAME)
        with con:
            con.execute("UPDATE sources SET document_key = 'guidelines/eucast'")
        con.close()
        (group,) = find_unchained_versions(SourceRegistry.load(store.sources_path))
        assert group.shares_one_key
        assert group.keys == ["guidelines/eucast"]


class TestLint:
    def _report(self, store: WikiStore):
        return lint_wiki(
            store.wiki_path,
            sources_dir=store.sources_path,
            sources_local_dir=store.sources_local_path,
        )

    def test_warns_and_names_both_remedies(self, tmp_path: Path) -> None:
        store = _wiki(tmp_path)
        _ingest(store, tmp_path, "eucast-2024.md")
        _ingest(store, tmp_path, "eucast-2026.md")
        (finding,) = [
            f
            for f in self._report(store).findings
            if f.kind == "unlinked-source-versions"
        ]
        assert finding.severity == Severity.WARNING
        # One document -> merge them.
        assert (
            "outmem sources rekey guidelines/eucast-2024 --to "
            "guidelines/eucast-2026" in finding.message
        )
        # Different documents -> rename one, which also silences this.
        assert "distinguishes it" in finding.message

    def test_the_message_carries_the_ingest_origins(self, tmp_path: Path) -> None:
        """The evidence that settles it: two rows from
        `.../amikacin/...` and `.../aztreonam/...` are visibly different
        documents even when nothing cites either yet."""
        store = _wiki(tmp_path)
        _ingest(store, tmp_path, "eucast-2024.md")
        _ingest(store, tmp_path, "eucast-2026.md")
        (finding,) = [
            f
            for f in self._report(store).findings
            if f.kind == "unlinked-source-versions"
        ]
        assert str(tmp_path / "eucast-2024.md") in finding.message

    def test_shared_key_reports_the_chain_repair(self, tmp_path: Path) -> None:
        import sqlite3

        from outmem.sources import REGISTRY_FILENAME

        store = _wiki(tmp_path)
        _ingest(store, tmp_path, "eucast-2024.md")
        _ingest(store, tmp_path, "eucast-2026.md")
        con = sqlite3.connect(store.sources_path / REGISTRY_FILENAME)
        with con:
            con.execute("UPDATE sources SET document_key = 'guidelines/eucast'")
        con.close()
        (finding,) = [
            f for f in self._report(store).findings if f.kind == "multiple-live-versions"
        ]
        assert "outmem sources rekey guidelines/eucast" in finding.message
        assert "--to" not in finding.message

    def test_clean_wiki_is_silent(self, tmp_path: Path) -> None:
        store = _wiki(tmp_path)
        _ingest(store, tmp_path, "eucast-2024.md")
        kinds = {f.kind for f in self._report(store).findings}
        assert "unlinked-source-versions" not in kinds
        assert "multiple-live-versions" not in kinds

    def test_rekey_takes_the_warning_to_zero(self, tmp_path: Path) -> None:
        """A warning that cannot reach zero gets suppressed wholesale, so
        the remedy has to actually clear it."""
        store = _wiki(tmp_path)
        _ingest(store, tmp_path, "eucast-2024.md")
        _ingest(store, tmp_path, "eucast-2026.md")
        assert any(
            f.kind == "unlinked-source-versions" for f in self._report(store).findings
        )
        store.rekey_document(
            "guidelines/eucast-2024", "guidelines/eucast-2026", dry_run=False
        )
        assert not any(
            f.kind == "unlinked-source-versions" for f in self._report(store).findings
        )

    def test_the_local_tree_is_checked_too(self, tmp_path: Path) -> None:
        """The recurring bug class in this codebase: a check that opens
        'the' registry when there are two."""
        store = _wiki(tmp_path)
        for name in ("eucast-2024.md", "eucast-2026.md"):
            src = tmp_path / name
            src.write_text(f"local {name}\n", encoding="utf-8")
            store.add_source(src, into_subdir="guidelines", local=True)
        (finding,) = [
            f
            for f in self._report(store).findings
            if f.kind == "unlinked-source-versions"
        ]
        assert finding.path.startswith("sources-local/")
