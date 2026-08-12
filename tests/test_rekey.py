"""``outmem sources rekey`` — moving a document, and re-chaining it.

The failure this exists for: two editions of one document ingested
*without* ``--as`` derive their identities from their filenames, so
``eucast-2024.md`` and ``eucast-2026.md`` become two unrelated documents.
No supersession edge is written, ``outmem stale`` reports nothing, and a
page compacted from the 2024 edition looks current forever.

Relabelling alone does not fix that. Two rows under one identity with no
edge between them leave :meth:`SourceRegistry.latest_for` silently
returning the newer — the same silence, one step further in. So every
test here checks the *edges*, not just the key.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from outmem.cli.__main__ import main
from outmem.exceptions import OutmemError
from outmem.sources import REGISTRY_FILENAME, SourceRegistry
from outmem.store import WikiStore


def _wiki_with_sibling_keys(tmp_path: Path) -> tuple[WikiStore, str, str]:
    """Two editions of one guideline, ingested without ``--as``.

    Exactly the shape a bulk ingest produces: the year is in the
    filename, so the derived keys differ and nothing links them.

    The two are stamped a year apart. ``registered_at`` is stored to the
    second, so ingesting both inside one test would tie — and a tie is a
    genuinely different case (see :class:`TestTiedIngestTimes`) that
    would otherwise contaminate every assertion about chain order here.
    """
    store = WikiStore.init(tmp_path / "w")
    old_file = tmp_path / "eucast-2024.md"
    old_file.write_text("breakpoints, 2024 edition\n", encoding="utf-8")
    old = store.add_source(old_file, into_subdir="guidelines")
    store.write_page(
        "clinical:breakpoints",
        title="Breakpoints",
        body="S <= 2 mg/L.\n",
        provenance=[f"sources/{old.rel_path}"],
    )
    new_file = tmp_path / "eucast-2026.md"
    new_file.write_text("breakpoints, 2026 edition\n", encoding="utf-8")
    new = store.add_source(new_file, into_subdir="guidelines")
    _stamp(store, {old.rel_path: "2024-01-01", new.rel_path: "2026-01-01"})
    return store, old.rel_path, new.rel_path


def _stamp(store: WikiStore, when: dict[str, str]) -> None:
    """Set ``registered_at`` per row, as ``YYYY-MM-DD``."""
    for rel_path, day in when.items():
        _out_of_band(
            store,
            "UPDATE sources SET registered_at = ? WHERE rel_path = ?",
            (f"{day}T00:00:00Z", rel_path),
        )


def _out_of_band(store: WikiStore, sql: str, params: tuple[object, ...]) -> None:
    """Edit ``.sources.db`` behind the store's back, then drop its snapshot.

    A separate connection on purpose: this reproduces the hand-repair
    that motivated the feature (``UPDATE sources SET document_key = …``),
    which is exactly the write path outmem's own API refuses to expose.
    """
    con = sqlite3.connect(store.sources_path / REGISTRY_FILENAME)
    with con:
        con.execute(sql, params)
    con.close()
    store._source_registry = None


def _registry(store: WikiStore) -> SourceRegistry:
    return SourceRegistry.load(store.sources_path)


class TestTheReportedFailure:
    def test_sibling_keys_are_two_documents_and_stale_is_silent(
        self, tmp_path: Path
    ) -> None:
        """The premise. Without this the rest of the file proves nothing."""
        store, old, new = _wiki_with_sibling_keys(tmp_path)
        registry = _registry(store)
        assert registry.entries[old].document_key == "guidelines/eucast-2024"
        assert registry.entries[new].document_key == "guidelines/eucast-2026"
        assert registry.entries[old].superseded_by is None
        assert store.stale_pages() == ([], [])

    def test_rekey_merges_them_and_stale_fires(self, tmp_path: Path) -> None:
        store, old, new = _wiki_with_sibling_keys(tmp_path)
        result = store.rekey_document(
            "guidelines/eucast-2024", "guidelines/eucast-2026", dry_run=False
        )
        assert result.applied
        assert result.chain == (old, new)
        assert result.merged_with == (new,)

        registry = _registry(store)
        assert registry.entries[old].document_key == "guidelines/eucast-2026"
        assert registry.entries[old].superseded_by == new
        assert registry.entries[new].superseded_by is None

        (stale,) = store.stale_pages()[0]
        assert stale.slug == "clinical:breakpoints"
        assert stale.cited == old
        assert stale.current == new


class TestRelabellingAloneIsNotEnough:
    """The bug in the obvious repair — a bare ``UPDATE document_key``.

    Two rows land on one identity with no edge between them.
    ``latest_for`` resolves that by ``max(registered_at)`` rather than
    complaining, so the registry looks fine and ``stale`` stays silent.
    ``rekey`` with no target is the repair.
    """

    def _hand_relabelled(self, tmp_path: Path) -> tuple[WikiStore, str, str]:
        store, old, new = _wiki_with_sibling_keys(tmp_path)
        _out_of_band(
            store,
            "UPDATE sources SET document_key = 'guidelines/eucast' "
            "WHERE rel_path IN (?, ?)",
            (old, new),
        )
        return store, old, new

    def test_two_live_heads_are_silent(self, tmp_path: Path) -> None:
        store, old, new = self._hand_relabelled(tmp_path)
        registry = _registry(store)
        live = [
            e
            for e in registry.entries.values()
            if e.document_key == "guidelines/eucast" and e.superseded_by is None
        ]
        assert len(live) == 2
        head = registry.latest_for("guidelines/eucast")
        assert head is not None and head.rel_path == new
        assert store.stale_pages() == ([], [])
        assert old  # both rows are live; neither is reported

    def test_rekey_without_a_target_chains_them(self, tmp_path: Path) -> None:
        store, old, new = self._hand_relabelled(tmp_path)
        result = store.rekey_document("guidelines/eucast", dry_run=False)
        assert result.applied
        assert result.moved == (old, new)
        assert result.merged_with == ()
        assert result.chain == (old, new)

        assert _registry(store).entries[old].superseded_by == new
        (stale,) = store.stale_pages()[0]
        assert stale.cited == old
        assert stale.current == new


class TestChainOrdering:
    def test_a_merge_interleaves_by_registered_at(self, tmp_path: Path) -> None:
        """Two chains merging is not a concatenation.

        A 2025 edition registered under its own key belongs *between* the
        2024 and 2026 editions, not after them — the chain follows
        ``registered_at``, which is what ``latest_for`` and ``stale``
        both read.
        """
        store, old, new = _wiki_with_sibling_keys(tmp_path)
        mid_file = tmp_path / "eucast-2025.md"
        mid_file.write_text("breakpoints, 2025 edition\n", encoding="utf-8")
        mid = store.add_source(mid_file, into_subdir="guidelines").rel_path
        # Ingested last, but belongs in the middle — the normal case when
        # an older edition is backfilled after a newer one.
        _stamp(store, {mid: "2025-01-01"})

        store.rekey_document(
            "guidelines/eucast-2024", "guidelines/eucast-2026", dry_run=False
        )
        result = store.rekey_document(
            "guidelines/eucast-2025", "guidelines/eucast-2026", dry_run=False
        )
        assert result.chain == (old, mid, new)
        registry = _registry(store)
        assert registry.entries[old].superseded_by == mid
        assert registry.entries[mid].superseded_by == new
        assert registry.entries[new].superseded_by is None

    def test_stale_points_at_the_head_not_the_next_link(self, tmp_path: Path) -> None:
        store, old, _new = _wiki_with_sibling_keys(tmp_path)
        store.rekey_document(
            "guidelines/eucast-2024", "guidelines/eucast-2026", dry_run=False
        )
        third = tmp_path / "eucast-2027.md"
        third.write_text("breakpoints, 2027 edition\n", encoding="utf-8")
        e3 = store.add_source(third, into_subdir="guidelines")
        _stamp(store, {e3.rel_path: "2027-01-01"})
        store.rekey_document(
            "guidelines/eucast-2027", "guidelines/eucast-2026", dry_run=False
        )
        (stale,) = [c for c in store.stale_pages()[0] if c.cited == old]
        assert stale.current == e3.rel_path


class TestTiedIngestTimes:
    """A bulk ingest registers many sources inside one second.

    ``registered_at`` is stored to the second, so two editions ingested
    by the same `xargs` run carry identical timestamps and the registry
    holds no evidence of which is newer. The chain still has to be
    *deterministic* — the dry run and the write must agree — but the
    order is a guess, and saying so is the whole job.
    """

    def _tied(self, tmp_path: Path) -> tuple[WikiStore, str, str]:
        store, old, new = _wiki_with_sibling_keys(tmp_path)
        _stamp(store, {old: "2025-06-01", new: "2025-06-01"})
        return store, old, new

    def test_the_tie_is_reported(self, tmp_path: Path) -> None:
        store, old, new = self._tied(tmp_path)
        result = store.rekey_document(
            "guidelines/eucast-2024", "guidelines/eucast-2026"
        )
        assert len(result.tied_order) == 1
        assert set(result.chain) == {old, new}

    def test_preview_and_write_agree(self, tmp_path: Path) -> None:
        """The order is arbitrary but not random: --apply must write the
        chain the dry run printed, or the operator approved something
        else."""
        store, _old, _new = self._tied(tmp_path)
        preview = store.rekey_document(
            "guidelines/eucast-2024", "guidelines/eucast-2026"
        )
        written = store.rekey_document(
            "guidelines/eucast-2024", "guidelines/eucast-2026", dry_run=False
        )
        assert written.chain == preview.chain

    def test_cli_says_so_before_apply(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        store, _old, _new = self._tied(tmp_path)
        store.close()
        main(
            [
                "sources", "rekey", "guidelines/eucast-2024",
                "--to", "guidelines/eucast-2026", "--root", str(tmp_path / "w"),
            ]
        )
        out = capsys.readouterr().out
        assert "same ingest second" in out
        assert "which version `outmem stale` calls current" in out

    def test_untied_chains_say_nothing(self, tmp_path: Path) -> None:
        store, _old, _new = _wiki_with_sibling_keys(tmp_path)
        result = store.rekey_document(
            "guidelines/eucast-2024", "guidelines/eucast-2026"
        )
        assert result.tied_order == ()


class TestNothingIsLost:
    def test_ingestion_history_and_page_refs_survive(self, tmp_path: Path) -> None:
        """The reason rekey exists rather than gc-and-re-ingest.

        Both tables cascade on delete, so the destructive repair the old
        refusal message recommended took the audit trail with it.
        """
        store, old, _new = _wiki_with_sibling_keys(tmp_path)
        store.record_ingestion(
            old,
            prompt="compact the 2024 breakpoints",
            pages_touched=["clinical:breakpoints"],
        )
        before = store.source_refs(old)
        store.rekey_document(
            "guidelines/eucast-2024", "guidelines/eucast-2026", dry_run=False
        )
        entry = _registry(store).entries[old]
        assert [i.prompt for i in entry.ingestions] == ["compact the 2024 breakpoints"]
        assert store.source_refs(old) == before


class TestDryRunAndIdempotence:
    def test_dry_run_writes_nothing(self, tmp_path: Path) -> None:
        store, old, _new = _wiki_with_sibling_keys(tmp_path)
        result = store.rekey_document(
            "guidelines/eucast-2024", "guidelines/eucast-2026"
        )
        assert not result.applied
        assert result.chain  # it still previews the outcome
        registry = _registry(store)
        assert registry.entries[old].document_key == "guidelines/eucast-2024"
        assert registry.entries[old].superseded_by is None

    def test_second_apply_is_a_no_op(self, tmp_path: Path) -> None:
        """The registry is a git-tracked binary — a repeated repair must
        not add a full blob to history for no change."""
        store, _old, _new = _wiki_with_sibling_keys(tmp_path)
        store.rekey_document(
            "guidelines/eucast-2024", "guidelines/eucast-2026", dry_run=False
        )
        head = store.head()
        again = store.rekey_document("guidelines/eucast-2026", dry_run=False)
        assert not again.applied
        assert store.head() == head


class TestRefusals:
    def test_unknown_key_names_the_key(self, tmp_path: Path) -> None:
        store, _old, _new = _wiki_with_sibling_keys(tmp_path)
        with pytest.raises(OutmemError, match="no source holds the identity"):
            store.rekey_document("guidelines/nope", "guidelines/eucast-2026")

    def test_an_edge_from_outside_the_document_is_refused(
        self, tmp_path: Path
    ) -> None:
        """Only reachable via an out-of-band edit, which is the point:
        rewriting the chain under such an edge would leave the outside
        row pointing into a document it is not a version of."""
        store, old, new = _wiki_with_sibling_keys(tmp_path)
        stray = tmp_path / "unrelated.md"
        stray.write_text("something else\n", encoding="utf-8")
        other = store.add_source(stray, into_subdir="guidelines").rel_path
        _out_of_band(
            store,
            "UPDATE sources SET superseded_by = ? WHERE rel_path = ?",
            (old, other),
        )
        with pytest.raises(OutmemError, match="from outside this document"):
            store.rekey_document(
                "guidelines/eucast-2024", "guidelines/eucast-2026", dry_run=False
            )
        assert _registry(store).entries[old].document_key == "guidelines/eucast-2024"
        assert _registry(store).entries[new].superseded_by is None

    def test_adopt_now_points_at_rekey_instead_of_gc(self, tmp_path: Path) -> None:
        """The old message recommended `gc` + re-ingest, which cascades
        the ingestion history away. rekey is the non-destructive answer."""
        store, old, _new = _wiki_with_sibling_keys(tmp_path)
        registry = _registry(store)
        with pytest.raises(OutmemError, match="sources rekey"):
            registry.adopt_document_key(old, "guidelines/something-else")


class TestTheLocalTree:
    """The recurring bug class here: code that opens "the" registry when
    there are two. A licensed handbook gets revised editions exactly like
    a tracked guideline does."""

    def _local_wiki(self, tmp_path: Path) -> WikiStore:
        store = WikiStore.init(tmp_path / "w")
        for name in ("hb-2024.md", "hb-2026.md"):
            src = tmp_path / name
            src.write_text(f"licensed {name}\n", encoding="utf-8")
            store.add_source(src, into_subdir="ref", local=True)
        return store

    def test_a_local_document_can_be_rekeyed(self, tmp_path: Path) -> None:
        store = self._local_wiki(tmp_path)
        result = store.rekey_document("ref/hb-2024", "ref/hb-2026", dry_run=False)
        assert result.applied
        registry = SourceRegistry.load(store.sources_local_path)
        assert registry.entries[result.chain[0]].superseded_by == result.chain[1]

    def test_a_local_rekey_commits_nothing(self, tmp_path: Path) -> None:
        """That registry lives inside the gitignored tree, like the
        sources it indexes."""
        store = self._local_wiki(tmp_path)
        head = store.head()
        store.rekey_document("ref/hb-2024", "ref/hb-2026", dry_run=False)
        assert store.head() == head

    def test_asking_for_a_tree_that_does_not_exist_is_refused(
        self, tmp_path: Path
    ) -> None:
        """Opening a registry creates its directory, and for the local
        tree that would leave one without the .gitignore entry beside
        it — the exact hole the split exists to close."""
        store = WikiStore.init(tmp_path / "w")
        with pytest.raises(OutmemError, match="no sources-local/ tree"):
            store.rekey_document("ref/hb-2024", local=True)
        assert not store.sources_local_path.exists()


class TestCli:
    def test_dry_run_prints_the_chain_then_applies(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        store, old, new = _wiki_with_sibling_keys(tmp_path)
        store.close()
        argv = [
            "sources", "rekey", "guidelines/eucast-2024",
            "--to", "guidelines/eucast-2026", "--root", str(tmp_path / "w"),
        ]
        assert main(argv) == 0
        out = capsys.readouterr().out
        assert old in out and new in out
        assert "(current)" in out
        assert "dry run" in out

        assert main([*argv, "--apply"]) == 0
        registry = SourceRegistry.load(tmp_path / "w" / "wiki" / "sources")
        assert registry.entries[old].superseded_by == new

    def test_unknown_key_exits_nonzero(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        store, _old, _new = _wiki_with_sibling_keys(tmp_path)
        store.close()
        rc = main(
            ["sources", "rekey", "guidelines/nope", "--root", str(tmp_path / "w")]
        )
        assert rc == 1
        assert "no source holds the identity" in capsys.readouterr().err
