"""``superseded_ok:`` — an acknowledgement that expires by itself.

Some pages cite an old version on purpose: a page that *compares* two
editions has to name both, and reporting it as stale forever trains the
reader to skip the report. So a citation can record why.

The design decision this file pins is that the acknowledgement is
**scoped to the version it was made against**, exactly as ``finding:``
is. "We deliberately cite 2024 while 2026 exists" says nothing about
2027. A permanent suppression would restore the silent staleness the
whole feature exists to break — one level up, and worse, because a human
has signed it.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest

from outmem.cli.__main__ import main
from outmem.lint import Severity, lint_wiki, provenance_annotation
from outmem.store import WikiStore


def _wiki(tmp_path: Path, *, ack: str | None, date: str | None = "2026-06-01"):
    """A page citing v1 of a source whose v2 was registered 2026-05-01."""
    store = WikiStore.init(tmp_path / "w")
    v1 = tmp_path / "v1.md"
    v1.write_text("breakpoints, 2024 edition\n", encoding="utf-8")
    e1 = store.add_source(v1, into_subdir="g", rename="doc.md", as_key="g/eucast")
    entry: dict[str, object] = {"path": f"sources/{e1.rel_path}"}
    if ack is not None:
        entry["superseded_ok"] = ack
    if date is not None:
        entry["date"] = date
    store.write_page(
        "clinical:breakpoints",
        title="Breakpoints",
        body="Compares the 2024 and 2026 tables.\n",
        provenance=[entry],
    )
    v2 = tmp_path / "v2.md"
    v2.write_text("breakpoints, 2026 edition\n", encoding="utf-8")
    e2 = store.add_source(v2, into_subdir="g", rename="doc.md", as_key="g/eucast")
    _stamp(store, e2.rel_path, "2026-05-01")
    return store, e1.rel_path, e2.rel_path


def _stamp(store: WikiStore, rel_path: str, day: str) -> None:
    import sqlite3

    from outmem.sources import REGISTRY_FILENAME

    con = sqlite3.connect(store.sources_path / REGISTRY_FILENAME)
    with con:
        con.execute(
            "UPDATE sources SET registered_at = ? WHERE rel_path = ?",
            (f"{day}T00:00:00Z", rel_path),
        )
    con.close()
    store._source_registry = None


class TestTheAcknowledgementHolds:
    def test_an_acked_citation_is_not_reported(self, tmp_path: Path) -> None:
        store, _v1, _v2 = _wiki(tmp_path, ack="compares both editions")
        assert store.stale_pages()[0] == []

    def test_it_is_still_there_when_asked_for(self, tmp_path: Path) -> None:
        """Hidden, not discarded — a curator auditing the decisions needs
        to see what was signed off and why."""
        store, v1, _v2 = _wiki(tmp_path, ack="compares both editions")
        (row,) = store.stale_pages(include_acknowledged=True)[0]
        assert row.cited == v1
        assert row.acknowledged == "compares both editions"

    def test_an_unacked_citation_is_reported_as_before(self, tmp_path: Path) -> None:
        store, v1, v2 = _wiki(tmp_path, ack=None)
        (row,) = store.stale_pages()[0]
        assert (row.cited, row.current) == (v1, v2)
        assert row.acknowledged is None

    def test_an_ack_dated_the_same_day_counts(self, tmp_path: Path) -> None:
        """Compared as dates, not instants: an ack written the same day a
        version landed is about that version, whatever hour each got."""
        store, _v1, _v2 = _wiki(tmp_path, ack="checked", date="2026-05-01")
        assert store.stale_pages()[0] == []


class TestTheAcknowledgementExpires:
    def test_a_newer_version_fires_the_row_again(self, tmp_path: Path) -> None:
        store, v1, _v2 = _wiki(tmp_path, ack="compares both editions")
        assert store.stale_pages()[0] == []
        v3 = tmp_path / "v3.md"
        v3.write_text("breakpoints, 2027 edition\n", encoding="utf-8")
        e3 = store.add_source(v3, into_subdir="g", rename="doc.md", as_key="g/eucast")
        _stamp(store, e3.rel_path, "2027-01-01")
        (row,) = store.stale_pages()[0]
        assert row.cited == v1
        assert row.current == e3.rel_path
        assert row.acknowledged is None

    def test_an_ack_predating_the_head_never_applied(self, tmp_path: Path) -> None:
        """Written before the version it is supposed to acknowledge, so
        it is about some earlier one and says nothing about this."""
        store, v1, _v2 = _wiki(tmp_path, ack="checked", date="2025-01-01")
        (row,) = store.stale_pages()[0]
        assert row.cited == v1
        assert row.acknowledged is None


class TestSuppressionNeedsSomethingToCompare:
    @pytest.mark.parametrize("date", [None, "not-a-date", ""])
    def test_no_usable_date_means_no_suppression(
        self, tmp_path: Path, date: str | None
    ) -> None:
        """Suppressing here would be a guess, and a guess that hides a
        stale clinical page is the wrong way to be wrong."""
        store, _v1, _v2 = _wiki(tmp_path, ack="checked", date=date)
        assert len(store.stale_pages()[0]) == 1


class TestAnnotationParsing:
    def test_yaml_may_hand_back_a_date_object_or_a_string(self) -> None:
        """An unquoted `2026-08-12` is parsed by PyYAML into a date and a
        quoted one is left a string, and the file looks the same either
        way. A suppression must not depend on reaching for quotes."""
        as_date = provenance_annotation({"superseded_ok": "x", "date": dt.date(2026, 8, 12)})
        as_text = provenance_annotation({"superseded_ok": "x", "date": "2026-08-12"})
        assert as_date.date == as_text.date == dt.date(2026, 8, 12)

    def test_a_bare_date_records_nothing(self) -> None:
        assert not provenance_annotation({"date": "2026-08-12"})

    def test_finding_and_ack_are_read_by_one_walker(self) -> None:
        annotation = provenance_annotation(
            {"path": "sources/x", "finding": "silent", "superseded_ok": "why"}
        )
        assert annotation.finding == "silent"
        assert annotation.superseded_ok == "why"
        assert annotation


class TestLint:
    def _kinds(self, store: WikiStore) -> list[str]:
        return [
            f.kind
            for f in lint_wiki(store.wiki_path, sources_dir=store.sources_path).findings
        ]

    def test_a_missing_date_is_named(self, tmp_path: Path) -> None:
        """Without it the ack silently does nothing, and the author has
        stopped looking at the row."""
        store, _v1, _v2 = _wiki(tmp_path, ack="checked", date=None)
        assert "invalid-supersession-ack" in self._kinds(store)

    def test_an_unparseable_date_is_named(self, tmp_path: Path) -> None:
        store, _v1, _v2 = _wiki(tmp_path, ack="checked", date="last tuesday")
        assert "invalid-supersession-ack" in self._kinds(store)

    def test_an_empty_reason_is_named(self, tmp_path: Path) -> None:
        store, _v1, _v2 = _wiki(tmp_path, ack="   ")
        report = lint_wiki(store.wiki_path, sources_dir=store.sources_path)
        (finding,) = [
            f for f in report.findings if f.kind == "invalid-supersession-ack"
        ]
        assert finding.severity == Severity.WARNING
        assert "say why" in finding.message

    def test_a_well_formed_ack_is_quiet(self, tmp_path: Path) -> None:
        store, _v1, _v2 = _wiki(tmp_path, ack="compares both editions")
        assert "invalid-supersession-ack" not in self._kinds(store)


class TestCli:
    def test_default_hides_acked_rows_but_counts_them(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        store, _v1, _v2 = _wiki(tmp_path, ack="compares both editions")
        store.close()
        rc = main(["stale", "--root", str(tmp_path / "w")])
        out = capsys.readouterr().out
        assert rc == 0
        assert "1 acknowledged" in out
        assert "`--all` to show" in out

    def test_all_shows_the_reason(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        store, _v1, _v2 = _wiki(tmp_path, ack="compares both editions")
        store.close()
        rc = main(["stale", "--all", "--root", str(tmp_path / "w")])
        out = capsys.readouterr().out
        assert rc == 1
        assert "compares both editions" in out

    def test_json_carries_the_ack_and_the_unreadable_pages(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        store, v1, _v2 = _wiki(tmp_path, ack="compares both editions")
        store.close()
        main(["stale", "--all", "--json", "--root", str(tmp_path / "w")])
        payload = json.loads(capsys.readouterr().out)
        assert payload["unreadable"] == []
        (row,) = payload["stale"]
        assert row["cited"] == v1
        assert row["acknowledged"] == "compares both editions"

    def test_json_counts_what_it_hid(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Without `--all` the rows are omitted, and a payload showing a
        bare empty list reads as 'nothing to do' — the silence this
        command exists to break."""
        store, _v1, _v2 = _wiki(tmp_path, ack="compares both editions")
        store.close()
        rc = main(["stale", "--json", "--root", str(tmp_path / "w")])
        payload = json.loads(capsys.readouterr().out)
        assert payload["stale"] == []
        assert payload["acknowledged"] == 1
        assert rc == 0

    def test_json_on_a_clean_wiki_exits_zero(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        store = WikiStore.init(tmp_path / "w")
        store.write_page("p", title="P", body="b\n")
        store.close()
        rc = main(["stale", "--json", "--root", str(tmp_path / "w")])
        assert rc == 0
        assert json.loads(capsys.readouterr().out) == {
            "stale": [],
            "acknowledged": 0,
            "unreadable": [],
        }

    def test_json_uses_the_same_exit_codes_as_the_text_report(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """2 means a page the check could not run on, in both modes. The
        codes are documented, so a CI gate reading one and getting the
        other silently changes meaning."""
        store, _v1, _v2 = _wiki(tmp_path, ack=None)
        (store.pages_path / "broken.md").write_text(
            "---\nnot: [valid\n---\nbody\n", encoding="utf-8"
        )
        store.close()
        root = str(tmp_path / "w")
        assert main(["stale", "--json", "--root", root]) == 2
        capsys.readouterr()
        assert main(["stale", "--root", root]) == 2
