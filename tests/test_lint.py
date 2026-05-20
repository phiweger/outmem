"""Tests for ``outmem.lint`` — static checks over the wiki."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from outmem.lint import Severity, format_report, lint_wiki
from outmem.store import WikiStore


def test_clean_wiki_has_no_findings(tmp_path: Path) -> None:
    store = WikiStore.init(tmp_path / "w")
    store.write_page("acme-msa", title="Acme", body="See [[pricing]] for terms.")
    store.write_page("pricing", title="Pricing", body="Cost-plus. [[acme-msa]] is the exception.")

    report = lint_wiki(store.wiki_path, log_dir=store.log_path)
    assert not report.has_findings


def test_broken_wikilink_is_error(tmp_path: Path) -> None:
    store = WikiStore.init(tmp_path / "w")
    store.write_page("alpha", title="A", body="See [[nonexistent]].")
    report = lint_wiki(store.wiki_path, log_dir=store.log_path)
    kinds = {f.kind for f in report.findings}
    assert "broken-wikilink" in kinds
    broken = [f for f in report.findings if f.kind == "broken-wikilink"]
    assert broken[0].severity == Severity.ERROR


def test_orphan_page_is_warning(tmp_path: Path) -> None:
    store = WikiStore.init(tmp_path / "w")
    store.write_page("orphan", title="Lonely", body="No inbound links.")
    store.write_page("hub", title="Hub", body="No outbound links to orphan.")
    report = lint_wiki(store.wiki_path, log_dir=store.log_path)
    orphans = {f.path for f in report.findings if f.kind == "orphan-page"}
    # Both pages are orphans since neither references the other.
    assert "wiki/orphan.md" in orphans
    assert "wiki/hub.md" in orphans
    for f in report.findings:
        if f.kind == "orphan-page":
            assert f.severity == Severity.WARNING


def test_orphan_with_log_mention_is_not_flagged(tmp_path: Path) -> None:
    store = WikiStore.init(tmp_path / "w")
    store.write_page("documented", title="Documented", body="No inbound wikilinks here.")
    store.append_log(topic="discovery", content="- found [[documented]] today")
    report = lint_wiki(store.wiki_path, log_dir=store.log_path)
    orphans = {f.path for f in report.findings if f.kind == "orphan-page"}
    assert "wiki/documented.md" not in orphans


def test_index_is_never_flagged_as_orphan(tmp_path: Path) -> None:
    store = WikiStore.init(tmp_path / "w")
    store.write_page("alpha", title="Alpha", body="body")
    report = lint_wiki(store.wiki_path, log_dir=store.log_path)
    paths = {f.path for f in report.findings if f.kind == "orphan-page"}
    assert "wiki/index.md" not in paths


def test_stale_provenance_is_warning(tmp_path: Path) -> None:
    store = WikiStore.init(tmp_path / "w")
    store.write_page(
        "alpha",
        title="Alpha",
        body="body",
        provenance=["raw/deleted.md"],
    )
    # Add a counter-link so alpha isn't also flagged as an orphan.
    store.write_page("ref", title="Ref", body="See [[alpha]].")
    report = lint_wiki(
        store.wiki_path,
        log_dir=store.log_path,
        raw_dir=store.raw_path,
    )
    stale = [f for f in report.findings if f.kind == "stale-provenance"]
    assert any("deleted.md" in f.message for f in stale)
    assert stale[0].severity == Severity.WARNING


def test_stale_provenance_dict_entry(tmp_path: Path) -> None:
    """Dict-form provenance is also checked (path: …, sha256: …)."""
    store = WikiStore.init(tmp_path / "w")
    store.write_page(
        "alpha",
        title="Alpha",
        body="body",
        provenance=[{"path": "raw/deleted.md", "sha256": "x"}],
    )
    store.write_page("ref", title="Ref", body="See [[alpha]].")
    report = lint_wiki(
        store.wiki_path,
        log_dir=store.log_path,
        raw_dir=store.raw_path,
    )
    assert any(f.kind == "stale-provenance" for f in report.findings)


def test_present_provenance_not_flagged(tmp_path: Path) -> None:
    store = WikiStore.init(tmp_path / "w")
    (store.raw_path / "real.md").write_text("real source\n", encoding="utf-8")
    store.write_page("alpha", title="Alpha", body="body", provenance=["raw/real.md"])
    store.write_page("ref", title="Ref", body="See [[alpha]].")
    report = lint_wiki(
        store.wiki_path,
        log_dir=store.log_path,
        raw_dir=store.raw_path,
    )
    stale = [f for f in report.findings if f.kind == "stale-provenance"]
    assert stale == []


def test_index_drift_detected(tmp_path: Path) -> None:
    """Simulate a human Obsidian edit that adds a page without going
    through outmem — the index goes stale."""
    store = WikiStore.init(tmp_path / "w")
    store.write_page("alpha", title="Alpha", body="body")
    # Now drop a file directly without using the WikiStore.
    (store.wiki_path / "rogue.md").write_text(
        "---\ntitle: Rogue\nslug: rogue\n---\n\nbody\n",
        encoding="utf-8",
    )
    report = lint_wiki(store.wiki_path, log_dir=store.log_path)
    assert any(f.kind == "index-drift" for f in report.findings)


def test_slug_filename_mismatch_is_error(tmp_path: Path) -> None:
    store = WikiStore.init(tmp_path / "w")
    store.write_page("alpha", title="Alpha", body="body")
    # Hand-edit the file to lie about its slug.
    bad = store.wiki_path / "alpha.md"
    text = bad.read_text().replace("slug: alpha", "slug: not-alpha")
    bad.write_text(text)
    report = lint_wiki(store.wiki_path, log_dir=store.log_path)
    kinds = {f.kind for f in report.findings}
    assert "slug-filename-mismatch" in kinds


def test_invalid_frontmatter_is_error(tmp_path: Path) -> None:
    store = WikiStore.init(tmp_path / "w")
    (store.wiki_path / "bad.md").write_text("no frontmatter at all", encoding="utf-8")
    report = lint_wiki(store.wiki_path, log_dir=store.log_path)
    assert any(f.kind == "frontmatter-invalid" for f in report.findings)


def test_format_report_no_findings() -> None:
    from outmem.lint import LintReport

    out = format_report(LintReport())
    assert "no issues" in out.lower()


def test_format_report_groups_by_kind(tmp_path: Path) -> None:
    store = WikiStore.init(tmp_path / "w")
    store.write_page("orphan", title="O", body="No links.")
    store.write_page("broken", title="B", body="[[nonexistent]]")
    report = lint_wiki(store.wiki_path, log_dir=store.log_path)
    out = format_report(report)
    assert "## broken-wikilink" in out
    assert "## orphan-page" in out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_lint_clean_wiki_exit_0(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from outmem.cli.__main__ import main

    store = WikiStore.init(tmp_path / "w")
    store.write_page("alpha", title="A", body="See [[beta]].")
    store.write_page("beta", title="B", body="See [[alpha]].")

    rc = main(["--root", str(store.root), "lint"])
    assert rc == 0
    assert "no issues" in capsys.readouterr().out.lower()


def test_cli_lint_warnings_exit_1(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from outmem.cli.__main__ import main

    store = WikiStore.init(tmp_path / "w")
    store.write_page("orphan", title="Lonely", body="no links here")
    store.write_page("hub", title="Hub", body="no outbound either")

    rc = main(["--root", str(store.root), "lint"])
    # Warning-only -> exit 1
    assert rc == 1


def test_cli_lint_errors_exit_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from outmem.cli.__main__ import main

    store = WikiStore.init(tmp_path / "w")
    store.write_page("alpha", title="A", body="See [[nonexistent]].")

    rc = main(["--root", str(store.root), "lint"])
    # Error present -> exit 2
    assert rc == 2


def test_cli_lint_default_command_help() -> None:
    """Sanity: `outmem lint --help` succeeds."""
    import argparse

    from outmem.cli.__main__ import build_parser

    parser = build_parser()
    with pytest.raises((SystemExit, argparse.ArgumentError)):
        parser.parse_args(["lint", "--help"])


def _unused_io() -> None:
    """Keeps the io import live for any future tests that need it."""
    _ = io.StringIO("")
