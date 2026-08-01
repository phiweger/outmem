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
    assert "wiki/pages/orphan.md" in orphans
    assert "wiki/pages/hub.md" in orphans
    for f in report.findings:
        if f.kind == "orphan-page":
            assert f.severity == Severity.WARNING


def test_orphan_with_log_mention_is_not_flagged(tmp_path: Path) -> None:
    store = WikiStore.init(tmp_path / "w")
    store.write_page("documented", title="Documented", body="No inbound wikilinks here.")
    store.append_log(topic="discovery", content="- found [[documented]] today")
    report = lint_wiki(store.wiki_path, log_dir=store.log_path)
    orphans = {f.path for f in report.findings if f.kind == "orphan-page"}
    assert "wiki/pages/documented.md" not in orphans


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
    (store.pages_path / "rogue.md").write_text(
        "---\ntitle: Rogue\nslug: rogue\n---\n\nbody\n",
        encoding="utf-8",
    )
    report = lint_wiki(store.wiki_path, log_dir=store.log_path)
    assert any(f.kind == "index-drift" for f in report.findings)


def test_slug_filename_mismatch_is_error(tmp_path: Path) -> None:
    store = WikiStore.init(tmp_path / "w")
    store.write_page("alpha", title="Alpha", body="body")
    # Hand-edit the file to lie about its slug.
    bad = store.pages_path / "alpha.md"
    text = bad.read_text().replace("slug: alpha", "slug: not-alpha")
    bad.write_text(text)
    report = lint_wiki(store.wiki_path, log_dir=store.log_path)
    kinds = {f.kind for f in report.findings}
    assert "slug-filename-mismatch" in kinds


def test_invalid_frontmatter_is_error(tmp_path: Path) -> None:
    store = WikiStore.init(tmp_path / "w")
    (store.pages_path / "bad.md").write_text("no frontmatter at all", encoding="utf-8")
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

    rc = main(["lint", "--root", str(store.root)])
    assert rc == 0
    assert "no issues" in capsys.readouterr().out.lower()


def test_cli_lint_warnings_exit_1(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from outmem.cli.__main__ import main

    store = WikiStore.init(tmp_path / "w")
    store.write_page("orphan", title="Lonely", body="no links here")
    store.write_page("hub", title="Hub", body="no outbound either")

    rc = main(["lint", "--root", str(store.root)])
    # Warning-only -> exit 1
    assert rc == 1


def test_cli_lint_errors_exit_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from outmem.cli.__main__ import main

    store = WikiStore.init(tmp_path / "w")
    store.write_page("alpha", title="A", body="See [[nonexistent]].")

    rc = main(["lint", "--root", str(store.root)])
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


# ---------------------------------------------------------------------------
# Integrity checks added for the slug-stability work
# ---------------------------------------------------------------------------


def test_dead_slug_mention_flags_prose_references(tmp_path: Path) -> None:
    """A dangling-link check is by construction blind to a slug written as
    prose. That is exactly where dead references accumulate after a
    namespace is reorganised — one production wiki carried them for months."""
    store = WikiStore.init(tmp_path / "w")
    store.write_page("sop:mikrobiologie:geraete:vitek2", title="Vitek", body="v")
    store.write_page(
        "notes:digest",
        title="Digest",
        body="Volltext-Digest: sop:mikrobiologie:vitek2 (alte Struktur)\n",
    )
    report = lint_wiki(store.wiki_path, log_dir=store.log_path, raw_dir=store.raw_path)
    dead = [f for f in report.findings if f.kind == "dead-slug-mention"]
    assert len(dead) == 1
    assert "sop:mikrobiologie:vitek2" in dead[0].message
    assert dead[0].severity == Severity.WARNING


def test_dead_slug_mention_ignores_times_ratios_and_live_pages(tmp_path: Path) -> None:
    """Slug grammar is permissive enough that `12:30` and `3:1` parse as
    slugs, and `clinical:x: prose` uses the second colon as punctuation.
    The namespace gate is what keeps those out."""
    store = WikiStore.init(tmp_path / "w")
    store.write_page("clinical:leishmaniose", title="Leish", body="x")
    store.write_page("clinical:sepsis", title="Sepsis", body="see [[clinical:leishmaniose]]")
    store.write_page(
        "notes:dosing",
        title="Dosing",
        body=(
            "clinical:leishmaniose: L-AmB 3 mg/kg, Abnahme 12:30, Verhaeltnis 3:1\n\n"
            "Link: [[clinical:sepsis]]\n"
        ),
    )
    report = lint_wiki(store.wiki_path, log_dir=store.log_path, raw_dir=store.raw_path)
    dead = [f for f in report.findings if f.kind == "dead-slug-mention"]
    assert dead == [], [f.message for f in dead]


def test_a_renamed_page_mentioned_in_prose_is_not_called_dead(tmp_path: Path) -> None:
    """`outmem rename` records an alias, so the old name still opens. Both
    slug checks predate aliases and tested membership in the canonical slug
    set, so a clean rename made the linter report a name that `read()`
    resolves fine — the linter contradicting the feature next to it."""
    store = WikiStore.init(tmp_path / "w")
    store.write_page("clinical:sepsis", title="Sepsis", body="s")
    store.write_page("clinical:other", title="Other", body="Vgl. clinical:sepsis\n")
    store.rename_page("clinical:sepsis", "clinical:infektion:sepsis")

    assert store.read("clinical:sepsis").slug == "clinical:infektion:sepsis"
    report = lint_wiki(store.wiki_path, log_dir=store.log_path, raw_dir=store.raw_path)
    assert [f for f in report.findings if f.kind == "dead-slug-mention"] == []
    # Prose is editable, so the old name is still debt worth retiring —
    # reported truthfully, the way `wikilink-via-alias` treats a link.
    (via,) = [f for f in report.findings if f.kind == "slug-mention-via-alias"]
    assert via.severity == Severity.WARNING
    assert "clinical:infektion:sepsis" in via.message


def test_a_source_referencing_an_aliased_slug_is_silent(tmp_path: Path) -> None:
    """The alias is doing exactly its job here. A content-addressed source
    cannot be edited, so a nudge would ask the operator to fix something
    unfixable — unlike prose, which is why only that side gets one."""
    store = WikiStore.init(tmp_path / "w")
    store.write_page("sop:vitek2", title="Vitek", body="v")
    store.sources_path.mkdir(parents=True, exist_ok=True)
    (store.sources_path / "t.md").write_text("Vgl. sop:vitek2\n", encoding="utf-8")
    store.rename_page("sop:vitek2", "sop:geraete:vitek2")

    report = lint_wiki(
        store.wiki_path,
        log_dir=store.log_path,
        raw_dir=store.raw_path,
        sources_dir=store.sources_path,
    )
    assert [f for f in report.findings if f.kind == "source-references-dead-slug"] == []
    assert [f for f in report.findings if f.kind == "slug-mention-via-alias"] == []


def test_a_slug_with_no_page_and_no_alias_is_still_dead(tmp_path: Path) -> None:
    """The guarantee the alias fix must not weaken: a genuinely dead
    reference — the 136 found in production — still reports."""
    store = WikiStore.init(tmp_path / "w")
    store.write_page("sop:geraete:vitek2", title="Vitek", body="v")
    store.write_page("sop:notes", title="Notes", body="Vgl. sop:mikrobiologie\n")
    store.sources_path.mkdir(parents=True, exist_ok=True)
    (store.sources_path / "t.md").write_text("Vgl. sop:mikrobiologie\n", encoding="utf-8")
    report = lint_wiki(
        store.wiki_path,
        log_dir=store.log_path,
        raw_dir=store.raw_path,
        sources_dir=store.sources_path,
    )
    assert len([f for f in report.findings if f.kind == "dead-slug-mention"]) == 1
    assert len([f for f in report.findings if f.kind == "source-references-dead-slug"]) == 1


def test_alias_namespaces_do_not_widen_the_false_positive_gate(tmp_path: Path) -> None:
    """Resolution and the namespace gate are separate questions. Folding
    aliases into the namespace set would admit tokens no live page
    justifies — and the gate is the only thing keeping `12:30` out."""
    store = WikiStore.init(tmp_path / "w")
    store.write_page("clinical:sepsis", title="Sepsis", body="s")
    store.write_page("notes:dosing", title="Dosing", body="Abnahme 12:30, Verhaeltnis 3:1\n")
    store.rename_page("clinical:sepsis", "clinical:infektion:sepsis")
    report = lint_wiki(store.wiki_path, log_dir=store.log_path, raw_dir=store.raw_path)
    assert [f for f in report.findings if f.kind == "dead-slug-mention"] == []


def test_duplicate_slug_is_an_error(tmp_path: Path) -> None:
    """Two files claiming one slug silently collided before — the loser
    vanished from every slug-keyed check with no signal."""
    store = WikiStore.init(tmp_path / "w")
    store.write_page("alpha", title="Alpha", body="a")
    rogue = store.pages_path / "beta.md"
    rogue.write_text("---\ntitle: Rogue\nslug: alpha\n---\n\nb\n", encoding="utf-8")
    report = lint_wiki(store.wiki_path, log_dir=store.log_path, raw_dir=store.raw_path)
    dupes = [f for f in report.findings if f.kind == "duplicate-slug"]
    assert len(dupes) == 1
    assert dupes[0].severity == Severity.ERROR


def test_repairable_frontmatter_warns_instead_of_erroring(tmp_path: Path) -> None:
    """A page that `read_page` self-heals must not be a CI-failing ERROR in
    lint while every other reader serves it — but it should still say
    'persist this'."""
    store = WikiStore.init(tmp_path / "w")
    page = store.pages_path / "flu.md"
    page.write_text(
        "---\ntitle: Influenza (Teil 1): Erkrankungen\nslug: flu\n---\n\nbody\n",
        encoding="utf-8",
    )
    report = lint_wiki(store.wiki_path, log_dir=store.log_path, raw_dir=store.raw_path)
    kinds = {f.kind for f in report.findings}
    assert "frontmatter-repairable" in kinds
    assert "frontmatter-invalid" not in kinds


def test_sources_registry_symmetry_both_directions(tmp_path: Path) -> None:
    """Nothing reconciled `.sources.db` against disk, so a registry could
    drift to double-digit percent junk unnoticed."""
    from outmem.sources import SourceRegistry, compute_sha256

    store = WikiStore.init(tmp_path / "w")
    store.sources_path.mkdir(parents=True, exist_ok=True)
    # (a) a row whose file is gone
    reg = SourceRegistry.load(store.sources_path)
    reg.register("gone/deadbeefcafe/x.md", sha256="a" * 64, size_bytes=10)
    # (b) a file with no row
    stray = store.sources_path / "stray.md"
    stray.write_text("unregistered\n", encoding="utf-8")

    report = lint_wiki(
        store.wiki_path,
        log_dir=store.log_path,
        raw_dir=store.raw_path,
        sources_dir=store.sources_path,
    )
    kinds = {f.kind for f in report.findings}
    assert "source-orphaned" in kinds
    assert "source-unregistered" in kinds
    assert compute_sha256(stray)  # sanity: the helper gc will reuse


def test_provenance_sha_mismatch_detects_a_rehashed_source(tmp_path: Path) -> None:
    """`stale-provenance` only stats the file. A source re-ingested after a
    content change lives at a new path, so a page citing the old sha points
    at content it was never compacted from."""
    from outmem.sources import SourceRegistry

    store = WikiStore.init(tmp_path / "w")
    store.sources_path.mkdir(parents=True, exist_ok=True)
    src = store.sources_path / "deck.md"
    src.write_text("v2 content\n", encoding="utf-8")
    reg = SourceRegistry.load(store.sources_path)
    reg.register("deck.md", sha256="b" * 64, size_bytes=11)

    store.write_page(
        "pricing",
        title="Pricing",
        body="p",
        provenance=[{"path": "deck.md", "sha256": "a" * 64}],  # the OLD sha
    )
    store.write_page("ref", title="Ref", body="See [[pricing]].")
    report = lint_wiki(
        store.wiki_path,
        log_dir=store.log_path,
        raw_dir=store.raw_path,
        sources_dir=store.sources_path,
    )
    mism = [f for f in report.findings if f.kind == "provenance-sha-mismatch"]
    assert len(mism) == 1
    assert mism[0].severity == Severity.WARNING


def test_cli_lint_error_only_ignores_warnings(tmp_path: Path) -> None:
    """A wiki carrying known warnings shouldn't fail CI just because a new
    warning-severity check shipped."""
    from outmem.cli.__main__ import main

    store = WikiStore.init(tmp_path / "w")
    store.write_page("orphan", title="Lonely", body="no links here")
    assert main(["lint", "--root", str(store.root)]) == 1
    assert main(["lint", "--root", str(store.root), "--error-only"]) == 0


def test_source_referencing_a_dead_slug_is_flagged(tmp_path: Path) -> None:
    """Content-addressed sources are frozen; page slugs move. A source that
    names slugs couples the two, and the reference rots undetected — 136
    such references were found in one production wiki, all of it in
    self-authored material filed as sources."""
    store = WikiStore.init(tmp_path / "w")
    store.write_page("sop:mikrobiologie:geraete:vitek2", title="Vitek", body="v")
    store.sources_path.mkdir(parents=True, exist_ok=True)
    (store.sources_path / "sop-transcript.md").write_text(
        "Vgl. sop:mikrobiologie:vitek2 (alte Struktur)\n", encoding="utf-8"
    )
    report = lint_wiki(
        store.wiki_path,
        log_dir=store.log_path,
        raw_dir=store.raw_path,
        sources_dir=store.sources_path,
    )
    hits = [f for f in report.findings if f.kind == "source-references-dead-slug"]
    assert len(hits) == 1
    assert "sop:mikrobiologie:vitek2" in hits[0].message
    assert hits[0].severity == Severity.WARNING


def test_source_referencing_a_live_slug_is_not_flagged(tmp_path: Path) -> None:
    """A live reference carries the risk but isn't a defect yet."""
    store = WikiStore.init(tmp_path / "w")
    store.write_page("sop:vitek2", title="Vitek", body="v")
    store.sources_path.mkdir(parents=True, exist_ok=True)
    (store.sources_path / "t.md").write_text("Vgl. sop:vitek2\n", encoding="utf-8")
    report = lint_wiki(
        store.wiki_path,
        log_dir=store.log_path,
        raw_dir=store.raw_path,
        sources_dir=store.sources_path,
    )
    assert [f for f in report.findings if f.kind == "source-references-dead-slug"] == []
