"""Lint verification of the access-control invariants.

The store maintains these rules at write time. Lint is what makes them
true of the corpus that is already on disk — the case write-time
enforcement structurally cannot reach, and the one a wiki hits the day
it turns restrictions on over years of existing content.

The doctrine is the same one the tracked/local containment checks
follow: a guarantee is only worth stating if something verifies it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from outmem.lint import lint_wiki
from outmem.restricted import RestrictedSettings, settings_from_dict
from outmem.store import WikiStore


def _settings(**block: object) -> RestrictedSettings:
    return settings_from_dict({"labels": ["hr", "legal"], **block})


@pytest.fixture
def store(tmp_path: Path) -> WikiStore:
    """A wiki that declares the labels the checks below use.

    Ingest validates `--restricted` against the declared set, so the
    config block has to be real even though lint takes its settings as an
    argument.
    """
    import yaml

    root = tmp_path / "w"
    store = WikiStore.init(root)
    raw = yaml.safe_load((root / "config.yaml").read_text()) or {}
    raw["restricted"] = {"labels": ["hr", "legal"]}
    (root / "config.yaml").write_text(yaml.safe_dump(raw))
    store.close()
    return WikiStore.open(root)


def _run(store: WikiStore, settings: RestrictedSettings | None = None):  # type: ignore[no-untyped-def]
    return lint_wiki(
        store.wiki_path,
        log_dir=store.log_path,
        sources_dir=store.sources_path,
        sources_local_dir=store.sources_local_path,
        repo_root=store.root,
        restricted=settings if settings is not None else _settings(),
    )


def _kinds(report) -> set[str]:  # type: ignore[no-untyped-def]
    return {f.kind for f in report.findings}


class TestDisabledByDefault:
    def test_no_settings_means_no_restriction_findings(
        self, store: WikiStore
    ) -> None:
        """A wiki that declares no labels must not start seeing findings
        about a feature it does not use."""
        store.write_page("a", title="A", body="Text.\n")
        report = lint_wiki(store.wiki_path, restricted=None)
        assert not any(f.kind.startswith("restricted-") for f in report.findings)

    def test_an_empty_declared_set_is_also_off(self, store: WikiStore) -> None:
        store.write_page("a", title="A", body="Text.\n")
        report = lint_wiki(store.wiki_path, restricted=RestrictedSettings())
        assert not any(f.kind.startswith("restricted-") for f in report.findings)


class TestLinkViolation:
    def test_a_visible_page_linking_to_a_restricted_one(
        self, store: WikiStore
    ) -> None:
        """Filtering the page list achieves nothing if the open page's
        body contains the restricted slug."""
        store.write_page(
            "hr:severance", title="S", body="Text.\n", extra={"restricted": ["hr"]}
        )
        store.write_page("public", title="P", body="See [[hr:severance]].\n")
        assert "restricted-link-violation" in _kinds(_run(store))

    def test_linking_up_is_fine(self, store: WikiStore) -> None:
        """Everyone who can see the HR page can already see the glossary."""
        store.write_page("glossary", title="G", body="Terms.\n")
        store.write_page(
            "hr:x",
            title="X",
            body="See [[glossary]].\n",
            extra={"restricted": ["hr"]},
        )
        assert "restricted-link-violation" not in _kinds(_run(store))

    def test_a_dangling_link_is_not_a_restriction_finding(
        self, store: WikiStore
    ) -> None:
        store.write_page("a", title="A", body="See [[nope]].\n")
        assert "restricted-link-violation" not in _kinds(_run(store))

    def test_the_message_gives_the_fix(self, store: WikiStore) -> None:
        store.write_page(
            "hr:severance", title="S", body="Text.\n", extra={"restricted": ["hr"]}
        )
        store.write_page("public", title="P", body="See [[hr:severance]].\n")
        finding = next(
            f for f in _run(store).findings if f.kind == "restricted-link-violation"
        )
        assert "outmem restrict public --label hr" in finding.message

    def test_path_rules_count_as_labels(self, store: WikiStore) -> None:
        """A namespace restricted only by config must still be protected
        by closure, or the safety net has a hole the frontmatter does not."""
        store.write_page("hr:pay", title="Pay", body="Bands.\n")
        store.write_page("public", title="P", body="See [[hr:pay]].\n")
        report = _run(store, _settings(paths={"hr:*": ["hr"]}))
        assert "restricted-link-violation" in _kinds(report)


class TestProvenanceViolation:
    def test_a_page_citing_a_more_restricted_source(
        self, store: WikiStore, tmp_path: Path
    ) -> None:
        """A source's path embeds its filename, so the citation itself is
        disclosure even when the file is unreadable."""
        doc = tmp_path / "severance-plan-2026.md"
        doc.write_text("Plan.\n")
        entry = store.add_source(doc, restricted=["hr"])
        store.write_page(
            "public", title="P", body="A fact.\n", provenance=[entry.citation_path]
        )
        assert "restricted-provenance-violation" in _kinds(_run(store))

    def test_a_matching_page_is_clean(
        self, store: WikiStore, tmp_path: Path
    ) -> None:
        doc = tmp_path / "memo.md"
        doc.write_text("Memo.\n")
        entry = store.add_source(doc, restricted=["hr"])
        store.write_page(
            "hr:derived",
            title="D",
            body="A fact.\n",
            provenance=[entry.citation_path],
            extra={"restricted": ["hr"]},
        )
        assert "restricted-provenance-violation" not in _kinds(_run(store))

    def test_an_open_source_raises_nothing(
        self, store: WikiStore, tmp_path: Path
    ) -> None:
        doc = tmp_path / "public.md"
        doc.write_text("Public.\n")
        entry = store.add_source(doc)
        store.write_page(
            "p", title="P", body="A fact.\n", provenance=[entry.citation_path]
        )
        assert "restricted-provenance-violation" not in _kinds(_run(store))


class TestSlugMentionedAsProse:
    def test_an_unlinked_mention_is_a_warning(self, store: WikiStore) -> None:
        """It discloses exactly as much as a link, and nothing else in the
        system would ever notice it."""
        store.write_page(
            "hr:severance", title="S", body="Text.\n", extra={"restricted": ["hr"]}
        )
        store.write_page(
            "public", title="P", body="The rules are in hr:severance.\n"
        )
        findings = [
            f for f in _run(store).findings if f.kind == "restricted-slug-mentioned"
        ]
        assert findings
        assert findings[0].severity.name == "WARNING"

    def test_it_reports_a_file_line_number(self, store: WikiStore) -> None:
        """A warning about prose is only actionable if it says where."""
        store.write_page(
            "hr:severance", title="S", body="Text.\n", extra={"restricted": ["hr"]}
        )
        store.write_page(
            "public", title="P", body="One.\n\nTwo hr:severance three.\n"
        )
        finding = next(
            f for f in _run(store).findings if f.kind == "restricted-slug-mentioned"
        )
        assert finding.line and finding.line > 1

    def test_a_page_mentioning_its_own_slug_is_fine(
        self, store: WikiStore
    ) -> None:
        store.write_page(
            "hr:severance",
            title="S",
            body="This page, hr:severance, covers it.\n",
            extra={"restricted": ["hr"]},
        )
        assert "restricted-slug-mentioned" not in _kinds(_run(store))


class TestUnknownLabel:
    def test_a_label_absent_from_config_is_an_error(
        self, store: WikiStore
    ) -> None:
        """A label nobody can hold hides the page from everyone, silently
        — including whoever it was written for."""
        store.write_page(
            "p", title="P", body="Text.\n", extra={"restricted": ["board"]}
        )
        assert "restricted-label-unknown" in _kinds(_run(store))

    def test_a_declared_label_is_clean(self, store: WikiStore) -> None:
        store.write_page(
            "p", title="P", body="Text.\n", extra={"restricted": ["hr"]}
        )
        assert "restricted-label-unknown" not in _kinds(_run(store))

    def test_the_message_says_what_the_consequence_is(
        self, store: WikiStore
    ) -> None:
        store.write_page(
            "p", title="P", body="Text.\n", extra={"restricted": ["board"]}
        )
        finding = next(
            f for f in _run(store).findings if f.kind == "restricted-label-unknown"
        )
        assert "hides this page from everyone" in finding.message


class TestUnparseableFrontmatter:
    def test_it_is_reported_in_access_control_terms(
        self, store: WikiStore
    ) -> None:
        """`frontmatter-invalid` already reports the parse failure. The
        consequence differs in kind with restrictions on: the page is not
        merely malformed, it is gone until somebody fixes it."""
        (store.pages_path / "broken.md").write_text("no frontmatter\n")
        kinds = _kinds(_run(store))
        assert "restricted-frontmatter-unparseable" in kinds
        assert "frontmatter-invalid" in kinds

    def test_a_clean_wiki_reports_neither(self, store: WikiStore) -> None:
        store.write_page("p", title="P", body="Text.\n")
        assert "restricted-frontmatter-unparseable" not in _kinds(_run(store))


class TestChainConsistency:
    def test_versions_with_different_labels_are_an_error(
        self, store: WikiStore, tmp_path: Path
    ) -> None:
        """`outmem stale` points pages at the current version, so an open
        head silently declassifies a restricted document."""
        from outmem._store.sources import get_registry

        v1 = tmp_path / "v1.md"
        v1.write_text("One.\n")
        v2 = tmp_path / "v2.md"
        v2.write_text("Two.\n")
        first = store.add_source(v1, as_key="policy/sev", restricted=["hr"])
        second = store.add_source(v2, as_key="policy/sev")
        assert second.restricted == frozenset({"hr"})  # inherited, as designed

        # Reproduce the state a pre-inheritance build (or a hand edit)
        # could leave behind, which is what the lint backstop is for.
        registry = get_registry(store, None)
        registry.set_restricted(second.rel_path, set(), allow_narrowing=True)
        assert first.rel_path  # the chain still exists

        assert "restricted-chain-inconsistent" in _kinds(_run(store))

    def test_a_consistent_chain_is_clean(
        self, store: WikiStore, tmp_path: Path
    ) -> None:
        v1 = tmp_path / "v1.md"
        v1.write_text("One.\n")
        v2 = tmp_path / "v2.md"
        v2.write_text("Two.\n")
        store.add_source(v1, as_key="policy/sev", restricted=["hr"])
        store.add_source(v2, as_key="policy/sev")
        assert "restricted-chain-inconsistent" not in _kinds(_run(store))

    def test_the_message_names_both_versions(
        self, store: WikiStore, tmp_path: Path
    ) -> None:
        from outmem._store.sources import get_registry

        v1 = tmp_path / "v1.md"
        v1.write_text("One.\n")
        v2 = tmp_path / "v2.md"
        v2.write_text("Two.\n")
        store.add_source(v1, as_key="policy/sev", restricted=["hr"])
        second = store.add_source(v2, as_key="policy/sev")
        get_registry(store, None).set_restricted(
            second.rel_path, set(), allow_narrowing=True
        )
        finding = next(
            f
            for f in _run(store).findings
            if f.kind == "restricted-chain-inconsistent"
        )
        assert "v1.md" in finding.message and "v2.md" in finding.message


class TestCliRunsThem:
    def test_the_lint_command_reports_a_violation(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        import yaml

        from outmem.cli.__main__ import main

        root = tmp_path / "w"
        store = WikiStore.init(root)
        raw = yaml.safe_load((root / "config.yaml").read_text()) or {}
        raw["restricted"] = {"labels": ["hr"]}
        (root / "config.yaml").write_text(yaml.safe_dump(raw))
        store.close()

        store = WikiStore.open(root)
        store.write_page(
            "hr:severance", title="S", body="Text.\n", extra={"restricted": ["hr"]}
        )
        store.write_page("public", title="P", body="See [[hr:severance]].\n")
        store.close()

        assert main(["lint", "--root", str(root)]) == 2
        assert "restricted-link-violation" in capsys.readouterr().out
