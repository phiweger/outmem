"""A `provenance:` citation must name a source the registry holds.

Provenance is the edge the rest of outmem hangs off: `outmem stale`
follows it to find pages citing a superseded version, `source_citations`
turns it into a liveness signal, `superseded_ok:` and `finding:` annotate
it, `provenance-sha-mismatch` compares it. A citation to nothing opts a
page out of every one of those while looking, on the page, exactly like a
citation to something.

Refused at the write for the same reason an elided body is: that is the
moment the author — or the model, with the source still in context — is
present and can fix it. Lint is a report somebody reads later.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from outmem.exceptions import UnregisteredProvenanceError
from outmem.git_ops import head_or_none
from outmem.store import WikiStore

MISSING = "guidelines/deadbeef1234/nope.pdf"


@pytest.fixture
def wiki(tmp_path: Path) -> WikiStore:
    return WikiStore.init(tmp_path / "w")


def _doc(tmp_path: Path, name: str = "guide.md", body: str = "Guidance.\n") -> Path:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


class TestWritesAreRefused:
    def test_a_bare_unregistered_ref_leaves_nothing_behind(
        self, wiki: WikiStore
    ) -> None:
        # The whole point of validating before any disk write: a refused
        # call must not half-apply — no page file, no regenerated index, no
        # commit. (`init` leaves its own scaffold untracked, so the test is
        # "HEAD did not move", not "the tree is clean".)
        wiki.write_page("anchor", title="Anchor", body="Body.\n")
        before = head_or_none(wiki.repo)
        index_before = (wiki.wiki_path / "index.md").read_text(encoding="utf-8")

        with pytest.raises(UnregisteredProvenanceError) as exc:
            wiki.write_page("abx:p1", title="P", body="Body.\n", provenance=[MISSING])
        assert MISSING in str(exc.value)
        assert exc.value.refs == (MISSING,)
        assert not wiki.exists("abx:p1")
        assert not (wiki.pages_path / "abx" / "p1.md").exists()
        assert head_or_none(wiki.repo) == before
        assert (wiki.wiki_path / "index.md").read_text(encoding="utf-8") == index_before

    def test_a_mapping_entry_with_a_finding_is_refused_the_same_way(
        self, wiki: WikiStore
    ) -> None:
        with pytest.raises(UnregisteredProvenanceError):
            wiki.write_page(
                "abx:p2",
                title="P",
                body="Body.\n",
                provenance=[
                    {
                        "path": MISSING,
                        "finding": "silent",
                        "scope": "x",
                        "note": "n",
                        "date": "2026-09-11",
                    }
                ],
            )
        assert not wiki.exists("abx:p2")

    def test_the_wiki_prefixed_spelling_is_refused_for_being_unregistered(
        self, wiki: WikiStore
    ) -> None:
        # Not for its prefix: `wiki/sources/…` is a supported spelling (it is
        # what `grep_wiki` prints). It is refused because no row names it,
        # which is the same reason as every other form here.
        with pytest.raises(UnregisteredProvenanceError):
            wiki.write_page(
                "abx:p3", title="P", body="Body.\n", provenance=[f"wiki/sources/{MISSING}"]
            )
        assert not wiki.exists("abx:p3")

    def test_the_error_names_the_accepted_forms(self, wiki: WikiStore) -> None:
        # The message is the only thing an agent caller gets to act on.
        with pytest.raises(UnregisteredProvenanceError) as exc:
            wiki.write_page("p", title="P", body="Body.\n", provenance=[MISSING])
        text = str(exc.value)
        assert "list_sources" in text
        assert "sources-local/<sha>/file.md" in text

    def test_every_offending_ref_is_named_not_just_the_first(
        self, wiki: WikiStore, tmp_path: Path
    ) -> None:
        good = wiki.add_source(_doc(tmp_path))
        with pytest.raises(UnregisteredProvenanceError) as exc:
            wiki.write_page(
                "p",
                title="P",
                body="Body.\n",
                provenance=[good.citation_path, MISSING, "other/beef/x.md"],
            )
        assert exc.value.refs == (MISSING, "other/beef/x.md")


class TestWritesThatShouldStillWork:
    def test_a_registered_ref_commits_and_is_cited(
        self, wiki: WikiStore, tmp_path: Path
    ) -> None:
        entry = wiki.add_source(_doc(tmp_path))
        wiki.write_page("p", title="P", body="Body.\n", provenance=[entry.citation_path])
        citations, _failures = wiki.source_citations()
        assert citations.get(entry.rel_path) == ["p"]

    @pytest.mark.parametrize("form", ["bare", "prefixed", "wiki-prefixed"])
    def test_all_three_ref_spellings_are_accepted(
        self, wiki: WikiStore, tmp_path: Path, form: str
    ) -> None:
        entry = wiki.add_source(_doc(tmp_path))
        ref = {
            "bare": entry.rel_path,
            "prefixed": entry.citation_path,
            "wiki-prefixed": f"wiki/{entry.citation_path}",
        }[form]
        wiki.write_page("p", title="P", body="Body.\n", provenance=[ref])
        assert wiki.exists("p")

    def test_a_superseded_row_is_legal_to_cite(
        self, wiki: WikiStore, tmp_path: Path
    ) -> None:
        # This is what `outmem stale` and `superseded_ok:` exist for. A row
        # existing is the criterion, not being current.
        v1 = wiki.add_source(_doc(tmp_path, "v1.md", "Edition 1.\n"), as_key="ref/g")
        wiki.add_source(_doc(tmp_path, "v2.md", "Edition 2.\n"), as_key="ref/g")
        wiki.write_page("p", title="P", body="From ed.1\n", provenance=[v1.citation_path])
        assert wiki.exists("p")
        stale, _failures = wiki.stale_pages()
        assert [c.slug for c in stale] == ["p"]

    def test_a_registered_row_whose_file_is_gone_is_still_citable(
        self, wiki: WikiStore, tmp_path: Path
    ) -> None:
        # Whether the file is on disk is a lint concern, not a write-time
        # one — the row is what the write path asks about.
        entry = wiki.add_source(_doc(tmp_path))
        (wiki.sources_path / entry.rel_path).unlink()
        wiki.write_page("p", title="P", body="Body.\n", provenance=[entry.citation_path])
        assert wiki.exists("p")

    def test_a_page_with_no_provenance_is_fine(self, wiki: WikiStore) -> None:
        # Navigation hubs have none; unchanged.
        wiki.write_page("hub", title="Hub", body="See [[other]].\n")
        assert wiki.exists("hub")

    def test_an_annotation_only_entry_is_left_to_lint(self, wiki: WikiStore) -> None:
        # No ref to resolve, so nothing for this guard to check.
        wiki.write_page(
            "p", title="P", body="Body.\n", provenance=[{"note": "checked by hand"}]
        )
        assert wiki.exists("p")


class TestLocalTree:
    def test_a_local_ref_that_resolves_commits(
        self, wiki: WikiStore, tmp_path: Path
    ) -> None:
        entry = wiki.add_source(_doc(tmp_path), local=True)
        wiki.write_page("p", title="P", body="Body.\n", provenance=[entry.citation_path])
        assert wiki.exists("p")

    def test_the_same_key_with_no_local_row_is_refused(
        self, wiki: WikiStore
    ) -> None:
        with pytest.raises(UnregisteredProvenanceError):
            wiki.write_page(
                "p", title="P", body="Body.\n", provenance=["sources-local/beef/x.md"]
            )


class TestExtendAndAppend:
    def test_extend_with_no_provenance_does_not_revalidate(
        self, wiki: WikiStore
    ) -> None:
        # A page carrying a dangling ref — written before the check, or with
        # the opt-out — must stay editable. Re-checking here would refuse an
        # unrelated body edit for a problem lint already reports.
        wiki.write_page(
            "p",
            title="P",
            body="Body.\n",
            provenance=[MISSING],
            allow_unregistered_provenance=True,
        )
        wiki.extend_page("p", body="Revised body.\n")
        assert "Revised" in wiki.read("p").body

    def test_extend_validates_what_it_sets(self, wiki: WikiStore) -> None:
        wiki.write_page("p", title="P", body="Body.\n")
        with pytest.raises(UnregisteredProvenanceError):
            wiki.extend_page("p", body="Revised.\n", provenance=[MISSING])
        assert "Revised" not in wiki.read("p").body

    def test_append_refuses_as_a_whole(self, wiki: WikiStore, tmp_path: Path) -> None:
        # One good ref and one bad: nothing is appended, so the caller does
        # not have to guess which half took.
        good = wiki.add_source(_doc(tmp_path))
        wiki.write_page("p", title="P", body="First.\n")
        with pytest.raises(UnregisteredProvenanceError):
            wiki.append_page(
                "p", body="Second.\n", provenance=[good.citation_path, MISSING]
            )
        page = wiki.read("p")
        assert "Second." not in page.body
        assert page.frontmatter.provenance == []


class TestOptOut:
    def test_it_restores_the_old_behaviour(self, wiki: WikiStore) -> None:
        wiki.write_page(
            "p",
            title="P",
            body="Body.\n",
            provenance=[MISSING],
            allow_unregistered_provenance=True,
        )
        assert wiki.exists("p")

    def test_the_tool_palette_does_not_offer_it(self, wiki: WikiStore) -> None:
        # Same precedent as `omitted:` and `allow_elision`: an escape hatch
        # in a tool argument is one a model learns to tick.
        import inspect

        from outmem.adapters.pydantic_ai import wiki_tools

        for tool in wiki_tools(wiki):
            params = inspect.signature(tool).parameters
            assert "allow_unregistered_provenance" not in params


class TestPydanticAiTool:
    @pytest.mark.parametrize(
        ("name", "kwargs"),
        [
            ("write_page", {"slug": "fresh", "title": "P"}),
            ("extend_page", {"slug": "p"}),
            ("append_page", {"slug": "p"}),
        ],
    )
    def test_the_refusal_comes_back_as_a_retry_not_a_string(
        self, wiki: WikiStore, name: str, kwargs: dict[str, str]
    ) -> None:
        # Returned as a string, an error arrives as commentary *after* the
        # call succeeded. ModelRetry is the only path that gets the model to
        # cite something real. All three write tools, because all three
        # validate.
        from pydantic_ai import ModelRetry

        from outmem.adapters.pydantic_ai import wiki_tools

        wiki.write_page("p", title="P", body="Existing.\n")
        tool = next(t for t in wiki_tools(wiki) if t.__name__ == name)
        with pytest.raises(ModelRetry) as exc:
            tool(body="Body.\n", provenance=[MISSING], **kwargs)
        assert MISSING in str(exc.value)
        assert "list_sources" in str(exc.value)
        assert not wiki.exists("fresh")
        assert wiki.read("p").body.strip() == "Existing."


class TestTheRegistryIsReReadBeforeRefusing:
    def test_a_source_registered_by_another_handle_is_accepted(
        self, tmp_path: Path
    ) -> None:
        # The deployment this guard was written for: outmem's write path
        # behind a long-lived server, with ingestion happening elsewhere.
        # The registry is cached for the store's lifetime, so the server's
        # snapshot predates the row — and refusing would hand back advice
        # ("cite one `list_sources` shows") that its own `list_sources`
        # cannot satisfy, because that is stale too.
        server = WikiStore.init(tmp_path / "w")
        server.list_sources()  # warm the cached snapshot

        ingester = WikiStore.open(tmp_path / "w", read_only=False)
        entry = ingester.add_source(_doc(tmp_path))

        server.write_page("p", title="P", body="Body.\n", provenance=[entry.citation_path])
        assert server.exists("p")
        # And the re-read leaves the store's own view correct, not just
        # this one call.
        assert [e.citation_path for e in server.list_sources()] == [entry.citation_path]

    def test_a_genuinely_absent_source_is_still_refused(self, wiki: WikiStore) -> None:
        # The re-read must not soften the check into "accept on a second
        # look": a ref nothing registered stays refused.
        wiki.list_sources()
        with pytest.raises(UnregisteredProvenanceError):
            wiki.write_page("p", title="P", body="Body.\n", provenance=[MISSING])


class TestARefUnrepresentableOnDisk:
    def test_it_is_refused_not_an_oserror(self, wiki: WikiStore) -> None:
        # A citation of several hundred junk characters is exactly what a
        # model invents when it guesses. Resolving it used to raise
        # ENAMETOOLONG straight out of the write, past every handler that
        # expects an OutmemError — so the agent's turn died instead of
        # retrying.
        with pytest.raises(UnregisteredProvenanceError):
            wiki.write_page("p", title="P", body="Body.\n", provenance=["x" * 500])
        assert not wiki.exists("p")

    def test_the_read_path_answers_rather_than_raising(self, wiki: WikiStore) -> None:
        # Same resolver, and the read tools reach it with whatever string
        # the model passed.
        assert wiki.get_source("x" * 500) is None


class TestLintSaysWhichProblem:
    def _kinds(self, store: WikiStore) -> dict[str, str]:
        from outmem.lint import lint_wiki

        report = lint_wiki(
            store.wiki_path,
            log_dir=store.log_path,
            sources_dir=store.sources_path,
            sources_local_dir=store.sources_local_path,
            repo_root=store.repo,
        )
        return {f.kind: f.message for f in report.findings}

    def test_never_registered_is_unregistered_provenance(
        self, wiki: WikiStore
    ) -> None:
        wiki.write_page(
            "p",
            title="P",
            body="Body.\n",
            provenance=[MISSING],
            allow_unregistered_provenance=True,
        )
        found = self._kinds(wiki)
        assert "unregistered-provenance" in found
        assert "stale-provenance" not in found
        # "Restore the source" is advice for a file that was deleted.
        assert "Register the source" in found["unregistered-provenance"]

    def test_registered_then_deleted_is_stale_provenance(
        self, wiki: WikiStore, tmp_path: Path
    ) -> None:
        entry = wiki.add_source(_doc(tmp_path))
        wiki.write_page("p", title="P", body="Body.\n", provenance=[entry.citation_path])
        (wiki.sources_path / entry.rel_path).unlink()
        found = self._kinds(wiki)
        assert "stale-provenance" in found
        assert "unregistered-provenance" not in found
        assert "restore the source" in found["stale-provenance"]

    def test_a_local_citation_with_a_missing_file_is_stale_not_unregistered(
        self, wiki: WikiStore, tmp_path: Path
    ) -> None:
        # The two trees carry separate registries. Without consulting the
        # local one, a `sources-local/` citation whose file went missing
        # would read as having no row at all — the wrong remedy. The file
        # has to be gone for the split to be reached at all.
        entry = wiki.add_source(_doc(tmp_path), local=True)
        wiki.write_page("p", title="P", body="Body.\n", provenance=[entry.citation_path])
        (wiki.sources_local_path / entry.rel_path).unlink()
        found = self._kinds(wiki)
        assert "stale-provenance" in found
        assert "unregistered-provenance" not in found

    def test_lint_agrees_with_the_store_on_the_wiki_prefixed_spelling(
        self, wiki: WikiStore, tmp_path: Path
    ) -> None:
        # `split_tree_prefix` accepts three spellings and the store
        # resolves all three, so `outmem stale` follows a `wiki/sources/…`
        # citation happily — but lint had its own resolver that did not
        # strip the wiki directory, and reported the page as citing
        # something nothing names. Two resolvers, two answers, and the
        # louder one was wrong.
        entry = wiki.add_source(_doc(tmp_path))
        prefixed = f"{wiki.config.wiki_dir}/{entry.citation_path}"
        wiki.write_page("p", title="P", body="Body.\n", provenance=[prefixed])

        found = self._kinds(wiki)
        assert "unregistered-provenance" not in found
        assert "stale-provenance" not in found
        # And it is genuinely resolved, not merely unreported.
        citations, _ = wiki.source_citations()
        assert citations[entry.rel_path] == ["p"]

    def test_the_registry_lookup_is_not_cached_between_runs(
        self, wiki: WikiStore, tmp_path: Path
    ) -> None:
        # The lookups are cached to avoid one sqlite open per provenance
        # entry within a run. Surviving between runs would make a second
        # lint describe a registry that has since changed — and which
        # remedy it advises depends on that answer.
        from outmem.sources import REGISTRY_FILENAME

        entry = wiki.add_source(_doc(tmp_path))
        wiki.write_page(
            "p", title="P", body="Body.\n", provenance=[entry.citation_path]
        )
        (wiki.sources_path / entry.rel_path).unlink()
        assert "stale-provenance" in self._kinds(wiki)  # the row is still there

        (wiki.sources_path / REGISTRY_FILENAME).unlink()  # now the row is gone too
        found = self._kinds(wiki)
        assert "unregistered-provenance" in found
        assert "stale-provenance" not in found
