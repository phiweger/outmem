"""Reading several wikis as one.

Nothing here filters. Every store in the set is one the session may read
in full, chosen once before the set existed — so these tests are about
relevance and addressing, not confidentiality. The one thing they *do*
guard is that a set contains what it was given and nothing else: a wiki
absent from the set contributes nothing, because it was never opened.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from outmem.exceptions import OutmemError
from outmem.repo import Repo
from outmem.semantic.testing import make_bag_of_words_handle
from outmem.store import WikiStore
from outmem.wikiset import WikiSet, split_qualified

from .conftest import _run_git

REGISTRY = """\
tags: {everyone: {}, hr: {}, legal: {}}
wikis:
  open:  {path: wikis/open,  audience: [everyone]}
  hr:    {path: wikis/hr,    audience: [hr]}
  legal: {path: wikis/legal, audience: [legal]}
"""


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "mem"
    root.mkdir()
    _run_git(["init", "--initial-branch", "main"], cwd=root)
    (root / "wikis.yaml").write_text(REGISTRY, encoding="utf-8")
    for name in ("open", "hr", "legal"):
        (root / "wikis" / name).mkdir(parents=True)
        WikiStore.init(root / "wikis" / name)
    o = WikiStore.open(root / "wikis" / "open")
    o.write_page("pricing", title="Pricing", body="List price is cost times 2.4.\n")
    o.write_page("shared", title="Open shared", body="The open version.\n")
    lg = WikiStore.open(root / "wikis" / "legal")
    lg.write_page("nda", title="NDA", body="Counsel reviews every NDA.\n")
    lg.write_page("shared", title="Legal shared", body="The legal version.\n")
    WikiStore.open(root / "wikis" / "hr").write_page(
        "leave", title="Leave", body="Thirty days.\n"
    )
    return root


class TestQualifierGrammar:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("legal/nda", ("legal", "nda")),
            ("nda", (None, "nda")),
            ("abx:penicillin", (None, "abx:penicillin")),
            ("legal/abx:penicillin", ("legal", "abx:penicillin")),
            # Only the first `/` qualifies; a slug cannot contain one, so
            # this is malformed rather than ambiguous, and the remainder
            # comes back intact for the error message.
            ("a/b/c", ("a", "b/c")),
        ],
    )
    def test_split(self, name: str, expected: tuple[str | None, str]) -> None:
        assert split_qualified(name) == expected


class TestSetShape:
    def test_an_empty_set_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one wiki"):
            WikiSet([])

    def test_duplicate_names_are_refused(self, repo: Path) -> None:
        store = WikiStore.open(repo / "wikis" / "open")
        with pytest.raises(ValueError, match="duplicate wiki name"):
            WikiSet([store, store])

    def test_a_standalone_wiki_is_named_for_its_directory(
        self, tmp_path: Path
    ) -> None:
        # It has no registry name; `None/slug` would be worse than useless.
        store = WikiStore.init(tmp_path / "solo")
        store.write_page("x", title="X", body="Body.\n")
        assert WikiSet([store]).list_slugs() == ["solo/x"]

    def test_order_is_the_order_given(self, repo: Path) -> None:
        r = Repo.open(repo)
        assert r.wikiset(audience={"everyone", "hr", "legal"}).names == (
            "open",
            "hr",
            "legal",
        )


class TestMembership:
    def test_a_wiki_outside_the_audience_is_absent(self, repo: Path) -> None:
        # Not filtered out — never opened. That is the whole design.
        wikis = Repo.open(repo).wikiset(audience={"everyone"})
        assert wikis.names == ("open",)
        assert wikis.list_slugs() == ["open/pricing", "open/shared"]
        assert not wikis.exists("nda")
        assert wikis.search("NDA") == []

    def test_an_audience_reaching_nothing_is_an_error(self, repo: Path) -> None:
        # Better than an empty set that silently answers "I don't know"
        # to everything.
        with pytest.raises(OutmemError, match="reach no wiki"):
            Repo.open(repo).wikiset(audience={"nobody"})


class TestResolution:
    def test_a_bare_slug_takes_the_first_wiki_holding_it(self, repo: Path) -> None:
        wikis = Repo.open(repo).wikiset(audience={"everyone", "legal"})
        found = wikis.resolve("shared")
        assert found.wiki == "open"
        assert found.shadowed == ("legal/shared",)

    def test_shadowing_is_reported_not_hidden(self, repo: Path) -> None:
        wikis = Repo.open(repo).wikiset(audience={"everyone", "legal"})
        rendered = wikis.read("shared")
        assert rendered.qualified_slug == "open/shared"
        assert wikis.resolve("shared").shadowed == ("legal/shared",)

    def test_a_qualified_name_goes_to_that_wiki_only(self, repo: Path) -> None:
        wikis = Repo.open(repo).wikiset(audience={"everyone", "legal"})
        assert wikis.read("legal/shared").page.title == "Legal shared"
        assert wikis.read("open/shared").page.title == "Open shared"

    def test_a_qualified_name_does_not_fall_through(self, repo: Path) -> None:
        # `open/nda` must not quietly serve `legal/nda`. Guessing across a
        # boundary the caller was explicit about is how a reader ends up
        # with the wrong wiki's page under the right wiki's name.
        wikis = Repo.open(repo).wikiset(audience={"everyone", "legal"})
        with pytest.raises(OutmemError, match="No such wiki page"):
            wikis.read("open/nda")

    def test_an_unknown_wiki_qualifier_is_an_error(self, repo: Path) -> None:
        wikis = Repo.open(repo).wikiset(audience={"everyone"})
        with pytest.raises(OutmemError, match="no such wiki"):
            wikis.read("legal/nda")

    def test_backlinks_come_back_qualified(self, repo: Path) -> None:
        store = WikiStore.open(repo / "wikis" / "open")
        store.write_page("cites", title="Cites", body="See [[pricing]].\n")
        wikis = Repo.open(repo).wikiset(audience={"everyone"})
        assert wikis.backlinks("pricing") == ("open/cites",)


class TestFannedSearch:
    def test_hits_carry_their_wiki(self, repo: Path) -> None:
        wikis = Repo.open(repo).wikiset(audience={"everyone", "legal"})
        hits = wikis.search("version")
        assert {h.wiki for h in hits} == {"open", "legal"}

    def test_results_follow_wiki_order(self, repo: Path) -> None:
        wikis = Repo.open(repo).wikiset(audience={"everyone", "legal"})
        seen = [h.wiki for h in wikis.search("version")]
        assert seen == sorted(seen, key=lambda w: wikis.names.index(w))


class TestSemanticMerge:
    @pytest.fixture
    def indexed(self, repo: Path, monkeypatch: pytest.MonkeyPatch) -> Repo:
        monkeypatch.setattr(
            "outmem.semantic.build_embedder",
            lambda _model: make_bag_of_words_handle(),
        )
        monkeypatch.setattr(
            "outmem.store.build_embedder",
            lambda _model: make_bag_of_words_handle(),
            raising=False,
        )
        for name in ("open", "hr", "legal"):
            cfg = repo / "wikis" / name / "config.yaml"
            cfg.write_text(
                cfg.read_text(encoding="utf-8").replace(
                    "similarity_threshold: 0.8", "similarity_threshold: 0.01"
                ),
                encoding="utf-8",
            )
            store = WikiStore.open(repo / "wikis" / name)
            store.semantic_reindex_all()
            store.close()
        return Repo.open(repo)

    def test_matches_carry_their_wiki_and_are_ranked_together(
        self, indexed: Repo
    ) -> None:
        wikis = indexed.wikiset(audience={"everyone", "hr", "legal"})
        matches = wikis.semantic_find_similar("counsel reviews every NDA", top_k=5)
        assert matches
        assert matches[0].wiki == "legal"
        scores = [m.match.similarity for m in matches]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_applies_to_the_merged_list(self, indexed: Repo) -> None:
        # Taking k per wiki first would let a wiki with nothing relevant
        # crowd out one that had everything.
        wikis = indexed.wikiset(audience={"everyone", "hr", "legal"})
        assert len(wikis.semantic_find_similar("thirty days of leave", top_k=2)) == 2

    def test_only_the_wikis_in_the_set_contribute(self, indexed: Repo) -> None:
        wikis = indexed.wikiset(audience={"everyone"})
        matches = wikis.semantic_find_similar("counsel reviews every NDA", top_k=10)
        assert {m.wiki for m in matches} <= {"open"}


class TestToolPalette:
    def _tools(self, repo: Path, audience: set[str]) -> dict[str, object]:
        from outmem.adapters.wikiset import wikiset_read_tools

        wikis = Repo.open(repo).wikiset(audience=audience)
        return {t.__name__: t for t in wikiset_read_tools(wikis)}

    def test_every_name_the_model_sees_is_qualified(self, repo: Path) -> None:
        tools = self._tools(repo, {"everyone", "legal"})
        listing = tools["list_pages"]()  # type: ignore[operator]
        assert set(listing.splitlines()) == {
            "open/pricing",
            "open/shared",
            "legal/nda",
            "legal/shared",
        }

    def test_the_docstrings_name_the_wikis_in_play(self, repo: Path) -> None:
        # The model has to know the qualifier exists, or it passes a bare
        # slug into a set where two wikis hold it and never learns why it
        # got the other one.
        tools = self._tools(repo, {"everyone", "legal"})
        doc = tools["read_page"].__doc__  # type: ignore[union-attr]
        assert "open, legal" in doc
        assert "qualified `wiki/slug`" in doc

    def test_grep_reports_the_wiki_per_hit(self, repo: Path) -> None:
        tools = self._tools(repo, {"everyone", "legal"})
        out = tools["grep_wiki"]("NDA")  # type: ignore[operator]
        assert "legal/" in out
        assert "open/" not in out

    def test_a_missing_page_is_a_message_not_an_exception(self, repo: Path) -> None:
        # Tool bodies return text to the model; raising would end the turn.
        tools = self._tools(repo, {"everyone"})
        assert "no such page" in tools["read_page"]("nope")  # type: ignore[operator]

    def test_reading_a_shadowed_page_names_the_alternative(
        self, repo: Path
    ) -> None:
        tools = self._tools(repo, {"everyone", "legal"})
        out = tools["read_page"]("shared")  # type: ignore[operator]
        assert out.startswith("# open/shared")
        assert "legal/shared" in out
