"""Opening a wiki out of a multi-wiki repository.

The whole access decision is here: which directory does this request
open. There is no filtering downstream, so this is the only place that
can get it wrong.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from outmem.exceptions import OutmemError
from outmem.repo import Repo
from outmem.store import WikiStore

from .conftest import _run_git

REGISTRY = """\
version: 1
tags:
  everyone: {description: "All employees"}
  hr:       {description: "People team"}
  legal:    {description: "Legal counsel"}
  exec:     {description: "Executive team"}
wikis:
  open:  {path: wikis/open,  title: "Handbook", audience: [everyone]}
  hr:    {path: wikis/hr,    title: "People",   audience: [hr]}
  legal: {path: wikis/legal, title: "Legal",    audience: [legal, exec]}
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
    return root


class TestAudienceFiltering:
    def test_one_shared_tag_is_enough(self, repo: Path) -> None:
        r = Repo.open(repo)
        assert r.wikis_for({"everyone"}) == ["open"]
        assert r.wikis_for({"exec"}) == ["legal"]
        assert r.wikis_for({"everyone", "hr"}) == ["open", "hr"]

    def test_holding_no_tags_reaches_nothing(self, repo: Path) -> None:
        assert Repo.open(repo).wikis_for(set()) == []

    def test_order_follows_the_registry(self, repo: Path) -> None:
        # Declared order is the resolution order a session will read in,
        # so it has to be the registry's, not a set's.
        r = Repo.open(repo)
        assert r.wikis_for({"everyone", "hr", "legal"}) == ["open", "hr", "legal"]


class TestOpening:
    def test_a_permitted_wiki_opens(self, repo: Path) -> None:
        store = Repo.open(repo).wiki("open", audience={"everyone"})
        assert store.root == repo / "wikis" / "open"
        assert store.wiki_name == "open"
        assert store.repo == repo

    def test_an_unpermitted_wiki_is_refused(self, repo: Path) -> None:
        with pytest.raises(OutmemError, match="no such wiki"):
            Repo.open(repo).wiki("legal", audience={"everyone"})

    def test_refusal_is_indistinguishable_from_absence(self, repo: Path) -> None:
        # A caller that can tell "not for you" from "does not exist" can
        # enumerate the wikis it may not open, and a wiki's *name* can be
        # the sensitive part.
        r = Repo.open(repo)
        with pytest.raises(OutmemError) as forbidden:
            r.wiki("legal", audience={"everyone"})
        with pytest.raises(OutmemError) as missing:
            r.wiki("no-such-thing", audience={"everyone"})
        assert str(forbidden.value).replace("legal", "X") == str(
            missing.value
        ).replace("no-such-thing", "X")

    def test_the_operator_path_needs_no_audience(self, repo: Path) -> None:
        store = Repo.open(repo).wiki_as_operator("legal")
        assert store.wiki_name == "legal"

    def test_there_is_no_argumentless_way_to_open_everything(self) -> None:
        # The guard rail is the shape of the API: `wiki` requires an
        # audience, and the unrestricted path is separately named, so
        # getting the whole repository is something a caller says out
        # loud rather than reaches by passing the obvious argument.
        import inspect

        sig = inspect.signature(Repo.wiki)
        assert sig.parameters["audience"].kind is inspect.Parameter.KEYWORD_ONLY
        assert sig.parameters["audience"].default is inspect.Parameter.empty

    def test_a_listed_wiki_with_no_directory_says_so(self, repo: Path) -> None:
        import shutil

        shutil.rmtree(repo / "wikis" / "hr")
        with pytest.raises(OutmemError, match="does not exist"):
            Repo.open(repo).wiki_as_operator("hr")

    def test_a_directory_absent_from_the_registry_is_unreachable(
        self, repo: Path
    ) -> None:
        (repo / "wikis" / "scratch").mkdir()
        WikiStore.init(repo / "wikis" / "scratch")
        r = Repo.open(repo)
        assert "scratch" not in r.registry.wikis
        with pytest.raises(OutmemError, match="no such wiki"):
            r.wiki_as_operator("scratch")


class TestRepoOpen:
    def test_a_plain_directory_is_not_a_repository(self, tmp_path: Path) -> None:
        with pytest.raises(OutmemError, match="not a multi-wiki repository"):
            Repo.open(tmp_path)


class TestCliWikiFlag:
    def test_wiki_resolves_against_the_repository(
        self, repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import io

        from outmem.cli.__main__ import main

        monkeypatch.setattr("sys.stdin", io.StringIO("Counsel reviews it.\n"))
        assert (
            main(
                ["write", "nda", "--root", str(repo), "--wiki", "legal",
                 "--title", "NDA"]
            )
            == 0
        )
        assert (repo / "wikis" / "legal" / "wiki" / "pages" / "nda.md").is_file()
        assert not (repo / "wikis" / "open" / "wiki" / "pages" / "nda.md").exists()

    def test_an_unknown_wiki_name_lists_what_exists(
        self, repo: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from outmem.cli.__main__ import main

        assert main(["read", "x", "--root", str(repo), "--wiki", "nope"]) == 1
        err = capsys.readouterr().err
        assert "no such wiki: 'nope'" in err
        # Naming what *does* exist is fine here: this is the operator at a
        # terminal with the repository in front of them, not a served
        # session.
        assert "hr, legal, open" in err

    def test_wiki_without_a_registry_explains_itself(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from outmem.cli.__main__ import main

        WikiStore.init(tmp_path / "solo")
        assert (
            main(["read", "x", "--root", str(tmp_path / "solo"), "--wiki", "open"])
            == 1
        )
        assert "needs a multi-wiki repository" in capsys.readouterr().err


class TestOpeningARegisteredNonWiki:
    def test_a_directory_that_was_never_scaffolded_is_refused(self, repo: Path) -> None:
        # `WikiStore.open` scaffolds `wiki/pages/` and `log/` into whatever
        # it is pointed at; `Repo` must not let a listed-but-empty
        # directory quietly become a wiki that opens and holds nothing.
        (repo / "wikis" / "hollow").mkdir()
        (repo / "wikis.yaml").write_text(
            REGISTRY + "  hollow: {path: wikis/hollow, audience: [hr]}\n", encoding="utf-8"
        )
        with pytest.raises(OutmemError, match="not a wiki"):
            Repo.open(repo).wiki_as_operator("hollow")
        assert not (repo / "wikis" / "hollow" / "wiki").exists()


class TestRepoSubcommandsTakeNoWiki:
    def test_wiki_is_not_an_option_for_repo_commands(self, repo: Path) -> None:
        # The subject of every `repo` command is the registry; a wiki name
        # has nothing to refer to there and is refused by the parser
        # rather than silently ignored.
        from outmem.cli.__main__ import main

        with pytest.raises(SystemExit) as exc:
            main(["repo", "list", "--root", str(repo), "--wiki", "open"])
        assert exc.value.code == 2
