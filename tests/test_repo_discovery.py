"""Discovery — what the host's user database provisions against.

The host maps its authenticated users to audience tags, so it has to be
able to *enumerate* the vocabulary rather than only filter with it. That
makes this payload a public interface: a provisioning script in someone
else's codebase reads it, and breaking its shape should fail here first.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from outmem.repo import CATALOGUE_VERSION, Repo
from outmem.store import WikiStore

from .conftest import _run_git

REGISTRY = """\
version: 1
tags:
  everyone: {description: "All employees"}
  hr:       {description: "People team"}
  atlas:    {description: "M&A working group"}
  ghost:    {description: "Nobody uses this"}
wikis:
  open:  {path: wikis/open,  title: "Handbook",      audience: [everyone]}
  hr:    {path: wikis/hr,    title: "People",        audience: [hr]}
  atlas: {path: wikis/atlas, title: "Project Atlas", audience: [atlas]}
  orphan: {path: wikis/orphan, title: "Orphan",      audience: []}
"""


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "mem"
    root.mkdir()
    _run_git(["init", "--initial-branch", "main"], cwd=root)
    (root / "wikis.yaml").write_text(REGISTRY, encoding="utf-8")
    for name in ("open", "hr", "atlas", "orphan"):
        (root / "wikis" / name).mkdir(parents=True)
        WikiStore.init(root / "wikis" / name)
    store = WikiStore.init(root / "wikis" / "open")
    store.write_page("pricing", title="Pricing", body="Body.\n")
    store.write_page("holidays", title="Holidays", body="Body.\n")
    return root


class TestTagVocabulary:
    def test_tags_carry_their_wiki_back_references(self, repo: Path) -> None:
        tags = {t.name: t for t in Repo.open(repo).tags()}
        assert tags["everyone"].wikis == ("open",)
        assert tags["hr"].description == "People team"

    def test_a_declared_tag_no_wiki_uses_is_still_listed(self, repo: Path) -> None:
        # It is part of the vocabulary a host provisions against. That it
        # reaches nothing is a finding for `reconcile`, not a reason to
        # hide it from the admin view.
        tags = {t.name: t for t in Repo.open(repo).tags()}
        assert "ghost" in tags
        assert tags["ghost"].wikis == ()


class TestCatalogue:
    def test_the_admin_view_lists_everything(self, repo: Path) -> None:
        catalogue = Repo.open(repo).catalogue()
        assert [w.name for w in catalogue.wikis] == ["open", "hr", "atlas", "orphan"]
        assert {t.name for t in catalogue.tags} == {
            "everyone",
            "hr",
            "atlas",
            "ghost",
        }

    def test_page_counts_are_reported(self, repo: Path) -> None:
        by_name = {w.name: w for w in Repo.open(repo).catalogue().wikis}
        assert by_name["open"].pages == 2
        assert by_name["hr"].pages == 0


class TestCatalogueForAnAudience:
    def test_it_names_no_wiki_outside_the_audience(self, repo: Path) -> None:
        catalogue = Repo.open(repo).catalogue_for({"everyone"})
        assert [w.name for w in catalogue.wikis] == ["open"]

    def test_it_names_no_tag_outside_the_audience(self, repo: Path) -> None:
        # The leak this surface can have. A tag name discloses as much as
        # a wiki name — the existence of an `atlas` compartment is the
        # secret, whatever the wiki behind it is called.
        catalogue = Repo.open(repo).catalogue_for({"everyone"})
        assert [t.name for t in catalogue.tags] == ["everyone"]

    def test_holding_nothing_shows_nothing(self, repo: Path) -> None:
        catalogue = Repo.open(repo).catalogue_for(set())
        assert catalogue.wikis == ()
        assert catalogue.tags == ()


class TestReconcile:
    def test_it_finds_a_tag_no_wiki_declares(self, repo: Path) -> None:
        # A stale grant or a typo. Its symptom is a user silently getting
        # nothing extra, which is why something has to go looking.
        result = Repo.open(repo).reconcile({"everyone", "hr", "atlas", "typoo"})
        assert result.unknown == ("typoo",)

    def test_a_declared_but_unassigned_tag_is_not_unknown(self, repo: Path) -> None:
        # `ghost` is declared in the registry; nobody holding it is a
        # different finding from nobody knowing what it is.
        assert Repo.open(repo).reconcile({"ghost"}).unknown == ()

    def test_it_finds_a_wiki_nobody_reaches(self, repo: Path) -> None:
        # Content nobody can see — how a compartment quietly dies.
        result = Repo.open(repo).reconcile({"everyone"})
        assert "orphan" in result.unreachable
        assert "hr" in result.unreachable
        assert "open" not in result.unreachable

    def test_a_wiki_with_no_audience_is_unreachable_by_anyone(
        self, repo: Path
    ) -> None:
        result = Repo.open(repo).reconcile({"everyone", "hr", "atlas", "ghost"})
        assert result.unreachable == ("orphan",)


class TestJsonContract:
    """The payload a provisioning script in another codebase reads."""

    def test_the_shape_is_pinned(self, repo: Path) -> None:
        payload = json.loads(json.dumps(Repo.open(repo).catalogue().as_dict()))
        assert set(payload) == {"version", "tags", "wikis"}
        assert payload["version"] == CATALOGUE_VERSION
        assert set(payload["tags"][0]) == {"name", "description", "wikis"}
        assert set(payload["wikis"][0]) == {"name", "title", "audience", "pages"}

    def test_it_survives_a_round_trip_through_json(self, repo: Path) -> None:
        # Frozensets and tuples do not serialise; the `as_dict` layer is
        # what makes this a contract rather than a repr.
        payload = Repo.open(repo).catalogue().as_dict()
        assert json.loads(json.dumps(payload)) == payload

    def test_the_cli_emits_it(
        self, repo: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from outmem.cli.__main__ import main

        assert main(["repo", "tags", "--root", str(repo), "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["version"] == CATALOGUE_VERSION
        assert {t["name"] for t in payload["tags"]} == {
            "everyone",
            "hr",
            "atlas",
            "ghost",
        }

    def test_the_audience_command_reports_only_unknown_tags(
        self, repo: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from outmem.cli.__main__ import main

        # Asked about one user, "wikis you cannot see" is the normal
        # condition rather than a finding, so only the `unknown` half of
        # a reconciliation belongs here.
        main(["repo", "audience", "--root", str(repo), "--tags",
              "everyone,typoo", "--json"])
        payload = json.loads(capsys.readouterr().out)
        assert payload["unknown_tags"] == ["typoo"]
        assert [w["name"] for w in payload["wikis"]] == ["open"]
        assert "unreachable" not in payload


class TestCliListing:
    def test_list_shows_wikis_with_audiences(
        self, repo: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from outmem.cli.__main__ import main

        assert main(["repo", "list", "--root", str(repo)]) == 0
        out = capsys.readouterr().out
        assert "open" in out and "[everyone]" in out
        assert "orphan" in out and "(nobody)" in out

    def test_list_can_be_narrowed_to_an_audience(
        self, repo: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from outmem.cli.__main__ import main

        assert main(
            ["repo", "list", "--root", str(repo), "--audience", "everyone"]
        ) == 0
        out = capsys.readouterr().out
        assert "open" in out
        assert "atlas" not in out
