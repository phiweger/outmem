"""Several wikis in one git repository.

A wiki is the compartment: within it everything is open, and separation
comes from which directory a request opens rather than from filtering
what a shared corpus returns. These tests hold the seam that makes that
true — a store addresses its own subtree and nothing else, while all of
them share one history.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from outmem.exceptions import OutmemError
from outmem.repo import find_repo_root, load_registry
from outmem.store import WikiStore

from .conftest import _commit, _run_git

REGISTRY = """\
version: 1
tags:
  everyone: {description: "All employees"}
  legal:    {description: "Legal counsel"}
wikis:
  open:  {path: wikis/open,  title: "Handbook", audience: [everyone]}
  legal: {path: wikis/legal, title: "Legal",    audience: [legal]}
"""


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A git repo holding two registered, not-yet-scaffolded wikis."""
    root = tmp_path / "mem"
    root.mkdir()
    _run_git(["init", "--initial-branch", "main"], cwd=root)
    (root / "wikis.yaml").write_text(REGISTRY, encoding="utf-8")
    for name in ("open", "legal"):
        (root / "wikis" / name).mkdir(parents=True)
    return root


@pytest.fixture
def wiki_pair(repo: Path) -> tuple[WikiStore, WikiStore]:
    return (
        WikiStore.init(repo / "wikis" / "open"),
        WikiStore.init(repo / "wikis" / "legal"),
    )


def _log(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "log", *args],
        cwd=str(repo),
        check=True,
        capture_output=True,
        text=True,
    ).stdout


class TestRootResolution:
    def test_each_wiki_resolves_the_shared_repo_and_its_own_prefix(
        self, repo: Path, wiki_pair: tuple[WikiStore, WikiStore]
    ) -> None:
        openw, legal = wiki_pair
        assert openw.repo == legal.repo == repo
        assert openw.root == repo / "wikis" / "open"
        assert openw.repo_prefix == "wikis/open/"
        assert legal.repo_prefix == "wikis/legal/"

    def test_a_standalone_wiki_is_its_own_repo(self, tmp_path: Path) -> None:
        store = WikiStore.init(tmp_path / "solo")
        assert store.repo == store.root
        # The empty prefix is what makes every existing wiki a no-op: a
        # wiki-relative path is already repo-relative.
        assert store.repo_prefix == ""

    def test_a_registered_wiki_does_not_start_its_own_repo(
        self, repo: Path, wiki_pair: tuple[WikiStore, WikiStore]
    ) -> None:
        openw, _legal = wiki_pair
        # A nested `.git` would make this wiki's commits invisible to the
        # repository that contains it.
        assert not (openw.root / ".git").exists()
        assert (repo / ".git").is_dir()


class TestOptInDiscovery:
    """Walking up for ``.git`` is only safe when somebody asked for it."""

    def test_a_wiki_inside_an_unrelated_repo_stays_standalone(
        self, tmp_path: Path
    ) -> None:
        # The `~/projects/notes` case: a wiki that happens to sit inside
        # an unrelated repository must not silently start committing into
        # it. No `wikis.yaml`, no adoption.
        project = tmp_path / "projects"
        project.mkdir()
        _run_git(["init", "--initial-branch", "main"], cwd=project)
        notes = project / "notes"
        notes.mkdir()

        found, prefix = find_repo_root(notes)
        assert found == notes
        assert prefix == ""

    def test_an_unlisted_directory_inside_a_multi_wiki_repo_is_not_adopted(
        self, repo: Path
    ) -> None:
        # `wikis.yaml` exists, but says nothing about this directory.
        stray = repo / "wikis" / "scratch"
        stray.mkdir()
        found, prefix = find_repo_root(stray)
        assert found == stray
        assert prefix == ""

    def test_a_listing_reached_by_a_dotdot_spelling_still_matches(
        self, repo: Path
    ) -> None:
        # A listing is about the directory, not the spelling used to
        # reach it.
        detour = repo / "wikis" / "legal" / ".." / "open"
        found, prefix = find_repo_root(detour)
        assert found == repo
        assert prefix == "wikis/open/"


class TestOneHistoryManyWikis:
    def test_a_write_only_ever_stages_its_own_subtree(
        self, repo: Path, wiki_pair: tuple[WikiStore, WikiStore]
    ) -> None:
        openw, legal = wiki_pair
        openw.write_page("pricing", title="Pricing", body="Open body.\n")
        legal.write_page("contract", title="Contract", body="Legal body.\n")

        for prefix, subject in (("wikis/open/", "pricing"), ("wikis/legal/", "contract")):
            touched = _log(repo, "--format=", "--name-only", "-1", f"--grep={subject}")
            paths = [line for line in touched.splitlines() if line]
            assert paths, subject
            assert all(p.startswith(prefix) for p in paths), (subject, paths)

    def test_the_wikis_share_one_history(
        self, repo: Path, wiki_pair: tuple[WikiStore, WikiStore]
    ) -> None:
        openw, legal = wiki_pair
        openw.write_page("pricing", title="Pricing", body="Open.\n")
        legal.write_page("contract", title="Contract", body="Legal.\n")
        subjects = _log(repo, "--format=%s").splitlines()
        assert any("pricing" in s for s in subjects)
        assert any("contract" in s for s in subjects)

    def test_each_wiki_sees_only_its_own_pages(
        self, wiki_pair: tuple[WikiStore, WikiStore]
    ) -> None:
        openw, legal = wiki_pair
        openw.write_page("pricing", title="Pricing", body="Open.\n")
        legal.write_page("contract", title="Contract", body="Legal.\n")
        assert openw.list_slugs() == ["pricing"]
        assert legal.list_slugs() == ["contract"]
        assert not openw.exists("contract")


class TestPathspecsAreRepoRelative:
    """``git log`` pathspecs resolve from the repo root, not the wiki."""

    def test_page_history_finds_the_page(
        self, wiki_pair: tuple[WikiStore, WikiStore]
    ) -> None:
        openw, _legal = wiki_pair
        openw.write_page("pricing", title="Pricing", body="One.\n")
        openw.extend_page("pricing", body="One.\nTwo.\n")
        assert len(openw.history("pricing")) == 2

    def test_steering_does_not_see_the_other_wikis_commits(
        self, repo: Path, wiki_pair: tuple[WikiStore, WikiStore]
    ) -> None:
        openw, legal = wiki_pair
        # Human edits in each wiki. Steering reports commits that are not
        # the agent's, narrowed to the wiki that asked.
        _commit(
            repo,
            file="wikis/open/wiki/pages/handbook.md",
            content="# Handbook\n",
            message="human: handbook",
        )
        _commit(
            repo,
            file="wikis/legal/wiki/pages/nda.md",
            content="# NDA\n",
            message="human: nda",
        )
        assert [c.subject for c in openw.steering(default_window="10 years")] == [
            "human: handbook"
        ]
        assert [c.subject for c in legal.steering(default_window="10 years")] == [
            "human: nda"
        ]

    def test_evolution_is_scoped_to_the_asking_wiki(
        self, repo: Path, wiki_pair: tuple[WikiStore, WikiStore]
    ) -> None:
        openw, legal = wiki_pair
        openw.write_page("pricing", title="Pricing", body="Open body text.\n")
        legal.write_page("contract", title="Contract", body="Legal body text.\n")
        stream = openw.evolution(["pricing"])
        assert "Open body text." in stream
        assert "Legal body text." not in stream


class TestRegistryParsing:
    def test_entries_carry_title_and_audience(self, repo: Path) -> None:
        registry = load_registry(repo)
        assert registry is not None
        assert set(registry.wikis) == {"open", "legal"}
        assert registry.wikis["legal"].title == "Legal"
        assert registry.wikis["legal"].audience == frozenset({"legal"})
        assert set(registry.tags) == {"everyone", "legal"}

    def test_a_directory_with_no_registry_is_not_an_error(
        self, tmp_path: Path
    ) -> None:
        assert load_registry(tmp_path) is None

    @pytest.mark.parametrize(
        "body",
        [
            "wikis: [not, a, mapping]\n",
            "wikis:\n  open: {path: /etc}\n",
            "wikis:\n  open: {path: ../escape}\n",
            "wikis:\n  open: {path: wikis/open, audience: 5}\n",
            "version: [1]\n",
            ": : :\n  bad yaml\n",
        ],
    )
    def test_a_malformed_registry_is_refused(self, tmp_path: Path, body: str) -> None:
        # Treating a broken registry as "no registry" would silently
        # demote a multi-wiki repo to a single wiki, and commit one
        # wiki's writes into another's history.
        (tmp_path / "wikis.yaml").write_text(body, encoding="utf-8")
        with pytest.raises(OutmemError):
            load_registry(tmp_path)
