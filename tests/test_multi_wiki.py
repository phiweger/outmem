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

        found, prefix, _name = find_repo_root(notes)
        assert found == notes
        assert prefix == ""

    def test_an_unlisted_directory_inside_a_multi_wiki_repo_is_not_adopted(
        self, repo: Path
    ) -> None:
        # `wikis.yaml` exists, but says nothing about this directory.
        stray = repo / "wikis" / "scratch"
        stray.mkdir()
        found, prefix, _name = find_repo_root(stray)
        assert found == stray
        assert prefix == ""

    def test_a_listing_reached_by_a_dotdot_spelling_still_matches(
        self, repo: Path
    ) -> None:
        # A listing is about the directory, not the spelling used to
        # reach it.
        detour = repo / "wikis" / "legal" / ".." / "open"
        found, prefix, _name = find_repo_root(detour)
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


class TestPathContainment:
    """One index, one history — a store may only stage its own subtree."""

    def test_a_path_leaving_the_wiki_is_refused(
        self, wiki_pair: tuple[WikiStore, WikiStore]
    ) -> None:
        openw, legal = wiki_pair
        legal.write_page("nda", title="NDA", body="Legal.\n")
        with pytest.raises(OutmemError, match="may not leave it"):
            openw._commit_paths(
                ["../legal/wiki/pages/nda.md"], subject="steal: nda"
            )

    def test_without_the_guard_one_wiki_could_rewrite_anothers_page(
        self, repo: Path, wiki_pair: tuple[WikiStore, WikiStore]
    ) -> None:
        # This is what the guard is for, stated as the attack: git
        # normalises `..` in a pathspec and stages the result happily, so
        # the open wiki can edit legal's page on disk and commit it under
        # its own name. Nothing downstream would notice — the commit is
        # well-formed and even carries `open/` as its qualifier.
        openw, legal = wiki_pair
        legal.write_page("nda", title="NDA", body="Legal.\n")
        (legal.pages_path / "nda.md").write_text(
            "---\ntitle: NDA\n---\n\nTampered.\n", encoding="utf-8"
        )
        with pytest.raises(OutmemError, match="may not leave it"):
            openw._commit_paths(
                ["../legal/wiki/pages/nda.md"], subject="tamper"
            )
        # The tampering is still on disk — the guard stops the commit, not
        # the filesystem — but it never entered the history.
        assert "Tampered." not in _log(repo, "-p", "--format=%s")

    def test_an_absolute_path_is_refused(
        self, wiki_pair: tuple[WikiStore, WikiStore]
    ) -> None:
        openw, _legal = wiki_pair
        with pytest.raises(OutmemError, match="may not leave it"):
            openw._commit_paths(["/etc/passwd"], subject="steal: passwd")

    def test_the_guard_holds_on_a_standalone_wiki_too(self, tmp_path: Path) -> None:
        # The prefix is empty here, so there is no translation to get
        # wrong — and the guard still refuses, because "relative to this
        # wiki" is the contract either way.
        store = WikiStore.init(tmp_path / "solo")
        with pytest.raises(OutmemError, match="may not leave it"):
            store._commit_paths(["../elsewhere.md"], subject="x")


class TestCommitSubjects:
    def test_subjects_name_the_wiki_they_belong_to(
        self, repo: Path, wiki_pair: tuple[WikiStore, WikiStore]
    ) -> None:
        openw, legal = wiki_pair
        openw.write_page("pricing", title="Pricing", body="Open.\n")
        legal.write_page("nda", title="NDA", body="Legal.\n")
        subjects = _log(repo, "--format=%s").splitlines()
        assert "open/ compact: pricing" in subjects
        assert "legal/ compact: nda" in subjects

    def test_a_standalone_wiki_keeps_bare_subjects(self, tmp_path: Path) -> None:
        store = WikiStore.init(tmp_path / "solo")
        store.write_page("pricing", title="Pricing", body="Body.\n")
        assert "compact: pricing" in _log(store.repo, "--format=%s").splitlines()

    def test_the_qualifier_round_trips(self) -> None:
        from outmem.repo import qualify_subject, split_subject

        assert split_subject(qualify_subject("compact: nda", "legal")) == (
            "legal",
            "compact: nda",
        )
        assert split_subject(qualify_subject("compact: nda", None)) == (
            None,
            "compact: nda",
        )

    def test_a_handwritten_subject_is_left_alone(self) -> None:
        from outmem.repo import split_subject

        # `fix: docs` is not a wiki name, so this is not a qualifier.
        assert split_subject("fix: docs/ typo") == (None, "fix: docs/ typo")

    def test_slug_extraction_survives_the_qualifier(self) -> None:
        from outmem.cli.__main__ import _slugs_from_commits

        assert _slugs_from_commits(("legal/ compact: nda", "open/ extend: pricing")) == [
            "nda",
            "pricing",
        ]


class TestPreCommitHookAcrossWikis:
    """One hook per clone, and a commit may span several wikis."""

    def _stage_page(self, repo: Path, wiki: str, slug: str, body: str) -> None:
        rel = f"wikis/{wiki}/wiki/pages/{slug}.md"
        path = repo / rel
        path.write_text(body, encoding="utf-8")
        _run_git(["add", "--", rel], cwd=repo)

    def test_each_wiki_indexes_its_own_staged_pages(
        self, repo: Path, wiki_pair: tuple[WikiStore, WikiStore]
    ) -> None:
        from outmem.cli.__main__ import _cmd_reindex_staged_repo

        openw, legal = wiki_pair
        self._stage_page(
            repo, "open", "handbook", "---\ntitle: Handbook\n---\n\nOpen.\n"
        )
        self._stage_page(repo, "legal", "nda", "---\ntitle: NDA\n---\n\nLegal.\n")

        assert _cmd_reindex_staged_repo(repo) == 0

        # Each wiki's index.md lists its own page and only its own.
        open_index = (openw.wiki_path / "index.md").read_text(encoding="utf-8")
        legal_index = (legal.wiki_path / "index.md").read_text(encoding="utf-8")
        assert "handbook" in open_index and "nda" not in open_index
        assert "nda" in legal_index and "handbook" not in legal_index

    def test_the_repo_root_is_not_scaffolded_into_a_wiki(self, repo: Path) -> None:
        from outmem.cli.__main__ import _cmd_reindex_staged_repo

        # The hook runs at the top of the working tree, which in a
        # multi-wiki repo is not a wiki. Opening a store there would
        # create `wiki/`, `log/` and a `config.yaml` beside `wikis.yaml`.
        _cmd_reindex_staged_repo(repo)
        assert not (repo / "wiki").exists()
        assert not (repo / "config.yaml").exists()
        assert not (repo / "log").exists()

    def test_a_standalone_wiki_still_reindexes(self, tmp_path: Path) -> None:
        from outmem.cli.__main__ import _cmd_reindex_staged_repo

        store = WikiStore.init(tmp_path / "solo")
        page = store.pages_path / "handbook.md"
        page.write_text("---\ntitle: Handbook\n---\n\nBody.\n", encoding="utf-8")
        _run_git(["add", "--", "wiki/pages/handbook.md"], cwd=store.root)

        assert _cmd_reindex_staged_repo(store.root) == 0
        assert "handbook" in (store.wiki_path / "index.md").read_text(encoding="utf-8")


class TestRegistryLint:
    """Both failures here are silent by construction.

    A wiki nobody can reach and a tag nobody can be granted produce no
    error anywhere — the content is simply gone, and the first sign is
    somebody asking why the assistant has never heard of the handbook.
    """

    def _kinds(self, root: Path) -> set[str]:
        from outmem.repo import lint_registry

        return {kind for _sev, kind, _msg in lint_registry(root)}

    def test_a_clean_repository_reports_nothing(
        self, repo: Path, wiki_pair: tuple[WikiStore, WikiStore]
    ) -> None:
        assert self._kinds(repo) == set()

    def test_a_listed_wiki_with_no_directory(self, repo: Path) -> None:
        # `wikis/open` and `wikis/legal` exist but were never scaffolded
        # in this fixture path, so name a wiki that truly is not there.
        (repo / "wikis.yaml").write_text(
            REGISTRY + "  ghost: {path: wikis/ghost, audience: [legal]}\n",
            encoding="utf-8",
        )
        assert "registry-missing-wiki" in self._kinds(repo)

    def test_an_audience_tag_no_tags_block_declares(self, repo: Path) -> None:
        (repo / "wikis.yaml").write_text(
            "tags: {everyone: {}}\n"
            "wikis:\n"
            "  open: {path: wikis/open, audience: [everyone]}\n"
            "  hr:   {path: wikis/hr, audience: [people-team]}\n",
            encoding="utf-8",
        )
        (repo / "wikis" / "hr").mkdir(parents=True)
        WikiStore.init(repo / "wikis" / "hr")
        assert "registry-undeclared-tag" in self._kinds(repo)

    def test_a_declared_tag_no_wiki_uses(self, repo: Path) -> None:
        (repo / "wikis.yaml").write_text(
            REGISTRY.replace("tags:", "tags:\n  ghost: {description: unused}"),
            encoding="utf-8",
        )
        assert "registry-unused-tag" in self._kinds(repo)

    def test_a_wiki_with_no_audience(self, repo: Path) -> None:
        (repo / "wikis.yaml").write_text(
            REGISTRY + "  orphan: {path: wikis/orphan, audience: []}\n",
            encoding="utf-8",
        )
        (repo / "wikis" / "orphan").mkdir(parents=True)
        WikiStore.init(repo / "wikis" / "orphan")
        assert "registry-unreachable-wiki" in self._kinds(repo)

    def test_a_wiki_shaped_directory_nobody_listed(
        self, repo: Path, wiki_pair: tuple[WikiStore, WikiStore]
    ) -> None:
        # Unreachable, and worse: a commit made in it would start its own
        # repository rather than joining this one.
        (repo / "wikis" / "stray").mkdir(parents=True)
        WikiStore.init(repo / "wikis" / "stray")
        assert "registry-unlisted-wiki" in self._kinds(repo)

    def test_a_directory_that_is_not_a_wiki_is_ignored(
        self, repo: Path, wiki_pair: tuple[WikiStore, WikiStore]
    ) -> None:
        (repo / "wikis" / "notes").mkdir(parents=True)
        (repo / "wikis" / "notes" / "README.md").write_text("hi", encoding="utf-8")
        assert "registry-unlisted-wiki" not in self._kinds(repo)


class TestCrossWikiLinkLint:
    def test_a_link_into_another_wiki_is_its_own_finding(
        self, wiki_pair: tuple[WikiStore, WikiStore]
    ) -> None:
        # "That page does not exist" is true but unhelpful: the page
        # usually does exist, in a wiki this one cannot link into.
        from outmem.lint import lint_wiki

        openw, legal = wiki_pair
        legal.write_page("nda", title="NDA", body="Legal.\n")
        openw.write_page("a", title="A", body="See [[legal/nda]].\n")
        report = lint_wiki(
            openw.wiki_path,
            log_dir=openw.log_path,
            sources_dir=openw.sources_path,
            sources_local_dir=openw.sources_local_path,
            repo_root=openw.repo,
        )
        kinds = {f.kind for f in report.findings}
        assert "cross-wiki-wikilink" in kinds
        assert "broken-wikilink" not in kinds

    def test_an_ordinary_typo_is_still_a_broken_link(
        self, wiki_pair: tuple[WikiStore, WikiStore]
    ) -> None:
        from outmem.lint import lint_wiki

        openw, _legal = wiki_pair
        openw.write_page("a", title="A", body="See [[pricng]].\n")
        report = lint_wiki(
            openw.wiki_path,
            log_dir=openw.log_path,
            sources_dir=openw.sources_path,
            sources_local_dir=openw.sources_local_path,
            repo_root=openw.repo,
        )
        kinds = {f.kind for f in report.findings}
        assert "broken-wikilink" in kinds
        assert "cross-wiki-wikilink" not in kinds


class TestRepoImport:
    def test_a_wiki_from_elsewhere_is_copied_and_registered(
        self, repo: Path, tmp_path: Path, wiki_pair: tuple[WikiStore, WikiStore]
    ) -> None:
        from outmem.cli.__main__ import main
        from outmem.repo import Repo

        outside = WikiStore.init(tmp_path / "elsewhere")
        outside.write_page("legacy", title="Legacy", body="Old content.\n")

        assert main(
            ["repo", "import", str(outside.root), "--root", str(repo),
             "--name", "legacy", "--audience", "legal"]
        ) == 0
        imported = Repo.open(repo).wiki_as_operator("legacy")
        assert imported.read("legacy").body.strip() == "Old content."
        assert imported.repo == repo
        # It joined this repository rather than bringing its own.
        assert not (imported.root / ".git").exists()

    def test_the_registry_move_is_committed(
        self, repo: Path, tmp_path: Path, wiki_pair: tuple[WikiStore, WikiStore]
    ) -> None:
        from outmem.cli.__main__ import main

        outside = WikiStore.init(tmp_path / "elsewhere")
        outside.write_page("legacy", title="Legacy", body="Old.\n")
        main(["repo", "import", str(outside.root), "--root", str(repo),
              "--name", "legacy", "--audience", "legal"])
        assert "repo: import legacy" in _log(repo, "--format=%s")

    def test_a_directory_that_is_not_a_wiki_is_refused(
        self, repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from outmem.cli.__main__ import main

        (tmp_path / "junk").mkdir()
        assert main(
            ["repo", "import", str(tmp_path / "junk"), "--root", str(repo),
             "--name", "junk"]
        ) == 1
        assert "does not look like a wiki" in capsys.readouterr().err

    def test_an_existing_name_is_refused(
        self, repo: Path, tmp_path: Path,
        wiki_pair: tuple[WikiStore, WikiStore],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from outmem.cli.__main__ import main

        outside = WikiStore.init(tmp_path / "elsewhere")
        assert main(
            ["repo", "import", str(outside.root), "--root", str(repo),
             "--name", "legal"]
        ) == 1
        assert "already registered" in capsys.readouterr().err
