"""Every commit a store makes can carry git trailers.

A commit made on behalf of a person, through a server, has to say who
acted, with which credential, and which batch it belongs to. The author
line carries the person; the trailers carry what it cannot. Until now no
argument on any write method, on the store, or on ``AgentIdentity``
reached the commit message, and ``--no-verify`` kept a ``commit-msg``
hook from adding them — so a server amended the commit outmem had just
made, changing the sha it had just been handed.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from outmem.exceptions import GitOperationError
from outmem.git_ops import commit_as, format_message
from outmem.store import WikiStore

from .conftest import _run_git

TRAILERS = {
    "Fleming-Actor": "kira",
    "Fleming-Token": "kira-desktop",
    "Fleming-Contribution": "26943db-e2a0268",
}


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=str(root), capture_output=True, text=True, check=True
    ).stdout


def _parsed_trailers(root: Path, rev: str = "HEAD") -> dict[str, str]:
    """What ``git interpret-trailers --parse`` reads back from ``rev``."""
    message = _git(root, "log", "-1", "--format=%B", rev)
    out = subprocess.run(
        ["git", "interpret-trailers", "--parse"],
        cwd=str(root), input=message, capture_output=True, text=True, check=True,
    ).stdout
    pairs = (line.split(": ", 1) for line in out.splitlines() if line.strip())
    return {k: v for k, v in pairs}


def _subject(root: Path, rev: str = "HEAD") -> str:
    return _git(root, "log", "-1", "--format=%s", rev).strip()


@pytest.fixture
def wiki(tmp_path: Path) -> WikiStore:
    store = WikiStore.init(tmp_path / "w")
    store.commit_trailers = TRAILERS
    return store


class TestEveryCommitPathCarriesThem:
    def test_write_page(self, wiki: WikiStore) -> None:
        wiki.write_page("abx:x", title="X", body="Body.\n")
        assert _parsed_trailers(wiki.repo) == TRAILERS
        assert _subject(wiki.repo) == "compact: abx:x"

    def test_append_log(self, wiki: WikiStore) -> None:
        wiki.append_log(topic="pricing", content="- noted\n")
        assert _parsed_trailers(wiki.repo) == TRAILERS
        assert _subject(wiki.repo) == "log: pricing"

    def test_add_source_and_record_ingestion(self, wiki: WikiStore, tmp_path: Path) -> None:
        doc = tmp_path / "doc.md"
        doc.write_text("# Doc\n\nbody\n", encoding="utf-8")
        entry = wiki.add_source(doc)
        assert _parsed_trailers(wiki.repo) == TRAILERS
        wiki.record_ingestion(entry.rel_path, prompt=None, pages_touched=[])
        assert _parsed_trailers(wiki.repo) == TRAILERS
        assert _subject(wiki.repo).startswith("ingest:")

    def test_rename_page(self, wiki: WikiStore) -> None:
        wiki.write_page("old", title="Old", body="Body.\n")
        wiki.rename_page("old", "new")
        assert _parsed_trailers(wiki.repo) == TRAILERS
        assert "old" in _subject(wiki.repo) and "new" in _subject(wiki.repo)

    def test_the_multi_wiki_prefix_stays_on_the_subject_line(self, tmp_path: Path) -> None:
        repo = tmp_path / "mem"
        repo.mkdir()
        _run_git(["init", "--initial-branch", "main"], cwd=repo)
        (repo / "wikis.yaml").write_text(
            "wikis:\n  legal: {path: wikis/legal}\n", encoding="utf-8"
        )
        (repo / "wikis" / "legal").mkdir(parents=True)
        WikiStore.init(repo / "wikis" / "legal").close()
        store = WikiStore.open(repo / "wikis" / "legal")
        store.commit_trailers = TRAILERS
        store.write_page("nda", title="NDA", body="Terms.\n")
        assert _subject(repo) == "legal/ compact: nda"
        assert _parsed_trailers(repo) == TRAILERS


class TestTheDefaultIsByteIdentical:
    def test_no_trailers_no_change(self, tmp_path: Path) -> None:
        store = WikiStore.init(tmp_path / "w")
        assert dict(store.commit_trailers) == {}
        store.write_page("abx:x", title="X", body="Body.\n")
        # Git stores exactly the subject and a newline; no trailing blank
        # line, nothing for interpret-trailers to find.
        assert _git(store.repo, "log", "-1", "--format=%B") == "compact: abx:x\n\n"
        assert _parsed_trailers(store.repo) == {}

    def test_format_message_leaves_an_empty_mapping_alone(self) -> None:
        assert format_message("compact: x", {}) == "compact: x"
        assert format_message("compact: x", None) == "compact: x"
        assert format_message("compact: x", {"K": "v"}) == "compact: x\n\nK: v\n"


class TestMalformedTrailersAreRefusedAtTheCall:
    @pytest.mark.parametrize(
        "trailers",
        [
            {"Bad Key": "v"},
            {"1st": "v"},
            {"Key_Under": "v"},
            {"": "v"},
            {"Key": "two\nlines"},
            {"Key": "cr\rhere"},
        ],
    )
    def test_assignment_refuses(self, tmp_path: Path, trailers: dict[str, str]) -> None:
        store = WikiStore.init(tmp_path / "w")
        with pytest.raises(GitOperationError):
            store.commit_trailers = trailers
        # Refused at assignment, so nothing downstream ever sees it and
        # the store keeps what it had.
        assert dict(store.commit_trailers) == {}

    def test_the_view_cannot_be_mutated_around_the_check(self, wiki: WikiStore) -> None:
        with pytest.raises(TypeError):
            wiki.commit_trailers["Bad Key"] = "v"  # type: ignore[index]

    def test_commit_as_refuses_too(self, tmp_path: Path) -> None:
        root = tmp_path / "r"
        root.mkdir()
        _run_git(["init", "--initial-branch", "main"], cwd=root)
        with pytest.raises(GitOperationError):
            commit_as(
                root, message="x", author_name="T", author_email="t@t.invalid",
                allow_empty=True, trailers={"Bad Key": "v"},
            )
        assert _git(root, "rev-list", "--all", "--count").strip() == "0"


class TestConfigSeedsAndTheStoreWins:
    def test_config_seeds_the_attribute(self, tmp_path: Path) -> None:
        root = tmp_path / "w"
        WikiStore.init(root).close()
        cfg = root / "config.yaml"
        text = cfg.read_text(encoding="utf-8")
        assert "  commit_trailers: {}" in text  # the template carries it
        cfg.write_text(
            text.replace("  commit_trailers: {}", "  commit_trailers:\n    Generated-By: outmem"),
            encoding="utf-8",
        )
        store = WikiStore.open(root)
        assert dict(store.commit_trailers) == {"Generated-By": "outmem"}
        store.write_page("abx:x", title="X", body="Body.\n")
        assert _parsed_trailers(root) == {"Generated-By": "outmem"}

    def test_a_value_set_on_the_store_wins(self, tmp_path: Path) -> None:
        root = tmp_path / "w"
        WikiStore.init(root).close()
        cfg = root / "config.yaml"
        cfg.write_text(
            cfg.read_text(encoding="utf-8").replace(
                "  commit_trailers: {}", "  commit_trailers:\n    Generated-By: outmem"
            ),
            encoding="utf-8",
        )
        store = WikiStore.open(root)
        store.commit_trailers = TRAILERS
        store.write_page("abx:x", title="X", body="Body.\n")
        assert _parsed_trailers(root) == TRAILERS

    def test_a_malformed_config_block_is_ignored_like_any_other(self, tmp_path: Path) -> None:
        # The config parser's house rule: a wrongly-typed value is ignored,
        # not fatal. A non-string value here therefore leaves the default.
        root = tmp_path / "w"
        WikiStore.init(root).close()
        cfg = root / "config.yaml"
        cfg.write_text(
            cfg.read_text(encoding="utf-8").replace(
                "  commit_trailers: {}", "  commit_trailers:\n    Count: 3"
            ),
            encoding="utf-8",
        )
        assert dict(WikiStore.open(root).commit_trailers) == {}
