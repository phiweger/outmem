"""Shared pytest fixtures for the outmem test suite."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


def _run_git(args: list[str], *, cwd: Path) -> str:
    """Test-only git runner. Mirrors outmem.git_ops but lives here so the
    fixture is independent of the code under test. Pins
    ``commit.gpgsign=false`` so signing-by-default sandboxes don't
    poison the suite."""
    result = subprocess.run(
        ["git", "-c", "commit.gpgsign=false", *args],
        cwd=str(cwd),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _commit(
    repo: Path,
    *,
    file: str,
    content: str,
    message: str,
    author_name: str = "Test User",
    author_email: str = "test@example.com",
) -> None:
    path = repo / file
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    _run_git(["add", "--", file], cwd=repo)
    _run_git(
        [
            "-c",
            f"user.name={author_name}",
            "-c",
            f"user.email={author_email}",
            "commit",
            "-m",
            message,
        ],
        cwd=repo,
    )


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """A fresh git repo at ``tmp_path/repo`` initialised on ``main``."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _run_git(["init", "--initial-branch", "main"], cwd=repo)
    return repo


@pytest.fixture
def populated_repo(git_repo: Path) -> Path:
    """Git repo with a small commit history across two authors.

    Layout::

        wiki/pricing-formula.md      (1 commit by alice, 1 by agent)
        wiki/acme-msa.md             (1 commit by bob)
        log/2026-05-04.md            (1 commit by agent)
    """
    _commit(
        git_repo,
        file="wiki/pricing-formula.md",
        content="---\ntitle: Pricing formula\nslug: pricing-formula\n---\n\nv1.\n",
        message="compact: pricing-formula",
        author_name="Alice",
        author_email="alice@example.com",
    )
    v2 = "---\ntitle: Pricing formula\nslug: pricing-formula\n---\n\nv2 with clarification.\n"
    _commit(
        git_repo,
        file="wiki/pricing-formula.md",
        content=v2,
        message="extend: pricing-formula",
        author_name="outmem agent",
        author_email="agent@host",
    )
    _commit(
        git_repo,
        file="wiki/acme-msa.md",
        content="---\ntitle: Acme MSA\nslug: acme-msa\n---\n\nContract notes.\n",
        message="compact: acme-msa",
        author_name="Bob",
        author_email="bob@example.com",
    )
    _commit(
        git_repo,
        file="log/2026-05-04.md",
        content="# 2026-05-04\n\n- noticed pricing inconsistency\n",
        message="log: pricing-inconsistency",
        author_name="outmem agent",
        author_email="agent@host",
    )
    return git_repo


@pytest.fixture
def bare_remote(tmp_path: Path) -> Path:
    """A bare git repo to act as ``origin`` for push/pull tests."""
    remote = tmp_path / "remote.git"
    remote.mkdir()
    _run_git(["init", "--bare", "--initial-branch", "main"], cwd=remote)
    return remote


@pytest.fixture
def sample_page_text() -> str:
    """A complete wiki page that exercises every frontmatter field."""
    return (
        "---\n"
        "title: Pricing formula\n"
        "slug: pricing-formula\n"
        "provenance:\n"
        "  - raw/pricing-deck-2026-Q1.md\n"
        "  - raw/acme-msa.md\n"
        "created: 2026-04-12T09:14:00Z\n"
        "updated: 2026-05-04T11:32:00Z\n"
        "tags: [pricing, contracts, finance]\n"
        "---\n"
        "\n"
        "The pricing formula is described below.\n"
        "\n"
        "See also [[acme-msa]] and [[discounts|the discount policy]].\n"
    )


@pytest.fixture
def page_with_rich_provenance() -> str:
    """A page whose provenance entries carry upstream ingestion metadata.

    Used to verify that the agent preserves the dict-valued entries
    verbatim during compaction (spec v0.5 §4).
    """
    return (
        "---\n"
        "title: Acme contract notes\n"
        "slug: acme-contract-notes\n"
        "provenance:\n"
        "  - path: raw/acme-msa.md\n"
        "    drive_path: /shared/contracts/acme/2026/MSA.pdf\n"
        "    sha256: a1b2c3\n"
        "    page_range: 4-7\n"
        "  - raw/acme-pricing.md\n"
        "---\n"
        "\n"
        "Notes body.\n"
    )


@pytest.fixture
def skills_root(tmp_path: Path) -> Path:
    """Build a small nested-layout skills directory under ``tmp_path``.

    Layout::

        skills/
        └── notes/
            ├── search/SKILL.md
            └── write/SKILL.md
    """
    root = tmp_path / "skills"
    notes = root / "notes"
    (notes / "search").mkdir(parents=True)
    (notes / "write").mkdir(parents=True)
    (notes / "search" / "SKILL.md").write_text(
        "---\n"
        "name: search\n"
        "description: Find a fact in the wiki by content or slug.\n"
        "---\n"
        "\n"
        "Run `outmem search <pattern>` first.\n"
    )
    (notes / "write" / "SKILL.md").write_text(
        "---\n"
        "name: write\n"
        "description: Compact a finding into the wiki and commit it back.\n"
        "---\n"
        "\n"
        "Write a new wiki page with `outmem write <slug>`.\n"
    )
    return root
