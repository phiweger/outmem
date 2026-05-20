"""Tests for ``outmem.history``."""

from __future__ import annotations

from pathlib import Path

import pytest

from outmem.exceptions import SlugError
from outmem.history import page_history, topic_evolution


def test_page_history_returns_commits(populated_repo: Path) -> None:
    history = page_history(populated_repo, "pricing-formula")
    assert len(history) == 2
    subjects = [c.subject for c in history]
    assert subjects == ["extend: pricing-formula", "compact: pricing-formula"]


def test_page_history_for_unknown_slug_is_empty(populated_repo: Path) -> None:
    history = page_history(populated_repo, "no-such-page")
    assert history == []


def test_page_history_rejects_bad_slug(populated_repo: Path) -> None:
    with pytest.raises(SlugError):
        page_history(populated_repo, "Bad Slug")


def test_topic_evolution_includes_diff(populated_repo: Path) -> None:
    out = topic_evolution(populated_repo, ["pricing-formula"])
    assert "diff --git" in out
    assert "compact: pricing-formula" in out
    assert "extend: pricing-formula" in out


def test_topic_evolution_pulls_in_log_dir_by_default(populated_repo: Path) -> None:
    out = topic_evolution(populated_repo, ["pricing-formula"])
    assert "log: pricing-inconsistency" in out


def test_topic_evolution_excludes_log_when_disabled(populated_repo: Path) -> None:
    out = topic_evolution(populated_repo, ["pricing-formula"], include_log=False)
    assert "log: pricing-inconsistency" not in out


def test_topic_evolution_requires_slugs(populated_repo: Path) -> None:
    with pytest.raises(ValueError, match="at least one slug"):
        topic_evolution(populated_repo, [])


def test_topic_evolution_validates_each_slug(populated_repo: Path) -> None:
    with pytest.raises(SlugError):
        topic_evolution(populated_repo, ["pricing-formula", "Bad Slug"])
