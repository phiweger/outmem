"""Tests for ``outmem.config`` — config.yaml + .env loaders."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from outmem.config import (
    DEFAULT_AGENT_EMAIL,
    DEFAULT_AGENT_NAME,
    DEFAULT_BRANCH,
    DEFAULT_MODEL,
    DEFAULT_REMOTE,
    OutmemConfig,
    load_dotenv_if_present,
    load_yaml_config,
    starter_yaml,
)

# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------


def test_load_missing_yaml_returns_defaults(tmp_path: Path) -> None:
    config = load_yaml_config(tmp_path)
    assert isinstance(config, OutmemConfig)
    assert config.model == DEFAULT_MODEL
    assert config.agent.name == DEFAULT_AGENT_NAME
    assert config.agent.email == DEFAULT_AGENT_EMAIL
    assert config.remote.name == DEFAULT_REMOTE
    assert config.remote.branch == DEFAULT_BRANCH
    assert config.git.remove_stale_lock is True
    assert config.git.stale_lock_seconds == 60


def test_load_yaml_overrides_defaults(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text(
        "model: openai:gpt-5\n"
        "agent:\n"
        "  name: custom-bot\n"
        "  email: bot@example.com\n"
        "remote:\n"
        "  name: upstream\n"
        "  branch: trunk\n"
        "git:\n"
        "  remove_stale_lock: false\n"
        "  stale_lock_seconds: 120\n"
        "  retry_on_lock: false\n",
        encoding="utf-8",
    )
    cfg = load_yaml_config(tmp_path)
    assert cfg.model == "openai:gpt-5"
    assert cfg.agent.name == "custom-bot"
    assert cfg.agent.email == "bot@example.com"
    assert cfg.remote.name == "upstream"
    assert cfg.remote.branch == "trunk"
    assert cfg.git.remove_stale_lock is False
    assert cfg.git.stale_lock_seconds == 120
    assert cfg.git.retry_on_lock is False


def test_load_yaml_partial_override(tmp_path: Path) -> None:
    """Unspecified sections keep their defaults."""
    (tmp_path / "config.yaml").write_text("model: openai:gpt-4o\n", encoding="utf-8")
    cfg = load_yaml_config(tmp_path)
    assert cfg.model == "openai:gpt-4o"
    assert cfg.agent.name == DEFAULT_AGENT_NAME
    assert cfg.git.remove_stale_lock is True


def test_load_yaml_preserves_unknown_keys(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text(
        "model: openai:gpt-5\nfuture_feature:\n  enabled: true\n",
        encoding="utf-8",
    )
    cfg = load_yaml_config(tmp_path)
    assert cfg.extra == {"future_feature": {"enabled": True}}


def test_logfire_disabled_by_default(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text("model: x\n", encoding="utf-8")
    cfg = load_yaml_config(tmp_path)
    assert cfg.logfire.project is None


def test_relevance_disabled_by_default(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text("model: x\n", encoding="utf-8")
    cfg = load_yaml_config(tmp_path)
    assert cfg.relevance.enabled is False


def test_relevance_block_parsed(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text(
        "relevance:\n"
        "  enabled: true\n"
        "  model: anthropic:claude-haiku-4-5\n"
        "  max_relevant: 5\n"
        "  max_candidates: 30\n"
        "  context: lines\n"
        "  context_chars_per_page: 1500\n",
        encoding="utf-8",
    )
    cfg = load_yaml_config(tmp_path)
    assert cfg.relevance.enabled is True
    assert cfg.relevance.model == "anthropic:claude-haiku-4-5"
    assert cfg.relevance.max_relevant == 5
    assert cfg.relevance.max_candidates == 30
    assert cfg.relevance.context == "lines"
    assert cfg.relevance.context_chars_per_page == 1500


def test_relevance_bad_context_ignored(tmp_path: Path) -> None:
    """An out-of-enum context value is rejected, keeping the default."""
    (tmp_path / "config.yaml").write_text(
        "relevance:\n  enabled: true\n  context: bogus\n", encoding="utf-8"
    )
    cfg = load_yaml_config(tmp_path)
    assert cfg.relevance.context == "page"  # default preserved


def test_logfire_project_from_yaml(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text(
        "logfire:\n  project: my-project\n",
        encoding="utf-8",
    )
    cfg = load_yaml_config(tmp_path)
    assert cfg.logfire.project == "my-project"


def test_logfire_explicit_null_yaml(tmp_path: Path) -> None:
    """``project: null`` (the starter default) is the disabled marker."""
    (tmp_path / "config.yaml").write_text(
        "logfire:\n  project: null\n",
        encoding="utf-8",
    )
    cfg = load_yaml_config(tmp_path)
    assert cfg.logfire.project is None


def test_load_yaml_malformed_returns_defaults(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text("model: [unterminated\n", encoding="utf-8")
    cfg = load_yaml_config(tmp_path)
    assert cfg.model == DEFAULT_MODEL  # silent fallback


def test_load_yaml_non_mapping_returns_defaults(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text("- just\n- a\n- list\n", encoding="utf-8")
    cfg = load_yaml_config(tmp_path)
    assert cfg.model == DEFAULT_MODEL


def test_load_yaml_empty_file_returns_defaults(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text("", encoding="utf-8")
    cfg = load_yaml_config(tmp_path)
    assert cfg.model == DEFAULT_MODEL


# ---------------------------------------------------------------------------
# .env loader
# ---------------------------------------------------------------------------


def test_load_dotenv_explicit_path_missing(tmp_path: Path) -> None:
    assert load_dotenv_if_present(tmp_path / ".env") is False


def test_load_dotenv_explicit_path_present(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OUTMEM_TEST_KEY", raising=False)
    env_path = tmp_path / ".env"
    env_path.write_text("OUTMEM_TEST_KEY=hello\n", encoding="utf-8")
    assert load_dotenv_if_present(env_path) is True
    assert os.environ.get("OUTMEM_TEST_KEY") == "hello"
    monkeypatch.delenv("OUTMEM_TEST_KEY", raising=False)


def test_load_dotenv_walks_upward_from_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No-arg form: standard python-dotenv search from CWD upward.

    Mirrors the production call site (``WikiStore.open``) — the user's
    .env lives at the project root they run ``outmem`` from, not at
    the wiki root.
    """
    monkeypatch.delenv("OUTMEM_TEST_KEY", raising=False)
    project = tmp_path / "project"
    nested = project / "subdir"
    nested.mkdir(parents=True)
    (project / ".env").write_text("OUTMEM_TEST_KEY=from-project\n", encoding="utf-8")
    monkeypatch.chdir(nested)
    assert load_dotenv_if_present() is True
    assert os.environ.get("OUTMEM_TEST_KEY") == "from-project"
    monkeypatch.delenv("OUTMEM_TEST_KEY", raising=False)


def test_load_dotenv_no_match_returns_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When neither CWD-upward nor the outmem repo has a .env, return False."""
    import outmem.config as cfg

    isolated = tmp_path / "isolated"
    isolated.mkdir()
    monkeypatch.chdir(isolated)
    # Stub the repo discovery so a dev's actual clone-level .env
    # doesn't poison the test.
    monkeypatch.setattr(cfg, "_outmem_repo_dotenv", lambda: None)
    assert load_dotenv_if_present() is False


def test_load_dotenv_falls_back_to_outmem_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When CWD-upward has no .env, fall back to a .env at the outmem
    package's repo root. Lets users keep one .env next to their outmem
    clone and have it found from any CWD."""
    import outmem.config as cfg

    monkeypatch.delenv("OUTMEM_TEST_REPO_KEY", raising=False)
    fake_repo_env = tmp_path / ".env"
    fake_repo_env.write_text("OUTMEM_TEST_REPO_KEY=from-repo\n", encoding="utf-8")

    isolated = tmp_path / "isolated"
    isolated.mkdir()
    monkeypatch.chdir(isolated)
    monkeypatch.setattr(cfg, "_outmem_repo_dotenv", lambda: fake_repo_env)

    assert load_dotenv_if_present() is True
    assert os.environ.get("OUTMEM_TEST_REPO_KEY") == "from-repo"
    monkeypatch.delenv("OUTMEM_TEST_REPO_KEY", raising=False)


def test_load_dotenv_cwd_wins_over_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If both CWD-upward and the outmem repo have a .env, the
    CWD-side wins (it's the user's explicit local override)."""
    import outmem.config as cfg

    monkeypatch.delenv("OUTMEM_TEST_KEY", raising=False)
    cwd_env = tmp_path / "cwd" / ".env"
    cwd_env.parent.mkdir()
    cwd_env.write_text("OUTMEM_TEST_KEY=from-cwd\n", encoding="utf-8")
    repo_env = tmp_path / "repo" / ".env"
    repo_env.parent.mkdir()
    repo_env.write_text("OUTMEM_TEST_KEY=from-repo\n", encoding="utf-8")

    monkeypatch.chdir(cwd_env.parent)
    monkeypatch.setattr(cfg, "_outmem_repo_dotenv", lambda: repo_env)

    assert load_dotenv_if_present() is True
    assert os.environ.get("OUTMEM_TEST_KEY") == "from-cwd"
    monkeypatch.delenv("OUTMEM_TEST_KEY", raising=False)


def test_load_dotenv_does_not_override_existing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OUTMEM_TEST_KEY", "pre-existing")
    env_path = tmp_path / ".env"
    env_path.write_text("OUTMEM_TEST_KEY=from-file\n", encoding="utf-8")
    load_dotenv_if_present(env_path)
    assert os.environ.get("OUTMEM_TEST_KEY") == "pre-existing"


# ---------------------------------------------------------------------------
# starter file rendering
# ---------------------------------------------------------------------------


def test_starter_yaml_contains_required_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import outmem.config as cfg

    monkeypatch.setattr(cfg, "_outmem_repo_defaults", lambda: cfg.OutmemConfig())
    text = starter_yaml()
    assert "model: anthropic:" in text
    assert "remove_stale_lock: true" in text
    assert "retry_on_lock: true" in text
    assert "name: outmem agent" in text


def test_starter_yaml_respects_supplied_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import outmem.config as cfg

    monkeypatch.setattr(cfg, "_outmem_repo_defaults", lambda: cfg.OutmemConfig())
    text = starter_yaml(agent_name="bob-bot", agent_email="bob@example.com")
    assert "name: bob-bot" in text
    assert "email: bob@example.com" in text


def test_starter_yaml_explicit_model_wins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``model`` is passed explicitly, it goes into the rendered YAML
    regardless of any repo-level defaults."""
    import outmem.config as cfg

    monkeypatch.setattr(cfg, "_outmem_repo_defaults", lambda: cfg.OutmemConfig())
    text = starter_yaml(model="anthropic:claude-haiku-4-5-20251001")
    assert "model: anthropic:claude-haiku-4-5-20251001" in text


def test_starter_yaml_reads_repo_default_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``config.yaml`` at the outmem repo root sets the per-user
    default model that ``outmem init`` seeds new wikis with."""
    import outmem.config as cfg

    (tmp_path / "config.yaml").write_text(
        "model: anthropic:claude-haiku-4-5-20251001\n",
        encoding="utf-8",
    )
    fake_defaults = cfg.load_yaml_config(tmp_path)
    monkeypatch.setattr(cfg, "_outmem_repo_defaults", lambda: fake_defaults)

    text = starter_yaml()
    assert "model: anthropic:claude-haiku-4-5-20251001" in text


def test_starter_yaml_falls_back_to_builtin_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No repo-level config.yaml → built-in DEFAULT_MODEL is rendered."""
    import outmem.config as cfg

    monkeypatch.setattr(cfg, "_outmem_repo_defaults", lambda: cfg.OutmemConfig())
    text = starter_yaml()
    assert f"model: {DEFAULT_MODEL}" in text
