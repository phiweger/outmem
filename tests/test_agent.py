"""Tests for ``outmem.agent``.

The agent runtime is exercised end-to-end via PydanticAI's
:class:`TestModel` with explicit tool-call scripts. The point is to
verify the orchestration contract (mandatory writeback, push retry,
last-run recording) — not to test PydanticAI itself.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from pydantic_ai.messages import (
    ModelResponse,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from outmem.agent import AskResult, ask_sync, build_agent, render_system_prompt
from outmem.exceptions import OutmemError, WritebackError
from outmem.store import WikiStore

# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> WikiStore:
    return WikiStore.init(tmp_path / "wiki")


@pytest.fixture
def seeded(store: WikiStore) -> WikiStore:
    store.write_page(
        "pricing-formula",
        title="Pricing formula",
        body="The pricing formula is cost-plus 35%.\n",
    )
    return store


def _model_that_calls(*calls: dict[str, object], reply: str = "done.") -> FunctionModel:
    """Build a FunctionModel that fires ``calls`` (one per turn) and
    then returns ``reply`` as text on the final turn.

    Each entry of ``calls`` is ``{"tool": "<name>", "args": {...}}``.
    """

    state = {"step": 0}

    async def _runner(messages: list[object], info: AgentInfo) -> ModelResponse:
        idx = state["step"]
        state["step"] = idx + 1
        if idx < len(calls):
            entry = calls[idx]
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name=str(entry["tool"]),
                        args=dict(entry["args"]),  # type: ignore[arg-type]
                    )
                ]
            )
        return ModelResponse(parts=[TextPart(content=reply)])

    return FunctionModel(_runner)


# ---------------------------------------------------------------------------
# System-prompt rendering
# ---------------------------------------------------------------------------


def test_render_system_prompt_contains_identity_and_root(store: WikiStore) -> None:
    prompt = render_system_prompt(store, include_steering=False)
    assert str(store.root) in prompt
    assert store.config.agent_identity.email in prompt
    # The orient/retrieve/compact discipline must show up.
    assert "PHASE 1 — ORIENT" in prompt
    assert "PHASE 3 — COMPACT" in prompt
    assert "MAY NOT produce neither" in prompt or "may not produce neither" in prompt.lower()


def test_render_system_prompt_includes_human_commits(populated_repo: Path) -> None:
    store = WikiStore.open(populated_repo)
    prompt = render_system_prompt(store, include_steering=True)
    # populated_repo has commits from alice and bob (humans) and from the
    # agent (excluded by steering()).
    assert "compact: pricing-formula" in prompt  # Alice's commit
    assert "compact: acme-msa" in prompt  # Bob's commit
    # Agent's own commits MUST NOT surface in the steering section,
    # even though `agent@host` appears in the identity boilerplate at
    # the top of the prompt.
    assert "extend: pricing-formula" not in prompt
    assert "log: pricing-inconsistency" not in prompt


def test_render_system_prompt_injects_bundled_skills(store: WikiStore) -> None:
    """Skill bodies (write / search / evolution) get rendered into the
    system prompt's `# Tool reference` section. Single source of
    truth: the bundled SKILL.md files under src/outmem/skills/notes/.

    Verifies anchors from each skill so a regression that loses one
    will fail this test, not be discovered mid-eval."""
    prompt = render_system_prompt(store, include_steering=False)
    assert "# Tool reference" in prompt
    # write skill anchors — the body= warning is the headline change
    # that fixed the Amikacin ingest crash.
    assert "## skill: write" in prompt
    assert "write_page(" in prompt
    assert "Required in every call" in prompt
    # search skill anchors
    assert "## skill: search" in prompt
    assert "search_wiki(" in prompt
    # evolution skill anchors
    assert "## skill: evolution" in prompt
    assert "topic_evolution(" in prompt
    # And the YAML frontmatter is NOT leaked into the prompt body.
    assert "name: write" not in prompt
    assert "name: search" not in prompt


def test_init_seeds_agents_md_under_wiki(tmp_path: Path) -> None:
    """`outmem init` writes `wiki/AGENTS.md` with starter conventions."""
    store = WikiStore.init(tmp_path / "w")
    assert store.agents_path.exists()
    body = store.agents_path.read_text(encoding="utf-8")
    assert "AGENTS.md" in body
    assert "What this wiki is for" in body


def test_render_system_prompt_injects_agents_md(tmp_path: Path) -> None:
    """A populated AGENTS.md lands verbatim under `# Wiki conventions`."""
    store = WikiStore.init(tmp_path / "w")
    store.agents_path.write_text(
        "## What this wiki is for\n\nPharma dosing wiki.\n", encoding="utf-8"
    )
    prompt = render_system_prompt(store, include_steering=False, inject_skills=())
    assert "# Wiki conventions" in prompt
    assert "Pharma dosing wiki." in prompt


def test_render_system_prompt_skips_wiki_conventions_when_agents_md_missing(
    tmp_path: Path,
) -> None:
    """No AGENTS.md → the conventions section is absent (existing wikis
    without the file keep working with built-in defaults)."""
    store = WikiStore.init(tmp_path / "w")
    store.agents_path.unlink()  # simulate a wiki without the file
    prompt = render_system_prompt(store, include_steering=False, inject_skills=())
    assert "# Wiki conventions" not in prompt


def test_render_system_prompt_can_suppress_skills(store: WikiStore) -> None:
    """Pass `inject_skills=()` to render the prompt without the tool
    reference appendix — useful for tests that pin the prompt size
    or callers who attach their own skill loader."""
    prompt = render_system_prompt(
        store, include_steering=False, inject_skills=()
    )
    assert "# Tool reference" not in prompt


def test_render_system_prompt_accepts_custom_skill_list(store: WikiStore) -> None:
    prompt = render_system_prompt(
        store, include_steering=False, inject_skills=("write",)
    )
    assert "## skill: write" in prompt
    assert "## skill: search" not in prompt
    assert "## skill: evolution" not in prompt


# ---------------------------------------------------------------------------
# build_agent: model resolution
# ---------------------------------------------------------------------------


def test_build_agent_requires_a_model(store: WikiStore, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OUTMEM_MODEL", raising=False)
    # Strip the seed-yaml model so resolution falls all the way through
    # to the error branch (the test's purpose pre-config).
    monkeypatch.setattr(store.config.outmem, "model", None)
    with pytest.raises(OutmemError, match="OUTMEM_MODEL"):
        build_agent(store)


def test_build_agent_uses_env_when_none_passed(
    store: WikiStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Env var is consulted but we don't actually invoke a real provider
    here — `Agent('test')` is what PydanticAI accepts for the in-memory
    TestModel string id."""
    monkeypatch.setenv("OUTMEM_MODEL", "test")
    agent = build_agent(store, include_steering=False)
    assert agent is not None


def test_build_agent_sets_anthropic_cache_defaults(store: WikiStore) -> None:
    """The Anthropic prompt-caching knobs ride on `model_settings`.

    The same settings dict is provider-agnostic — Anthropic picks the
    `anthropic_*` keys up; non-Anthropic models silently ignore them.
    Multi-turn ingest runs would otherwise re-ship the system prompt +
    tool defs + long tool results on every chat (~5-10x overspend).
    """
    from pydantic_ai.models.test import TestModel

    agent = build_agent(store, model=TestModel(), include_steering=False)
    settings = agent.model_settings or {}
    assert settings.get("anthropic_cache") is True
    assert settings.get("anthropic_cache_instructions") is True
    assert settings.get("anthropic_cache_tool_definitions") is True


def test_build_agent_caller_can_override_cache_settings(store: WikiStore) -> None:
    """Caller-supplied model_settings win over the defaults."""
    from pydantic_ai.models.test import TestModel

    agent = build_agent(
        store,
        model=TestModel(),
        include_steering=False,
        model_settings={"anthropic_cache": False, "max_tokens": 2048},
    )
    settings = agent.model_settings or {}
    assert settings.get("anthropic_cache") is False
    assert settings.get("max_tokens") == 2048
    # Other defaults still fill in.
    assert settings.get("anthropic_cache_instructions") is True


# ---------------------------------------------------------------------------
# ask_sync: end-to-end contract
# ---------------------------------------------------------------------------


def test_ask_writeback_happy_path(seeded: WikiStore) -> None:
    """Agent calls append_log, the service records the commit and
    returns the response."""
    model = _model_that_calls(
        {
            "tool": "append_log",
            "args": {"topic": "test-finding", "content": "- learned x"},
        },
        reply="x is learned.",
    )
    result = ask_sync(
        seeded,
        query="What is x?",
        model=model,
        push=False,
        record=False,
    )
    assert isinstance(result, AskResult)
    assert result.response == "x is learned."
    assert result.wrote_back
    assert result.head_before is not None
    assert result.head_after != result.head_before


def test_ask_no_writeback_raises(seeded: WikiStore) -> None:
    """If the model returns without calling any write tool, the service
    surfaces a WritebackError per spec v0.5 §9."""
    model = _model_that_calls(reply="I have nothing to say.")
    with pytest.raises(WritebackError, match="no commits"):
        ask_sync(seeded, query="x?", model=model, push=False, record=False)


def test_ask_writes_new_page(seeded: WikiStore) -> None:
    model = _model_that_calls(
        {
            "tool": "write_page",
            "args": {
                "slug": "from-agent",
                "title": "From agent",
                "body": "Body written by the agent.\n",
            },
        },
        reply="wrote a new page.",
    )
    result = ask_sync(seeded, query="please add a page", model=model, push=False, record=False)
    assert result.wrote_back
    page = seeded.read("from-agent")
    assert "Body written by the agent" in page.body


def test_ask_extends_existing_page(seeded: WikiStore) -> None:
    model = _model_that_calls(
        {
            "tool": "extend_page",
            "args": {"slug": "pricing-formula", "body": "Revised: 40%.\n"},
        },
        reply="updated.",
    )
    result = ask_sync(
        seeded,
        query="revise pricing",
        model=model,
        push=False,
        record=False,
    )
    assert result.wrote_back
    assert "40%" in seeded.read("pricing-formula").body


def test_ask_offline_mode_skips_pull_and_push(seeded: WikiStore) -> None:
    """No remote is configured; with push=False this must succeed."""
    model = _model_that_calls(
        {"tool": "append_log", "args": {"topic": "offline", "content": "- x"}},
        reply="ok.",
    )
    result = ask_sync(
        seeded,
        query="offline?",
        model=model,
        push=False,
        pull=False,
        record=False,
    )
    assert result.pushed is False


def test_ask_records_run_when_enabled(seeded: WikiStore) -> None:
    assert seeded.last_run() is None
    model = _model_that_calls(
        {"tool": "append_log", "args": {"topic": "x", "content": "- y"}},
        reply="ok.",
    )
    ask_sync(seeded, query="x?", model=model, push=False, record=True)
    marker = seeded.last_run()
    assert marker is not None
    assert marker.head == seeded.head()


# ---------------------------------------------------------------------------
# CLI: outmem ask
# ---------------------------------------------------------------------------


def test_cli_ask_writes_response_to_stdout(
    seeded: WikiStore,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI command should accept --stdin, no-push, and produce a
    real response when given a TestModel by env override.

    We patch `outmem.agent.service.build_agent` to return an agent
    backed by FunctionModel so we don't need a real provider."""
    from outmem.agent import service

    model = _model_that_calls(
        {"tool": "append_log", "args": {"topic": "cli", "content": "- cli-test"}},
        reply="cli reply.",
    )
    real_build = service.build_agent

    def fake_build(store, **kwargs):  # type: ignore[no-untyped-def]
        kwargs.pop("model", None)
        return real_build(store, model=model, **kwargs)

    monkeypatch.setattr(service, "build_agent", fake_build)

    from outmem.cli.__main__ import main

    monkeypatch.setattr("sys.stdin", io.StringIO("what?"))
    rc = main(
        [
            "--root",
            str(seeded.root),
            "ask",
            "--stdin",
            "--no-push",
            "--no-record",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "cli reply." in out


def test_cli_ask_empty_query_rejected(
    seeded: WikiStore,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from outmem.cli.__main__ import main

    monkeypatch.setattr("sys.stdin", io.StringIO("   "))
    rc = main(["--root", str(seeded.root), "ask", "--stdin"])
    assert rc == 2
    assert "empty" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Writeback contract — TOCTOU fix and push retry
# ---------------------------------------------------------------------------


def test_writeback_ignores_non_agent_commits(
    seeded: WikiStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A concurrent human commit moves HEAD but doesn't count as the
    agent writing back. The agent must produce its own commit or we
    surface WritebackError (regression test for the TOCTOU bug
    where ``head_before != head_after`` falsely declared success)."""
    from outmem.agent import service as svc
    from outmem.git_ops import add as git_add
    from outmem.git_ops import commit_as

    # Model that doesn't call any tools — should trigger WritebackError.
    model = _model_that_calls(reply="all done.")
    real_build = svc.build_agent

    def building_with_concurrent_human(store, **kwargs):  # type: ignore[no-untyped-def]
        kwargs.pop("model", None)
        # Simulate a concurrent human commit landing during the run —
        # exactly the window the old _new_commits() function would
        # have mistaken for an agent writeback.
        (store.root / "wiki/pages/human-page.md").write_text(
            "---\ntitle: Human\nslug: human-page\n---\n\nbody\n"
        )
        git_add(store.root, ["wiki/pages/human-page.md"])
        commit_as(
            store.root,
            message="compact: human-page",
            author_name="Human Bob",
            author_email="bob@example.com",
        )
        return real_build(store, model=model, **kwargs)

    monkeypatch.setattr(svc, "build_agent", building_with_concurrent_human)

    with pytest.raises(WritebackError, match="no commits"):
        ask_sync(seeded, query="x?", model=model, push=False, record=False)


def _wire_remote(store: WikiStore, bare_remote: Path) -> None:
    """Helper: attach a bare remote so the push-retry tests can exercise
    the *push* failure path (rather than the no-remote short-circuit)."""
    import subprocess

    subprocess.run(
        ["git", "remote", "add", "origin", str(bare_remote)],
        cwd=str(store.root),
        check=True,
        capture_output=True,
    )


def test_push_retry_second_failure_raises(
    seeded: WikiStore,
    bare_remote: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The mandatory-writeback contract (spec v0.5 §9) raises
    WritebackError if push fails after a single pull-rebase retry."""
    from outmem.agent.service import _push_with_retry
    from outmem.exceptions import GitOperationError

    _wire_remote(seeded, bare_remote)

    push_calls = {"count": 0}
    pull_calls = {"count": 0}

    def fail_push() -> None:
        push_calls["count"] += 1
        raise GitOperationError("simulated push rejection")

    def fake_pull() -> None:
        pull_calls["count"] += 1
        # Pull "succeeds" so the retry path tries push again.

    monkeypatch.setattr(seeded, "push", fail_push)
    monkeypatch.setattr(seeded, "pull", fake_pull)

    with pytest.raises(WritebackError, match="after one pull-rebase retry"):
        _push_with_retry(seeded)

    assert push_calls["count"] == 2  # initial + one retry
    assert pull_calls["count"] == 1  # exactly one pull-rebase


def test_push_retry_succeeds_flags_concurrent_commit(
    seeded: WikiStore,
    bare_remote: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When push fails once then succeeds after pull-rebase, the outcome
    flags concurrent_human_commit_landed so the caller can warn the user
    (spec §9: 're-read the affected file')."""
    from outmem.agent.service import _push_with_retry
    from outmem.exceptions import GitOperationError

    _wire_remote(seeded, bare_remote)

    push_calls = {"count": 0}

    def push_succeeds_on_retry() -> None:
        push_calls["count"] += 1
        if push_calls["count"] == 1:
            raise GitOperationError("initial push rejected")
        # second call succeeds

    monkeypatch.setattr(seeded, "push", push_succeeds_on_retry)
    monkeypatch.setattr(seeded, "pull", lambda: None)

    outcome = _push_with_retry(seeded)
    assert outcome.pushed is True
    assert outcome.concurrent_human_commit_landed is True


def test_ask_result_exposes_commit_shas_and_subjects(seeded: WikiStore) -> None:
    """commit_shas + commit_subjects expose what the agent actually
    wrote, not a copy of internal data."""
    model = _model_that_calls(
        {"tool": "append_log", "args": {"topic": "test-x", "content": "- ok"}},
        reply="ok.",
    )
    result = ask_sync(seeded, query="x?", model=model, push=False, record=False)
    assert len(result.commit_shas) == 1
    assert len(result.commit_shas[0]) == 40
    assert result.commit_subjects == ("log: test-x",)
    assert result.concurrent_human_commit_landed is False


# ---------------------------------------------------------------------------
# Local-only wikis — no remote, push must be skipped not retried
# ---------------------------------------------------------------------------


def test_push_skipped_when_no_remote_configured(seeded: WikiStore) -> None:
    """A local `git init` with no remote is the README quickstart case.
    The service must NOT try to push (and fail) — local commits ARE the
    writeback."""
    # seeded is a fresh WikiStore.init — no remote configured.
    model = _model_that_calls(
        {"tool": "append_log", "args": {"topic": "local-only", "content": "- ok"}},
        reply="local reply.",
    )
    result = ask_sync(seeded, query="x?", model=model, push=True, record=False)
    assert result.wrote_back
    assert result.pushed is False  # nothing to push to
    assert result.concurrent_human_commit_landed is False


def test_push_with_retry_skips_when_no_remote(seeded: WikiStore) -> None:
    from outmem.agent.service import _push_with_retry

    outcome = _push_with_retry(seeded)
    assert outcome.pushed is False
    assert outcome.concurrent_human_commit_landed is False


def test_has_remote_detects_origin(seeded: WikiStore, bare_remote: Path) -> None:
    """``has_remote`` distinguishes local-only from origin-wired repos."""
    from outmem.git_ops import has_remote

    assert has_remote(seeded.root) is False
    import subprocess

    subprocess.run(
        ["git", "remote", "add", "origin", str(bare_remote)],
        cwd=str(seeded.root),
        check=True,
        capture_output=True,
    )
    assert has_remote(seeded.root) is True


# ---------------------------------------------------------------------------
# Tool errors are recoverable (regression: bug in user trace)
# ---------------------------------------------------------------------------


def test_read_page_returns_error_string_on_missing_page(seeded: WikiStore) -> None:
    """Calling read_page with a non-existent slug must NOT raise — the
    model needs to see the error and try a different tool. This was the
    bug behind 'outmem: No such wiki page: pricing-deck-2026-q1'."""
    from outmem.adapters.pydantic_ai import wiki_tools

    tools = wiki_tools(seeded)
    read_page = next(t for t in tools if t.__name__ == "read_page")
    out = read_page(slug="pricing-deck-2026-q1")
    assert isinstance(out, str)
    assert "no such wiki page" in out.lower()
    # The hint should nudge the model toward search_wiki / list_pages.
    assert "search_wiki" in out or "list_pages" in out


def test_read_page_returns_error_string_on_invalid_slug(seeded: WikiStore) -> None:
    from outmem.adapters.pydantic_ai import wiki_tools

    read_page = next(t for t in wiki_tools(seeded) if t.__name__ == "read_page")
    out = read_page(slug="Bad Slug")
    assert "invalid slug" in out.lower()


def test_write_page_duplicate_returns_error_string(seeded: WikiStore) -> None:
    """`pricing-formula` already exists in the seeded store; trying to
    write_page over it must surface as a recoverable error string."""
    from outmem.adapters.pydantic_ai import wiki_tools

    write_page = next(t for t in wiki_tools(seeded) if t.__name__ == "write_page")
    out = write_page(
        slug="pricing-formula",
        title="X",
        body="duplicate attempt\n",
    )
    assert "write_page failed" in out
    assert "already exists" in out


def test_extend_page_unknown_returns_error_string(seeded: WikiStore) -> None:
    from outmem.adapters.pydantic_ai import wiki_tools

    extend_page = next(t for t in wiki_tools(seeded) if t.__name__ == "extend_page")
    out = extend_page(slug="does-not-exist", body="x\n")
    assert "extend_page failed" in out
    # Hint should point at write_page.
    assert "write_page" in out


# ---------------------------------------------------------------------------
# Tool-call logging
# ---------------------------------------------------------------------------


def test_tool_calls_emit_log_records(seeded: WikiStore, caplog: pytest.LogCaptureFixture) -> None:
    """Every tool call writes one INFO record to ``outmem.agent.tool``."""
    import logging

    from outmem.adapters.pydantic_ai import wiki_tools

    tools = wiki_tools(seeded)
    search_wiki = next(t for t in tools if t.__name__ == "search_wiki")
    read_page = next(t for t in tools if t.__name__ == "read_page")

    with caplog.at_level(logging.INFO, logger="outmem.agent.tool"):
        search_wiki(pattern="cost-plus")
        read_page(slug="pricing-formula")

    messages = [r.getMessage() for r in caplog.records]
    assert any("search_wiki" in m for m in messages)
    assert any("read_page" in m and "pricing-formula" in m for m in messages)


def test_log_summary_truncates_long_strings(
    seeded: WikiStore, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    from outmem.adapters.pydantic_ai import wiki_tools

    tools = wiki_tools(seeded)
    write_page = next(t for t in tools if t.__name__ == "write_page")

    long_body = "x" * 500
    with caplog.at_level(logging.INFO, logger="outmem.agent.tool"):
        write_page(slug="long-page", title="Long", body=long_body)

    msg = " ".join(r.getMessage() for r in caplog.records)
    # The body should be summarised as "(N chars)" not pasted in full.
    assert "(500 chars)" in msg
    assert "xxxxxxxxxxxxxxxxxxxxxxxxxxxx" not in msg


def test_format_validation_detail_extracts_pydantic_errors() -> None:
    """When PydanticAI runs out of tool retries it raises
    UnexpectedModelBehavior with the underlying ValidationError chained
    via ``__cause__``. The user-visible WritebackError message has to
    surface those validation details — otherwise the user just sees
    'exceeded max retries' with no clue what was wrong with the args."""
    from pydantic import BaseModel, ValidationError

    from outmem.exceptions import format_validation_detail

    class Args(BaseModel):
        slug: str
        title: str
        body: str

    try:
        Args.model_validate({"slug": "amikacin-side-effects", "title": "X"})
    except ValidationError as ve:
        try:
            raise RuntimeError("Tool 'write_page' exceeded max retries count of 3") from ve
        except RuntimeError as outer:
            detail = format_validation_detail(outer)
            assert "body" in detail
            assert "Field required" in detail
            assert detail.startswith(" — validation errors: ")


def test_format_validation_detail_empty_when_no_validation_error() -> None:
    """Non-validation failures (network, etc.) should produce no
    detail suffix — the message stays clean."""
    from outmem.exceptions import format_validation_detail

    assert format_validation_detail(RuntimeError("boom")) == ""


def test_format_validation_detail_handles_cause_cycle() -> None:
    """Pathological case: a circular ``__cause__`` chain. Must not loop
    forever — the walker terminates via a `seen` set."""
    from outmem.exceptions import format_validation_detail

    a = RuntimeError("a")
    b = RuntimeError("b")
    a.__cause__ = b
    b.__cause__ = a  # cycle

    # No ValidationError in the chain → empty detail, but the key thing
    # is that we return (not hang).
    assert format_validation_detail(a) == ""
