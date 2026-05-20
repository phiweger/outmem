"""Harness-only validation for the eval fixtures.

These tests run against the fixtures under ``evals/fixtures/wikis/``
**without** calling a real LLM — they exercise the parts of the eval
harness that we keep getting wrong (fixture layout, sources registry
JSON→DB conversion, ``SEED.md`` parsing, semantic reindex) so that
fixture bugs fail in regular ``pytest`` instead of mid-run on a real
eval pass.

Strategy:

* For every registered case, materialise its fixture into a tmp dir,
  flip its yaml flags, replay its ``SEED.md`` history.
* Open the resulting wiki with :class:`WikiStore` (so any
  malformed config / frontmatter / sources registry fails here).
* For ``semantic=True`` cases, call ``semantic_reindex_all`` with the
  bag-of-words stub — verifies the vector DB seeds cleanly.
* Run a scripted :class:`FunctionModel` agent that just calls
  ``append_log`` once to satisfy mandatory writeback. The agent's
  judgment is irrelevant here — we're only checking that the
  *plumbing* works.

If any of this throws, the case crashes locally in pytest before the
operator pays for a real model run.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

import evals.cases  # noqa: F401 — side-effect imports register cases
from evals.harness import (
    EvalCase,
    _flip_yaml_flags,
    _git_init_and_seed,
    _materialise_fixture,
    registered_cases,
)
from outmem.store import WikiStore


def _trivial_model(topic: str = "harness-validation") -> FunctionModel:
    """Always: call ``append_log`` once, then reply with text. Cheap and
    enough to satisfy mandatory writeback regardless of fixture."""
    state = {"step": 0}

    async def _runner(messages: list[object], info: AgentInfo) -> ModelResponse:
        idx = state["step"]
        state["step"] = idx + 1
        if idx == 0:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="append_log",
                        args={
                            "topic": topic,
                            "content": "- harness-validation smoke entry\n",
                        },
                    )
                ]
            )
        return ModelResponse(parts=[TextPart(content="done.")])

    return FunctionModel(_runner)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "wiki"
    ws.mkdir()
    return ws


@pytest.mark.parametrize(
    "case", registered_cases(), ids=lambda c: c.name
)
def test_fixture_materialises_and_seeds(case: EvalCase, workspace: Path) -> None:
    """Every fixture must copy, yaml-flip, and git-seed without errors."""
    from evals.harness import FIXTURES_ROOT

    fixture = FIXTURES_ROOT / case.wiki
    assert fixture.is_dir(), f"missing fixture dir: {fixture}"
    _materialise_fixture(fixture, workspace)
    _flip_yaml_flags(workspace, semantic=case.semantic, approval=case.approval)
    _git_init_and_seed(workspace, fixture)
    # Post-condition: git repo with at least one commit.
    assert (workspace / ".git").is_dir()


@pytest.mark.parametrize(
    "case", registered_cases(), ids=lambda c: c.name
)
def test_wiki_opens_cleanly(case: EvalCase, workspace: Path) -> None:
    """Opening the materialised fixture must not raise — catches
    malformed `config.yaml`, broken frontmatter, or a corrupt
    sources registry (we got bitten by this one)."""
    from evals.harness import FIXTURES_ROOT

    fixture = FIXTURES_ROOT / case.wiki
    _materialise_fixture(fixture, workspace)
    _flip_yaml_flags(workspace, semantic=case.semantic, approval=case.approval)
    _git_init_and_seed(workspace, fixture)
    store = WikiStore.open(workspace)
    # Exercises the registry parser (any registry-schema bug surfaces
    # here, not at agent run-time).
    store.list_sources()
    # And the page listing — catches frontmatter problems.
    store.list_slugs()


@pytest.mark.parametrize(
    "case",
    [c for c in registered_cases() if c.semantic],
    ids=lambda c: c.name,
)
def test_semantic_reindex_succeeds(case: EvalCase, workspace: Path) -> None:
    """Semantic-on fixtures must reindex with the bag-of-words stub
    without falling over. The eval harness does this before the
    agent runs; if it fails we want to know here."""
    from evals.harness import FIXTURES_ROOT

    fixture = FIXTURES_ROOT / case.wiki
    _materialise_fixture(fixture, workspace)
    _flip_yaml_flags(workspace, semantic=case.semantic, approval=case.approval)
    _git_init_and_seed(workspace, fixture)
    store = WikiStore.open(workspace)
    summary = store.semantic_reindex_all()
    assert summary["reindexed"] + summary["skipped"] >= 1, (
        f"{case.name}: nothing got indexed — fixture probably has no wiki "
        f"pages or registered sources"
    )


def test_seed_md_paths_exist_in_fixture() -> None:
    """If a SEED.md stanza references a path the fixture doesn't
    contain, the seed will silently skip it and stanza 1 will commit
    nothing — git refuses an empty commit and the case errors out.

    This static check catches the ``log/`` files-not-committed class
    of bug at lint time."""
    from evals.harness import FIXTURES_ROOT, _parse_seed_stanzas

    failures: list[str] = []
    for fixture in sorted(FIXTURES_ROOT.iterdir()):
        if not fixture.is_dir():
            continue
        seed = fixture / "SEED.md"
        if not seed.exists():
            continue
        for author, _email, subject, paths in _parse_seed_stanzas(
            seed.read_text(encoding="utf-8")
        ):
            for p in paths:
                target = fixture / p
                if not target.exists():
                    failures.append(
                        f"{fixture.name}/SEED.md: stanza "
                        f"`{author}|...|{subject}` references missing path "
                        f"`{p}` — add the file or fix the stanza"
                    )
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize(
    "case", registered_cases(), ids=lambda c: c.name
)
def test_harness_round_trip_with_scripted_agent(
    case: EvalCase, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end harness mechanics with a scripted agent.

    The agent is a deterministic :class:`FunctionModel` that just
    calls ``append_log`` once — enough to satisfy mandatory
    writeback regardless of the case's approval/semantic config.
    The case's assertions don't have to pass; we're checking that
    the harness ITSELF doesn't crash on any fixture.

    Approval-gated cases skip the deferred-tool resume loop because
    our scripted agent never calls a gated tool — the
    :class:`RecordingReviewer` is constructed but never consulted.
    """
    from evals.harness import FIXTURES_ROOT, _ToolCallRecorder
    from outmem.agent import RecordingReviewer, ask_sync

    fixture = FIXTURES_ROOT / case.wiki
    _materialise_fixture(fixture, workspace)
    _flip_yaml_flags(workspace, semantic=case.semantic, approval=case.approval)
    _git_init_and_seed(workspace, fixture)
    store = WikiStore.open(workspace)
    if case.semantic:
        store.semantic_reindex_all()

    reviewer = (
        RecordingReviewer(case.reviewer_verdicts) if case.approval else None
    )

    import logging

    recorder = _ToolCallRecorder()
    logger = logging.getLogger("outmem.agent.tool")
    logger.addHandler(recorder)
    logger.setLevel(logging.INFO)
    try:
        result = ask_sync(
            store,
            query="harness validation: just satisfy writeback",
            model=_trivial_model(topic=f"{case.name}-harness-smoke"),
            push=False,
            pull=False,
            record=False,
            reviewer=reviewer,
            include_steering=case.include_steering,
        )
    finally:
        logger.removeHandler(recorder)
    assert any(c.name == "append_log" for c in recorder.calls)
    assert any(c.subject.startswith("log:") for c in result.commits)


def teardown_module(_module: object) -> None:
    """Defensive cleanup — pytest's tmp_path fixture handles workspace,
    but we use ``shutil.copytree`` heavily and want any stray temp
    dirs swept regardless."""
    for path in Path("/tmp").glob("eval-validation-*"):
        shutil.rmtree(path, ignore_errors=True)
