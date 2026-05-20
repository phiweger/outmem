"""Harness: build a wiki from a fixture, run the agent, record everything.

Single :class:`EvalRun` carries the recorded tool calls, the commits
the agent produced, the final response, the model usage (token / cost
estimate), and provides the ``expect_*`` and ``judge`` assertion
helpers used by individual cases.

The harness does *not* assume pytest — cases are plain functions
registered via :func:`eval_case`. ``evals.run`` and the tiny pytest
shim in ``evals.cases.conftest`` both reuse the same :func:`run_case`.
"""

from __future__ import annotations

import logging
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

# Lazy: pulled in only when a case is actually invoked.
_REGISTRY: list[EvalCase] = []


@dataclass
class EvalCase:
    """A registered eval case (declarative metadata + body)."""

    name: str
    """Stable identifier — used by ``--case <name>`` and in reports."""

    wiki: str
    """Directory under ``evals/fixtures/wikis/`` to copy as the test wiki."""

    query: str
    """User prompt fed to ``ask_sync``."""

    body: Callable[[EvalRun], None]
    """The case function. Receives an :class:`EvalRun` and calls
    ``expect_*`` / ``judge`` against it."""

    description: str = ""
    """Optional one-line summary for the report."""

    semantic: bool = False
    """Whether to flip ``semantic.enabled: true`` in the wiki config."""

    approval: bool = False
    """Whether to flip ``approval.required_for_writes: true``."""

    reviewer_verdicts: dict[str, list[Any]] = field(default_factory=dict)
    """When ``approval=True``, programmed verdicts keyed by tool name.
    Forwarded into :class:`outmem.agent.RecordingReviewer`."""

    include_steering: bool = False
    """Whether to render recent human commits into the agent's system
    prompt (PHASE 1 steering). Off by default so seeded commit history
    doesn't pollute the prompt with noise; flip on for cases where the
    authored history IS the test (e.g. multi-author divergence)."""


def eval_case(
    *,
    wiki: str,
    query: str,
    description: str = "",
    semantic: bool = False,
    approval: bool = False,
    reviewer_verdicts: dict[str, list[Any]] | None = None,
    include_steering: bool = False,
) -> Callable[[Callable[[EvalRun], None]], Callable[[EvalRun], None]]:
    """Decorator to register an eval case.

    Example::

        @eval_case(wiki="pricing-cost-plus", query="what is the formula?")
        def case_pricing_lookup(r: EvalRun) -> None:
            r.expect_tool_called("read_page", slug="pricing-formula")
            r.judge("answer mentions cost-plus 35%")
    """

    def decorator(fn: Callable[[EvalRun], None]) -> Callable[[EvalRun], None]:
        case = EvalCase(
            name=fn.__name__.removeprefix("case_").replace("_", "-"),
            wiki=wiki,
            query=query,
            body=fn,
            description=description or (fn.__doc__ or "").strip().split("\n")[0],
            semantic=semantic,
            approval=approval,
            reviewer_verdicts=dict(reviewer_verdicts or {}),
            include_steering=include_steering,
        )
        _REGISTRY.append(case)
        fn._eval_case = case  # type: ignore[attr-defined]
        return fn

    return decorator


def registered_cases() -> list[EvalCase]:
    """Return every registered case (after the case modules have been imported)."""
    return list(_REGISTRY)


# ---------------------------------------------------------------------------
# Tool-call recorder
# ---------------------------------------------------------------------------


@dataclass
class ToolCall:
    """One recorded tool invocation."""

    name: str
    args: dict[str, Any]


class _ToolCallRecorder(logging.Handler):
    """Capture ``outmem.agent.tool`` LogRecords as :class:`ToolCall`.

    Relies on :func:`outmem.adapters.pydantic_ai._log_call` attaching
    ``tool_call=(name, kwargs)`` as a structured ``extra`` on each
    record. Errors emit a ``tool_error`` extra instead; we skip those
    for the trace (errors don't *do* anything trace-relevant — the
    agent's recovery via a second call is what matters).
    """

    def __init__(self, *, live_stream: IO[str] | None = None) -> None:
        super().__init__(level=logging.INFO)
        self.calls: list[ToolCall] = []
        self._stream = live_stream

    def emit(self, record: logging.LogRecord) -> None:
        tc = getattr(record, "tool_call", None)
        if tc is not None:
            name, args = tc
            self.calls.append(ToolCall(name=str(name), args=dict(args)))
            if self._stream is not None:
                # Re-use the same compact format the CLI uses for
                # `outmem ask`: `[tool] name k=v k=v` on one line.
                self._stream.write(f"    [tool] {record.getMessage()}\n")
                self._stream.flush()
            return
        te = getattr(record, "tool_error", None)
        if te is not None and self._stream is not None:
            name, exc_type, msg = te
            self._stream.write(f"    [tool] {name} → ERROR {exc_type}: {msg}\n")
            self._stream.flush()


# ---------------------------------------------------------------------------
# EvalRun: per-case state + assertion helpers
# ---------------------------------------------------------------------------


@dataclass
class AssertionRecord:
    """One trace or judge assertion + its outcome."""

    kind: str  # "trace" | "judge"
    description: str
    passed: bool
    detail: str = ""


@dataclass
class EvalRun:
    """Everything one case observed about an agent run."""

    case: EvalCase
    response: str
    commits: tuple[Any, ...]
    tool_calls: list[ToolCall]
    duration_s: float
    cost_estimate_usd: float | None = None
    assertions: list[AssertionRecord] = field(default_factory=list)
    progress: IO[str] | None = None  # set by ``run_case`` when --quiet is off
    skipped: bool = False
    skip_reason: str = ""

    # ----- trace -----

    def expect_tool_called(self, name: str, **arg_filters: Any) -> None:
        """Assert at least one recorded call to ``name`` whose args
        contain every ``(key, value)`` in ``arg_filters``.

        ``arg_filters`` accepts:
        * exact equality (``slug="pricing-formula"``);
        * substring match on string args via the special
          ``<key>__contains`` form (``pattern__contains="pricing"``);
        * a callable predicate (``slugs=lambda v: "discounts" in v``).
        """
        matches = [c for c in self.tool_calls if c.name == name]
        if not matches:
            self._record_trace(
                f"tool {name!r} called",
                False,
                detail=(
                    f"called tools were: "
                    f"{sorted({c.name for c in self.tool_calls})}"
                ),
            )
            return
        for call in matches:
            if _args_match(call.args, arg_filters):
                self._record_trace(
                    f"tool {name!r} called with {arg_filters!r}", True
                )
                return
        actual = [c.args for c in matches]
        self._record_trace(
            f"tool {name!r} called with {arg_filters!r}",
            False,
            detail=f"saw {name!r} called with: {actual[:5]}",
        )

    def expect_no_tool_called(self, name: str) -> None:
        """Negative trace assertion — typically used to confirm the agent
        did NOT escalate to a heavier tool (e.g. ``read_source``)."""
        hit = [c for c in self.tool_calls if c.name == name]
        if hit:
            self._record_trace(
                f"tool {name!r} NOT called",
                False,
                detail=f"saw {len(hit)} call(s): {[c.args for c in hit[:3]]}",
            )
        else:
            self._record_trace(f"tool {name!r} NOT called", True)

    def expect_no_commit(self, *, subject_matches: str) -> None:
        """Assert that NO agent commit subject matches ``subject_matches``.

        Use to pin negative outcomes — e.g. after a write was denied,
        assert no ``compact:`` or ``extend:`` commit landed.
        """
        import re

        pat = re.compile(subject_matches)
        offenders = [c.subject for c in self.commits if pat.search(c.subject)]
        if offenders:
            self._record_trace(
                f"no commit matching /{subject_matches}/",
                False,
                detail=f"unexpected commit(s): {offenders}",
            )
        else:
            self._record_trace(f"no commit matching /{subject_matches}/", True)

    def expect_commit(
        self,
        *,
        subject_matches: str | None = None,
        subject_contains: str | None = None,
    ) -> None:
        """Assert at least one agent-authored commit matched the pattern."""
        import re

        subjects = [c.subject for c in self.commits]
        if subject_matches is not None:
            pat = re.compile(subject_matches)
            if any(pat.search(s) for s in subjects):
                self._record_trace(
                    f"commit matching /{subject_matches}/", True
                )
                return
            self._record_trace(
                f"commit matching /{subject_matches}/",
                False,
                detail=f"saw subjects: {subjects}",
            )
            return
        if subject_contains is not None:
            if any(subject_contains in s for s in subjects):
                self._record_trace(
                    f"commit containing {subject_contains!r}", True
                )
                return
            self._record_trace(
                f"commit containing {subject_contains!r}",
                False,
                detail=f"saw subjects: {subjects}",
            )
            return
        # No filter — just assert there's at least one commit.
        if subjects:
            self._record_trace(f"at least one commit ({len(subjects)})", True)
        else:
            self._record_trace("at least one commit", False, detail="no commits")

    # ----- judge -----

    def judge(self, criterion: str) -> None:
        """LLM-judge assertion: did the response satisfy ``criterion``?

        Off when ``--no-judge`` was passed; the assertion is skipped
        with ``passed=True`` and a ``[skipped]`` detail so the case
        report still shows it.
        """
        from evals.judges.llm_judge import grade

        if not _JUDGE_ENABLED:
            self._record_assertion(
                AssertionRecord(
                    kind="judge",
                    description=criterion,
                    passed=True,
                    detail="[skipped — --no-judge]",
                )
            )
            return

        # Live progress: tell the operator we're paying for a judge
        # call BEFORE it blocks on the model. Cleared by the
        # final ✓/✗ line that ``_record_assertion`` prints.
        if self.progress is not None:
            self.progress.write(f"    … judging: {criterion}\n")
            self.progress.flush()

        verdict = grade(criterion=criterion, response=self.response)
        self._record_assertion(
            AssertionRecord(
                kind="judge",
                description=criterion,
                passed=verdict.passed,
                detail=verdict.reasoning,
            )
        )

    # ----- internals -----

    def _record_trace(self, description: str, passed: bool, *, detail: str = "") -> None:
        self._record_assertion(
            AssertionRecord(
                kind="trace",
                description=description,
                passed=passed,
                detail=detail,
            )
        )

    def _record_assertion(self, record: AssertionRecord) -> None:
        """Append + optionally echo to the progress stream.

        Live output uses ``✓`` / ``✗`` so success / failure pop visually;
        ``[skipped]`` judge calls keep their detail inline. The full
        report rendered at the end of the run remains unchanged — the
        progress lines are pure-additive UX."""
        self.assertions.append(record)
        if self.progress is None:
            return
        tick = "✓" if record.passed else "✗"
        label = "trace" if record.kind == "trace" else "judge"
        line = f"    {tick} {label}: {record.description}"
        self.progress.write(line + "\n")
        if record.detail and (not record.passed or record.detail.startswith("[skipped")):
            self.progress.write(f"        {record.detail}\n")
        self.progress.flush()

    @property
    def passed(self) -> bool:
        if self.skipped:
            # Skipped cases are NOT failures — they're "didn't run".
            # The report renderer surfaces them with [SKIP].
            return True
        return all(a.passed for a in self.assertions)


def _args_match(actual: dict[str, Any], filters: dict[str, Any]) -> bool:
    for key, expected in filters.items():
        if key.endswith("__contains"):
            base = key.removesuffix("__contains")
            value = actual.get(base)
            if not isinstance(value, str) or expected not in value:
                return False
            continue
        if callable(expected):
            if not expected(actual.get(key)):
                return False
            continue
        if actual.get(key) != expected:
            return False
    return True


# ---------------------------------------------------------------------------
# Judge enable/disable (module-global flag flipped by run.py / conftest)
# ---------------------------------------------------------------------------


_JUDGE_ENABLED = True


def set_judge_enabled(enabled: bool) -> None:
    """Toggle the LLM judge globally. ``False`` makes :meth:`EvalRun.judge`
    skip with a ``[skipped]`` detail so the report still includes the
    criterion."""
    global _JUDGE_ENABLED
    _JUDGE_ENABLED = enabled


# ---------------------------------------------------------------------------
# Case runner
# ---------------------------------------------------------------------------


FIXTURES_ROOT = Path(__file__).resolve().parent / "fixtures" / "wikis"


def run_case(
    case: EvalCase,
    *,
    workspace: Path,
    model: str | None = None,
    progress: IO[str] | None = None,
) -> EvalRun:
    """Execute one case end-to-end and return the resulting :class:`EvalRun`.

    Caller supplies an empty ``workspace`` directory (typically
    ``tmp_path`` under pytest). The fixture wiki is copied in, the
    repo is initialised, the agent runs against the user query, and
    everything observable is captured.

    When ``progress`` is a writable stream (typically ``sys.stderr``),
    the harness streams a live trace:

    * ``== <case-name>`` header + the user query.
    * ``[tool] …`` one line per agent tool call as they happen.
    * ``… judging: <criterion>`` immediately before each judge call.
    * ``✓ / ✗ <kind>: <description>`` one line per resolved assertion.
    * ``-- <PASS|FAIL> <case-name> (Ns, K tool calls, M commits)`` footer.

    Pass ``progress=None`` (CLI ``--quiet``) to suppress the live
    output; the final report rendered by :func:`evals.run.main` still
    contains everything.
    """
    from outmem.agent import RecordingReviewer, ask_sync
    from outmem.store import WikiStore

    fixture = FIXTURES_ROOT / case.wiki
    if not fixture.is_dir():
        raise FileNotFoundError(
            f"Fixture wiki {case.wiki!r} not found at {fixture}. "
            f"Add it under evals/fixtures/wikis/."
        )

    if progress is not None:
        progress.write(f"\n== {case.name}\n")
        if case.description:
            progress.write(f"   {case.description}\n")
        progress.write(f"   query: {case.query!r}\n")
        progress.flush()

    _materialise_fixture(fixture, workspace)
    _flip_yaml_flags(workspace, semantic=case.semantic, approval=case.approval)
    _git_init_and_seed(workspace, fixture)

    store = WikiStore.open(workspace)

    # If the case wants semantic retrieval, pre-populate the vector
    # index with whatever the fixture already had on disk. Without this
    # step, the agent's first `find_similar` call would query an empty
    # index — silently defeating any duplicate-trap style case.
    #
    # Embedder construction is a real network/auth boundary (OpenAI
    # default needs OPENAI_API_KEY). Surface that as a clean "skipped"
    # eval rather than a generic crash so the operator sees what's
    # missing.
    if case.semantic:
        try:
            store.semantic_reindex_all()
        except Exception as exc:
            reason = (
                f"semantic init failed ({type(exc).__name__}: {exc}). "
                "This case needs an embedder API key — likely "
                "OPENAI_API_KEY for the default text-embedding-3-small."
            )
            if progress is not None:
                progress.write(f"    [skipped] {reason}\n")
                progress.write(f"-- SKIP {case.name}\n")
                progress.flush()
            return EvalRun(
                case=case,
                response="",
                commits=(),
                tool_calls=[],
                duration_s=0.0,
                progress=progress,
                skipped=True,
                skip_reason=reason,
            )

    reviewer = None
    if case.approval:
        reviewer = RecordingReviewer(case.reviewer_verdicts)

    recorder = _ToolCallRecorder(live_stream=progress)
    logger = logging.getLogger("outmem.agent.tool")
    prev_level = logger.level
    logger.addHandler(recorder)
    logger.setLevel(logging.INFO)

    started = time.perf_counter()
    try:
        result = ask_sync(
            store,
            query=case.query,
            model=model,
            push=False,
            pull=False,
            record=False,
            reviewer=reviewer,
            include_steering=case.include_steering,
        )
    finally:
        logger.removeHandler(recorder)
        logger.setLevel(prev_level)
    duration = time.perf_counter() - started

    run = EvalRun(
        case=case,
        response=result.response,
        commits=tuple(result.commits),
        tool_calls=list(recorder.calls),
        duration_s=duration,
        progress=progress,
    )

    # Execute the case body — its ``expect_*`` and ``judge`` calls
    # accumulate AssertionRecords on the EvalRun (and stream to
    # ``progress`` as they resolve, when set).
    case.body(run)

    if progress is not None:
        if run.skipped:
            label = "SKIP"
        elif run.passed:
            label = "PASS"
        else:
            label = "FAIL"
        progress.write(
            f"-- {label} {case.name} "
            f"({duration:.1f}s, {len(run.tool_calls)} tool calls, "
            f"{len(run.commits)} commit(s))\n"
        )
        progress.flush()

    return run


def _materialise_fixture(fixture: Path, dest: Path) -> None:
    """Copy a fixture directory's contents into ``dest``.

    Symlinks are followed; ``.git`` is never copied (we re-init).
    """
    for entry in fixture.iterdir():
        if entry.name in (".git", "__pycache__"):
            continue
        target = dest / entry.name
        if entry.is_dir():
            shutil.copytree(entry, target)
        else:
            shutil.copy2(entry, target)


def _flip_yaml_flags(workspace: Path, *, semantic: bool, approval: bool) -> None:
    """Mutate the wiki's ``config.yaml`` to flip per-case feature flags
    without committing case-specific yaml under every fixture.

    For semantic cases we also swap in the deterministic
    ``test:bag-of-words`` stub embedder
    (:mod:`outmem.semantic.testing`) and lower the similarity
    threshold so hash-bucket BoW vectors actually clear it. This
    keeps evals free / offline / reproducible — there's no point
    paying OpenAI to verify the harness mechanics.

    To exercise a real embedder, set ``OUTMEM_EVAL_REAL_EMBEDDER=1``
    in the environment before calling ``python -m evals.run`` — the
    fixture's configured ``embedding_model`` is then left alone.
    """
    import os

    config = workspace / "config.yaml"
    if not config.exists():
        return
    text = config.read_text(encoding="utf-8")
    if semantic:
        text = text.replace(
            "semantic:\n  enabled: false",
            "semantic:\n  enabled: true",
        )
        if not os.environ.get("OUTMEM_EVAL_REAL_EMBEDDER"):
            text = text.replace(
                "embedding_model: openai:text-embedding-3-small",
                "embedding_model: test:bag-of-words",
            )
            # Bag-of-words on small fixture text rarely hits 0.8; the
            # case still cares that find_similar surfaces the duplicate,
            # not the exact similarity number. 0.2 lets paraphrased
            # overlap through while keeping unrelated chunks out.
            text = text.replace(
                "similarity_threshold: 0.8",
                "similarity_threshold: 0.2",
            )
    if approval:
        text = text.replace(
            "approval:\n  required_for_writes: false",
            "approval:\n  required_for_writes: true",
        )
    config.write_text(text, encoding="utf-8")


def _git_init_and_seed(workspace: Path, fixture: Path) -> None:
    """Initialise the wiki as a git repo and replay the SEED.md if present.

    The fixture may include a ``SEED.md`` describing per-file commit
    history (one commit per stanza, ordered). When absent, a single
    "seed" commit captures the whole tree.

    Every subprocess invocation surfaces stderr in the exception
    message on failure — the default ``CalledProcessError`` shows only
    the exit code, which is useless for debugging seed problems on
    operator machines (signing-key configs, hook scripts, etc.).
    """
    import subprocess

    def git(*args: str) -> None:
        proc = subprocess.run(
            ["git", "-c", "commit.gpgsign=false", *args],
            cwd=workspace,
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"git seed step failed (cwd={workspace}):\n"
                f"  command: {' '.join(['git', *args])}\n"
                f"  exit: {proc.returncode}\n"
                f"  stdout: {proc.stdout.strip() or '(empty)'}\n"
                f"  stderr: {proc.stderr.strip() or '(empty)'}"
            )

    git("init", "--initial-branch", "main")
    git("config", "user.name", "seed")
    git("config", "user.email", "seed@example.com")

    seed_file = fixture / "SEED.md"
    if not seed_file.exists():
        git("add", "-A")
        git("commit", "-m", "seed: fixture import")
        return

    # Stanza format: "## <author>|<email>|<subject>" then a list of
    # file paths to stage (one per line). Files are taken from the
    # ALREADY MATERIALISED workspace tree — i.e. SEED.md just dictates
    # the commit ordering and authorship over the copied content.
    stanzas = _parse_seed_stanzas(seed_file.read_text(encoding="utf-8"))
    for author, email, subject, paths in stanzas:
        for p in paths:
            if (workspace / p).exists():
                git("add", "--", p)
        git(
            "-c",
            f"user.name={author}",
            "-c",
            f"user.email={email}",
            "commit",
            "-m",
            subject,
        )

    # If anything's left unstaged after the scripted stanzas, sweep it
    # into a final seed commit so the working tree is clean.
    status_proc = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=False,
    )
    if status_proc.returncode != 0:
        raise RuntimeError(
            f"git status failed in seed (cwd={workspace}): "
            f"{status_proc.stderr.strip()}"
        )
    if status_proc.stdout.strip():
        git("add", "-A")
        git("commit", "-m", "seed: remainder")


def _parse_seed_stanzas(text: str) -> list[tuple[str, str, str, list[str]]]:
    stanzas: list[tuple[str, str, str, list[str]]] = []
    current: tuple[str, str, str] | None = None
    paths: list[str] = []
    for raw in text.splitlines():
        line = raw.rstrip()
        if line.startswith("## "):
            if current is not None:
                stanzas.append((*current, paths))
            header = line[3:].split("|")
            if len(header) != 3:
                raise ValueError(
                    f"SEED.md stanza header must be `## author|email|subject`, "
                    f"got {line!r}"
                )
            current = (header[0].strip(), header[1].strip(), header[2].strip())
            paths = []
        elif line.strip() and not line.startswith("#"):
            paths.append(line.strip())
    if current is not None:
        stanzas.append((*current, paths))
    return stanzas
