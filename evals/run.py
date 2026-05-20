"""CLI entry point for the eval suite.

Usage::

    python -m evals.run                        # all cases, trace + judge
    python -m evals.run --no-judge             # trace-only (still calls agent's LLM)
    python -m evals.run --case duplicate-trap  # one case by name
    python -m evals.run --case duplicate-trap --case approval-fallback
    python -m evals.run --model anthropic:claude-haiku-4-5
    python -m evals.run --judge-model anthropic:claude-sonnet-4-6
    python -m evals.run --json out/evals.json

Exit code = number of failing cases (0 on full pass).
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

from evals.harness import (
    EvalCase,
    EvalRun,
    registered_cases,
    run_case,
    set_judge_enabled,
)


def _load_cases() -> list[EvalCase]:
    # Importing ``evals.cases`` triggers each case module's
    # ``@eval_case`` decorators, populating the registry.
    import evals.cases  # noqa: F401 — side-effect import

    return registered_cases()


def _format_report(runs: list[EvalRun], *, errored: int = 0) -> str:
    lines: list[str] = ["", "== outmem evals =="]
    for r in runs:
        if r.skipped:
            label = "SKIP"
        elif r.passed:
            label = "PASS"
        else:
            label = "FAIL"
        lines.append(f"\n[{label}] {r.case.name}")
        if r.case.description:
            lines.append(f"  -- {r.case.description}")
        if r.skipped:
            lines.append(f"  -- skipped: {r.skip_reason}")
        for a in r.assertions:
            tick = "✓" if a.passed else "✗"
            kind = "trace" if a.kind == "trace" else "judge"
            lines.append(f"  {tick} {kind}: {a.description}")
            if a.detail and (not a.passed or a.detail.startswith("[skipped")):
                lines.append(f"     {a.detail}")
        lines.append(
            f"  ({len(r.tool_calls)} tool calls, "
            f"{len(r.commits)} commit(s), "
            f"{r.duration_s:.1f}s)"
        )

    total = len(runs) + errored
    passed = sum(1 for r in runs if not r.skipped and r.passed)
    failed = sum(1 for r in runs if not r.skipped and not r.passed)
    skipped = sum(1 for r in runs if r.skipped)
    parts = [f"{passed} passed"]
    if failed:
        parts.append(f"{failed} failed")
    if skipped:
        parts.append(f"{skipped} skipped")
    if errored:
        parts.append(f"{errored} errored")
    lines.append(
        f"\nTotal: {', '.join(parts)} (of {total}). "
        f"Cumulative wall time: {sum(r.duration_s for r in runs):.1f}s."
    )
    return "\n".join(lines)


def _json_report(runs: list[EvalRun]) -> str:
    payload = []
    for r in runs:
        payload.append(
            {
                "case": r.case.name,
                "description": r.case.description,
                "passed": r.passed,
                "duration_s": r.duration_s,
                "tool_calls": [asdict(c) for c in r.tool_calls],
                "commits": [
                    {"sha": c.sha, "subject": c.subject}
                    for c in r.commits
                ],
                "assertions": [asdict(a) for a in r.assertions],
                "response": r.response,
            }
        )
    return json.dumps(payload, indent=2)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m evals.run",
        description="Run outmem behavioural evaluations.",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Run only the named case(s). Repeat for multiple.",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip the LLM judge; only trace assertions run. The agent "
        "still uses a real model — only the judge step is suppressed.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="PydanticAI model id for the agent under test "
        "(defaults to $OUTMEM_MODEL or the wiki's config.yaml).",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="PydanticAI model id for the LLM judge "
        "(defaults to anthropic:claude-sonnet-4-6).",
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        default=None,
        help="Write a machine-readable report to this path.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress the live progress trace on stderr (per-case "
        "header, [tool] calls, ✓/✗ per assertion). The final report "
        "rendered to stdout is unaffected.",
    )
    args = parser.parse_args(argv)

    # Resolution chain for agent_model:
    #   CLI --model > $OUTMEM_MODEL (consumed inside outmem.agent.runtime)
    #     > repo config.yaml > fixture wiki's own config.yaml > built-in.
    from evals.config import load_repo_config

    repo_cfg = load_repo_config()
    agent_model = args.model
    if agent_model is None and repo_cfg.agent_model is not None:
        # Inject as fallback only — env var (consumed downstream) still
        # has higher precedence because outmem.agent.runtime reads it
        # before this default is consulted.
        agent_model = repo_cfg.agent_model

    if args.judge_model is not None:
        from evals.judges.llm_judge import set_judge_model

        set_judge_model(args.judge_model)
    set_judge_enabled(not args.no_judge)

    cases = _load_cases()
    if args.case:
        wanted = set(args.case)
        cases = [c for c in cases if c.name in wanted]
        missing = wanted - {c.name for c in cases}
        if missing:
            print(
                f"unknown case(s): {sorted(missing)}\n"
                f"available: {sorted(c.name for c in registered_cases())}",
                file=sys.stderr,
            )
            return 2

    if not cases:
        print("no cases to run", file=sys.stderr)
        return 2

    progress = None if args.quiet else sys.stderr

    runs: list[EvalRun] = []
    errored = 0
    for case in cases:
        with tempfile.TemporaryDirectory(prefix=f"eval-{case.name}-") as tmp:
            workspace = Path(tmp) / "wiki"
            workspace.mkdir()
            try:
                run = run_case(
                    case,
                    workspace=workspace,
                    model=agent_model,
                    progress=progress,
                )
            except Exception as exc:  # case crashed before assertions ran
                print(
                    f"[ERROR] {case.name}: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                errored += 1
                continue
            runs.append(run)

    print(_format_report(runs, errored=errored))

    if args.json_path:
        Path(args.json_path).write_text(_json_report(runs), encoding="utf-8")

    # Failing cases AND errored cases count toward the exit code; a
    # skipped case (missing credential) does NOT — that's a config
    # issue, not a regression.
    failed = sum(1 for r in runs if not r.skipped and not r.passed)
    return failed + errored


if __name__ == "__main__":
    sys.exit(main())
