"""Orchestrate one full agent run: pull → run → enforce → push → record.

This is where the mandatory-writeback contract is enforced (spec v0.5
§9). The contract has two halves:

1. *The agent must commit at least once per turn.* We count commits
   authored by the agent identity in ``head_before..head_after`` —
   not just ``head_before != head_after``, which a concurrent pull
   could move without the agent contributing anything. If the count is
   zero we raise :class:`WritebackError`.

2. *The writeback must reach the remote.* If push is rejected we
   pull-rebase once and retry; a second failure raises
   :class:`WritebackError`. When the retry succeeds we flag
   ``concurrent_human_commit_landed=True`` on :class:`AskResult` so
   the caller knows the agent's commit raced with another author —
   spec §9 says the agent should re-read the affected file in that
   case, which v0.1 surfaces as a flag rather than auto-re-runs.

The service is offline-tolerant for ``pull``: a misconfigured or
unreachable remote does not block the agent's retrieval. The push
step is the strict one — any push failure that survives one retry is
a hard error.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from outmem._time import utc_now
from outmem.agent.approval import Reviewer, apply_verdicts
from outmem.agent.runtime import build_agent
from outmem.exceptions import OutmemError, WritebackError, format_validation_detail
from outmem.git_ops import CommitInfo, has_remote, head_or_none, log_range
from outmem.store import WikiStore

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TokenUsage:
    """Token totals for one agent run.

    ``cache_read`` is the count of tokens served from Anthropic's prompt
    cache (billed at ~10% of normal input). High ``cache_read`` vs
    ``input`` is the sign that prompt caching is doing its job.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


@dataclass(frozen=True)
class AskResult:
    """The outcome of one full ask cycle."""

    response: str
    head_before: str | None
    head_after: str | None
    commits: tuple[CommitInfo, ...]
    pushed: bool
    concurrent_human_commit_landed: bool
    started_at: datetime
    finished_at: datetime
    usage: TokenUsage = field(default_factory=TokenUsage)

    @property
    def wrote_back(self) -> bool:
        return bool(self.commits)

    @property
    def commit_shas(self) -> tuple[str, ...]:
        """SHAs of the agent-authored commits produced this turn."""
        return tuple(c.sha for c in self.commits)

    @property
    def commit_subjects(self) -> tuple[str, ...]:
        """Subject lines of the agent-authored commits produced this turn."""
        return tuple(c.subject for c in self.commits)


@dataclass(frozen=True)
class _PushOutcome:
    """Result of :func:`_push_with_retry`."""

    pushed: bool
    concurrent_human_commit_landed: bool


@dataclass
class _RunOptions:
    """Internal options pinned per call."""

    push: bool = True
    pull: bool = True
    record: bool = True
    agent_kwargs: dict[str, Any] = field(default_factory=dict)


async def ask(
    store: WikiStore,
    *,
    query: str,
    model: Any | None = None,
    push: bool = True,
    pull: bool = True,
    record: bool = True,
    reviewer: Reviewer | None = None,
    **agent_kwargs: Any,
) -> AskResult:
    """One full orient → retrieve → compact cycle for ``query``.

    Args:
        store: The wiki to operate on.
        query: The user prompt.
        model: PydanticAI model id, ``Model`` instance, or ``None`` to
            read ``$OUTMEM_MODEL``.
        push: ``False`` for offline runs / tests — commits stay local.
        pull: ``False`` to skip the pre-run ``git pull --rebase``.
        record: ``False`` to skip stamping the last-run marker.
        reviewer: A :class:`outmem.agent.approval.Reviewer` that
            decides on each gated tool call when
            ``approval.required_for_writes`` is enabled. Required when
            that flag is on; ignored otherwise.
        agent_kwargs: Extra keyword arguments forwarded to
            :func:`outmem.agent.runtime.build_agent`.

    Raises:
        WritebackError: The agent skipped its writeback, or push failed
            after a single retry.
        OutmemError: Approval is required but no reviewer was supplied.
    """
    opts = _RunOptions(push=push, pull=pull, record=record, agent_kwargs=dict(agent_kwargs))
    started = utc_now()

    # Library entry point: honour `logfire.project` from config.yaml the
    # same way the CLI's `_open_store` does. Idempotent — re-calls from
    # subsequent ask() invocations in the same process are no-ops.
    from outmem._logfire import setup as _setup_logfire
    _setup_logfire(store.config.outmem.logfire)

    if store.config.outmem.approval.required_for_writes and reviewer is None:
        raise OutmemError(
            "approval.required_for_writes is on in config.yaml but no "
            "reviewer was provided to ask(...). Pass `reviewer=CliReviewer()` "
            "(see outmem.agent.approval)."
        )

    if opts.pull:
        # Best-effort: a misconfigured / unreachable remote should not
        # block local retrieval. Push is the strict step.
        with suppress(OutmemError):
            store.pull()

    head_before = head_or_none(store.root)

    agent = build_agent(store, model=model, **opts.agent_kwargs)
    try:
        result = await agent.run(query)

        # Approval gate: if any write tool call needs human review, the
        # output will be a DeferredToolRequests. Loop until the agent
        # produces a final string answer (or a deny chain that ends with
        # an append_log + a string answer).
        if reviewer is not None:
            from pydantic_ai.tools import DeferredToolRequests

            while isinstance(result.output, DeferredToolRequests):
                deferred_results = apply_verdicts(reviewer, result.output)
                result = await agent.run(
                    message_history=result.all_messages(),
                    deferred_tool_results=deferred_results,
                )
    except Exception as exc:
        # PydanticAI surfaces tool-validation crashes (model called a
        # tool with bad/missing args after exhausting `tool_retries`) as
        # `UnexpectedModelBehavior` with the underlying ValidationError
        # on ``__cause__``. Pull a digest of those validation details
        # into the user-visible message so the failure is debuggable —
        # the bare PydanticAI message ("Tool 'X' exceeded max retries")
        # says nothing about WHAT was wrong with the arguments.
        head_after_failure = head_or_none(store.root)
        partial_commits = _new_agent_commits(head_before, head_after_failure, store)
        detail = format_validation_detail(exc)
        raise WritebackError(
            f"agent run failed mid-flight ({type(exc).__name__}: {exc})"
            f"{detail}. {len(partial_commits)} agent commit(s) landed "
            "before the crash; re-run to continue from there."
        ) from exc

    response_text: str = result.output if isinstance(result.output, str) else str(result.output)
    usage = _extract_usage(result)

    head_after = head_or_none(store.root)
    commits = _new_agent_commits(head_before, head_after, store)

    if not commits:
        raise WritebackError(
            "Agent produced no commits this turn (mandatory writeback failed). "
            "The agent's system prompt should ensure at least one "
            "`write_page`, `extend_page`, or `append_log` call per run."
        )

    push_outcome = _PushOutcome(pushed=False, concurrent_human_commit_landed=False)
    if opts.push:
        push_outcome = _push_with_retry(store)

    if opts.record:
        store.record_run()

    finished = utc_now()
    return AskResult(
        response=response_text,
        usage=usage,
        head_before=head_before,
        head_after=head_after,
        commits=tuple(commits),
        pushed=push_outcome.pushed,
        concurrent_human_commit_landed=push_outcome.concurrent_human_commit_landed,
        started_at=started,
        finished_at=finished,
    )


def ask_sync(
    store: WikiStore,
    *,
    query: str,
    model: Any | None = None,
    push: bool = True,
    pull: bool = True,
    record: bool = True,
    reviewer: Reviewer | None = None,
    **agent_kwargs: Any,
) -> AskResult:
    """Blocking wrapper around :func:`ask` for CLI / sync call sites."""
    return asyncio.run(
        ask(
            store,
            query=query,
            model=model,
            push=push,
            pull=pull,
            record=record,
            reviewer=reviewer,
            **agent_kwargs,
        )
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _new_agent_commits(
    head_before: str | None,
    head_after: str | None,
    store: WikiStore,
) -> list[CommitInfo]:
    """Return commits authored by the agent identity between the two HEADs.

    The writeback contract (spec v0.5 §9) requires distinguishing *the
    agent committed* from *something else moved HEAD*. A naive
    ``head_before != head_after`` test would treat any concurrent pull
    or sibling process as a successful agent writeback. Instead we
    filter ``git log`` by the agent's email so only its own commits
    count.

    Returns an empty list when HEAD did not move (or moved only because
    of non-agent commits). The first-run case (``head_before is None``)
    is handled by listing all agent commits reachable from ``head_after``.
    """
    if head_after is None or head_after == head_before:
        return []
    agent_email = store.config.agent_identity.email
    range_spec = head_after if head_before is None else f"{head_before}..{head_after}"
    return log_range(store.root, range_spec=range_spec, author=agent_email)


def _push_with_retry(store: WikiStore) -> _PushOutcome:
    """Push, retry once on rejection via pull-rebase, hard-error on failure.

    Returns a :class:`_PushOutcome` describing what happened.

    Local-only wikis (no configured remote) are detected up-front via
    :func:`outmem.git_ops.has_remote` and the push step is skipped —
    the local commits are the writeback in that case, so failing the
    run because there's no remote to push to would be wrong.

    The retry succeeding implies ``git pull --rebase`` pulled in
    concurrent commits — that's surfaced as
    ``concurrent_human_commit_landed=True`` so the caller can warn the
    user that the writeback raced with another author (spec §9:
    re-read affected file).
    """
    if not has_remote(store.root, remote=store.config.remote):
        log.info(
            "no '%s' remote configured; local commits are the writeback",
            store.config.remote,
        )
        return _PushOutcome(pushed=False, concurrent_human_commit_landed=False)
    try:
        store.push()
        return _PushOutcome(pushed=True, concurrent_human_commit_landed=False)
    except OutmemError as first:
        log.warning("push rejected; pull-rebase and retry: %s", first)
        try:
            store.pull()
            store.push()
            return _PushOutcome(pushed=True, concurrent_human_commit_landed=True)
        except OutmemError as second:
            raise WritebackError(f"push failed after one pull-rebase retry: {second}") from second


def _extract_usage(run_result: Any) -> TokenUsage:
    """Pull PydanticAI's RunUsage into outmem's :class:`TokenUsage`.

    Defensive against API drift / missing fields — any AttributeError
    is treated as "no usage info available" and we return zeros.
    """
    u = getattr(run_result, "usage", None)
    if u is None:
        return TokenUsage()
    if callable(u):  # PydanticAI <2.0 exposed `.usage()` as a method
        try:
            u = u()
        except Exception:
            return TokenUsage()
    return TokenUsage(
        input_tokens=getattr(u, "input_tokens", 0) or 0,
        output_tokens=getattr(u, "output_tokens", 0) or 0,
        cache_read_tokens=getattr(u, "cache_read_tokens", 0) or 0,
        cache_write_tokens=getattr(u, "cache_write_tokens", 0) or 0,
    )
