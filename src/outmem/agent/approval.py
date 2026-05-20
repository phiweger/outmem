"""Human-in-the-loop approval for agent writes.

When ``approval.required_for_writes: true`` in ``config.yaml``, the
agent's ``write_page`` and ``extend_page`` tool calls are deferred:
the underlying git commit does not happen until a :class:`Reviewer`
returns a verdict for each pending call. ``append_log`` and the
read-only tools are not gated, so mandatory writeback (spec §9) still
has a path even after a denial.

The wiring uses PydanticAI's native deferred-tools primitives — see
the docs section on ``ApprovalRequiredToolset`` and
:class:`pydantic_ai.tools.DeferredToolRequests`. We don't re-implement
the pause/resume mechanics here; this module just maps
:class:`pydantic_ai.tools.ToolCallPart` proposals to verdicts.

Verdict shapes:

* :class:`pydantic_ai.tools.ToolApproved` — approve. Optional
  ``override_args`` revises the call (e.g. the reviewer rewrites the
  proposed body in place; the tool then runs with the revised args).
  This *is* the "comment that changes the edit" affordance — no extra
  re-prompt round-trip needed.
* :class:`pydantic_ai.tools.ToolDenied` — deny. ``message`` flows back
  to the model as the tool's result; the agent typically responds with
  an ``append_log`` to still satisfy mandatory writeback.
* The protocol also accepts ``True`` / ``False`` as sugar.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from outmem.exceptions import OutmemError

log = logging.getLogger(__name__)


@runtime_checkable
class Reviewer(Protocol):
    """Per-call verdict producer.

    Implementations return one of:

    * a :class:`pydantic_ai.tools.ToolApproved` (optionally with
      ``override_args``);
    * a :class:`pydantic_ai.tools.ToolDenied` (with a model-visible
      ``message``);
    * the bool sugar ``True`` / ``False``.
    """

    def review(self, call: Any) -> Any:
        """Inspect a single ``ToolCallPart`` and return a verdict."""
        ...


class AutoApproveReviewer:
    """Always approve. Useful for tests and for the rare in-process
    consumer that wants the deferred plumbing but no human gate."""

    def review(self, call: Any) -> Any:
        from pydantic_ai.tools import ToolApproved

        return ToolApproved()


class AutoDenyReviewer:
    """Always deny. The fail-loud default for non-interactive contexts
    when ``approval.required_for_writes`` is on but no reviewer was
    wired up — a CI pipeline that tries to ``outmem ask`` should not
    silently commit.

    Use :class:`OutmemError` from :func:`require_interactive_reviewer`
    if you'd rather abort before the agent run starts.
    """

    def review(self, call: Any) -> Any:
        from pydantic_ai.tools import ToolDenied

        return ToolDenied(
            message=(
                "Write rejected: no interactive reviewer is attached but "
                "`approval.required_for_writes` is enabled. Call `append_log` "
                "instead, or rerun in a terminal that can answer the prompt."
            )
        )


class RecordingReviewer:
    """Programmed verdicts keyed by tool name (FIFO per name).

    Tests build this with ``RecordingReviewer({"write_page": [True],
    "extend_page": [ToolDenied(message="...")]})``. Each ``review``
    call consumes the next verdict for the matching tool name; running
    out raises so tests catch missing verdicts.
    """

    def __init__(self, verdicts: dict[str, list[Any]]) -> None:
        self._verdicts = {k: list(v) for k, v in verdicts.items()}
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def review(self, call: Any) -> Any:
        name = call.tool_name
        # ToolCallPart.args is a JSON string or dict depending on the
        # provider; normalise to a dict for the test-side audit log.
        args = call.args_as_dict() if hasattr(call, "args_as_dict") else call.args
        self.calls.append((name, dict(args) if isinstance(args, dict) else {}))
        queue = self._verdicts.get(name)
        if not queue:
            raise AssertionError(
                f"RecordingReviewer ran out of verdicts for tool {name!r}; "
                f"got args={args!r}"
            )
        return queue.pop(0)


class CliReviewer:
    """Interactive stdin/stdout reviewer for ``outmem ask``.

    For each deferred call, prints the tool name + arguments and asks::

        [a]pprove  [d]eny  [e]dit-body-and-approve

    The "edit" path runs the proposed body through ``$VISUAL`` /
    ``$EDITOR`` (falling back to ``vi``) so the reviewer can revise it
    in place. The resulting body is sent back via
    :class:`ToolApproved` with ``override_args={"body": ...}``.
    """

    def __init__(
        self,
        *,
        stream: Any | None = None,
        input_fn: Callable[[str], str] | None = None,
        edit_fn: Callable[[str], str] | None = None,
    ) -> None:
        self._out = stream if stream is not None else sys.stdout
        self._input = input_fn if input_fn is not None else input
        self._edit = edit_fn if edit_fn is not None else _edit_via_external_editor

    def review(self, call: Any) -> Any:
        from pydantic_ai.tools import ToolApproved, ToolDenied

        args = call.args_as_dict() if hasattr(call, "args_as_dict") else call.args
        args_dict: dict[str, Any] = dict(args) if isinstance(args, dict) else {}

        self._render_proposal(call.tool_name, args_dict)
        while True:
            verdict = self._input(
                "  [a]pprove  [d]eny  [e]dit body and approve  > "
            ).strip().lower()
            if verdict in ("a", "approve", "y", "yes"):
                return ToolApproved()
            if verdict in ("d", "deny", "n", "no"):
                reason = self._input("  reason for denial: ").strip()
                return ToolDenied(
                    message=reason or "The reviewer denied this write."
                )
            if verdict in ("e", "edit"):
                old_body = args_dict.get("body", "")
                if not isinstance(old_body, str):
                    self._writeln(
                        "  (no `body` argument to edit; pick a or d)"
                    )
                    continue
                new_body = self._edit(old_body)
                if new_body == old_body:
                    self._writeln("  (no change made; falling back to approve)")
                return ToolApproved(override_args={**args_dict, "body": new_body})
            self._writeln(f"  unrecognised choice {verdict!r}; pick a, d, or e.")

    def _render_proposal(self, name: str, args: dict[str, Any]) -> None:
        self._writeln(f"\nPending write — {name}:")
        for key, value in args.items():
            if key == "body" and isinstance(value, str):
                preview = value if len(value) <= 400 else value[:397] + "…"
                self._writeln(f"  body ({len(value)} chars):")
                for line in preview.splitlines() or [""]:
                    self._writeln(f"  | {line}")
            else:
                self._writeln(f"  {key}: {value!r}")

    def _writeln(self, line: str) -> None:
        self._out.write(line + "\n")
        self._out.flush()


def require_interactive_reviewer(approval_required: bool) -> Reviewer | None:
    """CLI helper: pick the right reviewer for ``outmem ask`` / ``ingest``.

    Returns:
    * ``None`` when ``approval_required`` is ``False`` — caller passes
      no reviewer and the agent runs without the deferred-tool gate.
    * :class:`CliReviewer` when a tty is attached.

    Raises :class:`OutmemError` when approval is required but stdin is
    not a tty — CI / batch contexts must opt out of the gate explicitly
    rather than silently committing or silently failing forever.
    """
    if not approval_required:
        return None
    if not sys.stdin.isatty():
        raise OutmemError(
            "approval.required_for_writes is on but stdin is not a tty — "
            "no interactive reviewer can run. Either disable the flag in "
            "config.yaml for non-interactive runs, or use a wrapper that "
            "supplies a custom Reviewer to `ask_sync(...)`."
        )
    return CliReviewer()


def apply_verdicts(reviewer: Reviewer, requests: Any) -> Any:
    """Walk ``requests.approvals`` and build a ``DeferredToolResults``.

    Logs a one-line trace per call so the operator can see what the
    reviewer decided. Tests can substitute their own logger; the
    default goes to ``outmem.agent.approval``.
    """
    approvals: dict[str, Any] = {}
    for call in requests.approvals:
        verdict = reviewer.review(call)
        approvals[call.tool_call_id] = verdict
        log.info("review %s → %s", call.tool_name, _summarise_verdict(verdict))
    return requests.build_results(approvals=approvals)


def _summarise_verdict(verdict: Any) -> str:
    from pydantic_ai.tools import ToolApproved, ToolDenied

    if verdict is True:
        return "approve"
    if verdict is False:
        return "deny"
    if isinstance(verdict, ToolApproved):
        return "approve(override_args)" if verdict.override_args else "approve"
    if isinstance(verdict, ToolDenied):
        return f"deny({verdict.message[:60]!r})"
    return repr(verdict)


def _edit_via_external_editor(initial: str) -> str:
    """Open ``$VISUAL`` or ``$EDITOR`` on a tempfile seeded with ``initial``.

    Falls back to ``vi`` if neither env var is set. Returns the user's
    edited content; an empty result is treated as "no edit" by the
    caller. If the editor exits non-zero (e.g. ``:cq`` in vim), the
    initial content is returned unchanged.
    """
    import contextlib
    import os
    import subprocess
    import tempfile

    editor = os.environ.get("VISUAL") or os.environ.get("EDITOR") or "vi"
    with tempfile.NamedTemporaryFile(
        mode="w+", suffix=".md", encoding="utf-8", delete=False
    ) as tmp:
        tmp.write(initial)
        tmp_path = tmp.name
    try:
        rc = subprocess.call([editor, tmp_path])
        if rc != 0:
            log.warning("editor %r exited non-zero (%d); keeping original body", editor, rc)
            return initial
        with open(tmp_path, encoding="utf-8") as fh:
            return fh.read()
    finally:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
