"""Tests for ``outmem._progress`` — the shared stderr progress counter.

The throttle exists because progress you cannot scroll past is not
progress you can read: a `\\r`-updated counter costs one line on a
terminal however many ticks it has, but captured output got one line
*per tick*, so reindexing a few thousand pages buried everything else in
the log — including the install line above it.
"""

from __future__ import annotations

import pytest

from outmem._progress import report_progress


def _lines(capsys: pytest.CaptureFixture[str]) -> list[str]:
    return [ln for ln in capsys.readouterr().err.splitlines() if ln.strip()]


class TestCapturedOutputIsThrottled:
    def test_a_long_run_does_not_flood_the_log(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        for i in range(1, 1001):
            report_progress(None, i, 1000, label="reindex", unit="pages")
        lines = _lines(capsys)
        assert len(lines) <= 12, f"{len(lines)} lines would drown the log"

    def test_the_first_and_last_tick_always_survive(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The two that carry the information: that it started, and that
        it finished rather than hung."""
        for i in range(1, 1001):
            report_progress(None, i, 1000, label="reindex", unit="pages")
        lines = _lines(capsys)
        assert lines[0] == "reindex: 1/1000 pages"
        assert lines[-1] == "reindex: 1000/1000 pages"

    def test_a_short_run_is_not_throttled_away(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Throttling something already short would lose the signal
        entirely — the step must never exceed one tick."""
        for i in range(1, 6):
            report_progress(None, i, 5, label="bank", unit="pages")
        assert len(_lines(capsys)) == 5

    def test_a_single_item_run_reports_once(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        report_progress(None, 1, 1, label="reindex", unit="pages")
        assert _lines(capsys) == ["reindex: 1/1 pages"]

    def test_an_empty_run_does_not_divide_by_zero(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        report_progress(None, 0, 0, label="reindex", unit="pages")
        assert _lines(capsys) == ["reindex: 0/0 pages"]


class TestCallbackIsUnaffected:
    def test_every_tick_reaches_a_callback(self) -> None:
        """The throttle is about *display*. A caller wiring a progress bar
        or a logger asked for every tick and must still get them."""
        ticks: list[tuple[int, int]] = []
        for i in range(1, 1001):
            report_progress(
                lambda done, total: ticks.append((done, total)), i, 1000, label="x"
            )
        assert len(ticks) == 1000
        assert ticks[0] == (1, 1000) and ticks[-1] == (1000, 1000)

    def test_a_raising_callback_never_breaks_the_operation(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        def boom(done: int, total: int) -> None:
            raise RuntimeError("bar exploded")

        report_progress(boom, 1, 2, label="x")  # must not raise
        assert _lines(capsys) == []  # and must not fall back to stderr
