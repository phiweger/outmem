"""Refusing a page body that stops early — the write-time half.

Lint (see ``test_lint.py``) finds cuts already on disk; this refuses new
ones at the moment they are written, which is the only point where
recovery is cheap: the model still has the source in context. It is also
the first place outmem hands an error *back* to the model rather than
returning it as advisory text after the call already succeeded.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from outmem.exceptions import IncompleteBodyError
from outmem.store import WikiStore

TRUNCATED = "## Diagnostik\n\nDie PCR ist Methode der Wahl. […]\n"
QUOTED = 'Die Leitlinie sagt: "die Therapie [...] wird empfohlen".\n'


@pytest.fixture
def store(tmp_path: Path) -> WikiStore:
    s = WikiStore.init(tmp_path / "w")
    s.write_page("clinical:sepsis", title="Sepsis", body="## Erreger\n\nText.")
    return s


class TestStoreRefuses:
    """At the store layer, so the CLI, the Python API, and a downstream
    app driving its own agent are all covered — not just outmem's tools."""

    def test_write_page(self, store: WikiStore) -> None:
        with pytest.raises(IncompleteBodyError, match="elision marker"):
            store.write_page("clinical:neu", title="Neu", body=TRUNCATED)

    def test_extend_page(self, store: WikiStore) -> None:
        with pytest.raises(IncompleteBodyError):
            store.extend_page("clinical:sepsis", body=TRUNCATED)

    def test_append_page(self, store: WikiStore) -> None:
        with pytest.raises(IncompleteBodyError):
            store.append_page("clinical:sepsis", body=TRUNCATED)

    def test_message_points_at_append_page(self, store: WikiStore) -> None:
        """The ban is only safe because it names the alternative. Without
        somewhere to put the overflow, refusing the marker just pushes the
        model into truncating silently instead."""
        with pytest.raises(IncompleteBodyError) as caught:
            store.write_page("clinical:neu", title="Neu", body=TRUNCATED)
        assert "append_page" in str(caught.value)

    def test_message_quotes_the_offending_line(self, store: WikiStore) -> None:
        with pytest.raises(IncompleteBodyError) as caught:
            store.write_page("clinical:neu", title="Neu", body=TRUNCATED)
        assert "Die PCR ist Methode der Wahl." in str(caught.value)
        assert caught.value.markers

    def test_nothing_is_written_when_refused(self, store: WikiStore) -> None:
        """The guard runs before any disk write, so a refusal cannot leave
        a half-written page or a stray commit behind."""
        head_before = store.head()
        with pytest.raises(IncompleteBodyError):
            store.write_page("clinical:neu", title="Neu", body=TRUNCATED)
        assert not (store.pages_path / "clinical" / "neu.md").exists()
        assert store.head() == head_before

    def test_quotation_is_allowed_through(self, store: WikiStore) -> None:
        store.write_page("clinical:leitlinie", title="LL", body=QUOTED)
        assert "[...]" in store.read("clinical:leitlinie").body


class TestEscapeHatchIsHumanOnly:
    """A human writing an unusual page needs a way through; a model under
    budget pressure must not have one, or the guard becomes a checkbox."""

    def test_store_accepts_allow_elision(self, store: WikiStore) -> None:
        store.write_page(
            "clinical:neu", title="Neu", body=TRUNCATED, allow_elision=True
        )
        assert "[…]" in store.read("clinical:neu").body

    def test_extend_and_append_accept_it_too(self, store: WikiStore) -> None:
        store.extend_page("clinical:sepsis", body=TRUNCATED, allow_elision=True)
        store.append_page("clinical:sepsis", body=TRUNCATED, allow_elision=True)

    def test_tools_do_not_expose_it(self, store: WikiStore) -> None:
        import inspect

        from outmem.adapters.pydantic_ai import wiki_tools

        tools = {t.__name__: t for t in wiki_tools(store)}
        for name in ("write_page", "extend_page", "append_page"):
            params = inspect.signature(tools[name]).parameters
            assert "allow_elision" not in params, name


class TestToolRaisesModelRetry:
    """Returned as a string, an error arrives as commentary *after* the
    call succeeded. ModelRetry is the only path that gets the model to
    write the body again."""

    def _tool(self, store: WikiStore, name: str):  # type: ignore[no-untyped-def]
        from outmem.adapters.pydantic_ai import wiki_tools

        return next(t for t in wiki_tools(store) if t.__name__ == name)

    @pytest.mark.parametrize(
        ("name", "kwargs"),
        [
            ("write_page", {"slug": "clinical:neu", "title": "Neu"}),
            ("extend_page", {"slug": "clinical:sepsis"}),
            ("append_page", {"slug": "clinical:sepsis"}),
        ],
    )
    def test_raises_rather_than_returning_a_string(
        self, store: WikiStore, name: str, kwargs: dict[str, str]
    ) -> None:
        from pydantic_ai import ModelRetry

        with pytest.raises(ModelRetry) as caught:
            self._tool(store, name)(body=TRUNCATED, **kwargs)
        assert "append_page" in str(caught.value)

    def test_insisting_on_the_same_body_is_let_through(
        self, store: WikiStore
    ) -> None:
        """The guard is a fallible heuristic. Without a yield, a false
        positive costs the WHOLE turn: the model re-sends the same correct
        body until the retry budget is gone and the run dies with zero
        commits. A second identical submission is accepted and left for
        lint to report at WARNING."""
        from pydantic_ai import ModelRetry

        tool = self._tool(store, "write_page")
        kwargs = {"slug": "clinical:neu", "title": "Neu", "body": TRUNCATED}
        with pytest.raises(ModelRetry):
            tool(**kwargs)
        sha = tool(**kwargs)  # same body again → accepted
        assert isinstance(sha, str)
        assert "[…]" in store.read("clinical:neu").body

    def test_the_yield_is_per_body_not_per_tool(self, store: WikiStore) -> None:
        """Insisting on one body must not pre-authorise a different one —
        otherwise one false positive disables the guard for the turn."""
        from pydantic_ai import ModelRetry

        tool = self._tool(store, "write_page")
        with pytest.raises(ModelRetry):
            tool(slug="clinical:a", title="A", body=TRUNCATED)
        tool(slug="clinical:a", title="A", body=TRUNCATED)
        with pytest.raises(ModelRetry):
            tool(slug="clinical:b", title="B", body="Anderer Text. […]\n")

    def test_a_yielded_page_is_still_reported_by_lint(
        self, store: WikiStore
    ) -> None:
        """The bounded-damage half of the bargain: the write goes through,
        and the page is visible as `truncated-page` rather than silent."""
        from pydantic_ai import ModelRetry

        from outmem.lint import lint_wiki

        tool = self._tool(store, "write_page")
        kwargs = {"slug": "clinical:neu", "title": "Neu", "body": TRUNCATED}
        with pytest.raises(ModelRetry):
            tool(**kwargs)
        tool(**kwargs)
        report = lint_wiki(store.wiki_path, log_dir=store.log_path)
        assert [f for f in report.findings if f.kind == "truncated-page"]

    def test_message_offers_the_honest_escape(self, store: WikiStore) -> None:
        """A truncating model adds content; a model holding a correct
        quotation re-sends it. The message has to name that second path,
        or the model keeps 'fixing' text that was never broken."""
        with pytest.raises(IncompleteBodyError) as caught:
            store.write_page("clinical:neu", title="Neu", body=TRUNCATED)
        assert "same body again unchanged" in str(caught.value)

    def test_other_failures_still_return_strings(self, store: WikiStore) -> None:
        """Only the incomplete-body case is retryable; a bad slug is the
        model's mistake to read and correct, not to re-attempt blindly."""
        out = self._tool(store, "write_page")(
            slug="Not A Slug", title="X", body="Fine.\n"
        )
        assert isinstance(out, str)
        assert "invalid slug" in out


SENTINEL_BODY = (
    "## Erreger\n\n"
    "⟪ outmem: source truncated — 200000 of 512345 chars shown ⟫\n"
)


class TestToolSentinelIsRefusedAtWriteTime:
    """The same defect one step earlier: outmem withheld content from a
    tool result and the page was written from what was shown anyway.
    Caught here for the same reason as the elision — while the source is
    still in context, which lint (minutes to days later) is not."""

    def test_write_page_refuses_it(self, store: WikiStore) -> None:
        with pytest.raises(IncompleteBodyError, match="tool-output marker"):
            store.write_page("clinical:neu", title="Neu", body=SENTINEL_BODY)

    def test_the_message_says_to_read_the_source(self, store: WikiStore) -> None:
        with pytest.raises(IncompleteBodyError) as caught:
            store.write_page("clinical:neu", title="Neu", body=SENTINEL_BODY)
        assert "read_source" in str(caught.value)

    def test_allow_elision_overrides_it_too(self, store: WikiStore) -> None:
        store.write_page(
            "clinical:neu", title="Neu", body=SENTINEL_BODY, allow_elision=True
        )


class TestHumanAdjudicationIsHonoured:
    """The guard exists to catch a model truncating under budget pressure,
    not to overrule the person the approval gate was installed for."""

    def test_a_reviewed_body_passes(self, store: WikiStore) -> None:
        """Under the HITL gate the reviewer can edit the body; the edited
        text is re-run through the tool. Without registering it, the
        human's deliberate edit is bounced to the MODEL to rewrite."""
        from pydantic_ai.tools import ToolApproved

        from outmem.agent.approval import _allow_reviewed_body

        _allow_reviewed_body(store, ToolApproved(override_args={"body": TRUNCATED}))
        store.write_page("clinical:neu", title="Neu", body=TRUNCATED)
        assert "[…]" in store.read("clinical:neu").body

    def test_a_plain_approval_registers_nothing(self, store: WikiStore) -> None:
        """Approving the model's own body unchanged is not an adjudication
        of the text — only an edit is."""
        from pydantic_ai.tools import ToolApproved

        from outmem.agent.approval import _allow_reviewed_body

        _allow_reviewed_body(store, ToolApproved())
        with pytest.raises(IncompleteBodyError):
            store.write_page("clinical:neu", title="Neu", body=TRUNCATED)

    def test_allowance_is_by_text_not_slug(self, store: WikiStore) -> None:
        """An adjudication is about the words. Keying on the slug would
        make the same decision have to be taken again per page — and
        would break across an alias round-trip."""
        store.allow_elision_body(TRUNCATED)
        store.write_page("clinical:a", title="A", body=TRUNCATED)
        store.write_page("clinical:b", title="B", body=TRUNCATED)


def test_cli_offers_an_escape(tmp_path: Path) -> None:
    """The refusal message promises one; through the CLI it has to exist."""
    from outmem.cli.__main__ import main

    root = tmp_path / "w"
    WikiStore.init(root).close()
    import sys
    from io import StringIO

    sys.stdin = StringIO("Die Punkte sind: [...]\n")
    try:
        rc = main(["write", "q", "--title", "X", "--allow-elision", "--root", str(root)])
    finally:
        sys.stdin = sys.__stdin__
    assert rc == 0
    assert "[...]" in WikiStore.open(root).read("q").body
