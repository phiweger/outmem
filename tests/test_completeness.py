"""Elision detection — the check that a page is all there.

The design pressure this file pins: the marker vocabulary is shared
between a real defect and a legitimate one. A bracketed ellipsis is both
how a budget-truncated page marks its cut *and* how quoted material is
shortened. A checker that cannot tell them apart is one a team switches
off after the first false positive on a guideline quotation — so the
discriminator (position, not vocabulary) is what actually needs testing.
"""

from __future__ import annotations

import pytest

from outmem.completeness import (
    find_elision_markers,
    find_tool_sentinels,
    tool_note,
)


def _markers(text: str) -> list[str]:
    return [e.marker for e in find_elision_markers(text)]


class TestTruncationIsFlagged:
    """Text that stops at the marker — the shape budget pressure leaves."""

    @pytest.mark.parametrize(
        "body",
        [
            "Die PCR ist Methode der Wahl. […]",
            "Die PCR ist Methode der Wahl. [...]",
            "Die PCR ist Methode der Wahl. [ ... ]",
            "Die wichtigsten Punkte sind: […].",  # a cut wearing a full stop
        ],
    )
    def test_line_terminal_marker(self, body: str) -> None:
        assert len(find_elision_markers(body)) == 1

    def test_marker_alone_on_its_line(self) -> None:
        assert len(find_elision_markers("## Diagnostik\n\n[…]\n\n## Therapie")) == 1

    def test_bare_ellipsis_alone_on_a_line(self) -> None:
        assert len(find_elision_markers("Ein Satz.\n\n…\n")) == 1
        assert len(find_elision_markers("Ein Satz.\n\n...\n")) == 1

    @pytest.mark.parametrize(
        "comment",
        [
            "<!-- truncated -->",
            "<!-- content omitted for brevity -->",
            "<!-- gekürzt -->",
            "<!-- Fortsetzung folgt -->",
        ],
    )
    def test_html_comment_announcing_omission(self, comment: str) -> None:
        """Invisible when rendered, which makes it worse than a visible
        marker, not better — nobody reading the page can see it."""
        assert len(find_elision_markers(f"Ein Absatz.\n{comment}\n")) == 1

    def test_reports_line_and_text(self) -> None:
        (found,) = find_elision_markers("erste Zeile\nzweite Zeile […]\n")
        assert found.line == 2
        assert found.text == "zweite Zeile […]"

    def test_several_cuts_in_one_page(self) -> None:
        body = "## A\nText. […]\n\n## B\nMehr Text. […]\n"
        assert len(find_elision_markers(body)) == 2

    def test_one_finding_per_line(self) -> None:
        """A line with two markers is one cut, not two."""
        assert len(find_elision_markers("Text […] und mehr […]")) == 1


class TestQuotationIsNotFlagged:
    """The false positives that would get this check switched off."""

    def test_elision_inside_running_prose(self) -> None:
        """The standard way to shorten a quotation. Prose continues past
        the marker, so nothing was cut short."""
        assert _markers('Die Leitlinie sagt: "die Therapie [...] wird empfohlen".') == []

    def test_blockquote_is_left_alone(self) -> None:
        """Quoted material that stops at an ellipsis is a quotation the
        author chose to end early — their business, not the linter's."""
        assert _markers('> "Therapie über 7 Tage [...]"\n') == []

    def test_link_display_text(self) -> None:
        """`[…](url)` is markdown syntax, not an elision. Reported from a
        real wiki as the first false positive anyone hits."""
        assert _markers("Siehe [infektions…](http://www.lgl.bayern.de/seite).") == []

    def test_reference_link_display_text(self) -> None:
        assert _markers("Siehe [...][quelle] für Details.") == []

    def test_fenced_code_block(self) -> None:
        body = "Beispiel:\n\n```python\nitems = [...]\n# […]\n```\n\nWeiter im Text.\n"
        assert _markers(body) == []

    def test_tilde_fenced_code_block(self) -> None:
        assert _markers("~~~\nfoo = [...]\n~~~\n") == []

    def test_inline_code_span(self) -> None:
        assert _markers("Der Platzhalter `[...]` steht für die Liste.") == []

    def test_ellipsis_mid_sentence_without_brackets(self) -> None:
        """Bare ellipsis in prose is punctuation, not a cut — only a bare
        ellipsis *alone on its line* is a signal."""
        assert _markers("Er zögerte … und ging.") == []

    def test_clean_page(self) -> None:
        assert _markers("## Diagnostik\n\nDie PCR ist Methode der Wahl.\n") == []

    def test_html_comment_without_omission_vocabulary(self) -> None:
        assert _markers("<!-- reviewed 2026-08 -->\nText.\n") == []


class TestToolSentinel:
    """outmem's own marker for content it withheld from a tool result."""

    def test_note_is_not_itself_an_elision(self) -> None:
        """The point of a distinct sentinel: outmem must not emit the
        vocabulary it rejects, or it teaches the model the pattern."""
        note = tool_note("source truncated — 512345 chars total, 200000 shown")
        assert _markers(note) == []

    def test_sentinel_in_a_page_body_is_found(self) -> None:
        body = f"Ein Absatz.\n{tool_note('source truncated')}\nNoch einer.\n"
        (found,) = find_tool_sentinels(body)
        assert found.line == 2

    def test_clean_page_has_no_sentinels(self) -> None:
        assert find_tool_sentinels("## Diagnostik\n\nText.\n") == []

    @pytest.mark.parametrize(
        "text",
        [
            "source truncated — 200000 of 512345 chars shown (sources.max_chars)",
            "results truncated at the output cap — narrow the pattern",
            "excerpt spliced — middle omitted",
            "page truncated — 7331 more chars not shown",
        ],
    )
    def test_every_note_outmem_emits_survives_its_own_detector(
        self, text: str
    ) -> None:
        """The round trip that keeps the two halves honest. These are the
        real notes from read_source, grep_wiki, the rerank gate, and the
        optimizer's read_page. If any of them read as an elision, outmem
        would be emitting into model context the exact pattern it refuses
        in page bodies — and a page quoting a tool result would be
        unwritable."""
        note = tool_note(text)
        assert find_elision_markers(note) == []
        assert find_elision_markers(f"Ein Absatz.\n\n{note}\n\nNoch einer.\n") == []
