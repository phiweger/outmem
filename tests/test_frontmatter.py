"""Tests for ``outmem.frontmatter``."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from outmem.exceptions import FrontmatterError
from outmem.frontmatter import (
    WikiFrontmatter,
    parse_wiki_page,
    serialize_wiki_page,
    touch_updated,
)


def test_parse_required_fields(sample_page_text: str) -> None:
    fm, body = parse_wiki_page(sample_page_text)
    assert fm.title == "Pricing formula"
    assert fm.slug == "pricing-formula"
    assert fm.provenance == [
        "raw/pricing-deck-2026-Q1.md",
        "raw/acme-msa.md",
    ]
    assert fm.created == datetime(2026, 4, 12, 9, 14, tzinfo=UTC)
    assert fm.updated == datetime(2026, 5, 4, 11, 32, tzinfo=UTC)
    assert fm.tags == ["pricing", "contracts", "finance"]
    assert fm.extra == {}
    assert "The pricing formula" in body


def test_round_trip_preserves_provenance(page_with_rich_provenance: str) -> None:
    """Dict-valued provenance entries propagate verbatim through write/read."""
    fm, body = parse_wiki_page(page_with_rich_provenance)
    rendered = serialize_wiki_page(fm, body)
    fm2, body2 = parse_wiki_page(rendered)

    assert fm2.provenance == fm.provenance
    assert isinstance(fm2.provenance[0], dict)
    assert fm2.provenance[0]["drive_path"] == "/shared/contracts/acme/2026/MSA.pdf"
    assert fm2.provenance[1] == "raw/acme-pricing.md"
    assert body2.strip() == body.strip()


def test_extra_fields_preserved() -> None:
    text = (
        "---\n"
        "title: Notes\n"
        "slug: notes\n"
        "custom_owner: alice\n"
        "ingestion_run: 2026-05-10-001\n"
        "---\n"
        "\n"
        "body\n"
    )
    fm, _ = parse_wiki_page(text)
    assert fm.extra == {"custom_owner": "alice", "ingestion_run": "2026-05-10-001"}

    rendered = serialize_wiki_page(fm, "body\n")
    fm2, _ = parse_wiki_page(rendered)
    assert fm2.extra == fm.extra


def test_missing_frontmatter_raises() -> None:
    with pytest.raises(FrontmatterError, match="missing the YAML frontmatter"):
        parse_wiki_page("No frontmatter here.\n")


def test_missing_title_raises() -> None:
    text = "---\nslug: x\n---\n\nbody\n"
    with pytest.raises(FrontmatterError, match="title"):
        parse_wiki_page(text)


def test_missing_slug_raises() -> None:
    text = "---\ntitle: X\n---\n\nbody\n"
    with pytest.raises(FrontmatterError, match="slug"):
        parse_wiki_page(text)


def test_malformed_yaml_raises() -> None:
    text = "---\ntitle: [unterminated\n---\n\nbody\n"
    with pytest.raises(FrontmatterError, match="failed to parse"):
        parse_wiki_page(text)


def test_provenance_must_be_list() -> None:
    text = "---\ntitle: X\nslug: x\nprovenance: raw/file.md\n---\n\nbody\n"
    with pytest.raises(FrontmatterError, match="provenance"):
        parse_wiki_page(text)


@pytest.mark.parametrize(
    "written",
    [
        "2026",     # a bare year -> int
        "007",      # zero-padded ICD/indicator code -> int 7
        "12:30",    # YAML 1.1 sexagesimal -> int 750
        "010",      # YAML 1.1 octal -> int 8
        "0x1F",     # hex -> int 31
        "1_000",    # underscore separator -> int 1000
        "1.50",     # float -> 1.5, losing the trailing zero
        ".inf",     # float infinity
        "yes",      # bool True
        "no",       # bool False
    ],
)
def test_unquoted_tags_keep_their_authored_text(written: str) -> None:
    """YAML resolves these to non-strings, and every resolution is lossy.
    Rejecting them dropped the whole page from the index; `str()`-ing the
    resolved value would silently rewrite the author's tag (007 -> '7',
    12:30 -> '750'). The tag's original text is recovered instead."""
    text = f"---\ntitle: X\nslug: x\ntags: [a, {written}]\n---\n\nbody\n"
    fm, _ = parse_wiki_page(text)
    assert fm.tags == ["a", written]


def test_unquoted_tags_survive_a_write_round_trip() -> None:
    """The corruption would otherwise be persisted to disk by the next
    write_page/extend_page, permanently destroying the author's value."""
    text = "---\ntitle: X\nslug: x\ntags: [icd, 007, 12:30]\n---\n\nbody\n"
    fm, body = parse_wiki_page(text)
    out = serialize_wiki_page(fm, body)
    assert "'007'" in out and "'12:30'" in out
    assert parse_wiki_page(out)[0].tags == ["icd", "007", "12:30"]


def test_non_scalar_tags_still_rejected() -> None:
    text = "---\ntitle: X\nslug: x\ntags: [a, [nested]]\n---\n\nbody\n"
    with pytest.raises(FrontmatterError, match="Tags must be strings"):
        parse_wiki_page(text)


@pytest.mark.parametrize("bad_date", ["2026-02-30", "2026-13-01", "2026-04-31"])
def test_invalid_calendar_date_raises_frontmatter_error_not_valueerror(
    bad_date: str,
) -> None:
    """PyYAML's timestamp resolver matches the shape, then datetime.date()
    raises a bare ValueError — not a YAMLError. Letting that escape killed
    the whole reindex and blocked writeback, so parse_wiki_page must wrap
    every parse failure as FrontmatterError."""
    text = f"---\ntitle: X\nslug: x\ncreated: {bad_date}\n---\n\nbody\n"
    with pytest.raises(FrontmatterError):
        parse_wiki_page(text)


def test_date_only_timestamp_is_promoted_to_midnight_utc() -> None:
    """PyYAML loads a bare `created: 2026-07-23` as a `date`, which is NOT
    a `datetime`. Rejecting it silently cost imported wikis whole pages."""
    text = "---\ntitle: X\nslug: x\ncreated: 2026-07-23\n---\n\nbody\n"
    fm, _ = parse_wiki_page(text)
    assert fm.created == datetime(2026, 7, 23, 0, 0, 0, tzinfo=UTC)


def test_datetime_iso_with_z_suffix() -> None:
    text = "---\ntitle: X\nslug: x\ncreated: 2026-01-02T03:04:05Z\n---\n\nbody\n"
    fm, _ = parse_wiki_page(text)
    assert fm.created == datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)


def test_datetime_naive_is_assumed_utc() -> None:
    text = "---\ntitle: X\nslug: x\ncreated: 2026-01-02T03:04:05\n---\n\nbody\n"
    fm, _ = parse_wiki_page(text)
    assert fm.created is not None
    assert fm.created.tzinfo == UTC


def test_datetime_invalid_raises() -> None:
    text = "---\ntitle: X\nslug: x\ncreated: yesterday\n---\n\nbody\n"
    with pytest.raises(FrontmatterError, match="ISO-8601"):
        parse_wiki_page(text)


def test_serialise_emits_z_suffix() -> None:
    fm = WikiFrontmatter(
        title="X",
        slug="x",
        created=datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
    )
    rendered = serialize_wiki_page(fm, "body\n")
    assert "created: 2026-01-02T03:04:05Z" in rendered
    assert "+00:00" not in rendered


def test_touch_updated_sets_aware_utc() -> None:
    fm = WikiFrontmatter(title="X", slug="x")
    touch_updated(fm, now=datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC))
    assert fm.updated == datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC)


def test_serialise_omits_empty_optional_fields() -> None:
    fm = WikiFrontmatter(title="X", slug="x")
    rendered = serialize_wiki_page(fm, "body\n")
    assert "provenance" not in rendered
    assert "tags" not in rendered
    assert "created" not in rendered
    assert "updated" not in rendered


class TestRepairWikiPage:
    """`repair_wiki_page` for the imported-data failure mode: a top-level
    scalar value (most often `title:`) contains an unquoted `: ` and the
    file won't parse. The repair single-quotes such values; conservative
    on everything else."""

    def test_repairs_unquoted_colon_space_in_title(self) -> None:
        from outmem.frontmatter import parse_wiki_page, repair_wiki_page

        # Exact shape from the failing wiki: imported title contains "(Teil 1): "
        broken = (
            "---\n"
            "title: Influenza (Teil 1): Erkrankungen durch saisonale Influenza\n"
            "slug: rki:ratgeber:grippe\n"
            "---\n\nbody\n"
        )
        from outmem.exceptions import FrontmatterError

        with pytest.raises(FrontmatterError):
            parse_wiki_page(broken)  # the original failure

        fixed = repair_wiki_page(broken)
        assert fixed is not None
        fm, body = parse_wiki_page(fixed)  # round-trips cleanly
        assert fm.title == "Influenza (Teil 1): Erkrankungen durch saisonale Influenza"
        assert fm.slug == "rki:ratgeber:grippe"
        assert body.strip() == "body"

    def test_returns_none_when_already_parses(self) -> None:
        """A well-formed page is left untouched (repair is opt-in)."""
        from outmem.frontmatter import repair_wiki_page

        ok = "---\ntitle: Fine\nslug: ok\n---\n\nbody\n"
        assert repair_wiki_page(ok) is None

    def test_returns_none_when_no_frontmatter(self) -> None:
        from outmem.frontmatter import repair_wiki_page

        assert repair_wiki_page("just body text\n") is None

    def test_leaves_quoted_values_alone(self) -> None:
        """A value that's already single- or double-quoted is not re-wrapped
        (broken page must contain SOME unquoted ': ' to fire)."""
        from outmem.frontmatter import repair_wiki_page

        # Frontmatter has a bad line AND a correctly-quoted one — the quoted
        # line must come through unmodified.
        broken = (
            "---\n"
            'title: "Already: Quoted"\n'
            "subtitle: Bad: unquoted\n"
            "slug: x\n"
            "---\n\nbody\n"
        )
        fixed = repair_wiki_page(broken)
        assert fixed is not None
        assert 'title: "Already: Quoted"' in fixed  # untouched

    def test_escapes_embedded_single_quotes(self) -> None:
        """Single-quote YAML escape is doubling — verify the round-trip."""
        from outmem.frontmatter import parse_wiki_page, repair_wiki_page

        broken = (
            "---\n"
            "title: Bob's note: a thing\n"
            "slug: x\n"
            "---\n\nbody\n"
        )
        fixed = repair_wiki_page(broken)
        assert fixed is not None
        fm, _ = parse_wiki_page(fixed)
        assert fm.title == "Bob's note: a thing"

    def test_returns_none_when_repair_doesnt_help(self) -> None:
        """An indentation / structural break isn't in scope — the repair
        returns None rather than emitting something that still won't parse."""
        from outmem.frontmatter import repair_wiki_page

        # Mis-indented mapping under a key — repair won't touch this shape.
        broken = (
            "---\n"
            "title: T\n"
            "slug: x\n"
            "tags:\n"
            "[oops, this is wrong, no closing\n"
            "---\n\nbody\n"
        )
        assert repair_wiki_page(broken) is None
