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


def test_tags_must_be_strings() -> None:
    text = "---\ntitle: X\nslug: x\ntags: [a, 1, b]\n---\n\nbody\n"
    with pytest.raises(FrontmatterError, match="strings"):
        parse_wiki_page(text)


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
