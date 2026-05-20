"""Tests for ``outmem.identity``."""

from __future__ import annotations

from pathlib import Path

from outmem.identity import Contributor, load_contributors, parse_contributors


def test_parse_single_entry() -> None:
    text = "- Alice Liddell <alice@example.com>\n"
    contributors = parse_contributors(text)
    assert contributors.entries == [
        Contributor(name="Alice Liddell", primary_email="alice@example.com"),
    ]


def test_parse_with_aliases() -> None:
    text = "- Bob Roberts <bob@example.com> [aliases: bob@personal.dev, b.r@x.com]\n"
    contributors = parse_contributors(text)
    bob = contributors.entries[0]
    assert bob.name == "Bob Roberts"
    assert bob.primary_email == "bob@example.com"
    assert bob.aliases == ("bob@personal.dev", "b.r@x.com")


def test_lookup_by_primary_and_alias() -> None:
    text = "- Bob <bob@example.com> [aliases: bob@personal.dev]\n"
    contributors = parse_contributors(text)
    bob = contributors.lookup("bob@example.com")
    assert bob is not None
    assert bob.name == "Bob"
    assert contributors.lookup("bob@personal.dev") is bob


def test_lookup_is_case_insensitive() -> None:
    text = "- Alice <Alice@Example.com>\n"
    contributors = parse_contributors(text)
    assert contributors.lookup("alice@example.com") is not None
    assert contributors.lookup("ALICE@EXAMPLE.COM") is not None


def test_lookup_unknown_returns_none() -> None:
    contributors = parse_contributors("- Alice <alice@example.com>\n")
    assert contributors.lookup("eve@elsewhere.net") is None


def test_lookup_non_string_returns_none() -> None:
    contributors = parse_contributors("- Alice <alice@example.com>\n")
    assert contributors.lookup(None) is None  # type: ignore[arg-type]


def test_freeform_lines_are_ignored() -> None:
    text = (
        "# Team\n"
        "\n"
        "Active contributors:\n"
        "\n"
        "- Alice <alice@example.com>\n"
        "- Bob <bob@example.com>\n"
        "\n"
        "Notes: aliases get added on first commit collision.\n"
    )
    contributors = parse_contributors(text)
    assert [c.primary_email for c in contributors.entries] == [
        "alice@example.com",
        "bob@example.com",
    ]


def test_malformed_lines_are_skipped() -> None:
    text = (
        "- Missing email\n"
        "- <no-name@example.com>\n"
        "- Alice <alice@example.com>\n"
        "not a bullet\n"
        "- Trailing bracket > <bob@example.com>\n"
    )
    contributors = parse_contributors(text)
    emails = [c.primary_email for c in contributors.entries]
    assert "alice@example.com" in emails


def test_agent_identity_recognised() -> None:
    text = "- outmem agent <agent@host>\n- Alice <alice@example.com>\n"
    contributors = parse_contributors(text)
    agent = contributors.lookup("agent@host")
    assert agent is not None
    assert agent.name == "outmem agent"


def test_load_contributors_missing_file(tmp_path: Path) -> None:
    contributors = load_contributors(tmp_path / "CONTRIBUTORS.md")
    assert contributors.entries == []
    assert contributors.lookup("anyone@example.com") is None


def test_load_contributors_reads_file(tmp_path: Path) -> None:
    path = tmp_path / "CONTRIBUTORS.md"
    path.write_text(
        "# Team\n\n- Alice <alice@example.com>\n- Bob <bob@example.com>\n",
        encoding="utf-8",
    )
    contributors = load_contributors(path)
    assert len(contributors.entries) == 2
    assert contributors.lookup("alice@example.com") is not None


def test_all_emails_includes_primary_first() -> None:
    text = "- Bob <bob@example.com> [aliases: alt@x, alt2@y]\n"
    contributors = parse_contributors(text)
    bob = contributors.entries[0]
    assert bob.all_emails == ("bob@example.com", "alt@x", "alt2@y")
