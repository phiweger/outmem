"""Tests for the bundled skills under ``src/outmem/skills/notes/``.

These are content tests, not behavioural — they pin the skill
inventory so accidental renames break loudly and validate that every
SKILL.md has the metadata the runtime's prompt-renderer expects.

The loader / manifest logic is owned by the
`outskilled <https://github.com/phiweger/outskilled>`_ package and
isn't tested here; only the bundled CONTENT is.
"""

from __future__ import annotations

import pytest
from outskilled import SkillRegistry, parse_frontmatter

from outmem.skills import BUNDLED_SKILLS_DIR, bundled_registry

EXPECTED_CATEGORIES = {"notes"}
EXPECTED_SKILLS = {
    ("notes", "search"),
    ("notes", "evolution"),
    ("notes", "write"),
}
EXPECTED_SKILL_NAMES = {s for _, s in EXPECTED_SKILLS}


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def test_skills_dir_exists() -> None:
    assert BUNDLED_SKILLS_DIR.is_dir()


def test_expected_skills_present() -> None:
    found: set[tuple[str, str]] = set()
    for category in BUNDLED_SKILLS_DIR.iterdir():
        if not category.is_dir() or category.name.startswith("."):
            continue
        for skill in category.iterdir():
            if (skill / "SKILL.md").exists():
                found.add((category.name, skill.name))
    assert found == EXPECTED_SKILLS


def test_categories_match_expected() -> None:
    categories = {
        c.name for c in BUNDLED_SKILLS_DIR.iterdir() if c.is_dir() and not c.name.startswith(".")
    }
    assert categories == EXPECTED_CATEGORIES


# ---------------------------------------------------------------------------
# Frontmatter validity (Anthropic skill schema)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("category", "skill"), sorted(EXPECTED_SKILLS))
def test_skill_frontmatter_required_fields(category: str, skill: str) -> None:
    text = (BUNDLED_SKILLS_DIR / category / skill / "SKILL.md").read_text()
    fm, _ = parse_frontmatter(text)
    assert fm.get("name") == skill, f"{category}/{skill}: name must match folder"
    description = fm.get("description", "")
    assert isinstance(description, str)
    assert description.strip(), f"{category}/{skill}: description must be non-empty"


@pytest.mark.parametrize(("category", "skill"), sorted(EXPECTED_SKILLS))
def test_skill_name_kebab_constraints(category: str, skill: str) -> None:
    """Skill names must be lowercase + hyphens + digits, ≤64 chars, and
    must not contain 'claude' or 'anthropic' (Anthropic platform rules)."""
    text = (BUNDLED_SKILLS_DIR / category / skill / "SKILL.md").read_text()
    fm, _ = parse_frontmatter(text)
    name = fm["name"]
    assert isinstance(name, str)
    assert len(name) <= 64
    assert "claude" not in name.lower()
    assert "anthropic" not in name.lower()
    # No leading/trailing hyphens, no consecutive hyphens.
    assert name == name.strip("-")
    assert "--" not in name


@pytest.mark.parametrize(("category", "skill"), sorted(EXPECTED_SKILLS))
def test_skill_description_size(category: str, skill: str) -> None:
    text = (BUNDLED_SKILLS_DIR / category / skill / "SKILL.md").read_text()
    fm, _ = parse_frontmatter(text)
    description = fm["description"]
    assert isinstance(description, str)
    # Anthropic's platform caps description at 1024 chars.
    assert len(description) <= 1024


@pytest.mark.parametrize(("category", "skill"), sorted(EXPECTED_SKILLS))
def test_skill_body_is_substantive(category: str, skill: str) -> None:
    """Avoid shipping placeholder bodies — every skill should give the
    agent something to act on."""
    text = (BUNDLED_SKILLS_DIR / category / skill / "SKILL.md").read_text()
    # Find the body after the closing frontmatter delimiter.
    parts = text.split("---", 2)
    assert len(parts) >= 3
    body = parts[2].strip()
    assert len(body) > 200, f"{category}/{skill}: SKILL.md body looks too short"


# ---------------------------------------------------------------------------
# References folder
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("category", "skill"), sorted(EXPECTED_SKILLS))
def test_skill_has_at_least_one_reference(category: str, skill: str) -> None:
    references = BUNDLED_SKILLS_DIR / category / skill / "references"
    md_files = list(references.glob("*.md")) if references.exists() else []
    assert md_files, f"{category}/{skill}: no references/*.md files"


# ---------------------------------------------------------------------------
# outskilled integration sanity (registry discovers what we expect)
# ---------------------------------------------------------------------------


def test_bundled_registry_discovers_all_skills() -> None:
    registry = bundled_registry()
    assert set(registry.names()) == EXPECTED_SKILL_NAMES


@pytest.mark.parametrize("skill", sorted(EXPECTED_SKILL_NAMES))
def test_registry_load_returns_body_without_frontmatter(skill: str) -> None:
    body = bundled_registry().load(skill)
    # Frontmatter is stripped by outskilled.
    assert not body.lstrip().startswith("---")
    # And the body is non-trivial.
    assert len(body.strip()) > 200


def test_bundled_registry_is_cached() -> None:
    """``bundled_registry`` is memoised so repeated callers don't
    re-walk the directory tree on every system-prompt render."""
    assert bundled_registry() is bundled_registry()


def test_fresh_registry_over_outmem_skills() -> None:
    """Sanity: a hand-constructed `SkillRegistry` over the bundled
    directory finds the same set — verifies the memoised helper isn't
    cooking the books."""
    fresh = SkillRegistry([BUNDLED_SKILLS_DIR])
    assert set(fresh.names()) == EXPECTED_SKILL_NAMES
