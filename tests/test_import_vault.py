"""Tests for ``outmem import`` — Obsidian-style vault import.

Covers the full flow end-to-end: building a tmp vault tree, importing
it into a fresh WikiStore, and asserting the resulting slugs, frontmatter,
wikilinks, and commit shape.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from outmem._store.import_vault import _slugify, import_vault
from outmem.exceptions import OutmemError
from outmem.frontmatter import parse_wiki_page
from outmem.store import WikiStore


@pytest.fixture
def store(tmp_path: Path) -> WikiStore:
    return WikiStore.init(tmp_path / "w")


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    """A small Obsidian-shaped vault.

    Layout::

        vault/
        ├── Pricing Formula.md          # H1 + wikilink to "Acme MSA"
        ├── Acme MSA.md                 # bare note
        ├── projects/
        │   └── alpha.md
        ├── clients/
        │   └── alpha.md                # collides with projects/alpha
        └── .obsidian/
            └── workspace.md            # must be ignored
    """
    root = tmp_path / "vault"
    (root / "projects").mkdir(parents=True)
    (root / "clients").mkdir()
    (root / ".obsidian").mkdir()

    (root / "Pricing Formula.md").write_text(
        "# Pricing formula\n\nThe formula is cost-plus 35%. See [[Acme MSA]] for the exception.\n",
        encoding="utf-8",
    )
    (root / "Acme MSA.md").write_text(
        "# Acme MSA\n\nAcme's contract overrides the standard pricing.\n",
        encoding="utf-8",
    )
    (root / "projects" / "alpha.md").write_text(
        "# Project Alpha\n\nDelivery team is small.\n",
        encoding="utf-8",
    )
    (root / "clients" / "alpha.md").write_text(
        "# Client Alpha\n\nLargest by revenue.\n",
        encoding="utf-8",
    )
    (root / ".obsidian" / "workspace.md").write_text(
        "Obsidian config; should be skipped.\n", encoding="utf-8"
    )
    return root


# ---------------------------------------------------------------------------
# slugify recipe
# ---------------------------------------------------------------------------


class TestSlugifyForImport:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Pricing Formula", "pricing-formula"),
            ("Café au Lait", "cafe-au-lait"),
            ("my_great_note", "my-great-note"),
            ("Q3, 2026 — kick-off!", "q3-2026-kick-off"),
            ("---foo---", "foo"),
        ],
    )
    def test_recipe(self, raw: str, expected: str) -> None:
        assert _slugify(raw) == expected


# ---------------------------------------------------------------------------
# End-to-end import
# ---------------------------------------------------------------------------


def test_import_produces_correct_slugs_and_titles(
    store: WikiStore, vault: Path
) -> None:
    summary = import_vault(store, vault)

    # All four notes imported; .obsidian/ skipped.
    assert summary.pages_imported == 4

    pricing = parse_wiki_page(
        (store.pages_path / "pricing-formula.md").read_text(encoding="utf-8")
    )
    assert pricing[0].slug == "pricing-formula"
    assert pricing[0].title == "Pricing formula"  # from the H1

    acme = parse_wiki_page(
        (store.pages_path / "acme-msa.md").read_text(encoding="utf-8")
    )
    assert acme[0].slug == "acme-msa"
    assert acme[0].title == "Acme MSA"


def test_import_resolves_slug_collisions_with_parent_prefix(
    store: WikiStore, vault: Path
) -> None:
    """Two ``alpha.md`` files under different parents — the second one
    gets prefixed so the flat slug namespace stays unique."""
    summary = import_vault(store, vault)
    slugs = {p.stem for p in store.pages_path.glob("*.md") if p.name != "index.md"}
    assert "alpha" in slugs  # one of them wins the bare slug
    assert "clients-alpha" in slugs or "projects-alpha" in slugs
    assert len(summary.slug_collisions) == 1


def test_import_rewrites_wikilinks_to_slugs(
    store: WikiStore, vault: Path
) -> None:
    """`[[Acme MSA]]` in the body becomes `[[acme-msa|Acme MSA]]` —
    display preserved, slug machine-resolvable."""
    import_vault(store, vault)
    body = (store.pages_path / "pricing-formula.md").read_text(encoding="utf-8")
    assert "[[acme-msa|Acme MSA]]" in body
    assert "[[Acme MSA]]" not in body


def test_import_writes_provenance(store: WikiStore, vault: Path) -> None:
    """Every imported page carries a provenance pointer back to its
    original vault-relative path."""
    import_vault(store, vault)
    pricing, _ = parse_wiki_page(
        (store.pages_path / "pricing-formula.md").read_text(encoding="utf-8")
    )
    assert pricing.provenance == [
        {"path": "Pricing Formula.md", "source": "obsidian-import"}
    ]


def test_import_creates_single_commit(store: WikiStore, vault: Path) -> None:
    """The whole import lands as one ``import: <vault-name>`` commit."""
    store.append_log(topic="seed", content="- pre-import marker\n")
    head_before = store.head()
    import_vault(store, vault)
    head_after = store.head()
    assert head_before is not None and head_after is not None and head_before != head_after

    # Walk the parent chain: exactly one commit between head_before and head_after.
    from outmem.git_ops import log_since

    new_commits = [
        c for c in log_since(store.root, since=head_before)
        if c.sha != head_before
    ]
    assert len(new_commits) == 1
    assert new_commits[0].subject == "import: vault"


def test_import_skips_hidden_dirs(store: WikiStore, vault: Path) -> None:
    """`.obsidian/`, `.git/`, etc. are not imported."""
    import_vault(store, vault)
    slugs = {p.stem for p in store.pages_path.glob("*.md") if p.name != "index.md"}
    assert "workspace" not in slugs


def test_import_refuses_non_empty_wiki_without_force(
    store: WikiStore, vault: Path
) -> None:
    """First import succeeds; a second one refuses unless force=True."""
    import_vault(store, vault)
    with pytest.raises(OutmemError, match="already has"):
        import_vault(store, vault)


def test_import_force_clobbers_existing(store: WikiStore, vault: Path) -> None:
    import_vault(store, vault)
    # Now mutate the vault and re-import with force.
    (vault / "Pricing Formula.md").write_text(
        "# Pricing formula\n\nRevised: cost-plus 40%.\n",
        encoding="utf-8",
    )
    import_vault(store, vault, force=True)
    body = (store.pages_path / "pricing-formula.md").read_text(encoding="utf-8")
    assert "Revised: cost-plus 40%" in body


def test_import_rejects_non_directory(store: WikiStore, tmp_path: Path) -> None:
    not_a_dir = tmp_path / "nope.md"
    not_a_dir.write_text("x", encoding="utf-8")
    with pytest.raises(OutmemError, match="not a directory"):
        import_vault(store, not_a_dir)


def test_import_rejects_empty_vault(store: WikiStore, tmp_path: Path) -> None:
    empty = tmp_path / "empty-vault"
    empty.mkdir()
    with pytest.raises(OutmemError, match="no \\*\\.md files"):
        import_vault(store, empty)


def test_import_unresolved_wikilink_left_alone(
    store: WikiStore, tmp_path: Path
) -> None:
    """A wikilink pointing to a non-existent note stays as-is; lint
    surfaces it later."""
    v = tmp_path / "vault"
    v.mkdir()
    (v / "Note.md").write_text(
        "See [[Missing Note]] for context.\n", encoding="utf-8"
    )
    summary = import_vault(store, v)
    body = (store.pages_path / "note.md").read_text(encoding="utf-8")
    assert "[[Missing Note]]" in body
    assert summary.wikilinks_unresolved == 1
    assert summary.wikilinks_rewritten == 0
