"""Tests for ``outmem.store``.

The store wires together the rest of the package; these tests focus on
the integration seam — round-tripping a page through write/read/extend,
the commit-message grammar (TARS Retained depends on it), the steering
loop's agent-exclusion, and the layout :meth:`WikiStore.init` creates.
"""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from outmem.exceptions import FrontmatterError, OutmemError, SlugError
from outmem.git_ops import current_head, log_since
from outmem.store import AgentIdentity, WikiStore, WikiStoreConfig

# ---------------------------------------------------------------------------
# Fresh store fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fresh_store(tmp_path: Path) -> WikiStore:
    """A brand-new WikiStore initialised at tmp_path/wiki."""
    return WikiStore.init(tmp_path / "wiki")


@pytest.fixture
def store_with_remote(fresh_store: WikiStore, bare_remote: Path) -> WikiStore:
    """A store wired to a bare remote, with an initial commit on main."""
    subprocess.run(
        ["git", "remote", "add", "origin", str(bare_remote)],
        cwd=str(fresh_store.root),
        check=True,
        capture_output=True,
    )
    fresh_store.write_page(
        "seed",
        title="Seed page",
        body="The first page.",
    )
    fresh_store.push()
    return fresh_store


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_init_creates_layout(self, tmp_path: Path) -> None:
        store = WikiStore.init(tmp_path / "w")
        assert store.root.is_dir()
        assert (store.root / "wiki").is_dir()
        assert (store.root / "wiki/pages").is_dir()
        assert (store.root / "wiki/sources").is_dir()
        # sources-local/ is created lazily on first local ingest, so a
        # wiki that never touches restricted material stays identical to
        # one from before the split existed.
        assert not (store.root / "wiki/sources-local").exists()
        assert (store.root / "log").is_dir()
        assert (store.root / ".outmem").is_dir()
        assert (store.root / ".outmem/.gitignore").exists()
        assert (store.root / "CONTRIBUTORS.md").exists()
        assert (store.root / ".git").exists()

    def test_init_writes_agent_in_contributors(self, tmp_path: Path) -> None:
        store = WikiStore.init(
            tmp_path / "w",
            agent_identity=AgentIdentity(name="My Agent", email="my-agent@host"),
        )
        body = (store.root / "CONTRIBUTORS.md").read_text()
        assert "My Agent" in body
        assert "my-agent@host" in body

    def test_init_does_not_clobber_existing_contributors(self, tmp_path: Path) -> None:
        root = tmp_path / "w"
        root.mkdir()
        (root / "CONTRIBUTORS.md").write_text("# Custom\n- Alice <alice@example.com>\n")
        WikiStore.init(root)
        body = (root / "CONTRIBUTORS.md").read_text()
        assert "Alice" in body
        assert "outmem agent" not in body  # not appended

    def test_init_raises_when_git_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No `git` on PATH → clear error, no directories created."""
        import outmem.store as store_mod

        monkeypatch.setattr(store_mod, "git_available", lambda: False)
        root = tmp_path / "fresh-wiki"
        with pytest.raises(OutmemError, match="git"):
            WikiStore.init(root)
        assert not root.exists()

    def test_init_raises_when_rg_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No `rg` on PATH → clear error, no directories created."""
        import outmem.store as store_mod

        monkeypatch.setattr(store_mod, "rg_available", lambda: False)
        root = tmp_path / "fresh-wiki"
        with pytest.raises(OutmemError, match="rg \\(ripgrep\\)"):
            WikiStore.init(root)
        assert not root.exists()

    def test_open_existing_layout(self, fresh_store: WikiStore) -> None:
        reopened = WikiStore.open(fresh_store.root)
        assert reopened.wiki_path == fresh_store.wiki_path

    def test_open_missing_root_raises(self, tmp_path: Path) -> None:
        with pytest.raises(OutmemError, match="does not exist"):
            WikiStore.open(tmp_path / "noplace")


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


class TestRead:
    def test_read_after_write(self, fresh_store: WikiStore) -> None:
        fresh_store.write_page("alpha", title="Alpha", body="alpha body.")
        page = fresh_store.read("alpha")
        assert page.title == "Alpha"
        assert page.slug == "alpha"
        assert "alpha body" in page.body
        assert page.frontmatter.created is not None
        assert page.frontmatter.updated is not None

    def test_read_missing_raises(self, fresh_store: WikiStore) -> None:
        with pytest.raises(OutmemError, match="No such wiki page"):
            fresh_store.read("nope")

    def test_read_rejects_unsafe_slug(self, fresh_store: WikiStore) -> None:
        with pytest.raises(SlugError):
            fresh_store.read("../etc/passwd")

    def test_exists(self, fresh_store: WikiStore) -> None:
        assert fresh_store.exists("x") is False
        fresh_store.write_page("x", title="X", body="b")
        assert fresh_store.exists("x") is True
        assert fresh_store.exists("../escape") is False

    def test_list_slugs(self, fresh_store: WikiStore) -> None:
        assert fresh_store.list_slugs() == []
        fresh_store.write_page("b", title="B", body="x")
        fresh_store.write_page("a", title="A", body="x")
        assert fresh_store.list_slugs() == ["a", "b"]

    def test_read_malformed_page_raises(self, fresh_store: WikiStore) -> None:
        bad = fresh_store.pages_path / "broken.md"
        bad.parent.mkdir(parents=True, exist_ok=True)
        bad.write_text("no frontmatter here\n")
        with pytest.raises(FrontmatterError):
            fresh_store.read("broken")

    def test_read_self_heals_repairable_frontmatter(
        self, fresh_store: WikiStore, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A page with an unquoted ': ' in the title (the import failure
        mode) is repaired in memory on read — usable content, no raise —
        and a warning is logged. The on-disk file is left for the hook /
        repair_pages to persist."""
        import logging

        bad = fresh_store.pages_path / "rki.md"
        bad.parent.mkdir(parents=True, exist_ok=True)
        on_disk = "---\ntitle: Influenza (Teil 1): saisonale\nslug: rki\n---\n\nbody\n"
        bad.write_text(on_disk, encoding="utf-8")

        with caplog.at_level(logging.WARNING):
            page = fresh_store.read("rki")
        assert page.title == "Influenza (Teil 1): saisonale"  # usable content
        assert "self-healed" in caplog.text
        # read() did NOT rewrite the file (no surprise disk write).
        assert bad.read_text(encoding="utf-8") == on_disk

    def test_read_still_raises_when_unrepairable(
        self, fresh_store: WikiStore
    ) -> None:
        """A frontmatter break the repair doesn't cover still surfaces."""
        bad = fresh_store.pages_path / "wrecked.md"
        bad.parent.mkdir(parents=True, exist_ok=True)
        bad.write_text("---\ntitle: ok\ntags:\n[broken\n---\n\nbody\n", encoding="utf-8")
        with pytest.raises(FrontmatterError):
            fresh_store.read("wrecked")


# ---------------------------------------------------------------------------
# Write / extend / log
# ---------------------------------------------------------------------------


class TestWrite:
    def test_write_uses_compact_subject(self, fresh_store: WikiStore) -> None:
        fresh_store.write_page("alpha", title="Alpha", body="body")
        log = log_since(fresh_store.root)
        assert log[0].subject == "compact: alpha"

    def test_internal_commit_bypasses_pre_commit_hook(
        self, fresh_store: WikiStore
    ) -> None:
        """outmem's own commits must not fire the (possibly auto-installed)
        pre-commit hook — they maintain derived artefacts inline, so firing
        it would be redundant and, with auto-install, re-entrant. Install a
        hook that writes a sentinel; a write_page commit must not trip it."""
        hooks = fresh_store.root / ".git" / "hooks"
        hooks.mkdir(parents=True, exist_ok=True)
        sentinel = fresh_store.root / "HOOK_FIRED"
        (hooks / "pre-commit").write_text(
            f"#!/bin/sh\ntouch '{sentinel}'\n", encoding="utf-8"
        )
        (hooks / "pre-commit").chmod(0o755)

        fresh_store.write_page("alpha", title="Alpha", body="body")
        assert not sentinel.exists()  # --no-verify kept the hook from firing

    def test_write_persists_frontmatter(self, fresh_store: WikiStore) -> None:
        when = datetime(2026, 5, 11, 10, 0, 0, tzinfo=UTC)
        fresh_store.write_page(
            "alpha",
            title="Alpha",
            body="hi",
            provenance=["raw/source.md"],
            tags=["pricing"],
            created=when,
        )
        page = fresh_store.read("alpha")
        assert page.frontmatter.provenance == ["raw/source.md"]
        assert page.frontmatter.tags == ["pricing"]
        assert page.frontmatter.created == when

    def test_write_duplicate_slug_rejected(self, fresh_store: WikiStore) -> None:
        fresh_store.write_page("x", title="X", body="a")
        with pytest.raises(OutmemError, match="already exists"):
            fresh_store.write_page("x", title="X", body="b")

    def test_write_records_agent_identity(self, fresh_store: WikiStore) -> None:
        identity = fresh_store.config.agent_identity
        fresh_store.write_page("alpha", title="Alpha", body="x")
        log = log_since(fresh_store.root)
        assert log[0].author_name == identity.name
        assert log[0].author_email == identity.email

    def test_extend_uses_extend_subject(self, fresh_store: WikiStore) -> None:
        fresh_store.write_page("alpha", title="Alpha", body="v1")
        fresh_store.extend_page("alpha", body="v2 with new content")
        log = log_since(fresh_store.root)
        subjects = [c.subject for c in log]
        assert subjects[0] == "extend: alpha"
        assert subjects[1] == "compact: alpha"

    def test_extend_preserves_frontmatter_except_updated(self, fresh_store: WikiStore) -> None:
        when = datetime(2026, 5, 11, 10, 0, 0, tzinfo=UTC)
        fresh_store.write_page(
            "alpha",
            title="Alpha",
            body="v1",
            provenance=["raw/source.md"],
            tags=["x"],
            created=when,
        )
        page_v1 = fresh_store.read("alpha")
        fresh_store.extend_page("alpha", body="v2")
        page_v2 = fresh_store.read("alpha")
        assert page_v2.frontmatter.title == page_v1.frontmatter.title
        assert page_v2.frontmatter.provenance == page_v1.frontmatter.provenance
        assert page_v2.frontmatter.created == page_v1.frontmatter.created
        assert page_v2.frontmatter.tags == page_v1.frontmatter.tags
        assert page_v2.frontmatter.updated >= page_v1.frontmatter.updated
        assert "v2" in page_v2.body

    def test_extend_missing_page_raises(self, fresh_store: WikiStore) -> None:
        with pytest.raises(OutmemError, match="No such wiki page"):
            fresh_store.extend_page("ghost", body="x")

    def test_append_log_creates_today_file(self, fresh_store: WikiStore) -> None:
        when = datetime(2026, 5, 11, 12, 0, 0, tzinfo=UTC)
        fresh_store.append_log(topic="pricing", content="Noticed X.", when=when)
        log_file = fresh_store.log_path / "2026-05-11.md"
        assert log_file.exists()
        body = log_file.read_text()
        assert "# 2026-05-11" in body
        assert "Noticed X." in body

    def test_append_log_uses_log_subject(self, fresh_store: WikiStore) -> None:
        when = datetime(2026, 5, 11, 12, 0, 0, tzinfo=UTC)
        fresh_store.append_log(topic="pricing", content="x", when=when)
        log = log_since(fresh_store.root)
        assert log[0].subject == "log: pricing"

    def test_append_log_appends_to_existing(self, fresh_store: WikiStore) -> None:
        when = datetime(2026, 5, 11, 12, 0, 0, tzinfo=UTC)
        fresh_store.append_log(topic="a", content="first", when=when)
        fresh_store.append_log(topic="b", content="second", when=when)
        body = (fresh_store.log_path / "2026-05-11.md").read_text()
        assert "first" in body
        assert "second" in body

    def test_append_log_empty_topic_rejected(self, fresh_store: WikiStore) -> None:
        with pytest.raises(OutmemError, match="topic"):
            fresh_store.append_log(topic="   ", content="x")


# ---------------------------------------------------------------------------
# Search and graph
# ---------------------------------------------------------------------------


class TestNamespacedSlugs:
    """End-to-end coverage for the v0.2 slug grammar.

    Pins the workflow a downstream consumer sees: write → read → list
    → search → backlinks → history with namespaced slugs (``abx:foo``)
    coexisting with flat slugs (``pricing-formula``) and with mixed
    page-and-folder layouts (``abx`` + ``abx:penicillin``).
    """

    def test_write_read_namespaced(self, fresh_store: WikiStore) -> None:
        fresh_store.write_page(
            "abx:penicillin", title="Penicillin", body="A beta-lactam."
        )
        on_disk = fresh_store.pages_path / "abx" / "penicillin.md"
        assert on_disk.exists()
        page = fresh_store.read("abx:penicillin")
        assert page.frontmatter.slug == "abx:penicillin"
        assert page.frontmatter.title == "Penicillin"

    def test_deep_nesting(self, fresh_store: WikiStore) -> None:
        fresh_store.write_page(
            "a:b:c:d", title="Deep", body="four levels deep"
        )
        on_disk = fresh_store.pages_path / "a" / "b" / "c" / "d.md"
        assert on_disk.exists()

    def test_page_and_folder_coexist(self, fresh_store: WikiStore) -> None:
        # Write `abx:penicillin` first (creates wiki/pages/abx/ as a dir),
        # then write `abx` as a flat page sibling to that dir.
        fresh_store.write_page("abx:penicillin", title="P", body="x")
        fresh_store.write_page("abx", title="Overview", body="y")
        assert (fresh_store.pages_path / "abx.md").exists()
        assert (fresh_store.pages_path / "abx" / "penicillin.md").exists()
        # Both read back distinctly.
        assert fresh_store.read("abx").frontmatter.title == "Overview"
        assert fresh_store.read("abx:penicillin").frontmatter.title == "P"

    def test_list_slugs_uses_colon_notation(self, fresh_store: WikiStore) -> None:
        fresh_store.write_page("pricing-formula", title="P", body="x")
        fresh_store.write_page("abx:penicillin", title="P", body="x")
        fresh_store.write_page("abx:side-effects:misc", title="P", body="x")
        slugs = fresh_store.list_slugs()
        assert "pricing-formula" in slugs
        assert "abx:penicillin" in slugs
        assert "abx:side-effects:misc" in slugs

    def test_backlinks_across_namespaces(self, fresh_store: WikiStore) -> None:
        fresh_store.write_page("abx:penicillin", title="P", body="x")
        fresh_store.write_page(
            "infection:strep", title="Strep", body="Treat with [[abx:penicillin]]."
        )
        assert fresh_store.backlinks("abx:penicillin") == ("infection:strep",)

    def test_history_for_namespaced_slug(self, fresh_store: WikiStore) -> None:
        fresh_store.write_page("abx:penicillin", title="P", body="v1")
        fresh_store.extend_page("abx:penicillin", body="v2")
        hist = fresh_store.history("abx:penicillin")
        assert len(hist) == 2
        # Newest first.
        assert hist[0].subject == "extend: abx:penicillin"
        assert hist[1].subject == "compact: abx:penicillin"

    def test_index_renders_namespaced_slugs(self, fresh_store: WikiStore) -> None:
        fresh_store.write_page("pricing-formula", title="P", body="x")
        fresh_store.write_page("abx:penicillin", title="P", body="x")
        index = (fresh_store.wiki_path / "index.md").read_text(encoding="utf-8")
        assert "[[pricing-formula]]" in index
        assert "[[abx:penicillin]]" in index


class TestSearchAndGraph:
    def test_search_wiki_scope(self, fresh_store: WikiStore) -> None:
        fresh_store.write_page(
            "pricing-formula",
            title="Pricing",
            body="The pricing formula is cost-plus.",
        )
        result = fresh_store.search("cost-plus", fixed_strings=True)
        assert any(h.path == "pricing-formula.md" for h in result.hits)

    def test_search_sources_scope_covers_tracked_tree(
        self, fresh_store: WikiStore
    ) -> None:
        (fresh_store.sources_path / "doc.md").write_text("source text token-xyz\n")
        result = fresh_store.search("token-xyz", scope="sources")
        assert any(h.path.endswith("doc.md") for h in result.hits)

    def test_search_sources_scope_covers_local_tree(
        self, fresh_store: WikiStore
    ) -> None:
        """The trap this split exists to remove: an agent falling through
        to the sources must see BOTH trees. Missing the local one would
        silently hide half the corpus behind a distribution policy the
        agent has no reason to know about."""
        local = fresh_store.ensure_sources_local()
        (local / "restricted.md").write_text("licensed text token-abc\n")
        result = fresh_store.search("token-abc", scope="sources")
        assert any(h.path.endswith("restricted.md") for h in result.hits)

    def test_search_sources_scope_spans_both_trees_at_once(
        self, fresh_store: WikiStore
    ) -> None:
        (fresh_store.sources_path / "open.md").write_text("shared token-both\n")
        local = fresh_store.ensure_sources_local()
        (local / "closed.md").write_text("licensed token-both\n")
        hits = {h.path for h in fresh_store.search("token-both", scope="sources").hits}
        assert any(p.endswith("open.md") for p in hits)
        assert any(p.endswith("closed.md") for p in hits)

    def test_search_sources_scope_is_empty_when_no_tree_exists(
        self, tmp_path: Path
    ) -> None:
        """A read-only store skips layout creation, so neither tree need
        exist. That must read as "no hits", not as an rg failure."""
        seed = WikiStore.init(tmp_path / "w")
        seed.write_page("p", title="P", body="body")
        import shutil

        shutil.rmtree(seed.sources_path)
        store = WikiStore.open(seed.root, read_only=True)
        assert store.search("anything", scope="sources").hits == ()

    def test_search_unknown_scope_raises(self, fresh_store: WikiStore) -> None:
        with pytest.raises(OutmemError, match="scope"):
            fresh_store.search("x", scope="bogus")

    def test_backlinks(self, fresh_store: WikiStore) -> None:
        fresh_store.write_page("acme-msa", title="Acme MSA", body="x")
        fresh_store.write_page(
            "pricing-formula",
            title="Pricing",
            body="See [[acme-msa]].",
        )
        backlinks = fresh_store.backlinks("acme-msa")
        assert backlinks == ("pricing-formula",)

    def test_history(self, fresh_store: WikiStore) -> None:
        fresh_store.write_page("alpha", title="Alpha", body="v1")
        fresh_store.extend_page("alpha", body="v2")
        history = fresh_store.history("alpha")
        assert len(history) == 2

    def test_evolution_returns_diff_stream(self, fresh_store: WikiStore) -> None:
        fresh_store.write_page("alpha", title="Alpha", body="v1")
        fresh_store.extend_page("alpha", body="v2 different content")
        out = fresh_store.evolution(["alpha"])
        assert "diff --git" in out
        assert "v2 different content" in out


# ---------------------------------------------------------------------------
# Steering
# ---------------------------------------------------------------------------


class TestSteering:
    def test_steering_excludes_agent(self, populated_repo: Path) -> None:
        store = WikiStore.open(populated_repo)
        signal = store.steering()
        emails = {c.author_email for c in signal}
        assert "agent@host" not in emails
        assert {"alice@example.com", "bob@example.com"} <= emails

    def test_steering_uses_last_run_when_no_since(self, populated_repo: Path) -> None:
        store = WikiStore.open(populated_repo)
        # No last-run marker yet → returns everything (modulo exclusion).
        first = store.steering()
        assert len(first) >= 2
        # Record a run strictly in the future of the fixture's commits.
        # `git log --since` is inclusive at the second granularity, so a
        # marker at "now" can still surface commits made in the same
        # second; the +1h shim makes the test deterministic.
        store.record_run(when=datetime.now(UTC) + timedelta(hours=1))
        later = store.steering()
        assert later == []

    def test_steering_explicit_since(self, populated_repo: Path) -> None:
        store = WikiStore.open(populated_repo)
        future = datetime(2099, 1, 1, tzinfo=UTC)
        assert store.steering(since=future) == []


# ---------------------------------------------------------------------------
# Sync + run marker
# ---------------------------------------------------------------------------


class TestSync:
    def test_push_publishes_commit(self, store_with_remote: WikiStore, bare_remote: Path) -> None:
        head = current_head(store_with_remote.root)
        result = subprocess.run(
            ["git", "rev-parse", "refs/heads/main"],
            cwd=str(bare_remote),
            check=True,
            capture_output=True,
            text=True,
        )
        assert result.stdout.strip() == head

    def test_pull_invalidates_backlinks_cache(self, store_with_remote: WikiStore) -> None:
        # Seed an in-memory cache.
        store_with_remote.write_page("a", title="A", body="[[seed]]")
        head = store_with_remote.head()
        store_with_remote.backlinks_cache.graph_for(head)
        assert store_with_remote.backlinks_cache._memo is not None  # type: ignore[attr-defined]
        # Pull (no remote changes) clears the memo so we re-evaluate
        # against whatever HEAD ends up being post-rebase.
        store_with_remote.pull()
        assert store_with_remote.backlinks_cache._memo is None  # type: ignore[attr-defined]

    def test_head_none_before_first_commit(self, tmp_path: Path) -> None:
        store = WikiStore.init(tmp_path / "w")
        assert store.head() is None

    def test_record_and_read_last_run(self, fresh_store: WikiStore) -> None:
        fresh_store.write_page("a", title="A", body="x")
        marker = fresh_store.record_run()
        assert marker.head == fresh_store.head()
        assert fresh_store.last_run() == marker


# ---------------------------------------------------------------------------
# Contributors
# ---------------------------------------------------------------------------


class TestContributors:
    def test_contributors_loaded_from_seed(self, fresh_store: WikiStore) -> None:
        contributors = fresh_store.contributors()
        assert contributors.lookup(fresh_store.config.agent_identity.email) is not None

    def test_contributors_refresh(self, fresh_store: WikiStore) -> None:
        fresh_store.contributors_path.write_text("- Alice <alice@example.com>\n", encoding="utf-8")
        fresh_store.contributors(refresh=True)
        assert fresh_store.contributors().lookup("alice@example.com") is not None


# ---------------------------------------------------------------------------
# Construction config exposed publicly
# ---------------------------------------------------------------------------


def test_config_dataclass_is_constructable() -> None:
    cfg = WikiStoreConfig(root=Path("/tmp/x"))
    assert cfg.wiki_dir == "wiki"
    assert cfg.log_dir == "log"


# ---------------------------------------------------------------------------
# Config integration: config.yaml + .env loaded by WikiStore.open / init
# ---------------------------------------------------------------------------


class TestConfigIntegration:
    def test_init_seeds_config_yaml(self, tmp_path: Path) -> None:
        store = WikiStore.init(tmp_path / "w")
        yaml_path = store.root / "config.yaml"
        assert yaml_path.exists()
        body = yaml_path.read_text()
        assert "model: anthropic:" in body
        assert "remove_stale_lock: true" in body

    def test_init_gitignores_dotenv_as_safety_net(self, tmp_path: Path) -> None:
        """``.env`` belongs at the user's project root, not the wiki —
        but if one ends up here by accident, ``.gitignore`` still
        keeps it out of commits."""
        store = WikiStore.init(tmp_path / "w")
        gitignore = store.root / ".gitignore"
        assert gitignore.exists()
        assert ".env" in gitignore.read_text()

    def test_init_does_not_clobber_existing_yaml(self, tmp_path: Path) -> None:
        root = tmp_path / "w"
        root.mkdir()
        (root / "config.yaml").write_text("model: custom:override\n")
        WikiStore.init(root)
        assert (root / "config.yaml").read_text() == "model: custom:override\n"

    def test_open_loads_model_from_yaml(self, tmp_path: Path) -> None:
        store = WikiStore.init(tmp_path / "w")
        (store.root / "config.yaml").write_text(
            "model: openai:gpt-5\nagent:\n  name: x\n  email: x@y\n", encoding="utf-8"
        )
        reopened = WikiStore.open(store.root)
        assert reopened.config.outmem.model == "openai:gpt-5"

    def test_open_loads_agent_identity_from_yaml(self, tmp_path: Path) -> None:
        root = tmp_path / "w"
        root.mkdir()
        (root / "config.yaml").write_text(
            "agent:\n  name: yaml-bot\n  email: bot@yaml.test\n", encoding="utf-8"
        )
        store = WikiStore.open(root)
        assert store.config.agent_identity.name == "yaml-bot"
        assert store.config.agent_identity.email == "bot@yaml.test"

    def test_open_explicit_args_win_over_yaml(self, tmp_path: Path) -> None:
        root = tmp_path / "w"
        root.mkdir()
        (root / "config.yaml").write_text(
            "remote:\n  name: upstream\n  branch: trunk\n", encoding="utf-8"
        )
        store = WikiStore.open(root, remote="origin", branch="main")
        assert store.config.remote == "origin"
        assert store.config.branch == "main"

    def test_open_loads_dotenv_from_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``WikiStore.open`` calls ``load_dotenv()`` with no args; the
        search walks upward from CWD (typical: user runs ``outmem``
        from their project repo). The wiki itself can live anywhere."""
        import os

        monkeypatch.delenv("OUTMEM_INTEGRATION_KEY", raising=False)
        project = tmp_path / "project"
        wiki = tmp_path / "wiki"
        project.mkdir()
        wiki.mkdir()
        (project / ".env").write_text("OUTMEM_INTEGRATION_KEY=loaded\n", encoding="utf-8")
        monkeypatch.chdir(project)
        WikiStore.open(wiki)
        assert os.environ.get("OUTMEM_INTEGRATION_KEY") == "loaded"
        monkeypatch.delenv("OUTMEM_INTEGRATION_KEY", raising=False)

    def test_open_removes_stale_index_lock(self, tmp_path: Path) -> None:
        """A stale .git/index.lock left by a killed prior run gets cleaned up."""
        import os
        import time

        store = WikiStore.init(tmp_path / "w")
        lock = store.root / ".git/index.lock"
        lock.touch()
        # Backdate so the cleanup recognises it as stale.
        past = time.time() - 3600
        os.utime(lock, (past, past))

        WikiStore.open(store.root)
        assert not lock.exists()

    def test_open_keeps_fresh_index_lock(self, tmp_path: Path) -> None:
        """A fresh lock (likely another live git op) must NOT be removed."""
        store = WikiStore.init(tmp_path / "w")
        lock = store.root / ".git/index.lock"
        lock.touch()
        WikiStore.open(store.root)
        assert lock.exists()
        lock.unlink()

    def test_open_disable_stale_lock_cleanup_via_yaml(self, tmp_path: Path) -> None:
        import os
        import time

        store = WikiStore.init(tmp_path / "w")
        (store.root / "config.yaml").write_text(
            "git:\n  remove_stale_lock: false\n", encoding="utf-8"
        )
        lock = store.root / ".git/index.lock"
        lock.touch()
        past = time.time() - 3600
        os.utime(lock, (past, past))

        WikiStore.open(store.root)
        # Cleanup disabled — stale lock survives.
        assert lock.exists()
        lock.unlink()


# ---------------------------------------------------------------------------
# Read-only mode
# ---------------------------------------------------------------------------


class TestReadOnly:
    """``WikiStore.open(path, read_only=True)`` refuses every commit path.

    The store still reads cleanly; only the commit funnel
    (:meth:`WikiStore._commit_paths`) trips, so every write entry point
    (``write_page``, ``extend_page``, ``append_log``,
    ``rebuild_index``, ``add_source``, ``import_vault``) raises
    :class:`OutmemError` regardless of which method was called.
    """

    @pytest.fixture
    def seeded(self, tmp_path: Path) -> Path:
        """A wiki with one page committed, then closed."""
        store = WikiStore.init(tmp_path / "wiki")
        store.write_page(
            "pricing",
            title="Pricing",
            body="Cost-plus 35%.\n",
            tags=["pricing"],
        )
        store.close()
        return store.root

    def test_open_read_only_sets_flag(self, seeded: Path) -> None:
        store = WikiStore.open(seeded, read_only=True)
        assert store.config.read_only is True

    def test_open_default_is_writable(self, seeded: Path) -> None:
        # Round-trip a regular open to make sure the new kwarg doesn't
        # leak: the existing default behaviour must still commit.
        store = WikiStore.open(seeded)
        assert store.config.read_only is False
        store.append_log(topic="smoke", content="- still writable\n")

    def test_read_still_works(self, seeded: Path) -> None:
        store = WikiStore.open(seeded, read_only=True)
        page = store.read("pricing")
        assert page.title == "Pricing"
        assert "Cost-plus 35%" in page.body

    def test_search_still_works(self, seeded: Path) -> None:
        store = WikiStore.open(seeded, read_only=True)
        result = store.search("Cost-plus", scope="wiki")
        assert any("pricing" in hit.path for hit in result.hits)

    def test_list_slugs_still_works(self, seeded: Path) -> None:
        store = WikiStore.open(seeded, read_only=True)
        assert "pricing" in store.list_slugs()

    def test_history_still_works(self, seeded: Path) -> None:
        store = WikiStore.open(seeded, read_only=True)
        assert len(store.history("pricing")) >= 1

    def test_write_page_refused(self, seeded: Path) -> None:
        store = WikiStore.open(seeded, read_only=True)
        with pytest.raises(OutmemError, match="read-only"):
            store.write_page("new-page", title="New", body="x")

    def test_extend_page_refused(self, seeded: Path) -> None:
        store = WikiStore.open(seeded, read_only=True)
        with pytest.raises(OutmemError, match="read-only"):
            store.extend_page("pricing", body="Cost-plus 99%.\n")

    def test_append_log_refused(self, seeded: Path) -> None:
        store = WikiStore.open(seeded, read_only=True)
        with pytest.raises(OutmemError, match="read-only"):
            store.append_log(topic="anything", content="- blocked\n")

    def test_rebuild_index_refused_when_dirty(self, seeded: Path) -> None:
        # Drop a new page directly on disk so the index is genuinely
        # stale — rebuild_index would otherwise short-circuit when
        # the regenerated file matches HEAD.
        new_page = seeded / "wiki" / "pages" / "extra.md"
        new_page.parent.mkdir(parents=True, exist_ok=True)
        new_page.write_text(
            "---\ntitle: Extra\nslug: extra\n"
            "created: 2026-05-19T00:00:00Z\nupdated: 2026-05-19T00:00:00Z\n---\n\n"
            "body\n",
            encoding="utf-8",
        )
        store = WikiStore.open(seeded, read_only=True)
        with pytest.raises(OutmemError, match="read-only"):
            store.rebuild_index()

    def test_add_source_refused(self, seeded: Path, tmp_path: Path) -> None:
        source = tmp_path / "doc.md"
        source.write_text("# A document\n\nSome facts.\n", encoding="utf-8")
        store = WikiStore.open(seeded, read_only=True)
        with pytest.raises(OutmemError, match="read-only"):
            store.add_source(source)

    def test_import_vault_refused(self, tmp_path: Path) -> None:
        # Empty wiki so import_vault doesn't hit its own non-empty guard
        # before the read-only commit guard fires.
        empty = WikiStore.init(tmp_path / "empty")
        empty.close()
        vault = tmp_path / "vault"
        vault.mkdir()
        (vault / "note.md").write_text("# Note\n\nbody.\n", encoding="utf-8")
        store = WikiStore.open(empty.root, read_only=True)
        with pytest.raises(OutmemError, match="read-only"):
            store.import_vault(vault)

    def test_head_unchanged_after_refused_writes(self, seeded: Path) -> None:
        store = WikiStore.open(seeded, read_only=True)
        head_before = store.head()
        for fn in (
            lambda: store.write_page("p", title="T", body="b"),
            lambda: store.extend_page("pricing", body="x"),
            lambda: store.append_log(topic="t", content="c\n"),
        ):
            with pytest.raises(OutmemError):
                fn()
        assert store.head() == head_before

    def test_pull_refused(self, seeded: Path) -> None:
        """``git pull --rebase`` would mutate the working tree, so the
        read-only contract refuses it. ``push`` stays unguarded — with
        ``_commit_paths`` refusing every commit, there's nothing local
        to push, and propagating prior commits isn't a wiki mutation."""
        store = WikiStore.open(seeded, read_only=True)
        with pytest.raises(OutmemError, match="read-only"):
            store.pull()

    def test_does_not_create_outmem_dir(self, tmp_path: Path) -> None:
        """A wiki where ``.outmem/`` was never created (e.g. a curator
        shipped a fresh clone, or the cache was wiped) must open
        read-only without creating it. This makes ``read_only=True``
        safe to use against literally read-only filesystems."""
        import shutil

        seed = WikiStore.init(tmp_path / "w")
        seed.write_page("pricing", title="Pricing", body="Cost-plus 35%.\n")
        seed.close()
        shutil.rmtree(seed.root / ".outmem")  # curator delivered without the cache
        assert not (seed.root / ".outmem").exists()

        store = WikiStore.open(seed.root, read_only=True)
        # The store works — reads, search, backlinks — without ever
        # creating .outmem/.
        assert store.read("pricing").title == "Pricing"
        assert store.backlinks("pricing") == ()
        store.search("Cost-plus")
        assert not (seed.root / ".outmem").exists()

    def test_backlinks_memo_only_when_read_only(self, tmp_path: Path) -> None:
        """``BacklinkCache`` keeps the in-memory memo but skips the
        ``.outmem/backlinks.json`` persist when read-only. The first
        call computes; subsequent calls hit the memo; the on-disk file
        is never written."""
        import shutil

        seed = WikiStore.init(tmp_path / "w")
        seed.write_page("a", title="A", body="[[b]]\n")
        seed.write_page("b", title="B", body="just b.\n")
        seed.close()
        shutil.rmtree(seed.root / ".outmem", ignore_errors=True)

        store = WikiStore.open(seed.root, read_only=True)
        # First lookup forces a rebuild.
        assert "a" in store.backlinks("b")
        # Second lookup hits the in-memory memo (no disk side effect).
        assert "a" in store.backlinks("b")
        # The on-disk cache file was never written.
        assert not (seed.root / ".outmem" / "backlinks.json").exists()
        assert not (seed.root / ".outmem").exists()

    def test_backlinks_writes_persist_when_writable(self, tmp_path: Path) -> None:
        """Sanity-check the inverse: a writable store DOES persist the
        backlinks cache. Pins the behaviour we're conditionally
        suppressing so a future change to BacklinkCache doesn't silently
        flip both modes."""
        import shutil

        seed = WikiStore.init(tmp_path / "w")
        seed.write_page("a", title="A", body="[[b]]\n")
        seed.write_page("b", title="B", body="just b.\n")
        seed.close()
        shutil.rmtree(seed.root / ".outmem")

        store = WikiStore.open(seed.root)
        store.backlinks("b")
        assert (seed.root / ".outmem" / "backlinks.json").exists()


class TestRepairPages:
    """End-to-end of `WikiStore.repair_pages` — detects malformed-frontmatter
    pages, reports them in dry_run mode, repairs + commits them otherwise.

    The targeted shape: a top-level scalar containing an unquoted ': ',
    which YAML rejects with 'mapping values are not allowed here'.
    """

    def _seed_broken_page(self, store: WikiStore, slug: str, title: str) -> Path:
        """Write a page bypassing serialize_wiki_page (which auto-quotes), so
        we land an actually-broken file on disk — the shape we see in real
        imports. Uses the conftest's unsigned git runner so the
        signing-by-default sandbox doesn't poison the test."""
        from outmem.slug import slug_to_relpath
        from tests.conftest import _run_git

        path = store.pages_path / slug_to_relpath(slug)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"---\ntitle: {title}\nslug: {slug}\n---\n\nbody\n",
            encoding="utf-8",
        )
        _run_git(["add", str(path)], cwd=store.root)
        _run_git(["commit", "-m", f"seed broken {slug}"], cwd=store.root)
        return path

    def test_dry_run_reports_without_writing(self, fresh_store: WikiStore) -> None:
        fresh_store.write_page("good", title="Good page", body="ok\n")
        path = self._seed_broken_page(
            fresh_store, "rki:flu",
            "Influenza (Teil 1): Erkrankungen durch saisonale Influenza",
        )
        before = path.read_text(encoding="utf-8")

        report = fresh_store.repair_pages(dry_run=True)
        assert [slug for slug, _ in report] == ["rki:flu"]
        assert path.read_text(encoding="utf-8") == before  # untouched

    def test_repairs_and_commits(self, fresh_store: WikiStore) -> None:
        self._seed_broken_page(
            fresh_store, "rki:flu",
            "Influenza (Teil 1): Erkrankungen durch saisonale Influenza",
        )
        head_before = current_head(fresh_store.root)

        report = fresh_store.repair_pages(dry_run=False)
        assert [slug for slug, _ in report] == ["rki:flu"]
        # Page now reads cleanly and preserves the original title text.
        page = fresh_store.read("rki:flu")
        assert page.title == (
            "Influenza (Teil 1): Erkrankungen durch saisonale Influenza"
        )
        # A new commit landed for the repair.
        assert current_head(fresh_store.root) != head_before

    def test_returns_empty_when_nothing_to_repair(
        self, fresh_store: WikiStore
    ) -> None:
        fresh_store.write_page("good", title="Good", body="ok\n")
        assert fresh_store.repair_pages(dry_run=True) == []
        assert fresh_store.repair_pages(dry_run=False) == []


# ---------------------------------------------------------------------------
# Discovery vs reading (issue #6) — the catalogue must not advertise a page
# `read()` will reject, and a declared slug must never become a second address.
# ---------------------------------------------------------------------------


class TestDiscoveryMatchesReading:
    def _broken_wiki(self, tmp_path: Path) -> WikiStore:
        store = WikiStore.init(tmp_path / "w")
        store.write_page("ok", title="OK", body="fine\n")
        p = store.pages_path
        (p / "SOP-Upper" / "x").mkdir(parents=True, exist_ok=True)
        (p / "SOP-Upper" / "x" / "page.md").write_text(
            "---\ntitle: T\nslug: t\n---\n\nb\n", encoding="utf-8"
        )
        (p / "has_underscore").mkdir(parents=True, exist_ok=True)
        (p / "has_underscore" / "page.md").write_text(
            "---\ntitle: T\nslug: t\n---\n\nb\n", encoding="utf-8"
        )
        (p / "noslug.md").write_text("---\ntitle: T\n---\n\nb\n", encoding="utf-8")
        (p / "moved.md").write_text(
            "---\ntitle: T\nslug: somewhere-else\n---\n\nb\n", encoding="utf-8"
        )
        return store

    def test_every_listed_slug_is_readable(self, tmp_path: Path) -> None:
        """The core invariant from issue #6: discovery was more permissive
        than reading, so an agent burned a tool call per malformed page."""
        store = self._broken_wiki(tmp_path)
        for slug in store.list_slugs():
            store.read(slug)  # must not raise

    def test_ungrammatical_slugs_are_not_advertised(self, tmp_path: Path) -> None:
        store = self._broken_wiki(tmp_path)
        listed = store.list_slugs()
        assert "SOP-Upper:x:page" not in listed
        assert "has_underscore:page" not in listed

    def test_missing_slug_is_no_longer_fatal(self, tmp_path: Path) -> None:
        """Refusing to read a findable page over a missing label was pure
        downside — the path addresses it perfectly well."""
        store = self._broken_wiki(tmp_path)
        assert "noslug" in store.list_slugs()
        assert store.read("noslug").frontmatter.slug == "noslug"

    def test_declared_slug_never_becomes_a_second_address(self, tmp_path: Path) -> None:
        """`moved.md` declaring `slug: somewhere-else` must not put an
        unopenable name in the TOC — the path stays authoritative."""
        from outmem.index import render_index

        store = self._broken_wiki(tmp_path)
        toc = render_index(store.pages_path)
        assert "[[moved]]" in toc
        assert "somewhere-else" not in toc
        store.read("moved")  # still readable by its real name

    def test_unreadable_reports_all_three_defects(self, tmp_path: Path) -> None:
        store = self._broken_wiki(tmp_path)
        reasons = dict(store.unreadable())
        assert "SOP-Upper:x:page" in reasons
        assert "not addressable" in reasons["SOP-Upper:x:page"]
        assert "moved" in reasons
        assert "somewhere-else" in reasons["moved"]

    def test_clean_wiki_reports_nothing_unreadable(self, tmp_path: Path) -> None:
        """The assertion a downstream consumer wants in its own suite."""
        store = WikiStore.init(tmp_path / "w")
        store.write_page("abx:penicillin", title="Pen", body="b\n")
        store.write_page("pricing", title="P", body="b\n")
        assert store.unreadable() == []


class TestAliases:
    """A rename shouldn't be a hard break. `aliases:` makes an old slug keep
    resolving, so inbound links and outside references (tickets, configs)
    survive a reorganisation."""

    def _renamed(self, tmp_path: Path) -> WikiStore:
        store = WikiStore.init(tmp_path / "w")
        store.write_page(
            "clinical:sepsis", title="Sepsis", body="body\n",
            extra={"aliases": ["sepsis", "old:sepsis"]},
        )
        store.write_page("ref", title="Ref", body="See [[sepsis]].\n")
        return store

    def test_read_follows_an_alias(self, tmp_path: Path) -> None:
        store = self._renamed(tmp_path)
        assert store.read("sepsis").title == "Sepsis"
        assert store.exists("old:sepsis")

    def test_read_reports_the_canonical_slug(self, tmp_path: Path) -> None:
        """Otherwise the dashboard renders alias URLs and they never decay."""
        store = self._renamed(tmp_path)
        assert store.read("sepsis").slug == "clinical:sepsis"

    def test_a_live_page_always_wins_its_own_name(self, tmp_path: Path) -> None:
        """A stale alias must never shadow a real page."""
        store = self._renamed(tmp_path)
        store.write_page("sepsis-note", title="N", body="n\n")
        assert store.resolve_slug("sepsis-note") == "sepsis-note"

    def test_extend_page_via_alias_commits_cleanly(self, tmp_path: Path) -> None:
        """Regression: resolving only in _page_path made extend_page write the
        canonical file then `git add` the alias path, which doesn't exist —
        a half-applied write."""
        store = self._renamed(tmp_path)
        store.extend_page("sepsis", body="revised\n")
        assert "revised" in store.read("clinical:sepsis").body

    def test_history_via_alias_is_not_empty(self, tmp_path: Path) -> None:
        store = self._renamed(tmp_path)
        assert store.history("sepsis")  # would silently be [] without resolution

    def test_backlinks_fold_onto_the_canonical_page(self, tmp_path: Path) -> None:
        """Otherwise a renamed page's referrers split across two keys."""
        store = self._renamed(tmp_path)
        assert "ref" in store.backlinks("clinical:sepsis")

    def test_writing_a_page_at_an_alias_is_refused(self, tmp_path: Path) -> None:
        """It would succeed, win resolution file-first, and silently retarget
        every [[sepsis]] link in the corpus to the new stub."""
        store = self._renamed(tmp_path)
        with pytest.raises(OutmemError, match="is an alias of"):
            store.write_page("sepsis", title="Hijack", body="x\n")

    def test_alias_link_is_not_a_broken_link(self, tmp_path: Path) -> None:
        """Without this the feature makes `outmem lint` fail CI on every
        renamed page, which defeats the point."""
        from outmem.lint import lint_wiki

        store = self._renamed(tmp_path)
        report = lint_wiki(store.wiki_path, log_dir=store.log_path, sources_dir=store.sources_path)
        assert [f for f in report.findings if f.kind == "broken-wikilink"] == []
        assert [f for f in report.findings if f.kind == "wikilink-via-alias"]

    def test_alias_linked_page_is_not_an_orphan(self, tmp_path: Path) -> None:
        from outmem.lint import lint_wiki

        store = self._renamed(tmp_path)
        report = lint_wiki(store.wiki_path, log_dir=store.log_path, sources_dir=store.sources_path)
        orphans = {f.path for f in report.findings if f.kind == "orphan-page"}
        assert not any("sepsis" in p for p in orphans)

    def test_alias_collisions_are_lint_errors(self, tmp_path: Path) -> None:
        from outmem.lint import lint_wiki

        store = WikiStore.init(tmp_path / "w")
        store.write_page("a", title="A", body="a\n", extra={"aliases": ["shared"]})
        store.write_page("b", title="B", body="b\n", extra={"aliases": ["shared"]})
        store.write_page("c", title="C", body="c\n", extra={"aliases": ["a"]})
        report = lint_wiki(store.wiki_path, log_dir=store.log_path, sources_dir=store.sources_path)
        kinds = {f.kind for f in report.findings}
        assert "alias-conflict" in kinds
        assert "alias-shadowed" in kinds


class TestRenamePage:
    """`outmem rename` — file, frontmatter, inbound links and the alias, in
    one commit. Both bugs from a real hand-rolled rename script are pinned."""

    def _wiki(self, tmp_path: Path) -> WikiStore:
        store = WikiStore.init(tmp_path / "w")
        store.write_page("rki:ratgeber:influenza", title="Flu", body="body\n")
        store.write_page("ref", title="Ref", body="See [[rki:ratgeber:influenza]].\n")
        store.append_log(topic="t", content="- see [[rki:ratgeber:influenza]]\n")
        return store

    def test_moves_file_and_rewrites_links(self, tmp_path: Path) -> None:
        store = self._wiki(tmp_path)
        store.rename_page("rki:ratgeber:influenza", "clinical:influenza")
        assert store.read("clinical:influenza").title == "Flu"
        assert "[[clinical:influenza]]" in store.read("ref").body
        assert not (store.pages_path / "rki" / "ratgeber" / "influenza.md").exists()

    def test_multi_segment_namespace_keeps_every_segment(self, tmp_path: Path) -> None:
        """Regression: a hand-rolled slug_to_path dropped the FIRST segment,
        writing rki:ratgeber:x to ratgeber/x.md."""
        store = self._wiki(tmp_path)
        store.rename_page("rki:ratgeber:influenza", "a:b:c:d")
        assert (store.pages_path / "a" / "b" / "c" / "d.md").is_file()

    def test_old_slug_still_resolves_via_alias(self, tmp_path: Path) -> None:
        store = self._wiki(tmp_path)
        store.rename_page("rki:ratgeber:influenza", "clinical:influenza")
        assert store.read("rki:ratgeber:influenza").slug == "clinical:influenza"

    def test_prose_containing_the_slug_is_untouched(self, tmp_path: Path) -> None:
        """Regression: a text-substituting script also rewrote prose, and its
        guard against that blocked a legitimate case (`clinical:x: dose ...`
        where the second colon is punctuation)."""
        store = self._wiki(tmp_path)
        store.write_page(
            "notes", title="N",
            body="rki:ratgeber:influenza: 3 mg/kg laut [[rki:ratgeber:influenza]]\n",
        )
        store.rename_page("rki:ratgeber:influenza", "clinical:influenza")
        body = store.read("notes").body
        assert "rki:ratgeber:influenza: 3 mg/kg" in body  # prose untouched
        assert "[[clinical:influenza]]" in body  # link rewritten

    def test_does_not_rewrite_a_longer_slug_that_shares_the_prefix(
        self, tmp_path: Path
    ) -> None:
        store = self._wiki(tmp_path)
        store.write_page("rki:ratgeber:influenza:kinder", title="K", body="k\n")
        store.write_page("r2", title="R2", body="See [[rki:ratgeber:influenza:kinder]].\n")
        store.rename_page("rki:ratgeber:influenza", "clinical:influenza")
        assert "[[rki:ratgeber:influenza:kinder]]" in store.read("r2").body

    def test_rewrites_the_log_too(self, tmp_path: Path) -> None:
        store = self._wiki(tmp_path)
        store.rename_page("rki:ratgeber:influenza", "clinical:influenza")
        logs = "\n".join(p.read_text() for p in store.log_path.rglob("*.md"))
        assert "[[clinical:influenza]]" in logs

    def test_lands_as_one_commit_and_lints_clean(self, tmp_path: Path) -> None:
        from outmem.lint import lint_wiki

        store = self._wiki(tmp_path)
        before = store.head()
        store.rename_page("rki:ratgeber:influenza", "clinical:influenza")
        assert len(store.history("clinical:influenza")) >= 1
        assert store.head() != before
        report = lint_wiki(store.wiki_path, log_dir=store.log_path, sources_dir=store.sources_path)
        assert [f for f in report.findings if f.severity == "error"] == []

    def test_refuses_an_occupied_target(self, tmp_path: Path) -> None:
        store = self._wiki(tmp_path)
        store.write_page("taken", title="T", body="t\n")
        with pytest.raises(OutmemError, match="already exists"):
            store.rename_page("rki:ratgeber:influenza", "taken")

    def test_display_text_is_preserved(self, tmp_path: Path) -> None:
        store = self._wiki(tmp_path)
        store.write_page("d", title="D", body="See [[rki:ratgeber:influenza|Grippe]].\n")
        store.rename_page("rki:ratgeber:influenza", "clinical:influenza")
        assert "[[clinical:influenza|Grippe]]" in store.read("d").body
