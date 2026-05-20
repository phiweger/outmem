"""Tests for ``outmem.dashboard``.

The dashboard is mounted into a FastAPI app and exercised via httpx's
TestClient. We cover routing, markdown + wikilink rendering, the
backlinks panel, the per-page history view, and the security boundary
(html disabled in markdown-it).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from outmem.dashboard import create_app, router_for
from outmem.dashboard.service import (
    build_renderer,
    render_body,
    wikilinks_to_markdown,
)
from outmem.store import WikiStore

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def seeded_store(tmp_path: Path) -> WikiStore:
    store = WikiStore.init(tmp_path / "wiki")
    store.write_page(
        "pricing-formula",
        title="Pricing formula",
        body="The pricing formula is cost-plus 35%.\n\nSee [[acme-msa]] for terms.\n",
        provenance=["raw/pricing-deck-2026-Q1.md"],
        tags=["pricing", "contracts"],
    )
    store.write_page(
        "acme-msa",
        title="Acme MSA",
        body="Standard Master Service Agreement. See [[pricing-formula]].\n",
    )
    return store


@pytest.fixture
def client(seeded_store: WikiStore) -> TestClient:
    return TestClient(create_app(seeded_store))


# ---------------------------------------------------------------------------
# Service-layer helpers
# ---------------------------------------------------------------------------


def test_wikilinks_to_markdown_simple() -> None:
    body = "see [[pricing-formula]]."
    out = wikilinks_to_markdown(body)
    assert "[pricing-formula](/wiki/pricing-formula)" in out


def test_wikilinks_to_markdown_with_display() -> None:
    body = "see [[pricing-formula|the pricing page]]."
    out = wikilinks_to_markdown(body)
    assert "[the pricing page](/wiki/pricing-formula)" in out


def test_wikilinks_to_markdown_keeps_invalid_literal() -> None:
    body = "see [[Bad Slug]]."
    assert wikilinks_to_markdown(body) == "see [[Bad Slug]]."


def test_render_body_emits_anchor() -> None:
    html = render_body("see [[acme-msa]]")
    assert '<a href="/wiki/acme-msa"' in html


def test_render_body_escapes_raw_html() -> None:
    html = render_body("paragraph with <script>alert(1)</script>")
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_markdown_it_html_disabled() -> None:
    """Defence in depth — the renderer must refuse raw HTML."""
    md = build_renderer()
    assert md.options.get("html") is False


# ---------------------------------------------------------------------------
# Index + page routing
# ---------------------------------------------------------------------------


def test_index_lists_pages(client: TestClient) -> None:
    rsp = client.get("/wiki")
    assert rsp.status_code == 200
    assert "acme-msa" in rsp.text
    assert "pricing-formula" in rsp.text


def test_index_redirect_from_root(client: TestClient) -> None:
    rsp = client.get("/", follow_redirects=False)
    assert rsp.status_code == 307
    assert rsp.headers["location"] == "/wiki"


def test_page_renders_body(client: TestClient) -> None:
    rsp = client.get("/wiki/pricing-formula")
    assert rsp.status_code == 200
    assert "Pricing formula" in rsp.text
    assert "cost-plus 35%" in rsp.text


def test_page_wikilink_becomes_anchor(client: TestClient) -> None:
    rsp = client.get("/wiki/pricing-formula")
    assert '<a href="/wiki/acme-msa"' in rsp.text


def test_page_backlinks_panel_lists_referrers(client: TestClient) -> None:
    rsp = client.get("/wiki/pricing-formula")
    # acme-msa links back to pricing-formula.
    assert "Backlinks" in rsp.text
    assert ">acme-msa<" in rsp.text


def test_page_no_backlinks_shows_empty_state(client: TestClient) -> None:
    rsp = client.get("/wiki/acme-msa")
    # pricing-formula links into acme-msa, so acme-msa SHOULD have a
    # backlink. Make sure that the empty-state copy is not present.
    assert "No backlinks." not in rsp.text


def test_page_renders_tags(client: TestClient) -> None:
    rsp = client.get("/wiki/pricing-formula")
    assert "pricing" in rsp.text
    assert "contracts" in rsp.text


def test_page_renders_provenance(client: TestClient) -> None:
    rsp = client.get("/wiki/pricing-formula")
    assert "raw/pricing-deck-2026-Q1.md" in rsp.text


def test_unknown_slug_404(client: TestClient) -> None:
    rsp = client.get("/wiki/no-such-page")
    assert rsp.status_code == 404


def test_unsafe_slug_400(client: TestClient) -> None:
    rsp = client.get("/wiki/Bad Slug")
    assert rsp.status_code == 400


# ---------------------------------------------------------------------------
# History view
# ---------------------------------------------------------------------------


def test_history_view_lists_commits(seeded_store: WikiStore) -> None:
    seeded_store.extend_page("pricing-formula", body="updated body.\n")
    client = TestClient(create_app(seeded_store))
    rsp = client.get("/wiki/pricing-formula/history")
    assert rsp.status_code == 200
    assert "extend: pricing-formula" in rsp.text
    assert "compact: pricing-formula" in rsp.text


def test_history_for_unknown_slug_renders_empty(client: TestClient) -> None:
    rsp = client.get("/wiki/nonexistent/history")
    assert rsp.status_code == 200
    assert "No commits yet" in rsp.text


def test_history_unsafe_slug_400(client: TestClient) -> None:
    rsp = client.get("/wiki/Bad Slug/history")
    assert rsp.status_code == 400


# ---------------------------------------------------------------------------
# Mount-into-host scenario
# ---------------------------------------------------------------------------


def test_router_can_mount_with_prefix(seeded_store: WikiStore) -> None:
    """The router_for() entrypoint mounts under any prefix; consumers
    can put it behind their own auth."""
    app = FastAPI()
    app.include_router(router_for(seeded_store), prefix="/memory")
    client = TestClient(app)

    rsp = client.get("/memory/wiki/pricing-formula")
    assert rsp.status_code == 200
    assert "Pricing formula" in rsp.text


def test_router_custom_base_path(seeded_store: WikiStore) -> None:
    app = FastAPI()
    app.include_router(router_for(seeded_store, base_path="/pages"))
    client = TestClient(app)

    rsp = client.get("/pages")
    assert rsp.status_code == 200
    assert "pricing-formula" in rsp.text

    rsp = client.get("/pages/pricing-formula")
    assert rsp.status_code == 200


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


def test_cli_dashboard_subcommand_help() -> None:
    import argparse

    from outmem.cli.__main__ import build_parser

    parser = build_parser()
    with pytest.raises((SystemExit, argparse.ArgumentError)):
        parser.parse_args(["dashboard", "--help"])
