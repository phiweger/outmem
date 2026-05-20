"""FastAPI router for the read-only wiki dashboard.

The router is constructed with a :class:`WikiStore`; downstream apps
mount it under whatever path prefix and auth they prefer:

.. code-block:: python

    from fastapi import FastAPI
    from outmem import WikiStore
    from outmem.dashboard import router_for

    app = FastAPI()
    store = WikiStore.open("/srv/agent")
    app.include_router(router_for(store), prefix="/memory")

Routes (relative to the mount prefix)::

    GET /                      — wiki index (list of slugs)
    GET /wiki                  — wiki index (alias)
    GET /wiki/{slug}           — rendered page
    GET /wiki/{slug}/history   — git log for that page

The render path is read-only by design (spec v0.5 §5).
"""

from __future__ import annotations

from contextlib import suppress
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from jinja2 import Environment, FileSystemLoader, select_autoescape

from outmem.dashboard.service import render_body
from outmem.exceptions import FrontmatterError, OutmemError, SlugError
from outmem.store import WikiStore

_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


def _build_jinja_env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=select_autoescape(["html", "htm", "j2"]),
    )


def router_for(
    store: WikiStore,
    *,
    pull_on_request: bool = False,
    base_path: str = "/wiki",
) -> APIRouter:
    """Construct a router bound to ``store``.

    Args:
        store: The wiki the router renders.
        pull_on_request: If ``True``, run ``store.pull()`` at the start
            of every request. The simplest cache-freshness strategy
            (spec §5); use only for low-traffic dashboards. Default
            ``False`` — consumers can call ``store.pull()`` from a cron
            or webhook instead.
        base_path: URL path prefix for the wiki routes (default
            ``"/wiki"``). Combine with ``app.include_router(...,
            prefix="…")`` for richer mount layouts.
    """
    router = APIRouter()
    env = _build_jinja_env()

    def _wiki_url(slug: str) -> str:
        return f"{base_path}/{slug}"

    def _history_url(slug: str) -> str:
        return f"{base_path}/{slug}/history"

    def _template_globals() -> dict[str, object]:
        return {
            "wiki_index_url": base_path,
            "wiki_url": _wiki_url,
            "history_url": _history_url,
        }

    def _ensure_fresh() -> None:
        if pull_on_request:
            # Best-effort refresh; failures shouldn't prevent reads.
            with suppress(OutmemError):
                store.pull()

    @router.get(base_path, response_class=HTMLResponse, name="wiki_index")
    @router.get(base_path + "/", response_class=HTMLResponse, include_in_schema=False)
    async def wiki_index() -> HTMLResponse:
        _ensure_fresh()
        template = env.get_template("wiki_index.html.j2")
        html = template.render(slugs=store.list_slugs(), **_template_globals())
        return HTMLResponse(html)

    @router.get(
        base_path + "/{slug}/history",
        response_class=HTMLResponse,
        name="wiki_history",
    )
    async def wiki_history(slug: str) -> HTMLResponse:
        _ensure_fresh()
        try:
            commits = store.history(slug)
        except SlugError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        template = env.get_template("wiki_history.html.j2")
        html = template.render(
            slug=slug,
            commits=commits,
            **_template_globals(),
        )
        return HTMLResponse(html)

    @router.get(
        base_path + "/{slug}",
        response_class=HTMLResponse,
        name="wiki_page",
    )
    async def wiki_page(slug: str) -> HTMLResponse:
        _ensure_fresh()
        try:
            page = store.read(slug)
        except SlugError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FrontmatterError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except OutmemError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        backlinks = store.backlinks(slug)
        body_html = render_body(page.body, base=base_path + "/")
        rendered = {
            "slug": page.slug,
            "title": page.title,
            "body_html": body_html,
            "updated": page.frontmatter.updated,
            "backlinks": backlinks,
        }
        # Provenance entries are either strings or dicts; flatten dicts
        # to a single ``key: value`` line so the template stays simple.
        provenance: list[str] = []
        for entry in page.frontmatter.provenance:
            if isinstance(entry, dict):
                parts = ", ".join(f"{k}: {v}" for k, v in entry.items())
                provenance.append(parts)
            else:
                provenance.append(str(entry))

        template = env.get_template("wiki_page.html.j2")
        html = template.render(
            page=rendered,
            tags=page.frontmatter.tags,
            provenance=provenance,
            **_template_globals(),
        )
        return HTMLResponse(html)

    # ``GET /`` redirects to the wiki index so the dashboard feels alive
    # at the mount root.
    @router.get("/", include_in_schema=False)
    async def index_root() -> RedirectResponse:
        return RedirectResponse(url=base_path, status_code=307)

    return router
