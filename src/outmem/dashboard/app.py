"""Standalone FastAPI app — used by ``outmem dashboard``.

Returned by :func:`create_app` for cases where the dashboard is the
whole web surface (single-tenant, local dev, demo). For mounting into
an existing FastAPI app — including consumer auth — use
:func:`outmem.dashboard.router_for` directly.
"""

from __future__ import annotations

from fastapi import FastAPI

from outmem.dashboard.router import router_for
from outmem.store import WikiStore


def create_app(
    store: WikiStore,
    *,
    pull_on_request: bool = False,
    base_path: str = "/wiki",
    title: str = "outmem",
) -> FastAPI:
    """Build a FastAPI app serving the wiki at ``base_path``.

    No auth is wired in — adding it is the consumer's job. If you need
    auth on a single-tenant deployment, prefer
    :func:`outmem.dashboard.router_for` mounted under your own
    auth-gated app.
    """
    app = FastAPI(title=title, openapi_url=None)
    app.include_router(router_for(store, pull_on_request=pull_on_request, base_path=base_path))
    return app
