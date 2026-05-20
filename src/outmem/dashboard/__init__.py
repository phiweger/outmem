"""Read-only FastAPI dashboard for an outmem wiki.

Optional extra — install ``outmem[dashboard]`` to pull FastAPI, Uvicorn,
markdown-it-py, and Jinja2 into the same environment.

Two integration shapes:

* :func:`create_app` builds a standalone FastAPI app that serves the
  wiki and nothing else. Useful for ``outmem dashboard`` / local dev.
* :func:`router_for` returns an :class:`fastapi.APIRouter` you mount
  on your own app, behind your own auth. The router carries the same
  routes (``/wiki``, ``/wiki/{path:path}``, etc.).

Both rely on a :class:`outmem.store.WikiStore` passed in at construction
time. The dashboard is read-only by design (spec v0.5 §5): editing
happens through Obsidian against a local clone, never through the dashboard.
"""

from __future__ import annotations

from outmem.dashboard.app import create_app
from outmem.dashboard.router import router_for

__all__ = ["create_app", "router_for"]
