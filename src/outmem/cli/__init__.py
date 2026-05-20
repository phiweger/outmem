"""Command-line interface — ``outmem <subcommand>``.

The CLI is a thin shell over :class:`outmem.store.WikiStore` so the
same primitives downstream Python consumers use are also reachable
as shell tools.
"""

from __future__ import annotations

from outmem.cli.__main__ import main

__all__ = ["main"]
