"""Framework adapters for downstream agentic systems.

Each submodule shapes :class:`outmem.store.WikiStore` primitives into
the tool form a specific framework expects. Adapter modules are
optional and have no hard dependency edges in outmem core — installing
the relevant extra (``outmem[pydantic-ai]``) just guarantees the
framework itself is available for the consumer.
"""

from __future__ import annotations
