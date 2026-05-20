"""Semantic retrieval layer for outmem.

Optional v0.2 capability: index wiki pages and sources in a local
``sqlite-vec`` database so the agent can fall back to vector similarity
when ripgrep over compiled material yields nothing useful, and so
``outmem lint --semantic`` can surface near-duplicate / contradicting
chunks across pages (issue #7).

The semantic layer is opt-in (``semantic.enabled: true`` in
``config.yaml``) and ships in the ``outmem[semantic]`` extra:

    pip install "outmem[semantic]"

Storage lives at ``<root>/.vectors.db`` (tracked in git, sibling of
``wiki/``). Embeddings come from PydanticAI's
:class:`pydantic_ai.Embedder` so the choice of provider follows the
same model-id convention as the agent
(``openai:text-embedding-3-small`` by default).

Public API::

    from outmem.semantic import VectorStore, chunk_text
"""

from __future__ import annotations

from outmem.semantic.chunker import Chunk, chunk_text, hash_text
from outmem.semantic.embeddings import EmbedderHandle, build_embedder
from outmem.semantic.store import (
    DEFAULT_DB_FILENAME,
    Match,
    ReindexResult,
    VectorStore,
)

__all__ = [
    "DEFAULT_DB_FILENAME",
    "Chunk",
    "EmbedderHandle",
    "Match",
    "ReindexResult",
    "VectorStore",
    "build_embedder",
    "chunk_text",
    "hash_text",
]
