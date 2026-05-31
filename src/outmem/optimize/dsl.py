"""Controlled-vocabulary mini-DSL for ``retrieval.strategy``.

The optimizer's :class:`~outmem.optimize.blocks.RetrievalConfig` has many
knobs; production users only care about *which retrieval pipeline*. This
module exposes one string field that names the pipeline and parses it
into a config dict.

Vocabulary (every unsupported combination raises :class:`OutmemError`):

* ``lexical`` / ``bm25`` / ``semantic`` / ``hyde`` — atomic strategies
* ``rerank`` — short for ``rerank(lexical)``
* ``rerank(<source>)`` — LLM yes/no gate over a non-rerank atomic source
* ``a+b[+c…]`` (≥2 atomic legs) — RRF-fused hybrid with those legs

The split between atomic-source (parens) and fusion (``+``) keeps the
grammar context-free and lets us reject `bm25+rerank(semantic)`-style
oddities deterministically — kept out of the DSL even though the runtime
could technically build them, to honour the "controlled vocabulary"
contract.
"""

from __future__ import annotations

import re
from typing import Any

from outmem.exceptions import OutmemError

# Legs you can name in a fuse OR pass as a `rerank` source. NOT ``rerank``
# itself — no recursive rerank-over-rerank, and rerank-as-a-leg is excluded
# from the DSL on purpose (it complicates the grammar without any known win).
_DSL_ATOMICS = ("lexical", "bm25", "semantic", "hyde")

_RERANK_RE = re.compile(r"^rerank\(([a-z0-9_]+)\)$")


def parse_strategy(spec: str) -> dict[str, Any]:
    """Turn a DSL string into a partial ``RetrievalConfig`` dict.

    Returns ``{"strategy": ..., ...}`` ready to splat into
    :meth:`RetrievalConfig.from_dict`. Numeric knobs aren't touched here —
    they live on :class:`~outmem.config.RetrievalSettings` and get merged
    by the caller.
    """
    if not isinstance(spec, str) or not spec.strip():
        raise OutmemError("retrieval.strategy must be a non-empty string")
    text = spec.strip().lower()

    if "+" in text:
        legs = [leg.strip() for leg in text.split("+")]
        if len(legs) < 2:
            raise OutmemError(
                f"hybrid strategy {spec!r} needs >=2 legs joined with '+'"
            )
        for leg in legs:
            if leg not in _DSL_ATOMICS:
                raise OutmemError(
                    f"hybrid leg {leg!r} not in controlled vocabulary "
                    f"{_DSL_ATOMICS}; got {spec!r}"
                )
        if len(set(legs)) != len(legs):
            raise OutmemError(f"hybrid strategy {spec!r} has duplicate legs")
        return {"strategy": "hybrid", "fuse": legs}

    if text == "rerank":
        return {"strategy": "rerank", "rerank_source": "lexical"}
    if m := _RERANK_RE.match(text):
        source = m.group(1)
        if source not in _DSL_ATOMICS:
            raise OutmemError(
                f"rerank source {source!r} not in {_DSL_ATOMICS}; got {spec!r}"
            )
        return {"strategy": "rerank", "rerank_source": source}

    if text in _DSL_ATOMICS:
        return {"strategy": text}

    vocabulary = (*_DSL_ATOMICS, "rerank", "rerank(<source>)", "a+b[+c…]")
    raise OutmemError(
        f"unknown retrieval strategy {spec!r}; vocabulary: {vocabulary}"
    )


def format_strategy(cfg_dict: dict[str, Any]) -> str:
    """Inverse of :func:`parse_strategy` — render a config dict as the DSL.

    Used when the optimizer writes its winning config to
    ``retrieval.yaml``; the file shows the same string a user would type
    by hand, so a `git diff` is readable.
    """
    strategy = cfg_dict.get("strategy", "lexical")
    if strategy == "hybrid":
        legs = cfg_dict.get("fuse") or ()
        if len(legs) < 2:
            raise OutmemError(
                f"hybrid config has <2 fuse legs: {list(legs)!r}"
            )
        return "+".join(str(leg) for leg in legs)
    if strategy == "rerank":
        source = cfg_dict.get("rerank_source", "lexical")
        return f"rerank({source})"
    return str(strategy)
