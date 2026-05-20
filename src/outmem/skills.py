"""Outmem-specific skill bundle metadata.

The actual skill-loading machinery (frontmatter parsing, manifest
rendering, name validation) lives in the
`outskilled <https://github.com/phiweger/outskilled>`_ package. This
module only owns:

* :data:`BUNDLED_SKILLS_DIR` — the on-disk path to outmem's bundled
  ``notes/{search,write,evolution}/SKILL.md`` files.
* :func:`bundled_registry` — a thin constructor that returns a
  :class:`outskilled.SkillRegistry` over that directory. Memoised so
  consumers can call it freely without re-walking the tree.

Callers that want a different skills root (tests, downstream
consumers) construct their own :class:`SkillRegistry` directly —
``SkillRegistry([path])`` — rather than going through this module.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from outskilled import SkillRegistry  # type: ignore[import-untyped]

# The bundled skill files ship inside the package so ``outmem skill
# install`` (and downstream consumers via :func:`bundled_registry`)
# can find them without filesystem discovery.
BUNDLED_SKILLS_DIR = Path(__file__).resolve().parent / "skills"


@lru_cache(maxsize=1)
def bundled_registry() -> SkillRegistry:
    """Return a :class:`SkillRegistry` over outmem's bundled skills.

    Memoised — the registry walks the directory on construction, so
    repeated calls (e.g. one per system-prompt render) reuse the same
    parsed skills.
    """
    return SkillRegistry([BUNDLED_SKILLS_DIR])
