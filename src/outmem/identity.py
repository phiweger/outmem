"""Parse ``CONTRIBUTORS.md`` into a structured identity map.

The file lives at the wiki root and lists every team member known to
the steering loop (planning prompt phase 1). The agent reads it on
startup and uses it to recognise authors in ``git log`` — unknown
authors degrade the steering signal but do not block runs (spec v0.5
§3).

Schema (one team member per top-level list entry)::

    - Alice Liddell <alice@example.com>
    - Bob Roberts <bob@example.com> [aliases: bob@personal.dev, b.roberts@example.com]
    - The Agent <agent@host>

The lookup index includes both the primary email and every alias, so
``Contributors.lookup("bob@personal.dev")`` returns Bob's record
regardless of which address the commit was made under.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Contributor:
    """A single team member identity."""

    name: str
    primary_email: str
    aliases: tuple[str, ...] = field(default_factory=tuple)

    @property
    def all_emails(self) -> tuple[str, ...]:
        return (self.primary_email, *self.aliases)


@dataclass
class Contributors:
    """Collection of known team identities.

    The lookup methods are case-insensitive on the local-part-and-domain
    of the email because git's own email handling is case-insensitive in
    practice (most providers normalise) and we do not want capitalisation
    drift in a CONTRIBUTORS.md edit to silently break the steering loop.
    """

    entries: list[Contributor]
    _index: dict[str, Contributor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._index = {}
        for entry in self.entries:
            for email in entry.all_emails:
                self._index[email.lower()] = entry

    def lookup(self, email: str) -> Contributor | None:
        if not isinstance(email, str):
            return None
        return self._index.get(email.lower())


# A line we recognise: ``- Name <email> [aliases: a@x, b@y]`` (aliases optional).
_ENTRY_RE = re.compile(
    r"""
    ^\s*-\s+              # leading "- " bullet
    (?P<name>[^<]+?)\s*   # display name (greedy up to the angle bracket)
    <(?P<email>[^>\s]+)>  # primary email
    (?:\s+\[\s*aliases?\s*:\s*(?P<aliases>[^\]]+?)\s*\])?  # optional aliases block
    \s*$
    """,
    re.VERBOSE,
)


def parse_contributors(text: str) -> Contributors:
    """Parse ``CONTRIBUTORS.md`` content into a :class:`Contributors`.

    Lines that don't match the bullet schema are silently ignored — the
    file is human-edited markdown and may carry headers, blank lines,
    or freeform commentary.
    """
    entries: list[Contributor] = []
    for raw_line in text.splitlines():
        match = _ENTRY_RE.match(raw_line)
        if not match:
            continue
        name = match.group("name").strip()
        email = match.group("email").strip()
        aliases_text = match.group("aliases") or ""
        aliases = tuple(alias.strip() for alias in aliases_text.split(",") if alias.strip())
        if not name or not email:
            continue
        entries.append(Contributor(name=name, primary_email=email, aliases=aliases))
    return Contributors(entries=entries)


def load_contributors(path: Path) -> Contributors:
    """Read and parse ``CONTRIBUTORS.md`` from disk.

    Returns an empty :class:`Contributors` if the file is missing, so
    a wiki without one still works (every author will be unknown and
    surface as a warning, but the steering loop continues).
    """
    if not path.exists():
        return Contributors(entries=[])
    return parse_contributors(path.read_text(encoding="utf-8"))
