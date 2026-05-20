"""Internal datetime helpers — single source for ISO-8601 ``Z`` formatting.

Several modules persist UTC datetimes as ``"2026-05-12T10:00:00Z"``
strings (frontmatter YAML, ``.sources.db`` rows, ``.outmem`` state,
semantic metadata). They used to hand-roll the same ``astimezone`` +
microsecond-strip + ``+00:00`` ↔ ``Z`` swap dance; this module is
the single place to do it so the spelling can't drift.
"""

from __future__ import annotations

from datetime import UTC, datetime


def utc_now() -> datetime:
    """Current UTC time, seconds resolution, tz-aware."""
    return datetime.now(UTC).replace(microsecond=0)


def ensure_utc(ts: datetime) -> datetime:
    """Return ``ts`` as a UTC-aware datetime; naive inputs are assumed UTC."""
    aware = ts if ts.tzinfo else ts.replace(tzinfo=UTC)
    return aware.astimezone(UTC)


def format_iso_z(ts: datetime) -> str:
    """Serialise an aware (or naive-UTC) datetime as ``"...Z"``."""
    aware = ts if ts.tzinfo else ts.replace(tzinfo=UTC)
    return (
        aware.astimezone(UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def parse_iso_z(text: str) -> datetime:
    """Parse an ISO-8601 timestamp into an aware UTC datetime.

    Accepts the ``Z`` suffix as a shorthand for ``+00:00``. Raises
    :class:`ValueError` for inputs ``datetime.fromisoformat`` rejects.
    """
    normalised = text[:-1] + "+00:00" if text.endswith("Z") else text
    parsed = datetime.fromisoformat(normalised)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
