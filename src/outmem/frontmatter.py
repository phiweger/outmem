"""Wiki page frontmatter — parse, serialise, validate.

The page model is specified in ``docs/spec.md`` §4 (v0.5):

.. code-block:: yaml

    ---
    title: Pricing formula
    slug: pricing-formula
    provenance:                # source pointers into raw/
      - raw/pricing-deck-2026-Q1.md
      - raw/acme-msa.md
    created: 2026-04-12T09:14:00Z
    updated: 2026-05-04T11:32:00Z
    tags: [pricing, contracts, finance]
    ---

There is no ``authority`` field in v0.1 (dropped at spec v0.5). Provenance
entries are either plain path strings or dicts carrying upstream-ingestion
metadata (Drive paths, content hashes, page ranges). The agent preserves
upstream-supplied entries verbatim during compaction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import yaml

from outmem._time import ensure_utc, format_iso_z, parse_iso_z, utc_now
from outmem.exceptions import FrontmatterError

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", re.DOTALL)

# Provenance entries may be plain strings or dicts (upstream-ingestion metadata).
ProvenanceEntry = str | dict[str, Any]


@dataclass
class WikiFrontmatter:
    """Structured representation of a wiki page's YAML frontmatter.

    Required fields are ``title`` and ``slug``. Everything else is optional.
    Extra keys not in the named fields are preserved in ``extra`` so the
    agent does not accidentally drop ingestion-supplied metadata.
    """

    title: str
    slug: str
    provenance: list[ProvenanceEntry] = field(default_factory=list)
    created: datetime | None = None
    updated: datetime | None = None
    tags: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


def parse_wiki_page(content: str) -> tuple[WikiFrontmatter, str]:
    """Split a wiki page into frontmatter + body.

    The frontmatter block must be at the very start of the file and
    delimited by ``---`` lines.

    Raises:
        FrontmatterError: If the block is missing, malformed, or fails
            validation (missing required ``title`` or ``slug``).
    """
    match = _FRONTMATTER_RE.match(content)
    if not match:
        raise FrontmatterError("Wiki page is missing the YAML frontmatter block.")

    raw_yaml = match.group(1)
    body = match.group(2)

    try:
        data = yaml.safe_load(raw_yaml)
    except yaml.YAMLError as exc:
        raise FrontmatterError(f"Frontmatter YAML failed to parse: {exc}") from exc

    if not isinstance(data, dict):
        raise FrontmatterError(f"Frontmatter must be a YAML mapping, got {type(data).__name__}.")

    return _frontmatter_from_dict(data), body


def serialize_wiki_page(frontmatter: WikiFrontmatter, body: str) -> str:
    """Render a frontmatter + body pair back to the on-disk file form.

    Round-trips: parsing the output of this function yields the same
    ``WikiFrontmatter`` instance (modulo dict key ordering inside ``extra``).
    """
    data = _frontmatter_to_dict(frontmatter)
    yaml_text = yaml.dump(
        data,
        Dumper=_OutmemDumper,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
    ).rstrip("\n")

    # Keep a single blank line between the frontmatter and the body so the
    # files render predictably in Obsidian and markdown-it-py.
    body_text = body.lstrip("\n")
    return f"---\n{yaml_text}\n---\n\n{body_text}"


def touch_updated(frontmatter: WikiFrontmatter, *, now: datetime | None = None) -> None:
    """Set ``updated`` to the current UTC time (or the supplied value).

    Mutates the frontmatter in place. Use before writing back a page so
    ``updated`` reflects the most recent edit.
    """
    frontmatter.updated = now.replace(microsecond=0) if now else utc_now()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


_KNOWN_FIELDS = {"title", "slug", "provenance", "created", "updated", "tags"}


def _frontmatter_from_dict(data: dict[str, Any]) -> WikiFrontmatter:
    title = data.get("title")
    slug = data.get("slug")
    if not isinstance(title, str) or not title.strip():
        raise FrontmatterError("Frontmatter is missing a non-empty 'title'.")
    if not isinstance(slug, str) or not slug.strip():
        raise FrontmatterError("Frontmatter is missing a non-empty 'slug'.")

    provenance = _coerce_provenance(data.get("provenance", []))
    created = _coerce_datetime(data.get("created"), field_name="created")
    updated = _coerce_datetime(data.get("updated"), field_name="updated")
    tags = _coerce_tags(data.get("tags", []))
    extra = {k: v for k, v in data.items() if k not in _KNOWN_FIELDS}

    return WikiFrontmatter(
        title=title,
        slug=slug,
        provenance=provenance,
        created=created,
        updated=updated,
        tags=tags,
        extra=extra,
    )


def _frontmatter_to_dict(frontmatter: WikiFrontmatter) -> dict[str, Any]:
    data: dict[str, Any] = {
        "title": frontmatter.title,
        "slug": frontmatter.slug,
    }
    if frontmatter.provenance:
        data["provenance"] = list(frontmatter.provenance)
    if frontmatter.created is not None:
        data["created"] = frontmatter.created
    if frontmatter.updated is not None:
        data["updated"] = frontmatter.updated
    if frontmatter.tags:
        data["tags"] = list(frontmatter.tags)
    data.update(frontmatter.extra)
    return data


def _coerce_provenance(value: Any) -> list[ProvenanceEntry]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise FrontmatterError(f"'provenance' must be a list, got {type(value).__name__}.")
    out: list[ProvenanceEntry] = []
    for entry in value:
        if isinstance(entry, str):
            out.append(entry)
        elif isinstance(entry, dict):
            out.append(dict(entry))
        else:
            raise FrontmatterError(
                f"Provenance entries must be strings or mappings, got {type(entry).__name__}."
            )
    return out


def _coerce_datetime(value: Any, *, field_name: str) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return ensure_utc(value)
    if isinstance(value, str):
        try:
            return parse_iso_z(value.strip())
        except ValueError as exc:
            raise FrontmatterError(
                f"'{field_name}' is not a valid ISO-8601 timestamp: {value!r}"
            ) from exc
    raise FrontmatterError(
        f"'{field_name}' must be a datetime or ISO-8601 string, got {type(value).__name__}."
    )


def _coerce_tags(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise FrontmatterError(f"'tags' must be a list, got {type(value).__name__}.")
    out: list[str] = []
    for tag in value:
        if not isinstance(tag, str):
            raise FrontmatterError(f"Tags must be strings, got {type(tag).__name__}.")
        out.append(tag)
    return out


class _OutmemDumper(yaml.SafeDumper):
    """Custom YAML dumper that emits datetimes as plain ISO-8601 ``Z`` strings.

    PyYAML's default datetime representation uses ``YYYY-MM-DD HH:MM:SS+00:00``
    (space separator, ``+00:00`` offset). We prefer the form the spec
    example uses — ``YYYY-MM-DDTHH:MM:SSZ`` — because it matches Obsidian
    and round-trips through :func:`datetime.fromisoformat`.
    """


def _represent_datetime(dumper: yaml.SafeDumper, data: datetime) -> yaml.nodes.ScalarNode:
    return dumper.represent_scalar("tag:yaml.org,2002:timestamp", format_iso_z(data))


_OutmemDumper.add_representer(datetime, _represent_datetime)
