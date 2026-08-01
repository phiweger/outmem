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
from datetime import date, datetime
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


def parse_wiki_page(
    content: str, *, fallback_slug: str | None = None
) -> tuple[WikiFrontmatter, str]:
    """Split a wiki page into frontmatter + body.

    The frontmatter block must be at the very start of the file and
    delimited by ``---`` lines.

    ``fallback_slug`` is used when the page declares no ``slug:``. The
    path is what actually addresses a page — ``slug:`` is a *declaration
    of intent to check the path against*, not the address — so refusing
    to read an otherwise-findable page over a missing label is pure
    downside. Callers that know where the page lives (the loaders,
    ``WikiStore.read``) pass it; callers parsing free-floating text don't,
    and keep the strict behaviour.

    Raises:
        FrontmatterError: If the block is missing, malformed, or fails
            validation (missing ``title``, or missing ``slug`` with no
            ``fallback_slug`` to stand in).
    """
    match = _FRONTMATTER_RE.match(content)
    if not match:
        raise FrontmatterError("Wiki page is missing the YAML frontmatter block.")

    raw_yaml = match.group(1)
    body = match.group(2)

    try:
        data = yaml.safe_load(raw_yaml)
    except Exception as exc:
        # Deliberately broader than yaml.YAMLError: PyYAML's implicit
        # timestamp resolver matches `2026-02-30`, then calls
        # datetime.date(2026, 2, 30) and raises a bare ValueError. Letting
        # that escape kills a whole `outmem reindex` (and blocks writeback,
        # which must never fail on a bad page). Every parse failure this
        # function can produce is a FrontmatterError.
        raise FrontmatterError(f"Frontmatter YAML failed to parse: {exc}") from exc

    if not isinstance(data, dict):
        raise FrontmatterError(f"Frontmatter must be a YAML mapping, got {type(data).__name__}.")

    return _frontmatter_from_dict(data, raw_yaml, fallback_slug=fallback_slug), body


# Matches a top-level frontmatter line `<key>: <value>` where <value> is a
# bare scalar (no leading list/anchor/quote, no block indicator). The
# repair pass only touches lines like these — nested keys, lists, and
# already-quoted values are passed through verbatim.
_TOPLEVEL_SCALAR_RE = re.compile(
    r"^([A-Za-z_][A-Za-z0-9_-]*):[ \t]+([^>|\n][^\n]*)$"
)


def repair_wiki_page(content: str) -> str | None:
    """Best-effort repair of a frontmatter block YAML can't load.

    Targets the single failure mode we see in imported wikis: a top-level
    scalar value (most often ``title:``) that contains an unquoted
    ``: `` — ``title: Influenza (Teil 1): Erkrankungen…`` — which YAML
    interprets as a nested mapping and rejects with "mapping values are
    not allowed here". The repair single-quotes such values and only
    accepts the result if it now parses cleanly. Leaves nested structures
    and already-quoted lines alone.

    Returns the repaired full page text on success; ``None`` if the
    repair didn't fire (file already parses) or didn't help (file is
    broken in a way we don't pretend to fix — e.g. mis-indented blocks,
    truncated frontmatter). Conservative on purpose: silently rewriting
    arbitrary YAML risks masking other bugs."""
    match = _FRONTMATTER_RE.match(content)
    if not match:
        return None
    raw_yaml = match.group(1)
    body = match.group(2)

    # Already-parseable file: no repair needed. Catch broadly for the same
    # reason parse_wiki_page does — PyYAML's timestamp constructor raises a
    # bare ValueError on `created: 2026-02-30`, which is NOT a YAMLError.
    # Letting it escape here turns "this page needs repair" into a crash in
    # every caller of the repair path (read_page, the TOC, the backlink
    # graph, the indexer).
    try:
        yaml.safe_load(raw_yaml)
        return None
    except Exception:
        pass

    lines = raw_yaml.split("\n")
    changed = False
    for i, line in enumerate(lines):
        m = _TOPLEVEL_SCALAR_RE.match(line)
        if not m:
            continue
        key, value = m.group(1), m.group(2).rstrip()
        # Only quote if the value contains the trigger and isn't already quoted.
        if ": " not in value:
            continue
        if value.startswith(("'", '"')):
            continue
        escaped = value.replace("'", "''")  # YAML single-quote escape
        lines[i] = f"{key}: '{escaped}'"
        changed = True
    if not changed:
        return None

    repaired_yaml = "\n".join(lines)
    try:
        yaml.safe_load(repaired_yaml)
    except Exception:
        # Broad on purpose: only accept a repair that genuinely loads. A
        # narrow `except yaml.YAMLError` would let a page whose repaired
        # YAML still raises (e.g. an invalid calendar date) be returned as
        # "repaired", moving the crash to the caller instead of declining.
        return None  # didn't fix it; leave the file alone
    return f"---\n{repaired_yaml}\n---\n\n{body.lstrip(chr(10))}"


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


def _raw_tag_text(raw_yaml: str) -> list[str] | None:
    """The *as-written* text of each entry in the top-level ``tags:`` list.

    ``yaml.safe_load`` resolves scalars before we can see them, and the
    resolution is lossy in ways that matter for tags: ``007`` becomes the
    int 7, ``12:30`` becomes 750 (YAML 1.1 sexagesimal), ``0x1F`` becomes
    31, ``yes`` becomes True. Rendering *those* back with ``str()`` would
    write a tag the author never typed — and a later ``write_page`` would
    persist the corruption to disk.

    Composing the node tree instead gives us ``ScalarNode.value``, which is
    the original lexical text, so a non-string tag can be recovered exactly
    as authored. Returns ``None`` when the shape isn't a plain sequence of
    scalars (then the caller falls back to validating the parsed values).
    """
    try:
        node = yaml.compose(raw_yaml)
    except Exception:
        return None
    if not isinstance(node, yaml.MappingNode):
        return None
    for key_node, value_node in node.value:
        if getattr(key_node, "value", None) != "tags":
            continue
        if not isinstance(value_node, yaml.SequenceNode):
            return None
        out: list[str] = []
        for item in value_node.value:
            if not isinstance(item, yaml.ScalarNode):
                return None
            out.append(item.value)
        return out
    return None


def _frontmatter_from_dict(
    data: dict[str, Any], raw_yaml: str = "", *, fallback_slug: str | None = None
) -> WikiFrontmatter:
    title = data.get("title")
    slug = data.get("slug")
    if not isinstance(title, str) or not title.strip():
        raise FrontmatterError("Frontmatter is missing a non-empty 'title'.")
    if not isinstance(slug, str) or not slug.strip():
        if fallback_slug:
            slug = fallback_slug
        else:
            raise FrontmatterError("Frontmatter is missing a non-empty 'slug'.")

    provenance = _coerce_provenance(data.get("provenance", []))
    created = _coerce_datetime(data.get("created"), field_name="created")
    updated = _coerce_datetime(data.get("updated"), field_name="updated")
    tags = _coerce_tags(data.get("tags", []), _raw_tag_text(raw_yaml))
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
    # A bare ``created: 2026-07-23`` (no time) is loaded by PyYAML as a
    # ``date``, which is NOT a ``datetime`` (the subclass relation runs the
    # other way). Imported wikis write date-only stamps constantly, and the
    # intent is unambiguous — promote to midnight UTC rather than rejecting
    # the page. Checked before ``str`` since ``date`` is not a string.
    if isinstance(value, date):
        return ensure_utc(datetime(value.year, value.month, value.day))
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


def _coerce_tags(value: Any, raw_text: list[str] | None = None) -> list[str]:
    """Normalise the ``tags:`` list, keeping every tag exactly as authored.

    An unquoted year, ICD code, indicator ID or ``yes`` is valid YAML that
    resolves to a non-string. Raising on those used to make the semantic
    indexer drop the entire page, so instead we take the tag's original
    text from ``raw_text`` (see :func:`_raw_tag_text`) — never ``str()`` of
    the resolved value, which would rewrite ``007`` as ``7``.
    """
    if value is None:
        return []
    if not isinstance(value, list):
        raise FrontmatterError(f"'tags' must be a list, got {type(value).__name__}.")
    usable_raw = raw_text if raw_text is not None and len(raw_text) == len(value) else None
    out: list[str] = []
    for i, tag in enumerate(value):
        if isinstance(tag, str):
            out.append(tag)
            continue
        if usable_raw is not None:
            out.append(usable_raw[i])
            continue
        # No lexical text recovered (unusual shape, e.g. an anchor or a
        # nested collection). Refuse rather than guess at the author's
        # intent — str() here is exactly the corruption we're avoiding.
        raise FrontmatterError(
            f"Tags must be strings, got {type(tag).__name__}. Quote the tag."
        )
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
