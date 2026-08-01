"""Static wiki linter — orphans, broken links, stale provenance, drift.

Read-only mechanical checks against the on-disk wiki. Catches the
class of problems that don't need an LLM:

- Pages with malformed or missing frontmatter (and pages that only parse
  after self-heal — repairable, so a warning, not an error)
- Two pages claiming the same slug
- Broken ``[[wikilink]]`` references (target slug doesn't exist)
- Slugs written as *prose* that no longer resolve — a dangling-link check
  is blind to these, and they are where dead references accumulate after
  a namespace is reorganised
- Stale provenance (cited ``raw/`` or ``sources/`` file is missing) and
  provenance citing a sha256 the registry no longer holds
- ``.sources.db`` disagreeing with what is on disk, in either direction
- Orphan pages (zero inbound wikilinks, not referenced from ``log/``)
- Index drift (``wiki/index.md`` doesn't reflect current pages — happens
  when humans edit the wiki via Obsidian without running outmem)

Semantic contradictions ("page A says X, page B says Y about the
same thing") need an LLM pass — tracked as a v0.2 deferral, see GitHub
issue #7.

Output is a :class:`LintReport` listing :class:`LintFinding` objects.
The :func:`format_report` helper renders them for human consumption.
``outmem lint`` (CLI) exits non-zero when findings exist so it can
feed straight into CI.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from outmem.frontmatter import ProvenanceEntry, parse_wiki_page
from outmem.index import (
    INDEX_FILENAME,
    INDEX_SLUG,
    editorial_pages,
    load_page_text,
    render_index,
)
from outmem.slug import (
    PAGES_DIR,
    extract_slug_references,
    extract_wikilinks,
    relpath_to_slug,
)


class Severity(StrEnum):
    """How serious a finding is.

    ``error`` — something the wiki can't render cleanly (broken
    link, missing file). ``warning`` — something that needs human
    attention but doesn't break rendering (orphan, stale provenance).
    """

    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class LintFinding:
    """A single problem identified during lint."""

    kind: str
    severity: Severity
    path: str  # repo-relative
    message: str


@dataclass
class LintReport:
    """All findings from one lint pass."""

    findings: list[LintFinding] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(f.severity == Severity.ERROR for f in self.findings)

    @property
    def has_findings(self) -> bool:
        return bool(self.findings)

    def by_kind(self) -> dict[str, list[LintFinding]]:
        groups: dict[str, list[LintFinding]] = {}
        for f in self.findings:
            groups.setdefault(f.kind, []).append(f)
        return groups


def lint_wiki(
    wiki_dir: Path,
    *,
    log_dir: Path | None = None,
    raw_dir: Path | None = None,
    sources_dir: Path | None = None,
) -> LintReport:
    """Run every static check against ``wiki_dir``.

    ``log_dir`` is consulted for orphan detection — a page mentioned
    only in ``log/<date>.md`` still counts as referenced. ``raw_dir``
    and ``sources_dir`` are consulted for stale-provenance checks (if
    the cited source file is missing, the page is flagged).
    """
    report = LintReport()

    if not wiki_dir.is_dir():
        report.findings.append(
            LintFinding(
                kind="missing-wiki-dir",
                severity=Severity.ERROR,
                path=str(wiki_dir),
                message=f"wiki directory does not exist: {wiki_dir}",
            )
        )
        return report

    pages_dir = wiki_dir / PAGES_DIR
    pages = _load_pages(wiki_dir, pages_dir, report)

    _check_aliases(pages, report)
    # An alias a frozen source depends on is load-bearing, not debt — the
    # retirement nudges below must not tell you to remove it.
    pinned = _source_pinned_aliases(sources_dir, _alias_map(pages))
    _check_wikilinks(pages, pinned, report)
    _check_dead_slug_mentions(pages, pinned, report)
    _check_provenance(pages, raw_dir=raw_dir, sources_dir=sources_dir, report=report)
    _check_sources_registry(sources_dir, report)
    _check_source_slug_coupling(pages, sources_dir, report)
    _check_orphans(pages, log_dir=log_dir, report=report)
    _check_index_drift(wiki_dir, pages_dir, report)

    return report


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


@dataclass
class _LoadedPage:
    slug: str
    path: Path
    rel_path: str  # repo-relative, for messaging
    provenance: list[ProvenanceEntry]
    body: str
    outbound_links: tuple[str, ...]
    generated: bool
    aliases: tuple[str, ...] = ()


def _load_pages(
    wiki_dir: Path, pages_dir: Path, report: LintReport
) -> dict[str, _LoadedPage]:
    """Parse every ``wiki/pages/**/*.md``."""
    pages: dict[str, _LoadedPage] = {}
    for path in editorial_pages(pages_dir):
        expected_slug = relpath_to_slug(path.relative_to(pages_dir))
        rel = f"{wiki_dir.name}/{PAGES_DIR}/{path.relative_to(pages_dir).as_posix()}"
        try:
            # Same loader every other reader uses, so a page that
            # ``read_page`` self-heals isn't a CI-failing ERROR here while
            # the rest of outmem serves it happily. The repair is reported
            # below at WARNING instead — you still want to persist it.
            frontmatter, body, repaired = load_page_text(
                path.read_text(encoding="utf-8")
            )
        except Exception as exc:
            report.findings.append(
                LintFinding(
                    kind="frontmatter-invalid",
                    severity=Severity.ERROR,
                    path=rel,
                    message=str(exc),
                )
            )
            continue
        if repaired:
            report.findings.append(
                LintFinding(
                    kind="frontmatter-repairable",
                    severity=Severity.WARNING,
                    path=rel,
                    message=(
                        "frontmatter only parses after repair (usually an "
                        "unquoted ': ' in a value) — persist the fix with "
                        "`store.repair_pages(dry_run=False)` or a commit "
                        "through the pre-commit hook"
                    ),
                )
            )
        if frontmatter.slug in pages:
            # Silently overwriting here used to lose a page from every
            # slug-keyed check below (links, orphans) with no signal.
            report.findings.append(
                LintFinding(
                    kind="duplicate-slug",
                    severity=Severity.ERROR,
                    path=rel,
                    message=(
                        f"slug {frontmatter.slug!r} is already claimed by "
                        f"{pages[frontmatter.slug].rel_path} — one of the two "
                        "must change, they cannot both be linked to"
                    ),
                )
            )
        if frontmatter.slug != expected_slug:
            report.findings.append(
                LintFinding(
                    kind="slug-filename-mismatch",
                    severity=Severity.ERROR,
                    path=rel,
                    message=(
                        f"frontmatter slug {frontmatter.slug!r} does not "
                        f"match path-derived slug {expected_slug!r}"
                    ),
                )
            )
        links = tuple(link.slug for link in extract_wikilinks(body))
        generated = bool(frontmatter.extra.get("generated"))
        pages[frontmatter.slug] = _LoadedPage(
            slug=frontmatter.slug,
            path=path,
            rel_path=rel,
            provenance=list(frontmatter.provenance),
            body=body,
            outbound_links=links,
            generated=generated,
            aliases=tuple(frontmatter.aliases),
        )
    return pages


# A slug-shaped token: two or more ``:``-joined segments, matching the slug
# grammar in outmem.slug. Single-segment tokens are excluded — a bare word
# like "sepsis" is prose, not a reference.
_SLUG_TOKEN_RE = re.compile(
    r"(?<![\w:-])[a-z0-9]+(?:-[a-z0-9]+)*(?::[a-z0-9]+(?:-[a-z0-9]+)*)+(?![\w-])"
)


def _check_dead_slug_mentions(
    pages: dict[str, _LoadedPage],
    pinned: frozenset[str],
    report: LintReport,
) -> None:
    """Flag slugs written as prose that no longer resolve.

    A ``broken-wikilink`` check is *by construction* blind to these: a slug
    sitting in running text (``"Volltext-Digest: clinical:pflegeheim-x"``)
    or in a ``provenance.upstream`` string is not a ``[[link]]``, so nothing
    validates it. That is exactly where dead references accumulate after a
    namespace is reorganised, and they can sit undetected for months.

    False positives are the whole difficulty here. Slug grammar is permissive
    enough that ``12:30`` (a time) and ``3:1`` (a ratio) parse as slugs, and
    a sentence like ``clinical:leishmaniose: L-AmB 3 mg/kg`` uses the second
    colon as punctuation. The gate: only consider a token whose **namespace
    prefix already exists in this wiki**. A namespace survives a
    reorganisation (``sop:mikrobiologie:vitek2`` →
    ``sop:mikrobiologie:geraete:vitek2`` keeps ``sop``), while ``12`` and
    ``3`` are not namespaces, so times and ratios drop out.
    """
    known = set(pages.keys())
    aliases = _alias_map(pages)
    # Every namespace prefix in use, at any depth: `abx`, `abx:side-effects`.
    namespaces: set[str] = set()
    for slug in known:
        segments = slug.split(":")
        for i in range(1, len(segments)):
            namespaces.add(":".join(segments[:i]))
    if not namespaces:
        return

    for page in pages.values():
        # Drop the linked spans first — a real [[link]] is _check_wikilinks'
        # job, and double-reporting it here would be noise.
        prose = page.body
        for link in extract_wikilinks(page.body):
            prose = prose.replace(link.raw, " ")
        seen: set[str] = set()
        for match in _SLUG_TOKEN_RE.finditer(prose):
            token = match.group(0)
            if token in known or token in seen:
                continue
            namespace = token.rsplit(":", 1)[0]
            if namespace not in namespaces:
                continue  # not a wiki namespace — a time, a ratio, prose
            seen.add(token)
            if token in aliases:
                # It resolves — `read(token)` opens the page. Calling it "not
                # a page" would be false, and it would fire on every clean
                # `outmem rename`, which is the case aliases exist to make
                # safe. Still worth a nudge: prose is editable, so an old
                # name here is debt the author can retire, exactly as
                # `wikilink-via-alias` treats a link.
                report.findings.append(
                    LintFinding(
                        kind="slug-mention-via-alias",
                        severity=Severity.WARNING,
                        path=page.rel_path,
                        message=(
                            f"text mentions {token!r}, which resolves only via "
                            f"an alias on {aliases[token]!r} — update it to "
                            f"{aliases[token]!r}" + _alias_advice(token, pinned)
                        ),
                    )
                )
                continue
            report.findings.append(
                LintFinding(
                    kind="dead-slug-mention",
                    severity=Severity.WARNING,
                    path=page.rel_path,
                    message=(
                        f"text mentions {token!r}, which is not a page — the "
                        f"{namespace!r} namespace exists, so this is probably a "
                        "slug left behind by a rename. Update it or make it a "
                        "[[link]] so it gets checked."
                    ),
                )
            )


def _alias_advice(alias: str, pinned: frozenset[str]) -> str:
    """The trailing clause of an alias-retirement nudge.

    Retiring an alias is the right end state — that is what keeps aliases
    from becoming permanent — *unless* a content-addressed source names
    it. That file cannot be edited to stop needing it, so the alias is
    structural and the nudge applies only to the editable reference.
    """
    if alias in pinned:
        return (
            "; keep the alias — a frozen source under sources/ names "
            f"{alias!r} and cannot be rewritten"
        )
    return " so the alias can eventually go"


def _alias_map(pages: dict[str, _LoadedPage]) -> dict[str, str]:
    """Alias → canonical slug. File-first: a live page keeps its own name."""
    out: dict[str, str] = {}
    for page in pages.values():
        for alias in page.aliases:
            if alias not in pages:
                out.setdefault(alias, page.slug)
    return out


def _check_aliases(
    pages: dict[str, _LoadedPage],
    report: LintReport,
) -> None:
    """Aliases that can never resolve, or resolve ambiguously."""
    claims: dict[str, list[str]] = {}
    for page in pages.values():
        for alias in page.aliases:
            claims.setdefault(alias, []).append(page.slug)
    for alias, owners in sorted(claims.items()):
        if alias == INDEX_SLUG:
            report.findings.append(
                LintFinding(
                    kind="alias-reserved", severity=Severity.ERROR,
                    path=pages[owners[0]].rel_path,
                    message=f"alias {alias!r} is the reserved index slug",
                )
            )
        elif alias in pages:
            report.findings.append(
                LintFinding(
                    kind="alias-shadowed", severity=Severity.ERROR,
                    path=pages[owners[0]].rel_path,
                    message=(
                        f"alias {alias!r} is a real page, so it never resolves "
                        "to this one — a live page always wins its own name"
                    ),
                )
            )
        elif len(owners) > 1:
            for owner in owners:
                report.findings.append(
                    LintFinding(
                        kind="alias-conflict", severity=Severity.ERROR,
                        path=pages[owner].rel_path,
                        message=(
                            f"alias {alias!r} is also claimed by "
                            f"{[o for o in owners if o != owner]} — which wins "
                            "would depend on directory order"
                        ),
                    )
                )


def _check_wikilinks(
    pages: dict[str, _LoadedPage],
    pinned: frozenset[str],
    report: LintReport,
) -> None:
    known = set(pages.keys())
    aliases = _alias_map(pages)
    for page in pages.values():
        for target in page.outbound_links:
            if target == page.slug:
                continue  # self-links are accepted; backlinks already skips them
            if target in aliases:
                report.findings.append(
                    LintFinding(
                        kind="wikilink-via-alias",
                        severity=Severity.WARNING,
                        path=page.rel_path,
                        message=(
                            f"[[{target}]] resolves only via an alias on "
                            f"{aliases[target]!r} — rewrite it to "
                            f"[[{aliases[target]}]]" + _alias_advice(target, pinned)
                        ),
                    )
                )
                continue
            if target not in known:
                report.findings.append(
                    LintFinding(
                        kind="broken-wikilink",
                        severity=Severity.ERROR,
                        path=page.rel_path,
                        message=f"[[{target}]] refers to a page that does not exist",
                    )
                )


def _check_sources_registry(
    sources_dir: Path | None,
    report: LintReport,
) -> None:
    """Reconcile ``.sources.db`` against what is actually on disk.

    Both directions matter and neither was checked before. A row whose
    file is gone makes ``list_sources`` advertise material the agent then
    can't read; a file with no row is invisible to provenance. Because
    nothing reconciled them, a registry can drift to double-digit
    percentages of junk without anyone noticing.

    Reported here, actioned by ``outmem sources gc``.
    """
    if sources_dir is None or not sources_dir.is_dir():
        return
    from outmem.sources import REGISTRY_FILENAME, SourceRegistry

    registry = SourceRegistry.load(sources_dir)
    registered = set(registry.entries)
    for rel_path in sorted(registered):
        if not (sources_dir / rel_path).is_file():
            report.findings.append(
                LintFinding(
                    kind="source-orphaned",
                    severity=Severity.WARNING,
                    path=f"{sources_dir.name}/{rel_path}",
                    message=(
                        "registered in .sources.db but the file is gone — "
                        "run `outmem sources gc` to review and remove"
                    ),
                )
            )
    on_disk = {
        p.relative_to(sources_dir).as_posix()
        for p in sources_dir.rglob("*")
        if p.is_file() and p.name != REGISTRY_FILENAME
    }
    for rel_path in sorted(on_disk - registered):
        report.findings.append(
            LintFinding(
                kind="source-unregistered",
                severity=Severity.WARNING,
                path=f"{sources_dir.name}/{rel_path}",
                message=(
                    "file under sources/ with no .sources.db row — it has no "
                    "provenance and no ingestion history"
                ),
            )
        )


def _check_source_slug_coupling(
    pages: dict[str, _LoadedPage],
    sources_dir: Path | None,
    report: LintReport,
) -> None:
    """Flag frozen sources that reference mutable page slugs.

    Sources are content-addressed — their path embeds a sha, so their
    content is immutable by construction. A page slug is the opposite:
    it moves whenever the wiki is reorganised. A source that names page
    slugs therefore couples something that can never change to something
    that changes often, and the reference rots with no way to notice.

    Observed in production: 136 dead slugs across 129 source files — and
    tellingly, *none* in genuine third-party material (0 of 305 files
    across guidelines, publications, regulatory…). All of it was
    self-authored SOP transcripts and dictated notes filed as sources.
    That distribution is the real signal, and it is why outmem now gives
    sources *versions* (``--as`` / ``outmem stale``) rather than a
    separate tree: an SOP is a source that gets replaced, exactly like a
    republished guideline. What supersession does **not** fix is this
    coupling, so the check stays.

    Only fires on slugs that no longer resolve **at all**. A live
    reference is fine, and so is one that resolves through an alias —
    the alias is doing precisely its job, protecting a reference in a
    file that is content-addressed and therefore cannot be edited. There
    is no ``wikilink-via-alias``-style nudge here for that reason:
    reporting it would ask the operator to fix something unfixable.
    """
    if sources_dir is None or not sources_dir.is_dir():
        return
    # A token with a *recorded* mapping resolves by identity rather than
    # by string: the registry remembers what it meant at ingest and
    # `rename_page` keeps that current, so a rename can no longer break
    # it. When that mapping's target is gone the reference is *certainly*
    # dead — no heuristics, no namespace gate, and it works for
    # single-segment slugs the text scan can never see.
    recorded = _recorded_refs(sources_dir)
    for source_rel in sorted(recorded):
        for token, target in sorted(recorded[source_rel].items()):
            if target in pages:
                continue
            report.findings.append(
                LintFinding(
                    kind="source-references-dead-slug",
                    severity=Severity.WARNING,
                    path=f"{sources_dir.name}/{source_rel}",
                    message=(
                        f"frozen source references {token!r}, recorded at "
                        f"ingest as {target!r}, which no longer exists. A "
                        "rename would have been followed — this page was "
                        "deleted or moved outside outmem. Restore it, or "
                        "re-ingest the source without the slug."
                    ),
                )
            )

    # Resolution and the false-positive gate are separate questions. An
    # alias resolves, so it belongs in `resolvable`; but the namespace
    # gate keeps deriving from live pages only, because widening it is
    # what would let `12:30` back in. This half is the fallback for
    # sources ingested before the mapping existed.
    resolvable = set(pages) | set(_alias_map(pages))
    namespaces: set[str] = set()
    for slug in pages:
        segments = slug.split(":")
        for i in range(1, len(segments)):
            namespaces.add(":".join(segments[:i]))
    if not namespaces:
        return
    for path in sorted(sources_dir.rglob("*")):
        if not path.is_file() or path.name == ".sources.db":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        rel = f"{sources_dir.name}/{path.relative_to(sources_dir).as_posix()}"
        source_rel = path.relative_to(sources_dir).as_posix()
        mapped = recorded.get(source_rel, {})
        seen: set[str] = set()
        for match in _SLUG_TOKEN_RE.finditer(text):
            token = match.group(0)
            if token in resolvable or token in seen:
                continue
            if token in mapped:
                continue  # the mapping owns this token, reported above
            if token.rsplit(":", 1)[0] not in namespaces:
                continue
            seen.add(token)
            report.findings.append(
                LintFinding(
                    kind="source-references-dead-slug",
                    severity=Severity.WARNING,
                    path=rel,
                    message=(
                        f"frozen source references {token!r}, which no longer "
                        "resolves — not as a page and not as an alias. A "
                        "content-addressed source can never change; a page "
                        "slug changes whenever the wiki is reorganised. "
                        "Rename the page back, add the old name to its "
                        "`aliases:`, or re-ingest the source without the slug."
                    ),
                )
            )


def _check_provenance(
    pages: dict[str, _LoadedPage],
    *,
    raw_dir: Path | None,
    sources_dir: Path | None,
    report: LintReport,
) -> None:
    """Flag pages whose cited source files no longer exist."""
    for page in pages.values():
        for entry in page.provenance:
            ref = provenance_ref(entry)
            if ref is None:
                continue
            if not _provenance_exists(ref, raw_dir=raw_dir, sources_dir=sources_dir):
                report.findings.append(
                    LintFinding(
                        kind="stale-provenance",
                        severity=Severity.WARNING,
                        path=page.rel_path,
                        message=(
                            f"cites {ref!r} but the file is missing — either "
                            "restore the source or update the page"
                        ),
                    )
                )
                continue
            # The file existing is only half the question. A source that was
            # re-ingested after its content changed lives at a new
            # sha-addressed path, so a page still citing the old sha points
            # at content that is no longer what the page was compacted from.
            cited_sha = _provenance_sha(entry)
            if cited_sha and sources_dir is not None:
                actual = _registry_sha(sources_dir, ref)
                if actual is not None and actual != cited_sha:
                    report.findings.append(
                        LintFinding(
                            kind="provenance-sha-mismatch",
                            severity=Severity.WARNING,
                            path=page.rel_path,
                            message=(
                                f"cites {ref!r} with sha256 {cited_sha[:12]}… but "
                                f"the registry has {actual[:12]}… — the source was "
                                "re-ingested; re-check the page against it"
                            ),
                        )
                    )


def _provenance_sha(entry: Any) -> str | None:
    """The ``sha256`` a dict-shaped provenance entry claims, if any."""
    if isinstance(entry, dict):
        candidate = entry.get("sha256")
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def _recorded_refs(sources_dir: Path | None) -> dict[str, dict[str, str]]:
    """``source rel_path -> {token: page_slug}`` from the registry."""
    if sources_dir is None or not sources_dir.is_dir():
        return {}
    from outmem.sources import SourceRegistry

    try:
        registry = SourceRegistry.load(sources_dir)
    except Exception:  # a registry we can't read is not a lint failure
        return {}
    out: dict[str, dict[str, str]] = {}
    for ref in registry.refs():
        out.setdefault(ref.rel_path, {})[ref.token] = ref.page_slug
    return out


def _source_pinned_aliases(
    sources_dir: Path | None, aliases: dict[str, str]
) -> frozenset[str]:
    """Aliases a frozen source depends on, so retiring one would break it.

    Both alias nudges end with "so the alias can eventually go". For an
    alias that is the only thing keeping a content-addressed file's
    reference alive, that advice is wrong — the source cannot be edited
    to stop needing it. These are load-bearing, and the nudge says so.
    """
    if not aliases or sources_dir is None or not sources_dir.is_dir():
        return frozenset()
    from outmem.sources import REGISTRY_FILENAME as _REGISTRY_FILE
    pinned: set[str] = set()
    for path in sources_dir.rglob("*"):
        if not path.is_file() or path.name == _REGISTRY_FILE:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        pinned.update(ref.slug for ref in extract_slug_references(text) if ref.slug in aliases)
    return frozenset(pinned)


def _registry_sha(sources_dir: Path, ref: str) -> str | None:
    """The sha256 ``.sources.db`` holds for ``ref``, or None if unknown.

    Cached per sources_dir: ``_check_provenance`` runs per provenance
    entry across every page, and re-opening the sqlite registry each time
    would make lint O(entries) database opens.
    """
    key = str(sources_dir)
    cached = _REGISTRY_SHA_CACHE.get(key)
    if cached is None:
        from outmem.sources import SourceRegistry

        try:
            registry = SourceRegistry.load(sources_dir)
        except Exception:
            cached = {}
        else:
            cached = {rel: e.sha256 for rel, e in registry.entries.items()}
        _REGISTRY_SHA_CACHE[key] = cached
    # Provenance may cite the path with or without the `sources/` prefix.
    return cached.get(ref) or cached.get(ref.removeprefix(f"{sources_dir.name}/"))


_REGISTRY_SHA_CACHE: dict[str, dict[str, str]] = {}


def provenance_ref(entry: Any) -> str | None:
    """Extract a path-shaped reference from a provenance entry.

    Public because ``WikiStore.source_citations`` reads the same field
    for ``outmem stale``. A second, narrower extractor there meant a page
    citing its source under ``source:`` or ``file:`` — shapes lint
    resolves and sha-checks — was invisible to staleness: a silent miss
    of exactly the failure mode the feature exists to catch.
    """
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        candidate = entry.get("path") or entry.get("source") or entry.get("file")
        if isinstance(candidate, str):
            return candidate
    return None


def _provenance_exists(
    ref: str,
    *,
    raw_dir: Path | None,
    sources_dir: Path | None,
) -> bool:
    """A provenance reference resolves if the file exists in either
    ``raw/`` or ``sources/`` (matching either the bare path or the
    appropriate directory prefix)."""
    candidates: list[Path] = []
    if raw_dir is not None:
        candidates.append(raw_dir / ref)
        if ref.startswith("raw/"):
            candidates.append(raw_dir.parent / ref)
    if sources_dir is not None:
        candidates.append(sources_dir / ref)
        if ref.startswith("sources/"):
            candidates.append(sources_dir.parent / ref)
    return any(p.exists() for p in candidates)


def _check_orphans(
    pages: dict[str, _LoadedPage],
    *,
    log_dir: Path | None,
    report: LintReport,
) -> None:
    """Flag pages with zero inbound wikilinks and no mention in log/."""
    inbound: dict[str, set[str]] = {slug: set() for slug in pages}
    # A link arriving via an alias still references the page — without
    # folding, renaming a page makes it look orphaned.
    aliases = _alias_map(pages)
    for page in pages.values():
        if page.generated:
            # Generated pages (the auto-index) link to everything by
            # construction — those links are navigational, not
            # editorial. Don't let them rescue real orphans.
            continue
        for target in page.outbound_links:
            target = aliases.get(target, target)
            if target in inbound and target != page.slug:
                inbound[target].add(page.slug)

    log_mentions = _scan_log_for_mentions(log_dir, set(pages.keys()))

    for page in pages.values():
        if page.generated:
            # The index is intentionally a hub — never has inbound links.
            continue
        if inbound[page.slug]:
            continue
        if page.slug in log_mentions:
            continue
        report.findings.append(
            LintFinding(
                kind="orphan-page",
                severity=Severity.WARNING,
                path=page.rel_path,
                message=(
                    "no inbound wikilinks and no mentions in log/ — link it "
                    "from a related page or drop it"
                ),
            )
        )


def _scan_log_for_mentions(log_dir: Path | None, slugs: Iterable[str]) -> set[str]:
    mentioned: set[str] = set()
    if log_dir is None or not log_dir.is_dir():
        return mentioned
    for path in log_dir.glob("*.md"):
        text = path.read_text(encoding="utf-8")
        for link in extract_wikilinks(text):
            mentioned.add(link.slug)
        for slug in slugs:
            if slug in mentioned:
                continue
            if slug in text:
                mentioned.add(slug)
    return mentioned


def _check_index_drift(wiki_dir: Path, pages_dir: Path, report: LintReport) -> None:
    """Flag if ``wiki/index.md`` is out of sync with the page set.

    Compares the body (post-frontmatter) of the on-disk index against
    a freshly-rendered one. Mismatch usually means a human added a
    page via Obsidian and didn't run an outmem write — easily fixed
    by running any write or by ``outmem lint --fix`` (deferred).
    """
    on_disk = wiki_dir / INDEX_FILENAME
    if not on_disk.exists():
        if pages_dir.is_dir() and any(pages_dir.rglob("*.md")):
            # Pages exist but no index — drift.
            report.findings.append(
                LintFinding(
                    kind="index-missing",
                    severity=Severity.WARNING,
                    path=f"{wiki_dir.name}/{INDEX_FILENAME}",
                    message="wiki has pages but no index — next page write will create it",
                )
            )
        return

    try:
        on_disk_fm, on_disk_body = parse_wiki_page(on_disk.read_text(encoding="utf-8"))
    except Exception as exc:
        report.findings.append(
            LintFinding(
                kind="frontmatter-invalid",
                severity=Severity.ERROR,
                path=f"{wiki_dir.name}/{INDEX_FILENAME}",
                message=f"index.md has malformed frontmatter: {exc}",
            )
        )
        return

    if on_disk_fm.slug != INDEX_SLUG:
        report.findings.append(
            LintFinding(
                kind="index-malformed",
                severity=Severity.ERROR,
                path=f"{wiki_dir.name}/{INDEX_FILENAME}",
                message="index.md frontmatter slug is not 'index'",
            )
        )
        return

    expected = render_index(pages_dir)
    if _normalize(on_disk_body) != _normalize(expected):
        report.findings.append(
            LintFinding(
                kind="index-drift",
                severity=Severity.WARNING,
                path=f"{wiki_dir.name}/{INDEX_FILENAME}",
                message=(
                    "index.md doesn't reflect current pages — likely an "
                    "Obsidian edit added/removed a page. Re-run any outmem "
                    "write to regenerate."
                ),
            )
        )


def _normalize(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def format_report(report: LintReport) -> str:
    """Render a :class:`LintReport` for human consumption (CLI / log)."""
    if not report.has_findings:
        return "OK — no issues found.\n"
    lines: list[str] = []
    groups = report.by_kind()
    total = len(report.findings)
    errors = sum(1 for f in report.findings if f.severity == Severity.ERROR)
    warnings = total - errors
    lines.append(f"Found {total} issue(s): {errors} error(s), {warnings} warning(s).")
    lines.append("")
    for kind in sorted(groups):
        lines.append(f"## {kind}")
        for finding in groups[kind]:
            lines.append(f"  [{finding.severity.value}] {finding.path}: {finding.message}")
        lines.append("")
    return "\n".join(lines)
