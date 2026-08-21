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
- Stale provenance (the cited source file is missing) and
  provenance citing a sha256 the registry no longer holds
- ``.sources.db`` disagreeing with what is on disk, in either direction
- Source versions that should be one supersession chain and are not —
  two editions under keys differing only in a number, or one key held by
  several rows all reading as current
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

import datetime as dt
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from outmem.completeness import find_elision_markers, find_tool_sentinels
from outmem.exceptions import OutmemError
from outmem.frontmatter import ProvenanceEntry, parse_wiki_page
from outmem.git_ops import tracked_paths_under
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
from outmem.sources import (
    PROVENANCE_FINDINGS,
    SOURCES_DIR,
    SOURCES_LOCAL_DIR,
    UnchainedVersions,
    find_unchained_versions,
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
    # 1-based line within ``path``, when the finding is about one line.
    # Kept separate rather than smuggled into ``path`` as "file.md:13":
    # callers build real paths out of that field (``wiki_dir / path``),
    # and a line suffix silently turns those into nonexistent files.
    line: int | None = None


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
    sources_dir: Path | None = None,
    sources_local_dir: Path | None = None,
    repo_root: Path | None = None,
    indexed_paths: Iterable[str] | None = None,
) -> LintReport:
    """Run every static check against ``wiki_dir``.

    ``log_dir`` is consulted for orphan detection — a page mentioned
    only in ``log/<date>.md`` still counts as referenced.
    ``sources_dir`` and ``sources_local_dir`` are consulted for
    stale-provenance checks (if the cited source file is missing, the
    page is flagged); a page may legitimately cite either tree.

    ``repo_root`` and ``indexed_paths`` enable the containment checks
    that make the tracked/local split trustworthy rather than merely
    conventional — see :func:`_check_local_source_containment` and
    :func:`_check_local_source_not_indexed`. Both are optional so a
    library caller can run the page-level checks without paying for a
    git invocation or a vector-store open; the CLI passes them, so the
    checks run wherever a user would actually look.
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
    _check_page_completeness(pages, report)
    _check_declared_omissions(pages, report)
    _check_provenance(
        pages,
        sources_dir=sources_dir,
        sources_local_dir=sources_local_dir,
        report=report,
    )
    _check_sources_registry(sources_dir, report)
    _check_unchained_source_versions(sources_dir, sources_local_dir, report)
    _check_source_slug_coupling(pages, sources_dir, report)
    _check_orphans(pages, log_dir=log_dir, report=report)
    _check_index_drift(wiki_dir, pages_dir, report)
    _check_local_source_containment(
        sources_local_dir=sources_local_dir, repo_root=repo_root, report=report
    )
    _check_local_source_not_indexed(
        indexed_paths=indexed_paths,
        wiki_dir_name=wiki_dir.name,
        report=report,
    )

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
    # Lines the frontmatter occupies, so a body-relative line number can
    # be reported in file coordinates — the number is only useful if it
    # matches what an editor and `grep_wiki` show for the same page.
    body_line_offset: int = 0
    # `omitted:` entries — content the author says the page deliberately
    # leaves out. Round-trips through `extra`, so it needs no schema.
    omitted: tuple[str, ...] = ()


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
            raw = path.read_text(encoding="utf-8")
            frontmatter, body, repaired = load_page_text(raw)
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
            body_line_offset=len(raw.splitlines()) - len(body.splitlines()),
            omitted=_declared_omissions(frontmatter.extra.get("omitted")),
        )
    return pages


def _declared_omissions(value: Any) -> tuple[str, ...]:
    """Normalise an ``omitted:`` frontmatter value to a tuple of notes.

    Lenient about shape (a bare string is one note) because the field
    round-trips through ``extra`` and has no schema to enforce — the
    point is to surface what is there, not to reject how it was typed.
    """
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    if isinstance(value, list):
        return tuple(str(v).strip() for v in value if str(v).strip())
    if isinstance(value, dict):
        # `omitted: {therapie: "on clinical:therapie"}` is the shape an
        # author reaches for first. Silently returning () there would let
        # a page be made clean by announcing what it left out — the one
        # thing this check exists to prevent.
        return tuple(
            f"{k}: {v}".strip() for k, v in value.items() if str(v).strip()
        )
    return ()


def _check_declared_omissions(
    pages: dict[str, _LoadedPage], report: LintReport
) -> None:
    """Report every ``omitted:`` note as an open item.

    ``omitted:`` gives a deliberate scoping decision somewhere to live —
    "treatment is on another page", "the PCR protocol details stay in the
    source" — which is genuinely useful, and it is also the obvious place
    for budget-driven truncation to migrate to once the elision marker is
    refused. Reporting it is what keeps those apart: a declared gap is
    still a gap, so it costs a warning, and a page cannot be made clean by
    announcing what it left out.

    That is why there is no ``omitted`` tool argument. The convention is
    for a human curating scope, not a channel a model can use to make a
    short page legal.
    """
    for page in sorted(pages.values(), key=lambda p: p.slug):
        for note in page.omitted:
            report.findings.append(
                LintFinding(
                    kind="declared-omission",
                    severity=Severity.WARNING,
                    path=page.rel_path,
                    message=(
                        f"page declares omitted content: {note!r}. Deliberate "
                        "scoping is fine — this is reported so the gap stays "
                        "visible, not because it is wrong. Fill it, link the "
                        "page that covers it, or drop the note once it no "
                        "longer describes the page."
                    ),
                )
            )


def _check_page_completeness(
    pages: dict[str, _LoadedPage], report: LintReport
) -> None:
    """Flag pages that stop early, and tool output copied into a page.

    The only check outmem has that reads the body *as content*. It exists
    because a budget-truncated page is otherwise indistinguishable from a
    complete one: its provenance, hashes, links, and index entry are all
    correct, and those are what every other check inspects.

    WARNING rather than ERROR deliberately. The signal is positional and
    therefore fallible, and a false positive that turns CI red is how a
    completeness check gets switched off — which costs more than the
    occasional missed cut.
    """
    for page in sorted(pages.values(), key=lambda p: p.slug):
        if page.generated:
            continue  # wiki/index.md is machine-written, not authored
        for elision in find_elision_markers(page.body):
            report.findings.append(
                LintFinding(
                    kind="truncated-page",
                    severity=Severity.WARNING,
                    path=page.rel_path,
                    line=elision.line + page.body_line_offset,
                    message=(
                        f"page appears to stop early — {elision.marker!r} ends "
                        f"the line {elision.text!r}. If content is missing, "
                        f"restore it from the page's sources (`outmem sources "
                        f"list`) and append it with `extend_page`/`append_page`; "
                        f"if the ellipsis is part of a quotation, keep prose "
                        f"after it or move it into a blockquote."
                    ),
                )
            )
        for sentinel in find_tool_sentinels(page.body):
            report.findings.append(
                LintFinding(
                    kind="tool-output-in-page",
                    severity=Severity.WARNING,
                    path=page.rel_path,
                    line=sentinel.line + page.body_line_offset,
                    message=(
                        "an outmem tool-output marker was copied into the page "
                        f"({sentinel.text!r}) — that marker means outmem itself "
                        "withheld content from a tool result, so the page is "
                        "built on material it never actually saw. Re-read the "
                        "source in full and rewrite the passage."
                    ),
                )
            )


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


def _check_unchained_source_versions(
    sources_dir: Path | None,
    sources_local_dir: Path | None,
    report: LintReport,
) -> None:
    """Live rows that look like versions of one document but aren't chained.

    The defect supersession was built to remove, reappearing one level
    up. ``document_key`` links a revision to what it replaces — but a
    source ingested without ``--as`` has its identity *derived* from its
    filename, and the year is usually in the filename. So the 2024 and
    2026 editions of one guideline become two documents, no edge is
    written, and ``outmem stale`` never reports the page compacted from
    the older one. Nothing is wrong in the registry; the failure is
    entirely an absence.

    Both trees, each on its own: a tracked and a local source can hold
    the same key without being related, and supersession cannot span two
    registries anyway.
    """
    from outmem.sources import SourceRegistry

    for tree_dir in (sources_dir, sources_local_dir):
        if tree_dir is None or not tree_dir.is_dir():
            continue
        for group in find_unchained_versions(SourceRegistry.load(tree_dir)):
            report.findings.append(
                LintFinding(
                    kind="multiple-live-versions"
                    if group.shares_one_key
                    else "unlinked-source-versions",
                    severity=Severity.WARNING,
                    path=f"{tree_dir.name}/{group.entries[0].rel_path}",
                    message=_unchained_versions_message(group),
                )
            )


def _unchained_versions_message(group: UnchainedVersions) -> str:
    """Name the rows, then both ways out.

    Two derived keys resembling each other is a judgement outmem cannot
    make — "next edition of that" and "different document, similar name"
    look identical from the path. So the message carries the evidence
    (the ingest origins, which is where the distinguishing part of a
    pipeline path survives) and names the remedy for *either* answer,
    the same shape ``add_source``'s ambiguity refusal already uses.
    Naming only the merge would turn every legitimately-numbered pair
    into a warning with no way to reach zero.
    """
    rows = []
    for entry in group.entries:
        origin = f"\n      from {entry.origin_path}" if entry.origin_path else ""
        rows.append(f"    {entry.document_key}\n      {entry.rel_path}{origin}")
    listing = "\n".join(rows)
    if group.shares_one_key:
        return (
            f"{len(group.entries)} sources share the identity "
            f"{group.keys[0]!r} with nothing linking them, so all of them "
            "read as current and `outmem stale` reports none:\n"
            f"{listing}\n"
            f"  -> `outmem sources rekey {group.keys[0]}` chains them by "
            "ingest time"
        )
    newest = group.entries[-1].document_key
    # One per line: a corpus routinely has three or more editions, and
    # run together they read as a single command with stray arguments.
    merge = "\n".join(
        f"       outmem sources rekey {e.document_key} --to {newest}"
        for e in group.entries[:-1]
    )
    return (
        "these identities differ only in a number, so they look like "
        "editions of one document — but nothing links them, and a page "
        "citing the older one will never be reported stale:\n"
        f"{listing}\n"
        f"  -> if they are one document, merge them onto the newest:\n{merge}\n"
        "  -> if they are different documents: rekey one to a name that "
        "distinguishes it, which also stops this warning (a declared "
        "identity is never second-guessed)"
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


def _check_one_finding(
    entry: Any, *, page: _LoadedPage, report: LintReport
) -> None:
    """Validate a ``finding:`` on one provenance entry.

    Every branch here is the same failure: an entry the author believes
    records a checked absence, which the rest of outmem reads as an
    ordinary citation or as nothing at all. That is strictly worse than
    not writing it — the author stops looking, and no reader can tell.
    So each way of getting it subtly wrong gets named rather than
    ignored.
    """
    if not isinstance(entry, dict) or "finding" not in entry:
        return

    raw = entry["finding"]
    if not isinstance(raw, str) or not raw.strip():
        report.findings.append(
            LintFinding(
                kind="unknown-provenance-finding",
                severity=Severity.WARNING,
                path=page.rel_path,
                message=(
                    f"provenance finding must be a string, got {raw!r}. "
                    f"Expected one of {sorted(PROVENANCE_FINDINGS)}."
                ),
            )
        )
        return

    finding = raw.strip()
    if finding not in PROVENANCE_FINDINGS:
        report.findings.append(
            LintFinding(
                kind="unknown-provenance-finding",
                severity=Severity.WARNING,
                path=page.rel_path,
                message=(
                    f"provenance finding {finding!r} is not one of "
                    f"{sorted(PROVENANCE_FINDINGS)}. An unrecognised value "
                    "records nothing — the entry reads as an ordinary "
                    "citation, which is the opposite of what a checked "
                    "absence means."
                ),
            )
        )
        return

    if provenance_ref(entry) is None:
        report.findings.append(
            LintFinding(
                kind="finding-without-source",
                severity=Severity.WARNING,
                path=page.rel_path,
                message=(
                    f"provenance entry has finding {finding!r} but no "
                    "`path:` — 'we checked and it was silent' needs to say "
                    "*what* was checked. Without a source the entry is "
                    "invisible to `outmem stale`, so it will never be "
                    "re-checked when that source gets a new version."
                ),
            )
        )


def _check_one_ack(entry: Any, *, page: _LoadedPage, report: LintReport) -> None:
    """Validate a ``superseded_ok:`` on one provenance entry.

    Same failure in every branch as :func:`_check_one_finding`: the
    author believes they have recorded a decision, and outmem reads
    nothing. Here that cuts the dangerous way — the author expects
    ``outmem stale`` to stop reporting the row, so they stop looking at
    it, and an acknowledgement that never took effect is indistinguishable
    from one that did.
    """
    if not isinstance(entry, dict) or "superseded_ok" not in entry:
        return

    def warn(message: str) -> None:
        report.findings.append(
            LintFinding(
                kind="invalid-supersession-ack",
                severity=Severity.WARNING,
                path=page.rel_path,
                message=message,
            )
        )

    raw = entry["superseded_ok"]
    if not isinstance(raw, str) or not raw.strip():
        warn(
            f"`superseded_ok:` must say why, got {raw!r}. The reason is the "
            "whole record — a bare flag says a human looked, which is what "
            "the report already assumed."
        )
        return
    annotation = provenance_annotation(entry)
    if annotation.date is None:
        has = f"has {entry['date']!r}" if "date" in entry else "has none"
        warn(
            f"`superseded_ok:` needs a `date:` (YYYY-MM-DD) and this entry "
            f"{has}. An acknowledgement is scoped to the version that was "
            "current when it was made, so without a date there is nothing to "
            "compare against and the row keeps being reported."
        )
        return
    if provenance_ref(entry) is None:
        warn(
            "`superseded_ok:` on an entry with no `path:` — there is no "
            "citation to acknowledge, so it suppresses nothing."
        )


def _check_provenance(
    pages: dict[str, _LoadedPage],
    *,
    sources_dir: Path | None,
    sources_local_dir: Path | None,
    report: LintReport,
) -> None:
    """Flag pages whose cited source files no longer exist."""
    for page in pages.values():
        for entry in page.provenance:
            _check_one_finding(entry, page=page, report=report)
            _check_one_ack(entry, page=page, report=report)
            ref = provenance_ref(entry)
            if ref is None:
                continue
            if not _provenance_exists(
                ref, sources_dir=sources_dir, sources_local_dir=sources_local_dir
            ):
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


def provenance_finding(entry: Any) -> str | None:
    """The ``finding:`` on a provenance entry, if it carries one.

    Present when the citation records a *checked absence* rather than a
    claim drawn from the source — see
    :data:`outmem.sources.PROVENANCE_FINDINGS`. Returns the raw string
    even when it is not in the vocabulary, so the linter can name the
    typo rather than treating an unrecognised value as no value.
    """
    return provenance_annotation(entry).finding


@dataclass(frozen=True)
class ProvenanceAnnotation:
    """The non-path fields on a provenance entry that outmem acts on.

    ``scope`` and ``note`` are deliberately absent: they round-trip
    verbatim and nothing reads them, so listing them here would suggest
    otherwise.
    """

    finding: str | None = None
    """See :data:`outmem.sources.PROVENANCE_FINDINGS`. Raw, so the
    linter can name a typo instead of dropping it."""
    superseded_ok: str | None = None
    """Why citing a superseded version of this source is deliberate."""
    date: dt.date | None = None
    """When the entry was written, if it parses as a date.

    Decoration on a ``finding:``, load-bearing on a ``superseded_ok:`` —
    an acknowledgement is scoped to the version that was current when it
    was made, and this is what says which one that was.
    """

    def __bool__(self) -> bool:
        """Whether the entry records anything outmem acts on.

        A bare ``date:`` is not an annotation — it decorates one. Asking
        this rather than comparing against an empty instance keeps a
        caller correct when a field is added.
        """
        return bool(self.finding or self.superseded_ok)


def provenance_annotation(entry: Any) -> ProvenanceAnnotation:
    """Read the fields outmem acts on off one provenance entry.

    One walker for both ``finding:`` and ``superseded_ok:``. They are
    read together (``outmem stale`` needs both for the same row) and
    validated together, and two extractors over the same shape is how
    the second one silently stops seeing an entry shape the first one
    learned about.
    """
    if not isinstance(entry, dict):
        return ProvenanceAnnotation()
    return ProvenanceAnnotation(
        finding=_nonempty_str(entry.get("finding")),
        superseded_ok=_nonempty_str(entry.get("superseded_ok")),
        date=_as_date(entry.get("date")),
    )


def _nonempty_str(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _as_date(value: Any) -> dt.date | None:
    """``date:`` as a date, however YAML happened to type it.

    PyYAML parses an unquoted ``2026-08-12`` into a ``datetime.date``
    but leaves a quoted one a string, and the difference is invisible in
    the file. Accepting both keeps a suppression from depending on
    whether someone reached for quotes.
    """
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    if isinstance(value, str):
        try:
            return dt.date.fromisoformat(value.strip()[:10])
        except ValueError:
            return None
    return None


def _provenance_exists(
    ref: str,
    *,
    sources_dir: Path | None,
    sources_local_dir: Path | None,
) -> bool:
    """A provenance reference resolves if the cited file exists in either
    source tree.

    Both the bare form (``<sha>/file.md``, relative to a tree) and the
    prefixed form (``sources/<sha>/file.md``, relative to ``wiki/``) are
    accepted, because both appear in the wild — the registry keys on the
    former and pages tend to cite the latter.

    Note the prefixes are not ambiguous despite sharing a stem:
    ``"sources-local/x"`` does not start with ``"sources/"``, so a local
    ref never resolves against the tracked tree by accident.
    """
    candidates: list[Path] = []
    for tree, prefix in (
        (sources_dir, f"{SOURCES_DIR}/"),
        (sources_local_dir, f"{SOURCES_LOCAL_DIR}/"),
    ):
        if tree is None:
            continue
        candidates.append(tree / ref)
        if ref.startswith(prefix):
            candidates.append(tree.parent / ref)
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


def _check_local_source_containment(
    *,
    sources_local_dir: Path | None,
    repo_root: Path | None,
    report: LintReport,
) -> None:
    """Verify nothing under ``wiki/sources-local/`` reached git.

    This is the check that turns the tracked/local split from a naming
    convention into an enforced boundary. The whole point of the local
    tree is that its bytes never leave the machine; if git is tracking
    them, that guarantee is already broken and the user needs to know
    loudly — an ERROR, not a warning, because the remedy (history
    rewrite + rotate anything sensitive) gets harder the longer it sits.

    Skipped when either path is unknown, so library callers that only
    want the page-level checks don't pay for a git invocation.
    """
    if sources_local_dir is None or repo_root is None:
        return
    if not sources_local_dir.is_dir():
        return

    try:
        rel_dir = sources_local_dir.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        # Local tree lives outside the repo — unusual, but then git
        # cannot be tracking it and there is nothing to check.
        return

    try:
        tracked = tracked_paths_under(repo_root, rel_dir)
    except OutmemError:
        # No git, or git failed. Containment is unverifiable rather than
        # violated; staying quiet beats a spurious error.
        return

    if not tracked:
        return

    shown = ", ".join(tracked[:3])
    if len(tracked) > 3:
        shown += f", … ({len(tracked)} total)"
    report.findings.append(
        LintFinding(
            kind="local-source-tracked",
            severity=Severity.ERROR,
            path=rel_dir,
            message=(
                f"git is tracking {len(tracked)} file(s) under {rel_dir}/ "
                f"({shown}). This tree exists to hold material that must "
                "not be redistributed, so tracked bytes are already in "
                "history and will ship with any clone or push. Untrack "
                f"them (`git rm --cached -r {rel_dir}`), confirm "
                f"`{rel_dir}/` is in .gitignore, and rewrite history if "
                "the commits were pushed."
            ),
        )
    )


def _check_local_source_not_indexed(
    *,
    indexed_paths: Iterable[str] | None,
    wiki_dir_name: str,
    report: LintReport,
) -> None:
    """Verify no ``sources-local/`` chunk reached the vector index.

    The index stores each chunk's verbatim text next to its embedding
    and its DB is committed alongside the pages that triggered the
    write, so a local chunk in there is the same leak as a tracked
    file — just through a path that looks like a cache.

    :func:`outmem._store.semantic.load_for_index` refuses these
    unconditionally, so a hit means the index predates that rule (or was
    built by other means). Checking rather than trusting is the point:
    the guarantee is only worth stating if something verifies it.
    """
    if indexed_paths is None:
        return
    prefix = f"{wiki_dir_name}/{SOURCES_LOCAL_DIR}/"
    offenders = sorted(p for p in indexed_paths if p.startswith(prefix))
    if not offenders:
        return
    shown = ", ".join(offenders[:3])
    if len(offenders) > 3:
        shown += f", … ({len(offenders)} total)"
    report.findings.append(
        LintFinding(
            kind="local-source-indexed",
            severity=Severity.ERROR,
            path=prefix.rstrip("/"),
            message=(
                f"the semantic index contains {len(offenders)} chunk(s) from "
                f"{prefix} ({shown}). Chunk text is stored verbatim in the "
                "vector DB, and that DB is committed — so this material "
                "ships even though the source files do not. Run `outmem "
                "reindex --force` to rebuild without them."
            ),
        )
    )


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
            where = (
                f"{finding.path}:{finding.line}"
                if finding.line is not None
                else finding.path
            )
            lines.append(f"  [{finding.severity.value}] {where}: {finding.message}")
        lines.append("")
    return "\n".join(lines)
