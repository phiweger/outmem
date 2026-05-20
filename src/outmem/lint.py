"""Static wiki linter — orphans, broken links, stale provenance, drift.

Read-only mechanical checks against the on-disk wiki. Catches the
class of problems that don't need an LLM:

- Pages with malformed or missing frontmatter
- Broken ``[[wikilink]]`` references (target slug doesn't exist)
- Stale provenance (cited ``raw/`` or ``sources/`` file is missing)
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

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from outmem.frontmatter import ProvenanceEntry, parse_wiki_page
from outmem.index import INDEX_FILENAME, INDEX_SLUG, editorial_pages, render_index
from outmem.slug import PAGES_DIR, extract_wikilinks, relpath_to_slug


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

    _check_wikilinks(pages, report)
    _check_provenance(pages, raw_dir=raw_dir, sources_dir=sources_dir, report=report)
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


def _load_pages(
    wiki_dir: Path, pages_dir: Path, report: LintReport
) -> dict[str, _LoadedPage]:
    """Parse every ``wiki/pages/**/*.md``."""
    pages: dict[str, _LoadedPage] = {}
    for path in editorial_pages(pages_dir):
        expected_slug = relpath_to_slug(path.relative_to(pages_dir))
        rel = f"{wiki_dir.name}/{PAGES_DIR}/{path.relative_to(pages_dir).as_posix()}"
        try:
            frontmatter, body = parse_wiki_page(path.read_text(encoding="utf-8"))
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
        )
    return pages


def _check_wikilinks(
    pages: dict[str, _LoadedPage],
    report: LintReport,
) -> None:
    known = set(pages.keys())
    for page in pages.values():
        for target in page.outbound_links:
            if target == page.slug:
                continue  # self-links are accepted; backlinks already skips them
            if target not in known:
                report.findings.append(
                    LintFinding(
                        kind="broken-wikilink",
                        severity=Severity.ERROR,
                        path=page.rel_path,
                        message=f"[[{target}]] refers to a page that does not exist",
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
            ref = _provenance_ref(entry)
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


def _provenance_ref(entry: Any) -> str | None:
    """Extract a path-shaped reference from a provenance entry."""
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
    for page in pages.values():
        if page.generated:
            # Generated pages (the auto-index) link to everything by
            # construction — those links are navigational, not
            # editorial. Don't let them rescue real orphans.
            continue
        for target in page.outbound_links:
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
