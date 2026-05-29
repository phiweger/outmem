"""``outmem`` CLI entry point.

Subcommands mirror :class:`outmem.store.WikiStore` operations:

* ``outmem init <path>`` — scaffold a new wiki
* ``outmem read <slug>`` — print a wiki page
* ``outmem search <pattern>`` — ripgrep over wiki/raw/log
* ``outmem history <slug>`` — per-page commit log
* ``outmem evolution <slug> [slug...]`` — temporal-evolution diff stream
* ``outmem write <slug> --title T < body`` — create a page from stdin
* ``outmem extend <slug> < body`` — replace body of an existing page
* ``outmem log <topic> < body`` — append a log entry
* ``outmem pull`` / ``outmem push`` — sync with the configured remote
* ``outmem steering`` — phase-1 steering signal (human commits since last run)

Defaults to operating on the wiki at ``$OUTMEM_PATH`` or the current
working directory. ``--root`` overrides per invocation.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from outmem import __version__
from outmem._progress import report_progress
from outmem.config import SEMANTIC_DISABLED_HELP
from outmem.exceptions import OutmemError
from outmem.store import AgentIdentity, WikiStore

_TIMESTAMP_FMT = "%H:%M:%S"


def _status(msg: str) -> None:
    """Print a status / progress line to stdout, prefixed with a local
    ``[HH:MM:SS]`` timestamp. Reserve for human-readable progress messages;
    keep raw :func:`print` for structured output (commit SHAs, grep hits,
    paths) that downstream tools might pipe."""
    print(f"[{datetime.now().strftime(_TIMESTAMP_FMT)}] {msg}")


def _resolve_root(args: argparse.Namespace) -> Path:
    # ``--root`` uses argparse.SUPPRESS default (so nested subparsers don't
    # clobber outer-level values), which means it's ABSENT from args rather
    # than None when unset — use getattr to handle both forms cleanly.
    root = getattr(args, "root", None)
    if root:
        return Path(root).expanduser()
    env = os.environ.get("OUTMEM_PATH")
    if env:
        return Path(env).expanduser()
    return Path.cwd()


def _agent_identity() -> AgentIdentity:
    """Resolve the agent identity from env, falling back to defaults.

    ``OUTMEM_AGENT_NAME`` and ``OUTMEM_AGENT_EMAIL`` override the
    defaults so the same install can serve multiple wikis with different
    agent identities (one process per wiki).
    """
    name = os.environ.get("OUTMEM_AGENT_NAME")
    email = os.environ.get("OUTMEM_AGENT_EMAIL")
    if name and email:
        return AgentIdentity(name=name, email=email)
    return AgentIdentity()


def _open_store(args: argparse.Namespace) -> WikiStore:
    store = WikiStore.open(_resolve_root(args), agent_identity=_agent_identity())
    # Opt-in observability — short-circuit before importing logfire so
    # disabled wikis don't pay the import cost on every CLI invocation.
    if store.config.outmem.logfire.enabled:
        from outmem._logfire import setup as setup_logfire

        setup_logfire(store.config.outmem.logfire)
    return store


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_init(args: argparse.Namespace) -> int:
    root = Path(args.path).expanduser()
    WikiStore.init(root, agent_identity=_agent_identity())
    _status(f"Initialised wiki at {root}")
    return 0


def cmd_read(args: argparse.Namespace) -> int:
    store = _open_store(args)
    page = store.read(args.slug)
    if args.body_only:
        sys.stdout.write(page.body)
    else:
        sys.stdout.write(page.path.read_text(encoding="utf-8"))
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    store = _open_store(args)
    result = store.search(
        args.pattern,
        scope=args.scope,
        case_insensitive=args.ignore_case,
        fixed_strings=args.fixed_strings,
        max_hits=args.max_hits,
    )
    for hit in result.hits:
        print(f"{hit.path}:{hit.line_number}:{hit.text}")
    if result.truncated:
        print("(truncated — narrow the pattern or raise --max-hits)", file=sys.stderr)
    return 0 if result.hits else 1


def cmd_history(args: argparse.Namespace) -> int:
    store = _open_store(args)
    for commit in store.history(args.slug):
        print(
            f"{commit.sha[:10]}  {commit.date.isoformat()}  "
            f"{commit.author_name} <{commit.author_email}>  {commit.subject}"
        )
    return 0


def cmd_evolution(args: argparse.Namespace) -> int:
    store = _open_store(args)
    sys.stdout.write(store.evolution(args.slugs, include_log=not args.no_log))
    return 0


def cmd_write(args: argparse.Namespace) -> int:
    store = _open_store(args)
    body = sys.stdin.read()
    if not body.strip():
        print("write: refusing to write empty body (read from stdin).", file=sys.stderr)
        return 2
    sha = store.write_page(
        args.slug,
        title=args.title,
        body=body,
        provenance=args.provenance or None,
        tags=args.tag or None,
    )
    print(sha)
    return 0


def cmd_extend(args: argparse.Namespace) -> int:
    store = _open_store(args)
    body = sys.stdin.read()
    if not body.strip():
        print("extend: refusing to write empty body (read from stdin).", file=sys.stderr)
        return 2
    sha = store.extend_page(args.slug, body=body)
    print(sha)
    return 0


def cmd_log(args: argparse.Namespace) -> int:
    store = _open_store(args)
    content = sys.stdin.read()
    if not content.strip():
        print("log: refusing to log empty content (read from stdin).", file=sys.stderr)
        return 2
    sha = store.append_log(topic=args.topic, content=content)
    print(sha)
    return 0


def cmd_pull(args: argparse.Namespace) -> int:
    store = _open_store(args)
    store.pull()
    return 0


def cmd_push(args: argparse.Namespace) -> int:
    store = _open_store(args)
    store.push()
    return 0


def cmd_steering(args: argparse.Namespace) -> int:
    store = _open_store(args)
    for commit in store.steering():
        print(
            f"{commit.sha[:10]}  {commit.date.isoformat()}  "
            f"{commit.author_name} <{commit.author_email}>  {commit.subject}"
        )
    return 0


def cmd_record_run(args: argparse.Namespace) -> int:
    store = _open_store(args)
    marker = store.record_run()
    _status(f"recorded run at {marker.timestamp.isoformat()} head={marker.head}")
    return 0


def cmd_similar(args: argparse.Namespace) -> int:
    store = _open_store(args)
    if not store.semantic_enabled():
        print(f"outmem: {SEMANTIC_DISABLED_HELP}", file=sys.stderr)
        return 1

    if args.slug:
        try:
            page = store.read(args.slug)
        except OutmemError as exc:
            print(f"outmem: {exc}", file=sys.stderr)
            return 1
        query_text = page.body
        exclude = args.slug
    elif args.stdin:
        query_text = sys.stdin.read()
        exclude = None
    else:
        query_text = args.text or ""
        exclude = None

    if not query_text.strip():
        print("similar: empty query (use a text arg, --slug, or --stdin).", file=sys.stderr)
        return 2

    try:
        matches = store.semantic_find_similar(
            query_text,
            top_k=args.top_k,
            threshold=args.threshold,
            exclude_slug=exclude,
        )
    except OutmemError as exc:
        print(f"outmem: {exc}", file=sys.stderr)
        return 1

    if not matches:
        print("(no matches above threshold)")
        return 1
    for match in matches:
        preview = match.content.replace("\n", " ").strip()
        if len(preview) > 160:
            preview = preview[:157] + "…"
        print(
            f"{match.similarity:.3f}  {match.rel_path}#chunk{match.chunk_index}  {preview}"
        )
    return 0


def cmd_reindex(args: argparse.Namespace) -> int:
    store = _open_store(args)
    if not store.semantic_enabled():
        print(f"outmem: {SEMANTIC_DISABLED_HELP}", file=sys.stderr)
        return 1

    if args.staged:
        return _cmd_reindex_staged(store)

    try:
        if args.path:
            results = []
            for rel in args.path:
                if args.force:
                    store.semantic_remove_path(rel)
                result = store.semantic_reindex_path(rel)
                if result is not None:
                    results.append(result)
            for r in results:
                state = "skipped" if r.skipped else f"+{r.chunks_added} chunks"
                print(f"{r.rel_path}  {state}")
            return 0

        summary = store.semantic_reindex_all(
            force=args.force,
            on_progress=lambda done, total: report_progress(
                None, done, total, label="reindex", unit="files"
            ),
        )
    except OutmemError as exc:
        print(f"outmem: {exc}", file=sys.stderr)
        return 1
    _status(
        f"reindex: {summary['reindexed']} re-embedded, "
        f"{summary['skipped']} unchanged, "
        f"{summary['removed']} removed, "
        f"{summary['chunks_added']} chunks added, "
        f"{summary.get('embed_tokens', 0)} embed tokens"
    )
    return 0


def _cmd_reindex_staged(store: WikiStore) -> int:
    """Sync derived artefacts to staged changes (pre-commit hook).

    Two derived things need to track manual wiki edits:
    1. ``wiki/index.md`` — the auto-maintained slug index.
    2. The semantic vector DB (when ``semantic.enabled: true``).

    For each staged wiki page / source: reindex it (or remove its
    chunks if the change is a deletion), regenerate ``index.md`` if
    any wiki page touched in the commit, then re-stage both the
    rebuilt index and the vector DB so they land in the same
    commit. Exits 0 even on per-file errors — pre-commit hooks
    should not block commits over indexing issues.
    """
    from outmem.git_ops import add as git_add
    from outmem.git_ops import staged_changes
    from outmem.index import INDEX_FILENAME

    try:
        added, deleted = staged_changes(store.root)
    except OutmemError as exc:
        print(f"outmem: {exc}", file=sys.stderr)
        return 0  # do not block the commit

    # --- semantic index --------------------------------------------------
    for rel in deleted:
        try:
            store.semantic_remove_path(rel)
        except Exception as exc:
            print(f"outmem: semantic remove {rel} failed: {exc}", file=sys.stderr)
    for rel in added:
        try:
            store.semantic_reindex_path(rel)
        except Exception as exc:
            print(f"outmem: semantic reindex {rel} failed: {exc}", file=sys.stderr)
    db_rel = store.config.outmem.semantic.db_filename
    db_path = store.root / db_rel
    if db_path.exists():
        try:
            git_add(store.root, [db_rel])
        except OutmemError as exc:
            print(f"outmem: could not stage {db_rel}: {exc}", file=sys.stderr)

    # --- wiki/index.md ---------------------------------------------------
    # If any staged path is a wiki page (other than index.md itself),
    # regenerate the index and re-stage it so it lands in the same
    # commit as the manual edit. `commit=False` because we're inside
    # the pre-commit hook — the human's commit IS the commit.
    wiki_prefix = f"{store.config.wiki_dir}/"
    index_rel = f"{wiki_prefix}{INDEX_FILENAME}"
    touched_wiki = any(
        rel.startswith(wiki_prefix)
        and rel.endswith(".md")
        and rel != index_rel
        for rel in (*added, *deleted)
    )
    if touched_wiki:
        try:
            store.rebuild_index(commit=False)
            git_add(store.root, [index_rel])
        except OutmemError as exc:
            print(f"outmem: rebuilding index failed: {exc}", file=sys.stderr)

    return 0


HOOK_NAME = "pre-commit"
HOOK_MARKER = "# outmem pre-commit hook"
HOOK_SCRIPT = f"""#!/bin/sh
{HOOK_MARKER}
# Keeps the semantic vector DB in lockstep with externally edited wiki
# pages (Obsidian, etc.). Installed by `outmem hook install`. Safe to
# remove: `outmem hook uninstall`.
set -e
exec outmem reindex --staged
"""


def cmd_hook_install(args: argparse.Namespace) -> int:
    store = _open_store(args)
    hooks_dir = store.root / ".git" / "hooks"
    if not hooks_dir.is_dir():
        print(
            f"outmem: no .git/hooks at {store.root} — is this a git repo?",
            file=sys.stderr,
        )
        return 1
    target = hooks_dir / HOOK_NAME
    if target.exists() and not args.force:
        existing = target.read_text(encoding="utf-8", errors="replace")
        if HOOK_MARKER in existing:
            _status(f"{target} already installed.")
            return 0
        print(
            f"outmem: {target} exists and was not installed by outmem; "
            "rerun with --force to overwrite.",
            file=sys.stderr,
        )
        return 1
    target.write_text(HOOK_SCRIPT, encoding="utf-8")
    target.chmod(0o755)
    _status(f"installed pre-commit hook → {target}")
    return 0


def cmd_hook_uninstall(args: argparse.Namespace) -> int:
    store = _open_store(args)
    target = store.root / ".git" / "hooks" / HOOK_NAME
    if not target.exists():
        _status(f"{target} is not present.")
        return 0
    existing = target.read_text(encoding="utf-8", errors="replace")
    if HOOK_MARKER not in existing and not args.force:
        print(
            f"outmem: {target} was not installed by outmem; "
            "rerun with --force to remove anyway.",
            file=sys.stderr,
        )
        return 1
    target.unlink()
    _status(f"removed pre-commit hook ← {target}")
    return 0


def cmd_lint(args: argparse.Namespace) -> int:
    from outmem.lint import format_report, lint_wiki

    store = _open_store(args)
    report = lint_wiki(
        store.wiki_path,
        log_dir=store.log_path,
        raw_dir=store.raw_path,
        sources_dir=store.sources_path,
    )
    sys.stdout.write(format_report(report))
    if report.has_errors:
        return 2
    if report.has_findings:
        return 1
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    from pathlib import Path as _Path

    source = _Path(args.source).expanduser()
    if not source.exists():
        print(f"ingest: source not found: {source}", file=sys.stderr)
        return 2

    store = _open_store(args)
    try:
        entry = store.add_source(
            source,
            into_subdir=args.into,
            rename=args.rename,
            commit=True,
        )
    except OutmemError as exc:
        print(f"outmem: {exc}", file=sys.stderr)
        return 1
    _status(f"registered {entry.rel_path} (sha256: {entry.sha256[:12]}…)")

    if args.register_only:
        return 0

    # Hand off to the agent to do the actual extraction.
    try:
        from outmem.agent import ask_sync, require_interactive_reviewer
        from outmem.exceptions import WritebackError
    except ImportError as exc:
        print(
            f"outmem: agent extraction requires `outmem[agent]` ({exc}). "
            "Source registered; rerun with `--register-only` to skip this step.",
            file=sys.stderr,
        )
        return 1

    prompt_directive = (
        f"Focus: {args.prompt}." if args.prompt else "Extract any new facts relevant to the wiki."
    )
    query = (
        f"Ingest the new source `sources/{entry.rel_path}` (sha256: {entry.sha256}). "
        f"{prompt_directive} Compare to the existing wiki via `list_pages` and "
        "`search_wiki` before writing — do not duplicate facts that are already "
        "compiled. For every wiki page you write or extend, include this source "
        "in the `provenance:` argument as a dict entry (path, sha256). After you "
        "finish, the runtime will record the ingestion automatically."
    )

    _configure_tool_logging(quiet=args.quiet)

    try:
        reviewer = require_interactive_reviewer(
            store.config.outmem.approval.required_for_writes
        )
    except OutmemError as exc:
        print(f"outmem: {exc}", file=sys.stderr)
        return 1
    try:
        result = ask_sync(
            store,
            query=query,
            model=args.model,
            push=not args.no_push,
            pull=not args.no_pull,
            record=not args.no_record,
            reviewer=reviewer,
        )
    except WritebackError as exc:
        print(f"outmem: writeback failed — {exc}", file=sys.stderr)
        return 1
    except OutmemError as exc:
        print(f"outmem: {exc}", file=sys.stderr)
        return 1

    # The agent's commits are now in history; record this ingestion against
    # the source registry. pages_touched comes from the commit subjects.
    pages = _slugs_from_commits(result.commit_subjects)
    store.record_ingestion(
        entry.rel_path,
        prompt=args.prompt,
        pages_touched=pages,
        commit=True,
    )

    _status("agent response:")
    sys.stdout.write(result.response)
    if not result.response.endswith("\n"):
        sys.stdout.write("\n")
    u = result.usage
    cache_pct = (
        (100 * u.cache_read_tokens / (u.input_tokens + u.cache_read_tokens))
        if (u.input_tokens + u.cache_read_tokens) > 0
        else 0
    )
    print(
        f"--\ningested {entry.rel_path}; pages touched: {', '.join(pages) or '(none)'}\n"
        f"tokens: {u.input_tokens:,} in / {u.output_tokens:,} out "
        f"(cache {cache_pct:.0f}% hit, {u.cache_read_tokens:,} read)",
        file=sys.stderr,
    )
    return 0


def _slugs_from_commits(subjects: tuple[str, ...]) -> list[str]:
    """Extract slugs from ``compact: <slug>`` / ``extend: <slug>`` commits.

    ``log:`` subjects are skipped — they don't produce pages.
    """
    slugs: list[str] = []
    for subj in subjects:
        for prefix in ("compact: ", "extend: "):
            if subj.startswith(prefix):
                slug = subj[len(prefix) :].strip()
                if slug and slug not in slugs:
                    slugs.append(slug)
                break
    return slugs


# ---------------------------------------------------------------------------
# Index rebuild
# ---------------------------------------------------------------------------


def cmd_import(args: argparse.Namespace) -> int:
    """Bulk-import an existing markdown vault (Obsidian, etc.) into wiki/."""
    source = Path(args.source).expanduser()
    if not source.is_dir():
        print(f"outmem: import source is not a directory: {source}", file=sys.stderr)
        return 2
    store = _open_store(args)
    try:
        summary = store.import_vault(source, force=args.force)
    except OutmemError as exc:
        print(f"outmem: {exc}", file=sys.stderr)
        return 1
    _status(
        f"imported {summary.pages_imported} page(s) from {source} "
        f"(wikilinks: {summary.wikilinks_rewritten} rewritten, "
        f"{summary.wikilinks_unresolved} unresolved)"
    )
    if summary.slug_collisions:
        _status(
            f"{len(summary.slug_collisions)} slug collision(s) resolved by parent-prefix:"
        )
        for path, slug in summary.slug_collisions[:5]:
            print(f"  {path}  →  {slug}")
        if len(summary.slug_collisions) > 5:
            print(f"  … and {len(summary.slug_collisions) - 5} more")
    if summary.wikilinks_unresolved:
        _status("run `outmem lint` to surface unresolved wikilinks.")
    return 0


def cmd_index_rebuild(args: argparse.Namespace) -> int:
    store = _open_store(args)
    try:
        sha = store.rebuild_index(commit=not args.no_commit)
    except OutmemError as exc:
        print(f"outmem: {exc}", file=sys.stderr)
        return 1
    if sha is None:
        if args.no_commit:
            _status("index.md regenerated (uncommitted; rerun without --no-commit to land).")
        else:
            _status("index.md already in sync — nothing to commit.")
        return 0
    _status(f"rebuilt index.md ({sha[:10]})")
    return 0


def _configure_tool_logging(quiet: bool) -> None:
    """Pipe ``outmem.agent.tool`` log records to stderr so users can
    follow what the agent is doing during ``outmem ask``.

    Off when ``quiet=True``. Idempotent — repeated CLI invocations in
    the same process don't stack handlers.
    """
    import logging

    logger = logging.getLogger("outmem.agent.tool")
    # Drop any handler we previously attached so re-runs don't duplicate output.
    for existing in list(logger.handlers):
        if getattr(existing, "_outmem_cli_handler", False):
            logger.removeHandler(existing)
    if quiet:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(
        fmt="[%(asctime)s] [tool] %(message)s",
        datefmt=_TIMESTAMP_FMT,
    ))
    handler._outmem_cli_handler = True  # type: ignore[attr-defined]
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    # ``propagate`` deliberately left at its default (True). Standard
    # logging convention; consumers who configure their own root handler
    # will see tool calls flow there as well, which is usually what
    # they want.


def cmd_ask(args: argparse.Namespace) -> int:
    try:
        from outmem.agent import ask_sync, require_interactive_reviewer
        from outmem.exceptions import WritebackError
    except ImportError as exc:
        print(
            f"outmem: `ask` requires `outmem[agent]` ({exc}).",
            file=sys.stderr,
        )
        return 1

    query = sys.stdin.read().strip() if args.stdin else (args.query or "")
    if not query:
        print("ask: query is empty.", file=sys.stderr)
        return 2

    _configure_tool_logging(quiet=args.quiet)

    store = _open_store(args)
    try:
        reviewer = require_interactive_reviewer(
            store.config.outmem.approval.required_for_writes
        )
    except OutmemError as exc:
        print(f"outmem: {exc}", file=sys.stderr)
        return 1
    try:
        result = ask_sync(
            store,
            query=query,
            model=args.model,
            push=not args.no_push,
            pull=not args.no_pull,
            record=not args.no_record,
            reviewer=reviewer,
        )
    except WritebackError as exc:
        print(f"outmem: writeback failed — {exc}", file=sys.stderr)
        return 1
    except OutmemError as exc:
        print(f"outmem: {exc}", file=sys.stderr)
        return 1

    _status("agent response:")
    sys.stdout.write(result.response)
    if not result.response.endswith("\n"):
        sys.stdout.write("\n")
    if args.show_meta:
        commits = ", ".join(c.sha[:10] for c in result.commits) or "(none)"
        race_note = (
            " (concurrent human commit landed during retry)"
            if result.concurrent_human_commit_landed
            else ""
        )
        u = result.usage
        cache_pct = (
            (100 * u.cache_read_tokens / (u.input_tokens + u.cache_read_tokens))
            if (u.input_tokens + u.cache_read_tokens) > 0
            else 0
        )
        print(
            f"--\nturn {result.started_at.isoformat()} → {result.finished_at.isoformat()}\n"
            f"commits: {commits}\npushed: {result.pushed}{race_note}\n"
            f"tokens: {u.input_tokens:,} in / {u.output_tokens:,} out "
            f"(cache: {u.cache_read_tokens:,} read, {u.cache_write_tokens:,} write, "
            f"{cache_pct:.0f}% hit rate)",
            file=sys.stderr,
        )
    return 0


def cmd_dashboard(args: argparse.Namespace) -> int:
    try:
        import uvicorn

        from outmem.dashboard import create_app
    except ImportError as exc:
        print(
            f"outmem: dashboard requires `outmem[dashboard]` ({exc}).",
            file=sys.stderr,
        )
        return 1

    store = _open_store(args)
    app = create_app(store, pull_on_request=args.pull_on_request)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="outmem", description="Agentic RAG memory CLI.")
    parser.add_argument("--version", action="version", version=f"outmem {__version__}")
    # --root lives on a shared parent attached to every subcommand, so it
    # goes AFTER the subcommand (`outmem reindex --root X`) — what users
    # intuitively type. $OUTMEM_PATH covers the env/scripted case.
    # SUPPRESS the default so that an inner subparser (e.g. `hook install`)
    # without --root doesn't clobber the value set at the outer level
    # (`outmem hook --root X install`) — both positions must work.
    root_parent = argparse.ArgumentParser(add_help=False)
    root_parent.add_argument(
        "--root",
        default=argparse.SUPPRESS,
        help="Wiki root (defaults to $OUTMEM_PATH or the current directory).",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # `init` takes a positional PATH and doesn't use --root (it CREATES the
    # wiki at PATH rather than opening an existing one) — so it's the lone
    # subcommand that doesn't inherit root_parent.
    p_init = sub.add_parser("init", help="Scaffold a new wiki at PATH.")
    p_init.add_argument("path")
    p_init.set_defaults(func=cmd_init)

    p_read = sub.add_parser("read", help="Print a wiki page.", parents=[root_parent])
    p_read.add_argument("slug")
    p_read.add_argument(
        "--body-only",
        action="store_true",
        help="Print just the body (no frontmatter).",
    )
    p_read.set_defaults(func=cmd_read)

    p_search = sub.add_parser("search", help="Search the wiki.", parents=[root_parent])
    p_search.add_argument("pattern")
    p_search.add_argument(
        "--scope",
        choices=("wiki", "raw", "log", "all"),
        default="wiki",
    )
    p_search.add_argument("-i", "--ignore-case", action="store_true")
    p_search.add_argument("-F", "--fixed-strings", action="store_true")
    p_search.add_argument("--max-hits", type=int, default=None)
    p_search.set_defaults(func=cmd_search)

    p_history = sub.add_parser("history", help="Per-page commit history.", parents=[root_parent])
    p_history.add_argument("slug")
    p_history.set_defaults(func=cmd_history)

    p_evolution = sub.add_parser(
        "evolution",
        help="git log -p --follow across slugs (EXPANSION pattern).", parents=[root_parent])
    p_evolution.add_argument("slugs", nargs="+")
    p_evolution.add_argument(
        "--no-log",
        action="store_true",
        help="Exclude log/ entries from the diff stream.",
    )
    p_evolution.set_defaults(func=cmd_evolution)

    p_write = sub.add_parser(
        "write", help="Create a wiki page (body on stdin).", parents=[root_parent]
    )
    p_write.add_argument("slug")
    p_write.add_argument("--title", required=True)
    p_write.add_argument(
        "--provenance",
        action="append",
        default=[],
        help="Path into raw/. Repeat for multiple sources.",
    )
    p_write.add_argument(
        "--tag",
        action="append",
        default=[],
        help="Tag value. Repeat for multiple tags.",
    )
    p_write.set_defaults(func=cmd_write)

    p_extend = sub.add_parser(
        "extend",
        help="Replace body of an existing page (body on stdin).", parents=[root_parent])
    p_extend.add_argument("slug")
    p_extend.set_defaults(func=cmd_extend)

    p_log = sub.add_parser(
        "log", help="Append a log entry (content on stdin).", parents=[root_parent]
    )
    p_log.add_argument("topic")
    p_log.set_defaults(func=cmd_log)

    p_pull = sub.add_parser(
        "pull", help="git pull --rebase from the remote.", parents=[root_parent]
    )
    p_pull.set_defaults(func=cmd_pull)

    p_push = sub.add_parser("push", help="git push to the remote.", parents=[root_parent])
    p_push.set_defaults(func=cmd_push)

    p_steering = sub.add_parser(
        "steering",
        help="Phase-1 steering signal: human commits since last run.", parents=[root_parent])
    p_steering.set_defaults(func=cmd_steering)

    p_record = sub.add_parser(
        "record-run",
        help="Record a successful agent run (timestamp + HEAD).", parents=[root_parent])
    p_record.set_defaults(func=cmd_record_run)

    p_lint = sub.add_parser(
        "lint",
        help="Run static checks over the wiki (orphans, broken links, drift).",
        parents=[root_parent],
    )
    p_lint.set_defaults(func=cmd_lint)

    p_similar = sub.add_parser(
        "similar",
        help="Find semantically similar wiki / source chunks (requires outmem[semantic]).",
        parents=[root_parent],
    )
    p_similar.add_argument(
        "text",
        nargs="?",
        default=None,
        help="Query text. Omit when using --slug or --stdin.",
    )
    p_similar.add_argument(
        "--slug",
        default=None,
        help="Use the body of this wiki page as the query, excluding itself.",
    )
    p_similar.add_argument(
        "--stdin",
        action="store_true",
        help="Read the query from stdin.",
    )
    p_similar.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of matches to return (default 5).",
    )
    p_similar.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Minimum cosine similarity. Defaults to `semantic.similarity_threshold` "
        "in config.yaml — same gate the agent tool uses. Pass 0.0 to disable.",
    )
    p_similar.set_defaults(func=cmd_similar)

    p_reindex = sub.add_parser(
        "reindex",
        help="Resync the semantic vector DB with disk.", parents=[root_parent])
    p_reindex.add_argument(
        "--force",
        action="store_true",
        help="Rebuild every entry even when its content hash matches.",
    )
    p_reindex.add_argument(
        "--staged",
        action="store_true",
        help="Pre-commit hook mode: only reindex staged changes and "
        "re-stage the vector DB.",
    )
    p_reindex.add_argument(
        "--path",
        action="append",
        default=[],
        help="Reindex a specific repo-relative path. Repeat for multiple.",
    )
    p_reindex.set_defaults(func=cmd_reindex)

    p_hook = sub.add_parser(
        "hook",
        help="Install / uninstall the outmem git pre-commit hook.", parents=[root_parent])
    hook_sub = p_hook.add_subparsers(dest="hook_command", required=True)

    p_hook_install = hook_sub.add_parser(
        "install",
        help="Drop a pre-commit hook at .git/hooks/pre-commit.",
        parents=[root_parent],
    )
    p_hook_install.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing pre-commit hook.",
    )
    p_hook_install.set_defaults(func=cmd_hook_install)

    p_hook_uninstall = hook_sub.add_parser(
        "uninstall",
        help="Remove the outmem pre-commit hook.",
        parents=[root_parent],
    )
    p_hook_uninstall.add_argument(
        "--force",
        action="store_true",
        help="Remove the hook even if it wasn't installed by outmem.",
    )
    p_hook_uninstall.set_defaults(func=cmd_hook_uninstall)

    p_ingest = sub.add_parser(
        "ingest",
        help="Register a source file and run the agent to extract from it.", parents=[root_parent])
    p_ingest.add_argument("source", help="Path to the source file.")
    p_ingest.add_argument(
        "--into",
        default=None,
        help="Subdirectory under wiki/sources/ (e.g. 'veterinary').",
    )
    p_ingest.add_argument(
        "--rename",
        default=None,
        help="Rename the file in wiki/sources/.",
    )
    p_ingest.add_argument(
        "--prompt",
        default=None,
        help="Focus directive for the agent (e.g. 'extract drug dosages for cats').",
    )
    p_ingest.add_argument(
        "--register-only",
        action="store_true",
        help="Copy + register the source; do not invoke the agent.",
    )
    p_ingest.add_argument(
        "--model",
        default=None,
        help="PydanticAI model id (defaults to $OUTMEM_MODEL or config.yaml).",
    )
    p_ingest.add_argument("--no-push", action="store_true")
    p_ingest.add_argument("--no-pull", action="store_true")
    p_ingest.add_argument("--no-record", action="store_true")
    p_ingest.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress the per-tool-call trace on stderr.",
    )
    p_ingest.set_defaults(func=cmd_ingest)

    p_import = sub.add_parser(
        "import",
        help="Bulk-import an existing markdown vault (Obsidian, etc.) into wiki/.",
        parents=[root_parent],
    )
    p_import.add_argument(
        "source",
        help="Top-level vault directory. Recursively imports every *.md, "
        "skipping hidden dirs (.obsidian/, .git/, etc.).",
    )
    p_import.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing non-empty wiki/. Without this, the "
        "command refuses to clobber a wiki that already has pages.",
    )
    p_import.set_defaults(func=cmd_import)

    p_ask = sub.add_parser(
        "ask",
        help="Ask the agent a question (requires outmem[agent]).", parents=[root_parent])
    p_ask.add_argument("query", nargs="?", default=None, help="Question text.")
    p_ask.add_argument("--stdin", action="store_true", help="Read query from stdin.")
    p_ask.add_argument(
        "--model",
        default=None,
        help="PydanticAI model id (defaults to $OUTMEM_MODEL).",
    )
    p_ask.add_argument("--no-push", action="store_true")
    p_ask.add_argument("--no-pull", action="store_true")
    p_ask.add_argument("--no-record", action="store_true")
    p_ask.add_argument(
        "--show-meta",
        action="store_true",
        help="Print run metadata (commits, timing) on stderr after the response.",
    )
    p_ask.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress the per-tool-call trace on stderr (on by default).",
    )
    p_ask.set_defaults(func=cmd_ask)

    p_dashboard = sub.add_parser(
        "dashboard",
        help="Start the read-only FastAPI dashboard (requires outmem[dashboard]).",
        parents=[root_parent],
    )
    p_dashboard.add_argument("--host", default="127.0.0.1")
    p_dashboard.add_argument("--port", type=int, default=8765)
    p_dashboard.add_argument("--log-level", default="info")
    p_dashboard.add_argument(
        "--pull-on-request",
        action="store_true",
        help="git pull --rebase before every request (default off).",
    )
    p_dashboard.set_defaults(func=cmd_dashboard)

    p_index = sub.add_parser(
        "index",
        help="Operate on wiki/index.md (currently: rebuild).", parents=[root_parent])
    index_sub = p_index.add_subparsers(dest="index_command", required=True)

    p_index_rebuild = index_sub.add_parser(
        "rebuild",
        help="Regenerate wiki/index.md from the current wiki tree. Use "
        "after manual edits (Obsidian, vim) that add / rename / "
        "delete pages or change titles / tags. No-op when the index "
        "is already in sync.",
        parents=[root_parent],
    )
    p_index_rebuild.add_argument(
        "--no-commit",
        action="store_true",
        help="Rewrite index.md on disk but do not commit. Useful when "
        "you'll combine the index rebuild with other staged edits in "
        "the same commit.",
    )
    p_index_rebuild.set_defaults(func=cmd_index_rebuild)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except OutmemError as exc:
        print(f"outmem: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
