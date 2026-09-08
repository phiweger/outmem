"""``outmem repo …`` — the multi-wiki repository commands.

Registering wikis, inspecting the audience vocabulary, and bringing an
existing wiki in. Everything a wiki *contains* is still reached through
the ordinary subcommands with ``--wiki NAME``; this group is about the
repository around them.

These commands take ``--root`` as the *repository*, and deliberately not
``--wiki`` — there is no wiki to name when the subject is the registry.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from outmem.cli._common import agent_identity, base_root, status
from outmem.config import DEFAULT_BRANCH
from outmem.exceptions import OutmemError
from outmem.repo import (
    REGISTRY_FILENAME,
    Registry,
    Repo,
    is_wiki_root,
    load_registry,
    validate_wiki_path,
)
from outmem.store import WikiStore, ensure_gitignored

# Plain on purpose. `repo add` round-trips this file through the YAML
# loader, which drops comments — so guidance written here would survive
# exactly until the first wiki was added. The guidance lives in
# docs/multi-wiki.md, which `repo init` points at.
_STARTER_REGISTRY = "version: 1\ntags: {}\nwikis: {}\n"

# Repository-level ignores, appended one at a time and only when absent.
# `.outmem-repo/` carries its own self-ignoring `.gitignore`, so the entry
# here is belt-and-braces for anyone reading the repo root.
_REPO_IGNORES = (
    (".outmem-repo/", "# outmem: repository-level state, not part of any wiki."),
    (".env", "# outmem: secrets stay out of the repository."),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_raw(registry_path: Path, text: str) -> dict[str, object]:
    raw = yaml.safe_load(text) or {}
    if not isinstance(raw, dict):
        raise OutmemError(f"{registry_path}: must be a YAML mapping.")
    return raw


def _has_comments(text: str) -> bool:
    """True if a load-and-dump of ``text`` would drop something a person wrote.

    Exact without a comment-aware parser: every ``#`` in a YAML document
    is either inside a scalar — a key or a value, and both come back from
    the loader — or it is a comment. Count them on both sides; any surplus
    in the raw text is commentary. ``title: "Issue #12"`` is not a false
    positive, because that ``#`` is in the loaded string.
    """
    raw = text.count("#")
    if raw == 0:
        return False

    def inside(node: object) -> int:
        if isinstance(node, str):
            return node.count("#")
        if isinstance(node, dict):
            return sum(inside(k) + inside(v) for k, v in node.items())
        if isinstance(node, list):
            return sum(inside(x) for x in node)
        return 0

    return raw > inside(yaml.safe_load(text))


def _refuse_rewrite(registry_path: Path, name: str, *, then: str) -> int:
    """The registry is hand-maintained; say so and stop, changing nothing.

    ``wikis.yaml`` is the one file in the system a person is meant to
    keep — it is the contract with their user database — so it is exactly
    the file that carries "mirrors the IdP groups" and "see ADR-014".
    Rewriting it through the YAML loader drops every one of those and
    reformats the rest, and committed the result. outmem does not rewrite
    a file somebody maintains; it tells them what to add.
    """
    print(
        f"outmem: {registry_path} has comments that a rewrite would drop, so "
        f"it was left alone. Add an entry for {name!r} under `wikis:` by hand "
        f"(and any new tags under `tags:`), then run `{then}` again.",
        file=sys.stderr,
    )
    return 1


def _save_raw(registry_path: Path, raw: dict[str, object]) -> None:
    registry_path.write_text(
        yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def _open_registry(root: Path) -> Registry | None:
    """The parsed registry, or ``None`` after telling the user there is none."""
    registry = load_registry(root)
    if registry is None:
        print(
            f"outmem: {root} is not a multi-wiki repository — run `outmem repo init` first.",
            file=sys.stderr,
        )
    return registry


def _register_entry(
    raw: dict[str, object], *, name: str, rel: str, title: str | None, audience: list[str]
) -> dict[str, object]:
    """Add a wiki entry (and any new tags) to the loaded registry.

    Returns the ``wikis`` mapping the entry went into, so a caller that
    has to undo the write can. The shape checks cannot fail for a file
    `load_registry` just accepted; they are here so a surprise is an
    `OutmemError` with a location rather than an `AssertionError`.
    """
    wikis = raw.setdefault("wikis", {})
    tags = raw.setdefault("tags", {})
    if not isinstance(wikis, dict) or not isinstance(tags, dict):
        raise OutmemError(f"{REGISTRY_FILENAME}: `wikis` and `tags` must be mappings.")
    for tag in audience:
        tags.setdefault(tag, {"description": ""})
    wikis[name] = {"path": rel, "title": title or name, "audience": list(audience)}
    return wikis


def _commit_registry(root: Path, *, paths: list[str], subject: str) -> None:
    """Commit the registry itself.

    Unlike a wiki's scaffold — which `outmem init` leaves untracked for
    the author's first write to carry in — `wikis.yaml` is what makes a
    directory a multi-wiki repository, and `find_repo_root` reads it from
    the working tree. Untracked, it would not survive a clone: every wiki
    would look standalone, and the next `WikiStore.init` would nest a
    `.git` inside the repo instead of joining it.
    """
    from outmem.git_ops import add as git_add
    from outmem.git_ops import commit_as, staged_changes

    identity = agent_identity()
    git_add(root, paths)
    added, deleted = staged_changes(root)
    if not added and not deleted:
        # Re-running a setup command that changed nothing is a no-op, not
        # a failure. Without this, git's own "nothing to commit" surfaces
        # as an error from a command that did exactly what was asked.
        return
    commit_as(root, message=subject, author_name=identity.name, author_email=identity.email)


def _git_mv(root: Path, source: Path, target: Path) -> None:
    """``git mv`` inside ``root``, falling back to a plain rename.

    An untracked directory cannot be ``git mv``-ed — there is nothing to
    move in the index — but moving it is still the right thing, so the
    failure is not fatal.
    """
    rel_source = source.relative_to(root.resolve()).as_posix()
    rel_target = target.relative_to(root.resolve()).as_posix()
    result = subprocess.run(
        ["git", "mv", "--", rel_source, rel_target],
        cwd=str(root),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        source.rename(target)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_repo_init(args: argparse.Namespace) -> int:
    """Scaffold a multi-wiki repository: a git repo plus an empty registry."""
    from outmem.git_ops import init_repo

    root = base_root(args)
    root.mkdir(parents=True, exist_ok=True)
    registry_path = root / REGISTRY_FILENAME
    if registry_path.exists():
        print(f"outmem: {registry_path} already exists.", file=sys.stderr)
        return 1
    init_repo(root, initial_branch=args.branch)
    registry_path.write_text(_STARTER_REGISTRY, encoding="utf-8")
    # Append; never rewrite. `repo init` is routinely run in a directory
    # that already has a `.gitignore` — writing ours over it silently
    # dropped whatever was there (`__pycache__/`, `*.pyc`, `.vectors.db`)
    # and committed the loss.
    touched = [
        ensure_gitignored(root, pattern, comment=comment) for pattern, comment in _REPO_IGNORES
    ]
    paths = [REGISTRY_FILENAME] + ([".gitignore"] if any(touched) else [])
    try:
        _commit_registry(root, paths=paths, subject="repo: initialise")
    except OutmemError as exc:
        print(f"outmem: {exc}", file=sys.stderr)
        return 1
    status(f"initialised multi-wiki repository at {root}")
    status("add a wiki with `outmem repo add <name> --audience <tag>`; see docs/multi-wiki.md")
    return 0


def cmd_repo_add(args: argparse.Namespace) -> int:
    """Scaffold a wiki inside the repository, registering it first if needed.

    Two modes, chosen by whether ``wikis.yaml`` already lists the name.

    **Listed:** the registry is the source of truth and is not touched —
    the wiki is scaffolded at the path the entry gives. This is the path
    for a hand-maintained registry: edit the YAML, then ``repo add``.

    **Unlisted:** the entry is written and the wiki scaffolded — but only
    if the file carries no comments. A commented registry is somebody's
    document, and a load-and-dump would flatten it; they are told what to
    add by hand instead.
    """
    root = base_root(args)
    registry = _open_registry(root)
    if registry is None:
        return 1
    registry_path = root / REGISTRY_FILENAME
    if args.name in registry.wikis:
        return _scaffold_listed(registry, args)

    # Validate the flag before anything touches disk. Acting first and
    # letting the parser reject the entry afterwards created the directory
    # *outside* the repository and only then rolled the entry back.
    rel = validate_wiki_path(args.path or f"wikis/{args.name}", context="--path")
    text = registry_path.read_text(encoding="utf-8")
    if _has_comments(text):
        return _refuse_rewrite(registry_path, args.name, then=f"outmem repo add {args.name}")
    raw = _parse_raw(registry_path, text)
    wikis = _register_entry(
        raw, name=args.name, rel=rel, title=args.title, audience=args.audience
    )
    # Write the registry BEFORE scaffolding: `WikiStore.init` discovers its
    # repository by looking itself up here, and an unlisted directory would
    # nest a `.git` inside the repo instead of joining it.
    _save_raw(registry_path, raw)
    (root / rel).mkdir(parents=True, exist_ok=True)
    try:
        WikiStore.init(root / rel, agent_identity=agent_identity())
        _commit_registry(root, paths=[REGISTRY_FILENAME], subject=f"repo: add {args.name}")
    except OutmemError as exc:
        # Roll the entry back. Leaving it would list a wiki that is not
        # one — reachable by name, openable by nobody, and a state the
        # operator has no reason to expect after a command that failed.
        del wikis[args.name]
        _save_raw(registry_path, raw)
        print(f"outmem: {exc}", file=sys.stderr)
        print(
            f"outmem: rolled back the {REGISTRY_FILENAME} entry for "
            f"{args.name!r}; {rel} may need removing by hand.",
            file=sys.stderr,
        )
        return 1
    status(f"registered wiki {args.name!r} at {rel}")
    return 0


def _scaffold_listed(registry: Registry, args: argparse.Namespace) -> int:
    """``repo add`` for a name the registry already carries: scaffold only."""
    if args.audience or args.title or args.path:
        print(
            f"outmem: wiki {args.name!r} is already listed in {REGISTRY_FILENAME}; "
            "its audience, title and path come from there. Edit the file to "
            "change them, then run `repo add` with no flags to scaffold.",
            file=sys.stderr,
        )
        return 1
    path = registry.path_of(args.name)
    rel = registry.wikis[args.name].path
    if is_wiki_root(path):
        status(f"wiki {args.name!r} is already scaffolded at {rel}; nothing to do")
        return 0
    path.mkdir(parents=True, exist_ok=True)
    WikiStore.init(path, agent_identity=agent_identity())
    status(f"scaffolded wiki {args.name!r} at {rel} ({REGISTRY_FILENAME} untouched)")
    return 0


def cmd_repo_import(args: argparse.Namespace) -> int:
    """Move an existing wiki into a multi-wiki repository.

    Two cases, and they differ in what happens to history.

    A wiki already inside the repository is moved with ``git mv``, so
    every tracked file keeps its history and ``git log --follow`` still
    works across the move.

    A wiki from elsewhere is copied in and committed as new content. Its
    own history stays in its own repository — merging two histories is
    `git subtree`/`filter-repo` work, and doing it badly is worse than
    not doing it, so this says so rather than pretending.
    """
    from outmem.git_ops import is_git_repo

    root = base_root(args)
    registry = _open_registry(root)
    if registry is None:
        return 1
    registry_path = root / REGISTRY_FILENAME
    source = Path(args.path).expanduser().resolve()
    if not is_wiki_root(source):
        print(f"outmem: {source} does not look like a wiki (no config.yaml).", file=sys.stderr)
        return 1
    # Everything that can refuse, refuses here — before the move. Refusing
    # afterwards left the wiki relocated (outside the repository, for a
    # bad `--path-in-repo`) with an entry that no longer parsed.
    wanted = (
        validate_wiki_path(args.path_in_repo, context="--path-in-repo")
        if args.path_in_repo
        else None
    )
    listed = args.name in registry.wikis
    text = registry_path.read_text(encoding="utf-8")
    if listed:
        # The registry is the source of truth: the wiki goes where the
        # entry says, and the entry is not rewritten.
        entry = registry.wikis[args.name]
        if args.audience or args.title or (wanted is not None and wanted != entry.path):
            print(
                f"outmem: wiki {args.name!r} is already listed in {REGISTRY_FILENAME}; "
                "its audience, title and path come from there.",
                file=sys.stderr,
            )
            return 1
        rel = entry.path
    else:
        if _has_comments(text):
            return _refuse_rewrite(
                registry_path,
                args.name,
                then=f"outmem repo import {args.path} --name {args.name}",
            )
        rel = wanted or f"wikis/{args.name}"
    target = root / rel
    if target.exists():
        print(f"outmem: {target} already exists.", file=sys.stderr)
        return 1

    inside = source.is_relative_to(root.resolve())
    if inside and (source / ".git").exists():
        # Moving it would leave a nested repository inside this one: git
        # then treats the directory as a foreign checkout and refuses to
        # stage it ("does not have a commit checked out"). Removing the
        # nested `.git` discards that wiki's history, which is the
        # operator's call to make, not this command's.
        print(
            f"outmem: {source} has its own git repository. A move cannot "
            f"carry its history into this one — remove {source / '.git'} "
            "first (this discards that history), or import from outside "
            "the repository to copy the working tree.",
            file=sys.stderr,
        )
        return 1
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        if inside:
            _git_mv(root, source, target)
        else:
            shutil.copytree(source, target, ignore=shutil.ignore_patterns(".git"))
    except (OSError, OutmemError) as exc:
        print(f"outmem: could not move the wiki: {exc}", file=sys.stderr)
        return 1

    paths = [rel]
    if not listed:
        raw = _parse_raw(registry_path, text)
        _register_entry(raw, name=args.name, rel=rel, title=args.title, audience=args.audience)
        _save_raw(registry_path, raw)
        paths.insert(0, REGISTRY_FILENAME)
    try:
        _commit_registry(root, paths=paths, subject=f"repo: import {args.name}")
    except OutmemError as exc:
        print(f"outmem: {exc}", file=sys.stderr)
        return 1

    status(f"imported {source} as wiki {args.name!r} at {rel}")
    if not inside:
        if is_git_repo(source):
            status(
                f"note: {source} keeps its own git history — the copy at {rel} "
                "starts fresh. Move it inside the repository first if you need "
                "`git log --follow` across the boundary."
            )
        status(f"the original at {source} was left in place; remove it yourself")
    return 0


def cmd_repo_list(args: argparse.Namespace) -> int:
    """The catalogue: which wikis exist and which tags reach them."""
    repo = Repo.open(base_root(args))
    catalogue = repo.catalogue_for(args.audience) if args.audience else repo.catalogue()
    if args.json:
        print(json.dumps(catalogue.as_dict(), indent=2))
        return 0
    if not catalogue.wikis:
        print("outmem: no wikis registered.", file=sys.stderr)
        return 1
    width = max(len(w.name) for w in catalogue.wikis)
    for w in catalogue.wikis:
        audience = ", ".join(w.audience) or "(nobody)"
        print(f"{w.name:<{width}}  {w.pages:>5} pages  [{audience}]  {w.title}")
    return 0


def cmd_repo_tags(args: argparse.Namespace) -> int:
    """The declared vocabulary — what a user database provisions against."""
    repo = Repo.open(base_root(args))
    if args.json:
        print(json.dumps(repo.catalogue().as_dict(), indent=2))
        return 0
    tags = repo.tags()
    if not tags:
        print("outmem: no tags declared.", file=sys.stderr)
        return 1
    width = max(len(t.name) for t in tags)
    for tag in tags:
        reaches = ", ".join(tag.wikis) or "(nothing)"
        suffix = f"  — {tag.description}" if tag.description else ""
        print(f"{tag.name:<{width}}  -> {reaches}{suffix}")
    return 0


def cmd_repo_audience(args: argparse.Namespace) -> int:
    """What a user holding these tags would get. The support-ticket tool."""
    repo = Repo.open(base_root(args))
    held = set(args.tags)
    catalogue = repo.catalogue_for(held)
    # `reconcile` answers a repository-wide question — which tags nobody
    # holds, which wikis nobody reaches. Asked about one user, only the
    # `unknown` half means anything: "wikis you cannot see" is the normal
    # condition, not a finding.
    unknown = repo.reconcile(held).unknown
    if args.json:
        payload = dict(catalogue.as_dict())
        payload["unknown_tags"] = list(unknown)
        print(json.dumps(payload, indent=2))
        return 0
    if not catalogue.wikis:
        print(
            f"outmem: tags {', '.join(sorted(held)) or '(none)'} reach no wiki.",
            file=sys.stderr,
        )
    for w in catalogue.wikis:
        print(f"{w.name}  {w.title}")
    if unknown:
        print(
            f"outmem: no wiki declares {', '.join(unknown)} — a stale grant or a typo.",
            file=sys.stderr,
        )
    return 0 if catalogue.wikis else 1


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def register(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Attach the ``repo`` command group to the top-level subparsers."""
    # Its own parent: `--root` is the repository here, and `--wiki` would
    # be meaningless — the subject of every command below is the registry.
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument(
        "--root",
        default=argparse.SUPPRESS,
        help="Repository root (defaults to $OUTMEM_PATH or the current directory).",
    )

    p_repo = sub.add_parser(
        "repo",
        help="Multi-wiki repository: register wikis and inspect audience tags.",
        parents=[parent],
    )
    repo_sub = p_repo.add_subparsers(dest="repo_command", required=True)

    p_init = repo_sub.add_parser(
        "init",
        help="Scaffold a multi-wiki repository (git repo + wikis.yaml).",
        parents=[parent],
    )
    p_init.add_argument("--branch", default=DEFAULT_BRANCH)
    p_init.set_defaults(func=cmd_repo_init)

    p_add = repo_sub.add_parser(
        "add",
        help="Register and scaffold a new wiki inside the repository.",
        parents=[parent],
    )
    p_add.add_argument("name")
    p_add.add_argument(
        "--audience",
        action="append",
        default=[],
        metavar="TAG",
        help="Audience tag that reaches this wiki (repeatable).",
    )
    p_add.add_argument("--title", default=None)
    p_add.add_argument("--path", default=None, help="Directory, relative to the repo root.")
    p_add.set_defaults(func=cmd_repo_add)

    p_list = repo_sub.add_parser(
        "list",
        help="List registered wikis, their audience tags and page counts.",
        parents=[parent],
    )
    p_list.add_argument("--json", action="store_true")
    p_list.add_argument(
        "--audience",
        action="append",
        default=[],
        metavar="TAG",
        help="Show only what these tags reach (repeatable).",
    )
    p_list.set_defaults(func=cmd_repo_list)

    p_tags = repo_sub.add_parser(
        "tags",
        help="The declared tag vocabulary — provision your user DB from this.",
        parents=[parent],
    )
    p_tags.add_argument("--json", action="store_true")
    p_tags.set_defaults(func=cmd_repo_tags)

    p_audience = repo_sub.add_parser(
        "audience",
        help="What a user holding these tags would see.",
        parents=[parent],
    )
    p_audience.add_argument(
        "--tags",
        required=True,
        type=lambda s: [t for t in s.split(",") if t],
        help="Comma-separated tags the user holds.",
    )
    p_audience.add_argument("--json", action="store_true")
    p_audience.set_defaults(func=cmd_repo_audience)

    p_import = repo_sub.add_parser(
        "import",
        help="Move an existing wiki into the repository and register it.",
        parents=[parent],
    )
    p_import.add_argument("path", help="The wiki directory to import.")
    p_import.add_argument("--name", required=True)
    p_import.add_argument("--audience", action="append", default=[], metavar="TAG")
    p_import.add_argument("--title", default=None)
    p_import.add_argument(
        "--path-in-repo",
        default=None,
        help="Destination, relative to the repo root (default wikis/<name>).",
    )
    p_import.set_defaults(func=cmd_repo_import)
