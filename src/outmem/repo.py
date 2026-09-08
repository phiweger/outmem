"""Several wikis in one git repository — the ``wikis.yaml`` registry.

A wiki is the unit of access. Within a wiki everything is open; a
session is permitted a *set of wikis* up front, and every store it then
holds is one it may read in full. Nothing downstream decides what to
hide, because nothing is hidden — which is the whole point of the
arrangement.

The repository holds the wikis side by side::

    /srv/memory/
      wikis.yaml
      wikis/
        open/     <- an ordinary wiki root: config.yaml, wiki/, log/, …
        hr/
        legal/

Each wiki directory is exactly what ``outmem init`` produces, so a wiki
moves out of the repo and opens standalone, and a standalone wiki moves
in, with ``git mv`` and no conversion step.

``wikis.yaml`` names the wikis and the audience tags that reach them::

    version: 1
    tags:
      everyone: {description: "All employees"}
      hr:       {description: "People team"}
    wikis:
      open: {path: wikis/open, title: "Handbook", audience: [everyone]}
      hr:   {path: wikis/hr,   title: "People",   audience: [hr]}

An audience tag is opaque to outmem. The host application maps its
authenticated user to a set of tags and hands them in; outmem does one
set-overlap test on *names*, once, before a store exists. Nothing in
this file is derived from the wiki's contents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from outmem.exceptions import OutmemError

REGISTRY_FILENAME = "wikis.yaml"

# How far up the tree :func:`find_repo_root` will look. A wiki nested more
# deeply than this inside its repo is possible but not a layout outmem
# creates, and an unbounded walk on a broken path is worth avoiding.
_MAX_WALK_UP = 8


@dataclass(frozen=True)
class WikiEntry:
    """One wiki as ``wikis.yaml`` declares it."""

    name: str
    path: str
    title: str
    audience: frozenset[str]


@dataclass(frozen=True)
class Registry:
    """A parsed ``wikis.yaml``, with the repo root it was found at."""

    root: Path
    version: int = 1
    tags: dict[str, str] = field(default_factory=dict)
    wikis: dict[str, WikiEntry] = field(default_factory=dict)

    def path_of(self, name: str) -> Path:
        """Absolute path of the wiki directory named ``name``."""
        return self.root / self.wikis[name].path

    def name_at(self, wiki_root: Path) -> str | None:
        """The registry name for the directory at ``wiki_root``, if listed.

        Compared by resolved path rather than by string: a caller may
        reach the same directory through a symlink, a relative path, or a
        ``..`` detour, and a listing is about the directory itself.
        """
        try:
            target = wiki_root.resolve()
        except OSError:  # pragma: no cover — unreadable path
            return None
        for name in self.wikis:
            try:
                if self.path_of(name).resolve() == target:
                    return name
            except OSError:  # pragma: no cover — unreadable entry
                continue
        return None


def _fail(path: Path, detail: str) -> OutmemError:
    return OutmemError(f"{path}: {detail}")


def load_registry(root: Path) -> Registry | None:
    """Parse ``<root>/wikis.yaml``, or return ``None`` if there is none.

    A *missing* file is not an error — that is an ordinary single-wiki
    directory, and the overwhelming majority of them. A file that exists
    but does not parse is, because the alternative is treating a
    multi-wiki repository as a single wiki and committing one wiki's
    writes into another's history.
    """
    path = root / REGISTRY_FILENAME
    if not path.is_file():
        return None
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise _fail(path, f"could not be read as YAML — {exc}") from exc
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise _fail(path, "must be a YAML mapping.")

    version = raw.get("version", 1)
    if not isinstance(version, int):
        raise _fail(path, f"`version` must be an integer, got {version!r}.")

    tags = _parse_tags(path, raw.get("tags"))
    wikis = _parse_wikis(path, raw.get("wikis"))
    return Registry(root=root, version=version, tags=tags, wikis=wikis)


def _parse_tags(path: Path, block: object) -> dict[str, str]:
    """The declared tag vocabulary, as ``{name: description}``.

    Declaring the vocabulary is what makes a typo loud. A tag used in an
    ``audience:`` but never declared would otherwise make a wiki silently
    unreachable — nobody can be granted a tag nobody knows exists, so the
    content is simply gone with no error anywhere.
    """
    if block is None:
        return {}
    if not isinstance(block, dict):
        raise _fail(path, "`tags` must be a mapping of tag name to settings.")
    out: dict[str, str] = {}
    for name, settings in block.items():
        if not isinstance(name, str) or not name:
            raise _fail(path, f"tag names must be non-empty strings, got {name!r}.")
        if settings is None:
            out[name] = ""
        elif isinstance(settings, str):
            out[name] = settings
        elif isinstance(settings, dict):
            description = settings.get("description", "")
            out[name] = description if isinstance(description, str) else ""
        else:
            raise _fail(path, f"tag {name!r} must map to a mapping or a string.")
    return out


def _parse_wikis(path: Path, block: object) -> dict[str, WikiEntry]:
    if block is None:
        return {}
    if not isinstance(block, dict):
        raise _fail(path, "`wikis` must be a mapping of wiki name to settings.")
    out: dict[str, WikiEntry] = {}
    for name, settings in block.items():
        if not isinstance(name, str) or not name:
            raise _fail(path, f"wiki names must be non-empty strings, got {name!r}.")
        if not isinstance(settings, dict):
            raise _fail(path, f"wiki {name!r} must map to a mapping.")
        rel = settings.get("path", f"wikis/{name}")
        if not isinstance(rel, str) or not rel:
            raise _fail(path, f"wiki {name!r}: `path` must be a non-empty string.")
        if Path(rel).is_absolute() or ".." in Path(rel).parts:
            raise _fail(
                path,
                f"wiki {name!r}: `path` must stay inside the repository, "
                f"got {rel!r}.",
            )
        title = settings.get("title", name)
        audience = settings.get("audience", [])
        if isinstance(audience, str):
            audience = [audience]
        if not isinstance(audience, list) or not all(
            isinstance(t, str) for t in audience
        ):
            raise _fail(path, f"wiki {name!r}: `audience` must be a list of tags.")
        out[name] = WikiEntry(
            name=name,
            path=rel.rstrip("/"),
            title=title if isinstance(title, str) else name,
            audience=frozenset(audience),
        )
    return out


def find_repo_root(wiki_root: Path) -> tuple[Path, str]:
    """The git repository ``wiki_root`` commits into, and its prefix within it.

    Returns ``(repo_root, prefix)``, where ``prefix`` is the wiki's
    location relative to the repo as a POSIX string ending in ``/`` — or
    ``""`` when the wiki *is* the repo, which is every standalone wiki
    and therefore the overwhelmingly common case.

    Discovery is deliberately not "walk up until you find ``.git``". A
    wiki that happens to sit inside an unrelated repository
    (``~/projects/notes`` under ``~/projects/.git``) refuses commits
    today; an unconditional walk would silently start committing into
    that parent instead. So an ancestor is only accepted when its
    ``wikis.yaml`` *lists this directory*: multi-wiki is opted into by a
    file somebody deliberately wrote, and every existing layout keeps the
    behaviour it has.
    """
    try:
        start = wiki_root.resolve()
    except OSError:  # pragma: no cover — unreadable path
        return wiki_root, ""
    for ancestor in list(start.parents)[:_MAX_WALK_UP]:
        registry = load_registry(ancestor)
        if registry is None:
            continue
        if registry.name_at(start) is not None:
            return ancestor, f"{start.relative_to(ancestor).as_posix()}/"
    return wiki_root, ""
