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

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from outmem.exceptions import OutmemError

if TYPE_CHECKING:
    from outmem.store import WikiStore
    from outmem.wikiset import WikiSet

REGISTRY_FILENAME = "wikis.yaml"

# How far up the tree :func:`find_repo_root` will look. A wiki nested more
# deeply than this inside its repo is possible but not a layout outmem
# creates, and an unbounded walk on a broken path is worth avoiding.
_MAX_WALK_UP = 8

# A wiki name appears in commit subjects (``legal/ write: nda``) and, later,
# as the qualifier on a slug (``legal/nda``). Both need it to contain no
# whitespace and no ``/``, so the grammar stays unambiguous in both places.
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


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
        if not isinstance(name, str) or not _NAME_RE.match(name):
            raise _fail(
                path,
                f"wiki name {name!r} must be lowercase letters, digits, "
                "`.`, `_` or `-`, starting with a letter or digit — it "
                "appears in commit subjects and as a slug qualifier.",
            )
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


def find_repo_root(wiki_root: Path) -> tuple[Path, str, str | None]:
    """The git repository ``wiki_root`` commits into, its prefix, and its name.

    Returns ``(repo_root, prefix, name)``. ``prefix`` is the wiki's
    location relative to the repo as a POSIX string ending in ``/`` — or
    ``""`` when the wiki *is* the repo, which is every standalone wiki
    and therefore the overwhelmingly common case. ``name`` is the
    registry name, or ``None`` for a standalone wiki, which has none:
    there is nothing to distinguish it from.

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
        return wiki_root, "", None
    for ancestor in list(start.parents)[:_MAX_WALK_UP]:
        registry = load_registry(ancestor)
        if registry is None:
            continue
        name = registry.name_at(start)
        if name is not None:
            return ancestor, f"{start.relative_to(ancestor).as_posix()}/", name
    return wiki_root, "", None


def qualify_subject(subject: str, wiki_name: str | None) -> str:
    """Tag a commit subject with the wiki it belongs to.

    ``legal/ write: nda``. Several wikis share one history, so a bare
    ``write: nda`` in ``git log --oneline`` no longer says which wiki
    moved. A standalone wiki has no name and keeps its subjects exactly
    as they were.
    """
    return f"{wiki_name}/ {subject}" if wiki_name else subject


def split_subject(subject: str) -> tuple[str | None, str]:
    """Inverse of :func:`qualify_subject` — ``(wiki_name, rest)``.

    Lives next to its inverse because the two have to agree: the
    steering path recovers the item a commit is about by matching the
    verb at the front of the subject, and a qualifier it does not know
    to strip makes every commit in a multi-wiki repo unrecognisable.

    A subject that does not carry a qualifier comes back ``(None,
    subject)`` unchanged, which is both the standalone case and any
    commit a human wrote by hand.
    """
    head, sep, rest = subject.partition("/ ")
    if sep and _NAME_RE.match(head):
        return head, rest
    return None, subject


# ---------------------------------------------------------------------------
# Discovery
#
# The host's user database has to *name* the tags it assigns, so it needs to
# enumerate the vocabulary rather than only filter with it. Everything here
# is metadata about which wikis exist and who reaches them — never wiki
# content — and it is what a provisioning flow reads.
#
# The host stores tags, never wiki names. The tag vocabulary is the stable
# contract; wikis can be renamed, split, merged or moved underneath it
# without touching a single user record.
# ---------------------------------------------------------------------------

# Bumped when the shape below changes incompatibly. Provisioning scripts in
# other codebases read this payload, so it is a public interface.
CATALOGUE_VERSION = 1


@dataclass(frozen=True)
class TagInfo:
    """One declared audience tag, and which wikis it reaches."""

    name: str
    description: str
    wikis: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "description": self.description,
            "wikis": list(self.wikis),
        }


@dataclass(frozen=True)
class WikiInfo:
    """One wiki as the catalogue describes it."""

    name: str
    title: str
    audience: tuple[str, ...]
    pages: int

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "title": self.title,
            "audience": list(self.audience),
            "pages": self.pages,
        }


@dataclass(frozen=True)
class Catalogue:
    """What wikis exist and which tags reach them."""

    version: int
    tags: tuple[TagInfo, ...]
    wikis: tuple[WikiInfo, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "tags": [t.as_dict() for t in self.tags],
            "wikis": [w.as_dict() for w in self.wikis],
        }


@dataclass(frozen=True)
class Reconciliation:
    """Where the registry and the host's user table have drifted apart."""

    unknown: tuple[str, ...]
    unreachable: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "unknown": list(self.unknown),
            "unreachable": list(self.unreachable),
        }


def _count_pages(wiki_root: Path) -> int:
    """Editorial pages in a wiki, counted without opening a store.

    A catalogue may cover dozens of wikis and is read by provisioning
    code that wants a number, not a corpus — opening each wiki to get it
    would mean a SQLite handle and a config parse per row.
    """
    pages = wiki_root / "wiki" / "pages"
    if not pages.is_dir():
        return 0
    return sum(1 for p in pages.rglob("*.md") if p.is_file())


class Repo:
    """Several wikis in one repository, addressed by name.

    Opening is deliberately narrow. There is no accessor that returns
    every wiki without an argument: :meth:`wiki` takes the audience it
    is opening on behalf of, and the unrestricted path is the separately
    named :meth:`wiki_as_operator`. Getting the whole repository is
    therefore something a caller says out loud, not something they reach
    by passing the obvious argument.
    """

    def __init__(self, registry: Registry) -> None:
        self.registry = registry
        self.root = registry.root

    @classmethod
    def open(cls, path: str | Path) -> Repo:
        """Open the multi-wiki repository rooted at ``path``."""
        root = Path(path).expanduser()
        registry = load_registry(root)
        if registry is None:
            raise OutmemError(
                f"{root} is not a multi-wiki repository — no "
                f"{REGISTRY_FILENAME}. Create one with `outmem repo init`."
            )
        return cls(registry)

    # -- opening -------------------------------------------------------

    def wikis_for(self, audience: Iterable[str]) -> list[str]:
        """Names of the wikis an audience reaches, in declared order.

        Reachability is a plain set overlap: one shared tag is enough. A
        wiki declaring no audience is reachable by nobody, which
        :meth:`reconcile` reports rather than leaving to be discovered.
        """
        held = frozenset(audience)
        return [n for n, e in self.registry.wikis.items() if e.audience & held]

    def wiki(
        self, name: str, *, audience: Iterable[str], **kwargs: object
    ) -> WikiStore:
        """Open one wiki on behalf of an audience.

        A name the audience does not reach fails exactly as an unknown
        name does. The two are one message on purpose: a caller that can
        tell them apart can enumerate the wikis it may not open, and a
        wiki's *name* can itself be the sensitive part.
        """
        if name not in self.wikis_for(audience):
            raise OutmemError(f"no such wiki: {name!r}")
        return self._open(name, **kwargs)

    def wiki_as_operator(self, name: str, **kwargs: object) -> WikiStore:
        """Open one wiki with no audience check — maintenance and admin.

        Named for what it is so that reaching for it is a decision.
        """
        if name not in self.registry.wikis:
            raise OutmemError(f"no such wiki: {name!r}")
        return self._open(name, **kwargs)

    def wikiset(self, *, audience: Iterable[str], **kwargs: object) -> WikiSet:
        """Open every wiki this audience reaches, read as one.

        The ordinary shape of a served session: the open core plus
        whatever compartments the user's tags allow, presented to the
        model as a single knowledge base. The access decision is made
        here, once, by choosing which stores go in — everything
        downstream reads what it was handed in full.
        """
        from outmem.wikiset import WikiSet

        names = self.wikis_for(audience)
        if not names:
            raise OutmemError(
                "these audience tags reach no wiki in this repository."
            )
        return WikiSet([self._open(n, **kwargs) for n in names])

    def _open(self, name: str, **kwargs: object) -> WikiStore:
        from outmem.store import WikiStore as _WikiStore

        path = self.registry.path_of(name)
        if not path.is_dir():
            raise OutmemError(
                f"wiki {name!r} is listed in {REGISTRY_FILENAME} but "
                f"{path} does not exist."
            )
        return _WikiStore.open(path, **kwargs)  # type: ignore[arg-type]

    # -- discovery -----------------------------------------------------

    def tags(self) -> list[TagInfo]:
        """The declared vocabulary, with the wikis each tag reaches.

        This is what a provisioning flow reads to populate its own
        user-to-tag table.
        """
        return list(
            self._catalogue_tags(set(self.registry.wikis), include_unused=True)
        )

    def catalogue(self) -> Catalogue:
        """Every wiki and every tag — the provisioning and admin view."""
        return self._catalogue(set(self.registry.wikis), include_unused=True)

    def catalogue_for(self, audience: Iterable[str]) -> Catalogue:
        """Only what this audience reaches.

        Safe behind an end-user "which knowledge bases can I search?"
        picker, where :meth:`catalogue` would not be: a wiki's name and
        title are metadata, and a name can be the sensitive part.
        """
        return self._catalogue(
            set(self.wikis_for(audience)), include_unused=False
        )

    def reconcile(self, assigned: Iterable[str]) -> Reconciliation:
        """Compare the registry against the tags a host has actually assigned.

        ``unknown`` are assigned tags no wiki declares — a stale grant or
        a typo, whose symptom is a user silently getting nothing extra.
        ``unreachable`` are wikis no assigned tag opens — content nobody
        can see, which is how a compartment quietly dies.

        Both failures are silent by nature, which is the reason to have a
        call that goes looking for them.
        """
        held = frozenset(assigned)
        declared = {t for e in self.registry.wikis.values() for t in e.audience}
        declared |= set(self.registry.tags)
        reachable = set(self.wikis_for(held))
        return Reconciliation(
            unknown=tuple(sorted(held - declared)),
            unreachable=tuple(
                n for n in self.registry.wikis if n not in reachable
            ),
        )

    # -- internals -----------------------------------------------------

    def _catalogue_tags(
        self, names: set[str], *, include_unused: bool
    ) -> list[TagInfo]:
        used: dict[str, list[str]] = {}
        for name in self.registry.wikis:
            if name not in names:
                continue
            for tag in sorted(self.registry.wikis[name].audience):
                used.setdefault(tag, []).append(name)
        if include_unused:
            # Declared-but-unused tags belong in the *admin* vocabulary:
            # they are what a host provisions against, and one reaching
            # nothing is a finding for `reconcile`, not a reason to hide
            # it. They must not appear in an audience-filtered catalogue,
            # where a tag name discloses as much as a wiki name — the
            # existence of `project-atlas-acquisition` is the secret.
            for tag in self.registry.tags:
                used.setdefault(tag, [])
        return [
            TagInfo(
                name=tag,
                description=self.registry.tags.get(tag, ""),
                wikis=tuple(used[tag]),
            )
            for tag in sorted(used)
        ]

    def _catalogue(self, names: set[str], *, include_unused: bool) -> Catalogue:
        wikis = tuple(
            WikiInfo(
                name=name,
                title=entry.title,
                audience=tuple(sorted(entry.audience)),
                pages=_count_pages(self.registry.path_of(name)),
            )
            for name, entry in self.registry.wikis.items()
            if name in names
        )
        return Catalogue(
            version=CATALOGUE_VERSION,
            tags=tuple(self._catalogue_tags(names, include_unused=include_unused)),
            wikis=wikis,
        )


# ---------------------------------------------------------------------------
# Registry lint
#
# Both failures these look for are silent by construction. A wiki nobody can
# reach and a tag nobody can be granted produce no error anywhere — the
# content is simply gone, and the first sign is somebody asking why the
# assistant has never heard of the HR handbook.
# ---------------------------------------------------------------------------


def lint_registry(root: Path) -> list[tuple[str, str, str]]:
    """Check a repository's ``wikis.yaml``.

    Returns ``(severity, kind, message)`` triples — plain tuples rather
    than :class:`outmem.lint.LintFinding`, which is anchored to a path
    inside one wiki and has nowhere to put a repository-level problem.
    """
    registry = load_registry(root)
    if registry is None:
        return [
            (
                "error",
                "registry-missing",
                f"{root} has no {REGISTRY_FILENAME}.",
            )
        ]
    out: list[tuple[str, str, str]] = []
    declared = set(registry.tags)
    used: set[str] = set()

    for name, entry in registry.wikis.items():
        used |= entry.audience
        path = registry.path_of(name)
        if not path.is_dir():
            out.append(
                (
                    "error",
                    "registry-missing-wiki",
                    f"wiki {name!r} is listed but {entry.path} does not exist.",
                )
            )
        if not entry.audience:
            out.append(
                (
                    "warning",
                    "registry-unreachable-wiki",
                    f"wiki {name!r} declares no audience — nobody can open it.",
                )
            )
        for tag in sorted(entry.audience - declared):
            out.append(
                (
                    "error",
                    "registry-undeclared-tag",
                    f"wiki {name!r} lists audience tag {tag!r}, which no "
                    "`tags:` entry declares — nobody can be granted a tag "
                    "nobody knows exists, so the wiki is unreachable.",
                )
            )

    for tag in sorted(declared - used):
        out.append(
            (
                "warning",
                "registry-unused-tag",
                f"tag {tag!r} is declared but no wiki lists it — anyone "
                "granted it gains nothing.",
            )
        )

    for path in sorted(_wiki_shaped_dirs(root)):
        rel = path.relative_to(root).as_posix()
        if registry.name_at(path) is None:
            out.append(
                (
                    "warning",
                    "registry-unlisted-wiki",
                    f"{rel} looks like a wiki but no {REGISTRY_FILENAME} "
                    "entry names it — it is unreachable, and commits made "
                    "in it would start their own repository.",
                )
            )
    return out


def _wiki_shaped_dirs(root: Path) -> list[Path]:
    """Directories under ``root`` that look like a wiki root.

    "Looks like" is `config.yaml` beside a `wiki/pages/` directory —
    what `outmem init` produces. Only one level under `wikis/` and one
    under the root itself are searched; a deep walk of a repository with
    thousands of pages costs more than this check is worth.
    """
    candidates: list[Path] = []
    for parent in (root, root / "wikis"):
        if not parent.is_dir():
            continue
        for child in parent.iterdir():
            if not child.is_dir() or child.name.startswith("."):
                continue
            if (child / "config.yaml").is_file() and (
                child / "wiki" / "pages"
            ).is_dir():
                candidates.append(child)
    return candidates
