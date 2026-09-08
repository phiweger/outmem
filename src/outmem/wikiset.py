"""Reading across several wikis at once.

A session is usually entitled to more than one wiki — the open core plus
whatever compartments its audience reaches — and the model should see one
knowledge base, not three. :class:`WikiSet` fans a read across an ordered
list of stores and merges the answers.

The important property is what this is *not*. Nothing here decides what
to hide. Every store in the set is one the session may read in full,
chosen once by :meth:`outmem.repo.Repo.wikis_for` before any of these
objects existed. So a bug in the merging below costs relevance — a
result ranked oddly, a duplicate, a slug resolved to the wrong wiki —
and never confidentiality. That is the whole reason separation is
cheaper to get right than filtering: there is no code path here whose
failure mode is disclosure.

Names are qualified ``wiki/slug``. A slug never contains ``/`` — the
hierarchy separator is ``:`` (see :mod:`outmem.slug`) — so the qualifier
is unambiguous, and a bare slug still works, resolving in the set's
declared order.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from outmem.exceptions import OutmemError
from outmem.search import DEFAULT_RESULT_BYTES

if TYPE_CHECKING:
    from outmem.search import SearchHit
    from outmem.semantic.store import Match
    from outmem.store import WikiPage, WikiStore

QUALIFIER = "/"


@dataclass(frozen=True)
class QualifiedPage:
    """A page, and which wiki it came from."""

    wiki: str
    slug: str
    page: WikiPage

    @property
    def qualified_slug(self) -> str:
        return f"{self.wiki}{QUALIFIER}{self.slug}"


@dataclass(frozen=True)
class QualifiedHit:
    """A search hit, and which wiki it came from."""

    wiki: str
    hit: SearchHit

    @property
    def path(self) -> str:
        return f"{self.wiki}{QUALIFIER}{self.hit.path}"


@dataclass(frozen=True)
class QualifiedMatch:
    """A semantic match, and which wiki it came from."""

    wiki: str
    match: Match

    @property
    def rel_path(self) -> str:
        return f"{self.wiki}{QUALIFIER}{self.match.rel_path}"


@dataclass(frozen=True)
class FederatedSearch:
    """Merged search results, and which wikis had to clip theirs.

    `truncated` is a tuple of wiki names rather than a bool because the
    answer "some of this is missing" is only actionable if you know
    *where* from — and a single wiki hitting its output cap must not make
    the whole result look clipped.
    """

    hits: tuple[QualifiedHit, ...]
    truncated: tuple[str, ...] = ()


@dataclass(frozen=True)
class Resolution:
    """Where a bare slug landed, and what it shadowed getting there."""

    wiki: str
    slug: str
    shadowed: tuple[str, ...]


def split_qualified(name: str) -> tuple[str | None, str]:
    """``"legal/nda"`` → ``("legal", "nda")``; ``"nda"`` → ``(None, "nda")``.

    Only the first ``/`` is a qualifier. Everything after it is the slug,
    which cannot contain one, so a name with two is malformed rather than
    ambiguous — and comes back with the remainder intact so the caller
    reports the name the user actually typed.
    """
    wiki, sep, slug = name.partition(QUALIFIER)
    return (wiki, slug) if sep else (None, name)


class WikiSet:
    """An ordered, already-permitted collection of wikis, read as one.

    Order is the registry's, and it is the resolution order for a bare
    slug: the first wiki holding it wins, and the rest are reported as
    shadowed rather than silently dropped.
    """

    def __init__(self, stores: Sequence[WikiStore]) -> None:
        if not stores:
            raise ValueError("WikiSet needs at least one wiki.")
        named: dict[str, WikiStore] = {}
        for store in stores:
            # A standalone wiki has no registry name. Falling back to the
            # directory keeps the qualifier meaningful for the one-wiki
            # case rather than emitting `None/slug`.
            name = store.wiki_name or store.root.name
            if name in named:
                raise ValueError(f"duplicate wiki name in set: {name!r}")
            named[name] = store
        self._stores = named

    # -- shape ---------------------------------------------------------

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._stores)

    def __len__(self) -> int:
        return len(self._stores)

    def __iter__(self) -> Iterator[tuple[str, WikiStore]]:
        return iter(self._stores.items())

    def store(self, name: str) -> WikiStore:
        try:
            return self._stores[name]
        except KeyError:
            raise OutmemError(f"no such wiki: {name!r}") from None

    # -- resolution ----------------------------------------------------

    def resolve(self, name: str) -> Resolution:
        """Locate ``name``, qualified or bare.

        A qualified name goes to that wiki and only that wiki: if it is
        not there, the answer is "not found", not "let me look
        elsewhere". Guessing across a boundary the caller was explicit
        about is how a reader ends up reading the wrong wiki's page under
        the right wiki's name.
        """
        wiki, slug = split_qualified(name)
        if wiki is not None:
            store = self.store(wiki)
            if not store.exists(slug):
                raise OutmemError(f"No such wiki page: {name}")
            return Resolution(wiki=wiki, slug=store.resolve_slug(slug), shadowed=())

        holders = [n for n, s in self._stores.items() if s.exists(slug)]
        if not holders:
            raise OutmemError(f"No such wiki page: {name}")
        winner = holders[0]
        return Resolution(
            wiki=winner,
            slug=self._stores[winner].resolve_slug(slug),
            shadowed=tuple(f"{n}{QUALIFIER}{slug}" for n in holders[1:]),
        )

    def read(self, name: str) -> QualifiedPage:
        """Read one page by qualified or bare name."""
        found = self.resolve(name)
        store = self._stores[found.wiki]
        return QualifiedPage(
            wiki=found.wiki, slug=found.slug, page=store.read(found.slug)
        )

    def exists(self, name: str) -> bool:
        try:
            self.resolve(name)
        except OutmemError:
            return False
        return True

    # -- enumeration ---------------------------------------------------

    def list_slugs(self) -> list[str]:
        """Every page in the set, qualified, in declared wiki order."""
        return [
            f"{name}{QUALIFIER}{slug}"
            for name, store in self._stores.items()
            for slug in store.list_slugs()
        ]

    def backlinks(self, name: str) -> tuple[str, ...]:
        """Inbound links, qualified — and wiki-local by construction.

        Wikilinks do not cross a wiki boundary, so a page's referrers all
        live in its own wiki. Qualifying them anyway keeps every name the
        caller sees in one namespace.
        """
        found = self.resolve(name)
        store = self._stores[found.wiki]
        return tuple(
            f"{found.wiki}{QUALIFIER}{ref}" for ref in store.backlinks(found.slug)
        )

    # -- retrieval -----------------------------------------------------

    def search(
        self,
        pattern: str,
        *,
        scope: str = "wiki",
        case_insensitive: bool = False,
        fixed_strings: bool = False,
        context: int = 0,
        max_bytes: int = DEFAULT_RESULT_BYTES,
        max_hits: int | None = None,
    ) -> FederatedSearch:
        """Fan a literal/regex search across the set.

        Results come back in wiki order with their hits in each wiki's own
        order. `rg` scores nothing, so there is no cross-wiki ranking
        question here — that arrives with the semantic leg below.

        The per-wiki output cap applies per wiki, so any of them can clip
        independently. Which ones did is reported rather than folded into
        a single flag: a partial result that looks complete is worse than
        no result, and the caller needs to know which wiki to narrow.

        The keyword arguments mirror :meth:`outmem.store.WikiStore.search`
        and are spelled out rather than forwarded as ``**kwargs`` — this
        is a public surface, and a typo'd keyword should be a type error
        here, not a ``TypeError`` from inside a loop over three wikis.
        """
        hits: list[QualifiedHit] = []
        clipped: list[str] = []
        for name, store in self._stores.items():
            result = store.search(
                pattern,
                scope=scope,
                case_insensitive=case_insensitive,
                fixed_strings=fixed_strings,
                context=context,
                max_bytes=max_bytes,
                max_hits=max_hits,
            )
            hits.extend(QualifiedHit(wiki=name, hit=h) for h in result.hits)
            if result.truncated:
                clipped.append(name)
        return FederatedSearch(hits=tuple(hits), truncated=tuple(clipped))

    def semantic_find_similar(
        self,
        text: str,
        *,
        top_k: int | None = None,
        threshold: float | None = None,
        exclude_slug: str | None = None,
    ) -> list[QualifiedMatch]:
        """Nearest chunks across the set, best first.

        Each wiki is queried for its own candidates and the union is
        sorted by similarity. Cosine similarity against the same query
        vector is comparable between wikis as long as they embed with the
        same model — which is why `top_k` is applied to the *merged*
        list: taking the top k per wiki first would let a wiki with
        nothing relevant crowd out one that had everything.
        """
        merged: list[QualifiedMatch] = []
        for name, store in self._stores.items():
            for match in store.semantic_find_similar(
                text,
                top_k=top_k,
                threshold=threshold,
                exclude_slug=exclude_slug,
            ):
                merged.append(QualifiedMatch(wiki=name, match=match))
        merged.sort(key=lambda m: m.match.similarity, reverse=True)
        return merged[:top_k] if top_k is not None else merged

    def semantic_available(self) -> bool:
        """True if any wiki in the set has a semantic index."""
        return any(s.semantic_available() for s in self._stores.values())

    # -- lifecycle -----------------------------------------------------

    def close(self) -> None:
        """Release every store's SQLite handles.

        Each store opens the vector store and both source registries
        lazily and holds them. Dropping a set does *not* release them
        promptly: the objects sit in reference cycles, so they survive
        until the cycle collector runs. Measured over twenty
        open-and-drop cycles on three wikis, six connections were open at
        the end and `gc.collect()` took it back to zero — bounded, but by
        the collector's schedule rather than by anything the caller
        controls.

        Closing makes it deterministic, which is what a server wants.
        Build a set once per audience and keep it, or use it as a context
        manager.
        """
        for store in self._stores.values():
            store.close()

    def __enter__(self) -> WikiSet:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
