"""A PydanticAI read palette over several wikis at once.

The model should see one knowledge base, not three. These tools fan each
read across the :class:`~outmem.wikiset.WikiSet` and hand back names
qualified ``wiki/slug``, so a session that is entitled to the open core
plus two compartments searches all of them in one call.

Every wiki in the set is one the session may read in full — the access
decision was made before the set existed, by choosing which stores went
in. So nothing here filters, and nothing here can leak: the failure mode
of a bug below is a badly ordered result, not a disclosed one.

Reads are federated; writes are not, and cannot sensibly be. "Append this
to the wiki" has no answer when there are three of them, so a session
that writes takes a single :class:`~outmem.store.WikiStore` and the
ordinary palette in :mod:`outmem.adapters.pydantic_ai`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from outmem.exceptions import FrontmatterError, OutmemError, SlugError
from outmem.wikiset import split_qualified

if TYPE_CHECKING:
    from collections.abc import Callable

    from outmem.wikiset import WikiSet

# How much of a page to show per hit, so the model can tell whether it is
# worth a `read_page`. Matches the single-wiki tool: the two produce the
# same kind of result and should look the same doing it.
_PREVIEW_CHARS = 200


def _wiki_note(wikis: WikiSet) -> str:
    """One line naming the wikis in play, for the tool docstrings.

    The model has to know the qualifier exists, or it will pass a bare
    slug into a set where two wikis hold that name and never understand
    why it got the other one.
    """
    names = ", ".join(wikis.names)
    return (
        f"This session reads {len(wikis)} wiki(s): {names}. Page names are "
        f"qualified `wiki/slug` (for example `{wikis.names[0]}/some-page`). "
        "A bare slug resolves in the order above."
    )


def wikiset_read_tools(wikis: WikiSet) -> list[Callable[..., Any]]:
    """The federated read palette for ``wikis``.

    Mirrors the single-wiki read tools in
    :func:`outmem.adapters.pydantic_ai.wiki_read_tools`, with every name
    qualified and every search fanned across the set.
    """
    note = _wiki_note(wikis)

    def list_pages() -> str:
        """Return every page in every wiki this session can read.

        One qualified name per line, grouped by wiki in resolution
        order. Cheap — one directory listing per wiki. Use it to map the
        territory before a broad search, or to confirm a name exists
        before reading it.

        Example:
            list_pages()
        """
        slugs = wikis.list_slugs()
        return "\n".join(slugs) if slugs else "(no pages)"

    def read_page(name: str) -> str:
        """Read one page, by qualified name or bare slug.

        A qualified name (`legal/nda`) goes to that wiki and only that
        wiki. A bare slug (`nda`) resolves in wiki order; if more than
        one wiki holds it, the others are named at the end so you can ask
        for a specific one.

        Example:
            read_page(name="legal/nda")
            read_page(name="pricing-formula")

        Args:
            name: `wiki/slug`, or a bare slug from `list_pages` or
                `search_wiki`.
        """
        try:
            found = wikis.resolve(name)
        except SlugError as exc:
            return f"(invalid name {name!r}: {exc})"
        except OutmemError:
            return (
                f"(no such page: {name!r} — try `list_pages` to see what "
                "exists)"
            )
        store = wikis.store(found.wiki)
        try:
            page = store.read(found.slug)
        except FrontmatterError as exc:
            return f"(page {name!r} has malformed frontmatter: {exc})"
        body = page.path.read_text(encoding="utf-8")
        header = f"# {found.wiki}/{found.slug}\n\n"
        if found.shadowed:
            others = ", ".join(found.shadowed)
            header += f"(also present as: {others})\n\n"
        return header + body

    def grep_wiki(pattern: str, case_insensitive: bool = False) -> str:
        """Literal / regex search across every wiki in this session.

        Exact matching, not semantic — use it for identifiers, error
        strings, names, section headings, anything you can spell. For
        "what do we know about X", use `search_wiki`.

        Example:
            grep_wiki(pattern="cost times")
            grep_wiki(pattern="NDA", case_insensitive=True)

        Args:
            pattern: A regular expression, or a literal string.
            case_insensitive: Match without regard to case.
        """
        try:
            result = wikis.search(pattern, case_insensitive=case_insensitive)
        except OutmemError as exc:
            return f"(search failed: {exc})"
        lines = [
            f"{h.path}:{h.hit.line_number}: {h.hit.text.rstrip()}"
            for h in result.hits
            if h.hit.is_match
        ]
        if not lines:
            return f"(no matches for {pattern!r} in: {', '.join(wikis.names)})"
        if result.truncated:
            # Never let a clipped result read as a complete one. The model
            # will otherwise conclude the wiki holds nothing more and stop
            # looking.
            where = ", ".join(result.truncated)
            lines.append(
                f"(results from {where} were truncated at the output cap — "
                "narrow the pattern)"
            )
        return "\n".join(lines)

    def search_wiki(question: str, k: int = 5) -> str:
        """Find the pages most likely to answer a question, across every wiki.

        The primary search tool: ask a natural-language question, get back
        the most relevant whole pages, each qualified with the wiki it
        came from. Every wiki runs the pipeline *it* configures
        (`retrieval.strategy`), and the rankings are fused, so a
        compartment holding the answer outranks the open core rather than
        queueing behind it. Call `read_page` on the interesting ones.

        Use this for question-shaped queries. For the exact line a literal
        string appears on, use `grep_wiki`.

        Example:
            search_wiki(question="how is the list price calculated?")

        Args:
            question: A natural-language question.
            k: How many pages to return.
        """
        try:
            found = wikis.search_pages(question, k=k)
        except OutmemError as exc:
            return f"(search_wiki failed: {exc})"
        if not found.pages:
            # Never report an empty corpus when retrieval is what failed.
            # "We have nothing on that" is a load-bearing answer here, and
            # a caller who cannot tell it from an outage will state it
            # with confidence.
            where = ", ".join(found.searched) or "(none)"
            detail = f" (diagnostics: {'; '.join(found.notes)})" if found.notes else ""
            return (
                f"(no pages matched {question!r} in: {where} — try rephrasing, "
                f"or `grep_wiki` for literal keyword matches){detail}"
            )
        lines: list[str] = []
        for qualified in found.pages:
            wiki, slug = split_qualified(qualified)
            try:
                store = wikis.store(wiki) if wiki else None
                body = (
                    store.read(slug).body.replace("\n", " ").strip() if store else ""
                )
            except OutmemError:
                # A page can vanish between ranking and rendering; a missing
                # preview is better than losing the citation.
                body = ""
            preview = body[:_PREVIEW_CHARS] + ("…" if len(body) > _PREVIEW_CHARS else "")
            lines.append(f"  - [[{qualified}]] {preview}")
        if found.notes:
            lines.append(f"(diagnostics: {'; '.join(found.notes)})")
        return "\n".join(lines)

    def find_backlinks(name: str) -> str:
        """Which pages link to this one.

        Links never cross a wiki boundary, so every referrer is in the
        same wiki as the page itself.

        Example:
            find_backlinks(name="legal/nda")

        Args:
            name: `wiki/slug`, or a bare slug.
        """
        try:
            refs = wikis.backlinks(name)
        except OutmemError:
            return f"(no such page: {name!r})"
        return "\n".join(refs) if refs else f"(nothing links to {name})"

    for tool in (list_pages, read_page, grep_wiki, search_wiki, find_backlinks):
        tool.__doc__ = f"{tool.__doc__}\n\n        {note}"
    return [list_pages, read_page, grep_wiki, search_wiki, find_backlinks]
