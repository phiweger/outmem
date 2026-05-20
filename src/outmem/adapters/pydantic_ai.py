"""PydanticAI adapter — turn a :class:`WikiStore` into a tool palette.

The adapter returns plain Python callables with docstrings tuned for
PydanticAI's schema extraction. Consumers attach them to their own
:class:`pydantic_ai.Agent` via the ``tools=`` parameter::

    from pydantic_ai import Agent
    from outmem import WikiStore
    from outmem.adapters.pydantic_ai import wiki_tools

    store = WikiStore.open("/srv/agent")
    agent = Agent("anthropic:claude-sonnet-4-6", tools=wiki_tools(store))

No hard dependency on ``pydantic_ai`` — the functions are vanilla
Python and PydanticAI introspects them at attach time. Install
``outmem[pydantic-ai]`` to pull the framework into the same environment.

Tool docstrings follow AGENTS.md §"Tool docstrings — the JSON schema
the model can't escape": multi-arg tools lead with ``REQUIRES ALL
ARGUMENTS``, every tool shows one concrete example call, and ``Args:``
sections give concrete valid values.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from outmem.exceptions import (
    FrontmatterError,
    OutmemError,
    SlugError,
    WritebackError,
)
from outmem.skills import bundled_registry
from outmem.store import WikiStore

# Public type for the returned function list — kept as ``Any`` so we
# don't pretend a tight signature we can't enforce across nine arities.
WikiTool = Callable[..., Any]

# Logger every tool call writes a one-line trace to. Disabled by default
# (Python's logging emits nothing without a handler); the CLI's
# ``outmem ask`` attaches a stderr handler so users can see what the
# agent is doing. Consumer apps can do the same — see ``outmem.cli``.
_tool_log = logging.getLogger("outmem.agent.tool")


def _summarise(value: Any, *, limit: int = 60) -> str:
    """Compact repr for tool-call logging — long strings become "(N chars)"."""
    if isinstance(value, str):
        if len(value) > limit:
            return f"({len(value)} chars)"
        return repr(value)
    if isinstance(value, list | tuple):
        return repr(list(value)[:5]) + ("…" if len(value) > 5 else "")
    return repr(value)


def _log_call(name: str, **kwargs: Any) -> None:
    formatted = " ".join(f"{k}={_summarise(v)}" for k, v in kwargs.items())
    # ``tool_call`` carries the raw kwargs so logging handlers can do
    # structured analysis (e.g. eval recorders) without having to parse
    # the formatted string. Stays on the LogRecord as ``record.tool_call``.
    _tool_log.info("%s %s", name, formatted, extra={"tool_call": (name, dict(kwargs))})


def _log_error(name: str, exc: Exception) -> None:
    _tool_log.info(
        "%s → ERROR: %s",
        name,
        exc,
        extra={"tool_error": (name, type(exc).__name__, str(exc))},
    )


def _read_tools(store: WikiStore) -> list[WikiTool]:
    """Internal — closures for the read-only tool palette.

    Used by both :func:`wiki_tools` (which appends the write tools) and
    :func:`wiki_read_tools` (which returns this list verbatim). Lives
    behind a single factory so the "what counts as read-only" definition
    is the function body, not a string allowlist that could drift.
    """

    def search_wiki(
        pattern: str,
        scope: str = "wiki",
        case_insensitive: bool = False,
    ) -> str:
        """Search the wiki / raw / log directories with ripgrep.

        Use this as your first retrieval move. ``scope="wiki"`` (the
        default, Tier 1) searches compiled pages; ``scope="raw"`` (Tier 2)
        falls through to source material when the wiki did not contain
        the answer. Returns ``path:line:text`` rows, one per match.

        Example:
            search_wiki(pattern="cost-plus", scope="wiki")

        Args:
            pattern: Regex pattern, or a literal string when ``case_insensitive`` does the job.
            scope: One of ``"wiki"``, ``"raw"``, ``"log"``, ``"all"``. Default ``"wiki"``.
            case_insensitive: ``True`` to ignore case (``rg -i``).
        """
        _log_call("search_wiki", pattern=pattern, scope=scope, case_insensitive=case_insensitive)
        try:
            result = store.search(pattern, scope=scope, case_insensitive=case_insensitive)
        except OutmemError as exc:
            _log_error("search_wiki", exc)
            return f"(search failed: {exc})"
        if not result.hits:
            return "(no matches)"
        lines = [f"{hit.path}:{hit.line_number}:{hit.text}" for hit in result.hits]
        if result.truncated:
            lines.append("(truncated — narrow the pattern)")
        return "\n".join(lines)

    def read_page(slug: str) -> str:
        """Read a single wiki page by slug. Returns the full file
        (frontmatter + body) as a string.

        Use this after ``search_wiki`` has surfaced a candidate slug,
        or when you already know which page you want. If the slug
        names a raw source file rather than a wiki page, use
        ``search_wiki`` with ``scope="raw"`` instead.

        Example:
            read_page(slug="pricing-formula")

        Args:
            slug: A slug from ``list_pages`` or ``search_wiki`` (lowercase, hyphen-separated).
        """
        _log_call("read_page", slug=slug)
        try:
            page = store.read(slug)
        except SlugError as exc:
            _log_error("read_page", exc)
            return f"(invalid slug {slug!r}: lowercase letters/digits/hyphens only)"
        except FrontmatterError as exc:
            _log_error("read_page", exc)
            return f"(page {slug!r} has malformed frontmatter: {exc})"
        except OutmemError as exc:
            _log_error("read_page", exc)
            return (
                f"(no such wiki page: {slug!r} — try `list_pages` to see "
                "what exists, or `search_wiki` with scope='raw' for source material)"
            )
        return page.path.read_text(encoding="utf-8")

    def list_pages() -> str:
        """Return every wiki page slug, one per line, alphabetically.

        Cheap (one directory listing). Use to map the territory before
        a broad search, or to confirm a slug exists before reading it.

        Example:
            list_pages()
        """
        _log_call("list_pages")
        slugs = store.list_slugs()
        return "\n".join(slugs) if slugs else "(no pages)"

    def find_backlinks(slug: str) -> str:
        """List wiki pages that link to ``slug`` via ``[[wikilink]]``.

        Useful when answering "what depends on X" — backlinks surface
        the reverse direction of the wikilink graph.

        Example:
            find_backlinks(slug="pricing-formula")

        Args:
            slug: The target slug whose referrers you want.
        """
        _log_call("find_backlinks", slug=slug)
        try:
            refs = store.backlinks(slug)
        except SlugError as exc:
            _log_error("find_backlinks", exc)
            return f"(invalid slug {slug!r})"
        return "\n".join(refs) if refs else "(no backlinks)"

    def page_history(slug: str) -> str:
        """Per-page commit log: every commit that touched ``wiki/<slug>.md``.

        Returns ``sha  iso-date  author <email>  subject`` rows,
        newest first. Use to answer "when did X change and who changed it".
        For the actual diff content use ``topic_evolution``.

        Example:
            page_history(slug="pricing-formula")

        Args:
            slug: The page whose history you want.
        """
        _log_call("page_history", slug=slug)
        try:
            history = store.history(slug)
        except SlugError as exc:
            _log_error("page_history", exc)
            return f"(invalid slug {slug!r})"
        if not history:
            return "(no history)"
        return "\n".join(
            f"{c.sha[:10]}  {c.date.isoformat()}  {c.author_name} <{c.author_email}>  {c.subject}"
            for c in history
        )

    def topic_evolution(slugs: list[str], include_log: bool = True) -> str:
        """Return the raw ``git log -p`` diff stream across the given slugs.

        This is the EXPANSION-branch helper: read the diff sequence as-is
        to understand how thinking on the topic has shifted over time,
        rather than just retrieving the current state. Pass multiple
        related slugs to interleave their evolution chronologically.

        Example:
            topic_evolution(slugs=["pricing-formula", "discounts"], include_log=True)

        Args:
            slugs: One or more wiki slugs to walk.
            include_log: ``True`` to include ``log/`` entries in the timeline (default).
        """
        _log_call("topic_evolution", slugs=slugs, include_log=include_log)
        if not slugs:
            return "(topic_evolution requires at least one slug)"
        try:
            return store.evolution(slugs, include_log=include_log)
        except SlugError as exc:
            _log_error("topic_evolution", exc)
            return f"(invalid slug in {slugs}: {exc})"
        except OutmemError as exc:
            _log_error("topic_evolution", exc)
            return f"(evolution failed: {exc})"

    def list_sources() -> str:
        """List every registered source under ``wiki/sources/``.

        Returns ``relative/path  sha:...  N ingestion(s)`` rows, one
        per registered file. Use when answering "what raw material do
        we have?" or before deciding whether to write a new page from
        a source you already cited before.

        Example:
            list_sources()
        """
        _log_call("list_sources")
        entries = store.list_sources()
        if not entries:
            return "(no sources registered)"
        lines: list[str] = []
        for entry in entries:
            ingestions_summary = ""
            if entry.ingestions:
                prompts = [f'"{i.prompt}"' if i.prompt else "(no prompt)" for i in entry.ingestions]
                ingestions_summary = f"  {len(entry.ingestions)} ingestion(s): {'; '.join(prompts)}"
            lines.append(
                f"{entry.rel_path}  sha:{entry.sha256[:12]}…  "
                f"{entry.size_bytes}B{ingestions_summary}"
            )
        return "\n".join(lines)

    def read_source(rel_path: str) -> str:
        """Read a registered source file as plain text.

        Use during ingestion to actually look at the material before
        extracting facts into wiki pages. The output is capped at the
        configured ``sources.max_chars`` (default 200k chars) — if
        the file is larger, the tail is truncated with a marker.

        Example:
            read_source(rel_path="veterinary/drugs.md")

        Args:
            rel_path: Path relative to ``wiki/sources/`` (matches the
                ``rel_path`` field from ``list_sources``).
        """
        _log_call("read_source", rel_path=rel_path)
        try:
            return store.read_source(rel_path)
        except OutmemError as exc:
            _log_error("read_source", exc)
            return f"(read_source failed: {exc})"

    def find_similar(
        text: str,
        top_k: int = 5,
        exclude_slug: str | None = None,
    ) -> str:
        """Find wiki / source chunks that are semantically similar to ``text``.

        Use BEFORE writing a new page to spot near-duplicates (the wiki
        loses value when the same fact is compiled into two pages with
        different framings). Also useful when answering open-ended
        questions where keyword search misses paraphrased material.

        Requires ``semantic.enabled: true`` in ``config.yaml`` — returns
        an explanatory string if disabled. Similarity is cosine
        (``1.0`` is identical, ``0.0`` is orthogonal); the per-call
        threshold comes from config.

        Example:
            find_similar(text="cost-plus pricing formula", top_k=5)
            find_similar(text="...new page body...", exclude_slug="my-new-page")

        Args:
            text: The query — often the body of the page you're about to write.
            top_k: How many matches to return (default 5).
            exclude_slug: A slug to exclude from results (use when comparing
                a page against the rest of the wiki).
        """
        _log_call(
            "find_similar",
            text=text,
            top_k=top_k,
            exclude_slug=exclude_slug,
        )
        if not store.semantic_enabled():
            return (
                "(find_similar unavailable: set `semantic.enabled: true` in "
                "config.yaml and install `outmem[semantic]`)"
            )
        try:
            matches = store.semantic_find_similar(
                text,
                top_k=top_k,
                exclude_slug=exclude_slug,
            )
        except OutmemError as exc:
            _log_error("find_similar", exc)
            return f"(find_similar failed: {exc})"
        except Exception as exc:
            _log_error("find_similar", exc)
            return f"(find_similar failed: {exc})"
        if not matches:
            return "(no semantically similar chunks above threshold)"
        lines: list[str] = []
        for match in matches:
            preview = match.content.replace("\n", " ").strip()
            if len(preview) > 200:
                preview = preview[:197] + "…"
            lines.append(
                f"{match.rel_path}#chunk{match.chunk_index}  "
                f"sim={match.similarity:.3f}  {preview}"
            )
        return "\n".join(lines)

    tools: list[WikiTool] = [
        search_wiki,
        read_page,
        list_pages,
        find_backlinks,
        page_history,
        topic_evolution,
        list_sources,
        read_source,
    ]
    # find_similar is only exposed when the semantic index is enabled,
    # so the model isn't tempted to call a tool that always returns
    # "unavailable".
    if store.semantic_enabled():
        tools.append(find_similar)
    return tools


def _write_tools(store: WikiStore) -> list[WikiTool]:
    """Internal — closures for the commit-producing tool palette.

    Every function here funnels through :meth:`WikiStore._commit_paths`,
    so a read-only store will surface the refusal as an
    :class:`OutmemError` propagated back through the tool's return path.
    """

    def write_page(
        slug: str,
        title: str,
        body: str,
        provenance: list[str | dict[str, Any]] | None = None,
        tags: list[str] | None = None,
    ) -> str:
        """Create a new wiki page.

        REQUIRES ALL THREE OF: slug, title, body. This is not a multi-step
        "stage then commit" workflow — every call writes a complete page
        and produces exactly one commit (``compact: <slug>``). Returns
        the new HEAD SHA.

        Fails if a page with ``slug`` already exists — use ``extend_page``
        to edit existing pages.

        Example with plain-string provenance:
            write_page(
                slug="pricing-formula",
                title="Pricing formula",
                body="The pricing formula is cost-plus 35%.\\n",
                provenance=["raw/pricing-deck-2026-Q1.md"],
                tags=["pricing", "contracts"],
            )

        Example with structured provenance (for ingested sources):
            write_page(
                slug="amikacin-iv-dosing",
                title="Amikacin IV — Dosing",
                body="...",
                provenance=[{
                    "path": "sources/9b3d0d4e1a35/document.md",
                    "sha256": "9b3d0d4e1a35...",
                    "label": "Fachinformation Amikacin Eberth 250 mg/ml",
                }],
                tags=["amikacin", "dosing"],
            )

        Args:
            slug: Lowercase, hyphen-separated identifier. Becomes ``wiki/<slug>.md``.
            title: Human-readable page title for the frontmatter.
            body: The complete markdown body (no frontmatter — that is generated).
            provenance: Optional list of source pointers. Each entry is
                either a plain path string (e.g. ``"raw/...md"``) or a
                dict carrying additional metadata (``path``, ``sha256``,
                ``label``, etc.). Both shapes round-trip through the
                frontmatter unchanged.
            tags: Optional tag list for the frontmatter.
        """
        _log_call(
            "write_page",
            slug=slug,
            title=title,
            body=body,
            provenance=provenance,
            tags=tags,
        )
        try:
            return store.write_page(
                slug,
                title=title,
                body=body,
                provenance=list(provenance) if provenance else None,
                tags=list(tags) if tags else None,
            )
        except WritebackError:
            raise  # propagate; the service surfaces this to the caller
        except SlugError as exc:
            _log_error("write_page", exc)
            return f"(invalid slug {slug!r}: lowercase letters/digits/hyphens only)"
        except OutmemError as exc:
            _log_error("write_page", exc)
            return f"(write_page failed: {exc})"

    def extend_page(slug: str, body: str) -> str:
        """Replace the body of an existing wiki page.

        REQUIRES BOTH ARGUMENTS in a single call. Frontmatter (title,
        slug, provenance, tags, created) is preserved; ``updated`` is
        bumped to now. Produces exactly one commit (``extend: <slug>``)
        and returns the new HEAD SHA.

        Fails if the page does not exist — use ``write_page`` for new
        pages.

        Example:
            extend_page(
                slug="pricing-formula",
                body="The pricing formula is now cost-plus 40%, revised Q2.\\n",
            )

        Args:
            slug: Existing page slug.
            body: The complete replacement body.
        """
        _log_call("extend_page", slug=slug, body=body)
        try:
            return store.extend_page(slug, body=body)
        except WritebackError:
            raise
        except SlugError as exc:
            _log_error("extend_page", exc)
            return f"(invalid slug {slug!r})"
        except OutmemError as exc:
            _log_error("extend_page", exc)
            return f"(extend_page failed: {exc} — use `write_page` for new pages)"

    def append_log(topic: str, content: str) -> str:
        """Append an entry to ``log/<today>.md`` and commit.

        REQUIRES BOTH ARGUMENTS in a single call. Use this when a turn
        produced an observation, contradiction, or "no new compaction
        needed" note — anything that doesn't yet rise to a wiki page.
        The commit message is ``log: <topic>``; ``content`` is appended
        verbatim under today's date heading. Returns the new HEAD SHA.

        Mandatory writeback (spec v0.5 §9) means EVERY turn must end
        with at least one commit — when no wiki write was warranted,
        an ``append_log`` is the canonical "I did the work and here is
        the trail" outcome.

        Example:
            append_log(
                topic="pricing-inconsistency",
                content="- noticed acme-msa cites cost-plus 30%, pricing-formula says 35%.\\n",
            )

        Args:
            topic: Short topic for the commit subject (``log: <topic>``).
            content: The markdown content to append. End with a newline.
        """
        _log_call("append_log", topic=topic, content=content)
        try:
            return store.append_log(topic=topic, content=content)
        except WritebackError:
            raise
        except OutmemError as exc:
            _log_error("append_log", exc)
            return f"(append_log failed: {exc})"

    def record_ingestion(
        rel_path: str,
        prompt: str,
        pages_touched: list[str],
    ) -> str:
        """Record an ingestion against a registered source.

        REQUIRES ALL THREE OF: rel_path, prompt, pages_touched. Call
        this AFTER you've written the wiki pages — it appends an
        entry to ``wiki/sources/.sources.db`` linking the source to
        the pages produced, and commits the registry update as
        ``ingest: <rel_path>``. The runtime may also auto-record;
        calling this explicitly is the safer path during agent-driven
        ingestion.

        Example:
            record_ingestion(
                rel_path="veterinary/drugs.md",
                prompt="extract drug dosages for cats",
                pages_touched=["cat-drug-dosages"],
            )

        Args:
            rel_path: Source path (must already be registered).
            prompt: The focus directive you ingested under (or "" if none).
            pages_touched: Slugs you wrote or extended in this turn.
        """
        _log_call(
            "record_ingestion",
            rel_path=rel_path,
            prompt=prompt,
            pages_touched=pages_touched,
        )
        try:
            store.record_ingestion(
                rel_path,
                prompt=prompt or None,
                pages_touched=pages_touched,
                commit=True,
            )
        except OutmemError as exc:
            _log_error("record_ingestion", exc)
            return f"(record_ingestion failed: {exc})"
        return f"(recorded ingestion against {rel_path})"

    return [write_page, extend_page, append_log, record_ingestion]


def wiki_tools(store: WikiStore) -> list[WikiTool]:
    """Return the v0.1 PydanticAI tool palette bound to ``store``.

    Twelve tools (thirteen when ``semantic.enabled``): retrieval
    (``search_wiki`` / ``read_page`` / ``list_pages``), graph
    traversal (``find_backlinks`` / ``page_history``), the EXPANSION
    helper (``topic_evolution``), source inspection (``list_sources``
    / ``read_source`` / ``find_similar``), and the four writeback
    paths (``write_page`` / ``extend_page`` / ``append_log`` /
    ``record_ingestion``).

    Each call is a closure over ``store`` so consumers don't need to
    plumb a RunContext deps type — just pass ``tools=wiki_tools(store)``
    and the model gets everything bound.

    For consult-only / external agent integrations, use
    :func:`wiki_read_tools` instead — it drops the write tools so the
    model never sees a commit-producing API.
    """
    return _read_tools(store) + _write_tools(store)


def wiki_read_tools(store: WikiStore) -> list[WikiTool]:
    """Return the read-only subset of the v0.1 PydanticAI tool palette.

    Drops every commit-producing tool (``write_page``, ``extend_page``,
    ``append_log``, ``record_ingestion``). The survivors are pure
    retrieval / inspection paths: ``search_wiki``, ``read_page``,
    ``list_pages``, ``find_backlinks``, ``page_history``,
    ``topic_evolution``, ``list_sources``, ``read_source``, and
    ``find_similar`` (when the semantic index is enabled).

    Pair with ``WikiStore.open(path, read_only=True)`` when handing a
    curated wiki to an external agent system as a consult-only tool —
    the store's :meth:`~outmem.store.WikiStore._commit_paths` guard
    rejects any write attempt anyway, but exposing only the read tools
    means the model never even sees the write API and won't try to
    use it.

    Example::

        from pydantic_ai import Agent
        from outmem import WikiStore
        from outmem.adapters.pydantic_ai import wiki_read_tools

        store = WikiStore.open("/srv/curated-wiki", read_only=True)
        agent = Agent(
            "anthropic:claude-sonnet-4-6",
            tools=wiki_read_tools(store),
            system_prompt="You answer from the wiki only. Cite [[slugs]].",
        )
    """
    return _read_tools(store)


# Inner system prompt for the read-only consult subagent. Deliberately
# narrow — describes what this subagent must do (cite by [[slug]],
# signal absence explicitly) without re-stating workflow that the tool
# docstrings already make obvious. The "do not invent" line is the
# load-bearing instruction for the no-answer case tested by the
# subagent E2E eval.
_CONSULT_WIKI_SYSTEM_PROMPT = (
    "You answer questions from a curated knowledge base by searching "
    "and reading its wiki pages. The available tools describe how to "
    "search, read, and traverse the wiki — use them.\n\n"
    "Cite pages by `[[slug]]` when you draw from their content. If "
    "the wiki has no information on the asked topic, say so "
    "explicitly — phrase the answer so the caller can tell the "
    "difference between \"the wiki has nothing on this\" and a "
    "substantive reply. Do not invent answers from your background "
    "knowledge.\n\n"
    "The knowledge base is read-only; you cannot modify it."
)


# Match the full agent runtime — `agent.runtime.DEFAULT_MAX_TOKENS` and
# the Anthropic prompt-caching keys. Multi-page reads + synthesis
# regularly exceed PydanticAI's 4096-token Anthropic default and
# truncate mid-tool-call; the cache keys cut the bill 5-10x on
# repeated calls. Kept inline (not imported) so this module has no
# dependency on the optional `outmem.agent` runtime.
_CONSULT_MODEL_SETTINGS: dict[str, Any] = {
    "max_tokens": 16384,
    "anthropic_cache": True,
    "anthropic_cache_instructions": True,
    "anthropic_cache_tool_definitions": True,
}


def build_consult_wiki(
    wiki_path: str | Path,
    *,
    model: Any = "anthropic:claude-sonnet-4-6",
) -> Callable[[str], str]:
    """One-call factory: a ``consult_wiki(question) -> str`` tool function.

    Opens the wiki at ``wiki_path`` in read-only mode, builds an inner
    :class:`pydantic_ai.Agent` configured with the read-only tool palette
    (:func:`wiki_read_tools`), a tight system prompt that tells the
    inner agent to cite pages by ``[[slug]]`` and explicitly signal
    absence when the wiki has nothing on the topic, and the same
    ``max_tokens`` + Anthropic prompt-caching settings as the full
    ``outmem ask`` runtime.

    Returns a plain ``consult_wiki(question: str) -> str`` callable
    suitable for attaching to *your* outer PydanticAI agent via
    ``tools=[consult_wiki]``. The outer agent never sees outmem's
    internals — it just gets a black-box "ask the team's knowledge
    base" tool.

    The inner agent's answer is returned verbatim. With
    ``read_only=True`` the store refuses every commit-producing path
    (``write_page`` etc.) and skips the layout/cache-creating side
    effects in ``WikiStore.open`` — the wiki's filesystem state is
    left exactly as it was found. If you ALSO want the consult to
    leave a per-question trace in the wiki's ``log/`` (gap signal for
    the curator), open the wiki writable and use
    :func:`outmem.agent.ask_sync` instead — that path enforces
    mandatory writeback.

    The store is held alive by the returned callable's closure for the
    lifetime of the process; that's the right shape for long-running
    agent processes that build the consult once and reuse it. The
    store keeps lazy SQLite handles open for the source registry and
    (when enabled) the semantic vector store — if you're building a
    short-lived script, hold the returned callable in a context that
    eventually drops out of scope to let the handles close.

    Example::

        from pydantic_ai import Agent
        from outmem.adapters.pydantic_ai import build_consult_wiki

        consult_wiki = build_consult_wiki("/srv/curated-wiki")

        my_assistant = Agent(
            "anthropic:claude-sonnet-4-6",
            tools=[consult_wiki],
            system_prompt=(
                "You're a helpful assistant. For questions about "
                "internal policies or decisions, call `consult_wiki`."
            ),
        )
        result = my_assistant.run_sync("What's our pricing policy?")

    Args:
        wiki_path: Path to a curated wiki directory (must already
            exist; use ``outmem init`` to scaffold one).
        model: Anything :class:`pydantic_ai.Agent` accepts — a model ID
            string (``"anthropic:claude-sonnet-4-6"``), a
            :class:`~pydantic_ai.models.Model` instance, or
            :class:`~pydantic_ai.models.test.TestModel` for tests.
            Defaults to ``anthropic:claude-sonnet-4-6``.
    """
    from pydantic_ai import Agent

    store = WikiStore.open(wiki_path, read_only=True)
    # Library entry point: honour `logfire.project` from config.yaml the
    # same way the CLI's `_open_store` does. Idempotent process-wide.
    from outmem._logfire import setup as _setup_logfire
    _setup_logfire(store.config.outmem.logfire)
    # `model_settings` carries provider-specific Anthropic keys
    # (anthropic_cache*) that aren't in PydanticAI's TypedDict; splat as
    # **kwargs so mypy doesn't try to narrow the dict to ModelSettings.
    agent_kwargs: dict[str, Any] = {"model_settings": _CONSULT_MODEL_SETTINGS}
    inner_agent: Agent[None, str] = Agent(
        model,
        tools=wiki_read_tools(store),
        system_prompt=_CONSULT_WIKI_SYSTEM_PROMPT,
        **agent_kwargs,
    )

    def consult_wiki(question: str) -> str:
        """Ask the team's curated knowledge base about a question.

        Use this when the user asks about topics that might be in our
        internal documentation — policies, decisions, customer history,
        technical patterns, etc. Returns the wiki's synthesised answer
        with ``[[slug]]`` citations, or a clear "no record" if the
        wiki has nothing on the topic.

        Example:
            consult_wiki(question="What's our standard pricing formula?")

        Args:
            question: A natural-language question for the knowledge base.
        """
        result = inner_agent.run_sync(question)
        return str(result.output)

    return consult_wiki


# ---------------------------------------------------------------------------
# Skill text for system-prompt injection
# ---------------------------------------------------------------------------


def skill_text(
    skill_name: str,
    *,
    skills_dir: Path | None = None,
) -> str:
    """Return the SKILL.md body (frontmatter stripped) for a bundled
    outmem skill.

    Use this when a PydanticAI agent doesn't have a native skill-loader
    and you want to inject outmem's procedural guidance directly into
    the system prompt. With the default skills root (the bundled
    ``src/outmem/skills/`` directory) the available skills are
    ``search``, ``evolution``, and ``write`` — all under the
    ``notes/`` category.

    The YAML frontmatter (``name``, ``description``) is stripped — it's
    loader metadata, not content for the model. Loading is delegated
    to :class:`outskilled.SkillRegistry`.

    Example::

        from pydantic_ai import Agent
        from outmem.adapters.pydantic_ai import skill_text

        system = "You are an agent. " + skill_text("write")
        agent = Agent("anthropic:claude-sonnet-4-6", system_prompt=system, ...)

    Args:
        skill_name: One of the bundled skill names (e.g. ``"write"``).
        skills_dir: Override the skills root. Default is the bundled
            ``src/outmem/skills/`` directory.
    """
    if skills_dir is None:
        registry = bundled_registry()
    else:
        from outskilled import SkillRegistry  # type: ignore[import-untyped]

        registry = SkillRegistry([skills_dir])
    body: str = registry.load(skill_name)
    return body
