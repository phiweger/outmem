"""Test-data generation from the wiki — and the hand-authoring contract.

The bank is the held-constant "prepare.py": a set of questions with
known gold pages. Two ways to get one:

* **Generate it** (:func:`generate_bank`) — a model reads each page and
  emits natural questions that page answers; the gold label is the page
  slug (this measures *retrieval* — "given the fact lives on page X, does
  search surface X from a paraphrased question?" — which is not the
  circular trap, since the question is reworded NL, not the page text).
  Unanswerable questions are harvested from the gap log.
* **Hand-author it** — generation is optional. The bank is just JSON
  (:meth:`QuestionBank.save` / :meth:`QuestionBank.load`), so a team
  with sensitive content writes the file by hand and never sends a page
  to a model. Same downstream path.

Generation uses a model via a lazy ``pydantic_ai`` import, so importing
this module never requires the ``agent`` extra.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from outmem.config import ANTHROPIC_CACHE_SETTINGS
from outmem.exceptions import OutmemError

if TYPE_CHECKING:
    from outmem.store import WikiStore

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Question:
    """One bank question. ``gold_slugs`` empty ⇒ unanswerable (expect abstain)."""

    question: str
    gold_slugs: tuple[str, ...] = ()
    source: str | None = None  # provenance pointer, when grounded


@dataclass
class QuestionBank:
    answerable: list[Question] = field(default_factory=list)
    unanswerable: list[Question] = field(default_factory=list)

    def save(self, path: str | Path) -> None:
        payload = {
            "answerable": [_q_to_dict(q) for q in self.answerable],
            "unanswerable": [_q_to_dict(q) for q in self.unanswerable],
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> QuestionBank:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            answerable=[_q_from_dict(d) for d in data.get("answerable", [])],
            unanswerable=[_q_from_dict(d) for d in data.get("unanswerable", [])],
        )

    def __len__(self) -> int:
        return len(self.answerable) + len(self.unanswerable)


def generate_bank(
    store: WikiStore,
    *,
    model: Any,
    per_page: int = 2,
    slugs: list[str] | None = None,
    max_pages: int | None = None,
    include_unanswerable: bool = True,
    max_concurrency: int = 8,
    on_progress: Callable[[int, int], None] | None = None,
) -> QuestionBank:
    """Generate a provenance-labelled bank from the wiki.

    Volume knobs: ``per_page`` questions per page (a cap), over
    ``store.list_slugs()`` — or a ``slugs`` subset / the first
    ``max_pages``. So roughly ``per_page * pages`` answerable samples,
    plus up to 20 unanswerable harvested from the gap log.

    The per-page model calls run concurrently, at most ``max_concurrency``
    in flight (default 8). ``on_progress(done, total)`` fires as each page
    finishes; by default it prints a live counter to stderr (silent under
    pytest, which captures stderr) — pass your own to route it to a bar or
    a logger.
    """
    page_slugs = slugs if slugs is not None else store.list_slugs()
    if max_pages is not None:
        page_slugs = page_slugs[:max_pages]

    # Reuse outmem's Logfire wiring (no-op unless logfire.project is set) so
    # the generation model calls are traced like the rest of outmem.
    from outmem._logfire import setup as _setup_logfire

    _setup_logfire(store.config.outmem.logfire)

    # Read pages up front (cheap, sequential); skip unreadable ones so one
    # malformed page can't abort the bank.
    pages: list[tuple[str, str, str, str | None]] = []
    for slug in page_slugs:
        try:
            page = store.read(slug)
        except OutmemError as exc:
            log.warning("skipping unreadable page %r (%s)", slug, exc)
            continue
        pages.append((slug, page.title, page.body, _first_source(page.frontmatter)))

    # Generate for all pages concurrently — the slow, I/O-bound part.
    generated = (
        asyncio.run(_generate_pages(model, pages, per_page, max_concurrency, on_progress))
        if pages
        else []
    )
    answerable = [
        Question(question=q, gold_slugs=(slug,), source=source)
        for slug, source, questions in generated
        for q in questions
    ]

    # If pages were readable but EVERY one produced nothing, that's a
    # systemic failure (bad model id / API key) — surface it rather than
    # return a silently empty bank the optimizer would "tune" against.
    if page_slugs and not answerable:
        raise OutmemError(
            f"question generation produced no questions across {len(page_slugs)} "
            "page(s) — all unreadable, or the model id / API key is misconfigured"
        )

    unanswerable = harvest_unanswerable(store) if include_unanswerable else []
    return QuestionBank(answerable=answerable, unanswerable=unanswerable)


def harvest_unanswerable(store: WikiStore, *, limit: int = 20) -> list[Question]:
    """Best-effort: pull question-shaped lines from the gap log.

    The log records "I needed X and had nothing" entries; lines ending
    in ``?`` are real queries the wiki once couldn't answer. Caveat:
    some may have been filled since — treat as a noisy starting set, and
    prune/hand-edit. Returns ``[]`` if there's no log or the search fails.
    """
    try:
        result = store.search(r"\?\s*$", scope="log")
    except OutmemError:
        return []
    out: list[Question] = []
    seen: set[str] = set()
    for hit in result.hits:
        text = hit.text.strip().lstrip("#-* ").strip()
        if text and text.endswith("?") and text not in seen:
            seen.add(text)
            out.append(Question(question=text, gold_slugs=()))
        if len(out) >= limit:
            break
    return out


# --- internals -------------------------------------------------------------


@dataclass
class _Questions:
    questions: list[str] = field(default_factory=list)


_GEN_SYSTEM_PROMPT = (
    "You write evaluation questions for a retrieval benchmark. Given a "
    "wiki page (title + body), produce concise, natural questions a real "
    "user would ask THAT this page answers.\n\n"
    "Rules:\n"
    "- Reword in natural language; do NOT copy sentences from the page.\n"
    "- Each question must be answerable from THIS page alone.\n"
    "- No meta questions about the page itself ('what is this page about').\n"
    "- Vary phrasing and angle; prefer the specifics a user would search for."
)


async def _generate_pages(
    model: Any,
    pages: list[tuple[str, str, str, str | None]],
    per_page: int,
    max_concurrency: int,
    on_progress: Callable[[int, int], None] | None,
) -> list[tuple[str, str | None, list[str]]]:
    """Generate questions for every page, at most ``max_concurrency`` in
    flight. Returns ``(slug, source, questions)`` in input order."""
    from pydantic_ai import Agent

    agent_kwargs: dict[str, Any] = {
        "model_settings": {**ANTHROPIC_CACHE_SETTINGS, "max_tokens": 1024}
    }
    agent: Agent[None, _Questions] = Agent(
        model,
        output_type=_Questions,
        system_prompt=_GEN_SYSTEM_PROMPT,
        **agent_kwargs,
    )
    sem = asyncio.Semaphore(max(1, max_concurrency))
    total = len(pages)
    results: list[tuple[str, str | None, list[str]]] = [
        (slug, source, []) for slug, _, _, source in pages
    ]
    done = 0

    async def _one(index: int) -> None:
        nonlocal done
        slug, title, body, source = pages[index]
        async with sem:
            questions = await _gen_one(agent, title, body, per_page)
        results[index] = (slug, source, questions)
        done += 1
        _report_progress(on_progress, done, total)

    await asyncio.gather(*(_one(i) for i in range(total)))
    return results


async def _gen_one(agent: Any, title: str, body: str, n: int) -> list[str]:
    prompt = f"Write {n} questions.\n\nTITLE: {title}\n\nBODY:\n{body[:4000]}"
    try:
        run = await agent.run(prompt)
    except Exception as exc:  # one page failing must not abort the whole bank…
        log.warning("question generation failed for a page (%s); skipping", exc)
        return []
    out: list[str] = []
    for q in run.output.questions:
        q = q.strip()
        if q and q not in out:
            out.append(q)
        if len(out) >= n:
            break
    return out


def _report_progress(
    on_progress: Callable[[int, int], None] | None, done: int, total: int
) -> None:
    if on_progress is not None:
        try:
            on_progress(done, total)
        except Exception as exc:  # a progress callback must never break generation
            log.warning("on_progress raised (%s); ignoring", exc)
        return
    # Default: a live counter on stderr — `\r`-updated on a TTY, one line
    # per page otherwise (Jupyter, redirected output). pytest captures
    # stderr, so this is silent in test runs. Pass on_progress to route
    # progress elsewhere (a bar, a logger).
    end = "\r" if (sys.stderr.isatty() and done < total) else "\n"
    sys.stderr.write(f"generate_bank: {done}/{total} pages{end}")
    sys.stderr.flush()


def _first_source(frontmatter: Any) -> str | None:
    """First provenance pointer as a string, or None. Provenance entries
    are ``str`` (a path) or ``dict`` (path + ingestion metadata)."""
    prov = getattr(frontmatter, "provenance", None) or []
    if not prov:
        return None
    entry = prov[0]
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        for key in ("path", "source", "ref"):
            val = entry.get(key)
            if val:
                return str(val)
    return None


def _q_to_dict(q: Question) -> dict[str, Any]:
    return {"question": q.question, "gold_slugs": list(q.gold_slugs), "source": q.source}


def _q_from_dict(d: dict[str, Any]) -> Question:
    return Question(
        question=str(d["question"]),
        gold_slugs=tuple(d.get("gold_slugs", []) or []),
        source=d.get("source"),
    )
