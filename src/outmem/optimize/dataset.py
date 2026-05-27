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

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
) -> QuestionBank:
    """Generate a provenance-labelled bank from the wiki.

    For each page (optionally a ``slugs`` subset / ``max_pages`` cap),
    ask ``model`` for ``per_page`` natural questions the page answers;
    gold = that slug. Unanswerable questions are harvested from the gap
    log when ``include_unanswerable``.
    """
    page_slugs = slugs if slugs is not None else store.list_slugs()
    if max_pages is not None:
        page_slugs = page_slugs[:max_pages]

    answerable: list[Question] = []
    for slug in page_slugs:
        try:
            page = store.read(slug)
        except OutmemError as exc:  # one malformed page must not abort the bank
            log.warning("skipping unreadable page %r (%s)", slug, exc)
            continue
        source = _first_source(page.frontmatter)
        for q in _generate_for_page(model, page.title, page.body, per_page):
            answerable.append(Question(question=q, gold_slugs=(slug,), source=source))

    # …but if EVERY page produced nothing, that's a systemic failure (bad
    # model id, missing/invalid API key) — surface it loudly rather than
    # return a silently empty bank the optimizer would "tune" against.
    if page_slugs and not answerable:
        raise OutmemError(
            f"question generation produced no questions across {len(page_slugs)} "
            "page(s) — check the model id and API key"
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


def _generate_for_page(model: Any, title: str, body: str, n: int) -> list[str]:
    from pydantic_ai import Agent

    agent_kwargs: dict[str, Any] = {"model_settings": {"max_tokens": 1024}}
    agent: Agent[None, _Questions] = Agent(
        model,
        output_type=_Questions,
        system_prompt=_GEN_SYSTEM_PROMPT,
        **agent_kwargs,
    )
    prompt = f"Write {n} questions.\n\nTITLE: {title}\n\nBODY:\n{body[:4000]}"
    try:
        run = agent.run_sync(prompt)
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
