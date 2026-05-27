"""The metric — one scalar to hill-climb, two sub-rates to diagnose.

Per the design discussion: most provenance-labelled questions have a
single gold page, and outmem feeds the top-k to an LLM regardless of
internal order, so the natural retrieval metric is **Hit@k** (was the
gold page in the top k?), not F1. The unanswerable class wants the
opposite — return *empty* — measured by **abstention**. Neither folds
into the other, so we blend them into one accuracy-style scalar:

    score = mean over the bank of:
        answerable   → 1 if a gold slug is in top-k else 0   (Hit@k)
        unanswerable → 1 if retriever returned empty else 0  (abstained)

``score`` is the optimizer's objective; ``hit_at_k`` and ``abstention``
are reported so you can see *why* it moved. No F1 until list-style
(multi-gold) questions exist.
"""

from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from outmem.optimize.blocks import Retriever
from outmem.optimize.dataset import Question, QuestionBank


@dataclass(frozen=True)
class QuestionResult:
    question: str
    answerable: bool
    gold_slugs: tuple[str, ...]
    retrieved: tuple[str, ...]
    correct: bool  # answerable: gold in top-k; unanswerable: retrieved empty


@dataclass(frozen=True)
class Scorecard:
    score: float  # the one scalar — mean correctness over the whole bank
    hit_at_k: float  # answerable sub-rate
    abstention: float  # unanswerable sub-rate
    k: int
    n_answerable: int
    n_unanswerable: int
    results: tuple[QuestionResult, ...]

    @property
    def failures(self) -> tuple[QuestionResult, ...]:
        return tuple(r for r in self.results if not r.correct)


def evaluate(
    retriever: Retriever,
    bank: QuestionBank,
    *,
    k: int = 5,
    max_concurrency: int = 8,
    sample: int | None = None,
    seed: int = 0,
) -> Scorecard:
    """Run the bank through ``retriever`` and score it.

    The per-question ``retrieve`` calls run concurrently, at most
    ``max_concurrency`` in flight (default 8) — the win for the ``rerank``
    / ``hybrid`` blocks, which make one model call per question. Scoring
    itself is pure set/membership logic.

    ``sample`` caps how many *answerable* questions are scored — a seeded,
    reproducible subset (so different configs are compared on the *same*
    questions); the ``unanswerable`` set (small by construction) is always
    scored whole. Use it to bound per-eval cost while tuning, then
    re-score the winner on the full bank (``sample=None``).

    Note: the ``semantic`` / ``hybrid`` blocks hold a SQLite connection;
    if you hit a cross-thread SQLite error, score those with
    ``max_concurrency=1`` (``lexical`` / ``rerank`` are thread-safe).
    """
    answerable = bank.answerable
    if sample is not None and sample < len(answerable):
        answerable = random.Random(seed).sample(answerable, sample)

    items: list[tuple[Question, bool]] = [(q, True) for q in answerable] + [
        (q, False) for q in bank.unanswerable
    ]

    def _run(item: tuple[Question, bool]) -> QuestionResult:
        q, is_answerable = item
        retrieved = retriever.retrieve(q.question, k=k).slugs
        correct = (
            any(g in retrieved[:k] for g in q.gold_slugs)
            if is_answerable
            else len(retrieved) == 0
        )
        return QuestionResult(
            question=q.question,
            answerable=is_answerable,
            gold_slugs=q.gold_slugs if is_answerable else (),
            retrieved=retrieved,
            correct=correct,
        )

    if max_concurrency > 1 and len(items) > 1:
        with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
            results = list(pool.map(_run, items))  # preserves input order
    else:
        results = [_run(it) for it in items]

    n_ans = len(answerable)
    n_unans = len(bank.unanswerable)
    total = n_ans + n_unans
    answerable_hits = sum(r.correct for r in results if r.answerable)
    abstained = sum(r.correct for r in results if not r.answerable)
    return Scorecard(
        score=(answerable_hits + abstained) / total if total else 0.0,
        hit_at_k=answerable_hits / n_ans if n_ans else 0.0,
        abstention=abstained / n_unans if n_unans else 0.0,
        k=k,
        n_answerable=n_ans,
        n_unanswerable=n_unans,
        results=tuple(results),
    )
