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

from dataclasses import dataclass

from outmem.optimize.blocks import Retriever
from outmem.optimize.dataset import QuestionBank


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


def evaluate(retriever: Retriever, bank: QuestionBank, *, k: int = 5) -> Scorecard:
    """Run every bank question through ``retriever`` and score it.

    Deterministic given the retriever (the rerank block may call a model
    internally, but scoring is pure set/membership logic).
    """
    results: list[QuestionResult] = []

    answerable_hits = 0
    for q in bank.answerable:
        retrieved = retriever.retrieve(q.question, k=k).slugs
        top_k = retrieved[:k]
        correct = any(g in top_k for g in q.gold_slugs)
        answerable_hits += int(correct)
        results.append(
            QuestionResult(
                question=q.question,
                answerable=True,
                gold_slugs=q.gold_slugs,
                retrieved=retrieved,
                correct=correct,
            )
        )

    abstained = 0
    for q in bank.unanswerable:
        retrieved = retriever.retrieve(q.question, k=k).slugs
        correct = len(retrieved) == 0
        abstained += int(correct)
        results.append(
            QuestionResult(
                question=q.question,
                answerable=False,
                gold_slugs=(),
                retrieved=retrieved,
                correct=correct,
            )
        )

    n_ans = len(bank.answerable)
    n_unans = len(bank.unanswerable)
    total = n_ans + n_unans
    score = (answerable_hits + abstained) / total if total else 0.0
    return Scorecard(
        score=score,
        hit_at_k=answerable_hits / n_ans if n_ans else 0.0,
        abstention=abstained / n_unans if n_unans else 0.0,
        k=k,
        n_answerable=n_ans,
        n_unanswerable=n_unans,
        results=tuple(results),
    )
