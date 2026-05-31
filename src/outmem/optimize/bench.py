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

import contextvars
import math
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from outmem.config import DEFAULT_OPTIMIZE_CONCURRENCY, DEFAULT_OPTIMIZE_K
from outmem.optimize.blocks import Retriever
from outmem.optimize.dataset import Question, QuestionBank


@dataclass(frozen=True)
class QuestionResult:
    question: str
    answerable: bool
    gold_slugs: tuple[str, ...]
    retrieved: tuple[str, ...]
    correct: bool  # answerable: gold in top-k; unanswerable: retrieved empty
    latency_ms: float = 0.0  # wall-clock of this single retrieve() call


@dataclass(frozen=True)
class Scorecard:
    score: float  # the one scalar — mean correctness over the whole bank
    hit_at_k: float  # answerable sub-rate
    abstention: float  # unanswerable sub-rate
    k: int
    n_answerable: int
    n_unanswerable: int
    results: tuple[QuestionResult, ...]
    notes: tuple[str, ...] = ()  # retriever diagnostics (e.g. rerank fallbacks), deduped
    mean_latency_ms: float = 0.0  # mean per-search wall-clock
    p95_latency_ms: float = 0.0  # 95th-percentile per-search wall-clock

    @property
    def failures(self) -> tuple[QuestionResult, ...]:
        return tuple(r for r in self.results if not r.correct)


def evaluate(
    retriever: Retriever,
    bank: QuestionBank,
    *,
    k: int = DEFAULT_OPTIMIZE_K,
    max_concurrency: int = DEFAULT_OPTIMIZE_CONCURRENCY,
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

    The scorecard also reports per-search wall-clock (``mean_latency_ms`` /
    ``p95_latency_ms``, and ``latency_ms`` on each :class:`QuestionResult``)
    so a faster strategy can be preferred among configs that score alike.

    All shipped blocks are thread-safe under the pool (``semantic``
    serialises access to its shared sqlite connection on a per-instance
    lock; ``bm25`` uses a per-thread FTS5 connection), so the default
    concurrency is safe across strategies.
    """
    answerable = bank.answerable
    if sample is not None and sample < len(answerable):
        answerable = random.Random(seed).sample(answerable, sample)

    items: list[tuple[Question, bool]] = [(q, True) for q in answerable] + [
        (q, False) for q in bank.unanswerable
    ]

    def _run(item: tuple[Question, bool]) -> tuple[QuestionResult, str | None]:
        q, is_answerable = item
        t0 = time.perf_counter()
        result = retriever.retrieve(q.question, k=k)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        retrieved = result.slugs
        correct = (
            any(g in retrieved[:k] for g in q.gold_slugs)
            if is_answerable
            else len(retrieved) == 0
        )
        qr = QuestionResult(
            question=q.question,
            answerable=is_answerable,
            gold_slugs=q.gold_slugs if is_answerable else (),
            retrieved=retrieved,
            correct=correct,
            latency_ms=latency_ms,
        )
        return qr, result.note

    if max_concurrency > 1 and len(items) > 1:
        # Carry the current context (incl. the active OTEL/logfire span) into
        # each worker so per-question retrieval spans nest under the caller's
        # eval span instead of appearing as flat roots. contextvars is stdlib,
        # so this adds no otel/logfire dependency.
        ctx = contextvars.copy_context()

        def _run_in_ctx(item: tuple[Question, bool]) -> tuple[QuestionResult, str | None]:
            return ctx.copy().run(_run, item)

        # NOT a `with` block: ThreadPoolExecutor.__exit__ always does
        # shutdown(wait=True), which on Ctrl+C joins workers still blocked on
        # an in-flight embed — those waits can't see the (main-thread-only)
        # KeyboardInterrupt, so exit hangs. On abort we instead cancel the
        # shared embed loop's tasks (unblocking the workers' .result()) and
        # shut down without waiting, so the interrupt propagates promptly.
        pool = ThreadPoolExecutor(max_workers=max_concurrency)
        try:
            pairs = list(pool.map(_run_in_ctx, items))  # preserves input order
            pool.shutdown(wait=True)
        except BaseException:
            _cancel_inflight_embeds()
            pool.shutdown(wait=False, cancel_futures=True)
            raise
    else:
        pairs = [_run(it) for it in items]

    results = [qr for qr, _ in pairs]
    # Dedupe retriever diagnostics with a count (e.g. "…refusal" x 30 questions).
    note_counts = Counter(note for _, note in pairs if note)
    notes = tuple(f"{n} (x{c})" if c > 1 else n for n, c in note_counts.items())

    n_ans = len(answerable)
    n_unans = len(bank.unanswerable)
    total = n_ans + n_unans
    answerable_hits = sum(r.correct for r in results if r.answerable)
    abstained = sum(r.correct for r in results if not r.answerable)
    mean_latency, p95_latency = _latency_stats([r.latency_ms for r in results])
    return Scorecard(
        score=(answerable_hits + abstained) / total if total else 0.0,
        hit_at_k=answerable_hits / n_ans if n_ans else 0.0,
        abstention=abstained / n_unans if n_unans else 0.0,
        k=k,
        n_answerable=n_ans,
        n_unanswerable=n_unans,
        results=tuple(results),
        notes=notes,
        mean_latency_ms=mean_latency,
        p95_latency_ms=p95_latency,
    )


def _cancel_inflight_embeds() -> None:
    """Best-effort: cancel any in-flight embed calls on the shared loop so a
    Ctrl+C abort doesn't wedge on a worker join. Lazy import keeps bench
    free of the optional ``semantic`` extra for lexical/bm25-only evals."""
    try:
        from outmem.semantic.embeddings import cancel_inflight
    except ImportError:
        return
    cancel_inflight()


def _latency_stats(latencies: list[float]) -> tuple[float, float]:
    """(mean, p95) of per-search latencies in ms; (0, 0) if empty.

    Note: under concurrency these are per-call wall times measured inside
    worker threads — a fair latency-per-search signal, not a measure of
    total throughput (which the concurrency hides)."""
    if not latencies:
        return 0.0, 0.0
    mean = sum(latencies) / len(latencies)
    ordered = sorted(latencies)
    # nearest-rank p95: smallest rank covering ≥95% of samples (ceil, not
    # round — round's banker's tie-break drifts the rank inconsistently).
    rank = math.ceil(0.95 * len(ordered))  # 1-based
    idx = min(len(ordered), rank) - 1
    return mean, ordered[idx]
