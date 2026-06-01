"""Agent-driven config search — the user-facing "optimize" loop.

Not a grid sweep. An agent is given two tools — ``run_eval`` (score a
config on the bank) and ``read_page`` (inspect a wiki page) — and asked
to *navigate* the small retrieval search space: try a config, look at
which questions failed and what the gold pages actually say, form a
hypothesis ("lexical misses paraphrased questions → try rerank"), and
pick the next config to try. It stops when it stops improving or hits
the eval budget.

We **trust the metric, not the agent's self-report**: every config the
agent evaluates is recorded with its score, and :func:`optimize_retrieval`
returns the best-scoring config seen — the agent's closing rationale is
advisory commentary. A confused agent can waste budget but can't hand
back a worse config than it measured.

This is the *config-space* loop (safe: only picks among shipped, tested
blocks). The *code-space* loop that writes new blocks is the
maintainer-side PR-bot described in ``improve.md`` — deliberately not
here.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from outmem.config import (
    ANTHROPIC_CACHE_WITH_TOOLS,
    DEFAULT_OPTIMIZE_CONCURRENCY,
    DEFAULT_OPTIMIZE_K,
    DEFAULT_OPTIMIZE_MAX_CANDIDATES,
    DEFAULT_OPTIMIZE_MAX_EVALS,
    DEFAULT_OPTIMIZE_MAX_FAILURES_SHOWN,
    DEFAULT_OPTIMIZE_MAX_RELEVANT,
    DEFAULT_OPTIMIZE_RERANK_SOURCE,
    DEFAULT_OPTIMIZE_RRF_K,
    DEFAULT_OPTIMIZE_SEMANTIC_TOP_K,
    DEFAULT_OPTIMIZE_STRATEGY,
    DEFAULT_RELEVANCE_MODEL,
    RETRIEVAL_FILENAME,
)
from outmem.exceptions import OutmemError
from outmem.optimize.bench import Scorecard, evaluate
from outmem.optimize.blocks import RetrievalConfig, build_retriever
from outmem.optimize.dataset import QuestionBank

if TYPE_CHECKING:
    from outmem.store import WikiStore

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvalRow:
    """One row of the post-run summary table — compact per-eval stats.

    Kept lean (no per-question results) so a 12-eval run holds a dozen
    of these in memory regardless of bank size, but rich enough to print
    a useful scorecard and let the user pick a config by rank."""

    config: RetrievalConfig
    score: float
    hit_at_k: float
    abstention: float
    mean_latency_ms: float
    p95_latency_ms: float
    # Bank composition (same for every row of a run). Carried so the table
    # can hide the abstention column when there were no unanswerable
    # questions to measure it against — see `_format_summary_table`.
    n_unanswerable: int = 0


def _eval_row(cfg: RetrievalConfig, card: Scorecard) -> EvalRow:
    """Project a scored config + its :class:`Scorecard` into one summary row."""
    return EvalRow(
        config=cfg,
        score=card.score,
        hit_at_k=card.hit_at_k,
        abstention=card.abstention,
        mean_latency_ms=card.mean_latency_ms,
        p95_latency_ms=card.p95_latency_ms,
        n_unanswerable=card.n_unanswerable,
    )


@dataclass
class OptimizeResult:
    best_config: RetrievalConfig
    best_score: float
    scorecard: Scorecard
    trace: list[tuple[dict[str, Any], float]]  # (config, score) in eval order
    notes: str  # the agent's closing rationale (advisory)
    log: list[str] = field(default_factory=list)  # diagnostics (errors/fallbacks)
    # Per-eval summary rows. `__post_init__` sorts these into the same
    # score-descending (latency-ascending tiebreak) order the table prints
    # and `pick(rank)` indexes, so the rank a user reads off
    # `print_summary()` is exactly what `pick(rank)` / `save(rank)` resolve.
    # Distinct from `trace`, which preserves eval order for dedupe lookups
    # and the agent's own reasoning.
    summary: list[EvalRow] = field(default_factory=list)

    def __post_init__(self) -> None:
        # The optimizer appends rows in eval-arrival order; the leaderboard
        # is by score. Sort once here so `summary`, `summary_table()`, and
        # `pick(rank)` share one ordering — otherwise `save(1)` could
        # persist the first config tried instead of the highest-scoring one.
        self.summary.sort(key=lambda r: (-r.score, r.mean_latency_ms))

    def summary_table(self) -> str:
        """Format the post-run leaderboard as plain text.

        One row per scored config, ranked by score then by latency
        (lower wins). The top row is what :attr:`best_config` already
        names; ``pick(rank)`` / ``save(rank, ...)`` index in here."""
        return _format_summary_table(self.summary)

    def print_summary(self, stream: Any = None) -> None:
        """Print :meth:`summary_table` to ``stream`` (default ``sys.stderr``).

        Matches the optimizer's per-eval progress lines (also stderr) so
        the table lands in the same stream as the running output."""
        if stream is None:
            stream = sys.stderr
        stream.write(self.summary_table())
        stream.write("\n")
        stream.flush()

    def pick(self, rank: int) -> RetrievalConfig:
        """Return the :class:`RetrievalConfig` at the given 1-based rank
        in :attr:`summary`. Raises :class:`OutmemError` on out-of-range
        so callers don't silently get a default config."""
        if not self.summary:
            raise OutmemError("no evals were scored — nothing to pick")
        if not 1 <= rank <= len(self.summary):
            raise OutmemError(
                f"rank {rank} out of range; the table has "
                f"{len(self.summary)} rows (1..{len(self.summary)})"
            )
        return self.summary[rank - 1].config

    def save(
        self,
        rank: int,
        store: WikiStore,
        *,
        path: Path | None = None,
    ) -> Path:
        """Write the picked config to ``<wiki>/retrieval.yaml`` as a
        ``retrieval:`` block with ``from_optimization: true``.

        Overwrites the file wholesale — ``retrieval.yaml`` is a
        machine-owned artefact, so any hand-edits or comments in a prior
        copy are replaced. Returns the absolute path written.

        The write is atomic (temp file + ``os.replace``) and creates the
        parent directory, so an interrupted save can't leave a truncated
        file that the next load would reject. A read-only store, or any
        OS-level write failure, raises :class:`OutmemError` rather than
        leaking a bare ``OSError``.

        Unless ``path=`` overrides the destination, the in-memory
        ``store.config.outmem.retrieval`` is refreshed to match what was
        written, so an :class:`~pydantic_ai.Agent` built from this same
        ``store`` picks up the new pipeline without a reopen. With a custom
        ``path=`` (e.g. in tests) the store is left untouched.
        """
        cfg = self.pick(rank)
        default_dest = path is None
        target = path or (store.root / RETRIEVAL_FILENAME)
        if default_dest and store.config.read_only:
            raise OutmemError(
                "cannot save retrieval.yaml: store was opened read_only"
            )
        text = _render_retrieval_yaml(cfg)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(f"{target.name}.tmp")
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, target)
        except OSError as exc:
            raise OutmemError(f"failed to write {target}: {exc}") from exc
        if default_dest:
            # Keep the live store consistent with what's now on disk. Reload
            # from the file (single source of truth) rather than mirroring
            # the render, so this can't drift from `load_yaml_config`.
            from outmem.config import load_yaml_config

            store.config.outmem.retrieval = load_yaml_config(store.root).retrieval
        return target


@dataclass(frozen=True)
class EvalEvent:
    """One scored step of the optimizer loop — the ``on_eval`` payload (an
    "epoch": the config just tried, its metrics, and the best so far)."""

    index: int  # 1-based, among scored evals
    max_evals: int
    config: RetrievalConfig
    scorecard: Scorecard
    best_score: float  # best score seen so far, this eval included


_OPTIMIZER_SYSTEM_PROMPT = (
    "You are tuning a retrieval pipeline for a specific wiki. You cannot "
    "edit code; you choose among composable blocks via their config. Your "
    "job: find the config that MAXIMISES the benchmark score.\n\n"
    "Work empirically and frugally: evaluate a config with `run_eval`, then "
    "READ the failing questions' gold pages with `read_page` to understand "
    "WHY retrieval missed (wrong keywords? paraphrase the lexical block "
    "can't match? a reranker discarding the right page?). Form a hypothesis, "
    "try the next config, keep what the score rewards. Don't brute-force the "
    "grid — move deliberately.\n\n"
    "Cover the strategy families before declaring a winner: at minimum try "
    "one of {lexical, bm25}, one of {semantic, hyde}, one hybrid fuse, and "
    "the rerank gate over BOTH a lexical and a semantic candidate source "
    "(`rerank_source` knob — feeding the LLM yes/no judge a high-recall "
    "semantic shortlist instead of a keyword net is the actually-good "
    "rerank pairing on paraphrase-heavy banks). "
    "A perfect score on the first or second eval is almost always a small-"
    "sample illusion (10 questions, score=1.000 has a 95% CI lower bound of "
    "~0.69) — keep going to see whether a cheaper/faster strategy ties it, "
    "or whether a different family actually wins on a tighter sample. "
    "Stop early only when several distinct strategies have plateaued at "
    "similar scores, or when the budget is spent.\n\n"
    "Calling `run_eval` with a config you've already tried is a no-op — it "
    "returns the prior score without consuming an eval slot — so vary at "
    "least one parameter each turn."
)

_MODEL_SETTINGS: dict[str, Any] = {
    **ANTHROPIC_CACHE_WITH_TOOLS,  # the optimizer agent exposes run_eval/read_page
    "max_tokens": 8192,
}


def optimize_retrieval(
    store: WikiStore,
    bank: QuestionBank,
    *,
    optimizer_model: Any,
    rerank_model: Any = None,
    k: int = DEFAULT_OPTIMIZE_K,
    eval_concurrency: int = DEFAULT_OPTIMIZE_CONCURRENCY,
    eval_sample: int | None = None,
    max_evals: int = DEFAULT_OPTIMIZE_MAX_EVALS,
    max_failures_shown: int = DEFAULT_OPTIMIZE_MAX_FAILURES_SHOWN,
    on_eval: Callable[[EvalEvent], None] | None = None,
) -> OptimizeResult:
    """Let ``optimizer_model`` search the config space over ``bank``.

    ``rerank_model`` overrides the rerank block's model object (pass a
    cheap model / a ``FunctionModel`` in tests); ``None`` uses each
    config's ``rerank_model`` string. ``max_evals`` soft-caps how many
    configs the agent may score (the "turn budget").

    **Cost control.** ``rerank`` and ``hyde`` evals (and any ``hybrid``
    that fuses one of those) make one model call per bank question, so
    cost ≈ ``bank_size * (rerank + hyde) evals`` (plus the optimizer's
    own turns). Pure ``lexical`` / ``bm25`` / ``semantic`` / ``hybrid[
    lexical+semantic]`` evals are free of LLM cost; semantic query
    embeddings are cached per text, so repeated questions across evals
    re-embed at most once. Two knobs bound the expensive evals:
    ``eval_concurrency`` (default 8) runs each eval's per-question calls
    in parallel, and ``eval_sample`` caps the answerable questions scored
    *per eval* to a fixed seeded subset — the winner is then re-scored on
    the full bank so the reported score is honest. See
    ``docs/autoresearch.md`` for the full run + logging recipe.

    ``on_eval(EvalEvent)`` fires once per scored eval — an epoch-style
    progress hook carrying the config just tried, its metrics, and the
    best score so far. By default it prints one line per eval to stderr
    (silent under pytest), e.g. ``[eval 3/12] strategy=rerank score=0.620
    (hit@5=0.550 abstain=0.800) best=0.710``; wire it to your own display
    or a logger if you like.
    """
    # Reuse outmem's Logfire wiring (no-op unless logfire.enabled is set);
    # instrument_pydantic_ai is process-global, so this one call traces the
    # optimizer agent AND the per-question rerank calls in the loop.
    from outmem._logfire import setup as _setup_logfire
    from outmem._logfire import span as _span

    _setup_logfire(store.config.outmem.logfire)

    from pydantic_ai import Agent

    trace: list[tuple[dict[str, Any], float]] = []
    best: dict[str, Any] = {"score": -1.0, "cfg": None, "card": None}
    run_log: list[str] = []  # errors / fallbacks, surfaced on OptimizeResult.log
    # Per-eval summary rows, accumulated for the post-run leaderboard
    # (`OptimizeResult.summary` / `print_summary` / `pick` / `save`).
    summary_rows: list[EvalRow] = []
    # Dedupe cache: an agent can wander into the same (strategy, params) twice
    # across 12 turns. Returning the cached scorecard without burning an eval
    # slot keeps the budget for genuinely new configs and stops the trace
    # filling up with `semantic / semantic / semantic` lines.
    seen: dict[str, tuple[RetrievalConfig, Scorecard]] = {}

    def run_eval(
        strategy: str = DEFAULT_OPTIMIZE_STRATEGY,
        case_insensitive: bool = True,
        max_candidates: int = DEFAULT_OPTIMIZE_MAX_CANDIDATES,
        rerank_model_id: str = DEFAULT_RELEVANCE_MODEL,
        max_relevant: int = DEFAULT_OPTIMIZE_MAX_RELEVANT,
        rerank_source: str = DEFAULT_OPTIMIZE_RERANK_SOURCE,
        semantic_top_k: int = DEFAULT_OPTIMIZE_SEMANTIC_TOP_K,
        rrf_k: int = DEFAULT_OPTIMIZE_RRF_K,
        hyde_model_id: str = DEFAULT_RELEVANCE_MODEL,
        fuse: list[str] | None = None,
    ) -> str:
        """Score one retrieval config on the benchmark and report the
        result plus a sample of failing questions.

        Args:
            strategy: "lexical" (keyword frequency rank), "bm25" (SQLite
                FTS5 BM25 ranking — no model/index needed), "rerank"
                (candidate generator + cheap-model yes/no relevance gate;
                source picked via `rerank_source`), "semantic" (vector
                similarity), "hyde" (generate a hypothetical answer, then
                semantic-search on it — needs a model + the index), or
                "hybrid" (Reciprocal Rank Fusion of the `fuse` legs).
            case_insensitive: case-fold the keyword search.
            max_candidates: width of the candidate net before reranking.
            rerank_model_id: model id for the rerank block.
            max_relevant: cap on pages the rerank block keeps.
            rerank_source: which atomic block builds the rerank candidate
                shortlist ("lexical","bm25","semantic","hyde"). The classic
                pairing is "lexical"; the high-recall pairing is "semantic"
                — try both, they score very differently.
            semantic_top_k: neighbours for the semantic / hyde / hybrid blocks.
            rrf_k: Reciprocal Rank Fusion constant for the hybrid block.
            hyde_model_id: model id the hyde block uses to generate the
                hypothetical answer.
            fuse: for strategy="hybrid", the 2+ atomic legs to fuse, e.g.
                ["lexical","semantic"], ["bm25","semantic"], or
                ["semantic","hyde"]. Ignored for non-hybrid strategies.
        """
        if len(trace) >= max_evals:
            return (
                f"Eval budget exhausted ({max_evals}). Stop evaluating and "
                "summarise the best config you found."
            )
        # from_dict raises OutmemError on a bad strategy and _as_int does
        # the same on a bad number — keep it inside the try so a fumbled
        # config is reported back to the agent, not crashed out of the run.
        try:
            cfg_dict: dict[str, Any] = {
                "strategy": strategy,
                "case_insensitive": case_insensitive,
                "max_candidates": max_candidates,
                "rerank_model": rerank_model_id,
                "max_relevant": max_relevant,
                "rerank_source": rerank_source,
                "semantic_top_k": semantic_top_k,
                "rrf_k": rrf_k,
                "hyde_model": hyde_model_id,
            }
            if fuse is not None:
                cfg_dict["fuse"] = fuse
            cfg = RetrievalConfig.from_dict(cfg_dict)
            fingerprint = _config_fingerprint(cfg)
            if fingerprint in seen:
                prior_card = seen[fingerprint][1]
                return (
                    f"already evaluated this exact config on eval "
                    f"{_index_of(trace, fingerprint) + 1} "
                    f"(score={prior_card.score:.3f}); pick a different one. "
                    "Evals left unchanged."
                )
            retriever = build_retriever(store, cfg, model=rerank_model)
            # One span per eval nests this config's per-question retrieval
            # calls under it in the trace.
            with _span(f"eval {len(trace) + 1}: {cfg.strategy}", **cfg.to_dict()):
                card = evaluate(
                    retriever, bank, k=k,
                    max_concurrency=eval_concurrency, sample=eval_sample,
                )
        except OutmemError as exc:
            log.info("optimize: skipped strategy=%s (%s)", strategy, exc)
            run_log.append(f"[eval attempt] strategy={strategy} unavailable: {exc}")
            # Surface on stderr next to the epoch lines (default UI only).
            # Otherwise a whole family silently vanishing — e.g. semantic /
            # hyde / semantic-hybrid when the index isn't built — looks like
            # the agent ignored it, when really every such config was
            # refused. Seeing "unavailable: index is empty" tells the user to
            # run `outmem reindex` instead of wondering why no semantic ran.
            if on_eval is None:
                sys.stderr.write(f"[skip] {strategy} unavailable: {exc}\n")
                sys.stderr.flush()
            return f"config unavailable: {exc}"
        trace.append((cfg.to_dict(), card.score))
        seen[fingerprint] = (cfg, card)
        summary_rows.append(_eval_row(cfg, card))
        if card.score > best["score"]:
            best.update(score=card.score, cfg=cfg, card=card)
        for note in card.notes:  # e.g. a rerank model that refused on N questions
            run_log.append(f"[eval {len(trace)}] {cfg.strategy}: {note}")
        _report_eval(
            on_eval,
            EvalEvent(
                index=len(trace),
                max_evals=max_evals,
                config=cfg,
                scorecard=card,
                best_score=best["score"],
            ),
        )
        return _format_card(cfg, card, remaining=max_evals - len(trace),
                            max_failures_shown=max_failures_shown,
                            eval_sample=eval_sample)

    def read_page(slug: str) -> str:
        """Read a wiki page's body (truncated) to diagnose why retrieval
        missed it. Use on the gold slugs of failing questions."""
        try:
            return store.read(slug).body[:2000]
        except OutmemError as exc:
            return f"(no such page {slug!r}: {exc})"

    agent_kwargs: dict[str, Any] = {"model_settings": _MODEL_SETTINGS}
    agent: Agent[None, str] = Agent(
        optimizer_model,
        tools=[run_eval, read_page],
        system_prompt=_OPTIMIZER_SYSTEM_PROMPT,
        **agent_kwargs,
    )
    # One parent span nests the optimizer's own turns and every per-eval
    # span (and their per-question children) under a single run in the trace.
    _emit_metric_context(store, k, n_unanswerable=len(bank.unanswerable))
    # Warm the query-embedding cache up front so the FIRST semantic-family
    # eval isn't a latency outlier vs. the rest (see the helper's docstring).
    _prewarm_query_cache(store, bank, eval_concurrency)
    with _span("optimize_retrieval", max_evals=max_evals, k=k):
        run = agent.run_sync(_initial_prompt(bank, k, max_evals))
    notes = str(run.output)

    if best["cfg"] is None:  # agent never produced a scorable config
        cfg = RetrievalConfig()
        card = evaluate(
            build_retriever(store, cfg, model=rerank_model),
            bank, k=k, max_concurrency=eval_concurrency,
        )
        # Record the fallback eval so `summary`/`pick`/`save` aren't empty —
        # `best_config` is meaningful here, and a user should be able to
        # persist it the same way as any other winning row.
        summary_rows.append(_eval_row(cfg, card))
        return OptimizeResult(
            cfg, card.score, card, trace, notes,
            log=run_log, summary=summary_rows,
        )

    best_cfg: RetrievalConfig = best["cfg"]
    best_card: Scorecard = best["card"]
    if eval_sample is not None:  # winner chosen on a sample → re-score on full bank
        best_card = evaluate(
            build_retriever(store, best_cfg, model=rerank_model),
            bank, k=k, max_concurrency=eval_concurrency,
        )
    return OptimizeResult(
        best_cfg, best_card.score, best_card, trace, notes,
        log=run_log, summary=summary_rows,
    )


def _config_fingerprint(cfg: RetrievalConfig) -> str:
    """Stable string key for an evaluated config — sorted JSON of `to_dict()`
    so the dedupe cache treats equivalent dicts as the same entry."""
    return json.dumps(cfg.to_dict(), sort_keys=True, default=str)


def _index_of(
    trace: list[tuple[dict[str, Any], float]], fingerprint: str
) -> int:
    """0-based index in `trace` of the first config matching `fingerprint`.
    Caller bumps it by 1 for the human-facing "eval N" label."""
    for i, (cfg_dict, _) in enumerate(trace):
        if json.dumps(cfg_dict, sort_keys=True, default=str) == fingerprint:
            return i
    return -1


def _emit_metric_context(store: WikiStore, k: int, n_unanswerable: int = 0) -> None:
    """Print one line so the user can sanity-check whether the metric is
    informative on their corpus before reading any scores.

    With N pages and cutoff ``k``, the theoretical ceiling is ``min(k,N)/N``
    — well below 1.0 for big corpora, but on a 12-page wiki any retriever
    that returns 4+ top slots already covers a third of the corpus and
    Hit@k saturates near 1.0. The scores stop distinguishing strategies.
    A loud-but-cheap warning here saves an honest "score=1.000 is too good
    to be true" diagnosis after the fact. (Default ``k=1`` already dodges
    this on tiny corpora; the warning catches overrides.)

    Also flags an all-answerable bank up front: with no unanswerables the
    abstention half of the score is unmeasured (``score == hit@k``), which
    is worth knowing before spending the eval budget."""
    try:
        n = len(store.list_slugs())
    except Exception:
        return
    saturated = n > 0 and k / n > 0.25
    flag = "  (⚠ Hit@k saturates — k is a large fraction of the corpus)" if saturated else ""
    abst = "" if n_unanswerable else "  (no unanswerables: score = hit@k)"
    sys.stderr.write(f"corpus: {n} pages, k={k}{flag}{abst}\n")
    sys.stderr.flush()


def _prewarm_query_cache(
    store: WikiStore, bank: QuestionBank, max_concurrency: int
) -> None:
    """Embed every bank question once, *untimed*, before the eval loop.

    The embedder caches query vectors by text (see
    :class:`outmem.semantic.embeddings.EmbedderHandle`), so the optimizer
    re-asks the same bank questions cheaply across evals. The catch for the
    leaderboard: only the FIRST semantic/hyde/hybrid eval pays the cold
    network embedding cost — its per-search latency is then a ~10x outlier
    (e.g. semantic 1348ms cold vs 161ms warm), making the latency column
    non-comparable across rows. Warming the cache up front moves that
    one-off cost out of every timed eval, so all rows measure warm
    retrieval and rank fairly on speed.

    Best-effort and never fatal: a no-op when semantic is disabled or the
    index is empty (the agent will surface that per-config instead), and a
    per-question embed failure is swallowed — it'll resurface, attributed,
    in the real eval."""
    if not store.semantic_enabled():
        return
    try:
        if store.semantic_index_is_empty():
            return
    except Exception:
        return
    questions = [q.question for q in bank.answerable]
    questions += [q.question for q in bank.unanswerable]
    if not questions:
        return

    def _warm(q: str) -> None:
        # Best-effort: a real embed failure resurfaces, attributed, in the eval.
        with contextlib.suppress(Exception):
            store.semantic_find_similar(q, top_k=1, threshold=0.0)

    workers = max(1, min(max_concurrency, len(questions)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(_warm, questions))


def _initial_prompt(bank: QuestionBank, k: int, max_evals: int) -> str:
    return (
        f"Wiki bank: {len(bank.answerable)} answerable + "
        f"{len(bank.unanswerable)} unanswerable questions. Metric (maximise, "
        f"0..1): mean of [answerable: gold page in top-{k}] and "
        f"[unanswerable: retriever returned empty]. You have up to "
        f"{max_evals} `run_eval` calls. Start with the lexical baseline, "
        f"diagnose its failures by reading gold pages, then improve."
    )


def _describe_config(cfg: RetrievalConfig) -> str:
    """Compact, human-readable label of which blocks a trial used AND the
    knobs that distinguish near-identical trials.

    Includes the tuned numeric knobs (candidate width, kept count, fusion
    constant, neighbour count) so that tuning *within* a strategy family —
    e.g. ``rerank[bm25]`` at three different candidate widths — shows as
    three distinct lines instead of looking like the same config run
    repeatedly. The dedupe cache keys on the full config, so identical
    trials are already skipped; these are genuinely different configs."""
    if cfg.strategy == "hybrid":
        return f"hybrid[{'+'.join(cfg.fuse)} rrf={cfg.rrf_k}]"
    if cfg.strategy == "rerank":
        return (
            f"rerank[{cfg.rerank_source}→{_short_model(cfg.rerank_model)} "
            f"cand={cfg.max_candidates} keep={cfg.max_relevant}]"
        )
    if cfg.strategy == "hyde":
        return f"hyde({_short_model(cfg.hyde_model)} k={cfg.semantic_top_k})"
    if cfg.strategy == "semantic":
        return f"semantic[k={cfg.semantic_top_k}]"
    return cfg.strategy


def _short_model(model_id: str) -> str:
    """`anthropic:claude-haiku-4-5` → `haiku`-ish: last path segment, deprefixed."""
    tail = model_id.split(":")[-1]
    return tail.replace("claude-", "")


def _format_epoch(event: EvalEvent) -> str:
    c = event.scorecard
    star = " *" if c.score >= event.best_score else ""  # this eval is (tied) best
    # Only show abstain= when there were unanswerable questions to measure
    # it on; otherwise it's a structural 0.000 and score == hit@k.
    abstain = f" abstain={c.abstention:.3f}" if c.n_unanswerable else ""
    return (
        f"[eval {event.index}/{event.max_evals}] {_describe_config(event.config)} "
        f"score={c.score:.3f} (hit@{c.k}={c.hit_at_k:.3f}{abstain}) "
        f"{c.mean_latency_ms:.0f}ms/search best={event.best_score:.3f}{star}"
    )


def _report_eval(on_eval: Callable[[EvalEvent], None] | None, event: EvalEvent) -> None:
    if on_eval is not None:
        try:
            on_eval(event)
        except Exception as exc:  # a progress callback must never break the loop
            log.warning("on_eval raised (%s); ignoring", exc)
        return
    # Default: print each epoch to stderr (silent under pytest, which
    # captures it). Pass on_eval to route epochs to a logger / your own UI.
    sys.stderr.write(_format_epoch(event) + "\n")
    sys.stderr.flush()


def _format_card(
    cfg: RetrievalConfig,
    card: Scorecard,
    *,
    remaining: int,
    max_failures_shown: int,
    eval_sample: int | None,
) -> str:
    lines = [
        f"score={card.score:.3f}  hit@{card.k}={card.hit_at_k:.3f}  "
        f"abstain={card.abstention:.3f}  "
        f"latency={card.mean_latency_ms:.0f}ms/search (p95 {card.p95_latency_ms:.0f}ms)  "
        f"(evals left: {remaining})",
        f"config: {cfg.to_dict()}",
    ]
    failures = card.failures
    if failures:
        lines.append(f"failures ({len(failures)} total, showing up to {max_failures_shown}):")
        for r in failures[:max_failures_shown]:
            kind = "answerable" if r.answerable else "unanswerable"
            gold = list(r.gold_slugs) if r.gold_slugs else "(should abstain)"
            lines.append(
                f"  [{kind}] Q={r.question!r} gold={gold} got={list(r.retrieved[:card.k])}"
            )
    else:
        # A perfect score on a small sample is almost always sample noise,
        # not a real ceiling — surface that to the agent so it doesn't
        # early-stop on the first config that hits 1.000.
        if eval_sample is not None and eval_sample < 30 and card.score >= 0.99:
            lines.append(
                f"no failures, but eval_sample={eval_sample} is small — "
                "a perfect score here is sample-noise-limited (95% CI lower "
                "bound ~0.69 at n=10). Try other strategy families to see if "
                "they tie at lower cost/latency before declaring a winner."
            )
        else:
            lines.append("no failures.")
    return "\n".join(lines)


def _format_summary_table(rows: list[EvalRow]) -> str:
    """Render the post-run leaderboard.

    Columns: rank | config | score | hit@k | [abst] | ms/q (p95).
    Ordering: score desc, then mean latency asc (faster wins ties — the
    practical tiebreaker when two configs hit the same score).

    The ``abst`` (abstention) column is dropped when the bank had no
    unanswerable questions: with nothing to abstain on, the rate is a
    structural ``0.000`` for every row and ``score`` equals ``hit@k``.
    A banner says so, rather than showing a column of meaningless zeros."""
    if not rows:
        return "(no evals scored)"
    ranked = sorted(rows, key=lambda r: (-r.score, r.mean_latency_ms))
    show_abst = any(r.n_unanswerable > 0 for r in rows)

    def cells(i: int, r: EvalRow) -> tuple[str, ...]:
        base = (str(i), _describe_config(r.config), f"{r.score:.3f}", f"{r.hit_at_k:.3f}")
        abst = (f"{r.abstention:.3f}",) if show_abst else ()
        return (*base, *abst, f"{r.mean_latency_ms:.0f} ({r.p95_latency_ms:.0f})")

    header = ("#", "config", "score", "hit@k", *(("abst",) if show_abst else ()),
              "ms/q (p95)")
    body = [cells(i, r) for i, r in enumerate(ranked, 1)]
    cols = list(zip(header, *body, strict=False))
    widths = [max(len(cell) for cell in col) for col in cols]
    def fmt_row(cells_: tuple[str, ...]) -> str:
        return "  ".join(cell.ljust(w) for cell, w in zip(cells_, widths, strict=False))
    sep = "  ".join("-" * w for w in widths)
    lines = [fmt_row(header), sep, *(fmt_row(row) for row in body)]
    if not show_abst:
        lines.append(
            "\nNo unanswerable questions in the bank: score = hit@k, "
            "abstention not measured. Add some (questions the wiki should "
            "answer with nothing) to test abstention precision."
        )
    lines.append(
        "\nPick a row with `result.save(rank, store)` — writes "
        f"{RETRIEVAL_FILENAME} (from_optimization: true)."
    )
    return "\n".join(lines)


def _render_retrieval_yaml(cfg: RetrievalConfig) -> str:
    """Dump a picked :class:`RetrievalConfig` as ``retrieval.yaml`` text.

    The body goes through ``yaml.safe_dump`` (``sort_keys=False`` to keep
    the field order stable under git diff) so any scalar that needs
    quoting — a model id that parses as a YAML keyword/number, a value
    with a colon — is escaped correctly rather than emitted raw and
    silently dropped on reload. ``strategy`` is the DSL string a human
    would type (``bm25+semantic``, not the expanded ``{strategy: hybrid,
    fuse: [...]}``). A leading comment records provenance."""
    from outmem.optimize.dsl import format_strategy

    block = {
        "retrieval": {
            "strategy": format_strategy(cfg.to_dict()),
            "from_optimization": True,
            "semantic_top_k": cfg.semantic_top_k,
            "rrf_k": cfg.rrf_k,
            "max_candidates": cfg.max_candidates,
            "max_relevant": cfg.max_relevant,
            "rerank_model": cfg.rerank_model,
            "hyde_model": cfg.hyde_model,
            "case_insensitive": cfg.case_insensitive,
        }
    }
    header = (
        "# Written by `outmem.optimize.OptimizeResult.save(...)`.\n"
        "# Loaded by `outmem.config.load_yaml_config`; when present this\n"
        "# file fully defines retrieval. Delete it to revert to config.yaml's\n"
        "# retrieval block / defaults.\n"
    )
    body = yaml.safe_dump(
        block, sort_keys=False, default_flow_style=False, allow_unicode=True
    )
    return header + body
