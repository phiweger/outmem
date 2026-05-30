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

import json
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from outmem.config import (
    ANTHROPIC_CACHE_WITH_TOOLS,
    DEFAULT_OPTIMIZE_CONCURRENCY,
    DEFAULT_OPTIMIZE_K,
    DEFAULT_OPTIMIZE_MAX_CANDIDATES,
    DEFAULT_OPTIMIZE_MAX_EVALS,
    DEFAULT_OPTIMIZE_MAX_FAILURES_SHOWN,
    DEFAULT_OPTIMIZE_MAX_RELEVANT,
    DEFAULT_OPTIMIZE_RRF_K,
    DEFAULT_OPTIMIZE_SEMANTIC_TOP_K,
    DEFAULT_OPTIMIZE_STRATEGY,
    DEFAULT_RELEVANCE_MODEL,
)
from outmem.exceptions import OutmemError
from outmem.optimize.bench import Scorecard, evaluate
from outmem.optimize.blocks import RetrievalConfig, build_retriever
from outmem.optimize.dataset import QuestionBank

if TYPE_CHECKING:
    from outmem.store import WikiStore

log = logging.getLogger(__name__)


@dataclass
class OptimizeResult:
    best_config: RetrievalConfig
    best_score: float
    scorecard: Scorecard
    trace: list[tuple[dict[str, Any], float]]  # (config, score) in eval order
    notes: str  # the agent's closing rationale (advisory)
    log: list[str] = field(default_factory=list)  # diagnostics (errors/fallbacks)


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
    "one of {lexical, bm25}, one of {semantic, hyde}, and one hybrid fuse. "
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
                (keyword net + cheap-model relevance gate), "semantic"
                (vector similarity), "hyde" (generate a hypothetical answer,
                then semantic-search on it — needs a model + the index), or
                "hybrid" (Reciprocal Rank Fusion of the `fuse` legs).
            case_insensitive: case-fold the keyword search.
            max_candidates: width of the keyword net before reranking.
            rerank_model_id: model id for the rerank block.
            max_relevant: cap on pages the rerank block keeps.
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
            return f"config unavailable: {exc}"
        trace.append((cfg.to_dict(), card.score))
        seen[fingerprint] = (cfg, card)
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
    _emit_metric_context(store, k)
    with _span("optimize_retrieval", max_evals=max_evals, k=k):
        run = agent.run_sync(_initial_prompt(bank, k, max_evals))
    notes = str(run.output)

    if best["cfg"] is None:  # agent never produced a scorable config
        cfg = RetrievalConfig()
        card = evaluate(
            build_retriever(store, cfg, model=rerank_model),
            bank, k=k, max_concurrency=eval_concurrency,
        )
        return OptimizeResult(cfg, card.score, card, trace, notes, log=run_log)

    best_cfg: RetrievalConfig = best["cfg"]
    best_card: Scorecard = best["card"]
    if eval_sample is not None:  # winner chosen on a sample → re-score on full bank
        best_card = evaluate(
            build_retriever(store, best_cfg, model=rerank_model),
            bank, k=k, max_concurrency=eval_concurrency,
        )
    return OptimizeResult(best_cfg, best_card.score, best_card, trace, notes, log=run_log)


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


def _emit_metric_context(store: WikiStore, k: int) -> None:
    """Print one line so the user can sanity-check whether Hit@k is
    informative on their corpus before reading any scores.

    With N pages and cutoff ``k``, the theoretical ceiling is ``min(k,N)/N``
    — well below 1.0 for big corpora, but on a 12-page wiki any retriever
    that returns 4+ top slots already covers a third of the corpus and
    Hit@k saturates near 1.0. The scores stop distinguishing strategies.
    A loud-but-cheap warning here saves an honest "score=1.000 is too good
    to be true" diagnosis after the fact. (Default ``k=1`` already dodges
    this on tiny corpora; the warning catches overrides.)"""
    try:
        n = len(store.list_slugs())
    except Exception:
        return
    saturated = n > 0 and k / n > 0.25
    flag = "  (⚠ Hit@k saturates — k is a large fraction of the corpus)" if saturated else ""
    sys.stderr.write(f"corpus: {n} pages, k={k}{flag}\n")
    sys.stderr.flush()


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
    """Compact, human-readable label of which blocks a trial actually used,
    so the epoch line shows e.g. `hybrid[bm25+semantic]` or `rerank(haiku)`
    rather than a bare strategy name."""
    if cfg.strategy == "hybrid":
        return f"hybrid[{'+'.join(cfg.fuse)}]"
    if cfg.strategy == "rerank":
        return f"rerank({_short_model(cfg.rerank_model)})"
    if cfg.strategy == "hyde":
        return f"hyde({_short_model(cfg.hyde_model)})"
    return cfg.strategy


def _short_model(model_id: str) -> str:
    """`anthropic:claude-haiku-4-5` → `haiku`-ish: last path segment, deprefixed."""
    tail = model_id.split(":")[-1]
    return tail.replace("claude-", "")


def _format_epoch(event: EvalEvent) -> str:
    c = event.scorecard
    star = " *" if c.score >= event.best_score else ""  # this eval is (tied) best
    return (
        f"[eval {event.index}/{event.max_evals}] {_describe_config(event.config)} "
        f"score={c.score:.3f} (hit@{c.k}={c.hit_at_k:.3f} abstain={c.abstention:.3f}) "
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
