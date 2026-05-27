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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from outmem.exceptions import OutmemError
from outmem.optimize.bench import Scorecard, evaluate
from outmem.optimize.blocks import RetrievalConfig, build_retriever
from outmem.optimize.dataset import QuestionBank
from outmem.relevance import DEFAULT_RELEVANCE_MODEL

if TYPE_CHECKING:
    from outmem.store import WikiStore


@dataclass
class OptimizeResult:
    best_config: RetrievalConfig
    best_score: float
    scorecard: Scorecard
    trace: list[tuple[dict[str, Any], float]]  # (config, score) in eval order
    notes: str  # the agent's closing rationale (advisory)


_OPTIMIZER_SYSTEM_PROMPT = (
    "You are tuning a retrieval pipeline for a specific wiki. You cannot "
    "edit code; you choose among composable blocks via their config. Your "
    "job: find the config that MAXIMISES the benchmark score.\n\n"
    "Work empirically and frugally: evaluate a config with `run_eval`, then "
    "READ the failing questions' gold pages with `read_page` to understand "
    "WHY retrieval missed (wrong keywords? paraphrase the lexical block "
    "can't match? a reranker discarding the right page?). Form a hypothesis, "
    "try the next config, keep what the score rewards. Don't brute-force the "
    "grid — move deliberately. Stop when the score plateaus or your eval "
    "budget is spent, then summarise what worked and why."
)

_MODEL_SETTINGS: dict[str, Any] = {
    "max_tokens": 8192,
    "anthropic_cache": True,
    "anthropic_cache_instructions": True,
}


def optimize_retrieval(
    store: WikiStore,
    bank: QuestionBank,
    *,
    optimizer_model: Any,
    rerank_model: Any = None,
    k: int = 5,
    max_evals: int = 12,
    max_failures_shown: int = 6,
) -> OptimizeResult:
    """Let ``optimizer_model`` search the config space over ``bank``.

    ``rerank_model`` overrides the rerank block's model object (pass a
    cheap model / a ``FunctionModel`` in tests); ``None`` uses each
    config's ``rerank_model`` string. ``max_evals`` soft-caps how many
    configs the agent may score.
    """
    from pydantic_ai import Agent

    trace: list[tuple[dict[str, Any], float]] = []
    best: dict[str, Any] = {"score": -1.0, "cfg": None, "card": None}

    def run_eval(
        strategy: str = "lexical",
        case_insensitive: bool = True,
        max_candidates: int = 30,
        rerank_model_id: str = DEFAULT_RELEVANCE_MODEL,
        max_relevant: int = 8,
        semantic_top_k: int = 8,
        rrf_k: int = 60,
    ) -> str:
        """Score one retrieval config on the benchmark and report the
        result plus a sample of failing questions.

        Args:
            strategy: "lexical" (keyword only), "rerank" (keyword net +
                cheap-model relevance gate), "semantic" (vector
                similarity), or "hybrid" (RRF of lexical + semantic).
            case_insensitive: case-fold the keyword search.
            max_candidates: width of the keyword net before reranking.
            rerank_model_id: model id for the rerank block.
            max_relevant: cap on pages the rerank block keeps.
            semantic_top_k: neighbours for the semantic / hybrid blocks.
            rrf_k: Reciprocal Rank Fusion constant for the hybrid block.
        """
        if len(trace) >= max_evals:
            return (
                f"Eval budget exhausted ({max_evals}). Stop evaluating and "
                "summarise the best config you found."
            )
        cfg = RetrievalConfig.from_dict(
            {
                "strategy": strategy,
                "case_insensitive": case_insensitive,
                "max_candidates": max_candidates,
                "rerank_model": rerank_model_id,
                "max_relevant": max_relevant,
                "semantic_top_k": semantic_top_k,
                "rrf_k": rrf_k,
            }
        )
        try:
            retriever = build_retriever(store, cfg, model=rerank_model)
            card = evaluate(retriever, bank, k=k)
        except OutmemError as exc:
            return f"config unavailable: {exc}"
        trace.append((cfg.to_dict(), card.score))
        if card.score > best["score"]:
            best.update(score=card.score, cfg=cfg, card=card)
        return _format_card(cfg, card, remaining=max_evals - len(trace),
                            max_failures_shown=max_failures_shown)

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
    run = agent.run_sync(_initial_prompt(bank, k, max_evals))
    notes = str(run.output)

    if best["cfg"] is None:  # agent never produced a scorable config
        cfg = RetrievalConfig()
        card = evaluate(build_retriever(store, cfg, model=rerank_model), bank, k=k)
        return OptimizeResult(cfg, card.score, card, trace, notes)
    return OptimizeResult(best["cfg"], best["score"], best["card"], trace, notes)


def _initial_prompt(bank: QuestionBank, k: int, max_evals: int) -> str:
    return (
        f"Wiki bank: {len(bank.answerable)} answerable + "
        f"{len(bank.unanswerable)} unanswerable questions. Metric (maximise, "
        f"0..1): mean of [answerable: gold page in top-{k}] and "
        f"[unanswerable: retriever returned empty]. You have up to "
        f"{max_evals} `run_eval` calls. Start with the lexical baseline, "
        f"diagnose its failures by reading gold pages, then improve."
    )


def _format_card(
    cfg: RetrievalConfig, card: Scorecard, *, remaining: int, max_failures_shown: int
) -> str:
    lines = [
        f"score={card.score:.3f}  hit@{card.k}={card.hit_at_k:.3f}  "
        f"abstain={card.abstention:.3f}  (evals left: {remaining})",
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
        lines.append("no failures.")
    return "\n".join(lines)
