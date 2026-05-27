# Autoresearch: making outmem improve its own retrieval

outmem treats retrieval as a tunable, measurable system rather than a
fixed default. This document describes the loop that lets it improve —
what's implemented today (tune retrieval *config* to your wiki), and
what is deliberately **not yet** built (an agent that rewrites retrieval
*code*).

## Inspiration

The shape is borrowed from Andrej Karpathy's
[autoresearch](https://github.com/karpathy/autoresearch): an agent edits
a markdown spec (`program.md`), runs a short experiment (`train.py`),
grades against a single scalar (`val_bpb`, lower = better), and keeps or
discards the change — looping ~100× overnight. The human edits the
*guidance*, the agent hill-climbs the *system*, a cheap automatic metric
arbitrates.

Mapped onto outmem:

| autoresearch | outmem |
| --- | --- |
| `train.py` (the system, edited) | retrieval — the blocks + their config |
| `prepare.py` (fixed) | the wiki + a question bank + the harness |
| `program.md` (human guidance) | `improve.md` (for the code-space loop) |
| `val_bpb` (scalar metric) | retrieval score on the bank (`bench.evaluate`) |
| keep / discard | `git revert` (outmem is already commit-atomic) |

The key enabler outmem has that a generic setup doesn't: **provenance
gives relevance labels for free** (a question grounded in source `S` has
gold page = the page whose `provenance:` cites `S`), so the metric needs
no human labelling and — for the cheap retrieval metric — no model calls.

## Two loops

1. **Config-space (implemented, user-facing, safe):** compose shipped
   retrieval blocks via `RetrievalConfig`, score them on a `QuestionBank`,
   let an agent find the best config for *your* wiki. Writes no code.
2. **Code-space (not yet — see [#1](https://github.com/phiweger/outmem/issues/1)):**
   an agent rewrites retrieval *source* to invent new blocks, gated by
   tests + the benchmark, opening a PR. Maintainer-side library R&D.

Everything below the next heading is loop 1. The final section is loop 2.

---

## The config-space loop (current)

Package: `outmem.optimize` (`pip install "outmem[agent]"`; the `semantic`
block also needs `outmem[semantic]`).

### Lego blocks

Every strategy satisfies one contract (`outmem.optimize.blocks`):

```python
def retrieve(self, question: str, *, k: int) -> RetrievalResult  # ranked slugs; empty == abstain
```

*Empty == abstain* is load-bearing: returning nothing is the **correct**
answer to an unanswerable query, and it's how a block scores on that
half of the metric. Shipped blocks:

| block | how it matches | needs |
| --- | --- | --- |
| `lexical` | keyword ripgrep, pages ranked by hit frequency | nothing |
| `rerank` | wide keyword net → relevance-filter gate | a cheap model |
| `semantic` | vector cosine similarity over the index | `semantic` + index |
| `hybrid` | Reciprocal Rank Fusion of `lexical` + `semantic` | as `semantic` |

A `RetrievalConfig` names a block and its knobs; `build_retriever(store,
config)` composes it. The set is small and honest — adding `bm25` or a
smarter fusion is the *code-space* loop's job.

### Test-set generation (and the hand-author escape hatch)

A `QuestionBank` is questions with known gold page(s). Two ways to get one:

- **Generate** (`generate_bank`): a model reads each page and writes
  natural questions it answers; gold = that page's slug. This measures
  **retrieval** ("given the fact lives on page X, does search find X from
  a *reworded* question?") — not coverage, so it is **not** the circular
  trap of quizzing the wiki on its own verbatim text. Unanswerable
  questions are harvested from the gap log (`harvest_unanswerable`).
- **Hand-author**: generation is optional. The bank is plain JSON
  (`QuestionBank.save` / `.load`):

  ```json
  { "answerable":  [{ "question": "...", "gold_slugs": ["abx:penicillin"] }],
    "unanswerable": [{ "question": "...", "gold_slugs": [] }] }
  ```

  Teams with sensitive content write this by hand and **never send a page
  to a model**. Same downstream path.

### The metric — one scalar, two diagnostics

`bench.evaluate(retriever, bank, k)` returns a `Scorecard`:

```
score = mean over the bank of:
    answerable   → 1 if a gold page is in top-k   (Hit@k)
    unanswerable → 1 if the retriever returned empty (abstention)
```

`score` is the single number the optimizer maximises (the `val_bpb`
analogue). `hit_at_k` and `abstention` are reported separately so you can
see *why* it moved. We use Hit@k (not F1) because gold is usually a
single page and outmem feeds the top-k to an LLM regardless of internal
order; F1 only earns its place once multi-page (list) questions exist.

### The optimizer is an agent, not a grid sweep

`optimize_retrieval(store, bank, optimizer_model=...)` gives an agent two
tools — `run_eval(config)` and `read_page(slug)` — and asks it to
*navigate* the space: score a config, read the gold pages of failing
questions to understand *why* retrieval missed, form a hypothesis, try
the next config. It stops on plateau or budget.

It **trusts the metric, not the agent**: every evaluated config is
recorded with its score, and the function returns the best-*scoring* one
seen. A confused agent can waste budget but cannot hand back a config
worse than it measured.

### Quickstart

Progress prints to **stderr** as it runs — a live page counter for
`generate_bank`, one epoch line per eval for `optimize_retrieval`
(`[eval 3/12] rerank score=0.620 (hit@5=0.550 abstain=0.800) best=0.710 *`,
`*` marks a new best). No logging setup required.

```python
from outmem import WikiStore
from outmem.optimize import generate_bank, optimize_retrieval, QuestionBank

store = WikiStore.open("/srv/wiki")

# Build the bank (per-page calls run in parallel; progress on stderr).
bank = generate_bank(store, model="anthropic:claude-haiku-4-5", max_pages=50)
# …or, for sensitive content, hand-author the JSON and load it:
# bank = QuestionBank.load("bank.json")

result = optimize_retrieval(
    store,
    bank,
    optimizer_model="anthropic:claude-sonnet-4-6",
    eval_sample=30,       # score each config on 30 questions while tuning
    eval_concurrency=8,   # 8 retrievals in flight per eval
)
print(result.best_config, result.best_score)   # winning config + its full-bank score
for cfg, score in result.trace:                 # every config tried (sampled scores)
    print(score, cfg)
```

### Cost, scale & logging

Every step is LLM calls, and `rerank`/`hybrid` are the multiplier:

| step | model calls |
| --- | --- |
| `lexical` eval | 0 (pure ripgrep) |
| `semantic` eval | embeddings only |
| `rerank` / `hybrid` eval | **one filter call per bank question** |
| the optimizer agent | one per reasoning turn (propose / diagnose) |

So a single `rerank` eval over a 120-question bank is ~120 small-model
calls; the optimizer trying it a few times reaches the hundreds. Three
knobs bound it:

- **`eval_sample=N`** — score each config on a fixed, seeded subset of N
  answerable questions while tuning (the winner is re-scored on the full
  bank, so `best_score` stays honest). The biggest lever.
- **`eval_concurrency`** (and `generate_bank`'s `max_concurrency`) — run
  the per-question / per-page calls in parallel (default 8). Cuts wall
  time, not total cost.
- **bank size** — `generate_bank(..., max_pages=…, per_page=…, slugs=[…])`;
  `max_evals` caps the optimizer's turns.

**Logging.** Progress and epochs write to stderr directly, so you don't
need to configure logging to see them. If you *do* turn logging on, keep
the HTTP client quiet — otherwise its one-line-per-request output buries
everything:

```python
import logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
```

---

## What this is NOT (yet): the self-modifying loop

Loop 1 only ever *picks among shipped, tested blocks*. It cannot invent
a strategy outmem doesn't already have. The **code-space** loop — the
true autoresearch analogue, where an agent edits
`src/outmem/optimize/blocks.py` to add e.g. a BM25 block, runs the tests
+ benchmark, keeps the change only if the score improves, and opens a PR
— is **documented but not implemented**:

- the brief: [`improve.md`](../improve.md) (the `program.md` analogue —
  objective, the `Retriever` contract, the allowed edit surface, the
  keep/discard gate, safety rules);
- a skeleton CI job: [`.github/workflows/autoresearch.yml`](../.github/workflows/autoresearch.yml);
- tracked in [#1](https://github.com/phiweger/outmem/issues/1).

Why it's separate: a new block ships to *everyone*, so it's library R&D
validated across **multiple corpora** (a winner on one wiki can lose on
another), not per-deployment tuning. Mechanically it's one bounded agent
job (propose → `ruff && mypy && pytest` + benchmark → keep/revert,
time-boxed), output a PR, human-gated merge, run in the ephemeral CI VM,
and **treat wiki content as data, not instructions** (a page could carry
prompt-injection into the benchmark). See `improve.md` and #1 for the
full design.

## See also

- [features.md → Retrieval tuning](features.md#retrieval-tuning) — the user-facing summary.
- [search.md](search.md) — the retrieval *workflow* and the three tiers the blocks wrap.
