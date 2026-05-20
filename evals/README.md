# outmem evals

End-to-end behavioural evaluations. Different shape than `tests/`:

- **`tests/`** — deterministic unit / integration tests using
  `FunctionModel` (no real LLM). Fast, free, every commit.
- **`evals/`** — call **real** LLMs against realistic wiki fixtures
  and grade the agent's behaviour with both deterministic trace
  assertions (tool calls + commit subjects) and an LLM judge.
  Costs cents per case; gate behind nightly / release-tag CI, not
  per-PR.

## Install

```bash
pip install -e ".[agent,semantic,evals]"
```

Plus an API key for whichever provider you point the agent and judge
at (Anthropic by default — `ANTHROPIC_API_KEY` in your `.env` or the
shell env).

Semantic cases (`duplicate-trap`) use a deterministic in-process
**bag-of-words stub embedder** by default — no `OPENAI_API_KEY` needed
and no per-eval embedding cost. The stub
(`outmem.semantic.testing.BagOfWordsEmbeddingModel`) hashes each token
into a fixed-size bucket vector and L2-normalises, which is plenty to
verify `find_similar` ranking behaviour. To exercise a real embedder
end-to-end instead, set `OUTMEM_EVAL_REAL_EMBEDDER=1` — the fixture's
configured `embedding_model` will then be honoured as-is.

## Configure

Tool-level defaults live in the **repo-root** `config.yaml` (sibling
of `pyproject.toml`, *not* a wiki's own `config.yaml`):

```yaml
evals:
  # agent_model: anthropic:claude-sonnet-4-6   # optional fallback
  judge_model: anthropic:claude-sonnet-4-6
```

Resolution, highest priority first:
1. CLI flag (`--model`, `--judge-model`)
2. Env var (`OUTMEM_MODEL` for the agent)
3. Top-level `config.yaml`
4. Built-in default

## Run

```bash
# All 8 cases, trace + LLM judge (default judge: anthropic:claude-sonnet-4-6).
python -m evals.run

# Trace-only — still calls the agent's LLM, but skips the judge step
# (deterministic assertions still run).
python -m evals.run --no-judge

# Single case (or repeat --case for a subset).
python -m evals.run --case pricing-lookup
python -m evals.run --case duplicate-trap --case approval-fallback

# Use a specific agent model.
OUTMEM_MODEL=anthropic:claude-haiku-4-5 python -m evals.run

# Override the judge model.
python -m evals.run --judge-model anthropic:claude-sonnet-4-6

# Machine-readable report.
python -m evals.run --json out/evals.json

# Suppress the live trace (per-case header, [tool] calls, ✓/✗ per
# assertion). The final summary report is unaffected.
python -m evals.run --quiet
```

Exit code = number of failing cases (0 on full pass).

## Live trace

By default each case streams to stderr as it runs:

```
== pricing-lookup
   Tier-1 lookup: agent should search the wiki, read the page, …
   query: 'What is our standard pricing formula and where does it come from?'
    [tool] search_wiki pattern='pricing' scope='wiki' case_insensitive=True
    [tool] read_page slug='pricing-formula'
    [tool] extend_page slug='pricing-formula' body=(742 chars)
    ✓ trace: tool 'search_wiki' called with {'pattern__contains': 'pricing'}
    ✓ trace: tool 'read_page' called with {'slug': 'pricing-formula'}
    ✓ trace: commit matching /^(extend|log):/
    … judging: answer states the pricing formula is cost-plus 35%
    ✓ judge: answer states the pricing formula is cost-plus 35%
    … judging: answer attributes the formula to the Q1 2026 pricing deck
    ✓ judge: answer attributes the formula to the Q1 2026 pricing deck
-- PASS pricing-lookup (4.2s, 3 tool calls, 1 commit(s))
```

`--quiet` falls back to the at-end-only summary.

## Costs

Per case: ~3-10 agent turns × ~$0.01-0.05 + 2-4 judge calls × ~$0.005.
Full 8-case suite: **~$0.30-1.00** with Opus 4.7 (agent) + Sonnet 4.6
(judge). Use `--no-judge` and Haiku to cut to ~$0.05.

## Subagent end-to-end eval

`evals/subagent_e2e.py` is a standalone runner (not pytest — evals
have side effects + costs, run intentionally). Builds an outer
PydanticAI agent whose only tool is `consult_wiki` (which wraps
`outmem.agent.ask_sync`), seeds a tmp wiki, runs two scenarios:

1. *In-scope question* the wiki can answer — verifies outer delegates,
   subagent commits, response is grounded, no outmem-internal vocabulary
   leaks through the tool boundary.
2. *No-answer question* — verifies the subagent still commits
   (typically an `append_log` per mandatory-writeback) and the response
   signals absence rather than hallucinating.

Run:

```bash
ANTHROPIC_API_KEY=... python -m evals.subagent_e2e
```

Exits 0 on full pass, 1 on any failure, 2 if no API key. Tmp wiki
is cleaned up automatically. ~$0.10-0.20 per run on sonnet-4-6.

## The 8 cases

| Case | Wiki | What it pins down |
|---|---|---|
| `pricing-lookup` | `pricing-cost-plus` | CONVERGENCE: Tier-1 search → read → cite. |
| `pricing-history` | `temporal-evolution` | EXPANSION: agent reaches for `topic_evolution`. |
| `no-match-falls-back-to-log` | `no-match-query` | No relevant material → `append_log`, no fabrication. |
| `raw-contradicts-wiki` | `raw-contradicts-wiki` | wiki/raw disagreement surfaced, not averaged. |
| `duplicate-trap` | `duplicate-trap` *(semantic)* | `find_similar` catches a paraphrased duplicate. |
| `approval-fallback` | `approval-fallback` *(approval)* | HITL denial → log fallback satisfies writeback. |
| `stale-wikilink` | `stale-wikilink` | Dangling `[[discounts]]` link flagged, not invented. |
| `multi-author-divergence` | `multi-author-divergence` | Both authors named, divergence surfaced. |

Each fixture is checked in under `evals/fixtures/wikis/<name>/`. The
harness copies the fixture into a tmp dir, optionally replays a
`SEED.md` to seed commit history with realistic authorship, flips
per-case feature flags (`semantic`, `approval`), and runs `ask_sync`.

## Adding a case

1. Build a fixture directory under `evals/fixtures/wikis/<name>/`
   with at minimum a `config.yaml` (or copy the baseline). Drop in
   the `wiki/`, `raw/`, `log/`, `wiki/sources/` material your case
   needs. Optionally add a `SEED.md` with `## author|email|subject`
   stanzas + path lists to script the commit history.

2. Add a module under `evals/cases/` with a `case_*` function decorated
   with `@eval_case(...)`:

   ```python
   from evals import EvalRun, eval_case

   @eval_case(
       wiki="<your-fixture-dir>",
       query="...",
       semantic=False,           # flip if you need find_similar
       approval=False,           # flip if you want HITL gating
       reviewer_verdicts={...},  # required when approval=True
   )
   def case_my_thing(r: EvalRun) -> None:
       r.expect_tool_called("read_page", slug="...")
       r.expect_commit(subject_matches=r"^extend:")
       r.judge("answer mentions ...")
   ```

3. Register it in `evals/cases/__init__.py` by importing the module so
   the decorator side-effects run.

## Trace assertions cheat sheet

```python
r.expect_tool_called("search_wiki", pattern__contains="pricing")
r.expect_tool_called("read_page", slug="pricing-formula")
r.expect_tool_called("topic_evolution", slugs=lambda v: "pricing-formula" in v)
r.expect_no_tool_called("read_source")        # negative
r.expect_commit(subject_matches=r"^(extend|log):")
r.expect_commit(subject_contains="pricing-formula")
```

`arg_filters` accept:
- exact equality
- `<key>__contains=...` for substring match on string args
- a callable predicate (`lambda v: ...`) for anything else

## LLM judge

Each `r.judge("...")` call ships the criterion + the agent's response
to the judge agent (Sonnet 4.6 by default) and expects a structured
`JudgeVerdict(passed: bool, reasoning: str)` back. Failures surface
the reasoning in the report.

`--no-judge` skips the judge calls; trace assertions still run.
