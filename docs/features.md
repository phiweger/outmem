# Optional features

Every feature in this file is opt-in. Core outmem doesn't pull these
deps; install the relevant extra.

## Semantic index

Install: `pip install "outmem[semantic]"`. Off by default — flip
`semantic.enabled: true` in `config.yaml`. Everything below assumes
both.

When semantic is on, outmem maintains a local `sqlite-vec` database at
`<wiki>/.vectors.db` (tracked in git, sibling of `wiki/`) holding
paragraph-aware chunks of every wiki page and every text source under
`wiki/sources/`. Embeddings come from PydanticAI's `Embedder` — the
default is `openai:text-embedding-3-small` (1536 dims, ~ $0.02 / M
tokens). The DB is updated atomically with every write: `write_page`,
`extend_page`, and `add_source` re-chunk the affected file, re-embed
only what changed (skipped via a content hash when the body is
unchanged), and stage the updated `.vectors.db` in the same commit as
the page. No "rebooting" needed within a session.

For external edits (Obsidian, manual `git add`, etc.), install the
pre-commit hook so the index AND `wiki/index.md` keep step with the
human's commit:

```bash
outmem hook install        # → .git/hooks/pre-commit
outmem hook uninstall      # remove
```

The hook calls `outmem reindex --staged`, which walks the git index
for staged wiki pages and source files, reindexes them in the vector
DB, regenerates `wiki/index.md` if any wiki page changed, and stages
both so the commit carries everything in lockstep.

Tools the agent gains when semantic is on (added to its PydanticAI
palette automatically):

* `find_similar(text, top_k, exclude_slug)` — cosine-similarity
  lookup over every chunk in the index. The system prompt nudges the
  agent to call it before `write_page` so paraphrased duplicates get
  caught.

CLI:

```bash
outmem similar "cost-plus pricing"             # free-form query
outmem similar --slug pricing-formula           # use page body, exclude itself
outmem reindex                                  # full re-walk; skip if hash unchanged
outmem reindex --force                          # rebuild from scratch
outmem reindex --path wiki/pages/foo.md         # one file
```

Switching `embedding_model` invalidates the existing DB (sqlite-vec
bakes the dim into the virtual table). `VectorStore.open` detects the
mismatch and surfaces a clear error pointing at `outmem reindex --force`.

## Relevance filter

Install: `pip install "outmem[agent]"` (the triage model needs
PydanticAI). Off by default — flip `relevance.enabled: true` in
`config.yaml`.

For how this tier fits the end-to-end retrieval workflow (and when to
reach for it vs plain lexical or semantic), see
[search.md](search.md#three-retrieval-tiers).

A cheap-model **gate** between lexical `search_wiki` and the expensive
outer agent. Today `search_wiki` returns raw ripgrep lines, byte-capped
at 8 KiB, and the expensive model triages them in-context. With the
filter on, the swapped `search_wiki` casts a *wider* ripgrep net (64
KiB), hands the candidate pages to a small model (default
`anthropic:claude-haiku-4-5`), and keeps only the ones the model judges
relevant — each with a one-line "why". Triage moves off the expensive
model and out of its context; the candidate net can be wider than what
the outer agent should ever see.

It is a **filter, not a ranker**: per candidate the model answers "is
this relevant — yes/no". There is no score and no re-ordering. The
load-bearing invariant: **no model ever emits wiki content here.** The
filter model consumes deterministic file reads (the candidate
excerpts) and emits only decisions `{slug, reason}`; the supporting
lines shown are verbatim ripgrep hits, and the full page text the agent
reads still comes from `read_page` reading disk. The one-line `reason`
is the only model-generated string in the path.

Drop-in: the agent's mental model is unchanged ("call `search_wiki`,
then `read_page(slug)`"). Only wiki-scope searches are filtered;
`scope="raw"`/`"log"`/`"all"` stay plain path-shaped searches.

```yaml
relevance:
  enabled: true
  model: anthropic:claude-haiku-4-5
  max_relevant: 8               # cap on kept pages
  max_candidates: 20            # distinct slugs sent to the filter
  context: page                 # "page" (cap below) | "lines"
  context_chars_per_page: 2000  # per-page text the filter sees
```

Agent-facing output becomes:

```
abx:penicillin  why: IV penicillin dosing for endocarditis
  abx:penicillin:14:IV penicillin G 18-24 MU/day in divided doses
abx:ceftriaxone  why: listed as a beta-lactam alternative
  abx:ceftriaxone:9:ceftriaxone 2g IV q24h
```

**Reliability.** Any filter error/timeout/malformed output falls back to
the candidate hits in lexical order, **trimmed back to the normal 8 KiB
agent budget** (`fell_back=True`) — a filter failure never makes
retrieval worse than today, and never floods the expensive context with
the wide net. An empty result (nothing relevant) is a valid answer, not
a failure — a weak keyword isn't laundered into a false positive.

**Embedding in your own agent / consumer hook.** Pass an explicit
`RelevanceConfig` to the adapter factories — this is also how you get
the `on_filter` observability hook (outmem stays ignorant of your
tracing; it just hands you each `FilterOutcome`):

```python
from outmem import WikiStore, RelevanceConfig, FilterOutcome
from outmem.adapters.pydantic_ai import wiki_read_tools

def record(outcome: FilterOutcome) -> None:
    my_trace.append({
        "query": outcome.query,
        "candidates": outcome.candidates_considered,
        "model": outcome.model,          # records that Haiku did the triage
        "fell_back": outcome.fell_back,
        "kept": [(p.slug, p.reason) for p in outcome.kept],
    })

store = WikiStore.open("/srv/wiki")
cfg = RelevanceConfig(model="anthropic:claude-haiku-4-5", on_filter=record)
tools = wiki_read_tools(store, relevance=cfg)   # search_wiki is now filtered
```

`wiki_read_tools(store)` / `wiki_tools(store)` with no `relevance=` and
the config flag off is byte-for-byte today's behaviour. The standalone
`relevance_filter(store, query=..., model=...) -> FilterOutcome` is
public for non-PydanticAI consumers and tests.

This buys **precision + context savings, not semantic recall** — recall
stays bounded by ripgrep (keyword). If ripgrep returns nothing the
filter has nothing to keep and correctly returns empty; that empty
signal, logged via `on_filter`, is useful evidence for whether a
semantic tier (the `semantic` index above) would earn its keep.

## Retrieval tuning

Install: `pip install "outmem[agent]"` (the optimizer + bank generator
need a model; the `semantic` block additionally needs
`outmem[semantic]`).

Which retrieval strategy is best — keyword, keyword+rerank, semantic,
hybrid — is **corpus-dependent**; there's no universal winner. So
instead of shipping one default and hoping, outmem exposes retrieval as
composable **blocks** plus a script that searches the space for the
config that's best on *your* wiki.

**The blocks** (`outmem.optimize.blocks`) share one contract —
`retrieve(question, k) -> ranked slugs`, where an empty result means
"nothing relevant" (a deliberate abstention):

* `lexical` — keyword ripgrep, pages ranked by hit frequency (no model).
* `rerank` — wide keyword net → the relevance filter as a gate.
* `semantic` — vector similarity over the semantic index; recall for
  paraphrases that share no keywords (needs `semantic.enabled` + a
  built index).

**The benchmark.** A `QuestionBank` is questions with known gold
page(s). Generate one from the wiki — a model writes natural questions
per page, gold = that page (this measures *retrieval*: can search find
page X from a reworded question?) — or, if your content is sensitive,
**hand-author the JSON and never send a page to a model**; the bank is
just `{"answerable": [...], "unanswerable": [...]}`. The metric
(`bench.evaluate`) is one scalar to hill-climb: the mean of `Hit@k` on
answerable questions and *abstention* (returned empty) on unanswerable
ones — plus the two sub-rates for diagnosis. No F1 until you add
multi-page (list) questions.

**The optimizer is an agent, not a grid sweep.** It runs an eval, reads
the gold pages of failing questions to see *why* retrieval missed, forms
a hypothesis, and picks the next config to try. It returns the
best-*scoring* config it measured — the metric decides, not the agent's
self-report. Progress prints to stderr as epoch lines. Since a `rerank`
eval costs one model call per bank question, bound it with `eval_sample`
(score on a seeded subset) and `eval_concurrency` — see
[autoresearch.md](autoresearch.md#cost-scale--logging).

```python
from outmem import WikiStore
from outmem.optimize import generate_bank, optimize_retrieval

store = WikiStore.open("/srv/wiki")
bank = generate_bank(store, model="anthropic:claude-haiku-4-5")
# …or, for sensitive corpora: bank = QuestionBank.load("bank.json")
result = optimize_retrieval(store, bank, optimizer_model="anthropic:claude-sonnet-4-6")
print(result.best_config, result.best_score)   # then write it into config.yaml
```

This is the **config-space** loop: it picks among shipped, tested blocks
and writes no code. The **code-space** loop — an agent that writes *new*
blocks (BM25, hybrid fusion), gated by tests + the benchmark across
multiple corpora, opening a PR — is a maintainer activity, documented in
[`improve.md`](../improve.md) with a stub workflow at
`.github/workflows/autoresearch.yml`.

Full design (and the current-vs-future split) is in
[autoresearch.md](autoresearch.md).

## Write approval (HITL)

Off by default. When `approval.required_for_writes: true` in `config.yaml`,
the agent's `write_page` and `extend_page` tool calls are **deferred** —
the underlying git commit only happens after a human reviewer returns
a verdict. `append_log` and the read-only tools are not gated, so the
agent can still satisfy mandatory writeback (spec §9) by logging an
observation after a denial.

Under the hood we use PydanticAI's native deferred-tools primitives
(`FunctionToolset(requires_approval=True)` →
`DeferredToolRequests` → `DeferredToolResults`). `outmem ask` and
`outmem ingest` wire a CLI reviewer automatically when the flag is on.

**The three verdicts**

| Reviewer choice | What happens |
|---|---|
| `a` (approve) | Tool runs with the model's proposed args; commit lands. |
| `d` (deny) | A `ToolDenied` is returned to the model; the agent typically falls back to `append_log` to satisfy writeback. |
| `e` (edit) | `$VISUAL` / `$EDITOR` opens on the proposed body; on save, the tool runs with the **edited** body — no re-prompt round-trip. |

**Programmatic use**

```python
from outmem.agent import ask_sync, CliReviewer, RecordingReviewer
from pydantic_ai.tools import ToolApproved, ToolDenied

# Interactive (CLI):
ask_sync(store, query="…", reviewer=CliReviewer())

# Custom: e.g. a web dashboard reviewer
class DashboardReviewer:
    def review(self, call):
        # show call.tool_name + call.args_as_dict() in the UI
        # return True / False / ToolApproved(override_args=...) / ToolDenied(...)
        ...

ask_sync(store, query="…", reviewer=DashboardReviewer())

# Tests: pre-program the verdicts
reviewer = RecordingReviewer({
    "write_page": [ToolApproved(override_args={"body": "corrected.\n"})],
    "extend_page": [ToolDenied(message="stale source")],
})
```

**CI / non-interactive contexts**

When the flag is on but stdin is not a tty, `outmem ask` aborts with
a clear error before the agent starts — silent autocommit in batch
contexts is a footgun the gate is specifically there to prevent.
Either disable the flag for CI runs, or wire a custom `Reviewer`.

## Logfire instrumentation

Off by default. Set `logfire.enabled: true` in `config.yaml` to opt in.
Install `pip install "outmem[logfire]"` and set `$LOGFIRE_TOKEN` — the
token alone determines which project the data lands in.

Spans are tagged `service_name=outmem` so they're filterable when
other services publish to the same project. PydanticAI is
auto-instrumented (LLM calls, tool calls, tokens, latencies).

The CLI's `outmem ask`, the library `outmem.agent.ask_sync(store, ...)`,
and the read-only `build_consult_wiki(path)` factory all auto-call the
setup once per process when the config opts in — no manual wiring
needed. For custom integrations that attach `wiki_tools(store)` to
your own `pydantic_ai.Agent`, call the public helper once at startup:

```python
from outmem import WikiStore, setup_logfire

store = WikiStore.open("/srv/agent")
setup_logfire(store)   # honours store.config.outmem.logfire
```

`setup_logfire` is idempotent process-wide (later calls are no-ops),
returns `True` when instrumentation was activated and `False` when the
config has `logfire.enabled: false`, and raises `OutmemError` when the
config opts in but the `logfire` package isn't installed.

## Dashboard

A read-only FastAPI app. Editing happens through Obsidian against a
local git clone, never through the dashboard (spec §5).

Standalone:

```bash
outmem dashboard --port 8765
```

Mounted into your own FastAPI app (so you can add auth):

```python
from fastapi import FastAPI, Depends
from outmem import WikiStore
from outmem.dashboard import router_for

app = FastAPI()
store = WikiStore.open("/srv/agent")

# Mount under any prefix; the router carries the routes:
#   /wiki                       page index
#   /wiki/{slug:path}           rendered page + backlinks + provenance
#   /wiki/{slug:path}/history   git log timeline
#
# Namespaced slugs use ``/`` in URLs (``[[abx:penicillin]]`` →
# ``/wiki/abx/penicillin``); the router maps ``/`` back to ``:`` for
# the store lookup.
app.include_router(
    router_for(store, pull_on_request=False, base_path="/wiki"),
    prefix="/memory",
    dependencies=[Depends(your_auth)],
)
```

The render pipeline rewrites `[[wikilink]]` into markdown
`[label](/wiki/slug)` *before* feeding the body to `markdown-it-py`
with `html=False`. Defence in depth: raw HTML in wiki body is escaped,
not rendered.

## Bundled skill bodies

For internal completeness: the runtime's system prompt is composed of
the process layer (`src/outmem/agent/prompts/system.j2`), the user's
`wiki/AGENTS.md`, and three skill bodies that live under
`src/outmem/skills/notes/{search,write,evolution}/SKILL.md`. The
skill bodies are rendered verbatim into the prompt at runtime via
:func:`outmem.skills.bundled_registry`; you don't have to do anything
to "install" them.

If you're embedding outmem's tools in your own PydanticAI agent and
want the same skill text in your prompt, splice them in via
:func:`outmem.adapters.pydantic_ai.skill_text`:

```python
from outmem.adapters.pydantic_ai import skill_text

system_prompt = "You answer with citations.\n\n" + skill_text("search")
```
