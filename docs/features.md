# Optional features

Every feature in this file is opt-in. Core outmem doesn't pull these
deps; install the relevant extra.

## Semantic index

Install: `pip install "outmem[semantic]"`. There's no on/off flag —
semantic is active for a wiki once its index exists, so build it:
`outmem reindex`. A `semantic`/`hyde`/`*+semantic` retrieval strategy or
`find_similar` then uses it; with no index they fail loud ("run `outmem
reindex`").

When the index is built, outmem maintains a local `sqlite-vec` database at
`<wiki>/.vectors.db` (tracked in git, sibling of `wiki/`) holding
paragraph-aware chunks of every wiki page and every text source under
`wiki/sources/`. Set `semantic.index: pages` (or run `outmem reindex
--pages-only`) to index curated pages only — worth doing on a
source-heavy wiki, where raw sources are near-duplicates of the pages
distilled from them and crowd those pages out of the fixed-size candidate
window. Set `semantic.embed_frontmatter: true` to put each page's
`"<title> — <tags>"` in front of every chunk, without which titles and
tags are invisible to semantic search. Both are covered in
[configuration.md](configuration.md).

A page whose frontmatter outmem can't parse is **not** indexed, and would
otherwise vanish from search unnoticed. `outmem reindex` therefore names
every such page on stderr and exits non-zero; `outmem lint` reports the
offending field. Embeddings come from PydanticAI's `Embedder` — the
default is `openai:text-embedding-3-small` (1536 dims, ~ $0.02 / M
tokens). The DB is updated atomically with every write: `write_page`,
`extend_page`, and `add_source` re-chunk the affected file, re-embed
only what changed (skipped via a content hash when the body is
unchanged), and stage the updated `.vectors.db` in the same commit as
the page. No "rebooting" needed within a session.

For external edits (Obsidian, manual `git add`, etc.), the pre-commit
hook keeps the index AND `wiki/index.md` in step with the human's commit.
outmem auto-installs it on `init`/`open` (idempotent, never clobbers your
own hook; disable with `git.auto_install_hook: false`), so you normally
don't run anything:

```bash
outmem hook install        # explicit install / --force to replace
outmem hook uninstall      # one-off removal
```

The hook calls `outmem reindex --staged`, which walks the git index
for staged wiki pages and source files, repairs any whose frontmatter
won't parse (e.g. an externally pasted title containing `: `; a no-op
on well-formed pages), reindexes them in the vector DB, regenerates
`wiki/index.md` if any wiki page changed, and stages everything so the
commit carries it all in lockstep.

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
* `bm25` — SQLite FTS5 BM25 (IDF-weighted term ranking); no model, no
  index, no extra dependency. Often beats `lexical` on jargon-heavy text.
* `rerank` — a candidate shortlist (from a source block) → an LLM yes/no
  relevance gate. `rerank(bm25)` is the default; `rerank(semantic)` is the
  high-recall pairing.
* `semantic` — vector similarity over the semantic index; recall for
  paraphrases that share no keywords (needs a built index).
* `hyde` — generate a hypothetical answer to the question, then
  semantic-search on *that* (needs a model + the index).
* `hybrid` — Reciprocal Rank Fusion of 2+ atomic legs, named by the
  `fuse` knob (default `["lexical","semantic"]`; also e.g.
  `["bm25","semantic"]` or `["semantic","hyde"]`).

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
from outmem.optimize import generate_bank, optimize_retrieval, QuestionBank

store = WikiStore.open("/srv/wiki")
bank = generate_bank(store, model="anthropic:claude-haiku-4-5")
# …or, for sensitive corpora: bank = QuestionBank.load("bank.json")
result = optimize_retrieval(store, bank, optimizer_model="anthropic:claude-sonnet-4-6")
print(result.best_config, result.best_score)   # then write it into config.yaml
```

This is the **config-space** loop: it picks among shipped, tested blocks
and writes no code. The **code-space** loop — an agent that writes *new*
blocks (e.g. a learned query-formulation block, a smarter reranker),
gated by tests + the benchmark across multiple corpora, opening a PR —
is a maintainer activity, documented in
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
auto-instrumented (LLM calls, tool calls, tokens, latencies). Embedding
calls are also wrapped via `instrument_embedding_model`, so each
`embed_query` / `embed_documents` emits a span with the model, prompt
count, and per-call token usage — you see per-call cost (not just the
rolled-up `outmem.reindex` summary).

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

**Reindex cost.** `outmem reindex` (and `WikiStore.semantic_reindex_all`)
emit an `outmem.reindex` parent span with `files`, `force`, `reindexed`,
`chunks_added`, and **`embed_tokens`** attributes — that's where the
embedding spend lands in the Logfire UI. The CLI summary line also
reports the token count (`reindex: 12 re-embedded, …, 18234 embed tokens`).

**Terminal output.** outmem calls `logfire.configure(console=False)` so
spans don't echo to your terminal (outmem prints its own progress; the
console exporter just floods stdout during optimize loops). Spans still
go to the UI. To re-enable terminal spans, call `logfire.configure(...)`
yourself with `console=...` *before* importing/triggering outmem's
setup — the first `configure` wins.

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
