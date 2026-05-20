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

Off by default. Set `logfire.project: <name>` in `config.yaml` to
opt in. Install `pip install "outmem[logfire]"` and set
`$LOGFIRE_TOKEN` (the token determines which project the data lands
in; the config field is an opt-in marker plus self-documentation).

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
config has `logfire.project: null`, and raises `OutmemError` when the
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
