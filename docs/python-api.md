# Python API

The public surface lives at the package root:

```python
from outmem import (
    WikiStore,
    WikiStoreConfig,
    WikiPage,
    WikiFrontmatter,
    ProvenanceEntry,
    AgentIdentity,
    OutmemError,
    WritebackError,
    SlugError,
    FrontmatterError,
    GitOperationError,
    ConflictError,
    # Relevance filter (optional; see docs/features.md):
    RelevanceConfig,    # the config object a consumer passes to the adapter
    FilterOutcome,      # what on_filter receives per search
    RelevantPage,       # one kept page: slug + reason + supporting SearchHits
    SearchHit,          # a single ripgrep hit (path, line_number, text)
    relevance_filter,   # standalone core fn (non-PydanticAI consumers / tests)
)
```

## Opening / creating a store

```python
from outmem import WikiStore, AgentIdentity

# Scaffold a new wiki (creates raw/, wiki/, log/, .outmem/, CONTRIBUTORS.md, .git/).
# Also seeds wiki/AGENTS.md with starter conventions (see docs/configuration.md).
store = WikiStore.init("/srv/agent")

# Or open an existing wiki (creates missing subdirs but does not init git).
store = WikiStore.open("/srv/agent")

# Override the agent identity used for commits.
store = WikiStore.open(
    "/srv/agent",
    agent_identity=AgentIdentity(name="my-agent", email="my-agent@example.com"),
    remote="origin",
    branch="main",
)
```

## Reading

```python
page = store.read("pricing-formula")
page.slug                  # "pricing-formula"
page.title                 # "Pricing formula"
page.body                  # markdown body (no frontmatter)
page.frontmatter.provenance # list of strings or dicts
page.frontmatter.updated   # datetime | None

store.exists("pricing-formula")    # True
store.list_slugs()                 # ["acme-msa", "pricing-formula"]
```

## Searching

```python
result = store.search("cost-plus", scope="wiki")
# scope: "wiki" | "raw" | "log" | "all"
result.hits         # tuple[SearchHit, ...]; each SearchHit has .path, .line_number, .text
result.truncated    # True if the byte-cap clipped output

for hit in result.hits:
    print(f"{hit.path}:{hit.line_number}: {hit.text}")
```

## Backlinks, history, evolution

```python
store.backlinks("pricing-formula")    # ("acme-msa",) — pages that link in
store.history("pricing-formula")      # list[CommitInfo] (sha, author, date, subject, body)
store.evolution(["pricing-formula"])  # raw `git log -p --follow` diff stream (str)
```

## Writing — three paths

Every write produces exactly one commit. The commit subject grammar
(`compact:` / `extend:` / `log:`) is what makes the TARS *Retained*
metric a `git log --grep` filter (spec §9).

```python
# New page.
sha = store.write_page(
    "discounts",
    title="Discount tiers",
    body="Standard discount tiers are 5% / 10% / 15%.\n",
    provenance=["raw/pricing-deck-2026-Q1.md"],
    tags=["pricing"],
)
# Commits "compact: discounts" under the agent identity. Returns 40-char SHA.

# Edit existing page (replaces body, preserves frontmatter, bumps `updated`).
sha = store.extend_page("pricing-formula", body="Revised: cost-plus 40%.\n")
# Commits "extend: pricing-formula".

# Log entry — for findings that don't yet rise to a wiki page.
sha = store.append_log(
    topic="pricing-inconsistency",
    content="- noticed acme-msa cites cost-plus 30%, pricing-formula says 35%.\n",
)
# Commits "log: pricing-inconsistency". Appends to log/<today>.md.
```

## Sources

```python
entry = store.add_source(
    "/path/to/paper.md",
    into_subdir="research",
    rename="paper.md",
)
entry.rel_path           # "research/<sha[:12]>/paper.md"
entry.sha256             # full sha
entry.size_bytes
entry.registered_at      # datetime

store.list_sources()                  # list[SourceEntry]
store.get_source(entry.rel_path)      # SourceEntry | None
store.read_source(entry.rel_path)     # text, capped at config.sources.max_chars

# After the agent extracts pages from a source, record the link:
store.record_ingestion(
    entry.rel_path,
    prompt="extract dosages",
    pages_touched=["amikacin-iv-dosing"],
)
```

## Sync

```python
store.pull()       # git pull --rebase origin main
store.push()       # git push origin main
store.head()       # current HEAD SHA or None
```

## Steering (phase 1)

```python
# Returns CommitInfo list of non-agent commits since the last successful run.
# If no last-run marker exists, defaults to "30 days ago" (configurable).
signal = store.steering()
for commit in signal:
    print(f"{commit.author_name}: {commit.subject}")
```

## Record / read last-run marker

```python
marker = store.record_run()    # stamps .outmem/last_run.json with current HEAD + timestamp
store.last_run()                # LastRun(timestamp=..., head=...) or None
```

## Contributors

```python
contributors = store.contributors()           # parsed CONTRIBUTORS.md
contributors.lookup("bob@example.com")        # → Contributor(...) or None
contributors.lookup("bob@personal.dev")       # → same Contributor if listed as alias
```

## Error handling

Every operation that touches the filesystem can raise `OutmemError`
(or a subclass). Catch the parent class unless you need to discriminate.

```python
from outmem import OutmemError, WritebackError, FrontmatterError, SlugError

try:
    page = store.read("bad slug")
except SlugError:
    # Slug had spaces / wrong case / bad chars.
    ...

try:
    store.push()
except OutmemError as exc:
    # Network error, branch protection, etc.
    print(f"push failed: {exc}")
```

## Embedding in your own PydanticAI agent

The adapter returns plain-function tools you attach to your own
`pydantic_ai.Agent`. No PydanticAI dependency in outmem core — the
functions are vanilla Python that PydanticAI introspects at attach time.

```python
from pydantic_ai import Agent
from outmem import WikiStore
from outmem.adapters.pydantic_ai import wiki_tools, skill_text

store = WikiStore.open("/srv/agent")

agent = Agent(
    "anthropic:claude-sonnet-4-6",
    tools=wiki_tools(store),
    system_prompt=(
        "You answer with citations. "
        + skill_text("search")    # injects the bundled `search` skill body
    ),
)

result = await agent.run("what did we decide about pricing?")
```

The tools — twelve, plus `find_similar` when `semantic.enabled` (thirteen):

| Tool | Required args | Purpose |
|------|---------------|---------|
| `search_wiki(pattern, scope, case_insensitive)` | 1 | Ripgrep over wiki / raw / log / all |
| `read_page(slug)` | 1 | Full file (frontmatter + body) |
| `list_pages()` | 0 | Every slug, one per line |
| `find_backlinks(slug)` | 1 | Pages linking *to* slug |
| `page_history(slug)` | 1 | Commits touching the page |
| `topic_evolution(slugs, include_log)` | 1 | `git log -p --follow` diff stream |
| `list_sources()` | 0 | Registered source files under `wiki/sources/` |
| `read_source(rel_path)` | 1 | Full text of a registered source |
| `write_page(slug, title, body, provenance, tags)` | **3** | New page → commit `compact: <slug>` |
| `extend_page(slug, body)` | **2** | Replace body → commit `extend: <slug>` |
| `append_log(topic, content)` | **2** | Append entry → commit `log: <topic>` |
| `record_ingestion(rel_path, prompt, pages_touched)` | 1 | Note a source as ingested |
| `find_similar(text, top_k, exclude_slug)` | 1 | Vector search — only when `semantic.enabled` |

## Read-only consult — wiki as a tool in someone else's agent

When you've curated a wiki and want an *external* agentic system to
consult it (without ever modifying it), use the one-call factory:

```python
from pydantic_ai import Agent
from outmem.adapters.pydantic_ai import build_consult_wiki

consult_wiki = build_consult_wiki("/srv/curated-wiki")

my_assistant = Agent(
    "anthropic:claude-sonnet-4-6",
    tools=[consult_wiki],
    system_prompt=(
        "You're a helpful assistant. For questions about internal "
        "policies, decisions, or customer history, call `consult_wiki`."
    ),
)
result = my_assistant.run_sync("What's our pricing policy?")
```

`build_consult_wiki(path, *, model=...)` opens the wiki via
`WikiStore.open(path, read_only=True)`, builds an inner PydanticAI
agent with the read-only tool palette (`wiki_read_tools`) and a tight
system prompt ("cite by `[[slug]]`, say so explicitly if the wiki has
nothing on the topic"), and returns a single
`consult_wiki(question: str) -> str` callable. The outer agent gets a
black-box tool — no outmem-internal vocabulary leaks through the
boundary.

What "read-only" guarantees:

- Every commit-producing entry point on `WikiStore` (`write_page`,
  `extend_page`, `append_log`, `add_source`, `record_ingestion`,
  `rebuild_index`, `import_vault`) raises `OutmemError` via a single
  guard in `WikiStore._commit_paths`. The contract is defense in depth:
  even if the model somehow obtained a write tool, the commit funnel
  would still refuse.
- `pull()` is also refused — `git pull --rebase` would mutate the
  working tree. `push()` stays unguarded, since with `_commit_paths`
  refused there's nothing local to push.
- `WikiStore.open(read_only=True)` skips `_ensure_layout`,
  `_maybe_clear_stale_lock`, and runs `BacklinkCache` memo-only — the
  wiki's filesystem state (including `.outmem/`) is left exactly as
  the caller found it. The mode is safe to use against a literally
  read-only mount.
- The inner `consult_wiki` agent carries `max_tokens=16384` and the
  Anthropic prompt-caching keys (`anthropic_cache`,
  `anthropic_cache_instructions`, `anthropic_cache_tool_definitions`),
  matching the full `outmem ask` runtime. Without these, multi-page
  reads truncate against PydanticAI's 4096-token default and every
  call re-bills the system prompt + tool defs.

If you want lower-level control — your own system prompt, a different
tool subset, your own retry / wrap logic — assemble it manually:

```python
from pydantic_ai import Agent
from outmem import WikiStore
from outmem.adapters.pydantic_ai import wiki_read_tools

store = WikiStore.open("/srv/curated-wiki", read_only=True)
agent = Agent(
    "anthropic:claude-sonnet-4-6",
    tools=wiki_read_tools(store),    # 8 tools, no write paths
    system_prompt="You answer from the wiki only. Cite [[slugs]].",
)
```

Read-only mode is also useful in tests and notebooks: open a wiki
you don't want to accidentally modify and any commit attempt will
fail loudly rather than silently writing.

## Logfire from library entry points

The CLI auto-configures Pydantic Logfire from the wiki's
`config.yaml` (`logfire.project: <name>`). The library does the same —
`ask_sync` and `build_consult_wiki` both call the setup helper
internally, so library callers get instrumentation without extra
wiring.

For *custom* integrations (`wiki_tools(store)` + your own `Agent`),
call the public helper once at startup:

```python
from outmem import WikiStore, setup_logfire

store = WikiStore.open("/srv/agent")
setup_logfire(store)   # respects store.config.outmem.logfire
```

`setup_logfire(store)` returns `True` when instrumentation activated,
`False` when `logfire.project` is null. It's idempotent process-wide,
so calling from multiple entry points in the same process is safe.

## Standalone agent runtime

If you want outmem to *be* the agent (rather than embedding it into
your own), install `outmem[agent]` and use `outmem ask` or the
programmatic API:

```python
from outmem import WikiStore
from outmem.agent import ask_sync, build_agent

store = WikiStore.open("/srv/agent")

result = ask_sync(
    store,
    query="what did we decide about pricing?",
    model="anthropic:claude-sonnet-4-6",   # or None to read $OUTMEM_MODEL
)
result.response                          # the agent's text reply
result.wrote_back                        # True if the agent committed
result.commit_shas                       # tuple of new commit SHAs
result.commit_subjects                   # ('log: pricing', ...)
result.pushed                            # True if push succeeded
result.concurrent_human_commit_landed    # True if push-retry rebased over a human
```

The runtime enforces the spec §9 contract:

- **At least one agent commit per turn.** Returning without writing
  raises `WritebackError`. The TOCTOU-safe check filters
  `git log head_before..head_after --author=<agent_email>` so a
  concurrent pull can't fake-out the check.
- **Writeback must reach the remote.** Push failure triggers one
  `pull --rebase`; second failure raises `WritebackError`.
- **Concurrent commits surface as a flag**, not silent retry — spec §9
  says the agent should re-read the affected file in that case;
  v0.1 surfaces the flag so the caller can warn the user (full
  re-read is a v0.2 enhancement).

The system prompt comes from `src/outmem/agent/prompts/system.j2` plus
the user's `wiki/AGENTS.md` (see [configuration.md](configuration.md#wikiagentsmd))
— the runtime process layer + per-wiki conventions layer.

## Retrieval tuning — `outmem.optimize`

Optional (`pip install outmem[agent]`; the `semantic`/`hybrid` blocks also
need `outmem[semantic]`). Search composable retrieval blocks for the config
that scores best on *your* wiki. Design:
[features.md](features.md#retrieval-tuning) and [autoresearch.md](autoresearch.md).

```python
from outmem import WikiStore
from outmem.optimize import (
    RetrievalConfig,     # a point in the search space (strategy + knobs)
    build_retriever,     # RetrievalConfig -> a live Retriever
    Question, QuestionBank,
    generate_bank,       # provenance-labelled bank from the wiki (LLM)
    evaluate,            # score a retriever -> Scorecard
    optimize_retrieval,  # agent-driven config search -> OptimizeResult
)

store = WikiStore.open("/srv/wiki")
bank = generate_bank(store, model="anthropic:claude-haiku-4-5")  # or QuestionBank.load("bank.json")
result = optimize_retrieval(store, bank, optimizer_model="anthropic:claude-sonnet-4-6")
result.best_config   # winning strategy + knobs for this corpus
result.best_score    # the metric it achieved (Hit@k blended with abstention)
result.trace         # [(config_dict, score), ...] — every config tried
result.log           # diagnostics: errors/fallbacks during the run (why + which eval)
```

Entry-point signatures:

- `generate_bank(store, *, model, per_page=2, slugs=None, max_pages=None, include_unanswerable=True) -> QuestionBank`
- `optimize_retrieval(store, bank, *, optimizer_model, rerank_model=None, k=5, max_evals=12) -> OptimizeResult`
- `evaluate(retriever, bank, *, k=5) -> Scorecard` — `.score`, `.hit_at_k`, `.abstention`, `.failures`

The bank is plain JSON (`QuestionBank.save` / `.load`), so a team with
sensitive content can hand-author it and never send a page to a model.
