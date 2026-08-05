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
    SearchHit,          # one ripgrep row (path, line_number, text, is_match)
    # The LLM relevance gate the rerank retrieval strategy uses:
    RelevantPage,       # one kept page (slug)
    judge_relevance,    # gate over (slug, excerpt) candidates → kept slugs
)
```

## Opening / creating a store

```python
from outmem import WikiStore, AgentIdentity

# Scaffold a new wiki (creates wiki/pages/, wiki/sources/, log/, .outmem/,
# CONTRIBUTORS.md, .git/).
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
# scope: "wiki" | "sources" | "log" | "all"
# "sources" spans wiki/sources/ AND wiki/sources-local/ — see docs/sources.md
result.hits         # tuple[SearchHit, ...]: .path, .line_number, .text, .is_match
result.truncated    # True if the byte-cap clipped output

# context=N returns N lines either side of each match (rg -C). Those rows
# come back with .is_match False, interleaved in file/line order.
result = store.search("cost-plus", scope="wiki", context=2)

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
    provenance=["sources/a1b2c3d4e5f6/pricing-deck-2026-Q1.md"],
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
    as_key="research/marino-2026",   # which document this is a version of
)
entry.rel_path           # "research/<sha[:12]>/paper.md"
entry.sha256             # full sha
entry.size_bytes
entry.registered_at      # datetime
entry.document_key       # "research/marino-2026" — survives a revision
entry.superseded_by      # rel_path of the version that replaced this, or None
entry.origin_path        # where it was ingested from
entry.local              # True if it lives in the untracked sources-local/ tree
entry.citation_path      # "sources/<rel>" or "sources-local/<rel>" — cite THIS

# Material you may read but not redistribute goes in the untracked tree.
# Nothing is committed; the source bytes never leave this machine, while
# pages compiled from it are tracked as usual. See docs/sources.md.
store.add_source("/path/to/licensed-handbook.md", local=True)

store.list_sources()                  # both trees; rows whose file is gone excluded
store.get_source(entry.rel_path)      # SourceEntry | None
store.read_source(entry.rel_path)     # text, capped at config.sources.max_chars

# After the agent extracts pages from a source, record the link:
store.record_ingestion(
    entry.rel_path,
    prompt="extract dosages",
    pages_touched=["amikacin-iv-dosing"],
)
```

### Supersession and staleness

`rel_path` embeds the content hash, so a revised document is always a *new*
row. `as_key` is the identity that survives the revision: re-adding under the
same key links the old row to the new one instead of leaving two unrelated
sources.

```python
v2 = store.add_source("/path/to/paper-v2.md", into_subdir="research",
                      as_key="research/marino-2026")
store.get_source(entry.rel_path).superseded_by == v2.rel_path

citations, failures = store.source_citations()   # {source rel_path: [slug, …]}
                           # the reverse provenance edge, in one walk of pages/

stale, failures = store.stale_pages()
for c in stale:
    c.slug, c.cited, c.current, c.document_key, c.current_exists

# Re-compact against the new version — this is what clears the report.
store.extend_page(c.slug, body=new_body, provenance=[f"sources/{c.current}"])
```

`stale_pages()` follows the chain to the newest version and **reports only** —
whether a page still holds is a judgement call for a human or an explicit agent
run, not a side effect of ingest.

Both return the loader's `failures` alongside the result, per the shared loader
contract: a page whose frontmatter will not parse is a page the check could not
run on, and reporting a clean wiki while silently skipping it is the failure
this contract exists to prevent. `outmem stale` exits 2 when there are any.

`extend_page(provenance=…)` **replaces** the page's source pointers; omit it and
they are untouched. Without it there is no way to update the field, so a page
reported by `stale_pages()` would keep citing the superseded version forever.

Omit `as_key` and the identity is derived from the path; `add_source` raises
`OutmemError` when that derivation is already another document's identity,
rather than guessing which of "new version" / "different document, same
filename" was meant. The error quotes both ingest origins and proposes a name
from the first segment where they diverge.

Keys name a *document*, not a file, and are normalised on both paths — case
folded, and the source extension dropped — so a `.md` document re-exported as
`.txt` keeps its identity instead of silently starting a new one:

```python
from outmem.sources import normalize_document_key

normalize_document_key("Fachinfo/Document.MD")    # "fachinfo/document"
normalize_document_key("doi/10.1001-jama-2026")   # unchanged — not a source type
```

Otherwise free-form, so an external identifier (`awmf/113-001`) works as a key.
They use `/` rather than the `:` of page slugs so a source key is never mistaken
for a page.

For wikis registered before identity existed:

```python
store.source_refs()              # list[SourceRef] — which pages each frozen
                                 # source names, resolved at ingest
store.record_source_refs(rel)    # (re)scan one source; runs at ingest

candidates, failures = store.propose_document_keys()
for cand in candidates:
    cand.document_key, cand.rows, cand.held_by, cand.citing_pages, cand.origins
    cand.is_ambiguous   # several rows derive it, OR the name is already held

store.assign_document_keys([(rel_path, key), …])   # returns the count written
```

## Renaming

```python
store.rename_page("rki:ratgeber:influenza", "clinical:influenza")
# moves the file, updates slug:, rewrites inbound [[links]] in pages + log,
# appends the old slug to aliases: — one `rename: old -> new` commit

store.resolve_slug("rki:ratgeber:influenza")   # "clinical:influenza"
store.read("rki:ratgeber:influenza")           # still works, .slug is canonical
```

Aliases resolve **file-first**: a live page always wins its own name, so a
stale alias can never shadow a real page. They are followable but not
discoverable — `list_slugs()`, the TOC and the retrieval corpora stay
canonical-only, one entry per file.

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

## Checking a wiki is sound

```python
for slug, reason in store.unreadable():
    print(slug, "->", reason)
```

`unreadable()` is the load-time audit: every page under `wiki/pages/`
that isn't cleanly addressable, as `(slug, reason)`. It catches a filename
whose derived slug fails the slug grammar (uppercase, underscores), a page
whose frontmatter won't parse, and — the silent one — a page whose declared
`slug:` disagrees with its path, which reads fine but gives the same page two
names, only one of which resolves.

A clean wiki returns `[]`, which is the assertion to put in a consumer's own
test suite. It is one directory walk, not an O(n) `read()` sweep, and it does
not require running the whole linter.

**The path addresses a page; `slug:` is a declaration checked against it.**
A missing `slug:` is derived from the path rather than being fatal, and a
mismatched one is reported here rather than raising — the page stays
available either way. `list_slugs()` skips anything `read()` would reject, so
the catalogue never advertises a page it can't open.

## Resolving slugs — delegate, don't reimplement

```python
canonical = store.resolve_slug(slug)
return canonical if store.exists(canonical) else None
```

That is the whole contract, and writing it this way means future addressing
rules arrive for free. **Don't build your own slug map.** outmem has twice
taught the store a new way to resolve a name — 0.7.0 stopped treating a
missing `slug:` as fatal and began filtering ungrammatical ones, 0.8.0 added
`aliases:` — and a consumer's own resolver falls behind silently, because the
wiki is *sound* in both cases. `unreadable()` correctly says nothing is wrong.

The usual reason people reimplement is a browsing surface: you need titles,
and once you are walking `wiki/pages/` to get them, a slug map falls out of
the walk for free. Take the titles from the TOC instead and the walk never
happens:

```python
level = store.index_tree("abx", titles=True)
level.namespaces      # [("abx:side-effects", 3), …] — drill in with prefix
level.pages           # ["abx:penicillin", …]
level.titles          # {"abx:penicillin": "Penicillin", …}
```

`titles=True` is opt-in because it costs a frontmatter parse per page, where
the rest of `index_tree` is a directory walk.

### Conformance harness

If you genuinely can't delegate — you cache a resolution map, render offline,
or run without a live store — pin your resolver against outmem's:

```python
from outmem.testing import assert_resolver_conforms, build_conformance_wiki

def test_my_resolver(tmp_path):
    store = build_conformance_wiki(tmp_path / "wiki")
    assert_resolver_conforms(lambda s: my_resolve(store, s), store)
```

Your resolver takes a name and returns the canonical slug it addresses, or
`None`. The harness feeds it every addressing case outmem knows — canonical,
namespaced, aliased, declared-slug-mismatch, derived slug, ungrammatical,
missing, reserved index — and reports every disagreement with the feature it
belongs to and why the case exists.

**The corpus is built by outmem, not supplied by you.** That is the point: your
own wiki might contain no aliased page, so testing against it would have sailed
straight through the 0.8.0 break. Because the fixture lives here, a behaviour
added in a future release becomes a new case in *your* suite — your tests go red
on upgrade without you having heard of the feature.

A consequence worth knowing: a resolver that can't be pointed at a different
wiki can't be conformance-tested. Parameterise it by store or root.

Deliberate divergence is expressed by narrowing, so it lands in your suite as a
decision rather than a silence — and use subtraction, so a feature added in 0.9
is still checked:

```python
assert_resolver_conforms(r, store, features=ADDRESSING_FEATURES - {"reserved-index"})
```

The cheap variant, if the harness is more than you want: pin the feature set
itself. Cruder — it says *something* changed without saying what — but it is one
line.

```python
from outmem.testing import ADDRESSING_FEATURES

def test_addressing_surface_is_unchanged():
    assert ADDRESSING_FEATURES == {"canonical", "namespaces", "aliases", …}
```

`outmem.testing` imports nothing beyond the store — no pytest — so it works
with any runner. `build_conformance_wiki` creates a real wiki, so it needs
`git` and `rg` on PATH like any other outmem wiki; if your CI runs the harness,
it needs them too. The fixture deliberately contains states `outmem lint`
reports (a mismatched declared slug, a filename deriving an invalid slug) —
those are the cases a reimplemented resolver gets wrong, so a corpus without
them would test only the easy half.

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
        + skill_text("search")    # the bundled `search` skill body
        + "\n\n"
        + skill_text("write")     # writeback rules (`compact:`/`extend:`/`log:`)
        + "\n\n"
        + skill_text("ingest")    # source → page workflow + record_ingestion
        # `evolution` (page history + `topic_evolution`) is also available.
    ),
)

result = await agent.run("what did we decide about pricing?")
```

A short tool plan a competent agent will follow with this palette:

```text
search_index()                            # orient on an unfamiliar wiki
└─ search_index(prefix="abx")             # drill into a namespace
   └─ search_wiki(question="…")           # then ask the question
      └─ read_page(slug, peek=True)       # outline: which part to read
         └─ read_page(slug)               # full read on the winner
            └─ write_page / extend_page / append_log  # close the loop
```

The tools — fourteen, plus `find_similar` when the semantic index is
built (fifteen):

| Tool | Required args | Purpose |
|------|---------------|---------|
| `search_wiki(question, k)` | 1 | Strategy-driven page search → ranked `[[slug]]` citations |
| `grep_wiki(pattern, scope, case_insensitive, context)` | 1 | Ripgrep over wiki / sources / log / all; `context=N` adds N lines either side |
| `read_page(slug, peek, section)` | 1 | Full file (frontmatter + body); `peek=True` → section outline; `section=` → one section |
| `list_pages()` | 0 | Every slug, one per line (flat) |
| `search_index(prefix)` | 0 | Browse the slug namespaces (the TOC) one level at a time |
| `find_backlinks(slug)` | 1 | Pages linking *to* slug |
| `page_history(slug)` | 1 | Commits touching the page |
| `topic_evolution(slugs, include_log)` | 1 | `git log -p --follow` diff stream |
| `list_sources()` | 0 | Registered source files under `wiki/sources/` |
| `read_source(rel_path)` | 1 | Full text of a registered source |
| `write_page(slug, title, body, provenance, tags)` | **3** | New page → commit `compact: <slug>` |
| `extend_page(slug, body)` | **2** | Replace body → commit `extend: <slug>` |
| `append_log(topic, content)` | **2** | Append entry → commit `log: <topic>` |
| `record_ingestion(rel_path, prompt, pages_touched)` | 1 | Note a source as ingested |
| `find_similar(text, top_k, exclude_slug)` | 1 | Vector search — only when the index is built |

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
`config.yaml` (`logfire.enabled: true`). The library does the same —
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
`False` when `logfire.enabled` is false. It's idempotent process-wide,
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
    Retriever,           # protocol for custom blocks (improve.md uses this)
    RetrievalResult,     # what a Retriever returns: ranked slugs + optional note
    Question, QuestionBank,
    generate_bank,       # provenance-labelled bank from the wiki (LLM)
    harvest_unanswerable,  # pull gap-log questions for the unanswerable half
    evaluate,            # score a retriever -> Scorecard
    Scorecard,           # the metric scalar + sub-rates + per-Q results
    QuestionResult,      # per-question row inside Scorecard.results
    optimize_retrieval,  # agent-driven config search -> OptimizeResult
    EvalEvent,           # per-eval progress event (the on_eval hook payload)
)

store = WikiStore.open("/srv/wiki")
bank = generate_bank(store, model="anthropic:claude-haiku-4-5")  # or QuestionBank.load("bank.json")
result = optimize_retrieval(store, bank, optimizer_model="anthropic:claude-sonnet-4-6")
result.best_config   # winning strategy + knobs for this corpus
result.best_score    # the metric it achieved (Hit@k blended with abstention)
result.trace         # [(config_dict, score), ...] — every config tried
result.log           # diagnostics: errors/fallbacks during the run (why + which eval)
```

`RetrievalConfig` carries `strategy` (`lexical`/`bm25`/`rerank`/`semantic`/
`hyde`/`hybrid`) plus knobs: `max_candidates`, `max_relevant`,
`semantic_top_k`, `rrf_k`, `rerank_model`, `rerank_source`, `hyde_model`,
and `fuse` — the 2+ atomic legs the `hybrid` strategy RRF-fuses (default
`("lexical","semantic")`; see [autoresearch.md](autoresearch.md) for leg combos).
`rerank_source` picks which atomic block feeds candidates to the `rerank`
LLM gate (any of `lexical`/`bm25`/`semantic`/`hyde`); default `lexical`,
but `semantic` is the recall-first pairing that lets the gate prune false
positives from real recall instead of judging a keyword net the gold page
may not even be in.

Entry-point signatures:

- `generate_bank(store, *, model, per_page=2, slugs=None, max_pages=None, include_unanswerable=True, max_concurrency=8, on_progress=None) -> QuestionBank`
- `optimize_retrieval(store, bank, *, optimizer_model, rerank_model=None, k=1, eval_concurrency=8, eval_sample=None, max_evals=12, on_eval=None) -> OptimizeResult`
- `evaluate(retriever, bank, *, k=1, max_concurrency=8, sample=None) -> Scorecard` — `.score`, `.hit_at_k`, `.abstention`, `.failures`, `.mean_latency_ms`, `.p95_latency_ms`; each entry in `.results` is a `QuestionResult` with `.latency_ms` for per-question wall-clock.

Progress prints to stderr by default: a page counter for `generate_bank`,
and one epoch line per eval for `optimize_retrieval`
(`[eval 3/12] hybrid[bm25+semantic] score=0.71 (hit@5=0.66 abstain=0.80) 4ms/search best=0.71 *`) —
the bracketed part names which blocks the trial used.
Pass `on_progress(done, total)` / `on_eval(EvalEvent)` to redirect.

The bank is plain JSON (`QuestionBank.save` / `.load`), so a team with
sensitive content can hand-author it and never send a page to a model.
