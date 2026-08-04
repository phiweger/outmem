# Configuration

Four files configure outmem at startup:

* **`wiki/AGENTS.md`** — the **user-editable wiki-conventions doc**.
  Loaded into the agent's system prompt every run. The place to put
  domain framing ("what this wiki is for"), page structure templates,
  source-handling preferences, what-rises-to-a-page-vs-log heuristics.
  Seeded by `outmem init` with sparse placeholders. You and the agent
  co-evolve it. See [wiki/AGENTS.md](#wikiagentsmd) below.
* **`config.yaml`** at the **wiki root** — machine-readable settings
  (model, agent identity, git resilience, remote, optional features).
  Tracked in git so a team shares the same defaults. `outmem init`
  seeds a starter.
* **`config.yaml`** at the **outmem repo root** (optional, per-user) —
  defaults that `outmem init` seeds into every NEW wiki. Use this if
  you want `outmem init` to write e.g. `model: anthropic:claude-haiku-…`
  to new wikis without editing each one. The only field read today is
  `model:`. Any wiki you've already initialised is untouched — this
  only affects new scaffolding.
* **`.env`** — provider API keys (`ANTHROPIC_API_KEY`, `LOGFIRE_TOKEN`,
  …). Loaded via `python-dotenv` in two stages, neither of which
  overrides existing env vars:
  1. *Project-root*: CWD-upward search via `find_dotenv` — finds a
     `.env` next to wherever you invoked `outmem` from. Use this for
     per-project overrides.
  2. *Outmem repo root*: a `.env` sitting alongside outmem's own
     `pyproject.toml` (the cloned source repo). Loaded regardless of
     CWD, so you can keep one `.env` next to your outmem clone and
     have it found from anywhere. The wiki itself stays content-only —
     `.env` is never read from the wiki root.

## Resolution order (highest priority first)

1. Explicit CLI flag (`--model`, `--root`) or constructor kwarg
2. Environment variable (including anything `.env` loaded)
3. `config.yaml`
4. Built-in defaults

## wiki/AGENTS.md

This is the wiki owner's customization layer — read by every agent
turn between the runtime invariants (phases, tool tiers, response
style) and the user's actual query. It is the answer to the OG
pattern's question "what makes the LLM a *disciplined* wiki maintainer
for *this specific wiki* rather than a generic chatbot?".

`outmem init` writes a starter with four sections:

- **What this wiki is for** — one or two sentences of domain framing.
- **Page conventions** — page-structure templates if you have them.
- **What goes where** — heuristics for write_page vs extend_page vs
  append_log.
- **Anything else** — free-form notes.

The starter content is placeholder comments; populate as you discover
what your wiki needs. When you notice the agent making the same
mistake twice, write the rule in AGENTS.md so it stops making it a
third time.

Existing wikis without an `AGENTS.md` keep working — the conventions
section is simply absent from the prompt, and the runtime invariants
carry the agent. There's no migration path; create the file when you
want it.

## Sample `config.yaml`

```yaml
# config.yaml — wiki-level config for the agent runtime
model: anthropic:claude-sonnet-4-6

agent:
  name: outmem agent
  email: agent@host

remote:
  name: origin
  branch: main

git:
  remove_stale_lock: true       # clean .git/index.lock left by killed prior runs
  stale_lock_seconds: 60         # lock counts as stale after this many seconds
  retry_on_lock: true            # retry git ops once on transient index.lock failures
  auto_install_hook: true        # ensure the pre-commit hook on open (see below)

# Vector index (`pip install outmem[semantic]`). No on/off flag — it's
# active once built (`outmem reindex`); a semantic/hyde/*+semantic
# retrieval strategy or `find_similar` needs it. These keys configure how
# it's built and queried.
semantic:
  embedding_model: openai:text-embedding-3-small
  db_filename: .vectors.db                # tracked in git, sibling of wiki/
  index: pages                            # or `pages+sources` — see below
  embed_frontmatter: false                # prepend "<title> — <tags>" to chunks
  chunk_size: 2000                        # target characters per chunk
  chunk_max: 8000                         # hard ceiling for oversized paragraphs
  overlap_paragraphs: 1                   # paragraphs of overlap between chunks
  similarity_threshold: 0.80              # min cosine sim for find_similar
  top_k: 5

# Optional — HITL gate around write_page / extend_page. See features.md.
approval:
  required_for_writes: false              # flip on for review-before-commit

# Optional — Pydantic Logfire instrumentation. See features.md.
logfire:
  enabled: false                          # true + LOGFIRE_TOKEN in env → traces
```

#### `semantic.index` — what gets embedded

`pages` (default) indexes only curated pages. `pages+sources` additionally
indexes `wiki/sources/`.

**`pages` is the default because an indexed source can never be an answer.**
`search_wiki`'s semantic path maps each matched chunk back to a page slug and
drops everything that isn't under `wiki/pages/` — so a source chunk is fetched
into the fixed-size KNN window and then discarded. It cannot influence the
answer; it can only displace a page that would have. On a store that is half
raw source, roughly half of every candidate window is spent that way.

Index sources when you use `find_similar` / `outmem similar` over raw
material — that is the one path that can actually return them, and it's what
the agent's pre-write duplicate check uses. Narrowing the
scope also prunes already-indexed source chunks on the next
`outmem reindex` (they show up in the `removed` count).

Note the pruned rows are deleted, not compacted away: outmem never runs
`VACUUM`, so `.vectors.db` keeps its size on disk and the rewritten pages
add a fresh blob to git history. The win is retrieval quality and
embedding spend, not repo size.

`outmem reindex --pages-only` does the same thing for a single run
without editing config.

Sources stay fully available either way — they are registered, readable
via `read_source`, and greppable via `grep_wiki` with `scope="sources"`
(which spans both `wiki/sources/` and `wiki/sources-local/`). Only the
*vector* index is narrowed.

`pages+sources` means the **tracked** sources tree only.
`wiki/sources-local/` is never indexed under any setting: the vector DB
stores chunk text verbatim and is committed, so indexing local material
would ship the exact bytes that tree exists to withhold. See
[sources.md](sources.md#the-guarantees-and-what-enforces-them).

#### `semantic.embed_frontmatter` — make titles and tags searchable

Off by default. When on, each chunk is embedded with a
`"<title> — <tags>"` line in front of it.

This matters more than it sounds: `parse_wiki_page` splits frontmatter
off before the chunker runs, so by default **a page's own title and tags
are not in the embedded text at all**. A page whose body never repeats
its own title is effectively unretrievable by that title, and the entire
tag vocabulary contributes nothing to semantic search. The header is
applied to *every* chunk, not just the first, so continuation chunks stay
attributable to their page rather than being reachable only by whatever
happens to be in that slice of the body.

Two limits worth knowing. The header is added to the *document* side
only — `find_similar` embeds your query verbatim — so stored vectors
shift slightly away from raw-body queries; if you rely on `find_similar`
for near-duplicate detection, re-check `similarity_threshold` (0.80 was
tuned without headers). And it does not affect the `rerank` gate, which
re-reads the page body from disk by slug and never sees chunk text.

Turning it on changes what is embedded, so the next `outmem reindex`
re-embeds every page (the header participates in the content hash, so
this happens automatically — no `--force` needed). That is a real
embedding bill on a large wiki; budget for it.

### `retrieval:` — what the agent's wiki search runs

The `retrieval:` block of `config.yaml` configures `search_wiki` in
`wiki_read_tools`. `OptimizeResult.save(rank, store)` rewrites *this block*
(leaving the rest of `config.yaml` and its comments intact); you can also
hand-edit it.

```yaml
# config.yaml
retrieval:
  strategy: rerank(bm25)            # the DSL (see below); default rerank(bm25)
  from_optimization: false          # true ⇒ written by an optimize run
  semantic_top_k: 8                 # chunks fetched per semantic/hyde call
  rrf_k: 60                         # Reciprocal Rank Fusion constant (hybrid)
  max_candidates: 30                # candidate net width before rerank
  max_relevant: 8                   # cap on pages the rerank gate keeps
  rerank_model: anthropic:claude-haiku-4-5
  hyde_model: anthropic:claude-haiku-4-5
  case_insensitive: true
```

`strategy` is a controlled-vocabulary DSL string. A bad value warns and
falls back to the default (config.yaml's forgiving-load contract), it
doesn't crash the open:

| string | pipeline |
| --- | --- |
| `lexical` | ripgrep, ranked by hit frequency |
| `bm25` | SQLite FTS5 BM25 (free, no model) |
| `semantic` | vector cosine over the index |
| `hyde` | LLM hypothetical answer → semantic search |
| `rerank` | shortcut for `rerank(lexical)` |
| `rerank(<source>)` | LLM yes/no gate over `<source>` candidates |
| `rerank(bm25)` | **the default** — BM25 shortlist, Haiku-gated (1 model call/query) |
| `a+b[+c…]` | Reciprocal Rank Fusion of the named atomic legs |

Examples: `bm25+semantic`, `lexical+semantic`, `rerank(semantic)`,
`bm25+semantic+hyde`. Each leg must be one of
`lexical`/`bm25`/`semantic`/`hyde`; rerank is not a fuse leg. The rerank
source can be written inline (`strategy: rerank(semantic)`) or as a
sibling field (`strategy: rerank` + `rerank_source: semantic`).

Note `retrieval.semantic_top_k` is distinct from `semantic.top_k` (which
governs the `find_similar` tool): `search_wiki` with a `semantic`/`hyde`
strategy uses `retrieval.semantic_top_k`.

## Sample `.env`

```dotenv
# Anthropic (default for `model: anthropic:...`):
ANTHROPIC_API_KEY=sk-ant-...

# Or OpenAI:
# OPENAI_API_KEY=sk-...

# Optional — Pydantic Logfire:
# LOGFIRE_TOKEN=...
```

The `git:` block governs how outmem reacts to a stranded `.git/index.lock`
(e.g. from a previous `outmem ask` you Ctrl-C'd): on the next
`WikiStore.open()` we remove the lock file if it's older than
`stale_lock_seconds`. Live concurrent git operations are unaffected.
Set `remove_stale_lock: false` to disable.

## Environment variables

| Var | Effect |
|---|---|
| `OUTMEM_PATH` | Default wiki root (overridden by `--root` on the CLI, passed *after* the subcommand — e.g. `outmem reindex --root /srv/wiki`). |
| `OUTMEM_MODEL` | Model id for `outmem ask` / `build_agent`. Overrides `config.yaml`. |
| `OUTMEM_AGENT_NAME` | Override the agent's commit `user.name`. |
| `OUTMEM_AGENT_EMAIL` | Override the agent's commit `user.email`. |
| `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, … | Provider keys consumed by PydanticAI. Loaded from `.env` if present. |
| `LOGFIRE_TOKEN` | Routes Logfire data to the project the token belongs to (when `logfire.enabled` is true). |

`OUTMEM_AGENT_NAME` and `OUTMEM_AGENT_EMAIL` both have to be set to
take effect — setting one without the other falls back to the defaults.

## System requirements

- **Python ≥ 3.12**
- **`git`** on PATH (every wiki / log commit is a real `git commit`)
- **`ripgrep`** (`rg`) on PATH (search backend)
- Optional, per extra:
  - `[pydantic-ai]` / `[agent]` — pulls `pydantic-ai`
  - `[semantic]` — pulls `sqlite-vec`
  - `[dashboard]` — pulls `fastapi`, `uvicorn`, `markdown-it-py`, `jinja2`
  - `[logfire]` — pulls `logfire`
  - `[dev]` — pulls `pytest`, `ruff`, `mypy`, `types-pyyaml`

GPG signing is **off** for agent commits in v0.1 (spec §12). Outmem's
`commit_as` sets `-c commit.gpgsign=false` per commit; your global
`commit.gpgsign=true` is not affected (it just doesn't apply to the
agent's commits). Re-enabling signing is a v0.2 deferral.
