# outmem

**Agentic RAG memory over a git-versioned wiki.**

A standalone Python library + CLI for maintaining a directory of
plain-markdown notes that an LLM agent compiles, retrieves from, and
writes back to. Retrieval is shell-tool based (`ripgrep`, `git log`,
`cat`) — no vector index by default — and every agent turn is required
to produce a git commit so identical future queries don't re-pay
retrieval cost.

The idea is Andrej Karpathy's ["LLM Wiki" sketch][kw]: the LLM owns the
wiki, the human curates sources and asks questions, and the wiki is a
compounding artifact that gets richer with every source ingested.
outmem is a working implementation of it — full rationale and the v0.1
spec in [`specs/concept.md`](specs/concept.md).

---

## Main idea

outmem is a deliberate inversion of conventional RAG. Instead of
pre-indexing raw sources into a vector store and reaching into them
on every query, the agent **compiles** raw material into a wiki of
small markdown pages, **retrieves** over the compiled material with
shell tools (`ripgrep`, `git log`, `cat`), and is **required to
commit** at the end of every turn — so identical future queries
don't re-pay retrieval cost. The wiki compounds; the vector index
that would otherwise grow stale doesn't exist.

Three design choices anchor the rest of the system:

1. **Compaction first.** The cheapest retrieval is reading a compiled
   wiki page; raw sources are the fall-through, not the default. This
   directly attacks the *relevance trap* — the gap between embedding
   similarity and actual usefulness ([Raudaschl, "The Relevance
   Trap"][rt]; [Fleck, "Divergence Engines"][de]).
2. **Git as the substrate.** Every write produces a commit; `git log`
   is both the audit trail and the agent's steering signal (recent
   human commits become phase-1 planning context), and `git blame`
   tracks line-level authorship. The same move Claude Code makes for
   code search — agentic shell tools, no index — works here for
   prose ([Nicolai, "Claude Code Doesn't Index Your
   Codebase"][cc]; [SmartScope, "Settling the RAG Debate"][rd]).
3. **Mandatory writeback.** Every agent turn ends with at least one
   commit (`compact:` / `extend:` / `log:`), so the system records
   not just what was retrieved but what it was retrieved *for*.
   Adoption is measured via the **TARS** product
   metric — Target / Adopted / Retained / Satisfied — rather than
   recall@k or nDCG ([Raudaschl, "TARS"][tars]).

A single divergence primitive ships in v0.1 — `topic_evolution`, a
chronological `git log -p` over a topic — for the class of question
convergent retrieval can't answer ("how has our thinking on X
changed?"). The four other divergence primitives sketched in
`concept.md` (contradiction surfacer, negative-space query,
associative drift, cross-domain bridges) are queued for when the
first one has earned its place.

The full conceptual rationale and v0.1 implementation spec live in
[`specs/concept.md`](specs/concept.md) and
[`specs/spec.md`](specs/spec.md).

[rt]: https://breakingproduct.substack.com/p/the-relevance-trap
[tars]: https://uxdesign.cc/tars-a-product-metric-game-changer-c523f260306a
[de]: https://medium.com/@j0lian/divergence-engines-escaping-the-relevance-trap-1bbdbee55ea6
[cc]: https://vadim.blog/claude-code-no-indexing
[rd]: https://smartscope.blog/en/ai-development/practices/rag-debate-agentic-search-code-exploration/
[kw]: https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f

---

## Install

```bash
pip install outmem            # core: WikiStore + CLI
pip install outmem[all]       # everything: agent runtime, semantic, dashboard, logfire
```

Or pick a subset:

```bash
pip install outmem[agent]      # + standalone PydanticAI agent runtime
pip install outmem[semantic]   # + sqlite-vec index for find_similar
pip install outmem[dashboard]  # + read-only FastAPI dashboard
pip install outmem[logfire]    # + Pydantic Logfire instrumentation
```

System: Python 3.12+, `git`, and `ripgrep` (`rg`) on PATH. `outmem init`
checks for these and refuses to proceed if either is missing.

---

## The 60-second mental model

outmem maintains a handful of directories under one wiki root:

| Directory   | Tracked in git | Who writes | What lives there |
|-------------|----------------|------------|------------------|
| `wiki/pages/` | **yes**      | agent + humans (via Obsidian) | compiled knowledge, one concept per file, nested as deeply as the slug demands, YAML frontmatter + `[[wikilinks]]` |
| `wiki/sources/` | **yes**    | `outmem ingest` | ingested source documents, content-addressed under `[<into>/]<sha256[:12]>/<filename>` |
| `wiki/sources-local/` | no (auto-gitignored) | `outmem ingest --local` | sources you may **read but not redistribute** — licensed, copyrighted, embargoed |
| `log/`      | **yes**        | agent + humans | dated decision / observation trail |
| `.outmem/`  | no (auto-gitignored) | outmem | non-git state (backlinks cache, last-run marker) |

The two source trees are the same in every respect except one: git
sees one and never sees the other. That split is what lets a wiki be
compiled from a licensed handbook and still be shareable — the source
bytes stay on your machine, the pages you derived from them travel.
Both are equally readable, searchable, and citable by the agent. See
[`docs/sources.md`](docs/sources.md).

Plus three special wiki-root files:

- **`wiki/AGENTS.md`** — user-editable conventions doc loaded into the
  agent's system prompt every turn. Your customization layer for
  domain, page structure, source-handling preferences. The agent
  reads this to know which namespaces exist and what belongs where.
- **`wiki/index.md`** — auto-maintained slug list, regenerated on
  every write.
- **`wiki/CONTRIBUTORS.md`** — known team identities, used by phase-1
  steering.

Slugs are `:`-delimited: `pricing-formula` is flat, `abx:penicillin`
maps to `wiki/pages/abx/penicillin.md`, `abx:side-effects:misc` to
`wiki/pages/abx/side-effects/misc.md`. Wikilinks carry the same slug:
`[[abx:penicillin]]`.

The agent's loop per turn:

1. **Orient** — read recent human commits (steering signal); choose
   *convergence* (look up a fact) or *expansion* (walk history).
2. **Retrieve** — cheapest tool first: `rg wiki/pages/`, then the
   source documents, then `git log -p --follow` for the expansion path.
3. **Compact** — produce at least one commit before responding
   (`compact:` for new pages, `extend:` for edits, `log:` for
   observations). **Mandatory** — runs that skip it raise `WritebackError`.

---

## Quickstart

```bash
# Scaffold a fresh wiki.
outmem init /srv/my-wiki
export OUTMEM_PATH=/srv/my-wiki

# Optional: tell the agent what this wiki is for.
${EDITOR:-vi} /srv/my-wiki/wiki/AGENTS.md

# Optional: ask the agent something (requires outmem[agent] + an API key).
export OUTMEM_MODEL="anthropic:claude-sonnet-4-6"
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
outmem ask "what's our pricing policy?"
```

There's also a pre-populated example wiki at
[`examples/starter-wiki/`](examples/starter-wiki/) if you want to
poke at outmem before scaffolding your own.

---

## Common workflows

### Ask the agent

```bash
outmem ask "what is our pricing formula and where does it come from?"
```

Searches the pages first, falls back to the source documents if needed,
and produces at least one commit before responding — either extending a
wiki page (`extend: <slug>`) or logging the observation (`log: <topic>`).

### Ingest a source document

```bash
outmem ingest /path/to/some-paper.md \
    --into research \
    --prompt "extract methodology and headline results"
```

Copies the file under `wiki/sources/[<into>/]<sha256[:12]>/`,
registers it in `wiki/sources/.sources.db`, then runs the agent to
write/extend pages with `provenance:` pointing at the registered
source. Parallel `outmem ingest` runs are safe (SQLite serialises
writers). See [docs/cli.md](docs/cli.md#ingestion-requires-outmemagent)
for `--register-only`, re-ingest semantics, and file-type rules.

**Material you can read but not redistribute** — a licensed corpus, a
copyrighted book, an embargoed draft — takes `--local`:

```bash
outmem ingest /path/to/licensed-handbook.md --local --into reference
```

It lands in the gitignored `wiki/sources-local/` instead. The agent
reads, greps, and cites it exactly as it would any other source; the
bytes just never enter git, so the pages you compile from it stay
shareable. `outmem lint` errors if any of them leak into a commit, and
local material is never written into the (committed) semantic index.
Full contract in [docs/sources.md](docs/sources.md).

**When a source gets a new version**, name the document with `--as` so the
revision supersedes its predecessor instead of landing beside it:

```bash
outmem ingest fachinfo/amikacin/2026/document.md --into fachinfo \
    --as fachinfo/amikacin
outmem stale        # pages compacted from a version that is no longer current
```

Because the path embeds the content hash, a revision is otherwise
indistinguishable from an unrelated file. `--as` is what turns `provenance:`
from an audit trail into a signal: it tells you exactly which pages to
re-check when a guideline is republished. See
[docs/cli.md](docs/cli.md#as-which-document-is-this-a-version-of).

### Import an existing markdown vault (Obsidian, plain notes folder)

```bash
outmem import /path/to/obsidian-vault
outmem lint                            # surface anything not auto-resolved
```

One-shot bulk import: walks the source for `*.md`, generates
frontmatter, normalises slugs to outmem's flat namespace, rewrites
wikilinks, and commits everything as `import: <vault-name>`. Hidden
dirs (`.obsidian/`, `.git/`, …) are skipped. See
[docs/cli.md](docs/cli.md#import-existing-markdown-vault) for
collision handling and `--force` semantics.

### Edit wiki files manually (Obsidian, vim, VS Code)

Edit `wiki/pages/**/*.md` however you like — outmem keeps the agent happy
as long as you commit through git. Install the pre-commit hook once
so the auto-maintained `wiki/index.md` and the semantic vector DB
stay in lockstep with your edits:

```bash
outmem hook install
```

Without the hook, the explicit commands are `outmem index rebuild`
and `outmem reindex`. See [docs/cli.md](docs/cli.md#sync-derived-artefacts-after-manual-edits).

### Sync across machines

```bash
outmem pull
outmem push
```

The agent does pull-rebase-push around every `outmem ask` by default
(disable with `--no-pull` / `--no-push`). outmem's git operations
are vanilla; any other tool (Obsidian Git plugin, GitHub Desktop,
plain `git`) interoperates.

### Tune retrieval to your wiki

Which retrieval strategy wins depends on *your* corpus and how questions
are phrased. outmem can measure it: an agent generates a question bank
from your pages, then tries strategies and keeps the one with the best
Hit@k. This is a Python/API tool (not a CLI subcommand) — run it from a
script or REPL:

```python
from outmem import WikiStore
from outmem.optimize import generate_bank, optimize_retrieval

store = WikiStore.open("/srv/wiki")

# Build a question bank (defaults to ~1 question per page — cap big wikis).
bank = generate_bank(
    store,
    model="anthropic:claude-haiku-4-5",
    max_pages=60,          # sample at most 60 pages → ~60 questions (omit = all pages)
)

result = optimize_retrieval(
    store, bank,
    optimizer_model="anthropic:claude-sonnet-4-6",
    eval_sample=30,        # score each config on 30 questions, not the whole bank
    #                        (the winner is re-scored on the full bank, so it's honest).
    #                        THE cost lever: rerank/hyde make 1 model call PER question,
    #                        so a full 700-question bank = 700 calls per eval.
    # allowed_strategies=["bm25", "semantic"],   # also skip rerank/hyde entirely
)
result.print_summary()      # ranked leaderboard → stderr
#  #  config                 score  hit@k  abst   ms/q (p95)
#  1  rerank(semantic)       0.97   0.97   0.00  2112 (3194)
#  2  bm25+semantic          0.93   0.93   0.00    59 (  78)
#  3  bm25                   0.80   0.80   0.00     2 (   3)

result.save(1, store)       # rewrite config.yaml's retrieval: block from row 1
```

> **Watch the cost on big wikis.** Without `eval_sample`, *every* eval
> scores the *whole* bank — and a `rerank`/`hyde` eval makes one model
> call per question, so a 700-page wiki means ~700 calls per eval, ×12
> evals. Set `eval_sample` (per-eval question cap) and `max_pages`
> (bank size) to bound it; `allowed_strategies` without `rerank`/`hyde`
> removes the per-question model calls altogether.

#### The strategies (blocks)

A strategy is one of six **atomic blocks**, optionally composed:

| block | what it does | cost |
| --- | --- | --- |
| `lexical` | ripgrep keyword search, ranked by how often query terms appear | free, no index |
| `bm25` | SQLite FTS5 **BM25** keyword ranking — better term weighting than `lexical` | free, no index |
| `semantic` | vector cosine over the embedding index — finds pages that *mean* the same thing with no shared words | 1 embedding call / query |
| `hyde` | writes a hypothetical answer with a cheap model, then `semantic`-searches on *that* (closer in vector space than a terse question) | 1 model call / query |
| `rerank` | pulls a wide candidate shortlist, then an LLM yes/no-gates each for relevance — highest precision | **1 model call / query** |
| `hybrid` | **R**eciprocal **R**ank **F**usion: blends the rankings of 2+ atomic blocks | sum of its legs |

The out-of-box default is **`rerank(bm25)`** — a free BM25 keyword
shortlist gated by one Haiku call per query (lifts recall on paraphrased
questions; ~1.5 s/query, needs `ANTHROPIC_API_KEY`). Want zero model cost?
Set `strategy: bm25` for plain keyword ranking. `semantic`/`hyde` need
a built index (`outmem reindex`).

**Composing two (or more).** There are two ways to combine blocks:

- **Gate over a source** — `rerank(<source>)`: rerank fed by another
  block's shortlist. `rerank(semantic)` (rerank **+** semantic) is the
  recall-then-precision pairing that usually wins on paraphrase-heavy
  banks; bare `rerank` = `rerank(lexical)`.
- **Fuse legs** — `a+b[+c…]`: RRF over 2+ atomic legs, e.g.
  `bm25+semantic` (keyword precision **+** vector recall) or
  `lexical+bm25+semantic`. Legs must be atomic
  (`lexical`/`bm25`/`semantic`/`hyde`); `rerank` is not a fuse leg.

**Restrict what the optimizer tries** with `allowed_strategies` — the
biggest cost lever, since `rerank`/`hyde` are the only blocks that make a
model call *per question*:

```python
result = optimize_retrieval(
    store, bank, optimizer_model="anthropic:claude-sonnet-4-6",
    allowed_strategies=["lexical", "bm25", "semantic"],   # skip rerank/hyde entirely
)
```

It takes the **block names** above (not composed strings), and the
restriction covers **rerank sources and hybrid legs too** — so
`["rerank", "semantic"]` permits `rerank(semantic)` but *bounces*
`rerank(lexical)` (lexical wasn't allowed). A disallowed config is bounced
without burning an eval. (To cap the *number* of trials instead, lower
`max_evals`.)

#### Persisting the winner

`save(rank, store)` rewrites the **`retrieval:` block of your
`config.yaml`** in place (every other setting and comment left intact),
tagged `from_optimization: true` so a `git diff` shows the wiki was tuned.
From then on the agent's `search_wiki` tool runs that pipeline. You can also
just edit the block by hand — `strategy` is the composed string from above:

```yaml
# config.yaml
retrieval:
  strategy: rerank(semantic)    # or bm25+semantic, semantic, bm25, …
  from_optimization: false
```

Default `strategy` is `rerank(bm25)`; a bad value warns and falls back (no
crash). Full knobs and cost notes in
[docs/configuration.md](docs/configuration.md#retrieval--what-the-agents-wiki-search-runs)
and [docs/autoresearch.md](docs/autoresearch.md).

---

## Wiki page format

Every page under `wiki/pages/<slug-as-relpath>.md` opens with YAML
frontmatter. Slugs are flat (`pricing-formula`) or namespaced
(`abx:penicillin`, `abx:side-effects:misc`); each `:` becomes a
directory under `wiki/pages/`. Example frontmatter:

```yaml
---
title: Pricing formula
slug: pricing-formula
provenance:
  - path: sources/a1b2c3d4e5f6/pricing-deck-2026-Q1.md
    drive_path: /shared/Sales/2026-Q1-pricing-deck.pdf
    sha256: 9e2c1f00aa
created: 2026-01-15T10:30:00Z
updated: 2026-05-04T11:32:00Z
tags: [pricing, contracts, finance]
---

The 2026 pricing formula is **cost-plus 35%**…

See also [[acme-msa]] for the Acme exception.
```

`provenance:` accepts plain path strings or dicts with richer
ingestion metadata. There's no `authority` field — anyone (human or
agent) may edit any page; "who wrote what" is reconstructed from
`git log` / `git blame`. Wikilinks (`[[slug]]`) resolve to
the corresponding page under `wiki/pages/`. Both flat and namespaced
slugs are valid wikilink targets.

Full schema: [docs/python-api.md](docs/python-api.md#writing--three-paths).

---

## Embed in your own PydanticAI agent

If you want outmem as a *component* of a larger agent (virtual
assistant, RAG pipeline, etc.) rather than as its own runtime,
attach the tools + the same system prompt `outmem ask` uses to your
own `pydantic_ai.Agent`:

```python
from pydantic_ai import Agent

from outmem import WikiStore
from outmem.adapters.pydantic_ai import wiki_tools
from outmem.agent import render_system_prompt

store = WikiStore.open("/path/to/wiki")

agent = Agent(
    "anthropic:claude-sonnet-4-6",
    tools=wiki_tools(store),                   # 14 tools (15 with semantic)
    system_prompt=render_system_prompt(store), # identical to outmem ask's
)
```

`render_system_prompt(store)` returns the **exact same prompt string**
the bundled `outmem ask` runtime sends — the three-phase framing
(orient / retrieve / compact), recent human commits as steering
signal, the bundled skill bodies (`search` / `evolution` / `write` /
`ingest`), and your wiki's `AGENTS.md`. Tools + prompt → your agent
has prompt-level parity with `outmem ask`. Pass `include_steering=False` if you
don't want the phase-1 steering signal injected (handy for stateless
assistant turns).

**What you don't get in embed mode** (intentional — these are runtime
concerns):

- Mandatory writeback enforcement
- Pull-before / push-after / record-run lifecycle
- HITL approval gate around `write_page` / `extend_page`
- Default `max_tokens=16384` and Anthropic prompt caching

If you want those too, the all-in-one is `outmem.agent.ask_sync(store, query=…)`.

**Fine-grained control** — if you want to compose the prompt yourself
(e.g. prepend your own preamble, swap skill selection, omit AGENTS.md),
the building blocks are all public:

```python
from outmem.adapters.pydantic_ai import skill_text, wiki_tools

agents_md = store.read_agents_md() or ""
agent = Agent(
    "anthropic:claude-sonnet-4-6",
    tools=wiki_tools(store),
    system_prompt=(
        "You are a helpful assistant.\n\n"
        + skill_text("search")
        + skill_text("write")
        + (f"\n# Wiki conventions\n\n{agents_md}" if agents_md else "")
    ),
)
```

---

## Read-only consult — wiki as a tool in someone else's agent

When you've curated a wiki and want an *external* agent to consult it
without ever modifying it, there's a one-call factory:

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

`build_consult_wiki` opens the wiki via
`WikiStore.open(path, read_only=True)` and builds an inner PydanticAI
agent with the read-only tool palette (search / read / list / backlinks
/ history / evolution / sources) plus a tight "cite by `[[slug]]`,
explicitly say so if the wiki has nothing on the topic" system prompt
and the same `max_tokens=16384` + Anthropic prompt-caching settings as
the full `outmem ask` runtime. The outer agent gets a black-box
`consult_wiki(question) -> str` tool — outmem-internal vocabulary
never leaks across the boundary.

What `read_only=True` guarantees:

- Every commit-producing entry point on `WikiStore` (`write_page`,
  `extend_page`, `append_log`, `add_source`, `record_ingestion`,
  `rebuild_index`, `import_vault`) raises `OutmemError` via a single
  guard in `_commit_paths`. `pull()` is also refused (rebase mutates
  the working tree). `push()` stays unguarded — nothing local to push.
- The read-tool palette doesn't even expose write tools (defense in
  depth — the model never sees the write API).
- `WikiStore.open(read_only=True)` skips the directory-creating layout
  step, skips the stale `.git/index.lock` cleanup, and the backlinks
  cache runs memo-only (no writes to `.outmem/`). The wiki's
  filesystem state is left exactly as the caller found it, which makes
  the mode safe to use on a literally read-only mount.

For finer-grained control (custom system prompt, your own retry logic):

```python
from pydantic_ai import Agent
from outmem import WikiStore
from outmem.adapters.pydantic_ai import wiki_read_tools

store = WikiStore.open("/srv/curated-wiki", read_only=True)
agent = Agent(
    "anthropic:claude-sonnet-4-6",
    tools=wiki_read_tools(store),
    system_prompt="You answer from the wiki only. Cite [[slugs]].",
)
```

---

## Where to go next

- [`docs/cli.md`](docs/cli.md) — every subcommand with examples.
- [`docs/search.md`](docs/search.md) — the search & retrieval workflow:
  `search_wiki` (strategy-driven) + `grep_wiki` (literal), and the
  semantic tiers (when to reach for each).
- [`docs/sources.md`](docs/sources.md) — the two source trees, what
  `--local` is for, and the guarantees around material you may read
  but not redistribute.
- [`docs/python-api.md`](docs/python-api.md) — `WikiStore` + the
  PydanticAI adapter + the standalone agent runtime.
- [`docs/growing-the-wiki.md`](docs/growing-the-wiki.md) — reading
  the log + lint signals to figure out what to ingest next.
- [`docs/configuration.md`](docs/configuration.md) — `wiki/AGENTS.md`,
  `config.yaml`, `.env`, environment variables, system requirements.
- [`docs/features.md`](docs/features.md) — semantic index,
  filter, retrieval tuning, write approval, Logfire, dashboard (all opt-in).
- [`docs/autoresearch.md`](docs/autoresearch.md) — making outmem improve
  its own retrieval: generate a question bank, let an agent tune the
  retrieval pipeline, pick a winner and save it to `retrieval.yaml`
  (now) — and the self-modifying code loop (future).
- [`docs/development.md`](docs/development.md) — dev install,
  repository layout.
- [`CHANGELOG.md`](CHANGELOG.md) — what changed per release, including
  the breaking ones and how to migrate.
- [`specs/concept.md`](specs/concept.md) — the original pattern this
  implements.
- [`specs/spec.md`](specs/spec.md) — v0.1 implementation spec.

---

## Status

Package **0.10.0**, implementing spec v0.10. The core loop (compile →
retrieve → mandatory writeback) has been stable since v0.1; since then
the additions have been the SQLite source registry with supersession,
the opt-in semantic index, HITL write approval, retrieval tuning, and —
in 0.10.0 — the tracked/local source split. See
[`CHANGELOG.md`](CHANGELOG.md); 0.10.0 has breaking changes and a
migration guide.

Tests + ruff + mypy strict clean.

To give feedback, report at <https://github.com/phiweger/outmem/issues>.
