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

# Optional — requires `pip install outmem[semantic]`.
semantic:
  enabled: false                          # flip to true to turn the vector index on
  embedding_model: openai:text-embedding-3-small
  db_filename: .vectors.db                # tracked in git, sibling of wiki/
  chunk_size: 2000                        # target characters per chunk
  chunk_max: 8000                         # hard ceiling for oversized paragraphs
  overlap_paragraphs: 1                   # paragraphs of overlap between chunks
  similarity_threshold: 0.80              # min cosine sim for find_similar
  top_k: 5

# Optional — relevance filter over lexical search; requires `pip install outmem[agent]`. See features.md.
relevance:
  enabled: false                          # cheap-model gate; swaps the search_wiki tool
  model: anthropic:claude-haiku-4-5
  max_relevant: 8                         # cap on pages kept per search
  max_candidates: 20                      # distinct pages sent to the filter
  candidate_max_bytes: 65536              # width of the wide ripgrep net (bytes)
  context: page                           # "page" (cap below) | "lines"
  context_chars_per_page: 2000

# Optional — HITL gate around write_page / extend_page. See features.md.
approval:
  required_for_writes: false              # flip on for review-before-commit

# Optional — Pydantic Logfire instrumentation. See features.md.
logfire:
  project: null                            # any non-null string opts in
```

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
| `OUTMEM_PATH` | Default wiki root (overridden by `--root` on the CLI). |
| `OUTMEM_MODEL` | Model id for `outmem ask` / `build_agent`. Overrides `config.yaml`. |
| `OUTMEM_AGENT_NAME` | Override the agent's commit `user.name`. |
| `OUTMEM_AGENT_EMAIL` | Override the agent's commit `user.email`. |
| `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, … | Provider keys consumed by PydanticAI. Loaded from `.env` if present. |
| `LOGFIRE_TOKEN` | Routes Logfire data to the project the token belongs to (when `logfire.project` is set). |

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
