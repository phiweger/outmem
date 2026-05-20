# CLI reference

`outmem --help` lists subcommands. `outmem <cmd> --help` shows options
for each. Defaults: wiki root from `$OUTMEM_PATH` (or `--root`),
agent identity from `$OUTMEM_AGENT_NAME` + `$OUTMEM_AGENT_EMAIL` (or
the defaults `outmem agent <agent@host>`).

Status/progress messages are timestamped (`[HH:MM:SS]`); structured
output (commit SHAs, search hits, git-log style lines) prints raw so
downstream tooling can pipe it.

## Reading

```bash
outmem init /srv/agent
# → "Initialised wiki at /srv/agent"

outmem read pricing-formula
# → full file (frontmatter + body), printed to stdout

outmem read pricing-formula --body-only
# → just the body

outmem search "cost-plus" --scope wiki
# → wiki-only matches; one "path:line:text" per match

outmem search "cost-plus" --scope all -i           # case-insensitive over wiki+raw+log
outmem search "literal[bracket]" --fixed-strings   # treat pattern as literal
outmem search "x" --max-hits 5                     # hard cap

outmem history pricing-formula
# → sha (10 chars)  iso-date  Author <email>  subject

outmem evolution pricing-formula
outmem evolution pricing-formula acme-msa          # multiple slugs interleaved
outmem evolution pricing-formula --no-log          # drop log/ from the diff stream

outmem steering
# → commits authored by humans since the last record-run, formatted like history
```

## Writing

Body / content come from stdin so commands compose with shell pipes.

```bash
outmem write discounts \
    --title "Discount tiers" \
    --provenance raw/pricing-deck-2026-Q1.md \
    --tag pricing \
    --tag contracts \
    <<< "Standard tiers: 5% / 10% / 15%."
# → 40-char SHA, printed to stdout

outmem extend pricing-formula <<< "Revised: cost-plus 40%."
outmem log pricing <<< "- saw a contradiction between deck and msa."
outmem pull
outmem push
outmem record-run
# → "recorded run at 2026-05-11T12:00:00+00:00 head=abc1234..."
```

## Agent (requires `outmem[agent]`)

```bash
outmem ask "what is our pricing formula?"
# → agent response on stdout, exits 0 on success
# → tool calls logged to stderr: [HH:MM:SS] [tool] search_wiki pattern='…' …

outmem ask --stdin <<< "what is our pricing formula?"
outmem ask "explain pricing" --model anthropic:claude-sonnet-4-6
outmem ask "..." --quiet                          # suppress the per-tool-call trace
outmem ask "..." --no-push --no-record --show-meta
# --show-meta prints a "--\nturn ... commits: <shas> pushed: <bool>" footer on stderr
```

Push is skipped automatically when no `origin` remote is configured —
local-only wikis (`git init` + nothing else) work without flags.
With a remote present, the spec §9 push-retry contract applies:
single `pull --rebase` retry, then `WritebackError` on second failure.

## Ingestion (requires `outmem[agent]`)

```bash
# Just register a source file (copy + sha + record); no agent run.
outmem ingest path/to/guide.md --into veterinary --register-only

# Register + invoke the agent with an optional focus prompt.
outmem ingest path/to/guide.md --into veterinary \
    --prompt "extract drug dosages for cats"
```

Sources land at `wiki/sources/[<into>/]<sha256[:12]>/<filename>`
(tracked in git) and are registered in `wiki/sources/.sources.db`
(SQLite). The hash directory keeps the layout collision-free; the
same file ingested twice deduplicates to the same dir.

**Parallel ingest** is safe — `.sources.db` is SQLite, writers
serialise on the DB's busy-timeout instead of racing on JSON
read-modify-write. Use `xargs -P` across a batch.

Allowed source types: `.md`, `.txt`, `.csv`, `.json`, `.mmd`,
`.yaml` / `.yml`. Binary files are rejected — convert upstream.

## Import (existing markdown vault)

```bash
outmem import /path/to/obsidian-vault
outmem import /path/to/vault --force   # overwrite an existing non-empty wiki
```

Recursively imports every `*.md` under the source directory; hidden
dirs (`.obsidian/`, `.git/`, `.trash/`, …) are skipped automatically.
Each note becomes `wiki/pages/<slug-as-relpath>.md` with frontmatter
generated from the file (H1 → `title`, mtime → `created`/`updated`,
vault path → `provenance`). Wikilinks `[[Note Name]]` are rewritten to
`[[note-slug|Note Name]]` — display preserved, slug machine-resolvable.

Slug collisions across the flat namespace are resolved deterministically
by prefixing with the parent directory (`projects/alpha.md` + `clients/alpha.md`
→ `alpha`, `clients-alpha`). The whole import lands as one
`import: <vault-name>` commit. Wikilinks pointing at notes that
don't exist in the import are left as-is; run `outmem lint` after to
surface them.

## Lint

```bash
outmem lint
# → exit 0 if clean, 1 for warnings only, 2 for errors
```

Static checks: broken wikilinks, malformed frontmatter, slug /
filename mismatch (errors); orphans, stale provenance, index
drift (warnings). For semantic near-duplicate / contradiction
detection, see [features.md](features.md#semantic-index).

## Sync derived artefacts after manual edits

When you edit `wiki/` files directly (Obsidian, vim, VS Code), two
derived artefacts can fall out of sync: `wiki/index.md` (the slug
list) and the semantic vector DB (when on).

```bash
outmem index rebuild       # → `index: rebuild` commit (no-op if in sync)
outmem reindex             # full semantic walk; skip-if-hash-unchanged
```

Or install the pre-commit hook once and forget about both —
manual `git commit` of wiki pages will regenerate `index.md` AND
the vector DB into the same commit:

```bash
outmem hook install        # → .git/hooks/pre-commit
outmem hook uninstall
```

## Semantic search (requires `outmem[semantic]`)

```bash
outmem similar "cost-plus pricing"                # query text
outmem similar --slug pricing-formula             # use a page body, exclude itself
outmem similar --stdin                            # query on stdin
outmem reindex                                    # walk wiki + sources, skip-if-unchanged
outmem reindex --force                            # rebuild from scratch
outmem reindex --path wiki/pages/foo.md           # specific files
```

Set `semantic.enabled: true` in `config.yaml` first. Detailed
behaviour: [features.md](features.md#semantic-index).

## Dashboard (requires `outmem[dashboard]`)

```bash
outmem dashboard --host 127.0.0.1 --port 8765
# Browse http://127.0.0.1:8765/ → redirects to /wiki
outmem dashboard --pull-on-request   # git pull --rebase before each request
```

## Exit codes

| Code | Meaning |
|------|---------|
| `0`  | Success. |
| `1`  | An `OutmemError` was raised (network, git, malformed input, etc.). |
| `2`  | Bad invocation (e.g. empty body to `write`). |

`outmem search` exits `1` when the pattern matched nothing (mirrors `rg`).
