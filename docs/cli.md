# CLI reference

`outmem --help` lists subcommands. `outmem <cmd> --help` shows options
for each. Defaults: wiki root from `$OUTMEM_PATH` (or `--root`, passed
*after* the subcommand — e.g. `outmem reindex --root /srv/wiki`),
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
# → tool calls logged to stderr: [HH:MM:SS] [tool] search_wiki question='…' …

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

## Source registry maintenance

```bash
outmem sources gc            # dry run — report only
outmem sources gc --apply    # write + commit
```

Reconciles `wiki/sources/.sources.db` against what is on disk, in both
directions. Nothing did this before, which is how a registry drifts to
double-digit percentages of junk unnoticed: `list_sources` never stat'd the
filesystem, so rows whose file was gone were handed to the agent as readable
sources it then failed to open.

- **Rows whose file is missing** are removed (`--apply`); the FK cascade takes
  their ingestion history with it.
- **Files with no registry row** are reported, never deleted — removing your
  data to satisfy a registry is backwards. Re-register or remove them yourself.
- **Orphaned ingestion rows** (parent row gone) are swept; they were invisible
  before, skipped silently on read but resident forever.

Dry run is the default because `.sources.db` is tracked in git, so every apply
writes a full blob into history. There is no `VACUUM` and no in-file
tombstone — `git show HEAD~1:wiki/sources/.sources.db` already recovers the
exact pre-gc state.

`outmem lint` reports the same drift as `source-orphaned` / `source-unregistered`
warnings, so CI notices it the day it happens.

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
outmem lint --error-only     # exit non-zero only for errors
# → exit 0 if clean, 1 for warnings only, 2 for errors
```

Static checks. **Errors:** broken wikilinks, malformed frontmatter, slug /
filename mismatch, two pages claiming one slug. **Warnings:** orphans,
stale provenance, provenance citing a sha256 the registry no longer holds,
`.sources.db` disagreeing with disk in either direction, frontmatter that
only parses after self-heal, and *dead slug mentions* — a slug written as
prose (`Volltext-Digest: clinical:old-name`) that resolves to no page.

That last one matters because a broken-`[[link]]` check cannot see it:
prose isn't a link, so nothing validates it, and dead references pile up
silently after a namespace is reorganised. It only fires when the token's
namespace already exists in the wiki, which keeps times (`12:30`) and
ratios (`3:1`) out.

`--error-only` is for CI on a wiki that carries known warnings you don't
want to block on — they are still printed.

For semantic near-duplicate / contradiction detection, see
[features.md](features.md#semantic-index).

## Sync derived artefacts after manual edits

When you edit `wiki/` files directly (Obsidian, vim, VS Code), two
derived artefacts can fall out of sync: `wiki/index.md` (the slug
list) and the semantic vector DB (when on).

```bash
outmem index rebuild       # → `index: rebuild` commit (no-op if in sync)
outmem reindex             # full semantic walk; skip-if-hash-unchanged
```

The pre-commit hook does the same in the same commit on a manual
`git commit` of wiki pages: **repair** any staged page whose frontmatter
won't parse, regenerate `index.md`, and update the vector DB. outmem
**auto-installs it** on `init`/`open` (idempotent, never clobbers a hook
you wrote), so you normally don't run anything:

```bash
outmem hook install        # explicit install (rarely needed) / --force to replace
outmem hook uninstall      # one-off removal
```

To stop the auto-install permanently, set `git.auto_install_hook: false`
in `config.yaml` (otherwise the next `open` re-ensures it). The hook
prints `repaired frontmatter in <path>` so a fix is visible in the commit
output, not silent.

### Frontmatter self-healing

A wiki page whose YAML frontmatter won't parse — overwhelmingly the
imported-data case where a `title:` contains an unquoted `: `
(`title: Influenza (Teil 1): ...`) — is handled in three places, so a bad
page can't silently disappear from a question bank or search:

| When | What happens | Persists to disk? |
| --- | --- | --- |
| **On read** (`store.read`, so `generate_bank`, `search_wiki`, the agent) | repaired *in memory*, logged at WARNING — you get usable content, never a silent skip | no (read has no side effects) |
| **Pre-commit hook** (manual `git commit`) | repaired + re-staged into the commit | yes |
| **On demand** (`store.repair_pages(dry_run=False)`) | walks all pages, repairs, commits | yes |

The repair only ever touches a page that *currently* fails to parse —
it's a guaranteed no-op on a well-formed page, and it verifies the result
parses before accepting it, so it can't make a page worse. Breaks it
*doesn't* understand (mis-indented blocks, truncated frontmatter) are left
to surface loudly. outmem's own writes never produce this (the serializer
quotes correctly); it only arises from external tools or manual edits.

**One page, one verdict.** Every reader of `wiki/pages/` goes through the
same loader (`outmem.index.load_editorial_pages` / `load_page_text`), so
a page either works everywhere or fails everywhere:

| Reader | Repairable page | Unfixable page |
| --- | --- | --- |
| `read_page` / `search_wiki` | served | `FrontmatterError` |
| `wiki/index.md` (the TOC) | listed | omitted **+ WARNING** |
| Backlink graph | full frontmatter honoured | links still scanned from raw text |
| bm25 keyword net (backs the default `rerank(bm25)`) | indexed | omitted **+ WARNING** |
| Semantic index | indexed | omitted **+ WARNING**, and `outmem reindex` exits 2 |
| `outmem lint` | `frontmatter-repairable` **warning** | `frontmatter-invalid` error |

Before this was shared, each reader decided independently and three of
them dropped the page with no signal at all — so a page could be
`read_page`-able yet missing from the TOC and from the default search
path, with nothing in the repo detecting it. Lint was the last holdout:
it parsed directly and so reported a page as a hard error while every
other reader served it happily.

For pages already committed (e.g. pulled from a remote, which the hook
won't have seen), repair on demand:

```python
store.repair_pages(dry_run=True)    # report what's fixable
store.repair_pages(dry_run=False)   # repair + commit
```

## Semantic search (requires `outmem[semantic]`)

```bash
outmem similar "cost-plus pricing"                # query text
outmem similar --slug pricing-formula             # use a page body, exclude itself
outmem similar --stdin                            # query on stdin
outmem reindex                                    # walk wiki + sources, skip-if-unchanged
outmem reindex --force                            # rebuild from scratch
outmem reindex --pages-only                       # skip wiki/sources/ (and prune it)
outmem reindex --path wiki/pages/foo.md           # specific files
```

`reindex` embeds files concurrently (the network bottleneck; writes stay
serial) and prints a `reindex: done/total files` progress counter to
stderr, then a summary like:

```
reindex: 12 re-embedded, 334 unchanged, 0 removed, 47 chunks added, 18234 embed tokens
```

The embedding-token count is the spend you can multiply by your provider's
$/M-tokens to get cost. When Logfire is enabled, the same numbers land on
an `outmem.reindex` span. `--path` reindexes one repo-relative *file*;
`--root` selects the wiki (after the subcommand). `--pages-only` overrides
`semantic.index` for one run — sources already in the index are pruned
(they appear in `removed`).

**Exit code.** A wiki page that exists on disk but could not be indexed is
data loss — it becomes unreachable by `search_wiki`. `reindex` names every
such page on stderr and **exits 1**, so a CI step notices:

```
outmem: 1 page(s) NOT indexed and unreachable by search. `outmem lint` names the offending field:
  wiki/pages/regulatory/deqs-rl-sepsis.md
```

The usual cause is frontmatter that YAML accepts but outmem rejects. Run
`outmem lint` for the precise field.

Build the index first: `outmem reindex`. Detailed
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
| `2`  | Bad invocation (e.g. empty body to `write`), or the command ran but found errors in the wiki (`lint` findings, `reindex` dropped pages). |

`outmem search` exits `1` when the pattern matched nothing (mirrors `rg`).
