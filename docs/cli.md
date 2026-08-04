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
    --provenance sources/a1b2c3d4e5f6/pricing-deck-2026-Q1.md \
    --tag pricing \
    --tag contracts \
    <<< "Standard tiers: 5% / 10% / 15%."
# → 40-char SHA, printed to stdout

outmem extend pricing-formula <<< "Revised: cost-plus 40%."
# --provenance replaces the page's source pointers (repeat for several);
# omit it and they are left untouched. See "Staleness" below.
outmem extend pricing-formula --provenance sources/deck/a1b2c3d4e5f6/q2.md <<< "…"
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

# Say which document this file is a version of.
outmem ingest parsed/fachinfo/amikacin/output/a1b2/document.md \
    --into fachinfo --as fachinfo/amikacin
```

Sources land at `wiki/sources/[<into>/]<sha256[:12]>/<filename>`
(tracked in git) and are registered in `wiki/sources/.sources.db`
(SQLite). The hash directory keeps the layout collision-free; the
same file ingested twice deduplicates to the same dir.

### `--as`: which document is this a version of?

Because the path embeds the content hash, a *revised* document lands at a new
path and looks like an unrelated source. `--as` gives it a name that survives
the revision, so the new version **supersedes** the old one instead of
accumulating beside it. Nothing is deleted — the old file, its sha and its
ingestion history stay put, which is exactly what [`outmem stale`](#staleness)
needs to find the pages compacted from a version that is no longer current.

When you omit `--as`, outmem derives the name from the path — and **refuses the
ingest** if that name is already another file's identity:

```
outmem: 'fachinfo/document' is already the identity of 'fachinfo/9b3d0d4e1a35/document.md'.
    that row came from    parsed/fachinfo/aztreonam/output/9b3d0d/document.md
    this file comes from  parsed/fachinfo/amikacin/output/a1b2c3/document.md
If this file is a new version of that document, pass `--as fachinfo/document`.
If it is a different document that happens to share a filename, pass `--as` with a
name that distinguishes it, e.g. `--as fachinfo/amikacin`.
```

It refuses rather than guesses because the two cases are indistinguishable from
the *wiki* path — copying into `<into>/<sha>/<filename>` already discarded what
told them apart. The **ingest origins** still have it, so both are quoted and
the suggested name comes from the first segment where they diverge. You read an
answer rather than inventing one. (When the older row predates `origin_path`,
the refusal stands but the evidence lines are omitted rather than fabricated.)

A pipeline that emits `<drug>/output/<hash>/document.md` hits this on its second
document — pass `--as` from the start and it never comes up. Re-ingesting
*identical* content is still a no-op, not a refusal.

### What a key looks like

A key names a **document**, not a file, so the derived form drops the two
properties that change without the document changing:

| ingested as | key |
|---|---|
| `fachinfo/9b3d0d4e1a35/document.md` | `fachinfo/document` |
| `Fachinfo/Document.MD` | `fachinfo/document` |

Dropping the extension is the point: a pipeline that starts exporting `.txt`
where it used to export `.md` keeps the same identity. Keep it in the key and
that change starts a new identity **silently** — the exact failure mode
supersession exists to remove. Note which way this cuts: dropping the extension
makes *collisions* more likely, and a collision is a refusal, never a merge. So
`report.md` and `report.csv` under one `--into` now stop and ask rather than
quietly becoming unrelated documents.

Keys are free-form apart from that normalisation, so when a document has a real
external identifier, use it — `--as awmf/113-001`, `--as doi/10.1001-jama-2026`.
Only extensions outmem accepts as source types are stripped, so an identifier's
dot survives intact.

Keys use `/`, not the `:` of page slugs. That is deliberate: `[[abx:amikacin]]`
is a page and `fachinfo/amikacin` is a source document, and they appear side by
side in `outmem stale` output. Keeping them visually disjoint stops an agent
trying to `read_page` a source key.

**Parallel ingest** is safe — `.sources.db` is SQLite, writers
serialise on the DB's busy-timeout instead of racing on JSON
read-modify-write. Use `xargs -P` across a batch.

Allowed source types: `.md`, `.txt`, `.csv`, `.json`, `.mmd`,
`.yaml` / `.yml`. Binary files are rejected — convert upstream.

## Renaming / reorganising

```bash
outmem rename rki:ratgeber:influenza clinical:influenza
outmem rename old new --no-alias      # don't keep the old name resolving
outmem rename old new --no-rewrite    # leave inbound [[links]] pointing at the old slug
```

Moves the file, updates `slug:`, rewrites every inbound `[[wikilink]]` across
`wiki/pages/` and `log/`, and records the old slug in `aliases:` — all in one
`rename: <old> -> <new>` commit.

The alias is the point, not a nicety. A link rewrite can only reach what is
*inside* the wiki; references outside it — tickets, configs, a shipped answer
citing a slug — are ones outmem cannot touch. The alias keeps those resolving.

Link rewriting matches whole wikilinks, so prose containing the slug is left
alone and `[[old:slug:child]]` (a different page) isn't caught by a prefix
match. `outmem lint` reports surviving alias-resolved links as
`wikilink-via-alias` warnings so the aliases don't become permanent debt.

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

### Backfilling identity on an older wiki

```bash
outmem sources backfill            # dry run — propose only
outmem sources backfill --apply    # write the unambiguous ones
```

A wiki built before `--as` existed has rows with no identity, so re-ingesting a
document lands as an unrelated row and supersession never fires. `backfill`
reads the identity each path implies and assigns it — but **only where exactly
one row claims it**.

Where several rows share a name it stops and shows you the evidence that settles
it — the pages whose `provenance:` cites each row, and where each was ingested
from. Different citing pages, or different origins, means different documents:

```
2 identit(ies) are claimed by more than one row. …
  fachinfo/document
      fachinfo/9b3d0d4e1a35/document.md
          cited by [[abx:amikacin]]
          from     parsed/fachinfo/amikacin/output/9b3d0d/document.md
      fachinfo/c4f10ab7e221/document.md
          cited by [[abx:aztreonam]]
          from     parsed/fachinfo/aztreonam/output/c4f10a/document.md
      -> re-ingest each with `--as <name>` to resolve
```

The hash segment is *verified* against the row's own sha256 rather than
pattern-matched, so a directory legitimately named like a hash is never
mistaken for one. A name **already held** by another row is treated as
ambiguous no matter how few rows derive it — assigning it would put two live
rows on one identity, which is exactly the merge `--as` refuses to perform, and
doing it silently in a migration would be worse. An identity set explicitly with
`--as` is never overwritten, so re-running is safe.

To resolve an ambiguous group, re-ingest each file with `--as`. The bytes are
already registered, so this only sets the identity — nothing is copied twice.

<a id="staleness"></a>
## Staleness — pages citing a superseded source

```bash
outmem stale
# → 0 clean, 1 if any page cites a superseded version, 2 if a page failed to load
```

The payoff of supersession. Every page's `provenance:` names the source
versions it was compacted from; when one of those moves to a new version,
`outmem stale` reports the pages that may no longer hold:

```
1 page(s) cite a source that has been superseded:

  [[abx:amikacin]]
      cites   fachinfo/9b3d0d4e1a35/document.md
      current fachinfo/e77a10b4c503/document.md
```

It follows the chain to the *newest* version, not just the next one, so you
diff against something current. A page whose frontmatter will not parse is a
page this check could not run on, so those are named on stderr and the command
exits 2 rather than reading as clean.

To clear a report, re-compact the page and update its citation in the same
step — `--provenance` replaces the pointers, and is the only way to change
them:

```bash
outmem extend abx:amikacin --provenance sources/fachinfo/e77a10b4c503/document.md <<< "…"
```

It **reports only** — deciding whether a page still stands is a judgement call,
and on clinical content that belongs to a human (or to an explicit agent run
over this list), not to a side effect of ingest. Wire it into CI as a warning
gate, or run it after a batch re-ingest.

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

**Aliases resolve here too.** A slug that opens only via an alias is not
dead, so it is never reported as such. In *prose* — which you can edit — it
becomes a `slug-mention-via-alias` warning instead, the same nudge
`wikilink-via-alias` gives a link, so the alias can eventually be retired.
Inside a **frozen source** it is silent: the file is content-addressed and
cannot be edited, so the alias protecting that reference is the system
working as designed, and a warning would only ask you to fix something you
can't.

An alias a frozen source depends on is **load-bearing**, and neither nudge
tells you to retire it — the message says to keep it instead. Retiring it
would break a reference in a file you cannot edit.

### Frozen sources naming page slugs

A source under `wiki/sources/` can never change — the path embeds its sha. A
page slug changes whenever you reorganise. A source that names slugs couples
the two, and one production wiki carried 136 references that had rotted this
way, none of them in genuine third-party material.

outmem **records the coupling rather than forbidding it.** At ingest, every
page slug the source names is resolved and stored in the registry:

```
outmem: this source names 2 page slug(s); the references are recorded,
so a rename will follow them.
  prose     clinical:sepsis -> clinical:sepsis
  [[link]]  glossary -> glossary
  1 were matched heuristically from prose. Writing them as [[wikilinks]]
  before ingest makes them exact …
```

The bytes are never touched — a source is a faithful copy — but `outmem rename`
re-points the *mapping*, so the reference survives a reorganisation even with
`--no-alias`. `outmem lint` reads the mapping, so it reports only what is
genuinely gone, and names both what the file says and what it meant:

```
frozen source references 'clinical:sepsis', recorded at ingest as
'clinical:infektion:sepsis', which no longer exists. A rename would have been
followed — this page was deleted or moved outside outmem.
```

**Write references as `[[wikilinks]]`.** A bare prose slug is a *guess*: the
grammar needs at least one `:`, so a single-segment slug like `glossary` cannot
be detected at all, and `12:30` and `3:1` look exactly like slugs. A `[[link]]`
is exact — it works for single-segment slugs, survives a wiki with no
namespaces, and can never be a time or a ratio. Only sources you write yourself
can carry markup, which is fine: that is empirically the only material that
names slugs.

This protects against **renames, not deletions.** Delete a page and no mapping
brings it back — you just learn precisely which sources are now orphaned.

Sources ingested before outmem recorded this are caught up by
`outmem sources backfill --apply`, which resolves whatever is still live today.

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

Progress goes to stderr. On a terminal it is one `\r`-updated line; when
stderr is captured (CI, a redirected log, a subprocess reading the tail) it is
throttled to about a dozen lines plus the first and last, so a long reindex
stays visible without burying everything else that shares the log. Pass an
`on_progress` callback via the Python API to route every tick elsewhere.

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
| `2`  | Bad invocation (e.g. empty body to `write`), or the command ran but found errors in the wiki (`lint` findings, `reindex` dropped pages, `stale` pages that failed to load). |

`outmem search` exits `1` when the pattern matched nothing (mirrors `rg`).
`outmem stale` exits `1` when a page cites a superseded source.
