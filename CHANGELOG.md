# Changelog

Notable changes per release. Versions before 0.10.0 are in the git
history (`git log --grep '^release:'`).

## Unreleased

### Changed

- Default Sonnet bumped: `anthropic:claude-sonnet-4-6` →
  `anthropic:claude-sonnet-5` everywhere the string was pinned — the
  agent default (`DEFAULT_MODEL`), `consult_wiki`'s parameter default,
  the eval judge, and the repo-level `config.yaml` that `outmem init`
  seeds new wikis from. Existing wikis are untouched: the model is
  pinned in each wiki's own `config.yaml`, so the bump reaches new
  wikis (and anyone resolving through env/defaults), not old ones.
  Two things shift with the model: Sonnet 5 runs adaptive thinking by
  default when no `thinking` config is sent (outmem sends none), and
  thinking shares the `max_tokens` budget; its tokenizer also counts
  ~30% more tokens for the same text, so cost baselines move even
  though per-token pricing is unchanged. The rerank/HyDE gate stays
  on `claude-haiku-4-5`, which is still the current Haiku.

## 0.13.0

**Supersession stops depending on how you named the file.** Reported
from a ~1170-source clinical wiki. `document_key` links a revision to
what it replaces — but a source ingested without `--as` derives its
identity from its filename, and the edition marker is usually *in* the
filename. So `eucast-2024.md` and `eucast-2026.md` became two unrelated
documents, no edge was written, and `outmem stale` never reported the
page compacted from the older one. Nothing was wrong in the registry;
the failure was entirely an absence.

### Added

- **`outmem sources rekey KEY [--to KEY]`** — move a document to another
  identity and rebuild its supersession chain. The deliberate operation
  `adopt_document_key` already refused to perform as a side effect.

  It moves **every** row holding the key, so a chain is never split, and
  **rewrites the edges** across the result. That second half is the
  point: relabelling alone — the obvious hand repair, and the one a
  reporter is most likely to reach for — leaves two rows live under one
  identity, where `latest_for` silently returns the newer and `stale`
  still reports nothing. The same silence, one step further in.

  Merging is the normal case. With no `--to` it rebuilds the chain in
  place, which repairs a registry edited out of band. Nothing is
  deleted, so ingestion history and recorded page references survive —
  `gc` + re-ingest cascades both away.

  Ingest timestamps are stored to the second, so a bulk ingest can
  register two editions with the same one. The chain stays deterministic
  (the preview and the write must agree) but the order is then a guess,
  and the dry run marks it rather than presenting a coin flip as a fact.
- **Two lint warnings for versions that should be one chain.**
  `unlinked-source-versions` names derived identities differing only in
  a number. It carries the ingest origins as evidence, because "next
  edition of that" and "different document, similar name" look identical
  from the path — and it names the remedy for *either* answer, since a
  check that cannot reach zero gets silenced wholesale. Identities you
  set with `--as` are never second-guessed: `doi/10.1001-jama-2026` and
  `-2027` differ only in a digit and are different articles.

  `multiple-live-versions` is the exact version of the same defect — one
  identity, several rows all reading as current, which outmem's own
  write paths refuse to create.
- **`superseded_ok: "<reason>"` on a provenance entry**, with `date:`.
  A page that *compares* two editions has to cite both, and reporting it
  forever teaches the reader to skip the report.

  The acknowledgement is **scoped to the version it was made against**,
  exactly as `finding:` is: it holds only while the acknowledged head is
  still the head. A newer edition moves past the date and the row fires
  again. "We deliberately cite 2024 while 2026 exists" says nothing
  about 2027, and a permanent suppression would restore the silent
  staleness this feature exists to break — now with a human signature on
  it. That is what makes `date:` load-bearing rather than decorative;
  without a usable one nothing is suppressed and `outmem lint` says so
  (`invalid-supersession-ack`).
- `outmem stale --all` and `--json`. The default output counts what it
  hid instead of silently omitting it, and both modes use the same exit
  codes.
- `store.rekey_document()`, `store.provenance_annotations()`,
  `SourceRegistry.rekey()` / `.plan_rekey()`, `sources.sibling_form()`,
  `sources.version_order()`, `sources.find_unchained_versions()`,
  `StaleCitation.acknowledged`.

### Fixed

- **`outmem sources backfill` no longer dies on a source named only for
  its type.** `Path(".md.md").suffix` is `.md`, so such a file passes the
  extension check and ingests fine with an explicit `--as` — and its
  path then implies no usable identity, which the derivation reports by
  raising. Correct for a single ingest, where the operator is standing
  there; fatal for a pass over the whole registry, where one such row
  took the entire command down. Sweeps now skip it (it has nothing to
  propose and stays keyless, which is what it already was).
- **A read-only store no longer reaches the bare `SourceRegistry`
  constructor.** `SourceRegistry.empty()` names the deliberate
  no-database case, so the one legitimate caller says what it means.
- `_live_claimants` built entries from a partial column list, reporting
  `refs_scanned_at=None` for sources that had in fact been scanned. The
  row→entry mapping now exists once.

### Note

Once a document has been rekeyed, ingest its next edition with
`--as <key>` and supersession links on its own — the repair is one-time
per document, and the ingest skill now says so.

## 0.12.0

Two retrieval gaps reported from a ~600-page clinical wiki after a week
of production use. Both are cases where **the content was already there
and correct** and could not be found.

### Added

- **`semantic.embed_headings`** — each chunk is embedded with the
  heading trail of the section it starts in (`Diagnostik > Blutkulturen`),
  on the line below the page header. The chunker splits on blank lines,
  so a `## Heading` is just another paragraph: once a chunk boundary
  moves past it, every following chunk in that section is embedded with
  no record of which section it is in. A section whose body never
  repeats its own heading is unretrievable by that heading — the defect
  `embed_frontmatter` fixes one scope up.

  Off by default, like `embed_frontmatter`, because flipping it
  re-embeds the corpus. The flag participates in the content hash, so
  the flip invalidates correctly instead of leaving the index reporting
  `skipped` while serving vectors built under the old policy.

  On a fixture reproducing the reported case, the chunk stating
  "95 % negativ" — whose body never contains "Blutkultur" — gains 42 %
  similarity for the query naming its section.
- **`finding:` on a provenance entry** — `silent` | `contradicts` |
  `out-of-scope`, plus free-form `scope` / `note` / `date`. Records
  "we checked source X and it is silent on Q", which is otherwise
  indistinguishable from nobody having looked. `outmem lint` warns on
  an unrecognised value (one that isn't recognised records nothing and
  reads as an ordinary citation); `outmem stale` marks these
  `[silent — re-check]`, because an absence expires with the version it
  was checked against in a way a claim does not.
- `outmem.outline.heading_path_at()`, and `Section.start_char` /
  `end_char`. Line spans stay file-relative (grep parity), char spans
  body-relative (chunk parity).
- `Chunk.heading_path`, `StaleCitation.finding`,
  `store.provenance_findings()`.
- Two more lint warnings around `finding:`, both cases where an entry
  the author believes records a checked absence records nothing:
  `finding-without-source` (no `path:`, so `outmem stale` can never
  re-check it) and a non-string value, which the extractor would
  otherwise drop before the vocabulary check saw it.

### Not shipped

Premise gates — [#10](https://github.com/phiweger/outmem/issues/10).
Filed with the diagnosis intact rather than built: `embed_headings` plus
a section heading may already cover the reported cases, and the evidence
for that is a before/after the reporter offered to run.

## 0.11.0

**Two retrieval calls that couldn't finish the job they started.** Both
issues came from the consuming side ([phiweger/fleming][fleming]) with
session traces, and both had the same shape: the tool returned in
13–31 ms, then cost a ~3 s model round-trip to ask the obvious
follow-up. The unit that moves the wall clock is the call, not the file
read.

### Added

- **`grep_wiki(context=N)`** — `rg -C`, clamped 0–10. For the questions
  grep is best at (a threshold, a deadline, a paragraph reference) the
  answer is the matched line plus one or two either side; without it the
  caller opens the whole page to see what its own hit continues into.
  Matches render `slug:line:text`, context `slug-line-text` — ripgrep's
  own convention — and separate neighbourhoods are split by a blank
  line. The leading slug is identical on both, so a slug read off a
  *context* row still feeds `read_page`. **`context=0` output is
  byte-identical to 0.10.0.**
- **`outmem search -C N`** — the same context on the CLI, which the docs
  already described as `grep_wiki`'s equivalent.
- **`read_page(section="<heading>")`** — read one section instead of the
  page.
- **`outmem.outline`** — `parse_outline`, `preamble_chars`,
  `find_section`. ATX headings, fenced code excluded, nesting-aware
  spans. Useful independently of the tool layer.
- **`SearchHit.is_match`** — distinguishes match rows from context rows.

### Changed

- **`read_page(peek=True)` returns the page's outline, not its first
  1000 characters.** In the reported session three peeks were each
  followed immediately by a full read of the same slug: the peek never
  avoided a read, it only preceded one. That is the most a prefix can
  do — it answers "is this page on topic", which `search_wiki` already
  answered by returning an excerpt. The outline answers the question the
  caller actually has next: *where* in this page is the fact, and is it
  worth the context. It names every section however long the page,
  where a prefix is blind to anything in the fourth one.

  Line numbers are file-relative, so they agree with what `grep_wiki`
  prints for the same page. A page with no headings is returned whole
  rather than as an empty map — asking twice for something that small is
  the round-trip this is meant to remove.

  Callers that relied on the prefix should drop `peek` and read the page.
- `search(max_hits=…)` counts matches rather than rows, and a match's
  trailing context is no longer truncated away by the cap.

### Removed

- `outmem.adapters.pydantic_ai.PEEK_CHARS`. It was the prefix length,
  and there is no longer a prefix. Nothing in outmem imported it; a
  consumer that did wanted the old peek, which is now a full read.

[fleming]: https://github.com/phiweger/fleming

## 0.10.0

**The source tree splits in two.** A wiki can now be compiled from
material it may read but not redistribute — a licensed corpus, a
copyrighted book, an embargoed draft — and still be shareable. The
source bytes stay on your machine; the pages derived from them travel.

### Added

- **`wiki/sources-local/`** — an untracked sibling of `wiki/sources/`.
  Same content-addressed layout, same registry schema, same tooling;
  git never sees it. Created on first use, so a wiki that never needs
  it is byte-identical to one from 0.9.
- **`outmem ingest --local`** / `store.add_source(..., local=True)`.
  Nothing is committed: both the file and its registry live inside the
  gitignored tree.
- **`outmem sources list`** — every registered source, tree-qualified,
  with a count of how many are local-only. The ingest skill had
  documented this command for a while; it now exists.
- **`SourceEntry.local`** and **`SourceEntry.citation_path`**
  (`sources/…` or `sources-local/…`). `rel_path` remains the registry
  key.
- **Two lint errors** that verify rather than assume the split holds:
  `local-source-tracked` (git is tracking local bytes) and
  `local-source-indexed` (local chunks reached the vector index).
- **`git_ops.tracked_paths_under()`** — asks git what it *is* tracking,
  not what `.gitignore` says, because a committed file stays tracked
  whatever you write afterwards.

### Changed — breaking

- **`raw/` is removed.** It was untracked *and* unregistered *and*
  unmanaged; `sources-local/` keeps the registry and gives up only the
  git tracking. Existing `raw/` directories are left untouched on
  upgrade — outmem just stops reading them. Migration guide:
  [docs/sources.md](docs/sources.md#migrating-from-09-the-raw-directory).
- **`scope="raw"` → `scope="sources"`**, which spans *both* source
  trees. Searching one of them would have been the sharp edge worth
  avoiding: an agent told to "fall through to the sources" must not
  silently miss half the corpus because of a distribution policy it
  cannot see. `scope="raw"` now raises with a pointer to the migration
  guide.
- `WikiStore.raw_path` and `WikiStoreConfig.raw_dir` are gone.
- `lint_wiki(raw_dir=…)` → `lint_wiki(sources_dir=…,
  sources_local_dir=…, repo_root=…, indexed_paths=…)`.
- `search(paths=[])` now means "nothing in scope" and returns no hits;
  `paths=None` still means "the whole root". Callers computing a path
  list by filtering depend on the empty case staying empty.

### Fixed

- **Local material never enters the semantic index.** The vector DB
  stores each chunk's verbatim text and is staged into the same commit
  as the write that triggered it, so indexing local sources would have
  pushed the exact bytes the tree withholds into git through a path
  that looks like a cache. The exclusion is unconditional — it applies
  under `semantic.index: pages+sources`, which now means the *tracked*
  tree only.
- **`scope="all"` no longer hides the local tree.** ripgrep honours
  `.gitignore` while walking a directory, so handing it the repo root
  made "search everything" the one scope that skipped `sources-local/`.
  It now enumerates trees explicitly, which also keeps `.outmem/` and
  `.vectors.db` out of results.
- **`stale_pages()`, `source_refs()`, `rename_page()` and
  `sources_gc()` reach both registries.** Each previously opened "the"
  registry and so was blind to local sources — a revised licensed
  handbook never reported its stale pages, and renaming a page silently
  broke the recorded slug mapping in the one tree whose drift nobody
  can spot in a diff.
- **The pre-commit hook no longer rewrites ingested sources.**
  Frontmatter repair matched "markdown under `wiki/`", which includes
  sources. A source's path embeds its own `sha256[:12]` and the registry
  records the full hash, so repairing one broke content addressing,
  failed lint's provenance-sha check, and invalidated the immutability
  supersession assumes. Reachable on any `git add` of an uncommitted
  ingest, and unrelated to the split — found reviewing for this release.
- **A read-only store no longer writes.** `list_sources()` created
  `wiki/sources/.sources.db` on a wiki with no registered sources,
  writing into the tracked tree from a store contractually forbidden to
  write.

### Note

A page citing a local source records that source's **filename** in its
tracked `provenance:`. A citation is not a redistribution, so this is
the intended trade — but it means `sources-local/` protects
distribution rights, not secrecy. Material whose *name* must stay
private does not belong in either tree.
