# Changelog

Notable changes per release. Versions before 0.10.0 are in the git
history (`git log --grep '^release:'`).

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
