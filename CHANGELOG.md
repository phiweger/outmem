# Changelog

Notable changes per release. Versions before 0.10.0 are in the git
history (`git log --grep '^release:'`).

## 0.18.0

### Changed

- **A `provenance:` citation that names no registered source is now refused at
  the write**, by `write_page`, `extend_page` and `append_page`
  (`UnregisteredProvenanceError`), instead of being accepted and reported by
  `outmem lint` later.

  Provenance is the edge everything else hangs off: `outmem stale` follows it
  to find pages citing a superseded version, `source_citations` turns it into a
  liveness signal, `superseded_ok:` and `finding:` annotate it,
  `provenance-sha-mismatch` compares it. A citation to nothing silently opts a
  page out of all of that while looking, on the page, exactly like a citation
  to something — and a model asked to cite its source will produce a plausible
  `sources/<sha>/document.md` from the tool docstring's own example. The write
  is the moment the author is present and can fix it; lint is a report somebody
  reads later. Same argument 0.15 made for refusing a body that ends in an
  elision, and the same shape of fix.

  **A registry row existing is the criterion, not a file existing.** Citing a
  superseded version is legal — that is what `outmem stale` and `superseded_ok:`
  are for — and a row whose file was later deleted stays a lint concern.
  Validation happens before anything touches disk, so a refused write leaves no
  page file, no regenerated index and no commit. Accepted ref forms are
  `<sha>/file.md`, `sources/<sha>/file.md` and `sources-local/<sha>/file.md`,
  each optionally prefixed `wiki/`; all resolve to the same key. A page with no
  provenance at all is still fine — navigation hubs have none — and
  `extend_page(provenance=None)` leaves the field untouched without
  re-validating it, so a body edit to a page that already carries a dangling
  ref still goes through. `append_page` validates what it merges as a whole:
  one bad ref refuses the call and nothing is appended.

  `allow_unregistered_provenance=True` restores the old behaviour per call, for
  a migration script or an operator who knows. As with `allow_elision`, the
  PydanticAI write tools deliberately do not expose it — an escape hatch in a
  tool argument is an escape hatch a model will pull. Those tools instead hand
  the refusal back to the model as a retry naming the offending refs, the way
  they already do for an incomplete body, so it can call `list_sources` and
  cite a real key or drop the claim.

- **`outmem lint` splits the provenance finding in two.** `stale-provenance`
  now means what its advice always assumed — the registry row is there, the
  file is gone, "restore the source or update the page". The new
  `unregistered-provenance` covers a ref with no row at all, where the remedy
  is to register the source or remove the citation. Both are warnings. Nothing
  refuses a pre-existing page, so lint is what finds these, and it now says the
  right thing about each.

### Fixed

- **A source registered by another process was refused as unregistered.** The
  registry is cached in memory for a store's lifetime — right for reads, but it
  meant a long-lived store's snapshot predated any row registered elsewhere, so
  the new check refused a perfectly good citation and handed back advice ("cite
  one `list_sources` shows") that its own `list_sources` could not satisfy,
  being stale too. That is the deployment the guard was written for: outmem's
  write path behind a server, with ingestion happening in another process. The
  registries are now re-read once before a refusal, so the refusal path pays for
  itself and the common path does not. A genuinely absent source is still
  refused.
- **A citation the filesystem cannot represent raised `OSError`.** `Path.is_file`
  propagates ENAMETOOLONG, so a ref of several hundred junk characters — the
  shape a model invents when it guesses a path — escaped every handler that
  expects an `OutmemError`, killing an agent's turn instead of retrying it.
  Source resolution now treats a path the OS refuses as "does not resolve",
  which also fixes the read path (`get_source`, `read_source`) where the same
  resolver was reachable with any string a model passed.
- **`outmem lint` reported a perfectly good citation as dangling** when the page
  spelled it `wiki/sources/<sha>/file.md` — one of the three forms
  `split_tree_prefix` accepts, and the one `grep_wiki` prints, so an agent that
  grepped and then cited what it saw produced it routinely. The store resolved
  it, `source_citations` attributed the page to the registry row and `outmem
  stale` followed it; lint had a second resolver that did not strip the wiki
  directory and said the file was missing. Lint now normalises the ref the way
  the store does. (Found while splitting the finding above, which would have
  turned a vague wrong answer into a confident one: "no registry row names
  this" about a row that is right there.)

### Compatibility

Pages already carrying dangling refs are untouched; only new writes are
refused. Callers that register first — `outmem ingest`, the ingest agent,
downstream registration scripts — see no change. A caller that wrote refs to
nothing was writing something lint would have flagged, and now hears about it
at the write. Run `outmem lint` after upgrading to see which existing pages
carry `unregistered-provenance`.

## 0.17.6

### Fixed

- **Federated `search_wiki` returned nothing on a corpus that answered the
  question.** The multi-wiki path called `find_similar` directly instead of
  running each wiki's configured retrieval pipeline, which diverged from the
  single-wiki tool three ways: it ignored `retrieval.strategy`, it inherited
  `semantic.similarity_threshold` (0.8, tuned for whole-page duplicate
  detection) where the configured path forces `0.0` because question-vs-chunk
  cosines sit well below it, and it returned raw chunks — including chunks of
  ingested source documents — rather than deduped pages.

  The visible symptom was `(nothing close to …)` on an indexed store with a
  current index. In a grounded-answer system that is the most expensive
  failure available: "we have nothing on that" is a load-bearing answer, and
  it was being given because retrieval had broken.

  `WikiSet.search_pages()` now runs each wiki's own strategy and fuses the
  rankings by Reciprocal Rank Fusion, the method `hybrid` already uses for its
  legs. `search_wiki` returns `[[wiki/slug]]` page citations, matching the
  single-wiki contract.
- **One unindexed wiki blanked semantic search for the whole set.**
  `semantic_available()` is true if *any* wiki has an index, but the fan-out
  called every store and an unindexed one raised — so adding a wiki broke
  search until it was reindexed. Unindexed wikis are skipped.
- **One failing wiki no longer blanks the others.** A wiki whose retriever
  errors is reported as a diagnostic; the rest still answer.

### Changed

- **`search_wiki` works without a semantic index.** It used to refuse; it now
  falls back to bm25 for that query and says so, the same graceful degradation
  the single-wiki tool has always had.
- **The no-match message names the wikis searched and carries diagnostics**,
  so an empty corpus and a retrieval outage are distinguishable.

### Also fixed, from reviewing the above

- **The fusion constant was read per wiki**, inside the loop, so the value
  used to fuse was whichever wiki came last — a silent dependence on registry
  order. It is the set's constant now: a wiki's `rrf_k` governs fusion within
  its own hybrid strategy, which is a different fusion over different inputs.
- **`WikiSet.close()` left the retriever cache populated.** Retrievers hold
  their store and, for bm25, a built FTS5 table — exactly what closing exists
  to release.
- **Page previews were sized by the constant left over from chunk excerpts**
  (400 chars). They match the single-wiki tool's 200 now; the two produce the
  same kind of result and should look the same doing it.

### Note on the tests

The suite could not have caught the original bug. The fixture covering
federated semantic search rewrote `similarity_threshold` to `0.01` so the stub
embedder would produce hits — the exact knob that hid it. A second fixture now
leaves every retrieval setting at its default, and the regression tests run
against that one.

The fusion-constant fix needed the same care: an ordering assertion could not
catch it, because `rrf_k` changes rank only when candidates compete. That test
captures the argument the fusion is actually called with.

## 0.17.5

### Added

- **`registry-undescribed-tag`** — `outmem lint --repo` reports a declared
  audience tag whose `description:` is empty. That string is what `outmem
  repo tags --json` emits and what a host's user database provisions
  against, so an empty one reaches the provisioning UI as a blank while
  the meaning sits in a YAML comment the payload cannot see.

  outmem produced this state itself: `repo add --audience X` writes
  `description: ''` and offers no flag to fill it, and lint then called
  the result clean. A warning, not an error — terse self-explanatory tags
  (`legal`, `hr`) are a legitimate choice.

### Docs

- `docs/multi-wiki.md` draws the line the guidance assumed: comments carry
  rationale for the next reader, `description:` carries what a machine
  consumes.

## 0.17.4

A review round over 0.17.3.

### Fixed

- **`repo import --path-in-repo ../x` bricked the registry.** The path
  was acted on before the parser got to reject it: the wiki was copied
  *outside* the repository, the entry was written, `git add` refused,
  and nothing rolled back — so `wikis.yaml` no longer parsed and every
  command on the repository failed until somebody edited the file by
  hand. `repo add --path ../x` had the milder version: a directory
  created outside the repository, then the entry rolled back.

  Both commands now validate the path first, with the one rule the
  parser uses (`validate_wiki_path`, shared), so a value the registry
  would reject can never be written or acted on. Nothing touches disk
  until everything that can refuse has had its chance.
- **The refusal for a commented registry told `repo import` users to run
  `repo add`**, which would not have done what they wanted. It names the
  command that was run.

### Changed

- `repo add` and `repo import` check for the registry once, through the
  parser, rather than testing for the file and then parsing it; the
  `assert`-based narrowing in those paths is gone in favour of errors
  that survive `python -O`. A listed name's `--path-in-repo` is compared
  after normalisation, so `./wikis/legal/` and `wikis/legal` agree.

## 0.17.3

### Fixed

- **`repo add` and `repo import` flattened a hand-maintained `wikis.yaml`.**
  Both round-tripped the file through the YAML loader, which drops every
  comment and reformats what is left — and committed the result. 0.17.2
  worked around this by shipping a comment-free starter instead of fixing
  it. `wikis.yaml` is the one file a person is meant to keep, so it is
  exactly the file that carries "mirrors the IdP groups" and "see
  ADR-014"; outmem now never rewrites one that has comments.

### Changed

- **`repo add NAME` on a name the registry already lists scaffolds the
  wiki and leaves the file alone.** That is the path for a hand-maintained
  registry: edit the YAML, then `repo add`. It used to refuse with
  "already registered". Passing `--audience`, `--title` or `--path` for a
  listed name is refused rather than silently ignored.
- **For an unlisted name, the entry is written only if the file has no
  comments.** Otherwise the command stops before changing anything — for
  `repo import`, before moving the wiki — and says what to add by hand.
  Detection is exact: every `#` is either inside a loaded scalar or is a
  comment.
- `repo import` of a listed name moves the wiki to the entry's path
  without rewriting the registry.
- The `registry-missing-wiki` and `registry-not-a-wiki` lint messages, and
  the error from opening such an entry, point at `outmem repo add NAME`.

## 0.17.2

A review pass over the multi-wiki work: correctness, typing, and the shape
of the CLI. No new features.

### Fixed

- **`Repo` refused to open a listed directory that is not a wiki.** Before,
  `WikiStore.open` scaffolded `wiki/pages/` and `log/` into it as a side
  effect — the path by which a half-finished `repo add` became something
  that opened, linted clean, and held nothing. A missing directory and a
  present-but-empty one are now two distinct errors.
- **`wikis.yaml` rejects two entries naming one directory**, and a `path`
  of `.` or `""`. Two names for one directory meant `name_at` returned
  whichever came first and the other wiki silently *was* that one; `.`
  names the repository itself, which a walk of *parents* could never find.
- **`outmem lint --repo` opens each wiki read-only.** A lint has no
  business installing a pre-commit hook or creating `.outmem/` as a side
  effect, and it was doing both. A malformed `wikis.yaml` is now a
  `registry-malformed` finding rather than a crash — the linter is what
  somebody reaches for when a repository is misbehaving.

### Changed

- **`outmem lint --repo` is one report.** Findings from every wiki carry
  repo-relative paths (`wikis/legal/wiki/pages/nda.md`), so the path says
  which wiki without a header per wiki, and the exit-code rule is the one
  `outmem lint` already used. `lint_registry` in `outmem.repo` is replaced
  by `outmem.lint.lint_repository`, which returns a `LintReport` like
  everything else in that module.
- **`Repo.wiki`, `wiki_as_operator` and `wikiset` spell out their
  arguments** (`agent_identity`, `remote`, `branch`, `read_only`) instead of
  forwarding `**kwargs` past the type checker. Keyword callers are
  unaffected.
- **`Repo` and `WikiSet` are exported from `outmem`.**
- **The `repo` subcommands no longer accept `--wiki`**, which they ignored;
  the subject of every one of them is the registry. They live in
  `outmem.cli.repo` now, and the starter `wikis.yaml` is plain YAML — the
  comments it carried were dropped by the first `repo add`, which
  round-trips the file through the YAML loader.
- Federated search results carry which wikis truncated (`FederatedSearch`),
  and the read tools use the qualified accessors rather than re-formatting
  names by hand.

### Docs

- `docs/python-api.md` gains the multi-wiki section it was missing;
  `docs/features.md` a pointer; `docs/development.md`'s repository layout
  is current again.

## 0.17.1

### Fixed

- **`outmem repo init` overwrote an existing `.gitignore`** and committed
  the loss. Run in a directory that already had one — the ordinary case
  — it replaced the file wholesale, dropping rules like `__pycache__/`,
  `*.pyc` and `.vectors.db`. Entries are appended one at a time and only
  when absent now, reusing the same conservative helper `outmem init`
  has always used for a wiki's own `.gitignore`; the file is staged only
  if something was actually added to it.
- **`repo init` failed with a raw git error when there was nothing to
  commit.** Re-running a setup command that changed nothing is a no-op,
  not a failure.

## 0.17.0

**Multi-wiki.** Several wikis in one git repository, one per audience.
A wiki is the compartment: within it everything is open, and separation
comes from which wiki a request opens rather than from filtering what a
shared corpus returns. Full contract and host-integration guide in
[`docs/multi-wiki.md`](docs/multi-wiki.md).

### Removed — breaking

- **Restricted content (`restricted:`) is gone.** Per-item labels gated
  audiences inside one wiki; every read path filtered on
  `labels(item) <= mode`. It worked, but its value was entirely
  conditional on being correct, and reaching correct took four
  adversarial review rounds and ~29 fixes — every one of them a
  *filtering* bug, some path that returned content, identity or
  existence without consulting the mode. Under separation nothing
  decides what to hide, so nothing can decide wrong, and keeping the
  label machinery alongside would be an access-control surface
  maintained for no reader.

  Gone with it: the `restricted:` block in `config.yaml`, the
  `restricted:` frontmatter field, `WikiStore.as_viewer`, `Grants`,
  `restrict_page` / `restrict_source` / `compartment_hint`, `outmem
  restrict` and `outmem sources restrict`, the `restricted` column in
  `.sources.db`, the six `restricted-*` lint kinds, and the
  `log/<label-set>/` partitions.

  **If you use `restricted:`**, split the wiki by hand before upgrading
  — one wiki per label — or stay on 0.16.1. There is no automatic
  split: collapsing a label lattice into a partition is lossy, since a
  page labelled `{hr, legal}` has no single home.

### Added

- **`wikis.yaml`** — a repository registry declaring the wikis and the
  audience tags that reach them. A tag is opaque to outmem: the host
  maps its authenticated user to a set of tags and hands them in, and
  outmem does one set-overlap test on names, once, before a store
  exists.
- **`outmem.repo.Repo`** — `wiki(name, *, audience)`,
  `wikiset(audience=…)`, and the separately named `wiki_as_operator`.
  No accessor returns every wiki without an argument. A name an
  audience does not reach fails with the message an unknown name does.
- **Discovery** — `Repo.tags()`, `catalogue()`, `catalogue_for()` and
  `reconcile()`, plus `outmem repo tags --json` and `repo list --json`,
  emitting a versioned payload a provisioning script can read without
  importing Python. The host's user table stores *tags*, never wiki
  names, so wikis can be renamed or moved without touching a user
  record.
- **`outmem.wikiset.WikiSet`** — reads several wikis as one. Names come
  back qualified `wiki/slug`; a bare slug resolves in wiki order and
  reports what it shadowed. `outmem.adapters.wikiset.wikiset_read_tools`
  is the matching PydanticAI palette. Reads federate; writes name one
  wiki.
- **`outmem repo init` / `add` / `list` / `tags` / `audience` /
  `import`**, and `--wiki NAME` alongside `--root` on every subcommand.
- **`outmem lint --repo`** — the registry plus every wiki. Five new
  registry checks and `cross-wiki-wikilink`.

### Changed

- **`WikiStore.root` is split into `root` (content) and `repo` (git).**
  They are the same directory for a standalone wiki, and every existing
  wiki is unaffected — the pre-0.16 test suite passes unedited, which is
  the evidence. `repo_prefix` is the wiki's location inside its
  repository, applied in exactly one place so one wiki cannot name
  another's files.
- **Commit subjects carry the wiki** in a multi-wiki repository:
  `legal/ compact: nda`. A standalone wiki is unchanged.
- **Finding the repository is opt-in.** outmem does not walk up looking
  for `.git`; an ancestor is accepted only when its `wikis.yaml` lists
  the directory. A wiki inside an unrelated repository behaves exactly
  as before.

### Fixed

- **Stage-and-commit is serialised across the repository** with an
  `fcntl.flock`. A commit is two operations with the shared git index
  between them, and `WikiStore._write_lock` is per-store and
  per-process. Measured on three processes writing into three wikis of
  one repository: two to four commits per run carried another wiki's
  paths, and one or two pages `write_page` reported as written were
  absent from HEAD. This also hardens the pre-existing single-wiki
  multi-process case.
- **`resolve_source` canonicalises its path.** The tree-prefix strip was
  textual, so `sources/../../wiki/sources/<rel>` resolved to a real file
  under a key the registry never held — one file reachable under two
  spellings with rows under only one. (Found during the restricted-
  content work and kept; it is independent of labels.)

## 0.16.1

Fixes to the restricted-content boundary shipped in 0.16.0, found by
adversarial review and by a deployment report. **Upgrade if you use
`restricted:`** — one of these is a fail-open.

### Fixed

- **A source restricted by another process could stay readable
  indefinitely** on a wiki with no `.git`. `SourceRegistry` holds an
  in-memory snapshot taken at load, and the code path that serves such a
  wiki was missing the step that drops that snapshot before rebuilding
  the label index — so it re-read the labels this process saw last time
  and cached them under the *new* fingerprint, which meant it never
  healed. An open view both listed the source and read its bytes. Wikis
  with a repository were never affected.
- **Closure now covers log entries and source documents, not just
  pages.** A restricted slug named in an open log entry discloses
  exactly what one named in an open page does, and `append_log` was not
  checking. It is checked at write time now, against the labels of the
  partition the entry lands in, and `outmem lint` verifies the same
  invariant across all three trees — reported as a warning, since a log
  entry is a historical record and a source is frozen bytes.
- **A crash under concurrent reads** on a wiki with no `.git`: the
  cache's fast path could return `None` into a visibility check when
  another thread invalidated the index between the check and the read.

### Performance

- **A wiki with no `.git` no longer rebuilds the label index on every
  visibility check.** The index is keyed on HEAD, so a deployment that
  strips the repository had no token and rebuilt constantly — measured
  on 1200 pages at 648 ms per check against 0.10 ms with a repo, and
  entirely silent. It now caches for 5 seconds and says so once, naming
  the cause; when the window elapses it stats the page tree before
  parsing it, so a corpus that never changes costs one cheap walk per
  window rather than a full parse forever. Source labels can still move
  without a commit and are caught by the registry fingerprint rather
  than the clock. Keeping `.git` in the deployed copy remains strictly
  better and costs a few MB — a depth-1 clone is enough.
- **Packed refs and worktrees are off the subprocess path.** 0.16.0
  read HEAD from `.git/HEAD` but fell back to `git rev-parse` for both
  shapes — 2.1 ms per visibility check against 0.08 ms — and `git gc
  --auto` packs refs on any long-lived server-side repository, moving it
  there without anybody choosing it.

### Changed

- **The "no reachable HEAD" warning says which case it is.** A `.git`
  directory with no HEAD, an unreadable HEAD, and a branch with no
  commits were all told to keep their `.git` while `.git` was sitting
  right there.

## 0.16.0

**Restricted content.** Some wikis hold material only part of an
organisation may see. outmem now gates that deterministically, at the
store layer, before anything becomes prompt text — no instruction, no
system-prompt rule, and no tool description anywhere in the enforcement
path. Content stays open by default, and a wiki that never takes a view
behaves exactly as it did before: every enforcement point tests for a
view first, so an unrestricted store never reaches the label index at
all.

The deployment assumed is the one this was built for: outmem runs
server-side, one company, one repository, employees with no clone
rights. The repository is dark to them, so filtering at the store layer
is genuine access control rather than a display convention. Full
contract, threat boundary and rollout order in
[`docs/restricted-content.md`](docs/restricted-content.md).

### Added

- **`restricted:` in `config.yaml`** — a declared label vocabulary plus
  optional path rules for pages and for ingested sources. Absent by
  default. Unlike every other config block this one is not forgiving: a
  malformed block refuses to open the wiki, and unparseable YAML in a
  file that mentions `restricted:` is fatal, because the forgiving load
  would otherwise open a restricted wiki with access control silently
  off. Withdrawing a label declaration *hides* the content carrying it
  rather than releasing it — one deleted line must not publish a
  corpus.
- **`store.as_viewer(mode=…, grants=…)`** — the object a served request
  should hold. Restricted content is filtered before it becomes prompt
  text, so the model cannot disclose what it was never given. A view
  shares the store's registries, vector store, alias map, write lock
  and label index, so every view sees the same rows and the process
  opens one set of SQLite connections rather than one per request. It
  holds no reference back to the unrestricted store.
- **`outmem ingest --restricted hr`** — label a source at the moment
  someone is holding the document. Every page later compiled from it
  inherits the label, which is the highest-leverage rule in the design.
  Orthogonal to `--local`: that tree is about redistribution rights,
  this is about secrecy.
- **`outmem restrict <slug> --label hr [--cascade]`** and **`outmem
  sources restrict <path> --label hr`** — the operational verbs.
  Restricting is a graph operation, not a field edit: a page that
  visible pages already link to is refused until those links are
  resolved, since the inbound link would still name it in a body its
  readers can see. Neither will remove a label that a path rule or a
  cited source immediately reapplies — reporting success for a
  declassification that did not happen is worse than refusing.
- **Compartment hints.** When a search returns nothing, `search_wiki`
  tells a user who *holds* a label but is not scoped to it that the
  compartment exists and how much it holds. Counts only, per label,
  never for a label they do not hold. The count is a property of the
  corpus rather than of the question, and the hint takes no query at
  all: counting what *this* question matched would be a content oracle,
  since the model writes the question. Without the hint a cleared user
  asking about parental leave gets nothing and never learns to switch.
- **Six lint kinds** (`restricted-link-violation`,
  `restricted-provenance-violation`, `restricted-slug-mentioned`,
  `restricted-label-unknown`, `restricted-frontmatter-unparseable`,
  `restricted-chain-inconsistent`) verifying the invariants against
  content already on disk — the case write-time enforcement cannot
  reach, and the state a wiki is in the day it turns restrictions on.

### Breaking

- **`restricted:` is now a reserved frontmatter key.** It previously
  round-tripped through `extra` as an opaque value. It is now parsed as
  a list of restriction labels, so a page using the key for something
  else changes shape (a bare string becomes a one-item list, an empty
  value is dropped) and a value that is not a label list makes the page
  unparseable — reported by `outmem lint` as `frontmatter-invalid` (and,
  once you declare labels, also as
  `restricted-frontmatter-unparseable`), and hidden from every view
  until it is fixed. A `restricted: [a, b]` value whose entries are not
  declared labels is the quiet one: the page parses and is then hidden
  from every view. **Grep for `restricted:` under `wiki/pages/` before
  upgrading if you used the key.**
- **`rename_page` and `restrict_page` are operator-only** — refused to a
  view, as `import_vault` and `repair_pages` already were. Both write
  files the caller did not name: rename rewrites inbound links across
  the corpus with the new slug as content, and `--cascade` picks its
  targets from the backlink graph. There is no way to label-check a
  write whose targets are discovered rather than named. Neither was ever
  in a model-facing palette, so this affects only a downstream app that
  called them through a view; hold a bare store for administrative
  operations.

### Changed

- **`.sources.db` schema 3 → 4**, adding a `restricted` column.
  Migrated in place on open; NULL on existing rows reads as open, which
  is the same answer the wiki gave before the column existed.
- **Tool calls are traced without their content on a wiki that declares
  labels.** `_log_call` attaches its kwargs to every `LogRecord` and
  Logfire is a handler, so an unredacted `body` was exporting page text
  verbatim to an observability backend. Bodies, titles and log topics
  are withheld (lengths survive), the item *names* with them — a slug
  can be as disclosing as a filename — and a refusal's message is
  reduced to its exception type, since `IncompleteBodyError` quotes the
  body it refused. A wiki that declares no labels keeps byte-identical
  traces, so eval recorders reading `record.tool_call` are unaffected.
- **`build_consult_wiki` accepts an open store**, so a caller can pass a
  view. It previously always opened its own from a path, which would
  discard a caller's mode and grants.
- **Caches derived from page content are keyed on the corpus, not on
  HEAD.** BM25 snapshots page bodies at construction, so a retriever
  built before a page changed kept answering from the text it had — a
  staleness bug that predates this work and became a fail-open one with
  it. HEAD alone is not enough: a local-tree ingest, and a re-ingest
  that only sets labels, both write the registry and commit nothing, so
  the token carries the registries' fingerprints as well.
  `store.corpus_token()` exposes it, and is `None` for a wiki that
  declares no labels — which is how such a wiki keeps its old cache
  lifetime and skips a `git rev-parse` per query.
- **`page_history` and `topic_evolution` leave the served palette when a
  view is in play**, and the store refuses them. Both are answered by
  git, which knows nothing about labels, while the label index describes
  only the current commit. The system prompt and the injected
  `evolution` skill follow the palette, so a restricted session is no
  longer told to call tools it does not have.
- **A restricted session writes `log/<label-set>/<date>.md`.** The open
  mode is unpartitioned, so a wiki with no restrictions has nothing to
  migrate.
- **`store.corpus_token()` and the label index read HEAD from
  `.git/HEAD` rather than by forking `git rev-parse`.** A visibility
  check on a view was paying ~2 ms of subprocess per call; it is now
  two `stat`s. Unusual repository shapes (worktrees, packed refs) still
  fall back to git, so the token is never weaker — only cheaper.
- **`add_source` takes the write lock**, like every other
  commit-producing path. The registry was already safe across processes
  — SQLite serialises the writers — but the git half was not: two
  concurrent ingests interleaving between `git add` and `git commit`
  raced on the index and one of them failed.

## 0.15.0

**A page is now checked for being all there.** Reported from the same
~1200-page clinical wiki: a turn's output budget binds, and the model
does not necessarily fail — it emits a schema-valid `write_page` whose
body stops early and marks the cut with an ellipsis. Provenance,
hashes, links, and the index are all correct, so nothing downstream can
tell that page from a finished one. 181 such cuts across 75 pages, ~226k
characters, undetected for months, with the missing content sitting in
registered sources the whole time. outmem enforced every *structural*
invariant and only ever *asked* for completeness.

### Added

- **`append_page`** — add a section to an existing page, keeping what is
  there. The counterpart `extend_page` never had: `extend_page`
  *replaces* the body, so building a page in sections meant re-emitting
  everything written so far on every call, and the last call still had
  to hold the whole page in one turn — the very budget pressure that
  causes truncation. Appending removes it. Provenance here is additive
  (deduped by source path) so a section drawn from a new source can cite
  it without restating the page's earlier citations. Available as a
  tool, on `WikiStore`, and as `outmem append`.
- **Write paths refuse a body that stops early.** `write_page`,
  `extend_page`, and `append_page` raise `IncompleteBodyError` when the
  body ends at an elision marker, and the tools turn that into a
  `ModelRetry` naming `append_page` as the place to put the overflow —
  the first use of `ModelRetry` in outmem, and the only error handed
  *back* to the model rather than returned as advisory text after the
  call already committed. Banning the marker is only safe because the
  message names the alternative. A tool-output marker
  (`⟪ outmem: … ⟫`) copied into a body is refused the same way — that
  page would be built on material outmem itself declined to show. See
  the escape-hatch note below.
- **Three lint kinds that read the body as content**, for corpora that
  already have the damage: `truncated-page` (with a file line number and
  the offending line quoted), `tool-output-in-page` (an outmem
  withheld-content marker copied into a page), and `declared-omission`
  (an `omitted:` frontmatter note — deliberate scoping is fine, and is
  reported so a declared gap cannot be used to make a short page look
  finished). `omitted:` needs no schema; unknown frontmatter keys already
  round-trip. There is deliberately no `omitted` tool argument.
- **`AskResult.budget_truncated_writes`** and an unconditional CLI
  warning when a model turn ended because it ran out of output room
  while calling a write tool. This is the case the model did *not* mark,
  the only one that survives without its cooperation.

### Fixed

- **outmem no longer emits the elision vocabulary it refuses.** Four
  places announced *withheld content* with a bracketed marker — the
  `read_source` cap (which lands in the writing agent's context at
  exactly the moment it is reading source material), the `grep_wiki`
  cap, the rerank gate's splice (literally `[…]`, the banned string),
  and the optimizer's `read_page` diagnostic. All four now use one
  out-of-band sentinel, `⟪ outmem: … ⟫`, chosen so the ban and the
  notice can never collide; a test pins that every note outmem emits
  survives its own detector.

- **Page writes are serialised.** `write_page`, `extend_page`,
  `append_page`, `append_log`, and `rename_page` each read the current
  state, rewrite files, regenerate the index, and commit — and
  PydanticAI runs a response's tool calls concurrently. Since the
  guidance is now explicitly "one `append_page` per section", parallel
  appends to one page are the ordinary path: unguarded, four of them
  lost sections outright and raised `cannot lock ref 'HEAD'` and
  half-read-file `FrontmatterError`. A per-store re-entrant lock now
  spans read-through-commit.

### Notes on two design decisions

**Every caller has an escape from the elision guard, but not the same
one.** A human overrides in one step (`--allow-elision`,
`allow_elision=True`, or a reviewer edit under the HITL approval gate).
A model has no argument at all — its only route is to submit the
identical body again after being handed the refusal. A flag a model
could set becomes a checkbox it learns to tick; re-sending unchanged
costs a round trip and distinguishes "this text is right" from "I ran
out of room", since a truncating model adds content rather than
repeating itself. Allowances are keyed by the body text, not the slug:
an adjudication is about the words. Either way the page is still
reported as `truncated-page` — the bargain is bounded damage and a
visible record, not silence.

**The write guard yields rather than deadlocking.** The detector is a
fallible heuristic — which is why lint reports it at WARNING — so making
it a hard block risked the opposite failure: a false positive on a
legitimate quotation, the model re-sending the same correct body, and
the turn dying with zero commits once the retry budget ran out. A second
*identical* submission is therefore accepted and left for lint to
report, and the refusal message says so. One retry to fix a real
truncation; no way for one misread quotation to cost a whole turn.

The detector keys on **position, not vocabulary**: a marker counts as a
cut only when nothing but closing punctuation follows it on its line.
`"die Therapie [...] wird empfohlen"` is the standard way to shorten a
quotation and is not flagged, nor is anything in a blockquote, in code,
or used as link display text — those are the false positives that get a
completeness check switched off.

## 0.14.0

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
  though per-token pricing is unchanged. To keep the anti-truncation
  headroom `DEFAULT_MAX_TOKENS` exists for, the agent and
  `consult_wiki` budgets rise 16384 → 20480 — just under the ~21.3k
  ceiling the Anthropic SDK enforces for non-streaming requests. The
  rerank/HyDE gate stays on `claude-haiku-4-5`, which is still the
  current Haiku.

- **The rerank gate now sees evidence, not just the page opening.**
  A candidate excerpt used to be `body[:2000]` — frontmatter stripped,
  so the page's own title and tags never reached the gate, and a page
  shortlisted by lexical/bm25 *because* the query terms occur in it
  could show the gate an intro containing none of them. Combined with
  the gate's no-false-positives instruction, long pages were
  systematically under-selected. Excerpts now carry the page's
  `<title> — <tags>` line (the `embed_frontmatter` format, but always
  on: the gate prompt is ephemeral, nothing to re-embed), and long
  pages are spliced: the opening (identity) plus a window around the
  densest query-term cluster (evidence), prefixed with the window's
  markdown heading path (via the fence-aware outline parser, so a
  `# comment` in a code block never becomes a section label). Head-only
  when the page fits, nothing matches, or the match already sits in
  the opening. The optimizer's `read_page` diagnostic tool gets the
  same title/tags line and now says when a page was truncated.
  Behavior change to retrieval — worth re-running `outmem optimize`
  on tuned wikis.

### Fixed

- **Query tokenization is Unicode-aware.** `_keywords` — the shared
  extraction behind `lexical`, `bm25`, and the rerank gate's match
  window — split on ASCII alphanumerics, so umlauted terms fragmented:
  `häufig` degraded to the junk substring `ufig`, `hämolytisch` to
  `molytisch` — fragments that match unrelated words and mis-anchor
  windows. Tokens now keep any Unicode word character (underscores
  still separate), and the stopword set gains the German 80-20, which
  the fix makes newly relevant: `für`/`über` previously shattered into
  droppable shrapnel, and ASCII German function words (`und`, `der`,
  `die`…) always slipped through as search terms.

- **One-shot LLM calls no longer opt into Anthropic automatic prompt
  caching.** The rerank gate, HyDE, and `generate_bank`'s question
  generation sent `anthropic_cache: True`, which places the server-side
  cache breakpoint after the *whole* prompt — on calls whose prompt is
  unique every time (a fresh query + candidate set, a different page
  body), that wrote the entire prompt to cache at the 1.25× write rate
  with zero chance of a later read. Telemetry from a production wiki
  showed the signature plainly: `cache_creation ≈ input, cache_read = 0`
  on every rerank call — a flat ~25% surcharge on the dominant cost item
  of a `search_wiki` call. One-shot callers now use
  `ANTHROPIC_CACHE_ONESHOT` (system-prompt marker only, harmless);
  multi-turn agents (agent runtime, `consult_wiki`, the optimizer) keep
  automatic caching, which is what makes their growing conversations
  cheap. `ANTHROPIC_CACHE_SETTINGS` was renamed to
  `ANTHROPIC_CACHE_ONESHOT` so the name states the call shape it is
  safe for.

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
