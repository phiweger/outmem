# Restricted content — access control for a served outmem wiki

Status: **implemented** in 0.16.0. This file is the design record —
what was decided and why, including the alternatives that were rejected.
For how to *use* the feature, see
[`docs/restricted-content.md`](../docs/restricted-content.md).

Several things landed differently from the text below. Each was found
while building or reviewing it, and each is stricter than specified.

**§2.4's hint takes no query.** Counting the items that matched *this
question* and fell outside the mode is a content oracle, because the
model writes the question: "twelve" then "eleven", compare the counts,
and a fact has been read out of a restricted page without retrieving
it. The count is now a corpus property — how many pages a session
scoped to that label would gain — which is the same for every question.

**§8.3's two verbs are operator-only, not grant-gated.** `rename_page`
rewrites inbound links across the corpus and `restrict_page --cascade`
picks its targets from the backlink graph, so both write files the
caller did not name, with content the caller chooses. A write whose
targets are discovered rather than named cannot be label-checked, and a
refusal that has to list the referrers cannot be shown to a view.
`outmem sources restrict` was added as the source counterpart.

**Aliases derive labels only for names no live page occupies.** §5.2
says aliases inherit; applied to an occupied name it inverted, letting
an HR page's `aliases:` relabel an open page and open it to editing.

**Withdrawing a declaration hides content rather than releasing it.**
Skipping the index when `restricted.labels` is empty made one deleted
line publish the whole corpus. An undeclared label already collapsed to
DENY; the same rule now covers an emptied block.

**Source keys are normalised, not enumerated.** `resolve_source`
resolves against the filesystem, so `sources/./x` reached a file that a
map of literal keys did not cover — and a miss read as open.

**Path rules apply at lookup, not only at build.** The page map holds
only live `.md` files, so a `.txt` under a restricted namespace, or a
slug named in `AGENTS.md` before its page exists, read as open.

**`restrict_page` refuses a no-op declassification**, rather than
committing a label removal that a path rule or a cited source
immediately reapplies.

**`read_page` renders from the page object the store returns** rather
than re-reading the file, because re-reading served the stored
`wiki/index.md` — the one file that lists every slug — past the
per-viewer render in §7.1.

## 1. Scope

outmem is deployed **server-side**. A company runs one wiki in one git
repository; employees do **not** have clone rights and never touch the
filesystem. They interact through an application that authenticates
them and calls outmem on their behalf.

Content is **open by default**. Any page or source may carry one or
more *restriction labels*; content with no label is visible to
everyone. A user holds a set of labels, and sees only content whose
labels they hold.

Denial is **hiding**, not redaction: a user must not learn that
restricted content exists.

### 1.1 What this protects

Queries issued through the served tool surface. A user without the
`hr` label cannot obtain HR content, its titles, its slugs, its source
filenames, or evidence of its existence, through any tool call.

### 1.2 What this does not protect

- **The substrate.** The server operator, filesystem access, backups,
  and the git repository itself are outside this boundary and are
  protected by your infrastructure controls.
- **`.vectors.db`.** Restricted chunk text is stored verbatim in the
  semantic index (see §7.3). The file is therefore as sensitive as the
  most restricted item in it.
- **Inference.** An open page compiled from restricted sources may
  disclose them by implication. §3.4 makes such a page restricted by
  default, but a human deciding to publish a summary is an editorial
  judgment no mechanism here checks.
- **Anything a cleared user chooses to repeat** outside the system.

### 1.3 Non-goal: LLM-mediated enforcement

No decision in this spec may depend on a model behaving correctly.
Restricted content is filtered **before** it becomes prompt text, so
the model cannot disclose what it was never given. There is no
instruction, system-prompt rule, or tool description anywhere in the
enforcement path.

## 2. Model

### 2.1 Labels

A label is an opaque string (`hr`, `legal`, `board`). Labels are
**flat**: there is no hierarchy, no ordering, no implication. All
declared labels are listed in `config.yaml`; an undeclared label is a
lint error (§11), because a label nobody holds makes content invisible
to everyone, silently.

A page or source carries a set of labels. The empty set means open.

### 2.2 Grants

For each label a user holds one of three states:

| state | meaning |
|---|---|
| none | the label's content does not exist, as far as this user is concerned |
| `read` | may see content carrying the label |
| `read, write` | may also create and modify it |

`write` without `read` is **invalid** and MUST be rejected at grant
resolution. `extend_page` replaces a whole body and `append_page` must
avoid duplicating existing content; neither is safe without reading.

Grants are supplied by the calling application from its own
authentication. outmem does not model users, sessions, or identity.

### 2.3 Session mode

Every request runs in a **mode** `S`: a set of labels, chosen by the
application before the run and **immutable for its duration**.

`S` MUST satisfy `S ⊆ {labels the user may read}`.

`S` MUST NOT be derivable from, or modifiable by, model output. It is
bound at view construction. A mode the model could change would let it
read under one mode and write under another, which defeats §3.2.

**`S` defaults to `∅`, and restricted content is opt-in.** A session
that does not name a compartment retrieves open content only, exactly
as outmem behaves today: no restricted item is fetched, so none can
influence the answer or the writes. A caller opts in by naming the
compartment it intends to work in.

There is deliberately **no automatic escalation** — no mechanism by
which retrieving a restricted item widens `S`, and no session taint.
Two taint designs were considered and rejected:

- *Taint to read-only* — the first restricted item retrieved makes the
  session read-only. Safe, but it hard-fails the turn (`ask()` raises
  `WritebackError` on zero commits — `specs/spec.md` §9) and does so
  nondeterministically, depending on whether the retriever happened to
  surface a restricted page the user never asked about.
- *Taint to the write target* — writes are forced into the union of
  labels read. Also safe, but one incidentally-retrieved page
  permanently over-classifies an unrelated note, and content drifts
  into compartments where it does not belong.

Both over-react to the same accident. The precise signal — whether
restricted content was actually *used* — is a model judgment, which
§1.3 forbids relying on. So the trigger is removed rather than
refined: **content that is never retrieved cannot taint anything.**

### 2.4 Grants and mode are different things

**Grants are entitlement; mode is scope.** A user either holds `hr` or
does not. Within what they hold, mode says which compartment this
particular session is working in.

The hiding requirement (§1) is a property of **grants**, not of mode.
For a user who does not hold `hr`, HR content must be undetectable.
For a user who *does* hold `hr` but is running in mode `∅`, telling
them that matches exist elsewhere discloses nothing they are not
already entitled to see.

Retrieval SHOULD therefore return, alongside open results, a
**count-only, grant-gated hint**: `4 more results in hr`. It MUST name only labels
the user holds, MUST NOT include titles, slugs, excerpts, or anything
else derived from the matched items, and MUST be absent entirely for a
user without the grant.

This is what makes the opt-in default workable. Without it, a cleared
user asking about parental leave gets nothing and never learns to
switch compartment; with it, discoverability is preserved for exactly
the people entitled to it.

## 3. Rules

Four rules. Everything else in this spec is their consequence or their
enforcement.

### 3.1 Visibility

> An item is visible in mode `S` **iff** `labels(item) ⊆ S`.

Open content (`labels = ∅`) is visible in every mode, since `∅ ⊆ S`
for all `S`. Open-by-default is not a special case; it falls out of
the subset relation.

Note the consequence: mode `{hr}` sees open **and** HR content, not HR
alone. Restricting the session to HR-only would be stricter than
necessary — the unsafe direction is read-HR-then-write-open, whereas
reading open while writing to HR is a write *up*. An HR-only session
would also leave the agent composing HR pages with no access to the
company glossary.

### 3.2 Write

> In mode `S`, a write may target an item **iff** `labels(item) == S`,
> and the user holds `write` on every label in `S`.

Equality, not containment, and it is forced from both directions:

- `labels ⊆ S` — you must be able to see what you are modifying.
- `labels ⊇ S` — you must not carry facts from a more restricted
  context into a less restricted item (no write-down).

Consequences worth stating:

- In mode `∅` you may write only open items, however many labels you
  hold. Holding `write hr` does not let you edit an HR page from an
  open session; you must start a session in mode `{hr}`.
- To write a page labelled `{hr, legal}` you must run in mode
  `{hr, legal}`.
- Fact-laundering is blocked by the same rule: citing an HR source in
  mode `∅` makes the page's computed labels `{hr}` (§3.4), which is
  `≠ ∅`, so the write is refused. No separate mechanism is needed.

**A new item's labels default to `S`.** Without this, writing an HR
page in mode `{hr}` that happens to cite only open sources would
compute `∅ ≠ {hr}` and be refused unless someone remembered an
explicit label. Defaulting to the mode removes that friction and is
fail-safe: the default is always the *more* restricted option, and it
can only be narrowed by the privileged operation in §8.3.

### 3.3 Closure

> An item's labels MUST be a superset of the labels of everything it
> references — wikilinks, provenance entries, and aliases.

Equivalently: if you can see a page, you can see everything it points
at. This is what makes hiding real. Filtering the page list achieves
nothing if an open page's body contains `[[hr:severance-policy]]`,
because `read_page` on the open page hands over the slug.

Maintained by two operations, not one:

- **At write time** (§8.1): a write whose body links to a
  more-restricted page is refused.
- **At restriction time** (§8.3): restricting an item that visible
  items already reference is refused until the references are resolved.

Verified continuously by lint (§11).

### 3.4 Inheritance

> A page's labels include the union of the labels of every source it
> cites.

Restricting a document at ingest automatically restricts every page
ever compiled from it. This is the highest-leverage rule in the spec,
because ingest is the one moment when someone is holding the document
and knows what it is.

It also does double duty for §3.3: a source's `rel_path` embeds its
original filename, so `sources/policy/9b3d0d/severance-plan-2026.md`
in an open page's `provenance:` is disclosure even when the file
itself is unreadable.

## 4. Deployment shape

```python
store = WikiStore.open("/srv/wiki")              # full access, server-side
view  = store.as_viewer(mode={"hr"}, grants=g)   # per request
tools = wiki_read_tools(view)                    # model never sees `store`
```

`as_viewer` returns a store whose `_mode` and `_grants` are set. The
unrestricted `store` MUST NOT be reachable from `view` through any
public attribute. The existing closure pattern already gives the
required property: `wiki_tools(view)` binds the view at construction,
so no tool takes a clearance argument the model could set.

## 5. Label resolution

### 5.1 Sources of labels

A page's labels are the union of:

1. **Explicit** — `restricted: [hr]` in frontmatter. Primary
   mechanism; orthogonal to the wiki's topic taxonomy.
2. **Path rules** — `restricted_paths: {"hr:*": [hr]}` in config.
   A safety net that cannot be forgotten: a new page under `hr:` is
   restricted whether or not anyone edited its frontmatter.
3. **Inherited** — the union of its cited sources' labels (§3.4).

Explicit labels exist because forcing restriction onto the slug
namespace would make taxonomy and security the same axis; content
whose sensitivity does not follow its topic hierarchy would require
reorganising the wiki to secure it. Path rules exist because explicit
labels can be forgotten. Both are needed.

### 5.2 Fail-closed cases

- **Unparseable frontmatter.** If a page's frontmatter cannot be
  parsed, its explicit labels cannot be read. Such a page MUST be
  treated as denied to every mode, and reported by lint. Resolving to
  "no labels" would be fail-open.
- **Undeclared label.** Content carrying a label absent from
  `config.yaml` is denied to every mode (nobody can hold it) and
  reported by lint.
- **Aliases** inherit the labels of the page they resolve to.
  `resolve_slug` MUST NOT resolve an alias to a hidden page.

### 5.3 Sources

Source labels live in the registry, not in a tree and not in the path:
`.sources.db` gains a `restricted TEXT` column holding a JSON array.
`SCHEMA_VERSION` 3 → 4; `_migrate` handles existing registries.

```bash
outmem ingest severance-plan.md --into policy --restricted hr
```

Path rules apply here too, on the `--into` subdirectory:
`restricted_sources: {"hr/*": [hr]}`.

`sources-local` and `restricted` are **orthogonal axes, not levels**.
The local tree is about redistribution rights; `restricted` is about
secrecy. A source may be local-and-open (a licensed handbook everyone
may read but nobody may republish) or tracked-and-restricted (an
internal memo). Conflating them weakens both.

The agent cannot create sources — `add_source` is not in any tool
palette — so the labelling decision is always made by a person holding
the document.

## 6. Enforcement architecture

### 6.1 Where the check lives

Enforcement is on `WikiStore` itself, via `_mode` / `_grants` fields
(`None` = unrestricted), **not** on a wrapper that overrides methods.
One class, one predicate, and every method is covered at the point the
check is added. `as_viewer` returns a shallow copy.

### 6.2 The label index

`slug → frozenset[label]` and `rel_path → frozenset[label]`, computed
once per store open.

It MUST carry a validity token so a multi-worker deployment cannot
serve a stale index after another worker restricts something. **HEAD
sha is the token** — every outmem write produces a commit, so a moved
HEAD is exactly the invalidation signal.

### 6.3 The enumeration test

A test reflects over `WikiStore`'s public methods and asserts each
appears in exactly one of:

- `_VISIBILITY_ENFORCED` — with a test proving a restricted item is
  filtered from its result;
- `_NO_CONTENT` — returns no item content, identity, or existence.

Adding a public method fails the suite until it is classified. This is
the mechanism that keeps §7 complete as the codebase changes; the tool
palette already uses the same exhaustive-assertion pattern.

## 7. Per-path requirements

### 7.1 Reads

Every method returning content, identity, or existence:

| method | requirement |
|---|---|
| `read` | denied → raise the **byte-identical** error to a nonexistent slug |
| `exists` | `False` for hidden |
| `list_slugs`, `index_tree` | omit hidden; a namespace with no visible pages does not appear |
| `resolve_slug` | MUST NOT resolve an alias to a hidden page |
| `unreadable` | filter; unparseable pages are hidden from all (§5.2) |
| `search` | filter result rows by path, all scopes |
| `backlinks` | filter the returned list |
| `source_citations`, `provenance_annotations`, `provenance_findings`, `stale_pages` | filter by both page and source labels |
| `list_sources`, `get_source`, `read_source` | filter; `read_source` denial identical to nonexistent |
| `semantic_find_similar` | filter by chunk `rel_path`; see §7.3 |
| `read_agents_md` | MUST NOT describe restricted namespaces |

The `index` slug MUST be rendered live from the viewer's visible set,
never served from `wiki/index.md` on disk.

### 7.2 Palette reduction

`page_history` and `topic_evolution` are **removed from the served
palette**. `topic_evolution` returns raw `git log -p` diffs, so a page
restricted today but open last month would hand over its old body.
`store.steering()` is worse — it feeds `render_system_prompt`, so
commit subjects like `compact: hr:severance-policy` reach *every*
prompt regardless of who is asking.

Steering injection MUST be gated on the mode. Gate it on the **mode,
not the user**, so the system prompt is identical for all users in one
compartment and prompt caching still works.

Every tool not exposed is a bypass class that needs no code, no test,
and cannot regress. Shrink before filtering.

### 7.3 Retrieval

**Over-fetch is a security requirement.** Filtering after a fixed
top-k makes the result count an existence oracle: eight results for a
cleared user and three for an uncleared one reveals that five
restricted items sit near that topic. Retrieval MUST fetch until `k`
*visible* results are found, or a bounded ceiling is reached.

**Filter before the rerank gate.** The gate ships candidate excerpts
to an LLM provider. Restricted text reaching it has already left the
deployment, whatever the gate then returns. Filtering MUST happen at
candidate generation, not on the gate's output.

**The semantic index contains restricted content.** Restricted pages
and sources are indexed normally and filtered at query time, so
cleared users retain full semantic recall. The consequence is §1.2:
the DB file is as sensitive as its most restricted item.

**Compartment hints** (§2.4) are computed from the same over-fetch: a
count of items that matched the query but fall outside `S`. The count
MUST be broken down per label and filtered to labels the user holds —
a user holding `hr` but not `legal` sees the HR count and no
indication that anything in `legal` matched at all. An aggregate
"N more results" across all compartments would leak the existence of
compartments the user does not hold.

Timing differences from over-fetch are a theoretical oracle and are
out of scope.

### 7.4 Caches

Any cache spanning requests MUST be keyed by mode, or recomputed. The
backlink cache is currently global and on disk.

### 7.5 Logging

`_log_call` attaches `dict(kwargs)` — including a full page `body` —
to every `LogRecord`, and Logfire is a handler. Tool-argument logging
MUST redact content fields, or restricted page text is exported to the
observability backend.

`build_consult_wiki` opens its own store and MUST take a view instead;
as written it is a complete bypass.

## 8. Writes

### 8.1 Ordinary writes

`write_page`, `extend_page`, `append_page` MUST refuse unless
`labels(target) == S` (§3.2) and the user holds `write` on every label
in `S`. Refusal raises `RestrictionError` at the **store** layer, so
the CLI and Python API are covered, not only the tool palette.

A write whose body links to a page with labels ⊄ the target's labels
is refused (§3.3).

Slug collisions with hidden pages are refused with a generic message.
This is a bounded oracle — a user learns "something blocks this slug"
without learning what — and is mitigated by preferring path rules for
namespaces that contain restricted content.

### 8.2 The log

`append_log` writes to open `log/<date>.md`. In mode `S` that is a
write-down, and mandatory writeback (`specs/spec.md` §9) actively pushes the
agent there when nothing else was warranted. Logs MUST therefore be
partitioned per mode: `log/<label-set>/<date>.md`.

A session that cannot write at all is not viable, because `ask()`
raises `WritebackError` on zero commits. Read-only sessions MUST be an
explicit service-level mode that does not demand writeback.

### 8.3 Privileged operations

Both require a `declassify` grant, which MUST NOT appear in any
model-facing palette:

- **Removing a label** from a page or source.
- **`rename_page`** where the rename changes the page's effective
  labels. Under path rules `hr:x → notes:x` is silent declassification
  through an innocuous-looking tool.

`outmem restrict <slug> --label hr` is the operational verb. It MUST
report inbound references from items that would become non-conforming
under §3.3 and refuse until they are resolved, or cascade with
`--cascade`. Restricting is a graph operation, not a field edit.

### 8.4 Ingestion

`record_ingestion` is the only source-touching write in the palette,
and its `prompt` field is agent-written free text stored in the
registry. It MUST satisfy `labels(source) == S`.

## 9. Errors

`RestrictionError(OutmemError)`, raised at the store layer. Message
text MUST NOT name a hidden item. Denials of *reads* do not raise
`RestrictionError` — they raise or return exactly what a nonexistent
item would, since distinguishable errors are an existence oracle.

## 10. Configuration

```yaml
restricted:
  labels: [hr, legal, board]     # declared set; unknown labels are a lint error
  paths:                          # safety net for pages
    "hr:*": [hr]
  sources:                        # safety net for ingested documents
    "hr/*": [hr]
```

## 11. Lint

Mirrors the existing containment checks — the guarantee is only worth
stating if something verifies it.

| kind | severity | condition |
|---|---|---|
| `restricted-link-violation` | error | a visible page references a more-restricted page |
| `restricted-provenance-violation` | error | a page's labels ⊉ its cited sources' labels |
| `restricted-slug-mentioned` | warning | a restricted slug appears as prose in a less-restricted page |
| `restricted-label-unknown` | error | a label absent from `restricted.labels` |
| `restricted-frontmatter-unparseable` | error | labels unreadable, so the page is denied to everyone |
| `restricted-chain-inconsistent` | error | versions under one `document_key` carry different labels |

The last is outmem-specific and easy to miss: re-ingesting v2 of a
restricted document without `--restricted` leaves the document key
with an open current version, and `outmem stale` then quietly points
pages at it. Silent declassification.

## 12. Rollout

1. **Palette reduction** (§7.2). No new security code.
2. **Read boundary** — label index, `as_viewer`, primitive-level
   enforcement, enumeration test, per-viewer index rendering, logging
   redaction, `consult_wiki` threading.
3. **Retrieval** — over-fetch, pre-gate filtering, cache keying.
4. **Write boundary** — `RestrictionError`, closure enforcement,
   privileged operations, per-mode logs.
5. **Verification** — lint checks, and a documented threat model.

Phases 1–3 are complete and useful on their own: they make restricted
content unreachable through reads. Phase 4 prevents its creation in
the wrong place.

## 13. Residual risk

**Remembered facts.** Inheritance catches what an agent *cites*. In
mode `S` an agent may only write items labelled exactly `S` (§3.2),
which confines the damage to the compartment it read from — so the
write-down path is closed structurally rather than by trust. What
remains is a cleared human repeating restricted content into an open
session by hand, which is outside any mechanism here.

**Bounded existence oracles.** Slug-collision refusals (§8.1) and
retrieval timing (§7.3) leak a small amount. Both are documented
rather than eliminated.

**Availability, not confidentiality.** The opt-in default (§2.3) means
a cleared user who never switches compartment will not see content
they were entitled to. This is a deliberate trade: the alternative —
retrieving restricted content by default — reintroduces the taint
problem the default exists to remove. The grant-gated hint (§2.4) is
the mitigation, and the reason it is specified SHOULD rather than
MAY: an opt-in default without a hint is a wiki whose restricted half
is unfindable even by the people it was restricted *to*.
