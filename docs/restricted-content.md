# Restricted content — access control for a served wiki

Some wikis hold material that only part of the organisation may see.
outmem gates that **deterministically**, at the store layer, before
anything becomes prompt text. No decision here depends on a model
behaving correctly: there is no instruction, no system-prompt rule, and
no tool description anywhere in the enforcement path.

Content is **open by default**. A wiki that never takes a view behaves
exactly as it did before this feature existed and pays nothing for it —
every enforcement point tests for a view first, so an unrestricted store
never reaches the label index at all.

## The deployment this assumes

outmem runs **server-side**. One company, one wiki, one git repository.
Employees do not have clone rights and never touch the filesystem —
they interact through an application that authenticates them and calls
outmem on their behalf. The repository is dark to them, so the process
boundary is real and filtering at the store layer is genuine access
control rather than a display convention.

```python
store = WikiStore.open("/srv/wiki")               # server-side, full access
view  = store.as_viewer(mode={"hr"}, grants=g)    # per request
tools = wiki_read_tools(view)                     # the model never sees `store`
```

`as_viewer` returns a store that filters. Its lazily-opened resources —
both source registries, the vector store, the alias map — live on one
object shared with the original, so every view sees the same registry
rows and the process opens one set of SQLite connections rather than one
per request. The write lock and the label index are shared for the same
reason. It holds no reference back to the unrestricted store, so nothing
downstream can climb out.

### What this protects, and what it does not

Protected: everything reachable through the served tool surface. A user
without the `hr` label cannot obtain HR content, its titles, its slugs,
its source filenames, or evidence that any of it exists, through any
tool call.

Not protected, and out of scope by design:

- **The substrate.** The server operator, filesystem access, backups
  and the git repository are yours to protect with infrastructure
  controls.
- **`.vectors.db`.** Restricted chunk text is stored verbatim in the
  semantic index so cleared users keep full recall. The file is as
  sensitive as the most restricted item in it.
- **Inference.** An open page compiled from restricted sources can
  disclose them by implication. Inheritance (below) makes such a page
  restricted by default, but a human deciding to publish a summary is
  an editorial judgment no mechanism here checks.
- **Anything a cleared person repeats** outside the system.
- **The tool log, if you export it.** Tool calls are traced with their
  arguments; content fields are redacted, but the *references* are not,
  because a trace that does not say which page was read is not a trace.
  With Logfire enabled that record leaves the deployment and says which
  items a session touched, and a source's path embeds its original
  filename. That is a deployment choice about an observability backend,
  not something a filter can take back.

## Labels

A label is an opaque string: `hr`, `legal`, `board`. Labels are **flat**
— no hierarchy, no ordering, no implication. `hr` does not contain
`hr-payroll`; they are two unrelated strings.

Labels use the slug-segment grammar: lowercase ASCII, single hyphens.
`HR` is an error where you type it, rather than a second label nobody
holds and content nobody can see.

An item carries a set of labels. The empty set means open.

```yaml
# config.yaml
restricted:
  labels: [hr, legal, board]     # the declared vocabulary
  paths:                          # safety net for pages
    "hr:*": [hr]
  sources:                        # safety net for ingested documents
    "hr/*": [hr]
```

Everything about this block is fail-closed, and it is deliberately
**not** covered by config.yaml's usual forgiving load. Every other
setting degrades a feature when it is wrong; this one degrades a
boundary. A malformed block refuses to open the wiki, and unparseable
YAML that mentions `restricted:` is fatal rather than silently dropping
the file and opening with access control off.

A label used anywhere but absent from `restricted.labels` hides its
content from **everyone**, including the people it was meant for.
`outmem lint` reports it.

## Where a page's labels come from

Three sources, unioned:

1. **Explicit** — `restricted: [hr]` in the page's frontmatter. The
   primary mechanism, and deliberately orthogonal to the wiki's topic
   taxonomy: forcing restriction onto the slug namespace would put
   taxonomy and security on the same axis, so content whose sensitivity
   does not follow its topic hierarchy would need the wiki reorganised
   to secure it.
2. **Path rules** — `restricted.paths` in config. The safety net that
   cannot be forgotten: a new page under `hr:` is restricted whether or
   not anybody edited its frontmatter. `hr:*` covers the namespace root
   page `hr` too.
3. **Inherited** — the union of the labels of every source the page
   cites.

Inheritance is the highest-leverage rule here, because ingest is the
one moment when a person is holding the document and knows what it is:

```bash
outmem ingest severance-plan.md --into policy --restricted hr
```

Every page ever compiled from that document is restricted
automatically. It also closes a leak nothing else would: a source's
path embeds its original filename, so
`sources/policy/9b3d0d/severance-plan-2026.md` sitting in an open
page's `provenance:` is disclosure even when the file itself is
unreadable.

`--restricted` is **orthogonal to `--local`**. The local tree is about
redistribution rights; restriction is about secrecy. A licensed
handbook everyone may read but nobody may republish is local-and-open;
an internal memo is tracked-and-restricted.

## Grants and mode

**Grants are entitlement. Mode is scope.**

Grants come from the calling application's own authentication — outmem
models no users, sessions or identity. Per label a user holds nothing,
`read`, or `read` and `write`. `write` without `read` is rejected at
construction: `extend_page` replaces a whole body and `append_page` has
to avoid duplicating what is there, so neither is safe without reading.

```python
from outmem.restricted import Grants

g = Grants(read=frozenset({"hr", "legal"}), write=frozenset({"hr"}))
```

Mode is the set of compartments *this session* works in. It is chosen
by the application before the run, immutable for its duration, and must
be a subset of what the user may read.

**Mode defaults to empty, and restricted content is opt-in.** A session
that names no compartment retrieves open content only. There is no
automatic escalation, no mechanism by which retrieving something widens
the mode, and no session taint — content that is never retrieved cannot
taint anything.

Note that mode `{hr}` sees open content **and** HR content, not HR
alone. The unsafe direction is read-HR-then-write-open; reading open
while writing HR is a write *up*. An HR-only session would also leave
an agent composing HR pages with no access to the company glossary.

### The compartment hint

Hiding is a property of **grants**, not of mode. For a user who does
not hold `hr`, HR content must be undetectable. For one who *does* hold
it and is merely running in mode empty, being told the compartment
exists and has content discloses nothing they are not already entitled
to see.

So when a search returns nothing, `search_wiki` appends a
**count-only, per-label, grant-gated** note:

```
(this session is scoped to open content; you also have access to:
 hr (47 pages). Ask the user to start a session in that compartment.)
```

Never titles, slugs or excerpts. Never labels the user does not hold —
an aggregate "N more" would leak the existence of compartments they
cannot hold. An item whose labels are not wholly within their grants is
not counted at all, since no mode they could choose would show it.

**The count is a property of the corpus, not of the question**, and
that is the load-bearing part. The obvious design counts the items that
matched *this* question and fell outside the mode — and the model
writes the question, so asking for "twelve" and then "eleven" and
comparing the counts reads a fact out of a restricted page without ever
retrieving it, one keyword at a time. The model could then commit that
fact to an open page. Relying on it not to try is exactly what this
design forbids, so the hint takes no query at all: it reports how many
additional pages a session scoped to that label would see, which is the
same answer for every question and therefore carries no bits about any
of them.

The trigger is the empty result, for the same reason. That depends only
on the open corpus, which the user can already search, and it is the
case the hint exists for — without it a cleared user asking about
parental leave gets nothing and never learns to switch compartment.

## The two rules

**Visibility.** An item is visible in mode `S` iff `labels(item) ⊆ S`.
Open content is visible everywhere because `∅ ⊆ S` for all `S` —
open-by-default is not a special case, it falls out of the subset
relation.

**Write.** A write may target an item iff `labels(item) == S`, and the
user holds `write` on every label in `S`. Equality, not containment,
forced from both directions: `⊆ S` because you must be able to see what
you are modifying, `⊇ S` because you must not carry facts out of a more
restricted context into a less restricted item.

Consequences worth knowing:

- In mode empty you may write only open items, however many labels you
  hold. Holding `write hr` does not let you edit an HR page from an
  open session; start a session in mode `{hr}`.
- A new item's labels default to the mode. Writing an HR page that
  happens to cite only open sources would otherwise compute `∅ ≠ {hr}`
  and be refused unless somebody remembered a label.
- Fact-laundering is blocked by the same rule, with no separate
  mechanism: citing an HR source in mode empty makes the page's
  computed labels `{hr}`, which is `≠ ∅`, so the write is refused.

Two further invariants follow:

**Closure.** An item's labels must be a superset of the labels of
everything it references. Equivalently: if you can see a page, you can
see everything it points at. Filtering the page list achieves nothing if
an open page's body contains `[[hr:severance-policy]]`, because reading
the open page hands over the slug.

Each of the three kinds of reference is held to it by a different
mechanism. **Wikilinks** are checked in the body at write time.
**Provenance** is covered by inheritance: citing a restricted source
raises the page's own labels, so a page can never end up below what it
cites. **Aliases** cannot break it because an alias derives its labels
from the page it resolves to, and never from or onto a name a live page
already occupies.

**Per-mode logs.** A restricted session writes `log/<label-set>/<date>.md`
rather than `log/<date>.md`. The log is otherwise an open file and
mandatory writeback actively pushes an agent there when nothing else
was warranted, which in a restricted session is a write-down through
the one tool the runtime insists on calling. The open mode is
unpartitioned, so a wiki with no restrictions has nothing to migrate.

## Denials look like absence

A hidden page raises the byte-identical error a nonexistent slug
raises. `exists` returns `False`. A hidden source returns `None` and
the same not-found text an unregistered path produces. None of them
raise `RestrictionError` — a distinguishable error is an existence
oracle, because learning that something is hidden is learning that it
is there.

`RestrictionError` is for **writes** and operator-only paths, where the
caller already knows what they were trying to do.

## What a view cannot reach at all

Some methods are refused to a view outright rather than filtered. A
method a view cannot reach is a bypass class that needs no filtering
code, no test, and cannot regress.

- `history` and `evolution` are answered by git, which knows nothing
  about labels, while the label index describes only the current
  commit. A page restricted today was open last month and its old
  bodies are still in the diff stream. Both also leave the served tool
  palette when a view is in play.
- Bulk maintenance — reindexing, vault import, `sources gc`, registry
  backfill, `repair_pages` — walks the whole wiki by design.

The steering signal is filtered rather than refused, because it feeds
the **system prompt**: an unfiltered `compact: hr:severance-policy`
would reach every request with no tool call needed. It is filtered on
the *mode*, not the user, so everyone in one compartment shares a
system prompt and prompt caching still works.

`wiki/index.md` is rendered live per viewer rather than served from
disk: it is the one page whose entire purpose is to list every slug.
`AGENTS.md` drops lines naming a hidden slug, because a conventions
file is exactly where somebody writes "HR policies go under `hr:`".

## Restricting existing content

```bash
outmem restrict hr:severance --label hr
outmem restrict hr:severance --label hr --cascade
outmem sources restrict policy/9b3d0d/severance-plan.md --label hr
```

The source verb has more reach: every page compiled from a document
inherits its labels, so restricting it there restricts the whole
downstream without anybody having to find those pages.

Restricting is a **graph** operation, not a field edit. A page that
visible pages already link to cannot simply become restricted — the
inbound link would still name it in a body its readers can see. The
command reports the referrers and refuses until they are resolved, or
`--cascade` applies the same labels to them.

Both verbs, and `rename_page`, are **operator-only**: they run against
the bare store and are refused to a view. That is not a policy choice
about who is trusted, it is what their shape forces. Each writes files
the caller did not name — `rename_page` rewrites inbound links across
the corpus with the new slug as content, and `--cascade` picks its
targets from the backlink graph — and there is no way to label-check a
write whose targets are discovered rather than named. A refusal that has
to name the referrers cannot be shown to a view either.

Removing a label is **declassification**, and it is only available to
the operator for the same reason. Under path rules, `hr:x → notes:x`
would otherwise be declassification through an innocuous-looking rename.
Note that a label a path rule or a cited source supplies cannot be
removed by editing frontmatter at all: the verbs refuse rather than
committing a change that would be immediately undone.

The bare store is exempt throughout. It is the server-side operator, it
holds no mode and no grants, and gating it would block the tooling that
has to fix a mislabelled page in the first place.

## Verification

`outmem lint` checks the invariants against the corpus on disk — the
case write-time enforcement structurally cannot reach, and the one a
wiki hits the day it turns restrictions on over years of content.

| kind | severity | condition |
|---|---|---|
| `restricted-link-violation` | error | a visible page links to a more-restricted page |
| `restricted-provenance-violation` | error | a page's labels do not cover its cited sources' |
| `restricted-slug-mentioned` | warning | a restricted slug appears as prose |
| `restricted-label-unknown` | error | a label absent from `restricted.labels` |
| `restricted-frontmatter-unparseable` | error | labels unreadable, so the page is denied to everyone |
| `restricted-chain-inconsistent` | error | versions of one document carry different labels |

The last is outmem-specific and easy to miss. Re-ingesting v2 of a
restricted document without `--restricted` would leave the document key
with an open current version, and `outmem stale` then quietly points
pages at it. Ingest closes that at the cause — a new version inherits
at least its predecessor's labels — and the check is the backstop for
rows written by hand or by an older build.

## Rolling it out

1. Declare the labels in `config.yaml`. Nothing changes yet: no content
   carries them.
2. Add path rules for namespaces that will hold restricted content.
   This is the step that cannot be forgotten later.
3. Re-ingest or `outmem restrict` the material that needs labelling,
   working outward from sources — inheritance does most of the work.
4. Run `outmem lint` and resolve every `restricted-*` error. Expect
   link violations: closure is the invariant existing content is least
   likely to already satisfy.
5. Switch the serving application from `store` to
   `store.as_viewer(mode=…, grants=…)`.

Steps 1 to 4 are safe to do while still serving the bare store; nothing
is hidden from anyone until step 5.

## Residual risk

**Remembered facts.** Inheritance catches what an agent *cites*. In
mode `S` an agent may only write items labelled exactly `S`, which
confines the damage to the compartment it read from — the write-down
path is closed structurally rather than by trust. What remains is a
cleared human repeating restricted content into an open session by
hand, which is outside any mechanism here.

**Bounded existence oracles.** Writing to a slug held by a hidden page
is refused generically: the writer learns something blocks the slug,
not what. Retrieval timing under over-fetch leaks a little. Both are
documented rather than eliminated; prefer path rules for namespaces
that contain restricted content, which keeps the collision inside the
compartment.

**Availability, not confidentiality.** The opt-in default means a
cleared user who never switches compartment will not see content they
were entitled to. That is a deliberate trade: retrieving restricted
content by default reintroduces the write-down problem the default
exists to remove. The compartment hint is the mitigation.
