# Multi-wiki — design record

Written alongside the 0.17.0 implementation. Records why the design is shaped
the way it is, and where the implementation landed somewhere other than the
plan.

---

## 1. The problem, and the previous answer

Some organisations hold material only part of the company may see. outmem
0.16 answered that with **per-item labels**: every page, log entry and source
carried a label set, every session carried a mode, and every read path
filtered on `labels(item) ⊆ mode`.

It worked. It also cost ~3,750 lines across fifteen modules, 383 tests, and
fifteen interacting concepts — grants versus mode, closure over citations,
path rules, inheritance, the label index and its validity token, the
compartment hint, log partitions, write-down refusal, declassification.

The decisive number is not the line count. It is that **every one of the ~29
defects found across four adversarial review rounds was a filtering bug** —
some path that returned content, identity, or existence without consulting the
mode. `read_page` served the stored `index.md` past the per-viewer render. A
shallow copy in `as_viewer` diverged the lazy registries and failed open. A
`..` spelling of a source path bypassed the guard. Aliases could relabel a live
page. Deleting the `restricted:` block published everything. The compartment
hint was a per-keyword content oracle.

None of these were careless. They are what a filtering boundary costs: the
correctness obligation is *every* path that can return anything, forever,
including paths added next year by someone who has not read this document.

## 2. The answer: separation instead of filtering

**A wiki is the compartment.** Within a wiki everything is open. A session is
permitted a set of wikis up front, and every store it then holds is one it may
read in full.

Nothing decides what to hide, so nothing can decide wrong. The correctness
obligation collapses from "every read path, forever" to "which directory does
this request open", answered once, before anything is read.

The trade is real and worth naming. Filtering could share one corpus with
different views onto it; separation cannot. Content two audiences both need
must either be duplicated, or live in a shared wiki both can open. We chose the
latter — the "open core" — and made retrieval federate across the session's
set so the model still sees one knowledge base.

### 2.1 Why removal, not deprecation

The label machinery could have stayed alongside as a finer-grained option.
It did not, because its value was entirely conditional on being correct, and
reaching correct took four review rounds. Code like that in a repository where
nothing recommends it is an access-control surface maintained for no reader —
someone will find it, enable it, and inherit a boundary nobody is reviewing any
more.

0.16.0 and 0.16.1 were *entirely* that feature, so removal was a near-clean
reversal to v0.15.0. The verification was mechanical: after Phase 0,
`git diff v0.15.0 -- src/outmem/` had to reduce to one function.

### 2.2 The one thing kept

`_canonical_within` in `_store/sources.py`. `resolve_source` keyed the registry
off a textual prefix strip, so `sources/../../wiki/sources/<rel>` resolved to a
real file under a key the registry had never held — one file reachable under
two spellings, with rows under only one. That is a path canonicalisation bug
independent of labels, so it stayed, with three tests that all fail when it is
reverted.

## 3. The one design decision everything follows from

`WikiStore.root` was doing two jobs:

- **content root** — pages, sources, `log/`, `config.yaml`, `.vectors.db`,
  `.outmem/`;
- **git root** — every `git_ops` call took it as `repo_path`.

As long as a wiki was a repository these were the same directory, so nothing
distinguished them. Splitting them into `root` and `repo`, with `repo_prefix`
the wiki's location inside the repository, is what makes several wikis in one
repository expressible. Everything else in this document is a consequence.

The back-compat argument is the same fact read the other way: for a standalone
wiki the two are equal and the prefix is empty, so the pre-existing test suite
passes unedited. That, rather than any assertion, is the evidence the split is
a no-op for every wiki that exists today.

### 3.1 Translation happens in exactly one place

Wiki-relative paths become repo-relative inside `_commit_paths` and nowhere
else. Every caller keeps passing paths relative to its own wiki, so one wiki
cannot name another's files by construction rather than by each of eight
callers remembering to prefix.

The vector DB deliberately keeps *wiki-relative* keys. A key that shifted when
a wiki was placed in a repository would orphan every chunk already embedded.

## 4. Where the implementation landed stricter than the plan

**Finding the repository is opt-in.** The plan said "walk up for `.git`". That
would change behaviour for a wiki that happens to sit inside an unrelated
repository: `~/projects/notes` under `~/projects/.git` refuses commits today
and would silently start committing into the parent. An ancestor is accepted
only when its `wikis.yaml` lists that exact directory, compared by resolved
path so a symlink or `..` spelling still matches.

**Path containment is a guard, not a hope.** The plan called for refusing paths
outside the wiki. Verifying it as an attack first showed why it is
load-bearing: git normalises `..` in a pathspec and stages the result happily,
so without the guard the `open` wiki can edit `legal`'s page on disk, commit
it, and the commit is well-formed and even carries `open/` as its qualifier.

**The commit lock covers more than the plan claimed.** The plan framed it as
avoiding `index.lock: File exists`. Measurement showed the visible error is the
lesser half: without the lock, two to four commits per run carried another
wiki's paths, and one or two pages that `write_page` had reported as written
were absent from HEAD. The caller is told the write succeeded and the history
is simply wrong.

> **Amended in 0.19.** The lock still covered too little. It serialised the
> stage-and-commit pair, but what a write rewrites *before* it commits —
> `.sources.db`, `wiki/index.md`, `.vectors.db` — is shared with every other
> writer of the same wiki, and git refuses to stage a file that changes under
> its hash. Eight processes registering sources into one wiki lost a
> registration every round to `unstable object source data`. The lock is now
> re-entrant within a thread and held across each mutating method's whole
> body — the cross-process twin of `_write_lock` — with the commit funnel
> re-entering it. Found by a thread-contention test written for something else.

**`catalogue_for` filters tags as well as wikis.** The plan named the wiki leak
and missed the tag one. A tag name discloses as much as a wiki name — the
existence of an `atlas-acquisition` tag is the secret, whatever the wiki behind
it is called. Found by inspecting the first implementation's own output.

**`wikis.yaml` is committed.** The plan did not say. Following the convention
of `outmem init` — which leaves the scaffold for the author's first write to
carry in — would have left the registry untracked, so it would not survive a
clone: every wiki would look standalone and the next `WikiStore.init` would
nest a `.git` inside the repository.

**`repo audience` does not report a full reconciliation.** `reconcile` answers
a repository-wide question. Asked about one user, "wikis you cannot see" is the
normal condition rather than a finding, so only the unknown-tag half is
reported.

**A malformed `wikis.yaml` is an error.** Treating it as "no registry" would
quietly demote a multi-wiki repository to a single wiki and commit one wiki's
writes into another's history. This mirrors the fail-closed instinct from the
label work, in the one place it still applies.

## 5. Decisions worth recording

**Audience tags are opaque and reachability is overlap, not subset.** One
shared tag is enough. Subset semantics would mean a user needed *every* tag a
wiki lists, which makes `audience: [legal, exec]` mean "legal AND exec" rather
than the intended "either".

**The host stores tags, never wiki names.** The vocabulary is the stable
contract; wikis can be renamed, split, merged or moved underneath it. A user
table keyed on wiki names breaks the first time somebody reorganises.

**Declaring the vocabulary is kept from the label design.** It is the one idea
there that earned its place: a tag used but never declared makes a wiki
silently unreachable, which is precisely the failure the label vocabulary
existed to catch. It now applies to about six tags in one file rather than to a
lattice.

**Denial and absence are one message.** `Repo.wiki` fails identically for a
name outside the audience and a name that does not exist. Code that could tell
them apart could enumerate the wikis it may not open.

**There is no argumentless way to open everything.** `wiki(name, *, audience)`
requires the audience keyword; the unrestricted path is the separately named
`wiki_as_operator`. Reading the whole repository is something a caller says out
loud rather than reaches by passing the obvious argument. A test asserts the
signature, because the guard rail *is* the shape of the API.

**Wikilinks stay wiki-local.** A link is a claim that every reader of this page
can follow it, and the readers of two wikis are not the same people. One-way
links toward an open core would be checkable by lint in one pass with no
runtime cost, and are a coherent later extension — but they are the one place
the graph could start to reacquire structure, so they are not in 0.17.0.

**Retrieval merges before ranking.** Taking the top *k* from each wiki first
would let a wiki with nothing relevant crowd out one that had everything.
Cosine similarity against one query vector is comparable across wikis as long
as they embed with the same model, which is the condition that makes the merge
meaningful.

**Reads federate; writes do not.** "Append this to the wiki" has no answer when
there are three. A session that writes takes a single `WikiStore`.

## 6. What this design cannot do

- **Share a page between two wikis.** Duplicate it, or move it to a wiki both
  audiences can open.
- **Express overlapping compartments without duplication.** A label lattice
  could give `{hr}`, `{legal}` and `{hr, legal}` distinct views of one corpus.
  A partition cannot, which is exactly why migrating from labels needs a human.
- **Protect anything if users can read the repository.** The process boundary
  is the boundary. This is a deployment precondition, not a detail.

## 7. Verification approach

Every phase was checked by deliberately reverting the change it made and
confirming the tests fail. That found two genuine gaps: the first containment
mutation did not match the source and passed vacuously (the tests were
sharpened until removing the guard fails all four), and two of the three
`_canonical_within` tests passed with the fix reverted because they exercised
paths that resolve through the filesystem either way — they were rewritten
against `resolve_source` directly.

The concurrency tests are the same idea under load: they fail on every run with
the lock disabled and pass on every run with it.
