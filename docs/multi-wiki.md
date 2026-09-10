# Several wikis in one repository

Some organisations hold material only part of the company may see. outmem
separates that by **putting each audience in its own wiki**.

A wiki is the compartment. Within a wiki everything is open — there is no
per-page or per-source access control, and nothing at read time decides what
to hide. A session is permitted a *set of wikis* up front, and every store it
then holds is one it may read in full.

That is the whole mechanism, and its value is what it makes impossible. There
is no filtering code, so there is no filtering bug: no path that can return
content, a title, or the mere existence of something it should not. The only
question is which directory a request opens, and it is answered once, before
anything is read.

The deployment this assumes: outmem runs server-side, one repository,
employees with no clone rights. The repository is dark to them, so choosing
which wiki to open is genuine access control rather than a display convention.
**If your users can read the repository, none of this protects anything.**

---

## Layout

```
/srv/memory/                  git repo — one history, N wikis
  wikis.yaml                  the registry
  .outmem-repo/               gitignored: the commit lock
  wikis/
    open/                     an ordinary wiki root
      config.yaml  CONTRIBUTORS.md  .vectors.db  .outmem/  log/
      wiki/
        AGENTS.md  index.md
        pages/  sources/  sources-local/
    hr/                       same shape
    legal/                    same shape
```

Each wiki directory is exactly what `outmem init` produces. That is
deliberate: a wiki moves out of the repository and opens standalone, and a
standalone wiki moves in, with no conversion step. Everything you know about a
single outmem wiki still applies inside each one.

The wikis share one git history. Commits carry the wiki they belong to in the
subject:

```
$ git log --oneline
4ee0434 legal/ compact: contract-review
8730f7b open/ compact: pricing-formula
be9d6bf repo: add legal
```

A store can only ever stage paths under its own wiki — the guard is in
`WikiStore._commit_paths`, at the one point where paths cross into git.

---

## `wikis.yaml`

```yaml
version: 1

# The declared vocabulary. This is the contract with your user database.
tags:
  everyone: {description: "All employees"}
  hr:       {description: "People team"}
  legal:    {description: "Legal counsel"}
  exec:     {description: "Executive team"}

wikis:
  open:
    path: wikis/open
    title: "Company handbook"
    audience: [everyone]
  hr:
    path: wikis/hr
    title: "People team"
    audience: [hr]
  legal:
    path: wikis/legal
    title: "Legal"
    audience: [legal, exec]
```

An **audience tag** is opaque to outmem. Your application maps its
authenticated user to a set of tags and hands them in; outmem does one
set-overlap test on names, once, before a store exists. Nothing about a tag is
derived from the wiki's contents.

Reachability is a plain overlap: **one shared tag is enough**. `legal` above is
reachable by anyone holding `legal` *or* `exec`.

Declaring the vocabulary under `tags:` is not decoration. A tag used in an
`audience:` but never declared makes a wiki *silently unreachable* — nobody can
be granted a tag nobody knows exists, so the content is simply gone with no
error anywhere. `outmem lint --repo` reports it as an error.

---

## Getting started

```bash
outmem repo init --root /srv/memory
outmem repo add open  --root /srv/memory --audience everyone --title "Handbook"
outmem repo add legal --root /srv/memory --audience legal --audience exec

# every ordinary subcommand takes --wiki
echo "The list price is cost times 2.4." \
  | outmem write pricing-formula --root /srv/memory --wiki open --title "Pricing"

outmem lint --repo --root /srv/memory
```

To bring an existing wiki in:

```bash
outmem repo import /srv/old-hr-wiki --root /srv/memory --name hr --audience hr
```

A wiki already inside the repository is moved with `git mv`, so tracked files
keep their history. A wiki from elsewhere is copied and committed as new
content — its own history stays in its own repository. Merging two histories is
`git subtree` / `git filter-repo` work, and `repo import` says so rather than
pretending.

A wiki inside the repository that still carries **its own `.git`** is refused:
moving it would leave a nested repository, which git treats as a foreign
checkout and will not stage. Remove that `.git` first — which discards that
wiki's history, so it is your call, not the command's.

### Keeping `wikis.yaml` by hand

`wikis.yaml` is the one file here a person is meant to maintain, and comments
in it are welcome — "mirrors the IdP groups", "see ADR-014". **outmem never
rewrites a `wikis.yaml` that carries comments.** `repo add` and `repo import`
have two modes, chosen by whether the name is already listed:

- **Listed** → the registry is the source of truth and is not touched. The
  wiki is scaffolded (or moved) to the path the entry gives. Edit the YAML,
  commit it yourself, then `outmem repo add legal`.
- **Unlisted** → the entry is written for you, but only if the file has no
  comments. A commented file is refused with a note saying what to add.

So a hand-maintained registry and the convenience commands coexist: the
commands only ever write to a file they could regenerate anyway.

**Comments carry rationale; `description:` carries data.** A comment is for
the next person to read the file — "mirrors the IdP groups", "see ADR-014".
Anything a *machine* consumes belongs in the tag's `description:`, because
that is the field `outmem repo tags --json` emits and a provisioning UI
displays; a comment never reaches that consumer. `repo add --audience X`
writes an empty description and has no flag to fill it, so `outmem lint
--repo` reports an undescribed tag (`registry-undescribed-tag`, a warning) —
terse self-explanatory tags are a legitimate choice, so it does not fail the
run.

---

## Host integration

This is the part your application has to get right, and it is short.

```
wikis.yaml declares tags
      │
      ├─ Repo.tags()  ──▶  your provisioning UI / user_tags table
      ▼
authenticate the request, look up that user's tags
      ▼
Repo.wikiset(audience=tags)  ──▶  the session's tools
```

### Store tags, never wiki names

The tag vocabulary is the stable contract. Wikis can be renamed, split, merged
or moved underneath it without touching a single user record. A user table
keyed on wiki names breaks the first time somebody reorganises the repository.

### Discovering the vocabulary

Your user database has to *name* the tags it assigns, so it needs to enumerate
them rather than only filter with them:

```bash
outmem repo tags --json
```

```json
{
  "version": 1,
  "tags": [
    {"name": "everyone", "description": "All employees", "wikis": ["open"]},
    {"name": "legal", "description": "Legal counsel", "wikis": ["legal"]}
  ],
  "wikis": [
    {"name": "open", "title": "Company handbook", "audience": ["everyone"], "pages": 412},
    {"name": "legal", "title": "Legal", "audience": ["legal", "exec"], "pages": 87}
  ]
}
```

`version` is bumped when the shape changes incompatibly. Read it.

In Python:

```python
from outmem.repo import Repo

repo = Repo.open("/srv/memory")

repo.tags()                      # the vocabulary, with wiki back-references
repo.catalogue()                 # every wiki and tag — admin view
repo.catalogue_for({"everyone"}) # only what this audience reaches
```

`catalogue()` and `catalogue_for()` are separate calls on purpose.
`catalogue()` names every wiki and every tag, which is right for a provisioning
screen and wrong for an end-user picker: a name can itself be the secret. The
existence of an `atlas-acquisition` tag says something whatever the wiki behind
it is called.

### Serving a session

```python
repo = Repo.open("/srv/memory")
wikis = repo.wikiset(audience=tags_for(current_user))   # open core + compartments
```

**Close what you open.** A `WikiSet` holds one `WikiStore` per wiki, and each
opens SQLite handles lazily — the vector store and both source registries.
Dropping the set does not release them promptly: they sit in reference cycles
and survive until the cycle collector runs, so a busy server carries an
unpredictable number of open connections rather than a bounded one. Closing
makes it deterministic.

```python
with repo.wikiset(audience=tags_for(current_user)) as wikis:
    ...
# or: wikis.close()
```

Long-running servers usually want the other shape: build one set per distinct
audience at startup and keep it, rather than one per request.

`Repo.wiki(name, *, audience=…)` opens one. The audience argument is
keyword-only and required; the unrestricted path is the separately named
`Repo.wiki_as_operator(name)`, so reading the whole repository is something
your code says out loud rather than reaches by passing the obvious argument.

A name the audience does not reach fails with the **same message** an unknown
name does. Code that could tell them apart could enumerate the wikis it may not
open.

### Attaching the tools

```python
from pydantic_ai import Agent
from outmem.adapters.wikiset import wikiset_read_tools

agent = Agent(
    "anthropic:claude-sonnet-5",
    tools=wikiset_read_tools(wikis),
    system_prompt="Answer from the wiki only. Cite pages by name.",
)
```

The model sees one knowledge base. Page names come back qualified `wiki/slug`
(`legal/contract-review`); a slug never contains `/`, so the qualifier is
unambiguous. A bare slug resolves in wiki order and reports what it shadowed,
so a reader who got the open version of a name two wikis share can ask for the
other.

Each wiki runs the retrieval pipeline **it** configures (`retrieval.strategy`),
and the per-wiki rankings are fused by Reciprocal Rank Fusion — the same method
`hybrid` uses for its legs. Fusion rather than concatenation because retrievers
return order, not comparable scores: one wiki's third-best is not commensurable
with another's. A wiki whose strategy needs a semantic index it hasn't built
falls back to bm25 for that query and says so, and a wiki that fails outright
is reported without blanking the rest.

`search_wiki` returns qualified page citations (`[[legal/nda]]`), not raw
chunks — the same contract as the single-wiki tool. When nothing matches, the
message names the wikis searched and carries every diagnostic, because "we have
nothing on that" is a load-bearing answer and must not be confused with a
retrieval failure.

Search results carry which wikis had to clip theirs at the output cap, and the
tool passes that to the model: a partial result that reads as complete is worse
than no result, because the model concludes the wiki holds nothing more and
stops looking.

Reads federate. **Writes do not** — "append this to the wiki" has no answer
when there are three. A session that writes takes a single `WikiStore` and the
ordinary palette from `outmem.adapters.pydantic_ai`.

### Reconciling with your user table

Two failures are silent by construction, so something has to go looking:

```python
repo.reconcile(all_tags_assigned_in_your_db)
# → unknown:     tags you assign that no wiki declares (stale grant, or a typo)
# → unreachable: wikis no assigned tag opens (content nobody can see)
```

Worth a nightly job. For a single user's ticket — "why can't Sam see the HR
wiki?" — use:

```bash
outmem repo audience --root /srv/memory --tags hr,everyone
```

---

## What crosses a wiki boundary, and what does not

| | Crosses? |
|---|---|
| Retrieval (`search`, `find_similar`) | Yes — fanned across the session's set and merged |
| Page names | Yes — qualified `wiki/slug` |
| `[[wikilinks]]` | **No** — lint reports `cross-wiki-wikilink` |
| Provenance, backlinks, `index.md` | No — wiki-local |
| Writes | No — a write names one wiki |
| Git history | Shared, one repository |

Wikilinks stay wiki-local because a link is a claim that every reader of this
page can follow it, and the readers of two wikis are not the same people. To
reference material in another wiki, cite a shared source or restate what you
need.

Retrieval merges *before* ranking, not after: taking the top *k* from each wiki
first would let a wiki with nothing relevant crowd out one that had everything.

---

## Concurrency

Several wikis in one repository means concurrent writers are ordinary. A commit
is two operations — `git add` writes the index, `git commit` reads it back —
and between them the index is shared state.

outmem serialises the pair with an `fcntl.flock` on `<repo>/.outmem-repo/`.
Without it, measured on three processes writing into three wikis: two to four
commits per run carried another wiki's paths, and one or two pages that
`write_page` reported as written were absent from HEAD. The visible
`index.lock: File exists` is the lesser half; the real damage is silent.

Nothing is needed to enable this. If the lock cannot be taken — a read-only
mount, a permissions problem — the commit proceeds unserialised rather than
failing.

---

## The pre-commit hook

`.git/hooks` is per-clone, so there is exactly **one** hook for the whole
repository, and one commit can carry files from several wikis. `outmem reindex
--staged` groups staged paths by wiki and reindexes each. Install it once at
the repository root:

```bash
outmem hook install --root /srv/memory --wiki open
```

---

## Linting

```bash
outmem lint --repo --root /srv/memory
```

Checks the registry, then every wiki, as one report with repo-relative paths
(`wikis/legal/wiki/pages/nda.md`). Wikis are opened read-only for lint — it
installs no hook and creates nothing. Registry findings:

| Kind | Severity | Meaning |
|---|---|---|
| `registry-malformed` | error | `wikis.yaml` exists but does not parse |
| `registry-missing-wiki` | error | listed in `wikis.yaml`, no directory |
| `registry-not-a-wiki` | error | listed, directory exists, but it is not a wiki (no `config.yaml`) |
| `registry-undeclared-tag` | error | an `audience` tag no `tags:` entry declares — the wiki is unreachable |
| `registry-unreachable-wiki` | warning | a wiki with an empty `audience` |
| `registry-unused-tag` | warning | a declared tag no wiki lists |
| `registry-undescribed-tag` | warning | a declared tag with an empty `description:` — what `repo tags --json` emits, and what a user database provisions against |
| `registry-unlisted-wiki` | warning | a wiki-shaped directory absent from the registry — unreachable, and commits in it would start their own repository |

Plus `cross-wiki-wikilink` (error) on any `[[other-wiki/page]]` link.

---

## Notes and gotchas

**Finding the repository is opt-in.** outmem does not walk up looking for
`.git`. An ancestor is accepted only when its `wikis.yaml` lists the wiki
directory by name. A wiki that happens to sit inside an unrelated repository
(`~/projects/notes` under `~/projects/.git`) keeps behaving exactly as it does
today.

**`wikis.yaml` is committed.** It is what makes the directory a multi-wiki
repository and it is read from the working tree — untracked, it would not
survive a clone, every wiki would look standalone, and the next
`WikiStore.init` would nest a `.git` inside the repository.

**A malformed `wikis.yaml` is an error, not "no registry."** Treating it as
absent would quietly demote the repository to a single wiki and commit one
wiki's writes into another's history.

**Wiki names are slug-like** (lowercase letters, digits, `.`, `_`, `-`). They
appear in commit subjects and as slug qualifiers, so both grammars have to stay
unambiguous.

---

## Coming from `restricted:` (0.16.x)

The per-item label system is **removed in 0.17.0**. If you used `restricted:`,
split the wiki by hand before upgrading — one wiki per label — or stay on
0.16.1.

There is no automatic split. Collapsing a label lattice into a partition is
lossy: a page labelled `{hr, legal}` has no single home, and only a person who
knows the content can decide where it goes.
