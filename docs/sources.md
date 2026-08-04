# Sources — what the wiki is compiled from

A wiki page is a compaction of something. That something lives in a
**source tree**, and outmem keeps two of them:

| | `wiki/sources/` | `wiki/sources-local/` |
|---|---|---|
| Tracked in git | **yes** | **no** (gitignored) |
| Ships with a clone / push | yes | no |
| Registered (sha256, supersession, ingestion log) | yes | yes |
| Readable by the agent (`read_source`, `grep_wiki`) | yes | yes |
| Citable in a page's `provenance:` | yes | yes |
| In the semantic index | when `semantic.index: pages+sources` | **never** |
| Created | at `outmem init` | on first `--local` ingest |

Everything else is identical: same content-addressed layout
(`<sha256[:12]>/<filename>`), same registry schema, same tooling.

## Why two trees

The distinction is **redistribution rights**, not secrecy.

You can lawfully read a licensed handbook, a purchased corpus, or a
copyrighted paper, and you can lawfully write your own notes about
what it says. What you generally cannot do is republish the source
text. A wiki that compiles the first into the second is doing
something useful and legitimate — but only if the source bytes stay
put while the derived pages travel.

That is exactly the split:

```
wiki/sources-local/  →  the handbook           (stays on your machine)
wiki/pages/          →  your notes about it    (ships, and is yours)
```

## Ingesting

```bash
# Ships with the wiki (default).
outmem ingest ./press-release.md --into announcements

# Stays on this machine.
outmem ingest ./licensed-handbook.md --local --into reference
```

`--local` creates `wiki/sources-local/` on first use and adds it to
the wiki's `.gitignore` **before** copying anything in — a source
copied first and ignored second is one a concurrent `git add -A` can
still catch.

Nothing is committed for a local ingest. Both the file and the
registry that indexes it live inside the ignored tree.

From Python:

```python
store.add_source("./licensed-handbook.md", local=True)
```

## What the agent sees

Very little difference, deliberately. The agent reads and cites local
material like any other:

```python
grep_wiki(pattern="dosage", scope="sources")   # spans BOTH trees
read_source(rel_path="sources-local/a1b2c3/handbook.md")
list_sources()                                  # rows are tree-prefixed
```

`scope="sources"` covering both trees is not a convenience — it is the
point. An agent told to "fall through to the sources" must not
silently miss half the corpus because of a distribution policy it has
no reason to know about.

The one rule that *is* different is in the agent's system prompt and
the `ingest` skill: a page compiled from a local-only source should
carry synthesis rather than long verbatim passages, because the page
travels even though the source does not. Summarise, restructure, quote
briefly where exact wording carries meaning.

## The guarantees, and what enforces them

**1. Git never sees local bytes.** The `.gitignore` entry is written
before the tree is populated. `outmem lint` raises a
`local-source-tracked` **error** if git is nonetheless tracking
anything under it — which happens if someone ran `git add -f`, or if
the ignore rule was added after the files. The check asks git what it
*is* tracking (`git ls-files`) rather than reading `.gitignore`,
because a file already committed stays tracked no matter what the
ignore file says afterwards, and that is the case worth catching.

**2. The registries are separate.** The registry stores `rel_path`
(which embeds the filename), `sha256`, and `origin_path` (the absolute
path on the machine you ingested from). A single shared registry would
commit all three for every local source. Each tree therefore carries
its own `.sources.db`, and the local one lives inside the ignored tree.

**3. Local material never enters the semantic index.** This is the
subtle one. The vector DB stores each chunk's **verbatim text**
alongside its embedding, and it is staged into the same commit as the
write that triggered it. Indexing local sources would push the exact
bytes the tree exists to withhold straight into git, through a path
that looks like a cache.

So the exclusion is unconditional — it applies even under
`semantic.index: pages+sources`, which means "the *tracked* sources
tree". It is deliberately not "skip when the DB happens to be
untracked": a safety property should be checkable in one sentence, not
dependent on the current contents of an unrelated file. `outmem lint`
raises `local-source-indexed` if the index contains local chunks
anyway (from an index built before this rule existed); `outmem reindex
--force` rebuilds without them.

The cost is that `find_similar` cannot see local material. The pages
distilled from it *are* indexed, which is what recall usually wants.

## What still travels: filenames

A page citing a local source records
`sources-local/<sha>/<filename>` in its `provenance:`, and that page
is tracked. **The filename travels even though the bytes do not.**

That is the intended trade — a citation is not a redistribution, and a
provenance chain nobody can follow is not worth much. But it means
`sources-local/` protects *distribution rights*, not *secrecy*. Do not
use it for material whose name alone must stay private; that is a
different problem needing a different tool (an encrypted volume, or
simply keeping the document out of the wiki's directory entirely).

## Reclassifying

Material moves between trees by re-ingesting into the other one:

```bash
outmem ingest ./now-public.md            # was local, embargo lifted
```

Then remove the old copy and let `outmem sources gc` drop the stale
registry row. Supersession chains do not span the boundary — each tree
has its own registry — so treat a reclassification as a new document
rather than a new version of the old one.

If material moves *out* of tracked into local because you realise it
should never have shipped: moving it is not enough. The bytes are in
git history and must be treated as published — rewrite history, and
assume anyone who cloned already has them.

## See also

- [cli.md](cli.md#ingestion-requires-outmemagent) — `outmem ingest` in full
- [configuration.md](configuration.md) — `semantic.index` and what it covers
- [search.md](search.md) — how `scope="sources"` fits the retrieval tiers
