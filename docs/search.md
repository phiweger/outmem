# Search & retrieval

Retrieval in outmem is a **workflow**, not a single call: *orient* on
the wiki, *locate* a page, *read* it in full, *traverse* to neighbours.
This page explains the loop, the tools, and how the configured retrieval
strategy fits in.

## The tools at a glance

| tool | input | returns | use it for |
| --- | --- | --- | --- |
| **`search_wiki`** | a natural-language *question* | wiki pages ranked by relevance, as `[[slug]]` citations + excerpt | "which pages answer this?" — the primary recall move |
| **`grep_wiki`** | a *pattern* (regex/literal) + `scope` | `key:line:text` rows, one per match | the *exact line* a string is on; searching the source documents or the `log/` |
| **`search_index`** | a namespace *prefix* (optional) | child namespaces (with page counts) + leaf pages at that level | browsing the wiki's structure (the TOC) to orient before searching |
| **`list_pages`** | — | every slug, one per line | a flat existence check |
| **`read_page`** | a `slug` (+ `peek`) | the full page, or — with `peek=True` — the title + first ~1000 chars | reading a hit in full, or skimming it cheaply first |
| **`find_backlinks`** | a `slug` | slugs that link to it | "what references / depends on this page?" |
| **`find_similar`** | free *text* | cosine-similar chunks (needs a built index) | paraphrase / near-duplicate matches grep would miss |

`search_wiki` runs the pipeline configured by the wiki's
`retrieval.strategy` (the `retrieval:` block in `config.yaml`; default
`rerank(bm25)`). `grep_wiki` is plain ripgrep — no ranking, no model.
`search_index` and `list_pages` are cheap directory walks — no model, no
index. (`page_history` / `topic_evolution` traverse *time* rather than
content — see [growing-the-wiki.md](growing-the-wiki.md).)

## The core loop

```
search_wiki(question) ──▶ [[slug]] ──▶ read_page(slug) ──▶ traverse
   rank relevant pages       cite        full text         backlinks / [[links]] / history
```

0. **Orient (optional)** — on an unfamiliar wiki, `search_index()` shows
   its shape (top-level namespaces + page counts) before you search;
   drill in with `search_index(prefix="abx")`.

1. **`search_wiki(question="…")`** — ask a question, get the most
   relevant whole pages back as `[[slug]]` citations with a short
   excerpt. `read_page` the interesting ones.

2. **`read_page(slug)`** — the full page text. This is the step that
   matters: reasoning should happen over the whole page, not an excerpt.
   Unsure a hit is the right one? `read_page(slug, peek=True)` returns
   just the title + first ~1000 chars to triage it before you spend the
   context on a full read.

3. **Traverse** — `find_backlinks(slug)` (who links here), `[[slug]]`
   wikilinks in the body, `page_history(slug)`, `topic_evolution(slugs…)`.

`list_pages()` enumerates every slug flat; `search_index()` browses the
same slugs *by namespace* when you'd rather see the structure.

## Browsing the index (`search_index`)

Slugs are `:`-namespaced (`abx:penicillin`, `abx:side-effects:rash`),
mirroring directories under `wiki/pages/`. `search_index` walks that tree
one level at a time — the wiki's table of contents:

```
search_index()              # top-level namespaces (with counts) + top-level pages
search_index(prefix="abx")  # the namespaces and pages directly under abx:
```

Each call returns the child namespaces at that level (each with the count
of pages beneath it, ready to pass back as the next `prefix`) and the leaf
pages sitting directly there. Where `list_pages()` dumps every slug flat,
`search_index` shows the *shape* — reach for it to orient on an unfamiliar
wiki, then `read_page` (optionally `peek=True`) the leaves that look
relevant. It's a plain directory walk: no model call, no semantic index.

## `grep_wiki` — literal matches & non-wiki scopes

When you need the *exact line* a string appears on, or to search material
`search_wiki` can't reach, use `grep_wiki`:

```
grep_wiki(pattern="cost-plus 35%", scope="wiki")   # exact line in a page
grep_wiki(pattern="penicillin", scope="sources")    # source docs, both trees
grep_wiki(pattern="...", scope="log")               # the gap log
```

| scope  | searches                | row shape                     |
| ------ | ----------------------- | ----------------------------- |
| `wiki` | curated pages (default) | slug — `abx:penicillin:14:…`  |
| `sources` | source documents — **both** `wiki/sources/` and `wiki/sources-local/` | path — `sources/a1b2/deck.md:3:…` |
| `log`  | the append-only log     | path                          |
| `all`  | everything              | mixed                         |

Only `wiki` scope is slug-shaped; `sources`/`log`/`all` return real paths.
A `sources` hit's prefix tells you which tree it came from — see
[sources.md](sources.md).
Output is capped at **8 KiB** (a soft token ceiling); past that you get a
trailing `(truncated — narrow the pattern)`.

## Retrieval strategies (what `search_wiki` runs)

`search_wiki`'s pipeline is one `strategy` string — `lexical` | `bm25` |
`semantic` | `hyde` | `rerank(<source>)` | `a+b[+c…]` (RRF hybrid).
Default `rerank(bm25)`: a BM25 keyword shortlist gated by one cheap
model call per query. `semantic`/`hyde`/`*+semantic` need a built index
(`outmem reindex`) and fail loud without one.

Don't pick by hand — let outmem measure it. [Retrieval
tuning](autoresearch.md) scores the strategies on a question bank built
from your wiki and reports the best; `result.save(rank, store)` writes it
into the `retrieval:` block. See
[configuration.md](configuration.md#retrieval--what-the-agents-wiki-search-runs)
for the full strategy table and knobs.

## Semantic similarity (`find_similar`)

A parallel door, reached through `find_similar(text)` — vector cosine
similarity, exposed only when the semantic index is built. Use it when
the word you'd grep for isn't the word on the page: "have we seen
something like this before?", a paraphrase, a near-duplicate you want to
avoid writing again. (The system prompt nudges agents to `find_similar`
before `write_page`.) Needs `outmem[semantic]` + `outmem reindex`.

## CLI quick reference

```bash
outmem search "penicillin" --scope wiki        # ripgrep (grep_wiki's CLI); -i, -F, --max-hits N
outmem read abx:penicillin                      # full page by slug
outmem similar "beta-lactam alternative"        # semantic (needs [semantic] + reindex)
outmem similar --slug abx:penicillin            # use a page's body as the query
outmem history abx:penicillin                   # git log for one page
outmem evolution abx:penicillin abx:ceftriaxone # diff stream across pages
```

The CLI's `outmem search` is raw ripgrep (the `grep_wiki` behaviour); the
strategy-driven `search_wiki` is an agent tool (it can spend a model
call), not a CLI command.

## Embedding the tools in your own agent

The PydanticAI adapter hands you the same palette as plain callables:

```python
from outmem import WikiStore
from outmem.adapters.pydantic_ai import wiki_read_tools, wiki_tools

store = WikiStore.open("/srv/wiki")
tools = wiki_read_tools(store)        # retrieval only (read-only consult)
# tools = wiki_tools(store)           # + writeback paths
```

* **Black-box consult**: `build_consult_wiki(path)` returns a single
  `consult_wiki(question) -> str` tool that runs an inner agent over this
  whole workflow and returns a cited answer — when you want "ask the
  knowledge base", not the raw retrieval primitives. See
  [python-api.md](python-api.md).

## Failure & edge-case cheatsheet

| you see                                    | meaning                                                        |
| ------------------------------------------ | -------------------------------------------------------------- |
| `(no matches)`                             | `grep_wiki` found nothing — broaden, drop `-F`, or try `search_wiki` |
| `(no pages matched — …)`                   | `search_wiki` ranked nothing — rephrase, or `grep_wiki` for a literal |
| `(truncated — narrow the pattern)`         | hit the 8 KiB cap — tighten the query or raise `--max-hits`     |
| `(search failed: …)`                       | ripgrep not installed / bad regex — the message says which     |
| `(search_wiki failed: … not built)`        | a semantic strategy with no index — run `outmem reindex`        |
