---
name: search
description: >
  Look up facts, decisions, or notes in an outmem wiki — a git-versioned
  directory of compiled markdown maintained by the team. Use when the
  user asks "what did we decide about X", "find Y in our notes", "have
  we discussed Z", "look up our position on …", or any other recall
  question where the answer might already be compiled.
---

# search — find what we already know

Compiled knowledge is cheaper to read than raw sources are to
re-derive, and the wiki is *your* prior work — start there and only
fall through to raw material when the wiki was silent.

## Tools at a glance

The full search/navigation/read surface — reach for the right one:

| Tool | Reach for it when |
|------|-------------------|
| `search_wiki(question, k)` | **Primary recall.** Question-shaped lookup — ranks whole pages by relevance using the wiki's configured strategy (default `rerank(bm25)`). |
| `grep_wiki(pattern, scope)` | The *exact line* a literal/regex string sits on, or to search `raw/` sources / the gap `log` (`scope="raw"` / `"log"`). |
| `search_index(prefix)` | **Browse the structure.** Walk the slug namespaces (the table of contents) one level at a time to orient on an unfamiliar wiki. |
| `list_pages()` | A flat dump of *every* slug — a cheap existence check. |
| `read_page(slug, peek)` | Open a page; `peek=True` returns just the title + first ~1000 chars for a cheap skim before a full read. |
| `find_backlinks(slug)` | What links *to* a page (the reverse `[[wikilink]]` graph). |
| `find_similar(text, …)` | Paraphrase / semantic matches — only when a semantic index is built (see "Optional semantic tier" below). |

The workflow below is the canonical happy path through these.

## Workflow

Tool calls below show the primary API (PydanticAI tools attached to
the agent). The equivalent CLI is shown alongside for the
human-driven workflow.

1. **Search the wiki first.** Always. Ask a natural-language question:

   ```python
   search_wiki(question="What is our cost-plus pricing formula?")
   ```

   Returns the most relevant whole pages as `[[slug]]` citations with a
   short excerpt, ranked by the wiki's configured retrieval strategy.
   This is the primary recall move — `read_page` the interesting hits.

   *New to this wiki?* `search_index()` first shows its shape — the
   top-level namespaces with page counts — so you can `search_index(prefix="…")`
   to drill in, rather than guessing search terms blind.

2. **Need an exact line, or to search raw sources / the log?** Use
   `grep_wiki` — literal/regex ripgrep, not relevance ranking:

   ```python
   grep_wiki(pattern="cost-plus 35%", scope="wiki")   # exact line in a page
   grep_wiki(pattern="cost-plus", scope="raw")         # the raw/ source documents
   grep_wiki(pattern="...", scope="log")               # the gap log
   ```

   `scope="raw"` is Tier 2: uncompiled source material — slower to read
   and less authoritative than a wiki page. If you find an answer there,
   that's a strong signal to write the compacted version back to the
   wiki at the end of the turn (see the `write` skill).

   Equivalent CLI: `outmem search "<pattern>" --scope wiki|raw|log`.

3. **Read the candidate page** (skim first if you're unsure):

   ```python
   read_page(slug="pricing-formula", peek=True)   # title + first ~1000 chars
   read_page(slug="pricing-formula")               # the full file
   ```

   The full read returns the whole file (frontmatter + body); its
   frontmatter `provenance` field tells you which raw files the page
   was compiled from. Use `peek=True` to triage a candidate cheaply —
   confirm it's the right page before committing the whole body to
   context.

   Equivalent CLI: `outmem read "<slug>"`.

4. **Check what links to the page** (often points at related context):

   ```python
   find_backlinks(slug="pricing-formula")
   ```

   Lists wiki pages whose body contains `[[pricing-formula]]`.

   Equivalent CLI: `outmem search "[[<slug>]]" --scope wiki --fixed-strings`.

## Optional semantic tier (when the semantic index is built)

If the wiki has a semantic index (built with `outmem reindex`), you
also have `find_similar`:

```python
find_similar(text="cost-plus 35% pricing", top_k=5, exclude_slug=None)
```

Use when the user's question is paraphrased or you suspect a related
chunk exists under a different name than ripgrep would catch. The
tool returns cosine-similarity matches across both wiki pages and
registered sources.

## Anti-patterns

- **Don't reach for `raw/` before `wiki/`.** You will re-derive an
  answer that's already compiled, costing context and time.
- **Don't surface contradictions silently.** If `wiki/` and `raw/`
  disagree, lead the response with the contradiction — that's the
  highest-signal finding the turn can produce.
- **Don't chain more than three searches without a candidate
  answer.** If three calls haven't surfaced one, stop and say what
  you tried.

## After answering

Mandatory writeback (spec §9): every turn ends with at least one git
commit. If you used the search but didn't learn anything that
warrants a wiki write, log the search itself as a one-line entry —
see the `write` skill, section "When nothing rises to a wiki page".

See `references/patterns.md` for the convergence-vs-expansion
decision rule. See the `evolution` skill if the question is about
how thinking has changed over time rather than about the current
state.
