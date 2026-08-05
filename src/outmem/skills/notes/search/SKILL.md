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

Compiled knowledge is cheaper to read than source documents are to
re-derive, and the wiki is *your* prior work — start there and only
fall through to the source documents when the wiki was silent.

## Tools at a glance

The full search/navigation/read surface — reach for the right one:

| Tool | Reach for it when |
|------|-------------------|
| `search_wiki(question, k)` | **Primary recall.** Question-shaped lookup — ranks whole pages by relevance using the wiki's configured strategy (default `rerank(bm25)`). |
| `grep_wiki(pattern, scope, context)` | The *exact line* a literal/regex string sits on, or to search the source documents / the gap `log` (`scope="sources"` / `"log"`). `context=N` returns N lines either side. |
| `search_index(prefix)` | **Browse the structure.** Walk the slug namespaces (the table of contents) one level at a time to orient on an unfamiliar wiki. |
| `list_pages()` | A flat dump of *every* slug — a cheap existence check. |
| `read_page(slug, peek, section)` | Open a page. `peek=True` returns its **outline** (every heading, line span, size); `section="<heading>"` returns one section. |
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

2. **Need an exact line, or to search the sources / the log?** Use
   `grep_wiki` — literal/regex ripgrep, not relevance ranking:

   ```python
   grep_wiki(pattern="cost-plus 35%", scope="wiki")   # exact line in a page
   grep_wiki(pattern="§ 7 Abs. 1", context=2)          # …and its neighbourhood
   grep_wiki(pattern="cost-plus", scope="sources")     # source documents, both trees
   grep_wiki(pattern="...", scope="log")               # the gap log
   ```

   **Pass `context` when you intend to quote.** A matched line is often
   half a sentence — a threshold, a deadline, a cross-reference runs into
   the next line. `context=2` returns that in the same call; opening the
   page afterwards to see what your own hit continues into costs a whole
   extra round-trip. Matches show as `slug:line:text`, context rows as
   `slug-line-text`, and the slug is the same on both, so you can still
   feed it to `read_page`.

   `scope="sources"` is Tier 2: uncompiled source material — slower to read
   and less authoritative than a wiki page. If you find an answer there,
   that's a strong signal to write the compacted version back to the
   wiki at the end of the turn (see the `write` skill).

   Equivalent CLI: `outmem search "<pattern>" --scope wiki|sources|log`.

3. **Read the candidate page.**

   ```python
   read_page(slug="pricing-formula")                          # the full file
   read_page(slug="meldewesen:ifsg", peek=True)               # what's in it?
   read_page(slug="meldewesen:ifsg", section="§7 Abs. 1")     # just that part
   ```

   The full read returns the whole file (frontmatter + body); its
   frontmatter `provenance` field tells you which source files the page
   was compiled from.

   `peek=True` returns the page's **outline** — every heading with its
   line span and size — not a preview of the text. Reach for it when a
   page is large and you need to know *which part* to spend context on;
   then `section=` reads only that part. If you already know the page is
   right and it isn't huge, just read it: a peek followed by a full read
   of the same page is two round-trips for one answer. `search_wiki` has
   usually already told you the page is on topic, since it returns an
   excerpt.

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

- **Don't reach for the sources before the pages.** You will re-derive an
  answer that's already compiled, costing context and time.
- **Don't surface contradictions silently.** If a page and a source
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
