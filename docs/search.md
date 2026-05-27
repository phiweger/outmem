# Search & retrieval

Retrieval in outmem is a **workflow**, not a single call: *locate* a
page, *read* it in full, *traverse* to neighbours. `search_wiki` finds
where something lives; `read_page` is what actually feeds the model.
This page explains the loop, the three retrieval tiers (one default,
two opt-in), and how to choose between them.

## The core loop (always available)

```
search_wiki(pattern) ──▶ slug ──▶ read_page(slug) ──▶ traverse
   locate                            full text         find_backlinks / [[links]] / history
```

1. **`search_wiki(pattern, scope="wiki")`** — ripgrep over the wiki.
   For `scope="wiki"` each hit row is **slug-shaped** so it drops
   straight into the next step:

   ```
   abx:penicillin:14:IV penicillin G 18-24 MU/day in divided doses
   ```

   The leading token (`abx:penicillin`) is a slug, not a path — pass it
   verbatim to `read_page`. Output is capped at **8 KiB** (a soft
   token ceiling); past that you get a trailing `(truncated — narrow
   the pattern)` and should tighten the query.

2. **`read_page(slug)`** — the full page text. This is the step that
   matters: `search_wiki` only tells you *where* to look; reasoning
   should happen over the whole page, not a grep line.

3. **Traverse** — `find_backlinks(slug)` (who links here),
   `[[slug]]` wikilinks in the body, `page_history(slug)` (git log for
   one page), `topic_evolution(slugs...)` (diff stream across pages).

`list_pages()` enumerates every slug when you'd rather browse than
search.

### Scopes

`search_wiki` / `outmem search` take `--scope`:

| scope  | searches                | row shape                     |
| ------ | ----------------------- | ----------------------------- |
| `wiki` | curated pages (default) | slug — `abx:penicillin:14:…`  |
| `raw`  | unprocessed sources     | path — `raw/deck.md:3:…`      |
| `log`  | the append-only log     | path                          |
| `all`  | everything              | mixed                         |

Only `wiki` scope is slug-shaped (and only `wiki` scope is touched by
the relevance filter below); `raw`/`log`/`all` return real paths.

## Three retrieval tiers

| tier               | how it matches            | recall            | cost / deps                | turn it on            |
| ------------------ | ------------------------- | ----------------- | -------------------------- | --------------------- |
| **Lexical**        | ripgrep keyword / regex   | exact tokens only | free, always on            | default               |
| **Relevance filter** | lexical net → cheap-model yes/no gate | keyword-bounded, **higher precision** | one small-model call per search; `outmem[agent]` | `relevance.enabled` |
| **Semantic**       | vector cosine similarity  | **paraphrase / synonym recall** | embeddings + local vector DB; `outmem[semantic]` | `semantic.enabled` |

They are not alternatives you pick once — they compose:

* The **relevance filter wraps lexical**. When enabled it *replaces*
  the `search_wiki` tool with a variant that casts a wider ripgrep net,
  filters it through a small model, and returns only the relevant
  pages. Same `pattern` in, same `read_page(slug)` out — drop-in.
* **Semantic is a parallel door**, reached through a separate
  `find_similar(text)` tool, not through `search_wiki`. Use it when the
  word you'd grep for isn't the word on the page.

Full setup, YAML, and reliability details for the two opt-in tiers live
in [features.md](features.md#relevance-filter) and
[features.md](features.md#semantic-index); this page covers *when* and
*how they fit the workflow*.

Rather than pick by hand, you can let outmem search this space for you:
[retrieval tuning](features.md#retrieval-tuning) scores the blocks on a
question bank built from your wiki and reports the best config.

## Choosing a tier

* **You know the keyword** (`penicillin`, an error code, a customer
  name) → plain **lexical** `search_wiki`. Nothing else needed.
* **The keyword returns a noisy pile** and you're paying to triage it
  in the expensive model's context → turn on the **relevance filter**.
  It moves triage to a cheap model and only the relevant pages (each
  with a one-line *why*) reach the agent. Buys **precision + context
  savings, not recall** — if ripgrep wouldn't have found it, the filter
  can't either.
* **You're searching by meaning** — "have we seen something like this
  before?", a paraphrase, a synonym, a near-duplicate you want to avoid
  writing again → **semantic** `find_similar`. This is the only tier
  that recalls pages that share no keywords with your query.

A useful pattern: lexical/relevance to *answer*, semantic to *avoid
duplicating* (the system prompt nudges agents to `find_similar` before
`write_page`).

## The relevance filter, in the workflow

Before (plain lexical) the agent gets raw rows and triages them itself:

```
abx:penicillin:14:IV penicillin G 18-24 MU/day in divided doses
abx:ceftriaxone:9:ceftriaxone 2g IV q24h
pricing-formula:5:cost-plus 35% applied to penicillin product sales
```

After (filter on) the agent gets only the relevant pages, each with a
*why* and its supporting line — and `read_page(slug)` is still the next
step:

```
abx:penicillin  why: IV penicillin dosing for endocarditis
  abx:penicillin:14:IV penicillin G 18-24 MU/day in divided doses
```

Two invariants worth knowing as a consumer:

* **It's a filter, not a ranker, and it never writes content.** The
  model answers relevant *yes/no* per page (no score, no re-ordering)
  and emits only `{slug, reason}`. Every line shown is a verbatim
  ripgrep hit; full text always comes from `read_page` reading disk.
  The one-line `reason` is the only model-generated string.
* **Failure degrades, it never breaks.** A filter error/timeout falls
  back to the lexical hits trimmed to the normal 8 KiB budget; an empty
  result means "nothing relevant" (a weak keyword is not laundered into
  a false positive), not an error.

## CLI quick reference

The CLI exposes the **lexical tier only** — `outmem search` is raw
ripgrep (path-shaped rows, exit code 1 on no matches) and does **not**
apply the relevance filter; that wrapper lives in the agent tool
(`search_wiki`) so it can spend a model call. Semantic has its own
command.

```bash
outmem search "penicillin" --scope wiki        # lexical; -i, -F, --max-hits N
outmem read abx:penicillin                      # full page by slug
outmem similar "beta-lactam alternative"        # semantic (needs [semantic])
outmem similar --slug abx:penicillin            # use a page's body as the query
outmem history abx:penicillin                   # git log for one page
outmem evolution abx:penicillin abx:ceftriaxone # diff stream across pages
```

## Embedding the tools in your own agent

The PydanticAI adapter hands you the same palette as plain callables:

```python
from outmem import WikiStore
from outmem.adapters.pydantic_ai import wiki_read_tools, wiki_tools

store = WikiStore.open("/srv/wiki")
tools = wiki_read_tools(store)        # retrieval only (read-only consult)
# tools = wiki_tools(store)           # + writeback paths
```

* **Relevance filter**: pass `relevance=RelevanceConfig(model=…,
  on_filter=…)` to either factory to swap in the filtered
  `search_wiki` and tap every `FilterOutcome` for tracing. With no
  argument it follows the wiki's `relevance:` config block (off by
  default ⇒ byte-for-byte the plain tool). See
  [features.md](features.md#relevance-filter).
* **Black-box consult**: `build_consult_wiki(path)` returns a single
  `consult_wiki(question) -> str` tool that runs an inner agent over
  this whole workflow and returns a cited answer — when you want "ask
  the knowledge base", not the raw retrieval primitives. See
  [python-api.md](python-api.md).

## Failure & edge-case cheatsheet

| you see                                    | meaning                                                        |
| ------------------------------------------ | -------------------------------------------------------------- |
| `(no matches)`                             | ripgrep found nothing — broaden, drop `-F`, or try `find_similar` |
| `(no relevant pages — N candidate(s) …)`   | filter ran, judged none relevant — rephrase; the empty signal is real |
| `(truncated — narrow the pattern)`         | hit the 8 KiB cap — tighten the query or raise `--max-hits`     |
| `(search failed: …)`                       | ripgrep not installed / bad regex — the message says which     |
| filtered results but no `why:` lines       | the filter fell back to lexical (a model error) — still usable |
