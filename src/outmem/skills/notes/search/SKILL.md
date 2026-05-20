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

## Workflow

Tool calls below show the primary API (PydanticAI tools attached to
the agent). The equivalent CLI is shown alongside for the
human-driven workflow.

1. **Search the wiki first.** Always.

   ```python
   search_wiki(pattern="cost-plus", scope="wiki")
   ```

   Returns `path:line:text` rows, one per match. Hits in `wiki/`
   mean we already have a compiled answer; read the page and you're
   done.

   Equivalent CLI: `outmem search "<pattern>" --scope wiki`.

2. **If the wiki is silent, fall through to raw sources.**

   ```python
   search_wiki(pattern="cost-plus", scope="raw")
   ```

   Tier 2. Raw files are uncompiled source material — useful, but
   slower to read and less authoritative than a wiki page on the same
   topic. If you find an answer here, that's a strong signal you
   should also write the compacted version back to the wiki at the
   end of the turn (see the `write` skill).

   Equivalent CLI: `outmem search "<pattern>" --scope raw`.

3. **Read the candidate page.**

   ```python
   read_page(slug="pricing-formula")
   ```

   Returns the full file (frontmatter + body). The frontmatter
   `provenance` field tells you which raw files the page was
   compiled from.

   Equivalent CLI: `outmem read "<slug>"`.

4. **Check what links to the page** (often points at related context):

   ```python
   find_backlinks(slug="pricing-formula")
   ```

   Lists wiki pages whose body contains `[[pricing-formula]]`.

   Equivalent CLI: `outmem search "[[<slug>]]" --scope wiki --fixed-strings`.

## Optional semantic tier (when enabled in `config.yaml`)

If the wiki has `semantic.enabled: true`, you also have
`find_similar`:

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
