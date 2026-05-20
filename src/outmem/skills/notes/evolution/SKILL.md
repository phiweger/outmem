---
name: evolution
description: >
  Trace how thinking on a topic has changed over time in the outmem
  wiki. Reads `git log -p --follow` against wiki pages and `log/`
  entries to surface the chronological diff stream. Use when the user
  asks "how has our thinking on X changed", "what did we say about Y
  last week", "show the history of Z", or any question where the
  answer is about the trajectory rather than the current state.
---

# evolution — how thinking has shifted

The wiki's current state answers convergent questions; its *history*
answers divergent ones. This skill is the EXPANSION branch of the
retrieval discipline (see `search` skill, `references/patterns.md`).

## When to use this skill

Reach for evolution when:

- The question is about change ("how has X evolved", "what changed
  last month", "when did we stop saying Y").
- The framing itself is in question ("are we contradicting our older
  position on …").
- The user wants to compare two points in time.

If the user just wants the current value, use `search_wiki` /
`read_page` instead — evolution is more expensive and produces a
noisier output.

## Workflow

Tool calls below show the primary API (PydanticAI tools attached to
the agent). The equivalent CLI is shown alongside for the
human-driven workflow.

1. **Identify the relevant slugs.** If you don't know which, use
   `search_wiki(...)` or `list_pages()` first to surface candidates.

2. **Walk the chronological diff.**

   ```python
   topic_evolution(slugs=["pricing-formula", "discounts"], include_log=True)
   ```

   **Required: `slugs` (a list of one or more wiki slugs).**
   `include_log` defaults to `True` so log entries are interleaved
   with wiki page diffs; pass `False` for a wiki-only stream when
   the log noise hurts.

   Returns the raw `git log -p --follow` stream — every commit that
   touched the page(s), with the diff content.

   Equivalent CLI: `outmem evolution <slug> [<slug> ...] [--no-log]`.

3. **Or look at one page's history without the diffs.**

   ```python
   page_history(slug="pricing-formula")
   ```

   Returns `sha  iso-date  author <email>  subject` rows, newest
   first — useful for "who edited this and when" without the diff
   content.

   Equivalent CLI: `outmem history <slug>`.

4. **Read the diff stream as a story.** Each commit block tells you
   who edited, when, and what they changed. Pay attention to:
   - **Reversals**: a line that was added then deleted is a stronger
     signal than a line never written.
   - **Authorship shifts**: a page edited by Alice and later
     rewritten by Bob is a divergence to surface.
   - **Log entries between wiki commits**: those captured the *why*
     of an edit and often contain decision context the wiki page
     omitted.

5. **Surface contradictions in the response.** If you find that the
   page used to say X and now says Y, *and the user's question is
   adjacent*, lead the answer with the change — don't just report
   the current state.

## Anti-patterns

- **Don't run `topic_evolution` on a slug you haven't located yet.**
  Confirm the slug exists (`read_page(slug=...)` or `search_wiki`)
  first; otherwise you'll get an error and have to retry.
- **Don't include `log/` for purely-wiki questions.** Pass
  `include_log=False` to keep the diff stream tight when log
  entries would add noise.
- **Don't paraphrase the diff at length.** The diff *is* the answer
  for divergent questions — quote the meaningful changes directly.

See `references/git-log.md` for tips on reading `git log -p` output
efficiently.
