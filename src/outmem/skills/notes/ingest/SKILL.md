---
name: ingest
description: >
  Bring a raw source document into the wiki — read a registered file
  under `wiki/sources/`, extract the durable facts, write or extend the
  relevant pages, and record the ingestion so the source ↔ page link
  survives. Use when the user says "ingest this file", "process this
  source", "load this into the wiki", or any time a registered source
  hasn't yet been compacted into pages.
---

# ingest — fold a source into the wiki

Sources under `wiki/sources/` are raw material the agent hasn't yet
compiled. Ingestion is the workflow that turns them into wiki pages and
records the link in the ingestions registry (`wiki/sources/.sources.db`),
so a later question like "where does this fact come from?" answers itself
from the page's `provenance` field.

## Workflow

Tool calls below show the primary API (PydanticAI tools attached to
the agent). The equivalent CLI is shown alongside for the
human-driven workflow.

1. **List what's registered.**

   ```python
   list_sources()
   ```

   Returns one `relative/path  sha:<short>  <size>B  N ingestion(s)` row
   per registered source, with the prompts each prior ingestion ran
   under. A row with `0 ingestion(s)` is the obvious candidate; a row
   with prior ingestions tells you what's already been extracted (and
   under what focus directive) so you don't duplicate work.

   Equivalent CLI: `outmem sources`.

2. **Read the source you're ingesting.**

   ```python
   read_source(rel_path="veterinary/cat-drugs.md")
   ```

   Returns the full source text. Big sources are capped at
   `sources.max_chars`; if you hit the cap, narrow the prompt or pick a
   different section.

   Equivalent CLI: `outmem read-source veterinary/cat-drugs.md`.

3. **Check what already exists** before writing — ingestion is meant to
   *extend* the wiki, not duplicate it. Use the search workflow:

   ```python
   search_index(prefix="abx")          # any existing namespace for this topic?
   search_wiki(question="cat penicillin dose")   # any existing page?
   find_similar(text="<a key sentence from the source>")   # paraphrase match
   ```

   See the `search` skill for the full retrieval workflow.

4. **Write or extend the relevant pages.** Each durable fact in the
   source belongs in **one** page — synthesise across multiple sources
   on the same topic, don't append source-by-source:

   ```python
   # New topic — a page that didn't exist
   write_page(
       slug="abx:penicillin:cat-dose",
       title="Penicillin dosing in cats",
       body="<complete page body>\n",
       provenance=["veterinary/cat-drugs.md"],
       tags=["abx", "veterinary"],
   )

   # Existing topic — refine or add to it
   extend_page(
       slug="abx:penicillin",
       body="<complete replacement body, preserving what's worth keeping>\n",
   )
   ```

   Critical: `provenance` lists the **source rel_paths** you're
   compacting from. The same path you passed to `read_source` is what
   goes here. See the `write` skill for the full writeback rules
   (required args, slug grammar, body completeness, etc.).

5. **Record the ingestion.** Last step, after the page writes commit:

   ```python
   record_ingestion(
       rel_path="veterinary/cat-drugs.md",
       prompt="extract penicillin dosing for cats",
       pages_touched=["abx:penicillin:cat-dose", "abx:penicillin"],
   )
   ```

   REQUIRES ALL THREE: `rel_path`, `prompt`, `pages_touched`. Appends a
   row to `wiki/sources/.sources.db` and commits the registry update as
   `ingest: <rel_path>`. The `prompt` is the *focus directive* — what
   the agent was asked to extract — so a later ingestion under a
   different prompt (e.g. "extract drug interactions" rather than
   "extract dosing") can layer in without re-doing the first one.
   `pages_touched` is exactly the slugs you wrote or extended in this
   turn — not the slugs you read.

## Local-only sources

Some material may be read but not redistributed — licensed corpora,
copyrighted text, embargoed drafts. That lives in
`wiki/sources-local/`, a sibling tree git never sees (`outmem ingest
--local` puts it there).

For you at the keyboard, it changes almost nothing: it is greppable
via `grep_wiki(scope="sources")` like everything else, readable via
`read_source`, and citable in `provenance:` — a citation is not a
redistribution. `list_sources` prefixes each row so you can tell
which tree it came from.

What *does* change is how you compact it. A page built from a
local-only source travels even though the source does not, so it must
carry your synthesis rather than the original's prose. Summarise,
restructure, quote briefly where exact wording carries the meaning —
do not transcribe long passages into a page. The wiki is the thing
you are allowed to hand someone; keep it that way.

Two things are handled for you, so don't work around them: local
sources never enter the semantic index (the vector DB stores chunk
text verbatim and is committed), and `outmem lint` errors if any of
those bytes reach git.

## When NOT to ingest

- **The source restates an already-compacted page.** Skip — record an
  ingestion with `pages_touched=[]` and `prompt="<why-nothing-new>"` so
  the row reflects the no-op, then `append_log` a one-line note.
- **The source contradicts the wiki.** Surface the contradiction in
  `append_log` *first*; only then choose extend vs leave-alone. Don't
  silently overwrite — a page/source disagreement is the highest-signal
  finding the turn can produce.
- **The source is non-text / binary.** `read_source` will reject it; pick
  a different file or pre-convert the source outside outmem.

## Common mistakes

- **Forgetting `record_ingestion`.** The page provenance still works (the
  source is named in frontmatter) but the registry stops being a useful
  audit trail — `list_sources` then shows `0 ingestion(s)` forever, and
  the wiki can't tell whether a source has been processed.
- **One page per source.** A source covers multiple facts; a page covers
  one. If the source has dosing AND interaction AND side-effects, that's
  three pages (or three extensions to existing pages), not one
  source-shaped dumping ground.
- **Logging the source verbatim.** Compaction means re-stating in your
  own words against the wiki's grammar; verbatim copies belong in
  the source trees, not `wiki/pages/`.

## After ingesting

The writeback rules from the `write` skill apply — outmem commits each
page under the agent identity (`compact:`/`extend:`/`log:`), then the
`record_ingestion` adds a final `ingest:` commit. The standalone runtime
(`outmem ingest`) handles the pull-rebase-push loop for you.

See:
- `references/registry.md` for the source-and-ingestion data model
  (what `record_ingestion` actually writes, how `prompt` /
  `pages_touched` are scoped, and the N-sources-into-one-page pattern).
- The `search` skill for step 3's existence checks.
- The `write` skill for step 4's writeback contract (required args,
  body completeness, slug grammar, the `provenance` field shape).
