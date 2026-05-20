---
name: write
description: >
  Compact a finding, decision, or observation back into the outmem
  wiki. Use when the user says "remember that we decided X", "add this
  to memory", "save this finding", "write this down" — or any time a
  turn produced understanding worth keeping for the next one. Also
  covers log entries for findings that don't yet rise to a wiki page.
---

# write — close the loop

Mandatory writeback (spec §9): every turn ends with at least one git
commit. Without it, agentic search is an expensive way to re-derive
the same answer on every query. The wiki is *your* compounding
artifact; maintain it.

## Three writeback paths

Tool calls below show the primary API (PydanticAI tools attached to
the agent). The equivalent CLI is at the bottom of each section for
the human-driven workflow.

### A. New page — when the turn produced an uncovered topic

```python
write_page(
    slug="pricing-formula",
    title="Pricing formula",
    body="The 2026 pricing formula is cost-plus 35%.\n\nApplies to all SKUs.\n",
    provenance=["raw/pricing-deck-2026-Q1.md"],
    tags=["pricing", "contracts"],
)
```

**Required in every call: `slug`, `title`, AND `body`.** The body is
not stdin; it is a named argument and you MUST include it in the same
tool call. Omitting `body=` is the most common write_page failure
mode — the call fails validation and the model gets a single retry
before the run errors out.

Slugs are one or more `:`-separated segments; each segment is
lowercase, hyphen-separated. Flat (`pricing-formula`) or namespaced
(`abx:penicillin`, `abx:side-effects:misc`) — the namespace becomes
a directory on disk under `wiki/pages/`. Nest eagerly: if a parent
topic exists (per `wiki/AGENTS.md` or the slugs already in the
wiki), use it from the first page you write on that topic.

The body should be a complete page — not a stub. Provenance can be
repeated for multiple sources. The commit message becomes
`compact: <slug>`.

Equivalent CLI (body via stdin):

```bash
outmem write <slug> \
    --title "<Human-readable title>" \
    --provenance raw/<source-file>.md \
    --tag <tag> \
    <<< "<complete markdown body>"
```

### B. Edit existing page — when the turn refined or extended an existing topic

```python
extend_page(
    slug="pricing-formula",
    body="The pricing formula is now cost-plus 40%, revised Q2 2026.\n",
)
```

**Required in every call: `slug` AND `body`.** Body is a complete
replacement of the page's body section — there is no partial-edit
primitive at v0.1. To keep the old content, paste it into your
replacement before the new material.

Frontmatter (title, provenance, tags, created) is preserved;
`updated` is bumped. Commit message becomes `extend: <slug>`.

Equivalent CLI:

```bash
outmem extend <slug> <<< "<complete replacement body>"
```

### C. Log entry — when nothing rose to a wiki page

```python
append_log(
    topic="pricing-inconsistency",
    content="- noticed acme-msa cites cost-plus 30%, pricing-formula says 35%.\n",
)
```

**Required in every call: `topic` AND `content`.** Use for
observations, contradictions surfaced, questions raised but
unanswered, decisions not yet codified, or "no new compaction needed"
notes. Appends to `log/<today>.md`. Commit message becomes
`log: <topic>`.

**Critical:** "I didn't learn anything new this turn" is a legitimate
outcome — and it still produces a one-line log entry. Silent runs
corrupt the writeback-rate metric.

Equivalent CLI:

```bash
outmem log <topic> <<< "- <one-line observation>"
```

## When to write what

- Synthesis of multiple sources into one statement → **new wiki page**.
- Correction or refinement of an existing page → **extend**.
- Observed contradiction, decision, "TODO" item, or open question → **log**.
- Search that turned up nothing actionable → **log** (one line).

## Common mistakes

- **Forgetting `body=` on `write_page` or `extend_page`.** Body is a
  named argument, not stdin. Always include it in the same tool call,
  with the complete page text. The schema will reject the call without
  it.
- **Treating `extend_page` as a partial edit.** It replaces the whole
  body; copy any content you want to keep into the new body.
- **Adding speculation to wiki pages.** If the source doesn't support
  the claim, route the observation to `append_log` instead.

## The frontmatter you generate

When you write a new page, outmem fills in `slug`, `created`, and
`updated` from arguments and the current time. You supply:

- `title` (required, human-readable)
- `provenance` (list of raw/ paths, or dict entries with richer
  ingestion metadata; preserve any frontmatter the raw files
  carried — see `references/frontmatter.md`)
- `tags` (optional)

There is **no `authority` field** in v0.1 — humans and the agent both
edit any page, and git history is the audit trail.

## After your write

`outmem` commits under the agent identity (`OUTMEM_AGENT_NAME` /
`OUTMEM_AGENT_EMAIL`, defaulting to `outmem agent <agent@host>`). The
standalone runtime (`outmem ask` / `outmem ingest`) handles the
pull-rebase-push loop for you. If you're invoking the tools from a
custom agent, run:

```bash
outmem pull && outmem push
```

If `push` fails after a single pull-rebase retry, outmem raises a
hard error — do not respond to the user as if writeback had succeeded
(spec §9). Surface the error.

See:
- `references/frontmatter.md` for the page model.
- `references/commit-grammar.md` for the `compact:`/`extend:`/`log:`
  conventions (TARS Retained depends on these prefixes).
- `references/rebase-loop.md` for the pull-rebase-push contract.
