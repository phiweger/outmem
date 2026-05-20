# Commit message grammar

The wiki's commit messages encode intent so that `git log --grep`
filters are reliable. The TARS *Retained* metric (spec §10) is one
`git log --grep` call away from being computable — that breaks the
moment the prefix grammar drifts.

## The three prefixes

| Prefix | Produced by | Meaning |
|---|---|---|
| `compact: <slug>` | `outmem write` | A new wiki page was created. |
| `extend: <slug>` | `outmem extend` | An existing wiki page was edited. |
| `log: <topic>` | `outmem log` | A log entry was appended; no wiki write. |

These are pinned in the writeback service — every commit produced by
outmem matches one of these three shapes. The agent should never need
to write a commit message manually; the CLI handles it.

## What "topic" should be

`log: <topic>` is the only one where you pick the text. Keep it short
(2–5 words, hyphen-separated) and topical:

- `log: pricing-inconsistency`
- `log: contradiction-acme-msa`
- `log: no-new-compaction`
- `log: clarification-requested`

It becomes the search anchor when someone later runs
`git log --grep='^log: pricing'`. Vague topics ("notes", "stuff")
break that.

## Why this matters operationally

`git log --author=agent --grep='^compact:'` counts wiki creations.
`git log --author=agent --grep='^log:'` counts non-compaction runs.
Their ratio is the *Retained* signal. If you commit with a non-matching
subject (e.g. via raw `git commit` outside the CLI), it falls into the
"didn't happen" bucket and skews the metric.

The fix is procedural, not technical: always go through `outmem
write`, `outmem extend`, or `outmem log` for wiki/log changes.
