# Reading `git log -p --follow` efficiently

The diff stream is verbose. A few patterns help you skim it without
loading every byte into context.

## Anatomy of one commit block

```
commit 1a2b3c4d…                      ← sha (first 7 chars are enough)
Author: Alice <alice@example.com>     ← who
Date:   Thu May 11 09:14:00 2026     ← when
                                      ← subject + body
    compact: pricing-formula
                                      ← unified diff
diff --git a/wiki/pricing-formula.md b/wiki/pricing-formula.md
@@ -3,7 +3,7 @@                       ← hunk header (line numbers)
-old line                             ← removed
+new line                             ← added
```

## What to look for

- **Commit subjects matter.** Outmem encodes intent in the subject
  prefix: `compact:` (new page), `extend:` (edit), `log:` (decision/
  observation). Scanning subjects gives you the *story* without
  reading any diffs.
- **Hunks tell you where in the file the change happened.** A change
  in the frontmatter section is qualitatively different from a change
  in the body.
- **Author changes between commits are signal.** Two humans alternating
  on the same page often means the question itself is contested.
- **`log:` commits sandwiched between `extend:` commits often capture
  the rationale for the wiki edit on either side.**

## When to read everything vs. skim

If the user is asking "what changed last week", skim subjects only.
If they're asking "why was this rewritten", you need to read the
hunks — and probably the `log:` commits adjacent to the rewrite.

## Tip: pipe through your shell to filter

```bash
outmem evolution pricing-formula | grep -E '^(commit |Author:|Date:|    )' \
    | head -80
```

This drops the diff hunks and just shows the metadata + subject
lines — a fast structural skim before deciding whether to read full
diffs.
