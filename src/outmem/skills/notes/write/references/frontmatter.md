# The wiki page model

Every page under `wiki/pages/<slug-as-relpath>.md` begins with YAML
frontmatter (the slug's `:` separators map to `/` on disk; e.g.
`abx:penicillin` → `wiki/pages/abx/penicillin.md`):

```yaml
---
title: Pricing formula
slug: pricing-formula
provenance:
  - sources/a1b2c3d4e5f6/pricing-deck-2026-Q1.md
  - sources/f6e5d4c3b2a1/acme-msa.md
created: 2026-04-12T09:14:00Z
updated: 2026-05-04T11:32:00Z
tags: [pricing, contracts, finance]
---
```

## Fields

| Field | Required | Notes |
|---|---|---|
| `title` | yes | Human-readable display name. |
| `slug` | yes | Lowercase, hyphen-separated; matches the filename. |
| `provenance` | optional | List of pointers into the source trees — see below. |
| `created` | optional | ISO 8601 with `Z` suffix (UTC). Auto-set on new pages. |
| `updated` | optional | ISO 8601 with `Z` suffix. Auto-bumped on every edit. |
| `tags` | optional | List of strings. Free-form. |

No `authority` field — that was dropped in spec v0.5.

## Provenance — preserve upstream metadata verbatim

Plain string entries point at a source file:

```yaml
provenance:
  - sources/a1b2c3d4e5f6/pricing-deck-2026-Q1.md
```

But if the upstream ingestion pipeline embedded its own frontmatter in
the source file (drive paths, content hashes, page ranges, focus
instructions), preserve those richer entries as **dicts** rather than
flattening to strings:

```yaml
provenance:
  - path: sources/f6e5d4c3b2a1/acme-msa.md
    drive_path: /shared/contracts/acme/2026/MSA.pdf
    sha256: a1b2c3…
    page_range: 4-7
```

Outmem round-trips dict entries verbatim — you do not interpret or
generate this metadata, you just carry it through. The CLI's
`--provenance sources/<sha>/foo.md` flag adds plain string entries; richer dicts
require writing the YAML directly (use the Python API `write_page(...,
provenance=[{...}])` when needed).

### `finding:` — when you checked and the source said nothing

If you read a source expecting an answer and it **doesn't give one**,
that is a fact worth recording, not a dead end. Write it:

```yaml
provenance:
  - path: sources/7b6cb641da16/awmf-s3-hwi-2024.md
    finding: silent          # silent | contradicts | out-of-scope
    scope: "Therapiedauer der Pyelonephritis in der Schwangerschaft"
    note: "S3 fuehrt unter 12.2 ausdruecklich 'Keine Empfehlungen/Statements'."
    date: 2026-08-10
```

Why it matters: without it, "we checked and there is nothing" looks
exactly like "nobody looked" — so the next reader answers from their own
knowledge instead of reporting the gap. `outmem stale` also flags these
for re-check when the source gets a new version, since a *new* edition
is precisely where a previously-missing answer would appear.

`finding` must be one of the three values above (`outmem lint` warns
otherwise — an unrecognised value records nothing). `scope`, `note` and
`date` are free-form.

**Also give it a heading.** The frontmatter is metadata; search matches
body text. State the absence in a section whose heading names the
question:

```markdown
## Therapiedauer Pyelonephritis in der Schwangerschaft — keine Empfehlung

Die S3-Leitlinie trifft hierzu ausdruecklich keine Aussage (12.2).
```

That way someone asking the question finds the statement that nothing is
stated, which is the answer they need.

## Extra fields are preserved

Any frontmatter key outside the canonical set above is preserved
verbatim in an `extra` bag and serialised back out as-is. This is the
seam for ingestion-supplied metadata you don't want to lose: ingestion
pipelines can stamp arbitrary keys on raw files and outmem will
propagate them.
