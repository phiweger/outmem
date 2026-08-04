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

## Extra fields are preserved

Any frontmatter key outside the canonical set above is preserved
verbatim in an `extra` bag and serialised back out as-is. This is the
seam for ingestion-supplied metadata you don't want to lose: ingestion
pipelines can stamp arbitrary keys on raw files and outmem will
propagate them.
