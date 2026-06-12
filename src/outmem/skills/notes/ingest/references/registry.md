# The ingestions registry — what `record_ingestion` writes

The registry lives at `wiki/sources/.sources.db` (sqlite). It tracks
two things:

1. **Sources** — every file under `wiki/sources/`, by its
   `relative/path` from the wiki root, with a content `sha256` and the
   file `size_bytes`. Added automatically when a file lands under
   `wiki/sources/` and committed to git; you do not call a tool to
   register a source.

2. **Ingestions** — append-only rows that link a source to the wiki
   pages a turn produced from it. Each row carries the focus directive
   the agent ran under so the same source can be re-processed under a
   different prompt without redoing the first ingestion's work.

`record_ingestion` writes rows of the second kind:

```python
record_ingestion(
    rel_path="veterinary/cat-drugs.md",
    prompt="extract penicillin dosing for cats",
    pages_touched=["abx:penicillin:cat-dose", "abx:penicillin"],
)
```

The arguments map to a registry row roughly like:

```text
source_id   = (the sha256-resolved id for veterinary/cat-drugs.md)
prompt      = "extract penicillin dosing for cats"
pages       = ("abx:penicillin:cat-dose", "abx:penicillin")
created_at  = <now>
```

## Field rules

| field | rule | example |
|---|---|---|
| `rel_path` | exact registered path (same string you passed to `read_source`) | `veterinary/cat-drugs.md` |
| `prompt` | the focus directive — a noun phrase or imperative ≤ ~12 words; what you were asked to extract, not how you did it | `"extract penicillin dosing for cats"` |
| `pages_touched` | every slug your turn wrote or extended; **not** slugs you only read | `["abx:penicillin:cat-dose"]` |

`prompt` is what later `list_sources` rows show next to the source —
think of it as the title of the ingestion run. `pages_touched=[]` is
allowed when the source was a confirmed no-op (already-compacted,
contradicts something better-sourced, etc.) — pair it with a
`prompt="no-op: …"` that says why.

## Why this matters

- `list_sources` shows ingestion counts and prompts, so the agent can
  see at a glance "this source has been processed for dosing but not
  for interactions" — exactly the kind of layered ingestion the
  per-prompt design exists to enable.
- The page's `provenance: [<rel_path>]` and the registry's
  `(source → pages)` link are two halves of the same audit trail. Skip
  `record_ingestion` and only the page-side half survives; `outmem lint`
  doesn't (yet) cross-check the two, so silent drift is on you.

## Two-source ingestion (worth showing once)

Most ingestions are one source → one or two pages. The interesting
shape is **N sources → one page**, when you compact across files:

```python
write_page(
    slug="abx:penicillin:cat-dose",
    title="Penicillin dosing in cats",
    body="...synthesised across both sources...\n",
    provenance=["veterinary/cat-drugs.md", "veterinary/feline-formulary.md"],
    tags=["abx", "veterinary"],
)
record_ingestion("veterinary/cat-drugs.md",      prompt="…", pages_touched=["abx:penicillin:cat-dose"])
record_ingestion("veterinary/feline-formulary.md", prompt="…", pages_touched=["abx:penicillin:cat-dose"])
```

Two `record_ingestion` calls (one per source), one shared
`pages_touched` slug. The page's `provenance` lists both sources; the
registry has two rows pointing at the same page.
