# Growing the wiki

The wiki is supposed to *compound*: every question the agent answers
should leave the wiki slightly better-equipped to answer the next
question. This file describes how to read the signals the agent
leaves behind, so you know what to source / ingest / write next.

The short version: **read the log, then run lint, then ask the
agent**. In that order.

## 1. The log is the agent's TODO backlog

Every `outmem ask` run produces at least one commit (mandatory
writeback). For questions the agent couldn't answer from the wiki, it
appends a `log/` entry — the `write` skill makes this explicit:

> "Search that turned up nothing actionable → **log** (one line)."

So each `log:` commit is effectively the agent saying "I needed X and
had nothing." Those are your highest-priority gaps.

Find them:

```bash
# Every "log:" commit on the wiki, newest first.
git -C $OUTMEM_PATH log --grep '^log:' --oneline

# Or grep the log files directly (they're plain markdown).
outmem search "gap"            --scope log
outmem search "no record"      --scope log -i
outmem search "not aware|don't have|no information" --scope log -i

# Today's log.
outmem read $(date +%Y-%m-%d) --body-only  # if you've aliased read-by-date
cat $OUTMEM_PATH/log/$(date +%Y-%m-%d).md
```

Each entry usually carries the original query + a one-line "no
documented source for X" note. Take a few of them, find sources,
`outmem ingest` them.

## 2. Lint surfaces structural gaps

`outmem lint` catches several classes of "the wiki is reaching for
something it doesn't have":

- **Broken wikilinks** (error) — a page wrote `[[bluesky-corp]]` but
  no `bluesky-corp.md` exists. The agent connected a concept the wiki
  doesn't define. Strong signal to ingest.
- **Orphan pages** (warning) — pages with no inbound links. Less of a
  gap, more of "this knowledge is isolated" — usually wants a few
  wikilinks from related pages, not a new source.
- **Stale provenance** (warning) — a page cites a registered source
  whose file no longer exists. Re-ingest or fix the citation.
- **Unregistered provenance** (warning) — a page cites something the
  registry never held at all. Register the source (`outmem ingest`) or
  remove the citation. New writes are refused outright rather than
  reported here, so this only appears on pages written before 0.18.

```bash
outmem lint
# → exit 0 clean / 1 warnings / 2 errors
```

Treat the broken-wikilink list as a sibling backlog to the log
entries.

## 2b. Completeness — pages that stop early

Every check above is structural: it asks whether the wiki's *metadata*
is consistent. One check reads the body as content, because a page can
be a fraction of what its own source says while every structural
invariant around it is perfect.

The failure is quiet and specific. When a turn's output budget binds,
a model does not necessarily fail — it can emit a schema-valid
`write_page` whose body stops early and marks the cut with an ellipsis.
Provenance, hashes, links, and the index all stay correct, so nothing
downstream can tell that page from a finished one. One wiki carried 181
such cuts across 75 pages for months; the missing content was in
registered sources the whole time.

Three lint kinds cover it:

- **`truncated-page`** (warning) — the body ends at an elision marker.
  Reported with a file line number and the offending line quoted. Fix
  by restoring the content from the page's sources and appending it.
- **`tool-output-in-page`** (warning) — an `⟪ outmem: … ⟫` marker was
  copied into a page. That marker means outmem itself withheld content
  from a tool result (an oversized source, a spliced search excerpt),
  so the page is built on material it never actually saw.
- **`declared-omission`** (warning) — the page's frontmatter carries an
  `omitted:` note. Deliberate scoping is fine; it is reported so the
  gap stays visible rather than becoming a way to make a short page
  look finished.

New cuts are prevented rather than reported: `write_page`,
`extend_page`, and `append_page` refuse a body that ends at an elision
marker, and the agent is told to use `append_page` for the rest instead
of shortening. A quotation that continues past its ellipsis
(`"die Therapie [...] wird empfohlen"`) is not affected — the check
keys on a marker that *ends* its line, and skips blockquotes, code, and
link text.

**The refusal a model can insist past, and the one it cannot.** Because
the elision check is a positional heuristic, a model that re-sends the
*identical* body after being refused is taken at its word — the ellipsis
was quoted text, the page lands, and `outmem lint` reports it as
`truncated-page` so the bargain is bounded damage rather than silence. A
human has the one-step override instead (`--allow-elision`,
`allow_elision=True`), which the tools deliberately do not offer, because
a flag a model can set is a flag it learns to tick. The `⟪ outmem: … ⟫`
marker gets no such yield: outmem wrote that marker itself, so the page
is provably built on content the model was not shown, and re-sending
unchanged is refused again.

That yield is right inside outmem's own agent loop, where a false
positive would otherwise cost the whole turn against a bounded retry
budget. Over a connector it is a sentence a host model follows
reflexively, and the elided page lands on the second try. For a wiki
served that way, withdraw it:

```yaml
# config.yaml
completeness:
  elision_yield: false
```

or, from the serving code, `store.elision_yield = False` after opening.
The refusal then tells the model to rephrase so the marker does not end
its line, an identical resubmission is refused again, and only an
operator can pass the body as written (`--allow-elision`).

If a turn ends because it ran out of output room while writing a page,
`outmem ask` warns and names the page even when nothing was marked —
that is the case the model did not flag itself. Verify those pages
against their sources.

## 3. Ask the agent to summarise the gaps

The cleanest signal is just to ask. The agent reads the log itself
during phase 2 (RETRIEVE) and synthesises:

```bash
outmem ask "What are the biggest gaps in our knowledge, based on
recent log entries? Prioritise topics asked about more than once."
```

This triggers `grep_wiki(scope="log")` plus `topic_evolution` over
the relevant slugs and gives you a prioritised list. The agent itself
produces another `log:` entry recording the meta-question, which
shows up next time you ask the same.

## 4. Topic-level shape — what's moving?

For active areas, `outmem evolution <slug>` shows the `git log -p`
diff stream for one or more pages. Heavily-`extend:`ed pages are
where current attention is — usually a sign that more sources would
help. Untouched-for-months pages are settled.

```bash
outmem evolution acme-pricing
outmem evolution acme-pricing acme-msa     # interleaved across slugs
```

## A weekly review rhythm

For a wiki you're actively maintaining, this loop works:

```bash
# 1. What did I (and the agent) do this week?
outmem steering                                  # human commits since last record-run
git -C $OUTMEM_PATH log --since='7 days ago' --oneline

# 2. What did the agent flag as missing?
outmem search "gap" --scope log

# 3. Structural health.
outmem lint

# 4. Ingest a few sources to close the top gaps.
outmem ingest /path/to/source-1.md --into <topic> --prompt "..."
outmem ingest /path/to/source-2.md --into <topic> --prompt "..."

# 5. Mark the review done so next week's steering signal is bounded.
outmem record-run
```

Twenty minutes a week keeps a healthy wiki growing. Skipping it lets
the gap list build up but nothing breaks — the agent will keep
logging until you act.

## Anti-patterns

- **Writing stub pages by hand.** If you find yourself creating
  `bluesky-corp.md` with body "we have no information yet", stop:
  ingest a real source instead. Empty pages clutter the index and
  give the agent false hits on `search_wiki`.
- **Ignoring orphans long-term.** A page no one cites tends to drift
  out of date because the agent never has a reason to extend it. Add
  the wikilinks or merge into a parent topic.
- **Reading the index instead of the log.** `wiki/index.md` is the
  catalog of what you *have*; the log is the catalog of what you
  *don't*. The log is more useful for "what should I add?".
