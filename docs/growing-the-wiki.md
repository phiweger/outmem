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

`outmem lint` catches three classes of "the wiki is reaching for
something it doesn't have":

- **Broken wikilinks** (error) — a page wrote `[[bluesky-corp]]` but
  no `bluesky-corp.md` exists. The agent connected a concept the wiki
  doesn't define. Strong signal to ingest.
- **Orphan pages** (warning) — pages with no inbound links. Less of a
  gap, more of "this knowledge is isolated" — usually wants a few
  wikilinks from related pages, not a new source.
- **Stale provenance** (warning) — a page cites a `raw/` file that no
  longer exists. Re-ingest or fix the citation.

```bash
outmem lint
# → exit 0 clean / 1 warnings / 2 errors
```

Treat the broken-wikilink list as a sibling backlog to the log
entries.

## 3. Ask the agent to summarise the gaps

The cleanest signal is just to ask. The agent reads the log itself
during phase 2 (RETRIEVE) and synthesises:

```bash
outmem ask "What are the biggest gaps in our knowledge, based on
recent log entries? Prioritise topics asked about more than once."
```

This triggers `search_wiki(scope="log")` plus `topic_evolution` over
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
