# Agentic RAG — Technical Specification

**Status:** draft v0.10 (implementation in progress; v0.10 supersedes v0.9)

> v0.10 changes:
> - **SQLite source registry.** ``wiki/sources/`` carries a
>   ``.sources.db`` (SQLite, two tables — ``sources`` + ``ingestions``).
>   Concurrent ``outmem ingest`` runs against the same wiki are
>   parallel-safe: writers serialise on the DB's busy-timeout instead
>   of racing on a JSON read-modify-write. Rollback-journal mode means
>   no ``-wal`` / ``-shm`` companion files. See §7.

> v0.9 changes:
> - **Content-addressed source layout.** ``wiki/sources/`` files live
>   at ``[<into>/]<sha256[:12]>/<filename>``. The hash directory
>   makes the layout collision-free (two ``document.md`` files with
>   different content land in different dirs) and dedupes
>   identical-content re-ingests automatically. See §7.

> v0.8 changes:
> - **Opt-in HITL approval for writes.** ``approval.required_for_writes:
>   true`` in ``config.yaml`` defers the agent's ``write_page`` /
>   ``extend_page`` tool calls through PydanticAI's native deferred-tool
>   primitives. A :class:`Reviewer` (CLI by default) returns one of
>   approve / approve-with-edited-args / deny per call; the commit
>   only lands on approve. ``append_log`` stays free so mandatory
>   writeback (§9) is still satisfiable after a denial. Non-interactive
>   contexts must explicitly disable the flag or wire a custom
>   Reviewer — ``outmem ask`` fails loudly rather than silently
>   committing. See §9.
>
> v0.7 changes:
> - **Opt-in semantic index (Tier 3 retrieval, issue #7).** A local
>   ``sqlite-vec`` DB at ``<wiki>/.vectors.db`` (tracked in git) holds
>   paragraph-aware chunks of wiki pages + sources. Updated atomically
>   on every write; a pre-commit hook (``outmem hook install``) keeps
>   it in sync with external edits made through Obsidian. Adds
>   ``find_similar`` to the agent's tool palette and an ``outmem
>   similar`` CLI command. Embeddings via PydanticAI's ``Embedder``,
>   default ``openai:text-embedding-3-small``. See §8 "Tier 3".
>
> v0.6 changes:
> - **Ingestion is now in scope.** Sources live in ``wiki/sources/``
>   (tracked in git, flat-text only); the agent reads them via
>   ``read_source`` and produces wiki pages with provenance pointing
>   at ``sources/<rel-path>`` + sha256. ``outmem ingest <file>
>   [--prompt P]`` is the entry point. See §7.
> - **Auto-maintained ``wiki/index.md``** — every page write
>   regenerates it in the same commit. No manual command.
> - **``outmem lint``** for static checks (orphans, broken
>   wikilinks, stale provenance, index drift). Semantic
>   near-duplicate / contradiction detection lives behind
>   ``find_similar`` in v0.7 (see §8 Tier 3); a structured
>   ``outmem lint --semantic`` command may follow.
> - **``config.yaml`` + ``.env``** carry config; ``.env`` is loaded
>   from CWD walking upward (project root), ``config.yaml`` from the
>   wiki root.

---

**Status:** draft v0.5
**Companion documents:** `concept.md`, `planning_prompt.md`
**Scope:** memory formation and retrieval over a directory of plain text/markdown files versioned in git, with async multi-author editing through Obsidian. Single-server deployment.

**Changes from v0.4:** authorship-as-permission is removed. The `authority` frontmatter field, the blame-aware compaction rule, and the human-driven `agent → mixed` promotion all go. "Who wrote what" is left to git history; the agent freely edits any wiki page, and humans correct via normal commits. Identity discipline is reframed as steering-signal quality (degraded mode if misconfigured) rather than a write-side correctness invariant. The temporal-evolution divergence primitive becomes a documented shell pattern (`git log -p --follow`) rather than a dedicated tool. Backlinks move from per-render `rg` to a precomputed cache. Dashboard authorship indicator deferred to v0.2. Mixed-authority and elaborate rebase-on-writeback tests retire.

---

## 1. System overview

A Hetzner server hosts two processes: a FastAPI dashboard (auth, read-only wiki view) and the agent runtime (a separate long-running process). They share a single canonical store: a directory of markdown files versioned in git. Humans edit by cloning the repo locally and using Obsidian; the dashboard renders for reading only. Plain markdown/text source material appears in `raw/` via an external ingestion pipeline (out of scope here, see §7). The agent compiles raw material into the wiki, writes back its findings, and uses git history as both audit trail and steering signal.

The architecture is a deliberate inversion of conventional RAG: no global vector index over raw sources, agentic shell-tool retrieval as the default, and mandatory writeback so identical future queries don't re-pay retrieval cost. The conceptual rationale lives in `concept.md`.

---

## 2. Storage layout

The canonical store is a single directory on the server, conventionally `/srv/agent/`, containing three subdirectories.

`raw/` holds plain markdown or text source material populated by the upstream ingestion pipeline (§7). The agent reads from `raw/` but never writes to it. `raw/` is **not** tracked in git: it is treated as a working cache populated and curated externally, with its own version history living in whatever upstream system the ingestion pipeline draws from.

`wiki/` holds the compiled knowledge — plain markdown files, one concept or topic per file, interlinked with Obsidian-style `[[wikilinks]]`. Every file has YAML frontmatter (§4). `wiki/` **is** tracked in git. The directory is small — hundreds to low thousands of files — and diffs cleanly. Scale ceiling for this design: roughly 10k pages or 100MB of compiled markdown, beyond which ripgrep latency starts to bite (§12).

`log/` holds the decision and exploration trail — dated markdown files (`log/2026-05-04.md`) with append-only entries. Both agent and humans write here. `log/` **is** tracked in git. Its history *is* the long-term decision audit, and the temporal-evolution primitive (§8) reads `git log -p` against it to reconstruct how framings shifted over time.

---

## 3. Git as the synchronisation and audit substrate

Everything that touches `wiki/` or `log/` does so through git commits. There is no other write path. Obsidian Sync, Dropbox, iCloud, Syncthing on the wiki folder, and similar non-commit sync mechanisms are explicitly disqualified — they produce silent file changes with no authorship attribution, breaking the agent's steering loop, the audit trail, and the v0.2 upgrade path to write-time authority enforcement simultaneously.

The repo lives on a private git remote. Self-hosted Gitea on the same Hetzner box keeps everything on-prem; a private GitHub repo is acceptable for personal use. The `main` branch is the single line of development.

**Identity discipline matters for steering, not for write authority.** Each human commits under their own configured `user.name` and `user.email`, set in their local clone (which is what Obsidian's Git plugin will use). The agent commits under a fixed identity — `agent@<hostname>` — set explicitly in the agent's commit wrapper via `git -c user.name=agent -c user.email=agent@host commit`, never relying on global config. A `CONTRIBUTORS.md` at the repo root lists known team identities; the agent reads it on startup to recognise authors in the steering loop (§8, planning prompt phase 1). Misconfigured identities — Bob committing as his system default, two humans collapsing onto the same email — do not corrupt the wiki, but they do degrade the steering signal by merging or misattributing channels of intent. The runtime warns when `git log` surfaces an author not in `CONTRIBUTORS.md`; it does not refuse to run.

Two git queries form the agent's authorship API:

`git log --since=<last-run-timestamp> -- wiki/ log/` filtered to non-agent authors and grouped by author becomes the human-steering signal: each person's recent edits enter the agent's planning context as a separate channel, so divergent framings surface as contradictions rather than silently averaged. Read at the start of every agent run.

`git log --author=agent --since=<window>` returns the agent's own writeback rate — the implementation of the *Retained* TARS metric (§10). Run from the shell when needed; no dedicated dashboard route in v0.1.

`git blame` is not part of the write-side contract in v0.1. It remains available as a read-side tool — humans inspect history through it directly, and the dashboard may surface a blame-coloured authorship view in v0.2 (§12) — but no write decision the agent makes depends on its output.

Conflict handling is standard git: humans pull-then-edit-then-commit-then-push from their Obsidian clones, the agent runs `git pull --rebase` before its own writeback and re-reads the affected file if HEAD has moved before proceeding with compaction. When a human and the agent commit incompatibly to the same paragraph, the second to push hits a conflict and resolves through normal git mechanics — Obsidian surfaces conflict markers in the file, the human resolves, commits, pushes. The agent's writeback wrapper retries on push rejection by pulling and re-reading; if the rebase still fails or the push is rejected a second time, the agent treats this as a hard error, logs the failure, and surfaces it to the user rather than responding as if writeback had succeeded.

---

## 4. The wiki page model

Each `wiki/*.md` file begins with YAML frontmatter:

```yaml
---
title: Pricing formula
slug: pricing-formula
provenance:                  # source pointers into raw/
  - raw/pricing-deck-2026-Q1.md
  - raw/acme-msa.md
created: 2026-04-12T09:14:00Z
updated: 2026-05-04T11:32:00Z
tags: [pricing, contracts, finance]
---
```

There is no `authority` field. Any human or the agent may edit any wiki page. The system's defence against the agent making a mess is that humans see its commits in `git log` and correct them via normal edits — the next agent run reads those edits as part of phase 1's steering loop (§8) and adjusts. This trades a write-time guarantee (the agent provably cannot touch human lines) for a review-time one (humans can see and reverse anything the agent did), which is the right shape at v0.1 scale where history is visible and the team is small. If a future deployment needs write-time enforcement — adversarial settings, multi-tenant, or just a wiki large enough that humans miss bad edits — a page-level or section-level authority tag can be reintroduced as an additive feature (§12).

The `provenance` field is a list of pointers into `raw/`. The agent populates it during compaction and propagates any deeper provenance — Drive paths, page ranges, content hashes — that the ingestion pipeline embedded in the raw file's own frontmatter. The agent does not generate or interpret upstream provenance; it just preserves it.

Wikilinks use `[[slug]]` syntax, resolved natively by Obsidian and rendered to `/wiki/<slug>` in the dashboard's read view (§5).

---

## 5. Reading and editing surfaces

**Editing happens in Obsidian, on a local git clone. Period.**

Each user clones the repo to their laptop, opens the directory as an Obsidian vault, and installs the Obsidian Git community plugin [10] for one-keystroke commits and auto-pull. On iOS, Working Copy [11] handles git and Obsidian edits the files. On Android, the official Obsidian Git plugin works directly. Break-glass for emergency edits from anywhere is github.dev [12] or SSH plus any text editor.

All editing paths produce identical commits attributable to the configured user. The dashboard provides no editing surface — this is a deliberate constraint, not a missing feature. A second editing path would create a second commit-author resolution, a second conflict-handling story, and a second autosave-vs-explicit-save decision; the Obsidian-only path keeps git's authorship discipline simple.

**The dashboard provides a read view.**

A FastAPI route at `/wiki/{path:path}` reads the markdown file from disk, renders it to HTML via `markdown-it-py` [9] with a small wikilink-resolution pass that turns `[[slug]]` into anchor tags pointing at `/wiki/<slug>`, and serves it inside a thin template. Read-only. Gated behind the dashboard's existing auth.

The route's clone of `wiki/` is kept fresh either by `git pull` on every request (simplest, fine for a small team), by a webhook from the git remote (lower latency, requires remote configuration), or by a pull cron at fixed interval (compromise). Choose based on traffic; default is pull-on-request.

Two read-side features earn their keep in v0.1:

The **backlinks panel** at the bottom of each rendered page lists pages that wikilink to this one. The backlink index is precomputed — refreshed whenever the dashboard's clone moves to a new HEAD (post-pull, or on webhook) by running `rg "\[\[<slug>\]\]" wiki/` once per slug and caching the result on disk or in memory. Render reads the cache. At v0.1 scale a full rebuild is sub-second; an incremental update on the changed files is straightforward if it ever becomes necessary.

The **commit history view** for each page exposes `git log --follow -- wiki/<page>.md` as a vertical timeline — author, date, commit message. This is the user-facing surface of the temporal-evolution pattern: humans get the same affordance the agent does, and "why does this page say X now when it said Y last week" is answerable in one click.

The **authorship indicator** — a blame-coloured overlay showing which lines were written by which author — is deferred to v0.2 (§12). It is the natural read-side surface for "who wrote what" now that the write-side authority rule is gone, but it adds dashboard surface area and visual design choices that v0.1 can do without.

What the dashboard does not provide in v0.1: graph view, in-browser advanced search (ripgrep over the directory exposed as a `/search?q=` route beats most full-text UIs at this scale), draft mode (commits-as-versions handles this), authorship indicator (deferred to v0.2), or any editing affordance.

---

## 6. Process layout

**Two processes**, not one.

The **dashboard** is a FastAPI app handling auth and the read view. Stateless apart from session storage; restartable without affecting the agent. Has its own clone of the repo at a known path (e.g., `/srv/dashboard/wiki-clone/`), kept fresh by the strategy chosen in §5.

The **agent runtime** is a separate long-running Python process. Its lifecycle is independent of the dashboard — different scaling characteristics, different restart cadence, different failure modes. The dashboard does not call agent functions directly. Communication is via a small internal API: the agent exposes a localhost HTTP endpoint that accepts queries and returns results, or alternatively a queue (Redis, SQLite-backed, even a flat-file inbox — anything durable). v0.1 default is a localhost HTTP endpoint because it is the lowest-friction option; switch to a queue if and when the agent becomes a bottleneck.

The agent process owns its own working directory containing its own clone of the repo. Both processes pull from the remote on each operation that needs current state, and push their commits to the remote when they write. They coordinate purely through git.

The ingestion pipeline (§7) runs as its own process or processes, entirely outside this system's lifecycle.

---

## 7. The ingestion contract

This system does not ingest. Population of `raw/` is the responsibility of an upstream pipeline specified separately. This section defines the contract between that pipeline and this system, which is the only thing this system depends on.

**Inputs to `raw/`.** Plain UTF-8 text files, preferably with `.md` extension. The agent reads them with shell tools (`rg`, `cat`, `head`); anything that's valid input to those tools is valid input to the system. PDFs, .docx, .xlsx, audio, images, and other non-text formats are *not* valid inputs; the upstream pipeline is responsible for converting them to text or markdown before they arrive in `raw/`.

**Optional frontmatter.** A raw file *may* begin with a YAML frontmatter block recording its provenance — Drive path, content hash, source URL, page range, focus instructions, ingestion timestamp, ingestor identity, anything the upstream pipeline wants to record. The agent preserves this metadata in the `provenance:` field of any wiki page it compiles from the raw file (§4), but does not generate or interpret it. If frontmatter is absent, the agent uses the file's path as its only provenance pointer.

**File lifecycle.** Files appear in `raw/` when the upstream pipeline puts them there. They are read freely by the agent. They may be edited or removed by the upstream pipeline at any time; the agent does not depend on raw files being immutable. If a wiki page cites a raw file that subsequently disappears, the citation remains valid as a historical record — a `log/` entry should note the disappearance, but no automatic cleanup runs.

**Out of scope here.** How files get into `raw/`, what they're parsed from, how focus extraction works, how the user reviews and confirms candidates before they become inputs, how Drive paths or content hashes are tracked, how supersession or contradictions between source documents are surfaced — all of these are ingestion concerns, specified in the (separate) ingestion document.

**Promotion from `raw/` to `wiki/sources/`.** When the agent compiles a wiki page from a specific raw source via `outmem ingest`, that source gets copied into the wiki at a content-addressed path: `wiki/sources/[<into>/]<sha256[:12]>/<filename>`. The hash directory is what gives the layout three useful properties simultaneously: (a) no collisions when two different files share a basename (the user's `parsed/fachinfo/<drug>/output/<hash>/document.md` pipeline produces many such collisions), (b) re-ingesting the same content is a no-op rather than an error (same hash → same path), and (c) the registry rows (one per `rel_path`) are self-validating against the recorded sha256.

**Registry storage (v0.10).** The source registry lives in `wiki/sources/.sources.db` — a SQLite file with two tables (`sources`, `ingestions`); the ingestion FK is `ON DELETE CASCADE`. SQLite (busy-timeout 5 s, default rollback-journal) is parallel-write-safe at the file-system level, which a JSON registry couldn't be without an application-level lock — concurrent `outmem ingest` runs against the same wiki are legal. Rollback-journal mode (not WAL) keeps the main DB file always in sync with committed state, so it's a normal git-trackable binary with no `-wal` / `-shm` companions.

---

## 8. Retrieval — the agent's tool palette

**v0.1 ships two tiers.** A semantic sidecar is deliberately deferred to v0.2 (§12).

**Tier 1: shell tools over `wiki/`.** `ripgrep` for content search, `glob` for filename patterns, `ls` for directory enumeration, `cat`/`head`/`tail` for read. Wrapped as agent tools, direct shell calls under the hood. Default first move on any retrieval need: ripgrep over `wiki/`. Compiled material answers most queries faster and with higher fidelity than raw chunks ever could.

**Tier 2: same shell tools over `raw/`.** Falls back here when `wiki/` doesn't have the answer. Since `raw/` is plain text by contract (§7), no special tooling — same `rg`, same `cat`. If a future ingestion pipeline wants to enable structured queries (e.g., querying across CSV-derived markdown tables), that's an additional tool the agent can pick up, but the v0.1 default is text-only.

**One divergence pattern in v0.1: temporal evolution.** Not a separate tool — when phase 1 selects EXPANSION (planning prompt §C), the agent invokes the existing shell tool palette with `git log -p --follow -- <paths>` against the wiki pages and `log/` entries it judges relevant, and reads the chronological diff stream. Nearly free given git-as-substrate; the agent already has shell access, and the orchestrator's planning prompt teaches when to reach for the pattern. Promoting this to a dedicated wrapper is straightforward in v0.2 if session logs show the agent consistently mis-scoping the query (forgetting `--follow`, missing `log/`, etc.) — but a wrapper that adds nothing beyond what the agent can already do should not exist in v0.1. The other four primitives from `concept.md` (contradiction surfacer, negative-space query, associative drift, cross-domain bridges) are deferred to v0.2 (§12).

**Tier 3 (v0.2, opt-in): semantic index.** When the user installs `outmem[semantic]` and sets `semantic.enabled: true`, outmem maintains a paragraph-aware `sqlite-vec` index at `<wiki>/.vectors.db` over every wiki page and every text source under `wiki/sources/`. Embeddings come from PydanticAI's `Embedder` (default `openai:text-embedding-3-small`). The index is updated atomically with every write: agent `write_page` / `extend_page` / `add_source` re-chunk the affected file, re-embed only what changed (content-hash skip), and stage the DB in the same commit. For human edits made outside the agent (Obsidian, manual `git add`), `outmem hook install` drops a pre-commit hook that runs `outmem reindex --staged` so the DB stays in lockstep with the commit history. The agent gains one tool — `find_similar(text, top_k, exclude_slug)` — added to its palette automatically when semantic is on; the system prompt nudges it to check for paraphrased duplicates before writing a new page. Switching embedding models is a flagged operation: the dim baked into `chunks_vec` won't match the new model, and `VectorStore.open` raises a clear error pointing at `outmem reindex --force`. The remaining four divergence primitives are still deferred (§12).

The orchestrator's planning prompt (`planning_prompt.md`) is required to decide, before any retrieval, whether the query needs convergence (precision: pull the relevant fact) or expansion (map the possibility space, currently only via temporal evolution). Even with one divergence pattern the convergence/divergence decision still gets made — it just collapses to a binary about whether to walk git history before converging.

---

## 9. Writeback and compaction

Writeback is **mandatory**. At the end of every agent run, before responding, the agent must produce at least one git commit. Without this, agentic search is just an expensive way to re-derive the same answer on every query.

Two output paths in v0.1:

1. **Wiki write.** Either a new `wiki/<slug>.md` if the run produced understanding of an uncovered topic, or an edit to an existing page. The agent reads the file, decides on the edit, and commits — there is no blame-based gate, no authority lookup, no line-level permission check. If the agent edits something a human disagrees with, the human reverts or rewrites in their next commit, and the agent reads that commit as part of phase 1's steering signal on the following run. New pages include frontmatter and provenance (§4); existing pages preserve their frontmatter except for `updated:`.

2. **Log entry.** Append to `log/<today>.md` for findings that don't yet rise to a wiki page — observations, contradictions, questions raised but unanswered, decisions, or "no new compaction needed" notes. Timestamped and tagged with session ID.

Either path produces a commit. Promotion from log to wiki is deferred to v0.2 — for v0.1, humans manually elevate log entries to wiki pages when they decide it is worth doing.

**Optional HITL approval (v0.2, opt-in).** When `approval.required_for_writes: true`, the agent's `write_page` and `extend_page` calls go through PydanticAI's deferred-tool mechanism: the agent run yields a `DeferredToolRequests` instead of committing, the orchestrator consults a `Reviewer` (CLI by default; pluggable for web/dashboard surfaces), and resumes the run with a `DeferredToolResults` carrying one of three verdicts per call: **approve** (tool runs as proposed), **approve-with-override-args** (tool runs with reviewer-edited args — typically the body — so "comment that changes the edit" doesn't need a re-prompt round-trip), or **deny** (a `ToolDenied` message flows back to the model, which typically responds with `append_log` to still satisfy the writeback contract above). The gate is intentionally narrow: `append_log`, `add_source`, and all read-only tools are not deferred, so the agent has an unblocked path to mandatory writeback after a denial. Non-interactive contexts (CI, batch) must either disable the flag or wire a custom `Reviewer` — the CLI aborts before the agent run starts rather than silently autocommitting.

Before pushing, the agent runs `git pull --rebase`. If the rebase succeeds, it pushes. If push is rejected (another commit landed between pull and push) the agent retries the pull-rebase-push loop once. If the second push is still rejected, or if the rebase produces a conflict that cannot be resolved automatically, the agent treats writeback as failed: it logs the failure and surfaces a hard error to the caller. It does not respond to the user as if the commit succeeded, and it does not silently drop the writeback. A failed writeback is a system fault that needs attention, not a recoverable runtime condition to paper over.

Commit messages follow a structured convention so they're parseable: `compact: <slug>` for new wiki pages, `extend: <slug>` for updates, `log: <topic>` for log entries. This makes TARS *Retained* (§10) trivially computable as a `git log` filter.

---

## 10. Evaluation

Per Raudaschl [2] and the concept doc, the system tracks four metrics — but **only for features actually shipped in v0.1**: shell retrieval over `wiki/` (Tier 1), shell retrieval over `raw/` (Tier 2), and temporal evolution. Adding a feature without measuring it is the failure mode this evaluation is meant to prevent.

- **Target.** Fraction of agent sessions that invoke this feature. From agent run logs.
- **Adopted.** Of sessions where the feature *should* have been invoked (judged by replay against a held-out query set), fraction that *did*. Bootstrap a small held-out set during v0.1 — 30–50 hand-built queries with known good retrieval paths.
- **Retained.** Did invocation produce a commit? `git log --author=agent --grep=<feature-tag>` per session. Sessions that invoke a feature without producing writeback signal that compaction is broken for that feature.
- **Satisfied.** Did the human accept / act on / not contradict the result? Inferred from subsequent sessions — if the human rewrites the same page within 24h, the agent's output was probably wrong; if they cite it in a later session, probably right.

Three operational metrics from SmartScope [5]:

- **Time-to-first-relevant-fact.** Wall-clock from query received to first useful retrieval result.
- **Tokens-per-task.** Total tokens consumed per resolved query. Watch upward drift.
- **Staleness incidents.** Cases where the agent surfaced a wiki claim contradicted by a more recent `raw/` source.

Recall@k and nDCG do not appear. Similarity and usefulness are not the same thing [1, 3].

---

## 11. Testing

Distinct from evaluation: tests catch regressions when code changes, evaluation measures whether the system is useful.

A frozen **test corpus** lives in `tests/fixtures/` — a `raw/` of ~20 plain-text files and a hand-built target `wiki/`. CI runs the agent against this corpus from a clean state and verifies invariants, not exact outputs. Output equality tests against agent-generated content will be brittle and constantly false-positive.

**Property tests** to ship with v0.1:

Every wiki page has valid frontmatter (parseable YAML, required fields present). Every wiki page's `provenance:` entries point to files that exist in `raw/` at compile time (the test fixture freezes both, so this is checkable). No orphan wikilinks except those explicitly tracked in `log/`. Commit-message convention: every commit on the test branch matches one of `compact:`, `extend:`, or `log:` so the TARS *Retained* query (§10) does not silently drop results.

**Replay harness** for the steering loop. Capture real `git log` outputs from prior sessions, replay as steering input, snapshot the agent's planning context, assert that user-A's edits and user-B's edits arrive as separately tagged channels rather than a merged blob. This bug class is invisible until it bites and very hard to diagnose post-hoc.

**Writeback push-retry test.** A small integration test that, between the agent's pull and its push, inserts a conflicting commit on the remote, then asserts that the agent (a) detects the rejected push, (b) pulls again, (c) either succeeds with a new commit or surfaces a hard error per §9 — and in particular never silently drops the writeback. This is much less elaborate than the v0.4 "rebase fixture harness" because the property being tested is just "the retry loop is wired correctly": with blame-aware authority gone, there is no longer a class of "agent silently overwrites human content" failures hiding behind the test.

Tests *not* worth writing at v0.1: anything asserting specific retrieval results ("query X returns page Y"), anything asserting the agent's natural language output, anything testing the upstream ingestion pipeline (out of scope per §7), anything testing an in-browser save flow (no save flow exists), anything asserting write-time authority enforcement (no authority field exists). The first two belong in evaluation; the third belongs in the ingestion document's own tests; the last two are moot.

---

## 12. Out of scope for v0.1 — explicit deferrals

Each item below is something v0.1 does *not* ship. They are listed so future contributors can find the rationale rather than rediscover it.

**Ingestion.** Entirely out of scope per §7. Spec'd separately.

**In-browser editing.** Editing is Obsidian-only by design (§5). If an in-browser editor ever becomes desirable — for users who don't want to install Obsidian, for casual edits from any machine — the right approach is to add it as an optional surface that uses the same git-commit-with-user-identity flow. github.dev [12] already covers this case for users with GitHub access; a self-hosted CodeMirror-on-FastAPI editor could be added later if there's demand. Adding it in v0.1 doubles the edit-path surface area for negligible benefit.

**Semantic sidecar (Tier 3 retrieval).** Shipped as opt-in in v0.2 — see §8 "Tier 3 (v0.2, opt-in)". The v0.1 reasoning below is preserved for context: a FAISS [13] index over `wiki/` page embeddings, rebuilt on commit, with a local embedding model. Concept doc argues this is necessary; reality is it is necessary *at scale*, not at the start. Ship without it, measure how often Tier 1+2 fails to find what the agent needs, and add Tier 3 only if data justifies. The v0.2 implementation differs on two choices: (a) `sqlite-vec` rather than FAISS — a single git-tracked DB file is simpler than rebuilding a binary index, scales to ~100k chunks, and survives rebases cleanly; (b) provider-hosted embeddings via PydanticAI rather than a local model — easier to install, and the index is opt-in so users who object to sending content out of the box leave it off. Switching to a local model is a one-line config change (`embedding_model: local:bge-small-en`) when PydanticAI gains the provider.

**Four of five divergence primitives.** Contradiction surfacer (requires stance classification with calibrated confidence — a real engineering project), negative-space query, associative drift, cross-domain bridges. All deferred until temporal evolution proves the divergence layer carries weight in real use.

**Subagent fan-out for parallel exploration.** An answer to a token-budget problem v0.1 does not have. Single-agent serial exploration is fine for a wiki of this size.

**Cron-driven log-to-wiki promotion.** Humans manually elevate log entries to wiki pages in v0.1. Automatic promotion needs heuristics that should be designed against real logs, not imagined ones.

**Source supersession / contradiction handling across raw files.** When a new raw file replaces or contradicts an older one, who decides which to trust and how the wiki gets updated? Genuinely hard, partly an ingestion concern (the ingestion layer might mark supersession via frontmatter) and partly a memory concern (the contradiction-surfacer divergence primitive is the right tool, but it's deferred). For v0.1, the agent treats all `raw/` files as equally valid and surfaces contradictions in the answer when they're noticed during retrieval.

**Write-time authority enforcement.** v0.1 has none: any author can edit any page, and "who wrote what" is reconstructed from git history rather than enforced at write time. If a future deployment needs write-time gating — adversarial settings, multi-tenant, or wikis large enough that humans miss bad edits — the natural reintroduction is a page-level `authority:` field with `agent | human | mixed` values, plus a pre-commit hook that promotes pages on first human edit. The v0.4 design specified exactly this; it is intentionally on the shelf until use justifies it.

**Dashboard authorship indicator.** A blame-coloured overlay on the read view showing which lines belong to which author. The natural read-side surface for "who wrote what" once write-side authority is gone. Deferred from v0.1 only because of dashboard surface area and visual design choices, not because the data isn't there — `git blame` answers it directly.

**`agent_compactions` frontmatter field, `/audit` FastAPI route, GPG-signed agent commits.** All useful at scale or in adversarial settings. None earn their complexity at v0.1.

**Real-time collaborative editing.** Async git-based multi-author through Obsidian covers ~95% of actual team patterns. If real-time becomes necessary later, the right path is one of the Obsidian relay-sync community plugins (LiveSync and similar) which work peer-to-peer over Obsidian's vault — not a CRDT-in-the-browser approach, because there is no browser editor to put a CRDT in. Adding live sync between Obsidian instances is an Obsidian-side configuration concern with no impact on this system as long as commits still flow through git.

**Multi-tenant deployments.** Single-team, single-server. Multi-tenant isolation requires per-tenant repos, per-tenant agent runtimes, and a permissions story not yet written.

**Public Quartz-style published view.** The dashboard `/wiki/` route serves authenticated users only. Quartz [14] is the natural choice for a public subset, but selection / redaction / build / deploy is a separate project.

**Scale beyond ~10k wiki pages.** This architecture works well up to roughly 10k pages or 100MB compiled markdown. Beyond that, ripgrep latency degrades and `wiki/` needs a proper search index — not embeddings, but a lexical index like Tantivy [15] or Meilisearch [16]. The upgrade path is additive: keep the shell tools, layer the search index underneath them so `wiki_search` becomes a tool the agent picks when ripgrep would be too slow.

**Multiple concurrent agents.** "The agent" is singular throughout. If multiple agents (research, maintenance, triage) ever run concurrently against the same wiki, writeback contention needs an explicit story — probably per-agent branches with auto-merge, or a write-coordinator process.

---

## 13. Failure modes the design defends against

The relevance trap is addressed by mandatory writeback and the temporal-evolution primitive in v0.1, with the rest of the divergence layer queued for v0.2 once it has earned its place [1, 3].

Index staleness for the wiki is moot in v0.1 because there is no index. v0.2's opt-in semantic index defends against staleness by atomicity rather than rebuild: every agent write goes through `_commit_paths`, which re-chunks the affected file, re-embeds only what changed (content-hash skip), and stages the `.vectors.db` in the same commit. External edits made through Obsidian or raw `git add` are caught by the optional pre-commit hook (`outmem hook install`), which runs `outmem reindex --staged` before the commit lands. The hook is local — it fires on the human's working clone the moment they commit, no remote webhook required. Drift between disk and DB can still be repaired explicitly with `outmem reindex`. Staleness in `raw/` is the upstream pipeline's responsibility per the §7 contract.

Token blow-up on large corpora is mitigated by tiered tool cost (cheap shell tools first), writeback collapsing repeat queries toward zero marginal cost, and the explicit scale ceiling (§12) that says "stop and add a search index" before token usage degrades silently. Large source documents that would individually blow up the context window are an ingestion problem (§7).

Context rot from over-retrieval is addressed by the compaction-first design — the agent prefers reading a compiled wiki page over raw chunks.

Hallucinated tool calls are addressed by a small, well-documented tool palette (under ten tools in v0.1) with full observability — every call and every result logged.

Silent overwrites between human and agent are addressed at v0.1 by visibility rather than prevention: every commit is authored under a stable identity (§3), every edit is visible in `git log`, and the agent reads recent human commits as part of phase 1's steering loop before acting. If the agent writes something a human disagrees with, the human reverts or rewrites in a normal commit and the agent picks that up on the next turn. Non-git sync mechanisms that bypass commit-level visibility are disqualified by design — that is what keeps "silent" out of the failure mode. Because there is no in-browser editor and no `authority` field to maintain, there is no save-flow surface where authorship could be accidentally collapsed and no metadata path that can rot out of step with reality. Write-time enforcement remains available as an additive v0.2 feature (§12) if real use shows humans missing bad edits.

PII and security exposure for memory operations is addressed by keeping all data on the Hetzner box. v0.2's semantic sidecar is opt-in — users for whom embedding-provider egress is unacceptable simply leave `semantic.enabled: false`, and the system falls back to Tier 1+2. When the index is on, a local embedding model is a one-line config swap (the implementation goes through PydanticAI's `Embedder` interface, which already supports local providers). PII handling at the ingestion boundary is the upstream pipeline's concern.

---

## References

[1] Raudaschl, A. *The Relevance Trap.* Breaking Product. <https://breakingproduct.substack.com/p/the-relevance-trap>

[2] Raudaschl, A. *TARS: A Product Metric Game Changer.* UX Collective. <https://uxdesign.cc/tars-a-product-metric-game-changer-c523f260306a>

[3] Fleck, J. *Divergence Engines: Escaping the Relevance Trap.* Medium, October 2025. <https://medium.com/@j0lian/divergence-engines-escaping-the-relevance-trap-1bbdbee55ea6>

[4] Nicolai, V. *Claude Code Doesn't Index Your Codebase. Here's What It Does Instead.* Vadim's Blog. <https://vadim.blog/claude-code-no-indexing>

[5] SmartScope. *Settling the RAG Debate: Why Claude Code Dropped Vector DB-Based RAG and the Reality of Code Search.* <https://smartscope.blog/en/ai-development/practices/rag-debate-agentic-search-code-exploration/>

[6] Karpathy, A. *LLM Wiki — idea file.* GitHub Gist. <https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f>

[7] Hetzner Cloud. <https://www.hetzner.com/cloud>

[8] FastAPI. <https://fastapi.tiangolo.com/>

[9] markdown-it-py — Python markdown parser, used for the read view. <https://github.com/executablebooks/markdown-it-py>

[10] Obsidian Git plugin. <https://github.com/Vinzent03/obsidian-git>

[11] Working Copy — iOS git client. <https://workingcopy.app/>

[12] github.dev — VS Code in the browser, for any GitHub repo. <https://github.dev/>

[13] FAISS — local vector similarity index (deferred to v0.2). <https://github.com/facebookresearch/faiss>

[14] Quartz — static site generator for Obsidian vaults (deferred). <https://quartz.jzhao.xyz/>

[15] Tantivy — Rust full-text search library (scale upgrade path). <https://github.com/quickwit-oss/tantivy>

[16] Meilisearch — full-text search engine (scale upgrade path). <https://www.meilisearch.com/>
