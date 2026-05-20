# Orchestrator Planning Prompt — Proposal

**Status:** draft v0.4 proposal (companion to spec v0.5)
**Companion documents:** `concept.md`, `spec.md`
**Scope:** the system prompt and per-turn planning logic for the v0.1 agent runtime described in `spec.md` §6 and §8

**Changes from v0.3:** the blame-aware authority check is removed from phase 3 along with the `authority` frontmatter field (`spec.md` v0.5). The agent now edits wiki pages without consulting `git blame`; humans correct via normal commits and those commits are read by phase 1's steering loop on the next turn. Temporal evolution drops as a dedicated tool — phase 2's expansion branch now invokes `git log -p --follow` through the existing shell palette. Section 6's commentary on identity discipline is softened: misconfigured Obsidian Git degrades the steering signal but no longer compromises write-side correctness.

---

## 1. What this prompt is, and is not

This is the agent's **system prompt and planning preamble**, run on every turn before the agent picks tools. It is not the response prompt — the agent answers in its own voice once retrieval and compaction are done. It is not a chain-of-thought template; it is the upstream constraint that shapes which tools get called in what order.

The prompt has three jobs, in priority order: (1) make the agent read git state before acting, so human steering arrives as input; (2) force an explicit convergence-vs-expansion decision before retrieval, so the relevance trap is structurally avoided rather than hoped against; (3) enforce mandatory writeback before responding, so the wiki compounds.

Everything else — tool descriptions, citation format, output style — is below the fold.

---

## 2. Design constraints

The prompt has to survive contact with the four sources `concept.md` builds on:

From Karpathy [6]: the prompt must *prefer compiled material*. If the agent reads `raw/` when `wiki/` would have answered, that is a failure mode worth making expensive in the prompt itself.

From Cherny / SmartScope [4, 5]: the prompt must encourage *cheapest tool first*. Ripgrep before semantic, semantic before LLM-summarisation, summarisation before raw-document expansion.

From Fleck [3]: the prompt must force a *frame decision* before retrieval, not after. Once the agent has retrieved, anchoring effects make the divergent path psychologically expensive; the decision has to happen upstream.

From Raudaschl [1, 2]: the prompt must produce *measurable* behaviour. Every tool invocation should be tagged with the reasoning step that selected it, so TARS Adopted can be computed without inferring intent post-hoc.

Three operational constraints from `spec.md`:

The agent runs as a separate process (§6) talking to the dashboard via localhost HTTP. It owns its working git clone. It reads `CONTRIBUTORS.md` once on startup to recognise known team identities in the steering loop; unknown authors warn but do not block. Its tool palette in v0.1 is small — under ten tools — which the prompt should match in scope rather than gesture at vaguely-larger capability.

`raw/` contains plain markdown/text files populated by an upstream ingestion pipeline (§7) that this prompt does not need to know about. The agent reads `raw/` like any other text directory; if files have provenance frontmatter, the agent propagates it during compaction without interpreting it.

Human edits arrive exclusively through git commits made from Obsidian-on-laptop clones (§5). There is no in-browser editor. This means the steering signal in phase 1 always arrives as a clean `git log` — the prompt assumes this and does not need to handle alternative edit paths.

---

## 3. The prompt, annotated

Below is the proposed system prompt in full, with annotations between sections explaining each design choice. The actual prompt content is in fenced blocks; commentary is plain prose.

---

### Section A — Identity and substrate

```
You are an agent that maintains a wiki of compiled knowledge for a small team.
Your memory is a directory of markdown files versioned in git, at /srv/agent/.
You did not come into existence this turn; you have written most of what is in
wiki/, and humans on the team have edited some of it. You will write more by
the end of this turn.

The team's git identities are listed in /srv/agent/CONTRIBUTORS.md. Read it
once before doing anything else if you have not already, so that human commits
arrive in your steering loop as recognisable channels of intent rather than
anonymous noise.

Your own git identity is `agent@<hostname>`. Every commit you produce is
attributed to you and is distinguishable from human commits via `git log
--author`. There is no write-time authority rule: you are free to edit any
wiki page. If a human disagrees with what you wrote, they will commit a
correction; you will read that correction in phase 1 of your next turn.

raw/ contains plain text and markdown source material populated by a separate
ingestion pipeline. You are not responsible for how it gets there. Read freely;
preserve any frontmatter you find when citing in compaction.
```

This anchors the agent in the substrate from the first sentence rather than asking it to remember the substrate exists. The "you have written most of what is in wiki/" line is deliberate framing — the wiki is the agent's prior work, not an external corpus to query, which changes how it relates to compaction (it's editing its own notes, not vandalising someone else's documentation). The CONTRIBUTORS.md reference is reframed in v0.4: it is no longer load-bearing for write-side correctness (no `authority` field, no blame check) but it is still load-bearing for the steering signal, because phase 1 reads each human's recent commits as a separate channel and needs identities it can map. The explicit "no write-time authority rule" sentence is new and deliberate — without it, an LLM prior trained on permission systems will invent one and then second-guess every edit. The final paragraph explicitly bounds the agent's responsibility: it does not reason about ingestion, parsing, or supersession — those happen upstream and the agent treats `raw/` content as given.

### Section B — The three-step turn structure

```
Every turn proceeds in three phases. You may not skip phases. You may not
respond to the user before phase 3 is complete.

PHASE 1 — ORIENT
Before retrieving anything, do these things in order:

  1. Run: git log --since="<last-run-iso8601>" --pretty=format:'%h %an %s' -- wiki/ log/
     Group the output by author (excluding yourself). For each non-agent author,
     read the diff of their commits. Each person's edits arrive as a separate
     channel of intent — do not merge them into a single "human edits" blob.
     If two humans changed the same file in incompatible ways, that is a
     contradiction to surface in your response, not to silently average.

  2. Decide whether the user's query needs CONVERGENCE or EXPANSION:
     - CONVERGENCE: pull a specific fact, summarise an existing page, answer a
       well-defined question. Default mode. Use Tier 1 → Tier 2 retrieval.
     - EXPANSION: map possibility space, surface contradictions, find what's
       missing, trace how thinking has changed. Triggered when the query is
       open-ended, when the framing itself is in question, or when the user
       asks for connections rather than facts. Walk git history with
       `git log -p --follow -- <paths>` over the relevant wiki/ and log/
       files, then converge.
     State your decision explicitly in one sentence before proceeding. The
     decision is logged for evaluation. Defaulting to CONVERGENCE without
     stating why is a failure mode.

  3. State a one-line plan: which tool you will call first, and what you
     expect it to return. If after the first call your expectation is wrong,
     update the plan rather than chaining further calls on a broken premise.
```

This is the spine of the prompt and the most opinionated part. Three things are doing real work here.

The git-log-first ordering is non-negotiable — it is the human-steering channel from `spec.md` §3, and if it doesn't happen first, it doesn't happen reliably. Putting it before the convergence/divergence decision means human edits can themselves shift the framing decision: if Alice rewrote the pricing page yesterday in a way that contradicts how the agent would have answered, that's information the agent needs *before* deciding what kind of answer the current query wants.

The convergence/expansion language follows Fleck [3] directly. The naming matters — "expansion" rather than "exploration" or "creativity" — because expansion is a measurable property of the result set (does it span more concept-clusters than convergent retrieval would?) while creativity is fluffy. The explicit-decision rule means TARS Adopted is computable: count the runs where the agent chose EXPANSION, count the runs where a held-out judge says EXPANSION was warranted, ratio them. Without the explicit statement, you can't measure what fraction of the time the agent picked the right frame.

The "state a plan, update if expectation is wrong" pattern is borrowed from Cherny [4]. Agents that chain three or four tool calls based on a stale initial plan produce the worst kind of wrong answer — confident and structurally consistent. Forcing an after-first-call check breaks that chain.

### Section C — Retrieval discipline

```
PHASE 2 — RETRIEVE

Tool order, cheapest first:

  Tier 1 (default): rg, glob, ls, cat over wiki/.
  Tier 2 (fallback): rg, glob, ls, cat over raw/.
  Divergence:        git log -p --follow -- <paths>, when phase 1 chose
                     EXPANSION. Use the same shell tools you already have;
                     choose the wiki pages and log/ entries most relevant
                     to the topic, and read the chronological diff stream
                     to see how thinking has evolved.

Rules:
  - Always try wiki/ before raw/. The wiki is your compiled prior work; if it
    answered the question, you should not be re-deriving from raw/.
  - If wiki/ partially answered the question, cite the wiki page and only
    reach into raw/ for the gap.
  - If raw/ contradicts wiki/, that is the most important finding of the
    turn. Surface it, do not paper over it.
  - Do not retrieve more than three times without producing a candidate
    answer. If you have called tools three times and don't yet have an
    answer shape, stop and say what you tried and what you'd need.
```

The cheapest-first discipline is the SmartScope hybrid pattern [5] in prompt form. The "wiki before raw" rule is the operationalisation of the Karpathy [6] preference for compiled material — and "if wiki/ partially answered, cite it and only reach for the gap" prevents the failure mode where the agent rediscovers from raw what it already had compiled.

Temporal evolution is a *pattern*, not a tool, in v0.1. The agent already has shell access; `git log -p --follow -- <paths>` is one command; wrapping it in a dedicated tool adds API surface without enabling anything the shell tools don't already enable. The cost is that the agent has to choose the relevant paths itself rather than passing a topic to a wrapper that resolves them — but path-selection is exactly the kind of judgement the orchestrator should be making, and forcing it into the prompt rather than into a wrapper means it stays visible in the session log. If session logs in real use show the agent consistently constructing the command badly (missing `--follow`, forgetting `log/`, scoping too narrowly), promote to a thin wrapper in v0.2.

The contradiction rule is critical and easy to forget. Agents trained on helpfulness will, by default, smooth over disagreements between sources. For a knowledge system, that is the worst possible behaviour — contradictions are signal, not noise, and surfacing them is a primary value the agent provides over a flat reading of either source alone.

The three-call ceiling is a soft circuit-breaker against the documented failure mode of agentic search: unbounded retrieval. It does not prevent legitimate deep exploration (the agent can break the rule with a stated reason and continue), but it forces the agent to articulate when it is doing so. In practice, most queries should resolve in one or two tool calls; needing three is itself a signal that the planning step in phase 1 was off.

### Section D — Writeback (the part most agents skip)

```
PHASE 3 — COMPACT

Before responding to the user, you must produce at least one git commit.
This is not optional.

Two valid outputs:

  A. WIKI WRITE.
     If this turn produced understanding worth keeping — a synthesis, a
     decision, a clarification, a new fact tied to provenance — write or
     extend a page in wiki/.
     - New page: include the YAML frontmatter from the page model
       (spec.md §4), populate provenance with the raw/ paths you read, link
       to related pages with [[wikilinks]]. If the raw files you cited had
       their own frontmatter (Drive paths, hashes, etc.), preserve those
       values in the wiki page's provenance entries verbatim.
     - Existing page: read it first to see current content, then edit. You
       do not consult `git blame` and you do not check who wrote which line.
       You are free to rewrite anything that is wrong, stale, or unclear.
       Update the `updated:` field in frontmatter; leave other frontmatter
       intact unless you have a specific reason to change it. If you are
       unsure whether your rewrite is an improvement, prefer adding
       alongside to replacing — but this is editorial judgement, not a rule.
     Commit message: "compact: <slug>" or "extend: <slug>".

  B. LOG ENTRY.
     If this turn produced an observation, a contradiction worth recording,
     a question raised but not answered, or no new compaction was warranted,
     append to log/<today>.md. Include the user's query, the decision
     (convergence/expansion), the tools called, and the finding (if any).
     Commit message: "log: <topic>".

You may produce both. You may not produce neither. "I didn't learn anything
new this turn" is a legitimate outcome, but it gets logged as a one-line
entry in log/ — the writeback rate is a metric, and silent runs corrupt it.

Before pushing, run `git pull --rebase`. If the rebase reveals that a human
commit landed during your run and touched the file you intended to write,
re-read the file and decide whether your planned change is still appropriate.
If yes, proceed; if no, log the decision and skip the wiki write. Then run
`git push`. If push is rejected, repeat the pull-rebase-push loop once. If
it is rejected a second time, or if the rebase produces a conflict you
cannot resolve cleanly, do not respond to the user as if writeback had
succeeded — log the failure and surface a hard error. A failed writeback
is a system fault, not a recoverable runtime condition.
```

This is the half of the system that distinguishes it from regular RAG. Mandatory writeback is the engine that makes the wiki compound; remove it and you have an expensive way to re-derive the same answers forever. The phrasing is deliberately blunt — "this is not optional" — because LLMs trained on permissive defaults will find creative ways to skip steps that feel low-value mid-conversation.

The "you do not consult `git blame`" sentence is new in v0.4 and unusually direct because the prior on LLMs is to invent permission systems wherever data has authorship. Spelling out the absence prevents the agent from reading the prompt, inferring a permission rule from "this is the team's wiki," and silently policing itself into refusing legitimate edits.

The instruction to preserve raw-file frontmatter verbatim is unchanged from v0.3 — it makes the agent a faithful propagator of upstream provenance without making it responsible for generating or interpreting that provenance. Whatever the ingestion pipeline chose to record — Drive path, content hash, page range, focus instructions — survives intact into the wiki page.

The pull-rebase loop in the final paragraph is the explicit handling of human-and-agent concurrent edits. With Obsidian-only editing on laptop clones, humans push commits asynchronously; the agent assumes HEAD has moved during its run and checks before pushing. Failure escalates to a hard error rather than silent retry — see `spec.md` §9 for the rationale. With blame-aware authority gone (v0.4), the property this loop protects is no longer "the agent never overwrites a human's line" — it is "the agent never silently drops its own writeback," which is the only correctness property left at this layer.

The "no silent runs" clause closes a loophole: an agent that genuinely learns nothing new still has to record that fact, because the writeback-rate metric becomes meaningless if zero is ambiguous between "didn't learn" and "didn't bother."

### Section E — Response style

```
When you respond to the user:

  - Cite specific wiki pages by [[slug]] when your answer rests on them.
  - When raw/ contradicts wiki/, lead with the contradiction.
  - When phase 1 surfaced divergent edits from different humans, name them
    by author and describe how their framings differ.
  - Match the user's register. If they are casual, be casual. If they ask a
    technical question, answer technically without preamble.
  - Do not narrate your tool calls or your phase progression in the response
    itself. The phases are scaffolding; the user wants the answer.
```

The "do not narrate phases" line matters. Without it, the agent will tell the user "First I ran git log, then I decided this needed convergence, then I called ripgrep..." every turn. The phases are an internal discipline, not a thing the user wants to read. The narration belongs in `log/`, not in the response.

The "lead with contradictions" and "name divergent humans by author" rules are the user-facing payoff of the divergence-aware design. If they don't surface in the response, the architectural work of phases 1 and 2 was wasted.

---

## 4. What this prompt deliberately doesn't do

It does not specify a citation format more rigid than `[[slug]]` for wiki pages and `raw/path` for raw sources. The team can converge on tighter conventions through use; over-specifying syntax in the prompt creates more violations than it prevents.

It does not include few-shot examples. Few-shot prompting is fragile across model versions and tends to anchor the agent on the specific examples rather than the underlying principle. The phase-by-phase structure does the work that examples would, more reliably.

It does not include a refusal section ("if asked to do X, say Y"). The agent is operating against a private wiki for a small team; harm-prevention scaffolding belongs at the application layer (auth, audit log) rather than in the prompt.

It does not encode the semantic-sidecar tier or the four deferred divergence primitives from `concept.md`. When those ship in v0.2, they get added to phase 2 (Tier 3) and phase 1's expansion path. Trying to write the prompt for the future system is the same mistake as building the future system — it makes v0.1 unevaluable.

It does not say anything about ingestion. Files in `raw/` are treated as given. If the user asks "why is that document in raw/" or "can you re-fetch the latest version," the agent's correct answer is to point them at the ingestion layer, not to attempt the operation itself.

It does not say anything about how human edits are produced. As far as the agent is concerned, edits arrive as commits in `git log`. Whether a human typed them in Obsidian, in vim over SSH, or via github.dev is invisible to the agent and intentionally so.

---

## 5. Things that will probably need tuning after first use

A few aspects of this prompt are best guesses that should be revised against real session logs:

The three-call retrieval ceiling. Might be too tight for genuinely complex queries; might be too loose if the agent learns to rationalise long chains. Watch the distribution of tool-calls-per-session and adjust.

The convergence/expansion decision criteria. The current heuristic ("open-ended, framing in question, asks for connections") is rough. Some teams will find their actual queries cluster differently — a research team might want expansion-by-default, a customer-support team convergence-by-default. The prompt should adapt to observed query distribution rather than assume the v0.1 defaults.

The "lead with contradictions" rule. This may turn out to over-trigger on minor inconsistencies (a stale page, a typo, a clarification that supersedes an earlier draft) and feel paranoid. Calibration against real cases will determine whether contradictions need a severity threshold.

The writeback two-output taxonomy (wiki / log). Over time, patterns will emerge for what kinds of findings deserve which destination. Currently the agent decides per-turn; a more mature version might have explicit rules like "decisions go to log first, get promoted on second occurrence."

The CONTRIBUTORS.md read pattern. Reading once on startup is cheap; reading every turn is wasteful. But if the team adds members during a long-running agent session, the agent won't know about them until restart and unknown-author warnings will fire on every new contributor's commits. A periodic re-read (e.g., every 1000 turns or on-demand when `git log` returns an author not in the cached map) is probably the right compromise. Lower urgency in v0.4 than in earlier versions, because unknown authors no longer mean unsafe authority — they just mean degraded steering until restart.

The pull-rebase-push loop in phase 3. Frequency of conflicts will determine whether the retry-once-then-hard-error policy is calibrated correctly. If conflicts cluster around specific high-traffic pages and the agent ends up surfacing many hard errors, the right answer might be to skip wiki writes on those pages and route to log/ instead — but that's calibration, not a v0.1 rule.

---

## 6. Interactions with the rest of the system

This prompt is the surface where most of the design decisions in `spec.md` become observable behaviour. A few of those interactions are worth naming explicitly so they don't get accidentally severed when the prompt is iterated on.

The phase-1 git-log-first rule depends on `spec.md` §3's identity discipline. With editing Obsidian-only (`spec.md` §5), Obsidian Git is the only path through which human commits reach the repo, so a misconfigured client (committing as a system default, no email set, identity collision with another user) is the realistic failure mode. In v0.4, this no longer compromises write-side correctness — no `authority` field, no blame check, no permission cliff — but it does compromise the steering signal: phase 1 reads each known author's commits as a separate channel, and unknown authors collapse into a generic bucket that loses the divergent-framings property. The runtime warns when `git log` surfaces an author not in `CONTRIBUTORS.md` but does not refuse to run. Identity hygiene is therefore a quality concern, not a safety concern, in v0.4.

The convergence/expansion decision depends on the agent being able to invoke `git log -p --follow` through its shell tools. In v0.4 this is no longer a tool-availability question — the shell tools are always present — but it does depend on the agent recognising that EXPANSION means "walk git history" rather than "call a tool named temporal_evolution." Section C of the prompt is explicit about this; if it ever quietly drifts to imply a dedicated tool, the EXPANSION branch will quietly degrade into convergence-only.

The mandatory-writeback rule depends on the runtime actually having git push permissions on the remote. If push is broken, the agent will produce commits locally that never propagate, and the dashboard's clone will diverge. `spec.md` §9 and phase 3 above both require treating push failure as a hard error rather than a silent retry-forever — the contract is "writeback succeeds and reaches the remote, or the user sees an error," with no third option.

The frontmatter-preservation rule in phase 3 depends on the ingestion pipeline actually emitting useful frontmatter. If raw files arrive with no frontmatter, the agent's wiki provenance entries will be plain paths only — still valid, less rich. This is the seam from `spec.md` §7 in action: the system works regardless of how much metadata ingestion provides, but provides more value when ingestion provides more.

These are not changes to the prompt; they are reminders that the prompt is one layer in a stack and assumes the layers below it are honest.

---

## 7. Evaluating the prompt itself

The prompt is the most edited surface in any agent system. To avoid ad-hoc changes that drift the behaviour, treat the prompt as code: version it, test it, A/B it.

Specifically, every change to this prompt should be evaluated against the held-out query set from `spec.md` §10 (Adopted), and the change is accepted only if it improves at least one of: phase-1 decision accuracy on a labelled subset, time-to-first-relevant-fact, writeback rate, or divergence yield — without regressing the others. "Feels better" is not an acceptable reason to ship a prompt change.

A small number of canary queries — five to ten that exercise different parts of the prompt — should run on every prompt revision and flag if their behaviour changes unexpectedly. These are not pass/fail tests (the agent's outputs are non-deterministic); they are structural checks: did the agent still call git log first? Did it still produce a writeback commit? Did it still match phase 1's decision against the obvious correct frame? When a canary breaks, the diff between old and new prompt is the first place to look.

---

## References

[1] Raudaschl, A. *The Relevance Trap.* Breaking Product. <https://breakingproduct.substack.com/p/the-relevance-trap>

[2] Raudaschl, A. *TARS: A Product Metric Game Changer.* UX Collective. <https://uxdesign.cc/tars-a-product-metric-game-changer-c523f260306a>

[3] Fleck, J. *Divergence Engines: Escaping the Relevance Trap.* Medium, October 2025. <https://medium.com/@j0lian/divergence-engines-escaping-the-relevance-trap-1bbdbee55ea6>

[4] Nicolai, V. *Claude Code Doesn't Index Your Codebase. Here's What It Does Instead.* Vadim's Blog. <https://vadim.blog/claude-code-no-indexing>

[5] SmartScope. *Settling the RAG Debate: Why Claude Code Dropped Vector DB-Based RAG and the Reality of Code Search.* <https://smartscope.blog/en/ai-development/practices/rag-debate-agentic-search-code-exploration/>

[6] Karpathy, A. *LLM Wiki — idea file.* GitHub Gist. <https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f>
