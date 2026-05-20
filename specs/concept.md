# Agentic RAG — Conceptual Design

## Thesis

Similarity-based retrieval is a convergent dynamical system. Each turn pulls the next query toward what was already found, embeddings smooth away the edges, re-rankers encode majority taste, and the whole stack optimises for precision at the cost of adaptability. Fleck [3] calls this the relevance trap; Raudaschl [1, 2] names the product-side analogue when he warns against optimising for DCG-style vanity metrics; Cherny's team at Anthropic hit the same wall on code and concluded that agentic search beat RAG + local vector DB by a wide margin [4, 5]; Karpathy's reaction [6] was to stop asking and start compiling. A serious agentic RAG has to absorb all four critiques simultaneously, not just the convergence one. The architecture below is built on five inversions of the default RAG stack.

## Inversion 1 — Compile, don't index

Karpathy's pattern [6] is the right substrate: three directories, `raw/` (immutable sources), `wiki/` (current compiled consensus, plain markdown, interlinked), `log/` (decision and exploration trail). The wiki is the persistent, compounding artifact. Every successful exploration ends with a compaction step that commits a wiki delta — a synthesized page, a decision, a formula, a contradiction pair — with provenance pointers back into `raw/`. The next query that touches the same neighbourhood reads a compiled page, not raw chunks. This is the only mechanism that breaks the per-query rediscovery cost and the convergence loop simultaneously, because the wiki accumulates contradictions and temporal evolutions as first-class entries rather than averaging them away into an embedding centroid.

## Inversion 2 — No global vector index. Agentic search as the backbone

Following Cherny [4] and the SmartScope synthesis [5], the primary retrieval mechanism is a tiered tool palette the model drives itself, in cost order: ripgrep, glob, ls, read on `wiki/` first, then the same tools on `raw/`, then shell adjuncts (jq, duckdb against tabular sources, an LSP if code is in scope). The agent loops think → act → observe until it has enough, exactly the Claude Code pattern [4]. This sidesteps the four classical RAG failure modes the literature now consistently catalogues — staleness, the false-similarity problem (retrieved chunks that confidently answer a different question), the operational surface of maintaining an external index, and the security exposure of shipping embeddings off the box [5].

## Inversion 3 — Semantic search as fallback navigation, not source of truth

The one thing pure agentic search is genuinely bad at is concept search when you don't know the name or when spec terminology doesn't match implementation terminology [5]. So ship a small embeddings sidecar, but have it index `wiki/`, not `raw/`. The agent invokes it only when grep over compiled material returns nothing useful, treats its output as a list of *places to look*, and then resolves to actual files via read. The vector store is therefore tiny — hundreds to low thousands of compiled pages, not millions of raw chunks — cheap to keep fresh, and free of the noise that destroys production RAG. This is the hybrid SmartScope describes [5]: agentic backbone, semantic index only where it earns its keep.

**v0.1 status:** the semantic sidecar does not ship. Tier 1 (`wiki/`) and Tier 2 (`raw/`) shell tools only. Add Tier 3 when measured Tier 1+2 failure rates justify it — see `spec.md` §8 and §12.

## Inversion 4 — Divergence primitives as first-class tools

This is where Fleck's argument [3] operationalises. Alongside the convergence tools above, the agent has a parallel palette whose explicit job is *not* to return the most similar thing: a contradiction surfacer (search for passages whose stance opposes a target claim), a negative-space query (return what's missing from the coverage of a topic, not what's present), associative drift (a high-temperature walk over the wiki link graph), cross-domain bridges (deliberately sample from a distant cluster), and temporal evolution (diff how a concept has been treated over time in `log/` and successive wiki versions). The orchestrator's planning prompt forces it to decide, before retrieving, whether the query needs precision or expansion of the frame; non-trivial queries get a brief divergent pass to map the possibility space before converging on an answer. This is the missing half of cognition [3] that every current memory system skips when it runs on similarity alone.

**v0.1 status:** only temporal evolution ships, and as a documented shell pattern (`git log -p --follow`) rather than a dedicated tool. The other four primitives are deferred until temporal evolution proves the divergence layer carries weight in real use — see `spec.md` §8 and §12.

## Inversion 5 — Subagent fan-out and mandatory compaction for cost control

The honest counterargument to agentic search — token blow-up on large corpora, documented in Anthropic's own issue tracker and reiterated by several Milvus-adjacent critiques — is mitigated two ways. First, partition exploration across parallel subagents with isolated context windows, each returning only a condensed finding; this is Cherny's "agent topologies" idea applied internally [4]. Second, the writeback step is not optional — it is a turn the orchestrator is required to take before responding, so the marginal cost of identical future queries collapses toward zero. Without mandatory writeback, agentic search is just an expensive way to re-derive the same answer, and Karpathy's core complaint [6] stands: the system rediscovers knowledge from scratch on every question.

**v0.1 status:** mandatory writeback ships; subagent fan-out does not. Single-agent serial exploration is fine for the wiki size v0.1 targets (~10k pages). Subagents become an answer to a token-budget problem v0.1 does not have — see `spec.md` §12.

## Evaluation — where most RAG projects actually die

Borrow TARS [2] wholesale and refuse the vanity metrics. For each retrieval feature (the contradiction tool, the semantic sidecar, the cross-domain bridge), measure Target (what fraction of sessions actually need this kind of retrieval), Adopted (does the agent invoke it when it should), Retained (does invocation produce a wiki delta or a decision entry in `log/`), Satisfied (did the user act on the result). Map features to the TARS 2×2; kill the ones nobody uses, double down on the overperformers. On top of TARS, run the SmartScope operational triad [5] — time-to-first-relevant-fact, tokens-per-task, staleness incidents — and add a divergence-yield metric: the fraction of sessions in which a non-obvious connection was surfaced *and* acted upon. Recall@k and nDCG do not appear on this scorecard. Similarity and usefulness are not the same thing, and optimising the former actively harms the latter [1, 3].

## What the system explicitly refuses to do

It doesn't ship raw documents to a third-party embedding service. It doesn't pre-index anything that changes daily. It doesn't return chunks to the model without a compiled summary attempt first. It doesn't treat conversation history as memory — Fleck's static-factoid trap [3] — so relevant material gets extracted into `wiki/` or discarded. It doesn't run a single mega-agent with a 2M-token context, because the context-rot literature is now unambiguous that this just amplifies whatever noise retrieval introduced. And it doesn't optimise for the question you asked — it optimises for the answer you needed, which is often adjacent [3].

## Failure modes the design defends against

The relevance trap is addressed by explicit divergence tools plus mandatory writeback, which together prevent conversation history and retrieval from compounding similarity. Index staleness is addressed by having no global index over `raw/`; the semantic sidecar covers `wiki/` only, which the agent itself maintains. Token blow-up is addressed by tiered tool cost, parallel subagents, and writeback collapsing repeat queries. Context rot from over-retrieval is addressed by the compaction step and by using smaller child chunks with parent-section fallback when multiple children hit. Hallucinated tool calls are addressed by a constrained tool palette with full observability on every call.

## Next passes

If the concept is sound, the natural next passes are the orchestrator's planning prompt (convergence/divergence decision logic), the contradiction-surfacer's implementation (the trickiest divergence primitive, because it requires stance extraction not just embedding opposition), the wiki-compaction policy (what gets promoted from `log/` to `wiki/` and when), and the TARS scorecard with concrete thresholds and A/B protocols against an agentic-only baseline.

---

## References

[1] Raudaschl, A. *The Relevance Trap.* Breaking Product. <https://breakingproduct.substack.com/p/the-relevance-trap>

[2] Raudaschl, A. *TARS: A Product Metric Game Changer.* UX Collective. <https://uxdesign.cc/tars-a-product-metric-game-changer-c523f260306a>

[3] Fleck, J. *Divergence Engines: Escaping the Relevance Trap.* Medium, October 2025. <https://medium.com/@j0lian/divergence-engines-escaping-the-relevance-trap-1bbdbee55ea6>

[4] Nicolai, V. *Claude Code Doesn't Index Your Codebase. Here's What It Does Instead.* Vadim's Blog. <https://vadim.blog/claude-code-no-indexing>

[5] SmartScope. *Settling the RAG Debate: Why Claude Code Dropped Vector DB-Based RAG and the Reality of Code Search.* <https://smartscope.blog/en/ai-development/practices/rag-debate-agentic-search-code-exploration/>

[6] Karpathy, A. *LLM Wiki — idea file.* GitHub Gist. <https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f>
