# Search patterns: convergence vs. expansion

Two retrieval shapes; pick before you call any tool.

## CONVERGENCE — the default

The user wants a specific fact, the current value of something, a summary
of an existing page, a yes/no answer. Use `outmem search` (Tier 1 →
Tier 2) and stop as soon as you have an answer.

Examples:
- "What is the pricing formula?"
- "Have we written anything about Acme?"
- "When did we last update the discount policy?"

## EXPANSION — the divergence-aware mode

The user is asking about *change*, *contradiction*, *what's missing*,
or *the framing itself*. Convergence here gives the wrong shape of
answer — you don't want the latest state, you want the trajectory.
Reach for the `evolution` skill instead of (or before) `search`.

Examples:
- "How has our thinking on X changed?"
- "Where do our notes disagree?"
- "What hasn't been said about Y?"
- "Is there a hidden assumption in how we frame Z?"

## How to decide

State the decision in one sentence before you start retrieving:

> "Convergence: the user wants the current pricing formula."
> "Expansion: the user wants to see how our framing of Acme has shifted."

Defaulting to CONVERGENCE without saying why is a failure mode. Almost
all queries are convergent, but the few that aren't reward expansion
disproportionately.

## Pattern syntax

`outmem search` accepts regex by default. Use `--fixed-strings` for
literal matches (especially `[[wikilink]]` references which contain
regex metacharacters). `-i` for case-insensitive.

```bash
outmem search 'pricing|discount' --scope wiki     # regex alternation
outmem search '[[pricing-formula]]' --fixed-strings  # literal wikilink
outmem search 'TODO' -i --scope log                 # case-insensitive in log/
```
