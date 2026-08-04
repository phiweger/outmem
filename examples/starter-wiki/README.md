# Starter wiki

A small worked example showing the directories outmem maintains
(`wiki/pages/`, `wiki/sources/`, `log/`, and the eventual `.outmem/`
cache) populated with realistic content. Use it to try the library
against a non-trivial corpus without seeding one yourself.

## Layout

```
starter-wiki/
├── CONTRIBUTORS.md             # known team identities
├── wiki/
│   ├── pages/                  # compiled knowledge (one concept per file)
│   │   ├── pricing-formula.md
│   │   └── acme-msa.md
│   └── sources/                # source documents, tracked in git
│       ├── pricing-deck-2026-Q1.md
│       └── acme-msa-text.md
└── log/                        # decision + exploration trail
    └── 2026-05-01.md
```

Not shown because this example doesn't need one: `wiki/sources-local/`,
an untracked sibling of `sources/` for material you may read but not
redistribute. See [docs/sources.md](../../docs/sources.md).

## Try it

```bash
# 1. Copy the example into a working directory and turn it into a git repo.
cp -r examples/starter-wiki /tmp/wiki
cd /tmp/wiki
git init --initial-branch=main
git add . && git -c user.name="seed" -c user.email="seed@example.com" \
    -c commit.gpgsign=false commit -m "compact: seed starter wiki"

# 2. Point outmem at it.
export OUTMEM_PATH=/tmp/wiki
outmem read pricing-formula              # full page (frontmatter + body)
outmem search "cost-plus" --scope wiki   # ripgrep over wiki/
outmem search "Acme" --scope sources     # ripgrep over the source docs
outmem history pricing-formula           # commits touching the page
outmem evolution pricing-formula         # git log -p --follow stream

# 3. Add to it.
outmem log onboarding <<< "- explored the starter wiki, looks good"
outmem write discounts \
    --title "Discount tiers" \
    --provenance sources/pricing-deck-2026-Q1.md \
    --tag pricing \
    <<< "Standard discounts: 5% / 10% / 15% by volume."
outmem extend pricing-formula <<< "Revised: cost-plus 40% as of Q3."

# 4. With an LLM (requires `pip install "outmem[agent]"`).
#    Two options for credentials:
#    a) Export the key inline:
export ANTHROPIC_API_KEY="sk-ant-..."
#    b) Drop a `.env` at your CWD (or any ancestor) — outmem walks
#       upward from CWD looking for one:
#       echo 'ANTHROPIC_API_KEY=sk-ant-...' > .env
#
# The model already comes from `config.yaml` at the wiki root, but
# you can override with --model or $OUTMEM_MODEL.
outmem ask "what is our pricing formula and where does it come from?"
```

## What you can verify

- `outmem search "pricing"` finds matches in the pages; `--scope sources`
  reaches the source documents.
- `outmem backlinks` (via the dashboard or `WikiStore.backlinks`)
  shows `pricing-formula` referring to `acme-msa` and vice versa.
- Commit subjects in `git log` follow the `compact:` / `extend:` /
  `log:` grammar — `git log --grep='^compact:' | wc -l` is the TARS
  *Retained* signal in unprocessed form.
- `wiki/pages/pricing-formula.md` has a YAML frontmatter block; the
  `provenance` field carries a path back into `wiki/sources/`.
