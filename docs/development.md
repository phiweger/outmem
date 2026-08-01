# Development install

Editable install with every code path exercisable (tests, agent
runtime, dashboard):

```bash
pip install -e ".[dev,agent,dashboard,semantic]"
```

That's the canonical dev setup.

| Extra | Pulls in | Enables |
|---|---|---|
| `dev` | `pytest`, `pytest-asyncio`, `hypothesis`, `ruff`, `mypy`, `types-pyyaml` | `pytest`, `ruff check`, `mypy src/outmem` |
| `agent` | `pydantic-ai`, `jinja2` | `outmem ask`, `outmem.agent.*`, the PydanticAI adapter tests |
| `semantic` | `pydantic-ai`, `sqlite-vec` | `outmem similar` / `reindex`, vector store |
| `dashboard` | `fastapi`, `uvicorn[standard]`, `markdown-it-py`, `jinja2` | `outmem dashboard`, `outmem.dashboard.*` tests |
| `logfire` | `logfire` | Pydantic Logfire instrumentation |

You can skip `[pydantic-ai]` from the install list — `[agent]` already
includes it.

System deps must also be on PATH: **`git`** and **`ripgrep`** (`rg`).

Verify everything loaded:

```bash
pytest -q
ruff check src/outmem tests evals
mypy src/outmem
outmem --version
```

### Property-based tests

`tests/test_properties.py` uses [Hypothesis](https://hypothesis.readthedocs.io/)
for the pure functions with algebraic invariants — key normalisation,
frontmatter round-tripping, slug↔path round-tripping, and the derivation rule
`sources backfill` shares with the ingest-time refusal.

It is scoped deliberately. The most expensive bugs in outmem's history were
round-trip failures that shipped because the example-based tests happened not
to contain the input that broke: a tag written `007` came back as `7`, and a
body opening with an indented code block lost its indent. Both are properties a
generator finds mechanically. Anything touching git, SQLite or the filesystem
stays example-based — Hypothesis would be slow there and the value is in
integration, not algebra.

The profile sets `derandomize=True`, so a bad seed can't fail an unrelated PR's
CI. To explore beyond the fixed examples:

```bash
pytest tests/test_properties.py --hypothesis-seed=random -p no:randomly
```

If you'd rather keep the dev environment isolated:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,agent,dashboard,semantic]"
```

## Repository layout

```
src/outmem/
├── store.py                    # WikiStore — public API + write_page/extend_page/append_log
├── _store/                     # internal facets imported by store.py
│   ├── sources.py              #   add_source, list_sources, record_ingestion, ...
│   └── semantic.py             #   reindex, find_similar, the indexer ...
├── _sqlite.py / _time.py       # shared helpers used by both DBs and timestamp call sites
├── _logfire.py                 # opt-in Pydantic Logfire hook
├── config.py                   # OutmemConfig + YAML loader
├── frontmatter.py / slug.py    # page model + wikilink rewriter
├── git_ops.py / history.py     # subprocess wrappers + named queries
├── search.py / backlinks.py    # rg --json + HEAD-keyed cache
├── identity.py / state.py      # CONTRIBUTORS.md + .outmem/ state (fcntl-locked)
├── sources.py                  # sources registry (SQLite-backed)
├── skills.py                   # SKILL.md loader (uses `outskilled` dep)
├── lint.py / index.py          # outmem lint + wiki/index.md auto-maintenance
├── exceptions.py               # OutmemError hierarchy
├── adapters/pydantic_ai.py     # wiki_tools() + skill_text()
├── agent/                      # orient → retrieve → compact runtime + system.j2
├── dashboard/                  # FastAPI read view
├── semantic/                   # sqlite-vec wrapper + chunker + embedder probes
├── skills/notes/               # bundled SKILL.md files rendered into the system prompt
└── cli/__main__.py             # the `outmem` command

docs/                           # cli, python-api, features, configuration, this file
evals/                          # cases + harness + fixtures (eval suite, opt-in)
examples/starter-wiki/          # pre-populated example to try the library against
specs/                          # conceptual rationale, v0.1 spec, planning prompt
tests/                          # pytest suite, ruff + mypy strict clean
```
