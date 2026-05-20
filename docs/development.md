# Development install

Editable install with every code path exercisable (tests, agent
runtime, dashboard):

```bash
pip install -e ".[dev,agent,dashboard,semantic]"
```

That's the canonical dev setup.

| Extra | Pulls in | Enables |
|---|---|---|
| `dev` | `pytest`, `pytest-asyncio`, `ruff`, `mypy`, `types-pyyaml` | `pytest`, `ruff check`, `mypy src/outmem` |
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
├── skills.py                   # SKILL.md loader (uses `skillfull` dep)
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
