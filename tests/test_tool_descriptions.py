"""What the model actually receives for each tool.

pydantic-ai does not send a tool's docstring. It parses it with griffe and
sends three parts: the first plain-text section as the tool description,
the ``Args:`` entries as per-argument schema, and the first ``Returns:``
entry. Every other section is parsed out and dropped without a warning —
an ``Example:`` heading over an indented block (griffe reads it as an
admonition), and anything after it, which lands in a second text section.

Until 0.19.1 that dropped every example call outmem's tools carried, the
federated palette's note on how page names are qualified, and a paragraph
of ``extend_page`` telling the model when to pass ``provenance``. So these
tests check the definitions pydantic-ai builds, never ``__doc__``.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai import Agent, FunctionToolset
from pydantic_ai.models.test import TestModel

from outmem.adapters.pydantic_ai import build_consult_wiki, wiki_tools
from outmem.adapters.wikiset import wikiset_read_tools
from outmem.agent.runtime import build_agent
from outmem.store import WikiStore
from outmem.wikiset import WikiSet

_ARGS_HEADER = re.compile(r"^Args:\s*$", re.MULTILINE)


def _defs(agent: Agent[Any, Any]) -> dict[str, tuple[Callable[..., Any], Any]]:
    """Every function tool the agent will offer: ``name -> (fn, ToolDefinition)``."""
    out: dict[str, tuple[Callable[..., Any], Any]] = {}
    for toolset in agent.toolsets:
        if isinstance(toolset, FunctionToolset):
            for name, tool in toolset.tools.items():
                out[name] = (tool.function, tool.tool_def)
    return out


def _palette(tools: list[Callable[..., Any]]) -> dict[str, tuple[Callable[..., Any], Any]]:
    return _defs(Agent(TestModel(), tools=tools))


def _indexed(root: Path, *, gated: bool = False) -> WikiStore:
    """A wiki with a semantic index built, so ``find_similar`` — offered
    only when one exists — is in the palette too."""
    WikiStore.init(root).close()
    cfg = root / "config.yaml"
    text = cfg.read_text(encoding="utf-8").replace(
        "embedding_model:", "embedding_model: test:bag-of-words  #", 1
    )
    if gated:
        text = text.replace(
            "approval:\n  required_for_writes: false",
            "approval:\n  required_for_writes: true",
        )
    cfg.write_text(text, encoding="utf-8")
    store = WikiStore.open(root)
    store.write_page("seed", title="Seed", body="Seed body.\n")
    store.semantic_reindex_all()
    assert store.semantic_available()
    assert store.config.outmem.approval.required_for_writes is gated
    return store


@pytest.fixture(scope="module")
def palettes(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, dict[str, tuple[Callable[..., Any], Any]]]:
    """Every tool outmem hands a model, grouped by how it gets there.

    Module-scoped: building two indexed wikis is the slow part, and every
    test here only reads the tool definitions.
    """
    tmp_path = tmp_path_factory.mktemp("palettes")
    store = _indexed(tmp_path / "open")
    other = WikiStore.init(tmp_path / "legal")
    gated = _indexed(tmp_path / "gated", gated=True)

    found = {
        "wiki_tools": _palette(wiki_tools(store)),
        "consult_wiki": _palette([build_consult_wiki(store.root, model=TestModel())]),
        "wikiset_read_tools": _palette(wikiset_read_tools(WikiSet([store, other]))),
        # The palette as outmem's own agent registers it — through
        # `tools=` normally, through a FunctionToolset with
        # `requires_approval` when writes are gated.
        "runtime": _defs(build_agent(store, model=TestModel(), include_steering=False)),
        "runtime-approval": _defs(
            build_agent(gated, model=TestModel(), include_steering=False)
        ),
    }
    # Coverage that could otherwise vanish silently: the index-only tool,
    # and each runtime path — the gated one routes through a toolset of
    # its own — offering the whole palette rather than some of it.
    assert "find_similar" in found["wiki_tools"]
    for label in ("runtime", "runtime-approval"):
        assert set(found[label]) == set(found["wiki_tools"]), label
    return found


def _lead_and_rest(fn: Callable[..., Any]) -> tuple[str, str]:
    """The docstring split at its ``Args:`` header: what pydantic-ai should
    send as the description, and what should hold nothing but arguments."""
    doc = inspect.cleandoc(fn.__doc__ or "")
    match = _ARGS_HEADER.search(doc)
    if match is None:
        return doc.strip(), ""
    return doc[: match.start()].strip(), doc[match.end() :]


class TestNothingIsDropped:
    def test_the_description_is_the_whole_lead_text(self, palettes: dict[str, Any]) -> None:
        # Every character before `Args:` has to arrive. An `Example:`
        # block — or a `Note:`, or any `Title:` line with a blank line
        # above and an indented block directly below — becomes a section
        # pydantic-ai discards, and takes every later paragraph with it.
        #
        # Compared with whitespace normalised: the test is for a drop, and
        # CI runs the newest pydantic-ai (`>=0.1.0`), whose formatting of
        # the text it does send is not ours to pin.
        problems: dict[str, str] = {}
        for label, defs in palettes.items():
            for name, (fn, tool_def) in defs.items():
                lead = " ".join(_lead_and_rest(fn)[0].split())
                sent = " ".join((tool_def.description or "").split())
                if sent != lead:
                    cut = next(
                        (i for i, (a, b) in enumerate(zip(sent, lead, strict=False)) if a != b),
                        min(len(sent), len(lead)),
                    )
                    problems[f"{label}:{name}"] = (
                        f"description stops matching the docstring at char {cut}: "
                        f"{lead[cut : cut + 60]!r}"
                    )
        assert problems == {}

    def test_nothing_follows_the_arguments(self, palettes: dict[str, Any]) -> None:
        # A paragraph after `Args:` is a second text section, and only the
        # first is sent. The federated palette used to append its note on
        # qualified page names there — on every tool, never delivered.
        # A `Returns:` section would be sent, but only by wrapping the
        # description in XML; the tools say what they return in prose, so
        # this flags one too.
        problems: dict[str, str] = {}
        for label, defs in palettes.items():
            for name, (fn, _tool_def) in defs.items():
                _lead, rest = _lead_and_rest(fn)
                stray = [ln for ln in rest.splitlines() if ln.strip() and not ln[0].isspace()]
                if stray:
                    problems[f"{label}:{name}"] = f"text after Args: {stray[0]!r}"
        assert problems == {}


class TestEveryToolShowsAWholeCall:
    def test_the_description_carries_a_call_naming_every_required_argument(
        self, palettes: dict[str, Any]
    ) -> None:
        # The example exists so a model does not drop a required argument,
        # fail validation and burn a retry — the write tools take several.
        # So the call has to be in the part that is sent, and it has to be
        # whole: every required argument named in one call.
        problems: dict[str, str] = {}
        for label, defs in palettes.items():
            for name, (_fn, tool_def) in defs.items():
                desc = tool_def.description or ""
                required = tool_def.parameters_json_schema.get("required", [])
                calls = [
                    desc[m.start() :].split("\n\n", 1)[0]
                    for m in re.finditer(rf"(?<![\w.]){re.escape(name)}\(", desc)
                ]
                if not calls:
                    problems[f"{label}:{name}"] = "no example call in the description"
                    continue
                whole = any(
                    all(re.search(rf"(?<!\w){re.escape(arg)}=", call) for arg in required)
                    for call in calls
                )
                if not whole:
                    problems[f"{label}:{name}"] = f"no single call names all of {required}"
        assert problems == {}


class TestMultiArgumentToolsSayHowMany:
    def test_the_first_paragraph_carries_the_count(self, palettes: dict[str, Any]) -> None:
        # A model that drops one of several required arguments fails
        # validation and burns a retry. The count has to be in what is
        # sent, near the top where a model skimming a long description
        # still meets it. Derived from the schema, so a tool that gains a
        # required argument is held to it without anyone editing a list.
        words = {2: "REQUIRES BOTH", 3: "REQUIRES ALL THREE"}
        problems: dict[str, str] = {}
        for label, defs in palettes.items():
            for name, (_fn, tool_def) in defs.items():
                count = len(tool_def.parameters_json_schema.get("required", []))
                if count < 2:
                    continue
                expected = words.get(count)
                opening = (tool_def.description or "").split("\n\n")[:2]
                if expected is None or not any(expected in para for para in opening):
                    problems[f"{label}:{name}"] = f"{count} required; opening lacks {expected!r}"
        assert problems == {}


class TestTheFederatedNote:
    def test_it_reaches_every_federated_tool(self, palettes: dict[str, Any]) -> None:
        # "The model has to know the qualifier exists, or it will pass a
        # bare slug into a set where two wikis hold that name and never
        # understand why it got the other one."
        # The exact sentence: "open" and "legal" on their own also occur in
        # the tools' examples and in the note's own `open/some-page`, so a
        # substring check for the names proves little.
        defs = palettes["wikiset_read_tools"]
        assert defs
        for name, (_fn, tool_def) in defs.items():
            desc = tool_def.description or ""
            assert "This session reads 2 wiki(s): open, legal." in desc, name
            assert "Page names are qualified" in desc, name
