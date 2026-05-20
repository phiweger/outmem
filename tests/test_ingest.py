"""Integration tests for source registration + agent-driven ingestion."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from outmem.exceptions import OutmemError
from outmem.sources import (
    REGISTRY_FILENAME,
    SHA_PREFIX_LEN,
    SourceRegistry,
    compute_sha256,
)
from outmem.store import WikiStore


def _rel(src: Path, *, into: str | None = None, name: str | None = None) -> str:
    """Expected hash-layout rel_path for ``src``.

    Mirrors what :func:`copy_source` produces — encoded once here so
    individual tests don't have to recompute sha256s inline.
    """
    short = compute_sha256(src)[:SHA_PREFIX_LEN]
    parts = [into, short, name or src.name]
    return "/".join(p for p in parts if p)

# ---------------------------------------------------------------------------
# WikiStore-level source primitives
# ---------------------------------------------------------------------------


def test_add_source_creates_registry_entry(tmp_path: Path) -> None:
    store = WikiStore.init(tmp_path / "w")
    src = tmp_path / "guide.md"
    src.write_text("# Guide\n\nbody\n", encoding="utf-8")

    entry = store.add_source(src, into_subdir="veterinary")
    rel = _rel(src, into="veterinary")
    assert entry.rel_path == rel
    assert len(entry.sha256) == 64
    assert entry.ingestions == []

    # File copied
    assert (store.sources_path / rel).exists()
    # Registry persisted
    assert (store.sources_path / REGISTRY_FILENAME).exists()
    # Re-load matches
    reg = SourceRegistry.load(store.sources_path)
    assert rel in reg.entries


def test_add_source_commits_with_source_subject(tmp_path: Path) -> None:
    store = WikiStore.init(tmp_path / "w")
    src = tmp_path / "x.md"
    src.write_text("body\n", encoding="utf-8")
    store.add_source(src)
    from outmem.git_ops import log_since

    rel = _rel(src)
    log = log_since(store.root)
    assert log[0].subject == f"source: {rel}"


def test_add_source_re_add_same_content_no_change(tmp_path: Path) -> None:
    store = WikiStore.init(tmp_path / "w")
    src = tmp_path / "x.md"
    src.write_text("body\n", encoding="utf-8")
    entry1 = store.add_source(src)
    # Second add of identical content: same hash dir → idempotent.
    entry2 = store.add_source(src)
    assert entry1.sha256 == entry2.sha256
    assert entry1.rel_path == entry2.rel_path
    assert entry1.registered_at == entry2.registered_at


def test_list_sources(tmp_path: Path) -> None:
    store = WikiStore.init(tmp_path / "w")
    files = []
    for name in ("alpha.md", "beta.md"):
        f = tmp_path / name
        f.write_text(f"content of {name}\n", encoding="utf-8")
        store.add_source(f)
        files.append(f)

    sources = store.list_sources()
    paths = {s.rel_path for s in sources}
    assert paths == {_rel(files[0]), _rel(files[1])}


def test_read_source_returns_content(tmp_path: Path) -> None:
    store = WikiStore.init(tmp_path / "w")
    src = tmp_path / "x.md"
    src.write_text("hello world\n", encoding="utf-8")
    entry = store.add_source(src)
    out = store.read_source(entry.rel_path)
    assert "hello world" in out


def test_read_source_truncates(tmp_path: Path) -> None:
    store = WikiStore.init(tmp_path / "w")
    src = tmp_path / "x.md"
    src.write_text("z" * 5000, encoding="utf-8")
    entry = store.add_source(src)
    out = store.read_source(entry.rel_path, max_chars=100)
    assert "truncated" in out


def test_record_ingestion(tmp_path: Path) -> None:
    store = WikiStore.init(tmp_path / "w")
    src = tmp_path / "x.md"
    src.write_text("body\n", encoding="utf-8")
    entry = store.add_source(src)
    record = store.record_ingestion(
        entry.rel_path,
        prompt="extract X",
        pages_touched=["page-x"],
    )
    assert record.prompt == "extract X"
    assert record.pages_touched == ("page-x",)

    # Persisted
    reg = SourceRegistry.load(store.sources_path)
    assert reg.entries[entry.rel_path].ingestions[0].prompt == "extract X"


def test_record_ingestion_unknown_source_raises(tmp_path: Path) -> None:
    store = WikiStore.init(tmp_path / "w")
    with pytest.raises(OutmemError, match="not registered"):
        store.record_ingestion("ghost.md", prompt="x", pages_touched=[])


def test_source_max_chars_from_config(tmp_path: Path) -> None:
    """``config.yaml`` sets the source cap; WikiStore picks it up."""
    root = tmp_path / "w"
    root.mkdir()
    (root / "config.yaml").write_text("sources:\n  max_chars: 50\n", encoding="utf-8")
    WikiStore.init(root)
    store = WikiStore.open(root)
    src = tmp_path / "x.md"
    src.write_text("a" * 200, encoding="utf-8")
    entry = store.add_source(src)
    out = store.read_source(entry.rel_path)
    # Cap of 50 → ~50 chars plus the "[truncated …]" footer.
    assert "truncated" in out
    assert len(out) < 200


# ---------------------------------------------------------------------------
# Agent-side tools (list_sources / read_source / record_ingestion)
# ---------------------------------------------------------------------------


def _by_name(tools: list, name: str):
    for t in tools:
        if t.__name__ == name:
            return t
    raise AssertionError(f"missing tool: {name}")


def test_adapter_list_sources_empty(tmp_path: Path) -> None:
    from outmem.adapters.pydantic_ai import wiki_tools

    store = WikiStore.init(tmp_path / "w")
    out = _by_name(wiki_tools(store), "list_sources")()
    assert "no sources registered" in out


def test_adapter_list_sources_after_add(tmp_path: Path) -> None:
    from outmem.adapters.pydantic_ai import wiki_tools

    store = WikiStore.init(tmp_path / "w")
    src = tmp_path / "x.md"
    src.write_text("body\n", encoding="utf-8")
    entry = store.add_source(src)
    out = _by_name(wiki_tools(store), "list_sources")()
    assert entry.rel_path in out
    assert "sha:" in out


def test_adapter_read_source(tmp_path: Path) -> None:
    from outmem.adapters.pydantic_ai import wiki_tools

    store = WikiStore.init(tmp_path / "w")
    src = tmp_path / "x.md"
    src.write_text("hello\n", encoding="utf-8")
    entry = store.add_source(src)
    out = _by_name(wiki_tools(store), "read_source")(rel_path=entry.rel_path)
    assert "hello" in out


def test_adapter_read_source_unknown(tmp_path: Path) -> None:
    from outmem.adapters.pydantic_ai import wiki_tools

    store = WikiStore.init(tmp_path / "w")
    out = _by_name(wiki_tools(store), "read_source")(rel_path="ghost.md")
    assert "read_source failed" in out


def test_adapter_record_ingestion(tmp_path: Path) -> None:
    from outmem.adapters.pydantic_ai import wiki_tools

    store = WikiStore.init(tmp_path / "w")
    src = tmp_path / "x.md"
    src.write_text("body\n", encoding="utf-8")
    entry = store.add_source(src)
    out = _by_name(wiki_tools(store), "record_ingestion")(
        rel_path=entry.rel_path,
        prompt="extract X",
        pages_touched=["page-x"],
    )
    assert "recorded ingestion" in out
    reg = SourceRegistry.load(store.sources_path)
    assert reg.entries[entry.rel_path].ingestions[0].prompt == "extract X"


# ---------------------------------------------------------------------------
# CLI: outmem ingest
# ---------------------------------------------------------------------------


def _model_that_calls(*calls: dict[str, object], reply: str = "done.") -> FunctionModel:
    state = {"step": 0}

    async def runner(messages: list[object], info: AgentInfo) -> ModelResponse:
        idx = state["step"]
        state["step"] = idx + 1
        if idx < len(calls):
            entry = calls[idx]
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name=str(entry["tool"]),
                        args=dict(entry["args"]),  # type: ignore[arg-type]
                    )
                ]
            )
        return ModelResponse(parts=[TextPart(content=reply)])

    return FunctionModel(runner)


def test_cli_ingest_register_only(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from outmem.cli.__main__ import main

    store = WikiStore.init(tmp_path / "w")
    src = tmp_path / "guide.md"
    src.write_text("# Drugs for cats\nDosage: 5mg/kg.\n", encoding="utf-8")

    rc = main(
        [
            "--root",
            str(store.root),
            "ingest",
            str(src),
            "--into",
            "veterinary",
            "--register-only",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    rel = _rel(src, into="veterinary")
    assert f"registered {rel}" in out
    # Source file copied, registered, committed.
    assert (store.sources_path / rel).exists()
    reg = SourceRegistry.load(store.sources_path)
    assert rel in reg.entries
    # No ingestion recorded
    assert reg.entries[rel].ingestions == []


def test_cli_ingest_with_agent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Full ingestion run: register + agent writes a page + ingestion recorded."""
    from outmem.agent import service as svc
    from outmem.cli.__main__ import main

    store = WikiStore.init(tmp_path / "w")
    src = tmp_path / "drugs.md"
    src.write_text(
        "# Drugs\nCat dose: 5mg/kg.\nDog dose: 10mg/kg.\n",
        encoding="utf-8",
    )

    # The model fires a write_page (which produces a `compact: cat-doses`
    # commit) and then replies.
    model = _model_that_calls(
        {
            "tool": "write_page",
            "args": {
                "slug": "cat-doses",
                "title": "Cat doses",
                "body": "Cat dose: 5mg/kg per the drugs source.\n",
                "provenance": ["sources/drugs.md"],
            },
        },
        reply="wrote cat dosage page.",
    )

    real_build = svc.build_agent

    def building(store, **kwargs):  # type: ignore[no-untyped-def]
        kwargs.pop("model", None)
        return real_build(store, model=model, **kwargs)

    monkeypatch.setattr(svc, "build_agent", building)

    rc = main(
        [
            "--root",
            str(store.root),
            "ingest",
            str(src),
            "--prompt",
            "extract dosages for cats",
            "--no-push",
            "--no-record",
            "--quiet",
        ]
    )
    assert rc == 0

    # Agent's write produced wiki/pages/cat-doses.md.
    assert (store.pages_path / "cat-doses.md").exists()

    # Ingestion was recorded against the source.
    reg = SourceRegistry.load(store.sources_path)
    drugs = reg.entries[_rel(src)]
    assert len(drugs.ingestions) == 1
    assert drugs.ingestions[0].prompt == "extract dosages for cats"
    assert "cat-doses" in drugs.ingestions[0].pages_touched


def test_add_source_same_name_different_content_no_collision(tmp_path: Path) -> None:
    """Two files with the same basename but different content land in
    distinct hash dirs and both register cleanly — the bug behind the
    user's amikacin / aztreonam ingest collision."""
    store = WikiStore.init(tmp_path / "w")

    a = tmp_path / "amikacin-document.md"
    a.write_text("amikacin fachinfo\n", encoding="utf-8")
    b = tmp_path / "aztreonam-document.md"
    b.write_text("aztreonam fachinfo\n", encoding="utf-8")

    ea = store.add_source(a, into_subdir="abx", rename="document.md")
    eb = store.add_source(b, into_subdir="abx", rename="document.md")

    assert ea.rel_path != eb.rel_path
    assert ea.sha256 != eb.sha256
    reg = SourceRegistry.load(store.sources_path)
    assert ea.rel_path in reg.entries
    assert eb.rel_path in reg.entries


