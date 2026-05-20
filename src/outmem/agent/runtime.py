"""Build a PydanticAI agent bound to a :class:`WikiStore`.

The agent's tool palette comes from :mod:`outmem.adapters.pydantic_ai`;
its system prompt is rendered from ``prompts/system.j2`` (the
planning-prompt distilled into instructions). The model is supplied at
construction time — typically from ``OUTMEM_MODEL`` env — so the same
runtime serves any PydanticAI-supported provider.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from outmem.adapters.pydantic_ai import wiki_tools
from outmem.exceptions import OutmemError
from outmem.skills import bundled_registry
from outmem.store import WikiStore

if TYPE_CHECKING:
    from pydantic_ai import Agent

DEFAULT_MODEL_ENV = "OUTMEM_MODEL"
DEFAULT_PROMPT_NAME = "system"

# Skills the runtime injects into the system prompt by default. The
# bundled SKILL.md files under src/outmem/skills/notes/ are the single
# source of truth — rendered verbatim into the prompt here and also
# usable from an external PydanticAI agent via
# :func:`outmem.adapters.pydantic_ai.skill_text`.
DEFAULT_INJECTED_SKILLS: tuple[str, ...] = ("search", "evolution", "write")

_TEMPLATES_DIR = Path(__file__).resolve().parent / "prompts"


def _jinja_env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=False,  # the system prompt is plain text, not HTML
        undefined=StrictUndefined,
    )


def render_system_prompt(
    store: WikiStore,
    *,
    prompt_name: str = DEFAULT_PROMPT_NAME,
    include_steering: bool = True,
    inject_skills: tuple[str, ...] | None = None,
) -> str:
    """Render ``prompts/<prompt_name>.j2`` into the system prompt.

    Variables exposed to the template:

    - ``wiki_root`` — absolute path to the wiki on disk
    - ``agent_name`` / ``agent_email`` — the identity outmem commits under
    - ``recent_human_commits`` — phase-1 steering signal (list of
      :class:`outmem.git_ops.CommitInfo`), if ``include_steering=True``
      and a last-run marker exists; otherwise an empty list
    - ``semantic_enabled`` — bool, true if the wiki has the vector
      index turned on (controls a small "duplicate check" paragraph
      in the prompt)
    - ``agents_md`` — the body of ``wiki/AGENTS.md`` (user-editable
      wiki-conventions doc) if present, else ``None``. Injected as a
      section so users can teach the agent domain rules without
      having to fork the runtime prompt.
    - ``skills`` — list of ``(skill_name, skill_body)`` tuples
      injected verbatim into a `# Tool reference` appendix. Defaults
      to :data:`DEFAULT_INJECTED_SKILLS`; pass ``inject_skills=()``
      to suppress, or a custom tuple to override.

    The steering signal is rendered into the prompt itself rather than
    injected as a tool call — phase 1 happens before the model can
    plan, which means it has to be in the system prompt from the
    moment the agent boots.

    Skill bodies come from :mod:`outmem.skills` — the bundled
    SKILL.md files under ``src/outmem/skills/notes/``. Rendering
    them into the prompt here keeps a single source of truth for
    tool-usage docs that external PydanticAI agents can also splice
    in via :func:`outmem.adapters.pydantic_ai.skill_text`.
    """
    env = _jinja_env()
    template = env.get_template(f"{prompt_name}.j2")
    recent_human_commits: list[Any] = []
    if include_steering:
        recent_human_commits = store.steering()
    skill_names = inject_skills if inject_skills is not None else DEFAULT_INJECTED_SKILLS
    skills = _load_injected_skills(skill_names)
    return template.render(
        wiki_root=str(store.root),
        agent_name=store.config.agent_identity.name,
        agent_email=store.config.agent_identity.email,
        recent_human_commits=recent_human_commits,
        semantic_enabled=store.semantic_enabled(),
        agents_md=store.read_agents_md(),
        skills=skills,
    )


def _load_injected_skills(
    names: tuple[str, ...],
) -> list[tuple[str, str]]:
    """Load each named skill's body via :class:`outskilled.SkillRegistry`.

    Missing skills are skipped silently — a deployment that doesn't
    ship one of the defaults shouldn't break the prompt render.
    """
    from outskilled import UnknownSkillError  # type: ignore[import-untyped]

    registry = bundled_registry()
    out: list[tuple[str, str]] = []
    for name in names:
        try:
            body: str = registry.load(name)
        except UnknownSkillError:
            continue
        out.append((name, body.strip()))
    return out


def _resolve_model(model: Any | None, store: WikiStore | None = None) -> Any:
    """Pick the PydanticAI model, in priority order.

    1. Explicit ``model`` argument
    2. ``$OUTMEM_MODEL`` environment variable
    3. ``model:`` field from the store's ``config.yaml`` (if a store was supplied)
    4. Error — the user has to be explicit

    The ``.env`` file at the wiki root has already been loaded by
    :meth:`WikiStore.open`, so env vars set there appear in step 2.
    """
    if model is not None:
        return model
    env = os.environ.get(DEFAULT_MODEL_ENV)
    if env:
        return env
    if store is not None and store.config.outmem.model:
        return store.config.outmem.model
    raise OutmemError(
        "No model specified. Pass `model=...`, set "
        f"${DEFAULT_MODEL_ENV} (e.g. 'anthropic:claude-sonnet-4-6'), "
        "or add `model:` to config.yaml."
    )


_APPROVAL_GATED_TOOLS = frozenset({"write_page", "extend_page"})

# How many times PydanticAI retries a tool call that failed schema
# validation. PydanticAI's default is 1, which means a single missing-arg
# slip from the model (e.g. forgetting `body=` on a `write_page` call)
# blows up the whole agent run. 5 gives the model meaningful room to
# self-correct on multi-page ingests without burning unbounded tokens.
# When the retry budget IS exhausted, the wrapped WritebackError now
# surfaces the underlying validation errors (see :func:`service._format_validation_detail`).
DEFAULT_TOOL_RETRIES = 5
DEFAULT_OUTPUT_RETRIES = 3

# PydanticAI's Anthropic adapter defaults `max_tokens=4096` when nothing
# is specified. That's the model's TOTAL output budget for one turn —
# tool-call JSON + thinking + everything. A typical compacted wiki page
# body lands at 5-8k chars (~ 1.5-2k tokens); combined with the model's
# reasoning, 4096 tokens regularly runs out mid-tool-call, the response
# gets truncated, and PydanticAI sees a write_page call missing its
# `body` argument — surfaced to the user as "body: Field required" with
# no obvious cause. 16k is a comfortable headroom that still bounds
# token cost.
DEFAULT_MAX_TOKENS = 16384


def build_agent(
    store: WikiStore,
    *,
    model: Any | None = None,
    prompt_name: str = DEFAULT_PROMPT_NAME,
    include_steering: bool = True,
    **agent_kwargs: Any,
) -> Agent[None, Any]:
    """Construct a :class:`pydantic_ai.Agent` configured for outmem.

    ``model`` accepts anything :class:`pydantic_ai.Agent` accepts — a
    string ID (``"anthropic:claude-sonnet-4-6"``), a :class:`Model`
    instance, or :class:`TestModel` for testing. When ``None``, falls
    back to ``$OUTMEM_MODEL``.

    ``include_steering=True`` (the default) renders the recent human
    commits into the system prompt, so phase 1 of the planning prompt
    has its inputs before the model picks a tool. Set ``False`` to
    omit — useful for tests that don't want the steering noise.

    When ``store.config.outmem.approval.required_for_writes`` is
    ``True``, the ``write_page`` and ``extend_page`` tools are added to
    a :class:`pydantic_ai.toolsets.FunctionToolset` with
    ``requires_approval=True`` so the agent's run yields a
    :class:`pydantic_ai.tools.DeferredToolRequests` instead of
    committing. The orchestrator (:func:`outmem.agent.ask`) resumes
    the run with a :class:`pydantic_ai.tools.DeferredToolResults` once
    the human reviewer has spoken — see
    :mod:`outmem.agent.approval`. ``append_log`` and the read-only
    tools are never gated so the agent can still satisfy mandatory
    writeback (spec §9) after a denial.

    Additional ``agent_kwargs`` flow through to ``Agent(...)``.
    """
    # Imported lazily so the optional `pydantic_ai` dep doesn't make
    # the module ungettable when the user only wants `render_system_prompt`.
    from pydantic_ai import Agent

    resolved_model = _resolve_model(model, store)
    system_prompt = render_system_prompt(
        store,
        prompt_name=prompt_name,
        include_steering=include_steering,
    )

    # Defaults the caller can override via agent_kwargs (e.g. tests
    # pin tool_retries=0 to keep failures crisp).
    agent_kwargs.setdefault("tool_retries", DEFAULT_TOOL_RETRIES)
    agent_kwargs.setdefault("output_retries", DEFAULT_OUTPUT_RETRIES)
    existing_settings: dict[str, Any] = agent_kwargs.get("model_settings") or {}
    merged_settings: dict[str, Any] = {**existing_settings}
    merged_settings.setdefault("max_tokens", DEFAULT_MAX_TOKENS)
    # Anthropic prompt-caching knobs. The model_settings dict is shared
    # across providers; these keys are no-ops on non-Anthropic models
    # (silently ignored). For Anthropic, they cut the bill by ~5-10x
    # on multi-turn ingest runs where the system prompt + tool defs +
    # long tool results would otherwise be re-shipped on every chat.
    for key in (
        # Top-level auto-cache, moves the breakpoint with the conversation.
        "anthropic_cache",
        # Caches the system prompt block.
        "anthropic_cache_instructions",
        # Caches the tool-def array.
        "anthropic_cache_tool_definitions",
    ):
        merged_settings.setdefault(key, True)
    agent_kwargs["model_settings"] = merged_settings

    tools = wiki_tools(store)
    if not store.config.outmem.approval.required_for_writes:
        return Agent(
            resolved_model,
            tools=tools,
            system_prompt=system_prompt,
            **agent_kwargs,
        )

    from pydantic_ai.tools import DeferredToolRequests
    from pydantic_ai.toolsets import FunctionToolset

    toolset: FunctionToolset[None] = FunctionToolset()
    for fn in tools:
        toolset.add_function(
            fn,
            requires_approval=fn.__name__ in _APPROVAL_GATED_TOOLS,
        )
    # output_type accepts a list-union so the run can yield either a
    # final string or a DeferredToolRequests (when an approval-gated
    # call is pending).
    return Agent(
        resolved_model,
        toolsets=[toolset],
        output_type=[str, DeferredToolRequests],
        system_prompt=system_prompt,
        **agent_kwargs,
    )
