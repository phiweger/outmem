"""Restriction labels, grants, and the predicates that decide access.

This module is the whole *policy*. It holds no state, touches no disk,
and knows nothing about pages, sources, or git — it answers three
questions and nothing else:

- Is an item with these labels visible in this mode?  (:func:`visible`)
- May this mode write an item with these labels?      (:func:`writable`)
- Which labels does a path rule assign?               (:class:`RestrictedSettings`)

Keeping it separate from :mod:`outmem.store` is what makes the rules
testable in isolation, and what lets the enforcement code in the store
be a one-line call at each of the places it is needed rather than a
policy re-derived per method.

See ``specs/restricted-content.md`` for the model. The short version:

**Open by default.** An item carries a set of labels; the empty set
means open. A *mode* is the set of labels a session is working in. An
item is visible iff ``labels ⊆ mode``, so open content (``∅``) is
visible in every mode without needing a special case.

**Flat labels.** No hierarchy, no ordering, no implication. ``hr`` does
not contain ``hr-payroll``; they are two unrelated strings.

**Fail closed.** Every path that cannot determine an item's labels
resolves them to :data:`DENY`, a sentinel that is deliberately not a
legal label — so it can never be declared, never granted, never in any
mode, and ``labels ⊆ mode`` is therefore false for every mode. The
fail-closed case needs no branch in the enforcement code; it falls out
of the same subset test as everything else.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from typing import Any

from outmem.exceptions import OutmemError

__all__ = [
    "DENY",
    "DENY_SET",
    "MODE_SEPARATOR",
    "UNRESTRICTED_GRANTS",
    "Grants",
    "LabelError",
    "RestrictedSettings",
    "mode_dirname",
    "mode_from_dirname",
    "normalise_labels",
    "settings_from_dict",
    "validate_label",
    "visible",
    "writable",
]


class LabelError(OutmemError):
    """A restriction label is malformed, or a grant set is incoherent.

    Raised at the *entry* points — config load, ``--restricted`` on the
    CLI, grant construction, view construction — never during a read.
    A read that cannot resolve labels denies silently (§9: a
    distinguishable error is an existence oracle); this error is for the
    operator holding the document, at the moment they are labelling it.
    """


# Label grammar: one segment of the slug grammar (lowercase ASCII
# alphanumerics, single hyphens, no leading/trailing hyphen).
#
# Borrowed deliberately rather than invented. Labels appear in path
# rules next to slugs, in CLI flags, and in per-mode log directory names
# (§8.2) — one grammar for all three means a label is always a safe path
# component and always renders the same way. It also makes ``HR`` a hard
# error at the point someone types it, instead of a second label nobody
# holds and content nobody can see.
_LABEL_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

# The fail-closed sentinel. Not a legal label (``\x00`` can never pass
# ``_LABEL_RE``), so it can never be declared in config, never granted,
# and never appear in a mode. Any item resolved to this set is therefore
# invisible to everyone, including the operator — which is the intent:
# an item whose labels cannot be read is not an item whose labels are
# empty.
DENY = "\x00outmem:deny"

#: Convenience: the label set that denies an item to every mode.
DENY_SET = frozenset({DENY})


def validate_label(label: Any) -> str:
    """Return ``label`` normalised, or raise :class:`LabelError`.

    Normalisation is whitespace-stripping only. Case is *not* folded:
    ``HR`` is rejected rather than quietly rewritten to ``hr``, because
    silently mapping one label onto another is the one direction that
    can widen access.
    """
    if not isinstance(label, str):
        raise LabelError(
            f"restriction label must be a string, got {type(label).__name__}."
        )
    stripped = label.strip()
    if not stripped:
        raise LabelError("restriction label is empty.")
    if not _LABEL_RE.match(stripped):
        raise LabelError(
            f"restriction label {label!r} is not valid: lowercase ASCII "
            "alphanumerics with single hyphens only, no leading or trailing "
            "hyphen (e.g. 'hr', 'board', 'legal-privileged')."
        )
    return stripped


def normalise_labels(labels: Iterable[Any] | None) -> frozenset[str]:
    """Validate and de-duplicate an iterable of labels.

    ``None`` and the empty iterable both mean *open*.
    """
    if labels is None:
        return frozenset()
    if isinstance(labels, str):
        # A bare string is almost always a mistake that would otherwise
        # iterate character-by-character into a set of one-letter labels.
        raise LabelError(
            f"expected a list of restriction labels, got the string {labels!r} "
            "— pass ['" + labels + "'] instead."
        )
    return frozenset(validate_label(item) for item in labels)


# ---------------------------------------------------------------------------
# Grants — what a user is entitled to, independent of any one session
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Grants:
    """What a user may do with each label. Supplied by the application.

    outmem does not model users, sessions, or identity — the calling
    application authenticates the person and translates whatever it
    knows (an LDAP group, a JWT claim, a row in its own database) into
    one of these.

    Three nested sets, each a subset of the one before it:

    ``read``
        May see content carrying the label. A label absent from ``read``
        does not exist as far as this user is concerned.
    ``write``
        May create and modify it. Must be a subset of ``read``:
        ``extend_page`` replaces a whole body and ``append_page`` has to
        avoid duplicating what is already there, so neither is safe
        without reading first.
    ``declassify``
        May *remove* a label, or rename a page in a way that changes its
        effective labels (§8.3). Must be a subset of ``write``, and is
        never exposed through a model-facing tool.

    The nesting is enforced at construction rather than checked at use,
    so an incoherent grant is a loud failure in the application's
    authentication code and not a quiet widening at a call site.
    """

    read: frozenset[str] = frozenset()
    write: frozenset[str] = frozenset()
    declassify: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(self, "read", normalise_labels(self.read))
        object.__setattr__(self, "write", normalise_labels(self.write))
        object.__setattr__(self, "declassify", normalise_labels(self.declassify))
        if not self.write <= self.read:
            raise LabelError(
                "write without read is not a valid grant "
                f"({sorted(self.write - self.read)}): a write tool replaces or "
                "extends an existing body, which is unsafe without being able "
                "to see it."
            )
        if not self.declassify <= self.write:
            raise LabelError(
                "declassify without write is not a valid grant "
                f"({sorted(self.declassify - self.write)}): removing a label "
                "modifies the item."
            )

    @classmethod
    def none(cls) -> Grants:
        """A user entitled to open content only."""
        return cls()

    @classmethod
    def reader(cls, *labels: str) -> Grants:
        """Read-only entitlement to ``labels``."""
        return cls(read=frozenset(labels))

    @classmethod
    def writer(cls, *labels: str) -> Grants:
        """Read and write entitlement to ``labels`` (no declassify)."""
        return cls(read=frozenset(labels), write=frozenset(labels))

    def may_read(self, labels: Iterable[str]) -> bool:
        """True if every label in ``labels`` is readable by this user."""
        return frozenset(labels) <= self.read

    def may_write(self, labels: Iterable[str]) -> bool:
        return frozenset(labels) <= self.write

    def may_declassify(self, labels: Iterable[str]) -> bool:
        return frozenset(labels) <= self.declassify


#: The entitlement of a caller with no access control in play at all —
#: the server-side operator holding the bare store.
UNRESTRICTED_GRANTS = Grants()


# ---------------------------------------------------------------------------
# The two rules
# ---------------------------------------------------------------------------


def visible(labels: Iterable[str], mode: Iterable[str]) -> bool:
    """§3.1 — an item is visible in ``mode`` iff ``labels ⊆ mode``.

    Open content (``labels = ∅``) is visible in every mode because
    ``∅ ⊆ S`` for all ``S``; open-by-default is not a special case.

    Note the direction: mode ``{hr}`` sees open content *and* HR
    content, not HR alone. An HR-only session would be stricter than
    necessary — the unsafe direction is read-HR-then-write-open — and
    would leave the agent composing HR pages with no access to the
    company glossary.
    """
    return frozenset(labels) <= frozenset(mode)


def writable(labels: Iterable[str], mode: Iterable[str], grants: Grants) -> bool:
    """§3.2 — a write may target an item iff ``labels == mode``, with
    ``write`` held on every label in ``mode``.

    Equality, not containment, forced from both directions:

    - ``labels ⊆ mode`` — you must be able to see what you are modifying.
    - ``labels ⊇ mode`` — you must not carry facts out of a more
      restricted context into a less restricted item (no write-down).
    """
    label_set = frozenset(labels)
    mode_set = frozenset(mode)
    return label_set == mode_set and grants.may_write(mode_set)


# ---------------------------------------------------------------------------
# Per-mode log partitioning
# ---------------------------------------------------------------------------

#: Joins several labels into one directory name. Not in the label
#: grammar, so ``hr+legal`` can only ever mean the pair — a label
#: containing the separator is impossible by construction.
MODE_SEPARATOR = "+"


def mode_dirname(mode: Iterable[str]) -> str:
    """Directory under ``log/`` that a session in ``mode`` writes to.

    The open mode maps to ``""`` — ``log/<date>.md``, exactly where logs
    have always gone — so a wiki with no restrictions has no new
    directory level and nothing to migrate.

    Partitioning is not cosmetic. ``append_log`` writes to an open file,
    and mandatory writeback actively pushes an agent there when nothing
    else was warranted; in a restricted session that is a write-down of
    whatever the agent was just reading.
    """
    return MODE_SEPARATOR.join(sorted(mode))


def mode_from_dirname(name: str) -> frozenset[str] | None:
    """Parse a ``log/`` subdirectory name back into a label set.

    Returns ``None`` for a directory that is not a mode partition at all
    (someone's ``log/archive/``), which the caller should treat as open
    — it was not written by this mechanism. A name that *looks* like a
    partition but holds an invalid label returns :data:`DENY_SET`,
    because a directory whose audience cannot be determined must not be
    shown to an audience.
    """
    if not name:
        return frozenset()
    parts = name.split(MODE_SEPARATOR)
    if len(parts) == 1 and not _LABEL_RE.match(name):
        return None
    try:
        return normalise_labels(parts)
    except LabelError:
        return DENY_SET


# ---------------------------------------------------------------------------
# Configuration — the declared label set and the path safety nets
# ---------------------------------------------------------------------------


def _compile_rules(block: Any, *, what: str) -> dict[str, frozenset[str]]:
    """Parse a ``{pattern: [labels]}`` mapping from config.

    Unlike the rest of ``config.yaml``, a malformed rule here is *not*
    forgiven into a default. The forgiving-load contract exists so a
    typo cannot brick a wiki, but the failure mode differs: a dropped
    ``retrieval.strategy`` degrades search, whereas a dropped path rule
    silently publishes a namespace that was meant to be restricted.
    Raising is the fail-closed choice.
    """
    if block is None:
        return {}
    if not isinstance(block, Mapping):
        raise LabelError(
            f"restricted.{what} must be a mapping of pattern → [labels], got "
            f"{type(block).__name__}."
        )
    rules: dict[str, frozenset[str]] = {}
    for pattern, labels in block.items():
        if not isinstance(pattern, str) or not pattern.strip():
            raise LabelError(f"restricted.{what}: pattern {pattern!r} is not a string.")
        resolved = normalise_labels(labels)
        if not resolved:
            raise LabelError(
                f"restricted.{what}: pattern {pattern!r} assigns no labels. "
                "A rule that restricts nothing is almost certainly a mistake; "
                "delete it if it is intentional."
            )
        rules[pattern.strip()] = resolved
    return rules


def _matches(pattern: str, candidate: str) -> bool:
    """Glob match with one deliberate extension.

    A pattern ending in a separator plus ``*`` also matches the bare
    prefix: ``hr:*`` covers the namespace root page ``hr``, and ``hr/*``
    covers the source directory ``hr`` itself. Plain ``fnmatch`` would
    leave those two uncovered, and an uncovered item is an *open* item —
    the unsafe direction. "Everything under hr" is what the rule means,
    so that is what it does.
    """
    if fnmatchcase(candidate, pattern):
        return True
    for sep in (":", "/"):
        suffix = sep + "*"
        if pattern.endswith(suffix) and candidate == pattern[: -len(suffix)]:
            return True
    return False


@dataclass
class RestrictedSettings:
    """The ``restricted:`` block of ``config.yaml``.

    ::

        restricted:
          labels: [hr, legal, board]   # the declared set
          paths:                       # safety net for pages
            "hr:*": [hr]
          sources:                     # safety net for ingested documents
            "hr/*": [hr]

    ``labels`` is the whole declared vocabulary. A label used anywhere
    but absent from it is denied to every mode and reported by lint
    (§5.2, §11) — a label nobody can hold makes content invisible to
    everyone, silently, and that is exactly the failure the declaration
    exists to catch.

    ``paths`` and ``sources`` are *safety nets*, not the primary
    mechanism. Explicit ``restricted:`` frontmatter is primary, because
    forcing restriction onto the slug namespace would put taxonomy and
    security on the same axis and mean reorganising a wiki to secure it.
    Path rules exist because explicit labels can be forgotten: a new
    page under ``hr:`` is restricted whether or not anyone remembered.
    Both are needed; labels resolve to the union of the two.
    """

    labels: frozenset[str] = frozenset()
    paths: dict[str, frozenset[str]] = field(default_factory=dict)
    sources: dict[str, frozenset[str]] = field(default_factory=dict)

    @property
    def enabled(self) -> bool:
        """True once this wiki declares any label at all.

        The switch that decides whether outmem has restricted content to
        protect. A wiki that declares none behaves exactly as it did
        before this feature existed.
        """
        return bool(self.labels)

    def labels_for_slug(self, slug: str) -> frozenset[str]:
        """Labels assigned to ``slug`` by path rules (union of all matches)."""
        return self._match(self.paths, slug)

    def labels_for_source(self, rel_path: str) -> frozenset[str]:
        """Labels assigned to a source by path rules.

        ``rel_path`` is matched as stored in the registry — the
        ``--into`` subdirectory plus filename, without the
        ``sources/`` / ``sources-local/`` tree prefix, so one rule
        covers a directory in either tree.
        """
        return self._match(self.sources, rel_path)

    @staticmethod
    def _match(
        rules: Mapping[str, frozenset[str]], candidate: str
    ) -> frozenset[str]:
        matched: set[str] = set()
        for pattern, labels in rules.items():
            if _matches(pattern, candidate):
                matched |= labels
        return frozenset(matched)

    def undeclared(self, labels: Iterable[str]) -> frozenset[str]:
        """Which of ``labels`` are not in the declared set.

        Used by the fail-closed resolver and by lint. ``DENY`` is
        excluded: it is already the fail-closed outcome, and reporting
        it as "undeclared" would turn one finding into two.
        """
        return frozenset(labels) - self.labels - {DENY}

    def check_declared(self, labels: Iterable[str], *, what: str = "label") -> None:
        """Raise :class:`LabelError` for any label outside the declared set.

        Called at entry points where a human is choosing a label and can
        fix a typo immediately — ``outmem ingest --restricted``, view
        construction, the ``restrict`` verb. Reads never call this; they
        fail closed and silent instead (§9).
        """
        unknown = self.undeclared(labels)
        if unknown:
            declared = ", ".join(sorted(self.labels)) or "(none declared)"
            raise LabelError(
                f"unknown restriction {what}: {', '.join(sorted(unknown))}. "
                f"Declared labels are: {declared}. Add it under "
                "`restricted.labels` in config.yaml first — an undeclared "
                "label hides content from everyone, including you."
            )

    def resolve(self, labels: Iterable[str]) -> frozenset[str]:
        """Fail-closed label resolution for content read off disk.

        Any label outside the declared set collapses the whole set to
        :data:`DENY_SET`. Keeping the undeclared label as-is would work
        by accident (nobody holds it, so nothing sees it), but only for
        as long as nobody adds it to ``restricted.labels`` later and
        retroactively publishes the item to that compartment's holders.
        """
        resolved = frozenset(labels)
        if not resolved:
            return frozenset()
        if self.undeclared(resolved):
            return DENY_SET
        return resolved


def settings_from_dict(block: Any) -> RestrictedSettings:
    """Build :class:`RestrictedSettings` from the raw ``restricted:`` block."""
    if block is None:
        return RestrictedSettings()
    if not isinstance(block, Mapping):
        raise LabelError(
            f"`restricted` must be a mapping, got {type(block).__name__}."
        )
    declared = normalise_labels(block.get("labels"))
    settings = RestrictedSettings(
        labels=declared,
        paths=_compile_rules(block.get("paths"), what="paths"),
        sources=_compile_rules(block.get("sources"), what="sources"),
    )
    # A path rule naming a label nobody declared would restrict a whole
    # namespace to a compartment that cannot exist — content invisible to
    # everyone, created by a rule that looks correct. Catch it at load.
    for what, rules in (("paths", settings.paths), ("sources", settings.sources)):
        for pattern, labels in rules.items():
            settings.check_declared(labels, what=f"label in {what} rule {pattern!r}")
    return settings
