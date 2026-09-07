"""Exceptions raised by outmem.

The hierarchy is shallow on purpose: callers usually want to either retry
the whole operation, surface the error to a human, or treat it as a hard
fault. Fine-grained discrimination is rarely useful — the type is mostly
documentation about *where* in the pipeline a failure originated.
"""

from __future__ import annotations


class OutmemError(Exception):
    """Base class for all outmem exceptions."""


class FrontmatterError(OutmemError):
    """Wiki page frontmatter is missing, malformed, or fails validation."""


class SlugError(OutmemError):
    """A slug is empty, contains unsafe characters, or fails to resolve."""


class GitOperationError(OutmemError):
    """A git subprocess returned non-zero or produced unexpected output."""


class WritebackError(OutmemError):
    """A writeback (commit + push) failed after the configured retries.

    Raised by the runtime when ``git push`` is rejected a second time or
    when a rebase conflict cannot be resolved automatically (spec v0.5 §9).
    Treated as a hard fault — callers must not respond to the user as if
    writeback had succeeded.
    """


class ConflictError(OutmemError):
    """A merge or rebase conflict requires human resolution."""


class IncompleteBodyError(OutmemError):
    """A page body stops early — it ends at an elision marker.

    The one content-level refusal in a write path otherwise concerned
    with structure. It exists because the alternative failure is silent:
    a body truncated under output-budget pressure is schema-valid, so
    without this the call commits and nothing downstream can tell the
    page apart from a complete one.

    Carries ``markers`` (the offending lines) so a caller can quote them,
    and the adapter turns it into a ``ModelRetry`` — the model still has
    the source in context at that moment, which is the only point where
    recovery is cheap.
    """

    def __init__(self, message: str, *, markers: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.markers = markers


class RestrictionError(OutmemError):
    """A write or privileged operation is refused by the access rules.

    Raised at the **store** layer, so the CLI, the Python API, and a
    downstream app driving its own agent are all covered — not only
    outmem's tool palette.

    Only *writes* and operator-only paths raise it. A denied **read**
    raises or returns exactly what a nonexistent item would, because a
    distinguishable error is an existence oracle: learning that
    something is hidden is learning that it is there. Message text must
    therefore never name a hidden item.
    """


class IdentityWarning(OutmemError):
    """A git author was not found in ``CONTRIBUTORS.md``.

    Raised as a warning when the runtime wants strict identity resolution
    (e.g. tests). At runtime the steering loop normally logs and continues,
    treating unknown authors as a degraded steering signal rather than a
    fault (spec v0.5 §3).
    """


def format_validation_detail(exc: BaseException) -> str:
    """Walk ``__cause__`` for a pydantic ``ValidationError`` and summarise it.

    PydanticAI raises ``UnexpectedModelBehavior`` with the underlying
    ``pydantic_core.ValidationError`` chained via ``__cause__``. The
    validation error carries the actual reason the model's tool call was
    rejected — surface a one-liner per error so the user knows whether
    ``body`` was missing, ``tags`` had the wrong type, etc.

    Returns an empty string when no validation detail is available (the
    failure was something other than tool-arg validation). Cycle-safe:
    pathological cause chains (A → B → A) terminate via a ``seen`` set.
    """
    from pydantic_core import ValidationError

    seen: set[int] = set()
    cursor: BaseException | None = exc
    while cursor is not None and id(cursor) not in seen:
        seen.add(id(cursor))
        if isinstance(cursor, ValidationError):
            entries = []
            for err in cursor.errors(include_url=False, include_context=False)[:5]:
                loc = ".".join(str(p) for p in err.get("loc", ()))
                msg = err.get("msg", "")
                entries.append(f"{loc}: {msg}" if loc else msg)
            if entries:
                return " — validation errors: " + "; ".join(entries)
            return ""
        cursor = cursor.__cause__
    return ""
