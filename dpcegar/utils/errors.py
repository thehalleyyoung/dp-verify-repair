"""Custom exception hierarchy for DP-CEGAR.

Provides structured, informative error types for every stage of the
verification and repair pipeline.  Each exception carries optional
``details`` and ``source_loc`` fields so that error reporters can
produce rich diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Source location (shared by IR nodes and error messages)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SourceLoc:
    """Source location within a mechanism file.

    Attributes:
        file: Path to the source file (may be ``"<stdin>"``).
        line: 1-based line number.
        col:  1-based column number.
        end_line: Optional end line for multi-line spans.
        end_col:  Optional end column.
    """

    file: str = "<unknown>"
    line: int = 0
    col: int = 0
    end_line: int | None = None
    end_col: int | None = None

    def __str__(self) -> str:
        base = f"{self.file}:{self.line}:{self.col}"
        if self.end_line is not None:
            base += f"-{self.end_line}:{self.end_col}"
        return base


# ---------------------------------------------------------------------------
# Base exception
# ---------------------------------------------------------------------------

class DPCegarError(Exception):
    """Root of the DP-CEGAR exception hierarchy.

    All domain-specific exceptions inherit from this class so that callers
    can catch ``DPCegarError`` to handle any pipeline error generically.

    Attributes:
        message:    Human-readable error description.
        details:    Optional structured payload for programmatic consumers.
        source_loc: Optional source location where the error originated.
    """

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
        source_loc: SourceLoc | None = None,
    ) -> None:
        self.message = message
        self.details = details or {}
        self.source_loc = source_loc
        loc_prefix = f"{source_loc}: " if source_loc else ""
        super().__init__(f"{loc_prefix}{message}")

    def with_loc(self, loc: SourceLoc) -> DPCegarError:
        """Return a copy of this error with an attached source location."""
        return self.__class__(
            self.message,
            details=self.details,
            source_loc=loc,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise the error to a JSON-friendly dictionary."""
        d: dict[str, Any] = {
            "error_type": type(self).__name__,
            "message": self.message,
        }
        if self.details:
            d["details"] = self.details
        if self.source_loc:
            d["source_loc"] = str(self.source_loc)
        return d


# ---------------------------------------------------------------------------
# Parsing errors
# ---------------------------------------------------------------------------

class ParseError(DPCegarError):
    """Raised when the front-end parser encounters invalid syntax.

    Attributes:
        token: The offending token text (if available).
    """

    def __init__(
        self,
        message: str,
        *,
        token: str | None = None,
        details: dict[str, Any] | None = None,
        source_loc: SourceLoc | None = None,
    ) -> None:
        details = dict(details or {})
        if token is not None:
            details["token"] = token
        super().__init__(message, details=details, source_loc=source_loc)
        self.token = token


class LexError(ParseError):
    """Raised when the lexer cannot tokenise input."""


class SyntaxError_(ParseError):
    """Raised for grammar-level syntax errors (named with trailing
    underscore to avoid shadowing the built-in)."""


# ---------------------------------------------------------------------------
# Type-checking errors
# ---------------------------------------------------------------------------

class TypeCheckError(DPCegarError):
    """Raised when IR type-checking fails.

    Attributes:
        expected: The expected type descriptor.
        actual:   The actual type descriptor found.
    """

    def __init__(
        self,
        message: str,
        *,
        expected: str | None = None,
        actual: str | None = None,
        details: dict[str, Any] | None = None,
        source_loc: SourceLoc | None = None,
    ) -> None:
        details = dict(details or {})
        if expected is not None:
            details["expected"] = expected
        if actual is not None:
            details["actual"] = actual
        super().__init__(message, details=details, source_loc=source_loc)
        self.expected = expected
        self.actual = actual


class UndefinedVarError(TypeCheckError):
    """Raised when a variable is used before definition."""


class TypeMismatchError(TypeCheckError):
    """Raised when operand types are incompatible."""


# ---------------------------------------------------------------------------
# Verification / SMT errors
# ---------------------------------------------------------------------------

class VerificationError(DPCegarError):
    """Raised when the verification engine encounters a problem.

    This is *not* a privacy violation — it signals an internal failure
    in the verification pipeline itself.
    """


class SMTError(VerificationError):
    """Raised when the SMT solver reports an error or unknown status."""

    def __init__(
        self,
        message: str,
        *,
        solver_status: str | None = None,
        details: dict[str, Any] | None = None,
        source_loc: SourceLoc | None = None,
    ) -> None:
        details = dict(details or {})
        if solver_status is not None:
            details["solver_status"] = solver_status
        super().__init__(message, details=details, source_loc=source_loc)
        self.solver_status = solver_status


class TimeoutError_(SMTError):
    """Raised when the SMT solver times out (trailing underscore to
    avoid shadowing the built-in)."""


class UnsatCoreError(SMTError):
    """Raised when an unsat core cannot be extracted."""


# ---------------------------------------------------------------------------
# Privacy violation (counterexample found)
# ---------------------------------------------------------------------------

class PrivacyViolation(DPCegarError):
    """Raised (or returned) when a concrete privacy violation is found.

    Attributes:
        counterexample: Witness inputs demonstrating the violation.
        violated_budget: The privacy budget that was exceeded.
        actual_cost:     The computed privacy cost on the witness.
    """

    def __init__(
        self,
        message: str,
        *,
        counterexample: dict[str, Any] | None = None,
        violated_budget: Any = None,
        actual_cost: Any = None,
        details: dict[str, Any] | None = None,
        source_loc: SourceLoc | None = None,
    ) -> None:
        details = dict(details or {})
        if counterexample is not None:
            details["counterexample"] = counterexample
        super().__init__(message, details=details, source_loc=source_loc)
        self.counterexample = counterexample
        self.violated_budget = violated_budget
        self.actual_cost = actual_cost


# ---------------------------------------------------------------------------
# Repair errors
# ---------------------------------------------------------------------------

class RepairError(DPCegarError):
    """Raised when the repair engine fails to synthesise a fix."""


class NoRepairFoundError(RepairError):
    """Raised when all repair strategies are exhausted without success."""


class RepairBudgetExceeded(RepairError):
    """Raised when the repair search exceeds its iteration / time budget."""


# ---------------------------------------------------------------------------
# Configuration errors
# ---------------------------------------------------------------------------

class ConfigError(DPCegarError):
    """Raised for invalid or missing configuration values."""


class ConfigFileNotFoundError(ConfigError):
    """Raised when the configuration file cannot be located."""


class ConfigValidationError(ConfigError):
    """Raised when configuration values fail validation."""


# ---------------------------------------------------------------------------
# Internal / assertion errors
# ---------------------------------------------------------------------------

class InternalError(DPCegarError):
    """Raised when an internal invariant is violated.

    Indicates a bug in DP-CEGAR itself rather than in user input.
    """


class UnreachableError(InternalError):
    """Raised when control reaches a code path that should be unreachable."""

    def __init__(self, context: str = "") -> None:
        msg = "reached unreachable code"
        if context:
            msg += f": {context}"
        super().__init__(msg)


class NotImplementedYetError(InternalError):
    """Raised for features that are planned but not yet implemented."""

    def __init__(self, feature: str) -> None:
        super().__init__(f"not yet implemented: {feature}")


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def ensure(condition: bool, message: str, *, error_cls: type[DPCegarError] = InternalError) -> None:
    """Assert-like helper that raises *error_cls* when *condition* is False.

    Use instead of bare ``assert`` so that checks survive ``-O`` mode.
    """
    if not condition:
        raise error_cls(message)


def unreachable(context: str = "") -> Never:  # type: ignore[name-defined]
    """Marker for unreachable code; always raises :class:`UnreachableError`."""
    raise UnreachableError(context)
