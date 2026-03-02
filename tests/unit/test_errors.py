"""Tests for dpcegar.utils.errors – Error hierarchy and helpers."""

from __future__ import annotations

from typing import Any

import pytest

from dpcegar.utils.errors import (
    ConfigError,
    ConfigFileNotFoundError,
    ConfigValidationError,
    DPCegarError,
    InternalError,
    LexError,
    NoRepairFoundError,
    NotImplementedYetError,
    ParseError,
    PrivacyViolation,
    RepairBudgetExceeded,
    RepairError,
    SMTError,
    SourceLoc,
    SyntaxError_,
    TimeoutError_,
    TypeCheckError,
    TypeMismatchError,
    UndefinedVarError,
    UnreachableError,
    UnsatCoreError,
    VerificationError,
    ensure,
    unreachable,
)


# ═══════════════════════════════════════════════════════════════════════════
# SourceLoc
# ═══════════════════════════════════════════════════════════════════════════


class TestSourceLoc:
    """Tests for source location data class."""

    def test_default_values(self):
        loc = SourceLoc()
        assert loc.file == "<unknown>"
        assert loc.line == 0
        assert loc.col == 0

    def test_str_basic(self):
        loc = SourceLoc(file="test.py", line=10, col=5)
        assert str(loc) == "test.py:10:5"

    def test_str_with_end_range(self):
        loc = SourceLoc(file="f.py", line=1, col=1, end_line=3, end_col=10)
        s = str(loc)
        assert "1:1" in s
        assert "3:10" in s

    def test_is_frozen(self):
        loc = SourceLoc(file="a.py", line=1, col=1)
        with pytest.raises(AttributeError):
            loc.line = 2  # type: ignore[misc]


# ═══════════════════════════════════════════════════════════════════════════
# DPCegarError (base)
# ═══════════════════════════════════════════════════════════════════════════


class TestDPCegarError:
    """Tests for the base error class."""

    def test_message_preserved(self):
        err = DPCegarError("something went wrong")
        assert err.message == "something went wrong"

    def test_str_contains_message(self):
        err = DPCegarError("failure")
        assert "failure" in str(err)

    def test_details_default_empty(self):
        err = DPCegarError("x")
        assert err.details == {}

    def test_details_preserved(self):
        err = DPCegarError("x", details={"key": "val"})
        assert err.details["key"] == "val"

    def test_source_loc_in_str(self):
        loc = SourceLoc(file="f.py", line=5, col=3)
        err = DPCegarError("oops", source_loc=loc)
        assert "f.py:5:3" in str(err)

    def test_with_loc_returns_new_error(self):
        err = DPCegarError("oops")
        loc = SourceLoc(file="a.py", line=1, col=1)
        err2 = err.with_loc(loc)
        assert err2.source_loc == loc
        assert err.source_loc is None

    def test_to_dict(self):
        err = DPCegarError("fail", details={"x": 1})
        d = err.to_dict()
        assert d["error_type"] == "DPCegarError"
        assert d["message"] == "fail"
        assert d["details"]["x"] == 1

    def test_to_dict_with_loc(self):
        loc = SourceLoc(file="f.py", line=1, col=1)
        err = DPCegarError("fail", source_loc=loc)
        d = err.to_dict()
        assert "source_loc" in d

    def test_to_dict_no_details(self):
        err = DPCegarError("fail")
        d = err.to_dict()
        assert "details" not in d

    def test_is_exception(self):
        err = DPCegarError("test")
        assert isinstance(err, Exception)


# ═══════════════════════════════════════════════════════════════════════════
# ParseError hierarchy
# ═══════════════════════════════════════════════════════════════════════════


class TestParseErrors:
    """Tests for parse error classes."""

    def test_parse_error_with_token(self):
        err = ParseError("unexpected token", token="@")
        assert err.token == "@"
        assert err.details["token"] == "@"

    def test_parse_error_without_token(self):
        err = ParseError("bad input")
        assert err.token is None

    def test_lex_error_is_parse_error(self):
        err = LexError("can't lex")
        assert isinstance(err, ParseError)
        assert isinstance(err, DPCegarError)

    def test_syntax_error_is_parse_error(self):
        err = SyntaxError_("bad grammar")
        assert isinstance(err, ParseError)


# ═══════════════════════════════════════════════════════════════════════════
# TypeCheckError hierarchy
# ═══════════════════════════════════════════════════════════════════════════


class TestTypeCheckErrors:
    """Tests for type checking error classes."""

    def test_type_check_error_with_types(self):
        err = TypeCheckError("mismatch", expected="int", actual="bool")
        assert err.expected == "int"
        assert err.actual == "bool"

    def test_undefined_var_error_is_type_check(self):
        err = UndefinedVarError("var 'x' not defined")
        assert isinstance(err, TypeCheckError)

    def test_type_mismatch_error(self):
        err = TypeMismatchError("int vs bool", expected="int", actual="bool")
        assert isinstance(err, TypeCheckError)


# ═══════════════════════════════════════════════════════════════════════════
# Verification / SMT errors
# ═══════════════════════════════════════════════════════════════════════════


class TestVerificationErrors:
    """Tests for verification and SMT errors."""

    def test_verification_error_is_base(self):
        err = VerificationError("engine crash")
        assert isinstance(err, DPCegarError)

    def test_smt_error_with_status(self):
        err = SMTError("solver error", solver_status="unknown")
        assert err.solver_status == "unknown"
        assert err.details["solver_status"] == "unknown"

    def test_timeout_error_is_smt_error(self):
        err = TimeoutError_("timed out")
        assert isinstance(err, SMTError)

    def test_unsat_core_error(self):
        err = UnsatCoreError("no core available")
        assert isinstance(err, SMTError)


# ═══════════════════════════════════════════════════════════════════════════
# PrivacyViolation
# ═══════════════════════════════════════════════════════════════════════════


class TestPrivacyViolation:
    """Tests for privacy violation error."""

    def test_construction_with_counterexample(self):
        err = PrivacyViolation(
            "violation found",
            counterexample={"q": 5.0, "q_prime": 6.0},
            violated_budget="eps=1.0",
            actual_cost=2.5,
        )
        assert err.counterexample == {"q": 5.0, "q_prime": 6.0}
        assert err.violated_budget == "eps=1.0"
        assert err.actual_cost == 2.5

    def test_is_dpcegar_error(self):
        err = PrivacyViolation("violation")
        assert isinstance(err, DPCegarError)


# ═══════════════════════════════════════════════════════════════════════════
# Repair errors
# ═══════════════════════════════════════════════════════════════════════════


class TestRepairErrors:
    """Tests for repair error classes."""

    def test_repair_error_base(self):
        err = RepairError("repair failed")
        assert isinstance(err, DPCegarError)

    def test_no_repair_found(self):
        err = NoRepairFoundError("exhausted")
        assert isinstance(err, RepairError)

    def test_repair_budget_exceeded(self):
        err = RepairBudgetExceeded("too many attempts")
        assert isinstance(err, RepairError)


# ═══════════════════════════════════════════════════════════════════════════
# Config errors
# ═══════════════════════════════════════════════════════════════════════════


class TestConfigErrors:
    """Tests for configuration error classes."""

    def test_config_error_base(self):
        err = ConfigError("bad config")
        assert isinstance(err, DPCegarError)

    def test_config_file_not_found(self):
        err = ConfigFileNotFoundError("missing.toml")
        assert isinstance(err, ConfigError)

    def test_config_validation_error(self):
        err = ConfigValidationError("invalid value")
        assert isinstance(err, ConfigError)


# ═══════════════════════════════════════════════════════════════════════════
# Internal errors
# ═══════════════════════════════════════════════════════════════════════════


class TestInternalErrors:
    """Tests for internal / assertion error classes."""

    def test_internal_error(self):
        err = InternalError("invariant violated")
        assert isinstance(err, DPCegarError)

    def test_unreachable_error_no_context(self):
        err = UnreachableError()
        assert "unreachable" in str(err).lower()

    def test_unreachable_error_with_context(self):
        err = UnreachableError("switch case")
        assert "switch case" in str(err)

    def test_not_implemented_yet(self):
        err = NotImplementedYetError("fancy feature")
        assert "fancy feature" in str(err)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


class TestHelpers:
    """Tests for ensure() and unreachable()."""

    def test_ensure_passes_on_true(self):
        ensure(True, "should not fail")

    def test_ensure_raises_on_false(self):
        with pytest.raises(InternalError, match="oops"):
            ensure(False, "oops")

    def test_ensure_custom_error_class(self):
        with pytest.raises(ConfigError):
            ensure(False, "bad config", error_cls=ConfigError)

    def test_unreachable_always_raises(self):
        with pytest.raises(UnreachableError):
            unreachable("should not get here")
