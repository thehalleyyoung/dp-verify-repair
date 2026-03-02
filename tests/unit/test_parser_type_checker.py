"""Tests for dpcegar.parser.type_checker – DPImp type checking & validation.

Covers:
  • TypeEnvironment: scope stack, define, lookup, sensitivity tracking
  • SensitivityKind & SensitivityType: is_sensitive, CONSTANT, QUERY_RESULT, etc.
  • TypeChecker: assignment checks, expr type-checking, noise/query/loop/branch
  • TypeErrorInfo: construction and __str__
  • type_check / type_check_strict public API
  • DPImp restriction enforcement: no recursion, bounded loops, sensitivity flow
"""

from __future__ import annotations

import pytest

from dpcegar.parser.type_checker import (
    SensitivityKind,
    SensitivityType,
    TypeChecker,
    TypeEnvironment,
    TypeErrorInfo,
    type_check,
    type_check_strict,
)
from dpcegar.ir.types import (
    BinOp,
    BinOpKind,
    Const,
    FuncCall,
    IRType,
    NoiseKind,
    UnaryOp,
    UnaryOpKind,
    Var,
)
from dpcegar.ir.nodes import (
    AssignNode,
    BranchNode,
    LoopNode,
    MechIR,
    MergeNode,
    NoOpNode,
    NoiseDrawNode,
    ParamDecl,
    QueryNode,
    ReturnNode,
    SequenceNode,
)
from dpcegar.utils.errors import SourceLoc, TypeCheckError


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def env() -> TypeEnvironment:
    return TypeEnvironment()


@pytest.fixture
def checker() -> TypeChecker:
    return TypeChecker()


@pytest.fixture
def simple_mechir() -> MechIR:
    """Laplace mechanism: query → noise → return."""
    q = QueryNode(
        target=Var(ty=IRType.REAL, name="q"),
        query_name="count",
        args=(Var(ty=IRType.REAL, name="db"),),
        sensitivity=Const.real(1.0),
    )
    noise = NoiseDrawNode(
        target=Var(ty=IRType.REAL, name="eta"),
        noise_kind=NoiseKind.LAPLACE,
        center=Var(ty=IRType.REAL, name="q"),
        scale=Const.real(1.0),
    )
    ret = ReturnNode(value=Var(ty=IRType.REAL, name="eta"))
    body = SequenceNode(stmts=[q, noise, ret])
    return MechIR(
        name="test_mech",
        params=[
            ParamDecl(name="db", ty=IRType.ARRAY, is_database=True),
            ParamDecl(name="epsilon", ty=IRType.REAL),
        ],
        body=body,
        return_type=IRType.REAL,
    )


@pytest.fixture
def no_noise_mechir() -> MechIR:
    """Mechanism with a query but no noise (likely non-private)."""
    q = QueryNode(
        target=Var(ty=IRType.REAL, name="q"),
        query_name="count",
        args=(Var(ty=IRType.REAL, name="db"),),
        sensitivity=Const.real(1.0),
    )
    ret = ReturnNode(value=Var(ty=IRType.REAL, name="q"))
    body = SequenceNode(stmts=[q, ret])
    return MechIR(
        name="no_noise",
        params=[ParamDecl(name="db", ty=IRType.ARRAY, is_database=True)],
        body=body,
        return_type=IRType.REAL,
    )


@pytest.fixture
def no_return_mechir() -> MechIR:
    """Mechanism with no return statement."""
    assign = AssignNode(
        target=Var(ty=IRType.REAL, name="x"),
        value=Const.real(1.0),
    )
    return MechIR(
        name="no_return",
        params=[],
        body=assign,
        return_type=IRType.REAL,
    )


# ═══════════════════════════════════════════════════════════════════════════
# SensitivityKind
# ═══════════════════════════════════════════════════════════════════════════


class TestSensitivityKind:
    """Verify the SensitivityKind enum values."""

    def test_all_members(self):
        expected = {"CONSTANT", "QUERY_RESULT", "DERIVED", "NOISY", "UNKNOWN"}
        actual = {k.name for k in SensitivityKind}
        assert actual == expected

    def test_auto_values_distinct(self):
        values = [k.value for k in SensitivityKind]
        assert len(values) == len(set(values))


# ═══════════════════════════════════════════════════════════════════════════
# SensitivityType
# ═══════════════════════════════════════════════════════════════════════════


class TestSensitivityType:
    """SensitivityType dataclass behaviour."""

    def test_default_is_constant(self):
        st = SensitivityType()
        assert st.kind == SensitivityKind.CONSTANT

    def test_constant_not_sensitive(self):
        st = SensitivityType(kind=SensitivityKind.CONSTANT)
        assert not st.is_sensitive()

    def test_query_result_is_sensitive(self):
        st = SensitivityType(kind=SensitivityKind.QUERY_RESULT)
        assert st.is_sensitive()

    def test_derived_is_sensitive(self):
        st = SensitivityType(kind=SensitivityKind.DERIVED)
        assert st.is_sensitive()

    def test_noisy_not_sensitive(self):
        st = SensitivityType(kind=SensitivityKind.NOISY)
        assert not st.is_sensitive()

    def test_unknown_not_sensitive(self):
        st = SensitivityType(kind=SensitivityKind.UNKNOWN)
        assert not st.is_sensitive()

    def test_frozen(self):
        st = SensitivityType(kind=SensitivityKind.CONSTANT)
        with pytest.raises(AttributeError):
            st.kind = SensitivityKind.NOISY  # type: ignore[misc]

    def test_sensitivity_value(self):
        st = SensitivityType(kind=SensitivityKind.QUERY_RESULT, sensitivity=1.0)
        assert st.sensitivity == 1.0

    def test_source_var(self):
        st = SensitivityType(kind=SensitivityKind.QUERY_RESULT, source_var="q")
        assert st.source_var == "q"


# ═══════════════════════════════════════════════════════════════════════════
# TypeEnvironment
# ═══════════════════════════════════════════════════════════════════════════


class TestTypeEnvironment:
    """TypeEnvironment scope stack operations."""

    def test_initial_depth(self, env):
        assert env.depth == 1

    def test_push_pop(self, env):
        env.push_scope()
        assert env.depth == 2
        env.pop_scope()
        assert env.depth == 1

    def test_pop_beyond_base(self, env):
        env.pop_scope()
        assert env.depth == 1  # can't pop below 1

    def test_define_lookup(self, env):
        env.define("x", IRType.REAL)
        assert env.lookup("x") == IRType.REAL

    def test_lookup_undefined(self, env):
        assert env.lookup("nonexistent") is None

    def test_is_defined(self, env):
        env.define("x", IRType.INT)
        assert env.is_defined("x")
        assert not env.is_defined("y")

    def test_inner_scope_shadows(self, env):
        env.define("x", IRType.INT)
        env.push_scope()
        env.define("x", IRType.REAL)
        assert env.lookup("x") == IRType.REAL
        env.pop_scope()
        assert env.lookup("x") == IRType.INT

    def test_inner_scope_sees_outer(self, env):
        env.define("x", IRType.REAL)
        env.push_scope()
        assert env.lookup("x") == IRType.REAL

    def test_all_variables(self, env):
        env.define("a", IRType.INT)
        env.push_scope()
        env.define("b", IRType.REAL)
        all_vars = env.all_variables()
        assert "a" in all_vars
        assert "b" in all_vars

    def test_all_variables_inner_wins(self, env):
        env.define("x", IRType.INT)
        env.push_scope()
        env.define("x", IRType.BOOL)
        all_vars = env.all_variables()
        assert all_vars["x"] == IRType.BOOL

    def test_define_with_sensitivity(self, env):
        sens = SensitivityType(kind=SensitivityKind.QUERY_RESULT, source_var="q")
        env.define("q", IRType.REAL, sens)
        looked = env.lookup_sensitivity("q")
        assert looked is not None
        assert looked.kind == SensitivityKind.QUERY_RESULT

    def test_lookup_sensitivity_undefined(self, env):
        assert env.lookup_sensitivity("z") is None

    def test_default_sensitivity_is_constant(self, env):
        env.define("c", IRType.INT)
        sens = env.lookup_sensitivity("c")
        assert sens is not None
        assert sens.kind == SensitivityKind.CONSTANT

    def test_sensitive_variables(self, env):
        env.define("c", IRType.REAL)
        env.define(
            "q", IRType.REAL,
            SensitivityType(kind=SensitivityKind.QUERY_RESULT),
        )
        env.define(
            "d", IRType.REAL,
            SensitivityType(kind=SensitivityKind.DERIVED),
        )
        result = env.sensitive_variables()
        assert "q" in result
        assert "d" in result
        assert "c" not in result


# ═══════════════════════════════════════════════════════════════════════════
# TypeErrorInfo
# ═══════════════════════════════════════════════════════════════════════════


class TestTypeErrorInfo:
    """TypeErrorInfo construction and formatting."""

    def test_default_severity_error(self):
        info = TypeErrorInfo(message="bad")
        assert info.severity == "error"

    def test_str_without_loc(self):
        info = TypeErrorInfo(message="oops")
        assert "[error] oops" == str(info)

    def test_str_with_loc(self):
        loc = SourceLoc(file="f.py", line=10, col=5)
        info = TypeErrorInfo(message="oops", source_loc=loc)
        s = str(info)
        assert "f.py" in s
        assert "oops" in s

    def test_warning_severity(self):
        info = TypeErrorInfo(message="maybe", severity="warning")
        assert info.severity == "warning"
        assert "[warning]" in str(info)

    def test_category_default(self):
        info = TypeErrorInfo(message="x")
        assert info.category == "type"


# ═══════════════════════════════════════════════════════════════════════════
# TypeChecker — basic type checking
# ═══════════════════════════════════════════════════════════════════════════


class TestTypeCheckerBasic:
    """Basic type-checking of well-formed MechIR trees."""

    def test_valid_mech_passes(self, checker, simple_mechir):
        ok = checker.check(simple_mechir)
        assert ok
        assert len(checker.errors) == 0

    def test_params_registered(self, checker, simple_mechir):
        checker.check(simple_mechir)
        assert checker.env.is_defined("db")
        assert checker.env.is_defined("epsilon")

    def test_db_param_sensitivity(self, checker, simple_mechir):
        checker.check(simple_mechir)
        sens = checker.env.lookup_sensitivity("db")
        assert sens is not None
        assert sens.kind == SensitivityKind.QUERY_RESULT

    def test_non_db_param_constant(self, checker, simple_mechir):
        checker.check(simple_mechir)
        sens = checker.env.lookup_sensitivity("epsilon")
        assert sens is not None
        assert sens.kind == SensitivityKind.CONSTANT


class TestTypeCheckerAssignment:
    """Type-checking assignment nodes."""

    def test_assign_defines_variable(self, checker):
        assign = AssignNode(
            target=Var(ty=IRType.REAL, name="x"),
            value=Const.real(3.14),
        )
        ret = ReturnNode(value=Var(ty=IRType.REAL, name="x"))
        mechir = MechIR(
            name="m", params=[], body=SequenceNode(stmts=[assign, ret]),
        )
        ok = checker.check(mechir)
        assert ok
        assert checker.env.lookup("x") == IRType.REAL

    def test_assign_non_var_target_errors(self, checker):
        assign = AssignNode(
            target=Const.real(1.0),  # wrong target type
            value=Const.real(3.14),
        )
        ret = ReturnNode(value=Const.real(1.0))
        mechir = MechIR(
            name="m", params=[], body=SequenceNode(stmts=[assign, ret]),
        )
        ok = checker.check(mechir)
        assert not ok
        assert any("target" in e.message.lower() for e in checker.errors)


# ═══════════════════════════════════════════════════════════════════════════
# TypeChecker — expression checking
# ═══════════════════════════════════════════════════════════════════════════


class TestTypeCheckerExpressions:
    """Type inference for various expression types."""

    def test_const_int_type(self, checker):
        result = checker._check_expr(Const.int_(42))
        assert result == IRType.INT

    def test_const_real_type(self, checker):
        result = checker._check_expr(Const.real(3.14))
        assert result == IRType.REAL

    def test_const_bool_type(self, checker):
        result = checker._check_expr(Const.bool_(True))
        assert result == IRType.BOOL

    def test_binop_add_real(self, checker):
        expr = BinOp(
            ty=IRType.REAL, op=BinOpKind.ADD,
            left=Const.real(1.0), right=Const.real(2.0),
        )
        result = checker._check_expr(expr)
        assert result in (IRType.REAL, IRType.INT)

    def test_binop_add_int(self, checker):
        expr = BinOp(
            ty=IRType.INT, op=BinOpKind.ADD,
            left=Const.int_(1), right=Const.int_(2),
        )
        result = checker._check_expr(expr)
        assert result == IRType.INT

    def test_binop_div_returns_real(self, checker):
        expr = BinOp(
            ty=IRType.REAL, op=BinOpKind.DIV,
            left=Const.int_(1), right=Const.int_(2),
        )
        result = checker._check_expr(expr)
        assert result == IRType.REAL

    def test_comparison_returns_bool(self, checker):
        expr = BinOp(
            ty=IRType.BOOL, op=BinOpKind.GT,
            left=Const.real(1.0), right=Const.real(0.0),
        )
        result = checker._check_expr(expr)
        assert result == IRType.BOOL

    def test_logical_and_returns_bool(self, checker):
        expr = BinOp(
            ty=IRType.BOOL, op=BinOpKind.AND,
            left=Const.bool_(True), right=Const.bool_(False),
        )
        result = checker._check_expr(expr)
        assert result == IRType.BOOL

    def test_unary_neg(self, checker):
        expr = UnaryOp(
            ty=IRType.REAL, op=UnaryOpKind.NEG,
            operand=Const.real(1.0),
        )
        result = checker._check_expr(expr)
        assert result == IRType.REAL

    def test_unary_not(self, checker):
        expr = UnaryOp(
            ty=IRType.BOOL, op=UnaryOpKind.NOT,
            operand=Const.bool_(True),
        )
        result = checker._check_expr(expr)
        assert result == IRType.BOOL

    def test_funccall_abs(self, checker):
        expr = FuncCall(
            ty=IRType.REAL, name="abs",
            args=(Const.real(-1.0),),
        )
        result = checker._check_expr(expr)
        assert result == IRType.REAL

    def test_funccall_len(self, checker):
        expr = FuncCall(
            ty=IRType.INT, name="len",
            args=(Var(ty=IRType.ARRAY, name="arr"),),
        )
        result = checker._check_expr(expr)
        assert result == IRType.INT

    def test_undefined_var_warns(self, checker):
        var = Var(ty=IRType.REAL, name="undefined_var")
        checker._check_expr(var)
        assert any("undefined" in w.message.lower() for w in checker.warnings)


# ═══════════════════════════════════════════════════════════════════════════
# TypeChecker — DPImp restrictions
# ═══════════════════════════════════════════════════════════════════════════


class TestDPImpRestrictions:
    """Enforcement of DPImp restrictions."""

    def test_recursion_detected(self, checker):
        call = FuncCall(ty=IRType.REAL, name="mech", args=())
        assign = AssignNode(
            target=Var(ty=IRType.REAL, name="r"),
            value=call,
        )
        ret = ReturnNode(value=Var(ty=IRType.REAL, name="r"))
        mechir = MechIR(
            name="mech", params=[], body=SequenceNode(stmts=[assign, ret]),
        )
        ok = checker.check(mechir)
        assert not ok
        assert any("recursion" in e.message.lower() for e in checker.errors)

    def test_no_return_warns(self, checker, no_return_mechir):
        checker.check(no_return_mechir)
        assert any("no return" in w.message.lower() for w in checker.warnings)

    def test_no_noise_warns(self, checker, no_noise_mechir):
        checker.check(no_noise_mechir)
        assert any("no noise" in w.message.lower() for w in checker.warnings)


# ═══════════════════════════════════════════════════════════════════════════
# TypeChecker — Noise draw checking
# ═══════════════════════════════════════════════════════════════════════════


class TestTypeCheckerNoise:
    """Type checking NoiseDrawNode."""

    def test_noise_defines_var_as_real(self, checker):
        noise = NoiseDrawNode(
            target=Var(ty=IRType.REAL, name="eta"),
            noise_kind=NoiseKind.LAPLACE,
            center=Const.real(0.0),
            scale=Const.real(1.0),
        )
        ret = ReturnNode(value=Var(ty=IRType.REAL, name="eta"))
        mechir = MechIR(
            name="m", params=[], body=SequenceNode(stmts=[noise, ret]),
        )
        checker.check(mechir)
        assert checker.env.lookup("eta") == IRType.REAL

    def test_noise_var_marked_noisy(self, checker):
        noise = NoiseDrawNode(
            target=Var(ty=IRType.REAL, name="eta"),
            noise_kind=NoiseKind.GAUSSIAN,
            center=Const.real(0.0),
            scale=Const.real(1.0),
        )
        ret = ReturnNode(value=Var(ty=IRType.REAL, name="eta"))
        mechir = MechIR(
            name="m", params=[], body=SequenceNode(stmts=[noise, ret]),
        )
        checker.check(mechir)
        sens = checker.env.lookup_sensitivity("eta")
        assert sens is not None
        assert sens.kind == SensitivityKind.NOISY

    def test_noise_with_sensitivity(self, checker):
        noise = NoiseDrawNode(
            target=Var(ty=IRType.REAL, name="eta"),
            noise_kind=NoiseKind.LAPLACE,
            center=Const.real(0.0),
            scale=Const.real(1.0),
            sensitivity=Const.real(1.0),
        )
        ret = ReturnNode(value=Var(ty=IRType.REAL, name="eta"))
        mechir = MechIR(
            name="m", params=[], body=SequenceNode(stmts=[noise, ret]),
        )
        ok = checker.check(mechir)
        assert ok


# ═══════════════════════════════════════════════════════════════════════════
# TypeChecker — Branch checking
# ═══════════════════════════════════════════════════════════════════════════


class TestTypeCheckerBranch:
    """Type checking BranchNode."""

    def test_branch_condition_must_be_bool(self, checker):
        branch = BranchNode(
            condition=Const.real(1.0),  # not bool!
            true_branch=NoOpNode(),
            false_branch=NoOpNode(),
        )
        ret = ReturnNode(value=Const.real(0.0))
        mechir = MechIR(
            name="m", params=[], body=SequenceNode(stmts=[branch, ret]),
        )
        checker.check(mechir)
        assert any("bool" in e.message.lower() for e in checker.errors)

    def test_valid_branch_passes(self, checker):
        branch = BranchNode(
            condition=BinOp(
                ty=IRType.BOOL, op=BinOpKind.GT,
                left=Const.real(1.0), right=Const.real(0.0),
            ),
            true_branch=AssignNode(
                target=Var(ty=IRType.REAL, name="r"),
                value=Const.real(1.0),
            ),
            false_branch=AssignNode(
                target=Var(ty=IRType.REAL, name="r"),
                value=Const.real(0.0),
            ),
        )
        ret = ReturnNode(value=Var(ty=IRType.REAL, name="r"))
        mechir = MechIR(
            name="m", params=[], body=SequenceNode(stmts=[branch, ret]),
        )
        ok = checker.check(mechir)
        assert ok


# ═══════════════════════════════════════════════════════════════════════════
# TypeChecker — Loop checking
# ═══════════════════════════════════════════════════════════════════════════


class TestTypeCheckerLoop:
    """Type checking LoopNode."""

    def test_loop_constant_bound_ok(self, checker):
        loop = LoopNode(
            index_var=Var(ty=IRType.INT, name="i"),
            bound=Const.int_(10),
            body=NoOpNode(),
        )
        ret = ReturnNode(value=Const.real(0.0))
        mechir = MechIR(
            name="m", params=[], body=SequenceNode(stmts=[loop, ret]),
        )
        ok = checker.check(mechir)
        assert ok

    def test_loop_var_bound_with_param(self, checker):
        loop = LoopNode(
            index_var=Var(ty=IRType.INT, name="i"),
            bound=Var(ty=IRType.INT, name="n"),
            body=NoOpNode(),
        )
        ret = ReturnNode(value=Const.real(0.0))
        mechir = MechIR(
            name="m",
            params=[ParamDecl(name="n", ty=IRType.INT)],
            body=SequenceNode(stmts=[loop, ret]),
        )
        ok = checker.check(mechir)
        assert ok

    def test_loop_index_registered(self, checker):
        loop = LoopNode(
            index_var=Var(ty=IRType.INT, name="i"),
            bound=Const.int_(5),
            body=NoOpNode(),
        )
        ret = ReturnNode(value=Const.real(0.0))
        mechir = MechIR(
            name="m", params=[], body=SequenceNode(stmts=[loop, ret]),
        )
        checker.check(mechir)
        assert checker.env.lookup("i") == IRType.INT


# ═══════════════════════════════════════════════════════════════════════════
# TypeChecker — sensitivity inference
# ═══════════════════════════════════════════════════════════════════════════


class TestSensitivityInference:
    """_infer_sensitivity tracks data flow from queries."""

    def test_constant_expr(self, checker):
        sens = checker._infer_sensitivity(Const.real(1.0))
        assert sens.kind == SensitivityKind.CONSTANT

    def test_query_var_propagates(self, checker):
        checker.env.define(
            "q", IRType.REAL,
            SensitivityType(kind=SensitivityKind.QUERY_RESULT, source_var="q"),
        )
        sens = checker._infer_sensitivity(Var(ty=IRType.REAL, name="q"))
        assert sens.kind == SensitivityKind.QUERY_RESULT

    def test_derived_from_binop(self, checker):
        checker.env.define(
            "q", IRType.REAL,
            SensitivityType(kind=SensitivityKind.QUERY_RESULT, source_var="q"),
        )
        expr = BinOp(
            ty=IRType.REAL, op=BinOpKind.ADD,
            left=Var(ty=IRType.REAL, name="q"),
            right=Const.real(1.0),
        )
        sens = checker._infer_sensitivity(expr)
        assert sens.kind == SensitivityKind.DERIVED
        assert sens.source_var == "q"

    def test_constant_binop(self, checker):
        expr = BinOp(
            ty=IRType.REAL, op=BinOpKind.MUL,
            left=Const.real(2.0), right=Const.real(3.0),
        )
        sens = checker._infer_sensitivity(expr)
        assert sens.kind == SensitivityKind.CONSTANT

    def test_funccall_propagates(self, checker):
        checker.env.define(
            "q", IRType.REAL,
            SensitivityType(kind=SensitivityKind.QUERY_RESULT, source_var="q"),
        )
        expr = FuncCall(
            ty=IRType.REAL, name="abs",
            args=(Var(ty=IRType.REAL, name="q"),),
        )
        sens = checker._infer_sensitivity(expr)
        assert sens.kind == SensitivityKind.DERIVED

    def test_unaryop_propagates(self, checker):
        checker.env.define(
            "q", IRType.REAL,
            SensitivityType(kind=SensitivityKind.QUERY_RESULT, source_var="q"),
        )
        expr = UnaryOp(
            ty=IRType.REAL, op=UnaryOpKind.NEG,
            operand=Var(ty=IRType.REAL, name="q"),
        )
        sens = checker._infer_sensitivity(expr)
        assert sens.is_sensitive()


# ═══════════════════════════════════════════════════════════════════════════
# type_check / type_check_strict public API
# ═══════════════════════════════════════════════════════════════════════════


class TestPublicAPI:
    """Public functions type_check and type_check_strict."""

    def test_type_check_returns_tuple(self, simple_mechir):
        ok, errors = type_check(simple_mechir)
        assert isinstance(ok, bool)
        assert isinstance(errors, list)

    def test_type_check_valid(self, simple_mechir):
        ok, errors = type_check(simple_mechir)
        assert ok
        assert len(errors) == 0

    def test_type_check_strict_valid(self, simple_mechir):
        # Should not raise
        type_check_strict(simple_mechir)

    def test_type_check_strict_raises(self):
        assign = AssignNode(
            target=Const.real(1.0),  # invalid target
            value=Const.real(1.0),
        )
        ret = ReturnNode(value=Const.real(0.0))
        mechir = MechIR(
            name="m", params=[], body=SequenceNode(stmts=[assign, ret]),
        )
        with pytest.raises(TypeCheckError):
            type_check_strict(mechir)

    def test_type_check_multiple_errors(self):
        a1 = AssignNode(
            target=Const.real(1.0),
            value=Const.real(1.0),
        )
        a2 = AssignNode(
            target=Const.int_(2),
            value=Const.int_(2),
        )
        ret = ReturnNode(value=Const.real(0.0))
        mechir = MechIR(
            name="m", params=[], body=SequenceNode(stmts=[a1, a2, ret]),
        )
        ok, errors = type_check(mechir)
        assert not ok
        assert len(errors) >= 2


# ═══════════════════════════════════════════════════════════════════════════
# TypeChecker — query node
# ═══════════════════════════════════════════════════════════════════════════


class TestTypeCheckerQuery:
    """Query node type checking."""

    def test_query_registers_target(self, checker):
        q = QueryNode(
            target=Var(ty=IRType.REAL, name="q"),
            query_name="count",
            args=(Var(ty=IRType.REAL, name="db"),),
            sensitivity=Const.real(1.0),
        )
        ret = ReturnNode(value=Var(ty=IRType.REAL, name="q"))
        mechir = MechIR(
            name="m",
            params=[ParamDecl(name="db", ty=IRType.ARRAY, is_database=True)],
            body=SequenceNode(stmts=[q, ret]),
        )
        checker.check(mechir)
        assert checker.env.lookup("q") == IRType.REAL

    def test_query_sensitivity_is_query_result(self, checker):
        q = QueryNode(
            target=Var(ty=IRType.REAL, name="q"),
            query_name="count",
            args=(Var(ty=IRType.REAL, name="db"),),
            sensitivity=Const.real(1.0),
        )
        ret = ReturnNode(value=Var(ty=IRType.REAL, name="q"))
        mechir = MechIR(
            name="m",
            params=[ParamDecl(name="db", ty=IRType.ARRAY, is_database=True)],
            body=SequenceNode(stmts=[q, ret]),
        )
        checker.check(mechir)
        sens = checker.env.lookup_sensitivity("q")
        assert sens is not None
        assert sens.kind == SensitivityKind.QUERY_RESULT

    def test_empty_query_name_errors(self, checker):
        q = QueryNode(
            target=Var(ty=IRType.REAL, name="q"),
            query_name="",
            args=(),
            sensitivity=Const.real(1.0),
        )
        ret = ReturnNode(value=Var(ty=IRType.REAL, name="q"))
        mechir = MechIR(
            name="m", params=[], body=SequenceNode(stmts=[q, ret]),
        )
        ok = checker.check(mechir)
        assert not ok


# ═══════════════════════════════════════════════════════════════════════════
# TypeChecker — merge node
# ═══════════════════════════════════════════════════════════════════════════


class TestTypeCheckerMerge:
    """Type checking MergeNode."""

    def test_merge_no_sources_warns(self, checker):
        merge = MergeNode(
            target=Var(ty=IRType.REAL, name="x"),
            sources={},
        )
        ret = ReturnNode(value=Var(ty=IRType.REAL, name="x"))
        mechir = MechIR(
            name="m", params=[], body=SequenceNode(stmts=[merge, ret]),
        )
        checker.check(mechir)
        assert any("no sources" in w.message.lower() for w in checker.warnings)

    def test_merge_compatible_sources(self, checker):
        merge = MergeNode(
            target=Var(ty=IRType.REAL, name="x"),
            sources={
                0: Const.real(1.0),
                1: Const.real(2.0),
            },
        )
        ret = ReturnNode(value=Var(ty=IRType.REAL, name="x"))
        mechir = MechIR(
            name="m", params=[], body=SequenceNode(stmts=[merge, ret]),
        )
        ok = checker.check(mechir)
        assert ok


# ═══════════════════════════════════════════════════════════════════════════
# TypeChecker — return type consistency
# ═══════════════════════════════════════════════════════════════════════════


class TestReturnTypeConsistency:
    """Post-check for consistent return types."""

    def test_single_return_no_warning(self, checker, simple_mechir):
        checker.check(simple_mechir)
        type_warns = [w for w in checker.warnings if "return type" in w.message.lower()]
        assert len(type_warns) == 0

    def test_check_idempotent(self, checker, simple_mechir):
        ok1 = checker.check(simple_mechir)
        ok2 = checker.check(simple_mechir)
        assert ok1 == ok2
        # errors should be cleared between checks
        assert len(checker.errors) == 0
