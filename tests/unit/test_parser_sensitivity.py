"""Tests for dpcegar.parser.sensitivity – sensitivity analysis for DPImp.

Covers:
  • SensitivityNorm: L1, L2, LINF
  • SymbolicSens: construction, factories, arithmetic, properties
  • SensitivityResult, SensitivityCert
  • SensitivityAnalyzer: per-query and global analysis
  • sequential_compose / parallel_compose helpers
  • analyze_sensitivity / generate_sensitivity_certificate public API
"""

from __future__ import annotations

import math

import pytest

from dpcegar.parser.sensitivity import (
    NoiseInfo,
    QuerySensitivity,
    SensitivityAnalyzer,
    SensitivityCert,
    SensitivityNorm,
    SensitivityResult,
    SymbolicSens,
    analyze_sensitivity,
    generate_sensitivity_certificate,
    parallel_compose,
    sequential_compose,
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
    NoOpNode,
    NoiseDrawNode,
    ParamDecl,
    QueryNode,
    ReturnNode,
    SequenceNode,
)


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def laplace_mechir() -> MechIR:
    """Simple Laplace mechanism: query → noise → return."""
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
        name="lap_mech",
        params=[
            ParamDecl(name="db", ty=IRType.ARRAY, is_database=True),
            ParamDecl(name="epsilon", ty=IRType.REAL),
        ],
        body=body,
        return_type=IRType.REAL,
    )


@pytest.fixture
def two_query_mechir() -> MechIR:
    """Mechanism with two sequential queries."""
    q1 = QueryNode(
        target=Var(ty=IRType.REAL, name="q1"),
        query_name="count",
        args=(Var(ty=IRType.REAL, name="db"),),
        sensitivity=Const.real(1.0),
    )
    n1 = NoiseDrawNode(
        target=Var(ty=IRType.REAL, name="e1"),
        noise_kind=NoiseKind.LAPLACE,
        center=Var(ty=IRType.REAL, name="q1"),
        scale=Const.real(1.0),
    )
    q2 = QueryNode(
        target=Var(ty=IRType.REAL, name="q2"),
        query_name="sum_query",
        args=(Var(ty=IRType.REAL, name="db"),),
        sensitivity=Const.real(2.0),
    )
    n2 = NoiseDrawNode(
        target=Var(ty=IRType.REAL, name="e2"),
        noise_kind=NoiseKind.LAPLACE,
        center=Var(ty=IRType.REAL, name="q2"),
        scale=Const.real(2.0),
    )
    ret_val = BinOp(
        ty=IRType.REAL, op=BinOpKind.ADD,
        left=Var(ty=IRType.REAL, name="e1"),
        right=Var(ty=IRType.REAL, name="e2"),
    )
    ret = ReturnNode(value=ret_val)
    body = SequenceNode(stmts=[q1, n1, q2, n2, ret])
    return MechIR(
        name="two_query",
        params=[ParamDecl(name="db", ty=IRType.ARRAY, is_database=True)],
        body=body,
        return_type=IRType.REAL,
    )


@pytest.fixture
def loop_mechir() -> MechIR:
    """Mechanism with a loop containing a query."""
    inner_q = QueryNode(
        target=Var(ty=IRType.REAL, name="q"),
        query_name="count",
        args=(Var(ty=IRType.REAL, name="db"),),
        sensitivity=Const.real(1.0),
    )
    inner_noise = NoiseDrawNode(
        target=Var(ty=IRType.REAL, name="eta"),
        noise_kind=NoiseKind.LAPLACE,
        center=Var(ty=IRType.REAL, name="q"),
        scale=Const.real(1.0),
    )
    inner_assign = AssignNode(
        target=Var(ty=IRType.REAL, name="s"),
        value=BinOp(
            ty=IRType.REAL, op=BinOpKind.ADD,
            left=Var(ty=IRType.REAL, name="s"),
            right=Var(ty=IRType.REAL, name="eta"),
        ),
    )
    loop_body = SequenceNode(stmts=[inner_q, inner_noise, inner_assign])
    init = AssignNode(
        target=Var(ty=IRType.REAL, name="s"),
        value=Const.real(0.0),
    )
    loop = LoopNode(
        index_var=Var(ty=IRType.INT, name="i"),
        bound=Const.int_(5),
        body=loop_body,
    )
    ret = ReturnNode(value=Var(ty=IRType.REAL, name="s"))
    body = SequenceNode(stmts=[init, loop, ret])
    return MechIR(
        name="loop_mech",
        params=[ParamDecl(name="db", ty=IRType.ARRAY, is_database=True)],
        body=body,
        return_type=IRType.REAL,
    )


@pytest.fixture
def no_query_mechir() -> MechIR:
    """Mechanism with no query (pure constant)."""
    ret = ReturnNode(value=Const.real(42.0))
    return MechIR(name="const_mech", params=[], body=ret, return_type=IRType.REAL)


# ═══════════════════════════════════════════════════════════════════════════
# SensitivityNorm
# ═══════════════════════════════════════════════════════════════════════════


class TestSensitivityNorm:
    """Verify the SensitivityNorm enum."""

    def test_members(self):
        assert SensitivityNorm.L1 is not None
        assert SensitivityNorm.L2 is not None
        assert SensitivityNorm.LINF is not None

    def test_str_l1(self):
        assert str(SensitivityNorm.L1) == "L1"

    def test_str_l2(self):
        assert str(SensitivityNorm.L2) == "L2"

    def test_str_linf(self):
        assert "L" in str(SensitivityNorm.LINF)  # "L∞" or similar


# ═══════════════════════════════════════════════════════════════════════════
# SymbolicSens — construction & factories
# ═══════════════════════════════════════════════════════════════════════════


class TestSymbolicSensConstruction:
    """SymbolicSens factories and basic properties."""

    def test_zero(self):
        s = SymbolicSens.zero()
        assert s.value == 0.0
        assert s.is_zero
        assert s.is_concrete
        assert s.is_bounded

    def test_constant(self):
        s = SymbolicSens.constant(3.5)
        assert s.value == 3.5
        assert s.is_concrete
        assert not s.is_zero

    def test_unbounded(self):
        s = SymbolicSens.unbounded()
        assert s.value == float("inf")
        assert not s.is_bounded

    def test_from_var(self):
        s = SymbolicSens.from_var("n")
        assert s.symbolic == "n"
        assert "n" in s.depends_on
        assert not s.is_concrete

    def test_frozen(self):
        s = SymbolicSens.zero()
        with pytest.raises(AttributeError):
            s.value = 1.0  # type: ignore[misc]


# ═══════════════════════════════════════════════════════════════════════════
# SymbolicSens — arithmetic
# ═══════════════════════════════════════════════════════════════════════════


class TestSymbolicSensArithmetic:
    """Arithmetic operations on SymbolicSens."""

    def test_add_concrete(self):
        a = SymbolicSens.constant(1.0)
        b = SymbolicSens.constant(2.0)
        result = a.add(b)
        assert result.is_concrete
        assert result.value == 3.0

    def test_add_zero_identity(self):
        a = SymbolicSens.constant(5.0)
        z = SymbolicSens.zero()
        assert a.add(z) is a
        assert z.add(a) is a

    def test_add_symbolic(self):
        a = SymbolicSens.from_var("x")
        b = SymbolicSens.constant(1.0)
        result = a.add(b)
        assert result.symbolic is not None
        assert "x" in result.depends_on

    def test_multiply_concrete(self):
        a = SymbolicSens.constant(3.0)
        b = SymbolicSens.constant(4.0)
        result = a.multiply(b)
        assert result.value == 12.0

    def test_multiply_by_zero(self):
        a = SymbolicSens.constant(5.0)
        z = SymbolicSens.zero()
        result = a.multiply(z)
        assert result.is_zero

    def test_multiply_symbolic(self):
        a = SymbolicSens.from_var("x")
        b = SymbolicSens.from_var("y")
        result = a.multiply(b)
        assert "x" in result.depends_on
        assert "y" in result.depends_on

    def test_scale(self):
        a = SymbolicSens.constant(3.0)
        result = a.scale(2.0)
        assert result.value == 6.0

    def test_scale_negative_uses_abs(self):
        a = SymbolicSens.constant(3.0)
        result = a.scale(-2.0)
        assert result.value == 6.0

    def test_scale_zero(self):
        a = SymbolicSens.zero()
        result = a.scale(100.0)
        assert result.is_zero

    def test_max_concrete(self):
        a = SymbolicSens.constant(3.0)
        b = SymbolicSens.constant(5.0)
        result = a.max(b)
        assert result.value == 5.0

    def test_max_with_zero(self):
        a = SymbolicSens.constant(3.0)
        z = SymbolicSens.zero()
        assert a.max(z) is a
        assert z.max(a) is a

    def test_max_symbolic(self):
        a = SymbolicSens.from_var("x")
        b = SymbolicSens.from_var("y")
        result = a.max(b)
        assert result.symbolic is not None
        assert "max" in result.symbolic

    def test_sqrt_concrete(self):
        a = SymbolicSens.constant(9.0)
        result = a.sqrt()
        assert abs(result.value - 3.0) < 1e-10

    def test_sqrt_zero(self):
        z = SymbolicSens.zero()
        result = z.sqrt()
        assert result.is_zero

    def test_sqrt_symbolic(self):
        a = SymbolicSens.from_var("x")
        result = a.sqrt()
        assert result.symbolic is not None

    def test_str_concrete(self):
        s = SymbolicSens.constant(2.5)
        assert "2.5" in str(s)

    def test_str_zero(self):
        s = SymbolicSens.zero()
        assert "0" in str(s)

    def test_str_unbounded(self):
        s = SymbolicSens.unbounded()
        assert "∞" in str(s)

    def test_str_symbolic(self):
        s = SymbolicSens.from_var("n")
        assert "n" in str(s)


# ═══════════════════════════════════════════════════════════════════════════
# SensitivityResult
# ═══════════════════════════════════════════════════════════════════════════


class TestSensitivityResult:
    """SensitivityResult dataclass."""

    def test_default(self):
        r = SensitivityResult()
        assert r.mechanism_name == ""
        assert r.global_sensitivity.is_zero
        assert r.norm == SensitivityNorm.L1

    def test_str_output(self):
        r = SensitivityResult(
            mechanism_name="test",
            global_sensitivity=SymbolicSens.constant(1.0),
        )
        s = str(r)
        assert "test" in s
        assert "1" in s

    def test_with_queries(self):
        qs = QuerySensitivity(
            query_name="count",
            node_id=0,
            sensitivity=SymbolicSens.constant(1.0),
        )
        r = SensitivityResult(
            mechanism_name="m",
            local_sensitivities=[qs],
        )
        assert len(r.local_sensitivities) == 1
        assert "count" in str(r)

    def test_with_noise(self):
        ni = NoiseInfo(
            target_var="eta",
            noise_kind=NoiseKind.LAPLACE,
            scale=SymbolicSens.constant(1.0),
        )
        r = SensitivityResult(
            mechanism_name="m",
            noise_info=[ni],
        )
        assert len(r.noise_info) == 1
        assert "eta" in str(ni)


# ═══════════════════════════════════════════════════════════════════════════
# SensitivityCert
# ═══════════════════════════════════════════════════════════════════════════


class TestSensitivityCert:
    """SensitivityCert construction and serialization."""

    def test_default(self):
        cert = SensitivityCert()
        assert cert.is_valid
        assert cert.mechanism_name == ""

    def test_to_dict(self):
        cert = SensitivityCert(
            mechanism_name="test",
            global_sensitivity=1.0,
            norm=SensitivityNorm.L1,
        )
        d = cert.to_dict()
        assert d["mechanism_name"] == "test"
        assert d["global_sensitivity"] == 1.0
        assert d["is_valid"]

    def test_from_result(self):
        result = SensitivityResult(
            mechanism_name="m",
            global_sensitivity=SymbolicSens.constant(1.0),
            norm=SensitivityNorm.L2,
            composition="sequential",
        )
        cert = SensitivityCert.from_result(result)
        assert cert.mechanism_name == "m"
        assert cert.global_sensitivity == 1.0
        assert cert.norm == SensitivityNorm.L2
        assert cert.composition_type == "sequential"

    def test_from_result_with_queries(self):
        qs = QuerySensitivity(
            query_name="count",
            node_id=0,
            sensitivity=SymbolicSens.constant(1.0),
        )
        result = SensitivityResult(
            mechanism_name="m",
            global_sensitivity=SymbolicSens.constant(1.0),
            local_sensitivities=[qs],
        )
        cert = SensitivityCert.from_result(result)
        assert len(cert.queries) == 1
        assert cert.queries[0]["query_name"] == "count"

    def test_from_result_symbolic_sensitivity(self):
        qs = QuerySensitivity(
            query_name="q",
            node_id=0,
            sensitivity=SymbolicSens.from_var("n"),
        )
        result = SensitivityResult(
            mechanism_name="m",
            global_sensitivity=SymbolicSens.from_var("n"),
            local_sensitivities=[qs],
        )
        cert = SensitivityCert.from_result(result)
        assert cert.global_sensitivity is None  # not concrete
        assert "sensitivity_expr" in cert.queries[0]


# ═══════════════════════════════════════════════════════════════════════════
# SensitivityAnalyzer — basic analysis
# ═══════════════════════════════════════════════════════════════════════════


class TestSensitivityAnalyzerBasic:
    """Core sensitivity analysis on simple mechanisms."""

    def test_single_query_sensitivity(self, laplace_mechir):
        result = analyze_sensitivity(laplace_mechir)
        assert result.mechanism_name == "lap_mech"
        assert len(result.local_sensitivities) == 1
        qs = result.local_sensitivities[0]
        assert qs.sensitivity.is_concrete
        assert qs.sensitivity.value == 1.0

    def test_noise_info_recorded(self, laplace_mechir):
        result = analyze_sensitivity(laplace_mechir)
        assert len(result.noise_info) == 1
        ni = result.noise_info[0]
        assert ni.target_var == "eta"
        assert ni.noise_kind is NoiseKind.LAPLACE

    def test_global_sensitivity_single_query(self, laplace_mechir):
        result = analyze_sensitivity(laplace_mechir)
        assert result.global_sensitivity.value == 1.0

    def test_no_query_zero_sensitivity(self, no_query_mechir):
        result = analyze_sensitivity(no_query_mechir)
        assert result.global_sensitivity.is_zero

    def test_db_param_has_sensitivity_1(self, laplace_mechir):
        analyzer = SensitivityAnalyzer()
        analyzer.analyze(laplace_mechir)
        assert analyzer._var_sens.get("db") is not None
        assert analyzer._var_sens["db"].value == 1.0

    def test_non_db_param_zero_sensitivity(self, laplace_mechir):
        analyzer = SensitivityAnalyzer()
        analyzer.analyze(laplace_mechir)
        assert analyzer._var_sens["epsilon"].is_zero


# ═══════════════════════════════════════════════════════════════════════════
# SensitivityAnalyzer — expression sensitivity
# ═══════════════════════════════════════════════════════════════════════════


class TestExpressionSensitivity:
    """_expr_sensitivity for various expression types."""

    def test_constant_zero_sens(self):
        analyzer = SensitivityAnalyzer()
        s = analyzer._expr_sensitivity(Const.real(1.0))
        assert s.is_zero

    def test_undefined_var_zero_sens(self):
        analyzer = SensitivityAnalyzer()
        s = analyzer._expr_sensitivity(Var(ty=IRType.REAL, name="unknown"))
        assert s.is_zero

    def test_tracked_var_sens(self):
        analyzer = SensitivityAnalyzer()
        analyzer._var_sens["q"] = SymbolicSens.constant(1.0)
        s = analyzer._expr_sensitivity(Var(ty=IRType.REAL, name="q"))
        assert s.value == 1.0

    def test_add_sensitivities(self):
        analyzer = SensitivityAnalyzer()
        analyzer._var_sens["a"] = SymbolicSens.constant(1.0)
        analyzer._var_sens["b"] = SymbolicSens.constant(2.0)
        expr = BinOp(
            ty=IRType.REAL, op=BinOpKind.ADD,
            left=Var(ty=IRType.REAL, name="a"),
            right=Var(ty=IRType.REAL, name="b"),
        )
        s = analyzer._expr_sensitivity(expr)
        assert s.value == 3.0

    def test_sub_sensitivities(self):
        analyzer = SensitivityAnalyzer()
        analyzer._var_sens["a"] = SymbolicSens.constant(1.0)
        analyzer._var_sens["b"] = SymbolicSens.constant(2.0)
        expr = BinOp(
            ty=IRType.REAL, op=BinOpKind.SUB,
            left=Var(ty=IRType.REAL, name="a"),
            right=Var(ty=IRType.REAL, name="b"),
        )
        s = analyzer._expr_sensitivity(expr)
        assert s.value == 3.0

    def test_mul_by_constant(self):
        analyzer = SensitivityAnalyzer()
        analyzer._var_sens["q"] = SymbolicSens.constant(1.0)
        expr = BinOp(
            ty=IRType.REAL, op=BinOpKind.MUL,
            left=Const.real(3.0),
            right=Var(ty=IRType.REAL, name="q"),
        )
        s = analyzer._expr_sensitivity(expr)
        assert s.value == 3.0

    def test_div_by_constant(self):
        analyzer = SensitivityAnalyzer()
        analyzer._var_sens["q"] = SymbolicSens.constant(6.0)
        expr = BinOp(
            ty=IRType.REAL, op=BinOpKind.DIV,
            left=Var(ty=IRType.REAL, name="q"),
            right=Const.real(2.0),
        )
        s = analyzer._expr_sensitivity(expr)
        assert s.value == 3.0

    def test_div_by_sensitive_warns(self):
        analyzer = SensitivityAnalyzer()
        analyzer._var_sens["a"] = SymbolicSens.constant(1.0)
        analyzer._var_sens["b"] = SymbolicSens.constant(1.0)
        expr = BinOp(
            ty=IRType.REAL, op=BinOpKind.DIV,
            left=Var(ty=IRType.REAL, name="a"),
            right=Var(ty=IRType.REAL, name="b"),
        )
        s = analyzer._expr_sensitivity(expr)
        assert not s.is_bounded
        assert len(analyzer._warnings) >= 1

    def test_negation_preserves_sens(self):
        analyzer = SensitivityAnalyzer()
        analyzer._var_sens["q"] = SymbolicSens.constant(1.0)
        expr = UnaryOp(
            ty=IRType.REAL, op=UnaryOpKind.NEG,
            operand=Var(ty=IRType.REAL, name="q"),
        )
        s = analyzer._expr_sensitivity(expr)
        assert s.value == 1.0

    def test_abs_func_preserves_sens(self):
        analyzer = SensitivityAnalyzer()
        analyzer._var_sens["q"] = SymbolicSens.constant(1.0)
        expr = FuncCall(
            ty=IRType.REAL, name="abs",
            args=(Var(ty=IRType.REAL, name="q"),),
        )
        s = analyzer._expr_sensitivity(expr)
        assert s.value == 1.0

    def test_len_func_constant_sens(self):
        analyzer = SensitivityAnalyzer()
        expr = FuncCall(
            ty=IRType.INT, name="len",
            args=(Var(ty=IRType.ARRAY, name="arr"),),
        )
        s = analyzer._expr_sensitivity(expr)
        assert s.value == 1.0

    def test_comparison_max_sens(self):
        analyzer = SensitivityAnalyzer()
        analyzer._var_sens["a"] = SymbolicSens.constant(1.0)
        analyzer._var_sens["b"] = SymbolicSens.constant(2.0)
        expr = BinOp(
            ty=IRType.BOOL, op=BinOpKind.GT,
            left=Var(ty=IRType.REAL, name="a"),
            right=Var(ty=IRType.REAL, name="b"),
        )
        s = analyzer._expr_sensitivity(expr)
        assert s.value == 2.0


# ═══════════════════════════════════════════════════════════════════════════
# SensitivityAnalyzer — L1/L2/Linf norms
# ═══════════════════════════════════════════════════════════════════════════


class TestNormComposition:
    """Sensitivity composition under different norms."""

    def test_l1_sequential_sums(self, two_query_mechir):
        result = analyze_sensitivity(two_query_mechir, norm=SensitivityNorm.L1)
        assert result.global_sensitivity.value == 3.0  # 1.0 + 2.0

    def test_l2_sequential_sqrt(self, two_query_mechir):
        result = analyze_sensitivity(two_query_mechir, norm=SensitivityNorm.L2)
        expected = math.sqrt(1.0**2 + 2.0**2)
        assert abs(result.global_sensitivity.value - expected) < 1e-10

    def test_linf_takes_max(self, two_query_mechir):
        result = analyze_sensitivity(two_query_mechir, norm=SensitivityNorm.LINF)
        assert result.global_sensitivity.value == 2.0


# ═══════════════════════════════════════════════════════════════════════════
# SensitivityAnalyzer — branch handling
# ═══════════════════════════════════════════════════════════════════════════


class TestBranchSensitivity:
    """Sensitivity through branches takes max of both sides."""

    def test_branch_takes_max(self):
        cond = BinOp(
            ty=IRType.BOOL, op=BinOpKind.GT,
            left=Const.real(1.0), right=Const.real(0.0),
        )
        true_br = AssignNode(
            target=Var(ty=IRType.REAL, name="x"),
            value=Const.real(10.0),
        )
        false_br = AssignNode(
            target=Var(ty=IRType.REAL, name="x"),
            value=Const.real(20.0),
        )
        branch = BranchNode(
            condition=cond,
            true_branch=true_br,
            false_branch=false_br,
        )
        ret = ReturnNode(value=Var(ty=IRType.REAL, name="x"))
        mechir = MechIR(
            name="m",
            params=[ParamDecl(name="db", ty=IRType.ARRAY, is_database=True)],
            body=SequenceNode(stmts=[branch, ret]),
        )
        result = analyze_sensitivity(mechir)
        # No queries means zero global sensitivity
        assert result.global_sensitivity.is_zero


# ═══════════════════════════════════════════════════════════════════════════
# SensitivityAnalyzer — noise draw handling
# ═══════════════════════════════════════════════════════════════════════════


class TestNoiseDrawSensitivity:
    """NoiseDrawNode resets sensitivity to zero (after noise)."""

    def test_noise_zeroes_sensitivity(self, laplace_mechir):
        analyzer = SensitivityAnalyzer()
        analyzer.analyze(laplace_mechir)
        eta_sens = analyzer._var_sens.get("eta")
        assert eta_sens is not None
        assert eta_sens.is_zero

    def test_noise_info_scale(self, laplace_mechir):
        result = analyze_sensitivity(laplace_mechir)
        ni = result.noise_info[0]
        assert ni.scale.value == 1.0


# ═══════════════════════════════════════════════════════════════════════════
# SensitivityAnalyzer — annotation handling
# ═══════════════════════════════════════════════════════════════════════════


class TestAnnotationHandling:
    """Sensitivity annotations from node attributes."""

    def test_query_sensitivity_annotation(self):
        q = QueryNode(
            target=Var(ty=IRType.REAL, name="q"),
            query_name="count",
            args=(),
            sensitivity=Const.real(5.0),
        )
        noise = NoiseDrawNode(
            target=Var(ty=IRType.REAL, name="eta"),
            noise_kind=NoiseKind.LAPLACE,
            center=Var(ty=IRType.REAL, name="q"),
            scale=Const.real(5.0),
        )
        ret = ReturnNode(value=Var(ty=IRType.REAL, name="eta"))
        mechir = MechIR(
            name="m",
            params=[ParamDecl(name="db", ty=IRType.ARRAY, is_database=True)],
            body=SequenceNode(stmts=[q, noise, ret]),
        )
        result = analyze_sensitivity(mechir)
        assert result.local_sensitivities[0].sensitivity.value == 5.0


# ═══════════════════════════════════════════════════════════════════════════
# Certificate generation
# ═══════════════════════════════════════════════════════════════════════════


class TestCertificateGeneration:
    """generate_sensitivity_certificate and SensitivityAnalyzer.generate_certificate."""

    def test_public_api(self, laplace_mechir):
        cert = generate_sensitivity_certificate(laplace_mechir)
        assert isinstance(cert, SensitivityCert)
        assert cert.mechanism_name == "lap_mech"
        assert cert.global_sensitivity == 1.0

    def test_analyzer_generate(self, laplace_mechir):
        analyzer = SensitivityAnalyzer(norm=SensitivityNorm.L1)
        cert = analyzer.generate_certificate(laplace_mechir)
        assert cert.is_valid
        assert cert.global_sensitivity == 1.0

    def test_cert_dict_round_trip(self, laplace_mechir):
        cert = generate_sensitivity_certificate(laplace_mechir)
        d = cert.to_dict()
        assert isinstance(d, dict)
        assert d["mechanism_name"] == "lap_mech"

    def test_cert_with_norm(self, laplace_mechir):
        cert = generate_sensitivity_certificate(
            laplace_mechir, norm=SensitivityNorm.L2,
        )
        assert cert.norm == SensitivityNorm.L2


# ═══════════════════════════════════════════════════════════════════════════
# Composition helpers
# ═══════════════════════════════════════════════════════════════════════════


class TestCompositionHelpers:
    """sequential_compose and parallel_compose free functions."""

    def test_sequential_l1(self):
        assert sequential_compose([1.0, 2.0, 3.0], SensitivityNorm.L1) == 6.0

    def test_sequential_l2(self):
        result = sequential_compose([3.0, 4.0], SensitivityNorm.L2)
        assert abs(result - 5.0) < 1e-10

    def test_sequential_linf(self):
        assert sequential_compose([1.0, 5.0, 3.0], SensitivityNorm.LINF) == 5.0

    def test_sequential_empty(self):
        assert sequential_compose([], SensitivityNorm.L1) == 0.0

    def test_parallel_compose(self):
        assert parallel_compose([1.0, 5.0, 3.0]) == 5.0

    def test_parallel_compose_empty(self):
        assert parallel_compose([]) == 0.0

    def test_parallel_compose_single(self):
        assert parallel_compose([2.0]) == 2.0


# ═══════════════════════════════════════════════════════════════════════════
# SensitivityAnalyzer — composition mode detection
# ═══════════════════════════════════════════════════════════════════════════


class TestCompositionDetection:
    """_detect_composition_mode heuristic."""

    def test_single_query_sequential(self, laplace_mechir):
        result = analyze_sensitivity(laplace_mechir)
        assert result.composition == "sequential"

    def test_loop_forces_sequential(self, loop_mechir):
        result = analyze_sensitivity(loop_mechir)
        assert result.composition == "sequential"


# ═══════════════════════════════════════════════════════════════════════════
# QuerySensitivity / NoiseInfo dataclasses
# ═══════════════════════════════════════════════════════════════════════════


class TestDataclasses:
    """QuerySensitivity and NoiseInfo repr / str."""

    def test_query_sensitivity_str(self):
        qs = QuerySensitivity(
            query_name="count",
            node_id=42,
            sensitivity=SymbolicSens.constant(1.0),
            norm=SensitivityNorm.L1,
        )
        s = str(qs)
        assert "count" in s
        assert "1" in s

    def test_noise_info_str(self):
        ni = NoiseInfo(
            target_var="eta",
            noise_kind=NoiseKind.LAPLACE,
            scale=SymbolicSens.constant(1.0),
            query_sens=SymbolicSens.constant(1.0),
        )
        s = str(ni)
        assert "eta" in s
        assert "laplace" in s.lower() or "Lap" in s or "LAPLACE" in s

    def test_noise_info_no_query_sens(self):
        ni = NoiseInfo(
            target_var="eta",
            noise_kind=NoiseKind.GAUSSIAN,
            scale=SymbolicSens.constant(2.0),
        )
        s = str(ni)
        assert "eta" in s
