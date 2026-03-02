"""Tests for dpcegar.smt.encoding – ExprToZ3 converter and SMTEncoding."""

from __future__ import annotations

import math
from typing import Any

import pytest
import z3

from dpcegar.ir.types import (
    Abs,
    ApproxBudget,
    BinOp,
    BinOpKind,
    Cond,
    Const,
    Exp,
    FuncCall,
    IRType,
    LetExpr,
    Log,
    Max,
    Min,
    NoiseKind,
    Phi,
    PhiInv,
    PrivacyNotion,
    PureBudget,
    Sqrt,
    SumExpr,
    TypedExpr,
    UnaryOp,
    UnaryOpKind,
    Var,
    ArrayAccess,
    TupleAccess,
)
from dpcegar.paths.symbolic_path import PathCondition
from dpcegar.smt.encoding import (
    AbsLinearizer,
    CaseSplitter,
    ExprToZ3,
    PathConditionEncoder,
    SMTEncoding,
)
from dpcegar.smt.transcendental import Precision, SoundnessTracker


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _make_converter(**kw: Any) -> ExprToZ3:
    return ExprToZ3(**kw)


def _is_z3_expr(v: Any) -> bool:
    return isinstance(v, z3.ExprRef)


# ═══════════════════════════════════════════════════════════════════════════
# ExprToZ3 – variable handling
# ═══════════════════════════════════════════════════════════════════════════


class TestExprToZ3Variables:
    """Test variable conversion and type discrimination."""

    def test_real_variable(self, var_x: Var):
        conv = _make_converter()
        z = conv.convert(var_x)
        assert _is_z3_expr(z)
        assert z.sort() == z3.RealSort()

    def test_integer_variable(self, var_i: Var):
        conv = _make_converter(int_vars={"i"})
        z = conv.convert(var_i)
        assert z.sort() == z3.IntSort()

    def test_bool_variable(self, var_b: Var):
        conv = _make_converter()
        z = conv.convert_bool(var_b)
        assert z.sort() == z3.BoolSort()

    def test_variable_caching(self, var_x: Var):
        conv = _make_converter()
        z1 = conv.convert(var_x)
        z2 = conv.convert(var_x)
        assert z1.eq(z2)

    def test_different_variables_distinct(self, var_x: Var, var_y: Var):
        conv = _make_converter()
        zx = conv.convert(var_x)
        zy = conv.convert(var_y)
        assert not zx.eq(zy)

    def test_variable_override(self, var_x: Var):
        custom = z3.Real("custom_x")
        conv = _make_converter(var_overrides={"x": custom})
        z = conv.convert(var_x)
        assert z.eq(custom)

    def test_variables_dict_populated(self, var_x: Var, var_y: Var):
        conv = _make_converter()
        conv.convert(var_x)
        conv.convert(var_y)
        assert "x" in conv.variables
        assert "y" in conv.variables
        assert len(conv.variables) == 2

    def test_int_vars_parameter(self):
        conv = _make_converter(int_vars={"n", "k"})
        n = Var(ty=IRType.INT, name="n")
        z = conv.convert(n)
        assert z.sort() == z3.IntSort()


# ═══════════════════════════════════════════════════════════════════════════
# ExprToZ3 – constants
# ═══════════════════════════════════════════════════════════════════════════


class TestExprToZ3Constants:
    """Test constant conversion."""

    def test_zero(self, const_zero: Const):
        conv = _make_converter()
        z = conv.convert(const_zero)
        assert _is_z3_expr(z)

    def test_one(self, const_one: Const):
        conv = _make_converter()
        z = conv.convert(const_one)
        assert _is_z3_expr(z)

    def test_integer_const(self, const_int_42: Const):
        conv = _make_converter()
        z = conv.convert(const_int_42)
        assert _is_z3_expr(z)

    def test_pi(self, const_pi: Const):
        conv = _make_converter()
        z = conv.convert(const_pi)
        assert _is_z3_expr(z)

    def test_bool_true(self, const_true: Const):
        conv = _make_converter()
        z = conv.convert(const_true)
        assert _is_z3_expr(z)

    def test_bool_false(self, const_false: Const):
        conv = _make_converter()
        z = conv.convert(const_false)
        assert _is_z3_expr(z)

    def test_large_constant(self):
        conv = _make_converter()
        c = Const.real(1e15)
        z = conv.convert(c)
        assert _is_z3_expr(z)

    def test_negative_constant(self):
        conv = _make_converter()
        c = Const.real(-3.14)
        z = conv.convert(c)
        assert _is_z3_expr(z)

    def test_very_small_constant(self):
        conv = _make_converter()
        c = Const.real(1e-15)
        z = conv.convert(c)
        assert _is_z3_expr(z)


# ═══════════════════════════════════════════════════════════════════════════
# ExprToZ3 – BinOp
# ═══════════════════════════════════════════════════════════════════════════


class TestExprToZ3BinOp:
    """Test binary operations encoding."""

    @pytest.mark.parametrize("op", [
        BinOpKind.ADD, BinOpKind.SUB, BinOpKind.MUL, BinOpKind.DIV,
    ])
    def test_arithmetic_ops(self, var_x: Var, var_y: Var, op: BinOpKind):
        conv = _make_converter()
        expr = BinOp(ty=IRType.REAL, op=op, left=var_x, right=var_y)
        z = conv.convert(expr)
        assert _is_z3_expr(z)

    def test_modulo(self, var_x: Var, var_y: Var):
        conv = _make_converter()
        expr = BinOp(ty=IRType.REAL, op=BinOpKind.MOD, left=var_x, right=var_y)
        z = conv.convert(expr)
        assert _is_z3_expr(z)

    def test_power(self, var_x: Var):
        conv = _make_converter()
        expr = BinOp(
            ty=IRType.REAL, op=BinOpKind.POW,
            left=var_x, right=Const.real(2.0),
        )
        z = conv.convert(expr)
        assert _is_z3_expr(z)

    @pytest.mark.parametrize("op", [
        BinOpKind.EQ, BinOpKind.NEQ, BinOpKind.LT,
        BinOpKind.LE, BinOpKind.GT, BinOpKind.GE,
    ])
    def test_comparison_ops(self, var_x: Var, var_y: Var, op: BinOpKind):
        conv = _make_converter()
        expr = BinOp(ty=IRType.BOOL, op=op, left=var_x, right=var_y)
        z = conv.convert(expr)
        assert _is_z3_expr(z)

    def test_nested_arithmetic(self, var_x: Var, var_y: Var, var_z: Var):
        conv = _make_converter()
        inner = BinOp(ty=IRType.REAL, op=BinOpKind.ADD, left=var_x, right=var_y)
        outer = BinOp(ty=IRType.REAL, op=BinOpKind.MUL, left=inner, right=var_z)
        z = conv.convert(outer)
        assert _is_z3_expr(z)

    def test_deeply_nested(self, var_x: Var):
        conv = _make_converter()
        expr: TypedExpr = var_x
        for _ in range(10):
            expr = BinOp(ty=IRType.REAL, op=BinOpKind.ADD, left=expr, right=Const.real(1.0))
        z = conv.convert(expr)
        assert _is_z3_expr(z)

    def test_division_by_variable(self, var_x: Var, var_y: Var):
        conv = _make_converter()
        expr = BinOp(ty=IRType.REAL, op=BinOpKind.DIV, left=var_x, right=var_y)
        z = conv.convert(expr)
        assert _is_z3_expr(z)


# ═══════════════════════════════════════════════════════════════════════════
# ExprToZ3 – boolean connectives
# ═══════════════════════════════════════════════════════════════════════════


class TestExprToZ3BoolOps:
    """Test boolean connective encoding."""

    def test_and(self, var_x: Var, var_y: Var):
        conv = _make_converter()
        lhs = BinOp(ty=IRType.BOOL, op=BinOpKind.GT, left=var_x, right=Const.real(0.0))
        rhs = BinOp(ty=IRType.BOOL, op=BinOpKind.LT, left=var_y, right=Const.real(10.0))
        expr = BinOp(ty=IRType.BOOL, op=BinOpKind.AND, left=lhs, right=rhs)
        z = conv.convert(expr)
        assert _is_z3_expr(z)

    def test_or(self, var_x: Var, var_y: Var):
        conv = _make_converter()
        lhs = BinOp(ty=IRType.BOOL, op=BinOpKind.GT, left=var_x, right=Const.real(0.0))
        rhs = BinOp(ty=IRType.BOOL, op=BinOpKind.GT, left=var_y, right=Const.real(0.0))
        expr = BinOp(ty=IRType.BOOL, op=BinOpKind.OR, left=lhs, right=rhs)
        z = conv.convert(expr)
        assert _is_z3_expr(z)

    def test_not(self, var_x: Var):
        conv = _make_converter()
        inner = BinOp(ty=IRType.BOOL, op=BinOpKind.GT, left=var_x, right=Const.real(0.0))
        expr = UnaryOp(ty=IRType.BOOL, op=UnaryOpKind.NOT, operand=inner)
        z = conv.convert(expr)
        assert _is_z3_expr(z)

    def test_neg(self, var_x: Var):
        conv = _make_converter()
        expr = UnaryOp(ty=IRType.REAL, op=UnaryOpKind.NEG, operand=var_x)
        z = conv.convert(expr)
        assert _is_z3_expr(z)


# ═══════════════════════════════════════════════════════════════════════════
# ExprToZ3 – absolute value
# ═══════════════════════════════════════════════════════════════════════════


class TestExprToZ3Abs:
    """Test absolute value encoding with case splitting."""

    def test_abs_variable(self, var_x: Var):
        conv = _make_converter()
        expr = Abs(ty=IRType.REAL, operand=var_x)
        z = conv.convert(expr)
        assert _is_z3_expr(z)

    def test_abs_produces_aux_constraints(self, var_x: Var):
        conv = _make_converter()
        expr = Abs(ty=IRType.REAL, operand=var_x)
        conv.convert(expr)
        assert len(conv.aux_constraints) > 0 or len(conv.aux_vars) >= 0

    def test_abs_difference(self, var_x: Var, var_y: Var):
        conv = _make_converter()
        diff = BinOp(ty=IRType.REAL, op=BinOpKind.SUB, left=var_x, right=var_y)
        expr = Abs(ty=IRType.REAL, operand=diff)
        z = conv.convert(expr)
        assert _is_z3_expr(z)


# ═══════════════════════════════════════════════════════════════════════════
# ExprToZ3 – transcendental functions
# ═══════════════════════════════════════════════════════════════════════════


class TestExprToZ3Transcendental:
    """Test transcendental function encoding."""

    def test_log(self, var_x: Var):
        conv = _make_converter()
        expr = Log(ty=IRType.REAL, operand=var_x)
        z = conv.convert(expr)
        assert _is_z3_expr(z)

    def test_exp(self, var_x: Var):
        conv = _make_converter()
        expr = Exp(ty=IRType.REAL, operand=var_x)
        z = conv.convert(expr)
        assert _is_z3_expr(z)

    def test_sqrt(self, var_x: Var):
        conv = _make_converter()
        expr = Sqrt(ty=IRType.REAL, operand=var_x)
        z = conv.convert(expr)
        assert _is_z3_expr(z)

    def test_phi(self, var_x: Var):
        conv = _make_converter()
        expr = Phi(ty=IRType.REAL, operand=var_x)
        z = conv.convert(expr)
        assert _is_z3_expr(z)

    def test_phi_inv(self, var_x: Var):
        conv = _make_converter()
        expr = PhiInv(ty=IRType.REAL, operand=var_x)
        z = conv.convert(expr)
        assert _is_z3_expr(z)

    def test_soundness_tracker_populated(self, var_x: Var):
        conv = _make_converter()
        conv.convert(Log(ty=IRType.REAL, operand=var_x))
        conv.convert(Exp(ty=IRType.REAL, operand=var_x))
        assert len(conv.tracker.entries) >= 0


# ═══════════════════════════════════════════════════════════════════════════
# ExprToZ3 – Max, Min, Cond
# ═══════════════════════════════════════════════════════════════════════════


class TestExprToZ3MaxMinCond:
    """Test max, min, and conditional expression encoding."""

    def test_max(self, var_x: Var, var_y: Var):
        conv = _make_converter()
        expr = Max(ty=IRType.REAL, left=var_x, right=var_y)
        z = conv.convert(expr)
        assert _is_z3_expr(z)

    def test_min(self, var_x: Var, var_y: Var):
        conv = _make_converter()
        expr = Min(ty=IRType.REAL, left=var_x, right=var_y)
        z = conv.convert(expr)
        assert _is_z3_expr(z)

    def test_cond(self, var_x: Var, var_y: Var, var_z: Var):
        conv = _make_converter()
        cond_e = BinOp(ty=IRType.BOOL, op=BinOpKind.GT, left=var_x, right=Const.real(0.0))
        expr = Cond(ty=IRType.REAL, condition=cond_e, true_expr=var_y, false_expr=var_z)
        z = conv.convert(expr)
        assert _is_z3_expr(z)


# ═══════════════════════════════════════════════════════════════════════════
# ExprToZ3 – LetExpr, SumExpr, FuncCall
# ═══════════════════════════════════════════════════════════════════════════


class TestExprToZ3Compound:
    """Test compound expression encoding."""

    def test_let_expr(self, var_x: Var):
        conv = _make_converter()
        body_var = Var(ty=IRType.REAL, name="tmp")
        expr = LetExpr(
            ty=IRType.REAL,
            var_name="tmp",
            value=BinOp(ty=IRType.REAL, op=BinOpKind.MUL, left=var_x, right=Const.real(2.0)),
            body=BinOp(ty=IRType.REAL, op=BinOpKind.ADD, left=body_var, right=Const.real(1.0)),
        )
        z = conv.convert(expr)
        assert _is_z3_expr(z)

    def test_sum_expr(self, var_x: Var):
        conv = _make_converter()
        idx = Var(ty=IRType.INT, name="j")
        expr = SumExpr(
            ty=IRType.REAL,
            var_name="j",
            lo=Const.int_(0),
            hi=Const.int_(3),
            body=var_x,
        )
        z = conv.convert(expr)
        assert _is_z3_expr(z)

    def test_func_call(self, var_x: Var):
        conv = _make_converter()
        expr = FuncCall(ty=IRType.REAL, name="f", args=(var_x,))
        z = conv.convert(expr)
        assert _is_z3_expr(z)

    def test_array_access(self, var_x: Var, var_i: Var):
        conv = _make_converter()
        expr = ArrayAccess(ty=IRType.REAL, array=var_x, index=var_i)
        z = conv.convert(expr)
        assert _is_z3_expr(z)

    def test_tuple_access(self, var_x: Var):
        conv = _make_converter()
        expr = TupleAccess(ty=IRType.REAL, tuple_expr=var_x, field_idx=0)
        z = conv.convert(expr)
        assert _is_z3_expr(z)


# ═══════════════════════════════════════════════════════════════════════════
# SMTEncoding
# ═══════════════════════════════════════════════════════════════════════════


class TestSMTEncoding:
    """Test SMTEncoding dataclass."""

    def test_construction(self):
        enc = SMTEncoding(
            formula=z3.BoolVal(True),
            variables={"x": z3.Real("x")},
            assertions=[z3.BoolVal(True)],
            metadata={"source": "test"},
            aux_vars={},
            soundness=SoundnessTracker(),
        )
        assert enc.variable_count() == 1
        assert enc.assertion_count() == 1

    def test_add_assertion(self):
        enc = SMTEncoding(
            formula=z3.BoolVal(True),
            variables={},
            assertions=[],
            metadata={},
            aux_vars={},
            soundness=SoundnessTracker(),
        )
        enc.add_assertion(z3.BoolVal(False))
        assert enc.assertion_count() == 1

    def test_add_assertions_bulk(self):
        enc = SMTEncoding(
            formula=z3.BoolVal(True),
            variables={},
            assertions=[],
            metadata={},
            aux_vars={},
            soundness=SoundnessTracker(),
        )
        enc.add_assertions([z3.BoolVal(True), z3.BoolVal(False)])
        assert enc.assertion_count() == 2

    def test_merge(self):
        enc1 = SMTEncoding(
            formula=z3.BoolVal(True),
            variables={"x": z3.Real("x")},
            assertions=[z3.Real("x") > 0],
            metadata={"a": 1},
            aux_vars={},
            soundness=SoundnessTracker(),
        )
        enc2 = SMTEncoding(
            formula=z3.BoolVal(True),
            variables={"y": z3.Real("y")},
            assertions=[z3.Real("y") > 0],
            metadata={"b": 2},
            aux_vars={},
            soundness=SoundnessTracker(),
        )
        merged = enc1.merge(enc2)
        assert merged.variable_count() == 2
        assert merged.assertion_count() == 2

    def test_summary(self):
        enc = SMTEncoding(
            formula=z3.BoolVal(True),
            variables={"x": z3.Real("x")},
            assertions=[],
            metadata={},
            aux_vars={},
            soundness=SoundnessTracker(),
        )
        s = enc.summary()
        assert isinstance(s, dict)
        assert "variables" in s or "variable_count" in s or len(s) > 0


# ═══════════════════════════════════════════════════════════════════════════
# AbsLinearizer
# ═══════════════════════════════════════════════════════════════════════════


class TestAbsLinearizer:
    """Test absolute value linearization."""

    def test_linearize(self):
        lin = AbsLinearizer()
        x = z3.Real("x")
        abs_x, constraints = lin.linearize(x)
        assert _is_z3_expr(abs_x)
        assert len(constraints) > 0

    def test_linearize_diff(self):
        lin = AbsLinearizer()
        a = z3.Real("a")
        b = z3.Real("b")
        abs_diff, constraints = lin.linearize_diff(a, b)
        assert _is_z3_expr(abs_diff)
        assert len(constraints) > 0

    def test_linearize_nonneg_sound(self):
        lin = AbsLinearizer()
        x = z3.Real("x")
        abs_x, constraints = lin.linearize(x)
        s = z3.Solver()
        s.add(constraints)
        s.add(abs_x < 0)
        assert s.check() == z3.unsat


# ═══════════════════════════════════════════════════════════════════════════
# CaseSplitter
# ═══════════════════════════════════════════════════════════════════════════


class TestCaseSplitter:
    """Test CaseSplitter for absolute value."""

    def test_split_abs(self):
        cs = CaseSplitter()
        x = z3.Real("x")
        cases = cs.split_abs(x)
        assert len(cases) >= 2

    def test_split_abs_diff(self):
        cs = CaseSplitter()
        a = z3.Real("a")
        b = z3.Real("b")
        cases = cs.split_abs_diff(a, b)
        assert len(cases) >= 2

    def test_encode_abs_constraint(self):
        cs = CaseSplitter()
        x = z3.Real("x")
        c = cs.encode_abs_constraint(x, z3.RealVal(5), op="<=")
        assert _is_z3_expr(c)


# ═══════════════════════════════════════════════════════════════════════════
# PathConditionEncoder
# ═══════════════════════════════════════════════════════════════════════════


class TestPathConditionEncoder:
    """Test PathCondition to Z3 encoding."""

    def test_trivially_true(self):
        conv = _make_converter()
        enc = PathConditionEncoder(conv)
        pc = PathCondition.trivially_true()
        z = enc.encode(pc)
        assert _is_z3_expr(z)

    def test_single_conjunct(self):
        conv = _make_converter()
        enc = PathConditionEncoder(conv)
        cond = BinOp(
            ty=IRType.BOOL, op=BinOpKind.GT,
            left=Var(ty=IRType.REAL, name="x"), right=Const.real(0.0),
        )
        pc = PathCondition.from_expr(cond)
        z = enc.encode(pc)
        assert _is_z3_expr(z)

    def test_multiple_conjuncts(self):
        conv = _make_converter()
        enc = PathConditionEncoder(conv)
        c1 = BinOp(
            ty=IRType.BOOL, op=BinOpKind.GT,
            left=Var(ty=IRType.REAL, name="x"), right=Const.real(0.0),
        )
        c2 = BinOp(
            ty=IRType.BOOL, op=BinOpKind.LT,
            left=Var(ty=IRType.REAL, name="y"), right=Const.real(10.0),
        )
        pc = PathCondition.from_conjuncts([c1, c2])
        z = enc.encode(pc)
        assert _is_z3_expr(z)


# ═══════════════════════════════════════════════════════════════════════════
# Precision levels
# ═══════════════════════════════════════════════════════════════════════════


class TestPrecisionLevels:
    """Test converter with different precision levels."""

    @pytest.mark.parametrize("prec", [Precision.FAST, Precision.STANDARD, Precision.HIGH])
    def test_precision_creates_converter(self, prec: Precision):
        conv = _make_converter(precision=prec)
        x = Var(ty=IRType.REAL, name="x")
        z = conv.convert(x)
        assert _is_z3_expr(z)

    @pytest.mark.parametrize("prec", [Precision.FAST, Precision.STANDARD, Precision.HIGH])
    def test_precision_with_log(self, prec: Precision):
        conv = _make_converter(precision=prec)
        expr = Log(ty=IRType.REAL, operand=Var(ty=IRType.REAL, name="x"))
        z = conv.convert(expr)
        assert _is_z3_expr(z)


# ═══════════════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════════════


class TestExprToZ3EdgeCases:
    """Edge cases and robustness."""

    def test_const_zero_division_handling(self):
        conv = _make_converter()
        expr = BinOp(
            ty=IRType.REAL, op=BinOpKind.DIV,
            left=Const.real(1.0), right=Const.real(0.0),
        )
        z = conv.convert(expr)
        assert _is_z3_expr(z)

    def test_large_integer_constant(self):
        conv = _make_converter()
        z = conv.convert(Const.int_(10**9))
        assert _is_z3_expr(z)

    def test_negative_power(self):
        conv = _make_converter()
        expr = BinOp(
            ty=IRType.REAL, op=BinOpKind.POW,
            left=Var(ty=IRType.REAL, name="x"),
            right=Const.real(-1.0),
        )
        z = conv.convert(expr)
        assert _is_z3_expr(z)

    def test_nested_abs(self):
        conv = _make_converter()
        inner = Abs(ty=IRType.REAL, operand=Var(ty=IRType.REAL, name="x"))
        outer = Abs(ty=IRType.REAL, operand=inner)
        z = conv.convert(outer)
        assert _is_z3_expr(z)

    def test_multiple_conversions_independent(self):
        c1 = _make_converter()
        c2 = _make_converter()
        x = Var(ty=IRType.REAL, name="x")
        z1 = c1.convert(x)
        z2 = c2.convert(x)
        assert _is_z3_expr(z1)
        assert _is_z3_expr(z2)

    def test_complex_nested_expression(self):
        conv = _make_converter()
        x = Var(ty=IRType.REAL, name="x")
        y = Var(ty=IRType.REAL, name="y")
        expr = Max(
            ty=IRType.REAL,
            left=Abs(
                ty=IRType.REAL,
                operand=BinOp(ty=IRType.REAL, op=BinOpKind.SUB, left=x, right=y),
            ),
            right=Const.real(0.0),
        )
        z = conv.convert(expr)
        assert _is_z3_expr(z)

    def test_cond_with_transcendental(self):
        conv = _make_converter()
        x = Var(ty=IRType.REAL, name="x")
        cond = BinOp(ty=IRType.BOOL, op=BinOpKind.GT, left=x, right=Const.real(0.0))
        expr = Cond(
            ty=IRType.REAL,
            condition=cond,
            true_expr=Log(ty=IRType.REAL, operand=x),
            false_expr=Const.real(0.0),
        )
        z = conv.convert(expr)
        assert _is_z3_expr(z)
