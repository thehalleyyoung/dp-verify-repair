"""Unit tests for dpcegar.paths.feasibility — FeasibilityChecker."""

from __future__ import annotations

import pytest

from dpcegar.ir.types import (
    BinOp,
    BinOpKind,
    Const,
    IRType,
    UnaryOp,
    UnaryOpKind,
    Var,
)
from dpcegar.paths.feasibility import FeasibilityChecker, FeasibilityResult
from dpcegar.paths.symbolic_path import PathCondition


# ═══════════════════════════════════════════════════════════════════════════
# Construction
# ═══════════════════════════════════════════════════════════════════════════


class TestFeasibilityCheckerConstruction:
    """FeasibilityChecker can be instantiated with default and custom args."""

    def test_default_construction(self):
        fc = FeasibilityChecker()
        assert fc.stats["total_checks"] == 0

    def test_construction_with_max_cache(self):
        fc = FeasibilityChecker(max_cache_size=5)
        assert fc.stats["cache_size"] == 0

    def test_reset_stats(self):
        fc = FeasibilityChecker()
        fc.check(PathCondition.trivially_true())
        assert fc.stats["total_checks"] > 0
        fc.reset_stats()
        assert fc.stats["total_checks"] == 0

    def test_clear_cache(self):
        fc = FeasibilityChecker()
        cond = _make_simple_condition("x", BinOpKind.GT, 0.0)
        fc.check(cond)
        fc.clear_cache()
        assert fc.stats["cache_size"] == 0


# ═══════════════════════════════════════════════════════════════════════════
# Satisfiable path conditions
# ═══════════════════════════════════════════════════════════════════════════


class TestFeasibilityCheckerSatisfiable:
    """Satisfiable conditions should be marked feasible."""

    def test_trivially_true(self):
        fc = FeasibilityChecker()
        result = fc.check(PathCondition.trivially_true())
        assert result.is_feasible

    def test_simple_gt_zero(self):
        fc = FeasibilityChecker()
        cond = _make_simple_condition("x", BinOpKind.GT, 0.0)
        result = fc.check(cond)
        assert result.is_feasible

    def test_compatible_bounds(self):
        """x > 0 ∧ x < 10 is satisfiable."""
        fc = FeasibilityChecker()
        c1 = BinOp(
            ty=IRType.BOOL, op=BinOpKind.GT,
            left=Var(ty=IRType.REAL, name="x"),
            right=Const.real(0.0),
        )
        c2 = BinOp(
            ty=IRType.BOOL, op=BinOpKind.LT,
            left=Var(ty=IRType.REAL, name="x"),
            right=Const.real(10.0),
        )
        cond = PathCondition(conjuncts=[c1, c2])
        result = fc.check(cond)
        assert result.is_feasible


# ═══════════════════════════════════════════════════════════════════════════
# Unsatisfiable path conditions
# ═══════════════════════════════════════════════════════════════════════════


class TestFeasibilityCheckerUnsatisfiable:
    """Unsatisfiable conditions should be marked infeasible."""

    def test_trivially_false(self):
        fc = FeasibilityChecker()
        cond = PathCondition(conjuncts=[Const.bool_(False)])
        result = fc.check(cond)
        assert not result.is_feasible

    def test_false_constant(self):
        fc = FeasibilityChecker()
        cond = PathCondition(conjuncts=[Const.bool_(False)])
        result = fc.check(cond)
        assert not result.is_feasible

    def test_syntactic_contradiction(self):
        """x > 0 ∧ ¬(x > 0) is unsatisfiable."""
        fc = FeasibilityChecker()
        pos = BinOp(
            ty=IRType.BOOL, op=BinOpKind.GT,
            left=Var(ty=IRType.REAL, name="x"),
            right=Const.real(0.0),
        )
        neg = UnaryOp(ty=IRType.BOOL, op=UnaryOpKind.NOT, operand=pos)
        cond = PathCondition(conjuncts=[pos, neg])
        result = fc.check(cond)
        assert not result.is_feasible

    def test_conflicting_bounds(self):
        """x > 10 ∧ x < 5 is unsatisfiable (bounds conflict)."""
        fc = FeasibilityChecker()
        c1 = BinOp(
            ty=IRType.BOOL, op=BinOpKind.GT,
            left=Var(ty=IRType.REAL, name="x"),
            right=Const.real(10.0),
        )
        c2 = BinOp(
            ty=IRType.BOOL, op=BinOpKind.LT,
            left=Var(ty=IRType.REAL, name="x"),
            right=Const.real(5.0),
        )
        cond = PathCondition(conjuncts=[c1, c2])
        result = fc.check(cond)
        assert not result.is_feasible


# ═══════════════════════════════════════════════════════════════════════════
# Cache behaviour
# ═══════════════════════════════════════════════════════════════════════════


class TestFeasibilityCheckerCache:
    """Repeated checks should hit cache."""

    def test_cache_hit(self):
        fc = FeasibilityChecker()
        cond = _make_simple_condition("x", BinOpKind.GT, 0.0)
        fc.check(cond)
        fc.check(cond)
        assert fc.stats["cache_hits"] >= 1

    def test_cache_eviction(self):
        """With max_cache_size=2, third distinct check evicts oldest."""
        fc = FeasibilityChecker(max_cache_size=2)
        for val in [1.0, 2.0, 3.0]:
            fc.check(_make_simple_condition("x", BinOpKind.GT, val))
        assert fc.stats["cache_size"] <= 2


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _make_simple_condition(
    var_name: str, op: BinOpKind, value: float
) -> PathCondition:
    expr = BinOp(
        ty=IRType.BOOL, op=op,
        left=Var(ty=IRType.REAL, name=var_name),
        right=Const.real(value),
    )
    return PathCondition(conjuncts=[expr])
