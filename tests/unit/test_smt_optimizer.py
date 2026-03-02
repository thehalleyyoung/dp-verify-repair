"""Tests for dpcegar.smt.optimizer – OMTSolver, linear objectives, soft constraints."""

from __future__ import annotations

import pytest
import z3

from dpcegar.smt.encoding import SMTEncoding
from dpcegar.smt.optimizer import (
    LinearObjective,
    OMTResult,
    OMTSolver,
    SoftConstraint,
)
from dpcegar.smt.solver import CheckResult
from dpcegar.smt.transcendental import SoundnessTracker


# ═══════════════════════════════════════════════════════════════════════════
# LinearObjective
# ═══════════════════════════════════════════════════════════════════════════


class TestLinearObjective:
    def test_construction(self):
        obj = LinearObjective(terms=[(1.0, "x"), (2.0, "y")], constant=0.0, name="cost")
        assert obj.name == "cost"

    def test_add_term(self):
        obj = LinearObjective(terms=[], constant=0.0, name="obj")
        obj2 = obj.add_term(3.0, "z")
        assert len(obj2.terms) == 1

    def test_negate(self):
        obj = LinearObjective(terms=[(1.0, "x")], constant=5.0, name="obj")
        neg = obj.negate()
        assert neg.terms[0][0] == -1.0
        assert neg.constant == -5.0

    def test_scale(self):
        obj = LinearObjective(terms=[(2.0, "x")], constant=1.0, name="obj")
        scaled = obj.scale(3.0)
        assert scaled.terms[0][0] == 6.0
        assert scaled.constant == 3.0

    def test_to_z3(self):
        obj = LinearObjective(terms=[(1.0, "x"), (2.0, "y")], constant=3.0, name="obj")
        z = obj.to_z3()
        assert isinstance(z, z3.ExprRef)

    def test_str(self):
        obj = LinearObjective(terms=[(1.0, "x")], constant=0.0, name="obj")
        assert isinstance(str(obj), str)


# ═══════════════════════════════════════════════════════════════════════════
# OMTSolver – basic minimize/maximize
# ═══════════════════════════════════════════════════════════════════════════


class TestOMTSolverBasic:
    def test_construction(self):
        opt = OMTSolver()
        assert opt is not None

    def test_construction_with_timeout(self):
        opt = OMTSolver(timeout_ms=5000)
        assert opt is not None

    def test_minimize_simple(self):
        opt = OMTSolver()
        x = z3.Real("x")
        opt.add(x >= 0, x <= 10)
        opt.minimize(x, name="min_x")
        result = opt.check()
        assert result.feasible
        if result.optimal_value is not None:
            assert abs(result.optimal_value - 0.0) < 1e-6

    def test_maximize_simple(self):
        opt = OMTSolver()
        x = z3.Real("x")
        opt.add(x >= 0, x <= 10)
        opt.maximize(x, name="max_x")
        result = opt.check()
        assert result.feasible
        if result.optimal_value is not None:
            assert abs(result.optimal_value - 10.0) < 1e-6

    def test_minimize_linear_objective(self):
        opt = OMTSolver()
        x = z3.Real("x")
        y = z3.Real("y")
        opt.add(x >= 0, y >= 0, x + y >= 5)
        obj = LinearObjective(terms=[(1.0, "x"), (1.0, "y")], constant=0.0, name="sum")
        opt.minimize_linear(obj)
        result = opt.check()
        assert result.feasible

    def test_maximize_linear_objective(self):
        opt = OMTSolver()
        x = z3.Real("x")
        y = z3.Real("y")
        opt.add(x >= 0, y >= 0, x <= 5, y <= 5)
        obj = LinearObjective(terms=[(1.0, "x"), (1.0, "y")], constant=0.0, name="sum")
        opt.maximize_linear(obj)
        result = opt.check()
        assert result.feasible


# ═══════════════════════════════════════════════════════════════════════════
# OMTSolver – infeasible problems
# ═══════════════════════════════════════════════════════════════════════════


class TestOMTSolverInfeasible:
    def test_infeasible(self):
        opt = OMTSolver()
        x = z3.Real("x")
        opt.add(x > 10, x < 5)
        opt.minimize(x)
        result = opt.check()
        assert result.feasible is False

    def test_infeasible_values_empty(self):
        opt = OMTSolver()
        x = z3.Real("x")
        opt.add(x > 10, x < 5)
        opt.minimize(x)
        result = opt.check()
        assert len(result.values) == 0 or result.feasible is False


# ═══════════════════════════════════════════════════════════════════════════
# OMTSolver – absolute value linearization
# ═══════════════════════════════════════════════════════════════════════════


class TestOMTSolverAbsLinearize:
    def test_linearize_abs(self):
        opt = OMTSolver()
        x = z3.Real("x")
        abs_x, pos_constraint, neg_constraint = opt.linearize_abs(x, name_prefix="t")
        assert isinstance(abs_x, z3.ExprRef)

    def test_minimize_abs(self):
        opt = OMTSolver()
        x = z3.Real("x")
        opt.add(x >= -5, x <= 5)
        opt.minimize_abs(x, name="abs_x")
        result = opt.check()
        assert result.feasible
        if result.optimal_value is not None:
            assert result.optimal_value >= -1e-6


# ═══════════════════════════════════════════════════════════════════════════
# OMTSolver – soft constraints
# ═══════════════════════════════════════════════════════════════════════════


class TestOMTSolverSoftConstraints:
    def test_add_soft(self):
        opt = OMTSolver()
        x = z3.Real("x")
        opt.add(x >= 0, x <= 10)
        sc = SoftConstraint(constraint=x < 3, weight=1.0, group="g1", label="prefer_small")
        opt.add_soft(sc)
        result = opt.check()
        assert result.feasible

    def test_add_soft_constraints_batch(self):
        opt = OMTSolver()
        x = z3.Real("x")
        opt.add(x >= 0, x <= 10)
        scs = [
            SoftConstraint(constraint=x < 5, weight=1.0, group="g1", label="c1"),
            SoftConstraint(constraint=x > 2, weight=2.0, group="g1", label="c2"),
        ]
        opt.add_soft_constraints(scs)
        result = opt.check()
        assert result.feasible

    def test_conflicting_soft_constraints(self):
        opt = OMTSolver()
        x = z3.Real("x")
        opt.add(x >= 0, x <= 10)
        scs = [
            SoftConstraint(constraint=x < 1, weight=1.0, group="g1", label="small"),
            SoftConstraint(constraint=x > 9, weight=1.0, group="g2", label="large"),
        ]
        opt.add_soft_constraints(scs)
        result = opt.check()
        assert result.feasible  # hard constraints still satisfied


# ═══════════════════════════════════════════════════════════════════════════
# OMTSolver – cost function
# ═══════════════════════════════════════════════════════════════════════════


class TestOMTSolverCost:
    def test_minimize_repair_cost(self):
        opt = OMTSolver()
        x = z3.Real("x")
        opt.add(x >= 0)
        result = opt.minimize_repair_cost(
            param_vars={"x": x},
            original_values={"x": 5.0},
            constraints=[x >= 1],
        )
        assert result.feasible

    def test_minimize_repair_cost_infeasible(self):
        opt = OMTSolver()
        x = z3.Real("x")
        result = opt.minimize_repair_cost(
            param_vars={"x": x},
            original_values={"x": 5.0},
            constraints=[x > 10, x < 3],
        )
        assert result.feasible is False


# ═══════════════════════════════════════════════════════════════════════════
# OMTResult
# ═══════════════════════════════════════════════════════════════════════════


class TestOMTResult:
    def test_get_value(self):
        opt = OMTSolver()
        x = z3.Real("x")
        opt.add(x == 7)
        opt.minimize(x)
        result = opt.check()
        v = result.get_value("x")
        assert v is not None
        assert abs(v - 7.0) < 1e-6

    def test_summary(self):
        opt = OMTSolver()
        x = z3.Real("x")
        opt.add(x >= 0)
        opt.minimize(x)
        result = opt.check()
        s = result.summary()
        assert isinstance(s, str)

    def test_reset(self):
        opt = OMTSolver()
        x = z3.Real("x")
        opt.add(x > 10, x < 5)
        opt.minimize(x)
        r1 = opt.check()
        assert r1.feasible is False
        opt.reset()
        opt.add(x >= 0, x <= 5)
        opt.minimize(x)
        r2 = opt.check()
        assert r2.feasible


# ═══════════════════════════════════════════════════════════════════════════
# OMTSolver – encoding
# ═══════════════════════════════════════════════════════════════════════════


class TestOMTSolverEncoding:
    def test_add_encoding(self):
        opt = OMTSolver()
        x = z3.Real("x")
        enc = SMTEncoding(
            formula=z3.And(x > 0, x < 10),
            variables={"x": x},
            assertions=[x > 0, x < 10],
            metadata={},
            aux_vars={},
            soundness=SoundnessTracker(),
        )
        opt.add_encoding(enc)
        opt.minimize(x)
        result = opt.check()
        assert result.feasible
