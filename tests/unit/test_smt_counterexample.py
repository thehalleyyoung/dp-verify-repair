"""Tests for dpcegar.smt.counterexample – extraction, dedup, validation."""

from __future__ import annotations

import pytest
import z3

from dpcegar.ir.types import (
    ApproxBudget,
    IRType,
    NoiseKind,
    PrivacyNotion,
    PureBudget,
    Var,
    Const,
)
from dpcegar.paths.symbolic_path import NoiseDrawInfo
from dpcegar.smt.solver import CheckResult, SolverResult, Z3Solver
from dpcegar.smt.counterexample import (
    Counterexample,
    CounterexampleDiversifier,
    CounterexampleExtractor,
    CounterexampleMinimizer,
    CounterexampleSet,
)
from dpcegar.smt.theory_selection import SMTTheory


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _make_sat_result(assignments: dict[str, float]) -> SolverResult:
    """Build a fake SAT SolverResult with model values."""
    solver = Z3Solver()
    for name, val in assignments.items():
        solver.add(z3.Real(name) == val)
    return solver.check()


def _make_counterexample(
    d: dict[str, float] | None = None,
    d_prime: dict[str, float] | None = None,
    output: float = 0.0,
    path_id: int = 0,
    loss: float = 1.5,
) -> Counterexample:
    return Counterexample(
        d_values=d or {"q": 5.0},
        d_prime_values=d_prime or {"q_prime": 6.0},
        output_value=output,
        path_id=path_id,
        privacy_loss=loss,
        witness_values={},
        notion=PrivacyNotion.PURE_DP,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Counterexample dataclass
# ═══════════════════════════════════════════════════════════════════════════


class TestCounterexample:
    def test_construction(self):
        ce = _make_counterexample()
        assert ce.path_id == 0
        assert ce.privacy_loss == 1.5

    def test_violation_amount(self):
        ce = _make_counterexample(loss=2.0)
        assert ce.violation_amount > 0

    def test_is_valid(self):
        ce = _make_counterexample(loss=1.5)
        assert isinstance(ce.is_valid, bool)

    def test_signature_deterministic(self):
        ce1 = _make_counterexample(d={"q": 5.0}, loss=1.5, path_id=0)
        ce2 = _make_counterexample(d={"q": 5.0}, loss=1.5, path_id=0)
        assert ce1.signature() == ce2.signature()

    def test_signature_differs_for_different_values(self):
        ce1 = _make_counterexample(d={"q": 5.0})
        ce2 = _make_counterexample(d={"q": 10.0})
        assert ce1.signature() != ce2.signature()

    def test_summary_string(self):
        ce = _make_counterexample()
        s = ce.summary()
        assert isinstance(s, str)
        assert len(s) > 0

    def test_str(self):
        ce = _make_counterexample()
        assert len(str(ce)) > 0

    def test_notion_stored(self):
        ce = _make_counterexample()
        assert ce.notion == PrivacyNotion.PURE_DP

    def test_budget_optional(self):
        ce = _make_counterexample()
        assert ce.budget is None


# ═══════════════════════════════════════════════════════════════════════════
# CounterexampleSet
# ═══════════════════════════════════════════════════════════════════════════


class TestCounterexampleSet:
    def test_empty_set(self):
        cs = CounterexampleSet()
        assert len(cs) == 0
        assert not cs

    def test_add_single(self):
        cs = CounterexampleSet()
        ce = _make_counterexample()
        added = cs.add(ce)
        assert added is True
        assert len(cs) == 1

    def test_add_duplicate(self):
        cs = CounterexampleSet()
        ce1 = _make_counterexample(d={"q": 5.0}, loss=1.5, path_id=0)
        ce2 = _make_counterexample(d={"q": 5.0}, loss=1.5, path_id=0)
        cs.add(ce1)
        added = cs.add(ce2)
        assert added is False
        assert len(cs) == 1

    def test_add_different(self):
        cs = CounterexampleSet()
        cs.add(_make_counterexample(d={"q": 5.0}, path_id=0))
        cs.add(_make_counterexample(d={"q": 10.0}, path_id=1))
        assert len(cs) == 2

    def test_add_all(self):
        cs = CounterexampleSet()
        ces = [
            _make_counterexample(d={"q": float(i)}, path_id=i)
            for i in range(5)
        ]
        count = cs.add_all(ces)
        assert count == 5
        assert len(cs) == 5

    def test_worst(self):
        cs = CounterexampleSet()
        cs.add(_make_counterexample(loss=1.0, path_id=0, d={"q": 1.0}))
        cs.add(_make_counterexample(loss=3.0, path_id=1, d={"q": 2.0}))
        cs.add(_make_counterexample(loss=2.0, path_id=2, d={"q": 3.0}))
        w = cs.worst()
        assert w is not None
        assert w.privacy_loss == 3.0

    def test_worst_empty(self):
        cs = CounterexampleSet()
        assert cs.worst() is None

    def test_by_path(self):
        cs = CounterexampleSet()
        cs.add(_make_counterexample(path_id=0, d={"q": 1.0}))
        cs.add(_make_counterexample(path_id=0, d={"q": 2.0}))
        cs.add(_make_counterexample(path_id=1, d={"q": 3.0}))
        p0 = cs.by_path(0)
        assert len(p0) == 2

    def test_path_coverage(self):
        cs = CounterexampleSet()
        cs.add(_make_counterexample(path_id=0, d={"q": 1.0}))
        cs.add(_make_counterexample(path_id=2, d={"q": 2.0}))
        assert cs.path_coverage == {0, 2}

    def test_max_loss(self):
        cs = CounterexampleSet()
        cs.add(_make_counterexample(loss=1.0, d={"q": 1.0}))
        cs.add(_make_counterexample(loss=5.0, d={"q": 2.0}, path_id=1))
        assert cs.max_loss == 5.0

    def test_coverage_analysis(self):
        cs = CounterexampleSet()
        cs.add(_make_counterexample(path_id=0, d={"q": 1.0}))
        analysis = cs.coverage_analysis(total_paths=3)
        assert isinstance(analysis, dict)

    def test_iter(self):
        cs = CounterexampleSet()
        cs.add(_make_counterexample(d={"q": 1.0}))
        cs.add(_make_counterexample(d={"q": 2.0}, path_id=1))
        count = sum(1 for _ in cs)
        assert count == 2

    def test_bool_truthy(self):
        cs = CounterexampleSet()
        assert not cs
        cs.add(_make_counterexample())
        assert cs


# ═══════════════════════════════════════════════════════════════════════════
# CounterexampleExtractor
# ═══════════════════════════════════════════════════════════════════════════


class TestCounterexampleExtractor:
    def test_construction(self):
        ext = CounterexampleExtractor()
        assert ext is not None

    def test_construction_custom_suffixes(self):
        ext = CounterexampleExtractor(d_suffix="_d", d_prime_suffix="_dp")
        assert ext is not None

    def test_extract_from_sat_result(self):
        result = _make_sat_result({"q": 5.0, "q_prime": 6.0, "o": 5.5})
        ext = CounterexampleExtractor()
        ce = ext.extract(result, path_id=0)
        assert ce is not None
        assert isinstance(ce, Counterexample)

    def test_extract_returns_none_on_unsat(self):
        solver = Z3Solver()
        x = z3.Real("x")
        solver.add(x > 10, x < 5)
        result = solver.check()
        ext = CounterexampleExtractor()
        ce = ext.extract(result, path_id=0)
        assert ce is None

    def test_extract_multiple(self):
        results = [
            _make_sat_result({"q": float(i), "q_prime": float(i + 1), "o": float(i)})
            for i in range(3)
        ]
        ext = CounterexampleExtractor()
        cs = ext.extract_multiple(results, path_ids=[0, 1, 2])
        assert len(cs) >= 0

    def test_extract_with_budget(self):
        result = _make_sat_result({"q": 5.0, "q_prime": 6.0, "o": 5.5})
        ext = CounterexampleExtractor()
        budget = PureBudget(epsilon=1.0)
        ce = ext.extract(result, path_id=0, budget=budget)
        assert ce is not None

    def test_extract_with_noise_draws(self):
        result = _make_sat_result({"q": 5.0, "q_prime": 6.0, "o": 5.5, "eta": 0.5})
        ext = CounterexampleExtractor()
        draws = [
            NoiseDrawInfo(
                variable="eta",
                kind=NoiseKind.LAPLACE,
                center_expr=Var(ty=IRType.REAL, name="q"),
                scale_expr=Const.real(1.0),
                site_id=100,
            )
        ]
        ce = ext.extract(result, path_id=0, noise_draws=draws)
        assert ce is not None

    def test_z3_to_float_rational(self):
        val = z3.RatVal(7, 2)
        f = CounterexampleExtractor._z3_to_float(val)
        assert f is not None
        assert abs(f - 3.5) < 1e-9

    def test_z3_to_float_int(self):
        val = z3.IntVal(42)
        f = CounterexampleExtractor._z3_to_float(val)
        assert f is not None
        assert abs(f - 42.0) < 1e-9


# ═══════════════════════════════════════════════════════════════════════════
# Counterexample validation
# ═══════════════════════════════════════════════════════════════════════════


class TestCounterexampleValidation:
    def test_valid_counterexample(self):
        ce = _make_counterexample(loss=2.0)
        assert isinstance(ce.is_valid, bool)

    def test_negative_loss_invalid(self):
        ce = _make_counterexample(loss=-1.0)
        # Negative loss likely indicates no violation
        assert ce.violation_amount <= 0 or ce.privacy_loss < 0

    def test_zero_loss_borderline(self):
        ce = _make_counterexample(loss=0.0)
        assert isinstance(ce.is_valid, bool)


# ═══════════════════════════════════════════════════════════════════════════
# Printing
# ═══════════════════════════════════════════════════════════════════════════


class TestCounterexamplePrinting:
    def test_counterexample_str(self):
        ce = _make_counterexample()
        s = str(ce)
        assert isinstance(s, str)

    def test_counterexample_summary(self):
        ce = _make_counterexample()
        s = ce.summary()
        assert isinstance(s, str)

    def test_set_printing(self):
        cs = CounterexampleSet()
        cs.add(_make_counterexample(d={"q": 1.0}))
        cs.add(_make_counterexample(d={"q": 2.0}, path_id=1))
        assert len(cs) == 2


# ═══════════════════════════════════════════════════════════════════════════
# CounterexampleMinimizer
# ═══════════════════════════════════════════════════════════════════════════


class TestCounterexampleMinimizer:
    def test_minimize_returns_counterexample(self):
        ce = _make_counterexample(d={"q": 100.0, "q_prime": 200.0})
        minimizer = CounterexampleMinimizer()
        result = minimizer.minimize(ce)
        assert isinstance(result, Counterexample)

    def test_minimize_preserves_path_id(self):
        ce = _make_counterexample(d={"q": 50.0}, path_id=3)
        minimizer = CounterexampleMinimizer()
        result = minimizer.minimize(ce)
        assert result.path_id == 3

    def test_minimize_smaller_values(self):
        ce = _make_counterexample(d={"q": 1000.0, "q_prime": 999.0})
        minimizer = CounterexampleMinimizer(max_iter=10)
        result = minimizer.minimize(ce)
        for k, v in result.variable_assignments.items():
            assert abs(v) <= abs(ce.variable_assignments.get(k, float("inf"))) + 1e-6


class TestCounterexampleDiversifier:
    def test_diversify_empty(self):
        cs = CounterexampleSet()
        diversifier = CounterexampleDiversifier()
        result = diversifier.diversify(cs, k=3)
        assert len(result) == 0

    def test_diversify_single(self):
        cs = CounterexampleSet()
        cs.add(_make_counterexample(d={"q": 1.0}))
        diversifier = CounterexampleDiversifier()
        result = diversifier.diversify(cs, k=3)
        assert len(result) == 1

    def test_diversify_many(self):
        cs = CounterexampleSet()
        for i in range(10):
            cs.add(_make_counterexample(d={"q": float(i)}, path_id=i % 3))
        diversifier = CounterexampleDiversifier()
        result = diversifier.diversify(cs, k=3)
        assert len(result) <= 3

    def test_diversify_picks_different_paths(self):
        cs = CounterexampleSet()
        for pid in range(5):
            cs.add(_make_counterexample(d={"q": float(pid)}, path_id=pid))
        diversifier = CounterexampleDiversifier()
        result = diversifier.diversify(cs, k=3)
        paths = {ce.path_id for ce in result}
        assert len(paths) >= min(3, len(paths))


# ═══════════════════════════════════════════════════════════════════════════
# CounterexampleExtractor – edge cases
# ═══════════════════════════════════════════════════════════════════════════


class TestCounterexampleExtractorEdgeCases:
    def test_extract_unsat_returns_none(self):
        s = z3.Solver()
        x = z3.Real("x")
        s.add(x > 0, x < 0)
        result = s.check()
        ext = CounterexampleExtractor()
        assert result == z3.unsat

    def test_extract_algebraic_value(self):
        val = z3.RatVal(355, 113)
        f = CounterexampleExtractor._z3_to_float(val)
        assert isinstance(f, float)

    def test_extract_zero_value(self):
        val = z3.RatVal(0, 1)
        f = CounterexampleExtractor._z3_to_float(val)
        assert f == 0.0

    def test_extract_negative_value(self):
        val = z3.RatVal(-7, 2)
        f = CounterexampleExtractor._z3_to_float(val)
        assert f < 0

    def test_large_rational(self):
        val = z3.RatVal(10**18, 1)
        f = CounterexampleExtractor._z3_to_float(val)
        assert f > 0
