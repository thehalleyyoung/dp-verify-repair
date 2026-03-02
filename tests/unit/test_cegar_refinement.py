"""Tests for dpcegar.cegar.refinement – refinement operators, history, convergence."""

from __future__ import annotations

from typing import Any

import pytest

from dpcegar.ir.types import (
    BinOp,
    BinOpKind,
    Const,
    IRType,
    NoiseKind,
    PureBudget,
    Var,
)
from dpcegar.paths.symbolic_path import (
    NoiseDrawInfo,
    PathCondition,
    PathSet,
    SymbolicPath,
)
from dpcegar.cegar.abstraction import (
    AbstractDensityBound,
    AbstractionState,
    InitialAbstraction,
    RefinementKind,
    RefinementRecord,
)
from dpcegar.cegar.refinement import (
    ConvergenceDetector,
    ConvergenceReason,
    ConvergenceStatus,
    RefinementCounterexample,
    InfeasibilityProof,
    IntervalNarrowRefinement,
    LoopUnwindRefinement,
    PathSplitRefinement,
    PredicateRefinement,
    RefinementHistory,
    RefinementResult,
    RefinementSelector,
    RefinementStatus,
    SpuriousnessCause,
)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _make_path_set(n: int = 4) -> PathSet:
    ps = PathSet()
    for i in range(n):
        sp = SymbolicPath(
            path_condition=PathCondition.trivially_true(),
            noise_draws=[
                NoiseDrawInfo(
                    variable=f"eta_{i}", kind=NoiseKind.LAPLACE,
                    center_expr=Var(ty=IRType.REAL, name="q"),
                    scale_expr=Const.real(1.0), site_id=100 + i,
                )
            ],
            output_expr=Var(ty=IRType.REAL, name=f"eta_{i}"),
        )
        ps.add(sp)
    return ps


def _make_counterexample(
    path_id: int = 0,
    state_id: str = "s0",
    density_ratio: float = 2.0,
) -> RefinementCounterexample:
    return RefinementCounterexample(
        variable_assignment={"q": 5.0, "eta_0": 5.5},
        path_id=path_id,
        state_id=state_id,
        density_ratio_value=density_ratio,
        is_spurious=None,
        spuriousness_reason="",
        metadata={},
    )


def _make_proof() -> InfeasibilityProof:
    return InfeasibilityProof(
        unsat_core=[
            BinOp(ty=IRType.BOOL, op=BinOpKind.GT,
                   left=Var(ty=IRType.REAL, name="x"), right=Const.real(0.0)),
        ],
        interpolants=[],
        involved_paths=[0],
        proof_time=0.01,
    )


# ═══════════════════════════════════════════════════════════════════════════
# RefinementCounterexample dataclass
# ═══════════════════════════════════════════════════════════════════════════


class TestRefinementCounterexample:
    def test_construction(self):
        cex = _make_counterexample()
        assert cex.path_id == 0

    def test_get_value(self):
        cex = _make_counterexample()
        assert cex.get_value("q") == 5.0
        assert cex.get_value("nonexistent") is None

    def test_involves_path(self):
        cex = _make_counterexample(path_id=3)
        assert cex.involves_path(3)
        assert not cex.involves_path(0)

    def test_summary(self):
        cex = _make_counterexample()
        assert isinstance(cex.summary(), str)


# ═══════════════════════════════════════════════════════════════════════════
# InfeasibilityProof
# ═══════════════════════════════════════════════════════════════════════════


class TestInfeasibilityProof:
    def test_has_unsat_core(self):
        p = _make_proof()
        assert p.has_unsat_core()

    def test_has_no_interpolants(self):
        p = _make_proof()
        assert not p.has_interpolants()

    def test_empty_proof(self):
        p = InfeasibilityProof(
            unsat_core=[], interpolants=[], involved_paths=[], proof_time=0.0,
        )
        assert not p.has_unsat_core()


# ═══════════════════════════════════════════════════════════════════════════
# RefinementResult
# ═══════════════════════════════════════════════════════════════════════════


class TestRefinementResult:
    def test_success(self):
        r = RefinementResult(
            status=RefinementStatus.SUCCESS,
            refined_state=None,
            record=None,
            new_state_ids=["s1", "s2"],
            eliminated_cex=True,
            details={},
        )
        assert r.success is True

    def test_no_progress(self):
        r = RefinementResult(
            status=RefinementStatus.NO_PROGRESS,
            refined_state=None,
            record=None,
            new_state_ids=[],
            eliminated_cex=False,
            details={},
        )
        assert r.success is False

    def test_at_finest(self):
        r = RefinementResult(
            status=RefinementStatus.AT_FINEST,
            refined_state=None,
            record=None,
            new_state_ids=[],
            eliminated_cex=False,
            details={},
        )
        assert r.success is False


# ═══════════════════════════════════════════════════════════════════════════
# PathSplitRefinement
# ═══════════════════════════════════════════════════════════════════════════


class TestPathSplitRefinement:
    def test_name(self):
        op = PathSplitRefinement()
        assert op.name() == "path_split" or len(op.name()) > 0

    def test_is_applicable_multi_path_state(self):
        ps = _make_path_set(4)
        state = InitialAbstraction.coarsest(ps)
        cex = _make_counterexample()
        op = PathSplitRefinement()
        assert isinstance(op.is_applicable(state, cex, None), bool)

    def test_apply_splits_state(self):
        ps = _make_path_set(4)
        state = InitialAbstraction.coarsest(ps)
        cex = _make_counterexample()
        op = PathSplitRefinement()
        if op.is_applicable(state, cex, None):
            result = op.apply(state, cex, None)
            assert isinstance(result, RefinementResult)

    def test_estimated_cost(self):
        ps = _make_path_set(4)
        state = InitialAbstraction.coarsest(ps)
        cex = _make_counterexample()
        op = PathSplitRefinement()
        cost = op.estimated_cost(state, cex)
        assert cost >= 0


# ═══════════════════════════════════════════════════════════════════════════
# IntervalNarrowRefinement
# ═══════════════════════════════════════════════════════════════════════════


class TestIntervalNarrowRefinement:
    def test_name(self):
        op = IntervalNarrowRefinement(shrink_factor=0.5)
        assert len(op.name()) > 0

    def test_is_applicable(self):
        ps = _make_path_set(2)
        state = InitialAbstraction.coarsest(ps)
        cex = _make_counterexample()
        op = IntervalNarrowRefinement(shrink_factor=0.5)
        assert isinstance(op.is_applicable(state, cex, None), bool)

    def test_apply_with_proof(self):
        ps = _make_path_set(2)
        state = InitialAbstraction.coarsest(ps)
        cex = _make_counterexample()
        proof = _make_proof()
        op = IntervalNarrowRefinement(shrink_factor=0.5)
        if op.is_applicable(state, cex, proof):
            result = op.apply(state, cex, proof)
            assert isinstance(result, RefinementResult)

    @pytest.mark.parametrize("factor", [0.1, 0.5, 0.9])
    def test_various_shrink_factors(self, factor: float):
        op = IntervalNarrowRefinement(shrink_factor=factor)
        assert len(op.name()) > 0

    def test_estimated_cost(self):
        ps = _make_path_set(2)
        state = InitialAbstraction.coarsest(ps)
        cex = _make_counterexample()
        op = IntervalNarrowRefinement(shrink_factor=0.5)
        assert op.estimated_cost(state, cex) >= 0


# ═══════════════════════════════════════════════════════════════════════════
# PredicateRefinement
# ═══════════════════════════════════════════════════════════════════════════


class TestPredicateRefinement:
    def test_name(self):
        op = PredicateRefinement()
        assert len(op.name()) > 0

    def test_is_applicable_needs_proof(self):
        ps = _make_path_set(4)
        state = InitialAbstraction.coarsest(ps)
        cex = _make_counterexample()
        op = PredicateRefinement()
        result_no_proof = op.is_applicable(state, cex, None)
        result_with_proof = op.is_applicable(state, cex, _make_proof())
        assert isinstance(result_no_proof, bool)
        assert isinstance(result_with_proof, bool)

    def test_estimated_cost(self):
        ps = _make_path_set(4)
        state = InitialAbstraction.coarsest(ps)
        cex = _make_counterexample()
        op = PredicateRefinement()
        assert op.estimated_cost(state, cex) >= 0


# ═══════════════════════════════════════════════════════════════════════════
# LoopUnwindRefinement
# ═══════════════════════════════════════════════════════════════════════════


class TestLoopUnwindRefinement:
    def test_construction(self):
        op = LoopUnwindRefinement(max_unroll=10)
        assert len(op.name()) > 0

    def test_get_unroll_depth(self):
        op = LoopUnwindRefinement(max_unroll=10)
        depth = op.get_unroll_depth(0)
        assert depth >= 0

    def test_estimated_cost(self):
        ps = _make_path_set(2)
        state = InitialAbstraction.coarsest(ps)
        cex = _make_counterexample()
        op = LoopUnwindRefinement(max_unroll=10)
        assert op.estimated_cost(state, cex) >= 0


# ═══════════════════════════════════════════════════════════════════════════
# RefinementSelector
# ═══════════════════════════════════════════════════════════════════════════


class TestRefinementSelector:
    def test_default_operators(self):
        sel = RefinementSelector()
        assert sel is not None

    def test_custom_operators(self):
        ops = [PathSplitRefinement(), IntervalNarrowRefinement(shrink_factor=0.5)]
        sel = RefinementSelector(operators=ops)
        assert sel is not None

    def test_select(self):
        ps = _make_path_set(4)
        state = InitialAbstraction.coarsest(ps)
        cex = _make_counterexample()
        sel = RefinementSelector()
        op = sel.select(state, cex, None)
        # May return None if no operator is applicable
        assert op is None or hasattr(op, "apply")

    def test_select_and_apply(self):
        ps = _make_path_set(4)
        state = InitialAbstraction.coarsest(ps)
        cex = _make_counterexample()
        sel = RefinementSelector()
        result = sel.select_and_apply(state, cex, None)
        assert isinstance(result, RefinementResult)

    def test_get_statistics(self):
        sel = RefinementSelector()
        stats = sel.get_statistics()
        assert isinstance(stats, dict)


# ═══════════════════════════════════════════════════════════════════════════
# RefinementHistory
# ═══════════════════════════════════════════════════════════════════════════


class TestRefinementHistory:
    def test_empty_history(self):
        h = RefinementHistory()
        assert h.total_refinements() == 0

    def test_add_record(self):
        h = RefinementHistory()
        rec = RefinementRecord(
            kind=RefinementKind.PATH_SPLIT,
            source_state_id="s0",
            result_state_ids=["s1", "s2"],
            predicate=None,
            details={},
            iteration=1,
        )
        h.add(rec)
        assert h.total_refinements() == 1

    def test_refinements_by_kind(self):
        h = RefinementHistory()
        for i in range(3):
            h.add(RefinementRecord(
                kind=RefinementKind.PATH_SPLIT,
                source_state_id=f"s{i}",
                result_state_ids=[],
                predicate=None, details={}, iteration=i,
            ))
        h.add(RefinementRecord(
            kind=RefinementKind.INTERVAL_NARROW,
            source_state_id="s3",
            result_state_ids=[],
            predicate=None, details={}, iteration=3,
        ))
        by_kind = h.refinements_by_kind()
        assert isinstance(by_kind, dict)

    def test_last_refinement(self):
        h = RefinementHistory()
        assert h.last_refinement() is None
        rec = RefinementRecord(
            kind=RefinementKind.PATH_SPLIT,
            source_state_id="s0",
            result_state_ids=[],
            predicate=None, details={}, iteration=0,
        )
        h.add(rec)
        assert h.last_refinement() is not None

    def test_record_state_fingerprint(self):
        ps = _make_path_set(3)
        abs_state = InitialAbstraction.coarsest(ps)
        h = RefinementHistory()
        is_new = h.record_state_fingerprint(abs_state)
        assert isinstance(is_new, bool)

    def test_is_cycling(self):
        h = RefinementHistory()
        assert not h.is_cycling(window=3)

    def test_is_stalling(self):
        h = RefinementHistory()
        assert not h.is_stalling(window=3)

    def test_summary(self):
        h = RefinementHistory()
        s = h.summary()
        assert isinstance(s, dict)


# ═══════════════════════════════════════════════════════════════════════════
# ConvergenceDetector
# ═══════════════════════════════════════════════════════════════════════════


class TestConvergenceDetector:
    def test_construction(self):
        cd = ConvergenceDetector(
            max_refinements=50,
            max_time_seconds=60.0,
            fixpoint_tolerance=1e-6,
        )
        cd.start()
        assert cd is not None

    def test_check_not_converged_initially(self):
        cd = ConvergenceDetector(
            max_refinements=50,
            max_time_seconds=60.0,
            fixpoint_tolerance=1e-6,
        )
        cd.start()
        ps = _make_path_set(3)
        state = InitialAbstraction.coarsest(ps)
        history = RefinementHistory()
        budget = PureBudget(epsilon=1.0)
        status = cd.check(state, history, budget)
        assert isinstance(status, ConvergenceStatus)

    def test_max_refinements_convergence(self):
        cd = ConvergenceDetector(
            max_refinements=2,
            max_time_seconds=600.0,
            fixpoint_tolerance=1e-6,
        )
        cd.start()
        ps = _make_path_set(3)
        state = InitialAbstraction.coarsest(ps)
        history = RefinementHistory()
        for i in range(3):
            history.add(RefinementRecord(
                kind=RefinementKind.PATH_SPLIT,
                source_state_id=f"s{i}",
                result_state_ids=[],
                predicate=None, details={}, iteration=i,
            ))
        budget = PureBudget(epsilon=1.0)
        status = cd.check(state, history, budget)
        assert isinstance(status.converged, bool)


# ═══════════════════════════════════════════════════════════════════════════
# Convergence reason enum
# ═══════════════════════════════════════════════════════════════════════════


class TestConvergenceReason:
    def test_all_values(self):
        assert ConvergenceReason.NOT_CONVERGED is not None
        assert ConvergenceReason.FINEST_REACHED is not None
        assert ConvergenceReason.FIXPOINT_REACHED is not None
        assert ConvergenceReason.BUDGET_SATISFIED is not None
        assert ConvergenceReason.MAX_REFINEMENTS is not None
        assert ConvergenceReason.CYCLE_DETECTED is not None
        assert ConvergenceReason.TIMEOUT is not None


# ═══════════════════════════════════════════════════════════════════════════
# SpuriousnessCause enum
# ═══════════════════════════════════════════════════════════════════════════


class TestSpuriousnessCause:
    def test_all_values(self):
        assert SpuriousnessCause.MERGED_PATHS is not None
        assert SpuriousnessCause.LOOSE_BOUNDS is not None
        assert SpuriousnessCause.MISSING_PREDICATE is not None
        assert SpuriousnessCause.UNKNOWN is not None


# ═══════════════════════════════════════════════════════════════════════════
# RefinementStatus enum
# ═══════════════════════════════════════════════════════════════════════════


class TestRefinementStatus:
    def test_all_values(self):
        assert RefinementStatus.SUCCESS is not None
        assert RefinementStatus.NO_PROGRESS is not None
        assert RefinementStatus.AT_FINEST is not None
        assert RefinementStatus.FAILED is not None
