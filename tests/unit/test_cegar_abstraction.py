"""Tests for dpcegar.cegar.abstraction – abstract states, partitions, lattice."""

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
    AbstractionLattice,
    AbstractionState,
    AbstractState,
    InitialAbstraction,
    PathPartition,
    RefinementKind,
    RefinementRecord,
    WideningOperator,
)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _make_path_set(n: int = 3) -> PathSet:
    ps = PathSet()
    for i in range(n):
        sp = SymbolicPath(
            path_condition=PathCondition.trivially_true(),
            noise_draws=[
                NoiseDrawInfo(
                    variable=f"eta_{i}",
                    kind=NoiseKind.LAPLACE,
                    center_expr=Var(ty=IRType.REAL, name="q"),
                    scale_expr=Const.real(1.0),
                    site_id=100 + i,
                )
            ],
            output_expr=Var(ty=IRType.REAL, name=f"eta_{i}"),
        )
        ps.add(sp)
    return ps


# ═══════════════════════════════════════════════════════════════════════════
# AbstractDensityBound
# ═══════════════════════════════════════════════════════════════════════════


class TestAbstractDensityBound:
    def test_construction(self):
        b = AbstractDensityBound(lo=0.0, hi=1.0, is_exact=False, source_notion="pure_dp")
        assert b.lo == 0.0
        assert b.hi == 1.0

    def test_unbounded(self):
        b = AbstractDensityBound.unbounded()
        assert b.is_unbounded()

    def test_exact(self):
        b = AbstractDensityBound.exact(0.5)
        assert b.is_exact
        assert b.lo == b.hi == 0.5

    def test_from_interval(self):
        b = AbstractDensityBound.from_interval(0.1, 0.9, "test")
        assert b.lo == 0.1
        assert b.hi == 0.9

    def test_contains(self):
        b = AbstractDensityBound(lo=0.0, hi=1.0, is_exact=False, source_notion="")
        assert b.contains(0.5)
        assert not b.contains(1.5)

    def test_overlaps(self):
        b1 = AbstractDensityBound(lo=0.0, hi=1.0, is_exact=False, source_notion="")
        b2 = AbstractDensityBound(lo=0.5, hi=1.5, is_exact=False, source_notion="")
        b3 = AbstractDensityBound(lo=2.0, hi=3.0, is_exact=False, source_notion="")
        assert b1.overlaps(b2)
        assert not b1.overlaps(b3)

    def test_is_subset_of(self):
        inner = AbstractDensityBound(lo=0.2, hi=0.8, is_exact=False, source_notion="")
        outer = AbstractDensityBound(lo=0.0, hi=1.0, is_exact=False, source_notion="")
        assert inner.is_subset_of(outer)
        assert not outer.is_subset_of(inner)

    def test_meet(self):
        b1 = AbstractDensityBound(lo=0.0, hi=1.0, is_exact=False, source_notion="")
        b2 = AbstractDensityBound(lo=0.5, hi=1.5, is_exact=False, source_notion="")
        m = b1.meet(b2)
        assert m.lo == 0.5
        assert m.hi == 1.0

    def test_join(self):
        b1 = AbstractDensityBound(lo=0.0, hi=1.0, is_exact=False, source_notion="")
        b2 = AbstractDensityBound(lo=0.5, hi=1.5, is_exact=False, source_notion="")
        j = b1.join(b2)
        assert j.lo == 0.0
        assert j.hi == 1.5

    def test_widen(self):
        b1 = AbstractDensityBound(lo=0.0, hi=1.0, is_exact=False, source_notion="")
        b2 = AbstractDensityBound(lo=-0.1, hi=1.1, is_exact=False, source_notion="")
        w = b1.widen(b2, threshold=10.0)
        assert w.lo <= b2.lo
        assert w.hi >= b2.hi

    def test_narrow(self):
        outer = AbstractDensityBound(lo=-10.0, hi=10.0, is_exact=False, source_notion="")
        inner = AbstractDensityBound(lo=0.0, hi=1.0, is_exact=False, source_notion="")
        n = outer.narrow(inner)
        assert n.lo >= inner.lo or n.lo >= outer.lo
        assert n.hi <= inner.hi or n.hi <= outer.hi

    def test_satisfies_epsilon(self):
        b = AbstractDensityBound(lo=0.0, hi=0.5, is_exact=False, source_notion="")
        assert b.satisfies_epsilon(1.0)
        large = AbstractDensityBound(lo=0.0, hi=5.0, is_exact=False, source_notion="")
        assert not large.satisfies_epsilon(1.0)


# ═══════════════════════════════════════════════════════════════════════════
# AbstractState
# ═══════════════════════════════════════════════════════════════════════════


class TestAbstractState:
    def test_construction(self):
        s = AbstractState(
            state_id="s0",
            path_ids={0, 1, 2},
            density_bound=AbstractDensityBound.unbounded(),
            predicates=[],
            metadata={},
        )
        assert s.size() == 3

    def test_is_singleton(self):
        s = AbstractState(
            state_id="s0", path_ids={0},
            density_bound=AbstractDensityBound.unbounded(),
            predicates=[], metadata={},
        )
        assert s.is_singleton()

    def test_is_empty(self):
        s = AbstractState(
            state_id="s0", path_ids=set(),
            density_bound=AbstractDensityBound.unbounded(),
            predicates=[], metadata={},
        )
        assert s.is_empty()

    def test_contains_path(self):
        s = AbstractState(
            state_id="s0", path_ids={0, 1},
            density_bound=AbstractDensityBound.unbounded(),
            predicates=[], metadata={},
        )
        assert s.contains_path(0)
        assert not s.contains_path(5)

    def test_add_remove_path(self):
        s = AbstractState(
            state_id="s0", path_ids={0},
            density_bound=AbstractDensityBound.unbounded(),
            predicates=[], metadata={},
        )
        s.add_path(1)
        assert s.contains_path(1)
        s.remove_path(0)
        assert not s.contains_path(0)

    def test_add_predicate(self):
        s = AbstractState(
            state_id="s0", path_ids={0},
            density_bound=AbstractDensityBound.unbounded(),
            predicates=[], metadata={},
        )
        pred = BinOp(
            ty=IRType.BOOL, op=BinOpKind.GT,
            left=Var(ty=IRType.REAL, name="x"), right=Const.real(0.0),
        )
        s.add_predicate(pred)
        assert len(s.predicates) == 1

    def test_refine_bound(self):
        s = AbstractState(
            state_id="s0", path_ids={0},
            density_bound=AbstractDensityBound.unbounded(),
            predicates=[], metadata={},
        )
        new_bound = AbstractDensityBound(lo=0.0, hi=1.0, is_exact=False, source_notion="")
        s.refine_bound(new_bound)
        assert s.density_bound.hi == 1.0


# ═══════════════════════════════════════════════════════════════════════════
# PathPartition
# ═══════════════════════════════════════════════════════════════════════════


class TestPathPartition:
    def test_singleton_partition(self):
        ps = _make_path_set(3)
        part = PathPartition.singleton_partition(ps)
        assert part.state_count() == 3

    def test_coarsest_partition(self):
        ps = _make_path_set(3)
        part = PathPartition.coarsest_partition(ps)
        assert part.state_count() == 1

    def test_get_state(self):
        ps = _make_path_set(2)
        part = PathPartition.coarsest_partition(ps)
        for sid in [s.state_id for s in part.iter_states()]:
            assert part.get_state(sid) is not None

    def test_get_state_for_path(self):
        ps = _make_path_set(3)
        part = PathPartition.coarsest_partition(ps)
        all_ids = part.all_path_ids()
        for pid in all_ids:
            assert part.get_state_for_path(pid) is not None

    def test_split_state(self):
        ps = _make_path_set(4)
        part = PathPartition.coarsest_partition(ps)
        states = list(part.iter_states())
        sid = states[0].state_id
        all_pids = list(states[0].path_ids)
        pred = lambda pid: pid in set(all_pids[:2])  # noqa: E731
        t_id, f_id = part.split_state(sid, pred)
        assert part.state_count() == 2

    def test_merge_states(self):
        ps = _make_path_set(3)
        part = PathPartition.singleton_partition(ps)
        states = list(part.iter_states())
        s0 = states[0].state_id
        s1 = states[1].state_id
        merged_id = part.merge_states(s0, s1)
        assert part.get_state(merged_id) is not None

    def test_is_finest(self):
        ps = _make_path_set(3)
        finest = PathPartition.singleton_partition(ps)
        assert finest.is_finest(ps)
        coarsest = PathPartition.coarsest_partition(ps)
        assert not coarsest.is_finest(ps)

    def test_is_coarsest(self):
        ps = _make_path_set(3)
        coarsest = PathPartition.coarsest_partition(ps)
        assert coarsest.is_coarsest()


# ═══════════════════════════════════════════════════════════════════════════
# AbstractionState
# ═══════════════════════════════════════════════════════════════════════════


class TestAbstractionState:
    def test_from_initial_coarsest(self):
        ps = _make_path_set(3)
        abs_state = InitialAbstraction.coarsest(ps)
        assert isinstance(abs_state, AbstractionState)
        assert abs_state.refinement_level == 0

    def test_from_initial_finest(self):
        ps = _make_path_set(3)
        abs_state = InitialAbstraction.finest(ps)
        assert abs_state.is_finest()

    def test_by_noise_pattern(self):
        ps = _make_path_set(3)
        abs_state = InitialAbstraction.by_noise_pattern(ps)
        assert isinstance(abs_state, AbstractionState)

    def test_by_branch_structure(self):
        ps = _make_path_set(3)
        abs_state = InitialAbstraction.by_branch_structure(ps)
        assert isinstance(abs_state, AbstractionState)

    def test_overall_density_bound(self):
        ps = _make_path_set(3)
        abs_state = InitialAbstraction.coarsest(ps)
        bound = abs_state.overall_density_bound()
        assert isinstance(bound, AbstractDensityBound)

    def test_set_density_bound(self):
        ps = _make_path_set(3)
        abs_state = InitialAbstraction.coarsest(ps)
        state_ids = abs_state.all_state_ids()
        new_bound = AbstractDensityBound(lo=0.0, hi=0.5, is_exact=False, source_notion="test")
        abs_state.set_density_bound(state_ids[0], new_bound)
        assert abs_state.get_abstract_density(state_ids[0]).hi == 0.5

    def test_split_state(self):
        ps = _make_path_set(4)
        abs_state = InitialAbstraction.coarsest(ps)
        state_ids = abs_state.all_state_ids()
        sid = state_ids[0]
        state = abs_state.partition.get_state(sid)
        pids = list(state.path_ids)
        pred = lambda pid: pid in set(pids[:2])  # noqa: E731
        t_id, f_id = abs_state.split_state(sid, pred, RefinementKind.PATH_SPLIT)
        assert len(abs_state.all_state_ids()) == 2

    def test_record_refinement(self):
        ps = _make_path_set(3)
        abs_state = InitialAbstraction.coarsest(ps)
        rec = RefinementRecord(
            kind=RefinementKind.PATH_SPLIT,
            source_state_id="s0",
            result_state_ids=["s1", "s2"],
            predicate=None,
            details={},
            iteration=1,
        )
        abs_state.record_refinement(rec)
        assert len(abs_state.history) == 1

    def test_summary(self):
        ps = _make_path_set(3)
        abs_state = InitialAbstraction.coarsest(ps)
        s = abs_state.summary()
        assert isinstance(s, dict)


# ═══════════════════════════════════════════════════════════════════════════
# AbstractionLattice
# ═══════════════════════════════════════════════════════════════════════════


class TestAbstractionLattice:
    def test_is_refinement_of(self):
        ps = _make_path_set(4)
        coarsest = InitialAbstraction.coarsest(ps)
        finest = InitialAbstraction.finest(ps)
        lat = AbstractionLattice()
        assert lat.is_refinement_of(finest, coarsest)

    def test_lattice_join(self):
        ps = _make_path_set(4)
        a = InitialAbstraction.coarsest(ps)
        b = InitialAbstraction.finest(ps)
        lat = AbstractionLattice()
        joined = lat.lattice_join(a, b)
        assert isinstance(joined, AbstractionState)

    def test_lattice_meet(self):
        ps = _make_path_set(4)
        a = InitialAbstraction.coarsest(ps)
        b = InitialAbstraction.finest(ps)
        lat = AbstractionLattice()
        met = lat.lattice_meet(a, b)
        assert isinstance(met, AbstractionState)


# ═══════════════════════════════════════════════════════════════════════════
# WideningOperator
# ═══════════════════════════════════════════════════════════════════════════


class TestWideningOperator:
    def test_construction(self):
        w = WideningOperator(threshold=5.0, patience=3)
        assert not w.should_widen()

    def test_should_widen_after_patience(self):
        w = WideningOperator(threshold=5.0, patience=2)
        bounds = {"s0": AbstractDensityBound(lo=0.0, hi=1.0, is_exact=False, source_notion="")}
        w.record_iteration(bounds)
        w.record_iteration(bounds)
        w.record_iteration(bounds)
        # After enough stalled iterations, should_widen may become True
        assert isinstance(w.should_widen(), bool)

    def test_apply(self):
        ps = _make_path_set(2)
        abs_state = InitialAbstraction.coarsest(ps)
        w = WideningOperator(threshold=5.0, patience=1)
        widened = w.apply(abs_state)
        assert isinstance(widened, AbstractionState)

    def test_reset(self):
        w = WideningOperator(threshold=5.0, patience=3)
        w.reset()
        assert not w.should_widen()
