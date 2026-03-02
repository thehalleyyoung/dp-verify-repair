"""Tests for dpcegar.repair.synthesizer – CEGIS loop, repair verification, cost."""

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
from dpcegar.ir.nodes import (
    MechIR,
    NoiseDrawNode,
    ParamDecl,
    QueryNode,
    ReturnNode,
    SequenceNode,
)
from dpcegar.paths.symbolic_path import (
    NoiseDrawInfo,
    PathCondition,
    PathSet,
    SymbolicPath,
)
from dpcegar.density.ratio_builder import (
    DensityRatioExpr,
    DensityRatioResult,
)
from dpcegar.cegar.engine import CEGARConfig, CEGARResult, CEGARVerdict
from dpcegar.cegar.refinement import RefinementCounterexample as CegarCex
from dpcegar.repair.templates import (
    RepairSite,
    ScaleParam,
    ThresholdShift,
    TemplateEnumerator,
)
from dpcegar.repair.synthesizer import (
    CostFunction,
    CounterexampleAccumulator,
    MinimizationResult,
    RepairMinimizer,
    RepairResult,
    RepairStatistics,
    RepairSynthesizer,
    RepairVerdict,
    RepairVerifier,
    SynthesizerConfig,
)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _buggy_laplace(scale: float = 0.1) -> MechIR:
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
        scale=Const.real(scale),
    )
    ret = ReturnNode(value=Var(ty=IRType.REAL, name="eta"))
    body = SequenceNode(stmts=[q, noise, ret])
    return MechIR(
        name="buggy_laplace",
        params=[ParamDecl(name="db", ty=IRType.ARRAY, is_database=True)],
        body=body,
        return_type=IRType.REAL,
        budget=PureBudget(epsilon=1.0),
    )


def _make_cex(path_id: int = 0) -> CegarCex:
    return CegarCex(
        variable_assignment={"q": 5.0, "eta": 5.5},
        path_id=path_id,
        state_id="s0",
        density_ratio_value=10.0,
        is_spurious=False,
        spuriousness_reason="",
        metadata={},
    )


def _make_path_set() -> PathSet:
    ps = PathSet()
    sp = SymbolicPath(
        path_condition=PathCondition.trivially_true(),
        noise_draws=[
            NoiseDrawInfo(
                variable="eta", kind=NoiseKind.LAPLACE,
                center_expr=Var(ty=IRType.REAL, name="q"),
                scale_expr=Const.real(0.1), site_id=100,
            )
        ],
        output_expr=Var(ty=IRType.REAL, name="eta"),
    )
    ps.add(sp)
    return ps


# ═══════════════════════════════════════════════════════════════════════════
# RepairVerdict enum
# ═══════════════════════════════════════════════════════════════════════════


class TestRepairVerdict:
    def test_all_values(self):
        assert RepairVerdict.SUCCESS is not None
        assert RepairVerdict.NO_REPAIR is not None
        assert RepairVerdict.TIMEOUT is not None
        assert RepairVerdict.ERROR is not None


# ═══════════════════════════════════════════════════════════════════════════
# RepairResult
# ═══════════════════════════════════════════════════════════════════════════


class TestRepairResult:
    def test_default_no_repair(self):
        r = RepairResult()
        assert r.verdict == RepairVerdict.NO_REPAIR
        assert not r.success

    def test_successful_repair(self):
        r = RepairResult(
            verdict=RepairVerdict.SUCCESS,
            parameter_values={"scale": 1.0},
            repair_cost=0.9,
        )
        assert r.success

    def test_summary(self):
        r = RepairResult()
        s = r.summary()
        assert isinstance(s, str)

    def test_str(self):
        r = RepairResult()
        assert isinstance(str(r), str)


# ═══════════════════════════════════════════════════════════════════════════
# RepairStatistics
# ═══════════════════════════════════════════════════════════════════════════


class TestRepairStatistics:
    def test_default(self):
        s = RepairStatistics()
        assert s.cegis_iterations == 0
        assert s.templates_tried == 0

    def test_summary(self):
        s = RepairStatistics(cegis_iterations=5, templates_tried=3)
        d = s.summary()
        assert isinstance(d, dict)

    def test_str(self):
        s = RepairStatistics()
        assert isinstance(str(s), str)


# ═══════════════════════════════════════════════════════════════════════════
# CounterexampleAccumulator
# ═══════════════════════════════════════════════════════════════════════════


class TestCounterexampleAccumulator:
    def test_empty(self):
        acc = CounterexampleAccumulator()
        assert acc.is_empty()
        assert acc.size() == 0

    def test_add(self):
        acc = CounterexampleAccumulator()
        cex = _make_cex()
        added = acc.add(cex)
        assert added is True
        assert acc.size() == 1

    def test_add_duplicate(self):
        acc = CounterexampleAccumulator()
        cex1 = _make_cex()
        cex2 = _make_cex()
        acc.add(cex1)
        added = acc.add(cex2)
        assert added is False
        assert acc.size() == 1

    def test_add_from_model(self):
        acc = CounterexampleAccumulator()
        added = acc.add_from_model({"q": 5.0, "eta": 5.5}, path_id=0, density_value=2.0)
        assert added is True

    def test_all(self):
        acc = CounterexampleAccumulator()
        acc.add(_make_cex(path_id=0))
        acc.add(_make_cex(path_id=1))
        all_cexs = acc.all()
        assert len(all_cexs) == 2

    def test_clear(self):
        acc = CounterexampleAccumulator()
        acc.add(_make_cex())
        acc.clear()
        assert acc.is_empty()

    def test_max_size_eviction(self):
        acc = CounterexampleAccumulator(max_size=3)
        for i in range(5):
            acc.add_from_model({"q": float(i)}, path_id=i, density_value=float(i))
        assert acc.size() <= 3

    def test_constraints_for_template(self):
        acc = CounterexampleAccumulator()
        acc.add(_make_cex())
        site = RepairSite(node_id=1, node_type="NoiseDrawNode")
        template = ScaleParam(site=site, original_scale=0.1)
        budget = PureBudget(epsilon=1.0)
        constraints = acc.constraints_for_template(template, budget)
        assert isinstance(constraints, list)


# ═══════════════════════════════════════════════════════════════════════════
# CostFunction
# ═══════════════════════════════════════════════════════════════════════════


class TestCostFunction:
    def test_construction(self):
        cf = CostFunction()
        assert cf is not None

    def test_compute(self):
        cf = CostFunction()
        site = RepairSite(node_id=1, node_type="NoiseDrawNode")
        template = ScaleParam(site=site, original_scale=1.0)
        cost = cf.compute(template, {"new_scale": 2.0}, {"new_scale": 1.0})
        assert cost >= 0

    def test_custom_weights(self):
        cf = CostFunction(weights={"new_scale": 10.0})
        site = RepairSite(node_id=1, node_type="NoiseDrawNode")
        template = ScaleParam(site=site, original_scale=1.0)
        cost = cf.compute(template, {"new_scale": 2.0}, {"new_scale": 1.0})
        assert cost >= 0

    def test_symbolic_cost(self):
        cf = CostFunction()
        site = RepairSite(node_id=1, node_type="NoiseDrawNode")
        template = ScaleParam(site=site, original_scale=1.0)
        expr = cf.symbolic_cost(template, {"new_scale": 1.0})
        assert expr is not None


# ═══════════════════════════════════════════════════════════════════════════
# RepairMinimizer
# ═══════════════════════════════════════════════════════════════════════════


class TestRepairMinimizer:
    def test_construction(self):
        rm = RepairMinimizer()
        assert rm is not None

    def test_minimize(self):
        rm = RepairMinimizer()
        site = RepairSite(node_id=1, node_type="NoiseDrawNode")
        template = ScaleParam(site=site, original_scale=0.1)
        result = rm.minimize(
            template=template,
            constraints=[],
            original_values={"new_scale": 0.1},
            timeout_ms=5000,
        )
        assert isinstance(result, MinimizationResult)

    def test_minimization_result_str(self):
        r = MinimizationResult(success=True, parameter_values={"s": 1.0}, cost=0.5)
        assert isinstance(str(r), str)


# ═══════════════════════════════════════════════════════════════════════════
# RepairVerifier
# ═══════════════════════════════════════════════════════════════════════════


class TestRepairVerifier:
    def test_construction(self):
        rv = RepairVerifier()
        assert rv is not None

    def test_quick_check(self):
        rv = RepairVerifier()
        mech = _buggy_laplace(scale=1.0)
        budget = PureBudget(epsilon=1.0)
        result = rv.quick_check(mech, budget)
        assert isinstance(result, bool)


# ═══════════════════════════════════════════════════════════════════════════
# SynthesizerConfig
# ═══════════════════════════════════════════════════════════════════════════


class TestSynthesizerConfig:
    def test_default(self):
        cfg = SynthesizerConfig()
        assert cfg.max_cegis_iterations > 0
        assert cfg.max_templates > 0

    def test_custom(self):
        cfg = SynthesizerConfig(max_cegis_iterations=10, max_templates=5, timeout_seconds=30.0)
        assert cfg.max_cegis_iterations == 10


# ═══════════════════════════════════════════════════════════════════════════
# RepairSynthesizer
# ═══════════════════════════════════════════════════════════════════════════


class TestRepairSynthesizer:
    def test_construction(self):
        synth = RepairSynthesizer()
        assert synth is not None

    def test_construction_with_config(self):
        cfg = SynthesizerConfig(max_cegis_iterations=5, timeout_seconds=10.0)
        synth = RepairSynthesizer(config=cfg)
        assert synth is not None

    def test_synthesize_returns_result(self):
        cfg = SynthesizerConfig(max_cegis_iterations=3, timeout_seconds=10.0, max_templates=2)
        synth = RepairSynthesizer(config=cfg)
        mech = _buggy_laplace(scale=0.1)
        budget = PureBudget(epsilon=1.0)
        result = synth.synthesize(mech, budget)
        assert isinstance(result, RepairResult)

    def test_synthesize_with_template(self):
        cfg = SynthesizerConfig(max_cegis_iterations=3, timeout_seconds=10.0)
        synth = RepairSynthesizer(config=cfg)
        mech = _buggy_laplace(scale=0.1)
        budget = PureBudget(epsilon=1.0)
        site = RepairSite(
            node_id=mech.body.stmts[1].node_id,
            node_type="NoiseDrawNode",
        )
        template = ScaleParam(site=site, original_scale=0.1)
        result = synth.synthesize_with_template(mech, budget, template)
        assert isinstance(result, RepairResult)

    def test_get_statistics(self):
        synth = RepairSynthesizer()
        stats = synth.get_statistics()
        assert isinstance(stats, RepairStatistics)

    def test_no_repair_case(self):
        """Try to repair an already-correct mechanism – should still return."""
        cfg = SynthesizerConfig(max_cegis_iterations=2, timeout_seconds=5.0)
        synth = RepairSynthesizer(config=cfg)
        mech = _buggy_laplace(scale=1.0)
        budget = PureBudget(epsilon=1.0)
        result = synth.synthesize(mech, budget)
        assert isinstance(result, RepairResult)
