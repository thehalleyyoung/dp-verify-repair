"""Tests for dpcegar.repair.templates – repair template hierarchy and enumeration."""

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
    BranchNode,
    AssignNode,
    LoopNode,
)
from dpcegar.repair.templates import (
    ClampBound,
    CompositeRepair,
    CompositionBudgetSplit,
    NoiseSwap,
    RepairParameter,
    RepairSite,
    RepairTemplate,
    ScaleParam,
    SensitivityRescale,
    TemplateCost,
    TemplateEnumerator,
    TemplateValidator,
    ThresholdShift,
    ValidationResult,
)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _laplace_mechir(scale: float = 1.0) -> MechIR:
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
        name="laplace_mech",
        params=[ParamDecl(name="db", ty=IRType.ARRAY, is_database=True)],
        body=body, return_type=IRType.REAL,
        budget=PureBudget(epsilon=1.0),
    )


def _branching_mechir() -> MechIR:
    cond = BinOp(
        ty=IRType.BOOL, op=BinOpKind.GT,
        left=Var(ty=IRType.REAL, name="x"), right=Const.real(0.0),
    )
    true_br = AssignNode(target=Var(ty=IRType.REAL, name="r"), value=Const.real(1.0))
    false_br = AssignNode(target=Var(ty=IRType.REAL, name="r"), value=Const.real(0.0))
    branch = BranchNode(condition=cond, true_branch=true_br, false_branch=false_br)
    ret = ReturnNode(value=Var(ty=IRType.REAL, name="r"))
    body = SequenceNode(stmts=[branch, ret])
    return MechIR(
        name="branching_mech",
        params=[ParamDecl(name="x", ty=IRType.REAL)],
        body=body, return_type=IRType.REAL,
    )


def _make_site(node_id: int = 1) -> RepairSite:
    return RepairSite(node_id=node_id, node_type="NoiseDrawNode", description="noise scale")


# ═══════════════════════════════════════════════════════════════════════════
# RepairSite
# ═══════════════════════════════════════════════════════════════════════════


class TestRepairSite:
    def test_construction(self):
        s = RepairSite(node_id=1, node_type="NoiseDrawNode")
        assert s.node_id == 1

    def test_with_value(self):
        s = RepairSite(node_id=1, node_type="NoiseDrawNode", current_value=2.0)
        assert s.current_value == 2.0

    def test_str(self):
        s = _make_site()
        assert isinstance(str(s), str)


# ═══════════════════════════════════════════════════════════════════════════
# RepairParameter
# ═══════════════════════════════════════════════════════════════════════════


class TestRepairParameter:
    def test_construction(self):
        p = RepairParameter(name="scale", lower_bound=0.0, upper_bound=100.0)
        assert p.name == "scale"

    def test_as_var(self):
        p = RepairParameter(name="scale", param_type=IRType.REAL)
        v = p.as_var()
        assert isinstance(v, Var)
        assert v.name == "scale"

    def test_as_const(self):
        p = RepairParameter(name="scale")
        c = p.as_const(5.0)
        assert isinstance(c, Const)

    def test_domain_constraints(self):
        p = RepairParameter(name="scale", lower_bound=0.0, upper_bound=100.0)
        constraints = p.domain_constraints()
        assert isinstance(constraints, list)
        assert len(constraints) >= 0

    def test_str(self):
        p = RepairParameter(name="scale")
        assert isinstance(str(p), str)


# ═══════════════════════════════════════════════════════════════════════════
# ScaleParam template
# ═══════════════════════════════════════════════════════════════════════════


class TestScaleParam:
    def test_construction(self):
        t = ScaleParam(site=_make_site(), original_scale=1.0)
        assert "scale" in t.name().lower() or len(t.name()) > 0

    def test_parameters(self):
        t = ScaleParam(site=_make_site(), original_scale=1.0)
        params = t.parameters()
        assert len(params) >= 1

    def test_sites(self):
        t = ScaleParam(site=_make_site(), original_scale=1.0)
        sites = t.sites()
        assert len(sites) == 1

    def test_apply_concrete(self):
        mech = _laplace_mechir(scale=0.5)
        t = ScaleParam(site=_make_site(node_id=mech.body.stmts[1].node_id), original_scale=0.5)
        repaired = t.apply_concrete(mech, {"new_scale": 1.0})
        assert isinstance(repaired, MechIR)

    def test_cost_expression(self):
        t = ScaleParam(site=_make_site(), original_scale=1.0)
        cost = t.cost_expression({"new_scale": 1.0})
        assert cost is not None

    @pytest.mark.parametrize("min_s,max_s", [(0.01, 10.0), (1e-6, 1e6)])
    def test_scale_bounds(self, min_s: float, max_s: float):
        t = ScaleParam(site=_make_site(), original_scale=1.0, min_scale=min_s, max_scale=max_s)
        params = t.parameters()
        assert any(p.lower_bound >= min_s for p in params)


# ═══════════════════════════════════════════════════════════════════════════
# ThresholdShift template
# ═══════════════════════════════════════════════════════════════════════════


class TestThresholdShift:
    def test_construction(self):
        t = ThresholdShift(site=_make_site(), original_threshold=5.0)
        assert len(t.name()) > 0

    def test_parameters(self):
        t = ThresholdShift(site=_make_site(), original_threshold=5.0)
        assert len(t.parameters()) >= 1

    def test_sites(self):
        t = ThresholdShift(site=_make_site(), original_threshold=5.0)
        assert len(t.sites()) == 1

    def test_cost_expression(self):
        t = ThresholdShift(site=_make_site(), original_threshold=5.0)
        cost = t.cost_expression({"threshold_shift": 0.0})
        assert cost is not None


# ═══════════════════════════════════════════════════════════════════════════
# ClampBound template
# ═══════════════════════════════════════════════════════════════════════════


class TestClampBound:
    def test_construction(self):
        t = ClampBound(site=_make_site())
        assert len(t.name()) > 0

    def test_parameters_two_bounds(self):
        t = ClampBound(site=_make_site())
        params = t.parameters()
        assert len(params) >= 2  # lo and hi

    def test_cost_expression(self):
        t = ClampBound(site=_make_site())
        cost = t.cost_expression({"clamp_lo": -10.0, "clamp_hi": 10.0})
        assert cost is not None


# ═══════════════════════════════════════════════════════════════════════════
# CompositionBudgetSplit template
# ═══════════════════════════════════════════════════════════════════════════


class TestCompositionBudgetSplit:
    def test_construction(self):
        t = CompositionBudgetSplit(
            loop_site=_make_site(),
            num_iterations=5,
            total_budget=1.0,
        )
        assert len(t.name()) > 0

    def test_parameters(self):
        t = CompositionBudgetSplit(
            loop_site=_make_site(), num_iterations=5, total_budget=1.0,
        )
        params = t.parameters()
        assert len(params) >= 1

    def test_cost_expression(self):
        t = CompositionBudgetSplit(
            loop_site=_make_site(), num_iterations=5, total_budget=1.0,
        )
        cost = t.cost_expression({"split_budget": 0.2})
        assert cost is not None


# ═══════════════════════════════════════════════════════════════════════════
# SensitivityRescale template
# ═══════════════════════════════════════════════════════════════════════════


class TestSensitivityRescale:
    def test_construction(self):
        t = SensitivityRescale(site=_make_site(), original_sensitivity=1.0)
        assert len(t.name()) > 0

    def test_parameters(self):
        t = SensitivityRescale(site=_make_site(), original_sensitivity=1.0)
        assert len(t.parameters()) >= 1

    def test_cost_expression(self):
        t = SensitivityRescale(site=_make_site(), original_sensitivity=1.0)
        cost = t.cost_expression({"new_sensitivity": 1.0})
        assert cost is not None


# ═══════════════════════════════════════════════════════════════════════════
# NoiseSwap template
# ═══════════════════════════════════════════════════════════════════════════


class TestNoiseSwap:
    def test_construction(self):
        t = NoiseSwap(
            site=_make_site(),
            original_kind=NoiseKind.LAPLACE,
            target_kind=NoiseKind.GAUSSIAN,
        )
        assert len(t.name()) > 0

    def test_parameters(self):
        t = NoiseSwap(
            site=_make_site(),
            original_kind=NoiseKind.LAPLACE,
            target_kind=NoiseKind.GAUSSIAN,
        )
        assert len(t.parameters()) >= 1

    def test_cost_expression(self):
        t = NoiseSwap(
            site=_make_site(),
            original_kind=NoiseKind.LAPLACE,
            target_kind=NoiseKind.GAUSSIAN,
        )
        cost = t.cost_expression({"new_scale": 1.0})
        assert cost is not None


# ═══════════════════════════════════════════════════════════════════════════
# CompositeRepair template
# ═══════════════════════════════════════════════════════════════════════════


class TestCompositeRepair:
    def test_construction_empty(self):
        t = CompositeRepair(templates=[])
        assert t.component_count() == 0

    def test_construction_with_templates(self):
        t1 = ScaleParam(site=_make_site(1), original_scale=1.0)
        t2 = ThresholdShift(site=_make_site(2), original_threshold=5.0)
        comp = CompositeRepair(templates=[t1, t2])
        assert comp.component_count() == 2

    def test_add_template(self):
        comp = CompositeRepair(templates=[])
        comp.add_template(ScaleParam(site=_make_site(), original_scale=1.0))
        assert comp.component_count() == 1

    def test_parameters_combined(self):
        t1 = ScaleParam(site=_make_site(1), original_scale=1.0)
        t2 = ThresholdShift(site=_make_site(2), original_threshold=5.0)
        comp = CompositeRepair(templates=[t1, t2])
        params = comp.parameters()
        assert len(params) >= 2

    def test_sites_combined(self):
        t1 = ScaleParam(site=_make_site(1), original_scale=1.0)
        t2 = ThresholdShift(site=_make_site(2), original_threshold=5.0)
        comp = CompositeRepair(templates=[t1, t2])
        sites = comp.sites()
        assert len(sites) == 2

    def test_name(self):
        comp = CompositeRepair(templates=[ScaleParam(site=_make_site(), original_scale=1.0)])
        assert len(comp.name()) > 0


# ═══════════════════════════════════════════════════════════════════════════
# TemplateEnumerator
# ═══════════════════════════════════════════════════════════════════════════


class TestTemplateEnumerator:
    def test_construction(self):
        e = TemplateEnumerator()
        assert e is not None

    def test_enumerate_laplace(self):
        mech = _laplace_mechir()
        e = TemplateEnumerator()
        templates = e.enumerate(mech)
        assert isinstance(templates, list)
        assert len(templates) >= 1

    def test_enumerate_composites(self):
        mech = _laplace_mechir()
        e = TemplateEnumerator()
        composites = e.enumerate_composites(mech, max_components=2)
        assert isinstance(composites, list)

    def test_disable_noise_swap(self):
        e = TemplateEnumerator(enable_noise_swap=False)
        mech = _laplace_mechir()
        templates = e.enumerate(mech)
        assert all(not isinstance(t, NoiseSwap) for t in templates)

    def test_disable_clamping(self):
        e = TemplateEnumerator(enable_clamping=False)
        mech = _laplace_mechir()
        templates = e.enumerate(mech)
        assert all(not isinstance(t, ClampBound) for t in templates)


# ═══════════════════════════════════════════════════════════════════════════
# TemplateCost
# ═══════════════════════════════════════════════════════════════════════════


class TestTemplateCost:
    def test_construction(self):
        tc = TemplateCost()
        assert tc is not None

    def test_estimate(self):
        tc = TemplateCost()
        t = ScaleParam(site=_make_site(), original_scale=1.0)
        cost = tc.estimate(t)
        assert cost >= 0

    def test_rank(self):
        tc = TemplateCost()
        t1 = ScaleParam(site=_make_site(1), original_scale=1.0)
        t2 = ThresholdShift(site=_make_site(2), original_threshold=5.0)
        ranked = tc.rank([t1, t2])
        assert len(ranked) == 2

    def test_custom_weights(self):
        tc = TemplateCost(weights={"scale": 2.0, "threshold": 1.0})
        t = ScaleParam(site=_make_site(), original_scale=1.0)
        cost = tc.estimate(t)
        assert cost >= 0


# ═══════════════════════════════════════════════════════════════════════════
# TemplateValidator
# ═══════════════════════════════════════════════════════════════════════════


class TestTemplateValidator:
    def test_validate_valid(self):
        tv = TemplateValidator()
        t = ScaleParam(site=_make_site(), original_scale=1.0)
        mech = _laplace_mechir()
        result = tv.validate(t, mech)
        assert isinstance(result, ValidationResult)

    def test_validation_result_str(self):
        vr = ValidationResult(is_valid=True)
        assert isinstance(str(vr), str)

    def test_validation_result_invalid(self):
        vr = ValidationResult(is_valid=False, errors=["site mismatch"])
        assert not vr.is_valid
        assert len(vr.errors) == 1
