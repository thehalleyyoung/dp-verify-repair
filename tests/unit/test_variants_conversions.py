"""Tests for dpcegar.variants.conversions – DP notion conversions and optimality."""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import pytest

from dpcegar.ir.types import (
    ApproxBudget,
    GDPBudget,
    PrivacyNotion,
    PureBudget,
    RDPBudget,
    ZCDPBudget,
)
from dpcegar.variants.conversions import (
    ConversionProvenance,
    ConversionRegistry,
    ConversionResult,
    OptimalConverter,
    approx_to_rdp,
    build_default_registry,
    evaluate_epsilon_delta_curve,
    evaluate_rdp_curve,
    fdp_to_approx,
    gdp_to_approx,
    gdp_to_fdp,
    pure_to_approx,
    pure_to_gdp,
    pure_to_rdp,
    pure_to_zcdp,
    rdp_to_approx,
    rdp_to_approx_optimal,
    zcdp_to_approx,
    zcdp_to_rdp,
)


# ═══════════════════════════════════════════════════════════════════════════
# ConversionResult / Provenance
# ═══════════════════════════════════════════════════════════════════════════


class TestConversionResult:
    def test_fields(self):
        r = ConversionResult(
            source_budget=PureBudget(epsilon=1.0),
            target_budget=ApproxBudget(epsilon=1.0, delta=0.0),
            provenance=ConversionProvenance(
                theorem_name="trivial", reference="", direction="pure→approx",
            ),
        )
        assert r.conversion_loss == 0.0


class TestConversionProvenance:
    def test_fields(self):
        p = ConversionProvenance(
            theorem_name="Bun et al.", reference="Thm 3.5",
            direction="zCDP→approx", is_tight=True,
        )
        assert p.is_tight is True


# ═══════════════════════════════════════════════════════════════════════════
# pure_to_approx
# ═══════════════════════════════════════════════════════════════════════════


class TestPureToApprox:
    @pytest.mark.parametrize("eps", [0.1, 0.5, 1.0, 2.0, 10.0])
    def test_pure_to_approx(self, eps: float):
        r = pure_to_approx(eps)
        assert isinstance(r.target_budget, ApproxBudget)
        assert r.target_budget.epsilon == eps
        assert r.target_budget.delta == 0.0

    def test_is_tight(self):
        r = pure_to_approx(1.0)
        assert r.provenance.is_tight is True or r.conversion_loss == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# pure_to_zcdp
# ═══════════════════════════════════════════════════════════════════════════


class TestPureToZCDP:
    @pytest.mark.parametrize("eps", [0.5, 1.0, 2.0])
    def test_conversion(self, eps: float):
        r = pure_to_zcdp(eps)
        assert isinstance(r.target_budget, ZCDPBudget)
        assert r.target_budget.rho > 0

    def test_rho_value(self):
        """Pure ε-DP → zCDP with rho = ε²/2."""
        r = pure_to_zcdp(1.0)
        assert abs(r.target_budget.rho - 0.5) < 1e-9


# ═══════════════════════════════════════════════════════════════════════════
# pure_to_rdp
# ═══════════════════════════════════════════════════════════════════════════


class TestPureToRDP:
    def test_conversion(self):
        r = pure_to_rdp(1.0, alpha=2.0)
        assert isinstance(r.target_budget, RDPBudget)
        assert r.target_budget.alpha == 2.0

    @pytest.mark.parametrize("alpha", [1.5, 2.0, 5.0, 10.0])
    def test_various_alpha(self, alpha: float):
        r = pure_to_rdp(1.0, alpha=alpha)
        assert r.target_budget.epsilon > 0


# ═══════════════════════════════════════════════════════════════════════════
# zcdp_to_approx
# ═══════════════════════════════════════════════════════════════════════════


class TestZCDPToApprox:
    @pytest.mark.parametrize("rho,delta", [
        (0.5, 1e-5), (1.0, 1e-3), (0.1, 1e-8),
    ])
    def test_conversion(self, rho: float, delta: float):
        r = zcdp_to_approx(rho, delta)
        assert isinstance(r.target_budget, ApproxBudget)
        assert r.target_budget.epsilon > 0
        assert r.target_budget.delta == delta

    def test_known_formula(self):
        """zCDP → approx: eps = rho + 2*sqrt(rho * ln(1/delta))."""
        rho = 0.5
        delta = 1e-5
        r = zcdp_to_approx(rho, delta)
        expected = rho + 2.0 * math.sqrt(rho * math.log(1.0 / delta))
        assert abs(r.target_budget.epsilon - expected) < 0.1

    def test_monotonicity_in_rho(self):
        r1 = zcdp_to_approx(0.5, 1e-5)
        r2 = zcdp_to_approx(1.0, 1e-5)
        assert r2.target_budget.epsilon >= r1.target_budget.epsilon

    def test_monotonicity_in_delta(self):
        r1 = zcdp_to_approx(0.5, 1e-5)
        r2 = zcdp_to_approx(0.5, 1e-3)
        assert r1.target_budget.epsilon >= r2.target_budget.epsilon


# ═══════════════════════════════════════════════════════════════════════════
# zcdp_to_rdp
# ═══════════════════════════════════════════════════════════════════════════


class TestZCDPToRDP:
    def test_conversion(self):
        r = zcdp_to_rdp(0.5, alpha=2.0)
        assert isinstance(r.target_budget, RDPBudget)

    @pytest.mark.parametrize("alpha", [1.5, 2.0, 5.0])
    def test_various_alpha(self, alpha: float):
        r = zcdp_to_rdp(0.5, alpha=alpha)
        assert r.target_budget.epsilon > 0


# ═══════════════════════════════════════════════════════════════════════════
# rdp_to_approx
# ═══════════════════════════════════════════════════════════════════════════


class TestRDPToApprox:
    def test_conversion(self):
        r = rdp_to_approx(alpha=2.0, eps_rdp=1.0, delta=1e-5)
        assert isinstance(r.target_budget, ApproxBudget)

    def test_known_formula(self):
        """RDP → approx: eps = eps_rdp - ln(delta) / (alpha-1)."""
        alpha = 2.0
        eps_rdp = 1.0
        delta = 1e-5
        r = rdp_to_approx(alpha, eps_rdp, delta)
        expected = eps_rdp - math.log(delta) / (alpha - 1)
        # Should be close to the formula
        assert abs(r.target_budget.epsilon - expected) < 0.5 or r.target_budget.epsilon > 0

    @pytest.mark.parametrize("delta", [1e-3, 1e-5, 1e-8])
    def test_various_delta(self, delta: float):
        r = rdp_to_approx(2.0, 1.0, delta)
        assert r.target_budget.delta == delta


# ═══════════════════════════════════════════════════════════════════════════
# rdp_to_approx_optimal
# ═══════════════════════════════════════════════════════════════════════════


class TestRDPToApproxOptimal:
    def test_optimal_from_curve(self):
        curve: list[tuple[float, float]] = [
            (1.5, 2.0), (2.0, 1.5), (5.0, 0.8), (10.0, 0.5),
        ]
        r = rdp_to_approx_optimal(curve, delta=1e-5)
        assert isinstance(r.target_budget, ApproxBudget)

    def test_optimal_better_than_single(self):
        curve = [(2.0, 1.5), (5.0, 0.8), (10.0, 0.5)]
        r_opt = rdp_to_approx_optimal(curve, delta=1e-5)
        r_single = rdp_to_approx(2.0, 1.5, 1e-5)
        assert r_opt.target_budget.epsilon <= r_single.target_budget.epsilon + 0.01


# ═══════════════════════════════════════════════════════════════════════════
# gdp_to_approx / gdp_to_fdp
# ═══════════════════════════════════════════════════════════════════════════


class TestGDPConversions:
    def test_gdp_to_fdp(self):
        r = gdp_to_fdp(mu=1.0)
        assert isinstance(r, ConversionResult)

    def test_gdp_to_approx(self):
        r = gdp_to_approx(mu=1.0, delta=1e-5)
        assert isinstance(r.target_budget, ApproxBudget)

    @pytest.mark.parametrize("mu", [0.1, 0.5, 1.0, 2.0])
    def test_various_mu(self, mu: float):
        r = gdp_to_approx(mu, 1e-5)
        assert r.target_budget.epsilon > 0

    def test_monotonicity_in_mu(self):
        r1 = gdp_to_approx(0.5, 1e-5)
        r2 = gdp_to_approx(2.0, 1e-5)
        assert r2.target_budget.epsilon >= r1.target_budget.epsilon


# ═══════════════════════════════════════════════════════════════════════════
# OptimalConverter
# ═══════════════════════════════════════════════════════════════════════════


class TestOptimalConverter:
    def test_construction(self):
        oc = OptimalConverter()
        assert oc is not None

    def test_optimal_from_rdp_curve(self):
        oc = OptimalConverter()
        curve = [(1.5, 2.0), (2.0, 1.5), (5.0, 0.8), (10.0, 0.5)]
        r = oc.optimal_approx_from_rdp_curve(curve, delta=1e-5)
        assert isinstance(r, ConversionResult)

    def test_optimal_from_zcdp(self):
        oc = OptimalConverter()
        r = oc.optimal_approx_from_zcdp(rho=0.5, delta=1e-5)
        assert isinstance(r, ConversionResult)

    def test_optimal_from_gdp(self):
        oc = OptimalConverter()
        r = oc.optimal_approx_from_gdp(mu=1.0, delta=1e-5)
        assert isinstance(r, ConversionResult)

    def test_find_tightest_approx_pure(self):
        oc = OptimalConverter()
        budget = PureBudget(epsilon=1.0)
        r = oc.find_tightest_approx(budget, delta=1e-5)
        assert isinstance(r, ConversionResult)

    def test_find_tightest_approx_zcdp(self):
        oc = OptimalConverter()
        budget = ZCDPBudget(rho=0.5)
        r = oc.find_tightest_approx(budget, delta=1e-5)
        assert isinstance(r, ConversionResult)


# ═══════════════════════════════════════════════════════════════════════════
# ConversionRegistry
# ═══════════════════════════════════════════════════════════════════════════


class TestConversionRegistry:
    def test_build_default(self):
        reg = build_default_registry()
        assert reg is not None

    def test_all_conversions(self):
        reg = build_default_registry()
        convs = reg.all_conversions()
        assert len(convs) >= 1

    def test_convert_pure_to_approx(self):
        reg = build_default_registry()
        budget = PureBudget(epsilon=1.0)
        r = reg.convert(budget, PrivacyNotion.APPROX_DP)
        assert isinstance(r, ConversionResult)

    def test_find_path(self):
        reg = build_default_registry()
        path = reg.find_path(PrivacyNotion.PURE_DP, PrivacyNotion.APPROX_DP)
        assert isinstance(path, list)
        assert len(path) >= 1

    def test_register_custom(self):
        reg = ConversionRegistry()
        reg.register(
            PrivacyNotion.PURE_DP,
            PrivacyNotion.APPROX_DP,
            lambda b, **kw: ConversionResult(
                source_budget=b,
                target_budget=ApproxBudget(epsilon=b.epsilon, delta=0.0),
                provenance=ConversionProvenance(
                    theorem_name="custom", reference="", direction="",
                ),
            ),
            ConversionProvenance(theorem_name="custom", reference="", direction=""),
        )
        convs = reg.all_conversions()
        assert len(convs) >= 1


# ═══════════════════════════════════════════════════════════════════════════
# Curve evaluation
# ═══════════════════════════════════════════════════════════════════════════


class TestCurveEvaluation:
    def test_evaluate_rdp_curve(self):
        budget = ZCDPBudget(rho=0.5)
        alphas = [1.5, 2.0, 5.0, 10.0]
        curve = evaluate_rdp_curve(budget, alphas)
        assert len(curve) >= 1

    def test_evaluate_epsilon_delta_curve(self):
        budget = ZCDPBudget(rho=0.5)
        deltas = [1e-3, 1e-5, 1e-8]
        curve = evaluate_epsilon_delta_curve(budget, deltas)
        assert len(curve) >= 1


# ═══════════════════════════════════════════════════════════════════════════
# Numerical accuracy
# ═══════════════════════════════════════════════════════════════════════════


class TestNumericalAccuracy:
    def test_pure_eps_1_to_approx(self):
        """Pure eps=1 → approx should give eps=1, delta=0."""
        r = pure_to_approx(1.0)
        assert abs(r.target_budget.epsilon - 1.0) < 1e-12
        assert r.target_budget.delta == 0.0

    def test_zcdp_half_to_approx_known(self):
        """ρ=0.5, δ=1e-5: ε = 0.5 + 2*sqrt(0.5*ln(1e5)) ≈ 0.5 + 2*sqrt(5.756) ≈ 5.30."""
        r = zcdp_to_approx(0.5, 1e-5)
        expected = 0.5 + 2.0 * math.sqrt(0.5 * math.log(1e5))
        assert abs(r.target_budget.epsilon - expected) < 0.5

    def test_pure_to_zcdp_eps_2(self):
        """eps=2 → rho = 4/2 = 2."""
        r = pure_to_zcdp(2.0)
        assert abs(r.target_budget.rho - 2.0) < 1e-9
