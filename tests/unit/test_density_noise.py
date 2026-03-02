"""Unit tests for dpcegar.density.noise_models — noise distribution models."""

from __future__ import annotations

import math
import random
from typing import Any

import pytest

from dpcegar.ir.types import (
    Abs,
    BinOp,
    BinOpKind,
    Const,
    IRType,
    Log,
    NoiseKind,
    TypedExpr,
    Var,
)
from dpcegar.density.noise_models import (
    DiscreteGaussianNoise,
    DiscreteLaplaceNoise,
    ExponentialMechNoise,
    GaussianNoise,
    LaplaceNoise,
    MixtureNoise,
    NoiseModel,
    TruncatedGaussianNoise,
    TruncatedLaplaceNoise,
    get_noise_model,
    _NOISE_MODEL_REGISTRY,
)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _sym_vars():
    """Return standard symbolic variables for tests."""
    obs = Var(ty=IRType.REAL, name="o")
    c_d = Var(ty=IRType.REAL, name="c_d")
    c_dp = Var(ty=IRType.REAL, name="c_dp")
    scale = Var(ty=IRType.REAL, name="b")
    return obs, c_d, c_dp, scale


# ═══════════════════════════════════════════════════════════════════════════
# LaplaceNoise
# ═══════════════════════════════════════════════════════════════════════════


class TestLaplaceNoise:
    """Tests for LaplaceNoise."""

    def setup_method(self):
        self.model = LaplaceNoise()
        self.rng = random.Random(12345)

    # -- density -----------------------------------------------------------

    def test_density_at_center(self):
        """Density at the center should be 1/(2b)."""
        for b in [0.5, 1.0, 2.0, 10.0]:
            d = self.model.density(0.0, 0.0, b)
            assert d == pytest.approx(1.0 / (2.0 * b))

    def test_density_symmetry(self):
        """Laplace density is symmetric around the center."""
        d_plus = self.model.density(3.0, 0.0, 1.0)
        d_minus = self.model.density(-3.0, 0.0, 1.0)
        assert d_plus == pytest.approx(d_minus)

    def test_density_positive(self):
        """Density should always be positive."""
        for x in [-10.0, -1.0, 0.0, 1.0, 10.0]:
            assert self.model.density(x, 0.0, 1.0) > 0

    def test_density_shifted_center(self):
        """Density at center=5, x=5 equals density at center=0, x=0."""
        d1 = self.model.density(5.0, 5.0, 1.0)
        d2 = self.model.density(0.0, 0.0, 1.0)
        assert d1 == pytest.approx(d2)

    def test_density_zero_scale_raises(self):
        with pytest.raises(ValueError, match="positive"):
            self.model.density(0.0, 0.0, 0.0)

    def test_density_negative_scale_raises(self):
        with pytest.raises(ValueError, match="positive"):
            self.model.density(0.0, 0.0, -1.0)

    # -- log_density -------------------------------------------------------

    def test_log_density_at_center(self):
        d = self.model.log_density(0.0, 0.0, 1.0)
        assert d == pytest.approx(-math.log(2.0))

    def test_log_density_matches_density(self):
        for x in [-2.0, 0.0, 3.5]:
            ld = self.model.log_density(x, 1.0, 2.0)
            d = self.model.density(x, 1.0, 2.0)
            assert ld == pytest.approx(math.log(d))

    def test_log_density_zero_scale_raises(self):
        with pytest.raises(ValueError, match="positive"):
            self.model.log_density(0.0, 0.0, 0.0)

    # -- log_ratio ---------------------------------------------------------

    def test_log_ratio_same_center_is_zero(self):
        lr = self.model.log_ratio(5.0, 1.0, 1.0, 1.0)
        assert lr == pytest.approx(0.0)

    def test_log_ratio_formula(self):
        """log_ratio = (|x-c2| - |x-c1|) / b"""
        x, c1, c2, b = 3.0, 0.0, 1.0, 2.0
        expected = (abs(x - c2) - abs(x - c1)) / b
        assert self.model.log_ratio(x, c1, c2, b) == pytest.approx(expected)

    def test_log_ratio_bounded_by_sensitivity_over_scale(self):
        """For Laplace, |log_ratio| <= |c1 - c2| / b."""
        c1, c2, b = 0.0, 1.0, 2.0
        for x in [-10.0, -1.0, 0.0, 0.5, 1.0, 5.0, 10.0]:
            lr = self.model.log_ratio(x, c1, c2, b)
            assert abs(lr) <= abs(c1 - c2) / b + 1e-12

    def test_log_ratio_antisymmetric(self):
        """Swapping centers negates the ratio."""
        lr1 = self.model.log_ratio(2.0, 0.0, 1.0, 1.0)
        lr2 = self.model.log_ratio(2.0, 1.0, 0.0, 1.0)
        assert lr1 == pytest.approx(-lr2)

    def test_log_ratio_zero_scale_raises(self):
        with pytest.raises(ValueError, match="positive"):
            self.model.log_ratio(0.0, 0.0, 1.0, 0.0)

    # -- cdf ---------------------------------------------------------------

    def test_cdf_at_center_is_half(self):
        assert self.model.cdf(0.0, 0.0, 1.0) == pytest.approx(0.5)

    def test_cdf_monotone(self):
        prev = 0.0
        for x in [-5.0, -2.0, 0.0, 2.0, 5.0]:
            c = self.model.cdf(x, 0.0, 1.0)
            assert c >= prev
            prev = c

    def test_cdf_limits(self):
        assert self.model.cdf(-1000.0, 0.0, 1.0) < 1e-100
        assert self.model.cdf(1000.0, 0.0, 1.0) > 1.0 - 1e-10

    def test_cdf_zero_scale_raises(self):
        with pytest.raises(ValueError, match="positive"):
            self.model.cdf(0.0, 0.0, 0.0)

    # -- sample ------------------------------------------------------------

    def test_sample_returns_float(self):
        s = self.model.sample(0.0, 1.0, self.rng)
        assert isinstance(s, float)

    def test_sample_mean_approx_center(self):
        samples = [self.model.sample(5.0, 1.0, self.rng) for _ in range(10000)]
        mean = sum(samples) / len(samples)
        assert mean == pytest.approx(5.0, abs=0.1)

    def test_sample_with_none_rng(self):
        s = self.model.sample(0.0, 1.0, None)
        assert isinstance(s, float)

    # -- symbolic_log_ratio ------------------------------------------------

    def test_symbolic_log_ratio_returns_typed_expr(self):
        obs, c_d, c_dp, scale = _sym_vars()
        expr = self.model.symbolic_log_ratio(obs, c_d, c_dp, scale)
        assert isinstance(expr, TypedExpr)

    def test_symbolic_log_ratio_is_division(self):
        obs, c_d, c_dp, scale = _sym_vars()
        expr = self.model.symbolic_log_ratio(obs, c_d, c_dp, scale)
        assert isinstance(expr, BinOp)
        assert expr.op == BinOpKind.DIV

    def test_symbolic_log_ratio_numerator_has_abs(self):
        obs, c_d, c_dp, scale = _sym_vars()
        expr = self.model.symbolic_log_ratio(obs, c_d, c_dp, scale)
        numer = expr.left
        assert isinstance(numer, BinOp)
        assert isinstance(numer.left, Abs) or isinstance(numer.right, Abs)

    # -- symbolic_log_density ----------------------------------------------

    def test_symbolic_log_density_returns_typed_expr(self):
        obs = Var(ty=IRType.REAL, name="o")
        center = Var(ty=IRType.REAL, name="c")
        scale = Var(ty=IRType.REAL, name="b")
        expr = self.model.symbolic_log_density(obs, center, scale)
        assert isinstance(expr, TypedExpr)

    # -- max_privacy_loss --------------------------------------------------

    def test_max_privacy_loss(self):
        assert self.model.max_privacy_loss(1.0, 1.0) == pytest.approx(1.0)
        assert self.model.max_privacy_loss(2.0, 1.0) == pytest.approx(2.0)
        assert self.model.max_privacy_loss(1.0, 2.0) == pytest.approx(0.5)

    def test_max_privacy_loss_zero_scale_raises(self):
        with pytest.raises(ValueError, match="positive"):
            self.model.max_privacy_loss(1.0, 0.0)


# ═══════════════════════════════════════════════════════════════════════════
# GaussianNoise
# ═══════════════════════════════════════════════════════════════════════════


class TestGaussianNoise:
    """Tests for GaussianNoise."""

    def setup_method(self):
        self.model = GaussianNoise()
        self.rng = random.Random(54321)

    # -- density -----------------------------------------------------------

    def test_density_at_center(self):
        sigma = 2.0
        d = self.model.density(0.0, 0.0, sigma)
        expected = 1.0 / (sigma * math.sqrt(2.0 * math.pi))
        assert d == pytest.approx(expected)

    def test_density_symmetry(self):
        d_plus = self.model.density(2.0, 0.0, 1.0)
        d_minus = self.model.density(-2.0, 0.0, 1.0)
        assert d_plus == pytest.approx(d_minus)

    def test_density_positive(self):
        for x in [-10.0, 0.0, 10.0]:
            assert self.model.density(x, 0.0, 1.0) > 0

    def test_density_zero_scale_raises(self):
        with pytest.raises(ValueError, match="positive"):
            self.model.density(0.0, 0.0, 0.0)

    def test_density_negative_scale_raises(self):
        with pytest.raises(ValueError, match="positive"):
            self.model.density(0.0, 0.0, -1.0)

    # -- log_density -------------------------------------------------------

    def test_log_density_at_center(self):
        ld = self.model.log_density(0.0, 0.0, 1.0)
        expected = -0.5 * math.log(2.0 * math.pi)
        assert ld == pytest.approx(expected)

    def test_log_density_matches_density(self):
        for x in [-3.0, 0.0, 1.5]:
            ld = self.model.log_density(x, 1.0, 2.0)
            d = self.model.density(x, 1.0, 2.0)
            assert ld == pytest.approx(math.log(d))

    def test_log_density_zero_scale_raises(self):
        with pytest.raises(ValueError, match="positive"):
            self.model.log_density(0.0, 0.0, 0.0)

    # -- log_ratio ---------------------------------------------------------

    def test_log_ratio_same_center_is_zero(self):
        lr = self.model.log_ratio(5.0, 2.0, 2.0, 1.0)
        assert lr == pytest.approx(0.0)

    def test_log_ratio_formula(self):
        """log_ratio = ((x-c2)^2 - (x-c1)^2) / (2*sigma^2)"""
        x, c1, c2, s = 3.0, 0.0, 1.0, 2.0
        expected = ((x - c2) ** 2 - (x - c1) ** 2) / (2.0 * s ** 2)
        assert self.model.log_ratio(x, c1, c2, s) == pytest.approx(expected)

    def test_log_ratio_antisymmetric(self):
        lr1 = self.model.log_ratio(2.0, 0.0, 1.0, 1.0)
        lr2 = self.model.log_ratio(2.0, 1.0, 0.0, 1.0)
        assert lr1 == pytest.approx(-lr2)

    def test_log_ratio_zero_scale_raises(self):
        with pytest.raises(ValueError, match="positive"):
            self.model.log_ratio(0.0, 0.0, 1.0, 0.0)

    # -- cdf ---------------------------------------------------------------

    def test_cdf_at_center_is_half(self):
        assert self.model.cdf(0.0, 0.0, 1.0) == pytest.approx(0.5)

    def test_cdf_monotone(self):
        prev = 0.0
        for x in [-5.0, -2.0, 0.0, 2.0, 5.0]:
            c = self.model.cdf(x, 0.0, 1.0)
            assert c >= prev
            prev = c

    def test_cdf_limits(self):
        assert self.model.cdf(-10.0, 0.0, 1.0) < 1e-10
        assert self.model.cdf(10.0, 0.0, 1.0) > 1.0 - 1e-10

    # -- sample ------------------------------------------------------------

    def test_sample_returns_float(self):
        s = self.model.sample(0.0, 1.0, self.rng)
        assert isinstance(s, float)

    def test_sample_mean_approx_center(self):
        samples = [self.model.sample(3.0, 1.0, self.rng) for _ in range(10000)]
        mean = sum(samples) / len(samples)
        assert mean == pytest.approx(3.0, abs=0.1)

    # -- symbolic_log_ratio ------------------------------------------------

    def test_symbolic_log_ratio_returns_typed_expr(self):
        obs, c_d, c_dp, scale = _sym_vars()
        expr = self.model.symbolic_log_ratio(obs, c_d, c_dp, scale)
        assert isinstance(expr, TypedExpr)

    def test_symbolic_log_ratio_is_division(self):
        obs, c_d, c_dp, scale = _sym_vars()
        expr = self.model.symbolic_log_ratio(obs, c_d, c_dp, scale)
        assert isinstance(expr, BinOp)
        assert expr.op == BinOpKind.DIV

    # -- symbolic_log_density ----------------------------------------------

    def test_symbolic_log_density_returns_typed_expr(self):
        obs = Var(ty=IRType.REAL, name="o")
        center = Var(ty=IRType.REAL, name="c")
        scale = Var(ty=IRType.REAL, name="s")
        expr = self.model.symbolic_log_density(obs, center, scale)
        assert isinstance(expr, TypedExpr)

    # -- renyi_divergence --------------------------------------------------

    def test_renyi_divergence_same_center_is_zero(self):
        assert self.model.renyi_divergence(0.0, 0.0, 1.0, 2.0) == pytest.approx(0.0)

    def test_renyi_divergence_formula(self):
        c1, c2, s, alpha = 0.0, 1.0, 2.0, 3.0
        expected = alpha * (c1 - c2) ** 2 / (2.0 * s ** 2)
        assert self.model.renyi_divergence(c1, c2, s, alpha) == pytest.approx(expected)

    def test_renyi_divergence_zero_scale_raises(self):
        with pytest.raises(ValueError, match="positive"):
            self.model.renyi_divergence(0.0, 1.0, 0.0, 2.0)

    # -- zcdp_rho ----------------------------------------------------------

    def test_zcdp_rho(self):
        assert self.model.zcdp_rho(1.0, 1.0) == pytest.approx(0.5)
        assert self.model.zcdp_rho(2.0, 1.0) == pytest.approx(2.0)
        assert self.model.zcdp_rho(1.0, 2.0) == pytest.approx(0.125)

    def test_zcdp_rho_zero_scale_raises(self):
        with pytest.raises(ValueError, match="positive"):
            self.model.zcdp_rho(1.0, 0.0)

    # -- gdp_mu ------------------------------------------------------------

    def test_gdp_mu(self):
        assert self.model.gdp_mu(1.0, 1.0) == pytest.approx(1.0)
        assert self.model.gdp_mu(2.0, 1.0) == pytest.approx(2.0)
        assert self.model.gdp_mu(1.0, 2.0) == pytest.approx(0.5)

    def test_gdp_mu_zero_scale_raises(self):
        with pytest.raises(ValueError, match="positive"):
            self.model.gdp_mu(1.0, 0.0)


# ═══════════════════════════════════════════════════════════════════════════
# ExponentialMechNoise
# ═══════════════════════════════════════════════════════════════════════════


class TestExponentialMechNoise:
    """Tests for ExponentialMechNoise."""

    def setup_method(self):
        self.model = ExponentialMechNoise()

    def test_density_positive(self):
        d = self.model.density(0.0, 2.0, 1.0)
        assert d > 0

    def test_density_formula(self):
        """density = exp(center / scale)."""
        d = self.model.density(0.0, 2.0, 1.0)
        assert d == pytest.approx(math.exp(2.0))

    def test_density_zero_scale_raises(self):
        with pytest.raises(ValueError, match="positive"):
            self.model.density(0.0, 1.0, 0.0)

    def test_log_density_formula(self):
        """log_density = center / scale."""
        ld = self.model.log_density(0.0, 3.0, 2.0)
        assert ld == pytest.approx(1.5)

    def test_log_density_zero_scale_raises(self):
        with pytest.raises(ValueError, match="positive"):
            self.model.log_density(0.0, 1.0, 0.0)

    def test_log_ratio_formula(self):
        """log_ratio = (center1 - center2) / scale."""
        lr = self.model.log_ratio(0.0, 5.0, 3.0, 2.0)
        assert lr == pytest.approx(1.0)

    def test_log_ratio_same_center_is_zero(self):
        lr = self.model.log_ratio(0.0, 3.0, 3.0, 1.0)
        assert lr == pytest.approx(0.0)

    def test_log_ratio_zero_scale_raises(self):
        with pytest.raises(ValueError, match="positive"):
            self.model.log_ratio(0.0, 1.0, 2.0, 0.0)

    def test_cdf_returns_half(self):
        """CDF not well-defined; returns 0.5."""
        assert self.model.cdf(0.0, 1.0, 1.0) == 0.5

    def test_sample_returns_center(self):
        """Sample not sampleable; returns center."""
        assert self.model.sample(7.0, 1.0) == 7.0

    def test_symbolic_log_ratio_structure(self):
        obs, c_d, c_dp, scale = _sym_vars()
        expr = self.model.symbolic_log_ratio(obs, c_d, c_dp, scale)
        assert isinstance(expr, BinOp)
        assert expr.op == BinOpKind.DIV
        numer = expr.left
        assert isinstance(numer, BinOp)
        assert numer.op == BinOpKind.SUB

    def test_symbolic_log_density_structure(self):
        obs = Var(ty=IRType.REAL, name="o")
        center = Var(ty=IRType.REAL, name="c")
        scale = Var(ty=IRType.REAL, name="s")
        expr = self.model.symbolic_log_density(obs, center, scale)
        assert isinstance(expr, BinOp)
        assert expr.op == BinOpKind.DIV


# ═══════════════════════════════════════════════════════════════════════════
# TruncatedLaplaceNoise
# ═══════════════════════════════════════════════════════════════════════════


class TestTruncatedLaplaceNoise:
    """Tests for TruncatedLaplaceNoise."""

    def setup_method(self):
        self.model = TruncatedLaplaceNoise(lo=-5.0, hi=5.0)
        self.rng = random.Random(99)

    def test_density_outside_bounds_is_zero(self):
        assert self.model.density(-6.0, 0.0, 1.0) == 0.0
        assert self.model.density(6.0, 0.0, 1.0) == 0.0

    def test_density_inside_bounds_positive(self):
        d = self.model.density(0.0, 0.0, 1.0)
        assert d > 0

    def test_density_greater_than_untruncated(self):
        """Truncated density should be higher inside the bounds (re-normalized)."""
        base = LaplaceNoise()
        trunc_d = self.model.density(0.0, 0.0, 1.0)
        base_d = base.density(0.0, 0.0, 1.0)
        assert trunc_d >= base_d

    def test_log_density_outside_bounds_is_neg_inf(self):
        assert self.model.log_density(-6.0, 0.0, 1.0) == -math.inf

    def test_log_density_inside_matches(self):
        ld = self.model.log_density(0.0, 0.0, 1.0)
        d = self.model.density(0.0, 0.0, 1.0)
        assert ld == pytest.approx(math.log(d))

    def test_log_ratio_outside_bounds_is_zero(self):
        assert self.model.log_ratio(-6.0, 0.0, 1.0, 1.0) == 0.0

    def test_log_ratio_inside_bounds(self):
        lr = self.model.log_ratio(0.0, 0.0, 1.0, 1.0)
        assert isinstance(lr, float)

    def test_cdf_at_lower_bound_is_zero(self):
        assert self.model.cdf(-5.0, 0.0, 1.0) == pytest.approx(0.0)

    def test_cdf_at_upper_bound_is_one(self):
        assert self.model.cdf(5.0, 0.0, 1.0) == pytest.approx(1.0)

    def test_cdf_monotone(self):
        prev = 0.0
        for x in [-4.0, -2.0, 0.0, 2.0, 4.0]:
            c = self.model.cdf(x, 0.0, 1.0)
            assert c >= prev
            prev = c

    def test_sample_within_bounds(self):
        for _ in range(100):
            s = self.model.sample(0.0, 1.0, self.rng)
            assert -5.0 <= s <= 5.0

    def test_symbolic_delegates_to_base(self):
        obs, c_d, c_dp, scale = _sym_vars()
        trunc_expr = self.model.symbolic_log_ratio(obs, c_d, c_dp, scale)
        base_expr = LaplaceNoise().symbolic_log_ratio(obs, c_d, c_dp, scale)
        assert str(trunc_expr) == str(base_expr)


# ═══════════════════════════════════════════════════════════════════════════
# TruncatedGaussianNoise
# ═══════════════════════════════════════════════════════════════════════════


class TestTruncatedGaussianNoise:
    """Tests for TruncatedGaussianNoise."""

    def setup_method(self):
        self.model = TruncatedGaussianNoise(lo=-3.0, hi=3.0)
        self.rng = random.Random(77)

    def test_density_outside_bounds_is_zero(self):
        assert self.model.density(-4.0, 0.0, 1.0) == 0.0
        assert self.model.density(4.0, 0.0, 1.0) == 0.0

    def test_density_inside_bounds_positive(self):
        d = self.model.density(0.0, 0.0, 1.0)
        assert d > 0

    def test_log_density_outside_bounds_is_neg_inf(self):
        assert self.model.log_density(-4.0, 0.0, 1.0) == -math.inf

    def test_log_ratio_outside_bounds_is_zero(self):
        assert self.model.log_ratio(-4.0, 0.0, 1.0, 1.0) == 0.0

    def test_cdf_boundaries(self):
        assert self.model.cdf(-3.0, 0.0, 1.0) == pytest.approx(0.0)
        assert self.model.cdf(3.0, 0.0, 1.0) == pytest.approx(1.0)

    def test_sample_within_bounds(self):
        for _ in range(100):
            s = self.model.sample(0.0, 1.0, self.rng)
            assert -3.0 <= s <= 3.0

    def test_symbolic_delegates_to_base(self):
        obs, c_d, c_dp, scale = _sym_vars()
        trunc_expr = self.model.symbolic_log_ratio(obs, c_d, c_dp, scale)
        base_expr = GaussianNoise().symbolic_log_ratio(obs, c_d, c_dp, scale)
        assert str(trunc_expr) == str(base_expr)


# ═══════════════════════════════════════════════════════════════════════════
# DiscreteGaussianNoise
# ═══════════════════════════════════════════════════════════════════════════


class TestDiscreteGaussianNoise:
    """Tests for DiscreteGaussianNoise."""

    def setup_method(self):
        self.model = DiscreteGaussianNoise()

    def test_density_positive(self):
        d = self.model.density(0.0, 0.0, 1.0)
        assert d > 0

    def test_density_zero_scale_raises(self):
        with pytest.raises(ValueError, match="positive"):
            self.model.density(0.0, 0.0, 0.0)

    def test_log_density_matches_density(self):
        ld = self.model.log_density(0.0, 0.0, 1.0)
        d = self.model.density(0.0, 0.0, 1.0)
        assert ld == pytest.approx(math.log(d))

    def test_log_ratio_same_center_is_zero(self):
        lr = self.model.log_ratio(0.0, 0.0, 0.0, 1.0)
        assert lr == pytest.approx(0.0, abs=1e-10)

    def test_sample_returns_float(self):
        rng = random.Random(42)
        s = self.model.sample(0.0, 1.0, rng)
        assert isinstance(s, float)


# ═══════════════════════════════════════════════════════════════════════════
# DiscreteLaplaceNoise
# ═══════════════════════════════════════════════════════════════════════════


class TestDiscreteLaplaceNoise:
    """Tests for DiscreteLaplaceNoise."""

    def setup_method(self):
        self.model = DiscreteLaplaceNoise()

    def test_density_positive(self):
        d = self.model.density(0.0, 0.0, 1.0)
        assert d > 0

    def test_density_zero_scale_raises(self):
        with pytest.raises(ValueError, match="positive"):
            self.model.density(0.0, 0.0, 0.0)

    def test_log_ratio_same_as_laplace_form(self):
        lr = self.model.log_ratio(3.0, 0.0, 1.0, 2.0)
        expected = (abs(3 - 1) - abs(3 - 0)) / 2.0
        assert lr == pytest.approx(expected)

    def test_sample_returns_float(self):
        rng = random.Random(42)
        s = self.model.sample(0.0, 1.0, rng)
        assert isinstance(s, float)


# ═══════════════════════════════════════════════════════════════════════════
# MixtureNoise
# ═══════════════════════════════════════════════════════════════════════════


class TestMixtureNoise:
    """Tests for MixtureNoise."""

    def setup_method(self):
        self.model = MixtureNoise([
            (0.5, LaplaceNoise()),
            (0.5, GaussianNoise()),
        ])

    def test_num_components(self):
        assert self.model.num_components == 2

    def test_density_positive(self):
        d = self.model.density(0.0, 0.0, 1.0)
        assert d > 0

    def test_density_is_weighted_sum(self):
        x, c, s = 1.0, 0.0, 1.0
        d_lap = LaplaceNoise().density(x, c, s)
        d_gauss = GaussianNoise().density(x, c, s)
        d_mix = self.model.density(x, c, s)
        assert d_mix == pytest.approx(0.5 * d_lap + 0.5 * d_gauss)

    def test_log_density(self):
        d = self.model.density(1.0, 0.0, 1.0)
        ld = self.model.log_density(1.0, 0.0, 1.0)
        assert ld == pytest.approx(math.log(d))

    def test_log_ratio_same_center_is_zero(self):
        lr = self.model.log_ratio(0.0, 0.0, 0.0, 1.0)
        assert lr == pytest.approx(0.0)

    def test_cdf_is_weighted_sum(self):
        x, c, s = 0.0, 0.0, 1.0
        cdf_lap = LaplaceNoise().cdf(x, c, s)
        cdf_gauss = GaussianNoise().cdf(x, c, s)
        cdf_mix = self.model.cdf(x, c, s)
        assert cdf_mix == pytest.approx(0.5 * cdf_lap + 0.5 * cdf_gauss)

    def test_sample_returns_float(self):
        rng = random.Random(42)
        s = self.model.sample(0.0, 1.0, rng)
        assert isinstance(s, float)

    def test_zero_weights_raises(self):
        with pytest.raises(ValueError, match="positive"):
            MixtureNoise([(0.0, LaplaceNoise()), (0.0, GaussianNoise())])

    def test_symbolic_delegates_to_first(self):
        obs, c_d, c_dp, scale = _sym_vars()
        mix_expr = self.model.symbolic_log_ratio(obs, c_d, c_dp, scale)
        base_expr = LaplaceNoise().symbolic_log_ratio(obs, c_d, c_dp, scale)
        assert str(mix_expr) == str(base_expr)


# ═══════════════════════════════════════════════════════════════════════════
# get_noise_model / Registry
# ═══════════════════════════════════════════════════════════════════════════


class TestNoiseModelRegistry:
    """Tests for the noise model factory/registry."""

    @pytest.mark.parametrize("key,expected_cls", [
        ("laplace", LaplaceNoise),
        ("gaussian", GaussianNoise),
        ("exponential", ExponentialMechNoise),
        ("truncated_laplace", TruncatedLaplaceNoise),
        ("truncated_gaussian", TruncatedGaussianNoise),
        ("discrete_gaussian", DiscreteGaussianNoise),
        ("discrete_laplace", DiscreteLaplaceNoise),
    ])
    def test_get_by_string(self, key: str, expected_cls: type):
        model = get_noise_model(key)
        assert isinstance(model, expected_cls)

    def test_get_by_noise_kind_enum(self):
        model = get_noise_model(NoiseKind.LAPLACE)
        assert isinstance(model, LaplaceNoise)

    def test_get_by_noise_kind_gaussian(self):
        model = get_noise_model(NoiseKind.GAUSSIAN)
        assert isinstance(model, GaussianNoise)

    def test_get_by_noise_kind_exponential(self):
        model = get_noise_model(NoiseKind.EXPONENTIAL)
        assert isinstance(model, ExponentialMechNoise)

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            get_noise_model("nonexistent_noise")

    def test_registry_contains_expected_keys(self):
        expected_keys = {
            "laplace", "gaussian", "exponential",
            "truncated_laplace", "truncated_gaussian",
            "discrete_gaussian", "discrete_laplace",
        }
        assert expected_keys.issubset(set(_NOISE_MODEL_REGISTRY.keys()))


# ═══════════════════════════════════════════════════════════════════════════
# Parametrized tests for different scales / sensitivities
# ═══════════════════════════════════════════════════════════════════════════


class TestParametrizedScales:
    """Cross-model parametrized tests for various scales."""

    @pytest.mark.parametrize("scale", [0.1, 0.5, 1.0, 2.0, 10.0])
    def test_laplace_density_integrates_approx_one(self, scale: float):
        """Density should integrate to approximately 1 via Riemann sum."""
        model = LaplaceNoise()
        dx = 0.01
        xs = [i * dx for i in range(-5000, 5001)]
        total = sum(model.density(x, 0.0, scale) * dx for x in xs)
        assert total == pytest.approx(1.0, abs=0.01)

    @pytest.mark.parametrize("scale", [0.5, 1.0, 2.0, 5.0])
    def test_gaussian_density_integrates_approx_one(self, scale: float):
        model = GaussianNoise()
        dx = 0.01
        xs = [i * dx for i in range(-5000, 5001)]
        total = sum(model.density(x, 0.0, scale) * dx for x in xs)
        assert total == pytest.approx(1.0, abs=0.01)

    @pytest.mark.parametrize("sensitivity,scale,expected_eps", [
        (1.0, 1.0, 1.0),
        (1.0, 2.0, 0.5),
        (2.0, 1.0, 2.0),
        (0.5, 0.5, 1.0),
    ])
    def test_laplace_max_privacy_loss_parametrized(
        self, sensitivity: float, scale: float, expected_eps: float
    ):
        model = LaplaceNoise()
        assert model.max_privacy_loss(sensitivity, scale) == pytest.approx(expected_eps)

    @pytest.mark.parametrize("sensitivity,scale", [
        (1.0, 1.0),
        (1.0, 2.0),
        (2.0, 1.0),
    ])
    def test_gaussian_zcdp_rho_parametrized(self, sensitivity: float, scale: float):
        model = GaussianNoise()
        expected = sensitivity ** 2 / (2.0 * scale ** 2)
        assert model.zcdp_rho(sensitivity, scale) == pytest.approx(expected)


# ═══════════════════════════════════════════════════════════════════════════
# Edge cases: numerical stability
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Edge cases and numerical stability checks."""

    def test_laplace_density_very_large_x(self):
        """Should not raise for very large x."""
        model = LaplaceNoise()
        d = model.density(1000.0, 0.0, 1.0)
        assert d >= 0 and math.isfinite(d)

    def test_gaussian_density_very_large_x(self):
        model = GaussianNoise()
        d = model.density(100.0, 0.0, 1.0)
        assert d >= 0 and math.isfinite(d)

    def test_laplace_log_density_very_large_x(self):
        model = LaplaceNoise()
        ld = model.log_density(1000.0, 0.0, 1.0)
        assert math.isfinite(ld) and ld < 0

    def test_gaussian_log_density_very_large_x(self):
        model = GaussianNoise()
        ld = model.log_density(100.0, 0.0, 1.0)
        assert math.isfinite(ld) and ld < 0

    def test_laplace_log_ratio_large_displacement(self):
        model = LaplaceNoise()
        lr = model.log_ratio(0.0, 0.0, 100.0, 1.0)
        assert math.isfinite(lr)

    def test_gaussian_log_ratio_large_displacement(self):
        model = GaussianNoise()
        lr = model.log_ratio(0.0, 0.0, 100.0, 1.0)
        assert math.isfinite(lr)

    def test_laplace_very_small_scale(self):
        model = LaplaceNoise()
        d = model.density(0.0, 0.0, 1e-10)
        assert d > 0

    def test_gaussian_very_small_scale(self):
        model = GaussianNoise()
        d = model.density(0.0, 0.0, 1e-10)
        assert d > 0

    def test_laplace_negative_inputs(self):
        model = LaplaceNoise()
        d = model.density(-5.0, -3.0, 1.0)
        assert d > 0

    def test_gaussian_negative_inputs(self):
        model = GaussianNoise()
        d = model.density(-5.0, -3.0, 1.0)
        assert d > 0

    def test_truncated_with_infinite_bounds_same_as_base(self):
        """Truncation with [-inf, inf] should behave like the base."""
        trunc = TruncatedLaplaceNoise(lo=-math.inf, hi=math.inf)
        base = LaplaceNoise()
        d_t = trunc.density(1.0, 0.0, 1.0)
        d_b = base.density(1.0, 0.0, 1.0)
        assert d_t == pytest.approx(d_b, rel=1e-6)
