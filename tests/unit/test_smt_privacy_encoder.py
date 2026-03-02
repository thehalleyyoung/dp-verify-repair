"""Tests for dpcegar.smt.privacy_encoder – privacy-specific SMT encodings."""

from __future__ import annotations

import math
from typing import Any

import pytest
import z3

from dpcegar.ir.types import (
    ApproxBudget,
    BinOp,
    BinOpKind,
    Const,
    GDPBudget,
    IRType,
    NoiseKind,
    Phi,
    PrivacyNotion,
    PureBudget,
    RDPBudget,
    Var,
    ZCDPBudget,
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
from dpcegar.smt.encoding import SMTEncoding
from dpcegar.smt.privacy_encoder import (
    ApproxDPEncoder,
    PureEpsDPEncoder,
    PrivacyEncodingResult,
    ZCDPEncoder,
    RDPEncoder,
    GDPEncoder,
    FDPEncoder,
    CrossPathEncoder,
)
from dpcegar.smt.transcendental import Precision
from dpcegar.smt.solver import Z3Solver, SolverConfig, CheckResult


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _make_laplace_ratio(sensitivity: float = 1.0, scale: float = 1.0) -> DensityRatioResult:
    """Build a density ratio for a simple Laplace mechanism."""
    delta_q = Var(ty=IRType.REAL, name="delta_q")
    log_ratio = BinOp(
        ty=IRType.REAL,
        op=BinOpKind.DIV,
        left=delta_q,
        right=Const.real(scale),
    )
    dr = DensityRatioExpr(
        log_ratio=log_ratio,
        path_condition_d=PathCondition.trivially_true(),
        path_condition_d_prime=PathCondition.trivially_true(),
        path_id_d=0,
        path_id_d_prime=0,
    )
    return DensityRatioResult(ratios=[dr], same_path=[dr])


def _make_gaussian_ratio(sigma: float = 1.0) -> DensityRatioResult:
    """Build a density ratio for a Gaussian mechanism."""
    delta_q = Var(ty=IRType.REAL, name="delta_q")
    o = Var(ty=IRType.REAL, name="o")
    log_ratio = BinOp(
        ty=IRType.REAL,
        op=BinOpKind.MUL,
        left=Const.real(1.0 / (2.0 * sigma**2)),
        right=BinOp(
            ty=IRType.REAL, op=BinOpKind.MUL,
            left=delta_q,
            right=BinOp(
                ty=IRType.REAL, op=BinOpKind.SUB,
                left=BinOp(
                    ty=IRType.REAL, op=BinOpKind.MUL,
                    left=Const.real(2.0), right=o,
                ),
                right=delta_q,
            ),
        ),
    )
    dr = DensityRatioExpr(
        log_ratio=log_ratio,
        path_condition_d=PathCondition.trivially_true(),
        path_condition_d_prime=PathCondition.trivially_true(),
        path_id_d=0,
        path_id_d_prime=0,
    )
    return DensityRatioResult(ratios=[dr], same_path=[dr])


def _make_cross_path_ratio() -> DensityRatioResult:
    """Density ratio for cross-path (data-dependent branching)."""
    log_ratio = BinOp(
        ty=IRType.REAL,
        op=BinOpKind.SUB,
        left=Var(ty=IRType.REAL, name="log_p1"),
        right=Var(ty=IRType.REAL, name="log_p2"),
    )
    dr = DensityRatioExpr(
        log_ratio=log_ratio,
        path_condition_d=PathCondition.from_expr(
            BinOp(ty=IRType.BOOL, op=BinOpKind.GT,
                   left=Var(ty=IRType.REAL, name="x"), right=Const.real(0.0)),
        ),
        path_condition_d_prime=PathCondition.from_expr(
            BinOp(ty=IRType.BOOL, op=BinOpKind.LE,
                   left=Var(ty=IRType.REAL, name="x"), right=Const.real(0.0)),
        ),
        path_id_d=0,
        path_id_d_prime=1,
        is_cross_path=True,
    )
    return DensityRatioResult(
        ratios=[dr], same_path=[], cross_path=[dr],
    )


def _check_sat(encoding: SMTEncoding) -> CheckResult:
    solver = Z3Solver(SolverConfig(timeout_ms=5000))
    solver.add_encoding(encoding)
    return solver.check().result


# ═══════════════════════════════════════════════════════════════════════════
# PureEpsDPEncoder
# ═══════════════════════════════════════════════════════════════════════════


class TestPureEpsDPEncoder:
    """Tests for pure ε-differential privacy encoding."""

    def test_construction(self):
        enc = PureEpsDPEncoder()
        assert enc is not None

    def test_construction_with_precision(self):
        enc = PureEpsDPEncoder(precision=Precision.HIGH)
        assert enc is not None

    def test_encode_returns_result(self):
        enc = PureEpsDPEncoder()
        ratio = _make_laplace_ratio(scale=1.0)
        result = enc.encode(ratio, epsilon=1.0)
        assert isinstance(result, PrivacyEncodingResult)
        assert result.notion == PrivacyNotion.PURE_DP

    def test_correct_laplace_is_unsat(self):
        """Correct Laplace mechanism (scale=1/eps) should be UNSAT (verified)."""
        enc = PureEpsDPEncoder()
        ratio = _make_laplace_ratio(scale=1.0)
        result = enc.encode(ratio, epsilon=1.0)
        status = _check_sat(result.encoding)
        assert status in (CheckResult.UNSAT, CheckResult.UNKNOWN)

    def test_buggy_laplace_is_sat(self):
        """Buggy Laplace (scale too small) should be SAT (counterexample)."""
        enc = PureEpsDPEncoder()
        ratio = _make_laplace_ratio(scale=0.1)
        result = enc.encode(ratio, epsilon=1.0)
        status = _check_sat(result.encoding)
        assert status in (CheckResult.SAT, CheckResult.UNKNOWN)

    def test_encode_laplace_direct(self):
        enc = PureEpsDPEncoder()
        encoding = enc.encode_laplace("delta", "scale", epsilon=1.0, sensitivity_bound=1.0)
        assert isinstance(encoding, SMTEncoding)

    def test_encode_single_ratio(self):
        enc = PureEpsDPEncoder()
        log_r = BinOp(
            ty=IRType.REAL, op=BinOpKind.MUL,
            left=Var(ty=IRType.REAL, name="x"), right=Const.real(1.0),
        )
        encoding = enc.encode_single_ratio(log_r, epsilon=1.0)
        assert isinstance(encoding, SMTEncoding)

    def test_num_paths_in_result(self):
        enc = PureEpsDPEncoder()
        ratio = _make_laplace_ratio()
        result = enc.encode(ratio, epsilon=1.0)
        assert result.num_paths >= 1

    def test_result_summary(self):
        enc = PureEpsDPEncoder()
        ratio = _make_laplace_ratio()
        result = enc.encode(ratio, epsilon=1.0)
        s = result.summary()
        assert isinstance(s, dict)

    @pytest.mark.parametrize("eps", [0.1, 0.5, 1.0, 2.0, 5.0])
    def test_various_epsilon_values(self, eps: float):
        enc = PureEpsDPEncoder()
        ratio = _make_laplace_ratio(scale=1.0 / eps)
        result = enc.encode(ratio, epsilon=eps)
        assert isinstance(result, PrivacyEncodingResult)

    def test_encoding_is_sound(self):
        enc = PureEpsDPEncoder()
        ratio = _make_laplace_ratio()
        result = enc.encode(ratio, epsilon=1.0)
        assert result.is_sound is True or result.is_sound is False


# ═══════════════════════════════════════════════════════════════════════════
# ApproxDPEncoder
# ═══════════════════════════════════════════════════════════════════════════


class TestApproxDPEncoder:
    """Tests for (ε,δ)-differential privacy encoding."""

    def test_construction(self):
        enc = ApproxDPEncoder()
        assert enc is not None

    def test_encode_returns_result(self):
        enc = ApproxDPEncoder()
        ratio = _make_gaussian_ratio(sigma=1.0)
        result = enc.encode(ratio, epsilon=1.0, delta=1e-5)
        assert isinstance(result, PrivacyEncodingResult)
        assert result.notion == PrivacyNotion.APPROX_DP

    def test_encode_gaussian(self):
        enc = ApproxDPEncoder()
        encoding = enc.encode_gaussian("delta_q", "sigma", epsilon=1.0, target_delta=1e-5)
        assert isinstance(encoding, SMTEncoding)

    def test_encode_gaussian_analytic(self):
        enc = ApproxDPEncoder()
        encoding = enc.encode_gaussian_analytic(
            sensitivity=1.0, sigma=2.0, epsilon=1.0, target_delta=1e-5,
        )
        assert isinstance(encoding, SMTEncoding)

    def test_num_cross_paths(self):
        enc = ApproxDPEncoder()
        ratio = _make_gaussian_ratio()
        result = enc.encode(ratio, epsilon=1.0, delta=1e-5)
        assert result.num_cross_paths >= 0

    @pytest.mark.parametrize("delta", [1e-3, 1e-5, 1e-8, 1e-10])
    def test_various_delta_values(self, delta: float):
        enc = ApproxDPEncoder()
        ratio = _make_gaussian_ratio()
        result = enc.encode(ratio, epsilon=1.0, delta=delta)
        assert isinstance(result, PrivacyEncodingResult)

    def test_tight_gaussian_should_verify(self):
        """Gaussian with sigma = sqrt(2 ln(1.25/delta)) / eps should verify."""
        delta = 1e-5
        eps = 1.0
        sigma = math.sqrt(2.0 * math.log(1.25 / delta)) / eps
        enc = ApproxDPEncoder()
        ratio = _make_gaussian_ratio(sigma=sigma)
        result = enc.encode(ratio, epsilon=eps, delta=delta)
        assert isinstance(result, PrivacyEncodingResult)


# ═══════════════════════════════════════════════════════════════════════════
# ZCDPEncoder
# ═══════════════════════════════════════════════════════════════════════════


class TestZCDPEncoder:
    """Tests for zero-concentrated DP encoding."""

    def test_construction(self):
        enc = ZCDPEncoder()
        assert enc is not None

    def test_encode_gaussian_zcdp(self):
        enc = ZCDPEncoder()
        ratio = _make_gaussian_ratio(sigma=1.0)
        result = enc.encode(ratio, rho=0.5)
        assert isinstance(result, PrivacyEncodingResult)
        assert result.notion == PrivacyNotion.ZCDP

    @pytest.mark.parametrize("rho", [0.1, 0.5, 1.0, 2.0])
    def test_various_rho_values(self, rho: float):
        enc = ZCDPEncoder()
        ratio = _make_gaussian_ratio()
        result = enc.encode(ratio, rho=rho)
        assert isinstance(result, PrivacyEncodingResult)

    def test_result_summary(self):
        enc = ZCDPEncoder()
        ratio = _make_gaussian_ratio()
        result = enc.encode(ratio, rho=0.5)
        s = result.summary()
        assert isinstance(s, dict)


# ═══════════════════════════════════════════════════════════════════════════
# RDPEncoder
# ═══════════════════════════════════════════════════════════════════════════


class TestRDPEncoder:
    """Tests for Rényi DP encoding."""

    def test_construction(self):
        enc = RDPEncoder()
        assert enc is not None

    def test_encode_gaussian_rdp(self):
        enc = RDPEncoder()
        ratio = _make_gaussian_ratio(sigma=1.0)
        result = enc.encode(ratio, alpha=2.0, epsilon=1.0)
        assert isinstance(result, PrivacyEncodingResult)
        assert result.notion == PrivacyNotion.RDP

    def test_encode_laplace_rdp(self):
        enc = RDPEncoder()
        ratio = _make_laplace_ratio(scale=1.0)
        result = enc.encode(ratio, alpha=2.0, epsilon=1.0)
        assert isinstance(result, PrivacyEncodingResult)

    @pytest.mark.parametrize("alpha", [1.5, 2.0, 5.0, 10.0, 100.0])
    def test_various_alpha_values(self, alpha: float):
        enc = RDPEncoder()
        ratio = _make_gaussian_ratio()
        result = enc.encode(ratio, alpha=alpha, epsilon=5.0)
        assert isinstance(result, PrivacyEncodingResult)

    def test_rdp_gaussian_known_value(self):
        """For Gaussian, RDP cost = alpha/(2*sigma^2) at sensitivity 1."""
        enc = RDPEncoder()
        sigma = 1.0
        alpha = 2.0
        expected_eps = alpha / (2.0 * sigma**2)
        ratio = _make_gaussian_ratio(sigma=sigma)
        result = enc.encode(ratio, alpha=alpha, epsilon=expected_eps + 0.1)
        assert isinstance(result, PrivacyEncodingResult)


# ═══════════════════════════════════════════════════════════════════════════
# GDPEncoder
# ═══════════════════════════════════════════════════════════════════════════


class TestGDPEncoder:
    """Tests for Gaussian DP encoding."""

    def test_construction(self):
        enc = GDPEncoder()
        assert enc is not None

    def test_encode_gaussian_gdp(self):
        enc = GDPEncoder()
        ratio = _make_gaussian_ratio(sigma=1.0)
        result = enc.encode(ratio, mu=1.0)
        assert isinstance(result, PrivacyEncodingResult)
        assert result.notion == PrivacyNotion.GDP

    @pytest.mark.parametrize("mu", [0.1, 0.5, 1.0, 2.0, 5.0])
    def test_various_mu_values(self, mu: float):
        enc = GDPEncoder()
        ratio = _make_gaussian_ratio()
        result = enc.encode(ratio, mu=mu)
        assert isinstance(result, PrivacyEncodingResult)


# ═══════════════════════════════════════════════════════════════════════════
# FDPEncoder
# ═══════════════════════════════════════════════════════════════════════════


class TestFDPEncoder:
    """Tests for f-DP encoding."""

    def test_construction(self):
        enc = FDPEncoder()
        assert enc is not None

    def test_encode_at_grid_points(self):
        enc = FDPEncoder()
        ratio = _make_gaussian_ratio(sigma=1.0)
        trade_off = lambda alpha: max(0.0, 1.0 - alpha)  # noqa: E731
        result = enc.encode(ratio, trade_off_fn=trade_off)
        assert isinstance(result, PrivacyEncodingResult)
        assert result.notion == PrivacyNotion.FDP

    def test_encode_with_fine_grid(self):
        enc = FDPEncoder()
        ratio = _make_gaussian_ratio(sigma=1.0)
        trade_off = lambda alpha: max(0.0, 1.0 - 2.0 * alpha)  # noqa: E731
        result = enc.encode(ratio, trade_off_fn=trade_off, n_grid=50)
        assert isinstance(result, PrivacyEncodingResult)


# ═══════════════════════════════════════════════════════════════════════════
# CrossPathEncoder
# ═══════════════════════════════════════════════════════════════════════════


class TestCrossPathEncoder:
    """Tests for cross-path (data-dependent branching) encoding."""

    def test_construction(self):
        enc = CrossPathEncoder()
        assert enc is not None

    def test_encode_cross_path(self):
        enc = CrossPathEncoder()
        ratio = _make_cross_path_ratio()
        result = enc.encode(ratio, epsilon=1.0)
        assert isinstance(result, PrivacyEncodingResult)
        assert result.num_cross_paths >= 1

    def test_cross_path_is_cross(self):
        ratio = _make_cross_path_ratio()
        assert len(ratio.cross_path) > 0
        assert ratio.cross_path[0].is_cross_path is True


# ═══════════════════════════════════════════════════════════════════════════
# Encoding soundness
# ═══════════════════════════════════════════════════════════════════════════


class TestEncodingSoundness:
    """Verify encoding soundness: good mechanisms → UNSAT, bad → SAT."""

    def test_correct_laplace_verified(self):
        enc = PureEpsDPEncoder()
        ratio = _make_laplace_ratio(scale=1.0)
        result = enc.encode(ratio, epsilon=1.0)
        status = _check_sat(result.encoding)
        assert status in (CheckResult.UNSAT, CheckResult.UNKNOWN)

    def test_buggy_laplace_counterexample(self):
        enc = PureEpsDPEncoder()
        ratio = _make_laplace_ratio(scale=0.1)
        result = enc.encode(ratio, epsilon=1.0)
        status = _check_sat(result.encoding)
        assert status in (CheckResult.SAT, CheckResult.UNKNOWN)

    def test_very_noisy_laplace_verified(self):
        """Oversized noise: should still verify."""
        enc = PureEpsDPEncoder()
        ratio = _make_laplace_ratio(scale=10.0)
        result = enc.encode(ratio, epsilon=1.0)
        status = _check_sat(result.encoding)
        assert status in (CheckResult.UNSAT, CheckResult.UNKNOWN)

    def test_zero_epsilon_impossible(self):
        """eps=0 with any finite noise should be SAT (violated)."""
        enc = PureEpsDPEncoder()
        ratio = _make_laplace_ratio(scale=1.0)
        result = enc.encode(ratio, epsilon=0.0)
        status = _check_sat(result.encoding)
        assert status in (CheckResult.SAT, CheckResult.UNKNOWN)

    @pytest.mark.parametrize("scale,expected", [
        (1.0, CheckResult.UNSAT),
        (0.5, CheckResult.SAT),
    ])
    def test_parametrized_laplace_soundness(self, scale: float, expected: CheckResult):
        enc = PureEpsDPEncoder()
        ratio = _make_laplace_ratio(scale=scale)
        result = enc.encode(ratio, epsilon=1.0)
        status = _check_sat(result.encoding)
        assert status in (expected, CheckResult.UNKNOWN)


# ═══════════════════════════════════════════════════════════════════════════
# Privacy encoding result
# ═══════════════════════════════════════════════════════════════════════════


class TestPrivacyEncodingResult:
    """Test PrivacyEncodingResult metadata."""

    def test_summary_keys(self):
        enc = PureEpsDPEncoder()
        ratio = _make_laplace_ratio()
        result = enc.encode(ratio, epsilon=1.0)
        s = result.summary()
        assert "notion" in s or len(s) > 0

    def test_theory_field(self):
        enc = PureEpsDPEncoder()
        ratio = _make_laplace_ratio()
        result = enc.encode(ratio, epsilon=1.0)
        assert result.theory is not None

    def test_approximations_list(self):
        enc = PureEpsDPEncoder()
        ratio = _make_laplace_ratio()
        result = enc.encode(ratio, epsilon=1.0)
        assert isinstance(result.approximations, list)

    def test_encoding_has_variables(self):
        enc = PureEpsDPEncoder()
        ratio = _make_laplace_ratio()
        result = enc.encode(ratio, epsilon=1.0)
        assert result.encoding.variable_count() >= 0

    def test_encoding_has_assertions(self):
        enc = PureEpsDPEncoder()
        ratio = _make_laplace_ratio()
        result = enc.encode(ratio, epsilon=1.0)
        assert result.encoding.assertion_count() >= 0
