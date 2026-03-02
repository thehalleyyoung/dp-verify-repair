"""Tests for f-DP tensor product composition (FDP soundness fix)."""

from __future__ import annotations

import math

import pytest

from dpcegar.ir.types import FDPBudget, PrivacyNotion
from dpcegar.certificates.certificate import (
    Certificate,
    CertificateChain,
    CertificateType,
)

from scipy.stats import norm  # type: ignore[import-untyped]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def gaussian_tradeoff(mu: float):
    """Return the Gaussian trade-off function T(α) = Φ(Φ⁻¹(1-α) - μ)."""
    def T(alpha: float) -> float:
        alpha = max(0.0, min(1.0, alpha))
        if alpha >= 1.0:
            return 0.0
        if alpha <= 0.0:
            return 1.0
        return float(norm.cdf(norm.ppf(1.0 - alpha) - mu))
    return T


def _make_fdp_cert(trade_off_fn, *, valid: bool = True) -> Certificate:
    return Certificate(
        cert_type=CertificateType.VERIFICATION,
        mechanism_id="test-mech",
        mechanism_name="test",
        privacy_notion=PrivacyNotion.FDP,
        privacy_guarantee=FDPBudget(trade_off_fn=trade_off_fn),
        proof_data={"placeholder": True},
        valid=valid,
    )


# ═══════════════════════════════════════════════════════════════════════════
# FDPBudget.compose tests
# ═══════════════════════════════════════════════════════════════════════════


class TestFDPCompose:
    def test_fdp_compose_type_preserved(self):
        f = FDPBudget(trade_off_fn=gaussian_tradeoff(1.0))
        g = FDPBudget(trade_off_fn=gaussian_tradeoff(1.0))
        result = f.compose(g)
        assert isinstance(result, FDPBudget)

    def test_fdp_compose_vs_gaussian_closed_form(self):
        # Two Gaussian(μ=1) composed → Gaussian(μ=√2)
        f = FDPBudget(trade_off_fn=gaussian_tradeoff(1.0))
        g = FDPBudget(trade_off_fn=gaussian_tradeoff(1.0))
        composed = f.compose(g)
        assert isinstance(composed, FDPBudget)

        expected = gaussian_tradeoff(math.sqrt(2.0))
        for alpha in [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95]:
            assert abs(composed.trade_off_fn(alpha) - expected(alpha)) < 0.01, (
                f"Mismatch at α={alpha}: got {composed.trade_off_fn(alpha)}, "
                f"expected {expected(alpha)}"
            )

    def test_fdp_compose_identity(self):
        # Identity trade-off: T(α)=1-α (no privacy loss)
        identity = FDPBudget(trade_off_fn=lambda a: 1.0 - a)
        original = FDPBudget(trade_off_fn=gaussian_tradeoff(1.0))
        composed = original.compose(identity)
        assert isinstance(composed, FDPBudget)

        for alpha in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
            assert abs(composed.trade_off_fn(alpha) - original.trade_off_fn(alpha)) < 0.01, (
                f"Mismatch at α={alpha}"
            )

    def test_fdp_compose_trivial(self):
        # Trivial trade-off: T(α)=0 (total privacy loss)
        trivial = FDPBudget(trade_off_fn=lambda a: 0.0)
        original = FDPBudget(trade_off_fn=gaussian_tradeoff(1.0))
        composed = original.compose(trivial)
        assert isinstance(composed, FDPBudget)

        for alpha in [0.0, 0.1, 0.5, 0.9, 1.0]:
            assert composed.trade_off_fn(alpha) <= 1e-9, (
                f"Expected ~0 at α={alpha}, got {composed.trade_off_fn(alpha)}"
            )

    def test_fdp_compose_boundary_values(self):
        f = FDPBudget(trade_off_fn=gaussian_tradeoff(1.0))
        g = FDPBudget(trade_off_fn=gaussian_tradeoff(1.0))
        composed = f.compose(g)
        assert isinstance(composed, FDPBudget)
        # At α=0, trade-off should be close to 1
        assert composed.trade_off_fn(0.0) >= 0.99
        # At α=1, trade-off should be close to 0
        assert composed.trade_off_fn(1.0) <= 0.01

    def test_fdp_compose_monotonicity(self):
        f = FDPBudget(trade_off_fn=gaussian_tradeoff(1.0))
        g = FDPBudget(trade_off_fn=gaussian_tradeoff(0.5))
        composed = f.compose(g)
        assert isinstance(composed, FDPBudget)

        alphas = [i / 100 for i in range(101)]
        values = [composed.trade_off_fn(a) for a in alphas]
        for i in range(len(values) - 1):
            assert values[i] >= values[i + 1] - 1e-9, (
                f"Not decreasing at α={alphas[i]}: "
                f"{values[i]} < {values[i+1]}"
            )

    def test_fdp_sequential_three_way(self):
        # Three Gaussian(μ=0.5) → Gaussian(μ=√0.75)
        mu = 0.5
        b1 = FDPBudget(trade_off_fn=gaussian_tradeoff(mu))
        b2 = FDPBudget(trade_off_fn=gaussian_tradeoff(mu))
        b3 = FDPBudget(trade_off_fn=gaussian_tradeoff(mu))
        composed = b1.compose(b2).compose(b3)
        assert isinstance(composed, FDPBudget)

        expected = gaussian_tradeoff(math.sqrt(3 * mu ** 2))
        for alpha in [0.05, 0.1, 0.3, 0.5, 0.7, 0.9]:
            assert abs(composed.trade_off_fn(alpha) - expected(alpha)) < 0.02, (
                f"Mismatch at α={alpha}: got {composed.trade_off_fn(alpha)}, "
                f"expected {expected(alpha)}"
            )

    def test_fdp_parallel_max(self):
        # Parallel composition: pointwise max
        f_fn = gaussian_tradeoff(1.0)
        g_fn = gaussian_tradeoff(2.0)
        f = FDPBudget(trade_off_fn=f_fn)
        g = FDPBudget(trade_off_fn=g_fn)

        fns = [f.trade_off_fn, g.trade_off_fn]

        def parallel(alpha: float) -> float:
            return max(fn(alpha) for fn in fns)

        for alpha in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
            expected = parallel(alpha)
            # max of two Gaussians at each alpha → the one with less privacy loss
            assert expected == max(f_fn(alpha), g_fn(alpha))


# ═══════════════════════════════════════════════════════════════════════════
# CertificateChain composed_budget / validate_chain tests
# ═══════════════════════════════════════════════════════════════════════════


class TestFDPCertificateChain:
    def test_composed_budget_fdp_sequential(self):
        chain = CertificateChain(composition_type="sequential")
        chain.add(_make_fdp_cert(gaussian_tradeoff(1.0)))
        chain.add(_make_fdp_cert(gaussian_tradeoff(1.0)))
        result = chain.composed_budget()
        assert isinstance(result, FDPBudget)

        expected = gaussian_tradeoff(math.sqrt(2.0))
        for alpha in [0.05, 0.1, 0.5, 0.9]:
            assert abs(result.trade_off_fn(alpha) - expected(alpha)) < 0.01

    def test_composed_budget_fdp_parallel(self):
        f_fn = gaussian_tradeoff(1.0)
        g_fn = gaussian_tradeoff(2.0)
        chain = CertificateChain(composition_type="parallel")
        chain.add(_make_fdp_cert(f_fn))
        chain.add(_make_fdp_cert(g_fn))
        result = chain.composed_budget()
        assert isinstance(result, FDPBudget)

        for alpha in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
            expected = max(f_fn(alpha), g_fn(alpha))
            assert abs(result.trade_off_fn(alpha) - expected) < 1e-9

    def test_validate_chain_fdp_valid(self):
        # Composed guarantee matches actual composition
        f_fn = gaussian_tradeoff(1.0)
        chain = CertificateChain(composition_type="sequential")
        chain.add(_make_fdp_cert(f_fn))
        chain.add(_make_fdp_cert(f_fn))

        # Set composed_guarantee to exactly the composed result
        composed = chain.composed_budget()
        assert isinstance(composed, FDPBudget)
        chain.composed_guarantee = composed

        assert chain.validate_chain() is True

    def test_validate_chain_fdp_invalid(self):
        # Composed guarantee is TIGHTER than actual → should fail
        f_fn = gaussian_tradeoff(1.0)
        chain = CertificateChain(composition_type="sequential")
        chain.add(_make_fdp_cert(f_fn))
        chain.add(_make_fdp_cert(f_fn))

        # Set a guarantee that is too strong (lower μ = better privacy)
        too_strong = FDPBudget(trade_off_fn=gaussian_tradeoff(0.1))
        chain.composed_guarantee = too_strong

        assert chain.validate_chain() is False
