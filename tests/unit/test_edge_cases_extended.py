"""Unit tests for numerical edge cases and boundary scenarios.

Extends the existing test_edge_cases.py with additional coverage for
density ratios, path conditions, composition, budget limits, and
numerical precision.
"""

from __future__ import annotations

import math

import pytest

from dpcegar.ir.types import (
    ApproxBudget,
    BinOp,
    BinOpKind,
    Const,
    GDPBudget,
    IRType,
    NoiseKind,
    PrivacyNotion,
    PureBudget,
    RDPBudget,
    Var,
    ZCDPBudget,
)
from dpcegar.ir.nodes import (
    MechIR,
    NoiseDrawNode,
    ParamDecl,
    QueryNode,
    ReturnNode,
    SequenceNode,
)
from dpcegar.density.noise_models import GaussianNoise, LaplaceNoise
from dpcegar.density.composition import SequentialComposition
from dpcegar.density.ratio_builder import DensityRatioBuilder, DensityRatioResult
from dpcegar.paths.symbolic_path import (
    NoiseDrawInfo,
    PathCondition,
    PathSet,
    SymbolicPath,
)
from dpcegar.cegar.engine import CEGARConfig, CEGAREngine, CEGARResult, CEGARVerdict
from dpcegar.certificates.certificate import (
    CertificateChain,
    CertificateType,
    VerificationCertificate,
)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Density ratio with zero sensitivity
# ═══════════════════════════════════════════════════════════════════════════


class TestZeroSensitivity:
    """Sensitivity=0 means adjacent databases are identical → ratio = 0."""

    def test_laplace_zero_sensitivity_privacy_loss(self):
        lap = LaplaceNoise()
        loss = lap.max_privacy_loss(sensitivity=0.0, scale=1.0)
        assert loss == 0.0

    def test_gaussian_zero_sensitivity_zcdp(self):
        g = GaussianNoise()
        rho = g.zcdp_rho(sensitivity=0.0, scale=1.0)
        assert rho == 0.0

    def test_gaussian_zero_sensitivity_gdp(self):
        g = GaussianNoise()
        mu = g.gdp_mu(sensitivity=0.0, scale=1.0)
        assert mu == 0.0

    def test_laplace_zero_sensitivity_log_ratio(self):
        lap = LaplaceNoise()
        ratio = lap.log_ratio(0.0, 0.0, 0.0, 1.0)
        assert ratio == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# 2. Path condition with contradictory constraints
# ═══════════════════════════════════════════════════════════════════════════


class TestContradictoryPathConditions:
    """Contradictory path conditions should be representable."""

    def test_contradictory_gt_lt(self):
        """x > 0 AND x < 0 is contradictory but should not crash."""
        cond_gt = BinOp(ty=IRType.BOOL, op=BinOpKind.GT,
                        left=Var(ty=IRType.REAL, name="x"),
                        right=Const.real(0.0))
        cond_lt = BinOp(ty=IRType.BOOL, op=BinOpKind.LT,
                        left=Var(ty=IRType.REAL, name="x"),
                        right=Const.real(0.0))
        pc = PathCondition.from_expr(cond_gt)
        pc.conjuncts.append(cond_lt)
        assert len(pc.conjuncts) == 2

    def test_trivially_true_has_no_conjuncts(self):
        pc = PathCondition.trivially_true()
        assert len(pc.conjuncts) == 0

    def test_from_expr_has_one_conjunct(self):
        cond = BinOp(ty=IRType.BOOL, op=BinOpKind.GT,
                     left=Var(ty=IRType.REAL, name="x"),
                     right=Const.real(0.0))
        pc = PathCondition.from_expr(cond)
        assert len(pc.conjuncts) == 1


# ═══════════════════════════════════════════════════════════════════════════
# 3. Very large epsilon (should always verify)
# ═══════════════════════════════════════════════════════════════════════════


class TestVeryLargeEpsilon:
    """A very large privacy budget should always be satisfiable."""

    def test_large_epsilon_pure_dp(self):
        ps = PathSet()
        ps.add(SymbolicPath(
            path_condition=PathCondition.trivially_true(),
            noise_draws=[NoiseDrawInfo(
                variable="eta", kind=NoiseKind.LAPLACE,
                center_expr=Var(ty=IRType.REAL, name="q"),
                scale_expr=Const.real(1.0), site_id=100,
            )],
            output_expr=Var(ty=IRType.REAL, name="eta"),
        ))
        builder = DensityRatioBuilder()
        dr = builder.build(ps)
        engine = CEGAREngine(config=CEGARConfig(
            max_refinements=10, timeout_seconds=15.0, solver_timeout_seconds=5.0,
        ))
        result = engine.verify(ps, PureBudget(epsilon=1000.0), dr)
        assert result.verdict in (CEGARVerdict.VERIFIED, CEGARVerdict.UNKNOWN)

    def test_inf_epsilon(self):
        b = PureBudget(epsilon=float("inf"))
        assert b.epsilon == float("inf")


# ═══════════════════════════════════════════════════════════════════════════
# 4. Very small epsilon (should often fail)
# ═══════════════════════════════════════════════════════════════════════════


class TestVerySmallEpsilon:
    """A very small budget with fixed noise is likely to fail."""

    def test_tiny_epsilon_laplace(self):
        ps = PathSet()
        ps.add(SymbolicPath(
            path_condition=PathCondition.trivially_true(),
            noise_draws=[NoiseDrawInfo(
                variable="eta", kind=NoiseKind.LAPLACE,
                center_expr=Var(ty=IRType.REAL, name="q"),
                scale_expr=Const.real(1.0), site_id=100,
            )],
            output_expr=Var(ty=IRType.REAL, name="eta"),
        ))
        builder = DensityRatioBuilder()
        dr = builder.build(ps)
        engine = CEGAREngine(config=CEGARConfig(
            max_refinements=10, timeout_seconds=15.0, solver_timeout_seconds=5.0,
        ))
        # scale=1 gives max_loss=sensitivity/scale=1, so eps=0.001 should fail
        result = engine.verify(ps, PureBudget(epsilon=0.001), dr)
        assert result.verdict in (
            CEGARVerdict.COUNTEREXAMPLE, CEGARVerdict.UNKNOWN,
        )

    def test_zero_epsilon_budget(self):
        b = PureBudget(epsilon=0.0)
        assert b.epsilon == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# 5. Mechanism with single branch (degenerate path)
# ═══════════════════════════════════════════════════════════════════════════


class TestSingleBranchMechanism:
    """A mechanism with no branching produces a single path."""

    def test_single_path_mechanism(self):
        ps = PathSet()
        ps.add(SymbolicPath(
            path_condition=PathCondition.trivially_true(),
            noise_draws=[NoiseDrawInfo(
                variable="eta", kind=NoiseKind.LAPLACE,
                center_expr=Var(ty=IRType.REAL, name="q"),
                scale_expr=Const.real(1.0), site_id=100,
            )],
            output_expr=Var(ty=IRType.REAL, name="eta"),
        ))
        assert ps.size() == 1
        builder = DensityRatioBuilder()
        dr = builder.build(ps)
        assert isinstance(dr, DensityRatioResult)
        assert len(dr.ratios) >= 1


# ═══════════════════════════════════════════════════════════════════════════
# 6. Composition of 0 mechanisms
# ═══════════════════════════════════════════════════════════════════════════


class TestCompositionZeroMechanisms:
    """Composing zero mechanisms should produce zero cost."""

    def test_basic_composition_zero(self):
        result = SequentialComposition.basic([])
        assert result.num_mechanisms == 0
        eps, _ = result.budget.to_approx_dp()
        assert eps == 0.0

    def test_advanced_composition_zero(self):
        result = SequentialComposition.advanced([])
        assert result.num_mechanisms == 0


# ═══════════════════════════════════════════════════════════════════════════
# 7. Composition of 1 mechanism
# ═══════════════════════════════════════════════════════════════════════════


class TestCompositionSingleMechanism:
    """Composing a single mechanism should return the same budget."""

    def test_basic_composition_one_pure(self):
        result = SequentialComposition.basic([PureBudget(epsilon=1.0)])
        assert result.num_mechanisms == 1
        eps, _ = result.budget.to_approx_dp()
        assert abs(eps - 1.0) < 1e-12

    def test_basic_composition_one_approx(self):
        result = SequentialComposition.basic([
            ApproxBudget(epsilon=1.0, delta=1e-5)
        ])
        assert result.num_mechanisms == 1

    def test_advanced_composition_one(self):
        result = SequentialComposition.advanced([PureBudget(epsilon=1.0)])
        assert result.num_mechanisms == 1


# ═══════════════════════════════════════════════════════════════════════════
# 8. Budget with epsilon=0
# ═══════════════════════════════════════════════════════════════════════════


class TestEpsilonZeroBudget:
    """epsilon=0 is perfect privacy – only a constant mechanism satisfies it."""

    def test_pure_budget_epsilon_zero(self):
        b = PureBudget(epsilon=0.0)
        assert b.epsilon == 0.0
        assert b.notion is PrivacyNotion.PURE_DP

    def test_approx_budget_epsilon_zero_delta_zero(self):
        b = ApproxBudget(epsilon=0.0, delta=0.0)
        assert b.epsilon == 0.0
        assert b.delta == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# 9. Budget with delta=0 for approx-DP
# ═══════════════════════════════════════════════════════════════════════════


class TestDeltaZeroBudget:
    """delta=0 in approx-DP reduces to pure-DP semantics."""

    def test_approx_budget_delta_zero(self):
        b = ApproxBudget(epsilon=1.0, delta=0.0)
        assert b.delta == 0.0
        assert b.epsilon == 1.0

    def test_delta_zero_to_approx_dp(self):
        b = ApproxBudget(epsilon=1.0, delta=0.0)
        eps, delta = b.to_approx_dp()
        assert eps == 1.0
        assert delta == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# 10. Numerical precision near machine epsilon
# ═══════════════════════════════════════════════════════════════════════════


class TestNumericalPrecision:
    """Test density ratios and budget arithmetic near machine epsilon."""

    def test_laplace_log_ratio_near_zero_shift(self):
        """Very small center shift → log ratio near zero."""
        lap = LaplaceNoise()
        ratio = lap.log_ratio(0.0, 0.0, 1e-15, 1.0)
        assert abs(ratio) < 1e-14

    def test_gaussian_log_ratio_near_zero_shift(self):
        g = GaussianNoise()
        ratio = g.log_ratio(0.0, 0.0, 1e-15, 1.0)
        assert abs(ratio) < 1e-14

    def test_composition_many_small_budgets(self):
        """Compose many small budgets – check for float overflow."""
        budgets = [PureBudget(epsilon=0.001) for _ in range(1000)]
        result = SequentialComposition.basic(budgets)
        assert result.num_mechanisms == 1000
        eps, _ = result.budget.to_approx_dp()
        assert abs(eps - 1.0) < 1e-6

    def test_composition_large_budgets_no_overflow(self):
        """Compose budgets that sum to large values."""
        budgets = [PureBudget(epsilon=100.0) for _ in range(100)]
        result = SequentialComposition.basic(budgets)
        eps, _ = result.budget.to_approx_dp()
        assert math.isfinite(eps)
        assert eps == pytest.approx(10000.0)


# ═══════════════════════════════════════════════════════════════════════════
# 11. Float overflow in composition
# ═══════════════════════════════════════════════════════════════════════════


class TestFloatOverflow:
    """Very large composition should not raise exceptions."""

    def test_many_pure_budgets(self):
        budgets = [PureBudget(epsilon=1.0) for _ in range(500)]
        result = SequentialComposition.basic(budgets)
        assert result.num_mechanisms == 500
        eps, _ = result.budget.to_approx_dp()
        assert eps == pytest.approx(500.0)

    def test_many_approx_budgets(self):
        budgets = [ApproxBudget(epsilon=0.1, delta=1e-6) for _ in range(100)]
        result = SequentialComposition.basic(budgets)
        assert result.num_mechanisms == 100
        eps, _ = result.budget.to_approx_dp()
        assert math.isfinite(eps)


# ═══════════════════════════════════════════════════════════════════════════
# 12. Certificate chain edge cases
# ═══════════════════════════════════════════════════════════════════════════


class TestCertificateChainEdgeCases:
    """Edge cases for CertificateChain composition."""

    def test_chain_single_cert(self):
        cert = VerificationCertificate(
            cert_type=CertificateType.VERIFICATION,
            mechanism_id="m1",
            mechanism_name="m1",
            privacy_notion=PrivacyNotion.PURE_DP,
            privacy_guarantee=PureBudget(epsilon=1.0),
            encoding_hash="hash1",
            density_bound_summary={"k": 1},
            proof_data={"verdict": "VERIFIED"},
        )
        cert.valid = True
        chain = CertificateChain(
            certificates=[cert],
            composition_type="sequential",
            composed_guarantee=PureBudget(epsilon=1.0),
        )
        assert chain.validate_chain() is True
        composed = chain.composed_budget()
        assert isinstance(composed, PureBudget)
        assert composed.epsilon == 1.0

    def test_chain_empty(self):
        chain = CertificateChain(certificates=[])
        assert chain.validate_chain() is False

    def test_chain_composed_budget_none_when_empty(self):
        chain = CertificateChain(certificates=[])
        assert chain.composed_budget() is None


# ═══════════════════════════════════════════════════════════════════════════
# 13. RDP budget edge cases
# ═══════════════════════════════════════════════════════════════════════════


class TestRDPEdgeCases:
    """RDP with boundary alpha values."""

    def test_rdp_large_alpha(self):
        b = RDPBudget(alpha=100.0, epsilon=1.0)
        assert b.alpha == 100.0

    def test_rdp_alpha_near_one_raises(self):
        with pytest.raises(ValueError):
            RDPBudget(alpha=1.0, epsilon=1.0)

    def test_rdp_to_approx_dp(self):
        b = RDPBudget(alpha=2.0, epsilon=1.0)
        eps, delta = b.to_approx_dp()
        assert eps >= 0.0
        assert delta >= 0.0


# ═══════════════════════════════════════════════════════════════════════════
# 14. GDP / zCDP budget edge cases
# ═══════════════════════════════════════════════════════════════════════════


class TestGDPZCDPEdgeCases:
    """Boundary values for GDP and zCDP budgets."""

    def test_zcdp_zero_rho(self):
        b = ZCDPBudget(rho=0.0)
        assert b.rho == 0.0

    def test_gdp_zero_mu(self):
        b = GDPBudget(mu=0.0)
        assert b.mu == 0.0

    def test_zcdp_to_approx(self):
        b = ZCDPBudget(rho=0.5)
        eps, delta = b.to_approx_dp()
        assert eps >= 0.0

    def test_gdp_to_approx(self):
        b = GDPBudget(mu=1.0)
        eps, delta = b.to_approx_dp()
        assert eps >= 0.0
