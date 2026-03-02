"""Privacy loss computation for differential privacy verification.

The :class:`PrivacyLossComputer` takes density ratio expressions and
computes the privacy loss under various DP notions: pure DP, approximate
(ε,δ)-DP, zCDP, RDP, GDP, and f-DP.

For each notion it checks whether the mechanism's privacy loss stays
within the declared budget, using both closed-form formulas for standard
mechanisms and numerical verification for non-standard cases.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from dpcegar.ir.types import (
    Abs,
    BinOp,
    BinOpKind,
    Const,
    IRType,
    NoiseKind,
    PrivacyBudget,
    PrivacyNotion,
    PureBudget,
    ApproxBudget,
    ZCDPBudget,
    RDPBudget,
    GDPBudget,
    FDPBudget,
    TypedExpr,
    Var,
)
from dpcegar.utils.math_utils import (
    Interval,
    gaussian_privacy_loss_approx_dp,
    gaussian_privacy_loss_zcdp,
    laplace_privacy_loss,
    phi as std_phi,
    phi_inv as std_phi_inv,
    rdp_to_approx_dp,
    safe_exp,
    safe_log,
    zcdp_to_approx_dp,
)
from dpcegar.density.ratio_builder import DensityRatioExpr, DensityRatioResult
from dpcegar.density.noise_models import (
    GaussianNoise,
    LaplaceNoise,
    NoiseModel,
    get_noise_model,
)
from dpcegar.paths.symbolic_path import NoiseDrawInfo


# ═══════════════════════════════════════════════════════════════════════════
# PRIVACY LOSS RESULT
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class PrivacyLossResult:
    """Result of a privacy loss computation.

    Attributes:
        notion:            The DP notion used for the analysis.
        is_private:        True if the mechanism satisfies the budget.
        computed_cost:     The computed privacy cost as a budget.
        declared_budget:   The declared privacy budget.
        worst_case_ratio:  The worst-case density ratio expression.
        details:           Additional computation details.
    """

    notion: PrivacyNotion
    is_private: bool
    computed_cost: PrivacyBudget | None = None
    declared_budget: PrivacyBudget | None = None
    worst_case_ratio: DensityRatioExpr | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Return a human-readable summary."""
        status = "PRIVATE" if self.is_private else "NOT PRIVATE"
        cost_str = str(self.computed_cost) if self.computed_cost else "unknown"
        budget_str = str(self.declared_budget) if self.declared_budget else "none"
        return f"[{status}] cost={cost_str}, budget={budget_str}"

    def __str__(self) -> str:
        return self.summary()


@dataclass(slots=True)
class PerPathLoss:
    """Privacy loss for a single path or path pair.

    Attributes:
        path_id_d:      Path ID for dataset d.
        path_id_d_prime: Path ID for dataset d'.
        max_loss:       Maximum privacy loss on this path.
        epsilon:        Pure-DP cost.
        rho:            zCDP cost (if applicable).
        rdp_cost:       RDP cost as function of alpha (if applicable).
    """

    path_id_d: int
    path_id_d_prime: int
    max_loss: float
    epsilon: float = 0.0
    rho: float | None = None
    rdp_cost: dict[float, float] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
# PRIVACY LOSS COMPUTER
# ═══════════════════════════════════════════════════════════════════════════


class PrivacyLossComputer:
    """Compute privacy loss under various DP notions.

    Given density ratio expressions produced by the DensityRatioBuilder,
    this class evaluates the privacy cost and checks whether it fits
    within a declared budget.

    Usage::

        computer = PrivacyLossComputer()
        result = computer.check_pure_dp(ratio_result, epsilon=1.0)
        print(result.summary())

    Args:
        num_grid_points: Number of grid points for numerical integration.
        rdp_alphas:      List of Rényi orders for RDP computation.
    """

    def __init__(
        self,
        num_grid_points: int = 1000,
        rdp_alphas: list[float] | None = None,
    ) -> None:
        self._num_grid = num_grid_points
        self._rdp_alphas = rdp_alphas or [
            1.5, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0, 16.0, 32.0, 64.0,
        ]

    # -- Pure DP -----------------------------------------------------------

    def check_pure_dp(
        self,
        ratio_result: DensityRatioResult,
        epsilon: float,
        noise_draws: list[NoiseDrawInfo] | None = None,
        sensitivity: float | None = None,
    ) -> PrivacyLossResult:
        """Check pure ε-differential privacy.

        For pure DP we need: max|L(o)| ≤ ε for all observations o.

        If sensitivity and noise parameters are known, uses closed-form.
        Otherwise falls back to worst-case analysis of the ratio expressions.

        Args:
            ratio_result: The density ratio expressions.
            epsilon:      The privacy budget.
            noise_draws:  Optional noise draw info for closed-form analysis.
            sensitivity:  Optional global sensitivity.

        Returns:
            A :class:`PrivacyLossResult`.
        """
        budget = PureBudget(epsilon=epsilon)

        if noise_draws and sensitivity is not None:
            cost = self._pure_dp_closed_form(noise_draws, sensitivity)
            if cost is not None:
                is_private = cost.epsilon <= epsilon
                return PrivacyLossResult(
                    notion=PrivacyNotion.PURE_DP,
                    is_private=is_private,
                    computed_cost=cost,
                    declared_budget=budget,
                    details={"method": "closed-form"},
                )

        max_loss = self._compute_max_loss_bound(ratio_result)
        computed = PureBudget(epsilon=max_loss)
        return PrivacyLossResult(
            notion=PrivacyNotion.PURE_DP,
            is_private=max_loss <= epsilon,
            computed_cost=computed,
            declared_budget=budget,
            details={"method": "symbolic-bound", "max_loss": max_loss},
        )

    # -- Approximate DP ----------------------------------------------------

    def check_approx_dp(
        self,
        ratio_result: DensityRatioResult,
        epsilon: float,
        delta: float,
        noise_draws: list[NoiseDrawInfo] | None = None,
        sensitivity: float | None = None,
    ) -> PrivacyLossResult:
        """Check approximate (ε,δ)-differential privacy.

        Computes the hockey-stick divergence:
            δ' = E_o[max(0, p(o|d)/p(o|d') - exp(ε))]

        and checks δ' ≤ δ.

        Args:
            ratio_result: Density ratio expressions.
            epsilon:      Privacy parameter ε.
            delta:        Failure probability δ.
            noise_draws:  Optional noise draw info.
            sensitivity:  Optional sensitivity.

        Returns:
            A :class:`PrivacyLossResult`.
        """
        budget = ApproxBudget(epsilon=epsilon, delta=delta)

        if noise_draws and sensitivity is not None:
            cost = self._approx_dp_closed_form(noise_draws, sensitivity, delta)
            if cost is not None:
                is_private = cost.epsilon <= epsilon and cost.delta <= delta
                return PrivacyLossResult(
                    notion=PrivacyNotion.APPROX_DP,
                    is_private=is_private,
                    computed_cost=cost,
                    declared_budget=budget,
                    details={"method": "closed-form"},
                )

        hockey_stick_delta = self._compute_hockey_stick(ratio_result, epsilon)
        computed = ApproxBudget(epsilon=epsilon, delta=hockey_stick_delta)
        return PrivacyLossResult(
            notion=PrivacyNotion.APPROX_DP,
            is_private=hockey_stick_delta <= delta,
            computed_cost=computed,
            declared_budget=budget,
            details={
                "method": "hockey-stick",
                "computed_delta": hockey_stick_delta,
            },
        )

    # -- zCDP --------------------------------------------------------------

    def check_zcdp(
        self,
        ratio_result: DensityRatioResult,
        rho: float,
        noise_draws: list[NoiseDrawInfo] | None = None,
        sensitivity: float | None = None,
    ) -> PrivacyLossResult:
        """Check zero-concentrated differential privacy.

        For zCDP we need:
            D_α(M(d) ‖ M(d')) ≤ ρα  for all α > 1

        For the Gaussian mechanism: ρ = Δ²/(2σ²).

        Args:
            ratio_result: Density ratio expressions.
            rho:          zCDP parameter ρ.
            noise_draws:  Optional noise draw info.
            sensitivity:  Optional sensitivity.

        Returns:
            A :class:`PrivacyLossResult`.
        """
        budget = ZCDPBudget(rho=rho)

        if noise_draws and sensitivity is not None:
            cost_rho = self._zcdp_closed_form(noise_draws, sensitivity)
            if cost_rho is not None:
                computed = ZCDPBudget(rho=cost_rho)
                return PrivacyLossResult(
                    notion=PrivacyNotion.ZCDP,
                    is_private=cost_rho <= rho,
                    computed_cost=computed,
                    declared_budget=budget,
                    details={"method": "closed-form", "rho": cost_rho},
                )

        max_loss = self._compute_max_loss_bound(ratio_result)
        estimated_rho = max_loss ** 2 / 2.0 if max_loss > 0 else 0.0
        computed = ZCDPBudget(rho=estimated_rho)
        return PrivacyLossResult(
            notion=PrivacyNotion.ZCDP,
            is_private=estimated_rho <= rho,
            computed_cost=computed,
            declared_budget=budget,
            details={"method": "from-pure-dp", "estimated_rho": estimated_rho},
        )

    # -- RDP ---------------------------------------------------------------

    def check_rdp(
        self,
        ratio_result: DensityRatioResult,
        alpha: float,
        epsilon_rdp: float,
        noise_draws: list[NoiseDrawInfo] | None = None,
        sensitivity: float | None = None,
    ) -> PrivacyLossResult:
        """Check Rényi differential privacy of order α.

        D_α(M(d) ‖ M(d')) ≤ ε_RDP

        For Gaussian mechanism:
            D_α = α·Δ²/(2σ²)

        Args:
            ratio_result: Density ratio expressions.
            alpha:        Rényi order α > 1.
            epsilon_rdp:  RDP privacy parameter.
            noise_draws:  Optional noise draw info.
            sensitivity:  Optional sensitivity.

        Returns:
            A :class:`PrivacyLossResult`.
        """
        budget = RDPBudget(alpha=alpha, epsilon=epsilon_rdp)

        if noise_draws and sensitivity is not None:
            cost_eps = self._rdp_closed_form(noise_draws, sensitivity, alpha)
            if cost_eps is not None:
                computed = RDPBudget(alpha=alpha, epsilon=cost_eps)
                return PrivacyLossResult(
                    notion=PrivacyNotion.RDP,
                    is_private=cost_eps <= epsilon_rdp,
                    computed_cost=computed,
                    declared_budget=budget,
                    details={"method": "closed-form", "alpha": alpha, "epsilon": cost_eps},
                )

        max_loss = self._compute_max_loss_bound(ratio_result)
        estimated_eps = alpha * max_loss if max_loss > 0 else 0.0
        computed = RDPBudget(alpha=alpha, epsilon=estimated_eps)
        return PrivacyLossResult(
            notion=PrivacyNotion.RDP,
            is_private=estimated_eps <= epsilon_rdp,
            computed_cost=computed,
            declared_budget=budget,
            details={"method": "from-pure-dp", "estimated_eps": estimated_eps},
        )

    # -- GDP ---------------------------------------------------------------

    def check_gdp(
        self,
        ratio_result: DensityRatioResult,
        mu: float,
        noise_draws: list[NoiseDrawInfo] | None = None,
        sensitivity: float | None = None,
    ) -> PrivacyLossResult:
        """Check Gaussian differential privacy.

        A mechanism is μ-GDP if the privacy loss random variable is
        dominated by N(μ²/2, μ²).  For the Gaussian mechanism: μ = Δ/σ.

        Args:
            ratio_result: Density ratio expressions.
            mu:           GDP parameter μ.
            noise_draws:  Optional noise draw info.
            sensitivity:  Optional sensitivity.

        Returns:
            A :class:`PrivacyLossResult`.
        """
        budget = GDPBudget(mu=mu)

        if noise_draws and sensitivity is not None:
            cost_mu = self._gdp_closed_form(noise_draws, sensitivity)
            if cost_mu is not None:
                computed = GDPBudget(mu=cost_mu)
                return PrivacyLossResult(
                    notion=PrivacyNotion.GDP,
                    is_private=cost_mu <= mu,
                    computed_cost=computed,
                    declared_budget=budget,
                    details={"method": "closed-form", "mu": cost_mu},
                )

        max_loss = self._compute_max_loss_bound(ratio_result)
        estimated_mu = math.sqrt(2.0 * max_loss) if max_loss > 0 else 0.0
        computed = GDPBudget(mu=estimated_mu)
        return PrivacyLossResult(
            notion=PrivacyNotion.GDP,
            is_private=estimated_mu <= mu,
            computed_cost=computed,
            declared_budget=budget,
            details={"method": "from-pure-dp", "estimated_mu": estimated_mu},
        )

    # -- f-DP --------------------------------------------------------------

    def check_fdp(
        self,
        ratio_result: DensityRatioResult,
        trade_off_fn: Callable[[float], float],
        noise_draws: list[NoiseDrawInfo] | None = None,
        sensitivity: float | None = None,
        num_points: int = 100,
    ) -> PrivacyLossResult:
        """Check f-differential privacy via the trade-off function.

        Evaluates the trade-off function T(α) at a grid of points
        and checks that the mechanism's type-I/type-II error trade-off
        is at least as good.

        Args:
            ratio_result: Density ratio expressions.
            trade_off_fn: The declared trade-off function T: [0,1]→[0,1].
            noise_draws:  Optional noise draw info.
            sensitivity:  Optional sensitivity.
            num_points:   Grid resolution for evaluation.

        Returns:
            A :class:`PrivacyLossResult`.
        """
        budget = FDPBudget(trade_off_fn=trade_off_fn)

        if noise_draws and sensitivity is not None:
            computed_fn = self._fdp_trade_off(noise_draws, sensitivity)
            if computed_fn is not None:
                is_private = True
                for i in range(num_points + 1):
                    alpha = i / num_points
                    if computed_fn(alpha) < trade_off_fn(alpha) - 1e-9:
                        is_private = False
                        break
                return PrivacyLossResult(
                    notion=PrivacyNotion.FDP,
                    is_private=is_private,
                    computed_cost=FDPBudget(trade_off_fn=computed_fn),
                    declared_budget=budget,
                    details={"method": "closed-form", "num_points": num_points},
                )

        return PrivacyLossResult(
            notion=PrivacyNotion.FDP,
            is_private=False,
            declared_budget=budget,
            details={"method": "no-closed-form-available"},
        )

    # -- Per-path analysis -------------------------------------------------

    def compute_per_path_loss(
        self,
        ratio_result: DensityRatioResult,
        noise_draws_map: dict[int, list[NoiseDrawInfo]] | None = None,
        sensitivity: float | None = None,
    ) -> list[PerPathLoss]:
        """Compute privacy loss for each path or path pair.

        Args:
            ratio_result:    The density ratio result.
            noise_draws_map: Optional mapping from path ID to noise draws.
            sensitivity:     Optional global sensitivity.

        Returns:
            List of per-path loss results.
        """
        results: list[PerPathLoss] = []

        for ratio in ratio_result.ratios:
            max_loss = self._estimate_max_ratio(ratio)
            loss = PerPathLoss(
                path_id_d=ratio.path_id_d,
                path_id_d_prime=ratio.path_id_d_prime,
                max_loss=max_loss,
                epsilon=max_loss,
            )
            if sensitivity is not None and sensitivity > 0 and max_loss > 0:
                # max_loss = Δ/σ (the privacy loss bound for Gaussian),
                # so σ = Δ/max_loss. Then ρ = Δ²/(2σ²) = max_loss²/2.
                sigma_est = sensitivity / max_loss
                loss.rho = (sensitivity ** 2) / (2.0 * sigma_est ** 2)
                for alpha in self._rdp_alphas:
                    loss.rdp_cost[alpha] = alpha * (sensitivity ** 2) / (2.0 * sigma_est ** 2)
            results.append(loss)

        return results

    # -- Numerical verification --------------------------------------------

    def numerical_verify(
        self,
        noise_kind: NoiseKind,
        center1: float,
        center2: float,
        scale: float,
        epsilon: float,
        num_samples: int = 100_000,
    ) -> dict[str, Any]:
        """Numerically verify privacy loss bounds via sampling.

        Draws samples from both distributions and computes the empirical
        maximum and mean privacy loss.

        Args:
            noise_kind:  Distribution family.
            center1:     Centre for dataset d.
            center2:     Centre for dataset d'.
            scale:       Scale parameter.
            epsilon:     Declared privacy budget.
            num_samples: Number of Monte Carlo samples.

        Returns:
            Dictionary with empirical statistics.
        """
        import random
        rng = random.Random(42)
        model = get_noise_model(noise_kind)

        max_loss = -math.inf
        min_loss = math.inf
        total_loss = 0.0
        violations = 0

        for _ in range(num_samples):
            x = model.sample(center1, scale, rng)
            lr = model.log_ratio(x, center1, center2, scale)
            abs_lr = abs(lr)
            max_loss = max(max_loss, abs_lr)
            min_loss = min(min_loss, abs_lr)
            total_loss += abs_lr
            if abs_lr > epsilon:
                violations += 1

        mean_loss = total_loss / num_samples
        violation_rate = violations / num_samples

        return {
            "max_loss": max_loss,
            "min_loss": min_loss,
            "mean_loss": mean_loss,
            "violation_rate": violation_rate,
            "num_samples": num_samples,
            "epsilon": epsilon,
            "is_private_empirical": violation_rate == 0.0,
        }

    # -- Closed-form computations ------------------------------------------

    def _pure_dp_closed_form(
        self, noise_draws: list[NoiseDrawInfo], sensitivity: float
    ) -> PureBudget | None:
        """Closed-form pure-DP cost for standard mechanisms."""
        total_epsilon = 0.0
        for nd in noise_draws:
            if isinstance(nd.scale_expr, Const):
                scale = float(nd.scale_expr.value)
                if scale <= 0:
                    return None
                if nd.kind == NoiseKind.LAPLACE:
                    total_epsilon += laplace_privacy_loss(scale, sensitivity)
                elif nd.kind == NoiseKind.EXPONENTIAL:
                    total_epsilon += sensitivity / scale
                else:
                    return None
            else:
                return None
        return PureBudget(epsilon=total_epsilon)

    def _approx_dp_closed_form(
        self,
        noise_draws: list[NoiseDrawInfo],
        sensitivity: float,
        delta: float,
    ) -> ApproxBudget | None:
        """Closed-form (ε,δ)-DP cost for Gaussian mechanism."""
        total_epsilon = 0.0
        total_delta = 0.0
        for nd in noise_draws:
            if isinstance(nd.scale_expr, Const):
                scale = float(nd.scale_expr.value)
                if scale <= 0:
                    return None
                if nd.kind == NoiseKind.GAUSSIAN:
                    eps = gaussian_privacy_loss_approx_dp(scale, sensitivity, delta)
                    total_epsilon += eps
                    total_delta += delta
                elif nd.kind == NoiseKind.LAPLACE:
                    total_epsilon += laplace_privacy_loss(scale, sensitivity)
                else:
                    return None
            else:
                return None
        return ApproxBudget(epsilon=total_epsilon, delta=min(total_delta, 1.0))

    def _zcdp_closed_form(
        self, noise_draws: list[NoiseDrawInfo], sensitivity: float
    ) -> float | None:
        """Closed-form zCDP cost for Gaussian mechanism."""
        total_rho = 0.0
        for nd in noise_draws:
            if isinstance(nd.scale_expr, Const):
                scale = float(nd.scale_expr.value)
                if scale <= 0:
                    return None
                if nd.kind == NoiseKind.GAUSSIAN:
                    total_rho += gaussian_privacy_loss_zcdp(scale, sensitivity)
                elif nd.kind == NoiseKind.LAPLACE:
                    eps = laplace_privacy_loss(scale, sensitivity)
                    total_rho += 0.5 * eps ** 2
                else:
                    return None
            else:
                return None
        return total_rho

    def _rdp_closed_form(
        self,
        noise_draws: list[NoiseDrawInfo],
        sensitivity: float,
        alpha: float,
    ) -> float | None:
        """Closed-form RDP cost for Gaussian mechanism."""
        total_eps = 0.0
        for nd in noise_draws:
            if isinstance(nd.scale_expr, Const):
                scale = float(nd.scale_expr.value)
                if scale <= 0:
                    return None
                if nd.kind == NoiseKind.GAUSSIAN:
                    total_eps += alpha * sensitivity ** 2 / (2.0 * scale ** 2)
                elif nd.kind == NoiseKind.LAPLACE:
                    eps_lap = laplace_privacy_loss(scale, sensitivity)
                    if alpha > 1 and eps_lap > 0:
                        # Mironov 2017, Proposition 3: Laplace RDP
                        total_eps += (1.0 / (alpha - 1)) * math.log(
                            alpha / (2 * alpha - 1) * math.exp((alpha - 1) * eps_lap)
                            + (alpha - 1) / (2 * alpha - 1) * math.exp(-alpha * eps_lap)
                        )
                    else:
                        total_eps += eps_lap
                else:
                    return None
            else:
                return None
        return total_eps

    def _gdp_closed_form(
        self, noise_draws: list[NoiseDrawInfo], sensitivity: float
    ) -> float | None:
        """Closed-form GDP cost for Gaussian mechanism."""
        total_mu_sq = 0.0
        for nd in noise_draws:
            if isinstance(nd.scale_expr, Const):
                scale = float(nd.scale_expr.value)
                if scale <= 0:
                    return None
                if nd.kind == NoiseKind.GAUSSIAN:
                    mu = sensitivity / scale
                    total_mu_sq += mu ** 2
                else:
                    return None
            else:
                return None
        return math.sqrt(total_mu_sq) if total_mu_sq > 0 else 0.0

    def _fdp_trade_off(
        self,
        noise_draws: list[NoiseDrawInfo],
        sensitivity: float,
    ) -> Callable[[float], float] | None:
        """Compute f-DP trade-off function for Gaussian mechanism.

        T(α) = Φ(Φ⁻¹(1-α) - μ)  where μ = Δ/σ.
        """
        all_gaussian = all(nd.kind == NoiseKind.GAUSSIAN for nd in noise_draws)
        if not all_gaussian:
            return None

        total_mu_sq = 0.0
        for nd in noise_draws:
            if not isinstance(nd.scale_expr, Const):
                return None
            scale = float(nd.scale_expr.value)
            if scale <= 0:
                return None
            mu = sensitivity / scale
            total_mu_sq += mu ** 2

        mu_composed = math.sqrt(total_mu_sq) if total_mu_sq > 0 else 0.0

        def trade_off(alpha: float) -> float:
            """Evaluate the trade-off function at α."""
            if alpha <= 0.0:
                return 1.0
            if alpha >= 1.0:
                return 0.0
            return std_phi(std_phi_inv(1.0 - alpha) - mu_composed)

        return trade_off

    # -- Symbolic bound estimation -----------------------------------------

    def _compute_max_loss_bound(self, ratio_result: DensityRatioResult) -> float:
        """Estimate the maximum absolute privacy loss from ratio expressions.

        Uses a heuristic analysis of the symbolic ratio expressions to
        extract bounds.  This is conservative (may over-estimate).
        """
        max_loss = 0.0
        for ratio in ratio_result.ratios:
            loss = self._estimate_max_ratio(ratio)
            max_loss = max(max_loss, loss)
        return max_loss

    @staticmethod
    def _estimate_max_ratio(ratio: DensityRatioExpr) -> float:
        """Heuristically estimate the maximum of a density ratio expression.

        Looks for patterns like ``expr / scale`` where scale is a constant,
        and returns a conservative bound.
        """
        expr = ratio.log_ratio

        if isinstance(expr, Const):
            return abs(float(expr.value))

        if isinstance(expr, BinOp) and expr.op == BinOpKind.DIV:
            if isinstance(expr.right, Const):
                scale = abs(float(expr.right.value))
                if scale > 0:
                    numerator_bound = _estimate_numerator_bound(expr.left)
                    return numerator_bound / scale

        if isinstance(expr, BinOp) and expr.op == BinOpKind.ADD:
            left_bound = PrivacyLossComputer._estimate_max_ratio_expr(expr.left)
            right_bound = PrivacyLossComputer._estimate_max_ratio_expr(expr.right)
            return left_bound + right_bound

        return math.inf

    @staticmethod
    def _estimate_max_ratio_expr(expr: TypedExpr) -> float:
        """Estimate max of a subexpression."""
        if isinstance(expr, Const):
            return abs(float(expr.value))
        if isinstance(expr, BinOp) and expr.op == BinOpKind.DIV:
            if isinstance(expr.right, Const):
                scale = abs(float(expr.right.value))
                if scale > 0:
                    num_bound = _estimate_numerator_bound(expr.left)
                    return num_bound / scale
        return math.inf

    def _compute_hockey_stick(
        self, ratio_result: DensityRatioResult, epsilon: float
    ) -> float:
        """Estimate the hockey-stick divergence E[max(0, exp(L) - exp(ε))].

        Uses the max loss bound to estimate an upper bound on δ.
        For Laplace mechanism: δ = 0 (pure DP).
        For Gaussian mechanism: uses the analytic formula.
        """
        max_loss = self._compute_max_loss_bound(ratio_result)
        if max_loss <= epsilon:
            return 0.0
        # Correct upper bound: δ ≤ max(0, 1 - exp(-max_loss + ε))
        delta_bound = max(0.0, 1.0 - safe_exp(-max_loss + epsilon))
        return min(delta_bound, 1.0)


# ═══════════════════════════════════════════════════════════════════════════
# HELPER
# ═══════════════════════════════════════════════════════════════════════════

def _estimate_numerator_bound(expr: TypedExpr) -> float:
    """Estimate the maximum absolute value of a numerator expression.

    Handles patterns:
    - |x - y| → returns ∞ (unknown bound)
    - constant → returns the value
    - abs(...) - abs(...) → returns ∞
    """
    if isinstance(expr, Const):
        return abs(float(expr.value))
    if isinstance(expr, Abs):
        return math.inf
    if isinstance(expr, BinOp):
        if expr.op == BinOpKind.SUB:
            if isinstance(expr.left, Abs) and isinstance(expr.right, Abs):
                return math.inf
        if expr.op == BinOpKind.MUL:
            if isinstance(expr.left, Const):
                return abs(float(expr.left.value)) * _estimate_numerator_bound(expr.right)
            if isinstance(expr.right, Const):
                return _estimate_numerator_bound(expr.left) * abs(float(expr.right.value))
    return math.inf
