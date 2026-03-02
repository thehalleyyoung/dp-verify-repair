"""Composition theorems for differential privacy.

Provides implementations of standard composition theorems used to
combine privacy costs from multiple mechanism invocations:

- **Sequential composition**: basic and advanced (Kairouz et al. 2015)
- **Parallel composition**: for disjoint dataset partitions
- **Subsampling amplification**: Poisson subsampling, shuffling
- **Composition optimiser**: choose the tightest bound
- **RDP composition**: per-alpha additive composition
- **zCDP composition**: additive rho composition
- **GDP composition**: central-limit-theorem-based composition
- **Heterogeneous composition**: across different mechanism types
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

from dpcegar.ir.types import (
    ApproxBudget,
    FDPBudget,
    GDPBudget,
    NoiseKind,
    PrivacyBudget,
    PrivacyNotion,
    PureBudget,
    RDPBudget,
    ZCDPBudget,
)
from dpcegar.utils.math_utils import (
    phi as std_phi,
    phi_inv as std_phi_inv,
    rdp_to_approx_dp,
    safe_exp,
    safe_log,
    zcdp_to_approx_dp,
)


# ═══════════════════════════════════════════════════════════════════════════
# COMPOSITION RESULT
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class CompositionResult:
    """Result of applying a composition theorem.

    Attributes:
        budget:       The composed privacy budget.
        theorem:      Name of the theorem used.
        num_mechanisms: Number of mechanisms composed.
        tightness:    Relative tightness indicator (lower is tighter).
        details:      Additional computation details.
    """

    budget: PrivacyBudget
    theorem: str = ""
    num_mechanisms: int = 0
    tightness: float = 1.0
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"CompositionResult({self.theorem}: {self.budget})"


# ═══════════════════════════════════════════════════════════════════════════
# SEQUENTIAL COMPOSITION
# ═══════════════════════════════════════════════════════════════════════════


class SequentialComposition:
    """Sequential composition of multiple mechanisms on the same dataset.

    Implements both basic composition (Σεᵢ) and advanced composition
    (Kairouz, Oh, Viswanath 2015).
    """

    @staticmethod
    def basic(budgets: list[PrivacyBudget]) -> CompositionResult:
        """Basic sequential composition: sum of epsilons.

        For pure DP: ε_total = Σ εᵢ
        For approx DP: (ε_total, δ_total) = (Σεᵢ, Σδᵢ)

        Args:
            budgets: List of individual mechanism budgets.

        Returns:
            The composed budget.
        """
        if not budgets:
            return CompositionResult(
                budget=PureBudget(epsilon=0.0),
                theorem="basic-sequential",
                num_mechanisms=0,
            )

        if all(isinstance(b, PureBudget) for b in budgets):
            total_eps = sum(b.epsilon for b in budgets)  # type: ignore[union-attr]
            return CompositionResult(
                budget=PureBudget(epsilon=total_eps),
                theorem="basic-sequential",
                num_mechanisms=len(budgets),
            )

        total_eps = 0.0
        total_delta = 0.0
        for b in budgets:
            eps, delta = b.to_approx_dp()
            total_eps += eps
            total_delta += delta

        return CompositionResult(
            budget=ApproxBudget(epsilon=total_eps, delta=min(total_delta, 1.0)),
            theorem="basic-sequential",
            num_mechanisms=len(budgets),
        )

    @staticmethod
    def advanced(
        budgets: list[PrivacyBudget],
        delta_total: float = 1e-5,
    ) -> CompositionResult:
        """Advanced composition theorem (Kairouz et al. 2015).

        For k-fold ε-DP composition with target δ:
            ε_total = ε√(2k·ln(1/δ')) + k·ε·(eε - 1)

        where δ' is split from δ_total.

        Args:
            budgets:     List of individual mechanism budgets.
            delta_total: Total failure probability budget.

        Returns:
            The composed budget under advanced composition.
        """
        if not budgets:
            return CompositionResult(
                budget=ApproxBudget(epsilon=0.0, delta=0.0),
                theorem="advanced-sequential",
                num_mechanisms=0,
            )

        k = len(budgets)

        delta_prime = delta_total / 2.0
        delta_mechs = delta_total / 2.0

        if all(isinstance(b, PureBudget) for b in budgets):
            eps_values = [b.epsilon for b in budgets]  # type: ignore[union-attr]
            max_eps = max(eps_values)
            sum_eps_sq = sum(e ** 2 for e in eps_values)

            if delta_prime > 0:
                log_term = math.log(1.0 / delta_prime)
                sqrt_term = math.sqrt(2.0 * sum_eps_sq * log_term)

                exp_term = sum(e * (safe_exp(e) - 1.0) for e in eps_values)
                eps_composed = min(
                    sqrt_term + exp_term,
                    sum(eps_values),
                )
            else:
                eps_composed = sum(eps_values)

            return CompositionResult(
                budget=ApproxBudget(epsilon=eps_composed, delta=delta_total),
                theorem="advanced-sequential",
                num_mechanisms=k,
                details={
                    "sum_eps_sq": sum_eps_sq,
                    "delta_prime": delta_prime,
                },
            )

        total_eps = 0.0
        total_delta = delta_mechs / k if k > 0 else 0.0
        eps_values = []
        for b in budgets:
            eps, delta = b.to_approx_dp(delta=total_delta)
            eps_values.append(eps)
            total_delta += delta

        sum_eps_sq = sum(e ** 2 for e in eps_values)
        if delta_prime > 0:
            log_term = math.log(1.0 / delta_prime)
            sqrt_term = math.sqrt(2.0 * sum_eps_sq * log_term)
            exp_term = sum(e * (safe_exp(e) - 1.0) for e in eps_values)
            eps_composed = min(sqrt_term + exp_term, sum(eps_values))
        else:
            eps_composed = sum(eps_values)

        return CompositionResult(
            budget=ApproxBudget(
                epsilon=eps_composed,
                delta=min(total_delta + delta_prime, 1.0),
            ),
            theorem="advanced-sequential",
            num_mechanisms=k,
        )

    @staticmethod
    def optimal_advanced(
        budgets: list[PrivacyBudget],
        delta_total: float = 1e-5,
    ) -> CompositionResult:
        """Find the optimal split of delta for advanced composition.

        Searches over delta allocations to minimise the composed epsilon.

        Args:
            budgets:     List of individual mechanism budgets.
            delta_total: Total failure probability.

        Returns:
            The tightest composed budget found.
        """
        best_result = SequentialComposition.basic(budgets)

        for fraction in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            delta_split = delta_total * fraction
            result = SequentialComposition.advanced(budgets, delta_split)
            if isinstance(result.budget, ApproxBudget) and isinstance(best_result.budget, ApproxBudget):
                if result.budget.epsilon < best_result.budget.epsilon:
                    best_result = result
            elif isinstance(result.budget, ApproxBudget):
                best_result = result

        best_result.theorem = "optimal-advanced"
        return best_result


# ═══════════════════════════════════════════════════════════════════════════
# PARALLEL COMPOSITION
# ═══════════════════════════════════════════════════════════════════════════


class ParallelComposition:
    """Parallel composition for mechanisms on disjoint dataset partitions.

    When mechanisms operate on disjoint subsets of the dataset, the
    composed privacy cost is the maximum (not sum) of individual costs.
    """

    @staticmethod
    def compose(budgets: list[PrivacyBudget]) -> CompositionResult:
        """Parallel composition: take the maximum privacy cost.

        Args:
            budgets: List of budgets for mechanisms on disjoint partitions.

        Returns:
            The composed budget (max over individual budgets).
        """
        if not budgets:
            return CompositionResult(
                budget=PureBudget(epsilon=0.0),
                theorem="parallel",
                num_mechanisms=0,
            )

        if all(isinstance(b, PureBudget) for b in budgets):
            max_eps = max(b.epsilon for b in budgets)  # type: ignore[union-attr]
            return CompositionResult(
                budget=PureBudget(epsilon=max_eps),
                theorem="parallel",
                num_mechanisms=len(budgets),
            )

        max_eps = 0.0
        max_delta = 0.0
        for b in budgets:
            eps, delta = b.to_approx_dp()
            max_eps = max(max_eps, eps)
            max_delta = max(max_delta, delta)

        return CompositionResult(
            budget=ApproxBudget(epsilon=max_eps, delta=max_delta),
            theorem="parallel",
            num_mechanisms=len(budgets),
        )


# ═══════════════════════════════════════════════════════════════════════════
# SUBSAMPLING AMPLIFICATION
# ═══════════════════════════════════════════════════════════════════════════


class SubsamplingAmplification:
    """Privacy amplification via subsampling.

    When a mechanism is applied to a random subsample of the dataset,
    the effective privacy cost is reduced.
    """

    @staticmethod
    def poisson(
        budget: PrivacyBudget,
        sampling_rate: float,
    ) -> CompositionResult:
        """Privacy amplification by Poisson subsampling.

        Each record is included independently with probability q.
        For pure ε-DP: ε' ≈ ln(1 + q(eε - 1))
        For (ε,δ)-DP: δ' = q·δ, ε' ≈ ln(1 + q(eε - 1))

        Args:
            budget:        The base mechanism's privacy budget.
            sampling_rate: Probability of including each record (q ∈ (0,1]).

        Returns:
            The amplified budget.
        """
        q = sampling_rate
        if q <= 0 or q > 1:
            raise ValueError(f"Sampling rate must be in (0, 1], got {q}")

        if q == 1.0:
            return CompositionResult(
                budget=budget,
                theorem="poisson-subsampling",
                num_mechanisms=1,
                details={"sampling_rate": q, "amplification": 1.0},
            )

        if isinstance(budget, PureBudget):
            eps = budget.epsilon
            amplified_eps = math.log(1.0 + q * (math.exp(eps) - 1.0))
            return CompositionResult(
                budget=PureBudget(epsilon=amplified_eps),
                theorem="poisson-subsampling",
                num_mechanisms=1,
                details={
                    "sampling_rate": q,
                    "original_eps": eps,
                    "amplified_eps": amplified_eps,
                },
            )

        eps, delta = budget.to_approx_dp()
        amplified_eps = math.log(1.0 + q * (math.exp(eps) - 1.0))
        amplified_delta = q * delta

        return CompositionResult(
            budget=ApproxBudget(epsilon=amplified_eps, delta=amplified_delta),
            theorem="poisson-subsampling",
            num_mechanisms=1,
            details={
                "sampling_rate": q,
                "original_eps": eps,
                "original_delta": delta,
            },
        )

    @staticmethod
    def poisson_rdp(
        budget: RDPBudget,
        sampling_rate: float,
    ) -> CompositionResult:
        """RDP amplification by Poisson subsampling.

        Uses the bound from Mironov et al. 2019:
            ε'(α) ≤ (1/(α-1)) ln(1 + C(α)·q²)

        where C(α) depends on the base RDP guarantee.

        Args:
            budget:        The base mechanism's RDP budget.
            sampling_rate: Subsampling probability.

        Returns:
            The amplified RDP budget.
        """
        q = sampling_rate
        alpha = budget.alpha
        eps = budget.epsilon

        if q <= 0 or q > 1:
            raise ValueError(f"Sampling rate must be in (0,1], got {q}")

        if q == 1.0:
            return CompositionResult(budget=budget, theorem="poisson-rdp", num_mechanisms=1)

        bound_terms: list[float] = []
        for j in range(2, int(alpha) + 1):
            log_binom = _log_binomial(int(alpha), j)
            log_term = log_binom + j * math.log(q) + (j - 1) * eps
            bound_terms.append(log_term)

        if bound_terms:
            max_term = max(bound_terms)
            log_sum = max_term + math.log(sum(math.exp(t - max_term) for t in bound_terms))
            amplified_eps = (1.0 / (alpha - 1)) * math.log(1.0 + safe_exp(log_sum))
        else:
            amplified_eps = eps * q ** 2

        amplified_eps = min(amplified_eps, eps)

        return CompositionResult(
            budget=RDPBudget(alpha=alpha, epsilon=amplified_eps),
            theorem="poisson-rdp",
            num_mechanisms=1,
            details={"sampling_rate": q},
        )

    @staticmethod
    def shuffling(
        budget: PrivacyBudget,
        n: int,
    ) -> CompositionResult:
        """Privacy amplification by shuffling (Erlingsson et al. 2019).

        The shuffle model amplifies local DP guarantees.
        For ε-LDP with n users: ε' ≈ O(ε·√(ln(1/δ)/n))

        Args:
            budget: The local mechanism's privacy budget.
            n:      Number of records / users.

        Returns:
            The amplified (central DP) budget.
        """
        if n <= 0:
            raise ValueError(f"Number of records must be positive, got {n}")

        if isinstance(budget, PureBudget):
            eps = budget.epsilon
            if eps <= 0:
                return CompositionResult(
                    budget=PureBudget(epsilon=0.0),
                    theorem="shuffling",
                    num_mechanisms=1,
                )

            delta = 1.0 / (n ** 2)
            log_term = math.log(1.0 / delta)
            amplified_eps = eps * math.sqrt(log_term / n)
            amplified_eps = min(amplified_eps, eps)

            return CompositionResult(
                budget=ApproxBudget(epsilon=amplified_eps, delta=delta),
                theorem="shuffling",
                num_mechanisms=1,
                details={
                    "n": n,
                    "original_eps": eps,
                    "amplified_eps": amplified_eps,
                },
            )

        eps, delta = budget.to_approx_dp()
        return CompositionResult(
            budget=ApproxBudget(epsilon=eps, delta=delta),
            theorem="shuffling",
            num_mechanisms=1,
        )


# ═══════════════════════════════════════════════════════════════════════════
# RDP COMPOSITION
# ═══════════════════════════════════════════════════════════════════════════


class RDPComposition:
    """Rényi differential privacy composition.

    RDP composes additively: for mechanisms with RDP guarantees
    (α, ε₁), (α, ε₂), ..., the composed guarantee is (α, Σεᵢ).
    """

    @staticmethod
    def compose(
        budgets: list[RDPBudget],
    ) -> CompositionResult:
        """Compose RDP budgets at the same order α.

        Args:
            budgets: List of RDP budgets (all at the same α).

        Returns:
            The composed RDP budget.

        Raises:
            ValueError: If budgets have different α values.
        """
        if not budgets:
            return CompositionResult(
                budget=RDPBudget(alpha=2.0, epsilon=0.0),
                theorem="rdp-additive",
                num_mechanisms=0,
            )

        alpha = budgets[0].alpha
        for b in budgets[1:]:
            if abs(b.alpha - alpha) > 1e-12:
                raise ValueError(
                    f"RDP composition requires same α: {alpha} vs {b.alpha}"
                )

        total_eps = sum(b.epsilon for b in budgets)
        return CompositionResult(
            budget=RDPBudget(alpha=alpha, epsilon=total_eps),
            theorem="rdp-additive",
            num_mechanisms=len(budgets),
        )

    @staticmethod
    def compose_multi_alpha(
        budgets: list[dict[float, float]],
        delta: float = 1e-5,
    ) -> CompositionResult:
        """Compose RDP budgets across multiple α values and convert to (ε,δ)-DP.

        For each α, sums the RDP costs, then converts to (ε,δ)-DP and
        picks the tightest bound.

        Args:
            budgets: List of dicts mapping α → ε_RDP.
            delta:   Target δ for conversion.

        Returns:
            The tightest (ε,δ)-DP guarantee.
        """
        if not budgets:
            return CompositionResult(
                budget=ApproxBudget(epsilon=0.0, delta=delta),
                theorem="rdp-multi-alpha",
                num_mechanisms=0,
            )

        all_alphas: set[float] = set()
        for b in budgets:
            all_alphas.update(b.keys())

        best_eps = math.inf
        best_alpha = 2.0

        for alpha in sorted(all_alphas):
            total_eps_rdp = sum(b.get(alpha, 0.0) for b in budgets)
            if alpha > 1:
                eps_dp = rdp_to_approx_dp(alpha, total_eps_rdp, delta)
                if eps_dp < best_eps:
                    best_eps = eps_dp
                    best_alpha = alpha

        return CompositionResult(
            budget=ApproxBudget(epsilon=max(best_eps, 0.0), delta=delta),
            theorem="rdp-multi-alpha",
            num_mechanisms=len(budgets),
            details={"best_alpha": best_alpha, "best_eps_rdp": best_eps},
        )


# ═══════════════════════════════════════════════════════════════════════════
# zCDP COMPOSITION
# ═══════════════════════════════════════════════════════════════════════════


class ZCDPComposition:
    """Zero-concentrated differential privacy composition.

    zCDP composes additively: ρ_total = Σ ρᵢ.
    """

    @staticmethod
    def compose(budgets: list[ZCDPBudget]) -> CompositionResult:
        """Compose zCDP budgets additively.

        Args:
            budgets: List of zCDP budgets.

        Returns:
            The composed zCDP budget.
        """
        if not budgets:
            return CompositionResult(
                budget=ZCDPBudget(rho=0.0),
                theorem="zcdp-additive",
                num_mechanisms=0,
            )

        total_rho = sum(b.rho for b in budgets)
        return CompositionResult(
            budget=ZCDPBudget(rho=total_rho),
            theorem="zcdp-additive",
            num_mechanisms=len(budgets),
        )

    @staticmethod
    def to_approx_dp(
        budgets: list[ZCDPBudget],
        delta: float = 1e-5,
    ) -> CompositionResult:
        """Compose zCDP budgets and convert to (ε,δ)-DP.

        Uses ε = ρ + 2√(ρ·ln(1/δ)).

        Args:
            budgets: List of zCDP budgets.
            delta:   Target δ.

        Returns:
            The composed (ε,δ)-DP guarantee.
        """
        total_rho = sum(b.rho for b in budgets)
        eps = zcdp_to_approx_dp(total_rho, delta)
        return CompositionResult(
            budget=ApproxBudget(epsilon=eps, delta=delta),
            theorem="zcdp-to-approx",
            num_mechanisms=len(budgets),
            details={"total_rho": total_rho},
        )


# ═══════════════════════════════════════════════════════════════════════════
# GDP COMPOSITION
# ═══════════════════════════════════════════════════════════════════════════


class GDPComposition:
    """Gaussian differential privacy composition.

    GDP composes via the central limit theorem: μ_total = √(Σ μᵢ²).
    """

    @staticmethod
    def compose(budgets: list[GDPBudget]) -> CompositionResult:
        """Compose GDP budgets via CLT.

        Args:
            budgets: List of GDP budgets.

        Returns:
            The composed GDP budget.
        """
        if not budgets:
            return CompositionResult(
                budget=GDPBudget(mu=0.0),
                theorem="gdp-clt",
                num_mechanisms=0,
            )

        total_mu_sq = sum(b.mu ** 2 for b in budgets)
        composed_mu = math.sqrt(total_mu_sq)

        return CompositionResult(
            budget=GDPBudget(mu=composed_mu),
            theorem="gdp-clt",
            num_mechanisms=len(budgets),
            details={"total_mu_sq": total_mu_sq},
        )

    @staticmethod
    def to_approx_dp(
        budgets: list[GDPBudget],
        delta: float = 1e-5,
    ) -> CompositionResult:
        """Compose GDP budgets and convert to (ε,δ)-DP.

        Args:
            budgets: List of GDP budgets.
            delta:   Target δ.

        Returns:
            The composed (ε,δ)-DP guarantee.
        """
        composed = GDPComposition.compose(budgets)
        composed_budget = composed.budget
        eps, _ = composed_budget.to_approx_dp(delta=delta)
        return CompositionResult(
            budget=ApproxBudget(epsilon=max(eps, 0.0), delta=delta),
            theorem="gdp-to-approx",
            num_mechanisms=len(budgets),
        )


# ═══════════════════════════════════════════════════════════════════════════
# COMPOSITION OPTIMISER
# ═══════════════════════════════════════════════════════════════════════════


class CompositionOptimizer:
    """Choose the tightest composition bound from multiple theorems.

    Evaluates all applicable composition theorems and returns the one
    producing the smallest privacy cost.
    """

    @staticmethod
    def compose(
        budgets: list[PrivacyBudget],
        delta: float = 1e-5,
    ) -> CompositionResult:
        """Find the tightest composition bound.

        Tries basic, advanced, and notion-specific composition and
        returns the result with the smallest ε (for a given δ).

        Args:
            budgets: List of privacy budgets to compose.
            delta:   Target δ for conversion.

        Returns:
            The tightest :class:`CompositionResult`.
        """
        if not budgets:
            return CompositionResult(
                budget=PureBudget(epsilon=0.0),
                theorem="none",
                num_mechanisms=0,
            )

        candidates: list[CompositionResult] = []

        basic = SequentialComposition.basic(budgets)
        candidates.append(basic)

        try:
            advanced = SequentialComposition.advanced(budgets, delta)
            candidates.append(advanced)
        except (ValueError, OverflowError):
            logger.debug("Advanced composition failed for %d budgets", len(budgets))

        if all(isinstance(b, ZCDPBudget) for b in budgets):
            zcdp_result = ZCDPComposition.to_approx_dp(
                [b for b in budgets if isinstance(b, ZCDPBudget)], delta
            )
            candidates.append(zcdp_result)

        if all(isinstance(b, RDPBudget) for b in budgets):
            rdp_budgets = [b for b in budgets if isinstance(b, RDPBudget)]
            try:
                rdp_result = RDPComposition.compose(rdp_budgets)
                composed_rdp = rdp_result.budget
                if isinstance(composed_rdp, RDPBudget):
                    eps_dp = rdp_to_approx_dp(composed_rdp.alpha, composed_rdp.epsilon, delta)
                    candidates.append(CompositionResult(
                        budget=ApproxBudget(epsilon=max(eps_dp, 0.0), delta=delta),
                        theorem="rdp-composed",
                        num_mechanisms=len(budgets),
                    ))
            except ValueError:
                logger.debug("RDP composition failed for %d budgets", len(rdp_budgets))

        if all(isinstance(b, GDPBudget) for b in budgets):
            gdp_result = GDPComposition.to_approx_dp(
                [b for b in budgets if isinstance(b, GDPBudget)], delta
            )
            candidates.append(gdp_result)

        best = min(
            candidates,
            key=lambda r: _extract_eps(r.budget),
        )
        best.tightness = _extract_eps(best.budget) / max(_extract_eps(basic.budget), 1e-30)
        return best

    @staticmethod
    def heterogeneous_compose(
        budgets: list[PrivacyBudget],
        delta: float = 1e-5,
    ) -> CompositionResult:
        """Compose heterogeneous mechanisms (different DP notions).

        Converts all budgets to (ε,δ)-DP first, then applies composition.

        Args:
            budgets: List of budgets from different DP notions.
            delta:   Target δ.

        Returns:
            The composed (ε,δ)-DP guarantee.
        """
        approx_budgets: list[ApproxBudget] = []
        per_mech_delta = delta / max(len(budgets), 1)

        for b in budgets:
            eps, d = b.to_approx_dp(delta=per_mech_delta)
            approx_budgets.append(ApproxBudget(epsilon=eps, delta=d))

        return CompositionOptimizer.compose(approx_budgets, delta)


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _extract_eps(budget: PrivacyBudget) -> float:
    """Extract the epsilon value from any budget type."""
    if isinstance(budget, PureBudget):
        return budget.epsilon
    if isinstance(budget, ApproxBudget):
        return budget.epsilon
    eps, _ = budget.to_approx_dp()
    return eps


def advanced_composition(
    epsilons: list[float],
    delta_total: float,
) -> float:
    """Advanced composition theorem for homogeneous/heterogeneous mechanisms.

    Implements the advanced composition theorem (Kairouz et al. 2015):
        ε_total = √(2·k·ln(1/δ'))·ε + k·ε·(eᵋ−1)

    For heterogeneous epsilons the generalised form is used:
        ε_total = √(2·ln(1/δ')·Σεᵢ²) + Σεᵢ·(eᵋⁱ−1)

    where k = len(epsilons) and δ' = delta_total / 2.

    Args:
        epsilons:    Per-mechanism ε values.
        delta_total: Total failure probability budget (must be > 0).

    Returns:
        The composed ε under the advanced composition theorem.

    Raises:
        ValueError: If *delta_total* is non-positive or *epsilons* is empty.
    """
    if not epsilons:
        raise ValueError("epsilons must be non-empty")
    if delta_total <= 0:
        raise ValueError(f"delta_total must be positive, got {delta_total}")

    delta_prime = delta_total / 2.0

    sum_eps_sq = sum(e ** 2 for e in epsilons)
    sqrt_term = math.sqrt(2.0 * sum_eps_sq * math.log(1.0 / delta_prime))
    exp_term = sum(e * (safe_exp(e) - 1.0) for e in epsilons)

    # The advanced bound; clamp to the basic bound (sum of epsilons).
    return min(sqrt_term + exp_term, sum(epsilons))


def _log_binomial(n: int, k: int) -> float:
    """Compute log(C(n,k)) using lgamma for numerical stability."""
    if k < 0 or k > n:
        return -math.inf
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
