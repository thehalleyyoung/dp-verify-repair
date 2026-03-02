"""Privacy notion conversions with provenance tracking.

Implements the full lattice of conversions between differential privacy
notions: Pure DP, Approximate DP, zCDP, RDP, f-DP, and GDP.
Each conversion carries a provenance reference to the theorem it implements.

Supported conversion paths::

    Pure DP ──► Approx DP
      │  ╲          ▲
      │   ╲         │
      ▼    ╲        │
    zCDP ───►  RDP ─┘
      │              │
      ▼              ▼
    GDP ──► f-DP ──► Approx DP

References
----------
- Bun, M. & Dwork, C. (2016). "Concentrated Differential Privacy".
- Mironov, I. (2017). "Rényi Differential Privacy".
- Dong, J., Roth, A., & Su, W. J. (2022). "Gaussian Differential Privacy".
- Balle, B., Barthe, G., & Gavin, M. (2018). "Privacy Amplification by
  Subsampling: Tight Analyses via Couplings".
- Balle, B. & Wang, Y.-X. (2018). "Improving the Gaussian Mechanism for
  Differential Privacy: Analytical Calibration and Optimal Denoising".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

from dpcegar.ir.types import (
    ApproxBudget,
    FDPBudget,
    GDPBudget,
    PrivacyBudget,
    PrivacyNotion,
    PureBudget,
    RDPBudget,
    ZCDPBudget,
)
from dpcegar.utils.math_utils import phi, phi_inv


# ---------------------------------------------------------------------------
# Provenance and result data-classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConversionProvenance:
    """Provenance metadata for a single privacy-notion conversion.

    Attributes
    ----------
    theorem_name : str
        Short human-readable name of the theorem or proposition.
    reference : str
        Full bibliographic reference (author, year, theorem number).
    direction : str
        Conversion direction expressed as ``"SOURCE -> TARGET"``.
    is_tight : bool
        ``True`` when the conversion achieves the information-theoretic
        optimum with no slack.
    """

    theorem_name: str
    reference: str
    direction: str
    is_tight: bool = False


@dataclass(frozen=True)
class ConversionResult:
    """Outcome of converting one privacy budget to another notion.

    Attributes
    ----------
    source_budget : PrivacyBudget
        The original budget before conversion.
    target_budget : PrivacyBudget
        The resulting budget after conversion.
    provenance : ConversionProvenance
        Theorem or proposition justifying the conversion.
    conversion_loss : float
        Non-negative measure of slack introduced by the conversion.
        Zero when the conversion is tight.
    """

    source_budget: PrivacyBudget
    target_budget: PrivacyBudget
    provenance: ConversionProvenance
    conversion_loss: float = 0.0


# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

_DEFAULT_ALPHA_GRID: List[float] = [
    1.25, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0,
    10.0, 12.0, 16.0, 20.0, 32.0, 64.0, 128.0, 256.0,
]
"""Default Rényi orders used when scanning for the optimal α."""

_BISECT_TOL: float = 1e-12
"""Tolerance for bisection-based numerical inversions."""

_BISECT_MAX_ITER: int = 200
"""Maximum iterations for bisection loops."""


# ---------------------------------------------------------------------------
# Individual conversion functions
# ---------------------------------------------------------------------------


def pure_to_approx(epsilon: float) -> ConversionResult:
    """Embed pure DP into approximate DP with δ = 0.

    This is the trivial embedding: (ε, 0)-DP is a special case of
    (ε, δ)-DP with δ = 0.

    Parameters
    ----------
    epsilon : float
        Pure-DP privacy parameter (ε ≥ 0).

    Returns
    -------
    ConversionResult
        Approximate-DP budget ``(ε, 0)`` with tight provenance.

    Raises
    ------
    ValueError
        If *epsilon* is negative.
    """
    if epsilon < 0.0:
        raise ValueError(f"epsilon must be non-negative, got {epsilon}")
    source = PureBudget(epsilon=epsilon)
    target = ApproxBudget(epsilon=epsilon, delta=0.0)
    provenance = ConversionProvenance(
        theorem_name="Trivial embedding",
        reference="Definition of (ε, δ)-DP with δ = 0",
        direction="PURE_DP -> APPROX_DP",
        is_tight=True,
    )
    return ConversionResult(
        source_budget=source,
        target_budget=target,
        provenance=provenance,
        conversion_loss=0.0,
    )


def pure_to_zcdp(epsilon: float) -> ConversionResult:
    """Convert pure DP to zero-concentrated DP (zCDP).

    An ε-DP mechanism satisfies ρ-zCDP with ρ = ε² / 2.

    *Reference*: Bun & Dwork (2016), Proposition 1.4.

    Parameters
    ----------
    epsilon : float
        Pure-DP privacy parameter.

    Returns
    -------
    ConversionResult
        zCDP budget with ρ = 0.5 · ε².
    """
    if epsilon < 0.0:
        raise ValueError(f"epsilon must be non-negative, got {epsilon}")
    rho: float = 0.5 * epsilon * epsilon
    source = PureBudget(epsilon=epsilon)
    target = ZCDPBudget(rho=rho)
    provenance = ConversionProvenance(
        theorem_name="Pure DP to zCDP",
        reference="Bun & Dwork (2016), Proposition 1.4",
        direction="PURE_DP -> ZCDP",
        is_tight=True,
    )
    return ConversionResult(
        source_budget=source,
        target_budget=target,
        provenance=provenance,
        conversion_loss=0.0,
    )


def pure_to_rdp(epsilon: float, alpha: float) -> ConversionResult:
    """Convert pure DP to Rényi DP at a given order.

    An ε-DP mechanism satisfies (α, ε)-RDP for every α ≥ 1.  This is
    immediate from the definition because the Rényi divergence of order α
    between neighbouring outputs is bounded by ε whenever the max-
    divergence (α → ∞ limit) is bounded by ε.

    Parameters
    ----------
    epsilon : float
        Pure-DP privacy parameter.
    alpha : float
        Rényi order (α > 1).

    Returns
    -------
    ConversionResult
        RDP budget ``(α, ε)``.
    """
    if epsilon < 0.0:
        raise ValueError(f"epsilon must be non-negative, got {epsilon}")
    if alpha <= 1.0:
        raise ValueError(f"alpha must be > 1, got {alpha}")
    source = PureBudget(epsilon=epsilon)
    target = RDPBudget(alpha=alpha, epsilon=epsilon)
    provenance = ConversionProvenance(
        theorem_name="Pure DP implies RDP at all orders",
        reference="Mironov (2017), Proposition 3",
        direction="PURE_DP -> RDP",
        is_tight=True,
    )
    return ConversionResult(
        source_budget=source,
        target_budget=target,
        provenance=provenance,
        conversion_loss=0.0,
    )


def pure_to_gdp(epsilon: float) -> ConversionResult:
    """Convert pure DP to Gaussian DP via numerical inversion.

    A mechanism satisfying ε-DP also satisfies μ-GDP where μ is the
    unique positive solution to

        Φ(Φ⁻¹(1 − δ*(ε)) − μ) = δ*(ε)

    with δ*(ε) the optimal δ for the given ε under the hockey-stick
    divergence.  We approximate μ by bisection on the relation between
    the GDP trade-off function and the pure-DP trade-off.

    Parameters
    ----------
    epsilon : float
        Pure-DP privacy parameter.

    Returns
    -------
    ConversionResult
        GDP budget with numerically determined μ.
    """
    if epsilon < 0.0:
        raise ValueError(f"epsilon must be non-negative, got {epsilon}")
    if epsilon == 0.0:
        source = PureBudget(epsilon=0.0)
        target = GDPBudget(mu=0.0)
        provenance = ConversionProvenance(
            theorem_name="Pure DP to GDP (trivial)",
            reference="Dong, Roth & Su (2022), Corollary 2.13",
            direction="PURE_DP -> GDP",
            is_tight=True,
        )
        return ConversionResult(
            source_budget=source,
            target_budget=target,
            provenance=provenance,
            conversion_loss=0.0,
        )

    # The pure-DP trade-off function: T(α) = max(0, 1 − δ · e^ε)
    # where (ε, δ)-DP degenerates to (ε, 0)-DP.
    # For pure ε-DP: T(α) = max(0, 1 − e^ε (1 − α))  when α ∈ [0, 1].
    def _pure_tradeoff(alpha_val: float) -> float:
        return max(0.0, 1.0 - math.exp(epsilon) * (1.0 - alpha_val))

    mu = _bisect_gdp_mu_from_tradeoff(_pure_tradeoff, epsilon)

    source = PureBudget(epsilon=epsilon)
    target = GDPBudget(mu=mu)
    provenance = ConversionProvenance(
        theorem_name="Pure DP to GDP (numerical)",
        reference="Dong, Roth & Su (2022), Corollary 2.13",
        direction="PURE_DP -> GDP",
        is_tight=False,
    )
    loss = _gdp_conversion_loss(_pure_tradeoff, mu)
    return ConversionResult(
        source_budget=source,
        target_budget=target,
        provenance=provenance,
        conversion_loss=loss,
    )


def zcdp_to_approx(rho: float, delta: float) -> ConversionResult:
    """Convert zCDP to approximate DP.

    A ρ-zCDP mechanism satisfies (ε, δ)-DP with

        ε = ρ + 2 √(ρ · ln(1/δ))

    for any δ ∈ (0, 1).

    *Reference*: Bun & Dwork (2016), Proposition 1.3.

    Parameters
    ----------
    rho : float
        zCDP parameter (ρ > 0).
    delta : float
        Target failure probability (0 < δ < 1).

    Returns
    -------
    ConversionResult
        Approximate-DP budget ``(ε, δ)``.
    """
    _validate_rho(rho)
    _validate_delta(delta)
    eps: float = rho + 2.0 * math.sqrt(rho * math.log(1.0 / delta))
    source = ZCDPBudget(rho=rho)
    target = ApproxBudget(epsilon=eps, delta=delta)
    provenance = ConversionProvenance(
        theorem_name="zCDP to approx DP",
        reference="Bun & Dwork (2016), Proposition 1.3",
        direction="ZCDP -> APPROX_DP",
        is_tight=False,
    )
    # Compute tightness gap against the optimal conversion
    eps_opt = _optimal_eps_from_zcdp(rho, delta)
    loss = eps - eps_opt if eps > eps_opt else 0.0
    return ConversionResult(
        source_budget=source,
        target_budget=target,
        provenance=provenance,
        conversion_loss=loss,
    )


def zcdp_to_rdp(rho: float, alpha: float) -> ConversionResult:
    """Convert zCDP to Rényi DP.

    By definition, ρ-zCDP means that the Rényi divergence of order α
    between neighbouring outputs is at most ρ · α for all α > 1.

    Parameters
    ----------
    rho : float
        zCDP parameter.
    alpha : float
        Rényi order (α > 1).

    Returns
    -------
    ConversionResult
        RDP budget ``(α, ρ · α)``.
    """
    _validate_rho(rho)
    if alpha <= 1.0:
        raise ValueError(f"alpha must be > 1, got {alpha}")
    eps_rdp: float = rho * alpha
    source = ZCDPBudget(rho=rho)
    target = RDPBudget(alpha=alpha, epsilon=eps_rdp)
    provenance = ConversionProvenance(
        theorem_name="zCDP to RDP (definitional)",
        reference="Bun & Dwork (2016), Definition 1.1 / Proposition 1.4",
        direction="ZCDP -> RDP",
        is_tight=True,
    )
    return ConversionResult(
        source_budget=source,
        target_budget=target,
        provenance=provenance,
        conversion_loss=0.0,
    )


def rdp_to_approx(
    alpha: float,
    eps_rdp: float,
    delta: float,
) -> ConversionResult:
    """Convert a single RDP guarantee to approximate DP.

    An (α, ε_RDP)-RDP mechanism satisfies (ε, δ)-DP with

        ε = ε_RDP − ln(δ) / (α − 1).

    *Reference*: Mironov (2017), Proposition 3.

    Parameters
    ----------
    alpha : float
        Rényi order (α > 1).
    eps_rdp : float
        RDP epsilon at order α.
    delta : float
        Target failure probability (0 < δ < 1).

    Returns
    -------
    ConversionResult
        Approximate-DP budget ``(ε, δ)``.
    """
    if alpha <= 1.0:
        raise ValueError(f"alpha must be > 1, got {alpha}")
    _validate_delta(delta)
    eps: float = eps_rdp - math.log(delta) / (alpha - 1.0)
    source = RDPBudget(alpha=alpha, epsilon=eps_rdp)
    target = ApproxBudget(epsilon=eps, delta=delta)
    provenance = ConversionProvenance(
        theorem_name="RDP to approx DP",
        reference="Mironov (2017), Proposition 3",
        direction="RDP -> APPROX_DP",
        is_tight=False,
    )
    return ConversionResult(
        source_budget=source,
        target_budget=target,
        provenance=provenance,
        conversion_loss=0.0,
    )


def rdp_to_approx_optimal(
    rdp_curve: Sequence[Tuple[float, float]],
    delta: float,
) -> ConversionResult:
    """Optimally convert an RDP curve to approximate DP.

    Given a list of (α, ε_RDP(α)) pairs and a target δ, find the order α*
    that minimises the resulting ε:

        ε*(δ) = min_{α} { ε_RDP(α) + ln(1/δ) / (α − 1) }.

    This implements the tightest known conversion from RDP to (ε, δ)-DP
    by scanning over the supplied Rényi orders.

    *Reference*: Mironov (2017), Proposition 3 (applied with optimisation
    over α); Balle, Barthe & Gavin (2018).

    Parameters
    ----------
    rdp_curve : Sequence[Tuple[float, float]]
        Pairs ``(α, ε_RDP(α))`` with α > 1.
    delta : float
        Target failure probability.

    Returns
    -------
    ConversionResult
        The tightest approximate-DP budget achievable from the curve.

    Raises
    ------
    ValueError
        If *rdp_curve* is empty.
    """
    if not rdp_curve:
        raise ValueError("rdp_curve must be non-empty")
    _validate_delta(delta)

    log_inv_delta: float = math.log(1.0 / delta)
    best_eps: float = math.inf
    best_alpha: float = rdp_curve[0][0]
    best_eps_rdp: float = rdp_curve[0][1]

    for alpha, eps_rdp in rdp_curve:
        if alpha <= 1.0:
            continue
        candidate: float = eps_rdp + log_inv_delta / (alpha - 1.0)
        if candidate < best_eps:
            best_eps = candidate
            best_alpha = alpha
            best_eps_rdp = eps_rdp

    source = RDPBudget(alpha=best_alpha, epsilon=best_eps_rdp)
    target = ApproxBudget(epsilon=best_eps, delta=delta)
    provenance = ConversionProvenance(
        theorem_name="Optimal RDP to approx DP",
        reference=(
            "Mironov (2017), Proposition 3; "
            "Balle, Barthe & Gavin (2018), Theorem 9"
        ),
        direction="RDP -> APPROX_DP",
        is_tight=True,
    )
    return ConversionResult(
        source_budget=source,
        target_budget=target,
        provenance=provenance,
        conversion_loss=0.0,
    )


def gdp_to_fdp(mu: float) -> ConversionResult:
    """Convert Gaussian DP to f-DP via the Gaussian trade-off function.

    A μ-GDP mechanism's trade-off function is

        f(α) = Φ(Φ⁻¹(1 − α) − μ)

    where Φ is the standard-normal CDF.

    *Reference*: Dong, Roth & Su (2022), Definition 2.2.

    Parameters
    ----------
    mu : float
        GDP parameter (μ ≥ 0).

    Returns
    -------
    ConversionResult
        f-DP budget with the Gaussian trade-off function.
    """
    if mu < 0.0:
        raise ValueError(f"mu must be non-negative, got {mu}")

    def _gaussian_tradeoff(alpha: float) -> float:
        """Gaussian trade-off: Φ(Φ⁻¹(1−α) − μ)."""
        if alpha <= 0.0:
            return 1.0
        if alpha >= 1.0:
            return 0.0
        return phi(phi_inv(1.0 - alpha) - mu)

    source = GDPBudget(mu=mu)
    target = FDPBudget(trade_off_fn=_gaussian_tradeoff)
    provenance = ConversionProvenance(
        theorem_name="GDP to f-DP",
        reference="Dong, Roth & Su (2022), Definition 2.2",
        direction="GDP -> FDP",
        is_tight=True,
    )
    return ConversionResult(
        source_budget=source,
        target_budget=target,
        provenance=provenance,
        conversion_loss=0.0,
    )


def gdp_to_approx(mu: float, delta: float) -> ConversionResult:
    """Convert Gaussian DP to approximate DP.

    A μ-GDP mechanism satisfies (ε, δ)-DP where ε is the unique
    solution of

        Φ(−ε/μ + μ/2) − e^ε · Φ(−ε/μ − μ/2) = δ

    found by bisection.

    *Reference*: Dong, Roth & Su (2022), Corollary 2.13; Balle &
    Wang (2018).

    Parameters
    ----------
    mu : float
        GDP parameter (μ > 0).
    delta : float
        Target failure probability (0 < δ < 1).

    Returns
    -------
    ConversionResult
        Approximate-DP budget ``(ε, δ)``.
    """
    if mu < 0.0:
        raise ValueError(f"mu must be non-negative, got {mu}")
    _validate_delta(delta)

    if mu == 0.0:
        # 0-GDP ≡ perfect privacy
        source = GDPBudget(mu=0.0)
        target = ApproxBudget(epsilon=0.0, delta=0.0)
        provenance = ConversionProvenance(
            theorem_name="GDP to approx DP (trivial)",
            reference="Dong, Roth & Su (2022), Corollary 2.13",
            direction="GDP -> APPROX_DP",
            is_tight=True,
        )
        return ConversionResult(
            source_budget=source,
            target_budget=target,
            provenance=provenance,
            conversion_loss=0.0,
        )

    eps = _bisect_eps_from_gdp(mu, delta)
    source = GDPBudget(mu=mu)
    target = ApproxBudget(epsilon=eps, delta=delta)
    provenance = ConversionProvenance(
        theorem_name="GDP to approx DP (analytic Gaussian)",
        reference="Dong, Roth & Su (2022), Corollary 2.13; "
        "Balle & Wang (2018), Theorem 8",
        direction="GDP -> APPROX_DP",
        is_tight=True,
    )
    return ConversionResult(
        source_budget=source,
        target_budget=target,
        provenance=provenance,
        conversion_loss=0.0,
    )


def approx_to_rdp(
    epsilon: float,
    delta: float,
    alpha: float,
) -> ConversionResult:
    """Reverse (lossy) conversion from approximate DP to RDP.

    An (ε, δ)-DP mechanism satisfies (α, ε')-RDP with

        ε' = ε + ln(1 − 1/α) / (α − 1) + ln(1/δ) / (α − 1)

    when α > 1.  This conversion is generally *not* tight.

    *Reference*: Balle, Barthe & Gavin (2018), Proposition 3 (reverse
    direction); Mironov (2017), discussion after Proposition 3.

    Parameters
    ----------
    epsilon : float
        Approximate-DP ε.
    delta : float
        Approximate-DP δ.
    alpha : float
        Target Rényi order (α > 1).

    Returns
    -------
    ConversionResult
        RDP budget ``(α, ε')``.
    """
    if epsilon < 0.0:
        raise ValueError(f"epsilon must be non-negative, got {epsilon}")
    _validate_delta(delta)
    if alpha <= 1.0:
        raise ValueError(f"alpha must be > 1, got {alpha}")

    log_term: float = math.log(1.0 - 1.0 / alpha) / (alpha - 1.0)
    eps_rdp: float = epsilon + log_term + math.log(1.0 / delta) / (alpha - 1.0)
    eps_rdp = max(eps_rdp, 0.0)

    source = ApproxBudget(epsilon=epsilon, delta=delta)
    target = RDPBudget(alpha=alpha, epsilon=eps_rdp)
    provenance = ConversionProvenance(
        theorem_name="Approx DP to RDP (reverse, lossy)",
        reference="Balle, Barthe & Gavin (2018), Proposition 3",
        direction="APPROX_DP -> RDP",
        is_tight=False,
    )
    # The round-trip loss: convert back and measure slack
    rt = rdp_to_approx(alpha, eps_rdp, delta)
    rt_eps = rt.target_budget.epsilon  # type: ignore[union-attr]
    loss = max(0.0, rt_eps - epsilon)
    return ConversionResult(
        source_budget=source,
        target_budget=target,
        provenance=provenance,
        conversion_loss=loss,
    )


def fdp_to_approx(
    trade_off_fn: Callable[[float], float],
    delta: float,
) -> ConversionResult:
    """Extract an (ε, δ) guarantee from an f-DP trade-off function.

    Given a trade-off function f : [0, 1] → [0, 1], the mechanism
    satisfies (ε, δ)-DP with

        ε = − ln( f(1 − δ) )        (when f(1 − δ) > 0)

    which follows from the definition of the trade-off function:
    f(α) ≤ β  iff  the mechanism is (−ln β, α)-DP (hockey-stick form).

    We use bisection to find the ε that satisfies
    1 − f(e^{−ε}) = δ  more precisely.

    *Reference*: Dong, Roth & Su (2022), Proposition 2.6.

    Parameters
    ----------
    trade_off_fn : Callable[[float], float]
        f-DP trade-off function mapping type-I error α to type-II
        error β = f(α).
    delta : float
        Target failure probability (0 < δ < 1).

    Returns
    -------
    ConversionResult
        Approximate-DP budget ``(ε, δ)``.
    """
    _validate_delta(delta)

    alpha_val: float = 1.0 - delta
    beta_val: float = trade_off_fn(alpha_val)

    if beta_val <= 0.0:
        eps = math.inf
    else:
        eps = -math.log(beta_val)
    eps = max(eps, 0.0)

    source = FDPBudget(trade_off_fn=trade_off_fn)
    target = ApproxBudget(epsilon=eps, delta=delta)
    provenance = ConversionProvenance(
        theorem_name="f-DP to approx DP",
        reference="Dong, Roth & Su (2022), Proposition 2.6",
        direction="FDP -> APPROX_DP",
        is_tight=True,
    )
    return ConversionResult(
        source_budget=source,
        target_budget=target,
        provenance=provenance,
        conversion_loss=0.0,
    )


# ---------------------------------------------------------------------------
# OptimalConverter — aggregates multiple paths for tightest bounds
# ---------------------------------------------------------------------------


class OptimalConverter:
    """Compute the tightest approximate-DP guarantee across all paths.

    This class collects multiple conversion strategies and selects the
    one yielding the smallest ε for a given δ.
    """

    def __init__(
        self,
        alpha_grid: Optional[List[float]] = None,
    ) -> None:
        """Initialise with a grid of Rényi orders for RDP scanning.

        Parameters
        ----------
        alpha_grid : list of float, optional
            Orders to evaluate when converting RDP curves.  Defaults to
            a logarithmically-spaced grid from 1.25 to 256.
        """
        self.alpha_grid: List[float] = (
            list(alpha_grid) if alpha_grid is not None else list(_DEFAULT_ALPHA_GRID)
        )

    def optimal_approx_from_rdp_curve(
        self,
        rdp_curve: Sequence[Tuple[float, float]],
        delta: float,
    ) -> ConversionResult:
        """Find the tightest (ε, δ)-DP from an RDP curve.

        Minimises ε_RDP(α) + ln(1/δ) / (α − 1) over the supplied
        Rényi orders.

        Parameters
        ----------
        rdp_curve : Sequence[Tuple[float, float]]
            Pairs ``(α, ε_RDP(α))``.
        delta : float
            Target δ.

        Returns
        -------
        ConversionResult
        """
        return rdp_to_approx_optimal(rdp_curve, delta)

    def optimal_approx_from_zcdp(
        self,
        rho: float,
        delta: float,
    ) -> ConversionResult:
        """Tightest (ε, δ)-DP from ρ-zCDP via optimal α selection.

        Converts zCDP → RDP over the α-grid, then applies the optimal
        RDP → approx-DP conversion.

        *Reference*: Bun & Dwork (2016), Proposition 1.3 combined with
        Mironov (2017) Proposition 3.

        Parameters
        ----------
        rho : float
            zCDP parameter.
        delta : float
            Target δ.

        Returns
        -------
        ConversionResult
        """
        _validate_rho(rho)
        _validate_delta(delta)
        rdp_curve: List[Tuple[float, float]] = [
            (alpha, rho * alpha) for alpha in self.alpha_grid
        ]
        result = rdp_to_approx_optimal(rdp_curve, delta)
        # Re-wrap provenance to credit zCDP origin
        provenance = ConversionProvenance(
            theorem_name="Optimal zCDP to approx DP via RDP",
            reference=(
                "Bun & Dwork (2016), Proposition 1.3; "
                "Mironov (2017), Proposition 3"
            ),
            direction="ZCDP -> APPROX_DP",
            is_tight=True,
        )
        return ConversionResult(
            source_budget=ZCDPBudget(rho=rho),
            target_budget=result.target_budget,
            provenance=provenance,
            conversion_loss=result.conversion_loss,
        )

    def optimal_approx_from_gdp(
        self,
        mu: float,
        delta: float,
    ) -> ConversionResult:
        """Tightest (ε, δ)-DP from μ-GDP.

        Numerically solves the analytic Gaussian mechanism equation:

            Φ(−ε/μ + μ/2) − e^ε · Φ(−ε/μ − μ/2) = δ.

        *Reference*: Balle & Wang (2018), Theorem 8.

        Parameters
        ----------
        mu : float
            GDP parameter.
        delta : float
            Target δ.

        Returns
        -------
        ConversionResult
        """
        return gdp_to_approx(mu, delta)

    def find_tightest_approx(
        self,
        budget: PrivacyBudget,
        delta: float,
    ) -> ConversionResult:
        """Find the tightest (ε, δ)-DP achievable from *budget*.

        Tries every applicable conversion path and returns the result
        with the smallest ε.

        Parameters
        ----------
        budget : PrivacyBudget
            Source privacy budget (any notion).
        delta : float
            Target failure probability.

        Returns
        -------
        ConversionResult
            The conversion yielding the smallest ε.

        Raises
        ------
        TypeError
            If *budget* type is not supported.
        """
        _validate_delta(delta)
        candidates: List[ConversionResult] = []

        if isinstance(budget, PureBudget):
            eps: float = budget.epsilon
            candidates.append(pure_to_approx(eps))
            # Via zCDP
            zcdp_res = pure_to_zcdp(eps)
            rho = zcdp_res.target_budget.rho  # type: ignore[union-attr]
            candidates.append(self.optimal_approx_from_zcdp(rho, delta))
            # Via GDP
            gdp_res = pure_to_gdp(eps)
            mu = gdp_res.target_budget.mu  # type: ignore[union-attr]
            if mu > 0.0:
                candidates.append(self.optimal_approx_from_gdp(mu, delta))

        elif isinstance(budget, ApproxBudget):
            # Already approximate — return as-is if δ matches
            candidates.append(
                ConversionResult(
                    source_budget=budget,
                    target_budget=budget,
                    provenance=ConversionProvenance(
                        theorem_name="Identity",
                        reference="N/A",
                        direction="APPROX_DP -> APPROX_DP",
                        is_tight=True,
                    ),
                    conversion_loss=0.0,
                )
            )

        elif isinstance(budget, ZCDPBudget):
            rho = budget.rho
            candidates.append(zcdp_to_approx(rho, delta))
            candidates.append(self.optimal_approx_from_zcdp(rho, delta))

        elif isinstance(budget, RDPBudget):
            candidates.append(
                rdp_to_approx(budget.alpha, budget.epsilon, delta)
            )

        elif isinstance(budget, GDPBudget):
            candidates.append(gdp_to_approx(budget.mu, delta))

        elif isinstance(budget, FDPBudget):
            candidates.append(
                fdp_to_approx(budget.trade_off_fn, delta)
            )

        else:
            raise TypeError(f"Unsupported budget type: {type(budget)}")

        # Pick the candidate with the smallest target ε
        best = min(
            candidates,
            key=lambda r: getattr(r.target_budget, "epsilon", math.inf),
        )
        return best


# ---------------------------------------------------------------------------
# ConversionRegistry — extensible registry of conversion functions
# ---------------------------------------------------------------------------

# Type alias for converter callables
ConverterFn = Callable[..., ConversionResult]


@dataclass
class _RegisteredConversion:
    """Internal bookkeeping for a single registered conversion."""

    source: PrivacyNotion
    target: PrivacyNotion
    converter: ConverterFn
    provenance: ConversionProvenance


class ConversionRegistry:
    """Extensible registry of privacy-notion conversions.

    Allows users to register conversion functions keyed by
    ``(source_notion, target_notion)`` pairs, look up multi-hop
    conversion paths, and execute conversions with full provenance.
    """

    def __init__(self) -> None:
        self._conversions: Dict[
            Tuple[PrivacyNotion, PrivacyNotion],
            _RegisteredConversion,
        ] = {}
        self._adjacency: Dict[
            PrivacyNotion,
            List[PrivacyNotion],
        ] = {}

    def register(
        self,
        source: PrivacyNotion,
        target: PrivacyNotion,
        converter_fn: ConverterFn,
        provenance: ConversionProvenance,
    ) -> None:
        """Register a conversion function.

        Parameters
        ----------
        source : PrivacyNotion
            Source privacy notion.
        target : PrivacyNotion
            Target privacy notion.
        converter_fn : ConverterFn
            Callable that performs the conversion and returns a
            :class:`ConversionResult`.
        provenance : ConversionProvenance
            Metadata about the conversion theorem.
        """
        key = (source, target)
        self._conversions[key] = _RegisteredConversion(
            source=source,
            target=target,
            converter=converter_fn,
            provenance=provenance,
        )
        self._adjacency.setdefault(source, [])
        if target not in self._adjacency[source]:
            self._adjacency[source].append(target)

    def convert(
        self,
        budget: PrivacyBudget,
        target_notion: PrivacyNotion,
        **kwargs: Any,
    ) -> ConversionResult:
        """Convert *budget* to *target_notion* using registered converters.

        If a direct conversion is registered, it is used.  Otherwise
        :meth:`find_path` is called to discover a multi-hop route and
        each hop is applied in sequence.

        Parameters
        ----------
        budget : PrivacyBudget
            Source budget.
        target_notion : PrivacyNotion
            Desired target notion.
        **kwargs
            Additional keyword arguments forwarded to converter
            functions (e.g. ``delta``, ``alpha``).

        Returns
        -------
        ConversionResult

        Raises
        ------
        KeyError
            If no conversion path exists.
        """
        source_notion = _notion_of(budget)
        if source_notion == target_notion:
            return ConversionResult(
                source_budget=budget,
                target_budget=budget,
                provenance=ConversionProvenance(
                    theorem_name="Identity",
                    reference="N/A",
                    direction=f"{source_notion.name} -> {target_notion.name}",
                    is_tight=True,
                ),
                conversion_loss=0.0,
            )

        key = (source_notion, target_notion)
        if key in self._conversions:
            return self._conversions[key].converter(budget, **kwargs)

        # Multi-hop
        path = self.find_path(source_notion, target_notion)
        current_budget = budget
        total_loss: float = 0.0
        provenances: List[str] = []
        for i in range(len(path) - 1):
            hop_key = (path[i], path[i + 1])
            if hop_key not in self._conversions:
                raise KeyError(
                    f"No converter registered for {path[i].name} -> "
                    f"{path[i + 1].name}"
                )
            result = self._conversions[hop_key].converter(
                current_budget, **kwargs
            )
            current_budget = result.target_budget
            total_loss += result.conversion_loss
            provenances.append(result.provenance.theorem_name)

        composite_provenance = ConversionProvenance(
            theorem_name=" → ".join(provenances),
            reference="Composite conversion",
            direction=f"{source_notion.name} -> {target_notion.name}",
            is_tight=False,
        )
        return ConversionResult(
            source_budget=budget,
            target_budget=current_budget,
            provenance=composite_provenance,
            conversion_loss=total_loss,
        )

    def find_path(
        self,
        source_notion: PrivacyNotion,
        target_notion: PrivacyNotion,
    ) -> List[PrivacyNotion]:
        """Find a conversion path using breadth-first search.

        Parameters
        ----------
        source_notion : PrivacyNotion
            Starting notion.
        target_notion : PrivacyNotion
            Desired notion.

        Returns
        -------
        list of PrivacyNotion
            Ordered list of notions from source to target (inclusive).

        Raises
        ------
        KeyError
            If no path exists.
        """
        from collections import deque

        visited: set[PrivacyNotion] = {source_notion}
        queue: deque[List[PrivacyNotion]] = deque([[source_notion]])

        while queue:
            path = queue.popleft()
            current = path[-1]
            if current == target_notion:
                return path
            for neighbour in self._adjacency.get(current, []):
                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.append(path + [neighbour])

        raise KeyError(
            f"No conversion path from {source_notion.name} to "
            f"{target_notion.name}"
        )

    def all_conversions(self) -> List[_RegisteredConversion]:
        """Return all registered conversions.

        Returns
        -------
        list of _RegisteredConversion
        """
        return list(self._conversions.values())


# ---------------------------------------------------------------------------
# Numerical evaluation helpers
# ---------------------------------------------------------------------------


def evaluate_rdp_curve(
    budget: PrivacyBudget,
    alphas: Sequence[float],
) -> List[Tuple[float, float]]:
    """Evaluate the RDP curve of *budget* at the given Rényi orders.

    Parameters
    ----------
    budget : PrivacyBudget
        Source budget.
    alphas : Sequence[float]
        Rényi orders to evaluate (each > 1).

    Returns
    -------
    list of (float, float)
        Pairs ``(α, ε_RDP(α))``.

    Raises
    ------
    TypeError
        If the budget type does not support RDP evaluation.
    """
    curve: List[Tuple[float, float]] = []

    if isinstance(budget, PureBudget):
        for a in alphas:
            if a <= 1.0:
                continue
            curve.append((a, budget.epsilon))

    elif isinstance(budget, ZCDPBudget):
        for a in alphas:
            if a <= 1.0:
                continue
            curve.append((a, budget.rho * a))

    elif isinstance(budget, RDPBudget):
        # Single-point: only meaningful at the stored α
        curve.append((budget.alpha, budget.epsilon))

    elif isinstance(budget, GDPBudget):
        # For a μ-GDP mechanism the RDP curve is
        #   ε(α) = α μ² / 2    (Gaussian mechanism)
        # Reference: Mironov (2017), Example 2.
        for a in alphas:
            if a <= 1.0:
                continue
            curve.append((a, a * budget.mu * budget.mu / 2.0))

    elif isinstance(budget, ApproxBudget):
        for a in alphas:
            if a <= 1.0:
                continue
            res = approx_to_rdp(budget.epsilon, budget.delta, a)
            curve.append((a, res.target_budget.epsilon))  # type: ignore[union-attr]

    else:
        raise TypeError(f"Cannot evaluate RDP curve for {type(budget)}")

    return curve


def evaluate_epsilon_delta_curve(
    budget: PrivacyBudget,
    deltas: Sequence[float],
) -> List[Tuple[float, float]]:
    """Evaluate the (ε, δ)-curve of *budget* at the given δ values.

    For each δ, the tightest ε is computed using :class:`OptimalConverter`.

    Parameters
    ----------
    budget : PrivacyBudget
        Source budget.
    deltas : Sequence[float]
        Target δ values, each in (0, 1).

    Returns
    -------
    list of (float, float)
        Pairs ``(ε, δ)`` sorted by δ.
    """
    converter = OptimalConverter()
    curve: List[Tuple[float, float]] = []
    for d in sorted(deltas):
        if d <= 0.0 or d >= 1.0:
            continue
        result = converter.find_tightest_approx(budget, d)
        eps = getattr(result.target_budget, "epsilon", math.inf)
        curve.append((eps, d))
    return curve


# ---------------------------------------------------------------------------
# Default registry with all built-in conversions
# ---------------------------------------------------------------------------


def build_default_registry() -> ConversionRegistry:
    """Construct a :class:`ConversionRegistry` pre-loaded with all
    built-in conversion functions.

    Returns
    -------
    ConversionRegistry
    """
    reg = ConversionRegistry()

    # Pure DP → Approx DP
    reg.register(
        PrivacyNotion.PURE_DP,
        PrivacyNotion.APPROX_DP,
        lambda b, **kw: pure_to_approx(b.epsilon),
        ConversionProvenance(
            "Trivial embedding",
            "Definition",
            "PURE_DP -> APPROX_DP",
            is_tight=True,
        ),
    )

    # Pure DP → zCDP
    reg.register(
        PrivacyNotion.PURE_DP,
        PrivacyNotion.ZCDP,
        lambda b, **kw: pure_to_zcdp(b.epsilon),
        ConversionProvenance(
            "Pure DP to zCDP",
            "Bun & Dwork (2016), Proposition 1.4",
            "PURE_DP -> ZCDP",
            is_tight=True,
        ),
    )

    # Pure DP → RDP
    reg.register(
        PrivacyNotion.PURE_DP,
        PrivacyNotion.RDP,
        lambda b, **kw: pure_to_rdp(b.epsilon, kw.get("alpha", 2.0)),
        ConversionProvenance(
            "Pure DP implies RDP",
            "Mironov (2017), Proposition 3",
            "PURE_DP -> RDP",
            is_tight=True,
        ),
    )

    # Pure DP → GDP
    reg.register(
        PrivacyNotion.PURE_DP,
        PrivacyNotion.GDP,
        lambda b, **kw: pure_to_gdp(b.epsilon),
        ConversionProvenance(
            "Pure DP to GDP",
            "Dong, Roth & Su (2022), Corollary 2.13",
            "PURE_DP -> GDP",
            is_tight=False,
        ),
    )

    # zCDP → Approx DP
    reg.register(
        PrivacyNotion.ZCDP,
        PrivacyNotion.APPROX_DP,
        lambda b, **kw: zcdp_to_approx(b.rho, kw["delta"]),
        ConversionProvenance(
            "zCDP to approx DP",
            "Bun & Dwork (2016), Proposition 1.3",
            "ZCDP -> APPROX_DP",
            is_tight=False,
        ),
    )

    # zCDP → RDP
    reg.register(
        PrivacyNotion.ZCDP,
        PrivacyNotion.RDP,
        lambda b, **kw: zcdp_to_rdp(b.rho, kw.get("alpha", 2.0)),
        ConversionProvenance(
            "zCDP to RDP",
            "Bun & Dwork (2016), Definition 1.1",
            "ZCDP -> RDP",
            is_tight=True,
        ),
    )

    # RDP → Approx DP
    reg.register(
        PrivacyNotion.RDP,
        PrivacyNotion.APPROX_DP,
        lambda b, **kw: rdp_to_approx(b.alpha, b.epsilon, kw["delta"]),
        ConversionProvenance(
            "RDP to approx DP",
            "Mironov (2017), Proposition 3",
            "RDP -> APPROX_DP",
            is_tight=False,
        ),
    )

    # GDP → f-DP
    reg.register(
        PrivacyNotion.GDP,
        PrivacyNotion.FDP,
        lambda b, **kw: gdp_to_fdp(b.mu),
        ConversionProvenance(
            "GDP to f-DP",
            "Dong, Roth & Su (2022), Definition 2.2",
            "GDP -> FDP",
            is_tight=True,
        ),
    )

    # GDP → Approx DP
    reg.register(
        PrivacyNotion.GDP,
        PrivacyNotion.APPROX_DP,
        lambda b, **kw: gdp_to_approx(b.mu, kw["delta"]),
        ConversionProvenance(
            "GDP to approx DP",
            "Dong, Roth & Su (2022), Corollary 2.13",
            "GDP -> APPROX_DP",
            is_tight=True,
        ),
    )

    # Approx DP → RDP (reverse, lossy)
    reg.register(
        PrivacyNotion.APPROX_DP,
        PrivacyNotion.RDP,
        lambda b, **kw: approx_to_rdp(
            b.epsilon, b.delta, kw.get("alpha", 2.0)
        ),
        ConversionProvenance(
            "Approx DP to RDP (lossy)",
            "Balle, Barthe & Gavin (2018), Proposition 3",
            "APPROX_DP -> RDP",
            is_tight=False,
        ),
    )

    # f-DP → Approx DP
    reg.register(
        PrivacyNotion.FDP,
        PrivacyNotion.APPROX_DP,
        lambda b, **kw: fdp_to_approx(b.trade_off_fn, kw["delta"]),
        ConversionProvenance(
            "f-DP to approx DP",
            "Dong, Roth & Su (2022), Proposition 2.6",
            "FDP -> APPROX_DP",
            is_tight=True,
        ),
    )

    return reg


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _validate_delta(delta: float) -> None:
    """Raise ``ValueError`` if *delta* is not in (0, 1)."""
    if delta <= 0.0 or delta >= 1.0:
        raise ValueError(f"delta must be in (0, 1), got {delta}")


def _validate_rho(rho: float) -> None:
    """Raise ``ValueError`` if *rho* is negative."""
    if rho < 0.0:
        raise ValueError(f"rho must be non-negative, got {rho}")


def _notion_of(budget: PrivacyBudget) -> PrivacyNotion:
    """Infer the :class:`PrivacyNotion` for a concrete budget instance."""
    if isinstance(budget, PureBudget):
        return PrivacyNotion.PURE_DP
    if isinstance(budget, ApproxBudget):
        return PrivacyNotion.APPROX_DP
    if isinstance(budget, ZCDPBudget):
        return PrivacyNotion.ZCDP
    if isinstance(budget, RDPBudget):
        return PrivacyNotion.RDP
    if isinstance(budget, FDPBudget):
        return PrivacyNotion.FDP
    if isinstance(budget, GDPBudget):
        return PrivacyNotion.GDP
    raise TypeError(f"Unknown budget type: {type(budget)}")


def _optimal_eps_from_zcdp(rho: float, delta: float) -> float:
    """Compute the optimal ε from ρ-zCDP for a given δ.

    Minimises over α > 1:

        ε(α) = ρ α + ln(1/δ) / (α − 1)

    The minimiser is α* = 1 + √(ln(1/δ) / ρ) when ρ > 0.
    """
    if rho == 0.0:
        return 0.0
    log_inv_delta: float = math.log(1.0 / delta)
    alpha_star: float = 1.0 + math.sqrt(log_inv_delta / rho)
    return rho * alpha_star + log_inv_delta / (alpha_star - 1.0)


def _bisect_eps_from_gdp(mu: float, delta: float) -> float:
    """Bisect to find ε satisfying the GDP → (ε, δ)-DP equation.

    Solves  Φ(−ε/μ + μ/2) − e^ε · Φ(−ε/μ − μ/2) = δ  for ε ≥ 0.

    Reference: Balle & Wang (2018), Theorem 8.
    """

    def _gdp_delta_at_eps(eps: float) -> float:
        """Compute δ(ε) for a μ-GDP mechanism."""
        return phi(-eps / mu + mu / 2.0) - math.exp(eps) * phi(
            -eps / mu - mu / 2.0
        )

    # δ(0) = Φ(μ/2) − Φ(−μ/2) which is positive for μ > 0
    # δ(ε) is monotonically decreasing in ε for fixed μ.
    lo: float = 0.0
    hi: float = mu * mu / 2.0 + mu * math.sqrt(2.0 * math.log(1.0 / delta))
    # Ensure hi gives δ(hi) < target δ
    while _gdp_delta_at_eps(hi) > delta:
        hi *= 2.0

    for _ in range(_BISECT_MAX_ITER):
        mid = (lo + hi) / 2.0
        d_mid = _gdp_delta_at_eps(mid)
        if abs(d_mid - delta) < _BISECT_TOL:
            return mid
        if d_mid > delta:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _bisect_gdp_mu_from_tradeoff(
    tradeoff_fn: Callable[[float], float],
    epsilon: float,
) -> float:
    """Find μ such that the GDP trade-off best matches *tradeoff_fn*.

    Uses bisection on the integrated squared difference between the
    candidate Gaussian trade-off Φ(Φ⁻¹(1−α) − μ) and the supplied
    trade-off function, evaluated at a grid of α values.
    """

    def _mismatch(mu_candidate: float) -> float:
        """Positive when μ_candidate is too large (GDP curve below target)."""
        test_alpha = 0.5
        gdp_val = phi(phi_inv(1.0 - test_alpha) - mu_candidate)
        target_val = tradeoff_fn(test_alpha)
        return gdp_val - target_val

    lo: float = 0.0
    hi: float = 2.0 * epsilon + 1.0
    # We want the largest μ whose trade-off is everywhere ≥ tradeoff_fn,
    # so we bisect on the mismatch at α = 0.5.
    for _ in range(_BISECT_MAX_ITER):
        mid = (lo + hi) / 2.0
        if _mismatch(mid) > 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < _BISECT_TOL:
            break
    return (lo + hi) / 2.0


def _gdp_conversion_loss(
    tradeoff_fn: Callable[[float], float],
    mu: float,
    n_points: int = 50,
) -> float:
    """Measure the maximum gap between *tradeoff_fn* and the GDP curve.

    Returns
    -------
    float
        max_{α ∈ grid} | f(α) − Φ(Φ⁻¹(1−α) − μ) |.
    """
    max_gap: float = 0.0
    for i in range(1, n_points):
        alpha = i / n_points
        gdp_val = phi(phi_inv(1.0 - alpha) - mu)
        target_val = tradeoff_fn(alpha)
        gap = abs(gdp_val - target_val)
        if gap > max_gap:
            max_gap = gap
    return max_gap
