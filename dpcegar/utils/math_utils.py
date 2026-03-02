"""Mathematical utilities for differential privacy verification.

This module provides:
  - Interval arithmetic with correct rounding
  - Polynomial approximations of the normal CDF (Φ) and its inverse
  - Taylor series computation with certified remainder bounds
  - Symbolic simplification helpers for SMT pre-processing
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import reduce
from typing import Callable, Sequence


# ═══════════════════════════════════════════════════════════════════════════
# 1.  INTERVAL ARITHMETIC
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class Interval:
    """Closed interval [lo, hi] with conservative floating-point rounding.

    All arithmetic operations widen the result to account for IEEE-754
    rounding, ensuring that the true mathematical result is always
    contained within the returned interval.

    Examples::

        >>> Interval(1, 2) + Interval(3, 4)
        Interval(lo=4, hi=6)
        >>> Interval(-1, 1).contains(0.5)
        True
    """

    lo: float
    hi: float

    def __post_init__(self) -> None:
        if self.lo > self.hi:
            raise ValueError(f"Empty interval: [{self.lo}, {self.hi}]")

    # -- Predicates --------------------------------------------------------

    def contains(self, x: float) -> bool:
        """Return True if *x* is inside the interval."""
        return self.lo <= x <= self.hi

    def overlaps(self, other: Interval) -> bool:
        """Return True if the two intervals share at least one point."""
        return self.lo <= other.hi and other.lo <= self.hi

    def is_subset_of(self, other: Interval) -> bool:
        """Return True if *self* ⊆ *other*."""
        return other.lo <= self.lo and self.hi <= other.hi

    @property
    def width(self) -> float:
        """Width of the interval."""
        return self.hi - self.lo

    @property
    def midpoint(self) -> float:
        """Centre of the interval."""
        return (self.lo + self.hi) / 2.0

    @property
    def is_point(self) -> bool:
        """True when lo == hi."""
        return self.lo == self.hi

    @property
    def is_positive(self) -> bool:
        """True when the entire interval is ≥ 0."""
        return self.lo >= 0.0

    @property
    def is_negative(self) -> bool:
        """True when the entire interval is ≤ 0."""
        return self.hi <= 0.0

    # -- Factories ---------------------------------------------------------

    @classmethod
    def point(cls, v: float) -> Interval:
        """Create a degenerate interval [v, v]."""
        return cls(v, v)

    @classmethod
    def entire(cls) -> Interval:
        """The interval (-∞, +∞)."""
        return cls(-math.inf, math.inf)

    @classmethod
    def non_negative(cls) -> Interval:
        """[0, +∞)."""
        return cls(0.0, math.inf)

    @classmethod
    def hull(cls, intervals: Sequence[Interval]) -> Interval:
        """Smallest interval containing all given intervals."""
        lo = min(iv.lo for iv in intervals)
        hi = max(iv.hi for iv in intervals)
        return cls(lo, hi)

    # -- Arithmetic (conservative rounding) --------------------------------

    @staticmethod
    def _widen(lo: float, hi: float) -> tuple[float, float]:
        """Apply a tiny widening to guard against FP rounding errors."""
        eps = max(abs(lo), abs(hi), 1.0) * 1e-15
        return lo - eps, hi + eps

    def __add__(self, other: Interval | float) -> Interval:
        if isinstance(other, (int, float)):
            other = Interval.point(float(other))
        lo, hi = self._widen(self.lo + other.lo, self.hi + other.hi)
        return Interval(lo, hi)

    def __radd__(self, other: float) -> Interval:
        return self.__add__(other)

    def __neg__(self) -> Interval:
        return Interval(-self.hi, -self.lo)

    def __sub__(self, other: Interval | float) -> Interval:
        if isinstance(other, (int, float)):
            other = Interval.point(float(other))
        return self + (-other)

    def __rsub__(self, other: float) -> Interval:
        return Interval.point(float(other)) - self

    def __mul__(self, other: Interval | float) -> Interval:
        if isinstance(other, (int, float)):
            other = Interval.point(float(other))
        candidates = [
            self.lo * other.lo,
            self.lo * other.hi,
            self.hi * other.lo,
            self.hi * other.hi,
        ]
        lo, hi = self._widen(min(candidates), max(candidates))
        return Interval(lo, hi)

    def __rmul__(self, other: float) -> Interval:
        return self.__mul__(other)

    def __truediv__(self, other: Interval | float) -> Interval:
        if isinstance(other, (int, float)):
            other = Interval.point(float(other))
        if other.lo <= 0.0 <= other.hi:
            return Interval.entire()
        return self * Interval(1.0 / other.hi, 1.0 / other.lo)

    def __pow__(self, n: int) -> Interval:
        """Raise the interval to a non-negative integer power."""
        if n == 0:
            return Interval.point(1.0)
        if n == 1:
            return Interval(self.lo, self.hi)
        if n % 2 == 0:
            if self.lo >= 0:
                lo, hi = self._widen(self.lo ** n, self.hi ** n)
            elif self.hi <= 0:
                lo, hi = self._widen(self.hi ** n, self.lo ** n)
            else:
                lo = 0.0
                hi = max(self.lo ** n, self.hi ** n)
                _, hi = self._widen(lo, hi)
            return Interval(lo, hi)
        else:
            lo, hi = self._widen(self.lo ** n, self.hi ** n)
            return Interval(lo, hi)

    # -- Monotone function lifting -----------------------------------------

    def apply_monotone_increasing(self, fn: Callable[[float], float]) -> Interval:
        """Apply a monotonically increasing function to both endpoints."""
        lo, hi = self._widen(fn(self.lo), fn(self.hi))
        return Interval(lo, hi)

    def apply_monotone_decreasing(self, fn: Callable[[float], float]) -> Interval:
        """Apply a monotonically decreasing function to both endpoints."""
        lo, hi = self._widen(fn(self.hi), fn(self.lo))
        return Interval(lo, hi)

    # -- Set operations ----------------------------------------------------

    def intersect(self, other: Interval) -> Interval | None:
        """Return the intersection, or None if disjoint."""
        lo = max(self.lo, other.lo)
        hi = min(self.hi, other.hi)
        if lo > hi:
            return None
        return Interval(lo, hi)

    def union_hull(self, other: Interval) -> Interval:
        """Return the smallest interval containing both."""
        return Interval(min(self.lo, other.lo), max(self.hi, other.hi))

    # -- String representation ---------------------------------------------

    def __str__(self) -> str:
        return f"[{self.lo}, {self.hi}]"

    def __repr__(self) -> str:
        return f"Interval(lo={self.lo}, hi={self.hi})"


# ═══════════════════════════════════════════════════════════════════════════
# 2.  NORMAL CDF (Φ) AND INVERSE (Φ⁻¹) APPROXIMATIONS
# ═══════════════════════════════════════════════════════════════════════════

# Hart's rational approximation coefficients for erfc
_P_COEFFS = (
    0.3275911,
    0.2548296,
    -0.2844967,
    1.4214137,
    -1.4531520,
    1.0614054,
)


def phi(x: float) -> float:
    """Standard normal CDF Φ(x) = P(Z ≤ x) for Z ~ N(0,1).

    Uses the Abramowitz & Stegun approximation (formula 7.1.26) with
    maximum absolute error < 1.5 × 10⁻⁷.

    Args:
        x: Real-valued argument.

    Returns:
        Φ(x) in [0, 1].
    """
    if x >= 0:
        return 0.5 * (1.0 + _erf_approx(x / math.sqrt(2.0)))
    return 1.0 - phi(-x)


def _erf_approx(x: float) -> float:
    """Approximation to the error function erf(x) for x ≥ 0."""
    t = 1.0 / (1.0 + 0.3275911 * x)
    poly = t * (
        0.254829592
        + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))
    )
    return 1.0 - poly * math.exp(-x * x)


def phi_inv(p: float) -> float:
    """Inverse normal CDF Φ⁻¹(p): the quantile function.

    Uses the rational approximation from Peter Acklam with maximum
    relative error < 1.15 × 10⁻⁹.

    Args:
        p: Probability in (0, 1).

    Returns:
        x such that Φ(x) ≈ p.

    Raises:
        ValueError: If *p* is not in (0, 1).
    """
    if p <= 0.0 or p >= 1.0:
        raise ValueError(f"phi_inv requires p in (0, 1), got {p}")

    # Coefficients for the rational approximation
    a = (
        -3.969683028665376e+01,
         2.209460984245205e+02,
        -2.759285104469687e+02,
         1.383577518672690e+02,
        -3.066479806614716e+01,
         2.506628277459239e+00,
    )
    b = (
        -5.447609879822406e+01,
         1.615858368580409e+02,
        -1.556989798598866e+02,
         6.680131188771972e+01,
        -1.328068155288572e+01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
         4.374664141464968e+00,
         2.938163982698783e+00,
    )
    d = (
         7.784695709041462e-03,
         3.224671290700398e-01,
         2.445134137142996e+00,
         3.754408661907416e+00,
    )

    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        x = (
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        x = (
            ((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]
        ) * q / (
            ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0
        )
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = -(
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)

    return x


def phi_interval(iv: Interval) -> Interval:
    """Lift Φ over an interval (monotone increasing)."""
    return iv.apply_monotone_increasing(phi)


def phi_inv_interval(iv: Interval) -> Interval:
    """Lift Φ⁻¹ over an interval (monotone increasing on (0,1))."""
    lo = max(iv.lo, 1e-15)
    hi = min(iv.hi, 1.0 - 1e-15)
    if lo > hi:
        return Interval.entire()
    return Interval(lo, hi).apply_monotone_increasing(phi_inv)


# ═══════════════════════════════════════════════════════════════════════════
# 3.  TAYLOR SERIES WITH CERTIFIED REMAINDER BOUNDS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class TaylorTerm:
    """A single term c · (x - a)ⁿ in a Taylor expansion."""
    coefficient: float
    degree: int
    center: float = 0.0

    def evaluate(self, x: float) -> float:
        """Evaluate this term at a point."""
        return self.coefficient * (x - self.center) ** self.degree

    def evaluate_interval(self, iv: Interval) -> Interval:
        """Evaluate this term over an interval."""
        shifted = iv - self.center
        powered = shifted ** self.degree
        return Interval.point(self.coefficient) * powered


@dataclass(slots=True)
class TaylorSeries:
    """Taylor polynomial with a certified Lagrange remainder bound.

    The remainder bound guarantees that for all x in a given interval::

        |f(x) - T_n(x)| ≤ remainder_bound

    where T_n is the Taylor polynomial of degree n.

    Attributes:
        terms:           List of Taylor terms.
        remainder_bound: Upper bound on the absolute error over the
                         domain of interest.
        domain:          Interval over which the bound is valid.
    """

    terms: list[TaylorTerm]
    remainder_bound: float = 0.0
    domain: Interval = field(default_factory=Interval.entire)

    @property
    def degree(self) -> int:
        """Degree of the polynomial."""
        return max((t.degree for t in self.terms), default=0)

    def evaluate(self, x: float) -> float:
        """Evaluate the polynomial at a point (no remainder)."""
        return sum(t.evaluate(x) for t in self.terms)

    def evaluate_with_bound(self, x: float) -> Interval:
        """Evaluate at a point, returning an interval including the remainder."""
        v = self.evaluate(x)
        return Interval(v - self.remainder_bound, v + self.remainder_bound)

    def evaluate_interval(self, iv: Interval) -> Interval:
        """Conservative interval evaluation of polynomial + remainder."""
        result = Interval.point(0.0)
        for t in self.terms:
            result = result + t.evaluate_interval(iv)
        return Interval(
            result.lo - self.remainder_bound,
            result.hi + self.remainder_bound,
        )


def taylor_exp(center: float, order: int, domain: Interval) -> TaylorSeries:
    """Taylor series for exp(x) around *center* with Lagrange remainder.

    The remainder bound is::

        |R_n(x)| ≤ exp(max(|lo|, |hi|)) · |x - center|^(n+1) / (n+1)!

    Args:
        center: Expansion point.
        order:  Polynomial degree.
        domain: Interval over which the bound is computed.

    Returns:
        A :class:`TaylorSeries` with certified remainder.
    """
    terms: list[TaylorTerm] = []
    factorial = 1.0
    for k in range(order + 1):
        if k > 0:
            factorial *= k
        coeff = math.exp(center) / factorial
        terms.append(TaylorTerm(coefficient=coeff, degree=k, center=center))

    max_abs = max(abs(domain.lo), abs(domain.hi))
    max_deviation = max(abs(domain.lo - center), abs(domain.hi - center))
    remainder = math.exp(max_abs) * max_deviation ** (order + 1) / math.factorial(order + 1)

    return TaylorSeries(terms=terms, remainder_bound=remainder, domain=domain)


def taylor_log(center: float, order: int, domain: Interval) -> TaylorSeries:
    """Taylor series for ln(x) around *center* > 0 with remainder bound.

    Args:
        center: Expansion point (must be > 0).
        order:  Polynomial degree.
        domain: Interval over which the bound is computed (must be > 0).

    Returns:
        A :class:`TaylorSeries` with certified remainder.
    """
    if center <= 0:
        raise ValueError(f"Taylor log requires center > 0, got {center}")
    if domain.lo <= 0:
        raise ValueError(f"Taylor log requires domain > 0, got {domain}")

    terms: list[TaylorTerm] = [
        TaylorTerm(coefficient=math.log(center), degree=0, center=center),
    ]
    for k in range(1, order + 1):
        sign = (-1) ** (k + 1)
        coeff = sign / (k * center ** k)
        terms.append(TaylorTerm(coefficient=coeff, degree=k, center=center))

    max_deviation = max(abs(domain.lo - center), abs(domain.hi - center))
    min_val = min(center, domain.lo)
    remainder = (max_deviation / min_val) ** (order + 1) / (order + 1)

    return TaylorSeries(terms=terms, remainder_bound=remainder, domain=domain)


# ═══════════════════════════════════════════════════════════════════════════
# 4.  SYMBOLIC SIMPLIFICATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def is_close(a: float, b: float, rel_tol: float = 1e-9, abs_tol: float = 1e-12) -> bool:
    """Approximate equality check for floating-point values."""
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


def safe_log(x: float) -> float:
    """Logarithm that returns -inf for 0 and raises for negative input."""
    if x < 0:
        raise ValueError(f"Cannot take log of negative number: {x}")
    if x == 0:
        return -math.inf
    return math.log(x)


def safe_exp(x: float, cap: float = 700.0) -> float:
    """Exponential capped to avoid overflow."""
    if x > cap:
        return math.inf
    if x < -cap:
        return 0.0
    return math.exp(x)


def log_sum_exp(values: Sequence[float]) -> float:
    """Numerically stable log-sum-exp: log(Σ exp(xᵢ)).

    Uses the standard trick of subtracting the maximum value before
    exponentiation to avoid overflow.
    """
    if not values:
        return -math.inf
    m = max(values)
    if math.isinf(m) and m < 0:
        return -math.inf
    return m + math.log(sum(math.exp(x - m) for x in values))


def kl_divergence_gaussians(mu1: float, sigma1: float, mu2: float, sigma2: float) -> float:
    """KL divergence D_KL(N(μ₁,σ₁²) ‖ N(μ₂,σ₂²)).

    Returns:
        The KL divergence in nats.
    """
    if sigma1 <= 0 or sigma2 <= 0:
        raise ValueError("Standard deviations must be positive")
    return (
        math.log(sigma2 / sigma1)
        + (sigma1 ** 2 + (mu1 - mu2) ** 2) / (2 * sigma2 ** 2)
        - 0.5
    )


def renyi_divergence_gaussians(
    mu1: float, sigma1: float, mu2: float, sigma2: float, alpha: float
) -> float:
    """Rényi divergence D_α(N(μ₁,σ₁²) ‖ N(μ₂,σ₂²)) for α > 0, α ≠ 1.

    Uses the closed-form expression for Gaussians.
    """
    if alpha <= 0:
        raise ValueError(f"Alpha must be > 0, got {alpha}")
    if abs(alpha - 1.0) < 1e-12:
        return kl_divergence_gaussians(mu1, sigma1, mu2, sigma2)

    s1_sq = sigma1 ** 2
    s2_sq = sigma2 ** 2
    sigma_alpha_sq = (1 - alpha) * s2_sq + alpha * s1_sq
    if sigma_alpha_sq <= 0:
        return math.inf

    term1 = alpha * (mu1 - mu2) ** 2 / (2 * sigma_alpha_sq)
    term2 = (1 / (2 * (alpha - 1))) * math.log(sigma_alpha_sq / (s1_sq ** alpha * s2_sq ** (1 - alpha)))

    return term1 - term2


def laplace_privacy_loss(b: float, sensitivity: float) -> float:
    """Privacy loss (ε) for the Laplace mechanism.

    Args:
        b:           Laplace scale parameter.
        sensitivity: ℓ₁-sensitivity of the query.

    Returns:
        ε = Δf / b.
    """
    if b <= 0:
        raise ValueError(f"Laplace scale must be positive, got {b}")
    return sensitivity / b


def gaussian_privacy_loss_approx_dp(
    sigma: float, sensitivity: float, delta: float
) -> float:
    """Privacy loss (ε) for the Gaussian mechanism under (ε,δ)-DP.

    Uses the analytic formula:
        ε = Δf / σ · √(2 ln(1.25/δ))

    Args:
        sigma:       Gaussian noise standard deviation.
        sensitivity: ℓ₂-sensitivity of the query.
        delta:       Failure probability.

    Returns:
        ε for (ε, δ)-differential privacy.
    """
    if sigma <= 0:
        raise ValueError(f"Sigma must be positive, got {sigma}")
    if delta <= 0 or delta >= 1:
        raise ValueError(f"Delta must be in (0,1), got {delta}")
    return (sensitivity / sigma) * math.sqrt(2 * math.log(1.25 / delta))


def gaussian_privacy_loss_zcdp(sigma: float, sensitivity: float) -> float:
    """Privacy loss (ρ) for the Gaussian mechanism under zCDP.

    Returns:
        ρ = Δf² / (2σ²).
    """
    if sigma <= 0:
        raise ValueError(f"Sigma must be positive, got {sigma}")
    return sensitivity ** 2 / (2 * sigma ** 2)


def zcdp_to_approx_dp(rho: float, delta: float) -> float:
    """Convert zCDP guarantee ρ to (ε, δ)-DP.

    Uses the conversion: ε = ρ + 2√(ρ ln(1/δ)).
    """
    if rho < 0:
        raise ValueError(f"Rho must be non-negative, got {rho}")
    if delta <= 0 or delta >= 1:
        raise ValueError(f"Delta must be in (0,1), got {delta}")
    return rho + 2 * math.sqrt(rho * math.log(1.0 / delta))


def rdp_to_approx_dp(alpha: float, rdp_eps: float, delta: float) -> float:
    """Convert RDP guarantee (α, ε_RDP) to (ε, δ)-DP.

    Uses: ε = ε_RDP - ln(δ) / (α - 1).
    """
    if alpha <= 1:
        raise ValueError(f"Alpha must be > 1, got {alpha}")
    if delta <= 0 or delta >= 1:
        raise ValueError(f"Delta must be in (0,1), got {delta}")
    return rdp_eps - math.log(delta) / (alpha - 1)


# ═══════════════════════════════════════════════════════════════════════════
# 5.  MISCELLANEOUS NUMERIC UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def clamp(x: float, lo: float, hi: float) -> float:
    """Clamp *x* to the interval [lo, hi]."""
    return max(lo, min(x, hi))


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation: a + t·(b - a)."""
    return a + t * (b - a)


def product(values: Sequence[float]) -> float:
    """Product of a sequence of floats."""
    return reduce(operator.mul, values, 1.0)


def harmonic_number(n: int) -> float:
    """Compute the n-th harmonic number H_n = Σ 1/k for k=1..n."""
    return sum(1.0 / k for k in range(1, n + 1))


def binomial_coefficient(n: int, k: int) -> int:
    """Compute C(n, k) using the multiplicative formula."""
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result
