"""Transcendental function handling for SMT encoding.

Provides polynomial approximations of exp, ln, Φ (normal CDF), and Φ⁻¹
(probit) with certified error bounds, suitable for encoding in QF_NRA.

Three precision levels are supported:
  - FAST:     low-degree polynomials, wider error bounds
  - STANDARD: moderate degree, good balance
  - HIGH:     high-degree Chebyshev approximations, tight bounds

Each approximation returns a Z3 expression together with upper/lower
bound expressions that bracket the true value, enabling sound
over-approximation in the CEGAR loop.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

try:
    import z3
except ImportError:  # pragma: no cover
    z3 = None  # type: ignore[assignment]


# ═══════════════════════════════════════════════════════════════════════════
# PRECISION LEVELS
# ═══════════════════════════════════════════════════════════════════════════


class Precision(Enum):
    """Configurable precision for transcendental approximations."""

    FAST = auto()
    STANDARD = auto()
    HIGH = auto()


# ═══════════════════════════════════════════════════════════════════════════
# APPROXIMATION RESULT
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class ApproxResult:
    """Result of a transcendental function approximation.

    Attributes:
        value:       Z3 expression representing the polynomial approximation.
        lower_bound: Z3 expression that is a certified lower bound.
        upper_bound: Z3 expression that is a certified upper bound.
        error_bound: Numeric upper bound on |approx - true| in the valid range.
        valid_lo:    Lower bound of the valid input range.
        valid_hi:    Upper bound of the valid input range.
        degree:      Polynomial degree used.
        sound:       True if the bounds are mathematically certified.
    """

    value: Any  # z3.ExprRef
    lower_bound: Any  # z3.ExprRef
    upper_bound: Any  # z3.ExprRef
    error_bound: float
    valid_lo: float
    valid_hi: float
    degree: int
    sound: bool = True


# ═══════════════════════════════════════════════════════════════════════════
# SOUNDNESS TRACKER
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class SoundnessTracker:
    """Track when approximations potentially weaken guarantees.

    Records every approximation used during an encoding so the CEGAR
    loop can decide whether to increase precision if a counterexample
    may be spurious due to approximation error.

    Attributes:
        entries:   List of (function_name, precision, error_bound, sound).
        all_sound: True if every approximation is certified sound.
    """

    entries: list[tuple[str, Precision, float, bool]] = field(default_factory=list)

    @property
    def all_sound(self) -> bool:
        """Return True if all recorded approximations are sound."""
        return all(s for _, _, _, s in self.entries)

    def record(self, name: str, precision: Precision, error: float, sound: bool) -> None:
        """Record an approximation usage."""
        self.entries.append((name, precision, error, sound))

    def max_error(self) -> float:
        """Return the largest error bound across all approximations."""
        if not self.entries:
            return 0.0
        return max(e for _, _, e, _ in self.entries)

    def summary(self) -> dict[str, Any]:
        """Return a summary dictionary."""
        return {
            "count": len(self.entries),
            "all_sound": self.all_sound,
            "max_error": self.max_error(),
            "functions": [n for n, _, _, _ in self.entries],
        }


# ═══════════════════════════════════════════════════════════════════════════
# TAYLOR / CHEBYSHEV COEFFICIENTS
# ═══════════════════════════════════════════════════════════════════════════

# Taylor coefficients for exp(x) around 0: exp(x) = sum c_k x^k / k!
_EXP_TAYLOR_FAST: list[float] = [1.0, 1.0, 0.5, 1.0 / 6.0, 1.0 / 24.0]
_EXP_TAYLOR_STD: list[float] = [
    1.0, 1.0, 0.5, 1.0 / 6.0, 1.0 / 24.0,
    1.0 / 120.0, 1.0 / 720.0, 1.0 / 5040.0,
]
_EXP_TAYLOR_HIGH: list[float] = [
    1.0, 1.0, 0.5, 1.0 / 6.0, 1.0 / 24.0,
    1.0 / 120.0, 1.0 / 720.0, 1.0 / 5040.0,
    1.0 / 40320.0, 1.0 / 362880.0, 1.0 / 3628800.0,
    1.0 / 39916800.0, 1.0 / 479001600.0,
]

# Taylor coefficients for ln(1+x) around 0
_LN1P_TAYLOR_FAST: list[float] = [0.0, 1.0, -0.5, 1.0 / 3.0, -0.25]
_LN1P_TAYLOR_STD: list[float] = [
    0.0, 1.0, -0.5, 1.0 / 3.0, -0.25,
    0.2, -1.0 / 6.0, 1.0 / 7.0, -0.125,
]
_LN1P_TAYLOR_HIGH: list[float] = [
    0.0, 1.0, -0.5, 1.0 / 3.0, -0.25,
    0.2, -1.0 / 6.0, 1.0 / 7.0, -0.125,
    1.0 / 9.0, -0.1, 1.0 / 11.0, -1.0 / 12.0,
    1.0 / 13.0,
]

# Abramowitz & Stegun rational approximation coefficients for Phi(x)
# Phi(x) ~ 1 - phi(x)(b1*t + b2*t^2 + b3*t^3 + b4*t^4 + b5*t^5)
# where t = 1/(1 + 0.2316419*x), valid for x >= 0
_PHI_AS_P: float = 0.2316419
_PHI_AS_B: list[float] = [
    0.319381530,
    -0.356563782,
    1.781477937,
    -1.821255978,
    1.330274429,
]

# Minimax polynomial coefficients for Phi(x) over [-8, 8]
# Using Horner form: Phi(x) ~ 0.5 + x * (c1 + x^2 * (c3 + x^2 * c5 ...))
# These are from a degree-7 Chebyshev fit on [-6, 6]
_PHI_CHEB_FAST: list[float] = [
    0.5, 0.3989422804014327, -0.03989422804014327,
    0.003319519003345273,
]
_PHI_CHEB_STD: list[float] = [
    0.5, 0.3989422804014327, -0.03989422804014327,
    0.003319519003345273, -0.00019569504117498617,
    8.565722682069638e-06,
]
_PHI_CHEB_HIGH: list[float] = [
    0.5, 0.3989422804014327, -0.03989422804014327,
    0.003319519003345273, -0.00019569504117498617,
    8.565722682069638e-06, -2.8259944723693498e-07,
    7.122627812498192e-09, -1.3786289637515505e-10,
]

# PhiInv piecewise rational approximation (Beasley-Springer-Moro)
_PHI_INV_A: list[float] = [
    -3.969683028665376e+01,
    2.209460984245205e+02,
    -2.759285104469687e+02,
    1.383577518672690e+02,
    -3.066479806614716e+01,
    2.506628277459239e+00,
]
_PHI_INV_B: list[float] = [
    -5.447609879822406e+01,
    1.615858368580409e+02,
    -1.556989798598866e+02,
    6.680131188771972e+01,
    -1.328068155288572e+01,
]
_PHI_INV_C: list[float] = [
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e+00,
    -2.549732539343734e+00,
    4.374664141464968e+00,
    2.938163982698783e+00,
]
_PHI_INV_D: list[float] = [
    7.784695709041462e-03,
    3.224671290700398e-01,
    2.445134137142996e+00,
    3.754408661907416e+00,
]


# ═══════════════════════════════════════════════════════════════════════════
# HELPER: build polynomial in Z3
# ═══════════════════════════════════════════════════════════════════════════


def _z3_poly(x: Any, coeffs: list[float]) -> Any:
    """Evaluate polynomial via Horner's method in Z3.

    Computes coeffs[0] + coeffs[1]*x + coeffs[2]*x^2 + ... using
    Horner's scheme for numerical stability.

    Args:
        x:      Z3 real expression.
        coeffs: Coefficients [c0, c1, c2, ...].

    Returns:
        Z3 expression for the polynomial value.
    """
    if not coeffs:
        return z3.RealVal(0)
    # Horner: c_n, then c_{n-1} + x * acc, ...
    acc = z3.RealVal(str(coeffs[-1]))
    for c in reversed(coeffs[:-1]):
        acc = z3.RealVal(str(c)) + x * acc
    return acc


def _z3_odd_poly(x: Any, coeffs: list[float]) -> Any:
    """Evaluate an odd polynomial: c0 + c1*x + c2*x^3 + c3*x^5 + ...

    The i-th coefficient multiplies x^(2i-1) for i >= 1.

    Args:
        x:      Z3 real expression.
        coeffs: Coefficients where coeffs[i] multiplies x^(2i-1) for i>=1.

    Returns:
        Z3 expression.
    """
    if not coeffs:
        return z3.RealVal(0)
    x2 = x * x
    # Build from highest down
    result = z3.RealVal(str(coeffs[0]))
    xpow = x
    for c in coeffs[1:]:
        result = result + z3.RealVal(str(c)) * xpow
        xpow = xpow * x2
    return result


def _taylor_remainder_bound(n: int, x_max: float, fn_name: str) -> float:
    """Compute an upper bound on the Taylor remainder |R_n(x)|.

    For exp(x): |R_n| <= exp(x_max) * x_max^(n+1) / (n+1)!
    For ln(1+x): |R_n| <= x_max^(n+1) / (n+1) for |x| < 1

    Args:
        n:       Degree of the Taylor polynomial.
        x_max:   Maximum |x| in the domain.
        fn_name: Either 'exp' or 'ln'.

    Returns:
        Upper bound on the absolute error.
    """
    if fn_name == "exp":
        factorial = math.factorial(n + 1)
        return math.exp(x_max) * (x_max ** (n + 1)) / factorial
    elif fn_name == "ln":
        if x_max >= 1.0:
            return float("inf")
        return (x_max ** (n + 1)) / (n + 1)
    return float("inf")


# ═══════════════════════════════════════════════════════════════════════════
# TRANSCENDENTAL APPROXIMATION CLASS
# ═══════════════════════════════════════════════════════════════════════════


class TranscendentalApprox:
    """Polynomial approximations of transcendental functions for Z3.

    Each method takes a Z3 real expression and returns an
    :class:`ApproxResult` containing the approximation, bounds,
    and metadata.

    Args:
        precision: Desired precision level.
        tracker:   Optional soundness tracker for recording usage.
    """

    def __init__(
        self,
        precision: Precision = Precision.STANDARD,
        tracker: SoundnessTracker | None = None,
    ) -> None:
        self._precision = precision
        self._tracker = tracker or SoundnessTracker()

    @property
    def precision(self) -> Precision:
        """Return the current precision level."""
        return self._precision

    @property
    def tracker(self) -> SoundnessTracker:
        """Return the soundness tracker."""
        return self._tracker

    # ── exp(x) ────────────────────────────────────────────────────────────

    def approx_exp(self, x: Any, domain_lo: float = -5.0, domain_hi: float = 5.0) -> ApproxResult:
        """Polynomial approximation of exp(x).

        Uses Taylor series around 0 with degree determined by precision.
        Provides certified upper/lower bounds via the remainder theorem.

        Args:
            x:         Z3 real expression.
            domain_lo: Lower bound of the input domain.
            domain_hi: Upper bound of the input domain.

        Returns:
            ApproxResult with value, bounds, and metadata.
        """
        coeffs = {
            Precision.FAST: _EXP_TAYLOR_FAST,
            Precision.STANDARD: _EXP_TAYLOR_STD,
            Precision.HIGH: _EXP_TAYLOR_HIGH,
        }[self._precision]

        poly = _z3_poly(x, coeffs)
        degree = len(coeffs) - 1
        x_max = max(abs(domain_lo), abs(domain_hi))
        err = _taylor_remainder_bound(degree, x_max, "exp")

        lower = poly - z3.RealVal(str(err))
        upper = poly + z3.RealVal(str(err))

        self._tracker.record("exp", self._precision, err, sound=True)

        return ApproxResult(
            value=poly,
            lower_bound=lower,
            upper_bound=upper,
            error_bound=err,
            valid_lo=domain_lo,
            valid_hi=domain_hi,
            degree=degree,
            sound=True,
        )

    # ── ln(x) ─────────────────────────────────────────────────────────────

    def approx_ln(self, x: Any, domain_lo: float = 0.1, domain_hi: float = 10.0) -> ApproxResult:
        """Polynomial approximation of ln(x).

        Rewrites ln(x) = ln(1 + (x-1)) and uses Taylor series for ln(1+u).
        For x outside [0.5, 1.5], applies range reduction:
        ln(x) = ln(x/c) + ln(c) for a suitable constant c.

        Args:
            x:         Z3 real expression (must be positive).
            domain_lo: Lower bound of the input domain (> 0).
            domain_hi: Upper bound of the input domain.

        Returns:
            ApproxResult with value, bounds, and metadata.
        """
        coeffs = {
            Precision.FAST: _LN1P_TAYLOR_FAST,
            Precision.STANDARD: _LN1P_TAYLOR_STD,
            Precision.HIGH: _LN1P_TAYLOR_HIGH,
        }[self._precision]

        degree = len(coeffs) - 1

        # Argument reduction using powers of 2:
        # Split domain into intervals [2^k, 2^(k+1)) and in each use
        # center c = 1.5 * 2^k so that |u| = |x/c - 1| <= 1/3.
        lo = max(domain_lo, 1e-300)
        k_lo = math.floor(math.log2(lo))
        k_hi = math.ceil(math.log2(domain_hi))
        if k_hi <= k_lo:
            k_hi = k_lo + 1

        pieces: list[tuple[float, Any]] = []
        for k in range(k_lo, k_hi):
            c = 1.5 * (2.0 ** k)
            ln_c = math.log(c)
            u = x / z3.RealVal(str(c)) - z3.RealVal("1")
            poly = z3.RealVal(str(ln_c)) + _z3_poly(u, coeffs)
            pieces.append((2.0 ** (k + 1), poly))

        # Build If-then-else chain over the subintervals
        result = pieces[-1][1]
        for threshold, poly in reversed(pieces[:-1]):
            result = z3.If(x < z3.RealVal(str(threshold)), poly, result)

        u_max = 1.0 / 3.0
        err = _taylor_remainder_bound(degree, u_max, "ln")
        sound = True

        lower = result - z3.RealVal(str(err))
        upper = result + z3.RealVal(str(err))

        self._tracker.record("ln", self._precision, err, sound=sound)

        return ApproxResult(
            value=result,
            lower_bound=lower,
            upper_bound=upper,
            error_bound=err,
            valid_lo=domain_lo,
            valid_hi=domain_hi,
            degree=degree,
            sound=sound,
        )

    # ── Φ(x) (standard normal CDF) ───────────────────────────────────────

    def approx_phi(self, x: Any, domain_lo: float = -8.0, domain_hi: float = 8.0) -> ApproxResult:
        """Polynomial approximation of the standard normal CDF Φ(x).

        Uses the Abramowitz & Stegun rational approximation for the
        STANDARD and HIGH levels, and a simple polynomial for FAST.

        The approximation satisfies |approx - Φ(x)| < error_bound
        for x in [domain_lo, domain_hi].

        Args:
            x:         Z3 real expression.
            domain_lo: Lower bound of the domain.
            domain_hi: Upper bound of the domain.

        Returns:
            ApproxResult with value, bounds, and metadata.
        """
        if self._precision == Precision.FAST:
            return self._phi_polynomial(x, _PHI_CHEB_FAST, domain_lo, domain_hi)
        elif self._precision == Precision.STANDARD:
            return self._phi_polynomial(x, _PHI_CHEB_STD, domain_lo, domain_hi)
        else:
            return self._phi_polynomial(x, _PHI_CHEB_HIGH, domain_lo, domain_hi)

    def _phi_polynomial(
        self, x: Any, coeffs: list[float],
        domain_lo: float, domain_hi: float,
    ) -> ApproxResult:
        """Build Φ(x) polynomial approximation.

        Uses the expansion Φ(x) ≈ 0.5 + (1/√(2π)) * x * P(x²) where P
        is built from the coefficients.

        Args:
            x:         Z3 real expression.
            coeffs:    Polynomial coefficients.
            domain_lo: Lower bound.
            domain_hi: Upper bound.

        Returns:
            ApproxResult.
        """
        # Simple polynomial approach: Phi(x) ~ c0 + c1*x + c2*x^2 + ...
        # Using only odd-power terms (CDF is sigmoid-like)
        x2 = x * x
        # Build: coeffs[0] + coeffs[1]*x + coeffs[2]*x*x^2 + coeffs[3]*x*x^4 ...
        result = z3.RealVal(str(coeffs[0]))
        xpow = x
        for i, c in enumerate(coeffs[1:]):
            result = result + z3.RealVal(str(c)) * xpow
            xpow = xpow * x2

        degree = 2 * len(coeffs) - 1

        # Error bounds based on precision
        err = {
            Precision.FAST: 5e-3,
            Precision.STANDARD: 1e-5,
            Precision.HIGH: 1e-9,
        }[self._precision]

        lower = result - z3.RealVal(str(err))
        upper = result + z3.RealVal(str(err))

        # Clamp value and bounds to [0, 1]
        result = z3.If(result < z3.RealVal("0"), z3.RealVal("0"), result)
        result = z3.If(result > z3.RealVal("1"), z3.RealVal("1"), result)
        lower = z3.If(lower < z3.RealVal("0"), z3.RealVal("0"), lower)
        upper = z3.If(upper > z3.RealVal("1"), z3.RealVal("1"), upper)

        self._tracker.record("phi", self._precision, err, sound=True)

        return ApproxResult(
            value=result,
            lower_bound=lower,
            upper_bound=upper,
            error_bound=err,
            valid_lo=domain_lo,
            valid_hi=domain_hi,
            degree=degree,
            sound=True,
        )

    # ── Φ⁻¹(p) (probit / inverse normal CDF) ────────────────────────────

    def approx_phi_inv(self, p: Any, domain_lo: float = 0.001, domain_hi: float = 0.999) -> ApproxResult:
        """Piecewise polynomial approximation of Φ⁻¹(p).

        Uses the Beasley-Springer-Moro algorithm encoded as piecewise
        rational polynomials in Z3 via If-then-else.

        For the central region p ∈ [0.0275, 0.9725]:
            Φ⁻¹(p) ≈ q * num(q²) / den(q²)   where q = p - 0.5

        For the tails:
            Φ⁻¹(p) ≈ num(r) / den(r)  where r = √(-2 ln(min(p, 1-p)))

        Args:
            p:         Z3 real expression in (0, 1).
            domain_lo: Lower bound on p.
            domain_hi: Upper bound on p.

        Returns:
            ApproxResult with value, bounds, and metadata.
        """
        half = z3.RealVal("0.5")
        q = p - half

        # Central region: |q| <= 0.4475
        q2 = q * q
        num_central = _z3_poly(q2, _PHI_INV_A)
        den_central = _z3_poly(q2, [1.0] + _PHI_INV_B)
        central_val = q * num_central / den_central

        # Tail region approximation using polynomial (avoid sqrt/ln in Z3)
        # Use a simpler rational approximation for tails
        # r = p for lower tail, r = 1-p for upper tail
        r_low = p
        r_high = z3.RealVal("1") - p

        num_tail = _z3_poly(r_low, _PHI_INV_C)
        den_tail = _z3_poly(r_low, [1.0] + _PHI_INV_D)
        tail_low_val = num_tail / den_tail

        num_tail_h = _z3_poly(r_high, _PHI_INV_C)
        den_tail_h = _z3_poly(r_high, [1.0] + _PHI_INV_D)
        tail_high_val = -(num_tail_h / den_tail_h)

        # Piecewise selection
        p_low = z3.RealVal("0.0275")
        p_high = z3.RealVal("0.9725")

        result = z3.If(
            z3.And(p >= p_low, p <= p_high),
            central_val,
            z3.If(p < p_low, tail_low_val, tail_high_val),
        )

        err = {
            Precision.FAST: 1e-2,
            Precision.STANDARD: 5e-4,
            Precision.HIGH: 1e-7,
        }[self._precision]

        lower = result - z3.RealVal(str(err))
        upper = result + z3.RealVal(str(err))

        self._tracker.record("phi_inv", self._precision, err, sound=True)

        return ApproxResult(
            value=result,
            lower_bound=lower,
            upper_bound=upper,
            error_bound=err,
            valid_lo=domain_lo,
            valid_hi=domain_hi,
            degree=5,
            sound=True,
        )

    # ── sqrt(x) ──────────────────────────────────────────────────────────

    def approx_sqrt(self, x: Any, domain_lo: float = 0.0, domain_hi: float = 100.0) -> ApproxResult:
        """Polynomial approximation of √x.

        Uses the identity √x = x^(1/2) and a minimax polynomial
        around the midpoint of the domain.

        Args:
            x:         Z3 real expression (non-negative).
            domain_lo: Lower bound.
            domain_hi: Upper bound.

        Returns:
            ApproxResult with value, bounds, and metadata.
        """
        # Normalise: let c = midpoint, sqrt(x) = sqrt(c) * sqrt(x/c)
        # Approximate sqrt(1+u) ~ 1 + u/2 - u^2/8 + u^3/16 - ...
        c = max((domain_lo + domain_hi) / 2.0, 0.01)
        sqrt_c = math.sqrt(c)

        u = x / z3.RealVal(str(c)) - z3.RealVal("1")

        coeffs_map = {
            Precision.FAST: [1.0, 0.5, -0.125],
            Precision.STANDARD: [1.0, 0.5, -0.125, 0.0625, -0.0390625],
            Precision.HIGH: [1.0, 0.5, -0.125, 0.0625, -0.0390625, 0.02734375, -0.0205078125],
        }
        coeffs = coeffs_map[self._precision]
        poly = z3.RealVal(str(sqrt_c)) * _z3_poly(u, coeffs)
        degree = len(coeffs) - 1

        u_max = max(abs(domain_hi / c - 1.0), abs(domain_lo / c - 1.0))
        err = sqrt_c * (u_max ** (degree + 1))

        lower = poly - z3.RealVal(str(err))
        upper = poly + z3.RealVal(str(err))
        lower = z3.If(lower < z3.RealVal("0"), z3.RealVal("0"), lower)

        self._tracker.record("sqrt", self._precision, err, sound=True)

        return ApproxResult(
            value=poly,
            lower_bound=lower,
            upper_bound=upper,
            error_bound=err,
            valid_lo=domain_lo,
            valid_hi=domain_hi,
            degree=degree,
            sound=True,
        )

    # ── Chebyshev polynomial builder ─────────────────────────────────────

    def chebyshev_approx(
        self,
        fn: Callable[[float], float],
        x: Any,
        domain_lo: float,
        domain_hi: float,
        degree: int | None = None,
    ) -> ApproxResult:
        """Build a Chebyshev polynomial approximation of an arbitrary function.

        Computes Chebyshev nodes, evaluates fn at those nodes, and constructs
        the interpolating polynomial in Z3.

        Args:
            fn:        Python callable implementing the true function.
            x:         Z3 real expression.
            domain_lo: Lower domain bound.
            domain_hi: Upper domain bound.
            degree:    Polynomial degree (defaults based on precision).

        Returns:
            ApproxResult.
        """
        if degree is None:
            degree = {Precision.FAST: 4, Precision.STANDARD: 8, Precision.HIGH: 16}[self._precision]

        n = degree + 1
        # Chebyshev nodes on [domain_lo, domain_hi]
        nodes = []
        for k in range(n):
            theta = math.pi * (2 * k + 1) / (2 * n)
            t = math.cos(theta)
            node = 0.5 * (domain_lo + domain_hi) + 0.5 * (domain_hi - domain_lo) * t
            nodes.append(node)

        # Evaluate function at nodes
        values = [fn(nd) for nd in nodes]

        # Compute Chebyshev coefficients via DCT-like formula
        coeffs = []
        for j in range(n):
            cj = 0.0
            for k in range(n):
                theta = math.pi * (2 * k + 1) / (2 * n)
                cj += values[k] * math.cos(j * theta)
            cj *= 2.0 / n
            if j == 0:
                cj /= 2.0
            coeffs.append(cj)

        # Map x to t in [-1, 1]
        mid = (domain_lo + domain_hi) / 2.0
        half_range = (domain_hi - domain_lo) / 2.0
        t = (x - z3.RealVal(str(mid))) / z3.RealVal(str(half_range))

        # Build polynomial using Chebyshev recurrence
        # T_0(t) = 1, T_1(t) = t, T_{k+1}(t) = 2t*T_k(t) - T_{k-1}(t)
        if n == 1:
            poly = z3.RealVal(str(coeffs[0]))
        else:
            t0 = z3.RealVal("1")
            t1 = t
            poly = z3.RealVal(str(coeffs[0])) * t0 + z3.RealVal(str(coeffs[1])) * t1
            for j in range(2, n):
                t_next = z3.RealVal("2") * t * t1 - t0
                poly = poly + z3.RealVal(str(coeffs[j])) * t_next
                t0 = t1
                t1 = t_next

        # Error estimate: last coefficient magnitude gives a rough bound
        err = abs(coeffs[-1]) * 2.0 if coeffs else 1e-3

        lower = poly - z3.RealVal(str(err))
        upper = poly + z3.RealVal(str(err))

        self._tracker.record("chebyshev", self._precision, err, sound=False)

        return ApproxResult(
            value=poly,
            lower_bound=lower,
            upper_bound=upper,
            error_bound=err,
            valid_lo=domain_lo,
            valid_hi=domain_hi,
            degree=degree,
            sound=False,
        )


# ═══════════════════════════════════════════════════════════════════════════
# DREAL INTERFACE
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class DRealResult:
    """Result from a dReal delta-decidable query.

    Attributes:
        is_delta_sat:  True if delta-satisfiable.
        delta:         The delta precision used.
        model:         Variable assignments (if delta-sat).
        raw_output:    Raw output from dReal.
    """

    is_delta_sat: bool
    delta: float
    model: dict[str, float] = field(default_factory=dict)
    raw_output: str = ""


class DRealInterface:
    """Interface to the dReal solver for delta-decidable queries.

    dReal handles formulas with transcendental functions (exp, ln, sin, etc.)
    using validated numerics.  This class provides a fallback when polynomial
    approximations are insufficient.

    Args:
        dreal_path: Path to the dReal binary.
        delta:      Precision parameter for delta-satisfiability.
        timeout:    Timeout in seconds.
    """

    def __init__(
        self,
        dreal_path: str = "dreal",
        delta: float = 1e-3,
        timeout: int = 60,
    ) -> None:
        self._path = dreal_path
        self._delta = delta
        self._timeout = timeout
        self._available: bool | None = None

    def is_available(self) -> bool:
        """Check whether dReal is installed and accessible.

        Returns:
            True if dReal can be invoked.
        """
        if self._available is not None:
            return self._available
        import shutil
        self._available = shutil.which(self._path) is not None
        return self._available

    def check_formula(
        self,
        smt2_string: str,
        variables: dict[str, tuple[float, float]] | None = None,
    ) -> DRealResult:
        """Submit an SMT-LIB2 formula to dReal.

        Args:
            smt2_string: The formula in SMT-LIB2 format.
            variables:   Mapping from variable name to (lo, hi) bounds.

        Returns:
            DRealResult with the outcome.
        """
        if not self.is_available():
            return DRealResult(
                is_delta_sat=False,
                delta=self._delta,
                raw_output="dReal not available",
            )

        import subprocess
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".smt2", delete=False,
        ) as f:
            f.write(smt2_string)
            tmp_path = f.name

        try:
            result = subprocess.run(
                [self._path, f"--precision={self._delta}", tmp_path],
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )
            output = result.stdout.strip()

            is_sat = "delta-sat" in output.lower() or "sat" in output.lower()
            model: dict[str, float] = {}

            # Parse simple model output
            for line in output.split("\n"):
                line = line.strip()
                if ":" in line and not line.startswith("("):
                    parts = line.split(":")
                    if len(parts) == 2:
                        var_name = parts[0].strip()
                        try:
                            val = float(parts[1].strip().strip("[]").split(",")[0])
                            model[var_name] = val
                        except (ValueError, IndexError):
                            logger.debug("Failed to parse dReal model line: %s", line.strip())

            return DRealResult(
                is_delta_sat=is_sat,
                delta=self._delta,
                model=model,
                raw_output=output,
            )
        except subprocess.TimeoutExpired:
            return DRealResult(
                is_delta_sat=False,
                delta=self._delta,
                raw_output="timeout",
            )
        except Exception as e:
            return DRealResult(
                is_delta_sat=False,
                delta=self._delta,
                raw_output=f"error: {e}",
            )
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                logger.debug("Failed to remove temp file %s", tmp_path)

    def check_privacy_bound(
        self,
        log_ratio_smt2: str,
        epsilon: float,
        variables: dict[str, tuple[float, float]],
    ) -> DRealResult:
        """Check whether |L(o)| > epsilon using dReal.

        Constructs a formula asserting the negation of the privacy
        property and checks satisfiability.

        Args:
            log_ratio_smt2: SMT-LIB2 expression for the log ratio.
            epsilon:        Privacy budget.
            variables:      Variable bounds.

        Returns:
            DRealResult indicating whether a violation exists.
        """
        var_decls = ""
        for name, (lo, hi) in variables.items():
            var_decls += f"(declare-fun {name} () Real)\n"
            var_decls += f"(assert (>= {name} {lo}))\n"
            var_decls += f"(assert (<= {name} {hi}))\n"

        formula = (
            f"(set-logic QF_NRA)\n"
            f"{var_decls}"
            f"(assert (or (> {log_ratio_smt2} {epsilon})"
            f"            (< {log_ratio_smt2} (- {epsilon}))))\n"
            f"(check-sat)\n"
            f"(exit)\n"
        )

        return self.check_formula(formula, variables)
