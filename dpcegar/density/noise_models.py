"""Noise distribution models for density ratio computation.

Provides concrete :class:`NoiseModel` implementations for Laplace,
Gaussian, exponential mechanism, truncated, discrete, and mixture
distributions.  Each model offers density, log-density, log-ratio,
CDF, and sampling operations, as well as symbolic expression builders
for SMT encoding.

Classes
-------
NoiseModel            – abstract base
LaplaceNoise          – Laplace (double exponential)
GaussianNoise         – Gaussian (normal)
ExponentialMechNoise  – exponential mechanism (discrete, utility-based)
TruncatedLaplaceNoise – Laplace truncated to [lo, hi]
TruncatedGaussianNoise– Gaussian truncated to [lo, hi]
DiscreteGaussianNoise – discrete Gaussian (integer-valued)
DiscreteLaplaceNoise  – discrete Laplace (integer-valued)
MixtureNoise          – mixture of other noise models
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence, Tuple

from dpcegar.ir.types import (
    Abs,
    BinOp,
    BinOpKind,
    Const,
    Exp,
    IRType,
    Log,
    Sqrt,
    TypedExpr,
    Var,
)
from dpcegar.utils.math_utils import (
    Interval,
    phi as std_phi,
    phi_inv as std_phi_inv,
    safe_exp,
    safe_log,
)


# ═══════════════════════════════════════════════════════════════════════════
# BASE CLASS
# ═══════════════════════════════════════════════════════════════════════════


class NoiseModel(ABC):
    """Abstract base class for noise distribution models.

    Every subclass must implement the core operations needed for
    density ratio construction and numerical verification.
    """

    @abstractmethod
    def density(self, x: float, center: float, scale: float) -> float:
        """Evaluate the probability density (or mass) at *x*.

        Args:
            x:      Observation point.
            center: Location / centre parameter.
            scale:  Scale parameter.

        Returns:
            The density p(x | center, scale).
        """
        ...

    @abstractmethod
    def log_density(self, x: float, center: float, scale: float) -> float:
        """Evaluate the log-density at *x*.

        Args:
            x:      Observation point.
            center: Location / centre parameter.
            scale:  Scale parameter.

        Returns:
            ln p(x | center, scale).
        """
        ...

    @abstractmethod
    def log_ratio(
        self,
        x: float,
        center1: float,
        center2: float,
        scale: float,
    ) -> float:
        """Compute the log density ratio ln(p(x|c1,s) / p(x|c2,s)).

        This is the privacy loss at observation *x* for neighbouring
        datasets producing centres *center1* and *center2*.

        Args:
            x:       Observation point.
            center1: Centre for dataset d.
            center2: Centre for dataset d'.
            scale:   Shared scale parameter.

        Returns:
            The log density ratio.
        """
        ...

    @abstractmethod
    def cdf(self, x: float, center: float, scale: float) -> float:
        """Evaluate the cumulative distribution function at *x*.

        Args:
            x:      Evaluation point.
            center: Location parameter.
            scale:  Scale parameter.

        Returns:
            P(X ≤ x).
        """
        ...

    @abstractmethod
    def sample(self, center: float, scale: float, rng: random.Random | None = None) -> float:
        """Draw a single sample from the distribution.

        Args:
            center: Location parameter.
            scale:  Scale parameter.
            rng:    Optional random number generator.

        Returns:
            A random sample.
        """
        ...

    @abstractmethod
    def symbolic_log_ratio(
        self,
        obs: TypedExpr,
        center_d: TypedExpr,
        center_d_prime: TypedExpr,
        scale: TypedExpr,
    ) -> TypedExpr:
        """Build a symbolic expression for the log density ratio.

        Used by the density ratio builder to construct SMT-encodable
        privacy loss expressions.

        Args:
            obs:            Symbolic observation variable.
            center_d:       Symbolic centre for dataset d.
            center_d_prime: Symbolic centre for dataset d'.
            scale:          Symbolic scale parameter.

        Returns:
            A TypedExpr representing ln(p(obs|d)/p(obs|d')).
        """
        ...

    @abstractmethod
    def symbolic_log_density(
        self,
        obs: TypedExpr,
        center: TypedExpr,
        scale: TypedExpr,
    ) -> TypedExpr:
        """Build a symbolic log-density expression.

        Args:
            obs:    Symbolic observation.
            center: Symbolic centre.
            scale:  Symbolic scale.

        Returns:
            A TypedExpr representing ln p(obs | center, scale).
        """
        ...


# ═══════════════════════════════════════════════════════════════════════════
# LAPLACE NOISE
# ═══════════════════════════════════════════════════════════════════════════


class LaplaceNoise(NoiseModel):
    """Laplace (double exponential) noise model.

    Density: p(x | μ, b) = (1/(2b)) exp(-|x - μ| / b)
    """

    def density(self, x: float, center: float, scale: float) -> float:
        """Evaluate Laplace density at *x*."""
        if scale <= 0:
            raise ValueError(f"Laplace scale must be positive, got {scale}")
        return (1.0 / (2.0 * scale)) * math.exp(-abs(x - center) / scale)

    def log_density(self, x: float, center: float, scale: float) -> float:
        """Evaluate Laplace log-density at *x*."""
        if scale <= 0:
            raise ValueError(f"Laplace scale must be positive, got {scale}")
        return -math.log(2.0 * scale) - abs(x - center) / scale

    def log_ratio(
        self,
        x: float,
        center1: float,
        center2: float,
        scale: float,
    ) -> float:
        """Compute Laplace log density ratio.

        ln(p(x|c1)/p(x|c2)) = (|x - c2| - |x - c1|) / b
        """
        if scale <= 0:
            raise ValueError(f"Laplace scale must be positive, got {scale}")
        return (abs(x - center2) - abs(x - center1)) / scale

    def cdf(self, x: float, center: float, scale: float) -> float:
        """Evaluate Laplace CDF at *x*."""
        if scale <= 0:
            raise ValueError(f"Laplace scale must be positive, got {scale}")
        z = (x - center) / scale
        if z <= 0:
            return 0.5 * math.exp(z)
        return 1.0 - 0.5 * math.exp(-z)

    def sample(self, center: float, scale: float, rng: random.Random | None = None) -> float:
        """Draw a Laplace sample via inverse CDF."""
        r = rng or random.Random()
        u = r.random() - 0.5
        return center - scale * math.copysign(1.0, u) * math.log(1.0 - 2.0 * abs(u))

    def symbolic_log_ratio(
        self,
        obs: TypedExpr,
        center_d: TypedExpr,
        center_d_prime: TypedExpr,
        scale: TypedExpr,
    ) -> TypedExpr:
        """Symbolic Laplace log ratio: (|obs - c'| - |obs - c|) / b."""
        diff_d = BinOp(ty=IRType.REAL, op=BinOpKind.SUB, left=obs, right=center_d)
        diff_d_prime = BinOp(ty=IRType.REAL, op=BinOpKind.SUB, left=obs, right=center_d_prime)
        abs_d = Abs(ty=IRType.REAL, operand=diff_d)
        abs_d_prime = Abs(ty=IRType.REAL, operand=diff_d_prime)
        numerator = BinOp(ty=IRType.REAL, op=BinOpKind.SUB, left=abs_d_prime, right=abs_d)
        return BinOp(ty=IRType.REAL, op=BinOpKind.DIV, left=numerator, right=scale)

    def symbolic_log_density(
        self,
        obs: TypedExpr,
        center: TypedExpr,
        scale: TypedExpr,
    ) -> TypedExpr:
        """Symbolic Laplace log-density: -ln(2b) - |obs - μ| / b."""
        two_b = BinOp(ty=IRType.REAL, op=BinOpKind.MUL, left=Const.real(2.0), right=scale)
        log_norm = Log(ty=IRType.REAL, operand=two_b)
        neg_log_norm = BinOp(ty=IRType.REAL, op=BinOpKind.MUL, left=Const.real(-1.0), right=log_norm)
        diff = BinOp(ty=IRType.REAL, op=BinOpKind.SUB, left=obs, right=center)
        abs_diff = Abs(ty=IRType.REAL, operand=diff)
        ratio = BinOp(ty=IRType.REAL, op=BinOpKind.DIV, left=abs_diff, right=scale)
        return BinOp(ty=IRType.REAL, op=BinOpKind.SUB, left=neg_log_norm, right=ratio)

    def max_privacy_loss(self, sensitivity: float, scale: float) -> float:
        """Maximum privacy loss: Δf / b."""
        if scale <= 0:
            raise ValueError(f"Laplace scale must be positive, got {scale}")
        return sensitivity / scale


# ═══════════════════════════════════════════════════════════════════════════
# GAUSSIAN NOISE
# ═══════════════════════════════════════════════════════════════════════════


class GaussianNoise(NoiseModel):
    """Gaussian (normal) noise model.

    Density: p(x | μ, σ) = (1/(σ√(2π))) exp(-(x-μ)²/(2σ²))
    """

    _LOG_2PI = math.log(2.0 * math.pi)

    def density(self, x: float, center: float, scale: float) -> float:
        """Evaluate Gaussian density at *x*."""
        if scale <= 0:
            raise ValueError(f"Gaussian scale must be positive, got {scale}")
        z = (x - center) / scale
        return (1.0 / (scale * math.sqrt(2.0 * math.pi))) * math.exp(-0.5 * z * z)

    def log_density(self, x: float, center: float, scale: float) -> float:
        """Evaluate Gaussian log-density at *x*."""
        if scale <= 0:
            raise ValueError(f"Gaussian scale must be positive, got {scale}")
        z = (x - center) / scale
        return -0.5 * self._LOG_2PI - math.log(scale) - 0.5 * z * z

    def log_ratio(
        self,
        x: float,
        center1: float,
        center2: float,
        scale: float,
    ) -> float:
        """Compute Gaussian log density ratio.

        ln(p(x|c1,σ)/p(x|c2,σ)) = ((x-c2)² - (x-c1)²) / (2σ²)
        """
        if scale <= 0:
            raise ValueError(f"Gaussian scale must be positive, got {scale}")
        return ((x - center2) ** 2 - (x - center1) ** 2) / (2.0 * scale ** 2)

    def cdf(self, x: float, center: float, scale: float) -> float:
        """Evaluate Gaussian CDF via Φ."""
        if scale <= 0:
            raise ValueError(f"Gaussian scale must be positive, got {scale}")
        return std_phi((x - center) / scale)

    def sample(self, center: float, scale: float, rng: random.Random | None = None) -> float:
        """Draw a Gaussian sample via Box-Muller."""
        r = rng or random.Random()
        return r.gauss(center, scale)

    def symbolic_log_ratio(
        self,
        obs: TypedExpr,
        center_d: TypedExpr,
        center_d_prime: TypedExpr,
        scale: TypedExpr,
    ) -> TypedExpr:
        """Symbolic Gaussian log ratio: ((o-c')² - (o-c)²) / (2σ²)."""
        diff_d = BinOp(ty=IRType.REAL, op=BinOpKind.SUB, left=obs, right=center_d)
        diff_dp = BinOp(ty=IRType.REAL, op=BinOpKind.SUB, left=obs, right=center_d_prime)
        sq_d = BinOp(ty=IRType.REAL, op=BinOpKind.POW, left=diff_d, right=Const.real(2.0))
        sq_dp = BinOp(ty=IRType.REAL, op=BinOpKind.POW, left=diff_dp, right=Const.real(2.0))
        numer = BinOp(ty=IRType.REAL, op=BinOpKind.SUB, left=sq_dp, right=sq_d)
        sq_scale = BinOp(ty=IRType.REAL, op=BinOpKind.POW, left=scale, right=Const.real(2.0))
        denom = BinOp(ty=IRType.REAL, op=BinOpKind.MUL, left=Const.real(2.0), right=sq_scale)
        return BinOp(ty=IRType.REAL, op=BinOpKind.DIV, left=numer, right=denom)

    def symbolic_log_density(
        self,
        obs: TypedExpr,
        center: TypedExpr,
        scale: TypedExpr,
    ) -> TypedExpr:
        """Symbolic Gaussian log-density: -0.5*ln(2π) - ln(σ) - (x-μ)²/(2σ²)."""
        log_sigma = Log(ty=IRType.REAL, operand=scale)
        diff = BinOp(ty=IRType.REAL, op=BinOpKind.SUB, left=obs, right=center)
        sq_diff = BinOp(ty=IRType.REAL, op=BinOpKind.POW, left=diff, right=Const.real(2.0))
        sq_sigma = BinOp(ty=IRType.REAL, op=BinOpKind.POW, left=scale, right=Const.real(2.0))
        two_sq = BinOp(ty=IRType.REAL, op=BinOpKind.MUL, left=Const.real(2.0), right=sq_sigma)
        exp_term = BinOp(ty=IRType.REAL, op=BinOpKind.DIV, left=sq_diff, right=two_sq)

        const_part = Const.real(-0.5 * self._LOG_2PI)
        neg_log_sigma = BinOp(ty=IRType.REAL, op=BinOpKind.MUL, left=Const.real(-1.0), right=log_sigma)
        neg_exp = BinOp(ty=IRType.REAL, op=BinOpKind.MUL, left=Const.real(-1.0), right=exp_term)
        t1 = BinOp(ty=IRType.REAL, op=BinOpKind.ADD, left=const_part, right=neg_log_sigma)
        return BinOp(ty=IRType.REAL, op=BinOpKind.ADD, left=t1, right=neg_exp)

    def renyi_divergence(
        self, center1: float, center2: float, scale: float, alpha: float
    ) -> float:
        """Compute Rényi divergence of order α between two Gaussians with same σ.

        D_α(N(c1,σ²) ‖ N(c2,σ²)) = α·(c1-c2)² / (2σ²)
        """
        if scale <= 0:
            raise ValueError(f"Gaussian scale must be positive, got {scale}")
        return alpha * (center1 - center2) ** 2 / (2.0 * scale ** 2)

    def zcdp_rho(self, sensitivity: float, scale: float) -> float:
        """Compute zCDP parameter ρ = Δ²/(2σ²)."""
        if scale <= 0:
            raise ValueError(f"Gaussian scale must be positive, got {scale}")
        return sensitivity ** 2 / (2.0 * scale ** 2)

    def gdp_mu(self, sensitivity: float, scale: float) -> float:
        """Compute GDP parameter μ = Δ/σ."""
        if scale <= 0:
            raise ValueError(f"Gaussian scale must be positive, got {scale}")
        return sensitivity / scale


# ═══════════════════════════════════════════════════════════════════════════
# EXPONENTIAL MECHANISM NOISE
# ═══════════════════════════════════════════════════════════════════════════


class ExponentialMechNoise(NoiseModel):
    """Exponential mechanism noise model (discrete, utility-based).

    For a utility function u(d, r) and sensitivity Δu, the probability
    of selecting outcome r is proportional to exp(ε · u(d, r) / (2Δu)).
    """

    def density(self, x: float, center: float, scale: float) -> float:
        """Evaluate unnormalized density (proportional to exp(ε·u/(2Δu))).

        Here *center* plays the role of utility u(d, x) and *scale*
        is 2Δu/ε.
        """
        if scale <= 0:
            raise ValueError(f"Scale must be positive, got {scale}")
        return safe_exp(center / scale)

    def log_density(self, x: float, center: float, scale: float) -> float:
        """Log of the unnormalized density: u/scale."""
        if scale <= 0:
            raise ValueError(f"Scale must be positive, got {scale}")
        return center / scale

    def log_ratio(
        self,
        x: float,
        center1: float,
        center2: float,
        scale: float,
    ) -> float:
        """Compute log ratio: (u(d,r) - u(d',r)) / scale.

        Here center1 = u(d,r), center2 = u(d',r), scale = 2Δu/ε.
        The log ratio simplifies to ε·(u(d,r) - u(d',r)) / (2Δu).
        """
        if scale <= 0:
            raise ValueError(f"Scale must be positive, got {scale}")
        return (center1 - center2) / scale

    def cdf(self, x: float, center: float, scale: float) -> float:
        """CDF is not well-defined without the range; return 0.5."""
        return 0.5

    def sample(self, center: float, scale: float, rng: random.Random | None = None) -> float:
        """Not directly sampleable without the output space; return centre."""
        return center

    def symbolic_log_ratio(
        self,
        obs: TypedExpr,
        center_d: TypedExpr,
        center_d_prime: TypedExpr,
        scale: TypedExpr,
    ) -> TypedExpr:
        """Symbolic exp-mech log ratio: (u(d,r) - u(d',r)) / (2Δu/ε)."""
        numer = BinOp(ty=IRType.REAL, op=BinOpKind.SUB, left=center_d, right=center_d_prime)
        return BinOp(ty=IRType.REAL, op=BinOpKind.DIV, left=numer, right=scale)

    def symbolic_log_density(
        self,
        obs: TypedExpr,
        center: TypedExpr,
        scale: TypedExpr,
    ) -> TypedExpr:
        """Symbolic log-density: u / scale."""
        return BinOp(ty=IRType.REAL, op=BinOpKind.DIV, left=center, right=scale)


# ═══════════════════════════════════════════════════════════════════════════
# TRUNCATED NOISE MODELS
# ═══════════════════════════════════════════════════════════════════════════


class TruncatedLaplaceNoise(NoiseModel):
    """Laplace noise truncated to a bounded interval [lo, hi].

    The density is re-normalized: p_trunc(x) = p(x) / (CDF(hi) - CDF(lo)).
    """

    def __init__(self, lo: float = -math.inf, hi: float = math.inf) -> None:
        self._lo = lo
        self._hi = hi
        self._base = LaplaceNoise()

    def _normalization(self, center: float, scale: float) -> float:
        """Compute the normalizing constant CDF(hi) - CDF(lo)."""
        return self._base.cdf(self._hi, center, scale) - self._base.cdf(self._lo, center, scale)

    def density(self, x: float, center: float, scale: float) -> float:
        """Evaluate truncated Laplace density."""
        if x < self._lo or x > self._hi:
            return 0.0
        z = self._normalization(center, scale)
        if z <= 0:
            return 0.0
        return self._base.density(x, center, scale) / z

    def log_density(self, x: float, center: float, scale: float) -> float:
        """Evaluate truncated Laplace log-density."""
        if x < self._lo or x > self._hi:
            return -math.inf
        z = self._normalization(center, scale)
        if z <= 0:
            return -math.inf
        return self._base.log_density(x, center, scale) - math.log(z)

    def log_ratio(
        self, x: float, center1: float, center2: float, scale: float,
    ) -> float:
        """Log ratio for truncated Laplace."""
        if x < self._lo or x > self._hi:
            return 0.0
        z1 = self._normalization(center1, scale)
        z2 = self._normalization(center2, scale)
        if z1 <= 0 or z2 <= 0:
            return 0.0
        return self._base.log_ratio(x, center1, center2, scale) + math.log(z2) - math.log(z1)

    def cdf(self, x: float, center: float, scale: float) -> float:
        """Evaluate truncated Laplace CDF."""
        if x <= self._lo:
            return 0.0
        if x >= self._hi:
            return 1.0
        z = self._normalization(center, scale)
        if z <= 0:
            return 0.5
        return (self._base.cdf(x, center, scale) - self._base.cdf(self._lo, center, scale)) / z

    def sample(self, center: float, scale: float, rng: random.Random | None = None) -> float:
        """Draw a truncated Laplace sample via rejection."""
        r = rng or random.Random()
        for _ in range(10_000):
            s = self._base.sample(center, scale, r)
            if self._lo <= s <= self._hi:
                return s
        return (self._lo + self._hi) / 2.0

    def symbolic_log_ratio(
        self, obs: TypedExpr, center_d: TypedExpr, center_d_prime: TypedExpr, scale: TypedExpr,
    ) -> TypedExpr:
        """Delegate to base Laplace symbolic (ignoring truncation for SMT simplicity)."""
        return self._base.symbolic_log_ratio(obs, center_d, center_d_prime, scale)

    def symbolic_log_density(
        self, obs: TypedExpr, center: TypedExpr, scale: TypedExpr,
    ) -> TypedExpr:
        """Delegate to base Laplace symbolic."""
        return self._base.symbolic_log_density(obs, center, scale)


class TruncatedGaussianNoise(NoiseModel):
    """Gaussian noise truncated to a bounded interval [lo, hi]."""

    def __init__(self, lo: float = -math.inf, hi: float = math.inf) -> None:
        self._lo = lo
        self._hi = hi
        self._base = GaussianNoise()

    def _normalization(self, center: float, scale: float) -> float:
        """Normalizing constant Φ((hi-μ)/σ) - Φ((lo-μ)/σ)."""
        return std_phi((self._hi - center) / scale) - std_phi((self._lo - center) / scale)

    def density(self, x: float, center: float, scale: float) -> float:
        """Evaluate truncated Gaussian density."""
        if x < self._lo or x > self._hi:
            return 0.0
        z = self._normalization(center, scale)
        if z <= 0:
            return 0.0
        return self._base.density(x, center, scale) / z

    def log_density(self, x: float, center: float, scale: float) -> float:
        """Evaluate truncated Gaussian log-density."""
        if x < self._lo or x > self._hi:
            return -math.inf
        z = self._normalization(center, scale)
        if z <= 0:
            return -math.inf
        return self._base.log_density(x, center, scale) - math.log(z)

    def log_ratio(
        self, x: float, center1: float, center2: float, scale: float,
    ) -> float:
        """Log ratio for truncated Gaussian."""
        if x < self._lo or x > self._hi:
            return 0.0
        z1 = self._normalization(center1, scale)
        z2 = self._normalization(center2, scale)
        if z1 <= 0 or z2 <= 0:
            return 0.0
        return self._base.log_ratio(x, center1, center2, scale) + math.log(z2) - math.log(z1)

    def cdf(self, x: float, center: float, scale: float) -> float:
        """Evaluate truncated Gaussian CDF."""
        if x <= self._lo:
            return 0.0
        if x >= self._hi:
            return 1.0
        z = self._normalization(center, scale)
        if z <= 0:
            return 0.5
        return (self._base.cdf(x, center, scale) - self._base.cdf(self._lo, center, scale)) / z

    def sample(self, center: float, scale: float, rng: random.Random | None = None) -> float:
        """Draw a truncated Gaussian sample via rejection."""
        r = rng or random.Random()
        for _ in range(10_000):
            s = self._base.sample(center, scale, r)
            if self._lo <= s <= self._hi:
                return s
        return (self._lo + self._hi) / 2.0

    def symbolic_log_ratio(
        self, obs: TypedExpr, center_d: TypedExpr, center_d_prime: TypedExpr, scale: TypedExpr,
    ) -> TypedExpr:
        """Delegate to base Gaussian symbolic."""
        return self._base.symbolic_log_ratio(obs, center_d, center_d_prime, scale)

    def symbolic_log_density(
        self, obs: TypedExpr, center: TypedExpr, scale: TypedExpr,
    ) -> TypedExpr:
        """Delegate to base Gaussian symbolic."""
        return self._base.symbolic_log_density(obs, center, scale)


# ═══════════════════════════════════════════════════════════════════════════
# DISCRETE NOISE MODELS
# ═══════════════════════════════════════════════════════════════════════════


class DiscreteGaussianNoise(NoiseModel):
    """Discrete Gaussian noise over the integers.

    p(x | μ, σ) ∝ exp(-(x - μ)² / (2σ²))  for integer x.
    """

    def _unnormalized_log_prob(self, x: int, center: float, scale: float) -> float:
        """Compute unnormalized log probability."""
        return -((x - center) ** 2) / (2.0 * scale ** 2)

    def _log_normalizer(self, center: float, scale: float, max_range: int = 100) -> float:
        """Compute log of the normalizing constant via summation."""
        rounded = round(center)
        lo_bound = rounded - max_range
        hi_bound = rounded + max_range
        log_probs = [self._unnormalized_log_prob(k, center, scale) for k in range(lo_bound, hi_bound + 1)]
        max_lp = max(log_probs)
        return max_lp + math.log(sum(math.exp(lp - max_lp) for lp in log_probs))

    def density(self, x: float, center: float, scale: float) -> float:
        """Evaluate discrete Gaussian probability mass."""
        if scale <= 0:
            raise ValueError(f"Scale must be positive, got {scale}")
        xi = round(x)
        log_z = self._log_normalizer(center, scale)
        return math.exp(self._unnormalized_log_prob(xi, center, scale) - log_z)

    def log_density(self, x: float, center: float, scale: float) -> float:
        """Evaluate discrete Gaussian log probability."""
        if scale <= 0:
            raise ValueError(f"Scale must be positive, got {scale}")
        xi = round(x)
        log_z = self._log_normalizer(center, scale)
        return self._unnormalized_log_prob(xi, center, scale) - log_z

    def log_ratio(
        self, x: float, center1: float, center2: float, scale: float,
    ) -> float:
        """Compute discrete Gaussian log ratio."""
        if scale <= 0:
            raise ValueError(f"Scale must be positive, got {scale}")
        xi = round(x)
        log_z1 = self._log_normalizer(center1, scale)
        log_z2 = self._log_normalizer(center2, scale)
        return (
            self._unnormalized_log_prob(xi, center1, scale)
            - self._unnormalized_log_prob(xi, center2, scale)
            - log_z1 + log_z2
        )

    def cdf(self, x: float, center: float, scale: float) -> float:
        """Evaluate discrete Gaussian CDF (sum of probabilities for k ≤ x)."""
        if scale <= 0:
            raise ValueError(f"Scale must be positive, got {scale}")
        xi = int(math.floor(x))
        log_z = self._log_normalizer(center, scale)
        rounded = round(center)
        lo_bound = rounded - 100
        total = 0.0
        for k in range(lo_bound, xi + 1):
            total += math.exp(self._unnormalized_log_prob(k, center, scale) - log_z)
        return min(total, 1.0)

    def sample(self, center: float, scale: float, rng: random.Random | None = None) -> float:
        """Draw a discrete Gaussian sample via rejection from continuous."""
        r = rng or random.Random()
        for _ in range(10_000):
            s = round(r.gauss(center, scale))
            accept_prob = math.exp(
                self._unnormalized_log_prob(s, center, scale)
                - (-0.5 * ((s - center) / scale) ** 2)
            )
            if r.random() < min(accept_prob, 1.0):
                return float(s)
        return float(round(center))

    def symbolic_log_ratio(
        self, obs: TypedExpr, center_d: TypedExpr, center_d_prime: TypedExpr, scale: TypedExpr,
    ) -> TypedExpr:
        """Symbolic discrete Gaussian log ratio (approximated by continuous)."""
        return GaussianNoise().symbolic_log_ratio(obs, center_d, center_d_prime, scale)

    def symbolic_log_density(
        self, obs: TypedExpr, center: TypedExpr, scale: TypedExpr,
    ) -> TypedExpr:
        """Symbolic discrete Gaussian log-density (continuous approximation)."""
        return GaussianNoise().symbolic_log_density(obs, center, scale)


class DiscreteLaplaceNoise(NoiseModel):
    """Discrete Laplace noise over the integers.

    p(x | μ, b) ∝ exp(-|x - μ| / b)  for integer x.
    """

    def _normalizer(self, scale: float) -> float:
        """Normalizing constant: (1 - exp(-1/b)) / (1 + exp(-1/b))."""
        e = math.exp(-1.0 / scale)
        return (1.0 - e) / (1.0 + e)

    def density(self, x: float, center: float, scale: float) -> float:
        """Evaluate discrete Laplace probability mass."""
        if scale <= 0:
            raise ValueError(f"Scale must be positive, got {scale}")
        xi = round(x)
        z = self._normalizer(scale)
        return z * math.exp(-abs(xi - center) / scale)

    def log_density(self, x: float, center: float, scale: float) -> float:
        """Evaluate discrete Laplace log probability."""
        if scale <= 0:
            raise ValueError(f"Scale must be positive, got {scale}")
        xi = round(x)
        z = self._normalizer(scale)
        return math.log(z) - abs(xi - center) / scale

    def log_ratio(
        self, x: float, center1: float, center2: float, scale: float,
    ) -> float:
        """Compute discrete Laplace log ratio (same as continuous Laplace)."""
        if scale <= 0:
            raise ValueError(f"Scale must be positive, got {scale}")
        xi = round(x)
        return (abs(xi - center2) - abs(xi - center1)) / scale

    def cdf(self, x: float, center: float, scale: float) -> float:
        """Evaluate discrete Laplace CDF."""
        if scale <= 0:
            raise ValueError(f"Scale must be positive, got {scale}")
        xi = int(math.floor(x))
        rounded = round(center)
        lo = rounded - 200
        z = self._normalizer(scale)
        total = 0.0
        for k in range(lo, xi + 1):
            total += z * math.exp(-abs(k - center) / scale)
        return min(total, 1.0)

    def sample(self, center: float, scale: float, rng: random.Random | None = None) -> float:
        """Draw a discrete Laplace sample via geometric difference."""
        r = rng or random.Random()
        p = 1.0 - math.exp(-1.0 / scale)
        g1 = 0
        while r.random() > p:
            g1 += 1
        g2 = 0
        while r.random() > p:
            g2 += 1
        return float(round(center) + g1 - g2)

    def symbolic_log_ratio(
        self, obs: TypedExpr, center_d: TypedExpr, center_d_prime: TypedExpr, scale: TypedExpr,
    ) -> TypedExpr:
        """Symbolic discrete Laplace log ratio (same form as continuous)."""
        return LaplaceNoise().symbolic_log_ratio(obs, center_d, center_d_prime, scale)

    def symbolic_log_density(
        self, obs: TypedExpr, center: TypedExpr, scale: TypedExpr,
    ) -> TypedExpr:
        """Symbolic discrete Laplace log-density (continuous approximation)."""
        return LaplaceNoise().symbolic_log_density(obs, center, scale)


# ═══════════════════════════════════════════════════════════════════════════
# MIXTURE NOISE
# ═══════════════════════════════════════════════════════════════════════════


class MixtureNoise(NoiseModel):
    """Mixture of multiple noise models: p(x) = Σ wᵢ pᵢ(x).

    Used for composed mechanisms where the effective noise distribution
    is a weighted combination of component distributions.

    Args:
        components: List of (weight, NoiseModel) pairs.
    """

    def __init__(self, components: list[tuple[float, NoiseModel]]) -> None:
        total_w = sum(w for w, _ in components)
        if total_w <= 0:
            raise ValueError("Mixture weights must be positive")
        self._components = [(w / total_w, m) for w, m in components]

    @property
    def num_components(self) -> int:
        """Number of mixture components."""
        return len(self._components)

    def density(self, x: float, center: float, scale: float) -> float:
        """Evaluate mixture density at *x*."""
        return sum(w * m.density(x, center, scale) for w, m in self._components)

    def log_density(self, x: float, center: float, scale: float) -> float:
        """Evaluate mixture log-density via log-sum-exp."""
        log_terms = []
        for w, m in self._components:
            ld = m.log_density(x, center, scale)
            log_terms.append(math.log(w) + ld)
        if not log_terms:
            return -math.inf
        max_lt = max(log_terms)
        return max_lt + math.log(sum(math.exp(lt - max_lt) for lt in log_terms))

    def log_ratio(
        self, x: float, center1: float, center2: float, scale: float,
    ) -> float:
        """Compute mixture log density ratio."""
        d1 = self.density(x, center1, scale)
        d2 = self.density(x, center2, scale)
        if d1 <= 0 or d2 <= 0:
            return 0.0
        return math.log(d1) - math.log(d2)

    def cdf(self, x: float, center: float, scale: float) -> float:
        """Evaluate mixture CDF."""
        return sum(w * m.cdf(x, center, scale) for w, m in self._components)

    def sample(self, center: float, scale: float, rng: random.Random | None = None) -> float:
        """Draw from the mixture by first selecting a component."""
        r = rng or random.Random()
        u = r.random()
        cumulative = 0.0
        for w, m in self._components:
            cumulative += w
            if u <= cumulative:
                return m.sample(center, scale, r)
        return self._components[-1][1].sample(center, scale, r)

    def symbolic_log_ratio(
        self, obs: TypedExpr, center_d: TypedExpr, center_d_prime: TypedExpr, scale: TypedExpr,
    ) -> TypedExpr:
        """Symbolic mixture log ratio (use first component as approximation)."""
        if self._components:
            _, m = self._components[0]
            return m.symbolic_log_ratio(obs, center_d, center_d_prime, scale)
        return Const.real(0.0)

    def symbolic_log_density(
        self, obs: TypedExpr, center: TypedExpr, scale: TypedExpr,
    ) -> TypedExpr:
        """Symbolic mixture log-density (first component approximation)."""
        if self._components:
            _, m = self._components[0]
            return m.symbolic_log_density(obs, center, scale)
        return Const.real(0.0)


# ═══════════════════════════════════════════════════════════════════════════
# FACTORY
# ═══════════════════════════════════════════════════════════════════════════

_NOISE_MODEL_REGISTRY: dict[str, type[NoiseModel]] = {
    "laplace": LaplaceNoise,
    "gaussian": GaussianNoise,
    "exponential": ExponentialMechNoise,
    "truncated_laplace": TruncatedLaplaceNoise,
    "truncated_gaussian": TruncatedGaussianNoise,
    "discrete_gaussian": DiscreteGaussianNoise,
    "discrete_laplace": DiscreteLaplaceNoise,
}


def get_noise_model(kind: str | Any) -> NoiseModel:
    """Get a noise model instance by name or NoiseKind enum.

    Args:
        kind: Either a string key or a :class:`NoiseKind` enum member.

    Returns:
        An instance of the appropriate :class:`NoiseModel`.

    Raises:
        ValueError: If the noise kind is not recognised.
    """
    from dpcegar.ir.types import NoiseKind as NK

    if isinstance(kind, NK):
        mapping = {
            NK.LAPLACE: "laplace",
            NK.GAUSSIAN: "gaussian",
            NK.EXPONENTIAL: "exponential",
        }
        kind = mapping.get(kind, str(kind).lower())

    key = str(kind).lower()
    cls = _NOISE_MODEL_REGISTRY.get(key)
    if cls is None:
        raise ValueError(f"Unknown noise kind: {kind!r}")
    return cls()
