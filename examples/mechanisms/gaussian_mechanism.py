"""
Gaussian Mechanism implementations for DP-CEGAR verification.

The Gaussian mechanism adds noise drawn from N(0, σ²) where
σ = Δf · √(2 ln(1.25/δ)) / ε  for (ε,δ)-differential privacy.

Variants include:
  - Standard Gaussian mechanism (Dwork-Roth, Appendix A)
  - Analytic Gaussian mechanism (Balle & Wang, ICML 2018)
  - zCDP-calibrated Gaussian (Bun & Dwork, 2016)

This module provides correct and buggy versions with DPImp annotations.

References:
  Dwork & Roth. "The Algorithmic Foundations of DP." 2014, Appendix A.
  Balle & Wang. "Improving the Gaussian Mechanism for DP: Analytical
    Calibration and Optimal Denoising." ICML 2018.
  Bun & Dwork. "Concentrated Differential Privacy." 2016.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Source generators
# ---------------------------------------------------------------------------

def gaussian_mechanism_source(
    sensitivity: float = 1.0,
    epsilon: float = 1.0,
    delta: float = 1e-5,
    name: str = "gaussian_mechanism",
) -> str:
    """Generate source for the standard Gaussian mechanism.

    Uses σ = Δf · √(2·ln(1.25/δ)) / ε.

    Args:
        sensitivity: L2 sensitivity Δ₂f.
        epsilon: Privacy budget ε.
        delta: Privacy budget δ.
        name: Function name in generated source.

    Returns:
        Python source string with DPImp annotations.
    """
    sigma = sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon
    sigma_rounded = round(sigma, 6)
    return f'''
# @dp.mechanism(privacy="({epsilon},{delta})-dp", sensitivity={sensitivity})
def {name}(db, query):
    """Standard Gaussian mechanism: f(db) + N(0, σ²).

    σ = Δ₂f · √(2·ln(1.25/δ)) / ε = {sigma_rounded}
    """
    # @dp.sensitivity({sensitivity}, norm="L2")
    true_answer = query(db)
    # @dp.noise(kind="gaussian", sigma={sigma_rounded})
    noise = gaussian(0, {sigma_rounded})
    result = true_answer + noise
    return result
'''


def analytic_gaussian_source(
    sensitivity: float = 1.0,
    epsilon: float = 1.0,
    delta: float = 1e-5,
    name: str = "analytic_gaussian",
) -> str:
    """Generate source for the analytic Gaussian mechanism.

    Uses tighter σ calibration from Balle & Wang (2018).  The analytic
    calibration finds the smallest σ satisfying the (ε,δ) constraint
    via a numerical procedure.  Here we use the closed-form upper bound.

    Args:
        sensitivity: L2 sensitivity Δ₂f.
        epsilon: Privacy budget ε.
        delta: Privacy budget δ.
        name: Function name in generated source.

    Returns:
        Python source string with DPImp annotations.
    """
    # Analytic calibration upper bound: tighter than standard
    sigma = sensitivity * math.sqrt(2.0 * math.log(1.0 / delta)) / epsilon
    sigma_rounded = round(sigma, 6)
    return f'''
# @dp.mechanism(privacy="({epsilon},{delta})-dp", sensitivity={sensitivity})
def {name}(db, query):
    """Analytic Gaussian mechanism (Balle & Wang, 2018).

    Tighter σ calibration: σ = {sigma_rounded}
    """
    # @dp.sensitivity({sensitivity}, norm="L2")
    true_answer = query(db)
    # @dp.noise(kind="gaussian", sigma={sigma_rounded})
    noise = gaussian(0, {sigma_rounded})
    result = true_answer + noise
    return result
'''


def gaussian_zcdp_source(
    sensitivity: float = 1.0,
    rho: float = 0.5,
    name: str = "gaussian_zcdp",
) -> str:
    """Generate source for the zCDP-calibrated Gaussian mechanism.

    Under zCDP, the Gaussian mechanism with σ = Δ₂f / √(2ρ) satisfies ρ-zCDP.

    Args:
        sensitivity: L2 sensitivity Δ₂f.
        rho: zCDP budget ρ.
        name: Function name in generated source.

    Returns:
        Python source string with DPImp annotations.
    """
    sigma = sensitivity / math.sqrt(2.0 * rho)
    sigma_rounded = round(sigma, 6)
    return f'''
# @dp.mechanism(privacy="{rho}-zCDP", sensitivity={sensitivity})
def {name}(db, query):
    """zCDP-calibrated Gaussian: σ = Δ₂f / √(2ρ) = {sigma_rounded}."""
    # @dp.sensitivity({sensitivity}, norm="L2")
    true_answer = query(db)
    # @dp.noise(kind="gaussian", sigma={sigma_rounded})
    noise = gaussian(0, {sigma_rounded})
    result = true_answer + noise
    return result
'''


def gaussian_wrong_sigma_source(
    sensitivity: float = 1.0,
    epsilon: float = 1.0,
    delta: float = 1e-5,
    name: str = "gaussian_wrong_sigma",
) -> str:
    """Generate source for Gaussian mechanism with wrong σ.

    BUG: Uses σ = Δf/ε (missing the √(2·ln(1.25/δ)) factor), which is
    insufficient for (ε,δ)-DP.

    Args:
        sensitivity: L2 sensitivity Δ₂f.
        epsilon: Privacy budget ε.
        delta: Privacy budget δ.
        name: Function name in generated source.

    Returns:
        Python source string with DPImp annotations.
    """
    wrong_sigma = sensitivity / epsilon  # BUG: missing ln(1/δ) factor
    return f'''
# @dp.mechanism(privacy="({epsilon},{delta})-dp", sensitivity={sensitivity})
def {name}(db, query):
    """Gaussian mechanism with WRONG σ (missing ln(1/δ) factor).

    BUG: σ = Δf/ε = {wrong_sigma} instead of Δf·√(2·ln(1.25/δ))/ε
    """
    # @dp.sensitivity({sensitivity}, norm="L2")
    true_answer = query(db)
    # @dp.noise(kind="gaussian", sigma={wrong_sigma})
    noise = gaussian(0, {wrong_sigma})
    result = true_answer + noise
    return result
'''


def gaussian_wrong_norm_source(
    l1_sensitivity: float = 3.0,
    l2_sensitivity: float = 1.0,
    epsilon: float = 1.0,
    delta: float = 1e-5,
    name: str = "gaussian_wrong_norm",
) -> str:
    """Generate source for Gaussian mechanism using L1 instead of L2 sensitivity.

    BUG: Calibrates noise to the L1 sensitivity when the Gaussian mechanism
    requires the L2 sensitivity.  When L1 > L2 the noise is too large
    (conservative but wasteful); when L1 < L2 it is too small (violates DP).

    Args:
        l1_sensitivity: L1 sensitivity (incorrectly used).
        l2_sensitivity: True L2 sensitivity (should be used).
        epsilon: Privacy budget ε.
        delta: Privacy budget δ.
        name: Function name in generated source.

    Returns:
        Python source string with DPImp annotations.
    """
    # BUG: using L1 sensitivity for Gaussian mechanism
    wrong_sigma = l1_sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon
    wrong_sigma_rounded = round(wrong_sigma, 6)
    return f'''
# @dp.mechanism(privacy="({epsilon},{delta})-dp", sensitivity={l2_sensitivity})
def {name}(db, query):
    """Gaussian mechanism using WRONG sensitivity norm.

    BUG: Uses L1 sensitivity ({l1_sensitivity}) instead of L2 ({l2_sensitivity}).
    """
    # @dp.sensitivity({l1_sensitivity}, norm="L1")  # BUG: should be L2
    true_answer = query(db)
    # @dp.noise(kind="gaussian", sigma={wrong_sigma_rounded})
    noise = gaussian(0, {wrong_sigma_rounded})
    result = true_answer + noise
    return result
'''


def gaussian_multi_dim_source(
    dimension: int = 3,
    sensitivity: float = 1.0,
    epsilon: float = 1.0,
    delta: float = 1e-5,
    name: str = "gaussian_multi_dim",
) -> str:
    """Generate source for vector-valued Gaussian mechanism.

    Adds independent Gaussian noise to each coordinate of a d-dimensional
    query answer.  Same σ per coordinate suffices for L2 sensitivity.

    Args:
        dimension: Number of output dimensions.
        sensitivity: L2 sensitivity Δ₂f.
        epsilon: Privacy budget ε.
        delta: Privacy budget δ.
        name: Function name in generated source.

    Returns:
        Python source string with DPImp annotations.
    """
    sigma = sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon
    sigma_rounded = round(sigma, 6)
    dims = []
    for i in range(dimension):
        dims.append(f"""
    # @dp.noise(kind="gaussian", sigma={sigma_rounded})
    noise_{i} = gaussian(0, {sigma_rounded})
    result_{i} = answer[{i}] + noise_{i}""")
    body = "\n".join(dims)
    returns = ", ".join(f"result_{i}" for i in range(dimension))
    return f'''
# @dp.mechanism(privacy="({epsilon},{delta})-dp", sensitivity={sensitivity})
def {name}(db, query):
    """Vector-valued Gaussian mechanism ({dimension}-dimensional).

    σ = {sigma_rounded} for each coordinate.
    """
    # @dp.sensitivity({sensitivity}, norm="L2")
    answer = query(db)
{body}
    return ({returns})
'''


# ---------------------------------------------------------------------------
# Pre-built source constants
# ---------------------------------------------------------------------------

GAUSSIAN_CORRECT = gaussian_mechanism_source(
    sensitivity=1.0, epsilon=1.0, delta=1e-5
)
"""Correct Gaussian mechanism with Δ₂f=1, ε=1, δ=10⁻⁵."""

GAUSSIAN_ANALYTIC_CORRECT = analytic_gaussian_source(
    sensitivity=1.0, epsilon=1.0, delta=1e-5
)
"""Correct analytic Gaussian mechanism (Balle & Wang)."""

GAUSSIAN_ZCDP = gaussian_zcdp_source(sensitivity=1.0, rho=0.5)
"""Correct zCDP-calibrated Gaussian mechanism with ρ=0.5."""

GAUSSIAN_WRONG_SIGMA = gaussian_wrong_sigma_source(
    sensitivity=1.0, epsilon=1.0, delta=1e-5
)
"""Buggy Gaussian mechanism: σ = Δf/ε (missing log factor)."""

GAUSSIAN_WRONG_NORM = gaussian_wrong_norm_source(
    l1_sensitivity=3.0, l2_sensitivity=1.0, epsilon=1.0, delta=1e-5
)
"""Buggy Gaussian mechanism: uses L1 sensitivity instead of L2."""

GAUSSIAN_MULTI_DIM = gaussian_multi_dim_source(dimension=3)
"""Correct 3-dimensional Gaussian mechanism."""


# ---------------------------------------------------------------------------
# Benchmark configurations
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GaussianBenchConfig:
    """Configuration for a single Gaussian benchmark instance."""

    sensitivity: float
    epsilon: float
    delta: float
    name: str
    is_correct: bool
    privacy_notion: str = "approx_dp"
    description: str = ""

    def source(self) -> str:
        """Generate the mechanism source for this configuration."""
        if self.is_correct:
            return gaussian_mechanism_source(
                sensitivity=self.sensitivity,
                epsilon=self.epsilon,
                delta=self.delta,
                name=self.name,
            )
        return gaussian_wrong_sigma_source(
            sensitivity=self.sensitivity,
            epsilon=self.epsilon,
            delta=self.delta,
            name=self.name,
        )


GAUSSIAN_BENCH_CONFIGS: list[GaussianBenchConfig] = [
    GaussianBenchConfig(1.0, 0.1, 1e-5, "gauss_e01", True, description="Low ε"),
    GaussianBenchConfig(1.0, 0.5, 1e-5, "gauss_e05", True, description="Moderate ε"),
    GaussianBenchConfig(1.0, 1.0, 1e-5, "gauss_e10", True, description="Standard"),
    GaussianBenchConfig(1.0, 1.0, 1e-8, "gauss_d8", True, description="Small δ"),
    GaussianBenchConfig(1.0, 1.0, 1e-5, "gauss_bug", False, description="Wrong σ"),
]
"""Benchmark sweep configurations for Gaussian mechanism."""
