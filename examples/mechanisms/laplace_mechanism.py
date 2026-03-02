"""
Laplace Mechanism implementations for DP-CEGAR verification.

The Laplace mechanism is the canonical pure ε-differential privacy mechanism.
Given a numeric query f with sensitivity Δf, the mechanism adds noise drawn
from Lap(Δf/ε) to the true answer.

This module provides:
  - Correct implementations at various sensitivity levels
  - Known-buggy variants (wrong scale, missing noise)
  - Source strings with DPImp annotations for parsing

References:
  Dwork, McSherry, Nissim, Smith. "Calibrating Noise to Sensitivity in
  Private Data Analysis." TCC 2006.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Mechanism source strings (parseable by dpcegar.parser)
# ---------------------------------------------------------------------------

def laplace_mechanism_source(
    sensitivity: float = 1.0,
    epsilon: float = 1.0,
    name: str = "laplace_mechanism",
) -> str:
    """Generate source for a correct Laplace mechanism.

    Args:
        sensitivity: Query sensitivity Δf.
        epsilon: Privacy budget ε.
        name: Function name in generated source.

    Returns:
        Python source string with DPImp annotations.
    """
    scale = sensitivity / epsilon
    return f'''
# @dp.mechanism(privacy="{epsilon}-dp", sensitivity={sensitivity})
def {name}(db, query):
    """Standard Laplace mechanism: f(db) + Lap(Δf/ε)."""
    # @dp.sensitivity({sensitivity})
    true_answer = query(db)
    # @dp.noise(kind="laplace", scale={scale})
    noise = laplace(0, {scale})
    result = true_answer + noise
    return result
'''


def laplace_wrong_scale_source(
    sensitivity: float = 1.0,
    epsilon: float = 1.0,
    name: str = "laplace_wrong_scale",
) -> str:
    """Generate source for a Laplace mechanism with incorrect scale.

    The scale is set to sensitivity * epsilon instead of sensitivity / epsilon,
    which provides insufficient noise for privacy.

    Args:
        sensitivity: Query sensitivity Δf.
        epsilon: Privacy budget ε.
        name: Function name in generated source.

    Returns:
        Python source string with DPImp annotations.
    """
    wrong_scale = sensitivity * epsilon  # BUG: should be sensitivity / epsilon
    return f'''
# @dp.mechanism(privacy="{epsilon}-dp", sensitivity={sensitivity})
def {name}(db, query):
    """Laplace mechanism with WRONG scale (scale = Δf·ε instead of Δf/ε)."""
    # @dp.sensitivity({sensitivity})
    true_answer = query(db)
    # @dp.noise(kind="laplace", scale={wrong_scale})
    noise = laplace(0, {wrong_scale})
    result = true_answer + noise
    return result
'''


def laplace_missing_noise_source(
    sensitivity: float = 1.0,
    name: str = "laplace_missing_noise",
) -> str:
    """Generate source for a Laplace mechanism that forgets to add noise.

    This is the most basic bug — the mechanism returns the true answer directly.

    Args:
        sensitivity: Query sensitivity Δf.
        name: Function name in generated source.

    Returns:
        Python source string with DPImp annotations.
    """
    return f'''
# @dp.mechanism(privacy="1.0-dp", sensitivity={sensitivity})
def {name}(db, query):
    """Laplace mechanism that FORGETS to add noise (returns raw answer)."""
    # @dp.sensitivity({sensitivity})
    true_answer = query(db)
    # BUG: no noise added!
    return true_answer
'''


def laplace_multi_query_source(
    num_queries: int = 3,
    sensitivity: float = 1.0,
    epsilon: float = 1.0,
    name: str = "laplace_multi_query",
) -> str:
    """Generate source for a Laplace mechanism answering multiple queries.

    Uses sequential composition: total budget = num_queries * epsilon_per_query.

    Args:
        num_queries: Number of queries to answer.
        sensitivity: Per-query sensitivity.
        epsilon: Total privacy budget ε.
        name: Function name in generated source.

    Returns:
        Python source string with DPImp annotations.
    """
    eps_per = epsilon / num_queries
    scale = sensitivity / eps_per
    queries = []
    for i in range(num_queries):
        queries.append(f"""
    # @dp.sensitivity({sensitivity})
    ans_{i} = query_{i}(db)
    # @dp.noise(kind="laplace", scale={scale})
    noise_{i} = laplace(0, {scale})
    result_{i} = ans_{i} + noise_{i}""")
    body = "\n".join(queries)
    returns = ", ".join(f"result_{i}" for i in range(num_queries))
    return f'''
# @dp.mechanism(privacy="{epsilon}-dp", sensitivity={sensitivity})
def {name}(db, {", ".join(f"query_{i}" for i in range(num_queries))}):
    """Answer {num_queries} queries with sequential composition."""
{body}
    return ({returns})
'''


def laplace_clamped_source(
    sensitivity: float = 1.0,
    epsilon: float = 1.0,
    lo: float = 0.0,
    hi: float = 100.0,
    name: str = "laplace_clamped",
) -> str:
    """Generate source for Laplace mechanism with output clamping.

    Post-processing (clamping) does not affect privacy.

    Args:
        sensitivity: Query sensitivity Δf.
        epsilon: Privacy budget ε.
        lo: Lower clamp bound.
        hi: Upper clamp bound.
        name: Function name in generated source.

    Returns:
        Python source string with DPImp annotations.
    """
    scale = sensitivity / epsilon
    return f'''
# @dp.mechanism(privacy="{epsilon}-dp", sensitivity={sensitivity})
def {name}(db, query):
    """Laplace mechanism with output clamping (post-processing)."""
    # @dp.sensitivity({sensitivity})
    true_answer = query(db)
    # @dp.noise(kind="laplace", scale={scale})
    noise = laplace(0, {scale})
    noisy_answer = true_answer + noise
    # Clamping is post-processing; does not affect privacy
    if noisy_answer < {lo}:
        result = {lo}
    elif noisy_answer > {hi}:
        result = {hi}
    else:
        result = noisy_answer
    return result
'''


# ---------------------------------------------------------------------------
# Pre-built source constants for common configurations
# ---------------------------------------------------------------------------

LAPLACE_CORRECT = laplace_mechanism_source(sensitivity=1.0, epsilon=1.0)
"""Correct Laplace mechanism with Δf=1, ε=1."""

LAPLACE_WRONG_SCALE = laplace_wrong_scale_source(sensitivity=1.0, epsilon=1.0)
"""Buggy Laplace mechanism: scale = Δf·ε instead of Δf/ε."""

LAPLACE_MISSING_NOISE = laplace_missing_noise_source(sensitivity=1.0)
"""Buggy Laplace mechanism: no noise added at all."""

LAPLACE_MULTI_QUERY = laplace_multi_query_source(num_queries=3, epsilon=3.0)
"""Correct multi-query Laplace mechanism (3 queries, ε=3.0)."""

LAPLACE_CLAMPED = laplace_clamped_source(sensitivity=1.0, epsilon=1.0)
"""Correct Laplace mechanism with output clamping."""


# ---------------------------------------------------------------------------
# Sensitivity configurations for benchmark sweeps
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LaplaceBenchConfig:
    """Configuration for a single Laplace benchmark instance."""

    sensitivity: float
    epsilon: float
    name: str
    is_correct: bool
    description: str = ""

    def source(self) -> str:
        """Generate the mechanism source for this configuration."""
        if self.is_correct:
            return laplace_mechanism_source(
                sensitivity=self.sensitivity,
                epsilon=self.epsilon,
                name=self.name,
            )
        return laplace_wrong_scale_source(
            sensitivity=self.sensitivity,
            epsilon=self.epsilon,
            name=self.name,
        )


LAPLACE_BENCH_CONFIGS: list[LaplaceBenchConfig] = [
    LaplaceBenchConfig(1.0, 0.1, "lap_s1_e01", True, "Low ε, high noise"),
    LaplaceBenchConfig(1.0, 0.5, "lap_s1_e05", True, "Moderate ε"),
    LaplaceBenchConfig(1.0, 1.0, "lap_s1_e10", True, "Standard ε"),
    LaplaceBenchConfig(1.0, 2.0, "lap_s1_e20", True, "High ε, low noise"),
    LaplaceBenchConfig(2.0, 1.0, "lap_s2_e10", True, "Sensitivity 2"),
    LaplaceBenchConfig(5.0, 1.0, "lap_s5_e10", True, "Sensitivity 5"),
    LaplaceBenchConfig(1.0, 1.0, "lap_bug_s1_e10", False, "Wrong scale bug"),
    LaplaceBenchConfig(2.0, 0.5, "lap_bug_s2_e05", False, "Wrong scale, high sens"),
]
"""Benchmark sweep configurations for Laplace mechanism."""
