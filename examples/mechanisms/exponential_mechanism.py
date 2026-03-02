"""
Exponential Mechanism implementations for DP-CEGAR verification.

The exponential mechanism selects an output from a discrete set R with
probability proportional to exp(ε · u(db, r) / (2Δu)), where u is a
utility function with sensitivity Δu.

Variants:
  - Standard exponential mechanism
  - Report-noisy-max (equivalent for certain utility structures)
  - Permute-and-flip (improved sampling)

References:
  McSherry & Talwar. "Mechanism Design via Differential Privacy." FOCS 2007.
  Durfee & Rogers. "Practical Differentially Private Top-k Selection with
    Pay-what-you-get Composition." NeurIPS 2019.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Source generators
# ---------------------------------------------------------------------------

def exponential_mechanism_source(
    num_candidates: int = 5,
    sensitivity: float = 1.0,
    epsilon: float = 1.0,
    name: str = "exponential_mechanism",
) -> str:
    """Generate source for the standard exponential mechanism.

    Args:
        num_candidates: Number of candidate outputs |R|.
        sensitivity: Utility sensitivity Δu.
        epsilon: Privacy budget ε.
        name: Function name in generated source.

    Returns:
        Python source string with DPImp annotations.
    """
    scale = epsilon / (2.0 * sensitivity)
    scale_rounded = round(scale, 6)
    return f'''
# @dp.mechanism(privacy="{epsilon}-dp", sensitivity={sensitivity})
def {name}(db, utility_scores):
    """Standard exponential mechanism over {num_candidates} candidates.

    Selects candidate i with probability ∝ exp(ε · u(db,i) / (2Δu)).
    Scale factor = ε / (2Δu) = {scale_rounded}
    """
    n_candidates = {num_candidates}
    # @dp.sensitivity({sensitivity})
    scores = utility_scores(db)
    # @dp.noise(kind="exponential", scale={scale_rounded}, n_candidates={num_candidates})
    selected = exp_mech(scores, {scale_rounded}, n_candidates)
    return selected
'''


def report_noisy_max_source(
    num_candidates: int = 5,
    sensitivity: float = 1.0,
    epsilon: float = 1.0,
    name: str = "report_noisy_max",
) -> str:
    """Generate source for the report-noisy-max mechanism.

    Equivalent to the exponential mechanism for argmax queries.
    Adds Lap(Δu/ε) noise to each score and returns the argmax.

    Args:
        num_candidates: Number of candidate outputs |R|.
        sensitivity: Utility sensitivity Δu.
        epsilon: Privacy budget ε.
        name: Function name in generated source.

    Returns:
        Python source string with DPImp annotations.
    """
    scale = sensitivity / epsilon
    noise_lines = []
    for i in range(num_candidates):
        noise_lines.append(
            f"    noise_{i} = laplace(0, {scale})\n"
            f"    noisy_score_{i} = scores[{i}] + noise_{i}"
        )
    noise_block = "\n".join(noise_lines)
    max_logic = "    best_idx = 0\n    best_val = noisy_score_0\n"
    for i in range(1, num_candidates):
        max_logic += (
            f"    if noisy_score_{i} > best_val:\n"
            f"        best_idx = {i}\n"
            f"        best_val = noisy_score_{i}\n"
        )
    return f'''
# @dp.mechanism(privacy="{epsilon}-dp", sensitivity={sensitivity})
def {name}(db, utility_scores):
    """Report-noisy-max: add Lap(Δu/ε) to each score, return argmax.

    Equivalent to exponential mechanism for argmax queries.
    """
    # @dp.sensitivity({sensitivity})
    scores = utility_scores(db)
{noise_block}
{max_logic}
    return best_idx
'''


def exponential_wrong_sensitivity_source(
    num_candidates: int = 5,
    true_sensitivity: float = 1.0,
    wrong_sensitivity: float = 0.5,
    epsilon: float = 1.0,
    name: str = "exp_wrong_sensitivity",
) -> str:
    """Generate source for exponential mechanism with under-estimated sensitivity.

    BUG: Uses Δu/2 instead of Δu, resulting in too little noise.

    Args:
        num_candidates: Number of candidate outputs.
        true_sensitivity: Correct utility sensitivity.
        wrong_sensitivity: Incorrectly assumed sensitivity (too small).
        epsilon: Privacy budget ε.
        name: Function name in generated source.

    Returns:
        Python source string with DPImp annotations.
    """
    wrong_scale = epsilon / (2.0 * wrong_sensitivity)
    wrong_scale_rounded = round(wrong_scale, 6)
    return f'''
# @dp.mechanism(privacy="{epsilon}-dp", sensitivity={true_sensitivity})
def {name}(db, utility_scores):
    """Exponential mechanism with UNDER-ESTIMATED sensitivity.

    BUG: Uses Δu={wrong_sensitivity} instead of Δu={true_sensitivity}.
    Scale = ε / (2·{wrong_sensitivity}) = {wrong_scale_rounded} (too large → too concentrated).
    """
    n_candidates = {num_candidates}
    # @dp.sensitivity({true_sensitivity})
    scores = utility_scores(db)
    # @dp.noise(kind="exponential", scale={wrong_scale_rounded})
    selected = exp_mech(scores, {wrong_scale_rounded}, n_candidates)
    return selected
'''


def exponential_missing_factor_source(
    num_candidates: int = 5,
    sensitivity: float = 1.0,
    epsilon: float = 1.0,
    name: str = "exp_missing_2",
) -> str:
    """Generate source for exponential mechanism missing the factor of 2.

    BUG: Uses exp(ε·u/Δu) instead of exp(ε·u/(2Δu)).

    Args:
        num_candidates: Number of candidates.
        sensitivity: Utility sensitivity Δu.
        epsilon: Privacy budget ε.
        name: Function name in generated source.

    Returns:
        Python source string with DPImp annotations.
    """
    wrong_scale = epsilon / sensitivity  # Missing factor of 2
    wrong_scale_rounded = round(wrong_scale, 6)
    return f'''
# @dp.mechanism(privacy="{epsilon}-dp", sensitivity={sensitivity})
def {name}(db, utility_scores):
    """Exponential mechanism MISSING the factor of 2 in denominator.

    BUG: scale = ε/Δu = {wrong_scale_rounded} instead of ε/(2Δu).
    This only satisfies 2ε-DP, not ε-DP.
    """
    n_candidates = {num_candidates}
    # @dp.sensitivity({sensitivity})
    scores = utility_scores(db)
    # @dp.noise(kind="exponential", scale={wrong_scale_rounded})
    selected = exp_mech(scores, {wrong_scale_rounded}, n_candidates)
    return selected
'''


def noisy_max_wrong_noise_source(
    num_candidates: int = 5,
    sensitivity: float = 1.0,
    epsilon: float = 1.0,
    name: str = "noisy_max_wrong_noise",
) -> str:
    """Generate source for report-noisy-max using Gaussian instead of Laplace.

    BUG: Using Gaussian noise breaks the equivalence to exponential mechanism
    under pure DP.

    Args:
        num_candidates: Number of candidates.
        sensitivity: Utility sensitivity.
        epsilon: Privacy budget ε.
        name: Function name in generated source.

    Returns:
        Python source string with DPImp annotations.
    """
    scale = sensitivity / epsilon
    noise_lines = []
    for i in range(num_candidates):
        noise_lines.append(
            f"    noise_{i} = gaussian(0, {scale})  # BUG: should be laplace\n"
            f"    noisy_score_{i} = scores[{i}] + noise_{i}"
        )
    noise_block = "\n".join(noise_lines)
    max_logic = "    best_idx = 0\n    best_val = noisy_score_0\n"
    for i in range(1, num_candidates):
        max_logic += (
            f"    if noisy_score_{i} > best_val:\n"
            f"        best_idx = {i}\n"
            f"        best_val = noisy_score_{i}\n"
        )
    return f'''
# @dp.mechanism(privacy="{epsilon}-dp", sensitivity={sensitivity})
def {name}(db, utility_scores):
    """Report-noisy-max using GAUSSIAN noise instead of Laplace.

    BUG: Gaussian noise does not give pure ε-DP for noisy-max.
    """
    # @dp.sensitivity({sensitivity})
    scores = utility_scores(db)
{noise_block}
{max_logic}
    return best_idx
'''


# ---------------------------------------------------------------------------
# Pre-built source constants
# ---------------------------------------------------------------------------

EXPONENTIAL_CORRECT = exponential_mechanism_source(
    num_candidates=5, sensitivity=1.0, epsilon=1.0
)
"""Correct exponential mechanism with 5 candidates, Δu=1, ε=1."""

REPORT_NOISY_MAX_CORRECT = report_noisy_max_source(
    num_candidates=5, sensitivity=1.0, epsilon=1.0
)
"""Correct report-noisy-max mechanism."""

EXPONENTIAL_WRONG_SENSITIVITY = exponential_wrong_sensitivity_source(
    num_candidates=5, true_sensitivity=1.0, wrong_sensitivity=0.5, epsilon=1.0
)
"""Buggy exponential mechanism: underestimated sensitivity."""

EXPONENTIAL_MISSING_FACTOR = exponential_missing_factor_source(
    num_candidates=5, sensitivity=1.0, epsilon=1.0
)
"""Buggy exponential mechanism: missing factor of 2."""

NOISY_MAX_WRONG_NOISE = noisy_max_wrong_noise_source(
    num_candidates=5, sensitivity=1.0, epsilon=1.0
)
"""Buggy noisy-max: uses Gaussian instead of Laplace noise."""


# ---------------------------------------------------------------------------
# Benchmark configurations
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExponentialBenchConfig:
    """Configuration for a single exponential mechanism benchmark."""

    num_candidates: int
    sensitivity: float
    epsilon: float
    name: str
    is_correct: bool
    variant: str = "standard"
    description: str = ""

    def source(self) -> str:
        """Generate the mechanism source for this configuration."""
        if self.variant == "noisy_max":
            return report_noisy_max_source(
                num_candidates=self.num_candidates,
                sensitivity=self.sensitivity,
                epsilon=self.epsilon,
                name=self.name,
            )
        if self.is_correct:
            return exponential_mechanism_source(
                num_candidates=self.num_candidates,
                sensitivity=self.sensitivity,
                epsilon=self.epsilon,
                name=self.name,
            )
        return exponential_wrong_sensitivity_source(
            num_candidates=self.num_candidates,
            true_sensitivity=self.sensitivity,
            wrong_sensitivity=self.sensitivity / 2,
            epsilon=self.epsilon,
            name=self.name,
        )


EXPONENTIAL_BENCH_CONFIGS: list[ExponentialBenchConfig] = [
    ExponentialBenchConfig(3, 1.0, 1.0, "exp_3c", True, description="3 candidates"),
    ExponentialBenchConfig(5, 1.0, 1.0, "exp_5c", True, description="5 candidates"),
    ExponentialBenchConfig(10, 1.0, 1.0, "exp_10c", True, description="10 candidates"),
    ExponentialBenchConfig(5, 1.0, 1.0, "exp_nm", True, "noisy_max", "Noisy max"),
    ExponentialBenchConfig(5, 1.0, 1.0, "exp_bug", False, description="Wrong sens."),
]
"""Benchmark sweep configurations for exponential mechanism."""
