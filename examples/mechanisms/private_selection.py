"""
Private Selection mechanism implementations for DP-CEGAR verification.

Private selection uses the exponential mechanism to select among privately
computed statistics.  Variants include:
  - Exponential mechanism over pre-computed scores
  - Report noisy max (add noise to each score, return argmax)
  - Private top-k selection (select k items with privacy)

References:
  Liu & Talwar. "Private Selection from Private Candidates." STOC 2019.
  Durfee & Rogers. "Practical Differentially Private Top-k Selection." NeurIPS 2019.
  Papernot & Steinke. "Hyperparameter Tuning with Renyi DP." ICLR 2022.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Source generators
# ---------------------------------------------------------------------------

def private_selection_source(
    num_candidates: int = 5,
    sensitivity: float = 1.0,
    epsilon: float = 1.0,
    name: str = "private_selection",
) -> str:
    """Generate source for private selection mechanism.

    Given candidates whose scores are computed via DP mechanisms,
    select the best candidate using the exponential mechanism.

    Args:
        num_candidates: Number of candidate statistics.
        sensitivity: Sensitivity of the scoring function.
        epsilon: Privacy budget for selection.
        name: Function name.

    Returns:
        Python source with DPImp annotations.
    """
    scale = epsilon / (2.0 * sensitivity)
    scale_r = round(scale, 6)
    return f'''
# @dp.mechanism(privacy="{epsilon}-dp", sensitivity={sensitivity})
def {name}(db, score_functions):
    """Private selection: choose best candidate via exponential mechanism.

    Uses ε = {epsilon} for selection among {num_candidates} candidates.
    Scale = ε/(2Δu) = {scale_r}.
    """
    scores = []
    for i in range({num_candidates}):
        # @dp.sensitivity({sensitivity})
        s = score_functions[i](db)
        scores.append(s)
    # @dp.noise(kind="exponential", scale={scale_r}, n_candidates={num_candidates})
    selected = exp_mech(scores, {scale_r}, {num_candidates})
    return selected
'''


def report_noisy_max_private_source(
    num_candidates: int = 5,
    sensitivity: float = 1.0,
    epsilon: float = 1.0,
    name: str = "report_noisy_max_sel",
) -> str:
    """Generate source for report-noisy-max selection.

    Adds independent Lap(Δf/ε) noise to each score, returns the index
    of the maximum.  Privacy cost is ε (by the Lipschitz property of max).

    Args:
        num_candidates: Number of candidates.
        sensitivity: Score sensitivity.
        epsilon: Privacy budget.
        name: Function name.

    Returns:
        Python source with DPImp annotations.
    """
    scale = sensitivity / epsilon
    return f'''
# @dp.mechanism(privacy="{epsilon}-dp", sensitivity={sensitivity})
def {name}(db, score_functions):
    """Report noisy max for private selection.

    Add Lap({scale}) to each score, return argmax.
    """
    best_idx = 0
    best_val = -1000000
    for i in range({num_candidates}):
        # @dp.sensitivity({sensitivity})
        s = score_functions[i](db)
        # @dp.noise(kind="laplace", scale={scale})
        noisy_s = s + laplace(0, {scale})
        if noisy_s > best_val:
            best_val = noisy_s
            best_idx = i
    return best_idx
'''


def private_top_k_source(
    num_candidates: int = 10,
    k: int = 3,
    sensitivity: float = 1.0,
    epsilon: float = 1.0,
    name: str = "private_top_k",
) -> str:
    """Generate source for private top-k selection.

    Selects k items using sequential application of report-noisy-max
    with fresh noise.  Budget splits: ε/k per selection.

    Args:
        num_candidates: Total candidates.
        k: Number of items to select.
        sensitivity: Score sensitivity.
        epsilon: Total privacy budget.
        name: Function name.

    Returns:
        Python source with DPImp annotations.
    """
    eps_per = epsilon / k
    scale = sensitivity / eps_per
    scale_r = round(scale, 6)
    return f'''
# @dp.mechanism(privacy="{epsilon}-dp", sensitivity={sensitivity})
def {name}(db, score_functions):
    """Private top-{k} selection from {num_candidates} candidates.

    Sequential noisy-max with ε/k = {eps_per:.4f} per selection.
    """
    selected = []
    remaining = list(range({num_candidates}))
    for round_idx in range({k}):
        best_idx = -1
        best_val = -1000000
        for j in range(len(remaining)):
            idx = remaining[j]
            # @dp.sensitivity({sensitivity})
            s = score_functions[idx](db)
            # @dp.noise(kind="laplace", scale={scale_r})
            noisy_s = s + laplace(0, {scale_r})
            if noisy_s > best_val:
                best_val = noisy_s
                best_idx = j
        winner = remaining[best_idx]
        selected.append(winner)
        remaining.pop(best_idx)
    return selected
'''


def private_selection_wrong_budget_source(
    num_candidates: int = 5,
    sensitivity: float = 1.0,
    epsilon: float = 1.0,
    name: str = "selection_wrong_budget",
) -> str:
    """Generate source for private selection with wrong budget in scoring.

    BUG: The scoring function queries are not accounted for in the privacy
    budget — only the selection step uses ε, but the scores also leak info.

    Args:
        num_candidates: Number of candidates.
        sensitivity: Score sensitivity.
        epsilon: Claimed privacy budget.
        name: Function name.

    Returns:
        Python source with DPImp annotations.
    """
    scale = epsilon / (2.0 * sensitivity)
    scale_r = round(scale, 6)
    return f'''
# @dp.mechanism(privacy="{epsilon}-dp", sensitivity={sensitivity})
def {name}(db, raw_queries):
    """Private selection with WRONG budget accounting.

    BUG: Scores are computed from raw queries without noise,
    then selection uses ε — but the score computation also
    consumes privacy budget (not accounted for).
    """
    scores = []
    for i in range({num_candidates}):
        # BUG: raw query without noise — leaks info!
        s = raw_queries[i](db)
        scores.append(s)
    # @dp.noise(kind="exponential", scale={scale_r}, n_candidates={num_candidates})
    selected = exp_mech(scores, {scale_r}, {num_candidates})
    return selected
'''


def private_selection_double_dip_source(
    num_candidates: int = 5,
    sensitivity: float = 1.0,
    epsilon: float = 1.0,
    name: str = "selection_double_dip",
) -> str:
    """Generate source for private selection that reuses data.

    BUG: After selecting the best candidate, returns the raw (non-private)
    score of that candidate — this is a second use of the data that is not
    accounted for in the privacy analysis.

    Args:
        num_candidates: Number of candidates.
        sensitivity: Score sensitivity.
        epsilon: Claimed privacy budget.
        name: Function name.

    Returns:
        Python source with DPImp annotations.
    """
    scale = sensitivity / epsilon
    return f'''
# @dp.mechanism(privacy="{epsilon}-dp", sensitivity={sensitivity})
def {name}(db, score_functions):
    """Private selection with DOUBLE-DIP on data.

    BUG: Returns raw score of selected candidate — second data access
    not covered by privacy budget.
    """
    best_idx = 0
    best_val = -1000000
    for i in range({num_candidates}):
        # @dp.sensitivity({sensitivity})
        s = score_functions[i](db)
        # @dp.noise(kind="laplace", scale={scale})
        noisy_s = s + laplace(0, {scale})
        if noisy_s > best_val:
            best_val = noisy_s
            best_idx = i
    # BUG: returning raw score — double dip!
    raw_score = score_functions[best_idx](db)
    return (best_idx, raw_score)
'''


# ---------------------------------------------------------------------------
# Pre-built source constants
# ---------------------------------------------------------------------------

PRIVATE_SELECTION_CORRECT = private_selection_source(
    num_candidates=5, epsilon=1.0
)
"""Correct private selection mechanism."""

REPORT_NOISY_MAX_SELECTION = report_noisy_max_private_source(
    num_candidates=5, epsilon=1.0
)
"""Correct report-noisy-max selection."""

PRIVATE_TOP_K = private_top_k_source(
    num_candidates=10, k=3, epsilon=1.0
)
"""Correct private top-3 selection from 10 candidates."""

SELECTION_WRONG_BUDGET = private_selection_wrong_budget_source(
    num_candidates=5, epsilon=1.0
)
"""Buggy: scores not budget-accounted."""

SELECTION_DOUBLE_DIP = private_selection_double_dip_source(
    num_candidates=5, epsilon=1.0
)
"""Buggy: returns raw score of selected candidate."""


# ---------------------------------------------------------------------------
# Benchmark configurations
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SelectionBenchConfig:
    """Configuration for a private selection benchmark instance."""

    name: str
    num_candidates: int
    epsilon: float
    is_correct: bool
    variant: str = "exponential"
    description: str = ""

    def source(self) -> str:
        """Generate the mechanism source for this configuration."""
        if self.variant == "noisy_max":
            return report_noisy_max_private_source(
                num_candidates=self.num_candidates,
                epsilon=self.epsilon,
                name=self.name,
            )
        if self.variant == "top_k":
            return private_top_k_source(
                num_candidates=self.num_candidates,
                k=3,
                epsilon=self.epsilon,
                name=self.name,
            )
        if self.is_correct:
            return private_selection_source(
                num_candidates=self.num_candidates,
                epsilon=self.epsilon,
                name=self.name,
            )
        return private_selection_wrong_budget_source(
            num_candidates=self.num_candidates,
            epsilon=self.epsilon,
            name=self.name,
        )


SELECTION_BENCH_CONFIGS: list[SelectionBenchConfig] = [
    SelectionBenchConfig("sel_5", 5, 1.0, True, description="5 candidates"),
    SelectionBenchConfig("sel_10", 10, 1.0, True, description="10 candidates"),
    SelectionBenchConfig("sel_nm", 5, 1.0, True, "noisy_max", "Noisy max"),
    SelectionBenchConfig("sel_top3", 10, 1.0, True, "top_k", "Top-3"),
    SelectionBenchConfig("sel_bug", 5, 1.0, False, description="Wrong budget"),
]
"""Benchmark configurations for private selection."""
