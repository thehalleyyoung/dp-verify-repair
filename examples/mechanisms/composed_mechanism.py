"""
Composed Mechanism implementations for DP-CEGAR verification.

Demonstrates sequential, parallel, and adaptive composition of base
DP mechanisms.  Privacy budget tracking ensures total expenditure stays
within the claimed guarantee.

Composition theorems:
  - Sequential: ε_total = Σ εᵢ, δ_total = Σ δᵢ
  - Parallel: ε_total = max εᵢ  (disjoint subsets)
  - Advanced: ε_total = √(2k·ln(1/δ′))·ε + k·ε(eε-1),  (Dwork-Rothblum-Vadhan)

References:
  Dwork, Rothblum, Vadhan. "Boosting and Differential Privacy." FOCS 2010.
  Kairouz, Oh, Viswanath. "The Composition Theorem for DP." ICML 2015.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Source generators
# ---------------------------------------------------------------------------

def sequential_composition_source(
    num_queries: int = 3,
    sensitivity: float = 1.0,
    total_epsilon: float = 3.0,
    name: str = "sequential_composition",
) -> str:
    """Generate source for sequential composition of Laplace mechanisms.

    Each query uses ε/k budget.  Total budget is ε.

    Args:
        num_queries: Number of queries k.
        sensitivity: Per-query sensitivity.
        total_epsilon: Total privacy budget.
        name: Function name.

    Returns:
        Python source with DPImp annotations.
    """
    eps_per = total_epsilon / num_queries
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
# @dp.mechanism(privacy="{total_epsilon}-dp", composition="sequential")
def {name}(db, {", ".join(f"query_{i}" for i in range(num_queries))}):
    """Sequential composition of {num_queries} Laplace mechanisms.

    Each query uses ε/k = {eps_per:.4f} budget.
    Total budget: {total_epsilon} = {num_queries} × {eps_per:.4f}.
    """
{body}
    return ({returns})
'''


def parallel_composition_source(
    num_partitions: int = 3,
    sensitivity: float = 1.0,
    epsilon: float = 1.0,
    name: str = "parallel_composition",
) -> str:
    """Generate source for parallel composition on disjoint partitions.

    Each partition uses ε budget.  Total budget is max(ε) = ε because
    partitions are disjoint.

    Args:
        num_partitions: Number of disjoint data partitions.
        sensitivity: Per-query sensitivity.
        epsilon: Per-partition (and total) privacy budget.
        name: Function name.

    Returns:
        Python source with DPImp annotations.
    """
    scale = sensitivity / epsilon
    parts = []
    for i in range(num_partitions):
        parts.append(f"""
    # @dp.partition(index={i})
    partition_{i} = db[{i}]
    # @dp.sensitivity({sensitivity})
    ans_{i} = query(partition_{i})
    # @dp.noise(kind="laplace", scale={scale})
    noise_{i} = laplace(0, {scale})
    result_{i} = ans_{i} + noise_{i}""")
    body = "\n".join(parts)
    returns = ", ".join(f"result_{i}" for i in range(num_partitions))
    return f'''
# @dp.mechanism(privacy="{epsilon}-dp", composition="parallel")
def {name}(db, query):
    """Parallel composition over {num_partitions} disjoint partitions.

    Each partition uses ε = {epsilon}. By parallel composition,
    total budget = max(ε) = {epsilon}.
    """
{body}
    return ({returns})
'''


def adaptive_composition_source(
    num_rounds: int = 5,
    sensitivity: float = 1.0,
    total_epsilon: float = 1.0,
    delta_prime: float = 1e-5,
    name: str = "adaptive_composition",
) -> str:
    """Generate source for adaptive composition using advanced composition.

    Uses the Dwork-Rothblum-Vadhan advanced composition theorem to get
    a tighter total budget than naive sequential composition.

    Args:
        num_rounds: Number of adaptive rounds k.
        sensitivity: Per-query sensitivity.
        total_epsilon: Target total ε.
        delta_prime: Composition slack δ'.
        name: Function name.

    Returns:
        Python source with DPImp annotations.
    """
    # Per-round epsilon from advanced composition:
    # ε_total ≈ ε_round · √(2k · ln(1/δ')) + k · ε_round · (e^ε_round - 1)
    # For small ε_round: ε_total ≈ ε_round · √(2k · ln(1/δ'))
    eps_per_round = total_epsilon / math.sqrt(
        2.0 * num_rounds * math.log(1.0 / delta_prime)
    )
    eps_per_r = round(eps_per_round, 6)
    scale = round(sensitivity / eps_per_round, 6)
    return f'''
# @dp.mechanism(privacy="({total_epsilon},{delta_prime})-dp", composition="advanced")
def {name}(db, queries):
    """Adaptive composition over {num_rounds} rounds (advanced composition).

    Per-round ε = {eps_per_r}.
    Total ε ≈ ε_round · √(2k·ln(1/δ')) = {total_epsilon}.
    """
    results = []
    budget_spent = 0.0
    for i in range({num_rounds}):
        # Adaptive: query may depend on previous results
        # @dp.sensitivity({sensitivity})
        ans = queries[i](db, results)
        # @dp.noise(kind="laplace", scale={scale})
        noise = laplace(0, {scale})
        noisy_ans = ans + noise
        results.append(noisy_ans)
        budget_spent = budget_spent + {eps_per_r}
    return results
'''


def sequential_wrong_split_source(
    num_queries: int = 3,
    sensitivity: float = 1.0,
    total_epsilon: float = 1.0,
    name: str = "seq_wrong_split",
) -> str:
    """Generate source for sequential composition with wrong budget split.

    BUG: Each query uses the FULL budget ε instead of ε/k.

    Args:
        num_queries: Number of queries k.
        sensitivity: Per-query sensitivity.
        total_epsilon: Claimed total privacy budget.
        name: Function name.

    Returns:
        Python source with DPImp annotations.
    """
    # BUG: using full epsilon for each query
    scale = sensitivity / total_epsilon
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
# @dp.mechanism(privacy="{total_epsilon}-dp", composition="sequential")
def {name}(db, {", ".join(f"query_{i}" for i in range(num_queries))}):
    """Sequential composition with WRONG budget split.

    BUG: Each query uses full ε = {total_epsilon} instead of ε/k.
    True cost = {num_queries}ε = {num_queries * total_epsilon}.
    """
{body}
    return ({returns})
'''


def budget_tracker_source(
    max_queries: int = 10,
    sensitivity: float = 1.0,
    total_epsilon: float = 1.0,
    name: str = "budget_tracker",
) -> str:
    """Generate source for a mechanism with explicit budget tracking.

    Answers queries until the budget runs out, then refuses.

    Args:
        max_queries: Maximum number of queries to handle.
        sensitivity: Per-query sensitivity.
        total_epsilon: Total privacy budget.
        name: Function name.

    Returns:
        Python source with DPImp annotations.
    """
    eps_per = round(total_epsilon / max_queries, 6)
    scale = round(sensitivity / eps_per, 6)
    return f'''
# @dp.mechanism(privacy="{total_epsilon}-dp")
def {name}(db, queries):
    """Privacy budget tracker: stops answering when budget exhausted.

    Per-query budget: {eps_per}.  Max queries: {max_queries}.
    """
    eps_remaining = {total_epsilon}
    eps_per_query = {eps_per}
    results = []
    for i in range(len(queries)):
        if eps_remaining < eps_per_query:
            # Budget exhausted
            results.append(None)
        else:
            # @dp.sensitivity({sensitivity})
            ans = queries[i](db)
            # @dp.noise(kind="laplace", scale={scale})
            noise = laplace(0, {scale})
            results.append(ans + noise)
            eps_remaining = eps_remaining - eps_per_query
    return results
'''


# ---------------------------------------------------------------------------
# Pre-built source constants
# ---------------------------------------------------------------------------

SEQUENTIAL_CORRECT = sequential_composition_source(
    num_queries=3, total_epsilon=3.0
)
"""Correct sequential composition of 3 queries, total ε=3."""

PARALLEL_CORRECT = parallel_composition_source(
    num_partitions=3, epsilon=1.0
)
"""Correct parallel composition over 3 disjoint partitions."""

ADAPTIVE_CORRECT = adaptive_composition_source(
    num_rounds=5, total_epsilon=1.0, delta_prime=1e-5
)
"""Correct adaptive composition using advanced composition theorem."""

SEQUENTIAL_WRONG_SPLIT = sequential_wrong_split_source(
    num_queries=3, total_epsilon=1.0
)
"""Buggy sequential composition: each query uses full ε."""

BUDGET_TRACKER = budget_tracker_source(
    max_queries=10, total_epsilon=1.0
)
"""Correct budget-tracking mechanism."""


# ---------------------------------------------------------------------------
# Benchmark configurations
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CompositionBenchConfig:
    """Configuration for a single composition benchmark."""

    name: str
    variant: str  # "sequential", "parallel", "adaptive", "wrong_split"
    num_steps: int
    epsilon: float
    is_correct: bool
    description: str = ""

    def source(self) -> str:
        """Generate the mechanism source for this configuration."""
        if self.variant == "sequential":
            return sequential_composition_source(
                num_queries=self.num_steps, total_epsilon=self.epsilon,
                name=self.name,
            )
        elif self.variant == "parallel":
            return parallel_composition_source(
                num_partitions=self.num_steps, epsilon=self.epsilon,
                name=self.name,
            )
        elif self.variant == "adaptive":
            return adaptive_composition_source(
                num_rounds=self.num_steps, total_epsilon=self.epsilon,
                name=self.name,
            )
        else:
            return sequential_wrong_split_source(
                num_queries=self.num_steps, total_epsilon=self.epsilon,
                name=self.name,
            )


COMPOSITION_BENCH_CONFIGS: list[CompositionBenchConfig] = [
    CompositionBenchConfig("seq_2", "sequential", 2, 2.0, True, "2-query seq."),
    CompositionBenchConfig("seq_3", "sequential", 3, 3.0, True, "3-query seq."),
    CompositionBenchConfig("seq_5", "sequential", 5, 5.0, True, "5-query seq."),
    CompositionBenchConfig("seq_10", "sequential", 10, 10.0, True, "10-query seq."),
    CompositionBenchConfig("par_3", "parallel", 3, 1.0, True, "3-partition par."),
    CompositionBenchConfig("par_5", "parallel", 5, 1.0, True, "5-partition par."),
    CompositionBenchConfig("adv_5", "adaptive", 5, 1.0, True, "5-round adaptive"),
    CompositionBenchConfig("adv_10", "adaptive", 10, 1.0, True, "10-round adaptive"),
    CompositionBenchConfig("bug_split", "wrong_split", 3, 1.0, False, "Wrong split"),
]
"""Benchmark sweep configurations for composition mechanisms."""
