"""
Sparse Vector Technique (SVT) implementations for DP-CEGAR verification.

The Sparse Vector Technique answers a stream of threshold queries using a
fixed privacy budget, outputting only whether each query answer is above or
below a noisy threshold.  It is one of the most subtle DP algorithms and has
been the subject of numerous buggy implementations in the literature.

Variants:
  - Above Threshold (basic SVT)
  - Numeric Sparse (returns noisy answers for above-threshold queries)
  - Multiple known-buggy variants (Lyu et al., 2017)

References:
  Dwork & Roth. "The Algorithmic Foundations of DP." 2014, §3.6.
  Lyu, Su, Li. "Understanding the Sparse Vector Technique for DP." VLDB 2017.
"""

from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Source generators
# ---------------------------------------------------------------------------

def sparse_vector_source(
    threshold: float = 1.0,
    epsilon: float = 1.0,
    max_above: int = 1,
    num_queries: int = 5,
    name: str = "above_threshold",
) -> str:
    """Generate source for the correct Above Threshold (SVT) algorithm.

    Splits the budget: ε₁ = ε/2 for the threshold noise, ε₂ = ε/2 for
    per-query noise.  Halts after ``max_above`` above-threshold answers.

    Args:
        threshold: Public threshold T.
        epsilon: Total privacy budget ε.
        max_above: Maximum number of above-threshold answers before halting (c).
        num_queries: Number of queries in the stream.
        name: Function name in generated source.

    Returns:
        Python source string with DPImp annotations.
    """
    eps1 = epsilon / 2.0
    eps2 = epsilon / 2.0
    scale_t = 1.0 / eps1
    scale_q = 1.0 / eps2
    return f'''
# @dp.mechanism(privacy="{epsilon}-dp", sensitivity=1)
def {name}(db, queries, T={threshold}):
    """Above Threshold (correct SVT).

    Budget split: ε₁ = {eps1} (threshold), ε₂ = {eps2} (queries).
    Halts after {max_above} above-threshold answers.
    """
    c = {max_above}
    count = 0
    out = []
    # @dp.noise(kind="laplace", scale={scale_t})
    noisy_T = T + laplace(0, {scale_t})
    for i in range({num_queries}):
        # @dp.sensitivity(1)
        q_val = queries[i](db)
        # @dp.noise(kind="laplace", scale={scale_q})
        nu_i = laplace(0, {scale_q})
        if q_val + nu_i >= noisy_T:
            out.append(True)
            count = count + 1
            if count >= c:
                break
        else:
            out.append(False)
    return out
'''


def numeric_sparse_source(
    threshold: float = 1.0,
    epsilon: float = 1.0,
    max_above: int = 1,
    num_queries: int = 5,
    name: str = "numeric_sparse",
) -> str:
    """Generate source for the Numeric Sparse algorithm.

    Like Above Threshold, but returns noisy answers for above-threshold
    queries.  Uses a three-way budget split: ε₁, ε₂, ε₃.

    Args:
        threshold: Public threshold T.
        epsilon: Total privacy budget ε.
        max_above: Maximum above-threshold count c.
        num_queries: Number of queries in the stream.
        name: Function name in generated source.

    Returns:
        Python source string with DPImp annotations.
    """
    eps1 = epsilon / 3.0
    eps2 = epsilon / 3.0
    eps3 = epsilon / 3.0
    scale_t = 1.0 / eps1
    scale_q = 1.0 / eps2
    scale_a = 1.0 / eps3
    scale_t_r = round(scale_t, 6)
    scale_q_r = round(scale_q, 6)
    scale_a_r = round(scale_a, 6)
    return f'''
# @dp.mechanism(privacy="{epsilon}-dp", sensitivity=1)
def {name}(db, queries, T={threshold}):
    """Numeric Sparse: returns noisy values for above-threshold queries.

    Three-way budget split: ε₁=ε₂=ε₃ = ε/3.
    """
    c = {max_above}
    count = 0
    out = []
    # @dp.noise(kind="laplace", scale={scale_t_r})
    noisy_T = T + laplace(0, {scale_t_r})
    for i in range({num_queries}):
        # @dp.sensitivity(1)
        q_val = queries[i](db)
        # @dp.noise(kind="laplace", scale={scale_q_r})
        nu_i = laplace(0, {scale_q_r})
        if q_val + nu_i >= noisy_T:
            # @dp.noise(kind="laplace", scale={scale_a_r})
            noisy_ans = q_val + laplace(0, {scale_a_r})
            out.append(noisy_ans)
            count = count + 1
            if count >= c:
                break
        else:
            out.append(False)
    return out
'''


# ---------------------------------------------------------------------------
# Known buggy variants (Lyu et al., 2017)
# ---------------------------------------------------------------------------

def svt_bug1_no_threshold_noise(
    num_queries: int = 5,
    epsilon: float = 1.0,
    name: str = "svt_bug1",
) -> str:
    """SVT Bug 1: Missing noise on the threshold.

    The threshold is used raw without adding Laplace noise.
    This breaks privacy because the threshold comparison leaks information.

    Reference: Lyu et al. VLDB 2017, Bug #1.

    Args:
        num_queries: Number of queries.
        epsilon: Privacy budget.
        name: Function name.

    Returns:
        Python source with DPImp annotations.
    """
    scale_q = 2.0 / epsilon
    return f'''
# @dp.mechanism(privacy="{epsilon}-dp", sensitivity=1)
def {name}(db, queries, T=1.0):
    """SVT BUG 1: Missing noise on threshold.

    The threshold T is used without noise. This leaks information
    about whether q(db) is above T, violating DP.
    """
    count = 0
    out = []
    # BUG: no noise on threshold!
    noisy_T = T
    for i in range({num_queries}):
        # @dp.sensitivity(1)
        q_val = queries[i](db)
        # @dp.noise(kind="laplace", scale={scale_q})
        nu_i = laplace(0, {scale_q})
        if q_val + nu_i >= noisy_T:
            out.append(True)
            count = count + 1
            if count >= 1:
                break
        else:
            out.append(False)
    return out
'''


def svt_bug2_reuse_threshold(
    num_queries: int = 5,
    epsilon: float = 1.0,
    name: str = "svt_bug2",
) -> str:
    """SVT Bug 2: Fresh threshold noise on every query.

    Instead of drawing threshold noise once, this variant draws fresh
    noise per query.  This effectively composes the threshold noise
    across queries, consuming far more budget than intended.

    Reference: Lyu et al. VLDB 2017, Bug #2.

    Args:
        num_queries: Number of queries.
        epsilon: Privacy budget.
        name: Function name.

    Returns:
        Python source with DPImp annotations.
    """
    eps1 = epsilon / 2.0
    eps2 = epsilon / 2.0
    scale_t = 1.0 / eps1
    scale_q = 1.0 / eps2
    return f'''
# @dp.mechanism(privacy="{epsilon}-dp", sensitivity=1)
def {name}(db, queries, T=1.0):
    """SVT BUG 2: Fresh threshold noise per query (should be drawn once).

    Drawing fresh threshold noise in each iteration means the threshold
    noise composes across queries, breaking the budget analysis.
    """
    count = 0
    out = []
    for i in range({num_queries}):
        # BUG: threshold noise re-drawn each iteration!
        # @dp.noise(kind="laplace", scale={scale_t})
        noisy_T = T + laplace(0, {scale_t})
        # @dp.sensitivity(1)
        q_val = queries[i](db)
        # @dp.noise(kind="laplace", scale={scale_q})
        nu_i = laplace(0, {scale_q})
        if q_val + nu_i >= noisy_T:
            out.append(True)
            count = count + 1
            if count >= 1:
                break
        else:
            out.append(False)
    return out
'''


def svt_bug3_wrong_sensitivity(
    num_queries: int = 5,
    epsilon: float = 1.0,
    name: str = "svt_bug3",
) -> str:
    """SVT Bug 3: Wrong sensitivity in query noise.

    Uses sensitivity 2 instead of 1 for query noise calibration,
    wasting budget (or: uses sensitivity 1 when true sensitivity is 2).
    In the common formulation, the query noise uses scale 2/ε₂ to account
    for the difference query structure; this variant uses 1/ε₂.

    Reference: Lyu et al. VLDB 2017, Bug #3.

    Args:
        num_queries: Number of queries.
        epsilon: Privacy budget.
        name: Function name.

    Returns:
        Python source with DPImp annotations.
    """
    eps1 = epsilon / 2.0
    eps2 = epsilon / 2.0
    scale_t = 1.0 / eps1
    wrong_scale_q = 1.0 / eps2  # BUG: should be 2.0 / eps2 for sensitivity-2 queries
    return f'''
# @dp.mechanism(privacy="{epsilon}-dp", sensitivity=1)
def {name}(db, queries, T=1.0):
    """SVT BUG 3: Wrong sensitivity in query noise.

    Uses scale={wrong_scale_q} for query noise instead of the correct
    scale based on query sensitivity.
    """
    count = 0
    out = []
    # @dp.noise(kind="laplace", scale={scale_t})
    noisy_T = T + laplace(0, {scale_t})
    for i in range({num_queries}):
        # @dp.sensitivity(1)
        q_val = queries[i](db)
        # BUG: wrong sensitivity → wrong scale
        # @dp.noise(kind="laplace", scale={wrong_scale_q})
        nu_i = laplace(0, {wrong_scale_q})
        if q_val + nu_i >= noisy_T:
            out.append(True)
            count = count + 1
            if count >= 1:
                break
        else:
            out.append(False)
    return out
'''


def svt_bug4_no_halt(
    num_queries: int = 5,
    epsilon: float = 1.0,
    max_above: int = 1,
    name: str = "svt_bug4",
) -> str:
    """SVT Bug 4: Not halting after c above-threshold answers.

    The algorithm should stop after c above-threshold answers, but this
    variant continues answering all queries.  This breaks the privacy
    analysis which relies on bounded above-threshold answers.

    Reference: Lyu et al. VLDB 2017, Bug #4.

    Args:
        num_queries: Number of queries.
        epsilon: Privacy budget.
        max_above: c value (ignored — bug is that we don't halt).
        name: Function name.

    Returns:
        Python source with DPImp annotations.
    """
    eps1 = epsilon / 2.0
    eps2 = epsilon / 2.0
    scale_t = 1.0 / eps1
    scale_q = 1.0 / eps2
    return f'''
# @dp.mechanism(privacy="{epsilon}-dp", sensitivity=1)
def {name}(db, queries, T=1.0):
    """SVT BUG 4: Does not halt after c above-threshold answers.

    Missing the count check and break means all queries are answered,
    violating the budget analysis.
    """
    out = []
    # @dp.noise(kind="laplace", scale={scale_t})
    noisy_T = T + laplace(0, {scale_t})
    for i in range({num_queries}):
        # @dp.sensitivity(1)
        q_val = queries[i](db)
        # @dp.noise(kind="laplace", scale={scale_q})
        nu_i = laplace(0, {scale_q})
        if q_val + nu_i >= noisy_T:
            out.append(True)
            # BUG: no count tracking, no halt!
        else:
            out.append(False)
    return out
'''


def svt_bug5_wrong_budget(
    num_queries: int = 5,
    epsilon: float = 1.0,
    name: str = "svt_bug5",
) -> str:
    """SVT Bug 5: Wrong privacy budget allocation.

    Allocates the entire budget ε to both the threshold noise and the
    query noise, effectively using 2ε total.

    Reference: Lyu et al. VLDB 2017, Bug #5.

    Args:
        num_queries: Number of queries.
        epsilon: Privacy budget.
        name: Function name.

    Returns:
        Python source with DPImp annotations.
    """
    scale_t = 1.0 / epsilon  # BUG: uses full ε instead of ε/2
    scale_q = 1.0 / epsilon  # BUG: uses full ε instead of ε/2
    return f'''
# @dp.mechanism(privacy="{epsilon}-dp", sensitivity=1)
def {name}(db, queries, T=1.0):
    """SVT BUG 5: Wrong budget allocation (uses ε for both noise draws).

    Both threshold and query noise use scale 1/ε instead of 2/ε.
    Total privacy cost is 2ε, not ε.
    """
    count = 0
    out = []
    # BUG: scale should be 2/ε
    # @dp.noise(kind="laplace", scale={scale_t})
    noisy_T = T + laplace(0, {scale_t})
    for i in range({num_queries}):
        # @dp.sensitivity(1)
        q_val = queries[i](db)
        # BUG: scale should be 2/ε
        # @dp.noise(kind="laplace", scale={scale_q})
        nu_i = laplace(0, {scale_q})
        if q_val + nu_i >= noisy_T:
            out.append(True)
            count = count + 1
            if count >= 1:
                break
        else:
            out.append(False)
    return out
'''


def svt_bug6_output_value(
    num_queries: int = 5,
    epsilon: float = 1.0,
    name: str = "svt_bug6",
) -> str:
    """SVT Bug 6: Outputting noisy query value instead of ⊤/⊥.

    Above Threshold should only output True/False, but this variant
    leaks the noisy query value when above threshold.

    Reference: Lyu et al. VLDB 2017, Bug #6 / variant.

    Args:
        num_queries: Number of queries.
        epsilon: Privacy budget.
        name: Function name.

    Returns:
        Python source with DPImp annotations.
    """
    eps1 = epsilon / 2.0
    eps2 = epsilon / 2.0
    scale_t = 1.0 / eps1
    scale_q = 1.0 / eps2
    return f'''
# @dp.mechanism(privacy="{epsilon}-dp", sensitivity=1)
def {name}(db, queries, T=1.0):
    """SVT BUG 6: Outputs noisy query value instead of True/False.

    Leaking the noisy value q(db)+ν requires additional privacy budget
    that is not accounted for.
    """
    count = 0
    out = []
    # @dp.noise(kind="laplace", scale={scale_t})
    noisy_T = T + laplace(0, {scale_t})
    for i in range({num_queries}):
        # @dp.sensitivity(1)
        q_val = queries[i](db)
        # @dp.noise(kind="laplace", scale={scale_q})
        nu_i = laplace(0, {scale_q})
        noisy_val = q_val + nu_i
        if noisy_val >= noisy_T:
            # BUG: outputs the noisy value, not just True
            out.append(noisy_val)
            count = count + 1
            if count >= 1:
                break
        else:
            out.append(False)
    return out
'''


# ---------------------------------------------------------------------------
# Pre-built source constants
# ---------------------------------------------------------------------------

SVT_CORRECT = sparse_vector_source(
    threshold=1.0, epsilon=1.0, max_above=1, num_queries=5
)
"""Correct Above Threshold (SVT) with T=1, ε=1, c=1."""

NUMERIC_SPARSE_CORRECT = numeric_sparse_source(
    threshold=1.0, epsilon=1.0, max_above=1, num_queries=5
)
"""Correct Numeric Sparse variant."""

SVT_BUG1_NO_THRESHOLD_NOISE = svt_bug1_no_threshold_noise()
"""Bug 1: No noise on threshold."""

SVT_BUG2_REUSE_THRESHOLD = svt_bug2_reuse_threshold()
"""Bug 2: Fresh threshold noise per query."""

SVT_BUG3_WRONG_SENSITIVITY = svt_bug3_wrong_sensitivity()
"""Bug 3: Wrong sensitivity in query noise."""

SVT_BUG4_NO_HALT = svt_bug4_no_halt()
"""Bug 4: No halt after c above-threshold answers."""

SVT_BUG5_WRONG_BUDGET = svt_bug5_wrong_budget()
"""Bug 5: Wrong budget allocation."""

SVT_BUG6_OUTPUT_VALUE = svt_bug6_output_value()
"""Bug 6: Leaks noisy query value."""


# ---------------------------------------------------------------------------
# Benchmark configurations
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SVTBenchConfig:
    """Configuration for a single SVT benchmark instance."""

    name: str
    source_fn: str
    is_correct: bool
    bug_description: str = ""
    num_queries: int = 5
    epsilon: float = 1.0

    def source(self) -> str:
        """Generate the mechanism source for this configuration."""
        fn = globals()[self.source_fn]
        if self.source_fn in ("sparse_vector_source", "numeric_sparse_source"):
            return fn(
                num_queries=self.num_queries,
                epsilon=self.epsilon,
                name=self.name,
            )
        return fn(
            num_queries=self.num_queries,
            epsilon=self.epsilon,
            name=self.name,
        )


SVT_BENCH_CONFIGS: list[SVTBenchConfig] = [
    SVTBenchConfig("svt_correct_5", "sparse_vector_source", True, "Correct SVT"),
    SVTBenchConfig("svt_correct_10", "sparse_vector_source", True,
                   "Correct SVT, 10 queries", num_queries=10),
    SVTBenchConfig("svt_numeric", "numeric_sparse_source", True, "Numeric Sparse"),
    SVTBenchConfig("svt_b1", "svt_bug1_no_threshold_noise", False,
                   "No threshold noise"),
    SVTBenchConfig("svt_b2", "svt_bug2_reuse_threshold", False,
                   "Reused threshold noise"),
    SVTBenchConfig("svt_b3", "svt_bug3_wrong_sensitivity", False,
                   "Wrong sensitivity"),
    SVTBenchConfig("svt_b4", "svt_bug4_no_halt", False,
                   "No halt after c answers"),
    SVTBenchConfig("svt_b5", "svt_bug5_wrong_budget", False,
                   "Wrong budget split"),
    SVTBenchConfig("svt_b6", "svt_bug6_output_value", False,
                   "Leaks noisy value"),
]
"""Benchmark sweep configurations for SVT variants."""
