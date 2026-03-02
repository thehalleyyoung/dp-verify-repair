"""
Tier 2 benchmarks: composed mechanism verification.

Tests composition of DP mechanisms at various depths:
  - Sequential composition (2, 3, 5, 10, 20, 50 queries)
  - Parallel composition (2, 3, 5 partitions)
  - Adaptive composition (5, 10, 20 rounds)
  - SVT variants (different query counts)
  - Scaling analysis: how does verification time grow with composition depth?

Run:
    python -m benchmarks.benchmark_runner --suite tier2
"""

from __future__ import annotations

from benchmarks.benchmark_runner import BenchmarkCase
from examples.mechanisms.composed_mechanism import (
    sequential_composition_source,
    parallel_composition_source,
    adaptive_composition_source,
    sequential_wrong_split_source,
    budget_tracker_source,
    COMPOSITION_BENCH_CONFIGS,
)
from examples.mechanisms.sparse_vector import (
    sparse_vector_source,
    numeric_sparse_source,
    svt_bug1_no_threshold_noise,
    svt_bug2_reuse_threshold,
    svt_bug4_no_halt,
    SVT_BENCH_CONFIGS,
)
from examples.mechanisms.iterative_mechanism import (
    noisy_gradient_descent_source,
    dp_sgd_source,
    noisy_gd_wrong_composition_source,
    private_mean_estimation_source,
    ITERATIVE_BENCH_CONFIGS,
)


# ---------------------------------------------------------------------------
# Sequential composition scaling benchmarks
# ---------------------------------------------------------------------------

def _sequential_scaling_cases() -> list[BenchmarkCase]:
    """Generate sequential composition scaling benchmarks.

    Varies the number of composed queries from 2 to 50 to measure
    how verification time scales with composition depth.
    """
    cases: list[BenchmarkCase] = []
    depths = [2, 3, 5, 8, 10, 15, 20, 30, 50]

    for depth in depths:
        eps = float(depth)  # each query uses ε=1
        source = sequential_composition_source(
            num_queries=depth,
            total_epsilon=eps,
            name=f"seq_{depth}",
        )
        cases.append(BenchmarkCase(
            id=f"seq-scale-{depth}",
            name=f"Sequential ×{depth}",
            category="tier2-sequential-scaling",
            source=source,
            mechanism_name=f"seq_{depth}",
            privacy_notion="pure_dp",
            epsilon=eps,
            expected_verified=True,
            timeout=300.0,
            description=f"Sequential composition of {depth} Laplace queries.",
            tags=["composition", "sequential", "scaling"],
        ))

    return cases


# ---------------------------------------------------------------------------
# Parallel composition benchmarks
# ---------------------------------------------------------------------------

def _parallel_cases() -> list[BenchmarkCase]:
    """Generate parallel composition benchmark cases."""
    cases: list[BenchmarkCase] = []
    partitions = [2, 3, 5, 8, 10]

    for n_parts in partitions:
        source = parallel_composition_source(
            num_partitions=n_parts,
            epsilon=1.0,
            name=f"par_{n_parts}",
        )
        cases.append(BenchmarkCase(
            id=f"par-{n_parts}",
            name=f"Parallel ×{n_parts}",
            category="tier2-parallel",
            source=source,
            mechanism_name=f"par_{n_parts}",
            privacy_notion="pure_dp",
            epsilon=1.0,
            expected_verified=True,
            description=f"Parallel composition over {n_parts} disjoint partitions.",
            tags=["composition", "parallel"],
        ))

    return cases


# ---------------------------------------------------------------------------
# Adaptive composition benchmarks
# ---------------------------------------------------------------------------

def _adaptive_cases() -> list[BenchmarkCase]:
    """Generate adaptive composition benchmark cases."""
    cases: list[BenchmarkCase] = []
    rounds_list = [3, 5, 10, 20]

    for n_rounds in rounds_list:
        source = adaptive_composition_source(
            num_rounds=n_rounds,
            total_epsilon=1.0,
            delta_prime=1e-5,
            name=f"adaptive_{n_rounds}",
        )
        cases.append(BenchmarkCase(
            id=f"adv-{n_rounds}",
            name=f"Adaptive ×{n_rounds}",
            category="tier2-adaptive",
            source=source,
            mechanism_name=f"adaptive_{n_rounds}",
            privacy_notion="approx_dp",
            epsilon=1.0,
            delta=1e-5,
            expected_verified=True,
            timeout=300.0,
            description=f"Adaptive composition with {n_rounds} rounds.",
            tags=["composition", "adaptive", "advanced"],
        ))

    return cases


# ---------------------------------------------------------------------------
# Wrong composition benchmarks (buggy)
# ---------------------------------------------------------------------------

def _wrong_composition_cases() -> list[BenchmarkCase]:
    """Generate composition benchmarks with known bugs."""
    cases: list[BenchmarkCase] = []

    for depth in [2, 3, 5, 10]:
        source = sequential_wrong_split_source(
            num_queries=depth,
            total_epsilon=1.0,
            name=f"seq_wrong_{depth}",
        )
        cases.append(BenchmarkCase(
            id=f"seq-wrong-{depth}",
            name=f"Wrong split ×{depth}",
            category="tier2-wrong-composition",
            source=source,
            mechanism_name=f"seq_wrong_{depth}",
            privacy_notion="pure_dp",
            epsilon=1.0,
            expected_verified=False,
            description=f"Sequential ×{depth} using full ε per query.",
            tags=["composition", "buggy", "wrong_split"],
        ))

    return cases


# ---------------------------------------------------------------------------
# SVT scaling benchmarks
# ---------------------------------------------------------------------------

def _svt_scaling_cases() -> list[BenchmarkCase]:
    """Generate SVT benchmarks with varying query counts."""
    cases: list[BenchmarkCase] = []
    query_counts = [3, 5, 10, 20, 50]

    for n_q in query_counts:
        # Correct SVT
        source = sparse_vector_source(
            num_queries=n_q, epsilon=1.0, name=f"svt_q{n_q}",
        )
        cases.append(BenchmarkCase(
            id=f"svt-correct-{n_q}q",
            name=f"SVT correct ({n_q} queries)",
            category="tier2-svt-scaling",
            source=source,
            mechanism_name=f"svt_q{n_q}",
            privacy_notion="pure_dp",
            epsilon=1.0,
            expected_verified=True,
            timeout=300.0,
            description=f"Correct SVT with {n_q} queries.",
            tags=["svt", "correct", "scaling"],
        ))

        # Buggy SVT (no halt)
        source_bug = svt_bug4_no_halt(
            num_queries=n_q, epsilon=1.0, name=f"svt_b4_q{n_q}",
        )
        cases.append(BenchmarkCase(
            id=f"svt-bug4-{n_q}q",
            name=f"SVT bug4 ({n_q} queries)",
            category="tier2-svt-scaling",
            source=source_bug,
            mechanism_name=f"svt_b4_q{n_q}",
            privacy_notion="pure_dp",
            epsilon=1.0,
            expected_verified=False,
            timeout=300.0,
            description=f"SVT bug 4 (no halt) with {n_q} queries.",
            tags=["svt", "buggy", "scaling"],
        ))

    # Numeric sparse
    for n_q in [5, 10, 20]:
        source = numeric_sparse_source(
            num_queries=n_q, epsilon=1.0, name=f"numsparse_q{n_q}",
        )
        cases.append(BenchmarkCase(
            id=f"numsparse-{n_q}q",
            name=f"NumericSparse ({n_q} q)",
            category="tier2-svt-scaling",
            source=source,
            mechanism_name=f"numsparse_q{n_q}",
            privacy_notion="pure_dp",
            epsilon=1.0,
            expected_verified=True,
            timeout=300.0,
            description=f"Numeric Sparse with {n_q} queries.",
            tags=["svt", "numeric_sparse", "scaling"],
        ))

    return cases


# ---------------------------------------------------------------------------
# Iterative mechanism scaling benchmarks
# ---------------------------------------------------------------------------

def _iterative_scaling_cases() -> list[BenchmarkCase]:
    """Generate iterative mechanism scaling benchmarks."""
    cases: list[BenchmarkCase] = []
    iteration_counts = [3, 5, 10, 20, 50]

    for n_iter in iteration_counts:
        source = noisy_gradient_descent_source(
            num_iterations=n_iter, epsilon=1.0, delta=1e-5,
            name=f"gd_{n_iter}",
        )
        cases.append(BenchmarkCase(
            id=f"gd-scale-{n_iter}",
            name=f"Noisy GD ×{n_iter}",
            category="tier2-iterative-scaling",
            source=source,
            mechanism_name=f"gd_{n_iter}",
            privacy_notion="approx_dp",
            epsilon=1.0,
            delta=1e-5,
            expected_verified=True,
            timeout=300.0,
            description=f"Noisy GD with {n_iter} iterations.",
            tags=["iterative", "noisy_gd", "scaling"],
        ))

    # Wrong composition iterative
    for n_iter in [5, 10, 20]:
        source = noisy_gd_wrong_composition_source(
            num_iterations=n_iter, epsilon=1.0,
            name=f"gd_wrong_{n_iter}",
        )
        cases.append(BenchmarkCase(
            id=f"gd-wrong-{n_iter}",
            name=f"Wrong comp GD ×{n_iter}",
            category="tier2-iterative-scaling",
            source=source,
            mechanism_name=f"gd_wrong_{n_iter}",
            privacy_notion="approx_dp",
            epsilon=1.0,
            delta=1e-5,
            expected_verified=False,
            description=f"Wrong composition GD with {n_iter} iterations.",
            tags=["iterative", "buggy", "wrong_composition"],
        ))

    return cases


# ---------------------------------------------------------------------------
# Budget tracker benchmarks
# ---------------------------------------------------------------------------

def _budget_tracker_cases() -> list[BenchmarkCase]:
    """Generate budget tracker benchmark cases."""
    cases: list[BenchmarkCase] = []

    for max_q in [5, 10, 20]:
        source = budget_tracker_source(
            max_queries=max_q, epsilon=1.0, name=f"budget_{max_q}",
        )
        cases.append(BenchmarkCase(
            id=f"budget-{max_q}",
            name=f"Budget tracker ×{max_q}",
            category="tier2-budget",
            source=source,
            mechanism_name=f"budget_{max_q}",
            privacy_notion="pure_dp",
            epsilon=1.0,
            expected_verified=True,
            description=f"Budget tracker with {max_q} max queries.",
            tags=["composition", "budget_tracking"],
        ))

    return cases


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_tier2_cases() -> list[BenchmarkCase]:
    """Return all Tier 2 benchmark cases.

    Returns:
        List of BenchmarkCase objects covering composition benchmarks.
    """
    cases: list[BenchmarkCase] = []
    cases.extend(_sequential_scaling_cases())
    cases.extend(_parallel_cases())
    cases.extend(_adaptive_cases())
    cases.extend(_wrong_composition_cases())
    cases.extend(_svt_scaling_cases())
    cases.extend(_iterative_scaling_cases())
    cases.extend(_budget_tracker_cases())
    return cases


# ---------------------------------------------------------------------------
# Stand-alone execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cases = get_tier2_cases()
    print(f"Tier 2: {len(cases)} benchmark cases")
    by_cat: dict[str, list[BenchmarkCase]] = {}
    for c in cases:
        by_cat.setdefault(c.category, []).append(c)
    for cat, cat_cases in sorted(by_cat.items()):
        print(f"\n  {cat} ({len(cat_cases)} cases):")
        for c in cat_cases:
            exp = "✓" if c.expected_verified else "✗"
            print(f"    [{exp}] {c.id:<30} {c.name}")
