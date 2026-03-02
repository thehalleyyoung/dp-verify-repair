"""
Tier 4 benchmarks: stress tests.

Pushes the DP-CEGAR verification engine to its limits:
  - Large loop bounds (high unroll counts)
  - Many branches (exponential path explosion)
  - Complex noise compositions
  - Deep nesting
  - Timeout behaviour evaluation
  - Memory pressure tests

Run:
    python -m benchmarks.benchmark_runner --suite tier4
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from benchmarks.benchmark_runner import BenchmarkCase
from examples.mechanisms.composed_mechanism import (
    sequential_composition_source,
    adaptive_composition_source,
    sequential_wrong_split_source,
)
from examples.mechanisms.sparse_vector import sparse_vector_source
from examples.mechanisms.iterative_mechanism import (
    noisy_gradient_descent_source,
    dp_sgd_source,
)
from examples.mechanisms.noisy_histogram import noisy_histogram_source


# ---------------------------------------------------------------------------
# Large loop bound benchmarks
# ---------------------------------------------------------------------------

def _large_loop_mechanism(
    iterations: int,
    name: str = "large_loop",
) -> str:
    """Generate a mechanism with a large bounded loop.

    The mechanism iterates `iterations` times, adding noise each time.

    Args:
        iterations: Number of loop iterations.
        name: Function name.

    Returns:
        Python source string.
    """
    eps_per = 1.0 / iterations
    scale = round(1.0 / eps_per, 6)
    return f'''
# @dp.mechanism(privacy="1.0-dp", sensitivity=1)
def {name}(db, queries):
    """Mechanism with {iterations} loop iterations.

    Per-iteration ε = {eps_per:.6f}, scale = {scale}.
    Tests verification with large loop bounds.
    """
    results = []
    for i in range({iterations}):
        # @dp.sensitivity(1)
        ans = queries[i](db)
        # @dp.noise(kind="laplace", scale={scale})
        noise = laplace(0, {scale})
        results.append(ans + noise)
    return results
'''


def _large_loop_cases() -> list[BenchmarkCase]:
    """Generate large-loop stress benchmarks."""
    cases: list[BenchmarkCase] = []
    loop_sizes = [10, 20, 50, 100, 200, 500]

    for n in loop_sizes:
        source = _large_loop_mechanism(n, name=f"loop_{n}")
        timeout = min(600.0, 30.0 + n * 2.0)  # scale timeout
        cases.append(BenchmarkCase(
            id=f"stress-loop-{n}",
            name=f"Large loop ×{n}",
            category="tier4-large-loops",
            source=source,
            mechanism_name=f"loop_{n}",
            privacy_notion="pure_dp",
            epsilon=1.0,
            expected_verified=True,
            timeout=timeout,
            description=f"Mechanism with {n} loop iterations.",
            tags=["stress", "loop", "scaling"],
        ))

    return cases


# ---------------------------------------------------------------------------
# Many-branches benchmarks (path explosion)
# ---------------------------------------------------------------------------

def _branching_mechanism(
    depth: int,
    name: str = "branching",
) -> str:
    """Generate a mechanism with nested branching.

    Creates a binary tree of if-else branches, each with noise draws.
    Total paths = 2^depth.

    Args:
        depth: Nesting depth (number of conditional levels).
        name: Function name.

    Returns:
        Python source string.
    """
    total_paths = 2 ** depth
    # Per-path budget: allocate across worst-case path (depth noise draws)
    eps_per_draw = 1.0 / depth if depth > 0 else 1.0
    scale = round(1.0 / eps_per_draw, 6)

    def _build_branch(level: int, indent: int) -> str:
        pad = "    " * indent
        if level >= depth:
            return (
                f"{pad}# @dp.noise(kind=\"laplace\", scale={scale})\n"
                f"{pad}result = result + laplace(0, {scale})"
            )
        return (
            f"{pad}# @dp.sensitivity(1)\n"
            f"{pad}cond_{level} = query_{level}(db)\n"
            f"{pad}if cond_{level} > 0:\n"
            f"{pad}    # @dp.noise(kind=\"laplace\", scale={scale})\n"
            f"{pad}    result = result + laplace(0, {scale})\n"
            + _build_branch(level + 1, indent + 1) + "\n"
            + f"{pad}else:\n"
            f"{pad}    # @dp.noise(kind=\"laplace\", scale={scale})\n"
            f"{pad}    result = result + laplace(0, {scale})\n"
            + _build_branch(level + 1, indent + 1)
        )

    body = _build_branch(0, 1)
    query_args = ", ".join(f"query_{i}" for i in range(depth))
    return f'''
# @dp.mechanism(privacy="1.0-dp", sensitivity=1)
def {name}(db, {query_args}):
    """Mechanism with {depth} levels of branching ({total_paths} paths).

    Tests path explosion in verification.
    """
    result = 0
{body}
    return result
'''


def _branching_cases() -> list[BenchmarkCase]:
    """Generate branching stress benchmarks."""
    cases: list[BenchmarkCase] = []
    depths = [2, 3, 4, 5, 6, 7, 8, 10]

    for depth in depths:
        total_paths = 2 ** depth
        source = _branching_mechanism(depth, name=f"branch_{depth}")
        timeout = min(600.0, 30.0 + total_paths * 0.5)
        cases.append(BenchmarkCase(
            id=f"stress-branch-{depth}",
            name=f"Branch depth {depth} ({total_paths} paths)",
            category="tier4-branching",
            source=source,
            mechanism_name=f"branch_{depth}",
            privacy_notion="pure_dp",
            epsilon=1.0,
            expected_verified=True,
            timeout=timeout,
            description=f"Nested branching with {depth} levels, {total_paths} paths.",
            tags=["stress", "branching", "path_explosion"],
        ))

    return cases


# ---------------------------------------------------------------------------
# Complex noise composition benchmarks
# ---------------------------------------------------------------------------

def _complex_noise_mechanism(
    num_laplace: int = 3,
    num_gaussian: int = 3,
    name: str = "complex_noise",
) -> str:
    """Generate a mechanism with multiple noise types composed.

    Mixes Laplace and Gaussian noise draws in a single mechanism.

    Args:
        num_laplace: Number of Laplace noise draws.
        num_gaussian: Number of Gaussian noise draws.
        name: Function name.

    Returns:
        Python source string.
    """
    total = num_laplace + num_gaussian
    eps_per = 1.0 / total
    lap_scale = round(1.0 / eps_per, 6)
    gauss_sigma = round(
        math.sqrt(2.0 * math.log(1.25e5)) / eps_per, 6
    )

    noise_draws = []
    for i in range(num_laplace):
        noise_draws.append(f"""
    # @dp.sensitivity(1)
    ans_l{i} = query_l{i}(db)
    # @dp.noise(kind="laplace", scale={lap_scale})
    result_l{i} = ans_l{i} + laplace(0, {lap_scale})""")

    for i in range(num_gaussian):
        noise_draws.append(f"""
    # @dp.sensitivity(1)
    ans_g{i} = query_g{i}(db)
    # @dp.noise(kind="gaussian", sigma={gauss_sigma})
    result_g{i} = ans_g{i} + gaussian(0, {gauss_sigma})""")

    body = "\n".join(noise_draws)
    lap_ret = ", ".join(f"result_l{i}" for i in range(num_laplace))
    gauss_ret = ", ".join(f"result_g{i}" for i in range(num_gaussian))
    all_ret = ", ".join(filter(None, [lap_ret, gauss_ret]))
    query_args = ", ".join(
        [f"query_l{i}" for i in range(num_laplace)]
        + [f"query_g{i}" for i in range(num_gaussian)]
    )

    return f'''
# @dp.mechanism(privacy="(1.0,1e-5)-dp", sensitivity=1)
def {name}(db, {query_args}):
    """Complex noise: {num_laplace} Laplace + {num_gaussian} Gaussian draws.

    Mixed noise composition stress test.
    """
{body}
    return ({all_ret})
'''


def _complex_noise_cases() -> list[BenchmarkCase]:
    """Generate complex noise composition benchmarks."""
    configs = [
        (2, 2, "2L+2G"),
        (3, 3, "3L+3G"),
        (5, 5, "5L+5G"),
        (10, 0, "10L"),
        (0, 10, "10G"),
        (10, 10, "10L+10G"),
        (20, 20, "20L+20G"),
    ]
    cases: list[BenchmarkCase] = []

    for n_lap, n_gauss, label in configs:
        fn_name = f"noise_{label.lower().replace('+', '_')}"
        source = _complex_noise_mechanism(n_lap, n_gauss, name=fn_name)
        cases.append(BenchmarkCase(
            id=f"stress-noise-{label.lower().replace('+', '-')}",
            name=f"Complex noise {label}",
            category="tier4-complex-noise",
            source=source,
            mechanism_name=fn_name,
            privacy_notion="approx_dp",
            epsilon=1.0,
            delta=1e-5,
            expected_verified=True,
            timeout=300.0,
            description=f"Mixed noise composition: {label}.",
            tags=["stress", "complex_noise", "mixed"],
        ))

    return cases


# ---------------------------------------------------------------------------
# Large histogram benchmarks
# ---------------------------------------------------------------------------

def _large_histogram_cases() -> list[BenchmarkCase]:
    """Generate large histogram stress benchmarks."""
    cases: list[BenchmarkCase] = []
    bin_counts = [10, 25, 50, 100, 200, 500]

    for n_bins in bin_counts:
        source = noisy_histogram_source(
            num_bins=n_bins, epsilon=1.0, name=f"hist_{n_bins}",
        )
        timeout = min(600.0, 30.0 + n_bins * 1.0)
        cases.append(BenchmarkCase(
            id=f"stress-hist-{n_bins}",
            name=f"Histogram {n_bins} bins",
            category="tier4-large-histogram",
            source=source,
            mechanism_name=f"hist_{n_bins}",
            privacy_notion="pure_dp",
            epsilon=1.0,
            expected_verified=True,
            timeout=timeout,
            description=f"Noisy histogram with {n_bins} bins.",
            tags=["stress", "histogram", "scaling"],
        ))

    return cases


# ---------------------------------------------------------------------------
# Timeout behaviour benchmarks
# ---------------------------------------------------------------------------

def _timeout_cases() -> list[BenchmarkCase]:
    """Generate benchmarks designed to test timeout behaviour.

    These cases are intentionally difficult/large and should hit the
    timeout limit, allowing us to evaluate graceful degradation.
    """
    cases: list[BenchmarkCase] = []

    # Very large sequential composition
    source = sequential_composition_source(
        num_queries=100, total_epsilon=100.0,
        name="seq_100",
    )
    cases.append(BenchmarkCase(
        id="timeout-seq-100",
        name="Timeout: Seq ×100",
        category="tier4-timeout",
        source=source,
        mechanism_name="seq_100",
        privacy_notion="pure_dp",
        epsilon=100.0,
        expected_verified=True,
        timeout=10.0,  # Very short timeout
        description="Sequential ×100 with 10s timeout.",
        tags=["stress", "timeout"],
    ))

    # Very deep branching
    source = _branching_mechanism(12, name="branch_12")
    cases.append(BenchmarkCase(
        id="timeout-branch-12",
        name="Timeout: Branch depth 12",
        category="tier4-timeout",
        source=source,
        mechanism_name="branch_12",
        privacy_notion="pure_dp",
        epsilon=1.0,
        expected_verified=True,
        timeout=10.0,
        description="12-level branching (4096 paths) with 10s timeout.",
        tags=["stress", "timeout", "path_explosion"],
    ))

    # Large GD
    source = noisy_gradient_descent_source(
        num_iterations=200, epsilon=1.0, delta=1e-5,
        name="gd_200",
    )
    cases.append(BenchmarkCase(
        id="timeout-gd-200",
        name="Timeout: GD ×200",
        category="tier4-timeout",
        source=source,
        mechanism_name="gd_200",
        privacy_notion="approx_dp",
        epsilon=1.0,
        delta=1e-5,
        expected_verified=True,
        timeout=10.0,
        description="Noisy GD with 200 iterations, 10s timeout.",
        tags=["stress", "timeout", "iterative"],
    ))

    # Giant histogram
    source = noisy_histogram_source(
        num_bins=1000, epsilon=1.0, name="hist_1000",
    )
    cases.append(BenchmarkCase(
        id="timeout-hist-1000",
        name="Timeout: Histogram 1000 bins",
        category="tier4-timeout",
        source=source,
        mechanism_name="hist_1000",
        privacy_notion="pure_dp",
        epsilon=1.0,
        expected_verified=True,
        timeout=10.0,
        description="1000-bin histogram with 10s timeout.",
        tags=["stress", "timeout", "histogram"],
    ))

    return cases


# ---------------------------------------------------------------------------
# SVT large query count stress tests
# ---------------------------------------------------------------------------

def _svt_stress_cases() -> list[BenchmarkCase]:
    """Generate SVT stress tests with large query counts."""
    cases: list[BenchmarkCase] = []
    query_counts = [50, 100, 200, 500]

    for n_q in query_counts:
        source = sparse_vector_source(
            num_queries=n_q, epsilon=1.0, name=f"svt_stress_{n_q}",
        )
        timeout = min(600.0, 30.0 + n_q * 1.0)
        cases.append(BenchmarkCase(
            id=f"stress-svt-{n_q}q",
            name=f"SVT stress {n_q} queries",
            category="tier4-svt-stress",
            source=source,
            mechanism_name=f"svt_stress_{n_q}",
            privacy_notion="pure_dp",
            epsilon=1.0,
            expected_verified=True,
            timeout=timeout,
            description=f"SVT with {n_q} queries.",
            tags=["stress", "svt", "scaling"],
        ))

    return cases


# ---------------------------------------------------------------------------
# Adaptive composition stress tests
# ---------------------------------------------------------------------------

def _adaptive_stress_cases() -> list[BenchmarkCase]:
    """Generate adaptive composition stress tests."""
    cases: list[BenchmarkCase] = []
    round_counts = [50, 100, 200]

    for n_rounds in round_counts:
        source = adaptive_composition_source(
            num_rounds=n_rounds, total_epsilon=1.0, delta_prime=1e-5,
            name=f"adaptive_stress_{n_rounds}",
        )
        timeout = min(600.0, 30.0 + n_rounds * 2.0)
        cases.append(BenchmarkCase(
            id=f"stress-adaptive-{n_rounds}",
            name=f"Adaptive stress ×{n_rounds}",
            category="tier4-adaptive-stress",
            source=source,
            mechanism_name=f"adaptive_stress_{n_rounds}",
            privacy_notion="approx_dp",
            epsilon=1.0,
            delta=1e-5,
            expected_verified=True,
            timeout=timeout,
            description=f"Adaptive composition with {n_rounds} rounds.",
            tags=["stress", "adaptive", "scaling"],
        ))

    return cases


# ---------------------------------------------------------------------------
# Wide branching benchmarks (flat if-elif chains, 20+ branches)
# ---------------------------------------------------------------------------

def _wide_branching_mechanism(
    num_branches: int,
    name: str = "wide_branch",
) -> str:
    """Generate a mechanism with many flat branches (if-elif chain).

    Unlike the nested binary branching above, this creates a single-level
    dispatch with ``num_branches`` disjoint arms.  Each arm performs its
    own Laplace noise draw, stressing path enumeration and cross-path
    density ratio computation.

    Args:
        num_branches: Number of branches in the if-elif chain.
        name: Function name.

    Returns:
        Python source string.
    """
    eps_per_draw = 1.0
    scale = round(1.0 / eps_per_draw, 6)

    branch_lines: list[str] = []
    for i in range(num_branches):
        kw = "if" if i == 0 else "elif"
        branch_lines.append(
            f"    {kw} selector == {i}:\n"
            f"        # @dp.sensitivity(1)\n"
            f"        ans = query_{i}(db)\n"
            f"        # @dp.noise(kind=\"laplace\", scale={scale})\n"
            f"        result = ans + laplace(0, {scale})"
        )
    branch_lines.append(
        f"    else:\n"
        f"        # @dp.sensitivity(1)\n"
        f"        ans = query_default(db)\n"
        f"        # @dp.noise(kind=\"laplace\", scale={scale})\n"
        f"        result = ans + laplace(0, {scale})"
    )
    body = "\n".join(branch_lines)
    query_args = ", ".join(
        [f"query_{i}" for i in range(num_branches)] + ["query_default"]
    )
    return f'''
# @dp.mechanism(privacy="1.0-dp", sensitivity=1)
def {name}(db, selector, {query_args}):
    """Mechanism with {num_branches} flat branches (if-elif chain).

    Stresses path enumeration and cross-path density ratio computation.
    Total paths = {num_branches + 1} (including else).
    """
    result = 0
{body}
    return result
'''


def _wide_branching_cases() -> list[BenchmarkCase]:
    """Generate wide-branching stress benchmarks (20+ branches)."""
    cases: list[BenchmarkCase] = []
    branch_counts = [20, 30, 50, 75, 100]

    for n in branch_counts:
        source = _wide_branching_mechanism(n, name=f"wide_branch_{n}")
        timeout = min(600.0, 60.0 + n * 2.0)
        cases.append(BenchmarkCase(
            id=f"stress-wide-branch-{n}",
            name=f"Wide branch ×{n}",
            category="tier4-wide-branching",
            source=source,
            mechanism_name=f"wide_branch_{n}",
            privacy_notion="pure_dp",
            epsilon=1.0,
            expected_verified=True,
            timeout=timeout,
            description=(
                f"Flat if-elif chain with {n} branches; "
                f"stresses path enumeration and cross-path density ratios."
            ),
            tags=["stress", "branching", "wide", "path_enumeration",
                  "density_ratio"],
        ))

    return cases


# ---------------------------------------------------------------------------
# Deep composition benchmarks (100+ sequential queries)
# ---------------------------------------------------------------------------

def _deep_composition_mechanism(
    num_queries: int,
    name: str = "deep_comp",
) -> str:
    """Generate a mechanism with many sequentially composed queries.

    Each query uses Laplace noise with ε_i = total_ε / num_queries.

    Args:
        num_queries: Number of sequential queries.
        name: Function name.

    Returns:
        Python source string.
    """
    eps_per = 1.0 / num_queries
    scale = round(1.0 / eps_per, 6)

    query_lines: list[str] = []
    for i in range(num_queries):
        query_lines.append(
            f"    # @dp.sensitivity(1)\n"
            f"    ans_{i} = queries[{i}](db)\n"
            f"    # @dp.noise(kind=\"laplace\", scale={scale})\n"
            f"    results.append(ans_{i} + laplace(0, {scale}))"
        )
    body = "\n".join(query_lines)
    return f'''
# @dp.mechanism(privacy="1.0-dp", sensitivity=1)
def {name}(db, queries):
    """Deep composition: {num_queries} sequential Laplace queries.

    Per-query ε = {eps_per:.6f}, scale = {scale}.
    Tests verification scalability with deep composition chains.
    """
    results = []
{body}
    return results
'''


def _deep_composition_cases() -> list[BenchmarkCase]:
    """Generate deep-composition stress benchmarks (100+ queries)."""
    cases: list[BenchmarkCase] = []
    query_counts = [100, 150, 200, 300, 500]

    for n in query_counts:
        source = _deep_composition_mechanism(n, name=f"deep_comp_{n}")
        timeout = min(600.0, 60.0 + n * 1.5)
        cases.append(BenchmarkCase(
            id=f"stress-deep-comp-{n}",
            name=f"Deep composition ×{n}",
            category="tier4-deep-composition",
            source=source,
            mechanism_name=f"deep_comp_{n}",
            privacy_notion="pure_dp",
            epsilon=1.0,
            expected_verified=True,
            timeout=timeout,
            description=(
                f"Sequential composition of {n} Laplace queries; "
                f"stresses composition accounting and solver scalability."
            ),
            tags=["stress", "composition", "deep", "scaling"],
        ))

    return cases


# ---------------------------------------------------------------------------
# CEGAR iteration benchmarks (subtle privacy violations)
# ---------------------------------------------------------------------------

def _subtle_violation_mechanism(
    num_branches: int,
    violation_branch: int,
    name: str = "subtle_violation",
) -> str:
    """Generate a mechanism with a subtle privacy violation in one branch.

    All branches are correctly calibrated except ``violation_branch``,
    which uses a slightly too-small noise scale.  The verifier must
    refine abstractions repeatedly to isolate the violating branch.

    Args:
        num_branches: Total number of branches.
        violation_branch: Index of the branch with the violation.
        name: Function name.

    Returns:
        Python source string.
    """
    correct_scale = round(1.0 / 1.0, 6)  # ε = 1
    # Slightly insufficient noise: scale = 1/1.05 ≈ 0.952 (needs 1.0)
    wrong_scale = round(1.0 / 1.05, 6)

    branch_lines: list[str] = []
    for i in range(num_branches):
        kw = "if" if i == 0 else "elif"
        scale = wrong_scale if i == violation_branch else correct_scale
        branch_lines.append(
            f"    {kw} selector == {i}:\n"
            f"        # @dp.sensitivity(1)\n"
            f"        ans = query_{i}(db)\n"
            f"        # @dp.noise(kind=\"laplace\", scale={scale})\n"
            f"        result = ans + laplace(0, {scale})"
        )
    branch_lines.append(
        f"    else:\n"
        f"        # @dp.sensitivity(1)\n"
        f"        ans = query_default(db)\n"
        f"        # @dp.noise(kind=\"laplace\", scale={correct_scale})\n"
        f"        result = ans + laplace(0, {correct_scale})"
    )
    body = "\n".join(branch_lines)
    query_args = ", ".join(
        [f"query_{i}" for i in range(num_branches)] + ["query_default"]
    )
    return f'''
# @dp.mechanism(privacy="1.0-dp", sensitivity=1)
def {name}(db, selector, {query_args}):
    """Mechanism with a subtle privacy violation in branch {violation_branch}.

    {num_branches} branches, only branch {violation_branch} uses
    scale={wrong_scale} (should be {correct_scale}).
    Requires fine-grained path splitting to detect.
    """
    result = 0
{body}
    return result
'''


def _cegar_iteration_cases() -> list[BenchmarkCase]:
    """Generate benchmarks requiring many CEGAR iterations.

    These mechanisms contain subtle privacy violations hidden among
    many correct branches, forcing the CEGAR loop to iteratively
    refine until the violating path is isolated.
    """
    cases: list[BenchmarkCase] = []

    configs = [
        # (num_branches, violation_branch, label)
        (10, 7, "10b-v7"),
        (15, 12, "15b-v12"),
        (20, 17, "20b-v17"),
        (30, 25, "30b-v25"),
        (50, 42, "50b-v42"),
    ]

    for n_branches, v_branch, label in configs:
        source = _subtle_violation_mechanism(
            n_branches, v_branch, name=f"subtle_{label.replace('-', '_')}",
        )
        timeout = min(600.0, 60.0 + n_branches * 3.0)
        cases.append(BenchmarkCase(
            id=f"stress-cegar-{label}",
            name=f"CEGAR iterations {label}",
            category="tier4-cegar-iterations",
            source=source,
            mechanism_name=f"subtle_{label.replace('-', '_')}",
            privacy_notion="pure_dp",
            epsilon=1.0,
            expected_verified=False,
            timeout=timeout,
            description=(
                f"Subtle violation in branch {v_branch} of {n_branches}; "
                f"requires iterative CEGAR refinement to isolate."
            ),
            tags=["stress", "cegar", "subtle_violation", "refinement"],
        ))

    # Deep composition with a single wrong query in the middle
    for total_q in [50, 100, 200]:
        violation_idx = total_q // 2
        fn_name = f"deep_subtle_{total_q}"
        eps_per = 1.0 / total_q
        correct_scale = round(1.0 / eps_per, 6)
        wrong_scale = round(1.0 / (eps_per * 1.1), 6)  # slightly wrong

        query_lines: list[str] = []
        for i in range(total_q):
            s = wrong_scale if i == violation_idx else correct_scale
            query_lines.append(
                f"    # @dp.sensitivity(1)\n"
                f"    ans_{i} = queries[{i}](db)\n"
                f"    # @dp.noise(kind=\"laplace\", scale={s})\n"
                f"    results.append(ans_{i} + laplace(0, {s}))"
            )
        body = "\n".join(query_lines)
        source = f'''
# @dp.mechanism(privacy="1.0-dp", sensitivity=1)
def {fn_name}(db, queries):
    """Deep composition with subtle violation at query {violation_idx}.

    {total_q} sequential queries; query {violation_idx} uses
    scale={wrong_scale} (should be {correct_scale}).
    """
    results = []
{body}
    return results
'''
        timeout = min(600.0, 60.0 + total_q * 2.0)
        cases.append(BenchmarkCase(
            id=f"stress-cegar-deep-{total_q}",
            name=f"CEGAR deep ×{total_q}",
            category="tier4-cegar-iterations",
            source=source,
            mechanism_name=fn_name,
            privacy_notion="pure_dp",
            epsilon=1.0,
            expected_verified=False,
            timeout=timeout,
            description=(
                f"Sequential ×{total_q} with subtle violation at query "
                f"{violation_idx}; stresses CEGAR refinement depth."
            ),
            tags=["stress", "cegar", "deep", "subtle_violation"],
        ))

    return cases


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_tier4_cases() -> list[BenchmarkCase]:
    """Return all Tier 4 benchmark cases.

    Returns:
        List of BenchmarkCase objects covering stress tests.
    """
    cases: list[BenchmarkCase] = []
    cases.extend(_large_loop_cases())
    cases.extend(_branching_cases())
    cases.extend(_complex_noise_cases())
    cases.extend(_large_histogram_cases())
    cases.extend(_timeout_cases())
    cases.extend(_svt_stress_cases())
    cases.extend(_adaptive_stress_cases())
    cases.extend(_wide_branching_cases())
    cases.extend(_deep_composition_cases())
    cases.extend(_cegar_iteration_cases())
    return cases


# ---------------------------------------------------------------------------
# Stand-alone execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cases = get_tier4_cases()
    print(f"Tier 4: {len(cases)} stress benchmark cases")
    by_cat: dict[str, list[BenchmarkCase]] = {}
    for c in cases:
        by_cat.setdefault(c.category, []).append(c)
    for cat, cat_cases in sorted(by_cat.items()):
        print(f"\n  {cat} ({len(cat_cases)} cases):")
        for c in cat_cases:
            print(f"    {c.id:<34} {c.name:<36} timeout={c.timeout:.0f}s")
