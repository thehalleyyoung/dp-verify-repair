"""
Tier 1 benchmarks: basic mechanism verification.

Tests the fundamental mechanisms against their known privacy guarantees:
  - Laplace mechanism (pure DP)
  - Gaussian mechanism (approx DP, zCDP)
  - Exponential mechanism (pure DP)

Each mechanism is tested in both correct and buggy variants,
with expected verdicts for validation.

Run:
    python -m benchmarks.benchmark_runner --suite tier1
"""

from __future__ import annotations

from benchmarks.benchmark_runner import BenchmarkCase
from examples.mechanisms.laplace_mechanism import (
    LAPLACE_CORRECT,
    LAPLACE_WRONG_SCALE,
    LAPLACE_MISSING_NOISE,
    LAPLACE_CLAMPED,
    LAPLACE_MULTI_QUERY,
    LAPLACE_BENCH_CONFIGS,
    laplace_mechanism_source,
)
from examples.mechanisms.gaussian_mechanism import (
    GAUSSIAN_CORRECT,
    GAUSSIAN_ANALYTIC_CORRECT,
    GAUSSIAN_ZCDP,
    GAUSSIAN_WRONG_SIGMA,
    GAUSSIAN_WRONG_NORM,
    GAUSSIAN_MULTI_DIM,
    GAUSSIAN_BENCH_CONFIGS,
)
from examples.mechanisms.exponential_mechanism import (
    EXPONENTIAL_CORRECT,
    REPORT_NOISY_MAX_CORRECT,
    EXPONENTIAL_WRONG_SENSITIVITY,
    EXPONENTIAL_MISSING_FACTOR,
    NOISY_MAX_WRONG_NOISE,
    EXPONENTIAL_BENCH_CONFIGS,
)
from examples.mechanisms.noisy_histogram import (
    HISTOGRAM_CORRECT,
    STABILITY_HISTOGRAM,
    HISTOGRAM_WRONG_SCALE,
    HISTOGRAM_MISSING_NOISE,
    GAUSSIAN_HISTOGRAM,
    HISTOGRAM_BENCH_CONFIGS,
)
from examples.mechanisms.private_selection import (
    PRIVATE_SELECTION_CORRECT,
    REPORT_NOISY_MAX_SELECTION,
    PRIVATE_TOP_K,
    SELECTION_WRONG_BUDGET,
    SELECTION_DOUBLE_DIP,
)


# ---------------------------------------------------------------------------
# Laplace benchmarks
# ---------------------------------------------------------------------------

def _laplace_cases() -> list[BenchmarkCase]:
    """Generate Laplace mechanism benchmark cases."""
    cases = [
        BenchmarkCase(
            id="lap-correct-1",
            name="Laplace correct (ε=1)",
            category="tier1-laplace",
            source=LAPLACE_CORRECT,
            mechanism_name="laplace_mechanism",
            privacy_notion="pure_dp",
            epsilon=1.0,
            expected_verified=True,
            description="Standard correct Laplace mechanism.",
            tags=["laplace", "correct", "pure_dp"],
        ),
        BenchmarkCase(
            id="lap-wrong-scale",
            name="Laplace wrong scale",
            category="tier1-laplace",
            source=LAPLACE_WRONG_SCALE,
            mechanism_name="laplace_wrong_scale",
            privacy_notion="pure_dp",
            epsilon=1.0,
            expected_verified=False,
            description="Laplace with scale=Δf·ε instead of Δf/ε.",
            tags=["laplace", "buggy", "scale_bug"],
        ),
        BenchmarkCase(
            id="lap-missing-noise",
            name="Laplace missing noise",
            category="tier1-laplace",
            source=LAPLACE_MISSING_NOISE,
            mechanism_name="laplace_missing_noise",
            privacy_notion="pure_dp",
            epsilon=1.0,
            expected_verified=False,
            description="Laplace mechanism with no noise at all.",
            tags=["laplace", "buggy", "missing_noise"],
        ),
        BenchmarkCase(
            id="lap-clamped",
            name="Laplace with clamping",
            category="tier1-laplace",
            source=LAPLACE_CLAMPED,
            mechanism_name="laplace_clamped",
            privacy_notion="pure_dp",
            epsilon=1.0,
            expected_verified=True,
            description="Correct Laplace with post-processing clamp.",
            tags=["laplace", "correct", "post_processing"],
        ),
        BenchmarkCase(
            id="lap-multi-query",
            name="Laplace 3 queries",
            category="tier1-laplace",
            source=LAPLACE_MULTI_QUERY,
            mechanism_name="laplace_multi_query",
            privacy_notion="pure_dp",
            epsilon=3.0,
            expected_verified=True,
            description="Correct sequential composition of 3 queries.",
            tags=["laplace", "correct", "composition"],
        ),
    ]

    # Add sweep configs
    for cfg in LAPLACE_BENCH_CONFIGS:
        cases.append(BenchmarkCase(
            id=f"lap-sweep-{cfg.name}",
            name=f"Laplace sweep {cfg.name}",
            category="tier1-laplace-sweep",
            source=cfg.source(),
            mechanism_name=cfg.name,
            privacy_notion="pure_dp",
            epsilon=cfg.epsilon,
            expected_verified=cfg.is_correct,
            description=cfg.description,
            tags=["laplace", "sweep"],
        ))

    return cases


# ---------------------------------------------------------------------------
# Gaussian benchmarks
# ---------------------------------------------------------------------------

def _gaussian_cases() -> list[BenchmarkCase]:
    """Generate Gaussian mechanism benchmark cases."""
    cases = [
        BenchmarkCase(
            id="gauss-correct-1",
            name="Gaussian correct (ε=1,δ=1e-5)",
            category="tier1-gaussian",
            source=GAUSSIAN_CORRECT,
            mechanism_name="gaussian_mechanism",
            privacy_notion="approx_dp",
            epsilon=1.0,
            delta=1e-5,
            expected_verified=True,
            description="Standard correct Gaussian mechanism.",
            tags=["gaussian", "correct", "approx_dp"],
        ),
        BenchmarkCase(
            id="gauss-analytic",
            name="Analytic Gaussian",
            category="tier1-gaussian",
            source=GAUSSIAN_ANALYTIC_CORRECT,
            mechanism_name="analytic_gaussian",
            privacy_notion="approx_dp",
            epsilon=1.0,
            delta=1e-5,
            expected_verified=True,
            description="Analytic Gaussian mechanism (Balle & Wang).",
            tags=["gaussian", "correct", "analytic"],
        ),
        BenchmarkCase(
            id="gauss-zcdp",
            name="Gaussian zCDP",
            category="tier1-gaussian",
            source=GAUSSIAN_ZCDP,
            mechanism_name="gaussian_zcdp",
            privacy_notion="zcdp",
            epsilon=0.5,
            expected_verified=True,
            description="zCDP-calibrated Gaussian.",
            tags=["gaussian", "correct", "zcdp"],
        ),
        BenchmarkCase(
            id="gauss-wrong-sigma",
            name="Gaussian wrong σ",
            category="tier1-gaussian",
            source=GAUSSIAN_WRONG_SIGMA,
            mechanism_name="gaussian_wrong_sigma",
            privacy_notion="approx_dp",
            epsilon=1.0,
            delta=1e-5,
            expected_verified=False,
            description="Gaussian with σ=Δf/ε (missing ln factor).",
            tags=["gaussian", "buggy", "sigma_bug"],
        ),
        BenchmarkCase(
            id="gauss-wrong-norm",
            name="Gaussian wrong norm",
            category="tier1-gaussian",
            source=GAUSSIAN_WRONG_NORM,
            mechanism_name="gaussian_wrong_norm",
            privacy_notion="approx_dp",
            epsilon=1.0,
            delta=1e-5,
            expected_verified=False,
            description="Gaussian using L1 instead of L2 sensitivity.",
            tags=["gaussian", "buggy", "norm_bug"],
        ),
        BenchmarkCase(
            id="gauss-multi-dim",
            name="Gaussian 3-dim",
            category="tier1-gaussian",
            source=GAUSSIAN_MULTI_DIM,
            mechanism_name="gaussian_multi_dim",
            privacy_notion="approx_dp",
            epsilon=1.0,
            delta=1e-5,
            expected_verified=True,
            description="3-dimensional Gaussian mechanism.",
            tags=["gaussian", "correct", "multi_dim"],
        ),
    ]

    # Add sweep configs
    for cfg in GAUSSIAN_BENCH_CONFIGS:
        cases.append(BenchmarkCase(
            id=f"gauss-sweep-{cfg.name}",
            name=f"Gaussian sweep {cfg.name}",
            category="tier1-gaussian-sweep",
            source=cfg.source(),
            mechanism_name=cfg.name,
            privacy_notion=cfg.privacy_notion,
            epsilon=cfg.epsilon,
            delta=cfg.delta,
            expected_verified=cfg.is_correct,
            description=cfg.description,
            tags=["gaussian", "sweep"],
        ))

    return cases


# ---------------------------------------------------------------------------
# Exponential benchmarks
# ---------------------------------------------------------------------------

def _exponential_cases() -> list[BenchmarkCase]:
    """Generate exponential mechanism benchmark cases."""
    cases = [
        BenchmarkCase(
            id="exp-correct-5c",
            name="Exponential correct (5 cand.)",
            category="tier1-exponential",
            source=EXPONENTIAL_CORRECT,
            mechanism_name="exponential_mechanism",
            privacy_notion="pure_dp",
            epsilon=1.0,
            expected_verified=True,
            description="Standard exponential mechanism, 5 candidates.",
            tags=["exponential", "correct"],
        ),
        BenchmarkCase(
            id="exp-noisy-max",
            name="Report noisy max",
            category="tier1-exponential",
            source=REPORT_NOISY_MAX_CORRECT,
            mechanism_name="report_noisy_max",
            privacy_notion="pure_dp",
            epsilon=1.0,
            expected_verified=True,
            description="Report noisy max mechanism.",
            tags=["exponential", "correct", "noisy_max"],
        ),
        BenchmarkCase(
            id="exp-wrong-sens",
            name="Exponential wrong sensitivity",
            category="tier1-exponential",
            source=EXPONENTIAL_WRONG_SENSITIVITY,
            mechanism_name="exp_wrong_sensitivity",
            privacy_notion="pure_dp",
            epsilon=1.0,
            expected_verified=False,
            description="Exponential with halved sensitivity.",
            tags=["exponential", "buggy", "sensitivity_bug"],
        ),
        BenchmarkCase(
            id="exp-missing-2",
            name="Exponential missing ×2",
            category="tier1-exponential",
            source=EXPONENTIAL_MISSING_FACTOR,
            mechanism_name="exp_missing_2",
            privacy_notion="pure_dp",
            epsilon=1.0,
            expected_verified=False,
            description="Missing factor of 2 in denominator.",
            tags=["exponential", "buggy", "factor_bug"],
        ),
        BenchmarkCase(
            id="exp-wrong-noise",
            name="Noisy max wrong noise type",
            category="tier1-exponential",
            source=NOISY_MAX_WRONG_NOISE,
            mechanism_name="noisy_max_wrong_noise",
            privacy_notion="pure_dp",
            epsilon=1.0,
            expected_verified=False,
            description="Report noisy max using Gaussian instead of Laplace.",
            tags=["exponential", "buggy", "noise_type_bug"],
        ),
    ]

    for cfg in EXPONENTIAL_BENCH_CONFIGS:
        cases.append(BenchmarkCase(
            id=f"exp-sweep-{cfg.name}",
            name=f"Exp sweep {cfg.name}",
            category="tier1-exponential-sweep",
            source=cfg.source(),
            mechanism_name=cfg.name,
            privacy_notion="pure_dp",
            epsilon=cfg.epsilon,
            expected_verified=cfg.is_correct,
            description=cfg.description,
            tags=["exponential", "sweep"],
        ))

    return cases


# ---------------------------------------------------------------------------
# Histogram benchmarks
# ---------------------------------------------------------------------------

def _histogram_cases() -> list[BenchmarkCase]:
    """Generate histogram mechanism benchmark cases."""
    return [
        BenchmarkCase(
            id="hist-correct-5",
            name="Histogram correct (5 bins)",
            category="tier1-histogram",
            source=HISTOGRAM_CORRECT,
            mechanism_name="noisy_histogram",
            privacy_notion="pure_dp",
            epsilon=1.0,
            expected_verified=True,
            description="Standard noisy histogram.",
            tags=["histogram", "correct"],
        ),
        BenchmarkCase(
            id="hist-stability",
            name="Stability histogram",
            category="tier1-histogram",
            source=STABILITY_HISTOGRAM,
            mechanism_name="stability_histogram",
            privacy_notion="pure_dp",
            epsilon=1.0,
            expected_verified=True,
            description="Stability-based histogram.",
            tags=["histogram", "correct", "stability"],
        ),
        BenchmarkCase(
            id="hist-wrong-scale",
            name="Histogram wrong scale",
            category="tier1-histogram",
            source=HISTOGRAM_WRONG_SCALE,
            mechanism_name="histogram_wrong_scale",
            privacy_notion="pure_dp",
            epsilon=1.0,
            expected_verified=False,
            description="Noise scale divided by num bins.",
            tags=["histogram", "buggy", "scale_bug"],
        ),
        BenchmarkCase(
            id="hist-missing-noise",
            name="Histogram missing bin noise",
            category="tier1-histogram",
            source=HISTOGRAM_MISSING_NOISE,
            mechanism_name="histogram_missing_noise",
            privacy_notion="pure_dp",
            epsilon=1.0,
            expected_verified=False,
            description="Only first bin gets noise.",
            tags=["histogram", "buggy", "missing_noise"],
        ),
        BenchmarkCase(
            id="hist-gaussian",
            name="Gaussian histogram",
            category="tier1-histogram",
            source=GAUSSIAN_HISTOGRAM,
            mechanism_name="gaussian_histogram",
            privacy_notion="approx_dp",
            epsilon=1.0,
            delta=1e-5,
            expected_verified=True,
            description="Gaussian noisy histogram.",
            tags=["histogram", "correct", "gaussian"],
        ),
    ]


# ---------------------------------------------------------------------------
# Selection benchmarks
# ---------------------------------------------------------------------------

def _selection_cases() -> list[BenchmarkCase]:
    """Generate private selection benchmark cases."""
    return [
        BenchmarkCase(
            id="sel-correct",
            name="Private selection correct",
            category="tier1-selection",
            source=PRIVATE_SELECTION_CORRECT,
            mechanism_name="private_selection",
            privacy_notion="pure_dp",
            epsilon=1.0,
            expected_verified=True,
            description="Correct exponential-based selection.",
            tags=["selection", "correct"],
        ),
        BenchmarkCase(
            id="sel-noisy-max",
            name="Noisy max selection",
            category="tier1-selection",
            source=REPORT_NOISY_MAX_SELECTION,
            mechanism_name="report_noisy_max_sel",
            privacy_notion="pure_dp",
            epsilon=1.0,
            expected_verified=True,
            description="Report noisy max selection.",
            tags=["selection", "correct", "noisy_max"],
        ),
        BenchmarkCase(
            id="sel-top-k",
            name="Private top-3",
            category="tier1-selection",
            source=PRIVATE_TOP_K,
            mechanism_name="private_top_k",
            privacy_notion="pure_dp",
            epsilon=1.0,
            expected_verified=True,
            description="Private top-3 from 10 candidates.",
            tags=["selection", "correct", "top_k"],
        ),
        BenchmarkCase(
            id="sel-wrong-budget",
            name="Selection wrong budget",
            category="tier1-selection",
            source=SELECTION_WRONG_BUDGET,
            mechanism_name="selection_wrong_budget",
            privacy_notion="pure_dp",
            epsilon=1.0,
            expected_verified=False,
            description="Scores not budget-accounted.",
            tags=["selection", "buggy", "budget_bug"],
        ),
        BenchmarkCase(
            id="sel-double-dip",
            name="Selection double dip",
            category="tier1-selection",
            source=SELECTION_DOUBLE_DIP,
            mechanism_name="selection_double_dip",
            privacy_notion="pure_dp",
            epsilon=1.0,
            expected_verified=False,
            description="Returns raw score of selected candidate.",
            tags=["selection", "buggy", "double_dip"],
        ),
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_tier1_cases() -> list[BenchmarkCase]:
    """Return all Tier 1 benchmark cases.

    Returns:
        List of BenchmarkCase objects covering basic mechanisms.
    """
    cases: list[BenchmarkCase] = []
    cases.extend(_laplace_cases())
    cases.extend(_gaussian_cases())
    cases.extend(_exponential_cases())
    cases.extend(_histogram_cases())
    cases.extend(_selection_cases())
    return cases


# ---------------------------------------------------------------------------
# Stand-alone execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cases = get_tier1_cases()
    print(f"Tier 1: {len(cases)} benchmark cases")
    for c in cases:
        exp = "✓" if c.expected_verified else ("✗" if c.expected_verified is False else "?")
        print(f"  [{exp}] {c.id:<30} {c.name}")
