"""
Tier 3 benchmarks: repair synthesis.

Tests the CEGIS-based repair synthesis on known-buggy mechanisms:
  - SVT buggy variants (6 known bugs)
  - Laplace wrong scale repair
  - Gaussian wrong σ repair
  - Exponential wrong sensitivity repair
  - Repair success rate, minimality, and timing

Run:
    python -m benchmarks.benchmark_runner --suite tier3
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from benchmarks.benchmark_runner import BenchmarkCase
from examples.mechanisms.laplace_mechanism import (
    LAPLACE_WRONG_SCALE,
    LAPLACE_MISSING_NOISE,
    laplace_wrong_scale_source,
)
from examples.mechanisms.gaussian_mechanism import (
    GAUSSIAN_WRONG_SIGMA,
    GAUSSIAN_WRONG_NORM,
    gaussian_wrong_sigma_source,
)
from examples.mechanisms.exponential_mechanism import (
    EXPONENTIAL_WRONG_SENSITIVITY,
    EXPONENTIAL_MISSING_FACTOR,
)
from examples.mechanisms.sparse_vector import (
    SVT_BUG1_NO_THRESHOLD_NOISE,
    SVT_BUG2_REUSE_THRESHOLD,
    SVT_BUG3_WRONG_SENSITIVITY,
    SVT_BUG4_NO_HALT,
    SVT_BUG5_WRONG_BUDGET,
    SVT_BUG6_OUTPUT_VALUE,
    svt_bug1_no_threshold_noise,
    svt_bug3_wrong_sensitivity,
    svt_bug5_wrong_budget,
)
from examples.mechanisms.noisy_histogram import (
    HISTOGRAM_WRONG_SCALE,
    HISTOGRAM_MISSING_NOISE,
)
from examples.mechanisms.iterative_mechanism import (
    NOISY_GD_WRONG_COMPOSITION,
    NOISY_GD_NO_CLIPPING,
)
from examples.mechanisms.private_selection import (
    SELECTION_WRONG_BUDGET,
    SELECTION_DOUBLE_DIP,
)
from examples.mechanisms.composed_mechanism import (
    SEQUENTIAL_WRONG_SPLIT,
    sequential_wrong_split_source,
)


# ---------------------------------------------------------------------------
# Repair benchmark case (extends BenchmarkCase with repair-specific fields)
# ---------------------------------------------------------------------------

@dataclass
class RepairBenchmarkCase(BenchmarkCase):
    """A benchmark case for repair synthesis evaluation.

    Extends BenchmarkCase with repair-specific metadata.
    """

    bug_type: str = ""
    expected_repair_template: str = ""
    expected_repair_possible: bool = True
    repair_timeout: float = 120.0
    min_repair_cost: float = 0.0
    literature_reference: str = ""


# ---------------------------------------------------------------------------
# SVT repair benchmarks
# ---------------------------------------------------------------------------

def _svt_repair_cases() -> list[RepairBenchmarkCase]:
    """Generate repair benchmarks for all SVT bug variants."""
    return [
        RepairBenchmarkCase(
            id="repair-svt-bug1",
            name="Repair SVT bug 1",
            category="tier3-svt-repair",
            source=SVT_BUG1_NO_THRESHOLD_NOISE,
            mechanism_name="svt_bug1",
            privacy_notion="pure_dp",
            epsilon=1.0,
            expected_verified=False,
            bug_type="missing_noise",
            expected_repair_template="ScaleParam",
            expected_repair_possible=True,
            description="SVT bug 1: missing threshold noise.",
            literature_reference="Lyu et al. VLDB 2017, Bug #1",
            tags=["svt", "repair", "missing_noise"],
        ),
        RepairBenchmarkCase(
            id="repair-svt-bug2",
            name="Repair SVT bug 2",
            category="tier3-svt-repair",
            source=SVT_BUG2_REUSE_THRESHOLD,
            mechanism_name="svt_bug2",
            privacy_notion="pure_dp",
            epsilon=1.0,
            expected_verified=False,
            bug_type="structural",
            expected_repair_template="CompositionBudgetSplit",
            expected_repair_possible=False,  # structural — hard to repair
            description="SVT bug 2: fresh threshold noise each iteration.",
            literature_reference="Lyu et al. VLDB 2017, Bug #2",
            tags=["svt", "repair", "structural"],
        ),
        RepairBenchmarkCase(
            id="repair-svt-bug3",
            name="Repair SVT bug 3",
            category="tier3-svt-repair",
            source=SVT_BUG3_WRONG_SENSITIVITY,
            mechanism_name="svt_bug3",
            privacy_notion="pure_dp",
            epsilon=1.0,
            expected_verified=False,
            bug_type="sensitivity",
            expected_repair_template="SensitivityRescale",
            expected_repair_possible=True,
            description="SVT bug 3: wrong query noise sensitivity.",
            literature_reference="Lyu et al. VLDB 2017, Bug #3",
            tags=["svt", "repair", "sensitivity_bug"],
        ),
        RepairBenchmarkCase(
            id="repair-svt-bug4",
            name="Repair SVT bug 4",
            category="tier3-svt-repair",
            source=SVT_BUG4_NO_HALT,
            mechanism_name="svt_bug4",
            privacy_notion="pure_dp",
            epsilon=1.0,
            expected_verified=False,
            bug_type="control_flow",
            expected_repair_template="ThresholdShift",
            expected_repair_possible=False,  # needs structural change
            description="SVT bug 4: no halt after c answers.",
            literature_reference="Lyu et al. VLDB 2017, Bug #4",
            tags=["svt", "repair", "control_flow"],
        ),
        RepairBenchmarkCase(
            id="repair-svt-bug5",
            name="Repair SVT bug 5",
            category="tier3-svt-repair",
            source=SVT_BUG5_WRONG_BUDGET,
            mechanism_name="svt_bug5",
            privacy_notion="pure_dp",
            epsilon=1.0,
            expected_verified=False,
            bug_type="budget_split",
            expected_repair_template="ScaleParam",
            expected_repair_possible=True,
            description="SVT bug 5: wrong budget allocation.",
            literature_reference="Lyu et al. VLDB 2017, Bug #5",
            tags=["svt", "repair", "budget_bug"],
        ),
        RepairBenchmarkCase(
            id="repair-svt-bug6",
            name="Repair SVT bug 6",
            category="tier3-svt-repair",
            source=SVT_BUG6_OUTPUT_VALUE,
            mechanism_name="svt_bug6",
            privacy_notion="pure_dp",
            epsilon=1.0,
            expected_verified=False,
            bug_type="information_leak",
            expected_repair_template="ClampBound",
            expected_repair_possible=False,  # needs output removal
            description="SVT bug 6: outputs noisy query value.",
            literature_reference="Lyu et al. VLDB 2017, Bug #6",
            tags=["svt", "repair", "info_leak"],
        ),
    ]


# ---------------------------------------------------------------------------
# Basic mechanism repair benchmarks
# ---------------------------------------------------------------------------

def _basic_repair_cases() -> list[RepairBenchmarkCase]:
    """Generate repair benchmarks for basic mechanism bugs."""
    return [
        RepairBenchmarkCase(
            id="repair-lap-scale",
            name="Repair Laplace scale",
            category="tier3-basic-repair",
            source=LAPLACE_WRONG_SCALE,
            mechanism_name="laplace_wrong_scale",
            privacy_notion="pure_dp",
            epsilon=1.0,
            expected_verified=False,
            bug_type="scale",
            expected_repair_template="ScaleParam",
            expected_repair_possible=True,
            description="Laplace with wrong scale (Δf·ε instead of Δf/ε).",
            tags=["laplace", "repair", "scale_bug"],
        ),
        RepairBenchmarkCase(
            id="repair-lap-missing",
            name="Repair Laplace missing noise",
            category="tier3-basic-repair",
            source=LAPLACE_MISSING_NOISE,
            mechanism_name="laplace_missing_noise",
            privacy_notion="pure_dp",
            epsilon=1.0,
            expected_verified=False,
            bug_type="missing_noise",
            expected_repair_template="ScaleParam",
            expected_repair_possible=False,  # no noise draw to adjust
            description="Laplace with no noise at all.",
            tags=["laplace", "repair", "missing_noise"],
        ),
        RepairBenchmarkCase(
            id="repair-gauss-sigma",
            name="Repair Gaussian σ",
            category="tier3-basic-repair",
            source=GAUSSIAN_WRONG_SIGMA,
            mechanism_name="gaussian_wrong_sigma",
            privacy_notion="approx_dp",
            epsilon=1.0,
            delta=1e-5,
            expected_verified=False,
            bug_type="scale",
            expected_repair_template="ScaleParam",
            expected_repair_possible=True,
            description="Gaussian with wrong σ (missing ln factor).",
            tags=["gaussian", "repair", "sigma_bug"],
        ),
        RepairBenchmarkCase(
            id="repair-gauss-norm",
            name="Repair Gaussian norm",
            category="tier3-basic-repair",
            source=GAUSSIAN_WRONG_NORM,
            mechanism_name="gaussian_wrong_norm",
            privacy_notion="approx_dp",
            epsilon=1.0,
            delta=1e-5,
            expected_verified=False,
            bug_type="sensitivity",
            expected_repair_template="SensitivityRescale",
            expected_repair_possible=True,
            description="Gaussian using L1 instead of L2 sensitivity.",
            tags=["gaussian", "repair", "norm_bug"],
        ),
        RepairBenchmarkCase(
            id="repair-exp-sens",
            name="Repair Exp wrong sensitivity",
            category="tier3-basic-repair",
            source=EXPONENTIAL_WRONG_SENSITIVITY,
            mechanism_name="exp_wrong_sensitivity",
            privacy_notion="pure_dp",
            epsilon=1.0,
            expected_verified=False,
            bug_type="sensitivity",
            expected_repair_template="ScaleParam",
            expected_repair_possible=True,
            description="Exponential with underestimated sensitivity.",
            tags=["exponential", "repair", "sensitivity_bug"],
        ),
        RepairBenchmarkCase(
            id="repair-exp-factor",
            name="Repair Exp missing ×2",
            category="tier3-basic-repair",
            source=EXPONENTIAL_MISSING_FACTOR,
            mechanism_name="exp_missing_2",
            privacy_notion="pure_dp",
            epsilon=1.0,
            expected_verified=False,
            bug_type="scale",
            expected_repair_template="ScaleParam",
            expected_repair_possible=True,
            description="Exponential missing factor of 2.",
            tags=["exponential", "repair", "factor_bug"],
        ),
    ]


# ---------------------------------------------------------------------------
# Composition repair benchmarks
# ---------------------------------------------------------------------------

def _composition_repair_cases() -> list[RepairBenchmarkCase]:
    """Generate repair benchmarks for composition bugs."""
    cases: list[RepairBenchmarkCase] = []

    for depth in [2, 3, 5]:
        cases.append(RepairBenchmarkCase(
            id=f"repair-seq-wrong-{depth}",
            name=f"Repair wrong split ×{depth}",
            category="tier3-composition-repair",
            source=sequential_wrong_split_source(
                num_queries=depth, total_epsilon=1.0,
                name=f"seq_wrong_{depth}",
            ),
            mechanism_name=f"seq_wrong_{depth}",
            privacy_notion="pure_dp",
            epsilon=1.0,
            expected_verified=False,
            bug_type="composition",
            expected_repair_template="CompositionBudgetSplit",
            expected_repair_possible=True,
            description=f"Sequential ×{depth} with wrong budget split.",
            tags=["composition", "repair", "budget_split"],
        ))

    # Iterative mechanism repairs
    cases.append(RepairBenchmarkCase(
        id="repair-gd-wrong-comp",
        name="Repair GD wrong comp",
        category="tier3-composition-repair",
        source=NOISY_GD_WRONG_COMPOSITION,
        mechanism_name="noisy_gd_wrong_comp",
        privacy_notion="approx_dp",
        epsilon=1.0,
        delta=1e-5,
        expected_verified=False,
        bug_type="composition",
        expected_repair_template="ScaleParam",
        expected_repair_possible=True,
        description="Noisy GD with wrong composition accounting.",
        tags=["iterative", "repair", "composition_bug"],
    ))

    cases.append(RepairBenchmarkCase(
        id="repair-gd-no-clip",
        name="Repair GD no clipping",
        category="tier3-composition-repair",
        source=NOISY_GD_NO_CLIPPING,
        mechanism_name="noisy_gd_no_clip",
        privacy_notion="approx_dp",
        epsilon=1.0,
        delta=1e-5,
        expected_verified=False,
        bug_type="sensitivity",
        expected_repair_template="ClampBound",
        expected_repair_possible=False,  # needs structural change
        description="Noisy GD without gradient clipping.",
        tags=["iterative", "repair", "clipping_bug"],
    ))

    return cases


# ---------------------------------------------------------------------------
# Histogram and selection repair benchmarks
# ---------------------------------------------------------------------------

def _other_repair_cases() -> list[RepairBenchmarkCase]:
    """Generate repair benchmarks for histogram and selection bugs."""
    return [
        RepairBenchmarkCase(
            id="repair-hist-scale",
            name="Repair histogram scale",
            category="tier3-other-repair",
            source=HISTOGRAM_WRONG_SCALE,
            mechanism_name="histogram_wrong_scale",
            privacy_notion="pure_dp",
            epsilon=1.0,
            expected_verified=False,
            bug_type="scale",
            expected_repair_template="ScaleParam",
            expected_repair_possible=True,
            description="Histogram with noise scale ÷ num_bins.",
            tags=["histogram", "repair", "scale_bug"],
        ),
        RepairBenchmarkCase(
            id="repair-hist-missing",
            name="Repair histogram missing noise",
            category="tier3-other-repair",
            source=HISTOGRAM_MISSING_NOISE,
            mechanism_name="histogram_missing_noise",
            privacy_notion="pure_dp",
            epsilon=1.0,
            expected_verified=False,
            bug_type="missing_noise",
            expected_repair_template="ScaleParam",
            expected_repair_possible=False,  # needs structural addition
            description="Histogram missing noise on bins 1..4.",
            tags=["histogram", "repair", "missing_noise"],
        ),
        RepairBenchmarkCase(
            id="repair-sel-budget",
            name="Repair selection budget",
            category="tier3-other-repair",
            source=SELECTION_WRONG_BUDGET,
            mechanism_name="selection_wrong_budget",
            privacy_notion="pure_dp",
            epsilon=1.0,
            expected_verified=False,
            bug_type="budget",
            expected_repair_template="CompositionBudgetSplit",
            expected_repair_possible=False,  # structural issue
            description="Selection with unaccounted score queries.",
            tags=["selection", "repair", "budget_bug"],
        ),
        RepairBenchmarkCase(
            id="repair-sel-double-dip",
            name="Repair selection double dip",
            category="tier3-other-repair",
            source=SELECTION_DOUBLE_DIP,
            mechanism_name="selection_double_dip",
            privacy_notion="pure_dp",
            epsilon=1.0,
            expected_verified=False,
            bug_type="information_leak",
            expected_repair_template="ClampBound",
            expected_repair_possible=False,
            description="Selection returning raw score (double dip).",
            tags=["selection", "repair", "double_dip"],
        ),
    ]


# ---------------------------------------------------------------------------
# Repair minimality evaluation
# ---------------------------------------------------------------------------

def _minimality_cases() -> list[RepairBenchmarkCase]:
    """Generate cases for evaluating repair minimality.

    Tests whether the repair synthesizer finds the smallest change
    that fixes the bug (minimum cost).
    """
    cases: list[RepairBenchmarkCase] = []

    # Laplace wrong scale with different ε values
    for eps in [0.1, 0.5, 1.0, 2.0, 5.0]:
        source = laplace_wrong_scale_source(
            sensitivity=1.0, epsilon=eps,
            name=f"lap_min_e{str(eps).replace('.', '')}",
        )
        cases.append(RepairBenchmarkCase(
            id=f"repair-min-lap-e{str(eps).replace('.', '')}",
            name=f"Minimality Lap ε={eps}",
            category="tier3-minimality",
            source=source,
            mechanism_name=f"lap_min_e{str(eps).replace('.', '')}",
            privacy_notion="pure_dp",
            epsilon=eps,
            expected_verified=False,
            bug_type="scale",
            expected_repair_template="ScaleParam",
            expected_repair_possible=True,
            min_repair_cost=0.0,
            description=f"Repair minimality test at ε={eps}.",
            tags=["minimality", "repair"],
        ))

    # SVT wrong budget with different ε values
    for eps in [0.5, 1.0, 2.0]:
        source = svt_bug5_wrong_budget(
            epsilon=eps, name=f"svt_min_e{str(eps).replace('.', '')}",
        )
        cases.append(RepairBenchmarkCase(
            id=f"repair-min-svt-e{str(eps).replace('.', '')}",
            name=f"Minimality SVT ε={eps}",
            category="tier3-minimality",
            source=source,
            mechanism_name=f"svt_min_e{str(eps).replace('.', '')}",
            privacy_notion="pure_dp",
            epsilon=eps,
            expected_verified=False,
            bug_type="budget_split",
            expected_repair_template="ScaleParam",
            expected_repair_possible=True,
            description=f"SVT budget repair minimality at ε={eps}.",
            tags=["minimality", "repair", "svt"],
        ))

    return cases


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_tier3_cases() -> list[BenchmarkCase]:
    """Return all Tier 3 benchmark cases.

    Returns:
        List of RepairBenchmarkCase objects covering repair synthesis.
    """
    cases: list[BenchmarkCase] = []
    cases.extend(_svt_repair_cases())
    cases.extend(_basic_repair_cases())
    cases.extend(_composition_repair_cases())
    cases.extend(_other_repair_cases())
    cases.extend(_minimality_cases())
    return cases


# ---------------------------------------------------------------------------
# Stand-alone execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cases = get_tier3_cases()
    print(f"Tier 3: {len(cases)} repair benchmark cases")
    by_cat: dict[str, list[BenchmarkCase]] = {}
    for c in cases:
        by_cat.setdefault(c.category, []).append(c)
    for cat, cat_cases in sorted(by_cat.items()):
        print(f"\n  {cat} ({len(cat_cases)} cases):")
        for c in cat_cases:
            rc = c if isinstance(c, RepairBenchmarkCase) else None
            repairable = "R" if (rc and rc.expected_repair_possible) else "N"
            print(f"    [{repairable}] {c.id:<32} {c.name}")
