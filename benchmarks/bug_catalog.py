"""
Bug Catalog: a database of known DP implementation bugs.

Each entry contains:
  - Unique bug ID and human-readable name
  - The mechanism source containing the bug
  - Expected violation type and privacy notion
  - Minimal counterexample (if known)
  - Category: scale, composition, sensitivity, branching, threshold, structural
  - Literature reference (if applicable)

This catalog serves as ground truth for validating the DP-CEGAR verifier:
the verifier should reject every buggy mechanism and accept every correct one.

References:
  Lyu, Su, Li. "Understanding the Sparse Vector Technique for DP." VLDB 2017.
  Bichsel, Gehr, Drachsler-Cohen, Tsankov, Vechev. "DP-Finder." CCS 2018.
  Ding, Wang, Zhang. "Detecting Violations of DP." CCS 2018.
  Barthe, Gaboardi, Grégoire, Hsu, Strub. "Proving DP via Probabilistic
    Couplings." LICS 2016.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Bug entry data type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BugEntry:
    """A single entry in the DP bug catalog.

    Attributes:
        bug_id: Unique bug identifier (e.g. "SCALE-001").
        name: Human-readable bug name.
        description: Detailed description of the bug.
        category: Bug category (scale, composition, sensitivity, etc.).
        mechanism_source: Python source containing the bug.
        mechanism_name: Function name in the source.
        privacy_notion: DP notion that is violated.
        claimed_epsilon: Claimed ε budget.
        claimed_delta: Claimed δ budget.
        actual_epsilon: Actual ε cost (if known), None if unbounded.
        actual_delta: Actual δ cost (if known).
        violation_type: Type of violation (insufficient_noise, budget_overflow,
            sensitivity_error, information_leak, structural).
        minimal_counterexample: Description of minimal counterexample.
        repair_hint: Suggested repair (for documentation).
        is_repairable_by_scale: Whether adjusting a scale parameter fixes it.
        literature_ref: Citation to paper/source.
        tags: Descriptive tags.
    """

    bug_id: str
    name: str
    description: str
    category: str
    mechanism_source: str
    mechanism_name: str
    privacy_notion: str = "pure_dp"
    claimed_epsilon: float = 1.0
    claimed_delta: float = 0.0
    actual_epsilon: float | None = None
    actual_delta: float | None = None
    violation_type: str = "insufficient_noise"
    minimal_counterexample: str = ""
    repair_hint: str = ""
    is_repairable_by_scale: bool = False
    literature_ref: str = ""
    tags: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Bug catalog class
# ---------------------------------------------------------------------------

class BugCatalog:
    """Database of known DP implementation bugs.

    Provides access to curated bug entries, filterable by category,
    violation type, and other attributes.

    Usage::

        catalog = BugCatalog()
        all_bugs = catalog.all()
        scale_bugs = catalog.by_category("scale")
        for bug in catalog.repairable():
            print(bug.bug_id, bug.name)
    """

    def __init__(self) -> None:
        """Initialise the catalog with all known bugs."""
        self._entries: dict[str, BugEntry] = {}
        self._build_catalog()

    def _build_catalog(self) -> None:
        """Populate the catalog with all known bugs."""
        for entry in _ALL_BUGS:
            self._entries[entry.bug_id] = entry

    # -- Access methods ----------------------------------------------------

    def all(self) -> list[BugEntry]:
        """Return all bug entries."""
        return list(self._entries.values())

    def get(self, bug_id: str) -> BugEntry | None:
        """Get a bug entry by ID."""
        return self._entries.get(bug_id)

    def by_category(self, category: str) -> list[BugEntry]:
        """Filter bugs by category.

        Args:
            category: One of scale, composition, sensitivity, branching,
                threshold, structural, information_leak.

        Returns:
            List of matching BugEntry objects.
        """
        return [e for e in self._entries.values() if e.category == category]

    def by_violation_type(self, vtype: str) -> list[BugEntry]:
        """Filter bugs by violation type.

        Args:
            vtype: One of insufficient_noise, budget_overflow,
                sensitivity_error, information_leak, structural.

        Returns:
            List of matching BugEntry objects.
        """
        return [e for e in self._entries.values() if e.violation_type == vtype]

    def by_notion(self, notion: str) -> list[BugEntry]:
        """Filter bugs by privacy notion violated."""
        return [e for e in self._entries.values() if e.privacy_notion == notion]

    def repairable(self) -> list[BugEntry]:
        """Return bugs that can be repaired by adjusting a scale parameter."""
        return [e for e in self._entries.values() if e.is_repairable_by_scale]

    def structural(self) -> list[BugEntry]:
        """Return bugs requiring structural repairs (not just scale)."""
        return [e for e in self._entries.values() if not e.is_repairable_by_scale]

    def by_tag(self, tag: str) -> list[BugEntry]:
        """Filter bugs by tag."""
        return [e for e in self._entries.values() if tag in e.tags]

    def categories(self) -> list[str]:
        """Return all unique categories."""
        return sorted(set(e.category for e in self._entries.values()))

    def summary(self) -> dict[str, Any]:
        """Return a summary of the catalog."""
        entries = self.all()
        return {
            "total": len(entries),
            "by_category": {
                cat: len(self.by_category(cat))
                for cat in self.categories()
            },
            "repairable": len(self.repairable()),
            "structural": len(self.structural()),
        }

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries.values())

    def print_catalog(self) -> None:
        """Print a formatted catalog listing."""
        print(f"DP Bug Catalog — {len(self)} entries")
        print("=" * 78)
        for cat in self.categories():
            bugs = self.by_category(cat)
            print(f"\n  [{cat.upper()}] ({len(bugs)} bugs)")
            print(f"  {'-' * 74}")
            for b in bugs:
                repair_icon = "R" if b.is_repairable_by_scale else "S"
                print(
                    f"  [{repair_icon}] {b.bug_id:<14} {b.name:<40} "
                    f"{b.violation_type}"
                )
                if b.literature_ref:
                    print(f"      Ref: {b.literature_ref}")


# ---------------------------------------------------------------------------
# Import bug sources
# ---------------------------------------------------------------------------

from examples.mechanisms.laplace_mechanism import (
    LAPLACE_WRONG_SCALE,
    LAPLACE_MISSING_NOISE,
)
from examples.mechanisms.gaussian_mechanism import (
    GAUSSIAN_WRONG_SIGMA,
    GAUSSIAN_WRONG_NORM,
)
from examples.mechanisms.exponential_mechanism import (
    EXPONENTIAL_WRONG_SENSITIVITY,
    EXPONENTIAL_MISSING_FACTOR,
    NOISY_MAX_WRONG_NOISE,
)
from examples.mechanisms.sparse_vector import (
    SVT_BUG1_NO_THRESHOLD_NOISE,
    SVT_BUG2_REUSE_THRESHOLD,
    SVT_BUG3_WRONG_SENSITIVITY,
    SVT_BUG4_NO_HALT,
    SVT_BUG5_WRONG_BUDGET,
    SVT_BUG6_OUTPUT_VALUE,
)
from examples.mechanisms.noisy_histogram import (
    HISTOGRAM_WRONG_SCALE,
    HISTOGRAM_MISSING_NOISE,
)
from examples.mechanisms.composed_mechanism import SEQUENTIAL_WRONG_SPLIT
from examples.mechanisms.iterative_mechanism import (
    NOISY_GD_WRONG_COMPOSITION,
    NOISY_GD_NO_CLIPPING,
)
from examples.mechanisms.private_selection import (
    SELECTION_WRONG_BUDGET,
    SELECTION_DOUBLE_DIP,
)


# ---------------------------------------------------------------------------
# Bug entries
# ---------------------------------------------------------------------------

_ALL_BUGS: list[BugEntry] = [
    # --- Scale bugs -------------------------------------------------------
    BugEntry(
        bug_id="SCALE-001",
        name="Laplace wrong scale (multiply vs divide)",
        description=(
            "Laplace noise scale set to Δf·ε instead of Δf/ε. "
            "When ε < 1, the noise is far too small; when ε > 1, "
            "it is too large (over-private)."
        ),
        category="scale",
        mechanism_source=LAPLACE_WRONG_SCALE,
        mechanism_name="laplace_wrong_scale",
        privacy_notion="pure_dp",
        claimed_epsilon=1.0,
        actual_epsilon=None,  # depends on ε
        violation_type="insufficient_noise",
        minimal_counterexample=(
            "db1 = [0], db2 = [1]; output o=0.5: "
            "ratio = exp(|0-1|/scale) = exp(1/1) vs claimed exp(1)."
        ),
        repair_hint="Change scale from Δf·ε to Δf/ε.",
        is_repairable_by_scale=True,
        literature_ref="Common implementation error",
        tags=("laplace", "scale", "arithmetic"),
    ),
    BugEntry(
        bug_id="SCALE-002",
        name="Gaussian missing log factor in σ",
        description=(
            "Gaussian noise σ set to Δf/ε, missing the √(2·ln(1.25/δ)) "
            "factor.  Results in σ that is too small for (ε,δ)-DP."
        ),
        category="scale",
        mechanism_source=GAUSSIAN_WRONG_SIGMA,
        mechanism_name="gaussian_wrong_sigma",
        privacy_notion="approx_dp",
        claimed_epsilon=1.0,
        claimed_delta=1e-5,
        actual_epsilon=None,
        violation_type="insufficient_noise",
        minimal_counterexample=(
            "σ = 1.0 instead of ~4.87. "
            "Tail probability exceeds δ for outputs far from mean."
        ),
        repair_hint="Set σ = Δf·√(2·ln(1.25/δ))/ε.",
        is_repairable_by_scale=True,
        literature_ref="Dwork & Roth, Appendix A (common mistake)",
        tags=("gaussian", "scale", "log_factor"),
    ),
    BugEntry(
        bug_id="SCALE-003",
        name="Exponential mechanism missing factor of 2",
        description=(
            "Exponential mechanism uses scale ε/Δu instead of ε/(2Δu). "
            "This satisfies only 2ε-DP, not ε-DP."
        ),
        category="scale",
        mechanism_source=EXPONENTIAL_MISSING_FACTOR,
        mechanism_name="exp_missing_2",
        privacy_notion="pure_dp",
        claimed_epsilon=1.0,
        actual_epsilon=2.0,
        violation_type="insufficient_noise",
        minimal_counterexample=(
            "For two candidates with utility gap Δu=1: "
            "probability ratio = exp(ε·1/Δu) = exp(ε) > exp(ε/2)."
        ),
        repair_hint="Change scale from ε/Δu to ε/(2Δu).",
        is_repairable_by_scale=True,
        literature_ref="McSherry & Talwar, FOCS 2007",
        tags=("exponential", "scale", "factor_2"),
    ),
    BugEntry(
        bug_id="SCALE-004",
        name="Histogram noise scale divided by bins",
        description=(
            "Noisy histogram uses scale 1/(n·ε) instead of 1/ε. "
            "Dividing noise by bin count provides insufficient noise "
            "per bin."
        ),
        category="scale",
        mechanism_source=HISTOGRAM_WRONG_SCALE,
        mechanism_name="histogram_wrong_scale",
        privacy_notion="pure_dp",
        claimed_epsilon=1.0,
        actual_epsilon=5.0,  # 5 bins → 5× too little noise
        violation_type="insufficient_noise",
        minimal_counterexample=(
            "Changing one record affects one bin by 1; "
            "noise is only Lap(1/5) instead of Lap(1)."
        ),
        repair_hint="Use scale 1/ε for each bin independently.",
        is_repairable_by_scale=True,
        tags=("histogram", "scale", "bin_division"),
    ),

    # --- Sensitivity bugs -------------------------------------------------
    BugEntry(
        bug_id="SENS-001",
        name="Exponential mechanism under-estimated sensitivity",
        description=(
            "Uses Δu/2 instead of Δu for sensitivity, resulting in "
            "noise that is too concentrated around the maximum."
        ),
        category="sensitivity",
        mechanism_source=EXPONENTIAL_WRONG_SENSITIVITY,
        mechanism_name="exp_wrong_sensitivity",
        privacy_notion="pure_dp",
        claimed_epsilon=1.0,
        actual_epsilon=2.0,
        violation_type="sensitivity_error",
        minimal_counterexample=(
            "For neighbouring db, db': utility changes by Δu=1 "
            "but mechanism assumes change is only 0.5."
        ),
        repair_hint="Use correct sensitivity Δu.",
        is_repairable_by_scale=True,
        literature_ref="Common error in exponential mechanism",
        tags=("exponential", "sensitivity"),
    ),
    BugEntry(
        bug_id="SENS-002",
        name="Gaussian using L1 instead of L2 sensitivity",
        description=(
            "Calibrates Gaussian noise to L1 sensitivity instead of L2. "
            "When L1 ≠ L2, the noise may be miscalibrated."
        ),
        category="sensitivity",
        mechanism_source=GAUSSIAN_WRONG_NORM,
        mechanism_name="gaussian_wrong_norm",
        privacy_notion="approx_dp",
        claimed_epsilon=1.0,
        claimed_delta=1e-5,
        violation_type="sensitivity_error",
        minimal_counterexample=(
            "For d-dimensional query with L2 sensitivity 1 but "
            "L1 sensitivity d: noise is too large or too small."
        ),
        repair_hint="Use L2 sensitivity for Gaussian mechanism.",
        is_repairable_by_scale=True,
        literature_ref="Dwork & Roth, Appendix A",
        tags=("gaussian", "sensitivity", "norm"),
    ),
    BugEntry(
        bug_id="SENS-003",
        name="SVT wrong query noise sensitivity",
        description=(
            "SVT uses query noise scale calibrated to sensitivity 1 "
            "when the effective sensitivity in the comparison is 2 "
            "(or vice versa)."
        ),
        category="sensitivity",
        mechanism_source=SVT_BUG3_WRONG_SENSITIVITY,
        mechanism_name="svt_bug3",
        privacy_notion="pure_dp",
        claimed_epsilon=1.0,
        violation_type="sensitivity_error",
        minimal_counterexample=(
            "Neighbouring datasets differ by sensitivity 1 per query, "
            "but the comparison q(db)+ν vs T̃ has combined sensitivity "
            "that is not accounted for."
        ),
        repair_hint="Adjust query noise scale for correct sensitivity.",
        is_repairable_by_scale=True,
        literature_ref="Lyu et al. VLDB 2017, Bug #3",
        tags=("svt", "sensitivity"),
    ),

    # --- Composition bugs -------------------------------------------------
    BugEntry(
        bug_id="COMP-001",
        name="Sequential composition using full ε per query",
        description=(
            "Each of k queries uses the full ε budget instead of ε/k. "
            "True privacy cost is k·ε."
        ),
        category="composition",
        mechanism_source=SEQUENTIAL_WRONG_SPLIT,
        mechanism_name="seq_wrong_split",
        privacy_notion="pure_dp",
        claimed_epsilon=1.0,
        actual_epsilon=3.0,  # 3 queries × ε
        violation_type="budget_overflow",
        minimal_counterexample=(
            "Three queries each using ε=1 noise: total budget = 3ε, "
            "not ε."
        ),
        repair_hint="Split budget: use ε/k per query.",
        is_repairable_by_scale=True,
        literature_ref="Basic composition theorem violation",
        tags=("composition", "sequential", "budget"),
    ),
    BugEntry(
        bug_id="COMP-002",
        name="SVT wrong budget allocation (ε for both)",
        description=(
            "SVT uses the full ε for both threshold noise and query "
            "noise, effectively spending 2ε total."
        ),
        category="composition",
        mechanism_source=SVT_BUG5_WRONG_BUDGET,
        mechanism_name="svt_bug5",
        privacy_notion="pure_dp",
        claimed_epsilon=1.0,
        actual_epsilon=2.0,
        violation_type="budget_overflow",
        minimal_counterexample=(
            "Threshold noise uses ε; query noise uses ε; "
            "total = 2ε by sequential composition."
        ),
        repair_hint="Split budget: ε₁=ε/2 for threshold, ε₂=ε/2 for queries.",
        is_repairable_by_scale=True,
        literature_ref="Lyu et al. VLDB 2017, Bug #5",
        tags=("svt", "composition", "budget"),
    ),
    BugEntry(
        bug_id="COMP-003",
        name="Noisy GD wrong composition accounting",
        description=(
            "Iterative noisy GD uses the full ε per iteration instead "
            "of accounting for composition across iterations."
        ),
        category="composition",
        mechanism_source=NOISY_GD_WRONG_COMPOSITION,
        mechanism_name="noisy_gd_wrong_comp",
        privacy_notion="approx_dp",
        claimed_epsilon=1.0,
        claimed_delta=1e-5,
        actual_epsilon=10.0,  # 10 iterations × ε
        violation_type="budget_overflow",
        minimal_counterexample=(
            "10 iterations, each using ε=1 noise: "
            "total budget ≈ 10·ε by basic composition."
        ),
        repair_hint="Use ε/√T per iteration (advanced composition).",
        is_repairable_by_scale=True,
        literature_ref="Abadi et al. CCS 2016",
        tags=("iterative", "composition", "gradient_descent"),
    ),

    # --- Threshold / branching bugs ---------------------------------------
    BugEntry(
        bug_id="THRESH-001",
        name="SVT missing threshold noise",
        description=(
            "SVT uses the raw threshold T without adding Laplace noise. "
            "The comparison q(db)+ν ≥ T leaks the exact threshold relation."
        ),
        category="threshold",
        mechanism_source=SVT_BUG1_NO_THRESHOLD_NOISE,
        mechanism_name="svt_bug1",
        privacy_notion="pure_dp",
        claimed_epsilon=1.0,
        violation_type="insufficient_noise",
        minimal_counterexample=(
            "If q(db)=T−0.01, the output is almost certainly ⊥, "
            "but for db' with q(db')=T+0.01, output is almost certainly ⊤. "
            "Ratio is unbounded."
        ),
        repair_hint="Add Lap(2/ε) noise to threshold.",
        is_repairable_by_scale=False,  # need to ADD noise draw
        literature_ref="Lyu et al. VLDB 2017, Bug #1",
        tags=("svt", "threshold", "missing_noise"),
    ),
    BugEntry(
        bug_id="THRESH-002",
        name="SVT re-drawing threshold noise each iteration",
        description=(
            "Instead of drawing threshold noise once and reusing it, "
            "this variant draws fresh noise per query. This means "
            "threshold noise composes across queries."
        ),
        category="threshold",
        mechanism_source=SVT_BUG2_REUSE_THRESHOLD,
        mechanism_name="svt_bug2",
        privacy_notion="pure_dp",
        claimed_epsilon=1.0,
        violation_type="budget_overflow",
        minimal_counterexample=(
            "With 5 queries, threshold noise is drawn 5 times, "
            "consuming 5× the intended threshold budget."
        ),
        repair_hint="Draw threshold noise once before the loop.",
        is_repairable_by_scale=False,  # structural change
        literature_ref="Lyu et al. VLDB 2017, Bug #2",
        tags=("svt", "threshold", "structural"),
    ),

    # --- Structural / control flow bugs -----------------------------------
    BugEntry(
        bug_id="STRUCT-001",
        name="SVT not halting after c above-threshold answers",
        description=(
            "SVT should stop after c ⊤ answers, but this variant "
            "continues processing all queries. The privacy analysis "
            "relies on bounded above-threshold count."
        ),
        category="structural",
        mechanism_source=SVT_BUG4_NO_HALT,
        mechanism_name="svt_bug4",
        privacy_notion="pure_dp",
        claimed_epsilon=1.0,
        violation_type="structural",
        minimal_counterexample=(
            "Without halting, arbitrarily many ⊤ answers are released, "
            "each consuming budget. True cost = unbounded."
        ),
        repair_hint="Add count tracking and break after c ⊤ answers.",
        is_repairable_by_scale=False,
        literature_ref="Lyu et al. VLDB 2017, Bug #4",
        tags=("svt", "structural", "control_flow"),
    ),
    BugEntry(
        bug_id="STRUCT-002",
        name="Noisy GD missing gradient clipping",
        description=(
            "DP-GD assumes bounded gradient sensitivity via clipping, "
            "but this variant omits the clipping step. Actual sensitivity "
            "is unbounded."
        ),
        category="structural",
        mechanism_source=NOISY_GD_NO_CLIPPING,
        mechanism_name="noisy_gd_no_clip",
        privacy_notion="approx_dp",
        claimed_epsilon=1.0,
        claimed_delta=1e-5,
        violation_type="sensitivity_error",
        minimal_counterexample=(
            "A single outlier record can produce an arbitrarily large "
            "gradient, overwhelming the fixed noise σ."
        ),
        repair_hint="Add gradient clipping: g = g * min(1, C/||g||).",
        is_repairable_by_scale=False,
        literature_ref="Abadi et al. CCS 2016",
        tags=("iterative", "structural", "clipping"),
    ),

    # --- Information leak bugs --------------------------------------------
    BugEntry(
        bug_id="LEAK-001",
        name="SVT outputting noisy query value",
        description=(
            "Above Threshold should output only ⊤/⊥, but this variant "
            "outputs the noisy query value q(db)+ν for above-threshold "
            "queries. This leaks additional information."
        ),
        category="information_leak",
        mechanism_source=SVT_BUG6_OUTPUT_VALUE,
        mechanism_name="svt_bug6",
        privacy_notion="pure_dp",
        claimed_epsilon=1.0,
        violation_type="information_leak",
        minimal_counterexample=(
            "Outputting q(db)+ν reveals approximate query value; "
            "this requires additional ε budget per release."
        ),
        repair_hint="Output only True/False, not the noisy value.",
        is_repairable_by_scale=False,
        literature_ref="Lyu et al. VLDB 2017, Bug #6",
        tags=("svt", "information_leak", "output"),
    ),
    BugEntry(
        bug_id="LEAK-002",
        name="Laplace mechanism with no noise",
        description=(
            "Mechanism returns the raw query answer without adding "
            "any noise. This is a complete privacy violation — "
            "the output deterministically reveals the query value."
        ),
        category="information_leak",
        mechanism_source=LAPLACE_MISSING_NOISE,
        mechanism_name="laplace_missing_noise",
        privacy_notion="pure_dp",
        claimed_epsilon=1.0,
        actual_epsilon=float("inf"),
        violation_type="information_leak",
        minimal_counterexample=(
            "db1=[0], db2=[1]: mechanism outputs 0 and 1 respectively. "
            "Density ratio is infinite (0 probability on wrong output)."
        ),
        repair_hint="Add Lap(Δf/ε) noise to the answer.",
        is_repairable_by_scale=False,
        literature_ref="Most basic DP violation",
        tags=("laplace", "information_leak", "no_noise"),
    ),
    BugEntry(
        bug_id="LEAK-003",
        name="Report noisy max using wrong noise distribution",
        description=(
            "Report noisy max should use Laplace noise for pure ε-DP. "
            "Using Gaussian noise breaks the equivalence to the "
            "exponential mechanism and may not satisfy pure DP."
        ),
        category="information_leak",
        mechanism_source=NOISY_MAX_WRONG_NOISE,
        mechanism_name="noisy_max_wrong_noise",
        privacy_notion="pure_dp",
        claimed_epsilon=1.0,
        violation_type="insufficient_noise",
        minimal_counterexample=(
            "Gaussian noise has sub-exponential tails; the density ratio "
            "of Gaussian noise grows quadratically in the output, "
            "violating the bounded-ratio requirement of pure DP."
        ),
        repair_hint="Use Laplace noise instead of Gaussian.",
        is_repairable_by_scale=False,  # needs distribution swap
        literature_ref="Pure DP requires Laplace for additive noise",
        tags=("exponential", "noise_type", "distribution"),
    ),
    BugEntry(
        bug_id="LEAK-004",
        name="Private selection double-dip on data",
        description=(
            "After selecting the best candidate via noisy max, the "
            "mechanism returns the raw (non-private) score of the "
            "selected candidate. This second data access is not "
            "accounted for in the privacy budget."
        ),
        category="information_leak",
        mechanism_source=SELECTION_DOUBLE_DIP,
        mechanism_name="selection_double_dip",
        privacy_notion="pure_dp",
        claimed_epsilon=1.0,
        violation_type="information_leak",
        minimal_counterexample=(
            "The raw score s(db, i*) reveals exact information about "
            "the winning candidate's quality on db."
        ),
        repair_hint="Only return the index, not the raw score.",
        is_repairable_by_scale=False,
        literature_ref="Common mistake in private selection",
        tags=("selection", "double_dip", "information_leak"),
    ),
    BugEntry(
        bug_id="LEAK-005",
        name="Histogram missing noise on bins 1..N-1",
        description=(
            "Only the first bin gets Laplace noise; all other bins are "
            "released without noise. This violates DP for any record "
            "that falls in bins 1..N-1."
        ),
        category="information_leak",
        mechanism_source=HISTOGRAM_MISSING_NOISE,
        mechanism_name="histogram_missing_noise",
        privacy_notion="pure_dp",
        claimed_epsilon=1.0,
        actual_epsilon=float("inf"),
        violation_type="information_leak",
        minimal_counterexample=(
            "A record in bin 2: moving it to a neighbouring dataset "
            "changes bin 2's count by 1 with probability 1."
        ),
        repair_hint="Add Lap(1/ε) noise to every bin.",
        is_repairable_by_scale=False,  # need to ADD noise draws
        literature_ref="Basic histogram DP requirement",
        tags=("histogram", "missing_noise", "partial"),
    ),
    BugEntry(
        bug_id="LEAK-006",
        name="Private selection unaccounted score computation",
        description=(
            "Selection mechanism computes scores from raw data queries "
            "without noise, then selects among them using the exponential "
            "mechanism. The score computation itself leaks information "
            "that is not budgeted."
        ),
        category="information_leak",
        mechanism_source=SELECTION_WRONG_BUDGET,
        mechanism_name="selection_wrong_budget",
        privacy_notion="pure_dp",
        claimed_epsilon=1.0,
        violation_type="budget_overflow",
        minimal_counterexample=(
            "Score computation queries the database k times without noise; "
            "combined with selection, total budget exceeds claimed ε."
        ),
        repair_hint="Add noise to score computation or account for it.",
        is_repairable_by_scale=False,
        literature_ref="Liu & Talwar, STOC 2019",
        tags=("selection", "budget", "unaccounted"),
    ),
]


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def get_bug_catalog() -> BugCatalog:
    """Create and return a populated BugCatalog instance.

    Returns:
        BugCatalog with all known bugs loaded.
    """
    return BugCatalog()


# ---------------------------------------------------------------------------
# Stand-alone execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    catalog = get_bug_catalog()
    catalog.print_catalog()
    print(f"\nSummary: {catalog.summary()}")
