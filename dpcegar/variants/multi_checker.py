"""Multi-variant differential privacy verification.

Orchestrates verification across multiple DP notions, using the
implication lattice to minimize redundant SMT queries.  Verified
results propagate to weaker notions; counterexamples propagate
to stronger ones.

Classes
-------
VariantStatus           – status of a single variant check
VariantResult           – result for one DP notion
DerivedGuarantee        – a guarantee derived via lattice implication
MultiVariantResult      – aggregated result across all notions
MultiVariantStatistics  – performance statistics for multi-variant runs
VerificationPlan        – ordered schedule of variant checks
PlanOptimizer           – builds an efficient verification plan
MultiVariantChecker     – main orchestrator
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Sequence

from dpcegar.ir.types import (
    ApproxBudget,
    FDPBudget,
    GDPBudget,
    PrivacyBudget,
    PrivacyNotion,
    PureBudget,
    RDPBudget,
    ZCDPBudget,
)
from dpcegar.ir.nodes import MechIR
from dpcegar.cegar.engine import CEGAREngine, CEGARResult, CEGARVerdict, CEGARConfig
from dpcegar.paths.symbolic_path import PathSet
from dpcegar.variants.lattice import ImplicationLattice, NodeStatus, PrivacyLatticeNode

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# VARIANT STATUS
# ═══════════════════════════════════════════════════════════════════════════


class VariantStatus(Enum):
    """Outcome of verifying a single DP variant.

    Values:
        VERIFIED:  The CEGAR engine proved the mechanism satisfies this notion.
        FALSIFIED: A genuine counterexample was found.
        UNKNOWN:   The CEGAR engine returned unknown (resource exhaustion, etc.).
        SKIPPED:   Verification was skipped (e.g. no budget supplied).
        DERIVED:   The result was inferred via lattice propagation without
                   invoking the SMT solver.
    """

    VERIFIED = auto()
    FALSIFIED = auto()
    UNKNOWN = auto()
    SKIPPED = auto()
    DERIVED = auto()


# ═══════════════════════════════════════════════════════════════════════════
# VARIANT RESULT
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class VariantResult:
    """Result of verifying a mechanism under a single DP notion.

    Attributes:
        notion:       The DP notion that was checked.
        budget:       The privacy budget that was checked.
        status:       Outcome of the check.
        cegar_result: Full CEGAR result (``None`` if derived or skipped).
        derived_from: If the result was derived via lattice, the source notion.
        time_seconds: Wall-clock time spent on this variant.
    """

    notion: PrivacyNotion
    budget: PrivacyBudget
    status: VariantStatus
    cegar_result: CEGARResult | None = None
    derived_from: PrivacyNotion | None = None
    time_seconds: float = 0.0

    # ------------------------------------------------------------------
    # Convenience predicates
    # ------------------------------------------------------------------

    @property
    def is_verified(self) -> bool:
        """Return True if the variant was verified (directly or derived)."""
        return self.status in (VariantStatus.VERIFIED, VariantStatus.DERIVED)

    @property
    def is_falsified(self) -> bool:
        """Return True if a genuine counterexample was found."""
        return self.status is VariantStatus.FALSIFIED

    @property
    def was_derived(self) -> bool:
        """Return True if the result was inferred via the lattice."""
        return self.status is VariantStatus.DERIVED

    def summary(self) -> str:
        """Return a one-line human-readable summary."""
        parts = [f"{self.notion.name}: {self.status.name}"]
        if self.derived_from is not None:
            parts.append(f"(derived from {self.derived_from.name})")
        parts.append(f"[{self.time_seconds:.3f}s]")
        return " ".join(parts)

    def __str__(self) -> str:
        return self.summary()


# ═══════════════════════════════════════════════════════════════════════════
# DERIVED GUARANTEE
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class DerivedGuarantee:
    """A privacy guarantee obtained by converting a verified budget.

    When a mechanism is verified under one DP notion, the implication
    lattice may allow deriving guarantees under other (weaker) notions
    via known conversion theorems.

    Attributes:
        notion:              The target DP notion.
        budget:              The derived privacy budget in the target notion.
        source_notion:       The DP notion that was directly verified.
        source_budget:       The budget under the source notion.
        conversion_theorem:  Human-readable name of the conversion theorem.
        is_tight:            Whether the conversion is known to be tight.
    """

    notion: PrivacyNotion
    budget: PrivacyBudget
    source_notion: PrivacyNotion
    source_budget: PrivacyBudget
    conversion_theorem: str
    is_tight: bool

    def summary(self) -> str:
        """Return a one-line summary of the derived guarantee."""
        tight_label = "tight" if self.is_tight else "lossy"
        return (
            f"{self.source_notion.name}({self.source_budget}) "
            f"⟹ {self.notion.name}({self.budget}) "
            f"[{self.conversion_theorem}, {tight_label}]"
        )

    def __str__(self) -> str:
        return self.summary()


# ═══════════════════════════════════════════════════════════════════════════
# MULTI-VARIANT STATISTICS
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class MultiVariantStatistics:
    """Aggregate performance statistics for a multi-variant run.

    Attributes:
        per_variant_time:     Wall-clock time spent per notion.
        total_smt_queries:    Total number of SMT solver calls across all variants.
        lattice_propagations: Number of times a result was propagated via lattice.
        queries_saved:        Estimated number of solver queries avoided by
                              lattice propagation.
    """

    per_variant_time: dict[PrivacyNotion, float] = field(default_factory=dict)
    total_smt_queries: int = 0
    lattice_propagations: int = 0
    queries_saved: int = 0

    def record_variant(self, notion: PrivacyNotion, elapsed: float) -> None:
        """Record timing for a completed variant check."""
        self.per_variant_time[notion] = elapsed

    def record_propagation(self, estimated_queries: int = 1) -> None:
        """Record a lattice propagation event."""
        self.lattice_propagations += 1
        self.queries_saved += estimated_queries

    def record_smt_queries(self, count: int) -> None:
        """Accumulate solver query count."""
        self.total_smt_queries += count

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary dictionary."""
        return {
            "per_variant_time": {
                n.name: round(t, 4) for n, t in self.per_variant_time.items()
            },
            "total_smt_queries": self.total_smt_queries,
            "lattice_propagations": self.lattice_propagations,
            "queries_saved": self.queries_saved,
            "total_time": round(sum(self.per_variant_time.values()), 4),
        }


# ═══════════════════════════════════════════════════════════════════════════
# MULTI-VARIANT RESULT
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class MultiVariantResult:
    """Aggregated verification result across multiple DP notions.

    Attributes:
        results:             Per-notion verification results.
        derived_guarantees:  Guarantees inferred via the implication lattice.
        lattice:             The implication lattice used during verification.
        total_time:          Total wall-clock time for the entire run.
        queries_saved:       Number of SMT queries avoided by propagation.
        statistics:          Detailed performance statistics.
    """

    results: dict[PrivacyNotion, VariantResult] = field(default_factory=dict)
    derived_guarantees: list[DerivedGuarantee] = field(default_factory=list)
    lattice: ImplicationLattice | None = None
    total_time: float = 0.0
    queries_saved: int = 0
    statistics: MultiVariantStatistics = field(
        default_factory=MultiVariantStatistics
    )

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def verified_notions(self) -> list[PrivacyNotion]:
        """Return the list of notions that were verified (including derived)."""
        return [
            n for n, r in self.results.items()
            if r.status in (VariantStatus.VERIFIED, VariantStatus.DERIVED)
        ]

    def falsified_notions(self) -> list[PrivacyNotion]:
        """Return the list of notions for which a counterexample was found."""
        return [
            n for n, r in self.results.items()
            if r.status is VariantStatus.FALSIFIED
        ]

    def unknown_notions(self) -> list[PrivacyNotion]:
        """Return the list of notions with unknown status."""
        return [
            n for n, r in self.results.items()
            if r.status is VariantStatus.UNKNOWN
        ]

    def get_tightest_approx(self) -> ApproxBudget | None:
        """Return the tightest (ε, δ) guarantee across all verified notions.

        Converts every verified budget to approximate DP and returns the one
        with the smallest ε (breaking ties by δ).  Returns ``None`` if no
        notion was verified.
        """
        candidates: list[tuple[float, float, ApproxBudget]] = []

        for notion in self.verified_notions():
            vr = self.results[notion]
            eps, delta = vr.budget.to_approx_dp()
            candidates.append((eps, delta, ApproxBudget(epsilon=eps, delta=delta)))

        for dg in self.derived_guarantees:
            eps, delta = dg.budget.to_approx_dp()
            candidates.append((eps, delta, ApproxBudget(epsilon=eps, delta=delta)))

        if not candidates:
            return None

        # Sort by ε ascending, then δ ascending
        candidates.sort(key=lambda t: (t[0], t[1]))
        return candidates[0][2]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a multi-line human-readable summary."""
        lines: list[str] = [
            f"Multi-variant verification  ({self.total_time:.3f}s total, "
            f"{self.queries_saved} queries saved)",
            "─" * 60,
        ]
        for notion, vr in self.results.items():
            lines.append(f"  {vr.summary()}")
        if self.derived_guarantees:
            lines.append("")
            lines.append("Derived guarantees:")
            for dg in self.derived_guarantees:
                lines.append(f"  {dg.summary()}")
        tightest = self.get_tightest_approx()
        if tightest is not None:
            lines.append("")
            lines.append(f"Tightest (ε,δ)-DP: {tightest}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary representation."""
        return {
            "results": {
                notion.name: {
                    "status": vr.status.name,
                    "budget": str(vr.budget),
                    "derived_from": (
                        vr.derived_from.name if vr.derived_from else None
                    ),
                    "time_seconds": round(vr.time_seconds, 4),
                }
                for notion, vr in self.results.items()
            },
            "derived_guarantees": [
                {
                    "notion": dg.notion.name,
                    "budget": str(dg.budget),
                    "source_notion": dg.source_notion.name,
                    "source_budget": str(dg.source_budget),
                    "theorem": dg.conversion_theorem,
                    "is_tight": dg.is_tight,
                }
                for dg in self.derived_guarantees
            ],
            "total_time": round(self.total_time, 4),
            "queries_saved": self.queries_saved,
            "verified_notions": [n.name for n in self.verified_notions()],
            "falsified_notions": [n.name for n in self.falsified_notions()],
            "statistics": self.statistics.summary(),
        }

    def __str__(self) -> str:
        return self.summary()


# ═══════════════════════════════════════════════════════════════════════════
# VERIFICATION PLAN
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class VerificationPlan:
    """Ordered schedule of variant verification checks.

    The plan specifies which notions to verify and in what order, along
    with groups of incomparable notions that may be checked in parallel.

    Attributes:
        ordered_variants:  Notions paired with their budgets, ordered so
                           that lattice propagation is maximally effective.
        skip_if_derived:   If True, skip a variant when the lattice has
                           already determined its status.
        parallel_groups:   Groups of mutually-incomparable notions that
                           can be checked concurrently without losing
                           propagation benefits.
    """

    ordered_variants: list[tuple[PrivacyNotion, PrivacyBudget]] = field(
        default_factory=list
    )
    skip_if_derived: bool = True
    parallel_groups: list[list[PrivacyNotion]] = field(default_factory=list)

    def total_variants(self) -> int:
        """Return the number of variants in the plan."""
        return len(self.ordered_variants)

    def notions(self) -> list[PrivacyNotion]:
        """Return the ordered list of notions."""
        return [n for n, _ in self.ordered_variants]

    def budget_for(self, notion: PrivacyNotion) -> PrivacyBudget | None:
        """Look up the budget for a given notion, or ``None``."""
        for n, b in self.ordered_variants:
            if n is notion:
                return b
        return None

    def summary(self) -> str:
        """Return a human-readable summary of the plan."""
        lines = [f"VerificationPlan ({self.total_variants()} variants):"]
        for i, (n, b) in enumerate(self.ordered_variants, 1):
            lines.append(f"  {i}. {n.name}  budget={b}")
        if self.parallel_groups:
            lines.append("Parallel groups:")
            for j, group in enumerate(self.parallel_groups, 1):
                names = ", ".join(n.name for n in group)
                lines.append(f"  group {j}: [{names}]")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


# ═══════════════════════════════════════════════════════════════════════════
# PLAN OPTIMIZER
# ═══════════════════════════════════════════════════════════════════════════


# Heuristic SMT cost per notion (lower = cheaper to verify).
_DEFAULT_COSTS: dict[PrivacyNotion, float] = {
    PrivacyNotion.PURE_DP: 1.0,
    PrivacyNotion.APPROX_DP: 1.5,
    PrivacyNotion.ZCDP: 2.0,
    PrivacyNotion.RDP: 3.0,
    PrivacyNotion.GDP: 2.5,
    PrivacyNotion.FDP: 4.0,
}


class PlanOptimizer:
    """Builds an efficient :class:`VerificationPlan`.

    The optimizer orders variants so that verification results
    propagate maximally through the implication lattice:

    * Start from the weakest (most general) notion — typically
      (ε, δ)-DP — so that a verification success there can be
      propagated to all weaker specialisations.
    * Group incomparable notions for potential parallel checking.
    * Estimate solver cost per notion to break ties.
    """

    def __init__(
        self,
        cost_model: dict[PrivacyNotion, float] | None = None,
    ) -> None:
        """Initialise the optimizer.

        Args:
            cost_model: Optional per-notion cost estimates (higher = more
                        expensive).  Defaults to built-in heuristics.
        """
        self._costs = cost_model or dict(_DEFAULT_COSTS)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(
        self,
        requested: list[tuple[PrivacyNotion, PrivacyBudget]],
        lattice: ImplicationLattice,
    ) -> VerificationPlan:
        """Build an optimised verification plan.

        Args:
            requested: The set of (notion, budget) pairs the caller wants
                       to verify.
            lattice:   The implication lattice governing notion ordering.

        Returns:
            An ordered :class:`VerificationPlan`.
        """
        if not requested:
            return VerificationPlan()

        ordered = self._compute_order(requested, lattice)
        parallel_groups = self._find_parallel_groups(
            [n for n, _ in ordered], lattice
        )

        return VerificationPlan(
            ordered_variants=ordered,
            skip_if_derived=True,
            parallel_groups=parallel_groups,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_order(
        self,
        requested: list[tuple[PrivacyNotion, PrivacyBudget]],
        lattice: ImplicationLattice,
    ) -> list[tuple[PrivacyNotion, PrivacyBudget]]:
        """Order variants for maximal lattice propagation.

        Strategy:
          1. Stronger notions first — a positive result propagates
             downward to all weaker notions.
          2. Among incomparable notions, cheaper ones first.

        The lattice provides ``implies(a, b)`` to determine strength
        ordering.  We perform a topological sort on the subset of
        requested notions according to the lattice's partial order,
        processing strongest first.
        """
        notion_to_budget = {n: b for n, b in requested}
        notions = list(notion_to_budget.keys())

        # Build a DAG on the requested notions using the lattice.
        # Edge (a, b) means "a implies b" (a is stronger).
        stronger_than: dict[PrivacyNotion, list[PrivacyNotion]] = {
            n: [] for n in notions
        }
        in_degree: dict[PrivacyNotion, int] = {n: 0 for n in notions}

        for a in notions:
            for b in notions:
                if a is not b and lattice.implies(a, b):
                    stronger_than[a].append(b)
                    in_degree[b] += 1

        # Kahn's algorithm with cost-based tie-breaking.
        queue: list[PrivacyNotion] = sorted(
            [n for n in notions if in_degree[n] == 0],
            key=lambda n: self._estimate_cost(n),
        )
        ordered: list[tuple[PrivacyNotion, PrivacyBudget]] = []

        while queue:
            node = queue.pop(0)
            ordered.append((node, notion_to_budget[node]))
            for child in stronger_than[node]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
            queue.sort(key=lambda n: self._estimate_cost(n))

        # If cyclic edges caused some notions to be dropped, append them.
        seen = {n for n, _ in ordered}
        for n in notions:
            if n not in seen:
                ordered.append((n, notion_to_budget[n]))

        return ordered

    def _find_parallel_groups(
        self,
        notions: list[PrivacyNotion],
        lattice: ImplicationLattice,
    ) -> list[list[PrivacyNotion]]:
        """Group mutually-incomparable notions for parallel checking.

        Two notions are *incomparable* when neither implies the other.
        Checking them in parallel is safe because neither result can
        be derived from the other.

        Returns a list of groups (each group is a list of notions).
        """
        groups: list[list[PrivacyNotion]] = []
        remaining = list(notions)

        while remaining:
            group = [remaining.pop(0)]
            still_remaining: list[PrivacyNotion] = []

            for candidate in remaining:
                is_comparable = any(
                    lattice.implies(candidate, g) or lattice.implies(g, candidate)
                    for g in group
                )
                if is_comparable:
                    still_remaining.append(candidate)
                else:
                    group.append(candidate)

            if len(group) > 1:
                groups.append(group)
            remaining = still_remaining

        return groups

    def _estimate_cost(self, notion: PrivacyNotion) -> float:
        """Return a heuristic cost for verifying *notion*.

        Lower values mean cheaper / faster.  Used to break ties in the
        topological sort.
        """
        return self._costs.get(notion, 5.0)


# ═══════════════════════════════════════════════════════════════════════════
# CONVERSION THEOREM REGISTRY
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class _ConversionEntry:
    """Internal record of a known conversion theorem."""

    source: PrivacyNotion
    target: PrivacyNotion
    theorem_name: str
    is_tight: bool
    convert: Callable[[PrivacyBudget], PrivacyBudget | None]


def _pure_to_approx(budget: PrivacyBudget) -> PrivacyBudget | None:
    """Convert pure ε-DP to (ε, 0)-DP."""
    if isinstance(budget, PureBudget):
        return ApproxBudget(epsilon=budget.epsilon, delta=0.0)
    return None


def _zcdp_to_approx(budget: PrivacyBudget) -> PrivacyBudget | None:
    """Convert ρ-zCDP to (ε, δ)-DP via Bun–Steinke '16."""
    if isinstance(budget, ZCDPBudget):
        delta = 1e-5
        eps = budget.rho + 2.0 * math.sqrt(budget.rho * math.log(1.0 / delta))
        return ApproxBudget(epsilon=eps, delta=delta)
    return None


def _rdp_to_approx(budget: PrivacyBudget) -> PrivacyBudget | None:
    """Convert (α, ε)-RDP to (ε', δ)-DP via Mironov '17."""
    if isinstance(budget, RDPBudget):
        delta = 1e-5
        eps = budget.epsilon - math.log(delta) / (budget.alpha - 1.0)
        return ApproxBudget(epsilon=max(eps, 0.0), delta=delta)
    return None


def _gdp_to_approx(budget: PrivacyBudget) -> PrivacyBudget | None:
    """Convert μ-GDP to (ε, δ)-DP via Dong–Roth–Su '19."""
    if isinstance(budget, GDPBudget):
        delta = 1e-5
        # Use the Gaussian mechanism's conversion: ε = μ·Φ⁻¹(1−δ) + μ²/2
        # Approximate Φ⁻¹(1−δ) ≈ √(2·ln(1/δ)) for small δ.
        phi_inv_approx = math.sqrt(2.0 * math.log(1.0 / delta))
        eps = budget.mu * phi_inv_approx + 0.5 * budget.mu ** 2
        return ApproxBudget(epsilon=max(eps, 0.0), delta=delta)
    return None


def _pure_to_zcdp(budget: PrivacyBudget) -> PrivacyBudget | None:
    """Convert ε-DP to (ε²/2)-zCDP via Bun–Steinke '16."""
    if isinstance(budget, PureBudget):
        return ZCDPBudget(rho=0.5 * budget.epsilon ** 2)
    return None


def _zcdp_to_rdp(budget: PrivacyBudget) -> PrivacyBudget | None:
    """Convert ρ-zCDP to (α, αρ)-RDP."""
    if isinstance(budget, ZCDPBudget):
        alpha = 2.0
        return RDPBudget(alpha=alpha, epsilon=alpha * budget.rho)
    return None


_BUILTIN_CONVERSIONS: list[_ConversionEntry] = [
    _ConversionEntry(
        PrivacyNotion.PURE_DP, PrivacyNotion.APPROX_DP,
        "Pure ⊆ Approx (trivial)", True, _pure_to_approx,
    ),
    _ConversionEntry(
        PrivacyNotion.ZCDP, PrivacyNotion.APPROX_DP,
        "zCDP → (ε,δ)-DP (Bun–Steinke '16)", False, _zcdp_to_approx,
    ),
    _ConversionEntry(
        PrivacyNotion.RDP, PrivacyNotion.APPROX_DP,
        "RDP → (ε,δ)-DP (Mironov '17)", False, _rdp_to_approx,
    ),
    _ConversionEntry(
        PrivacyNotion.GDP, PrivacyNotion.APPROX_DP,
        "GDP → (ε,δ)-DP (Dong–Roth–Su '19)", False, _gdp_to_approx,
    ),
    _ConversionEntry(
        PrivacyNotion.PURE_DP, PrivacyNotion.ZCDP,
        "ε-DP → (ε²/2)-zCDP (Bun–Steinke '16)", False, _pure_to_zcdp,
    ),
    _ConversionEntry(
        PrivacyNotion.ZCDP, PrivacyNotion.RDP,
        "ρ-zCDP → (α,αρ)-RDP", True, _zcdp_to_rdp,
    ),
]


# ═══════════════════════════════════════════════════════════════════════════
# MULTI-VARIANT CHECKER
# ═══════════════════════════════════════════════════════════════════════════


class MultiVariantChecker:
    """Orchestrate verification across multiple DP notions.

    The checker walks the implication lattice so that:

    * A **verified** result on a *stronger* notion is propagated to
      all *weaker* notions it implies — those weaker checks are skipped.
    * A **counterexample** found for a *weaker* notion is propagated
      *upward* to all *stronger* notions — they are marked falsified.

    This can save significant solver time when a mechanism is verified
    under a strong notion (e.g. pure DP), automatically covering all
    weaker notions (approx DP, zCDP, etc.).

    Usage::

        checker = MultiVariantChecker(engine=my_engine)
        result = checker.check_all(mech, {
            PrivacyNotion.PURE_DP:   PureBudget(epsilon=1.0),
            PrivacyNotion.APPROX_DP: ApproxBudget(epsilon=1.0, delta=1e-5),
        })
        print(result.summary())
    """

    def __init__(
        self,
        engine: CEGAREngine | None = None,
        config: CEGARConfig | None = None,
        lattice: ImplicationLattice | None = None,
        plan_optimizer: PlanOptimizer | None = None,
        conversions: list[_ConversionEntry] | None = None,
    ) -> None:
        """Initialise the multi-variant checker.

        Args:
            engine:         A pre-configured CEGAR engine.  If ``None``, a
                            fresh engine is constructed from *config*.
            config:         CEGAR configuration (used only when *engine* is
                            ``None``).
            lattice:        Implication lattice.  If ``None``, a default
                            lattice is built.
            plan_optimizer: Optional custom plan optimizer.
            conversions:    Optional custom conversion theorems.
        """
        self._config = config or CEGARConfig()
        self._engine = engine or CEGAREngine(config=self._config)
        self._lattice = lattice or ImplicationLattice()
        self._optimizer = plan_optimizer or PlanOptimizer()
        self._conversions = conversions if conversions is not None else list(
            _BUILTIN_CONVERSIONS
        )
        self._stats = MultiVariantStatistics()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_all(
        self,
        mechanism: MechIR,
        budgets: dict[PrivacyNotion, PrivacyBudget],
        path_set: PathSet | None = None,
    ) -> MultiVariantResult:
        """Check a mechanism against every requested DP notion.

        Args:
            mechanism: The mechanism IR to verify.
            budgets:   Mapping from DP notion to the budget to verify.
            path_set:  Optional pre-enumerated symbolic paths.

        Returns:
            A :class:`MultiVariantResult` aggregating all per-notion results.
        """
        start = time.monotonic()
        self._stats = MultiVariantStatistics()

        if not budgets:
            budgets = self._build_default_budgets(mechanism)

        requested = list(budgets.items())
        plan = self._optimizer.optimize(requested, self._lattice)
        logger.info("Verification plan: %s", plan)

        result = MultiVariantResult(lattice=self._lattice)

        for notion, budget in plan.ordered_variants:
            if plan.skip_if_derived and self._should_skip(notion, result):
                logger.info("Skipping %s (already derived)", notion.name)
                continue

            vr = self.check_single(mechanism, notion, budget, path_set)
            result.results[notion] = vr
            self._stats.record_variant(notion, vr.time_seconds)

            if vr.cegar_result and vr.cegar_result.statistics:
                self._stats.record_smt_queries(
                    vr.cegar_result.statistics.solver_calls
                )

            self._propagate_result(notion, vr, result)

        derived = self._compute_derived_guarantees(result)
        result.derived_guarantees = derived

        result.total_time = time.monotonic() - start
        result.queries_saved = self._stats.queries_saved
        result.statistics = self._stats
        return result

    def check_single(
        self,
        mechanism: MechIR,
        notion: PrivacyNotion,
        budget: PrivacyBudget,
        path_set: PathSet | None = None,
    ) -> VariantResult:
        """Run the CEGAR engine for a single DP notion.

        Args:
            mechanism: The mechanism IR to verify.
            notion:    The DP notion to check.
            budget:    The privacy budget.
            path_set:  Optional pre-enumerated paths.

        Returns:
            A :class:`VariantResult` for this notion.
        """
        start = time.monotonic()
        logger.info("Checking %s with budget %s", notion.name, budget)

        cegar_result = self._engine.verify_mechanism(
            mechanism, budget, path_set=path_set
        )
        elapsed = time.monotonic() - start

        status = _verdict_to_variant_status(cegar_result.verdict)

        return VariantResult(
            notion=notion,
            budget=budget,
            status=status,
            cegar_result=cegar_result,
            derived_from=None,
            time_seconds=elapsed,
        )

    # ------------------------------------------------------------------
    # Default budget construction
    # ------------------------------------------------------------------

    def _build_default_budgets(
        self, mechanism: MechIR
    ) -> dict[PrivacyNotion, PrivacyBudget]:
        """Construct default budgets from a mechanism's declared budget.

        If the mechanism already declares a budget, convert it to every
        notion in the lattice.  Otherwise, fall back to a conservative
        ``PureBudget(ε=1.0)`` and its conversions.

        Args:
            mechanism: The mechanism IR.

        Returns:
            A mapping from each notion to a budget.
        """
        base = mechanism.budget or PureBudget(epsilon=1.0)
        budgets: dict[PrivacyNotion, PrivacyBudget] = {base.notion: base}

        for entry in self._conversions:
            if entry.source == base.notion:
                converted = entry.convert(base)
                if converted is not None:
                    budgets[entry.target] = converted

        # If the base was not ApproxDP, ensure we have an ApproxDP budget.
        if PrivacyNotion.APPROX_DP not in budgets:
            eps, delta = base.to_approx_dp()
            budgets[PrivacyNotion.APPROX_DP] = ApproxBudget(
                epsilon=eps, delta=delta
            )

        return budgets

    # ------------------------------------------------------------------
    # Lattice propagation
    # ------------------------------------------------------------------

    def _propagate_result(
        self,
        notion: PrivacyNotion,
        variant_result: VariantResult,
        multi_result: MultiVariantResult,
    ) -> None:
        """Propagate a verification or falsification through the lattice.

        Verified results propagate *downward* (strong ⟹ weak):
            If Pure DP is verified, Approx DP is automatically verified.

        Falsification results propagate *upward* (weak ⟹ strong):
            If Approx DP is falsified, Pure DP is automatically falsified.

        Args:
            notion:         The notion whose result is being propagated.
            variant_result: The result to propagate.
            multi_result:   The aggregated result to update.
        """
        lattice = self._lattice

        if variant_result.status is VariantStatus.VERIFIED:
            # Verified under a strong notion → all weaker notions hold too.
            for other_notion in PrivacyNotion:
                if other_notion is notion:
                    continue
                if other_notion in multi_result.results:
                    continue
                if lattice.implies(notion, other_notion):
                    converted_budget = self._convert_budget(
                        variant_result.budget, notion, other_notion
                    )
                    if converted_budget is not None:
                        multi_result.results[other_notion] = VariantResult(
                            notion=other_notion,
                            budget=converted_budget,
                            status=VariantStatus.DERIVED,
                            cegar_result=None,
                            derived_from=notion,
                            time_seconds=0.0,
                        )
                        self._stats.record_propagation()
                        logger.info(
                            "Derived %s from %s",
                            other_notion.name, notion.name,
                        )

        elif variant_result.status is VariantStatus.FALSIFIED:
            # Falsified under a weak notion → all stronger notions fail too.
            for other_notion in PrivacyNotion:
                if other_notion is notion:
                    continue
                if other_notion in multi_result.results:
                    continue
                if lattice.implies(other_notion, notion):
                    multi_result.results[other_notion] = VariantResult(
                        notion=other_notion,
                        budget=variant_result.budget,
                        status=VariantStatus.FALSIFIED,
                        cegar_result=variant_result.cegar_result,
                        derived_from=notion,
                        time_seconds=0.0,
                    )
                    self._stats.record_propagation()
                    logger.info(
                        "Propagated falsification of %s to %s",
                        notion.name, other_notion.name,
                    )

    def _convert_budget(
        self,
        budget: PrivacyBudget,
        source: PrivacyNotion,
        target: PrivacyNotion,
    ) -> PrivacyBudget | None:
        """Convert a budget from one notion to another.

        Searches the registered conversion theorems for a direct
        conversion from *source* to *target*.  Falls back to the
        budget's ``to_approx_dp`` if the target is ApproxDP.

        Args:
            budget: The source budget.
            source: Source DP notion.
            target: Target DP notion.

        Returns:
            The converted budget, or ``None`` if no conversion is known.
        """
        for entry in self._conversions:
            if entry.source == source and entry.target == target:
                result = entry.convert(budget)
                if result is not None:
                    return result

        # Fallback: any budget can convert to ApproxDP
        if target is PrivacyNotion.APPROX_DP:
            eps, delta = budget.to_approx_dp()
            return ApproxBudget(epsilon=eps, delta=delta)

        return None

    def _should_skip(
        self,
        notion: PrivacyNotion,
        multi_result: MultiVariantResult,
    ) -> bool:
        """Return True if the lattice has already determined *notion*.

        A notion should be skipped if it already appears in the results
        (either directly checked or derived).

        Args:
            notion:       The notion to check.
            multi_result: The current aggregated result.

        Returns:
            True if the notion can be skipped.
        """
        return notion in multi_result.results

    # ------------------------------------------------------------------
    # Derived guarantees
    # ------------------------------------------------------------------

    def _compute_derived_guarantees(
        self,
        multi_result: MultiVariantResult,
    ) -> list[DerivedGuarantee]:
        """Compute all derived guarantees from verified results.

        For every verified notion, apply all known conversion theorems
        to produce guarantees under other notions.  Includes both
        direct propagation results and additional conversions that may
        not have been requested in the original budget set.

        Args:
            multi_result: The current aggregated result.

        Returns:
            A list of derived guarantees.
        """
        guarantees: list[DerivedGuarantee] = []
        seen: set[tuple[PrivacyNotion, PrivacyNotion]] = set()

        for notion, vr in multi_result.results.items():
            if vr.status not in (VariantStatus.VERIFIED, VariantStatus.DERIVED):
                continue

            for entry in self._conversions:
                if entry.source != notion:
                    continue
                key = (notion, entry.target)
                if key in seen:
                    continue
                seen.add(key)

                converted = entry.convert(vr.budget)
                if converted is None:
                    continue

                guarantees.append(DerivedGuarantee(
                    notion=entry.target,
                    budget=converted,
                    source_notion=notion,
                    source_budget=vr.budget,
                    conversion_theorem=entry.theorem_name,
                    is_tight=entry.is_tight,
                ))

        return guarantees


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════


def _verdict_to_variant_status(verdict: CEGARVerdict) -> VariantStatus:
    """Map a CEGAR verdict to a variant status.

    Args:
        verdict: The CEGAR engine's verdict.

    Returns:
        The corresponding :class:`VariantStatus`.
    """
    mapping: dict[CEGARVerdict, VariantStatus] = {
        CEGARVerdict.VERIFIED: VariantStatus.VERIFIED,
        CEGARVerdict.COUNTEREXAMPLE: VariantStatus.FALSIFIED,
        CEGARVerdict.UNKNOWN: VariantStatus.UNKNOWN,
        CEGARVerdict.TIMEOUT: VariantStatus.UNKNOWN,
        CEGARVerdict.ERROR: VariantStatus.UNKNOWN,
    }
    return mapping.get(verdict, VariantStatus.UNKNOWN)


def check_mechanism_all_variants(
    mechanism: MechIR,
    budgets: dict[PrivacyNotion, PrivacyBudget] | None = None,
    config: CEGARConfig | None = None,
    path_set: PathSet | None = None,
) -> MultiVariantResult:
    """Convenience function: verify *mechanism* under all requested notions.

    This is the simplest entry point for multi-variant verification.
    It constructs a :class:`MultiVariantChecker` internally, builds
    default budgets if none are given, and returns the aggregated result.

    Args:
        mechanism: The mechanism IR to verify.
        budgets:   Optional mapping from notion to budget.  If ``None``,
                   defaults are derived from the mechanism's budget.
        config:    Optional CEGAR configuration.
        path_set:  Optional pre-enumerated paths.

    Returns:
        A :class:`MultiVariantResult` with results for every notion.

    Example::

        result = check_mechanism_all_variants(
            mechanism,
            budgets={
                PrivacyNotion.PURE_DP: PureBudget(epsilon=1.0),
                PrivacyNotion.ZCDP:    ZCDPBudget(rho=0.5),
            },
        )
        print(result.summary())
    """
    checker = MultiVariantChecker(config=config)
    return checker.check_all(mechanism, budgets or {}, path_set=path_set)
