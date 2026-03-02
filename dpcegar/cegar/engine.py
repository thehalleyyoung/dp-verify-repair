"""Main CEGAR loop for differential privacy verification.

Implements the Counter-Example Guided Abstraction Refinement (CEGAR) loop
for verifying that a mechanism satisfies a given privacy budget.

The loop proceeds as:
  1. **Abstract verify** — check if abstract density bounds imply privacy
  2. **Extract candidate** — if verification fails, extract a candidate
     counterexample witnessing the violation
  3. **Concretize** — check if the candidate is a genuine violation
  4. **Refine** — if spurious, refine the abstraction and repeat

Classes
-------
CEGARResult        – outcome of the full CEGAR verification
CEGARConfig        – configuration for the CEGAR engine
CEGAREngine        – main CEGAR loop driver
AbstractVerifier   – abstract privacy checking
CandidateExtractor – counterexample extraction from abstract failures
ConcreteChecker    – concretization of candidate counterexamples
SpuriousnessAnalyzer – analysis of spurious counterexamples
ConvergenceTracker – track progress and estimate remaining work
CEGARStatistics    – timing and counter statistics
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)

from dpcegar.ir.types import (
    BinOp,
    BinOpKind,
    Const,
    IRType,
    NoiseKind,
    PrivacyBudget,
    PrivacyNotion,
    PureBudget,
    ApproxBudget,
    ZCDPBudget,
    TypedExpr,
    Var,
)
from dpcegar.ir.nodes import MechIR, NoiseDrawNode
from dpcegar.paths.symbolic_path import (
    NoiseDrawInfo,
    PathCondition,
    PathSet,
    SymbolicPath,
)
from dpcegar.density.ratio_builder import DensityRatioExpr, DensityRatioResult
from dpcegar.density.privacy_loss import PrivacyLossResult
from dpcegar.cegar.abstraction import (
    AbstractDensityBound,
    AbstractionState,
    AbstractState,
    InitialAbstraction,
    PathPartition,
    RefinementKind,
    RefinementRecord,
    WideningOperator,
)
from dpcegar.cegar.refinement import (
    RefinementCounterexample,
    ConvergenceDetector,
    ConvergenceReason,
    ConvergenceStatus,
    InfeasibilityProof,
    RefinementHistory,
    RefinementOperator,
    RefinementResult,
    RefinementSelector,
    RefinementStatus,
)
from dpcegar.utils.errors import (
    DPCegarError,
    InternalError,
    PrivacyViolation,
    VerificationError,
    ensure,
)


# ═══════════════════════════════════════════════════════════════════════════
# CEGAR RESULT
# ═══════════════════════════════════════════════════════════════════════════


class CEGARVerdict(Enum):
    """Outcome of CEGAR verification."""

    VERIFIED = auto()
    COUNTEREXAMPLE = auto()
    UNKNOWN = auto()
    TIMEOUT = auto()
    ERROR = auto()


@dataclass(slots=True)
class CEGARResult:
    """Result of the CEGAR verification loop.

    Attributes:
        verdict: The verification outcome.
        budget: The privacy budget that was checked.
        certificate: Verification certificate (if VERIFIED).
        counterexample: Concrete counterexample (if COUNTEREXAMPLE).
        final_bounds: Final density bounds from the abstraction.
        statistics: Performance and iteration statistics.
        final_abstraction: The final abstraction state.
        details: Additional information.
    """

    verdict: CEGARVerdict = CEGARVerdict.UNKNOWN
    budget: PrivacyBudget | None = None
    certificate: VerificationCertificate | None = None
    counterexample: RefinementCounterexample | None = None
    final_bounds: AbstractDensityBound | None = None
    statistics: CEGARStatistics | None = None
    final_abstraction: AbstractionState | None = None
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def is_verified(self) -> bool:
        """Return True if the mechanism was verified private."""
        return self.verdict == CEGARVerdict.VERIFIED

    @property
    def is_violated(self) -> bool:
        """Return True if a genuine privacy violation was found."""
        return self.verdict == CEGARVerdict.COUNTEREXAMPLE

    def summary(self) -> str:
        """Return a human-readable summary."""
        parts = [f"CEGAR: {self.verdict.name}"]
        if self.budget:
            parts.append(f"budget={self.budget}")
        if self.final_bounds:
            parts.append(f"bounds={self.final_bounds}")
        if self.statistics:
            parts.append(f"iters={self.statistics.iterations}")
        return ", ".join(parts)

    def __str__(self) -> str:
        return self.summary()


@dataclass(slots=True)
class VerificationCertificate:
    """Certificate that a mechanism satisfies a privacy budget.

    Attributes:
        budget: The verified privacy budget.
        density_bounds: Per-state density bounds that prove privacy.
        abstraction_state: The final abstraction used in the proof.
        proof_steps: High-level description of the proof strategy.
    """

    budget: PrivacyBudget | None = None
    density_bounds: dict[str, AbstractDensityBound] = field(default_factory=dict)
    abstraction_state: AbstractionState | None = None
    proof_steps: list[str] = field(default_factory=list)

    def is_valid(self) -> bool:
        """Check if the certificate is internally consistent.

        Returns:
            True if all density bounds satisfy the budget.
        """
        if self.budget is None:
            return False
        eps, _ = self.budget.to_approx_dp()
        for sid, bound in self.density_bounds.items():
            if not bound.satisfies_epsilon(eps):
                return False
        return True

    def __str__(self) -> str:
        return f"Certificate(budget={self.budget}, states={len(self.density_bounds)})"


# ═══════════════════════════════════════════════════════════════════════════
# CEGAR CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class CEGARConfig:
    """Configuration for the CEGAR engine.

    Attributes:
        max_refinements: Maximum number of refinement iterations.
        timeout_seconds: Overall timeout for the CEGAR loop.
        solver_timeout_seconds: Per-query SMT solver timeout.
        initial_abstraction: Strategy for creating the initial abstraction.
        enable_widening: Whether to enable the widening operator.
        widening_threshold: Finite bound for widening.
        widening_patience: Iterations before triggering widening.
        enable_narrowing: Whether to apply narrowing after fixpoint.
        fixpoint_tolerance: Tolerance for fixpoint detection.
        progress_callback: Optional callback for progress updates.
    """

    max_refinements: int = 100
    timeout_seconds: float = 300.0
    solver_timeout_seconds: float = 30.0
    initial_abstraction: str = "noise_pattern"
    enable_widening: bool = True
    widening_threshold: float = 100.0
    widening_patience: int = 5
    enable_narrowing: bool = True
    fixpoint_tolerance: float = 1e-8
    enable_incremental_feasibility: bool = True
    progress_callback: Callable[[str, int, int], None] | None = None

    def validate(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If any parameter is out of range.
        """
        if self.max_refinements < 1:
            raise ValueError("max_refinements must be >= 1")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")
        if self.solver_timeout_seconds <= 0:
            raise ValueError("solver_timeout_seconds must be > 0")
        if self.initial_abstraction not in (
            "coarsest", "noise_pattern", "branch_structure", "finest"
        ):
            raise ValueError(
                f"Unknown initial_abstraction: {self.initial_abstraction}"
            )


# ═══════════════════════════════════════════════════════════════════════════
# CEGAR STATISTICS
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class CEGARStatistics:
    """Performance statistics for a CEGAR run.

    Attributes:
        iterations: Number of CEGAR iterations completed.
        refinements: Number of refinements applied.
        abstract_verifications: Number of abstract verification calls.
        concrete_checks: Number of concrete concretization checks.
        spurious_cex_count: Number of spurious counterexamples encountered.
        genuine_cex_count: Number of genuine counterexamples found.
        solver_calls: Total number of SMT solver calls.
        total_time: Total wall-clock time.
        abstract_verify_time: Time spent in abstract verification.
        concrete_check_time: Time spent in concretization checks.
        refinement_time: Time spent in refinement.
        solver_time: Total SMT solver time.
        peak_abstract_states: Maximum number of abstract states.
        final_abstract_states: Number of abstract states at termination.
    """

    iterations: int = 0
    refinements: int = 0
    abstract_verifications: int = 0
    concrete_checks: int = 0
    spurious_cex_count: int = 0
    genuine_cex_count: int = 0
    solver_calls: int = 0
    total_time: float = 0.0
    abstract_verify_time: float = 0.0
    concrete_check_time: float = 0.0
    refinement_time: float = 0.0
    solver_time: float = 0.0
    peak_abstract_states: int = 0
    final_abstract_states: int = 0

    @property
    def refinement_count(self) -> int:
        """Number of refinements recorded."""
        return self.refinements

    @property
    def smt_call_count(self) -> int:
        """Number of SMT solver calls recorded."""
        return self.solver_calls

    @property
    def total_time_seconds(self) -> float:
        """Total wall-clock time in seconds."""
        return self.total_time

    def record_refinement(self, time_seconds: float) -> None:
        """Record a refinement step."""
        self.refinements += 1
        self.refinement_time += time_seconds
        self.total_time += time_seconds

    def record_smt_call(self, time_seconds: float, result: Any = None) -> None:
        """Record an SMT solver call."""
        self.solver_calls += 1
        self.solver_time += time_seconds
        self.total_time += time_seconds

    def update_peak_states(self, current: int) -> None:
        """Update the peak abstract state count.

        Args:
            current: Current number of abstract states.
        """
        self.peak_abstract_states = max(self.peak_abstract_states, current)

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary summary of all statistics."""
        return {
            "iterations": self.iterations,
            "refinements": self.refinements,
            "refinement_count": self.refinements,
            "abstract_verifications": self.abstract_verifications,
            "concrete_checks": self.concrete_checks,
            "spurious_counterexamples": self.spurious_cex_count,
            "genuine_counterexamples": self.genuine_cex_count,
            "solver_calls": self.solver_calls,
            "total_time_s": round(self.total_time, 3),
            "abstract_verify_time_s": round(self.abstract_verify_time, 3),
            "concrete_check_time_s": round(self.concrete_check_time, 3),
            "refinement_time_s": round(self.refinement_time, 3),
            "peak_abstract_states": self.peak_abstract_states,
            "final_abstract_states": self.final_abstract_states,
        }

    def summary(self) -> str:
        """Return a human-readable summary string."""
        return (
            f"CEGARStats(iters={self.iterations}, "
            f"refine={self.refinements}, "
            f"smt_calls={self.solver_calls}, "
            f"time={self.total_time:.2f}s)"
        )

    def __str__(self) -> str:
        return (
            f"CEGARStats(iters={self.iterations}, "
            f"refine={self.refinements}, "
            f"time={self.total_time:.2f}s)"
        )


# ═══════════════════════════════════════════════════════════════════════════
# SMT SOLVER INTERFACE (protocol for decoupling from z3)
# ═══════════════════════════════════════════════════════════════════════════


class SMTSolverInterface(Protocol):
    """Protocol for the SMT solver used in CEGAR.

    This defines the interface that the SMT module must implement.
    The CEGAR engine calls these methods without importing z3 directly.
    """

    def check_sat(
        self,
        constraints: list[TypedExpr],
        timeout_ms: int,
    ) -> SMTResult:
        """Check satisfiability of a set of constraints.

        Args:
            constraints: List of boolean-typed expressions.
            timeout_ms: Solver timeout in milliseconds.

        Returns:
            An SMTResult with sat/unsat/unknown status.
        """
        ...

    def get_model(self) -> dict[str, float]:
        """Extract a satisfying model after a SAT result.

        Returns:
            Mapping from variable names to concrete values.
        """
        ...

    def get_unsat_core(self) -> list[TypedExpr]:
        """Extract an unsat core after an UNSAT result.

        Returns:
            Subset of constraints that are unsatisfiable.
        """
        ...

    def compute_interpolant(
        self,
        group_a: list[TypedExpr],
        group_b: list[TypedExpr],
    ) -> TypedExpr | None:
        """Compute a Craig interpolant between two constraint groups.

        Args:
            group_a: First group of constraints.
            group_b: Second group of constraints.

        Returns:
            An interpolant, or None if not supported.
        """
        ...

    def maximize(
        self,
        objective: TypedExpr,
        constraints: list[TypedExpr],
        timeout_ms: int,
    ) -> tuple[str, float | None]:
        """Maximize an objective under constraints (for OMT).

        Args:
            objective: Expression to maximize.
            constraints: Hard constraints.
            timeout_ms: Solver timeout.

        Returns:
            Tuple of (status, optimal_value) where status is "optimal"/"unsat"/"unknown".
        """
        ...


class SMTStatus(Enum):
    """Status returned by the SMT solver."""

    SAT = auto()
    UNSAT = auto()
    UNKNOWN = auto()
    TIMEOUT = auto()


@dataclass(slots=True)
class SMTResult:
    """Result from an SMT solver query.

    Attributes:
        status: SAT/UNSAT/UNKNOWN/TIMEOUT.
        model: Satisfying assignment (if SAT).
        unsat_core: Unsatisfiable core (if UNSAT).
        time_seconds: Time taken for the query.
    """

    status: SMTStatus = SMTStatus.UNKNOWN
    model: dict[str, float] = field(default_factory=dict)
    unsat_core: list[TypedExpr] = field(default_factory=list)
    time_seconds: float = 0.0

    @property
    def is_sat(self) -> bool:
        """Return True if the result is SAT."""
        return self.status == SMTStatus.SAT

    @property
    def is_unsat(self) -> bool:
        """Return True if the result is UNSAT."""
        return self.status == SMTStatus.UNSAT


# ═══════════════════════════════════════════════════════════════════════════
# ABSTRACT VERIFIER
# ═══════════════════════════════════════════════════════════════════════════


class AbstractVerifier:
    """Check if abstract density bounds imply the privacy predicate.

    For pure ε-DP, this checks: ∀ abstract state s, bound(s) ⊆ [-ε, ε].
    For approximate (ε,δ)-DP, it computes the hockey-stick divergence
    bound from the abstract intervals.
    """

    def verify(
        self,
        abstraction: AbstractionState,
        budget: PrivacyBudget,
    ) -> AbstractVerificationResult:
        """Verify privacy under the current abstraction.

        Checks whether the abstract density bounds for all states
        are sufficient to guarantee the privacy budget.

        Args:
            abstraction: Current abstraction state.
            budget: Privacy budget to verify against.

        Returns:
            An AbstractVerificationResult.
        """
        eps, delta = budget.to_approx_dp()

        violating_states: list[str] = []
        max_loss = 0.0

        for sid in abstraction.all_state_ids():
            bound = abstraction.get_abstract_density(sid)
            abs_max = max(abs(bound.lo), abs(bound.hi))
            max_loss = max(max_loss, abs_max)

            if not bound.satisfies_epsilon(eps):
                violating_states.append(sid)

        if not violating_states:
            return AbstractVerificationResult(
                verified=True,
                max_abstract_loss=max_loss,
                details={
                    "epsilon": eps,
                    "delta": delta,
                    "states_checked": len(abstraction.all_state_ids()),
                },
            )

        return AbstractVerificationResult(
            verified=False,
            violating_state_ids=violating_states,
            max_abstract_loss=max_loss,
            details={
                "epsilon": eps,
                "violating_count": len(violating_states),
            },
        )

    def verify_per_notion(
        self,
        abstraction: AbstractionState,
        budget: PrivacyBudget,
    ) -> AbstractVerificationResult:
        """Notion-specific abstract verification.

        Uses tighter analysis for specific DP notions (zCDP, RDP)
        that can leverage the additive structure of the density ratio.

        Args:
            abstraction: Current abstraction state.
            budget: Privacy budget.

        Returns:
            AbstractVerificationResult.
        """
        if isinstance(budget, PureBudget):
            return self._verify_pure_dp(abstraction, budget.epsilon)
        elif isinstance(budget, ApproxBudget):
            return self._verify_approx_dp(abstraction, budget.epsilon, budget.delta)
        elif isinstance(budget, ZCDPBudget):
            return self._verify_zcdp(abstraction, budget.rho)
        else:
            return self.verify(abstraction, budget)

    def _verify_pure_dp(
        self,
        abstraction: AbstractionState,
        epsilon: float,
    ) -> AbstractVerificationResult:
        """Pure ε-DP verification: max|L(o)| ≤ ε.

        Args:
            abstraction: Current abstraction.
            epsilon: Privacy parameter.

        Returns:
            AbstractVerificationResult.
        """
        violating: list[str] = []
        max_loss = 0.0

        for sid in abstraction.all_state_ids():
            bound = abstraction.get_abstract_density(sid)
            abs_max = max(abs(bound.lo), abs(bound.hi))
            max_loss = max(max_loss, abs_max)
            if abs_max > epsilon:
                violating.append(sid)

        return AbstractVerificationResult(
            verified=len(violating) == 0,
            violating_state_ids=violating,
            max_abstract_loss=max_loss,
            details={"notion": "pure_dp", "epsilon": epsilon},
        )

    def _verify_approx_dp(
        self,
        abstraction: AbstractionState,
        epsilon: float,
        delta: float,
    ) -> AbstractVerificationResult:
        """Approximate (ε,δ)-DP verification.

        Uses the hockey-stick divergence bound:
          δ_abstract ≤ Σ_s max(0, exp(bound_s.hi) - exp(ε)) · weight(s)
        where weight(s) is proportional to the number of paths.

        Args:
            abstraction: Current abstraction.
            epsilon: Privacy parameter.
            delta: Failure probability.

        Returns:
            AbstractVerificationResult.
        """
        total_paths = sum(
            s.size() for s in abstraction.partition.iter_states()
        )
        if total_paths == 0:
            return AbstractVerificationResult(verified=True)

        delta_bound = 0.0
        violating: list[str] = []

        for sid in abstraction.all_state_ids():
            bound = abstraction.get_abstract_density(sid)
            state = abstraction.partition.get_state(sid)
            weight = state.size() / total_paths if state else 0.0

            if bound.hi > epsilon:
                try:
                    excess = math.exp(bound.hi) - math.exp(epsilon)
                except OverflowError:
                    excess = float("inf")
                contribution = max(0.0, excess) * weight
                delta_bound += contribution
                violating.append(sid)

        verified = delta_bound <= delta
        return AbstractVerificationResult(
            verified=verified,
            violating_state_ids=violating if not verified else [],
            max_abstract_loss=delta_bound,
            details={
                "notion": "approx_dp",
                "epsilon": epsilon,
                "delta": delta,
                "computed_delta": delta_bound,
            },
        )

    def _verify_zcdp(
        self,
        abstraction: AbstractionState,
        rho: float,
    ) -> AbstractVerificationResult:
        """Zero-concentrated DP verification.

        Bounds the Rényi divergence using the abstract density intervals:
          D_α ≤ max_s (bound_s.hi²) / 2

        Args:
            abstraction: Current abstraction.
            rho: zCDP parameter.

        Returns:
            AbstractVerificationResult.
        """
        max_rho = 0.0
        violating: list[str] = []

        for sid in abstraction.all_state_ids():
            bound = abstraction.get_abstract_density(sid)
            abs_max = max(abs(bound.lo), abs(bound.hi))
            state_rho = abs_max ** 2 / 2.0
            max_rho = max(max_rho, state_rho)
            if state_rho > rho:
                violating.append(sid)

        return AbstractVerificationResult(
            verified=len(violating) == 0,
            violating_state_ids=violating,
            max_abstract_loss=max_rho,
            details={"notion": "zcdp", "rho": rho, "computed_rho": max_rho},
        )


@dataclass(slots=True)
class AbstractVerificationResult:
    """Result of abstract verification.

    Attributes:
        verified: True if privacy holds abstractly.
        violating_state_ids: States whose bounds violate the budget.
        max_abstract_loss: Maximum abstract privacy loss across states.
        details: Additional verification information.
    """

    verified: bool = False
    violating_state_ids: list[str] = field(default_factory=list)
    max_abstract_loss: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        status = "VERIFIED" if self.verified else "FAILED"
        return f"AbstractVerification({status}, loss={self.max_abstract_loss:.4f})"


# ═══════════════════════════════════════════════════════════════════════════
# CANDIDATE EXTRACTOR
# ═══════════════════════════════════════════════════════════════════════════


class CandidateExtractor:
    """Extract candidate counterexamples from abstract verification failure.

    When abstract verification fails, this component identifies the
    violating abstract state and constructs a concrete candidate
    counterexample by choosing specific paths and variable assignments.
    """

    def extract(
        self,
        abstraction: AbstractionState,
        verification_result: AbstractVerificationResult,
        path_set: PathSet | None = None,
    ) -> RefinementCounterexample | None:
        """Extract a candidate counterexample from a failed verification.

        Selects the violating state with the worst density bound and
        picks a representative path from it.

        Args:
            abstraction: Current abstraction.
            verification_result: The failed verification result.
            path_set: Optional path set for concrete path selection.

        Returns:
            A candidate counterexample, or None if extraction fails.
        """
        if verification_result.verified:
            return None

        violating = verification_result.violating_state_ids
        if not violating:
            return None

        worst_sid = self._select_worst_state(abstraction, violating)
        state = abstraction.partition.get_state(worst_sid)
        if state is None or state.is_empty():
            return None

        representative_pid = self._select_representative_path(
            state, path_set
        )
        bound = abstraction.get_abstract_density(worst_sid)
        cex_value = bound.hi if abs(bound.hi) >= abs(bound.lo) else bound.lo

        cex = RefinementCounterexample(
            path_id=representative_pid,
            state_id=worst_sid,
            density_ratio_value=cex_value,
            metadata={"source": "abstract_verification"},
        )

        if path_set is not None:
            path = path_set.get(representative_pid)
            if path is not None:
                for var_name, expr in path.assignments.items():
                    if isinstance(expr, Const):
                        cex.variable_assignment[var_name] = float(expr.value)

        return cex

    def _select_worst_state(
        self,
        abstraction: AbstractionState,
        violating_ids: list[str],
    ) -> str:
        """Select the violating state with the largest bound magnitude.

        Args:
            abstraction: Current abstraction.
            violating_ids: IDs of violating states.

        Returns:
            The state ID with the worst bound.
        """
        worst_id = violating_ids[0]
        worst_magnitude = 0.0

        for sid in violating_ids:
            bound = abstraction.get_abstract_density(sid)
            mag = max(abs(bound.lo), abs(bound.hi))
            if mag > worst_magnitude:
                worst_magnitude = mag
                worst_id = sid

        return worst_id

    def _select_representative_path(
        self,
        state: AbstractState,
        path_set: PathSet | None,
    ) -> int:
        """Select a representative path from the abstract state.

        Prefers paths with more noise draws (more likely to produce
        a genuine counterexample).

        Args:
            state: The abstract state to pick from.
            path_set: Optional path set for informed selection.

        Returns:
            A path ID from the state.
        """
        if path_set is None or state.size() == 1:
            return next(iter(state.path_ids))

        best_pid = next(iter(state.path_ids))
        best_noise = -1

        for pid in state.path_ids:
            path = path_set.get(pid)
            if path is not None:
                n = len(path.noise_draws)
                if n > best_noise:
                    best_noise = n
                    best_pid = pid

        return best_pid


# ═══════════════════════════════════════════════════════════════════════════
# CONCRETE CHECKER
# ═══════════════════════════════════════════════════════════════════════════


class ConcreteChecker:
    """Check candidate counterexamples against concrete semantics.

    Given a candidate counterexample from the abstract domain, this
    component formulates the concrete privacy check as an SMT query
    and determines whether the counterexample is genuine or spurious.
    """

    def __init__(self, solver: SMTSolverInterface | None = None) -> None:
        """Initialise with an optional SMT solver.

        Args:
            solver: SMT solver interface.  If None, uses syntactic checking.
        """
        self._solver = solver

    def check(
        self,
        counterexample: RefinementCounterexample,
        path_set: PathSet,
        density_ratios: list[DensityRatioExpr] | None = None,
        budget: PrivacyBudget | None = None,
        timeout_ms: int = 30000,
    ) -> ConcreteCheckResult:
        """Check if a candidate counterexample is genuine.

        Constructs the concrete constraints for the counterexample's
        path and checks their satisfiability.

        Args:
            counterexample: The candidate to check.
            path_set: The set of symbolic paths.
            density_ratios: Optional density ratio expressions.
            budget: The privacy budget being verified.
            timeout_ms: Solver timeout.

        Returns:
            A ConcreteCheckResult.
        """
        path = path_set.get(counterexample.path_id)
        if path is None:
            return ConcreteCheckResult(
                is_genuine=False,
                is_spurious=True,
                reason="path_not_found",
            )

        if path.path_condition.is_trivially_false():
            return ConcreteCheckResult(
                is_genuine=False,
                is_spurious=True,
                reason="trivially_infeasible",
                proof=InfeasibilityProof(
                    involved_paths=[counterexample.path_id]
                ),
            )

        if not path.is_feasible():
            return ConcreteCheckResult(
                is_genuine=False,
                is_spurious=True,
                reason="syntactically_infeasible",
                proof=InfeasibilityProof(
                    involved_paths=[counterexample.path_id]
                ),
            )

        if self._solver is not None:
            return self._smt_check(
                counterexample, path, density_ratios, budget, timeout_ms
            )

        return self._syntactic_check(counterexample, path, budget)

    def _smt_check(
        self,
        counterexample: RefinementCounterexample,
        path: SymbolicPath,
        density_ratios: list[DensityRatioExpr] | None,
        budget: PrivacyBudget | None,
        timeout_ms: int,
    ) -> ConcreteCheckResult:
        """Check using the SMT solver.

        Encodes the path condition and privacy violation as SMT
        constraints and queries the solver.

        Args:
            counterexample: Candidate counterexample.
            path: The symbolic path.
            density_ratios: Density ratio expressions.
            budget: Privacy budget.
            timeout_ms: Solver timeout.

        Returns:
            ConcreteCheckResult.
        """
        assert self._solver is not None

        constraints: list[TypedExpr] = list(path.path_condition.conjuncts)

        if budget is not None and density_ratios:
            eps, _ = budget.to_approx_dp()
            for ratio in density_ratios:
                if ratio.path_id_d == path.path_id:
                    violation = BinOp(
                        ty=IRType.BOOL,
                        op=BinOpKind.GT,
                        left=ratio.log_ratio,
                        right=Const.real(eps),
                    )
                    constraints.append(violation)

        result = self._solver.check_sat(constraints, timeout_ms)

        if result.is_sat:
            counterexample.variable_assignment = result.model
            counterexample.is_spurious = False
            return ConcreteCheckResult(
                is_genuine=True,
                is_spurious=False,
                reason="sat",
                model=result.model,
            )
        elif result.is_unsat:
            core = self._solver.get_unsat_core()
            interpolant = None
            if len(constraints) >= 2:
                mid = len(constraints) // 2
                interpolant = self._solver.compute_interpolant(
                    constraints[:mid], constraints[mid:]
                )

            proof = InfeasibilityProof(
                unsat_core=core,
                interpolants=[interpolant] if interpolant else [],
                involved_paths=[counterexample.path_id],
                proof_time=result.time_seconds,
            )
            counterexample.is_spurious = True
            return ConcreteCheckResult(
                is_genuine=False,
                is_spurious=True,
                reason="unsat",
                proof=proof,
            )
        else:
            return ConcreteCheckResult(
                is_genuine=None,
                is_spurious=None,
                reason="unknown",
            )

    def _syntactic_check(
        self,
        counterexample: RefinementCounterexample,
        path: SymbolicPath,
        budget: PrivacyBudget | None,
    ) -> ConcreteCheckResult:
        """Syntactic feasibility check without SMT solver.

        Conservative: can prove spuriousness but cannot confirm
        genuineness.

        Args:
            counterexample: Candidate counterexample.
            path: The symbolic path.
            budget: Privacy budget.

        Returns:
            ConcreteCheckResult.
        """
        if path.path_condition.is_trivially_false():
            counterexample.is_spurious = True
            return ConcreteCheckResult(
                is_genuine=False,
                is_spurious=True,
                reason="trivially_false",
            )

        return ConcreteCheckResult(
            is_genuine=None,
            is_spurious=None,
            reason="no_solver_available",
        )


@dataclass(slots=True)
class ConcreteCheckResult:
    """Result of concrete counterexample checking.

    Attributes:
        is_genuine: True if the cex is confirmed genuine. None if unknown.
        is_spurious: True if the cex is confirmed spurious. None if unknown.
        reason: Explanation of the outcome.
        model: Concrete variable assignment (if genuine).
        proof: Infeasibility proof (if spurious).
    """

    is_genuine: bool | None = None
    is_spurious: bool | None = None
    reason: str = ""
    model: dict[str, float] = field(default_factory=dict)
    proof: InfeasibilityProof | None = None

    def __str__(self) -> str:
        if self.is_genuine:
            return f"Genuine({self.reason})"
        if self.is_spurious:
            return f"Spurious({self.reason})"
        return f"Unknown({self.reason})"


# ═══════════════════════════════════════════════════════════════════════════
# SPURIOUSNESS ANALYSER
# ═══════════════════════════════════════════════════════════════════════════


class SpuriousnessAnalyzer:
    """Analyse why a counterexample is spurious.

    Given a spurious counterexample and its infeasibility proof,
    determines which abstraction aspect caused the spuriousness and
    recommends a refinement strategy.
    """

    def analyze(
        self,
        counterexample: RefinementCounterexample,
        proof: InfeasibilityProof | None,
        abstraction: AbstractionState,
        path_set: PathSet | None = None,
    ) -> SpuriousnessAnalysis:
        """Analyse the cause of spuriousness.

        Args:
            counterexample: The spurious counterexample.
            proof: Infeasibility proof from the SMT solver.
            abstraction: Current abstraction state.
            path_set: Optional path set.

        Returns:
            Analysis with recommended refinement strategy.
        """
        causes: list[SpuriousnessCause] = []
        recommended_kind = RefinementKind.PATH_SPLIT

        state = abstraction.partition.get_state(counterexample.state_id)
        if state is not None and state.size() > 1:
            causes.append(SpuriousnessCause.MERGED_PATHS)

        bound = abstraction.get_abstract_density(counterexample.state_id)
        if bound.is_unbounded() or bound.width > 10.0:
            causes.append(SpuriousnessCause.LOOSE_BOUNDS)
            recommended_kind = RefinementKind.INTERVAL_NARROW

        if proof is not None and proof.has_interpolants():
            causes.append(SpuriousnessCause.MISSING_PREDICATE)
            recommended_kind = RefinementKind.PREDICATE_ADD

        if proof is not None and proof.has_unsat_core():
            core_vars: set[str] = set()
            for expr in proof.unsat_core:
                core_vars.update(expr.free_vars())

            if path_set is not None:
                path = path_set.get(counterexample.path_id)
                if path is not None:
                    noise_vars = path.get_noise_vars()
                    if core_vars & noise_vars:
                        causes.append(SpuriousnessCause.NOISE_STRUCTURE)

        if not causes:
            causes.append(SpuriousnessCause.UNKNOWN)

        return SpuriousnessAnalysis(
            causes=causes,
            recommended_refinement=recommended_kind,
            proof=proof,
            details={
                "state_size": state.size() if state else 0,
                "bound_width": bound.width,
            },
        )


class SpuriousnessCause(Enum):
    """Possible causes of spuriousness."""

    MERGED_PATHS = auto()
    LOOSE_BOUNDS = auto()
    MISSING_PREDICATE = auto()
    NOISE_STRUCTURE = auto()
    LOOP_APPROXIMATION = auto()
    UNKNOWN = auto()


@dataclass(slots=True)
class SpuriousnessAnalysis:
    """Analysis of why a counterexample is spurious.

    Attributes:
        causes: Identified causes of spuriousness.
        recommended_refinement: Recommended refinement strategy.
        proof: The infeasibility proof.
        details: Additional analysis details.
    """

    causes: list[SpuriousnessCause] = field(default_factory=list)
    recommended_refinement: RefinementKind = RefinementKind.PATH_SPLIT
    proof: InfeasibilityProof | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        cause_names = [c.name for c in self.causes]
        return f"SpuriousnessAnalysis({cause_names}, recommend={self.recommended_refinement.name})"


# ═══════════════════════════════════════════════════════════════════════════
# CONVERGENCE TRACKER
# ═══════════════════════════════════════════════════════════════════════════


class ConvergenceTracker:
    """Track progress and estimate remaining work in the CEGAR loop.

    Provides progress percentage, estimated time to completion, and
    convergence trend analysis.
    """

    def __init__(self) -> None:
        """Initialise the convergence tracker."""
        self._bound_history: list[float] = []
        self._time_history: list[float] = []
        self._state_count_history: list[int] = []
        self._start_time: float = time.monotonic()

    def record(
        self,
        iteration: int,
        max_loss: float,
        state_count: int,
    ) -> None:
        """Record metrics for the current iteration.

        Args:
            iteration: Current iteration number.
            max_loss: Current maximum abstract privacy loss.
            state_count: Current number of abstract states.
        """
        self._bound_history.append(max_loss)
        self._time_history.append(time.monotonic() - self._start_time)
        self._state_count_history.append(state_count)

    def estimated_progress(self, budget_epsilon: float) -> float:
        """Estimate progress as a fraction [0, 1].

        Based on how close the current bounds are to the budget.

        Args:
            budget_epsilon: The target privacy parameter.

        Returns:
            Estimated progress in [0, 1].
        """
        if not self._bound_history:
            return 0.0
        current = self._bound_history[-1]
        if current <= budget_epsilon:
            return 1.0
        initial = self._bound_history[0] if self._bound_history else current
        if initial <= budget_epsilon:
            return 1.0
        reduction = initial - current
        needed = initial - budget_epsilon
        if needed <= 0:
            return 1.0
        return min(1.0, max(0.0, reduction / needed))

    def estimated_remaining_time(self) -> float | None:
        """Estimate remaining time based on convergence trend.

        Returns:
            Estimated seconds remaining, or None if insufficient data.
        """
        if len(self._bound_history) < 3:
            return None

        recent_bounds = self._bound_history[-3:]
        recent_times = self._time_history[-3:]

        rates = []
        for i in range(1, len(recent_bounds)):
            dt = recent_times[i] - recent_times[i - 1]
            db = recent_bounds[i - 1] - recent_bounds[i]
            if dt > 0 and db > 0:
                rates.append(db / dt)

        if not rates:
            return None

        avg_rate = sum(rates) / len(rates)
        remaining = self._bound_history[-1]
        if avg_rate <= 0:
            return None
        return remaining / avg_rate

    def is_progressing(self, window: int = 5) -> bool:
        """Check if the loop is making progress.

        Args:
            window: Number of recent iterations to check.

        Returns:
            True if the maximum loss is decreasing.
        """
        if len(self._bound_history) < window:
            return True
        recent = self._bound_history[-window:]
        return recent[-1] < recent[0]

    def trend(self) -> str:
        """Describe the convergence trend.

        Returns:
            One of "converging", "stalling", "diverging".
        """
        if len(self._bound_history) < 3:
            return "insufficient_data"
        recent = self._bound_history[-3:]
        if recent[-1] < recent[0]:
            return "converging"
        elif recent[-1] > recent[0]:
            return "diverging"
        else:
            return "stalling"


# ═══════════════════════════════════════════════════════════════════════════
# CEGAR ENGINE — Main CEGAR loop
# ═══════════════════════════════════════════════════════════════════════════


class CEGAREngine:
    """Main CEGAR loop driver for differential privacy verification.

    Orchestrates the abstract-verify → extract → concretize → refine
    loop until convergence or resource exhaustion.

    Usage::

        engine = CEGAREngine(config=CEGARConfig())
        result = engine.verify(path_set, budget)
        print(result.summary())
    """

    def __init__(
        self,
        config: CEGARConfig | None = None,
        solver: SMTSolverInterface | None = None,
        refinement_operators: list[RefinementOperator] | None = None,
    ) -> None:
        """Initialise the CEGAR engine.

        Args:
            config: CEGAR configuration.
            solver: Optional SMT solver interface.
            refinement_operators: Optional custom refinement operators.
        """
        self._config = config or CEGARConfig()
        self._config.validate()
        self._solver = solver

        self._abstract_verifier = AbstractVerifier()
        self._candidate_extractor = CandidateExtractor()
        self._concrete_checker = ConcreteChecker(solver=solver)
        self._spuriousness_analyzer = SpuriousnessAnalyzer()
        self._refinement_selector = RefinementSelector(refinement_operators)
        self._convergence_detector = ConvergenceDetector(
            max_refinements=self._config.max_refinements,
            max_time_seconds=self._config.timeout_seconds,
            fixpoint_tolerance=self._config.fixpoint_tolerance,
        )
        self._widening = WideningOperator(
            threshold=self._config.widening_threshold,
            patience=self._config.widening_patience,
        ) if self._config.enable_widening else None

        self._history = RefinementHistory()
        self._tracker = ConvergenceTracker()
        self._stats = CEGARStatistics()
        self._concrete_cache: dict[tuple, ConcreteCheckResult] = {}

    def verify(
        self,
        path_set: PathSet,
        budget: PrivacyBudget,
        density_ratios: list[DensityRatioExpr] | None = None,
    ) -> CEGARResult:
        """Run the CEGAR loop to verify or refute a privacy budget.

        This is the main entry point.  It creates an initial abstraction,
        then iteratively verifies/refines until convergence.

        Args:
            path_set: Enumerated symbolic paths from the mechanism.
            budget: The privacy budget to verify.
            density_ratios: Optional pre-computed density ratio expressions.

        Returns:
            A CEGARResult with the verification outcome.
        """
        start_time = time.monotonic()
        self._convergence_detector.start()
        self._history = RefinementHistory()
        self._tracker = ConvergenceTracker()
        self._stats = CEGARStatistics()
        self._concrete_cache.clear()

        abstraction = self._create_initial_abstraction(path_set)

        try:
            result = self._cegar_loop(
                abstraction, budget, path_set, density_ratios
            )
        except DPCegarError as e:
            result = CEGARResult(
                verdict=CEGARVerdict.ERROR,
                budget=budget,
                details={"error": str(e)},
            )

        self._stats.total_time = time.monotonic() - start_time
        self._stats.final_abstract_states = abstraction.partition.state_count()
        result.statistics = self._stats
        result.final_abstraction = abstraction
        return result

    def verify_mechanism(
        self,
        mechanism: MechIR,
        budget: PrivacyBudget,
        path_set: PathSet | None = None,
        density_ratios: list[DensityRatioExpr] | None = None,
    ) -> CEGARResult:
        """Verify a mechanism IR directly.

        Convenience method that accepts a MechIR instead of a path set.
        If no path set is provided, uses an empty one (caller should
        have enumerated paths beforehand).

        Args:
            mechanism: The mechanism IR.
            budget: Privacy budget.
            path_set: Optional pre-enumerated paths.
            density_ratios: Optional density ratios.

        Returns:
            CEGARResult.
        """
        if path_set is None:
            path_set = PathSet()
        return self.verify(path_set, budget, density_ratios)

    def _create_initial_abstraction(self, path_set: PathSet) -> AbstractionState:
        """Create the initial abstraction from configuration.

        Args:
            path_set: The paths to abstract.

        Returns:
            The initial abstraction state.
        """
        strategy = self._config.initial_abstraction

        if strategy == "coarsest":
            return InitialAbstraction.coarsest(path_set)
        elif strategy == "noise_pattern":
            return InitialAbstraction.by_noise_pattern(path_set)
        elif strategy == "branch_structure":
            return InitialAbstraction.by_branch_structure(path_set)
        elif strategy == "finest":
            return InitialAbstraction.finest(path_set)
        else:
            return InitialAbstraction.by_noise_pattern(path_set)

    def _cegar_loop(
        self,
        abstraction: AbstractionState,
        budget: PrivacyBudget,
        path_set: PathSet,
        density_ratios: list[DensityRatioExpr] | None,
    ) -> CEGARResult:
        """Execute the core CEGAR loop.

        Args:
            abstraction: Initial abstraction.
            budget: Privacy budget.
            path_set: Symbolic paths.
            density_ratios: Optional density ratios.

        Returns:
            CEGARResult.
        """
        iteration = 0

        while True:
            iteration += 1
            self._stats.iterations = iteration
            self._stats.update_peak_states(abstraction.partition.state_count())

            if self._config.progress_callback:
                self._config.progress_callback(
                    "cegar_iteration", iteration, self._config.max_refinements
                )

            conv_status = self._convergence_detector.check(
                abstraction, self._history, budget
            )
            if conv_status.converged:
                return self._handle_convergence(
                    conv_status, abstraction, budget
                )

            t0 = time.monotonic()
            abs_result = self._abstract_verifier.verify_per_notion(
                abstraction, budget
            )
            self._stats.abstract_verify_time += time.monotonic() - t0
            self._stats.abstract_verifications += 1

            self._tracker.record(
                iteration,
                abs_result.max_abstract_loss,
                abstraction.partition.state_count(),
            )

            if abs_result.verified:
                certificate = self._build_certificate(abstraction, budget)
                return CEGARResult(
                    verdict=CEGARVerdict.VERIFIED,
                    budget=budget,
                    certificate=certificate,
                    final_bounds=abstraction.overall_density_bound(),
                )

            candidate = self._candidate_extractor.extract(
                abstraction, abs_result, path_set
            )
            if candidate is None:
                return CEGARResult(
                    verdict=CEGARVerdict.UNKNOWN,
                    budget=budget,
                    final_bounds=abstraction.overall_density_bound(),
                    details={"reason": "no_candidate_extracted"},
                )

            t1 = time.monotonic()
            cache_key = (candidate.path_id, candidate.state_id)
            cached_concrete = self._concrete_cache.get(cache_key)
            if cached_concrete is not None:
                concrete_result = cached_concrete
            else:
                concrete_result = self._concrete_checker.check(
                    candidate, path_set, density_ratios, budget,
                    timeout_ms=int(self._config.solver_timeout_seconds * 1000),
                )
                if concrete_result.is_genuine or concrete_result.is_spurious:
                    self._concrete_cache[cache_key] = concrete_result
            self._stats.concrete_check_time += time.monotonic() - t1
            self._stats.concrete_checks += 1
            self._stats.solver_calls += 1

            if concrete_result.is_genuine:
                self._stats.genuine_cex_count += 1
                candidate.is_spurious = False
                return CEGARResult(
                    verdict=CEGARVerdict.COUNTEREXAMPLE,
                    budget=budget,
                    counterexample=candidate,
                    final_bounds=abstraction.overall_density_bound(),
                )

            if concrete_result.is_spurious:
                self._stats.spurious_cex_count += 1
                candidate.is_spurious = True
                candidate.spuriousness_reason = concrete_result.reason

                analysis = self._spuriousness_analyzer.analyze(
                    candidate, concrete_result.proof, abstraction, path_set
                )

                t2 = time.monotonic()
                ref_result = self._refinement_selector.select_and_apply(
                    abstraction, candidate, concrete_result.proof
                )
                self._stats.refinement_time += time.monotonic() - t2

                if ref_result.success:
                    self._stats.refinements += 1
                    if ref_result.record:
                        self._history.add(ref_result.record)

                    # Invalidate cache for the refined state
                    self._concrete_cache.pop(cache_key, None)

                    if self._widening:
                        self._widening.record_iteration(dict(abstraction.density_bounds))
                        if self._widening.should_widen():
                            self._widening.apply(abstraction)

                    is_cycle = self._history.record_state_fingerprint(abstraction)
                    if is_cycle:
                        return CEGARResult(
                            verdict=CEGARVerdict.UNKNOWN,
                            budget=budget,
                            final_bounds=abstraction.overall_density_bound(),
                            details={"reason": "cycle_detected"},
                        )
                else:
                    return CEGARResult(
                        verdict=CEGARVerdict.UNKNOWN,
                        budget=budget,
                        final_bounds=abstraction.overall_density_bound(),
                        details={"reason": "refinement_failed"},
                    )
            else:
                return CEGARResult(
                    verdict=CEGARVerdict.UNKNOWN,
                    budget=budget,
                    final_bounds=abstraction.overall_density_bound(),
                    details={"reason": "concretization_inconclusive"},
                )

    def _handle_convergence(
        self,
        status: ConvergenceStatus,
        abstraction: AbstractionState,
        budget: PrivacyBudget,
    ) -> CEGARResult:
        """Handle convergence of the CEGAR loop.

        Args:
            status: The convergence status.
            abstraction: Final abstraction.
            budget: Privacy budget.

        Returns:
            CEGARResult based on convergence reason.
        """
        overall = abstraction.overall_density_bound()
        eps, _ = budget.to_approx_dp()

        if status.reason == ConvergenceReason.BUDGET_SATISFIED:
            cert = self._build_certificate(abstraction, budget)
            return CEGARResult(
                verdict=CEGARVerdict.VERIFIED,
                budget=budget,
                certificate=cert,
                final_bounds=overall,
            )

        if status.reason == ConvergenceReason.FINEST_REACHED:
            if overall.satisfies_epsilon(eps):
                cert = self._build_certificate(abstraction, budget)
                return CEGARResult(
                    verdict=CEGARVerdict.VERIFIED,
                    budget=budget,
                    certificate=cert,
                    final_bounds=overall,
                )

        if status.reason == ConvergenceReason.TIMEOUT:
            verdict = CEGARVerdict.TIMEOUT
        else:
            verdict = CEGARVerdict.UNKNOWN

        return CEGARResult(
            verdict=verdict,
            budget=budget,
            final_bounds=overall,
            details={
                "convergence_reason": status.reason.name,
                "convergence_details": status.details,
            },
        )

    def _build_certificate(
        self,
        abstraction: AbstractionState,
        budget: PrivacyBudget,
    ) -> VerificationCertificate:
        """Build a verification certificate from the final abstraction.

        Args:
            abstraction: The final abstraction state.
            budget: The verified budget.

        Returns:
            A VerificationCertificate.
        """
        steps: list[str] = [
            f"Initial abstraction: {self._config.initial_abstraction}",
            f"Refinement iterations: {self._stats.refinements}",
            f"Final states: {abstraction.partition.state_count()}",
        ]

        for record in abstraction.history:
            steps.append(
                f"  {record.kind.name}: {record.source_state_id} → "
                f"{record.result_state_ids}"
            )

        return VerificationCertificate(
            budget=budget,
            density_bounds=dict(abstraction.density_bounds),
            abstraction_state=abstraction,
            proof_steps=steps,
        )

    def get_statistics(self) -> CEGARStatistics:
        """Return the current statistics.

        Returns:
            The CEGARStatistics object.
        """
        return self._stats

    def get_history(self) -> RefinementHistory:
        """Return the refinement history.

        Returns:
            The RefinementHistory object.
        """
        return self._history
