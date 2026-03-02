"""Refinement operators for CEGAR-based differential privacy verification.

This module provides the refinement operators that drive the CEGAR loop
toward convergence.  When abstract verification fails to prove privacy
and the extracted counterexample is spurious, a refinement operator
modifies the abstraction to rule out that spurious counterexample.

Classes
-------
RefinementOperator       – abstract base for all refinement operators
PathSplitRefinement      – split an abstract state by path condition
IntervalNarrowRefinement – tighten density interval from spurious cex
PredicateRefinement      – add a distinguishing predicate from an
                           infeasibility proof
LoopUnwindRefinement     – increase loop unrolling depth
RefinementSelector       – choose the best refinement operator
RefinementHistory        – track all refinements and detect cycles
ConvergenceDetector      – detect when CEGAR has converged
"""

from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
)

from dpcegar.ir.types import (
    BinOp,
    BinOpKind,
    Const,
    IRType,
    NoiseKind,
    PrivacyBudget,
    TypedExpr,
    UnaryOp,
    UnaryOpKind,
    Var,
)
from dpcegar.paths.symbolic_path import (
    NoiseDrawInfo,
    PathCondition,
    PathSet,
    SymbolicPath,
)
from dpcegar.density.ratio_builder import DensityRatioExpr
from dpcegar.cegar.abstraction import (
    AbstractDensityBound,
    AbstractionState,
    AbstractState,
    PathPartition,
    RefinementKind,
    RefinementRecord,
    WideningOperator,
)
from dpcegar.utils.errors import InternalError, ensure


# ═══════════════════════════════════════════════════════════════════════════
# COUNTEREXAMPLE REPRESENTATION
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class RefinementCounterexample:
    """A candidate counterexample to the privacy property.

    Represents a concrete assignment to program variables that
    witnesses a privacy violation (or a spurious one in the abstract).

    Attributes:
        variable_assignment: Mapping from variable name to concrete value.
        path_id: The path ID along which the violation occurs.
        state_id: The abstract state containing the violating path.
        density_ratio_value: The concrete privacy loss at this point.
        is_spurious: Whether this cex has been determined to be spurious.
        spuriousness_reason: Explanation if spurious.
        metadata: Additional context.
    """

    variable_assignment: dict[str, float] = field(default_factory=dict)
    path_id: int = -1
    state_id: str = ""
    density_ratio_value: float = 0.0
    is_spurious: bool | None = None
    spuriousness_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_value(self, var_name: str) -> float | None:
        """Return the concrete value assigned to *var_name*, or None."""
        return self.variable_assignment.get(var_name)

    def involves_path(self, path_id: int) -> bool:
        """Return True if this counterexample involves the given path."""
        return self.path_id == path_id

    def summary(self) -> str:
        """Return a human-readable summary."""
        status = "spurious" if self.is_spurious else "genuine" if self.is_spurious is False else "unknown"
        return (
            f"Counterexample(path={self.path_id}, "
            f"loss={self.density_ratio_value:.4f}, {status})"
        )

    def __str__(self) -> str:
        return self.summary()


# Backward compatibility alias (deprecated)
Counterexample = RefinementCounterexample


# ═══════════════════════════════════════════════════════════════════════════
# REFINEMENT RESULT
# ═══════════════════════════════════════════════════════════════════════════


class RefinementStatus(Enum):
    """Outcome of a refinement attempt."""

    SUCCESS = auto()
    NO_PROGRESS = auto()
    AT_FINEST = auto()
    FAILED = auto()


@dataclass(slots=True)
class RefinementResult:
    """Result of applying a refinement operator.

    Attributes:
        status: Whether refinement succeeded.
        refined_state: The refined abstraction state (if successful).
        record: The refinement record for the history.
        new_state_ids: IDs of newly created abstract states.
        eliminated_cex: Whether the triggering counterexample was eliminated.
        details: Additional information.
    """

    status: RefinementStatus = RefinementStatus.FAILED
    refined_state: AbstractionState | None = None
    record: RefinementRecord | None = None
    new_state_ids: list[str] = field(default_factory=list)
    eliminated_cex: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Return True if the refinement made progress."""
        return self.status == RefinementStatus.SUCCESS

    def __str__(self) -> str:
        return f"RefinementResult({self.status.name})"


# ═══════════════════════════════════════════════════════════════════════════
# INFEASIBILITY PROOF (interface for SMT layer)
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class InfeasibilityProof:
    """Proof that a counterexample is infeasible (spurious).

    This is the interface object produced by the SMT layer when it
    determines that a candidate counterexample cannot occur in the
    concrete semantics.

    Attributes:
        unsat_core: Subset of constraints that are mutually unsatisfiable.
        interpolants: Craig interpolants extracted from the proof.
        involved_paths: Path IDs whose conditions contributed to unsat.
        proof_time: Time in seconds to produce the proof.
    """

    unsat_core: list[TypedExpr] = field(default_factory=list)
    interpolants: list[TypedExpr] = field(default_factory=list)
    involved_paths: list[int] = field(default_factory=list)
    proof_time: float = 0.0

    def has_interpolants(self) -> bool:
        """Return True if interpolants are available."""
        return len(self.interpolants) > 0

    def has_unsat_core(self) -> bool:
        """Return True if an unsat core is available."""
        return len(self.unsat_core) > 0


# ═══════════════════════════════════════════════════════════════════════════
# REFINEMENT OPERATOR BASE CLASS
# ═══════════════════════════════════════════════════════════════════════════


class RefinementOperator(ABC):
    """Abstract base class for refinement operators.

    Each operator implements a specific strategy for refining the
    abstraction to eliminate a spurious counterexample.
    """

    @abstractmethod
    def name(self) -> str:
        """Return a human-readable name for this operator."""
        ...

    @abstractmethod
    def is_applicable(
        self,
        state: AbstractionState,
        counterexample: RefinementCounterexample,
        proof: InfeasibilityProof | None = None,
    ) -> bool:
        """Check whether this operator can refine the given state.

        Args:
            state: Current abstraction state.
            counterexample: The spurious counterexample to eliminate.
            proof: Optional infeasibility proof from the SMT layer.

        Returns:
            True if this operator can make progress.
        """
        ...

    @abstractmethod
    def apply(
        self,
        state: AbstractionState,
        counterexample: RefinementCounterexample,
        proof: InfeasibilityProof | None = None,
    ) -> RefinementResult:
        """Apply this refinement operator to the abstraction.

        Args:
            state: Current abstraction state (modified in place).
            counterexample: The spurious counterexample to eliminate.
            proof: Optional infeasibility proof.

        Returns:
            A RefinementResult describing the outcome.
        """
        ...

    @abstractmethod
    def estimated_cost(
        self,
        state: AbstractionState,
        counterexample: RefinementCounterexample,
    ) -> float:
        """Estimate the cost of applying this operator.

        Lower cost means the operator is preferred.  Cost accounts
        for the expected increase in abstract states (more states =
        more SMT queries) and solver difficulty.

        Args:
            state: Current abstraction state.
            counterexample: The counterexample to eliminate.

        Returns:
            Estimated cost (lower is better).
        """
        ...


# ═══════════════════════════════════════════════════════════════════════════
# PATH SPLIT REFINEMENT
# ═══════════════════════════════════════════════════════════════════════════


class PathSplitRefinement(RefinementOperator):
    """Split an abstract state by separating the counterexample path.

    The simplest refinement: move the path that produced the spurious
    counterexample into its own abstract state.  This is always
    applicable when the state contains more than one path.
    """

    def name(self) -> str:
        """Return the operator name."""
        return "PathSplit"

    def is_applicable(
        self,
        state: AbstractionState,
        counterexample: RefinementCounterexample,
        proof: InfeasibilityProof | None = None,
    ) -> bool:
        """Check if the counterexample's state has multiple paths.

        Args:
            state: Current abstraction.
            counterexample: Spurious counterexample.
            proof: Unused.

        Returns:
            True if the containing state has > 1 path.
        """
        abs_state = state.partition.get_state(counterexample.state_id)
        if abs_state is None:
            abs_state = state.partition.get_state_for_path(counterexample.path_id)
        return abs_state is not None and abs_state.size() > 1

    def apply(
        self,
        state: AbstractionState,
        counterexample: RefinementCounterexample,
        proof: InfeasibilityProof | None = None,
    ) -> RefinementResult:
        """Split the counterexample's path into its own state.

        Args:
            state: Current abstraction (modified in place).
            counterexample: Spurious counterexample.
            proof: Unused.

        Returns:
            RefinementResult with the split outcome.
        """
        target_state = state.partition.get_state(counterexample.state_id)
        if target_state is None:
            target_state = state.partition.get_state_for_path(counterexample.path_id)
        if target_state is None or target_state.size() <= 1:
            return RefinementResult(
                status=RefinementStatus.NO_PROGRESS,
                details={"reason": "state not found or singleton"},
            )

        cex_path_id = counterexample.path_id
        t_id, f_id = state.split_state(
            target_state.state_id,
            predicate=lambda pid: pid == cex_path_id,
            kind=RefinementKind.PATH_SPLIT,
        )

        return RefinementResult(
            status=RefinementStatus.SUCCESS,
            refined_state=state,
            record=state.history[-1] if state.history else None,
            new_state_ids=[t_id, f_id],
            eliminated_cex=True,
        )

    def estimated_cost(
        self,
        state: AbstractionState,
        counterexample: RefinementCounterexample,
    ) -> float:
        """Path splits are cheap: cost = 1.0.

        Args:
            state: Current abstraction.
            counterexample: The counterexample.

        Returns:
            Cost value of 1.0.
        """
        return 1.0


# ═══════════════════════════════════════════════════════════════════════════
# INTERVAL NARROW REFINEMENT
# ═══════════════════════════════════════════════════════════════════════════


class IntervalNarrowRefinement(RefinementOperator):
    """Tighten density interval bounds from a spurious counterexample.

    When the SMT solver shows that the counterexample's density ratio
    value is infeasible, this operator narrows the abstract density
    bound to exclude that value.
    """

    def __init__(self, shrink_factor: float = 0.5) -> None:
        """Initialise with a shrink factor.

        Args:
            shrink_factor: How much to shrink the interval (0.5 = halve).
        """
        self._shrink_factor = shrink_factor

    def name(self) -> str:
        """Return the operator name."""
        return "IntervalNarrow"

    def is_applicable(
        self,
        state: AbstractionState,
        counterexample: RefinementCounterexample,
        proof: InfeasibilityProof | None = None,
    ) -> bool:
        """Check if the density bound can be tightened.

        Applicable when the counterexample's density ratio value is
        outside the range that can be achieved by concrete paths.

        Args:
            state: Current abstraction.
            counterexample: Spurious counterexample.
            proof: Optional proof.

        Returns:
            True if the interval is not already exact.
        """
        bound = state.get_abstract_density(counterexample.state_id)
        return not bound.is_exact and bound.width > 1e-12

    def apply(
        self,
        state: AbstractionState,
        counterexample: RefinementCounterexample,
        proof: InfeasibilityProof | None = None,
    ) -> RefinementResult:
        """Narrow the density bound for the counterexample's state.

        If the counterexample's density value is at the boundary, we
        shrink the interval from that side.

        Args:
            state: Current abstraction (modified in place).
            counterexample: Spurious counterexample.
            proof: Optional infeasibility proof.

        Returns:
            RefinementResult.
        """
        sid = counterexample.state_id
        old_bound = state.get_abstract_density(sid)
        cex_val = counterexample.density_ratio_value

        if old_bound.is_exact:
            return RefinementResult(
                status=RefinementStatus.NO_PROGRESS,
                details={"reason": "bound already exact"},
            )

        mid = old_bound.midpoint
        if cex_val >= mid:
            new_hi = old_bound.lo + (old_bound.hi - old_bound.lo) * self._shrink_factor
            new_bound = AbstractDensityBound.from_interval(
                old_bound.lo, min(new_hi, old_bound.hi), source="narrowed"
            )
        else:
            new_lo = old_bound.hi - (old_bound.hi - old_bound.lo) * self._shrink_factor
            new_bound = AbstractDensityBound.from_interval(
                max(new_lo, old_bound.lo), old_bound.hi, source="narrowed"
            )

        if proof and proof.has_unsat_core():
            core_bounds = self._extract_bounds_from_core(proof.unsat_core, old_bound)
            if core_bounds is not None:
                new_bound = core_bounds

        state.set_density_bound(sid, new_bound)
        state.refinement_level += 1

        record = RefinementRecord(
            kind=RefinementKind.INTERVAL_NARROW,
            source_state_id=sid,
            result_state_ids=[sid],
            details={
                "old_bound": str(old_bound),
                "new_bound": str(new_bound),
                "cex_value": cex_val,
            },
            iteration=state.refinement_level,
        )
        state.record_refinement(record)

        return RefinementResult(
            status=RefinementStatus.SUCCESS,
            refined_state=state,
            record=record,
            new_state_ids=[sid],
            eliminated_cex=True,
        )

    def _extract_bounds_from_core(
        self,
        unsat_core: list[TypedExpr],
        current: AbstractDensityBound,
    ) -> AbstractDensityBound | None:
        """Extract tighter bounds from an unsat core.

        Scans the unsat core for comparison constraints and infers
        tighter interval endpoints.

        Args:
            unsat_core: List of unsatisfiable core constraints.
            current: The current density bound.

        Returns:
            Tighter bound if extractable, else None.
        """
        new_lo = current.lo
        new_hi = current.hi

        for expr in unsat_core:
            if isinstance(expr, BinOp):
                if expr.op == BinOpKind.LE and isinstance(expr.right, Const):
                    val = float(expr.right.value)
                    if val < new_hi:
                        new_hi = val
                elif expr.op == BinOpKind.GE and isinstance(expr.right, Const):
                    val = float(expr.right.value)
                    if val > new_lo:
                        new_lo = val

        if new_lo > current.lo or new_hi < current.hi:
            if new_lo <= new_hi:
                return AbstractDensityBound.from_interval(
                    new_lo, new_hi, source="unsat-core"
                )
        return None

    def estimated_cost(
        self,
        state: AbstractionState,
        counterexample: RefinementCounterexample,
    ) -> float:
        """Interval narrowing is moderately expensive: cost = 2.0.

        Args:
            state: Current abstraction.
            counterexample: The counterexample.

        Returns:
            Cost value of 2.0.
        """
        return 2.0


# ═══════════════════════════════════════════════════════════════════════════
# PREDICATE REFINEMENT
# ═══════════════════════════════════════════════════════════════════════════


class PredicateRefinement(RefinementOperator):
    """Add a distinguishing predicate from an infeasibility proof.

    When the SMT solver provides Craig interpolants from an infeasibility
    proof, this operator uses them as predicates to split abstract states.
    This is the most precise refinement strategy but requires interpolant
    support from the solver.
    """

    def name(self) -> str:
        """Return the operator name."""
        return "PredicateRefinement"

    def is_applicable(
        self,
        state: AbstractionState,
        counterexample: RefinementCounterexample,
        proof: InfeasibilityProof | None = None,
    ) -> bool:
        """Check if an infeasibility proof with interpolants is available.

        Args:
            state: Current abstraction.
            counterexample: Spurious counterexample.
            proof: Infeasibility proof (must have interpolants).

        Returns:
            True if interpolants are available.
        """
        if proof is None:
            return False
        return proof.has_interpolants() or proof.has_unsat_core()

    def apply(
        self,
        state: AbstractionState,
        counterexample: RefinementCounterexample,
        proof: InfeasibilityProof | None = None,
    ) -> RefinementResult:
        """Split using interpolants/core predicates.

        If interpolants are available, use the first interpolant as a
        distinguishing predicate.  Otherwise, derive a predicate from
        the unsat core.

        Args:
            state: Current abstraction (modified in place).
            counterexample: Spurious counterexample.
            proof: Infeasibility proof with interpolants.

        Returns:
            RefinementResult.
        """
        if proof is None:
            return RefinementResult(
                status=RefinementStatus.FAILED,
                details={"reason": "no proof provided"},
            )

        predicate = self._select_predicate(proof, counterexample)
        if predicate is None:
            return RefinementResult(
                status=RefinementStatus.FAILED,
                details={"reason": "no suitable predicate found"},
            )

        target_sid = counterexample.state_id
        if not target_sid:
            target_state = state.partition.get_state_for_path(counterexample.path_id)
            if target_state is None:
                return RefinementResult(
                    status=RefinementStatus.FAILED,
                    details={"reason": "path not found in partition"},
                )
            target_sid = target_state.state_id

        path_set = state.path_set
        if path_set is None:
            cex_pid = counterexample.path_id
            t_id, f_id = state.split_state(
                target_sid,
                predicate=lambda pid: pid == cex_pid,
                kind=RefinementKind.PREDICATE_ADD,
                distinguishing_pred=predicate,
            )
        else:
            satisfying_paths = self._evaluate_predicate_on_paths(
                predicate, path_set, state.partition.get_state(target_sid)
            )
            t_id, f_id = state.split_state(
                target_sid,
                predicate=lambda pid: pid in satisfying_paths,
                kind=RefinementKind.PREDICATE_ADD,
                distinguishing_pred=predicate,
            )

        true_state = state.partition.get_state(t_id)
        if true_state is not None:
            true_state.add_predicate(predicate)
        false_state = state.partition.get_state(f_id)
        if false_state is not None:
            negated = PathCondition.negate_expr(predicate)
            false_state.add_predicate(negated)

        return RefinementResult(
            status=RefinementStatus.SUCCESS,
            refined_state=state,
            record=state.history[-1] if state.history else None,
            new_state_ids=[t_id, f_id],
            eliminated_cex=True,
            details={"predicate": str(predicate)},
        )

    def _select_predicate(
        self,
        proof: InfeasibilityProof,
        counterexample: RefinementCounterexample,
    ) -> TypedExpr | None:
        """Select the best distinguishing predicate from the proof.

        Prefers interpolants over unsat core elements, and selects
        the predicate with the fewest free variables.

        Args:
            proof: Infeasibility proof.
            counterexample: The counterexample being eliminated.

        Returns:
            The selected predicate, or None.
        """
        candidates: list[TypedExpr] = []

        if proof.has_interpolants():
            candidates.extend(proof.interpolants)

        if not candidates and proof.has_unsat_core():
            candidates.extend(proof.unsat_core)

        if not candidates:
            return None

        scored = []
        for pred in candidates:
            fvs = pred.free_vars()
            score = len(fvs)
            scored.append((score, pred))

        scored.sort(key=lambda x: x[0])
        return scored[0][1]

    def _evaluate_predicate_on_paths(
        self,
        predicate: TypedExpr,
        path_set: PathSet,
        abstract_state: AbstractState | None,
    ) -> set[int]:
        """Evaluate which paths in the abstract state satisfy the predicate.

        Uses syntactic matching: a path satisfies the predicate if it
        appears in the path condition or if the predicate is implied
        by the path condition.

        Args:
            predicate: The distinguishing predicate.
            path_set: Full set of symbolic paths.
            abstract_state: The state being split.

        Returns:
            Set of path IDs that satisfy the predicate.
        """
        if abstract_state is None:
            return set()

        satisfying: set[int] = set()
        pred_str = str(predicate)

        for pid in abstract_state.path_ids:
            path = path_set.get(pid)
            if path is None:
                continue
            pc_str = str(path.path_condition)
            if pred_str in pc_str:
                satisfying.add(pid)
            elif path.path_condition.implies(PathCondition.from_expr(predicate)):
                satisfying.add(pid)

        if not satisfying:
            for pid in abstract_state.path_ids:
                path = path_set.get(pid)
                if path is None:
                    continue
                fvs = predicate.free_vars()
                path_vars = path.get_free_vars()
                if fvs.issubset(path_vars):
                    satisfying.add(pid)
                    break

        return satisfying

    def estimated_cost(
        self,
        state: AbstractionState,
        counterexample: RefinementCounterexample,
    ) -> float:
        """Predicate refinement is expensive: cost = 5.0.

        Args:
            state: Current abstraction.
            counterexample: The counterexample.

        Returns:
            Cost value of 5.0.
        """
        return 5.0


# ═══════════════════════════════════════════════════════════════════════════
# LOOP UNWIND REFINEMENT
# ═══════════════════════════════════════════════════════════════════════════


class LoopUnwindRefinement(RefinementOperator):
    """Increase loop unrolling depth to expose more paths.

    When a counterexample involves a loop, this operator increases the
    unrolling depth for that loop, generating new paths that may
    disambiguate the spurious counterexample.
    """

    def __init__(self, max_unroll: int = 20) -> None:
        """Initialise with a maximum unrolling depth.

        Args:
            max_unroll: Maximum number of loop iterations to unroll.
        """
        self._max_unroll = max_unroll
        self._current_depths: dict[int, int] = {}

    def name(self) -> str:
        """Return the operator name."""
        return "LoopUnwind"

    def is_applicable(
        self,
        state: AbstractionState,
        counterexample: RefinementCounterexample,
        proof: InfeasibilityProof | None = None,
    ) -> bool:
        """Check if the counterexample involves a loop that can be further unrolled.

        Args:
            state: Current abstraction.
            counterexample: Spurious counterexample.
            proof: Optional proof.

        Returns:
            True if there are loops that haven't reached max unroll depth.
        """
        loop_nodes = counterexample.metadata.get("loop_nodes", [])
        if not loop_nodes:
            return False
        for node_id in loop_nodes:
            current = self._current_depths.get(node_id, 1)
            if current < self._max_unroll:
                return True
        return False

    def apply(
        self,
        state: AbstractionState,
        counterexample: RefinementCounterexample,
        proof: InfeasibilityProof | None = None,
    ) -> RefinementResult:
        """Increase unrolling depth for loops in the counterexample.

        This records the new unrolling depth and marks the abstraction
        as needing re-enumeration.

        Args:
            state: Current abstraction (modified in place).
            counterexample: Spurious counterexample.
            proof: Optional proof.

        Returns:
            RefinementResult.
        """
        loop_nodes = counterexample.metadata.get("loop_nodes", [])
        if not loop_nodes:
            return RefinementResult(
                status=RefinementStatus.NO_PROGRESS,
                details={"reason": "no loop nodes found"},
            )

        increased: list[int] = []
        for node_id in loop_nodes:
            current = self._current_depths.get(node_id, 1)
            if current < self._max_unroll:
                self._current_depths[node_id] = current + 1
                increased.append(node_id)

        if not increased:
            return RefinementResult(
                status=RefinementStatus.NO_PROGRESS,
                details={"reason": "all loops at max depth"},
            )

        state.refinement_level += 1
        record = RefinementRecord(
            kind=RefinementKind.LOOP_UNWIND,
            source_state_id=counterexample.state_id,
            result_state_ids=[counterexample.state_id],
            details={
                "increased_loops": increased,
                "new_depths": {nid: self._current_depths[nid] for nid in increased},
            },
            iteration=state.refinement_level,
        )
        state.record_refinement(record)
        state.metadata["needs_reenumeration"] = True
        state.metadata["loop_depths"] = dict(self._current_depths)

        return RefinementResult(
            status=RefinementStatus.SUCCESS,
            refined_state=state,
            record=record,
            new_state_ids=[],
            eliminated_cex=False,
            details={"reenumeration_needed": True},
        )

    def get_unroll_depth(self, loop_node_id: int) -> int:
        """Return the current unrolling depth for a loop node.

        Args:
            loop_node_id: The loop node ID.

        Returns:
            Current unrolling depth.
        """
        return self._current_depths.get(loop_node_id, 1)

    def estimated_cost(
        self,
        state: AbstractionState,
        counterexample: RefinementCounterexample,
    ) -> float:
        """Loop unrolling is very expensive: cost = 10.0.

        Increases exponentially with current depth since each unrolling
        may double the path count.

        Args:
            state: Current abstraction.
            counterexample: The counterexample.

        Returns:
            Cost value (base 10.0, scaled by depth).
        """
        loop_nodes = counterexample.metadata.get("loop_nodes", [])
        max_depth = max(
            (self._current_depths.get(nid, 1) for nid in loop_nodes),
            default=1,
        )
        return 10.0 * (2.0 ** max_depth)


# ═══════════════════════════════════════════════════════════════════════════
# REFINEMENT SELECTOR
# ═══════════════════════════════════════════════════════════════════════════


class RefinementSelector:
    """Choose the best refinement operator based on counterexample analysis.

    The selector ranks applicable operators by a cost heuristic and
    selects the cheapest one.  The cost accounts for:
      - Expected increase in abstract states
      - Solver difficulty
      - Historical success rate of each operator

    Attributes:
        operators: Ordered list of refinement operators.
        success_counts: Per-operator success counter.
        failure_counts: Per-operator failure counter.
    """

    def __init__(
        self,
        operators: list[RefinementOperator] | None = None,
    ) -> None:
        """Initialise with a list of operators.

        If no operators are provided, uses the default set:
        PathSplit, IntervalNarrow, PredicateRefinement, LoopUnwind.

        Args:
            operators: Optional custom operator list.
        """
        if operators is None:
            self._operators: list[RefinementOperator] = [
                PathSplitRefinement(),
                IntervalNarrowRefinement(),
                PredicateRefinement(),
                LoopUnwindRefinement(),
            ]
        else:
            self._operators = list(operators)

        self._success_counts: dict[str, int] = {
            op.name(): 0 for op in self._operators
        }
        self._failure_counts: dict[str, int] = {
            op.name(): 0 for op in self._operators
        }

    def select(
        self,
        state: AbstractionState,
        counterexample: RefinementCounterexample,
        proof: InfeasibilityProof | None = None,
    ) -> RefinementOperator | None:
        """Select the best applicable refinement operator.

        Operators are scored by:
          estimated_cost / (1 + success_rate)
        where success_rate = successes / (successes + failures + 1).

        Args:
            state: Current abstraction.
            counterexample: Spurious counterexample.
            proof: Optional infeasibility proof.

        Returns:
            The best operator, or None if none are applicable.
        """
        candidates: list[tuple[float, RefinementOperator]] = []

        for op in self._operators:
            if not op.is_applicable(state, counterexample, proof):
                continue
            base_cost = op.estimated_cost(state, counterexample)
            name = op.name()
            successes = self._success_counts.get(name, 0)
            failures = self._failure_counts.get(name, 0)
            success_rate = successes / (successes + failures + 1)
            adjusted_cost = base_cost / (1.0 + success_rate)
            candidates.append((adjusted_cost, op))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

    def select_and_apply(
        self,
        state: AbstractionState,
        counterexample: RefinementCounterexample,
        proof: InfeasibilityProof | None = None,
    ) -> RefinementResult:
        """Select the best operator and apply it.

        Updates success/failure counters based on the outcome.

        Args:
            state: Current abstraction.
            counterexample: Spurious counterexample.
            proof: Optional infeasibility proof.

        Returns:
            RefinementResult from the chosen operator.
        """
        op = self.select(state, counterexample, proof)
        if op is None:
            return RefinementResult(
                status=RefinementStatus.FAILED,
                details={"reason": "no applicable operator"},
            )

        result = op.apply(state, counterexample, proof)
        name = op.name()

        if result.success:
            self._success_counts[name] = self._success_counts.get(name, 0) + 1
        else:
            self._failure_counts[name] = self._failure_counts.get(name, 0) + 1

        result.details["operator"] = name
        return result

    def get_statistics(self) -> dict[str, dict[str, int]]:
        """Return per-operator success/failure statistics.

        Returns:
            Mapping from operator name to success/failure counts.
        """
        return {
            name: {
                "successes": self._success_counts.get(name, 0),
                "failures": self._failure_counts.get(name, 0),
            }
            for name in self._success_counts
        }


# ═══════════════════════════════════════════════════════════════════════════
# REFINEMENT HISTORY
# ═══════════════════════════════════════════════════════════════════════════


class RefinementHistory:
    """Track all refinements and detect cycles/stalls.

    Maintains the full refinement history and provides analysis methods
    for detecting when CEGAR is cycling (applying the same refinements
    repeatedly) or stalling (no progress).

    Attributes:
        records: Ordered list of all refinement records.
        _fingerprints: Set of state fingerprints for cycle detection.
    """

    def __init__(self) -> None:
        """Initialise an empty history."""
        self._records: list[RefinementRecord] = []
        self._fingerprints: set[str] = set()
        self._timestamps: list[float] = []

    def add(self, record: RefinementRecord) -> None:
        """Add a refinement record to the history.

        Args:
            record: The refinement to record.
        """
        self._records.append(record)
        self._timestamps.append(time.monotonic())

    def record_state_fingerprint(self, state: AbstractionState) -> bool:
        """Record the current abstraction fingerprint and check for cycles.

        The fingerprint is based on the partition structure (which paths
        are in which states).

        Args:
            state: Current abstraction.

        Returns:
            True if this fingerprint has been seen before (cycle detected).
        """
        fp = self._compute_fingerprint(state)
        if fp in self._fingerprints:
            return True
        self._fingerprints.add(fp)
        return False

    def _compute_fingerprint(self, state: AbstractionState) -> str:
        """Compute a string fingerprint of the partition structure.

        Args:
            state: The abstraction state to fingerprint.

        Returns:
            A canonical string representing the partition.
        """
        parts = []
        for sid in sorted(state.partition.states.keys()):
            s = state.partition.states[sid]
            pids = sorted(s.path_ids)
            parts.append(f"{sid}:{pids}")
        return "|".join(parts)

    def is_cycling(self, window: int = 10) -> bool:
        """Detect whether CEGAR is cycling.

        A cycle is detected if the same refinement operation (same kind
        and same source state) appears more than twice in the last
        *window* refinements.

        Args:
            window: Number of recent refinements to examine.

        Returns:
            True if a cycle is detected.
        """
        if len(self._records) < window:
            return False

        recent = self._records[-window:]
        signatures: dict[str, int] = {}
        for r in recent:
            sig = f"{r.kind.name}:{r.source_state_id}"
            signatures[sig] = signatures.get(sig, 0) + 1

        return any(count > 2 for count in signatures.values())

    def is_stalling(self, window: int = 5) -> bool:
        """Detect whether CEGAR is stalling (no progress).

        Stalling is detected when the last *window* refinements all
        resulted in NO_PROGRESS or the refinement level hasn't changed.

        Args:
            window: Number of recent refinements to examine.

        Returns:
            True if stalling is detected.
        """
        if len(self._records) < window:
            return False

        recent = self._records[-window:]
        levels = set()
        for r in recent:
            levels.add(r.iteration)

        return len(levels) <= 1

    def total_refinements(self) -> int:
        """Return the total number of refinements applied.

        Returns:
            Count of refinement records.
        """
        return len(self._records)

    def refinements_by_kind(self) -> dict[str, int]:
        """Return counts of refinements grouped by kind.

        Returns:
            Mapping from kind name to count.
        """
        counts: dict[str, int] = {}
        for r in self._records:
            k = r.kind.name
            counts[k] = counts.get(k, 0) + 1
        return counts

    def last_refinement(self) -> RefinementRecord | None:
        """Return the most recent refinement record.

        Returns:
            The last record, or None if empty.
        """
        return self._records[-1] if self._records else None

    def elapsed_time(self) -> float:
        """Return the elapsed wall-clock time since the first refinement.

        Returns:
            Seconds elapsed, or 0.0 if no refinements.
        """
        if len(self._timestamps) < 2:
            return 0.0
        return self._timestamps[-1] - self._timestamps[0]

    def summary(self) -> dict[str, Any]:
        """Return a summary of the refinement history.

        Returns:
            Dictionary with total, by-kind counts, and cycle/stall status.
        """
        return {
            "total": self.total_refinements(),
            "by_kind": self.refinements_by_kind(),
            "is_cycling": self.is_cycling(),
            "is_stalling": self.is_stalling(),
            "elapsed_time": self.elapsed_time(),
        }

    def __len__(self) -> int:
        return len(self._records)

    def __str__(self) -> str:
        return f"RefinementHistory({len(self._records)} records)"


# ═══════════════════════════════════════════════════════════════════════════
# CONVERGENCE DETECTOR
# ═══════════════════════════════════════════════════════════════════════════


class ConvergenceDetector:
    """Detect when the CEGAR loop has converged.

    Convergence can be:
      1. Exact: abstraction is the finest possible
      2. Fixpoint: density bounds have stabilised
      3. Bounded: bounds are within the privacy budget
      4. Timeout: resource limit reached

    Attributes:
        max_refinements: Maximum number of refinement steps.
        max_time_seconds: Maximum wall-clock time.
        fixpoint_tolerance: Tolerance for detecting bound stability.
    """

    def __init__(
        self,
        max_refinements: int = 100,
        max_time_seconds: float = 300.0,
        fixpoint_tolerance: float = 1e-8,
    ) -> None:
        """Initialise convergence parameters.

        Args:
            max_refinements: Maximum refinement iterations.
            max_time_seconds: Maximum time in seconds.
            fixpoint_tolerance: Tolerance for fixpoint detection.
        """
        self._max_refinements = max_refinements
        self._max_time = max_time_seconds
        self._tolerance = fixpoint_tolerance
        self._start_time: float | None = None
        self._prev_bounds: dict[str, AbstractDensityBound] | None = None

    def start(self) -> None:
        """Mark the start of the CEGAR loop."""
        self._start_time = time.monotonic()
        self._prev_bounds = None

    def check(
        self,
        state: AbstractionState,
        history: RefinementHistory,
        budget: PrivacyBudget | None = None,
    ) -> ConvergenceStatus:
        """Check all convergence criteria.

        Args:
            state: Current abstraction.
            history: Refinement history.
            budget: Optional privacy budget to check against.

        Returns:
            The convergence status.
        """
        if self._start_time is not None:
            elapsed = time.monotonic() - self._start_time
            if elapsed > self._max_time:
                return ConvergenceStatus(
                    converged=True,
                    reason=ConvergenceReason.TIMEOUT,
                    details={"elapsed": elapsed, "limit": self._max_time},
                )

        if history.total_refinements() >= self._max_refinements:
            return ConvergenceStatus(
                converged=True,
                reason=ConvergenceReason.MAX_REFINEMENTS,
                details={"refinements": history.total_refinements()},
            )

        if history.is_cycling():
            return ConvergenceStatus(
                converged=True,
                reason=ConvergenceReason.CYCLE_DETECTED,
                details={"refinements": history.total_refinements()},
            )

        if state.is_finest():
            return ConvergenceStatus(
                converged=True,
                reason=ConvergenceReason.FINEST_REACHED,
            )

        if self._check_fixpoint(state):
            return ConvergenceStatus(
                converged=True,
                reason=ConvergenceReason.FIXPOINT_REACHED,
            )

        if budget is not None:
            overall = state.overall_density_bound()
            eps, _ = budget.to_approx_dp()
            if overall.satisfies_epsilon(eps):
                return ConvergenceStatus(
                    converged=True,
                    reason=ConvergenceReason.BUDGET_SATISFIED,
                    details={"bound": str(overall), "budget": str(budget)},
                )

        return ConvergenceStatus(converged=False)

    def _check_fixpoint(self, state: AbstractionState) -> bool:
        """Check if density bounds have stabilised.

        Args:
            state: Current abstraction.

        Returns:
            True if bounds haven't changed since last check.
        """
        curr = dict(state.density_bounds)

        if self._prev_bounds is None:
            self._prev_bounds = curr
            return False

        if set(curr.keys()) != set(self._prev_bounds.keys()):
            self._prev_bounds = curr
            return False

        for sid in curr:
            c = curr[sid]
            p = self._prev_bounds[sid]
            if abs(c.lo - p.lo) > self._tolerance or abs(c.hi - p.hi) > self._tolerance:
                self._prev_bounds = curr
                return False

        self._prev_bounds = curr
        return True


class ConvergenceReason(Enum):
    """Reason why convergence was detected."""

    NOT_CONVERGED = auto()
    FINEST_REACHED = auto()
    FIXPOINT_REACHED = auto()
    BUDGET_SATISFIED = auto()
    MAX_REFINEMENTS = auto()
    CYCLE_DETECTED = auto()
    TIMEOUT = auto()


@dataclass(frozen=True, slots=True)
class ConvergenceStatus:
    """Result of a convergence check.

    Attributes:
        converged: Whether convergence has been detected.
        reason: The reason for convergence.
        details: Additional information.
    """

    converged: bool = False
    reason: ConvergenceReason = ConvergenceReason.NOT_CONVERGED
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        if not self.converged:
            return "NotConverged"
        return f"Converged({self.reason.name})"


def __getattr__(name: str):
    if name == "SpuriousnessCause":
        from dpcegar.cegar.engine import SpuriousnessCause
        return SpuriousnessCause
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
