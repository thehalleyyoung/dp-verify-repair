"""Abstract domain for CEGAR-based differential privacy verification.

This module provides the abstract state representation used in the CEGAR
loop.  Paths enumerated from the mechanism IR are grouped into
equivalence classes (abstract states), each annotated with interval
bounds on the log density ratio.  The abstraction lattice supports
merge, split, and widening operations needed for convergence.

Classes
-------
AbstractDensityBound   – interval bound on log density ratio
AbstractState          – single abstract state (path class + bound)
PathPartition          – partition of symbolic paths into classes
AbstractionState       – full abstraction snapshot for the CEGAR loop
AbstractionLattice     – lattice ordering and comparison
InitialAbstraction     – factory for the coarsest abstraction
WideningOperator       – widening for convergence acceleration
RefinementRecord       – record of a single refinement step
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterator,
    List,
    Mapping,
    Optional,
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
    PureBudget,
    ApproxBudget,
    TypedExpr,
    Var,
)
from dpcegar.paths.symbolic_path import (
    NoiseDrawInfo,
    PathCondition,
    PathSet,
    SymbolicPath,
)
from dpcegar.density.ratio_builder import DensityRatioExpr
from dpcegar.utils.errors import InternalError, ensure


# ═══════════════════════════════════════════════════════════════════════════
# ABSTRACT DENSITY BOUND
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class AbstractDensityBound:
    """Interval [lo, hi] bounding the log density ratio for an abstract class.

    The privacy loss random variable L(o) = ln(p(o|d)/p(o|d')) is bounded
    within this interval for all observations *o* consistent with the path
    conditions in the abstract state.

    Attributes:
        lo: Lower bound on the log density ratio.
        hi: Upper bound on the log density ratio.
        is_exact: True if the bounds are tight (no over-approximation).
        source_notion: Which privacy notion produced these bounds.
    """

    lo: float = -math.inf
    hi: float = math.inf
    is_exact: bool = False
    source_notion: str = "unknown"

    def __post_init__(self) -> None:
        """Validate that the interval is non-empty."""
        if self.lo > self.hi:
            raise ValueError(
                f"Empty density bound interval: [{self.lo}, {self.hi}]"
            )

    @classmethod
    def unbounded(cls) -> AbstractDensityBound:
        """Create a fully unbounded density interval (⊤ in the lattice)."""
        return cls(lo=-math.inf, hi=math.inf, is_exact=False, source_notion="top")

    @classmethod
    def exact(cls, value: float) -> AbstractDensityBound:
        """Create a point bound [v, v] for a known exact density ratio."""
        return cls(lo=value, hi=value, is_exact=True, source_notion="exact")

    @classmethod
    def from_interval(cls, lo: float, hi: float, source: str = "computed") -> AbstractDensityBound:
        """Create bounds from explicit lower/upper values."""
        return cls(lo=lo, hi=hi, is_exact=(lo == hi), source_notion=source)

    # -- Predicates -------------------------------------------------------

    def contains(self, value: float) -> bool:
        """Return True if *value* lies inside [lo, hi]."""
        return self.lo <= value <= self.hi

    def overlaps(self, other: AbstractDensityBound) -> bool:
        """Return True if the two intervals share at least one point."""
        return self.lo <= other.hi and other.lo <= self.hi

    def is_subset_of(self, other: AbstractDensityBound) -> bool:
        """Return True if *self* ⊆ *other*."""
        return other.lo <= self.lo and self.hi <= other.hi

    def is_unbounded(self) -> bool:
        """Return True if either endpoint is infinite."""
        return math.isinf(self.lo) or math.isinf(self.hi)

    @property
    def width(self) -> float:
        """Width of the interval (may be inf)."""
        return self.hi - self.lo

    @property
    def midpoint(self) -> float:
        """Centre of the interval (0.0 if unbounded both ways)."""
        if math.isinf(self.lo) and math.isinf(self.hi):
            return 0.0
        if math.isinf(self.lo):
            return self.hi - 1.0
        if math.isinf(self.hi):
            return self.lo + 1.0
        return (self.lo + self.hi) / 2.0

    # -- Operations -------------------------------------------------------

    def meet(self, other: AbstractDensityBound) -> AbstractDensityBound:
        """Greatest lower bound (intersection, tightest enclosing)."""
        new_lo = max(self.lo, other.lo)
        new_hi = min(self.hi, other.hi)
        if new_lo > new_hi:
            raise ValueError("Meet of non-overlapping intervals is empty")
        return AbstractDensityBound(
            lo=new_lo,
            hi=new_hi,
            is_exact=(new_lo == new_hi),
            source_notion="meet",
        )

    def join(self, other: AbstractDensityBound) -> AbstractDensityBound:
        """Least upper bound (convex hull)."""
        return AbstractDensityBound(
            lo=min(self.lo, other.lo),
            hi=max(self.hi, other.hi),
            is_exact=False,
            source_notion="join",
        )

    def widen(self, other: AbstractDensityBound, threshold: float = 100.0) -> AbstractDensityBound:
        """Widening operator: push unstable bounds to infinity.

        If *other* extends beyond *self*, push the corresponding bound
        to ±threshold (not ±∞ to keep things finite for SMT).  This
        guarantees termination of the fixpoint iteration.

        Args:
            other: The new bound to incorporate.
            threshold: Maximum finite bound magnitude.

        Returns:
            Widened interval.
        """
        new_lo = self.lo if other.lo >= self.lo else -threshold
        new_hi = self.hi if other.hi <= self.hi else threshold
        return AbstractDensityBound(
            lo=new_lo,
            hi=new_hi,
            is_exact=False,
            source_notion="widened",
        )

    def narrow(self, other: AbstractDensityBound) -> AbstractDensityBound:
        """Narrowing operator: tighten infinite bounds using *other*.

        Dual of widening; used after a stable fixpoint is reached to
        recover precision.

        Args:
            other: Tighter bound information.

        Returns:
            Narrowed interval.
        """
        new_lo = other.lo if math.isinf(self.lo) or self.lo == -100.0 else self.lo
        new_hi = other.hi if math.isinf(self.hi) or self.hi == 100.0 else self.hi
        return AbstractDensityBound(
            lo=new_lo,
            hi=new_hi,
            is_exact=(new_lo == new_hi),
            source_notion="narrowed",
        )

    def satisfies_epsilon(self, epsilon: float) -> bool:
        """Return True if the entire interval is within [-ε, ε]."""
        return self.lo >= -epsilon and self.hi <= epsilon

    def __str__(self) -> str:
        lo_s = f"{self.lo:.4f}" if not math.isinf(self.lo) else "-∞"
        hi_s = f"{self.hi:.4f}" if not math.isinf(self.hi) else "+∞"
        exact = " (exact)" if self.is_exact else ""
        return f"[{lo_s}, {hi_s}]{exact}"

    def __repr__(self) -> str:
        return f"AbstractDensityBound(lo={self.lo}, hi={self.hi})"


# ═══════════════════════════════════════════════════════════════════════════
# REFINEMENT RECORD
# ═══════════════════════════════════════════════════════════════════════════


class RefinementKind(Enum):
    """Kind of refinement operation applied."""

    PATH_SPLIT = auto()
    INTERVAL_NARROW = auto()
    PREDICATE_ADD = auto()
    LOOP_UNWIND = auto()
    MERGE = auto()


@dataclass(frozen=True, slots=True)
class RefinementRecord:
    """Record of a single refinement step in the CEGAR loop.

    Attributes:
        kind: The type of refinement applied.
        source_state_id: The abstract state that was refined.
        result_state_ids: Resulting abstract state IDs after refinement.
        predicate: The distinguishing predicate (for predicate refinement).
        details: Additional metadata about the refinement.
        iteration: CEGAR iteration number when this refinement was applied.
    """

    kind: RefinementKind
    source_state_id: str
    result_state_ids: list[str] = field(default_factory=list)
    predicate: TypedExpr | None = None
    details: dict[str, Any] = field(default_factory=dict)
    iteration: int = 0

    def __str__(self) -> str:
        return (
            f"Refinement({self.kind.name}, "
            f"{self.source_state_id} → {self.result_state_ids})"
        )


# ═══════════════════════════════════════════════════════════════════════════
# ABSTRACT STATE
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class AbstractState:
    """A single abstract state grouping a set of symbolic paths.

    Each abstract state represents an equivalence class of paths that
    share the same density ratio bounds.  The state tracks which path
    IDs belong to it and the abstracted density interval.

    Attributes:
        state_id: Unique identifier for this abstract state.
        path_ids: Set of symbolic path IDs belonging to this class.
        density_bound: Interval bound on the log density ratio.
        predicates: Set of distinguishing predicates used to define this state.
        metadata: Additional annotations.
    """

    state_id: str
    path_ids: set[int] = field(default_factory=set)
    density_bound: AbstractDensityBound = field(
        default_factory=AbstractDensityBound.unbounded
    )
    predicates: list[TypedExpr] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def size(self) -> int:
        """Return the number of paths in this abstract state."""
        return len(self.path_ids)

    def is_singleton(self) -> bool:
        """Return True if this state contains exactly one path."""
        return len(self.path_ids) == 1

    def is_empty(self) -> bool:
        """Return True if this state contains no paths."""
        return len(self.path_ids) == 0

    def contains_path(self, path_id: int) -> bool:
        """Return True if the given path belongs to this state."""
        return path_id in self.path_ids

    def add_path(self, path_id: int) -> None:
        """Add a path to this abstract state."""
        self.path_ids.add(path_id)

    def remove_path(self, path_id: int) -> None:
        """Remove a path from this abstract state."""
        self.path_ids.discard(path_id)

    def add_predicate(self, pred: TypedExpr) -> None:
        """Add a distinguishing predicate to this state's definition."""
        self.predicates.append(pred)

    def refine_bound(self, new_bound: AbstractDensityBound) -> None:
        """Tighten the density bound for this state.

        The new bound must be a subset of the current bound.

        Args:
            new_bound: Tighter density ratio interval.
        """
        if self.density_bound.is_unbounded():
            self.density_bound = new_bound
        else:
            self.density_bound = self.density_bound.meet(new_bound)

    def __str__(self) -> str:
        return (
            f"AbstractState({self.state_id}, "
            f"paths={len(self.path_ids)}, "
            f"bound={self.density_bound})"
        )

    def __repr__(self) -> str:
        return (
            f"AbstractState(id={self.state_id!r}, "
            f"paths={self.path_ids!r}, bound={self.density_bound!r})"
        )


# ═══════════════════════════════════════════════════════════════════════════
# PATH PARTITION
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class PathPartition:
    """A partition of symbolic paths into equivalence classes.

    Maintains the invariant that every path in the universe appears in
    exactly one class.

    Attributes:
        states: Mapping from state ID to AbstractState.
        path_to_state: Reverse mapping from path ID to state ID.
    """

    states: dict[str, AbstractState] = field(default_factory=dict)
    path_to_state: dict[int, str] = field(default_factory=dict)

    @classmethod
    def singleton_partition(cls, path_set: PathSet) -> PathPartition:
        """Create a partition where each path is its own class.

        This is the finest possible partition.

        Args:
            path_set: The set of paths to partition.

        Returns:
            A partition with one state per path.
        """
        partition = cls()
        for path in path_set:
            state_id = f"s_{path.path_id}"
            state = AbstractState(
                state_id=state_id,
                path_ids={path.path_id},
            )
            partition.states[state_id] = state
            partition.path_to_state[path.path_id] = state_id
        return partition

    @classmethod
    def coarsest_partition(cls, path_set: PathSet) -> PathPartition:
        """Create a single-class partition merging all paths.

        This is the coarsest possible partition (⊤ in the lattice).

        Args:
            path_set: The set of paths to partition.

        Returns:
            A partition with one state containing all paths.
        """
        all_ids = {p.path_id for p in path_set}
        state = AbstractState(state_id="s_top", path_ids=all_ids)
        partition = cls()
        partition.states["s_top"] = state
        for pid in all_ids:
            partition.path_to_state[pid] = "s_top"
        return partition

    def get_state(self, state_id: str) -> AbstractState | None:
        """Return the abstract state with the given ID, or None."""
        return self.states.get(state_id)

    def get_state_for_path(self, path_id: int) -> AbstractState | None:
        """Return the abstract state containing the given path, or None."""
        state_id = self.path_to_state.get(path_id)
        if state_id is None:
            return None
        return self.states.get(state_id)

    def state_count(self) -> int:
        """Return the number of abstract states in the partition."""
        return len(self.states)

    def all_path_ids(self) -> set[int]:
        """Return all path IDs across all states."""
        return set(self.path_to_state.keys())

    def split_state(
        self,
        state_id: str,
        predicate: Callable[[int], bool],
        true_id: str | None = None,
        false_id: str | None = None,
    ) -> tuple[str, str]:
        """Split an abstract state into two based on a predicate.

        Paths for which predicate(path_id) is True go into the first
        new state; the rest go into the second.

        Args:
            state_id: ID of the state to split.
            predicate: Function mapping path ID to True/False.
            true_id: Optional ID for the 'true' partition.
            false_id: Optional ID for the 'false' partition.

        Returns:
            Tuple of (true_state_id, false_state_id).

        Raises:
            InternalError: If the state does not exist.
        """
        old_state = self.states.get(state_id)
        ensure(old_state is not None, f"State {state_id} not found")
        assert old_state is not None  # for type checker

        true_paths: set[int] = set()
        false_paths: set[int] = set()
        for pid in old_state.path_ids:
            if predicate(pid):
                true_paths.add(pid)
            else:
                false_paths.add(pid)

        if not true_paths or not false_paths:
            return (state_id, state_id)

        t_id = true_id or f"{state_id}_T"
        f_id = false_id or f"{state_id}_F"

        true_state = AbstractState(
            state_id=t_id,
            path_ids=true_paths,
            density_bound=copy.copy(old_state.density_bound),
            predicates=list(old_state.predicates),
        )
        false_state = AbstractState(
            state_id=f_id,
            path_ids=false_paths,
            density_bound=copy.copy(old_state.density_bound),
            predicates=list(old_state.predicates),
        )

        del self.states[state_id]
        self.states[t_id] = true_state
        self.states[f_id] = false_state

        for pid in true_paths:
            self.path_to_state[pid] = t_id
        for pid in false_paths:
            self.path_to_state[pid] = f_id

        return (t_id, f_id)

    def merge_states(self, state_id_a: str, state_id_b: str, merged_id: str | None = None) -> str:
        """Merge two abstract states into one.

        The merged state contains the union of paths and the join of
        density bounds.

        Args:
            state_id_a: First state to merge.
            state_id_b: Second state to merge.
            merged_id: Optional ID for the merged state.

        Returns:
            ID of the merged state.

        Raises:
            InternalError: If either state does not exist.
        """
        state_a = self.states.get(state_id_a)
        state_b = self.states.get(state_id_b)
        ensure(state_a is not None, f"State {state_id_a} not found")
        ensure(state_b is not None, f"State {state_id_b} not found")
        assert state_a is not None and state_b is not None

        m_id = merged_id or f"{state_id_a}+{state_id_b}"
        merged_paths = state_a.path_ids | state_b.path_ids
        merged_bound = state_a.density_bound.join(state_b.density_bound)
        shared_preds = [
            p for p in state_a.predicates
            if any(str(p) == str(q) for q in state_b.predicates)
        ]

        merged = AbstractState(
            state_id=m_id,
            path_ids=merged_paths,
            density_bound=merged_bound,
            predicates=shared_preds,
        )

        del self.states[state_id_a]
        del self.states[state_id_b]
        self.states[m_id] = merged

        for pid in merged_paths:
            self.path_to_state[pid] = m_id

        return m_id

    def is_finest(self, path_set: PathSet) -> bool:
        """Return True if every state contains at most one path."""
        return all(s.is_singleton() or s.is_empty() for s in self.states.values())

    def is_coarsest(self) -> bool:
        """Return True if the partition has exactly one state."""
        return len(self.states) == 1

    def iter_states(self) -> Iterator[AbstractState]:
        """Iterate over all abstract states."""
        return iter(self.states.values())

    def __len__(self) -> int:
        return len(self.states)

    def __str__(self) -> str:
        return f"PathPartition({len(self.states)} states)"


# ═══════════════════════════════════════════════════════════════════════════
# ABSTRACTION STATE — Full CEGAR abstraction snapshot
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class AbstractionState:
    """Complete abstraction state for the CEGAR loop.

    Bundles the path partition, density bounds, and refinement history.
    This is the object passed between the abstract verifier and the
    refinement engine at each CEGAR iteration.

    Attributes:
        partition: The current path partition.
        density_bounds: Mapping from abstract state ID to density bound.
        refinement_level: How many refinements have been applied.
        history: Ordered list of refinement operations.
        path_set: Reference to the underlying path set.
        metadata: Additional metadata.
    """

    partition: PathPartition = field(default_factory=PathPartition)
    density_bounds: dict[str, AbstractDensityBound] = field(default_factory=dict)
    refinement_level: int = 0
    history: list[RefinementRecord] = field(default_factory=list)
    path_set: PathSet | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_abstract_density(self, state_id: str) -> AbstractDensityBound:
        """Return the density bound for the given abstract state.

        If no bound has been computed, returns ⊤ (unbounded).

        Args:
            state_id: Abstract state identifier.

        Returns:
            The density bound for that state.
        """
        if state_id in self.density_bounds:
            return self.density_bounds[state_id]
        state = self.partition.get_state(state_id)
        if state is not None:
            return state.density_bound
        return AbstractDensityBound.unbounded()

    def set_density_bound(self, state_id: str, bound: AbstractDensityBound) -> None:
        """Set the density bound for an abstract state.

        Args:
            state_id: Abstract state identifier.
            bound: The density interval to assign.
        """
        self.density_bounds[state_id] = bound
        state = self.partition.get_state(state_id)
        if state is not None:
            state.density_bound = bound

    def overall_density_bound(self) -> AbstractDensityBound:
        """Compute the join of all per-state density bounds.

        This gives the coarsest sound bound on the whole mechanism's
        privacy loss.

        Returns:
            The overall density bound (join of all states).
        """
        if not self.density_bounds and not self.partition.states:
            return AbstractDensityBound.unbounded()

        bounds = list(self.density_bounds.values())
        for state in self.partition.iter_states():
            if state.state_id not in self.density_bounds:
                bounds.append(state.density_bound)

        if not bounds:
            return AbstractDensityBound.unbounded()

        result = bounds[0]
        for b in bounds[1:]:
            result = result.join(b)
        return result

    def merge_states(self, state_id_a: str, state_id_b: str) -> str:
        """Merge two abstract states, updating bounds and history.

        Args:
            state_id_a: First state.
            state_id_b: Second state.

        Returns:
            ID of the merged state.
        """
        merged_id = self.partition.merge_states(state_id_a, state_id_b)

        bound_a = self.density_bounds.pop(state_id_a, AbstractDensityBound.unbounded())
        bound_b = self.density_bounds.pop(state_id_b, AbstractDensityBound.unbounded())
        self.density_bounds[merged_id] = bound_a.join(bound_b)

        self.history.append(RefinementRecord(
            kind=RefinementKind.MERGE,
            source_state_id=state_id_a,
            result_state_ids=[merged_id],
            details={"merged_with": state_id_b},
            iteration=self.refinement_level,
        ))

        return merged_id

    def split_state(
        self,
        state_id: str,
        predicate: Callable[[int], bool],
        kind: RefinementKind = RefinementKind.PATH_SPLIT,
        distinguishing_pred: TypedExpr | None = None,
    ) -> tuple[str, str]:
        """Split an abstract state, updating bounds and history.

        Args:
            state_id: State to split.
            predicate: Classifier for path IDs.
            kind: The refinement kind for the history record.
            distinguishing_pred: Optional predicate expression.

        Returns:
            Tuple of (true_state_id, false_state_id).
        """
        old_bound = self.density_bounds.pop(state_id, AbstractDensityBound.unbounded())
        t_id, f_id = self.partition.split_state(state_id, predicate)

        if t_id != state_id:
            self.density_bounds[t_id] = old_bound
            self.density_bounds[f_id] = old_bound
            self.refinement_level += 1
            self.history.append(RefinementRecord(
                kind=kind,
                source_state_id=state_id,
                result_state_ids=[t_id, f_id],
                predicate=distinguishing_pred,
                iteration=self.refinement_level,
            ))

        return (t_id, f_id)

    def is_finest(self) -> bool:
        """Return True if every abstract state is a singleton."""
        ps = self.path_set
        if ps is None:
            return all(
                s.is_singleton() for s in self.partition.iter_states()
            )
        return self.partition.is_finest(ps)

    def record_refinement(self, record: RefinementRecord) -> None:
        """Append a refinement record to the history.

        Args:
            record: The refinement record to store.
        """
        self.history.append(record)
        self.refinement_level = max(self.refinement_level, record.iteration)

    def all_state_ids(self) -> list[str]:
        """Return all current abstract state IDs."""
        return list(self.partition.states.keys())

    def summary(self) -> dict[str, Any]:
        """Return a summary of the current abstraction state."""
        return {
            "num_states": self.partition.state_count(),
            "refinement_level": self.refinement_level,
            "history_length": len(self.history),
            "overall_bound": str(self.overall_density_bound()),
        }

    def __str__(self) -> str:
        return (
            f"AbstractionState(states={self.partition.state_count()}, "
            f"level={self.refinement_level})"
        )


# ═══════════════════════════════════════════════════════════════════════════
# INITIAL ABSTRACTION FACTORY
# ═══════════════════════════════════════════════════════════════════════════


class InitialAbstraction:
    """Factory for creating the initial (coarsest) abstraction.

    The initial abstraction merges all paths into a single abstract state
    (or groups by noise pattern), providing the starting point for CEGAR.
    """

    @staticmethod
    def coarsest(path_set: PathSet) -> AbstractionState:
        """Create the coarsest abstraction: all paths in one class.

        Args:
            path_set: Enumerated symbolic paths.

        Returns:
            An AbstractionState with a single abstract state.
        """
        partition = PathPartition.coarsest_partition(path_set)
        state = AbstractionState(
            partition=partition,
            density_bounds={"s_top": AbstractDensityBound.unbounded()},
            refinement_level=0,
            path_set=path_set,
        )
        return state

    @staticmethod
    def by_noise_pattern(path_set: PathSet) -> AbstractionState:
        """Create an abstraction grouping paths by noise draw sites.

        Paths that draw noise from the same sites (in the same order)
        are placed in the same abstract state.  This is a natural
        initial grouping since paths with different noise patterns
        typically have different density ratio structures.

        Args:
            path_set: Enumerated symbolic paths.

        Returns:
            An AbstractionState with one state per noise pattern.
        """
        groups: dict[str, set[int]] = {}
        for path in path_set:
            key = ",".join(str(nd.site_id) for nd in path.noise_draws)
            groups.setdefault(key, set()).add(path.path_id)

        partition = PathPartition()
        density_bounds: dict[str, AbstractDensityBound] = {}

        for idx, (pattern, pids) in enumerate(groups.items()):
            state_id = f"s_noise_{idx}"
            partition.states[state_id] = AbstractState(
                state_id=state_id,
                path_ids=pids,
                metadata={"noise_pattern": pattern},
            )
            density_bounds[state_id] = AbstractDensityBound.unbounded()
            for pid in pids:
                partition.path_to_state[pid] = state_id

        return AbstractionState(
            partition=partition,
            density_bounds=density_bounds,
            refinement_level=0,
            path_set=path_set,
        )

    @staticmethod
    def by_branch_structure(path_set: PathSet) -> AbstractionState:
        """Create an abstraction grouping paths by branching pattern.

        Paths that visit the same IR nodes (ignoring order) are placed
        in the same abstract state.

        Args:
            path_set: Enumerated symbolic paths.

        Returns:
            An AbstractionState with one state per branch pattern.
        """
        groups: dict[str, set[int]] = {}
        for path in path_set:
            key = ",".join(str(nid) for nid in sorted(set(path.source_nodes)))
            groups.setdefault(key, set()).add(path.path_id)

        partition = PathPartition()
        density_bounds: dict[str, AbstractDensityBound] = {}

        for idx, (pattern, pids) in enumerate(groups.items()):
            state_id = f"s_branch_{idx}"
            partition.states[state_id] = AbstractState(
                state_id=state_id,
                path_ids=pids,
                metadata={"branch_pattern": pattern},
            )
            density_bounds[state_id] = AbstractDensityBound.unbounded()
            for pid in pids:
                partition.path_to_state[pid] = state_id

        return AbstractionState(
            partition=partition,
            density_bounds=density_bounds,
            refinement_level=0,
            path_set=path_set,
        )

    @staticmethod
    def finest(path_set: PathSet) -> AbstractionState:
        """Create the finest abstraction: one class per path.

        Useful for debugging or when the path count is small enough
        to verify each path individually.

        Args:
            path_set: Enumerated symbolic paths.

        Returns:
            An AbstractionState with singleton states.
        """
        partition = PathPartition.singleton_partition(path_set)
        density_bounds: dict[str, AbstractDensityBound] = {}
        for sid in partition.states:
            density_bounds[sid] = AbstractDensityBound.unbounded()

        return AbstractionState(
            partition=partition,
            density_bounds=density_bounds,
            refinement_level=0,
            path_set=path_set,
        )


# ═══════════════════════════════════════════════════════════════════════════
# ABSTRACTION LATTICE
# ═══════════════════════════════════════════════════════════════════════════


class AbstractionLattice:
    """Track lattice position and provide comparison operations.

    The abstraction lattice is ordered by refinement: A ⊑ B iff every
    class in B is a union of classes in A (A is finer than B).

    Methods allow checking whether one abstraction is a refinement of
    another and computing joins/meets in the lattice.
    """

    @staticmethod
    def is_refinement_of(finer: AbstractionState, coarser: AbstractionState) -> bool:
        """Check if *finer* refines *coarser* in the partition lattice.

        A refines B iff every class of A is contained within some class
        of B.  Equivalently, A's partition is finer than B's.

        Args:
            finer: Candidate finer abstraction.
            coarser: Candidate coarser abstraction.

        Returns:
            True if finer ⊑ coarser.
        """
        for state in finer.partition.iter_states():
            if state.is_empty():
                continue
            representative = next(iter(state.path_ids))
            coarse_sid = coarser.partition.path_to_state.get(representative)
            if coarse_sid is None:
                return False
            coarse_state = coarser.partition.get_state(coarse_sid)
            if coarse_state is None:
                return False
            if not state.path_ids.issubset(coarse_state.path_ids):
                return False
        return True

    @staticmethod
    def lattice_join(a: AbstractionState, b: AbstractionState) -> AbstractionState:
        """Compute the lattice join (coarsest common refinement).

        The join merges classes that overlap between the two abstractions,
        producing the least upper bound.

        Args:
            a: First abstraction.
            b: Second abstraction.

        Returns:
            The join abstraction.
        """
        all_pids = a.partition.all_path_ids() | b.partition.all_path_ids()

        parent: dict[int, int] = {pid: pid for pid in all_pids}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry

        for state in a.partition.iter_states():
            pids = list(state.path_ids)
            for i in range(1, len(pids)):
                union(pids[0], pids[i])

        for state in b.partition.iter_states():
            pids = list(state.path_ids)
            for i in range(1, len(pids)):
                union(pids[0], pids[i])

        groups: dict[int, set[int]] = {}
        for pid in all_pids:
            root = find(pid)
            groups.setdefault(root, set()).add(pid)

        partition = PathPartition()
        density_bounds: dict[str, AbstractDensityBound] = {}
        for idx, (_, pids) in enumerate(groups.items()):
            sid = f"s_join_{idx}"
            partition.states[sid] = AbstractState(state_id=sid, path_ids=pids)
            bound_a = AbstractDensityBound.unbounded()
            bound_b = AbstractDensityBound.unbounded()
            for pid in pids:
                a_sid = a.partition.path_to_state.get(pid)
                if a_sid and a_sid in a.density_bounds:
                    bound_a = bound_a.join(a.density_bounds[a_sid])
                b_sid = b.partition.path_to_state.get(pid)
                if b_sid and b_sid in b.density_bounds:
                    bound_b = bound_b.join(b.density_bounds[b_sid])
            density_bounds[sid] = bound_a.join(bound_b)
            for pid in pids:
                partition.path_to_state[pid] = sid

        return AbstractionState(
            partition=partition,
            density_bounds=density_bounds,
            refinement_level=max(a.refinement_level, b.refinement_level),
            path_set=a.path_set or b.path_set,
        )

    @staticmethod
    def lattice_meet(a: AbstractionState, b: AbstractionState) -> AbstractionState:
        """Compute the lattice meet (finest common coarsening).

        The meet intersects classes, producing the greatest lower bound.

        Args:
            a: First abstraction.
            b: Second abstraction.

        Returns:
            The meet abstraction.
        """
        partition = PathPartition()
        density_bounds: dict[str, AbstractDensityBound] = {}
        idx = 0

        for sa in a.partition.iter_states():
            for sb in b.partition.iter_states():
                overlap = sa.path_ids & sb.path_ids
                if not overlap:
                    continue
                sid = f"s_meet_{idx}"
                idx += 1
                partition.states[sid] = AbstractState(
                    state_id=sid, path_ids=overlap
                )
                bound_a = a.density_bounds.get(
                    sa.state_id, AbstractDensityBound.unbounded()
                )
                bound_b = b.density_bounds.get(
                    sb.state_id, AbstractDensityBound.unbounded()
                )
                if bound_a.overlaps(bound_b):
                    density_bounds[sid] = bound_a.meet(bound_b)
                else:
                    density_bounds[sid] = bound_a.join(bound_b)
                for pid in overlap:
                    partition.path_to_state[pid] = sid

        return AbstractionState(
            partition=partition,
            density_bounds=density_bounds,
            refinement_level=max(a.refinement_level, b.refinement_level),
            path_set=a.path_set or b.path_set,
        )


# ═══════════════════════════════════════════════════════════════════════════
# WIDENING OPERATOR
# ═══════════════════════════════════════════════════════════════════════════


class WideningOperator:
    """Widening operator for convergence acceleration.

    Prevents the CEGAR loop from diverging by forcing density bounds
    to stabilise.  After a configurable number of iterations without
    progress, widening pushes unstable interval endpoints toward a
    finite threshold.

    Attributes:
        threshold: Maximum finite bound magnitude.
        patience: Number of iterations before applying widening.
        _iteration_bounds: History of per-iteration bounds for stability detection.
    """

    def __init__(
        self,
        threshold: float = 100.0,
        patience: int = 5,
    ) -> None:
        """Initialise the widening operator.

        Args:
            threshold: Finite ceiling/floor for widened bounds.
            patience: How many rounds of instability before widening.
        """
        self._threshold = threshold
        self._patience = patience
        self._iteration_bounds: list[dict[str, AbstractDensityBound]] = []

    def should_widen(self) -> bool:
        """Return True if widening should be applied now.

        Widening is triggered when bounds have been oscillating or
        growing for *patience* consecutive iterations.

        Returns:
            True if widening is needed.
        """
        if len(self._iteration_bounds) < self._patience:
            return False
        recent = self._iteration_bounds[-self._patience:]
        for sid in recent[0]:
            widths = []
            for iteration_bounds in recent:
                b = iteration_bounds.get(sid)
                if b is not None:
                    widths.append(b.width)
            if len(widths) >= 2 and all(
                widths[i] <= widths[i + 1] for i in range(len(widths) - 1)
            ):
                return True
        return False

    def record_iteration(self, bounds: dict[str, AbstractDensityBound]) -> None:
        """Record the bounds from the current iteration.

        Args:
            bounds: The per-state density bounds for this iteration.
        """
        self._iteration_bounds.append(dict(bounds))

    def apply(self, state: AbstractionState) -> AbstractionState:
        """Apply widening to the abstraction state.

        For each abstract state whose bounds have been growing, widen
        the interval to the threshold.

        Args:
            state: Current abstraction state.

        Returns:
            The widened abstraction state (mutated in place and returned).
        """
        if len(self._iteration_bounds) < 2:
            return state

        prev_bounds = self._iteration_bounds[-2]
        curr_bounds = self._iteration_bounds[-1]

        for sid in list(state.density_bounds.keys()):
            curr = curr_bounds.get(sid)
            prev = prev_bounds.get(sid)
            if curr is None or prev is None:
                continue
            if curr.lo < prev.lo or curr.hi > prev.hi:
                widened = prev.widen(curr, self._threshold)
                state.set_density_bound(sid, widened)

        return state

    def apply_narrowing(self, state: AbstractionState) -> AbstractionState:
        """Apply narrowing to recover precision after widening.

        Uses the latest concrete bounds to tighten any interval that was
        pushed to the threshold by widening.

        Args:
            state: Current abstraction state (post-fixpoint).

        Returns:
            The narrowed abstraction state.
        """
        if not self._iteration_bounds:
            return state

        latest = self._iteration_bounds[-1]
        for sid in list(state.density_bounds.keys()):
            concrete_bound = latest.get(sid)
            if concrete_bound is None:
                continue
            current = state.density_bounds[sid]
            narrowed = current.narrow(concrete_bound)
            state.set_density_bound(sid, narrowed)

        return state

    def reset(self) -> None:
        """Clear all recorded iteration data."""
        self._iteration_bounds.clear()

    def __str__(self) -> str:
        return (
            f"WideningOperator(threshold={self._threshold}, "
            f"patience={self._patience}, "
            f"recorded={len(self._iteration_bounds)})"
        )
