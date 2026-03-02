"""Path enumeration engine for MechIR programs.

The :class:`PathEnumerator` performs symbolic execution of a MechIR
mechanism, systematically exploring all feasible control-flow paths.
At each branch it forks into true/false continuations; loops are
unrolled via the :class:`LoopUnroller`; and noise draws are recorded
symbolically.  Infeasible paths are pruned early using the
:class:`FeasibilityChecker`.

The result is a :class:`PathSet` containing all enumerated
:class:`SymbolicPath` objects.
"""

from __future__ import annotations

from collections import ChainMap
import copy
import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

from dpcegar.ir.types import (
    BinOp,
    BinOpKind,
    Const,
    IRType,
    NoiseKind,
    TypedExpr,
    UnaryOp,
    UnaryOpKind,
    Var,
)
from dpcegar.ir.nodes import (
    AssignNode,
    BranchNode,
    IRNode,
    LoopNode,
    MechIR,
    MergeNode,
    NoOpNode,
    NoiseDrawNode,
    ParamDecl,
    QueryNode,
    ReturnNode,
    SequenceNode,
)
from dpcegar.paths.symbolic_path import (
    NoiseDrawInfo,
    PathCondition,
    PathSet,
    SymbolicPath,
)
from dpcegar.paths.path_condition import (
    IntervalEnv,
    PathConditionManager,
)
from dpcegar.paths.feasibility import FeasibilityChecker
from dpcegar.paths.loop_unroller import (
    LoopUnroller,
    UnrollConfig,
)


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class EnumeratorConfig:
    """Configuration for the path enumerator.

    Attributes:
        max_paths:         Maximum number of paths to enumerate.
        prune_infeasible:  Whether to prune infeasible paths eagerly.
        merge_identical:   Whether to merge paths with identical suffixes.
        unroll_config:     Configuration for the loop unroller.
        track_stats:       Whether to track detailed enumeration statistics.
    """

    max_paths: int = 10_000
    prune_infeasible: bool = True
    merge_identical: bool = True
    unroll_config: UnrollConfig = field(default_factory=UnrollConfig)
    track_stats: bool = True
    enable_incremental_feasibility: bool = True


# ═══════════════════════════════════════════════════════════════════════════
# SYMBOLIC EXECUTION STATE
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class SymExecState:
    """Symbolic execution state tracked during path enumeration.

    Attributes:
        bindings:       Current variable-to-expression bindings.
        path_condition: Accumulated path condition (list of conjuncts).
        noise_draws:    Noise draws encountered so far.
        source_nodes:   IR node IDs visited so far.
    """

    bindings: dict[str, TypedExpr] = field(default_factory=dict)
    path_condition: list[TypedExpr] = field(default_factory=list)
    noise_draws: list[NoiseDrawInfo] = field(default_factory=list)
    source_nodes: list[int] = field(default_factory=list)

    def clone(self) -> SymExecState:
        """Return a shallow-forked copy of this state.

        Uses ChainMap for O(1) fork of bindings.
        """
        return SymExecState(
            bindings=ChainMap({}, self.bindings),
            path_condition=list(self.path_condition),
            noise_draws=list(self.noise_draws),
            source_nodes=list(self.source_nodes),
        )

    def record_node(self, node_id: int) -> None:
        """Record a visited node."""
        self.source_nodes.append(node_id)

    def bind(self, name: str, expr: TypedExpr) -> None:
        """Bind a variable to an expression."""
        self.bindings[name] = expr

    def lookup(self, name: str) -> TypedExpr | None:
        """Look up a variable binding."""
        return self.bindings.get(name)

    def resolve_expr(self, expr: TypedExpr) -> TypedExpr:
        """Resolve variables in *expr* using current bindings."""
        return expr.substitute(self.bindings)

    def add_condition(self, cond: TypedExpr) -> None:
        """Append a conjunct to the path condition."""
        self.path_condition.append(cond)

    def add_noise_draw(self, draw: NoiseDrawInfo) -> None:
        """Record a noise draw."""
        self.noise_draws.append(draw)


# ═══════════════════════════════════════════════════════════════════════════
# ENUMERATION STATISTICS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class EnumStats:
    """Statistics collected during path enumeration.

    Attributes:
        paths_explored:   Total paths fully explored.
        paths_pruned:     Paths pruned as infeasible.
        paths_merged:     Paths merged with identical suffixes.
        paths_budget_cut: Paths dropped due to budget exhaustion.
        branches_visited: Total branch nodes visited.
        loops_unrolled:   Total loops unrolled.
        noise_draws_seen: Total noise draws recorded.
        max_depth:        Maximum recursion / nesting depth.
    """

    paths_explored: int = 0
    paths_pruned: int = 0
    paths_merged: int = 0
    paths_budget_cut: int = 0
    branches_visited: int = 0
    loops_unrolled: int = 0
    noise_draws_seen: int = 0
    max_depth: int = 0

    def to_dict(self) -> dict[str, int]:
        """Convert to a dictionary."""
        return {
            "paths_explored": self.paths_explored,
            "paths_pruned": self.paths_pruned,
            "paths_merged": self.paths_merged,
            "paths_budget_cut": self.paths_budget_cut,
            "branches_visited": self.branches_visited,
            "loops_unrolled": self.loops_unrolled,
            "noise_draws_seen": self.noise_draws_seen,
            "max_depth": self.max_depth,
        }


# ═══════════════════════════════════════════════════════════════════════════
# PATH ENUMERATOR
# ═══════════════════════════════════════════════════════════════════════════


class PathEnumerator:
    """Enumerate all feasible control-flow paths through a MechIR program.

    The enumerator performs symbolic execution, forking at branches and
    unrolling loops.  Infeasible paths are pruned early via interval
    analysis, and identical-suffix paths may be merged for efficiency.

    Usage::

        enumerator = PathEnumerator()
        path_set = enumerator.enumerate(mechir)
        for path in path_set:
            print(path.pretty())

    Args:
        config: Enumeration configuration.
    """

    def __init__(self, config: EnumeratorConfig | None = None) -> None:
        self._config = config or EnumeratorConfig()
        self._feasibility = FeasibilityChecker()
        self._pc_manager = PathConditionManager()
        self._unroller = LoopUnroller(self._config.unroll_config)
        self._stats = EnumStats()
        self._depth = 0
        self._completed_paths: list[SymbolicPath] = []
        self._budget_exhausted = False

    # -- Public API --------------------------------------------------------

    def enumerate(self, mechir: MechIR) -> PathSet:
        """Enumerate all feasible paths in the given mechanism.

        Args:
            mechir: The mechanism IR to analyse.

        Returns:
            A :class:`PathSet` containing all enumerated paths.
        """
        self._completed_paths = []
        self._stats = EnumStats()
        self._budget_exhausted = False
        self._depth = 0

        initial_state = self._make_initial_state(mechir)
        self._explore_node(mechir.body, initial_state)

        path_set = PathSet(paths=list(self._completed_paths))

        if self._config.merge_identical and path_set.size() > 1:
            merged = path_set.merge_compatible()
            self._stats.paths_merged = path_set.size() - merged.size()
            path_set = merged

        path_set.metadata = {
            "stats": self._stats.to_dict(),
            "config": {
                "max_paths": self._config.max_paths,
                "prune_infeasible": self._config.prune_infeasible,
                "merge_identical": self._config.merge_identical,
            },
        }

        return path_set

    @property
    def stats(self) -> EnumStats:
        """Return the enumeration statistics."""
        return self._stats

    # -- Internal: node dispatch -------------------------------------------

    def _explore_node(self, node: IRNode, state: SymExecState) -> None:
        """Recursively explore an IR node, forking as needed."""
        if self._budget_exhausted:
            return

        if len(self._completed_paths) >= self._config.max_paths:
            self._budget_exhausted = True
            self._stats.paths_budget_cut += 1
            return

        self._depth += 1
        self._stats.max_depth = max(self._stats.max_depth, self._depth)
        state.record_node(node.node_id)

        try:
            if isinstance(node, SequenceNode):
                self._explore_sequence(node, state)
            elif isinstance(node, AssignNode):
                self._explore_assign(node, state)
            elif isinstance(node, NoiseDrawNode):
                self._explore_noise_draw(node, state)
            elif isinstance(node, BranchNode):
                self._explore_branch(node, state)
            elif isinstance(node, LoopNode):
                self._explore_loop(node, state)
            elif isinstance(node, QueryNode):
                self._explore_query(node, state)
            elif isinstance(node, ReturnNode):
                self._explore_return(node, state)
            elif isinstance(node, MergeNode):
                self._explore_merge(node, state)
            elif isinstance(node, NoOpNode):
                pass
            else:
                pass  # unknown node type, skip
        finally:
            self._depth -= 1

    def _explore_sequence(self, node: SequenceNode, state: SymExecState) -> None:
        """Execute a sequence of statements."""
        current_states = [state]

        for i, stmt in enumerate(node.stmts):
            if self._budget_exhausted:
                return

            next_states: list[SymExecState] = []
            for s in current_states:
                before_count = len(self._completed_paths)
                completed_snapshot = list(self._completed_paths)

                if isinstance(stmt, ReturnNode):
                    self._explore_return(stmt, s)
                elif isinstance(stmt, BranchNode):
                    branch_paths: list[SymExecState] = []
                    self._fork_branch(stmt, s, branch_paths)
                    if i < len(node.stmts) - 1:
                        next_states.extend(branch_paths)
                else:
                    self._explore_node(stmt, s)
                    if len(self._completed_paths) == before_count:
                        next_states.append(s)
                    else:
                        new_completed = self._completed_paths[before_count:]
                        self._completed_paths = completed_snapshot
                        next_states.append(s)

            current_states = next_states
            if not current_states:
                return

    def _explore_assign(self, node: AssignNode, state: SymExecState) -> None:
        """Execute an assignment: bind the target variable."""
        resolved_value = state.resolve_expr(node.value)
        state.bind(node.target.name, resolved_value)

    def _explore_noise_draw(self, node: NoiseDrawNode, state: SymExecState) -> None:
        """Record a noise draw and bind the target variable."""
        self._stats.noise_draws_seen += 1

        center = state.resolve_expr(node.center)
        scale = state.resolve_expr(node.scale)

        draw = NoiseDrawInfo(
            variable=node.target.name,
            kind=node.noise_kind,
            center_expr=center,
            scale_expr=scale,
            site_id=node.node_id,
        )
        state.add_noise_draw(draw)

        noise_var = Var(ty=IRType.REAL, name=node.target.name)
        state.bind(node.target.name, noise_var)

    def _explore_branch(self, node: BranchNode, state: SymExecState) -> None:
        """Fork execution at a branch node."""
        self._stats.branches_visited += 1

        true_guard, false_guard = self._pc_manager.extract_guard(node)
        resolved_true = state.resolve_expr(true_guard)
        resolved_false = state.resolve_expr(false_guard)

        # True branch
        if self._should_explore_branch(state, resolved_true):
            true_state = state.clone()
            true_state.add_condition(resolved_true)
            self._explore_node(node.true_branch, true_state)

        # False branch
        if self._should_explore_branch(state, resolved_false):
            false_state = state.clone()
            false_state.add_condition(resolved_false)
            self._explore_node(node.false_branch, false_state)

    def _fork_branch(
        self,
        node: BranchNode,
        state: SymExecState,
        continuation_states: list[SymExecState],
    ) -> None:
        """Fork at a branch and collect continuation states (not completed paths)."""
        self._stats.branches_visited += 1

        true_guard, false_guard = self._pc_manager.extract_guard(node)
        resolved_true = state.resolve_expr(true_guard)
        resolved_false = state.resolve_expr(false_guard)

        if self._should_explore_branch(state, resolved_true):
            true_state = state.clone()
            true_state.add_condition(resolved_true)
            self._explore_subtree_for_continuation(node.true_branch, true_state, continuation_states)

        if self._should_explore_branch(state, resolved_false):
            false_state = state.clone()
            false_state.add_condition(resolved_false)
            self._explore_subtree_for_continuation(node.false_branch, false_state, continuation_states)

    def _explore_subtree_for_continuation(
        self,
        node: IRNode,
        state: SymExecState,
        continuation_states: list[SymExecState],
    ) -> None:
        """Explore a subtree, collecting states that don't end in return."""
        state.record_node(node.node_id)

        if isinstance(node, ReturnNode):
            self._explore_return(node, state)
        elif isinstance(node, SequenceNode):
            has_return = any(isinstance(s, ReturnNode) for s in node.stmts)
            if has_return:
                self._explore_sequence(node, state)
            else:
                for stmt in node.stmts:
                    self._explore_node(stmt, state)
                continuation_states.append(state)
        elif isinstance(node, NoOpNode):
            continuation_states.append(state)
        else:
            self._explore_node(node, state)
            continuation_states.append(state)

    def _explore_loop(self, node: LoopNode, state: SymExecState) -> None:
        """Unroll a loop and explore each iteration."""
        self._stats.loops_unrolled += 1

        result = self._unroller.unroll(node)
        current_state = state

        for iteration in result.iterations:
            if self._budget_exhausted:
                return

            iter_state = current_state.clone()
            resolved_cond = iter_state.resolve_expr(iteration.condition)
            iter_state.add_condition(resolved_cond)

            for var_name, expr in iteration.substitution.items():
                iter_state.bind(var_name, expr)

            if self._config.prune_infeasible:
                feas = self._feasibility.check_conjuncts(iter_state.path_condition)
                if not feas.is_feasible:
                    self._stats.paths_pruned += 1
                    continue

            self._explore_node(iteration.body, iter_state)
            current_state = iter_state

    def _explore_query(self, node: QueryNode, state: SymExecState) -> None:
        """Record a query result as a symbolic variable."""
        query_var = Var(ty=IRType.REAL, name=node.target.name)
        state.bind(node.target.name, query_var)

    def _explore_return(self, node: ReturnNode, state: SymExecState) -> None:
        """Complete a path at a return statement."""
        self._stats.paths_explored += 1

        output = state.resolve_expr(node.value)

        path = SymbolicPath(
            path_condition=PathCondition(conjuncts=list(state.path_condition)),
            noise_draws=list(state.noise_draws),
            output_expr=output,
            assignments=dict(state.bindings),
            source_nodes=list(state.source_nodes),
        )
        self._completed_paths.append(path)

    def _explore_merge(self, node: MergeNode, state: SymExecState) -> None:
        """Handle a merge (phi) node by selecting the appropriate source."""
        for pred_id, expr in node.sources.items():
            if pred_id in state.source_nodes:
                resolved = state.resolve_expr(expr)
                state.bind(node.target.name, resolved)
                return
        if node.sources:
            first_expr = next(iter(node.sources.values()))
            state.bind(node.target.name, state.resolve_expr(first_expr))

    # -- Helpers -----------------------------------------------------------

    def _make_initial_state(self, mechir: MechIR) -> SymExecState:
        """Create the initial symbolic execution state from mechanism params."""
        state = SymExecState()
        for param in mechir.params:
            param_var = Var(ty=param.ty, name=param.name)
            state.bind(param.name, param_var)
        return state

    def _should_explore_branch(
        self, state: SymExecState, guard: TypedExpr
    ) -> bool:
        """Check if a branch should be explored (not pruned).

        Returns False if the guard is a constant False or if the
        resulting path condition is infeasible.
        """
        if isinstance(guard, Const) and guard.ty == IRType.BOOL:
            return bool(guard.value)

        if self._config.prune_infeasible:
            if self._config.enable_incremental_feasibility and state.path_condition:
                feas = self._feasibility.check_incremental(
                    state.path_condition, guard,
                )
            else:
                conjuncts = state.path_condition + [guard]
                feas = self._feasibility.check_conjuncts(conjuncts)
            if not feas.is_feasible:
                self._stats.paths_pruned += 1
                return False

        return True
