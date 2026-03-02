"""Loop unrolling for bounded loops in MechIR programs.

The :class:`LoopUnroller` converts bounded ``LoopNode`` constructs into
flat sequences of per-iteration bodies.  For small bounds the loop is
fully unrolled; for large bounds a partial unrolling with a conservative
summary is produced.  Nested loops are handled recursively with a
composition budget that tracks total unrolled iterations.
"""

from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from dpcegar.ir.types import (
    BinOp,
    BinOpKind,
    Const,
    IRType,
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
    MergeNode,
    NoOpNode,
    NoiseDrawNode,
    QueryNode,
    ReturnNode,
    SequenceNode,
)
from dpcegar.paths.symbolic_path import (
    NoiseDrawInfo,
    PathCondition,
    SymbolicPath,
)

_unroll_id_counter = itertools.count()


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class UnrollConfig:
    """Configuration for the loop unroller.

    Attributes:
        max_full_unroll:     Maximum bound for full unrolling.
        max_partial_iters:   Number of iterations to unroll for partial.
        max_total_budget:    Total composition budget across all loops.
        max_nesting_depth:   Maximum nesting depth to unroll.
    """

    max_full_unroll: int = 64
    max_partial_iters: int = 8
    max_total_budget: int = 256
    max_nesting_depth: int = 4


# ═══════════════════════════════════════════════════════════════════════════
# LOOP ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class LoopAnalysis:
    """Analysis results for a single loop.

    Attributes:
        static_bound:        Concrete bound if statically known, else None.
        index_var:           Name of the loop index variable.
        has_noise_draw:      Whether the body contains a noise draw.
        has_branch:          Whether the body contains a branch.
        has_nested_loop:     Whether the body contains another loop.
        carried_deps:        Variables that are both read and written
                             across iterations (loop-carried dependencies).
        body_vars_written:   Variables written in the loop body.
        body_vars_read:      Variables read in the loop body.
    """

    static_bound: int | None = None
    index_var: str = ""
    has_noise_draw: bool = False
    has_branch: bool = False
    has_nested_loop: bool = False
    carried_deps: frozenset[str] = field(default_factory=frozenset)
    body_vars_written: frozenset[str] = field(default_factory=frozenset)
    body_vars_read: frozenset[str] = field(default_factory=frozenset)


def analyse_loop(loop: LoopNode) -> LoopAnalysis:
    """Analyse a LoopNode for unrolling decisions.

    Args:
        loop: The loop node to analyse.

    Returns:
        A :class:`LoopAnalysis` with the extracted information.
    """
    static_bound = _extract_static_bound(loop.bound)
    written: set[str] = set()
    read: set[str] = set()
    has_noise = False
    has_branch = False
    has_nested = False

    for node in loop.body.walk():
        if isinstance(node, AssignNode):
            written.add(node.target.name)
            read.update(node.value.free_vars())
        elif isinstance(node, NoiseDrawNode):
            has_noise = True
            written.add(node.target.name)
            read.update(node.center.free_vars())
            read.update(node.scale.free_vars())
        elif isinstance(node, QueryNode):
            written.add(node.target.name)
            for a in node.args:
                read.update(a.free_vars())
        elif isinstance(node, BranchNode):
            has_branch = True
            read.update(node.condition.free_vars())
        elif isinstance(node, LoopNode):
            has_nested = True
        elif isinstance(node, ReturnNode):
            read.update(node.value.free_vars())

    carried = frozenset(written & read)
    return LoopAnalysis(
        static_bound=static_bound,
        index_var=loop.index_var.name,
        has_noise_draw=has_noise,
        has_branch=has_branch,
        has_nested_loop=has_nested,
        carried_deps=carried,
        body_vars_written=frozenset(written),
        body_vars_read=frozenset(read),
    )


def _extract_static_bound(bound_expr: TypedExpr) -> int | None:
    """Try to extract a concrete integer bound from the expression.

    Returns None if the bound is symbolic or cannot be determined.
    """
    if isinstance(bound_expr, Const):
        v = bound_expr.value
        if isinstance(v, (int, float)) and v == int(v) and v >= 0:
            return int(v)
    return None


# ═══════════════════════════════════════════════════════════════════════════
# UNROLLED ITERATION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class UnrolledIteration:
    """A single unrolled loop iteration.

    Attributes:
        iteration:      The iteration index (0-based).
        index_var:      Name of the loop index variable.
        body:           The loop body node (shared reference).
        condition:      Path condition specific to this iteration.
        substitution:   Variable mapping for this iteration
                        (e.g. ``{i: Const.int_(3)}``).
    """

    iteration: int
    index_var: str
    body: IRNode
    condition: TypedExpr
    substitution: dict[str, TypedExpr] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"Iteration({self.index_var}={self.iteration})"


# ═══════════════════════════════════════════════════════════════════════════
# UNROLL RESULT
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class UnrollResult:
    """Result of loop unrolling.

    Attributes:
        iterations:      List of unrolled iterations.
        is_full:         True if the loop was fully unrolled.
        original_bound:  The original bound expression.
        unrolled_count:  Number of iterations actually unrolled.
        summary:         Optional summary for the remaining iterations
                         (when partially unrolled).
        budget_used:     Composition budget consumed.
    """

    iterations: list[UnrolledIteration] = field(default_factory=list)
    is_full: bool = True
    original_bound: TypedExpr = field(default_factory=lambda: Const.int_(0))
    unrolled_count: int = 0
    summary: LoopSummary | None = None
    budget_used: int = 0

    def __str__(self) -> str:
        mode = "full" if self.is_full else f"partial({self.unrolled_count})"
        return f"UnrollResult({mode}, budget={self.budget_used})"


@dataclass(slots=True)
class LoopSummary:
    """Conservative summary for un-unrolled loop iterations.

    Used when partial unrolling is applied.  Captures the worst-case
    effect of the remaining iterations.

    Attributes:
        remaining_iters:  Expression for the number of remaining iterations.
        noise_per_iter:   Number of noise draws per iteration.
        carried_vars:     Variables with loop-carried dependencies.
        worst_case_noise:  Maximum number of total noise draws from
                          remaining iterations.
    """

    remaining_iters: TypedExpr = field(default_factory=lambda: Const.int_(0))
    noise_per_iter: int = 0
    carried_vars: frozenset[str] = field(default_factory=frozenset)
    worst_case_noise: int = 0

    def __str__(self) -> str:
        return (
            f"LoopSummary(remaining={self.remaining_iters}, "
            f"noise/iter={self.noise_per_iter})"
        )


# ═══════════════════════════════════════════════════════════════════════════
# LOOP UNROLLER
# ═══════════════════════════════════════════════════════════════════════════


class LoopUnroller:
    """Unroll bounded loops in MechIR programs.

    The unroller converts :class:`LoopNode` constructs into flat sequences
    of per-iteration nodes.  For small bounds the loop is fully unrolled;
    for large bounds a partial unrolling with a conservative summary is
    produced.

    Args:
        config: Unrolling configuration parameters.
    """

    def __init__(self, config: UnrollConfig | None = None) -> None:
        self._config = config or UnrollConfig()
        self._budget_remaining = self._config.max_total_budget
        self._nesting_depth = 0
        self._stats = _UnrollerStats()

    # -- Public API --------------------------------------------------------

    def unroll(self, loop: LoopNode) -> UnrollResult:
        """Unroll a single loop node.

        Decides between full and partial unrolling based on the static
        bound and remaining budget.

        Args:
            loop: The loop node to unroll.

        Returns:
            An :class:`UnrollResult` describing the unrolled iterations.
        """
        self._stats.loops_processed += 1
        analysis = analyse_loop(loop)

        if self._nesting_depth >= self._config.max_nesting_depth:
            self._stats.depth_limited += 1
            return self._make_summary_only(loop, analysis)

        if analysis.static_bound is not None:
            bound = analysis.static_bound
            if bound == 0:
                return UnrollResult(
                    iterations=[],
                    is_full=True,
                    original_bound=loop.bound,
                    unrolled_count=0,
                    budget_used=0,
                )
            if bound <= self._config.max_full_unroll and bound <= self._budget_remaining:
                return self._full_unroll(loop, bound, analysis)
            else:
                n_partial = min(
                    self._config.max_partial_iters,
                    self._budget_remaining,
                    bound,
                )
                return self._partial_unroll(loop, n_partial, bound, analysis)
        else:
            n_partial = min(
                self._config.max_partial_iters,
                self._budget_remaining,
            )
            return self._partial_unroll_symbolic(loop, n_partial, analysis)

    def unroll_nested(self, loop: LoopNode) -> UnrollResult:
        """Unroll a loop that appears inside another loop body.

        Tracks nesting depth and reduces the budget proportionally.
        """
        self._nesting_depth += 1
        try:
            return self.unroll(loop)
        finally:
            self._nesting_depth -= 1

    @property
    def budget_remaining(self) -> int:
        """Return the remaining composition budget."""
        return self._budget_remaining

    @property
    def stats(self) -> dict[str, int]:
        """Return unrolling statistics."""
        return {
            "loops_processed": self._stats.loops_processed,
            "full_unrolls": self._stats.full_unrolls,
            "partial_unrolls": self._stats.partial_unrolls,
            "depth_limited": self._stats.depth_limited,
            "total_iterations": self._stats.total_iterations,
            "budget_used": self._config.max_total_budget - self._budget_remaining,
        }

    # -- Internal ----------------------------------------------------------

    def _full_unroll(
        self, loop: LoopNode, bound: int, analysis: LoopAnalysis
    ) -> UnrollResult:
        """Fully unroll a loop with a known static bound."""
        self._stats.full_unrolls += 1
        iterations: list[UnrolledIteration] = []
        idx_name = loop.index_var.name

        for i in range(bound):
            sub = {idx_name: Const.int_(i)}
            cond = BinOp(
                ty=IRType.BOOL,
                op=BinOpKind.EQ,
                left=Var(ty=IRType.INT, name=idx_name),
                right=Const.int_(i),
            )
            it = UnrolledIteration(
                iteration=i,
                index_var=idx_name,
                body=loop.body,
                condition=cond,
                substitution=sub,
            )
            iterations.append(it)

        cost = bound
        self._budget_remaining -= cost
        self._stats.total_iterations += bound

        return UnrollResult(
            iterations=iterations,
            is_full=True,
            original_bound=loop.bound,
            unrolled_count=bound,
            budget_used=cost,
        )

    def _partial_unroll(
        self,
        loop: LoopNode,
        n_iters: int,
        total_bound: int,
        analysis: LoopAnalysis,
    ) -> UnrollResult:
        """Partially unroll a loop with a known bound."""
        self._stats.partial_unrolls += 1
        iterations: list[UnrolledIteration] = []
        idx_name = loop.index_var.name

        for i in range(n_iters):
            sub = {idx_name: Const.int_(i)}
            cond = BinOp(
                ty=IRType.BOOL,
                op=BinOpKind.EQ,
                left=Var(ty=IRType.INT, name=idx_name),
                right=Const.int_(i),
            )
            it = UnrolledIteration(
                iteration=i,
                index_var=idx_name,
                body=loop.body,
                condition=cond,
                substitution=sub,
            )
            iterations.append(it)

        remaining = total_bound - n_iters
        noise_count = self._count_noise_draws(loop.body)
        summary = LoopSummary(
            remaining_iters=Const.int_(remaining),
            noise_per_iter=noise_count,
            carried_vars=analysis.carried_deps,
            worst_case_noise=remaining * noise_count,
        )

        cost = n_iters
        self._budget_remaining -= cost
        self._stats.total_iterations += n_iters

        return UnrollResult(
            iterations=iterations,
            is_full=False,
            original_bound=loop.bound,
            unrolled_count=n_iters,
            summary=summary,
            budget_used=cost,
        )

    def _partial_unroll_symbolic(
        self,
        loop: LoopNode,
        n_iters: int,
        analysis: LoopAnalysis,
    ) -> UnrollResult:
        """Partially unroll a loop with a symbolic (unknown) bound."""
        self._stats.partial_unrolls += 1
        iterations: list[UnrolledIteration] = []
        idx_name = loop.index_var.name

        for i in range(n_iters):
            sub = {idx_name: Const.int_(i)}
            guard = BinOp(
                ty=IRType.BOOL,
                op=BinOpKind.LT,
                left=Const.int_(i),
                right=loop.bound,
            )
            it = UnrolledIteration(
                iteration=i,
                index_var=idx_name,
                body=loop.body,
                condition=guard,
                substitution=sub,
            )
            iterations.append(it)

        remaining_expr = BinOp(
            ty=IRType.INT,
            op=BinOpKind.SUB,
            left=loop.bound,
            right=Const.int_(n_iters),
        )
        noise_count = self._count_noise_draws(loop.body)
        summary = LoopSummary(
            remaining_iters=remaining_expr,
            noise_per_iter=noise_count,
            carried_vars=analysis.carried_deps,
            worst_case_noise=0,
        )

        cost = n_iters
        self._budget_remaining -= cost
        self._stats.total_iterations += n_iters

        return UnrollResult(
            iterations=iterations,
            is_full=False,
            original_bound=loop.bound,
            unrolled_count=n_iters,
            summary=summary,
            budget_used=cost,
        )

    def _make_summary_only(
        self, loop: LoopNode, analysis: LoopAnalysis
    ) -> UnrollResult:
        """Create a summary-only result when nesting depth is exceeded."""
        noise_count = self._count_noise_draws(loop.body)
        bound_val = analysis.static_bound or 0
        summary = LoopSummary(
            remaining_iters=loop.bound,
            noise_per_iter=noise_count,
            carried_vars=analysis.carried_deps,
            worst_case_noise=bound_val * noise_count if bound_val else 0,
        )
        return UnrollResult(
            iterations=[],
            is_full=False,
            original_bound=loop.bound,
            unrolled_count=0,
            summary=summary,
            budget_used=0,
        )

    @staticmethod
    def _count_noise_draws(node: IRNode) -> int:
        """Count the number of noise draw nodes in a sub-tree."""
        count = 0
        for n in node.walk():
            if isinstance(n, NoiseDrawNode):
                count += 1
        return count

    def iteration_condition(self, loop: LoopNode, iteration: int) -> TypedExpr:
        """Build the path condition for a specific iteration.

        The condition asserts that the loop index equals *iteration*
        and is less than the bound.

        Args:
            loop:      The loop node.
            iteration: The 0-based iteration index.

        Returns:
            A boolean TypedExpr.
        """
        idx = Var(ty=IRType.INT, name=loop.index_var.name)
        bound_check = BinOp(
            ty=IRType.BOOL,
            op=BinOpKind.LT,
            left=Const.int_(iteration),
            right=loop.bound,
        )
        idx_check = BinOp(
            ty=IRType.BOOL,
            op=BinOpKind.EQ,
            left=idx,
            right=Const.int_(iteration),
        )
        return BinOp(
            ty=IRType.BOOL,
            op=BinOpKind.AND,
            left=bound_check,
            right=idx_check,
        )

    def carried_dependency_analysis(self, loop: LoopNode) -> dict[str, list[str]]:
        """Analyse loop-carried dependencies.

        Returns a mapping from each written variable to the list of
        variables it depends on that are also written in the loop body.

        Args:
            loop: The loop node to analyse.

        Returns:
            Dependency graph as adjacency list.
        """
        analysis = analyse_loop(loop)
        deps: dict[str, list[str]] = {}

        for node in loop.body.walk():
            if isinstance(node, AssignNode):
                target = node.target.name
                reads = node.value.free_vars()
                carried = [r for r in reads if r in analysis.body_vars_written]
                deps[target] = carried
            elif isinstance(node, NoiseDrawNode):
                target = node.target.name
                reads = node.center.free_vars() | node.scale.free_vars()
                carried = [r for r in reads if r in analysis.body_vars_written]
                deps[target] = carried

        return deps


@dataclass(slots=True)
class _UnrollerStats:
    """Internal statistics for the loop unroller."""

    loops_processed: int = 0
    full_unrolls: int = 0
    partial_unrolls: int = 0
    depth_limited: int = 0
    total_iterations: int = 0
