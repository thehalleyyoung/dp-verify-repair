"""Quick feasibility checking for symbolic paths.

Provides :class:`FeasibilityChecker` which uses interval-based analysis
and constraint propagation to quickly determine whether a path condition
is satisfiable *without* calling an SMT solver.  This is used as a
pre-filter to prune dead paths early in the enumeration.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set

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
from dpcegar.utils.math_utils import Interval
from dpcegar.paths.path_condition import (
    IntervalEnv,
    PathConditionManager,
    _eval_interval,
    _propagate_constraint,
    _canonical_key,
)
from dpcegar.paths.symbolic_path import PathCondition, SymbolicPath


# ═══════════════════════════════════════════════════════════════════════════
# FEASIBILITY RESULT
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class FeasibilityResult:
    """Result of a feasibility check.

    Attributes:
        is_feasible:  True if the condition might be satisfiable.
        reason:       Human-readable explanation (for diagnostics).
        var_bounds:   Refined variable bounds (if feasible).
    """

    is_feasible: bool
    reason: str = ""
    var_bounds: dict[str, Interval] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.is_feasible

    def __str__(self) -> str:
        status = "feasible" if self.is_feasible else "infeasible"
        return f"FeasibilityResult({status}: {self.reason})"


# ═══════════════════════════════════════════════════════════════════════════
# FEASIBILITY CHECKER
# ═══════════════════════════════════════════════════════════════════════════


class FeasibilityChecker:
    """Interval-based feasibility analyser for path conditions.

    Maintains a cache of previously checked conditions to avoid
    redundant work during path enumeration.

    Args:
        initial_bounds: Optional pre-populated variable bounds from
                        mechanism parameter declarations.
        max_cache_size: Maximum number of cached results.
    """

    def __init__(
        self,
        initial_bounds: dict[str, Interval] | None = None,
        max_cache_size: int = 10_000,
        max_prefix_depth: int = 10,
        max_prefix_cache_size: int = 128,
    ) -> None:
        self._base_env = IntervalEnv()
        if initial_bounds:
            for name, iv in initial_bounds.items():
                self._base_env.set(name, iv)
        self._cache: dict[str, FeasibilityResult] = {}
        self._max_cache: int = max_cache_size
        self._prefix_cache: OrderedDict[str, IntervalEnv] = OrderedDict()
        self._max_prefix_depth: int = max_prefix_depth
        self._max_prefix_cache_size: int = max_prefix_cache_size
        self._stats = _CheckerStats()

    # -- Public API --------------------------------------------------------

    def check(self, condition: PathCondition) -> FeasibilityResult:
        """Check feasibility of a path condition.

        Uses a layered strategy:
        1. Trivially true/false detection.
        2. Cache lookup.
        3. Syntactic contradiction detection.
        4. Interval propagation.

        Returns:
            A :class:`FeasibilityResult` indicating (in)feasibility.
        """
        self._stats.total_checks += 1

        if condition.is_trivially_true():
            return FeasibilityResult(is_feasible=True, reason="trivially true")
        if condition.is_trivially_false():
            self._stats.infeasible_count += 1
            return FeasibilityResult(is_feasible=False, reason="trivially false")

        cache_key = self._cache_key(condition)
        if cache_key in self._cache:
            self._stats.cache_hits += 1
            return self._cache[cache_key]

        result = self._check_syntactic(condition)
        if not result.is_feasible:
            self._store(cache_key, result)
            return result

        result = self._check_intervals(condition)
        self._store(cache_key, result)
        return result

    def check_conjuncts(self, conjuncts: list[TypedExpr]) -> FeasibilityResult:
        """Check feasibility of an explicit conjunct list."""
        return self.check(PathCondition(conjuncts=conjuncts))

    def check_path(self, path: SymbolicPath) -> FeasibilityResult:
        """Check feasibility of a symbolic path's condition."""
        return self.check(path.path_condition)

    def is_dead_path(self, path: SymbolicPath) -> bool:
        """Return True if the path is definitely infeasible."""
        return not self.check_path(path).is_feasible

    def eliminate_dead_paths(self, paths: list[SymbolicPath]) -> list[SymbolicPath]:
        """Filter out definitely-infeasible paths.

        Returns the list of paths whose conditions might be satisfiable.
        """
        alive: list[SymbolicPath] = []
        for p in paths:
            result = self.check_path(p)
            if result.is_feasible:
                alive.append(p)
            else:
                self._stats.eliminated_count += 1
        return alive

    def check_incremental(
        self,
        base_conjuncts: list[TypedExpr],
        new_conjunct: TypedExpr,
    ) -> FeasibilityResult:
        """Check feasibility of base_conjuncts ∧ new_conjunct.

        Reuses the interval environment from the base to avoid
        re-propagating all base constraints from scratch.  Cached
        prefix environments are stored in an LRU cache bounded by
        ``max_prefix_depth`` and ``max_prefix_cache_size``.
        """
        self._stats.total_checks += 1

        prefix_key = self._prefix_key(base_conjuncts)
        cached_env = self._prefix_cache_get(prefix_key)

        if cached_env is not None:
            env = cached_env.clone()
        else:
            env = self._base_env.clone()
            for c in base_conjuncts:
                if not _propagate_constraint(c, env, True):
                    self._stats.infeasible_count += 1
                    return FeasibilityResult(is_feasible=False, reason="base infeasible")
            if len(base_conjuncts) <= self._max_prefix_depth:
                self._prefix_cache_put(prefix_key, env.clone())

        if not _propagate_constraint(new_conjunct, env, True):
            self._stats.infeasible_count += 1
            return FeasibilityResult(
                is_feasible=False,
                reason="new conjunct conflicts with base",
            )

        return FeasibilityResult(
            is_feasible=True,
            reason="intervals consistent",
            var_bounds=dict(env.bounds),
        )

    # -- Statistics --------------------------------------------------------

    @property
    def stats(self) -> dict[str, int]:
        """Return checker statistics."""
        return {
            "total_checks": self._stats.total_checks,
            "cache_hits": self._stats.cache_hits,
            "infeasible_count": self._stats.infeasible_count,
            "eliminated_count": self._stats.eliminated_count,
            "cache_size": len(self._cache),
            "prefix_cache_size": len(self._prefix_cache),
        }

    def reset_stats(self) -> None:
        """Reset checker statistics."""
        self._stats = _CheckerStats()

    def clear_cache(self) -> None:
        """Clear the feasibility cache and prefix cache."""
        self._cache.clear()
        self._prefix_cache.clear()

    # -- Internal ----------------------------------------------------------

    def _check_syntactic(self, condition: PathCondition) -> FeasibilityResult:
        """Detect simple syntactic contradictions."""
        pos_keys: set[str] = set()
        neg_keys: set[str] = set()

        for c in condition.conjuncts:
            if isinstance(c, Const) and c.ty == IRType.BOOL:
                if c.value is False or c.value == 0:
                    self._stats.infeasible_count += 1
                    return FeasibilityResult(
                        is_feasible=False, reason="contains false constant"
                    )
                continue

            key = _canonical_key(c)
            if isinstance(c, UnaryOp) and c.op == UnaryOpKind.NOT:
                inner_key = _canonical_key(c.operand)
                if inner_key in pos_keys:
                    self._stats.infeasible_count += 1
                    return FeasibilityResult(
                        is_feasible=False,
                        reason=f"contradiction: {c.operand} ∧ ¬{c.operand}",
                    )
                neg_keys.add(inner_key)
            else:
                if key in neg_keys:
                    self._stats.infeasible_count += 1
                    return FeasibilityResult(
                        is_feasible=False,
                        reason=f"contradiction: {c} ∧ ¬{c}",
                    )
                pos_keys.add(key)

        # Check for conflicting comparisons: x < c1 ∧ x > c2 where c1 <= c2
        comp_bounds = self._extract_comparison_bounds(condition.conjuncts)
        for var_name, (lo, hi) in comp_bounds.items():
            if lo is not None and hi is not None and lo > hi:
                self._stats.infeasible_count += 1
                return FeasibilityResult(
                    is_feasible=False,
                    reason=f"conflicting bounds on {var_name}: [{lo}, {hi}]",
                )

        return FeasibilityResult(is_feasible=True, reason="no syntactic contradiction")

    def _check_intervals(self, condition: PathCondition) -> FeasibilityResult:
        """Check feasibility via interval constraint propagation."""
        env = self._base_env.clone()

        for c in condition.conjuncts:
            if not _propagate_constraint(c, env, True):
                self._stats.infeasible_count += 1
                return FeasibilityResult(
                    is_feasible=False,
                    reason="interval propagation detected infeasibility",
                )

        for name, iv in env.bounds.items():
            if iv.lo > iv.hi:
                self._stats.infeasible_count += 1
                return FeasibilityResult(
                    is_feasible=False,
                    reason=f"empty interval for {name}: {iv}",
                )

        return FeasibilityResult(
            is_feasible=True,
            reason="intervals consistent",
            var_bounds=dict(env.bounds),
        )

    @staticmethod
    def _extract_comparison_bounds(
        conjuncts: list[TypedExpr],
    ) -> dict[str, tuple[float | None, float | None]]:
        """Extract upper/lower bounds from simple var-vs-const comparisons."""
        bounds: dict[str, list[float | None]] = {}

        def _ensure(name: str) -> None:
            if name not in bounds:
                bounds[name] = [None, None]  # [lo, hi]

        for c in conjuncts:
            if not isinstance(c, BinOp) or not c.op.is_comparison:
                continue
            if isinstance(c.left, Var) and isinstance(c.right, Const):
                var_name = c.left.name
                val = float(c.right.value) if not isinstance(c.right.value, bool) else float(c.right.value)
                _ensure(var_name)
                if c.op in (BinOpKind.LT, BinOpKind.LE):
                    current_hi = bounds[var_name][1]
                    if current_hi is None or val < current_hi:
                        bounds[var_name][1] = val
                elif c.op in (BinOpKind.GT, BinOpKind.GE):
                    current_lo = bounds[var_name][0]
                    if current_lo is None or val > current_lo:
                        bounds[var_name][0] = val

        result: dict[str, tuple[float | None, float | None]] = {}
        for name, (lo, hi) in bounds.items():
            result[name] = (lo, hi)
        return result

    @staticmethod
    def _cache_key(condition: PathCondition) -> str:
        """Compute a cache key from the conjuncts."""
        keys = sorted(_canonical_key(c) for c in condition.conjuncts)
        return "|".join(keys)

    @staticmethod
    def _prefix_key(conjuncts: list[TypedExpr]) -> str:
        """Compute a cache key for a prefix of conjuncts."""
        keys = [_canonical_key(c) for c in conjuncts]
        return "|".join(keys)

    def _prefix_cache_get(self, key: str) -> IntervalEnv | None:
        """Look up *key* in the prefix cache, promoting on hit (LRU)."""
        if key in self._prefix_cache:
            self._prefix_cache.move_to_end(key)
            return self._prefix_cache[key]
        return None

    def _prefix_cache_put(self, key: str, env: IntervalEnv) -> None:
        """Insert into the prefix cache, evicting LRU if full."""
        if key in self._prefix_cache:
            self._prefix_cache.move_to_end(key)
            self._prefix_cache[key] = env
            return
        if len(self._prefix_cache) >= self._max_prefix_cache_size:
            self._prefix_cache.popitem(last=False)
        self._prefix_cache[key] = env

    def _store(self, cache_key: str, result: FeasibilityResult) -> None:
        """Store a result in the cache, evicting old entries if needed."""
        if len(self._cache) >= self._max_cache:
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        self._cache[cache_key] = result


@dataclass(slots=True)
class _CheckerStats:
    """Internal statistics for the feasibility checker."""

    total_checks: int = 0
    cache_hits: int = 0
    infeasible_count: int = 0
    eliminated_count: int = 0
