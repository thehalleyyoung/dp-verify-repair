"""Path condition management for the path enumeration engine.

Provides :class:`PathConditionManager` which handles conjunction,
disjunction, negation, simplification, satisfiability quick-checks,
implication testing, substitution, canonical form normalisation, and
guard extraction from branch nodes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

from dpcegar.ir.types import (
    Abs,
    BinOp,
    BinOpKind,
    Const,
    IRType,
    TypedExpr,
    UnaryOp,
    UnaryOpKind,
    Var,
)
from dpcegar.ir.nodes import BranchNode, IRNode
from dpcegar.utils.math_utils import Interval


# ═══════════════════════════════════════════════════════════════════════════
# INTERVAL ENVIRONMENT
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class IntervalEnv:
    """Maps variable names to their known interval bounds.

    Used for quick satisfiability checking via interval propagation.
    """

    bounds: dict[str, Interval] = field(default_factory=dict)

    def get(self, name: str) -> Interval:
        """Return the interval for *name*, defaulting to (-∞, ∞)."""
        return self.bounds.get(name, Interval.entire())

    def set(self, name: str, iv: Interval) -> None:
        """Update the interval for *name*."""
        self.bounds[name] = iv

    def refine(self, name: str, iv: Interval) -> bool:
        """Intersect *iv* with the current interval for *name*.

        Returns False if the intersection is empty (infeasible).
        """
        current = self.get(name)
        result = current.intersect(iv)
        if result is None:
            return False
        self.bounds[name] = result
        return True

    def clone(self) -> IntervalEnv:
        """Return a deep copy."""
        return IntervalEnv(bounds=dict(self.bounds))


# ═══════════════════════════════════════════════════════════════════════════
# EXPRESSION EVALUATOR (interval)
# ═══════════════════════════════════════════════════════════════════════════

def _eval_interval(expr: TypedExpr, env: IntervalEnv) -> Interval:
    """Evaluate a TypedExpr to an Interval using the given environment.

    Returns a conservative over-approximation of all possible values.
    """
    if isinstance(expr, Const):
        v = float(expr.value) if not isinstance(expr.value, bool) else (1.0 if expr.value else 0.0)
        return Interval.point(v)

    if isinstance(expr, Var):
        return env.get(expr.name)

    if isinstance(expr, BinOp):
        left = _eval_interval(expr.left, env)
        right = _eval_interval(expr.right, env)
        if expr.op == BinOpKind.ADD:
            return left + right
        if expr.op == BinOpKind.SUB:
            return left - right
        if expr.op == BinOpKind.MUL:
            return left * right
        if expr.op == BinOpKind.DIV:
            return left / right
        if expr.op == BinOpKind.POW:
            if right.is_point and right.lo == int(right.lo) and right.lo >= 0:
                return left ** int(right.lo)
            return Interval.entire()
        return Interval.entire()

    if isinstance(expr, UnaryOp):
        operand = _eval_interval(expr.operand, env)
        if expr.op == UnaryOpKind.NEG:
            return -operand
        return Interval.entire()

    if isinstance(expr, Abs):
        inner = _eval_interval(expr.operand, env)
        if inner.lo >= 0:
            return inner
        if inner.hi <= 0:
            return -inner
        return Interval(0.0, max(abs(inner.lo), abs(inner.hi)))

    return Interval.entire()


# ═══════════════════════════════════════════════════════════════════════════
# CONSTRAINT PROPAGATION
# ═══════════════════════════════════════════════════════════════════════════

def _propagate_constraint(expr: TypedExpr, env: IntervalEnv, assume_true: bool) -> bool:
    """Propagate a boolean constraint into the interval environment.

    Returns False if a contradiction is detected (infeasible).
    """
    if isinstance(expr, Const) and expr.ty == IRType.BOOL:
        val = bool(expr.value)
        if val == assume_true:
            return True
        return False

    if isinstance(expr, UnaryOp) and expr.op == UnaryOpKind.NOT:
        return _propagate_constraint(expr.operand, env, not assume_true)

    if isinstance(expr, BinOp):
        if expr.op == BinOpKind.AND:
            if assume_true:
                return (_propagate_constraint(expr.left, env, True)
                        and _propagate_constraint(expr.right, env, True))
            return True  # can't propagate OR negations tightly

        if expr.op == BinOpKind.OR:
            if not assume_true:
                return (_propagate_constraint(expr.left, env, False)
                        and _propagate_constraint(expr.right, env, False))
            return True

        if expr.op.is_comparison:
            return _propagate_comparison(expr, env, assume_true)

    return True  # conservative: can't refine


def _propagate_comparison(expr: BinOp, env: IntervalEnv, assume_true: bool) -> bool:
    """Refine intervals for simple variable-vs-constant comparisons.

    Handles patterns like ``x < 5``, ``x >= y``, ``x == 3``.
    """
    left = expr.left
    right = expr.right
    op = expr.op

    if not assume_true:
        negated = _negate_comparison(op)
        if negated is not None:
            op = negated
        else:
            return True

    left_iv = _eval_interval(left, env)
    right_iv = _eval_interval(right, env)

    if isinstance(left, Var) and isinstance(right, Const):
        val = float(right.value) if not isinstance(right.value, bool) else float(right.value)
        return _refine_var_const(left.name, op, val, env)

    if isinstance(right, Var) and isinstance(left, Const):
        val = float(left.value) if not isinstance(left.value, bool) else float(left.value)
        flipped = _flip_comparison(op)
        if flipped is not None:
            return _refine_var_const(right.name, flipped, val, env)

    if op == BinOpKind.LT:
        if left_iv.lo >= right_iv.hi:
            return False
    elif op == BinOpKind.LE:
        if left_iv.lo > right_iv.hi:
            return False
    elif op == BinOpKind.GT:
        if left_iv.hi <= right_iv.lo:
            return False
    elif op == BinOpKind.GE:
        if left_iv.hi < right_iv.lo:
            return False
    elif op == BinOpKind.EQ:
        overlap = left_iv.intersect(right_iv)
        if overlap is None:
            return False

    return True


def _refine_var_const(var_name: str, op: BinOpKind, val: float, env: IntervalEnv) -> bool:
    """Refine the interval for *var_name* given ``var op val``."""
    if op == BinOpKind.LT:
        return env.refine(var_name, Interval(-math.inf, val - 1e-15))
    if op == BinOpKind.LE:
        return env.refine(var_name, Interval(-math.inf, val))
    if op == BinOpKind.GT:
        return env.refine(var_name, Interval(val + 1e-15, math.inf))
    if op == BinOpKind.GE:
        return env.refine(var_name, Interval(val, math.inf))
    if op == BinOpKind.EQ:
        return env.refine(var_name, Interval(val, val))
    if op == BinOpKind.NEQ:
        current = env.get(var_name)
        if current.is_point and current.lo == val:
            return False
        return True
    return True


def _negate_comparison(op: BinOpKind) -> BinOpKind | None:
    """Return the negation of a comparison operator."""
    _neg_map = {
        BinOpKind.LT: BinOpKind.GE,
        BinOpKind.LE: BinOpKind.GT,
        BinOpKind.GT: BinOpKind.LE,
        BinOpKind.GE: BinOpKind.LT,
        BinOpKind.EQ: BinOpKind.NEQ,
        BinOpKind.NEQ: BinOpKind.EQ,
    }
    return _neg_map.get(op)


def _flip_comparison(op: BinOpKind) -> BinOpKind | None:
    """Flip a comparison for when operands are swapped: ``c op x`` → ``x op' c``."""
    _flip_map = {
        BinOpKind.LT: BinOpKind.GT,
        BinOpKind.LE: BinOpKind.GE,
        BinOpKind.GT: BinOpKind.LT,
        BinOpKind.GE: BinOpKind.LE,
        BinOpKind.EQ: BinOpKind.EQ,
        BinOpKind.NEQ: BinOpKind.NEQ,
    }
    return _flip_map.get(op)


# ═══════════════════════════════════════════════════════════════════════════
# CANONICAL FORM
# ═══════════════════════════════════════════════════════════════════════════

def _canonical_key(expr: TypedExpr) -> str:
    """Compute a canonical string key for an expression.

    Orders commutative operands lexicographically so that
    ``x + y`` and ``y + x`` receive the same key.
    """
    if isinstance(expr, Const):
        return f"C({expr.value})"
    if isinstance(expr, Var):
        return f"V({expr.ssa_name})"
    if isinstance(expr, BinOp):
        lk = _canonical_key(expr.left)
        rk = _canonical_key(expr.right)
        if expr.op in (BinOpKind.ADD, BinOpKind.MUL, BinOpKind.AND, BinOpKind.OR, BinOpKind.EQ, BinOpKind.NEQ):
            lk, rk = min(lk, rk), max(lk, rk)
        return f"B({expr.op.value},{lk},{rk})"
    if isinstance(expr, UnaryOp):
        return f"U({expr.op.value},{_canonical_key(expr.operand)})"
    return str(expr)


def _normalize_expr(expr: TypedExpr) -> TypedExpr:
    """Normalize an expression to canonical form.

    - Sorts operands of commutative ops lexicographically.
    - Eliminates double negations.
    - Folds constants.
    """
    if isinstance(expr, UnaryOp) and expr.op == UnaryOpKind.NOT:
        inner = _normalize_expr(expr.operand)
        if isinstance(inner, UnaryOp) and inner.op == UnaryOpKind.NOT:
            return inner.operand
        if isinstance(inner, Const) and inner.ty == IRType.BOOL:
            return Const.bool_(not inner.value)
        return UnaryOp(ty=IRType.BOOL, op=UnaryOpKind.NOT, operand=inner)

    if isinstance(expr, BinOp):
        left = _normalize_expr(expr.left)
        right = _normalize_expr(expr.right)
        if expr.op in (BinOpKind.ADD, BinOpKind.MUL, BinOpKind.AND, BinOpKind.OR):
            lk = _canonical_key(left)
            rk = _canonical_key(right)
            if lk > rk:
                left, right = right, left
        return expr.simplify()

    return expr.simplify()


# ═══════════════════════════════════════════════════════════════════════════
# PATH CONDITION MANAGER
# ═══════════════════════════════════════════════════════════════════════════


class PathConditionManager:
    """Manager for creating, combining, simplifying, and checking path conditions.

    Provides a higher-level interface over raw :class:`PathCondition`
    objects, adding interval-based satisfiability checks and canonical
    form normalisation.
    """

    def __init__(self, initial_env: IntervalEnv | None = None) -> None:
        """Initialise the manager.

        Args:
            initial_env: Optional pre-populated interval environment for
                         known variable bounds (e.g. from mechanism params).
        """
        self._base_env = initial_env or IntervalEnv()
        self._cache: dict[str, bool] = {}

    # -- Construction ------------------------------------------------------

    @staticmethod
    def make_true() -> list[TypedExpr]:
        """Return an empty conjunct list (trivially true)."""
        return []

    @staticmethod
    def make_from_expr(expr: TypedExpr) -> list[TypedExpr]:
        """Wrap a single boolean expression as a conjunct list."""
        return [expr]

    # -- Combination -------------------------------------------------------

    @staticmethod
    def conjunction(left: list[TypedExpr], right: list[TypedExpr]) -> list[TypedExpr]:
        """Return the conjunction of two conjunct lists."""
        return left + right

    @staticmethod
    def disjunction_expr(left: TypedExpr, right: TypedExpr) -> TypedExpr:
        """Build a disjunction expression."""
        if isinstance(left, Const) and left.ty == IRType.BOOL and left.value:
            return Const.bool_(True)
        if isinstance(right, Const) and right.ty == IRType.BOOL and right.value:
            return Const.bool_(True)
        if isinstance(left, Const) and left.ty == IRType.BOOL and not left.value:
            return right
        if isinstance(right, Const) and right.ty == IRType.BOOL and not right.value:
            return left
        return BinOp(ty=IRType.BOOL, op=BinOpKind.OR, left=left, right=right)

    @staticmethod
    def negation(expr: TypedExpr) -> TypedExpr:
        """Return the logical negation of *expr* with simplification."""
        if isinstance(expr, Const) and expr.ty == IRType.BOOL:
            return Const.bool_(not expr.value)
        if isinstance(expr, UnaryOp) and expr.op == UnaryOpKind.NOT:
            return expr.operand
        if isinstance(expr, BinOp) and expr.op.is_comparison:
            neg_op = _negate_comparison(expr.op)
            if neg_op is not None:
                return BinOp(ty=IRType.BOOL, op=neg_op, left=expr.left, right=expr.right)
        return UnaryOp(ty=IRType.BOOL, op=UnaryOpKind.NOT, operand=expr)

    # -- Simplification ----------------------------------------------------

    def simplify(self, conjuncts: list[TypedExpr]) -> list[TypedExpr]:
        """Simplify a conjunct list by algebraic rules and deduplication.

        - Simplifies each conjunct individually.
        - Drops trivially-true conjuncts.
        - Detects trivially-false conjuncts.
        - Removes duplicate conjuncts (canonical key comparison).
        """
        simplified: list[TypedExpr] = []
        seen_keys: set[str] = set()

        for c in conjuncts:
            s = _normalize_expr(c)
            if isinstance(s, Const) and s.ty == IRType.BOOL:
                if s.value is True or s.value == 1:
                    continue
                if s.value is False or s.value == 0:
                    return [Const.bool_(False)]
            key = _canonical_key(s)
            if key not in seen_keys:
                seen_keys.add(key)
                simplified.append(s)

        return simplified

    # -- Satisfiability ----------------------------------------------------

    def is_satisfiable(self, conjuncts: list[TypedExpr]) -> bool:
        """Quick satisfiability check via interval propagation.

        Returns False if the conjunction is definitely unsatisfiable.
        Returns True if it might be satisfiable (conservative).
        """
        cache_key = ";".join(_canonical_key(c) for c in conjuncts)
        if cache_key in self._cache:
            return self._cache[cache_key]

        env = self._base_env.clone()
        for c in conjuncts:
            if isinstance(c, Const) and c.ty == IRType.BOOL:
                if c.value is False or c.value == 0:
                    self._cache[cache_key] = False
                    return False
                continue
            if not _propagate_constraint(c, env, True):
                self._cache[cache_key] = False
                return False

        self._cache[cache_key] = True
        return True

    def is_unsatisfiable(self, conjuncts: list[TypedExpr]) -> bool:
        """Return True if the conjunction is definitely unsatisfiable."""
        return not self.is_satisfiable(conjuncts)

    # -- Implication -------------------------------------------------------

    def implies(self, premises: list[TypedExpr], conclusion: list[TypedExpr]) -> bool:
        """Check whether *premises* implies *conclusion*.

        Uses two strategies:
        1. Syntactic: every conjunct of conclusion appears in premises.
        2. Interval: assuming premises, check if conclusion is satisfied.
        """
        premise_keys = {_canonical_key(c) for c in premises}
        conclusion_keys = {_canonical_key(c) for c in conclusion}
        if conclusion_keys.issubset(premise_keys):
            return True

        env = self._base_env.clone()
        for c in premises:
            _propagate_constraint(c, env, True)

        for c in conclusion:
            if isinstance(c, BinOp) and c.op.is_comparison:
                left_iv = _eval_interval(c.left, env)
                right_iv = _eval_interval(c.right, env)
                if not _interval_satisfies_comparison(c.op, left_iv, right_iv):
                    return False
            elif isinstance(c, Const) and c.ty == IRType.BOOL:
                if not c.value:
                    return False

        return True

    # -- Substitution ------------------------------------------------------

    @staticmethod
    def substitute(
        conjuncts: list[TypedExpr], mapping: dict[str, TypedExpr]
    ) -> list[TypedExpr]:
        """Apply variable substitution to every conjunct."""
        return [c.substitute(mapping) for c in conjuncts]

    # -- Canonical form ----------------------------------------------------

    def canonicalize(self, conjuncts: list[TypedExpr]) -> list[TypedExpr]:
        """Normalize and sort conjuncts into canonical form.

        Canonical form enables structural comparison of path conditions.
        """
        normalized = [_normalize_expr(c) for c in conjuncts]
        normalized.sort(key=_canonical_key)
        return self.simplify(normalized)

    # -- Guard extraction --------------------------------------------------

    @staticmethod
    def extract_guard(branch: BranchNode) -> tuple[TypedExpr, TypedExpr]:
        """Extract the true-branch and false-branch guards from a BranchNode.

        Returns:
            A tuple ``(true_guard, false_guard)`` where false_guard is
            the negation of the branch condition.
        """
        cond = branch.condition
        true_guard = cond
        if isinstance(cond, UnaryOp) and cond.op == UnaryOpKind.NOT:
            false_guard = cond.operand
        elif isinstance(cond, BinOp) and cond.op.is_comparison:
            neg_op = _negate_comparison(cond.op)
            if neg_op is not None:
                false_guard = BinOp(ty=IRType.BOOL, op=neg_op, left=cond.left, right=cond.right)
            else:
                false_guard = UnaryOp(ty=IRType.BOOL, op=UnaryOpKind.NOT, operand=cond)
        elif isinstance(cond, Const) and cond.ty == IRType.BOOL:
            false_guard = Const.bool_(not cond.value)
        else:
            false_guard = UnaryOp(ty=IRType.BOOL, op=UnaryOpKind.NOT, operand=cond)
        return true_guard, false_guard

    def extract_var_bounds(self, conjuncts: list[TypedExpr]) -> dict[str, Interval]:
        """Extract known variable bounds from a conjunct list.

        Propagates all constraints through interval arithmetic and returns
        the resulting variable-to-interval mapping.
        """
        env = self._base_env.clone()
        for c in conjuncts:
            _propagate_constraint(c, env, True)
        return dict(env.bounds)

    def clear_cache(self) -> None:
        """Clear the satisfiability cache."""
        self._cache.clear()


# ═══════════════════════════════════════════════════════════════════════════
# HELPER
# ═══════════════════════════════════════════════════════════════════════════

def _interval_satisfies_comparison(
    op: BinOpKind, left: Interval, right: Interval
) -> bool:
    """Check whether a comparison is guaranteed to hold for all values."""
    if op == BinOpKind.LT:
        return left.hi < right.lo
    if op == BinOpKind.LE:
        return left.hi <= right.lo
    if op == BinOpKind.GT:
        return left.lo > right.hi
    if op == BinOpKind.GE:
        return left.lo >= right.hi
    if op == BinOpKind.EQ:
        return left.is_point and right.is_point and left.lo == right.lo
    if op == BinOpKind.NEQ:
        return left.hi < right.lo or left.lo > right.hi
    return False
