"""Core SMT encoding for differential privacy verification.

Translates MechIR typed expressions into Z3 formulas, handling all
expression types (arithmetic, comparison, logical, transcendental)
with type-aware encoding (int → z3.Int, real → z3.Real).

The central class :class:`ExprToZ3` implements the :class:`ExprVisitor`
pattern from our IR to produce ``z3.ExprRef`` values.  Auxiliary
classes handle absolute-value linearisation, case splitting, and
constraint construction for privacy predicates.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

try:
    import z3
except ImportError:  # pragma: no cover
    z3 = None  # type: ignore[assignment]

from dpcegar.ir.types import (
    Abs,
    BinOp,
    BinOpKind,
    Cond,
    Const,
    Exp,
    ExprVisitor,
    FuncCall,
    IRType,
    LetExpr,
    Log,
    Max,
    Min,
    Phi,
    PhiInv,
    Sqrt,
    SumExpr,
    TypedExpr,
    UnaryOp,
    UnaryOpKind,
    Var,
    ArrayAccess,
    TupleAccess,
    PrivacyNotion,
)
from dpcegar.paths.symbolic_path import PathCondition, SymbolicPath
from dpcegar.smt.transcendental import (
    ApproxResult,
    Precision,
    SoundnessTracker,
    TranscendentalApprox,
)
from dpcegar.smt.theory_selection import (
    SMTTheory,
    TheoryAnalyzer,
    TheoryAnalysisResult,
)


# ═══════════════════════════════════════════════════════════════════════════
# SMT ENCODING DATACLASS
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class SMTEncoding:
    """A complete SMT encoding ready for solving.

    Attributes:
        formula:      The conjunction of all assertions as a Z3 expression.
        variables:    Map from variable name to Z3 variable.
        assertions:   Individual assertion expressions.
        metadata:     Encoding metadata (theory, approximations, etc.).
        aux_vars:     Auxiliary variables introduced during encoding.
        soundness:    Soundness tracker for approximations used.
    """

    formula: Any = None  # z3.BoolRef
    variables: dict[str, Any] = field(default_factory=dict)
    assertions: list[Any] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    aux_vars: dict[str, Any] = field(default_factory=dict)
    soundness: SoundnessTracker = field(default_factory=SoundnessTracker)

    def add_assertion(self, assertion: Any) -> None:
        """Add a single assertion to the encoding.

        Args:
            assertion: Z3 boolean expression.
        """
        self.assertions.append(assertion)
        self._rebuild_formula()

    def add_assertions(self, assertions: Sequence[Any]) -> None:
        """Add multiple assertions to the encoding.

        Args:
            assertions: Sequence of Z3 boolean expressions.
        """
        self.assertions.extend(assertions)
        self._rebuild_formula()

    def _rebuild_formula(self) -> None:
        """Rebuild the conjunction formula from the assertion list."""
        if not self.assertions:
            self.formula = z3.BoolVal(True)
        elif len(self.assertions) == 1:
            self.formula = self.assertions[0]
        else:
            self.formula = z3.And(*self.assertions)

    def merge(self, other: SMTEncoding) -> SMTEncoding:
        """Merge another encoding into this one.

        Args:
            other: Another SMTEncoding to merge.

        Returns:
            Self, for chaining.
        """
        self.variables.update(other.variables)
        self.assertions.extend(other.assertions)
        self.aux_vars.update(other.aux_vars)
        for entry in other.soundness.entries:
            self.soundness.entries.append(entry)
        self.metadata.update(other.metadata)
        self._rebuild_formula()
        return self

    def variable_count(self) -> int:
        """Return the number of variables in this encoding.

        Returns:
            Total count of regular + auxiliary variables.
        """
        return len(self.variables) + len(self.aux_vars)

    def assertion_count(self) -> int:
        """Return the number of assertions.

        Returns:
            Number of assertions.
        """
        return len(self.assertions)

    def summary(self) -> dict[str, Any]:
        """Return a summary of the encoding.

        Returns:
            Dictionary with encoding statistics.
        """
        return {
            "variables": len(self.variables),
            "aux_vars": len(self.aux_vars),
            "assertions": len(self.assertions),
            "sound": self.soundness.all_sound,
            "metadata": dict(self.metadata),
        }


# ═══════════════════════════════════════════════════════════════════════════
# AUXILIARY VARIABLE COUNTER
# ═══════════════════════════════════════════════════════════════════════════

_aux_counter = itertools.count()


def _fresh_aux(prefix: str = "aux") -> str:
    """Generate a fresh auxiliary variable name.

    Args:
        prefix: Name prefix.

    Returns:
        Unique auxiliary variable name.
    """
    return f"__{prefix}_{next(_aux_counter)}"


# ═══════════════════════════════════════════════════════════════════════════
# EXPRESSION TO Z3 CONVERTER
# ═══════════════════════════════════════════════════════════════════════════


class ExprToZ3(ExprVisitor[Any]):
    """Convert MechIR TypedExpr nodes to Z3 expressions.

    Implements the ExprVisitor pattern: each ``visit_*`` method converts
    the corresponding IR node into a Z3 expression.  Variables are
    created lazily and cached in the ``variables`` dict.

    Handles:
      - Var:        z3.Int or z3.Real based on IRType
      - Const:      z3.IntVal, z3.RealVal, z3.BoolVal
      - BinOp:      all arithmetic, comparison, and logical ops
      - UnaryOp:    negation and logical not
      - Abs:        |x| via z3.If(x >= 0, x, -x)
      - Max/Min:    via z3.If
      - Log/Exp:    via TranscendentalApprox
      - Sqrt:       via TranscendentalApprox
      - Phi/PhiInv: via polynomial approximation
      - Cond:       z3.If
      - LetExpr:    substitution
      - SumExpr:    unrolled finite sum

    Args:
        precision:     Precision level for transcendental approximations.
        int_vars:      Set of variable names to encode as z3.Int.
        var_overrides: Pre-existing Z3 variables to use.
    """

    def __init__(
        self,
        precision: Precision = Precision.STANDARD,
        int_vars: set[str] | None = None,
        var_overrides: dict[str, Any] | None = None,
    ) -> None:
        self._precision = precision
        self._int_vars = int_vars or set()
        self._variables: dict[str, Any] = dict(var_overrides or {})
        self._aux_vars: dict[str, Any] = {}
        self._aux_constraints: list[Any] = []
        self._tracker = SoundnessTracker()
        self._transcendental = TranscendentalApprox(precision, self._tracker)
        self._let_env: dict[str, Any] = {}

    @property
    def variables(self) -> dict[str, Any]:
        """Return all Z3 variables created during conversion.

        Returns:
            Dictionary mapping variable names to Z3 constants.
        """
        return dict(self._variables)

    @property
    def aux_vars(self) -> dict[str, Any]:
        """Return auxiliary Z3 variables introduced during conversion.

        Returns:
            Dictionary mapping aux var names to Z3 constants.
        """
        return dict(self._aux_vars)

    @property
    def aux_constraints(self) -> list[Any]:
        """Return auxiliary constraints (e.g. for abs linearisation).

        Returns:
            List of Z3 boolean expressions.
        """
        return list(self._aux_constraints)

    @property
    def tracker(self) -> SoundnessTracker:
        """Return the soundness tracker.

        Returns:
            SoundnessTracker instance.
        """
        return self._tracker

    def convert(self, expr: TypedExpr) -> Any:
        """Convert a TypedExpr to a Z3 expression.

        This is the main entry point.  Calls the visitor dispatch.

        Args:
            expr: The IR expression to convert.

        Returns:
            Z3 expression (z3.ExprRef).
        """
        return self.visit(expr)

    def convert_bool(self, expr: TypedExpr) -> Any:
        """Convert a TypedExpr expected to be boolean.

        Wraps non-boolean results with a != 0 check.

        Args:
            expr: The IR expression to convert.

        Returns:
            Z3 boolean expression.
        """
        result = self.visit(expr)
        if z3.is_bool(result):
            return result
        return result != z3.RealVal(0)

    def _get_or_create_var(self, name: str, ty: IRType) -> Any:
        """Get an existing Z3 variable or create a new one.

        Args:
            name: Variable name.
            ty:   IR type.

        Returns:
            Z3 constant (Int, Real, or Bool).
        """
        if name in self._let_env:
            return self._let_env[name]
        if name in self._variables:
            return self._variables[name]
        if ty == IRType.INT or name in self._int_vars:
            var = z3.Int(name)
        elif ty == IRType.BOOL:
            var = z3.Bool(name)
        else:
            var = z3.Real(name)
        self._variables[name] = var
        return var

    def _make_aux_real(self, prefix: str = "aux") -> Any:
        """Create a fresh auxiliary real variable.

        Args:
            prefix: Name prefix.

        Returns:
            Z3 Real constant.
        """
        name = _fresh_aux(prefix)
        var = z3.Real(name)
        self._aux_vars[name] = var
        return var

    # ── Leaf visitors ────────────────────────────────────────────────────

    def visit_Var(self, expr: Var) -> Any:
        """Convert a Var to a Z3 constant.

        Args:
            expr: IR Var node.

        Returns:
            Z3 constant.
        """
        return self._get_or_create_var(expr.ssa_name, expr.ty)

    def visit_Const(self, expr: Const) -> Any:
        """Convert a Const to a Z3 value.

        Args:
            expr: IR Const node.

        Returns:
            Z3 value (IntVal, RealVal, or BoolVal).
        """
        if expr.ty == IRType.BOOL:
            return z3.BoolVal(bool(expr.value))
        if expr.ty == IRType.INT:
            return z3.IntVal(int(expr.value))
        return z3.RealVal(str(float(expr.value)))

    # ── Binary operations ────────────────────────────────────────────────

    def visit_BinOp(self, expr: BinOp) -> Any:
        """Convert a BinOp to a Z3 expression.

        Handles arithmetic (+, -, *, /, %, **), comparison (==, !=, <, >,
        <=, >=), and logical (&&, ||) operators.

        Args:
            expr: IR BinOp node.

        Returns:
            Z3 expression.
        """
        left = self.visit(expr.left)
        right = self.visit(expr.right)

        op_map = {
            BinOpKind.ADD: lambda l, r: l + r,
            BinOpKind.SUB: lambda l, r: l - r,
            BinOpKind.MUL: lambda l, r: l * r,
            BinOpKind.EQ: lambda l, r: l == r,
            BinOpKind.NEQ: lambda l, r: l != r,
            BinOpKind.LT: lambda l, r: l < r,
            BinOpKind.LE: lambda l, r: l <= r,
            BinOpKind.GT: lambda l, r: l > r,
            BinOpKind.GE: lambda l, r: l >= r,
        }

        if expr.op in op_map:
            return op_map[expr.op](left, right)

        if expr.op == BinOpKind.DIV:
            # Use real division
            if z3.is_int(left) and z3.is_int(right):
                return z3.ToReal(left) / z3.ToReal(right)
            if z3.is_int(left):
                return z3.ToReal(left) / right
            if z3.is_int(right):
                return left / z3.ToReal(right)
            return left / right

        if expr.op == BinOpKind.MOD:
            # For real-valued operands, use left - right * ToInt(left / right)
            if not (z3.is_int(left) and z3.is_int(right)):
                return left - right * z3.ToInt(left / right)
            return left % right

        if expr.op == BinOpKind.POW:
            # Check for integer constant exponent
            if isinstance(expr.right, Const) and isinstance(expr.right.value, int):
                n = int(expr.right.value)
                if n == 0:
                    return z3.RealVal(1)
                if n == 1:
                    return left
                if n == 2:
                    return left * left
                if n >= 3:
                    result = left
                    for _ in range(n - 1):
                        result = result * left
                    return result
            # General power: use nonlinear encoding
            return left ** right

        if expr.op == BinOpKind.AND:
            left_b = left if z3.is_bool(left) else (left != z3.RealVal(0))
            right_b = right if z3.is_bool(right) else (right != z3.RealVal(0))
            return z3.And(left_b, right_b)

        if expr.op == BinOpKind.OR:
            left_b = left if z3.is_bool(left) else (left != z3.RealVal(0))
            right_b = right if z3.is_bool(right) else (right != z3.RealVal(0))
            return z3.Or(left_b, right_b)

        raise ValueError(f"Unsupported BinOp kind: {expr.op}")

    # ── Unary operations ─────────────────────────────────────────────────

    def visit_UnaryOp(self, expr: UnaryOp) -> Any:
        """Convert a UnaryOp to a Z3 expression.

        Args:
            expr: IR UnaryOp node.

        Returns:
            Z3 expression.
        """
        operand = self.visit(expr.operand)
        if expr.op == UnaryOpKind.NEG:
            return -operand
        if expr.op == UnaryOpKind.NOT:
            if z3.is_bool(operand):
                return z3.Not(operand)
            return operand == z3.RealVal(0)
        raise ValueError(f"Unsupported UnaryOp kind: {expr.op}")

    # ── Mathematical functions ───────────────────────────────────────────

    def visit_Abs(self, expr: Abs) -> Any:
        """Convert Abs to Z3 using If-then-else.

        Encodes |x| as If(x >= 0, x, -x).

        Args:
            expr: IR Abs node.

        Returns:
            Z3 expression for |operand|.
        """
        operand = self.visit(expr.operand)
        zero = z3.RealVal(0) if z3.is_real(operand) else z3.IntVal(0)
        return z3.If(operand >= zero, operand, -operand)

    def visit_Max(self, expr: Max) -> Any:
        """Convert Max to Z3 using If-then-else.

        Encodes max(a, b) as If(a >= b, a, b).

        Args:
            expr: IR Max node.

        Returns:
            Z3 expression.
        """
        left = self.visit(expr.left)
        right = self.visit(expr.right)
        return z3.If(left >= right, left, right)

    def visit_Min(self, expr: Min) -> Any:
        """Convert Min to Z3 using If-then-else.

        Encodes min(a, b) as If(a <= b, a, b).

        Args:
            expr: IR Min node.

        Returns:
            Z3 expression.
        """
        left = self.visit(expr.left)
        right = self.visit(expr.right)
        return z3.If(left <= right, left, right)

    def visit_Log(self, expr: Log) -> Any:
        """Convert Log to Z3 via polynomial approximation.

        Args:
            expr: IR Log node.

        Returns:
            Z3 expression for ln(operand).
        """
        operand = self.visit(expr.operand)
        result = self._transcendental.approx_ln(operand)
        return result.value

    def visit_Exp(self, expr: Exp) -> Any:
        """Convert Exp to Z3 via polynomial approximation.

        Args:
            expr: IR Exp node.

        Returns:
            Z3 expression for exp(operand).
        """
        operand = self.visit(expr.operand)
        result = self._transcendental.approx_exp(operand)
        return result.value

    def visit_Sqrt(self, expr: Sqrt) -> Any:
        """Convert Sqrt to Z3 via polynomial approximation.

        Args:
            expr: IR Sqrt node.

        Returns:
            Z3 expression for √(operand).
        """
        operand = self.visit(expr.operand)
        result = self._transcendental.approx_sqrt(operand)
        return result.value

    def visit_Phi(self, expr: Phi) -> Any:
        """Convert Phi (normal CDF) to Z3 via polynomial approximation.

        Args:
            expr: IR Phi node.

        Returns:
            Z3 expression for Φ(operand).
        """
        operand = self.visit(expr.operand)
        result = self._transcendental.approx_phi(operand)
        return result.value

    def visit_PhiInv(self, expr: PhiInv) -> Any:
        """Convert PhiInv (inverse normal CDF) to Z3 via polynomial approx.

        Args:
            expr: IR PhiInv node.

        Returns:
            Z3 expression for Φ⁻¹(operand).
        """
        operand = self.visit(expr.operand)
        result = self._transcendental.approx_phi_inv(operand)
        return result.value

    # ── Control-flow expressions ─────────────────────────────────────────

    def visit_Cond(self, expr: Cond) -> Any:
        """Convert Cond (if-then-else expression) to Z3.

        Args:
            expr: IR Cond node.

        Returns:
            Z3 If expression.
        """
        cond = self.convert_bool(expr.condition)
        true_val = self.visit(expr.true_expr)
        false_val = self.visit(expr.false_expr)
        return z3.If(cond, true_val, false_val)

    def visit_LetExpr(self, expr: LetExpr) -> Any:
        """Convert LetExpr by substituting the bound value.

        Binds the variable in a local environment and evaluates the body.

        Args:
            expr: IR LetExpr node.

        Returns:
            Z3 expression for the body with substitution applied.
        """
        val = self.visit(expr.value)
        old = self._let_env.get(expr.var_name)
        self._let_env[expr.var_name] = val
        result = self.visit(expr.body)
        if old is not None:
            self._let_env[expr.var_name] = old
        else:
            self._let_env.pop(expr.var_name, None)
        return result

    def visit_SumExpr(self, expr: SumExpr) -> Any:
        """Convert SumExpr by unrolling the bounded sum.

        If bounds are constant, unrolls the sum.  Otherwise creates
        a symbolic variable for the sum result.

        Args:
            expr: IR SumExpr node.

        Returns:
            Z3 expression for the sum.
        """
        if isinstance(expr.lo, Const) and isinstance(expr.hi, Const):
            lo_val = int(expr.lo.value)
            hi_val = int(expr.hi.value)
            if hi_val - lo_val > 100:
                # Too many iterations; use abstract variable
                return self._make_aux_real("sum")
            result = z3.RealVal(0)
            for i in range(lo_val, hi_val + 1):
                old = self._let_env.get(expr.var_name)
                self._let_env[expr.var_name] = z3.IntVal(i)
                term = self.visit(expr.body)
                if z3.is_int(term):
                    term = z3.ToReal(term)
                result = result + term
                if old is not None:
                    self._let_env[expr.var_name] = old
                else:
                    self._let_env.pop(expr.var_name, None)
            return result
        return self._make_aux_real("sum")

    def visit_FuncCall(self, expr: FuncCall) -> Any:
        """Convert a FuncCall to Z3.

        Known functions (exp, log, sqrt, phi, phi_inv) are handled
        via transcendental approximations.  Unknown functions produce
        an uninterpreted function application.

        Args:
            expr: IR FuncCall node.

        Returns:
            Z3 expression.
        """
        if expr.name in ("exp",):
            if expr.args:
                arg = self.visit(expr.args[0])
                return self._transcendental.approx_exp(arg).value
        elif expr.name in ("log", "ln"):
            if expr.args:
                arg = self.visit(expr.args[0])
                return self._transcendental.approx_ln(arg).value
        elif expr.name in ("sqrt",):
            if expr.args:
                arg = self.visit(expr.args[0])
                return self._transcendental.approx_sqrt(arg).value
        elif expr.name in ("phi", "Phi"):
            if expr.args:
                arg = self.visit(expr.args[0])
                return self._transcendental.approx_phi(arg).value
        elif expr.name in ("phi_inv", "PhiInv"):
            if expr.args:
                arg = self.visit(expr.args[0])
                return self._transcendental.approx_phi_inv(arg).value
        elif expr.name in ("abs",):
            if expr.args:
                arg = self.visit(expr.args[0])
                return z3.If(arg >= z3.RealVal(0), arg, -arg)
        elif expr.name in ("max",) and len(expr.args) >= 2:
            a = self.visit(expr.args[0])
            b = self.visit(expr.args[1])
            return z3.If(a >= b, a, b)
        elif expr.name in ("min",) and len(expr.args) >= 2:
            a = self.visit(expr.args[0])
            b = self.visit(expr.args[1])
            return z3.If(a <= b, a, b)

        # Uninterpreted function
        z3_args = [self.visit(a) for a in expr.args]
        f = z3.Function(
            expr.name,
            *([z3.RealSort()] * len(z3_args)),
            z3.RealSort(),
        )
        return f(*z3_args)

    def visit_ArrayAccess(self, expr: ArrayAccess) -> Any:
        """Convert ArrayAccess to Z3 via uninterpreted function.

        Args:
            expr: IR ArrayAccess node.

        Returns:
            Z3 expression.
        """
        arr = self.visit(expr.array)
        idx = self.visit(expr.index)
        f = z3.Function("array_access", z3.RealSort(), z3.IntSort(), z3.RealSort())
        return f(arr, idx)

    def visit_TupleAccess(self, expr: TupleAccess) -> Any:
        """Convert TupleAccess to Z3 via uninterpreted function.

        Args:
            expr: IR TupleAccess node.

        Returns:
            Z3 expression.
        """
        tup = self.visit(expr.tuple_expr)
        f = z3.Function(
            f"tuple_field_{expr.field_idx}",
            z3.RealSort(),
            z3.RealSort(),
        )
        return f(tup)


# ═══════════════════════════════════════════════════════════════════════════
# ABSOLUTE VALUE LINEARISATION
# ═══════════════════════════════════════════════════════════════════════════


class AbsLinearizer:
    """Linearise absolute value expressions using auxiliary variables.

    Encodes |x| = x⁺ + x⁻ where x = x⁺ - x⁻ and x⁺, x⁻ ≥ 0.
    This transforms nonlinear |x| into linear constraints, allowing
    QF_LRA solving.

    Usage::

        lin = AbsLinearizer()
        abs_var, constraints = lin.linearize(z3_expr)
        solver.add(*constraints)
        # abs_var now represents |z3_expr|
    """

    def __init__(self) -> None:
        self._aux_count = itertools.count()

    def linearize(self, x: Any) -> tuple[Any, list[Any]]:
        """Linearise |x| using auxiliary variables.

        Introduces x⁺ and x⁻ such that:
          - x = x⁺ - x⁻
          - x⁺ ≥ 0, x⁻ ≥ 0
          - |x| = x⁺ + x⁻

        Args:
            x: Z3 real expression.

        Returns:
            Tuple of (|x| expression, list of constraints).
        """
        idx = next(self._aux_count)
        x_pos = z3.Real(f"__abs_pos_{idx}")
        x_neg = z3.Real(f"__abs_neg_{idx}")

        constraints = [
            x == x_pos - x_neg,
            x_pos >= z3.RealVal(0),
            x_neg >= z3.RealVal(0),
        ]

        abs_x = x_pos + x_neg
        return abs_x, constraints

    def linearize_diff(self, a: Any, b: Any) -> tuple[Any, list[Any]]:
        """Linearise |a - b| using auxiliary variables.

        Args:
            a: Z3 real expression.
            b: Z3 real expression.

        Returns:
            Tuple of (|a - b| expression, list of constraints).
        """
        return self.linearize(a - b)


# ═══════════════════════════════════════════════════════════════════════════
# CASE SPLITTER
# ═══════════════════════════════════════════════════════════════════════════


class CaseSplitter:
    """Case-split absolute value expressions.

    Instead of linearisation, produces two cases:
      Case 1: x ≥ 0 → |x| = x
      Case 2: x < 0 → |x| = -x

    This can be more precise than linearisation for verification.

    Usage::

        splitter = CaseSplitter()
        cases = splitter.split_abs(expr)
        for guard, value in cases:
            # guard is a Z3 boolean, value is the |expr| in that case
    """

    def split_abs(self, x: Any) -> list[tuple[Any, Any]]:
        """Split |x| into two cases.

        Args:
            x: Z3 real expression.

        Returns:
            List of (guard, value) pairs.
        """
        zero = z3.RealVal(0)
        return [
            (x >= zero, x),
            (x < zero, -x),
        ]

    def split_abs_diff(self, a: Any, b: Any) -> list[tuple[Any, Any]]:
        """Split |a - b| into two cases.

        Args:
            a: Z3 real expression.
            b: Z3 real expression.

        Returns:
            List of (guard, value) pairs.
        """
        diff = a - b
        return self.split_abs(diff)

    def encode_abs_constraint(self, x: Any, bound: Any, op: str = "<=") -> Any:
        """Encode |x| op bound via case split.

        For |x| <= bound:  (x >= 0 → x <= bound) ∧ (x < 0 → -x <= bound)
        This is equivalent to: -bound <= x ∧ x <= bound

        Args:
            x:     Z3 real expression.
            bound: Z3 real expression for the bound.
            op:    Comparison operator ('<=', '<', '>=', '>').

        Returns:
            Z3 boolean expression.
        """
        if op == "<=":
            return z3.And(x <= bound, x >= -bound)
        elif op == "<":
            return z3.And(x < bound, x > -bound)
        elif op == ">=":
            return z3.Or(x >= bound, x <= -bound)
        elif op == ">":
            return z3.Or(x > bound, x < -bound)
        raise ValueError(f"Unsupported operator: {op}")


# ═══════════════════════════════════════════════════════════════════════════
# PATH CONDITION ENCODER
# ═══════════════════════════════════════════════════════════════════════════


class PathConditionEncoder:
    """Encode PathCondition objects as Z3 boolean formulas.

    Converts the conjunction of IR boolean expressions in a PathCondition
    to a single Z3 boolean formula.

    Args:
        converter: ExprToZ3 instance for expression conversion.
    """

    def __init__(self, converter: ExprToZ3) -> None:
        self._converter = converter

    def encode(self, pc: PathCondition) -> Any:
        """Encode a PathCondition as a Z3 formula.

        Args:
            pc: The path condition to encode.

        Returns:
            Z3 boolean expression (conjunction of all conjuncts).
        """
        if pc.is_trivially_true():
            return z3.BoolVal(True)
        if pc.is_trivially_false():
            return z3.BoolVal(False)

        conjuncts = []
        for c in pc.conjuncts:
            z3_c = self._converter.convert_bool(c)
            conjuncts.append(z3_c)

        if len(conjuncts) == 1:
            return conjuncts[0]
        return z3.And(*conjuncts)

    def encode_negated(self, pc: PathCondition) -> Any:
        """Encode the negation of a PathCondition.

        Args:
            pc: The path condition to negate.

        Returns:
            Z3 boolean expression for ¬(pc).
        """
        return z3.Not(self.encode(pc))


# ═══════════════════════════════════════════════════════════════════════════
# CONSTRAINT BUILDER
# ═══════════════════════════════════════════════════════════════════════════


class ConstraintBuilder:
    """Build SMT constraints for privacy verification.

    Provides methods for constructing common constraint patterns:
    - Variable bounds
    - Adjacency constraints (d ~ d')
    - Privacy loss bounds
    - Path feasibility

    Args:
        converter: ExprToZ3 instance.
    """

    def __init__(self, converter: ExprToZ3) -> None:
        self._conv = converter
        self._pc_encoder = PathConditionEncoder(converter)
        self._linearizer = AbsLinearizer()
        self._splitter = CaseSplitter()

    def build_adjacency(
        self,
        d_var: str,
        d_prime_var: str,
        sensitivity: float = 1.0,
    ) -> list[Any]:
        """Build adjacency constraints: |d - d'| <= sensitivity.

        Args:
            d_var:       Name of the database variable for d.
            d_prime_var: Name of the database variable for d'.
            sensitivity: The adjacency bound (ℓ₁ sensitivity).

        Returns:
            List of Z3 constraints.
        """
        d = z3.Real(d_var)
        dp = z3.Real(d_prime_var)
        delta = d - dp
        bound = z3.RealVal(str(sensitivity))
        return [
            delta <= bound,
            delta >= -bound,
        ]

    def build_variable_bounds(
        self,
        var_name: str,
        lo: float | None = None,
        hi: float | None = None,
    ) -> list[Any]:
        """Build variable bound constraints.

        Args:
            var_name: Variable name.
            lo:       Lower bound (None for unbounded).
            hi:       Upper bound (None for unbounded).

        Returns:
            List of Z3 constraints.
        """
        v = z3.Real(var_name)
        constraints: list[Any] = []
        if lo is not None:
            constraints.append(v >= z3.RealVal(str(lo)))
        if hi is not None:
            constraints.append(v <= z3.RealVal(str(hi)))
        return constraints

    def build_positivity(self, var_name: str) -> Any:
        """Assert that a variable is strictly positive.

        Args:
            var_name: Variable name.

        Returns:
            Z3 constraint.
        """
        return z3.Real(var_name) > z3.RealVal(0)

    def build_path_condition(self, path: SymbolicPath) -> Any:
        """Encode a symbolic path's condition.

        Args:
            path: The symbolic path.

        Returns:
            Z3 boolean expression for the path guard.
        """
        return self._pc_encoder.encode(path.path_condition)

    def build_abs_bound(
        self,
        expr: Any,
        bound: Any,
        use_linearization: bool = True,
    ) -> tuple[Any, list[Any]]:
        """Build |expr| <= bound constraint.

        Args:
            expr:               Z3 expression.
            bound:              Z3 expression for the bound.
            use_linearization:  If True, use aux vars; else use case split.

        Returns:
            Tuple of (constraint, additional_constraints).
        """
        if use_linearization:
            abs_e, aux_constraints = self._linearizer.linearize(expr)
            return abs_e <= bound, aux_constraints
        else:
            return self._splitter.encode_abs_constraint(expr, bound), []


# ═══════════════════════════════════════════════════════════════════════════
# FULL ENCODING BUILDER
# ═══════════════════════════════════════════════════════════════════════════


class EncodingBuilder:
    """Build a complete SMTEncoding from paths and privacy constraints.

    Orchestrates the ExprToZ3, ConstraintBuilder, and TheoryAnalyzer
    to produce a self-contained SMTEncoding.

    Args:
        precision:        Transcendental approximation precision.
        use_linearization: Whether to linearise absolute values.
    """

    def __init__(
        self,
        precision: Precision = Precision.STANDARD,
        use_linearization: bool = True,
    ) -> None:
        self._precision = precision
        self._use_linearization = use_linearization

    def encode_path(
        self,
        path: SymbolicPath,
        output_var: str = "o",
    ) -> SMTEncoding:
        """Encode a single symbolic path as an SMT formula.

        Produces constraints for:
          1. Path condition (guard)
          2. Output expression (o == path.output_expr)
          3. Variable assignments

        Args:
            path:       The symbolic path to encode.
            output_var: Name for the output variable.

        Returns:
            SMTEncoding for the path.
        """
        converter = ExprToZ3(precision=self._precision)
        assertions: list[Any] = []

        # Path condition
        pc_encoder = PathConditionEncoder(converter)
        pc_z3 = pc_encoder.encode(path.path_condition)
        assertions.append(pc_z3)

        # Output expression
        output_z3 = converter.convert(path.output_expr)
        o_var = z3.Real(output_var)
        assertions.append(o_var == output_z3)

        # Variable assignments
        for var_name, var_expr in path.assignments.items():
            var_z3 = z3.Real(var_name)
            val_z3 = converter.convert(var_expr)
            assertions.append(var_z3 == val_z3)

        # Auxiliary constraints
        assertions.extend(converter.aux_constraints)

        encoding = SMTEncoding(
            variables=converter.variables,
            assertions=assertions,
            aux_vars=converter.aux_vars,
            soundness=converter.tracker,
            metadata={
                "path_id": path.path_id,
                "output_var": output_var,
            },
        )
        encoding._rebuild_formula()
        return encoding

    def encode_log_ratio(
        self,
        log_ratio_expr: TypedExpr,
        path_cond_d: PathCondition | None = None,
        path_cond_dp: PathCondition | None = None,
    ) -> SMTEncoding:
        """Encode a log density ratio expression.

        Args:
            log_ratio_expr: The log ratio L(o) as an IR expression.
            path_cond_d:    Path condition for dataset d.
            path_cond_dp:   Path condition for dataset d'.

        Returns:
            SMTEncoding with the ratio and path conditions.
        """
        converter = ExprToZ3(precision=self._precision)
        assertions: list[Any] = []

        # Log ratio
        lr_z3 = converter.convert(log_ratio_expr)
        lr_var = z3.Real("L")
        assertions.append(lr_var == lr_z3)

        # Path conditions
        pc_encoder = PathConditionEncoder(converter)
        if path_cond_d is not None:
            assertions.append(pc_encoder.encode(path_cond_d))
        if path_cond_dp is not None:
            assertions.append(pc_encoder.encode(path_cond_dp))

        assertions.extend(converter.aux_constraints)

        encoding = SMTEncoding(
            variables=converter.variables,
            assertions=assertions,
            aux_vars=converter.aux_vars,
            soundness=converter.tracker,
            metadata={"type": "log_ratio"},
        )
        encoding._rebuild_formula()
        return encoding

    def encode_privacy_negation(
        self,
        log_ratio_expr: TypedExpr,
        epsilon: float,
        path_cond_d: PathCondition | None = None,
        path_cond_dp: PathCondition | None = None,
    ) -> SMTEncoding:
        """Encode the negation of a pure-DP privacy property.

        Asserts that |L(o)| > epsilon (the negation of |L(o)| <= epsilon)
        so that SAT means a privacy violation exists.

        Args:
            log_ratio_expr: Log ratio expression.
            epsilon:        Privacy budget.
            path_cond_d:    Path condition for d.
            path_cond_dp:   Path condition for d'.

        Returns:
            SMTEncoding asserting the negation.
        """
        converter = ExprToZ3(precision=self._precision)
        builder = ConstraintBuilder(converter)
        pc_encoder = PathConditionEncoder(converter)
        assertions: list[Any] = []

        # Path conditions
        if path_cond_d is not None:
            assertions.append(pc_encoder.encode(path_cond_d))
        if path_cond_dp is not None:
            assertions.append(pc_encoder.encode(path_cond_dp))

        # Log ratio
        lr_z3 = converter.convert(log_ratio_expr)
        eps_z3 = z3.RealVal(str(epsilon))

        # Negate: |L(o)| > eps  ↔  L > eps ∨ L < -eps
        violation = z3.Or(lr_z3 > eps_z3, lr_z3 < -eps_z3)
        assertions.append(violation)

        assertions.extend(converter.aux_constraints)

        encoding = SMTEncoding(
            variables=converter.variables,
            assertions=assertions,
            aux_vars=converter.aux_vars,
            soundness=converter.tracker,
            metadata={
                "type": "privacy_negation",
                "epsilon": epsilon,
                "notion": "pure_dp",
            },
        )
        encoding._rebuild_formula()
        return encoding

    def analyze_theory(self, encoding: SMTEncoding) -> TheoryAnalysisResult:
        """Analyse the theory requirements of an encoding.

        Args:
            encoding: The SMT encoding to analyse.

        Returns:
            TheoryAnalysisResult with theory and feature information.
        """
        analyzer = TheoryAnalyzer()
        if encoding.formula is not None:
            return analyzer.analyze_z3(encoding.formula)
        return TheoryAnalysisResult()
