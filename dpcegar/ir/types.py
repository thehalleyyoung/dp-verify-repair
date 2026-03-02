"""Core type system for the DP-CEGAR intermediate representation.

This module defines every expression and type node used in MechIR.
All nodes are immutable dataclasses with ``__slots__`` for performance.

Hierarchy
---------
IRType          – enumeration of base types
NoiseKind       – enumeration of noise distributions
PrivacyNotion   – enumeration of DP variants

TypedExpr       – abstract base for all expression nodes
  ├─ Var        – variable reference (with SSA version)
  ├─ Const      – literal constant
  ├─ BinOp      – binary operator  (arith / logic / comparison)
  ├─ UnaryOp    – unary operator   (negate / not / abs / …)
  ├─ FuncCall   – named function application
  ├─ ArrayAccess– arr[index]
  ├─ TupleAccess– tup.i
  ├─ Abs        – |x|
  ├─ Max / Min  – max(a,b) / min(a,b)
  ├─ Log / Exp  – natural log / exponential
  ├─ Sqrt       – square root
  ├─ Phi        – standard normal CDF Φ(x)
  ├─ PhiInv     – inverse normal CDF Φ⁻¹(p)
  ├─ Cond       – if-then-else expression
  ├─ LetExpr    – let binding
  └─ SumExpr    – bounded summation

PrivacyBudget   – abstract base for privacy budgets
  ├─ PureBudget      (ε)
  ├─ ApproxBudget    (ε, δ)
  ├─ ZCDPBudget      (ρ)
  ├─ RDPBudget       (α, ε)
  ├─ FDPBudget       (trade-off function)
  └─ GDPBudget       (μ)

ExprVisitor     – generic visitor over expressions (return type T)
"""

from __future__ import annotations

import itertools
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Generic,
    Iterator,
    Sequence,
    TypeVar,
)

# ═══════════════════════════════════════════════════════════════════════════
# ENUMERATIONS
# ═══════════════════════════════════════════════════════════════════════════


class IRType(Enum):
    """Primitive types in the IR type system."""

    INT = auto()
    REAL = auto()
    BOOL = auto()
    DIST = auto()
    ARRAY = auto()
    TUPLE = auto()

    def is_numeric(self) -> bool:
        """True for INT and REAL."""
        return self in (IRType.INT, IRType.REAL)

    def __str__(self) -> str:
        return self.name.lower()


class NoiseKind(Enum):
    """Supported noise distributions."""

    LAPLACE = auto()
    GAUSSIAN = auto()
    EXPONENTIAL = auto()

    def __str__(self) -> str:
        return self.name.lower()


class PrivacyNotion(Enum):
    """Supported differential privacy variants."""

    PURE_DP = auto()
    APPROX_DP = auto()
    ZCDP = auto()
    RDP = auto()
    FDP = auto()
    GDP = auto()

    def __str__(self) -> str:
        _names = {
            PrivacyNotion.PURE_DP: "ε-DP",
            PrivacyNotion.APPROX_DP: "(ε,δ)-DP",
            PrivacyNotion.ZCDP: "ρ-zCDP",
            PrivacyNotion.RDP: "(α,ε)-RDP",
            PrivacyNotion.FDP: "f-DP",
            PrivacyNotion.GDP: "μ-GDP",
        }
        return _names.get(self, self.name)


class BinOpKind(Enum):
    """Binary operators."""

    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"
    MOD = "%"
    POW = "**"
    AND = "&&"
    OR = "||"
    EQ = "=="
    NEQ = "!="
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="

    @property
    def is_arithmetic(self) -> bool:
        return self in (
            BinOpKind.ADD, BinOpKind.SUB, BinOpKind.MUL,
            BinOpKind.DIV, BinOpKind.MOD, BinOpKind.POW,
        )

    @property
    def is_comparison(self) -> bool:
        return self in (
            BinOpKind.EQ, BinOpKind.NEQ, BinOpKind.LT,
            BinOpKind.LE, BinOpKind.GT, BinOpKind.GE,
        )

    @property
    def is_logical(self) -> bool:
        return self in (BinOpKind.AND, BinOpKind.OR)

    def __str__(self) -> str:
        return self.value


class UnaryOpKind(Enum):
    """Unary operators."""

    NEG = "-"
    NOT = "!"

    def __str__(self) -> str:
        return self.value


# ═══════════════════════════════════════════════════════════════════════════
# EXPRESSION BASE & VISITOR
# ═══════════════════════════════════════════════════════════════════════════

T = TypeVar("T")

# Global counter for unique expression IDs
_expr_id_counter = itertools.count()


class ExprVisitor(ABC, Generic[T]):
    """Visitor pattern for expression trees.

    Subclass and override the ``visit_*`` methods you need.  The default
    implementation of each method calls :meth:`generic_visit`.
    """

    def visit(self, expr: TypedExpr) -> T:
        """Dispatch to the appropriate ``visit_*`` method."""
        method_name = f"visit_{type(expr).__name__}"
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(expr)

    def generic_visit(self, expr: TypedExpr) -> T:
        """Called when no specific visitor exists.  Override for default behaviour."""
        raise NotImplementedError(
            f"{type(self).__name__} has no handler for {type(expr).__name__}"
        )

    # --- Leaf nodes ---
    def visit_Var(self, expr: Var) -> T:
        return self.generic_visit(expr)

    def visit_Const(self, expr: Const) -> T:
        return self.generic_visit(expr)

    # --- Compound nodes ---
    def visit_BinOp(self, expr: BinOp) -> T:
        return self.generic_visit(expr)

    def visit_UnaryOp(self, expr: UnaryOp) -> T:
        return self.generic_visit(expr)

    def visit_FuncCall(self, expr: FuncCall) -> T:
        return self.generic_visit(expr)

    def visit_ArrayAccess(self, expr: ArrayAccess) -> T:
        return self.generic_visit(expr)

    def visit_TupleAccess(self, expr: TupleAccess) -> T:
        return self.generic_visit(expr)

    def visit_Abs(self, expr: Abs) -> T:
        return self.generic_visit(expr)

    def visit_Max(self, expr: Max) -> T:
        return self.generic_visit(expr)

    def visit_Min(self, expr: Min) -> T:
        return self.generic_visit(expr)

    def visit_Log(self, expr: Log) -> T:
        return self.generic_visit(expr)

    def visit_Exp(self, expr: Exp) -> T:
        return self.generic_visit(expr)

    def visit_Sqrt(self, expr: Sqrt) -> T:
        return self.generic_visit(expr)

    def visit_Phi(self, expr: Phi) -> T:
        return self.generic_visit(expr)

    def visit_PhiInv(self, expr: PhiInv) -> T:
        return self.generic_visit(expr)

    def visit_Cond(self, expr: Cond) -> T:
        return self.generic_visit(expr)

    def visit_LetExpr(self, expr: LetExpr) -> T:
        return self.generic_visit(expr)

    def visit_SumExpr(self, expr: SumExpr) -> T:
        return self.generic_visit(expr)


# ═══════════════════════════════════════════════════════════════════════════
# EXPRESSION NODES
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class TypedExpr(ABC):
    """Abstract base for all typed expressions in MechIR.

    Every expression carries its inferred :class:`IRType` and a unique
    ``expr_id`` for identity-based caching.
    """

    ty: IRType
    expr_id: int = field(default_factory=lambda: next(_expr_id_counter), compare=False, repr=False)

    def accept(self, visitor: ExprVisitor[T]) -> T:
        """Double-dispatch to the visitor."""
        return visitor.visit(self)

    @abstractmethod
    def children(self) -> tuple[TypedExpr, ...]:
        """Return immediate sub-expressions."""
        ...

    @abstractmethod
    def map_children(self, fn: Callable[[TypedExpr], TypedExpr]) -> TypedExpr:
        """Return a copy with *fn* applied to each child expression."""
        ...

    def free_vars(self) -> frozenset[str]:
        """Collect free variable names from this expression tree."""
        collector = _FreeVarCollectorExpr()
        collector.visit(self)
        return frozenset(collector.vars)

    def substitute(self, mapping: dict[str, TypedExpr]) -> TypedExpr:
        """Replace free variables according to *mapping*."""
        sub = _ExprSubstituterImpl(mapping)
        return sub.visit(self)

    def simplify(self) -> TypedExpr:
        """Apply basic algebraic simplifications (constant folding, identities)."""
        return _simplify(self)


# --- Leaf nodes -----------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Var(TypedExpr):
    """Variable reference, optionally with SSA version number.

    Attributes:
        name:    Base variable name.
        version: SSA version (None before SSA numbering).
    """

    name: str = ""
    version: int | None = None

    def children(self) -> tuple[TypedExpr, ...]:
        return ()

    def map_children(self, fn: Callable[[TypedExpr], TypedExpr]) -> Var:
        return self

    @property
    def ssa_name(self) -> str:
        """Fully qualified SSA name, e.g. ``x_3``."""
        if self.version is not None:
            return f"{self.name}_{self.version}"
        return self.name

    def __str__(self) -> str:
        return self.ssa_name

    def __repr__(self) -> str:
        v = f", v={self.version}" if self.version is not None else ""
        return f"Var({self.name!r}{v}, {self.ty})"


@dataclass(frozen=True, slots=True)
class Const(TypedExpr):
    """Literal constant.

    Attributes:
        value: The constant value (int, float, or bool).
    """

    value: int | float | bool = 0

    def children(self) -> tuple[TypedExpr, ...]:
        return ()

    def map_children(self, fn: Callable[[TypedExpr], TypedExpr]) -> Const:
        return self

    @classmethod
    def int_(cls, v: int) -> Const:
        return cls(ty=IRType.INT, value=v)

    @classmethod
    def real(cls, v: float) -> Const:
        return cls(ty=IRType.REAL, value=v)

    @classmethod
    def bool_(cls, v: bool) -> Const:
        return cls(ty=IRType.BOOL, value=v)

    @classmethod
    def zero(cls, ty: IRType = IRType.REAL) -> Const:
        return cls(ty=ty, value=0 if ty == IRType.INT else 0.0)

    @classmethod
    def one(cls, ty: IRType = IRType.REAL) -> Const:
        return cls(ty=ty, value=1 if ty == IRType.INT else 1.0)

    @property
    def is_zero(self) -> bool:
        return self.value == 0

    @property
    def is_one(self) -> bool:
        return self.value == 1

    def __str__(self) -> str:
        if isinstance(self.value, bool):
            return str(self.value).lower()
        return str(self.value)

    def __repr__(self) -> str:
        return f"Const({self.value!r}, {self.ty})"


# --- Compound nodes -------------------------------------------------------

@dataclass(frozen=True, slots=True)
class BinOp(TypedExpr):
    """Binary operation.

    Attributes:
        op:    The operator kind.
        left:  Left operand.
        right: Right operand.
    """

    op: BinOpKind = BinOpKind.ADD
    left: TypedExpr = field(default_factory=lambda: Const.zero())
    right: TypedExpr = field(default_factory=lambda: Const.zero())

    def children(self) -> tuple[TypedExpr, ...]:
        return (self.left, self.right)

    def map_children(self, fn: Callable[[TypedExpr], TypedExpr]) -> BinOp:
        return BinOp(ty=self.ty, op=self.op, left=fn(self.left), right=fn(self.right))

    def __str__(self) -> str:
        return f"({self.left} {self.op} {self.right})"

    def __repr__(self) -> str:
        return f"BinOp({self.op}, {self.left!r}, {self.right!r})"


@dataclass(frozen=True, slots=True)
class UnaryOp(TypedExpr):
    """Unary operation (negation, logical not).

    Attributes:
        op:      The unary operator kind.
        operand: The operand expression.
    """

    op: UnaryOpKind = UnaryOpKind.NEG
    operand: TypedExpr = field(default_factory=lambda: Const.zero())

    def children(self) -> tuple[TypedExpr, ...]:
        return (self.operand,)

    def map_children(self, fn: Callable[[TypedExpr], TypedExpr]) -> UnaryOp:
        return UnaryOp(ty=self.ty, op=self.op, operand=fn(self.operand))

    def __str__(self) -> str:
        return f"({self.op}{self.operand})"

    def __repr__(self) -> str:
        return f"UnaryOp({self.op}, {self.operand!r})"


@dataclass(frozen=True, slots=True)
class FuncCall(TypedExpr):
    """Named function application.

    Attributes:
        name: Function name.
        args: Positional arguments.
    """

    name: str = ""
    args: tuple[TypedExpr, ...] = ()

    def children(self) -> tuple[TypedExpr, ...]:
        return self.args

    def map_children(self, fn: Callable[[TypedExpr], TypedExpr]) -> FuncCall:
        return FuncCall(ty=self.ty, name=self.name, args=tuple(fn(a) for a in self.args))

    def __str__(self) -> str:
        args_str = ", ".join(str(a) for a in self.args)
        return f"{self.name}({args_str})"

    def __repr__(self) -> str:
        return f"FuncCall({self.name!r}, {self.args!r})"


@dataclass(frozen=True, slots=True)
class ArrayAccess(TypedExpr):
    """Array element access: ``array[index]``.

    Attributes:
        array: The array expression.
        index: The index expression (must be INT).
    """

    array: TypedExpr = field(default_factory=lambda: Const.zero())
    index: TypedExpr = field(default_factory=lambda: Const.int_(0))

    def children(self) -> tuple[TypedExpr, ...]:
        return (self.array, self.index)

    def map_children(self, fn: Callable[[TypedExpr], TypedExpr]) -> ArrayAccess:
        return ArrayAccess(ty=self.ty, array=fn(self.array), index=fn(self.index))

    def __str__(self) -> str:
        return f"{self.array}[{self.index}]"


@dataclass(frozen=True, slots=True)
class TupleAccess(TypedExpr):
    """Tuple element access: ``tup.i``.

    Attributes:
        tuple_expr: The tuple expression.
        field_idx:  0-based field index.
    """

    tuple_expr: TypedExpr = field(default_factory=lambda: Const.zero())
    field_idx: int = 0

    def children(self) -> tuple[TypedExpr, ...]:
        return (self.tuple_expr,)

    def map_children(self, fn: Callable[[TypedExpr], TypedExpr]) -> TupleAccess:
        return TupleAccess(ty=self.ty, tuple_expr=fn(self.tuple_expr), field_idx=self.field_idx)

    def __str__(self) -> str:
        return f"{self.tuple_expr}.{self.field_idx}"


# --- Mathematical functions -----------------------------------------------

@dataclass(frozen=True, slots=True)
class Abs(TypedExpr):
    """Absolute value |operand|."""

    operand: TypedExpr = field(default_factory=lambda: Const.zero())

    def children(self) -> tuple[TypedExpr, ...]:
        return (self.operand,)

    def map_children(self, fn: Callable[[TypedExpr], TypedExpr]) -> Abs:
        return Abs(ty=self.ty, operand=fn(self.operand))

    def __str__(self) -> str:
        return f"|{self.operand}|"


@dataclass(frozen=True, slots=True)
class Max(TypedExpr):
    """Maximum of two expressions: max(left, right)."""

    left: TypedExpr = field(default_factory=lambda: Const.zero())
    right: TypedExpr = field(default_factory=lambda: Const.zero())

    def children(self) -> tuple[TypedExpr, ...]:
        return (self.left, self.right)

    def map_children(self, fn: Callable[[TypedExpr], TypedExpr]) -> Max:
        return Max(ty=self.ty, left=fn(self.left), right=fn(self.right))

    def __str__(self) -> str:
        return f"max({self.left}, {self.right})"


@dataclass(frozen=True, slots=True)
class Min(TypedExpr):
    """Minimum of two expressions: min(left, right)."""

    left: TypedExpr = field(default_factory=lambda: Const.zero())
    right: TypedExpr = field(default_factory=lambda: Const.zero())

    def children(self) -> tuple[TypedExpr, ...]:
        return (self.left, self.right)

    def map_children(self, fn: Callable[[TypedExpr], TypedExpr]) -> Min:
        return Min(ty=self.ty, left=fn(self.left), right=fn(self.right))

    def __str__(self) -> str:
        return f"min({self.left}, {self.right})"


@dataclass(frozen=True, slots=True)
class Log(TypedExpr):
    """Natural logarithm: ln(operand)."""

    operand: TypedExpr = field(default_factory=lambda: Const.one())

    def children(self) -> tuple[TypedExpr, ...]:
        return (self.operand,)

    def map_children(self, fn: Callable[[TypedExpr], TypedExpr]) -> Log:
        return Log(ty=self.ty, operand=fn(self.operand))

    def __str__(self) -> str:
        return f"ln({self.operand})"


@dataclass(frozen=True, slots=True)
class Exp(TypedExpr):
    """Natural exponential: exp(operand)."""

    operand: TypedExpr = field(default_factory=lambda: Const.zero())

    def children(self) -> tuple[TypedExpr, ...]:
        return (self.operand,)

    def map_children(self, fn: Callable[[TypedExpr], TypedExpr]) -> Exp:
        return Exp(ty=self.ty, operand=fn(self.operand))

    def __str__(self) -> str:
        return f"exp({self.operand})"


@dataclass(frozen=True, slots=True)
class Sqrt(TypedExpr):
    """Square root: √(operand)."""

    operand: TypedExpr = field(default_factory=lambda: Const.zero())

    def children(self) -> tuple[TypedExpr, ...]:
        return (self.operand,)

    def map_children(self, fn: Callable[[TypedExpr], TypedExpr]) -> Sqrt:
        return Sqrt(ty=self.ty, operand=fn(self.operand))

    def __str__(self) -> str:
        return f"√({self.operand})"


@dataclass(frozen=True, slots=True)
class Phi(TypedExpr):
    """Standard normal CDF: Φ(operand)."""

    operand: TypedExpr = field(default_factory=lambda: Const.zero())

    def children(self) -> tuple[TypedExpr, ...]:
        return (self.operand,)

    def map_children(self, fn: Callable[[TypedExpr], TypedExpr]) -> Phi:
        return Phi(ty=self.ty, operand=fn(self.operand))

    def __str__(self) -> str:
        return f"Φ({self.operand})"


@dataclass(frozen=True, slots=True)
class PhiInv(TypedExpr):
    """Inverse normal CDF: Φ⁻¹(operand)."""

    operand: TypedExpr = field(default_factory=lambda: Const.zero())

    def children(self) -> tuple[TypedExpr, ...]:
        return (self.operand,)

    def map_children(self, fn: Callable[[TypedExpr], TypedExpr]) -> PhiInv:
        return PhiInv(ty=self.ty, operand=fn(self.operand))

    def __str__(self) -> str:
        return f"Φ⁻¹({self.operand})"


# --- Control-flow expressions --------------------------------------------

@dataclass(frozen=True, slots=True)
class Cond(TypedExpr):
    """Conditional (if-then-else) expression.

    Attributes:
        condition:  Boolean guard.
        true_expr:  Value when condition is true.
        false_expr: Value when condition is false.
    """

    condition: TypedExpr = field(default_factory=lambda: Const.bool_(True))
    true_expr: TypedExpr = field(default_factory=lambda: Const.zero())
    false_expr: TypedExpr = field(default_factory=lambda: Const.zero())

    def children(self) -> tuple[TypedExpr, ...]:
        return (self.condition, self.true_expr, self.false_expr)

    def map_children(self, fn: Callable[[TypedExpr], TypedExpr]) -> Cond:
        return Cond(
            ty=self.ty,
            condition=fn(self.condition),
            true_expr=fn(self.true_expr),
            false_expr=fn(self.false_expr),
        )

    def __str__(self) -> str:
        return f"(if {self.condition} then {self.true_expr} else {self.false_expr})"


@dataclass(frozen=True, slots=True)
class LetExpr(TypedExpr):
    """Let-binding expression: ``let var = value in body``.

    Attributes:
        var_name: Bound variable name.
        value:    Expression bound to the variable.
        body:     Body in which the binding is visible.
    """

    var_name: str = ""
    value: TypedExpr = field(default_factory=lambda: Const.zero())
    body: TypedExpr = field(default_factory=lambda: Const.zero())

    def children(self) -> tuple[TypedExpr, ...]:
        return (self.value, self.body)

    def map_children(self, fn: Callable[[TypedExpr], TypedExpr]) -> LetExpr:
        return LetExpr(ty=self.ty, var_name=self.var_name, value=fn(self.value), body=fn(self.body))

    def __str__(self) -> str:
        return f"(let {self.var_name} = {self.value} in {self.body})"


@dataclass(frozen=True, slots=True)
class SumExpr(TypedExpr):
    """Bounded summation: Σ_{var=lo}^{hi} body.

    Attributes:
        var_name: Summation index variable.
        lo:       Lower bound (inclusive).
        hi:       Upper bound (inclusive).
        body:     Summand expression.
    """

    var_name: str = "i"
    lo: TypedExpr = field(default_factory=lambda: Const.int_(0))
    hi: TypedExpr = field(default_factory=lambda: Const.int_(0))
    body: TypedExpr = field(default_factory=lambda: Const.zero())

    def children(self) -> tuple[TypedExpr, ...]:
        return (self.lo, self.hi, self.body)

    def map_children(self, fn: Callable[[TypedExpr], TypedExpr]) -> SumExpr:
        return SumExpr(
            ty=self.ty, var_name=self.var_name,
            lo=fn(self.lo), hi=fn(self.hi), body=fn(self.body),
        )

    def __str__(self) -> str:
        return f"Σ({self.var_name}={self.lo}..{self.hi}, {self.body})"


# ═══════════════════════════════════════════════════════════════════════════
# PRIVACY BUDGETS
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class PrivacyBudget(ABC):
    """Abstract base for privacy budget specifications.

    Each variant encodes the parameters of a particular DP notion
    and provides comparison / composition operations.
    """

    notion: PrivacyNotion

    @abstractmethod
    def is_satisfied_by(self, cost: PrivacyBudget) -> bool:
        """Return True if *cost* fits within this budget."""
        ...

    @abstractmethod
    def compose(self, other: PrivacyBudget) -> PrivacyBudget:
        """Return the composed budget (sequential composition)."""
        ...

    @abstractmethod
    def to_approx_dp(self, delta: float | None = None) -> tuple[float, float]:
        """Convert to an (ε, δ) guarantee (possibly lossy)."""
        ...


@dataclass(frozen=True, slots=True)
class PureBudget(PrivacyBudget):
    """Pure (ε)-differential privacy budget.

    Attributes:
        epsilon: Privacy parameter ε ≥ 0.
    """

    epsilon: float = 0.0
    notion: PrivacyNotion = field(default=PrivacyNotion.PURE_DP, init=False)

    def __post_init__(self) -> None:
        if self.epsilon < 0:
            raise ValueError(f"epsilon must be ≥ 0, got {self.epsilon}")

    def is_satisfied_by(self, cost: PrivacyBudget) -> bool:
        if isinstance(cost, PureBudget):
            return cost.epsilon <= self.epsilon
        eps, _ = cost.to_approx_dp()
        return eps <= self.epsilon

    def compose(self, other: PrivacyBudget) -> PrivacyBudget:
        if isinstance(other, PureBudget):
            return PureBudget(epsilon=self.epsilon + other.epsilon)
        return other.compose(self)

    def to_approx_dp(self, delta: float | None = None) -> tuple[float, float]:
        return (self.epsilon, 0.0)

    def __str__(self) -> str:
        return f"ε={self.epsilon}"

    def __repr__(self) -> str:
        return f"PureBudget(epsilon={self.epsilon})"


@dataclass(frozen=True, slots=True)
class ApproxBudget(PrivacyBudget):
    """Approximate (ε, δ)-differential privacy budget.

    Attributes:
        epsilon: Privacy parameter ε ≥ 0.
        delta:   Failure probability δ ∈ [0, 1].
    """

    epsilon: float = 0.0
    delta: float = 0.0
    notion: PrivacyNotion = field(default=PrivacyNotion.APPROX_DP, init=False)

    def __post_init__(self) -> None:
        if self.epsilon < 0:
            raise ValueError(f"epsilon must be ≥ 0, got {self.epsilon}")
        if not (0 <= self.delta <= 1):
            raise ValueError(f"delta must be in [0,1], got {self.delta}")

    def is_satisfied_by(self, cost: PrivacyBudget) -> bool:
        eps, delt = cost.to_approx_dp(delta=self.delta)
        return eps <= self.epsilon and delt <= self.delta

    def compose(self, other: PrivacyBudget) -> PrivacyBudget:
        eps_o, del_o = other.to_approx_dp()
        return ApproxBudget(
            epsilon=self.epsilon + eps_o,
            delta=self.delta + del_o,
        )

    def to_approx_dp(self, delta: float | None = None) -> tuple[float, float]:
        return (self.epsilon, self.delta)

    def __str__(self) -> str:
        return f"(ε={self.epsilon}, δ={self.delta})"

    def __repr__(self) -> str:
        return f"ApproxBudget(epsilon={self.epsilon}, delta={self.delta})"


@dataclass(frozen=True, slots=True)
class ZCDPBudget(PrivacyBudget):
    """Zero-concentrated differential privacy budget.

    Attributes:
        rho: Concentration parameter ρ ≥ 0.
    """

    rho: float = 0.0
    notion: PrivacyNotion = field(default=PrivacyNotion.ZCDP, init=False)

    def __post_init__(self) -> None:
        if self.rho < 0:
            raise ValueError(f"rho must be ≥ 0, got {self.rho}")

    def is_satisfied_by(self, cost: PrivacyBudget) -> bool:
        if isinstance(cost, ZCDPBudget):
            return cost.rho <= self.rho
        return False

    def compose(self, other: PrivacyBudget) -> PrivacyBudget:
        if isinstance(other, ZCDPBudget):
            return ZCDPBudget(rho=self.rho + other.rho)
        return ApproxBudget(*self.to_approx_dp()).compose(other)

    def to_approx_dp(self, delta: float | None = None) -> tuple[float, float]:
        d = delta if delta is not None else 1e-5
        if d <= 0 or d >= 1:
            d = 1e-5
        eps = self.rho + 2 * math.sqrt(self.rho * math.log(1.0 / d))
        return (eps, d)

    def __str__(self) -> str:
        return f"ρ={self.rho}"


@dataclass(frozen=True, slots=True)
class RDPBudget(PrivacyBudget):
    """Rényi differential privacy budget.

    Attributes:
        alpha:   Rényi order α > 1.
        epsilon: RDP privacy parameter ε ≥ 0.
    """

    alpha: float = 2.0
    epsilon: float = 0.0
    notion: PrivacyNotion = field(default=PrivacyNotion.RDP, init=False)

    def __post_init__(self) -> None:
        if self.alpha <= 1:
            raise ValueError(f"alpha must be > 1, got {self.alpha}")
        if self.epsilon < 0:
            raise ValueError(f"epsilon must be ≥ 0, got {self.epsilon}")

    def is_satisfied_by(self, cost: PrivacyBudget) -> bool:
        if isinstance(cost, RDPBudget) and cost.alpha == self.alpha:
            return cost.epsilon <= self.epsilon
        return False

    def compose(self, other: PrivacyBudget) -> PrivacyBudget:
        if isinstance(other, RDPBudget) and other.alpha == self.alpha:
            return RDPBudget(alpha=self.alpha, epsilon=self.epsilon + other.epsilon)
        return ApproxBudget(*self.to_approx_dp()).compose(other)

    def to_approx_dp(self, delta: float | None = None) -> tuple[float, float]:
        d = delta if delta is not None else 1e-5
        if d <= 0 or d >= 1:
            d = 1e-5
        eps = self.epsilon - math.log(d) / (self.alpha - 1)
        return (max(eps, 0.0), d)

    def __str__(self) -> str:
        return f"(α={self.alpha}, ε={self.epsilon})-RDP"


@dataclass(frozen=True, slots=True)
class FDPBudget(PrivacyBudget):
    """f-differential privacy budget (trade-off function).

    Attributes:
        trade_off_fn: A callable T : [0,1] → [0,1] representing the
                      optimal trade-off function.
    """

    trade_off_fn: Callable[[float], float] = field(default=lambda a: 1.0 - a)
    notion: PrivacyNotion = field(default=PrivacyNotion.FDP, init=False)

    def is_satisfied_by(self, cost: PrivacyBudget) -> bool:
        # Point-wise comparison at a grid
        if not isinstance(cost, FDPBudget):
            return False
        for alpha in [i / 100 for i in range(101)]:
            if cost.trade_off_fn(alpha) > self.trade_off_fn(alpha) + 1e-9:
                return False
        return True

    def compose(self, other: PrivacyBudget) -> PrivacyBudget:
        if isinstance(other, FDPBudget):
            import numpy as np

            f = self.trade_off_fn
            g = other.trade_off_fn
            grid_points = 1001
            m = grid_points - 1

            alphas = np.linspace(0.0, 1.0, grid_points)
            f_v = np.array([f(float(a)) for a in alphas])
            g_v = np.array([g(float(a)) for a in alphas])

            h = 1.0 / m
            # Degenerate case: T(0)≈0 means no privacy (PLD undefined)
            f_dec_raw = -np.diff(f_v)
            g_dec_raw = -np.diff(g_v)
            if np.sum(np.maximum(f_dec_raw, 0.0)) < 1e-12 or np.sum(np.maximum(g_dec_raw, 0.0)) < 1e-12:
                return FDPBudget(trade_off_fn=lambda a: 0.0)
            # Discrete PLD: H1 masses are decrements of trade-off function
            f_dec = np.maximum(f_dec_raw, 1e-300)
            g_dec = np.maximum(g_dec_raw, 1e-300)
            # Privacy losses
            f_lam = np.log(f_dec / h)
            g_lam = np.log(g_dec / h)
            # Combined PLD over all pairs
            lam_all = (f_lam[:, None] + g_lam[None, :]).ravel()
            h1_all = (f_dec[:, None] * g_dec[None, :]).ravel()
            # Sort by decreasing privacy loss (Neyman-Pearson ordering)
            order = np.argsort(-lam_all)
            h1_sorted = h1_all[order]
            h0_each = h * h
            cum_alpha = np.arange(1, len(order) + 1) * h0_each
            cum_power = np.cumsum(h1_sorted)
            cum_alpha = np.concatenate([[0.0], cum_alpha])
            cum_power = np.concatenate([[0.0], cum_power])
            # Interpolate trade-off on output grid: T(α) = 1 - power(α)
            power_at_grid = np.interp(alphas, cum_alpha, cum_power)
            result = np.clip(1.0 - power_at_grid, 0.0, 1.0)

            cached_values = result.tolist()
            n = grid_points

            def composed(alpha: float) -> float:
                alpha = max(0.0, min(1.0, alpha))
                pos = alpha * (n - 1)
                lo = int(pos)
                if lo >= n - 1:
                    return cached_values[-1]
                frac = pos - lo
                return cached_values[lo] + frac * (cached_values[lo + 1] - cached_values[lo])

            return FDPBudget(trade_off_fn=composed)
        return ApproxBudget(*self.to_approx_dp()).compose(other)

    def to_approx_dp(self, delta: float | None = None) -> tuple[float, float]:
        d = delta if delta is not None else 1e-5
        beta = self.trade_off_fn(0.0)
        if beta <= d:
            return (0.0, d)
        # Find ε via binary search on 1 - trade_off(e^{-ε})
        lo, hi = 0.0, 50.0
        for _ in range(100):
            mid = (lo + hi) / 2
            alpha_test = math.exp(-mid)
            if 1.0 - self.trade_off_fn(alpha_test) > d:
                lo = mid
            else:
                hi = mid
        return (hi, d)

    def __str__(self) -> str:
        return "f-DP(trade_off_fn)"


@dataclass(frozen=True, slots=True)
class GDPBudget(PrivacyBudget):
    """Gaussian differential privacy budget.

    Attributes:
        mu: GDP parameter μ ≥ 0.
    """

    mu: float = 0.0
    notion: PrivacyNotion = field(default=PrivacyNotion.GDP, init=False)

    def __post_init__(self) -> None:
        if self.mu < 0:
            raise ValueError(f"mu must be ≥ 0, got {self.mu}")

    def is_satisfied_by(self, cost: PrivacyBudget) -> bool:
        if isinstance(cost, GDPBudget):
            return cost.mu <= self.mu
        return False

    def compose(self, other: PrivacyBudget) -> PrivacyBudget:
        if isinstance(other, GDPBudget):
            # Central limit theorem composition
            return GDPBudget(mu=math.sqrt(self.mu ** 2 + other.mu ** 2))
        return ApproxBudget(*self.to_approx_dp()).compose(other)

    def to_approx_dp(self, delta: float | None = None) -> tuple[float, float]:
        from dpcegar.utils.math_utils import phi, phi_inv
        d = delta if delta is not None else 1e-5
        if d <= 0 or d >= 1:
            d = 1e-5
        eps = self.mu * phi_inv(1.0 - d) + 0.5 * self.mu ** 2
        return (max(eps, 0.0), d)

    def __str__(self) -> str:
        return f"μ={self.mu}-GDP"


# ═══════════════════════════════════════════════════════════════════════════
# EXPRESSION UTILITIES (private)
# ═══════════════════════════════════════════════════════════════════════════


class _FreeVarCollectorExpr(ExprVisitor[None]):
    """Collect free variable names from an expression tree."""

    def __init__(self) -> None:
        self.vars: set[str] = set()

    def generic_visit(self, expr: TypedExpr) -> None:
        for child in expr.children():
            self.visit(child)

    def visit_Var(self, expr: Var) -> None:
        self.vars.add(expr.name)

    def visit_LetExpr(self, expr: LetExpr) -> None:
        self.visit(expr.value)
        inner = _FreeVarCollectorExpr()
        inner.visit(expr.body)
        inner.vars.discard(expr.var_name)
        self.vars.update(inner.vars)

    def visit_SumExpr(self, expr: SumExpr) -> None:
        self.visit(expr.lo)
        self.visit(expr.hi)
        inner = _FreeVarCollectorExpr()
        inner.visit(expr.body)
        inner.vars.discard(expr.var_name)
        self.vars.update(inner.vars)


class _ExprSubstituterImpl(ExprVisitor[TypedExpr]):
    """Substitute free variables according to a mapping."""

    def __init__(self, mapping: dict[str, TypedExpr]) -> None:
        self.mapping = mapping

    def generic_visit(self, expr: TypedExpr) -> TypedExpr:
        return expr.map_children(lambda c: self.visit(c))

    def visit_Var(self, expr: Var) -> TypedExpr:
        return self.mapping.get(expr.name, expr)

    def visit_LetExpr(self, expr: LetExpr) -> TypedExpr:
        new_value = self.visit(expr.value)
        inner_mapping = {k: v for k, v in self.mapping.items() if k != expr.var_name}
        inner_sub = _ExprSubstituterImpl(inner_mapping)
        new_body = inner_sub.visit(expr.body)
        return LetExpr(ty=expr.ty, var_name=expr.var_name, value=new_value, body=new_body)

    def visit_SumExpr(self, expr: SumExpr) -> TypedExpr:
        new_lo = self.visit(expr.lo)
        new_hi = self.visit(expr.hi)
        inner_mapping = {k: v for k, v in self.mapping.items() if k != expr.var_name}
        inner_sub = _ExprSubstituterImpl(inner_mapping)
        new_body = inner_sub.visit(expr.body)
        return SumExpr(
            ty=expr.ty, var_name=expr.var_name,
            lo=new_lo, hi=new_hi, body=new_body,
        )


# ═══════════════════════════════════════════════════════════════════════════
# EXPRESSION SIMPLIFICATION
# ═══════════════════════════════════════════════════════════════════════════

_BINOP_EVAL: dict[BinOpKind, Callable[[Any, Any], Any]] = {
    BinOpKind.ADD: lambda a, b: a + b,
    BinOpKind.SUB: lambda a, b: a - b,
    BinOpKind.MUL: lambda a, b: a * b,
    BinOpKind.DIV: lambda a, b: a / b if b != 0 else None,
    BinOpKind.MOD: lambda a, b: a % b if b != 0 else None,
    BinOpKind.POW: lambda a, b: a ** b,
    BinOpKind.AND: lambda a, b: a and b,
    BinOpKind.OR: lambda a, b: a or b,
    BinOpKind.EQ: lambda a, b: a == b,
    BinOpKind.NEQ: lambda a, b: a != b,
    BinOpKind.LT: lambda a, b: a < b,
    BinOpKind.LE: lambda a, b: a <= b,
    BinOpKind.GT: lambda a, b: a > b,
    BinOpKind.GE: lambda a, b: a >= b,
}


def _simplify(expr: TypedExpr) -> TypedExpr:
    """Apply basic algebraic simplifications bottom-up.

    Simplification rules:
      - Constant folding: evaluate ops on two constants
      - Additive identity: x + 0 → x, 0 + x → x
      - Multiplicative identity: x * 1 → x, 1 * x → x
      - Multiplicative annihilator: x * 0 → 0
      - Double negation: --x → x
      - Boolean identities: !true → false, x && true → x, x || false → x
      - Exp/Log cancellation: ln(exp(x)) → x, exp(ln(x)) → x
    """
    # Recursively simplify children first
    expr = expr.map_children(_simplify)

    # -- Constant folding for BinOp --
    if isinstance(expr, BinOp):
        left, right = expr.left, expr.right
        if isinstance(left, Const) and isinstance(right, Const):
            fn = _BINOP_EVAL.get(expr.op)
            if fn is not None:
                result = fn(left.value, right.value)
                if result is not None:
                    if isinstance(result, bool):
                        return Const.bool_(result)
                    elif isinstance(result, int) and expr.ty == IRType.INT:
                        return Const.int_(result)
                    elif isinstance(result, (int, float)):
                        return Const.real(float(result))

        # Additive identity
        if expr.op == BinOpKind.ADD:
            if isinstance(right, Const) and right.is_zero:
                return left
            if isinstance(left, Const) and left.is_zero:
                return right

        # Subtractive identity
        if expr.op == BinOpKind.SUB:
            if isinstance(right, Const) and right.is_zero:
                return left

        # Multiplicative identity
        if expr.op == BinOpKind.MUL:
            if isinstance(right, Const) and right.is_one:
                return left
            if isinstance(left, Const) and left.is_one:
                return right
            if isinstance(right, Const) and right.is_zero:
                return Const.zero(expr.ty)
            if isinstance(left, Const) and left.is_zero:
                return Const.zero(expr.ty)

        # Power identities
        if expr.op == BinOpKind.POW:
            if isinstance(right, Const) and right.is_zero:
                return Const.one(expr.ty)
            if isinstance(right, Const) and right.is_one:
                return left

        # Boolean identities
        if expr.op == BinOpKind.AND:
            if isinstance(left, Const) and left.value is True:
                return right
            if isinstance(right, Const) and right.value is True:
                return left
            if isinstance(left, Const) and left.value is False:
                return Const.bool_(False)
            if isinstance(right, Const) and right.value is False:
                return Const.bool_(False)

        if expr.op == BinOpKind.OR:
            if isinstance(left, Const) and left.value is False:
                return right
            if isinstance(right, Const) and right.value is False:
                return left
            if isinstance(left, Const) and left.value is True:
                return Const.bool_(True)
            if isinstance(right, Const) and right.value is True:
                return Const.bool_(True)

    # -- Double negation --
    if isinstance(expr, UnaryOp) and expr.op == UnaryOpKind.NEG:
        if isinstance(expr.operand, UnaryOp) and expr.operand.op == UnaryOpKind.NEG:
            return expr.operand.operand
        if isinstance(expr.operand, Const):
            if expr.ty == IRType.INT:
                return Const.int_(-int(expr.operand.value))
            return Const.real(-float(expr.operand.value))

    if isinstance(expr, UnaryOp) and expr.op == UnaryOpKind.NOT:
        if isinstance(expr.operand, Const) and isinstance(expr.operand.value, bool):
            return Const.bool_(not expr.operand.value)
        if isinstance(expr.operand, UnaryOp) and expr.operand.op == UnaryOpKind.NOT:
            return expr.operand.operand

    # -- Exp/Log cancellation --
    if isinstance(expr, Log) and isinstance(expr.operand, Exp):
        return expr.operand.operand
    if isinstance(expr, Exp) and isinstance(expr.operand, Log):
        return expr.operand.operand

    # -- Constant folding for unary math --
    if isinstance(expr, Abs) and isinstance(expr.operand, Const):
        return Const.real(abs(float(expr.operand.value)))
    if isinstance(expr, Log) and isinstance(expr.operand, Const):
        v = float(expr.operand.value)
        if v > 0:
            return Const.real(math.log(v))
    if isinstance(expr, Exp) and isinstance(expr.operand, Const):
        return Const.real(math.exp(float(expr.operand.value)))
    if isinstance(expr, Sqrt) and isinstance(expr.operand, Const):
        v = float(expr.operand.value)
        if v >= 0:
            return Const.real(math.sqrt(v))

    # -- Conditional simplification --
    if isinstance(expr, Cond) and isinstance(expr.condition, Const):
        if expr.condition.value is True:
            return expr.true_expr
        if expr.condition.value is False:
            return expr.false_expr

    return expr


# ═══════════════════════════════════════════════════════════════════════════
# EXPRESSION BUILDER HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def var(name: str, ty: IRType = IRType.REAL, version: int | None = None) -> Var:
    """Shorthand constructor for :class:`Var`."""
    return Var(ty=ty, name=name, version=version)


def const_int(value: int) -> Const:
    """Shorthand for integer constant."""
    return Const.int_(value)


def const_real(value: float) -> Const:
    """Shorthand for real constant."""
    return Const.real(value)


def const_bool(value: bool) -> Const:
    """Shorthand for boolean constant."""
    return Const.bool_(value)


def add(left: TypedExpr, right: TypedExpr) -> BinOp:
    """Create an addition expression."""
    return BinOp(ty=left.ty, op=BinOpKind.ADD, left=left, right=right)


def sub(left: TypedExpr, right: TypedExpr) -> BinOp:
    """Create a subtraction expression."""
    return BinOp(ty=left.ty, op=BinOpKind.SUB, left=left, right=right)


def mul(left: TypedExpr, right: TypedExpr) -> BinOp:
    """Create a multiplication expression."""
    return BinOp(ty=left.ty, op=BinOpKind.MUL, left=left, right=right)


def div(left: TypedExpr, right: TypedExpr) -> BinOp:
    """Create a division expression."""
    return BinOp(ty=left.ty, op=BinOpKind.DIV, left=left, right=right)


def lt(left: TypedExpr, right: TypedExpr) -> BinOp:
    """Create a less-than comparison."""
    return BinOp(ty=IRType.BOOL, op=BinOpKind.LT, left=left, right=right)


def le(left: TypedExpr, right: TypedExpr) -> BinOp:
    """Create a less-than-or-equal comparison."""
    return BinOp(ty=IRType.BOOL, op=BinOpKind.LE, left=left, right=right)


def eq(left: TypedExpr, right: TypedExpr) -> BinOp:
    """Create an equality comparison."""
    return BinOp(ty=IRType.BOOL, op=BinOpKind.EQ, left=left, right=right)


def and_(left: TypedExpr, right: TypedExpr) -> BinOp:
    """Create a logical AND."""
    return BinOp(ty=IRType.BOOL, op=BinOpKind.AND, left=left, right=right)


def or_(left: TypedExpr, right: TypedExpr) -> BinOp:
    """Create a logical OR."""
    return BinOp(ty=IRType.BOOL, op=BinOpKind.OR, left=left, right=right)


def neg(operand: TypedExpr) -> UnaryOp:
    """Create a negation expression."""
    return UnaryOp(ty=operand.ty, op=UnaryOpKind.NEG, operand=operand)


def not_(operand: TypedExpr) -> UnaryOp:
    """Create a logical NOT expression."""
    return UnaryOp(ty=IRType.BOOL, op=UnaryOpKind.NOT, operand=operand)
