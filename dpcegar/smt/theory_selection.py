"""Theory selection for SMT encoding of privacy verification formulas.

Analyses a Z3 formula (or our IR expressions) to determine the minimal
SMT theory required, then selects the appropriate solver configuration.

Theory hierarchy (weakest → strongest):
    QF_LIA  – quantifier-free linear integer arithmetic
    QF_LRA  – quantifier-free linear real arithmetic
    QF_LIRA – mixed linear integer/real
    QF_NIA  – quantifier-free nonlinear integer arithmetic
    QF_NRA  – quantifier-free nonlinear real arithmetic
    QF_NIRA – mixed nonlinear integer/real
    QF_NRAT – nonlinear real arithmetic with transcendentals (dReal)

Choosing the weakest sufficient theory enables faster decision procedures.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

try:
    import z3
except ImportError:  # pragma: no cover
    z3 = None  # type: ignore[assignment]

from dpcegar.ir.types import (
    Abs,
    BinOp,
    BinOpKind,
    Const,
    Cond,
    Exp,
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
    FuncCall,
)


# ═══════════════════════════════════════════════════════════════════════════
# THEORY ENUM
# ═══════════════════════════════════════════════════════════════════════════


class SMTTheory(enum.Enum):
    """SMT theories ordered from weakest to strongest."""

    QF_LIA = "QF_LIA"
    QF_LRA = "QF_LRA"
    QF_LIRA = "QF_LIRA"
    QF_NIA = "QF_NIA"
    QF_NRA = "QF_NRA"
    QF_NIRA = "QF_NIRA"
    QF_NRAT = "QF_NRAT"

    @property
    def is_linear(self) -> bool:
        """Return True if the theory only handles linear arithmetic."""
        return self in (SMTTheory.QF_LIA, SMTTheory.QF_LRA, SMTTheory.QF_LIRA)

    @property
    def has_integers(self) -> bool:
        """Return True if the theory supports integer variables."""
        return self in (
            SMTTheory.QF_LIA, SMTTheory.QF_LIRA,
            SMTTheory.QF_NIA, SMTTheory.QF_NIRA,
        )

    @property
    def has_reals(self) -> bool:
        """Return True if the theory supports real variables."""
        return self not in (SMTTheory.QF_LIA, SMTTheory.QF_NIA)

    @property
    def supports_transcendentals(self) -> bool:
        """Return True if the theory natively handles exp/ln/Φ."""
        return self == SMTTheory.QF_NRAT

    def __lt__(self, other: SMTTheory) -> bool:
        """Compare theories by strength."""
        order = list(SMTTheory)
        return order.index(self) < order.index(other)

    def __le__(self, other: SMTTheory) -> bool:
        """Compare theories by strength."""
        return self == other or self < other


# ═══════════════════════════════════════════════════════════════════════════
# ANALYSIS RESULT
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class TheoryAnalysisResult:
    """Result of analysing a formula's theory requirements.

    Attributes:
        theory:             The minimal sufficient theory.
        has_multiplication: True if variable-variable multiplication exists.
        has_division:       True if non-constant division exists.
        has_power:          True if non-trivial exponentiation exists.
        has_transcendental: True if exp/ln/Φ/Φ⁻¹ are used.
        has_integers:       True if integer-typed variables appear.
        has_reals:          True if real-typed variables appear.
        has_abs:            True if absolute value is used.
        num_variables:      Number of distinct variables.
        num_nonlinear:      Count of nonlinear sub-expressions.
        num_transcendental: Count of transcendental sub-expressions.
        fallback_reason:    Reason for fallback to stronger theory (if any).
    """

    theory: SMTTheory = SMTTheory.QF_LRA
    has_multiplication: bool = False
    has_division: bool = False
    has_power: bool = False
    has_transcendental: bool = False
    has_integers: bool = False
    has_reals: bool = True
    has_abs: bool = False
    num_variables: int = 0
    num_nonlinear: int = 0
    num_transcendental: int = 0
    fallback_reason: str = ""

    def summary(self) -> str:
        """Return a human-readable summary."""
        parts = [f"theory={self.theory.value}"]
        if self.has_multiplication:
            parts.append("nonlinear_mul")
        if self.has_division:
            parts.append("nonlinear_div")
        if self.has_transcendental:
            parts.append("transcendental")
        if self.has_integers:
            parts.append("integers")
        parts.append(f"vars={self.num_variables}")
        return ", ".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# THEORY ANALYSER
# ═══════════════════════════════════════════════════════════════════════════


class TheoryAnalyzer:
    """Analyse IR expressions or Z3 formulas to determine minimal theory.

    Walks the expression tree, tracking which features are used, then
    selects the weakest SMT theory that can express the formula.

    Usage::

        analyzer = TheoryAnalyzer()
        result = analyzer.analyze_expr(expr)
        print(result.theory)  # e.g. SMTTheory.QF_NRA
    """

    def __init__(self) -> None:
        self._has_mul = False
        self._has_div = False
        self._has_pow = False
        self._has_transcendental = False
        self._has_int = False
        self._has_real = False
        self._has_abs = False
        self._variables: set[str] = set()
        self._nonlinear_count = 0
        self._transcendental_count = 0

    def _reset(self) -> None:
        """Reset analysis state."""
        self._has_mul = False
        self._has_div = False
        self._has_pow = False
        self._has_transcendental = False
        self._has_int = False
        self._has_real = False
        self._has_abs = False
        self._variables = set()
        self._nonlinear_count = 0
        self._transcendental_count = 0

    def analyze_expr(self, expr: TypedExpr) -> TheoryAnalysisResult:
        """Analyse an IR expression tree.

        Args:
            expr: The expression to analyse.

        Returns:
            TheoryAnalysisResult with the minimal theory and feature flags.
        """
        self._reset()
        self._walk_expr(expr)
        return self._build_result()

    def analyze_exprs(self, exprs: Sequence[TypedExpr]) -> TheoryAnalysisResult:
        """Analyse multiple IR expressions (e.g. all path constraints).

        Args:
            exprs: Expressions to analyse.

        Returns:
            TheoryAnalysisResult for the combined formula.
        """
        self._reset()
        for expr in exprs:
            self._walk_expr(expr)
        return self._build_result()

    def analyze_z3(self, formula: Any) -> TheoryAnalysisResult:
        """Analyse a Z3 formula to determine its theory requirements.

        Walks the Z3 AST checking for nonlinear arithmetic, integer
        variables, and transcendental-like patterns.

        Args:
            formula: A Z3 expression (z3.ExprRef or z3.BoolRef).

        Returns:
            TheoryAnalysisResult.
        """
        self._reset()
        if z3 is not None:
            self._walk_z3(formula)
        return self._build_result()

    # ── IR expression walker ─────────────────────────────────────────────

    def _walk_expr(self, expr: TypedExpr) -> None:
        """Recursively walk an IR expression tree."""
        if isinstance(expr, Var):
            self._variables.add(expr.ssa_name)
            if expr.ty == IRType.INT:
                self._has_int = True
            elif expr.ty == IRType.REAL:
                self._has_real = True

        elif isinstance(expr, Const):
            pass  # constants don't affect theory choice

        elif isinstance(expr, BinOp):
            if expr.op == BinOpKind.MUL:
                if not self._is_const_or_param(expr.left) and not self._is_const_or_param(expr.right):
                    self._has_mul = True
                    self._nonlinear_count += 1
            elif expr.op == BinOpKind.DIV:
                if not self._is_const_or_param(expr.right):
                    self._has_div = True
                    self._nonlinear_count += 1
            elif expr.op == BinOpKind.POW:
                self._has_pow = True
                self._nonlinear_count += 1
            elif expr.op == BinOpKind.MOD:
                self._has_int = True
            for child in expr.children():
                self._walk_expr(child)
            return

        elif isinstance(expr, UnaryOp):
            pass  # negation and not don't affect theory

        elif isinstance(expr, Abs):
            self._has_abs = True

        elif isinstance(expr, (Max, Min)):
            pass  # encoded as If-then-else, no theory impact

        elif isinstance(expr, (Log, Exp)):
            self._has_transcendental = True
            self._transcendental_count += 1

        elif isinstance(expr, Sqrt):
            self._has_pow = True
            self._nonlinear_count += 1

        elif isinstance(expr, (Phi, PhiInv)):
            self._has_transcendental = True
            self._transcendental_count += 1

        elif isinstance(expr, FuncCall):
            if expr.name in ("exp", "log", "ln", "phi", "phi_inv", "sqrt"):
                self._has_transcendental = True
                self._transcendental_count += 1

        elif isinstance(expr, SumExpr):
            self._has_int = True

        # Recurse into children
        for child in expr.children():
            self._walk_expr(child)

    def _is_const_or_param(self, expr: TypedExpr) -> bool:
        """Check if an expression is a constant (no free variables).

        Args:
            expr: Expression to check.

        Returns:
            True if the expression has no free variables.
        """
        if isinstance(expr, Const):
            return True
        return len(expr.free_vars()) == 0

    # ── Z3 AST walker ────────────────────────────────────────────────────

    def _walk_z3(self, expr: Any, depth: int = 0) -> None:
        """Recursively walk a Z3 AST.

        Args:
            expr:  Z3 expression.
            depth: Current recursion depth (for cycle protection).
        """
        if depth > 500:
            return

        if z3.is_int(expr):
            self._has_int = True
        if z3.is_real(expr):
            self._has_real = True

        if z3.is_const(expr) and expr.decl().kind() == z3.Z3_OP_UNINTERPRETED:
            name = str(expr)
            self._variables.add(name)
            if z3.is_int(expr):
                self._has_int = True
            else:
                self._has_real = True
            return

        kind = expr.decl().kind() if hasattr(expr, "decl") else None

        if kind == z3.Z3_OP_MUL:
            # Check if it's variable * variable (nonlinear)
            children = [expr.arg(i) for i in range(expr.num_args())]
            non_const = sum(1 for c in children if not z3.is_rational_value(c) and not z3.is_int_value(c))
            if non_const >= 2:
                self._has_mul = True
                self._nonlinear_count += 1

        elif kind == z3.Z3_OP_DIV or kind == z3.Z3_OP_IDIV:
            if expr.num_args() >= 2:
                denom = expr.arg(1)
                if not z3.is_rational_value(denom) and not z3.is_int_value(denom):
                    self._has_div = True
                    self._nonlinear_count += 1

        elif kind == z3.Z3_OP_POWER:
            self._has_pow = True
            self._nonlinear_count += 1

        # Recurse
        if hasattr(expr, "num_args"):
            for i in range(expr.num_args()):
                self._walk_z3(expr.arg(i), depth + 1)

    # ── Result builder ───────────────────────────────────────────────────

    def _build_result(self) -> TheoryAnalysisResult:
        """Build the analysis result from collected flags.

        Returns:
            TheoryAnalysisResult with the minimal theory.
        """
        theory = self._select_theory()
        fallback = ""

        if self._has_transcendental and theory != SMTTheory.QF_NRAT:
            fallback = (
                "Transcendental functions present but polynomial "
                "approximations will be used instead of QF_NRAT"
            )

        return TheoryAnalysisResult(
            theory=theory,
            has_multiplication=self._has_mul,
            has_division=self._has_div,
            has_power=self._has_pow,
            has_transcendental=self._has_transcendental,
            has_integers=self._has_int,
            has_reals=self._has_real,
            has_abs=self._has_abs,
            num_variables=len(self._variables),
            num_nonlinear=self._nonlinear_count,
            num_transcendental=self._transcendental_count,
            fallback_reason=fallback,
        )

    def _select_theory(self) -> SMTTheory:
        """Select the minimal SMT theory based on collected flags.

        Returns:
            The weakest sufficient SMTTheory.
        """
        is_nonlinear = self._has_mul or self._has_div or self._has_pow
        has_both = self._has_int and self._has_real

        if self._has_transcendental:
            # Transcendentals require QF_NRAT (dReal) or approximation into NRA
            return SMTTheory.QF_NRAT

        if is_nonlinear:
            if has_both:
                return SMTTheory.QF_NIRA
            if self._has_int and not self._has_real:
                return SMTTheory.QF_NIA
            return SMTTheory.QF_NRA

        # Linear
        if has_both:
            return SMTTheory.QF_LIRA
        if self._has_int and not self._has_real:
            return SMTTheory.QF_LIA
        return SMTTheory.QF_LRA


# ═══════════════════════════════════════════════════════════════════════════
# AUTO-CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class SolverRecommendation:
    """Recommended solver configuration based on theory analysis.

    Attributes:
        theory:         The SMT theory to declare.
        solver_name:    Recommended solver ('z3', 'cvc5', 'dreal').
        use_portfolio:  Whether to use portfolio solving.
        timeout:        Recommended timeout in milliseconds.
        tactics:        Suggested Z3 tactics to apply before solving.
        use_approx:     Whether to use polynomial approximations.
        notes:          Human-readable notes about the recommendation.
    """

    theory: SMTTheory = SMTTheory.QF_LRA
    solver_name: str = "z3"
    use_portfolio: bool = False
    timeout: int = 30000
    tactics: list[str] = field(default_factory=list)
    use_approx: bool = False
    notes: str = ""


def auto_configure(analysis: TheoryAnalysisResult) -> SolverRecommendation:
    """Select solver configuration based on theory analysis.

    Args:
        analysis: Result from TheoryAnalyzer.

    Returns:
        SolverRecommendation with the best configuration.
    """
    theory = analysis.theory

    if theory == SMTTheory.QF_LRA:
        return SolverRecommendation(
            theory=theory,
            solver_name="z3",
            timeout=10000,
            tactics=["simplify", "solve-eqs", "smt"],
            notes="Linear real arithmetic: fast simplex-based solving.",
        )

    if theory == SMTTheory.QF_LIA:
        return SolverRecommendation(
            theory=theory,
            solver_name="z3",
            timeout=10000,
            tactics=["simplify", "solve-eqs", "smt"],
            notes="Linear integer arithmetic: Omega test / branch-and-bound.",
        )

    if theory == SMTTheory.QF_LIRA:
        return SolverRecommendation(
            theory=theory,
            solver_name="z3",
            timeout=15000,
            tactics=["simplify", "solve-eqs", "smt"],
            notes="Mixed linear arithmetic.",
        )

    if theory in (SMTTheory.QF_NRA, SMTTheory.QF_NIA, SMTTheory.QF_NIRA):
        return SolverRecommendation(
            theory=theory,
            solver_name="z3",
            use_portfolio=True,
            timeout=60000,
            tactics=["simplify", "purify-arith", "nlsat"],
            notes="Nonlinear arithmetic: nlsat + portfolio recommended.",
        )

    if theory == SMTTheory.QF_NRAT:
        return SolverRecommendation(
            theory=SMTTheory.QF_NRA,
            solver_name="z3",
            use_portfolio=True,
            timeout=120000,
            tactics=["simplify", "nlsat"],
            use_approx=True,
            notes=(
                "Transcendentals present: using polynomial approximations "
                "with QF_NRA. Consider dReal for delta-decidable queries."
            ),
        )

    return SolverRecommendation(
        theory=theory,
        solver_name="z3",
        timeout=30000,
        notes="Default configuration.",
    )


# ═══════════════════════════════════════════════════════════════════════════
# FALLBACK LOGIC
# ═══════════════════════════════════════════════════════════════════════════


class TheoryFallbackChain:
    """Try theories in order from weakest to strongest.

    When the weakest sufficient theory times out or returns unknown,
    this class steps up to the next stronger theory.

    Usage::

        chain = TheoryFallbackChain(start=SMTTheory.QF_LRA)
        while chain.has_next():
            theory = chain.current()
            # try solving with this theory
            if solved:
                break
            chain.step_up()

    Args:
        start: Initial (weakest) theory to try.
    """

    _ORDER: list[SMTTheory] = [
        SMTTheory.QF_LRA,
        SMTTheory.QF_NRA,
        SMTTheory.QF_NIRA,
        SMTTheory.QF_NRAT,
    ]

    def __init__(self, start: SMTTheory = SMTTheory.QF_LRA) -> None:
        try:
            self._idx = self._ORDER.index(start)
        except ValueError:
            self._idx = 0

    def current(self) -> SMTTheory:
        """Return the current theory.

        Returns:
            The SMTTheory at the current position.
        """
        return self._ORDER[self._idx]

    def has_next(self) -> bool:
        """Return True if there is a stronger theory to try.

        Returns:
            True if step_up() can be called.
        """
        return self._idx < len(self._ORDER) - 1

    def step_up(self) -> SMTTheory:
        """Move to the next stronger theory.

        Returns:
            The new (stronger) theory.

        Raises:
            StopIteration: If already at the strongest theory.
        """
        if not self.has_next():
            raise StopIteration("Already at strongest theory")
        self._idx += 1
        return self._ORDER[self._idx]

    def reset(self, theory: SMTTheory | None = None) -> None:
        """Reset to a specific theory (or the weakest).

        Args:
            theory: Theory to reset to (default: QF_LRA).
        """
        if theory is None:
            self._idx = 0
        else:
            try:
                self._idx = self._ORDER.index(theory)
            except ValueError:
                self._idx = 0
