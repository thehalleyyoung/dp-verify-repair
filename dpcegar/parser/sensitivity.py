"""Sensitivity analysis for DPImp mechanisms.

Computes symbolic sensitivity bounds through abstract interpretation
over MechIR trees.  Supports L1, L2, and L∞ sensitivity norms, as well
as sequential and parallel composition.

Classes
-------
SensitivityNorm   – which norm is used (L1, L2, Linf)
SymbolicSens      – symbolic sensitivity value
SensitivityResult – complete sensitivity analysis result
SensitivityCert   – exportable sensitivity certificate
SensitivityAnalyzer – main analysis driver
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from dpcegar.ir.types import (
    BinOp,
    BinOpKind,
    Const,
    FuncCall,
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
    IRNodeVisitorBase,
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
from dpcegar.utils.errors import SourceLoc


# ═══════════════════════════════════════════════════════════════════════════
# SENSITIVITY NORM
# ═══════════════════════════════════════════════════════════════════════════


class SensitivityNorm(Enum):
    """Which ℓ-norm is used for sensitivity measurement."""

    L1 = auto()     # ℓ₁ (Manhattan) norm
    L2 = auto()     # ℓ₂ (Euclidean) norm
    LINF = auto()   # ℓ∞ (Chebyshev) norm

    def __str__(self) -> str:
        names = {SensitivityNorm.L1: "L1", SensitivityNorm.L2: "L2", SensitivityNorm.LINF: "L∞"}
        return names.get(self, self.name)


# ═══════════════════════════════════════════════════════════════════════════
# SYMBOLIC SENSITIVITY
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class SymbolicSens:
    """Symbolic sensitivity value.

    Represents the sensitivity of a variable or expression as either
    a concrete bound or a symbolic expression.

    Attributes:
        value:      Concrete numeric bound (if known).
        symbolic:   Symbolic expression string (if not concrete).
        is_bounded: True if sensitivity is finitely bounded.
        depends_on: Variables this sensitivity depends on.
    """

    value: float | None = None
    symbolic: str | None = None
    is_bounded: bool = True
    depends_on: frozenset[str] = field(default_factory=frozenset)

    @classmethod
    def zero(cls) -> SymbolicSens:
        """Zero sensitivity (constant value)."""
        return cls(value=0.0)

    @classmethod
    def constant(cls, val: float) -> SymbolicSens:
        """Concrete sensitivity value."""
        return cls(value=val)

    @classmethod
    def unbounded(cls) -> SymbolicSens:
        """Unbounded sensitivity."""
        return cls(value=float("inf"), is_bounded=False)

    @classmethod
    def from_var(cls, name: str) -> SymbolicSens:
        """Sensitivity that depends on a variable."""
        return cls(symbolic=name, depends_on=frozenset({name}))

    @property
    def is_zero(self) -> bool:
        """True if this is provably zero sensitivity."""
        return self.value is not None and self.value == 0.0

    @property
    def is_concrete(self) -> bool:
        """True if this has a concrete numeric value."""
        return self.value is not None and self.symbolic is None

    def add(self, other: SymbolicSens) -> SymbolicSens:
        """Sensitivity under addition: Δ(f+g) = Δf + Δg."""
        if self.is_zero:
            return other
        if other.is_zero:
            return self
        if self.is_concrete and other.is_concrete:
            return SymbolicSens.constant(self.value + other.value)  # type: ignore[operator]
        left = str(self)
        right = str(other)
        return SymbolicSens(
            symbolic=f"({left} + {right})",
            depends_on=self.depends_on | other.depends_on,
        )

    def multiply(self, other: SymbolicSens) -> SymbolicSens:
        """Sensitivity under multiplication: Δ(f*c) = |c| * Δf (when c is const)."""
        if self.is_zero or other.is_zero:
            return SymbolicSens.zero()
        if self.is_concrete and other.is_concrete:
            return SymbolicSens.constant(self.value * other.value)  # type: ignore[operator]
        left = str(self)
        right = str(other)
        return SymbolicSens(
            symbolic=f"({left} * {right})",
            depends_on=self.depends_on | other.depends_on,
        )

    def scale(self, factor: float) -> SymbolicSens:
        """Scale sensitivity by a constant factor."""
        if self.is_zero:
            return SymbolicSens.zero()
        if self.is_concrete:
            return SymbolicSens.constant(self.value * abs(factor))  # type: ignore[operator]
        return SymbolicSens(
            symbolic=f"({abs(factor)} * {self})",
            depends_on=self.depends_on,
        )

    def max(self, other: SymbolicSens) -> SymbolicSens:
        """Pointwise maximum of two sensitivities."""
        if self.is_zero:
            return other
        if other.is_zero:
            return self
        if self.is_concrete and other.is_concrete:
            return SymbolicSens.constant(max(self.value, other.value))  # type: ignore[arg-type]
        left = str(self)
        right = str(other)
        return SymbolicSens(
            symbolic=f"max({left}, {right})",
            depends_on=self.depends_on | other.depends_on,
        )

    def sqrt(self) -> SymbolicSens:
        """Square root for L2 sensitivity computation."""
        if self.is_zero:
            return SymbolicSens.zero()
        if self.is_concrete:
            return SymbolicSens.constant(math.sqrt(self.value))  # type: ignore[arg-type]
        return SymbolicSens(
            symbolic=f"√({self})",
            depends_on=self.depends_on,
        )

    def __str__(self) -> str:
        if self.value is not None and self.symbolic is None:
            if self.value == float("inf"):
                return "∞"
            return f"{self.value:g}"
        if self.symbolic is not None:
            return self.symbolic
        return "?"


# ═══════════════════════════════════════════════════════════════════════════
# ANALYSIS RESULTS
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class QuerySensitivity:
    """Sensitivity information for a single query.

    Attributes:
        query_name:  Name of the query function.
        node_id:     IR node ID.
        sensitivity: Declared or computed sensitivity.
        norm:        Which norm was used.
        source_loc:  Source location.
    """

    query_name: str
    node_id: int
    sensitivity: SymbolicSens
    norm: SensitivityNorm = SensitivityNorm.L1
    source_loc: SourceLoc | None = None

    def __str__(self) -> str:
        return f"{self.query_name}: Δ={self.sensitivity} ({self.norm})"


@dataclass
class NoiseInfo:
    """Information about a noise draw and its relationship to queries.

    Attributes:
        target_var:    Variable receiving the noisy value.
        noise_kind:    Distribution family.
        scale:         Scale parameter.
        query_sens:    Sensitivity of the underlying query.
        node_id:       IR node ID.
        source_loc:    Source location.
    """

    target_var: str
    noise_kind: NoiseKind
    scale: SymbolicSens
    query_sens: SymbolicSens | None = None
    node_id: int = 0
    source_loc: SourceLoc | None = None

    def __str__(self) -> str:
        sens = f", Δ={self.query_sens}" if self.query_sens else ""
        return f"{self.target_var} ~ {self.noise_kind}(σ={self.scale}{sens})"


@dataclass
class SensitivityResult:
    """Complete result of sensitivity analysis.

    Attributes:
        mechanism_name:      Name of the mechanism.
        global_sensitivity:  Overall global sensitivity.
        local_sensitivities: Per-query sensitivity bounds.
        noise_info:          Information about noise draws.
        composition:         Composition mode used.
        norm:                Sensitivity norm.
        warnings:            Any warnings generated.
    """

    mechanism_name: str = ""
    global_sensitivity: SymbolicSens = field(default_factory=SymbolicSens.zero)
    local_sensitivities: list[QuerySensitivity] = field(default_factory=list)
    noise_info: list[NoiseInfo] = field(default_factory=list)
    composition: str = "sequential"
    norm: SensitivityNorm = SensitivityNorm.L1
    warnings: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            f"Sensitivity Analysis: {self.mechanism_name}",
            f"  Norm: {self.norm}",
            f"  Composition: {self.composition}",
            f"  Global sensitivity: {self.global_sensitivity}",
        ]
        if self.local_sensitivities:
            lines.append("  Queries:")
            for qs in self.local_sensitivities:
                lines.append(f"    {qs}")
        if self.noise_info:
            lines.append("  Noise draws:")
            for ni in self.noise_info:
                lines.append(f"    {ni}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# SENSITIVITY CERTIFICATE
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class SensitivityCert:
    """Exportable sensitivity certificate.

    Contains all the information needed to verify the sensitivity
    claims of a mechanism.

    Attributes:
        mechanism_name:     Name of the mechanism.
        global_sensitivity: Overall global sensitivity bound.
        norm:               Sensitivity norm.
        queries:            Per-query sensitivity information.
        noise_draws:        Noise draw information.
        composition_type:   Sequential or parallel.
        is_valid:           Whether the certificate is valid.
        validation_notes:   Notes from validation.
    """

    mechanism_name: str = ""
    global_sensitivity: float | None = None
    norm: SensitivityNorm = SensitivityNorm.L1
    queries: list[dict[str, Any]] = field(default_factory=list)
    noise_draws: list[dict[str, Any]] = field(default_factory=list)
    composition_type: str = "sequential"
    is_valid: bool = True
    validation_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "mechanism_name": self.mechanism_name,
            "global_sensitivity": self.global_sensitivity,
            "norm": str(self.norm),
            "queries": self.queries,
            "noise_draws": self.noise_draws,
            "composition_type": self.composition_type,
            "is_valid": self.is_valid,
            "validation_notes": self.validation_notes,
        }

    @classmethod
    def from_result(cls, result: SensitivityResult) -> SensitivityCert:
        """Build a certificate from a sensitivity analysis result."""
        cert = cls(
            mechanism_name=result.mechanism_name,
            norm=result.norm,
            composition_type=result.composition,
        )

        if result.global_sensitivity.is_concrete:
            cert.global_sensitivity = result.global_sensitivity.value

        for qs in result.local_sensitivities:
            entry: dict[str, Any] = {
                "query_name": qs.query_name,
                "norm": str(qs.norm),
            }
            if qs.sensitivity.is_concrete:
                entry["sensitivity"] = qs.sensitivity.value
            else:
                entry["sensitivity_expr"] = str(qs.sensitivity)
            cert.queries.append(entry)

        for ni in result.noise_info:
            entry = {
                "variable": ni.target_var,
                "distribution": str(ni.noise_kind),
            }
            if ni.scale.is_concrete:
                entry["scale"] = ni.scale.value
            if ni.query_sens and ni.query_sens.is_concrete:
                entry["query_sensitivity"] = ni.query_sens.value
            cert.noise_draws.append(entry)

        return cert


# ═══════════════════════════════════════════════════════════════════════════
# SENSITIVITY ANALYZER
# ═══════════════════════════════════════════════════════════════════════════


class SensitivityAnalyzer(IRNodeVisitorBase):
    """Compute sensitivity bounds for a DP mechanism via abstract interpretation.

    Walks the MechIR tree and tracks how database-dependent values flow
    through computations.  Produces a :class:`SensitivityResult` with
    per-query and global sensitivity bounds.

    Usage::

        analyzer = SensitivityAnalyzer(norm=SensitivityNorm.L1)
        result = analyzer.analyze(mechir)
        print(result.global_sensitivity)
    """

    def __init__(
        self,
        norm: SensitivityNorm = SensitivityNorm.L1,
    ) -> None:
        """Initialize the sensitivity analyzer.

        Args:
            norm: Which sensitivity norm to use (L1, L2, or L∞).
        """
        self.norm: SensitivityNorm = norm
        self._var_sens: dict[str, SymbolicSens] = {}
        self._query_results: list[QuerySensitivity] = []
        self._noise_info: list[NoiseInfo] = []
        self._warnings: list[str] = []
        self._param_names: set[str] = set()
        self._db_params: set[str] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, mechir: MechIR) -> SensitivityResult:
        """Analyze the sensitivity of a mechanism.

        Args:
            mechir: The mechanism IR to analyze.

        Returns:
            A :class:`SensitivityResult` with sensitivity bounds.
        """
        self._var_sens.clear()
        self._query_results.clear()
        self._noise_info.clear()
        self._warnings.clear()
        self._param_names.clear()
        self._db_params.clear()

        # Register parameters
        for param in mechir.params:
            self._param_names.add(param.name)
            if param.is_database:
                self._db_params.add(param.name)
                self._var_sens[param.name] = SymbolicSens.constant(1.0)
            else:
                self._var_sens[param.name] = SymbolicSens.zero()

        # Analyze the body
        self.visit(mechir.body)

        # Compute global sensitivity
        global_sens = self._compute_global_sensitivity()

        return SensitivityResult(
            mechanism_name=mechir.name,
            global_sensitivity=global_sens,
            local_sensitivities=list(self._query_results),
            noise_info=list(self._noise_info),
            composition=self._detect_composition_mode(mechir),
            norm=self.norm,
            warnings=list(self._warnings),
        )

    def generate_certificate(self, mechir: MechIR) -> SensitivityCert:
        """Analyze and produce an exportable sensitivity certificate.

        Args:
            mechir: The mechanism IR to analyze.

        Returns:
            A :class:`SensitivityCert`.
        """
        result = self.analyze(mechir)
        return SensitivityCert.from_result(result)

    # ------------------------------------------------------------------
    # Node visitors
    # ------------------------------------------------------------------

    def visit_AssignNode(self, node: AssignNode) -> None:
        """Track sensitivity through assignments."""
        sens = self._expr_sensitivity(node.value)
        if isinstance(node.target, Var):
            self._var_sens[node.target.name] = sens

    def visit_NoiseDrawNode(self, node: NoiseDrawNode) -> None:
        """Record noise draw information."""
        center_sens = self._expr_sensitivity(node.center)
        scale_sens = self._expr_sensitivity(node.scale)

        query_sens: SymbolicSens | None = None
        if node.sensitivity is not None:
            query_sens = self._expr_sensitivity(node.sensitivity)
            if query_sens.is_zero:
                query_sens = self._const_to_sens(node.sensitivity)
        elif center_sens.is_concrete and not center_sens.is_zero:
            query_sens = center_sens

        scale_val = self._const_to_sens(node.scale)

        if isinstance(node.target, Var):
            # After noise, the sensitivity is "protected"
            self._var_sens[node.target.name] = SymbolicSens.zero()

            self._noise_info.append(NoiseInfo(
                target_var=node.target.name,
                noise_kind=node.noise_kind,
                scale=scale_val,
                query_sens=query_sens,
                node_id=node.node_id,
                source_loc=node.source_loc,
            ))

    def visit_BranchNode(self, node: BranchNode) -> None:
        """Track sensitivity through branches (take max of both sides)."""
        saved = dict(self._var_sens)

        self.visit(node.true_branch)
        true_sens = dict(self._var_sens)

        self._var_sens = dict(saved)
        self.visit(node.false_branch)
        false_sens = dict(self._var_sens)

        # Merge: take maximum sensitivity from either branch
        all_vars = set(true_sens.keys()) | set(false_sens.keys())
        for name in all_vars:
            ts = true_sens.get(name, SymbolicSens.zero())
            fs = false_sens.get(name, SymbolicSens.zero())
            self._var_sens[name] = ts.max(fs)

    def visit_MergeNode(self, node: MergeNode) -> None:
        """Handle merge nodes: take max sensitivity of sources."""
        sens_list: list[SymbolicSens] = []
        for _, expr in node.sources.items():
            sens_list.append(self._expr_sensitivity(expr))

        merged = SymbolicSens.zero()
        for s in sens_list:
            merged = merged.max(s)

        if isinstance(node.target, Var):
            self._var_sens[node.target.name] = merged

    def visit_LoopNode(self, node: LoopNode) -> None:
        """Track sensitivity through loops.

        For L1: sensitivity scales linearly with iterations.
        For L2: sensitivity scales with sqrt(iterations).
        For L∞: sensitivity is the max per-iteration sensitivity.
        """
        # Determine iteration count
        iter_count = self._extract_bound_value(node.bound)

        if isinstance(node.index_var, Var):
            self._var_sens[node.index_var.name] = SymbolicSens.zero()

        saved = dict(self._var_sens)
        self.visit(node.body)

        # Scale body sensitivity by iteration count
        if iter_count is not None and iter_count > 0:
            for name in self._var_sens:
                if name in saved:
                    body_sens = self._var_sens[name]
                    orig_sens = saved[name]
                    if not body_sens.is_zero and body_sens != orig_sens:
                        delta = body_sens.add(orig_sens.scale(-1.0))
                        if self.norm == SensitivityNorm.L1:
                            self._var_sens[name] = delta.scale(iter_count)
                        elif self.norm == SensitivityNorm.L2:
                            self._var_sens[name] = delta.scale(
                                math.sqrt(iter_count)
                            )
                        else:  # LINF
                            self._var_sens[name] = delta

    def visit_QueryNode(self, node: QueryNode) -> None:
        """Record query sensitivity."""
        sens = self._const_to_sens(node.sensitivity)

        self._query_results.append(QuerySensitivity(
            query_name=node.query_name,
            node_id=node.node_id,
            sensitivity=sens,
            norm=self.norm,
            source_loc=node.source_loc,
        ))

        if isinstance(node.target, Var):
            self._var_sens[node.target.name] = sens

    def visit_ReturnNode(self, node: ReturnNode) -> None:
        """Track sensitivity of return value."""
        self._expr_sensitivity(node.value)

    def visit_SequenceNode(self, node: SequenceNode) -> None:
        """Process statements in order."""
        for stmt in node.stmts:
            self.visit(stmt)

    def visit_NoOpNode(self, node: NoOpNode) -> None:
        """No-op: nothing to analyze."""

    # ------------------------------------------------------------------
    # Expression sensitivity
    # ------------------------------------------------------------------

    def _expr_sensitivity(self, expr: TypedExpr) -> SymbolicSens:
        """Compute the sensitivity of an expression.

        Uses abstract interpretation rules:
        - Constants have zero sensitivity
        - Variables carry their tracked sensitivity
        - Addition: Δ(f+g) = Δf + Δg
        - Multiplication by constant: Δ(c*f) = |c| * Δf
        - General multiplication: Δ(f*g) ≤ |f|*Δg + |g|*Δf (but we
          approximate conservatively)
        """
        if isinstance(expr, Const):
            return SymbolicSens.zero()

        if isinstance(expr, Var):
            return self._var_sens.get(expr.name, SymbolicSens.zero())

        if isinstance(expr, BinOp):
            return self._binop_sensitivity(expr)

        if isinstance(expr, UnaryOp):
            return self._unaryop_sensitivity(expr)

        if isinstance(expr, FuncCall):
            return self._funccall_sensitivity(expr)

        # For other expression types, compute from children
        child_sens = [self._expr_sensitivity(c) for c in expr.children()]
        result = SymbolicSens.zero()
        for s in child_sens:
            result = result.add(s)
        return result

    def _binop_sensitivity(self, expr: BinOp) -> SymbolicSens:
        """Compute sensitivity of a binary operation."""
        left_sens = self._expr_sensitivity(expr.left)
        right_sens = self._expr_sensitivity(expr.right)

        if expr.op == BinOpKind.ADD or expr.op == BinOpKind.SUB:
            return left_sens.add(right_sens)

        if expr.op == BinOpKind.MUL:
            # If one side is constant, scale the other
            if left_sens.is_zero:
                const_val = self._const_value(expr.left)
                if const_val is not None:
                    return right_sens.scale(const_val)
                return right_sens
            if right_sens.is_zero:
                const_val = self._const_value(expr.right)
                if const_val is not None:
                    return left_sens.scale(const_val)
                return left_sens
            # Both sensitive: conservative
            return left_sens.multiply(right_sens)

        if expr.op == BinOpKind.DIV:
            if right_sens.is_zero:
                const_val = self._const_value(expr.right)
                if const_val is not None and const_val != 0:
                    return left_sens.scale(1.0 / const_val)
                return left_sens
            self._warnings.append(
                "division by a sensitive value may have unbounded sensitivity"
            )
            return SymbolicSens.unbounded()

        if expr.op == BinOpKind.POW:
            if right_sens.is_zero:
                const_exp = self._const_value(expr.right)
                if const_exp is not None:
                    # Δ(x^n) ≈ n * x^(n-1) * Δx (approximation)
                    return left_sens.scale(abs(const_exp))
            return left_sens.multiply(right_sens)

        if expr.op.is_comparison or expr.op.is_logical:
            # Boolean operations: sensitivity is max of operands
            return left_sens.max(right_sens)

        return left_sens.add(right_sens)

    def _unaryop_sensitivity(self, expr: UnaryOp) -> SymbolicSens:
        """Compute sensitivity of a unary operation."""
        operand_sens = self._expr_sensitivity(expr.operand)
        if expr.op == UnaryOpKind.NEG:
            return operand_sens  # negation preserves sensitivity
        if expr.op == UnaryOpKind.NOT:
            return operand_sens
        return operand_sens

    def _funccall_sensitivity(self, expr: FuncCall) -> SymbolicSens:
        """Compute sensitivity of a function call."""
        arg_sens = [self._expr_sensitivity(a) for a in expr.args]

        if expr.name == "abs":
            return arg_sens[0] if arg_sens else SymbolicSens.zero()
        if expr.name in ("max", "min"):
            if len(arg_sens) >= 2:
                return arg_sens[0].max(arg_sens[1])
            return arg_sens[0] if arg_sens else SymbolicSens.zero()
        if expr.name in ("int", "float", "round"):
            return arg_sens[0] if arg_sens else SymbolicSens.zero()
        if expr.name == "len":
            return SymbolicSens.constant(1.0)

        # Unknown function: take sum of arg sensitivities
        result = SymbolicSens.zero()
        for s in arg_sens:
            result = result.add(s)
        return result

    # ------------------------------------------------------------------
    # Global sensitivity computation
    # ------------------------------------------------------------------

    def _compute_global_sensitivity(self) -> SymbolicSens:
        """Compute the global sensitivity of the mechanism.

        The global sensitivity is the maximum over all query sensitivities,
        composed according to the composition mode.
        """
        if not self._query_results:
            return SymbolicSens.zero()

        if len(self._query_results) == 1:
            return self._query_results[0].sensitivity

        # Sequential composition
        sens_values = [qs.sensitivity for qs in self._query_results]
        return self._compose_sensitivities(sens_values)

    def _compose_sensitivities(
        self, sens_list: list[SymbolicSens]
    ) -> SymbolicSens:
        """Compose multiple sensitivity values.

        For L1 (sequential): sum of sensitivities.
        For L2 (sequential): sqrt of sum of squares.
        For L∞ (parallel):   max of sensitivities.
        """
        if not sens_list:
            return SymbolicSens.zero()

        if self.norm == SensitivityNorm.LINF:
            # Parallel composition: take max
            result = sens_list[0]
            for s in sens_list[1:]:
                result = result.max(s)
            return result

        if self.norm == SensitivityNorm.L2:
            # L2 composition: sqrt(sum of squares)
            if all(s.is_concrete for s in sens_list):
                total = sum(
                    (s.value or 0.0) ** 2 for s in sens_list  # type: ignore[operator]
                )
                return SymbolicSens.constant(math.sqrt(total))
            # Symbolic fallback
            result = SymbolicSens.zero()
            for s in sens_list:
                result = result.add(s.multiply(s))
            return result.sqrt()

        # L1 composition: sum
        result = SymbolicSens.zero()
        for s in sens_list:
            result = result.add(s)
        return result

    # ------------------------------------------------------------------
    # Composition mode detection
    # ------------------------------------------------------------------

    def _detect_composition_mode(self, mechir: MechIR) -> str:
        """Detect whether queries compose sequentially or in parallel.

        Sequential composition: queries are on the same data.
        Parallel composition: queries are on disjoint subsets.
        """
        queries = mechir.queries()
        if len(queries) <= 1:
            return "sequential"

        # Check if queries are in a loop (sequential composition)
        for node in mechir.all_nodes():
            if isinstance(node, LoopNode):
                for inner in node.body.walk():
                    if isinstance(inner, (QueryNode, NoiseDrawNode)):
                        return "sequential"

        # Check if queries access disjoint subsets (simplified heuristic)
        query_args: list[set[str]] = []
        for q in queries:
            arg_vars: set[str] = set()
            for arg in q.args:
                arg_vars.update(arg.free_vars())
            query_args.append(arg_vars)

        # If queries share no arguments, they may be parallel
        is_parallel = True
        for i in range(len(query_args)):
            for j in range(i + 1, len(query_args)):
                if query_args[i] & query_args[j]:
                    is_parallel = False
                    break
            if not is_parallel:
                break

        return "parallel" if is_parallel else "sequential"

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _const_to_sens(self, expr: TypedExpr) -> SymbolicSens:
        """Convert a constant expression to a SymbolicSens value."""
        if isinstance(expr, Const):
            val = expr.value
            if isinstance(val, (int, float)):
                return SymbolicSens.constant(float(val))
        if isinstance(expr, Var):
            sens = self._var_sens.get(expr.name)
            if sens is not None:
                return sens
            return SymbolicSens.from_var(expr.name)
        return SymbolicSens(symbolic=str(expr))

    def _const_value(self, expr: TypedExpr) -> float | None:
        """Extract a constant numeric value from an expression."""
        if isinstance(expr, Const):
            val = expr.value
            if isinstance(val, (int, float)):
                return float(val)
        return None

    def _extract_bound_value(self, expr: TypedExpr) -> float | None:
        """Extract a numeric bound from an expression."""
        if isinstance(expr, Const):
            val = expr.value
            if isinstance(val, (int, float)):
                return float(val)
        return None


# ═══════════════════════════════════════════════════════════════════════════
# COMPOSITION HELPERS
# ═══════════════════════════════════════════════════════════════════════════


def sequential_compose(
    sensitivities: list[float],
    norm: SensitivityNorm = SensitivityNorm.L1,
) -> float:
    """Compute composed sensitivity under sequential composition.

    Args:
        sensitivities: List of individual query sensitivities.
        norm:          Sensitivity norm.

    Returns:
        The composed sensitivity bound.
    """
    if not sensitivities:
        return 0.0

    if norm == SensitivityNorm.L1:
        return sum(sensitivities)
    if norm == SensitivityNorm.L2:
        return math.sqrt(sum(s * s for s in sensitivities))
    if norm == SensitivityNorm.LINF:
        return max(sensitivities)

    return sum(sensitivities)


def parallel_compose(
    sensitivities: list[float],
    norm: SensitivityNorm = SensitivityNorm.L1,
) -> float:
    """Compute composed sensitivity under parallel composition.

    Under parallel composition, queries access disjoint subsets of the
    database, so the composed sensitivity is the maximum.

    Args:
        sensitivities: List of individual query sensitivities.
        norm:          Sensitivity norm.

    Returns:
        The composed sensitivity bound (max for parallel).
    """
    if not sensitivities:
        return 0.0
    return max(sensitivities)


# ═══════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════


def analyze_sensitivity(
    mechir: MechIR,
    norm: SensitivityNorm = SensitivityNorm.L1,
) -> SensitivityResult:
    """Analyze the sensitivity of a mechanism.

    Args:
        mechir: The mechanism IR to analyze.
        norm:   Sensitivity norm to use.

    Returns:
        A :class:`SensitivityResult` with sensitivity bounds.
    """
    analyzer = SensitivityAnalyzer(norm=norm)
    return analyzer.analyze(mechir)


def generate_sensitivity_certificate(
    mechir: MechIR,
    norm: SensitivityNorm = SensitivityNorm.L1,
) -> SensitivityCert:
    """Analyze a mechanism and produce a sensitivity certificate.

    Args:
        mechir: The mechanism IR to analyze.
        norm:   Sensitivity norm to use.

    Returns:
        A :class:`SensitivityCert`.
    """
    analyzer = SensitivityAnalyzer(norm=norm)
    return analyzer.generate_certificate(mechir)
