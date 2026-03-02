"""Type checker for DPImp programs.

Validates DPImp restrictions on MechIR trees, ensuring that programs
conform to the subset that can be verified by DP-CEGAR:
- No recursion
- Bounded loops only
- No dynamic dispatch, closures, or classes
- Known noise primitives only
- Numeric types only (int, float, bool)

Classes
-------
TypeEnvironment – scope stack for type bindings
TypeError_      – type checking error with source location
TypeChecker     – visitor that validates DPImp restrictions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Sequence

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
from dpcegar.utils.errors import SourceLoc, TypeCheckError


# ═══════════════════════════════════════════════════════════════════════════
# SENSITIVITY TYPE
# ═══════════════════════════════════════════════════════════════════════════


class SensitivityKind(Enum):
    """How a variable relates to query inputs (sensitivity tracking)."""

    CONSTANT = auto()       # Independent of database input
    QUERY_RESULT = auto()   # Directly from a database query
    DERIVED = auto()        # Computed from query results
    NOISY = auto()          # Has had noise added
    UNKNOWN = auto()        # Cannot determine


@dataclass(frozen=True, slots=True)
class SensitivityType:
    """Tracks how data flows from database queries to outputs.

    Attributes:
        kind:        How this value relates to database input.
        sensitivity: Numeric sensitivity bound (if known).
        source_var:  The query variable this derives from (if any).
    """

    kind: SensitivityKind = SensitivityKind.CONSTANT
    sensitivity: float | None = None
    source_var: str | None = None

    def is_sensitive(self) -> bool:
        """Return True if this value depends on database input."""
        return self.kind in (
            SensitivityKind.QUERY_RESULT,
            SensitivityKind.DERIVED,
        )


# ═══════════════════════════════════════════════════════════════════════════
# TYPE ENVIRONMENT
# ═══════════════════════════════════════════════════════════════════════════


class TypeEnvironment:
    """Scoped type environment for type checking.

    Maintains a stack of scopes, each mapping variable names to their
    IR types and sensitivity information.

    Usage::

        env = TypeEnvironment()
        env.push_scope()
        env.define("x", IRType.REAL)
        ty = env.lookup("x")
        env.pop_scope()
    """

    def __init__(self) -> None:
        """Initialize with a single empty scope."""
        self._type_scopes: list[dict[str, IRType]] = [{}]
        self._sens_scopes: list[dict[str, SensitivityType]] = [{}]

    def push_scope(self) -> None:
        """Push a new scope onto the stack."""
        self._type_scopes.append({})
        self._sens_scopes.append({})

    def pop_scope(self) -> None:
        """Pop the top scope from the stack."""
        if len(self._type_scopes) > 1:
            self._type_scopes.pop()
            self._sens_scopes.pop()

    @property
    def depth(self) -> int:
        """Current nesting depth (1 = top-level)."""
        return len(self._type_scopes)

    def define(
        self,
        name: str,
        ty: IRType,
        sens: SensitivityType | None = None,
    ) -> None:
        """Define a variable in the current scope.

        Args:
            name: Variable name.
            ty:   IR type.
            sens: Sensitivity type (defaults to CONSTANT).
        """
        self._type_scopes[-1][name] = ty
        self._sens_scopes[-1][name] = sens or SensitivityType()

    def lookup(self, name: str) -> IRType | None:
        """Look up the type of a variable.

        Searches from innermost to outermost scope.

        Returns:
            The IR type, or ``None`` if not found.
        """
        for scope in reversed(self._type_scopes):
            if name in scope:
                return scope[name]
        return None

    def lookup_sensitivity(self, name: str) -> SensitivityType | None:
        """Look up the sensitivity type of a variable.

        Returns:
            The sensitivity type, or ``None`` if not found.
        """
        for scope in reversed(self._sens_scopes):
            if name in scope:
                return scope[name]
        return None

    def is_defined(self, name: str) -> bool:
        """Return True if *name* is defined in any scope."""
        return any(name in scope for scope in self._type_scopes)

    def all_variables(self) -> dict[str, IRType]:
        """Return all visible variable bindings (innermost wins)."""
        result: dict[str, IRType] = {}
        for scope in self._type_scopes:
            result.update(scope)
        return result

    def sensitive_variables(self) -> dict[str, SensitivityType]:
        """Return all variables that carry sensitivity information."""
        result: dict[str, SensitivityType] = {}
        for scope in self._sens_scopes:
            for name, sens in scope.items():
                if sens.is_sensitive():
                    result[name] = sens
        return result


# ═══════════════════════════════════════════════════════════════════════════
# TYPE CHECK ERROR
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class TypeErrorInfo:
    """A single type-checking finding.

    Attributes:
        message:    Human-readable description.
        severity:   'error' or 'warning'.
        source_loc: Source location (if available).
        node_id:    IR node ID (if available).
        category:   Error category for filtering.
    """

    message: str
    severity: str = "error"
    source_loc: SourceLoc | None = None
    node_id: int | None = None
    category: str = "type"

    def __str__(self) -> str:
        loc = f" at {self.source_loc}" if self.source_loc else ""
        return f"[{self.severity}] {self.message}{loc}"


# ═══════════════════════════════════════════════════════════════════════════
# TYPE CHECKER
# ═══════════════════════════════════════════════════════════════════════════


class TypeChecker(IRNodeVisitorBase):
    """Validates DPImp restrictions on MechIR trees.

    Checks:
    - All variables are defined before use
    - Operand types are compatible
    - Noise primitives use valid distributions
    - Loops are bounded (range with constant or parameter bound)
    - No recursion, closures, or dynamic dispatch
    - Sensitivity flow from queries to noise
    - Return type consistency

    Usage::

        checker = TypeChecker()
        ok = checker.check(mechir)
        for err in checker.errors:
            print(err)
    """

    def __init__(self) -> None:
        """Initialize the type checker."""
        self.env: TypeEnvironment = TypeEnvironment()
        self.errors: list[TypeErrorInfo] = []
        self.warnings: list[TypeErrorInfo] = []
        self._function_names: set[str] = set()
        self._current_func: str | None = None
        self._loop_depth: int = 0
        self._max_loop_depth: int = 0
        self._has_return: bool = False
        self._noise_vars: set[str] = set()
        self._query_vars: set[str] = set()
        self._return_types: list[IRType] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, mechir: MechIR) -> bool:
        """Run the type checker on a MechIR.

        Args:
            mechir: The mechanism IR to check.

        Returns:
            True if no errors were found.
        """
        self.errors.clear()
        self.warnings.clear()
        self._current_func = mechir.name
        self._function_names = {mechir.name}
        self._loop_depth = 0
        self._max_loop_depth = 0
        self._has_return = False
        self._noise_vars.clear()
        self._query_vars.clear()
        self._return_types.clear()

        # Register parameters
        for param in mechir.params:
            sens = SensitivityType(
                kind=SensitivityKind.CONSTANT,
            )
            if param.is_database:
                sens = SensitivityType(kind=SensitivityKind.QUERY_RESULT)
            self.env.define(param.name, param.ty, sens)

        # Type-check the body
        self.visit(mechir.body)

        # Post-checks
        self._post_check(mechir)

        return len(self.errors) == 0

    # ------------------------------------------------------------------
    # Post-analysis checks
    # ------------------------------------------------------------------

    def _post_check(self, mechir: MechIR) -> None:
        """Run checks after the main traversal."""
        # Check that there's at least one return
        if not self._has_return:
            self._warn(
                "mechanism has no return statement",
                category="structure",
            )

        # Check return type consistency
        if len(set(self._return_types)) > 1:
            types_str = ", ".join(t.name for t in self._return_types)
            self._warn(
                f"inconsistent return types: {types_str}",
                category="type",
            )

        # Check that noise is applied to sensitive data
        if self._query_vars and not self._noise_vars:
            self._warn(
                "mechanism queries database but adds no noise; "
                "this is unlikely to be differentially private",
                category="privacy",
            )

        # Verify declared return type matches inferred
        if self._return_types and mechir.return_type != IRType.REAL:
            actual = self._return_types[-1]
            if actual != mechir.return_type and actual != IRType.REAL:
                self._warn(
                    f"declared return type {mechir.return_type.name} "
                    f"does not match inferred type {actual.name}",
                    category="type",
                )

    # ------------------------------------------------------------------
    # Node visitors
    # ------------------------------------------------------------------

    def visit_AssignNode(self, node: AssignNode) -> None:
        """Type-check an assignment node."""
        # Check RHS expression
        rhs_ty = self._check_expr(node.value)

        # Check target is a Var
        if not isinstance(node.target, Var):
            self._err("assignment target must be a variable", node)
            return

        # Infer sensitivity
        sens = self._infer_sensitivity(node.value)

        # Define variable
        self.env.define(node.target.name, rhs_ty, sens)

    def visit_NoiseDrawNode(self, node: NoiseDrawNode) -> None:
        """Type-check a noise draw node."""
        # Check distribution is valid
        if not isinstance(node.noise_kind, NoiseKind):
            self._err(f"invalid noise distribution: {node.noise_kind}", node)

        # Check center expression
        center_ty = self._check_expr(node.center)
        if not center_ty.is_numeric():
            self._err(
                f"noise center must be numeric, got {center_ty.name}", node
            )

        # Check scale expression
        scale_ty = self._check_expr(node.scale)
        if not scale_ty.is_numeric():
            self._err(
                f"noise scale must be numeric, got {scale_ty.name}", node
            )

        # Check sensitivity if provided
        if node.sensitivity is not None:
            sens_ty = self._check_expr(node.sensitivity)
            if not sens_ty.is_numeric():
                self._err(
                    f"sensitivity must be numeric, got {sens_ty.name}", node
                )

        # Register as noise variable
        if isinstance(node.target, Var):
            self._noise_vars.add(node.target.name)
            self.env.define(
                node.target.name,
                IRType.REAL,
                SensitivityType(kind=SensitivityKind.NOISY),
            )

    def visit_BranchNode(self, node: BranchNode) -> None:
        """Type-check a branch (if/else) node."""
        cond_ty = self._check_expr(node.condition)
        if cond_ty != IRType.BOOL:
            self._err(
                f"branch condition must be bool, got {cond_ty.name}", node
            )

        self.env.push_scope()
        self.visit(node.true_branch)
        self.env.pop_scope()

        self.env.push_scope()
        self.visit(node.false_branch)
        self.env.pop_scope()

    def visit_MergeNode(self, node: MergeNode) -> None:
        """Type-check a merge (phi) node."""
        if not node.sources:
            self._warn("merge node has no sources", node)

        # Check all source expressions
        source_types: list[IRType] = []
        for pred_id, expr in node.sources.items():
            ty = self._check_expr(expr)
            source_types.append(ty)

        # All source types should be compatible
        if source_types:
            base = source_types[0]
            for ty in source_types[1:]:
                if ty != base and not (ty.is_numeric() and base.is_numeric()):
                    self._warn(
                        f"merge node has incompatible source types: "
                        f"{base.name} vs {ty.name}",
                        node,
                    )

        if isinstance(node.target, Var):
            merged_ty = source_types[0] if source_types else IRType.REAL
            self.env.define(node.target.name, merged_ty)

    def visit_LoopNode(self, node: LoopNode) -> None:
        """Type-check a loop node."""
        self._loop_depth += 1
        self._max_loop_depth = max(self._max_loop_depth, self._loop_depth)

        # Check that bound is integral
        bound_ty = self._check_expr(node.bound)
        if bound_ty != IRType.INT and not isinstance(node.bound, Const):
            self._warn(
                f"loop bound should be integer, got {bound_ty.name}", node
            )

        # Check that bound is bounded (constant or parameter)
        self._check_loop_bound(node)

        # Register index variable
        if isinstance(node.index_var, Var):
            self.env.define(node.index_var.name, IRType.INT)

        self.env.push_scope()
        self.visit(node.body)
        self.env.pop_scope()

        self._loop_depth -= 1

    def visit_QueryNode(self, node: QueryNode) -> None:
        """Type-check a query node."""
        if not node.query_name:
            self._err("query node has no query name", node)

        for arg in node.args:
            self._check_expr(arg)

        sens_ty = self._check_expr(node.sensitivity)
        if not sens_ty.is_numeric():
            self._err(
                f"query sensitivity must be numeric, got {sens_ty.name}", node
            )

        if isinstance(node.target, Var):
            self._query_vars.add(node.target.name)
            self.env.define(
                node.target.name,
                IRType.REAL,
                SensitivityType(
                    kind=SensitivityKind.QUERY_RESULT,
                    source_var=node.target.name,
                ),
            )

    def visit_ReturnNode(self, node: ReturnNode) -> None:
        """Type-check a return node."""
        ret_ty = self._check_expr(node.value)
        self._has_return = True
        self._return_types.append(ret_ty)

    def visit_SequenceNode(self, node: SequenceNode) -> None:
        """Type-check a sequence of statements."""
        for stmt in node.stmts:
            self.visit(stmt)

    def visit_NoOpNode(self, node: NoOpNode) -> None:
        """No-op: nothing to check."""

    # ------------------------------------------------------------------
    # Expression type checking
    # ------------------------------------------------------------------

    def _check_expr(self, expr: TypedExpr) -> IRType:
        """Recursively type-check an expression, returning its type.

        This validates:
        - Variable references are defined
        - Operand types are compatible
        - Function calls have valid arguments

        Args:
            expr: The expression to check.

        Returns:
            The inferred IR type of the expression.
        """
        if isinstance(expr, Var):
            return self._check_var(expr)
        if isinstance(expr, Const):
            return self._check_const(expr)
        if isinstance(expr, BinOp):
            return self._check_binop(expr)
        if isinstance(expr, UnaryOp):
            return self._check_unaryop(expr)
        if isinstance(expr, FuncCall):
            return self._check_funccall(expr)
        # For other expression types, trust the annotated type
        return expr.ty

    def _check_var(self, var: Var) -> IRType:
        """Check that a variable is defined and return its type."""
        env_ty = self.env.lookup(var.name)
        if env_ty is None:
            # Could be a parameter or external; treat as REAL with warning
            self._warn(f"reference to possibly undefined variable: {var.name}")
            return var.ty
        # Verify consistency
        if env_ty != var.ty and var.ty != IRType.REAL:
            self._warn(
                f"variable {var.name} has type {env_ty.name} in environment "
                f"but {var.ty.name} in expression"
            )
        return env_ty

    def _check_const(self, const: Const) -> IRType:
        """Type-check a constant literal."""
        if isinstance(const.value, bool):
            if const.ty != IRType.BOOL:
                self._warn(f"bool constant has type {const.ty.name}")
            return IRType.BOOL
        if isinstance(const.value, int) and not isinstance(const.value, bool):
            if const.ty not in (IRType.INT, IRType.REAL):
                self._warn(f"int constant has unexpected type {const.ty.name}")
            return const.ty
        if isinstance(const.value, float):
            if const.ty != IRType.REAL:
                self._warn(f"float constant has type {const.ty.name}")
            return IRType.REAL
        return const.ty

    def _check_binop(self, binop: BinOp) -> IRType:
        """Type-check a binary operation."""
        left_ty = self._check_expr(binop.left)
        right_ty = self._check_expr(binop.right)

        if binop.op.is_arithmetic:
            if not left_ty.is_numeric():
                self._err(
                    f"left operand of {binop.op} must be numeric, "
                    f"got {left_ty.name}"
                )
            if not right_ty.is_numeric():
                self._err(
                    f"right operand of {binop.op} must be numeric, "
                    f"got {right_ty.name}"
                )
            if binop.op == BinOpKind.DIV:
                return IRType.REAL
            if left_ty == IRType.REAL or right_ty == IRType.REAL:
                return IRType.REAL
            return IRType.INT

        if binop.op.is_comparison:
            if not (left_ty.is_numeric() and right_ty.is_numeric()):
                if left_ty != right_ty:
                    self._warn(
                        f"comparison {binop.op} between "
                        f"{left_ty.name} and {right_ty.name}"
                    )
            return IRType.BOOL

        if binop.op.is_logical:
            if left_ty != IRType.BOOL:
                self._warn(
                    f"left operand of {binop.op} should be bool, "
                    f"got {left_ty.name}"
                )
            if right_ty != IRType.BOOL:
                self._warn(
                    f"right operand of {binop.op} should be bool, "
                    f"got {right_ty.name}"
                )
            return IRType.BOOL

        return binop.ty

    def _check_unaryop(self, unaryop: UnaryOp) -> IRType:
        """Type-check a unary operation."""
        operand_ty = self._check_expr(unaryop.operand)

        if unaryop.op == UnaryOpKind.NEG:
            if not operand_ty.is_numeric():
                self._err(
                    f"negation requires numeric operand, got {operand_ty.name}"
                )
            return operand_ty

        if unaryop.op == UnaryOpKind.NOT:
            return IRType.BOOL

        return unaryop.ty

    def _check_funccall(self, call: FuncCall) -> IRType:
        """Type-check a function call expression."""
        # Check for recursion
        if call.name == self._current_func:
            self._err(
                f"recursion is not allowed in DPImp: "
                f"{call.name} calls itself"
            )

        # Check arguments
        for arg in call.args:
            self._check_expr(arg)

        # Known function return types
        known_types: dict[str, IRType] = {
            "abs": IRType.REAL,
            "max": IRType.REAL,
            "min": IRType.REAL,
            "len": IRType.INT,
            "int": IRType.INT,
            "float": IRType.REAL,
            "round": IRType.INT,
            "bool": IRType.BOOL,
        }
        return known_types.get(call.name, call.ty)

    # ------------------------------------------------------------------
    # Loop bound checking
    # ------------------------------------------------------------------

    def _check_loop_bound(self, node: LoopNode) -> None:
        """Verify that a loop bound is statically bounded.

        DPImp requires all loops to have bounds that are either:
        - Constant literals
        - Function parameters
        - Expressions of constants and parameters
        """
        bound = node.bound
        if isinstance(bound, Const):
            return  # constant bound: always OK

        if isinstance(bound, Var):
            # Check if it's a parameter (defined at top scope)
            if self.env.lookup(bound.name) is not None:
                return  # parameter or prior variable: OK
            self._warn(
                f"loop bound variable {bound.name} may not be statically bounded",
                node,
            )
            return

        if isinstance(bound, BinOp):
            # Allow arithmetic of constants and parameters
            free_vars = bound.free_vars()
            for v in free_vars:
                if not self.env.is_defined(v):
                    self._err(
                        f"loop bound references undefined variable: {v}",
                        node,
                    )
                    return
            return  # expression of known variables: OK

        if isinstance(bound, FuncCall):
            if bound.name == "len":
                return  # len() is acceptable
            self._warn(
                f"loop bound uses function call {bound.name}(); "
                f"may not be statically bounded",
                node,
            )
            return

        self._warn("loop bound may not be statically bounded", node)

    # ------------------------------------------------------------------
    # Sensitivity inference
    # ------------------------------------------------------------------

    def _infer_sensitivity(self, expr: TypedExpr) -> SensitivityType:
        """Infer the sensitivity type of an expression.

        Tracks how database query results flow through computations.
        """
        if isinstance(expr, Const):
            return SensitivityType(kind=SensitivityKind.CONSTANT)

        if isinstance(expr, Var):
            sens = self.env.lookup_sensitivity(expr.name)
            if sens is not None:
                return sens
            return SensitivityType(kind=SensitivityKind.CONSTANT)

        if isinstance(expr, BinOp):
            left_sens = self._infer_sensitivity(expr.left)
            right_sens = self._infer_sensitivity(expr.right)
            if left_sens.is_sensitive() or right_sens.is_sensitive():
                source = left_sens.source_var or right_sens.source_var
                return SensitivityType(
                    kind=SensitivityKind.DERIVED,
                    source_var=source,
                )
            return SensitivityType(kind=SensitivityKind.CONSTANT)

        if isinstance(expr, UnaryOp):
            return self._infer_sensitivity(expr.operand)

        if isinstance(expr, FuncCall):
            for arg in expr.args:
                arg_sens = self._infer_sensitivity(arg)
                if arg_sens.is_sensitive():
                    return SensitivityType(
                        kind=SensitivityKind.DERIVED,
                        source_var=arg_sens.source_var,
                    )
            return SensitivityType(kind=SensitivityKind.CONSTANT)

        return SensitivityType(kind=SensitivityKind.UNKNOWN)

    # ------------------------------------------------------------------
    # Error helpers
    # ------------------------------------------------------------------

    def _err(
        self,
        message: str,
        node: IRNode | None = None,
        category: str = "type",
    ) -> None:
        """Record a type-checking error."""
        loc = node.source_loc if node else None
        nid = node.node_id if node else None
        self.errors.append(
            TypeErrorInfo(
                message=message,
                severity="error",
                source_loc=loc,
                node_id=nid,
                category=category,
            )
        )

    def _warn(
        self,
        message: str,
        node: IRNode | None = None,
        category: str = "type",
    ) -> None:
        """Record a type-checking warning."""
        loc = node.source_loc if node else None
        nid = node.node_id if node else None
        self.warnings.append(
            TypeErrorInfo(
                message=message,
                severity="warning",
                source_loc=loc,
                node_id=nid,
                category=category,
            )
        )


# ═══════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════


def type_check(mechir: MechIR) -> tuple[bool, list[TypeErrorInfo]]:
    """Type-check a MechIR and return results.

    Args:
        mechir: The mechanism IR to check.

    Returns:
        A tuple of (ok, errors). ``ok`` is True if no errors were found.
    """
    checker = TypeChecker()
    ok = checker.check(mechir)
    return ok, checker.errors


def type_check_strict(mechir: MechIR) -> None:
    """Type-check a MechIR, raising on the first error.

    Args:
        mechir: The mechanism IR to check.

    Raises:
        TypeCheckError: On the first type error found.
    """
    checker = TypeChecker()
    ok = checker.check(mechir)
    if not ok:
        err = checker.errors[0]
        raise TypeCheckError(
            err.message,
            source_loc=err.source_loc,
        )
