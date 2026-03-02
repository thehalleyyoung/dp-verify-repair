"""Python AST → MechIR bridge.

Uses Python's built-in ``ast`` module to parse DPImp source code, then
walks the resulting AST to construct a MechIR tree.  This is the primary
entry point for converting user-written mechanism code into the IR used
by the DP-CEGAR verification engine.

Classes
-------
ASTBridgeError – errors specific to AST-to-IR lowering
ASTVisitor     – walks Python AST and builds MechIR

Functions
---------
parse_mechanism – convenience function: source code → MechIR
"""

from __future__ import annotations

import ast
import textwrap
from dataclasses import dataclass, field
from typing import Any, Sequence

from dpcegar.ir.types import (
    BinOp as IRBinOp,
    BinOpKind,
    Const,
    FuncCall,
    IRType,
    NoiseKind,
    TypedExpr,
    UnaryOp as IRUnaryOp,
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
from dpcegar.parser.source_map import SourceMap, SourceRange
from dpcegar.utils.errors import ParseError, SourceLoc


# ═══════════════════════════════════════════════════════════════════════════
# AST BRIDGE ERROR
# ═══════════════════════════════════════════════════════════════════════════


class ASTBridgeError(ParseError):
    """Raised when Python AST cannot be lowered to MechIR."""


# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════


_NOISE_NAMES: dict[str, NoiseKind] = {
    "laplace": NoiseKind.LAPLACE,
    "lap": NoiseKind.LAPLACE,
    "gaussian": NoiseKind.GAUSSIAN,
    "gauss": NoiseKind.GAUSSIAN,
    "exponential_mechanism": NoiseKind.EXPONENTIAL,
    "exp_mech": NoiseKind.EXPONENTIAL,
}

_QUERY_PREFIXES: frozenset[str] = frozenset({
    "query", "q_", "count", "sum_query", "mean_query",
    "median_query", "histogram_query",
})

_BINOP_MAP: dict[type, BinOpKind] = {
    ast.Add: BinOpKind.ADD,
    ast.Sub: BinOpKind.SUB,
    ast.Mult: BinOpKind.MUL,
    ast.Div: BinOpKind.DIV,
    ast.Mod: BinOpKind.MOD,
    ast.Pow: BinOpKind.POW,
    ast.FloorDiv: BinOpKind.DIV,
}

_CMPOP_MAP: dict[type, BinOpKind] = {
    ast.Eq: BinOpKind.EQ,
    ast.NotEq: BinOpKind.NEQ,
    ast.Lt: BinOpKind.LT,
    ast.LtE: BinOpKind.LE,
    ast.Gt: BinOpKind.GT,
    ast.GtE: BinOpKind.GE,
}

_BOOLOP_MAP: dict[type, BinOpKind] = {
    ast.And: BinOpKind.AND,
    ast.Or: BinOpKind.OR,
}

_UNARYOP_MAP: dict[type, UnaryOpKind] = {
    ast.USub: UnaryOpKind.NEG,
    ast.Not: UnaryOpKind.NOT,
}


# ═══════════════════════════════════════════════════════════════════════════
# SSA ENVIRONMENT
# ═══════════════════════════════════════════════════════════════════════════


class _SSAEnv:
    """Tracks variable definitions for SSA conversion during lowering."""

    def __init__(self) -> None:
        """Initialize with an empty scope stack."""
        self._scopes: list[dict[str, int]] = [{}]
        self._types: dict[str, IRType] = {}
        self._counters: dict[str, int] = {}

    def push_scope(self) -> None:
        """Push a new variable scope."""
        self._scopes.append({})

    def pop_scope(self) -> None:
        """Pop the current variable scope."""
        if len(self._scopes) > 1:
            self._scopes.pop()

    def define(self, name: str, ty: IRType = IRType.REAL) -> Var:
        """Define (or re-define) a variable, returning a fresh SSA Var."""
        version = self._counters.get(name, 0)
        self._counters[name] = version + 1
        self._scopes[-1][name] = version
        self._types[name] = ty
        return Var(ty=ty, name=name, version=version)

    def lookup(self, name: str) -> Var | None:
        """Look up the current SSA version of a variable."""
        for scope in reversed(self._scopes):
            if name in scope:
                ty = self._types.get(name, IRType.REAL)
                return Var(ty=ty, name=name, version=scope[name])
        return None

    def lookup_type(self, name: str) -> IRType:
        """Look up the type of a variable."""
        return self._types.get(name, IRType.REAL)

    def current_version(self, name: str) -> int:
        """Return the current SSA version of a variable, or -1."""
        for scope in reversed(self._scopes):
            if name in scope:
                return scope[name]
        return -1

    def snapshot(self) -> dict[str, int]:
        """Snapshot the current variable versions for branch merging."""
        result: dict[str, int] = {}
        for scope in self._scopes:
            result.update(scope)
        return result

    def is_defined(self, name: str) -> bool:
        """Return True if *name* has been defined in any scope."""
        return any(name in scope for scope in self._scopes)


# ═══════════════════════════════════════════════════════════════════════════
# DECORATOR INFO
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class DecoratorInfo:
    """Parsed information from DP-specific decorators.

    Attributes:
        epsilon:       Declared ε budget.
        delta:         Declared δ budget.
        sensitivity:   Declared global sensitivity.
        noise_type:    Preferred noise distribution.
        is_mechanism:  True if @dp_mechanism was found.
        metadata:      Additional key-value pairs from decorators.
    """

    epsilon: float | None = None
    delta: float | None = None
    sensitivity: float | None = None
    noise_type: NoiseKind | None = None
    is_mechanism: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
# AST VISITOR
# ═══════════════════════════════════════════════════════════════════════════


class ASTVisitor(ast.NodeVisitor):
    """Walk a Python AST and build a MechIR tree.

    This visitor handles the DPImp subset:
    - Function definitions → MechIR
    - Assignments → AssignNode
    - If/elif/else → BranchNode
    - For loops with range() → LoopNode
    - Return statements → ReturnNode
    - Noise primitive calls → NoiseDrawNode
    - Query function calls → QueryNode

    Unsupported constructs (classes, closures, recursion, etc.) raise
    :class:`ASTBridgeError`.

    Usage::

        visitor = ASTVisitor(file="mech.py")
        mechir = visitor.parse(source_code)
    """

    def __init__(self, file: str = "<unknown>") -> None:
        """Initialize the AST visitor.

        Args:
            file: Source file name for diagnostics.
        """
        self.file: str = file
        self.source_map: SourceMap = SourceMap()
        self.errors: list[ASTBridgeError] = []
        self._env: _SSAEnv = _SSAEnv()
        self._decorator_info: DecoratorInfo = DecoratorInfo()
        self._function_names: set[str] = set()
        self._current_func: str | None = None
        self._sensitivity_annotations: dict[str, float] = {}
        self._temp_counter: int = 0
        self._source_text: str = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, source: str) -> MechIR:
        """Parse DPImp source code and return a MechIR.

        Args:
            source: Python source code defining a DP mechanism.

        Returns:
            A complete :class:`MechIR` representation.

        Raises:
            ASTBridgeError: If the source cannot be lowered to MechIR.
        """
        self._source_text = source
        self.source_map.set_source_text(self.file, source)

        try:
            tree = ast.parse(source, filename=self.file)
        except SyntaxError as e:
            raise ASTBridgeError(
                f"Python syntax error: {e.msg}",
                source_loc=SourceLoc(
                    file=self.file,
                    line=e.lineno or 0,
                    col=e.offset or 0,
                ),
            ) from e

        # First pass: collect function names for recursion detection
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                self._function_names.add(node.name)

        # Extract sensitivity annotations from comments
        self._extract_sensitivity_comments(source)

        # Find the mechanism function (first function with @dp_mechanism,
        # or just the first function)
        func_node = self._find_mechanism_function(tree)
        if func_node is None:
            raise ASTBridgeError(
                "No function definition found in source",
                source_loc=SourceLoc(file=self.file, line=1, col=1),
            )

        return self._lower_function(func_node)

    # ------------------------------------------------------------------
    # Function lowering
    # ------------------------------------------------------------------

    def _find_mechanism_function(
        self, tree: ast.Module
    ) -> ast.FunctionDef | None:
        """Find the mechanism function definition in the module.

        Prefers functions decorated with @dp_mechanism; falls back to
        the first function definition.
        """
        functions: list[ast.FunctionDef] = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                functions.append(node)

        if not functions:
            return None

        # Prefer @dp_mechanism decorated functions
        for func in functions:
            for dec in func.decorator_list:
                dec_name = self._decorator_name(dec)
                if dec_name == "dp_mechanism":
                    return func

        return functions[0]

    def _lower_function(self, func: ast.FunctionDef) -> MechIR:
        """Lower a Python function definition to MechIR.

        Args:
            func: The AST function definition node.

        Returns:
            A :class:`MechIR` with params, body, and metadata.
        """
        self._current_func = func.name
        self._decorator_info = self._parse_decorators(func.decorator_list)

        # Parse parameters
        params = self._lower_params(func.args)

        # Register params in the SSA environment
        for p in params:
            self._env.define(p.name, p.ty)

        # Lower body statements
        body = self._lower_body(func.body)

        # Determine return type from annotations or inference
        return_type = self._infer_return_type(func)

        # Build budget from decorator info
        budget = None
        if self._decorator_info.epsilon is not None:
            from dpcegar.ir.types import PureBudget, ApproxBudget
            if self._decorator_info.delta is not None and self._decorator_info.delta > 0:
                budget = ApproxBudget(
                    epsilon=self._decorator_info.epsilon,
                    delta=self._decorator_info.delta,
                )
            else:
                budget = PureBudget(epsilon=self._decorator_info.epsilon)

        mechir = MechIR(
            name=func.name,
            params=params,
            body=body,
            return_type=return_type,
            budget=budget,
            metadata=self._decorator_info.metadata,
        )

        loc = self._loc(func)
        if loc:
            self.source_map.add(mechir.body.node_id, SourceRange(
                file=self.file,
                start_line=func.lineno,
                start_col=func.col_offset + 1,
                end_line=func.end_lineno or func.lineno,
                end_col=func.end_col_offset or 0,
            ))

        self._current_func = None
        return mechir

    def _lower_params(self, args: ast.arguments) -> list[ParamDecl]:
        """Lower function parameters to ParamDecl list."""
        params: list[ParamDecl] = []
        for arg in args.args:
            ty = self._annotation_to_type(arg.annotation)
            is_db = self._is_database_param(arg)
            params.append(ParamDecl(name=arg.arg, ty=ty, is_database=is_db))
        return params

    def _is_database_param(self, arg: ast.arg) -> bool:
        """Check if a parameter represents the database input."""
        name = arg.arg.lower()
        if name in ("db", "database", "dataset", "data", "x"):
            return True
        if arg.annotation:
            ann_str = ast.dump(arg.annotation)
            if "Database" in ann_str or "Dataset" in ann_str or "List" in ann_str:
                return True
        return False

    def _annotation_to_type(self, annotation: ast.expr | None) -> IRType:
        """Convert a Python type annotation to an IRType."""
        if annotation is None:
            return IRType.REAL
        if isinstance(annotation, ast.Name):
            type_map: dict[str, IRType] = {
                "int": IRType.INT,
                "float": IRType.REAL,
                "bool": IRType.BOOL,
                "list": IRType.ARRAY,
                "List": IRType.ARRAY,
                "tuple": IRType.TUPLE,
                "Tuple": IRType.TUPLE,
            }
            return type_map.get(annotation.id, IRType.REAL)
        if isinstance(annotation, ast.Constant):
            if isinstance(annotation.value, str):
                type_map2: dict[str, IRType] = {
                    "int": IRType.INT, "float": IRType.REAL, "bool": IRType.BOOL,
                }
                return type_map2.get(annotation.value, IRType.REAL)
        return IRType.REAL

    def _infer_return_type(self, func: ast.FunctionDef) -> IRType:
        """Infer the return type of a function."""
        if func.returns:
            return self._annotation_to_type(func.returns)
        # Check for return statements
        for node in ast.walk(func):
            if isinstance(node, ast.Return) and node.value:
                return self._infer_expr_type(node.value)
        return IRType.REAL

    # ------------------------------------------------------------------
    # Decorator parsing
    # ------------------------------------------------------------------

    def _parse_decorators(
        self, decorator_list: list[ast.expr]
    ) -> DecoratorInfo:
        """Parse DP-specific decorators from a function.

        Recognises:
        - @dp_mechanism(epsilon=..., delta=...)
        - @sensitivity(value=...)
        """
        info = DecoratorInfo()
        for dec in decorator_list:
            name = self._decorator_name(dec)
            if name == "dp_mechanism":
                info.is_mechanism = True
                if isinstance(dec, ast.Call):
                    for kw in dec.keywords:
                        val = self._const_value(kw.value)
                        if kw.arg == "epsilon" and val is not None:
                            info.epsilon = float(val)
                        elif kw.arg == "delta" and val is not None:
                            info.delta = float(val)
                        elif kw.arg == "noise" and isinstance(val, str):
                            info.noise_type = _NOISE_NAMES.get(val.lower())
                        elif kw.arg is not None and val is not None:
                            info.metadata[kw.arg] = val
            elif name == "sensitivity":
                if isinstance(dec, ast.Call):
                    # positional arg or keyword
                    if dec.args:
                        val = self._const_value(dec.args[0])
                        if val is not None:
                            info.sensitivity = float(val)
                    for kw in dec.keywords:
                        val = self._const_value(kw.value)
                        if kw.arg == "value" and val is not None:
                            info.sensitivity = float(val)
                        elif kw.arg == "norm" and isinstance(val, str):
                            info.metadata["sensitivity_norm"] = val
        return info

    def _decorator_name(self, dec: ast.expr) -> str:
        """Extract the name from a decorator expression."""
        if isinstance(dec, ast.Name):
            return dec.id
        if isinstance(dec, ast.Call):
            if isinstance(dec.func, ast.Name):
                return dec.func.id
            if isinstance(dec.func, ast.Attribute):
                return dec.func.attr
        if isinstance(dec, ast.Attribute):
            return dec.attr
        return ""

    # ------------------------------------------------------------------
    # Body lowering
    # ------------------------------------------------------------------

    def _lower_body(self, stmts: list[ast.stmt]) -> IRNode:
        """Lower a list of Python statements to a single IR node.

        Args:
            stmts: List of AST statement nodes.

        Returns:
            A :class:`SequenceNode` (or single node if only one stmt).
        """
        ir_nodes: list[IRNode] = []
        for stmt in stmts:
            nodes = self._lower_stmt(stmt)
            ir_nodes.extend(nodes)

        if len(ir_nodes) == 0:
            return NoOpNode()
        if len(ir_nodes) == 1:
            return ir_nodes[0]
        return SequenceNode(stmts=ir_nodes)

    def _lower_stmt(self, stmt: ast.stmt) -> list[IRNode]:
        """Lower a single Python statement to IR node(s).

        Returns a list because some statements expand to multiple IR nodes.
        """
        loc = self._loc(stmt)

        if isinstance(stmt, ast.Assign):
            return self._lower_assign(stmt, loc)
        if isinstance(stmt, ast.AugAssign):
            return self._lower_aug_assign(stmt, loc)
        if isinstance(stmt, ast.AnnAssign):
            return self._lower_ann_assign(stmt, loc)
        if isinstance(stmt, ast.Return):
            return self._lower_return(stmt, loc)
        if isinstance(stmt, ast.If):
            return [self._lower_if(stmt, loc)]
        if isinstance(stmt, ast.For):
            return [self._lower_for(stmt, loc)]
        if isinstance(stmt, ast.Expr):
            # Expression statement (e.g., docstrings, bare calls)
            if isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
                return []  # docstring, skip
            return self._lower_expr_stmt(stmt, loc)
        if isinstance(stmt, ast.Pass):
            return []
        if isinstance(stmt, ast.FunctionDef):
            self._error(
                "nested function definitions are not supported in DPImp",
                stmt,
            )
            return []
        if isinstance(stmt, ast.ClassDef):
            self._error(
                "class definitions are not supported in DPImp",
                stmt,
            )
            return []
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            # Imports are handled by the preprocessor
            return []
        if isinstance(stmt, ast.While):
            self._error(
                "while loops are not supported in DPImp; use bounded for loops",
                stmt,
            )
            return []
        if isinstance(stmt, ast.Assert):
            # Assertions stripped during lowering
            return []

        self._error(f"unsupported statement type: {type(stmt).__name__}", stmt)
        return []

    # ------------------------------------------------------------------
    # Assignment lowering
    # ------------------------------------------------------------------

    def _lower_assign(
        self, stmt: ast.Assign, loc: SourceLoc | None
    ) -> list[IRNode]:
        """Lower a simple assignment: target = value."""
        nodes: list[IRNode] = []
        for target in stmt.targets:
            if isinstance(target, ast.Name):
                value_expr = self._lower_expr(stmt.value)
                # Check if RHS is a noise call
                noise = self._check_noise_call(stmt.value, target.id)
                if noise is not None:
                    noise.source_loc = loc
                    nodes.append(noise)
                    continue
                # Check if RHS is a query call
                query = self._check_query_call(stmt.value, target.id)
                if query is not None:
                    query.source_loc = loc
                    nodes.append(query)
                    continue
                # Regular assignment
                ty = self._infer_expr_type(stmt.value)
                var = self._env.define(target.id, ty)
                node = AssignNode(target=var, value=value_expr, source_loc=loc)
                self._register_loc(node, stmt)
                nodes.append(node)
            elif isinstance(target, ast.Tuple):
                # Tuple unpacking: a, b = expr
                self._error(
                    "tuple unpacking assignments are not yet supported",
                    stmt,
                )
            else:
                self._error(
                    f"unsupported assignment target: {type(target).__name__}",
                    stmt,
                )
        return nodes

    def _lower_aug_assign(
        self, stmt: ast.AugAssign, loc: SourceLoc | None
    ) -> list[IRNode]:
        """Lower an augmented assignment: target op= value.

        Desugars ``x += e`` into ``x = x + e``.
        """
        if not isinstance(stmt.target, ast.Name):
            self._error("augmented assignment target must be a name", stmt)
            return []

        name = stmt.target.id
        current_var = self._env.lookup(name)
        if current_var is None:
            self._error(f"undefined variable in augmented assignment: {name}", stmt)
            return []

        op_kind = _BINOP_MAP.get(type(stmt.op))
        if op_kind is None:
            self._error(
                f"unsupported augmented assignment operator: {type(stmt.op).__name__}",
                stmt,
            )
            return []

        rhs_expr = self._lower_expr(stmt.value)
        result_ty = self._binop_result_type(current_var.ty, rhs_expr.ty, op_kind)
        combined = IRBinOp(
            ty=result_ty, op=op_kind, left=current_var, right=rhs_expr
        )
        new_var = self._env.define(name, result_ty)
        node = AssignNode(target=new_var, value=combined, source_loc=loc)
        self._register_loc(node, stmt)
        return [node]

    def _lower_ann_assign(
        self, stmt: ast.AnnAssign, loc: SourceLoc | None
    ) -> list[IRNode]:
        """Lower an annotated assignment: target: type = value."""
        if stmt.value is None:
            # Declaration only, no assignment
            if isinstance(stmt.target, ast.Name):
                ty = self._annotation_to_type(stmt.annotation)
                self._env.define(stmt.target.id, ty)
            return []

        if isinstance(stmt.target, ast.Name):
            ty = self._annotation_to_type(stmt.annotation)
            value_expr = self._lower_expr(stmt.value)
            # Check noise/query
            noise = self._check_noise_call(stmt.value, stmt.target.id)
            if noise is not None:
                noise.source_loc = loc
                return [noise]
            query = self._check_query_call(stmt.value, stmt.target.id)
            if query is not None:
                query.source_loc = loc
                return [query]
            var = self._env.define(stmt.target.id, ty)
            node = AssignNode(target=var, value=value_expr, source_loc=loc)
            self._register_loc(node, stmt)
            return [node]

        self._error("unsupported annotated assignment target", stmt)
        return []

    # ------------------------------------------------------------------
    # Control flow lowering
    # ------------------------------------------------------------------

    def _lower_return(
        self, stmt: ast.Return, loc: SourceLoc | None
    ) -> list[IRNode]:
        """Lower a return statement."""
        if stmt.value is None:
            expr: TypedExpr = Const.zero()
        else:
            expr = self._lower_expr(stmt.value)
        node = ReturnNode(value=expr, source_loc=loc)
        self._register_loc(node, stmt)
        return [node]

    def _lower_if(
        self, stmt: ast.If, loc: SourceLoc | None
    ) -> IRNode:
        """Lower an if/elif/else chain to nested BranchNodes."""
        cond_expr = self._lower_expr(stmt.test)
        if cond_expr.ty != IRType.BOOL:
            cond_expr = self._coerce_to_bool(cond_expr)

        # Save SSA state for merge
        pre_branch = self._env.snapshot()

        self._env.push_scope()
        true_body = self._lower_body(stmt.body)
        true_versions = self._env.snapshot()
        self._env.pop_scope()

        self._env.push_scope()
        if stmt.orelse:
            if len(stmt.orelse) == 1 and isinstance(stmt.orelse[0], ast.If):
                # elif chain
                false_body = self._lower_if(stmt.orelse[0], self._loc(stmt.orelse[0]))
            else:
                false_body = self._lower_body(stmt.orelse)
        else:
            false_body = NoOpNode()
        false_versions = self._env.snapshot()
        self._env.pop_scope()

        branch = BranchNode(
            condition=cond_expr,
            true_branch=true_body,
            false_branch=false_body,
            source_loc=loc,
        )
        self._register_loc(branch, stmt)

        # Insert merge nodes for variables modified in branches
        merge_nodes = self._create_merge_nodes(
            pre_branch, true_versions, false_versions, branch.node_id
        )
        if merge_nodes:
            return SequenceNode(stmts=[branch] + merge_nodes)
        return branch

    def _lower_for(
        self, stmt: ast.For, loc: SourceLoc | None
    ) -> IRNode:
        """Lower a for-loop with range() to a LoopNode.

        Only ``for x in range(N)`` is supported. Other iteration patterns
        are rejected.
        """
        if not isinstance(stmt.target, ast.Name):
            self._error("for-loop target must be a simple name", stmt)
            return NoOpNode(source_loc=loc)

        bound = self._extract_range_bound(stmt.iter, stmt)
        if bound is None:
            return NoOpNode(source_loc=loc)

        index_var = self._env.define(stmt.target.id, IRType.INT)

        self._env.push_scope()
        body = self._lower_body(stmt.body)
        self._env.pop_scope()

        node = LoopNode(
            index_var=index_var,
            bound=bound,
            body=body,
            source_loc=loc,
        )
        self._register_loc(node, stmt)
        return node

    def _extract_range_bound(
        self, iter_expr: ast.expr, stmt: ast.stmt
    ) -> TypedExpr | None:
        """Extract the upper bound from a range() call.

        Handles range(N) and range(0, N). Returns ``None`` on error.
        """
        if not isinstance(iter_expr, ast.Call):
            self._error(
                "for-loop iterator must be a range() call", stmt
            )
            return None

        func = iter_expr.func
        if isinstance(func, ast.Name) and func.id == "range":
            args = iter_expr.args
            if len(args) == 1:
                return self._lower_expr(args[0])
            if len(args) == 2:
                # range(start, stop) — we only use stop
                return self._lower_expr(args[1])
            if len(args) == 3:
                # range(start, stop, step)
                return self._lower_expr(args[1])
            self._error("range() requires 1-3 arguments", stmt)
            return None

        self._error(
            "for-loop iterator must be a range() call; "
            f"got {ast.dump(func)}",
            stmt,
        )
        return None

    # ------------------------------------------------------------------
    # Expression statement lowering
    # ------------------------------------------------------------------

    def _lower_expr_stmt(
        self, stmt: ast.Expr, loc: SourceLoc | None
    ) -> list[IRNode]:
        """Lower an expression statement (bare function call, etc.)."""
        if isinstance(stmt.value, ast.Call):
            # Check for noise calls without assignment
            name = self._call_name(stmt.value)
            if name and name.lower() in _NOISE_NAMES:
                tmp = self._fresh_temp()
                noise = self._check_noise_call(stmt.value, tmp)
                if noise:
                    noise.source_loc = loc
                    return [noise]
        return []

    # ------------------------------------------------------------------
    # Expression lowering
    # ------------------------------------------------------------------

    def _lower_expr(self, expr: ast.expr) -> TypedExpr:
        """Lower a Python expression to a MechIR TypedExpr.

        Args:
            expr: The AST expression node.

        Returns:
            An IR :class:`TypedExpr`.
        """
        if isinstance(expr, ast.Constant):
            return self._lower_constant(expr)
        if isinstance(expr, ast.Name):
            return self._lower_name(expr)
        if isinstance(expr, ast.BinOp):
            return self._lower_binop(expr)
        if isinstance(expr, ast.UnaryOp):
            return self._lower_unaryop(expr)
        if isinstance(expr, ast.BoolOp):
            return self._lower_boolop(expr)
        if isinstance(expr, ast.Compare):
            return self._lower_compare(expr)
        if isinstance(expr, ast.Call):
            return self._lower_call(expr)
        if isinstance(expr, ast.IfExp):
            return self._lower_ifexp(expr)
        if isinstance(expr, ast.Subscript):
            return self._lower_subscript(expr)
        if isinstance(expr, ast.Attribute):
            return self._lower_attribute(expr)
        if isinstance(expr, ast.NameConstant):
            # Python 3.7 compat
            return self._lower_name_constant(expr)

        self._error(f"unsupported expression type: {type(expr).__name__}", expr)
        return Const.zero()

    def _lower_constant(self, node: ast.Constant) -> TypedExpr:
        """Lower a constant literal."""
        val = node.value
        if isinstance(val, bool):
            return Const.bool_(val)
        if isinstance(val, int):
            return Const.int_(val)
        if isinstance(val, float):
            return Const.real(val)
        if val is None:
            return Const.zero()
        # String constants are not supported as values
        return Const.zero()

    def _lower_name(self, node: ast.Name) -> TypedExpr:
        """Lower a variable reference."""
        var = self._env.lookup(node.id)
        if var is not None:
            return var
        # Could be a built-in or not-yet-defined
        return Var(ty=IRType.REAL, name=node.id)

    def _lower_name_constant(self, node: Any) -> TypedExpr:
        """Lower a NameConstant (Python 3.7 compat)."""
        val = node.value
        if isinstance(val, bool):
            return Const.bool_(val)
        if val is None:
            return Const.zero()
        return Const.zero()

    def _lower_binop(self, node: ast.BinOp) -> TypedExpr:
        """Lower a binary operation."""
        left = self._lower_expr(node.left)
        right = self._lower_expr(node.right)
        op_kind = _BINOP_MAP.get(type(node.op))
        if op_kind is None:
            self._error(f"unsupported binary operator: {type(node.op).__name__}", node)
            return Const.zero()
        result_ty = self._binop_result_type(left.ty, right.ty, op_kind)
        return IRBinOp(ty=result_ty, op=op_kind, left=left, right=right)

    def _lower_unaryop(self, node: ast.UnaryOp) -> TypedExpr:
        """Lower a unary operation."""
        operand = self._lower_expr(node.operand)
        op_kind = _UNARYOP_MAP.get(type(node.op))
        if op_kind is None:
            self._error(
                f"unsupported unary operator: {type(node.op).__name__}", node
            )
            return operand
        if op_kind == UnaryOpKind.NOT:
            return IRUnaryOp(ty=IRType.BOOL, op=op_kind, operand=operand)
        return IRUnaryOp(ty=operand.ty, op=op_kind, operand=operand)

    def _lower_boolop(self, node: ast.BoolOp) -> TypedExpr:
        """Lower a boolean operation (and/or) to a chain of BinOps."""
        op_kind = _BOOLOP_MAP[type(node.op)]
        result = self._lower_expr(node.values[0])
        for val in node.values[1:]:
            right = self._lower_expr(val)
            result = IRBinOp(ty=IRType.BOOL, op=op_kind, left=result, right=right)
        return result

    def _lower_compare(self, node: ast.Compare) -> TypedExpr:
        """Lower a comparison to a chain of BinOps."""
        left = self._lower_expr(node.left)
        result: TypedExpr | None = None
        for op, comparator in zip(node.ops, node.comparators):
            right = self._lower_expr(comparator)
            op_kind = _CMPOP_MAP.get(type(op))
            if op_kind is None:
                self._error(
                    f"unsupported comparison operator: {type(op).__name__}",
                    node,
                )
                return Const.bool_(True)
            cmp_expr = IRBinOp(ty=IRType.BOOL, op=op_kind, left=left, right=right)
            if result is None:
                result = cmp_expr
            else:
                result = IRBinOp(
                    ty=IRType.BOOL, op=BinOpKind.AND, left=result, right=cmp_expr
                )
            left = right
        return result if result is not None else Const.bool_(True)

    def _lower_call(self, node: ast.Call) -> TypedExpr:
        """Lower a function call expression."""
        name = self._call_name(node)
        if not name:
            self._error("unsupported call expression", node)
            return Const.zero()

        # Check for recursion
        if name == self._current_func:
            self._error(
                f"recursion is not allowed in DPImp: {name} calls itself",
                node,
            )
            return Const.zero()

        # Built-in math functions
        math_funcs = {"abs", "max", "min", "len", "int", "float", "round"}
        if name in math_funcs:
            return self._lower_math_call(name, node)

        # Generic function call
        args = tuple(self._lower_expr(a) for a in node.args)
        return FuncCall(ty=IRType.REAL, name=name, args=args)

    def _lower_math_call(self, name: str, node: ast.Call) -> TypedExpr:
        """Lower a built-in math function call."""
        args = [self._lower_expr(a) for a in node.args]
        if name == "abs" and len(args) == 1:
            from dpcegar.ir.types import Abs
            return Abs(ty=args[0].ty, operand=args[0])
        if name == "max" and len(args) == 2:
            from dpcegar.ir.types import Max
            return Max(ty=args[0].ty, left=args[0], right=args[1])
        if name == "min" and len(args) == 2:
            from dpcegar.ir.types import Min
            return Min(ty=args[0].ty, left=args[0], right=args[1])
        if name == "int" and len(args) == 1:
            return FuncCall(ty=IRType.INT, name="int", args=(args[0],))
        if name == "float" and len(args) == 1:
            return FuncCall(ty=IRType.REAL, name="float", args=(args[0],))
        if name == "len" and len(args) == 1:
            return FuncCall(ty=IRType.INT, name="len", args=(args[0],))
        if name == "round" and len(args) >= 1:
            return FuncCall(ty=IRType.INT, name="round", args=tuple(args))
        return FuncCall(ty=IRType.REAL, name=name, args=tuple(args))

    def _lower_ifexp(self, node: ast.IfExp) -> TypedExpr:
        """Lower a conditional expression (ternary)."""
        from dpcegar.ir.types import Cond
        cond = self._lower_expr(node.test)
        true_val = self._lower_expr(node.body)
        false_val = self._lower_expr(node.orelse)
        if cond.ty != IRType.BOOL:
            cond = self._coerce_to_bool(cond)
        return Cond(
            ty=true_val.ty,
            condition=cond,
            true_expr=true_val,
            false_expr=false_val,
        )

    def _lower_subscript(self, node: ast.Subscript) -> TypedExpr:
        """Lower a subscript expression (array access)."""
        from dpcegar.ir.types import ArrayAccess
        array = self._lower_expr(node.value)
        if isinstance(node.slice, ast.Index):
            # Python 3.8 compat
            index = self._lower_expr(node.slice.value)  # type: ignore[attr-defined]
        else:
            index = self._lower_expr(node.slice)
        return ArrayAccess(ty=IRType.REAL, array=array, index=index)

    def _lower_attribute(self, node: ast.Attribute) -> TypedExpr:
        """Lower an attribute access expression."""
        value = self._lower_expr(node.value)
        return FuncCall(
            ty=IRType.REAL,
            name=f"getattr_{node.attr}",
            args=(value,),
        )

    # ------------------------------------------------------------------
    # Noise and query detection
    # ------------------------------------------------------------------

    def _check_noise_call(
        self, expr: ast.expr, target_name: str
    ) -> NoiseDrawNode | None:
        """Check if an expression is a noise primitive call.

        Returns a NoiseDrawNode if so, otherwise None.
        """
        if not isinstance(expr, ast.Call):
            return None
        name = self._call_name(expr)
        if name is None:
            return None
        noise_kind = _NOISE_NAMES.get(name.lower())
        if noise_kind is None:
            return None

        ty = IRType.REAL
        var = self._env.define(target_name, ty)

        # Parse arguments: noise(center, scale) or noise(value, scale=...)
        center = Const.zero()
        scale = Const.one()
        sensitivity: TypedExpr | None = None

        args = expr.args
        if len(args) >= 1:
            center = self._lower_expr(args[0])
        if len(args) >= 2:
            scale = self._lower_expr(args[1])

        for kw in expr.keywords:
            val = self._lower_expr(kw.value)
            if kw.arg in ("scale", "b", "sigma", "noise_scale"):
                scale = val
            elif kw.arg in ("loc", "center", "mu", "value", "mean"):
                center = val
            elif kw.arg in ("sensitivity", "sens", "delta_f"):
                sensitivity = val

        # Check decorator-level sensitivity
        if sensitivity is None and self._decorator_info.sensitivity is not None:
            sensitivity = Const.real(self._decorator_info.sensitivity)

        # Check comment-based sensitivity for this variable
        if sensitivity is None and target_name in self._sensitivity_annotations:
            sensitivity = Const.real(self._sensitivity_annotations[target_name])

        return NoiseDrawNode(
            target=var,
            noise_kind=noise_kind,
            center=center,
            scale=scale,
            sensitivity=sensitivity,
        )

    def _check_query_call(
        self, expr: ast.expr, target_name: str
    ) -> QueryNode | None:
        """Check if an expression is a query function call.

        Returns a QueryNode if so, otherwise None.
        """
        if not isinstance(expr, ast.Call):
            return None
        name = self._call_name(expr)
        if name is None:
            return None

        is_query = any(name.startswith(prefix) for prefix in _QUERY_PREFIXES)
        if not is_query:
            return None

        ty = IRType.REAL
        var = self._env.define(target_name, ty)

        args = tuple(self._lower_expr(a) for a in expr.args)

        sensitivity = Const.one()
        for kw in expr.keywords:
            if kw.arg in ("sensitivity", "sens"):
                sensitivity = self._lower_expr(kw.value)

        if name in self._sensitivity_annotations:
            sensitivity = Const.real(self._sensitivity_annotations[name])

        return QueryNode(
            target=var,
            query_name=name,
            args=args,
            sensitivity=sensitivity,
        )

    # ------------------------------------------------------------------
    # SSA merge helpers
    # ------------------------------------------------------------------

    def _create_merge_nodes(
        self,
        pre_branch: dict[str, int],
        true_versions: dict[str, int],
        false_versions: dict[str, int],
        branch_id: int,
    ) -> list[IRNode]:
        """Create MergeNodes for variables modified in branches."""
        merges: list[IRNode] = []
        all_vars = set(true_versions.keys()) | set(false_versions.keys())
        for name in sorted(all_vars):
            tv = true_versions.get(name)
            fv = false_versions.get(name)
            pv = pre_branch.get(name)
            if tv is not None and fv is not None and tv != fv:
                ty = self._env.lookup_type(name)
                new_var = self._env.define(name, ty)
                sources: dict[int, TypedExpr] = {
                    branch_id: Var(ty=ty, name=name, version=tv),
                    branch_id + 1: Var(ty=ty, name=name, version=fv),
                }
                merges.append(MergeNode(target=new_var, sources=sources))
            elif tv is not None and pv is not None and tv != pv:
                ty = self._env.lookup_type(name)
                new_var = self._env.define(name, ty)
                sources = {
                    branch_id: Var(ty=ty, name=name, version=tv),
                    branch_id + 1: Var(ty=ty, name=name, version=pv),
                }
                merges.append(MergeNode(target=new_var, sources=sources))
        return merges

    # ------------------------------------------------------------------
    # Sensitivity comment extraction
    # ------------------------------------------------------------------

    def _extract_sensitivity_comments(self, source: str) -> None:
        """Extract sensitivity annotations from source comments.

        Recognises patterns like:
        - ``# sensitivity: 1.0``
        - ``# sensitivity(query_name): 2.0``
        - ``# sens: L1 = 1.0``
        """
        import re
        pattern = re.compile(
            r"#\s*sensitivity(?:\((\w+)\))?:\s*(?:L[12∞]\s*=\s*)?([\d.]+)"
        )
        for i, line in enumerate(source.splitlines(), 1):
            m = pattern.search(line)
            if m:
                name = m.group(1) or f"_line_{i}"
                value = float(m.group(2))
                self._sensitivity_annotations[name] = value

    # ------------------------------------------------------------------
    # Type inference helpers
    # ------------------------------------------------------------------

    def _infer_expr_type(self, expr: ast.expr) -> IRType:
        """Infer the IR type of a Python expression.

        This is a best-effort heuristic used during lowering.
        """
        if isinstance(expr, ast.Constant):
            if isinstance(expr.value, bool):
                return IRType.BOOL
            if isinstance(expr.value, int):
                return IRType.INT
            if isinstance(expr.value, float):
                return IRType.REAL
            return IRType.REAL
        if isinstance(expr, ast.Name):
            return self._env.lookup_type(expr.id)
        if isinstance(expr, ast.BinOp):
            lt = self._infer_expr_type(expr.left)
            rt = self._infer_expr_type(expr.right)
            if isinstance(expr.op, ast.Div):
                return IRType.REAL
            if lt == IRType.REAL or rt == IRType.REAL:
                return IRType.REAL
            return IRType.INT
        if isinstance(expr, ast.UnaryOp):
            if isinstance(expr.op, ast.Not):
                return IRType.BOOL
            return self._infer_expr_type(expr.operand)
        if isinstance(expr, ast.BoolOp):
            return IRType.BOOL
        if isinstance(expr, ast.Compare):
            return IRType.BOOL
        if isinstance(expr, ast.Call):
            name = self._call_name(expr)
            if name in ("int", "len", "round"):
                return IRType.INT
            if name in ("float",):
                return IRType.REAL
            if name in ("bool",):
                return IRType.BOOL
            return IRType.REAL
        if isinstance(expr, ast.IfExp):
            return self._infer_expr_type(expr.body)
        return IRType.REAL

    def _binop_result_type(
        self, left_ty: IRType, right_ty: IRType, op: BinOpKind
    ) -> IRType:
        """Determine the result type of a binary operation."""
        if op.is_comparison or op.is_logical:
            return IRType.BOOL
        if op == BinOpKind.DIV:
            return IRType.REAL
        if left_ty == IRType.REAL or right_ty == IRType.REAL:
            return IRType.REAL
        return IRType.INT

    def _coerce_to_bool(self, expr: TypedExpr) -> TypedExpr:
        """Coerce a numeric expression to boolean (nonzero test)."""
        return IRBinOp(
            ty=IRType.BOOL,
            op=BinOpKind.NEQ,
            left=expr,
            right=Const.zero(expr.ty),
        )

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _call_name(self, node: ast.Call) -> str | None:
        """Extract the function name from a Call node."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        if isinstance(node.func, ast.Attribute):
            return node.func.attr
        return None

    def _const_value(self, node: ast.expr) -> Any:
        """Extract a constant value from an AST expression, or None."""
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Num):
            return node.n  # type: ignore[attr-defined]
        if isinstance(node, ast.Str):
            return node.s  # type: ignore[attr-defined]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            inner = self._const_value(node.operand)
            if inner is not None:
                return -inner
        return None

    def _fresh_temp(self) -> str:
        """Generate a fresh temporary variable name."""
        self._temp_counter += 1
        return f"_tmp_{self._temp_counter}"

    def _loc(self, node: ast.AST) -> SourceLoc | None:
        """Extract a SourceLoc from an AST node."""
        if hasattr(node, "lineno"):
            return SourceLoc(
                file=self.file,
                line=node.lineno,
                col=getattr(node, "col_offset", 0) + 1,
                end_line=getattr(node, "end_lineno", None),
                end_col=getattr(node, "end_col_offset", None),
            )
        return None

    def _register_loc(self, ir_node: IRNode, ast_node: ast.AST) -> None:
        """Register the source location of an IR node."""
        if hasattr(ast_node, "lineno"):
            self.source_map.add(
                ir_node.node_id,
                SourceRange(
                    file=self.file,
                    start_line=ast_node.lineno,
                    start_col=getattr(ast_node, "col_offset", 0) + 1,
                    end_line=getattr(ast_node, "end_lineno", ast_node.lineno),
                    end_col=getattr(ast_node, "end_col_offset", 0),
                ),
            )

    def _error(self, message: str, node: ast.AST | None = None) -> None:
        """Record a lowering error."""
        loc = self._loc(node) if node else None
        err = ASTBridgeError(message, source_loc=loc)
        self.errors.append(err)


# ═══════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════


def parse_mechanism(
    source: str,
    file: str = "<unknown>",
) -> MechIR:
    """Parse DPImp source code and return a MechIR.

    This is the main entry point for converting mechanism source code
    into the IR used by the verification engine.

    Args:
        source: Python source code defining a DP mechanism.
        file:   Source file name for error messages.

    Returns:
        A complete :class:`MechIR` representation.

    Raises:
        ASTBridgeError: If the source cannot be lowered.
    """
    visitor = ASTVisitor(file=file)
    mechir = visitor.parse(source)
    if visitor.errors:
        raise visitor.errors[0]
    return mechir


def parse_mechanism_lenient(
    source: str,
    file: str = "<unknown>",
) -> tuple[MechIR, list[ASTBridgeError]]:
    """Parse DPImp source code, collecting errors without raising.

    Args:
        source: Python source code defining a DP mechanism.
        file:   Source file name for error messages.

    Returns:
        A tuple of (MechIR, errors). The MechIR may be partial if
        errors occurred.
    """
    visitor = ASTVisitor(file=file)
    try:
        mechir = visitor.parse(source)
    except ASTBridgeError as e:
        mechir = MechIR(name="<error>")
        visitor.errors.append(e)
    return mechir, visitor.errors


def get_source_map(
    source: str,
    file: str = "<unknown>",
) -> tuple[MechIR, SourceMap]:
    """Parse source and return both the MechIR and its source map.

    Args:
        source: Python source code.
        file:   Source file name.

    Returns:
        A tuple of (MechIR, SourceMap).
    """
    visitor = ASTVisitor(file=file)
    mechir = visitor.parse(source)
    return mechir, visitor.source_map
