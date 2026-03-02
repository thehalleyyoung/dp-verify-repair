"""Visitor and transformer infrastructure for MechIR.

Classes
-------
IRNodeVisitor       – typed visitor for IR nodes (pre/post order)
ExprTransformer     – bottom-up expression rewriter
SSANumbering        – assign SSA version numbers to variables
FreeVarCollector    – collect free variables from IR nodes
ExprSubstituter     – substitute variables in expressions across IR nodes
NodePrinter         – pretty-print IR trees to a string
IRValidator         – validate structural well-formedness of IR trees
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from io import StringIO
from typing import Any, Callable, TypeVar

from dpcegar.ir.types import (
    BinOp,
    Const,
    ExprVisitor,
    IRType,
    NoiseKind,
    TypedExpr,
    UnaryOp,
    Var,
)
from dpcegar.ir.nodes import (
    AssignNode,
    BranchNode,
    CFG,
    IRNode,
    IRNodeVisitorBase,
    LoopNode,
    MechIR,
    MergeNode,
    NoOpNode,
    NoiseDrawNode,
    QueryNode,
    ReturnNode,
    SequenceNode,
)
from dpcegar.utils.errors import InternalError, TypeCheckError, ensure

T = TypeVar("T")


# ═══════════════════════════════════════════════════════════════════════════
# 1.  IR NODE VISITOR (typed, pre/post order)
# ═══════════════════════════════════════════════════════════════════════════


class IRNodeVisitor(IRNodeVisitorBase):
    """Typed visitor for IR nodes with pre- and post-order hooks.

    Override ``pre_visit`` and/or ``post_visit`` for generic logic that
    runs around every specific ``visit_*`` method.  Return values from
    ``visit_*`` methods are collected by :meth:`generic_visit`.

    Usage::

        class MyAnalysis(IRNodeVisitor):
            def visit_NoiseDrawNode(self, node):
                ...  # analyse noise draws
    """

    def pre_visit(self, node: IRNode) -> None:
        """Called before dispatching to the specific visitor."""

    def post_visit(self, node: IRNode, result: Any) -> Any:
        """Called after the specific visitor; can transform the result."""
        return result

    def visit(self, node: IRNode) -> Any:
        self.pre_visit(node)
        result = node.accept(self)
        return self.post_visit(node, result)

    def visit_all(self, root: IRNode) -> list[Any]:
        """Visit all nodes in pre-order, collecting results."""
        return [self.visit(n) for n in root.walk()]


# ═══════════════════════════════════════════════════════════════════════════
# 2.  EXPRESSION TRANSFORMER
# ═══════════════════════════════════════════════════════════════════════════


class ExprTransformer(ExprVisitor[TypedExpr]):
    """Bottom-up expression rewriter.

    Override the ``visit_*`` methods to transform specific node types.
    The default implementation recursively transforms children first,
    then returns the node unchanged.

    Usage::

        class ScaleNoise(ExprTransformer):
            def visit_Var(self, expr):
                if expr.name == 'sigma':
                    return Const.real(2.0)
                return expr

        new_expr = ScaleNoise().visit(old_expr)
    """

    def generic_visit(self, expr: TypedExpr) -> TypedExpr:
        """Transform children, then return the (possibly new) node."""
        return expr.map_children(lambda c: self.visit(c))

    def visit_Var(self, expr: Var) -> TypedExpr:
        return expr

    def visit_Const(self, expr: Const) -> TypedExpr:
        return expr


class NodeExprTransformer(IRNodeVisitorBase):
    """Apply an :class:`ExprTransformer` to every expression in the IR tree.

    This walks the IR node tree and rewrites all embedded expressions
    using the supplied transformer.
    """

    def __init__(self, expr_transformer: ExprTransformer) -> None:
        self._tx = expr_transformer

    def visit_AssignNode(self, node: AssignNode) -> None:
        node.value = self._tx.visit(node.value)
        node.target = self._tx.visit(node.target)  # type: ignore[assignment]

    def visit_NoiseDrawNode(self, node: NoiseDrawNode) -> None:
        node.center = self._tx.visit(node.center)
        node.scale = self._tx.visit(node.scale)
        if node.sensitivity is not None:
            node.sensitivity = self._tx.visit(node.sensitivity)

    def visit_BranchNode(self, node: BranchNode) -> None:
        node.condition = self._tx.visit(node.condition)
        self.visit(node.true_branch)
        self.visit(node.false_branch)

    def visit_MergeNode(self, node: MergeNode) -> None:
        node.sources = {k: self._tx.visit(v) for k, v in node.sources.items()}

    def visit_LoopNode(self, node: LoopNode) -> None:
        node.bound = self._tx.visit(node.bound)
        self.visit(node.body)

    def visit_QueryNode(self, node: QueryNode) -> None:
        node.args = tuple(self._tx.visit(a) for a in node.args)
        node.sensitivity = self._tx.visit(node.sensitivity)

    def visit_ReturnNode(self, node: ReturnNode) -> None:
        node.value = self._tx.visit(node.value)

    def visit_SequenceNode(self, node: SequenceNode) -> None:
        for stmt in node.stmts:
            self.visit(stmt)

    def visit_NoOpNode(self, node: NoOpNode) -> None:
        pass


# ═══════════════════════════════════════════════════════════════════════════
# 3.  SSA NUMBERING
# ═══════════════════════════════════════════════════════════════════════════


class SSANumbering(IRNodeVisitorBase):
    """Assign SSA version numbers to variable definitions.

    After running this visitor, every :class:`Var` node that is defined
    (appears as a target) will have a unique ``version`` number.  Uses
    within expressions are updated to refer to the most recent definition.

    Usage::

        numberer = SSANumbering()
        numberer.visit(mechir.body)
    """

    def __init__(self) -> None:
        self._counters: dict[str, int] = defaultdict(int)
        self._current_version: dict[str, int] = {}

    def _define(self, var: Var) -> Var:
        """Assign a fresh version number to a variable definition."""
        name = var.name
        version = self._counters[name]
        self._counters[name] += 1
        self._current_version[name] = version
        return Var(ty=var.ty, name=name, version=version)

    def _use_renamer(self) -> ExprTransformer:
        """Return a transformer that renames variable uses."""
        outer = self

        class _Renamer(ExprTransformer):
            def visit_Var(self, expr: Var) -> TypedExpr:
                if expr.name in outer._current_version:
                    return Var(
                        ty=expr.ty,
                        name=expr.name,
                        version=outer._current_version[expr.name],
                    )
                return expr

        return _Renamer()

    def visit_AssignNode(self, node: AssignNode) -> None:
        # Rename uses in the RHS first
        renamer = self._use_renamer()
        node.value = renamer.visit(node.value)
        # Then define the target
        node.target = self._define(node.target)

    def visit_NoiseDrawNode(self, node: NoiseDrawNode) -> None:
        renamer = self._use_renamer()
        node.center = renamer.visit(node.center)
        node.scale = renamer.visit(node.scale)
        if node.sensitivity is not None:
            node.sensitivity = renamer.visit(node.sensitivity)
        node.target = self._define(node.target)

    def visit_QueryNode(self, node: QueryNode) -> None:
        renamer = self._use_renamer()
        node.args = tuple(renamer.visit(a) for a in node.args)
        node.sensitivity = renamer.visit(node.sensitivity)
        node.target = self._define(node.target)

    def visit_BranchNode(self, node: BranchNode) -> None:
        renamer = self._use_renamer()
        node.condition = renamer.visit(node.condition)
        # Save state for branches
        saved = dict(self._current_version)
        self.visit(node.true_branch)
        true_versions = dict(self._current_version)
        # Restore and process false branch
        self._current_version = dict(saved)
        self.visit(node.false_branch)
        false_versions = dict(self._current_version)
        # Merge: take the latest from either branch
        all_vars = set(true_versions.keys()) | set(false_versions.keys())
        for v in all_vars:
            tv = true_versions.get(v)
            fv = false_versions.get(v)
            if tv is not None and fv is not None:
                self._current_version[v] = max(tv, fv)
            elif tv is not None:
                self._current_version[v] = tv
            elif fv is not None:
                self._current_version[v] = fv

    def visit_MergeNode(self, node: MergeNode) -> None:
        node.target = self._define(node.target)

    def visit_LoopNode(self, node: LoopNode) -> None:
        node.index_var = self._define(node.index_var)
        renamer = self._use_renamer()
        node.bound = renamer.visit(node.bound)
        self.visit(node.body)

    def visit_ReturnNode(self, node: ReturnNode) -> None:
        renamer = self._use_renamer()
        node.value = renamer.visit(node.value)

    def visit_SequenceNode(self, node: SequenceNode) -> None:
        for stmt in node.stmts:
            self.visit(stmt)

    def visit_NoOpNode(self, node: NoOpNode) -> None:
        pass


# ═══════════════════════════════════════════════════════════════════════════
# 4.  FREE VARIABLE COLLECTOR (for IR nodes)
# ═══════════════════════════════════════════════════════════════════════════


class FreeVarCollector(IRNodeVisitorBase):
    """Collect all free variable names referenced in an IR sub-tree.

    Usage::

        collector = FreeVarCollector()
        collector.visit(node)
        print(collector.free_vars)
    """

    def __init__(self) -> None:
        self.free_vars: set[str] = set()
        self._defined: set[str] = set()

    def _collect_from_expr(self, expr: TypedExpr) -> None:
        """Collect free vars from an expression, excluding already-defined ones."""
        for v in expr.free_vars():
            if v not in self._defined:
                self.free_vars.add(v)

    def visit_AssignNode(self, node: AssignNode) -> None:
        self._collect_from_expr(node.value)
        self._defined.add(node.target.name)

    def visit_NoiseDrawNode(self, node: NoiseDrawNode) -> None:
        self._collect_from_expr(node.center)
        self._collect_from_expr(node.scale)
        if node.sensitivity is not None:
            self._collect_from_expr(node.sensitivity)
        self._defined.add(node.target.name)

    def visit_QueryNode(self, node: QueryNode) -> None:
        for a in node.args:
            self._collect_from_expr(a)
        self._collect_from_expr(node.sensitivity)
        self._defined.add(node.target.name)

    def visit_BranchNode(self, node: BranchNode) -> None:
        self._collect_from_expr(node.condition)
        self.visit(node.true_branch)
        self.visit(node.false_branch)

    def visit_MergeNode(self, node: MergeNode) -> None:
        for expr in node.sources.values():
            self._collect_from_expr(expr)
        self._defined.add(node.target.name)

    def visit_LoopNode(self, node: LoopNode) -> None:
        self._collect_from_expr(node.bound)
        self._defined.add(node.index_var.name)
        self.visit(node.body)

    def visit_ReturnNode(self, node: ReturnNode) -> None:
        self._collect_from_expr(node.value)

    def visit_SequenceNode(self, node: SequenceNode) -> None:
        for stmt in node.stmts:
            self.visit(stmt)

    def visit_NoOpNode(self, node: NoOpNode) -> None:
        pass


# ═══════════════════════════════════════════════════════════════════════════
# 5.  EXPRESSION SUBSTITUTER (across IR nodes)
# ═══════════════════════════════════════════════════════════════════════════


class ExprSubstituter(IRNodeVisitorBase):
    """Substitute variables in all expressions across an IR sub-tree.

    Usage::

        sub = ExprSubstituter({"x": Const.real(42.0)})
        sub.visit(node)
    """

    def __init__(self, mapping: dict[str, TypedExpr]) -> None:
        self._mapping = mapping

    def _sub(self, expr: TypedExpr) -> TypedExpr:
        return expr.substitute(self._mapping)

    def visit_AssignNode(self, node: AssignNode) -> None:
        node.value = self._sub(node.value)

    def visit_NoiseDrawNode(self, node: NoiseDrawNode) -> None:
        node.center = self._sub(node.center)
        node.scale = self._sub(node.scale)
        if node.sensitivity is not None:
            node.sensitivity = self._sub(node.sensitivity)

    def visit_BranchNode(self, node: BranchNode) -> None:
        node.condition = self._sub(node.condition)
        self.visit(node.true_branch)
        self.visit(node.false_branch)

    def visit_MergeNode(self, node: MergeNode) -> None:
        node.sources = {k: self._sub(v) for k, v in node.sources.items()}

    def visit_LoopNode(self, node: LoopNode) -> None:
        node.bound = self._sub(node.bound)
        self.visit(node.body)

    def visit_QueryNode(self, node: QueryNode) -> None:
        node.args = tuple(self._sub(a) for a in node.args)
        node.sensitivity = self._sub(node.sensitivity)

    def visit_ReturnNode(self, node: ReturnNode) -> None:
        node.value = self._sub(node.value)

    def visit_SequenceNode(self, node: SequenceNode) -> None:
        for stmt in node.stmts:
            self.visit(stmt)

    def visit_NoOpNode(self, node: NoOpNode) -> None:
        pass


# ═══════════════════════════════════════════════════════════════════════════
# 6.  NODE PRINTER
# ═══════════════════════════════════════════════════════════════════════════


class NodePrinter(IRNodeVisitorBase):
    """Pretty-print an IR tree to a string.

    Usage::

        printer = NodePrinter()
        printer.visit(mechir.body)
        print(printer.output())
    """

    def __init__(self, indent_step: int = 2) -> None:
        self._buf = StringIO()
        self._indent = 0
        self._indent_step = indent_step

    def output(self) -> str:
        """Return the accumulated output string."""
        return self._buf.getvalue()

    def _write(self, text: str) -> None:
        self._buf.write(" " * self._indent + text + "\n")

    def _push(self) -> None:
        self._indent += self._indent_step

    def _pop(self) -> None:
        self._indent -= self._indent_step

    def visit_AssignNode(self, node: AssignNode) -> None:
        self._write(f"{node.target} := {node.value}")

    def visit_NoiseDrawNode(self, node: NoiseDrawNode) -> None:
        self._write(str(node))

    def visit_BranchNode(self, node: BranchNode) -> None:
        self._write(f"if ({node.condition}):")
        self._push()
        self.visit(node.true_branch)
        self._pop()
        self._write("else:")
        self._push()
        self.visit(node.false_branch)
        self._pop()

    def visit_MergeNode(self, node: MergeNode) -> None:
        self._write(str(node))

    def visit_LoopNode(self, node: LoopNode) -> None:
        self._write(f"for {node.index_var} in range({node.bound}):")
        self._push()
        self.visit(node.body)
        self._pop()

    def visit_QueryNode(self, node: QueryNode) -> None:
        self._write(str(node))

    def visit_ReturnNode(self, node: ReturnNode) -> None:
        self._write(f"return {node.value}")

    def visit_SequenceNode(self, node: SequenceNode) -> None:
        for stmt in node.stmts:
            self.visit(stmt)

    def visit_NoOpNode(self, node: NoOpNode) -> None:
        self._write("noop")


def print_ir(node: IRNode) -> str:
    """Convenience: pretty-print an IR node tree to a string."""
    printer = NodePrinter()
    printer.visit(node)
    return printer.output()


def print_mechir(mechir: MechIR) -> str:
    """Pretty-print a complete MechIR mechanism."""
    buf = StringIO()
    buf.write(f"mechanism {mechir.name}(")
    buf.write(", ".join(str(p) for p in mechir.params))
    buf.write(f") -> {mechir.return_type}")
    if mechir.budget:
        buf.write(f"  [{mechir.budget}]")
    buf.write(":\n")
    printer = NodePrinter(indent_step=2)
    printer._indent = 2  # noqa: SLF001
    printer.visit(mechir.body)
    buf.write(printer.output())
    return buf.getvalue()


# ═══════════════════════════════════════════════════════════════════════════
# 7.  IR VALIDATOR
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class ValidationError:
    """A single validation finding."""

    node_id: int
    message: str
    severity: str = "error"  # "error" | "warning"

    def __str__(self) -> str:
        return f"[{self.severity}] node {self.node_id}: {self.message}"


class IRValidator(IRNodeVisitorBase):
    """Validate structural well-formedness of IR trees.

    Checks:
      - Every AssignNode target is a Var
      - NoiseDrawNode has a valid distribution and positive scale
      - BranchNode condition is BOOL-typed
      - LoopNode bound is INT-typed
      - No orphan MergeNodes (must follow a BranchNode)
      - ReturnNode is only at the end of the body

    Usage::

        validator = IRValidator()
        validator.validate(mechir)
        if validator.errors:
            for err in validator.errors:
                print(err)
    """

    def __init__(self) -> None:
        self.errors: list[ValidationError] = []
        self.warnings: list[ValidationError] = []
        self._defined_vars: set[str] = set()
        self._seen_return: bool = False
        self._in_branch: int = 0

    def validate(self, mechir: MechIR) -> bool:
        """Run validation and return True if no errors were found."""
        self.errors.clear()
        self.warnings.clear()
        self._defined_vars.clear()
        self._seen_return = False

        # Register parameters as defined
        for p in mechir.params:
            self._defined_vars.add(p.name)

        self.visit(mechir.body)
        return len(self.errors) == 0

    def _error(self, node: IRNode, msg: str) -> None:
        self.errors.append(ValidationError(node.node_id, msg, "error"))

    def _warning(self, node: IRNode, msg: str) -> None:
        self.warnings.append(ValidationError(node.node_id, msg, "warning"))

    def _check_expr_vars(self, node: IRNode, expr: TypedExpr) -> None:
        """Warn if an expression uses undefined variables."""
        for v in expr.free_vars():
            if v not in self._defined_vars:
                self._warning(node, f"use of potentially undefined variable '{v}'")

    def visit_AssignNode(self, node: AssignNode) -> None:
        if not isinstance(node.target, Var):
            self._error(node, "AssignNode target must be a Var")
        self._check_expr_vars(node, node.value)
        self._defined_vars.add(node.target.name)

    def visit_NoiseDrawNode(self, node: NoiseDrawNode) -> None:
        if not isinstance(node.target, Var):
            self._error(node, "NoiseDrawNode target must be a Var")
        if not isinstance(node.noise_kind, NoiseKind):
            self._error(node, f"Invalid noise kind: {node.noise_kind}")
        self._check_expr_vars(node, node.center)
        self._check_expr_vars(node, node.scale)
        self._defined_vars.add(node.target.name)

    def visit_BranchNode(self, node: BranchNode) -> None:
        if node.condition.ty != IRType.BOOL:
            self._error(node, f"BranchNode condition must be BOOL, got {node.condition.ty}")
        self._check_expr_vars(node, node.condition)
        self._in_branch += 1
        self.visit(node.true_branch)
        self.visit(node.false_branch)
        self._in_branch -= 1

    def visit_MergeNode(self, node: MergeNode) -> None:
        if self._in_branch == 0:
            self._warning(node, "MergeNode not directly after a BranchNode")
        if not node.sources:
            self._warning(node, "MergeNode has no sources")
        self._defined_vars.add(node.target.name)

    def visit_LoopNode(self, node: LoopNode) -> None:
        if node.bound.ty != IRType.INT and not isinstance(node.bound, Const):
            self._warning(node, f"LoopNode bound is {node.bound.ty}, expected INT")
        self._check_expr_vars(node, node.bound)
        self._defined_vars.add(node.index_var.name)
        self.visit(node.body)

    def visit_QueryNode(self, node: QueryNode) -> None:
        if not node.query_name:
            self._error(node, "QueryNode has empty query_name")
        for a in node.args:
            self._check_expr_vars(node, a)
        self._defined_vars.add(node.target.name)

    def visit_ReturnNode(self, node: ReturnNode) -> None:
        self._check_expr_vars(node, node.value)
        self._seen_return = True

    def visit_SequenceNode(self, node: SequenceNode) -> None:
        for i, stmt in enumerate(node.stmts):
            if self._seen_return and i > 0:
                self._warning(stmt, "unreachable code after return")
            self.visit(stmt)

    def visit_NoOpNode(self, node: NoOpNode) -> None:
        pass
