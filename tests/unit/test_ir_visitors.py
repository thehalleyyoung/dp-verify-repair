"""Unit tests for dpcegar.ir.visitors — visitor, transformer, SSA, free vars, printer, validator."""

from __future__ import annotations

import pytest

from dpcegar.ir.types import (
    BinOp,
    BinOpKind,
    Const,
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
from dpcegar.ir.visitors import (
    ExprTransformer,
    ExprSubstituter,
    FreeVarCollector,
    IRNodeVisitor,
    IRValidator,
    NodeExprTransformer,
    NodePrinter,
    SSANumbering,
    ValidationError,
    print_ir,
    print_mechir,
)


# ═══════════════════════════════════════════════════════════════════════════
# IRNodeVisitor (pre/post order)
# ═══════════════════════════════════════════════════════════════════════════


class TestIRNodeVisitor:
    def test_pre_visit_called(self):
        class _Tracker(IRNodeVisitor):
            pre_visited = []
            def pre_visit(self, node):
                self.pre_visited.append(type(node).__name__)

        v = _Tracker()
        seq = SequenceNode(stmts=[AssignNode(), ReturnNode()])
        v.visit(seq)
        assert "SequenceNode" in v.pre_visited

    def test_post_visit_called(self):
        class _Tracker(IRNodeVisitor):
            post_visited = []
            def post_visit(self, node, result):
                self.post_visited.append(type(node).__name__)
                return result

        v = _Tracker()
        v.visit(AssignNode())
        assert "AssignNode" in v.post_visited

    def test_post_visit_can_transform(self):
        class _Transformer(IRNodeVisitor):
            def post_visit(self, node, result):
                return "transformed"

        v = _Transformer()
        result = v.visit(NoOpNode())
        assert result == "transformed"

    def test_visit_all_preorder(self):
        class _Counter(IRNodeVisitor):
            count = 0
            def pre_visit(self, node):
                self.count += 1

        v = _Counter()
        seq = SequenceNode(stmts=[AssignNode(), AssignNode(), ReturnNode()])
        v.visit_all(seq)
        assert v.count == 7  # seq + 2 assign + return + children walked recursively

    def test_visit_all_returns_list(self):
        class _NoneVisitor(IRNodeVisitor):
            pass

        v = _NoneVisitor()
        seq = SequenceNode(stmts=[NoOpNode(), NoOpNode()])
        results = v.visit_all(seq)
        assert isinstance(results, list)
        assert len(results) == 3  # seq + 2 noop

    def test_specific_visitor_override(self):
        class _NoiseCounter(IRNodeVisitor):
            noise_count = 0
            def visit_NoiseDrawNode(self, node):
                self.noise_count += 1
                return self.noise_count

        v = _NoiseCounter()
        seq = SequenceNode(stmts=[
            AssignNode(),
            NoiseDrawNode(),
            NoiseDrawNode(),
            ReturnNode(),
        ])
        v.visit_all(seq)
        assert v.noise_count == 4  # visit_all walks recursively

    def test_branch_traversal(self):
        class _Collector(IRNodeVisitor):
            types = []
            def pre_visit(self, node):
                self.types.append(type(node).__name__)

        v = _Collector()
        branch = BranchNode(
            condition=Const.bool_(True),
            true_branch=AssignNode(),
            false_branch=NoOpNode(),
        )
        v.visit(branch)
        assert "BranchNode" in v.types


# ═══════════════════════════════════════════════════════════════════════════
# ExprTransformer
# ═══════════════════════════════════════════════════════════════════════════


class TestExprTransformer:
    def test_identity_transform(self):
        t = ExprTransformer()
        x = Var(ty=IRType.REAL, name="x")
        result = t.visit(x)
        assert isinstance(result, Var) and result.name == "x"

    def test_const_unchanged(self):
        t = ExprTransformer()
        c = Const.real(3.14)
        result = t.visit(c)
        assert isinstance(result, Const)

    def test_custom_var_replacement(self):
        class _Replacer(ExprTransformer):
            def visit_Var(self, expr):
                if expr.name == "x":
                    return Const.real(42.0)
                return expr

        t = _Replacer()
        expr = BinOp(ty=IRType.REAL, op=BinOpKind.ADD,
                     left=Var(ty=IRType.REAL, name="x"),
                     right=Var(ty=IRType.REAL, name="y"))
        result = t.visit(expr)
        assert isinstance(result, BinOp)
        assert isinstance(result.left, Const) and result.left.value == 42.0
        assert isinstance(result.right, Var) and result.right.name == "y"

    def test_bottom_up_transform(self):
        """Inner expressions should be transformed before outer."""
        class _Counter(ExprTransformer):
            order = []
            def visit_Var(self, expr):
                self.order.append(expr.name)
                return expr

        t = _Counter()
        inner = BinOp(ty=IRType.REAL, op=BinOpKind.ADD,
                      left=Var(ty=IRType.REAL, name="a"),
                      right=Var(ty=IRType.REAL, name="b"))
        outer = BinOp(ty=IRType.REAL, op=BinOpKind.MUL,
                      left=inner,
                      right=Var(ty=IRType.REAL, name="c"))
        t.visit(outer)
        assert t.order == ["a", "b", "c"]


# ═══════════════════════════════════════════════════════════════════════════
# NodeExprTransformer
# ═══════════════════════════════════════════════════════════════════════════


class TestNodeExprTransformer:
    def test_transform_assign(self):
        class _DoubleConst(ExprTransformer):
            def visit_Const(self, expr):
                if isinstance(expr.value, (int, float)) and not isinstance(expr.value, bool):
                    return Const.real(float(expr.value) * 2)
                return expr

        node = AssignNode(
            target=Var(ty=IRType.REAL, name="x"),
            value=Const.real(5.0),
        )
        net = NodeExprTransformer(_DoubleConst())
        net.visit(node)
        assert isinstance(node.value, Const) and node.value.value == 10.0

    def test_transform_noise_draw(self):
        class _RenameVar(ExprTransformer):
            def visit_Var(self, expr):
                if expr.name == "q":
                    return Var(ty=expr.ty, name="q_renamed")
                return expr

        node = NoiseDrawNode(
            target=Var(ty=IRType.REAL, name="eta"),
            center=Var(ty=IRType.REAL, name="q"),
            scale=Const.real(1.0),
        )
        net = NodeExprTransformer(_RenameVar())
        net.visit(node)
        assert node.center.name == "q_renamed"

    def test_transform_sequence(self):
        class _ZeroOut(ExprTransformer):
            def visit_Var(self, expr):
                return Const.real(0.0)

        seq = SequenceNode(stmts=[
            AssignNode(target=Var(ty=IRType.REAL, name="x"),
                       value=Var(ty=IRType.REAL, name="y")),
            ReturnNode(value=Var(ty=IRType.REAL, name="x")),
        ])
        net = NodeExprTransformer(_ZeroOut())
        net.visit(seq)
        ret = seq.stmts[1]
        assert isinstance(ret, ReturnNode)
        assert isinstance(ret.value, Const) and ret.value.value == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# SSANumbering
# ═══════════════════════════════════════════════════════════════════════════


class TestSSANumbering:
    def test_single_assign(self):
        node = AssignNode(
            target=Var(ty=IRType.REAL, name="x"),
            value=Const.real(1.0),
        )
        ssa = SSANumbering()
        ssa.visit(node)
        assert node.target.version == 0

    def test_two_assigns_different_versions(self):
        a1 = AssignNode(target=Var(ty=IRType.REAL, name="x"), value=Const.real(1.0))
        a2 = AssignNode(target=Var(ty=IRType.REAL, name="x"), value=Var(ty=IRType.REAL, name="x"))
        seq = SequenceNode(stmts=[a1, a2])
        ssa = SSANumbering()
        ssa.visit(seq)
        assert a1.target.version == 0
        assert a2.target.version == 1

    def test_use_renamed(self):
        a1 = AssignNode(target=Var(ty=IRType.REAL, name="x"), value=Const.real(1.0))
        a2 = AssignNode(target=Var(ty=IRType.REAL, name="y"),
                        value=Var(ty=IRType.REAL, name="x"))
        seq = SequenceNode(stmts=[a1, a2])
        ssa = SSANumbering()
        ssa.visit(seq)
        # a2's RHS should reference x_0
        assert isinstance(a2.value, Var)
        assert a2.value.version == 0

    def test_branch_merge_versions(self):
        a1 = AssignNode(target=Var(ty=IRType.REAL, name="x"), value=Const.real(0.0))
        true_a = AssignNode(target=Var(ty=IRType.REAL, name="x"), value=Const.real(1.0))
        false_a = AssignNode(target=Var(ty=IRType.REAL, name="x"), value=Const.real(2.0))
        branch = BranchNode(
            condition=Const.bool_(True),
            true_branch=true_a,
            false_branch=false_a,
        )
        seq = SequenceNode(stmts=[a1, branch])
        ssa = SSANumbering()
        ssa.visit(seq)
        assert a1.target.version == 0
        assert true_a.target.version == 1
        assert false_a.target.version == 2

    def test_noise_draw_numbering(self):
        nd = NoiseDrawNode(
            target=Var(ty=IRType.REAL, name="eta"),
            center=Const.real(0.0),
            scale=Const.real(1.0),
        )
        ssa = SSANumbering()
        ssa.visit(nd)
        assert nd.target.version == 0

    def test_query_numbering(self):
        q = QueryNode(
            target=Var(ty=IRType.REAL, name="q"),
            query_name="count",
            args=(Const.real(0.0),),
            sensitivity=Const.real(1.0),
        )
        ssa = SSANumbering()
        ssa.visit(q)
        assert q.target.version == 0

    def test_loop_index_numbered(self):
        loop = LoopNode(
            index_var=Var(ty=IRType.INT, name="i"),
            bound=Const.int_(3),
            body=NoOpNode(),
        )
        ssa = SSANumbering()
        ssa.visit(loop)
        assert loop.index_var.version == 0

    def test_return_uses_latest(self):
        a = AssignNode(target=Var(ty=IRType.REAL, name="x"), value=Const.real(1.0))
        ret = ReturnNode(value=Var(ty=IRType.REAL, name="x"))
        seq = SequenceNode(stmts=[a, ret])
        ssa = SSANumbering()
        ssa.visit(seq)
        assert isinstance(ret.value, Var)
        assert ret.value.version == 0

    def test_noop_handled(self):
        ssa = SSANumbering()
        ssa.visit(NoOpNode())  # Should not raise

    def test_merge_numbering(self):
        m = MergeNode(target=Var(ty=IRType.REAL, name="phi"))
        ssa = SSANumbering()
        ssa.visit(m)
        assert m.target.version == 0


# ═══════════════════════════════════════════════════════════════════════════
# FreeVarCollector
# ═══════════════════════════════════════════════════════════════════════════


class TestFreeVarCollectorVisitor:
    def test_assign_defines(self):
        a = AssignNode(target=Var(ty=IRType.REAL, name="x"), value=Const.real(1.0))
        fvc = FreeVarCollector()
        fvc.visit(a)
        assert "x" not in fvc.free_vars

    def test_assign_uses_free(self):
        a = AssignNode(target=Var(ty=IRType.REAL, name="x"),
                       value=Var(ty=IRType.REAL, name="y"))
        fvc = FreeVarCollector()
        fvc.visit(a)
        assert "y" in fvc.free_vars
        assert "x" not in fvc.free_vars

    def test_sequence_defines_then_uses(self):
        a1 = AssignNode(target=Var(ty=IRType.REAL, name="x"), value=Const.real(1.0))
        a2 = AssignNode(target=Var(ty=IRType.REAL, name="y"),
                        value=Var(ty=IRType.REAL, name="x"))
        seq = SequenceNode(stmts=[a1, a2])
        fvc = FreeVarCollector()
        fvc.visit(seq)
        # x is defined before use, so only no free vars
        assert "x" not in fvc.free_vars
        assert "y" not in fvc.free_vars

    def test_use_before_define(self):
        a = AssignNode(target=Var(ty=IRType.REAL, name="y"),
                       value=Var(ty=IRType.REAL, name="z"))
        fvc = FreeVarCollector()
        fvc.visit(a)
        assert "z" in fvc.free_vars

    def test_noise_draw(self):
        nd = NoiseDrawNode(
            target=Var(ty=IRType.REAL, name="eta"),
            center=Var(ty=IRType.REAL, name="q"),
            scale=Var(ty=IRType.REAL, name="b"),
        )
        fvc = FreeVarCollector()
        fvc.visit(nd)
        assert "q" in fvc.free_vars
        assert "b" in fvc.free_vars
        assert "eta" not in fvc.free_vars

    def test_query_node(self):
        q = QueryNode(
            target=Var(ty=IRType.REAL, name="res"),
            query_name="count",
            args=(Var(ty=IRType.REAL, name="db"),),
            sensitivity=Const.real(1.0),
        )
        fvc = FreeVarCollector()
        fvc.visit(q)
        assert "db" in fvc.free_vars
        assert "res" not in fvc.free_vars

    def test_branch_condition_free(self):
        branch = BranchNode(
            condition=Var(ty=IRType.BOOL, name="flag"),
            true_branch=NoOpNode(),
            false_branch=NoOpNode(),
        )
        fvc = FreeVarCollector()
        fvc.visit(branch)
        assert "flag" in fvc.free_vars

    def test_loop_bound_free(self):
        loop = LoopNode(
            index_var=Var(ty=IRType.INT, name="i"),
            bound=Var(ty=IRType.INT, name="n"),
            body=NoOpNode(),
        )
        fvc = FreeVarCollector()
        fvc.visit(loop)
        assert "n" in fvc.free_vars
        assert "i" not in fvc.free_vars

    def test_return_free(self):
        ret = ReturnNode(value=Var(ty=IRType.REAL, name="result"))
        fvc = FreeVarCollector()
        fvc.visit(ret)
        assert "result" in fvc.free_vars

    def test_merge_free(self):
        m = MergeNode(
            target=Var(ty=IRType.REAL, name="merged"),
            sources={1: Var(ty=IRType.REAL, name="a"), 2: Var(ty=IRType.REAL, name="b")},
        )
        fvc = FreeVarCollector()
        fvc.visit(m)
        assert "a" in fvc.free_vars and "b" in fvc.free_vars
        assert "merged" not in fvc.free_vars


# ═══════════════════════════════════════════════════════════════════════════
# ExprSubstituter (across IR nodes)
# ═══════════════════════════════════════════════════════════════════════════


class TestExprSubstituterVisitor:
    def test_substitute_assign(self):
        a = AssignNode(target=Var(ty=IRType.REAL, name="x"),
                       value=Var(ty=IRType.REAL, name="y"))
        sub = ExprSubstituter({"y": Const.real(42.0)})
        sub.visit(a)
        assert isinstance(a.value, Const) and a.value.value == 42.0

    def test_substitute_noise_draw(self):
        nd = NoiseDrawNode(
            target=Var(ty=IRType.REAL, name="eta"),
            center=Var(ty=IRType.REAL, name="q"),
            scale=Var(ty=IRType.REAL, name="b"),
        )
        sub = ExprSubstituter({"q": Const.real(10.0)})
        sub.visit(nd)
        assert isinstance(nd.center, Const) and nd.center.value == 10.0

    def test_substitute_branch_condition(self):
        branch = BranchNode(
            condition=Var(ty=IRType.BOOL, name="flag"),
            true_branch=NoOpNode(),
            false_branch=NoOpNode(),
        )
        sub = ExprSubstituter({"flag": Const.bool_(True)})
        sub.visit(branch)
        assert isinstance(branch.condition, Const) and branch.condition.value is True

    def test_substitute_return(self):
        ret = ReturnNode(value=Var(ty=IRType.REAL, name="x"))
        sub = ExprSubstituter({"x": Const.real(99.0)})
        sub.visit(ret)
        assert isinstance(ret.value, Const) and ret.value.value == 99.0

    def test_substitute_loop_bound(self):
        loop = LoopNode(
            index_var=Var(ty=IRType.INT, name="i"),
            bound=Var(ty=IRType.INT, name="n"),
            body=NoOpNode(),
        )
        sub = ExprSubstituter({"n": Const.int_(10)})
        sub.visit(loop)
        assert isinstance(loop.bound, Const) and loop.bound.value == 10

    def test_substitute_query_args(self):
        q = QueryNode(
            target=Var(ty=IRType.REAL, name="res"),
            query_name="count",
            args=(Var(ty=IRType.REAL, name="db"),),
            sensitivity=Var(ty=IRType.REAL, name="delta"),
        )
        sub = ExprSubstituter({"delta": Const.real(1.0)})
        sub.visit(q)
        assert isinstance(q.sensitivity, Const)

    def test_substitute_merge_sources(self):
        m = MergeNode(
            target=Var(ty=IRType.REAL, name="merged"),
            sources={1: Var(ty=IRType.REAL, name="a")},
        )
        sub = ExprSubstituter({"a": Const.real(5.0)})
        sub.visit(m)
        assert isinstance(m.sources[1], Const)

    def test_substitute_noop(self):
        sub = ExprSubstituter({"x": Const.real(0.0)})
        sub.visit(NoOpNode())  # Should not raise

    def test_substitute_sequence(self):
        a = AssignNode(target=Var(ty=IRType.REAL, name="x"),
                       value=Var(ty=IRType.REAL, name="y"))
        ret = ReturnNode(value=Var(ty=IRType.REAL, name="y"))
        seq = SequenceNode(stmts=[a, ret])
        sub = ExprSubstituter({"y": Const.real(7.0)})
        sub.visit(seq)
        assert isinstance(a.value, Const) and a.value.value == 7.0
        assert isinstance(ret.value, Const) and ret.value.value == 7.0


# ═══════════════════════════════════════════════════════════════════════════
# NodePrinter
# ═══════════════════════════════════════════════════════════════════════════


class TestNodePrinter:
    def test_assign_output(self):
        a = AssignNode(target=Var(ty=IRType.REAL, name="x"), value=Const.real(1.0))
        printer = NodePrinter()
        printer.visit(a)
        out = printer.output()
        assert ":=" in out and "x" in out

    def test_noise_draw_output(self):
        nd = NoiseDrawNode(
            target=Var(ty=IRType.REAL, name="eta"),
            noise_kind=NoiseKind.LAPLACE,
            center=Const.real(0.0),
            scale=Const.real(1.0),
        )
        printer = NodePrinter()
        printer.visit(nd)
        out = printer.output()
        assert "~" in out

    def test_branch_output(self):
        b = BranchNode(
            condition=Const.bool_(True),
            true_branch=AssignNode(target=Var(ty=IRType.REAL, name="x"),
                                   value=Const.real(1.0)),
            false_branch=NoOpNode(),
        )
        printer = NodePrinter()
        printer.visit(b)
        out = printer.output()
        assert "if" in out and "else" in out

    def test_loop_output(self):
        loop = LoopNode(
            index_var=Var(ty=IRType.INT, name="i"),
            bound=Const.int_(5),
            body=NoOpNode(),
        )
        printer = NodePrinter()
        printer.visit(loop)
        out = printer.output()
        assert "for" in out and "range" in out

    def test_return_output(self):
        ret = ReturnNode(value=Const.real(0.0))
        printer = NodePrinter()
        printer.visit(ret)
        assert "return" in printer.output()

    def test_sequence_output(self):
        seq = SequenceNode(stmts=[
            AssignNode(target=Var(ty=IRType.REAL, name="x"), value=Const.real(1.0)),
            ReturnNode(value=Var(ty=IRType.REAL, name="x")),
        ])
        printer = NodePrinter()
        printer.visit(seq)
        out = printer.output()
        assert ":=" in out and "return" in out

    def test_noop_output(self):
        printer = NodePrinter()
        printer.visit(NoOpNode())
        assert "noop" in printer.output()

    def test_indent_step(self):
        b = BranchNode(
            condition=Const.bool_(True),
            true_branch=AssignNode(target=Var(ty=IRType.REAL, name="x"),
                                   value=Const.real(1.0)),
            false_branch=NoOpNode(),
        )
        printer = NodePrinter(indent_step=4)
        printer.visit(b)
        out = printer.output()
        lines = out.strip().split("\n")
        # Indented lines should have 4-space indent
        assert any(line.startswith("    ") for line in lines)


class TestPrintIRFunction:
    def test_print_ir(self):
        seq = SequenceNode(stmts=[
            AssignNode(target=Var(ty=IRType.REAL, name="x"), value=Const.real(1.0)),
            ReturnNode(value=Var(ty=IRType.REAL, name="x")),
        ])
        out = print_ir(seq)
        assert ":=" in out and "return" in out

    def test_print_mechir(self, simple_mechir):
        out = print_mechir(simple_mechir)
        assert "mechanism" in out and "laplace_mech" in out


# ═══════════════════════════════════════════════════════════════════════════
# IRValidator
# ═══════════════════════════════════════════════════════════════════════════


class TestIRValidator:
    def test_valid_mechanism(self, simple_mechir):
        v = IRValidator()
        is_valid = v.validate(simple_mechir)
        assert is_valid
        assert len(v.errors) == 0

    def test_empty_query_name(self):
        q = QueryNode(target=Var(ty=IRType.REAL, name="r"), query_name="")
        m = MechIR(body=SequenceNode(stmts=[q, ReturnNode()]))
        v = IRValidator()
        v.validate(m)
        assert len(v.errors) > 0
        assert any("empty query_name" in e.message for e in v.errors)

    def test_branch_non_bool_condition(self):
        b = BranchNode(
            condition=Const.real(1.0),  # Not BOOL
            true_branch=NoOpNode(),
            false_branch=NoOpNode(),
        )
        m = MechIR(body=b)
        v = IRValidator()
        v.validate(m)
        assert len(v.errors) > 0
        assert any("BOOL" in e.message for e in v.errors)

    def test_undefined_var_warning(self):
        a = AssignNode(target=Var(ty=IRType.REAL, name="x"),
                       value=Var(ty=IRType.REAL, name="undefined_var"))
        m = MechIR(body=a)
        v = IRValidator()
        v.validate(m)
        assert len(v.warnings) > 0

    def test_merge_no_sources_warning(self):
        merge = MergeNode(target=Var(ty=IRType.REAL, name="phi"), sources={})
        m = MechIR(body=merge)
        v = IRValidator()
        v.validate(m)
        assert len(v.warnings) > 0

    def test_params_are_defined(self):
        a = AssignNode(target=Var(ty=IRType.REAL, name="y"),
                       value=Var(ty=IRType.REAL, name="x"))
        m = MechIR(
            params=[ParamDecl(name="x", ty=IRType.REAL)],
            body=SequenceNode(stmts=[a, ReturnNode(value=Var(ty=IRType.REAL, name="y"))]),
        )
        v = IRValidator()
        v.validate(m)
        # x is a parameter, so no undefined warning for it
        assert not any("undefined variable 'x'" in w.message for w in v.warnings)

    def test_unreachable_code_warning(self):
        seq = SequenceNode(stmts=[
            ReturnNode(value=Const.real(0.0)),
            AssignNode(target=Var(ty=IRType.REAL, name="x"), value=Const.real(1.0)),
        ])
        m = MechIR(body=seq)
        v = IRValidator()
        v.validate(m)
        assert any("unreachable" in w.message for w in v.warnings)

    def test_validation_error_str(self):
        ve = ValidationError(node_id=42, message="test error", severity="error")
        s = str(ve)
        assert "42" in s and "test error" in s

    def test_valid_noise_draw(self):
        nd = NoiseDrawNode(
            target=Var(ty=IRType.REAL, name="eta"),
            noise_kind=NoiseKind.LAPLACE,
            center=Const.real(0.0),
            scale=Const.real(1.0),
        )
        m = MechIR(body=SequenceNode(stmts=[nd, ReturnNode(value=Var(ty=IRType.REAL, name="eta"))]))
        v = IRValidator()
        is_valid = v.validate(m)
        assert is_valid

    def test_validate_returns_true_no_errors(self):
        m = MechIR(body=ReturnNode(value=Const.real(0.0)))
        v = IRValidator()
        assert v.validate(m) is True

    def test_loop_validation(self):
        loop = LoopNode(
            index_var=Var(ty=IRType.INT, name="i"),
            bound=Const.int_(5),
            body=NoOpNode(),
        )
        m = MechIR(body=SequenceNode(stmts=[loop, ReturnNode()]))
        v = IRValidator()
        v.validate(m)
        assert len(v.errors) == 0

    def test_merge_outside_branch_warning(self):
        merge = MergeNode(
            target=Var(ty=IRType.REAL, name="phi"),
            sources={1: Const.real(1.0)},
        )
        m = MechIR(body=SequenceNode(stmts=[merge, ReturnNode()]))
        v = IRValidator()
        v.validate(m)
        assert any("not directly after" in w.message for w in v.warnings)
