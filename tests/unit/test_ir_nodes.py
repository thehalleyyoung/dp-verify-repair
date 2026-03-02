"""Unit tests for dpcegar.ir.nodes — IR node types, MechIR, CFGBuilder, CFG."""

from __future__ import annotations

import pytest

from dpcegar.ir.types import (
    BinOp,
    BinOpKind,
    Const,
    IRType,
    NoiseKind,
    PureBudget,
    Var,
)
from dpcegar.ir.nodes import (
    AssignNode,
    BranchNode,
    CFG,
    CFGBuilder,
    CFGEdge,
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
from dpcegar.utils.errors import InternalError, SourceLoc


# ═══════════════════════════════════════════════════════════════════════════
# AssignNode
# ═══════════════════════════════════════════════════════════════════════════


class TestAssignNode:
    def test_construction(self, assign_node, var_x, const_pi):
        assert assign_node.target is var_x
        assert assign_node.value is const_pi

    def test_children_empty(self, assign_node):
        assert assign_node.children() == []

    def test_unique_node_id(self):
        n1 = AssignNode()
        n2 = AssignNode()
        assert n1.node_id != n2.node_id

    def test_str(self, assign_node):
        s = str(assign_node)
        assert ":=" in s

    def test_repr(self, assign_node):
        assert "AssignNode" in repr(assign_node)

    def test_annotations(self):
        node = AssignNode()
        node.annotate("key", "value")
        assert node.get_annotation("key") == "value"

    def test_annotation_default(self):
        node = AssignNode()
        assert node.get_annotation("missing", "default") == "default"

    def test_source_loc(self, source_loc):
        node = AssignNode(source_loc=source_loc)
        assert node.source_loc is not None
        assert node.source_loc.line == 10


# ═══════════════════════════════════════════════════════════════════════════
# NoiseDrawNode
# ═══════════════════════════════════════════════════════════════════════════


class TestNoiseDrawNode:
    def test_construction(self, noise_draw_node):
        assert noise_draw_node.noise_kind == NoiseKind.LAPLACE
        assert noise_draw_node.target.name == "eta"

    def test_default_scale(self):
        n = NoiseDrawNode()
        assert n.scale == Const.one()

    def test_children_empty(self, noise_draw_node):
        assert noise_draw_node.children() == []

    def test_str(self, noise_draw_node):
        s = str(noise_draw_node)
        assert "~" in s

    def test_repr(self, noise_draw_node):
        assert "NoiseDrawNode" in repr(noise_draw_node)

    def test_sensitivity(self):
        n = NoiseDrawNode(sensitivity=Const.real(2.0))
        assert n.sensitivity is not None
        s = str(n)
        assert "Δ" in s

    def test_gaussian_noise(self):
        n = NoiseDrawNode(noise_kind=NoiseKind.GAUSSIAN)
        assert n.noise_kind == NoiseKind.GAUSSIAN

    def test_exponential_noise(self):
        n = NoiseDrawNode(noise_kind=NoiseKind.EXPONENTIAL)
        assert n.noise_kind == NoiseKind.EXPONENTIAL


# ═══════════════════════════════════════════════════════════════════════════
# BranchNode
# ═══════════════════════════════════════════════════════════════════════════


class TestBranchNode:
    def test_construction(self, branch_node, comparison_expr):
        assert branch_node.condition is comparison_expr

    def test_children(self, branch_node):
        children = branch_node.children()
        assert len(children) == 2
        assert isinstance(children[0], AssignNode)
        assert isinstance(children[1], AssignNode)

    def test_str(self, branch_node):
        assert "if" in str(branch_node)

    def test_repr(self, branch_node):
        assert "BranchNode" in repr(branch_node)

    def test_default_branches(self):
        b = BranchNode()
        assert isinstance(b.true_branch, NoOpNode)
        assert isinstance(b.false_branch, NoOpNode)

    def test_walk_includes_branches(self, branch_node):
        nodes = list(branch_node.walk())
        assert len(nodes) == 3  # branch + true assign + false assign


# ═══════════════════════════════════════════════════════════════════════════
# MergeNode
# ═══════════════════════════════════════════════════════════════════════════


class TestMergeNode:
    def test_construction(self):
        m = MergeNode(
            target=Var(ty=IRType.REAL, name="phi_x"),
            sources={1: Const.real(1.0), 2: Const.real(2.0)},
        )
        assert m.target.name == "phi_x"
        assert len(m.sources) == 2

    def test_add_source(self):
        m = MergeNode()
        m.add_source(10, Const.real(3.14))
        assert 10 in m.sources

    def test_children_empty(self):
        m = MergeNode()
        assert m.children() == []

    def test_str(self):
        m = MergeNode(
            target=Var(ty=IRType.REAL, name="r"),
            sources={1: Const.real(1.0)},
        )
        s = str(m)
        assert "φ" in s


# ═══════════════════════════════════════════════════════════════════════════
# LoopNode
# ═══════════════════════════════════════════════════════════════════════════


class TestLoopNode:
    def test_construction(self, loop_node):
        assert loop_node.index_var.name == "i"
        assert isinstance(loop_node.bound, Const)

    def test_children(self, loop_node):
        children = loop_node.children()
        assert len(children) == 1
        assert isinstance(children[0], AssignNode)

    def test_str(self, loop_node):
        s = str(loop_node)
        assert "for" in s and "range" in s

    def test_unroll_count(self):
        ln = LoopNode(unroll_count=3)
        assert ln.unroll_count == 3
        assert "unroll" in str(ln)

    def test_walk(self, loop_node):
        nodes = list(loop_node.walk())
        assert len(nodes) == 2  # loop + body assign


# ═══════════════════════════════════════════════════════════════════════════
# QueryNode
# ═══════════════════════════════════════════════════════════════════════════


class TestQueryNode:
    def test_construction(self, query_node):
        assert query_node.query_name == "count"
        assert query_node.target.name == "q_result"

    def test_sensitivity(self, query_node):
        assert isinstance(query_node.sensitivity, Const)

    def test_children_empty(self, query_node):
        assert query_node.children() == []

    def test_str(self, query_node):
        s = str(query_node)
        assert "count" in s and "Δ" in s


# ═══════════════════════════════════════════════════════════════════════════
# ReturnNode
# ═══════════════════════════════════════════════════════════════════════════


class TestReturnNode:
    def test_construction(self, return_node, var_x):
        assert return_node.value is var_x

    def test_children_empty(self, return_node):
        assert return_node.children() == []

    def test_str(self, return_node):
        assert "return" in str(return_node)


# ═══════════════════════════════════════════════════════════════════════════
# SequenceNode
# ═══════════════════════════════════════════════════════════════════════════


class TestSequenceNode:
    def test_construction(self, sequence_node):
        assert len(sequence_node) == 2

    def test_children(self, sequence_node):
        children = sequence_node.children()
        assert len(children) == 2

    def test_append(self):
        s = SequenceNode()
        s.append(NoOpNode())
        assert len(s) == 1

    def test_iter(self, sequence_node):
        nodes = list(sequence_node)
        assert len(nodes) == 2

    def test_str(self, sequence_node):
        assert "seq" in str(sequence_node) and "2" in str(sequence_node)

    def test_empty(self):
        s = SequenceNode()
        assert len(s) == 0
        assert list(s) == []

    def test_walk_preorder(self, sequence_node):
        nodes = list(sequence_node.walk())
        assert nodes[0] is sequence_node
        assert len(nodes) == 3  # seq + assign + return


# ═══════════════════════════════════════════════════════════════════════════
# NoOpNode
# ═══════════════════════════════════════════════════════════════════════════


class TestNoOpNode:
    def test_construction(self):
        n = NoOpNode()
        assert n.children() == []

    def test_str(self):
        n = NoOpNode()
        assert str(n) == "noop"


# ═══════════════════════════════════════════════════════════════════════════
# IRNode Walk Methods
# ═══════════════════════════════════════════════════════════════════════════


class TestIRNodeWalk:
    def test_walk_pre_order_simple(self):
        a = AssignNode(target=Var(ty=IRType.REAL, name="x"), value=Const.real(1.0))
        nodes = list(a.walk())
        assert len(nodes) == 1
        assert nodes[0] is a

    def test_walk_pre_order_sequence(self):
        a = AssignNode()
        b = ReturnNode()
        seq = SequenceNode(stmts=[a, b])
        nodes = list(seq.walk())
        assert nodes[0] is seq
        assert a in nodes and b in nodes

    def test_walk_post_order(self):
        a = AssignNode()
        b = ReturnNode()
        seq = SequenceNode(stmts=[a, b])
        nodes = list(seq.walk_post_order())
        assert nodes[-1] is seq
        assert nodes[0] is a or nodes[0] is b

    def test_walk_branch(self, branch_node):
        nodes = list(branch_node.walk())
        assert len(nodes) == 3

    def test_walk_nested(self):
        inner_seq = SequenceNode(stmts=[AssignNode(), AssignNode()])
        outer_seq = SequenceNode(stmts=[inner_seq, ReturnNode()])
        nodes = list(outer_seq.walk())
        assert len(nodes) == 5  # outer + inner + 2 assign + return


# ═══════════════════════════════════════════════════════════════════════════
# IRNodeVisitorBase
# ═══════════════════════════════════════════════════════════════════════════


class TestIRNodeVisitorBase:
    def test_double_dispatch(self):
        class _Counter(IRNodeVisitorBase):
            count = 0
            def visit_AssignNode(self, node):
                self.count += 1
                return self.count

        v = _Counter()
        a = AssignNode()
        result = v.visit(a)
        assert result == 1

    def test_generic_visit(self):
        class _Lister(IRNodeVisitorBase):
            visited = []
            def visit_AssignNode(self, node):
                self.visited.append("assign")
            def visit_ReturnNode(self, node):
                self.visited.append("return")

        v = _Lister()
        seq = SequenceNode(stmts=[AssignNode(), ReturnNode()])
        v.visit(seq)
        assert "assign" in v.visited and "return" in v.visited

    def test_accept_method(self):
        class _Inspector(IRNodeVisitorBase):
            def visit_NoOpNode(self, node):
                return "noop"

        v = _Inspector()
        n = NoOpNode()
        assert n.accept(v) == "noop"


# ═══════════════════════════════════════════════════════════════════════════
# ParamDecl
# ═══════════════════════════════════════════════════════════════════════════


class TestParamDecl:
    def test_construction(self):
        p = ParamDecl(name="db", ty=IRType.ARRAY, is_database=True)
        assert p.name == "db"
        assert p.ty == IRType.ARRAY
        assert p.is_database

    def test_str(self):
        p = ParamDecl(name="eps", ty=IRType.REAL)
        s = str(p)
        assert "eps" in s and "real" in s

    def test_str_database(self):
        p = ParamDecl(name="db", ty=IRType.ARRAY, is_database=True)
        assert "[DB]" in str(p)


# ═══════════════════════════════════════════════════════════════════════════
# MechIR
# ═══════════════════════════════════════════════════════════════════════════


class TestMechIR:
    def test_construction(self, simple_mechir):
        assert simple_mechir.name == "laplace_mech"
        assert simple_mechir.return_type == IRType.REAL
        assert len(simple_mechir.params) == 2

    def test_all_nodes(self, simple_mechir):
        nodes = list(simple_mechir.all_nodes())
        assert len(nodes) > 0

    def test_node_count(self, simple_mechir):
        assert simple_mechir.node_count() == 4  # seq + query + noise + return

    def test_noise_draws(self, simple_mechir):
        draws = simple_mechir.noise_draws()
        assert len(draws) == 1
        assert draws[0].noise_kind == NoiseKind.LAPLACE

    def test_queries(self, simple_mechir):
        queries = simple_mechir.queries()
        assert len(queries) == 1
        assert queries[0].query_name == "count"

    def test_find_node(self, simple_mechir):
        draws = simple_mechir.noise_draws()
        found = simple_mechir.find_node(draws[0].node_id)
        assert found is draws[0]

    def test_find_node_not_found(self, simple_mechir):
        assert simple_mechir.find_node(-999) is None

    def test_str(self, simple_mechir):
        s = str(simple_mechir)
        assert "mechanism" in s and "laplace_mech" in s

    def test_budget(self, simple_mechir):
        assert isinstance(simple_mechir.budget, PureBudget)

    def test_metadata(self):
        m = MechIR(metadata={"version": 1})
        assert m.metadata["version"] == 1

    def test_no_noise_draws(self):
        m = MechIR(body=ReturnNode(value=Const.real(0.0)))
        assert m.noise_draws() == []

    def test_no_queries(self):
        m = MechIR(body=ReturnNode(value=Const.real(0.0)))
        assert m.queries() == []


# ═══════════════════════════════════════════════════════════════════════════
# CFGEdge
# ═══════════════════════════════════════════════════════════════════════════


class TestCFGEdge:
    def test_construction(self):
        e = CFGEdge(src=1, dst=2)
        assert e.src == 1 and e.dst == 2

    def test_str(self):
        e = CFGEdge(src=1, dst=2, label="back")
        s = str(e)
        assert "→" in s and "back" in s

    def test_condition(self):
        cond = Const.bool_(True)
        e = CFGEdge(src=1, dst=2, condition=cond)
        assert e.condition is cond


# ═══════════════════════════════════════════════════════════════════════════
# CFG
# ═══════════════════════════════════════════════════════════════════════════


class TestCFG:
    def test_empty_cfg(self):
        cfg = CFG()
        assert cfg.node_count() == 0
        assert cfg.edge_count() == 0

    def test_add_node(self):
        cfg = CFG()
        n = AssignNode()
        cfg.add_node(n)
        assert cfg.node_count() == 1
        assert n.node_id in cfg.nodes

    def test_add_edge(self):
        cfg = CFG()
        n1 = AssignNode()
        n2 = ReturnNode()
        cfg.add_node(n1)
        cfg.add_node(n2)
        edge = cfg.add_edge(n1.node_id, n2.node_id)
        assert cfg.edge_count() == 1
        assert edge.src == n1.node_id

    def test_successors(self):
        cfg = CFG()
        n1 = NoOpNode()
        n2 = NoOpNode()
        cfg.add_node(n1)
        cfg.add_node(n2)
        cfg.add_edge(n1.node_id, n2.node_id)
        succs = cfg.successors(n1.node_id)
        assert len(succs) == 1 and succs[0].dst == n2.node_id

    def test_predecessors(self):
        cfg = CFG()
        n1 = NoOpNode()
        n2 = NoOpNode()
        cfg.add_node(n1)
        cfg.add_node(n2)
        cfg.add_edge(n1.node_id, n2.node_id)
        preds = cfg.predecessors(n2.node_id)
        assert len(preds) == 1 and preds[0].src == n1.node_id

    def test_successor_ids(self):
        cfg = CFG()
        n1 = NoOpNode()
        n2 = NoOpNode()
        n3 = NoOpNode()
        cfg.add_node(n1)
        cfg.add_node(n2)
        cfg.add_node(n3)
        cfg.add_edge(n1.node_id, n2.node_id)
        cfg.add_edge(n1.node_id, n3.node_id)
        ids = cfg.successor_ids(n1.node_id)
        assert set(ids) == {n2.node_id, n3.node_id}

    def test_predecessor_ids(self):
        cfg = CFG()
        n1 = NoOpNode()
        n2 = NoOpNode()
        n3 = NoOpNode()
        cfg.add_node(n1)
        cfg.add_node(n2)
        cfg.add_node(n3)
        cfg.add_edge(n1.node_id, n3.node_id)
        cfg.add_edge(n2.node_id, n3.node_id)
        ids = cfg.predecessor_ids(n3.node_id)
        assert set(ids) == {n1.node_id, n2.node_id}

    def test_reachable_from(self):
        cfg = CFG()
        n1 = NoOpNode()
        n2 = NoOpNode()
        n3 = NoOpNode()
        n4 = NoOpNode()  # unreachable
        cfg.add_node(n1)
        cfg.add_node(n2)
        cfg.add_node(n3)
        cfg.add_node(n4)
        cfg.add_edge(n1.node_id, n2.node_id)
        cfg.add_edge(n2.node_id, n3.node_id)
        reachable = cfg.reachable_from(n1.node_id)
        assert n1.node_id in reachable
        assert n2.node_id in reachable
        assert n3.node_id in reachable
        assert n4.node_id not in reachable

    def test_topological_order(self):
        cfg = CFG()
        n1 = NoOpNode()
        n2 = NoOpNode()
        n3 = NoOpNode()
        cfg.add_node(n1)
        cfg.add_node(n2)
        cfg.add_node(n3)
        cfg.add_edge(n1.node_id, n2.node_id)
        cfg.add_edge(n2.node_id, n3.node_id)
        cfg.entry = n1.node_id
        cfg.exit_node = n3.node_id
        order = cfg.topological_order()
        assert order.index(n1.node_id) < order.index(n2.node_id)
        assert order.index(n2.node_id) < order.index(n3.node_id)

    def test_str(self):
        cfg = CFG()
        n = NoOpNode()
        cfg.add_node(n)
        cfg.entry = n.node_id
        cfg.exit_node = n.node_id
        s = str(cfg)
        assert "CFG" in s

    def test_dominators_linear(self):
        cfg = CFG()
        n1 = NoOpNode()
        n2 = NoOpNode()
        n3 = NoOpNode()
        cfg.add_node(n1)
        cfg.add_node(n2)
        cfg.add_node(n3)
        cfg.add_edge(n1.node_id, n2.node_id)
        cfg.add_edge(n2.node_id, n3.node_id)
        cfg.entry = n1.node_id
        cfg.exit_node = n3.node_id
        dom = cfg.dominators()
        # n1 dominates everything
        assert n1.node_id in dom[n2.node_id]
        assert n1.node_id in dom[n3.node_id]
        # n2 dominates n3
        assert n2.node_id in dom[n3.node_id]
        # Every node dominates itself
        assert n1.node_id in dom[n1.node_id]

    def test_dominators_diamond(self):
        cfg = CFG()
        entry = NoOpNode()
        left = NoOpNode()
        right = NoOpNode()
        merge = NoOpNode()
        for n in [entry, left, right, merge]:
            cfg.add_node(n)
        cfg.add_edge(entry.node_id, left.node_id)
        cfg.add_edge(entry.node_id, right.node_id)
        cfg.add_edge(left.node_id, merge.node_id)
        cfg.add_edge(right.node_id, merge.node_id)
        cfg.entry = entry.node_id
        cfg.exit_node = merge.node_id
        dom = cfg.dominators()
        # entry dominates all
        for n in [entry, left, right, merge]:
            assert entry.node_id in dom[n.node_id]
        # left does not dominate merge
        assert left.node_id not in dom[merge.node_id]
        # right does not dominate merge
        assert right.node_id not in dom[merge.node_id]

    def test_post_dominators(self):
        cfg = CFG()
        n1 = NoOpNode()
        n2 = NoOpNode()
        n3 = NoOpNode()
        cfg.add_node(n1)
        cfg.add_node(n2)
        cfg.add_node(n3)
        cfg.add_edge(n1.node_id, n2.node_id)
        cfg.add_edge(n2.node_id, n3.node_id)
        cfg.entry = n1.node_id
        cfg.exit_node = n3.node_id
        pdom = cfg.post_dominators()
        # n3 post-dominates everything
        assert n3.node_id in pdom[n1.node_id]
        assert n3.node_id in pdom[n2.node_id]


# ═══════════════════════════════════════════════════════════════════════════
# CFGBuilder
# ═══════════════════════════════════════════════════════════════════════════


class TestCFGBuilder:
    def test_build_simple(self, simple_mechir):
        builder = CFGBuilder()
        cfg = builder.build(simple_mechir)
        assert cfg.entry >= 0
        assert cfg.exit_node >= 0
        assert cfg.node_count() > 0
        assert cfg.edge_count() > 0

    def test_build_has_entry_exit(self, simple_mechir):
        builder = CFGBuilder()
        cfg = builder.build(simple_mechir)
        entry_node = cfg.nodes[cfg.entry]
        assert entry_node.get_annotation("role") == "entry"
        exit_node = cfg.nodes[cfg.exit_node]
        assert exit_node.get_annotation("role") == "exit"

    def test_build_branching(self, branching_mechir):
        builder = CFGBuilder()
        cfg = builder.build(branching_mechir)
        # Should have: entry, seq(branch, return), exit + internal nodes
        assert cfg.node_count() > 4
        # The branch node should have multiple successors
        for nid, node in cfg.nodes.items():
            if isinstance(node, BranchNode):
                succs = cfg.successor_ids(nid)
                assert len(succs) >= 2
                break

    def test_build_loop(self):
        loop = LoopNode(
            index_var=Var(ty=IRType.INT, name="i"),
            bound=Const.int_(3),
            body=AssignNode(
                target=Var(ty=IRType.REAL, name="s"),
                value=Const.real(1.0),
            ),
        )
        ret = ReturnNode(value=Var(ty=IRType.REAL, name="s"))
        m = MechIR(name="loop_test", body=SequenceNode(stmts=[loop, ret]))
        builder = CFGBuilder()
        cfg = builder.build(m)
        # Loop should create a back edge
        has_back_edge = any(e.label == "back" for e in cfg.edges)
        assert has_back_edge

    def test_build_noop(self):
        m = MechIR(body=NoOpNode())
        builder = CFGBuilder()
        cfg = builder.build(m)
        assert cfg.node_count() >= 3  # entry + noop + exit

    def test_build_reachability(self, simple_mechir):
        builder = CFGBuilder()
        cfg = builder.build(simple_mechir)
        reachable = cfg.reachable_from(cfg.entry)
        assert cfg.exit_node in reachable

    def test_build_multiple_calls(self):
        """Builder can be reused."""
        builder = CFGBuilder()
        m1 = MechIR(body=ReturnNode(value=Const.real(0.0)))
        m2 = MechIR(body=ReturnNode(value=Const.real(1.0)))
        cfg1 = builder.build(m1)
        cfg2 = builder.build(m2)
        assert cfg1.entry != cfg2.entry  # Different node IDs


# ═══════════════════════════════════════════════════════════════════════════
# Node Annotations and Metadata
# ═══════════════════════════════════════════════════════════════════════════


class TestNodeAnnotations:
    def test_annotate_and_get(self):
        n = AssignNode()
        n.annotate("sensitivity", 1.0)
        assert n.get_annotation("sensitivity") == 1.0

    def test_multiple_annotations(self):
        n = AssignNode()
        n.annotate("a", 1)
        n.annotate("b", "two")
        n.annotate("c", [1, 2, 3])
        assert n.get_annotation("a") == 1
        assert n.get_annotation("b") == "two"
        assert n.get_annotation("c") == [1, 2, 3]

    def test_overwrite_annotation(self):
        n = AssignNode()
        n.annotate("key", "old")
        n.annotate("key", "new")
        assert n.get_annotation("key") == "new"

    def test_default_for_missing(self):
        n = AssignNode()
        assert n.get_annotation("x") is None
        assert n.get_annotation("x", 42) == 42


# ═══════════════════════════════════════════════════════════════════════════
# Source Location Tracking
# ═══════════════════════════════════════════════════════════════════════════


class TestSourceLocationTracking:
    def test_assign_with_loc(self, source_loc):
        n = AssignNode(source_loc=source_loc)
        assert n.source_loc.file == "test.py"
        assert n.source_loc.line == 10
        assert n.source_loc.col == 5

    def test_no_source_loc(self):
        n = AssignNode()
        assert n.source_loc is None

    def test_source_loc_str(self, source_loc):
        s = str(source_loc)
        assert "test.py" in s and "10" in s

    def test_source_loc_with_end(self):
        loc = SourceLoc(file="a.py", line=1, col=1, end_line=3, end_col=10)
        s = str(loc)
        assert "3" in s and "10" in s

    def test_all_node_types_accept_source_loc(self, source_loc):
        nodes = [
            AssignNode(source_loc=source_loc),
            NoiseDrawNode(source_loc=source_loc),
            BranchNode(source_loc=source_loc),
            MergeNode(source_loc=source_loc),
            LoopNode(source_loc=source_loc),
            QueryNode(source_loc=source_loc),
            ReturnNode(source_loc=source_loc),
            SequenceNode(source_loc=source_loc),
            NoOpNode(source_loc=source_loc),
        ]
        for n in nodes:
            assert n.source_loc is source_loc
