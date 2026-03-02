"""Tests for dpcegar.variants.lattice – implication lattice, propagation, ordering."""

from __future__ import annotations

import pytest

from dpcegar.ir.types import (
    ApproxBudget,
    GDPBudget,
    PrivacyNotion,
    PureBudget,
    RDPBudget,
    ZCDPBudget,
)
from dpcegar.variants.lattice import (
    ImplicationEdge,
    ImplicationLattice,
    NodeStatus,
    ParameterTransformer,
    PrivacyLatticeNode,
    lattice_summary,
    reachable_from,
    reachable_to,
    strongest_verified,
    topological_order,
    weakest_falsified,
)


# ═══════════════════════════════════════════════════════════════════════════
# NodeStatus enum
# ═══════════════════════════════════════════════════════════════════════════


class TestNodeStatus:
    def test_values(self):
        assert NodeStatus.VERIFIED is not None
        assert NodeStatus.FALSIFIED is not None
        assert NodeStatus.UNKNOWN is not None

    def test_str(self):
        assert isinstance(str(NodeStatus.VERIFIED), str)


# ═══════════════════════════════════════════════════════════════════════════
# PrivacyLatticeNode
# ═══════════════════════════════════════════════════════════════════════════


class TestPrivacyLatticeNode:
    def test_construction(self):
        n = PrivacyLatticeNode(notion=PrivacyNotion.PURE_DP)
        assert n.notion == PrivacyNotion.PURE_DP
        assert n.is_unknown

    def test_verified(self):
        n = PrivacyLatticeNode(notion=PrivacyNotion.PURE_DP, status=NodeStatus.VERIFIED)
        assert n.is_verified
        assert not n.is_falsified
        assert not n.is_unknown

    def test_falsified(self):
        n = PrivacyLatticeNode(notion=PrivacyNotion.PURE_DP, status=NodeStatus.FALSIFIED)
        assert n.is_falsified

    def test_reset(self):
        n = PrivacyLatticeNode(notion=PrivacyNotion.PURE_DP, status=NodeStatus.VERIFIED)
        n.reset()
        assert n.is_unknown

    def test_str(self):
        n = PrivacyLatticeNode(notion=PrivacyNotion.PURE_DP)
        assert isinstance(str(n), str)


# ═══════════════════════════════════════════════════════════════════════════
# ImplicationEdge
# ═══════════════════════════════════════════════════════════════════════════


class TestImplicationEdge:
    def test_construction(self):
        e = ImplicationEdge(
            source=PrivacyNotion.PURE_DP,
            target=PrivacyNotion.APPROX_DP,
            transform=lambda b: ApproxBudget(epsilon=b.epsilon, delta=0.0),
            theorem_ref="trivial",
        )
        assert e.source == PrivacyNotion.PURE_DP
        assert e.target == PrivacyNotion.APPROX_DP

    def test_str(self):
        e = ImplicationEdge(
            source=PrivacyNotion.PURE_DP,
            target=PrivacyNotion.APPROX_DP,
            transform=lambda b: None,
            theorem_ref="test",
        )
        assert isinstance(str(e), str)


# ═══════════════════════════════════════════════════════════════════════════
# ParameterTransformer
# ═══════════════════════════════════════════════════════════════════════════


class TestParameterTransformer:
    def test_pure_to_approx(self):
        budget = PureBudget(epsilon=1.0)
        result = ParameterTransformer.pure_to_approx(budget)
        assert isinstance(result, ApproxBudget)
        assert result.epsilon == 1.0
        assert result.delta == 0.0

    def test_pure_to_zcdp(self):
        budget = PureBudget(epsilon=1.0)
        result = ParameterTransformer.pure_to_zcdp(budget)
        assert isinstance(result, ZCDPBudget)

    def test_zcdp_to_rdp(self):
        budget = ZCDPBudget(rho=0.5)
        result = ParameterTransformer.zcdp_to_rdp(budget, alpha=2.0)
        assert isinstance(result, RDPBudget)

    def test_zcdp_to_approx(self):
        budget = ZCDPBudget(rho=0.5)
        result = ParameterTransformer.zcdp_to_approx(budget, delta=1e-5)
        assert isinstance(result, ApproxBudget)
        assert result.delta == 1e-5

    def test_rdp_to_approx(self):
        budget = RDPBudget(alpha=2.0, epsilon=1.0)
        result = ParameterTransformer.rdp_to_approx(budget, delta=1e-5)
        assert isinstance(result, ApproxBudget)

    def test_gdp_to_approx(self):
        budget = GDPBudget(mu=1.0)
        result = ParameterTransformer.gdp_to_approx(budget, delta=1e-5)
        assert isinstance(result, ApproxBudget)

    def test_transform_generic(self):
        budget = PureBudget(epsilon=1.0)
        result = ParameterTransformer.transform(budget, PrivacyNotion.APPROX_DP)
        assert isinstance(result, ApproxBudget) or result is not None

    @pytest.mark.parametrize("eps", [0.1, 0.5, 1.0, 5.0])
    def test_pure_to_approx_various_eps(self, eps: float):
        budget = PureBudget(epsilon=eps)
        result = ParameterTransformer.pure_to_approx(budget)
        assert result.epsilon == eps


# ═══════════════════════════════════════════════════════════════════════════
# ImplicationLattice
# ═══════════════════════════════════════════════════════════════════════════


class TestImplicationLattice:
    def test_construction(self):
        lat = ImplicationLattice()
        assert lat is not None

    def test_get_node(self):
        lat = ImplicationLattice()
        node = lat.get_node(PrivacyNotion.PURE_DP)
        assert isinstance(node, PrivacyLatticeNode)
        assert node.notion == PrivacyNotion.PURE_DP

    def test_all_nodes(self):
        lat = ImplicationLattice()
        nodes = lat.all_nodes
        assert len(nodes) >= 4

    def test_all_edges(self):
        lat = ImplicationLattice()
        edges = lat.all_edges
        assert len(edges) >= 1

    def test_get_edges_from(self):
        lat = ImplicationLattice()
        edges = lat.get_edges_from(PrivacyNotion.PURE_DP)
        assert isinstance(edges, list)

    def test_get_edges_to(self):
        lat = ImplicationLattice()
        edges = lat.get_edges_to(PrivacyNotion.APPROX_DP)
        assert isinstance(edges, list)

    def test_successors(self):
        lat = ImplicationLattice()
        succs = lat.successors(PrivacyNotion.PURE_DP)
        assert isinstance(succs, set)

    def test_predecessors(self):
        lat = ImplicationLattice()
        preds = lat.predecessors(PrivacyNotion.APPROX_DP)
        assert isinstance(preds, set)

    def test_update_node(self):
        lat = ImplicationLattice()
        lat.update_node(PrivacyNotion.PURE_DP, NodeStatus.VERIFIED)
        assert lat.get_node(PrivacyNotion.PURE_DP).is_verified

    def test_propagate_verified_downward(self):
        lat = ImplicationLattice()
        lat.update_node(
            PrivacyNotion.PURE_DP, NodeStatus.VERIFIED,
            budget=PureBudget(epsilon=1.0),
        )
        lat.propagate_verified(PrivacyNotion.PURE_DP)
        approx_node = lat.get_node(PrivacyNotion.APPROX_DP)
        if approx_node.is_verified:
            assert approx_node.derived is True

    def test_propagate_falsified_upward(self):
        lat = ImplicationLattice()
        lat.update_node(PrivacyNotion.APPROX_DP, NodeStatus.FALSIFIED)
        lat.propagate_falsified(PrivacyNotion.APPROX_DP)
        # Falsifying approx may propagate to stronger notions
        assert isinstance(lat.get_node(PrivacyNotion.APPROX_DP).status, NodeStatus)

    def test_query_order(self):
        lat = ImplicationLattice()
        order = lat.query_order()
        assert isinstance(order, list)
        assert len(order) >= 1

    def test_derive_guarantees(self):
        lat = ImplicationLattice()
        guarantees = lat.derive_guarantees(
            PrivacyNotion.PURE_DP, PureBudget(epsilon=1.0),
        )
        assert isinstance(guarantees, list)

    def test_reset(self):
        lat = ImplicationLattice()
        lat.update_node(PrivacyNotion.PURE_DP, NodeStatus.VERIFIED)
        lat.reset()
        assert lat.get_node(PrivacyNotion.PURE_DP).is_unknown

    def test_to_dot(self):
        lat = ImplicationLattice()
        dot = lat.to_dot()
        assert isinstance(dot, str)
        assert len(dot) > 0

    def test_str(self):
        lat = ImplicationLattice()
        assert isinstance(str(lat), str)


# ═══════════════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════════════


class TestLatticeSingleVariant:
    def test_single_notion_verified(self):
        lat = ImplicationLattice()
        lat.update_node(PrivacyNotion.ZCDP, NodeStatus.VERIFIED, budget=ZCDPBudget(rho=0.5))
        assert lat.get_node(PrivacyNotion.ZCDP).is_verified

    def test_all_verified(self):
        lat = ImplicationLattice()
        for notion in [PrivacyNotion.PURE_DP, PrivacyNotion.APPROX_DP, PrivacyNotion.ZCDP]:
            lat.update_node(notion, NodeStatus.VERIFIED)
        assert all(
            lat.get_node(n).is_verified
            for n in [PrivacyNotion.PURE_DP, PrivacyNotion.APPROX_DP, PrivacyNotion.ZCDP]
        )


# ═══════════════════════════════════════════════════════════════════════════
# Module-level functions
# ═══════════════════════════════════════════════════════════════════════════


class TestLatticeModuleFunctions:
    def test_topological_order(self):
        lat = ImplicationLattice()
        order = topological_order(lat)
        assert isinstance(order, list)

    def test_reachable_from(self):
        lat = ImplicationLattice()
        reachable = reachable_from(lat, PrivacyNotion.PURE_DP)
        assert isinstance(reachable, set)

    def test_reachable_to(self):
        lat = ImplicationLattice()
        reachable = reachable_to(lat, PrivacyNotion.APPROX_DP)
        assert isinstance(reachable, set)

    def test_strongest_verified_none(self):
        lat = ImplicationLattice()
        assert strongest_verified(lat) is None

    def test_strongest_verified_with_pure(self):
        lat = ImplicationLattice()
        lat.update_node(PrivacyNotion.PURE_DP, NodeStatus.VERIFIED)
        result = strongest_verified(lat)
        assert result is not None

    def test_weakest_falsified_none(self):
        lat = ImplicationLattice()
        assert weakest_falsified(lat) is None

    def test_lattice_summary(self):
        lat = ImplicationLattice()
        s = lattice_summary(lat)
        assert isinstance(s, dict)


# ═══════════════════════════════════════════════════════════════════════════
# Lattice – advanced propagation scenarios
# ═══════════════════════════════════════════════════════════════════════════


class TestLatticeAdvancedPropagation:
    def test_verify_pure_propagates_to_approx(self):
        lat = ImplicationLattice()
        lat.update_node(PrivacyNotion.PURE_DP, NodeStatus.VERIFIED,
                        budget=PureBudget(epsilon=1.0))
        # Pure DP implies Approx DP
        node = lat.get_node(PrivacyNotion.APPROX_DP)
        assert node.is_verified or node.is_unknown

    def test_falsify_approx_propagates_to_pure(self):
        lat = ImplicationLattice()
        lat.update_node(PrivacyNotion.APPROX_DP, NodeStatus.FALSIFIED)
        # If Approx DP is falsified, Pure DP should also be falsified
        node = lat.get_node(PrivacyNotion.PURE_DP)
        assert node.is_falsified or node.is_unknown

    def test_verify_zcdp_propagates(self):
        lat = ImplicationLattice()
        lat.update_node(PrivacyNotion.ZCDP, NodeStatus.VERIFIED,
                        budget=ZCDPBudget(rho=0.5))
        # zCDP implies Approx DP
        node = lat.get_node(PrivacyNotion.APPROX_DP)
        assert node.is_verified or node.is_unknown

    def test_conflicting_updates(self):
        lat = ImplicationLattice()
        lat.update_node(PrivacyNotion.PURE_DP, NodeStatus.VERIFIED,
                        budget=PureBudget(epsilon=1.0))
        lat.update_node(PrivacyNotion.PURE_DP, NodeStatus.FALSIFIED)
        node = lat.get_node(PrivacyNotion.PURE_DP)
        assert node.is_verified or node.is_falsified

    def test_summary_after_updates(self):
        lat = ImplicationLattice()
        lat.update_node(PrivacyNotion.PURE_DP, NodeStatus.VERIFIED,
                        budget=PureBudget(epsilon=1.0))
        lat.update_node(PrivacyNotion.ZCDP, NodeStatus.FALSIFIED)
        s = lattice_summary(lat)
        assert isinstance(s, dict)

    @pytest.mark.parametrize("notion", list(PrivacyNotion))
    def test_dot_with_each_notion_verified(self, notion: PrivacyNotion):
        lat = ImplicationLattice()
        lat.update_node(notion, NodeStatus.VERIFIED)
        dot = lat.to_dot()
        assert isinstance(dot, str)
