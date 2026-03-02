"""MechIR CFG nodes for differential privacy mechanisms.

This module defines the control-flow graph (CFG) representation used
throughout DP-CEGAR.  Mechanism programs are lowered to MechIR, which
is then analysed by the path enumerator, density lifter, and SMT
encoder.

Node hierarchy
--------------
IRNode (abstract)
  ├─ AssignNode      – variable assignment (SSA form)
  ├─ NoiseDrawNode   – noise sampling (Lap / Gauss / Exp)
  ├─ BranchNode      – if-then-else
  ├─ MergeNode       – SSA φ-node after branches
  ├─ LoopNode        – bounded for-loop
  ├─ QueryNode       – database query with sensitivity
  ├─ ReturnNode      – mechanism output
  ├─ SequenceNode    – ordered list of statements
  └─ NoOpNode        – placeholder / no-operation

Top-level containers
--------------------
MechIR  – a complete mechanism: name, params, body, return type
CFG     – explicit control-flow graph (nodes + edges)
CFGBuilder – construct CFG from a MechIR tree
"""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Sequence

from dpcegar.ir.types import (
    IRType,
    NoiseKind,
    PrivacyBudget,
    TypedExpr,
    Var,
    Const,
)
from dpcegar.utils.errors import SourceLoc, InternalError, ensure


# ═══════════════════════════════════════════════════════════════════════════
# NODE ID GENERATOR
# ═══════════════════════════════════════════════════════════════════════════

_node_id_counter = itertools.count()


def _fresh_node_id() -> int:
    """Return a globally unique node identifier."""
    return next(_node_id_counter)


# ═══════════════════════════════════════════════════════════════════════════
# BASE NODE
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class IRNode(ABC):
    """Abstract base for all MechIR control-flow nodes.

    Attributes:
        node_id:     Globally unique identifier for this node.
        source_loc:  Optional source location for diagnostics.
        annotations: Free-form metadata (e.g. sensitivity hints).
    """

    node_id: int = field(default_factory=_fresh_node_id)
    source_loc: SourceLoc | None = None
    annotations: dict[str, Any] = field(default_factory=dict)

    @abstractmethod
    def children(self) -> list[IRNode]:
        """Return immediate child nodes."""
        ...

    @abstractmethod
    def accept(self, visitor: IRNodeVisitorBase) -> Any:
        """Double-dispatch to the visitor."""
        ...

    def walk(self) -> Iterator[IRNode]:
        """Yield this node and all descendants in pre-order."""
        stack: list[IRNode] = [self]
        while stack:
            node = stack.pop()
            yield node
            stack.extend(reversed(node.children()))

    def walk_post_order(self) -> Iterator[IRNode]:
        """Yield all descendants then this node (post-order)."""
        for child in self.children():
            yield from child.walk_post_order()
        yield self

    def annotate(self, key: str, value: Any) -> None:
        """Add or update an annotation."""
        self.annotations[key] = value

    def get_annotation(self, key: str, default: Any = None) -> Any:
        """Retrieve an annotation."""
        return self.annotations.get(key, default)


# ═══════════════════════════════════════════════════════════════════════════
# CONCRETE NODES
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class AssignNode(IRNode):
    """Variable assignment in SSA form.

    Attributes:
        target:  Variable being assigned (with SSA version).
        value:   Right-hand-side expression.
    """

    target: Var = field(default_factory=lambda: Var(ty=IRType.REAL, name="_"))
    value: TypedExpr = field(default_factory=lambda: Const.zero())

    def children(self) -> list[IRNode]:
        return []

    def accept(self, visitor: IRNodeVisitorBase) -> Any:
        return visitor.visit_AssignNode(self)

    def __str__(self) -> str:
        return f"{self.target} := {self.value}"

    def __repr__(self) -> str:
        return f"AssignNode(id={self.node_id}, {self.target} := {self.value})"


@dataclass(slots=True)
class NoiseDrawNode(IRNode):
    """Noise sampling from a distribution.

    Attributes:
        target:      Variable receiving the noisy value.
        noise_kind:  Distribution family (Laplace, Gaussian, etc.).
        center:      Centre / location parameter (expression).
        scale:       Scale parameter (expression, e.g. b for Lap(b)).
        sensitivity: Declared sensitivity of the underlying query.
    """

    target: Var = field(default_factory=lambda: Var(ty=IRType.REAL, name="_noise"))
    noise_kind: NoiseKind = NoiseKind.LAPLACE
    center: TypedExpr = field(default_factory=lambda: Const.zero())
    scale: TypedExpr = field(default_factory=lambda: Const.one())
    sensitivity: TypedExpr | None = None

    def children(self) -> list[IRNode]:
        return []

    def accept(self, visitor: IRNodeVisitorBase) -> Any:
        return visitor.visit_NoiseDrawNode(self)

    def __str__(self) -> str:
        sens = f", Δ={self.sensitivity}" if self.sensitivity else ""
        return f"{self.target} ~ {self.noise_kind}(μ={self.center}, σ={self.scale}{sens})"

    def __repr__(self) -> str:
        return (
            f"NoiseDrawNode(id={self.node_id}, {self.target} ~ "
            f"{self.noise_kind}({self.center}, {self.scale}))"
        )


@dataclass(slots=True)
class BranchNode(IRNode):
    """Conditional branch (if-then-else).

    Attributes:
        condition:    Boolean guard expression.
        true_branch:  Node(s) executed when condition is true.
        false_branch: Node(s) executed when condition is false (may be NoOp).
    """

    condition: TypedExpr = field(default_factory=lambda: Const.bool_(True))
    true_branch: IRNode = field(default_factory=lambda: NoOpNode())
    false_branch: IRNode = field(default_factory=lambda: NoOpNode())

    def children(self) -> list[IRNode]:
        return [self.true_branch, self.false_branch]

    def accept(self, visitor: IRNodeVisitorBase) -> Any:
        return visitor.visit_BranchNode(self)

    def __str__(self) -> str:
        return f"if ({self.condition}) then ... else ..."

    def __repr__(self) -> str:
        return f"BranchNode(id={self.node_id}, cond={self.condition})"


@dataclass(slots=True)
class MergeNode(IRNode):
    """SSA φ-node: merges values from different control-flow predecessors.

    Attributes:
        target:  Variable receiving the merged value.
        sources: Mapping from predecessor node_id → expression.
    """

    target: Var = field(default_factory=lambda: Var(ty=IRType.REAL, name="_phi"))
    sources: dict[int, TypedExpr] = field(default_factory=dict)

    def children(self) -> list[IRNode]:
        return []

    def accept(self, visitor: IRNodeVisitorBase) -> Any:
        return visitor.visit_MergeNode(self)

    def add_source(self, pred_id: int, expr: TypedExpr) -> None:
        """Register a predecessor's contributed expression."""
        self.sources[pred_id] = expr

    def __str__(self) -> str:
        srcs = ", ".join(f"n{k}→{v}" for k, v in self.sources.items())
        return f"{self.target} = φ({srcs})"


@dataclass(slots=True)
class LoopNode(IRNode):
    """Bounded for-loop.

    Semantics: ``for index_var in range(bound): body``

    Attributes:
        index_var:      Loop index variable.
        bound:          Upper bound expression (exclusive).
        body:           Loop body node.
        unroll_count:   Hint for how many iterations to unroll
                        (None = use global config).
    """

    index_var: Var = field(default_factory=lambda: Var(ty=IRType.INT, name="i"))
    bound: TypedExpr = field(default_factory=lambda: Const.int_(1))
    body: IRNode = field(default_factory=lambda: NoOpNode())
    unroll_count: int | None = None

    def children(self) -> list[IRNode]:
        return [self.body]

    def accept(self, visitor: IRNodeVisitorBase) -> Any:
        return visitor.visit_LoopNode(self)

    def __str__(self) -> str:
        uc = f" [unroll={self.unroll_count}]" if self.unroll_count else ""
        return f"for {self.index_var} in range({self.bound}){uc}: ..."


@dataclass(slots=True)
class QueryNode(IRNode):
    """Database query with sensitivity annotation.

    Attributes:
        target:      Variable receiving the query result.
        query_name:  Identifier for the query function.
        args:        Arguments passed to the query.
        sensitivity: Declared ℓ₁ or ℓ₂ sensitivity.
    """

    target: Var = field(default_factory=lambda: Var(ty=IRType.REAL, name="_q"))
    query_name: str = ""
    args: tuple[TypedExpr, ...] = ()
    sensitivity: TypedExpr = field(default_factory=lambda: Const.one())

    def children(self) -> list[IRNode]:
        return []

    def accept(self, visitor: IRNodeVisitorBase) -> Any:
        return visitor.visit_QueryNode(self)

    def __str__(self) -> str:
        a = ", ".join(str(x) for x in self.args)
        return f"{self.target} = {self.query_name}({a})  [Δ={self.sensitivity}]"


@dataclass(slots=True)
class ReturnNode(IRNode):
    """Mechanism output / return statement.

    Attributes:
        value: The expression being returned.
    """

    value: TypedExpr = field(default_factory=lambda: Const.zero())

    def children(self) -> list[IRNode]:
        return []

    def accept(self, visitor: IRNodeVisitorBase) -> Any:
        return visitor.visit_ReturnNode(self)

    def __str__(self) -> str:
        return f"return {self.value}"


@dataclass(slots=True)
class SequenceNode(IRNode):
    """Ordered sequence of statements.

    Attributes:
        stmts: List of child nodes executed in order.
    """

    stmts: list[IRNode] = field(default_factory=list)

    def children(self) -> list[IRNode]:
        return list(self.stmts)

    def accept(self, visitor: IRNodeVisitorBase) -> Any:
        return visitor.visit_SequenceNode(self)

    def append(self, node: IRNode) -> None:
        """Append a statement to the sequence."""
        self.stmts.append(node)

    def __len__(self) -> int:
        return len(self.stmts)

    def __iter__(self) -> Iterator[IRNode]:
        return iter(self.stmts)

    def __str__(self) -> str:
        return f"seq[{len(self.stmts)} stmts]"


@dataclass(slots=True)
class NoOpNode(IRNode):
    """Placeholder / no-operation node."""

    def children(self) -> list[IRNode]:
        return []

    def accept(self, visitor: IRNodeVisitorBase) -> Any:
        return visitor.visit_NoOpNode(self)

    def __str__(self) -> str:
        return "noop"


# ═══════════════════════════════════════════════════════════════════════════
# VISITOR BASE (for IR nodes)
# ═══════════════════════════════════════════════════════════════════════════


class IRNodeVisitorBase(ABC):
    """Visitor interface for IR nodes.

    Override the ``visit_*`` methods you need; the default for each
    delegates to :meth:`generic_visit`.
    """

    def visit(self, node: IRNode) -> Any:
        return node.accept(self)

    def generic_visit(self, node: IRNode) -> Any:
        """Fallback: visit all children."""
        results = []
        for child in node.children():
            results.append(self.visit(child))
        return results

    def visit_AssignNode(self, node: AssignNode) -> Any:
        return self.generic_visit(node)

    def visit_NoiseDrawNode(self, node: NoiseDrawNode) -> Any:
        return self.generic_visit(node)

    def visit_BranchNode(self, node: BranchNode) -> Any:
        return self.generic_visit(node)

    def visit_MergeNode(self, node: MergeNode) -> Any:
        return self.generic_visit(node)

    def visit_LoopNode(self, node: LoopNode) -> Any:
        return self.generic_visit(node)

    def visit_QueryNode(self, node: QueryNode) -> Any:
        return self.generic_visit(node)

    def visit_ReturnNode(self, node: ReturnNode) -> Any:
        return self.generic_visit(node)

    def visit_SequenceNode(self, node: SequenceNode) -> Any:
        return self.generic_visit(node)

    def visit_NoOpNode(self, node: NoOpNode) -> Any:
        return self.generic_visit(node)


# ═══════════════════════════════════════════════════════════════════════════
# TOP-LEVEL IR CONTAINER
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class ParamDecl:
    """A mechanism parameter declaration.

    Attributes:
        name: Parameter name.
        ty:   Parameter type.
        is_database: True if this is the database input (for adjacency).
    """

    name: str
    ty: IRType
    is_database: bool = False

    def __str__(self) -> str:
        db = " [DB]" if self.is_database else ""
        return f"{self.name}: {self.ty}{db}"


@dataclass(slots=True)
class MechIR:
    """Top-level container for a differential privacy mechanism.

    Attributes:
        name:        Mechanism name.
        params:      Parameter declarations.
        body:        Root node of the mechanism body.
        return_type: Return type.
        budget:      Declared privacy budget (if any).
        metadata:    Free-form metadata.
    """

    name: str = "unnamed"
    params: list[ParamDecl] = field(default_factory=list)
    body: IRNode = field(default_factory=NoOpNode)
    return_type: IRType = IRType.REAL
    budget: PrivacyBudget | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def all_nodes(self) -> Iterator[IRNode]:
        """Iterate over all nodes in the body (pre-order)."""
        return self.body.walk()

    def node_count(self) -> int:
        """Return the total number of IR nodes in the body."""
        return sum(1 for _ in self.all_nodes())

    def noise_draws(self) -> list[NoiseDrawNode]:
        """Return all noise-draw nodes in the mechanism."""
        return [n for n in self.all_nodes() if isinstance(n, NoiseDrawNode)]

    def queries(self) -> list[QueryNode]:
        """Return all query nodes in the mechanism."""
        return [n for n in self.all_nodes() if isinstance(n, QueryNode)]

    def find_node(self, node_id: int) -> IRNode | None:
        """Find a node by its ID, or return None."""
        for n in self.all_nodes():
            if n.node_id == node_id:
                return n
        return None

    def __str__(self) -> str:
        params = ", ".join(str(p) for p in self.params)
        budget = f" [{self.budget}]" if self.budget else ""
        return f"mechanism {self.name}({params}) -> {self.return_type}{budget}"


# ═══════════════════════════════════════════════════════════════════════════
# CONTROL-FLOW GRAPH
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class CFGEdge:
    """An edge in the control-flow graph.

    Attributes:
        src:       Source node ID.
        dst:       Destination node ID.
        condition: Guard condition (None for unconditional edges).
        label:     Human-readable label for debugging.
    """

    src: int
    dst: int
    condition: TypedExpr | None = None
    label: str = ""

    def __str__(self) -> str:
        cond = f" [{self.condition}]" if self.condition else ""
        lbl = f" ({self.label})" if self.label else ""
        return f"n{self.src} → n{self.dst}{cond}{lbl}"


@dataclass(slots=True)
class CFG:
    """Explicit control-flow graph representation.

    Nodes are stored by ID in a dictionary; edges are stored both as
    a flat list and in adjacency maps for fast lookup.

    Attributes:
        nodes:      Mapping from node_id to IRNode.
        edges:      List of all edges.
        entry:      Entry node ID.
        exit_node:  Exit node ID (may be a synthetic merge).
    """

    nodes: dict[int, IRNode] = field(default_factory=dict)
    edges: list[CFGEdge] = field(default_factory=list)
    entry: int = -1
    exit_node: int = -1

    # -- Adjacency caches (built lazily) --
    _successors: dict[int, list[CFGEdge]] = field(default_factory=lambda: defaultdict(list), repr=False)
    _predecessors: dict[int, list[CFGEdge]] = field(default_factory=lambda: defaultdict(list), repr=False)
    _dirty: bool = field(default=True, repr=False)

    def _rebuild_adjacency(self) -> None:
        """Rebuild successor/predecessor maps from the edge list."""
        self._successors = defaultdict(list)
        self._predecessors = defaultdict(list)
        for e in self.edges:
            self._successors[e.src].append(e)
            self._predecessors[e.dst].append(e)
        self._dirty = False

    def add_node(self, node: IRNode) -> None:
        """Add a node to the CFG."""
        self.nodes[node.node_id] = node
        self._dirty = True

    def add_edge(self, src: int, dst: int, condition: TypedExpr | None = None, label: str = "") -> CFGEdge:
        """Add an edge to the CFG."""
        edge = CFGEdge(src=src, dst=dst, condition=condition, label=label)
        self.edges.append(edge)
        self._dirty = True
        return edge

    def successors(self, node_id: int) -> list[CFGEdge]:
        """Return outgoing edges from *node_id*."""
        if self._dirty:
            self._rebuild_adjacency()
        return self._successors.get(node_id, [])

    def predecessors(self, node_id: int) -> list[CFGEdge]:
        """Return incoming edges to *node_id*."""
        if self._dirty:
            self._rebuild_adjacency()
        return self._predecessors.get(node_id, [])

    def successor_ids(self, node_id: int) -> list[int]:
        """Return IDs of successor nodes."""
        return [e.dst for e in self.successors(node_id)]

    def predecessor_ids(self, node_id: int) -> list[int]:
        """Return IDs of predecessor nodes."""
        return [e.src for e in self.predecessors(node_id)]

    # -- Graph algorithms --------------------------------------------------

    def reachable_from(self, start: int) -> set[int]:
        """Return all node IDs reachable from *start* via BFS."""
        visited: set[int] = set()
        queue: deque[int] = deque([start])
        while queue:
            nid = queue.popleft()
            if nid in visited:
                continue
            visited.add(nid)
            queue.extend(self.successor_ids(nid))
        return visited

    def topological_order(self) -> list[int]:
        """Return node IDs in topological order (Kahn's algorithm).

        Raises :class:`InternalError` if the CFG contains a cycle
        (which should not happen after loop unrolling).
        """
        if self._dirty:
            self._rebuild_adjacency()

        in_degree: dict[int, int] = {nid: 0 for nid in self.nodes}
        for e in self.edges:
            in_degree[e.dst] = in_degree.get(e.dst, 0) + 1

        queue: deque[int] = deque(nid for nid, d in in_degree.items() if d == 0)
        order: list[int] = []

        while queue:
            nid = queue.popleft()
            order.append(nid)
            for e in self.successors(nid):
                in_degree[e.dst] -= 1
                if in_degree[e.dst] == 0:
                    queue.append(e.dst)

        if len(order) != len(self.nodes):
            raise InternalError("CFG contains a cycle (topological sort failed)")
        return order

    def dominators(self) -> dict[int, set[int]]:
        """Compute dominator sets for each node.

        Returns:
            Mapping from node_id to the set of node_ids that dominate it.
        """
        all_ids = set(self.nodes.keys())
        dom: dict[int, set[int]] = {nid: set(all_ids) for nid in all_ids}
        dom[self.entry] = {self.entry}

        changed = True
        while changed:
            changed = False
            for nid in self.nodes:
                if nid == self.entry:
                    continue
                preds = self.predecessor_ids(nid)
                if not preds:
                    new_dom = {nid}
                else:
                    new_dom = set.intersection(*(dom[p] for p in preds))
                    new_dom = new_dom | {nid}
                if new_dom != dom[nid]:
                    dom[nid] = new_dom
                    changed = True
        return dom

    def post_dominators(self) -> dict[int, set[int]]:
        """Compute post-dominator sets for each node.

        Returns:
            Mapping from node_id to the set of node_ids that post-dominate it.
        """
        all_ids = set(self.nodes.keys())
        pdom: dict[int, set[int]] = {nid: set(all_ids) for nid in all_ids}
        pdom[self.exit_node] = {self.exit_node}

        changed = True
        while changed:
            changed = False
            for nid in self.nodes:
                if nid == self.exit_node:
                    continue
                succs = self.successor_ids(nid)
                if not succs:
                    new_pdom = {nid}
                else:
                    new_pdom = set.intersection(*(pdom[s] for s in succs))
                    new_pdom = new_pdom | {nid}
                if new_pdom != pdom[nid]:
                    pdom[nid] = new_pdom
                    changed = True
        return pdom

    def node_count(self) -> int:
        """Number of nodes in the CFG."""
        return len(self.nodes)

    def edge_count(self) -> int:
        """Number of edges in the CFG."""
        return len(self.edges)

    def __str__(self) -> str:
        return (
            f"CFG(nodes={self.node_count()}, edges={self.edge_count()}, "
            f"entry=n{self.entry}, exit=n{self.exit_node})"
        )


# ═══════════════════════════════════════════════════════════════════════════
# CFG BUILDER
# ═══════════════════════════════════════════════════════════════════════════


class CFGBuilder:
    """Construct a CFG from a MechIR tree.

    Usage::

        mechir = MechIR(...)
        builder = CFGBuilder()
        cfg = builder.build(mechir)
    """

    def __init__(self) -> None:
        self._cfg = CFG()

    def build(self, mechir: MechIR) -> CFG:
        """Build and return the CFG for the given mechanism.

        Args:
            mechir: The mechanism IR to convert.

        Returns:
            A fully constructed :class:`CFG`.
        """
        self._cfg = CFG()

        # Create synthetic entry and exit nodes
        entry = NoOpNode()
        entry.annotate("role", "entry")
        exit_node = NoOpNode()
        exit_node.annotate("role", "exit")

        self._cfg.add_node(entry)
        self._cfg.add_node(exit_node)
        self._cfg.entry = entry.node_id
        self._cfg.exit_node = exit_node.node_id

        # Process body
        last_ids = self._process_node(mechir.body, [entry.node_id])

        # Connect last nodes to exit
        for lid in last_ids:
            self._cfg.add_edge(lid, exit_node.node_id)

        return self._cfg

    def _process_node(self, node: IRNode, pred_ids: list[int]) -> list[int]:
        """Process an IR node, add it to the CFG, connect to predecessors.

        Returns the list of node IDs that represent the "tail" of this
        sub-graph (i.e. nodes that should be connected to whatever comes
        next).
        """
        self._cfg.add_node(node)

        if isinstance(node, SequenceNode):
            current_preds = pred_ids
            for stmt in node.stmts:
                current_preds = self._process_node(stmt, current_preds)
            return current_preds

        elif isinstance(node, BranchNode):
            # Connect predecessors to the branch node
            for pid in pred_ids:
                self._cfg.add_edge(pid, node.node_id)

            # Process true branch
            true_tails = self._process_node(
                node.true_branch, [node.node_id]
            )
            # Process false branch
            false_tails = self._process_node(
                node.false_branch, [node.node_id]
            )

            return true_tails + false_tails

        elif isinstance(node, LoopNode):
            # Connect predecessors to the loop header
            for pid in pred_ids:
                self._cfg.add_edge(pid, node.node_id)

            # Process body
            body_tails = self._process_node(node.body, [node.node_id])

            # Back-edge from body tail to loop header
            for tid in body_tails:
                self._cfg.add_edge(tid, node.node_id, label="back")

            return [node.node_id]

        else:
            # Simple node: connect all predecessors to this node
            for pid in pred_ids:
                self._cfg.add_edge(pid, node.node_id)
            return [node.node_id]
