"""Proof extraction and format conversion.

Extracts UNSAT proofs from SMT solver results and converts them
to standard proof formats (LFSC, Alethe) with mechanism-specific
annotations for differential privacy verification.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Sequence

from dpcegar.ir.types import PrivacyBudget, PrivacyNotion, TypedExpr


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ProofFormat(Enum):
    """Supported proof output formats."""

    INTERNAL = auto()
    LFSC = auto()
    ALETHE = auto()
    SMTLIB2 = auto()
    DOT = auto()


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------


@dataclass
class ProofNode:
    """A single node in an UNSAT proof tree.

    Each node records one inference step: the rule applied,
    the formula it derives, and which earlier nodes serve as
    premises.
    """

    node_id: int
    rule: str
    conclusion: str
    premises: list[int] = field(default_factory=list)
    theory: str = "core"
    annotations: dict[str, Any] = field(default_factory=dict)

    @property
    def is_leaf(self) -> bool:
        """True when the node is an axiom or assumption (no premises)."""
        return len(self.premises) == 0


@dataclass
class ProofTree:
    """Complete UNSAT proof represented as a directed acyclic graph.

    Nodes are keyed by their integer *node_id*; the proof's
    conclusion is at ``root_id``.
    """

    nodes: dict[int, ProofNode] = field(default_factory=dict)
    root_id: int = 0

    # -- mutators ----------------------------------------------------------

    def add_node(self, node: ProofNode) -> None:
        """Insert *node* into the tree, keyed by ``node.node_id``."""
        self.nodes[node.node_id] = node

    # -- accessors ---------------------------------------------------------

    def get_node(self, node_id: int) -> ProofNode | None:
        """Return the node with the given id, or ``None``."""
        return self.nodes.get(node_id)

    def depth(self) -> int:
        """Longest path from root to any leaf."""
        if not self.nodes:
            return 0

        memo: dict[int, int] = {}

        def _depth(nid: int) -> int:
            if nid in memo:
                return memo[nid]
            node = self.nodes.get(nid)
            if node is None or node.is_leaf:
                memo[nid] = 0
                return 0
            d = 1 + max(_depth(p) for p in node.premises)
            memo[nid] = d
            return d

        return _depth(self.root_id)

    def size(self) -> int:
        """Total number of nodes in the tree."""
        return len(self.nodes)

    def leaves(self) -> list[ProofNode]:
        """Return all leaf nodes (axioms and assumptions)."""
        return [n for n in self.nodes.values() if n.is_leaf]

    def theories_used(self) -> set[str]:
        """Collect the set of SMT theories referenced by any node."""
        return {n.theory for n in self.nodes.values()}

    # -- serialisation -----------------------------------------------------

    def to_dot(self) -> str:
        """Render the proof as a Graphviz DOT string."""
        lines: list[str] = ["digraph proof {", "  rankdir=BT;"]
        for nid, node in sorted(self.nodes.items()):
            label = f"{node.rule}\\n{node.conclusion[:40]}"
            shape = "box" if node.is_leaf else "ellipse"
            lines.append(f'  n{nid} [label="{label}", shape={shape}];')
            for pid in node.premises:
                lines.append(f"  n{pid} -> n{nid};")
        lines.append("}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


@dataclass
class ProofStatistics:
    """Aggregate metrics for a proof tree (optionally after simplification)."""

    total_nodes: int = 0
    depth: int = 0
    theories: set[str] = field(default_factory=set)
    leaf_count: int = 0
    rule_histogram: dict[str, int] = field(default_factory=dict)
    compression_ratio: float = 1.0


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


class ProofExtractor:
    """Build a :class:`ProofTree` from raw SMT solver output.

    Handles Z3 proof objects and generic UNSAT-core lists produced by
    the DP-CEGAR verification loop.
    """

    def extract_from_smt_result(
        self,
        result: Any,
        encoding_info: dict[str, Any],
    ) -> ProofTree | None:
        """Extract a proof tree from an :class:`SMTResult`.

        Returns ``None`` when the result does not contain a proof
        (e.g. the status is SAT or UNKNOWN).
        """
        from dpcegar.cegar.engine import SMTStatus

        if result.status is not SMTStatus.UNSAT:
            return None

        raw_proof = getattr(result, "proof", None)
        if raw_proof is not None:
            tree = self._build_proof_tree(raw_proof)
        elif result.unsat_core:
            typed_core = [
                expr if isinstance(expr, TypedExpr) else expr
                for expr in result.unsat_core
            ]
            tree = self.extract_unsat_core_proof(typed_core)
        else:
            tree = ProofTree()
            tree.add_node(
                ProofNode(
                    node_id=0,
                    rule="unsat",
                    conclusion="false",
                    theory="core",
                )
            )
            tree.root_id = 0

        return self._map_constraints_to_mechanism(tree, encoding_info)

    def extract_unsat_core_proof(
        self,
        unsat_core: list[Any],
    ) -> ProofTree:
        """Construct a skeletal proof tree from an UNSAT core.

        Each core element becomes a leaf assumption; a single
        ``resolution`` step derives ``false``.
        """
        tree = ProofTree()
        premise_ids: list[int] = []
        for idx, expr in enumerate(unsat_core):
            conclusion = str(expr)
            node = ProofNode(
                node_id=idx,
                rule="assumption",
                conclusion=conclusion,
                theory="core",
                annotations={"core_index": idx},
            )
            tree.add_node(node)
            premise_ids.append(idx)

        root_id = len(unsat_core)
        root = ProofNode(
            node_id=root_id,
            rule="resolution",
            conclusion="false",
            premises=premise_ids,
            theory="core",
        )
        tree.add_node(root)
        tree.root_id = root_id
        return tree

    # -- internal helpers --------------------------------------------------

    def _build_proof_tree(self, raw_proof: Any) -> ProofTree:
        """Dispatch to solver-specific extraction logic."""
        type_name = type(raw_proof).__name__
        if "ExprRef" in type_name or "Proof" in type_name:
            return self._extract_z3_proof(raw_proof)

        tree = ProofTree()
        tree.add_node(
            ProofNode(
                node_id=0,
                rule="opaque",
                conclusion=str(raw_proof)[:200],
                theory="core",
                annotations={"raw_type": type_name},
            )
        )
        tree.root_id = 0
        return tree

    def _extract_z3_proof(self, z3_proof: Any) -> ProofTree:
        """Walk a Z3 proof object and populate a :class:`ProofTree`.

        Nodes are assigned sequential IDs via a post-order traversal so
        that every premise is numbered before the step that uses it.
        """
        tree = ProofTree()
        visited: dict[int, int] = {}
        counter: list[int] = [0]

        def _visit(expr: Any) -> int:
            eid = id(expr)
            if eid in visited:
                return visited[eid]

            children = (
                list(expr.children()) if hasattr(expr, "children") else []
            )
            premise_ids = [_visit(c) for c in children]

            nid = counter[0]
            counter[0] += 1

            decl_name = ""
            if hasattr(expr, "decl"):
                try:
                    decl_name = str(expr.decl())
                except Exception:
                    decl_name = "unknown"

            node = ProofNode(
                node_id=nid,
                rule=decl_name or "z3_step",
                conclusion=str(expr)[:200],
                premises=premise_ids,
                theory=self._infer_theory(decl_name),
            )
            tree.add_node(node)
            visited[eid] = nid
            return nid

        root_nid = _visit(z3_proof)
        tree.root_id = root_nid
        return tree

    @staticmethod
    def _infer_theory(rule_name: str) -> str:
        """Guess the SMT theory a proof rule belongs to."""
        rule_lower = rule_name.lower()
        if any(k in rule_lower for k in ("arith", "lra", "nra", "farkas")):
            return "LRA"
        if any(k in rule_lower for k in ("nla", "nonlinear")):
            return "NRA"
        if any(k in rule_lower for k in ("quant", "forall", "exists", "inst")):
            return "quantifiers"
        return "core"

    def _map_constraints_to_mechanism(
        self,
        proof: ProofTree,
        encoding_info: dict[str, Any],
    ) -> ProofTree:
        """Annotate proof nodes with the DP mechanism constraints they stem from."""
        constraint_map: dict[str, str] = encoding_info.get(
            "constraint_origins", {}
        )
        for node in proof.nodes.values():
            for fragment, origin in constraint_map.items():
                if fragment in node.conclusion:
                    node.annotations["mechanism_origin"] = origin
                    break
        return proof


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------


class ProofAnnotator:
    """Enrich a proof tree with DP-specific metadata.

    Annotations allow downstream consumers (e.g. certificate checkers)
    to relate proof steps back to the privacy argument.
    """

    def annotate(
        self,
        proof: ProofTree,
        mechanism_name: str,
        budget: PrivacyBudget,
    ) -> ProofTree:
        """Apply all annotation passes and return the enriched tree."""
        proof = self._annotate_density_bounds(proof)
        proof = self._annotate_privacy_predicate(proof, budget)
        proof = self._add_mechanism_context(proof, mechanism_name)
        return proof

    def _annotate_density_bounds(self, proof: ProofTree) -> ProofTree:
        """Mark nodes whose conclusions involve density-ratio bounds."""
        density_keywords = ("density", "ratio", "exp(", "log(", "prob")
        for node in proof.nodes.values():
            concl_lower = node.conclusion.lower()
            if any(kw in concl_lower for kw in density_keywords):
                node.annotations["density_bound"] = True
        return proof

    def _annotate_privacy_predicate(
        self,
        proof: ProofTree,
        budget: PrivacyBudget,
    ) -> ProofTree:
        """Tag nodes that reference the overall privacy predicate."""
        budget_str = str(budget).lower()
        privacy_keywords = ("epsilon", "delta", "privacy", "dp", budget_str)
        for node in proof.nodes.values():
            concl_lower = node.conclusion.lower()
            if any(kw in concl_lower for kw in privacy_keywords if kw):
                node.annotations["privacy_predicate"] = True
                node.annotations["budget_snapshot"] = str(budget)
        return proof

    def _add_mechanism_context(
        self,
        proof: ProofTree,
        mechanism_name: str,
    ) -> ProofTree:
        """Store the mechanism name on the root node."""
        root = proof.get_node(proof.root_id)
        if root is not None:
            root.annotations["mechanism"] = mechanism_name
            root.annotations["annotated_at"] = time.time()
        return proof


# ---------------------------------------------------------------------------
# Simplification
# ---------------------------------------------------------------------------


class ProofSimplifier:
    """Reduce proof size while preserving logical validity."""

    def simplify(self, proof: ProofTree) -> ProofTree:
        """Run all simplification passes and return a smaller tree."""
        proof = self._remove_irrelevant_lemmas(proof)
        proof = self._collapse_chains(proof)
        proof = self._remove_redundant_steps(proof)
        return proof

    def _remove_irrelevant_lemmas(self, proof: ProofTree) -> ProofTree:
        """Drop lemma nodes that are not reachable from the root."""
        reachable: set[int] = set()

        def _walk(nid: int) -> None:
            if nid in reachable:
                return
            reachable.add(nid)
            node = proof.get_node(nid)
            if node is not None:
                for pid in node.premises:
                    _walk(pid)

        _walk(proof.root_id)

        trimmed = ProofTree(root_id=proof.root_id)
        for nid in reachable:
            node = proof.get_node(nid)
            if node is not None:
                trimmed.add_node(node)
        return trimmed

    def _collapse_chains(self, proof: ProofTree) -> ProofTree:
        """Collapse single-premise inference chains into one step.

        A chain ``A -> B -> C`` where B has exactly one premise
        is shortened to ``A -> C`` by inheriting A's premises.
        """
        child_count: dict[int, int] = {nid: 0 for nid in proof.nodes}
        for node in proof.nodes.values():
            for pid in node.premises:
                if pid in child_count:
                    child_count[pid] += 1

        collapsed = ProofTree(root_id=proof.root_id)
        skip: set[int] = set()

        for nid, node in proof.nodes.items():
            if nid in skip:
                continue
            if (
                len(node.premises) == 1
                and node.premises[0] in proof.nodes
                and child_count.get(node.premises[0], 0) == 1
                and nid != proof.root_id
            ):
                parent = proof.nodes[node.premises[0]]
                new_node = ProofNode(
                    node_id=nid,
                    rule=f"{parent.rule}+{node.rule}",
                    conclusion=node.conclusion,
                    premises=list(parent.premises),
                    theory=node.theory,
                    annotations={**parent.annotations, **node.annotations},
                )
                collapsed.add_node(new_node)
                skip.add(node.premises[0])
            else:
                collapsed.add_node(node)

        return collapsed

    def _remove_redundant_steps(self, proof: ProofTree) -> ProofTree:
        """Deduplicate nodes that derive the same conclusion via the same rule."""
        seen: dict[tuple[str, str], int] = {}
        id_remap: dict[int, int] = {}

        deduped = ProofTree(root_id=proof.root_id)

        for nid in sorted(proof.nodes):
            node = proof.nodes[nid]
            key = (node.rule, node.conclusion)
            if key in seen and node.is_leaf:
                id_remap[nid] = seen[key]
            else:
                seen[key] = nid
                id_remap[nid] = nid
                deduped.add_node(node)

        for node in deduped.nodes.values():
            node.premises = [id_remap.get(p, p) for p in node.premises]

        deduped.root_id = id_remap.get(proof.root_id, proof.root_id)
        return deduped

    def statistics(
        self,
        original: ProofTree,
        simplified: ProofTree,
    ) -> ProofStatistics:
        """Compute aggregate metrics comparing *original* and *simplified*."""
        rule_hist: dict[str, int] = {}
        for node in simplified.nodes.values():
            rule_hist[node.rule] = rule_hist.get(node.rule, 0) + 1

        orig_size = max(original.size(), 1)
        return ProofStatistics(
            total_nodes=simplified.size(),
            depth=simplified.depth(),
            theories=simplified.theories_used(),
            leaf_count=len(simplified.leaves()),
            rule_histogram=rule_hist,
            compression_ratio=simplified.size() / orig_size,
        )


# ---------------------------------------------------------------------------
# Format converters
# ---------------------------------------------------------------------------


class LFSCFormatter:
    """Emit a proof tree in LFSC (Logical Framework with Side Conditions)."""

    def format(self, proof: ProofTree) -> str:
        """Return the full LFSC proof string."""
        lines: list[str] = [
            ";; LFSC proof generated by dp-cegar proof extractor",
            "(check",
        ]
        for nid in sorted(proof.nodes):
            lines.append(self._format_node(proof.nodes[nid]))
        lines.append(")")
        return "\n".join(lines)

    def _format_node(self, node: ProofNode) -> str:
        """Format a single proof node as an LFSC declaration."""
        lfsc_rule = self._rule_to_lfsc(node.rule)
        lfsc_formula = self._formula_to_lfsc(node.conclusion)
        if node.is_leaf:
            return f"  (assume _{node.node_id} {lfsc_formula})"
        premise_refs = " ".join(f"_{p}" for p in node.premises)
        return f"  ({lfsc_rule} _{node.node_id} {lfsc_formula} {premise_refs})"

    @staticmethod
    def _rule_to_lfsc(rule: str) -> str:
        """Map an internal rule name to its LFSC equivalent."""
        mapping: dict[str, str] = {
            "resolution": "R",
            "assumption": "assume",
            "unit_resolution": "ur",
            "rewrite": "rw",
            "monotonicity": "mono",
            "transitivity": "trans",
            "reflexivity": "refl",
            "symmetry": "symm",
            "modus_ponens": "mp",
        }
        return mapping.get(rule, rule)

    @staticmethod
    def _formula_to_lfsc(formula: str) -> str:
        """Translate a formula string to LFSC s-expression syntax."""
        result = formula
        replacements = [
            ("&&", "and"),
            ("||", "or"),
            ("!", "not"),
            ("==", "="),
            ("!=", "distinct"),
            ("<=", "<="),
            (">=", ">="),
        ]
        for src, dst in replacements:
            result = result.replace(src, dst)
        if " " in result and not result.startswith("("):
            result = f"({result})"
        return result


class AletheFormatter:
    """Emit a proof tree in Alethe format (used by cvc5 / veriT)."""

    def format(self, proof: ProofTree) -> str:
        """Return the full Alethe proof string."""
        lines: list[str] = [
            ";; Alethe proof generated by dp-cegar proof extractor",
            "(set-logic ALL)",
        ]
        for nid in sorted(proof.nodes):
            lines.append(self._format_step(proof.nodes[nid]))
        return "\n".join(lines)

    def _format_step(self, node: ProofNode) -> str:
        """Format one Alethe ``step`` or ``assume`` line."""
        alethe_rule = self._rule_to_alethe(node.rule)
        if node.is_leaf:
            return f"(assume t{node.node_id} {node.conclusion})"
        premises_str = " ".join(f"t{p}" for p in node.premises)
        return (
            f"(step t{node.node_id} "
            f"(cl {node.conclusion}) "
            f":rule {alethe_rule} "
            f":premises ({premises_str}))"
        )

    @staticmethod
    def _rule_to_alethe(rule: str) -> str:
        """Map an internal rule name to the Alethe equivalent."""
        mapping: dict[str, str] = {
            "resolution": "resolution",
            "assumption": "assume",
            "rewrite": "eq_simplify",
            "unit_resolution": "th_resolution",
            "monotonicity": "cong",
            "transitivity": "trans",
            "reflexivity": "refl",
            "symmetry": "symm",
            "modus_ponens": "implies",
        }
        return mapping.get(rule, rule)


class SMTLIB2Formatter:
    """Emit a proof tree as annotated SMT-LIB2 declarations."""

    def format(self, proof: ProofTree) -> str:
        """Return the full SMT-LIB2 proof string.

        Each node is rendered as a ``define-fun`` with a ``:proof-rule``
        attribute so that external tools can reconstruct the derivation.
        """
        lines: list[str] = [
            ";; SMT-LIB2 annotated proof",
            ";; generated by dp-cegar proof extractor",
            "(set-logic ALL)",
        ]

        for nid in sorted(proof.nodes):
            node = proof.nodes[nid]
            if node.is_leaf:
                lines.append(
                    f"(assert (! {node.conclusion} "
                    f":named step_{nid} "
                    f":proof-rule {node.rule}))"
                )
            else:
                deps = " ".join(f"step_{p}" for p in node.premises)
                lines.append(
                    f"(assert (! {node.conclusion} "
                    f":named step_{nid} "
                    f":proof-rule {node.rule} "
                    f":proof-deps ({deps})))"
                )

        annotation_nodes = [
            n
            for n in proof.nodes.values()
            if n.annotations
        ]
        if annotation_nodes:
            lines.append(";; mechanism annotations")
            for node in annotation_nodes:
                for key, val in node.annotations.items():
                    lines.append(
                        f"(set-info :dp-annotation-{node.node_id}-{key} "
                        f'"{val}")'
                    )

        lines.append("(exit)")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


class ProofHasher:
    """Deterministic hashing for proof deduplication and caching."""

    def hash_proof(self, proof: ProofTree) -> str:
        """Compute a SHA-256 digest covering every node in the tree.

        The hash is independent of insertion order because nodes are
        visited in sorted-id order.
        """
        h = hashlib.sha256()
        for nid in sorted(proof.nodes):
            node = proof.nodes[nid]
            h.update(f"{nid}|{node.rule}|{node.conclusion}|".encode())
            h.update(",".join(str(p) for p in node.premises).encode())
            h.update(f"|{node.theory}\n".encode())
        return h.hexdigest()

    def hash_conclusion(self, proof: ProofTree) -> str:
        """Hash only the root's conclusion for quick equivalence checks."""
        root = proof.get_node(proof.root_id)
        conclusion = root.conclusion if root is not None else ""
        return hashlib.sha256(conclusion.encode()).hexdigest()
