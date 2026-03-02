"""Symbolic path representation for differential privacy mechanism analysis.

A symbolic path records the control-flow decisions, noise draws, variable
assignments, and output expression encountered along a single feasible
execution of a MechIR program.  The path enumeration engine produces a
:class:`PathSet` of :class:`SymbolicPath` objects, which are then fed
into the density ratio builder and SMT encoder.

Classes
-------
NoiseDrawInfo   – metadata for a single noise draw along a path
PathCondition   – conjunction of boolean predicates (path guard)
SymbolicPath    – complete symbolic execution trace
PathSet         – collection of SymbolicPaths with query operations
"""

from __future__ import annotations

import copy
import itertools
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple

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

# Boolean expressions are just TypedExpr with ty == IRType.BOOL.
BoolExpr = TypedExpr

_path_id_counter = itertools.count()


def _fresh_path_id() -> int:
    """Return a globally unique path identifier."""
    return next(_path_id_counter)


# ═══════════════════════════════════════════════════════════════════════════
# NOISE DRAW INFO
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class NoiseDrawInfo:
    """Metadata for a single noise draw encountered on a symbolic path.

    Attributes:
        variable:    Name of the variable receiving the noisy value.
        kind:        Distribution family (LAPLACE, GAUSSIAN, EXPONENTIAL).
        center_expr: Symbolic expression for the centre/location parameter.
        scale_expr:  Symbolic expression for the scale parameter.
        site_id:     Unique identifier of the corresponding NoiseDrawNode.
    """

    variable: str
    kind: NoiseKind
    center_expr: TypedExpr
    scale_expr: TypedExpr
    site_id: int

    def with_substitution(self, mapping: dict[str, TypedExpr]) -> NoiseDrawInfo:
        """Return a copy with *mapping* applied to symbolic expressions."""
        return NoiseDrawInfo(
            variable=self.variable,
            kind=self.kind,
            center_expr=self.center_expr.substitute(mapping),
            scale_expr=self.scale_expr.substitute(mapping),
            site_id=self.site_id,
        )

    def free_vars(self) -> frozenset[str]:
        """Return free variable names appearing in centre and scale."""
        return self.center_expr.free_vars() | self.scale_expr.free_vars()

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly dictionary."""
        return {
            "variable": self.variable,
            "kind": self.kind.name,
            "center_expr": str(self.center_expr),
            "scale_expr": str(self.scale_expr),
            "site_id": self.site_id,
        }

    def __str__(self) -> str:
        return (
            f"{self.variable} ~ {self.kind.name}"
            f"(center={self.center_expr}, scale={self.scale_expr})"
        )

    def __repr__(self) -> str:
        return (
            f"NoiseDrawInfo(variable={self.variable!r}, kind={self.kind}, "
            f"site_id={self.site_id})"
        )


# ═══════════════════════════════════════════════════════════════════════════
# PATH CONDITION
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class PathCondition:
    """Conjunction of boolean predicates forming the guard for a path.

    Internally the condition is stored as a list of conjuncts.  The
    overall condition is their logical AND.

    Attributes:
        conjuncts: Individual boolean predicates that are all required.
    """

    conjuncts: list[TypedExpr] = field(default_factory=list)

    # -- Construction ------------------------------------------------------

    @classmethod
    def trivially_true(cls) -> PathCondition:
        """Create a condition that is always satisfied."""
        return cls(conjuncts=[])

    @classmethod
    def from_expr(cls, expr: TypedExpr) -> PathCondition:
        """Create a condition from a single boolean expression."""
        return cls(conjuncts=[expr])

    @classmethod
    def from_conjuncts(cls, exprs: Sequence[TypedExpr]) -> PathCondition:
        """Create a condition from an explicit list of conjuncts."""
        return cls(conjuncts=list(exprs))

    # -- Combination -------------------------------------------------------

    def and_(self, other: PathCondition) -> PathCondition:
        """Return the conjunction of *self* and *other*."""
        return PathCondition(conjuncts=self.conjuncts + other.conjuncts)

    def and_expr(self, expr: TypedExpr) -> PathCondition:
        """Append a single conjunct and return a new PathCondition."""
        return PathCondition(conjuncts=self.conjuncts + [expr])

    def negate_last(self) -> PathCondition:
        """Return a copy where the last conjunct is negated.

        Useful for forking into the false branch of a conditional.

        Raises:
            ValueError: If there are no conjuncts to negate.
        """
        if not self.conjuncts:
            raise ValueError("Cannot negate an empty path condition")
        negated = UnaryOp(ty=IRType.BOOL, op=UnaryOpKind.NOT, operand=self.conjuncts[-1])
        return PathCondition(conjuncts=self.conjuncts[:-1] + [negated])

    @staticmethod
    def negate_expr(expr: TypedExpr) -> TypedExpr:
        """Return the logical negation of *expr*."""
        if isinstance(expr, UnaryOp) and expr.op == UnaryOpKind.NOT:
            return expr.operand
        if isinstance(expr, Const) and expr.ty == IRType.BOOL:
            return Const.bool_(not expr.value)
        return UnaryOp(ty=IRType.BOOL, op=UnaryOpKind.NOT, operand=expr)

    # -- Simplification ----------------------------------------------------

    def simplify(self) -> PathCondition:
        """Simplify each conjunct and drop trivially-true ones."""
        simplified: list[TypedExpr] = []
        for c in self.conjuncts:
            s = c.simplify()
            if isinstance(s, Const) and s.ty == IRType.BOOL:
                if s.value is True or s.value == 1:
                    continue  # drop trivially true
                if s.value is False or s.value == 0:
                    return PathCondition(conjuncts=[Const.bool_(False)])
            simplified.append(s)
        return PathCondition(conjuncts=simplified)

    def is_trivially_true(self) -> bool:
        """Return True if the condition has no conjuncts (always true)."""
        return len(self.conjuncts) == 0

    def is_trivially_false(self) -> bool:
        """Return True if any conjunct is the constant False."""
        for c in self.conjuncts:
            if isinstance(c, Const) and c.ty == IRType.BOOL and (c.value is False or c.value == 0):
                return True
        return False

    # -- Queries -----------------------------------------------------------

    def free_vars(self) -> frozenset[str]:
        """Return all free variable names in the condition."""
        result: set[str] = set()
        for c in self.conjuncts:
            result.update(c.free_vars())
        return frozenset(result)

    def substitute(self, mapping: dict[str, TypedExpr]) -> PathCondition:
        """Apply variable substitution to every conjunct."""
        return PathCondition(
            conjuncts=[c.substitute(mapping) for c in self.conjuncts]
        )

    def as_expr(self) -> TypedExpr:
        """Convert the conjunction into a single TypedExpr (AND-tree).

        Returns Const(True) if no conjuncts are present.
        """
        if not self.conjuncts:
            return Const.bool_(True)
        result = self.conjuncts[0]
        for c in self.conjuncts[1:]:
            result = BinOp(ty=IRType.BOOL, op=BinOpKind.AND, left=result, right=c)
        return result

    def implies(self, other: PathCondition) -> bool:
        """Quick syntactic check: does *self* imply *other*?

        This is a conservative over-approximation: returns True only when
        every conjunct of *other* appears verbatim in *self*.
        """
        other_strs = {str(c) for c in other.conjuncts}
        self_strs = {str(c) for c in self.conjuncts}
        return other_strs.issubset(self_strs)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly dictionary."""
        return {"conjuncts": [str(c) for c in self.conjuncts]}

    def __len__(self) -> int:
        return len(self.conjuncts)

    def __str__(self) -> str:
        if not self.conjuncts:
            return "true"
        return " ∧ ".join(str(c) for c in self.conjuncts)

    def __repr__(self) -> str:
        return f"PathCondition({len(self.conjuncts)} conjuncts)"


# ═══════════════════════════════════════════════════════════════════════════
# SYMBOLIC PATH
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class SymbolicPath:
    """A single symbolic execution path through a MechIR program.

    Captures everything needed to build the density ratio and encode
    the path in SMT:

    * **path_condition** – the boolean guard under which this path is taken
    * **noise_draws** – noise samples encountered along the path
    * **output_expr** – the symbolic expression for the mechanism output
    * **assignments** – variable bindings accumulated during execution
    * **source_nodes** – IR node IDs visited along this path

    Attributes:
        path_id:        Unique identifier for this path.
        path_condition: Guard under which this path executes.
        noise_draws:    Noise draws encountered, in order.
        output_expr:    The mechanism's return expression on this path.
        assignments:    Map from variable name to its symbolic expression.
        source_nodes:   List of IR node IDs traversed on this path.
    """

    path_id: int = field(default_factory=_fresh_path_id)
    path_condition: PathCondition = field(default_factory=PathCondition.trivially_true)
    noise_draws: list[NoiseDrawInfo] = field(default_factory=list)
    output_expr: TypedExpr = field(default_factory=lambda: Const.zero())
    assignments: dict[str, TypedExpr] = field(default_factory=dict)
    source_nodes: list[int] = field(default_factory=list)

    # -- Feasibility -------------------------------------------------------

    def is_feasible(self) -> bool:
        """Quick feasibility check via syntactic analysis.

        Returns False if the path condition is trivially unsatisfiable
        (e.g. contains a constant-False conjunct or contradictory
        comparisons).  Returns True otherwise (conservative).
        """
        simplified = self.path_condition.simplify()
        if simplified.is_trivially_false():
            return False
        return not self._has_contradiction(simplified)

    @staticmethod
    def _has_contradiction(pc: PathCondition) -> bool:
        """Detect simple contradictions like x < 5 ∧ x >= 5."""
        seen_pos: set[str] = set()
        seen_neg: set[str] = set()
        for c in pc.conjuncts:
            s = str(c)
            if isinstance(c, UnaryOp) and c.op == UnaryOpKind.NOT:
                inner = str(c.operand)
                if inner in seen_pos:
                    return True
                seen_neg.add(inner)
            else:
                if s in seen_neg:
                    return True
                seen_pos.add(s)
        return False

    # -- Variable queries --------------------------------------------------

    def get_free_vars(self) -> frozenset[str]:
        """Return all free variable names across the entire path."""
        result: set[str] = set()
        result.update(self.path_condition.free_vars())
        result.update(self.output_expr.free_vars())
        for nd in self.noise_draws:
            result.update(nd.free_vars())
        for expr in self.assignments.values():
            result.update(expr.free_vars())
        return frozenset(result)

    def get_noise_vars(self) -> frozenset[str]:
        """Return the set of noise variable names."""
        return frozenset(nd.variable for nd in self.noise_draws)

    def get_noise_sites(self) -> list[int]:
        """Return the ordered list of noise-draw site IDs."""
        return [nd.site_id for nd in self.noise_draws]

    # -- Transformation ----------------------------------------------------

    def substitute(self, mapping: dict[str, TypedExpr]) -> SymbolicPath:
        """Return a new path with *mapping* applied to all expressions."""
        return SymbolicPath(
            path_id=self.path_id,
            path_condition=self.path_condition.substitute(mapping),
            noise_draws=[nd.with_substitution(mapping) for nd in self.noise_draws],
            output_expr=self.output_expr.substitute(mapping),
            assignments={k: v.substitute(mapping) for k, v in self.assignments.items()},
            source_nodes=list(self.source_nodes),
        )

    def clone(self) -> SymbolicPath:
        """Return a deep copy with a fresh path_id."""
        return SymbolicPath(
            path_id=_fresh_path_id(),
            path_condition=PathCondition(conjuncts=list(self.path_condition.conjuncts)),
            noise_draws=list(self.noise_draws),
            output_expr=self.output_expr,
            assignments=dict(self.assignments),
            source_nodes=list(self.source_nodes),
        )

    def merge_with(self, other: SymbolicPath) -> SymbolicPath | None:
        """Attempt to merge two paths with identical suffixes.

        Returns a merged path if the output expressions and noise draws
        are structurally identical, combining path conditions via
        disjunction.  Returns None if merging is not possible.
        """
        if str(self.output_expr) != str(other.output_expr):
            return None
        if len(self.noise_draws) != len(other.noise_draws):
            return None
        for a, b in zip(self.noise_draws, other.noise_draws):
            if a.site_id != b.site_id:
                return None
            if str(a.center_expr) != str(b.center_expr):
                return None
            if str(a.scale_expr) != str(b.scale_expr):
                return None

        combined_cond = BinOp(
            ty=IRType.BOOL,
            op=BinOpKind.OR,
            left=self.path_condition.as_expr(),
            right=other.path_condition.as_expr(),
        )
        merged_assignments = dict(self.assignments)
        merged_assignments.update(other.assignments)
        return SymbolicPath(
            path_id=_fresh_path_id(),
            path_condition=PathCondition.from_expr(combined_cond),
            noise_draws=list(self.noise_draws),
            output_expr=self.output_expr,
            assignments=merged_assignments,
            source_nodes=list(set(self.source_nodes) | set(other.source_nodes)),
        )

    def extend_condition(self, expr: TypedExpr) -> SymbolicPath:
        """Return a new path with *expr* appended to the path condition."""
        return SymbolicPath(
            path_id=self.path_id,
            path_condition=self.path_condition.and_expr(expr),
            noise_draws=list(self.noise_draws),
            output_expr=self.output_expr,
            assignments=dict(self.assignments),
            source_nodes=list(self.source_nodes),
        )

    def add_noise_draw(self, draw: NoiseDrawInfo) -> None:
        """Append a noise draw to this path (mutating)."""
        self.noise_draws.append(draw)

    def set_output(self, expr: TypedExpr) -> None:
        """Set the output expression for this path (mutating)."""
        self.output_expr = expr

    def record_assignment(self, var_name: str, expr: TypedExpr) -> None:
        """Record a variable assignment (mutating)."""
        self.assignments[var_name] = expr

    def record_node(self, node_id: int) -> None:
        """Record a visited IR node ID (mutating)."""
        self.source_nodes.append(node_id)

    # -- Serialization -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly dictionary."""
        return {
            "path_id": self.path_id,
            "path_condition": self.path_condition.to_dict(),
            "noise_draws": [nd.to_dict() for nd in self.noise_draws],
            "output_expr": str(self.output_expr),
            "assignments": {k: str(v) for k, v in self.assignments.items()},
            "source_nodes": self.source_nodes,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialise to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    # -- Pretty printing ---------------------------------------------------

    def pretty(self) -> str:
        """Return a human-readable multi-line representation."""
        lines = [f"Path #{self.path_id}"]
        lines.append(f"  Condition: {self.path_condition}")
        if self.noise_draws:
            lines.append("  Noise draws:")
            for nd in self.noise_draws:
                lines.append(f"    {nd}")
        if self.assignments:
            lines.append("  Assignments:")
            for var, expr in self.assignments.items():
                lines.append(f"    {var} = {expr}")
        lines.append(f"  Output: {self.output_expr}")
        return "\n".join(lines)

    def __str__(self) -> str:
        nd_count = len(self.noise_draws)
        return (
            f"Path(id={self.path_id}, cond={self.path_condition}, "
            f"draws={nd_count}, out={self.output_expr})"
        )

    def __repr__(self) -> str:
        return f"SymbolicPath(id={self.path_id})"


# ═══════════════════════════════════════════════════════════════════════════
# PATH SET
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class PathSet:
    """Collection of symbolic paths with query and transformation operations.

    Attributes:
        paths:    List of symbolic paths.
        metadata: Free-form metadata (e.g. enumeration statistics).
    """

    paths: list[SymbolicPath] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- Basic operations --------------------------------------------------

    def add(self, path: SymbolicPath) -> None:
        """Add a path to the set."""
        self.paths.append(path)

    def add_all(self, paths: Sequence[SymbolicPath]) -> None:
        """Add multiple paths to the set."""
        self.paths.extend(paths)

    def size(self) -> int:
        """Return the number of paths."""
        return len(self.paths)

    def is_empty(self) -> bool:
        """Return True if the set is empty."""
        return len(self.paths) == 0

    def get(self, path_id: int) -> SymbolicPath | None:
        """Find a path by its ID, or return None."""
        for p in self.paths:
            if p.path_id == path_id:
                return p
        return None

    # -- Iteration ---------------------------------------------------------

    def __iter__(self) -> Iterator[SymbolicPath]:
        return iter(self.paths)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> SymbolicPath:
        return self.paths[idx]

    # -- Filtering and partitioning ----------------------------------------

    def filter(self, predicate: Callable[[SymbolicPath], bool]) -> PathSet:
        """Return a new PathSet containing only paths satisfying *predicate*."""
        return PathSet(paths=[p for p in self.paths if predicate(p)])

    def feasible_only(self) -> PathSet:
        """Return a new PathSet containing only feasible paths."""
        return self.filter(lambda p: p.is_feasible())

    def with_noise(self) -> PathSet:
        """Return paths that contain at least one noise draw."""
        return self.filter(lambda p: len(p.noise_draws) > 0)

    def without_noise(self) -> PathSet:
        """Return paths that contain no noise draws."""
        return self.filter(lambda p: len(p.noise_draws) == 0)

    def partition(
        self, key: Callable[[SymbolicPath], str]
    ) -> dict[str, PathSet]:
        """Partition paths by the given key function.

        Returns a mapping from key values to PathSets.
        """
        groups: dict[str, list[SymbolicPath]] = {}
        for p in self.paths:
            k = key(p)
            groups.setdefault(k, []).append(p)
        return {k: PathSet(paths=v) for k, v in groups.items()}

    def partition_by_noise_pattern(self) -> dict[str, PathSet]:
        """Partition paths by their noise draw site patterns.

        Two paths share a pattern if they draw noise at the same sites
        in the same order.
        """
        def noise_key(p: SymbolicPath) -> str:
            return ",".join(str(nd.site_id) for nd in p.noise_draws)
        return self.partition(noise_key)

    def group_by_output(self) -> dict[str, PathSet]:
        """Group paths by their output expression (string equality)."""
        return self.partition(lambda p: str(p.output_expr))

    # -- Transformation ----------------------------------------------------

    def substitute_all(self, mapping: dict[str, TypedExpr]) -> PathSet:
        """Apply *mapping* to every path and return a new PathSet."""
        return PathSet(paths=[p.substitute(mapping) for p in self.paths])

    def merge_compatible(self) -> PathSet:
        """Attempt to merge paths with identical suffixes.

        Returns a new PathSet where mergeable paths have been combined.
        This can reduce the number of paths without losing precision.
        """
        merged: list[SymbolicPath] = []
        used: set[int] = set()

        for i, pi in enumerate(self.paths):
            if pi.path_id in used:
                continue
            current = pi
            for j in range(i + 1, len(self.paths)):
                pj = self.paths[j]
                if pj.path_id in used:
                    continue
                result = current.merge_with(pj)
                if result is not None:
                    current = result
                    used.add(pj.path_id)
            merged.append(current)
            used.add(pi.path_id)

        return PathSet(paths=merged, metadata=dict(self.metadata))

    # -- Statistics --------------------------------------------------------

    def all_noise_sites(self) -> set[int]:
        """Return the set of all noise-draw site IDs across all paths."""
        sites: set[int] = set()
        for p in self.paths:
            for nd in p.noise_draws:
                sites.add(nd.site_id)
        return sites

    def max_noise_draws(self) -> int:
        """Return the maximum number of noise draws on any single path."""
        if not self.paths:
            return 0
        return max(len(p.noise_draws) for p in self.paths)

    def total_noise_draws(self) -> int:
        """Return the total number of noise draws across all paths."""
        return sum(len(p.noise_draws) for p in self.paths)

    def all_free_vars(self) -> frozenset[str]:
        """Return all free variables across all paths."""
        result: set[str] = set()
        for p in self.paths:
            result.update(p.get_free_vars())
        return frozenset(result)

    def summary(self) -> dict[str, Any]:
        """Return a summary of the path set for diagnostics."""
        return {
            "total_paths": self.size(),
            "feasible_paths": self.feasible_only().size(),
            "max_noise_draws": self.max_noise_draws(),
            "total_noise_draws": self.total_noise_draws(),
            "noise_sites": len(self.all_noise_sites()),
            "free_vars": len(self.all_free_vars()),
            "metadata": self.metadata,
        }

    # -- Serialization -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly dictionary."""
        return {
            "paths": [p.to_dict() for p in self.paths],
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialise to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def __str__(self) -> str:
        return f"PathSet({self.size()} paths)"

    def __repr__(self) -> str:
        return f"PathSet(size={self.size()}, meta={self.metadata})"
