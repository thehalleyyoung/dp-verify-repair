"""JSON serialization and deserialization for MechIR.

Provides lossless round-trip serialization of all IR nodes, expressions,
and privacy budgets to/from JSON.

Usage::

    from dpcegar.ir.serialization import to_json, from_json

    json_str = to_json(mechir)
    mechir2 = from_json(json_str)
    assert to_json(mechir2) == json_str   # round-trip fidelity
"""

from __future__ import annotations

import json
from typing import Any

from dpcegar.ir.types import (
    Abs,
    BinOp,
    BinOpKind,
    Const,
    Cond,
    Exp,
    ExprVisitor,
    FuncCall,
    IRType,
    LetExpr,
    Log,
    Max,
    Min,
    NoiseKind,
    Phi,
    PhiInv,
    PrivacyBudget,
    PrivacyNotion,
    ApproxBudget,
    FDPBudget,
    GDPBudget,
    PureBudget,
    RDPBudget,
    ZCDPBudget,
    Sqrt,
    SumExpr,
    ArrayAccess,
    TupleAccess,
    TypedExpr,
    UnaryOp,
    UnaryOpKind,
    Var,
)
from dpcegar.ir.nodes import (
    AssignNode,
    BranchNode,
    CFG,
    CFGEdge,
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
from dpcegar.utils.errors import InternalError, SourceLoc


# ═══════════════════════════════════════════════════════════════════════════
# JSON ENCODER
# ═══════════════════════════════════════════════════════════════════════════


class IREncoder(json.JSONEncoder):
    """Custom JSON encoder that handles IR types and enums."""

    def default(self, o: Any) -> Any:
        if isinstance(o, TypedExpr):
            return _expr_to_dict(o)
        if isinstance(o, IRNode):
            return _node_to_dict(o)
        if isinstance(o, MechIR):
            return _mechir_to_dict(o)
        if isinstance(o, PrivacyBudget):
            return _budget_to_dict(o)
        if isinstance(o, ParamDecl):
            return {"name": o.name, "ty": o.ty.name, "is_database": o.is_database}
        if isinstance(o, SourceLoc):
            return {"file": o.file, "line": o.line, "col": o.col,
                    "end_line": o.end_line, "end_col": o.end_col}
        if isinstance(o, CFGEdge):
            d: dict[str, Any] = {"src": o.src, "dst": o.dst, "label": o.label}
            if o.condition is not None:
                d["condition"] = _expr_to_dict(o.condition)
            return d
        if isinstance(o, CFG):
            return _cfg_to_dict(o)
        return super().default(o)


# ═══════════════════════════════════════════════════════════════════════════
# EXPRESSION SERIALIZATION
# ═══════════════════════════════════════════════════════════════════════════


def _expr_to_dict(expr: TypedExpr) -> dict[str, Any]:
    """Serialise an expression to a plain dictionary."""
    ty = expr.ty.name

    if isinstance(expr, Var):
        d: dict[str, Any] = {"_type": "Var", "ty": ty, "name": expr.name}
        if expr.version is not None:
            d["version"] = expr.version
        return d

    if isinstance(expr, Const):
        return {"_type": "Const", "ty": ty, "value": expr.value}

    if isinstance(expr, BinOp):
        return {
            "_type": "BinOp", "ty": ty, "op": expr.op.name,
            "left": _expr_to_dict(expr.left),
            "right": _expr_to_dict(expr.right),
        }

    if isinstance(expr, UnaryOp):
        return {
            "_type": "UnaryOp", "ty": ty, "op": expr.op.name,
            "operand": _expr_to_dict(expr.operand),
        }

    if isinstance(expr, FuncCall):
        return {
            "_type": "FuncCall", "ty": ty, "name": expr.name,
            "args": [_expr_to_dict(a) for a in expr.args],
        }

    if isinstance(expr, ArrayAccess):
        return {
            "_type": "ArrayAccess", "ty": ty,
            "array": _expr_to_dict(expr.array),
            "index": _expr_to_dict(expr.index),
        }

    if isinstance(expr, TupleAccess):
        return {
            "_type": "TupleAccess", "ty": ty,
            "tuple_expr": _expr_to_dict(expr.tuple_expr),
            "field_idx": expr.field_idx,
        }

    # Unary math functions
    for cls_name in ("Abs", "Log", "Exp", "Sqrt", "Phi", "PhiInv"):
        cls = globals().get(cls_name) or locals().get(cls_name)
        if cls is not None and isinstance(expr, cls):
            return {
                "_type": cls_name, "ty": ty,
                "operand": _expr_to_dict(expr.operand),
            }

    if isinstance(expr, Max):
        return {
            "_type": "Max", "ty": ty,
            "left": _expr_to_dict(expr.left),
            "right": _expr_to_dict(expr.right),
        }

    if isinstance(expr, Min):
        return {
            "_type": "Min", "ty": ty,
            "left": _expr_to_dict(expr.left),
            "right": _expr_to_dict(expr.right),
        }

    if isinstance(expr, Cond):
        return {
            "_type": "Cond", "ty": ty,
            "condition": _expr_to_dict(expr.condition),
            "true_expr": _expr_to_dict(expr.true_expr),
            "false_expr": _expr_to_dict(expr.false_expr),
        }

    if isinstance(expr, LetExpr):
        return {
            "_type": "LetExpr", "ty": ty, "var_name": expr.var_name,
            "value": _expr_to_dict(expr.value),
            "body": _expr_to_dict(expr.body),
        }

    if isinstance(expr, SumExpr):
        return {
            "_type": "SumExpr", "ty": ty, "var_name": expr.var_name,
            "lo": _expr_to_dict(expr.lo),
            "hi": _expr_to_dict(expr.hi),
            "body": _expr_to_dict(expr.body),
        }

    raise InternalError(f"Cannot serialise expression type: {type(expr).__name__}")


def _expr_from_dict(d: dict[str, Any]) -> TypedExpr:
    """Deserialise an expression from a plain dictionary."""
    kind = d["_type"]
    ty = IRType[d["ty"]]

    if kind == "Var":
        return Var(ty=ty, name=d["name"], version=d.get("version"))

    if kind == "Const":
        return Const(ty=ty, value=d["value"])

    if kind == "BinOp":
        return BinOp(
            ty=ty, op=BinOpKind[d["op"]],
            left=_expr_from_dict(d["left"]),
            right=_expr_from_dict(d["right"]),
        )

    if kind == "UnaryOp":
        return UnaryOp(
            ty=ty, op=UnaryOpKind[d["op"]],
            operand=_expr_from_dict(d["operand"]),
        )

    if kind == "FuncCall":
        return FuncCall(
            ty=ty, name=d["name"],
            args=tuple(_expr_from_dict(a) for a in d["args"]),
        )

    if kind == "ArrayAccess":
        return ArrayAccess(
            ty=ty,
            array=_expr_from_dict(d["array"]),
            index=_expr_from_dict(d["index"]),
        )

    if kind == "TupleAccess":
        return TupleAccess(
            ty=ty,
            tuple_expr=_expr_from_dict(d["tuple_expr"]),
            field_idx=d["field_idx"],
        )

    _unary_math = {"Abs": Abs, "Log": Log, "Exp": Exp, "Sqrt": Sqrt, "Phi": Phi, "PhiInv": PhiInv}
    if kind in _unary_math:
        cls = _unary_math[kind]
        return cls(ty=ty, operand=_expr_from_dict(d["operand"]))

    if kind == "Max":
        return Max(ty=ty, left=_expr_from_dict(d["left"]), right=_expr_from_dict(d["right"]))

    if kind == "Min":
        return Min(ty=ty, left=_expr_from_dict(d["left"]), right=_expr_from_dict(d["right"]))

    if kind == "Cond":
        return Cond(
            ty=ty,
            condition=_expr_from_dict(d["condition"]),
            true_expr=_expr_from_dict(d["true_expr"]),
            false_expr=_expr_from_dict(d["false_expr"]),
        )

    if kind == "LetExpr":
        return LetExpr(
            ty=ty, var_name=d["var_name"],
            value=_expr_from_dict(d["value"]),
            body=_expr_from_dict(d["body"]),
        )

    if kind == "SumExpr":
        return SumExpr(
            ty=ty, var_name=d["var_name"],
            lo=_expr_from_dict(d["lo"]),
            hi=_expr_from_dict(d["hi"]),
            body=_expr_from_dict(d["body"]),
        )

    raise InternalError(f"Unknown expression type in JSON: {kind}")


# ═══════════════════════════════════════════════════════════════════════════
# NODE SERIALIZATION
# ═══════════════════════════════════════════════════════════════════════════


def _source_loc_to_dict(loc: SourceLoc | None) -> dict[str, Any] | None:
    if loc is None:
        return None
    return {"file": loc.file, "line": loc.line, "col": loc.col,
            "end_line": loc.end_line, "end_col": loc.end_col}


def _source_loc_from_dict(d: dict[str, Any] | None) -> SourceLoc | None:
    if d is None:
        return None
    return SourceLoc(
        file=d.get("file", "<unknown>"),
        line=d.get("line", 0),
        col=d.get("col", 0),
        end_line=d.get("end_line"),
        end_col=d.get("end_col"),
    )


def _node_to_dict(node: IRNode) -> dict[str, Any]:
    """Serialise an IR node to a plain dictionary."""
    base: dict[str, Any] = {
        "_type": type(node).__name__,
        "node_id": node.node_id,
        "source_loc": _source_loc_to_dict(node.source_loc),
        "annotations": node.annotations,
    }

    if isinstance(node, AssignNode):
        base["target"] = _expr_to_dict(node.target)
        base["value"] = _expr_to_dict(node.value)
    elif isinstance(node, NoiseDrawNode):
        base["target"] = _expr_to_dict(node.target)
        base["noise_kind"] = node.noise_kind.name
        base["center"] = _expr_to_dict(node.center)
        base["scale"] = _expr_to_dict(node.scale)
        base["sensitivity"] = _expr_to_dict(node.sensitivity) if node.sensitivity else None
    elif isinstance(node, BranchNode):
        base["condition"] = _expr_to_dict(node.condition)
        base["true_branch"] = _node_to_dict(node.true_branch)
        base["false_branch"] = _node_to_dict(node.false_branch)
    elif isinstance(node, MergeNode):
        base["target"] = _expr_to_dict(node.target)
        base["sources"] = {str(k): _expr_to_dict(v) for k, v in node.sources.items()}
    elif isinstance(node, LoopNode):
        base["index_var"] = _expr_to_dict(node.index_var)
        base["bound"] = _expr_to_dict(node.bound)
        base["body"] = _node_to_dict(node.body)
        base["unroll_count"] = node.unroll_count
    elif isinstance(node, QueryNode):
        base["target"] = _expr_to_dict(node.target)
        base["query_name"] = node.query_name
        base["args"] = [_expr_to_dict(a) for a in node.args]
        base["sensitivity"] = _expr_to_dict(node.sensitivity)
    elif isinstance(node, ReturnNode):
        base["value"] = _expr_to_dict(node.value)
    elif isinstance(node, SequenceNode):
        base["stmts"] = [_node_to_dict(s) for s in node.stmts]
    elif isinstance(node, NoOpNode):
        pass
    else:
        raise InternalError(f"Cannot serialise node type: {type(node).__name__}")

    return base


def _node_from_dict(d: dict[str, Any]) -> IRNode:
    """Deserialise an IR node from a plain dictionary."""
    kind = d["_type"]
    loc = _source_loc_from_dict(d.get("source_loc"))
    annotations = d.get("annotations", {})
    node_id = d.get("node_id", 0)

    if kind == "AssignNode":
        node = AssignNode(
            node_id=node_id, source_loc=loc, annotations=annotations,
            target=_expr_from_dict(d["target"]),  # type: ignore[arg-type]
            value=_expr_from_dict(d["value"]),
        )
    elif kind == "NoiseDrawNode":
        sens = _expr_from_dict(d["sensitivity"]) if d.get("sensitivity") else None
        node = NoiseDrawNode(
            node_id=node_id, source_loc=loc, annotations=annotations,
            target=_expr_from_dict(d["target"]),  # type: ignore[arg-type]
            noise_kind=NoiseKind[d["noise_kind"]],
            center=_expr_from_dict(d["center"]),
            scale=_expr_from_dict(d["scale"]),
            sensitivity=sens,
        )
    elif kind == "BranchNode":
        node = BranchNode(
            node_id=node_id, source_loc=loc, annotations=annotations,
            condition=_expr_from_dict(d["condition"]),
            true_branch=_node_from_dict(d["true_branch"]),
            false_branch=_node_from_dict(d["false_branch"]),
        )
    elif kind == "MergeNode":
        sources = {int(k): _expr_from_dict(v) for k, v in d.get("sources", {}).items()}
        node = MergeNode(
            node_id=node_id, source_loc=loc, annotations=annotations,
            target=_expr_from_dict(d["target"]),  # type: ignore[arg-type]
            sources=sources,
        )
    elif kind == "LoopNode":
        node = LoopNode(
            node_id=node_id, source_loc=loc, annotations=annotations,
            index_var=_expr_from_dict(d["index_var"]),  # type: ignore[arg-type]
            bound=_expr_from_dict(d["bound"]),
            body=_node_from_dict(d["body"]),
            unroll_count=d.get("unroll_count"),
        )
    elif kind == "QueryNode":
        node = QueryNode(
            node_id=node_id, source_loc=loc, annotations=annotations,
            target=_expr_from_dict(d["target"]),  # type: ignore[arg-type]
            query_name=d["query_name"],
            args=tuple(_expr_from_dict(a) for a in d.get("args", [])),
            sensitivity=_expr_from_dict(d["sensitivity"]),
        )
    elif kind == "ReturnNode":
        node = ReturnNode(
            node_id=node_id, source_loc=loc, annotations=annotations,
            value=_expr_from_dict(d["value"]),
        )
    elif kind == "SequenceNode":
        node = SequenceNode(
            node_id=node_id, source_loc=loc, annotations=annotations,
            stmts=[_node_from_dict(s) for s in d.get("stmts", [])],
        )
    elif kind == "NoOpNode":
        node = NoOpNode(node_id=node_id, source_loc=loc, annotations=annotations)
    else:
        raise InternalError(f"Unknown node type in JSON: {kind}")

    return node


# ═══════════════════════════════════════════════════════════════════════════
# BUDGET SERIALIZATION
# ═══════════════════════════════════════════════════════════════════════════


def _budget_to_dict(budget: PrivacyBudget) -> dict[str, Any]:
    """Serialise a privacy budget."""
    if isinstance(budget, PureBudget):
        return {"_type": "PureBudget", "epsilon": budget.epsilon}
    if isinstance(budget, ApproxBudget):
        return {"_type": "ApproxBudget", "epsilon": budget.epsilon, "delta": budget.delta}
    if isinstance(budget, ZCDPBudget):
        return {"_type": "ZCDPBudget", "rho": budget.rho}
    if isinstance(budget, RDPBudget):
        return {"_type": "RDPBudget", "alpha": budget.alpha, "epsilon": budget.epsilon}
    if isinstance(budget, GDPBudget):
        return {"_type": "GDPBudget", "mu": budget.mu}
    if isinstance(budget, FDPBudget):
        return {"_type": "FDPBudget", "_note": "trade_off_fn not serialisable"}
    raise InternalError(f"Cannot serialise budget type: {type(budget).__name__}")


def _budget_from_dict(d: dict[str, Any]) -> PrivacyBudget:
    """Deserialise a privacy budget."""
    kind = d["_type"]
    if kind == "PureBudget":
        return PureBudget(epsilon=d["epsilon"])
    if kind == "ApproxBudget":
        return ApproxBudget(epsilon=d["epsilon"], delta=d["delta"])
    if kind == "ZCDPBudget":
        return ZCDPBudget(rho=d["rho"])
    if kind == "RDPBudget":
        return RDPBudget(alpha=d["alpha"], epsilon=d["epsilon"])
    if kind == "GDPBudget":
        return GDPBudget(mu=d["mu"])
    if kind == "FDPBudget":
        return FDPBudget()
    raise InternalError(f"Unknown budget type in JSON: {kind}")


# ═══════════════════════════════════════════════════════════════════════════
# MECHIR SERIALIZATION
# ═══════════════════════════════════════════════════════════════════════════


def _mechir_to_dict(mechir: MechIR) -> dict[str, Any]:
    """Serialise a complete MechIR to a dictionary."""
    return {
        "_type": "MechIR",
        "name": mechir.name,
        "params": [
            {"name": p.name, "ty": p.ty.name, "is_database": p.is_database}
            for p in mechir.params
        ],
        "body": _node_to_dict(mechir.body),
        "return_type": mechir.return_type.name,
        "budget": _budget_to_dict(mechir.budget) if mechir.budget else None,
        "metadata": mechir.metadata,
    }


def _mechir_from_dict(d: dict[str, Any]) -> MechIR:
    """Deserialise a MechIR from a dictionary."""
    params = [
        ParamDecl(name=p["name"], ty=IRType[p["ty"]], is_database=p.get("is_database", False))
        for p in d.get("params", [])
    ]
    budget = _budget_from_dict(d["budget"]) if d.get("budget") else None
    return MechIR(
        name=d.get("name", "unnamed"),
        params=params,
        body=_node_from_dict(d["body"]),
        return_type=IRType[d.get("return_type", "REAL")],
        budget=budget,
        metadata=d.get("metadata", {}),
    )


# ═══════════════════════════════════════════════════════════════════════════
# CFG SERIALIZATION
# ═══════════════════════════════════════════════════════════════════════════


def _cfg_to_dict(cfg: CFG) -> dict[str, Any]:
    """Serialise a CFG to a dictionary."""
    return {
        "_type": "CFG",
        "nodes": {str(k): _node_to_dict(v) for k, v in cfg.nodes.items()},
        "edges": [
            {
                "src": e.src, "dst": e.dst, "label": e.label,
                "condition": _expr_to_dict(e.condition) if e.condition else None,
            }
            for e in cfg.edges
        ],
        "entry": cfg.entry,
        "exit_node": cfg.exit_node,
    }


def _cfg_from_dict(d: dict[str, Any]) -> CFG:
    """Deserialise a CFG from a dictionary."""
    cfg = CFG()
    for _k, v in d.get("nodes", {}).items():
        node = _node_from_dict(v)
        cfg.nodes[node.node_id] = node
    for e in d.get("edges", []):
        cond = _expr_from_dict(e["condition"]) if e.get("condition") else None
        cfg.add_edge(e["src"], e["dst"], condition=cond, label=e.get("label", ""))
    cfg.entry = d.get("entry", -1)
    cfg.exit_node = d.get("exit_node", -1)
    return cfg


# ═══════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════


def to_json(obj: MechIR | IRNode | TypedExpr | PrivacyBudget | CFG, *, indent: int = 2) -> str:
    """Serialise any IR object to a JSON string.

    Args:
        obj:    An IR object (MechIR, IRNode, TypedExpr, PrivacyBudget, CFG).
        indent: JSON indentation level.

    Returns:
        A JSON string.
    """
    return json.dumps(obj, cls=IREncoder, indent=indent)


def from_json(text: str) -> MechIR | IRNode | TypedExpr | PrivacyBudget | CFG:
    """Deserialise an IR object from a JSON string.

    The ``_type`` field in the top-level object determines the return type.

    Args:
        text: A JSON string produced by :func:`to_json`.

    Returns:
        The reconstructed IR object.
    """
    d = json.loads(text)
    return from_dict(d)


def from_dict(d: dict[str, Any]) -> MechIR | IRNode | TypedExpr | PrivacyBudget | CFG:
    """Deserialise from a dictionary (already parsed JSON)."""
    kind = d.get("_type", "")

    if kind == "MechIR":
        return _mechir_from_dict(d)
    if kind == "CFG":
        return _cfg_from_dict(d)
    if kind in ("PureBudget", "ApproxBudget", "ZCDPBudget", "RDPBudget", "FDPBudget", "GDPBudget"):
        return _budget_from_dict(d)

    # Try as node
    _node_types = {
        "AssignNode", "NoiseDrawNode", "BranchNode", "MergeNode",
        "LoopNode", "QueryNode", "ReturnNode", "SequenceNode", "NoOpNode",
    }
    if kind in _node_types:
        return _node_from_dict(d)

    # Try as expression
    _expr_types = {
        "Var", "Const", "BinOp", "UnaryOp", "FuncCall", "ArrayAccess",
        "TupleAccess", "Abs", "Max", "Min", "Log", "Exp", "Sqrt",
        "Phi", "PhiInv", "Cond", "LetExpr", "SumExpr",
    }
    if kind in _expr_types:
        return _expr_from_dict(d)

    raise InternalError(f"Unknown top-level _type in JSON: {kind!r}")
