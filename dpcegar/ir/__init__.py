"""Intermediate representation for differential privacy mechanisms.

This sub-package defines the core IR used throughout DP-CEGAR:
  - types.py   : Type system, expression nodes, privacy budgets
  - nodes.py   : MechIR CFG nodes (assign, noise, branch, loop, ...)
  - visitors.py: Visitor and transformer infrastructure
  - serialization.py: JSON round-trip serialization
"""

from dpcegar.ir.types import (
    IRType,
    NoiseKind,
    PrivacyNotion,
    TypedExpr,
    Var,
    Const,
    BinOp,
    UnaryOp,
    PrivacyBudget,
    PureBudget,
    ApproxBudget,
)
from dpcegar.ir.nodes import (
    IRNode,
    AssignNode,
    NoiseDrawNode,
    BranchNode,
    MergeNode,
    LoopNode,
    QueryNode,
    ReturnNode,
    SequenceNode,
    NoOpNode,
    MechIR,
    CFG,
)

__all__ = [
    "IRType", "NoiseKind", "PrivacyNotion",
    "TypedExpr", "Var", "Const", "BinOp", "UnaryOp",
    "PrivacyBudget", "PureBudget", "ApproxBudget",
    "IRNode", "AssignNode", "NoiseDrawNode", "BranchNode",
    "MergeNode", "LoopNode", "QueryNode", "ReturnNode",
    "SequenceNode", "NoOpNode", "MechIR", "CFG",
]
