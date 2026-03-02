"""Shared fixtures for DP-CEGAR unit tests."""

from __future__ import annotations

import math
import random
from typing import Any

import pytest

from dpcegar.ir.types import (
    Abs,
    ApproxBudget,
    BinOp,
    BinOpKind,
    Cond,
    Const,
    Exp,
    FDPBudget,
    FuncCall,
    GDPBudget,
    IRType,
    LetExpr,
    Log,
    Max,
    Min,
    NoiseKind,
    Phi,
    PhiInv,
    PrivacyNotion,
    PureBudget,
    RDPBudget,
    Sqrt,
    SumExpr,
    TypedExpr,
    UnaryOp,
    UnaryOpKind,
    Var,
    ZCDPBudget,
)
from dpcegar.ir.nodes import (
    AssignNode,
    BranchNode,
    CFG,
    CFGBuilder,
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
from dpcegar.paths.symbolic_path import (
    NoiseDrawInfo,
    PathCondition,
    PathSet,
    SymbolicPath,
)
from dpcegar.density.ratio_builder import (
    DensityRatioBuilder,
    DensityRatioExpr,
    DensityRatioResult,
)
from dpcegar.utils.errors import SourceLoc


# ═══════════════════════════════════════════════════════════════════════════
# Expression fixtures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def var_x() -> Var:
    return Var(ty=IRType.REAL, name="x")


@pytest.fixture
def var_y() -> Var:
    return Var(ty=IRType.REAL, name="y")


@pytest.fixture
def var_z() -> Var:
    return Var(ty=IRType.REAL, name="z")


@pytest.fixture
def var_i() -> Var:
    return Var(ty=IRType.INT, name="i")


@pytest.fixture
def var_b() -> Var:
    return Var(ty=IRType.BOOL, name="b")


@pytest.fixture
def const_zero() -> Const:
    return Const.zero()


@pytest.fixture
def const_one() -> Const:
    return Const.one()


@pytest.fixture
def const_pi() -> Const:
    return Const.real(math.pi)


@pytest.fixture
def const_int_42() -> Const:
    return Const.int_(42)


@pytest.fixture
def const_true() -> Const:
    return Const.bool_(True)


@pytest.fixture
def const_false() -> Const:
    return Const.bool_(False)


@pytest.fixture
def add_expr(var_x: Var, var_y: Var) -> BinOp:
    return BinOp(ty=IRType.REAL, op=BinOpKind.ADD, left=var_x, right=var_y)


@pytest.fixture
def mul_expr(var_x: Var, const_pi: Const) -> BinOp:
    return BinOp(ty=IRType.REAL, op=BinOpKind.MUL, left=var_x, right=const_pi)


@pytest.fixture
def comparison_expr(var_x: Var, const_zero: Const) -> BinOp:
    return BinOp(ty=IRType.BOOL, op=BinOpKind.GT, left=var_x, right=const_zero)


# ═══════════════════════════════════════════════════════════════════════════
# Node fixtures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def assign_node(var_x: Var, const_pi: Const) -> AssignNode:
    return AssignNode(target=var_x, value=const_pi)


@pytest.fixture
def noise_draw_node() -> NoiseDrawNode:
    return NoiseDrawNode(
        target=Var(ty=IRType.REAL, name="eta"),
        noise_kind=NoiseKind.LAPLACE,
        center=Var(ty=IRType.REAL, name="q"),
        scale=Const.real(1.0),
    )


@pytest.fixture
def branch_node(comparison_expr: BinOp) -> BranchNode:
    return BranchNode(
        condition=comparison_expr,
        true_branch=AssignNode(
            target=Var(ty=IRType.REAL, name="r"),
            value=Const.real(1.0),
        ),
        false_branch=AssignNode(
            target=Var(ty=IRType.REAL, name="r"),
            value=Const.real(0.0),
        ),
    )


@pytest.fixture
def return_node(var_x: Var) -> ReturnNode:
    return ReturnNode(value=var_x)


@pytest.fixture
def sequence_node(assign_node: AssignNode, return_node: ReturnNode) -> SequenceNode:
    return SequenceNode(stmts=[assign_node, return_node])


@pytest.fixture
def loop_node() -> LoopNode:
    return LoopNode(
        index_var=Var(ty=IRType.INT, name="i"),
        bound=Const.int_(5),
        body=AssignNode(
            target=Var(ty=IRType.REAL, name="s"),
            value=BinOp(
                ty=IRType.REAL,
                op=BinOpKind.ADD,
                left=Var(ty=IRType.REAL, name="s"),
                right=Const.real(1.0),
            ),
        ),
    )


@pytest.fixture
def query_node() -> QueryNode:
    return QueryNode(
        target=Var(ty=IRType.REAL, name="q_result"),
        query_name="count",
        args=(Var(ty=IRType.REAL, name="db"),),
        sensitivity=Const.real(1.0),
    )


@pytest.fixture
def source_loc() -> SourceLoc:
    return SourceLoc(file="test.py", line=10, col=5)


# ═══════════════════════════════════════════════════════════════════════════
# MechIR fixtures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def simple_mechir() -> MechIR:
    """A simple Laplace mechanism: return query(db) + Lap(1.0)."""
    q_node = QueryNode(
        target=Var(ty=IRType.REAL, name="q"),
        query_name="count",
        args=(Var(ty=IRType.REAL, name="db"),),
        sensitivity=Const.real(1.0),
    )
    noise_node = NoiseDrawNode(
        target=Var(ty=IRType.REAL, name="eta"),
        noise_kind=NoiseKind.LAPLACE,
        center=Var(ty=IRType.REAL, name="q"),
        scale=Const.real(1.0),
    )
    ret = ReturnNode(value=Var(ty=IRType.REAL, name="eta"))
    body = SequenceNode(stmts=[q_node, noise_node, ret])

    return MechIR(
        name="laplace_mech",
        params=[
            ParamDecl(name="db", ty=IRType.ARRAY, is_database=True),
            ParamDecl(name="epsilon", ty=IRType.REAL),
        ],
        body=body,
        return_type=IRType.REAL,
        budget=PureBudget(epsilon=1.0),
    )


@pytest.fixture
def branching_mechir() -> MechIR:
    """A mechanism with a branch."""
    cond = BinOp(
        ty=IRType.BOOL,
        op=BinOpKind.GT,
        left=Var(ty=IRType.REAL, name="x"),
        right=Const.real(0.0),
    )
    true_br = AssignNode(
        target=Var(ty=IRType.REAL, name="r"),
        value=Const.real(1.0),
    )
    false_br = AssignNode(
        target=Var(ty=IRType.REAL, name="r"),
        value=Const.real(0.0),
    )
    branch = BranchNode(condition=cond, true_branch=true_br, false_branch=false_br)
    ret = ReturnNode(value=Var(ty=IRType.REAL, name="r"))
    body = SequenceNode(stmts=[branch, ret])

    return MechIR(
        name="branching_mech",
        params=[ParamDecl(name="x", ty=IRType.REAL)],
        body=body,
        return_type=IRType.REAL,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Budget fixtures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def pure_budget() -> PureBudget:
    return PureBudget(epsilon=1.0)


@pytest.fixture
def approx_budget() -> ApproxBudget:
    return ApproxBudget(epsilon=1.0, delta=1e-5)


@pytest.fixture
def zcdp_budget() -> ZCDPBudget:
    return ZCDPBudget(rho=0.5)


@pytest.fixture
def rdp_budget() -> RDPBudget:
    return RDPBudget(alpha=2.0, epsilon=1.0)


@pytest.fixture
def gdp_budget() -> GDPBudget:
    return GDPBudget(mu=1.0)


@pytest.fixture
def seeded_rng() -> random.Random:
    return random.Random(42)


# ═══════════════════════════════════════════════════════════════════════════
# Path / Density fixtures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def noise_draw_info_laplace() -> NoiseDrawInfo:
    return NoiseDrawInfo(
        variable="eta",
        kind=NoiseKind.LAPLACE,
        center_expr=Var(ty=IRType.REAL, name="q"),
        scale_expr=Const.real(1.0),
        site_id=100,
    )


@pytest.fixture
def noise_draw_info_gaussian() -> NoiseDrawInfo:
    return NoiseDrawInfo(
        variable="eta_g",
        kind=NoiseKind.GAUSSIAN,
        center_expr=Var(ty=IRType.REAL, name="q"),
        scale_expr=Const.real(1.0),
        site_id=101,
    )


@pytest.fixture
def trivial_path_condition() -> PathCondition:
    return PathCondition.trivially_true()


@pytest.fixture
def simple_path_condition() -> PathCondition:
    cond = BinOp(
        ty=IRType.BOOL,
        op=BinOpKind.GT,
        left=Var(ty=IRType.REAL, name="x"),
        right=Const.real(0.0),
    )
    return PathCondition.from_expr(cond)


@pytest.fixture
def symbolic_path_laplace(noise_draw_info_laplace: NoiseDrawInfo) -> SymbolicPath:
    return SymbolicPath(
        path_condition=PathCondition.trivially_true(),
        noise_draws=[noise_draw_info_laplace],
        output_expr=Var(ty=IRType.REAL, name="eta"),
    )


@pytest.fixture
def symbolic_path_gaussian(noise_draw_info_gaussian: NoiseDrawInfo) -> SymbolicPath:
    return SymbolicPath(
        path_condition=PathCondition.trivially_true(),
        noise_draws=[noise_draw_info_gaussian],
        output_expr=Var(ty=IRType.REAL, name="eta_g"),
    )


@pytest.fixture
def simple_path_set(symbolic_path_laplace: SymbolicPath) -> PathSet:
    ps = PathSet()
    ps.add(symbolic_path_laplace)
    return ps


@pytest.fixture
def two_path_set(
    symbolic_path_laplace: SymbolicPath,
    symbolic_path_gaussian: SymbolicPath,
) -> PathSet:
    ps = PathSet()
    ps.add(symbolic_path_laplace)
    ps.add(symbolic_path_gaussian)
    return ps


@pytest.fixture
def simple_density_ratio() -> DensityRatioExpr:
    log_ratio = BinOp(
        ty=IRType.REAL,
        op=BinOpKind.MUL,
        left=Const.real(1.0),
        right=Var(ty=IRType.REAL, name="delta_q"),
    )
    return DensityRatioExpr(
        log_ratio=log_ratio,
        path_condition_d=PathCondition.trivially_true(),
        path_condition_d_prime=PathCondition.trivially_true(),
        path_id_d=0,
        path_id_d_prime=0,
    )


@pytest.fixture
def simple_density_result(simple_density_ratio: DensityRatioExpr) -> DensityRatioResult:
    return DensityRatioResult(
        ratios=[simple_density_ratio],
        same_path=[simple_density_ratio],
    )


# ═══════════════════════════════════════════════════════════════════════════
# Gaussian mechanism MechIR
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def gaussian_mechir() -> MechIR:
    """Gaussian mechanism: return query(db) + N(0, sigma^2)."""
    q_node = QueryNode(
        target=Var(ty=IRType.REAL, name="q"),
        query_name="count",
        args=(Var(ty=IRType.REAL, name="db"),),
        sensitivity=Const.real(1.0),
    )
    noise_node = NoiseDrawNode(
        target=Var(ty=IRType.REAL, name="eta"),
        noise_kind=NoiseKind.GAUSSIAN,
        center=Var(ty=IRType.REAL, name="q"),
        scale=Const.real(1.0),
    )
    ret = ReturnNode(value=Var(ty=IRType.REAL, name="eta"))
    body = SequenceNode(stmts=[q_node, noise_node, ret])

    return MechIR(
        name="gaussian_mech",
        params=[
            ParamDecl(name="db", ty=IRType.ARRAY, is_database=True),
            ParamDecl(name="sigma", ty=IRType.REAL),
        ],
        body=body,
        return_type=IRType.REAL,
        budget=ApproxBudget(epsilon=1.0, delta=1e-5),
    )


@pytest.fixture
def buggy_laplace_mechir() -> MechIR:
    """Buggy Laplace mechanism with insufficient noise (scale too small)."""
    q_node = QueryNode(
        target=Var(ty=IRType.REAL, name="q"),
        query_name="count",
        args=(Var(ty=IRType.REAL, name="db"),),
        sensitivity=Const.real(1.0),
    )
    noise_node = NoiseDrawNode(
        target=Var(ty=IRType.REAL, name="eta"),
        noise_kind=NoiseKind.LAPLACE,
        center=Var(ty=IRType.REAL, name="q"),
        scale=Const.real(0.1),  # Too small for eps=1
    )
    ret = ReturnNode(value=Var(ty=IRType.REAL, name="eta"))
    body = SequenceNode(stmts=[q_node, noise_node, ret])

    return MechIR(
        name="buggy_laplace",
        params=[
            ParamDecl(name="db", ty=IRType.ARRAY, is_database=True),
        ],
        body=body,
        return_type=IRType.REAL,
        budget=PureBudget(epsilon=1.0),
    )
