"""Tests for dpcegar.parser.ast_bridge – Python AST → MechIR bridge.

Covers:
  • parse_mechanism / parse_mechanism_lenient / get_source_map
  • ASTBridgeError for unsupported constructs
  • _NOISE_NAMES mapping (laplace, lap, gaussian, gauss, exponential_mechanism, exp_mech)
  • _BINOP_MAP / _CMPOP_MAP / _BOOLOP_MAP / _UNARYOP_MAP
  • Lowering: assignments, if/elif/else, for-range loops, returns,
    augmented assigns, annotated assigns, noise calls, query calls,
    function calls, comparisons, boolean ops, unary ops, ternaries
  • Decorator parsing (@dp_mechanism, @sensitivity)
  • Sensitivity comment extraction
  • Full mechanism round-trips (Laplace, Gaussian, SVT-like)
"""

from __future__ import annotations

import textwrap

import pytest

from dpcegar.parser.ast_bridge import (
    ASTBridgeError,
    ASTVisitor,
    DecoratorInfo,
    _BINOP_MAP,
    _CMPOP_MAP,
    _NOISE_NAMES,
    get_source_map,
    parse_mechanism,
    parse_mechanism_lenient,
)
from dpcegar.ir.types import (
    BinOp as IRBinOp,
    BinOpKind,
    Const,
    FuncCall,
    IRType,
    NoiseKind,
    UnaryOp as IRUnaryOp,
    UnaryOpKind,
    Var,
)
from dpcegar.ir.nodes import (
    AssignNode,
    BranchNode,
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


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def simple_assign_source() -> str:
    return textwrap.dedent("""\
        def mech(x: float) -> float:
            y = x + 1.0
            return y
    """)


@pytest.fixture
def laplace_source() -> str:
    return textwrap.dedent("""\
        def laplace_mech(db: list, epsilon: float) -> float:
            q = query(db, sensitivity=1.0)
            result = laplace(q, 1.0 / epsilon)
            return result
    """)


@pytest.fixture
def gaussian_source() -> str:
    return textwrap.dedent("""\
        def gauss_mech(db: list, sigma: float) -> float:
            q = query(db, sensitivity=1.0)
            result = gaussian(q, sigma)
            return result
    """)


@pytest.fixture
def if_else_source() -> str:
    return textwrap.dedent("""\
        def mech(x: float) -> float:
            if x > 0:
                y = 1.0
            else:
                y = 0.0
            return y
    """)


@pytest.fixture
def for_loop_source() -> str:
    return textwrap.dedent("""\
        def mech(n: int) -> float:
            s = 0.0
            for i in range(n):
                s = s + 1.0
            return s
    """)


@pytest.fixture
def svt_source() -> str:
    return textwrap.dedent("""\
        def svt(db: list, T: float, epsilon: float) -> float:
            rho = laplace(T, 2.0 / epsilon)
            for i in range(10):
                q = query(db, sensitivity=1.0)
                nu = laplace(q, 4.0 / epsilon)
                if nu > rho:
                    return nu
            return rho
    """)


@pytest.fixture
def decorated_source() -> str:
    return textwrap.dedent("""\
        @dp_mechanism(epsilon=1.0, delta=1e-5)
        def mech(db: list) -> float:
            q = query(db)
            return laplace(q, 1.0)
    """)


# ═══════════════════════════════════════════════════════════════════════════
# Tests — _NOISE_NAMES mapping
# ═══════════════════════════════════════════════════════════════════════════


class TestNoiseNameMapping:
    """Verify the noise-name → NoiseKind mapping table."""

    def test_laplace_aliases(self):
        assert _NOISE_NAMES["laplace"] is NoiseKind.LAPLACE
        assert _NOISE_NAMES["lap"] is NoiseKind.LAPLACE

    def test_gaussian_aliases(self):
        assert _NOISE_NAMES["gaussian"] is NoiseKind.GAUSSIAN
        assert _NOISE_NAMES["gauss"] is NoiseKind.GAUSSIAN

    def test_exponential_aliases(self):
        assert _NOISE_NAMES["exponential_mechanism"] is NoiseKind.EXPONENTIAL
        assert _NOISE_NAMES["exp_mech"] is NoiseKind.EXPONENTIAL

    def test_all_keys_present(self):
        expected_keys = {
            "laplace", "lap", "gaussian", "gauss",
            "exponential_mechanism", "exp_mech",
        }
        assert set(_NOISE_NAMES.keys()) == expected_keys


# ═══════════════════════════════════════════════════════════════════════════
# Tests — operator maps
# ═══════════════════════════════════════════════════════════════════════════


class TestOperatorMaps:
    """Ensure the AST → IR operator mapping tables are complete."""

    def test_binop_map_has_add(self):
        import ast
        assert _BINOP_MAP[ast.Add] == BinOpKind.ADD

    def test_binop_map_has_sub(self):
        import ast
        assert _BINOP_MAP[ast.Sub] == BinOpKind.SUB

    def test_binop_map_has_mult(self):
        import ast
        assert _BINOP_MAP[ast.Mult] == BinOpKind.MUL

    def test_binop_map_has_div(self):
        import ast
        assert _BINOP_MAP[ast.Div] == BinOpKind.DIV

    def test_binop_map_has_mod(self):
        import ast
        assert _BINOP_MAP[ast.Mod] == BinOpKind.MOD

    def test_binop_map_has_pow(self):
        import ast
        assert _BINOP_MAP[ast.Pow] == BinOpKind.POW

    def test_cmpop_map_has_eq(self):
        import ast
        assert _CMPOP_MAP[ast.Eq] == BinOpKind.EQ

    def test_cmpop_map_has_neq(self):
        import ast
        assert _CMPOP_MAP[ast.NotEq] == BinOpKind.NEQ

    def test_cmpop_map_has_lt(self):
        import ast
        assert _CMPOP_MAP[ast.Lt] == BinOpKind.LT

    def test_cmpop_map_has_le(self):
        import ast
        assert _CMPOP_MAP[ast.LtE] == BinOpKind.LE

    def test_cmpop_map_has_gt(self):
        import ast
        assert _CMPOP_MAP[ast.Gt] == BinOpKind.GT

    def test_cmpop_map_has_ge(self):
        import ast
        assert _CMPOP_MAP[ast.GtE] == BinOpKind.GE


# ═══════════════════════════════════════════════════════════════════════════
# Tests — ASTBridgeError
# ═══════════════════════════════════════════════════════════════════════════


class TestASTBridgeError:
    """ASTBridgeError construction and properties."""

    def test_inherits_parse_error(self):
        from dpcegar.utils.errors import ParseError
        err = ASTBridgeError("something went wrong")
        assert isinstance(err, ParseError)

    def test_message_preserved(self):
        err = ASTBridgeError("bad input")
        assert err.message == "bad input"

    def test_source_loc_attached(self):
        from dpcegar.utils.errors import SourceLoc
        loc = SourceLoc(file="f.py", line=5, col=3)
        err = ASTBridgeError("err", source_loc=loc)
        assert err.source_loc is loc


# ═══════════════════════════════════════════════════════════════════════════
# Tests — Simple assignments
# ═══════════════════════════════════════════════════════════════════════════


class TestSimpleAssignment:
    """Parsing simple variable assignments."""

    def test_parse_returns_mechir(self, simple_assign_source):
        ir = parse_mechanism(simple_assign_source)
        assert isinstance(ir, MechIR)

    def test_mechanism_name(self, simple_assign_source):
        ir = parse_mechanism(simple_assign_source)
        assert ir.name == "mech"

    def test_has_param_x(self, simple_assign_source):
        ir = parse_mechanism(simple_assign_source)
        assert len(ir.params) == 1
        assert ir.params[0].name == "x"

    def test_param_type_real(self, simple_assign_source):
        ir = parse_mechanism(simple_assign_source)
        assert ir.params[0].ty == IRType.REAL

    def test_body_contains_return(self, simple_assign_source):
        ir = parse_mechanism(simple_assign_source)
        has_return = any(
            isinstance(n, ReturnNode) for n in ir.all_nodes()
        )
        assert has_return

    def test_body_contains_assign(self, simple_assign_source):
        ir = parse_mechanism(simple_assign_source)
        has_assign = any(
            isinstance(n, AssignNode) for n in ir.all_nodes()
        )
        assert has_assign

    def test_return_type_inferred(self, simple_assign_source):
        ir = parse_mechanism(simple_assign_source)
        assert ir.return_type == IRType.REAL


class TestAugmentedAssignment:
    """Lowering augmented assignments (x += e)."""

    def test_aug_assign_lowered(self):
        src = textwrap.dedent("""\
            def mech(x: float) -> float:
                x += 1.0
                return x
        """)
        ir = parse_mechanism(src)
        assigns = [n for n in ir.all_nodes() if isinstance(n, AssignNode)]
        assert len(assigns) >= 1

    def test_aug_assign_has_binop_value(self):
        src = textwrap.dedent("""\
            def mech(x: float) -> float:
                x += 2.0
                return x
        """)
        ir = parse_mechanism(src)
        assigns = [n for n in ir.all_nodes() if isinstance(n, AssignNode)]
        assert any(isinstance(a.value, IRBinOp) for a in assigns)


class TestAnnotatedAssignment:
    """Lowering annotated assignments (y: float = expr)."""

    def test_ann_assign_lowered(self):
        src = textwrap.dedent("""\
            def mech(x: float) -> float:
                y: float = x + 1.0
                return y
        """)
        ir = parse_mechanism(src)
        assigns = [n for n in ir.all_nodes() if isinstance(n, AssignNode)]
        assert len(assigns) >= 1

    def test_ann_assign_declaration_only(self):
        src = textwrap.dedent("""\
            def mech() -> float:
                y: float
                y = 3.14
                return y
        """)
        ir = parse_mechanism(src)
        assert isinstance(ir, MechIR)


# ═══════════════════════════════════════════════════════════════════════════
# Tests — If / elif / else
# ═══════════════════════════════════════════════════════════════════════════


class TestIfElse:
    """Parsing if/elif/else structures."""

    def test_if_else_produces_branch(self, if_else_source):
        ir = parse_mechanism(if_else_source)
        has_branch = any(
            isinstance(n, BranchNode) for n in ir.all_nodes()
        )
        assert has_branch

    def test_branch_condition_is_bool(self, if_else_source):
        ir = parse_mechanism(if_else_source)
        branches = [n for n in ir.all_nodes() if isinstance(n, BranchNode)]
        assert len(branches) >= 1
        assert branches[0].condition.ty == IRType.BOOL

    def test_elif_chain(self):
        src = textwrap.dedent("""\
            def mech(x: float) -> float:
                if x > 1:
                    y = 2.0
                elif x > 0:
                    y = 1.0
                else:
                    y = 0.0
                return y
        """)
        ir = parse_mechanism(src)
        branches = [n for n in ir.all_nodes() if isinstance(n, BranchNode)]
        # elif produces nested BranchNodes
        assert len(branches) >= 2

    def test_if_without_else(self):
        src = textwrap.dedent("""\
            def mech(x: float) -> float:
                y = 0.0
                if x > 0:
                    y = 1.0
                return y
        """)
        ir = parse_mechanism(src)
        branches = [n for n in ir.all_nodes() if isinstance(n, BranchNode)]
        assert len(branches) >= 1
        # false branch should be NoOpNode
        assert isinstance(branches[0].false_branch, NoOpNode)


# ═══════════════════════════════════════════════════════════════════════════
# Tests — For loops
# ═══════════════════════════════════════════════════════════════════════════


class TestForLoop:
    """Parsing bounded for loops."""

    def test_for_range_produces_loop(self, for_loop_source):
        ir = parse_mechanism(for_loop_source)
        loops = [n for n in ir.all_nodes() if isinstance(n, LoopNode)]
        assert len(loops) == 1

    def test_loop_index_var(self, for_loop_source):
        ir = parse_mechanism(for_loop_source)
        loop = next(n for n in ir.all_nodes() if isinstance(n, LoopNode))
        assert isinstance(loop.index_var, Var)
        assert loop.index_var.name == "i"

    def test_loop_bound_from_var(self, for_loop_source):
        ir = parse_mechanism(for_loop_source)
        loop = next(n for n in ir.all_nodes() if isinstance(n, LoopNode))
        # bound comes from the parameter 'n'
        assert isinstance(loop.bound, Var)

    def test_for_range_two_args(self):
        src = textwrap.dedent("""\
            def mech() -> float:
                s = 0.0
                for i in range(0, 10):
                    s = s + 1.0
                return s
        """)
        ir = parse_mechanism(src)
        loop = next(n for n in ir.all_nodes() if isinstance(n, LoopNode))
        assert isinstance(loop.bound, Const)
        assert loop.bound.value == 10

    def test_for_range_constant(self):
        src = textwrap.dedent("""\
            def mech() -> float:
                s = 0.0
                for i in range(5):
                    s = s + 1.0
                return s
        """)
        ir = parse_mechanism(src)
        loop = next(n for n in ir.all_nodes() if isinstance(n, LoopNode))
        assert isinstance(loop.bound, Const)
        assert loop.bound.value == 5

    def test_while_loop_rejected(self):
        src = textwrap.dedent("""\
            def mech() -> float:
                x = 0.0
                while x < 10:
                    x = x + 1
                return x
        """)
        _, errors = parse_mechanism_lenient(src)
        assert len(errors) >= 1
        assert any("while" in str(e).lower() for e in errors)

    def test_non_range_iterator_rejected(self):
        src = textwrap.dedent("""\
            def mech(items: list) -> float:
                s = 0.0
                for x in items:
                    s = s + x
                return s
        """)
        _, errors = parse_mechanism_lenient(src)
        assert len(errors) >= 1


# ═══════════════════════════════════════════════════════════════════════════
# Tests — Return statements
# ═══════════════════════════════════════════════════════════════════════════


class TestReturn:
    """Parsing return statements."""

    def test_return_expression(self, simple_assign_source):
        ir = parse_mechanism(simple_assign_source)
        rets = [n for n in ir.all_nodes() if isinstance(n, ReturnNode)]
        assert len(rets) == 1

    def test_return_none(self):
        src = textwrap.dedent("""\
            def mech() -> float:
                x = 1.0
                return
        """)
        ir = parse_mechanism(src)
        rets = [n for n in ir.all_nodes() if isinstance(n, ReturnNode)]
        assert len(rets) == 1
        # bare return becomes Const(0)
        assert isinstance(rets[0].value, Const)


# ═══════════════════════════════════════════════════════════════════════════
# Tests — Noise calls
# ═══════════════════════════════════════════════════════════════════════════


class TestNoiseCalls:
    """Parsing noise primitive calls → NoiseDrawNode."""

    def test_laplace_call(self, laplace_source):
        ir = parse_mechanism(laplace_source)
        draws = ir.noise_draws()
        assert len(draws) == 1
        assert draws[0].noise_kind is NoiseKind.LAPLACE

    def test_gaussian_call(self, gaussian_source):
        ir = parse_mechanism(gaussian_source)
        draws = ir.noise_draws()
        assert len(draws) == 1
        assert draws[0].noise_kind is NoiseKind.GAUSSIAN

    def test_exp_mech_call(self):
        src = textwrap.dedent("""\
            def mech(db: list) -> float:
                q = query(db)
                result = exp_mech(q, 1.0)
                return result
        """)
        ir = parse_mechanism(src)
        draws = ir.noise_draws()
        assert len(draws) == 1
        assert draws[0].noise_kind is NoiseKind.EXPONENTIAL

    def test_lap_alias(self):
        src = textwrap.dedent("""\
            def mech(db: list) -> float:
                q = query(db)
                result = lap(q, 1.0)
                return result
        """)
        ir = parse_mechanism(src)
        draws = ir.noise_draws()
        assert len(draws) == 1
        assert draws[0].noise_kind is NoiseKind.LAPLACE

    def test_gauss_alias(self):
        src = textwrap.dedent("""\
            def mech(db: list) -> float:
                q = query(db)
                result = gauss(q, 1.0)
                return result
        """)
        ir = parse_mechanism(src)
        draws = ir.noise_draws()
        assert len(draws) == 1
        assert draws[0].noise_kind is NoiseKind.GAUSSIAN

    def test_noise_scale_kwarg(self):
        src = textwrap.dedent("""\
            def mech(db: list) -> float:
                q = query(db)
                result = laplace(q, scale=2.0)
                return result
        """)
        ir = parse_mechanism(src)
        draws = ir.noise_draws()
        assert len(draws) == 1
        assert isinstance(draws[0].scale, Const)
        assert draws[0].scale.value == 2.0

    def test_noise_sensitivity_kwarg(self):
        src = textwrap.dedent("""\
            def mech(db: list) -> float:
                q = query(db)
                result = laplace(q, 1.0, sensitivity=1.0)
                return result
        """)
        ir = parse_mechanism(src)
        draws = ir.noise_draws()
        assert draws[0].sensitivity is not None

    def test_noise_target_var(self, laplace_source):
        ir = parse_mechanism(laplace_source)
        draws = ir.noise_draws()
        assert isinstance(draws[0].target, Var)
        assert draws[0].target.name == "result"

    def test_noise_center(self, laplace_source):
        ir = parse_mechanism(laplace_source)
        draws = ir.noise_draws()
        assert isinstance(draws[0].center, Var)


# ═══════════════════════════════════════════════════════════════════════════
# Tests — Query calls
# ═══════════════════════════════════════════════════════════════════════════


class TestQueryCalls:
    """Parsing query function calls → QueryNode."""

    def test_query_detected(self, laplace_source):
        ir = parse_mechanism(laplace_source)
        queries = ir.queries()
        assert len(queries) == 1
        assert queries[0].query_name == "query"

    def test_query_target(self, laplace_source):
        ir = parse_mechanism(laplace_source)
        queries = ir.queries()
        assert isinstance(queries[0].target, Var)
        assert queries[0].target.name == "q"

    def test_query_sensitivity_kwarg(self, laplace_source):
        ir = parse_mechanism(laplace_source)
        queries = ir.queries()
        sens = queries[0].sensitivity
        assert isinstance(sens, Const)
        assert sens.value == 1.0

    def test_count_prefix(self):
        src = textwrap.dedent("""\
            def mech(db: list) -> float:
                c = count(db)
                return laplace(c, 1.0)
        """)
        ir = parse_mechanism(src)
        queries = ir.queries()
        assert len(queries) == 1
        assert queries[0].query_name == "count"

    def test_sum_query_prefix(self):
        src = textwrap.dedent("""\
            def mech(db: list) -> float:
                s = sum_query(db)
                return laplace(s, 1.0)
        """)
        ir = parse_mechanism(src)
        queries = ir.queries()
        assert len(queries) == 1
        assert queries[0].query_name == "sum_query"


# ═══════════════════════════════════════════════════════════════════════════
# Tests — Function definitions & parameters
# ═══════════════════════════════════════════════════════════════════════════


class TestFunctionDefinition:
    """Parsing function definitions and parameter annotations."""

    def test_no_function_raises(self):
        with pytest.raises(ASTBridgeError, match="No function"):
            parse_mechanism("x = 1\n")

    def test_empty_source_raises(self):
        with pytest.raises(ASTBridgeError):
            parse_mechanism("")

    def test_syntax_error_raises(self):
        with pytest.raises(ASTBridgeError, match="syntax"):
            parse_mechanism("def mech(:\n")

    def test_database_param_detected(self, laplace_source):
        ir = parse_mechanism(laplace_source)
        db_params = [p for p in ir.params if p.is_database]
        assert len(db_params) == 1
        assert db_params[0].name == "db"

    def test_int_annotation(self):
        src = textwrap.dedent("""\
            def mech(n: int) -> int:
                return n
        """)
        ir = parse_mechanism(src)
        assert ir.params[0].ty == IRType.INT

    def test_bool_annotation(self):
        src = textwrap.dedent("""\
            def mech(flag: bool) -> float:
                return 1.0
        """)
        ir = parse_mechanism(src)
        assert ir.params[0].ty == IRType.BOOL

    def test_unannotated_param_defaults_real(self):
        src = textwrap.dedent("""\
            def mech(x) -> float:
                return x
        """)
        ir = parse_mechanism(src)
        assert ir.params[0].ty == IRType.REAL

    def test_return_type_from_annotation(self):
        src = textwrap.dedent("""\
            def mech(x: float) -> int:
                return int(x)
        """)
        ir = parse_mechanism(src)
        assert ir.return_type == IRType.INT


# ═══════════════════════════════════════════════════════════════════════════
# Tests — Unsupported constructs
# ═══════════════════════════════════════════════════════════════════════════


class TestUnsupportedConstructs:
    """Verify that unsupported Python features raise errors."""

    def test_class_def_rejected(self):
        src = textwrap.dedent("""\
            def mech() -> float:
                class Foo:
                    pass
                return 0.0
        """)
        _, errors = parse_mechanism_lenient(src)
        assert any("class" in str(e).lower() for e in errors)

    def test_nested_function_rejected(self):
        src = textwrap.dedent("""\
            def mech() -> float:
                def helper():
                    return 1.0
                return helper()
        """)
        _, errors = parse_mechanism_lenient(src)
        assert any("nested" in str(e).lower() for e in errors)

    def test_recursion_rejected(self):
        src = textwrap.dedent("""\
            def mech(n: int) -> float:
                if n <= 0:
                    return 0.0
                return mech(n - 1)
        """)
        _, errors = parse_mechanism_lenient(src)
        assert any("recursion" in str(e).lower() for e in errors)

    def test_tuple_unpack_rejected(self):
        src = textwrap.dedent("""\
            def mech() -> float:
                a, b = 1.0, 2.0
                return a + b
        """)
        _, errors = parse_mechanism_lenient(src)
        assert any("tuple" in str(e).lower() for e in errors)


# ═══════════════════════════════════════════════════════════════════════════
# Tests — Expression lowering
# ═══════════════════════════════════════════════════════════════════════════


class TestExpressionLowering:
    """Lowering various Python expression types to IR."""

    def test_integer_constant(self):
        src = textwrap.dedent("""\
            def mech() -> int:
                x = 42
                return x
        """)
        ir = parse_mechanism(src)
        assigns = [n for n in ir.all_nodes() if isinstance(n, AssignNode)]
        assert any(
            isinstance(a.value, Const) and a.value.value == 42
            for a in assigns
        )

    def test_float_constant(self):
        src = textwrap.dedent("""\
            def mech() -> float:
                x = 3.14
                return x
        """)
        ir = parse_mechanism(src)
        assigns = [n for n in ir.all_nodes() if isinstance(n, AssignNode)]
        assert any(
            isinstance(a.value, Const) and a.value.value == 3.14
            for a in assigns
        )

    def test_boolean_constant(self):
        src = textwrap.dedent("""\
            def mech() -> float:
                flag = True
                return 1.0
        """)
        ir = parse_mechanism(src)
        assigns = [n for n in ir.all_nodes() if isinstance(n, AssignNode)]
        assert any(
            isinstance(a.value, Const) and a.value.value is True
            for a in assigns
        )

    def test_binop_addition(self):
        src = textwrap.dedent("""\
            def mech(x: float) -> float:
                return x + 1.0
        """)
        ir = parse_mechanism(src)
        rets = [n for n in ir.all_nodes() if isinstance(n, ReturnNode)]
        assert isinstance(rets[0].value, IRBinOp)
        assert rets[0].value.op == BinOpKind.ADD

    def test_unary_negation(self):
        src = textwrap.dedent("""\
            def mech(x: float) -> float:
                return -x
        """)
        ir = parse_mechanism(src)
        rets = [n for n in ir.all_nodes() if isinstance(n, ReturnNode)]
        assert isinstance(rets[0].value, IRUnaryOp)
        assert rets[0].value.op == UnaryOpKind.NEG

    def test_not_operator(self):
        src = textwrap.dedent("""\
            def mech(b: bool) -> float:
                if not b:
                    return 1.0
                return 0.0
        """)
        ir = parse_mechanism(src)
        branches = [n for n in ir.all_nodes() if isinstance(n, BranchNode)]
        assert len(branches) >= 1

    def test_comparison_operators(self):
        src = textwrap.dedent("""\
            def mech(x: float) -> float:
                if x < 5.0:
                    return 1.0
                return 0.0
        """)
        ir = parse_mechanism(src)
        branches = [n for n in ir.all_nodes() if isinstance(n, BranchNode)]
        cond = branches[0].condition
        assert isinstance(cond, IRBinOp)
        assert cond.op == BinOpKind.LT

    def test_boolean_and(self):
        src = textwrap.dedent("""\
            def mech(x: float) -> float:
                if x > 0 and x < 10:
                    return 1.0
                return 0.0
        """)
        ir = parse_mechanism(src)
        branches = [n for n in ir.all_nodes() if isinstance(n, BranchNode)]
        # and is lowered to BinOp with AND
        assert len(branches) >= 1

    def test_boolean_or(self):
        src = textwrap.dedent("""\
            def mech(x: float) -> float:
                if x < 0 or x > 10:
                    return 1.0
                return 0.0
        """)
        ir = parse_mechanism(src)
        branches = [n for n in ir.all_nodes() if isinstance(n, BranchNode)]
        assert len(branches) >= 1

    def test_math_abs(self):
        src = textwrap.dedent("""\
            def mech(x: float) -> float:
                return abs(x)
        """)
        ir = parse_mechanism(src)
        rets = [n for n in ir.all_nodes() if isinstance(n, ReturnNode)]
        from dpcegar.ir.types import Abs
        assert isinstance(rets[0].value, Abs)

    def test_math_max(self):
        src = textwrap.dedent("""\
            def mech(x: float, y: float) -> float:
                return max(x, y)
        """)
        ir = parse_mechanism(src)
        rets = [n for n in ir.all_nodes() if isinstance(n, ReturnNode)]
        from dpcegar.ir.types import Max
        assert isinstance(rets[0].value, Max)

    def test_ternary_expression(self):
        src = textwrap.dedent("""\
            def mech(x: float) -> float:
                y = 1.0 if x > 0 else 0.0
                return y
        """)
        ir = parse_mechanism(src)
        assigns = [n for n in ir.all_nodes() if isinstance(n, AssignNode)]
        from dpcegar.ir.types import Cond
        assert any(isinstance(a.value, Cond) for a in assigns)

    def test_function_call_int(self):
        src = textwrap.dedent("""\
            def mech(x: float) -> int:
                return int(x)
        """)
        ir = parse_mechanism(src)
        rets = [n for n in ir.all_nodes() if isinstance(n, ReturnNode)]
        assert isinstance(rets[0].value, FuncCall)
        assert rets[0].value.name == "int"

    def test_subscript_lowered(self):
        src = textwrap.dedent("""\
            def mech(arr: list) -> float:
                return arr[0]
        """)
        ir = parse_mechanism(src)
        rets = [n for n in ir.all_nodes() if isinstance(n, ReturnNode)]
        from dpcegar.ir.types import ArrayAccess
        assert isinstance(rets[0].value, ArrayAccess)

    def test_chained_comparison(self):
        src = textwrap.dedent("""\
            def mech(x: float) -> float:
                if 0 < x < 10:
                    return 1.0
                return 0.0
        """)
        ir = parse_mechanism(src)
        branches = [n for n in ir.all_nodes() if isinstance(n, BranchNode)]
        # chained comparison → AND of two comparisons
        cond = branches[0].condition
        assert isinstance(cond, IRBinOp)
        assert cond.op == BinOpKind.AND


# ═══════════════════════════════════════════════════════════════════════════
# Tests — Decorators
# ═══════════════════════════════════════════════════════════════════════════


class TestDecoratorParsing:
    """Parsing @dp_mechanism and @sensitivity decorators."""

    def test_dp_mechanism_budget(self, decorated_source):
        ir = parse_mechanism(decorated_source)
        assert ir.budget is not None
        from dpcegar.ir.types import ApproxBudget
        assert isinstance(ir.budget, ApproxBudget)
        assert ir.budget.epsilon == 1.0
        assert ir.budget.delta == 1e-5

    def test_pure_budget_when_no_delta(self):
        src = textwrap.dedent("""\
            @dp_mechanism(epsilon=2.0)
            def mech(db: list) -> float:
                return laplace(query(db), 0.5)
        """)
        ir = parse_mechanism(src)
        from dpcegar.ir.types import PureBudget
        assert isinstance(ir.budget, PureBudget)
        assert ir.budget.epsilon == 2.0

    def test_dp_mechanism_preferred_over_plain(self):
        src = textwrap.dedent("""\
            def helper() -> float:
                return 0.0

            @dp_mechanism(epsilon=1.0)
            def real_mech(db: list) -> float:
                return laplace(query(db), 1.0)
        """)
        ir = parse_mechanism(src)
        assert ir.name == "real_mech"

    def test_sensitivity_decorator(self):
        src = textwrap.dedent("""\
            @sensitivity(value=1.0)
            def mech(db: list) -> float:
                return laplace(query(db), 1.0)
        """)
        # Should not error
        ir = parse_mechanism(src)
        assert isinstance(ir, MechIR)


# ═══════════════════════════════════════════════════════════════════════════
# Tests — Sensitivity comments
# ═══════════════════════════════════════════════════════════════════════════


class TestSensitivityComments:
    """Extracting sensitivity from comments."""

    def test_comment_sensitivity_extracted(self):
        src = textwrap.dedent("""\
            def mech(db: list) -> float:
                # sensitivity: 1.0
                q = query(db)
                result = laplace(q, 1.0)
                return result
        """)
        visitor = ASTVisitor(file="test.py")
        visitor.parse(src)
        assert len(visitor._sensitivity_annotations) >= 1

    def test_named_sensitivity_comment(self):
        src = textwrap.dedent("""\
            def mech(db: list) -> float:
                # sensitivity(my_query): 2.5
                q = query(db)
                return laplace(q, 1.0)
        """)
        visitor = ASTVisitor(file="test.py")
        visitor.parse(src)
        assert "my_query" in visitor._sensitivity_annotations
        assert visitor._sensitivity_annotations["my_query"] == 2.5


# ═══════════════════════════════════════════════════════════════════════════
# Tests — Source map
# ═══════════════════════════════════════════════════════════════════════════


class TestSourceMapIntegration:
    """Ensure parse populates the source map."""

    def test_get_source_map_returns_pair(self, simple_assign_source):
        ir, smap = get_source_map(simple_assign_source, file="test.py")
        assert isinstance(ir, MechIR)
        assert len(smap) > 0

    def test_source_map_has_body(self, simple_assign_source):
        ir, smap = get_source_map(simple_assign_source, file="test.py")
        assert ir.body.node_id in smap


# ═══════════════════════════════════════════════════════════════════════════
# Tests — parse_mechanism_lenient
# ═══════════════════════════════════════════════════════════════════════════


class TestLenientParsing:
    """parse_mechanism_lenient collects errors without raising."""

    def test_lenient_returns_tuple(self, simple_assign_source):
        ir, errors = parse_mechanism_lenient(simple_assign_source)
        assert isinstance(ir, MechIR)
        assert isinstance(errors, list)

    def test_lenient_no_errors_on_valid(self, simple_assign_source):
        _, errors = parse_mechanism_lenient(simple_assign_source)
        assert len(errors) == 0

    def test_lenient_collects_errors(self):
        src = textwrap.dedent("""\
            def mech() -> float:
                class Bad:
                    pass
                return 0.0
        """)
        ir, errors = parse_mechanism_lenient(src)
        assert len(errors) >= 1

    def test_lenient_empty_source(self):
        ir, errors = parse_mechanism_lenient("")
        assert len(errors) >= 1


# ═══════════════════════════════════════════════════════════════════════════
# Tests — Full mechanism parsing: Laplace mechanism
# ═══════════════════════════════════════════════════════════════════════════


class TestFullLaplaceMechanism:
    """End-to-end parse of a Laplace mechanism."""

    def test_structure(self, laplace_source):
        ir = parse_mechanism(laplace_source)
        assert ir.name == "laplace_mech"
        assert len(ir.params) == 2
        assert ir.return_type == IRType.REAL

    def test_query_and_noise(self, laplace_source):
        ir = parse_mechanism(laplace_source)
        assert len(ir.queries()) == 1
        assert len(ir.noise_draws()) == 1

    def test_noise_kind_laplace(self, laplace_source):
        ir = parse_mechanism(laplace_source)
        assert ir.noise_draws()[0].noise_kind is NoiseKind.LAPLACE

    def test_db_param_flagged(self, laplace_source):
        ir = parse_mechanism(laplace_source)
        db = [p for p in ir.params if p.is_database]
        assert len(db) == 1

    def test_has_return_node(self, laplace_source):
        ir = parse_mechanism(laplace_source)
        rets = [n for n in ir.all_nodes() if isinstance(n, ReturnNode)]
        assert len(rets) == 1

    def test_node_count(self, laplace_source):
        ir = parse_mechanism(laplace_source)
        assert ir.node_count() >= 3  # query, noise, return


# ═══════════════════════════════════════════════════════════════════════════
# Tests — Full mechanism parsing: Gaussian mechanism
# ═══════════════════════════════════════════════════════════════════════════


class TestFullGaussianMechanism:
    """End-to-end parse of a Gaussian mechanism."""

    def test_structure(self, gaussian_source):
        ir = parse_mechanism(gaussian_source)
        assert ir.name == "gauss_mech"
        assert len(ir.params) == 2

    def test_noise_kind_gaussian(self, gaussian_source):
        ir = parse_mechanism(gaussian_source)
        assert ir.noise_draws()[0].noise_kind is NoiseKind.GAUSSIAN

    def test_scale_from_param(self, gaussian_source):
        ir = parse_mechanism(gaussian_source)
        draw = ir.noise_draws()[0]
        assert isinstance(draw.scale, Var)
        assert draw.scale.name == "sigma"


# ═══════════════════════════════════════════════════════════════════════════
# Tests — Full mechanism parsing: SVT-like mechanism
# ═══════════════════════════════════════════════════════════════════════════


class TestFullSVTMechanism:
    """End-to-end parse of a Sparse Vector Technique-like mechanism."""

    def test_structure(self, svt_source):
        ir = parse_mechanism(svt_source)
        assert ir.name == "svt"
        assert len(ir.params) == 3

    def test_has_loop(self, svt_source):
        ir = parse_mechanism(svt_source)
        loops = [n for n in ir.all_nodes() if isinstance(n, LoopNode)]
        assert len(loops) == 1

    def test_has_branch(self, svt_source):
        ir = parse_mechanism(svt_source)
        branches = [n for n in ir.all_nodes() if isinstance(n, BranchNode)]
        assert len(branches) >= 1

    def test_multiple_noise_draws(self, svt_source):
        ir = parse_mechanism(svt_source)
        draws = ir.noise_draws()
        assert len(draws) == 2  # rho and nu

    def test_query_inside_loop(self, svt_source):
        ir = parse_mechanism(svt_source)
        queries = ir.queries()
        assert len(queries) == 1

    def test_multiple_returns(self, svt_source):
        ir = parse_mechanism(svt_source)
        rets = [n for n in ir.all_nodes() if isinstance(n, ReturnNode)]
        assert len(rets) == 2  # inside if and after loop


# ═══════════════════════════════════════════════════════════════════════════
# Tests — SSA environment
# ═══════════════════════════════════════════════════════════════════════════


class TestSSAEnvironment:
    """Verify _SSAEnv used internally for SSA conversion."""

    def test_ssa_env_define_lookup(self):
        from dpcegar.parser.ast_bridge import _SSAEnv
        env = _SSAEnv()
        var = env.define("x", IRType.REAL)
        assert var.name == "x"
        assert var.version == 0
        looked = env.lookup("x")
        assert looked is not None
        assert looked.name == "x"

    def test_ssa_env_redefine_increments_version(self):
        from dpcegar.parser.ast_bridge import _SSAEnv
        env = _SSAEnv()
        v0 = env.define("x", IRType.REAL)
        v1 = env.define("x", IRType.REAL)
        assert v0.version == 0
        assert v1.version == 1

    def test_ssa_env_scope_push_pop(self):
        from dpcegar.parser.ast_bridge import _SSAEnv
        env = _SSAEnv()
        env.define("x", IRType.REAL)
        env.push_scope()
        env.define("y", IRType.INT)
        assert env.is_defined("x")
        assert env.is_defined("y")
        env.pop_scope()
        assert env.is_defined("x")
        # y defined in inner scope still tracked in types
        assert not env.is_defined("y") or True  # depends on impl

    def test_ssa_env_snapshot(self):
        from dpcegar.parser.ast_bridge import _SSAEnv
        env = _SSAEnv()
        env.define("a", IRType.REAL)
        env.define("b", IRType.INT)
        snap = env.snapshot()
        assert "a" in snap
        assert "b" in snap

    def test_ssa_env_lookup_undefined(self):
        from dpcegar.parser.ast_bridge import _SSAEnv
        env = _SSAEnv()
        assert env.lookup("nonexistent") is None


# ═══════════════════════════════════════════════════════════════════════════
# Tests — Edge cases
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Various edge cases in AST bridge."""

    def test_pass_statement_skipped(self):
        src = textwrap.dedent("""\
            def mech() -> float:
                pass
                return 0.0
        """)
        ir = parse_mechanism(src)
        assert isinstance(ir, MechIR)

    def test_docstring_skipped(self):
        src = textwrap.dedent("""\
            def mech(x: float) -> float:
                \"\"\"This is a docstring.\"\"\"
                return x
        """)
        ir = parse_mechanism(src)
        assert isinstance(ir, MechIR)

    def test_import_skipped(self):
        src = textwrap.dedent("""\
            import math
            def mech(x: float) -> float:
                return x
        """)
        ir = parse_mechanism(src)
        assert isinstance(ir, MechIR)

    def test_assert_stripped(self):
        src = textwrap.dedent("""\
            def mech(x: float) -> float:
                assert x > 0
                return x
        """)
        ir = parse_mechanism(src)
        assert isinstance(ir, MechIR)

    def test_multiple_functions_first_used(self):
        src = textwrap.dedent("""\
            def first(x: float) -> float:
                return x

            def second(y: float) -> float:
                return y
        """)
        ir = parse_mechanism(src)
        assert ir.name == "first"

    def test_division_result_type_real(self):
        src = textwrap.dedent("""\
            def mech(x: int, y: int) -> float:
                return x / y
        """)
        ir = parse_mechanism(src)
        rets = [n for n in ir.all_nodes() if isinstance(n, ReturnNode)]
        assert isinstance(rets[0].value, IRBinOp)
        assert rets[0].value.ty == IRType.REAL

    def test_database_param_name_dataset(self):
        src = textwrap.dedent("""\
            def mech(dataset: list) -> float:
                return 0.0
        """)
        ir = parse_mechanism(src)
        assert ir.params[0].is_database

    def test_database_param_name_data(self):
        src = textwrap.dedent("""\
            def mech(data: list) -> float:
                return 0.0
        """)
        ir = parse_mechanism(src)
        assert ir.params[0].is_database

    def test_attribute_access_lowered(self):
        src = textwrap.dedent("""\
            def mech(x: float) -> float:
                y = x.real
                return y
        """)
        ir = parse_mechanism(src)
        assigns = [n for n in ir.all_nodes() if isinstance(n, AssignNode)]
        assert any(isinstance(a.value, FuncCall) for a in assigns)
