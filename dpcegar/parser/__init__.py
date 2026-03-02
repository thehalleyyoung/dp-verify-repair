"""Front-end parsers for mechanism description languages.

This sub-package converts DPImp source code (a Python subset for
differential privacy mechanisms) into MechIR for verification.

Modules
-------
lexer         – Token definitions and lexer for DPImp
ast_bridge    – Python AST → MechIR lowering
type_checker  – DPImp type checking and validation
sensitivity   – Sensitivity analysis via abstract interpretation
preprocessor  – Source preprocessing (imports, macros, inlining)
source_map    – Source location tracking for diagnostics
"""

from dpcegar.parser.lexer import (
    Lexer,
    Token,
    TokenType,
    tokenize,
    filter_tokens,
    strip_comments,
)
from dpcegar.parser.ast_bridge import (
    ASTBridgeError,
    ASTVisitor,
    parse_mechanism,
    parse_mechanism_lenient,
    get_source_map,
)
from dpcegar.parser.type_checker import (
    TypeEnvironment,
    TypeChecker,
    TypeErrorInfo,
    SensitivityKind,
    SensitivityType,
    type_check,
    type_check_strict,
)
from dpcegar.parser.sensitivity import (
    SensitivityNorm,
    SymbolicSens,
    SensitivityResult,
    SensitivityCert,
    SensitivityAnalyzer,
    QuerySensitivity,
    NoiseInfo,
    analyze_sensitivity,
    generate_sensitivity_certificate,
    sequential_compose,
    parallel_compose,
)
from dpcegar.parser.preprocessor import (
    ImportResolver,
    ImportInfo,
    PreprocessResult,
    preprocess,
    expand_macros,
    inline_helpers,
    normalize,
    strip_runtime,
    expand_elif_chains,
)
from dpcegar.parser.source_map import (
    SourceRange,
    SourceMap,
)

__all__ = [
    # lexer
    "Lexer", "Token", "TokenType",
    "tokenize", "filter_tokens", "strip_comments",
    # ast_bridge
    "ASTBridgeError", "ASTVisitor",
    "parse_mechanism", "parse_mechanism_lenient", "get_source_map",
    # type_checker
    "TypeEnvironment", "TypeChecker", "TypeErrorInfo",
    "SensitivityKind", "SensitivityType",
    "type_check", "type_check_strict",
    # sensitivity
    "SensitivityNorm", "SymbolicSens", "SensitivityResult",
    "SensitivityCert", "SensitivityAnalyzer",
    "QuerySensitivity", "NoiseInfo",
    "analyze_sensitivity", "generate_sensitivity_certificate",
    "sequential_compose", "parallel_compose",
    # preprocessor
    "ImportResolver", "ImportInfo", "PreprocessResult",
    "preprocess", "expand_macros", "inline_helpers",
    "normalize", "strip_runtime", "expand_elif_chains",
    # source_map
    "SourceRange", "SourceMap",
]
