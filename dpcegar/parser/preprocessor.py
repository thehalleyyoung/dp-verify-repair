"""Source preprocessor for DPImp mechanism code.

Handles import resolution, macro expansion, helper inlining, and
code normalisation before the main AST lowering pass.

Functions
---------
preprocess          – full preprocessing pipeline
resolve_imports     – handle DP library imports
expand_macros       – expand common DP patterns
inline_helpers      – inline simple helper functions
normalize           – normalise code to canonical form
strip_runtime       – strip non-essential runtime code
"""

from __future__ import annotations

import ast
import re
import textwrap
from dataclasses import dataclass, field
from typing import Any, Sequence

from dpcegar.utils.errors import ParseError, SourceLoc


# ═══════════════════════════════════════════════════════════════════════════
# IMPORT RESOLUTION
# ═══════════════════════════════════════════════════════════════════════════

# Known DP library modules and their provided names
_DP_LIBRARY_NAMES: dict[str, dict[str, str]] = {
    "dp_mechanisms": {
        "laplace": "laplace",
        "gaussian": "gaussian",
        "exponential_mechanism": "exponential_mechanism",
        "lap": "laplace",
        "gauss": "gaussian",
    },
    "dp_noise": {
        "laplace_noise": "laplace",
        "gaussian_noise": "gaussian",
        "add_laplace": "laplace",
        "add_gaussian": "gaussian",
    },
    "dp_queries": {
        "count_query": "count_query",
        "sum_query": "sum_query",
        "mean_query": "mean_query",
        "histogram_query": "histogram_query",
    },
    "dp_budget": {
        "dp_mechanism": "dp_mechanism",
        "sensitivity": "sensitivity",
        "privacy_budget": "privacy_budget",
    },
}

# All known DP names across modules
_ALL_DP_NAMES: dict[str, str] = {}
for _mod_names in _DP_LIBRARY_NAMES.values():
    _ALL_DP_NAMES.update(_mod_names)


@dataclass
class ImportInfo:
    """Information about a resolved import.

    Attributes:
        module:     The source module name.
        name:       The imported name.
        alias:      Alias (if any).
        canonical:  Canonical name in our system.
        line:       Source line number.
    """

    module: str
    name: str
    alias: str | None = None
    canonical: str = ""
    line: int = 0


class ImportResolver:
    """Resolves imports from DP mechanism libraries.

    Tracks which DP primitives are imported and their aliases, so the
    AST bridge can correctly identify noise calls even when imported
    under different names.

    Usage::

        resolver = ImportResolver()
        cleaned_source, imports = resolver.resolve(source)
    """

    def __init__(self) -> None:
        """Initialize the import resolver."""
        self.imports: list[ImportInfo] = []
        self.name_map: dict[str, str] = {}
        self.warnings: list[str] = []

    def resolve(self, source: str) -> tuple[str, list[ImportInfo]]:
        """Resolve imports in the source code.

        Strips known DP library imports and builds a name mapping.
        Unknown imports are left in place.

        Args:
            source: The DPImp source code.

        Returns:
            A tuple of (cleaned_source, import_list).
        """
        self.imports.clear()
        self.name_map.clear()

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return source, []

        lines = source.splitlines(keepends=True)
        lines_to_remove: set[int] = set()

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ImportFrom):
                self._process_import_from(node, lines_to_remove)
            elif isinstance(node, ast.Import):
                self._process_import(node, lines_to_remove)

        # Remove resolved import lines
        result_lines = [
            line for i, line in enumerate(lines, 1) if i not in lines_to_remove
        ]
        return "".join(result_lines), self.imports

    def _process_import_from(
        self, node: ast.ImportFrom, to_remove: set[int]
    ) -> None:
        """Process a 'from module import ...' statement."""
        module = node.module or ""
        is_dp_module = any(
            module.endswith(known) or module == known
            for known in _DP_LIBRARY_NAMES
        )

        for alias in node.names:
            name = alias.name
            asname = alias.asname

            canonical = _ALL_DP_NAMES.get(name, "")
            if canonical or is_dp_module:
                self.imports.append(ImportInfo(
                    module=module,
                    name=name,
                    alias=asname,
                    canonical=canonical or name,
                    line=node.lineno,
                ))
                # Map alias → canonical name
                used_name = asname if asname else name
                self.name_map[used_name] = canonical or name
                to_remove.add(node.lineno)
                # Handle multi-line imports
                if node.end_lineno:
                    for line in range(node.lineno, node.end_lineno + 1):
                        to_remove.add(line)

    def _process_import(
        self, node: ast.Import, to_remove: set[int]
    ) -> None:
        """Process a bare 'import module' statement."""
        for alias in node.names:
            name = alias.name
            if name in _DP_LIBRARY_NAMES:
                to_remove.add(node.lineno)
                self.name_map[alias.asname or name] = name

    def get_canonical_name(self, name: str) -> str:
        """Get the canonical name for an imported identifier.

        Args:
            name: The name as used in source code.

        Returns:
            The canonical name, or the original name if unknown.
        """
        return self.name_map.get(name, name)


# ═══════════════════════════════════════════════════════════════════════════
# MACRO EXPANSION
# ═══════════════════════════════════════════════════════════════════════════


# Common DP patterns that can be expanded
_MACRO_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # noisy_count(db, eps) → count_result = count_query(db); laplace(count_result, 1.0/eps)
    (
        re.compile(
            r"(\w+)\s*=\s*noisy_count\(\s*(\w+)\s*,\s*([^)]+)\s*\)"
        ),
        r"\1_q = count_query(\2)\n\1 = laplace(\1_q, 1.0 / (\3))",
    ),
    # noisy_sum(db, eps, sensitivity) → sum_result = sum_query(db); laplace(sum_result, sensitivity/eps)
    (
        re.compile(
            r"(\w+)\s*=\s*noisy_sum\(\s*(\w+)\s*,\s*([^,]+)\s*,\s*([^)]+)\s*\)"
        ),
        r"\1_q = sum_query(\2)\n\1 = laplace(\1_q, (\4) / (\3))",
    ),
    # noisy_mean(db, eps, sensitivity) → mean_result = mean_query(db); laplace(mean_result, sensitivity/eps)
    (
        re.compile(
            r"(\w+)\s*=\s*noisy_mean\(\s*(\w+)\s*,\s*([^,]+)\s*,\s*([^)]+)\s*\)"
        ),
        r"\1_q = mean_query(\2)\n\1 = laplace(\1_q, (\4) / (\3))",
    ),
    # above_threshold(db, queries, threshold, eps) → expansion
    (
        re.compile(
            r"(\w+)\s*=\s*above_threshold\(\s*(\w+)\s*,\s*(\w+)\s*,\s*([^,]+)\s*,\s*([^)]+)\s*\)"
        ),
        r"_noisy_T = laplace(\4, 2.0 / (\5))\nfor _i in range(len(\3)):\n"
        r"    _q_result = \3[_i](\2)\n    _noisy_q = laplace(_q_result, 4.0 / (\5))\n"
        r"    if _noisy_q >= _noisy_T:\n        \1 = _i\n        break",
    ),
]


def expand_macros(source: str) -> str:
    """Expand common DP pattern macros in source code.

    Replaces shorthand patterns (noisy_count, noisy_sum, etc.) with
    their expanded forms using primitive noise operations.

    Args:
        source: The source code to process.

    Returns:
        Source code with macros expanded.
    """
    result = source
    for pattern, replacement in _MACRO_PATTERNS:
        result = pattern.sub(replacement, result)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# HELPER INLINING
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class _HelperFunc:
    """A simple helper function that can be inlined.

    Attributes:
        name:   Function name.
        params: Parameter names.
        body:   Function body as source text.
        node:   AST node for the function definition.
    """

    name: str
    params: list[str]
    body: str
    node: ast.FunctionDef


def _is_simple_helper(func: ast.FunctionDef) -> bool:
    """Check if a function is simple enough to inline.

    A function is inlinable if:
    - It has no decorators
    - It has at most 5 statements
    - It contains no loops or nested functions
    - It contains no noise/query calls
    """
    if func.decorator_list:
        return False
    if len(func.body) > 5:
        return False

    for node in ast.walk(func):
        if isinstance(node, (ast.For, ast.While, ast.FunctionDef, ast.ClassDef)):
            return False
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in (
                    "laplace", "gaussian", "exponential_mechanism",
                    "query", "count_query", "sum_query",
                ):
                    return False
    return True


def _extract_helpers(source: str) -> tuple[list[_HelperFunc], ast.FunctionDef | None]:
    """Extract helper functions and the main mechanism function.

    Returns:
        A tuple of (helpers, main_func).
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [], None

    helpers: list[_HelperFunc] = []
    main_func: ast.FunctionDef | None = None

    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]

    for func in functions:
        # Check if this is the mechanism function
        is_mechanism = any(
            (isinstance(d, ast.Name) and d.id == "dp_mechanism")
            or (isinstance(d, ast.Call) and isinstance(d.func, ast.Name)
                and d.func.id == "dp_mechanism")
            for d in func.decorator_list
        )
        if is_mechanism:
            main_func = func
            continue

        if _is_simple_helper(func):
            params = [arg.arg for arg in func.args.args]
            body = ast.get_source_segment(source, func)
            if body is not None:
                helpers.append(_HelperFunc(
                    name=func.name,
                    params=params,
                    body=body,
                    node=func,
                ))

    if main_func is None and functions:
        # Last function is the mechanism
        main_func = functions[-1]

    return helpers, main_func


def inline_helpers(source: str) -> str:
    """Inline simple helper functions into the mechanism.

    Replaces calls to simple helpers with their body, substituting
    parameter names.

    Args:
        source: The source code to process.

    Returns:
        Source code with helpers inlined.
    """
    helpers, main_func = _extract_helpers(source)
    if not helpers:
        return source

    result = source
    for helper in helpers:
        # Build a pattern to match calls to this helper
        param_pattern = r",\s*".join(r"([^,)]+)" for _ in helper.params)
        call_pattern = re.compile(
            rf"(\w+)\s*=\s*{re.escape(helper.name)}\s*\(\s*{param_pattern}\s*\)"
        )

        def _replace(m: re.Match[str], h: _HelperFunc = helper) -> str:
            """Replace a helper call with inlined body."""
            target = m.group(1)
            args = [m.group(i + 2).strip() for i in range(len(h.params))]
            # Extract the return expression from the helper
            for node in ast.walk(h.node):
                if isinstance(node, ast.Return) and node.value is not None:
                    ret_src = ast.get_source_segment(
                        source, node.value
                    )
                    if ret_src:
                        inlined = ret_src
                        for param, arg in zip(h.params, args):
                            inlined = re.sub(
                                rf"\b{re.escape(param)}\b", arg, inlined
                            )
                        return f"{target} = {inlined}"
            return m.group(0)

        result = call_pattern.sub(_replace, result)

    return result


# ═══════════════════════════════════════════════════════════════════════════
# CODE NORMALIZATION
# ═══════════════════════════════════════════════════════════════════════════


def normalize(source: str) -> str:
    """Normalize DPImp source code to canonical form.

    Transformations:
    - Expand elif chains to nested if/else
    - Normalize whitespace
    - Remove trailing whitespace
    - Ensure final newline

    Args:
        source: The source code to normalize.

    Returns:
        Normalized source code.
    """
    # Remove trailing whitespace from lines
    lines = source.splitlines()
    lines = [line.rstrip() for line in lines]

    # Remove leading blank lines
    while lines and not lines[0].strip():
        lines.pop(0)

    # Remove trailing blank lines
    while lines and not lines[-1].strip():
        lines.pop()

    # Ensure final newline
    result = "\n".join(lines)
    if result and not result.endswith("\n"):
        result += "\n"

    return result


def expand_elif_chains(source: str) -> str:
    """Expand elif chains into nested if/else blocks.

    Converts::

        if a:
            ...
        elif b:
            ...
        else:
            ...

    Into::

        if a:
            ...
        else:
            if b:
                ...
            else:
                ...

    Args:
        source: The source code to transform.

    Returns:
        Source code with elif chains expanded.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    # Check if there are any elif chains
    has_elif = False
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and node.orelse:
            if (
                len(node.orelse) == 1
                and isinstance(node.orelse[0], ast.If)
            ):
                has_elif = True
                break

    if not has_elif:
        return source

    # Use ast.unparse (Python 3.9+) if available
    try:
        return ast.unparse(tree)
    except AttributeError:
        return source


# ═══════════════════════════════════════════════════════════════════════════
# RUNTIME STRIPPING
# ═══════════════════════════════════════════════════════════════════════════


def strip_runtime(source: str) -> str:
    """Strip non-essential runtime code from the source.

    Removes:
    - Logging calls (print, logging.*)
    - Assertions (assert statements)
    - Type checking guards (isinstance checks outside if conditions)
    - Docstrings (module-level and standalone string expressions)
    - __main__ blocks

    Args:
        source: The source code to process.

    Returns:
        Stripped source code.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    lines = source.splitlines(keepends=True)
    lines_to_remove: set[int] = set()

    for node in ast.iter_child_nodes(tree):
        # Remove if __name__ == '__main__': blocks
        if isinstance(node, ast.If):
            if _is_main_guard(node):
                for line in range(
                    node.lineno, (node.end_lineno or node.lineno) + 1
                ):
                    lines_to_remove.add(line)
                continue

        # Remove standalone print/logging calls
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            name = _call_name(node.value)
            if name in ("print", "logging", "log", "debug", "info", "warning"):
                for line in range(
                    node.lineno, (node.end_lineno or node.lineno) + 1
                ):
                    lines_to_remove.add(line)
                continue

        # Remove standalone docstrings
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            if isinstance(node.value.value, str):
                for line in range(
                    node.lineno, (node.end_lineno or node.lineno) + 1
                ):
                    lines_to_remove.add(line)
                continue

        # Remove assert statements at module level
        if isinstance(node, ast.Assert):
            for line in range(
                node.lineno, (node.end_lineno or node.lineno) + 1
            ):
                lines_to_remove.add(line)

    # Also strip asserts and prints inside functions
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for stmt in node.body:
                if isinstance(stmt, ast.Assert):
                    for line in range(
                        stmt.lineno, (stmt.end_lineno or stmt.lineno) + 1
                    ):
                        lines_to_remove.add(line)
                if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                    name = _call_name(stmt.value)
                    if name in (
                        "print", "logging", "log", "debug", "info", "warning"
                    ):
                        for line in range(
                            stmt.lineno,
                            (stmt.end_lineno or stmt.lineno) + 1,
                        ):
                            lines_to_remove.add(line)

    result_lines = [
        line for i, line in enumerate(lines, 1) if i not in lines_to_remove
    ]
    return "".join(result_lines)


def _is_main_guard(node: ast.If) -> bool:
    """Check if an if-statement is ``if __name__ == '__main__':``."""
    test = node.test
    if isinstance(test, ast.Compare):
        if (
            len(test.ops) == 1
            and isinstance(test.ops[0], ast.Eq)
            and isinstance(test.left, ast.Name)
            and test.left.id == "__name__"
            and len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value == "__main__"
        ):
            return True
    return False


def _call_name(node: ast.Call) -> str | None:
    """Extract the function name from a Call node."""
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


# ═══════════════════════════════════════════════════════════════════════════
# PREPROCESSING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class PreprocessResult:
    """Result of the preprocessing pipeline.

    Attributes:
        source:    The preprocessed source code.
        imports:   Resolved import information.
        name_map:  Mapping of import aliases to canonical names.
        warnings:  Any warnings generated.
    """

    source: str
    imports: list[ImportInfo] = field(default_factory=list)
    name_map: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def preprocess(
    source: str,
    *,
    resolve_imports_flag: bool = True,
    expand_macros_flag: bool = True,
    inline_flag: bool = True,
    normalize_flag: bool = True,
    strip_flag: bool = True,
) -> PreprocessResult:
    """Run the full preprocessing pipeline on DPImp source code.

    Args:
        source:               Raw source code.
        resolve_imports_flag:  Whether to resolve imports.
        expand_macros_flag:    Whether to expand macros.
        inline_flag:           Whether to inline helpers.
        normalize_flag:        Whether to normalize whitespace.
        strip_flag:            Whether to strip runtime code.

    Returns:
        A :class:`PreprocessResult` with the processed source.
    """
    result = PreprocessResult(source=source)

    # Step 1: Strip runtime code
    if strip_flag:
        result.source = strip_runtime(result.source)

    # Step 2: Resolve imports
    if resolve_imports_flag:
        resolver = ImportResolver()
        result.source, result.imports = resolver.resolve(result.source)
        result.name_map = dict(resolver.name_map)
        result.warnings.extend(resolver.warnings)

    # Step 3: Expand macros
    if expand_macros_flag:
        result.source = expand_macros(result.source)

    # Step 4: Inline helpers
    if inline_flag:
        result.source = inline_helpers(result.source)

    # Step 5: Normalize
    if normalize_flag:
        result.source = normalize(result.source)

    return result
