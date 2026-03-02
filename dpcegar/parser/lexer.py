"""Token definitions and lexer for the DPImp language.

DPImp is a Python subset tailored for differential privacy mechanisms.
This module provides the tokeniser that is used by the AST bridge and
preprocessor to convert raw source text into structured token streams.

Classes
-------
TokenType  – enumeration of all token kinds
Token      – single token with type, value, and source position
Lexer      – stateful tokeniser for DPImp source
"""

from __future__ import annotations

import keyword
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Iterator, Sequence

from dpcegar.utils.errors import LexError, SourceLoc


# ═══════════════════════════════════════════════════════════════════════════
# TOKEN TYPE ENUMERATION
# ═══════════════════════════════════════════════════════════════════════════


class TokenType(Enum):
    """All token kinds recognised by the DPImp lexer."""

    # -- Literals --
    IDENT = auto()
    INT_LIT = auto()
    FLOAT_LIT = auto()
    STRING_LIT = auto()
    BOOL_TRUE = auto()
    BOOL_FALSE = auto()
    NONE_LIT = auto()

    # -- Arithmetic operators --
    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    SLASH = auto()
    DSTAR = auto()       # **
    PERCENT = auto()     # %

    # -- Comparison operators --
    EQ = auto()          # ==
    NEQ = auto()         # !=
    LT = auto()
    GT = auto()
    LE = auto()          # <=
    GE = auto()          # >=

    # -- Logical / boolean keywords --
    AND = auto()
    OR = auto()
    NOT = auto()

    # -- Assignment --
    ASSIGN = auto()      # =
    PLUS_ASSIGN = auto()  # +=
    MINUS_ASSIGN = auto()  # -=
    STAR_ASSIGN = auto()   # *=
    SLASH_ASSIGN = auto()  # /=

    # -- Delimiters --
    LPAREN = auto()
    RPAREN = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    LBRACE = auto()
    RBRACE = auto()
    COMMA = auto()
    COLON = auto()
    SEMICOLON = auto()
    DOT = auto()
    ARROW = auto()       # ->
    AT = auto()          # @

    # -- Keywords --
    KW_DEF = auto()
    KW_IF = auto()
    KW_ELIF = auto()
    KW_ELSE = auto()
    KW_FOR = auto()
    KW_IN = auto()
    KW_RANGE = auto()
    KW_RETURN = auto()
    KW_WHILE = auto()
    KW_IMPORT = auto()
    KW_FROM = auto()
    KW_AS = auto()
    KW_PASS = auto()
    KW_BREAK = auto()
    KW_CONTINUE = auto()
    KW_CLASS = auto()
    KW_LAMBDA = auto()

    # -- Noise primitives --
    NOISE_LAPLACE = auto()
    NOISE_GAUSSIAN = auto()
    NOISE_EXPONENTIAL = auto()

    # -- DP-specific decorators / annotations --
    DECORATOR_DP_MECHANISM = auto()
    DECORATOR_SENSITIVITY = auto()

    # -- Whitespace / structure --
    NEWLINE = auto()
    INDENT = auto()
    DEDENT = auto()

    # -- Special --
    COMMENT = auto()
    EOF = auto()

    def __str__(self) -> str:
        """Return a human-readable name for this token type."""
        return self.name


# ═══════════════════════════════════════════════════════════════════════════
# TOKEN DATACLASS
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class Token:
    """A single lexical token.

    Attributes:
        type:  The token kind.
        value: The raw textual value of the token.
        line:  1-based line number.
        col:   1-based column number.
        file:  Source file name (for diagnostics).
    """

    type: TokenType
    value: str
    line: int
    col: int
    file: str = "<unknown>"

    def source_loc(self) -> SourceLoc:
        """Convert to a :class:`SourceLoc`."""
        return SourceLoc(
            file=self.file,
            line=self.line,
            col=self.col,
        )

    def is_type(self, *types: TokenType) -> bool:
        """Return True if this token matches any of *types*."""
        return self.type in types

    def __str__(self) -> str:
        return f"Token({self.type}, {self.value!r}, {self.line}:{self.col})"


# ═══════════════════════════════════════════════════════════════════════════
# KEYWORD AND SYMBOL TABLES
# ═══════════════════════════════════════════════════════════════════════════

_KEYWORDS: dict[str, TokenType] = {
    "def": TokenType.KW_DEF,
    "if": TokenType.KW_IF,
    "elif": TokenType.KW_ELIF,
    "else": TokenType.KW_ELSE,
    "for": TokenType.KW_FOR,
    "in": TokenType.KW_IN,
    "range": TokenType.KW_RANGE,
    "return": TokenType.KW_RETURN,
    "while": TokenType.KW_WHILE,
    "True": TokenType.BOOL_TRUE,
    "False": TokenType.BOOL_FALSE,
    "None": TokenType.NONE_LIT,
    "and": TokenType.AND,
    "or": TokenType.OR,
    "not": TokenType.NOT,
    "import": TokenType.KW_IMPORT,
    "from": TokenType.KW_FROM,
    "as": TokenType.KW_AS,
    "pass": TokenType.KW_PASS,
    "break": TokenType.KW_BREAK,
    "continue": TokenType.KW_CONTINUE,
    "class": TokenType.KW_CLASS,
    "lambda": TokenType.KW_LAMBDA,
}

_NOISE_PRIMITIVES: dict[str, TokenType] = {
    "laplace": TokenType.NOISE_LAPLACE,
    "gaussian": TokenType.NOISE_GAUSSIAN,
    "exponential_mechanism": TokenType.NOISE_EXPONENTIAL,
}

# Multi-character operators (longest match first)
_MULTI_OPS: list[tuple[str, TokenType]] = [
    ("**", TokenType.DSTAR),
    ("->", TokenType.ARROW),
    ("==", TokenType.EQ),
    ("!=", TokenType.NEQ),
    ("<=", TokenType.LE),
    (">=", TokenType.GE),
    ("+=", TokenType.PLUS_ASSIGN),
    ("-=", TokenType.MINUS_ASSIGN),
    ("*=", TokenType.STAR_ASSIGN),
    ("/=", TokenType.SLASH_ASSIGN),
]

_SINGLE_OPS: dict[str, TokenType] = {
    "+": TokenType.PLUS,
    "-": TokenType.MINUS,
    "*": TokenType.STAR,
    "/": TokenType.SLASH,
    "%": TokenType.PERCENT,
    "<": TokenType.LT,
    ">": TokenType.GT,
    "=": TokenType.ASSIGN,
    "(": TokenType.LPAREN,
    ")": TokenType.RPAREN,
    "[": TokenType.LBRACKET,
    "]": TokenType.RBRACKET,
    "{": TokenType.LBRACE,
    "}": TokenType.RBRACE,
    ",": TokenType.COMMA,
    ":": TokenType.COLON,
    ";": TokenType.SEMICOLON,
    ".": TokenType.DOT,
    "@": TokenType.AT,
}


# ═══════════════════════════════════════════════════════════════════════════
# LEXER
# ═══════════════════════════════════════════════════════════════════════════


# Pre-compiled patterns for hot inner loop
_RE_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_RE_FLOAT = re.compile(r"\d+\.\d*([eE][+-]?\d+)?|\d+[eE][+-]?\d+|\.\d+([eE][+-]?\d+)?")
_RE_INT = re.compile(r"\d+")
_RE_STRING_DQ = re.compile(r'"(?:[^"\\]|\\.)*"')
_RE_STRING_SQ = re.compile(r"'(?:[^'\\]|\\.)*'")
_RE_TRIPLE_DQ = re.compile(r'"""[\s\S]*?"""')
_RE_TRIPLE_SQ = re.compile(r"'''[\s\S]*?'''")


class Lexer:
    """Tokeniser for DPImp source code.

    Supports the Python subset used by DP mechanism definitions, plus
    DP-specific noise primitives and decorator annotations.

    Usage::

        lexer = Lexer(source_code, file="mech.py")
        tokens = lexer.tokenize()

    Attributes:
        source:   The full source string.
        file:     Name of the source file for diagnostics.
        tokens:   Resulting token list (populated after :meth:`tokenize`).
        errors:   Any lexer errors encountered.
    """

    def __init__(self, source: str, file: str = "<unknown>") -> None:
        """Initialize the lexer.

        Args:
            source: The DPImp source code to tokenise.
            file:   Source file name for error messages.
        """
        self.source: str = source
        self.file: str = file
        self.tokens: list[Token] = []
        self.errors: list[LexError] = []

        self._pos: int = 0
        self._line: int = 1
        self._col: int = 1
        self._indent_stack: list[int] = [0]
        self._paren_depth: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tokenize(self) -> list[Token]:
        """Tokenise the entire source and return the token list.

        Returns:
            List of :class:`Token` instances, ending with ``EOF``.

        Raises:
            LexError: If a fatal lexing error is encountered and
                      :attr:`errors` is not being checked.
        """
        self.tokens.clear()
        self.errors.clear()
        self._pos = 0
        self._line = 1
        self._col = 1
        self._indent_stack = [0]
        self._paren_depth = 0

        while self._pos < len(self.source):
            self._scan_token()

        # Emit DEDENT tokens for remaining indentation levels
        while len(self._indent_stack) > 1:
            self._indent_stack.pop()
            self._emit(TokenType.DEDENT, "")

        self._emit(TokenType.EOF, "")
        return self.tokens

    def tokenize_iter(self) -> Iterator[Token]:
        """Lazily tokenise the source, yielding tokens one at a time."""
        for tok in self.tokenize():
            yield tok

    # ------------------------------------------------------------------
    # Token creation helpers
    # ------------------------------------------------------------------

    def _emit(self, ttype: TokenType, value: str) -> Token:
        """Create and append a token to the list."""
        tok = Token(
            type=ttype,
            value=value,
            line=self._line,
            col=self._col,
            file=self.file,
        )
        self.tokens.append(tok)
        return tok

    def _emit_at(self, ttype: TokenType, value: str, line: int, col: int) -> Token:
        """Create and append a token at a specific position."""
        tok = Token(type=ttype, value=value, line=line, col=col, file=self.file)
        self.tokens.append(tok)
        return tok

    def _error(self, message: str) -> None:
        """Record a lex error at the current position."""
        loc = SourceLoc(file=self.file, line=self._line, col=self._col)
        self.errors.append(LexError(message, source_loc=loc))

    # ------------------------------------------------------------------
    # Character access
    # ------------------------------------------------------------------

    def _peek(self, offset: int = 0) -> str:
        """Return the character at current position + offset, or ''."""
        idx = self._pos + offset
        if idx < len(self.source):
            return self.source[idx]
        return ""

    def _advance(self, count: int = 1) -> str:
        """Advance position by *count* characters, updating line/col."""
        result = self.source[self._pos: self._pos + count]
        for ch in result:
            if ch == "\n":
                self._line += 1
                self._col = 1
            else:
                self._col += 1
        self._pos += count
        return result

    def _match(self, text: str) -> bool:
        """Return True and advance if the source matches *text* at pos."""
        if self.source[self._pos: self._pos + len(text)] == text:
            self._advance(len(text))
            return True
        return False

    def _remaining(self) -> str:
        """Return the un-consumed portion of the source."""
        return self.source[self._pos:]

    # ------------------------------------------------------------------
    # Main scanner dispatch
    # ------------------------------------------------------------------

    def _scan_token(self) -> None:
        """Scan and emit the next token."""
        # Handle start-of-line indentation
        if self._col == 1 and self._paren_depth == 0:
            self._handle_indentation()
            if self._pos >= len(self.source):
                return

        ch = self._peek()

        # Skip spaces/tabs (not at line start)
        if ch in (" ", "\t") and self._col > 1:
            self._advance()
            return

        # Newline
        if ch == "\n":
            if self._paren_depth == 0:
                self._emit(TokenType.NEWLINE, "\\n")
            self._advance()
            return

        # Carriage return (handle \r\n)
        if ch == "\r":
            self._advance()
            if self._peek() == "\n":
                self._advance()
            if self._paren_depth == 0:
                self._emit(TokenType.NEWLINE, "\\n")
            return

        # Comments
        if ch == "#":
            comment = self._scan_comment()
            self._emit(TokenType.COMMENT, comment)
            return

        # Triple-quoted strings
        if self.source[self._pos: self._pos + 3] in ('"""', "'''"):
            self._scan_triple_string()
            return

        # Strings
        if ch in ('"', "'"):
            self._scan_string(ch)
            return

        # Numbers
        if ch.isdigit() or (ch == "." and self._peek(1).isdigit()):
            self._scan_number()
            return

        # Identifiers / keywords
        if ch.isalpha() or ch == "_":
            self._scan_identifier()
            return

        # Multi-character operators
        for op_str, op_type in _MULTI_OPS:
            if self.source[self._pos: self._pos + len(op_str)] == op_str:
                start_col = self._col
                self._advance(len(op_str))
                self._emit_at(op_type, op_str, self._line, start_col)
                return

        # Single-character operators / delimiters
        if ch in _SINGLE_OPS:
            start_col = self._col
            self._advance()
            tt = _SINGLE_OPS[ch]
            if tt == TokenType.LPAREN:
                self._paren_depth += 1
            elif tt == TokenType.RPAREN:
                self._paren_depth = max(0, self._paren_depth - 1)
            elif tt == TokenType.LBRACKET:
                self._paren_depth += 1
            elif tt == TokenType.RBRACKET:
                self._paren_depth = max(0, self._paren_depth - 1)
            elif tt == TokenType.LBRACE:
                self._paren_depth += 1
            elif tt == TokenType.RBRACE:
                self._paren_depth = max(0, self._paren_depth - 1)
            self._emit_at(tt, ch, self._line, start_col)
            return

        # Backslash line continuation
        if ch == "\\":
            self._advance()
            if self._peek() == "\n":
                self._advance()
            return

        # Unknown character
        self._error(f"unexpected character: {ch!r}")
        self._advance()

    # ------------------------------------------------------------------
    # Indentation handling
    # ------------------------------------------------------------------

    def _handle_indentation(self) -> None:
        """Process indentation at the start of a line."""
        indent = 0
        while self._pos < len(self.source) and self.source[self._pos] in (" ", "\t"):
            if self.source[self._pos] == "\t":
                indent = (indent // 4 + 1) * 4
            else:
                indent += 1
            self._pos += 1
            self._col += 1

        # Blank line or comment-only line: skip indentation processing
        if self._pos >= len(self.source) or self.source[self._pos] in ("\n", "\r", "#"):
            return

        current = self._indent_stack[-1]
        if indent > current:
            self._indent_stack.append(indent)
            self._emit(TokenType.INDENT, "")
        elif indent < current:
            while len(self._indent_stack) > 1 and self._indent_stack[-1] > indent:
                self._indent_stack.pop()
                self._emit(TokenType.DEDENT, "")
            if self._indent_stack[-1] != indent:
                self._error(
                    f"inconsistent indentation: expected {self._indent_stack[-1]} "
                    f"spaces, got {indent}"
                )

    # ------------------------------------------------------------------
    # Specific scanners
    # ------------------------------------------------------------------

    def _scan_comment(self) -> str:
        """Scan a comment from '#' to end of line."""
        start = self._pos
        while self._pos < len(self.source) and self.source[self._pos] != "\n":
            self._pos += 1
            self._col += 1
        return self.source[start: self._pos]

    def _scan_string(self, quote: str) -> None:
        """Scan a single-quoted or double-quoted string literal."""
        start_line = self._line
        start_col = self._col
        self._advance()  # consume opening quote
        chars: list[str] = []
        while self._pos < len(self.source):
            ch = self._peek()
            if ch == "\\":
                self._advance()
                esc = self._peek()
                escape_map = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\"}
                chars.append(escape_map.get(esc, esc))
                self._advance()
            elif ch == quote:
                self._advance()
                val = "".join(chars)
                self._emit_at(TokenType.STRING_LIT, val, start_line, start_col)
                return
            elif ch == "\n":
                self._error("unterminated string literal")
                return
            else:
                chars.append(ch)
                self._advance()
        self._error("unterminated string literal at end of file")

    def _scan_triple_string(self) -> None:
        """Scan a triple-quoted string literal."""
        start_line = self._line
        start_col = self._col
        quote = self.source[self._pos: self._pos + 3]
        self._advance(3)
        chars: list[str] = []
        while self._pos < len(self.source):
            if self.source[self._pos: self._pos + 3] == quote:
                self._advance(3)
                val = "".join(chars)
                self._emit_at(TokenType.STRING_LIT, val, start_line, start_col)
                return
            ch = self._peek()
            if ch == "\\":
                self._advance()
                esc = self._peek()
                chars.append(esc)
                self._advance()
            else:
                chars.append(ch)
                self._advance()
        self._error("unterminated triple-quoted string at end of file")

    def _scan_number(self) -> None:
        """Scan an integer or floating-point literal."""
        start_line = self._line
        start_col = self._col
        remaining = self._remaining()

        # Try float first (more specific)
        m = _RE_FLOAT.match(remaining)
        if m:
            val = m.group()
            self._advance(len(val))
            self._emit_at(TokenType.FLOAT_LIT, val, start_line, start_col)
            return

        # Integer
        m = _RE_INT.match(remaining)
        if m:
            val = m.group()
            self._advance(len(val))
            # Check for float suffix (e.g. 1e5)
            if self._peek() == "." and self._peek(1).isdigit():
                self._advance()  # consume '.'
                m2 = _RE_INT.match(self._remaining())
                if m2:
                    frac = m2.group()
                    self._advance(len(frac))
                    val = val + "." + frac
                self._emit_at(TokenType.FLOAT_LIT, val, start_line, start_col)
            else:
                self._emit_at(TokenType.INT_LIT, val, start_line, start_col)
            return

        self._error(f"invalid number literal")
        self._advance()

    def _scan_identifier(self) -> None:
        """Scan an identifier, keyword, or noise primitive."""
        start_line = self._line
        start_col = self._col
        remaining = self._remaining()
        m = _RE_IDENT.match(remaining)
        if not m:
            self._error("invalid identifier")
            self._advance()
            return

        word = m.group()
        self._advance(len(word))

        # Check for DP decorator patterns (after @)
        if (
            self.tokens
            and self.tokens[-1].type == TokenType.AT
        ):
            if word == "dp_mechanism":
                self._emit_at(
                    TokenType.DECORATOR_DP_MECHANISM, word, start_line, start_col
                )
                return
            if word == "sensitivity":
                self._emit_at(
                    TokenType.DECORATOR_SENSITIVITY, word, start_line, start_col
                )
                return

        # Keywords
        if word in _KEYWORDS:
            self._emit_at(_KEYWORDS[word], word, start_line, start_col)
            return

        # Noise primitives
        if word in _NOISE_PRIMITIVES:
            self._emit_at(_NOISE_PRIMITIVES[word], word, start_line, start_col)
            return

        # Plain identifier
        self._emit_at(TokenType.IDENT, word, start_line, start_col)


# ═══════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════


def tokenize(source: str, file: str = "<unknown>") -> list[Token]:
    """Tokenise DPImp source code.

    Args:
        source: The source code string.
        file:   File name for diagnostics.

    Returns:
        List of tokens.

    Raises:
        LexError: On fatal lexing errors.
    """
    lexer = Lexer(source, file=file)
    tokens = lexer.tokenize()
    if lexer.errors:
        raise lexer.errors[0]
    return tokens


def filter_tokens(
    tokens: Sequence[Token],
    *,
    exclude: set[TokenType] | None = None,
    include: set[TokenType] | None = None,
) -> list[Token]:
    """Filter a token list by type.

    Args:
        tokens:  Input token list.
        exclude: Token types to remove (if provided).
        include: Token types to keep (if provided; overrides *exclude*).

    Returns:
        Filtered list of tokens.
    """
    if include is not None:
        return [t for t in tokens if t.type in include]
    if exclude is not None:
        return [t for t in tokens if t.type not in exclude]
    return list(tokens)


def strip_comments(tokens: Sequence[Token]) -> list[Token]:
    """Remove comment tokens from a token list."""
    return [t for t in tokens if t.type != TokenType.COMMENT]
