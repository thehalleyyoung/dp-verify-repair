"""Source-location tracking for MechIR ↔ source mapping.

Maps IR node IDs back to their originating source locations so that
error messages, counterexample reports, and repair suggestions can
point users at the exact source lines that matter.

Classes
-------
SourceRange  – file + line/column span
SourceMap    – bidirectional mapping between IR nodes and source ranges
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Sequence

from dpcegar.utils.errors import SourceLoc


# ═══════════════════════════════════════════════════════════════════════════
# SOURCE RANGE
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class SourceRange:
    """A contiguous range of source text.

    Attributes:
        file:       Path to the source file.
        start_line: 1-based starting line number.
        start_col:  1-based starting column number.
        end_line:   1-based ending line number (inclusive).
        end_col:    1-based ending column number (inclusive).
    """

    file: str = "<unknown>"
    start_line: int = 0
    start_col: int = 0
    end_line: int = 0
    end_col: int = 0

    # ------------------------------------------------------------------
    # Conversions
    # ------------------------------------------------------------------

    def to_source_loc(self) -> SourceLoc:
        """Convert to a :class:`SourceLoc` for attaching to IR nodes."""
        return SourceLoc(
            file=self.file,
            line=self.start_line,
            col=self.start_col,
            end_line=self.end_line,
            end_col=self.end_col,
        )

    @classmethod
    def from_source_loc(cls, loc: SourceLoc) -> SourceRange:
        """Construct from a :class:`SourceLoc`."""
        return cls(
            file=loc.file,
            start_line=loc.line,
            start_col=loc.col,
            end_line=loc.end_line if loc.end_line is not None else loc.line,
            end_col=loc.end_col if loc.end_col is not None else loc.col,
        )

    @classmethod
    def from_line(cls, file: str, line: int) -> SourceRange:
        """Create a range covering a single line."""
        return cls(file=file, start_line=line, start_col=1, end_line=line, end_col=1)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def contains_line(self, line: int) -> bool:
        """Return True if *line* falls within this range."""
        return self.start_line <= line <= self.end_line

    def overlaps(self, other: SourceRange) -> bool:
        """Return True if this range overlaps with *other*."""
        if self.file != other.file:
            return False
        return self.start_line <= other.end_line and other.start_line <= self.end_line

    def merge(self, other: SourceRange) -> SourceRange:
        """Return the smallest range enclosing both *self* and *other*.

        Both ranges must refer to the same file.
        """
        if self.file != other.file:
            raise ValueError(
                f"Cannot merge ranges from different files: "
                f"{self.file!r} vs {other.file!r}"
            )
        if self.start_line < other.start_line or (
            self.start_line == other.start_line and self.start_col <= other.start_col
        ):
            s_line, s_col = self.start_line, self.start_col
        else:
            s_line, s_col = other.start_line, other.start_col
        if self.end_line > other.end_line or (
            self.end_line == other.end_line and self.end_col >= other.end_col
        ):
            e_line, e_col = self.end_line, self.end_col
        else:
            e_line, e_col = other.end_line, other.end_col
        return SourceRange(
            file=self.file,
            start_line=s_line,
            start_col=s_col,
            end_line=e_line,
            end_col=e_col,
        )

    @property
    def line_count(self) -> int:
        """Number of lines spanned (inclusive)."""
        return max(self.end_line - self.start_line + 1, 1)

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "file": self.file,
            "start_line": self.start_line,
            "start_col": self.start_col,
            "end_line": self.end_line,
            "end_col": self.end_col,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SourceRange:
        """Deserialize from a dictionary."""
        return cls(
            file=d.get("file", "<unknown>"),
            start_line=d.get("start_line", 0),
            start_col=d.get("start_col", 0),
            end_line=d.get("end_line", 0),
            end_col=d.get("end_col", 0),
        )

    def __str__(self) -> str:
        """Human-readable representation."""
        if self.start_line == self.end_line:
            return f"{self.file}:{self.start_line}:{self.start_col}-{self.end_col}"
        return (
            f"{self.file}:{self.start_line}:{self.start_col}"
            f"-{self.end_line}:{self.end_col}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# SOURCE MAP
# ═══════════════════════════════════════════════════════════════════════════


class SourceMap:
    """Bidirectional mapping between IR node IDs and source ranges.

    Usage::

        smap = SourceMap()
        smap.add(node.node_id, SourceRange("mech.py", 10, 5, 10, 30))
        loc = smap.get(node.node_id)
    """

    def __init__(self) -> None:
        """Initialize an empty source map."""
        self._node_to_range: dict[int, SourceRange] = {}
        self._source_lines: dict[str, list[str]] = {}

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, node_id: int, source_range: SourceRange) -> None:
        """Register a mapping from *node_id* to *source_range*."""
        self._node_to_range[node_id] = source_range

    def add_from_loc(self, node_id: int, loc: SourceLoc) -> None:
        """Register a mapping from *node_id* using a :class:`SourceLoc`."""
        self._node_to_range[node_id] = SourceRange.from_source_loc(loc)

    def set_source_text(self, file: str, text: str) -> None:
        """Store the full source text for a file (for snippet extraction)."""
        self._source_lines[file] = text.splitlines()

    def remove(self, node_id: int) -> None:
        """Remove the mapping for *node_id*, if present."""
        self._node_to_range.pop(node_id, None)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get(self, node_id: int) -> SourceRange | None:
        """Return the source range for *node_id*, or ``None``."""
        return self._node_to_range.get(node_id)

    def get_loc(self, node_id: int) -> SourceLoc | None:
        """Return a :class:`SourceLoc` for *node_id*, or ``None``."""
        sr = self._node_to_range.get(node_id)
        return sr.to_source_loc() if sr else None

    def nodes_in_range(self, source_range: SourceRange) -> list[int]:
        """Return node IDs whose ranges overlap with *source_range*."""
        return [
            nid
            for nid, sr in self._node_to_range.items()
            if sr.overlaps(source_range)
        ]

    def nodes_at_line(self, file: str, line: int) -> list[int]:
        """Return node IDs that span the given *line*."""
        return [
            nid
            for nid, sr in self._node_to_range.items()
            if sr.file == file and sr.contains_line(line)
        ]

    def all_node_ids(self) -> list[int]:
        """Return all registered node IDs."""
        return list(self._node_to_range.keys())

    def __len__(self) -> int:
        """Number of registered mappings."""
        return len(self._node_to_range)

    def __contains__(self, node_id: int) -> bool:
        """True if *node_id* has a registered mapping."""
        return node_id in self._node_to_range

    # ------------------------------------------------------------------
    # Snippet extraction
    # ------------------------------------------------------------------

    def get_snippet(
        self,
        node_id: int,
        context_lines: int = 2,
    ) -> str | None:
        """Extract a source snippet for the given node.

        Args:
            node_id:       The IR node ID.
            context_lines: Number of lines of context above and below.

        Returns:
            A formatted snippet string, or ``None`` if no mapping or
            source text is available.
        """
        sr = self._node_to_range.get(node_id)
        if sr is None:
            return None
        return self.get_snippet_for_range(sr, context_lines=context_lines)

    def get_snippet_for_range(
        self,
        source_range: SourceRange,
        context_lines: int = 2,
    ) -> str | None:
        """Extract a source snippet for a given range.

        Args:
            source_range:  The source range to extract.
            context_lines: Lines of context above and below.

        Returns:
            A formatted snippet string with line numbers and underline
            markers, or ``None`` if source text is unavailable.
        """
        lines = self._source_lines.get(source_range.file)
        if not lines:
            return None

        start = max(source_range.start_line - context_lines - 1, 0)
        end = min(source_range.end_line + context_lines, len(lines))

        max_lineno_width = len(str(end + 1))
        result_lines: list[str] = []

        for idx in range(start, end):
            lineno = idx + 1
            prefix = f"{lineno:>{max_lineno_width}} | "
            line_text = lines[idx] if idx < len(lines) else ""
            marker = ""
            if source_range.start_line <= lineno <= source_range.end_line:
                prefix = f"{lineno:>{max_lineno_width}} > "
                if lineno == source_range.start_line:
                    col_start = max(source_range.start_col - 1, 0)
                    if lineno == source_range.end_line:
                        col_end = source_range.end_col
                    else:
                        col_end = len(line_text)
                    underline_pad = " " * (max_lineno_width + 3 + col_start)
                    underline_len = max(col_end - col_start, 1)
                    marker = f"\n{underline_pad}{'~' * underline_len}"
                elif lineno == source_range.end_line:
                    col_end = source_range.end_col
                    underline_pad = " " * (max_lineno_width + 3)
                    marker = f"\n{underline_pad}{'~' * max(col_end, 1)}"
            result_lines.append(f"{prefix}{line_text}{marker}")

        return "\n".join(result_lines)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize the source map to a JSON-compatible dictionary."""
        return {
            "mappings": {
                str(nid): sr.to_dict()
                for nid, sr in self._node_to_range.items()
            }
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SourceMap:
        """Deserialize a source map from a dictionary."""
        smap = cls()
        for nid_str, sr_dict in d.get("mappings", {}).items():
            smap._node_to_range[int(nid_str)] = SourceRange.from_dict(sr_dict)
        return smap

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, text: str) -> SourceMap:
        """Deserialize from a JSON string."""
        return cls.from_dict(json.loads(text))

    def __repr__(self) -> str:
        return f"SourceMap({len(self._node_to_range)} mappings)"
