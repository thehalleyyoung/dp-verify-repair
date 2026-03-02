"""Code patch generation for differential privacy mechanism repair.

Applies repair templates to MechIR to produce repaired mechanisms and
generates human-readable source code patches in unified diff format.

Classes
-------
PatchGenerator  – apply templates and generate patches
PatchEntry      – single parameter change in a patch
PatchReport     – full report of a repair application
SourcePatcher   – source-level repair application
"""

from __future__ import annotations

import copy
import difflib
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

from dpcegar.ir.types import (
    BinOp,
    BinOpKind,
    Const,
    IRType,
    NoiseKind,
    TypedExpr,
    Var,
)
from dpcegar.ir.nodes import (
    MechIR,
    NoiseDrawNode,
    BranchNode,
    QueryNode,
    ReturnNode,
    LoopNode,
    IRNode,
    SequenceNode,
)
from dpcegar.repair.templates import (
    RepairTemplate,
    RepairSite,
    RepairParameter,
)
from dpcegar.utils.errors import InternalError, RepairError, ensure


# ═══════════════════════════════════════════════════════════════════════════
# PATCH ENTRY
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class PatchEntry:
    """A single parameter change in a repair patch.

    Attributes:
        parameter_name: Name of the modified parameter.
        site_node_id: IR node ID of the repair site.
        site_description: Human-readable site description.
        original_value: Value before repair.
        repaired_value: Value after repair.
        change_kind: Type of change (scale, threshold, etc.).
    """

    parameter_name: str
    site_node_id: int
    site_description: str = ""
    original_value: float = 0.0
    repaired_value: float = 0.0
    change_kind: str = "parameter_change"

    @property
    def absolute_change(self) -> float:
        """Return the absolute change in value."""
        return abs(self.repaired_value - self.original_value)

    @property
    def relative_change(self) -> float:
        """Return the relative change (ratio) in value."""
        if self.original_value == 0:
            return float("inf") if self.repaired_value != 0 else 0.0
        return abs(self.repaired_value - self.original_value) / abs(self.original_value)

    def __str__(self) -> str:
        return (
            f"{self.parameter_name}: {self.original_value:.6g} → "
            f"{self.repaired_value:.6g} ({self.change_kind})"
        )


# ═══════════════════════════════════════════════════════════════════════════
# PATCH REPORT
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class PatchReport:
    """Full report of a repair application.

    Attributes:
        mechanism_name: Name of the repaired mechanism.
        template_name: Name of the repair template used.
        entries: Individual parameter changes.
        original_source: Source code before repair.
        repaired_source: Source code after repair.
        unified_diff: Unified diff between original and repaired.
        total_cost: Total repair cost.
        is_valid: Whether the patch passed validation.
        validation_notes: Notes from validation.
    """

    mechanism_name: str = ""
    template_name: str = ""
    entries: list[PatchEntry] = field(default_factory=list)
    original_source: str = ""
    repaired_source: str = ""
    unified_diff: str = ""
    total_cost: float = 0.0
    is_valid: bool = True
    validation_notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            f"Patch Report: {self.mechanism_name}",
            f"  Template: {self.template_name}",
            f"  Changes: {len(self.entries)}",
            f"  Cost: {self.total_cost:.4f}",
            f"  Valid: {self.is_valid}",
        ]
        for entry in self.entries:
            lines.append(f"    {entry}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


# ═══════════════════════════════════════════════════════════════════════════
# SOURCE CODE GENERATOR
# ═══════════════════════════════════════════════════════════════════════════


class MechIRPrinter:
    """Pretty-print a MechIR as pseudo-source code.

    Generates a readable representation of the mechanism that can
    be used for diffing before/after repair.
    """

    def __init__(self, indent_width: int = 2) -> None:
        """Initialise with indentation width.

        Args:
            indent_width: Number of spaces per indentation level.
        """
        self._indent = indent_width

    def print(self, mechanism: MechIR) -> str:
        """Generate source code for a mechanism.

        Args:
            mechanism: The mechanism IR.

        Returns:
            Multi-line source code string.
        """
        lines: list[str] = []

        params = ", ".join(str(p) for p in mechanism.params)
        budget_str = f"  // budget: {mechanism.budget}" if mechanism.budget else ""
        lines.append(f"mechanism {mechanism.name}({params}) -> {mechanism.return_type} {{{budget_str}")

        self._print_node(mechanism.body, lines, level=1)

        lines.append("}")
        return "\n".join(lines)

    def _print_node(self, node: IRNode, lines: list[str], level: int) -> None:
        """Recursively print an IR node.

        Args:
            node: The IR node to print.
            lines: Output line accumulator.
            level: Current indentation level.
        """
        indent = " " * (self._indent * level)

        if isinstance(node, SequenceNode):
            for stmt in node.stmts:
                self._print_node(stmt, lines, level)

        elif isinstance(node, NoiseDrawNode):
            sens = f"  // sensitivity={node.sensitivity}" if node.sensitivity else ""
            lines.append(
                f"{indent}{node.target} ~ {node.noise_kind.name}("
                f"center={node.center}, scale={node.scale}){sens}"
            )

        elif isinstance(node, BranchNode):
            lines.append(f"{indent}if ({node.condition}) {{")
            self._print_node(node.true_branch, lines, level + 1)
            lines.append(f"{indent}}} else {{")
            self._print_node(node.false_branch, lines, level + 1)
            lines.append(f"{indent}}}")

        elif isinstance(node, LoopNode):
            budget_ann = node.get_annotation("per_iteration_budget")
            budget_str = f"  // per_iter_budget={budget_ann}" if budget_ann else ""
            lines.append(
                f"{indent}for {node.index_var} in range({node.bound}) {{{budget_str}"
            )
            self._print_node(node.body, lines, level + 1)
            lines.append(f"{indent}}}")

        elif isinstance(node, QueryNode):
            args = ", ".join(str(a) for a in node.args)
            lines.append(
                f"{indent}{node.target} = query {node.query_name}({args})"
                f"  // Δ={node.sensitivity}"
            )

        elif isinstance(node, ReturnNode):
            lines.append(f"{indent}return {node.value}")

        else:
            s = str(node)
            if s and s != "noop":
                lines.append(f"{indent}{s}")


# ═══════════════════════════════════════════════════════════════════════════
# PATCH GENERATOR
# ═══════════════════════════════════════════════════════════════════════════


class PatchGenerator:
    """Apply repair templates and generate human-readable patches.

    Orchestrates the application of a repair template to a mechanism,
    generates before/after source code, and produces a unified diff.
    """

    def __init__(self) -> None:
        """Initialise the patch generator."""
        self._printer = MechIRPrinter()

    def generate(
        self,
        mechanism: MechIR,
        template: RepairTemplate,
        parameter_values: dict[str, float],
        original_values: dict[str, float] | None = None,
    ) -> PatchReport:
        """Generate a patch report for a repair.

        Args:
            mechanism: The original mechanism.
            template: The repair template applied.
            parameter_values: Concrete repair parameter values.
            original_values: Original parameter values for comparison.

        Returns:
            A PatchReport with full before/after source and diff.
        """
        orig_values = original_values or self._extract_originals(
            mechanism, template
        )

        repaired = template.apply_concrete(mechanism, parameter_values)

        entries = self._build_entries(template, parameter_values, orig_values)

        original_source = self._printer.print(mechanism)
        repaired_source = self._printer.print(repaired)

        diff = self._generate_diff(
            original_source, repaired_source, mechanism.name
        )

        total_cost = sum(e.absolute_change for e in entries)

        report = PatchReport(
            mechanism_name=mechanism.name,
            template_name=template.name(),
            entries=entries,
            original_source=original_source,
            repaired_source=repaired_source,
            unified_diff=diff,
            total_cost=total_cost,
        )

        validation = self._validate_patch(mechanism, repaired, template)
        report.is_valid = validation[0]
        report.validation_notes = validation[1]

        return report

    def apply_and_diff(
        self,
        mechanism: MechIR,
        template: RepairTemplate,
        parameter_values: dict[str, float],
    ) -> tuple[MechIR, str]:
        """Apply a repair and return the diff.

        Args:
            mechanism: Original mechanism.
            template: Repair template.
            parameter_values: Parameter values.

        Returns:
            Tuple of (repaired mechanism, unified diff string).
        """
        repaired = template.apply_concrete(mechanism, parameter_values)
        original_source = self._printer.print(mechanism)
        repaired_source = self._printer.print(repaired)
        diff = self._generate_diff(
            original_source, repaired_source, mechanism.name
        )
        return repaired, diff

    def _build_entries(
        self,
        template: RepairTemplate,
        values: dict[str, float],
        original_values: dict[str, float],
    ) -> list[PatchEntry]:
        """Build patch entries from parameter changes.

        Args:
            template: Repair template.
            values: New parameter values.
            original_values: Original parameter values.

        Returns:
            List of PatchEntry objects.
        """
        entries: list[PatchEntry] = []

        site_map: dict[str, RepairSite] = {}
        for site in template.sites():
            site_map[str(site.node_id)] = site

        for param in template.parameters():
            new_val = values.get(param.name, param.initial_value)
            orig_val = original_values.get(param.name, param.initial_value)

            node_id = -1
            desc = ""
            for site in template.sites():
                if str(site.node_id) in param.name:
                    node_id = site.node_id
                    desc = site.description
                    break

            kind = "parameter_change"
            if "scale" in param.name:
                kind = "noise_scale"
            elif "threshold" in param.name:
                kind = "threshold"
            elif "clamp" in param.name:
                kind = "clamp_bound"
            elif "sensitivity" in param.name:
                kind = "sensitivity"
            elif "budget" in param.name:
                kind = "budget_split"

            entries.append(PatchEntry(
                parameter_name=param.name,
                site_node_id=node_id,
                site_description=desc,
                original_value=orig_val,
                repaired_value=new_val,
                change_kind=kind,
            ))

        return entries

    def _generate_diff(
        self,
        original: str,
        repaired: str,
        name: str,
    ) -> str:
        """Generate a unified diff between original and repaired source.

        Args:
            original: Original source code.
            repaired: Repaired source code.
            name: Mechanism name for the diff header.

        Returns:
            Unified diff string.
        """
        original_lines = original.splitlines(keepends=True)
        repaired_lines = repaired.splitlines(keepends=True)

        diff = difflib.unified_diff(
            original_lines,
            repaired_lines,
            fromfile=f"a/{name}.dp",
            tofile=f"b/{name}.dp",
            lineterm="",
        )
        return "\n".join(diff)

    def _extract_originals(
        self,
        mechanism: MechIR,
        template: RepairTemplate,
    ) -> dict[str, float]:
        """Extract original parameter values from the mechanism.

        Args:
            mechanism: Original mechanism.
            template: Repair template.

        Returns:
            Mapping from parameter name to original value.
        """
        values: dict[str, float] = {}
        for param in template.parameters():
            values[param.name] = param.initial_value
        return values

    def _validate_patch(
        self,
        original: MechIR,
        repaired: MechIR,
        template: RepairTemplate,
    ) -> tuple[bool, list[str]]:
        """Validate a repair patch.

        Checks that the repaired mechanism is structurally sound.

        Args:
            original: Original mechanism.
            repaired: Repaired mechanism.
            template: Template that was applied.

        Returns:
            Tuple of (is_valid, notes).
        """
        notes: list[str] = []
        is_valid = True

        orig_nodes = original.node_count()
        rep_nodes = repaired.node_count()
        if rep_nodes < orig_nodes:
            notes.append(f"Warning: node count decreased ({orig_nodes} → {rep_nodes})")

        for node in repaired.all_nodes():
            if isinstance(node, NoiseDrawNode):
                if isinstance(node.scale, Const):
                    if float(node.scale.value) <= 0:
                        notes.append(f"Error: non-positive scale at node {node.node_id}")
                        is_valid = False

        return is_valid, notes


# ═══════════════════════════════════════════════════════════════════════════
# SOURCE PATCHER
# ═══════════════════════════════════════════════════════════════════════════


class SourcePatcher:
    """Apply repairs at the source code level.

    Given a source code string and a patch report, applies the
    parameter changes to produce repaired source code.
    """

    def apply(
        self,
        source: str,
        patch_report: PatchReport,
    ) -> str:
        """Apply a patch report to source code.

        Performs textual substitution of parameter values in the source.

        Args:
            source: Original source code string.
            patch_report: The patch to apply.

        Returns:
            Repaired source code string.
        """
        result = source
        for entry in patch_report.entries:
            result = self._apply_entry(result, entry)
        return result

    def _apply_entry(self, source: str, entry: PatchEntry) -> str:
        """Apply a single patch entry to source code.

        Performs a textual replacement of the original value with
        the repaired value.

        Args:
            source: Current source code.
            entry: The patch entry to apply.

        Returns:
            Updated source code.
        """
        orig_str = self._format_value(entry.original_value)
        new_str = self._format_value(entry.repaired_value)

        if entry.change_kind == "noise_scale":
            patterns = [
                f"scale={orig_str}",
                f"scale = {orig_str}",
                f"b={orig_str}",
                f"sigma={orig_str}",
                f"σ={orig_str}",
            ]
        elif entry.change_kind == "threshold":
            patterns = [
                f"< {orig_str}",
                f"> {orig_str}",
                f"<= {orig_str}",
                f">= {orig_str}",
                f"== {orig_str}",
            ]
        elif entry.change_kind == "sensitivity":
            patterns = [
                f"sensitivity={orig_str}",
                f"Δ={orig_str}",
            ]
        else:
            patterns = [orig_str]

        for pattern in patterns:
            if pattern in source:
                new_pattern = pattern.replace(orig_str, new_str)
                source = source.replace(pattern, new_pattern, 1)
                break

        return source

    @staticmethod
    def _format_value(value: float) -> str:
        """Format a numeric value for source code.

        Args:
            value: The numeric value.

        Returns:
            String representation.
        """
        if value == int(value) and abs(value) < 1e10:
            return str(int(value))
        return f"{value:.6g}"

    def preview(
        self,
        source: str,
        patch_report: PatchReport,
        context_lines: int = 3,
    ) -> str:
        """Preview the patch with context.

        Shows the regions of the source that will change, with
        surrounding context lines.

        Args:
            source: Original source code.
            patch_report: The patch to preview.
            context_lines: Number of context lines around changes.

        Returns:
            Preview string.
        """
        repaired = self.apply(source, patch_report)
        original_lines = source.splitlines()
        repaired_lines = repaired.splitlines()

        changed_lines: set[int] = set()
        for i, (orig, rep) in enumerate(
            zip(original_lines, repaired_lines)
        ):
            if orig != rep:
                changed_lines.add(i)

        if not changed_lines:
            return "No changes"

        preview_lines: list[str] = []
        for line_no in sorted(changed_lines):
            start = max(0, line_no - context_lines)
            end = min(len(original_lines), line_no + context_lines + 1)

            preview_lines.append(f"--- Change at line {line_no + 1} ---")
            for i in range(start, end):
                prefix = "  "
                if i in changed_lines:
                    prefix = "- "
                    if i < len(original_lines):
                        preview_lines.append(f"{prefix}{original_lines[i]}")
                    prefix = "+ "
                    if i < len(repaired_lines):
                        preview_lines.append(f"{prefix}{repaired_lines[i]}")
                else:
                    if i < len(original_lines):
                        preview_lines.append(f"{prefix}{original_lines[i]}")

        return "\n".join(preview_lines)
