"""Output formatters for the DP-CEGAR CLI.

Provides pluggable formatters that render verification and repair results
in different formats: plain text, JSON, rich terminal markup, ASCII
tables, and unified-diff style repair summaries.

Example::

    formatter = TextFormatter()
    print(formatter.format_verification(result))
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any, Sequence

from dpcegar.cegar.engine import CEGARResult, CEGARVerdict
from dpcegar.repair.synthesizer import RepairResult, RepairVerdict
from dpcegar.ir.types import (
    PrivacyBudget,
    PureBudget,
    ApproxBudget,
    ZCDPBudget,
    RDPBudget,
    FDPBudget,
    GDPBudget,
)


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _budget_to_dict(budget: PrivacyBudget) -> dict[str, Any]:
    """Convert a privacy budget into a JSON-friendly dictionary.

    Parameters
    ----------
    budget:
        Any concrete ``PrivacyBudget`` subclass.

    Returns
    -------
    dict[str, Any]
        A dictionary with ``"type"`` and the relevant numeric fields.
    """
    if isinstance(budget, PureBudget):
        return {"type": "pure", "epsilon": budget.epsilon}
    elif isinstance(budget, ApproxBudget):
        return {"type": "approx", "epsilon": budget.epsilon, "delta": budget.delta}
    elif isinstance(budget, ZCDPBudget):
        return {"type": "zcdp", "rho": budget.rho}
    elif isinstance(budget, RDPBudget):
        return {"type": "rdp", "alpha": budget.alpha, "epsilon": budget.epsilon}
    elif isinstance(budget, GDPBudget):
        return {"type": "gdp", "mu": budget.mu}
    elif isinstance(budget, FDPBudget):
        return {"type": "fdp"}
    return {"type": "unknown"}


def _budget_to_str(budget: PrivacyBudget) -> str:
    """Render a budget as a compact human-readable string.

    Parameters
    ----------
    budget:
        Any concrete ``PrivacyBudget`` subclass.

    Returns
    -------
    str
        E.g. ``"ε=1.0"`` or ``"(ε=1.0, δ=1e-05)"``.
    """
    if isinstance(budget, PureBudget):
        return f"ε={budget.epsilon}"
    elif isinstance(budget, ApproxBudget):
        return f"(ε={budget.epsilon}, δ={budget.delta})"
    elif isinstance(budget, ZCDPBudget):
        return f"ρ={budget.rho}"
    elif isinstance(budget, RDPBudget):
        return f"(α={budget.alpha}, ε={budget.epsilon})"
    elif isinstance(budget, GDPBudget):
        return f"μ={budget.mu}"
    elif isinstance(budget, FDPBudget):
        return "f-DP(trade-off)"
    return str(budget)


def _safe_serialize(obj: Any) -> Any:
    """Recursively convert dataclasses / enums for JSON serialization."""
    if isinstance(obj, Enum):
        return obj.value
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _safe_serialize(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_serialize(v) for v in obj]
    if callable(obj):
        return "<callable>"
    return obj


# ---------------------------------------------------------------------------
# Base formatter
# ---------------------------------------------------------------------------


class ResultFormatter(ABC):
    """Abstract base for all result formatters.

    Subclasses must implement the four ``format_*`` methods.  Each method
    receives the relevant result object and returns a formatted string.
    """

    @abstractmethod
    def format_verification(self, result: CEGARResult) -> str:
        """Format a single verification result.

        Parameters
        ----------
        result:
            The CEGAR verification result.

        Returns
        -------
        str
        """

    @abstractmethod
    def format_repair(self, result: RepairResult) -> str:
        """Format a single repair result.

        Parameters
        ----------
        result:
            The repair synthesis result.

        Returns
        -------
        str
        """

    @abstractmethod
    def format_multi_variant(self, results: dict[str, CEGARResult]) -> str:
        """Format results from multiple privacy-notion checks.

        Parameters
        ----------
        results:
            Mapping from notion name to the verification result.

        Returns
        -------
        str
        """

    @abstractmethod
    def format_profile(self, profile_data: list[dict[str, Any]]) -> str:
        """Format profiling data across multiple alpha values.

        Parameters
        ----------
        profile_data:
            List of dicts with ``alpha``, ``verdict``, and ``bounds``.

        Returns
        -------
        str
        """


# ---------------------------------------------------------------------------
# JSON formatter
# ---------------------------------------------------------------------------


class JSONFormatter(ResultFormatter):
    """Structured JSON output with proper serialisation of budgets."""

    def __init__(self, indent: int = 2) -> None:
        self._indent = indent

    def format_verification(self, result: CEGARResult) -> str:
        """Return *result* as a JSON string."""
        data: dict[str, Any] = {
            "verdict": result.verdict.value if isinstance(result.verdict, Enum) else str(result.verdict),
            "budget": _budget_to_dict(result.budget) if result.budget else None,
            "certificate": _safe_serialize(result.certificate) if result.certificate else None,
            "counterexample": _safe_serialize(result.counterexample) if result.counterexample else None,
            "final_bounds": _safe_serialize(result.final_bounds) if result.final_bounds else None,
            "statistics": _safe_serialize(result.statistics) if result.statistics else None,
        }
        return json.dumps(data, indent=self._indent, default=str)

    def format_repair(self, result: RepairResult) -> str:
        """Return *result* as a JSON string."""
        data: dict[str, Any] = {
            "verdict": result.verdict.value if isinstance(result.verdict, Enum) else str(result.verdict),
            "parameter_values": _safe_serialize(result.parameter_values) if result.parameter_values else None,
            "repair_cost": result.repair_cost,
            "statistics": _safe_serialize(result.statistics) if result.statistics else None,
        }
        return json.dumps(data, indent=self._indent, default=str)

    def format_multi_variant(self, results: dict[str, CEGARResult]) -> str:
        """Return all notion results as a JSON object keyed by notion name."""
        data: dict[str, Any] = {}
        for notion, res in results.items():
            data[notion] = {
                "verdict": res.verdict.value if isinstance(res.verdict, Enum) else str(res.verdict),
                "budget": _budget_to_dict(res.budget) if res.budget else None,
                "statistics": _safe_serialize(res.statistics) if res.statistics else None,
            }
        return json.dumps(data, indent=self._indent, default=str)

    def format_profile(self, profile_data: list[dict[str, Any]]) -> str:
        """Return profiling data as a JSON array."""
        return json.dumps(_safe_serialize(profile_data), indent=self._indent, default=str)


# ---------------------------------------------------------------------------
# Text formatter
# ---------------------------------------------------------------------------


class TextFormatter(ResultFormatter):
    """Human-readable plain-text output with sections and light framing."""

    _SECTION_WIDTH = 60

    def format_verification(self, result: CEGARResult) -> str:
        """Render a readable verification summary."""
        verdict_str = result.verdict.value if isinstance(result.verdict, Enum) else str(result.verdict)
        icon = "✓" if result.verdict == CEGARVerdict.VERIFIED else "✗"
        lines: list[str] = [
            self._header("Verification Result"),
            f"  Verdict : {icon} {verdict_str}",
        ]
        if result.budget:
            lines.append(f"  Budget  : {_budget_to_str(result.budget)}")
        if result.final_bounds:
            lines.append(f"  Bounds  : {result.final_bounds}")
        if result.statistics:
            lines.append(self._header("Statistics"))
            for k, v in (result.statistics if isinstance(result.statistics, dict) else {}).items():
                lines.append(f"  {k:<24} : {v}")
        if result.certificate:
            lines.append(self._header("Certificate"))
            lines.append(f"  {result.certificate}")
        lines.append(self._rule())
        return "\n".join(lines)

    def format_repair(self, result: RepairResult) -> str:
        """Render a readable repair summary."""
        verdict_str = result.verdict.value if isinstance(result.verdict, Enum) else str(result.verdict)
        icon = "✓" if result.verdict == RepairVerdict.SUCCESS else "✗"
        lines: list[str] = [
            self._header("Repair Result"),
            f"  Verdict : {icon} {verdict_str}",
        ]
        if result.repair_cost is not None:
            lines.append(f"  Cost    : {result.repair_cost:.4f}")
        if result.parameter_values:
            lines.append(self._header("Parameter Changes"))
            for k, v in result.parameter_values.items():
                lines.append(f"  {k:<24} = {v}")
        lines.append(self._rule())
        return "\n".join(lines)

    def format_multi_variant(self, results: dict[str, CEGARResult]) -> str:
        """Render a table summarising all notion results."""
        lines: list[str] = [
            self._header("Multi-Notion Check"),
            f"  {'Notion':<12} {'Verdict':<16} Budget",
            "  " + "-" * (self._SECTION_WIDTH - 2),
        ]
        for notion, res in results.items():
            verdict_str = res.verdict.value if isinstance(res.verdict, Enum) else str(res.verdict)
            budget_str = _budget_to_str(res.budget) if res.budget else "—"
            lines.append(f"  {notion:<12} {verdict_str:<16} {budget_str}")
        lines.append(self._rule())
        return "\n".join(lines)

    def format_profile(self, profile_data: list[dict[str, Any]]) -> str:
        """Render profiling data as a text table."""
        lines: list[str] = [
            self._header("Privacy Profile"),
            f"  {'Alpha':>8}  {'Epsilon':>12}  {'Verdict':<16}  Bounds",
            "  " + "-" * (self._SECTION_WIDTH - 2),
        ]
        for row in profile_data:
            alpha = row.get('alpha', '—')
            epsilon = row.get('epsilon', '—')
            verdict = row.get('verdict', '—')
            bounds = row.get('bounds', '—')
            alpha_s = f"{alpha:>8.1f}" if isinstance(alpha, (int, float)) else f"{alpha:>8}"
            epsilon_s = f"{epsilon:>12.6f}" if isinstance(epsilon, (int, float)) else f"{epsilon:>12}"
            lines.append(f"  {alpha_s}  {epsilon_s}  {verdict:<16}  {bounds}")
        lines.append(self._rule())
        return "\n".join(lines)

    # -- helpers -------------------------------------------------------------

    def _header(self, title: str) -> str:
        return f"\n{'═' * 4} {title} {'═' * (self._SECTION_WIDTH - 6 - len(title))}"

    def _rule(self) -> str:
        return "═" * self._SECTION_WIDTH


# ---------------------------------------------------------------------------
# Rich formatter (markup strings, no direct ``rich`` import)
# ---------------------------------------------------------------------------


class RichFormatter(ResultFormatter):
    """Terminal output using rich markup strings.

    Produces strings with ``[bold]``, ``[green]``, etc. markup that can
    be rendered by the ``rich`` library's ``Console.print``.  The
    formatter itself does **not** import ``rich`` directly; callers may
    pass the resulting string to ``rich.print()`` or strip markup for
    plain terminals.
    """

    def format_verification(self, result: CEGARResult) -> str:
        """Render verification result with rich markup."""
        verdict = result.verdict
        colour = "green" if verdict == CEGARVerdict.VERIFIED else "red"
        verdict_str = verdict.value if isinstance(verdict, Enum) else str(verdict)
        lines: list[str] = [
            "[bold]Verification Result[/bold]",
            f"  Verdict : [{colour}]{verdict_str}[/{colour}]",
        ]
        if result.budget:
            lines.append(f"  Budget  : [cyan]{_budget_to_str(result.budget)}[/cyan]")
        if result.final_bounds:
            lines.append(f"  Bounds  : [yellow]{result.final_bounds}[/yellow]")
        if result.statistics and isinstance(result.statistics, dict):
            lines.append("[bold]Statistics[/bold]")
            for k, v in result.statistics.items():
                lines.append(f"  [dim]{k:<24}[/dim] : {v}")
        if result.certificate:
            lines.append("[bold]Certificate[/bold]")
            lines.append(f"  [green]{result.certificate}[/green]")
        return "\n".join(lines)

    def format_repair(self, result: RepairResult) -> str:
        """Render repair result with rich markup."""
        verdict = result.verdict
        colour = "green" if verdict == RepairVerdict.SUCCESS else "red"
        verdict_str = verdict.value if isinstance(verdict, Enum) else str(verdict)
        lines: list[str] = [
            "[bold]Repair Result[/bold]",
            f"  Verdict : [{colour}]{verdict_str}[/{colour}]",
        ]
        if result.repair_cost is not None:
            lines.append(f"  Cost    : [yellow]{result.repair_cost:.4f}[/yellow]")
        if result.parameter_values:
            lines.append("[bold]Parameter Changes[/bold]")
            for k, v in result.parameter_values.items():
                lines.append(f"  [cyan]{k:<24}[/cyan] = {v}")
        return "\n".join(lines)

    def format_multi_variant(self, results: dict[str, CEGARResult]) -> str:
        """Render multi-notion results with rich markup."""
        lines: list[str] = ["[bold]Multi-Notion Check[/bold]"]
        for notion, res in results.items():
            verdict = res.verdict
            colour = "green" if verdict == CEGARVerdict.VERIFIED else "red"
            verdict_str = verdict.value if isinstance(verdict, Enum) else str(verdict)
            budget_str = _budget_to_str(res.budget) if res.budget else "—"
            lines.append(
                f"  [bold]{notion:<12}[/bold] [{colour}]{verdict_str:<16}[/{colour}] [cyan]{budget_str}[/cyan]"
            )
        return "\n".join(lines)

    def format_profile(self, profile_data: list[dict[str, Any]]) -> str:
        """Render profiling data with rich markup."""
        lines: list[str] = ["[bold]Privacy Profile[/bold]"]
        for row in profile_data:
            lines.append(
                f"  [cyan]α={row['alpha']:<6.1f}[/cyan]  {row['verdict']:<16}  {row.get('bounds', '—')}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CSV formatter
# ---------------------------------------------------------------------------


class CSVFormatter(ResultFormatter):
    """Comma-separated values output for CI/CD integration."""

    def format_verification(self, result: CEGARResult) -> str:
        """Return *result* as CSV rows (header + data)."""
        verdict = result.verdict.value if isinstance(result.verdict, Enum) else str(result.verdict)
        budget_str = _budget_to_str(result.budget) if result.budget else ""
        bounds = str(result.final_bounds) if result.final_bounds else ""
        lines = ["verdict,budget,bounds"]
        lines.append(f"{verdict},{budget_str},{bounds}")
        return "\n".join(lines)

    def format_repair(self, result: RepairResult) -> str:
        """Return *result* as CSV rows."""
        verdict = result.verdict.value if isinstance(result.verdict, Enum) else str(result.verdict)
        cost = str(result.repair_cost) if result.repair_cost is not None else ""
        lines = ["verdict,repair_cost"]
        lines.append(f"{verdict},{cost}")
        return "\n".join(lines)

    def format_multi_variant(self, results: dict[str, CEGARResult]) -> str:
        """Return multi-notion results as CSV."""
        lines = ["notion,verdict,budget"]
        for notion, res in results.items():
            verdict = res.verdict.value if isinstance(res.verdict, Enum) else str(res.verdict)
            budget_str = _budget_to_str(res.budget) if res.budget else ""
            lines.append(f"{notion},{verdict},{budget_str}")
        return "\n".join(lines)

    def format_profile(self, profile_data: list[dict[str, Any]]) -> str:
        """Return profiling data as CSV."""
        lines = ["alpha,verdict,bounds"]
        for row in profile_data:
            lines.append(f"{row.get('alpha', '')},{row.get('verdict', '')},{row.get('bounds', '')}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# SARIF formatter (GitHub Code Scanning)
# ---------------------------------------------------------------------------


class SARIFFormatter(ResultFormatter):
    """SARIF 2.1.0 output for GitHub Code Scanning integration."""

    SARIF_VERSION = "2.1.0"
    SARIF_SCHEMA = "https://json.schemastore.org/sarif-2.1.0.json"

    def format_verification(self, result: CEGARResult) -> str:
        """Return *result* as a SARIF JSON document."""
        verdict = result.verdict.value if isinstance(result.verdict, Enum) else str(result.verdict)
        is_violation = result.verdict != CEGARVerdict.VERIFIED

        results_list: list[dict[str, Any]] = []
        if is_violation:
            sarif_result: dict[str, Any] = {
                "ruleId": "dp-cegar/privacy-violation",
                "level": "error",
                "message": {
                    "text": f"Privacy violation detected: mechanism does not satisfy the requested budget. Verdict: {verdict}",
                },
                "locations": [
                    {
                        "physicalLocation": {
                            "artifactLocation": {"uri": "mechanism"},
                            "region": {"startLine": 1},
                        }
                    }
                ],
            }
            if result.counterexample:
                sarif_result["message"]["text"] += f" Counterexample: {_safe_serialize(result.counterexample)}"
            results_list.append(sarif_result)

        sarif: dict[str, Any] = {
            "$schema": self.SARIF_SCHEMA,
            "version": self.SARIF_VERSION,
            "runs": [
                {
                    "tool": {
                        "driver": {
                            "name": "DP-CEGAR",
                            "version": "0.1.0",
                            "informationUri": "https://github.com/dp-cegar/dp-cegar",
                            "rules": [
                                {
                                    "id": "dp-cegar/privacy-violation",
                                    "name": "PrivacyViolation",
                                    "shortDescription": {
                                        "text": "Differential privacy budget exceeded",
                                    },
                                    "fullDescription": {
                                        "text": "The mechanism does not satisfy the specified differential privacy budget. A counterexample demonstrates a pair of neighboring databases where the privacy loss exceeds the allowed budget.",
                                    },
                                    "defaultConfiguration": {"level": "error"},
                                    "helpUri": "https://github.com/dp-cegar/dp-cegar#readme",
                                },
                            ],
                        }
                    },
                    "results": results_list,
                }
            ],
        }
        return json.dumps(sarif, indent=2, default=str)

    def format_repair(self, result: RepairResult) -> str:
        """Return *result* as a SARIF JSON document."""
        return JSONFormatter().format_repair(result)

    def format_multi_variant(self, results: dict[str, CEGARResult]) -> str:
        """Return multi-notion results as a SARIF JSON document."""
        return JSONFormatter().format_multi_variant(results)

    def format_profile(self, profile_data: list[dict[str, Any]]) -> str:
        """Return profiling data as JSON (SARIF not applicable)."""
        return JSONFormatter().format_profile(profile_data)


# ---------------------------------------------------------------------------
# Table formatter (ASCII)
# ---------------------------------------------------------------------------


class TableFormatter:
    """Tabular comparison output using simple ASCII box-drawing.

    This is a utility formatter; it does not subclass ``ResultFormatter``
    because it focuses on side-by-side comparisons rather than single
    results.
    """

    def format_table(
        self,
        headers: Sequence[str],
        rows: Sequence[Sequence[str]],
        col_widths: Sequence[int] | None = None,
    ) -> str:
        """Render an ASCII table.

        Parameters
        ----------
        headers:
            Column header strings.
        rows:
            2-D sequence of cell values.
        col_widths:
            Optional explicit column widths.  If *None*, widths are
            auto-computed from the data.

        Returns
        -------
        str
            The complete table as a multi-line string.
        """
        if col_widths is None:
            col_widths = self._compute_widths(headers, rows)
        sep = self._separator(col_widths)
        header_line = self._row(headers, col_widths)
        body_lines = [self._row(row, col_widths) for row in rows]
        return "\n".join([sep, header_line, sep, *body_lines, sep])

    def format_comparison(
        self,
        results_a: dict[str, CEGARResult],
        results_b: dict[str, CEGARResult],
        label_a: str = "Run A",
        label_b: str = "Run B",
    ) -> str:
        """Compare two sets of verification results side-by-side.

        Parameters
        ----------
        results_a, results_b:
            Dicts mapping notion/mechanism name to ``CEGARResult``.
        label_a, label_b:
            Column header labels for each run.

        Returns
        -------
        str
        """
        all_keys = list(dict.fromkeys(list(results_a) + list(results_b)))
        headers = ["Name", label_a, label_b, "Match"]
        rows: list[list[str]] = []
        for key in all_keys:
            va = results_a.get(key)
            vb = results_b.get(key)
            va_str = (va.verdict.value if isinstance(va.verdict, Enum) else str(va.verdict)) if va else "—"
            vb_str = (vb.verdict.value if isinstance(vb.verdict, Enum) else str(vb.verdict)) if vb else "—"
            match = "✓" if va_str == vb_str else "✗"
            rows.append([key, va_str, vb_str, match])
        return self.format_table(headers, rows)

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _compute_widths(
        headers: Sequence[str], rows: Sequence[Sequence[str]]
    ) -> list[int]:
        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                if i < len(widths):
                    widths[i] = max(widths[i], len(str(cell)))
        return [w + 2 for w in widths]

    @staticmethod
    def _separator(widths: Sequence[int]) -> str:
        return "+" + "+".join("-" * w for w in widths) + "+"

    @staticmethod
    def _row(cells: Sequence[str], widths: Sequence[int]) -> str:
        parts: list[str] = []
        for cell, w in zip(cells, widths):
            parts.append(f" {str(cell):<{w - 1}}")
        return "|" + "|".join(parts) + "|"


# ---------------------------------------------------------------------------
# Diff formatter
# ---------------------------------------------------------------------------


class DiffFormatter:
    """Unified-diff style output for repair parameter changes.

    Shows the original and repaired parameter values in a format
    reminiscent of ``diff -u`` so that changes are immediately visible.
    """

    def format_repair_diff(
        self,
        original_params: dict[str, Any],
        repaired_params: dict[str, Any],
        mechanism_name: str = "mechanism",
    ) -> str:
        """Generate a unified-diff view of parameter changes.

        Parameters
        ----------
        original_params:
            Original parameter name→value mapping.
        repaired_params:
            Repaired parameter name→value mapping.
        mechanism_name:
            Name used in the diff header.

        Returns
        -------
        str
            Multi-line unified-diff style string.
        """
        lines: list[str] = [
            f"--- a/{mechanism_name}.params",
            f"+++ b/{mechanism_name}.params",
            "@@ parameters @@",
        ]
        all_keys = list(dict.fromkeys(list(original_params) + list(repaired_params)))
        for key in all_keys:
            old_val = original_params.get(key)
            new_val = repaired_params.get(key)
            if old_val == new_val:
                lines.append(f" {key} = {old_val}")
            else:
                if old_val is not None:
                    lines.append(f"-{key} = {old_val}")
                if new_val is not None:
                    lines.append(f"+{key} = {new_val}")
        return "\n".join(lines)

    def format_repair_summary(self, result: RepairResult) -> str:
        """Generate a diff summary from a ``RepairResult``.

        Uses ``result.parameter_values`` as the repaired values and
        falls back to an empty mapping for original values when the
        repaired mechanism is not available.

        Parameters
        ----------
        result:
            The repair synthesis result.

        Returns
        -------
        str
        """
        repaired = result.parameter_values or {}
        original: dict[str, Any] = {}
        if result.repaired_mechanism and hasattr(result.repaired_mechanism, "parameters"):
            original = dict(result.repaired_mechanism.parameters)

        # Build a minimal original map with "?" for unknown prior values
        for key in repaired:
            if key not in original:
                original[key] = "?"

        name = "mechanism"
        if result.repaired_mechanism and hasattr(result.repaired_mechanism, "name"):
            name = result.repaired_mechanism.name

        return self.format_repair_diff(original, repaired, mechanism_name=name)
