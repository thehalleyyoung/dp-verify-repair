"""Human-readable report generation.

Generates verification and repair reports in multiple formats:
plain text, JSON, HTML, and Markdown.  Supports rich terminal
output with colors and tables via the ``rich`` library.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Sequence
from pathlib import Path

from dpcegar.ir.types import (
    PrivacyBudget,
    PrivacyNotion,
    PureBudget,
    ApproxBudget,
    ZCDPBudget,
    RDPBudget,
)


# ---------------------------------------------------------------------------
# Format enum
# ---------------------------------------------------------------------------


class ReportFormat(Enum):
    """Supported output formats for generated reports."""

    TEXT = auto()
    JSON = auto()
    HTML = auto()
    MARKDOWN = auto()
    RICH = auto()


_FORMAT_EXTENSIONS: dict[ReportFormat, str] = {
    ReportFormat.TEXT: ".txt",
    ReportFormat.JSON: ".json",
    ReportFormat.HTML: ".html",
    ReportFormat.MARKDOWN: ".md",
    ReportFormat.RICH: ".txt",
}


# ---------------------------------------------------------------------------
# Report data-classes
# ---------------------------------------------------------------------------


@dataclass
class ReportSection:
    """A titled section that may contain nested subsections.

    Attributes:
        title: Human-readable section heading.
        content: Free-form text body of the section.
        subsections: Ordered child sections.
        data: Arbitrary key/value pairs rendered as a table.
    """

    title: str = ""
    content: str = ""
    subsections: list[ReportSection] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationReport:
    """Complete report for a single verification run.

    Attributes:
        mechanism_name: Identifier of the mechanism under verification.
        privacy_notion: The privacy notion used during verification.
        budget: The target privacy budget.
        verdict: Outcome – VERIFIED, COUNTEREXAMPLE, UNKNOWN, or TIMEOUT.
        certificate_summary: Optional textual description of the proof
            certificate when verification succeeds.
        counterexample_details: Structured counterexample data when a
            violation is found.
        statistics: Timing, iteration counts, and solver statistics.
        timestamp: Unix epoch when the report was created.
        sections: Ordered list of sections for the full report body.
    """

    mechanism_name: str = ""
    privacy_notion: PrivacyNotion = PrivacyNotion.PURE_DP
    budget: PrivacyBudget = field(default_factory=PureBudget)
    verdict: str = "UNKNOWN"
    certificate_summary: str | None = None
    counterexample_details: dict[str, Any] | None = None
    statistics: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    sections: list[ReportSection] = field(default_factory=list)

    @classmethod
    def from_cegar_result(
        cls,
        mechanism_name: str,
        result: Any,
    ) -> VerificationReport:
        """Build a report from a :class:`CEGARResult` instance.

        Args:
            mechanism_name: Name of the mechanism that was verified.
            result: A ``CEGARResult`` produced by the CEGAR engine.

        Returns:
            A populated ``VerificationReport``.
        """
        verdict_str = result.verdict.name if hasattr(result.verdict, "name") else str(result.verdict)
        cert_summary: str | None = None
        if result.certificate is not None:
            cert_summary = str(result.certificate)

        cex_details: dict[str, Any] | None = None
        if result.counterexample is not None:
            cex_details = {
                "counterexample": str(result.counterexample),
            }

        stats: dict[str, Any] = {}
        if result.statistics is not None:
            stats = {
                "statistics": str(result.statistics),
            }
        stats.update(result.details)

        sections: list[ReportSection] = []
        sections.append(
            ReportSection(
                title="Overview",
                content=f"Mechanism {mechanism_name!r} verified under "
                f"{result.budget.notion.name if result.budget else 'N/A'}.",
                data={"verdict": verdict_str},
            )
        )

        if cert_summary is not None:
            sections.append(
                ReportSection(title="Certificate", content=cert_summary)
            )

        if cex_details is not None:
            sections.append(
                ReportSection(
                    title="Counterexample",
                    content="A privacy violation was detected.",
                    data=cex_details,
                )
            )

        return cls(
            mechanism_name=mechanism_name,
            privacy_notion=result.budget.notion if result.budget else PrivacyNotion.PURE_DP,
            budget=result.budget or PureBudget(),
            verdict=verdict_str,
            certificate_summary=cert_summary,
            counterexample_details=cex_details,
            statistics=stats,
            sections=sections,
        )


@dataclass
class RepairReport:
    """Report summarising a mechanism repair attempt.

    Attributes:
        mechanism_name: Identifier of the mechanism that was repaired.
        original_budget: Budget the original mechanism was tested against.
        repair_verdict: Outcome string (SUCCESS, NO_REPAIR, TIMEOUT, ERROR).
        repair_description: Human-readable explanation of the repair.
        parameter_changes: Mapping of parameter name to (old, new) values.
        repair_cost: Scalar cost of the repair.
        verification_of_repair: If the repaired mechanism was re-verified,
            the corresponding report.
        timestamp: Unix epoch when the report was created.
    """

    mechanism_name: str = ""
    original_budget: PrivacyBudget = field(default_factory=PureBudget)
    repair_verdict: str = "NO_REPAIR"
    repair_description: str = ""
    parameter_changes: dict[str, tuple[float, float]] = field(default_factory=dict)
    repair_cost: float = float("inf")
    verification_of_repair: VerificationReport | None = None
    timestamp: float = field(default_factory=time.time)

    @classmethod
    def from_repair_result(
        cls,
        mechanism_name: str,
        result: Any,
    ) -> RepairReport:
        """Build a report from a :class:`RepairResult` instance.

        Args:
            mechanism_name: Name of the mechanism that was repaired.
            result: A ``RepairResult`` produced by the repair synthesizer.

        Returns:
            A populated ``RepairReport``.
        """
        verdict_str = result.verdict.name if hasattr(result.verdict, "name") else str(result.verdict)

        description_parts: list[str] = []
        if result.template is not None:
            description_parts.append(f"Template: {result.template}")
        if result.parameter_values:
            description_parts.append(
                "Parameters: " + ", ".join(
                    f"{k}={v}" for k, v in result.parameter_values.items()
                )
            )
        description = "; ".join(description_parts) if description_parts else "No repair applied."

        param_changes: dict[str, tuple[float, float]] = {}
        for name, new_val in result.parameter_values.items():
            param_changes[name] = (0.0, float(new_val))

        ver_report: VerificationReport | None = None
        if result.verification_certificate is not None:
            ver_report = VerificationReport.from_cegar_result(
                mechanism_name, result.verification_certificate
            )

        return cls(
            mechanism_name=mechanism_name,
            original_budget=result.verification_certificate.budget
            if result.verification_certificate and result.verification_certificate.budget
            else PureBudget(),
            repair_verdict=verdict_str,
            repair_description=description,
            parameter_changes=param_changes,
            repair_cost=result.repair_cost,
            verification_of_repair=ver_report,
        )


@dataclass
class MultiVariantReport:
    """Report spanning multiple privacy-notion variants.

    Attributes:
        mechanism_name: Identifier of the mechanism analysed.
        variant_results: Mapping of variant label to result data.
        derived_guarantees: List of derived guarantee records.
        lattice_summary: Human-readable summary of the privacy lattice.
        timestamp: Unix epoch when the report was created.
    """

    mechanism_name: str = ""
    variant_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    derived_guarantees: list[dict[str, Any]] = field(default_factory=list)
    lattice_summary: str = ""
    timestamp: float = field(default_factory=time.time)

    @classmethod
    def from_multi_result(
        cls,
        mechanism_name: str,
        result: Any,
    ) -> MultiVariantReport:
        """Build a report from a multi-variant analysis result.

        Args:
            mechanism_name: Name of the mechanism analysed.
            result: An object carrying ``variant_results``,
                ``derived_guarantees``, and ``lattice_summary`` attributes
                or dictionary keys.

        Returns:
            A populated ``MultiVariantReport``.
        """
        if isinstance(result, dict):
            return cls(
                mechanism_name=mechanism_name,
                variant_results=result.get("variant_results", {}),
                derived_guarantees=result.get("derived_guarantees", []),
                lattice_summary=result.get("lattice_summary", ""),
            )
        return cls(
            mechanism_name=mechanism_name,
            variant_results=getattr(result, "variant_results", {}),
            derived_guarantees=getattr(result, "derived_guarantees", []),
            lattice_summary=getattr(result, "lattice_summary", ""),
        )


# ---------------------------------------------------------------------------
# Budget formatting helpers
# ---------------------------------------------------------------------------


def _budget_label(budget: PrivacyBudget) -> str:
    """Return a concise human-readable label for *budget*."""
    if isinstance(budget, PureBudget):
        return f"ε = {budget.epsilon}"
    if isinstance(budget, ApproxBudget):
        return f"(ε, δ) = ({budget.epsilon}, {budget.delta})"
    if isinstance(budget, ZCDPBudget):
        return f"ρ = {budget.rho}"
    if isinstance(budget, RDPBudget):
        return f"(α, ε) = ({budget.alpha}, {budget.epsilon})"
    return str(budget)


def _budget_to_dict(budget: PrivacyBudget) -> dict[str, Any]:
    """Serialise *budget* to a JSON-friendly dictionary."""
    base: dict[str, Any] = {"notion": budget.notion.name}
    if isinstance(budget, PureBudget):
        base["epsilon"] = budget.epsilon
    elif isinstance(budget, ApproxBudget):
        base["epsilon"] = budget.epsilon
        base["delta"] = budget.delta
    elif isinstance(budget, ZCDPBudget):
        base["rho"] = budget.rho
    elif isinstance(budget, RDPBudget):
        base["alpha"] = budget.alpha
        base["epsilon"] = budget.epsilon
    return base


# ---------------------------------------------------------------------------
# Text formatter
# ---------------------------------------------------------------------------


class TextReportFormatter:
    """Render reports as plain-text strings."""

    def format_verification(self, report: VerificationReport) -> str:
        """Format a verification report as plain text.

        Args:
            report: The verification report to format.

        Returns:
            A multi-line plain-text string.
        """
        lines: list[str] = []
        lines.append(self._format_header("Verification Report"))
        lines.append(f"Mechanism : {report.mechanism_name}")
        lines.append(f"Notion    : {report.privacy_notion.name}")
        lines.append(f"Budget    : {self._format_budget(report.budget)}")
        lines.append(f"Verdict   : {report.verdict}")
        lines.append(f"Timestamp : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(report.timestamp))}")
        lines.append("")

        if report.certificate_summary:
            lines.append("Certificate:")
            lines.append(f"  {report.certificate_summary}")
            lines.append("")

        if report.counterexample_details:
            lines.append(self._format_counterexample(report.counterexample_details))

        if report.statistics:
            lines.append("Statistics:")
            for key, val in report.statistics.items():
                lines.append(f"  {key}: {val}")
            lines.append("")

        for section in report.sections:
            lines.append(self._format_section(section))

        lines.append(self._format_header("End of Report"))
        return "\n".join(lines)

    def format_repair(self, report: RepairReport) -> str:
        """Format a repair report as plain text.

        Args:
            report: The repair report to format.

        Returns:
            A multi-line plain-text string.
        """
        lines: list[str] = []
        lines.append(self._format_header("Repair Report"))
        lines.append(f"Mechanism       : {report.mechanism_name}")
        lines.append(f"Original Budget : {self._format_budget(report.original_budget)}")
        lines.append(f"Repair Verdict  : {report.repair_verdict}")
        lines.append(f"Repair Cost     : {report.repair_cost}")
        lines.append(f"Timestamp       : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(report.timestamp))}")
        lines.append("")

        if report.repair_description:
            lines.append("Description:")
            lines.append(f"  {report.repair_description}")
            lines.append("")

        if report.parameter_changes:
            lines.append("Parameter Changes:")
            for name, (old, new) in report.parameter_changes.items():
                lines.append(f"  {name}: {old} -> {new}")
            lines.append("")

        if report.verification_of_repair is not None:
            lines.append("Re-verification of repaired mechanism:")
            lines.append(
                TextReportFormatter().format_verification(report.verification_of_repair)
            )

        lines.append(self._format_header("End of Repair Report"))
        return "\n".join(lines)

    def format_multi_variant(self, report: MultiVariantReport) -> str:
        """Format a multi-variant report as plain text.

        Args:
            report: The multi-variant report to format.

        Returns:
            A multi-line plain-text string.
        """
        lines: list[str] = []
        lines.append(self._format_header("Multi-Variant Report"))
        lines.append(f"Mechanism : {report.mechanism_name}")
        lines.append(f"Timestamp : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(report.timestamp))}")
        lines.append("")

        if report.lattice_summary:
            lines.append("Lattice Summary:")
            lines.append(f"  {report.lattice_summary}")
            lines.append("")

        if report.variant_results:
            lines.append("Variant Results:")
            for label, data in report.variant_results.items():
                lines.append(f"  [{label}]")
                for k, v in data.items():
                    lines.append(f"    {k}: {v}")
            lines.append("")

        if report.derived_guarantees:
            lines.append("Derived Guarantees:")
            for idx, guar in enumerate(report.derived_guarantees, 1):
                lines.append(f"  {idx}. {guar}")
            lines.append("")

        lines.append(self._format_header("End of Multi-Variant Report"))
        return "\n".join(lines)

    # -- helpers --

    def _format_header(self, title: str, width: int = 70) -> str:
        """Return a centred header bar.

        Args:
            title: Title text.
            width: Total character width of the bar.

        Returns:
            A decorated header string.
        """
        pad = max(width - len(title) - 4, 0)
        left = pad // 2
        right = pad - left
        return f"{'=' * left}  {title}  {'=' * right}"

    def _format_section(self, section: ReportSection, indent: int = 0) -> str:
        """Recursively format a report section.

        Args:
            section: The section to render.
            indent: Current indentation level.

        Returns:
            A multi-line string for the section.
        """
        prefix = "  " * indent
        lines: list[str] = []
        if section.title:
            lines.append(f"{prefix}[{section.title}]")
        if section.content:
            for line in section.content.splitlines():
                lines.append(f"{prefix}  {line}")
        if section.data:
            for key, val in section.data.items():
                lines.append(f"{prefix}  {key}: {val}")
        for sub in section.subsections:
            lines.append(self._format_section(sub, indent + 1))
        lines.append("")
        return "\n".join(lines)

    def _format_budget(self, budget: PrivacyBudget) -> str:
        """Return a compact budget string.

        Args:
            budget: The privacy budget to format.

        Returns:
            A short textual representation.
        """
        return _budget_label(budget)

    def _format_counterexample(self, cex: dict[str, Any]) -> str:
        """Format counterexample details.

        Args:
            cex: Dictionary of counterexample data.

        Returns:
            A multi-line plain-text block.
        """
        lines: list[str] = ["Counterexample Details:"]
        for key, val in cex.items():
            lines.append(f"  {key}: {val}")
        lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON formatter
# ---------------------------------------------------------------------------


class JSONReportFormatter:
    """Render reports as JSON strings."""

    def format_verification(self, report: VerificationReport) -> str:
        """Serialise a verification report to JSON.

        Args:
            report: The verification report.

        Returns:
            A pretty-printed JSON string.
        """
        obj: dict[str, Any] = {
            "type": "verification",
            "mechanism_name": report.mechanism_name,
            "privacy_notion": report.privacy_notion.name,
            "budget": self._budget_to_json(report.budget),
            "verdict": report.verdict,
            "certificate_summary": report.certificate_summary,
            "counterexample_details": report.counterexample_details,
            "statistics": report.statistics,
            "timestamp": report.timestamp,
        }
        return json.dumps(obj, indent=2, default=str)

    def format_repair(self, report: RepairReport) -> str:
        """Serialise a repair report to JSON.

        Args:
            report: The repair report.

        Returns:
            A pretty-printed JSON string.
        """
        ver_json: dict[str, Any] | None = None
        if report.verification_of_repair is not None:
            ver_json = json.loads(self.format_verification(report.verification_of_repair))

        obj: dict[str, Any] = {
            "type": "repair",
            "mechanism_name": report.mechanism_name,
            "original_budget": self._budget_to_json(report.original_budget),
            "repair_verdict": report.repair_verdict,
            "repair_description": report.repair_description,
            "parameter_changes": {
                k: {"old": v[0], "new": v[1]}
                for k, v in report.parameter_changes.items()
            },
            "repair_cost": report.repair_cost,
            "verification_of_repair": ver_json,
            "timestamp": report.timestamp,
        }
        return json.dumps(obj, indent=2, default=str)

    def format_multi_variant(self, report: MultiVariantReport) -> str:
        """Serialise a multi-variant report to JSON.

        Args:
            report: The multi-variant report.

        Returns:
            A pretty-printed JSON string.
        """
        obj: dict[str, Any] = {
            "type": "multi_variant",
            "mechanism_name": report.mechanism_name,
            "variant_results": report.variant_results,
            "derived_guarantees": report.derived_guarantees,
            "lattice_summary": report.lattice_summary,
            "timestamp": report.timestamp,
        }
        return json.dumps(obj, indent=2, default=str)

    def _budget_to_json(self, budget: PrivacyBudget) -> dict[str, Any]:
        """Convert a budget to a JSON-ready dict.

        Args:
            budget: Privacy budget object.

        Returns:
            Dictionary suitable for ``json.dumps``.
        """
        return _budget_to_dict(budget)


# ---------------------------------------------------------------------------
# HTML formatter
# ---------------------------------------------------------------------------


class HTMLReportFormatter:
    """Render reports as self-contained HTML documents."""

    def format_verification(self, report: VerificationReport) -> str:
        """Render a verification report as an HTML document.

        Args:
            report: The verification report.

        Returns:
            A complete HTML string.
        """
        parts: list[str] = [self._html_header(f"Verification: {report.mechanism_name}")]
        parts.append(f"<h2>Verification Report &mdash; {report.mechanism_name}</h2>")
        parts.append(self._verdict_badge(report.verdict))
        parts.append(
            self._html_table(
                ["Property", "Value"],
                [
                    ["Mechanism", report.mechanism_name],
                    ["Notion", report.privacy_notion.name],
                    ["Budget", _budget_label(report.budget)],
                    ["Verdict", report.verdict],
                    [
                        "Timestamp",
                        time.strftime(
                            "%Y-%m-%d %H:%M:%S",
                            time.localtime(report.timestamp),
                        ),
                    ],
                ],
            )
        )

        if report.certificate_summary:
            parts.append("<h3>Certificate</h3>")
            parts.append(f"<p>{report.certificate_summary}</p>")

        if report.counterexample_details:
            parts.append("<h3>Counterexample</h3>")
            parts.append(
                self._html_table(
                    ["Key", "Value"],
                    [[k, str(v)] for k, v in report.counterexample_details.items()],
                )
            )

        if report.statistics:
            parts.append("<h3>Statistics</h3>")
            parts.append(
                self._html_table(
                    ["Metric", "Value"],
                    [[k, str(v)] for k, v in report.statistics.items()],
                )
            )

        parts.append(self._html_footer())
        return "\n".join(parts)

    def format_repair(self, report: RepairReport) -> str:
        """Render a repair report as an HTML document.

        Args:
            report: The repair report.

        Returns:
            A complete HTML string.
        """
        parts: list[str] = [self._html_header(f"Repair: {report.mechanism_name}")]
        parts.append(f"<h2>Repair Report &mdash; {report.mechanism_name}</h2>")
        parts.append(self._verdict_badge(report.repair_verdict))
        parts.append(
            self._html_table(
                ["Property", "Value"],
                [
                    ["Mechanism", report.mechanism_name],
                    ["Original Budget", _budget_label(report.original_budget)],
                    ["Repair Verdict", report.repair_verdict],
                    ["Repair Cost", str(report.repair_cost)],
                    ["Description", report.repair_description],
                ],
            )
        )

        if report.parameter_changes:
            parts.append("<h3>Parameter Changes</h3>")
            parts.append(
                self._html_table(
                    ["Parameter", "Old", "New"],
                    [[k, str(v[0]), str(v[1])] for k, v in report.parameter_changes.items()],
                )
            )

        if report.verification_of_repair is not None:
            parts.append("<h3>Re-verification</h3>")
            inner = HTMLReportFormatter().format_verification(report.verification_of_repair)
            parts.append(f"<div class='nested'>{inner}</div>")

        parts.append(self._html_footer())
        return "\n".join(parts)

    def format_multi_variant(self, report: MultiVariantReport) -> str:
        """Render a multi-variant report as an HTML document.

        Args:
            report: The multi-variant report.

        Returns:
            A complete HTML string.
        """
        parts: list[str] = [self._html_header(f"Multi-Variant: {report.mechanism_name}")]
        parts.append(f"<h2>Multi-Variant Report &mdash; {report.mechanism_name}</h2>")

        if report.lattice_summary:
            parts.append(f"<p><strong>Lattice:</strong> {report.lattice_summary}</p>")

        if report.variant_results:
            parts.append("<h3>Variant Results</h3>")
            for label, data in report.variant_results.items():
                parts.append(f"<h4>{label}</h4>")
                parts.append(
                    self._html_table(
                        ["Key", "Value"],
                        [[k, str(v)] for k, v in data.items()],
                    )
                )

        if report.derived_guarantees:
            parts.append("<h3>Derived Guarantees</h3>")
            parts.append("<ol>")
            for guar in report.derived_guarantees:
                parts.append(f"  <li>{guar}</li>")
            parts.append("</ol>")

        parts.append(self._html_footer())
        return "\n".join(parts)

    # -- helpers --

    def _html_header(self, title: str) -> str:
        """Return the opening ``<html>`` / ``<head>`` boilerplate.

        Args:
            title: Page ``<title>`` content.

        Returns:
            HTML string up to and including ``<body>``.
        """
        return (
            "<!DOCTYPE html>\n"
            "<html lang='en'>\n<head>\n"
            f"  <title>{title}</title>\n"
            "  <meta charset='utf-8'>\n"
            "  <style>\n"
            "    body { font-family: sans-serif; margin: 2em; }\n"
            "    table { border-collapse: collapse; margin: 1em 0; }\n"
            "    th, td { border: 1px solid #ccc; padding: .4em .8em; text-align: left; }\n"
            "    th { background: #f5f5f5; }\n"
            "    .badge { display: inline-block; padding: .2em .6em;\n"
            "             border-radius: 4px; color: #fff; font-weight: bold; }\n"
            "    .badge-verified { background: #28a745; }\n"
            "    .badge-counterexample { background: #dc3545; }\n"
            "    .badge-unknown { background: #6c757d; }\n"
            "    .badge-timeout { background: #ffc107; color: #333; }\n"
            "    .badge-success { background: #28a745; }\n"
            "    .badge-no_repair { background: #dc3545; }\n"
            "    .badge-error { background: #dc3545; }\n"
            "    .nested { margin-left: 2em; border-left: 3px solid #ccc; padding-left: 1em; }\n"
            "  </style>\n"
            "</head>\n<body>"
        )

    def _html_footer(self) -> str:
        """Return the closing ``</body></html>`` tags.

        Returns:
            HTML closing string.
        """
        return "</body>\n</html>"

    def _html_table(self, headers: list[str], rows: list[list[str]]) -> str:
        """Render an HTML ``<table>``.

        Args:
            headers: Column header labels.
            rows: List of row data (each row is a list of cell strings).

        Returns:
            An HTML table string.
        """
        parts: list[str] = ["<table>", "  <tr>"]
        for h in headers:
            parts.append(f"    <th>{h}</th>")
        parts.append("  </tr>")
        for row in rows:
            parts.append("  <tr>")
            for cell in row:
                parts.append(f"    <td>{cell}</td>")
            parts.append("  </tr>")
        parts.append("</table>")
        return "\n".join(parts)

    def _verdict_badge(self, verdict: str) -> str:
        """Return a coloured HTML badge for *verdict*.

        Args:
            verdict: Verdict string (e.g. ``VERIFIED``).

        Returns:
            An HTML ``<span>`` with the appropriate CSS class.
        """
        css_class = f"badge-{verdict.lower()}"
        return f"<span class='badge {css_class}'>{verdict}</span>"


# ---------------------------------------------------------------------------
# Markdown formatter
# ---------------------------------------------------------------------------


class MarkdownReportFormatter:
    """Render reports as Markdown text."""

    def format_verification(self, report: VerificationReport) -> str:
        """Format a verification report as Markdown.

        Args:
            report: The verification report.

        Returns:
            A Markdown string.
        """
        lines: list[str] = [
            f"# Verification Report – {report.mechanism_name}",
            "",
            f"**Verdict:** `{report.verdict}`  ",
            f"**Notion:** {report.privacy_notion.name}  ",
            f"**Budget:** {_budget_label(report.budget)}  ",
            f"**Timestamp:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(report.timestamp))}",
            "",
        ]

        if report.certificate_summary:
            lines.append("## Certificate")
            lines.append("")
            lines.append(report.certificate_summary)
            lines.append("")

        if report.counterexample_details:
            lines.append("## Counterexample")
            lines.append("")
            lines.append(
                self._md_table(
                    ["Key", "Value"],
                    [[k, str(v)] for k, v in report.counterexample_details.items()],
                )
            )
            lines.append("")

        if report.statistics:
            lines.append("## Statistics")
            lines.append("")
            lines.append(
                self._md_table(
                    ["Metric", "Value"],
                    [[k, str(v)] for k, v in report.statistics.items()],
                )
            )
            lines.append("")

        return "\n".join(lines)

    def format_repair(self, report: RepairReport) -> str:
        """Format a repair report as Markdown.

        Args:
            report: The repair report.

        Returns:
            A Markdown string.
        """
        lines: list[str] = [
            f"# Repair Report – {report.mechanism_name}",
            "",
            f"**Verdict:** `{report.repair_verdict}`  ",
            f"**Original Budget:** {_budget_label(report.original_budget)}  ",
            f"**Repair Cost:** {report.repair_cost}  ",
            f"**Timestamp:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(report.timestamp))}",
            "",
        ]

        if report.repair_description:
            lines.append("## Description")
            lines.append("")
            lines.append(report.repair_description)
            lines.append("")

        if report.parameter_changes:
            lines.append("## Parameter Changes")
            lines.append("")
            lines.append(
                self._md_table(
                    ["Parameter", "Old", "New"],
                    [[k, str(v[0]), str(v[1])] for k, v in report.parameter_changes.items()],
                )
            )
            lines.append("")

        if report.verification_of_repair is not None:
            lines.append("## Re-verification of Repair")
            lines.append("")
            lines.append(
                MarkdownReportFormatter().format_verification(report.verification_of_repair)
            )

        return "\n".join(lines)

    def format_multi_variant(self, report: MultiVariantReport) -> str:
        """Format a multi-variant report as Markdown.

        Args:
            report: The multi-variant report.

        Returns:
            A Markdown string.
        """
        lines: list[str] = [
            f"# Multi-Variant Report – {report.mechanism_name}",
            "",
            f"**Timestamp:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(report.timestamp))}",
            "",
        ]

        if report.lattice_summary:
            lines.append("## Lattice Summary")
            lines.append("")
            lines.append(report.lattice_summary)
            lines.append("")

        if report.variant_results:
            lines.append("## Variant Results")
            lines.append("")
            for label, data in report.variant_results.items():
                lines.append(f"### {label}")
                lines.append("")
                lines.append(
                    self._md_table(
                        ["Key", "Value"],
                        [[k, str(v)] for k, v in data.items()],
                    )
                )
                lines.append("")

        if report.derived_guarantees:
            lines.append("## Derived Guarantees")
            lines.append("")
            for idx, guar in enumerate(report.derived_guarantees, 1):
                lines.append(f"{idx}. {guar}")
            lines.append("")

        return "\n".join(lines)

    # -- helpers --

    def _md_table(self, headers: list[str], rows: list[list[str]]) -> str:
        """Build a GitHub-flavoured Markdown table.

        Args:
            headers: Column headers.
            rows: List of rows (each a list of cell strings).

        Returns:
            A Markdown table string.
        """
        header_line = "| " + " | ".join(headers) + " |"
        sep_line = "| " + " | ".join("---" for _ in headers) + " |"
        body_lines = [
            "| " + " | ".join(cells) + " |" for cells in rows
        ]
        return "\n".join([header_line, sep_line, *body_lines])

    def _md_code_block(self, code: str, lang: str = "") -> str:
        """Wrap *code* in a fenced code block.

        Args:
            code: The code text.
            lang: Optional language identifier for syntax highlighting.

        Returns:
            A fenced Markdown code block.
        """
        return f"```{lang}\n{code}\n```"


# ---------------------------------------------------------------------------
# Rich formatter (plain-text with ANSI-like markup)
# ---------------------------------------------------------------------------


class RichReportFormatter:
    """Render reports using ``rich`` markup for terminal display.

    The output uses ``rich``-compatible markup tags such as
    ``[bold]``, ``[green]``, etc.  When rendered through the
    ``rich`` library these produce coloured terminal output; when
    printed directly they appear as bracketed annotations.
    """

    def format_verification(self, report: VerificationReport) -> str:
        """Format a verification report with rich markup.

        Args:
            report: The verification report.

        Returns:
            A string containing rich markup.
        """
        colour = self._verdict_color(report.verdict)
        lines: list[str] = [
            "[bold underline]Verification Report[/bold underline]",
            "",
            f"[bold]Mechanism:[/bold] {report.mechanism_name}",
            f"[bold]Notion:[/bold]    {report.privacy_notion.name}",
            f"[bold]Budget:[/bold]    {_budget_label(report.budget)}",
            f"[bold]Verdict:[/bold]   [{colour}]{report.verdict}[/{colour}]",
            "",
        ]

        if report.statistics:
            lines.append(
                self._rich_table(
                    "Statistics",
                    ["Metric", "Value"],
                    [[k, str(v)] for k, v in report.statistics.items()],
                )
            )
            lines.append("")

        if report.counterexample_details:
            lines.append(
                self._rich_table(
                    "Counterexample",
                    ["Key", "Value"],
                    [
                        [k, str(v)]
                        for k, v in report.counterexample_details.items()
                    ],
                )
            )
            lines.append("")

        if report.certificate_summary:
            lines.append(f"[bold]Certificate:[/bold] {report.certificate_summary}")
            lines.append("")

        return "\n".join(lines)

    def format_repair(self, report: RepairReport) -> str:
        """Format a repair report with rich markup.

        Args:
            report: The repair report.

        Returns:
            A string containing rich markup.
        """
        colour = self._verdict_color(report.repair_verdict)
        lines: list[str] = [
            "[bold underline]Repair Report[/bold underline]",
            "",
            f"[bold]Mechanism:[/bold]       {report.mechanism_name}",
            f"[bold]Original Budget:[/bold] {_budget_label(report.original_budget)}",
            f"[bold]Verdict:[/bold]         [{colour}]{report.repair_verdict}[/{colour}]",
            f"[bold]Repair Cost:[/bold]     {report.repair_cost}",
            "",
        ]

        if report.repair_description:
            lines.append(f"[bold]Description:[/bold] {report.repair_description}")
            lines.append("")

        if report.parameter_changes:
            lines.append(
                self._rich_table(
                    "Parameter Changes",
                    ["Parameter", "Old", "New"],
                    [
                        [k, str(v[0]), str(v[1])]
                        for k, v in report.parameter_changes.items()
                    ],
                )
            )
            lines.append("")

        if report.verification_of_repair is not None:
            lines.append("[bold]Re-verification:[/bold]")
            lines.append(self.format_verification(report.verification_of_repair))

        return "\n".join(lines)

    # -- helpers --

    def _rich_table(
        self,
        title: str,
        headers: list[str],
        rows: list[list[str]],
    ) -> str:
        """Build an ASCII table decorated with rich markup.

        Args:
            title: Table title.
            headers: Column header labels.
            rows: Row data.

        Returns:
            A string table suitable for ``rich.print``.
        """
        col_widths: list[int] = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                if i < len(col_widths):
                    col_widths[i] = max(col_widths[i], len(cell))

        def _row_str(cells: list[str]) -> str:
            parts = [cell.ljust(col_widths[i]) for i, cell in enumerate(cells)]
            return "| " + " | ".join(parts) + " |"

        separator = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
        lines: list[str] = [
            f"[bold]{title}[/bold]",
            separator,
            _row_str(headers),
            separator,
        ]
        for row in rows:
            lines.append(_row_str(row))
        lines.append(separator)
        return "\n".join(lines)

    def _verdict_color(self, verdict: str) -> str:
        """Map *verdict* to a rich colour name.

        Args:
            verdict: Verdict string.

        Returns:
            A rich colour tag name (e.g. ``green``, ``red``).
        """
        mapping: dict[str, str] = {
            "VERIFIED": "green",
            "SUCCESS": "green",
            "COUNTEREXAMPLE": "red",
            "NO_REPAIR": "red",
            "ERROR": "red",
            "UNKNOWN": "yellow",
            "TIMEOUT": "yellow",
        }
        return mapping.get(verdict.upper(), "white")


# ---------------------------------------------------------------------------
# Unified generator
# ---------------------------------------------------------------------------


_FORMATTERS: dict[ReportFormat, TextReportFormatter | JSONReportFormatter | HTMLReportFormatter | MarkdownReportFormatter | RichReportFormatter] = {
    ReportFormat.TEXT: TextReportFormatter(),
    ReportFormat.JSON: JSONReportFormatter(),
    ReportFormat.HTML: HTMLReportFormatter(),
    ReportFormat.MARKDOWN: MarkdownReportFormatter(),
    ReportFormat.RICH: RichReportFormatter(),
}


class ReportGenerator:
    """Facade that dispatches to format-specific renderers.

    Usage::

        gen = ReportGenerator()
        text = gen.generate_verification(report, ReportFormat.MARKDOWN)
        gen.save(text, Path("report.md"), ReportFormat.MARKDOWN)
    """

    def generate_verification(
        self,
        report: VerificationReport,
        fmt: ReportFormat = ReportFormat.TEXT,
    ) -> str:
        """Render a verification report in the requested format.

        Args:
            report: The verification report.
            fmt: Desired output format.

        Returns:
            The formatted report string.
        """
        formatter = _FORMATTERS[fmt]
        return formatter.format_verification(report)

    def generate_repair(
        self,
        report: RepairReport,
        fmt: ReportFormat = ReportFormat.TEXT,
    ) -> str:
        """Render a repair report in the requested format.

        Args:
            report: The repair report.
            fmt: Desired output format.

        Returns:
            The formatted report string.
        """
        formatter = _FORMATTERS[fmt]
        return formatter.format_repair(report)

    def generate_multi_variant(
        self,
        report: MultiVariantReport,
        fmt: ReportFormat = ReportFormat.TEXT,
    ) -> str:
        """Render a multi-variant report in the requested format.

        Args:
            report: The multi-variant report.
            fmt: Desired output format.

        Returns:
            The formatted report string.

        Raises:
            AttributeError: If the chosen formatter does not support
                multi-variant reports (e.g. the Rich formatter).
        """
        formatter = _FORMATTERS[fmt]
        return formatter.format_multi_variant(report)

    def save(
        self,
        content: str,
        path: Path,
        fmt: ReportFormat = ReportFormat.TEXT,
    ) -> None:
        """Write *content* to *path*, creating parent directories.

        Args:
            content: The rendered report string.
            path: Destination file path.
            fmt: Format (used only to infer encoding; all formats use UTF-8).
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
