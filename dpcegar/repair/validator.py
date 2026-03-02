"""Repair validation for differential privacy mechanism repairs.

Validates that a repair is correct, consistent, and doesn't break
other properties of the mechanism.  Performs functional equivalence
checking, parameter range validation, regression testing, numerical
sanity checks, and certificate verification.

Classes
-------
RepairValidator         – main validation orchestrator
FunctionalChecker       – verify functional equivalence
ParameterRangeChecker   – validate parameter ranges
RegressionTester        – test repair against other DP notions
NumericalSanityChecker  – numerical sanity checks
CertificateChecker      – verify CEGAR certificates
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
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
    PrivacyBudget,
    PrivacyNotion,
    PureBudget,
    ApproxBudget,
    ZCDPBudget,
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
)
from dpcegar.paths.symbolic_path import PathSet
from dpcegar.cegar.abstraction import AbstractDensityBound
from dpcegar.cegar.engine import (
    CEGARResult,
    CEGARVerdict,
    VerificationCertificate,
)
from dpcegar.repair.templates import RepairTemplate, RepairSite
from dpcegar.utils.errors import InternalError, RepairError, ensure


# ═══════════════════════════════════════════════════════════════════════════
# VALIDATION RESULT
# ═══════════════════════════════════════════════════════════════════════════


class ValidationSeverity(Enum):
    """Severity of a validation finding."""

    ERROR = auto()
    WARNING = auto()
    INFO = auto()


@dataclass(frozen=True, slots=True)
class ValidationFinding:
    """A single finding from the validation process.

    Attributes:
        severity: Severity level.
        category: Category of the finding.
        message: Human-readable description.
        details: Additional structured data.
    """

    severity: ValidationSeverity
    category: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{self.severity.name}] {self.category}: {self.message}"


@dataclass(slots=True)
class ValidationReport:
    """Full validation report for a repair.

    Attributes:
        is_valid: Whether the repair passes all critical checks.
        findings: All validation findings.
        checks_passed: Number of checks that passed.
        checks_failed: Number of checks that failed.
        checks_warned: Number of checks that produced warnings.
    """

    is_valid: bool = True
    findings: list[ValidationFinding] = field(default_factory=list)
    checks_passed: int = 0
    checks_failed: int = 0
    checks_warned: int = 0

    def add_finding(self, finding: ValidationFinding) -> None:
        """Add a finding to the report.

        Updates is_valid if the finding is an error.

        Args:
            finding: The validation finding.
        """
        self.findings.append(finding)
        if finding.severity == ValidationSeverity.ERROR:
            self.is_valid = False
            self.checks_failed += 1
        elif finding.severity == ValidationSeverity.WARNING:
            self.checks_warned += 1
        else:
            self.checks_passed += 1

    def add_pass(self, category: str, message: str) -> None:
        """Record a passing check.

        Args:
            category: Check category.
            message: Description.
        """
        self.checks_passed += 1
        self.findings.append(ValidationFinding(
            severity=ValidationSeverity.INFO,
            category=category,
            message=message,
        ))

    def add_error(self, category: str, message: str, **details: Any) -> None:
        """Record a failing check.

        Args:
            category: Check category.
            message: Description.
            **details: Additional details.
        """
        self.is_valid = False
        self.checks_failed += 1
        self.findings.append(ValidationFinding(
            severity=ValidationSeverity.ERROR,
            category=category,
            message=message,
            details=dict(details),
        ))

    def add_warning(self, category: str, message: str, **details: Any) -> None:
        """Record a warning.

        Args:
            category: Check category.
            message: Description.
            **details: Additional details.
        """
        self.checks_warned += 1
        self.findings.append(ValidationFinding(
            severity=ValidationSeverity.WARNING,
            category=category,
            message=message,
            details=dict(details),
        ))

    def errors(self) -> list[ValidationFinding]:
        """Return only error findings.

        Returns:
            List of error findings.
        """
        return [f for f in self.findings if f.severity == ValidationSeverity.ERROR]

    def warnings(self) -> list[ValidationFinding]:
        """Return only warning findings.

        Returns:
            List of warning findings.
        """
        return [f for f in self.findings if f.severity == ValidationSeverity.WARNING]

    def summary(self) -> str:
        """Return a human-readable summary.

        Returns:
            Summary string.
        """
        status = "VALID" if self.is_valid else "INVALID"
        return (
            f"Validation: {status} "
            f"(passed={self.checks_passed}, "
            f"failed={self.checks_failed}, "
            f"warned={self.checks_warned})"
        )

    def __str__(self) -> str:
        return self.summary()


# ═══════════════════════════════════════════════════════════════════════════
# FUNCTIONAL EQUIVALENCE CHECKER
# ═══════════════════════════════════════════════════════════════════════════


class FunctionalChecker:
    """Verify that the repair preserves the query function structure.

    Checks that the repaired mechanism computes the same underlying
    query (only noise parameters may change).
    """

    def check(
        self,
        original: MechIR,
        repaired: MechIR,
    ) -> list[ValidationFinding]:
        """Check functional equivalence.

        Verifies that the query structure and branching structure are
        preserved by the repair.

        Args:
            original: Original mechanism.
            repaired: Repaired mechanism.

        Returns:
            List of validation findings.
        """
        findings: list[ValidationFinding] = []

        orig_queries = original.queries()
        rep_queries = repaired.queries()

        if len(orig_queries) != len(rep_queries):
            findings.append(ValidationFinding(
                severity=ValidationSeverity.WARNING,
                category="functional",
                message=(
                    f"Query count changed: {len(orig_queries)} → "
                    f"{len(rep_queries)}"
                ),
            ))
        else:
            for oq, rq in zip(orig_queries, rep_queries):
                if oq.query_name != rq.query_name:
                    findings.append(ValidationFinding(
                        severity=ValidationSeverity.ERROR,
                        category="functional",
                        message=(
                            f"Query name changed: {oq.query_name} → "
                            f"{rq.query_name}"
                        ),
                    ))

        orig_noise = original.noise_draws()
        rep_noise = repaired.noise_draws()

        if len(orig_noise) != len(rep_noise):
            findings.append(ValidationFinding(
                severity=ValidationSeverity.WARNING,
                category="functional",
                message=(
                    f"Noise draw count changed: {len(orig_noise)} → "
                    f"{len(rep_noise)}"
                ),
            ))

        if not findings:
            findings.append(ValidationFinding(
                severity=ValidationSeverity.INFO,
                category="functional",
                message="Functional structure preserved",
            ))

        return findings

    def check_query_preservation(
        self,
        original: MechIR,
        repaired: MechIR,
    ) -> bool:
        """Quick check if query functions are preserved.

        Args:
            original: Original mechanism.
            repaired: Repaired mechanism.

        Returns:
            True if all queries are preserved.
        """
        orig_names = sorted(q.query_name for q in original.queries())
        rep_names = sorted(q.query_name for q in repaired.queries())
        return orig_names == rep_names


# ═══════════════════════════════════════════════════════════════════════════
# PARAMETER RANGE CHECKER
# ═══════════════════════════════════════════════════════════════════════════


class ParameterRangeChecker:
    """Validate that repair parameters are in valid ranges.

    Checks that noise scales are positive, sensitivities are
    non-negative, clamp bounds are ordered, etc.
    """

    def check(
        self,
        repaired: MechIR,
        parameter_values: dict[str, float] | None = None,
    ) -> list[ValidationFinding]:
        """Validate parameter ranges in the repaired mechanism.

        Args:
            repaired: The repaired mechanism.
            parameter_values: Optional explicit parameter values.

        Returns:
            List of validation findings.
        """
        findings: list[ValidationFinding] = []

        for node in repaired.all_nodes():
            if isinstance(node, NoiseDrawNode):
                findings.extend(self._check_noise_params(node))
            elif isinstance(node, QueryNode):
                findings.extend(self._check_query_params(node))

        if parameter_values:
            findings.extend(self._check_explicit_params(parameter_values))

        if not findings:
            findings.append(ValidationFinding(
                severity=ValidationSeverity.INFO,
                category="parameter_range",
                message="All parameters in valid ranges",
            ))

        return findings

    def _check_noise_params(self, node: NoiseDrawNode) -> list[ValidationFinding]:
        """Check noise draw parameter validity.

        Args:
            node: The noise draw node.

        Returns:
            Findings.
        """
        findings: list[ValidationFinding] = []

        if isinstance(node.scale, Const):
            scale = float(node.scale.value)
            if scale <= 0:
                findings.append(ValidationFinding(
                    severity=ValidationSeverity.ERROR,
                    category="parameter_range",
                    message=f"Non-positive noise scale {scale} at node {node.node_id}",
                    details={"node_id": node.node_id, "scale": scale},
                ))
            elif scale < 1e-10:
                findings.append(ValidationFinding(
                    severity=ValidationSeverity.WARNING,
                    category="parameter_range",
                    message=f"Very small noise scale {scale} at node {node.node_id}",
                    details={"node_id": node.node_id, "scale": scale},
                ))
            elif scale > 1e8:
                findings.append(ValidationFinding(
                    severity=ValidationSeverity.WARNING,
                    category="parameter_range",
                    message=f"Very large noise scale {scale} at node {node.node_id}",
                    details={"node_id": node.node_id, "scale": scale},
                ))

        return findings

    def _check_query_params(self, node: QueryNode) -> list[ValidationFinding]:
        """Check query parameter validity.

        Args:
            node: The query node.

        Returns:
            Findings.
        """
        findings: list[ValidationFinding] = []

        if isinstance(node.sensitivity, Const):
            sens = float(node.sensitivity.value)
            if sens < 0:
                findings.append(ValidationFinding(
                    severity=ValidationSeverity.ERROR,
                    category="parameter_range",
                    message=f"Negative sensitivity {sens} at node {node.node_id}",
                    details={"node_id": node.node_id, "sensitivity": sens},
                ))
            elif sens == 0:
                findings.append(ValidationFinding(
                    severity=ValidationSeverity.WARNING,
                    category="parameter_range",
                    message=f"Zero sensitivity at node {node.node_id}",
                ))

        return findings

    def _check_explicit_params(
        self, values: dict[str, float]
    ) -> list[ValidationFinding]:
        """Check explicitly provided parameter values.

        Args:
            values: Parameter name to value mapping.

        Returns:
            Findings.
        """
        findings: list[ValidationFinding] = []

        for name, value in values.items():
            if math.isnan(value):
                findings.append(ValidationFinding(
                    severity=ValidationSeverity.ERROR,
                    category="parameter_range",
                    message=f"NaN value for parameter {name}",
                ))
            elif math.isinf(value):
                findings.append(ValidationFinding(
                    severity=ValidationSeverity.ERROR,
                    category="parameter_range",
                    message=f"Infinite value for parameter {name}",
                ))

            if "scale" in name and value <= 0:
                findings.append(ValidationFinding(
                    severity=ValidationSeverity.ERROR,
                    category="parameter_range",
                    message=f"Non-positive scale parameter {name}={value}",
                ))

            if "sensitivity" in name and value < 0:
                findings.append(ValidationFinding(
                    severity=ValidationSeverity.ERROR,
                    category="parameter_range",
                    message=f"Negative sensitivity parameter {name}={value}",
                ))

        return findings


# ═══════════════════════════════════════════════════════════════════════════
# REGRESSION TESTER
# ═══════════════════════════════════════════════════════════════════════════


class RegressionTester:
    """Test repair against other privacy notions.

    Verifies that a repair designed for one DP notion doesn't break
    privacy guarantees under related notions.
    """

    def __init__(self) -> None:
        """Initialise the regression tester."""
        self._related_notions: dict[PrivacyNotion, list[PrivacyBudget]] = {}

    def add_regression_budget(self, budget: PrivacyBudget) -> None:
        """Add a budget for regression testing.

        Args:
            budget: A privacy budget to test against.
        """
        notion = budget.notion
        self._related_notions.setdefault(notion, []).append(budget)

    def check(
        self,
        repaired: MechIR,
        primary_budget: PrivacyBudget,
        primary_result: CEGARResult | None = None,
    ) -> list[ValidationFinding]:
        """Run regression tests.

        Checks if the repair maintains privacy under related notions.
        Without a full CEGAR run, performs conservative estimates.

        Args:
            repaired: Repaired mechanism.
            primary_budget: The primary budget that was repaired for.
            primary_result: CEGAR result from primary verification.

        Returns:
            List of validation findings.
        """
        findings: list[ValidationFinding] = []

        if primary_result and primary_result.final_bounds:
            bound = primary_result.final_bounds
            findings.extend(
                self._check_related_notions(bound, primary_budget)
            )

        findings.extend(self._check_noise_adequacy(repaired, primary_budget))

        if not findings:
            findings.append(ValidationFinding(
                severity=ValidationSeverity.INFO,
                category="regression",
                message="No regression issues detected",
            ))

        return findings

    def _check_related_notions(
        self,
        bound: AbstractDensityBound,
        primary_budget: PrivacyBudget,
    ) -> list[ValidationFinding]:
        """Check related notions using density bounds.

        Args:
            bound: Final density bounds from primary verification.
            primary_budget: Primary budget.

        Returns:
            Findings.
        """
        findings: list[ValidationFinding] = []

        for notion, budgets in self._related_notions.items():
            for budget in budgets:
                eps, delta = budget.to_approx_dp()
                if bound.satisfies_epsilon(eps):
                    findings.append(ValidationFinding(
                        severity=ValidationSeverity.INFO,
                        category="regression",
                        message=f"Also satisfies {budget} ({notion.name})",
                    ))
                else:
                    findings.append(ValidationFinding(
                        severity=ValidationSeverity.WARNING,
                        category="regression",
                        message=f"May not satisfy {budget} ({notion.name})",
                        details={"bound": str(bound), "budget": str(budget)},
                    ))

        return findings

    def _check_noise_adequacy(
        self,
        repaired: MechIR,
        budget: PrivacyBudget,
    ) -> list[ValidationFinding]:
        """Check if noise scales are adequate for the budget.

        Conservative check using closed-form formulas for standard
        mechanisms.

        Args:
            repaired: Repaired mechanism.
            budget: Privacy budget.

        Returns:
            Findings.
        """
        findings: list[ValidationFinding] = []
        eps, _ = budget.to_approx_dp()

        for node in repaired.all_nodes():
            if not isinstance(node, NoiseDrawNode):
                continue
            if not isinstance(node.scale, Const):
                continue
            scale = float(node.scale.value)
            if scale <= 0:
                continue

            sens = 1.0
            if node.sensitivity is not None and isinstance(node.sensitivity, Const):
                sens = float(node.sensitivity.value)

            if node.noise_kind == NoiseKind.LAPLACE:
                required_scale = sens / eps if eps > 0 else float("inf")
                if scale < required_scale * 0.99:
                    findings.append(ValidationFinding(
                        severity=ValidationSeverity.WARNING,
                        category="regression",
                        message=(
                            f"Laplace scale {scale:.4g} may be insufficient "
                            f"(need ≥ {required_scale:.4g} for ε={eps})"
                        ),
                        details={
                            "node_id": node.node_id,
                            "scale": scale,
                            "required": required_scale,
                        },
                    ))

            elif node.noise_kind == NoiseKind.GAUSSIAN:
                if eps > 0:
                    required_sigma = sens * math.sqrt(2 * math.log(1.25)) / eps
                    if scale < required_sigma * 0.99:
                        findings.append(ValidationFinding(
                            severity=ValidationSeverity.WARNING,
                            category="regression",
                            message=(
                                f"Gaussian σ={scale:.4g} may be insufficient "
                                f"(need ≥ {required_sigma:.4g} for ε={eps})"
                            ),
                        ))

        return findings


# ═══════════════════════════════════════════════════════════════════════════
# NUMERICAL SANITY CHECKER
# ═══════════════════════════════════════════════════════════════════════════


class NumericalSanityChecker:
    """Perform numerical sanity checks on repaired mechanisms.

    Catches common numerical issues like overflow, underflow,
    division by zero, and extreme parameter values.
    """

    def __init__(
        self,
        scale_warning_threshold: float = 1e6,
        scale_error_threshold: float = 1e12,
    ) -> None:
        """Initialise with thresholds.

        Args:
            scale_warning_threshold: Scale above which to warn.
            scale_error_threshold: Scale above which to error.
        """
        self._warn_threshold = scale_warning_threshold
        self._error_threshold = scale_error_threshold

    def check(
        self,
        repaired: MechIR,
        parameter_values: dict[str, float] | None = None,
    ) -> list[ValidationFinding]:
        """Run numerical sanity checks.

        Args:
            repaired: Repaired mechanism.
            parameter_values: Optional parameter values.

        Returns:
            List of findings.
        """
        findings: list[ValidationFinding] = []

        findings.extend(self._check_overflow_risk(repaired))

        if parameter_values:
            findings.extend(self._check_parameter_magnitudes(parameter_values))

        findings.extend(self._check_division_safety(repaired))

        if not findings:
            findings.append(ValidationFinding(
                severity=ValidationSeverity.INFO,
                category="numerical",
                message="No numerical issues detected",
            ))

        return findings

    def _check_overflow_risk(self, mechanism: MechIR) -> list[ValidationFinding]:
        """Check for potential overflow in exponential operations.

        Args:
            mechanism: The mechanism to check.

        Returns:
            Findings.
        """
        findings: list[ValidationFinding] = []

        for node in mechanism.all_nodes():
            if isinstance(node, NoiseDrawNode):
                if isinstance(node.scale, Const):
                    scale = float(node.scale.value)
                    if scale > self._error_threshold:
                        findings.append(ValidationFinding(
                            severity=ValidationSeverity.ERROR,
                            category="numerical",
                            message=f"Extreme scale {scale:.2e} risks overflow",
                            details={"node_id": node.node_id},
                        ))
                    elif scale > self._warn_threshold:
                        findings.append(ValidationFinding(
                            severity=ValidationSeverity.WARNING,
                            category="numerical",
                            message=f"Large scale {scale:.2e} may cause precision loss",
                            details={"node_id": node.node_id},
                        ))

        return findings

    def _check_parameter_magnitudes(
        self, values: dict[str, float]
    ) -> list[ValidationFinding]:
        """Check parameter value magnitudes.

        Args:
            values: Parameter values.

        Returns:
            Findings.
        """
        findings: list[ValidationFinding] = []

        for name, value in values.items():
            abs_val = abs(value)
            if abs_val > self._error_threshold:
                findings.append(ValidationFinding(
                    severity=ValidationSeverity.ERROR,
                    category="numerical",
                    message=f"Extreme parameter {name}={value:.2e}",
                ))
            elif abs_val > self._warn_threshold:
                findings.append(ValidationFinding(
                    severity=ValidationSeverity.WARNING,
                    category="numerical",
                    message=f"Large parameter {name}={value:.2e}",
                ))
            elif abs_val > 0 and abs_val < 1e-15:
                findings.append(ValidationFinding(
                    severity=ValidationSeverity.WARNING,
                    category="numerical",
                    message=f"Near-zero parameter {name}={value:.2e}",
                ))

        return findings

    def _check_division_safety(self, mechanism: MechIR) -> list[ValidationFinding]:
        """Check for potential division by zero.

        Args:
            mechanism: The mechanism to check.

        Returns:
            Findings.
        """
        findings: list[ValidationFinding] = []

        for node in mechanism.all_nodes():
            if isinstance(node, NoiseDrawNode):
                if isinstance(node.scale, Const):
                    if float(node.scale.value) == 0:
                        findings.append(ValidationFinding(
                            severity=ValidationSeverity.ERROR,
                            category="numerical",
                            message=f"Zero scale causes division by zero at node {node.node_id}",
                        ))

        return findings


# ═══════════════════════════════════════════════════════════════════════════
# CERTIFICATE CHECKER
# ═══════════════════════════════════════════════════════════════════════════


class CertificateChecker:
    """Verify CEGAR certificates for repaired mechanisms.

    Checks that the verification certificate is internally consistent
    and that the density bounds actually prove privacy.
    """

    def check(
        self,
        certificate: VerificationCertificate,
        budget: PrivacyBudget,
    ) -> list[ValidationFinding]:
        """Verify a CEGAR certificate.

        Args:
            certificate: The verification certificate.
            budget: The privacy budget it should prove.

        Returns:
            List of findings.
        """
        findings: list[ValidationFinding] = []

        if certificate.budget is None:
            findings.append(ValidationFinding(
                severity=ValidationSeverity.ERROR,
                category="certificate",
                message="Certificate has no budget",
            ))
            return findings

        cert_eps, cert_delta = certificate.budget.to_approx_dp()
        budget_eps, budget_delta = budget.to_approx_dp()

        if cert_eps > budget_eps + 1e-9:
            findings.append(ValidationFinding(
                severity=ValidationSeverity.ERROR,
                category="certificate",
                message=(
                    f"Certificate ε={cert_eps} exceeds budget ε={budget_eps}"
                ),
            ))

        if cert_delta > budget_delta + 1e-12:
            findings.append(ValidationFinding(
                severity=ValidationSeverity.ERROR,
                category="certificate",
                message=(
                    f"Certificate δ={cert_delta} exceeds budget δ={budget_delta}"
                ),
            ))

        for sid, bound in certificate.density_bounds.items():
            if not bound.satisfies_epsilon(budget_eps):
                findings.append(ValidationFinding(
                    severity=ValidationSeverity.ERROR,
                    category="certificate",
                    message=(
                        f"State {sid} bound {bound} violates ε={budget_eps}"
                    ),
                    details={"state_id": sid, "bound": str(bound)},
                ))

        if certificate.is_valid():
            findings.append(ValidationFinding(
                severity=ValidationSeverity.INFO,
                category="certificate",
                message="Certificate is internally consistent",
            ))
        else:
            findings.append(ValidationFinding(
                severity=ValidationSeverity.ERROR,
                category="certificate",
                message="Certificate internal consistency check failed",
            ))

        return findings


# ═══════════════════════════════════════════════════════════════════════════
# REPAIR VALIDATOR — Main orchestrator
# ═══════════════════════════════════════════════════════════════════════════


class RepairValidator:
    """Main repair validation orchestrator.

    Runs all validation checks and produces a comprehensive report.
    """

    def __init__(self) -> None:
        """Initialise all sub-checkers."""
        self._functional = FunctionalChecker()
        self._parameter = ParameterRangeChecker()
        self._regression = RegressionTester()
        self._numerical = NumericalSanityChecker()
        self._certificate = CertificateChecker()

    def validate(
        self,
        original: MechIR,
        repaired: MechIR,
        budget: PrivacyBudget,
        parameter_values: dict[str, float] | None = None,
        certificate: VerificationCertificate | None = None,
        cegar_result: CEGARResult | None = None,
    ) -> ValidationReport:
        """Run all validation checks and produce a report.

        Args:
            original: Original mechanism.
            repaired: Repaired mechanism.
            budget: Privacy budget.
            parameter_values: Repair parameter values.
            certificate: Optional CEGAR certificate.
            cegar_result: Optional full CEGAR result.

        Returns:
            A comprehensive ValidationReport.
        """
        report = ValidationReport()

        for f in self._functional.check(original, repaired):
            report.add_finding(f)

        for f in self._parameter.check(repaired, parameter_values):
            report.add_finding(f)

        for f in self._regression.check(repaired, budget, cegar_result):
            report.add_finding(f)

        for f in self._numerical.check(repaired, parameter_values):
            report.add_finding(f)

        if certificate is not None:
            for f in self._certificate.check(certificate, budget):
                report.add_finding(f)

        return report

    def quick_validate(
        self,
        repaired: MechIR,
        budget: PrivacyBudget,
    ) -> bool:
        """Quick validation: check only critical constraints.

        Args:
            repaired: Repaired mechanism.
            budget: Privacy budget.

        Returns:
            True if no critical errors found.
        """
        for node in repaired.all_nodes():
            if isinstance(node, NoiseDrawNode):
                if isinstance(node.scale, Const):
                    if float(node.scale.value) <= 0:
                        return False

        return True

    def add_regression_budget(self, budget: PrivacyBudget) -> None:
        """Add a budget for regression testing.

        Args:
            budget: Privacy budget to test against.
        """
        self._regression.add_regression_budget(budget)
