"""Privacy-specific SMT encodings for differential privacy verification.

Provides encoder classes for each DP variant:
  - PureEpsDPEncoder:   |L(o)| ≤ ε
  - ApproxDPEncoder:    hockey-stick divergence
  - ZCDPEncoder:        zero-concentrated DP (Rényi divergence)
  - RDPEncoder:         Rényi DP
  - GDPEncoder:         Gaussian DP
  - FDPEncoder:         f-DP trade-off function
  - CrossPathEncoder:   data-dependent branching

Each encoder produces an :class:`SMTEncoding` containing all necessary
Z3 constraints, metadata about the theory and approximations used, and
soundness flags.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import z3
except ImportError:  # pragma: no cover
    z3 = None  # type: ignore[assignment]

from dpcegar.ir.types import (
    Abs,
    BinOp,
    BinOpKind,
    Const,
    Exp,
    IRType,
    Log,
    NoiseKind,
    Phi,
    PhiInv,
    PrivacyBudget,
    PrivacyNotion,
    PureBudget,
    ApproxBudget,
    ZCDPBudget,
    RDPBudget,
    GDPBudget,
    FDPBudget,
    TypedExpr,
    Var,
)
from dpcegar.paths.symbolic_path import (
    NoiseDrawInfo,
    PathCondition,
    SymbolicPath,
)
from dpcegar.density.ratio_builder import DensityRatioExpr, DensityRatioResult
from dpcegar.smt.encoding import (
    AbsLinearizer,
    CaseSplitter,
    ConstraintBuilder,
    EncodingBuilder,
    ExprToZ3,
    PathConditionEncoder,
    SMTEncoding,
    _fresh_aux,
)
from dpcegar.smt.transcendental import (
    Precision,
    SoundnessTracker,
    TranscendentalApprox,
)
from dpcegar.smt.theory_selection import SMTTheory, TheoryAnalysisResult


# ═══════════════════════════════════════════════════════════════════════════
# PRIVACY ENCODING RESULT
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class PrivacyEncodingResult:
    """Result of privacy-specific encoding.

    Attributes:
        encoding:        The SMT encoding.
        theory:          The SMT theory used.
        notion:          The DP notion encoded.
        num_paths:       Number of paths encoded.
        num_cross_paths: Number of cross-path pairs.
        approximations:  List of approximation descriptions.
        is_sound:        Whether all approximations are certified sound.
    """

    encoding: SMTEncoding
    theory: SMTTheory = SMTTheory.QF_LRA
    notion: PrivacyNotion = PrivacyNotion.PURE_DP
    num_paths: int = 0
    num_cross_paths: int = 0
    approximations: list[str] = field(default_factory=list)
    is_sound: bool = True

    def summary(self) -> dict[str, Any]:
        """Return a summary dictionary.

        Returns:
            Dictionary with encoding statistics.
        """
        return {
            "notion": self.notion.name,
            "theory": self.theory.value,
            "paths": self.num_paths,
            "cross_paths": self.num_cross_paths,
            "sound": self.is_sound,
            "approximations": self.approximations,
            "variables": self.encoding.variable_count(),
            "assertions": self.encoding.assertion_count(),
        }


# ═══════════════════════════════════════════════════════════════════════════
# BASE ENCODER
# ═══════════════════════════════════════════════════════════════════════════


class _BasePrivacyEncoder:
    """Base class for privacy encoders.

    Provides common utilities for converting expressions, building
    constraints, and managing auxiliary variables.

    Args:
        precision: Transcendental approximation precision level.
    """

    def __init__(self, precision: Precision = Precision.STANDARD) -> None:
        self._precision = precision
        self._converter = ExprToZ3(precision=precision)
        self._builder = ConstraintBuilder(self._converter)
        self._linearizer = AbsLinearizer()
        self._splitter = CaseSplitter()
        self._tracker = SoundnessTracker()
        self._transcendental = TranscendentalApprox(precision, self._tracker)

    def _reset_converter(self) -> None:
        """Reset the expression converter for a fresh encoding."""
        self._converter = ExprToZ3(precision=self._precision)
        self._builder = ConstraintBuilder(self._converter)

    def _encode_path_conditions(
        self,
        ratio: DensityRatioExpr,
    ) -> list[Any]:
        """Encode path conditions for a density ratio.

        Args:
            ratio: The density ratio expression with path conditions.

        Returns:
            List of Z3 constraints for the path conditions.
        """
        pc_encoder = PathConditionEncoder(self._converter)
        constraints: list[Any] = []

        if not ratio.path_condition_d.is_trivially_true():
            constraints.append(pc_encoder.encode(ratio.path_condition_d))
        if not ratio.path_condition_d_prime.is_trivially_true():
            constraints.append(pc_encoder.encode(ratio.path_condition_d_prime))

        return constraints

    def _encode_log_ratio(self, ratio: DensityRatioExpr) -> Any:
        """Convert a log ratio expression to Z3.

        Args:
            ratio: Density ratio expression.

        Returns:
            Z3 expression for the log ratio.
        """
        return self._converter.convert(ratio.log_ratio)


# ═══════════════════════════════════════════════════════════════════════════
# PURE ε-DP ENCODER
# ═══════════════════════════════════════════════════════════════════════════


class PureEpsDPEncoder(_BasePrivacyEncoder):
    """Encode pure ε-differential privacy constraints.

    Pure DP requires |L(o)| ≤ ε for all outputs o and all adjacent
    databases (d, d').  This encoder produces the *negation*:
    ∃ o, d, d' such that |L(o)| > ε, so SAT = violation.

    For Laplace noise with sensitivity Δ and scale b:
        |L(o)| = |Δ|/b ≤ ε  →  |Δ| ≤ ε·b

    The encoder case-splits absolute values to get linear constraints
    when possible.

    Args:
        precision: Approximation precision level.
    """

    def __init__(self, precision: Precision = Precision.STANDARD) -> None:
        super().__init__(precision)

    def encode(
        self,
        ratio_result: DensityRatioResult,
        epsilon: float,
        sensitivity_bound: float = 1.0,
    ) -> PrivacyEncodingResult:
        """Encode the negation of |L(o)| ≤ ε for all paths.

        For each density ratio expression, asserts that path conditions
        hold AND |L(o)| > ε.  The overall formula is a disjunction
        over all ratio expressions (any one violation suffices).

        Args:
            ratio_result: All density ratio expressions.
            epsilon:      Privacy budget ε.
            sensitivity_bound: Upper bound on query sensitivity.

        Returns:
            PrivacyEncodingResult with the encoding.
        """
        self._reset_converter()
        eps_z3 = z3.RealVal(str(epsilon))
        all_disjuncts: list[Any] = []
        all_constraints: list[Any] = []
        num_cross = 0

        for ratio in ratio_result.ratios:
            path_constraints = self._encode_path_conditions(ratio)
            lr_z3 = self._encode_log_ratio(ratio)

            # |L(o)| > eps  ↔  L > eps ∨ L < -eps
            violation = z3.Or(lr_z3 > eps_z3, lr_z3 < -eps_z3)

            if path_constraints:
                disjunct = z3.And(*path_constraints, violation)
            else:
                disjunct = violation

            all_disjuncts.append(disjunct)
            if ratio.is_cross_path:
                num_cross += 1

        if not all_disjuncts:
            formula = z3.BoolVal(False)
        elif len(all_disjuncts) == 1:
            formula = all_disjuncts[0]
        else:
            formula = z3.Or(*all_disjuncts)

        all_constraints.append(formula)

        # Adjacency: bound sensitivity variables |delta_q| <= sensitivity_bound
        sens_z3 = z3.RealVal(str(sensitivity_bound))
        for name, var in self._converter.variables.items():
            if name.startswith("delta"):
                all_constraints.append(var <= sens_z3)
                all_constraints.append(var >= -sens_z3)

        all_constraints.extend(self._converter.aux_constraints)

        encoding = SMTEncoding(
            variables=self._converter.variables,
            assertions=all_constraints,
            aux_vars=self._converter.aux_vars,
            soundness=self._tracker,
            metadata={
                "notion": "pure_dp",
                "epsilon": epsilon,
                "num_ratios": len(ratio_result.ratios),
            },
        )
        encoding._rebuild_formula()

        return PrivacyEncodingResult(
            encoding=encoding,
            theory=SMTTheory.QF_LRA,
            notion=PrivacyNotion.PURE_DP,
            num_paths=len(ratio_result.ratios),
            num_cross_paths=num_cross,
            is_sound=self._tracker.all_sound,
        )

    def encode_laplace(
        self,
        delta_var: str,
        scale_var: str,
        epsilon: float,
        sensitivity_bound: float = 1.0,
    ) -> SMTEncoding:
        """Encode Laplace mechanism pure-DP constraint.

        For Laplace(b), pure DP requires Δ/b ≤ ε, i.e. Δ ≤ ε·b.
        This is a linear constraint.

        Args:
            delta_var:         Name of the sensitivity variable.
            scale_var:         Name of the scale variable.
            epsilon:           Privacy budget.
            sensitivity_bound: Upper bound on sensitivity.

        Returns:
            SMTEncoding with the constraint.
        """
        delta = z3.Real(delta_var)
        b = z3.Real(scale_var)
        eps = z3.RealVal(str(epsilon))
        sens_bound = z3.RealVal(str(sensitivity_bound))

        assertions = [
            # Adjacency: |Δ| <= sensitivity_bound
            delta <= sens_bound,
            delta >= -sens_bound,
            # Scale positive
            b > z3.RealVal(0),
            # Negate privacy: |Δ|/b > ε
            # Case split: Δ >= 0 → Δ > ε·b, or Δ < 0 → -Δ > ε·b
            z3.Or(delta > eps * b, -delta > eps * b),
        ]

        encoding = SMTEncoding(
            variables={delta_var: delta, scale_var: b},
            assertions=assertions,
            metadata={"type": "laplace_pure_dp", "epsilon": epsilon},
        )
        encoding._rebuild_formula()
        return encoding

    def encode_single_ratio(
        self,
        log_ratio_expr: TypedExpr,
        epsilon: float,
        path_cond: PathCondition | None = None,
    ) -> SMTEncoding:
        """Encode the negation of |L(o)| ≤ ε for a single ratio.

        Args:
            log_ratio_expr: The log ratio as an IR expression.
            epsilon:        Privacy budget.
            path_cond:      Optional path condition.

        Returns:
            SMTEncoding asserting the violation.
        """
        self._reset_converter()
        assertions: list[Any] = []

        if path_cond is not None:
            pc_enc = PathConditionEncoder(self._converter)
            assertions.append(pc_enc.encode(path_cond))

        lr_z3 = self._converter.convert(log_ratio_expr)
        eps_z3 = z3.RealVal(str(epsilon))
        assertions.append(z3.Or(lr_z3 > eps_z3, lr_z3 < -eps_z3))
        assertions.extend(self._converter.aux_constraints)

        encoding = SMTEncoding(
            variables=self._converter.variables,
            assertions=assertions,
            aux_vars=self._converter.aux_vars,
            soundness=self._tracker,
            metadata={"type": "single_ratio_pure_dp", "epsilon": epsilon},
        )
        encoding._rebuild_formula()
        return encoding


# ═══════════════════════════════════════════════════════════════════════════
# APPROXIMATE (ε,δ)-DP ENCODER
# ═══════════════════════════════════════════════════════════════════════════


class ApproxDPEncoder(_BasePrivacyEncoder):
    """Encode approximate (ε,δ)-differential privacy constraints.

    For Gaussian noise with sensitivity Δ and scale σ, the hockey-stick
    divergence gives:

        δ = Φ(-ε·σ/Δ + Δ/(2σ)) - e^ε · Φ(-ε·σ/Δ - Δ/(2σ))

    The encoder uses polynomial approximations of Φ for SMT encoding.

    Args:
        precision: Approximation precision level.
    """

    def __init__(self, precision: Precision = Precision.STANDARD) -> None:
        super().__init__(precision)

    def encode(
        self,
        ratio_result: DensityRatioResult,
        epsilon: float,
        delta: float,
    ) -> PrivacyEncodingResult:
        """Encode the negation of (ε,δ)-DP for all paths.

        Asserts the existence of parameters where the hockey-stick
        divergence exceeds δ at the given ε.

        Args:
            ratio_result: Density ratio expressions.
            epsilon:      Privacy parameter ε.
            delta:        Privacy parameter δ.

        Returns:
            PrivacyEncodingResult.
        """
        self._reset_converter()
        all_disjuncts: list[Any] = []
        approx_notes: list[str] = []
        num_cross = 0

        for ratio in ratio_result.ratios:
            path_constraints = self._encode_path_conditions(ratio)
            lr_z3 = self._encode_log_ratio(ratio)

            # Hockey-stick: max(0, e^L - e^eps) integrated
            # For verification negation: ∃ o s.t. E[max(0, e^{L(o)} - e^ε)] > δ
            eps_z3 = z3.RealVal(str(epsilon))
            delta_z3 = z3.RealVal(str(delta))

            # Approximate e^L via polynomial
            exp_L = self._transcendental.approx_exp(lr_z3)
            exp_eps = self._transcendental.approx_exp(eps_z3)
            approx_notes.append(f"exp approx: err={exp_L.error_bound}")

            # hockey-stick integrand: max(0, e^L - e^eps)
            integrand = z3.If(
                exp_L.value > exp_eps.value,
                exp_L.value - exp_eps.value,
                z3.RealVal(0),
            )

            # Violation: integrand > delta
            violation = integrand > delta_z3

            if path_constraints:
                disjunct = z3.And(*path_constraints, violation)
            else:
                disjunct = violation

            all_disjuncts.append(disjunct)
            if ratio.is_cross_path:
                num_cross += 1

        assertions: list[Any] = []
        if not all_disjuncts:
            assertions.append(z3.BoolVal(False))
        elif len(all_disjuncts) == 1:
            assertions.append(all_disjuncts[0])
        else:
            assertions.append(z3.Or(*all_disjuncts))

        assertions.extend(self._converter.aux_constraints)

        encoding = SMTEncoding(
            variables=self._converter.variables,
            assertions=assertions,
            aux_vars=self._converter.aux_vars,
            soundness=self._tracker,
            metadata={
                "notion": "approx_dp",
                "epsilon": epsilon,
                "delta": delta,
            },
        )
        encoding._rebuild_formula()

        return PrivacyEncodingResult(
            encoding=encoding,
            theory=SMTTheory.QF_NRA,
            notion=PrivacyNotion.APPROX_DP,
            num_paths=len(ratio_result.ratios),
            num_cross_paths=num_cross,
            approximations=approx_notes,
            is_sound=self._tracker.all_sound,
        )

    def encode_gaussian(
        self,
        delta_var: str,
        sigma_var: str,
        epsilon: float,
        target_delta: float,
    ) -> SMTEncoding:
        """Encode Gaussian mechanism (ε,δ)-DP constraint.

        For Gaussian noise, the exact formula is:
            δ = Φ(-ε·σ/Δ + Δ/(2σ)) - e^ε · Φ(-ε·σ/Δ - Δ/(2σ))

        We negate this: assert δ_computed > target_delta.

        Args:
            delta_var:    Name of the sensitivity variable Δ.
            sigma_var:    Name of the noise scale variable σ.
            epsilon:      Privacy parameter ε.
            target_delta: Target failure probability δ.

        Returns:
            SMTEncoding asserting the violation.
        """
        delta = z3.Real(delta_var)
        sigma = z3.Real(sigma_var)
        eps = z3.RealVal(str(epsilon))
        target = z3.RealVal(str(target_delta))

        # t1 = -ε·σ/Δ + Δ/(2σ)
        t1 = -eps * sigma / delta + delta / (z3.RealVal("2") * sigma)
        # t2 = -ε·σ/Δ - Δ/(2σ)
        t2 = -eps * sigma / delta - delta / (z3.RealVal("2") * sigma)

        phi_t1 = self._transcendental.approx_phi(t1)
        phi_t2 = self._transcendental.approx_phi(t2)
        exp_eps = self._transcendental.approx_exp(eps)

        # δ_computed = Φ(t1) - e^ε · Φ(t2)
        delta_computed = phi_t1.value - exp_eps.value * phi_t2.value

        assertions = [
            delta > z3.RealVal(0),
            sigma > z3.RealVal(0),
            delta_computed > target,
        ]

        encoding = SMTEncoding(
            variables={delta_var: delta, sigma_var: sigma},
            assertions=assertions,
            soundness=self._tracker,
            metadata={
                "type": "gaussian_approx_dp",
                "epsilon": epsilon,
                "target_delta": target_delta,
            },
        )
        encoding._rebuild_formula()
        return encoding

    def encode_gaussian_analytic(
        self,
        sensitivity: float,
        sigma: float,
        epsilon: float,
        target_delta: float,
    ) -> SMTEncoding:
        """Encode the analytic Gaussian mechanism check.

        Verifies numerically whether the Gaussian mechanism with given
        parameters satisfies (ε,δ)-DP.

        Args:
            sensitivity: Query sensitivity Δ.
            sigma:       Noise standard deviation σ.
            epsilon:     Privacy parameter ε.
            target_delta: Target failure probability.

        Returns:
            SMTEncoding asserting the violation (SAT = not private).
        """
        delta_z3 = z3.RealVal(str(sensitivity))
        sigma_z3 = z3.RealVal(str(sigma))
        eps_z3 = z3.RealVal(str(epsilon))
        target_z3 = z3.RealVal(str(target_delta))

        t1 = -eps_z3 * sigma_z3 / delta_z3 + delta_z3 / (z3.RealVal("2") * sigma_z3)
        t2 = -eps_z3 * sigma_z3 / delta_z3 - delta_z3 / (z3.RealVal("2") * sigma_z3)

        phi_t1 = self._transcendental.approx_phi(t1)
        phi_t2 = self._transcendental.approx_phi(t2)
        exp_eps = self._transcendental.approx_exp(eps_z3)

        computed = phi_t1.value - exp_eps.value * phi_t2.value

        # SAT if computed delta > target
        # Use a witness variable to allow model extraction
        witness = z3.Real("__witness")
        assertions = [
            witness == computed,
            witness > target_z3,
        ]

        encoding = SMTEncoding(
            variables={"__witness": witness},
            assertions=assertions,
            soundness=self._tracker,
            metadata={
                "type": "gaussian_analytic",
                "sensitivity": sensitivity,
                "sigma": sigma,
                "epsilon": epsilon,
            },
        )
        encoding._rebuild_formula()
        return encoding


# ═══════════════════════════════════════════════════════════════════════════
# ZCDP ENCODER
# ═══════════════════════════════════════════════════════════════════════════


class ZCDPEncoder(_BasePrivacyEncoder):
    """Encode zero-concentrated differential privacy (zCDP) constraints.

    zCDP requires: D_α(M(d) || M(d')) ≤ ρ·α for all α > 1,
    where D_α is the Rényi divergence of order α.

    For Gaussian noise with sensitivity Δ and scale σ:
        D_α = α·Δ²/(2σ²) ≤ ρ·α  →  Δ²/(2σ²) ≤ ρ

    Args:
        precision: Approximation precision level.
    """

    def __init__(self, precision: Precision = Precision.STANDARD) -> None:
        super().__init__(precision)

    def encode(
        self,
        ratio_result: DensityRatioResult,
        rho: float,
    ) -> PrivacyEncodingResult:
        """Encode the negation of ρ-zCDP for all paths.

        For each ratio, asserts ∃ α > 1 such that α·L²/(2) > ρ·α,
        which simplifies to L² > 2·ρ (sufficient for Gaussian).

        Args:
            ratio_result: Density ratio expressions.
            rho:          zCDP parameter ρ.

        Returns:
            PrivacyEncodingResult.
        """
        self._reset_converter()
        disjuncts: list[Any] = []
        num_cross = 0

        for ratio in ratio_result.ratios:
            path_constraints = self._encode_path_conditions(ratio)
            lr_z3 = self._encode_log_ratio(ratio)

            rho_z3 = z3.RealVal(str(rho))
            two = z3.RealVal("2")

            # zCDP violation: L² > 2ρ (for Gaussian, independent of α)
            violation = lr_z3 * lr_z3 > two * rho_z3

            if path_constraints:
                disjunct = z3.And(*path_constraints, violation)
            else:
                disjunct = violation
            disjuncts.append(disjunct)

            if ratio.is_cross_path:
                num_cross += 1

        assertions: list[Any] = []
        if not disjuncts:
            assertions.append(z3.BoolVal(False))
        elif len(disjuncts) == 1:
            assertions.append(disjuncts[0])
        else:
            assertions.append(z3.Or(*disjuncts))

        assertions.extend(self._converter.aux_constraints)

        encoding = SMTEncoding(
            variables=self._converter.variables,
            assertions=assertions,
            aux_vars=self._converter.aux_vars,
            soundness=self._tracker,
            metadata={"notion": "zcdp", "rho": rho},
        )
        encoding._rebuild_formula()

        return PrivacyEncodingResult(
            encoding=encoding,
            theory=SMTTheory.QF_NRA,
            notion=PrivacyNotion.ZCDP,
            num_paths=len(ratio_result.ratios),
            num_cross_paths=num_cross,
            is_sound=True,
        )

    def encode_gaussian(
        self,
        delta_var: str,
        sigma_var: str,
        rho: float,
    ) -> SMTEncoding:
        """Encode Gaussian mechanism zCDP constraint.

        zCDP for Gaussian: Δ²/(2σ²) ≤ ρ.
        Negation: Δ²/(2σ²) > ρ → Δ² > 2ρσ².

        Args:
            delta_var: Sensitivity variable name.
            sigma_var: Scale variable name.
            rho:       zCDP parameter.

        Returns:
            SMTEncoding asserting the violation.
        """
        delta = z3.Real(delta_var)
        sigma = z3.Real(sigma_var)
        rho_z3 = z3.RealVal(str(rho))
        two = z3.RealVal("2")

        assertions = [
            delta > z3.RealVal(0),
            sigma > z3.RealVal(0),
            # Negate: Δ² > 2ρσ²
            delta * delta > two * rho_z3 * sigma * sigma,
        ]

        encoding = SMTEncoding(
            variables={delta_var: delta, sigma_var: sigma},
            assertions=assertions,
            metadata={"type": "gaussian_zcdp", "rho": rho},
        )
        encoding._rebuild_formula()
        return encoding

    def encode_alpha_universal(
        self,
        delta_var: str,
        sigma_var: str,
        rho: float,
        alpha_values: list[float] | None = None,
    ) -> SMTEncoding:
        """Encode zCDP with explicit α-universality checking.

        Checks the Rényi divergence bound at multiple α values to
        verify ∀ α > 1: D_α ≤ ρ·α.

        Args:
            delta_var:    Sensitivity variable name.
            sigma_var:    Scale variable name.
            rho:          zCDP parameter.
            alpha_values: List of α values to check.

        Returns:
            SMTEncoding.
        """
        if alpha_values is None:
            alpha_values = [1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0]

        delta = z3.Real(delta_var)
        sigma = z3.Real(sigma_var)

        assertions = [
            delta > z3.RealVal(0),
            sigma > z3.RealVal(0),
        ]

        # For Gaussian: D_α = α·Δ²/(2σ²)
        # Violation at any α: α·Δ²/(2σ²) > ρ·α → Δ²/(2σ²) > ρ
        violation_disjuncts = []
        for alpha in alpha_values:
            alpha_z3 = z3.RealVal(str(alpha))
            rho_z3 = z3.RealVal(str(rho))
            two = z3.RealVal("2")
            # D_α = alpha * delta^2 / (2 * sigma^2) > rho * alpha
            renyi = alpha_z3 * delta * delta / (two * sigma * sigma)
            violation_disjuncts.append(renyi > rho_z3 * alpha_z3)

        assertions.append(z3.Or(*violation_disjuncts))

        encoding = SMTEncoding(
            variables={delta_var: delta, sigma_var: sigma},
            assertions=assertions,
            metadata={
                "type": "zcdp_alpha_universal",
                "rho": rho,
                "alphas": alpha_values,
            },
        )
        encoding._rebuild_formula()
        return encoding


# ═══════════════════════════════════════════════════════════════════════════
# RDP ENCODER
# ═══════════════════════════════════════════════════════════════════════════


class RDPEncoder(_BasePrivacyEncoder):
    """Encode Rényi differential privacy constraints.

    RDP of order α requires: D_α(M(d) || M(d')) ≤ ε_α.

    For Gaussian: D_α = α·Δ²/(2σ²)
    For Laplace:  D_α = (1/(α-1))·ln((α/(2α-1))·exp((α-1)/b) +
                        ((α-1)/(2α-1))·exp(-α/b))

    Args:
        precision: Approximation precision level.
    """

    def __init__(self, precision: Precision = Precision.STANDARD) -> None:
        super().__init__(precision)

    def encode(
        self,
        ratio_result: DensityRatioResult,
        alpha: float,
        eps_alpha: float | None = None,
        *,
        epsilon: float | None = None,
    ) -> PrivacyEncodingResult:
        """Encode the negation of (α, ε_α)-RDP for all paths.

        Args:
            ratio_result: Density ratio expressions.
            alpha:        Rényi order α > 1.
            eps_alpha:    RDP bound ε_α.
            epsilon:      Alias for eps_alpha.

        Returns:
            PrivacyEncodingResult.
        """
        if eps_alpha is None:
            eps_alpha = epsilon
        if eps_alpha is None:
            raise ValueError("Must provide eps_alpha or epsilon")
        self._reset_converter()
        disjuncts: list[Any] = []
        num_cross = 0

        for ratio in ratio_result.ratios:
            path_constraints = self._encode_path_conditions(ratio)
            lr_z3 = self._encode_log_ratio(ratio)

            alpha_z3 = z3.RealVal(str(alpha))
            eps_z3 = z3.RealVal(str(eps_alpha))

            # RDP violation: α·L²/2 > ε_α (for Gaussian-like)
            violation = alpha_z3 * lr_z3 * lr_z3 / z3.RealVal("2") > eps_z3

            if path_constraints:
                disjunct = z3.And(*path_constraints, violation)
            else:
                disjunct = violation
            disjuncts.append(disjunct)

            if ratio.is_cross_path:
                num_cross += 1

        assertions: list[Any] = []
        if not disjuncts:
            assertions.append(z3.BoolVal(False))
        elif len(disjuncts) == 1:
            assertions.append(disjuncts[0])
        else:
            assertions.append(z3.Or(*disjuncts))

        assertions.extend(self._converter.aux_constraints)

        encoding = SMTEncoding(
            variables=self._converter.variables,
            assertions=assertions,
            aux_vars=self._converter.aux_vars,
            soundness=self._tracker,
            metadata={
                "notion": "rdp",
                "alpha": alpha,
                "eps_alpha": eps_alpha,
            },
        )
        encoding._rebuild_formula()

        return PrivacyEncodingResult(
            encoding=encoding,
            theory=SMTTheory.QF_NRA,
            notion=PrivacyNotion.RDP,
            num_paths=len(ratio_result.ratios),
            num_cross_paths=num_cross,
            is_sound=True,
        )

    def encode_gaussian(
        self,
        delta_var: str,
        sigma_var: str,
        alpha: float,
        eps_alpha: float,
    ) -> SMTEncoding:
        """Encode Gaussian mechanism RDP constraint.

        For Gaussian: D_α = α·Δ²/(2σ²) ≤ ε_α.
        Negation: α·Δ²/(2σ²) > ε_α.

        Args:
            delta_var: Sensitivity variable name.
            sigma_var: Scale variable name.
            alpha:     Rényi order.
            eps_alpha: RDP bound.

        Returns:
            SMTEncoding asserting the violation.
        """
        delta = z3.Real(delta_var)
        sigma = z3.Real(sigma_var)
        alpha_z3 = z3.RealVal(str(alpha))
        eps_z3 = z3.RealVal(str(eps_alpha))
        two = z3.RealVal("2")

        assertions = [
            delta > z3.RealVal(0),
            sigma > z3.RealVal(0),
            alpha_z3 * delta * delta / (two * sigma * sigma) > eps_z3,
        ]

        encoding = SMTEncoding(
            variables={delta_var: delta, sigma_var: sigma},
            assertions=assertions,
            metadata={
                "type": "gaussian_rdp",
                "alpha": alpha,
                "eps_alpha": eps_alpha,
            },
        )
        encoding._rebuild_formula()
        return encoding

    def encode_laplace(
        self,
        delta_var: str,
        scale_var: str,
        alpha: float,
        eps_alpha: float,
    ) -> SMTEncoding:
        """Encode Laplace mechanism RDP constraint.

        For Laplace(b): D_α = (1/(α-1))·ln(
            (α/(2α-1))·exp((α-1)·Δ/b) + ((α-1)/(2α-1))·exp(-α·Δ/b)
        )

        Args:
            delta_var: Sensitivity variable name.
            scale_var: Scale variable name.
            alpha:     Rényi order α > 1.
            eps_alpha: RDP bound ε_α.

        Returns:
            SMTEncoding asserting the violation.
        """
        delta = z3.Real(delta_var)
        b = z3.Real(scale_var)
        alpha_z3 = z3.RealVal(str(alpha))
        alpha_m1 = z3.RealVal(str(alpha - 1.0))
        eps_z3 = z3.RealVal(str(eps_alpha))
        two_alpha_m1 = z3.RealVal(str(2 * alpha - 1.0))

        # exp((α-1)·Δ/b) and exp(-α·Δ/b)
        exp_arg1 = alpha_m1 * delta / b
        exp_arg2 = -alpha_z3 * delta / b
        exp1 = self._transcendental.approx_exp(exp_arg1)
        exp2 = self._transcendental.approx_exp(exp_arg2)

        # term = (α/(2α-1))·exp1 + ((α-1)/(2α-1))·exp2
        coeff1 = alpha_z3 / two_alpha_m1
        coeff2 = alpha_m1 / two_alpha_m1
        term = coeff1 * exp1.value + coeff2 * exp2.value

        # D_α = ln(term) / (α-1)
        ln_term = self._transcendental.approx_ln(term)
        renyi_div = ln_term.value / alpha_m1

        assertions = [
            delta > z3.RealVal(0),
            b > z3.RealVal(0),
            renyi_div > eps_z3,
        ]

        encoding = SMTEncoding(
            variables={delta_var: delta, scale_var: b},
            assertions=assertions,
            soundness=self._tracker,
            metadata={
                "type": "laplace_rdp",
                "alpha": alpha,
                "eps_alpha": eps_alpha,
            },
        )
        encoding._rebuild_formula()
        return encoding


# ═══════════════════════════════════════════════════════════════════════════
# GDP ENCODER
# ═══════════════════════════════════════════════════════════════════════════


class GDPEncoder(_BasePrivacyEncoder):
    """Encode Gaussian differential privacy (GDP) constraints.

    μ-GDP is a distributional property defined by the trade-off function
    T(α) = Φ(Φ⁻¹(1-α) - μ).  The encoder delegates to a grid-based
    trade-off check identical to :class:`FDPEncoder`.

    Args:
        precision:   Approximation precision level.
        grid_points: Number of grid points for checking the trade-off.
    """

    def __init__(
        self,
        precision: Precision = Precision.STANDARD,
        grid_points: int = 50,
    ) -> None:
        super().__init__(precision)
        self._grid_points = grid_points

    def encode(
        self,
        ratio_result: DensityRatioResult,
        mu: float,
    ) -> PrivacyEncodingResult:
        """Encode the negation of μ-GDP for all paths.

        GDP is checked via the Gaussian trade-off function
        T(α) = Φ(Φ⁻¹(1-α) - μ) evaluated at a grid of α values.

        Args:
            ratio_result: Density ratio expressions.
            mu:           GDP parameter μ.

        Returns:
            PrivacyEncodingResult.
        """
        from dpcegar.utils.math_utils import phi as std_phi, phi_inv as std_phi_inv

        def gaussian_tradeoff(alpha: float) -> float:
            """T(α) = Φ(Φ⁻¹(1-α) - μ) for μ-GDP."""
            if alpha <= 0.0 or alpha >= 1.0:
                return 0.0
            try:
                return std_phi(std_phi_inv(1.0 - alpha) - mu)
            except (ValueError, ZeroDivisionError):
                return 0.0

        fdp_encoder = FDPEncoder(
            precision=self._precision,
            grid_points=self._grid_points,
        )
        fdp_result = fdp_encoder.encode(ratio_result, trade_off_fn=gaussian_tradeoff)

        return PrivacyEncodingResult(
            encoding=fdp_result.encoding,
            theory=fdp_result.theory,
            notion=PrivacyNotion.GDP,
            num_paths=fdp_result.num_paths,
            num_cross_paths=fdp_result.num_cross_paths,
            approximations=fdp_result.approximations,
            is_sound=fdp_result.is_sound,
        )

    def encode_sensitivity_ratio(
        self,
        delta_var: str,
        sigma_var: str,
        mu: float,
    ) -> SMTEncoding:
        """Encode GDP constraint: Δ/σ ≤ μ.

        Negation: Δ/σ > μ → Δ > μ·σ.

        Args:
            delta_var: Sensitivity variable name.
            sigma_var: Scale variable name.
            mu:        GDP parameter.

        Returns:
            SMTEncoding asserting the violation.
        """
        delta = z3.Real(delta_var)
        sigma = z3.Real(sigma_var)
        mu_z3 = z3.RealVal(str(mu))

        assertions = [
            delta > z3.RealVal(0),
            sigma > z3.RealVal(0),
            # Negate: Δ > μ·σ
            delta > mu_z3 * sigma,
        ]

        encoding = SMTEncoding(
            variables={delta_var: delta, sigma_var: sigma},
            assertions=assertions,
            metadata={"type": "gdp", "mu": mu},
        )
        encoding._rebuild_formula()
        return encoding


# ═══════════════════════════════════════════════════════════════════════════
# FDP ENCODER
# ═══════════════════════════════════════════════════════════════════════════


class FDPEncoder(_BasePrivacyEncoder):
    """Encode f-differential privacy constraints.

    f-DP is defined by a trade-off function T(α) such that for all
    tests at significance level α, the type II error is ≥ T(α).

    The encoder checks the trade-off at a grid of α values.

    Args:
        precision:   Approximation precision level.
        grid_points: Number of grid points for checking the trade-off.
    """

    def __init__(
        self,
        precision: Precision = Precision.STANDARD,
        grid_points: int = 50,
    ) -> None:
        super().__init__(precision)
        self._grid_points = grid_points

    def encode(
        self,
        ratio_result: DensityRatioResult,
        trade_off_fn: Any,
        n_grid: int | None = None,
    ) -> PrivacyEncodingResult:
        """Encode the negation of f-DP at grid points.

        For each grid point α ∈ [0, 1], checks whether the mechanism's
        actual trade-off exceeds the declared trade-off function.

        Args:
            ratio_result: Density ratio expressions.
            trade_off_fn: Callable α → T(α), the declared trade-off.

        Returns:
            PrivacyEncodingResult.
        """
        if n_grid is not None:
            self._grid_points = n_grid
        self._reset_converter()
        all_violations: list[Any] = []
        num_cross = 0

        for ratio in ratio_result.ratios:
            path_constraints = self._encode_path_conditions(ratio)
            lr_z3 = self._encode_log_ratio(ratio)

            # Check at grid points
            grid_violations: list[Any] = []
            for i in range(self._grid_points + 1):
                alpha = i / self._grid_points
                t_alpha = trade_off_fn(alpha)
                alpha_z3 = z3.RealVal(str(alpha))
                t_z3 = z3.RealVal(str(t_alpha))

                # The actual type II error at significance α
                # For a likelihood ratio test: β = 1 - Φ(Φ⁻¹(1-α) - L)
                # Violation: β < T(α)
                phi_inv_arg = z3.RealVal(str(1.0 - alpha)) if alpha < 1.0 else z3.RealVal("0")
                phi_inv_val = self._transcendental.approx_phi_inv(phi_inv_arg)
                phi_arg = phi_inv_val.value - lr_z3
                phi_val = self._transcendental.approx_phi(phi_arg)

                # β = 1 - Φ(Φ⁻¹(1-α) - L)
                beta = z3.RealVal("1") - phi_val.value

                # Violation: β < T(α) at this grid point
                grid_violations.append(beta < t_z3)

            # Any grid point violation suffices
            if grid_violations:
                ratio_violation = z3.Or(*grid_violations)
                if path_constraints:
                    all_violations.append(z3.And(*path_constraints, ratio_violation))
                else:
                    all_violations.append(ratio_violation)

            if ratio.is_cross_path:
                num_cross += 1

        assertions: list[Any] = []
        if not all_violations:
            assertions.append(z3.BoolVal(False))
        elif len(all_violations) == 1:
            assertions.append(all_violations[0])
        else:
            assertions.append(z3.Or(*all_violations))

        assertions.extend(self._converter.aux_constraints)

        encoding = SMTEncoding(
            variables=self._converter.variables,
            assertions=assertions,
            aux_vars=self._converter.aux_vars,
            soundness=self._tracker,
            metadata={
                "notion": "fdp",
                "grid_points": self._grid_points,
            },
        )
        encoding._rebuild_formula()

        return PrivacyEncodingResult(
            encoding=encoding,
            theory=SMTTheory.QF_NRA,
            notion=PrivacyNotion.FDP,
            num_paths=len(ratio_result.ratios),
            num_cross_paths=num_cross,
            approximations=["phi_approx", "phi_inv_approx"],
            is_sound=self._tracker.all_sound,
        )

    def encode_gaussian_tradeoff(
        self,
        delta_var: str,
        sigma_var: str,
        mu: float,
        grid_points: int | None = None,
    ) -> SMTEncoding:
        """Encode Gaussian f-DP trade-off function check.

        For μ-GDP, the trade-off function is:
            T(α) = Φ(Φ⁻¹(1 - α) - μ)

        Args:
            delta_var:   Sensitivity variable name.
            sigma_var:   Scale variable name.
            mu:          GDP parameter (determines the trade-off).
            grid_points: Number of grid points.

        Returns:
            SMTEncoding.
        """
        n_pts = grid_points or self._grid_points
        delta = z3.Real(delta_var)
        sigma = z3.Real(sigma_var)

        assertions = [
            delta > z3.RealVal(0),
            sigma > z3.RealVal(0),
        ]

        # Actual ratio: Δ/σ
        actual_ratio = delta / sigma

        violations: list[Any] = []
        for i in range(1, n_pts):
            alpha = i / n_pts
            # Declared trade-off for μ-GDP
            from dpcegar.utils.math_utils import phi as std_phi, phi_inv as std_phi_inv
            try:
                threshold = std_phi(std_phi_inv(1.0 - alpha) - mu)
            except (ValueError, ZeroDivisionError):
                continue

            t_z3 = z3.RealVal(str(threshold))
            # Actual trade-off: Φ(Φ⁻¹(1-α) - Δ/σ)
            phi_inv_val = self._transcendental.approx_phi_inv(z3.RealVal(str(1.0 - alpha)))
            phi_val = self._transcendental.approx_phi(phi_inv_val.value - actual_ratio)

            # Violation: actual < declared
            violations.append(phi_val.value < t_z3)

        if violations:
            assertions.append(z3.Or(*violations))
        else:
            assertions.append(z3.BoolVal(False))

        encoding = SMTEncoding(
            variables={delta_var: delta, sigma_var: sigma},
            assertions=assertions,
            soundness=self._tracker,
            metadata={
                "type": "gaussian_fdp",
                "mu": mu,
                "grid_points": n_pts,
            },
        )
        encoding._rebuild_formula()
        return encoding


# ═══════════════════════════════════════════════════════════════════════════
# CROSS-PATH ENCODER
# ═══════════════════════════════════════════════════════════════════════════


class CrossPathEncoder(_BasePrivacyEncoder):
    """Handle data-dependent branching with path pair enumeration.

    When a mechanism has branches that depend on the database,
    different paths may be taken for d and d'.  This encoder
    constructs the privacy constraint for all pairs (πᵢ, πⱼ)
    where πᵢ is taken on d and πⱼ on d'.

    Args:
        precision:     Approximation precision level.
        inner_encoder: The privacy encoder to use for each pair.
    """

    def __init__(
        self,
        precision: Precision = Precision.STANDARD,
        inner_encoder: _BasePrivacyEncoder | None = None,
    ) -> None:
        super().__init__(precision)
        self._inner = inner_encoder

    def encode(
        self,
        ratio_result: DensityRatioResult,
        **kwargs: Any,
    ) -> PrivacyEncodingResult:
        """Convenience wrapper that delegates to encode_cross_paths.

        Constructs a PureBudget from keyword arguments and delegates.

        Args:
            ratio_result: Density ratio result.
            **kwargs:     Budget parameters (e.g. epsilon).

        Returns:
            PrivacyEncodingResult.
        """
        epsilon = kwargs.get("epsilon", 1.0)
        budget = PureBudget(epsilon=epsilon)
        return self.encode_cross_paths(ratio_result, budget)

    def encode_cross_paths(
        self,
        ratio_result: DensityRatioResult,
        budget: PrivacyBudget,
    ) -> PrivacyEncodingResult:
        """Encode privacy constraints across all path pairs.

        Enumerates cross-path ratios and encodes the appropriate
        privacy constraint based on the budget type.

        Args:
            ratio_result: Density ratio result with cross-path ratios.
            budget:       The declared privacy budget.

        Returns:
            PrivacyEncodingResult.
        """
        self._reset_converter()
        disjuncts: list[Any] = []

        for ratio in ratio_result.cross_path:
            path_constraints = self._encode_path_conditions(ratio)
            lr_z3 = self._encode_log_ratio(ratio)

            violation = self._encode_budget_violation(lr_z3, budget)

            if path_constraints:
                disjunct = z3.And(*path_constraints, violation)
            else:
                disjunct = violation
            disjuncts.append(disjunct)

        # Also include same-path ratios
        for ratio in ratio_result.same_path:
            path_constraints = self._encode_path_conditions(ratio)
            lr_z3 = self._encode_log_ratio(ratio)

            violation = self._encode_budget_violation(lr_z3, budget)

            if path_constraints:
                disjunct = z3.And(*path_constraints, violation)
            else:
                disjunct = violation
            disjuncts.append(disjunct)

        assertions: list[Any] = []
        if not disjuncts:
            assertions.append(z3.BoolVal(False))
        elif len(disjuncts) == 1:
            assertions.append(disjuncts[0])
        else:
            assertions.append(z3.Or(*disjuncts))

        assertions.extend(self._converter.aux_constraints)

        encoding = SMTEncoding(
            variables=self._converter.variables,
            assertions=assertions,
            aux_vars=self._converter.aux_vars,
            soundness=self._tracker,
            metadata={
                "type": "cross_path",
                "num_same_path": len(ratio_result.same_path),
                "num_cross_path": len(ratio_result.cross_path),
            },
        )
        encoding._rebuild_formula()

        notion = budget.notion
        return PrivacyEncodingResult(
            encoding=encoding,
            theory=SMTTheory.QF_NRA,
            notion=notion,
            num_paths=len(ratio_result.ratios),
            num_cross_paths=len(ratio_result.cross_path),
            is_sound=self._tracker.all_sound,
        )

    def _encode_budget_violation(self, lr_z3: Any, budget: PrivacyBudget) -> Any:
        """Encode budget violation for the given privacy notion.

        Args:
            lr_z3:  Z3 expression for the log ratio.
            budget: Privacy budget.

        Returns:
            Z3 boolean expression asserting violation.
        """
        if isinstance(budget, PureBudget):
            eps_z3 = z3.RealVal(str(budget.epsilon))
            return z3.Or(lr_z3 > eps_z3, lr_z3 < -eps_z3)

        elif isinstance(budget, ApproxBudget):
            eps_z3 = z3.RealVal(str(budget.epsilon))
            delta_z3 = z3.RealVal(str(budget.delta))
            exp_L = self._transcendental.approx_exp(lr_z3)
            exp_eps = self._transcendental.approx_exp(eps_z3)
            integrand = z3.If(
                exp_L.value > exp_eps.value,
                exp_L.value - exp_eps.value,
                z3.RealVal(0),
            )
            return integrand > delta_z3

        elif isinstance(budget, ZCDPBudget):
            rho_z3 = z3.RealVal(str(budget.rho))
            return lr_z3 * lr_z3 > z3.RealVal("2") * rho_z3

        elif isinstance(budget, RDPBudget):
            alpha_z3 = z3.RealVal(str(budget.alpha))
            eps_z3 = z3.RealVal(str(budget.epsilon))
            return alpha_z3 * lr_z3 * lr_z3 / z3.RealVal("2") > eps_z3

        elif isinstance(budget, GDPBudget):
            mu_z3 = z3.RealVal(str(budget.mu))
            return z3.Or(lr_z3 > mu_z3, lr_z3 < -mu_z3)

        else:
            # Default: pure DP with epsilon from approx conversion
            eps, _ = budget.to_approx_dp()
            eps_z3 = z3.RealVal(str(eps))
            return z3.Or(lr_z3 > eps_z3, lr_z3 < -eps_z3)

    def enumerate_path_pairs(
        self,
        paths: Sequence[SymbolicPath],
    ) -> list[tuple[SymbolicPath, SymbolicPath]]:
        """Enumerate all pairs of paths for cross-path analysis.

        Each pair (πᵢ, πⱼ) represents path πᵢ taken on d and πⱼ on d'.

        Args:
            paths: List of symbolic paths.

        Returns:
            List of (d-path, d'-path) pairs.
        """
        pairs = []
        for pi_d in paths:
            for pi_dp in paths:
                pairs.append((pi_d, pi_dp))
        return pairs

    def filter_feasible_pairs(
        self,
        pairs: list[tuple[SymbolicPath, SymbolicPath]],
    ) -> list[tuple[SymbolicPath, SymbolicPath]]:
        """Filter path pairs that are syntactically infeasible.

        Removes pairs where the conjunction of path conditions is
        trivially unsatisfiable.

        Args:
            pairs: List of path pairs.

        Returns:
            Filtered list of feasible pairs.
        """
        feasible = []
        for pi_d, pi_dp in pairs:
            combined = pi_d.path_condition.and_(pi_dp.path_condition)
            if not combined.simplify().is_trivially_false():
                feasible.append((pi_d, pi_dp))
        return feasible
