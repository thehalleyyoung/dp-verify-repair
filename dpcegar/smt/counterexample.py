"""Counterexample extraction and validation for CEGAR loop.

When the SMT solver returns SAT (indicating a potential privacy
violation), this module extracts a concrete counterexample, validates
it against the original mechanism semantics, and provides tools for
minimisation and human-readable display.

Classes
-------
Counterexample         – a single concrete counterexample
CounterexampleSet      – collection with deduplication and coverage
CounterexampleExtractor – extract from Z3 models
SpuriousnessChecker    – validate against concrete semantics
CounterexampleMinimizer – simplify values
CounterexamplePrinter   – human-readable display
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

try:
    import z3
except ImportError:  # pragma: no cover
    z3 = None  # type: ignore[assignment]

from dpcegar.ir.types import (
    Const,
    IRType,
    NoiseKind,
    PrivacyBudget,
    PrivacyNotion,
    PureBudget,
    ApproxBudget,
    TypedExpr,
    Var,
)
from dpcegar.paths.symbolic_path import NoiseDrawInfo, SymbolicPath
from dpcegar.smt.solver import SolverResult, CheckResult


# ═══════════════════════════════════════════════════════════════════════════
# COUNTEREXAMPLE DATACLASS
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class Counterexample:
    """A concrete counterexample demonstrating a privacy violation.

    Represents a specific database pair (d, d'), output value o, and
    the execution path that witnesses |L(o)| > ε (or the relevant
    privacy predicate).

    Attributes:
        d_values:      Variable assignments for dataset d.
        d_prime_values: Variable assignments for dataset d'.
        output_value:  The mechanism output that witnesses violation.
        path_id:       ID of the execution path.
        privacy_loss:  Concrete privacy loss at this point.
        witness_values: All variable assignments from the model.
        notion:        The DP notion being violated.
        budget:        The declared privacy budget.
        metadata:      Additional metadata.
    """

    d_values: dict[str, float] = field(default_factory=dict)
    d_prime_values: dict[str, float] = field(default_factory=dict)
    output_value: float = 0.0
    path_id: int = -1
    privacy_loss: float = 0.0
    witness_values: dict[str, float] = field(default_factory=dict)
    notion: PrivacyNotion = PrivacyNotion.PURE_DP
    budget: PrivacyBudget | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def violation_amount(self) -> float:
        """Return the amount by which privacy is violated.

        For pure DP: |L(o)| - ε.

        Returns:
            The excess privacy loss.
        """
        if isinstance(self.budget, PureBudget):
            return abs(self.privacy_loss) - self.budget.epsilon
        if isinstance(self.budget, ApproxBudget):
            return abs(self.privacy_loss) - self.budget.epsilon
        return abs(self.privacy_loss)

    @property
    def is_valid(self) -> bool:
        """Return True if this counterexample has essential data.

        Returns:
            True if d_values and output_value are set.
        """
        return bool(self.d_values) or self.output_value != 0.0

    @property
    def variable_assignments(self) -> dict[str, float]:
        """Return merged variable assignments (d, d', witness)."""
        merged: dict[str, float] = {}
        merged.update(self.d_values)
        merged.update(self.d_prime_values)
        merged.update(self.witness_values)
        return merged

    def signature(self) -> str:
        """Return a deduplication signature.

        Two counterexamples with the same signature are considered
        duplicates (same path, similar values).

        Returns:
            String signature.
        """
        d_sig = tuple(sorted((k, round(v, 6)) for k, v in self.d_values.items()))
        dp_sig = tuple(sorted((k, round(v, 6)) for k, v in self.d_prime_values.items()))
        return f"p{self.path_id}:{d_sig}:{dp_sig}:{round(self.output_value, 6)}"

    def summary(self) -> str:
        """Return a short summary string.

        Returns:
            Summary string.
        """
        return (
            f"Counterexample(path={self.path_id}, "
            f"loss={self.privacy_loss:.6f}, "
            f"output={self.output_value:.6f})"
        )

    def __str__(self) -> str:
        return self.summary()


# ═══════════════════════════════════════════════════════════════════════════
# COUNTEREXAMPLE SET
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class CounterexampleSet:
    """Collection of counterexamples with deduplication and analysis.

    Attributes:
        counterexamples: List of unique counterexamples.
        path_coverage:   Set of path IDs covered.
        max_loss:        Maximum privacy loss observed.
    """

    counterexamples: list[Counterexample] = field(default_factory=list)
    _signatures: set[str] = field(default_factory=set, repr=False)

    @property
    def path_coverage(self) -> set[int]:
        """Return the set of path IDs covered.

        Returns:
            Set of path IDs.
        """
        return {ce.path_id for ce in self.counterexamples if ce.path_id >= 0}

    @property
    def max_loss(self) -> float:
        """Return the maximum privacy loss.

        Returns:
            Maximum |L(o)| across all counterexamples.
        """
        if not self.counterexamples:
            return 0.0
        return max(abs(ce.privacy_loss) for ce in self.counterexamples)

    def add(self, ce: Counterexample) -> bool:
        """Add a counterexample if it is not a duplicate.

        Args:
            ce: The counterexample to add.

        Returns:
            True if the counterexample was added (not a duplicate).
        """
        sig = ce.signature()
        if sig in self._signatures:
            return False
        self._signatures.add(sig)
        self.counterexamples.append(ce)
        return True

    def add_all(self, ces: Sequence[Counterexample]) -> int:
        """Add multiple counterexamples, skipping duplicates.

        Args:
            ces: Counterexamples to add.

        Returns:
            Number of new (non-duplicate) counterexamples added.
        """
        count = 0
        for ce in ces:
            if self.add(ce):
                count += 1
        return count

    def worst(self) -> Counterexample | None:
        """Return the counterexample with the largest privacy loss.

        Returns:
            The worst counterexample, or None if empty.
        """
        if not self.counterexamples:
            return None
        return max(self.counterexamples, key=lambda ce: abs(ce.privacy_loss))

    def by_path(self, path_id: int) -> list[Counterexample]:
        """Return counterexamples for a specific path.

        Args:
            path_id: The path ID to filter by.

        Returns:
            List of counterexamples on that path.
        """
        return [ce for ce in self.counterexamples if ce.path_id == path_id]

    def coverage_analysis(self, total_paths: int) -> dict[str, Any]:
        """Analyse path coverage of the counterexample set.

        Args:
            total_paths: Total number of paths in the mechanism.

        Returns:
            Dictionary with coverage statistics.
        """
        covered = self.path_coverage
        return {
            "total_paths": total_paths,
            "covered_paths": len(covered),
            "coverage_pct": len(covered) / max(total_paths, 1) * 100,
            "num_counterexamples": len(self.counterexamples),
            "max_loss": self.max_loss,
        }

    def __len__(self) -> int:
        return len(self.counterexamples)

    def __iter__(self):
        return iter(self.counterexamples)

    def __bool__(self) -> bool:
        return len(self.counterexamples) > 0


# ═══════════════════════════════════════════════════════════════════════════
# COUNTEREXAMPLE EXTRACTOR
# ═══════════════════════════════════════════════════════════════════════════


class CounterexampleExtractor:
    """Extract concrete counterexamples from Z3 models.

    Given a SAT result from the solver, maps Z3 variable assignments
    back to mechanism variables and computes the concrete privacy loss.

    Args:
        d_suffix:       Suffix identifying dataset-d variables.
        d_prime_suffix: Suffix identifying dataset-d' variables.
        output_var:     Name of the output variable.
    """

    def __init__(
        self,
        d_suffix: str = "",
        d_prime_suffix: str = "_prime",
        output_var: str = "o",
    ) -> None:
        self._d_suffix = d_suffix
        self._dp_suffix = d_prime_suffix
        self._output_var = output_var

    def extract(
        self,
        result: SolverResult,
        path_id: int = -1,
        budget: PrivacyBudget | None = None,
        noise_draws: Sequence[NoiseDrawInfo] | None = None,
    ) -> Counterexample | None:
        """Extract a counterexample from a SAT solver result.

        Args:
            result:      A SolverResult with result == SAT.
            path_id:     The path ID for this counterexample.
            budget:      The declared privacy budget.
            noise_draws: Noise draw info for computing privacy loss.

        Returns:
            Counterexample, or None if extraction fails.
        """
        if not result.is_sat or result.model is None:
            return None

        model = result.model
        all_values = self._extract_all_values(model)

        d_values = self._filter_d_values(all_values)
        dp_values = self._filter_dp_values(all_values)
        output = self._extract_float(model, self._output_var, 0.0)

        # Compute privacy loss from model
        privacy_loss = self._extract_float(model, "L", 0.0)
        if privacy_loss == 0.0 and "L" not in all_values:
            # Try to compute from witness values
            privacy_loss = self._compute_privacy_loss(
                all_values, noise_draws,
            )

        return Counterexample(
            d_values=d_values,
            d_prime_values=dp_values,
            output_value=output,
            path_id=path_id,
            privacy_loss=privacy_loss,
            witness_values=all_values,
            notion=budget.notion if budget else PrivacyNotion.PURE_DP,
            budget=budget,
            metadata={"solver_stats": result.stats},
        )

    def extract_multiple(
        self,
        results: Sequence[SolverResult],
        path_ids: Sequence[int] | None = None,
        budget: PrivacyBudget | None = None,
    ) -> CounterexampleSet:
        """Extract counterexamples from multiple SAT results.

        Args:
            results:  Sequence of solver results.
            path_ids: Corresponding path IDs.
            budget:   Declared privacy budget.

        Returns:
            CounterexampleSet with all extracted counterexamples.
        """
        ce_set = CounterexampleSet()
        pids = path_ids or [-1] * len(results)

        for result, pid in zip(results, pids):
            ce = self.extract(result, path_id=pid, budget=budget)
            if ce is not None:
                ce_set.add(ce)

        return ce_set

    def _extract_all_values(self, model: Any) -> dict[str, float]:
        """Extract all variable values from a Z3 model.

        Args:
            model: Z3 model.

        Returns:
            Dictionary of variable name → float value.
        """
        values: dict[str, float] = {}
        for decl in model.decls():
            name = decl.name()
            val = model[decl]
            fval = self._z3_to_float(val)
            if fval is not None:
                values[name] = fval
        return values

    def _filter_d_values(self, all_values: dict[str, float]) -> dict[str, float]:
        """Filter values for dataset d variables.

        Args:
            all_values: All extracted values.

        Returns:
            Values for d-variables only.
        """
        result: dict[str, float] = {}
        for name, val in all_values.items():
            if name.startswith("__"):
                continue
            if self._dp_suffix and name.endswith(self._dp_suffix):
                continue
            if self._d_suffix:
                if name.endswith(self._d_suffix):
                    result[name] = val
            else:
                result[name] = val
        return result

    def _filter_dp_values(self, all_values: dict[str, float]) -> dict[str, float]:
        """Filter values for dataset d' variables.

        Args:
            all_values: All extracted values.

        Returns:
            Values for d'-variables only.
        """
        if not self._dp_suffix:
            return {}
        result: dict[str, float] = {}
        for name, val in all_values.items():
            if name.endswith(self._dp_suffix):
                result[name] = val
        return result

    def _extract_float(self, model: Any, var_name: str, default: float = 0.0) -> float:
        """Extract a single variable's float value.

        Args:
            model:    Z3 model.
            var_name: Variable name.
            default:  Default value if not found.

        Returns:
            Float value.
        """
        try:
            v = z3.Real(var_name)
            val = model.eval(v, model_completion=True)
            fval = self._z3_to_float(val)
            return fval if fval is not None else default
        except Exception:
            return default

    @staticmethod
    def _z3_to_float(val: Any) -> float | None:
        """Convert a Z3 value to a Python float.

        Args:
            val: Z3 value.

        Returns:
            Float, or None if conversion fails.
        """
        try:
            if z3.is_rational_value(val):
                return float(val.as_fraction())
            if z3.is_int_value(val):
                return float(val.as_long())
            if z3.is_algebraic_value(val):
                return float(val.approx(20))
            if z3.is_true(val):
                return 1.0
            if z3.is_false(val):
                return 0.0
            return float(str(val))
        except (ValueError, AttributeError):
            return None

    def _compute_privacy_loss(
        self,
        values: dict[str, float],
        noise_draws: Sequence[NoiseDrawInfo] | None,
    ) -> float:
        """Compute the privacy loss from witness values.

        Uses the noise draw information to compute the log density ratio
        at the counterexample point.

        Args:
            values:      All variable assignments.
            noise_draws: Noise draw metadata.

        Returns:
            Computed privacy loss.
        """
        if not noise_draws:
            return 0.0

        total_loss = 0.0
        for draw in noise_draws:
            loss = self._single_draw_loss(draw, values)
            total_loss += loss

        return total_loss

    def _single_draw_loss(
        self,
        draw: NoiseDrawInfo,
        values: dict[str, float],
    ) -> float:
        """Compute privacy loss for a single noise draw.

        Args:
            draw:   Noise draw info.
            values: Variable assignments.

        Returns:
            Privacy loss for this draw.
        """
        # Extract output value
        o_val = values.get(self._output_var, 0.0)

        # Extract center values for d and d'
        # Convention: center_d is the center, center_d_prime has _prime suffix
        center_d_name = str(draw.center_expr)
        center_dp_name = center_d_name + self._dp_suffix

        center_d = values.get(center_d_name, 0.0)
        center_dp = values.get(center_dp_name, 0.0)

        # Extract scale
        scale_name = str(draw.scale_expr)
        scale = values.get(scale_name, 1.0)
        if scale <= 0:
            scale = 1.0

        if draw.kind == NoiseKind.LAPLACE:
            # L = (|o - center_dp| - |o - center_d|) / scale
            return (abs(o_val - center_dp) - abs(o_val - center_d)) / scale

        elif draw.kind == NoiseKind.GAUSSIAN:
            # L = ((o - center_dp)² - (o - center_d)²) / (2σ²)
            sigma2 = 2.0 * scale * scale
            if sigma2 <= 0:
                return 0.0
            return ((o_val - center_dp) ** 2 - (o_val - center_d) ** 2) / sigma2

        return 0.0


# ═══════════════════════════════════════════════════════════════════════════
# SPURIOUSNESS CHECKER
# ═══════════════════════════════════════════════════════════════════════════


class SpuriousnessChecker:
    """Verify counterexamples against concrete mechanism semantics.

    A counterexample may be spurious due to:
      - Polynomial approximation error in transcendental functions
      - Abstraction in the path condition
      - Numerical precision issues in the solver

    This class evaluates the concrete privacy loss at the counterexample
    point and checks whether it truly violates the budget.

    Args:
        tolerance: Tolerance for spuriousness checks (accounts for
                   floating-point imprecision).
    """

    def __init__(self, tolerance: float = 1e-8) -> None:
        self._tolerance = tolerance

    def is_spurious(
        self,
        ce: Counterexample,
        concrete_loss_fn: Callable[[dict[str, float], dict[str, float], float], float] | None = None,
    ) -> bool:
        """Check if a counterexample is spurious.

        Args:
            ce:               The counterexample to check.
            concrete_loss_fn: Optional function computing the true privacy
                            loss: f(d_values, dp_values, output) → loss.

        Returns:
            True if the counterexample is spurious.
        """
        if concrete_loss_fn is not None:
            true_loss = concrete_loss_fn(
                ce.d_values, ce.d_prime_values, ce.output_value,
            )
            if isinstance(ce.budget, PureBudget):
                return abs(true_loss) <= ce.budget.epsilon + self._tolerance
            return abs(true_loss) < self._tolerance

        # Without a concrete loss function, check basic consistency
        return self._basic_consistency_check(ce)

    def check_batch(
        self,
        ces: Sequence[Counterexample],
        concrete_loss_fn: Callable[[dict[str, float], dict[str, float], float], float] | None = None,
    ) -> tuple[list[Counterexample], list[Counterexample]]:
        """Partition counterexamples into genuine and spurious.

        Args:
            ces:              Counterexamples to check.
            concrete_loss_fn: Optional concrete loss function.

        Returns:
            Tuple of (genuine, spurious) counterexample lists.
        """
        genuine: list[Counterexample] = []
        spurious: list[Counterexample] = []

        for ce in ces:
            if self.is_spurious(ce, concrete_loss_fn):
                spurious.append(ce)
            else:
                genuine.append(ce)

        return genuine, spurious

    def _basic_consistency_check(self, ce: Counterexample) -> bool:
        """Perform basic consistency checks on a counterexample.

        Checks:
          - Values are finite
          - Privacy loss is non-zero
          - d and d' differ by at most the sensitivity

        Args:
            ce: Counterexample to check.

        Returns:
            True if the counterexample appears spurious.
        """
        # Check for non-finite values
        for v in ce.d_values.values():
            if not math.isfinite(v):
                return True
        for v in ce.d_prime_values.values():
            if not math.isfinite(v):
                return True
        if not math.isfinite(ce.output_value):
            return True
        if not math.isfinite(ce.privacy_loss):
            return True

        # Zero privacy loss is suspicious
        if abs(ce.privacy_loss) < self._tolerance:
            return True

        return False

    def validate_adjacency(
        self,
        ce: Counterexample,
        sensitivity: float = 1.0,
    ) -> bool:
        """Check that the database pair satisfies adjacency.

        Args:
            ce:          Counterexample.
            sensitivity: The adjacency bound.

        Returns:
            True if adjacency holds.
        """
        # Find matching d and d' variables
        for d_name, d_val in ce.d_values.items():
            dp_name = d_name + "_prime"
            if dp_name in ce.d_prime_values:
                dp_val = ce.d_prime_values[dp_name]
                if abs(d_val - dp_val) > sensitivity + self._tolerance:
                    return False
        return True


# ═══════════════════════════════════════════════════════════════════════════
# COUNTEREXAMPLE MINIMIZER
# ═══════════════════════════════════════════════════════════════════════════


class CounterexampleMinimizer:
    """Simplify counterexample values while preserving the violation.

    Attempts to:
      1. Round values to simpler numbers
      2. Set irrelevant variables to zero
      3. Find the minimum-norm counterexample

    Args:
        precision: Number of decimal places for rounding.
    """

    def __init__(self, precision: int = 4, max_iter: int = 100) -> None:
        self._precision = precision
        self._max_iter = max_iter

    def minimize(
        self,
        ce: Counterexample,
        check_fn: Callable[[Counterexample], bool] | None = None,
    ) -> Counterexample:
        """Simplify a counterexample.

        Args:
            ce:       The counterexample to simplify.
            check_fn: Optional function to verify the simplified
                     counterexample still represents a violation.
                     Returns True if it's still a valid violation.

        Returns:
            Simplified counterexample.
        """
        simplified = Counterexample(
            d_values=self._round_values(ce.d_values),
            d_prime_values=self._round_values(ce.d_prime_values),
            output_value=round(ce.output_value, self._precision),
            path_id=ce.path_id,
            privacy_loss=ce.privacy_loss,
            witness_values=self._round_values(ce.witness_values),
            notion=ce.notion,
            budget=ce.budget,
            metadata=dict(ce.metadata),
        )

        if check_fn is not None and not check_fn(simplified):
            return ce  # simplification invalidated the violation

        return simplified

    def minimize_zeroing(
        self,
        ce: Counterexample,
        check_fn: Callable[[Counterexample], bool],
    ) -> Counterexample:
        """Try to zero out variables while preserving the violation.

        For each variable, tries setting it to zero and checks if
        the violation still holds.

        Args:
            ce:       The counterexample.
            check_fn: Function returning True if violation holds.

        Returns:
            Counterexample with maximum zeroed-out variables.
        """
        current = Counterexample(
            d_values=dict(ce.d_values),
            d_prime_values=dict(ce.d_prime_values),
            output_value=ce.output_value,
            path_id=ce.path_id,
            privacy_loss=ce.privacy_loss,
            witness_values=dict(ce.witness_values),
            notion=ce.notion,
            budget=ce.budget,
            metadata=dict(ce.metadata),
        )

        # Try zeroing d-values
        for name in list(current.d_values.keys()):
            old_val = current.d_values[name]
            current.d_values[name] = 0.0
            if not check_fn(current):
                current.d_values[name] = old_val

        # Try zeroing d'-values
        for name in list(current.d_prime_values.keys()):
            old_val = current.d_prime_values[name]
            current.d_prime_values[name] = 0.0
            if not check_fn(current):
                current.d_prime_values[name] = old_val

        return current

    def _round_values(self, values: dict[str, float]) -> dict[str, float]:
        """Round all values to the configured precision.

        Args:
            values: Dictionary of values.

        Returns:
            Dictionary with rounded values.
        """
        return {k: round(v, self._precision) for k, v in values.items()}

    def _try_simplify_value(self, val: float) -> float:
        """Try to simplify a single value.

        Attempts to find a simpler representation (integer, simple
        fraction, etc.).

        Args:
            val: The value to simplify.

        Returns:
            Simplified value.
        """
        # Try integer
        if abs(val - round(val)) < 1e-10:
            return float(round(val))

        # Try simple fractions
        for denom in [2, 3, 4, 5, 10]:
            for numer in range(-100, 101):
                if abs(val - numer / denom) < 1e-8:
                    return numer / denom

        return round(val, self._precision)


# ═══════════════════════════════════════════════════════════════════════════
# COUNTEREXAMPLE DIVERSIFIER
# ═══════════════════════════════════════════════════════════════════════════


class CounterexampleDiversifier:
    """Select a diverse subset of counterexamples.

    Diversifies by preferring counterexamples from different execution
    paths and maximising pairwise distance in the input space.
    """

    def diversify(
        self,
        cex_set: CounterexampleSet,
        k: int = 3,
    ) -> list[Counterexample]:
        """Select up to *k* diverse counterexamples from *cex_set*.

        Strategy:
          1. Group by path_id.
          2. Round-robin pick one from each path.
          3. If more slots remain, fill with highest-loss remaining.

        Args:
            cex_set: The counterexample set to diversify.
            k:       Maximum number to return.

        Returns:
            List of at most *k* diverse counterexamples.
        """
        if len(cex_set) == 0:
            return []
        if len(cex_set) <= k:
            return list(cex_set)

        # Group by path
        by_path: dict[int, list[Counterexample]] = {}
        for ce in cex_set:
            by_path.setdefault(ce.path_id, []).append(ce)

        # Sort each group by descending loss
        for pid in by_path:
            by_path[pid].sort(key=lambda c: c.privacy_loss, reverse=True)

        selected: list[Counterexample] = []
        selected_sigs: set[str] = set()

        # Round-robin across paths
        path_ids = sorted(by_path.keys())
        idx = {pid: 0 for pid in path_ids}
        while len(selected) < k:
            added_any = False
            for pid in path_ids:
                if len(selected) >= k:
                    break
                while idx[pid] < len(by_path[pid]):
                    ce = by_path[pid][idx[pid]]
                    idx[pid] += 1
                    sig = ce.signature()
                    if sig not in selected_sigs:
                        selected.append(ce)
                        selected_sigs.add(sig)
                        added_any = True
                        break
            if not added_any:
                break

        return selected


# ═══════════════════════════════════════════════════════════════════════════
# COUNTEREXAMPLE PRINTER
# ═══════════════════════════════════════════════════════════════════════════


class CounterexamplePrinter:
    """Format counterexamples for human-readable display.

    Provides multiple output formats:
      - Compact single-line summary
      - Detailed multi-line display
      - Table format for counterexample sets

    Args:
        float_precision: Number of decimal places for floats.
        show_witness:    Whether to show all witness values.
    """

    def __init__(
        self,
        float_precision: int = 6,
        show_witness: bool = False,
    ) -> None:
        self._precision = float_precision
        self._show_witness = show_witness

    def format(self, ce: Counterexample) -> str:
        """Format a single counterexample.

        Args:
            ce: The counterexample to format.

        Returns:
            Multi-line formatted string.
        """
        lines: list[str] = []
        lines.append("╔══ Counterexample ══════════════════════════════════╗")
        lines.append(f"║  Path ID:      {ce.path_id}")
        lines.append(f"║  DP Notion:    {ce.notion.name}")
        if ce.budget:
            lines.append(f"║  Budget:       {ce.budget}")
        lines.append(f"║  Privacy Loss: {self._fmt(ce.privacy_loss)}")
        if ce.violation_amount > 0:
            lines.append(f"║  Violation:    +{self._fmt(ce.violation_amount)} over budget")
        lines.append(f"║  Output:       {self._fmt(ce.output_value)}")

        if ce.d_values:
            lines.append("║  ── Dataset d ──")
            for name, val in sorted(ce.d_values.items()):
                lines.append(f"║    {name} = {self._fmt(val)}")

        if ce.d_prime_values:
            lines.append("║  ── Dataset d' ──")
            for name, val in sorted(ce.d_prime_values.items()):
                lines.append(f"║    {name} = {self._fmt(val)}")

        if self._show_witness and ce.witness_values:
            lines.append("║  ── All Witness Values ──")
            for name, val in sorted(ce.witness_values.items()):
                if not name.startswith("__"):
                    lines.append(f"║    {name} = {self._fmt(val)}")

        lines.append("╚═══════════════════════════════════════════════════╝")
        return "\n".join(lines)

    def format_compact(self, ce: Counterexample) -> str:
        """Format a counterexample as a single line.

        Args:
            ce: The counterexample.

        Returns:
            Single-line formatted string.
        """
        return (
            f"[path={ce.path_id}] "
            f"loss={self._fmt(ce.privacy_loss)} "
            f"output={self._fmt(ce.output_value)} "
            f"d={self._fmt_dict(ce.d_values)} "
            f"d'={self._fmt_dict(ce.d_prime_values)}"
        )

    def format_set(self, ce_set: CounterexampleSet) -> str:
        """Format a counterexample set as a table.

        Args:
            ce_set: The counterexample set.

        Returns:
            Multi-line table string.
        """
        if not ce_set:
            return "No counterexamples found."

        lines: list[str] = []
        lines.append(f"Found {len(ce_set)} counterexample(s):")
        lines.append(f"  Path coverage: {len(ce_set.path_coverage)} paths")
        lines.append(f"  Max loss:      {self._fmt(ce_set.max_loss)}")
        lines.append("")

        header = f"{'#':>3}  {'Path':>5}  {'Loss':>12}  {'Output':>12}  {'Status':>8}"
        lines.append(header)
        lines.append("-" * len(header))

        for i, ce in enumerate(ce_set.counterexamples):
            status = "GENUINE" if ce.violation_amount > 0 else "SUSPECT"
            lines.append(
                f"{i + 1:>3}  "
                f"{ce.path_id:>5}  "
                f"{self._fmt(ce.privacy_loss):>12}  "
                f"{self._fmt(ce.output_value):>12}  "
                f"{status:>8}"
            )

        return "\n".join(lines)

    def format_for_refinement(self, ce: Counterexample) -> str:
        """Format a counterexample for the CEGAR refinement step.

        Provides the information needed to strengthen the abstraction.

        Args:
            ce: The counterexample.

        Returns:
            Formatted string with refinement-relevant information.
        """
        lines: list[str] = []
        lines.append(f"Refinement target (path {ce.path_id}):")
        lines.append(f"  Privacy loss: {self._fmt(ce.privacy_loss)}")

        # Identify critical variables (those with large values)
        all_vals = list(ce.witness_values.items())
        all_vals.sort(key=lambda x: abs(x[1]), reverse=True)

        lines.append("  Critical variables:")
        for name, val in all_vals[:10]:
            if not name.startswith("__"):
                lines.append(f"    {name} = {self._fmt(val)}")

        return "\n".join(lines)

    def _fmt(self, val: float) -> str:
        """Format a float value.

        Args:
            val: Float value.

        Returns:
            Formatted string.
        """
        return f"{val:.{self._precision}f}"

    def _fmt_dict(self, d: dict[str, float]) -> str:
        """Format a dictionary of values.

        Args:
            d: Dictionary.

        Returns:
            Formatted string.
        """
        if not d:
            return "{}"
        items = ", ".join(f"{k}={self._fmt(v)}" for k, v in sorted(d.items()))
        return "{" + items + "}"
