"""Optimization Modulo Theories (OMT) for privacy verification.

Wraps Z3's Optimize solver to support:
  - Minimisation / maximisation of linear or nonlinear objectives
  - Soft constraints with priorities
  - Pareto-optimal solving for multi-objective problems
  - Linearisation of |x| for LP-compatible objectives
  - Cost functions for repair minimality in CEGAR

Classes
-------
LinearObjective   – a weighted sum of variables
OMTResult         – result of an optimisation query
OMTSolver         – wrapper around z3.Optimize
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import z3
except ImportError:  # pragma: no cover
    z3 = None  # type: ignore[assignment]

from dpcegar.smt.encoding import SMTEncoding
from dpcegar.smt.solver import CheckResult, SolverResult


# ═══════════════════════════════════════════════════════════════════════════
# LINEAR OBJECTIVE
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class LinearObjective:
    """A linear objective function: Σ wᵢ · xᵢ + constant.

    Attributes:
        terms:    List of (weight, variable_name) pairs.
        constant: Constant offset.
        name:     Human-readable name for the objective.
    """

    terms: list[tuple[float, str]] = field(default_factory=list)
    constant: float = 0.0
    name: str = "objective"

    def add_term(self, weight: float, var_name: str) -> LinearObjective:
        """Add a weighted term to the objective.

        Args:
            weight:   Coefficient.
            var_name: Variable name.

        Returns:
            Self, for chaining.
        """
        self.terms.append((weight, var_name))
        return self

    def to_z3(self) -> Any:
        """Convert to a Z3 expression.

        Returns:
            Z3 arithmetic expression.
        """
        expr = z3.RealVal(str(self.constant))
        for weight, var_name in self.terms:
            expr = expr + z3.RealVal(str(weight)) * z3.Real(var_name)
        return expr

    def negate(self) -> LinearObjective:
        """Return the negation of this objective.

        Useful for converting minimisation to maximisation.

        Returns:
            Negated LinearObjective.
        """
        negated_terms = [(-w, v) for w, v in self.terms]
        return LinearObjective(
            terms=negated_terms,
            constant=-self.constant,
            name=f"-({self.name})",
        )

    def scale(self, factor: float) -> LinearObjective:
        """Scale all weights by a factor.

        Args:
            factor: Scaling factor.

        Returns:
            Scaled LinearObjective.
        """
        scaled_terms = [(w * factor, v) for w, v in self.terms]
        return LinearObjective(
            terms=scaled_terms,
            constant=self.constant * factor,
            name=self.name,
        )

    def __str__(self) -> str:
        parts = []
        for w, v in self.terms:
            if w == 1.0:
                parts.append(v)
            elif w == -1.0:
                parts.append(f"-{v}")
            else:
                parts.append(f"{w}*{v}")
        if self.constant != 0:
            parts.append(str(self.constant))
        return " + ".join(parts) if parts else "0"


# ═══════════════════════════════════════════════════════════════════════════
# OMT RESULT
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class OMTResult:
    """Result of an Optimization Modulo Theories query.

    Attributes:
        feasible:      True if a feasible solution exists.
        optimal_value: The optimal objective value (if feasible).
        model:         Z3 model at the optimum (if feasible).
        values:        Variable assignments at the optimum.
        solve_time:    Wall-clock solve time in seconds.
        status:        SAT/UNSAT/UNKNOWN.
        objective_name: Name of the objective.
    """

    feasible: bool = False
    optimal_value: float | None = None
    model: Any = None  # z3.ModelRef
    values: dict[str, float] = field(default_factory=dict)
    solve_time: float = 0.0
    status: CheckResult = CheckResult.UNKNOWN
    objective_name: str = ""

    def get_value(self, var_name: str) -> float | None:
        """Get the optimal value of a variable.

        Args:
            var_name: Variable name.

        Returns:
            Float value, or None if not available.
        """
        return self.values.get(var_name)

    def summary(self) -> str:
        """Return a human-readable summary.

        Returns:
            Summary string.
        """
        if not self.feasible:
            return f"OMT: infeasible ({self.status.value})"
        opt_str = f"{self.optimal_value:.6f}" if self.optimal_value is not None else "?"
        return (
            f"OMT: optimal={opt_str}, "
            f"vars={len(self.values)}, "
            f"time={self.solve_time:.3f}s"
        )


# ═══════════════════════════════════════════════════════════════════════════
# SOFT CONSTRAINT
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class SoftConstraint:
    """A soft constraint with a weight and optional priority group.

    Attributes:
        constraint: Z3 boolean expression.
        weight:     Cost of violating this constraint.
        group:      Priority group name (higher priority = must be
                    satisfied before lower priority).
        label:      Human-readable label.
    """

    constraint: Any  # z3.BoolRef
    weight: float = 1.0
    group: str = "default"
    label: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# OMT SOLVER
# ═══════════════════════════════════════════════════════════════════════════


class OMTSolver:
    """Wrapper around z3.Optimize for Optimization Modulo Theories.

    Provides:
      - Minimisation / maximisation of objectives
      - Soft constraints with priorities
      - Pareto-optimal solving for multi-objective
      - Model extraction
      - Linearisation of |x| for LP objectives

    Args:
        timeout_ms: Solver timeout in milliseconds.
    """

    def __init__(self, timeout_ms: int = 60000) -> None:
        self._timeout_ms = timeout_ms
        self._optimizer = z3.Optimize()
        self._optimizer.set("timeout", timeout_ms)
        self._objectives: list[tuple[str, Any, str]] = []  # (name, handle, direction)
        self._soft_groups: dict[str, list[SoftConstraint]] = {}

    def add(self, *constraints: Any) -> None:
        """Add hard constraints.

        Args:
            constraints: Z3 boolean expressions that must be satisfied.
        """
        for c in constraints:
            self._optimizer.add(c)

    def add_encoding(self, encoding: SMTEncoding) -> None:
        """Add all assertions from an SMT encoding.

        Args:
            encoding: The encoding whose assertions to add.
        """
        for assertion in encoding.assertions:
            self._optimizer.add(assertion)

    def add_soft(self, constraint: SoftConstraint) -> None:
        """Add a soft constraint.

        Args:
            constraint: The soft constraint to add.
        """
        self._optimizer.add_soft(
            constraint.constraint,
            weight=str(constraint.weight),
            id=constraint.group,
        )
        if constraint.group not in self._soft_groups:
            self._soft_groups[constraint.group] = []
        self._soft_groups[constraint.group].append(constraint)

    def add_soft_constraints(self, constraints: Sequence[SoftConstraint]) -> None:
        """Add multiple soft constraints.

        Args:
            constraints: Sequence of soft constraints.
        """
        for c in constraints:
            self.add_soft(c)

    def minimize(self, objective: Any, name: str = "min_obj") -> Any:
        """Add a minimisation objective.

        Args:
            objective: Z3 expression to minimise.
            name:      Name for the objective.

        Returns:
            Z3 optimisation handle.
        """
        handle = self._optimizer.minimize(objective)
        self._objectives.append((name, handle, "min"))
        return handle

    def maximize(self, objective: Any, name: str = "max_obj") -> Any:
        """Add a maximisation objective.

        Args:
            objective: Z3 expression to maximise.
            name:      Name for the objective.

        Returns:
            Z3 optimisation handle.
        """
        handle = self._optimizer.maximize(objective)
        self._objectives.append((name, handle, "max"))
        return handle

    def minimize_linear(self, obj: LinearObjective) -> Any:
        """Minimise a linear objective.

        Args:
            obj: Linear objective to minimise.

        Returns:
            Z3 optimisation handle.
        """
        return self.minimize(obj.to_z3(), name=obj.name)

    def maximize_linear(self, obj: LinearObjective) -> Any:
        """Maximise a linear objective.

        Args:
            obj: Linear objective to maximise.

        Returns:
            Z3 optimisation handle.
        """
        return self.maximize(obj.to_z3(), name=obj.name)

    def check(self) -> OMTResult:
        """Solve the optimisation problem.

        Returns:
            OMTResult with the outcome.
        """
        start = time.monotonic()

        try:
            z3_result = self._optimizer.check()
        except z3.Z3Exception as e:
            elapsed = time.monotonic() - start
            return OMTResult(
                feasible=False,
                solve_time=elapsed,
                status=CheckResult.ERROR,
            )

        elapsed = time.monotonic() - start

        if z3_result == z3.sat:
            model = self._optimizer.model()
            values = self._extract_model(model)

            # Extract optimal values from objectives
            optimal = None
            obj_name = ""
            for name, handle, direction in self._objectives:
                try:
                    val = handle.value()
                    fval = self._z3_to_float(val)
                    if fval is not None:
                        optimal = fval
                        obj_name = name
                except Exception:
                    pass

            return OMTResult(
                feasible=True,
                optimal_value=optimal,
                model=model,
                values=values,
                solve_time=elapsed,
                status=CheckResult.SAT,
                objective_name=obj_name,
            )

        status = CheckResult.UNSAT if z3_result == z3.unsat else CheckResult.UNKNOWN
        return OMTResult(
            feasible=False,
            solve_time=elapsed,
            status=status,
        )

    def _extract_model(self, model: Any) -> dict[str, float]:
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

    @staticmethod
    def _z3_to_float(val: Any) -> float | None:
        """Convert a Z3 value to float.

        Args:
            val: Z3 value.

        Returns:
            Float value, or None.
        """
        try:
            if z3.is_rational_value(val):
                return float(val.as_fraction())
            if z3.is_int_value(val):
                return float(val.as_long())
            if z3.is_algebraic_value(val):
                return float(val.approx(20))
            return float(str(val))
        except (ValueError, AttributeError):
            return None

    def reset(self) -> None:
        """Reset the optimiser to its initial state."""
        self._optimizer = z3.Optimize()
        self._optimizer.set("timeout", self._timeout_ms)
        self._objectives = []
        self._soft_groups = {}

    # ── Linearisation utilities ──────────────────────────────────────────

    def linearize_abs(
        self,
        x: Any,
        name_prefix: str = "abs",
    ) -> tuple[Any, Any, Any]:
        """Linearise |x| using auxiliary variables.

        Introduces x⁺ and x⁻ such that:
          x = x⁺ - x⁻
          x⁺ ≥ 0, x⁻ ≥ 0
          |x| = x⁺ + x⁻

        Args:
            x:           Z3 expression.
            name_prefix: Prefix for auxiliary variable names.

        Returns:
            Tuple of (|x| expression, x_pos variable, x_neg variable).
        """
        x_pos = z3.Real(f"__{name_prefix}_pos")
        x_neg = z3.Real(f"__{name_prefix}_neg")

        self._optimizer.add(x == x_pos - x_neg)
        self._optimizer.add(x_pos >= z3.RealVal("0"))
        self._optimizer.add(x_neg >= z3.RealVal("0"))

        return x_pos + x_neg, x_pos, x_neg

    def minimize_abs(
        self,
        x: Any,
        name: str = "abs_obj",
    ) -> Any:
        """Minimise |x| using linearisation.

        Args:
            x:    Z3 expression.
            name: Objective name.

        Returns:
            Z3 optimisation handle.
        """
        abs_x, _, _ = self.linearize_abs(x, name_prefix=name)
        return self.minimize(abs_x, name=name)

    # ── Repair cost computation ──────────────────────────────────────────

    def minimize_repair_cost(
        self,
        param_vars: dict[str, Any],
        original_values: dict[str, float],
        constraints: Sequence[Any],
    ) -> OMTResult:
        """Find the minimum-cost parameter repair.

        Minimises the sum of |new_param - original_param| subject
        to the privacy constraints.

        Args:
            param_vars:      Mapping from param name to Z3 variable.
            original_values: Original parameter values.
            constraints:     Privacy constraints that must be satisfied.

        Returns:
            OMTResult with the minimum-cost repair.
        """
        self.reset()

        # Add privacy constraints
        for c in constraints:
            self._optimizer.add(c)

        # Build repair cost: sum of |param_i - original_i|
        total_cost = z3.RealVal("0")
        for name, var in param_vars.items():
            if name in original_values:
                orig = z3.RealVal(str(original_values[name]))
                diff = var - orig
                abs_diff, _, _ = self.linearize_abs(diff, name_prefix=f"repair_{name}")
                total_cost = total_cost + abs_diff

        self.minimize(total_cost, name="repair_cost")
        return self.check()

    def find_pareto_front(
        self,
        objectives: list[tuple[str, Any]],
        constraints: Sequence[Any],
        num_points: int = 10,
    ) -> list[OMTResult]:
        """Find points on the Pareto front for multi-objective OMT.

        Uses the ε-constraint method: optimise one objective while
        constraining the others to be within varying bounds.

        Args:
            objectives:  List of (name, z3_expr) pairs to optimise.
            constraints: Hard constraints.
            num_points:  Number of Pareto-front points to compute.

        Returns:
            List of OMTResults on the Pareto front.
        """
        if len(objectives) < 2:
            self.reset()
            for c in constraints:
                self._optimizer.add(c)
            if objectives:
                self.minimize(objectives[0][1], name=objectives[0][0])
            result = self.check()
            return [result] if result.feasible else []

        # First pass: find individual optima
        bounds: list[tuple[float, float]] = []  # (min, max) for each objective
        for name, expr in objectives:
            # Minimise
            self.reset()
            for c in constraints:
                self._optimizer.add(c)
            self.minimize(expr, name=f"{name}_min")
            min_result = self.check()

            # Maximise
            self.reset()
            for c in constraints:
                self._optimizer.add(c)
            self.maximize(expr, name=f"{name}_max")
            max_result = self.check()

            lo = min_result.optimal_value if min_result.feasible else 0.0
            hi = max_result.optimal_value if max_result.feasible else 1.0
            bounds.append((lo or 0.0, hi or 1.0))

        # ε-constraint method: optimise first objective, constrain second
        pareto_results: list[OMTResult] = []
        primary_name, primary_expr = objectives[0]
        secondary_name, secondary_expr = objectives[1]
        sec_lo, sec_hi = bounds[1]

        for i in range(num_points):
            eps = sec_lo + (sec_hi - sec_lo) * i / max(num_points - 1, 1)

            self.reset()
            for c in constraints:
                self._optimizer.add(c)

            # Constrain secondary objective
            self._optimizer.add(secondary_expr <= z3.RealVal(str(eps)))

            # Optimise primary
            self.minimize(primary_expr, name=primary_name)
            result = self.check()

            if result.feasible:
                pareto_results.append(result)

        return pareto_results
