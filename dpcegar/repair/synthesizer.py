"""CEGIS-based repair synthesis for differential privacy mechanisms.

Implements Counter-Example Guided Inductive Synthesis (CEGIS) to find
a minimum-cost repair that restores privacy.  The inner loop alternates
between:
  1. **Synthesise** — find parameter values satisfying all accumulated
     counterexamples using OMT (optimisation modulo theories)
  2. **Verify** — check the candidate repair via CEGAR
  3. **Accumulate** — if verification finds a new counterexample, add
     it to the set and repeat

Classes
-------
RepairResult              – outcome of repair synthesis
RepairSynthesizer         – main CEGIS repair loop
CounterexampleAccumulator – manage the counterexample set
RepairVerifier            – verify repaired mechanisms
CostFunction              – weighted L1 cost with linearisation
RepairMinimizer           – OMT-based minimum-cost repair
RepairStatistics          – synthesis statistics
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
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
    PureBudget,
    ApproxBudget,
    TypedExpr,
    Var,
)
from dpcegar.ir.nodes import MechIR
from dpcegar.paths.symbolic_path import PathSet, SymbolicPath
from dpcegar.density.ratio_builder import DensityRatioExpr
from dpcegar.cegar.engine import (
    CEGARConfig,
    CEGAREngine,
    CEGARResult,
    CEGARVerdict,
    SMTSolverInterface,
    SMTResult,
    SMTStatus,
)
from dpcegar.cegar.refinement import RefinementCounterexample
from dpcegar.repair.templates import (
    RepairParameter,
    RepairSite,
    RepairTemplate,
    CompositeRepair,
    ScaleParam,
    TemplateCost,
    TemplateEnumerator,
    TemplateValidator,
)
from dpcegar.utils.errors import (
    DPCegarError,
    InternalError,
    RepairError,
    NoRepairFoundError,
    ensure,
)


# ═══════════════════════════════════════════════════════════════════════════
# REPAIR RESULT
# ═══════════════════════════════════════════════════════════════════════════


class RepairVerdict(Enum):
    """Outcome of repair synthesis."""

    SUCCESS = auto()
    NO_REPAIR = auto()
    TIMEOUT = auto()
    ERROR = auto()


@dataclass(slots=True)
class RepairResult:
    """Result of the repair synthesis process.

    Attributes:
        verdict: Whether repair succeeded.
        template: The repair template used (if successful).
        parameter_values: Concrete parameter values (if successful).
        repaired_mechanism: The repaired MechIR (if successful).
        repair_cost: Cost of the repair.
        verification_certificate: CEGAR certificate for the repair.
        statistics: Synthesis statistics.
        details: Additional information.
    """

    verdict: RepairVerdict = RepairVerdict.NO_REPAIR
    template: RepairTemplate | None = None
    parameter_values: dict[str, float] = field(default_factory=dict)
    repaired_mechanism: MechIR | None = None
    repair_cost: float = float("inf")
    verification_certificate: CEGARResult | None = None
    statistics: RepairStatistics | None = None
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Return True if repair was successful."""
        return self.verdict == RepairVerdict.SUCCESS

    def summary(self) -> str:
        """Return a human-readable summary."""
        parts = [f"Repair: {self.verdict.name}"]
        if self.template:
            parts.append(f"template={self.template.name()}")
        if self.repair_cost < float("inf"):
            parts.append(f"cost={self.repair_cost:.4f}")
        if self.statistics:
            parts.append(f"iters={self.statistics.cegis_iterations}")
        return ", ".join(parts)

    def __str__(self) -> str:
        return self.summary()


# ═══════════════════════════════════════════════════════════════════════════
# REPAIR STATISTICS
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class RepairStatistics:
    """Statistics for the repair synthesis process.

    Attributes:
        cegis_iterations: Number of CEGIS iterations.
        templates_tried: Number of templates attempted.
        counterexamples_accumulated: Total counterexamples collected.
        solver_calls: Number of OMT/SMT solver calls.
        verification_calls: Number of CEGAR verification calls.
        total_time: Total wall-clock time.
        synthesis_time: Time spent in synthesis (OMT).
        verification_time: Time spent in verification (CEGAR).
        best_cost: Best repair cost found.
    """

    cegis_iterations: int = 0
    templates_tried: int = 0
    counterexamples_accumulated: int = 0
    solver_calls: int = 0
    verification_calls: int = 0
    total_time: float = 0.0
    synthesis_time: float = 0.0
    verification_time: float = 0.0
    best_cost: float = float("inf")

    def summary(self) -> dict[str, Any]:
        """Return all statistics as a dictionary.

        Returns:
            Statistics dictionary.
        """
        return {
            "cegis_iterations": self.cegis_iterations,
            "templates_tried": self.templates_tried,
            "counterexamples": self.counterexamples_accumulated,
            "solver_calls": self.solver_calls,
            "verification_calls": self.verification_calls,
            "total_time_s": round(self.total_time, 3),
            "synthesis_time_s": round(self.synthesis_time, 3),
            "verification_time_s": round(self.verification_time, 3),
            "best_cost": self.best_cost,
        }

    def __str__(self) -> str:
        return (
            f"RepairStats(iters={self.cegis_iterations}, "
            f"cex={self.counterexamples_accumulated}, "
            f"time={self.total_time:.2f}s)"
        )


# ═══════════════════════════════════════════════════════════════════════════
# COUNTEREXAMPLE ACCUMULATOR
# ═══════════════════════════════════════════════════════════════════════════


class CounterexampleAccumulator:
    """Collect and manage the counterexample set for CEGIS.

    Maintains a set of counterexamples (concrete variable assignments)
    that any valid repair must satisfy.  Provides deduplication and
    subsumption checking.
    """

    def __init__(self, max_size: int = 1000) -> None:
        """Initialise with a maximum counterexample set size.

        Args:
            max_size: Maximum number of counterexamples to retain.
        """
        self._counterexamples: list[RefinementCounterexample] = []
        self._max_size = max_size
        self._fingerprints: set[str] = set()

    def add(self, cex: RefinementCounterexample) -> bool:
        """Add a counterexample to the accumulator.

        Deduplicates by checking variable assignment fingerprints.

        Args:
            cex: The counterexample to add.

        Returns:
            True if the counterexample was new and added.
        """
        fp = self._fingerprint(cex)
        if fp in self._fingerprints:
            return False

        if len(self._counterexamples) >= self._max_size:
            self._evict_oldest()

        self._counterexamples.append(cex)
        self._fingerprints.add(fp)
        return True

    def add_from_model(
        self,
        model: dict[str, float],
        path_id: int = -1,
        density_value: float = 0.0,
    ) -> bool:
        """Add a counterexample from an SMT model.

        Args:
            model: Variable assignment from the solver.
            path_id: Associated path ID.
            density_value: Privacy loss at this point.

        Returns:
            True if new.
        """
        cex = RefinementCounterexample(
            variable_assignment=dict(model),
            path_id=path_id,
            density_ratio_value=density_value,
            is_spurious=False,
        )
        return self.add(cex)

    def all(self) -> list[RefinementCounterexample]:
        """Return all accumulated counterexamples.

        Returns:
            List of counterexamples.
        """
        return list(self._counterexamples)

    def size(self) -> int:
        """Return the number of accumulated counterexamples.

        Returns:
            Count.
        """
        return len(self._counterexamples)

    def is_empty(self) -> bool:
        """Return True if no counterexamples have been accumulated.

        Returns:
            True if empty.
        """
        return len(self._counterexamples) == 0

    def clear(self) -> None:
        """Clear all accumulated counterexamples."""
        self._counterexamples.clear()
        self._fingerprints.clear()

    def constraints_for_template(
        self,
        template: RepairTemplate,
        budget: PrivacyBudget,
    ) -> list[TypedExpr]:
        """Generate SMT constraints from accumulated counterexamples.

        For each counterexample, generates the constraint that the
        repaired mechanism's privacy loss (with template parameters)
        stays within the budget at that point.

        Args:
            template: The repair template (provides parameter variables).
            budget: The privacy budget.

        Returns:
            List of constraint expressions.
        """
        eps, _ = budget.to_approx_dp()
        constraints: list[TypedExpr] = []

        for param in template.parameters():
            constraints.extend(param.domain_constraints())

        for cex in self._counterexamples:
            cex_constraint = self._cex_to_constraint(cex, template, eps)
            if cex_constraint is not None:
                constraints.append(cex_constraint)

        return constraints

    def _cex_to_constraint(
        self,
        cex: RefinementCounterexample,
        template: RepairTemplate,
        epsilon: float,
    ) -> TypedExpr | None:
        """Convert a counterexample to a privacy constraint.

        Uses ``template.symbolic_loss()`` when available to produce a
        parametric constraint ``loss(params) ≤ ε``.  Falls back to the
        constant-based approach only when ``symbolic_loss()`` returns
        ``None``.

        Args:
            cex: RefinementCounterexample.
            template: Repair template.
            epsilon: Privacy parameter.

        Returns:
            Constraint expression, or None if not applicable.
        """
        if not cex.variable_assignment:
            return None

        sensitivity = abs(cex.density_ratio_value)

        # Prefer symbolic_loss() for a parametric constraint
        loss_expr = template.symbolic_loss(sensitivity)
        if loss_expr is not None:
            return BinOp(
                ty=IRType.BOOL,
                op=BinOpKind.LE,
                left=loss_expr,
                right=Const.real(epsilon),
            )

        # Fallback: constant-only constraint (no template params)
        return BinOp(
            ty=IRType.BOOL,
            op=BinOpKind.LE,
            left=Const.real(sensitivity),
            right=Const.real(epsilon),
        )

    def _fingerprint(self, cex: RefinementCounterexample) -> str:
        """Compute a fingerprint for deduplication.

        Args:
            cex: RefinementCounterexample to fingerprint.

        Returns:
            String fingerprint.
        """
        items = sorted(cex.variable_assignment.items())
        parts = [f"{k}={v:.6g}" for k, v in items]
        return f"p{cex.path_id}|" + "|".join(parts)

    def _evict_oldest(self) -> None:
        """Remove the oldest counterexample to make room."""
        if self._counterexamples:
            old = self._counterexamples.pop(0)
            fp = self._fingerprint(old)
            self._fingerprints.discard(fp)


# ═══════════════════════════════════════════════════════════════════════════
# REPAIR VERIFIER
# ═══════════════════════════════════════════════════════════════════════════


class RepairVerifier:
    """Verify repaired mechanisms via full CEGAR.

    After synthesising candidate parameter values, this verifies
    the repaired mechanism using the CEGAR engine.
    """

    def __init__(
        self,
        cegar_config: CEGARConfig | None = None,
        solver: SMTSolverInterface | None = None,
    ) -> None:
        """Initialise with CEGAR configuration.

        Args:
            cegar_config: Configuration for verification runs.
            solver: Optional SMT solver.
        """
        self._config = cegar_config or CEGARConfig(
            max_refinements=50,
            timeout_seconds=60.0,
        )
        self._solver = solver

    def verify(
        self,
        repaired_mechanism: MechIR,
        budget: PrivacyBudget,
        path_set: PathSet | None = None,
        density_ratios: list[DensityRatioExpr] | None = None,
    ) -> CEGARResult:
        """Verify the repaired mechanism.

        Args:
            repaired_mechanism: The mechanism after repair.
            budget: The privacy budget to verify.
            path_set: Optional paths (will be re-enumerated otherwise).
            density_ratios: Optional density ratios.

        Returns:
            CEGARResult from the verification.
        """
        engine = CEGAREngine(
            config=self._config,
            solver=self._solver,
        )

        if path_set is not None:
            return engine.verify(path_set, budget, density_ratios)
        else:
            return engine.verify_mechanism(repaired_mechanism, budget)

    def quick_check(
        self,
        repaired_mechanism: MechIR,
        budget: PrivacyBudget,
    ) -> bool:
        """Quick syntactic check if repair might satisfy budget.

        Performs lightweight checks without full CEGAR.

        Args:
            repaired_mechanism: Repaired mechanism.
            budget: Privacy budget.

        Returns:
            True if the quick check passes.
        """
        eps, _ = budget.to_approx_dp()

        for node in repaired_mechanism.all_nodes():
            from dpcegar.ir.nodes import NoiseDrawNode
            if isinstance(node, NoiseDrawNode):
                if isinstance(node.scale, Const):
                    scale = float(node.scale.value)
                    if scale <= 0:
                        return False
                if node.sensitivity is not None and isinstance(node.sensitivity, Const):
                    sens = float(node.sensitivity.value)
                    if isinstance(node.scale, Const):
                        scale = float(node.scale.value)
                        if scale > 0 and sens / scale > eps:
                            return False

        return True


# ═══════════════════════════════════════════════════════════════════════════
# COST FUNCTION
# ═══════════════════════════════════════════════════════════════════════════


class CostFunction:
    """Weighted L1 cost function for repair optimisation.

    The cost is a weighted sum of absolute deviations of repair
    parameters from their original values:

        cost = Σ_i  w_i · |p_i - p_i_orig|

    For OMT, absolute values are linearised using auxiliary variables:
        |x| = max(x, -x)  encoded as  t ≥ x ∧ t ≥ -x ∧ minimize t
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        default_weight: float = 1.0,
    ) -> None:
        """Initialise with per-parameter weights.

        Args:
            weights: Mapping from parameter name to weight.
            default_weight: Weight for parameters not in the mapping.
        """
        self._weights = weights or {}
        self._default_weight = default_weight

    def compute(
        self,
        template: RepairTemplate,
        values: dict[str, float],
        original_values: dict[str, float],
    ) -> float:
        """Compute the concrete cost for given parameter values.

        Args:
            template: The repair template.
            values: Repair parameter values.
            original_values: Original parameter values.

        Returns:
            The total cost.
        """
        cost = 0.0
        for param in template.parameters():
            w = self._weights.get(param.name, self._default_weight)
            v = values.get(param.name, param.initial_value)
            orig = original_values.get(param.name, param.initial_value)
            cost += w * abs(v - orig)
        return cost

    def symbolic_cost(
        self,
        template: RepairTemplate,
        original_values: dict[str, float],
    ) -> TypedExpr:
        """Build the symbolic cost expression for OMT.

        Args:
            template: The repair template.
            original_values: Original parameter values.

        Returns:
            Symbolic cost expression.
        """
        return template.cost_expression(original_values)

    def linearize_absolute_values(
        self,
        template: RepairTemplate,
        original_values: dict[str, float],
    ) -> tuple[list[TypedExpr], TypedExpr, list[RepairParameter]]:
        """Linearise |p - p_orig| for OMT using auxiliary variables.

        Introduces auxiliary variables t_i ≥ 0 and constraints:
            t_i ≥ p_i - p_i_orig
            t_i ≥ -(p_i - p_i_orig)

        The objective is: minimize Σ w_i · t_i.

        Args:
            template: The repair template.
            original_values: Original parameter values.

        Returns:
            Tuple of (constraints, objective, auxiliary_params).
        """
        constraints: list[TypedExpr] = []
        aux_params: list[RepairParameter] = []
        objective_terms: list[TypedExpr] = []

        for param in template.parameters():
            w = self._weights.get(param.name, self._default_weight)
            orig = original_values.get(param.name, param.initial_value)

            aux_name = f"_abs_{param.name}"
            aux = RepairParameter(
                name=aux_name,
                param_type=IRType.REAL,
                lower_bound=0.0,
                upper_bound=1e6,
            )
            aux_params.append(aux)
            aux_var = aux.as_var()

            diff = BinOp(
                ty=IRType.REAL, op=BinOpKind.SUB,
                left=param.as_var(),
                right=Const.real(orig),
            )

            # t >= p - orig
            constraints.append(BinOp(
                ty=IRType.BOOL, op=BinOpKind.GE,
                left=aux_var, right=diff,
            ))

            # t >= -(p - orig) = orig - p
            neg_diff = BinOp(
                ty=IRType.REAL, op=BinOpKind.SUB,
                left=Const.real(orig),
                right=param.as_var(),
            )
            constraints.append(BinOp(
                ty=IRType.BOOL, op=BinOpKind.GE,
                left=aux_var, right=neg_diff,
            ))

            weighted = BinOp(
                ty=IRType.REAL, op=BinOpKind.MUL,
                left=Const.real(w),
                right=aux_var,
            )
            objective_terms.append(weighted)

        if not objective_terms:
            return constraints, Const.real(0.0), aux_params

        objective = objective_terms[0]
        for term in objective_terms[1:]:
            objective = BinOp(
                ty=IRType.REAL, op=BinOpKind.ADD,
                left=objective, right=term,
            )

        return constraints, objective, aux_params


# ═══════════════════════════════════════════════════════════════════════════
# REPAIR MINIMIZER
# ═══════════════════════════════════════════════════════════════════════════


class RepairMinimizer:
    """Find minimum-cost repair via OMT (Optimisation Modulo Theories).

    Uses the SMT solver's optimisation capabilities to find the
    parameter values that minimise the repair cost while satisfying
    all privacy constraints from accumulated counterexamples.
    """

    def __init__(
        self,
        solver: SMTSolverInterface | None = None,
        cost_function: CostFunction | None = None,
    ) -> None:
        """Initialise with solver and cost function.

        Args:
            solver: SMT solver with OMT support.
            cost_function: Cost function for the objective.
        """
        self._solver = solver
        self._cost_fn = cost_function or CostFunction()

    def minimize(
        self,
        template: RepairTemplate,
        constraints: list[TypedExpr],
        original_values: dict[str, float],
        timeout_ms: int = 30000,
    ) -> MinimizationResult:
        """Find minimum-cost parameter values.

        Constructs the OMT problem:
          minimise  cost(params)
          subject to  privacy_constraints ∧ domain_constraints

        Args:
            template: The repair template.
            constraints: Privacy constraints from counterexamples.
            original_values: Original parameter values.
            timeout_ms: Solver timeout.

        Returns:
            MinimizationResult with optimal values.
        """
        lin_constraints, objective, aux_params = \
            self._cost_fn.linearize_absolute_values(template, original_values)

        all_constraints = list(constraints) + lin_constraints
        for param in template.parameters():
            all_constraints.extend(param.domain_constraints())
        for aux in aux_params:
            all_constraints.extend(aux.domain_constraints())

        if self._solver is not None:
            return self._solver_minimize(
                template, all_constraints, objective, timeout_ms
            )

        return self._fallback_minimize(template, original_values)

    def _solver_minimize(
        self,
        template: RepairTemplate,
        constraints: list[TypedExpr],
        objective: TypedExpr,
        timeout_ms: int,
    ) -> MinimizationResult:
        """Use the SMT solver's OMT to minimise cost.

        Args:
            template: Repair template.
            constraints: All constraints.
            objective: Objective to minimise.
            timeout_ms: Timeout.

        Returns:
            MinimizationResult.
        """
        assert self._solver is not None

        status, opt_value = self._solver.maximize(
            BinOp(
                ty=IRType.REAL, op=BinOpKind.MUL,
                left=Const.real(-1.0),
                right=objective,
            ),
            constraints,
            timeout_ms,
        )

        if status == "optimal":
            model = self._solver.get_model()
            values = {
                p.name: model.get(p.name, p.initial_value)
                for p in template.parameters()
            }
            cost = -opt_value if opt_value is not None else float("inf")
            return MinimizationResult(
                success=True,
                parameter_values=values,
                cost=cost,
            )
        elif status == "unsat":
            return MinimizationResult(
                success=False,
                reason="infeasible",
            )
        else:
            return MinimizationResult(
                success=False,
                reason=f"solver_status_{status}",
            )

    def _fallback_minimize(
        self,
        template: RepairTemplate,
        original_values: dict[str, float],
    ) -> MinimizationResult:
        """Fallback: return initial values as the candidate.

        When no solver is available, use the initial parameter values
        as a best guess.

        Args:
            template: Repair template.
            original_values: Original values.

        Returns:
            MinimizationResult with initial values.
        """
        values = {
            p.name: p.initial_value for p in template.parameters()
        }
        cost = self._cost_fn.compute(template, values, original_values)
        return MinimizationResult(
            success=True,
            parameter_values=values,
            cost=cost,
            details={"method": "fallback"},
        )


@dataclass(slots=True)
class MinimizationResult:
    """Result of the OMT minimization.

    Attributes:
        success: Whether minimization succeeded.
        parameter_values: Optimal parameter values.
        cost: Optimal cost.
        reason: Reason for failure (if unsuccessful).
        details: Additional information.
    """

    success: bool = False
    parameter_values: dict[str, float] = field(default_factory=dict)
    cost: float = float("inf")
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        if self.success:
            return f"Optimal(cost={self.cost:.4f})"
        return f"Failed({self.reason})"


# ═══════════════════════════════════════════════════════════════════════════
# REPAIR SYNTHESIZER — Main CEGIS loop
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class SynthesizerConfig:
    """Configuration for the repair synthesizer.

    Attributes:
        max_cegis_iterations: Maximum CEGIS loop iterations.
        max_templates: Maximum templates to try.
        timeout_seconds: Overall synthesis timeout.
        solver_timeout_ms: Per-query solver timeout.
        cegar_config: Config for verification sub-calls.
        enable_composite_repairs: Whether to try composite repairs.
        max_composite_components: Maximum components in a composite.
    """

    max_cegis_iterations: int = 50
    max_templates: int = 20
    timeout_seconds: float = 300.0
    solver_timeout_ms: int = 30000
    cegar_config: CEGARConfig = field(default_factory=lambda: CEGARConfig(
        max_refinements=50, timeout_seconds=60.0
    ))
    enable_composite_repairs: bool = True
    max_composite_components: int = 3


class RepairSynthesizer:
    """CEGIS-based repair synthesizer for differential privacy mechanisms.

    Searches for a minimum-cost repair that restores privacy by
    iterating between synthesis and verification.

    Usage::

        synth = RepairSynthesizer(config=SynthesizerConfig())
        result = synth.synthesize(mechanism, budget)
        if result.success:
            print(f"Repair found: cost={result.repair_cost}")
            repaired = result.repaired_mechanism
    """

    def __init__(
        self,
        config: SynthesizerConfig | None = None,
        solver: SMTSolverInterface | None = None,
        templates: list[RepairTemplate] | None = None,
    ) -> None:
        """Initialise the synthesizer.

        Args:
            config: Synthesis configuration.
            solver: Optional SMT solver.
            templates: Optional pre-selected templates.
        """
        self._config = config or SynthesizerConfig()
        self._solver = solver
        self._fixed_templates = templates
        self._enumerator = TemplateEnumerator()
        self._cost_fn = CostFunction()
        self._minimizer = RepairMinimizer(solver, self._cost_fn)
        self._verifier = RepairVerifier(
            cegar_config=self._config.cegar_config,
            solver=solver,
        )
        self._validator = TemplateValidator()
        self._template_cost = TemplateCost()
        self._stats = RepairStatistics()

    def synthesize(
        self,
        mechanism: MechIR,
        budget: PrivacyBudget,
        initial_counterexample: RefinementCounterexample | None = None,
        path_set: PathSet | None = None,
        density_ratios: list[DensityRatioExpr] | None = None,
    ) -> RepairResult:
        """Synthesise a minimum-cost repair for the mechanism.

        This is the main entry point.  It:
          1. Enumerates applicable templates
          2. Ranks them by cost
          3. For each template, runs the CEGIS loop
          4. Returns the best repair found

        Args:
            mechanism: The mechanism to repair.
            budget: The privacy budget to satisfy.
            initial_counterexample: Optional seed counterexample.
            path_set: Optional pre-enumerated paths.
            density_ratios: Optional density ratio expressions.

        Returns:
            RepairResult with the best repair found.
        """
        start_time = time.monotonic()
        self._stats = RepairStatistics()

        templates = self._get_templates(mechanism)
        if not templates:
            self._stats.total_time = time.monotonic() - start_time
            return RepairResult(
                verdict=RepairVerdict.NO_REPAIR,
                statistics=self._stats,
                details={"reason": "no applicable templates"},
            )

        ranked = self._template_cost.rank(templates)
        max_templates = min(len(ranked), self._config.max_templates)

        best_result: RepairResult | None = None
        best_cost = float("inf")

        for idx, template in enumerate(ranked[:max_templates]):
            if time.monotonic() - start_time > self._config.timeout_seconds:
                break

            self._stats.templates_tried += 1
            validation = self._validator.validate(template, mechanism)
            if not validation.is_valid:
                continue

            result = self._cegis_loop(
                mechanism, budget, template,
                initial_counterexample, path_set, density_ratios,
            )

            if result.success and result.repair_cost < best_cost:
                best_cost = result.repair_cost
                best_result = result

            if best_result is not None and best_cost <= 0.0:
                break

        self._stats.total_time = time.monotonic() - start_time
        self._stats.best_cost = best_cost

        if best_result is not None:
            best_result.statistics = self._stats
            return best_result

        return RepairResult(
            verdict=RepairVerdict.NO_REPAIR,
            statistics=self._stats,
            details={"reason": "no template produced a valid repair"},
        )

    def synthesize_with_template(
        self,
        mechanism: MechIR,
        budget: PrivacyBudget,
        template: RepairTemplate,
        initial_counterexample: RefinementCounterexample | None = None,
        path_set: PathSet | None = None,
        density_ratios: list[DensityRatioExpr] | None = None,
    ) -> RepairResult:
        """Synthesise a repair using a specific template.

        Args:
            mechanism: The mechanism to repair.
            budget: The privacy budget.
            template: The repair template to use.
            initial_counterexample: Optional seed counterexample.
            path_set: Optional paths.
            density_ratios: Optional density ratios.

        Returns:
            RepairResult.
        """
        start_time = time.monotonic()
        self._stats = RepairStatistics()
        self._stats.templates_tried = 1

        result = self._cegis_loop(
            mechanism, budget, template,
            initial_counterexample, path_set, density_ratios,
        )

        self._stats.total_time = time.monotonic() - start_time
        result.statistics = self._stats
        return result

    def _get_templates(self, mechanism: MechIR) -> list[RepairTemplate]:
        """Get applicable repair templates.

        Args:
            mechanism: The mechanism.

        Returns:
            List of applicable templates.
        """
        if self._fixed_templates is not None:
            return self._fixed_templates

        if self._config.enable_composite_repairs:
            return self._enumerator.enumerate_composites(
                mechanism, self._config.max_composite_components
            )
        return self._enumerator.enumerate(mechanism)

    def _cegis_loop(
        self,
        mechanism: MechIR,
        budget: PrivacyBudget,
        template: RepairTemplate,
        initial_cex: RefinementCounterexample | None,
        path_set: PathSet | None,
        density_ratios: list[DensityRatioExpr] | None,
    ) -> RepairResult:
        """Execute the inner CEGIS loop for a single template.

        The loop:
          1. Synthesise candidate parameter values via OMT
          2. Apply template to produce repaired mechanism
          3. Verify repaired mechanism via CEGAR
          4. If counterexample found, accumulate and repeat

        Args:
            mechanism: Original mechanism.
            budget: Privacy budget.
            template: Repair template.
            initial_cex: Optional initial counterexample.
            path_set: Optional paths.
            density_ratios: Optional density ratios.

        Returns:
            RepairResult for this template.
        """
        accumulator = CounterexampleAccumulator()
        if initial_cex is not None:
            accumulator.add(initial_cex)

        original_values = self._extract_original_values(mechanism, template)
        start_time = time.monotonic()

        for iteration in range(self._config.max_cegis_iterations):
            self._stats.cegis_iterations += 1

            if time.monotonic() - start_time > self._config.timeout_seconds / 2:
                return RepairResult(
                    verdict=RepairVerdict.TIMEOUT,
                    template=template,
                    details={"reason": "cegis_timeout", "iteration": iteration},
                )

            constraints = accumulator.constraints_for_template(template, budget)

            t_synth = time.monotonic()
            min_result = self._minimizer.minimize(
                template, constraints, original_values,
                timeout_ms=self._config.solver_timeout_ms,
            )
            self._stats.synthesis_time += time.monotonic() - t_synth
            self._stats.solver_calls += 1

            if not min_result.success:
                return RepairResult(
                    verdict=RepairVerdict.NO_REPAIR,
                    template=template,
                    details={
                        "reason": f"synthesis_failed: {min_result.reason}",
                        "iteration": iteration,
                    },
                )

            repaired = template.apply_concrete(mechanism, min_result.parameter_values)

            if not self._verifier.quick_check(repaired, budget):
                dummy_cex = RefinementCounterexample(
                    variable_assignment=min_result.parameter_values,
                    density_ratio_value=float("inf"),
                )
                accumulator.add(dummy_cex)
                self._stats.counterexamples_accumulated += 1
                continue

            t_verify = time.monotonic()
            cegar_result = self._verifier.verify(
                repaired, budget, path_set, density_ratios
            )
            self._stats.verification_time += time.monotonic() - t_verify
            self._stats.verification_calls += 1

            if cegar_result.is_verified:
                cost = self._cost_fn.compute(
                    template, min_result.parameter_values, original_values
                )
                return RepairResult(
                    verdict=RepairVerdict.SUCCESS,
                    template=template,
                    parameter_values=min_result.parameter_values,
                    repaired_mechanism=repaired,
                    repair_cost=cost,
                    verification_certificate=cegar_result,
                )

            if cegar_result.counterexample is not None:
                new = accumulator.add(cegar_result.counterexample)
                if new:
                    self._stats.counterexamples_accumulated += 1
                else:
                    return RepairResult(
                        verdict=RepairVerdict.NO_REPAIR,
                        template=template,
                        details={
                            "reason": "duplicate_counterexample",
                            "iteration": iteration,
                        },
                    )
            else:
                return RepairResult(
                    verdict=RepairVerdict.NO_REPAIR,
                    template=template,
                    details={
                        "reason": "verification_inconclusive",
                        "verdict": cegar_result.verdict.name,
                    },
                )

        return RepairResult(
            verdict=RepairVerdict.TIMEOUT,
            template=template,
            details={
                "reason": "max_iterations",
                "iterations": self._config.max_cegis_iterations,
                "counterexamples": accumulator.size(),
            },
        )

    def _extract_original_values(
        self,
        mechanism: MechIR,
        template: RepairTemplate,
    ) -> dict[str, float]:
        """Extract original parameter values from the mechanism.

        Args:
            mechanism: The original mechanism.
            template: The repair template.

        Returns:
            Mapping from parameter name to original value.
        """
        values: dict[str, float] = {}
        for param in template.parameters():
            values[param.name] = param.initial_value

        for site in template.sites():
            node = mechanism.find_node(site.node_id)
            if node is None:
                continue

            from dpcegar.ir.nodes import NoiseDrawNode, QueryNode, BranchNode
            if isinstance(node, NoiseDrawNode):
                if isinstance(node.scale, Const):
                    for param in template.parameters():
                        if f"scale_{site.node_id}" == param.name:
                            values[param.name] = float(node.scale.value)
            elif isinstance(node, QueryNode):
                if isinstance(node.sensitivity, Const):
                    for param in template.parameters():
                        if f"sensitivity_{site.node_id}" == param.name:
                            values[param.name] = float(node.sensitivity.value)
            elif isinstance(node, BranchNode):
                if isinstance(node.condition, BinOp) and isinstance(node.condition.right, Const):
                    for param in template.parameters():
                        if f"threshold_{site.node_id}" == param.name:
                            values[param.name] = float(node.condition.right.value)

        return values

    def get_statistics(self) -> RepairStatistics:
        """Return the current synthesis statistics.

        Returns:
            The RepairStatistics object.
        """
        return self._stats
