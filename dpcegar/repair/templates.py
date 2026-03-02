"""Repair template grammar for differential privacy mechanism repair.

Defines the space of possible repairs as a template grammar.  Each
template describes a parameterised modification to the mechanism that
may restore privacy.  The synthesiser searches over instantiations of
these templates to find a minimum-cost repair.

Classes
-------
RepairTemplate          – abstract base for all repair templates
ScaleParam              – modify noise scale parameter at a site
ThresholdShift          – modify conditional threshold
ClampBound              – add output clamping [lo, hi]
CompositionBudgetSplit  – adjust per-iteration privacy budget
SensitivityRescale      – adjust sensitivity parameter
NoiseSwap               – change noise distribution family
CompositeRepair         – multiple repairs applied together
TemplateEnumerator      – enumerate applicable templates
TemplateCost            – compute repair cost
TemplateValidator       – check applicability constraints
"""

from __future__ import annotations

import copy
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
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
    PureBudget,
    ApproxBudget,
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
from dpcegar.paths.symbolic_path import PathSet, SymbolicPath, NoiseDrawInfo
from dpcegar.utils.errors import InternalError, RepairError, ensure


# ═══════════════════════════════════════════════════════════════════════════
# REPAIR SITE
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class RepairSite:
    """A location in the mechanism IR where a repair can be applied.

    Attributes:
        node_id: IR node ID targeted by the repair.
        node_type: Type name of the IR node.
        description: Human-readable description of the site.
        current_value: Current parameter value at the site (if applicable).
    """

    node_id: int
    node_type: str
    description: str = ""
    current_value: float | None = None

    def __str__(self) -> str:
        return f"Site(node={self.node_id}, type={self.node_type})"


# ═══════════════════════════════════════════════════════════════════════════
# REPAIR PARAMETER
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class RepairParameter:
    """A synthesis parameter in a repair template.

    Represents an unknown value that the CEGIS synthesiser will solve
    for.  Each parameter has a name, type, domain, and an optional
    initial guess.

    Attributes:
        name: Parameter name (used as SMT variable name).
        param_type: Type of the parameter.
        lower_bound: Lower bound of the search domain.
        upper_bound: Upper bound of the search domain.
        initial_value: Starting value for the search.
        is_integer: Whether the parameter must be integer-valued.
    """

    name: str = ""
    param_type: IRType = IRType.REAL
    lower_bound: float = -1e6
    upper_bound: float = 1e6
    initial_value: float = 0.0
    is_integer: bool = False

    def as_var(self) -> Var:
        """Create an IR variable for this parameter.

        Returns:
            A Var node referencing this parameter.
        """
        return Var(ty=self.param_type, name=self.name)

    def as_const(self, value: float) -> Const:
        """Create a constant from a concrete parameter value.

        Args:
            value: The concrete value.

        Returns:
            A Const node.
        """
        if self.is_integer:
            return Const.int_(int(value))
        return Const.real(value)

    def domain_constraints(self) -> list[TypedExpr]:
        """Return SMT constraints bounding this parameter.

        Returns:
            List of constraint expressions.
        """
        v = self.as_var()
        constraints: list[TypedExpr] = []
        if not math.isinf(self.lower_bound):
            constraints.append(BinOp(
                ty=IRType.BOOL, op=BinOpKind.GE,
                left=v, right=Const.real(self.lower_bound),
            ))
        if not math.isinf(self.upper_bound):
            constraints.append(BinOp(
                ty=IRType.BOOL, op=BinOpKind.LE,
                left=v, right=Const.real(self.upper_bound),
            ))
        return constraints

    def __str__(self) -> str:
        return f"{self.name} ∈ [{self.lower_bound}, {self.upper_bound}]"


# ═══════════════════════════════════════════════════════════════════════════
# REPAIR TEMPLATE BASE CLASS
# ═══════════════════════════════════════════════════════════════════════════


class RepairTemplate(ABC):
    """Abstract base class for repair templates.

    Each template describes a parameterised modification to the
    mechanism.  The synthesiser instantiates the parameters to
    find a repair that satisfies the privacy budget.
    """

    @abstractmethod
    def name(self) -> str:
        """Return a human-readable name for this template."""
        ...

    @abstractmethod
    def parameters(self) -> list[RepairParameter]:
        """Return the synthesis parameters for this template."""
        ...

    @abstractmethod
    def sites(self) -> list[RepairSite]:
        """Return the IR sites targeted by this template."""
        ...

    @abstractmethod
    def apply_symbolic(self, mechanism: MechIR) -> MechIR:
        """Apply the template symbolically (with parameter variables).

        Produces a modified MechIR where template parameters appear
        as free variables.  The synthesiser solves for their values.

        Args:
            mechanism: The original mechanism IR.

        Returns:
            Modified MechIR with symbolic parameters.
        """
        ...

    @abstractmethod
    def apply_concrete(self, mechanism: MechIR, values: dict[str, float]) -> MechIR:
        """Apply the template with concrete parameter values.

        Args:
            mechanism: The original mechanism IR.
            values: Mapping from parameter name to concrete value.

        Returns:
            The repaired mechanism IR.
        """
        ...

    @abstractmethod
    def cost_expression(self, original_values: dict[str, float]) -> TypedExpr:
        """Return the cost of this repair as an expression over parameters.

        The cost measures how much the repair deviates from the original
        mechanism.  The synthesiser minimises this.

        Args:
            original_values: Original parameter values for comparison.

        Returns:
            Symbolic cost expression.
        """
        ...

    def is_applicable(self, mechanism: MechIR) -> bool:
        """Check if this template can be applied to the mechanism.

        Args:
            mechanism: The mechanism to check.

        Returns:
            True if the template is applicable.
        """
        return len(self.sites()) > 0

    def symbolic_loss(self, sensitivity: float = 1.0) -> Optional[TypedExpr]:
        """Return the symbolic privacy loss expression for this template.

        The expression is parameterised by the template's synthesis
        variables.  For a Laplace mechanism with scale *s* and query
        sensitivity Δ the loss is Δ/s.

        Args:
            sensitivity: The query sensitivity (Δ) at the repair site.

        Returns:
            A ``TypedExpr`` whose free variables are the template
            parameters, or ``None`` if symbolic loss is not meaningful
            for this template (e.g. discrete choices).
        """
        return None

    def description(self) -> str:
        """Return a human-readable description of the repair.

        Returns:
            Description string.
        """
        return f"{self.name()}: {len(self.parameters())} params, {len(self.sites())} sites"


# ═══════════════════════════════════════════════════════════════════════════
# SCALE PARAMETER REPAIR
# ═══════════════════════════════════════════════════════════════════════════


class ScaleParam(RepairTemplate):
    """Modify noise scale parameter at a noise draw site.

    Changes the scale (e.g. Laplace b or Gaussian σ) of a noise draw
    to increase the noise and restore privacy.
    """

    def __init__(
        self,
        site: RepairSite,
        original_scale: float,
        min_scale: float = 1e-6,
        max_scale: float = 1e6,
    ) -> None:
        """Initialise with the target site and bounds.

        Args:
            site: The noise draw site to modify.
            original_scale: The original scale value.
            min_scale: Minimum allowed scale.
            max_scale: Maximum allowed scale.
        """
        self._site = site
        self._original = original_scale
        self._param = RepairParameter(
            name=f"scale_{site.node_id}",
            param_type=IRType.REAL,
            lower_bound=min_scale,
            upper_bound=max_scale,
            initial_value=original_scale,
        )

    def name(self) -> str:
        """Return the template name."""
        return f"ScaleParam(node={self._site.node_id})"

    def parameters(self) -> list[RepairParameter]:
        """Return the scale parameter."""
        return [self._param]

    def sites(self) -> list[RepairSite]:
        """Return the targeted noise draw site."""
        return [self._site]

    def apply_symbolic(self, mechanism: MechIR) -> MechIR:
        """Replace the noise scale with a symbolic parameter.

        Args:
            mechanism: Original mechanism.

        Returns:
            MechIR with symbolic scale.
        """
        mech = copy.deepcopy(mechanism)
        for node in mech.all_nodes():
            if isinstance(node, NoiseDrawNode) and node.node_id == self._site.node_id:
                node.scale = self._param.as_var()
                break
        return mech

    def apply_concrete(self, mechanism: MechIR, values: dict[str, float]) -> MechIR:
        """Replace the noise scale with a concrete value.

        Args:
            mechanism: Original mechanism.
            values: Parameter values.

        Returns:
            Repaired MechIR.
        """
        mech = copy.deepcopy(mechanism)
        new_scale = values.get(self._param.name, self._original)
        for node in mech.all_nodes():
            if isinstance(node, NoiseDrawNode) and node.node_id == self._site.node_id:
                node.scale = Const.real(new_scale)
                break
        return mech

    def cost_expression(self, original_values: dict[str, float]) -> TypedExpr:
        """Cost is the absolute deviation from the original scale.

        Args:
            original_values: Original parameter values.

        Returns:
            |new_scale - original_scale| expression.
        """
        from dpcegar.ir.types import Abs as IRabs
        orig = original_values.get(self._param.name, self._original)
        diff = BinOp(
            ty=IRType.REAL, op=BinOpKind.SUB,
            left=self._param.as_var(),
            right=Const.real(orig),
        )
        return IRabs(ty=IRType.REAL, operand=diff)

    def symbolic_loss(self, sensitivity: float = 1.0) -> Optional[TypedExpr]:
        """Privacy loss for Laplace mechanism: Δ / scale_var."""
        return BinOp(
            ty=IRType.REAL, op=BinOpKind.DIV,
            left=Const.real(sensitivity),
            right=self._param.as_var(),
        )


# ═══════════════════════════════════════════════════════════════════════════
# THRESHOLD SHIFT REPAIR
# ═══════════════════════════════════════════════════════════════════════════


class ThresholdShift(RepairTemplate):
    """Modify a conditional threshold to change branching behaviour.

    Shifts the threshold in a comparison (e.g. ``x < T``) to move
    the decision boundary and potentially restore privacy.
    """

    def __init__(
        self,
        site: RepairSite,
        original_threshold: float,
        min_shift: float = -100.0,
        max_shift: float = 100.0,
    ) -> None:
        """Initialise with the branch site and shift bounds.

        Args:
            site: The branch node site.
            original_threshold: Original threshold value.
            min_shift: Minimum allowed shift.
            max_shift: Maximum allowed shift.
        """
        self._site = site
        self._original = original_threshold
        self._param = RepairParameter(
            name=f"threshold_{site.node_id}",
            param_type=IRType.REAL,
            lower_bound=original_threshold + min_shift,
            upper_bound=original_threshold + max_shift,
            initial_value=original_threshold,
        )

    def name(self) -> str:
        """Return the template name."""
        return f"ThresholdShift(node={self._site.node_id})"

    def parameters(self) -> list[RepairParameter]:
        """Return the threshold parameter."""
        return [self._param]

    def sites(self) -> list[RepairSite]:
        """Return the targeted branch site."""
        return [self._site]

    def apply_symbolic(self, mechanism: MechIR) -> MechIR:
        """Replace the branch threshold with a symbolic parameter.

        Args:
            mechanism: Original mechanism.

        Returns:
            MechIR with symbolic threshold.
        """
        mech = copy.deepcopy(mechanism)
        for node in mech.all_nodes():
            if isinstance(node, BranchNode) and node.node_id == self._site.node_id:
                if isinstance(node.condition, BinOp):
                    node.condition = BinOp(
                        ty=node.condition.ty,
                        op=node.condition.op,
                        left=node.condition.left,
                        right=self._param.as_var(),
                    )
                break
        return mech

    def apply_concrete(self, mechanism: MechIR, values: dict[str, float]) -> MechIR:
        """Replace the branch threshold with a concrete value.

        Args:
            mechanism: Original mechanism.
            values: Parameter values.

        Returns:
            Repaired MechIR.
        """
        mech = copy.deepcopy(mechanism)
        new_threshold = values.get(self._param.name, self._original)
        for node in mech.all_nodes():
            if isinstance(node, BranchNode) and node.node_id == self._site.node_id:
                if isinstance(node.condition, BinOp):
                    node.condition = BinOp(
                        ty=node.condition.ty,
                        op=node.condition.op,
                        left=node.condition.left,
                        right=Const.real(new_threshold),
                    )
                break
        return mech

    def cost_expression(self, original_values: dict[str, float]) -> TypedExpr:
        """Cost is the absolute shift from the original threshold.

        Args:
            original_values: Original values.

        Returns:
            Cost expression.
        """
        from dpcegar.ir.types import Abs as IRabs
        orig = original_values.get(self._param.name, self._original)
        diff = BinOp(
            ty=IRType.REAL, op=BinOpKind.SUB,
            left=self._param.as_var(),
            right=Const.real(orig),
        )
        return IRabs(ty=IRType.REAL, operand=diff)

    def symbolic_loss(self, sensitivity: float = 1.0) -> Optional[TypedExpr]:
        """Privacy loss for threshold mechanism: Δ / threshold_var."""
        return BinOp(
            ty=IRType.REAL, op=BinOpKind.DIV,
            left=Const.real(sensitivity),
            right=self._param.as_var(),
        )


# ═══════════════════════════════════════════════════════════════════════════
# CLAMP BOUND REPAIR
# ═══════════════════════════════════════════════════════════════════════════


class ClampBound(RepairTemplate):
    """Add output clamping [lo, hi] to bound the mechanism output.

    Clamping limits the sensitivity of the mechanism by bounding
    its output range.
    """

    def __init__(
        self,
        site: RepairSite,
        default_lo: float = -100.0,
        default_hi: float = 100.0,
    ) -> None:
        """Initialise with the return site and default bounds.

        Args:
            site: The return node site.
            default_lo: Default lower clamp.
            default_hi: Default upper clamp.
        """
        self._site = site
        self._lo_param = RepairParameter(
            name=f"clamp_lo_{site.node_id}",
            param_type=IRType.REAL,
            lower_bound=-1e6,
            upper_bound=0.0,
            initial_value=default_lo,
        )
        self._hi_param = RepairParameter(
            name=f"clamp_hi_{site.node_id}",
            param_type=IRType.REAL,
            lower_bound=0.0,
            upper_bound=1e6,
            initial_value=default_hi,
        )

    def name(self) -> str:
        """Return the template name."""
        return f"ClampBound(node={self._site.node_id})"

    def parameters(self) -> list[RepairParameter]:
        """Return the lo/hi clamp parameters."""
        return [self._lo_param, self._hi_param]

    def sites(self) -> list[RepairSite]:
        """Return the targeted return site."""
        return [self._site]

    def apply_symbolic(self, mechanism: MechIR) -> MechIR:
        """Wrap the return expression with a clamp.

        Args:
            mechanism: Original mechanism.

        Returns:
            MechIR with clamped output.
        """
        mech = copy.deepcopy(mechanism)
        for node in mech.all_nodes():
            if isinstance(node, ReturnNode) and node.node_id == self._site.node_id:
                from dpcegar.ir.types import Max as IRmax, Min as IRmin
                clamped = IRmax(
                    ty=IRType.REAL,
                    left=self._lo_param.as_var(),
                    right=IRmin(
                        ty=IRType.REAL,
                        left=node.value,
                        right=self._hi_param.as_var(),
                    ),
                )
                node.value = clamped
                break
        return mech

    def apply_concrete(self, mechanism: MechIR, values: dict[str, float]) -> MechIR:
        """Apply concrete clamp bounds.

        Args:
            mechanism: Original mechanism.
            values: Parameter values.

        Returns:
            Repaired MechIR.
        """
        mech = copy.deepcopy(mechanism)
        lo = values.get(self._lo_param.name, self._lo_param.initial_value)
        hi = values.get(self._hi_param.name, self._hi_param.initial_value)
        for node in mech.all_nodes():
            if isinstance(node, ReturnNode) and node.node_id == self._site.node_id:
                from dpcegar.ir.types import Max as IRmax, Min as IRmin
                clamped = IRmax(
                    ty=IRType.REAL,
                    left=Const.real(lo),
                    right=IRmin(
                        ty=IRType.REAL,
                        left=node.value,
                        right=Const.real(hi),
                    ),
                )
                node.value = clamped
                break
        return mech

    def cost_expression(self, original_values: dict[str, float]) -> TypedExpr:
        """Cost penalises tight clamping (hi - lo should be large).

        Args:
            original_values: Unused for clamp.

        Returns:
            Cost expression: 1 / (hi - lo + 1).
        """
        width = BinOp(
            ty=IRType.REAL, op=BinOpKind.SUB,
            left=self._hi_param.as_var(),
            right=self._lo_param.as_var(),
        )
        return BinOp(
            ty=IRType.REAL, op=BinOpKind.DIV,
            left=Const.real(1.0),
            right=BinOp(
                ty=IRType.REAL, op=BinOpKind.ADD,
                left=width,
                right=Const.real(1.0),
            ),
        )

    def symbolic_loss(self, sensitivity: float = 1.0) -> Optional[TypedExpr]:
        """Privacy loss for clamped output: |hi - lo| / scale (range-based)."""
        from dpcegar.ir.types import Abs as IRabs
        width = BinOp(
            ty=IRType.REAL, op=BinOpKind.SUB,
            left=self._hi_param.as_var(),
            right=self._lo_param.as_var(),
        )
        return IRabs(ty=IRType.REAL, operand=width)


# ═══════════════════════════════════════════════════════════════════════════
# COMPOSITION BUDGET SPLIT
# ═══════════════════════════════════════════════════════════════════════════


class CompositionBudgetSplit(RepairTemplate):
    """Adjust per-iteration privacy budget in composed mechanisms.

    For mechanisms using composition (e.g. iterative queries), adjusts
    the privacy budget allocated to each iteration.
    """

    def __init__(
        self,
        loop_site: RepairSite,
        num_iterations: int,
        total_budget: float,
    ) -> None:
        """Initialise with loop information and total budget.

        Args:
            loop_site: The loop node site.
            num_iterations: Number of loop iterations.
            total_budget: Total privacy budget available.
        """
        self._site = loop_site
        self._num_iters = num_iterations
        self._total_budget = total_budget
        self._param = RepairParameter(
            name=f"iter_budget_{loop_site.node_id}",
            param_type=IRType.REAL,
            lower_bound=1e-6,
            upper_bound=total_budget,
            initial_value=total_budget / max(num_iterations, 1),
        )

    def name(self) -> str:
        """Return the template name."""
        return f"CompositionBudgetSplit(node={self._site.node_id})"

    def parameters(self) -> list[RepairParameter]:
        """Return the per-iteration budget parameter."""
        return [self._param]

    def sites(self) -> list[RepairSite]:
        """Return the loop site."""
        return [self._site]

    def apply_symbolic(self, mechanism: MechIR) -> MechIR:
        """Annotate the loop with a symbolic per-iteration budget.

        Args:
            mechanism: Original mechanism.

        Returns:
            MechIR with annotated loop.
        """
        mech = copy.deepcopy(mechanism)
        for node in mech.all_nodes():
            if isinstance(node, LoopNode) and node.node_id == self._site.node_id:
                node.annotate("per_iteration_budget", self._param.name)
                break
        return mech

    def apply_concrete(self, mechanism: MechIR, values: dict[str, float]) -> MechIR:
        """Annotate the loop with a concrete per-iteration budget.

        Args:
            mechanism: Original mechanism.
            values: Parameter values.

        Returns:
            Repaired MechIR.
        """
        mech = copy.deepcopy(mechanism)
        budget = values.get(self._param.name, self._param.initial_value)
        for node in mech.all_nodes():
            if isinstance(node, LoopNode) and node.node_id == self._site.node_id:
                node.annotate("per_iteration_budget", budget)
                break
        return mech

    def cost_expression(self, original_values: dict[str, float]) -> TypedExpr:
        """Cost penalises small per-iteration budgets (more noise).

        Args:
            original_values: Original values.

        Returns:
            Cost expression.
        """
        orig = original_values.get(
            self._param.name, self._total_budget / max(self._num_iters, 1)
        )
        from dpcegar.ir.types import Abs as IRabs
        diff = BinOp(
            ty=IRType.REAL, op=BinOpKind.SUB,
            left=self._param.as_var(),
            right=Const.real(orig),
        )
        return IRabs(ty=IRType.REAL, operand=diff)


# ═══════════════════════════════════════════════════════════════════════════
# SENSITIVITY RESCALE
# ═══════════════════════════════════════════════════════════════════════════


class SensitivityRescale(RepairTemplate):
    """Adjust the sensitivity parameter of a query.

    Changes the declared sensitivity, which may require corresponding
    noise adjustments.
    """

    def __init__(
        self,
        site: RepairSite,
        original_sensitivity: float,
    ) -> None:
        """Initialise with the query site.

        Args:
            site: The query node site.
            original_sensitivity: Original declared sensitivity.
        """
        self._site = site
        self._original = original_sensitivity
        self._param = RepairParameter(
            name=f"sensitivity_{site.node_id}",
            param_type=IRType.REAL,
            lower_bound=1e-6,
            upper_bound=1e6,
            initial_value=original_sensitivity,
        )

    def name(self) -> str:
        """Return the template name."""
        return f"SensitivityRescale(node={self._site.node_id})"

    def parameters(self) -> list[RepairParameter]:
        """Return the sensitivity parameter."""
        return [self._param]

    def sites(self) -> list[RepairSite]:
        """Return the query site."""
        return [self._site]

    def apply_symbolic(self, mechanism: MechIR) -> MechIR:
        """Replace the sensitivity with a symbolic parameter.

        Args:
            mechanism: Original mechanism.

        Returns:
            MechIR with symbolic sensitivity.
        """
        mech = copy.deepcopy(mechanism)
        for node in mech.all_nodes():
            if isinstance(node, QueryNode) and node.node_id == self._site.node_id:
                node.sensitivity = self._param.as_var()
                break
        return mech

    def apply_concrete(self, mechanism: MechIR, values: dict[str, float]) -> MechIR:
        """Replace the sensitivity with a concrete value.

        Args:
            mechanism: Original mechanism.
            values: Parameter values.

        Returns:
            Repaired MechIR.
        """
        mech = copy.deepcopy(mechanism)
        new_sens = values.get(self._param.name, self._original)
        for node in mech.all_nodes():
            if isinstance(node, QueryNode) and node.node_id == self._site.node_id:
                node.sensitivity = Const.real(new_sens)
                break
        return mech

    def cost_expression(self, original_values: dict[str, float]) -> TypedExpr:
        """Cost is the ratio of change in sensitivity.

        Args:
            original_values: Original values.

        Returns:
            Cost expression.
        """
        from dpcegar.ir.types import Abs as IRabs
        orig = original_values.get(self._param.name, self._original)
        if orig == 0:
            return self._param.as_var()
        ratio = BinOp(
            ty=IRType.REAL, op=BinOpKind.DIV,
            left=BinOp(
                ty=IRType.REAL, op=BinOpKind.SUB,
                left=self._param.as_var(),
                right=Const.real(orig),
            ),
            right=Const.real(orig),
        )
        return IRabs(ty=IRType.REAL, operand=ratio)

    def symbolic_loss(self, sensitivity: float = 1.0) -> Optional[TypedExpr]:
        """Privacy loss: sensitivity_var / scale."""
        return BinOp(
            ty=IRType.REAL, op=BinOpKind.DIV,
            left=self._param.as_var(),
            right=Const.real(sensitivity),
        )


# ═══════════════════════════════════════════════════════════════════════════
# NOISE SWAP REPAIR
# ═══════════════════════════════════════════════════════════════════════════


class NoiseSwap(RepairTemplate):
    """Change the noise distribution family at a noise draw site.

    Swaps between Laplace, Gaussian, and Exponential mechanisms.
    After swapping, the scale must also be adjusted via ScaleParam.
    """

    def __init__(
        self,
        site: RepairSite,
        original_kind: NoiseKind,
        target_kind: NoiseKind,
        scale_conversion_factor: float = 1.0,
    ) -> None:
        """Initialise with distribution swap information.

        Args:
            site: The noise draw site.
            original_kind: Original distribution.
            target_kind: Target distribution.
            scale_conversion_factor: Conversion for scale parameter.
        """
        self._site = site
        self._original_kind = original_kind
        self._target_kind = target_kind
        self._conversion = scale_conversion_factor

    def name(self) -> str:
        """Return the template name."""
        return f"NoiseSwap({self._original_kind.name}→{self._target_kind.name})"

    def parameters(self) -> list[RepairParameter]:
        """Return a scale parameter for the swapped distribution.

        Returns:
            List with a scale parameter.
        """
        return [
            RepairParameter(
                name="new_scale",
                lower_bound=1e-6,
                upper_bound=1e6,
                param_type=IRType.REAL,
            )
        ]

    def sites(self) -> list[RepairSite]:
        """Return the targeted noise site."""
        return [self._site]

    def apply_symbolic(self, mechanism: MechIR) -> MechIR:
        """Swap the distribution kind.

        Args:
            mechanism: Original mechanism.

        Returns:
            MechIR with swapped noise kind.
        """
        mech = copy.deepcopy(mechanism)
        for node in mech.all_nodes():
            if isinstance(node, NoiseDrawNode) and node.node_id == self._site.node_id:
                node.noise_kind = self._target_kind
                if self._conversion != 1.0:
                    node.scale = BinOp(
                        ty=IRType.REAL, op=BinOpKind.MUL,
                        left=node.scale,
                        right=Const.real(self._conversion),
                    )
                break
        return mech

    def apply_concrete(self, mechanism: MechIR, values: dict[str, float]) -> MechIR:
        """Swap is fully determined; values are unused.

        Args:
            mechanism: Original mechanism.
            values: Unused.

        Returns:
            MechIR with swapped noise.
        """
        return self.apply_symbolic(mechanism)

    def cost_expression(self, original_values: dict[str, float]) -> TypedExpr:
        """Noise swap has a fixed cost of 5.0 (discrete change penalty).

        Args:
            original_values: Unused.

        Returns:
            Constant cost expression.
        """
        return Const.real(5.0)


# ═══════════════════════════════════════════════════════════════════════════
# COMPOSITE REPAIR
# ═══════════════════════════════════════════════════════════════════════════


class CompositeRepair(RepairTemplate):
    """Multiple repairs applied together.

    Composes several repair templates into a single compound repair.
    The cost is the sum of individual costs.
    """

    def __init__(self, templates: list[RepairTemplate]) -> None:
        """Initialise with a list of component templates.

        Args:
            templates: Individual repair templates to compose.
        """
        self._templates = list(templates)

    def name(self) -> str:
        """Return the composite template name."""
        names = [t.name() for t in self._templates]
        return f"Composite({', '.join(names)})"

    def parameters(self) -> list[RepairParameter]:
        """Return the union of all component parameters."""
        params: list[RepairParameter] = []
        seen: set[str] = set()
        for t in self._templates:
            for p in t.parameters():
                if p.name not in seen:
                    params.append(p)
                    seen.add(p.name)
        return params

    def sites(self) -> list[RepairSite]:
        """Return the union of all component sites."""
        sites: list[RepairSite] = []
        seen: set[int] = set()
        for t in self._templates:
            for s in t.sites():
                if s.node_id not in seen:
                    sites.append(s)
                    seen.add(s.node_id)
        return sites

    def apply_symbolic(self, mechanism: MechIR) -> MechIR:
        """Apply all component templates symbolically in sequence.

        Args:
            mechanism: Original mechanism.

        Returns:
            MechIR with all symbolic modifications.
        """
        result = mechanism
        for t in self._templates:
            result = t.apply_symbolic(result)
        return result

    def apply_concrete(self, mechanism: MechIR, values: dict[str, float]) -> MechIR:
        """Apply all component templates with concrete values.

        Args:
            mechanism: Original mechanism.
            values: Parameter values for all components.

        Returns:
            Repaired MechIR.
        """
        result = mechanism
        for t in self._templates:
            result = t.apply_concrete(result, values)
        return result

    def cost_expression(self, original_values: dict[str, float]) -> TypedExpr:
        """Sum of individual component costs.

        Args:
            original_values: Original parameter values.

        Returns:
            Sum cost expression.
        """
        if not self._templates:
            return Const.real(0.0)

        total = self._templates[0].cost_expression(original_values)
        for t in self._templates[1:]:
            total = BinOp(
                ty=IRType.REAL, op=BinOpKind.ADD,
                left=total,
                right=t.cost_expression(original_values),
            )
        return total

    def add_template(self, template: RepairTemplate) -> None:
        """Add another template to the composition.

        Args:
            template: Template to add.
        """
        self._templates.append(template)

    def component_count(self) -> int:
        """Return the number of component templates.

        Returns:
            Component count.
        """
        return len(self._templates)


# ═══════════════════════════════════════════════════════════════════════════
# TEMPLATE ENUMERATOR
# ═══════════════════════════════════════════════════════════════════════════


class TemplateEnumerator:
    """Enumerate applicable repair templates for a mechanism.

    Scans the mechanism IR to find all possible repair sites and
    generates appropriate templates for each.
    """

    def __init__(
        self,
        enable_noise_swap: bool = True,
        enable_clamping: bool = True,
        enable_threshold: bool = True,
        enable_sensitivity: bool = True,
        enable_composition: bool = True,
    ) -> None:
        """Configure which template types to enumerate.

        Args:
            enable_noise_swap: Include noise distribution swaps.
            enable_clamping: Include output clamping.
            enable_threshold: Include threshold shifts.
            enable_sensitivity: Include sensitivity rescaling.
            enable_composition: Include composition budget splits.
        """
        self._enable_noise_swap = enable_noise_swap
        self._enable_clamping = enable_clamping
        self._enable_threshold = enable_threshold
        self._enable_sensitivity = enable_sensitivity
        self._enable_composition = enable_composition

    def enumerate(self, mechanism: MechIR) -> list[RepairTemplate]:
        """Enumerate all applicable templates for the mechanism.

        Args:
            mechanism: The mechanism IR to analyse.

        Returns:
            List of applicable repair templates.
        """
        templates: list[RepairTemplate] = []

        for node in mechanism.all_nodes():
            if isinstance(node, NoiseDrawNode):
                templates.extend(self._noise_templates(node))
            elif isinstance(node, BranchNode) and self._enable_threshold:
                templates.extend(self._branch_templates(node))
            elif isinstance(node, QueryNode) and self._enable_sensitivity:
                templates.extend(self._query_templates(node))
            elif isinstance(node, ReturnNode) and self._enable_clamping:
                templates.extend(self._return_templates(node))
            elif isinstance(node, LoopNode) and self._enable_composition:
                templates.extend(self._loop_templates(node, mechanism))

        return templates

    def enumerate_composites(
        self,
        mechanism: MechIR,
        max_components: int = 3,
    ) -> list[RepairTemplate]:
        """Enumerate composite repairs combining multiple templates.

        Generates all subsets of applicable templates up to
        *max_components* elements.

        Args:
            mechanism: The mechanism IR.
            max_components: Maximum templates per composite.

        Returns:
            List of composite repair templates.
        """
        singles = self.enumerate(mechanism)
        composites: list[RepairTemplate] = list(singles)

        if max_components >= 2:
            for i in range(len(singles)):
                for j in range(i + 1, len(singles)):
                    sites_i = {s.node_id for s in singles[i].sites()}
                    sites_j = {s.node_id for s in singles[j].sites()}
                    if not sites_i & sites_j:
                        composites.append(
                            CompositeRepair([singles[i], singles[j]])
                        )

        return composites

    def _noise_templates(self, node: NoiseDrawNode) -> list[RepairTemplate]:
        """Generate templates for a noise draw node.

        Args:
            node: The noise draw node.

        Returns:
            List of applicable templates.
        """
        site = RepairSite(
            node_id=node.node_id,
            node_type="NoiseDrawNode",
            description=f"Noise draw: {node.target}",
        )
        templates: list[RepairTemplate] = []

        original_scale = 1.0
        if isinstance(node.scale, Const):
            original_scale = float(node.scale.value)

        templates.append(ScaleParam(site, original_scale))

        if self._enable_noise_swap:
            for kind in NoiseKind:
                if kind != node.noise_kind:
                    conversion = self._noise_conversion(node.noise_kind, kind)
                    templates.append(NoiseSwap(site, node.noise_kind, kind, conversion))

        return templates

    def _branch_templates(self, node: BranchNode) -> list[RepairTemplate]:
        """Generate templates for a branch node.

        Args:
            node: The branch node.

        Returns:
            List of applicable templates.
        """
        if not isinstance(node.condition, BinOp):
            return []
        if not isinstance(node.condition.right, Const):
            return []

        site = RepairSite(
            node_id=node.node_id,
            node_type="BranchNode",
            description=f"Branch: {node.condition}",
            current_value=float(node.condition.right.value),
        )
        return [ThresholdShift(site, float(node.condition.right.value))]

    def _query_templates(self, node: QueryNode) -> list[RepairTemplate]:
        """Generate templates for a query node.

        Args:
            node: The query node.

        Returns:
            List of applicable templates.
        """
        orig_sens = 1.0
        if isinstance(node.sensitivity, Const):
            orig_sens = float(node.sensitivity.value)

        site = RepairSite(
            node_id=node.node_id,
            node_type="QueryNode",
            description=f"Query: {node.query_name}",
            current_value=orig_sens,
        )
        return [SensitivityRescale(site, orig_sens)]

    def _return_templates(self, node: ReturnNode) -> list[RepairTemplate]:
        """Generate templates for a return node.

        Args:
            node: The return node.

        Returns:
            List of applicable templates.
        """
        site = RepairSite(
            node_id=node.node_id,
            node_type="ReturnNode",
            description="Output clamping",
        )
        return [ClampBound(site)]

    def _loop_templates(self, node: LoopNode, mechanism: MechIR) -> list[RepairTemplate]:
        """Generate templates for a loop node.

        Args:
            node: The loop node.
            mechanism: Full mechanism for budget info.

        Returns:
            List of applicable templates.
        """
        site = RepairSite(
            node_id=node.node_id,
            node_type="LoopNode",
            description=f"Loop: {node.index_var}",
        )
        num_iters = 1
        if isinstance(node.bound, Const):
            num_iters = int(node.bound.value)

        total_budget = 1.0
        if mechanism.budget is not None:
            eps, _ = mechanism.budget.to_approx_dp()
            total_budget = eps

        return [CompositionBudgetSplit(site, num_iters, total_budget)]

    @staticmethod
    def _noise_conversion(from_kind: NoiseKind, to_kind: NoiseKind) -> float:
        """Compute scale conversion factor between noise distributions.

        Args:
            from_kind: Source distribution.
            to_kind: Target distribution.

        Returns:
            Scale conversion factor.
        """
        conversions = {
            (NoiseKind.LAPLACE, NoiseKind.GAUSSIAN): math.sqrt(2.0),
            (NoiseKind.GAUSSIAN, NoiseKind.LAPLACE): 1.0 / math.sqrt(2.0),
            (NoiseKind.LAPLACE, NoiseKind.EXPONENTIAL): 1.0,
            (NoiseKind.EXPONENTIAL, NoiseKind.LAPLACE): 1.0,
            (NoiseKind.GAUSSIAN, NoiseKind.EXPONENTIAL): 1.0 / math.sqrt(2.0),
            (NoiseKind.EXPONENTIAL, NoiseKind.GAUSSIAN): math.sqrt(2.0),
        }
        return conversions.get((from_kind, to_kind), 1.0)


# ═══════════════════════════════════════════════════════════════════════════
# TEMPLATE COST
# ═══════════════════════════════════════════════════════════════════════════


class TemplateCost:
    """Compute and compare costs of repair templates.

    The cost reflects how much the repair changes the mechanism from
    its original form.  Lower cost means a more desirable repair.
    """

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        """Initialise with optional per-template-type weights.

        Args:
            weights: Mapping from template type name to weight multiplier.
        """
        self._weights = weights or {
            "ScaleParam": 1.0,
            "ThresholdShift": 2.0,
            "ClampBound": 3.0,
            "SensitivityRescale": 2.0,
            "NoiseSwap": 5.0,
            "CompositionBudgetSplit": 2.0,
            "Composite": 1.0,
        }

    def estimate(self, template: RepairTemplate) -> float:
        """Estimate the cost of a template before synthesis.

        Args:
            template: The repair template.

        Returns:
            Estimated cost (lower is better).
        """
        base_name = type(template).__name__
        weight = self._weights.get(base_name, 1.0)
        num_params = len(template.parameters())
        num_sites = len(template.sites())
        return weight * (1 + num_params) * (1 + num_sites * 0.5)

    def rank(self, templates: list[RepairTemplate]) -> list[RepairTemplate]:
        """Rank templates by estimated cost (cheapest first).

        Args:
            templates: Templates to rank.

        Returns:
            Templates sorted by ascending cost.
        """
        scored = [(self.estimate(t), t) for t in templates]
        scored.sort(key=lambda x: x[0])
        return [t for _, t in scored]


# ═══════════════════════════════════════════════════════════════════════════
# TEMPLATE VALIDATOR
# ═══════════════════════════════════════════════════════════════════════════


class TemplateValidator:
    """Check template applicability constraints.

    Validates that a template's parameters are within valid ranges
    and that the repair sites exist in the mechanism.
    """

    def validate(
        self, template: RepairTemplate, mechanism: MechIR
    ) -> ValidationResult:
        """Validate a template against a mechanism.

        Args:
            template: The repair template to validate.
            mechanism: The target mechanism.

        Returns:
            A ValidationResult.
        """
        errors: list[str] = []
        warnings: list[str] = []

        for site in template.sites():
            node = mechanism.find_node(site.node_id)
            if node is None:
                errors.append(f"Site node {site.node_id} not found in mechanism")
            elif type(node).__name__ != site.node_type:
                warnings.append(
                    f"Site {site.node_id} expected {site.node_type}, "
                    f"got {type(node).__name__}"
                )

        for param in template.parameters():
            if param.lower_bound > param.upper_bound:
                errors.append(
                    f"Parameter {param.name}: lower > upper "
                    f"({param.lower_bound} > {param.upper_bound})"
                )
            if param.initial_value < param.lower_bound:
                warnings.append(
                    f"Parameter {param.name}: initial < lower "
                    f"({param.initial_value} < {param.lower_bound})"
                )
            if param.initial_value > param.upper_bound:
                warnings.append(
                    f"Parameter {param.name}: initial > upper "
                    f"({param.initial_value} > {param.upper_bound})"
                )

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Result of template validation.

    Attributes:
        is_valid: Whether the template is valid.
        errors: List of validation errors.
        warnings: List of validation warnings.
    """

    is_valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        if self.is_valid:
            return f"Valid({len(self.warnings)} warnings)"
        return f"Invalid({len(self.errors)} errors)"
