"""Implication lattice for differential privacy notions.

Models the implication relationships between DP variants as a directed
acyclic graph.  Verified results propagate downward (stronger → weaker)
and counterexamples propagate upward (weaker → stronger).

Lattice structure::

    PureDP ──→ ApproxDP
    PureDP ──→ zCDP ──→ RDP ──→ ApproxDP
    GDP ──→ fDP ──→ ApproxDP

The lattice enables the DP-CEGAR loop to:

1.  **Skip redundant checks**: if PureDP is verified, all weaker notions
    (ApproxDP, zCDP, RDP) are automatically satisfied.
2.  **Prune early**: if ApproxDP is falsified, all stronger notions
    (PureDP, zCDP, RDP, GDP, fDP) are immediately falsified.
3.  **Derive concrete budgets**: given a verified PureDP budget ε, compute
    the implied (ε, δ)-DP, ρ-zCDP, and (α, ε_RDP)-RDP budgets via the
    standard conversion theorems.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Iterator, Sequence

logger = logging.getLogger(__name__)

from dpcegar.ir.types import (
    ApproxBudget,
    FDPBudget,
    GDPBudget,
    PrivacyBudget,
    PrivacyNotion,
    PureBudget,
    RDPBudget,
    ZCDPBudget,
)
from dpcegar.utils.math_utils import phi, phi_inv


# ═══════════════════════════════════════════════════════════════════════════
# 1.  NODE STATUS
# ═══════════════════════════════════════════════════════════════════════════


class NodeStatus(Enum):
    """Verification status of a privacy notion in the lattice."""

    VERIFIED = auto()
    """The mechanism satisfies this DP notion."""

    FALSIFIED = auto()
    """A counterexample disproves this DP notion."""

    UNKNOWN = auto()
    """Not yet determined."""

    def __str__(self) -> str:
        return self.name.lower()


# ═══════════════════════════════════════════════════════════════════════════
# 2.  LATTICE NODE
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class PrivacyLatticeNode:
    """A single node in the implication lattice.

    Each node corresponds to one :class:`PrivacyNotion` and tracks
    its verification status, the associated budget (if any), and
    whether the result was derived from another node.

    Attributes:
        notion:              The DP variant this node represents.
        status:              Current verification status.
        budget:              Concrete budget, if known.
        verification_result: Opaque handle to the full CEGAR result
                             (e.g., a ``CEGARResult`` instance).
        derived:             ``True`` when the status was inferred from
                             another node via lattice propagation.
        derived_from:        The notion from which the status was derived,
                             or ``None`` if it was verified directly.
    """

    notion: PrivacyNotion
    status: NodeStatus = NodeStatus.UNKNOWN
    budget: PrivacyBudget | None = None
    verification_result: Any = None
    derived: bool = False
    derived_from: PrivacyNotion | None = None

    # -- helpers -----------------------------------------------------------

    @property
    def is_verified(self) -> bool:
        """``True`` if this notion has been verified."""
        return self.status is NodeStatus.VERIFIED

    @property
    def is_falsified(self) -> bool:
        """``True`` if this notion has been falsified."""
        return self.status is NodeStatus.FALSIFIED

    @property
    def is_unknown(self) -> bool:
        """``True`` if the status is still unknown."""
        return self.status is NodeStatus.UNKNOWN

    def reset(self) -> None:
        """Clear all verification state, restoring ``UNKNOWN``."""
        self.status = NodeStatus.UNKNOWN
        self.budget = None
        self.verification_result = None
        self.derived = False
        self.derived_from = None

    def __str__(self) -> str:
        tag = " (derived)" if self.derived else ""
        budget_str = f", budget={self.budget}" if self.budget is not None else ""
        return f"{self.notion.name}: {self.status}{budget_str}{tag}"


# ═══════════════════════════════════════════════════════════════════════════
# 3.  IMPLICATION EDGE
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class ImplicationEdge:
    """A directed edge in the implication lattice.

    An edge from *source* to *target* means:

        *source*-DP  ⟹  *target*-DP

    The *transform* callable converts a budget for the source notion
    into a budget for the target notion using the conversion theorem
    identified by *theorem_ref*.

    Attributes:
        source:      Stronger DP notion.
        target:      Weaker DP notion (implied by *source*).
        transform:   Budget conversion function.
        theorem_ref: Literature reference for the conversion theorem.
        is_tight:    ``True`` if the conversion is known to be tight.
    """

    source: PrivacyNotion
    target: PrivacyNotion
    transform: Callable[..., PrivacyBudget]
    theorem_ref: str
    is_tight: bool = False

    def __str__(self) -> str:
        tight = " (tight)" if self.is_tight else ""
        return f"{self.source.name} → {self.target.name}{tight}  [{self.theorem_ref}]"


# ═══════════════════════════════════════════════════════════════════════════
# 4.  PARAMETER TRANSFORMER
# ═══════════════════════════════════════════════════════════════════════════


class ParameterTransformer:
    """Stateless helper that converts privacy budgets between DP notions.

    Every public method implements a conversion theorem from the
    differential privacy literature.  The ``transform`` dispatcher
    selects the right method based on the *target_notion*.
    """

    # -- PureDP conversions ------------------------------------------------

    @staticmethod
    def pure_to_approx(budget: PureBudget) -> ApproxBudget:
        r"""Convert (ε)-DP → (ε, 0)-DP.

        Pure DP is a strict special case of approximate DP with δ = 0.

        Reference:
            Dwork & Roth, *The Algorithmic Foundations of Differential
            Privacy*, Proposition 3.2.
        """
        return ApproxBudget(epsilon=budget.epsilon, delta=0.0)

    @staticmethod
    def pure_to_zcdp(budget: PureBudget) -> ZCDPBudget:
        r"""Convert (ε)-DP → ρ-zCDP with ρ = ε²/2.

        Reference:
            Bun & Murtagh (2016), *Concentrated Differential Privacy*,
            Proposition 1.4.  An ε-DP mechanism is (ε²/2)-zCDP.
        """
        rho = (budget.epsilon ** 2) / 2.0
        return ZCDPBudget(rho=rho)

    # -- zCDP conversions --------------------------------------------------

    @staticmethod
    def zcdp_to_rdp(budget: ZCDPBudget, alpha: float = 2.0) -> RDPBudget:
        r"""Convert ρ-zCDP → (α, ρ·α)-RDP for a given Rényi order α > 1.

        A ρ-zCDP mechanism satisfies (α, ρα)-RDP for every α > 1.

        Reference:
            Bun & Murtagh (2016), Proposition 1.3.
        """
        if alpha <= 1.0:
            raise ValueError(f"Rényi order α must be > 1, got {alpha}")
        return RDPBudget(alpha=alpha, epsilon=budget.rho * alpha)

    @staticmethod
    def zcdp_to_approx(budget: ZCDPBudget, delta: float = 1e-5) -> ApproxBudget:
        r"""Convert ρ-zCDP → (ε, δ)-DP.

        Uses the optimal conversion:

        .. math::

            \varepsilon = \rho + 2\sqrt{\rho \ln(1/\delta)}

        Reference:
            Bun & Murtagh (2016), Proposition 1.6.
        """
        if delta <= 0.0 or delta >= 1.0:
            raise ValueError(f"δ must be in (0, 1), got {delta}")
        rho = budget.rho
        epsilon = rho + 2.0 * math.sqrt(rho * math.log(1.0 / delta))
        return ApproxBudget(epsilon=epsilon, delta=delta)

    # -- RDP conversions ---------------------------------------------------

    @staticmethod
    def rdp_to_approx(budget: RDPBudget, delta: float = 1e-5) -> ApproxBudget:
        r"""Convert (α, ε_RDP)-RDP → (ε, δ)-DP.

        .. math::

            \varepsilon = \varepsilon_{\mathrm{RDP}}
                          + \frac{\ln(1/\delta)}{\alpha - 1}

        Reference:
            Mironov (2017), *Rényi Differential Privacy*, Proposition 3.
        """
        if delta <= 0.0 or delta >= 1.0:
            raise ValueError(f"δ must be in (0, 1), got {delta}")
        alpha = budget.alpha
        epsilon = budget.epsilon + math.log(1.0 / delta) / (alpha - 1.0)
        return ApproxBudget(epsilon=epsilon, delta=delta)

    # -- GDP conversions ---------------------------------------------------

    @staticmethod
    def gdp_to_fdp(budget: GDPBudget) -> FDPBudget:
        r"""Convert μ-GDP → f-DP with trade-off function T(α).

        The trade-off function for μ-GDP is:

        .. math::

            T(\alpha) = \Phi\bigl(\Phi^{-1}(1 - \alpha) - \mu\bigr)

        where Φ is the standard normal CDF.

        Reference:
            Dong, Roth & Su (2022), *Gaussian Differential Privacy*,
            Theorem 2.1.
        """
        mu = budget.mu

        def _trade_off(alpha: float) -> float:
            """Trade-off function T(α) = Φ(Φ⁻¹(1−α) − μ)."""
            if alpha <= 0.0:
                return 1.0
            if alpha >= 1.0:
                return 0.0
            return phi(phi_inv(1.0 - alpha) - mu)

        return FDPBudget(trade_off_fn=_trade_off)

    @staticmethod
    def fdp_to_approx(
        budget: FDPBudget,
        delta: float = 1e-5,
        tol: float = 1e-10,
        max_iter: int = 200,
    ) -> ApproxBudget:
        r"""Convert f-DP → (ε, δ)-DP via binary search.

        For a trade-off function *T*, the mechanism satisfies (ε, δ)-DP
        when *T(1 − δ) ≤ e^{−ε} · δ*, which is equivalent to finding
        the smallest ε such that:

        .. math::

            1 - \delta - T(1 - \delta) \;\leq\; (e^{\varepsilon} - 1)\,\delta

        We binary-search for the tightest ε satisfying the dual
        characterisation:

        .. math::

            \delta \;\geq\;
            \inf_{\alpha \in (0,1)}
            \frac{T(\alpha) - (1-\alpha)}{e^{\varepsilon}\alpha - (1-\alpha)}

        but in practice we use the simpler point-wise check on a grid
        and then refine with bisection.

        Reference:
            Dong, Roth & Su (2022), Section 3.
        """
        if delta <= 0.0 or delta >= 1.0:
            raise ValueError(f"δ must be in (0, 1), got {delta}")

        T = budget.trade_off_fn

        # Upper bound: try to find a finite ε using a coarse grid.
        eps_lo = 0.0
        eps_hi = 50.0

        def _satisfies(eps: float) -> bool:
            """Check if (eps, delta)-DP holds given trade-off T."""
            n_points = 500
            for i in range(1, n_points):
                alpha = i / n_points
                # For (eps, delta)-DP the trade-off must satisfy
                # T(alpha) >= max(0, 1 - delta - e^eps * alpha,
                #                    e^{-eps} * (1 - delta - alpha))
                lower_1 = 1.0 - delta - math.exp(eps) * alpha
                lower_2 = math.exp(-eps) * (1.0 - delta - alpha)
                lower_bound = max(0.0, lower_1, lower_2)
                if T(alpha) < lower_bound - 1e-12:
                    return False
            return True

        # Binary search for tightest ε.
        if _satisfies(eps_lo):
            return ApproxBudget(epsilon=0.0, delta=delta)

        # Ensure upper bound is sufficient; double if needed.
        for _ in range(20):
            if _satisfies(eps_hi):
                break
            eps_hi *= 2.0
        else:
            # Return a conservative large epsilon
            return ApproxBudget(epsilon=eps_hi, delta=delta)

        for _ in range(max_iter):
            eps_mid = (eps_lo + eps_hi) / 2.0
            if eps_hi - eps_lo < tol:
                break
            if _satisfies(eps_mid):
                eps_hi = eps_mid
            else:
                eps_lo = eps_mid

        return ApproxBudget(epsilon=eps_hi, delta=delta)

    # -- GDP direct to ApproxDP --------------------------------------------

    @staticmethod
    def gdp_to_approx(budget: GDPBudget, delta: float = 1e-5) -> ApproxBudget:
        r"""Convert μ-GDP → (ε, δ)-DP directly.

        Uses the closed-form:

        .. math::

            \delta = \Phi(-\varepsilon/\mu + \mu/2)
                    - e^{\varepsilon}\,\Phi(-\varepsilon/\mu - \mu/2)

        and binary-searches for the smallest ε satisfying the given δ.

        Reference:
            Dong, Roth & Su (2022), Corollary 2.13.
        """
        if delta <= 0.0 or delta >= 1.0:
            raise ValueError(f"δ must be in (0, 1), got {delta}")
        mu = budget.mu
        if mu == 0.0:
            return ApproxBudget(epsilon=0.0, delta=delta)

        def _delta_of_eps(eps: float) -> float:
            """Compute δ(ε) for μ-GDP."""
            return (
                phi(-eps / mu + mu / 2.0)
                - math.exp(eps) * phi(-eps / mu - mu / 2.0)
            )

        eps_lo = 0.0
        eps_hi = mu * 10.0 + 50.0

        for _ in range(200):
            eps_mid = (eps_lo + eps_hi) / 2.0
            if eps_hi - eps_lo < 1e-12:
                break
            if _delta_of_eps(eps_mid) > delta:
                eps_lo = eps_mid
            else:
                eps_hi = eps_mid

        return ApproxBudget(epsilon=eps_hi, delta=delta)

    # -- Dispatch ----------------------------------------------------------

    @staticmethod
    def transform(
        source_budget: PrivacyBudget,
        target_notion: PrivacyNotion,
        **kwargs: Any,
    ) -> PrivacyBudget:
        """Convert *source_budget* into an equivalent budget for *target_notion*.

        This is the main entry point.  It dispatches to the appropriate
        conversion method based on the source and target notions.

        Args:
            source_budget:  Budget to convert.
            target_notion:  Desired target DP notion.
            **kwargs:       Extra parameters forwarded to the conversion
                            method (e.g. ``delta``, ``alpha``).

        Returns:
            A :class:`PrivacyBudget` for the target notion.

        Raises:
            ValueError: If the conversion is not supported.
        """
        src = source_budget.notion
        key = (src, target_notion)

        dispatch: dict[
            tuple[PrivacyNotion, PrivacyNotion],
            Callable[..., PrivacyBudget],
        ] = {
            (PrivacyNotion.PURE_DP, PrivacyNotion.APPROX_DP): lambda: ParameterTransformer.pure_to_approx(source_budget),  # type: ignore[arg-type]
            (PrivacyNotion.PURE_DP, PrivacyNotion.ZCDP): lambda: ParameterTransformer.pure_to_zcdp(source_budget),  # type: ignore[arg-type]
            (PrivacyNotion.ZCDP, PrivacyNotion.RDP): lambda: ParameterTransformer.zcdp_to_rdp(source_budget, **kwargs),  # type: ignore[arg-type]
            (PrivacyNotion.ZCDP, PrivacyNotion.APPROX_DP): lambda: ParameterTransformer.zcdp_to_approx(source_budget, **kwargs),  # type: ignore[arg-type]
            (PrivacyNotion.RDP, PrivacyNotion.APPROX_DP): lambda: ParameterTransformer.rdp_to_approx(source_budget, **kwargs),  # type: ignore[arg-type]
            (PrivacyNotion.GDP, PrivacyNotion.FDP): lambda: ParameterTransformer.gdp_to_fdp(source_budget),  # type: ignore[arg-type]
            (PrivacyNotion.GDP, PrivacyNotion.APPROX_DP): lambda: ParameterTransformer.gdp_to_approx(source_budget, **kwargs),  # type: ignore[arg-type]
            (PrivacyNotion.FDP, PrivacyNotion.APPROX_DP): lambda: ParameterTransformer.fdp_to_approx(source_budget, **kwargs),  # type: ignore[arg-type]
        }

        if key not in dispatch:
            raise ValueError(
                f"No direct conversion from {src.name} to {target_notion.name}. "
                f"Supported edges: {', '.join(f'{s.name}→{t.name}' for s, t in dispatch)}"
            )

        return dispatch[key]()


# Singleton transformer instance
_TRANSFORMER = ParameterTransformer()


# ═══════════════════════════════════════════════════════════════════════════
# 5.  IMPLICATION LATTICE
# ═══════════════════════════════════════════════════════════════════════════


# Recommended verification order: weakest first, so that a falsification
# at ApproxDP immediately prunes all stronger notions.
_QUERY_ORDER: list[PrivacyNotion] = [
    PrivacyNotion.APPROX_DP,
    PrivacyNotion.FDP,
    PrivacyNotion.RDP,
    PrivacyNotion.ZCDP,
    PrivacyNotion.GDP,
    PrivacyNotion.PURE_DP,
]


class ImplicationLattice:
    """Directed acyclic graph of DP notion implications.

    The lattice maintains a :class:`PrivacyLatticeNode` for each of the
    six supported :class:`PrivacyNotion` values and a set of directed
    :class:`ImplicationEdge` instances encoding the implication
    relationships.

    The two key propagation operations are:

    *  **propagate_verified(notion)** – if *notion* is verified, every
       weaker notion reachable via forward edges is also verified.
    *  **propagate_falsified(notion)** – if *notion* is falsified, every
       stronger notion reachable via backward edges is also falsified.

    Example::

        lattice = ImplicationLattice()
        lattice.update_node(PrivacyNotion.PURE_DP, NodeStatus.VERIFIED,
                            budget=PureBudget(epsilon=1.0))
        assert lattice.get_node(PrivacyNotion.APPROX_DP).is_verified
    """

    def __init__(self) -> None:
        self._nodes: dict[PrivacyNotion, PrivacyLatticeNode] = {}
        self._edges: list[ImplicationEdge] = []
        self._forward: dict[PrivacyNotion, list[ImplicationEdge]] = defaultdict(list)
        self._backward: dict[PrivacyNotion, list[ImplicationEdge]] = defaultdict(list)
        self._transformer: ParameterTransformer = _TRANSFORMER
        self._build_lattice()

    # -- Construction ------------------------------------------------------

    def _build_lattice(self) -> None:
        """Populate nodes and edges.

        Nodes
        -----
        One node for each :class:`PrivacyNotion` member.

        Edges (stronger → weaker)
        -------------------------
        1.  PureDP  → ApproxDP   (ε → (ε, 0))
        2.  PureDP  → zCDP       (ε → ρ = ε²/2)
        3.  zCDP    → RDP        (ρ → (α, ρα))
        4.  RDP     → ApproxDP   ((α, ε_RDP) → (ε_RDP + ln(1/δ)/(α−1), δ))
        5.  GDP     → fDP        (μ → T(α) = Φ(Φ⁻¹(1−α) − μ))
        6.  fDP     → ApproxDP   (T → (ε, δ) via binary search)
        7.  zCDP    → ApproxDP   (ρ → (ρ + 2√(ρ ln(1/δ)), δ))
        """
        # Nodes
        for notion in PrivacyNotion:
            self._nodes[notion] = PrivacyLatticeNode(notion=notion)

        # Helper to register an edge
        def _add(
            src: PrivacyNotion,
            tgt: PrivacyNotion,
            xform: Callable[..., PrivacyBudget],
            ref: str,
            tight: bool = False,
        ) -> None:
            edge = ImplicationEdge(
                source=src,
                target=tgt,
                transform=xform,
                theorem_ref=ref,
                is_tight=tight,
            )
            self._edges.append(edge)
            self._forward[src].append(edge)
            self._backward[tgt].append(edge)

        # Edge 1: PureDP → ApproxDP
        _add(
            PrivacyNotion.PURE_DP,
            PrivacyNotion.APPROX_DP,
            lambda b, **kw: ParameterTransformer.pure_to_approx(b),
            "Dwork & Roth, Proposition 3.2",
            tight=True,
        )

        # Edge 2: PureDP → zCDP
        _add(
            PrivacyNotion.PURE_DP,
            PrivacyNotion.ZCDP,
            lambda b, **kw: ParameterTransformer.pure_to_zcdp(b),
            "Bun & Murtagh 2016, Proposition 1.4",
            tight=True,
        )

        # Edge 3: zCDP → RDP
        _add(
            PrivacyNotion.ZCDP,
            PrivacyNotion.RDP,
            lambda b, **kw: ParameterTransformer.zcdp_to_rdp(b, **kw),
            "Bun & Murtagh 2016, Proposition 1.3",
            tight=True,
        )

        # Edge 4: RDP → ApproxDP
        _add(
            PrivacyNotion.RDP,
            PrivacyNotion.APPROX_DP,
            lambda b, **kw: ParameterTransformer.rdp_to_approx(b, **kw),
            "Mironov 2017, Proposition 3",
            tight=False,
        )

        # Edge 5: GDP → fDP
        _add(
            PrivacyNotion.GDP,
            PrivacyNotion.FDP,
            lambda b, **kw: ParameterTransformer.gdp_to_fdp(b),
            "Dong, Roth & Su 2022, Theorem 2.1",
            tight=True,
        )

        # Edge 6: fDP → ApproxDP
        _add(
            PrivacyNotion.FDP,
            PrivacyNotion.APPROX_DP,
            lambda b, **kw: ParameterTransformer.fdp_to_approx(b, **kw),
            "Dong, Roth & Su 2022, Section 3",
            tight=False,
        )

        # Edge 7: zCDP → ApproxDP  (direct, tighter than zCDP→RDP→ApproxDP)
        _add(
            PrivacyNotion.ZCDP,
            PrivacyNotion.APPROX_DP,
            lambda b, **kw: ParameterTransformer.zcdp_to_approx(b, **kw),
            "Bun & Murtagh 2016, Proposition 1.6",
            tight=False,
        )

    # -- Node access -------------------------------------------------------

    def get_node(self, notion: PrivacyNotion) -> PrivacyLatticeNode:
        """Return the lattice node for *notion*.

        Args:
            notion: The DP variant to look up.

        Returns:
            The corresponding :class:`PrivacyLatticeNode`.

        Raises:
            KeyError: If *notion* is not in the lattice.
        """
        if notion not in self._nodes:
            raise KeyError(f"Unknown notion: {notion}")
        return self._nodes[notion]

    def get_edges_from(self, notion: PrivacyNotion) -> list[ImplicationEdge]:
        """Return all edges originating from *notion* (outgoing implications).

        Args:
            notion: Source DP notion.

        Returns:
            List of edges where *notion* is the stronger (source) side.
        """
        return list(self._forward.get(notion, []))

    def get_edges_to(self, notion: PrivacyNotion) -> list[ImplicationEdge]:
        """Return all edges terminating at *notion* (incoming implications).

        Args:
            notion: Target DP notion.

        Returns:
            List of edges where *notion* is the weaker (target) side.
        """
        return list(self._backward.get(notion, []))

    @property
    def all_edges(self) -> list[ImplicationEdge]:
        """Return a copy of the full edge list."""
        return list(self._edges)

    @property
    def all_nodes(self) -> list[PrivacyLatticeNode]:
        """Return a copy of the full node list in enum order."""
        return [self._nodes[n] for n in PrivacyNotion]

    # -- Successors / predecessors -----------------------------------------

    def successors(self, notion: PrivacyNotion) -> set[PrivacyNotion]:
        """Return the set of notions *directly* implied by *notion*.

        These are the immediate successors (one hop) in the lattice.

        Args:
            notion: Source notion.

        Returns:
            Set of directly implied (weaker) notions.
        """
        return {e.target for e in self._forward.get(notion, [])}

    def predecessors(self, notion: PrivacyNotion) -> set[PrivacyNotion]:
        """Return the set of notions that *directly* imply *notion*.

        These are the immediate predecessors (one hop) in the lattice.

        Args:
            notion: Target notion.

        Returns:
            Set of notions that directly imply (are stronger than) *notion*.
        """
        return {e.source for e in self._backward.get(notion, [])}

    def implies(self, source: PrivacyNotion, target: PrivacyNotion) -> bool:
        """Return ``True`` if *source* implies *target* via forward edges (BFS)."""
        if source == target:
            return True
        visited: set[PrivacyNotion] = set()
        queue: list[PrivacyNotion] = [source]
        while queue:
            current = queue.pop(0)
            if current == target:
                return True
            if current in visited:
                continue
            visited.add(current)
            for edge in self._forward.get(current, []):
                if edge.target not in visited:
                    queue.append(edge.target)
        return False

    # -- Status update and propagation -------------------------------------

    def update_node(
        self,
        notion: PrivacyNotion,
        status: NodeStatus,
        result: Any = None,
        budget: PrivacyBudget | None = None,
    ) -> None:
        """Update a node's status and trigger lattice propagation.

        If *status* is ``VERIFIED``, all reachable weaker notions are
        marked as verified (with ``derived=True``).  If *status* is
        ``FALSIFIED``, all reachable stronger notions are marked as
        falsified.

        Args:
            notion:  The DP notion to update.
            status:  New verification status.
            result:  Optional opaque verification result.
            budget:  Optional concrete budget.
        """
        node = self.get_node(notion)
        node.status = status
        node.verification_result = result
        if budget is not None:
            node.budget = budget

        if status is NodeStatus.VERIFIED:
            self.propagate_verified(notion)
        elif status is NodeStatus.FALSIFIED:
            self.propagate_falsified(notion)

    def propagate_verified(self, notion: PrivacyNotion) -> None:
        """Mark all weaker notions reachable from *notion* as verified.

        Traverses forward edges (stronger → weaker) using BFS.  Each
        reached node is marked ``VERIFIED`` with ``derived=True`` and
        its budget is computed via the edge transform when a source
        budget is available.

        Args:
            notion: The verified notion to propagate from.
        """
        source_node = self.get_node(notion)
        queue: deque[PrivacyNotion] = deque([notion])
        visited: set[PrivacyNotion] = {notion}

        while queue:
            current = queue.popleft()
            current_node = self.get_node(current)

            for edge in self._forward.get(current, []):
                target = edge.target
                if target in visited:
                    continue
                visited.add(target)

                target_node = self.get_node(target)
                # Only propagate if the target is still unknown.
                if target_node.is_unknown:
                    target_node.status = NodeStatus.VERIFIED
                    target_node.derived = True
                    target_node.derived_from = notion

                    # Derive the budget if we have one on the current node.
                    if current_node.budget is not None:
                        try:
                            derived_budget = edge.transform(current_node.budget)
                            target_node.budget = derived_budget
                        except (ValueError, TypeError):
                            logger.debug("Budget conversion failed for edge to %s", target)

                queue.append(target)

    def propagate_falsified(self, notion: PrivacyNotion) -> None:
        """Mark all stronger notions reachable from *notion* as falsified.

        Traverses backward edges (weaker → stronger) using BFS.  Each
        reached node is marked ``FALSIFIED`` with ``derived=True``.

        Args:
            notion: The falsified notion to propagate from.
        """
        queue: deque[PrivacyNotion] = deque([notion])
        visited: set[PrivacyNotion] = {notion}

        while queue:
            current = queue.popleft()
            for edge in self._backward.get(current, []):
                source = edge.source
                if source in visited:
                    continue
                visited.add(source)

                source_node = self.get_node(source)
                if source_node.is_unknown:
                    source_node.status = NodeStatus.FALSIFIED
                    source_node.derived = True
                    source_node.derived_from = notion

                queue.append(source)

    # -- Query order -------------------------------------------------------

    def query_order(self) -> list[PrivacyNotion]:
        """Return the optimal order for checking DP notions.

        The CEGAR loop should check the *weakest* notion first (ApproxDP).
        If ApproxDP is falsified, all stronger notions are immediately
        pruned.  If it is verified, proceed to the next weakest notion.

        Returns:
            A list of :class:`PrivacyNotion` from weakest to strongest,
            excluding notions whose status is already known.
        """
        return [n for n in _QUERY_ORDER if self.get_node(n).is_unknown]

    # -- Deriving guarantees -----------------------------------------------

    def derive_guarantees(
        self,
        notion: PrivacyNotion,
        budget: PrivacyBudget,
        **kwargs: Any,
    ) -> list[tuple[PrivacyNotion, PrivacyBudget]]:
        """Compute all derived budgets reachable from *notion* via edges.

        Starting from a verified budget for *notion*, walks the forward
        edges and applies each transform to produce concrete budgets for
        every reachable weaker notion.

        Args:
            notion: The starting (verified) notion.
            budget: The verified budget.
            **kwargs: Extra parameters forwarded to transforms
                      (e.g. ``delta``, ``alpha``).

        Returns:
            A list of ``(notion, budget)`` pairs for all reachable
            weaker notions.
        """
        results: list[tuple[PrivacyNotion, PrivacyBudget]] = []
        queue: deque[tuple[PrivacyNotion, PrivacyBudget]] = deque([(notion, budget)])
        visited: set[PrivacyNotion] = {notion}

        while queue:
            current_notion, current_budget = queue.popleft()

            for edge in self._forward.get(current_notion, []):
                target = edge.target
                if target in visited:
                    continue
                visited.add(target)

                try:
                    derived = edge.transform(current_budget, **kwargs)
                    results.append((target, derived))
                    queue.append((target, derived))
                except (ValueError, TypeError):
                    # Cannot convert with the given parameters; skip this
                    # edge but don't block further traversal from other paths.
                    logger.debug("Budget conversion failed for edge to %s", target)

        return results

    # -- Reset -------------------------------------------------------------

    def reset(self) -> None:
        """Clear all verification results, restoring every node to UNKNOWN."""
        for node in self._nodes.values():
            node.reset()

    # -- Visualisation -----------------------------------------------------

    def to_dot(self) -> str:
        """Generate a DOT (Graphviz) representation of the lattice.

        Nodes are colour-coded:

        * **green**: verified
        * **red**: falsified
        * **gray**: unknown

        Tight edges are drawn with a solid line; non-tight edges are
        dashed.

        Returns:
            A string in DOT format.
        """
        _color_map: dict[NodeStatus, str] = {
            NodeStatus.VERIFIED: "#4CAF50",   # green
            NodeStatus.FALSIFIED: "#F44336",  # red
            NodeStatus.UNKNOWN: "#9E9E9E",    # gray
        }

        _font_color: dict[NodeStatus, str] = {
            NodeStatus.VERIFIED: "white",
            NodeStatus.FALSIFIED: "white",
            NodeStatus.UNKNOWN: "black",
        }

        lines: list[str] = [
            "digraph DPLattice {",
            '    rankdir=TB;',
            '    node [shape=box, style="filled,rounded", fontname="Helvetica"];',
            '    edge [fontname="Helvetica", fontsize=9];',
            "",
        ]

        # Nodes
        for notion in PrivacyNotion:
            node = self._nodes[notion]
            color = _color_map[node.status]
            fc = _font_color[node.status]
            label = f"{notion.name}\\n({node.status})"
            if node.budget is not None:
                label += f"\\n{node.budget}"
            if node.derived:
                label += "\\n[derived]"
            lines.append(
                f'    {notion.name} [label="{label}", '
                f'fillcolor="{color}", fontcolor="{fc}"];'
            )

        lines.append("")

        # Edges
        for edge in self._edges:
            style = "solid" if edge.is_tight else "dashed"
            label = edge.theorem_ref
            lines.append(
                f'    {edge.source.name} -> {edge.target.name} '
                f'[style={style}, label="{label}"];'
            )

        lines.append("}")
        return "\n".join(lines)

    # -- String summary ----------------------------------------------------

    def __str__(self) -> str:
        """Return a multi-line text summary of the lattice state."""
        header = "DP Implication Lattice"
        sep = "=" * len(header)
        node_lines = [f"  {node}" for node in self.all_nodes]
        edge_lines = [f"  {edge}" for edge in self._edges]

        parts = [
            header,
            sep,
            "Nodes:",
            *node_lines,
            "",
            "Edges:",
            *edge_lines,
        ]
        return "\n".join(parts)

    def __repr__(self) -> str:
        verified = sum(1 for n in self._nodes.values() if n.is_verified)
        falsified = sum(1 for n in self._nodes.values() if n.is_falsified)
        unknown = sum(1 for n in self._nodes.values() if n.is_unknown)
        return (
            f"ImplicationLattice(verified={verified}, "
            f"falsified={falsified}, unknown={unknown})"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 6.  LATTICE TRAVERSAL HELPERS
# ═══════════════════════════════════════════════════════════════════════════


def topological_order(lattice: ImplicationLattice) -> list[PrivacyNotion]:
    """Return a topological ordering of the lattice (strongest first).

    Uses Kahn's algorithm.  The returned list starts with notions that
    have no incoming edges (strongest) and ends with the weakest notions.

    Args:
        lattice: The implication lattice to traverse.

    Returns:
        A list of :class:`PrivacyNotion` in topological order.
    """
    in_degree: dict[PrivacyNotion, int] = {n: 0 for n in PrivacyNotion}
    adjacency: dict[PrivacyNotion, list[PrivacyNotion]] = defaultdict(list)

    for edge in lattice.all_edges:
        adjacency[edge.source].append(edge.target)
        in_degree[edge.target] += 1

    queue: deque[PrivacyNotion] = deque(
        n for n in PrivacyNotion if in_degree[n] == 0
    )
    result: list[PrivacyNotion] = []

    while queue:
        current = queue.popleft()
        result.append(current)
        for successor in adjacency[current]:
            in_degree[successor] -= 1
            if in_degree[successor] == 0:
                queue.append(successor)

    if len(result) != len(PrivacyNotion):
        raise RuntimeError(
            "Cycle detected in implication lattice — this should be impossible. "
            f"Processed {len(result)} of {len(PrivacyNotion)} notions."
        )

    return result


def reachable_from(
    lattice: ImplicationLattice,
    notion: PrivacyNotion,
) -> set[PrivacyNotion]:
    """Return all notions reachable from *notion* via forward edges.

    This is the set of weaker notions implied by *notion*, not including
    *notion* itself.

    Args:
        lattice: The implication lattice to traverse.
        notion:  Starting notion.

    Returns:
        Set of reachable (weaker) notions.
    """
    visited: set[PrivacyNotion] = set()
    queue: deque[PrivacyNotion] = deque([notion])

    while queue:
        current = queue.popleft()
        for edge in lattice.get_edges_from(current):
            if edge.target not in visited:
                visited.add(edge.target)
                queue.append(edge.target)

    return visited


def reachable_to(
    lattice: ImplicationLattice,
    notion: PrivacyNotion,
) -> set[PrivacyNotion]:
    """Return all notions that can reach *notion* via forward edges.

    This is the set of stronger notions that imply *notion*, not
    including *notion* itself.

    Args:
        lattice: The implication lattice to traverse.
        notion:  Target notion.

    Returns:
        Set of reachable (stronger) notions.
    """
    visited: set[PrivacyNotion] = set()
    queue: deque[PrivacyNotion] = deque([notion])

    while queue:
        current = queue.popleft()
        for edge in lattice.get_edges_to(current):
            if edge.source not in visited:
                visited.add(edge.source)
                queue.append(edge.source)

    return visited


def strongest_verified(lattice: ImplicationLattice) -> PrivacyNotion | None:
    """Return the strongest verified notion, or ``None``.

    Strength is determined by the topological order: a notion is
    *stronger* if it has fewer incoming edges (appears earlier in
    the topological sort).

    Args:
        lattice: The implication lattice to inspect.

    Returns:
        The strongest verified :class:`PrivacyNotion`, or ``None``
        if no notion has been verified.
    """
    for notion in topological_order(lattice):
        node = lattice.get_node(notion)
        if node.is_verified and not node.derived:
            return notion

    # Fall back to any verified node (including derived).
    for notion in topological_order(lattice):
        if lattice.get_node(notion).is_verified:
            return notion

    return None


def weakest_falsified(lattice: ImplicationLattice) -> PrivacyNotion | None:
    """Return the weakest falsified notion, or ``None``.

    Weakness is determined by the reverse topological order: a notion
    is *weaker* if it has more incoming edges (appears later in the
    topological sort).

    Args:
        lattice: The implication lattice to inspect.

    Returns:
        The weakest falsified :class:`PrivacyNotion`, or ``None``
        if no notion has been falsified.
    """
    for notion in reversed(topological_order(lattice)):
        node = lattice.get_node(notion)
        if node.is_falsified and not node.derived:
            return notion

    # Fall back to any falsified node (including derived).
    for notion in reversed(topological_order(lattice)):
        if lattice.get_node(notion).is_falsified:
            return notion

    return None


def lattice_summary(lattice: ImplicationLattice) -> dict[str, Any]:
    """Return a machine-readable summary of the lattice state.

    Useful for serialisation, logging, or integration with the CEGAR
    driver.

    Args:
        lattice: The lattice to summarise.

    Returns:
        A dictionary with keys ``"nodes"``, ``"edges"``,
        ``"strongest_verified"``, ``"weakest_falsified"``, and
        ``"remaining"``.
    """
    nodes_info: list[dict[str, Any]] = []
    for node in lattice.all_nodes:
        entry: dict[str, Any] = {
            "notion": node.notion.name,
            "status": node.status.name,
            "derived": node.derived,
        }
        if node.derived_from is not None:
            entry["derived_from"] = node.derived_from.name
        if node.budget is not None:
            entry["budget"] = str(node.budget)
        nodes_info.append(entry)

    edges_info = [
        {
            "source": e.source.name,
            "target": e.target.name,
            "theorem": e.theorem_ref,
            "tight": e.is_tight,
        }
        for e in lattice.all_edges
    ]

    sv = strongest_verified(lattice)
    wf = weakest_falsified(lattice)

    return {
        "nodes": nodes_info,
        "edges": edges_info,
        "strongest_verified": sv.name if sv else None,
        "weakest_falsified": wf.name if wf else None,
        "remaining": [n.name for n in lattice.query_order()],
    }
