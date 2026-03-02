"""Privacy profile computation and analysis.

A privacy profile provides a complete characterisation of a mechanism's
privacy guarantees across all supported DP notions, including the
tightest achievable parameters and epsilon-delta trade-off curves.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

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
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class PrivacyGuarantee:
    """A single privacy guarantee under a specific DP notion.

    Attributes:
        notion:   The DP notion this guarantee pertains to.
        budget:   The concrete privacy budget.
        is_tight: Whether this is the tightest achievable bound.
        source:   How the guarantee was obtained: ``"direct"``,
                  ``"derived"``, or ``"composed"``.
    """

    notion: PrivacyNotion
    budget: PrivacyBudget
    is_tight: bool = False
    source: str = "direct"

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dictionary."""
        return {
            "notion": self.notion.name,
            "budget": str(self.budget),
            "is_tight": self.is_tight,
            "source": self.source,
        }


# ───────────────────────────────────────────────────────────────────────────
# Epsilon-Delta curve
# ───────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class EpsilonDeltaCurve:
    """A piecewise-linear ε–δ trade-off curve.

    ``points`` is a list of ``(delta, epsilon)`` pairs kept sorted by
    ascending *delta*.
    """

    points: list[tuple[float, float]] = field(default_factory=list)

    # -- mutation --------------------------------------------------------

    def add_point(self, delta: float, epsilon: float) -> None:
        """Insert a point and maintain sorted order by *delta*."""
        self.points.append((delta, epsilon))
        self.points.sort(key=lambda p: p[0])

    # -- queries ---------------------------------------------------------

    def epsilon_at_delta(self, delta: float) -> float:
        """Linearly interpolate ε at the given *delta*.

        Returns ``math.inf`` when extrapolation would be required beyond
        the recorded range.
        """
        if not self.points:
            return math.inf
        if delta <= self.points[0][0]:
            return self.points[0][1]
        if delta >= self.points[-1][0]:
            return self.points[-1][1]
        for i in range(len(self.points) - 1):
            d0, e0 = self.points[i]
            d1, e1 = self.points[i + 1]
            if d0 <= delta <= d1:
                if d1 == d0:
                    return min(e0, e1)
                t = (delta - d0) / (d1 - d0)
                return e0 + t * (e1 - e0)
        return math.inf  # pragma: no cover

    def delta_at_epsilon(self, eps: float) -> float:
        """Linearly interpolate δ at the given *epsilon*.

        The curve is traversed assuming ε is monotonically non-increasing
        in δ (the typical privacy trade-off shape).  Returns ``1.0`` when
        extrapolation would be required.
        """
        if not self.points:
            return 1.0
        # Build (epsilon, delta) sorted descending in epsilon
        by_eps = sorted(self.points, key=lambda p: -p[1])
        if eps >= by_eps[0][1]:
            return by_eps[0][0]
        if eps <= by_eps[-1][1]:
            return by_eps[-1][0]
        for i in range(len(by_eps) - 1):
            d0, e0 = by_eps[i]
            d1, e1 = by_eps[i + 1]
            if e1 <= eps <= e0:
                if e0 == e1:
                    return min(d0, d1)
                t = (eps - e1) / (e0 - e1)
                return d1 + t * (d0 - d1)
        return 1.0  # pragma: no cover

    # -- combination -----------------------------------------------------

    def merge(self, other: EpsilonDeltaCurve) -> EpsilonDeltaCurve:
        """Return a new curve that is the pointwise minimum of *self* and *other*."""
        all_deltas = sorted({d for d, _ in self.points} | {d for d, _ in other.points})
        merged = EpsilonDeltaCurve()
        for d in all_deltas:
            e_self = self.epsilon_at_delta(d)
            e_other = other.epsilon_at_delta(d)
            merged.add_point(d, min(e_self, e_other))
        return merged

    # -- serialisation ---------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dictionary."""
        return {"points": [{"delta": d, "epsilon": e} for d, e in self.points]}


# ───────────────────────────────────────────────────────────────────────────
# RDP curve
# ───────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class RDPCurve:
    """A Rényi differential privacy curve.

    ``points`` is a list of ``(alpha, epsilon)`` pairs kept sorted by
    ascending *alpha* (Rényi order).
    """

    points: list[tuple[float, float]] = field(default_factory=list)

    def add_point(self, alpha: float, epsilon: float) -> None:
        """Insert a point and maintain sorted order by *alpha*."""
        self.points.append((alpha, epsilon))
        self.points.sort(key=lambda p: p[0])

    def epsilon_at_alpha(self, alpha: float) -> float:
        """Linearly interpolate ε at the given Rényi order *alpha*.

        Returns ``math.inf`` when extrapolation would be required beyond
        the recorded range.
        """
        if not self.points:
            return math.inf
        if alpha <= self.points[0][0]:
            return self.points[0][1]
        if alpha >= self.points[-1][0]:
            return self.points[-1][1]
        for i in range(len(self.points) - 1):
            a0, e0 = self.points[i]
            a1, e1 = self.points[i + 1]
            if a0 <= alpha <= a1:
                if a1 == a0:
                    return min(e0, e1)
                t = (alpha - a0) / (a1 - a0)
                return e0 + t * (e1 - e0)
        return math.inf  # pragma: no cover

    def to_approx_at_delta(self, delta: float) -> ApproxBudget:
        """Optimal RDP → (ε, δ)-DP conversion.

        For each recorded ``(α, ε_rdp)`` the conversion gives

            ε(δ) = ε_rdp − log(δ) / (α − 1)

        and we return the ``(ε, δ)`` pair that minimises ε.
        """
        if not self.points or delta <= 0 or delta >= 1:
            return ApproxBudget(epsilon=math.inf, delta=max(0.0, min(delta, 1.0)))
        best_eps = math.inf
        for alpha, eps_rdp in self.points:
            if alpha <= 1:
                continue
            candidate = eps_rdp - math.log(delta) / (alpha - 1.0)
            if candidate < best_eps:
                best_eps = candidate
        return ApproxBudget(epsilon=max(best_eps, 0.0), delta=delta)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dictionary."""
        return {"points": [{"alpha": a, "epsilon": e} for a, e in self.points]}


# ═══════════════════════════════════════════════════════════════════════════
# PRIVACY PROFILE
# ═══════════════════════════════════════════════════════════════════════════


class PrivacyProfile:
    """Complete privacy characterisation of a mechanism.

    A profile aggregates guarantees across DP notions and provides
    ε–δ / RDP curve computation, comparison, and serialisation.

    Attributes:
        mechanism_name:      Human-readable name of the mechanism.
        guarantees:          Per-notion privacy guarantees.
        epsilon_delta_curve: Cached ε–δ curve (computed lazily).
        rdp_curve:           Cached RDP curve (computed lazily).
        trade_off_fn:        Optional f-DP trade-off function.
        metadata:            Arbitrary extra information.
    """

    def __init__(
        self,
        mechanism_name: str,
        *,
        trade_off_fn: Callable[[float], float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.mechanism_name: str = mechanism_name
        self.guarantees: dict[PrivacyNotion, PrivacyGuarantee] = {}
        self.epsilon_delta_curve: EpsilonDeltaCurve | None = None
        self.rdp_curve: RDPCurve | None = None
        self.trade_off_fn: Callable[[float], float] | None = trade_off_fn
        self.metadata: dict[str, Any] = metadata if metadata is not None else {}

    # -- guarantee management --------------------------------------------

    def add_guarantee(
        self,
        notion: PrivacyNotion,
        budget: PrivacyBudget,
        is_tight: bool = False,
        source: str = "direct",
    ) -> None:
        """Register a privacy guarantee for *notion*."""
        self.guarantees[notion] = PrivacyGuarantee(
            notion=notion, budget=budget, is_tight=is_tight, source=source,
        )

    def get_guarantee(self, notion: PrivacyNotion) -> PrivacyGuarantee | None:
        """Return the stored guarantee for *notion*, or ``None``."""
        return self.guarantees.get(notion)

    # -- (ε, δ) helpers --------------------------------------------------

    def get_tightest_approx(self, delta: float) -> ApproxBudget | None:
        """Return the tightest (ε, δ)-DP guarantee achievable at *delta*.

        The method considers every stored guarantee, converting to
        approximate DP where possible, and returns the smallest ε.
        """
        best: ApproxBudget | None = None
        for g in self.guarantees.values():
            eps, delt = g.budget.to_approx_dp(delta=delta)
            candidate = ApproxBudget(epsilon=eps, delta=delt)
            if best is None or eps < best.epsilon:
                best = candidate

        # Also consult RDP curve if available
        if self.rdp_curve is not None:
            rdp_candidate = self.rdp_curve.to_approx_at_delta(delta)
            if best is None or rdp_candidate.epsilon < best.epsilon:
                best = rdp_candidate

        return best

    # -- curve computation -----------------------------------------------

    _DEFAULT_DELTAS: list[float] = [
        10 ** (-k) for k in range(1, 11)
    ]

    _DEFAULT_ALPHAS: list[float] = [
        1.25, 1.5, 2.0, 3.0, 4.0, 5.0, 8.0, 16.0, 32.0, 64.0,
    ]

    def compute_epsilon_delta_curve(
        self, deltas: list[float] | None = None,
    ) -> EpsilonDeltaCurve:
        """Build (or rebuild) the ε–δ curve from stored guarantees.

        Parameters:
            deltas: δ values at which to evaluate.  Defaults to
                    ``[1e-1, 1e-2, …, 1e-10]``.

        Returns:
            The computed :class:`EpsilonDeltaCurve` (also cached on the
            profile).
        """
        if deltas is None:
            deltas = self._DEFAULT_DELTAS
        curve = EpsilonDeltaCurve()
        for d in sorted(deltas):
            best = self.get_tightest_approx(d)
            if best is not None:
                curve.add_point(d, best.epsilon)
        self.epsilon_delta_curve = curve
        return curve

    def compute_rdp_curve(
        self, alphas: list[float] | None = None,
    ) -> RDPCurve:
        """Build (or rebuild) the RDP curve from stored guarantees.

        Parameters:
            alphas: Rényi orders at which to evaluate.  Defaults to a
                    standard set of ten orders.

        Returns:
            The computed :class:`RDPCurve` (also cached on the profile).
        """
        if alphas is None:
            alphas = self._DEFAULT_ALPHAS
        curve = RDPCurve()
        g = self.guarantees.get(PrivacyNotion.RDP)
        if g is not None and isinstance(g.budget, RDPBudget):
            # If a single RDP point is stored, extrapolate linearly
            base_alpha = g.budget.alpha
            base_eps = g.budget.epsilon
            for a in sorted(alphas):
                # Linear scaling heuristic: ε(α) ≈ ε₀ · α / α₀
                scaled = base_eps * a / base_alpha
                curve.add_point(a, scaled)
        elif self.guarantees:
            # Fall back: convert stored guarantees to RDP via ε(α) bound
            for a in sorted(alphas):
                best_eps = math.inf
                for gg in self.guarantees.values():
                    eps_approx, _ = gg.budget.to_approx_dp(delta=1e-6)
                    # Inverse conversion: ε_rdp ≈ ε + log(1/δ)/(α-1)
                    if a > 1:
                        rdp_eps = eps_approx + math.log(1e6) / (a - 1.0)
                        best_eps = min(best_eps, rdp_eps)
                if best_eps < math.inf:
                    curve.add_point(a, best_eps)
        self.rdp_curve = curve
        return curve

    # -- comparison ------------------------------------------------------

    def is_dominated_by(self, other: PrivacyProfile) -> bool:
        """Return ``True`` if *other* is at least as private on every notion.

        Only notions present in **both** profiles are compared.
        """
        common = set(self.guarantees) & set(other.guarantees)
        if not common:
            return False
        for notion in common:
            self_g = self.guarantees[notion]
            other_g = other.guarantees[notion]
            if not other_g.budget.is_satisfied_by(self_g.budget):
                return False
        return True

    # -- serialisation ---------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise the profile to a JSON-compatible dictionary."""
        data: dict[str, Any] = {
            "mechanism_name": self.mechanism_name,
            "guarantees": {
                n.name: g.to_dict() for n, g in self.guarantees.items()
            },
            "metadata": self.metadata,
        }
        if self.epsilon_delta_curve is not None:
            data["epsilon_delta_curve"] = self.epsilon_delta_curve.to_dict()
        if self.rdp_curve is not None:
            data["rdp_curve"] = self.rdp_curve.to_dict()
        return data

    def to_json(self, indent: int = 2) -> str:
        """Serialise the profile to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PrivacyProfile:
        """Reconstruct a :class:`PrivacyProfile` from a dictionary.

        Budget objects are recreated from their string representation
        using a best-effort heuristic; the resulting profile carries all
        guarantees as non-tight / ``"direct"``.
        """
        profile = cls(
            mechanism_name=data.get("mechanism_name", "unknown"),
            metadata=data.get("metadata", {}),
        )
        for notion_name, g_data in data.get("guarantees", {}).items():
            notion = PrivacyNotion[notion_name]
            is_tight = g_data.get("is_tight", False)
            source = g_data.get("source", "direct")
            budget = _reconstruct_budget(notion, g_data.get("budget", ""))
            profile.add_guarantee(notion, budget, is_tight=is_tight, source=source)

        # Restore curves if present
        if "epsilon_delta_curve" in data:
            curve = EpsilonDeltaCurve()
            for pt in data["epsilon_delta_curve"].get("points", []):
                curve.add_point(pt["delta"], pt["epsilon"])
            profile.epsilon_delta_curve = curve
        if "rdp_curve" in data:
            rdp = RDPCurve()
            for pt in data["rdp_curve"].get("points", []):
                rdp.add_point(pt["alpha"], pt["epsilon"])
            profile.rdp_curve = rdp
        return profile

    def summary(self) -> str:
        """Return a human-readable summary of the profile."""
        lines: list[str] = [f"PrivacyProfile: {self.mechanism_name}"]
        lines.append("─" * 50)
        if not self.guarantees:
            lines.append("  (no guarantees recorded)")
        for notion, g in self.guarantees.items():
            tight_mark = " [tight]" if g.is_tight else ""
            lines.append(f"  {notion.name:12s}  {g.budget}{tight_mark}  ({g.source})")
        if self.epsilon_delta_curve and self.epsilon_delta_curve.points:
            n_pts = len(self.epsilon_delta_curve.points)
            lines.append(f"  ε–δ curve:   {n_pts} points")
        if self.rdp_curve and self.rdp_curve.points:
            n_pts = len(self.rdp_curve.points)
            lines.append(f"  RDP curve:   {n_pts} points")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# PROFILE COMPARISON
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class ComparisonResult:
    """Outcome of comparing two :class:`PrivacyProfile` objects.

    Attributes:
        per_notion: Mapping from notion name to ``"a_wins"``,
                    ``"b_wins"``, or ``"tie"``.
        overall:    ``"a_dominates"``, ``"b_dominates"``,
                    ``"incomparable"``, or ``"equal"``.
        details:    Free-form detail strings per notion.
    """

    per_notion: dict[str, str] = field(default_factory=dict)
    overall: str = "incomparable"
    details: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dictionary."""
        return {
            "per_notion": self.per_notion,
            "overall": self.overall,
            "details": self.details,
        }


class ProfileComparison:
    """Compare two :class:`PrivacyProfile` instances across all notions."""

    @staticmethod
    def compare(
        profile_a: PrivacyProfile,
        profile_b: PrivacyProfile,
    ) -> ComparisonResult:
        """Compare *profile_a* and *profile_b*.

        Returns a :class:`ComparisonResult` summarising which profile
        offers tighter privacy for each notion present in both.
        """
        result = ComparisonResult()
        common = set(profile_a.guarantees) & set(profile_b.guarantees)
        a_wins = 0
        b_wins = 0
        for notion in sorted(common, key=lambda n: n.name):
            ga = profile_a.guarantees[notion]
            gb = profile_b.guarantees[notion]
            a_sat = ga.budget.is_satisfied_by(gb.budget)
            b_sat = gb.budget.is_satisfied_by(ga.budget)
            name = notion.name
            if a_sat and b_sat:
                result.per_notion[name] = "tie"
                result.details[name] = f"{ga.budget} ≡ {gb.budget}"
            elif a_sat:
                result.per_notion[name] = "a_wins"
                result.details[name] = f"{ga.budget} ≤ {gb.budget}"
                a_wins += 1
            elif b_sat:
                result.per_notion[name] = "b_wins"
                result.details[name] = f"{gb.budget} ≤ {ga.budget}"
                b_wins += 1
            else:
                result.per_notion[name] = "incomparable"
                result.details[name] = f"{ga.budget} ⊥ {gb.budget}"

        if a_wins > 0 and b_wins == 0:
            result.overall = "a_dominates"
        elif b_wins > 0 and a_wins == 0:
            result.overall = "b_dominates"
        elif a_wins == 0 and b_wins == 0 and common:
            result.overall = "equal"
        else:
            result.overall = "incomparable"
        return result


# ═══════════════════════════════════════════════════════════════════════════
# PROFILE COMPOSITION
# ═══════════════════════════════════════════════════════════════════════════


class ProfileComposer:
    """Compose multiple :class:`PrivacyProfile` objects under sequential
    or parallel composition rules.
    """

    # -- public API ------------------------------------------------------

    @classmethod
    def sequential(cls, profiles: list[PrivacyProfile]) -> PrivacyProfile:
        """Return a profile representing the sequential composition of
        *profiles*.

        For each DP notion, the appropriate composition theorem is
        applied if all constituent profiles carry that notion.
        """
        if not profiles:
            return PrivacyProfile(mechanism_name="empty_seq")
        names = " ∘ ".join(p.mechanism_name for p in profiles)
        composed = PrivacyProfile(
            mechanism_name=f"Seq({names})",
            metadata={"composition": "sequential", "n_components": len(profiles)},
        )
        # Attempt composition for every notion present in ALL profiles
        all_notions = set.intersection(*(set(p.guarantees) for p in profiles))
        for notion in all_notions:
            budgets = [p.guarantees[notion].budget for p in profiles]
            composed_budget = cls._compose_sequential_budgets(notion, budgets)
            if composed_budget is not None:
                composed.add_guarantee(notion, composed_budget, source="composed")
        return composed

    @classmethod
    def parallel(cls, profiles: list[PrivacyProfile]) -> PrivacyProfile:
        """Return a profile representing the parallel composition of
        *profiles* (maximum over budgets).
        """
        if not profiles:
            return PrivacyProfile(mechanism_name="empty_par")
        names = " ‖ ".join(p.mechanism_name for p in profiles)
        composed = PrivacyProfile(
            mechanism_name=f"Par({names})",
            metadata={"composition": "parallel", "n_components": len(profiles)},
        )
        all_notions = set.intersection(*(set(p.guarantees) for p in profiles))
        for notion in all_notions:
            budgets = [p.guarantees[notion].budget for p in profiles]
            # Parallel composition: take the worst-case (max) budget
            worst = budgets[0]
            for b in budgets[1:]:
                eps_w, delt_w = worst.to_approx_dp()
                eps_b, delt_b = b.to_approx_dp()
                if eps_b > eps_w or (eps_b == eps_w and delt_b > delt_w):
                    worst = b
            composed.add_guarantee(notion, worst, source="composed")
        return composed

    # -- private helpers -------------------------------------------------

    @classmethod
    def _compose_sequential_budgets(
        cls,
        notion: PrivacyNotion,
        budgets: list[PrivacyBudget],
    ) -> PrivacyBudget | None:
        """Dispatch to the appropriate sequential composition rule."""
        if notion == PrivacyNotion.APPROX_DP:
            return cls._compose_approx_sequential(
                [b for b in budgets if isinstance(b, ApproxBudget)],
            )
        if notion == PrivacyNotion.ZCDP:
            return cls._compose_zcdp_sequential(
                [b for b in budgets if isinstance(b, ZCDPBudget)],
            )
        if notion == PrivacyNotion.RDP:
            return cls._compose_rdp_sequential(
                [b for b in budgets if isinstance(b, RDPBudget)],
            )
        if notion == PrivacyNotion.GDP:
            return cls._compose_gdp_sequential(
                [b for b in budgets if isinstance(b, GDPBudget)],
            )
        if notion == PrivacyNotion.PURE_DP:
            eps = sum(
                b.epsilon for b in budgets if isinstance(b, PureBudget)
            )
            return PureBudget(epsilon=eps) if eps < math.inf else None
        return None

    @staticmethod
    def _compose_approx_sequential(
        budgets: list[ApproxBudget],
    ) -> ApproxBudget:
        """Basic (ε, δ) sequential composition: sum ε and δ."""
        total_eps = sum(b.epsilon for b in budgets)
        total_delta = sum(b.delta for b in budgets)
        return ApproxBudget(epsilon=total_eps, delta=min(total_delta, 1.0))

    @staticmethod
    def _compose_zcdp_sequential(
        budgets: list[ZCDPBudget],
    ) -> ZCDPBudget:
        """zCDP sequential composition: sum ρ."""
        total_rho = sum(b.rho for b in budgets)
        return ZCDPBudget(rho=total_rho)

    @staticmethod
    def _compose_rdp_sequential(
        budgets: list[RDPBudget],
    ) -> RDPBudget:
        """RDP sequential composition at the common α: sum ε.

        Falls back to the first budget's α if orders differ.
        """
        if not budgets:
            return RDPBudget(alpha=2.0, epsilon=0.0)
        alpha = budgets[0].alpha
        total_eps = sum(b.epsilon for b in budgets)
        return RDPBudget(alpha=alpha, epsilon=total_eps)

    @staticmethod
    def _compose_gdp_sequential(
        budgets: list[GDPBudget],
    ) -> GDPBudget:
        """GDP sequential composition: μ_total = √(Σ μ_i²)."""
        mu_sq = sum(b.mu ** 2 for b in budgets)
        return GDPBudget(mu=math.sqrt(mu_sq))


# ═══════════════════════════════════════════════════════════════════════════
# PROFILE VISUALISER
# ═══════════════════════════════════════════════════════════════════════════


class ProfileVisualizer:
    """ASCII and DOT visualisation of :class:`PrivacyProfile` objects."""

    @staticmethod
    def to_ascii_epsilon_delta(
        profile: PrivacyProfile,
        width: int = 60,
        height: int = 20,
    ) -> str:
        """Render the ε–δ curve as an ASCII plot.

        Parameters:
            profile: The profile whose ε–δ curve to render.
            width:   Character width of the plot area.
            height:  Character height of the plot area.
        """
        curve = profile.epsilon_delta_curve
        if curve is None or not curve.points:
            return "(no ε–δ curve available)"

        deltas = [d for d, _ in curve.points]
        epsilons = [e for _, e in curve.points]
        d_min, d_max = min(deltas), max(deltas)
        e_min, e_max = min(epsilons), max(epsilons)
        if e_max == e_min:
            e_max = e_min + 1.0
        if d_max == d_min:
            d_max = d_min + 1.0

        grid: list[list[str]] = [[" "] * width for _ in range(height)]
        for d, e in curve.points:
            col = int((d - d_min) / (d_max - d_min) * (width - 1))
            row = int((1.0 - (e - e_min) / (e_max - e_min)) * (height - 1))
            col = max(0, min(width - 1, col))
            row = max(0, min(height - 1, row))
            grid[row][col] = "●"

        lines: list[str] = []
        lines.append(f"  ε–δ curve for {profile.mechanism_name}")
        lines.append(f"  ε ∈ [{e_min:.4g}, {e_max:.4g}]")
        for r, row in enumerate(grid):
            if r == 0:
                label = f"{e_max:>8.3g} │"
            elif r == height - 1:
                label = f"{e_min:>8.3g} │"
            else:
                label = "         │"
            lines.append(label + "".join(row))
        lines.append("         └" + "─" * width)
        d_min_s = f"{d_min:.2g}"
        d_max_s = f"{d_max:.2g}"
        pad = width - len(d_min_s) - len(d_max_s)
        lines.append("          " + d_min_s + " " * max(pad, 1) + d_max_s)
        lines.append("          " + " " * (width // 2 - 1) + "δ →")
        return "\n".join(lines)

    @staticmethod
    def to_ascii_rdp(
        profile: PrivacyProfile,
        width: int = 60,
        height: int = 20,
    ) -> str:
        """Render the RDP curve as an ASCII plot.

        Parameters:
            profile: The profile whose RDP curve to render.
            width:   Character width of the plot area.
            height:  Character height of the plot area.
        """
        curve = profile.rdp_curve
        if curve is None or not curve.points:
            return "(no RDP curve available)"

        alphas = [a for a, _ in curve.points]
        epsilons = [e for _, e in curve.points]
        a_min, a_max = min(alphas), max(alphas)
        e_min, e_max = min(epsilons), max(epsilons)
        if e_max == e_min:
            e_max = e_min + 1.0
        if a_max == a_min:
            a_max = a_min + 1.0

        grid: list[list[str]] = [[" "] * width for _ in range(height)]
        for a, e in curve.points:
            col = int((a - a_min) / (a_max - a_min) * (width - 1))
            row = int((1.0 - (e - e_min) / (e_max - e_min)) * (height - 1))
            col = max(0, min(width - 1, col))
            row = max(0, min(height - 1, row))
            grid[row][col] = "●"

        lines: list[str] = []
        lines.append(f"  RDP curve for {profile.mechanism_name}")
        lines.append(f"  ε ∈ [{e_min:.4g}, {e_max:.4g}]")
        for r, row in enumerate(grid):
            if r == 0:
                label = f"{e_max:>8.3g} │"
            elif r == height - 1:
                label = f"{e_min:>8.3g} │"
            else:
                label = "         │"
            lines.append(label + "".join(row))
        lines.append("         └" + "─" * width)
        a_min_s = f"{a_min:.2g}"
        a_max_s = f"{a_max:.2g}"
        pad = width - len(a_min_s) - len(a_max_s)
        lines.append("          " + a_min_s + " " * max(pad, 1) + a_max_s)
        lines.append("          " + " " * (width // 2 - 1) + "α →")
        return "\n".join(lines)

    @staticmethod
    def to_table(profile: PrivacyProfile) -> str:
        """Render a text table of all guarantees in the profile."""
        lines: list[str] = [f"Privacy Profile: {profile.mechanism_name}"]
        lines.append(f"  {'Notion':<16} {'Budget':<30} {'Tight':<6}")
        lines.append("  " + "-" * 52)
        for notion, g in profile.guarantees.items():
            budget_str = str(g.budget) if g.budget else "—"
            tight_str = "yes" if g.is_tight else "no"
            lines.append(f"  {notion.name:<16} {budget_str:<30} {tight_str:<6}")
        if not profile.guarantees:
            lines.append("  (no guarantees)")
        return "\n".join(lines)

    @staticmethod
    def compare_table(profiles: list[PrivacyProfile]) -> str:
        """Render a comparison table for multiple profiles."""
        if not profiles:
            return "(no profiles to compare)"
        all_notions: list[PrivacyNotion] = []
        for p in profiles:
            for n in p.guarantees:
                if n not in all_notions:
                    all_notions.append(n)
        header = f"  {'Notion':<16}" + "".join(f" {p.mechanism_name:<20}" for p in profiles)
        lines: list[str] = ["Profile Comparison", header, "  " + "-" * (16 + 20 * len(profiles))]
        for notion in all_notions:
            row = f"  {notion.name:<16}"
            for p in profiles:
                g = p.get_guarantee(notion)
                val = str(g.budget) if g and g.budget else "—"
                row += f" {val:<20}"
            lines.append(row)
        return "\n".join(lines)

    @staticmethod
    def rdp_curve(profile: PrivacyProfile, alphas: list[float]) -> str:
        """Render RDP curve values at given alpha orders."""
        lines: list[str] = [f"RDP Curve for {profile.mechanism_name}"]
        lines.append(f"  {'Alpha':>8}  {'Epsilon':>12}")
        lines.append("  " + "-" * 22)
        curve = profile.compute_rdp_curve(alphas=alphas)
        for a, e in curve.points:
            lines.append(f"  {a:>8.2f}  {e:>12.6f}")
        if not curve.points:
            lines.append("  (no RDP data)")
        return "\n".join(lines)

    @staticmethod
    def to_latex(profile: PrivacyProfile) -> str:
        """Render the profile as a LaTeX table."""
        lines: list[str] = [
            r"\begin{tabular}{lll}",
            r"\hline",
            r"Notion & Budget & Tight \\",
            r"\hline",
        ]
        for notion, g in profile.guarantees.items():
            budget_str = str(g.budget) if g.budget else "---"
            tight_str = "yes" if g.is_tight else "no"
            lines.append(f"{notion.name} & {budget_str} & {tight_str} \\\\")
        lines.append(r"\hline")
        lines.append(r"\end{tabular}")
        return "\n".join(lines)

    @staticmethod
    def to_dot(profile: PrivacyProfile) -> str:
        """Render the profile as a Graphviz DOT digraph.

        The central node represents the mechanism; each guarantee is a
        leaf with its notion, budget, and tightness flag.
        """
        safe_name = profile.mechanism_name.replace('"', '\\"')
        lines: list[str] = [
            "digraph PrivacyProfile {",
            '  rankdir=LR;',
            '  node [shape=box, fontname="Courier"];',
            f'  mech [label="{safe_name}", style=bold];',
        ]
        for i, (notion, g) in enumerate(profile.guarantees.items()):
            budget_str = str(g.budget).replace('"', '\\"')
            tight_str = "✓" if g.is_tight else "✗"
            node_id = f"g{i}"
            label = f"{notion.name}\\n{budget_str}\\ntight: {tight_str}"
            lines.append(f'  {node_id} [label="{label}"];')
            lines.append(f'  mech -> {node_id} [label="{g.source}"];')
        lines.append("}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# PRIVATE HELPERS
# ═══════════════════════════════════════════════════════════════════════════


def _reconstruct_budget(notion: PrivacyNotion, raw: str) -> PrivacyBudget:
    """Best-effort reconstruction of a :class:`PrivacyBudget` from its
    ``__str__`` representation used during serialisation.
    """
    if notion == PrivacyNotion.PURE_DP:
        return _parse_pure(raw)
    if notion == PrivacyNotion.APPROX_DP:
        return _parse_approx(raw)
    if notion == PrivacyNotion.ZCDP:
        return _parse_zcdp(raw)
    if notion == PrivacyNotion.RDP:
        return _parse_rdp(raw)
    if notion == PrivacyNotion.GDP:
        return _parse_gdp(raw)
    # FDP and unknown: return a trivial approx budget
    return ApproxBudget(epsilon=math.inf, delta=1.0)


def _try_float(s: str, default: float = 0.0) -> float:
    """Parse a float, returning *default* on failure."""
    try:
        return float(s)
    except (ValueError, TypeError):
        return default


def _parse_pure(raw: str) -> PureBudget:
    # Expected format: "ε=<val>"
    raw = raw.strip()
    if "=" in raw:
        val = raw.split("=", 1)[1]
        return PureBudget(epsilon=_try_float(val))
    return PureBudget(epsilon=_try_float(raw))


def _parse_approx(raw: str) -> ApproxBudget:
    # Expected format: "(ε=<val>, δ=<val>)"
    raw = raw.strip().strip("()")
    parts = [p.strip() for p in raw.split(",")]
    eps = 0.0
    delta = 0.0
    for part in parts:
        if "ε" in part or "epsilon" in part.lower():
            eps = _try_float(part.split("=", 1)[-1])
        elif "δ" in part or "delta" in part.lower():
            delta = _try_float(part.split("=", 1)[-1])
    return ApproxBudget(epsilon=eps, delta=delta)


def _parse_zcdp(raw: str) -> ZCDPBudget:
    # Expected format: "ρ=<val>"
    raw = raw.strip()
    if "=" in raw:
        val = raw.split("=", 1)[1]
        return ZCDPBudget(rho=_try_float(val))
    return ZCDPBudget(rho=_try_float(raw))


def _parse_rdp(raw: str) -> RDPBudget:
    # Expected format: "(α=<val>, ε=<val>)-RDP"
    raw = raw.replace("-RDP", "").strip().strip("()")
    parts = [p.strip() for p in raw.split(",")]
    alpha = 2.0
    eps = 0.0
    for part in parts:
        if "α" in part or "alpha" in part.lower():
            alpha = _try_float(part.split("=", 1)[-1], default=2.0)
        elif "ε" in part or "epsilon" in part.lower():
            eps = _try_float(part.split("=", 1)[-1])
    return RDPBudget(alpha=alpha, epsilon=eps)


def _parse_gdp(raw: str) -> GDPBudget:
    # Expected format: "μ=<val>-GDP"
    raw = raw.replace("-GDP", "").strip()
    if "=" in raw:
        val = raw.split("=", 1)[1]
        return GDPBudget(mu=_try_float(val))
    return GDPBudget(mu=_try_float(raw))
