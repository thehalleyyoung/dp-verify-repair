"""Density ratio construction for differential privacy verification.

The :class:`DensityRatioBuilder` takes a :class:`PathSet` produced by
the path enumerator and constructs the log density ratio expression

    L(o) = ln(p(o | d) / p(o | d'))

for each path.  This is the privacy loss random variable whose worst-case
value must be bounded for the mechanism to satisfy differential privacy.

For a single noise draw from Laplace(b) the ratio is:
    (|o - f(d')| - |o - f(d)|) / b

For Gaussian(σ) the ratio is:
    ((o - f(d'))² - (o - f(d))²) / (2σ²)

For composed mechanisms with multiple independent noise draws, the
total log ratio is the sum of individual log ratios.

When data-dependent branching creates different paths for d and d',
the builder enumerates cross-path pairs (πᵢ, πⱼ) and constructs
the ratio for each pair.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dpcegar.ir.types import (
    Abs,
    BinOp,
    BinOpKind,
    Const,
    IRType,
    Log,
    NoiseKind,
    TypedExpr,
    Var,
)
from dpcegar.paths.symbolic_path import (
    NoiseDrawInfo,
    PathCondition,
    PathSet,
    SymbolicPath,
)
from dpcegar.density.noise_models import (
    NoiseModel,
    LaplaceNoise,
    GaussianNoise,
    ExponentialMechNoise,
    get_noise_model,
)


# ═══════════════════════════════════════════════════════════════════════════
# DENSITY RATIO EXPRESSION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class DensityRatioExpr:
    """A single density ratio expression for a path or path pair.

    Attributes:
        log_ratio:        Symbolic expression for the log density ratio L(o).
        path_condition_d: Path condition under dataset d.
        path_condition_d_prime: Path condition under dataset d'.
        path_id_d:        Path ID for dataset d.
        path_id_d_prime:  Path ID for dataset d'.
        noise_sites:      Noise draw site IDs involved.
        is_cross_path:    True if d and d' follow different paths.
        metadata:         Additional metadata.
    """

    log_ratio: TypedExpr
    path_condition_d: PathCondition
    path_condition_d_prime: PathCondition
    path_id_d: int
    path_id_d_prime: int
    noise_sites: list[int] = field(default_factory=list)
    is_cross_path: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def pretty(self) -> str:
        """Return a human-readable representation."""
        kind = "cross-path" if self.is_cross_path else "same-path"
        return (
            f"DensityRatio({kind})\n"
            f"  d-path:  #{self.path_id_d}\n"
            f"  d'-path: #{self.path_id_d_prime}\n"
            f"  L(o) = {self.log_ratio}\n"
            f"  noise sites: {self.noise_sites}"
        )

    def __str__(self) -> str:
        return f"DensityRatio(L={self.log_ratio}, cross={self.is_cross_path})"


@dataclass(slots=True)
class DensityRatioResult:
    """Collection of density ratio expressions for a mechanism.

    Attributes:
        ratios:         All density ratio expressions.
        same_path:      Ratios where d and d' follow the same path.
        cross_path:     Ratios where d and d' follow different paths.
        metadata:       Computation metadata and statistics.
    """

    ratios: list[DensityRatioExpr] = field(default_factory=list)
    same_path: list[DensityRatioExpr] = field(default_factory=list)
    cross_path: list[DensityRatioExpr] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def all_log_ratios(self) -> list[TypedExpr]:
        """Return all log-ratio expressions."""
        return [r.log_ratio for r in self.ratios]

    def summary(self) -> dict[str, int]:
        """Return summary statistics."""
        return {
            "total_ratios": len(self.ratios),
            "same_path_ratios": len(self.same_path),
            "cross_path_ratios": len(self.cross_path),
        }


# ═══════════════════════════════════════════════════════════════════════════
# SENSITIVITY TEMPLATE
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class SensitivityTemplate:
    """Parameterised density ratio template for a given noise kind.

    Instead of building a concrete ratio, this captures the ratio as a
    function of the sensitivity parameter Δ, enabling sensitivity
    inference during CEGAR refinement.

    Attributes:
        kind:             Noise distribution family.
        ratio_template:   Symbolic expression parameterised by ``delta``.
        sensitivity_var:  Variable representing the sensitivity.
        scale_var:        Variable representing the noise scale.
    """

    kind: NoiseKind
    ratio_template: TypedExpr
    sensitivity_var: Var
    scale_var: Var

    def instantiate(self, sensitivity: TypedExpr, scale: TypedExpr) -> TypedExpr:
        """Substitute concrete sensitivity and scale into the template."""
        mapping = {
            self.sensitivity_var.name: sensitivity,
            self.scale_var.name: scale,
        }
        return self.ratio_template.substitute(mapping)


# ═══════════════════════════════════════════════════════════════════════════
# DENSITY RATIO BUILDER
# ═══════════════════════════════════════════════════════════════════════════


class DensityRatioBuilder:
    """Build log density ratio expressions from enumerated paths.

    For each pair of paths (one for dataset d, one for d'), constructs
    the symbolic privacy loss expression L(o) = ln(p(o|d)/p(o|d')).

    Usage::

        builder = DensityRatioBuilder()
        result = builder.build(path_set, sensitivity_map)
        for ratio in result.ratios:
            print(ratio.pretty())

    Args:
        d_suffix:       Suffix for dataset-d variables (e.g. ``""``).
        d_prime_suffix: Suffix for dataset-d' variables (e.g. ``"_prime"``).
    """

    def __init__(
        self,
        d_suffix: str = "",
        d_prime_suffix: str = "_prime",
    ) -> None:
        self._d_suffix = d_suffix
        self._dp_suffix = d_prime_suffix
        self._noise_models: dict[NoiseKind, NoiseModel] = {
            NoiseKind.LAPLACE: LaplaceNoise(),
            NoiseKind.GAUSSIAN: GaussianNoise(),
            NoiseKind.EXPONENTIAL: ExponentialMechNoise(),
        }
        self._center_cache: dict[tuple, TypedExpr] = {}

    # -- Public API --------------------------------------------------------

    def build(
        self,
        path_set: PathSet,
        sensitivity_map: dict[int, TypedExpr] | None = None,
    ) -> DensityRatioResult:
        """Build density ratio expressions for all path pairs.

        Args:
            path_set:        The enumerated paths.
            sensitivity_map: Optional mapping from noise-draw site ID
                             to the sensitivity expression for that query.

        Returns:
            A :class:`DensityRatioResult` with all ratio expressions.
        """
        sens_map = sensitivity_map or {}
        result = DensityRatioResult()

        paths = path_set.paths
        for path in paths:
            same_ratio = self._build_same_path_ratio(path, sens_map)
            if same_ratio is not None:
                result.ratios.append(same_ratio)
                result.same_path.append(same_ratio)

        if len(paths) > 1:
            cross_ratios = self._build_cross_path_ratios(paths, sens_map)
            result.ratios.extend(cross_ratios)
            result.cross_path.extend(cross_ratios)

        result.metadata = {
            "total_paths": len(paths),
            "same_path_count": len(result.same_path),
            "cross_path_count": len(result.cross_path),
        }
        return result

    def build_single_ratio(
        self,
        noise_draw: NoiseDrawInfo,
        obs_var: TypedExpr,
        center_d: TypedExpr,
        center_d_prime: TypedExpr,
        scale: TypedExpr,
    ) -> TypedExpr:
        """Build the log density ratio for a single noise draw.

        Args:
            noise_draw:     The noise draw info.
            obs_var:        Symbolic observation variable.
            center_d:       Centre expression for dataset d.
            center_d_prime: Centre expression for dataset d'.
            scale:          Scale expression.

        Returns:
            Symbolic log density ratio expression.
        """
        model = self._noise_models.get(noise_draw.kind)
        if model is None:
            model = get_noise_model(noise_draw.kind)
        return model.symbolic_log_ratio(obs_var, center_d, center_d_prime, scale)

    def build_composed_ratio(
        self,
        noise_draws: list[NoiseDrawInfo],
        obs_vars: dict[str, TypedExpr],
        center_d_map: dict[str, TypedExpr],
        center_d_prime_map: dict[str, TypedExpr],
    ) -> TypedExpr:
        """Build the composed log density ratio for multiple independent draws.

        For independent noise draws, the total log ratio is the sum:
            L(o) = Σᵢ ln(pᵢ(oᵢ|d) / pᵢ(oᵢ|d'))

        Args:
            noise_draws:         List of noise draw infos.
            obs_vars:            Map from noise variable name to obs expression.
            center_d_map:        Map from noise variable name to d-centre.
            center_d_prime_map:  Map from noise variable name to d'-centre.

        Returns:
            The composed log density ratio expression.
        """
        if not noise_draws:
            return Const.real(0.0)

        terms: list[TypedExpr] = []
        for nd in noise_draws:
            obs = obs_vars.get(nd.variable, Var(ty=IRType.REAL, name=nd.variable))
            c_d = center_d_map.get(nd.variable, nd.center_expr)
            c_dp = center_d_prime_map.get(nd.variable, nd.center_expr)
            ratio = self.build_single_ratio(nd, obs, c_d, c_dp, nd.scale_expr)
            terms.append(ratio)

        result = terms[0]
        for t in terms[1:]:
            result = BinOp(ty=IRType.REAL, op=BinOpKind.ADD, left=result, right=t)
        return result

    def make_sensitivity_template(self, kind: NoiseKind) -> SensitivityTemplate:
        """Create a sensitivity-parameterised ratio template.

        The template uses symbolic variables for sensitivity (Δ) and
        scale, allowing the CEGAR loop to instantiate them later.

        Args:
            kind: The noise distribution family.

        Returns:
            A :class:`SensitivityTemplate`.
        """
        delta_var = Var(ty=IRType.REAL, name="__delta")
        scale_var = Var(ty=IRType.REAL, name="__scale")
        obs_var = Var(ty=IRType.REAL, name="__obs")
        center_d = Var(ty=IRType.REAL, name="__center_d")
        center_dp = BinOp(
            ty=IRType.REAL, op=BinOpKind.ADD,
            left=center_d, right=delta_var,
        )

        model = self._noise_models.get(kind)
        if model is None:
            model = get_noise_model(kind)

        template = model.symbolic_log_ratio(obs_var, center_d, center_dp, scale_var)
        return SensitivityTemplate(
            kind=kind,
            ratio_template=template,
            sensitivity_var=delta_var,
            scale_var=scale_var,
        )

    # -- Internal ----------------------------------------------------------

    def _build_same_path_ratio(
        self,
        path: SymbolicPath,
        sens_map: dict[int, TypedExpr],
    ) -> DensityRatioExpr | None:
        """Build the density ratio for d and d' both following *path*.

        The output expressions differ only in the query results, so
        the ratio captures how noise draws produce different densities
        under neighbouring datasets.
        """
        if not path.noise_draws:
            return None

        obs_vars: dict[str, TypedExpr] = {}
        center_d: dict[str, TypedExpr] = {}
        center_dp: dict[str, TypedExpr] = {}
        noise_sites: list[int] = []

        for nd in path.noise_draws:
            obs_name = f"o_{nd.variable}"
            obs_var = Var(ty=IRType.REAL, name=obs_name)
            obs_vars[nd.variable] = obs_var
            center_d[nd.variable] = nd.center_expr
            center_dp[nd.variable] = self._make_d_prime_center(nd, sens_map)
            noise_sites.append(nd.site_id)

        log_ratio = self.build_composed_ratio(
            path.noise_draws, obs_vars, center_d, center_dp
        )

        return DensityRatioExpr(
            log_ratio=log_ratio,
            path_condition_d=path.path_condition,
            path_condition_d_prime=path.path_condition,
            path_id_d=path.path_id,
            path_id_d_prime=path.path_id,
            noise_sites=noise_sites,
            is_cross_path=False,
        )

    def _build_cross_path_ratios(
        self,
        paths: list[SymbolicPath],
        sens_map: dict[int, TypedExpr],
    ) -> list[DensityRatioExpr]:
        """Build density ratios for all cross-path pairs.

        When data-dependent branching causes d and d' to follow different
        paths π_i and π_j, the density ratio accounts for different
        output expressions and potentially different noise draws.

        Paths with identical noise draw signatures are pruned since their
        cross-path ratios are structurally equivalent to same-path ratios.
        """
        ratios: list[DensityRatioExpr] = []

        # Group paths by noise draw signature for pruning
        def _noise_sig(path: SymbolicPath) -> frozenset:
            return frozenset((nd.site_id, nd.kind) for nd in path.noise_draws)

        sigs = {id(p): _noise_sig(p) for p in paths}

        for pi, pj in itertools.combinations(paths, 2):
            if not pi.noise_draws and not pj.noise_draws:
                continue

            # Skip pairs with identical noise signatures
            if sigs[id(pi)] == sigs[id(pj)]:
                continue

            ratio = self._build_one_cross_path_ratio(pi, pj, sens_map)
            if ratio is not None:
                ratios.append(ratio)
                reverse_ratio = self._build_one_cross_path_ratio(pj, pi, sens_map)
                if reverse_ratio is not None:
                    ratios.append(reverse_ratio)

        return ratios

    def _build_one_cross_path_ratio(
        self,
        path_d: SymbolicPath,
        path_dp: SymbolicPath,
        sens_map: dict[int, TypedExpr],
    ) -> DensityRatioExpr | None:
        """Build the density ratio for d on path_d, d' on path_dp.

        If the paths share common noise sites, the ratio is the sum
        of per-site log ratios.  For unmatched sites, we compute the
        density contribution from each path separately.
        """
        if not path_d.noise_draws and not path_dp.noise_draws:
            return None

        d_sites = {nd.site_id: nd for nd in path_d.noise_draws}
        dp_sites = {nd.site_id: nd for nd in path_dp.noise_draws}
        all_sites = set(d_sites.keys()) | set(dp_sites.keys())

        terms: list[TypedExpr] = []
        noise_site_list: list[int] = list(all_sites)

        for site_id in sorted(all_sites):
            nd_d = d_sites.get(site_id)
            nd_dp = dp_sites.get(site_id)

            if nd_d is not None and nd_dp is not None:
                obs_var = Var(ty=IRType.REAL, name=f"o_{nd_d.variable}")
                c_dp = self._make_d_prime_center(nd_d, sens_map)
                model = self._noise_models.get(nd_d.kind)
                if model is None:
                    model = get_noise_model(nd_d.kind)
                ratio = model.symbolic_log_ratio(
                    obs_var, nd_d.center_expr, c_dp, nd_d.scale_expr
                )
                terms.append(ratio)
            elif nd_d is not None:
                obs_var = Var(ty=IRType.REAL, name=f"o_{nd_d.variable}")
                model = self._noise_models.get(nd_d.kind)
                if model is None:
                    model = get_noise_model(nd_d.kind)
                log_d = model.symbolic_log_density(obs_var, nd_d.center_expr, nd_d.scale_expr)
                terms.append(log_d)
            elif nd_dp is not None:
                obs_var = Var(ty=IRType.REAL, name=f"o_{nd_dp.variable}")
                model = self._noise_models.get(nd_dp.kind)
                if model is None:
                    model = get_noise_model(nd_dp.kind)
                log_dp = model.symbolic_log_density(obs_var, nd_dp.center_expr, nd_dp.scale_expr)
                neg_log_dp = BinOp(
                    ty=IRType.REAL, op=BinOpKind.MUL,
                    left=Const.real(-1.0), right=log_dp,
                )
                terms.append(neg_log_dp)

        if not terms:
            return None

        log_ratio = terms[0]
        for t in terms[1:]:
            log_ratio = BinOp(ty=IRType.REAL, op=BinOpKind.ADD, left=log_ratio, right=t)

        return DensityRatioExpr(
            log_ratio=log_ratio,
            path_condition_d=path_d.path_condition,
            path_condition_d_prime=path_dp.path_condition,
            path_id_d=path_d.path_id,
            path_id_d_prime=path_dp.path_id,
            noise_sites=noise_site_list,
            is_cross_path=True,
        )

    def _make_d_prime_center(
        self, nd: NoiseDrawInfo, sens_map: dict[int, TypedExpr]
    ) -> TypedExpr:
        """Create the centre expression for d' from a noise draw.

        If a sensitivity is known for the site, the d' centre is
        ``center + delta`` where delta is bounded by the sensitivity.
        Otherwise, a fresh symbolic variable is introduced.

        Results are cached by (site_id, variable) for reuse.
        """
        cache_key = (nd.site_id, nd.variable)
        cached = self._center_cache.get(cache_key)
        if cached is not None:
            return cached

        if nd.site_id in sens_map:
            delta_var = Var(ty=IRType.REAL, name=f"delta_{nd.variable}")
            result = BinOp(
                ty=IRType.REAL, op=BinOpKind.ADD,
                left=nd.center_expr, right=delta_var,
            )
        else:
            dp_name = f"{nd.variable}{self._dp_suffix}"
            result = Var(ty=IRType.REAL, name=dp_name)

        self._center_cache[cache_key] = result
        return result

    def simplify_ratio(self, ratio: TypedExpr) -> TypedExpr:
        """Apply algebraic simplification to a density ratio expression.

        This performs bottom-up simplification including:
        - Constant folding.
        - Cancellation of terms.
        - Combining absolute value expressions.

        Args:
            ratio: The ratio expression to simplify.

        Returns:
            The simplified expression.
        """
        return ratio.simplify()
