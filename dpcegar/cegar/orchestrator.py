"""High-level orchestration for the DP-CEGAR verification pipeline.

Coordinates the full pipeline: parse → paths → density → CEGAR → result.
Provides progress callbacks, caching, timeout management, and batch
verification across multiple mechanisms or privacy budgets.

Classes
-------
PipelineStage          – enum of pipeline stages
StageResult            – result of a single pipeline stage
VerificationOrchestrator – full pipeline orchestration
ResultAggregator       – combine results from multiple runs
BatchVerifier          – verify multiple mechanisms or budgets
PipelineCache          – caching of intermediate results
"""

from __future__ import annotations

import copy
import logging
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

logger = logging.getLogger(__name__)

from dpcegar.ir.types import (
    NoiseKind,
    PrivacyBudget,
    PrivacyNotion,
    PureBudget,
    ApproxBudget,
    ZCDPBudget,
    TypedExpr,
)
from dpcegar.ir.nodes import MechIR
from dpcegar.paths.symbolic_path import PathSet, SymbolicPath
from dpcegar.density.ratio_builder import DensityRatioExpr, DensityRatioResult
from dpcegar.density.privacy_loss import PrivacyLossResult
from dpcegar.cegar.abstraction import AbstractDensityBound, AbstractionState
from dpcegar.cegar.engine import (
    CEGARConfig,
    CEGAREngine,
    CEGARResult,
    CEGARStatistics,
    CEGARVerdict,
    SMTSolverInterface,
)
from dpcegar.utils.errors import (
    DPCegarError,
    InternalError,
    VerificationError,
    ensure,
)
from dpcegar.certificates.certificate import (
    Certificate,
    CertificateType,
    VerificationCertificate as FullVerificationCertificate,
    RefutationCertificate,
)
from dpcegar.repair.synthesizer import (
    RepairResult,
    RepairSynthesizer,
    SynthesizerConfig,
)


# ═══════════════════════════════════════════════════════════════════════════
# PIPELINE STAGES
# ═══════════════════════════════════════════════════════════════════════════


class PipelineStage(Enum):
    """Stages of the verification pipeline."""

    PARSING = auto()
    PATH_ENUMERATION = auto()
    DENSITY_CONSTRUCTION = auto()
    PRIVACY_ANALYSIS = auto()
    CEGAR_VERIFICATION = auto()
    REPAIR = auto()
    COMPLETED = auto()
    FAILED = auto()


@dataclass(slots=True)
class StageResult:
    """Result of a single pipeline stage.

    Attributes:
        stage: Which stage produced this result.
        success: Whether the stage succeeded.
        elapsed_time: Time spent in this stage.
        output: Stage-specific output data.
        error: Error message if failed.
        details: Additional stage-specific information.
    """

    stage: PipelineStage = PipelineStage.COMPLETED
    success: bool = True
    elapsed_time: float = 0.0
    output: Any = None
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        status = "OK" if self.success else "FAIL"
        return f"StageResult({self.stage.name}, {status}, {self.elapsed_time:.2f}s)"


# ═══════════════════════════════════════════════════════════════════════════
# PIPELINE CACHE
# ═══════════════════════════════════════════════════════════════════════════


class PipelineCache:
    """Cache intermediate results across pipeline stages.

    Stores results keyed by mechanism name and stage, allowing
    repeated verification runs (e.g. with different budgets) to
    skip already-computed stages.

    Attributes:
        _cache: Internal storage keyed by (mechanism_name, stage).
    """

    def __init__(self) -> None:
        """Initialise an empty cache."""
        self._cache: dict[tuple[str, PipelineStage], StageResult] = {}
        self._hit_count: int = 0
        self._miss_count: int = 0

    def get(
        self, mechanism_name: str, stage: PipelineStage
    ) -> StageResult | None:
        """Retrieve a cached stage result.

        Args:
            mechanism_name: Name of the mechanism.
            stage: The pipeline stage.

        Returns:
            Cached result, or None if not cached.
        """
        key = (mechanism_name, stage)
        result = self._cache.get(key)
        if result is not None:
            self._hit_count += 1
        else:
            self._miss_count += 1
        return result

    def put(
        self, mechanism_name: str, stage: PipelineStage, result: StageResult
    ) -> None:
        """Store a stage result in the cache.

        Args:
            mechanism_name: Name of the mechanism.
            stage: The pipeline stage.
            result: The result to cache.
        """
        self._cache[(mechanism_name, stage)] = result

    def invalidate(self, mechanism_name: str) -> None:
        """Invalidate all cached results for a mechanism.

        Args:
            mechanism_name: Name of the mechanism.
        """
        keys_to_remove = [
            k for k in self._cache if k[0] == mechanism_name
        ]
        for k in keys_to_remove:
            del self._cache[k]

    def clear(self) -> None:
        """Clear the entire cache."""
        self._cache.clear()
        self._hit_count = 0
        self._miss_count = 0

    def statistics(self) -> dict[str, int]:
        """Return cache hit/miss statistics.

        Returns:
            Dictionary with hits, misses, and total entries.
        """
        return {
            "hits": self._hit_count,
            "misses": self._miss_count,
            "entries": len(self._cache),
        }

    def __len__(self) -> int:
        return len(self._cache)


# ═══════════════════════════════════════════════════════════════════════════
# PROGRESS TRACKER
# ═══════════════════════════════════════════════════════════════════════════


class ProgressTracker:
    """Track and report pipeline progress.

    Provides a uniform progress reporting interface across all
    pipeline stages.

    Attributes:
        _callback: User-provided callback for progress updates.
        _current_stage: The currently executing stage.
        _stage_progress: Per-stage progress [0, 1].
    """

    def __init__(
        self,
        callback: Callable[[PipelineStage, float, str], None] | None = None,
    ) -> None:
        """Initialise with an optional callback.

        Args:
            callback: Called with (stage, progress, message) on updates.
        """
        self._callback = callback
        self._current_stage = PipelineStage.PARSING
        self._stage_progress: dict[PipelineStage, float] = {}

    def enter_stage(self, stage: PipelineStage) -> None:
        """Mark the start of a pipeline stage.

        Args:
            stage: The stage being entered.
        """
        self._current_stage = stage
        self._stage_progress[stage] = 0.0
        self._notify(stage, 0.0, f"Starting {stage.name}")

    def update(self, progress: float, message: str = "") -> None:
        """Update progress within the current stage.

        Args:
            progress: Progress fraction [0, 1].
            message: Optional status message.
        """
        self._stage_progress[self._current_stage] = progress
        self._notify(self._current_stage, progress, message)

    def complete_stage(self, stage: PipelineStage) -> None:
        """Mark a stage as complete.

        Args:
            stage: The completed stage.
        """
        self._stage_progress[stage] = 1.0
        self._notify(stage, 1.0, f"Completed {stage.name}")

    def overall_progress(self) -> float:
        """Compute overall pipeline progress.

        Returns:
            Overall progress in [0, 1].
        """
        total_stages = len(PipelineStage) - 2  # exclude COMPLETED and FAILED
        if total_stages == 0:
            return 1.0
        completed = sum(
            1 for s, p in self._stage_progress.items()
            if p >= 1.0 and s not in (PipelineStage.COMPLETED, PipelineStage.FAILED)
        )
        return completed / total_stages

    def _notify(self, stage: PipelineStage, progress: float, message: str) -> None:
        """Send a progress notification.

        Args:
            stage: Current stage.
            progress: Stage progress.
            message: Status message.
        """
        if self._callback is not None:
            self._callback(stage, progress, message)


# ═══════════════════════════════════════════════════════════════════════════
# TIMEOUT MANAGER
# ═══════════════════════════════════════════════════════════════════════════


class TimeoutManager:
    """Manage timeouts across pipeline stages.

    Distributes a total time budget across stages, ensuring no single
    stage can monopolise the entire budget.

    Attributes:
        total_budget: Total time budget in seconds.
        _stage_budgets: Per-stage time allocation.
        _start_time: Overall start time.
        _stage_starts: Per-stage start times.
    """

    _DEFAULT_WEIGHTS: dict[PipelineStage, float] = {
        PipelineStage.PARSING: 0.05,
        PipelineStage.PATH_ENUMERATION: 0.15,
        PipelineStage.DENSITY_CONSTRUCTION: 0.10,
        PipelineStage.PRIVACY_ANALYSIS: 0.05,
        PipelineStage.CEGAR_VERIFICATION: 0.55,
        PipelineStage.REPAIR: 0.10,
    }

    def __init__(
        self,
        total_budget: float = 600.0,
        stage_weights: dict[PipelineStage, float] | None = None,
    ) -> None:
        """Initialise with a total time budget.

        Args:
            total_budget: Total time in seconds.
            stage_weights: Optional per-stage weight allocation.
        """
        self.total_budget = total_budget
        weights = stage_weights or self._DEFAULT_WEIGHTS
        total_weight = sum(weights.values())
        self._stage_budgets = {
            stage: (w / total_weight) * total_budget
            for stage, w in weights.items()
        }
        self._start_time = time.monotonic()
        self._stage_starts: dict[PipelineStage, float] = {}

    def start_stage(self, stage: PipelineStage) -> None:
        """Mark the start of a stage.

        Args:
            stage: The stage starting.
        """
        self._stage_starts[stage] = time.monotonic()

    def remaining_for_stage(self, stage: PipelineStage) -> float:
        """Return the remaining time for a stage.

        Args:
            stage: The stage to check.

        Returns:
            Remaining seconds (may be 0 if exceeded).
        """
        start = self._stage_starts.get(stage)
        if start is None:
            return self._stage_budgets.get(stage, 30.0)
        elapsed = time.monotonic() - start
        budget = self._stage_budgets.get(stage, 30.0)
        return max(0.0, budget - elapsed)

    def remaining_total(self) -> float:
        """Return the remaining total time.

        Returns:
            Remaining seconds.
        """
        elapsed = time.monotonic() - self._start_time
        return max(0.0, self.total_budget - elapsed)

    def is_expired(self, stage: PipelineStage | None = None) -> bool:
        """Check if time has expired.

        Args:
            stage: Optional specific stage to check.

        Returns:
            True if the time budget is exhausted.
        """
        if self.remaining_total() <= 0:
            return True
        if stage is not None and self.remaining_for_stage(stage) <= 0:
            return True
        return False

    def stage_budget(self, stage: PipelineStage) -> float:
        """Return the allocated budget for a stage.

        Args:
            stage: The stage.

        Returns:
            Time budget in seconds.
        """
        return self._stage_budgets.get(stage, 30.0)


# ═══════════════════════════════════════════════════════════════════════════
# PATH ENUMERATOR INTERFACE
# ═══════════════════════════════════════════════════════════════════════════


class PathEnumeratorInterface(Protocol):
    """Protocol for the path enumeration module."""

    def enumerate(self, mechanism: MechIR, max_depth: int) -> PathSet:
        """Enumerate symbolic paths through the mechanism.

        Args:
            mechanism: The mechanism IR.
            max_depth: Maximum path depth / loop unrolling.

        Returns:
            A PathSet of symbolic paths.
        """
        ...


class DensityBuilderInterface(Protocol):
    """Protocol for the density ratio builder."""

    def build(
        self, path_set: PathSet, sensitivity_map: dict[int, float] | None
    ) -> DensityRatioResult:
        """Build density ratio expressions from paths.

        Args:
            path_set: Enumerated paths.
            sensitivity_map: Optional per-query sensitivity.

        Returns:
            DensityRatioResult.
        """
        ...


# ═══════════════════════════════════════════════════════════════════════════
# VERIFICATION ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class OrchestratorConfig:
    """Configuration for the verification orchestrator.

    Attributes:
        cegar_config: Configuration for the CEGAR engine.
        total_timeout: Overall pipeline timeout in seconds.
        enable_caching: Whether to cache intermediate results.
        max_path_depth: Maximum path enumeration depth.
        sensitivity_map: Per-query sensitivity values.
        enable_repair: Whether to attempt repair on counterexample.
    """

    cegar_config: CEGARConfig = field(default_factory=CEGARConfig)
    total_timeout: float = 600.0
    enable_caching: bool = True
    max_path_depth: int = 10
    sensitivity_map: dict[int, float] = field(default_factory=dict)
    enable_repair: bool = False


@dataclass(slots=True)
class OrchestratorResult:
    """Full pipeline result from the orchestrator.

    Attributes:
        mechanism_name: Name of the verified mechanism.
        cegar_result: The CEGAR verification result.
        repair_result: The repair synthesis result (if repair was attempted).
        certificate: Certificate from the certificates module (if produced).
        stage_results: Per-stage results.
        total_time: Total pipeline time.
        details: Additional information.
    """

    mechanism_name: str = ""
    cegar_result: CEGARResult | None = None
    repair_result: RepairResult | None = None
    certificate: Certificate | None = None
    stage_results: dict[PipelineStage, StageResult] = field(default_factory=dict)
    total_time: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def is_verified(self) -> bool:
        """Return True if the mechanism was verified private."""
        return self.cegar_result is not None and self.cegar_result.is_verified

    @property
    def verdict(self) -> CEGARVerdict:
        """Return the verification verdict."""
        if self.cegar_result is None:
            return CEGARVerdict.ERROR
        return self.cegar_result.verdict

    def summary(self) -> str:
        """Return a human-readable summary."""
        parts = [f"Pipeline: {self.mechanism_name}"]
        for stage, sr in self.stage_results.items():
            parts.append(f"  {stage.name}: {sr}")
        if self.cegar_result:
            parts.append(f"  Result: {self.cegar_result.summary()}")
        parts.append(f"  Total: {self.total_time:.2f}s")
        return "\n".join(parts)

    def __str__(self) -> str:
        v = self.verdict.name
        return f"OrchestratorResult({self.mechanism_name}, {v}, {self.total_time:.2f}s)"


class VerificationOrchestrator:
    """Full pipeline orchestrator: parse → paths → density → CEGAR → result.

    Coordinates all pipeline stages, manages timeouts, caches intermediate
    results, and provides progress callbacks.

    Usage::

        orchestrator = VerificationOrchestrator(config=OrchestratorConfig())
        result = orchestrator.verify(mechanism, budget)
        print(result.summary())
    """

    def __init__(
        self,
        config: OrchestratorConfig | None = None,
        solver: SMTSolverInterface | None = None,
        path_enumerator: PathEnumeratorInterface | None = None,
        density_builder: DensityBuilderInterface | None = None,
        progress_callback: Callable[[PipelineStage, float, str], None] | None = None,
    ) -> None:
        """Initialise the orchestrator.

        Args:
            config: Pipeline configuration.
            solver: Optional SMT solver.
            path_enumerator: Optional path enumerator.
            density_builder: Optional density builder.
            progress_callback: Optional progress callback.
        """
        self._config = config or OrchestratorConfig()
        self._solver = solver
        self._enumerator = path_enumerator
        self._density_builder = density_builder
        self._progress = ProgressTracker(progress_callback)
        self._cache = PipelineCache() if self._config.enable_caching else None
        self._timeout = TimeoutManager(self._config.total_timeout)

    def verify(
        self,
        mechanism: MechIR,
        budget: PrivacyBudget,
        path_set: PathSet | None = None,
        density_ratios: DensityRatioResult | None = None,
    ) -> OrchestratorResult:
        """Run the full verification pipeline.

        Executes each stage in sequence, using cached results when
        available and respecting timeouts.

        Args:
            mechanism: The mechanism IR to verify.
            budget: The privacy budget.
            path_set: Optional pre-enumerated paths (skips enumeration).
            density_ratios: Optional pre-built density ratios.

        Returns:
            An OrchestratorResult with all stage results.
        """
        start_time = time.monotonic()
        stage_results: dict[PipelineStage, StageResult] = {}
        result = OrchestratorResult(mechanism_name=mechanism.name)

        try:
            paths = self._run_path_enumeration(
                mechanism, path_set, stage_results
            )
            if paths is None:
                result.stage_results = stage_results
                result.total_time = time.monotonic() - start_time
                return result

            ratios = self._run_density_construction(
                mechanism, paths, density_ratios, stage_results
            )

            cegar_result = self._run_cegar_verification(
                paths, budget, ratios, stage_results
            )
            result.cegar_result = cegar_result

            # Create a proper Certificate from the certificates module
            if cegar_result.verdict == CEGARVerdict.VERIFIED:
                result.certificate = FullVerificationCertificate.from_cegar_result(
                    mechanism_id=mechanism.name,
                    mechanism_name=mechanism.name,
                    result=cegar_result,
                )
            elif cegar_result.verdict == CEGARVerdict.COUNTEREXAMPLE:
                cx = cegar_result.counterexample
                cx_dict: dict[str, float] = {}
                violation = 0.0
                path_id = 0
                if cx is not None:
                    cx_dict = getattr(cx, "assignment", {}) or {}
                    violation = getattr(cx, "violation_magnitude", 0.0)
                    path_id = getattr(cx, "path_id", 0)
                notion = _budget_to_notion(budget)
                result.certificate = RefutationCertificate(
                    mechanism_id=mechanism.name,
                    mechanism_name=mechanism.name,
                    privacy_notion=notion,
                    privacy_guarantee=budget,
                    counterexample=cx_dict,
                    violation_magnitude=violation,
                    path_id=path_id,
                )

                if self._config.enable_repair:
                    repair_result = self._run_repair(
                        mechanism, budget, cegar_result,
                        paths, ratios, stage_results,
                    )
                    result.repair_result = repair_result

        except DPCegarError as e:
            stage_results[PipelineStage.FAILED] = StageResult(
                stage=PipelineStage.FAILED,
                success=False,
                error=str(e),
            )

        result.stage_results = stage_results
        result.total_time = time.monotonic() - start_time
        return result

    def _run_path_enumeration(
        self,
        mechanism: MechIR,
        existing: PathSet | None,
        stage_results: dict[PipelineStage, StageResult],
    ) -> PathSet | None:
        """Run or skip the path enumeration stage.

        Args:
            mechanism: The mechanism IR.
            existing: Pre-enumerated paths (if any).
            stage_results: Stage result accumulator.

        Returns:
            The path set, or None on failure.
        """
        stage = PipelineStage.PATH_ENUMERATION
        self._progress.enter_stage(stage)
        self._timeout.start_stage(stage)

        if existing is not None:
            sr = StageResult(
                stage=stage, success=True, output=existing,
                details={"source": "pre-enumerated", "count": existing.size()},
            )
            stage_results[stage] = sr
            self._progress.complete_stage(stage)
            return existing

        if self._cache is not None:
            cached = self._cache.get(mechanism.name, stage)
            if cached is not None and cached.success:
                stage_results[stage] = cached
                self._progress.complete_stage(stage)
                return cached.output

        if self._enumerator is None:
            path_set = PathSet()
            sr = StageResult(
                stage=stage, success=True, output=path_set,
                details={"source": "empty", "reason": "no enumerator"},
            )
        else:
            t0 = time.monotonic()
            try:
                path_set = self._enumerator.enumerate(
                    mechanism, self._config.max_path_depth
                )
                elapsed = time.monotonic() - t0
                sr = StageResult(
                    stage=stage, success=True, output=path_set,
                    elapsed_time=elapsed,
                    details={"count": path_set.size()},
                )
            except DPCegarError as e:
                sr = StageResult(
                    stage=stage, success=False, error=str(e),
                    elapsed_time=time.monotonic() - t0,
                )
                stage_results[stage] = sr
                return None

        stage_results[stage] = sr
        if self._cache is not None:
            self._cache.put(mechanism.name, stage, sr)
        self._progress.complete_stage(stage)
        return sr.output

    def _run_density_construction(
        self,
        mechanism: MechIR,
        path_set: PathSet,
        existing: DensityRatioResult | None,
        stage_results: dict[PipelineStage, StageResult],
    ) -> DensityRatioResult | None:
        """Run or skip the density construction stage.

        Args:
            mechanism: The mechanism IR.
            path_set: Enumerated paths.
            existing: Pre-built density ratios.
            stage_results: Stage result accumulator.

        Returns:
            The density ratio result, or None.
        """
        stage = PipelineStage.DENSITY_CONSTRUCTION
        self._progress.enter_stage(stage)
        self._timeout.start_stage(stage)

        if existing is not None:
            sr = StageResult(
                stage=stage, success=True, output=existing,
                details={"source": "pre-computed"},
            )
            stage_results[stage] = sr
            self._progress.complete_stage(stage)
            return existing

        if self._cache is not None:
            cached = self._cache.get(mechanism.name, stage)
            if cached is not None and cached.success:
                stage_results[stage] = cached
                self._progress.complete_stage(stage)
                return cached.output

        if self._density_builder is None:
            sr = StageResult(
                stage=stage, success=True, output=None,
                details={"reason": "no density builder"},
            )
            stage_results[stage] = sr
            self._progress.complete_stage(stage)
            return None

        t0 = time.monotonic()
        try:
            ratios = self._density_builder.build(
                path_set, self._config.sensitivity_map or None
            )
            elapsed = time.monotonic() - t0
            sr = StageResult(
                stage=stage, success=True, output=ratios,
                elapsed_time=elapsed,
                details={"ratio_count": len(ratios.ratios)},
            )
        except DPCegarError as e:
            sr = StageResult(
                stage=stage, success=False, error=str(e),
                elapsed_time=time.monotonic() - t0,
            )
            stage_results[stage] = sr
            self._progress.complete_stage(stage)
            return None

        stage_results[stage] = sr
        if self._cache is not None:
            self._cache.put(mechanism.name, stage, sr)
        self._progress.complete_stage(stage)
        return ratios

    def _run_cegar_verification(
        self,
        path_set: PathSet,
        budget: PrivacyBudget,
        density_ratios: DensityRatioResult | None,
        stage_results: dict[PipelineStage, StageResult],
    ) -> CEGARResult:
        """Run the CEGAR verification stage.

        Args:
            path_set: Enumerated paths.
            budget: Privacy budget.
            density_ratios: Density ratio expressions.
            stage_results: Stage result accumulator.

        Returns:
            The CEGAR result.
        """
        stage = PipelineStage.CEGAR_VERIFICATION
        self._progress.enter_stage(stage)
        self._timeout.start_stage(stage)

        remaining = self._timeout.remaining_for_stage(stage)
        cegar_config = copy.copy(self._config.cegar_config)
        cegar_config.timeout_seconds = min(
            cegar_config.timeout_seconds, remaining
        )

        def cegar_progress(msg: str, current: int, total: int) -> None:
            frac = current / max(total, 1)
            self._progress.update(frac, f"CEGAR: {msg} ({current}/{total})")

        cegar_config.progress_callback = cegar_progress

        engine = CEGAREngine(
            config=cegar_config,
            solver=self._solver,
        )

        t0 = time.monotonic()
        ratios_list = density_ratios.ratios if density_ratios else None
        cegar_result = engine.verify(path_set, budget, ratios_list)
        elapsed = time.monotonic() - t0

        sr = StageResult(
            stage=stage,
            success=cegar_result.verdict != CEGARVerdict.ERROR,
            output=cegar_result,
            elapsed_time=elapsed,
            details={"verdict": cegar_result.verdict.name},
        )
        stage_results[stage] = sr
        self._progress.complete_stage(stage)
        return cegar_result

    def _run_repair(
        self,
        mechanism: MechIR,
        budget: PrivacyBudget,
        cegar_result: CEGARResult,
        path_set: PathSet,
        density_ratios: DensityRatioResult | None,
        stage_results: dict[PipelineStage, StageResult],
    ) -> RepairResult | None:
        """Run the repair synthesis stage.

        Args:
            mechanism: The mechanism IR.
            budget: The privacy budget.
            cegar_result: The CEGAR result with a counterexample.
            path_set: Enumerated paths.
            density_ratios: Density ratio expressions.
            stage_results: Stage result accumulator.

        Returns:
            The repair result, or None on failure.
        """
        stage = PipelineStage.REPAIR
        self._progress.enter_stage(stage)
        self._timeout.start_stage(stage)

        remaining = self._timeout.remaining_for_stage(stage)
        if remaining <= 0:
            sr = StageResult(
                stage=stage, success=False,
                error="no time remaining for repair",
            )
            stage_results[stage] = sr
            self._progress.complete_stage(stage)
            return None

        synth_config = SynthesizerConfig(
            timeout_seconds=remaining,
        )
        synthesizer = RepairSynthesizer(
            config=synth_config,
            solver=self._solver,
        )

        initial_cex = cegar_result.counterexample
        ratios_list = density_ratios.ratios if density_ratios else None

        t0 = time.monotonic()
        try:
            repair_result = synthesizer.synthesize(
                mechanism, budget,
                initial_counterexample=initial_cex,
                path_set=path_set,
                density_ratios=ratios_list,
            )
            elapsed = time.monotonic() - t0
            sr = StageResult(
                stage=stage,
                success=repair_result.success,
                output=repair_result,
                elapsed_time=elapsed,
                details={"verdict": repair_result.verdict.name},
            )
        except Exception:
            logger.debug("Repair synthesis failed", exc_info=True)
            elapsed = time.monotonic() - t0
            sr = StageResult(
                stage=stage, success=False,
                elapsed_time=elapsed,
                error="repair synthesis raised an exception",
            )
            stage_results[stage] = sr
            self._progress.complete_stage(stage)
            return None

        stage_results[stage] = sr
        self._progress.complete_stage(stage)
        return repair_result

    def get_cache_statistics(self) -> dict[str, int]:
        """Return cache statistics.

        Returns:
            Cache hit/miss/entry counts.
        """
        if self._cache is None:
            return {"hits": 0, "misses": 0, "entries": 0}
        return self._cache.statistics()


# ═══════════════════════════════════════════════════════════════════════════
# RESULT AGGREGATOR
# ═══════════════════════════════════════════════════════════════════════════


class ResultAggregator:
    """Combine results from multiple verification runs.

    Useful for aggregating results when verifying under multiple privacy
    notions or multiple budget levels.
    """

    def __init__(self) -> None:
        """Initialise an empty aggregator."""
        self._results: list[OrchestratorResult] = []

    def add(self, result: OrchestratorResult) -> None:
        """Add a result to the aggregation.

        Args:
            result: A pipeline result to include.
        """
        self._results.append(result)

    def all_verified(self) -> bool:
        """Return True if all mechanisms were verified.

        Returns:
            True if every result has verdict VERIFIED.
        """
        return all(r.is_verified for r in self._results)

    def any_violated(self) -> bool:
        """Return True if any mechanism has a violation.

        Returns:
            True if any result has verdict COUNTEREXAMPLE.
        """
        return any(
            r.verdict == CEGARVerdict.COUNTEREXAMPLE for r in self._results
        )

    def by_verdict(self) -> dict[str, list[OrchestratorResult]]:
        """Group results by verdict.

        Returns:
            Mapping from verdict name to list of results.
        """
        groups: dict[str, list[OrchestratorResult]] = {}
        for r in self._results:
            k = r.verdict.name
            groups.setdefault(k, []).append(r)
        return groups

    def total_time(self) -> float:
        """Return the total time across all results.

        Returns:
            Sum of all result times.
        """
        return sum(r.total_time for r in self._results)

    def summary(self) -> dict[str, Any]:
        """Return a summary of all results.

        Returns:
            Aggregated statistics.
        """
        verdicts = self.by_verdict()
        return {
            "total_mechanisms": len(self._results),
            "verified": len(verdicts.get("VERIFIED", [])),
            "counterexamples": len(verdicts.get("COUNTEREXAMPLE", [])),
            "unknown": len(verdicts.get("UNKNOWN", [])),
            "timeout": len(verdicts.get("TIMEOUT", [])),
            "errors": len(verdicts.get("ERROR", [])),
            "total_time": self.total_time(),
        }

    def __len__(self) -> int:
        return len(self._results)

    def __str__(self) -> str:
        s = self.summary()
        return (
            f"ResultAggregator({s['total_mechanisms']} mechanisms, "
            f"{s['verified']} verified, {s['counterexamples']} violated)"
        )


# ═══════════════════════════════════════════════════════════════════════════
# BATCH VERIFIER
# ═══════════════════════════════════════════════════════════════════════════


class BatchVerifier:
    """Verify multiple mechanisms or multiple privacy budgets.

    Supports:
      - Verifying one mechanism against multiple budgets
      - Verifying multiple mechanisms against one budget
      - Full cross-product verification

    Uses a shared orchestrator and cache for efficiency.
    """

    def __init__(
        self,
        orchestrator: VerificationOrchestrator | None = None,
        config: OrchestratorConfig | None = None,
    ) -> None:
        """Initialise the batch verifier.

        Args:
            orchestrator: Optional shared orchestrator.
            config: Configuration (used if orchestrator is not provided).
        """
        self._orchestrator = orchestrator or VerificationOrchestrator(config)
        self._aggregator = ResultAggregator()

    def verify_budgets(
        self,
        mechanism: MechIR,
        budgets: list[PrivacyBudget],
        path_set: PathSet | None = None,
    ) -> ResultAggregator:
        """Verify one mechanism against multiple budgets.

        Caches path enumeration and density construction across runs.

        Args:
            mechanism: The mechanism to verify.
            budgets: List of privacy budgets.
            path_set: Optional pre-enumerated paths.

        Returns:
            Aggregated results across all budgets.
        """
        aggregator = ResultAggregator()

        for budget in budgets:
            result = self._orchestrator.verify(
                mechanism, budget, path_set=path_set
            )
            aggregator.add(result)

        return aggregator

    def verify_mechanisms(
        self,
        mechanisms: list[MechIR],
        budget: PrivacyBudget,
    ) -> ResultAggregator:
        """Verify multiple mechanisms against one budget.

        Args:
            mechanisms: List of mechanisms to verify.
            budget: The privacy budget.

        Returns:
            Aggregated results across all mechanisms.
        """
        aggregator = ResultAggregator()

        for mechanism in mechanisms:
            result = self._orchestrator.verify(mechanism, budget)
            aggregator.add(result)

        return aggregator

    def verify_cross_product(
        self,
        mechanisms: list[MechIR],
        budgets: list[PrivacyBudget],
    ) -> dict[tuple[str, str], OrchestratorResult]:
        """Verify all (mechanism, budget) pairs.

        Args:
            mechanisms: List of mechanisms.
            budgets: List of budgets.

        Returns:
            Mapping from (mechanism_name, budget_str) to result.
        """
        results: dict[tuple[str, str], OrchestratorResult] = {}

        for mechanism in mechanisms:
            for budget in budgets:
                result = self._orchestrator.verify(mechanism, budget)
                key = (mechanism.name, str(budget))
                results[key] = result

        return results

    def get_aggregator(self) -> ResultAggregator:
        """Return the accumulated aggregator.

        Returns:
            The ResultAggregator.
        """
        return self._aggregator


def _budget_to_notion(budget: PrivacyBudget | None) -> PrivacyNotion:
    """Map a budget object to a PrivacyNotion enum value."""
    if budget is None:
        return PrivacyNotion.PURE_DP
    if isinstance(budget, PureBudget):
        return PrivacyNotion.PURE_DP
    if isinstance(budget, ApproxBudget):
        return PrivacyNotion.APPROX_DP
    if isinstance(budget, ZCDPBudget):
        return PrivacyNotion.ZCDP
    type_name = type(budget).__name__.upper()
    for notion in PrivacyNotion:
        if notion.name in type_name:
            return notion
    return PrivacyNotion.PURE_DP
