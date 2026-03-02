"""CEGAR loop orchestration for differential privacy verification.

Submodules
----------
abstraction   – abstract domain, path partitions, widening
refinement    – refinement operators and convergence detection
engine        – main CEGAR loop driver
orchestrator  – full pipeline orchestration and batch verification
"""

from dpcegar.cegar.abstraction import (
    AbstractDensityBound,
    AbstractionState,
    AbstractState,
    InitialAbstraction,
    AbstractionLattice,
    PathPartition,
    RefinementKind,
    RefinementRecord,
    WideningOperator,
)
from dpcegar.cegar.refinement import (
    RefinementCounterexample,
    Counterexample,
    ConvergenceDetector,
    ConvergenceReason,
    ConvergenceStatus,
    InfeasibilityProof,
    IntervalNarrowRefinement,
    LoopUnwindRefinement,
    PathSplitRefinement,
    PredicateRefinement,
    RefinementHistory,
    RefinementOperator,
    RefinementResult,
    RefinementSelector,
    RefinementStatus,
)
from dpcegar.cegar.engine import (
    AbstractVerificationResult,
    AbstractVerifier,
    CEGARConfig,
    CEGAREngine,
    CEGARResult,
    CEGARStatistics,
    CEGARVerdict,
    CandidateExtractor,
    ConcreteCheckResult,
    ConcreteChecker,
    SMTResult,
    SMTSolverInterface,
    SMTStatus,
    SpuriousnessAnalysis,
    SpuriousnessAnalyzer,
    SpuriousnessCause,
    VerificationCertificate,
)
from dpcegar.cegar.orchestrator import (
    BatchVerifier,
    OrchestratorConfig,
    OrchestratorResult,
    PipelineCache,
    PipelineStage,
    ProgressTracker,
    ResultAggregator,
    StageResult,
    TimeoutManager,
    VerificationOrchestrator,
)

__all__ = [
    # abstraction
    "AbstractDensityBound",
    "AbstractionState",
    "AbstractState",
    "InitialAbstraction",
    "AbstractionLattice",
    "PathPartition",
    "RefinementKind",
    "RefinementRecord",
    "WideningOperator",
    # refinement
    "RefinementCounterexample",
    "Counterexample",
    "ConvergenceDetector",
    "ConvergenceReason",
    "ConvergenceStatus",
    "InfeasibilityProof",
    "IntervalNarrowRefinement",
    "LoopUnwindRefinement",
    "PathSplitRefinement",
    "PredicateRefinement",
    "RefinementHistory",
    "RefinementOperator",
    "RefinementResult",
    "RefinementSelector",
    "RefinementStatus",
    # engine
    "AbstractVerificationResult",
    "AbstractVerifier",
    "CEGARConfig",
    "CEGAREngine",
    "CEGARResult",
    "CEGARStatistics",
    "CEGARVerdict",
    "CandidateExtractor",
    "ConcreteCheckResult",
    "ConcreteChecker",
    "SMTResult",
    "SMTSolverInterface",
    "SMTStatus",
    "SpuriousnessAnalysis",
    "SpuriousnessAnalyzer",
    "SpuriousnessCause",
    "VerificationCertificate",
    # orchestrator
    "BatchVerifier",
    "OrchestratorConfig",
    "OrchestratorResult",
    "PipelineCache",
    "PipelineStage",
    "ProgressTracker",
    "ResultAggregator",
    "StageResult",
    "TimeoutManager",
    "VerificationOrchestrator",
]
