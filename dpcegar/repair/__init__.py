"""Automated repair strategies for differential privacy mechanisms.

Submodules
----------
templates    – repair template grammar (ScaleParam, ClampBound, …)
synthesizer  – CEGIS-based repair synthesis
patcher      – code patch generation and source-level application
validator    – repair validation and regression testing
"""

from dpcegar.repair.templates import (
    ClampBound,
    CompositeRepair,
    CompositionBudgetSplit,
    NoiseSwap,
    RepairParameter,
    RepairSite,
    RepairTemplate,
    ScaleParam,
    SensitivityRescale,
    TemplateCost,
    TemplateEnumerator,
    TemplateValidator,
    ThresholdShift,
    ValidationResult,
)
from dpcegar.repair.synthesizer import (
    CostFunction,
    CounterexampleAccumulator,
    MinimizationResult,
    RepairMinimizer,
    RepairResult,
    RepairStatistics,
    RepairSynthesizer,
    RepairVerdict,
    RepairVerifier,
    SynthesizerConfig,
)
from dpcegar.repair.patcher import (
    MechIRPrinter,
    PatchEntry,
    PatchGenerator,
    PatchReport,
    SourcePatcher,
)
from dpcegar.repair.validator import (
    CertificateChecker,
    FunctionalChecker,
    NumericalSanityChecker,
    ParameterRangeChecker,
    RegressionTester,
    RepairValidator,
    ValidationFinding,
    ValidationReport,
    ValidationSeverity,
)

__all__ = [
    # templates
    "ClampBound",
    "CompositeRepair",
    "CompositionBudgetSplit",
    "NoiseSwap",
    "RepairParameter",
    "RepairSite",
    "RepairTemplate",
    "ScaleParam",
    "SensitivityRescale",
    "TemplateCost",
    "TemplateEnumerator",
    "TemplateValidator",
    "ThresholdShift",
    "ValidationResult",
    # synthesizer
    "CostFunction",
    "CounterexampleAccumulator",
    "MinimizationResult",
    "RepairMinimizer",
    "RepairResult",
    "RepairStatistics",
    "RepairSynthesizer",
    "RepairVerdict",
    "RepairVerifier",
    "SynthesizerConfig",
    # patcher
    "MechIRPrinter",
    "PatchEntry",
    "PatchGenerator",
    "PatchReport",
    "SourcePatcher",
    # validator
    "CertificateChecker",
    "FunctionalChecker",
    "NumericalSanityChecker",
    "ParameterRangeChecker",
    "RegressionTester",
    "RepairValidator",
    "ValidationFinding",
    "ValidationReport",
    "ValidationSeverity",
]
