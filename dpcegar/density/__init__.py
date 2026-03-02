"""Density function lifting and manipulation.

Sub-modules
-----------
noise_models    – NoiseModel, LaplaceNoise, GaussianNoise, etc.
ratio_builder   – DensityRatioBuilder, DensityRatioExpr
privacy_loss    – PrivacyLossComputer, PrivacyLossResult
composition     – SequentialComposition, ParallelComposition, etc.
"""

from dpcegar.density.noise_models import (
    NoiseModel,
    LaplaceNoise,
    GaussianNoise,
    ExponentialMechNoise,
    TruncatedLaplaceNoise,
    TruncatedGaussianNoise,
    DiscreteGaussianNoise,
    DiscreteLaplaceNoise,
    MixtureNoise,
    get_noise_model,
)
from dpcegar.density.ratio_builder import (
    DensityRatioBuilder,
    DensityRatioExpr,
    DensityRatioResult,
    SensitivityTemplate,
)
from dpcegar.density.privacy_loss import (
    PrivacyLossComputer,
    PrivacyLossResult,
    PerPathLoss,
)
from dpcegar.density.composition import (
    CompositionOptimizer,
    CompositionResult,
    GDPComposition,
    ParallelComposition,
    RDPComposition,
    SequentialComposition,
    SubsamplingAmplification,
    ZCDPComposition,
    advanced_composition,
)

__all__ = [
    "NoiseModel", "LaplaceNoise", "GaussianNoise", "ExponentialMechNoise",
    "TruncatedLaplaceNoise", "TruncatedGaussianNoise",
    "DiscreteGaussianNoise", "DiscreteLaplaceNoise",
    "MixtureNoise", "get_noise_model",
    "DensityRatioBuilder", "DensityRatioExpr", "DensityRatioResult",
    "SensitivityTemplate",
    "PrivacyLossComputer", "PrivacyLossResult", "PerPathLoss",
    "CompositionOptimizer", "CompositionResult",
    "GDPComposition", "ParallelComposition",
    "RDPComposition", "SequentialComposition",
    "SubsamplingAmplification", "ZCDPComposition",
    "advanced_composition",
]
