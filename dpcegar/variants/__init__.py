"""DP variant support: zCDP, RDP, f-DP, GDP conversions and verification.

Submodules:
    lattice         – Implication lattice between DP notions
    multi_checker   – Multi-variant verification orchestrator
    conversions     – Privacy notion conversion functions
    privacy_profile – Privacy profile computation and analysis
"""

from dpcegar.variants.lattice import (
    ImplicationEdge,
    ImplicationLattice,
    NodeStatus,
    ParameterTransformer,
    PrivacyLatticeNode,
)
from dpcegar.variants.multi_checker import (
    DerivedGuarantee,
    MultiVariantChecker,
    MultiVariantResult,
    MultiVariantStatistics,
    PlanOptimizer,
    VariantResult,
    VariantStatus,
    VerificationPlan,
)
from dpcegar.variants.conversions import (
    ConversionProvenance,
    ConversionRegistry,
    ConversionResult,
    OptimalConverter,
    build_default_registry,
    gdp_to_approx,
    gdp_to_fdp,
    pure_to_approx,
    pure_to_rdp,
    pure_to_zcdp,
    rdp_to_approx,
    rdp_to_approx_optimal,
    zcdp_to_approx,
    zcdp_to_rdp,
)
from dpcegar.variants.privacy_profile import (
    EpsilonDeltaCurve,
    PrivacyGuarantee,
    PrivacyProfile,
    ProfileComparison,
    ProfileComposer,
    ProfileVisualizer,
    RDPCurve,
)

__all__ = [
    "ConversionProvenance",
    "ConversionRegistry",
    "ConversionResult",
    "DerivedGuarantee",
    "EpsilonDeltaCurve",
    "ImplicationEdge",
    "ImplicationLattice",
    "MultiVariantChecker",
    "MultiVariantResult",
    "MultiVariantStatistics",
    "NodeStatus",
    "OptimalConverter",
    "ParameterTransformer",
    "PlanOptimizer",
    "PrivacyGuarantee",
    "PrivacyLatticeNode",
    "PrivacyProfile",
    "ProfileComparison",
    "ProfileComposer",
    "ProfileVisualizer",
    "RDPCurve",
    "VariantResult",
    "VariantStatus",
    "VerificationPlan",
    "build_default_registry",
    "gdp_to_approx",
    "gdp_to_fdp",
    "pure_to_approx",
    "pure_to_rdp",
    "pure_to_zcdp",
    "rdp_to_approx",
    "rdp_to_approx_optimal",
    "zcdp_to_approx",
    "zcdp_to_rdp",
]
