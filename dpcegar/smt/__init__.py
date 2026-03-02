"""SMT encoding and solver interface for privacy verification.

This package provides the SMT-based verification backend for DP-CEGAR:

Modules
-------
encoding          – Core IR-to-Z3 translation (ExprToZ3, SMTEncoding)
privacy_encoder   – Privacy-specific encoders for all DP variants
solver            – Z3/CVC5 solver wrappers with portfolio solving
counterexample    – Counterexample extraction, validation, minimisation
transcendental    – Polynomial approximations of exp, ln, Φ, Φ⁻¹
optimizer         – Optimization Modulo Theories for repair minimality
theory_selection  – Automatic SMT theory selection
"""

from dpcegar.smt.encoding import (
    SMTEncoding,
    ExprToZ3,
    AbsLinearizer,
    CaseSplitter,
    PathConditionEncoder,
    ConstraintBuilder,
    EncodingBuilder,
)
from dpcegar.smt.privacy_encoder import (
    PrivacyEncodingResult,
    PureEpsDPEncoder,
    ApproxDPEncoder,
    ZCDPEncoder,
    RDPEncoder,
    GDPEncoder,
    FDPEncoder,
    CrossPathEncoder,
)
from dpcegar.smt.solver import (
    CheckResult,
    SolverResult,
    SolverConfig,
    SolverStats,
    Z3Solver,
    PortfolioSolver,
    SolverPool,
)
from dpcegar.smt.counterexample import (
    Counterexample,
    CounterexampleSet,
    CounterexampleExtractor,
    SpuriousnessChecker,
    CounterexampleMinimizer,
    CounterexamplePrinter,
)
from dpcegar.smt.transcendental import (
    Precision,
    ApproxResult,
    SoundnessTracker,
    TranscendentalApprox,
    DRealInterface,
    DRealResult,
)
from dpcegar.smt.optimizer import (
    LinearObjective,
    OMTResult,
    OMTSolver,
    SoftConstraint,
)
from dpcegar.smt.theory_selection import (
    SMTTheory,
    TheoryAnalysisResult,
    TheoryAnalyzer,
    SolverRecommendation,
    TheoryFallbackChain,
    auto_configure,
)

__all__ = [
    # encoding
    "SMTEncoding",
    "ExprToZ3",
    "AbsLinearizer",
    "CaseSplitter",
    "PathConditionEncoder",
    "ConstraintBuilder",
    "EncodingBuilder",
    # privacy_encoder
    "PrivacyEncodingResult",
    "PureEpsDPEncoder",
    "ApproxDPEncoder",
    "ZCDPEncoder",
    "RDPEncoder",
    "GDPEncoder",
    "FDPEncoder",
    "CrossPathEncoder",
    # solver
    "CheckResult",
    "SolverResult",
    "SolverConfig",
    "SolverStats",
    "Z3Solver",
    "PortfolioSolver",
    "SolverPool",
    # counterexample
    "Counterexample",
    "CounterexampleSet",
    "CounterexampleExtractor",
    "SpuriousnessChecker",
    "CounterexampleMinimizer",
    "CounterexamplePrinter",
    # transcendental
    "Precision",
    "ApproxResult",
    "SoundnessTracker",
    "TranscendentalApprox",
    "DRealInterface",
    "DRealResult",
    # optimizer
    "LinearObjective",
    "OMTResult",
    "OMTSolver",
    "SoftConstraint",
    # theory_selection
    "SMTTheory",
    "TheoryAnalysisResult",
    "TheoryAnalyzer",
    "SolverRecommendation",
    "TheoryFallbackChain",
    "auto_configure",
]
