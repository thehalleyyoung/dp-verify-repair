"""
Differential privacy mechanism implementations for testing and demonstration.

This package contains reference implementations of standard DP mechanisms,
including both correct versions and known-buggy variants for verification
and repair benchmarking.
"""

from examples.mechanisms.laplace_mechanism import (
    LAPLACE_CORRECT,
    LAPLACE_WRONG_SCALE,
    LAPLACE_MISSING_NOISE,
    laplace_mechanism_source,
)
from examples.mechanisms.gaussian_mechanism import (
    GAUSSIAN_CORRECT,
    GAUSSIAN_ANALYTIC_CORRECT,
    GAUSSIAN_WRONG_SIGMA,
    gaussian_mechanism_source,
)
from examples.mechanisms.exponential_mechanism import (
    EXPONENTIAL_CORRECT,
    EXPONENTIAL_WRONG_SENSITIVITY,
    exponential_mechanism_source,
)
from examples.mechanisms.sparse_vector import (
    SVT_CORRECT,
    SVT_BUG1_NO_THRESHOLD_NOISE,
    SVT_BUG2_REUSE_THRESHOLD,
    SVT_BUG3_WRONG_SENSITIVITY,
    SVT_BUG4_NO_HALT,
    SVT_BUG5_WRONG_BUDGET,
    sparse_vector_source,
)

__all__ = [
    "LAPLACE_CORRECT",
    "LAPLACE_WRONG_SCALE",
    "LAPLACE_MISSING_NOISE",
    "laplace_mechanism_source",
    "GAUSSIAN_CORRECT",
    "GAUSSIAN_ANALYTIC_CORRECT",
    "GAUSSIAN_WRONG_SIGMA",
    "gaussian_mechanism_source",
    "EXPONENTIAL_CORRECT",
    "EXPONENTIAL_WRONG_SENSITIVITY",
    "exponential_mechanism_source",
    "SVT_CORRECT",
    "SVT_BUG1_NO_THRESHOLD_NOISE",
    "SVT_BUG2_REUSE_THRESHOLD",
    "SVT_BUG3_WRONG_SENSITIVITY",
    "SVT_BUG4_NO_HALT",
    "SVT_BUG5_WRONG_BUDGET",
    "sparse_vector_source",
]
