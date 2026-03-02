"""
Noisy Histogram mechanism implementations for DP-CEGAR verification.

A noisy histogram adds independent noise to each bin count.  The standard
variant uses Laplace noise.  The stability-based variant exploits the
observation that most bins have 0 count on neighbouring datasets, so only
bins with sufficiently large noisy count need to be released.

Variants:
  - Standard noisy histogram (add Lap(1/ε) to every bin)
  - Stability-based histogram (only release "stable" bins)
  - Buggy: wrong noise scale, missing bin noise

References:
  Dwork & Roth. "The Algorithmic Foundations of DP." 2014, §3.4.
  Korolova, Kenthapadi, Mishra, Ntoulas. "Releasing Search Queries and
    Clicks Privately." WWW 2009.
"""

from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Source generators
# ---------------------------------------------------------------------------

def noisy_histogram_source(
    num_bins: int = 5,
    sensitivity: float = 1.0,
    epsilon: float = 1.0,
    name: str = "noisy_histogram",
) -> str:
    """Generate source for a standard noisy histogram.

    Adds Lap(Δf/ε) noise independently to each bin.  Sensitivity is 1
    for counting queries (changing one record changes one bin by 1).

    Args:
        num_bins: Number of histogram bins.
        sensitivity: Sensitivity per bin (typically 1).
        epsilon: Privacy budget ε.
        name: Function name.

    Returns:
        Python source with DPImp annotations.
    """
    scale = sensitivity / epsilon
    bins = []
    for i in range(num_bins):
        bins.append(f"""
    # @dp.noise(kind="laplace", scale={scale})
    noise_{i} = laplace(0, {scale})
    noisy_bin_{i} = counts[{i}] + noise_{i}""")
    body = "\n".join(bins)
    returns = ", ".join(f"noisy_bin_{i}" for i in range(num_bins))
    return f'''
# @dp.mechanism(privacy="{epsilon}-dp", sensitivity={sensitivity})
def {name}(db):
    """Standard noisy histogram with {num_bins} bins.

    Each bin gets independent Lap({scale}) noise.
    Sensitivity = {sensitivity} (one record changes one bin by 1).
    """
    # @dp.sensitivity({sensitivity})
    counts = histogram_query(db, n_bins={num_bins})
{body}
    return ({returns})
'''


def stability_histogram_source(
    num_bins: int = 10,
    sensitivity: float = 1.0,
    epsilon: float = 1.0,
    threshold: float = 2.0,
    name: str = "stability_histogram",
) -> str:
    """Generate source for a stability-based histogram.

    Only releases bins whose noisy count exceeds a threshold.  This
    exploits the observation that changing one record affects at most 2
    bins (one increases, one decreases), so the mechanism's sensitivity
    is low and budget usage is efficient.

    Args:
        num_bins: Number of histogram bins.
        sensitivity: Per-bin sensitivity.
        epsilon: Privacy budget.
        threshold: Release threshold.
        name: Function name.

    Returns:
        Python source with DPImp annotations.
    """
    scale = sensitivity / epsilon
    return f'''
# @dp.mechanism(privacy="{epsilon}-dp", sensitivity={sensitivity})
def {name}(db):
    """Stability-based histogram: only release bins above threshold.

    Threshold = {threshold}, noise scale = {scale}.
    Bins with noisy count < threshold are suppressed (set to 0).
    """
    # @dp.sensitivity({sensitivity})
    counts = histogram_query(db, n_bins={num_bins})
    out = []
    for i in range({num_bins}):
        # @dp.noise(kind="laplace", scale={scale})
        noise_i = laplace(0, {scale})
        noisy_count = counts[i] + noise_i
        if noisy_count >= {threshold}:
            out.append(noisy_count)
        else:
            out.append(0)
    return out
'''


def histogram_wrong_scale_source(
    num_bins: int = 5,
    sensitivity: float = 1.0,
    epsilon: float = 1.0,
    name: str = "histogram_wrong_scale",
) -> str:
    """Generate source for noisy histogram with wrong noise scale.

    BUG: Uses Lap(1/(n·ε)) instead of Lap(1/ε).  This divides the
    noise by the number of bins, providing far too little noise.

    Args:
        num_bins: Number of bins.
        sensitivity: Per-bin sensitivity.
        epsilon: Privacy budget.
        name: Function name.

    Returns:
        Python source with DPImp annotations.
    """
    wrong_scale = sensitivity / (num_bins * epsilon)  # BUG: divides by n_bins
    bins = []
    for i in range(num_bins):
        bins.append(f"""
    # @dp.noise(kind="laplace", scale={wrong_scale})
    noise_{i} = laplace(0, {wrong_scale})
    noisy_bin_{i} = counts[{i}] + noise_{i}""")
    body = "\n".join(bins)
    returns = ", ".join(f"noisy_bin_{i}" for i in range(num_bins))
    return f'''
# @dp.mechanism(privacy="{epsilon}-dp", sensitivity={sensitivity})
def {name}(db):
    """Noisy histogram with WRONG scale.

    BUG: scale = 1/(n·ε) = {wrong_scale} instead of 1/ε.
    Divides noise by number of bins — far too little noise.
    """
    # @dp.sensitivity({sensitivity})
    counts = histogram_query(db, n_bins={num_bins})
{body}
    return ({returns})
'''


def histogram_missing_bin_noise_source(
    num_bins: int = 5,
    sensitivity: float = 1.0,
    epsilon: float = 1.0,
    name: str = "histogram_missing_noise",
) -> str:
    """Generate source for noisy histogram where some bins get no noise.

    BUG: Only the first bin gets noise; the rest are released raw.

    Args:
        num_bins: Number of bins.
        sensitivity: Per-bin sensitivity.
        epsilon: Privacy budget.
        name: Function name.

    Returns:
        Python source with DPImp annotations.
    """
    scale = sensitivity / epsilon
    first_bin = f"""
    # @dp.noise(kind="laplace", scale={scale})
    noise_0 = laplace(0, {scale})
    noisy_bin_0 = counts[0] + noise_0"""
    rest = []
    for i in range(1, num_bins):
        rest.append(f"""
    # BUG: no noise on bin {i}!
    noisy_bin_{i} = counts[{i}]""")
    body = first_bin + "\n".join(rest)
    returns = ", ".join(f"noisy_bin_{i}" for i in range(num_bins))
    return f'''
# @dp.mechanism(privacy="{epsilon}-dp", sensitivity={sensitivity})
def {name}(db):
    """Noisy histogram MISSING noise on bins 1..{num_bins - 1}.

    BUG: Only bin 0 gets Laplace noise; rest are released raw.
    """
    # @dp.sensitivity({sensitivity})
    counts = histogram_query(db, n_bins={num_bins})
{body}
    return ({returns})
'''


def noisy_histogram_gaussian_source(
    num_bins: int = 5,
    sensitivity: float = 1.0,
    epsilon: float = 1.0,
    delta: float = 1e-5,
    name: str = "gaussian_histogram",
) -> str:
    """Generate source for a Gaussian noisy histogram under (ε,δ)-DP.

    Args:
        num_bins: Number of bins.
        sensitivity: L2 sensitivity (√2 for add/remove, 1 for substitution).
        epsilon: Privacy budget ε.
        delta: Privacy budget δ.
        name: Function name.

    Returns:
        Python source with DPImp annotations.
    """
    import math
    sigma = sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon
    sigma_r = round(sigma, 6)
    bins = []
    for i in range(num_bins):
        bins.append(f"""
    # @dp.noise(kind="gaussian", sigma={sigma_r})
    noise_{i} = gaussian(0, {sigma_r})
    noisy_bin_{i} = counts[{i}] + noise_{i}""")
    body = "\n".join(bins)
    returns = ", ".join(f"noisy_bin_{i}" for i in range(num_bins))
    return f'''
# @dp.mechanism(privacy="({epsilon},{delta})-dp", sensitivity={sensitivity})
def {name}(db):
    """Gaussian noisy histogram with {num_bins} bins.

    Each bin: N(0, σ²) with σ = {sigma_r}.
    """
    # @dp.sensitivity({sensitivity}, norm="L2")
    counts = histogram_query(db, n_bins={num_bins})
{body}
    return ({returns})
'''


# ---------------------------------------------------------------------------
# Pre-built source constants
# ---------------------------------------------------------------------------

HISTOGRAM_CORRECT = noisy_histogram_source(num_bins=5, epsilon=1.0)
"""Correct noisy histogram with 5 bins."""

STABILITY_HISTOGRAM = stability_histogram_source(num_bins=10, epsilon=1.0)
"""Correct stability-based histogram."""

HISTOGRAM_WRONG_SCALE = histogram_wrong_scale_source(num_bins=5, epsilon=1.0)
"""Buggy histogram: scale divided by number of bins."""

HISTOGRAM_MISSING_NOISE = histogram_missing_bin_noise_source(num_bins=5, epsilon=1.0)
"""Buggy histogram: missing noise on bins 1..4."""

GAUSSIAN_HISTOGRAM = noisy_histogram_gaussian_source(num_bins=5, epsilon=1.0)
"""Correct Gaussian noisy histogram."""


# ---------------------------------------------------------------------------
# Benchmark configurations
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HistogramBenchConfig:
    """Configuration for a histogram benchmark instance."""

    name: str
    num_bins: int
    epsilon: float
    is_correct: bool
    variant: str = "standard"
    description: str = ""

    def source(self) -> str:
        """Generate the mechanism source for this configuration."""
        if self.variant == "stability":
            return stability_histogram_source(
                num_bins=self.num_bins, epsilon=self.epsilon, name=self.name,
            )
        if self.variant == "gaussian":
            return noisy_histogram_gaussian_source(
                num_bins=self.num_bins, epsilon=self.epsilon, name=self.name,
            )
        if self.is_correct:
            return noisy_histogram_source(
                num_bins=self.num_bins, epsilon=self.epsilon, name=self.name,
            )
        return histogram_wrong_scale_source(
            num_bins=self.num_bins, epsilon=self.epsilon, name=self.name,
        )


HISTOGRAM_BENCH_CONFIGS: list[HistogramBenchConfig] = [
    HistogramBenchConfig("hist_5", 5, 1.0, True, description="5 bins"),
    HistogramBenchConfig("hist_10", 10, 1.0, True, description="10 bins"),
    HistogramBenchConfig("hist_50", 50, 1.0, True, description="50 bins"),
    HistogramBenchConfig("hist_stab", 10, 1.0, True, "stability", "Stability"),
    HistogramBenchConfig("hist_gauss", 5, 1.0, True, "gaussian", "Gaussian"),
    HistogramBenchConfig("hist_bug", 5, 1.0, False, description="Wrong scale"),
]
"""Benchmark sweep configurations for histogram mechanisms."""
