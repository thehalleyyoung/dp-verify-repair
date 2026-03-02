"""
Iterative Mechanism implementations for DP-CEGAR verification.

Iterative DP mechanisms apply noise repeatedly over multiple rounds,
accumulating privacy cost.  Common examples include:

  - Noisy gradient descent (DP-GD)
  - DP-SGD (Abadi et al., 2016)
  - Private iterative optimization

Budget accounting across iterations is critical and a frequent source
of bugs.

References:
  Abadi, Chu, Goodfellow, McMahan, Mironov, Talwar, Zhang. "Deep Learning
    with Differential Privacy." CCS 2016.
  Bassily, Smith, Thakurta. "Private Empirical Risk Minimization." FOCS 2014.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Source generators
# ---------------------------------------------------------------------------

def noisy_gradient_descent_source(
    num_iterations: int = 10,
    learning_rate: float = 0.1,
    sensitivity: float = 1.0,
    epsilon: float = 1.0,
    delta: float = 1e-5,
    name: str = "noisy_gd",
) -> str:
    """Generate source for noisy gradient descent.

    At each iteration, computes the gradient on the full dataset, clips to
    sensitivity bound, and adds Gaussian noise.  Uses (ε,δ)-DP per
    iteration via composition.

    Args:
        num_iterations: Number of GD iterations T.
        learning_rate: Step size η.
        sensitivity: Gradient clipping bound C.
        epsilon: Total privacy budget ε.
        delta: Total privacy budget δ.
        name: Function name.

    Returns:
        Python source with DPImp annotations.
    """
    # Per-iteration budget via advanced composition
    delta_per = delta / (2.0 * num_iterations)
    eps_per = epsilon / math.sqrt(2.0 * num_iterations * math.log(1.0 / delta_per))
    eps_per_r = round(eps_per, 6)
    sigma = sensitivity * math.sqrt(2.0 * math.log(1.25 / delta_per)) / eps_per
    sigma_r = round(sigma, 6)
    return f'''
# @dp.mechanism(privacy="({epsilon},{delta})-dp", sensitivity={sensitivity})
def {name}(db, initial_model, gradient_fn):
    """Noisy gradient descent: {num_iterations} iterations.

    Per-iteration: ε_i = {eps_per_r}, σ = {sigma_r}, η = {learning_rate}.
    Gradient clipping bound: {sensitivity}.
    """
    model = initial_model
    for t in range({num_iterations}):
        # @dp.sensitivity({sensitivity})
        grad = gradient_fn(db, model)
        # Clip gradient to sensitivity bound
        grad_norm = sqrt(dot(grad, grad))
        if grad_norm > {sensitivity}:
            grad = grad * ({sensitivity} / grad_norm)
        # @dp.noise(kind="gaussian", sigma={sigma_r})
        noise = gaussian(0, {sigma_r})
        noisy_grad = grad + noise
        model = model - {learning_rate} * noisy_grad
    return model
'''


def dp_sgd_source(
    num_iterations: int = 10,
    batch_size: int = 32,
    learning_rate: float = 0.01,
    clip_norm: float = 1.0,
    noise_multiplier: float = 1.1,
    name: str = "dp_sgd",
) -> str:
    """Generate source for simplified DP-SGD (Abadi et al., 2016).

    Per iteration:
      1. Sample a mini-batch via Poisson sampling (rate q = batch_size / n).
      2. Clip per-sample gradients to norm C.
      3. Average and add N(0, σ²C²) noise, where σ = noise_multiplier.

    Args:
        num_iterations: Number of SGD iterations.
        batch_size: Mini-batch size.
        learning_rate: Step size η.
        clip_norm: Per-sample gradient clipping norm C.
        noise_multiplier: σ (noise multiplier).
        name: Function name.

    Returns:
        Python source with DPImp annotations.
    """
    sigma_actual = noise_multiplier * clip_norm
    sigma_r = round(sigma_actual, 6)
    return f'''
# @dp.mechanism(privacy="rdp", sensitivity={clip_norm})
def {name}(db, initial_model, loss_fn):
    """Simplified DP-SGD (Abadi et al., 2016).

    Iterations: {num_iterations}, batch: {batch_size}, C={clip_norm}, σ={noise_multiplier}.
    Per-iteration noise: N(0, (σC)²) = N(0, {sigma_r}²).
    """
    model = initial_model
    n = len(db)
    q = {batch_size} / n  # Sampling rate
    for t in range({num_iterations}):
        # Poisson sub-sampling
        batch = poisson_sample(db, q)
        # Per-sample gradient computation and clipping
        clipped_grads = []
        for sample in batch:
            # @dp.sensitivity({clip_norm})
            g = gradient(loss_fn, model, sample)
            g_norm = sqrt(dot(g, g))
            if g_norm > {clip_norm}:
                g = g * ({clip_norm} / g_norm)
            clipped_grads.append(g)
        # Average gradients
        avg_grad = mean(clipped_grads)
        # @dp.noise(kind="gaussian", sigma={sigma_r})
        noise = gaussian(0, {sigma_r})
        noisy_grad = avg_grad + noise / {batch_size}
        model = model - {learning_rate} * noisy_grad
    return model
'''


def noisy_gd_wrong_composition_source(
    num_iterations: int = 10,
    sensitivity: float = 1.0,
    epsilon: float = 1.0,
    delta: float = 1e-5,
    name: str = "noisy_gd_wrong_comp",
) -> str:
    """Generate source for noisy GD with wrong composition accounting.

    BUG: Uses total ε for each iteration instead of splitting. The mechanism
    claims ε-DP but actually uses T·ε.

    Args:
        num_iterations: Number of GD iterations.
        sensitivity: Gradient clipping bound.
        epsilon: Claimed total privacy budget.
        delta: Privacy parameter δ.
        name: Function name.

    Returns:
        Python source with DPImp annotations.
    """
    delta_per = delta / num_iterations
    # BUG: using full epsilon per iteration
    sigma = sensitivity * math.sqrt(2.0 * math.log(1.25 / delta_per)) / epsilon
    sigma_r = round(sigma, 6)
    return f'''
# @dp.mechanism(privacy="({epsilon},{delta})-dp", sensitivity={sensitivity})
def {name}(db, initial_model, gradient_fn):
    """Noisy GD with WRONG composition accounting.

    BUG: Uses ε = {epsilon} per iteration instead of ε/√T.
    Actual cost = {num_iterations}·ε = {num_iterations * epsilon} (naive) or more.
    """
    model = initial_model
    for t in range({num_iterations}):
        # @dp.sensitivity({sensitivity})
        grad = gradient_fn(db, model)
        grad_norm = sqrt(dot(grad, grad))
        if grad_norm > {sensitivity}:
            grad = grad * ({sensitivity} / grad_norm)
        # BUG: sigma calibrated to full epsilon, not per-iteration
        # @dp.noise(kind="gaussian", sigma={sigma_r})
        noise = gaussian(0, {sigma_r})
        noisy_grad = grad + noise
        model = model - 0.1 * noisy_grad
    return model
'''


def noisy_gd_no_clipping_source(
    num_iterations: int = 5,
    sensitivity: float = 1.0,
    epsilon: float = 1.0,
    delta: float = 1e-5,
    name: str = "noisy_gd_no_clip",
) -> str:
    """Generate source for noisy GD without gradient clipping.

    BUG: Omits gradient clipping, so the actual sensitivity is unbounded.
    The noise calibration assumes bounded sensitivity, but gradients can
    be arbitrarily large.

    Args:
        num_iterations: Number of iterations.
        sensitivity: Assumed (but not enforced) gradient bound.
        epsilon: Privacy budget.
        delta: Privacy budget δ.
        name: Function name.

    Returns:
        Python source with DPImp annotations.
    """
    delta_per = delta / num_iterations
    eps_per = epsilon / math.sqrt(2.0 * num_iterations * math.log(1.0 / delta_per))
    sigma = sensitivity * math.sqrt(2.0 * math.log(1.25 / delta_per)) / eps_per
    sigma_r = round(sigma, 6)
    return f'''
# @dp.mechanism(privacy="({epsilon},{delta})-dp", sensitivity={sensitivity})
def {name}(db, initial_model, gradient_fn):
    """Noisy GD with NO gradient clipping.

    BUG: Sensitivity is assumed to be {sensitivity} but gradient is not clipped,
    so actual sensitivity can be unbounded.
    """
    model = initial_model
    for t in range({num_iterations}):
        # @dp.sensitivity({sensitivity})
        grad = gradient_fn(db, model)
        # BUG: no clipping! Gradient norm can exceed sensitivity bound.
        # @dp.noise(kind="gaussian", sigma={sigma_r})
        noise = gaussian(0, {sigma_r})
        noisy_grad = grad + noise
        model = model - 0.1 * noisy_grad
    return model
'''


def private_mean_estimation_source(
    num_dimensions: int = 3,
    sensitivity: float = 1.0,
    epsilon: float = 1.0,
    name: str = "private_mean",
) -> str:
    """Generate source for private mean estimation.

    Computes the mean of a bounded dataset with Laplace noise.
    Sensitivity = range / n  for mean queries.

    Args:
        num_dimensions: Number of dimensions.
        sensitivity: Sensitivity of the mean query (after clamping).
        epsilon: Privacy budget.
        name: Function name.

    Returns:
        Python source with DPImp annotations.
    """
    eps_per = epsilon / num_dimensions
    scale = sensitivity / eps_per
    scale_r = round(scale, 6)
    dims = []
    for i in range(num_dimensions):
        dims.append(f"""
    # @dp.sensitivity({sensitivity})
    mean_{i} = mean_query(db, dim={i})
    # @dp.noise(kind="laplace", scale={scale_r})
    noisy_mean_{i} = mean_{i} + laplace(0, {scale_r})""")
    body = "\n".join(dims)
    returns = ", ".join(f"noisy_mean_{i}" for i in range(num_dimensions))
    return f'''
# @dp.mechanism(privacy="{epsilon}-dp", sensitivity={sensitivity})
def {name}(db):
    """Private mean estimation in {num_dimensions} dimensions.

    Per-dimension: ε/d = {eps_per:.4f}, scale = {scale_r}.
    """
{body}
    return ({returns})
'''


# ---------------------------------------------------------------------------
# Pre-built source constants
# ---------------------------------------------------------------------------

NOISY_GD_CORRECT = noisy_gradient_descent_source(
    num_iterations=10, epsilon=1.0, delta=1e-5
)
"""Correct noisy gradient descent with 10 iterations."""

DP_SGD = dp_sgd_source(num_iterations=10)
"""Simplified DP-SGD implementation."""

NOISY_GD_WRONG_COMPOSITION = noisy_gd_wrong_composition_source(
    num_iterations=10, epsilon=1.0
)
"""Buggy noisy GD: wrong composition accounting."""

NOISY_GD_NO_CLIPPING = noisy_gd_no_clipping_source(
    num_iterations=5, epsilon=1.0
)
"""Buggy noisy GD: missing gradient clipping."""

PRIVATE_MEAN = private_mean_estimation_source(
    num_dimensions=3, epsilon=1.0
)
"""Correct private mean estimation in 3 dimensions."""


# ---------------------------------------------------------------------------
# Benchmark configurations
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IterativeBenchConfig:
    """Configuration for an iterative mechanism benchmark."""

    name: str
    num_iterations: int
    epsilon: float
    is_correct: bool
    variant: str = "noisy_gd"
    description: str = ""

    def source(self) -> str:
        """Generate the mechanism source for this configuration."""
        if self.variant == "dp_sgd":
            return dp_sgd_source(
                num_iterations=self.num_iterations, name=self.name,
            )
        if self.variant == "noisy_gd_wrong":
            return noisy_gd_wrong_composition_source(
                num_iterations=self.num_iterations,
                epsilon=self.epsilon,
                name=self.name,
            )
        if self.variant == "noisy_gd_no_clip":
            return noisy_gd_no_clipping_source(
                num_iterations=self.num_iterations,
                epsilon=self.epsilon,
                name=self.name,
            )
        return noisy_gradient_descent_source(
            num_iterations=self.num_iterations,
            epsilon=self.epsilon,
            name=self.name,
        )


ITERATIVE_BENCH_CONFIGS: list[IterativeBenchConfig] = [
    IterativeBenchConfig("gd_5", 5, 1.0, True, description="5 iterations"),
    IterativeBenchConfig("gd_10", 10, 1.0, True, description="10 iterations"),
    IterativeBenchConfig("gd_50", 50, 1.0, True, description="50 iterations"),
    IterativeBenchConfig("sgd_10", 10, 1.0, True, "dp_sgd", "DP-SGD 10 iter."),
    IterativeBenchConfig("gd_bug_comp", 10, 1.0, False, "noisy_gd_wrong",
                         "Wrong composition"),
    IterativeBenchConfig("gd_bug_clip", 5, 1.0, False, "noisy_gd_no_clip",
                         "No clipping"),
]
"""Benchmark configurations for iterative mechanisms."""
