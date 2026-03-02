"""
End-to-end verification demonstration.

Walks through the full DP-CEGAR verification pipeline:
  1. Parse a mechanism from Python source
  2. Build the MechIR intermediate representation
  3. Enumerate symbolic paths
  4. Build density ratio expressions
  5. Run CEGAR verification loop
  6. Print results with formatting

Run:
    python -m examples.demo_verify
"""

from __future__ import annotations

import sys
import time
from typing import Any

from examples.mechanisms.laplace_mechanism import (
    LAPLACE_CORRECT,
    LAPLACE_WRONG_SCALE,
    laplace_mechanism_source,
)
from examples.mechanisms.gaussian_mechanism import (
    GAUSSIAN_CORRECT,
    GAUSSIAN_WRONG_SIGMA,
    gaussian_mechanism_source,
)
from examples.mechanisms.exponential_mechanism import EXPONENTIAL_CORRECT
from examples.mechanisms.sparse_vector import SVT_CORRECT, SVT_BUG1_NO_THRESHOLD_NOISE


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _header(title: str, width: int = 70) -> str:
    """Return a formatted section header."""
    line = "=" * width
    return f"\n{line}\n  {title}\n{line}"


def _subheader(title: str, width: int = 70) -> str:
    """Return a formatted sub-header."""
    line = "-" * width
    return f"\n{line}\n  {title}\n{line}"


def _verdict_str(verdict_name: str) -> str:
    """Return a coloured verdict string (ANSI escape codes)."""
    colours = {
        "VERIFIED": "\033[92m✓ VERIFIED\033[0m",
        "COUNTEREXAMPLE": "\033[91m✗ COUNTEREXAMPLE\033[0m",
        "UNKNOWN": "\033[93m? UNKNOWN\033[0m",
        "TIMEOUT": "\033[93m⏱ TIMEOUT\033[0m",
        "ERROR": "\033[91m⚠ ERROR\033[0m",
    }
    return colours.get(verdict_name, verdict_name)


def _print_stats(stats: dict[str, Any]) -> None:
    """Pretty-print verification statistics."""
    print(f"  CEGAR iterations:      {stats.get('iterations', 'N/A')}")
    print(f"  Refinements:           {stats.get('refinements', 'N/A')}")
    print(f"  Solver calls:          {stats.get('solver_calls', 'N/A')}")
    print(f"  Total time:            {stats.get('total_time', 0):.3f}s")
    print(f"  Abstract verify time:  {stats.get('abstract_verify_time', 0):.3f}s")
    print(f"  Concrete check time:   {stats.get('concrete_check_time', 0):.3f}s")
    print(f"  Refinement time:       {stats.get('refinement_time', 0):.3f}s")
    print(f"  Solver time:           {stats.get('solver_time', 0):.3f}s")
    print(f"  Spurious cex count:    {stats.get('spurious_cex_count', 0)}")
    print(f"  Peak abstract states:  {stats.get('peak_abstract_states', 0)}")


# ---------------------------------------------------------------------------
# Core verification workflow
# ---------------------------------------------------------------------------

def parse_and_verify(
    source: str,
    mechanism_name: str,
    privacy_notion: str = "pure_dp",
    epsilon: float = 1.0,
    delta: float = 0.0,
    timeout: float = 120.0,
    verbose: bool = True,
) -> dict[str, Any]:
    """Parse a mechanism and run CEGAR verification.

    This function encapsulates the full pipeline:
      source → parse → IR → paths → density → CEGAR

    Args:
        source: Python source with DPImp annotations.
        mechanism_name: Name of the mechanism function.
        privacy_notion: One of pure_dp, approx_dp, zcdp, rdp, fdp, gdp.
        epsilon: Privacy budget ε.
        delta: Privacy budget δ (for approx_dp).
        timeout: CEGAR timeout in seconds.
        verbose: Whether to print progress info.

    Returns:
        Dictionary with verification results and timing.
    """
    from dpcegar.parser.ast_bridge import parse_mechanism
    from dpcegar.ir.types import (
        PrivacyNotion,
        PureBudget,
        ApproxBudget,
        ZCDPBudget,
        RDPBudget,
    )
    from dpcegar.paths.enumerator import PathEnumerator
    from dpcegar.density.ratio_builder import DensityRatioBuilder
    from dpcegar.cegar.engine import CEGAREngine, CEGARConfig
    from dpcegar.smt.solver import Z3Solver

    result: dict[str, Any] = {
        "mechanism": mechanism_name,
        "privacy_notion": privacy_notion,
        "epsilon": epsilon,
        "delta": delta,
    }
    t0 = time.perf_counter()

    # Step 1: Parse
    if verbose:
        print(f"  [1/5] Parsing mechanism '{mechanism_name}'...")
    mechir = parse_mechanism(source, mechanism_name)
    result["num_nodes"] = sum(1 for _ in mechir.all_nodes())
    result["num_noise_draws"] = len(mechir.noise_draws())
    result["parse_time"] = time.perf_counter() - t0

    # Step 2: Build privacy budget
    if verbose:
        print(f"  [2/5] Building privacy budget ({privacy_notion})...")
    notion_map = {
        "pure_dp": PrivacyNotion.PURE_DP,
        "approx_dp": PrivacyNotion.APPROX_DP,
        "zcdp": PrivacyNotion.ZCDP,
        "rdp": PrivacyNotion.RDP,
    }
    budget_builders = {
        "pure_dp": lambda: PureBudget(epsilon=epsilon),
        "approx_dp": lambda: ApproxBudget(epsilon=epsilon, delta=delta),
        "zcdp": lambda: ZCDPBudget(rho=epsilon),
        "rdp": lambda: RDPBudget(alpha=2.0, epsilon=epsilon),
    }
    budget = budget_builders.get(privacy_notion, budget_builders["pure_dp"])()
    mechir = mechir  # already have it

    # Step 3: Enumerate paths
    if verbose:
        print("  [3/5] Enumerating symbolic paths...")
    t_paths = time.perf_counter()
    enumerator = PathEnumerator(mechir)
    path_set = enumerator.enumerate()
    result["num_paths"] = sum(1 for _ in path_set.all_ids())
    result["path_enum_time"] = time.perf_counter() - t_paths

    # Step 4: Build density ratios
    if verbose:
        print("  [4/5] Building density ratio expressions...")
    t_density = time.perf_counter()
    ratio_builder = DensityRatioBuilder(mechir)
    density_result = ratio_builder.build_density_ratios()
    result["density_build_time"] = time.perf_counter() - t_density

    # Step 5: Run CEGAR
    if verbose:
        print("  [5/5] Running CEGAR verification loop...")
    t_cegar = time.perf_counter()
    config = CEGARConfig(
        max_refinements=100,
        timeout_seconds=timeout,
        solver_timeout_seconds=30.0,
        initial_abstraction="noise_pattern",
        enable_widening=True,
    )
    solver = Z3Solver()
    engine = CEGAREngine(mechir, budget, solver, config)
    cegar_result = engine.verify()
    result["cegar_time"] = time.perf_counter() - t_cegar

    # Collect results
    result["verdict"] = cegar_result.verdict.name
    result["is_verified"] = cegar_result.is_verified
    result["is_violated"] = cegar_result.is_violated
    result["total_time"] = time.perf_counter() - t0

    if hasattr(cegar_result, "statistics"):
        stats = cegar_result.statistics
        result["stats"] = stats.summary() if hasattr(stats, "summary") else {}
    else:
        result["stats"] = {}

    if cegar_result.is_violated and hasattr(cegar_result, "counterexample"):
        cex = cegar_result.counterexample
        result["counterexample"] = str(cex) if cex else None

    return result


# ---------------------------------------------------------------------------
# Demonstration scenarios
# ---------------------------------------------------------------------------

def demo_laplace_correct(verbose: bool = True) -> dict[str, Any]:
    """Verify a correct Laplace mechanism — should pass."""
    if verbose:
        print(_subheader("Laplace Mechanism (correct, ε=1)"))
    result = parse_and_verify(
        LAPLACE_CORRECT,
        "laplace_mechanism",
        privacy_notion="pure_dp",
        epsilon=1.0,
        verbose=verbose,
    )
    if verbose:
        print(f"\n  Verdict: {_verdict_str(result['verdict'])}")
        print(f"  Nodes: {result['num_nodes']}, Paths: {result['num_paths']}")
        print(f"  Total time: {result['total_time']:.3f}s")
        if result.get("stats"):
            _print_stats(result["stats"])
    return result


def demo_laplace_buggy(verbose: bool = True) -> dict[str, Any]:
    """Verify a Laplace mechanism with wrong scale — should fail."""
    if verbose:
        print(_subheader("Laplace Mechanism (WRONG SCALE, ε=1)"))
    result = parse_and_verify(
        LAPLACE_WRONG_SCALE,
        "laplace_wrong_scale",
        privacy_notion="pure_dp",
        epsilon=1.0,
        verbose=verbose,
    )
    if verbose:
        print(f"\n  Verdict: {_verdict_str(result['verdict'])}")
        if result.get("counterexample"):
            print(f"  Counterexample: {result['counterexample']}")
        print(f"  Total time: {result['total_time']:.3f}s")
    return result


def demo_gaussian_correct(verbose: bool = True) -> dict[str, Any]:
    """Verify a correct Gaussian mechanism under (ε,δ)-DP."""
    if verbose:
        print(_subheader("Gaussian Mechanism (correct, ε=1, δ=1e-5)"))
    result = parse_and_verify(
        GAUSSIAN_CORRECT,
        "gaussian_mechanism",
        privacy_notion="approx_dp",
        epsilon=1.0,
        delta=1e-5,
        verbose=verbose,
    )
    if verbose:
        print(f"\n  Verdict: {_verdict_str(result['verdict'])}")
        print(f"  Total time: {result['total_time']:.3f}s")
    return result


def demo_gaussian_buggy(verbose: bool = True) -> dict[str, Any]:
    """Verify a Gaussian mechanism with wrong sigma — should fail."""
    if verbose:
        print(_subheader("Gaussian Mechanism (WRONG σ)"))
    result = parse_and_verify(
        GAUSSIAN_WRONG_SIGMA,
        "gaussian_wrong_sigma",
        privacy_notion="approx_dp",
        epsilon=1.0,
        delta=1e-5,
        verbose=verbose,
    )
    if verbose:
        print(f"\n  Verdict: {_verdict_str(result['verdict'])}")
        print(f"  Total time: {result['total_time']:.3f}s")
    return result


def demo_exponential(verbose: bool = True) -> dict[str, Any]:
    """Verify a correct exponential mechanism."""
    if verbose:
        print(_subheader("Exponential Mechanism (correct, ε=1)"))
    result = parse_and_verify(
        EXPONENTIAL_CORRECT,
        "exponential_mechanism",
        privacy_notion="pure_dp",
        epsilon=1.0,
        verbose=verbose,
    )
    if verbose:
        print(f"\n  Verdict: {_verdict_str(result['verdict'])}")
        print(f"  Total time: {result['total_time']:.3f}s")
    return result


def demo_svt_correct(verbose: bool = True) -> dict[str, Any]:
    """Verify the correct Above Threshold (SVT) algorithm."""
    if verbose:
        print(_subheader("Sparse Vector Technique (correct)"))
    result = parse_and_verify(
        SVT_CORRECT,
        "above_threshold",
        privacy_notion="pure_dp",
        epsilon=1.0,
        verbose=verbose,
    )
    if verbose:
        print(f"\n  Verdict: {_verdict_str(result['verdict'])}")
        print(f"  Total time: {result['total_time']:.3f}s")
    return result


def demo_svt_buggy(verbose: bool = True) -> dict[str, Any]:
    """Verify SVT Bug 1 (no threshold noise) — should fail."""
    if verbose:
        print(_subheader("SVT Bug 1: No Threshold Noise"))
    result = parse_and_verify(
        SVT_BUG1_NO_THRESHOLD_NOISE,
        "svt_bug1",
        privacy_notion="pure_dp",
        epsilon=1.0,
        verbose=verbose,
    )
    if verbose:
        print(f"\n  Verdict: {_verdict_str(result['verdict'])}")
        if result.get("counterexample"):
            print(f"  Counterexample: {result['counterexample']}")
        print(f"  Total time: {result['total_time']:.3f}s")
    return result


def demo_multi_notion(verbose: bool = True) -> list[dict[str, Any]]:
    """Verify a Gaussian mechanism under multiple privacy notions."""
    if verbose:
        print(_subheader("Multi-Notion Verification: Gaussian Mechanism"))
    notions = [
        ("approx_dp", 1.0, 1e-5),
        ("zcdp", 0.5, 0.0),
        ("rdp", 1.0, 0.0),
    ]
    results = []
    for notion, eps, delt in notions:
        if verbose:
            print(f"\n  → Checking under {notion} (ε={eps}, δ={delt})...")
        r = parse_and_verify(
            GAUSSIAN_CORRECT,
            "gaussian_mechanism",
            privacy_notion=notion,
            epsilon=eps,
            delta=delt,
            verbose=False,
        )
        if verbose:
            print(f"    Verdict: {_verdict_str(r['verdict'])}  "
                  f"({r['total_time']:.3f}s)")
        results.append(r)
    return results


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def _summary_table(results: list[dict[str, Any]]) -> str:
    """Format a summary table of verification results."""
    header = f"{'Mechanism':<30} {'Notion':<12} {'Verdict':<16} {'Time':>8}"
    sep = "-" * len(header)
    rows = [header, sep]
    for r in results:
        verdict = r.get("verdict", "?")
        icon = "✓" if r.get("is_verified") else ("✗" if r.get("is_violated") else "?")
        rows.append(
            f"{r['mechanism']:<30} {r['privacy_notion']:<12} "
            f"{icon} {verdict:<13} {r.get('total_time', 0):>7.3f}s"
        )
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run all verification demonstrations."""
    print(_header("DP-CEGAR Verification Demo"))
    print("  Demonstrating end-to-end verification of DP mechanisms.\n")

    all_results: list[dict[str, Any]] = []

    # Basic mechanism demonstrations
    all_results.append(demo_laplace_correct())
    all_results.append(demo_laplace_buggy())
    all_results.append(demo_gaussian_correct())
    all_results.append(demo_gaussian_buggy())
    all_results.append(demo_exponential())
    all_results.append(demo_svt_correct())
    all_results.append(demo_svt_buggy())

    # Multi-notion
    multi = demo_multi_notion()
    all_results.extend(multi)

    # Summary
    print(_header("Summary"))
    print(_summary_table(all_results))

    # Final statistics
    verified = sum(1 for r in all_results if r.get("is_verified"))
    violated = sum(1 for r in all_results if r.get("is_violated"))
    total_time = sum(r.get("total_time", 0) for r in all_results)
    print(f"\n  Verified: {verified}/{len(all_results)}")
    print(f"  Violations: {violated}/{len(all_results)}")
    print(f"  Total time: {total_time:.3f}s")


if __name__ == "__main__":
    main()
