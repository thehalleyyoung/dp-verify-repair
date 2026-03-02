"""
End-to-end repair demonstration.

Walks through the full DP-CEGAR repair pipeline:
  1. Parse a buggy mechanism
  2. Verify it (find the bug)
  3. Synthesize a repair via CEGIS
  4. Verify the repaired mechanism
  5. Show before/after code and generate certificate

Run:
    python -m examples.demo_repair
"""

from __future__ import annotations

import sys
import time
from typing import Any

from examples.mechanisms.laplace_mechanism import (
    LAPLACE_WRONG_SCALE,
    LAPLACE_MISSING_NOISE,
    laplace_mechanism_source,
)
from examples.mechanisms.gaussian_mechanism import GAUSSIAN_WRONG_SIGMA
from examples.mechanisms.sparse_vector import (
    SVT_BUG1_NO_THRESHOLD_NOISE,
    SVT_BUG3_WRONG_SENSITIVITY,
    SVT_BUG5_WRONG_BUDGET,
)


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
    """Return a coloured verdict string."""
    colours = {
        "VERIFIED": "\033[92m✓ VERIFIED\033[0m",
        "COUNTEREXAMPLE": "\033[91m✗ COUNTEREXAMPLE\033[0m",
        "SUCCESS": "\033[92m✓ REPAIR FOUND\033[0m",
        "NO_REPAIR": "\033[91m✗ NO REPAIR\033[0m",
        "TIMEOUT": "\033[93m⏱ TIMEOUT\033[0m",
        "ERROR": "\033[91m⚠ ERROR\033[0m",
    }
    return colours.get(verdict_name, verdict_name)


def _diff_sources(original: str, repaired: str) -> str:
    """Generate a simple inline diff between original and repaired source."""
    orig_lines = original.strip().splitlines()
    rep_lines = repaired.strip().splitlines()
    output = []
    max_lines = max(len(orig_lines), len(rep_lines))
    for i in range(max_lines):
        ol = orig_lines[i] if i < len(orig_lines) else ""
        rl = rep_lines[i] if i < len(rep_lines) else ""
        if ol == rl:
            output.append(f"  {rl}")
        else:
            if ol:
                output.append(f"\033[91m- {ol}\033[0m")
            if rl:
                output.append(f"\033[92m+ {rl}\033[0m")
    return "\n".join(output)


# ---------------------------------------------------------------------------
# Core repair workflow
# ---------------------------------------------------------------------------

def verify_and_repair(
    source: str,
    mechanism_name: str,
    privacy_notion: str = "pure_dp",
    epsilon: float = 1.0,
    delta: float = 0.0,
    timeout: float = 120.0,
    verbose: bool = True,
) -> dict[str, Any]:
    """Parse, verify, and repair a buggy mechanism.

    Pipeline:
      1. Parse mechanism
      2. Verify (expect failure)
      3. Synthesize repair
      4. Verify repaired mechanism
      5. Generate certificate

    Args:
        source: Python source with DPImp annotations.
        mechanism_name: Name of the mechanism function.
        privacy_notion: DP notion to check.
        epsilon: Privacy budget ε.
        delta: Privacy budget δ.
        timeout: Timeout for each CEGAR/CEGIS invocation.
        verbose: Print progress.

    Returns:
        Dictionary with repair results.
    """
    from dpcegar.parser.ast_bridge import parse_mechanism
    from dpcegar.ir.types import (
        PrivacyNotion,
        PureBudget,
        ApproxBudget,
        ZCDPBudget,
    )
    from dpcegar.cegar.engine import CEGAREngine, CEGARConfig
    from dpcegar.repair.synthesizer import RepairSynthesizer, SynthesizerConfig
    from dpcegar.repair.validator import RepairValidator
    from dpcegar.smt.solver import Z3Solver

    result: dict[str, Any] = {
        "mechanism": mechanism_name,
        "privacy_notion": privacy_notion,
        "epsilon": epsilon,
    }
    t0 = time.perf_counter()

    # Step 1: Parse
    if verbose:
        print(f"  [1/5] Parsing '{mechanism_name}'...")
    mechir = parse_mechanism(source, mechanism_name)

    # Step 2: Build budget
    budget_map = {
        "pure_dp": lambda: PureBudget(epsilon=epsilon),
        "approx_dp": lambda: ApproxBudget(epsilon=epsilon, delta=delta),
        "zcdp": lambda: ZCDPBudget(rho=epsilon),
    }
    budget = budget_map.get(privacy_notion, budget_map["pure_dp"])()

    # Step 3: Initial verification (expect bug)
    if verbose:
        print("  [2/5] Running initial verification (expecting failure)...")
    solver = Z3Solver()
    config = CEGARConfig(
        max_refinements=50,
        timeout_seconds=timeout,
        initial_abstraction="noise_pattern",
    )
    engine = CEGAREngine(mechir, budget, solver, config)
    verify_result = engine.verify()
    result["initial_verdict"] = verify_result.verdict.name
    result["initial_verified"] = verify_result.is_verified

    if verbose:
        print(f"    Initial verdict: {_verdict_str(verify_result.verdict.name)}")

    if verify_result.is_verified:
        if verbose:
            print("    Mechanism already verified — no repair needed!")
        result["repair_verdict"] = "NOT_NEEDED"
        result["total_time"] = time.perf_counter() - t0
        return result

    # Step 4: Synthesize repair
    if verbose:
        print("  [3/5] Synthesizing repair via CEGIS...")
    t_repair = time.perf_counter()
    synth_config = SynthesizerConfig(
        max_iterations=50,
        timeout_seconds=timeout,
        minimize_cost=True,
    )
    synthesizer = RepairSynthesizer(mechir, budget, solver, synth_config)
    repair_result = synthesizer.synthesize()
    result["repair_time"] = time.perf_counter() - t_repair
    result["repair_verdict"] = repair_result.verdict.name
    result["repair_success"] = repair_result.success

    if verbose:
        print(f"    Repair verdict: {_verdict_str(repair_result.verdict.name)}")

    if not repair_result.success:
        if verbose:
            print("    No repair found within budget/timeout.")
        result["total_time"] = time.perf_counter() - t0
        return result

    # Collect repair details
    result["template_name"] = (
        repair_result.template.name() if repair_result.template else "N/A"
    )
    result["parameter_values"] = repair_result.parameter_values
    result["repair_cost"] = repair_result.repair_cost

    if verbose:
        print(f"    Template: {result['template_name']}")
        print(f"    Parameters: {result['parameter_values']}")
        print(f"    Repair cost: {result['repair_cost']:.4f}")

    # Step 5: Verify repaired mechanism
    if verbose:
        print("  [4/5] Verifying repaired mechanism...")
    repaired_mechir = repair_result.repaired_mechanism
    engine2 = CEGAREngine(repaired_mechir, budget, solver, config)
    verify2 = engine2.verify()
    result["repaired_verified"] = verify2.is_verified
    result["repaired_verdict"] = verify2.verdict.name

    if verbose:
        print(f"    Repaired verdict: {_verdict_str(verify2.verdict.name)}")

    # Step 6: Validate repair
    if verbose:
        print("  [5/5] Validating repair integrity...")
    validator = RepairValidator(mechir, repaired_mechir, budget)
    validation = validator.validate()
    result["validation_passed"] = validation.is_valid

    if verbose:
        if validation.is_valid:
            print("    Validation: \033[92m✓ PASSED\033[0m")
        else:
            print("    Validation: \033[91m✗ FAILED\033[0m")
            for finding in validation.findings:
                print(f"      - {finding}")

    # Statistics
    if hasattr(repair_result, "statistics"):
        stats = repair_result.statistics
        result["stats"] = {
            "cegis_iterations": stats.cegis_iterations,
            "templates_tried": stats.templates_tried,
            "counterexamples_accumulated": stats.counterexamples_accumulated,
            "solver_calls": stats.solver_calls,
            "synthesis_time": stats.synthesis_time,
            "verification_time": stats.verification_time,
        }

    result["total_time"] = time.perf_counter() - t0
    return result


# ---------------------------------------------------------------------------
# Demonstration scenarios
# ---------------------------------------------------------------------------

def demo_laplace_wrong_scale(verbose: bool = True) -> dict[str, Any]:
    """Repair Laplace mechanism with wrong scale."""
    if verbose:
        print(_subheader("Repair: Laplace Wrong Scale"))
        print("  Bug: scale = Δf·ε instead of Δf/ε")
    result = verify_and_repair(
        LAPLACE_WRONG_SCALE,
        "laplace_wrong_scale",
        privacy_notion="pure_dp",
        epsilon=1.0,
        verbose=verbose,
    )
    if verbose and result.get("repair_success"):
        print(f"\n  Repair: change noise scale to {result['parameter_values']}")
    return result


def demo_laplace_missing_noise(verbose: bool = True) -> dict[str, Any]:
    """Attempt repair of Laplace mechanism with missing noise."""
    if verbose:
        print(_subheader("Repair: Laplace Missing Noise"))
        print("  Bug: no noise added (returns raw query answer)")
    result = verify_and_repair(
        LAPLACE_MISSING_NOISE,
        "laplace_missing_noise",
        privacy_notion="pure_dp",
        epsilon=1.0,
        verbose=verbose,
    )
    return result


def demo_gaussian_wrong_sigma(verbose: bool = True) -> dict[str, Any]:
    """Repair Gaussian mechanism with wrong σ."""
    if verbose:
        print(_subheader("Repair: Gaussian Wrong σ"))
        print("  Bug: σ = Δf/ε (missing √(2·ln(1.25/δ)) factor)")
    result = verify_and_repair(
        GAUSSIAN_WRONG_SIGMA,
        "gaussian_wrong_sigma",
        privacy_notion="approx_dp",
        epsilon=1.0,
        delta=1e-5,
        verbose=verbose,
    )
    return result


def demo_svt_no_threshold_noise(verbose: bool = True) -> dict[str, Any]:
    """Repair SVT Bug 1: missing threshold noise."""
    if verbose:
        print(_subheader("Repair: SVT Bug 1 — No Threshold Noise"))
        print("  Bug: threshold T used without Laplace noise")
    result = verify_and_repair(
        SVT_BUG1_NO_THRESHOLD_NOISE,
        "svt_bug1",
        privacy_notion="pure_dp",
        epsilon=1.0,
        verbose=verbose,
    )
    return result


def demo_svt_wrong_budget(verbose: bool = True) -> dict[str, Any]:
    """Repair SVT Bug 5: wrong budget allocation."""
    if verbose:
        print(_subheader("Repair: SVT Bug 5 — Wrong Budget Allocation"))
        print("  Bug: uses ε for both threshold and query noise (2ε total)")
    result = verify_and_repair(
        SVT_BUG5_WRONG_BUDGET,
        "svt_bug5",
        privacy_notion="pure_dp",
        epsilon=1.0,
        verbose=verbose,
    )
    return result


# ---------------------------------------------------------------------------
# Certificate generation
# ---------------------------------------------------------------------------

def _generate_certificate(result: dict[str, Any]) -> str:
    """Generate a textual verification/repair certificate."""
    lines = [
        "╔══════════════════════════════════════════════════════════════╗",
        "║              DP-CEGAR Repair Certificate                   ║",
        "╚══════════════════════════════════════════════════════════════╝",
        "",
        f"  Mechanism:        {result.get('mechanism', 'N/A')}",
        f"  Privacy Notion:   {result.get('privacy_notion', 'N/A')}",
        f"  Budget:           ε = {result.get('epsilon', 'N/A')}",
        "",
        f"  Initial Verdict:  {result.get('initial_verdict', 'N/A')}",
        f"  Repair Template:  {result.get('template_name', 'N/A')}",
        f"  Parameters:       {result.get('parameter_values', {})}",
        f"  Repair Cost:      {result.get('repair_cost', 'N/A')}",
        "",
        f"  Repaired Verdict: {result.get('repaired_verdict', 'N/A')}",
        f"  Validation:       {'PASSED' if result.get('validation_passed') else 'FAILED'}",
        f"  Total Time:       {result.get('total_time', 0):.3f}s",
        "",
    ]
    if result.get("stats"):
        lines.append("  Repair Statistics:")
        for k, v in result["stats"].items():
            lines.append(f"    {k}: {v}")
    lines.extend([
        "",
        "  This certificate attests that the repaired mechanism satisfies",
        "  the stated privacy guarantee as verified by DP-CEGAR.",
        "",
    ])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _summary_table(results: list[dict[str, Any]]) -> str:
    """Format a summary table of repair results."""
    header = (
        f"{'Mechanism':<28} {'Initial':<14} {'Repair':<14} "
        f"{'Verified':<10} {'Time':>8}"
    )
    sep = "-" * len(header)
    rows = [header, sep]
    for r in results:
        init = r.get("initial_verdict", "?")
        repair = r.get("repair_verdict", "?")
        verified = "✓" if r.get("repaired_verified") else (
            "N/A" if r.get("repair_verdict") == "NOT_NEEDED" else "✗"
        )
        rows.append(
            f"{r['mechanism']:<28} {init:<14} {repair:<14} "
            f"{verified:<10} {r.get('total_time', 0):>7.3f}s"
        )
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run all repair demonstrations."""
    print(_header("DP-CEGAR Repair Demo"))
    print("  Demonstrating automated repair of buggy DP mechanisms.\n")

    all_results: list[dict[str, Any]] = []

    # Run each scenario
    all_results.append(demo_laplace_wrong_scale())
    all_results.append(demo_laplace_missing_noise())
    all_results.append(demo_gaussian_wrong_sigma())
    all_results.append(demo_svt_no_threshold_noise())
    all_results.append(demo_svt_wrong_budget())

    # Print certificates for successful repairs
    print(_header("Repair Certificates"))
    for r in all_results:
        if r.get("repair_success"):
            print(_generate_certificate(r))

    # Summary
    print(_header("Summary"))
    print(_summary_table(all_results))

    repaired_count = sum(1 for r in all_results if r.get("repair_success"))
    total_time = sum(r.get("total_time", 0) for r in all_results)
    print(f"\n  Repairs successful: {repaired_count}/{len(all_results)}")
    print(f"  Total time: {total_time:.3f}s")


if __name__ == "__main__":
    main()
