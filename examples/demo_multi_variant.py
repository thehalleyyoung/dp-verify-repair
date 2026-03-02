"""
Multi-variant checking demonstration.

Demonstrates checking a single DP mechanism against all six supported
privacy notions simultaneously, using the variant checker:
  - Pure ε-DP
  - Approximate (ε,δ)-DP
  - Zero-Concentrated DP (ρ-zCDP)
  - Rényi DP ((α,ε)-RDP)
  - f-DP
  - Gaussian DP (μ-GDP)

Shows the implication lattice among notions and derives guarantees.

Run:
    python -m examples.demo_multi_variant
"""

from __future__ import annotations

import sys
import time
from typing import Any

from examples.mechanisms.laplace_mechanism import LAPLACE_CORRECT
from examples.mechanisms.gaussian_mechanism import (
    GAUSSIAN_CORRECT,
    GAUSSIAN_ZCDP,
    gaussian_mechanism_source,
)
from examples.mechanisms.sparse_vector import SVT_CORRECT


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


def _notion_str(notion: str, verified: bool) -> str:
    """Format a notion with verdict colouring."""
    icon = "\033[92m✓\033[0m" if verified else "\033[91m✗\033[0m"
    return f"  {icon} {notion}"


# ---------------------------------------------------------------------------
# Privacy notion definitions
# ---------------------------------------------------------------------------

PRIVACY_NOTIONS = [
    {
        "name": "Pure ε-DP",
        "key": "pure_dp",
        "description": (
            "For all neighbouring datasets D, D' and all measurable sets S: "
            "Pr[M(D) ∈ S] ≤ eε · Pr[M(D') ∈ S]"
        ),
        "epsilon": 1.0,
        "delta": 0.0,
    },
    {
        "name": "Approximate (ε,δ)-DP",
        "key": "approx_dp",
        "description": (
            "Pr[M(D) ∈ S] ≤ eε · Pr[M(D') ∈ S] + δ"
        ),
        "epsilon": 1.0,
        "delta": 1e-5,
    },
    {
        "name": "Zero-Concentrated DP (ρ-zCDP)",
        "key": "zcdp",
        "description": (
            "For all α > 1: Dα(M(D) || M(D')) ≤ ρα, "
            "where Dα is the Rényi divergence of order α."
        ),
        "epsilon": 0.5,  # interpreted as rho
        "delta": 0.0,
    },
    {
        "name": "Rényi DP ((α,ε)-RDP)",
        "key": "rdp",
        "description": (
            "Dα(M(D) || M(D')) ≤ ε for a specific order α."
        ),
        "epsilon": 1.0,
        "delta": 0.0,
    },
    {
        "name": "f-DP",
        "key": "fdp",
        "description": (
            "Trade-off function: T(M, D, D') ≥ f, where f describes the "
            "type I / type II error trade-off."
        ),
        "epsilon": 1.0,
        "delta": 0.0,
    },
    {
        "name": "Gaussian DP (μ-GDP)",
        "key": "gdp",
        "description": (
            "T(M, D, D') ≥ Gμ, where Gμ is the trade-off function of "
            "testing N(0,1) vs N(μ,1)."
        ),
        "epsilon": 1.0,  # interpreted as mu
        "delta": 0.0,
    },
]


# ---------------------------------------------------------------------------
# Implication lattice
# ---------------------------------------------------------------------------

IMPLICATION_EDGES = [
    ("pure_dp", "approx_dp", "ε-DP ⟹ (ε,0)-DP"),
    ("pure_dp", "zcdp", "ε-DP ⟹ ε²/2-zCDP"),
    ("pure_dp", "rdp", "ε-DP ⟹ (α,ε)-RDP for all α"),
    ("zcdp", "approx_dp", "ρ-zCDP ⟹ (ε,δ)-DP with ε = ρ + 2√(ρ·ln(1/δ))"),
    ("zcdp", "rdp", "ρ-zCDP ⟹ (α, ρα)-RDP"),
    ("rdp", "approx_dp", "(α,ε)-RDP ⟹ (ε - ln(δ)/(α-1), δ)-DP"),
    ("gdp", "fdp", "μ-GDP ⟹ f-DP with f = Gμ"),
    ("gdp", "approx_dp", "μ-GDP ⟹ (ε,δ(ε))-DP via Gaussian CDF"),
]


def print_implication_lattice() -> None:
    """Print the privacy notion implication lattice."""
    print("\n  Implication Lattice:")
    print("  ┌─────────────────────────────────────────────────────────┐")
    print("  │                    Pure ε-DP                           │")
    print("  │                   /    |    \\                          │")
    print("  │                  /     |     \\                         │")
    print("  │              zCDP    RDP    GDP                        │")
    print("  │                \\     /       |                         │")
    print("  │                 \\   /        |                         │")
    print("  │              Approx DP     f-DP                        │")
    print("  └─────────────────────────────────────────────────────────┘")
    print()
    for src, dst, desc in IMPLICATION_EDGES:
        print(f"  {desc}")


# ---------------------------------------------------------------------------
# Multi-notion verification
# ---------------------------------------------------------------------------

def check_all_notions(
    source: str,
    mechanism_name: str,
    verbose: bool = True,
) -> dict[str, Any]:
    """Check a mechanism against all six privacy notions.

    Args:
        source: Python source with DPImp annotations.
        mechanism_name: Name of the mechanism function.
        verbose: Print progress.

    Returns:
        Dictionary mapping notion key → verification result.
    """
    from dpcegar.parser.ast_bridge import parse_mechanism
    from dpcegar.ir.types import (
        PrivacyNotion,
        PureBudget,
        ApproxBudget,
        ZCDPBudget,
        RDPBudget,
    )
    from dpcegar.cegar.engine import CEGAREngine, CEGARConfig
    from dpcegar.smt.solver import Z3Solver

    mechir = parse_mechanism(source, mechanism_name)
    solver = Z3Solver()
    config = CEGARConfig(
        max_refinements=50,
        timeout_seconds=60.0,
        initial_abstraction="noise_pattern",
    )

    budget_builders = {
        "pure_dp": lambda n: PureBudget(epsilon=n["epsilon"]),
        "approx_dp": lambda n: ApproxBudget(
            epsilon=n["epsilon"], delta=n["delta"]
        ),
        "zcdp": lambda n: ZCDPBudget(rho=n["epsilon"]),
        "rdp": lambda n: RDPBudget(alpha=2.0, epsilon=n["epsilon"]),
        "fdp": lambda n: ApproxBudget(
            epsilon=n["epsilon"], delta=1e-5
        ),
        "gdp": lambda n: ApproxBudget(
            epsilon=n["epsilon"], delta=1e-5
        ),
    }

    results: dict[str, Any] = {}
    for notion in PRIVACY_NOTIONS:
        key = notion["key"]
        if verbose:
            print(f"  Checking {notion['name']}...", end=" ", flush=True)
        t0 = time.perf_counter()
        budget = budget_builders[key](notion)
        engine = CEGAREngine(mechir, budget, solver, config)
        cegar_result = engine.verify()
        elapsed = time.perf_counter() - t0

        entry = {
            "notion": notion["name"],
            "key": key,
            "verdict": cegar_result.verdict.name,
            "verified": cegar_result.is_verified,
            "time": elapsed,
        }
        results[key] = entry

        if verbose:
            icon = "✓" if cegar_result.is_verified else "✗"
            colour = "\033[92m" if cegar_result.is_verified else "\033[91m"
            print(f"{colour}{icon}\033[0m ({elapsed:.3f}s)")

    return results


def derive_guarantees(
    results: dict[str, Any],
    verbose: bool = True,
) -> list[str]:
    """Derive additional guarantees from verified notions via implications.

    If a stronger notion is verified, all weaker notions hold automatically.

    Args:
        results: Mapping from notion key to verification result.
        verbose: Print derived guarantees.

    Returns:
        List of derived guarantee descriptions.
    """
    derived: list[str] = []
    for src, dst, desc in IMPLICATION_EDGES:
        src_entry = results.get(src, {})
        dst_entry = results.get(dst, {})
        if src_entry.get("verified") and not dst_entry.get("verified"):
            derived.append(f"  DERIVED: {desc}")
            if verbose:
                print(f"  \033[94m→ Derived:\033[0m {desc}")
    return derived


# ---------------------------------------------------------------------------
# Demonstration scenarios
# ---------------------------------------------------------------------------

def demo_laplace_all_notions(verbose: bool = True) -> dict[str, Any]:
    """Check Laplace mechanism against all 6 notions."""
    if verbose:
        print(_subheader("Laplace Mechanism — All Notions"))
    results = check_all_notions(LAPLACE_CORRECT, "laplace_mechanism", verbose)
    if verbose:
        derive_guarantees(results, verbose)
    return results


def demo_gaussian_all_notions(verbose: bool = True) -> dict[str, Any]:
    """Check Gaussian mechanism against all 6 notions."""
    if verbose:
        print(_subheader("Gaussian Mechanism — All Notions"))
    results = check_all_notions(GAUSSIAN_CORRECT, "gaussian_mechanism", verbose)
    if verbose:
        derive_guarantees(results, verbose)
    return results


def demo_svt_all_notions(verbose: bool = True) -> dict[str, Any]:
    """Check SVT mechanism against all 6 notions."""
    if verbose:
        print(_subheader("Sparse Vector Technique — All Notions"))
    results = check_all_notions(SVT_CORRECT, "above_threshold", verbose)
    if verbose:
        derive_guarantees(results, verbose)
    return results


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def _notion_matrix(
    all_results: dict[str, dict[str, Any]],
) -> str:
    """Format a matrix of mechanisms × notions."""
    notions = [n["key"] for n in PRIVACY_NOTIONS]
    notion_short = {
        "pure_dp": "PureDP",
        "approx_dp": "ApxDP",
        "zcdp": "zCDP",
        "rdp": "RDP",
        "fdp": "fDP",
        "gdp": "GDP",
    }
    header_parts = [f"{'Mechanism':<20}"]
    for n in notions:
        header_parts.append(f"{notion_short[n]:>8}")
    header = " ".join(header_parts)
    sep = "-" * len(header)
    rows = [header, sep]
    for mech_name, results in all_results.items():
        parts = [f"{mech_name:<20}"]
        for n in notions:
            entry = results.get(n, {})
            if entry.get("verified"):
                parts.append(f"{'✓':>8}")
            else:
                parts.append(f"{'✗':>8}")
        rows.append(" ".join(parts))
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run multi-variant checking demonstrations."""
    print(_header("DP-CEGAR Multi-Variant Checking Demo"))
    print("  Checking mechanisms against all 6 DP notions.\n")

    # Show the lattice
    print_implication_lattice()

    # Run checks
    all_results: dict[str, dict[str, Any]] = {}

    all_results["Laplace"] = demo_laplace_all_notions()
    all_results["Gaussian"] = demo_gaussian_all_notions()
    all_results["SVT"] = demo_svt_all_notions()

    # Summary matrix
    print(_header("Notion Verification Matrix"))
    print(_notion_matrix(all_results))

    # Total stats
    total_checks = sum(len(r) for r in all_results.values())
    verified_checks = sum(
        sum(1 for v in r.values() if v.get("verified"))
        for r in all_results.values()
    )
    total_time = sum(
        sum(v.get("time", 0) for v in r.values())
        for r in all_results.values()
    )
    print(f"\n  Total checks: {total_checks}")
    print(f"  Verified: {verified_checks}")
    print(f"  Total time: {total_time:.3f}s")


if __name__ == "__main__":
    main()
