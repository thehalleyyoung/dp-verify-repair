"""
Profiling infrastructure for DP-CEGAR verification pipeline.

Provides tools to:
  - Profile verification runs and identify hotspots
  - Break down solver time by phase (encode, check, refine)
  - Track memory usage across pipeline stages
  - Generate profiling reports

Usage::

    profiler = PipelineProfiler()
    profiler.profile_mechanism(source, name, budget)
    report = profiler.generate_report()
    profiler.print_report(report)

Run:
    python -m benchmarks.profile_analysis
"""

from __future__ import annotations

import cProfile
import io
import pstats
import sys
import time
import tracemalloc
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Profiling data types
# ---------------------------------------------------------------------------

@dataclass
class StageProfile:
    """Profiling data for a single pipeline stage."""

    name: str
    wall_time: float = 0.0
    cpu_time: float = 0.0
    memory_before_kb: float = 0.0
    memory_after_kb: float = 0.0
    memory_peak_kb: float = 0.0
    call_count: int = 0
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def memory_delta_kb(self) -> float:
        """Memory change during this stage."""
        return self.memory_after_kb - self.memory_before_kb


@dataclass
class SolverBreakdown:
    """Detailed breakdown of solver time."""

    encode_time: float = 0.0
    check_time: float = 0.0
    refine_time: float = 0.0
    model_extraction_time: float = 0.0
    total_solver_calls: int = 0
    sat_calls: int = 0
    unsat_calls: int = 0
    timeout_calls: int = 0
    avg_call_time: float = 0.0
    max_call_time: float = 0.0

    @property
    def total_time(self) -> float:
        """Total solver time across all phases."""
        return (
            self.encode_time
            + self.check_time
            + self.refine_time
            + self.model_extraction_time
        )


@dataclass
class ProfileReport:
    """Complete profiling report for a verification run."""

    mechanism_name: str
    total_wall_time: float
    total_cpu_time: float
    peak_memory_kb: float
    stages: list[StageProfile]
    solver: SolverBreakdown
    hotspots: list[dict[str, Any]]
    cprofile_stats: str = ""

    def stage_by_name(self, name: str) -> StageProfile | None:
        """Find a stage profile by name."""
        for s in self.stages:
            if s.name == name:
                return s
        return None


# ---------------------------------------------------------------------------
# Pipeline Profiler
# ---------------------------------------------------------------------------

class PipelineProfiler:
    """Profile the DP-CEGAR verification pipeline.

    Instruments each stage (parse, path enumeration, density building,
    CEGAR loop) with timing and memory measurements.

    Usage::

        profiler = PipelineProfiler()
        report = profiler.profile_mechanism(source, name, budget_kwargs)
        profiler.print_report(report)
    """

    def __init__(self, track_memory: bool = True, cprofile: bool = False):
        """Initialise the profiler.

        Args:
            track_memory: Whether to track memory via tracemalloc.
            cprofile: Whether to also run cProfile for call-level stats.
        """
        self.track_memory = track_memory
        self.use_cprofile = cprofile
        self._reports: list[ProfileReport] = []

    # -- Timing helpers ----------------------------------------------------

    @staticmethod
    def _time_stage(
        name: str,
        fn: Any,
        track_mem: bool = True,
    ) -> tuple[Any, StageProfile]:
        """Time a single pipeline stage.

        Args:
            name: Stage name.
            fn: Callable to execute (no args).
            track_mem: Whether to track memory.

        Returns:
            Tuple of (function result, StageProfile).
        """
        profile = StageProfile(name=name)

        if track_mem:
            tracemalloc.start()
            mem_before = tracemalloc.get_traced_memory()[0]
            profile.memory_before_kb = mem_before / 1024.0

        t_wall_start = time.perf_counter()
        t_cpu_start = time.process_time()

        result = fn()

        t_cpu_end = time.process_time()
        t_wall_end = time.perf_counter()

        profile.wall_time = t_wall_end - t_wall_start
        profile.cpu_time = t_cpu_end - t_cpu_start

        if track_mem:
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            profile.memory_after_kb = current / 1024.0
            profile.memory_peak_kb = peak / 1024.0

        profile.call_count = 1
        return result, profile

    # -- Core profiling ----------------------------------------------------

    def profile_mechanism(
        self,
        source: str,
        mechanism_name: str,
        privacy_notion: str = "pure_dp",
        epsilon: float = 1.0,
        delta: float = 0.0,
        timeout: float = 120.0,
    ) -> ProfileReport:
        """Profile a full verification pipeline run.

        Args:
            source: Mechanism source code.
            mechanism_name: Function name in source.
            privacy_notion: DP notion to verify.
            epsilon: Privacy budget ε.
            delta: Privacy budget δ.
            timeout: CEGAR timeout.

        Returns:
            ProfileReport with detailed stage-by-stage profiling.
        """
        from dpcegar.parser.ast_bridge import parse_mechanism
        from dpcegar.ir.types import (
            PureBudget,
            ApproxBudget,
            ZCDPBudget,
            RDPBudget,
        )
        from dpcegar.paths.enumerator import PathEnumerator
        from dpcegar.density.ratio_builder import DensityRatioBuilder
        from dpcegar.cegar.engine import CEGAREngine, CEGARConfig
        from dpcegar.smt.solver import Z3Solver

        stages: list[StageProfile] = []
        t_total_start = time.perf_counter()
        t_cpu_total_start = time.process_time()

        overall_peak_mem = 0.0

        # Stage 1: Parse
        mechir, parse_prof = self._time_stage(
            "parse",
            lambda: parse_mechanism(source, mechanism_name),
            track_mem=self.track_memory,
        )
        parse_prof.details["num_nodes"] = sum(1 for _ in mechir.all_nodes())
        parse_prof.details["num_noise_draws"] = len(mechir.noise_draws())
        stages.append(parse_prof)
        overall_peak_mem = max(overall_peak_mem, parse_prof.memory_peak_kb)

        # Budget
        budget_map = {
            "pure_dp": lambda: PureBudget(epsilon=epsilon),
            "approx_dp": lambda: ApproxBudget(epsilon=epsilon, delta=delta),
            "zcdp": lambda: ZCDPBudget(rho=epsilon),
            "rdp": lambda: RDPBudget(alpha=2.0, epsilon=epsilon),
        }
        budget = budget_map.get(privacy_notion, budget_map["pure_dp"])()

        # Stage 2: Path enumeration
        def _enumerate():
            enum = PathEnumerator()
            return enum.enumerate(mechir)

        path_set, path_prof = self._time_stage(
            "path_enumeration", _enumerate, self.track_memory,
        )
        path_prof.details["num_paths"] = path_set.size()
        stages.append(path_prof)
        overall_peak_mem = max(overall_peak_mem, path_prof.memory_peak_kb)

        # Stage 3: Density ratio building
        def _build_density():
            builder = DensityRatioBuilder(mechir)
            return builder.build_density_ratios()

        density_result, density_prof = self._time_stage(
            "density_building", _build_density, self.track_memory,
        )
        stages.append(density_prof)
        overall_peak_mem = max(overall_peak_mem, density_prof.memory_peak_kb)

        # Stage 4: CEGAR
        cegar_result = None
        cegar_stats = None

        def _run_cegar():
            nonlocal cegar_result, cegar_stats
            config = CEGARConfig(
                max_refinements=100,
                timeout_seconds=timeout,
                solver_timeout_seconds=30.0,
                initial_abstraction="noise_pattern",
                enable_widening=True,
            )
            solver = Z3Solver()
            engine = CEGAREngine(config=config, solver=solver)
            cegar_result = engine.verify_mechanism(mechir, budget)
            if hasattr(cegar_result, "statistics"):
                cegar_stats = cegar_result.statistics

        _, cegar_prof = self._time_stage(
            "cegar_verification", _run_cegar, self.track_memory,
        )

        if cegar_stats:
            cegar_prof.details["iterations"] = getattr(
                cegar_stats, "iterations", 0
            )
            cegar_prof.details["solver_calls"] = getattr(
                cegar_stats, "solver_calls", 0
            )
            cegar_prof.details["verdict"] = (
                cegar_result.verdict.name if cegar_result else "N/A"
            )
        stages.append(cegar_prof)
        overall_peak_mem = max(overall_peak_mem, cegar_prof.memory_peak_kb)

        # Solver breakdown
        solver_breakdown = SolverBreakdown()
        if cegar_stats:
            solver_breakdown.total_solver_calls = getattr(
                cegar_stats, "solver_calls", 0
            )
            solver_breakdown.check_time = getattr(
                cegar_stats, "solver_time", 0.0
            )
            solver_breakdown.encode_time = getattr(
                cegar_stats, "abstract_verify_time", 0.0
            )
            solver_breakdown.refine_time = getattr(
                cegar_stats, "refinement_time", 0.0
            )
            if solver_breakdown.total_solver_calls > 0:
                solver_breakdown.avg_call_time = (
                    solver_breakdown.check_time
                    / solver_breakdown.total_solver_calls
                )

        # Hotspot analysis
        hotspots = self._analyze_hotspots(stages)

        # cProfile (optional)
        cprofile_output = ""
        if self.use_cprofile:
            cprofile_output = self._run_cprofile(
                source, mechanism_name, budget, timeout
            )

        # Build report
        total_wall = time.perf_counter() - t_total_start
        total_cpu = time.process_time() - t_cpu_total_start

        report = ProfileReport(
            mechanism_name=mechanism_name,
            total_wall_time=total_wall,
            total_cpu_time=total_cpu,
            peak_memory_kb=overall_peak_mem,
            stages=stages,
            solver=solver_breakdown,
            hotspots=hotspots,
            cprofile_stats=cprofile_output,
        )
        self._reports.append(report)
        return report

    # -- Hotspot analysis --------------------------------------------------

    @staticmethod
    def _analyze_hotspots(
        stages: list[StageProfile],
    ) -> list[dict[str, Any]]:
        """Identify the hottest stages by wall time.

        Args:
            stages: List of stage profiles.

        Returns:
            Sorted list of hotspot dicts (name, time, pct).
        """
        total = sum(s.wall_time for s in stages) or 1e-9
        hotspots = []
        for s in stages:
            hotspots.append({
                "name": s.name,
                "wall_time": s.wall_time,
                "pct": s.wall_time / total * 100,
                "memory_delta_kb": s.memory_delta_kb,
            })
        hotspots.sort(key=lambda h: h["wall_time"], reverse=True)
        return hotspots

    # -- cProfile integration ----------------------------------------------

    @staticmethod
    def _run_cprofile(
        source: str,
        mechanism_name: str,
        budget: Any,
        timeout: float,
    ) -> str:
        """Run cProfile on the full pipeline and capture output.

        Returns:
            Formatted cProfile output string.
        """
        from dpcegar.parser.ast_bridge import parse_mechanism
        from dpcegar.cegar.engine import CEGAREngine, CEGARConfig
        from dpcegar.smt.solver import Z3Solver

        profiler = cProfile.Profile()
        profiler.enable()

        mechir = parse_mechanism(source, mechanism_name)
        config = CEGARConfig(
            max_refinements=50,
            timeout_seconds=timeout,
        )
        engine = CEGAREngine(config=config, solver=Z3Solver())
        engine.verify_mechanism(mechir, budget)

        profiler.disable()

        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.sort_stats("cumulative")
        stats.print_stats(30)
        return stream.getvalue()

    # -- Reporting ---------------------------------------------------------

    def print_report(self, report: ProfileReport) -> None:
        """Print a formatted profiling report.

        Args:
            report: ProfileReport to display.
        """
        print(f"\n{'=' * 70}")
        print(f"  Profiling Report: {report.mechanism_name}")
        print(f"{'=' * 70}")
        print(f"  Total wall time:   {report.total_wall_time:.4f}s")
        print(f"  Total CPU time:    {report.total_cpu_time:.4f}s")
        print(f"  Peak memory:       {report.peak_memory_kb:.0f} KB")
        print()

        # Stage breakdown
        print(f"  {'Stage':<22} {'Wall Time':>10} {'CPU Time':>10} "
              f"{'Mem Δ (KB)':>12} {'Pct':>7}")
        print(f"  {'-' * 65}")
        for s in report.stages:
            pct = (
                s.wall_time / report.total_wall_time * 100
                if report.total_wall_time > 0 else 0
            )
            print(
                f"  {s.name:<22} {s.wall_time:>9.4f}s {s.cpu_time:>9.4f}s "
                f"{s.memory_delta_kb:>11.0f} {pct:>6.1f}%"
            )
        print()

        # Solver breakdown
        sv = report.solver
        if sv.total_solver_calls > 0:
            print("  Solver Breakdown:")
            print(f"    Encode time:        {sv.encode_time:.4f}s")
            print(f"    Check (SAT) time:   {sv.check_time:.4f}s")
            print(f"    Refine time:        {sv.refine_time:.4f}s")
            print(f"    Total solver calls: {sv.total_solver_calls}")
            print(f"    Avg call time:      {sv.avg_call_time:.4f}s")
            print()

        # Hotspots
        print("  Hotspots (by wall time):")
        for h in report.hotspots[:5]:
            bar_len = int(h["pct"] / 2)
            bar = "█" * bar_len
            print(
                f"    {h['name']:<22} {h['wall_time']:>9.4f}s "
                f"({h['pct']:>5.1f}%) {bar}"
            )
        print()

        # Stage details
        for s in report.stages:
            if s.details:
                print(f"  {s.name} details:")
                for k, v in s.details.items():
                    print(f"    {k}: {v}")
                print()

        if report.cprofile_stats:
            print("  cProfile Top Functions:")
            print(report.cprofile_stats[:2000])

    def generate_csv_report(
        self,
        reports: list[ProfileReport] | None = None,
        path: str | Path | None = None,
    ) -> str:
        """Generate a CSV profiling report.

        Args:
            reports: Reports to include; None for all collected.
            path: Optional file path to write CSV.

        Returns:
            CSV content as string.
        """
        import csv

        target = reports or self._reports
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow([
            "mechanism", "total_wall_time", "total_cpu_time",
            "peak_memory_kb", "parse_time", "path_enum_time",
            "density_time", "cegar_time", "solver_calls",
            "solver_time", "verdict",
        ])

        for r in target:
            parse = r.stage_by_name("parse")
            paths = r.stage_by_name("path_enumeration")
            density = r.stage_by_name("density_building")
            cegar = r.stage_by_name("cegar_verification")
            writer.writerow([
                r.mechanism_name,
                f"{r.total_wall_time:.6f}",
                f"{r.total_cpu_time:.6f}",
                f"{r.peak_memory_kb:.0f}",
                f"{parse.wall_time:.6f}" if parse else "",
                f"{paths.wall_time:.6f}" if paths else "",
                f"{density.wall_time:.6f}" if density else "",
                f"{cegar.wall_time:.6f}" if cegar else "",
                r.solver.total_solver_calls,
                f"{r.solver.check_time:.6f}",
                cegar.details.get("verdict", "") if cegar else "",
            ])

        content = buf.getvalue()
        if path:
            with open(path, "w") as f:
                f.write(content)
        return content


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def profile_stress_suite() -> list[ProfileReport]:
    """Profile the tier-4 stress benchmarks to identify scaling bottlenecks.

    Runs profiling on representative stress-test mechanisms:
      - Wide branching (path enumeration / density ratio hotspot)
      - Deep composition (composition accounting hotspot)
      - CEGAR iteration (refinement loop hotspot)

    Returns:
        List of ProfileReport objects for each profiled mechanism.
    """
    from benchmarks.tier4_stress import (
        _wide_branching_mechanism,
        _deep_composition_mechanism,
        _subtle_violation_mechanism,
    )

    profiler = PipelineProfiler(track_memory=True, cprofile=False)
    reports: list[ProfileReport] = []

    stress_mechanisms = [
        # (source, name, notion, eps, delta, timeout)
        (
            _wide_branching_mechanism(30, name="wide_branch_30"),
            "wide_branch_30", "pure_dp", 1.0, 0.0, 120.0,
        ),
        (
            _deep_composition_mechanism(150, name="deep_comp_150"),
            "deep_comp_150", "pure_dp", 1.0, 0.0, 120.0,
        ),
        (
            _subtle_violation_mechanism(20, 17, name="subtle_20b_v17"),
            "subtle_20b_v17", "pure_dp", 1.0, 0.0, 120.0,
        ),
    ]

    print("Running stress-suite profiling...")
    for source, name, notion, eps, delta, timeout in stress_mechanisms:
        print(f"\n  Profiling stress case '{name}'...")
        report = profiler.profile_mechanism(
            source, name, notion, eps, delta, timeout=timeout,
        )
        profiler.print_report(report)
        reports.append(report)

    csv_content = profiler.generate_csv_report(reports)
    print("\nStress Profiling CSV Report:")
    print(csv_content)

    return reports


def main() -> None:
    """Run profiling on a set of example mechanisms."""
    import argparse

    from examples.mechanisms.laplace_mechanism import LAPLACE_CORRECT
    from examples.mechanisms.gaussian_mechanism import GAUSSIAN_CORRECT
    from examples.mechanisms.sparse_vector import SVT_CORRECT

    parser = argparse.ArgumentParser(
        description="DP-CEGAR Pipeline Profiler",
    )
    parser.add_argument(
        "--suite",
        choices=["basic", "stress", "all"],
        default="basic",
        help="Profiling suite to run (default: basic).",
    )
    args = parser.parse_args()

    profiler = PipelineProfiler(track_memory=True, cprofile=False)

    if args.suite in ("basic", "all"):
        mechanisms = [
            (LAPLACE_CORRECT, "laplace_mechanism", "pure_dp", 1.0, 0.0),
            (GAUSSIAN_CORRECT, "gaussian_mechanism", "approx_dp", 1.0, 1e-5),
            (SVT_CORRECT, "above_threshold", "pure_dp", 1.0, 0.0),
        ]

        print("Running basic profiling analysis...")
        for source, name, notion, eps, delta in mechanisms:
            print(f"\n  Profiling '{name}'...")
            report = profiler.profile_mechanism(
                source, name, notion, eps, delta, timeout=60.0,
            )
            profiler.print_report(report)

        # CSV output
        csv_content = profiler.generate_csv_report()
        print("\nCSV Report:")
        print(csv_content)

    if args.suite in ("stress", "all"):
        profile_stress_suite()


if __name__ == "__main__":
    main()
