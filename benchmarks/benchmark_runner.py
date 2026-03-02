"""
Benchmark runner infrastructure for DP-CEGAR verification and repair.

Provides a ``BenchmarkRunner`` class that:
  - Runs benchmark suites with timing, memory tracking
  - Outputs results as CSV
  - Computes statistical summaries (mean, median, p95)
  - Compares against baseline measurements
  - Reports progress

Run:
    python -m benchmarks.benchmark_runner --suite tier1
"""

from __future__ import annotations

import csv
import gc
import io
import math
import os
import statistics
import sys
import time
import traceback
import tracemalloc
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Protocol


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkCase:
    """A single benchmark case to be run."""

    id: str
    name: str
    category: str
    source: str
    mechanism_name: str
    privacy_notion: str = "pure_dp"
    epsilon: float = 1.0
    delta: float = 0.0
    expected_verified: bool | None = None
    timeout: float = 120.0
    description: str = ""
    tags: list[str] = field(default_factory=list)


@dataclass
class BenchmarkMeasurement:
    """Raw measurement from a single benchmark run."""

    case_id: str
    iteration: int
    wall_time: float
    parse_time: float = 0.0
    path_enum_time: float = 0.0
    density_build_time: float = 0.0
    cegar_time: float = 0.0
    solver_time: float = 0.0
    peak_memory_kb: float = 0.0
    cegar_iterations: int = 0
    solver_calls: int = 0
    num_paths: int = 0
    num_nodes: int = 0
    verdict: str = ""
    verified: bool | None = None
    error: str | None = None


@dataclass
class BenchmarkStats:
    """Statistical summary of benchmark measurements."""

    case_id: str
    case_name: str
    category: str
    num_runs: int
    mean_time: float
    median_time: float
    min_time: float
    max_time: float
    std_time: float
    p95_time: float
    mean_memory_kb: float
    mean_solver_time: float
    mean_cegar_iterations: float
    verdict: str
    expected_pass: bool | None
    actual_pass: bool | None

    @property
    def matches_expected(self) -> bool | None:
        """Whether actual result matches expected."""
        if self.expected_pass is None or self.actual_pass is None:
            return None
        return self.expected_pass == self.actual_pass


@dataclass
class BaselineEntry:
    """A baseline measurement for comparison."""

    case_id: str
    mean_time: float
    median_time: float
    verdict: str


# ---------------------------------------------------------------------------
# Progress callback protocol
# ---------------------------------------------------------------------------

class ProgressCallback(Protocol):
    """Protocol for progress reporting."""

    def __call__(
        self,
        case_id: str,
        case_num: int,
        total_cases: int,
        iteration: int,
        total_iterations: int,
    ) -> None: ...


def default_progress(
    case_id: str,
    case_num: int,
    total_cases: int,
    iteration: int,
    total_iterations: int,
) -> None:
    """Default progress reporter: print to stderr."""
    pct = (case_num - 1) / total_cases * 100 if total_cases > 0 else 0
    sys.stderr.write(
        f"\r  [{pct:5.1f}%] {case_id} "
        f"(run {iteration}/{total_iterations}) "
        f"[{case_num}/{total_cases}]"
    )
    sys.stderr.flush()


# ---------------------------------------------------------------------------
# BenchmarkRunner
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """Run DP-CEGAR benchmarks with measurement and reporting.

    Usage::

        runner = BenchmarkRunner(iterations=3)
        runner.add_case(BenchmarkCase(...))
        results = runner.run()
        runner.write_csv(results, "results.csv")
        stats = runner.compute_stats(results)
        runner.print_summary(stats)
    """

    def __init__(
        self,
        iterations: int = 3,
        warmup: int = 1,
        timeout: float = 120.0,
        track_memory: bool = True,
        progress: ProgressCallback | None = None,
    ) -> None:
        """Initialise the benchmark runner.

        Args:
            iterations: Number of timed iterations per case.
            warmup: Number of warm-up iterations (not measured).
            timeout: Default per-case timeout in seconds.
            track_memory: Whether to track peak memory via tracemalloc.
            progress: Progress callback; None for default stderr reporter.
        """
        self.iterations = iterations
        self.warmup = warmup
        self.timeout = timeout
        self.track_memory = track_memory
        self.progress = progress or default_progress
        self._cases: list[BenchmarkCase] = []
        self._baselines: dict[str, BaselineEntry] = {}

    # -- Case management ---------------------------------------------------

    def add_case(self, case: BenchmarkCase) -> None:
        """Add a benchmark case."""
        self._cases.append(case)

    def add_cases(self, cases: list[BenchmarkCase]) -> None:
        """Add multiple benchmark cases."""
        self._cases.extend(cases)

    @property
    def num_cases(self) -> int:
        """Number of registered cases."""
        return len(self._cases)

    def filter_cases(self, category: str | None = None, tags: list[str] | None = None):
        """Filter cases by category and/or tags."""
        filtered = self._cases
        if category:
            filtered = [c for c in filtered if c.category == category]
        if tags:
            tag_set = set(tags)
            filtered = [c for c in filtered if tag_set.intersection(c.tags)]
        return filtered

    # -- Baseline management -----------------------------------------------

    def load_baselines(self, path: str | Path) -> None:
        """Load baseline measurements from a CSV file.

        Args:
            path: Path to CSV with columns: case_id, mean_time, median_time, verdict.
        """
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                entry = BaselineEntry(
                    case_id=row["case_id"],
                    mean_time=float(row["mean_time"]),
                    median_time=float(row["median_time"]),
                    verdict=row.get("verdict", ""),
                )
                self._baselines[entry.case_id] = entry

    # -- Core benchmark execution ------------------------------------------

    def _run_single(self, case: BenchmarkCase) -> BenchmarkMeasurement:
        """Execute a single benchmark measurement.

        Args:
            case: The benchmark case to run.

        Returns:
            A BenchmarkMeasurement with timing and result data.
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

        m = BenchmarkMeasurement(case_id=case.id, iteration=0, wall_time=0.0)

        gc.collect()
        if self.track_memory:
            tracemalloc.start()

        t_start = time.perf_counter()

        try:
            # Parse
            t0 = time.perf_counter()
            mechir = parse_mechanism(case.source, case.mechanism_name)
            m.parse_time = time.perf_counter() - t0
            m.num_nodes = sum(1 for _ in mechir.all_nodes())

            # Budget
            budget_map = {
                "pure_dp": lambda: PureBudget(epsilon=case.epsilon),
                "approx_dp": lambda: ApproxBudget(
                    epsilon=case.epsilon, delta=case.delta
                ),
                "zcdp": lambda: ZCDPBudget(rho=case.epsilon),
                "rdp": lambda: RDPBudget(alpha=2.0, epsilon=case.epsilon),
            }
            budget = budget_map.get(
                case.privacy_notion, budget_map["pure_dp"]
            )()

            # Paths
            t0 = time.perf_counter()
            enumerator = PathEnumerator()
            path_set = enumerator.enumerate(mechir)
            m.path_enum_time = time.perf_counter() - t0
            m.num_paths = path_set.size()

            # Density
            t0 = time.perf_counter()
            builder = DensityRatioBuilder(mechir)
            builder.build_density_ratios()
            m.density_build_time = time.perf_counter() - t0

            # CEGAR
            t0 = time.perf_counter()
            config = CEGARConfig(
                max_refinements=100,
                timeout_seconds=case.timeout,
                solver_timeout_seconds=30.0,
                initial_abstraction="noise_pattern",
                enable_widening=True,
            )
            solver = Z3Solver()
            engine = CEGAREngine(config=config, solver=solver)
            result = engine.verify_mechanism(mechir, budget)
            m.cegar_time = time.perf_counter() - t0

            m.verdict = result.verdict.name
            m.verified = result.is_verified

            if hasattr(result, "statistics"):
                stats = result.statistics
                m.cegar_iterations = getattr(stats, "iterations", 0)
                m.solver_calls = getattr(stats, "solver_calls", 0)
                m.solver_time = getattr(stats, "solver_time", 0.0)

        except Exception as exc:
            m.error = f"{type(exc).__name__}: {exc}"
            m.verdict = "ERROR"

        m.wall_time = time.perf_counter() - t_start

        if self.track_memory:
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            m.peak_memory_kb = peak / 1024.0

        return m

    def run(
        self,
        cases: list[BenchmarkCase] | None = None,
    ) -> list[BenchmarkMeasurement]:
        """Run all benchmark cases.

        Args:
            cases: Specific cases to run; None for all registered cases.

        Returns:
            List of all measurements (warmup excluded).
        """
        target_cases = cases or self._cases
        if not target_cases:
            sys.stderr.write("  No benchmark cases to run.\n")
            return []

        measurements: list[BenchmarkMeasurement] = []
        total = len(target_cases)

        for case_num, case in enumerate(target_cases, 1):
            # Warm-up
            for w in range(self.warmup):
                self.progress(case.id, case_num, total, w + 1, self.warmup)
                self._run_single(case)

            # Timed runs
            for it in range(self.iterations):
                self.progress(
                    case.id, case_num, total, it + 1, self.iterations
                )
                m = self._run_single(case)
                m.iteration = it + 1
                measurements.append(m)

        sys.stderr.write("\n")
        return measurements

    # -- Statistics --------------------------------------------------------

    def compute_stats(
        self,
        measurements: list[BenchmarkMeasurement],
    ) -> list[BenchmarkStats]:
        """Compute statistical summaries grouped by case ID.

        Args:
            measurements: Raw measurements from run().

        Returns:
            List of BenchmarkStats, one per case.
        """
        grouped: dict[str, list[BenchmarkMeasurement]] = defaultdict(list)
        for m in measurements:
            grouped[m.case_id].append(m)

        case_map = {c.id: c for c in self._cases}
        stats_list: list[BenchmarkStats] = []

        for case_id, runs in sorted(grouped.items()):
            times = [r.wall_time for r in runs]
            mem = [r.peak_memory_kb for r in runs]
            solver = [r.solver_time for r in runs]
            iters = [r.cegar_iterations for r in runs]

            case = case_map.get(case_id)
            n = len(times)

            # Compute percentile 95
            sorted_times = sorted(times)
            p95_idx = max(0, int(math.ceil(0.95 * n)) - 1)
            p95 = sorted_times[p95_idx] if sorted_times else 0.0

            entry = BenchmarkStats(
                case_id=case_id,
                case_name=case.name if case else case_id,
                category=case.category if case else "",
                num_runs=n,
                mean_time=statistics.mean(times) if times else 0.0,
                median_time=statistics.median(times) if times else 0.0,
                min_time=min(times) if times else 0.0,
                max_time=max(times) if times else 0.0,
                std_time=statistics.stdev(times) if len(times) > 1 else 0.0,
                p95_time=p95,
                mean_memory_kb=statistics.mean(mem) if mem else 0.0,
                mean_solver_time=statistics.mean(solver) if solver else 0.0,
                mean_cegar_iterations=(
                    statistics.mean(iters) if iters else 0.0
                ),
                verdict=runs[0].verdict if runs else "",
                expected_pass=(
                    case.expected_verified if case else None
                ),
                actual_pass=runs[0].verified if runs else None,
            )
            stats_list.append(entry)

        return stats_list

    # -- CSV output --------------------------------------------------------

    def write_csv(
        self,
        measurements: list[BenchmarkMeasurement],
        path: str | Path,
    ) -> None:
        """Write raw measurements to a CSV file.

        Args:
            measurements: Raw measurements from run().
            path: Output file path.
        """
        fieldnames = [
            "case_id", "iteration", "wall_time", "parse_time",
            "path_enum_time", "density_build_time", "cegar_time",
            "solver_time", "peak_memory_kb", "cegar_iterations",
            "solver_calls", "num_paths", "num_nodes", "verdict",
            "verified", "error",
        ]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for m in measurements:
                writer.writerow({
                    "case_id": m.case_id,
                    "iteration": m.iteration,
                    "wall_time": f"{m.wall_time:.6f}",
                    "parse_time": f"{m.parse_time:.6f}",
                    "path_enum_time": f"{m.path_enum_time:.6f}",
                    "density_build_time": f"{m.density_build_time:.6f}",
                    "cegar_time": f"{m.cegar_time:.6f}",
                    "solver_time": f"{m.solver_time:.6f}",
                    "peak_memory_kb": f"{m.peak_memory_kb:.1f}",
                    "cegar_iterations": m.cegar_iterations,
                    "solver_calls": m.solver_calls,
                    "num_paths": m.num_paths,
                    "num_nodes": m.num_nodes,
                    "verdict": m.verdict,
                    "verified": m.verified,
                    "error": m.error or "",
                })

    def write_stats_csv(
        self,
        stats: list[BenchmarkStats],
        path: str | Path,
    ) -> None:
        """Write statistical summaries to a CSV file.

        Args:
            stats: Stats computed via compute_stats().
            path: Output file path.
        """
        fieldnames = [
            "case_id", "case_name", "category", "num_runs",
            "mean_time", "median_time", "min_time", "max_time",
            "std_time", "p95_time", "mean_memory_kb", "mean_solver_time",
            "mean_cegar_iterations", "verdict", "expected_pass",
            "actual_pass", "matches_expected",
        ]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for s in stats:
                writer.writerow({
                    "case_id": s.case_id,
                    "case_name": s.case_name,
                    "category": s.category,
                    "num_runs": s.num_runs,
                    "mean_time": f"{s.mean_time:.6f}",
                    "median_time": f"{s.median_time:.6f}",
                    "min_time": f"{s.min_time:.6f}",
                    "max_time": f"{s.max_time:.6f}",
                    "std_time": f"{s.std_time:.6f}",
                    "p95_time": f"{s.p95_time:.6f}",
                    "mean_memory_kb": f"{s.mean_memory_kb:.1f}",
                    "mean_solver_time": f"{s.mean_solver_time:.6f}",
                    "mean_cegar_iterations": f"{s.mean_cegar_iterations:.1f}",
                    "verdict": s.verdict,
                    "expected_pass": s.expected_pass,
                    "actual_pass": s.actual_pass,
                    "matches_expected": s.matches_expected,
                })

    # -- Baseline comparison -----------------------------------------------

    def compare_to_baselines(
        self,
        stats: list[BenchmarkStats],
    ) -> list[dict[str, Any]]:
        """Compare current stats against loaded baselines.

        Args:
            stats: Current benchmark stats.

        Returns:
            List of comparison dicts with speedup/slowdown info.
        """
        comparisons: list[dict[str, Any]] = []
        for s in stats:
            baseline = self._baselines.get(s.case_id)
            if baseline is None:
                continue
            speedup = (
                baseline.mean_time / s.mean_time
                if s.mean_time > 0 else float("inf")
            )
            comparisons.append({
                "case_id": s.case_id,
                "baseline_mean": baseline.mean_time,
                "current_mean": s.mean_time,
                "speedup": speedup,
                "verdict_changed": baseline.verdict != s.verdict,
                "baseline_verdict": baseline.verdict,
                "current_verdict": s.verdict,
            })
        return comparisons

    # -- Reporting ---------------------------------------------------------

    def print_summary(self, stats: list[BenchmarkStats]) -> None:
        """Print a human-readable summary of benchmark results.

        Args:
            stats: Computed benchmark statistics.
        """
        header = (
            f"{'Case':<28} {'Verdict':<14} {'Mean':>8} {'Med':>8} "
            f"{'P95':>8} {'Mem(KB)':>10} {'Match':>6}"
        )
        sep = "-" * len(header)
        print(f"\n{sep}")
        print(f"  Benchmark Summary ({len(stats)} cases)")
        print(sep)
        print(header)
        print(sep)

        for s in stats:
            match_icon = ""
            if s.matches_expected is True:
                match_icon = "✓"
            elif s.matches_expected is False:
                match_icon = "✗"
            else:
                match_icon = "-"

            print(
                f"{s.case_name:<28} {s.verdict:<14} "
                f"{s.mean_time:>7.3f}s {s.median_time:>7.3f}s "
                f"{s.p95_time:>7.3f}s {s.mean_memory_kb:>9.0f} "
                f"{match_icon:>6}"
            )
        print(sep)

        # Aggregate stats
        total_time = sum(s.mean_time for s in stats)
        correct_count = sum(
            1 for s in stats if s.matches_expected is True
        )
        total_expected = sum(
            1 for s in stats if s.matches_expected is not None
        )
        print(f"  Total mean time:  {total_time:.3f}s")
        print(
            f"  Correctness:      {correct_count}/{total_expected} "
            f"match expected"
        )

    def print_baseline_comparison(
        self,
        comparisons: list[dict[str, Any]],
    ) -> None:
        """Print baseline comparison results.

        Args:
            comparisons: Output of compare_to_baselines().
        """
        if not comparisons:
            print("  No baseline comparisons available.")
            return

        header = (
            f"{'Case':<28} {'Baseline':>10} {'Current':>10} "
            f"{'Speedup':>10} {'Verdict':>10}"
        )
        sep = "-" * len(header)
        print(f"\n{sep}")
        print("  Baseline Comparison")
        print(sep)
        print(header)
        print(sep)
        for c in comparisons:
            verdict_str = (
                "CHANGED" if c["verdict_changed"] else "same"
            )
            print(
                f"{c['case_id']:<28} "
                f"{c['baseline_mean']:>9.3f}s {c['current_mean']:>9.3f}s "
                f"{c['speedup']:>9.2f}x {verdict_str:>10}"
            )
        print(sep)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run benchmarks from the command line."""
    import argparse

    parser = argparse.ArgumentParser(
        description="DP-CEGAR Benchmark Runner"
    )
    parser.add_argument(
        "--suite", choices=["tier1", "tier2", "tier3", "tier4", "all"],
        default="tier1", help="Benchmark suite to run."
    )
    parser.add_argument(
        "--iterations", type=int, default=3,
        help="Number of timed iterations per case."
    )
    parser.add_argument(
        "--warmup", type=int, default=1,
        help="Number of warm-up iterations."
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="CSV output file path."
    )
    parser.add_argument(
        "--baseline", type=str, default=None,
        help="Path to baseline CSV for comparison."
    )
    args = parser.parse_args()

    runner = BenchmarkRunner(
        iterations=args.iterations,
        warmup=args.warmup,
    )

    # Import benchmark suites
    if args.suite in ("tier1", "all"):
        from benchmarks.tier1_basic import get_tier1_cases
        runner.add_cases(get_tier1_cases())
    if args.suite in ("tier2", "all"):
        from benchmarks.tier2_composition import get_tier2_cases
        runner.add_cases(get_tier2_cases())
    if args.suite in ("tier3", "all"):
        from benchmarks.tier3_repair import get_tier3_cases
        runner.add_cases(get_tier3_cases())
    if args.suite in ("tier4", "all"):
        from benchmarks.tier4_stress import get_tier4_cases
        runner.add_cases(get_tier4_cases())

    if args.baseline:
        runner.load_baselines(args.baseline)

    print(f"Running {runner.num_cases} benchmark cases "
          f"({args.iterations} iterations each)...")
    measurements = runner.run()

    if args.output:
        runner.write_csv(measurements, args.output)
        print(f"  Raw results written to {args.output}")

    stats = runner.compute_stats(measurements)
    runner.print_summary(stats)

    if args.baseline:
        comparisons = runner.compare_to_baselines(stats)
        runner.print_baseline_comparison(comparisons)


if __name__ == "__main__":
    main()
