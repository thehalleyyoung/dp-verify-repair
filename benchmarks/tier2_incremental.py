"""
Tier 2 benchmark: incremental feasibility with prefix caching.

Compares full-recheck feasibility against incremental feasibility
on path conditions with shared prefixes (simulating a CEGAR loop).

Run:
    python -m benchmarks.tier2_incremental
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

from dpcegar.ir.types import BinOp, BinOpKind, Const, IRType, Var
from dpcegar.paths.feasibility import FeasibilityChecker
from dpcegar.paths.symbolic_path import PathCondition


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _var(name: str) -> Var:
    return Var(ty=IRType.REAL, name=name)


def _gt(name: str, val: float) -> BinOp:
    return BinOp(
        ty=IRType.BOOL, op=BinOpKind.GT,
        left=_var(name), right=Const.real(val),
    )


def _lt(name: str, val: float) -> BinOp:
    return BinOp(
        ty=IRType.BOOL, op=BinOpKind.LT,
        left=_var(name), right=Const.real(val),
    )


def _generate_path_conditions(
    n_paths: int, prefix_length: int, suffix_length: int
) -> list[list[BinOp]]:
    """Generate N path conditions sharing a common prefix.

    Each condition has `prefix_length` shared conjuncts (simulating
    the common prefix of a CEGAR loop) followed by `suffix_length`
    unique conjuncts.
    """
    prefix = [_gt(f"p{i}", float(i)) for i in range(prefix_length)]
    conditions: list[list[BinOp]] = []
    for path_idx in range(n_paths):
        suffix = [
            _lt(f"s{path_idx}_{j}", float(100 + j))
            for j in range(suffix_length)
        ]
        conditions.append(prefix + suffix)
    return conditions


@dataclass
class BenchmarkResult:
    label: str
    n_paths: int
    prefix_len: int
    total_time_ms: float
    mean_per_check_us: float


# ---------------------------------------------------------------------------
# Benchmark routines
# ---------------------------------------------------------------------------


def _bench_full_check(conditions: list[list[BinOp]]) -> float:
    """Time full feasibility check (no prefix reuse)."""
    fc = FeasibilityChecker()
    t0 = time.perf_counter()
    for conjuncts in conditions:
        fc.check(PathCondition(conjuncts=conjuncts))
    return time.perf_counter() - t0


def _bench_incremental(conditions: list[list[BinOp]], prefix_len: int) -> float:
    """Time incremental feasibility check with prefix caching."""
    fc = FeasibilityChecker(max_prefix_depth=prefix_len + 10)
    t0 = time.perf_counter()
    for conjuncts in conditions:
        base = conjuncts[:prefix_len]
        for new in conjuncts[prefix_len:]:
            fc.check_incremental(base, new)
            base = base + [new]
    return time.perf_counter() - t0


def run_benchmark(
    n_paths: int = 200,
    prefix_length: int = 10,
    suffix_length: int = 5,
    warmup_rounds: int = 2,
    measure_rounds: int = 5,
) -> tuple[BenchmarkResult, BenchmarkResult]:
    """Run the incremental-vs-full benchmark and return results."""
    conditions = _generate_path_conditions(n_paths, prefix_length, suffix_length)

    # Warm-up
    for _ in range(warmup_rounds):
        _bench_full_check(conditions)
        _bench_incremental(conditions, prefix_length)

    full_times: list[float] = []
    inc_times: list[float] = []
    for _ in range(measure_rounds):
        full_times.append(_bench_full_check(conditions))
        inc_times.append(_bench_incremental(conditions, prefix_length))

    full_median = statistics.median(full_times)
    inc_median = statistics.median(inc_times)

    full_result = BenchmarkResult(
        label="full_check",
        n_paths=n_paths,
        prefix_len=prefix_length,
        total_time_ms=full_median * 1000,
        mean_per_check_us=(full_median / n_paths) * 1e6,
    )
    inc_result = BenchmarkResult(
        label="incremental",
        n_paths=n_paths,
        prefix_len=prefix_length,
        total_time_ms=inc_median * 1000,
        mean_per_check_us=(inc_median / n_paths) * 1e6,
    )
    return full_result, inc_result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    configs = [
        {"n_paths": 100, "prefix_length": 5, "suffix_length": 3},
        {"n_paths": 200, "prefix_length": 10, "suffix_length": 5},
        {"n_paths": 500, "prefix_length": 20, "suffix_length": 5},
    ]

    print("=" * 72)
    print("Tier 2 Benchmark: Incremental Feasibility with Prefix Caching")
    print("=" * 72)

    for cfg in configs:
        full, inc = run_benchmark(**cfg)
        speedup = full.total_time_ms / inc.total_time_ms if inc.total_time_ms > 0 else float("inf")
        print(
            f"\n  N={cfg['n_paths']:>4d}  prefix={cfg['prefix_length']:>2d}  "
            f"suffix={cfg['suffix_length']:>2d}"
        )
        print(f"    full_check : {full.total_time_ms:8.2f} ms  "
              f"({full.mean_per_check_us:8.1f} μs/check)")
        print(f"    incremental: {inc.total_time_ms:8.2f} ms  "
              f"({inc.mean_per_check_us:8.1f} μs/check)")
        print(f"    speedup    : {speedup:.2f}×")

    print("\n" + "=" * 72)


if __name__ == "__main__":
    main()
