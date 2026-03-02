"""Performance timing decorators and context managers.

Lightweight instrumentation for profiling DP-CEGAR pipeline stages.
All timing data is collected into a global registry that can be queried
or dumped at the end of a run.
"""

from __future__ import annotations

import functools
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Generator, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


# ---------------------------------------------------------------------------
# Timing record
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class TimingRecord:
    """Accumulated timing statistics for a named span."""

    name: str
    calls: int = 0
    total_seconds: float = 0.0
    min_seconds: float = float("inf")
    max_seconds: float = 0.0

    @property
    def avg_seconds(self) -> float:
        """Mean wall-clock time per call."""
        return self.total_seconds / self.calls if self.calls else 0.0

    def record(self, elapsed: float) -> None:
        """Record a single observation."""
        self.calls += 1
        self.total_seconds += elapsed
        if elapsed < self.min_seconds:
            self.min_seconds = elapsed
        if elapsed > self.max_seconds:
            self.max_seconds = elapsed

    def __str__(self) -> str:
        return (
            f"{self.name}: calls={self.calls}  "
            f"total={self.total_seconds:.4f}s  "
            f"avg={self.avg_seconds:.4f}s  "
            f"min={self.min_seconds:.4f}s  "
            f"max={self.max_seconds:.4f}s"
        )


# ---------------------------------------------------------------------------
# Global registry
# ---------------------------------------------------------------------------

class TimingRegistry:
    """Thread-local-ish registry of timing records.

    In the future this can be made truly thread-safe with a lock;
    for now DP-CEGAR is single-threaded.
    """

    __slots__ = ("_records",)

    def __init__(self) -> None:
        self._records: dict[str, TimingRecord] = {}

    def get(self, name: str) -> TimingRecord:
        """Return (or create) the record for *name*."""
        if name not in self._records:
            self._records[name] = TimingRecord(name=name)
        return self._records[name]

    def record(self, name: str, elapsed: float) -> None:
        """Record an observation under *name*."""
        self.get(name).record(elapsed)

    def clear(self) -> None:
        """Drop all accumulated records."""
        self._records.clear()

    def summary(self) -> list[TimingRecord]:
        """Return all records sorted by total time descending."""
        return sorted(
            self._records.values(),
            key=lambda r: r.total_seconds,
            reverse=True,
        )

    def dump(self) -> str:
        """Human-readable summary string."""
        lines = ["=== Timing Summary ==="]
        for rec in self.summary():
            lines.append(f"  {rec}")
        return "\n".join(lines)


_GLOBAL_REGISTRY = TimingRegistry()


def global_registry() -> TimingRegistry:
    """Return the process-wide timing registry."""
    return _GLOBAL_REGISTRY


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------

def timed(name: str | None = None, registry: TimingRegistry | None = None) -> Callable[[F], F]:
    """Decorator that records wall-clock time of each call.

    Args:
        name: Metric name.  Defaults to ``module.qualname`` of the function.
        registry: Where to store the record.  Defaults to the global registry.

    Usage::

        @timed()
        def expensive_solve(formula):
            ...

        @timed("cegar.refine")
        def refine(abstraction, cex):
            ...
    """
    reg = registry or _GLOBAL_REGISTRY

    def decorator(fn: F) -> F:
        span_name = name or f"{fn.__module__}.{fn.__qualname__}"

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                reg.record(span_name, time.perf_counter() - t0)

        return wrapper  # type: ignore[return-value]

    return decorator


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------

@contextmanager
def timed_block(
    name: str,
    registry: TimingRegistry | None = None,
) -> Generator[None, None, None]:
    """Context manager variant of :func:`timed`.

    Usage::

        with timed_block("smt.solve"):
            solver.check()
    """
    reg = registry or _GLOBAL_REGISTRY
    t0 = time.perf_counter()
    try:
        yield
    finally:
        reg.record(name, time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# One-shot timer
# ---------------------------------------------------------------------------

class Stopwatch:
    """Simple stopwatch for manual start/stop instrumentation."""

    __slots__ = ("_start", "_elapsed")

    def __init__(self) -> None:
        self._start: float | None = None
        self._elapsed: float = 0.0

    def start(self) -> Stopwatch:
        """Start (or restart) the stopwatch."""
        self._start = time.perf_counter()
        return self

    def stop(self) -> float:
        """Stop the stopwatch and return elapsed seconds since start."""
        if self._start is None:
            return 0.0
        self._elapsed = time.perf_counter() - self._start
        self._start = None
        return self._elapsed

    @property
    def elapsed(self) -> float:
        """Elapsed seconds (updated on :meth:`stop`)."""
        if self._start is not None:
            return time.perf_counter() - self._start
        return self._elapsed
