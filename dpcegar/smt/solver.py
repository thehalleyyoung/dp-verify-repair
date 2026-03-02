"""SMT solver management for differential privacy verification.

Provides wrapper classes around Z3 (and optionally CVC5) with:
  - Incremental solving with push/pop
  - Timeout management
  - Proof and unsat-core extraction
  - Model extraction and interpretation
  - Portfolio solving (run Z3 + CVC5 in parallel)
  - Statistics tracking
"""

from __future__ import annotations

import enum
import logging
import time
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

try:
    import z3
except ImportError:  # pragma: no cover
    z3 = None  # type: ignore[assignment]

from dpcegar.smt.encoding import SMTEncoding
from dpcegar.smt.theory_selection import SMTTheory, auto_configure, TheoryAnalyzer


# ═══════════════════════════════════════════════════════════════════════════
# SOLVER RESULT
# ═══════════════════════════════════════════════════════════════════════════


class CheckResult(enum.Enum):
    """Possible outcomes of a satisfiability check."""

    SAT = "sat"
    UNSAT = "unsat"
    UNKNOWN = "unknown"
    TIMEOUT = "timeout"
    ERROR = "error"

    @property
    def is_sat(self) -> bool:
        """Return True if the result is satisfiable."""
        return self == CheckResult.SAT

    @property
    def is_unsat(self) -> bool:
        """Return True if the result is unsatisfiable."""
        return self == CheckResult.UNSAT

    @property
    def is_definitive(self) -> bool:
        """Return True if the result is SAT or UNSAT."""
        return self in (CheckResult.SAT, CheckResult.UNSAT)


@dataclass(slots=True)
class SolverResult:
    """Result of a satisfiability check.

    Attributes:
        result:      SAT, UNSAT, UNKNOWN, TIMEOUT, or ERROR.
        model:       Z3 model if SAT (None otherwise).
        proof:       Z3 proof if UNSAT and proof generation enabled.
        unsat_core:  Unsat core if UNSAT and core tracking enabled.
        stats:       Solver statistics.
        solve_time:  Wall-clock solve time in seconds.
        solver_name: Which solver produced this result.
        theory:      SMT theory used.
    """

    result: CheckResult = CheckResult.UNKNOWN
    model: Any = None  # z3.ModelRef
    proof: Any = None  # z3.ExprRef
    unsat_core: list[Any] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    solve_time: float = 0.0
    solver_name: str = "z3"
    theory: SMTTheory = SMTTheory.QF_LRA

    @property
    def is_sat(self) -> bool:
        """Return True if satisfiable."""
        return self.result.is_sat

    @property
    def is_unsat(self) -> bool:
        """Return True if unsatisfiable."""
        return self.result.is_unsat

    def get_value(self, var_name: str) -> Any:
        """Extract the value of a variable from the model.

        Args:
            var_name: Variable name.

        Returns:
            The model value, or None if not available.
        """
        if self.model is None:
            return None
        try:
            # Try as Real
            v = z3.Real(var_name)
            val = self.model.eval(v, model_completion=True)
            return val
        except Exception:
            try:
                v = z3.Int(var_name)
                val = self.model.eval(v, model_completion=True)
                return val
            except Exception:
                return None

    def get_float(self, var_name: str) -> float | None:
        """Extract the value of a variable as a Python float.

        Args:
            var_name: Variable name.

        Returns:
            Float value, or None if not available.
        """
        val = self.get_value(var_name)
        if val is None:
            return None
        try:
            if z3.is_rational_value(val):
                return float(val.as_fraction())
            if z3.is_int_value(val):
                return float(val.as_long())
            if z3.is_algebraic_value(val):
                return float(val.approx(20))
            return float(str(val))
        except (ValueError, AttributeError):
            return None

    def model_values(self, var_names: Sequence[str]) -> dict[str, Any]:
        """Extract model values for multiple variables.

        Args:
            var_names: Variable names to extract.

        Returns:
            Dictionary mapping variable names to model values.
        """
        return {name: self.get_value(name) for name in var_names}

    def summary(self) -> str:
        """Return a human-readable summary.

        Returns:
            Summary string.
        """
        parts = [
            f"result={self.result.value}",
            f"time={self.solve_time:.3f}s",
            f"solver={self.solver_name}",
        ]
        if self.model is not None:
            parts.append(f"model_vars={len(self.model.decls())}")
        if self.unsat_core:
            parts.append(f"core_size={len(self.unsat_core)}")
        return ", ".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# SOLVER CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class SolverConfig:
    """Configuration for SMT solver instances.

    Attributes:
        timeout_ms:     Solver timeout in milliseconds.
        theory:         SMT theory to use.
        enable_proof:   Enable proof generation for UNSAT results.
        enable_core:    Enable unsat core extraction.
        incremental:    Use incremental solving (push/pop).
        portfolio:      Use portfolio solving (Z3 + CVC5).
        seed:           Random seed for the solver.
        max_memory_mb:  Memory limit in MB.
        tactics:        Z3 tactics to apply.
        extra_params:   Additional solver parameters.
    """

    timeout_ms: int = 30000
    theory: SMTTheory = SMTTheory.QF_LRA
    enable_proof: bool = False
    enable_core: bool = False
    incremental: bool = False
    portfolio: bool = False
    seed: int = 42
    max_memory_mb: int = 4096
    tactics: list[str] = field(default_factory=list)
    extra_params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def default(cls) -> SolverConfig:
        """Create a default solver configuration.

        Returns:
            Default SolverConfig.
        """
        return cls()

    @classmethod
    def for_verification(cls) -> SolverConfig:
        """Create a configuration optimised for verification queries.

        Returns:
            SolverConfig with proof and core enabled.
        """
        return cls(
            timeout_ms=60000,
            enable_proof=True,
            enable_core=True,
            incremental=True,
        )

    @classmethod
    def for_counterexample(cls) -> SolverConfig:
        """Create a configuration optimised for counterexample finding.

        Returns:
            SolverConfig with model generation focus.
        """
        return cls(
            timeout_ms=30000,
            enable_proof=False,
            enable_core=False,
        )


# ═══════════════════════════════════════════════════════════════════════════
# SOLVER STATISTICS
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class SolverStats:
    """Aggregate solver statistics.

    Attributes:
        total_queries:     Total number of queries.
        sat_count:         Number of SAT results.
        unsat_count:       Number of UNSAT results.
        unknown_count:     Number of UNKNOWN results.
        timeout_count:     Number of timeouts.
        total_time:        Total solving time in seconds.
        max_time:          Maximum single query time.
        theory_counts:     Count of queries per theory.
        restart_count:     Number of solver restarts.
    """

    total_queries: int = 0
    sat_count: int = 0
    unsat_count: int = 0
    unknown_count: int = 0
    timeout_count: int = 0
    total_time: float = 0.0
    max_time: float = 0.0
    theory_counts: dict[str, int] = field(default_factory=dict)
    restart_count: int = 0

    def record(self, result: SolverResult) -> None:
        """Record statistics from a solver result.

        Args:
            result: The solver result to record.
        """
        self.total_queries += 1
        self.total_time += result.solve_time
        self.max_time = max(self.max_time, result.solve_time)

        if result.result == CheckResult.SAT:
            self.sat_count += 1
        elif result.result == CheckResult.UNSAT:
            self.unsat_count += 1
        elif result.result == CheckResult.TIMEOUT:
            self.timeout_count += 1
        else:
            self.unknown_count += 1

        theory_key = result.theory.value
        self.theory_counts[theory_key] = self.theory_counts.get(theory_key, 0) + 1

    @property
    def avg_time(self) -> float:
        """Return average solve time per query.

        Returns:
            Average time in seconds.
        """
        if self.total_queries == 0:
            return 0.0
        return self.total_time / self.total_queries

    def summary(self) -> dict[str, Any]:
        """Return a summary dictionary.

        Returns:
            Dictionary with statistics.
        """
        return {
            "total_queries": self.total_queries,
            "sat": self.sat_count,
            "unsat": self.unsat_count,
            "unknown": self.unknown_count,
            "timeout": self.timeout_count,
            "total_time": round(self.total_time, 3),
            "avg_time": round(self.avg_time, 3),
            "max_time": round(self.max_time, 3),
            "theories": dict(self.theory_counts),
            "restarts": self.restart_count,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Z3 SOLVER WRAPPER
# ═══════════════════════════════════════════════════════════════════════════


class Z3Solver:
    """Wrapper around z3.Solver with enhanced functionality.

    Provides:
      - Incremental solving with push/pop
      - Timeout management
      - Proof extraction
      - Model extraction and interpretation
      - Unsat core extraction

    Args:
        config: Solver configuration.
    """

    def __init__(self, config: SolverConfig | None = None) -> None:
        self._config = config or SolverConfig.default()
        self._solver = self._create_solver()
        self._stats = SolverStats()
        self._push_count = 0
        self._encoding_cache: dict[int, SolverResult] = {}

    def _create_solver(self) -> Any:
        """Create and configure a Z3 solver.

        Returns:
            Configured z3.Solver instance.
        """
        if self._config.enable_proof:
            z3.set_param("proof", True)

        if self._config.tactics:
            try:
                tactic = z3.Then(*[z3.Tactic(t) for t in self._config.tactics])
                solver = tactic.solver()
            except Exception:
                solver = z3.Solver()
        else:
            solver = z3.Solver()

        solver.set("timeout", self._config.timeout_ms)
        if self._config.seed is not None:
            solver.set("random_seed", self._config.seed)

        for key, val in self._config.extra_params.items():
            try:
                solver.set(key, val)
            except Exception:
                logger.debug("Failed to set solver param %s=%s", key, val, exc_info=True)

        return solver

    @property
    def config(self) -> SolverConfig:
        """Return the solver configuration.

        Returns:
            SolverConfig.
        """
        return self._config

    @property
    def stats(self) -> SolverStats:
        """Return aggregate solver statistics.

        Returns:
            SolverStats.
        """
        return self._stats

    def add(self, *constraints: Any) -> None:
        """Add constraints to the solver.

        Args:
            constraints: Z3 boolean expressions to assert.
        """
        for c in constraints:
            self._solver.add(c)

    def add_encoding(self, encoding: SMTEncoding) -> None:
        """Add all assertions from an SMT encoding.

        Args:
            encoding: The encoding to add.
        """
        for assertion in encoding.assertions:
            self._solver.add(assertion)

    def assert_and_track(self, constraint: Any, label: str) -> None:
        """Add a constraint with tracking for unsat core extraction.

        Args:
            constraint: Z3 boolean expression.
            label:      Label for the constraint (used in unsat core).
        """
        p = z3.Bool(label)
        self._solver.assert_and_track(constraint, p)

    def push(self) -> None:
        """Create a new scope for incremental solving.

        All assertions added after push() are removed by pop().
        """
        self._solver.push()
        self._push_count += 1

    def pop(self, n: int = 1) -> None:
        """Remove the most recent n scopes.

        Args:
            n: Number of scopes to remove.
        """
        actual = min(n, self._push_count)
        if actual > 0:
            self._solver.pop(actual)
            self._push_count -= actual

    def reset(self) -> None:
        """Reset the solver to its initial state."""
        self._solver.reset()
        self._push_count = 0
        self._stats.restart_count += 1

    def check(self, *assumptions: Any) -> SolverResult:
        """Check satisfiability of the current assertions.

        Args:
            assumptions: Optional assumptions for the check.

        Returns:
            SolverResult with the outcome.
        """
        start = time.monotonic()
        try:
            if assumptions:
                z3_result = self._solver.check(*assumptions)
            else:
                z3_result = self._solver.check()
        except z3.Z3Exception as e:
            elapsed = time.monotonic() - start
            result = SolverResult(
                result=CheckResult.ERROR,
                solve_time=elapsed,
                solver_name="z3",
                theory=self._config.theory,
                stats={"error": str(e)},
            )
            self._stats.record(result)
            return result

        elapsed = time.monotonic() - start

        if z3_result == z3.sat:
            result_enum = CheckResult.SAT
            model = self._solver.model()
        elif z3_result == z3.unsat:
            result_enum = CheckResult.UNSAT
            model = None
        else:
            result_enum = CheckResult.UNKNOWN
            model = None

        # Extract proof if available
        proof = None
        if result_enum == CheckResult.UNSAT and self._config.enable_proof:
            try:
                proof = self._solver.proof()
            except Exception:
                logger.debug("Failed to extract proof", exc_info=True)

        # Extract unsat core if available
        core: list[Any] = []
        if result_enum == CheckResult.UNSAT and self._config.enable_core:
            try:
                core = list(self._solver.unsat_core())
            except Exception:
                logger.debug("Failed to extract unsat core", exc_info=True)

        # Extract statistics
        stats_dict: dict[str, Any] = {}
        try:
            z3_stats = self._solver.statistics()
            for k in z3_stats.keys():
                stats_dict[k] = z3_stats.get_key_value(k)
        except Exception:
            logger.debug("Failed to extract solver statistics", exc_info=True)

        result = SolverResult(
            result=result_enum,
            model=model,
            proof=proof,
            unsat_core=core,
            stats=stats_dict,
            solve_time=elapsed,
            solver_name="z3",
            theory=self._config.theory,
        )
        self._stats.record(result)
        return result

    def check_encoding(self, encoding: SMTEncoding) -> SolverResult:
        """Check an SMT encoding for satisfiability.

        Pushes a new scope, adds the encoding, checks, and pops.
        Caches definitive (SAT/UNSAT) results keyed by assertion hash.

        Args:
            encoding: The encoding to check.

        Returns:
            SolverResult.
        """
        cache_key = hash(tuple(str(a) for a in encoding.assertions))
        cached = self._encoding_cache.get(cache_key)
        if cached is not None:
            self._stats.record(cached)
            return cached

        self.push()
        try:
            self.add_encoding(encoding)
            result = self.check()
        finally:
            self.pop()

        if result.result.is_definitive:
            self._encoding_cache[cache_key] = result
        return result

    def clear_cache(self) -> None:
        """Clear the encoding result cache."""
        self._encoding_cache.clear()

    def to_smt2(self) -> str:
        """Export the current assertions as SMT-LIB2 string.

        Returns:
            SMT-LIB2 string representation.
        """
        return self._solver.to_smt2()

    def assertions_list(self) -> list[Any]:
        """Return the current list of assertions.

        Returns:
            List of Z3 expressions.
        """
        return list(self._solver.assertions())

    def num_assertions(self) -> int:
        """Return the number of current assertions.

        Returns:
            Assertion count.
        """
        return len(self._solver.assertions())

    def model_to_dict(self, result: SolverResult) -> dict[str, float]:
        """Extract all model values as a Python dictionary.

        Args:
            result: A SAT solver result with a model.

        Returns:
            Dictionary mapping variable names to float values.
        """
        if result.model is None:
            return {}

        values: dict[str, float] = {}
        for decl in result.model.decls():
            name = decl.name()
            val = result.model[decl]
            try:
                if z3.is_rational_value(val):
                    values[name] = float(val.as_fraction())
                elif z3.is_int_value(val):
                    values[name] = float(val.as_long())
                elif z3.is_algebraic_value(val):
                    values[name] = float(val.approx(20))
                elif z3.is_true(val):
                    values[name] = 1.0
                elif z3.is_false(val):
                    values[name] = 0.0
                else:
                    values[name] = float(str(val))
            except (ValueError, AttributeError):
                logger.debug("Failed to convert model value for %s", name)

        return values


# ═══════════════════════════════════════════════════════════════════════════
# PORTFOLIO SOLVER
# ═══════════════════════════════════════════════════════════════════════════


class PortfolioSolver:
    """Run multiple solvers in parallel and return the first result.

    Currently supports Z3 with different configurations.  CVC5 can be
    added when the pycvc5 package is available.

    Args:
        configs:   List of solver configurations to try.
        timeout_ms: Overall timeout in milliseconds.
    """

    def __init__(
        self,
        configs: list[SolverConfig] | None = None,
        timeout_ms: int = 60000,
    ) -> None:
        if configs:
            self._configs = configs
        else:
            self._configs = [
                SolverConfig(
                    timeout_ms=timeout_ms,
                    theory=SMTTheory.QF_LRA,
                    tactics=["simplify", "solve-eqs", "smt"],
                ),
                SolverConfig(
                    timeout_ms=timeout_ms,
                    theory=SMTTheory.QF_NRA,
                    tactics=["simplify", "nlsat"],
                ),
                SolverConfig(
                    timeout_ms=timeout_ms,
                    theory=SMTTheory.QF_NRA,
                    seed=123,
                ),
            ]
        self._timeout_ms = timeout_ms
        self._stats = SolverStats()

    @property
    def stats(self) -> SolverStats:
        """Return aggregate statistics across all solvers.

        Returns:
            SolverStats.
        """
        return self._stats

    def check(self, encoding: SMTEncoding) -> SolverResult:
        """Check an encoding using portfolio solving.

        Runs each solver configuration in a separate thread and returns
        the first definitive (SAT/UNSAT) result.

        Args:
            encoding: The SMT encoding to check.

        Returns:
            Best SolverResult found.
        """
        results: list[SolverResult] = []
        result_lock = threading.Lock()
        done_event = threading.Event()

        def run_solver(config: SolverConfig) -> None:
            """Run a single solver in a thread."""
            solver = Z3Solver(config)
            result = solver.check_encoding(encoding)
            with result_lock:
                results.append(result)
                if result.result.is_definitive:
                    done_event.set()

        threads: list[threading.Thread] = []
        for config in self._configs:
            t = threading.Thread(target=run_solver, args=(config,), daemon=True)
            threads.append(t)
            t.start()

        # Wait for first definitive result or all threads to complete
        done_event.wait(timeout=self._timeout_ms / 1000.0)

        # Find best result
        best: SolverResult | None = None
        for r in results:
            if r.result.is_definitive:
                if best is None or r.solve_time < best.solve_time:
                    best = r

        if best is None:
            if results:
                best = results[0]
            else:
                best = SolverResult(
                    result=CheckResult.TIMEOUT,
                    solve_time=self._timeout_ms / 1000.0,
                    solver_name="portfolio",
                )

        self._stats.record(best)
        return best

    def check_with_fallback(
        self,
        encoding: SMTEncoding,
        theories: list[SMTTheory] | None = None,
    ) -> SolverResult:
        """Check encoding trying theories in order.

        Tries the weakest theory first and falls back to stronger
        theories on UNKNOWN/TIMEOUT.

        Args:
            encoding: The SMT encoding.
            theories: Ordered list of theories to try.

        Returns:
            Best SolverResult.
        """
        if theories is None:
            theories = [SMTTheory.QF_LRA, SMTTheory.QF_NRA]

        for theory in theories:
            config = SolverConfig(
                timeout_ms=self._timeout_ms // len(theories),
                theory=theory,
            )
            solver = Z3Solver(config)
            result = solver.check_encoding(encoding)
            self._stats.record(result)

            if result.result.is_definitive:
                return result

        # All theories exhausted
        return SolverResult(
            result=CheckResult.UNKNOWN,
            solver_name="portfolio",
            theory=theories[-1] if theories else SMTTheory.QF_LRA,
        )


# ═══════════════════════════════════════════════════════════════════════════
# SOLVER POOL
# ═══════════════════════════════════════════════════════════════════════════


class SolverPool:
    """Manage multiple solver instances for parallel queries.

    Pre-creates a pool of Z3 solver instances that can be checked out
    for individual queries, avoiding the overhead of creating a new
    solver for each query.

    Args:
        pool_size: Number of solver instances to create.
        config:    Configuration for all solvers in the pool.
    """

    def __init__(
        self,
        pool_size: int = 4,
        config: SolverConfig | None = None,
    ) -> None:
        self._config = config or SolverConfig.default()
        self._pool: list[Z3Solver] = []
        self._available: list[bool] = []
        self._lock = threading.Lock()
        self._stats = SolverStats()

        for _ in range(pool_size):
            self._pool.append(Z3Solver(self._config))
            self._available.append(True)

    @property
    def pool_size(self) -> int:
        """Return the pool size.

        Returns:
            Number of solvers in the pool.
        """
        return len(self._pool)

    @property
    def stats(self) -> SolverStats:
        """Return aggregate statistics.

        Returns:
            SolverStats.
        """
        return self._stats

    def acquire(self) -> tuple[int, Z3Solver] | None:
        """Acquire an available solver from the pool.

        Returns:
            Tuple of (index, solver), or None if no solver is available.
        """
        with self._lock:
            for i, avail in enumerate(self._available):
                if avail:
                    self._available[i] = False
                    return (i, self._pool[i])
        return None

    def release(self, idx: int) -> None:
        """Release a solver back to the pool.

        Args:
            idx: Pool index of the solver to release.
        """
        with self._lock:
            if 0 <= idx < len(self._available):
                self._pool[idx].reset()
                self._available[idx] = True

    def check(self, encoding: SMTEncoding) -> SolverResult:
        """Check an encoding using a pooled solver.

        Acquires a solver, checks the encoding, and releases it.

        Args:
            encoding: The encoding to check.

        Returns:
            SolverResult.
        """
        acquired = self.acquire()
        if acquired is None:
            # All solvers busy; create a temporary one
            solver = Z3Solver(self._config)
            result = solver.check_encoding(encoding)
            self._stats.record(result)
            return result

        idx, solver = acquired
        try:
            result = solver.check_encoding(encoding)
            self._stats.record(result)
            return result
        finally:
            self.release(idx)

    def check_batch(
        self,
        encodings: Sequence[SMTEncoding],
    ) -> list[SolverResult]:
        """Check multiple encodings in parallel using the pool.

        Args:
            encodings: Sequence of encodings to check.

        Returns:
            List of SolverResults in the same order.
        """
        results: list[SolverResult | None] = [None] * len(encodings)
        threads: list[threading.Thread] = []
        remaining = list(range(len(encodings)))

        def run_one(idx: int, encoding: SMTEncoding) -> None:
            """Check one encoding and store the result."""
            result = self.check(encoding)
            results[idx] = result

        for i in remaining:
            t = threading.Thread(
                target=run_one, args=(i, encodings[i]), daemon=True,
            )
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=self._config.timeout_ms / 1000.0 + 5)

        # Fill in any missing results
        for i in range(len(results)):
            if results[i] is None:
                results[i] = SolverResult(
                    result=CheckResult.TIMEOUT,
                    solver_name="pool",
                )

        return results  # type: ignore[return-value]

    def available_count(self) -> int:
        """Return the number of available solvers.

        Returns:
            Count of available solvers.
        """
        with self._lock:
            return sum(1 for a in self._available if a)

    def reset_all(self) -> None:
        """Reset all solvers in the pool."""
        with self._lock:
            for i, solver in enumerate(self._pool):
                solver.reset()
                self._available[i] = True
