"""Configuration management with Pydantic.

Centralises all tuneable parameters for the DP-CEGAR pipeline.
Configurations can be loaded from TOML / JSON files, environment
variables (prefixed ``DPCEGAR_``), or constructed programmatically.
"""

from __future__ import annotations

import json
import os
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enumerations for constrained choices
# ---------------------------------------------------------------------------

class SolverBackend(str, Enum):
    """Supported SMT solver back-ends."""
    Z3 = "z3"
    CVC5 = "cvc5"


class RepairStrategy(str, Enum):
    """Available automated repair strategies."""
    NOISE_SCALE = "noise_scale"
    CLIPPING = "clipping"
    COMPOSITION = "composition"
    SUBSAMPLING = "subsampling"
    COMBINED = "combined"


class OutputFormat(str, Enum):
    """Output format for verification results."""
    TEXT = "text"
    JSON = "json"
    RICH = "rich"


class LogLevel(str, Enum):
    """Logging verbosity levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


# ---------------------------------------------------------------------------
# Sub-configurations
# ---------------------------------------------------------------------------

class SolverConfig(BaseModel):
    """Configuration for the SMT solver layer."""

    backend: SolverBackend = SolverBackend.Z3
    timeout_ms: int = Field(default=30_000, ge=100, le=3_600_000)
    random_seed: int = Field(default=42, ge=0)
    incremental: bool = True
    produce_unsat_cores: bool = True
    fp_precision: int = Field(default=64, ge=32, le=128)
    model_completion: bool = True


class CEGARConfig(BaseModel):
    """Configuration for the CEGAR loop."""

    max_iterations: int = Field(default=100, ge=1, le=10_000)
    max_path_depth: int = Field(default=50, ge=1, le=1_000)
    loop_unroll_bound: int = Field(default=10, ge=1, le=1_000)
    use_abstraction_cache: bool = True
    refinement_strategy: Literal["lazy", "eager", "hybrid"] = "lazy"
    counterexample_minimise: bool = True
    early_termination: bool = True


class RepairConfig(BaseModel):
    """Configuration for the automated repair engine."""

    strategy: RepairStrategy = RepairStrategy.COMBINED
    max_attempts: int = Field(default=50, ge=1, le=10_000)
    timeout_seconds: float = Field(default=300.0, ge=1.0)
    allow_relaxation: bool = False
    noise_scale_search: Literal["binary", "linear", "exponential"] = "binary"
    clip_bound_candidates: int = Field(default=20, ge=1)
    validate_repairs: bool = True


class PathConfig(BaseModel):
    """Configuration for path enumeration and symbolic execution."""

    max_paths: int = Field(default=10_000, ge=1)
    prune_infeasible: bool = True
    merge_symmetric: bool = True
    symbolic_arrays: bool = False
    path_timeout_ms: int = Field(default=5_000, ge=100)


class DensityConfig(BaseModel):
    """Configuration for density function lifting."""

    cdf_approximation: Literal["taylor", "rational", "lookup"] = "taylor"
    taylor_order: int = Field(default=12, ge=4, le=50)
    interval_precision: int = Field(default=128, ge=32, le=512)
    simplify_densities: bool = True


class OutputConfig(BaseModel):
    """Configuration for result output and reporting."""

    format: OutputFormat = OutputFormat.RICH
    show_counterexamples: bool = True
    show_timing: bool = True
    show_certificate: bool = False
    verbose_smt: bool = False
    output_file: str | None = None


class LoggingConfig(BaseModel):
    """Configuration for the logging subsystem."""

    level: LogLevel = LogLevel.INFO
    show_time: bool = True
    show_path: bool = False
    json_output: bool = False
    log_file: str | None = None
    rich_tracebacks: bool = True


# ---------------------------------------------------------------------------
# Top-level configuration
# ---------------------------------------------------------------------------

class DPCegarConfig(BaseModel):
    """Top-level configuration for the DP-CEGAR engine.

    Example usage::

        cfg = DPCegarConfig()                     # all defaults
        cfg = DPCegarConfig.from_file("dpcegar.toml")
        cfg = DPCegarConfig.from_env()

    Sections:
        solver  – SMT solver parameters
        cegar   – CEGAR loop parameters
        repair  – Repair engine parameters
        paths   – Path enumeration parameters
        density – Density lifting parameters
        output  – Output / reporting parameters
        logging – Logging parameters
    """

    solver: SolverConfig = Field(default_factory=SolverConfig)
    cegar: CEGARConfig = Field(default_factory=CEGARConfig)
    repair: RepairConfig = Field(default_factory=RepairConfig)
    paths: PathConfig = Field(default_factory=PathConfig)
    density: DensityConfig = Field(default_factory=DensityConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    @classmethod
    def from_file(cls, path: str | Path) -> DPCegarConfig:
        """Load configuration from a JSON or TOML file.

        Args:
            path: Path to the configuration file.

        Returns:
            A validated :class:`DPCegarConfig` instance.

        Raises:
            dpcegar.utils.errors.ConfigFileNotFoundError: If the file
                does not exist.
            dpcegar.utils.errors.ConfigValidationError: If the file
                contents fail validation.
        """
        from dpcegar.utils.errors import ConfigFileNotFoundError, ConfigValidationError

        p = Path(path)
        if not p.exists():
            raise ConfigFileNotFoundError(f"Config file not found: {p}")

        text = p.read_text(encoding="utf-8")
        suffix = p.suffix.lower()

        if suffix == ".json":
            data = json.loads(text)
        elif suffix in (".toml",):
            try:
                import tomllib  # Python 3.11+
            except ModuleNotFoundError:
                import tomli as tomllib  # type: ignore[no-redef]
            data = tomllib.loads(text)
        else:
            raise ConfigValidationError(
                f"Unsupported config file format: {suffix}"
            )

        try:
            return cls.model_validate(data)
        except Exception as exc:
            raise ConfigValidationError(str(exc)) from exc

    @classmethod
    def from_env(cls, prefix: str = "DPCEGAR_") -> DPCegarConfig:
        """Construct a config by reading ``DPCEGAR_*`` environment variables.

        Flat keys are mapped to nested fields using ``__`` as separator.
        For example ``DPCEGAR_SOLVER__TIMEOUT_MS=60000``.
        """
        overrides: dict[str, Any] = {}
        for key, value in os.environ.items():
            if not key.startswith(prefix):
                continue
            parts = key[len(prefix):].lower().split("__")
            d = overrides
            for part in parts[:-1]:
                d = d.setdefault(part, {})
            d[parts[-1]] = value
        return cls.model_validate(overrides)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DPCegarConfig:
        """Construct a config from an arbitrary dictionary."""
        return cls.model_validate(data)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_json(self, indent: int = 2) -> str:
        """Serialise the configuration to a JSON string."""
        return self.model_dump_json(indent=indent)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the configuration to a plain dictionary."""
        return self.model_dump(mode="python")

    # ------------------------------------------------------------------
    # Merge / override helpers
    # ------------------------------------------------------------------

    def merge(self, overrides: dict[str, Any]) -> DPCegarConfig:
        """Return a new config with *overrides* applied on top of ``self``."""
        base = self.to_dict()
        _deep_merge(base, overrides)
        return self.__class__.model_validate(base)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> None:
    """Recursively merge *overrides* into *base* (mutates *base*)."""
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


# ---------------------------------------------------------------------------
# Default singleton
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG: DPCegarConfig | None = None


def get_config() -> DPCegarConfig:
    """Return the global default configuration (lazy-initialised)."""
    global _DEFAULT_CONFIG
    if _DEFAULT_CONFIG is None:
        _DEFAULT_CONFIG = DPCegarConfig()
    return _DEFAULT_CONFIG


def set_config(cfg: DPCegarConfig) -> None:
    """Replace the global default configuration."""
    global _DEFAULT_CONFIG
    _DEFAULT_CONFIG = cfg


def reset_config() -> None:
    """Reset the global configuration to defaults (useful in tests)."""
    global _DEFAULT_CONFIG
    _DEFAULT_CONFIG = None
