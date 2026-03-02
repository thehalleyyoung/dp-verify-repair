"""Configuration loading, validation, and profile management for the CLI.

Merges configuration from three sources in priority order (highest last):

1. Config file (TOML / JSON / YAML)
2. Environment variables with a common prefix
3. CLI flag overrides

Provides preset profiles (``fast``, ``standard``, ``thorough``) that
supply reasonable defaults for common workflows.

Example::

    loader = ConfigLoader()
    cfg = loader.load("dpcegar.toml")

    warnings = ConfigValidator().validate(cfg)
    for w in warnings:
        print(w)
"""

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

from dpcegar.utils.config import DPCegarConfig


# ---------------------------------------------------------------------------
# Profile presets
# ---------------------------------------------------------------------------

PROFILE_PRESETS: dict[str, dict[str, Any]] = {
    "fast": {
        "cegar": {
            "max_refinements": 5,
            "timeout_seconds": 10,
            "fixpoint_tolerance": 1e-4,
        },
        "solver": {
            "timeout_ms": 5000,
        },
        "repair": {
            "max_iterations": 10,
        },
        "output": {
            "produce_certificate": False,
        },
    },
    "standard": {
        "cegar": {
            "max_refinements": 50,
            "timeout_seconds": 60,
            "fixpoint_tolerance": 1e-8,
        },
        "solver": {
            "timeout_ms": 30000,
        },
        "repair": {
            "max_iterations": 100,
        },
        "output": {
            "produce_certificate": True,
        },
    },
    "thorough": {
        "cegar": {
            "max_refinements": 500,
            "timeout_seconds": 600,
            "fixpoint_tolerance": 1e-12,
        },
        "solver": {
            "timeout_ms": 120000,
        },
        "repair": {
            "max_iterations": 1000,
        },
        "output": {
            "produce_certificate": True,
        },
    },
}


# ---------------------------------------------------------------------------
# EnvironmentOverrideApplier
# ---------------------------------------------------------------------------


class EnvironmentOverrideApplier:
    """Extract configuration overrides from environment variables.

    Environment variables are expected to follow the pattern
    ``{prefix}{SECTION}__{KEY}`` where double-underscore separates
    nesting levels.  For example ``DPCEGAR_CEGAR__TIMEOUT_SECONDS=120``
    maps to ``{"cegar": {"timeout_seconds": "120"}}``.
    """

    def apply(self, base: dict[str, Any], prefix: str = "DPCEGAR_") -> dict[str, Any]:
        """Overlay environment-variable overrides onto *base*.

        Parameters
        ----------
        base:
            The configuration dictionary to update **in-place**.
        prefix:
            Environment variable prefix to scan for.

        Returns
        -------
        dict[str, Any]
            The mutated *base* dict (same object).
        """
        for key, value in os.environ.items():
            if not key.startswith(prefix):
                continue
            suffix = key[len(prefix):]
            parts = [p.lower() for p in suffix.split("__")]
            self._set_nested(base, parts, self._coerce(value))
        return base

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _set_nested(d: dict[str, Any], keys: list[str], value: Any) -> None:
        """Set a value in a nested dict following *keys* as the path."""
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value

    @staticmethod
    def _coerce(value: str) -> int | float | bool | str:
        """Best-effort coercion of a string to a Python scalar."""
        if value.lower() in ("true", "1", "yes"):
            return True
        if value.lower() in ("false", "0", "no"):
            return False
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
        return value


# ---------------------------------------------------------------------------
# ConfigLoader
# ---------------------------------------------------------------------------


class ConfigLoader:
    """Load and merge configuration from file, environment, and CLI flags.

    The loader supports JSON, TOML, and YAML config files (detected by
    extension).  TOML requires Python 3.11+ ``tomllib`` or the
    ``tomli`` package.
    """

    def __init__(self) -> None:
        self._env_applier = EnvironmentOverrideApplier()

    def load(
        self,
        path: str | None = None,
        env_prefix: str = "DPCEGAR_",
        cli_overrides: dict[str, Any] | None = None,
        profile: str | None = None,
    ) -> DPCegarConfig:
        """Build a ``DPCegarConfig`` by merging all configuration sources.

        Parameters
        ----------
        path:
            Optional path to a config file (JSON / TOML / YAML).
        env_prefix:
            Prefix for environment variable overrides.
        cli_overrides:
            Additional key-value overrides from CLI flags.
        profile:
            Name of a preset profile to apply before other overrides.

        Returns
        -------
        DPCegarConfig
            The fully resolved configuration.
        """
        file_data = self._load_from_file(path) if path else {}
        env_data = self._load_from_env(env_prefix)

        merged = self._merge_sources(file_data, env_data, cli_overrides or {})

        if profile:
            merged = self._apply_profile(merged, profile)

        return DPCegarConfig.from_dict(merged)

    # -- source loaders ------------------------------------------------------

    def _load_from_file(self, path: str) -> dict[str, Any]:
        """Read a configuration file and return its contents as a dict.

        Parameters
        ----------
        path:
            Filesystem path.  Supported extensions: ``.json``,
            ``.toml``, ``.yaml`` / ``.yml``.

        Returns
        -------
        dict[str, Any]

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        ValueError
            If the file extension is not supported.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        text = p.read_text(encoding="utf-8")

        if p.suffix == ".json":
            return json.loads(text)

        if p.suffix == ".toml":
            try:
                import tomllib  # Python 3.11+
            except ModuleNotFoundError:
                import tomli as tomllib  # type: ignore[no-redef]
            return tomllib.loads(text)

        if p.suffix in {".yaml", ".yml"}:
            import yaml  # type: ignore[import-untyped]
            return yaml.safe_load(text) or {}

        raise ValueError(f"Unsupported config file format: {p.suffix}")

    def _load_from_env(self, prefix: str) -> dict[str, Any]:
        """Gather overrides from environment variables.

        Parameters
        ----------
        prefix:
            Only variables starting with *prefix* are considered.

        Returns
        -------
        dict[str, Any]
        """
        data: dict[str, Any] = {}
        self._env_applier.apply(data, prefix)
        return data

    # -- merging -------------------------------------------------------------

    @staticmethod
    def _merge_sources(
        file_data: dict[str, Any],
        env_data: dict[str, Any],
        cli_overrides: dict[str, Any],
    ) -> dict[str, Any]:
        """Deep-merge three configuration layers (file < env < CLI).

        Later layers override earlier ones at the leaf level.

        Parameters
        ----------
        file_data:
            Settings loaded from a config file.
        env_data:
            Settings extracted from environment variables.
        cli_overrides:
            Settings explicitly passed via CLI flags.

        Returns
        -------
        dict[str, Any]
            The merged configuration dictionary.
        """
        merged = deepcopy(file_data)
        _deep_merge(merged, env_data)
        _deep_merge(merged, cli_overrides)
        return merged

    @staticmethod
    def _apply_profile(data: dict[str, Any], profile_name: str) -> dict[str, Any]:
        """Apply a named preset profile underneath the existing data.

        The profile acts as a *base*: explicit values in *data* take
        precedence over the profile's defaults.

        Parameters
        ----------
        data:
            The current merged configuration dict.
        profile_name:
            One of the keys in :data:`PROFILE_PRESETS`.

        Returns
        -------
        dict[str, Any]
            A new dict with profile defaults filled in.

        Raises
        ------
        ValueError
            If *profile_name* is not a known preset.
        """
        preset = PROFILE_PRESETS.get(profile_name)
        if preset is None:
            raise ValueError(
                f"Unknown profile '{profile_name}'. "
                f"Available: {', '.join(PROFILE_PRESETS)}"
            )
        base = deepcopy(preset)
        _deep_merge(base, data)
        return base


# ---------------------------------------------------------------------------
# ConfigValidator
# ---------------------------------------------------------------------------


class ConfigValidator:
    """Validate a ``DPCegarConfig`` and return human-readable warnings.

    The validator does **not** raise exceptions; instead it collects
    warning / error strings so that the CLI can present them all at once.
    """

    def validate(self, config: DPCegarConfig) -> list[str]:
        """Run all validation checks.

        Parameters
        ----------
        config:
            The resolved configuration object.

        Returns
        -------
        list[str]
            A (possibly empty) list of warning or error messages.
        """
        warnings: list[str] = []
        warnings.extend(self._check_solver_config(config))
        warnings.extend(self._check_cegar_config(config))
        warnings.extend(self._check_repair_config(config))
        return warnings

    def _check_solver_config(self, cfg: DPCegarConfig) -> list[str]:
        """Validate solver-related settings.

        Parameters
        ----------
        cfg:
            The full config.

        Returns
        -------
        list[str]
        """
        warnings: list[str] = []
        solver = getattr(cfg, "solver", None)
        if solver is None:
            return warnings
        timeout = getattr(solver, "timeout_ms", None)
        if timeout is not None and timeout < 100:
            warnings.append(
                f"Solver timeout ({timeout} ms) is very low; "
                "verification may time out prematurely."
            )
        if timeout is not None and timeout > 600_000:
            warnings.append(
                f"Solver timeout ({timeout} ms) is very high; "
                "consider using a lower value to avoid long waits."
            )
        return warnings

    def _check_cegar_config(self, cfg: DPCegarConfig) -> list[str]:
        """Validate CEGAR loop settings.

        Parameters
        ----------
        cfg:
            The full config.

        Returns
        -------
        list[str]
        """
        warnings: list[str] = []
        cegar = getattr(cfg, "cegar", None)
        if cegar is None:
            return warnings
        max_ref = getattr(cegar, "max_refinements", None)
        if max_ref is not None and max_ref < 1:
            warnings.append(
                "CEGAR max_refinements must be ≥ 1; "
                f"got {max_ref}."
            )
        timeout = getattr(cegar, "timeout_seconds", None)
        if timeout is not None and timeout <= 0:
            warnings.append(
                "CEGAR timeout_seconds must be positive; "
                f"got {timeout}."
            )
        tol = getattr(cegar, "fixpoint_tolerance", None)
        if tol is not None and tol <= 0:
            warnings.append(
                "fixpoint_tolerance must be positive; "
                f"got {tol}."
            )
        return warnings

    def _check_repair_config(self, cfg: DPCegarConfig) -> list[str]:
        """Validate repair synthesis settings.

        Parameters
        ----------
        cfg:
            The full config.

        Returns
        -------
        list[str]
        """
        warnings: list[str] = []
        repair = getattr(cfg, "repair", None)
        if repair is None:
            return warnings
        max_iter = getattr(repair, "max_iterations", None)
        if max_iter is not None and max_iter < 1:
            warnings.append(
                "Repair max_iterations must be ≥ 1; "
                f"got {max_iter}."
            )
        return warnings


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> None:
    """Recursively merge *override* into *base* in-place.

    Parameters
    ----------
    base:
        The dictionary to update.
    override:
        Values that should take precedence.
    """
    for key, value in override.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            _deep_merge(base[key], value)
        else:
            base[key] = value
