"""Command-line interface for DP-CEGAR.

Submodules:
    main          – Click-based CLI entry point
    formatters    – Output formatting (JSON, text, rich, table, diff)
    config_loader – Configuration loading from files, env, and presets
"""

from dpcegar.cli.formatters import (
    DiffFormatter,
    JSONFormatter,
    ResultFormatter,
    RichFormatter,
    TableFormatter,
    TextFormatter,
)
from dpcegar.cli.config_loader import (
    ConfigLoader,
    ConfigValidator,
    EnvironmentOverrideApplier,
    PROFILE_PRESETS,
)

__all__ = [
    "ConfigLoader",
    "ConfigValidator",
    "DiffFormatter",
    "EnvironmentOverrideApplier",
    "JSONFormatter",
    "PROFILE_PRESETS",
    "ResultFormatter",
    "RichFormatter",
    "TableFormatter",
    "TextFormatter",
]
