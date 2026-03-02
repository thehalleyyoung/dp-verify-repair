"""Structured logging with Rich console output.

Provides a unified logging setup for the entire DP-CEGAR pipeline.
Supports structured fields, coloured output via Rich, and optional
JSON-line output for CI / machine consumption.
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

# ---------------------------------------------------------------------------
# Theme & global console
# ---------------------------------------------------------------------------

_THEME = Theme(
    {
        "info": "cyan",
        "warning": "yellow bold",
        "error": "red bold",
        "success": "green bold",
        "debug": "dim",
        "phase": "magenta bold",
        "metric": "blue",
    }
)

console = Console(theme=_THEME, stderr=True)

# ---------------------------------------------------------------------------
# Logger names (one per pipeline stage)
# ---------------------------------------------------------------------------

LOGGER_ROOT = "dpcegar"
LOGGER_PARSER = "dpcegar.parser"
LOGGER_PATHS = "dpcegar.paths"
LOGGER_DENSITY = "dpcegar.density"
LOGGER_SMT = "dpcegar.smt"
LOGGER_CEGAR = "dpcegar.cegar"
LOGGER_REPAIR = "dpcegar.repair"
LOGGER_VARIANTS = "dpcegar.variants"
LOGGER_CERTS = "dpcegar.certificates"

ALL_LOGGERS = [
    LOGGER_ROOT, LOGGER_PARSER, LOGGER_PATHS, LOGGER_DENSITY,
    LOGGER_SMT, LOGGER_CEGAR, LOGGER_REPAIR, LOGGER_VARIANTS, LOGGER_CERTS,
]

# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class LogConfig:
    """Configuration knobs for the logging subsystem."""

    level: str = "INFO"
    show_time: bool = True
    show_path: bool = False
    json_output: bool = False
    log_file: str | None = None
    rich_tracebacks: bool = True


def setup_logging(cfg: LogConfig | None = None) -> None:
    """Initialise the logging subsystem.

    Call once at application start-up (e.g. from the CLI entry-point).
    Subsequent calls reset handlers so that reconfiguration is safe in tests.

    Args:
        cfg: Logging configuration.  Uses sensible defaults when ``None``.
    """
    cfg = cfg or LogConfig()
    level = getattr(logging, cfg.level.upper(), logging.INFO)

    root = logging.getLogger(LOGGER_ROOT)
    root.setLevel(level)

    # Remove existing handlers to allow re-initialisation
    root.handlers.clear()

    # Rich console handler
    rich_handler = RichHandler(
        console=console,
        show_time=cfg.show_time,
        show_path=cfg.show_path,
        rich_tracebacks=cfg.rich_tracebacks,
        markup=True,
        log_time_format="[%X]",
    )
    rich_handler.setLevel(level)
    root.addHandler(rich_handler)

    # Optional file handler
    if cfg.log_file:
        fmt = logging.Formatter(
            "%(asctime)s | %(name)-24s | %(levelname)-7s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        fh = logging.FileHandler(cfg.log_file, mode="a", encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        root.addHandler(fh)

    # Prevent double-logging from propagation
    for name in ALL_LOGGERS:
        lg = logging.getLogger(name)
        lg.propagate = (name != LOGGER_ROOT)


def get_logger(name: str = LOGGER_ROOT) -> logging.Logger:
    """Return a named logger under the ``dpcegar`` hierarchy."""
    if not name.startswith(LOGGER_ROOT):
        name = f"{LOGGER_ROOT}.{name}"
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# Structured log record helpers
# ---------------------------------------------------------------------------

def log_phase(logger: logging.Logger, phase: str, **extra: Any) -> None:
    """Emit a phase-transition log line with optional structured fields."""
    parts = [f"[phase]▶ {phase}[/phase]"]
    for k, v in extra.items():
        parts.append(f"[metric]{k}[/metric]={v}")
    logger.info(" ".join(parts))


def log_metric(logger: logging.Logger, name: str, value: Any, unit: str = "") -> None:
    """Emit a single metric observation."""
    suffix = f" {unit}" if unit else ""
    logger.info("[metric]%s[/metric] = %s%s", name, value, suffix)


def log_counterexample(logger: logging.Logger, cex: dict[str, Any]) -> None:
    """Log a counterexample witness in a readable format."""
    logger.warning("[warning]Counterexample found:[/warning]")
    for k, v in cex.items():
        logger.warning("  %s = %s", k, v)


# ---------------------------------------------------------------------------
# Context managers
# ---------------------------------------------------------------------------

@contextmanager
def log_section(logger: logging.Logger, title: str) -> Generator[None, None, None]:
    """Context manager that logs entry/exit of a named section with timing."""
    logger.info("[phase]┌─ %s[/phase]", title)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        logger.info(
            "[phase]└─ %s[/phase] [metric](%.3fs)[/metric]", title, elapsed
        )


@contextmanager
def suppress_logging(level: int = logging.CRITICAL + 1) -> Generator[None, None, None]:
    """Temporarily silence all ``dpcegar`` loggers (useful in tests)."""
    root = logging.getLogger(LOGGER_ROOT)
    old_level = root.level
    root.setLevel(level)
    try:
        yield
    finally:
        root.setLevel(old_level)


# ---------------------------------------------------------------------------
# Verbosity presets
# ---------------------------------------------------------------------------

def set_verbosity(verbosity: int) -> None:
    """Map a CLI ``-v`` count to a logging level.

    0  → WARNING
    1  → INFO
    2+ → DEBUG
    """
    level_map = {0: logging.WARNING, 1: logging.INFO}
    level = level_map.get(verbosity, logging.DEBUG)
    logging.getLogger(LOGGER_ROOT).setLevel(level)
