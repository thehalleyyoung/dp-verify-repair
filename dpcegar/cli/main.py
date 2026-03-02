"""Command-line interface for the DP-CEGAR verification and repair engine.

Provides Click-based commands for verification, repair, multi-notion
checking, privacy profiling, and benchmarking of differential privacy
mechanisms.

Mechanisms can be supplied as annotated Python source files (.py),
JSON MechIR files (.json), or YAML MechIR files (.yaml/.yml).

Usage::

    dpcegar verify mechanism.py --budget "eps=1.0"
    dpcegar verify mechanism.json --budget "eps=1.0,delta=1e-5" --notion approx
    dpcegar check mechanism.py
    dpcegar init-mechanism my_mech.py --template laplace
    dpcegar repair mechanism.py --budget "eps=1.0" --strategy noise_scale
    dpcegar check-all mechanism.json --delta 1e-5
    dpcegar profile mechanism.json --alphas "2,4,8,16"
    dpcegar benchmark --suite standard --output-dir results/
    dpcegar info mechanism.json
    dpcegar dump-config
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import click

from dpcegar import __version__
from dpcegar.ir.types import (
    PrivacyBudget,
    PrivacyNotion,
    PureBudget,
    ApproxBudget,
    ZCDPBudget,
    RDPBudget,
    FDPBudget,
    GDPBudget,
)
from dpcegar.ir.nodes import MechIR
from dpcegar.cegar.engine import CEGAREngine, CEGARResult, CEGARVerdict, CEGARConfig
from dpcegar.repair.synthesizer import RepairResult, RepairVerdict
from dpcegar.utils.config import DPCegarConfig, get_config, set_config, OutputFormat
from dpcegar.utils.logging import setup_logging, get_logger, LogConfig, set_verbosity

from dpcegar.cli.formatters import (
    ResultFormatter,
    TextFormatter,
    JSONFormatter,
    RichFormatter,
    CSVFormatter,
    SARIFFormatter,
)
from dpcegar.cli.config_loader import ConfigLoader, ConfigValidator


# ---------------------------------------------------------------------------
# Notion string ↔ enum mapping
# ---------------------------------------------------------------------------

_NOTION_MAP: dict[str, PrivacyNotion] = {
    "pure": PrivacyNotion.PURE_DP,
    "approx": PrivacyNotion.APPROX_DP,
    "zcdp": PrivacyNotion.ZCDP,
    "rdp": PrivacyNotion.RDP,
    "fdp": PrivacyNotion.FDP,
    "gdp": PrivacyNotion.GDP,
}

_FORMATTER_MAP: dict[str, type[ResultFormatter]] = {
    "text": TextFormatter,
    "json": JSONFormatter,
    "rich": RichFormatter,
    "csv": CSVFormatter,
    "sarif": SARIFFormatter,
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _parse_budget(budget_str: str, notion: str) -> PrivacyBudget:
    """Parse a budget string like ``"eps=1.0"`` or ``"eps=1.0,delta=1e-5"``.

    Parameters
    ----------
    budget_str:
        Comma-separated ``key=value`` pairs.
    notion:
        Privacy notion name (``pure``, ``approx``, ``zcdp``, ``rdp``,
        ``fdp``, ``gdp``).

    Returns
    -------
    PrivacyBudget
        A concrete budget instance matching *notion*.

    Raises
    ------
    click.BadParameter
        If the string cannot be parsed for the requested notion.
    """
    parts: dict[str, float] = {}
    for token in budget_str.split(","):
        token = token.strip()
        if "=" not in token:
            raise click.BadParameter(
                f"Invalid budget token '{token}'; expected key=value"
            )
        key, value = token.split("=", 1)
        try:
            parts[key.strip()] = float(value.strip())
        except ValueError:
            raise click.BadParameter(
                f"Non-numeric budget value for '{key.strip()}': {value.strip()}"
            )

    notion_enum = _NOTION_MAP.get(notion)
    if notion_enum is None:
        raise click.BadParameter(f"Unknown privacy notion: {notion}")

    if notion_enum == PrivacyNotion.PURE_DP:
        return PureBudget(epsilon=parts.get("eps", parts.get("epsilon", 1.0)))
    elif notion_enum == PrivacyNotion.APPROX_DP:
        return ApproxBudget(
            epsilon=parts.get("eps", parts.get("epsilon", 1.0)),
            delta=parts.get("delta", 1e-5),
        )
    elif notion_enum == PrivacyNotion.ZCDP:
        return ZCDPBudget(rho=parts.get("rho", 1.0))
    elif notion_enum == PrivacyNotion.RDP:
        return RDPBudget(
            alpha=parts.get("alpha", 2.0),
            epsilon=parts.get("eps", parts.get("epsilon", 1.0)),
        )
    elif notion_enum == PrivacyNotion.FDP:
        return FDPBudget(trade_off_fn=lambda x: x)
    elif notion_enum == PrivacyNotion.GDP:
        return GDPBudget(mu=parts.get("mu", 1.0))
    else:
        raise click.BadParameter(f"Unsupported notion: {notion}")


def _load_mechanism(path: str) -> MechIR:
    """Load a mechanism IR from a JSON, YAML, or Python source file.

    Parameters
    ----------
    path:
        Filesystem path to the mechanism specification.  Supported formats:
        ``.json``, ``.yaml``/``.yml`` (serialised MechIR), and ``.py``
        (annotated Python source parsed via ``parse_mechanism``).

    Returns
    -------
    MechIR
        The parsed intermediate representation.
    """
    mech_path = Path(path)
    if mech_path.suffix == ".json":
        from dpcegar.ir.serialization import from_json

        with open(mech_path, "r") as fh:
            return from_json(fh.read())
    elif mech_path.suffix in {".yaml", ".yml"}:
        from dpcegar.ir.serialization import from_yaml

        with open(mech_path, "r") as fh:
            return from_yaml(fh.read())
    elif mech_path.suffix == ".py":
        from dpcegar.parser.ast_bridge import parse_mechanism

        source = mech_path.read_text(encoding="utf-8")
        return parse_mechanism(source, file=str(mech_path))
    else:
        raise click.BadParameter(
            f"Unsupported file format: {mech_path.suffix}. "
            "Use .py (annotated source), .json, or .yaml."
        )


def _format_result(result: Any, fmt: str) -> str:
    """Render a verification / repair result using the chosen formatter.

    Parameters
    ----------
    result:
        A ``CEGARResult``, ``RepairResult``, or dict.
    fmt:
        One of ``text``, ``json``, ``rich``.

    Returns
    -------
    str
        The formatted string.
    """
    formatter_cls = _FORMATTER_MAP.get(fmt, TextFormatter)
    formatter = formatter_cls()
    if isinstance(result, CEGARResult):
        return formatter.format_verification(result)
    elif isinstance(result, RepairResult):
        return formatter.format_repair(result)
    elif isinstance(result, dict):
        return formatter.format_multi_variant(result)
    return str(result)


def _setup_config(ctx: click.Context) -> DPCegarConfig:
    """Build the effective configuration from CLI context.

    Merges config-file settings, environment overrides, and CLI flags.

    Returns
    -------
    DPCegarConfig
        The resolved configuration object.
    """
    loader = ConfigLoader()
    config_path: str | None = ctx.obj.get("config_path")
    cfg = loader.load(config_path)

    output_format: str | None = ctx.obj.get("output_format")
    if output_format:
        cfg.output.format = OutputFormat(output_format)

    timeout: int | None = ctx.obj.get("timeout")
    if timeout is not None:
        cfg.cegar.timeout_seconds = timeout

    warnings = ConfigValidator().validate(cfg)
    if warnings:
        logger = get_logger("cli")
        for w in warnings:
            logger.warning("Config warning: %s", w)

    set_config(cfg)
    return cfg


def _write_output(text: str, output_path: str | None) -> None:
    """Write *text* to *output_path* or stdout."""
    if output_path:
        Path(output_path).write_text(text, encoding="utf-8")
        click.echo(f"Output written to {output_path}", err=True)
    else:
        click.echo(text)


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group()
@click.version_option(version=__version__, prog_name="dpcegar")
@click.option("-v", "--verbose", count=True, help="Increase verbosity (-v, -vv, -vvv).")
@click.option("--config", type=click.Path(exists=False), default=None, help="Config file path.")
@click.option(
    "--output-format",
    type=click.Choice(["text", "json", "csv", "rich"]),
    default="text",
    help="Output format.",
)
@click.option("--timeout", type=int, default=None, help="Global solver timeout in seconds.")
@click.pass_context
def cli(
    ctx: click.Context,
    verbose: int,
    config: str | None,
    output_format: str,
    timeout: int | None,
) -> None:
    """DP-CEGAR: Differential Privacy CEGAR Verification & Repair Engine."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["config_path"] = config
    ctx.obj["output_format"] = output_format
    ctx.obj["timeout"] = timeout

    setup_logging(LogConfig())
    set_verbosity(verbose)


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("mechanism", type=click.Path(exists=True))
@click.option("--budget", "-b", required=True, help="Privacy budget, e.g. 'eps=1.0' or 'eps=1.0,delta=1e-5'.")
@click.option(
    "--notion", "-n",
    type=click.Choice(["pure", "approx", "zcdp", "rdp", "fdp", "gdp"]),
    default="pure",
    help="Privacy notion.",
)
@click.option("--timeout", "-t", type=int, default=None, help="Solver timeout override in seconds.")
@click.option("--output", "-o", type=click.Path(), default=None, help="Write result to file.")
@click.option("--certificate/--no-certificate", default=True, help="Produce a verification certificate.")
@click.option(
    "--output-format",
    type=click.Choice(["text", "json", "csv"]),
    default=None,
    help="Output format for verification results (overrides global --output-format). Useful for CI/CD integration.",
)
@click.option("--sarif", is_flag=True, default=False, help="Output results in SARIF format for GitHub Code Scanning.")
@click.pass_context
def verify(
    ctx: click.Context,
    mechanism: str,
    budget: str,
    notion: str,
    timeout: int | None,
    output: str | None,
    certificate: bool,
    output_format: str | None,
    sarif: bool,
) -> None:
    """Verify that a mechanism satisfies a differential privacy budget."""
    logger = get_logger("cli")
    cfg = _setup_config(ctx)

    if timeout is not None:
        cfg.cegar.timeout_seconds = timeout

    logger.info("Verifying mechanism: %s", mechanism)
    mechir = _load_mechanism(mechanism)
    parsed_budget = _parse_budget(budget, notion)

    engine = CEGAREngine(config=CEGARConfig(
        timeout_seconds=cfg.cegar.timeout_seconds,
    ))
    result: CEGARResult = engine.verify(mechir, parsed_budget, produce_certificate=certificate)

    if sarif:
        fmt = "sarif"
    elif output_format:
        fmt = output_format
    else:
        fmt = ctx.obj.get("output_format", "text")
    _write_output(_format_result(result, fmt), output)

    if result.verdict != CEGARVerdict.VERIFIED:
        ctx.exit(1)


# ---------------------------------------------------------------------------
# repair
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("mechanism", type=click.Path(exists=True))
@click.option("--budget", "-b", required=True, help="Target privacy budget.")
@click.option(
    "--strategy", "-s",
    type=click.Choice(["noise_scale", "clipping", "composition", "combined"]),
    default="combined",
    help="Repair strategy.",
)
@click.option("--max-cost", type=float, default=None, help="Maximum acceptable repair cost.")
@click.option("--output", "-o", type=click.Path(), default=None, help="Write result to file.")
@click.option("--cost-weights", default=None, help="Comma-separated cost weights, e.g. 'noise=1.0,clip=0.5'.")
@click.pass_context
def repair(
    ctx: click.Context,
    mechanism: str,
    budget: str,
    strategy: str,
    max_cost: float | None,
    output: str | None,
    cost_weights: str | None,
) -> None:
    """Repair a mechanism to satisfy a differential privacy budget."""
    logger = get_logger("cli")
    cfg = _setup_config(ctx)

    logger.info("Repairing mechanism: %s with strategy: %s", mechanism, strategy)
    mechir = _load_mechanism(mechanism)
    parsed_budget = _parse_budget(budget, "pure")

    weights: dict[str, float] = {}
    if cost_weights:
        for tok in cost_weights.split(","):
            k, v = tok.strip().split("=", 1)
            weights[k.strip()] = float(v.strip())

    from dpcegar.repair.synthesizer import RepairSynthesizer

    synthesizer = RepairSynthesizer(config=cfg.repair)
    result: RepairResult = synthesizer.repair(
        mechir,
        parsed_budget,
        strategy=strategy,
        max_cost=max_cost,
        cost_weights=weights or None,
    )

    fmt = ctx.obj.get("output_format", "text")
    _write_output(_format_result(result, fmt), output)

    if result.verdict != RepairVerdict.SUCCESS:
        ctx.exit(1)


# ---------------------------------------------------------------------------
# check-all
# ---------------------------------------------------------------------------


@cli.command("check-all")
@click.argument("mechanism", type=click.Path(exists=True))
@click.option("--delta", type=float, default=1e-5, help="Default delta for approximate notions.")
@click.option("--output", "-o", type=click.Path(), default=None, help="Write result to file.")
@click.option("--parallel/--no-parallel", default=False, help="Run checks in parallel.")
@click.pass_context
def check_all(
    ctx: click.Context,
    mechanism: str,
    delta: float,
    output: str | None,
    parallel: bool,
) -> None:
    """Verify a mechanism under all supported privacy notions."""
    logger = get_logger("cli")
    cfg = _setup_config(ctx)

    mechir = _load_mechanism(mechanism)

    from dpcegar.variants.multi_checker import MultiVariantChecker, MultiVariantResult

    engine = CEGAREngine(config=CEGARConfig(timeout_seconds=cfg.cegar.timeout_seconds))

    budgets_map: dict[PrivacyNotion, PrivacyBudget] = {
        PrivacyNotion.PURE_DP: PureBudget(epsilon=1.0),
        PrivacyNotion.APPROX_DP: ApproxBudget(epsilon=1.0, delta=delta),
        PrivacyNotion.ZCDP: ZCDPBudget(rho=1.0),
        PrivacyNotion.RDP: RDPBudget(alpha=2.0, epsilon=1.0),
        PrivacyNotion.GDP: GDPBudget(mu=1.0),
        PrivacyNotion.FDP: FDPBudget(trade_off_fn=lambda x: x),
    }

    checker = MultiVariantChecker(engine=engine)
    multi_result: MultiVariantResult = checker.check_all(mechir, budgets_map)

    # Convert to dict[str, CEGARResult] for the formatter
    results: dict[str, CEGARResult] = {}
    for notion, vr in multi_result.results.items():
        if vr.cegar_result is not None:
            results[notion.name.lower()] = vr.cegar_result
        else:
            # Build a stub CEGARResult for derived/skipped variants
            results[notion.name.lower()] = CEGARResult(
                verdict=CEGARVerdict.VERIFIED if vr.is_verified else CEGARVerdict.UNKNOWN,
                budget=vr.budget,
            )

    fmt = ctx.obj.get("output_format", "text")
    _write_output(_format_result(results, fmt), output)


# ---------------------------------------------------------------------------
# profile
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("mechanism", type=click.Path(exists=True))
@click.option("--delta", type=float, default=1e-5, help="Delta parameter for profiling.")
@click.option("--alphas", default="2,4,8,16,32,64", help="Comma-separated RDP alpha values.")
@click.option("--output", "-o", type=click.Path(), default=None, help="Write result to file.")
@click.option(
    "--format", "fmt",
    type=click.Choice(["text", "json", "csv"]),
    default="text",
    help="Profile output format.",
)
@click.pass_context
def profile(
    ctx: click.Context,
    mechanism: str,
    delta: float,
    alphas: str,
    output: str | None,
    fmt: str,
) -> None:
    """Profile a mechanism's privacy across multiple alpha values."""
    logger = get_logger("cli")
    cfg = _setup_config(ctx)

    mechir = _load_mechanism(mechanism)
    alpha_list = [float(a.strip()) for a in alphas.split(",")]

    engine = CEGAREngine(config=CEGARConfig(timeout_seconds=cfg.cegar.timeout_seconds))

    profile_data: list[dict[str, Any]] = []
    for alpha in alpha_list:
        budget = RDPBudget(alpha=alpha, epsilon=1.0)
        result = engine.verify(mechir, budget)
        profile_data.append({
            "alpha": alpha,
            "verdict": result.verdict.value if hasattr(result.verdict, "value") else str(result.verdict),
            "bounds": result.final_bounds,
        })

    if fmt == "json":
        text = json.dumps(profile_data, indent=2, default=str)
    elif fmt == "csv":
        lines = ["alpha,verdict,bounds"]
        for row in profile_data:
            lines.append(f"{row['alpha']},{row['verdict']},{row['bounds']}")
        text = "\n".join(lines)
    else:
        lines = [f"{'Alpha':>8}  {'Verdict':<16}  Bounds"]
        lines.append("-" * 50)
        for row in profile_data:
            lines.append(f"{row['alpha']:>8.1f}  {row['verdict']:<16}  {row['bounds']}")
        text = "\n".join(lines)

    _write_output(text, output)


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------


@cli.command()
@click.option(
    "--suite",
    type=click.Choice(["basic", "standard", "full"]),
    default="standard",
    help="Benchmark suite to run.",
)
@click.option("--output-dir", type=click.Path(), default="benchmark_results", help="Output directory.")
@click.option("--timeout", type=int, default=60, help="Per-mechanism timeout in seconds.")
@click.option("--compare", type=click.Path(exists=True), default=None, help="Previous results file to compare against.")
@click.pass_context
def benchmark(
    ctx: click.Context,
    suite: str,
    output_dir: str,
    timeout: int,
    compare: str | None,
) -> None:
    """Run verification benchmarks across a suite of mechanisms."""
    logger = get_logger("cli")
    cfg = _setup_config(ctx)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    suite_mechanisms = _get_benchmark_suite(suite)
    engine = CEGAREngine(config=CEGARConfig(timeout_seconds=timeout))

    results: list[dict[str, Any]] = []
    for entry in suite_mechanisms:
        name = entry["name"]
        logger.info("Benchmarking: %s", name)
        start = time.monotonic()
        try:
            mechir = _load_mechanism(entry["path"])
            budget = _parse_budget(entry["budget"], entry.get("notion", "pure"))
            result = engine.verify(mechir, budget)
            elapsed = time.monotonic() - start
            results.append({
                "name": name,
                "verdict": result.verdict.value if hasattr(result.verdict, "value") else str(result.verdict),
                "time_seconds": round(elapsed, 3),
                "statistics": result.statistics,
            })
        except Exception as exc:
            elapsed = time.monotonic() - start
            results.append({
                "name": name,
                "verdict": "ERROR",
                "time_seconds": round(elapsed, 3),
                "error": str(exc),
            })

    results_file = out_path / "results.json"
    results_file.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    click.echo(f"Benchmark results written to {results_file}")

    if compare:
        prev = json.loads(Path(compare).read_text(encoding="utf-8"))
        _print_benchmark_comparison(prev, results)


def _get_benchmark_suite(suite: str) -> list[dict[str, Any]]:
    """Return the list of mechanisms for a benchmark suite."""
    suites: dict[str, list[dict[str, Any]]] = {
        "basic": [
            {"name": "laplace", "path": "benchmarks/laplace.json", "budget": "eps=1.0", "notion": "pure"},
        ],
        "standard": [
            {"name": "laplace", "path": "benchmarks/laplace.json", "budget": "eps=1.0", "notion": "pure"},
            {"name": "gaussian", "path": "benchmarks/gaussian.json", "budget": "eps=1.0,delta=1e-5", "notion": "approx"},
            {"name": "exponential", "path": "benchmarks/exponential.json", "budget": "eps=2.0", "notion": "pure"},
        ],
        "full": [
            {"name": "laplace", "path": "benchmarks/laplace.json", "budget": "eps=1.0", "notion": "pure"},
            {"name": "gaussian", "path": "benchmarks/gaussian.json", "budget": "eps=1.0,delta=1e-5", "notion": "approx"},
            {"name": "exponential", "path": "benchmarks/exponential.json", "budget": "eps=2.0", "notion": "pure"},
            {"name": "sparse_vector", "path": "benchmarks/sparse_vector.json", "budget": "eps=1.0", "notion": "pure"},
            {"name": "noisy_max", "path": "benchmarks/noisy_max.json", "budget": "eps=1.0", "notion": "pure"},
        ],
    }
    return suites.get(suite, suites["standard"])


def _print_benchmark_comparison(
    previous: list[dict[str, Any]], current: list[dict[str, Any]]
) -> None:
    """Print a comparison between previous and current benchmark runs."""
    prev_map = {r["name"]: r for r in previous}
    lines = [
        f"\n{'Mechanism':<20} {'Prev Verdict':<14} {'Curr Verdict':<14} {'Prev Time':>10} {'Curr Time':>10} {'Δ':>8}",
        "-" * 76,
    ]
    for entry in current:
        name = entry["name"]
        prev = prev_map.get(name, {})
        prev_t = prev.get("time_seconds", float("nan"))
        curr_t = entry.get("time_seconds", float("nan"))
        delta = curr_t - prev_t if prev_t == prev_t and curr_t == curr_t else float("nan")
        lines.append(
            f"{name:<20} {prev.get('verdict', 'N/A'):<14} {entry.get('verdict', 'N/A'):<14} "
            f"{prev_t:>9.3f}s {curr_t:>9.3f}s {delta:>+7.3f}s"
        )
    click.echo("\n".join(lines))


# ---------------------------------------------------------------------------
# dump-config
# ---------------------------------------------------------------------------


@cli.command("dump-config")
@click.pass_context
def dump_config(ctx: click.Context) -> None:
    """Dump the current (or default) configuration as JSON."""
    cfg = _setup_config(ctx)
    click.echo(cfg.to_json(indent=2))


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("mechanism", type=click.Path(exists=True))
@click.pass_context
def info(ctx: click.Context, mechanism: str) -> None:
    """Display information about a mechanism."""
    mechir = _load_mechanism(mechanism)
    fmt = ctx.obj.get("output_format", "text")

    if fmt == "json":
        data = {
            "type": type(mechir).__name__,
            "repr": str(mechir),
        }
        click.echo(json.dumps(data, indent=2, default=str))
    else:
        click.echo(f"Mechanism type : {type(mechir).__name__}")
        click.echo(f"Representation : {mechir}")


# ---------------------------------------------------------------------------
# check (parse-only validation)
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("mechanism", type=click.Path(exists=True))
@click.pass_context
def check(ctx: click.Context, mechanism: str) -> None:
    """Parse and validate a mechanism file without running verification.

    Useful for fast feedback when writing or editing mechanism files.
    Supports .py (annotated Python source), .json, and .yaml formats.
    """
    logger = get_logger("cli")
    mech_path = Path(mechanism)

    if mech_path.suffix == ".py":
        from dpcegar.parser.ast_bridge import parse_mechanism_lenient

        source = mech_path.read_text(encoding="utf-8")
        mechir, errors = parse_mechanism_lenient(source, file=str(mech_path))
        if errors:
            for err in errors:
                click.echo(f"Error: {err}", err=True)
            ctx.exit(1)
        else:
            click.echo(f"✓ {mechanism} parsed successfully")
            click.echo(f"  Mechanism: {mechir.name}")
    else:
        try:
            mechir = _load_mechanism(mechanism)
            click.echo(f"✓ {mechanism} loaded successfully")
            click.echo(f"  Mechanism: {type(mechir).__name__}")
        except Exception as exc:
            click.echo(f"Error: {exc}", err=True)
            ctx.exit(1)


# ---------------------------------------------------------------------------
# init-mechanism (generate starter .py template)
# ---------------------------------------------------------------------------

_MECHANISM_TEMPLATES: dict[str, str] = {
    "laplace": '''\
# @dp.mechanism(privacy="1.0-dp", sensitivity=1.0)
def my_mechanism(db, query):
    """Laplace mechanism: adds Lap(sensitivity/epsilon) noise."""
    # @dp.sensitivity(1.0)
    true_answer = query(db)
    # @dp.noise(kind="laplace", scale=1.0)
    noise = laplace(0, 1.0)
    result = true_answer + noise
    return result
''',
    "gaussian": '''\
# @dp.mechanism(privacy="(1.0, 1e-5)-dp", sensitivity=1.0)
def my_mechanism(db, query):
    """Gaussian mechanism: adds N(0, sigma^2) noise."""
    # @dp.sensitivity(1.0)
    true_answer = query(db)
    # @dp.noise(kind="gaussian", sigma=1.0)
    noise = gaussian(0, 1.0)
    result = true_answer + noise
    return result
''',
    "exponential": '''\
# @dp.mechanism(privacy="1.0-dp", sensitivity=1.0)
def my_mechanism(db, scores):
    """Exponential mechanism: selects an item with probability proportional to exp(epsilon * score / (2 * sensitivity))."""
    # @dp.sensitivity(1.0)
    score_values = scores(db)
    # @dp.noise(kind="exponential", sensitivity=1.0, epsilon=1.0)
    selected = exponential_select(score_values, 1.0, 1.0)
    return selected
''',
    "custom": '''\
# @dp.mechanism(privacy="1.0-dp", sensitivity=1.0)
def my_mechanism(db, query):
    """Custom DP mechanism — edit annotations and logic below."""
    # @dp.sensitivity(1.0)
    true_answer = query(db)
    # @dp.noise(kind="laplace", scale=1.0)
    noise = laplace(0, 1.0)
    result = true_answer + noise
    return result
''',
}


@cli.command("init-mechanism")
@click.argument("name", default="my_mechanism.py")
@click.option(
    "--template", "-t",
    type=click.Choice(list(_MECHANISM_TEMPLATES.keys())),
    default="laplace",
    help="Mechanism template to generate.",
)
def init_mechanism(name: str, template: str) -> None:
    """Generate a starter .py mechanism file from a template.

    Creates a new annotated Python file that can be verified with
    ``dpcegar verify`` or checked with ``dpcegar check``.
    """
    dest = Path(name)
    if dest.exists():
        raise click.ClickException(f"File already exists: {dest}")

    content = _MECHANISM_TEMPLATES[template]
    dest.write_text(content, encoding="utf-8")
    click.echo(f"Created {dest} (template: {template})")
    click.echo(f"  Verify: dpcegar verify {dest} --budget \"eps=1.0\"")
    click.echo(f"  Check:  dpcegar check {dest}")


if __name__ == "__main__":
    cli()
