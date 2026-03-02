# DP-CEGAR API Reference

## CLI Commands

### `dpcegar verify <mechanism> --budget <budget>`

Verify that a mechanism satisfies a differential privacy budget.

**Arguments:**
- `mechanism` — Path to a `.py`, `.json`, or `.yaml` mechanism file.
- `--budget`, `-b` — Privacy budget (e.g., `"eps=1.0"` or `"eps=1.0,delta=1e-5"`).
- `--notion`, `-n` — Privacy notion: `pure`, `approx`, `zcdp`, `rdp`, `fdp`, `gdp` (default: `pure`).
- `--timeout`, `-t` — Solver timeout in seconds.
- `--output`, `-o` — Write result to file.
- `--certificate/--no-certificate` — Produce a verification certificate (default: on).

```bash
dpcegar verify my_mechanism.py --budget "eps=1.0"
dpcegar verify mechanism.json --budget "eps=1.0,delta=1e-5" --notion approx
```

### `dpcegar repair <mechanism> --budget <budget>`

Repair a mechanism to satisfy a differential privacy budget.

**Arguments:**
- `mechanism` — Path to a `.py`, `.json`, or `.yaml` mechanism file.
- `--budget`, `-b` — Target privacy budget.
- `--strategy`, `-s` — Repair strategy: `noise_scale`, `clipping`, `composition`, `combined` (default: `combined`).
- `--max-cost` — Maximum acceptable repair cost.
- `--output`, `-o` — Write result to file.
- `--cost-weights` — Comma-separated cost weights (e.g., `"noise=1.0,clip=0.5"`).

```bash
dpcegar repair my_mechanism.py --budget "eps=1.0" --strategy noise_scale
```

### `dpcegar check <mechanism>`

Parse and validate a mechanism file without running verification. Useful for
fast feedback when writing or editing mechanism files.

```bash
dpcegar check my_mechanism.py
# ✓ my_mechanism.py parsed successfully
#   Mechanism: my_mechanism
```

### `dpcegar init-mechanism [name] --template <template>`

Generate a starter `.py` mechanism file from a template.

**Arguments:**
- `name` — Output filename (default: `my_mechanism.py`).
- `--template`, `-t` — Template: `laplace`, `gaussian`, `exponential`, `custom` (default: `laplace`).

```bash
dpcegar init-mechanism my_mech.py --template gaussian
```

### `dpcegar check-all <mechanism>`

Verify a mechanism under all six supported privacy notions.

```bash
dpcegar check-all mechanism.py --delta 1e-5
```

### `dpcegar profile <mechanism>`

Profile a mechanism's privacy across multiple RDP alpha values.

```bash
dpcegar profile mechanism.py --alphas "2,4,8,16,32,64"
```

### `dpcegar info <mechanism>`

Display information about a parsed mechanism.

```bash
dpcegar info mechanism.py
```

### `dpcegar benchmark`

Run verification benchmarks across a suite of mechanisms.

```bash
dpcegar benchmark --suite standard --output-dir results/
```

---

## Supported Input Formats

| Format | Extension | Description |
|---|---|---|
| Annotated Python | `.py` | Python source with `# @dp.*` comment annotations or `@dp_mechanism()` decorators. This is the recommended format for writing new mechanisms. |
| JSON MechIR | `.json` | Serialised intermediate representation (for tool interop). |
| YAML MechIR | `.yaml`, `.yml` | Same as JSON but in YAML format. |

---

## Python Library API

### Parsing

```python
from dpcegar.parser.ast_bridge import parse_mechanism, parse_mechanism_lenient

# Parse a mechanism from Python source (raises on error)
source = open("my_mechanism.py").read()
mechir = parse_mechanism(source, file="my_mechanism.py")

# Parse with error collection (does not raise)
mechir, errors = parse_mechanism_lenient(source, file="my_mechanism.py")
```

### Verification

```python
from dpcegar.cegar.engine import CEGAREngine, CEGARConfig
from dpcegar.ir.types import PureBudget

engine = CEGAREngine(config=CEGARConfig())
budget = PureBudget(epsilon=1.0)
result = engine.verify(mechir, budget)

print(result.verdict)       # CEGARVerdict.VERIFIED or .VIOLATED
print(result.counterexample) # None if verified
```

### Repair

```python
from dpcegar.repair.synthesizer import RepairSynthesizer

synthesizer = RepairSynthesizer()
result = synthesizer.repair(mechir, budget, strategy="noise_scale")

print(result.verdict)   # RepairVerdict.SUCCESS or .FAILURE
print(result.patch)     # Minimal code patch
```

### Annotation Format

Mechanisms are annotated Python functions. Use comment annotations:

```python
# @dp.mechanism(privacy="1.0-dp", sensitivity=1.0)
def my_mechanism(db, query):
    # @dp.sensitivity(1.0)
    true_answer = query(db)
    # @dp.noise(kind="laplace", scale=1.0)
    noise = laplace(0, 1.0)
    result = true_answer + noise
    return result
```

Or decorator syntax:

```python
@dp_mechanism(epsilon=1.0)
def my_mechanism(db, query):
    true_count = query(db, "COUNT", sensitivity=1)
    noise = laplace(0, 1.0)
    return true_count + noise
```

## Experiment API

### Running Experiments

```bash
python experiments/run_experiments.py
```

This produces `experiments/results.json` with structured results for all 5
research questions (bug detection, multi-notion coverage, repair, scalability,
certificates).

### Programmatic Privacy Loss Checking

```python
from dpcegar.density.privacy_loss import PrivacyLossComputer
from dpcegar.density.ratio_builder import DensityRatioBuilder
from dpcegar.paths.symbolic_path import PathSet, SymbolicPath, PathCondition, NoiseDrawInfo
from dpcegar.ir.types import NoiseKind, IRType, Var, Const

# Build a path set for a Laplace mechanism
ps = PathSet()
ps.add(SymbolicPath(
    path_condition=PathCondition.trivially_true(),
    noise_draws=[NoiseDrawInfo(
        variable="eta", kind=NoiseKind.LAPLACE,
        center_expr=Var(ty=IRType.REAL, name="q"),
        scale_expr=Const.real(1.0), site_id=100,
    )],
    output_expr=Var(ty=IRType.REAL, name="eta"),
))

# Check privacy
dr = DensityRatioBuilder().build(ps)
comp = PrivacyLossComputer()
result = comp.check_pure_dp(dr, epsilon=1.0,
    noise_draws=ps.paths[0].noise_draws, sensitivity=1.0)
print(f"Private: {result.is_private}, Loss: {result.computed_cost}")
# Output: Private: True, Loss: ε=1.0
```
