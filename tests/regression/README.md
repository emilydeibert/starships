# STARSHIPS Regression Tests

These tests verify that numerical results are unchanged after non-scientific
code modifications (cleanup, refactoring, dependency updates).

## How It Works

1. **Before any cleanup**: generate *golden outputs* (reference values)
2. **After each change**: compare new outputs against the references
3. If outputs differ → the change affected the scientific results

## Global Configuration (`~/.starships/config.yaml`)

STARSHIPS stores machine-specific paths in `~/.starships/config.yaml`.
Set it once on each machine:

```python
from pathlib import Path
from starships.config import edit_config

edit_config(
    data_dir=str(Path.home() / 'scratch' / 'starships_data'),
    output_dir=str(Path.home() / 'scratch' / 'starships_output'),
    scratch_dir=str(Path.home() / 'scratch' / 'starships_scratch'),
    prt_input_data_path=str(Path.home() / 'projects' / 'def-dlafre' / 'bouchea3' / 'input_data'),
    regression_golden_dir=str(Path.home() / 'scratch' / 'starships_regression' / 'golden'),
)
```

## Regression-Specific Configuration

```bash
# Copy template and fill in data paths
cp tests/regression/config_template.yaml ~/.starships/regression_config.yaml
# Edit ~/.starships/regression_config.yaml
```

## Setup on Narval (first time)

```bash
# 1. Set global paths (copy-paste as-is, no edits needed)
python -c "
from pathlib import Path
from starships.config import edit_config
edit_config(
    data_dir=str(Path.home() / 'scratch' / 'starships_data'),
    output_dir=str(Path.home() / 'scratch' / 'starships_output'),
    scratch_dir=str(Path.home() / 'scratch' / 'starships_scratch'),
    prt_input_data_path=str(Path.home() / 'projects' / 'def-dlafre' / 'bouchea3' / 'input_data'),
    regression_golden_dir=str(Path.home() / 'scratch' / 'starships_regression' / 'golden'),
)
"

# 2. Configure datasets
cp tests/regression/config_template.yaml ~/.starships/regression_config.yaml
# Edit the file with real data paths

# 3. Generate golden outputs (run ONCE with current code)
python tests/regression/generate_golden.py
# or via sbatch:
sbatch --export=MODE=generate tests/regression/run_narval.sh
```

## Running the Tests

```bash
# After each code change
pytest tests/regression/ -v

# Unit tests + regression in one go
sbatch --export=MODE=validate tests/regression/run_narval.sh
```

Tests are automatically **skipped** if `~/.starships/regression_config.yaml`
or the golden outputs are missing, so they do not break local development.

## What Is Tested

For each configured dataset (WASP-33b night 1, night 2, WASP-127b, ...):

| Test | Description | Tolerance |
|------|-------------|-----------|
| `test_logl_profile_unchanged` | Full logL(RV) profile identical to reference | rtol=1e-10 |
| `test_spot_checks_exact` | A few individual values exact | rtol=1e-12 |
| `test_peak_rv_unchanged` | logL peak position unchanged | ±step/2 km/s |

## Adding a New Dataset

1. Add an entry to `~/.starships/regression_config.yaml`
2. Re-run `generate_golden.py`

## Re-generating References After an Intentional Change

If you deliberately changed the logL formula (scientific improvement),
re-generate the golden outputs with the new code:

```bash
python tests/regression/generate_golden.py
```
