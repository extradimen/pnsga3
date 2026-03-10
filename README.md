# PNSGA3 — Parallel NSGA-III experiments

Experiments comparing **NSGA-III** and **Parallel NSGA3** (multi-island, reference-direction based). Uses a vendored [pymoo](https://pymoo.org/) with custom `NSGA3` and `ParallelNSGA3` in `pymoo/algorithms/moo/nsga3.py`. Results are written under `exp_logs/` (or a path you pass) for resume and plotting.

## Features

- **Grid experiments**: parameter sweeps over problem, `n_var`, `n_obj`, `pop_size`, `n_gen`, `n_partitions`, `n_islands`, migration, seed — with run-level resume via `SUMMARY/*.npy`.
- **CLI**: all parameters via `--params "key=value ..."` or individual `--problem`, `--n_obj`, etc.
- **Plotting**: line plots (IGD/HV by generation) and scatter plots (Pareto front) from disk; see `plot.py`.
- **Summary loading**: build a single DataFrame from `SUMMARY/*.npy`; see `load_summary.py`.

## Installation

- **Python**: 3.8+
- **Dependencies**: numpy, matplotlib, pandas (pymoo is vendored under `pymoo/`).

**Recommended: clone repo → create venv → install dependencies**

```bash
# 1. Clone the repository
git clone https://github.com/extradimen/pnsga3.git
cd pnsga3

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate   # Linux / macOS
# or: venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

Run all commands (experiment, plot) from the **repository root** so the in-tree `pymoo` package is importable. No need to set `PYTHONPATH` if you are in the project directory and use the same Python that has the venv activated.

## Usage

### 1. Run experiments (CLI)

Recommended usage is to drive experiments from a single YAML config file so that experiments and plots always use the same parameters.

```yaml
# config/experiments.yaml
experiments:
  - name: c1dtlz1_13_16_19
    problem: c1dtlz1
    n_var: [12]
    n_obj: [13, 16, 19]
    pop_size: [120]
    n_gen: [50]
    n_partitions: [3]
    n_islands: [4]
    migration_interval: [3]
    migration_rate: [0.1]
    seed: [1]

    # Flags (see config/experiments.yaml for full parameter reference)
    pnsga3_only: true
    output_dir: exp_logs
    metrics_every_gen: false
    hv_enabled: false
    pymoo_timing: true
```

**Config parameter reference:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | str | Experiment identifier; used by `--exp_name` and in plot output paths. |
| `problem` | str | Problem name (e.g. `c1dtlz1`, `dtlz1`). |
| `n_var` | list[int] | Number of decision variables. |
| `n_obj` | list[int] | Number of objectives. |
| `pop_size` | list[int] | Population size. |
| `n_gen` | list[int] | Number of generations. |
| `n_partitions` | list[int] | Reference direction partitions. |
| `n_islands` | list[int] | Number of islands in ParallelNSGA3. |
| `migration_interval` | list[int] | Generations between migrations. |
| `migration_rate` | list[float] | Fraction of population to migrate (0.0–1.0). |
| `seed` | list[int] | Random seed(s); multiple = multiple runs per config. |
| `output_dir` | str | Directory for logs and caches (default: `exp_logs`). |

**Boolean flags:**

| Flag | `true` | `false` |
|------|--------|---------|
| `pnsga3_only` | Run only ParallelNSGA3; skip NSGA3 baseline. | Run both NSGA3 and ParallelNSGA3 for comparison. |
| `metrics_every_gen` | Compute IGD/HV every generation (needed for line plots). | Compute IGD/HV only on last generation (faster; SUMMARY still has final values). |
| `hv_enabled` | Compute Hypervolume (HV) in addition to IGD. | Disable HV; only IGD/GD (HV columns = NaN; faster for high n_obj). |
| `pymoo_timing` | Print per-generation timing (for debugging). | No per-generation timing output. |
| `plot_after_run` | After each run, show IGD/HV+scatter figure and block until closed. | Do not show figure (for batch/grid; use `plot.py` later). |

See `config/experiments.yaml` for inline comments on each parameter.

Run the grid for a given experiment name:

```bash
python nsga3_experiment.py \
  --config config/experiments.yaml \
  --exp_name c1dtlz1_13_16_19
```

Legacy usage (still supported) – pass all parameters explicitly:

```bash
# Default grid (see --help for defaults)
python nsga3_experiment.py

# Single --params string (lists comma-separated; spaces after commas allowed)
python nsga3_experiment.py --params "problem=c1dtlz1 n_var=12 n_obj=6,7,8 pop_size=100,200 n_gen=50 n_partitions=6 n_islands=6,8 migration_interval=3 migration_rate=0.1 seed=1 pnsga3_only=0 output_dir=exp_logs"

# Or individual args
python nsga3_experiment.py --problem c1dtlz1 --n_obj 6,7,8 --pop_size 100,200 --output_dir exp_logs

# Advanced: timing + IGD/HV control (when not using YAML flags)
# - --pymoo_timing          print [timing]/[advance]/[survival] each generation (for performance debugging)
# - --no_metrics_every_gen  compute IGD/HV only on the last generation (SUMMARY still records final IGD/HV)
# - --no_hv                 disable all HV computations (IGD only, HV columns become NaN)
python nsga3_experiment.py \
  --params "problem=c1dtlz1 n_var=12 n_obj=13,16,19 pop_size=120 n_gen=50 n_partitions=3 n_islands=4 migration_interval=3 migration_rate=0.1 seed=1 pnsga3_only=1 output_dir=exp_logs" \
  --pymoo_timing \
  --no_metrics_every_gen \
  --no_hv
```

Output layout under `output_dir` (default `exp_logs/`):

- `output_dir/<problem>/NSGA3/*_final.npz` — NSGA-III cache
- `output_dir/<problem>/ParallelNSGA3/*.npy` — PNSGA3 history (for line plots); `*_iter*.npz` — per-generation fronts (for scatter)
- `output_dir/<problem>/SUMMARY/*.npy` — one file per parameter combo for resume

### 2. Plot (line + scatter)

Run from the repo root (with venv activated), using the **same YAML config and experiment name** as the grid runner:

```bash
# --mode: line | scatter | both
# --metrics: igd | hv | both (for line plots only)
python plot.py \
  --config config/experiments.yaml \
  --exp_name c1dtlz1_13_16_19 \
  --mode both \
  --metrics both
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `--config` | Path to YAML config (required). |
| `--exp_name` | Experiment name in config; required when config has multiple experiments. |
| `--mode` | `line` — IGD/HV line charts only; `scatter` — Pareto front scatter only; `both` — both (default). |
| `--metrics` | For line plots: `igd` — IGD only; `hv` — HV only; `both` — both metrics (default). |

**Line chart grouping:**

- **One figure per** `(n_var, n_obj, pop_size, n_gen, n_partitions)` — e.g. 3×2×1×1 = 6 figures for `n_obj=[7,8,9]`, `pop_size=[120,150]`, `n_gen=[50]`, `n_partitions=[3]`.
- **Curves within each figure** = PNSGA3 configs `(n_islands, migration_interval, migration_rate, seed)` + NSGA3 baseline (one per seed). NSGA3 uses dark gray; PNSGA3 uses color.

**Output paths** (plots are grouped by `exp_name` to avoid overwriting):

- Line plots: `output_dir/<problem>/line_plots/<exp_name>/*.png`
- Scatter: `output_dir/<problem>/SCATTER/<exp_name>/*.png`

### 3. Export summary to CSV

After running experiments, you can merge all `SUMMARY/*.npy` files into a single CSV table:

```bash
python load_summary.py \
  --root_dir exp_logs \
  --verbose
```

By default, the CSV is saved to `exp_summary/summary.csv` (exp_summary is parallel to `root_dir`). Example layout:

```
project/
  exp_logs/           # --root_dir
    c1dtlz1/
      SUMMARY/*.npy
  exp_summary/       # parallel to exp_logs
    summary.csv      # output
```

Use `--csv_path` to override the output path. This scans only `SUMMARY/*.npy` under `root_dir`, normalizes them to a common schema, and writes a CSV with the same columns as the experiment summary DataFrame.

## Project layout

- `nsga3_experiment.py` — grid runner and single-run logic (NSGA3 + ParallelNSGA3, cache, plots)
- `plot.py` — line and scatter plotting from disk (by problem / by param lists)
- `load_summary.py` — build DataFrame from `SUMMARY/*.npy`
- `pymoo/` — vendored pymoo with `NSGA3` and `ParallelNSGA3` in `pymoo/algorithms/moo/nsga3.py`

## License

This project is licensed under the Apache License 2.0.  
Vendored `pymoo` code is also under Apache License 2.0; original copyright
and license notices are preserved.
