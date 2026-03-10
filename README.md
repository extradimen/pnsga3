# PNSGA3 — Parallel NSGA-III experiments

Experiments comparing **NSGA-III** and **Parallel NSGA3** (multi-island, reference-direction based). Uses a vendored [pymoo](https://pymoo.org/) with custom `NSGA3` and `ParallelNSGA3` in `pymoo/algorithms/moo/nsga3.py`. Results are written under `nsga_logs/` (or a path you pass) for resume and plotting.

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

    # Flags
    pnsga3_only: true
    output_dir: nsga_logs
    metrics_every_gen: false   # false = IGD/HV only on last generation
    hv_enabled: false          # false = disable HV entirely (IGD only)
    pymoo_timing: true         # true  = print detailed per-generation timing
```

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
python nsga3_experiment.py --params "problem=c1dtlz1 n_var=12 n_obj=6,7,8 pop_size=100,200 n_gen=50 n_partitions=6 n_islands=6,8 migration_interval=3 migration_rate=0.1 seed=1 pnsga3_only=0 output_dir=nsga_logs"

# Or individual args
python nsga3_experiment.py --problem c1dtlz1 --n_obj 6,7,8 --pop_size 100,200 --output_dir nsga_logs

# Advanced: timing + IGD/HV control (when not using YAML flags)
# - --pymoo_timing          print [timing]/[advance]/[survival] each generation (for performance debugging)
# - --no_metrics_every_gen  compute IGD/HV only on the last generation (SUMMARY still records final IGD/HV)
# - --no_hv                 disable all HV computations (IGD only, HV columns become NaN)
python nsga3_experiment.py \
  --params "problem=c1dtlz1 n_var=12 n_obj=13,16,19 pop_size=120 n_gen=50 n_partitions=3 n_islands=4 migration_interval=3 migration_rate=0.1 seed=1 pnsga3_only=1 output_dir=nsga_logs" \
  --pymoo_timing \
  --no_metrics_every_gen \
  --no_hv
```

Output layout under `output_dir` (default `nsga_logs/`):

- `output_dir/<problem>/NSGA3/*_final.npz` — NSGA-III cache
- `output_dir/<problem>/ParallelNSGA3/*.npy` — PNSGA3 history (for line plots); `*_iter*.npz` — per-generation fronts (for scatter)
- `output_dir/<problem>/SUMMARY/*.npy` — one file per parameter combo for resume

### 2. Plot (line + scatter)

Run from the repo root (with venv activated), using the **same YAML config and experiment name** as the grid runner:

```bash
python plot.py \
  --config config/experiments.yaml \
  --exp_name c1dtlz1_13_16_19 \
  --mode both    # or: line / scatter
```

This will:

- Plot IGD/HV line charts for the parameter combination selected by the config (`metrics=("igd","hv")` by default).
- Plot ParallelNSGA3 scatter plots (objective space fronts every few generations) into:
  - `output_dir/<problem>/SCATTER/*.png` (separate from the npz/npy caches).

### 3. Load summary DataFrame

```python
from load_summary import load_df_from_disk

df = load_df_from_disk(root_dir="nsga_logs", verbose=True)
```

Scans only `SUMMARY/*.npy` under `root_dir` and returns a DataFrame with the same columns as the experiment script.

## Project layout

- `nsga3_experiment.py` — grid runner and single-run logic (NSGA3 + ParallelNSGA3, cache, plots)
- `plot.py` — line and scatter plotting from disk (by problem / by param lists)
- `load_summary.py` — build DataFrame from `SUMMARY/*.npy`
- `pymoo/` — vendored pymoo with `NSGA3` and `ParallelNSGA3` in `pymoo/algorithms/moo/nsga3.py`

## License

This project is licensed under the Apache License 2.0.  
Vendored `pymoo` code is also under Apache License 2.0; original copyright
and license notices are preserved.
