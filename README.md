# PNSGA3 — Parallel NSGA-III experiments

Experiments comparing **NSGA-III** and **Parallel NSGA3** (multi-island, reference-direction based). Uses a vendored [pymoo](https://pymoo.org/) with custom `NSGA3` and `ParallelNSGA3` in `pymoo/algorithms/moo/nsga3.py`. Results are written under `nsga_logs/` (or a path you pass) for resume and plotting.

## Features

- **Grid experiments**: parameter sweeps over problem, `n_var`, `n_obj`, `pop_size`, `n_gen`, `n_partitions`, `n_islands`, migration, seed — with run-level resume via `SUMMARY/*.npy`.
- **CLI**: all parameters via `--params "key=value ..."` or individual `--problem`, `--n_obj`, etc.
- **Plotting**: line plots (IGD/HV by generation) and scatter plots (Pareto front) from disk; see `plot.py`.
- **Summary loading**: build a single DataFrame from `SUMMARY/*.npy`; see `load_summary.py`.

## Requirements

- Python 3.8+
- numpy, matplotlib, pandas  
- **pymoo** is included under `pymoo/`; run from the repo root so `from pymoo...` resolves.

Install dependencies (pymoo is in-tree):

```bash
pip install numpy matplotlib pandas
```

Optional: install pymoo from the repo for development:

```bash
cd pnsga3
pip install -e .
```

If you do not install pymoo, run scripts from the **repository root** so the local `pymoo` package is on `PYTHONPATH`:

```bash
cd pnsga3
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python nsga3_experiment.py --help
```

## Usage

### 1. Run experiments (CLI)

All parameters can be passed in one string or separately:

```bash
# Default grid (see --help for defaults)
python nsga3_experiment.py

# Single --params string (lists comma-separated; spaces after commas allowed)
python nsga3_experiment.py --params "problem=c1dtlz1 n_var=12 n_obj=6,7,8 pop_size=100,200 n_gen=50 n_partitions=6 n_islands=6,8 migration_interval=3 migration_rate=0.1 seed=1 pnsga3_only=0 output_dir=nsga_logs"

# Or individual args
python nsga3_experiment.py --problem c1dtlz1 --n_obj 6,7,8 --pop_size 100,200 --output_dir nsga_logs
```

Output layout under `output_dir` (default `nsga_logs/`):

- `output_dir/<problem>/NSGA3/*_final.npz` — NSGA-III cache
- `output_dir/<problem>/ParallelNSGA3/*.npy` — PNSGA3 history (for line plots); `*_iter*.npz` — per-generation fronts (for scatter)
- `output_dir/<problem>/SUMMARY/*.npy` — one file per parameter combo for resume

### 2. Plot (line + scatter)

From repo root (so `plot` and `pymoo` are importable):

```bash
cd pnsga3
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python plot.py
```

Edit the `if __name__ == "__main__"` block in `plot.py` to set `problem_name`, list args, and `output_dir`, or call from your own script:

```python
from plot import plot_line_by_problem_from_lists, plot_scatter_from_lists

plot_line_by_problem_from_lists("c1dtlz1", n_var_list=[12], n_obj_list=[6,8], ...)
plot_scatter_from_lists("c1dtlz1", ..., output_dir="nsga_logs")
```

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

See repository license (e.g. MIT or same as pymoo if you keep their license).
