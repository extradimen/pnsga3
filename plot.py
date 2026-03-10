# -*- coding: utf-8 -*-
"""
Plot NSGA3 vs ParallelNSGA3 line charts (by problem) and scatter plots (ParallelNSGA3 every 5 gens) from npy/npz on disk.

If you still get ValueError (ambiguous truth value) after editing this file, try Kernel -> Restart Kernel and re-run all cells;
or before importing run: import importlib; import plot; importlib.reload(plot)

Path conventions (aligned with structure under output_dir, e.g. exp_logs/):
  output_dir / <problem_name> /  e.g. exp_logs/c1dtlz1/
    NSGA3 /
      <problem>_NSGA3_var<n_var>_obj<n_obj>_pop<pop_size>_gen<n_gen>_seed<seed>_np<n_partitions>_final.npz
    SUMMARY /
      <problem>_SUMMARY_var<n_var>_obj<n_obj>_pop<pop_size>_gen<n_gen>_isl<ni>_mi<mi>_mr<mr>_seed<seed>_np<n_partitions>.npy
    ParallelNSGA3 /   (flat directory, no subdirs)
      <problem>_ParallelNSGA3_var<n_var>_obj<n_obj>_pop<pop_size>_gen<n_gen>_isl<ni>_mi<mi>_mr<mr>_seed<seed>_np<n_partitions>.npy  (for line plots)
      <problem>_ParallelNSGA3_var<n_var>_obj<n_obj>_pop<pop_size>_gen<n_gen>_seed<seed>_isl<ni>_mi<mi>_mr<mr>_np<n_partitions>_iter<gen>.npz  (for scatter; np optional)
"""

from pathlib import Path
import re
import itertools
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, Dict, Any, List, Tuple

import yaml

DEFAULT_OUTPUT_DIR = "exp_logs"


def _get_history(d: Dict[str, Any], *keys: str) -> Any:
    """Return the value of the first existing key in the dict; avoids ambiguous truth value when using 'or' on numpy arrays."""
    for k in keys:
        v = d.get(k)
        if v is not None:
            return v
    return None


def _resolve_base(output_dir: str, problem_name: str) -> Optional[Path]:
    """Resolve data root: must have NSGA3 or ParallelNSGA3 subdir to count as found; tries output_dir, DEFAULT, ~/exp_logs in order."""
    def has_data(p: Path) -> bool:
        return p.exists() and ((p / "NSGA3").exists() or (p / "ParallelNSGA3").exists())

    candidates = [
        Path(output_dir).resolve() / problem_name,
        Path(DEFAULT_OUTPUT_DIR) / problem_name,
        Path.home() / "exp_logs" / problem_name,
    ]
    for base in candidates:
        if has_data(base):
            return base
    print(f"[_resolve_base] No valid data directory found. Tried: {[str(c) for c in candidates]}")
    return None

# ---------------------------------------------------------------------------
# Path and filename conventions (aligned with actual writes under output_dir)
# ---------------------------------------------------------------------------

def _mr_str(x: float) -> str:
    """migration_rate in filenames, e.g. 0.1 -> 0.10"""
    return f"{x:.2f}"


def summary_basename_ref(
    problem_name: str,
    n_var: int,
    n_obj: int,
    pop_size: int,
    n_gen: int,
    seed: int,
) -> str:
    """SUMMARY basename used as reference for NSGA3 (not used in current layout; NSGA3 uses _final.npz)."""
    return f"{problem_name}_SUMMARY_var{n_var}_obj{n_obj}_pop{pop_size}_gen{n_gen}_seed{seed}_ref.npy"


def summary_basename_pnsga3(
    problem_name: str,
    n_var: int,
    n_obj: int,
    pop_size: int,
    n_gen: int,
    n_islands: int,
    migration_interval: int,
    migration_rate: float,
    seed: int,
    n_partitions: int,
) -> str:
    """ParallelNSGA3 SUMMARY basename (includes n_partitions)."""
    return (
        f"{problem_name}_SUMMARY_var{n_var}_obj{n_obj}_pop{pop_size}_gen{n_gen}"
        f"_isl{n_islands}_mi{migration_interval}_mr{_mr_str(migration_rate)}_seed{seed}_np{n_partitions}.npy"
    )


def nsga3_final_basename(
    problem_name: str,
    n_var: int,
    n_obj: int,
    pop_size: int,
    n_gen: int,
    seed: int,
    n_partitions: Optional[int] = None,
) -> str:
    """NSGA3 _final.npz basename; when n_partitions is None, compatible with legacy format (no _np suffix)."""
    base = f"{problem_name}_NSGA3_var{n_var}_obj{n_obj}_pop{pop_size}_gen{n_gen}_seed{seed}"
    if n_partitions is not None:
        base += f"_np{n_partitions}"
    return base + "_final.npz"


def pnsga3_iter_basename(
    problem_name: str,
    n_var: int,
    n_obj: int,
    pop_size: int,
    n_gen: int,
    n_islands: int,
    migration_interval: int,
    migration_rate: float,
    seed: int,
    gen: int,
    n_partitions: Optional[int] = None,
) -> str:
    """ParallelNSGA3 npz basename for a given generation (flat under ParallelNSGA3/); when n_partitions is None, compatible with legacy format."""
    mid = (
        f"{problem_name}_ParallelNSGA3_var{n_var}_obj{n_obj}_pop{pop_size}_gen{n_gen}_seed{seed}"
        f"_isl{n_islands}_mi{migration_interval}_mr{_mr_str(migration_rate)}"
    )
    if n_partitions is not None:
        mid += f"_np{n_partitions}"
    return mid + f"_iter{gen:04d}.npz"


# Backward compatibility (basename without problem no longer used; keep _safe_float_str for possible save_path)
def _safe_float_str(x: float) -> str:
    """For filenames: e.g. 0.1 -> 0p1 (legacy convention; currently _mr_str is used)."""
    return str(x).replace(".", "p")


def _parse_summary_basename_ref(basename: str) -> Optional[Dict[str, Any]]:
    """Parse NSGA3 reference SUMMARY basename (current layout has no ref in SUMMARY; kept for compatibility)."""
    m = re.match(
        r"^(.+)_SUMMARY_var(\d+)_obj(\d+)_pop(\d+)_gen(\d+)_seed(\d+)_ref\.npy$",
        basename,
    )
    if not m:
        return None
    return {
        "problem_name": m.group(1),
        "n_var": int(m.group(2)),
        "n_obj": int(m.group(3)),
        "pop_size": int(m.group(4)),
        "n_gen": int(m.group(5)),
        "seed": int(m.group(6)),
        "is_ref": True,
        "n_islands": None,
        "migration_interval": None,
        "migration_rate": None,
    }


def _parse_summary_basename_pnsga3(basename: str) -> Optional[Dict[str, Any]]:
    """Parse ParallelNSGA3 SUMMARY basename: _SUMMARY_var..._isl..._seed*.npy, optional _np<n_partitions> at end."""
    m = re.match(
        r"^(.+)_SUMMARY_var(\d+)_obj(\d+)_pop(\d+)_gen(\d+)_isl(\d+)_mi(\d+)_mr([\d.]+)_seed(\d+)(?:_np(\d+))?\.npy$",
        basename,
    )
    if not m:
        return None
    out = {
        "problem_name": m.group(1),
        "n_var": int(m.group(2)),
        "n_obj": int(m.group(3)),
        "pop_size": int(m.group(4)),
        "n_gen": int(m.group(5)),
        "n_islands": int(m.group(6)),
        "migration_interval": int(m.group(7)),
        "migration_rate": float(m.group(8)),
        "seed": int(m.group(9)),
        "is_ref": False,
    }
    out["n_partitions"] = int(m.group(10)) if m.group(10) else None
    return out


def _parse_pnsga3_npy_basename(basename: str) -> Optional[Dict[str, Any]]:
    """Parse ParallelNSGA3 directory npy basename: same param format for _ParallelNSGA3_ or _SUMMARY_."""
    m = re.match(
        r"^(.+)_ParallelNSGA3_var(\d+)_obj(\d+)_pop(\d+)_gen(\d+)_isl(\d+)_mi(\d+)_mr([\d.]+)_seed(\d+)(?:_np(\d+))?\.npy$",
        basename,
    )
    if m:
        out = {
            "problem_name": m.group(1),
            "n_var": int(m.group(2)),
            "n_obj": int(m.group(3)),
            "pop_size": int(m.group(4)),
            "n_gen": int(m.group(5)),
            "n_islands": int(m.group(6)),
            "migration_interval": int(m.group(7)),
            "migration_rate": float(m.group(8)),
            "seed": int(m.group(9)),
            "is_ref": False,
        }
        out["n_partitions"] = int(m.group(10)) if m.group(10) else None
        return out
    return _parse_summary_basename_pnsga3(basename)


def _match_problem(params: Dict[str, Any], n_var: int, n_obj: int, pop_size: int, n_gen: int, seed: int) -> bool:
    return (
        params["n_var"] == n_var
        and params["n_obj"] == n_obj
        and params["pop_size"] == pop_size
        and params["n_gen"] == n_gen
        and params["seed"] == seed
    )


# ---------------------------------------------------------------------------
# Line plots: by problem, one figure per problem, different algo params = different curves
# ---------------------------------------------------------------------------

def plot_line_by_problem(
    problem_name: str,
    n_var: int,
    n_obj: int,
    pop_size: int,
    n_gen: int,
    n_partitions: Optional[int],
    seed_list: List[int],
    output_dir: str = DEFAULT_OUTPUT_DIR,
    *,
    n_islands_list: Optional[List[int]] = None,
    migration_interval_list: Optional[List[int]] = None,
    migration_rate_list: Optional[List[float]] = None,
    metrics: Tuple[str, ...] = ("igd", "hv"),
    figsize_per_metric: Tuple[float, float] = (8, 5),
    save_dir: Optional[str] = None,
    title_prefix: str = "",
) -> List[Path]:
    """
    Plot line charts by problem: one figure per (problem, n_var, n_obj, pop_size, n_gen, n_partitions);
    within each figure, curves = (n_islands, migration_interval, migration_rate, seed) for PNSGA3 + NSGA3 per seed.

    Grouping:
    - Chart = (problem_name, n_var, n_obj, pop_size, n_gen, n_partitions) -> one figure.
    - Curves = (n_islands, migration_interval, migration_rate, seed) for PNSGA3 + NSGA3 baseline (one per seed).
    - NSGA3: dark gray; PNSGA3: colorful.

    Data sources:
    - NSGA3: from output_dir/<problem>/NSGA3/ (*_final.npz);
    - ParallelNSGA3: from output_dir/<problem>/ParallelNSGA3/*.npy or SUMMARY/*.npy.

    Returns list of saved image paths.
    """
    base = _resolve_base(output_dir, problem_name)
    if base is None:
        print(f"[plot_line_by_problem] No data found: {output_dir}/{problem_name} (also tried {DEFAULT_OUTPUT_DIR}/{problem_name})")
        return []
    nsga3_dir = base / "NSGA3"
    pnsga3_dir = base / "ParallelNSGA3"
    if save_dir is None:
        save_dir = base / "line_plots"
    else:
        save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1) Collect all algo configs and their histories for this chart
    curves: List[Dict[str, Any]] = []  # [{"label": str, "igd_history": array, "hv_history": array, "is_nsga3": bool}, ...]

    # NSGA3 baseline: one curve per seed
    for seed in seed_list:
        final_npz = nsga3_dir / nsga3_final_basename(problem_name, n_var, n_obj, pop_size, n_gen, seed, n_partitions=n_partitions)
        if not final_npz.exists():
            continue
        try:
            data = np.load(final_npz, allow_pickle=True)
            igd = data.get("igd_history")
            hv = data.get("hv_history")
            if igd is not None or hv is not None:
                curves.append({
                    "label": f"NSGA3 seed={seed}",
                    "igd_history": np.asarray(igd) if igd is not None else None,
                    "hv_history": np.asarray(hv) if hv is not None else None,
                    "is_nsga3": True,
                })
        except Exception as e:
            print(f"[plot_line_by_problem] Failed to load NSGA3 {final_npz.name}: {e}")

    # 2) ParallelNSGA3 curves: (n_islands, migration_interval, migration_rate, seed) per combo
    algo_set = None
    if n_islands_list is not None and migration_interval_list is not None and migration_rate_list is not None:
        algo_set = set(
            (ni, mi, round(mr, 4), s)
            for ni, mi, mr, s in itertools.product(
                n_islands_list, migration_interval_list, migration_rate_list, seed_list
            )
        )

    def _match_pnsga3(params: Dict[str, Any]) -> bool:
        if params.get("n_partitions") != n_partitions:
            return False
        return (
            params["n_var"] == n_var
            and params["n_obj"] == n_obj
            and params["pop_size"] == pop_size
            and params["n_gen"] == n_gen
        )

    n_pnsga3_added = 0

    def _load_pnsga3_curves_from_dir(directory: Path, from_summary: bool = False) -> None:
        nonlocal n_pnsga3_added
        if not directory.exists():
            return
        for f in directory.glob("*.npy"):
            params = _parse_pnsga3_npy_basename(f.name) if not from_summary else _parse_summary_basename_pnsga3(f.name)
            if params is None:
                continue
            if (params.get("problem_name") or "").lower() != (problem_name or "").lower() or not _match_pnsga3(params):
                continue
            if params["seed"] not in seed_list:
                continue
            if algo_set is not None:
                key_algo = (params["n_islands"], params["migration_interval"], round(params["migration_rate"], 4), params["seed"])
                if key_algo not in algo_set:
                    continue
            try:
                summary = np.load(f, allow_pickle=True).item()
            except Exception as e:
                print(f"[plot_line_by_problem] Skipped (load failed) {f.name}: {e}")
                continue
            igd = _get_history(summary, "igd_history", "IGD_history", "igd")
            hv = _get_history(summary, "hv_history", "HV_history", "hv")
            if igd is not None:
                igd = np.asarray(igd)
            if hv is not None:
                hv = np.asarray(hv)
            if (igd is not None and len(igd) > 0) or (hv is not None and len(hv) > 0):
                label = f"PNSGA3 ni={params['n_islands']} mi={params['migration_interval']} mr={params['migration_rate']} s={params['seed']}"
                curves.append({
                    "label": label,
                    "igd_history": igd,
                    "hv_history": hv,
                    "is_nsga3": False,
                })
                n_pnsga3_added += 1

    _load_pnsga3_curves_from_dir(pnsga3_dir, from_summary=False)
    if n_pnsga3_added == 0:
        summary_dir = base / "SUMMARY"
        _load_pnsga3_curves_from_dir(summary_dir, from_summary=True)

    if not curves:
        print(f"[plot_line_by_problem] No valid curve data (dir: {base})")
        return []

    # 3) Plot: NSGA3 dark gray, PNSGA3 colorful
    saved = []
    np_str = f" np={n_partitions}" if n_partitions is not None else ""
    title_base = f"{title_prefix}{problem_name} (n_var={n_var}, n_obj={n_obj}, pop={pop_size}, gen={n_gen}{np_str})"
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(figsize_per_metric[0] * n_metrics, figsize_per_metric[1]))
    if n_metrics == 1:
        axes = [axes]
    try:
        cmap = plt.colormaps.get_cmap("tab10")
    except AttributeError:
        cmap = plt.get_cmap("tab10")
    for ax, metric in zip(axes, metrics):
        key = f"{metric}_history"
        nsga3_idx = 0
        pnsga3_idx = 0
        for c in curves:
            hist = c.get(key)
            if hist is None or len(hist) == 0:
                continue
            gens = np.arange(len(hist))
            if c.get("is_nsga3"):
                color = np.array([0.35, 0.35, 0.35]) - nsga3_idx * 0.05
                color = np.clip(color, 0.1, 0.9)
                ax.plot(gens, hist, label=c["label"], alpha=0.9, color=tuple(color))
                nsga3_idx += 1
            else:
                color = cmap(pnsga3_idx % 10)
                ax.plot(gens, hist, label=c["label"], alpha=0.8, color=color)
                pnsga3_idx += 1
        ax.set_xlabel("Generation")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"{metric.upper()} history")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle(title_base, y=1.02)
    fig.tight_layout()
    np_suffix = f"_np{n_partitions}" if n_partitions is not None else ""
    out = save_dir / f"line_{problem_name}_nv{n_var}_no{n_obj}_p{pop_size}_g{n_gen}{np_suffix}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(out)
    return saved


# ---------------------------------------------------------------------------
# Retrieve by param lists and plot: line charts (one per problem), scatter (one set per problem+algo params)
# ---------------------------------------------------------------------------

def plot_line_by_problem_from_lists(
    problem_name: str,
    n_var_list: List[int],
    n_obj_list: List[int],
    pop_size_list: List[int],
    n_gen_list: List[int],
    n_partitions_list: List[Optional[int]],
    seed_list: List[int],
    n_islands_list: List[int],
    migration_interval_list: List[int],
    migration_rate_list: List[float],
    output_dir: str = DEFAULT_OUTPUT_DIR,
    *,
    metrics: Tuple[str, ...] = ("igd", "hv"),
    figsize_per_metric: Tuple[float, float] = (8, 5),
    save_dir: Optional[str] = None,
    title_prefix: str = "",
) -> List[Path]:
    """
    Plot line charts: one figure per (n_var, n_obj, pop_size, n_gen, n_partitions);
    within each figure, curves = (n_islands, migration_interval, migration_rate, seed) for PNSGA3 + NSGA3 per seed.

    Chart grouping (one figure per combo): problem, n_var, n_obj, pop_size, n_gen, n_partitions.
    Line grouping (curves within figure): n_islands, migration_interval, migration_rate, seed + NSGA3 per seed.

    Returns list of all saved image paths.
    """
    all_saved: List[Path] = []
    for n_var, n_obj, pop_size, n_gen, n_partitions in itertools.product(
        n_var_list, n_obj_list, pop_size_list, n_gen_list, n_partitions_list
    ):
        saved = plot_line_by_problem(
            problem_name=problem_name,
            n_var=n_var,
            n_obj=n_obj,
            pop_size=pop_size,
            n_gen=n_gen,
            n_partitions=n_partitions,
            seed_list=seed_list,
            output_dir=output_dir,
            n_islands_list=n_islands_list,
            migration_interval_list=migration_interval_list,
            migration_rate_list=migration_rate_list,
            metrics=metrics,
            figsize_per_metric=figsize_per_metric,
            save_dir=save_dir,
            title_prefix=title_prefix,
        )
        all_saved.extend(saved)
    return all_saved


def plot_scatter_from_lists(
    problem_name: str,
    n_var_list: List[int],
    n_obj_list: List[int],
    pop_size_list: List[int],
    n_gen_list: List[int],
    seed_list: List[int],
    n_islands_list: List[int],
    migration_interval_list: List[int],
    migration_rate_list: List[float],
    n_partitions_list: Optional[List[int]] = None,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    *,
    gen_interval: int = 5,
    obj_indices: Optional[Tuple[int, ...]] = None,
    figsize_subplot: Tuple[float, float] = (5, 5),
) -> List[Path]:
    """
    Retrieve by param lists and plot one set of scatter plots per (problem + algo params); ParallelNSGA3 only.
    Same problem+params = one set: one subplot every gen_interval generations (e.g. 20 gens, gen_interval=5 -> 4 subplots).

    Parameters:
    - problem_name: problem name
    - n_var_list, n_obj_list, pop_size_list, n_gen_list, seed_list: problem param lists
    - n_islands_list, migration_interval_list, migration_rate_list: algo param lists
    - n_partitions_list: optional; if set, one set of plots per n_partitions (iter npz basename must include _np<n_partitions>).

    For each (n_var, n_obj, pop_size, n_gen, seed) and each (n_islands, migration_interval, migration_rate[, n_partitions])
    Cartesian product, if corresponding ParallelNSGA3 iter npz exist on disk, plot one set of scatter plots.

    Returns list of all saved image paths.
    """
    saved: List[Path] = []
    problem_tuples = list(itertools.product(n_var_list, n_obj_list, pop_size_list, n_gen_list, seed_list))
    algo_tuples = list(itertools.product(n_islands_list, migration_interval_list, migration_rate_list))
    if n_partitions_list is not None:
        for (n_var, n_obj, pop_size, n_gen, seed), (n_islands, migration_interval, migration_rate), n_partitions in itertools.product(
            problem_tuples, algo_tuples, n_partitions_list
        ):
            path = plot_scatter_parallel_nsga3(
                problem_name=problem_name,
                n_var=n_var,
                n_obj=n_obj,
                pop_size=pop_size,
                n_gen=n_gen,
                n_islands=n_islands,
                migration_interval=migration_interval,
                migration_rate=migration_rate,
                seed=seed,
                n_partitions=n_partitions,
                output_dir=output_dir,
                gen_interval=gen_interval,
                obj_indices=obj_indices,
                figsize_subplot=figsize_subplot,
            )
            if path is not None:
                saved.append(path)
    else:
        for (n_var, n_obj, pop_size, n_gen, seed), (n_islands, migration_interval, migration_rate) in itertools.product(
            problem_tuples, algo_tuples
        ):
            path = plot_scatter_parallel_nsga3(
                problem_name=problem_name,
                n_var=n_var,
                n_obj=n_obj,
                pop_size=pop_size,
                n_gen=n_gen,
                n_islands=n_islands,
                migration_interval=migration_interval,
                migration_rate=migration_rate,
                seed=seed,
                output_dir=output_dir,
                gen_interval=gen_interval,
                obj_indices=obj_indices,
                figsize_subplot=figsize_subplot,
            )
            if path is not None:
                saved.append(path)
    return saved


# ---------------------------------------------------------------------------
# Scatter plots: ParallelNSGA3 only, one subplot every 5 gens, one set of figures per config
# ---------------------------------------------------------------------------

def plot_scatter_parallel_nsga3(
    problem_name: str,
    n_var: int,
    n_obj: int,
    pop_size: int,
    n_gen: int,
    n_islands: int,
    migration_interval: int,
    migration_rate: float,
    seed: int,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    *,
    n_partitions: Optional[int] = None,
    gen_interval: int = 5,
    obj_indices: Optional[Tuple[int, ...]] = None,
    figsize_subplot: Tuple[float, float] = (5, 5),
    save_path: Optional[str] = None,
) -> Optional[Path]:
    """
    For one ParallelNSGA3 param set, plot one set of scatter plots: one subplot every gen_interval generations (e.g. 20 gens -> 0, 5, 10, 15, 4 subplots).
    Objective space: if n_obj==2 use f1-f2; if n_obj>=3 default 3D (f1,f2,f3), override via obj_indices.
    Data: flat *_iterXXXX.npz under output_dir/<problem>/ParallelNSGA3/; if n_partitions passed, basename includes _np<n_partitions>.
    """
    base = _resolve_base(output_dir, problem_name)
    if base is None:
        print(f"[plot_scatter_parallel_nsga3] Directory not found: {output_dir}/{problem_name} (also tried {DEFAULT_OUTPUT_DIR}/{problem_name})")
        return None
    run_dir = base / "ParallelNSGA3"
    if not run_dir.exists():
        print(f"[plot_scatter_parallel_nsga3] Directory does not exist: {run_dir}")
        return None
    scatter_dir = base / "SCATTER"

    # Generations to plot: 0, gen_interval, 2*gen_interval, ... <= n_gen
    gen_list = list(range(0, n_gen + 1, gen_interval))
    if gen_list[-1] != n_gen:
        gen_list.append(n_gen)
    gen_list = sorted(set(gen_list))

    def _load_iter_npz(n_partitions_opt: Optional[int]) -> List[Tuple[int, np.ndarray]]:
        out: List[Tuple[int, np.ndarray]] = []
        for g in gen_list:
            npz_path = run_dir / pnsga3_iter_basename(
                problem_name, n_var, n_obj, pop_size, n_gen,
                n_islands, migration_interval, migration_rate, seed, g,
                n_partitions=n_partitions_opt,
            )
            if not npz_path.exists():
                continue
            data = np.load(npz_path, allow_pickle=True)
            F = data.get("F")
            if F is None and "F" not in data.files:
                for k in data.files:
                    if data[k].ndim == 2 and data[k].shape[1] >= 2:
                        F = data[k]
                        break
            if F is not None and F.ndim == 2:
                out.append((g, np.asarray(F)))
        return out

    loaded = _load_iter_npz(n_partitions)
    # Fallback: if n_partitions given but no _np files found, try legacy format (no _np)
    if not loaded and n_partitions is not None:
        loaded = _load_iter_npz(None)
        if loaded:
            print(f"[plot_scatter_parallel_nsga3] Note: no iter npz with _np{n_partitions} found; used legacy (no _np) files for plot.")
    if not loaded:
        # Diagnostic: list actual files in dir and expected basename to see if missing or naming mismatch
        actual = list(run_dir.glob("*.npz"))[:5]
        expected_name = pnsga3_iter_basename(
            problem_name, n_var, n_obj, pop_size, n_gen,
            n_islands, migration_interval, migration_rate, seed, gen_list[0],
            n_partitions=n_partitions if n_partitions is not None else None,
        )
        print(f"[plot_scatter_parallel_nsga3] No valid iter npz: {run_dir}")
        if actual:
            print(f"  Sample npz in dir: {[f.name for f in actual]}")
        else:
            print(f"  No .npz in dir; run experiment with iter save (save_interval) first.")
        print(f"  Expected basename example (gen={gen_list[0]}): {expected_name}")
        return None

    # Objective dimensions: n_obj>=3 default 3D (f1,f2,f3), else 2D; override via obj_indices
    if obj_indices is not None:
        dims = obj_indices
    else:
        dims = (0, 1, 2) if n_obj >= 3 else ((0, 1) if n_obj >= 2 else (0,))
    n_dims = len(dims)
    if n_dims not in (2, 3):
        dims = (0, 1, 2) if n_obj >= 3 else (0, 1)
        n_dims = 3 if n_obj >= 3 else 2

    n_sub = len(loaded)
    n_col = min(4, n_sub)
    n_row = (n_sub + n_col - 1) // n_col
    use_3d = n_dims == 3
    if use_3d:
        fig = plt.figure(figsize=(figsize_subplot[0] * n_col, figsize_subplot[1] * n_row))
        axes_flat = [
            fig.add_subplot(n_row, n_col, i + 1, projection="3d") for i in range(n_sub)
        ]
    else:
        fig, axes = plt.subplots(n_row, n_col, figsize=(figsize_subplot[0] * n_col, figsize_subplot[1] * n_row))
        if n_sub == 1:
            axes = np.array([[axes]])
        elif axes.ndim == 1:
            axes = axes.reshape(1, -1)
        axes_flat = axes.ravel()

    for idx, (gen, F) in enumerate(loaded):
        ax = axes_flat[idx]
        if n_dims == 2:
            ax.scatter(F[:, dims[0]], F[:, dims[1]], s=8, alpha=0.6)
            ax.set_xlabel(f"$f_{dims[0]+1}$")
            ax.set_ylabel(f"$f_{dims[1]+1}$")
        else:
            ax.scatter(F[:, dims[0]], F[:, dims[1]], F[:, dims[2]], s=8, alpha=0.6)
            ax.set_xlabel(f"$f_{dims[0]+1}$")
            ax.set_ylabel(f"$f_{dims[1]+1}$")
            ax.set_zlabel(f"$f_{dims[2]+1}$")
        ax.set_title(f"gen {gen}")
        ax.grid(True, alpha=0.3)
    for j in range(len(loaded), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        f"{problem_name} — ParallelNSGA3 ni={n_islands} mi={migration_interval} mr={migration_rate}"
        + (f" np={n_partitions}" if n_partitions is not None else "") + f" (seed={seed})"
    )
    fig.tight_layout()

    if save_path is None:
        save_path = scatter_dir / (
            f"scatter_var{n_var}_obj{n_obj}_pop{pop_size}_gen{n_gen}"
            f"_ni{n_islands}_mi{migration_interval}_mr{_mr_str(migration_rate)}"
            + (f"_np{n_partitions}" if n_partitions is not None else "") + f"_s{seed}.png"
        )
    else:
        save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ---------------------------------------------------------------------------
# Usage example (run when executing this file directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot NSGA3 / ParallelNSGA3 results from disk using the same experiment parameters as the grid runner."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to YAML config file with experiments[]. If omitted, you must call plotting functions from Python.",
    )
    parser.add_argument(
        "--exp_name",
        default=None,
        help="Experiment name in config (experiments[].name). If config contains multiple experiments, this is required.",
    )
    parser.add_argument(
        "--mode",
        choices=["line", "scatter", "both"],
        default="both",
        help="What to plot: line charts (IGD/HV), scatter plots, or both.",
    )
    args = parser.parse_args()

    if args.config is None:
        raise SystemExit(
            "Please provide --config pointing to the same YAML file used for nsga3_experiment.py (with experiments[].)."
        )

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise SystemExit(f"Config file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    experiments: List[Dict[str, Any]] = cfg.get("experiments") or []
    if not experiments:
        raise SystemExit(f"No 'experiments' list found in config: {cfg_path}")

    if args.exp_name:
        matches = [e for e in experiments if e.get("name") == args.exp_name]
        if not matches:
            names = [e.get("name") for e in experiments]
            raise SystemExit(f"Experiment '{args.exp_name}' not found in config. Available: {names}")
        exp = matches[0]
    else:
        if len(experiments) > 1:
            names = [e.get("name") for e in experiments]
            raise SystemExit(
                "Config defines multiple experiments. Please select one via --exp_name. "
                f"Available: {names}"
            )
        exp = experiments[0]

    def _to_list(key: str, default=None):
        """Get value as list from experiment."""
        if key not in exp:
            return [default] if default is not None else [None]
        v = exp[key]
        if isinstance(v, list):
            return v if v else ([default] if default is not None else [None])
        return [v]

    problem_name = exp.get("problem", "c1dtlz1")
    n_var_list = _to_list("n_var", 12)
    n_obj_list = _to_list("n_obj", 6)
    pop_size_list = _to_list("pop_size", 100)
    n_gen_list = _to_list("n_gen", 50)
    n_partitions_list = _to_list("n_partitions", None)
    seed_list = _to_list("seed", 1)
    n_islands_list = _to_list("n_islands", 6)
    migration_interval_list = _to_list("migration_interval", 3)
    migration_rate_list = _to_list("migration_rate", 0.1)
    output_dir = exp.get("output_dir", DEFAULT_OUTPUT_DIR)

    # Line plots (IGD/HV history): chart=(n_var,n_obj,pop_size,n_gen,n_partitions), curves=(n_islands,mi,mr,seed)+NSGA3
    if args.mode in ("line", "both"):
        plot_line_by_problem_from_lists(
            problem_name=problem_name,
            n_var_list=n_var_list,
            n_obj_list=n_obj_list,
            pop_size_list=pop_size_list,
            n_gen_list=n_gen_list,
            n_partitions_list=n_partitions_list,
            seed_list=seed_list,
            n_islands_list=n_islands_list,
            migration_interval_list=migration_interval_list,
            migration_rate_list=migration_rate_list,
            output_dir=output_dir,
        )

    # Scatter plots (ParallelNSGA3 fronts over generations)
    if args.mode in ("scatter", "both"):
        plot_scatter_from_lists(
            problem_name=problem_name,
            n_var_list=n_var_list,
            n_obj_list=n_obj_list,
            pop_size_list=pop_size_list,
            n_gen_list=n_gen_list,
            seed_list=seed_list,
            n_islands_list=n_islands_list,
            migration_interval_list=migration_interval_list,
            migration_rate_list=migration_rate_list,
            n_partitions_list=n_partitions_list,
            output_dir=output_dir,
            gen_interval=5,
        )
