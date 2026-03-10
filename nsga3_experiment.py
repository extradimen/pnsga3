# -*- coding: utf-8 -*-
"""
NSGA3 vs ParallelNSGA3 experiment script.
- NSGA3 results are written to _final.npz under NSGA3/
- ParallelNSGA3 igd_history/hv_history are written to .npy under ParallelNSGA3/ for nsga_plot_utils line plots
- Run-level resume uses .npy under SUMMARY/ to decide whether to skip
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import itertools
import os
import pandas as pd
import time
from typing import Any, Dict, List

import yaml

from pymoo.algorithms.moo.nsga3 import NSGA3, ParallelNSGA3
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.problems import get_problem
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.core.callback import Callback
from pymoo.problems.multi.dascmop import DIFFICULTIES   # Used for DASCMOP
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.util.display.output import pareto_front_if_possible
from pymoo.indicators.gd import GD as GDIndicator


# -------------------------------
# 1. Problem builder
# -------------------------------
def build_problem(problem_name, n_var, n_obj, difficulty_index=9):
    """Return the corresponding pymoo Problem instance for the given problem_name."""

    if problem_name in [
        "dascmop1", "dascmop2", "dascmop3",
        "dascmop4", "dascmop5", "dascmop6"
    ]:
        return get_problem(problem_name, difficulty=difficulty_index)
    elif problem_name in ["dascmop7", "dascmop8", "dascmop9"]:
        difficulty_factors = DIFFICULTIES[difficulty_index - 1]
        return get_problem(problem_name, difficulty_factors=difficulty_factors)
    elif problem_name.startswith("df"):
        return get_problem(problem_name, n_var=n_var)
    else:
        return get_problem(problem_name, n_var=n_var, n_obj=n_obj)


def _ensure_list(x):
    """Ensure a value is a list (used for config-driven experiments)."""
    if isinstance(x, list):
        return x
    return [x]


def _load_experiment_from_config(path: str, exp_name: str | None) -> Dict[str, Any]:
    """
    Load one experiment definition from a YAML config file.

    Expected structure:

    experiments:
      - name: my_experiment
        problem: c1dtlz1
        n_var: [12]
        n_obj: [13,16,19]
        pop_size: [120]
        n_gen: [50]
        n_partitions: [3]
        n_islands: [4]
        migration_interval: [3]
        migration_rate: [0.1]
        seed: [1]
        pnsga3_only: true
        output_dir: nsga_logs
        metrics_every_gen: false
        hv_enabled: false
        pymoo_timing: true
    """
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    experiments: List[Dict[str, Any]] = cfg.get("experiments") or []
    if not experiments:
        raise ValueError(f"No 'experiments' list found in config: {cfg_path}")

    exp: Dict[str, Any]
    if exp_name:
        matches = [e for e in experiments if e.get("name") == exp_name]
        if not matches:
            available = [e.get("name") for e in experiments]
            raise ValueError(f"Experiment '{exp_name}' not found in config. Available: {available}")
        exp = matches[0]
    else:
        if len(experiments) > 1:
            names = [e.get("name") for e in experiments]
            raise ValueError(
                f"Config {cfg_path} defines multiple experiments. "
                f"Please select one via --exp_name. Available: {names}"
            )
        exp = experiments[0]

    def get_list(key: str, default=None):
        if key not in exp:
            return default
        return _ensure_list(exp[key])

    out: Dict[str, Any] = {}
    out["problem_list"] = get_list("problem", ["c1dtlz1"])
    out["n_var_list"] = get_list("n_var", [12])
    out["n_obj_list"] = get_list("n_obj", [6])
    out["pop_size_list"] = get_list("pop_size", [100])
    out["n_gen_list"] = get_list("n_gen", [50])
    out["n_partitions_list"] = get_list("n_partitions", [6])
    out["n_islands_list"] = get_list("n_islands", [6])
    out["migration_interval_list"] = get_list("migration_interval", [3])
    out["migration_rate_list"] = get_list("migration_rate", [0.1])
    out["seed_list"] = get_list("seed", [1])

    out["pnsga3_only"] = bool(exp.get("pnsga3_only", False))
    out["output_dir"] = exp.get("output_dir", "nsga_logs")
    out["metrics_every_gen"] = bool(exp.get("metrics_every_gen", True))
    out["hv_enabled"] = bool(exp.get("hv_enabled", True))
    out["pymoo_timing"] = bool(exp.get("pymoo_timing", False))

    return out


def _nsga3_cache_path(output_dir, problem_name, n_var, n_obj, pop_size, n_gen, seed, n_partitions):
    """NSGA-III result cache path: determined by (problem, n_var, n_obj, pop_size, n_gen, seed, n_partitions)."""
    d = Path(output_dir) / problem_name / "NSGA3"
    d.mkdir(parents=True, exist_ok=True)
    return d / (
        f"{problem_name}_NSGA3"
        f"_var{n_var}_obj{n_obj}_pop{pop_size}_gen{n_gen}_seed{seed}_np{n_partitions}_final.npz"
    )


def _pnsga3_summary_path(output_dir, problem_name,
                         n_var, n_obj, pop_size, n_gen,
                         n_islands, migration_interval, migration_rate, seed, n_partitions):
    """Path to the combined ParallelNSGA3 + NSGA3 experiment summary (one per parameter combo), used for grid resume."""
    d = Path(output_dir) / problem_name / "SUMMARY"
    d.mkdir(parents=True, exist_ok=True)
    return d / (
        f"{problem_name}_SUMMARY"
        f"_var{n_var}_obj{n_obj}_pop{pop_size}_gen{n_gen}"
        f"_isl{n_islands}_mi{migration_interval}_mr{migration_rate:.2f}"
        f"_seed{seed}_np{n_partitions}.npy"
    )


def _pnsga3_result_path(output_dir, problem_name, n_var, n_obj, pop_size, n_gen,
                        n_islands, migration_interval, migration_rate, seed, n_partitions):
    """Path for a single ParallelNSGA3 run result (igd_history, hv_history), under ParallelNSGA3/ for plotting."""
    d = Path(output_dir) / problem_name / "ParallelNSGA3"
    d.mkdir(parents=True, exist_ok=True)
    return d / (
        f"{problem_name}_ParallelNSGA3"
        f"_var{n_var}_obj{n_obj}_pop{pop_size}_gen{n_gen}"
        f"_isl{n_islands}_mi{migration_interval}_mr{migration_rate:.2f}"
        f"_seed{seed}_np{n_partitions}.npy"
    )


def _load_nsga3_cache(cache_path, igd_indicator, hv_indicator):
    """Load F, igd/hv history and final metrics from npz for NSGA-III."""
    data = np.load(cache_path, allow_pickle=True)
    F = data["F"]
    igd_hist = list(data["igd_history"])
    hv_hist = list(data["hv_history"])
    igd_final = float(data["igd_final"])
    hv_final = float(data["hv_final"])
    gen_times = list(data["gen_times"]) if "gen_times" in data.files else []
    return F, igd_hist, hv_hist, igd_final, hv_final, gen_times


def _save_nsga3_cache(cache_path, F, igd_history, hv_history, igd_final, hv_final, gen_times=None):
    """Write NSGA-III final F, history and metrics to cache."""
    if gen_times is None:
        gen_times = []
    np.savez(
        cache_path,
        F=F,
        igd_history=np.array(igd_history),
        hv_history=np.array(hv_history),
        igd_final=np.array(igd_final),
        hv_final=np.array(hv_final),
        gen_times=np.array(gen_times),
    )


# -------------------------------
# 2. Callback: metrics + saving F/X
# -------------------------------
class MetricsCallback(Callback):
    """Record IGD/HV per generation and save F/X to disk at intervals."""

    def __init__(
        self,
        label,
        igd_indicator,
        hv_indicator,
        log_interval,
        save_interval,
        output_root,
        algo_name,
        problem_name,
        n_var,
        n_obj,
        pop_size,
        n_gen_total,
        n_islands,
        migration_interval,
        migration_rate,
        seed,
        n_partitions=None,
        metrics_every_gen=True,
    ):
        super().__init__()
        self.label = label
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.igd_indicator = igd_indicator
        self.hv_indicator = hv_indicator
        self.metrics_every_gen = metrics_every_gen
        self.igd_history = []
        self.hv_history = []

        self.algo_name = algo_name
        self.problem_name = problem_name
        self.n_var = n_var
        self.n_obj = n_obj
        self.pop_size = pop_size
        self.n_gen_total = n_gen_total
        self.n_islands = n_islands
        self.migration_interval = migration_interval
        self.migration_rate = migration_rate
        self.seed = seed
        self.n_partitions = n_partitions

        self.output_dir = Path(output_root) / problem_name / algo_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.gen_times = []   # Per-generation wall time (seconds)
        self._last_time = time.time()

    def notify(self, algorithm):
        now = time.time()
        self.gen_times.append(now - self._last_time)
        self._last_time = now

        F = algorithm.pop.get("F")

        is_last_gen = algorithm.n_iter >= self.n_gen_total
        do_metrics = self.metrics_every_gen or is_last_gen

        if do_metrics:
            if self.igd_indicator is not None:
                igd = self.igd_indicator(F)
            else:
                igd = np.nan

            if self.hv_indicator is not None:
                hv = self.hv_indicator(F)
            else:
                hv = np.nan
        else:
            igd = np.nan
            hv = np.nan

        self.igd_history.append(igd)
        self.hv_history.append(hv)

        if algorithm.n_iter % self.log_interval == 0:
            if do_metrics:
                print(f"    gen {algorithm.n_iter}/{self.n_gen_total} done (IGD={igd:.4f})", flush=True)
            else:
                print(f"    gen {algorithm.n_iter}/{self.n_gen_total} done (IGD/HV=last gen only)", flush=True)
            print(f"{self.label} - completed iteration {algorithm.n_iter}", end="\r")

        if algorithm.n_iter % self.save_interval == 0:
            filename = (
                f"{self.problem_name}_{self.algo_name}"
                f"_var{self.n_var}"
                f"_obj{self.n_obj}"
                f"_pop{self.pop_size}"
                f"_gen{self.n_gen_total}"
                f"_seed{self.seed}"
                f"_isl{self.n_islands}"
                f"_mi{self.migration_interval}"
                f"_mr{self.migration_rate:.2f}"
            )
            if self.n_partitions is not None:
                filename += f"_np{self.n_partitions}"
            filename += f"_iter{algorithm.n_iter:04d}.npz"
            path = self.output_dir / filename
            X = algorithm.pop.get("X")
            np.savez(
                path,
                F=F,
                X=X,
                iter=algorithm.n_iter,
                n_var=self.n_var,
                n_obj=self.n_obj,
                pop_size=self.pop_size,
                n_gen_total=self.n_gen_total,
                n_islands=self.n_islands,
                migration_interval=self.migration_interval,
                migration_rate=self.migration_rate,
                seed=self.seed,
                n_partitions=getattr(self, "n_partitions", None),
            )


class MultiObjectiveOutputLastGenOnly(MultiObjectiveOutput):
    """Like MultiObjectiveOutput but when metrics_every_gen=False only computes IGD/GD/HV on last generation (saves time).
    If hv_enabled=False, HV is never computed (only IGD/GD)."""

    def __init__(self, metrics_every_gen=True, hv_enabled=True):
        super().__init__()
        self._metrics_every_gen = metrics_every_gen
        self._hv_enabled = hv_enabled

    def update(self, algorithm):
        from pymoo.util.display.output import Output
        Output.update(self, algorithm)
        for col in [self.igd, self.gd, self.hv, self.eps, self.indicator]:
            col.set(None)

        n_max = getattr(algorithm.termination, "n_max_gen", None)
        is_last_gen = n_max is not None and algorithm.n_iter >= n_max
        if not self._metrics_every_gen and not is_last_gen:
            return

        F, feas = algorithm.opt.get("F", "feas")
        F = F[feas]
        if len(F) == 0:
            return

        problem = algorithm.problem
        if hasattr(problem, "time"):
            self.pf = pareto_front_if_possible(problem)
        if self.pf is not None and feas.sum() > 0:
            self.igd.set(IGD(self.pf, zero_to_one=True).do(F))
            self.gd.set(GDIndicator(self.pf, zero_to_one=True).do(F))
            if self._hv_enabled and self.hv in self.columns:
                from pymoo.indicators.hv import Hypervolume
                self.hv.set(Hypervolume(pf=self.pf, zero_to_one=True).do(F))
        if self.indicator_no_pf is not None:
            ind = self.indicator_no_pf
            ind.update(algorithm)
            valid = ind.delta_ideal is not None
            if valid:
                if ind.delta_ideal > ind.tol:
                    max_from, eps = "ideal", ind.delta_ideal
                elif ind.delta_nadir > ind.tol:
                    max_from, eps = "nadir", ind.delta_nadir
                else:
                    max_from, eps = "f", ind.delta_f
                self.eps.set(eps)
                self.indicator.set(max_from)


# -------------------------------
# 3. Experiment runner
# -------------------------------
def run_parallel_nsga3_experiment(
    problem_name="c3dtlz4",
    n_var=12,
    n_obj=8,
    pop_size=1000,
    n_gen=200,
    n_islands=6,
    migration_interval=3,
    migration_rate=0.05,
    difficulty_index=9,
    seed=1,
    n_partitions=12,
    output_dir="nsga_logs",
    save_interval=10,
    pnsga3_only=False,
    metrics_every_gen=True,
    hv_enabled=True,
):
    """Compare NSGA-III and ParallelNSGA3 under given parameters and plot IGD/HV curves + front scatter. If pnsga3_only=True, run only PNSGA3 and skip NSGA3. metrics_every_gen: if True compute IGD/HV every generation; if False only on last generation (faster, SUMMARY still has final IGD/HV)."""

    print("  [setup] building problem...", flush=True)
    problem = build_problem(problem_name, n_var, n_obj, difficulty_index)

    print("  [setup] getting reference directions...", flush=True)
    ref_dirs = get_reference_directions(
        "das-dennis",
        problem.n_obj,
        n_partitions=n_partitions,
    )

    print("  [setup] setting up IGD/HV (pareto_front)...", flush=True)
    try:
        pf = problem.pareto_front(ref_dirs)
    except Exception as e:
        print(
            f"Warning: could not obtain exact PF for {problem_name} ({e}). "
            f"IGD/HV will be NaN."
        )
        pf = None

    if pf is not None:
        # Normalize IGD for consistency with pymoo display
        igd_indicator = IGD(pf, zero_to_one=True)
        nadir = np.asarray(np.max(pf, axis=0)).flatten()
        ideal = np.asarray(np.min(pf, axis=0)).flatten()
        if hv_enabled:
            ref_point = nadir + 0.5 * np.maximum(nadir - ideal, 1e-6)
            hv_indicator = HV(pf=pf, ref_point=ref_point)
        else:
            hv_indicator = None
    else:
        igd_indicator = None
        hv_indicator = None

    # Complexity: separate for NSGA3 and Parallel NSGA3 (single pop vs per-island pop)
    M = problem.n_obj
    # Selection/survival (objective space): O(M N^2), no n_var
    complexity_formula_nsga3 = "O(M N^2), N=pop_size (single population)"
    complexity_M_N2_nsga3 = M * (pop_size ** 2)
    # Parallel NSGA3: per-island pop ~ pop_size/n_islands; total complexity = per-island O(M N^2) * n_islands
    pop_per_island = max(1, pop_size // n_islands)
    complexity_formula_pnsga3 = "O(M N^2) per island × n_islands, N=pop_size/n_islands"
    complexity_M_N2_pnsga3 = n_islands * M * (pop_per_island ** 2)
    # Variation (crossover/mutation in decision space): O(N * n_var) per gen; same total for NSGA3 and PNSGA3
    complexity_variation = pop_size * n_var
    print(f"  [Complexity NSGA3] selection: {complexity_formula_nsga3} => M*N^2 = {M}*{pop_size}^2 = {complexity_M_N2_nsga3}", flush=True)
    print(f"  [Complexity PNSGA3] selection: {complexity_formula_pnsga3} => {n_islands}*{M}*{pop_per_island}^2 = {complexity_M_N2_pnsga3}", flush=True)
    print(f"  [Complexity both] variation: O(N*n_var) => N*n_var = {pop_size}*{n_var} = {complexity_variation} (per gen)", flush=True)
    print("  (Note: selection is objective-space; wall time often dominated by fitness evals and ref_dirs when ref_dirs >> pop_size)", flush=True)

    if not metrics_every_gen:
        print("  [metrics_every_gen=False] IGD/HV and table igd/gd only computed on last generation (faster).", flush=True)

    #print("  [4/5] Running NSGA-III...", flush=True)
    def run_algorithm(algorithm, label, algo_name, n_partitions_val=None):
        cb = MetricsCallback(
            label=label,
            igd_indicator=igd_indicator,
            hv_indicator=hv_indicator,
            log_interval=1,
            save_interval=save_interval,
            output_root=output_dir,
            algo_name=algo_name,
            problem_name=problem_name,
            n_var=n_var,
            n_obj=problem.n_obj,
            pop_size=pop_size,
            n_gen_total=n_gen,
            n_islands=n_islands if hasattr(algorithm, "n_islands") else 1,
            migration_interval=getattr(algorithm, "migration_interval", 0),
            migration_rate=getattr(algorithm, "migration_rate", 0.0),
            seed=seed,
            n_partitions=n_partitions_val,
            metrics_every_gen=metrics_every_gen,
        )

        print(f"    (n_obj={problem.n_obj}, pop={pop_size}: first gen may take minutes...)", flush=True)
        # verbose=True shows pymoo progress (e.g. Evaluating / Gen 0) to see if stuck in eval or selection
        # Optional: set env PYMOO_TIMING=1 for per-step timing (infill / eval / advance; survival: nds / norm / associate / niching)
        res = minimize(
            problem,
            algorithm,
            termination=("n_gen", n_gen),
            seed=seed,
            verbose=True,
            callback=cb,
        )

        F = res.pop.get("F")

        print("  Computing final IGD/HV for summary...", flush=True)
        if igd_indicator is not None:
            igd_final = igd_indicator(F)
        else:
            igd_final = np.nan

        if hv_indicator is not None:
            hv_final = hv_indicator(F)
        else:
            hv_final = np.nan
        print("  Final IGD/HV done.", flush=True)

        print(f"{label} - finished. Final IGD = {igd_final:.6f}, HV = {hv_final:.6f}")

        return F, cb.igd_history, cb.hv_history, igd_final, hv_final, cb.gen_times

    n_obj_actual = problem.n_obj
    if pnsga3_only:
        # Run only PNSGA3: skip NSGA3; use empty/NaN placeholders for plotting and summary
        M_obj = problem.n_obj
        F_nsga3 = np.empty((0, M_obj))
        igd_hist_nsga3 = []
        hv_hist_nsga3 = []
        igd_nsga3 = np.nan
        hv_nsga3 = np.nan
        gen_times_nsga3 = []
        print("  [Skipping NSGA-III] pnsga3_only=True", flush=True)
    else:
        nsga3_cache_path = _nsga3_cache_path(
            output_dir, problem_name, n_var, n_obj_actual, pop_size, n_gen, seed, n_partitions
        )
        if nsga3_cache_path.exists():
            F_nsga3, igd_hist_nsga3, hv_hist_nsga3, igd_nsga3, hv_nsga3, gen_times_nsga3 = _load_nsga3_cache(
                nsga3_cache_path, igd_indicator, hv_indicator
            )
            print("NSGA-III - using cached result (skipped run)")
        else:
            alg_nsga3 = NSGA3(
                ref_dirs=ref_dirs,
                pop_size=pop_size,
                output=MultiObjectiveOutputLastGenOnly(metrics_every_gen=metrics_every_gen, hv_enabled=hv_enabled),
            )
            F_nsga3, igd_hist_nsga3, hv_hist_nsga3, igd_nsga3, hv_nsga3, gen_times_nsga3 = run_algorithm(
                alg_nsga3,
                label="NSGA-III",
                algo_name="NSGA3",
                n_partitions_val=n_partitions,
            )
            _save_nsga3_cache(
                nsga3_cache_path,
                F_nsga3,
                igd_hist_nsga3,
                hv_hist_nsga3,
                igd_nsga3,
                hv_nsga3,
                gen_times_nsga3,
            )

    # print("  [5/5] Running ParallelNSGA3...", flush=True)
    alg_pnsga3 = ParallelNSGA3(
        ref_dirs=ref_dirs,
        pop_size=pop_size,
        n_islands=n_islands,
        migration_interval=migration_interval,
        migration_rate=migration_rate,
        output=MultiObjectiveOutputLastGenOnly(metrics_every_gen=metrics_every_gen, hv_enabled=hv_enabled),
    )
    F_pnsga3, igd_hist_pnsga3, hv_hist_pnsga3, igd_pnsga3, hv_pnsga3, gen_times_pnsga3 = run_algorithm(
        alg_pnsga3,
        label="ParallelNSGA3",
        algo_name="ParallelNSGA3",
        n_partitions_val=n_partitions,
    )

    # Write ParallelNSGA3 igd_history/hv_history to .npy under ParallelNSGA3/ for nsga_plot_utils line plots
    pnsga3_result_path = _pnsga3_result_path(
        output_dir, problem_name, n_var, problem.n_obj,
        pop_size, n_gen, n_islands, migration_interval, migration_rate, seed, n_partitions
    )
    pnsga3_plot_data = {
        "igd_history": np.array(igd_hist_pnsga3),
        "hv_history": np.array(hv_hist_pnsga3),
        "gen_times": np.array(gen_times_pnsga3),
        "problem_name": problem_name,
        "n_var": n_var,
        "n_obj": problem.n_obj,
        "pop_size": pop_size,
        "n_gen": n_gen,
        "n_islands": n_islands,
        "migration_interval": migration_interval,
        "migration_rate": migration_rate,
        "seed": seed,
        "n_partitions": n_partitions,
        "igd_final": igd_pnsga3,
        "hv_final": hv_pnsga3,
    }
    np.save(pnsga3_result_path, pnsga3_plot_data)

    # 5. Plots: IGD/HV curves + scatters
    gens_nsga3 = np.arange(1, len(igd_hist_nsga3) + 1)
    gens_pnsga3 = np.arange(1, len(igd_hist_pnsga3) + 1)

    fig = plt.figure(figsize=(12, 10))

    ax_igd = fig.add_subplot(2, 2, 1)
    ax_igd.plot(gens_nsga3, igd_hist_nsga3, label=f"NSGA-III (final IGD={igd_nsga3:.3g})")
    ax_igd.plot(gens_pnsga3, igd_hist_pnsga3, label=f"ParallelNSGA3 (final IGD={igd_pnsga3:.3g})")
    ax_igd.set_xlabel("Generation")
    ax_igd.set_ylabel("IGD (lower is better)")
    ax_igd.set_title(f"IGD vs. Generations ({problem_name})")
    ax_igd.grid(True, alpha=0.3)
    ax_igd.legend()

    ax_hv = fig.add_subplot(2, 2, 2)
    ax_hv.plot(gens_nsga3, hv_hist_nsga3, label=f"NSGA-III (final HV={hv_nsga3:.3g})")
    ax_hv.plot(gens_pnsga3, hv_hist_pnsga3, label=f"ParallelNSGA3 (final HV={hv_pnsga3:.3g})")
    ax_hv.set_xlabel("Generation")
    ax_hv.set_ylabel("Hypervolume (higher is better)")
    ax_hv.set_title(f"HV vs. Generations ({problem_name})")
    ax_hv.grid(True, alpha=0.3)
    ax_hv.legend()

    if problem.n_obj >= 3:
        ax_scatter_nsga3 = fig.add_subplot(2, 2, 3, projection='3d')
        ax_scatter_nsga3.scatter(F_nsga3[:, 0], F_nsga3[:, 1], F_nsga3[:, 2],
                                 s=8, c='tab:blue')
        ax_scatter_nsga3.set_title(
            f"NSGA-III Front\nIGD={igd_nsga3:.4f}, HV={hv_nsga3:.4f}"
        )
        ax_scatter_nsga3.set_xlabel("f1")
        ax_scatter_nsga3.set_ylabel("f2")
        ax_scatter_nsga3.set_zlabel("f3")

        ax_scatter_pnsga3 = fig.add_subplot(2, 2, 4, projection='3d')
        ax_scatter_pnsga3.scatter(F_pnsga3[:, 0], F_pnsga3[:, 1], F_pnsga3[:, 2],
                                  s=8, c='tab:orange')
        ax_scatter_pnsga3.set_title(
            f"ParallelNSGA3 Front\nIGD={igd_pnsga3:.4f}, HV={hv_pnsga3:.4f}"
        )
        ax_scatter_pnsga3.set_xlabel("f1")
        ax_scatter_pnsga3.set_ylabel("f2")
        ax_scatter_pnsga3.set_zlabel("f3")
    else:
        ax_scatter_nsga3 = fig.add_subplot(2, 2, 3)
        ax_scatter_nsga3.scatter(F_nsga3[:, 0], F_nsga3[:, 1], s=8, c='tab:blue')
        ax_scatter_nsga3.set_title(
            f"NSGA-III Front\nIGD={igd_nsga3:.4f}, HV={hv_nsga3:.4f}"
        )
        ax_scatter_nsga3.set_xlabel("f1")
        ax_scatter_nsga3.set_ylabel("f2")

        ax_scatter_pnsga3 = fig.add_subplot(2, 2, 4)
        ax_scatter_pnsga3.scatter(F_pnsga3[:, 0], F_pnsga3[:, 1], s=8, c='tab:orange')
        ax_scatter_pnsga3.set_title(
            f"ParallelNSGA3 Front\nIGD={igd_pnsga3:.4f}, HV={hv_pnsga3:.4f}"
        )
        ax_scatter_pnsga3.set_xlabel("f1")
        ax_scatter_pnsga3.set_ylabel("f2")

    plt.tight_layout()
    plt.show()

    summary = {
        "problem_name": problem_name,
        "n_var": n_var,
        "n_obj": problem.n_obj,
        "pop_size": pop_size,
        "n_gen": n_gen,
        "n_islands": n_islands,
        "migration_interval": migration_interval,
        "migration_rate": migration_rate,
        "seed": seed,
        "n_partitions": n_partitions,
        "complexity_formula_nsga3": complexity_formula_nsga3,
        "complexity_M_N2_nsga3": complexity_M_N2_nsga3,
        "complexity_formula_pnsga3": complexity_formula_pnsga3,
        "complexity_M_N2_pnsga3": complexity_M_N2_pnsga3,
        "complexity_variation": complexity_variation,
        "gen_time_avg_nsga3": float(np.mean(gen_times_nsga3)) if gen_times_nsga3 else np.nan,
        "gen_time_avg_pnsga3": float(np.mean(gen_times_pnsga3)) if gen_times_pnsga3 else np.nan,
        "igd_nsga3": igd_nsga3,
        "hv_nsga3": hv_nsga3,
        "igd_pnsga3": igd_pnsga3,
        "hv_pnsga3": hv_pnsga3,
    }

    summary_path = _pnsga3_summary_path(
        output_dir, problem_name, n_var, problem.n_obj,
        pop_size, n_gen, n_islands, migration_interval, migration_rate, seed, n_partitions
    )
    np.save(summary_path, summary)

    return summary


# -------------------------------
# 4. Grid of experiments (parameters from CLI or caller)
# -------------------------------
def run_grid(
    problem_list,
    n_var_list,
    n_obj_list,
    pop_size_list,
    n_gen_list,
    n_partitions_list,
    n_islands_list,
    migration_interval_list,
    migration_rate_list,
    seed_list,
    pnsga3_only=False,
    output_dir="nsga_logs",
    server_index=None,
    num_servers=None,
    worker_index=None,
    num_workers=None,
    metrics_every_gen=True,
    hv_enabled=True,
):
    """Run the full parameter grid; all list args are iterables (e.g. lists from CLI).
    If server_index, num_servers, worker_index, num_workers are set, only run tasks assigned to this (server, worker).
    """
    results = []
    _total_lens = (
        len(problem_list),
        len(n_var_list),
        len(n_obj_list),
        len(pop_size_list),
        len(n_gen_list),
        len(n_partitions_list),
        len(n_islands_list),
        len(migration_interval_list),
        len(migration_rate_list),
        len(seed_list),
    )
    total_tasks = 1
    for L in _total_lens:
        total_tasks *= L
    tasks_per_server = ((total_tasks + num_servers - 1) // num_servers) if num_servers else total_tasks
    run_filter = (
        server_index is not None
        and num_servers is not None
        and worker_index is not None
        and num_workers is not None
    )
    if run_filter:
        print(f">>> Parameter grid: {total_tasks} tasks | server {server_index}/{num_servers} worker {worker_index}/{num_workers} (filtered)\n", flush=True)
    else:
        print(f">>> Parameter grid: {total_tasks} tasks\n", flush=True)

    for task_index, (
        problem_name,
        n_var,
        n_obj,
        pop_size,
        n_gen,
        n_partitions,
        n_islands,
        migration_interval,
        migration_rate,
        seed,
    ) in enumerate(
        itertools.product(
            problem_list,
            n_var_list,
            n_obj_list,
            pop_size_list,
            n_gen_list,
            n_partitions_list,
            n_islands_list,
            migration_interval_list,
            migration_rate_list,
            seed_list,
        ),
        start=1,
    ):
        t = task_index - 1  # 0-based
        if run_filter:
            if (t // tasks_per_server) != server_index or (t % num_workers) != worker_index:
                continue
        print(f"\n>>> Starting task {task_index}/{total_tasks}...", flush=True)
        print(
            f"=== Task {task_index}/{total_tasks} | {problem_name}, n_var={n_var}, n_obj={n_obj}, "
            f"pop={pop_size}, n_partitions={n_partitions}, islands={n_islands}, "
            f"mig_int={migration_interval}, mig_rate={migration_rate}, seed={seed} ===",
            flush=True,
        )
        summary_path = _pnsga3_summary_path(
            output_dir,
            problem_name,
            n_var,
            n_obj,
            pop_size,
            n_gen,
            n_islands,
            migration_interval,
            migration_rate,
            seed,
            n_partitions,
        )
        if summary_path.exists():
            summary = np.load(summary_path, allow_pickle=True).item()
            print(">>> Skipping run (summary cache found).")
        else:
            summary = run_parallel_nsga3_experiment(
                problem_name=problem_name,
                n_var=n_var,
                n_obj=n_obj,
                pop_size=pop_size,
                n_gen=n_gen,
                n_partitions=n_partitions,
                n_islands=n_islands,
                migration_interval=migration_interval,
                migration_rate=migration_rate,
                seed=seed,
                output_dir=output_dir,
                pnsga3_only=pnsga3_only,
                metrics_every_gen=metrics_every_gen,
                hv_enabled=hv_enabled,
            )
        results.append(summary)
    return pd.DataFrame(results)


def _parse_int_list(s: str):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_float_list(s: str):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_params_string(s: str):
    """
    Parse a single string into grid parameters. Format: key=value pairs separated by spaces.
    List values are comma-separated (e.g. n_obj=6,7,8); spaces after commas are allowed (e.g. n_var=10, 12).
    If a token has no "=", it is appended to the previous value (so seed=1 , 2 => seed=[1,2]).
    Keys: problem, n_var, n_obj, pop_size, n_gen, n_partitions, n_islands, migration_interval,
    migration_rate, seed, pnsga3_only, output_dir, pymoo_timing, metrics_every_gen, hv_enabled.
    """
    out = {}
    parts = s.split()
    i = 0
    while i < len(parts):
        part = parts[i]
        if "=" not in part:
            i += 1
            continue
        k, v = part.split("=", 1)
        k = k.strip().lower()
        v = v.strip()
        i += 1
        # Consume following tokens that don't look like key=value (append to v)
        while i < len(parts) and "=" not in parts[i]:
            v += " " + parts[i]
            i += 1
        v = v.strip()
        if k == "problem":
            out["problem_list"] = [x.strip() for x in v.split(",") if x.strip()]
        elif k == "n_var":
            out["n_var_list"] = _parse_int_list(v)
        elif k == "n_obj":
            out["n_obj_list"] = _parse_int_list(v)
        elif k == "pop_size":
            out["pop_size_list"] = _parse_int_list(v)
        elif k == "n_gen":
            out["n_gen_list"] = _parse_int_list(v)
        elif k == "n_partitions":
            out["n_partitions_list"] = _parse_int_list(v)
        elif k == "n_islands":
            out["n_islands_list"] = _parse_int_list(v)
        elif k == "migration_interval":
            out["migration_interval_list"] = _parse_int_list(v)
        elif k == "migration_rate":
            out["migration_rate_list"] = _parse_float_list(v)
        elif k == "seed":
            out["seed_list"] = _parse_int_list(v)
        elif k == "pnsga3_only":
            out["pnsga3_only"] = v in ("1", "true", "True", "yes")
        elif k == "pymoo_timing":
            out["pymoo_timing"] = v in ("1", "true", "True", "yes")
        elif k == "metrics_every_gen":
            out["metrics_every_gen"] = v in ("1", "true", "True", "yes")
        elif k == "hv_enabled":
            out["hv_enabled"] = v in ("1", "true", "True", "yes")
        elif k == "output_dir":
            out["output_dir"] = v
    return out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(
        description="NSGA3 / ParallelNSGA3 grid experiment. Use either individual args or a single --params string."
    )
    p.add_argument("--config", default=None, help="Path to YAML config file with experiments[].")
    p.add_argument("--exp_name", default=None, help="Experiment name in config (experiments[].name).")
    p.add_argument("--params", default=None, help=
        "All params in one string: key=value pairs space-separated, lists comma-separated. "
        "Example: problem=c1dtlz1 n_var=12 n_obj=6,7,8 pop_size=100,105 n_gen=50 n_partitions=6 "
        "n_islands=6,7,8 migration_interval=3 migration_rate=0.1 seed=1 pnsga3_only=0 output_dir=nsga_logs"
    )
    p.add_argument("--problem", default="c1dtlz1", help="Problem name(s), comma-separated (e.g. c1dtlz1,c3dtlz4).")
    p.add_argument("--n_var", default="12", help="n_var list, comma-separated (e.g. 12).")
    p.add_argument("--n_obj", default="6,7,8,9,10,11,12", help="n_obj list (e.g. 6,7,8,9,10,11,12).")
    p.add_argument("--pop_size", default="100,105,110,115,120,125,130,135,140", help="pop_size list, comma-separated.")
    p.add_argument("--n_gen", default="50", help="n_gen list (e.g. 50).")
    p.add_argument("--n_partitions", default="6", help="n_partitions list (reference direction partition).")
    p.add_argument("--n_islands", default="6,7,8,9,10", help="n_islands list, comma-separated.")
    p.add_argument("--migration_interval", default="3", help="migration_interval list.")
    p.add_argument("--migration_rate", default="0.1", help="migration_rate list, comma-separated.")
    p.add_argument("--seed", default="1", help="seed list, comma-separated.")
    p.add_argument("--pnsga3_only", action="store_true", help="Skip NSGA3 each run; run only ParallelNSGA3.")
    p.add_argument("--output_dir", default="nsga_logs", help="Output directory for logs and caches.")
    p.add_argument("--server_index", type=int, default=None, help="Server index 0..num_servers-1 for distributed run.")
    p.add_argument("--num_servers", type=int, default=None, help="Total number of servers (with --server_index, --worker_index, --num_workers).")
    p.add_argument("--worker_index", type=int, default=None, help="Worker index 0..num_workers-1 on this server.")
    p.add_argument("--num_workers", type=int, default=None, help="Number of parallel workers per server.")
    p.add_argument("--pymoo_timing", action="store_true", help="Print [timing] and [survival] each gen (env PYMOO_TIMING=1).")
    p.add_argument("--no_metrics_every_gen", action="store_true", help="Only compute IGD/HV on last generation (faster); SUMMARY still has final IGD/HV.")
    p.add_argument("--no_hv", action="store_true", help="Disable HV computation entirely (faster); IGD only.")
    args = p.parse_args()

    # Highest priority: YAML config if provided
    if args.config is not None:
        cfg = _load_experiment_from_config(args.config, args.exp_name)
        problem_list = cfg["problem_list"]
        n_var_list = cfg["n_var_list"]
        n_obj_list = cfg["n_obj_list"]
        pop_size_list = cfg["pop_size_list"]
        n_gen_list = cfg["n_gen_list"]
        n_partitions_list = cfg["n_partitions_list"]
        n_islands_list = cfg["n_islands_list"]
        migration_interval_list = cfg["migration_interval_list"]
        migration_rate_list = cfg["migration_rate_list"]
        seed_list = cfg["seed_list"]
        pnsga3_only = cfg["pnsga3_only"]
        output_dir = cfg["output_dir"]
        metrics_every_gen = cfg["metrics_every_gen"]
        hv_enabled = cfg["hv_enabled"]
        pymoo_timing = cfg["pymoo_timing"]

    elif args.params:
        # All parameters from single --params string (ignore other CLI args for grid)
        parsed = _parse_params_string(args.params)
        problem_list = parsed.get("problem_list", ["c1dtlz1"])
        n_var_list = parsed.get("n_var_list", [12])
        n_obj_list = parsed.get("n_obj_list", [6, 7, 8, 9, 10, 11, 12])
        pop_size_list = parsed.get("pop_size_list", [100])
        n_gen_list = parsed.get("n_gen_list", [50])
        n_partitions_list = parsed.get("n_partitions_list", [6])
        n_islands_list = parsed.get("n_islands_list", [6])
        migration_interval_list = parsed.get("migration_interval_list", [3])
        migration_rate_list = parsed.get("migration_rate_list", [0.1])
        seed_list = parsed.get("seed_list", [1])
        pnsga3_only = parsed.get("pnsga3_only", False)
        output_dir = parsed.get("output_dir", "nsga_logs")
        pymoo_timing = parsed.get("pymoo_timing", False)
        metrics_every_gen = parsed.get("metrics_every_gen", True)
        hv_enabled = parsed.get("hv_enabled", True)
    else:
        problem_list = [x.strip() for x in args.problem.split(",") if x.strip()]
        n_var_list = _parse_int_list(args.n_var)
        n_obj_list = _parse_int_list(args.n_obj)
        pop_size_list = _parse_int_list(args.pop_size)
        n_gen_list = _parse_int_list(args.n_gen)
        n_partitions_list = _parse_int_list(args.n_partitions)
        n_islands_list = _parse_int_list(args.n_islands)
        migration_interval_list = _parse_int_list(args.migration_interval)
        migration_rate_list = _parse_float_list(args.migration_rate)
        seed_list = _parse_int_list(args.seed)
        pnsga3_only = args.pnsga3_only
        output_dir = args.output_dir
        pymoo_timing = getattr(args, "pymoo_timing", False)
        metrics_every_gen = not getattr(args, "no_metrics_every_gen", False)
        hv_enabled = not getattr(args, "no_hv", False)
        if getattr(args, "no_metrics_every_gen", False):
            metrics_every_gen = False
        if getattr(args, "no_hv", False):
            hv_enabled = False

    if pymoo_timing:
        os.environ["PYMOO_TIMING"] = "1"
        print("PYMOO_TIMING=1 enabled: will print [timing] and [survival] each generation.", flush=True)

    df = run_grid(
        problem_list=problem_list,
        n_var_list=n_var_list,
        n_obj_list=n_obj_list,
        pop_size_list=pop_size_list,
        n_gen_list=n_gen_list,
        n_partitions_list=n_partitions_list,
        n_islands_list=n_islands_list,
        migration_interval_list=migration_interval_list,
        migration_rate_list=migration_rate_list,
        seed_list=seed_list,
        pnsga3_only=pnsga3_only,
        output_dir=output_dir,
        server_index=args.server_index,
        num_servers=args.num_servers,
        worker_index=args.worker_index,
        num_workers=args.num_workers,
        metrics_every_gen=metrics_every_gen,
        hv_enabled=hv_enabled,
    )
    print(df)