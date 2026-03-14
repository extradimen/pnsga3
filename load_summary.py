# -*- coding: utf-8 -*-
"""
Scan SUMMARY/*.npy on disk and build a DataFrame with the same columns as the experiment script df.
Only reads .npy under SUMMARY/; does not read npz or other directories.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json


# Columns consistent with experiment summary (for normalization and table building)
SUMMARY_KEYS = [
    "problem_name", "n_var", "n_obj", "pop_size", "n_gen",
    "n_islands", "migration_interval", "migration_rate", "seed", "n_partitions", "focus_alpha",
    "complexity_formula_nsga3", "complexity_M_N2_nsga3",
    "complexity_formula_pnsga3", "complexity_M_N2_pnsga3",
    "gen_time_avg_nsga3", "gen_time_avg_pnsga3",
    "feasible_ratio_final_nsga3", "feasible_ratio_final_pnsga3",
    "feasible_ratio_mean_nsga3", "feasible_ratio_mean_pnsga3",
    "front1_ratio_final_nsga3", "front1_ratio_final_pnsga3",
    "front_sizes_last_nsga3", "front_sizes_last_pnsga3",
    "ideal_point_final_nsga3", "ideal_point_final_pnsga3",
    "distribution_std_final_nsga3", "distribution_std_final_pnsga3",
    "n_ref_covered_final_nsga3", "n_ref_covered_final_pnsga3",
    "igd_nsga3", "hv_nsga3", "igd_pnsga3", "hv_pnsga3",
]

ID_KEYS = [
    "problem_name", "n_var", "n_obj", "pop_size", "n_gen",
    "n_islands", "migration_interval", "migration_rate", "seed", "n_partitions", "focus_alpha",
]


def _nsga3_cache_path(root_dir, problem_name, n_var, n_obj, pop_size, n_gen, seed, n_partitions):
    root = Path(root_dir)
    d = root / problem_name / "NSGA3"
    return d / (
        f"{problem_name}_NSGA3"
        f"_var{n_var}_obj{n_obj}_pop{pop_size}_gen{n_gen}_seed{seed}_np{n_partitions}_final.npz"
    )


def _pnsga3_result_path(root_dir, problem_name, n_var, n_obj, pop_size, n_gen,
                        n_islands, migration_interval, migration_rate, seed, n_partitions, focus_alpha):
    root = Path(root_dir)
    d = root / problem_name / "ParallelNSGA3"
    return d / (
        f"{problem_name}_ParallelNSGA3"
        f"_var{n_var}_obj{n_obj}_pop{pop_size}_gen{n_gen}"
        f"_isl{n_islands}_mi{migration_interval}_mr{float(migration_rate):.2f}"
        f"_seed{seed}_np{n_partitions}_fa{float(focus_alpha):.3f}.npy"
    )


def _as_json_text(x):
    """Convert list/ndarray/object to a single-cell JSON string (for CSV single-column storage)."""
    if x is None:
        return ""
    try:
        if isinstance(x, np.ndarray):
            x = x.tolist()
        return json.dumps(x, ensure_ascii=False)
    except Exception:
        return str(x)


def _load_pnsga3_plot_data(path: Path):
    data = np.load(path, allow_pickle=True)
    if hasattr(data, "item"):
        return data.item()
    return dict(data) if hasattr(data, "keys") else {}


def _load_nsga3_histories(npz_path: Path):
    d = np.load(npz_path, allow_pickle=True)

    def _get_list(key, default=None):
        if default is None:
            default = []
        if key not in d.files:
            return default
        arr = d[key]
        # object arrays for front_sizes_history
        if getattr(arr, "dtype", None) == object:
            return list(arr.flat)
        return list(arr)

    feasible = _get_list("feasible_ratio_history", [])
    front1 = _get_list("front1_ratio_history", [])
    n_ref = _get_list("n_ref_covered_history", [])
    front_sizes = _get_list("front_sizes_history", [])

    # 2D arrays -> list of row vectors
    ideal = []
    if "ideal_point_history" in d.files and len(d["ideal_point_history"]) > 0:
        ideal = [d["ideal_point_history"][i] for i in range(len(d["ideal_point_history"]))]
    std = []
    if "distribution_std_history" in d.files and len(d["distribution_std_history"]) > 0:
        std = [d["distribution_std_history"][i] for i in range(len(d["distribution_std_history"]))]

    return dict(
        igd_history=_get_list("igd_history", []),
        igd_plus_history=_get_list("igd_plus_history", []),
        feasible_ratio_history=feasible,
        front1_ratio_history=front1,
        front_sizes_history=front_sizes,
        ideal_point_history=ideal,
        distribution_std_history=std,
        n_ref_covered_history=n_ref,
    )


def _extract_timeseries_rows(summary: dict, root_dir: Path):
    """Return expanded per-generation rows for NSGA3 and PNSGA3 for one run summary."""
    rows = []

    # identity columns (copy from summary)
    ident = {k: summary.get(k) for k in ID_KEYS}

    # --- NSGA3 (npz cache) ---
    nsga3_path = _nsga3_cache_path(
        root_dir, ident["problem_name"], ident["n_var"], ident["n_obj"],
        ident["pop_size"], ident["n_gen"], ident["seed"], ident["n_partitions"],
    )
    if nsga3_path.exists():
        h = _load_nsga3_histories(nsga3_path)
        T = int(ident["n_gen"]) if ident.get("n_gen") is not None else len(h["feasible_ratio_history"])
        T = min(T, len(h["feasible_ratio_history"]) or T)
        for t in range(T):
            rows.append({
                **ident,
                "algo": "nsga3",
                "gen": t + 1,
                "igd": h["igd_history"][t] if t < len(h["igd_history"]) else np.nan,
                "igd_plus": h["igd_plus_history"][t] if t < len(h["igd_plus_history"]) else np.nan,
                "feasible_ratio": h["feasible_ratio_history"][t] if t < len(h["feasible_ratio_history"]) else np.nan,
                "front1_ratio": h["front1_ratio_history"][t] if t < len(h["front1_ratio_history"]) else np.nan,
                "front_sizes": _as_json_text(h["front_sizes_history"][t] if t < len(h["front_sizes_history"]) else []),
                "ideal_point": _as_json_text(h["ideal_point_history"][t] if t < len(h["ideal_point_history"]) else []),
                "distribution_std": _as_json_text(h["distribution_std_history"][t] if t < len(h["distribution_std_history"]) else []),
                "n_ref_covered": h["n_ref_covered_history"][t] if t < len(h["n_ref_covered_history"]) else np.nan,
            })

    # --- PNSGA3 (plot data .npy) ---
    pnsga3_path = _pnsga3_result_path(
        root_dir, ident["problem_name"], ident["n_var"], ident["n_obj"],
        ident["pop_size"], ident["n_gen"], ident["n_islands"], ident["migration_interval"],
        ident["migration_rate"], ident["seed"], ident["n_partitions"], ident["focus_alpha"],
    )
    if pnsga3_path.exists():
        d = _load_pnsga3_plot_data(pnsga3_path)
        igd = list(np.asarray(d.get("igd_history", []), dtype=float)) if d.get("igd_history") is not None else []
        igd_plus = list(np.asarray(d.get("igd_plus_history", []), dtype=float)) if d.get("igd_plus_history") is not None else []
        feasible = list(np.asarray(d.get("feasible_ratio_history", []), dtype=float)) if d.get("feasible_ratio_history") is not None else []
        front1 = list(np.asarray(d.get("front1_ratio_history", []), dtype=float)) if d.get("front1_ratio_history") is not None else []
        n_ref = list(np.asarray(d.get("n_ref_covered_history", []), dtype=float)) if d.get("n_ref_covered_history") is not None else []
        front_sizes = list(d.get("front_sizes_history", [])) if d.get("front_sizes_history") is not None else []
        ideal = list(d.get("ideal_point_history", [])) if d.get("ideal_point_history") is not None else []
        std = list(d.get("distribution_std_history", [])) if d.get("distribution_std_history") is not None else []

        T = int(ident["n_gen"]) if ident.get("n_gen") is not None else len(feasible)
        T = min(T, len(feasible) or T)
        for t in range(T):
            rows.append({
                **ident,
                "algo": "pnsga3",
                "gen": t + 1,
                "igd": igd[t] if t < len(igd) else np.nan,
                "igd_plus": igd_plus[t] if t < len(igd_plus) else np.nan,
                "feasible_ratio": feasible[t] if t < len(feasible) else np.nan,
                "front1_ratio": front1[t] if t < len(front1) else np.nan,
                "front_sizes": _as_json_text(front_sizes[t] if t < len(front_sizes) else []),
                "ideal_point": _as_json_text(ideal[t] if t < len(ideal) else []),
                "distribution_std": _as_json_text(std[t] if t < len(std) else []),
                "n_ref_covered": n_ref[t] if t < len(n_ref) else np.nan,
            })

    return rows


def save_timeseries_csv(root_dir="exp_logs", csv_path=None, verbose=False):
    """
    Expand per-generation process metrics into a long-table CSV.

    Output rows:
      for each run in SUMMARY and each algorithm (nsga3/pnsga3) found on disk,
      emit ~n_gen rows with columns:
        - ID_KEYS (same as summary for distinguishing runs)
        - algo: "nsga3" or "pnsga3"
        - gen: 1..n_gen
        - feasible_ratio, front1_ratio, n_ref_covered
        - front_sizes, ideal_point, distribution_std (stored as single-cell JSON text)
    """
    # resolve root like load_df_from_disk
    if root_dir == "exp_logs":
        try:
            _script_dir = Path(__file__).resolve().parent
            _parallel = _script_dir / "exp_logs"
            root = _parallel if _parallel.exists() else Path(root_dir).resolve()
        except NameError:
            root = Path("exp_logs").resolve()
    else:
        root = Path(root_dir).resolve()

    if not root.exists():
        if verbose:
            print(f">>> Root directory does not exist: {root}")
        return

    if csv_path is None:
        csv_path = root.parent / "exp_summary" / "timeseries.csv"
    else:
        csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    summary_paths = [p for p in root.rglob("*.npy") if "SUMMARY" in p.parts]
    if verbose:
        print(f">>> Root: {root}")
        print(f">>> Found SUMMARY/*.npy: {len(summary_paths)}")

    for path in summary_paths:
        try:
            data = np.load(path, allow_pickle=True)
            summary = data.item() if hasattr(data, "item") else (dict(data) if hasattr(data, "keys") else None)
            if summary is None or not isinstance(summary, dict) or not _is_summary_like(summary):
                continue
            rows.extend(_extract_timeseries_rows(summary, root))
        except Exception as e:
            if verbose:
                print(f">>> Skipped {path}: {e}")
            continue

    df = pd.DataFrame(rows)
    # stable column order
    cols = ID_KEYS + ["algo", "gen", "igd", "igd_plus", "feasible_ratio", "front1_ratio", "n_ref_covered",
                      "front_sizes", "ideal_point", "distribution_std"]
    df = df[[c for c in cols if c in df.columns]]
    df.to_csv(csv_path, index=False)
    if verbose:
        print(f">>> Saved timeseries CSV with {len(df)} rows to {csv_path}")


def _normalize_summary(summary):
    """Convert summary to a dict with unified keys; compatible with old cache (gen_times_* / single complexity_*)."""
    out = {k: summary.get(k) for k in SUMMARY_KEYS}
    # Old cache: gen_times_* -> gen_time_avg_*
    if out["gen_time_avg_nsga3"] is None and "gen_times_nsga3" in summary:
        gt = summary["gen_times_nsga3"]
        out["gen_time_avg_nsga3"] = float(np.mean(gt)) if gt is not None and len(gt) > 0 else np.nan
    if out["gen_time_avg_pnsga3"] is None and "gen_times_pnsga3" in summary:
        gt = summary["gen_times_pnsga3"]
        out["gen_time_avg_pnsga3"] = float(np.mean(gt)) if gt is not None and len(gt) > 0 else np.nan
    # Old cache: single complexity_* -> complexity_*_nsga3
    if out["complexity_formula_nsga3"] is None and "complexity_formula" in summary:
        out["complexity_formula_nsga3"] = summary["complexity_formula"]
        out["complexity_M_N2_nsga3"] = summary.get("complexity_M_N2")
    return out


def _is_summary_like(data):
    """Whether data is an experiment-summary-like dict (has problem_name and n_var/n_obj or NSGA3/PNSGA3 metrics)."""
    if not isinstance(data, dict):
        return False
    if "problem_name" not in data:
        return False
    return (
        "igd_nsga3" in data or "igd_pnsga3" in data
        or "n_var" in data or "n_obj" in data
        or "pop_size" in data or "n_gen" in data
    )


def load_df_from_disk(root_dir="exp_logs", verbose=False):
    """
    Scan only SUMMARY/*.npy under root_dir and build a DataFrame with the same columns as the experiment df.

    Parameters
    ----------
    root_dir : str or Path
        Root directory; default "exp_logs".
    verbose : bool
        Whether to print scanned paths and file count; default False.

    Returns
    -------
    df : pandas.DataFrame
        Same columns as the experiment script df; extra column "source", all 'SUMMARY'.
    """
    # When root_dir is "exp_logs": if run from .py use exp_logs next to script; if run in Jupyter use cwd
    if root_dir == "exp_logs":
        try:
            _script_dir = Path(__file__).resolve().parent
            _parallel = _script_dir / "exp_logs"
            root = _parallel if _parallel.exists() else Path(root_dir).resolve()
        except NameError:
            # When executing this block directly in Jupyter there is no __file__; use cwd (usually notebook dir)
            root = Path("exp_logs").resolve()
        if not root.exists() and verbose:
            print(f">>> Root directory does not exist: {root} (if in Jupyter, ensure exp_logs exists under cwd)")
    else:
        root = Path(root_dir).resolve()
    if not root.exists():
        if verbose:
            print(f">>> Root directory does not exist: {root}")
        return pd.DataFrame(columns=SUMMARY_KEYS + ["source"])

    if verbose:
        summary_npy = [p for p in root.rglob("*.npy") if "SUMMARY" in p.parts]
        print(f">>> Root: {root}")
        print(f">>> Found SUMMARY/*.npy: {len(summary_npy)}")

    rows = []

    for path in root.rglob("*.npy"):
        if "SUMMARY" not in path.parts:
            continue
        try:
            data = np.load(path, allow_pickle=True)
            if hasattr(data, "item"):
                data = data.item()
            else:
                data = dict(data) if hasattr(data, "keys") else None
            if data is None or not isinstance(data, dict):
                continue
            if not _is_summary_like(data):
                continue
            row = _normalize_summary(data)
            row["source"] = "SUMMARY"
            rows.append(row)
        except Exception as e:
            if verbose:
                print(f">>> Skipped {path}: {e}")
            continue

    if verbose and rows:
        print(f">>> Parsed and added: {len(rows)} rows")

    if not rows:
        return pd.DataFrame(columns=SUMMARY_KEYS + ["source"])

    df = pd.DataFrame(rows)
    # Column order: standard keys + source
    cols = [c for c in SUMMARY_KEYS if c in df.columns]
    if "source" in df.columns:
        cols.append("source")
    df = df[[c for c in cols if c in df.columns]]
    return df


def save_summary_csv(root_dir="exp_logs", csv_path=None, verbose=False):
    """
    Load SUMMARY/*.npy under root_dir and save the normalized summary table as a CSV file.

    Parameters
    ----------
    root_dir : str or Path
        Root directory; default "exp_logs".
    csv_path : str or Path or None
        Destination CSV path. If None, defaults to exp_summary/summary.csv (exp_summary
        is parallel to root_dir, e.g. exp_logs and exp_summary under the same parent).
    verbose : bool
        Whether to print basic information (row count, save path).
    """
    df = load_df_from_disk(root_dir=root_dir, verbose=verbose)
    if csv_path is None:
        csv_path = Path(root_dir).resolve().parent / "exp_summary" / "summary.csv"
    else:
        csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    if verbose:
        print(f">>> Saved summary CSV with {len(df)} rows to {csv_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Export summary CSV and/or per-generation timeseries CSV from experiment logs."
    )
    parser.add_argument(
        "--root_dir",
        default="exp_logs",
        help='Root directory containing <problem>/SUMMARY/*.npy (default: "exp_logs").',
    )
    parser.add_argument(
        "--csv_path",
        default=None,
        help='Output CSV path. If omitted, saves to exp_summary/summary.csv (exp_summary parallel to root_dir).',
    )
    parser.add_argument(
        "--timeseries_csv_path",
        default=None,
        help='Output timeseries CSV path. If omitted, saves to exp_summary/timeseries.csv (parallel to root_dir).',
    )
    parser.add_argument(
        "--skip_summary",
        action="store_true",
        help="Skip writing the summary CSV.",
    )
    parser.add_argument(
        "--skip_timeseries",
        action="store_true",
        help="Skip writing the per-generation timeseries CSV.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print scanned files and summary info.",
    )
    args = parser.parse_args()

    if not args.skip_summary:
        save_summary_csv(root_dir=args.root_dir, csv_path=args.csv_path, verbose=args.verbose)
    if not args.skip_timeseries:
        save_timeseries_csv(root_dir=args.root_dir, csv_path=args.timeseries_csv_path, verbose=args.verbose)

