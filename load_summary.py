# -*- coding: utf-8 -*-
"""
Scan SUMMARY/*.npy on disk and build a DataFrame with the same columns as the experiment script df.
Only reads .npy under SUMMARY/; does not read npz or other directories.
"""

import numpy as np
import pandas as pd
from pathlib import Path


# Columns consistent with experiment summary (for normalization and table building)
SUMMARY_KEYS = [
    "problem_name", "n_var", "n_obj", "pop_size", "n_gen",
    "n_islands", "migration_interval", "migration_rate", "seed", "n_partitions",
    "complexity_formula_nsga3", "complexity_M_N2_nsga3",
    "complexity_formula_pnsga3", "complexity_M_N2_pnsga3",
    "gen_time_avg_nsga3", "gen_time_avg_pnsga3",
    "igd_nsga3", "hv_nsga3", "igd_pnsga3", "hv_pnsga3",
]


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
        description="Load SUMMARY/*.npy under a root directory and save the merged summary as a CSV file."
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
        "--verbose",
        action="store_true",
        help="Print scanned files and summary info.",
    )
    args = parser.parse_args()

    save_summary_csv(root_dir=args.root_dir, csv_path=args.csv_path, verbose=args.verbose)

