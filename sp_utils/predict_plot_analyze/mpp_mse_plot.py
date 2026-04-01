#!/usr/bin/env python3
"""
mpp_mse_plot.py

5x3 grid (cumulative column removed from rendering for now; code kept commented):

  Col 1: Mean(GT vs Pred) vs timestep  (GT in red, Pred in blue)
  Col 2: Mean(Pred - GT) vs timestep   (difference; same units/scale as Col 1)
  Col 3: Step MSE vs timestep          (its own y-scale)

Rows (order via FIELD_ORDER):
  1) Uvelocity
  2) Wvelocity
  3) density_maincharge
  4) av_density
  5) Mean of All Fields (computed from the 4 fields)

Notes:
- Requires mse_summary.csv to include: dataset, expt_id, t_idx, ns, var, field, step_mse, pred_path
- pred_path must point to pred.npz containing: pred_sel, gt_sel, fields
- Does NOT create output directories.

Optional (kept but not rendered):
- Running cumulative mean MSE column can be re-enabled later (see commented code).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


FIELD_ORDER: List[str] = [
    "Uvelocity",
    "Wvelocity",
    "density_maincharge",
    "av_density",
]

DATASET_DISPLAY: Dict[str, str] = {
    "clx": "Cylex",
}


def _to_int_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def _read_csv(csv_in: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_in)

    required = {"dataset", "expt_id", "t_idx", "ns", "var", "field", "step_mse", "pred_path"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    df["dataset"] = df["dataset"].astype(str).str.strip()
    df["field"] = df["field"].astype(str).str.strip()
    df["pred_path"] = df["pred_path"].astype(str).str.strip()

    df["t_idx"] = _to_int_series(df["t_idx"])
    df["ns"] = _to_int_series(df["ns"])
    df["var"] = _to_int_series(df["var"])

    df["expt_id_str"] = df["expt_id"].astype(str).str.strip()
    df["expt_id_int"] = _to_int_series(df["expt_id"])

    return df


def _filter_df(df: pd.DataFrame, dataset: str, expt_id: str, ns: int, var: int) -> pd.DataFrame:
    dataset = str(dataset).strip()
    expt_id_str = str(expt_id).strip()

    try:
        expt_id_int = int(expt_id_str)
    except Exception:
        expt_id_int = None

    base = (df["dataset"] == dataset) & (df["ns"] == ns) & (df["var"] == var)

    out = df[base & (df["expt_id_str"] == expt_id_str)].copy()
    if out.empty and expt_id_int is not None:
        out = df[base & (df["expt_id_int"] == expt_id_int)].copy()

    return out


def _plot_series(ax, t: np.ndarray, y: np.ndarray, label: str) -> None:
    if t.size == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return
    ax.plot(t, y, label=label)
    ax.legend(loc="best", fontsize="small")


def _prep_step_mse(df_f: pd.DataFrame, field: str) -> Tuple[np.ndarray, np.ndarray]:
    d = df_f[df_f["field"] == field].copy()
    d = d.dropna(subset=["t_idx", "step_mse"])
    if d.empty:
        return np.array([]), np.array([])
    d = d.groupby("t_idx", as_index=False)[["step_mse"]].mean(numeric_only=True).sort_values("t_idx")
    return d["t_idx"].to_numpy(dtype=int), d["step_mse"].to_numpy(dtype=float)


def _pred_gt_mean_timeseries(
    df_f: pd.DataFrame,
    field: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For a given field, compute gt_mean(t) and pred_mean(t) across timesteps,
    using pred_path to load pred.npz at each timestep.
    """
    d = df_f[df_f["field"] == field].copy()
    d = d.dropna(subset=["t_idx", "pred_path"])
    if d.empty:
        return np.array([]), np.array([]), np.array([])

    d = d.sort_values("t_idx").drop_duplicates(subset=["t_idx"], keep="first")

    t_list: List[int] = []
    gt_list: List[float] = []
    pr_list: List[float] = []

    cache: Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]] = {}

    for _, row in d.iterrows():
        t = int(row["t_idx"])
        p = row["pred_path"]

        try:
            if p not in cache:
                with np.load(p, allow_pickle=True) as z:
                    pred_sel = z["pred_sel"]
                    gt_sel = z["gt_sel"]
                    fields = z["fields"].tolist()
                fields = [str(x) for x in fields]
                cache[p] = (pred_sel, gt_sel, fields)

            pred_sel, gt_sel, fields = cache[p]
            if field not in fields:
                continue
            c = fields.index(field)

            if pred_sel.ndim == 4:
                pred_c = pred_sel[-1, c]
                gt_c = gt_sel[-1, c]
            else:
                pred_c = pred_sel[c]
                gt_c = gt_sel[c]

            t_list.append(t)
            gt_list.append(float(np.mean(gt_c)))
            pr_list.append(float(np.mean(pred_c)))

        except FileNotFoundError:
            print(f"[WARN] {Path(__file__).name}: missing pred_path: {p} (skipping t_idx={t})", file=sys.stderr)
            continue
        except Exception as e:
            print(f"[WARN] {Path(__file__).name}: failed reading {p} at t_idx={t}: {e} (skipping)", file=sys.stderr)
            continue

    if not t_list:
        return np.array([]), np.array([]), np.array([])

    t_arr = np.array(t_list, dtype=int)
    gt_arr = np.array(gt_list, dtype=float)
    pr_arr = np.array(pr_list, dtype=float)

    order = np.argsort(t_arr)
    return t_arr[order], gt_arr[order], pr_arr[order]


def _mean_over_fields_at_t(
    series_by_field: Dict[str, Tuple[np.ndarray, np.ndarray]]
) -> Tuple[np.ndarray, np.ndarray]:
    """Mean y(t) across fields at each t; averages over fields present at that t."""
    all_ts = sorted(set(int(t) for field in series_by_field for t in series_by_field[field][0].tolist()))
    if not all_ts:
        return np.array([]), np.array([])

    maps: Dict[str, Dict[int, float]] = {}
    for f, (t_arr, y_arr) in series_by_field.items():
        maps[f] = {int(t): float(y) for t, y in zip(t_arr, y_arr)}

    y_mean: List[float] = []
    for t in all_ts:
        vals = [maps[f][t] for f in maps if t in maps[f]]
        y_mean.append(float(np.mean(vals)) if vals else np.nan)

    t_out = np.array(all_ts, dtype=int)
    y_out = np.array(y_mean, dtype=float)
    good = np.isfinite(y_out)
    return t_out[good], y_out[good]


# --- kept for later (not rendered right now) ---
# def _running_cum_mean(y: np.ndarray) -> np.ndarray:
#     if y.size == 0:
#         return np.array([])
#     return np.cumsum(y) / np.arange(1, len(y) + 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_in", required=True, help="Path to mse_summary.csv (long-form)")
    ap.add_argument("--dataset", required=True, help="Dataset key, e.g., clx")
    ap.add_argument("--expt_id", required=True, help="Experiment ID, e.g., 00387 (or 387)")
    ap.add_argument("--ns", required=True, type=int, help="Context window (ns), e.g., 1")
    ap.add_argument("--var", required=True, type=int, help="Var, e.g., 3 or 6")
    ap.add_argument("--out_png", required=True, help="Output PNG path")
    ap.add_argument("--dpi", type=int, default=600, help="Output DPI (default 600)")
    args = ap.parse_args()

    csv_in = Path(args.csv_in)
    out_png = Path(args.out_png)

    if not csv_in.exists():
        print(f"[WARN] {Path(__file__).name}: missing csv_in: {csv_in} (skipping)", file=sys.stderr)
        return 0

    try:
        df = _read_csv(csv_in)
    except Exception as e:
        print(f"[WARN] {Path(__file__).name}: failed to read/validate {csv_in}: {e} (skipping)", file=sys.stderr)
        return 0

    df_f = _filter_df(df, args.dataset, args.expt_id, args.ns, args.var)
    if df_f.empty:
        print(
            f"[WARN] {Path(__file__).name}: no matching rows for "
            f"dataset={args.dataset} expt_id={args.expt_id} ns={args.ns} var={args.var} in {csv_in} (skipping)",
            file=sys.stderr,
        )
        return 0

    if not out_png.parent.exists():
        print(
            f"[WARN] {Path(__file__).name}: output directory does not exist, not creating it: {out_png.parent} (skipping)",
            file=sys.stderr,
        )
        return 0

    dataset_disp = DATASET_DISPLAY.get(args.dataset, args.dataset)
    expt_id_5 = f"{int(args.expt_id):05d}" if str(args.expt_id).isdigit() else str(args.expt_id)
    ns_2 = f"{int(args.ns):02d}"

    fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(18, 18), sharex="col")
    fig.suptitle(
        f"Dataset: {dataset_disp} | Sim ID={expt_id_5} | Context Window={ns_2} | Var={args.var}",
        fontsize=16,
    )

    # --- Rows 1-4: per-field ---
    gt_series_by_field: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    pr_series_by_field: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for r, field in enumerate(FIELD_ORDER):
        # Col 1: GT vs Pred means
        t_g, gt_mean, pr_mean = _pred_gt_mean_timeseries(df_f, field)
        gt_series_by_field[field] = (t_g, gt_mean)
        pr_series_by_field[field] = (t_g, pr_mean)

        ax_gp = axes[r, 0]
        ax_gp.set_title(f"{field} — Mean(GT vs Pred)")
        if t_g.size == 0:
            ax_gp.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax_gp.transAxes)
        else:
            ax_gp.plot(t_g, gt_mean, label="GT (mean)")
            ax_gp.plot(t_g, pr_mean, label="Pred (mean)")
            ax_gp.legend(loc="best", fontsize="small")
        ax_gp.set_ylabel("Value")
        ax_gp.grid(True, alpha=0.3)

        # Col 2: Pred - GT mean (same scale/units as Col 1)
        ax_diff = axes[r, 1]
        ax_diff.set_title(f"{field} — Mean(Pred - GT)")
        if t_g.size == 0:
            ax_diff.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax_diff.transAxes)
        else:
            diff = pr_mean - gt_mean
            ax_diff.plot(t_g, diff, label="Pred-GT (mean)")
            ax_diff.legend(loc="best", fontsize="small")
        ax_diff.set_ylabel("Value")
        ax_diff.grid(True, alpha=0.3)

        # Col 3: Step MSE (own y-scale)
        t_m, step_mse = _prep_step_mse(df_f, field)
        ax_mse = axes[r, 2]
        ax_mse.set_title(f"{field} — Step MSE")
        _plot_series(ax_mse, t_m, step_mse, "step_mse")
        ax_mse.set_ylabel("MSE")
        ax_mse.grid(True, alpha=0.3)

        # --- kept for later (not rendered) ---
        # run_cum = _running_cum_mean(step_mse)
        # ax_cum = axes[r, 3]  # would require ncols=4
        # ...

    # --- Row 5: mean over fields ---
    # Mean GT/PRED series
    gt_mean_t, gt_mean_all = _mean_over_fields_at_t(gt_series_by_field)
    pr_mean_t, pr_mean_all = _mean_over_fields_at_t(pr_series_by_field)

    ax_gp_m = axes[4, 0]
    ax_gp_m.set_title("Mean of All Fields — Mean(GT vs Pred)")
    if gt_mean_t.size == 0:
        ax_gp_m.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax_gp_m.transAxes)
    else:
        ax_gp_m.plot(gt_mean_t, gt_mean_all, label="GT (mean)")
        ax_gp_m.plot(pr_mean_t, pr_mean_all, label="Pred (mean)")
        ax_gp_m.legend(loc="best", fontsize="small")
    ax_gp_m.set_ylabel("Value")
    ax_gp_m.grid(True, alpha=0.3)

    ax_diff_m = axes[4, 1]
    ax_diff_m.set_title("Mean of All Fields — Mean(Pred - GT)")
    if gt_mean_t.size == 0:
        ax_diff_m.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax_diff_m.transAxes)
    else:
        diff_all = pr_mean_all - gt_mean_all
        ax_diff_m.plot(gt_mean_t, diff_all, label="Pred-GT (mean)")
        ax_diff_m.legend(loc="best", fontsize="small")
    ax_diff_m.set_ylabel("Value")
    ax_diff_m.grid(True, alpha=0.3)

    # Mean step_mse at each timestep (mean across fields)
    step_series_by_field: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for f in FIELD_ORDER:
        t_s, s_s = _prep_step_mse(df_f, f)
        step_series_by_field[f] = (t_s, s_s)
    t_step_mean, step_mean_all = _mean_over_fields_at_t(step_series_by_field)

    ax_mse_m = axes[4, 2]
    ax_mse_m.set_title("Mean of All Fields — Step MSE")
    _plot_series(ax_mse_m, t_step_mean, step_mean_all, "mean_step_mse")
    ax_mse_m.set_ylabel("MSE")
    ax_mse_m.grid(True, alpha=0.3)

    # X labels on bottom row
    axes[4, 0].set_xlabel("Time step (t_idx)")
    axes[4, 1].set_xlabel("Time step (t_idx)")
    axes[4, 2].set_xlabel("Time step (t_idx)")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"[INFO] wrote {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
