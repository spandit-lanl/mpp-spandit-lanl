#!/usr/bin/env python3
"""mpp_rmse_plot_2023_02_23.py (fixed)

Reads rmse_summary.csv produced by mpp_rmse_summary_2023_02_23.py and produces:
- Per-field step RMSE (lines vs t_idx) split by var, legend shows ns only
- Per-field cumulative RMSE (bar) split by var, x-axis shows ns only
- All-fields combined step/cum, split by var

Also overlays a GT reference as a horizontal dashed line:
- If CSV contains gt_rms_step / gt_rms_cum columns, uses those.
- For ALLFIELDS, if gt_rms_allfields_step / gt_rms_allfields_cum exist uses those.
  Otherwise it computes GT RMS across fields from gt_rms_step/gt_rms_cum by
  sqrt(mean(gt_rms^2 over fields)).

Var is NOT written anywhere on plot text/legend; it is encoded only in filenames:
  *_var-3.png, *_var-6.png, etc.

The script tolerates expt_id in CSV being '1750' while CLI may pass '01750'.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Print every saved figure path ---
import os
_orig_savefig = plt.savefig
def _savefig_with_print(fname, *args, **kwargs):
    _orig_savefig(fname, *args, **kwargs)
    try:
        p = Path(fname).expanduser().resolve()
    except Exception:
        p = fname
    print(f"SAVED_PLOT: {p}")
plt.savefig = _savefig_with_print
# --- end wrapper ---



def savefig(path: Path, dpi: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    print(f"SAVED_PLOT: {path.resolve()}")
    plt.close()

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to rmse_summary.csv")
    ap.add_argument("--dataset", default=None, help="Optional dataset filter (e.g., clx)")
    ap.add_argument("--expt_id", default=None, help="Optional expt filter (e.g., 01750)")
    ap.add_argument("--out_dir", default=None, help="Where to write plots (default: alongside CSV)")
    ap.add_argument("--dpi", type=int, default=160)
    return ap.parse_args()


def norm_expt_targets(expt_id: str):
    raw = str(expt_id).strip()
    targets = {raw}
    try:
        i = int(raw)
        targets.add(str(i))
        targets.add(str(i).zfill(5))
    except Exception:
        if raw.isdigit():
            targets.add(raw.zfill(5))
    return targets


def clean_expt_series(s: pd.Series) -> pd.Series:
    # turn 1750.0 -> 1750, keep strings
    out = s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    return out


def label_ns(ns):
    # groupby keys can be tuples in some pandas versions
    if isinstance(ns, tuple):
        ns = ns[0]
    return f"ns={int(ns):02d}"


def draw_gt_line(y, label="GT RMS"):
    plt.axhline(y, linestyle="--", linewidth=1.5, label=label)


def safe_mean(series):
    series = pd.to_numeric(series, errors="coerce")
    series = series.dropna()
    return float(series.mean()) if len(series) else None


def step_plot_field(dfv: pd.DataFrame, field: str, var: int, out_dir: Path, dpi: int):
    dff = dfv[dfv["field"] == field].copy()
    if dff.empty:
        return
    dff = dff.sort_values(["ns", "t_idx"])
    plt.figure()
    for ns, g in dff.groupby(["ns"]):
        plt.plot(g["t_idx"], g["step_rmse"], marker="o", label=label_ns(ns))
    # GT step RMS reference if present
    if "gt_rms_step" in dff.columns:
        gt = safe_mean(dff["gt_rms_step"])
        if gt is not None:
            draw_gt_line(gt)
    plt.xlabel("t_idx")
    plt.ylabel(f"Step RMSE vs GT ({field})")
    plt.title(f"Step RMSE vs GT — field={field}")
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, borderaxespad=0.0)
    plt.tight_layout(rect=[0, 0, 0.80, 1])
    plt.tight_layout()
    plt.savefig(out_dir / f"rmse_step_field-{field}_var-{var}.png", dpi=dpi)
    plt.close()


def cum_plot_field(dfv: pd.DataFrame, field: str, var: int, out_dir: Path, dpi: int):
    dff = dfv[dfv["field"] == field].copy()
    if dff.empty:
        return
    # cumulative constant across t for a fixed (ns,var,field); pick first per ns
    g0 = (dff.drop_duplicates(subset=["ns", "field"]).sort_values(["ns"]))
    plt.figure()
    x = np.arange(len(g0))
    y = g0["cum_rmse"].astype(float).to_numpy()
    labels = [label_ns(ns) for ns in g0["ns"].tolist()]
    plt.bar(x, y)
    # GT cumulative RMS reference if present
    if "gt_rms_cum" in g0.columns:
        gt = safe_mean(g0["gt_rms_cum"])
        if gt is not None:
            draw_gt_line(gt)
    plt.xticks(x, labels, rotation=45, ha="right", fontsize=8)
    plt.ylabel(f"Cumulative RMSE vs GT ({field})")
    plt.title(f"Cumulative RMSE vs GT — field={field}")
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, borderaxespad=0.0)
    plt.tight_layout(rect=[0, 0, 0.80, 1])
    plt.tight_layout()
    plt.savefig(out_dir / f"rmse_cum_field-{field}_var-{var}.png", dpi=dpi)
    plt.close()


def compute_allfields(dfv: pd.DataFrame):
    # step_all: per (t_idx, ns, var)
    step = (dfv.groupby(["t_idx", "ns", "var"], as_index=False)
              .agg(step_mse_mean=("step_mse", "mean")))
    step["step_rmse_allfields"] = np.sqrt(step["step_mse_mean"].astype(float))

    # cum_all: per (ns,var)
    cum = (dfv.groupby(["ns", "var"], as_index=False)
             .agg(cum_mse_mean=("cum_mse", "mean")))
    cum["cum_rmse_allfields"] = np.sqrt(cum["cum_mse_mean"].astype(float))

    # GT RMS allfields if possible
    if "gt_rms_step" in dfv.columns:
        gt_step = (dfv.groupby(["t_idx", "ns", "var"], as_index=False)
                     .agg(gtm2=("gt_rms_step", lambda s: float(np.mean(np.square(pd.to_numeric(s, errors='coerce').dropna()))))))
        step["gt_rms_allfields_step"] = np.sqrt(gt_step["gtm2"].astype(float))

    if "gt_rms_cum" in dfv.columns:
        gt_cum = (dfv.groupby(["ns", "var"], as_index=False)
                    .agg(gtm2=("gt_rms_cum", lambda s: float(np.mean(np.square(pd.to_numeric(s, errors='coerce').dropna()))))))
        cum["gt_rms_allfields_cum"] = np.sqrt(gt_cum["gtm2"].astype(float))

    return step, cum


def step_plot_allfields(step: pd.DataFrame, var: int, out_dir: Path, dpi: int):
    d = step[step["var"] == var].sort_values(["ns", "t_idx"])
    if d.empty:
        return
    plt.figure()
    for ns, g in d.groupby(["ns"]):
        plt.plot(g["t_idx"], g["step_rmse_allfields"], marker="o", label=label_ns(ns))
    if "gt_rms_allfields_step" in d.columns:
        gt = safe_mean(d["gt_rms_allfields_step"])
        if gt is not None:
            draw_gt_line(gt)
    plt.xlabel("t_idx")
    plt.ylabel("Step RMSE vs GT (ALL FIELDS)")
    plt.title("Step RMSE vs GT — ALL FIELDS")
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, borderaxespad=0.0)
    plt.tight_layout(rect=[0, 0, 0.80, 1])
    plt.tight_layout()
    plt.savefig(out_dir / f"rmse_step_ALLFIELDS_var-{var}.png", dpi=dpi)
    plt.close()


def cum_plot_allfields(cum: pd.DataFrame, var: int, out_dir: Path, dpi: int):
    d = cum[cum["var"] == var].sort_values(["ns"])
    if d.empty:
        return
    plt.figure()
    x = np.arange(len(d))
    y = d["cum_rmse_allfields"].astype(float).to_numpy()
    labels = [label_ns(ns) for ns in d["ns"].tolist()]
    plt.bar(x, y)
    if "gt_rms_allfields_cum" in d.columns:
        gt = safe_mean(d["gt_rms_allfields_cum"])
        if gt is not None:
            draw_gt_line(gt)
    plt.xticks(x, labels, rotation=45, ha="right", fontsize=8)
    plt.ylabel("Cumulative RMSE vs GT (ALL FIELDS)")
    plt.title("Cumulative RMSE vs GT — ALL FIELDS")
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, borderaxespad=0.0)
    plt.tight_layout(rect=[0, 0, 0.80, 1])
    plt.tight_layout()
    plt.savefig(out_dir / f"rmse_cum_ALLFIELDS_var-{var}.png", dpi=dpi)
    plt.close()


def main():
    args = parse_args()
    csv_path = Path(args.csv).expanduser().resolve()
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    # numeric coercions
    for col in ["t_idx","ns","var"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for col in ["step_rmse","step_mse","cum_rmse","cum_mse","gt_rms_step","gt_rms_cum"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if args.dataset:
        df = df[df["dataset"].astype(str) == str(args.dataset)]

    if args.expt_id is not None and "expt_id" in df.columns:
        targets = norm_expt_targets(args.expt_id)
        expt_str = clean_expt_series(df["expt_id"])
        expt_pad = expt_str.where(~expt_str.str.match(r"^\d+$"), expt_str.str.zfill(5))
        df = df[expt_str.isin(targets) | expt_pad.isin(targets)]

    df = df.dropna(subset=["ns","var","t_idx","field","step_rmse","step_mse","cum_rmse","cum_mse"])
    if df.empty:
        raise SystemExit("No rows after filtering; check dataset/expt_id filters or CSV contents.")

    # ensure ints
    df["ns"] = df["ns"].astype(int)
    df["var"] = df["var"].astype(int)
    df["t_idx"] = df["t_idx"].astype(int)
    df["field"] = df["field"].astype(str)

    # Per-var split plotting
    vars_present = sorted(df["var"].unique().tolist())
    fields = sorted(df["field"].unique().tolist())

    for var in vars_present:
        dfv = df[df["var"] == var]
        for field in fields:
            step_plot_field(dfv, field, var, out_dir, args.dpi)
            cum_plot_field(dfv, field, var, out_dir, args.dpi)

    # All-fields combined
    step_all, cum_all = compute_allfields(df)
    for var in vars_present:
        step_plot_allfields(step_all, var, out_dir, args.dpi)
        cum_plot_allfields(cum_all, var, out_dir, args.dpi)

    print(f"Wrote plots to: {out_dir}")


if __name__ == "__main__":
    main()
