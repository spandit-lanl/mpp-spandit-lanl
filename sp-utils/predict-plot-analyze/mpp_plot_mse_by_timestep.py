#!/usr/bin/env python3
"""
Plot mean MSE vs timestep across all simulations from a wide metrics CSV.
Layout:
  Row 1: Uvelocity, Wvelocity
  Row 2: density_maincharge, av_density
  Row 3: mse_avg_over_fields (same size as others, centered)

Expected columns:
  timestep,
  mse_av_density, mse_Uvelocity, mse_Wvelocity, mse_density_maincharge, mse_mean_fields
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


# (column_name, plot_title, color)
PLOTS = {
    "Uvelocity": ("mse_Uvelocity", "Uvelocity", "#ff7f0e"),
    "Wvelocity": ("mse_Wvelocity", "Wvelocity", "#2ca02c"),
    "density_maincharge": ("mse_density_maincharge", "density_maincharge", "#d62728"),
    "av_density": ("mse_av_density", "av_density", "#1f77b4"),
    "mse_avg_over_fields": ("mse_mean_fields", "mse_avg_over_fields", "#000000"),
}


def center_axis_same_size(ax, ref_ax):
    """Center ax horizontally while keeping its size equal to ref_ax."""
    ref_pos = ref_ax.get_position()
    pos = ax.get_position()
    new_x0 = 0.5 - pos.width / 2.0
    ax.set_position([new_x0, pos.y0, pos.width, pos.height])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        default="/lustre/scratch5/exempt/artimis/mpmm/spandit/runs-ch/mpp/final/pred/clx/metrics_wide.csv",
        help="Path to metrics_wide.csv",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output PDF path (default: alongside CSV as mse_by_timestep.pdf)",
    )
    args = ap.parse_args()

    csv_path = args.csv
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if "timestep" not in df.columns:
        raise ValueError("CSV must contain a 'timestep' column")

    required_cols = [v[0] for v in PLOTS.values()]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    # Coerce timestep to int for grouping/sorting
    df["timestep"] = pd.to_numeric(df["timestep"], errors="coerce")
    df = df.dropna(subset=["timestep"]).copy()
    df["timestep"] = df["timestep"].astype(int)

    # Mean across simulations for each timestep
    agg = (
        df.groupby("timestep", as_index=False)[required_cols]
        .mean()
        .sort_values("timestep")
    )

    out_path = args.out
    if out_path is None:
        out_path = os.path.join(os.path.dirname(os.path.abspath(csv_path)), "mse_by_timestep.pdf")
    if not out_path.lower().endswith(".pdf"):
        out_path += ".pdf"
    # 3 rows x 2 cols; bottom plot centered and same size
    fig = plt.figure(figsize=(14, 10))

    fig.suptitle(
        "Mean MSE vs Timestep (Averaged Across Simulations)\nCylex Test Dataset",
        fontsize=18,
        fontweight="bold",
        y=0.985,
    )

    gs = fig.add_gridspec(
        3, 2,
        left=0.06, right=0.98,
        bottom=0.06,
        top=0.90,          # <-- lower this to add more space under suptitle
        wspace=0.18,
        hspace=0.28
    )

    ax_u = fig.add_subplot(gs[0, 0])
    ax_w = fig.add_subplot(gs[0, 1])
    ax_dm = fig.add_subplot(gs[1, 0])
    ax_ad = fig.add_subplot(gs[1, 1])
    ax_mean = fig.add_subplot(gs[2, 0])  # create in left cell, then center it

    layout = [
        (ax_u, "Uvelocity"),
        (ax_w, "Wvelocity"),
        (ax_dm, "density_maincharge"),
        (ax_ad, "av_density"),
        (ax_mean, "mse_avg_over_fields"),
    ]

    for ax, key in layout:
        col, title, color = PLOTS[key]
        ax.plot(agg["timestep"].values, agg[col].values, color=color, linewidth=1.8)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("timestep")
        ax.set_ylabel("mean MSE")

        # crisp grid
        ax.grid(True, which="both", alpha=0.35, linewidth=0.8)

    # Center the bottom axis horizontally while keeping its size the same as others
    center_axis_same_size(ax_mean, ax_u)

    fig.savefig(out_path, format="pdf")
    print(f"Wrote plot: {out_path}")


if __name__ == "__main__":
    main()
