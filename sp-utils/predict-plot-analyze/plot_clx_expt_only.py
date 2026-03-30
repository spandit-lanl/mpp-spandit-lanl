#!/usr/bin/env python3
"""
plot_clx_expt_only.py

Reads:  combined_losses_clx_expt.csv
Writes: One PDF per ns (Temporal Context Window size), each PDF contains a grid of
        subplots for variations 1..8. Each subplot overlays train and valid loss.

Key behaviors:
- Global Y-axis scale is the SAME across all subplots in all PDFs (computed from all
  train/valid loss columns found in the CSV).
- Figure aspect is made more rectangular: height = 70% of width.

Expected column naming in combined CSV:
  clx_expt_ns02_var1_train_loss
  clx_expt_ns02_var1_valid_loss
  ...
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

import pandas as pd
import matplotlib.pyplot as plt


COL_RE = re.compile(r"^clx_expt_ns(\d+)_var(\d+)_(train_loss|valid_loss)$")


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def _coerce_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _global_ylim(df: pd.DataFrame, loss_cols: List[str]) -> tuple[float, float]:
    # Compute global min/max across all provided loss columns, ignoring NaNs.
    vals = pd.concat([_coerce_float_series(df[c]) for c in loss_cols], axis=0)
    vals = vals.dropna()
    if vals.empty:
        # Fallback
        return (0.0, 1.0)
    ymin = float(vals.min())
    ymax = float(vals.max())
    if ymin == ymax:
        # Avoid zero-range plots
        pad = 0.1 * abs(ymax) if ymax != 0 else 0.1
        return (ymin - pad, ymax + pad)
    # Add a small padding so curves don't touch borders
    pad = 0.05 * (ymax - ymin)
    return (ymin - pad, ymax + pad)


def main() -> int:
    out_dir = _script_dir()
    in_csv = out_dir / "combined_losses_clx_expt.csv"
    if not in_csv.exists():
        raise FileNotFoundError(f"Expected input not found: {in_csv}")

    df = pd.read_csv(in_csv)

    if "epoch" not in df.columns:
        raise ValueError("combined_losses_clx_expt.csv must contain an 'epoch' column")

    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    df = df.dropna(subset=["epoch"]).sort_values("epoch")
    epochs = df["epoch"].to_numpy()

    # mapping: ns -> var -> {"train_loss": colname, "valid_loss": colname}
    ns_map: Dict[int, Dict[int, Dict[str, str]]] = {}
    loss_cols: List[str] = []

    for col in df.columns:
        if col == "epoch":
            continue
        m = COL_RE.match(col)
        if not m:
            continue
        ns = int(m.group(1))
        var = int(m.group(2))
        kind = m.group(3)
        ns_map.setdefault(ns, {}).setdefault(var, {})[kind] = col
        loss_cols.append(col)

    if not ns_map:
        raise ValueError(
            "No columns matched the expected pattern "
            "'clx_expt_ns##_var#_(train_loss|valid_loss)'. "
            "Check combined_losses_clx_expt.csv column names."
        )

    ylow, yhigh = _global_ylim(df, loss_cols)

    # Force consistent x-axis maxima for specific ns values
    # ns corresponds to Temporal Context Window size (n_steps)
    XMAX_BY_NS = {1: 240, 2: 100, 4: 60, 6: 60}

    # Figure size: height = 70% of width
    fig_width = 16.0
    fig_height = 7  # ~70% of the original 8.5" height: vertically squeezed (Y direction)
    nrows, ncols = 2, 4
    vars_expected = list(range(1, 9))

    # Plot one PDF per ns
    for ns in sorted(ns_map.keys()):
        var_map = ns_map[ns]

        fig = plt.figure(figsize=(fig_width, fig_height))
        fig.suptitle(f"Temporal Context Window size {ns}", fontsize=16)

        for i, var in enumerate(vars_expected, start=1):
            ax = fig.add_subplot(nrows, ncols, i)
            ax.set_title(f"Variation {var}", fontsize=11)

            cols = var_map.get(var, {})
            train_col = cols.get("train_loss")
            valid_col = cols.get("valid_loss")

            plotted_any = False
            if train_col and train_col in df.columns:
                y_train = _coerce_float_series(df[train_col]).to_numpy()
                ax.plot(epochs, y_train, label="Train")
                plotted_any = True

            if valid_col and valid_col in df.columns:
                y_valid = _coerce_float_series(df[valid_col]).to_numpy()
                ax.plot(epochs, y_valid, label="Valid")
                plotted_any = True

            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            # Apply requested fixed x-axis maximum for some ns values
            if ns in XMAX_BY_NS:
                ax.set_xlim(left=0, right=XMAX_BY_NS[ns])
            ax.set_ylim(ylow, yhigh)
            ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)

            if plotted_any:
                ax.legend(fontsize=9)
            else:
                ax.text(
                    0.5, 0.5, "No data",
                    ha="center", va="center",
                    transform=ax.transAxes, fontsize=10
                )

        fig.tight_layout(rect=[0, 0.02, 1, 0.93])
        out_pdf = out_dir / f"clx_expt_ns{ns:02d}_loss_grid.pdf"
        fig.savefig(out_pdf, format="pdf")
        plt.close(fig)
        print(f"Wrote {out_pdf}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
