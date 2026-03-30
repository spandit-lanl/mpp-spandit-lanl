#!/usr/bin/env python3
"""
CLX experiment loss pipeline (replaces):
  01_extract_losses_clx_expt.bash
  02_combine_losses_clx_expt.py
  04_plot_grid_losses_uniq_EPOCH_SCALES_clx_expt.py

Steps:
  1) Extract per-run loss CSVs from log files
  2) Combine per-run loss CSVs into a wide CSV
  3) Plot per-ns grids into PDFs

Default behavior matches prior scripts:
  - Reads logs from a fixed LOG_DIR (override via --log-dir)
  - Writes per-log loss CSVs into the current working directory
  - Writes combined_losses_clx_expt.csv into the current working directory
  - Writes clx_expt_ns-NN_grid.pdf into --outdir (default: current working directory)

Expected log filename pattern:
  clx_expt_ns-<2 digits>_var-<digits>.log

Expected combined column naming (produced by this script):
  clx_expt_nsNN_varV_train_loss
  clx_expt_nsNN_varV_valid_loss
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# -----------------------------
# Configuration / patterns
# -----------------------------

DEFAULT_LOG_DIR = "/users/spandit/projects/artimis/mpp/mpp-spandit-npz-clx/final"

LOG_FILE_GLOB = "clx_expt_*.log"
LOG_NAME_RE = re.compile(r"^clx_expt_ns-(\d{2})_var-(\d+)\.log$")

# Log line pattern:
# Epoch: <int>. Train loss: tensor([<float>], ...). Valid loss: <float>
LOSS_LINE_RE = re.compile(
    r"Epoch:\s*(\d+)\.\s*Train loss:\s*tensor\(\[([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\]"
    r".*?\)\.\s*Valid loss:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
)

# Extracted CSV files written into PWD:
# loss_clx_expt_ns-##_var-#.csv
LOSS_CSV_NAME_RE = re.compile(r"^loss_clx_expt_ns-(\d{2})_var-(\d+)\.csv$")

COMBINED_OUT_DEFAULT = "combined_losses_clx_expt.csv"

# Combined columns tag format:
# clx_expt_nsNN_varV_{train|valid}_loss
COMBINED_TAG_RE = re.compile(r"^clx_expt_ns(\d{2})_var(\d+)$")


# -----------------------------
# Data structures
# -----------------------------

@dataclass(frozen=True)
class RunId:
    ns: int
    var: int

    @property
    def ns_str(self) -> str:
        return f"{self.ns:02d}"

    @property
    def var_str(self) -> str:
        return str(self.var)

    @property
    def tag(self) -> str:
        # used in combined CSV column prefixes
        return f"clx_expt_ns{self.ns_str}_var{self.var_str}"

    @property
    def loss_csv_name(self) -> str:
        return f"loss_clx_expt_ns-{self.ns_str}_var-{self.var_str}.csv"

    @property
    def pdf_name(self) -> str:
        # user requested dash after ns: ns-NN
        return f"clx_expt_ns-{self.ns_str}_grid.pdf"


# -----------------------------
# Step 1: extract loss CSVs
# -----------------------------

def iter_log_files(log_dir: Path) -> List[Tuple[RunId, Path]]:
    """Return list of (RunId, log_path) for logs matching expected naming."""
    candidates = sorted(log_dir.glob(LOG_FILE_GLOB))
    runs: List[Tuple[RunId, Path]] = []
    for p in candidates:
        m = LOG_NAME_RE.match(p.name)
        if not m:
            continue
        ns = int(m.group(1))
        var = int(m.group(2))
        runs.append((RunId(ns=ns, var=var), p))
    return runs


def parse_losses_from_log(log_path: Path) -> pd.DataFrame:
    """Parse epoch/train_loss/valid_loss from a log file."""
    epochs: List[int] = []
    train_losses: List[float] = []
    valid_losses: List[float] = []

    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = LOSS_LINE_RE.search(line)
            if not m:
                continue
            epochs.append(int(m.group(1)))
            train_losses.append(float(m.group(2)))
            valid_losses.append(float(m.group(3)))

    df = pd.DataFrame(
        {"epoch": epochs, "train_loss": train_losses, "valid_loss": valid_losses}
    )
    if not df.empty:
        df = df.sort_values("epoch").drop_duplicates(subset=["epoch"], keep="last").reset_index(drop=True)
    return df


def write_loss_csv(df: pd.DataFrame, out_path: Path) -> None:
    """Write extracted losses CSV with required header; even if empty, write header."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "valid_loss"])
        if not df.empty:
            for _, row in df.iterrows():
                w.writerow([int(row["epoch"]), row["train_loss"], row["valid_loss"]])


def extract_all_losses(log_dir: Path, out_dir: Path) -> List[Path]:
    """
    Extract per-run loss CSVs from log_dir and write into out_dir (PWD by default).
    Returns list of written CSV paths.
    """
    runs = iter_log_files(log_dir)
    if not runs:
        raise FileNotFoundError(
            f"No parseable logs found in {log_dir} matching clx_expt_ns-##_var-#.log"
        )

    written: List[Path] = []
    for run_id, log_path in runs:
        df = parse_losses_from_log(log_path)
        out_csv = out_dir / run_id.loss_csv_name
        write_loss_csv(df, out_csv)
        written.append(out_csv)
    return written


# -----------------------------
# Step 2: combine loss CSVs
# -----------------------------

def iter_loss_csv_files(search_pattern: str) -> List[Tuple[RunId, Path]]:
    """Find per-run loss CSVs and return sorted list of (RunId, path)."""
    paths = [Path(p) for p in glob.glob(search_pattern, recursive=True)]
    runs: List[Tuple[RunId, Path]] = []
    for p in paths:
        m = LOSS_CSV_NAME_RE.match(p.name)
        if not m:
            continue
        ns = int(m.group(1))
        var = int(m.group(2))
        runs.append((RunId(ns=ns, var=var), p))
    runs.sort(key=lambda t: (t[0].ns, t[0].var))
    return runs


def read_loss_csv(path: Path) -> pd.DataFrame:
    """Read a per-run loss CSV; tolerate empty."""
    df = pd.read_csv(path)
    # Ensure expected columns exist even if empty
    for col in ["epoch", "train_loss", "valid_loss"]:
        if col not in df.columns:
            raise ValueError(f"{path} missing required column '{col}'")
    return df


def combine_losses(
    loss_csv_pattern: str,
    combined_out: Path,
) -> pd.DataFrame:
    """
    Combine per-run CSVs into a wide table keyed by epoch.
    Produces columns: epoch + <tag>_train_loss + <tag>_valid_loss
    """
    runs = iter_loss_csv_files(loss_csv_pattern)
    if not runs:
        raise FileNotFoundError(
            f"No input files matched pattern {loss_csv_pattern} with expected name "
            f"loss_clx_expt_ns-##_var-#.csv"
        )

    # Union of epochs across all runs
    all_epochs: set[int] = set()
    per_run: Dict[RunId, pd.DataFrame] = {}

    for run_id, p in runs:
        df = read_loss_csv(p)
        per_run[run_id] = df
        all_epochs.update(int(e) for e in df["epoch"].dropna().tolist())

    epoch_list = sorted(all_epochs)
    out = pd.DataFrame({"epoch": epoch_list})

    for run_id, _p in runs:
        df = per_run[run_id]
        df2 = df[["epoch", "train_loss", "valid_loss"]].copy()
        df2["epoch"] = df2["epoch"].astype(int)

        out = out.merge(
            df2.rename(
                columns={
                    "train_loss": f"{run_id.tag}_train_loss",
                    "valid_loss": f"{run_id.tag}_valid_loss",
                }
            ),
            on="epoch",
            how="left",
        )

    combined_out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(combined_out, index=False)
    return out


def eight_ticks(max_epoch: int) -> List[int]:
    """Return exactly 8 integer tick locations from 0..max_epoch (inclusive)."""
    if max_epoch <= 0:
        return [0, 1, 2, 3, 4, 5, 6, 7]
    raw = np.linspace(0, max_epoch, 8)
    ticks = [int(round(x)) for x in raw]
    ticks[0] = 0
    ticks[-1] = int(max_epoch)
    for i in range(1, len(ticks)):
        if ticks[i] <= ticks[i - 1]:
            ticks[i] = ticks[i - 1] + 1
    if ticks[-1] > max_epoch:
        ticks[-1] = int(max_epoch)
        for i in range(len(ticks) - 2, -1, -1):
            if ticks[i] >= ticks[i + 1]:
                ticks[i] = max(0, ticks[i + 1] - 1)
    return ticks


# -----------------------------
# Step 3: plot grid per ns
# -----------------------------

def discover_runs_from_combined(df: pd.DataFrame) -> Dict[int, List[RunId]]:
    """
    From combined dataframe columns, identify runs and group by ns.
    Expects column names like:
      clx_expt_nsNN_varV_train_loss
      clx_expt_nsNN_varV_valid_loss
    """
    runs_by_ns: Dict[int, Dict[int, RunId]] = {}
    for c in df.columns:
        if not c.endswith("_train_loss"):
            continue
        prefix = c[: -len("_train_loss")]
        m = COMBINED_TAG_RE.match(prefix)
        if not m:
            continue
        ns = int(m.group(1))
        var = int(m.group(2))
        runs_by_ns.setdefault(ns, {})[var] = RunId(ns=ns, var=var)

    out: Dict[int, List[RunId]] = {}
    for ns, by_var in runs_by_ns.items():
        out[ns] = [by_var[v] for v in sorted(by_var.keys())]
    return out


def plot_ns_grid(
    df: pd.DataFrame,
    ns: int,
    runs: Sequence[RunId],
    out_path: Path,
    ncols: int = 4,
    y_limits: Optional[Tuple[float, float]] = None,  # NEW: fixed y-lims across all PDFs
) -> None:
    """
    Plot a grid for a single ns.

    Requirements enforced (within this grid):
      - Identical X limits for every panel (0..max_epoch)
      - X axis is integer-only and has exactly 8 ticks
      - Identical Y limits for every panel
        * If y_limits is provided: use those fixed limits (same across all ns PDFs)
        * Else: fall back to per-ns global min/max (previous behavior)
    """
    if "epoch" not in df.columns:
        raise ValueError("Combined dataframe missing 'epoch' column")

    # Ensure epochs are integers (avoids float tick labeling surprises)
    df = df.copy()
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce").astype("Int64")
    df = df[df["epoch"].notna()].copy()
    df["epoch"] = df["epoch"].astype(int)

    n = len(runs)
    nrows = max(1, math.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4.2 * ncols, 3.2 * nrows),
        squeeze=False,
        sharex=True,
        sharey=True,
    )

    # 1) Determine max_epoch for this ns (used for *all* panels)
    max_epoch = 0
    for run in runs:
        tcol = f"{run.tag}_train_loss"
        vcol = f"{run.tag}_valid_loss"
        if tcol in df.columns:
            e = df.loc[pd.to_numeric(df[tcol], errors="coerce").notna(), "epoch"].max()
            if pd.notna(e):
                max_epoch = max(max_epoch, int(e))
        if vcol in df.columns:
            e = df.loc[pd.to_numeric(df[vcol], errors="coerce").notna(), "epoch"].max()
            if pd.notna(e):
                max_epoch = max(max_epoch, int(e))

    # Fallback (shouldn't happen, but keep robust)
    if max_epoch <= 0:
        max_epoch = int(df["epoch"].max()) if pd.notna(df["epoch"].max()) else 0

    # Exactly 8 integer ticks, same in every subplot
    xticks = eight_ticks(max_epoch)

    # 2) Y limits: either fixed global (preferred) or per-ns fallback
    if y_limits is not None:
        y_lo, y_hi = y_limits
    else:
        # Previous behavior: per-ns global y-range over epochs <= max_epoch
        d_all = df[df["epoch"] <= max_epoch]
        y_min = float("inf")
        y_max = float("-inf")
        for run in runs:
            for suffix in ("_train_loss", "_valid_loss"):
                col = f"{run.tag}{suffix}"
                if col in d_all.columns:
                    s = pd.to_numeric(d_all[col], errors="coerce")
                    if s.notna().any():
                        y_min = min(y_min, float(s.min()))
                        y_max = max(y_max, float(s.max()))

        y_lo = y_hi = None
        if math.isfinite(y_min) and math.isfinite(y_max):
            pad = 0.05 * (y_max - y_min) if y_max > y_min else 0.1
            y_lo = y_min - pad
            y_hi = y_max + pad

    # 3) Plot each panel with identical axis limits/ticks
    for idx, run in enumerate(runs):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r][c]

        tcol = f"{run.tag}_train_loss"
        vcol = f"{run.tag}_valid_loss"

        if tcol not in df.columns and vcol not in df.columns:
            ax.set_title(f"var-{run.var} (missing cols)")
            ax.axis("off")
            continue

        d = df[df["epoch"] <= max_epoch]

        if tcol in d.columns:
            ax.plot(d["epoch"], pd.to_numeric(d[tcol], errors="coerce"), label="train")
        if vcol in d.columns:
            ax.plot(d["epoch"], pd.to_numeric(d[vcol], errors="coerce"), label="valid")

        ax.set_title(f"var-{run.var}")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")

        ax.set_xlim(0, max_epoch)
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(int(x)) for x in xticks])  # force integer labels

        if y_lo is not None and y_hi is not None:
            ax.set_ylim(y_lo, y_hi)

        ax.legend()

    # Turn off any unused axes
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle(f"CLX losses grid (ns-{ns:02d})", y=0.995)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf")
    plt.close(fig)


def plot_all_ns_grids(
    combined_csv: Path,
    outdir: Path,
    ncols: int = 4,
) -> List[Path]:
    df = pd.read_csv(combined_csv)
    runs_by_ns = discover_runs_from_combined(df)
    if not runs_by_ns:
        raise ValueError(
            f"No runs discovered from columns in {combined_csv}. "
            f"Expected columns like clx_expt_nsNN_varV_train_loss"
        )

    # NEW: Compute ONE global Y range across *all* train/valid loss columns (all ns)
    y_min = float("inf")
    y_max = float("-inf")
    for c in df.columns:
        if c.endswith("_train_loss") or c.endswith("_valid_loss"):
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().any():
                y_min = min(y_min, float(s.min()))
                y_max = max(y_max, float(s.max()))

    y_limits: Optional[Tuple[float, float]] = None
    if math.isfinite(y_min) and math.isfinite(y_max):
        pad = 0.05 * (y_max - y_min) if y_max > y_min else 0.1
        y_limits = (y_min - pad, y_max + pad)

    written: List[Path] = []
    for ns in sorted(runs_by_ns.keys()):
        runs = runs_by_ns[ns]
        out_path = outdir / RunId(ns=ns, var=0).pdf_name  # var unused in pdf_name
        plot_ns_grid(df=df, ns=ns, runs=runs, out_path=out_path, ncols=ncols, y_limits=y_limits)
        written.append(out_path)
    return written


# -----------------------------
# Orchestration / CLI
# -----------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run CLX loss pipeline (extract → combine → plot).")
    p.add_argument("--log-dir", type=str, default=DEFAULT_LOG_DIR, help="Directory containing clx_expt_*.log files.")
    p.add_argument(
        "--loss-csv-pattern",
        type=str,
        default="loss_clx_expt_ns-*_var-*.csv",
        help="Glob pattern (recursive allowed) for extracted per-run loss CSVs.",
    )
    p.add_argument(
        "--combined-out",
        type=str,
        default=COMBINED_OUT_DEFAULT,
        help="Output combined CSV filename.",
    )
    p.add_argument(
        "--outdir",
        type=str,
        default=".",
        help="Directory for PDF outputs (and by default per-run CSVs are written to PWD).",
    )
    p.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip extraction step (assume per-run loss CSVs already exist).",
    )
    p.add_argument(
        "--skip-combine",
        action="store_true",
        help="Skip combine step (assume combined CSV already exists).",
    )
    p.add_argument(
        "--skip-plot",
        action="store_true",
        help="Skip plotting step.",
    )
    p.add_argument(
        "--ncols",
        type=int,
        default=4,
        help="Number of columns in the plot grid.",
    )
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    log_dir = Path(args.log_dir)
    outdir = Path(args.outdir)
    combined_out = Path(args.combined_out)

    # Step 1: extract (writes into PWD like the old flow)
    if not args.skip_extract:
        written = extract_all_losses(log_dir=log_dir, out_dir=Path("."))
        print(f"[extract] wrote {len(written)} loss CSV files into {Path('.').resolve()}", flush=True)

    # Step 2: combine
    if not args.skip_combine:
        df = combine_losses(loss_csv_pattern=args.loss_csv_pattern, combined_out=combined_out)
        print(f"[combine] wrote {combined_out} with shape {df.shape}", flush=True)

    # Step 3: plot
    if not args.skip_plot:
        pdfs = plot_all_ns_grids(combined_csv=combined_out, outdir=outdir, ncols=args.ncols)
        print(f"[plot] wrote {len(pdfs)} PDFs into {outdir.resolve()}", flush=True)
        for p in pdfs:
            print(f"  - {p}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

