#!/usr/bin/env python3
import argparse
import csv
import math
import os
import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
#from matplotlib.ticker import MultipleLocator

# OLD: pretrain_n01_TrainLoss
COL_RE_OLD = re.compile(r"^(pretrain|finetune)_n(\d+)_([A-Za-z]+)$")

# NEW: pretrain_B_01_lr-3_opt-adam_wd-3_TrainLoss
COL_RE_NEW = re.compile(r"^(pretrain|finetune)_(B|L)_(\d+)_(.*)_(TrainLoss|ValidLoss)$")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="combined_losses.csv")
    p.add_argument("--phase", required=True, choices=["pretrain", "finetune"])
    p.add_argument(
        "--which",
        choices=["all", "even", "odd"],
        default="all",
        help=(
            "OLD headers: filter by nsteps parity. "
            "NEW headers: filter by run-id parity. "
            "(default: all)"
        ),
    )
    p.add_argument(
        "--variant",
        choices=["all", "B", "L"],
        default="all",
        help="For NEW headers, include only B or L runs (default: all). Ignored for OLD headers.",
    )
    p.add_argument(
        "--match",
        default=None,
        help="Optional substring filter applied to NEW run label (e.g. 'lr-3' or 'opt-adam').",
    )
    p.add_argument("--output", default=None)
    return p.parse_args()


def _sort_key(k):
    # k is either ('old', nstep:int) or ('new', variant:str, runid:int, config:str)
    if k[0] == "old":
        return (0, k[1])
    variant_key = 0 if k[1] == "B" else 1
    return (1, variant_key, k[2], k[3])


def load_combined_for_phase(path, phase, variant_filter="all", match_substr=None):
    if not os.path.exists(path):
        raise SystemExit(f"Input file {path!r} not found.")

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        rows = list(reader)

    if "Epoch" not in header:
        raise SystemExit("Expected an 'Epoch' column in the CSV header.")

    meta = {}  # key -> {"train_col": str|None, "valid_col": str|None, "label": str}

    for col in header:
        if col == "Epoch":
            continue

        m_old = COL_RE_OLD.match(col)
        if m_old:
            col_phase, nstep_str, kind = m_old.groups()
            if col_phase != phase:
                continue
            nstep = int(nstep_str)
            key = ("old", nstep)
            entry = meta.setdefault(key, {"train_col": None, "valid_col": None, "label": f"n={nstep}"})
            kl = kind.lower()
            if kl.startswith("train"):
                entry["train_col"] = col
            elif kl.startswith("valid"):
                entry["valid_col"] = col
            continue

        m_new = COL_RE_NEW.match(col)
        if m_new:
            col_phase, variant, runid_str, config, kind = m_new.groups()
            if col_phase != phase:
                continue
            if variant_filter != "all" and variant != variant_filter:
                continue

            label = f"{variant}_{int(runid_str):02d}\n{config}"
            if match_substr and match_substr not in label:
                continue

            key = ("new", variant, int(runid_str), config)
            entry = meta.setdefault(key, {"train_col": None, "valid_col": None, "label": label})
            if kind == "TrainLoss":
                entry["train_col"] = col
            else:
                entry["valid_col"] = col
            continue

    epochs = [int(r["Epoch"]) for r in rows]

    runs = {}
    for key, m in meta.items():
        tcol = m["train_col"]
        vcol = m["valid_col"]
        train_vals, valid_vals = [], []

        for r in rows:
            s = (r.get(tcol) or "").strip() if tcol else ""
            train_vals.append(float(s) if s else None)

            s = (r.get(vcol) or "").strip() if vcol else ""
            valid_vals.append(float(s) if s else None)

        runs[key] = {"train": train_vals, "valid": valid_vals, "label": m["label"]}

    return epochs, runs


def apply_parity_filter(keys, which):
    if which == "all":
        return keys
    out = []
    for k in keys:
        n = k[1] if k[0] == "old" else k[2]  # old: nsteps; new: runid
        if which == "even" and n % 2 == 0:
            out.append(k)
        if which == "odd" and n % 2 == 1:
            out.append(k)
    return out


def main():
    args = parse_args()

    epochs, runs = load_combined_for_phase(
        args.input, args.phase, variant_filter=args.variant, match_substr=args.match
    )
    if not runs:
        raise SystemExit(f"No runs found for phase={args.phase!r} in {args.input}")

    keys = sorted(runs.keys(), key=_sort_key)
    keys = apply_parity_filter(keys, args.which)

    if not keys:
        raise SystemExit(f"No runs remaining after applying filter which={args.which!r}")

    if args.output is None:
        suffix = f"{args.phase}_{args.which}"
        if args.variant != "all":
            suffix += f"_{args.variant}"
        if args.match:
            suffix += "_match"
        args.output = f"{suffix}_grid.png"

    # --- Global outlier cutoff based on median across all selected runs ---
    all_vals = []
    for k in keys:
        all_vals.extend([v for v in runs[k]["train"] if v is not None])
        all_vals.extend([v for v in runs[k]["valid"] if v is not None])

    if not all_vals:
        raise SystemExit("No loss values found to plot.")

    arr = np.array(all_vals, dtype=float)
    median_val = float(np.median(arr))
    factor = 100
    cutoff = median_val * factor
    if cutoff <= 0:
        cutoff = float(np.quantile(arr, 0.99))
    print(f"Global median loss = {median_val:.4g}, global cutoff = {cutoff:.4g}")

    # --- Build filtered series per run, and collect all plotted y-values to set y_max ---
    series_by_key = {}
    plotted = []

    for k in keys:
        train_vals = runs[k]["train"]
        valid_vals = runs[k]["valid"]

        # early-epoch average from first 5 points (same logic as before)
        train_early = [t for t in train_vals[:5] if t is not None]
        valid_early = [v for v in valid_vals[:5] if v is not None]
        train_avg = sum(train_early) / len(train_early) if train_early else None
        valid_avg = sum(valid_early) / len(valid_early) if valid_early else None

        tx, ty, vx, vy = [], [], [], []

        # IMPORTANT: no epoch cap — we iterate all rows in combined_losses.csv
        for idx, (e, t, v) in enumerate(zip(epochs, train_vals, valid_vals)):
            #if t is not None and t <= cutoff:
            if t is not None:
                if idx < 5 or train_avg is None or t <= train_avg:
                     tx.append(e)
                     ty.append(t)
                     plotted.append(t)
            #if v is not None and v <= cutoff:
            if v is not None:
                if idx < 5 or valid_avg is None or v <= valid_avg:
                   vx.append(e)
                   vy.append(v)
                   plotted.append(v)

        series_by_key[k] = {"train_x": tx, "train_y": ty, "valid_x": vx, "valid_y": vy}

    if not plotted:
        raise SystemExit("After filtering, no points remain to plot.")
    y_max = max(plotted) * 1.05

    # --- Grid size: 4 plots per row (plot ALL runs) ---
    n = len(keys)
    ncols = min(4, n)  # up to 4 columns, fewer if <4 runs
    nrows = math.ceil(n / ncols)

    # Scale figure size with grid, but keep it reasonable.
    # With 4 columns, use a slightly narrower per-subplot width.
    fig_w = max(12, 3.6 * ncols)
    fig_h = max(8, 3.0 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), sharex=True, sharey=True)

    # axes can be a single Axes, 1D, or 2D; normalize to 1D list
    if not isinstance(axes, (list, np.ndarray)):
        axes_list = [axes]
    else:
        axes_list = np.array(axes).ravel().tolist()

    # Hide any unused axes
    for ax in axes_list[n:]:
        ax.set_visible(False)

    # Plot each run
    for i, k in enumerate(keys):
        ax = axes_list[i]
        s = series_by_key[k]
        if s["train_x"]:
            ax.plot(s["train_x"], s["train_y"], label="Train")
        if s["valid_x"]:
            ax.plot(s["valid_x"], s["valid_y"], label="Valid")

        ax.set_title(runs[k]["label"], fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0, top=y_max)
        #ax.xaxis.set_major_locator(MaxNLocator(integer=True, step=2))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        #ax.xaxis.set_major_locator(MultipleLocator(2))  # every 2 epochs

    fig.text(0.5, 0.02, "Epoch", ha="center")
    fig.text(0.02, 0.5, "Loss", va="center", rotation="vertical")

    title = f"{args.phase.capitalize()} — {args.which} (Train & Valid Loss)"
    if args.variant != "all":
        title += f" [{args.variant}]"
    if args.match:
        title += " [match]"
    fig.suptitle(title, fontsize=14)

    # Global legend (use first visible axis that has labels)
    for ax in axes_list:
        h, l = ax.get_legend_handles_labels()
        if h:
            fig.legend(h, l, loc="upper right")
            break

    plt.tight_layout(rect=[0.04, 0.04, 0.98, 0.95])
    plt.savefig(args.output, dpi=150)
    print(f"Saved {args.output} (runs={n}, grid={nrows}x{ncols})")


if __name__ == "__main__":
    main()

