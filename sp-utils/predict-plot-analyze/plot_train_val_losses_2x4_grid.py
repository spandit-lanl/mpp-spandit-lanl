#!/usr/bin/env python3
"""
plot_grid_even_odd_nsteps.py

Example:
  python plot_grid_even_odd_nsteps.py \
    --pattern "out_pretrain_b_MPP_nsteps_*.txt" \
    --parity even \
    --out pretrain_even_grid.png

Each of the 8 panels (2x4) shows Train & Valid loss vs epoch for one log file.
"""

import argparse
import glob
import os
import re
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt


# ---- Regex to parse from log lines ----
EPOCH_RE = re.compile(r"\bEpoch\b[:\s]+(\d+)\b", re.IGNORECASE)

# Matches: Train loss: tensor([8.2894], device='cuda:0')
TRAIN_RE = re.compile(
    r"Train\s+loss:\s*tensor\(\[\s*([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)\s*\]",
    re.IGNORECASE,
)

# Matches: Valid loss: 8.915376663208008
VALID_RE = re.compile(
    r"Valid\s+loss:\s*([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)",
    re.IGNORECASE,
)

# Extract nsteps from filename like "...nsteps_03..."
NSTEPS_RE = re.compile(r"nsteps[_-](\d+)", re.IGNORECASE)


def extract_nsteps(path: str) -> Optional[int]:
    m = NSTEPS_RE.search(os.path.basename(path))
    return int(m.group(1)) if m else None


def parse_log(path: str, max_loss: float) -> Tuple[List[int], List[float], List[float]]:
    """
    Returns (epochs, train_losses, valid_losses) after filtering out epochs where
    train OR valid loss > max_loss.

    If epoch is not found, we fallback to an internal counter.
    """
    epochs: List[int] = []
    train_losses: List[float] = []
    valid_losses: List[float] = []

    current_epoch = None
    fallback_epoch_counter = 0

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m_ep = EPOCH_RE.search(line)
            if m_ep:
                current_epoch = int(m_ep.group(1))

            m_tr = TRAIN_RE.search(line)
            m_va = VALID_RE.search(line)

            if m_tr and m_va:
                tr = float(m_tr.group(1))
                va = float(m_va.group(1))

                if tr > max_loss or va > max_loss:
                    continue

                if current_epoch is None:
                    # fallback if epoch not printed
                    fallback_epoch_counter += 1
                    ep = fallback_epoch_counter
                else:
                    ep = current_epoch

                epochs.append(ep)
                train_losses.append(tr)
                valid_losses.append(va)

    return epochs, train_losses, valid_losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", required=True,
                    help='Glob for log files, e.g. "out_pretrain_b_MPP_nsteps_*.txt"')
    ap.add_argument("--parity", choices=["even", "odd"], required=True,
                    help="Select only even or odd n_steps files")
    ap.add_argument("--max_loss", type=float, default=0.5,
                    help="Ignore epochs where train OR valid loss exceeds this threshold")
    ap.add_argument("--out", default="loss_grid.png",
                    help="Output image filename (png)")
    ap.add_argument("--title", default="Train/Valid Loss vs Epoch (Pretraining Logs)",
                    help="Figure title")
    ap.add_argument("--limit", type=int, default=8,
                    help="How many panels to plot (default 8 for a 2x4 grid)")
    args = ap.parse_args()

    files = sorted(glob.glob(args.pattern))
    if not files:
        raise SystemExit(f"No files matched pattern: {args.pattern}")

    # Build (nsteps, path) list and filter by parity
    items = []
    for p in files:
        ns = extract_nsteps(p)
        if ns is None:
            continue
        if args.parity == "even" and (ns % 2 != 0):
            continue
        if args.parity == "odd" and (ns % 2 != 1):
            continue
        items.append((ns, p))

    if not items:
        raise SystemExit(f"No files matched parity '{args.parity}' with nsteps in filename.")

    # sort by nsteps
    items.sort(key=lambda x: x[0])

    # take up to limit (8)
    items = items[:args.limit]

    # Prepare 2x4 grid
    nrows, ncols = 2, 4
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 7))
    fig.suptitle(args.title)

    # Flatten axes for easy iteration
    axes_flat = [ax for row in axes for ax in row]

    for ax in axes_flat[len(items):]:
        ax.axis("off")

    for idx, (ns, path) in enumerate(items):
        ax = axes_flat[idx]
        epochs, tr, va = parse_log(path, max_loss=args.max_loss)

        if not epochs:
            ax.set_title(f"Context Window Size = {ns}\n(no points after filter)")
            ax.axis("off")
            continue

        # One panel per file: plot both lines
        ax.plot(epochs, tr, marker="o", markersize=2, linewidth=1.0, color="tab:blue", label="Train")
        ax.plot(epochs, va, marker="o", markersize=2, linewidth=1.0, color="tab:orange", label="Valid")

        ax.set_title(f"Context Window Size = {ns}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, linestyle="--", linewidth=0.4)

        # Keep legend small
        ax.legend(fontsize=8, loc="best")

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.savefig(args.out, dpi=200)
    print(f"Saved: {args.out}")
    print("Plotted files:")
    for ns, p in items:
        print(f"  Context Window Size = {ns}: {p}")


if __name__ == "__main__":
    main()

