#!/usr/bin/env python3
"""
plot_train_val_losses_grid.py

Plot Train/Valid loss vs Epoch for pretrain/finetune logs across context window sizes.

Features:
- Select mode: --mode pretrain|finetune (filters filenames by prefix "LOSS_<mode>_" by default)
- Select parity: --parity odd|even|all
  * odd/even -> 2x4 grid (8 panels)
  * all      -> 4x4 grid (16 panels)
- Uniform Y-axis scale across ALL panels (global min/max after filters)
- Epoch cutoff: --max_epoch (default 460)
- Loss threshold filter: --max_loss (default 1e9; set e.g. 0.5)
  Filters out points where train OR valid loss exceeds threshold
- Output naming:
  default: "<mode>_<parity>_grid.png"
  override with: --out <filename>

Input formats supported per line:
- "Train loss: tensor([0.5246], device='cuda:0'). Valid loss: 0.6605"
- Optionally prefixed: "Epoch 001: Train loss: ... Valid loss: ..."
- CSV files are treated as plain text (one record per line) with the same pattern.

Typical usage:
  python plot_train_val_losses_grid.py --mode pretrain --parity odd
  python plot_train_val_losses_grid.py --mode finetune --parity all --max_loss 0.8 --max_epoch 500
  python plot_train_val_losses_grid.py --mode finetune --parity even --out my_custom.png
"""

import argparse
import glob
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


# -------------------------
# Regex parsing
# -------------------------
# Optional "Epoch XXX:" prefix
EPOCH_PREFIX_RE = re.compile(r"\bEpoch\b\s*(\d+)\s*:\s*", re.IGNORECASE)

# Optional explicit "Epoch: N" variant (rare)
EPOCH_INLINE_RE = re.compile(r"\bEpoch\b[:\s]+(\d+)\b", re.IGNORECASE)

# Train loss: tensor([0.5246], device='cuda:0')
TRAIN_RE = re.compile(
    r"Train\s+loss:\s*tensor\(\[\s*([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)\s*\]",    re.IGNORECASE,
)

# Valid loss: 0.6605148911476135
VALID_RE = re.compile(
    r"Valid\s+loss:\s*([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)",    re.IGNORECASE,
)

# Extract context window size from filename
# Supports:
#   ...nsteps_02...  (txt logs)
#   LOSS_pretrain_04.csv / LOSS_finetune_16.csv
NSTEPS_RE_1 = re.compile(r"nsteps[_-](\d+)", re.IGNORECASE)
NSTEPS_RE_2 = re.compile(r"(?:^|[_-])(pretrain|finetune)[_-](\d+)\.(?:txt|csv)$", re.IGNORECASE)
NSTEPS_RE_3 = re.compile(r"LOSS_(?:pretrain|finetune)_(\d+)\.(?:txt|csv)$", re.IGNORECASE)


def extract_nsteps(path: str) -> Optional[int]:
    base = os.path.basename(path)

    m = NSTEPS_RE_1.search(base)
    if m:
        return int(m.group(1))

    m = NSTEPS_RE_3.search(base)
    if m:
        return int(m.group(1))

    m = NSTEPS_RE_2.search(base)
    if m:
        return int(m.group(2))

    return None


def parse_epoch_from_line(line: str) -> Optional[int]:
    m = EPOCH_PREFIX_RE.search(line)
    if m:
        return int(m.group(1))
    m = EPOCH_INLINE_RE.search(line)
    if m:
        return int(m.group(1))
    return None


def parse_log(path: str, max_loss: float, max_epoch: int) -> Tuple[List[int], List[float], List[float]]:
    """
    Parse a log file and return epochs, train_losses, valid_losses after filtering.
    - If epoch isn't present in lines, uses line counter as epoch.
    - Drops points where ep > max_epoch.
    - Drops points where train OR valid loss > max_loss.
    """
    epochs: List[int] = []
    train_losses: List[float] = []
    valid_losses: List[float] = []

    fallback_ep = 0

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            tr_m = TRAIN_RE.search(line)
            va_m = VALID_RE.search(line)
            if not (tr_m and va_m):
                continue

            tr = float(tr_m.group(1))
            va = float(va_m.group(1))

            if tr > max_loss or va > max_loss:
                continue

            ep = parse_epoch_from_line(line)
            if ep is None:
                fallback_ep += 1
                ep = fallback_ep

            if ep > max_epoch:
                continue

            epochs.append(ep)
            train_losses.append(tr)
            valid_losses.append(va)

    return epochs, train_losses, valid_losses


def build_file_list(mode: str, pattern: Optional[str]) -> List[str]:
    """
    Get files to plot.
    If pattern is provided, use it.
    Else default to LOSS_<mode>_*.csv and LOSS_<mode>_*.txt.
    """
    if pattern:
        return sorted(glob.glob(pattern))

    pat_csv = f"out_{mode}_*.csv"
    pat_txt = f"out_{mode}_*.txt"
    return sorted(set(glob.glob(pat_csv) + glob.glob(pat_txt)))


def choose_grid(parity: str) -> Tuple[int, int, int]:
    """Return (nrows, ncols, limit)."""
    if parity in ("odd", "even"):
        return 2, 4, 8
    if parity == "all":
        return 4, 4, 16
    raise ValueError(f"Unknown parity: {parity}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pretrain", "finetune"], required=True,
                    help="Which logs to plot.")
    ap.add_argument("--parity", choices=["odd", "even", "all"], required=True,
                    help="Which context window sizes to include.")
    ap.add_argument("--pattern", default=None,
                    help=("Optional glob override. If not provided, defaults to "

                          "LOSS_<mode>_*.csv and LOSS_<mode>_*.txt in current directory."))
    ap.add_argument("--max_loss", type=float, default=1e9,
                    help="Ignore points where train OR valid loss exceeds this threshold (default: no filter).")
    ap.add_argument("--max_epoch", type=int, default=460,
                    help="Only plot points with epoch <= max_epoch (default: 460).")
    ap.add_argument("--out", default=None,
                    help=("Optional output PNG filename override. If omitted, defaults to <mode>_<parity>_grid.png"))
    ap.add_argument("--title", default=None,
                    help="Optional figure title override.")
    args = ap.parse_args()

    out_name = args.out or f"{args.mode}_{args.parity}_grid.png"
    out_path = Path(out_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    files = build_file_list(args.mode, args.pattern)
    if not files:
        raise SystemExit(f"No files matched. mode={args.mode}, pattern={args.pattern!r}")

    items: List[Tuple[int, str]] = []
    skipped_no_nsteps = 0
    for p in files:
        ns = extract_nsteps(p)
        if ns is None:
            skipped_no_nsteps += 1
            continue

        if args.parity == "odd" and (ns % 2 != 1):
            continue
        if args.parity == "even" and (ns % 2 != 0):
            continue

        items.append((ns, p))

    if not items:
        raise SystemExit(
            f"No files passed filters. mode={args.mode}, parity={args.parity}. "

            f"Matched files={len(files)}, skipped(no nsteps)={skipped_no_nsteps}"
        )

    items.sort(key=lambda x: x[0])
    nrows, ncols, limit = choose_grid(args.parity)
    items = items[:limit]

    parsed_cache: Dict[int, Tuple[List[int], List[float], List[float]]] = {}
    global_min = float("inf")
    global_max = float("-inf")

    for ns, path in items:
        epochs, tr, va = parse_log(path, max_loss=args.max_loss, max_epoch=args.max_epoch)
        parsed_cache[ns] = (epochs, tr, va)
        if epochs:
            global_min = min(global_min, min(tr), min(va))
            global_max = max(global_max, max(tr), max(va))

    if global_min == float("inf"):
        global_min, global_max = 0.0, 1.0

    pad = 0.05 * (global_max - global_min) if global_max > global_min else 0.01
    global_min = max(0.0, global_min - pad)
    global_max = global_max + pad

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 3.6*nrows))
    # Flatten axes
    try:
        axes_list = list(axes.ravel())
    except Exception:
        axes_list = [axes]

    default_title = f"Train/Valid Loss vs Epoch ({args.mode.capitalize()} Logs)"
    fig.suptitle(args.title or default_title)

    for idx, (ns, _path) in enumerate(items):
        ax = axes_list[idx]
        epochs, tr, va = parsed_cache[ns]

        ax.set_title(f"Context Window Size = {ns}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, linestyle="--", linewidth=0.5)

        if not epochs:
            ax.text(0.5, 0.5, "No points\n(after filters)", ha="center", va="center")
            ax.set_ylim(global_min, global_max)
            continue

        ax.plot(epochs, tr, marker="o", markersize=2, linewidth=1.0, label="Train")
        ax.plot(epochs, va, marker="o", markersize=2, linewidth=1.0, label="Valid")
        ax.set_ylim(global_min, global_max)
        ax.legend(fontsize=8, loc="best")

    for j in range(len(items), nrows * ncols):
        axes_list[j].axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    fig.savefig(str(out_path), dpi=200)
    plt.close(fig)

    print(f"Saved: {out_path.resolve()}")
    print("Plotted files:")
    for ns, p in items:
        print(f"  Context Window Size={ns:02d}  {p}")


if __name__ == "__main__":
    main()
