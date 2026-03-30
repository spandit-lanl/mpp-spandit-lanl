#!/usr/bin/env python3
import argparse
import csv
import math
import os
import re
from typing import Dict, List, Optional, Tuple, Set

import matplotlib.pyplot as plt
import numpy as np

# Combined column scheme:
#   <phase>-<phasedata>_nsNN_train_loss
#   <phase>-<phasedata>_nsNN_valid_loss
COL_RE = re.compile(
    r"^(?P<phase>pretrain|dtrain|finetune)-(?P<phasedata>[A-Za-z0-9]+)_ns(?P<ns>\d{2})_(?P<kind>train_loss|valid_loss)$"
)

VALID_PHASEDATA = {
    "pretrain": {"pdebenchfull", "pdebenchpart"},
    "dtrain": {"LSC", "CLX"},
    "finetune": {"LSC", "CLX"},
}

PHASE_ORDER = {"pretrain": 0, "dtrain": 1, "finetune": 2}


def safe_float(s: str) -> Optional[float]:
    try:
        return float(s)
    except Exception:
        return None

def eight_ticks(max_epoch: int) -> List[int]:
    """
    Return up to 8 increasing tick positions (excluding 0), ending at max_epoch.
    Guaranteed to terminate for all max_epoch >= 0.
    """
    if max_epoch <= 0:
        return [1, 2, 3, 4, 5, 6, 7, 8]

    # If the run is short, just use 1..max_epoch
    if max_epoch <= 8:
        return list(range(1, max_epoch + 1))

    step = max(1, max_epoch // 8)
    ticks = [step * i for i in range(1, 9)]
    ticks[-1] = max_epoch

    ticks = sorted(set(ticks))

    # Pad (should be rare for max_epoch>8, but keep it safe)
    i = 1
    while len(ticks) < 8 and i <= max_epoch:
        if i not in ticks:
            ticks.append(i)
            ticks = sorted(set(ticks))
        i += 1

    # Downsample if somehow too many
    if len(ticks) > 8:
        idxs = [round(i * (len(ticks) - 1) / 7) for i in range(8)]
        ticks = [ticks[i] for i in idxs]
        ticks[-1] = max_epoch

    return ticks

'''
def eight_ticks(max_epoch: int) -> List[int]:
    """Exactly 8 ticks excluding 0, evenly spaced, ending at max_epoch."""
    if max_epoch <= 0:
        return [1, 2, 3, 4, 5, 6, 7, 8]

    step = max(1, max_epoch // 8)
    ticks = [step * i for i in range(1, 9)]
    ticks[-1] = max_epoch

    ticks = sorted(set(ticks))
    while len(ticks) < 8:
        cand = int(round(max_epoch * (len(ticks) + 1) / 9))
        if cand > 0:
            ticks.append(cand)
        ticks = sorted(set(ticks))
        if len(ticks) < 8 and ticks[-1] < max_epoch:
            ticks.append(min(max_epoch, ticks[-1] + 1))
            ticks = sorted(set(ticks))

    if len(ticks) > 8:
        idxs = [round(i * (len(ticks) - 1) / 7) for i in range(8)]
        ticks = [ticks[i] for i in idxs]
        ticks[-1] = max_epoch

    return ticks
'''

def parse_args():
    p = argparse.ArgumentParser(description="Plot loss curves in grids by phase-phasedata.")
    p.add_argument("--input", default="combined_losses.csv")
    p.add_argument("--outdir", default=".", help="Directory to write PDFs.")
    p.add_argument("--ncols", type=int, default=4)
    p.add_argument("--max_plots", type=int, default=64, help="Max panels per PDF.")
    return p.parse_args()


def discover_groups(header: List[str]) -> Dict[Tuple[str, str], Set[str]]:
    """
    Returns mapping: (phase, phasedata) -> set of ns strings present,
    only when BOTH train and valid columns exist for that ns.
    """
    seen: Dict[Tuple[str, str, str], Set[str]] = {}  # (phase, phasedata, ns) -> kinds

    for col in header:
        m = COL_RE.match(col)
        if not m:
            continue
        phase = m.group("phase")
        phasedata = m.group("phasedata")
        ns = m.group("ns")
        kind = m.group("kind")

        allowed = VALID_PHASEDATA.get(phase)
        if allowed is None or phasedata not in allowed:
            continue

        key = (phase, phasedata, ns)
        seen.setdefault(key, set()).add(kind)

    groups: Dict[Tuple[str, str], Set[str]] = {}
    for (phase, phasedata, ns), kinds in seen.items():
        if "train_loss" in kinds and "valid_loss" in kinds:
            groups.setdefault((phase, phasedata), set()).add(ns)

    return groups


def group_max_epoch(rows: List[dict], epochs: np.ndarray, cols: List[str]) -> int:
    """
    Compute max epoch for a group based on the last epoch where ANY of the
    given columns has a numeric value.
    """
    max_ep = 0
    for r in rows:
        ep = int(r["epoch"])
        for c in cols:
            s = (r.get(c) or "").strip()
            if not s:
                continue
            v = safe_float(s)
            #if v is None or math.isnan(v):
            if v is None or not math.isfinite(v):
                continue

            max_ep = max(max_ep, ep)
            break

    if max_ep <= 0 and epochs.size:
        max_ep = int(np.max(epochs))
    return max_ep


def main():
    args = parse_args()
    if not os.path.exists(args.input):
        raise SystemExit(f"Input file {args.input!r} not found.")

    with open(args.input, newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        rows = list(reader)

    if "epoch" not in header:
        raise SystemExit("Expected 'epoch' column in combined CSV.")

    epochs = np.array([int(r["epoch"]) for r in rows], dtype=int)

    groups = discover_groups(header)
    if not groups:
        raise SystemExit("No matching <phase>-<phasedata>_nsNN_(train|valid)_loss columns found.")

    os.makedirs(args.outdir, exist_ok=True)

    # Sort groups: phase order, then phasedata
    group_keys = sorted(groups.keys(), key=lambda k: (PHASE_ORDER.get(k[0], 999), k[1]))

    for (phase, phasedata) in group_keys:
        ns_list = sorted(groups[(phase, phasedata)], key=lambda s: int(s))
        ns_list = ns_list[: args.max_plots]

        if not ns_list:
            continue

        # Per-group X range (fix for long pretrain vs short finetune)
        xcols = []
        for ns in ns_list:
            xcols.append(f"{phase}-{phasedata}_ns{ns}_train_loss")
            xcols.append(f"{phase}-{phasedata}_ns{ns}_valid_loss")

        max_epoch = group_max_epoch(rows, epochs, xcols)
        xticks = eight_ticks(max_epoch)

        ncols = max(1, args.ncols)
        nrows = math.ceil(len(ns_list) / ncols)

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(4.8 * ncols, 3.6 * nrows),
            squeeze=False,
            sharex=True,
            sharey=True,
        )
        axes = axes.ravel()
        for ax in axes:
            ax.set_visible(False)

        # Global Y range across this group
        y_min = float("inf")
        y_max = float("-inf")

        def update_range(colname: str):
            nonlocal y_min, y_max
            for r in rows:
                s = (r.get(colname) or "").strip()
                if not s:
                    continue
                v = safe_float(s)
                if v is None:
                    continue
                y_min = min(y_min, v)
                y_max = max(y_max, v)

        for ns in ns_list:
            tcol = f"{phase}-{phasedata}_ns{ns}_train_loss"
            vcol = f"{phase}-{phasedata}_ns{ns}_valid_loss"
            if tcol in header:
                update_range(tcol)
            if vcol in header:
                update_range(vcol)

        if not (math.isfinite(y_min) and math.isfinite(y_max)):
            raise SystemExit(f"Could not compute Y range for {phase}-{phasedata} (no numeric values).")

        pad = 0.05 * (y_max - y_min) if y_max > y_min else 0.1
        y_lo = max(0.0, y_min - pad)
        y_hi = y_max + pad

        # Suptitle must include <phase>-<phasedata>
        fig.suptitle(f"{phase}-{phasedata}", fontsize=22, fontweight="bold", y=0.995)

        for i, ns in enumerate(ns_list):
            ax = axes[i]
            ax.set_visible(True)

            tcol = f"{phase}-{phasedata}_ns{ns}_train_loss"
            vcol = f"{phase}-{phasedata}_ns{ns}_valid_loss"

            train_y = np.array(
                [
                    safe_float((r.get(tcol) or "").strip()) if (r.get(tcol) or "").strip() else np.nan
                    for r in rows
                ],
                dtype=float,
            )
            valid_y = np.array(
                [
                    safe_float((r.get(vcol) or "").strip()) if (r.get(vcol) or "").strip() else np.nan
                    for r in rows
                ],
                dtype=float,
            )

            ax.plot(epochs, train_y, label="Train")
            ax.plot(epochs, valid_y, label="Valid")

            # Individual plot titles must include ns
            ax.set_title(f"ns={ns}", fontsize=11)

            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, max_epoch)
            ax.set_xticks(xticks)
            ax.set_ylim(y_lo, y_hi)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend(loc="best", fontsize=9)

        for j in range(len(ns_list), nrows * ncols):
            axes[j].axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.965])

        outpath = os.path.join(args.outdir, f"{phase}-{phasedata}_grid.pdf")
        plt.savefig(outpath)
        plt.close(fig)
        print(f"Saved {outpath} (panels={len(ns_list)}, grid={nrows}x{ncols}, x_max={max_epoch})")


if __name__ == "__main__":
    main()

