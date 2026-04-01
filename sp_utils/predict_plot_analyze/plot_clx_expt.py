#!/usr/bin/env python3
import re
import glob
import math
from pathlib import Path

import matplotlib.pyplot as plt

# Matches lines like:
# Epoch: 30. Train loss: tensor([0.9123], device='cuda:0'). Valid loss: 0.9353660345077515
LINE_RE = re.compile(
    r"Epoch:\s*(\d+)\.\s*Train loss:\s*tensor\(\[([0-9eE+\-\.]+)\].*?\)\.\s*Valid loss:\s*([0-9eE+\-\.]+)"
)

# Matches filenames like:
#   clx_expt_ns-01_var-1.log
#   clx_expt_ns-02_var-8.log
#   clx_expt_ns-04_var-3.log
FNAME_RE = re.compile(r"clx_expt_ns-(\d+)_var-(\d+)\.log$")

def parse_log(path: str):
    epochs, train_losses, valid_losses = [], [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = LINE_RE.search(line)
            if not m:
                continue
            epochs.append(int(m.group(1)))
            train_losses.append(float(m.group(2)))
            valid_losses.append(float(m.group(3)))

    # Sort by epoch in case logs are out of order
    if epochs:
        order = sorted(range(len(epochs)), key=lambda i: epochs[i])
        epochs = [epochs[i] for i in order]
        train_losses = [train_losses[i] for i in order]
        valid_losses = [valid_losses[i] for i in order]

    return epochs, train_losses, valid_losses


def group_and_sort_logs(files):
    """Group log files by ns (e.g., 1, 2, 4) and sort within each group by var."""
    groups = {}
    for fp in files:
        name = Path(fp).name
        m = FNAME_RE.search(name)
        if not m:
            # Ignore other clx_expt_*.log files that don't match the ns/var pattern.
            continue
        ns = int(m.group(1))
        var = int(m.group(2))
        groups.setdefault(ns, []).append((var, fp))

    # Sort by var within each ns
    out = {}
    for ns, items in groups.items():
        out[ns] = [fp for (var, fp) in sorted(items, key=lambda t: t[0])]
    return out


def plot_grid(files, out_png, title):
    """Plot up to 8 logs in a 2x4 grid."""
    nrows, ncols = 2, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 8), sharex=False, sharey=False)
    axes = axes.ravel()

    for ax_i, ax in enumerate(axes):
        if ax_i >= len(files):
            ax.axis("off")
            continue

        fp = files[ax_i]
        epochs, train_l, valid_l = parse_log(fp)

        ax.set_title(Path(fp).name)

        if not epochs:
            ax.text(0.5, 0.5, "No matching lines found", ha="center", va="center")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            continue

        ax.plot(epochs, train_l, label="train")
        ax.plot(epochs, valid_l, label="valid")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=200)
    print(f"Saved: {out_png}")
    plt.close(fig)

def main():
    # Pick up your files; adjust pattern if needed
    files = sorted(glob.glob("clx_expt_*.log"))
    if not files:
        raise SystemExit("No files matched pattern: clx_expt_*.log (run from the directory with logs)")

    groups = group_and_sort_logs(files)
    if not groups:
        raise SystemExit(
            "No files matched expected pattern clx_expt_ns-##_var-#.log (found clx_expt_*.log but none parseable)"
        )

    # For now you'll typically have ns-01 and ns-02.
    # If/when ns-04 exists, this will automatically produce a 3rd grid.
    for ns in sorted(groups.keys()):
        out = f"clx_loss_grid_ns-{ns:02d}_2x4.png"
        title = f"Training vs Validation Loss (clx_expt ns-{ns:02d})"
        plot_grid(groups[ns], out, title)

if __name__ == "__main__":
    main()

