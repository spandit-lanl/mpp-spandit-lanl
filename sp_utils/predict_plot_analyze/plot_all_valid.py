#!/usr/bin/env python3
"""
Plot validation loss vs epoch for all runs in combined_losses.csv.

Usage:
  python plot_all_valid.py --input combined_losses.csv --phase all --output all_valid.png
"""

import argparse
import matplotlib.pyplot as plt

from loss_utils import load_combined_losses


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        default="combined_losses.csv",
        help="Path to combined losses CSV (default: combined_losses.csv)",
    )
    p.add_argument(
        "--phase",
        choices=["all", "pretrain", "finetune"],
        default="all",
        help="Filter runs by phase (default: all)",
    )
    p.add_argument(
        "--output",
        default="all_valid.png",
        help="Output PNG filename (default: all_valid.png)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    epochs, runs = load_combined_losses(args.input)

    plt.figure()
    for tag, info in sorted(runs.items(), key=lambda kv: (kv[1]["phase"], kv[1]["nstep"])):
        if args.phase != "all" and info["phase"] != args.phase:
            continue

        xs = []
        ys = []
        for e, v in zip(epochs, info["valid"]):
            if v is not None:
                xs.append(e)
                ys.append(v)

        if not xs:
            continue

        label = f"{info['phase']}, n={info['nstep']}"
        plt.plot(xs, ys, label=label)

    plt.xlabel("Epoch")
    plt.ylabel("Validation loss")
    plt.title(f"Validation loss vs Epoch ({args.phase})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()

