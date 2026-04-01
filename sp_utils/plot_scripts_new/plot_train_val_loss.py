#!/usr/bin/env python3
"""
Plot training and validation losses from log files.

Expected input filenames look like:
    out_cyl-cx241203_full_ns01.log
    out_pli-lsc240420_full_ns01.log

This script:
- scans an input directory for matching log files
- groups them by dataset + experiment ID + image size
- parses only lines containing "Valid loss"
- extracts epoch, train loss, valid loss
- plots train loss in blue and valid loss in red
- uses one global y-axis range across all plots in a figure
- uses at most 8 x-tick labels per subplot
- saves one PDF per group, e.g.:
    plot_cyl-cx241203_full.pdf
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt


FILENAME_RE = re.compile(
    r"^out_(?P<dataset>cyl|pli)-(?P<experiment_id>[^_]+)_(?P<image_size>full|half)_ns(?P<context>\d{2})\.log$"
)

LOSS_LINE_RE = re.compile(
    r"Epoch:\s*(?P<epoch>\d+)\.\s*"
    r"Train loss:\s*tensor\(\[(?P<train>[0-9eE+\-.]+)\].*?"
    r"Valid loss:\s*(?P<valid>[0-9eE+\-.]+)"
)


@dataclass(frozen=True)
class LogFileInfo:
    path: Path
    dataset: str
    experiment_id: str
    image_size: str
    context_window: int


@dataclass
class LossSeries:
    epochs: List[int]
    train_losses: List[float]
    valid_losses: List[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot train/valid loss curves from matching log files."
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        default=".",
        help="Directory containing log files. Default: current working directory.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=".",
        help="Directory where PDF plots will be saved. Default: current working directory.",
    )
    return parser.parse_args()


def discover_log_files(input_dir: Path) -> List[LogFileInfo]:
    files: List[LogFileInfo] = []

    for path in sorted(input_dir.iterdir()):
        if not path.is_file():
            continue

        match = FILENAME_RE.match(path.name)
        if not match:
            continue

        files.append(
            LogFileInfo(
                path=path,
                dataset=match.group("dataset"),
                experiment_id=match.group("experiment_id"),
                image_size=match.group("image_size"),
                context_window=int(match.group("context")),
            )
        )

    return files


def parse_loss_lines(log_path: Path) -> Optional[LossSeries]:
    epochs: List[int] = []
    train_losses: List[float] = []
    valid_losses: List[float] = []

    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if "Valid loss" not in line:
                continue

            match = LOSS_LINE_RE.search(line)
            if not match:
                continue

            epoch = int(match.group("epoch"))
            train_loss = round(float(match.group("train")), 4)
            valid_loss = round(float(match.group("valid")), 4)

            epochs.append(epoch)
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)

    if not epochs:
        return None

    return LossSeries(
        epochs=epochs,
        train_losses=train_losses,
        valid_losses=valid_losses,
    )


def group_log_files(
    files: Sequence[LogFileInfo],
) -> Dict[Tuple[str, str, str], List[LogFileInfo]]:
    groups: Dict[Tuple[str, str, str], List[LogFileInfo]] = {}

    for info in files:
        key = (info.dataset, info.experiment_id, info.image_size)
        groups.setdefault(key, []).append(info)

    for key in groups:
        groups[key].sort(key=lambda item: item.context_window)

    return groups


def compute_global_y_limits(series_list: Sequence[LossSeries]) -> Tuple[float, float]:
    all_values: List[float] = []

    for series in series_list:
        all_values.extend(series.train_losses)
        all_values.extend(series.valid_losses)

    y_min = min(all_values)
    y_max = max(all_values)

    if math.isclose(y_min, y_max):
        pad = 0.05 if y_min == 0 else abs(y_min) * 0.05
        return y_min - pad, y_max + pad

    pad = 0.05 * (y_max - y_min)
    return y_min - pad, y_max + pad


def choose_grid(n_plots: int) -> Tuple[int, int]:
    if n_plots <= 1:
        return 1, 1
    if n_plots <= 4:
        return 1, n_plots
    rows = math.ceil(n_plots / 4)
    return rows, 4


def select_xticks(epochs: Sequence[int], max_ticks: int = 8) -> List[int]:
    if not epochs:
        return []

    if len(epochs) <= max_ticks:
        return list(epochs)

    start = epochs[0]
    end = epochs[-1]

    if start == end:
        return [start]

    ticks = []
    for i in range(max_ticks):
        value = round(start + i * (end - start) / (max_ticks - 1))
        ticks.append(int(value))

    deduped = []
    seen = set()
    for tick in ticks:
        if tick not in seen:
            deduped.append(tick)
            seen.add(tick)

    return deduped


def make_suptitle(dataset: str, experiment_id: str, image_size: str) -> str:
    dataset_label = "Cylex" if dataset == "cyl" else "Perturbed Layer Interface"
    return f"{dataset_label} | {experiment_id} | {image_size}"


def plot_group(
    group_key: Tuple[str, str, str],
    group_files: Sequence[LogFileInfo],
    output_dir: Path,
) -> Optional[Path]:
    dataset, experiment_id, image_size = group_key

    parsed: List[Tuple[LogFileInfo, LossSeries]] = []
    for info in group_files:
        series = parse_loss_lines(info.path)
        if series is not None:
            parsed.append((info, series))

    if not parsed:
        return None

    n_plots = len(parsed)
    nrows, ncols = choose_grid(n_plots)

    series_list = [series for _, series in parsed]
    y_min, y_max = compute_global_y_limits(series_list)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5 * ncols, 3.8 * nrows),
        squeeze=False,
    )

    flat_axes = list(axes.flat)

    for idx, (info, series) in enumerate(parsed):
        ax = flat_axes[idx]

        ax.plot(series.epochs, series.train_losses, color="blue", label="Train")
        ax.plot(series.epochs, series.valid_losses, color="red", label="Valid")
        ax.set_title(f"ns{info.context_window:02d}")
        ax.set_ylim(y_min, y_max)

        xticks = select_xticks(series.epochs, max_ticks=8)
        ax.set_xticks(xticks)

        row = idx // ncols
        col = idx % ncols

        is_bottom_row = row == (nrows - 1)
        is_leftmost = col == 0

        if nrows == 1 or is_bottom_row:
            ax.set_xlabel("Epoch")
        else:
            ax.tick_params(axis="x", labelbottom=False)

        if ncols == 1 or is_leftmost:
            ax.set_ylabel("Loss")
        else:
            ax.tick_params(axis="y", labelleft=False)

        ax.grid(True, alpha=0.3)

    for idx in range(n_plots, len(flat_axes)):
        flat_axes[idx].set_visible(False)

    # One legend for the whole figure.
    handles, labels = flat_axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")

    fig.suptitle(make_suptitle(dataset, experiment_id, image_size), fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    output_path = output_dir / f"plot_{dataset}-{experiment_id}_{image_size}.pdf"
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    return output_path


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    files = discover_log_files(input_dir)
    groups = group_log_files(files)

    if not groups:
        print(f"No matching log files found in: {input_dir}")
        return

    saved_paths: List[Path] = []

    for group_key, group_files in sorted(groups.items()):
        saved = plot_group(group_key, group_files, output_dir)
        if saved is not None:
            saved_paths.append(saved)

    if not saved_paths:
        print("No plots were created. Matching files were found, but no valid loss lines were parsed.")
        return

    print("Saved plot files:")
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()
