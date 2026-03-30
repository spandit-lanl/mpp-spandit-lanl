#!/usr/bin/env python3
"""
End-to-end loss pipeline (extract -> combine -> plot) replacing:
  - 01_extract_losses.bash
  - 02_combine_losses.py
  - 04_plot_grid_losses_uniq_EPOCH_SCALES.py

Typical usage (run from your PLOTS_FINAL dir):
  python3 00_extract_combine_plot.py \
    --log-dir /users/spandit/projects/artimis/mpp/mpp-spandit-npz-clx/final \
    --log-glob "out_final_L_*.log"

Outputs (in --outdir, default "."):
  - loss_final_L_<phase>-<phasedata>_lr-..._ns-<NN>[_var-<V>].csv  (one per log)
  - combined_losses.csv
  - combined_losses_var-3.csv
  - combined_losses_var-6.csv
  - <phase>-<phasedata>[_var-<V>]_grid.pdf                         (one per group)
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import glob as _glob
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Extraction (from logs -> per-run loss CSVs)
# -----------------------------------------------------------------------------

# Example line:
# Epoch: 30. Train loss: tensor([0.9123], device='cuda:0'). Valid loss: 0.9353660345077515
# or
# Epoch: 30. Train loss: 0.9123. Valid loss: 0.9353660345077515
LINE_RE = re.compile(
    r"Epoch:\s*(?P<epoch>\d+)\.\s*"
    r"Train loss:\s*(?:tensor\(\[(?P<train_t>[0-9eE+\-\.]+)\].*?\)|(?P<train_f>[0-9eE+\-\.]+))\.\s*"
    r"Valid loss:\s*(?P<valid>[0-9eE+\-\.]+)"
)

# Logs look like:
# out_final_L_finetune-CLX_lr-X_opt-adan_wd-3_ns-01.log
# out_final_L_finetune-CLX_lr-X_opt-adan_wd-3_ns-01_var-3.log
# out_final_L_finetune-CLX_lr-X_opt-adan_wd-3_ns-01_var-6.log
LOG_RE = re.compile(
    r"^out_final_(?P<variant>L|B)_"
    r"(?P<phase>pretrain|dtrain|finetune)-(?P<phasedata>[A-Za-z0-9]+)"
    r"_lr-(?P<lr>[A-Za-z0-9]+)"
    r"_opt-(?P<opt>[A-Za-z0-9]+)"
    r"_wd-(?P<wd>[A-Za-z0-9]+)"
    r"_ns-(?P<ns>[0-9]{2})"
    r"(?:_var-(?P<var>[0-9]+))?"
    r"\.log$"
)

VALID_PHASEDATA: Dict[str, Set[str]] = {
    "pretrain": {"pdebenchfull", "pdebenchpart"},
    "dtrain": {"LSC", "CLX"},
    "finetune": {"LSC", "CLX"},
}


@dataclass(frozen=True)
class RunId:
    variant: str   # "L" or "B"
    phase: str     # pretrain|dtrain|finetune
    phasedata: str # e.g. pdebenchfull, CLX
    lr: str
    opt: str
    wd: str
    ns: str        # "01", "02", ...
    var: Optional[str] = None  # e.g. "3", "6"

    @property
    def loss_csv_name(self) -> str:
        suffix = f"_var-{self.var}" if self.var is not None else ""
        return (
            f"loss_final_{self.variant}_{self.phase}-{self.phasedata}"
            f"_lr-{self.lr}_opt-{self.opt}_wd-{self.wd}_ns-{self.ns}{suffix}.csv"
        )

    @property
    def tag(self) -> str:
        suffix = f"_var{self.var}" if self.var is not None else ""
        return f"{self.phase}-{self.phasedata}_ns{int(self.ns):02d}{suffix}"


def parse_run_id_from_log_name(name: str) -> Optional[RunId]:
    m = LOG_RE.match(name)
    if not m:
        return None
    d = m.groupdict()
    phase = d["phase"]
    phasedata = d["phasedata"]
    if phasedata not in VALID_PHASEDATA.get(phase, set()):
        return None
    return RunId(
        variant=d["variant"],
        phase=phase,
        phasedata=phasedata,
        lr=d["lr"],
        opt=d["opt"],
        wd=d["wd"],
        ns=d["ns"],
        var=d.get("var"),
    )


def iter_logs(log_dir: Path, log_glob: str) -> List[Tuple[RunId, Path]]:
    out: List[Tuple[RunId, Path]] = []
    for p in sorted(log_dir.glob(log_glob)):
        rid = parse_run_id_from_log_name(p.name)
        if rid is None:
            continue
        out.append((rid, p))
    return out


def extract_losses_from_log(path: Path) -> List[Tuple[int, float, float]]:
    rows: List[Tuple[int, float, float]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = LINE_RE.search(line)
            if not m:
                continue
            epoch = int(m.group("epoch"))
            train_s = m.group("train_t") or m.group("train_f")
            if train_s is None:
                continue
            train = float(train_s)
            valid = float(m.group("valid"))
            rows.append((epoch, train, valid))

    rows.sort(key=lambda t: t[0])
    if not rows:
        return rows
    dedup: Dict[int, Tuple[int, float, float]] = {}
    for r in rows:
        dedup[r[0]] = r
    return [dedup[k] for k in sorted(dedup.keys())]


def write_loss_csv(rows: Sequence[Tuple[int, float, float]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "valid_loss"])
        for epoch, train, valid in rows:
            w.writerow([epoch, f"{train:.6g}", f"{valid:.6g}"])


def extract_all_losses(log_dir: Path, log_glob: str, out_dir: Path) -> List[Path]:
    logs = iter_logs(log_dir, log_glob)
    if not logs:
        raise FileNotFoundError(
            f"No parseable logs found in {log_dir} matching {log_glob} with expected name "
            f"out_final_(L|B)_(pretrain|dtrain|finetune)-<PHASEDATA>_lr-..._ns-<NN>.log"
        )

    written: List[Path] = []
    for rid, log_path in logs:
        rows = extract_losses_from_log(log_path)
        out_path = out_dir / rid.loss_csv_name
        print(f"writing {out_path.name}")
        # Always write (header-only is OK)
        write_loss_csv(rows, out_path)
        written.append(out_path)
    return written


# -----------------------------------------------------------------------------
# Combine (per-run loss CSVs -> combined CSVs)
# -----------------------------------------------------------------------------

LOSS_CSV_RE = re.compile(
    r"^loss_final_(?P<variant>L|B)_"
    r"(?P<phase>pretrain|dtrain|finetune)-(?P<phasedata>[A-Za-z0-9]+)"
    r"_lr-(?P<lr>[A-Za-z0-9]+)"
    r"_opt-(?P<opt>[A-Za-z0-9]+)"
    r"_wd-(?P<wd>[A-Za-z0-9]+)"
    r"_ns-(?P<ns>[0-9]{2})"
    r"(?:_var-(?P<var>[0-9]+))?"
    r"\.csv$"
)

PHASE_SORT_KEY = {"pretrain": 0, "dtrain": 1, "finetune": 2}


def parse_run_id_from_loss_csv_name(name: str) -> Optional[Tuple[Tuple, RunId]]:
    m = LOSS_CSV_RE.match(name)
    if not m:
        return None
    d = m.groupdict()
    phase = d["phase"]
    phasedata = d["phasedata"]
    if phasedata not in VALID_PHASEDATA.get(phase, set()):
        return None

    rid = RunId(
        variant=d["variant"],
        phase=phase,
        phasedata=phasedata,
        lr=d["lr"],
        opt=d["opt"],
        wd=d["wd"],
        ns=d["ns"],
        var=d.get("var"),
    )
    sort_key = (
        PHASE_SORT_KEY[phase],
        phasedata,
        rid.variant,
        int(rid.ns),
        int(rid.var) if rid.var is not None else -1,
        rid.lr,
        rid.opt,
        rid.wd,
    )
    return sort_key, rid


def combine_losses(pattern: str, output_csv: Path, *, var_filter: Optional[str] = None) -> None:
    files = sorted([Path(p) for p in _glob.glob(pattern, recursive=True)])
    if not files:
        raise SystemExit(f"No files match pattern {pattern!r}")

    all_epochs: Set[int] = set()
    runs: List[Dict[str, object]] = []
    skipped: List[str] = []

    for path in files:
        parsed = parse_run_id_from_loss_csv_name(path.name)
        if not parsed:
            skipped.append(path.name)
            continue
        sort_key, rid = parsed

        # Filter base vs var runs
        if var_filter is None:
            if rid.var is not None:
                continue
        else:
            if rid.var != var_filter:
                continue

        data: Dict[int, Tuple[str, str]] = {}
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or "epoch" not in reader.fieldnames:
                skipped.append(path.name)
                continue
            for row in reader:
                try:
                    epoch = int(row["epoch"])
                except Exception:
                    continue
                train = (row.get("train_loss") or "").strip()
                valid = (row.get("valid_loss") or "").strip()
                data[epoch] = (train, valid)
                all_epochs.add(epoch)

        runs.append({"sort_key": sort_key, "rid": rid, "data": data})

    if not runs:
        raise SystemExit("No valid loss_final_*.csv files found (filenames/headers did not match expected format).")

    runs.sort(key=lambda r: r["sort_key"])  # type: ignore[index]
    epochs_sorted = sorted(all_epochs)

    # IMPORTANT:
    # Plotting expects columns like: <phase>-<phasedata>_nsNN_(train|valid)_loss
    # For var-specific combined files we intentionally OMIT var from the column tag.
    header = ["epoch"]
    for r in runs:
        rid: RunId = r["rid"]  # type: ignore[assignment]
        if var_filter is not None:
            tag = f"{rid.phase}-{rid.phasedata}_ns{int(rid.ns):02d}"
        else:
            tag = rid.tag
        header += [f"{tag}_train_loss", f"{tag}_valid_loss"]

    with output_csv.open("w", newline="") as out_f:
        writer = csv.writer(out_f)
        writer.writerow(header)

        for epoch in epochs_sorted:
            row_out: List[object] = [epoch]
            for r in runs:
                data: Dict[int, Tuple[str, str]] = r["data"]  # type: ignore[assignment]
                vals = data.get(epoch)
                if vals is None:
                    row_out.extend(["", ""])
                else:
                    train_str, valid_str = vals
                    try:
                        train_out = f"{float(train_str):.4f}" if train_str != "" else ""
                    except ValueError:
                        train_out = train_str

                    if valid_str == "":
                        valid_out = ""
                    else:
                        try:
                            valid_out = f"{float(valid_str):.4f}"
                        except ValueError:
                            valid_out = valid_str

                    row_out.extend([train_out, valid_out])

            writer.writerow(row_out)

    counts = {"pretrain": 0, "dtrain": 0, "finetune": 0}
    for r in runs:
        rid: RunId = r["rid"]  # type: ignore[assignment]
        counts[rid.phase] += 1

    print(f"Wrote {output_csv} with {len(epochs_sorted)} epochs and {len(runs)} runs.")
    print(f"Runs by phase: {counts}")
    if skipped:
        print(f"Skipped {len(skipped)} file(s) that didn’t match expected pattern/headers.")


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

COL_RE = re.compile(
    r"^(?P<phase>pretrain|dtrain|finetune)-(?P<phasedata>[A-Za-z0-9]+)_ns(?P<ns>[0-9]{2})_(?P<kind>train_loss|valid_loss)$"
)

PHASE_ORDER = {"pretrain": 0, "dtrain": 1, "finetune": 2}


def safe_float(s: str) -> Optional[float]:
    try:
        v = float(s)
        if not math.isfinite(v):
            return None
        return v
    except Exception:
        return None


def eight_ticks(max_epoch: int) -> List[int]:
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


def discover_groups(header: List[str]) -> Dict[Tuple[str, str], Set[str]]:
    seen: Dict[Tuple[str, str, str], Set[str]] = {}

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


def group_max_epoch(rows: List[dict], cols: List[str]) -> int:
    max_ep = 0
    for r in rows:
        ep = int(r["epoch"])
        for c in cols:
            s = (r.get(c) or "").strip()
            if not s:
                continue
            v = safe_float(s)
            if v is None:
                continue
            max_ep = max(max_ep, ep)
            break
    return max_ep


def compute_global_y_limits(combined_csvs: Sequence[Path]) -> Tuple[float, float]:
    """
    Compute ONE y-range to apply to ALL plots across ALL combined CSVs.
    """
    y_min = float("inf")
    y_max = float("-inf")

    for p in combined_csvs:
        if not p.exists():
            continue
        with p.open(newline="") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames or []
            rows = list(reader)

        # only scan columns that match COL_RE (loss columns)
        loss_cols = [c for c in header if COL_RE.match(c)]
        for c in loss_cols:
            for r in rows:
                s = (r.get(c) or "").strip()
                if not s:
                    continue
                v = safe_float(s)
                if v is None:
                    continue
                y_min = min(y_min, v)
                y_max = max(y_max, v)

    if not (math.isfinite(y_min) and math.isfinite(y_max)):
        raise SystemExit("Could not compute a global Y range (no numeric loss values found).")

    pad = 0.05 * (y_max - y_min) if y_max > y_min else 0.1
    y_lo = max(0.0, y_min - pad)
    y_hi = y_max + pad
    return y_lo, y_hi


def plot_grids(
    combined_csv: Path,
    outdir: Path,
    ncols: int = 4,
    max_plots: int = 64,
    *,
    pdf_suffix: str = "",
    y_limits: Optional[Tuple[float, float]] = None,
) -> None:
    if not combined_csv.exists():
        raise SystemExit(f"Input file {str(combined_csv)!r} not found.")

    with combined_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        rows = list(reader)

    if "epoch" not in header:
        raise SystemExit("Expected 'epoch' column in combined CSV.")

    groups = discover_groups(header)
    if not groups:
        raise SystemExit("No matching <phase>-<phasedata>_nsNN_(train|valid)_loss columns found.")

    outdir.mkdir(parents=True, exist_ok=True)

    group_keys = sorted(groups.keys(), key=lambda k: (PHASE_ORDER.get(k[0], 999), k[1]))
    for (phase, phasedata) in group_keys:
        ns_list = sorted(groups[(phase, phasedata)], key=lambda s: int(s))
        ns_list = ns_list[:max_plots]
        if not ns_list:
            continue

        xcols: List[str] = []
        for ns in ns_list:
            xcols.append(f"{phase}-{phasedata}_ns{ns}_train_loss")
            xcols.append(f"{phase}-{phasedata}_ns{ns}_valid_loss")

        max_epoch = group_max_epoch(rows, xcols)
        if max_epoch <= 0:
            max_epoch = max(int(r["epoch"]) for r in rows) if rows else 0
        xticks = eight_ticks(max_epoch)

        ncols_eff = max(1, ncols)
        nrows = math.ceil(len(ns_list) / ncols_eff)

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols_eff,
            figsize=(4.8 * ncols_eff, 3.6 * nrows),
            squeeze=False,
            sharex=True,
            sharey=True,
        )
        axes_flat = axes.ravel()
        for ax in axes_flat:
            ax.set_visible(False)

        # GLOBAL y-limits for all plots (across grids and files)
        if y_limits is None:
            # fallback: compute per-group (old behavior)
            y_min = float("inf")
            y_max = float("-inf")

            def update_range(colname: str) -> None:
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
        else:
            y_lo, y_hi = y_limits

        fig.suptitle(f"{phase}-{phasedata}", fontsize=22, fontweight="bold", y=0.995)

        epochs = np.array([int(r["epoch"]) for r in rows], dtype=int)

        for i, ns in enumerate(ns_list):
            ax = axes_flat[i]
            ax.set_visible(True)

            tcol = f"{phase}-{phasedata}_ns{ns}_train_loss"
            vcol = f"{phase}-{phasedata}_ns{ns}_valid_loss"

            train_y = np.array(
                [safe_float((r.get(tcol) or "").strip()) if (r.get(tcol) or "").strip() else np.nan for r in rows],
                dtype=float,
            )
            valid_y = np.array(
                [safe_float((r.get(vcol) or "").strip()) if (r.get(vcol) or "").strip() else np.nan for r in rows],
                dtype=float,
            )

            mask = epochs <= max_epoch
            ax.plot(epochs[mask], train_y[mask], label="train", linewidth=1.2)
            ax.plot(epochs[mask], valid_y[mask], label="valid", linewidth=1.2)

            ax.set_title(f"ns{ns}", fontsize=14)
            ax.set_xlim(0, max_epoch)
            ax.set_ylim(y_lo, y_hi)
            ax.set_xticks(xticks)
            ax.grid(True, alpha=0.25)
            if i == 0:
                ax.legend(loc="best", fontsize=10)

        for ax in axes_flat[(nrows - 1) * ncols_eff : nrows * ncols_eff]:
            if ax.get_visible():
                ax.set_xlabel("Epoch")
        for r_i in range(nrows):
            ax = axes[r_i, 0]
            if ax.get_visible():
                ax.set_ylabel("Loss")

        out_path = outdir / f"{phase}-{phasedata}{pdf_suffix}_grid.pdf"
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(out_path)
        plt.close(fig)

        print(f"Saved {out_path} (panels={len(ns_list)}, grid={nrows}x{ncols_eff}, x_max={max_epoch})")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract, combine, and plot loss curves.")
    p.add_argument("--log-dir", type=Path, required=True, help="Directory containing out_final_*.log files.")
    p.add_argument("--log-glob", default="out_final_L_*.log", help="Glob pattern within --log-dir.")
    p.add_argument("--outdir", type=Path, default=Path("."), help="Where to write CSVs/PDFs (default: .)")
    p.add_argument("--loss-pattern", default="**/loss_final_*.csv", help="Glob for per-run loss CSVs for combining.")
    p.add_argument("--ncols", type=int, default=4, help="Number of columns in plot grid.")
    p.add_argument("--max-plots", type=int, default=64, help="Max panels per PDF.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # 1) extract
    extract_all_losses(log_dir=args.log_dir, log_glob=args.log_glob, out_dir=args.outdir)

    # 2) combine base + var-3 + var-6
    pattern = str(args.outdir / args.loss_pattern) if str(args.outdir) != "." else args.loss_pattern

    combined_base = args.outdir / Path("combined_losses.csv")
    if combined_base.exists():
        combined_base.unlink()
    combine_losses(pattern=pattern, output_csv=combined_base, var_filter=None)

    combined_v3 = args.outdir / Path("combined_losses_var-3.csv")
    if combined_v3.exists():
        combined_v3.unlink()
    combine_losses(pattern=pattern, output_csv=combined_v3, var_filter="3")

    combined_v6 = args.outdir / Path("combined_losses_var-6.csv")
    if combined_v6.exists():
        combined_v6.unlink()
    combine_losses(pattern=pattern, output_csv=combined_v6, var_filter="6")

    # 3) compute ONE global Y range across ALL combined files
    global_y = compute_global_y_limits([combined_base, combined_v3, combined_v6])
    print(f"Global Y limits: {global_y[0]:.6g} .. {global_y[1]:.6g}")

    # 4) plot with shared Y across everything
    plot_grids(combined_csv=combined_base, outdir=args.outdir, ncols=args.ncols, max_plots=args.max_plots, pdf_suffix="", y_limits=global_y)
    plot_grids(combined_csv=combined_v3, outdir=args.outdir, ncols=args.ncols, max_plots=args.max_plots, pdf_suffix="_var-3", y_limits=global_y)
    plot_grids(combined_csv=combined_v6, outdir=args.outdir, ncols=args.ncols, max_plots=args.max_plots, pdf_suffix="_var-6", y_limits=global_y)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
