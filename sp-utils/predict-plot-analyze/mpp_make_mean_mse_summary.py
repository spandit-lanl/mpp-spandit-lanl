#!/usr/bin/env python3
"""
mpp_make_mean_mse_summary.py

Create a master mean_mse_summary.csv by averaging step_mse across experiments.

Inputs:
  --ids_file : text file with one 5-digit expt id per line (e.g., 00005)
  --out_root : root directory containing <dataset>/expt-<#####>/mse_summary.csv
  --dataset  : dataset name (e.g., clx)
  --out_csv  : output CSV path (default: ./mean_mse_summary.csv)

Behavior:
  - Reads each expt's mse_summary.csv (skips missing files, logs warning).
  - Uses ONLY columns: dataset, t_idx, field, step_mse
  - Computes mean_step_mse across experiments for each (dataset, t_idx, field)
  - Outputs n_expts_used = number of experiments contributing to that group.

Notes:
  - Does NOT create directories.
  - Does NOT require all experiments to have the same timesteps.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import pandas as pd


def read_ids(ids_file: Path) -> list[str]:
    ids: list[str] = []
    for line in ids_file.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("expt-"):
            s = s.split("expt-", 1)[1]
        if not s.isdigit():
            print(f"[WARN] {Path(__file__).name}: skipping non-numeric id line: {line!r}", file=sys.stderr)
            continue
        ids.append(f"{int(s):05d}")
    return ids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids_file", required=True, help="Text file with expt ids (5-digit or numeric), one per line.")
    ap.add_argument("--out_root", required=True, help="Root dir containing <dataset>/expt-<#####>/mse_summary.csv")
    ap.add_argument("--dataset", required=True, help="Dataset subdir name, e.g. clx")
    ap.add_argument("--out_csv", default="./mean_mse_summary.csv", help="Output CSV path (default ./mean_mse_summary.csv)")
    ap.add_argument("--quiet", action="store_true", help="Reduce logging")
    args = ap.parse_args()

    ids_file = Path(args.ids_file)
    out_root = Path(args.out_root)
    dataset = str(args.dataset).strip()
    out_csv = Path(args.out_csv)

    if not ids_file.exists():
        print(f"[ERROR] {Path(__file__).name}: ids_file not found: {ids_file}", file=sys.stderr)
        return 2
    if not out_root.exists():
        print(f"[ERROR] {Path(__file__).name}: out_root not found: {out_root}", file=sys.stderr)
        return 2

    expt_ids = read_ids(ids_file)
    if not expt_ids:
        print(f"[ERROR] {Path(__file__).name}: no valid expt ids found in {ids_file}", file=sys.stderr)
        return 2

    frames = []
    n_missing = 0
    n_read = 0

    for expt in expt_ids:
        csv_path = out_root / dataset / f"expt-{expt}" / "mse_summary.csv"
        if not csv_path.exists():
            n_missing += 1
            if not args.quiet:
                print(f"[WARN] missing: {csv_path}", file=sys.stderr)
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            if not args.quiet:
                print(f"[WARN] failed reading {csv_path}: {e}", file=sys.stderr)
            continue

        required = {"dataset", "t_idx", "field", "step_mse"}
        if not required.issubset(df.columns):
            if not args.quiet:
                miss = sorted(required - set(df.columns))
                print(f"[WARN] {csv_path} missing required columns {miss}; skipping", file=sys.stderr)
            continue

        df2 = df.loc[:, ["dataset", "t_idx", "field", "step_mse"]].copy()
        df2["dataset"] = df2["dataset"].astype(str).str.strip()
        df2["field"] = df2["field"].astype(str).str.strip()
        df2["t_idx"] = pd.to_numeric(df2["t_idx"], errors="coerce")
        df2["step_mse"] = pd.to_numeric(df2["step_mse"], errors="coerce")
        df2 = df2.dropna(subset=["t_idx", "field", "step_mse", "dataset"])

        df2 = df2[df2["dataset"] == dataset]
        df2["expt_id"] = expt

        frames.append(df2)
        n_read += 1

    if not frames:
        print(f"[ERROR] {Path(__file__).name}: no mse_summary.csv files could be read.", file=sys.stderr)
        return 2

    all_df = pd.concat(frames, ignore_index=True)

    grp = all_df.groupby(["dataset", "t_idx", "field"], as_index=False)
    out = grp["step_mse"].agg(mean_step_mse="mean")

    expt_counts = all_df.groupby(["dataset", "t_idx", "field"])["expt_id"].nunique().reset_index(name="n_expts_used")
    out = out.merge(expt_counts, on=["dataset", "t_idx", "field"], how="left")

    out["t_idx"] = out["t_idx"].astype(int)
    out = out.sort_values(["dataset", "field", "t_idx"]).reset_index(drop=True)

    if out_csv.parent != Path(".") and not out_csv.parent.exists():
        print(f"[ERROR] {Path(__file__).name}: output directory does not exist (not creating): {out_csv.parent}", file=sys.stderr)
        return 2

    out.to_csv(out_csv, index=False)
    if not args.quiet:
        print(f"[INFO] wrote {out_csv} (read {n_read} CSVs, missing {n_missing})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
