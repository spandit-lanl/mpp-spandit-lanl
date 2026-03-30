#!/usr/bin/env python3
"""
Compute per-field RMSE from MPP stage-4 prediction outputs.

Assumptions:
- Each run directory contains pred.npz with arrays:
    - pred_sel: (C,H,W) or (T,C,H,W)
    - gt_sel:   same shape as pred_sel
    - fields:   (C,) list/array of field names
    - t_idx:    int (optional)
- Directory layout (as agreed):
    <out_root>/<dataset>/expt-<EXPT_ID>/t<#####>/ns-<##>_var-<v>/pred.npz

Outputs:
- CSV with one row per (run, field):
    dataset,expt_id,t_idx,ns,var,field,step_rmse,step_mse,cum_rmse,cum_mse,n_pixels,n_frames,run_dir,pred_path

Cumulative RMSE:
- Defined across timesteps encountered in the scan for each (dataset, expt_id, ns, var, field):
    cum_mse = mean(step_mse over timesteps)
    cum_rmse = sqrt(cum_mse)
"""

import argparse
import math
import os
import re
from pathlib import Path
import numpy as np


RUN_RE = re.compile(
    r"(?:^|/)"
    r"(?P<dataset>[^/]+)/"
    r"expt-(?P<expt_id>\d+)/"
    r"t(?P<t_idx>\d{5})/"
    r"ns-(?P<ns>\d{2})_var-(?P<var>\d+)"
    r"(?:/|$)"
)

def parse_run_fields(pred_path: Path):
    """Return metadata parsed from the pred_path directory structure."""
    m = RUN_RE.search(str(pred_path))
    if not m:
        return None
    d = m.groupdict()
    return {
        "dataset": d["dataset"],
        "expt_id": d["expt_id"],
        "t_idx": int(d["t_idx"]),
        "ns": int(d["ns"]),
        "var": int(d["var"]),
        "run_dir": str(pred_path.parent),
        "pred_path": str(pred_path),
    }

def load_pred_npz(pred_path: Path):
    z = np.load(pred_path, allow_pickle=True)
    # robust fetch
    pred = z["pred_sel"] if "pred_sel" in z else (z["pred"] if "pred" in z else None)
    gt   = z["gt_sel"]   if "gt_sel"   in z else (z["gt"]   if "gt"   in z else None)
    fields = z["fields"] if "fields" in z else None
    if pred is None or gt is None or fields is None:
        raise KeyError(f"Missing required keys in {pred_path}. Need pred_sel/gt_sel/fields.")
    # normalize fields to list[str]
    fields = [f.decode() if isinstance(f, (bytes, bytearray)) else str(f) for f in fields.tolist()]
    return pred, gt, fields, z

def mse_rmse(a, b):
    diff = (a.astype(np.float64) - b.astype(np.float64))
    mse = float(np.mean(diff * diff))
    rmse = math.sqrt(mse)
    return mse, rmse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", required=True, help="Root output folder that contains <dataset>/expt-<id>/...")
    ap.add_argument("--dataset", default=None, help="If set, only scan this dataset subdir (e.g. clx)")
    ap.add_argument("--expt_id", required=True, help="Experiment id digits, e.g. 01750 (will match expt-01750)")
    ap.add_argument("--csv_out", default=None, help="Output CSV path. Default: <out_root>/<dataset>/expt-<id>/rmse_summary.csv (or <out_root>/expt-<id>/rmse_summary.csv if dataset not provided)")
    ap.add_argument("--strict_layout", action="store_true", help="Fail if a pred.npz doesn't match expected directory layout.")
    args = ap.parse_args()

    out_root = Path(args.out_root).expanduser().resolve()
    if not out_root.exists():
        raise SystemExit(f"--out_root does not exist: {out_root}")

    # Choose scan root
    if args.dataset:
        scan_root = out_root / args.dataset / f"expt-{args.expt_id}"
    else:
        # scan all datasets under out_root for this expt
        scan_root = out_root
    if not scan_root.exists():
        raise SystemExit(f"Scan root does not exist: {scan_root}")

    pred_paths = list(scan_root.rglob("pred.npz"))
    if not pred_paths:
        raise SystemExit(f"No pred.npz found under {scan_root}")

    rows = []
    # For cumulative: collect step_mse by key
    # key = (dataset, expt_id, ns, var, field)
    accum = {}

    for p in sorted(pred_paths):
        meta = parse_run_fields(p)
        if meta is None:
            if args.strict_layout:
                raise SystemExit(f"pred.npz path does not match expected layout: {p}")
            else:
                continue

        # Filter expt_id match strictly
        if meta["expt_id"] != args.expt_id:
            continue

        pred, gt, fields, z = load_pred_npz(p)

        # Accept either (C,H,W) or (T,C,H,W); compute step on the LAST frame if T present
        if pred.ndim == 4:
            # (T,C,H,W)
            pred_step = pred[-1]
            gt_step   = gt[-1]
            n_frames = pred.shape[0]
        elif pred.ndim == 3:
            pred_step = pred
            gt_step   = gt
            n_frames = 1
        else:
            raise ValueError(f"Unexpected pred_sel dims {pred.shape} in {p}")

        if pred_step.shape != gt_step.shape:
            raise ValueError(f"Shape mismatch pred vs gt in {p}: {pred_step.shape} vs {gt_step.shape}")

        C = pred_step.shape[0]
        if len(fields) != C:
            raise ValueError(f"fields length {len(fields)} != C {C} in {p}")

        n_pixels = int(np.prod(pred_step.shape[1:]))

        for ci, field in enumerate(fields):
            step_mse, step_rmse = mse_rmse(pred_step[ci], gt_step[ci])

            row = dict(
                dataset=meta["dataset"],
                expt_id=meta["expt_id"],
                t_idx=meta["t_idx"],
                ns=meta["ns"],
                var=meta["var"],
                field=field,
                step_rmse=step_rmse,
                step_mse=step_mse,
                cum_rmse="",  # fill after
                cum_mse="",
                n_pixels=n_pixels,
                n_frames=n_frames,
                run_dir=meta["run_dir"],
                pred_path=meta["pred_path"],
            )
            rows.append(row)

            k = (meta["dataset"], meta["expt_id"], meta["ns"], meta["var"], field)
            accum.setdefault(k, []).append(step_mse)

    if not rows:
        raise SystemExit(f"No matching pred.npz found for expt_id={args.expt_id} under {scan_root}")

    # Compute cumulative by averaging step_mse across timesteps (per key)
    cum_stats = {k: (float(np.mean(v)), math.sqrt(float(np.mean(v))), len(v)) for k, v in accum.items()}

    for r in rows:
        k = (r["dataset"], r["expt_id"], r["ns"], r["var"], r["field"])
        cmse, crmse, n_steps = cum_stats[k]
        r["cum_mse"] = cmse
        r["cum_rmse"] = crmse
        r["n_steps"] = n_steps

    # Default csv_out
    if args.csv_out:
        csv_out = Path(args.csv_out).expanduser()
    else:
        if args.dataset:
            csv_out = out_root / args.dataset / f"expt-{args.expt_id}" / "rmse_summary.csv"
        else:
            csv_out = out_root / f"expt-{args.expt_id}" / "rmse_summary.csv"
    csv_out.parent.mkdir(parents=True, exist_ok=True)

    # Write CSV
    fieldnames = ["dataset","expt_id","t_idx","ns","var","field","step_rmse","step_mse","cum_rmse","cum_mse","n_steps","n_pixels","n_frames","run_dir","pred_path"]
    with csv_out.open("w", newline="") as f:
        import csv
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Wrote: {csv_out} ({len(rows)} rows)")

if __name__ == "__main__":
    main()
