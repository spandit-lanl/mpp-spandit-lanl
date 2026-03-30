#!/usr/bin/env python3
"""
predict_mpp_incremental_stage4_fixed.py

Incremental script through Stage-4.

Stage-1: DATA I/O sanity
Stage-2: MODEL + CKPT wiring sanity
Stage-3: ONE forward pass sanity (enabled with --run_forward)
Stage-4: SAVE artifacts + append metrics.csv (enabled with --save_outputs)

Key fix vs prior version:
- AViT in this repo uses *subsampled* state selection: the input should contain ONLY the
  selected channels (Csel=len(fields)), while state_labels provides the mapping to global
  state indices (e.g., [12,13,14,15]). Do NOT pad to n_states=16.

Everything is explicit: no path deduction.

Required:
  --repo_root, --config, --ckpt, --data_dir, --ns, --t_idx, --var, --fields
  and either --sample_id or --expt_id

Stage-3 requires:
  --run_forward
  --state_indices (len must match fields)
Defaults:
  --yparams_section finetune_resume
  --state_indices 12,13,14,15

Stage-4 requires:
  --save_outputs
  --out_root (default ./final/pred)

Outputs (Stage-4):
  {out_root}/{dataset}/expt-{expt_id}/t{t_idx:05d}/ns-{ns:02d}_var-{var}/
    pred.npz, metrics.json, meta.json
  and appends (one per experiment):
    {out_root}/{dataset}/expt-{expt_id}/metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import torch

def safe_torch_load(path, map_location):
    """Load a torch checkpoint safely.
    Returns the loaded object, or None if unreadable/corrupt.
    """
    import os, time, torch
    try:
        if not os.path.exists(path):
            print(f"SKIP: missing checkpoint: {path}")
            return None
        if os.path.getsize(path) == 0:
            print(f"SKIP: empty checkpoint: {path}")
            return None
    except Exception as e:
        print(f"SKIP: checkpoint stat failed: {path} ({e})")
        return None

    last_err = None
    for attempt in range(1, 4):
        try:
            return torch.load(path, map_location=map_location)
        except (EOFError, OSError, RuntimeError, ValueError) as e:
            last_err = e
            print(f"SKIP: checkpoint unreadable (attempt {attempt}/3): {path} ({type(e).__name__}: {e})")
            time.sleep(0.5 * attempt)
    print(f"SKIP: giving up on checkpoint: {path} ({type(last_err).__name__}: {last_err})")
    return None

from utils.YParams import YParams
from models.avit import build_avit


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def mkdir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def print_args(args) -> None:
    print("=== INPUT ARGS ===")
    for k, v in sorted(vars(args).items()):
        print(f"{k}: {v}")


def parse_fields(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_int_list(s: Optional[str]) -> Optional[List[int]]:
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    return [int(x) for x in s.split(",")]


def resolve_sample_id(sample_id: Optional[str], expt_id: Optional[str], series: Optional[str], suffix: str) -> str:
    if sample_id:
        return sample_id
    if expt_id is None:
        raise ValueError("Provide either --sample_id or --expt_id")
    series2 = series or "cx241203"
    return f"{series2}_id{expt_id}_{suffix}"


def fpath(data_dir: str, template: str, sample_id: str, t_idx: int) -> str:
    return os.path.join(data_dir, template.format(sample_id=sample_id, t_idx=t_idx))


def sanitize(a: np.ndarray) -> np.ndarray:
    b = np.array(a, copy=True)
    b[~np.isfinite(b)] = 0.0
    return b


def finite_frac(a: np.ndarray) -> float:
    return float(np.isfinite(a).mean())


def stat_str(a: np.ndarray) -> str:
    af = a[np.isfinite(a)]
    if af.size == 0:
        return f"dtype={a.dtype} shape={tuple(a.shape)} all-nonfinite"
    return f"dtype={a.dtype} shape={tuple(a.shape)} min={af.min():.6g} max={af.max():.6g} finite={finite_frac(a):.6g}"


def load_npz_fields(path: str, fields: List[str]) -> List[np.ndarray]:
    d = np.load(path)
    return [d[f] for f in fields]


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def rmse(pred: np.ndarray, gt: np.ndarray) -> float:
    mask = np.isfinite(pred) & np.isfinite(gt)
    if mask.sum() == 0:
        return float("nan")
    return float(np.sqrt(((pred[mask] - gt[mask]) ** 2).mean()))


def append_metrics_csv(csv_path: str, row: Dict[str, object]) -> None:
    mkdir(os.path.dirname(csv_path))
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--repo_root", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--yparams_section", default="finetune_resume")
    ap.add_argument("--ckpt", required=True)

    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--dataset", default="clx")
    ap.add_argument("--ns", type=int, required=True)
    ap.add_argument("--t_idx", type=int, required=True)
    ap.add_argument("--var", type=int, required=True)
    ap.add_argument("--fields", required=True)

    ap.add_argument("--sample_id", default=None)
    ap.add_argument("--expt_id", default=None)
    ap.add_argument("--series", default=None)
    ap.add_argument("--suffix", default="pvi")

    ap.add_argument("--filename_template", default="{sample_id}_idx{t_idx:05d}.npz")
    ap.add_argument("--print_keys", action="store_true")
    ap.add_argument("--sanitize_inputs", action="store_true")
    ap.add_argument("--device", default=None)

    ap.add_argument("--run_forward", action="store_true")
    ap.add_argument("--state_indices", default="12,13,14,15")

    ap.add_argument("--save_outputs", action="store_true")
    ap.add_argument("--out_root", default="./final/pred")
    ap.add_argument("--tag", default="")

    args = ap.parse_args()

    repo_root_abs = os.path.abspath(args.repo_root)
    if repo_root_abs not in sys.path:
        sys.path.insert(0, repo_root_abs)

    print_args(args)

    # -------- Stage-1 --------
    print("\n================ STAGE 1: DATA I/O ================")
    data_dir = os.path.abspath(args.data_dir)
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"data_dir not found: {data_dir}")
    if args.ns < 1:
        raise ValueError("--ns must be >= 1")
    if args.t_idx < args.ns:
        raise ValueError(f"--t_idx ({args.t_idx}) must be >= --ns ({args.ns})")

    sample_id = resolve_sample_id(args.sample_id, args.expt_id, args.series, args.suffix)
    fields = parse_fields(args.fields)

    ctx_indices = list(range(args.t_idx - args.ns, args.t_idx))
    tgt_index = args.t_idx
    ctx_files = [fpath(data_dir, args.filename_template, sample_id, i) for i in ctx_indices]
    tgt_file = fpath(data_dir, args.filename_template, sample_id, tgt_index)

    missing_files = [fp for fp in ctx_files + [tgt_file] if not os.path.exists(fp)]
    if missing_files:
        print("MISSING FILES:")
        for fp in missing_files[:50]:
            print(f"  - {fp}")
        raise FileNotFoundError(f"{len(missing_files)} required NPZ files missing")

    d0 = np.load(ctx_files[0])
    keys0 = list(d0.keys())
    if args.print_keys:
        print("\nNPZ keys (first context file):")
        for k in keys0:
            print(k)

    missing_fields = [f for f in fields if f not in keys0]
    if missing_fields:
        raise KeyError(f"Requested fields missing from NPZ: {missing_fields}")

    X_list = []
    ref_shape = None
    first_arrs = load_npz_fields(ctx_files[0], fields)
    if args.sanitize_inputs:
        first_arrs = [sanitize(a) for a in first_arrs]
    print("\nField stats (first context timestep):")
    for f, a in zip(fields, first_arrs):
        print(f"  {f}: {stat_str(a)}")

    for fp in ctx_files:
        arrs = load_npz_fields(fp, fields)
        if args.sanitize_inputs:
            arrs = [sanitize(a) for a in arrs]
        stacked = np.stack(arrs, axis=0)  # (Csel,H,W)
        if ref_shape is None:
            ref_shape = stacked.shape
        if stacked.shape != ref_shape:
            raise ValueError(f"Shape mismatch at {fp}: got {stacked.shape}, expected {ref_shape}")
        X_list.append(stacked)

    X = np.stack(X_list, axis=0)  # (T,Csel,H,W)
    Y_arrs = load_npz_fields(tgt_file, fields)
    if args.sanitize_inputs:
        Y_arrs = [sanitize(a) for a in Y_arrs]
    Y = np.stack(Y_arrs, axis=0)  # (Csel,H,W)

    print(f"\nX shape: {X.shape} (T,Csel,H,W)")
    print(f"Y shape: {Y.shape} (Csel,H,W)")
    if Y.shape != X.shape[1:]:
        raise ValueError(f"Target shape {Y.shape} != context channel/spatial shape {X.shape[1:]}")

    # -------- Stage-2 --------
    print("\n================ STAGE 2: MODEL + CKPT ================")
    config_abs = os.path.abspath(args.config)
    ckpt_abs = os.path.abspath(args.ckpt)
    if not os.path.isfile(config_abs):
        raise FileNotFoundError(f"Config not found: {config_abs}")
    if not os.path.isfile(ckpt_abs):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_abs}")

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    params = YParams(config_abs, args.yparams_section)
    model = build_avit(params).to(device).eval()
    n_params = count_params(model)
    print(f"model build OK, params={n_params:,}")

    ckpt = safe_torch_load(ckpt_abs, map_location=device)
    if ckpt is None:
        return  # skip this run without crashing
    if not isinstance(ckpt, dict):
        raise TypeError(f"Checkpoint is not a dict: {type(ckpt)}")
    state = None
    for k in ["model_state", "model", "model_state_dict", "state_dict", "net", "network"]:
        if k in ckpt and isinstance(ckpt[k], dict):
            state = ckpt[k]
            print(f"using state key '{k}', len={len(state)}")
            break
    if state is None:
        raise RuntimeError(f"Could not find model state dict in ckpt keys={list(ckpt.keys())}")

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"load_state_dict done: missing={len(missing)} unexpected={len(unexpected)}")

    if not args.run_forward:
        print("\nDONE: stages 1-2 passed (no forward pass).")
        return

    # -------- Stage-3 --------
    print("\n================ STAGE 3: SINGLE FORWARD ================")
    state_indices = parse_int_list(args.state_indices)
    if state_indices is None:
        raise ValueError("--state_indices must be provided")
    if len(state_indices) != len(fields):
        raise ValueError(f"--state_indices length {len(state_indices)} must match fields length {len(fields)}")

    # IMPORTANT: feed only the selected channels (Csel), with mapping in state_labels.
    x_t = torch.from_numpy(X.astype(np.float32)).to(device=device, dtype=torch.float32).unsqueeze(1)  # (T,1,Csel,H,W)
    state_labels = [torch.tensor(state_indices, device=device, dtype=torch.long)]
    bcs = torch.zeros((1, 2), device=device, dtype=torch.float32)

    with torch.no_grad():
        y_full = model(x_t, state_labels, bcs)

    y_np = y_full.detach().cpu().numpy()
    # expected (1,Csel,H,W) but handle a couple common possibilities
    if y_np.ndim == 4 and y_np.shape[0] == 1:
        pred_sel = y_np[0]  # (Csel,H,W)
    elif y_np.ndim == 3:
        pred_sel = y_np
    else:
        raise RuntimeError(f"Unexpected y_full shape: {y_np.shape}")

    rmse_all = rmse(pred_sel, Y.astype(np.float32))
    per_field = {fields[i]: rmse(pred_sel[i], Y[i].astype(np.float32)) for i in range(len(fields))}

    print(f"pred_sel shape: {pred_sel.shape}  rmse_all={rmse_all:.6g}")
    for f in fields:
        print(f"  rmse {f}: {per_field[f]:.6g}")

    if not args.save_outputs:
        print("\nDONE: stage 3 complete (no saving).")
        return

    # -------- Stage-4 --------
    print("\n================ STAGE 4: SAVE OUTPUTS ================")
    out_root = os.path.abspath(args.out_root)
    expt = args.expt_id if args.expt_id is not None else "NA"
    # Output layout (agreed):
    #   {out_root}/{dataset}/expt-<id>/t<####>/ns-<##>_var-<v>/
    out_dir = os.path.join(
        out_root,
        args.dataset,
        f"expt-{expt}",
        f"t{args.t_idx:05d}",
        f"ns-{args.ns:02d}_var-{args.var}",
    )
    mkdir(out_dir)

    pred_path = os.path.join(out_dir, "pred.npz")
    metrics_path = os.path.join(out_dir, "metrics.json")
    meta_path = os.path.join(out_dir, "meta.json")

    np.savez_compressed(
        pred_path,
        pred_sel=pred_sel.astype(np.float32),
        gt_sel=Y.astype(np.float32),
        fields=np.array(fields),
        state_indices=np.array(state_indices, dtype=np.int64),
        ctx_indices=np.array(ctx_indices, dtype=np.int64),
        t_idx=args.t_idx,
        ns=args.ns,
        var=args.var,
        dataset=args.dataset,
        sample_id=sample_id,
        expt_id=expt,
        ckpt_path=ckpt_abs,
        config_path=config_abs,
        yparams_section=args.yparams_section,
        tag=args.tag,
    )

    metrics = {
        "timestamp": now_iso(),
        "dataset": args.dataset,
        "ns": args.ns,
        "var": args.var,
        "expt_id": expt,
        "t_idx": args.t_idx,
        "fields": fields,
        "state_indices": state_indices,
        "rmse_all": float(rmse_all),
        "rmse_per_field": {k: float(v) for k, v in per_field.items()},
        "finite_frac_pred": float(np.isfinite(pred_sel).mean()),
        "pred_path": pred_path,
        "tag": args.tag,
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    meta = {
        "timestamp": now_iso(),
        "args": vars(args),
        "ctx_files": ctx_files,
        "tgt_file": tgt_file,
        "device": str(device),
        "model_param_count": int(n_params),
        "missing_keys_count": int(len(missing)),
        "unexpected_keys_count": int(len(unexpected)),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    metrics_csv = os.path.join(out_root, args.dataset, f"expt-{expt}", "metrics.csv")
    row = {
        "timestamp": metrics["timestamp"],
        "dataset": args.dataset,
        "ns": args.ns,
        "var": args.var,
        "expt_id": expt,
        "t_idx": args.t_idx,
        "rmse_all": float(rmse_all),
        "finite_frac_pred": metrics["finite_frac_pred"],
        "pred_path": pred_path,
        "tag": args.tag,
        "ckpt_path": ckpt_abs,
        "config_path": config_abs,
        "yparams_section": args.yparams_section,
    }
    for f in fields:
        row[f"rmse_{f}"] = float(per_field[f])
    append_metrics_csv(metrics_csv, row)

    print(f"saved: {pred_path}")
    print(f"saved: {metrics_path}")
    print(f"saved: {meta_path}")
    print(f"appended: {metrics_csv}")
    print("\nDONE: stage 4 complete.")


if __name__ == "__main__":
    main()
