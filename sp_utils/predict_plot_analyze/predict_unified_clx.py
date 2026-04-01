#!/usr/bin/env python3
"""
Unified CLX single-step prediction script.
"""
from __future__ import annotations
import argparse, csv, datetime as dt, re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

def _import_model_and_params():
    try:
        from utils import YParams  # type: ignore
    except Exception:
        from YParams import YParams  # type: ignore

    try:
        from models import build_avit  # type: ignore
        return build_avit, YParams
    except Exception:
        from avit import AViT  # type: ignore
        def build_avit(params):
            return AViT(params)
        return build_avit, YParams

def safe_load_state_dict(model, ckpt: Dict):
    state = ckpt
    for key in ("model_state", "state_dict", "model", "net"):
        if isinstance(state, dict) and key in state and isinstance(state[key], dict):
            state = state[key]
            break
    if not isinstance(state, dict):
        raise TypeError("Checkpoint does not contain a state dict in expected keys.")
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    return missing, unexpected

def parse_ns_var_from_ckpt(ckpt_path: Path) -> Tuple[str, str]:
    m = re.search(r"clx_expt_ns-(\d+)_var-(\d+)", str(ckpt_path))
    if not m:
        raise ValueError(f"Could not parse ns/var from ckpt path: {ckpt_path}")
    return m.group(1).zfill(2), m.group(2)

def deduce_config_path(ckpt_path: Path, config_root: Path) -> Path:
    ns, var = parse_ns_var_from_ckpt(ckpt_path)
    return config_root / f"mpp_avit_L_clx_{ns}_var_{var}.yaml"

def read_npz_fields(npz_path: Path, fields: List[str], dtype_out) -> np.ndarray:
    with np.load(npz_path) as z:
        arrs = []
        for f in fields:
            if f not in z:
                raise KeyError(f"Missing field '{f}' in {npz_path}")
            a = z[f]
            if a.ndim != 2:
                raise ValueError(f"Field '{f}' has shape {a.shape}, expected 2D.")
            arrs.append(a.astype(dtype_out, copy=False))
        return np.stack(arrs, axis=0)

def finite_rmse(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, int, int]:
    mask = np.isfinite(pred) & np.isfinite(gt)
    n_fin = int(mask.sum())
    n_tot = int(mask.size)
    if n_fin == 0:
        return float("nan"), 0, n_tot
    d = pred[mask] - gt[mask]
    return float(np.sqrt(np.mean(d * d))), n_fin, n_tot

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def append_csv(csv_path: Path, row: Dict):
    ensure_dir(csv_path.parent)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)

def append_runlog(log_path: Path, lines: List[str]):
    ensure_dir(log_path.parent)
    with log_path.open("a") as f:
        for ln in lines:
            f.write(ln.rstrip("\n") + "\n")

def save_badmask(path: Path, pred: np.ndarray, gt: np.ndarray):
    np.savez_compressed(path, badmask_pred=~np.isfinite(pred), badmask_gt=~np.isfinite(gt))

def make_viz_png(png_path: Path, fields: List[str], gt: np.ndarray, pred: np.ndarray):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C = gt.shape[0]
    fig, axes = plt.subplots(3, C, figsize=(4*C, 10), squeeze=False)
    err = np.abs(pred - gt)
    for c in range(C):
        for r, img, title in [
            (0, gt[c], f"GT: {fields[c]}"),
            (1, pred[c], f"PRED: {fields[c]}"),
            (2, err[c], f"|ERR|: {fields[c]}"),
        ]:
            ax = axes[r][c]
            im = ax.imshow(img)
            ax.set_title(title)
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--npz_dir", default="/lustre/scratch5/exempt/artimis/mpmm/spandit/data/fp_full/clx_ft_test")
    p.add_argument("--sample_id", required=True)
    p.add_argument("--t_idx", type=int, required=True)
    p.add_argument("--n_steps", type=int, required=True)
    p.add_argument("--fields", default="vf,pf,ef,nvf")
    p.add_argument("--label_offset", type=int, default=0)
    p.add_argument("--sanitize_inputs", action="store_true")
    p.add_argument("--n_states", type=int, default=None)
    p.add_argument(
        "--state_indices",
        default=None,
        help=(
            "Comma-separated list of length Csel mapping each selected field to a channel index in the full n_states tensor. "
            "Example for 8 fields: --state_indices 12,13,14,15,16,17,18,19. "
            "If omitted, fields map to indices 0..Csel-1."
        ),
    )
    p.add_argument("--config_root",
                   default="/users/spandit/projects/artimis/mpp/mpp-spandit-npz-clx/config/finetune_clx/expt")
    p.add_argument("--out_root", default=None)
    args = p.parse_args()

    ckpt_path = Path(args.ckpt).expanduser().resolve()
    npz_dir = Path(args.npz_dir).expanduser().resolve()
    config_root = Path(args.config_root).expanduser().resolve()

    config_path = deduce_config_path(ckpt_path, config_root)
    if not config_path.exists():
        raise FileNotFoundError(f"Deduced config not found: {config_path}")

    run_dir = ckpt_path.parent.parent if ckpt_path.parent.name == "training_checkpoints" else ckpt_path.parent
    out_root = Path(args.out_root).expanduser().resolve() if args.out_root else (run_dir / "inference")

    fields = [x.strip() for x in args.fields.split(",") if x.strip()]
    t, n_steps, sample = args.t_idx, args.n_steps, args.sample_id

    out_dir = out_root / sample / f"ctx{n_steps}" / f"idx{t:05d}"
    ensure_dir(out_dir)
    pred_npz = out_dir / f"pred_{sample}_idx{t:05d}_ctx{n_steps}.npz"
    viz_png = out_dir / f"viz_{sample}_idx{t:05d}_ctx{n_steps}.png"
    badmask_npz = out_dir / f"badmask_{sample}_idx{t:05d}_ctx{n_steps}.npz"
    metrics_csv = out_root / "metrics.csv"
    runlog = out_root / "run.log"

    start_ts = dt.datetime.now().isoformat(timespec="seconds")
    append_runlog(runlog, [
        "="*80,
        f"[{start_ts}] START ckpt={ckpt_path}",
        f"config={config_path}",
        f"npz_dir={npz_dir}",
        f"sample_id={sample} t_idx={t} n_steps={n_steps}",
        f"fields={fields} label_offset={args.label_offset} sanitize_inputs={args.sanitize_inputs}",
        f"out_dir={out_dir}",
    ])

    build_model, YParams = _import_model_and_params()

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params = YParams(str(config_path), "default")

    n_states = args.n_states
    if n_states is None:
        try:
            n_states = int(getattr(params, "n_states"))
        except Exception:
            try:
                n_states = int(params.get("n_states"))  # type: ignore
            except Exception:
                n_states = len(fields)

    model = build_model(params).to(device)
    model.eval()

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    missing, unexpected = safe_load_state_dict(model, ckpt)
    append_runlog(runlog, [f"Loaded ckpt: missing={len(missing)} unexpected={len(unexpected)}"])

    def npz_for(i: int) -> Path:
        return npz_dir / f"{sample}_idx{i:05d}.npz"

    ctx = list(range(t - n_steps, t))
    if min(ctx) < 0:
        raise ValueError("t_idx too small for n_steps")
    xs = [read_npz_fields(npz_for(i), fields, np.float16) for i in ctx]
    gt_sel = read_npz_fields(npz_for(t), fields, np.float16)

    if args.sanitize_inputs:
        xs = [np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0) for a in xs]
        gt_sel = np.nan_to_num(gt_sel, nan=0.0, posinf=0.0, neginf=0.0)

    xs = [a.astype(np.float32, copy=False) for a in xs]
    gt_sel = gt_sel.astype(np.float32, copy=False)

    Csel, H, W = gt_sel.shape

    if args.state_indices is None:
        state_indices = list(range(Csel))
    else:
        state_indices = [int(x.strip()) for x in args.state_indices.split(",") if x.strip()]
        if len(state_indices) != Csel:
            raise ValueError(
                f"--state_indices must have length {Csel} (one per selected field); got {len(state_indices)}"
            )
        if any(i < 0 or i >= n_states for i in state_indices):
            raise ValueError(f"--state_indices must be within [0, n_states-1]; n_states={n_states}")
        if len(set(state_indices)) != len(state_indices):
            raise ValueError("--state_indices contains duplicates; each field must map to a unique state index")
    x_full = np.zeros((n_steps, 1, n_states, H, W), dtype=np.float32)
    gt_full = np.zeros((n_states, H, W), dtype=np.float32)
    x_stack = np.stack(xs, axis=0)  # [T, Csel, H, W]
    for j, si in enumerate(state_indices):
        x_full[:, 0, si] = x_stack[:, j]
        gt_full[si] = gt_sel[j]

    x_t = torch.from_numpy(x_full).to(device)
    labels = torch.arange(n_states, device=device, dtype=torch.long) + int(args.label_offset)

    with torch.no_grad():
        try:
            y = model(x_t, labels, None)
        except TypeError:
            y = model(x_t, labels)
        if isinstance(y, (tuple, list)):
            y = y[0]
        y = y.detach().float().cpu().numpy()
        pred_full = y[0] if y.ndim == 4 else y

    pred_sel = pred_full[state_indices]
    rmse, n_fin, n_tot = finite_rmse(pred_sel, gt_sel)
    save_badmask(badmask_npz, pred_sel, gt_sel)

    np.savez_compressed(
        pred_npz,
        pred=pred_sel, gt=gt_sel, fields=np.array(fields),
        sample_id=sample, t_idx=t, n_steps=n_steps,
        ckpt=str(ckpt_path), config=str(config_path),
        label_offset=int(args.label_offset),
        sanitize_inputs=bool(args.sanitize_inputs),
        n_states=int(n_states),
        state_indices=np.array(state_indices, dtype=np.int32),
        rmse=float(rmse), n_finite=int(n_fin), n_total=int(n_tot),
    )
    make_viz_png(viz_png, fields, gt_sel, pred_sel)

    end_ts = dt.datetime.now().isoformat(timespec="seconds")
    append_runlog(runlog, [
        f"RMSE={rmse} finite={n_fin}/{n_tot}",
        f"wrote {pred_npz}",
        f"wrote {viz_png}",
        f"wrote {badmask_npz}",
        f"[{end_ts}] END",
    ])

    ns, var = parse_ns_var_from_ckpt(ckpt_path)
    append_csv(metrics_csv, {
        "timestamp": end_ts,
        "ns": ns, "var": var,
        "ckpt": str(ckpt_path),
        "config": str(config_path),
        "run_dir": str(run_dir),
        "npz_dir": str(npz_dir),
        "sample_id": sample,
        "t_idx": t,
        "n_steps": n_steps,
        "fields": ",".join(fields),
        "label_offset": int(args.label_offset),
        "sanitize_inputs": bool(args.sanitize_inputs),
        "n_states": int(n_states),
        "rmse": rmse,
        "n_finite": n_fin,
        "n_total": n_tot,
        "pred_npz": str(pred_npz),
        "viz_png": str(viz_png),
        "badmask_npz": str(badmask_npz),
    })

    print(f"OK RMSE={rmse:.6g} finite={n_fin}/{n_tot}")
    print(f"out_dir: {out_dir}")
    print(f"metrics: {metrics_csv}")
    print(f"runlog:  {runlog}")

if __name__ == "__main__":
    main()
