#!/usr/bin/env python3
"""
predict_lsc_timestep.py

Predict a single LSC timestep from the immediately preceding timesteps using a finetuned AViT checkpoint.

Expected LSC filenames:
  <sample_prefix>_idx00098.npz
Example:
  lsc240420_id00647_pvi_idx00098.npz
In that case, sample_prefix="lsc240420_id00647_pvi".

Typical usage (resume-style config recommended for finetuned checkpoints):
  python predict_lsc_timestep.py \
    --config config_spandit/mpp_avit_b_config_nsteps_01.yaml \
    --config_block finetune_resume \
    --ckpt /users/spandit/projects/artimis/mpp/mpp-spandit-npz/mpp-runs-finetune/finetune/finetune_b_LSC_nsteps_01/training_checkpoints/best_ckpt.tar \
    --npz_dir /lustre/scratch5/exempt/artimis/data/mpp_finetune_on_lsc/test_10_percent_lsc \
    --sample_prefix lsc240420_id00647_pvi \
    --pred_tstep 98 \
    --n_steps 1 \
    --out_dir ./predictions/finetune_b_LSC_nsteps_01

Notes:
- This script assumes each .npz contains keys matching the selected fields.
- If your training used a label offset (e.g., embedding_offset=12), pass --label_offset 12.
"""

import argparse
import os
from typing import List, Tuple

import numpy as np
import torch

from YParams import YParams
from avit import AViT


# -----------------------------
# Helpers
# -----------------------------
DEFAULT_FIELDS: List[str] = [
    # Keep these aligned with how your LSC .npz files are structured.
    # You can override via --fields.
    "vf", "pf", "ef", "nvf",
]


def ensure_file_exists(path: str, desc: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{desc} not found: {path}")


def parse_fields(fields_csv: str) -> List[str]:
    fields = [f.strip() for f in fields_csv.split(",") if f.strip()]
    if not fields:
        raise ValueError("--fields produced an empty field list")
    return fields


def load_npz_fields(path: str, fields: List[str]) -> torch.Tensor:
    """
    Loads fields from a single .npz as a torch Tensor with shape [C, H, W].
    """
    with np.load(path) as data:
        missing = [k for k in fields if k not in data]
        if missing:
            raise KeyError(f"Missing keys {missing} in {path}. Available keys: {list(data.keys())[:20]}...")
        #arrays = [np.array(data[k], dtype=np.float32) for k in fields]
        arrays = [np.array(data[k], dtype=np.float16) for k in fields]  # load fp16 from disk
    #return torch.from_numpy(np.stack(arrays, axis=0))
    return torch.from_numpy(np.stack(arrays, axis=0)).to(torch.float32)  # model input fp32 (Option A)


def build_input_stack(
    pred_idx: int,
    n_steps: int,
    fields: List[str],
    npz_dir: str,
    sample_prefix: str,
) -> Tuple[List[str], torch.Tensor]:
    """
    Builds x of shape [T, 1, C, H, W] from timesteps [pred_idx-n_steps, ..., pred_idx-1].
    Returns (paths_used, x_tensor).
    """
    paths_used: List[str] = []
    tensors: List[torch.Tensor] = []
    for t in range(pred_idx - n_steps, pred_idx):
        fname = f"{sample_prefix}_idx{t:05d}.npz"
        fpath = os.path.join(npz_dir, fname)
        ensure_file_exists(fpath, "input npz")
        paths_used.append(fpath)
        tensors.append(load_npz_fields(fpath, fields))

    # [T, C, H, W] -> [T, 1, C, H, W]
    x = torch.stack(tensors, dim=0).unsqueeze(1)
    return paths_used, x


def load_ground_truth(
    pred_idx: int,
    fields: List[str],
    npz_dir: str,
    sample_prefix: str,
) -> Tuple[str, torch.Tensor]:
    fname = f"{sample_prefix}_idx{pred_idx:05d}.npz"
    fpath = os.path.join(npz_dir, fname)
    ensure_file_exists(fpath, "ground truth npz")
    y = load_npz_fields(fpath, fields)
    return fpath, y


def safe_load_state_dict(model: torch.nn.Module, state: dict) -> None:
    """
    Loads a state dict, stripping 'module.' prefix if needed.
    """
    try:
        model.load_state_dict(state, strict=True)
        return
    except RuntimeError as e:
        # Try stripping 'module.' prefix ONLY when present
        stripped = {}
        changed = False
        for k, v in state.items():
            if k.startswith("module."):
                stripped[k[len("module."):]] = v
                changed = True
            else:
                stripped[k] = v
        if changed:
            model.load_state_dict(stripped, strict=True)
            return
        raise e


def compute_rmse(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item())


def save_prediction_npz(
    out_dir: str,
    sample_prefix: str,
    pred_idx: int,
    n_steps: int,
    fields: List[str],
    y_pred: torch.Tensor,
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"pred_{sample_prefix}_idx{pred_idx:05d}_ctx{n_steps:02d}.npz")
    arr = y_pred.detach().cpu().numpy()
    np.savez(out_path, **{fields[i]: arr[i] for i in range(len(fields))})
    return out_path


# -----------------------------
# Main
# -----------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Predict a single LSC timestep using a finetuned AViT checkpoint.")

    p.add_argument("--config", required=True, help="Path to YAML config file.")
    p.add_argument("--config_block", default="finetune_resume",
                   help="Config block name inside YAML (e.g., finetune, finetune_resume). Default: finetune_resume")

    p.add_argument("--ckpt", required=True, help="Path to checkpoint (.tar), e.g. best_ckpt.tar")
    p.add_argument("--npz_dir", required=True, help="Directory containing LSC .npz files.")
    p.add_argument("--sample_prefix", default=None,
                   help=("Prefix before _idx in filenames, e.g. lsc240420_id00647_pvi. "
                         "If omitted, it will be constructed from --sim_id and --prefix_template."))

    p.add_argument("--sim_id", type=int, default=None,
                   help=("Simulation ID (e.g., 647 for id00647). Used to construct sample_prefix "
                         "if --sample_prefix is not provided."))

    p.add_argument("--prefix_template", default="lsc240420_id{sim_id:05d}_pvi",
                   help=("Template to construct sample_prefix from sim_id. "
                         "Default: lsc240420_id{sim_id:05d}_pvi"))

    p.add_argument("--pred_tstep", type=int, required=True, help="Timestep index to predict (e.g., 98).")
    p.add_argument("--n_steps", type=int, required=True, help="Number of prior timesteps to use (1..16).")

    p.add_argument("--fields", default=",".join(DEFAULT_FIELDS),
                   help="Comma-separated field keys to load from .npz (default: vf,pf,ef,nvf)")

    p.add_argument("--label_offset", type=int, default=0,
                   help=("Offset added to field labels (if training used embedding_offset). "
                         "Example: --label_offset 12"))

    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                   help="Device: cuda or cpu (default: cuda if available).")

    p.add_argument("--out_dir", default="./predictions",
                   help="Output directory for predicted .npz. (default: ./predictions)")

    p.add_argument("--dry_run", action="store_true",
                   help="Only validate paths and print what would run; do not load model or predict.")

    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    # Resolve sample_prefix
    if args.sample_prefix is None:
        if args.sim_id is None:
            raise ValueError('Provide either --sample_prefix or --sim_id')
        try:
            args.sample_prefix = args.prefix_template.format(sim_id=int(args.sim_id))
        except Exception as e:
            raise ValueError(f'Failed to construct sample_prefix from --sim_id and --prefix_template: {e}')

    # Validate basic inputs
    if args.n_steps <= 0:
        raise ValueError("--n_steps must be > 0")
    if args.pred_tstep < args.n_steps:
        raise ValueError("--pred_tstep must be >= --n_steps so there are enough prior frames")

    ensure_file_exists(args.config, "config yaml")
    ensure_file_exists(args.ckpt, "checkpoint")

    fields = parse_fields(args.fields)

    # Validate required npz inputs exist
    in_paths, _ = build_input_stack(args.pred_tstep, args.n_steps, fields, args.npz_dir, args.sample_prefix)
    gt_path, _ = load_ground_truth(args.pred_tstep, fields, args.npz_dir, args.sample_prefix)

    print("✅ Inputs validated")
    print(f"  config:  {args.config} (block: {args.config_block})")
    print(f"  ckpt:    {args.ckpt}")
    print(f"  npz_dir: {args.npz_dir}")
    print(f"  sample:  {args.sample_prefix}")
    print(f"  predict: idx{args.pred_tstep:05d} using ctx={args.n_steps}")
    print(f"  fields:  {fields}")
    print(f"  label_offset: {args.label_offset}")
    print("  input files:")
    for pth in in_paths:
        print(f"    - {pth}")
    print(f"  ground truth: {gt_path}")

    if args.dry_run:
        print("🟡 Dry run requested; exiting before inference.")
        return

    # Load params + build model
    params = YParams(args.config, args.config_block)
    device = torch.device(args.device)

    model = AViT(params).to(device)
    model.eval()

    ckpt = torch.load(args.ckpt, map_location=device)
    if "model_state" not in ckpt:
        raise KeyError(f"Checkpoint missing 'model_state' key. Keys present: {list(ckpt.keys())}")
    safe_load_state_dict(model, ckpt["model_state"])
    print("✅ Model weights loaded")

    # Prepare tensors
    _, x = build_input_stack(args.pred_tstep, args.n_steps, fields, args.npz_dir, args.sample_prefix)
    _, y_true = load_ground_truth(args.pred_tstep, fields, args.npz_dir, args.sample_prefix)

    x = x.to(device)  # [T,1,C,H,W]
    y_true = y_true.to(device)  # [C,H,W]

    # Labels: one label per channel/field
    C = x.shape[2]
    labels = (torch.arange(C, device=device) + int(args.label_offset)).unsqueeze(0)  # [1,C]
    bcs = torch.zeros(1, 2, device=device)

    # Forward: expect output [1, C, H, W] for next-step
    with torch.no_grad():
        y_pred = model(x, labels, bcs)  # [1,C,H,W] in your training code
        if y_pred.dim() == 4 and y_pred.shape[0] == 1:
            y_pred = y_pred[0]  # [C,H,W]

    rmse = compute_rmse(y_true, y_pred)
    print(f"📊 RMSE vs ground truth: {rmse:.6f}")

    out_path = save_prediction_npz(args.out_dir, args.sample_prefix, args.pred_tstep, args.n_steps, fields, y_pred)
    print(f"💾 Saved prediction: {out_path}")


if __name__ == "__main__":
    main()
