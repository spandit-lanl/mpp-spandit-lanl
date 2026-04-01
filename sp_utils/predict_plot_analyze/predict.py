#!/usr/bin/env python3
"""
Predict a single timestep from prior timesteps using AViT and log/visualize results.

This script:
  1) Validates required config/ckpt/input files
  2) Builds a stacked input tensor from N prior timesteps
  3) Loads the AViT model, runs inference for PREDICT_TIMESTEP
  4) Saves the prediction .npz, logs RMSE to CSV, and writes a PNG comparison plot

CLI examples:
  python predict.py --pred_tstep 94 --n_steps 2
  python predict.py --pred_tstep 100 --n_steps 10 --dry_run
"""

import os
import sys
import torch
import numpy as np
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple

# ------------------------------------------------------------
# Ensure project root (e.g., mpp-spandit-npz/) is in sys.path
# ------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.avit import build_avit
from utils.YParams import YParams

# ------------------------------------------------------------
# Static configuration
# ------------------------------------------------------------
# Input directory for .npz files (unchanged)
NPZ_DIR = '/lustre/scratch5/exempt/artimis/data/lsc240420/'

# Fields to use from each .npz file
SELECTED_FIELDS: List[str] = [
    'Uvelocity', 'Wvelocity', 'density_case', 'density_cushion',
    'density_maincharge', 'density_outside_air', 'density_striker', 'density_throw'
]


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def ensure_file_exists(path: str, label: str = "file") -> None:
    """Raise if path does not exist; print a friendly checkmark otherwise."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Required {label} not found: {path}")
    print(f"✅ Required {label} found: {path}")


def _npz_to_tensor(npz_path: str, fields: List[str]) -> torch.Tensor:
    """Load a set of fields from an .npz file, sanitize NaNs/inf, and stack into [C, H, W] tensor."""
    with np.load(npz_path) as data:
        arrays = [
            np.nan_to_num(
                data[key].astype(np.float32),
                nan=0.0, posinf=0.0, neginf=0.0
            ) for key in fields
        ]
        return torch.tensor(np.stack(arrays, axis=0), dtype=torch.float32)


def _build_input_tensors(
    predict_idx: int,
    n_steps: int,
    fields: List[str],
    npz_dir: str,
    dry_run: bool
) -> List[torch.Tensor]:
    """Collect input tensors for offsets [predict_idx - n_steps, ..., predict_idx-1]."""
    inputs: List[torch.Tensor] = []
    for offset in range(predict_idx - n_steps, predict_idx):
        fname = f"lsc240420_id05300_pvi_idx{offset:05d}.npz"
        fpath = os.path.join(npz_dir, fname)
        print(f"📥 Validating input: {fname}")
        ensure_file_exists(fpath, "input .npz")
        if not dry_run:
            inputs.append(_npz_to_tensor(fpath, fields))
    return inputs


def _load_ground_truth(predict_idx: int, fields: List[str], npz_dir: str) -> Tuple[str, torch.Tensor]:
    """Load ground-truth tensor for predict_idx and return (path, tensor)."""
    target_fname = f"lsc240420_id05300_pvi_idx{predict_idx:05d}.npz"
    target_path = os.path.join(npz_dir, target_fname)
    print(f"📥 Validating ground truth: {target_fname}")
    ensure_file_exists(target_path, "ground truth .npz")
    y_true = _npz_to_tensor(target_path, fields)
    return target_path, y_true


def _save_prediction(fields: List[str], y_pred: torch.Tensor, out_dir: str, predict_idx: int, n_steps_2d: str) -> str:
    """Save prediction tensor as .npz (keys match SELECTED_FIELDS); return saved path."""
    output_fname = f"lsc240420_id05300_pvi_idx{predict_idx:05d}_nstep_{n_steps_2d}_PRED.npz"
    output_path = os.path.join(out_dir, output_fname)
    np.savez(output_path, **{key: y_pred[i].cpu().numpy() for i, key in enumerate(fields)})
    print(f"✅ Saved prediction to: {output_path}")
    return output_path


def _compute_rmse(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Compute RMSE between prediction and ground truth."""
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item()


def _append_log_csv(
    log_csv_path: str,
    predict_idx: int,
    n_steps: int,
    rmse: float,
    ckpt_path: str,
    config_path: str,
    output_path: str
) -> None:
    """Append a single-row RMSE log to CSV (create if missing)."""
    log_entry = pd.DataFrame([{
        'pred_tstep': predict_idx,
        'n_steps': n_steps,
        'rmse': rmse,
        'checkpoint': ckpt_path,
        'config': config_path,
        'output_file': output_path
    }])

    if os.path.exists(log_csv_path):
        log_entry.to_csv(log_csv_path, mode='a', header=False, index=False)
    else:
        log_entry.to_csv(log_csv_path, index=False)

    print(f"📝 Logged RMSE to {log_csv_path}")


def _plot_results(
    fields: List[str],
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    out_dir: str,
    predict_idx: int,
    n_steps_2d: str
) -> str:
    """Plot GT, Pred, |Error| per channel and save PNG; return plot path."""
    fig, axes = plt.subplots(3, len(fields), figsize=(3 * len(fields), 9))

    for i, name in enumerate(fields):
        # Row 0: Ground Truth
        axes[0, i].imshow(y_true[i].cpu(), cmap='viridis')
        axes[0, i].set_title(name, fontsize=10)
        axes[0, i].axis('off')

        # Row 1: Prediction
        axes[1, i].imshow(y_pred[i].cpu(), cmap='viridis')
        axes[1, i].axis('off')

        # Row 2: Absolute error
        diff = torch.abs(y_pred[i] - y_true[i])
        axes[2, i].imshow(diff.cpu(), cmap='hot')
        axes[2, i].axis('off')

    row_labels = ["Ground Truth", "Predicted", "Absolute Error"]
    for row_idx, label in enumerate(row_labels):
        fig.text(0.01, 0.92 - 0.31 * row_idx, label, va='top', ha='left', fontsize=14, rotation=90)

    plt.tight_layout(rect=[0.05, 0, 1, 1])

    plot_path = os.path.join(
        out_dir,
        f"lsc240420_id05300_pvi_idx{predict_idx:05d}_nsteps{n_steps_2d}_PRED_vs_GT.png"
    )
    plt.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"📷 Saved visualization to: {plot_path}")
    return plot_path


# ------------------------------------------------------------
# Argument parsing (kept identical to your original UX)
# ------------------------------------------------------------
def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='predict.py',
        description="Run MPP prediction for a given timestep.",
        add_help=False,
        formatter_class=argparse.RawTextHelpFormatter
    )

    # "optional arguments" group (help)
    opt_group = parser.add_argument_group('optional arguments:')
    opt_group.add_argument('-h', '--help', action='help',
                           help='Show this help message and exit.')

    # "required arguments" group
    req_group = parser.add_argument_group('required arguments:')
    req_group.add_argument('--pred_tstep', type=int, required=True,
                           metavar='PREDICT_TIMESTEP',
                           help='Timestep to predict (e.g., 94).')
    req_group.add_argument('--n_steps', type=int, required=True,
                           metavar='N_STEPS',
                           help='Number of prior timesteps to use (e.g., 2).')

    # "optional flags" group
    flags_group = parser.add_argument_group('optional flags:')
    flags_group.add_argument('--dry_run', action='store_true',
                             help=('Only validate file existence; do not run inference.\n'
                                   '        Useful for checking all required input and configuration files.'))
    return parser


# ------------------------------------------------------------
# Main entry
# ------------------------------------------------------------
def main() -> None:
    # Parse CLI
    args = _build_arg_parser().parse_args()
    PREDICT_IDX: int = args.pred_tstep
    N_STEPS: int = args.n_steps
    DRY_RUN: bool = args.dry_run

    # Basic argument validation
    if N_STEPS <= 0:
        raise ValueError(f"--n_steps must be > 0; got {N_STEPS}")
    if PREDICT_IDX < N_STEPS:
        raise ValueError(
            f"--pred_tstep ({PREDICT_IDX}) must be >= --n_steps ({N_STEPS}) "
            "so there are enough prior frames."
        )

    # Two-digit, zero-padded steps string for 1..9; >=10 stays as-is
    N_STEPS_2D = f"{N_STEPS:02d}"

    # Dynamic paths (use two-digit N_STEPS_2D consistently)
    CONFIG_PATH = f'./config/mpp_lsc_avit_ti_config_nsteps_{N_STEPS_2D}.yaml'
    CKPT_PATH = f'./mpp-runs/basic_config/lsc240420_nsteps_{N_STEPS_2D}/training_checkpoints/best_ckpt.tar'
    OUT_DIR = f'./mpp-pred/basic_config/lsc240420_nsteps_{N_STEPS_2D}'
    os.makedirs(OUT_DIR, exist_ok=True)

    LOG_PATH = os.path.join(OUT_DIR, f'prediction_rmse_log_nsteps{N_STEPS_2D}.csv')

    # Config & checkpoint presence
    ensure_file_exists(CONFIG_PATH, "config")
    ensure_file_exists(CKPT_PATH, "checkpoint")

    # Build inputs and load ground truth (also presence-check every input file)
    input_tensors = _build_input_tensors(PREDICT_IDX, N_STEPS, SELECTED_FIELDS, NPZ_DIR, DRY_RUN)
    target_path, y_true = _load_ground_truth(PREDICT_IDX, SELECTED_FIELDS, NPZ_DIR)

    if DRY_RUN:
        print("✅ DRY RUN complete. All files exist.")
        return

    # -------------------------------
    # Load Model
    # -------------------------------
    params = YParams(CONFIG_PATH, 'basic_config')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Using device: {device}")

    model = build_avit(params).to(device)
    model.eval()

    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    print(f"✅ Model loaded from: {CKPT_PATH}")

    # -------------------------------
    # Prepare inputs
    # -------------------------------
    # Stack to [T, 1, C, H, W] then send to device
    x = torch.stack(input_tensors, dim=0).unsqueeze(1).to(device)
    # Labels & bcs mirror your original setup
    labels = torch.arange(x.shape[2]).unsqueeze(0).to(device)
    bcs = torch.zeros(1, 2).to(device)

    # Align ground truth to device
    y_true = y_true.to(device)

    # -------------------------------
    # Run prediction
    # -------------------------------
    with torch.no_grad():
        y_pred = model(x, labels, bcs)[0]

    # -------------------------------
    # Save, log, and plot
    # -------------------------------
    output_path = _save_prediction(SELECTED_FIELDS, y_pred, OUT_DIR, PREDICT_IDX, N_STEPS_2D)

    rmse = _compute_rmse(y_true, y_pred)
    print(f"📊 RMSE vs ground truth: {rmse:.4f}")

    _append_log_csv(LOG_PATH, PREDICT_IDX, N_STEPS, rmse, CKPT_PATH, CONFIG_PATH, output_path)

    _plot_results(SELECTED_FIELDS, y_true, y_pred, OUT_DIR, PREDICT_IDX, N_STEPS_2D)


if __name__ == "__main__":
    main()

