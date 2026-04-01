#!/usr/bin/env python3
import argparse
import os
from typing import List, Optional, Tuple

import numpy as np
import torch

try:
    from utils.YParams import YParams
    from models.avit import build_avit
except Exception:
    from .utils.YParams import YParams
    from .models.avit import build_avit


DEFAULT_DATA_BASE = "/lustre/scratch5/exempt/artimis/mpmm/spandit/finetune"
DEFAULT_CKPT_BASE = "/lustre/scratch5/exempt/artimis/mpmm/spandit/runs-ch/mpp/final"

TEST_DIR_BY_DATASET = {
    "LSC": "lsc240420_fp16_full_direct_test",
    "CLX": "cx241203_fp16_full_direct_test",
}

CONFIG_DIR_BY_DTRAIN = {
    "dtrain_lsc": os.path.join(".", "config", "dtrain_lsc"),
    "dtrain_clx": os.path.join(".", "config", "dtrain_clx"),
}

OUTPUT_SUBDIR_BY_DATASET = {
    "LSC": "lsc-dtrain",
    "CLX": "clx-dtrain",
}

DEFAULT_LSC_FIELDS: List[str] = [
    "Uvelocity",
    "Wvelocity",
    "density_case",
    "density_cushion",
    "density_maincharge",
    "density_outside_air",
    "density_striker",
    "density_throw",
]


def ensure_file_exists(path: str, desc: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{desc} not found: {path}")


def ensure_dir_exists(path: str, desc: str) -> None:
    if not os.path.isdir(path):
        raise NotADirectoryError(f"{desc} not found or not a directory: {path}")


def parse_fields(fields_csv: str) -> List[str]:
    fields = [f.strip() for f in fields_csv.split(",") if f.strip()]
    if not fields:
        raise ValueError("--fields produced an empty field list")
    return fields


def print_args(args: argparse.Namespace) -> None:
    print("🧾 Parsed command-line arguments (including defaults):")
    for k, v in sorted(vars(args).items()):
        print(f"  {k}: {v}")


def finite_min_max(t: torch.Tensor) -> Tuple[Optional[float], Optional[float], int]:
    td = t.detach()
    finite = torch.isfinite(td)
    n_finite = int(finite.sum().item())
    if n_finite == 0:
        return None, None, 0
    vals = td[finite]
    return float(vals.min().item()), float(vals.max().item()), n_finite


def tensor_sanity(name: str, t: torch.Tensor) -> None:
    td = t.detach()
    finite = torch.isfinite(td)
    n_bad = int((~finite).sum().item())
    tmin, tmax, n_fin = finite_min_max(td)
    if tmin is None:
        minmax_str = "min=NA max=NA (no finite entries)"
    else:
        minmax_str = f"min={tmin:.6g} max={tmax:.6g} (finite n={n_fin})"
    print(
        f"🔎 {name}: shape={tuple(td.shape)} dtype={td.dtype} device={td.device} "
        f"{minmax_str} bad(nan/inf)={n_bad}"
    )
    if n_bad > 0:
        idx = (~finite).nonzero(as_tuple=False)[0].tolist()
        val = td[tuple(idx)].item()
        print(f"   first bad at index {idx} value={val}")


def canonical_dataset_name(dataset_arg: str) -> str:
    d = dataset_arg.strip().lower()
    if d == "lsc":
        return "LSC"
    if d in ("clx", "cx"):
        return "CLX"
    raise ValueError(f"Unsupported --dataset '{dataset_arg}'. Expected one of: lsc, clx, cx.")


def dtrain_dir_from_canonical_dataset(dataset_canonical: str) -> str:
    if dataset_canonical == "LSC":
        return "dtrain_lsc"
    if dataset_canonical == "CLX":
        return "dtrain_clx"
    raise ValueError(f"Unknown canonical dataset '{dataset_canonical}'")


def config_dir_from_dataset(dataset_canonical: str) -> str:
    return CONFIG_DIR_BY_DTRAIN[dtrain_dir_from_canonical_dataset(dataset_canonical)]


def config_filename_from(dataset_canonical: str, n_steps: int) -> str:
    if not (1 <= n_steps <= 16):
        raise ValueError(f"--n_steps must be in [1,16], got {n_steps}")
    return f"mpp_avit_L_dtrain-{dataset_canonical}_lr-X_opt-adan_wd-3_ns-{n_steps:02d}.yaml"


def run_dirname_from(dataset_canonical: str, n_steps: int) -> str:
    if not (1 <= n_steps <= 16):
        raise ValueError(f"--n_steps must be in [1,16], got {n_steps}")
    return f"final_L_dtrain-{dataset_canonical}_lr-X_opt-adan_wd-3_ns-{n_steps:02d}"


def auto_config_path(dataset_canonical: str, n_steps: int) -> str:
    cfg_dir = config_dir_from_dataset(dataset_canonical)
    return os.path.join(cfg_dir, config_filename_from(dataset_canonical, n_steps))


def auto_npz_dir_from_dataset(data_base: str, dataset_canonical: str) -> str:
    if dataset_canonical == "LSC":
        return os.path.join(data_base, TEST_DIR_BY_DATASET["LSC"])
    return os.path.join(data_base, TEST_DIR_BY_DATASET["CLX"])


def auto_ckpt_path(ckpt_base: str, dataset_canonical: str, n_steps: int) -> str:
    dtrain_dir = dtrain_dir_from_canonical_dataset(dataset_canonical)
    run_dir = run_dirname_from(dataset_canonical, n_steps)
    return os.path.join(
        ckpt_base,
        dtrain_dir,
        "basic_config",
        run_dir,
        "training_checkpoints",
        "best_ckpt.tar",
    )


def default_prefix_template(dataset_canonical: str) -> str:
    if dataset_canonical == "LSC":
        return "lsc240420_id{sim_id:05d}_pvi"
    return "cx241203_id{sim_id:05d}_pvi"


def load_npz_fields(path: str, fields: List[str]) -> torch.Tensor:
    """Return [C,H,W] tensor (float32) for selected fields only."""
    with np.load(path) as data:
        missing = [k for k in fields if k not in data]
        if missing:
            raise KeyError(f"Missing keys {missing} in {path}. Available keys: {list(data.keys())[:20]}...")
        arrays = [np.array(data[k], dtype=np.float16) for k in fields]
    return torch.from_numpy(np.stack(arrays, axis=0)).to(torch.float32)


def sanitize_inplace(t: torch.Tensor, fill_value: float = 0.0) -> torch.Tensor:
    """Replace NaN/Inf with fill_value."""
    finite = torch.isfinite(t)
    if not finite.all():
        t = t.clone()
        t[~finite] = fill_value
    return t


def build_input_stack_paths(pred_idx: int, n_steps: int, npz_dir: str, sample_prefix: str) -> List[str]:
    return [os.path.join(npz_dir, f"{sample_prefix}_idx{t:05d}.npz") for t in range(pred_idx - n_steps, pred_idx)]


def ground_truth_path(pred_idx: int, npz_dir: str, sample_prefix: str) -> str:
    return os.path.join(npz_dir, f"{sample_prefix}_idx{pred_idx:05d}.npz")


def output_dir_for_dataset(out_dir: str, dataset_canonical: str) -> str:
    sub = OUTPUT_SUBDIR_BY_DATASET.get(dataset_canonical, dataset_canonical.lower())
    return os.path.join(out_dir, sub)


def build_output_path(out_dir: str, dataset_canonical: str, sample_prefix: str, pred_idx: int, n_steps: int) -> str:
    out_dir2 = output_dir_for_dataset(out_dir, dataset_canonical)
    os.makedirs(out_dir2, exist_ok=True)
    return os.path.join(out_dir2, f"pred_{sample_prefix}_idx{pred_idx:05d}_ctx{n_steps:02d}_PRED.npz")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prediction script for finetuned AViT (dry-run first).")

    p.add_argument("--dataset", required=True, choices=["lsc", "clx", "cx"])
    p.add_argument("--n_steps", required=True, type=int, choices=list(range(1, 17)))

    p.add_argument("--sim_id", type=int, default=None)
    p.add_argument("--sample_prefix", default=None)
    p.add_argument("--prefix_template", default=None)

    p.add_argument("--pred_tstep", type=int, required=True, help="0..100")

    p.add_argument("--data_base", default=DEFAULT_DATA_BASE)
    p.add_argument("--npz_dir", default=None)

    p.add_argument("--ckpt_base", default=DEFAULT_CKPT_BASE)
    p.add_argument("--ckpt", default=None)

    p.add_argument("--fields", default=",".join(DEFAULT_LSC_FIELDS))
    p.add_argument("--label_offset", type=int, default=0)

    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir", default="./predictions")

    p.add_argument("--dry_run", action="store_true")
    p.add_argument(
        "--sanitize_inputs",
        action="store_true",
        help="Replace NaN/Inf in inputs/ground-truth selected fields with 0 before inference/RMSE.",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    print_args(args)

    if not (0 <= args.pred_tstep <= 100):
        raise ValueError(f"--pred_tstep must be in [0,100], got {args.pred_tstep}")
    if args.pred_tstep < args.n_steps:
        raise ValueError(f"--pred_tstep must be >= n_steps ({args.n_steps})")

    dataset_canonical = canonical_dataset_name(args.dataset)
    config_block = "basic_config"

    cfg_dir = config_dir_from_dataset(dataset_canonical)
    config_path = auto_config_path(dataset_canonical, args.n_steps)

    if args.npz_dir is None:
        args.npz_dir = auto_npz_dir_from_dataset(args.data_base, dataset_canonical)

    if args.ckpt is None:
        args.ckpt = auto_ckpt_path(args.ckpt_base, dataset_canonical, args.n_steps)

    if args.sample_prefix is None:
        if args.sim_id is None:
            raise ValueError("Provide either --sample_prefix or --sim_id")
        tpl = args.prefix_template if args.prefix_template else default_prefix_template(dataset_canonical)
        args.sample_prefix = tpl.format(sim_id=int(args.sim_id))

    ensure_dir_exists(cfg_dir, "config_dir (deduced)")
    ensure_file_exists(config_path, "config yaml (deduced)")
    ensure_dir_exists(args.npz_dir, "npz_dir")
    ensure_file_exists(args.ckpt, "checkpoint (best_ckpt.tar)")

    fields = parse_fields(args.fields)

    input_paths = build_input_stack_paths(args.pred_tstep, args.n_steps, args.npz_dir, args.sample_prefix)
    gt_path = ground_truth_path(args.pred_tstep, args.npz_dir, args.sample_prefix)
    for pth in input_paths:
        ensure_file_exists(pth, "input npz")
    ensure_file_exists(gt_path, "ground truth npz")

    # Read params (basic_config only)
    params = YParams(config_path, config_block)
    params.n_steps = int(args.n_steps)
    n_states = int(getattr(params, "n_states", len(fields)))

    intended_out_path = build_output_path(args.out_dir, dataset_canonical, args.sample_prefix, args.pred_tstep, args.n_steps)

    # Preview y_true for selected fields only
    y_true_sel = load_npz_fields(gt_path, fields)  # [Csel,H,W]
    Csel = int(y_true_sel.shape[0])

    print("\n✅ DRY-RUN VALIDATION SUMMARY")
    print(f"  dataset (canonical): {dataset_canonical}")
    print(f"  n_steps:            {args.n_steps:02d}")
    print(f"  config_dir:         {cfg_dir}")
    print(f"  config:             {config_path} (block: {config_block})")
    print(f"  run_dir:            {run_dirname_from(dataset_canonical, args.n_steps)}")
    print(f"  npz_dir:            {args.npz_dir}")
    print(f"  ckpt:               {args.ckpt}")
    print(f"  sample_prefix:      {args.sample_prefix}")
    print(f"  pred idx:           idx{args.pred_tstep:05d}")
    print(f"  selected y shape:   {tuple(y_true_sel.shape)}  # [Csel,H,W]")
    print(f"  basic_config n_states: {n_states}")

    print("\n📦 Intended output:")
    print(f"  out_dir:      {args.out_dir}")
    print(f"  out_subdir:   {output_dir_for_dataset(args.out_dir, dataset_canonical)}")
    print(f"  out_file:     {os.path.basename(intended_out_path)}")
    print(f"  out_path:     {intended_out_path}")

    if args.dry_run:
        print("\n🟡 Dry run requested; exiting before inference.")
        return

    device = torch.device(args.device)

    # Build model correctly from params
    model = build_avit(params).to(device)
    model.eval()

    ckpt = torch.load(args.ckpt, map_location=device)
    if "model_state" not in ckpt:
        raise KeyError(f"Checkpoint missing 'model_state'. Keys: {list(ckpt.keys())}")

    try:
        model.load_state_dict(ckpt["model_state"], strict=True)
    except RuntimeError:
        stripped = {k.replace("module.", "", 1): v for k, v in ckpt["model_state"].items()}
        model.load_state_dict(stripped, strict=True)

    # Build FULL-state inputs for the model: [T,1,n_states,H,W]
    def sel_to_full(t_sel: torch.Tensor) -> torch.Tensor:
        # t_sel is [Csel,H,W]
        H, W = int(t_sel.shape[1]), int(t_sel.shape[2])
        t_full = torch.zeros((n_states, H, W), dtype=t_sel.dtype)
        # Place selected channels in first Csel slots
        t_full[:Csel] = t_sel
        return t_full

    x_sel_list = [load_npz_fields(pth, fields) for pth in input_paths]  # each [Csel,H,W]
    y_true_sel = load_npz_fields(gt_path, fields)                       # [Csel,H,W]

    if args.sanitize_inputs:
        x_sel_list = [sanitize_inplace(x, 0.0) for x in x_sel_list]
        y_true_sel = sanitize_inplace(y_true_sel, 0.0)

    x_full_list = [sel_to_full(x) for x in x_sel_list]                  # each [n_states,H,W]
    x = torch.stack(x_full_list, dim=0).unsqueeze(1).to(device)         # [T,1,n_states,H,W]

    # y_true_full is only for sanity; RMSE computed on selected only
    y_true_sel = y_true_sel.to(device)                                  # [Csel,H,W]

    # Labels should match n_states (NOT Csel)
    labels_1d = (torch.arange(n_states, device=device, dtype=torch.long) + int(args.label_offset))
    state_labels = [labels_1d]
    bcs = torch.zeros(1, 2, device=device)

    with torch.no_grad():
        y_pred_full = model(x, state_labels, bcs)

        # Accept [n_states,H,W] or [1,n_states,H,W]
        if y_pred_full.dim() == 4 and y_pred_full.shape[0] == 1:
            y_pred_full = y_pred_full[0]
        if y_pred_full.dim() != 3:
            raise RuntimeError(f"Unexpected output shape {tuple(y_pred_full.shape)}; expected [n_states,H,W] or [1,n_states,H,W].")

    # Slice back to selected channels for RMSE + saving
    y_pred_sel = y_pred_full[:Csel]

    # Diagnostics for NaNs/Infs (selected only, per your rule)
    tensor_sanity("y_true_selected", y_true_sel)
    tensor_sanity("y_pred_selected", y_pred_sel)

    diff = y_true_sel - y_pred_sel
    mask = torch.isfinite(diff)
    if not mask.any():
        print("⚠️ No finite entries in diff (selected fields). RMSE will be NaN.")
        rmse = float("nan")
    else:
        if not mask.all():
            n_bad = int((~mask).sum().item())
            print(f"⚠️ diff(selected) has {n_bad} non-finite entries; computing RMSE over finite entries only")
        rmse = float(torch.sqrt(torch.mean((diff[mask]) ** 2)).item())

    print(f"\n📊 RMSE vs ground truth (selected fields only): {rmse:.6f}")

    # Save ONLY selected fields
    y_arr = y_pred_sel.detach().cpu().numpy()
    np.savez(intended_out_path, **{fields[i]: y_arr[i] for i in range(Csel)})
    print(f"💾 Saved prediction: {intended_out_path}")

    pred_bad = (~torch.isfinite(y_pred_sel)).detach().cpu().numpy()
    if pred_bad.any():
        badmask_path = intended_out_path.replace(".npz", "_badmask.npy")
        np.save(badmask_path, pred_bad)
        print(f"💾 Saved badmask (selected): {badmask_path}")


if __name__ == "__main__":
    main()

