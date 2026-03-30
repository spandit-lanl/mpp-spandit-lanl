#!/usr/bin/env python3
"""
visualize_prediction_v2.py

Same intent as before, but fixes aspect/layout:
- uses constrained_layout (no tight_layout warning)
- uses aspect='equal' to preserve image geometry
- chooses figsize based on H/W so tall fields look tall (1120x400)
"""

from __future__ import annotations
import argparse, os
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt

def _decode_fields(arr) -> List[str]:
    fields = arr.tolist() if hasattr(arr, "tolist") else list(arr)
    out = []
    for f in fields:
        out.append(f.decode() if isinstance(f, (bytes, np.bytes_)) else str(f))
    return out

def load_pred_npz(path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    d = np.load(path, allow_pickle=True)
    return d["pred_sel"], d["gt_sel"], _decode_fields(d["fields"])

def robust_minmax(vals: np.ndarray, p_lo: float, p_hi: float) -> Tuple[float, float]:
    x = vals[np.isfinite(vals)]
    if x.size == 0:
        return 0.0, 1.0
    return float(np.percentile(x, p_lo)), float(np.percentile(x, p_hi))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="Root that contains dataset/ (e.g. final/pred)")
    ap.add_argument("--dataset", default="clx")
    ap.add_argument("--expt_id", required=True)
    ap.add_argument("--t_idx", type=int, required=True)
    ap.add_argument("--vars", default="3,6")
    ap.add_argument("--ns_min", type=int, default=1)
    ap.add_argument("--ns_max", type=int, default=8)
    ap.add_argument("--out_dir", default="viz")
    ap.add_argument("--dpi", type=int, default=170)

    ap.add_argument("--cmap_main", default="magma")
    ap.add_argument("--cmap_diff", default="seismic")
    ap.add_argument("--robust", action="store_true")
    ap.add_argument("--p_lo", type=float, default=1.0)
    ap.add_argument("--p_hi", type=float, default=99.0)
    ap.add_argument("--diff_p", type=float, default=99.0)
    ap.add_argument("--interpolation", default="nearest")

    args = ap.parse_args()

    vars_list = [int(x) for x in args.vars.split(",") if x.strip()]
    root = os.path.abspath(args.root)
    t_str = f"t{args.t_idx:05d}"

    out_dir = args.out_dir
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(root, out_dir, args.dataset, f"expt-{args.expt_id}", t_str)
    os.makedirs(out_dir, exist_ok=True)

    table: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
    fields_ref: List[str] = []

    for ns in range(args.ns_min, args.ns_max + 1):
        for var in vars_list:
            p = os.path.join(root, args.dataset, f"ns-{ns:02d}_var-{var}",
                             f"expt-{args.expt_id}", t_str, "pred.npz")
            if not os.path.exists(p):
                continue
            pred, gt, fields = load_pred_npz(p)
            if not fields_ref:
                fields_ref = fields
            elif fields != fields_ref:
                raise ValueError(f"Field list mismatch in {p}: {fields} vs {fields_ref}")
            table[(ns, var)] = (pred, gt)

    if not table:
        raise FileNotFoundError(f"No pred.npz found under {root}/{args.dataset}/... expt-{args.expt_id}/{t_str}")

    # Determine H/W for sizing
    pred0, gt0 = next(iter(table.values()))
    C, H, W = pred0.shape
    hw_ratio = H / max(W, 1)

    # Scaling
    vmin_main = [0.0]*C; vmax_main = [1.0]*C; diff_lim = [1.0]*C
    for ci in range(C):
        vals = []
        diffs = []
        for (ns, var), (pred, gt) in table.items():
            vals.append(pred[ci].astype(np.float32).ravel())
            vals.append(gt[ci].astype(np.float32).ravel())
            diffs.append((pred[ci]-gt[ci]).astype(np.float32).ravel())
        vals_flat = np.concatenate(vals)
        diffs_flat = np.concatenate(diffs)

        if args.robust:
            vmin, vmax = robust_minmax(vals_flat, args.p_lo, args.p_hi)
            vmin_main[ci], vmax_main[ci] = vmin, vmax
            absd = np.abs(diffs_flat[np.isfinite(diffs_flat)])
            diff_lim[ci] = float(np.percentile(absd, args.diff_p)) if absd.size else 1.0
        else:
            finite = vals_flat[np.isfinite(vals_flat)]
            vmin_main[ci] = float(finite.min()) if finite.size else 0.0
            vmax_main[ci] = float(finite.max()) if finite.size else 1.0
            absd = np.abs(diffs_flat[np.isfinite(diffs_flat)])
            diff_lim[ci] = float(absd.max()) if absd.size else 1.0

        if vmax_main[ci] <= vmin_main[ci]:
            vmax_main[ci] = vmin_main[ci] + 1e-6
        if diff_lim[ci] <= 0:
            diff_lim[ci] = 1e-6

    ns_values = list(range(args.ns_min, args.ns_max + 1))

    for ci, field in enumerate(fields_ref):
        # Layout: GT | Pred v3 | Pred v6 | Diff v3 | Diff v6
        ncols = 1 + len(vars_list) + len(vars_list)
        nrows = len(ns_values)

        # Choose subplot size driven by H/W
        cell_w = 2.8
        cell_h = cell_w * hw_ratio
        fig_w = cell_w * ncols
        fig_h = cell_h * nrows

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(fig_w, fig_h),
            constrained_layout=True,
            squeeze=False
        )
        fig.suptitle(f"{args.dataset} expt-{args.expt_id} {t_str}  field={field}", fontsize=14)

        for r, ns in enumerate(ns_values):
            # pick GT from any available var
            gt_img = None
            for var in vars_list:
                if (ns, var) in table:
                    gt_img = table[(ns, var)][1][ci]
                    break

            # col 0: GT
            ax = axes[r][0]
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0: ax.set_title("GT")
            ax.set_ylabel(f"ns={ns:02d}", rotation=0, labelpad=28, va="center")
            if gt_img is None:
                ax.text(0.5, 0.5, "missing", ha="center", va="center")
            else:
                ax.imshow(gt_img, cmap=args.cmap_main,
                          vmin=vmin_main[ci], vmax=vmax_main[ci],
                          aspect="equal", interpolation=args.interpolation)

            # Pred columns
            for j, var in enumerate(vars_list):
                axp = axes[r][1+j]
                axp.set_xticks([]); axp.set_yticks([])
                if r == 0: axp.set_title(f"Pred v{var}")
                if (ns, var) not in table:
                    axp.text(0.5, 0.5, "missing", ha="center", va="center")
                else:
                    pred_img = table[(ns, var)][0][ci]
                    axp.imshow(pred_img, cmap=args.cmap_main,
                               vmin=vmin_main[ci], vmax=vmax_main[ci],
                               aspect="equal", interpolation=args.interpolation)

            # Diff columns
            for j, var in enumerate(vars_list):
                axd = axes[r][1+len(vars_list)+j]
                axd.set_xticks([]); axd.set_yticks([])
                if r == 0: axd.set_title(f"Diff v{var}")
                if gt_img is None or (ns, var) not in table:
                    axd.text(0.5, 0.5, "missing", ha="center", va="center")
                else:
                    pred_img = table[(ns, var)][0][ci]
                    diff = (pred_img - gt_img).astype(np.float32)
                    lim = diff_lim[ci]
                    axd.imshow(diff, cmap=args.cmap_diff,
                               vmin=-lim, vmax=lim,
                               aspect="equal", interpolation=args.interpolation)

        # Colorbars (one for main, one for diff)
        # main: use first axis image
        m_main = None
        for r in range(nrows):
            if axes[r][0].images:
                m_main = axes[r][0].images[0]
                break
        if m_main is not None:
            fig.colorbar(m_main, ax=axes[:, :1+len(vars_list)].ravel().tolist(), shrink=0.6, location="right")

        m_diff = None
        for r in range(nrows):
            if axes[r][1+len(vars_list)].images:
                m_diff = axes[r][1+len(vars_list)].images[0]
                break
        if m_diff is not None:
            fig.colorbar(m_diff, ax=axes[:, 1+len(vars_list):].ravel().tolist(), shrink=0.6, location="right")

        out_path = os.path.join(out_dir, f"{field}_GT_PRED_DIFF.png")
        fig.savefig(out_path, dpi=args.dpi)
        plt.close(fig)
        print(f"Wrote {out_path}")

    print(f"\nAll outputs in: {out_dir}")

if __name__ == "__main__":
    main()
