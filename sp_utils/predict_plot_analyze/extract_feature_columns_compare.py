#!/usr/bin/env python3
"""
extract_feature_columns_compare.py

Given multiple "PRED_vs_GT" PNGs that are arranged as a 3x8 grid (rows: GT / Pred / Error,
cols: features), this script extracts the corresponding feature *column* from each PNG and
concatenates them side-by-side.

Output:
- 8 PNGs (one per feature column), each showing (GT/Pred/Error) for all input PNGs.

This script is designed for figures like:
  lsc240420_idXXXXX_pvi_idxYYYYY_nstepsK_PRED_vs_GT.png

Usage examples:
  python extract_feature_columns_compare.py \
    --inputs lsc*_nsteps1_PRED_vs_GT.png lsc*_nsteps4_PRED_vs_GT.png lsc*_nsteps8_PRED_vs_GT.png \
    --out_dir out_columns

  # Optional: provide feature names (8 items). Used only for output filenames/titles.
  python extract_feature_columns_compare.py \
    --inputs ... \
    --out_dir out_columns \
    --feature_names pressure_throw,density_throw,temperature_throw,density_case,pressure_case,temperature_case,Uvelocity,Wvelocity \
    --label_nsteps
"""

import argparse
import os
import re
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _nonwhite_mask(arr: np.ndarray, thresh: int = 245) -> np.ndarray:
    """True where pixel is not 'near-white'."""
    return (arr[..., 0] < thresh) | (arr[..., 1] < thresh) | (arr[..., 2] < thresh)


def _contiguous_true_segments(b: np.ndarray) -> List[Tuple[int, int]]:
    """Return list of (start, end_exclusive) for contiguous True runs in 1D boolean array."""
    segs: List[Tuple[int, int]] = []
    in_run = False
    start = 0
    for i, v in enumerate(b.tolist()):
        if v and not in_run:
            in_run = True
            start = i
        elif (not v) and in_run:
            in_run = False
            segs.append((start, i))
    if in_run:
        segs.append((start, len(b)))
    return segs


def infer_column_bounds(img: Image.Image, expected_cols: int = 8) -> Tuple[List[Tuple[int, int]], Tuple[int, int]]:
    """
    Infer 8 column x-bounds and a common y-bound for the grid area using non-white pixel structure.
    Strategy:
      1) Compute non-white mask.
      2) Find x-runs with non-white pixels.
      3) Pick the 'expected_cols' widest runs as the feature columns (filters out left-side labels).
      4) Compute common y-min/y-max across those columns.
    """
    arr = np.array(img.convert("RGB"))
    mask = _nonwhite_mask(arr)

    # Vertical projection: any non-white in each x column
    x_any = mask.any(axis=0)
    segs = _contiguous_true_segments(x_any)

    if len(segs) < expected_cols:
        raise RuntimeError(f"Could not detect enough vertical segments: found {len(segs)}, expected {expected_cols}.")

    # Rank by width, take widest expected_cols segments as feature columns
    segs_sorted = sorted(segs, key=lambda ab: (ab[1] - ab[0]), reverse=True)[:expected_cols]
    segs_sorted = sorted(segs_sorted, key=lambda ab: ab[0])  # left-to-right

    # Compute y bounds across union of these column segments
    col_mask = np.zeros(mask.shape, dtype=bool)
    for x0, x1 in segs_sorted:
        col_mask[:, x0:x1] |= mask[:, x0:x1]

    y_any = col_mask.any(axis=1)
    y_segs = _contiguous_true_segments(y_any)
    if not y_segs:
        raise RuntimeError("Could not infer y-bounds for grid area.")

    # Take the largest y segment (should correspond to the main grid)
    y0, y1 = max(y_segs, key=lambda ab: ab[1] - ab[0])

    return segs_sorted, (y0, y1)


def parse_nsteps_from_name(p: str) -> Optional[str]:
    m = re.search(r"nsteps(\d+)", os.path.basename(p))
    return m.group(1) if m else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="Input PNG files (different nsteps).")
    ap.add_argument("--out_dir", required=True, help="Directory to write the 8 output PNGs.")
    ap.add_argument(
        "--feature_names",
        default=None,
        help="Comma-separated list of 8 feature names (used in output filenames).",
    )
    ap.add_argument(
        "--label_nsteps",
        action="store_true",
        help="If set, draw 'nsteps=K' above each column in the output image.",
    )
    ap.add_argument(
        "--tight",
        action="store_true",
        help="If set, do not add any extra padding around crops.",
    )
    args = ap.parse_args()

    in_paths = [Path(p) for p in args.inputs]
    for p in in_paths:
        if not p.exists():
            raise SystemExit(f"Missing input: {p}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.feature_names:
        parts = [s.strip() for s in args.feature_names.split(",") if s.strip()]
        if len(parts) != 8:
            raise SystemExit(f"--feature_names must have exactly 8 comma-separated names; got {len(parts)}")
        feature_names = parts
    else:
        feature_names = [f"feature_{i+1:02d}" for i in range(8)]

    # Infer bounds from the first image and reuse for all (assumes consistent layout)
    first_img = Image.open(in_paths[0])
    col_bounds, (y0, y1) = infer_column_bounds(first_img, expected_cols=8)

    pad = 0 if args.tight else 2

    imgs = [Image.open(p).convert("RGB") for p in in_paths]
    nsteps_labels = [parse_nsteps_from_name(str(p)) for p in in_paths]

    for j, (x0, x1) in enumerate(col_bounds):
        crops = []
        for img in imgs:
            cx0 = max(0, x0 - pad)
            cx1 = min(img.width, x1 + pad)
            cy0 = max(0, y0 - pad)
            cy1 = min(img.height, y1 + pad)
            crops.append(img.crop((cx0, cy0, cx1, cy1)))

        total_w = sum(c.width for c in crops)
        max_h = max(c.height for c in crops)

        label_h = 32 if args.label_nsteps else 0
        out_img = Image.new("RGB", (total_w, max_h + label_h), (255, 255, 255))

        x_cursor = 0
        for c in crops:
            out_img.paste(c, (x_cursor, label_h))
            x_cursor += c.width

        if args.label_nsteps:
            draw = ImageDraw.Draw(out_img)
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", 16)
            except Exception:
                font = ImageFont.load_default()

            x_cursor = 0
            for idx, c in enumerate(crops):
                lab = nsteps_labels[idx]
                lab_txt = f"nsteps={lab}" if lab else f"img{idx+1}"
                # textbbox is better when available; fallback otherwise
                try:
                    bbox = draw.textbbox((0, 0), lab_txt, font=font)
                    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                except Exception:
                    tw, th = draw.textsize(lab_txt, font=font)
                draw.text((x_cursor + (c.width - tw) // 2, (label_h - th) // 2), lab_txt, fill=(0, 0, 0), font=font)
                x_cursor += c.width

        fname = feature_names[j]
        safe = re.sub(r"[^A-Za-z0-9_\-]+", "_", fname)
        out_file = out_dir / f"{j+1:02d}_{safe}_compare.png"
        out_img.save(out_file)
        print(f"Saved: {out_file}")

    print("Done.")


if __name__ == "__main__":
    main()
