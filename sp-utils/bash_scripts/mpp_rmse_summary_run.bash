#!/bin/bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/users/spandit/projects/artimis/mpp/mpp-spandit-npz-clx}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/final/pred}"

DATASET="${DATASET:-clx}"
EXPT_ID="${EXPT_ID:-00005}"

SCRIPT="${SCRIPT:-$REPO_ROOT/mpp_rmse_summary.py}"

# Optional: require strict expected directory layout
STRICT_LAYOUT="${STRICT_LAYOUT:-0}"

cmd=(python "$SCRIPT"
  --out_root "$OUT_ROOT"
  --dataset "$DATASET"
  --expt_id "$EXPT_ID"
)

if [[ "$STRICT_LAYOUT" == "1" ]]; then
  cmd+=(--strict_layout)
fi

echo "RUN: ${cmd[*]}"
"${cmd[@]}"

