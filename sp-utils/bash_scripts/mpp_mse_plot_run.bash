#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/users/spandit/projects/artimis/mpp/mpp-spandit-npz-clx}"

# Lustre root (same as summary driver)
OUT_ROOT="${OUT_ROOT:-/lustre/scratch5/exempt/artimis/mpmm/spandit/runs-ch/mpp/final/pred}"
DATASET="${DATASET:-clx}"

IDS_FILE="${IDS_FILE:-./uniq_expt_ids_5digit_sorted.txt}"

# Plot script (uploaded/your repo copy)
PLOT_SCRIPT="${PLOT_SCRIPT:-${REPO_ROOT}/mpp_mse_plot.py}"

# Plot params
NS="${NS:-1}"
VAR="${VAR:-3}"
DPI="${DPI:-600}"
DRY_RUN="${DRY_RUN:-0}"

echo "OUT_ROOT     = ${OUT_ROOT}"
echo "DATASET      = ${DATASET}"
echo "IDS_FILE     = ${IDS_FILE}"
echo "PLOT_SCRIPT  = ${PLOT_SCRIPT}"
echo "NS           = ${NS}"
echo "VAR          = ${VAR}"
echo "DPI          = ${DPI}"
echo "DRY_RUN      = ${DRY_RUN}"
echo

if [[ ! -f "${IDS_FILE}" ]]; then
  echo "ERROR: cannot find IDS_FILE: ${IDS_FILE}" >&2
  exit 1
fi
if [[ ! -f "${PLOT_SCRIPT}" ]]; then
  echo "ERROR: cannot find PLOT_SCRIPT: ${PLOT_SCRIPT}" >&2
  exit 1
fi
if [[ ! -d "${OUT_ROOT}/${DATASET}" ]]; then
  echo "ERROR: dataset root dir not found: ${OUT_ROOT}/${DATASET}" >&2
  exit 1
fi

while IFS= read -r EXPT_ID; do
  [[ -z "${EXPT_ID}" ]] && continue
  [[ "${EXPT_ID}" =~ ^# ]] && continue

  # Octal-safe and 5-digit
  EXPT_ID="$(printf "%05d" $((10#${EXPT_ID})))"

  EXPT_DIR="${OUT_ROOT}/${DATASET}/expt-${EXPT_ID}"
  CSV_IN="${EXPT_DIR}/mse_summary.csv"
  OUT_PNG="${EXPT_DIR}/mse_expt-${EXPT_ID}.png"

  if [[ ! -d "${EXPT_DIR}" ]]; then
    echo "[WARN] missing expt dir, skipping: ${EXPT_DIR}" >&2
    continue
  fi
  if [[ ! -f "${CSV_IN}" ]]; then
    echo "[WARN] missing CSV, skipping: ${CSV_IN}" >&2
    continue
  fi

  CMD=(python "${PLOT_SCRIPT}"
       --csv_in "${CSV_IN}"
       --dataset "${DATASET}"
       --expt_id "${EXPT_ID}"
       --ns "${NS}"
       --var "${VAR}"
       --out_png "${OUT_PNG}"
       --dpi "${DPI}")

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[DRY] ${CMD[*]}"
  else
    echo "==> expt-${EXPT_ID}"
    "${CMD[@]}"
  fi

done < "${IDS_FILE}"

echo "Done."
