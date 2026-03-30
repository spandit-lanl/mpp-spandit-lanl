#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# mpp_run_make_mean_mse_summary.bash
#
# Create ./mean_mse_summary.csv in PWD by averaging step_mse across experiments
# listed in IDS_FILE. Reads per-expt:
#   ${OUT_ROOT}/${DATASET}/expt-${EXPT_ID}/mse_summary.csv
#
# Does NOT create any directories.
# -----------------------------------------------------------------------------

REPO_ROOT="${REPO_ROOT:-/users/spandit/projects/artimis/mpp/mpp-spandit-npz-clx}"
OUT_ROOT="${OUT_ROOT:-/lustre/scratch5/exempt/artimis/mpmm/spandit/runs-ch/mpp/final/pred}"
DATASET="${DATASET:-clx}"

IDS_FILE="${IDS_FILE:-./uniq_expt_ids_5digit_sorted.txt}"

SCRIPT="${SCRIPT:-${REPO_ROOT}/mpp_make_mean_mse_summary.py}"
OUT_CSV="${OUT_CSV:-./mean_mse_summary.csv}"

echo "OUT_ROOT = ${OUT_ROOT}"
echo "DATASET  = ${DATASET}"
echo "IDS_FILE = ${IDS_FILE}"
echo "SCRIPT   = ${SCRIPT}"
echo "OUT_CSV  = ${OUT_CSV}"
echo

if [[ ! -f "${IDS_FILE}" ]]; then
  echo "ERROR: IDS_FILE not found: ${IDS_FILE}" >&2
  exit 1
fi

if [[ ! -f "${SCRIPT}" ]]; then
  echo "ERROR: SCRIPT not found: ${SCRIPT}" >&2
  exit 1
fi

python "${SCRIPT}" \
  --ids_file "${IDS_FILE}" \
  --out_root "${OUT_ROOT}" \
  --dataset "${DATASET}" \
  --out_csv "${OUT_CSV}"
