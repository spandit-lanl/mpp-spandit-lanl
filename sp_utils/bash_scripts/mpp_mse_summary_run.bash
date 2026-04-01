#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Run mpp_mse_summary.py for ONLY the expt IDs listed in IDS_FILE.
# Writes output to:
#   ${OUT_ROOT}/${DATASET}/expt-${EXPT_ID}/mse_summary.csv
#
# No directory creation. Missing expt dirs are skipped.
# Octal-safe expt id formatting (10#...).
# -----------------------------------------------------------------------------

REPO_ROOT="${REPO_ROOT:-/users/spandit/projects/artimis/mpp/mpp-spandit-npz-clx}"

# Default to LUSTRE (can override)
OUT_ROOT="${OUT_ROOT:-/lustre/scratch5/exempt/artimis/mpmm/spandit/runs-ch/mpp/final/pred}"

DATASET="${DATASET:-clx}"

# File with 5-digit expt IDs (one per line), generated earlier
IDS_FILE="${IDS_FILE:-./uniq_expt_ids_5digit_sorted.txt}"

SCRIPT="${SCRIPT:-${REPO_ROOT}/mpp_mse_summary.py}"

echo "REPO_ROOT = ${REPO_ROOT}"
echo "OUT_ROOT  = ${OUT_ROOT}"
echo "DATASET   = ${DATASET}"
echo "IDS_FILE  = ${IDS_FILE}"
echo "SCRIPT    = ${SCRIPT}"
echo

if [[ ! -f "${IDS_FILE}" ]]; then
  echo "ERROR: cannot find IDS_FILE: ${IDS_FILE}" >&2
  exit 1
fi

if [[ ! -f "${SCRIPT}" ]]; then
  echo "ERROR: cannot find SCRIPT: ${SCRIPT}" >&2
  exit 1
fi

if [[ ! -d "${OUT_ROOT}/${DATASET}" ]]; then
  echo "ERROR: dataset root dir not found: ${OUT_ROOT}/${DATASET}" >&2
  exit 1
fi

while IFS= read -r EXPT_ID; do
  [[ -z "${EXPT_ID}" ]] && continue
  [[ "${EXPT_ID}" =~ ^# ]] && continue

  # Ensure 5-digit formatting; force base-10 to avoid octal issues (e.g., 00028)
  EXPT_ID="$(printf "%05d" $((10#${EXPT_ID})))"

  EXPT_DIR="${OUT_ROOT}/${DATASET}/expt-${EXPT_ID}"
  CSV_OUT="${EXPT_DIR}/mse_summary.csv"

  # Do NOT create directories; if missing, skip
  if [[ ! -d "${EXPT_DIR}" ]]; then
    echo "[WARN] missing expt dir, skipping: ${EXPT_DIR}" >&2
    continue
  fi

  echo "==> EXPT_ID=${EXPT_ID}"
  echo "    CSV_OUT=${CSV_OUT}"

  python "${SCRIPT}" \
    --out_root "${OUT_ROOT}" \
    --dataset "${DATASET}" \
    --expt_id "${EXPT_ID}" \
    --csv_out "${CSV_OUT}"

  echo
done < "${IDS_FILE}"

echo "Done."
