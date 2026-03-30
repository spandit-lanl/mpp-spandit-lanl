#!/bin/bash
set -euo pipefail

# Single-shot prediction driver (loads ckpt once; loops inside Python)

PROJ_ROOT="${PROJ_ROOT:-/users/spandit/projects/artimis/mpp/mpp-spandit-npz-clx}"
DATASET="${DATASET:-clx}"
DATA_DIR="${DATA_DIR:-/lustre/scratch5/exempt/artimis/mpmm/spandit/finetune/cx241203_fp16_full_finetune_test}"
SIM_IDS_FILE="${SIM_IDS_FILE:-./sim_ids.txt}"


CKPT="${CKPT:-/lustre/scratch5/exempt/artimis/mpmm/spandit/runs-ch/mpp/final/finetune_clx/finetune/final_L_finetune-CLX_lr-4_opt-adan_wd-3_ns-01/training_checkpoints/best_ckpt.tar}"
CFG="${CFG:-/users/spandit/projects/artimis/mpp/mpp-spandit-npz-clx/config/finetune_clx/mpp_avit_L_finetune-CLX_lr-4_opt-adan_wd-3_ns-01_var-3.yaml}"

SCRIPT="${SCRIPT:-${PROJ_ROOT}/mpp_predict_new.py}"

FIELDS="${FIELDS:-av_density,Uvelocity,Wvelocity,density_maincharge}"
STATE_INDICES="${STATE_INDICES:-12,13,14,15}"
YPARAMS_SECTION="${YPARAMS_SECTION:-finetune_resume}"

OUT_ROOT="${OUT_ROOT:-./final/pred}"
MIN_TIMESTEP="${MIN_TIMESTEP:-2}"

mkdir -p logs
LOG="logs/predict_loop_${DATASET}_$(date +%Y%m%dT%H%M%S).log"

python "${SCRIPT}" \
  --proj_root "${PROJ_ROOT}" \
  --config "${CFG}" \
  --yparams_section "${YPARAMS_SECTION}" \
  --ckpt "${CKPT}" \
  --data_dir "${DATA_DIR}" \
  --dataset "${DATASET}" \
  --sim_ids_file "${SIM_IDS_FILE}" \
  --min_timestep "${MIN_TIMESTEP}" \
  --fields "${FIELDS}" \
  --state_indices "${STATE_INDICES}" \
  --out_root "${OUT_ROOT}" \
  --run_forward \
  --sanitize_inputs \
  --series cx241203 \
  2>&1 | tee "${LOG}"

echo "DONE. Log: ${LOG}"
