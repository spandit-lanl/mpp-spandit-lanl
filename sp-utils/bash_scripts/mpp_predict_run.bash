#!/bin/bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/users/spandit/projects/artimis/mpp/mpp-spandit-npz-clx}"
RUNS_ROOT="${RUNS_ROOT:-/lustre/scratch5/exempt/artimis/mpmm/spandit/runs-ch/mpp}"

DATASET="${DATASET:-clx}"
DATA_DIR="${DATA_DIR:-/lustre/scratch5/exempt/artimis/mpmm/spandit/data/cx241203_fp16_full_finetune_test}"
EXPT_ID="${EXPT_ID:-01750}"
T_IDX="${T_IDX:-23}"

SCRIPT="${SCRIPT:-${REPO_ROOT}/mpp_predict.py}"

FIELDS="${FIELDS:-av_density,Uvelocity,Wvelocity,density_maincharge}"
STATE_INDICES="${STATE_INDICES:-12,13,14,15}"
YPARAMS_SECTION="${YPARAMS_SECTION:-finetune_resume}"

OUT_ROOT="${OUT_ROOT:-./final/pred}"
TAG="${TAG:-}"

SANITIZE="${SANITIZE:---sanitize_inputs}"
PRINT_KEYS="${PRINT_KEYS:-}"   # set to --print_keys if desired

RUN_FORWARD="${RUN_FORWARD:-1}"
SAVE_OUTPUTS="${SAVE_OUTPUTS:-1}"

FWD_FLAG=""; [[ "${RUN_FORWARD}" == "1" ]] && FWD_FLAG="--run_forward"
SAVE_FLAG=""; [[ "${SAVE_OUTPUTS}" == "1" ]] && SAVE_FLAG="--save_outputs"

mkdir -p logs
: > out.txt

for expt_id in `cat ./uniq_sim_ids_finetuning_testing_60_plus.txt`; do
	EXPT_ID="${expt_id}"
	echo $EXPT_ID

	for t_id in `seq  1 1 54`; do
		T_IDX="${t_id:-23}"
		#for ns_i in $(seq 1 8); do
		for ns_i in $(seq 1 1); do
		  ns2=$(printf "%02d" "$ns_i")
		  #for var in 3 6; do
		  for var in 3; do
		    CFG="${REPO_ROOT}/config/finetune_${DATASET}/mpp_avit_L_finetune-CLX_lr-4_opt-adan_wd-3_ns-${ns2}_var-${var}.yaml"
		    CKPT="${RUNS_ROOT}/final/finetune_${DATASET}/finetune/final_L_finetune-CLX_lr-4_opt-adan_wd-3_ns-${ns2}_var-${var}/training_checkpoints/best_ckpt.tar"

		    echo -e "\n============================================================"
		    echo "EXPT_ID=${EXPT_ID}, T_IDX=${T_IDX}, dataset=${DATASET} ns=${ns2} (arg ns=${ns_i}) var=${var}"
		    echo "CFG:  ${CFG}"
		    echo "CKPT: ${CKPT}"
		    echo "============================================================"

		    if [[ ! -f "${CFG}" ]]; then
		      echo "SKIP: missing config ${CFG}" | tee -a out.txt
		      continue
		    fi
		    if [[ ! -f "${CKPT}" ]]; then
		      echo "SKIP: missing checkpoint ${CKPT}" | tee -a out.txt
		      continue
		    fi

		    LOG="logs/incremental_${DATASET}_ns-${ns2}_var-${var}.log"

		    {
		      echo "RUN $(date -Is) dataset=${DATASET} ns=${ns2} var=${var}"
		      python "${SCRIPT}" \
			--repo_root "${REPO_ROOT}" \
			--config "${CFG}" \
			--yparams_section "${YPARAMS_SECTION}" \
			--ckpt "${CKPT}" \
			--data_dir "${DATA_DIR}" \
			--dataset "${DATASET}" \
			--expt_id "${EXPT_ID}" \
			--ns "${ns_i}" \
			--t_idx "${T_IDX}" \
			--var "${var}" \
			--fields "${FIELDS}" \
			--state_indices "${STATE_INDICES}" \
			--out_root "${OUT_ROOT}" \
			${TAG:+--tag "${TAG}"} \
			${PRINT_KEYS} \
			${SANITIZE} \
			${FWD_FLAG} \
			${SAVE_FLAG}
		      echo "DONE $(date -Is) dataset=${DATASET} ns=${ns2} var=${var}"
		    } 2>&1 | tee "${LOG}" | tee -a out.txt

		    echo "OK: wrote ${LOG}" | tee -a out.txt
		  done
		done
	done
done


read
read
read




echo "ALL DONE. Global log: out.txt"

