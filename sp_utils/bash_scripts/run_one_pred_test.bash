REPO_ROOT=/users/spandit/projects/artimis/mpp/mpp-spandit-npz-clx
RUNS_ROOT=/lustre/scratch5/exempt/artimis/mpmm/spandit/runs-ch/mpp

DATASET=clx
DATA_DIR=/lustre/scratch5/exempt/artimis/mpmm/spandit/data/cx241203_fp16_full_finetune_test

EXPT_ID=00005
T_IDX=20
NS=1
VAR=3

CFG=${REPO_ROOT}/config/finetune_${DATASET}/mpp_avit_L_finetune-CLX_lr-4_opt-adan_wd-3_ns-01_var-${VAR}.yaml
CKPT=${RUNS_ROOT}/final/finetune_${DATASET}/finetune/final_L_finetune-CLX_lr-4_opt-adan_wd-3_ns-01_var-${VAR}/training_checkpoints/best_ckpt.tar

python ${REPO_ROOT}/mpp_predict.py \
  --repo_root "${REPO_ROOT}" \
  --config "${CFG}" \
  --yparams_section finetune_resume \
  --ckpt "${CKPT}" \
  --data_dir "${DATA_DIR}" \
  --dataset "${DATASET}" \
  --expt_id "${EXPT_ID}" \
  --ns "${NS}" \
  --t_idx "${T_IDX}" \
  --var "${VAR}" \
  --fields "av_density,Uvelocity,Wvelocity,density_maincharge" \
  --state_indices "11,12,13,14" \
  --out_root ./final/pred \
  --sanitize_inputs \
  --run_forward \
  --save_outputs
