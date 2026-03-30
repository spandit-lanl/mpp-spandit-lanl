#bin/bash

SP_SCRATCH="/lustre/scratch5/exempt/artimis/mpmm/spandit"
RUN_ROOT="${SP_SCRATCH}/runs-ch/mpp/final"
CONFIG_ROOT="/users/spandit/projects/artimis/mpp/mpp-spandit-npz-clx/config/finetune_clx/expt"
CKPT_ROOT="${RUN_ROOT}/finetune_clx/finetune"

python sp-utils/predict_unified_clx.py                                   \
  --ckpt ${CKPT_ROOT}/clx_expt_ns-04_var-7/training_checkpoints/ckpt.tar \
  --config ${CONFIG_ROOT}/mpp_avit_L_clx_04_var_7.yaml                   \
  --npz_dir ${SP_SCRATCH}/data/fp_full/clx_ft_test                   	 \
  --sample_id YOUR_SAMPLE_PREFIX_HERE                                    \
  --t_idx 98                                                             \
  --n_steps 12                                                           \
  --label_offset 12                                                      \
  --sanitize_inputs                                                      \
  --out_root ${SP_SCRATCH}/preds/clx_expt_ns-04_var-7

#  --fields vf,pf,ef,nvf \
#  --n_states 32 \
#  --fields vf,pf,ef,nvf \
#  --state_indices 12,13,14,15 \
#  --fields vf,pf,ef,nvf \
