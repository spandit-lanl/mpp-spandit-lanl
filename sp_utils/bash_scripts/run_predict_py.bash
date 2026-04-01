#!/bin/bash

python ./predict_timestep.py                                                          \
  --config_block basic_config                                                         \
  --config ./config/dtrain_lsc/mpp_avit_L_dtrain-LSC_lr-X_opt-adan_wd-3_ns-01.yaml    \
  --ckpt  /lustre/scratch5/exempt/artimis/mpmm/spandit/runs-ch/mpp/final/dtrain_lsc/basic_config/final_L_dtrain-LSC_lr-X_opt-adan_wd-3_ns-01/training_checkpoints/best_ckpt.tar                                                             \
  --out_dir ./predictions/finetune_LSC_ns-01                                          \
  --npz_dir ./data/data_lsc_test                                                      \
  --sim_id  3739                                                                      \
  --pred_tstep 98                                                                     \
  --n_steps 1


#  --config config_spandit/mpp_avit_b_config_nsteps_01.yaml \
#  --config_block finetune_resume \
