#!/bin/bash

python predict_lsc_timestep_finetune.py \
  --config config_spandit/mpp_avit_b_config_nsteps_01.yaml \
  --config_block finetune_resume \
  --ckpt /users/spandit/projects/artimis/mpp/mpp-spandit-npz/mpp-runs-finetune/finetune/finetune_b_LSC_nsteps_01/training_checkpoints/best_ckpt.tar \
  --npz_dir /lustre/scratch5/exempt/artimis/data/mpp_finetune_on_lsc/test_10_percent_lsc \
  --sim_id 647 \
  --pred_tstep 98 \
  --n_steps 1 \
  --out_dir ./predictions/finetune_b_LSC_nsteps_01
