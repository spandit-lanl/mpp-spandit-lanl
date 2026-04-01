#!/bin/bash

set -euo pipefail
#rm -f *.pdf loss_clx*.csv combined_losses_clx_expt.csv loss_final*.csv  combined_losses.csv

## This plots CLX expt plots only
#python 00_extract_combine_plot_clx_expt.py

## This plots pretraina and finetune plots only
python3 00_extract_combine_plot.py \
  --log-dir /users/spandit/projects/artimis/mpp/mpp-spandit-npz-clx/final \
  --log-glob "out_final_L_*.log"

<< COMMENT
## ######################################## ##
## This was the old flow                    ##
## ######################################## ##
./01_extract_losses.bash
python3 02_combine_losses.py

if ! python3 03_verify_combines_losses.py; then
  echo "WARNING: verify failed; continuing to plots anyway" # Don't let verify stop plotting; warn and continue
fi

python3 ./04_plot_grid_losses.py
python3 ./04_plot_grid_losses_uniq_EPOCH_SCALES.py

echo "Done. Outputs:"
ls -1 *.png 2>/dev/null || true

COMMENT
#01_extract_losses_clx_expt.bash
#01_extract_losses_train_phases.bash
#02_combine_losses_clx_expt.bash
#02_combine_losses_train_phases.bash
