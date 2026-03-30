#!/bin/bash

set -x
python 04_plot_grid_losses.py --phase pretrain --which all --output pretrain_all_grid.png
#python 04_plot_grid_losses.py --phase pretrain --which odd --output pretrain_odd_grid.png
#python 04_plot_grid_losses.py --phase pretrain --which even --output pretrain_even_grid.png

python 04_plot_grid_losses.py --phase finetune --which all --output finetune_all_grid.png
#python 04_plot_grid_losses.py --phase finetune --which odd --output finetune_odd_grid.png
#python 04_plot_grid_losses.py --phase finetune --which even --output finetune_even_grid.png
#python 05_plot_grid_losses_range.py --phase pretrain --which even --nstep_min 1 --nstep_max 8  --output pretrain_1to8_grid.png
#python 05_plot_grid_losses_range.py --phase finetune --which even --nstep_min 1 --nstep_max 8  --output finetune_odd_1to8_grid.png
set +x
