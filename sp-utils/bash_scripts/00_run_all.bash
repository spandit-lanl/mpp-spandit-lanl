#!/bin/bash

#./01_extract_losses.bash

#python3 02_combine_losses.py

#python3 03_verify_combines_losses.py

#python3 04_plot_grid_losses.py --phase finetune --which all --output finetune_all_grid.png
#python3 04_plot_grid_losses.py --phase pretrain --which all --output pretrain_all_grid.png
python3 04_plot_grid_losses.py --phase direct --which all --output pretrain_all_grid.png

#python3 04_plot_grid_losses.py --phase finetune --which odd --output finetune_odd_grid.png
#python3 04_plot_grid_losses.py --phase pretrain --which odd --output pretrain_odd_grid.png

#python3 04_plot_grid_losses.py --phase pretrain --which even --output pretrain_even_grid.png
#python3 04_plot_grid_losses.py --phase finetune --which even --output finetune_even_grid.png

#python3 05_updated_plot_grid_losses_range.py --phase finetune --which all --nstep_min 1 --nstep_max 8  --output finetune_1to8_grid.png
#python3 05_updated_plot_grid_losses_range.py --phase finetune --which odd --nstep_min 1 --nstep_max 8  --output finetune_odd_1to8_grid.png
