#!/bin/bash


#python plot_train_val_losses_2x4_grid_uniformy.py 	\
#			--pattern "out_pretrain_b_MPP_nsteps_*.txt" 	\
#			--parity odd                                  \
#			--max_epoch 460                               \
#			--out pretrain_odd_grid.png
#
#python plot_train_val_losses_2x4_grid_uniformy.py   \
#			--pattern "out_pretrain_b_MPP_nsteps_*.txt"   \
#			--parity even                                 \
#			--max_epoch 460                               \
#			--out pretrain_even_grid.png
#

#python plot_train_val_losses_grid.py --mode pretrain --parity  odd  --max_loss 0.3
#python plot_train_val_losses_grid.py --mode pretrain --parity even  --max_loss 0.3
#python plot_train_val_losses_grid.py --mode pretrain --parity  all  --max_loss 0.3


python plot_train_val_losses_grid.py --mode finetune --parity  odd  --max_loss 1.3
python plot_train_val_losses_grid.py --mode finetune --parity even  --max_loss 1.3
python plot_train_val_losses_grid.py --mode finetune --parity  all  --max_loss 1.3

