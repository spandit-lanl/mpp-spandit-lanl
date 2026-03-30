#!/bin/bash

DATA_ROOT="/lustre/scratch5/exempt/artimis/spandit/datasets"

#DATASET="pli"
DATASET="cyl"

PRECISION="fp16"
#PRECISION="fp32"
#PRECISION="fp64"

#IMG_SIZE="half"
IMG_SIZE="full"

echo $DATA_ROOT
echo $DATA_PLI
echo $DATA_CYL
tgt_data=${DATA_ROOT}/$DATASET/${PRECISION}/${IMG_SIZE}

echo $tgt_data

ls $tgt_data

read
read

echo -e "\nStarting with /lustre/scratch5/exempt/artimis/data/cx241203_fp16_full - 1\n"
echo "--train_out /lustre/scratch5/exempt/artimis/mpmm/spandit/data/mpp_finetune/cx241203_fp16_full_finetune_train "
echo "--test_out /lustre/scratch5/exempt/artimis/mpmm/spandit/data/mpp_finetune/cx241203_fp16_full_finetune_test"
python random_select_data.py --train_pct 35 --test_pct 20 \
 --seed 62 \
 --input /lustre/scratch5/exempt/artimis/data/cx241203_fp16_full \
 --train_out /lustre/scratch5/exempt/artimis/mpmm/spandit/data/mpp_finetune/cx241203_fp16_full_finetune_train \
 --test_out /lustre/scratch5/exempt/artimis/mpmm/spandit/data/mpp_finetune/cx241203_fp16_full_finetune_test
echo -e "\nDone with /lustre/scratch5/exempt/artimis/data/cx241203_fp16_full - 1\n"

echo -e "\nStarting with /lustre/scratch5/exempt/artimis/data/lsc240420_fp16_full - 1\n"
echo "--train_out /lustre/scratch5/exempt/artimis/mpmm/spandit/data/mpp_finetune/lsc240420_fp16_full_finetune_train "
echo "--test_out /lustre/scratch5/exempt/artimis/mpmm/spandit/data/mpp_finetune/lsc240420_fp16_full_finetune_test"
python random_select_data.py --train_pct 35 --test_pct 20 \
 --seed 62 \
 --input /lustre/scratch5/exempt/artimis/data/lsc240420_fp16_full \
 --train_out /lustre/scratch5/exempt/artimis/mpmm/spandit/data/mpp_finetune/lsc240420_fp16_full_finetune_train \
 --test_out /lustre/scratch5/exempt/artimis/mpmm/spandit/data/mpp_finetune/lsc240420_fp16_full_finetune_test
echo -e "\nDone with /lustre/scratch5/exempt/artimis/data/lsc240420_fp16_full - 1\n"

echo -e "\nStarting with /lustre/scratch5/exempt/artimis/data/cx241203_fp16_half - 1\n"
echo "--train_out /lustre/scratch5/exempt/artimis/mpmm/spandit/data/mpp_finetune/cx241203_fp16_half_finetune_train"
echo "--test_out /lustre/scratch5/exempt/artimis/mpmm/spandit/data/mpp_finetune/cx241203_fp16_half_finetune_test"
python random_select_data.py --train_pct 35 --test_pct 20 \
 --seed 52 \
 --input /lustre/scratch5/exempt/artimis/data/cx241203_fp16_half \
 --train_out /lustre/scratch5/exempt/artimis/mpmm/spandit/data/mpp_finetune/cx241203_fp16_half_finetune_train \
 --test_out /lustre/scratch5/exempt/artimis/mpmm/spandit/data/mpp_finetune/cx241203_fp16_half_finetune_test
echo -e "\nDone with /lustre/scratch5/exempt/artimis/data/cx241203_fp16_half - 1\n"

echo -e "\nStarting with /lustre/scratch5/exempt/artimis/data/lsc240420_fp16_half - 1\n"
echo "--train_out /lustre/scratch5/exempt/artimis/mpmm/spandit/data/mpp_finetune/lsc240420_fp16_half_finetune_train "
echo "--test_out /lustre/scratch5/exempt/artimis/mpmm/spandit/data/mpp_finetune/lsc240420_fp16_half_finetune_test"
python random_select_data.py --train_pct 35 --test_pct 20 \
 --seed 52 \
 --input /lustre/scratch5/exempt/artimis/data/lsc240420_fp16_half \
 --train_out /lustre/scratch5/exempt/artimis/mpmm/spandit/data/mpp_finetune/lsc240420_fp16_half_finetune_train \
 --test_out /lustre/scratch5/exempt/artimis/mpmm/spandit/data/mpp_finetune/lsc240420_fp16_half_finetune_test
echo -e "\nDone with /lustre/scratch5/exempt/artimis/data/lsc240420_fp16_half - 1\n"

echo -e "\nStarting with /lustre/scratch5/exempt/artimis/data/cx241203 - 1\n"
echo "--train_out /lustre/scratch5/exempt/artimis/mpmm/spandit/data/mpp_finetune/cx241203_finetune_train "
echo "--test_out /lustre/scratch5/exempt/artimis/mpmm/spandit/data/mpp_finetune/cx241203_finetune_test"
python random_select_data.py --train_pct 35 --test_pct 20 \
 --seed 72 \
 --input /lustre/scratch5/exempt/artimis/data/cx241203 \
 --train_out /lustre/scratch5/exempt/artimis/mpmm/spandit/data/mpp_finetune/cx241203_finetune_train \
 --test_out /lustre/scratch5/exempt/artimis/mpmm/spandit/data/mpp_finetune/cx241203_finetune_test
echo -e "\nDone with /lustre/scratch5/exempt/artimis/data/cx241203 - 1\n"

echo -e "\nStarting with /lustre/scratch5/exempt/artimis/data/lsc240420 - 1\n"
echo "--train_out /lustre/scratch5/exempt/artimis/mpmm/spandit/data/mpp_finetune/lsc240420_finetune_train "
echo "--test_out /lustre/scratch5/exempt/artimis/mpmm/spandit/data/mpp_finetune/lsc240420_finetune_test"
python random_select_data.py --train_pct 35 --test_pct 20 \
 --seed 72 \
 --input /lustre/scratch5/exempt/artimis/data/lsc240420 \
 --train_out /lustre/scratch5/exempt/artimis/mpmm/spandit/data/mpp_finetune/lsc240420_finetune_train \
 --test_out /lustre/scratch5/exempt/artimis/mpmm/spandit/data/mpp_finetune/lsc240420_finetune_test
echo -e "\nDone with /lustre/scratch5/exempt/artimis/data/lsc240420 - 1\n"
