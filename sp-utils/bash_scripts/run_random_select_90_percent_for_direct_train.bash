#!/bin/bash

echo -e "\nStarting with /lustre/scratch5/exempt/artimis/data/cx241203_fp16_full - 1\n"
echo "--train_out /lustre/scratch5/exempt/artimis/mpmm/spandit/data/mpp_finetune/cx241203_fp16_full_direct_train"
echo "--test_out /lustre/scratch5/exempt/artimis/mpmm/spandit/data/mpp_finetune/cx241203_fp16_full_direct_test"
python random_select_data.py --train_pct 90 --test_pct 100 \
 --seed 62 \
 --input /lustre/scratch5/exempt/artimis/data/cx241203_fp16_full \
 --train_out /lustre/scratch5/exempt/artimis/mpmm/spandit/data/mpp_finetune/cx241203_fp16_full_direct_train \
 --test_out /lustre/scratch5/exempt/artimis/mpmm/spandit/data/mpp_finetune/cx241203_fp16_full_direct_test
echo -e "\nDone with /lustre/scratch5/exempt/artimis/data/cx241203_fp16_full - 1\n"

echo -e "\nStarting with /lustre/scratch5/exempt/artimis/data/lsc240420_fp16_full - 1\n"
echo "--train_out /lustre/scratch5/exempt/artimis/mpmm/spandit/data/mpp_finetune/lsc240420_fp16_full_direct_train"
echo "--test_out /lustre/scratch5/exempt/artimis/mpmm/spandit/data/mpp_finetune/lsc240420_fp16_full_direct_test"
python random_select_data.py --train_pct 90 --test_pct 100 \
 --seed 62 \
 --input /lustre/scratch5/exempt/artimis/data/lsc240420_fp16_full \
 --train_out /lustre/scratch5/exempt/artimis/mpmm/spandit/data/mpp_finetune/lsc240420_fp16_full_direct_train \
 --test_out /lustre/scratch5/exempt/artimis/mpmm/spandit/data/mpp_finetune/lsc240420_fp16_full_direct_test
echo -e "\nDone with /lustre/scratch5/exempt/artimis/data/lsc240420_fp16_full - 1\n"

echo -e "\nStarting with /lustre/scratch5/exempt/artimis/data/cx241203_fp16_half - 1\n"
echo "--train_out /lustre/scratch5/exempt/artimis/mpmm/spandit/data/mpp_finetune/cx241203_fp16_half_direct_train"
echo "--test_out /lustre/scratch5/exempt/artimis/mpmm/spandit/data/mpp_finetune/cx241203_fp16_half_direct_test"
python random_select_data.py --train_pct 90 --test_pct 100 \
 --seed 52 \
 --input /lustre/scratch5/exempt/artimis/data/cx241203_fp16_half \
 --train_out /lustre/scratch5/exempt/artimis/mpmm/spandit/data/mpp_finetune/cx241203_fp16_half_direct_train \
 --test_out /lustre/scratch5/exempt/artimis/mpmm/spandit/data/mpp_finetune/cx241203_fp16_half_direct_test
echo -e "\nDone with /lustre/scratch5/exempt/artimis/data/cx241203_fp16_half - 1\n"

echo -e "\nStarting with /lustre/scratch5/exempt/artimis/data/lsc240420_fp16_half - 1\n"
echo "--train_out /lustre/scratch5/exempt/artimis/mpmm/spandit/data/mpp_finetune/lsc240420_fp16_half_direct_train"
echo "--test_out /lustre/scratch5/exempt/artimis/mpmm/spandit/data/mpp_finetune/lsc240420_fp16_half_direct_test"
python random_select_data.py --train_pct 90 --test_pct 100 \
 --seed 52 \
 --input /lustre/scratch5/exempt/artimis/data/lsc240420_fp16_half \
 --train_out /lustre/scratch5/exempt/artimis/mpmm/spandit/data/mpp_finetune/lsc240420_fp16_half_direct_train \
 --test_out /lustre/scratch5/exempt/artimis/mpmm/spandit/data/mpp_finetune/lsc240420_fp16_half_direct_test
echo -e "\nDone with /lustre/scratch5/exempt/artimis/data/lsc240420_fp16_half - 1\n"

echo -e "\nStarting with /lustre/scratch5/exempt/artimis/data/cx241203 - 1\n"
echo "--train_out /lustre/scratch5/exempt/artimis/mpmm/spandit/data/mpp_finetune/cx241203_direct_train"
echo "--test_out /lustre/scratch5/exempt/artimis/mpmm/spandit/data/mpp_finetune/cx241203_direct_tes"
python random_select_data.py --train_pct 90 --test_pct 100 \
 --seed 72 \
 --input /lustre/scratch5/exempt/artimis/data/cx241203 \
 --train_out /lustre/scratch5/exempt/artimis/mpmm/spandit/data/mpp_finetune/cx241203_direct_train \
 --test_out /lustre/scratch5/exempt/artimis/mpmm/spandit/data/mpp_finetune/cx241203_direct_test
echo -e "\nDone with /lustre/scratch5/exempt/artimis/data/cx241203 - 1\n"

echo -e "\nStarting with /lustre/scratch5/exempt/artimis/data/lsc240420 - 1\n"
echo "--train_out /lustre/scratch5/exempt/artimis/mpmm/spandit/data/mpp_finetune/lsc240420_direct_train"
echo "--test_out /lustre/scratch5/exempt/artimis/mpmm/spandit/data/mpp_finetune/lsc240420_direct_tes"
python random_select_data.py --train_pct 90 --test_pct 100 \
 --seed 72 \
 --input /lustre/scratch5/exempt/artimis/data/lsc240420 \
 --train_out /lustre/scratch5/exempt/artimis/mpmm/spandit/data/mpp_finetune/lsc240420_direct_train \
 --test_out /lustre/scratch5/exempt/artimis/mpmm/spandit/data/mpp_finetune/lsc240420_direct_test
echo -e "\nDone with /lustre/scratch5/exempt/artimis/data/lsc240420 - 1\n"
