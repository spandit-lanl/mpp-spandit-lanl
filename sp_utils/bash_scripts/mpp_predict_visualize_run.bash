#!/bin/bash
set -euo pipefail

ROOT=/lustre/scratch5/exempt/artimis/mpmm/spandit/runs-ch/mpp/final/pred \
DATASET=clx \
EXPT_ID=01750 \
T_IDX=56 \
VARS=3,6 \
NS_MIN=1 \
NS_MAX=8 \
./mpp_run_viz.bash


