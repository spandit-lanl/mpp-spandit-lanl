#!/bin/bash

set -euo pipefail

./01_extract_losses_clx_expt.bash
./01_extract_losses_train_phases.bash
./02_combine_losses_clx_expt.bash
./02_combine_losses_train_phases.bash
