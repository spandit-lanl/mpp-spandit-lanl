#!/usr/bin/env bash

set -euo pipefail

usage() {
    echo "Usage: $0 <cyl|pli> <1-16> <full|half> <basic_config|finetune|finetune_resume> [--dry-run]" && exit 1
}

dry_run=0

# Validate number of arguments first
# Optional 5th arg is --dry-run
if [ "$#" -ne 4 ] && [ "$#" -ne 5 ]; then
    echo "Error: invalid number of arguments." && usage
fi

# Assign positional args immediately
dataset="$1"
n_steps="$2"
img_size="$3"
config="$4"

# Optional 5th arg
if [ "$#" -eq 5 ]; then
    if [ "$5" = "--dry-run" ]; then
        dry_run=1
    else
        echo "Error: Fifth argument, if provided, must be --dry-run."
        usage
    fi
fi

# First argument
case "$dataset" in
    cyl)
	yaml_prefix=${dataset}
        dataset="cyl_cx241203"
        ;;
    pli)
	yaml_prefix=${dataset}
        dataset="pli_lsc240402"
        ;;
    *)
        echo "Error: First argument must be 'cyl' or 'pli'."
        usage
        ;;
esac

# Second argument: integer 1..16
if ! [[ "$n_steps" =~ ^[0-9]+$ ]]; then
    echo "Error: Second argument must be an integer between 1 and 16."
    usage
fi

# Force decimal interpretation so 08 works
n_steps=$((10#$n_steps))

if [ "$n_steps" -lt 1 ] || [ "$n_steps" -gt 16 ]; then
    echo "Error: Second argument must be between 1 and 16."
    usage
fi

ns=$(printf "%02d" "$n_steps")

# Third argument
case "$img_size" in
    full|half)
        ;;
    *)
        echo "Error: Third argument must be 'full' or 'half'."
        usage
        ;;
esac

# Fourth argument
case "$config" in
    basic_config|finetune|finetune_resume)
        ;;
    *)
        echo "Error: Fourth argument must be 'basic_config', 'finetune'  or 'finetune_resume'."
        usage
        ;;
esac

cmd="python -u train_basic.py --run_name ${dataset}_${img_size}_ns${ns} --config ${config} --yaml_config config/${yaml_prefix}_finetune/mpp_avit_L_finetune_${yaml_prefix}_lr_4_opt_adan_wd_3_ns_${ns}.yaml &>> ./../runs/out_${dataset}_${img_size}_ns${ns}.log"

echo "$cmd"

if [ "$dry_run" -eq 0 ]; then
    echo "CAME to eval"
    eval "$cmd"
fi
