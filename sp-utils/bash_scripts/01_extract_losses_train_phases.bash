#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# Directory containing logs
LOG_DIR="/users/spandit/projects/artimis/mpp/mpp-spandit-npz-clx/final"

if [[ ! -d "$LOG_DIR" ]]; then
  echo "ERROR: LOG_DIR does not exist: $LOG_DIR" >&2
  exit 1
fi

# We handle two log families:
#  1) out_final_L_*.log  -> loss_final_L_*.csv
logs=( "$LOG_DIR"/out_final_L_*.log )
if (( ${#logs[@]} == 0 )); then
  echo "No logs found matching: $LOG_DIR/out_final_L_*.log" >&2
  exit 0
fi

for f in "${logs[@]}"; do
  base="${f##*/}"             # filename only
  base_noext="${base%.log}"   # strip .log

  if [[ "$base_noext" == out_final_L_* ]]; then
    # Expected:
    # out_final_L_<phase>-<phasedata>_lr-X_opt-adan_wd-3_ns-<NN>.log
    if [[ ! "$base_noext" =~ ^out_final_L_(pretrain|dtrain|finetune)-([A-Za-z0-9]+)_lr-X_opt-adan_wd-3_ns-([0-9]{2})$ ]]; then
      continue
    fi

    phase="${BASH_REMATCH[1]}"
    phasedata="${BASH_REMATCH[2]}"

    # Optional consistency check (per your rules)
    if [[ "$phase" == "pretrain" ]]; then
      if [[ "$phasedata" != "pdebenchfull" && "$phasedata" != "pdebenchpart" ]]; then
        echo "WARNING: skipping inconsistent file (phase=$phase, phasedata=$phasedata): $base" >&2
        continue
      fi
    else
      if [[ "$phasedata" != "LSC" && "$phasedata" != "CLX" ]]; then
        echo "WARNING: skipping inconsistent file (phase=$phase, phasedata=$phasedata): $base" >&2
        continue
      fi
    fi

    # input: out_final_*  -> output: loss_final_*
    outfile="loss_${base_noext}.csv"
    outfile="${outfile/loss_out_final_/loss_final_}"

  else
    continue
  fi

  echo "writing $outfile"

  {
    echo "epoch,train_loss,valid_loss"
    awk '
      /Epoch:/ && /Train loss:/ && /Valid loss:/ {
        epoch=""; train=""; valid=""

        if (match($0, /Epoch:[[:space:]]*([0-9]+)/, e)) epoch = e[1]

        # Train loss variants:
        if (match($0, /Train loss:[[:space:]]*tensor\(\[([0-9eE+\-\.]+)\]/, t)) train = t[1]
        else if (match($0, /Train loss:[[:space:]]*tensor\(([0-9eE+\-\.]+)/, t2)) train = t2[1]
        else if (match($0, /Train loss:[[:space:]]*([0-9eE+\-\.]+)/, t3)) train = t3[1]

        # Valid loss variants:
        if (match($0, /Valid loss:[[:space:]]*tensor\(\[([0-9eE+\-\.]+)\]/, v)) valid = v[1]
        else if (match($0, /Valid loss:[[:space:]]*tensor\(([0-9eE+\-\.]+)/, v2)) valid = v2[1]
        else if (match($0, /Valid loss:[[:space:]]*([0-9eE+\-\.]+)/, v3)) valid = v3[1]

        if (epoch != "" && train != "" && valid != "") {
          printf "%s,%.4f,%.4f\n", epoch, train+0.0, valid+0.0
        }
      }
    ' "$f"
  } > "$outfile"
done

# Keep the old cleanup behavior if you still want it
rm -f *old.csv

