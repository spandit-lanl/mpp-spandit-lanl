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
#  1) clx_expt_*.log     -> loss_clx_expt_*.csv
logs=( "$LOG_DIR"/clx_expt_*.log )
if (( ${#logs[@]} == 0 )); then
  echo "No logs found matching: $LOG_DIR/clx_expt_*.log" >&2
  exit 0
fi

for f in "${logs[@]}"; do
  base="${f##*/}"             # filename only
  base_noext="${base%.log}"   # strip .log

  if [[ "$base_noext" == clx_expt_* ]]; then
    # input: clx_expt_* -> output: loss_clx_expt_*
    outfile="loss_${base_noext}.csv"

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

