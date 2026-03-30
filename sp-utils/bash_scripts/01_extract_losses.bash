#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

for f in ./../*log ./../*.txt; do
  base=${f##*/}          # filename only
  base=${base%.txt}
  base=${base%.log}

  # Strip leading "out_" if present for the output stem
  stem="${base#out_}"

  # If stem starts with pretrain_ or finetune_:
  # - if it already has _L_ immediately after phase, keep it
  # - otherwise inject _B_ immediately after phase
  if [[ "$stem" =~ ^(pretrain|finetune)_(L)_ ]]; then
    # already has _L_
    outstem="$stem"
  elif [[ "$stem" =~ ^(pretrain|finetune)_(B)_ ]]; then
    # already has _B_
    outstem="$stem"
  elif [[ "$stem" =~ ^(pretrain|finetune)_ ]]; then
    # does not have _L_ or _B_ after phase -> insert _B_
    phase="${BASH_REMATCH[1]}"
    rest="${stem#${phase}_}"
    outstem="${phase}_B_${rest}"
  else
    # fallback: just use stem as-is
    outstem="$stem"
  fi

  outfile="loss_${outstem}.csv"

  echo "writing $outfile"

  {
    echo "Epoch,TrainLoss,ValidLoss"

    awk '
      BEGIN { OFS="," }
      FNR == 1 { seq = 0 }

      /Valid loss/ {
        seq++

        # epoch: use "Epoch: N" if present, else sequential counter
        epoch = seq
        if (match($0, /Epoch:[[:space:]]*([0-9]+)/, m)) epoch = m[1]

        # train loss from tensor([ ... ])
        train = ""
        if (match($0, /Train loss:[[:space:]]*tensor\(\[([0-9.eE+-]+)/, t)) train = t[1]

        # valid loss number after "Valid loss:"
        valid = ""
        if (match($0, /Valid loss:[[:space:]]*([0-9.eE+-]+)/, v)) valid = v[1]

        # print only if we found both numbers (skip malformed lines)
        if (train != "" && valid != "")
          print epoch, train, valid
      }
    ' "$f"
  } > "$outfile"
done

rm -f *old.csv

