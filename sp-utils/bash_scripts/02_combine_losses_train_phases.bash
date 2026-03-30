#!/usr/bin/env bash
set -euo pipefail

# Combine per-run loss CSVs into one wide CSV, mirroring the behavior of 02_combine_losses.py
# for training phases (pretrain, dtrain, finetune).
#
# IMPORTANT BEHAVIOR (as requested):
# - Operates ONLY in the directory where this script resides.
# - Reads ONLY loss_*.csv files from that same directory.
# - Writes combined_losses_train_phases.csv in that same directory.
# - Ignores loss_clx_* files for now (different naming; handled later).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="/users/spandit/projects/artimis/mpp/mpp-spandit-npz-clx/sp-utils/PLOTS_FINAL"
# If the script was copied elsewhere, we still follow the rule: operate in the script's directory.
# LOG_DIR is retained for clarity/documentation.

PATTERN='loss_*.csv'
OUTPUT='combined_losses_train_phases.csv'

# Phase ordering
phase_key() {
  case "$1" in
    pretrain) echo 0;;
    dtrain)   echo 1;;
    finetune) echo 2;;
    *)        echo 9;;
  esac
}

# Validate phasedata for a given phase (mirrors 02_combine_losses.py)
valid_phasedata() {
  local phase="$1" pd="$2"
  case "$phase" in
    pretrain) [[ "$pd" == "pdebenchfull" || "$pd" == "pdebenchpart" ]];;
    dtrain|finetune) [[ "$pd" == "LSC" || "$pd" == "CLX" ]];;
    *) return 1;;
  esac
}

shopt -s nullglob
files=( $PATTERN )
shopt -u nullglob

if (( ${#files[@]} == 0 )); then
  echo "No files match pattern '$PATTERN' in $SCRIPT_DIR" >&2
  exit 1
fi

# Parse filenames and build sortable list
# Expected (same as python script):
#   loss_final_(L|B)_(pretrain|dtrain|finetune)-PHASEDATA_lr-LR_opt-OPT_wd-WD_ns-NN.csv
# Tag used for columns:
#   <phase>-<phasedata>_nsNN
entries_tmp=$(mktemp)
sorted_tmp=$(mktemp)
trap 'rm -f "$entries_tmp" "$sorted_tmp"' EXIT

fname_re='^loss_final_(L|B)_(pretrain|dtrain|finetune)-([A-Za-z0-9]+)_lr-([A-Za-z0-9]+)_opt-([A-Za-z0-9]+)_wd-([A-Za-z0-9]+)_ns-([0-9]{2})\.csv$'

skipped=0
for path in "${files[@]}"; do
  base=$(basename "$path")

  # Ignore loss_clx_* (handled later)
  if [[ "$base" == loss_clx_* ]]; then
    ((skipped++)) || true
    continue
  fi

  if [[ "$base" =~ $fname_re ]]; then
    variant="${BASH_REMATCH[1]}"
    phase="${BASH_REMATCH[2]}"
    phasedata="${BASH_REMATCH[3]}"
    lr="${BASH_REMATCH[4]}"
    opt="${BASH_REMATCH[5]}"
    wd="${BASH_REMATCH[6]}"
    ns="${BASH_REMATCH[7]}"

    if valid_phasedata "$phase" "$phasedata"; then
      pk=$(phase_key "$phase")
      tag="${phase}-${phasedata}_ns${ns}"
      # sort key: (phase_key, phasedata, variant, ns, lr, opt, wd)
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$pk" "$phasedata" "$variant" "$ns" "$lr" "$opt" "$wd" "$path" "$tag" "$phase" >> "$entries_tmp"
    else
      echo "Skipping $base"
      ((skipped++)) || true
    fi
  else
      echo "Skipping $base"
    ((skipped++)) || true
  fi
done

if [[ ! -s "$entries_tmp" ]]; then
  echo "No valid loss_final_*.csv files found in $SCRIPT_DIR (filenames did not match expected format)." >&2
  exit 1
fi

sort -t$'\t' -k1,1n -k2,2 -k3,3 -k4,4 -k5,5 -k6,6 -k7,7 "$entries_tmp" > "$sorted_tmp"

run_paths=()
run_tags=()
run_phases=()
while IFS=$'\t' read -r pk phasedata variant ns lr opt wd path tag phase; do
  run_paths+=("$path")
  run_tags+=("$tag")
  run_phases+=("$phase")
done < "$sorted_tmp"

# Outer-join on epoch across runs.
# Input CSV header must include: epoch,train_loss,valid_loss
# Float formatting to 4 decimals when numeric.
awk -v OUT="$OUTPUT" \
    -v NTAGS="${#run_tags[@]}" \
    -v TAGS_STR="$(IFS=$'\t'; echo "${run_tags[*]}")" \
    'BEGIN{
      FS=","; OFS=",";
      split(TAGS_STR, TAGS, "\t");
      header="epoch";
      for(i=1;i<=NTAGS;i++){
        header = header OFS TAGS[i] "_train_loss" OFS TAGS[i] "_valid_loss";
      }
      print header > OUT;
    }
    function trim(s){ gsub(/^ +| +$/, "", s); return s }
    function fmt4(s,   x){
      s=trim(s);
      if(s=="") return "";
      if(s ~ /^-?[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?$/){
        x = s + 0;
        return sprintf("%.4f", x);
      }
      return s;
    }
    FNR==1{
      file_idx++;
      epoch_col=train_col=valid_col=0;
      for(i=1;i<=NF;i++){
        if($i=="epoch") epoch_col=i;
        else if($i=="train_loss") train_col=i;
        else if($i=="valid_loss") valid_col=i;
      }
      next;
    }
    {
      if(epoch_col==0) next;
      e = $epoch_col;
      if(e ~ /^[0-9]+$/){
        epochs[e]=1;
        key = file_idx SUBSEP e;
        train[key] = (train_col? $train_col : "");
        valid[key] = (valid_col? $valid_col : "");
      }
    }
    END{
      n=0;
      for(e in epochs){ elist[++n]=e }
      for(i=1;i<=n;i++){
        for(j=i+1;j<=n;j++) if(elist[i]+0 > elist[j]+0){ tmp=elist[i]; elist[i]=elist[j]; elist[j]=tmp }
      }
      for(i=1;i<=n;i++){
        e=elist[i];
        line=e;
        for(fi=1;fi<=file_idx;fi++){
          key = fi SUBSEP e;
          line = line OFS fmt4(train[key]) OFS fmt4(valid[key]);
        }
        print line >> OUT;
      }
    }' \
    "${run_paths[@]}"

counts_pre=0; counts_dt=0; counts_ft=0
for ph in "${run_phases[@]}"; do
  case "$ph" in
    pretrain) ((counts_pre++))||true;;
    dtrain) ((counts_dt++))||true;;
    finetune) ((counts_ft++))||true;;
  esac
done

echo "Wrote $SCRIPT_DIR/$OUTPUT with ${#run_tags[@]} runs."
echo "Runs by phase: pretrain=$counts_pre dtrain=$counts_dt finetune=$counts_ft"
if (( skipped > 0 )); then
  echo "Skipped $skipped loss_*.csv file(s) that didn't match expected loss_final_* filename format (and loss_clx_* intentionally)." >&2
fi

