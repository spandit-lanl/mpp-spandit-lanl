#!/usr/bin/env bash
set -euo pipefail

# Runs run_mpp.bash over the full Cartesian product of args.
# Default: dry-run (prints commands from run_mpp.bash). Set EXECUTE=1 to actually run.
#
# Usage:
#   chmod +x run_all.sh
#   ./run_all.sh
#   EXECUTE=1 ./run_all.sh         # actually run jobs
#   DS_LIST="lsc" EXECUTE=1 ./run_all.sh   # restrict dataset(s)
#
# Assumes run_mpp.bash is in the same directory as this script; adjust RUN_SCRIPT if not.

RUN_SCRIPT="${RUN_SCRIPT:-./run_mpp.bash}"
EXECUTE="${EXECUTE:-0}"   # 0=dry-run, 1=execute

# Parameter grids (override via env vars if you want)
DS_LIST=${DS_LIST:-"lsc clx"}
MS_LIST=${MS_LIST:-"B L"}     # MS_LIST=${MS_LIST:-"ti B L"}
NS_LIST=${NS_LIST:-"1"}       # NS_LIST=${NS_LIST:-"1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16"}
LR_LIST=${LR_LIST:-"4 5"}       # LR_LIST=${LR_LIST:-"4 5"}
OPT_LIST=${OPT_LIST:-"adam adan sgd"}
WD_LIST=${WD_LIST:-"3 4"}       # WD_LIST=${WD_LIST:-"3 4"}
CFG_LIST=${CFG_LIST:-"pretrain finetune finetune_resume"}

if [[ ! -x "$RUN_SCRIPT" ]]; then
  echo "Error: RUN_SCRIPT not found or not executable: $RUN_SCRIPT"
  echo "Tip: chmod +x run_mpp.bash"
  exit 1
fi

# Optional: write a log of everything printed by run_mpp.bash
OUT_LIST="${OUT_LIST:-all_commands.list}"
: > "$OUT_LIST"

xflag=()
if [[ "$EXECUTE" == "1" ]]; then
  xflag=(-x)
fi

for ds in $DS_LIST; do
  for ms in $MS_LIST; do
    for ns in $NS_LIST; do
      for lr in $LR_LIST; do
        for opt in $OPT_LIST; do
          for wd in $WD_LIST; do
            for cfg in $CFG_LIST; do
              # run_mpp.bash prints the final python command; we also tee it into a list file
              "$RUN_SCRIPT" -d "$ds" -m "$ms" -n "$ns" -l "$lr" -o "$opt" -w "$wd" -c "$cfg" "${xflag[@]}" \
                | tee -a "$OUT_LIST"
              #echo "$RUN_SCRIPT" -d "$ds" -m "$ms" -n "$ns" -l "$lr" -o "$opt" -w "$wd" -c "$cfg" "${xflag[@]}" \
              #  | tee -a "$OUT_LIST"
            done
          done
        done
      done
    done
  done
done

echo "Done. Commands captured in: $OUT_LIST"
if [[ "$EXECUTE" != "1" ]]; then
  echo "Note: This was a dry-run. Set EXECUTE=1 to actually run."
fi

