#!/usr/bin/env bash
set -euo pipefail

# Combine CLX experiment loss CSVs (loss_clx_expt_*.csv) into one wide CSV.
# - Looks only in the directory this script is in
# - Writes combined_losses_clx_expt.csv in the same directory
# - Outer-joins on epoch across runs
# - Columns: epoch, <tag>_train_loss, <tag>_valid_loss for each run
#
# Expected input filenames:
#   loss_clx_expt_ns-01_var-1.csv
#   loss_clx_expt_ns-06_var-8.csv
#
# Expected CSV header:
#   epoch,train_loss,valid_loss

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

OUT_CSV="combined_losses_clx_expt.csv"

shopt -s nullglob
FILES=( loss_clx_expt_*.csv )
shopt -u nullglob

skipped_tmp="$(mktemp)"
runs_tmp="$(mktemp)"
trap 'rm -f "$skipped_tmp" "$runs_tmp"' EXIT

if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "No loss_clx_expt_*.csv files found in: $SCRIPT_DIR" >&2
  exit 1
fi

# Collect runs: ns, var, file, tag
# Sort key: ns (numeric), var (numeric)
for f in "${FILES[@]}"; do
  b="$(basename "$f")"
  if [[ "$b" =~ ^loss_clx_expt_ns-([0-9]{2})_var-([0-9]+)\.csv$ ]]; then
    ns="${BASH_REMATCH[1]}"
    var="${BASH_REMATCH[2]}"
    tag="clx_expt_ns${ns}_var${var}"
    # sanity: file has at least 2 lines
    if [[ ! -s "$f" ]] || [[ "$(wc -l < "$f")" -lt 2 ]]; then
      echo "$b -> empty or header-only CSV" >> "$skipped_tmp"
      continue
    fi
    printf "%s\t%s\t%s\t%s\n" "$ns" "$var" "$f" "$tag" >> "$runs_tmp"
  else
    echo "$b -> filename did not match expected pattern loss_clx_expt_ns-##_var-#.csv" >> "$skipped_tmp"
  fi
done

if [[ ! -s "$runs_tmp" ]]; then
  echo "No valid CLX loss CSVs to combine." >&2
  if [[ -s "$skipped_tmp" ]]; then
    echo "--- skipped file list (filename -> reason) ---" >&2
    cat "$skipped_tmp" >&2
  fi
  exit 1
fi

# Sort runs
sorted_tmp="$(mktemp)"
trap 'rm -f "$skipped_tmp" "$runs_tmp" "$sorted_tmp"' EXIT
sort -t $'\t' -k1,1n -k2,2n "$runs_tmp" > "$sorted_tmp"

mapfile -t RUN_LINES < "$sorted_tmp"

# Build header
header="epoch"
tags=()
files=()
for line in "${RUN_LINES[@]}"; do
  IFS=$'\t' read -r ns var f tag <<< "$line"
  files+=( "$f" )
  tags+=( "$tag" )
  header+=",${tag}_train_loss,${tag}_valid_loss"
done

# Combine with gawk (outer join on epoch)
gawk -v header="$header" -v filelist="$(IFS=$'\n'; echo "${files[*]}")" -v taglist="$(IFS=$'\n'; echo "${tags[*]}")" '
BEGIN {
  print header
  nfiles = split(filelist, files, "\n")
  ntags  = split(taglist, tags, "\n")
  if (nfiles != ntags) {
    print "ERROR: internal mismatch between files and tags" > "/dev/stderr"
    exit 2
  }
  # column indices: for file i, train col = 2*i-1, valid col = 2*i
  for (i=1; i<=nfiles; i++) {
    train_col[i] = 2*i - 1
    valid_col[i] = 2*i
  }
}

function trim(s) { gsub(/^[ \t\r\n]+|[ \t\r\n]+$/, "", s); return s }

function fmt(x,  y) {
  x = trim(x)
  if (x == "" || x ~ /^(nan|NaN|None|null)$/) return ""
  # If it looks numeric, format to 4 decimals
  if (x ~ /^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$/) {
    y = x + 0
    return sprintf("%.4f", y)
  }
  return x
}

# Read each file sequentially
{
  # Not used: we handle reading via getline in END
}

END {
  # Load all files
  for (i=1; i<=nfiles; i++) {
    f = files[i]
    # Read header line
    if ((getline line < f) <= 0) { close(f); continue }
    # Determine indices by header names (robust to column order)
    n = split(line, h, ",")
    epoch_idx = train_idx = valid_idx = 0
    for (j=1; j<=n; j++) {
      hj = trim(h[j])
      if (hj == "epoch") epoch_idx = j
      else if (hj == "train_loss") train_idx = j
      else if (hj == "valid_loss") valid_idx = j
    }
    if (epoch_idx == 0) epoch_idx = 1
    if (train_idx == 0) train_idx = 2
    if (valid_idx == 0) valid_idx = 3

    while ((getline line < f) > 0) {
      if (line ~ /^[ \t\r\n]*$/) continue
      m = split(line, a, ",")
      e = trim(a[epoch_idx])
      if (e == "" || e !~ /^[0-9]+$/) continue
      epochs[e] = 1
      t = (train_idx <= m) ? a[train_idx] : ""
      v = (valid_idx <= m) ? a[valid_idx] : ""
      data[e, train_col[i]] = fmt(t)
      data[e, valid_col[i]] = fmt(v)
    }
    close(f)
  }

  # Sort epochs numerically
  ne = 0
  for (e in epochs) { epoch_list[++ne] = e + 0 }
  asort(epoch_list)

  # Emit rows
  for (k=1; k<=ne; k++) {
    e = epoch_list[k]
    row = e
    # append 2*nfiles columns
    for (i=1; i<=nfiles; i++) {
      tc = train_col[i]
      vc = valid_col[i]
      row = row "," data[e, tc] "," data[e, vc]
    }
    print row
  }
}
' /dev/null > "$OUT_CSV"

# Report skipped
if [[ -s "$skipped_tmp" ]]; then
  echo "Skipped $(wc -l < "$skipped_tmp") file(s):" >&2
  echo "--- skipped file list (filename -> reason) ---" >&2
  cat "$skipped_tmp" >&2
fi

echo "Wrote: $OUT_CSV"

