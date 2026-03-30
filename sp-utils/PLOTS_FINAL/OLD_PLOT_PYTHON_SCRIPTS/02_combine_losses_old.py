#!/usr/bin/env python3
import argparse
import csv
import glob
import os
import re
from typing import Optional, Tuple, Dict, Any, List

# New format produced by 01_extract_losses.bash:
#   loss_final_L_<phase>-<phasedata>_lr-X_opt-adan_wd-3_ns-<NN>.csv
FNAME_RE = re.compile(
    r"^loss_final_(L|B)_"
    r"(pretrain|dtrain|finetune)-([A-Za-z0-9]+)"
    r"_lr-([A-Za-z0-9]+)"
    r"_opt-([A-Za-z0-9]+)"
    r"_wd-([A-Za-z0-9]+)"
    r"_ns-([0-9]{2})"
    r"\.csv$"
)

VALID_PHASEDATA = {
    "pretrain": {"pdebenchfull", "pdebenchpart"},
    "dtrain": {"LSC", "CLX"},
    "finetune": {"LSC", "CLX"},
}

PHASE_SORT_KEY = {"pretrain": 0, "dtrain": 1, "finetune": 2}


def parse_args():
    p = argparse.ArgumentParser(description="Combine per-run loss CSVs into one wide CSV.")
    p.add_argument(
        "--pattern",
        default="**/loss_final_*.csv",
        help="Glob for loss csvs. Default searches recursively: **/loss_final_*.csv",
    )
    p.add_argument("--output", default="combined_losses.csv")
    return p.parse_args()


def parse_filename(basename: str) -> Optional[Tuple[Tuple[Any, ...], str, str]]:
    """
    Returns (sort_key, tag, phase) or None if not matching/invalid.
    tag is the base used for column prefixes.
    """
    m = FNAME_RE.match(basename)
    if not m:
        return None

    variant, phase, phasedata, lr, opt, wd, ns_str = m.groups()
    ns = int(ns_str)

    # Validate phase/phasedata consistency
    allowed = VALID_PHASEDATA.get(phase)
    if allowed is None or phasedata not in allowed:
        return None

    # Column base tag:
    # drop 'final_L_' and drop '_lrX_optadan_wd3_' from columns
    # => <phase>-<phasedata>_nsNN
    tag = f"{phase}-{phasedata}_ns{ns:02d}"

    sort_key = (PHASE_SORT_KEY[phase], phasedata, variant, ns, lr, opt, wd)
    return sort_key, tag, phase


def main():
    args = parse_args()

    files = sorted(glob.glob(args.pattern, recursive=True))
    if not files:
        raise SystemExit(f"No files match pattern {args.pattern!r}")

    all_epochs = set()
    runs: List[Dict[str, Any]] = []
    skipped: List[str] = []

    for path in files:
        #print (f"path : {path}", flush=True)
        base = os.path.basename(path)
        parsed = parse_filename(base)
        if not parsed:
            skipped.append(base)
            continue

        sort_key, tag, phase = parsed
        data: Dict[int, Tuple[str, str]] = {}

        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            # New lowercase headers from 01_extract_losses.bash
            if reader.fieldnames is None or "epoch" not in reader.fieldnames:
                skipped.append(base)
                continue

            for row in reader:
                try:
                    epoch = int(row["epoch"])
                except Exception:
                    continue

                train = (row.get("train_loss") or "").strip()
                valid = (row.get("valid_loss") or "").strip()
                data[epoch] = (train, valid)
                all_epochs.add(epoch)

        runs.append({"sort_key": sort_key, "tag": tag, "phase": phase, "data": data, "path": path})

    if not runs:
        raise SystemExit("No valid loss_final_*.csv files found (filenames/headers did not match expected format).")

    runs.sort(key=lambda r: r["sort_key"])
    epochs_sorted = sorted(all_epochs)

    header = ["epoch"]
    for r in runs:
        #print (f"run {r}", flush=True)
        header += [f"{r['tag']}_train_loss", f"{r['tag']}_valid_loss"]

    with open(args.output, "w", newline="") as out_f:
        writer = csv.writer(out_f)
        writer.writerow(header)

        for epoch in epochs_sorted:
            #print (f"\t epoch {epoch}", flush=True)
            row = [epoch]
            for r in runs:
                #print (f"\t Epoch {r}", flush=True)
                vals = r["data"].get(epoch)
                if vals is None:
                    row.extend(["", ""])
                else:
                    train_str, valid_str = vals

                    # keep 4 decimals if parseable
                    try:
                        train_out = f"{float(train_str):.4f}" if train_str != "" else ""
                    except ValueError:
                        train_out = train_str

                    if valid_str == "":
                        valid_out = ""
                    else:
                        try:
                            valid_out = f"{float(valid_str):.4f}"
                        except ValueError:
                            valid_out = valid_str

                    row.extend([train_out, valid_out])
            writer.writerow(row)

    counts = {"pretrain": 0, "dtrain": 0, "finetune": 0}
    for r in runs:
        counts[r["phase"]] += 1

    print(f"Wrote {args.output} with {len(epochs_sorted)} epochs and {len(runs)} runs.")
    print(f"Runs by phase: {counts}")
    if skipped:
        print(f"Skipped {len(skipped)} file(s) that didn’t match expected pattern/headers.")


if __name__ == "__main__":
    main()

