#!/usr/bin/env python3
import argparse
import csv
import glob
import os
import re

# Old style:
#   loss_pretrain_b_MPP_nsteps_01.csv
#   loss_finetune_b_LSC_nsteps_05.csv
OLD_FNAME_RE = re.compile(r"^loss_(pretrain|finetune)_.*_nsteps_(\d+)\.csv$")

# New style (your updated extractor output):
#   loss_pretrain_B_01_lr-3_opt-adam_wd-3.csv
#   loss_pretrain_L_01_lr-3_opt-adam_wd-3.csv
#   loss_finetune_B_02_....
#   loss_finetune_L_02_....
NEW_BL_FNAME_RE = re.compile(r"^loss_(pretrain|finetune)_(B|L)_(\d+)(?:_(.*))?\.csv$")

# Optional backwards-compat support (in case you still have any):
#   loss_pretrain_01_lr-3_opt-adam_wd-3.csv
NEW_NOFLAG_FNAME_RE = re.compile(r"^loss_(pretrain|finetune)_(\d+)(?:_(.*))?\.csv$")


def parse_args():
    p = argparse.ArgumentParser(
        description="Combine per-run loss CSVs into a single wide CSV."
    )
    p.add_argument(
        "--pattern",
        default="loss_*.csv",
        help="Glob pattern for input CSVs (default: loss_*.csv)",
    )
    p.add_argument(
        "--output",
        default="combined_losses.csv",
        help="Output CSV filename (default: combined_losses.csv)",
    )
    return p.parse_args()


def parse_filename(name: str):
    """
    Return:
      sort_key: tuple used for stable ordering
      tag: string used in output column names (must be unique)
    """
    m = OLD_FNAME_RE.match(name)
    if m:
        phase, nstep_str = m.groups()
        nstep = int(nstep_str)
        tag = f"{phase}_n{nstep:02d}"
        # sort: pretrain first, then finetune, then old-style nstep
        sort_key = (0 if phase == "pretrain" else 1, 0, nstep, "")
        return sort_key, tag

    m = NEW_BL_FNAME_RE.match(name)
    if m:
        phase, variant, runid_str, extra = m.groups()
        runid = int(runid_str)
        extra = extra or ""
        # include B/L so B and L runs don't collide
        tag = f"{phase}_{variant}_{runid:02d}" + (f"_{extra}" if extra else "")
        # sort: pretrain first, then finetune; B before L; then runid; then extra
        variant_key = 0 if variant == "B" else 1
        sort_key = (0 if phase == "pretrain" else 1, 1 + variant_key, runid, extra)
        return sort_key, tag

    m = NEW_NOFLAG_FNAME_RE.match(name)
    if m:
        phase, runid_str, extra = m.groups()
        runid = int(runid_str)
        extra = extra or ""
        tag = f"{phase}_{runid:02d}" + (f"_{extra}" if extra else "")
        # sort after the explicit B/L runs
        sort_key = (0 if phase == "pretrain" else 1, 3, runid, extra)
        return sort_key, tag

    return None


def main():
    args = parse_args()

    files = sorted(glob.glob(args.pattern))
    if not files:
        raise SystemExit(f"No files match pattern {args.pattern!r}")

    all_epochs = set()
    runs = []  # each: {sort_key, tag, data: {epoch: (train_str, valid_str)}}

    for path in files:
        name = os.path.basename(path)
        parsed = parse_filename(name)
        if not parsed:
            # skip unexpected filenames
            continue

        sort_key, tag = parsed

        data = {}
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                epoch = int(row["Epoch"])
                train = row["TrainLoss"].strip()
                valid = row["ValidLoss"].strip()
                data[epoch] = (train, valid)
                all_epochs.add(epoch)

        runs.append({"sort_key": sort_key, "tag": tag, "data": data})

    if not runs:
        raise SystemExit("No valid loss_*.csv files found.")

    runs.sort(key=lambda r: r["sort_key"])

    # Header
    header = ["Epoch"]
    for r in runs:
        tag = r["tag"]
        header.append(f"{tag}_TrainLoss")
        header.append(f"{tag}_ValidLoss")

    epochs_sorted = sorted(all_epochs)

    with open(args.output, "w", newline="") as out_f:
        writer = csv.writer(out_f)
        writer.writerow(header)

        for epoch in epochs_sorted:
            row = [epoch]
            for r in runs:
                vals = r["data"].get(epoch)
                if vals is None:
                    # Experiment hasn't reached this epoch yet
                    row.extend(["", ""])
                else:
                    train_str, valid_str = vals

                    # TrainLoss: keep as in original file
                    train_out = train_str

                    # ValidLoss: round to 4 decimal places if possible
                    if valid_str == "":
                        valid_out = ""
                    else:
                        try:
                            valid_out = f"{float(valid_str):.4f}"
                        except ValueError:
                            valid_out = valid_str

                    row.extend([train_out, valid_out])

            writer.writerow(row)

    print(
        f"Wrote {args.output} with {len(epochs_sorted)} epochs and {len(runs)} runs "
        f"(ValidLoss rounded to 4 decimal places)."
    )


if __name__ == "__main__":
    main()

