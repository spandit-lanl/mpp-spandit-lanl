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

# New style (extract_losses.bash output):
#   loss_pretrain_B_01_lr-3_opt-adam_wd-3.csv
#   loss_pretrain_L_01_lr-3_opt-adam_wd-3.csv
#   loss_finetune_B_02_....
#   loss_finetune_L_02_....
NEW_BL_FNAME_RE = re.compile(
    r"^loss_(pretrain|finetune)_(B|L)_(\d+)(?:_(.*))?\.csv$"
)

# Optional backwards-compat (if any exist):
#   loss_pretrain_01_lr-3_opt-adam_wd-3.csv
NEW_NOFLAG_FNAME_RE = re.compile(r"^loss_(pretrain|finetune)_(\d+)(?:_(.*))?\.csv$")


def parse_args():
    p = argparse.ArgumentParser(
        description="Verify combined_losses.csv matches the original loss_*.csv files."
    )
    p.add_argument(
        "--pattern",
        default="loss_*.csv",
        help="Glob pattern for input CSVs (default: loss_*.csv)",
    )
    p.add_argument(
        "--combined",
        default="combined_losses.csv",
        help="Combined CSV filename (default: combined_losses.csv)",
    )
    return p.parse_args()


def load_combined(path):
    """Load combined CSV into dict: epoch -> row dict."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = {}
        for row in reader:
            epoch = int(row["Epoch"])
            rows[epoch] = row
        header = reader.fieldnames or []
    return header, rows


def filename_to_tag(name: str):
    """
    Must match 02_combine_losses.py tag rules.

    Old style -> tag = f"{phase}_n{nstep:02d}"
    New B/L   -> tag = f"{phase}_{variant}_{runid:02d}" + (f"_{extra}" if extra else "")
    No-flag   -> tag = f"{phase}_{runid:02d}" + (f"_{extra}" if extra else "")
    """
    m = OLD_FNAME_RE.match(name)
    if m:
        phase, nstep_str = m.groups()
        nstep = int(nstep_str)
        return f"{phase}_n{nstep:02d}"

    m = NEW_BL_FNAME_RE.match(name)
    if m:
        phase, variant, runid_str, extra = m.groups()
        runid = int(runid_str)
        extra = extra or ""
        return f"{phase}_{variant}_{runid:02d}" + (f"_{extra}" if extra else "")

    m = NEW_NOFLAG_FNAME_RE.match(name)
    if m:
        phase, runid_str, extra = m.groups()
        runid = int(runid_str)
        extra = extra or ""
        return f"{phase}_{runid:02d}" + (f"_{extra}" if extra else "")

    return None


def main():
    args = parse_args()

    if not os.path.exists(args.combined):
        raise SystemExit(f"Combined file {args.combined!r} not found.")

    header, combined_rows = load_combined(args.combined)

    files = sorted(glob.glob(args.pattern))
    if not files:
        raise SystemExit(f"No files match pattern {args.pattern!r}")

    total_values_checked = 0
    total_issues = 0
    matched_files = 0
    skipped_files = 0

    for path in files:
        name = os.path.basename(path)
        tag = filename_to_tag(name)
        if not tag:
            print(f"Skipping {name}: does not match expected patterns.")
            skipped_files += 1
            continue

        matched_files += 1
        train_col = f"{tag}_TrainLoss"
        valid_col = f"{tag}_ValidLoss"

        file_issues = []

        if train_col not in header or valid_col not in header:
            file_issues.append(
                f"Columns {train_col} / {valid_col} not found in {args.combined}"
            )
        else:
            with open(path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    epoch = int(row["Epoch"])
                    train = row["TrainLoss"].strip()
                    valid = row["ValidLoss"].strip()

                    comb_row = combined_rows.get(epoch)
                    if comb_row is None:
                        file_issues.append(f"Epoch {epoch} missing in combined file")
                        continue

                    comb_train = (comb_row.get(train_col) or "").strip()
                    comb_valid = (comb_row.get(valid_col) or "").strip()

                    # TrainLoss: expect exact match if non-empty
                    if train:
                        if comb_train != train:
                            file_issues.append(
                                f"Epoch {epoch} TrainLoss mismatch "
                                f"(orig={train}, combined={comb_train})"
                            )
                        else:
                            total_values_checked += 1

                    # ValidLoss: combined stores value rounded to 4 decimals if possible
                    if valid:
                        try:
                            expected_valid = f"{float(valid):.4f}"
                        except ValueError:
                            expected_valid = valid

                        if comb_valid != expected_valid:
                            file_issues.append(
                                f"Epoch {epoch} ValidLoss mismatch "
                                f"(orig={valid} -> {expected_valid}, combined={comb_valid})"
                            )
                        else:
                            total_values_checked += 1

        if file_issues:
            total_issues += len(file_issues)
            print(f"❌ {name} — {len(file_issues)} issue(s):")
            for msg in file_issues:
                print(f"   - {msg}")
        else:
            print(f"✅ {name} — validated")

    print()
    print(
        f"Checked {total_values_checked} non-empty values across "
        f"{matched_files} matched files (skipped {skipped_files})."
    )
    if total_issues:
        print(f"Total mismatches / issues found: {total_issues}")
    else:
        print("All checked values match the combined file ✅")


if __name__ == "__main__":
    main()

