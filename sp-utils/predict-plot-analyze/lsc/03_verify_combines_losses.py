#!/usr/bin/env python3
import argparse
import csv
import glob
import os
import re

FNAME_RE = re.compile(
    r"^loss_(pretrain|finetune|direct_train)"
    r"(?:_([A-Za-z0-9]+))?"
    r"_(L|B)"
    r"_(\d+)"
    r"(?:_(.*))?\.csv$"
)

def parse_args():
    p = argparse.ArgumentParser(description="Verify combined_losses.csv matches original loss_*.csv files.")
    p.add_argument("--pattern", default="loss_*.csv")
    p.add_argument("--combined", default="combined_losses.csv")
    return p.parse_args()

def filename_to_tag(name: str):
    m = FNAME_RE.match(name)
    if not m:
        return None
    phase, prefix, variant, ns_str, rest = m.groups()
    ns = int(ns_str)
    rest = rest or ""
    if prefix:
        tag_base = f"{phase}_{prefix}_{variant}_{ns:02d}"
    else:
        tag_base = f"{phase}_{variant}_{ns:02d}"
    return tag_base + (f"_{rest}" if rest else "")

def load_combined(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        rows = {int(r["Epoch"]): r for r in reader}
    return header, rows

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
            skipped_files += 1
            continue

        matched_files += 1
        train_col = f"{tag}_TrainLoss"
        valid_col = f"{tag}_ValidLoss"

        file_issues = []
        if train_col not in header or valid_col not in header:
            file_issues.append(f"Missing columns {train_col} / {valid_col} in {args.combined}")
        else:
            with open(path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    epoch = int(row["Epoch"])
                    train = (row.get("TrainLoss") or "").strip()
                    valid = (row.get("ValidLoss") or "").strip()

                    comb_row = combined_rows.get(epoch)
                    if comb_row is None:
                        file_issues.append(f"Epoch {epoch} missing in combined file")
                        continue

                    comb_train = (comb_row.get(train_col) or "").strip()
                    comb_valid = (comb_row.get(valid_col) or "").strip()

                    if train:
                        if comb_train != train:
                            file_issues.append(f"Epoch {epoch} TrainLoss mismatch (orig={train}, combined={comb_train})")
                        else:
                            total_values_checked += 1

                    if valid:
                        try:
                            expected_valid = f"{float(valid):.4f}"
                        except ValueError:
                            expected_valid = valid

                        if comb_valid != expected_valid:
                            file_issues.append(
                                f"Epoch {epoch} ValidLoss mismatch (orig={valid} -> {expected_valid}, combined={comb_valid})"
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
    print(f"Checked {total_values_checked} non-empty values across {matched_files} matched files (skipped {skipped_files}).")
    if total_issues:
        print(f"Total mismatches / issues found: {total_issues}")
    else:
        print("All checked values match the combined file ✅")

if __name__ == "__main__":
    main()

