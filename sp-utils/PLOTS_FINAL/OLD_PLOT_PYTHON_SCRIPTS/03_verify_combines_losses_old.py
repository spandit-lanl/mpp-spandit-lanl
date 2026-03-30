#!/usr/bin/env python3
import argparse
import csv
import glob
import os
import re

# Per-run file pattern:
# loss_final_L_<phase>-<phasedata>_lr-X_opt-adan_wd-3_ns-<NN>.csv
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

def parse_args():
    p = argparse.ArgumentParser(description="Verify combined_losses.csv matches original loss_final_*.csv files.")
    p.add_argument("--pattern", default="loss_final_*.csv",
                  help="Glob for per-run csvs (default: loss_final_*.csv)")
    p.add_argument("--combined", default="combined_losses.csv",
                  help="Combined file to verify (default: combined_losses.csv)")
    return p.parse_args()

def filename_to_tag(name: str):
    """
    Returns the base tag used in combined column names:
      <phase>-<phasedata>_nsNN
    or None if filename invalid / inconsistent.
    """
    m = FNAME_RE.match(name)
    if not m:
        return None

    variant, phase, phasedata, lr, opt, wd, ns_str = m.groups()

    allowed = VALID_PHASEDATA.get(phase)
    if allowed is None or phasedata not in allowed:
        return None

    ns = int(ns_str)
    return f"{phase}-{phasedata}_ns{ns:02d}"

def load_combined(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        if "epoch" not in header:
            raise SystemExit(f"Combined file {path!r} missing required column 'epoch'.")
        rows = {}
        for r in reader:
            try:
                e = int(r["epoch"])
            except Exception:
                continue
            rows[e] = r
    return header, rows

def fmt4(x: str) -> str:
    x = (x or "").strip()
    if x == "":
        return ""
    try:
        return f"{float(x):.4f}"
    except ValueError:
        return x

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
        train_col = f"{tag}_train_loss"
        valid_col = f"{tag}_valid_loss"

        file_issues = []
        if train_col not in header or valid_col not in header:
            file_issues.append(f"Missing columns {train_col} / {valid_col} in {args.combined}")
        else:
            with open(path, newline="") as f:
                reader = csv.DictReader(f)

                # Per-run CSV headers expected from 01_extract_losses.bash
                if reader.fieldnames is None or "epoch" not in reader.fieldnames:
                    file_issues.append("Per-run CSV missing required header 'epoch' (expected: epoch,train_loss,valid_loss)")
                else:
                    for row in reader:
                        try:
                            epoch = int(row["epoch"])
                        except Exception:
                            continue

                        train = fmt4(row.get("train_loss") or "")
                        valid = fmt4(row.get("valid_loss") or "")

                        comb_row = combined_rows.get(epoch)
                        if comb_row is None:
                            file_issues.append(f"Epoch {epoch} missing in combined file")
                            continue

                        comb_train = fmt4(comb_row.get(train_col) or "")
                        comb_valid = fmt4(comb_row.get(valid_col) or "")

                        if train != "":
                            if comb_train != train:
                                file_issues.append(
                                    f"Epoch {epoch} train_loss mismatch (orig={train}, combined={comb_train})"
                                )
                            else:
                                total_values_checked += 1

                        if valid != "":
                            if comb_valid != valid:
                                file_issues.append(
                                    f"Epoch {epoch} valid_loss mismatch (orig={valid}, combined={comb_valid})"
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

