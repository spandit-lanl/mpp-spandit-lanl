#!/usr/bin/env python3
import glob
import os
import shlex
from pathlib import Path

ROOT = Path(".").resolve()

LIST_GLOB = "run_*.list"

RUN_ROOT = Path("runs/lsc")
PRETRAIN_SUBDIR = Path("basic_config")
FINETUNE_SUBDIR = Path("finetune")  # change to Path("finetune_resume") if that's your actual layout

def parse_line(line: str):
    """
    Parse a command line like:
      python train_basic.py --run_name X --config Y --yaml_config Z &>> LOG
    Returns dict with keys: run_name, yaml_path, log_path
    """
    # Split respecting quotes
    toks = shlex.split(line)

    # Extract flag values
    def get_flag_value(flag: str):
        if flag in toks:
            i = toks.index(flag)
            if i + 1 < len(toks):
                return toks[i + 1]
        return None

    run_name = get_flag_value("--run_name")
    yaml_path = get_flag_value("--yaml_config")

    # Find &>> redirection target (last token after &>>)
    log_path = None
    if "&>>" in toks:
        i = toks.index("&>>")
        if i + 1 < len(toks):
            log_path = toks[i + 1]

    return {
        "run_name": run_name,
        "yaml_path": yaml_path,
        "log_path": log_path,
    }

def expected_run_dir(run_name: str):
    rn = run_name.lower()
    if "lsc" not in rn:
        return None

    if "pretrain" in rn:
        return RUN_ROOT / PRETRAIN_SUBDIR / run_name

    # Match finetune_resume before finetune (since it contains finetune)
    if "finetune_resume" in rn:
        return RUN_ROOT / FINETUNE_SUBDIR / run_name

    if "finetune" in rn:
        return RUN_ROOT / FINETUNE_SUBDIR / run_name

    return None

def main():
    list_files = sorted(glob.glob(LIST_GLOB))
    if not list_files:
        print(f"ERROR: No files match {LIST_GLOB} in {ROOT}")
        raise SystemExit(2)

    missing_yaml = []
    missing_log_dirs = []
    missing_run_dirs = []
    bad_lines = []

    total_cmds = 0

    for lf in list_files:
        p = Path(lf)
        for lineno, raw in enumerate(p.read_text().splitlines(), start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "python" not in line:
                continue

            total_cmds += 1
            info = parse_line(line)

            run_name = info["run_name"]
            yaml_path = info["yaml_path"]
            log_path = info["log_path"]

            if not run_name or not yaml_path or not log_path:
                bad_lines.append((lf, lineno, raw))
                continue

            y = Path(yaml_path)
            if not y.exists():
                missing_yaml.append((lf, lineno, yaml_path))

            logp = Path(log_path)
            if not logp.parent.exists():
                missing_log_dirs.append((lf, lineno, str(logp.parent)))

            exp = expected_run_dir(run_name)
            if exp is not None and not exp.exists():
                missing_run_dirs.append((lf, lineno, str(exp)))

    print(f"Scanned {len(list_files)} list files, {total_cmds} commands.\n")

    def report(title, items, max_show=200):
        print(f"{title}: {len(items)}")
        for i, item in enumerate(items[:max_show], start=1):
            print(f"  {i:>3}. {item}")
        if len(items) > max_show:
            print(f"  ... ({len(items)-max_show} more)")
        print()

    report("BAD LINES (could not parse run_name/yaml/log)", bad_lines)
    report("MISSING YAML FILES", missing_yaml)
    report("MISSING LOG DIRECTORIES (parent dirs)", missing_log_dirs)
    report("MISSING RUN DIRECTORIES", missing_run_dirs)

    if bad_lines or missing_yaml or missing_log_dirs or missing_run_dirs:
        print("RESULT: ❌ Some required paths are missing (or some lines couldn't be parsed).")
        raise SystemExit(1)
    else:
        print("RESULT: ✅ All required YAML paths, log directories, and run directories exist.")
        raise SystemExit(0)

if __name__ == "__main__":
    main()

