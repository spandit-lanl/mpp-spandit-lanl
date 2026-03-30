import sys
from pathlib import Path
import numpy as np

def print_npz_keys(input_dir):
    input_path = Path(input_dir)
    npz_files = sorted(input_path.glob("*.npz"))

    if not npz_files:
        print(f"No .npz files found in {input_dir}")
        return

    first_npz = npz_files[0]
    print(f"Reading keys from: {first_npz.name}")

    with np.load(first_npz) as data:
        print("Keys in the .npz file:")
        for key in data.files:
            print(f"  - {key}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python print_npz_keys.py <INDIR>")
        sys.exit(1)

    input_directory = sys.argv[1]
    print_npz_keys(input_directory)

