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

def inspect_npz_structure(file_path):
    try:
        # Load the .npz file
        with np.load(file_path) as data:
            print(f"Inspecting file: {file_path}")
            print(f"{'-'*40}")

            # Loop through all arrays inside
            for key in data.files:
                array = data[key]
                print(f"Key: '{key}': ", end=' ')
                print(f"  Shape: {array.shape}", end=' ')
                print(f"  Dtype: {array.dtype}")
                #print(f"{'-'*40}")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")


if __name__ == "__main__":
#    if len(sys.argv) < 2:
#        print("Usage: python print_npz_keys.py <INDIR>")
#        sys.exit(1)

    inspect_npz_structure("input.npz")

#    input_directory = sys.argv[1]
#    print_npz_keys(input_directory)

