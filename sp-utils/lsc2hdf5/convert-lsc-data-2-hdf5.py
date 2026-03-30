import os
import sys
import numpy as np
import h5py
import hashlib
import json
import shutil
from pathlib import Path
from datetime import datetime

OUTDIR = "OUTDIR"
os.makedirs(OUTDIR, exist_ok=True)

TARGET_KEYS = {
    "density_case",
    "density_cushion",
    "density_maincharge",
    "density_outside_air",
    "density_striker",
    "density_throw",
    "Uvelocity",
    "Wvelocity"
}

def sha256_checksum(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes()).hexdigest()

def verify_conversion(npz_data, h5_path):
    with h5py.File(h5_path, 'r') as h5f:
        npz_keys = set(npz_data.files)
        h5_keys = set(h5f.keys())
        filtered_keys = TARGET_KEYS.intersection(npz_keys)

        if filtered_keys != h5_keys:
            print(f"❌ Key mismatch. Expected: {filtered_keys}, Found: {h5_keys}")
            return False

        for key in filtered_keys:
            npz_array = npz_data[key]
            h5_array = h5f[key][()]

            if npz_array.shape != h5_array.shape:
                print(f"❌ Shape mismatch for '{key}'")
                return False
            if npz_array.dtype != h5_array.dtype:
                print(f"❌ Dtype mismatch for '{key}'")
                return False
            if npz_array.tobytes() != h5_array.tobytes():
                print(f"❌ Byte-level data mismatch for '{key}'")
                return False

    return True

def process_npz_files(input_dir):
    input_dir = Path(input_dir)
    npz_files = sorted(input_dir.glob("*.npz"))[:4000]

    for npz_file in npz_files:
        base_name = npz_file.stem
        print(f"🔄 Processing {npz_file.name}")

        try:
            npz_data = np.load(npz_file)
            temp_h5_path = npz_file.with_suffix(".tmp.h5")
            final_h5_path = Path(OUTDIR) / f"{base_name}.h5"

            with h5py.File(temp_h5_path, 'w') as h5f:
                checksum_dict = {}

                for key in TARGET_KEYS:
                    if key not in npz_data:
                        print(f"⚠️  Missing key '{key}', skipping.")
                        continue

                    arr = npz_data[key]
                    dset = h5f.create_dataset(key, data=arr, dtype=arr.dtype)
                    dset.attrs["sha256"] = sha256_checksum(arr)
                    checksum_dict[key] = dset.attrs["sha256"]

                h5f.attrs["source_file"] = str(npz_file.name)
                h5f.attrs["created"] = datetime.utcnow().isoformat() + "Z"
                h5f.attrs["checksums"] = json.dumps(checksum_dict)

            # Run verification before moving
            if verify_conversion(npz_data, temp_h5_path):
                shutil.move(temp_h5_path, final_h5_path)
                print(f"✅ Successfully converted and verified → {final_h5_path.name}\n")
            else:
                os.remove(temp_h5_path)
                print(f"❌ Conversion failed for {npz_file.name}, no output written\n")

        except Exception as e:
            print(f"❌ Error processing {npz_file.name}: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <INDIR>")
        sys.exit(1)

    input_directory = sys.argv[1]
    process_npz_files(input_directory)

