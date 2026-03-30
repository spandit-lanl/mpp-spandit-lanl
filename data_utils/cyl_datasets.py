# File: data_utils/npz_datasets.py

import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset

'''
IMAGE_WIDTH_HALF_IMAGE = 560
IMAGE_WIDTH_FULL_IMAGE = 1120

PATCH_SIZE = 16
IMAGE_HEIGHT_RAW = 400
IMAGE_HEIGHT_CROPPED = (IMAGE_HEIGHT_RAW // PATCH_SIZE) * PATCH_SIZE
IMAGE_HEIGHT_PADDED  = ((IMAGE_HEIGHT_RAW // PATCH_SIZE) + 1) * PATCH_SIZE

#IMAGE_WIDTH = IMAGE_WIDTH_FULL_IMAGE
#IMAGE_HEIGHT = IMAGE_HEIGHT_PADDED
#IMAGE_HEIGHT = IMAGE_HEIGHT_CROPPED
'''

IMAGE_WIDTH = 400
IMAGE_HEIGHT = 1120

class CylNpzDataset(Dataset):
    def __init__(self,
                 path,
                 include_string='',
                 n_steps=5,
                 dt=1,
                 split='train',
                 train_val_test=(0.8, 0.1, 0.1),
                 subname=None,
                 extra_specific=False):

        self.root_dir = path
        self.n_steps = n_steps
        self.dt = dt
        self.split = split
        self.train_val_test = train_val_test
        self.include_string = include_string
        self.type = 'cyl_npz'
        self.field_names = self._specifics()[2]
        self.title = 'cyl_npz'

        self.file_list = sorted([
            f for f in os.listdir(self.root_dir)
            if f.endswith(".npz") and include_string in f
        ], key=self._extract_timestep)

        # Determine split range
        total = len(self.file_list) - (n_steps + dt - 1)
        train_end = int(train_val_test[0] * total)
        val_end = train_end + int(train_val_test[1] * total)

        if split == 'train':
            self.indices = range(0, train_end)
        elif split == 'val':
            self.indices = range(train_end, val_end)
        else:
            self.indices = range(val_end, total)

    def _extract_timestep(self, filename):
        match = re.search(r"idx(\d+)", filename)
        return int(match.group(1)) if match else -1

    def __len__(self):
        return len(self.indices)

    def get_name(self, full_name=False):
        return "cyl_npz"  # or whatever label you want to appear in logs

    def __getitem__(self, idx: int):
        selected_fields = [
			#"Wvelocity", "av_pressure", "av_temperature", "burn_frac_booster", "burn_frac_maincharge",
			#"density_booster", "energy_booster", "pressure_maincharge",
            "av_density", "Uvelocity", "Wvelocity", "density_maincharge",
        ]

        def load_tensor(fpath):
            try:
                with np.load(fpath) as data:
                    arrays = []
                    for key in selected_fields:
                        if key not in data:
                            print(f"[SKIP] Missing key {key} in {fpath}")
                            raise KeyError(f"Missing field '{key}' in file {fpath}")
                            return None
                        arr = data[key]
                        #print(f"DEBUG: {key} raw shape: {arr.shape}")
                        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                        arr = arr.astype(np.float32)
                        #arr = arr.astype(np.float16)  # load fp16 from disk (Option A); convert to fp32 when making torch tensor
                        if arr.ndim != 2:
                            print(f"[SKIP] {key} in {fpath} is not 2D")
                            return None
                        arrays.append(arr)
                    stacked = np.stack(arrays, axis=0)[:, :IMAGE_HEIGHT, :IMAGE_WIDTH]
                    #print(f"DEBUG: stacked shape after crop: {stacked.shape}")
                    t = torch.tensor(stacked, dtype=torch.float32)

                    # Tripwire: catch non-finite values early + tell us which file/field triggered it
                    if not torch.isfinite(t).all():
                        # Note: use torch.nan_to_num-style stats safely
                        t_min = torch.nanmin(torch.where(torch.isfinite(t), t, torch.tensor(float("nan"), device=t.device))).item()
                        t_max = torch.nanmax(torch.where(torch.isfinite(t), t, torch.tensor(float("nan"), device=t.device))).item()
                        print(f"[BAD DATA] non-finite tensor from {fpath} | min={t_min} max={t_max}")
                        return None

                    #return torch.tensor(stacked, dtype=torch.float32)
                    return t

            except Exception as e:
                print(f"[SKIP] Failed to load {fpath}: {e}")
                return None

        max_retries = 5
        attempt = 0
        failed_paths = set()

        while attempt < max_retries:
            true_idx = self.indices[idx]
            input_files = self.file_list[true_idx : true_idx + self.n_steps]
            target_file = self.file_list[true_idx + self.n_steps]

            input_tensors = []
            for f in input_files:
                fpath = os.path.join(self.root_dir, f)
                t = load_tensor(fpath)
                if t is None:
                    failed_paths.add(fpath)
                    idx = (idx + 1) % len(self)
                    attempt += 1
                    break
                input_tensors.append(t)
            else:
                y_path = os.path.join(self.root_dir, target_file)
                y = load_tensor(y_path)
                if y is None:
                    failed_paths.add(y_path)
                    idx = (idx + 1) % len(self)
                    attempt += 1
                    continue
                bcs = torch.zeros(2)

                return torch.stack(input_tensors, dim=0).float(), bcs.float(), y.float()
                #return torch.stack(input_tensors, dim=0), bcs, y

        print(f"[SKIP] Too many failed attempts at idx={idx}. Failed files:")
        for p in sorted(failed_paths):
            print(f" - {p}")
        raise IndexError(f"Skipping idx={idx} after {max_retries} failures.")


    @staticmethod
    def _specifics():
        time_index = 0
        sample_index = None
        field_names = [
			#"Wvelocity", "av_pressure", "av_temperature", "burn_frac_booster", "burn_frac_maincharge",
			#"density_booster", "energy_booster", "pressure_maincharge",
            "av_density", "Uvelocity", "Wvelocity", "density_maincharge",
        ]
        type = 'cyl_npz'
        split_level = 'sample'
        return time_index, sample_index, field_names, type, split_level

    def _get_specific_stats(self, f=None):
        total_timesteps = len(self.file_list)
        return 1, total_timesteps

    def _get_specific_bcs(self, f=None):
        return [0, 0]  # 0 = non-periodic, 1 = periodic

    def get_per_file_dsets(self):
        # No sub-files — treat whole dataset as one logical file
        return [self]




