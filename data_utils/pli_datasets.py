# File: data_utils/pli_datasets.py
"""LSC rho-to-rho datasets for MPP expt-kyle.

This module keeps the LSC/Loderunner temporal data-loading semantics and adds a
thin MPP adapter so the existing MPP trainer can still consume (x, bcs, y).

expt-kyle convention:
    start_img = image at timestep t-1
    end_img   = image at timestep t
    Dt        = 1 index timestep
    half_image = False by default, i.e. full image via left-right reflection.
"""

import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset
import sys
import random
from pathlib import Path
from typing import Union

'''
IMAGE_WIDTH_HALF_IMAGE = 200
IMAGE_WIDTH_FULL_IMAGE = 400

PATCH_SIZE = 16
IMAGE_HEIGHT_RAW = 1120
IMAGE_HEIGHT_CROPPED = (IMAGE_HEIGHT_RAW // PATCH_SIZE) * PATCH_SIZE
IMAGE_HEIGHT_PADDED  = ((IMAGE_HEIGHT_RAW // PATCH_SIZE) + 1) * PATCH_SIZE
'''

IMAGE_WIDTH = 400
IMAGE_HEIGHT = 1120


class PliNpzDataset(Dataset):
    def __init__(
        self,
        path,
        include_string='',
        n_steps=5,
        dt=1,
        split='train',
        train_val_test=(0.8, 0.1, 0.1),
        subname=None,
        extra_specific=False,
    ):
        self.root_dir = path
        self.n_steps = n_steps
        self.dt = dt
        self.split = split
        self.train_val_test = train_val_test
        self.include_string = include_string
        self.type = 'pli_npz'
        self.field_names = self._specifics()[2]
        self.title = 'pli_npz'

        self.file_list = sorted(
            [
                f for f in os.listdir(self.root_dir)
                if f.endswith(".npz") and include_string in f
            ],
            key=self._extract_timestep,
        )

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
        return "pli_npz"

    def __getitem__(self, idx: int):
        selected_fields = [
            'Uvelocity',
            'Wvelocity',
            'density_case',
            'density_cushion',
            'density_maincharge',
            'density_outside_air',
            'density_striker',
            'density_throw',
        ]

        def load_tensor(fpath):
            try:
                with np.load(fpath) as data:
                    arrays = []
                    for key in selected_fields:
                        if key not in data:
                            print(f"[SKIP] Missing key {key} in {fpath}")
                            raise KeyError(f"Missing field '{key}' in file {fpath}")

                        arr = data[key]
                        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                        arr = arr.astype(np.float32)

                        if arr.ndim != 2:
                            print(f"[SKIP] {key} in {fpath} is not 2D")
                            return None

                        arrays.append(arr)

                    stacked = np.stack(arrays, axis=0)[:, :IMAGE_HEIGHT, :IMAGE_WIDTH]
                    t = torch.tensor(stacked, dtype=torch.float32)

                    if not torch.isfinite(t).all():
                        print(f"[BAD DATA] non-finite tensor from {fpath}")
                        return None

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

        print(f"[SKIP] Too many failed attempts at idx={idx}. Failed files:")
        for p in sorted(failed_paths):
            print(f" - {p}")

        raise IndexError(f"Skipping idx={idx} after {max_retries} failures.")

    @staticmethod
    def _specifics():
        time_index = 0
        sample_index = None
        field_names = [
            'Uvelocity',
            'Wvelocity',
            'density_case',
            'density_cushion',
            'density_maincharge',
            'density_outside_air',
            'density_striker',
            'density_throw',
        ]
        type = 'pli_npz'
        split_level = 'sample'
        return time_index, sample_index, field_names, type, split_level

    def _get_specific_stats(self, f=None):
        total_timesteps = len(self.file_list)
        return 1, total_timesteps

    def _get_specific_bcs(self, f=None):
        return [0, 0]  # 0 = non-periodic, 1 = periodic

    def get_per_file_dsets(self):
        return [self]



def LSCread_npz_NaN(npz: np.lib.npyio.NpzFile, field: str) -> np.ndarray:
    """Extract a field from an NPZ file and replace NaNs/Infs with finite zeros."""
    return np.nan_to_num(npz[field], nan=0.0, posinf=0.0, neginf=0.0)


def volfrac_density(tmp_img: np.ndarray, npz_file: np.lib.npyio.NpzFile, hfield: str) -> np.ndarray:
    """Apply LSC/Loderunner density weighting: density_* -> density_* x vofm_*."""
    prefix = "density_"
    if not hfield.startswith(prefix):
        return tmp_img

    suffix = hfield[len(prefix):]
    if not suffix:
        print(f"[pli_datasets.py] Could not extract suffix from hfield: {hfield}", file=sys.stderr)
        return tmp_img

    vofm_hfield = "vofm_" + suffix
    if vofm_hfield not in npz_file:
        raise KeyError(f"Missing required volume-fraction field '{vofm_hfield}' for '{hfield}'")

    vofm = LSCread_npz_NaN(npz_file, vofm_hfield)
    return tmp_img * vofm


class LSC_rho2rho_temporal_DataSet(Dataset):
    """LSC/Loderunner-style temporal dataset returning (start_img, end_img, Dt).

    This follows the LSC channel/preprocessing semantics:
      - density channels first, velocity channels last
      - density_* channels are multiplied by matching vofm_* channels
      - half_image=False reflects the image with np.concatenate((fliplr(img), img), axis=1)

    For expt-kyle, set timeIDX_offset=1 and force_positive_offset=True.
    """

    def __init__(
        self,
        LSC_NPZ_DIR: str,
        file_prefix_list: Union[str, None] = None,
        max_timeIDX_offset: int = 1,
        max_file_checks: int = 100,
        half_image: bool = False,
        hydro_fields: np.ndarray = np.array([
            "density_case",
            "density_cushion",
            "density_maincharge",
            "density_outside_air",
            "density_striker",
            "density_throw",
            "Uvelocity",
            "Wvelocity",
        ]),
        include_string: str = "",
        split: str = "train",
        train_val_test=(0.8, 0.1, 0.1),
        timeIDX_offset: int = 1,
        force_positive_offset: bool = True,
        random_pair_sampling: bool = False,
    ) -> None:
        self.LSC_NPZ_DIR = LSC_NPZ_DIR if LSC_NPZ_DIR.endswith(os.sep) else LSC_NPZ_DIR + os.sep
        self.max_timeIDX_offset = max_timeIDX_offset
        self.max_file_checks = max_file_checks
        self.half_image = half_image
        self.hydro_fields = np.array(hydro_fields)
        self.include_string = include_string
        self.split = split
        self.train_val_test = train_val_test
        self.timeIDX_offset = int(timeIDX_offset)
        self.force_positive_offset = force_positive_offset
        self.random_pair_sampling = random_pair_sampling
        self.rng = np.random.default_rng()

        if file_prefix_list is not None and os.path.isfile(file_prefix_list):
            with open(file_prefix_list) as f:
                prefixes = [line.rstrip() for line in f if line.rstrip()]
        else:
            prefixes = self._discover_prefixes(self.LSC_NPZ_DIR, include_string)
        random.shuffle(prefixes)
        self.file_prefix_list = prefixes
        self.Nsamples = len(self.file_prefix_list)

        self.valid_pairs = self._build_valid_pairs()
        if len(self.valid_pairs) == 0:
            raise ValueError(
                f"No valid LSC rho2rho pairs found in {self.LSC_NPZ_DIR} "
                f"with include_string='{include_string}' and timeIDX_offset={self.timeIDX_offset}"
            )
        self.valid_pairs = self._split_pairs(self.valid_pairs, split, train_val_test)

    @staticmethod
    def _discover_prefixes(root_dir: str, include_string: str = ""):
        prefixes = set()
        pattern = re.compile(r"(?P<prefix>.+)_pvi_idx(?P<idx>\d+)\.npz$")
        for fname in os.listdir(root_dir):
            if not fname.endswith(".npz") or include_string not in fname:
                continue
            match = pattern.match(fname)
            if match:
                prefixes.add(match.group("prefix"))
        return sorted(prefixes)

    @staticmethod
    def _extract_idx(fname: str) -> int:
        match = re.search(r"_pvi_idx(?P<idx>\d+)\.npz$", fname)
        if match is None:
            raise ValueError(f"Cannot extract LSC time index from filename: {fname}")
        return int(match.group("idx"))

    def _build_valid_pairs(self):
        valid_pairs = []
        for prefix in self.file_prefix_list:
            files = sorted(Path(self.LSC_NPZ_DIR).glob(f"{prefix}_pvi_idx*.npz"))
            idx_to_file = {self._extract_idx(f.name): f.name for f in files}
            for start_idx in sorted(idx_to_file):
                end_idx = start_idx + self.timeIDX_offset
                if end_idx in idx_to_file:
                    valid_pairs.append((idx_to_file[start_idx], idx_to_file[end_idx], self.timeIDX_offset))
        return valid_pairs

    @staticmethod
    def _split_pairs(pairs, split, train_val_test):
        n = len(pairs)
        train_end = int(train_val_test[0] * n)
        val_end = train_end + int(train_val_test[1] * n)
        if split == "train":
            return pairs[:train_end]
        if split == "val":
            return pairs[train_end:val_end]
        return pairs[val_end:]

    def __len__(self) -> int:
        return len(self.valid_pairs)

    def _load_image(self, filename: str) -> torch.Tensor:
        npz_path = self.LSC_NPZ_DIR + filename
        with np.load(npz_path) as data_npz:
            img_list = []
            for hfield in self.hydro_fields:
                tmp_img = LSCread_npz_NaN(data_npz, str(hfield))
                tmp_img = volfrac_density(tmp_img, data_npz, str(hfield))
                if not self.half_image:
                    tmp_img = np.concatenate((np.fliplr(tmp_img), tmp_img), axis=1)
                img_list.append(tmp_img.astype(np.float32))
        return torch.tensor(np.stack(img_list, axis=0), dtype=torch.float32)

    def __getitem__(self, index: int):
        index = index % len(self.valid_pairs)
        start_file, end_file, dt_idx = self.valid_pairs[index]
        start_img = self._load_image(start_file)
        end_img = self._load_image(end_file)

        # expt-kyle wants Dt=1 timestep. Keep this in index-timestep units.
        Dt = torch.tensor(float(dt_idx), dtype=torch.float32)
        return start_img, end_img, Dt

class LscRho2RhoNpzDatasetMPPWrapper(Dataset):
    """MPP adapter for LSC_rho2rho_temporal_DataSet.

    MPP MixedDataset expects each sub-dataset to return (x, bcs, y), where x has
    shape [T, C, H, W]. The raw LSC dataset returns (start_img, end_img, Dt), so
    this adapter performs:
        x   = start_img[None, ...]
        bcs = zeros(2)
        y   = end_img

    For this dataset type, the MPP "include_string" slot is intentionally reused
    as the LSC file_prefix_list path.
    """

    def __init__(
        self,
        path,
        include_string='',
        n_steps=1,
        dt=1,
        split='train',
        train_val_test=(0.8, 0.1, 0.1),
        subname=None,
        extra_specific=False,
    ):
        self.root_dir = path if path.endswith("/") else path + "/"
        self.n_steps = 1
        self.dt = 1
        self.split = split
        self.train_val_test = train_val_test
        self.file_prefix_list = include_string
        self.type = 'lsc_npz_rho2rho'
        self.title = 'lsc_npz_rho2rho'
        self.field_names = self._specifics()[2]

        if not self.file_prefix_list:
            raise ValueError(
                "lsc_npz_rho2rho requires the third train_data_paths field "
                "to be a file-prefix-list path, but it was empty."
            )

        if not os.path.isfile(self.file_prefix_list):
            raise FileNotFoundError(
                f"LSC file_prefix_list not found: {self.file_prefix_list}"
            )

        self.lsc_dataset = LSC_rho2rho_temporal_DataSet(
            LSC_NPZ_DIR=self.root_dir,
            file_prefix_list=self.file_prefix_list,
            max_timeIDX_offset=1,
            max_file_checks=100,
            half_image=False,
            hydro_fields=np.array(self.field_names),
            split=split,
            train_val_test=train_val_test,
            timeIDX_offset=1,
            force_positive_offset=True,
            random_pair_sampling=False,
        )

    def __len__(self):
        return len(self.lsc_dataset)

    def get_name(self, full_name=False):
        return self.type

    def get_per_file_dsets(self):
        return [self]

    def __getitem__(self, idx: int):
        start_img, end_img, Dt = self.lsc_dataset[idx]

        x = start_img.unsqueeze(0).float()  # [1, C, H, W]
        y = end_img.float()                 # [C, H, W]
        bcs = torch.zeros(2, dtype=torch.float32)

        return x, bcs, y

    @staticmethod
    def _specifics():
        time_index = 0
        sample_index = None
        field_names = [
            "density_case",
            "density_cushion",
            "density_maincharge",
            "density_outside_air",
            "density_striker",
            "density_throw",
            "Uvelocity",
            "Wvelocity",
        ]
        type = "lsc_npz_rho2rho"
        split_level = "sample"
        return time_index, sample_index, field_names, type, split_level

    def _get_specific_stats(self, f=None):
        return 1, len(self.lsc_dataset)

    def _get_specific_bcs(self, f=None):
        return [0, 0]

