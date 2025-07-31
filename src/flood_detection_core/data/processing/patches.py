"""
TODO: Rework this module completely

- Separate into: Pretrain Sampler and an actual PatchesExtractor that focuses on only extracting patches

- Pretrain Sampler:
    - Maybe merge into the Pretrain dataset? Focus on sampling sites/tiles and image sequences (4 from 8).
    - Output is a list of paths to 4-image sequences.

- Patches Extractor:
    - Given a list of image paths, extract patches from them
    - Focus on the extract_one_sequence method, but the date picking logic should be moved to the Pretrain Sampler
    - Can be used for both pretraining and normal training (FloodEventDataset).
"""

import csv
import datetime
import random
import re
from pathlib import Path
from typing import Literal

import numpy as np
import rasterio


class NotEnoughImagesError(Exception):
    pass


class TileTooSmallError(Exception):
    pass


def get_img_path(tile_dir: Path, idx: int, extension: str = "tif") -> tuple[Path, datetime.datetime]:
    if idx == 0:
        raise ValueError("idx starts with 1")
    paths = list(tile_dir.glob(f"pre_flood_{idx}_*.{extension}"))
    if len(paths) != 1:
        raise ValueError(f"Expected 1 path, got {len(paths)}")
    path = paths[0]
    img_date = datetime.datetime.strptime(path.stem.split("_")[-1].replace(".tif", ""), "%Y-%m-%d")
    return path, img_date


def get_patches_cache_key(num_patches: int, num_temporal_length: int, patch_size: int, patch_stride: int) -> str:
    return f"cache_{num_patches}_{num_temporal_length}_{patch_size}_{patch_stride}"


class PreTrainPatchesExtractor:
    def __init__(
        self,
        split_csv_path: Path,
        output_dir: Path,
        num_patches: int = 100,
        num_temporal_length: int = 4,
        patch_size: int = 16,
        patch_stride: int = 16,
    ) -> None:
        self.split_csv_path = split_csv_path
        self.num_patches = num_patches
        self.num_temporal_length = num_temporal_length
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.cache_key = get_patches_cache_key(
            self.num_patches, self.num_temporal_length, self.patch_size, self.patch_stride
        )
        self.chosen_tile_paths = self.get_random_tile_paths(self.load_pre_flood_split_csv(self.split_csv_path))
        self.cache_dir = output_dir / self.cache_key

    def load_pre_flood_split_csv(self, split_csv_path: Path) -> dict[str, list[Path]]:
        data = {}
        with open(split_csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["dataset_type"] == "pretrain":
                    # data.append(row)
                    if row["tile"] not in data:
                        data[row["tile"]] = []
                    data[row["tile"]].append(row["path"])
        return data

    def get_random_tile_paths(self, tile_paths_mapping: dict[str, list[Path]]) -> dict[str, list[Path]]:
        chosen_tiles = random.choices(list(tile_paths_mapping.keys()), k=self.num_patches)
        chosen_tile_paths = {tile: tile_paths_mapping[tile] for tile in chosen_tiles}
        return chosen_tile_paths

    def extract_one_sequence(self, tile_name: str, paths: list[Path]) -> list[np.ndarray] | None:
        ts_length = len(paths)
        n_ts_length = self.num_temporal_length
        path_indices = [int(re.search(r"pre_flood_(\d+)_", path.name).group(1)) for path in paths]

        # sort paths based on indices
        sorted_paths = [path for _, path in sorted(zip(path_indices, paths))]

        if ts_length < n_ts_length:
            raise NotEnoughImagesError(
                f"Tile {tile_name} has fewer than {n_ts_length} images ({ts_length} < {n_ts_length})"
            )

        # randomly pick n_ts_length CONSECUTIVE images
        ts_start_idx = random.randint(1, ts_length - n_ts_length)
        ts_end_idx = ts_start_idx + n_ts_length - 1

        chosen_tile_paths = sorted_paths[ts_start_idx : ts_end_idx]

        images_data = []
        min_height, min_width = float("inf"), float("inf")
        for img_path in chosen_tile_paths:
            if img_path.suffix == ".tif":
                with rasterio.open(img_path) as src:
                    data = src.read()
                    data = np.transpose(data, (1, 2, 0))
            elif img_path.suffix == ".npy":
                data = np.load(img_path)
            else:
                raise ValueError(f"Invalid extension: {img_path.suffix}")

            images_data.append(data)
            height, width = data.shape[:2]
            min_height = min(min_height, height)
            min_width = min(min_width, width)

        patch_size = self.patch_size
        if min_height < patch_size or min_width < patch_size:
            raise TileTooSmallError(f"Tile {tile_name} is too small ({min_height}x{min_width})")

        start_y = random.randint(0, min_height - patch_size)
        start_x = random.randint(0, min_width - patch_size)

        patch_sequence = []
        for i, data in enumerate(images_data):
            patch = data[start_y : start_y + patch_size, start_x : start_x + patch_size, :]
            patch_sequence.append(patch)

        return patch_sequence

    def extract(self, idx: int) -> list[np.ndarray]:
        if idx < 0 or idx >= self.num_patches:
            raise ValueError(f"idx {idx} is out of range [0, {self.num_patches})")

        # check if cache exists and is complete
        patch_cache_dir = self.cache_dir / f"{idx}"
        patch_sequence = []
        if patch_cache_dir.exists():
            cached_files = list(patch_cache_dir.glob("*.npy"))
            if len(cached_files) == self.num_temporal_length:
                for patch_path in sorted(cached_files):  # sort to ensure consistent order
                    patch = np.load(patch_path)
                    patch_sequence.append(patch)
            else:
                # incomplete cache, remove and regenerate
                import shutil

                shutil.rmtree(patch_cache_dir)

        if not patch_sequence:  # if cache didn't exist or was incomplete
            tile_name = random.choice(list(self.chosen_tile_paths.keys()))
            while True:
                try:
                    patch_sequence = self.extract_one_sequence(tile_name, self.chosen_tile_paths[tile_name])
                    break
                except (NotEnoughImagesError, TileTooSmallError):
                    tile_name = random.choice(list(self.chosen_tile_paths.keys()))
                    continue

        # only cache if we generated new data (not loaded from cache)
        if not (self.cache_dir / f"{idx}").exists():
            self.cache(patch_sequence, idx)
        return patch_sequence

    def cache(self, patch_sequence: list[np.ndarray], idx: int) -> None:
        for i, patch in enumerate(patch_sequence):
            patch_dir = self.cache_dir / f"{idx}"
            patch_path = patch_dir / f"{i}.npy"
            patch_dir.mkdir(parents=True, exist_ok=True)
            np.save(patch_path, patch)

    def __call__(self, idx: int) -> list[np.ndarray]:
        return self.extract(idx)


def extract_flood_event_patches(
    tile_paths: list[Path],
    dates: list[datetime.datetime] | None = None,
    format: Literal["geotiff", "numpy"] = "geotiff",
    patch_size: int = 16,
    patch_stride: int = 1,
) -> list[list[np.ndarray]]:
    """
    Extract a sequence of patches from a list of tile paths and dates.
    Dates are optional and will be used for sorting the patches.

    Parameters
    ----------
    tile_paths : list[Path]
        List of tile paths to extract patches from.
    dates : list[datetime.datetime] | None, optional
        List of dates to sort the patches by. If None, the patches will keep the order of the tile paths.

    Returns
    -------
    list[list[np.ndarray]]
        List of patch sequences, where each sequence contains patches from all time steps at the same spatial location.
    """

    if dates:
        # sort tile paths by dates
        tile_paths, _ = zip(*sorted(zip(tile_paths, dates), key=lambda x: x[1]))

    images_data = []
    min_height, min_width = float("inf"), float("inf")
    for img_path in tile_paths:
        if format == "geotiff":
            with rasterio.open(img_path) as src:
                data = src.read()
                data = np.transpose(data, (1, 2, 0))
        elif format == "numpy":
            data = np.load(img_path)
        else:
            raise ValueError(f"Invalid format: {format}")

        images_data.append(data)
        height, width = data.shape[:2]
        min_height = min(min_height, height)
        min_width = min(min_width, width)

    if min_height < patch_size or min_width < patch_size:
        raise TileTooSmallError(f"Tile {tile_paths[0].parent.name} is too small ({min_height}x{min_width})")

    num_patches_x = (min_width - patch_size) // patch_stride + 1
    num_patches_y = (min_height - patch_size) // patch_stride + 1

    all_patch_sequences = []

    for y_idx in range(num_patches_y):
        for x_idx in range(num_patches_x):
            start_y = y_idx * patch_stride
            start_x = x_idx * patch_stride
            patch_sequence = []
            for data in images_data:
                patch = data[start_y : start_y + patch_size, start_x : start_x + patch_size, :]
                patch_sequence.append(patch)
            all_patch_sequences.append(patch_sequence)

    return all_patch_sequences
