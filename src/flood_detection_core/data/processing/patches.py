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

import datetime
import random
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
    if idx > len(list(tile_dir.glob(f"*.{extension}"))):
        raise ValueError(f"idx {idx} is out of range")
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
        input_dir: Path,
        input_format: Literal["geotiff", "numpy"],
        output_dir: Path,
        num_patches: int = 100,
        num_temporal_length: int = 4,
        patch_size: int = 16,
        patch_stride: int = 16,
        replacement: bool = False,
        stratified: bool = True,
    ) -> None:
        self.input_dir = input_dir
        self.num_patches = num_patches
        self.num_temporal_length = num_temporal_length
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.replacement = replacement
        self.stratified = stratified
        self.cache_key = get_patches_cache_key(
            self.num_patches, self.num_temporal_length, self.patch_size, self.patch_stride
        )
        self.cache_dir = output_dir / self.cache_key
        self.selected_tile_dirs = (
            self.get_site_stratified_tile_dirs() if self.stratified else self.get_random_tile_dirs()
        )
        if input_format == "numpy":
            self.extension = "npy"
        elif input_format == "geotiff":
            self.extension = "tif"
        else:
            raise ValueError(f"Invalid output format: {input_format}")

    @property
    def all_tile_dirs(self) -> list[Path]:
        return list(self.input_dir.glob("*/*/"))

    def get_random_tile_dirs(self) -> list[Path]:
        if self.replacement:
            return random.choices(self.all_tile_dirs, k=self.num_patches)
        else:
            return random.sample(self.all_tile_dirs, k=self.num_patches)

    def get_site_stratified_tile_dirs(self) -> list[Path]:
        """Sample tiles with equal probability per site"""
        sites = {}
        for tile_dir in self.all_tile_dirs:
            site_name = tile_dir.parent.name
            if site_name not in sites:
                sites[site_name] = []
            sites[site_name].append(tile_dir)

        patches_per_site = self.num_patches // len(sites)
        remainder = self.num_patches % len(sites)

        selected_tiles = []
        for i, (site_name, site_tiles) in enumerate(sites.items()):
            n_from_site = patches_per_site + (1 if i < remainder else 0)
            if self.replacement:
                selected_tiles.extend(random.choices(site_tiles, k=n_from_site))
            else:
                selected_tiles.extend(random.sample(site_tiles, k=min(n_from_site, len(site_tiles))))

        return selected_tiles

    def extract_one_sequence(self, tile_dir: Path) -> list[np.ndarray] | None:
        ts_length = len(list(tile_dir.glob(f"*.{self.extension}")))
        n_ts_length = self.num_temporal_length
        if ts_length < n_ts_length:
            raise NotEnoughImagesError(
                f"Tile {tile_dir.name} has fewer than {n_ts_length} images ({ts_length} < {n_ts_length})"
            )

        # randomly pick 4 CONSECUTIVE images
        ts_start_idx = random.randint(1, ts_length - n_ts_length)
        ts_end_idx = ts_start_idx + n_ts_length - 1

        chosen_tile_paths = []
        chosen_dates = []
        for i in range(ts_start_idx, ts_end_idx + 1):
            img_path, img_date = get_img_path(tile_dir, i)
            chosen_tile_paths.append(img_path)
            chosen_dates.append(img_date)

        # sort by date
        chosen_tile_paths, chosen_dates = zip(*sorted(zip(chosen_tile_paths, chosen_dates), key=lambda x: x[1]))

        images_data = []
        min_height, min_width = float("inf"), float("inf")
        for img_path in chosen_tile_paths:
            if self.extension == "tif":
                with rasterio.open(img_path) as src:
                    data = src.read()
                    data = np.transpose(data, (1, 2, 0))
            else:
                data = np.load(img_path)

            images_data.append(data)
            height, width = data.shape[:2]
            min_height = min(min_height, height)
            min_width = min(min_width, width)

        patch_size = self.patch_size
        if min_height < patch_size or min_width < patch_size:
            raise TileTooSmallError(f"Tile {tile_dir.name} is too small ({min_height}x{min_width})")

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
            tile_dir = random.choice(self.selected_tile_dirs)
            while True:
                try:
                    patch_sequence = self.extract_one_sequence(tile_dir)
                    break
                except (NotEnoughImagesError, TileTooSmallError):
                    tile_dir = random.choice(self.selected_tile_dirs)
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
