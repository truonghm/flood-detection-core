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


class PatchesExtractor:
    def __init__(
        self,
        pre_flood_dir: Path,
        pre_flood_format: Literal["geotiff", "numpy"],
        output_dir: Path,
        num_patches: int = 100,
        num_temporal_length: int = 4,
        patch_size: int = 16,
        patch_stride: int = 16,
        replacement: bool = False,
    ) -> None:
        self.pre_flood_dir = pre_flood_dir
        self.num_patches = num_patches
        self.num_temporal_length = num_temporal_length
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.replacement = replacement
        self.cache_key = get_patches_cache_key(
            self.num_patches, self.num_temporal_length, self.patch_size, self.patch_stride
        )
        self.cache_dir = output_dir / self.cache_key

        if pre_flood_format == "numpy":
            self.extension = "npy"
        elif pre_flood_format == "geotiff":
            self.extension = "tif"
        else:
            raise ValueError(f"Invalid output format: {pre_flood_format}")

    @property
    def all_tile_dirs(self) -> list[Path]:
        return list(self.pre_flood_dir.glob("*/*/"))

    @property
    def random_tile_dirs(self) -> list[Path]:
        if self.replacement:
            return random.choices(self.all_tile_dirs, k=self.num_patches)
        else:
            return random.sample(self.all_tile_dirs, k=self.num_patches)

    def extract_one_sequence(self, tile_dir: Path) -> list[tuple[np.ndarray, datetime.datetime]] | None:
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
        for i, (data, date) in enumerate(zip(images_data, chosen_dates)):
            patch = data[start_y : start_y + patch_size, start_x : start_x + patch_size, :]
            patch_sequence.append((patch, date))

        return patch_sequence

    def extract(self, idx: int) -> list[tuple[np.ndarray, datetime.datetime]]:
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
                    patch_sequence.append((patch, datetime.datetime.strptime(patch_path.stem, "%Y-%m-%d")))
            else:
                # incomplete cache, remove and regenerate
                import shutil
                shutil.rmtree(patch_cache_dir)

        if not patch_sequence:  # if cache didn't exist or was incomplete
            tile_dir = random.choice(self.random_tile_dirs)
            while True:
                try:
                    patch_sequence = self.extract_one_sequence(tile_dir)
                    break
                except (NotEnoughImagesError, TileTooSmallError):
                    tile_dir = random.choice(self.random_tile_dirs)
                    continue

        # only cache if we generated new data (not loaded from cache)
        if not (self.cache_dir / f"{idx}").exists():
            self.cache(patch_sequence, idx)
        return patch_sequence

    def cache(self, patch_sequence: list[tuple[np.ndarray, datetime.datetime]], idx: int) -> None:
        for i, (patch, date) in enumerate(patch_sequence):
            date_str = date.strftime("%Y-%m-%d")
            patch_dir = self.cache_dir / f"{idx}"
            patch_path = patch_dir / f"{date_str}.npy"
            patch_dir.mkdir(parents=True, exist_ok=True)
            np.save(patch_path, patch)

    def __call__(self, idx: int) -> list[tuple[np.ndarray, datetime.datetime]]:
        return self.extract(idx)
