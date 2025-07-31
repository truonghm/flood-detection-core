import csv
import datetime
import random
import re
from pathlib import Path
from typing import Callable

import numpy as np
import rasterio
from rich import print
from rich.progress import Progress

from flood_detection_core.data.processing.utils import choose_pre_flood_paths, get_image_size
from flood_detection_core.exceptions import NotEnoughImagesError, TileTooSmallError
from flood_detection_core.data.processing.augmentation import augment_data


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
        chosen_tile_paths = choose_pre_flood_paths(paths, self.num_temporal_length)

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


def create_patch_metadata(
    tile_pairs: list[dict], num_temporal_length: int, patch_size: int, patch_stride: int
) -> list[dict]:
    total_patches = 0
    tile_patch_counts = []

    for tile_pair in tile_pairs:
        try:
            chosen_pre_flood_paths = choose_pre_flood_paths(
                tile_pair["pre_flood_paths"], num_temporal_length=num_temporal_length
            )
            all_image_paths = chosen_pre_flood_paths + [tile_pair["post_flood_path"]]

            min_height, min_width = float("inf"), float("inf")

            for img_path in all_image_paths:
                img_height, img_width = get_image_size(img_path)
                min_height = min(min_height, img_height)
                min_width = min(min_width, img_width)

            if min_height < patch_size or min_width < patch_size:
                continue

            max_i = min_height - patch_size + 1
            max_j = min_width - patch_size + 1
            patch_count = len(range(0, max_i, patch_stride)) * len(range(0, max_j, patch_stride))
            tile_patch_counts.append(
                (
                    {
                        "tile_id": tile_pair["tile_id"],
                        "site": tile_pair["site"],
                        "pre_flood_paths": chosen_pre_flood_paths,
                        "post_flood_path": tile_pair["post_flood_path"],
                        "ground_truth_path": tile_pair["ground_truth_path"],
                    },
                    min_height,
                    min_width,
                    max_i,
                    max_j,
                    patch_count,
                )
            )
            total_patches += patch_count
        except Exception:
            tile_patch_counts.append(None)

    patch_metadata = [None] * total_patches

    idx = 0
    for tile_data in tile_patch_counts:
        if tile_data is None:
            continue

        tile_pair, min_height, min_width, max_i, max_j, patch_count = tile_data

        for i in range(0, max_i, patch_stride):
            for j in range(0, max_j, patch_stride):
                patch_metadata[idx] = {
                    "tile_pair": tile_pair,
                    "patch_coords": (i, j),
                    "img_dims": (min_height, min_width),
                }
                idx += 1

            # for i in range(0, max_i, self.patch_stride):
            #     for j in range(0, max_j, self.patch_stride):
            #         patch_metadata.append(
            #             {"tile_pair": tile_pair, "patch_coords": (i, j), "img_dims": (min_height, min_width)}
            #         )

    return patch_metadata


def extract_patches_at_coords(
    image_paths: list[Path | str],
    patch_coords: tuple[int, int],
    patch_size: int,
    vv_clipped_range: tuple[float, float] | None = None,
    vh_clipped_range: tuple[float, float] | None = None,
) -> list[np.ndarray]:
    patches = []
    i, j = patch_coords

    for img_path in image_paths:
        if isinstance(img_path, str):
            img_path = Path(img_path)

        if img_path.suffix == ".tif":
            with rasterio.open(img_path) as src:
                data = src.read()
                data = np.transpose(data, (1, 2, 0))
        elif img_path.suffix == ".npy":
            data = np.load(img_path)
        else:
            raise ValueError(f"Invalid format: {img_path}")

        if vv_clipped_range is not None:
            # handle nan values
            vv_band = data[:, :, 0].copy()
            vv_band = np.where(np.isnan(vv_band), vv_clipped_range[0], vv_band)
            data[:, :, 0] = np.clip(
                (vv_band - vv_clipped_range[0]) / (vv_clipped_range[1] - vv_clipped_range[0]),
                0,
                1,
            )
        if vh_clipped_range is not None:
            vh_band = data[:, :, 1].copy()
            vh_band = np.where(np.isnan(vh_band), vh_clipped_range[0], vh_band)
            data[:, :, 1] = np.clip(
                (vh_band - vh_clipped_range[0]) / (vh_clipped_range[1] - vh_clipped_range[0]),
                0,
                1,
            )

        patch = data[i : i + patch_size, j : j + patch_size, :]
        patches.append(patch)

    return patches


def extract_patches_to_binary(
    patch_metadata: list[dict],
    patch_size: int,
    output_path: Path,
    num_temporal_length: int,
    vv_clipped_range: tuple[float, float] | None = None,
    vh_clipped_range: tuple[float, float] | None = None,
    transform: Callable | None = None,
    batch_size: int = 1000,
) -> None:
    num_samples = len(patch_metadata)

    # Create memory-mapped array with fixed shape
    # Shape: (num_samples, max_temporal_length + 1, patch_size, patch_size, 2)
    # Last index is for post-flood image
    memmap_array = np.memmap(
        output_path,
        dtype=np.float32,
        mode="w+",
        shape=(num_samples, num_temporal_length + 1, patch_size, patch_size, 2),
    )

    print(f"Created memory-mapped file: {output_path}")
    print(f"Shape: {memmap_array.shape}")
    print(f"Size: {memmap_array.nbytes / (1024**3):.2f} GB")

    with Progress() as progress:
        # Calculate total number of batches correctly (using ceiling division)
        total_batches = (num_samples + batch_size - 1) // batch_size
        batch_task = progress.add_task("Processing batches", total=total_batches)
        sample_task = progress.add_task("Extracting patches per batch", total=batch_size)

        # Process in batches to manage memory
        for batch_start in range(0, num_samples, batch_size):
            batch_end = min(batch_start + batch_size, num_samples)

            print(
                f"Processing batch {batch_start // batch_size + 1}/{total_batches}: "
                f"samples {batch_start} to {batch_end - 1}"
            )

            progress.update(sample_task, completed=0)
            for idx in range(batch_start, batch_end):
                patch_meta = patch_metadata[idx]
                tile_pair = patch_meta["tile_pair"]
                patch_coords = patch_meta["patch_coords"]

                # Extract pre-flood patches
                pre_flood_patches = extract_patches_at_coords(
                    tile_pair["pre_flood_paths"],
                    patch_coords,
                    patch_size=patch_size,
                    vv_clipped_range=vv_clipped_range,
                    vh_clipped_range=vh_clipped_range,
                )

                # Extract post-flood patch
                post_flood_patches = extract_patches_at_coords(
                    [tile_pair["post_flood_path"]],
                    patch_coords,
                    patch_size=patch_size,
                    vv_clipped_range=vv_clipped_range,
                    vh_clipped_range=vh_clipped_range,
                )

                # Make arrays contiguous
                pre_flood_patches = [np.ascontiguousarray(p) for p in pre_flood_patches]
                post_flood_patches = [np.ascontiguousarray(p) for p in post_flood_patches]

                # Pre-allocate arrays for this sample
                num_pre_flood = len(pre_flood_patches)

                # Fill pre-flood patches (pad with zeros if fewer than max_temporal_length)
                for i, patch in enumerate(pre_flood_patches):
                    memmap_array[idx, i] = patch

                # Fill remaining pre-flood slots with zeros if needed
                for i in range(num_pre_flood, num_temporal_length):
                    memmap_array[idx, i] = 0

                # Fill post-flood patch (always at the last index)
                memmap_array[idx, num_temporal_length] = post_flood_patches[0]

                # Apply transform if provided
                if transform:
                    # Transform operates on the entire sample
                    sample = memmap_array[idx]  # Shape: (max_temporal_length + 1, patch_size, patch_size, 2)
                    transformed_sample = transform(sample)
                    memmap_array[idx] = transformed_sample
                progress.update(sample_task, advance=1)

            # Flush batch to disk by deleting and recreating memmap
            # print("Flushing batch to disk...")
            del memmap_array
            memmap_array = np.memmap(
                output_path,
                dtype=np.float32,
                mode="r+",
                shape=(num_samples, num_temporal_length + 1, patch_size, patch_size, 2),
            )
            progress.update(batch_task, advance=1)
    # Final cleanup
    del memmap_array
    print(f"Successfully saved {num_samples} samples to {output_path}")


if __name__ == "__main__":
    from flood_detection_core.config import CLVAEConfig, DataConfig
    from flood_detection_core.data.processing.split import get_flood_event_tile_pairs

    data_config = DataConfig.from_yaml(Path("yamls/data.yaml"))
    clvae_config = CLVAEConfig.from_yaml(Path("yamls/model_clvae.yaml"))

    patch_size = clvae_config.site_specific.patch_size
    patch_stride = clvae_config.site_specific.patch_stride

    tile_pairs = get_flood_event_tile_pairs(
        dataset_type="train",
        pre_flood_split_csv_path=data_config.splits.pre_flood_split,
        post_flood_split_csv_path=data_config.splits.post_flood_split,
    )
    patch_metadata = create_patch_metadata(
        tile_pairs=tile_pairs,
        num_temporal_length=clvae_config.site_specific.num_temporal_length,
        patch_size=patch_size,
        patch_stride=patch_stride,
    )
    extract_patches_to_binary(
        patch_metadata=patch_metadata,
        patch_size=patch_size,
        output_path=f"data/train_patches_{patch_size}_{patch_stride}.dat",
        num_temporal_length=clvae_config.site_specific.num_temporal_length,
        vv_clipped_range=clvae_config.site_specific.vv_clipped_range,
        vh_clipped_range=clvae_config.site_specific.vh_clipped_range,
        transform=lambda x: augment_data(x, clvae_config.augmentation, False),
        batch_size=10000,
    )
