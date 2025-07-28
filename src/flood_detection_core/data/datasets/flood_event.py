import datetime
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

from flood_detection_core.data.constants import HandLabeledSen1Flood11Sites
from flood_detection_core.data.loaders.tif import get_image_size

FloodPathsType = dict[str, dict[str, list[Path]]]


def get_image_size_v2(file_path: Path) -> tuple[int, int]:
    with rasterio.open(file_path) as src:
        return src.height, src.width


class FloodEventDataset(Dataset):
    """
    **Purpose**: Handles pre/post flood image pairs for site-specific training and inference
    **Logic**:
    - Loads paired pre-flood (configurable length) and post-flood (1 image) data
    - Extracts 16×16×2 patches with stride 1 for inference
    - For training: creates positive/negative pairs for contrastive learning
    - Implements same augmentation as PreFloodDataset

    **Output format**:
    - Pre-flood sequence: (T, C, H, W) where T=num_temporal_length, C=2, H=W=16
    - Post-flood image: (1, C, H, W) where C=2, H=W=16
    - After batching: (B, T, C, H, W) and (B, 1, C, H, W)

    IMPORTANT: images can have different sizes (not fixed at 512x512)
    """

    def __init__(
        self,
        pre_flood_dir: Path,
        pre_flood_format: Literal["numpy", "geotiff"],
        post_flood_dir: Path,
        post_flood_format: Literal["numpy", "geotiff"],
        sites: list[str] = HandLabeledSen1Flood11Sites,
        num_temporal_length: int = 4,
        patch_size: int = 16,
        patch_stride: int = 1,
        transform: Callable | None = None,
    ) -> None:
        self.pre_flood_dir = pre_flood_dir
        self.pre_flood_format = pre_flood_format
        self.post_flood_dir = post_flood_dir
        self.post_flood_format = post_flood_format
        self.num_temporal_length = num_temporal_length
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.transform = transform
        self.sites = sites

        self.pre_flood_paths = self._get_pre_flood_paths()
        self.post_flood_paths = self._get_post_flood_paths(self.pre_flood_paths)
        self.tile_pairs = self._create_tile_pairs(self.pre_flood_paths, self.post_flood_paths)
        self.patch_metadata = self._create_patch_metadata()

    def _get_pre_flood_paths(self) -> FloodPathsType:
        site_path_mapping = {}
        for site in self.sites:
            tile_path_mapping = {}
            for tile_dir in self.pre_flood_dir.glob(f"{site}/*/"):
                tile_paths = []
                tile_dates = []
                for tile_path in tile_dir.glob("*.tif"):
                    img_date = datetime.datetime.strptime(tile_path.stem.split("_")[-1].replace(".tif", ""), "%Y-%m-%d")
                    tile_paths.append(tile_path)
                    tile_dates.append(img_date)

                tile_paths, tile_dates = zip(*sorted(zip(tile_paths, tile_dates), key=lambda x: x[1]))
                tile_paths = list(tile_paths)[: self.num_temporal_length]
                tile_path_mapping[tile_dir.name] = tile_paths

            site_path_mapping[site] = tile_path_mapping

        return site_path_mapping

    def _get_post_flood_paths(self, pre_flood_paths: FloodPathsType) -> FloodPathsType:
        site_path_mapping = {}
        for site, tile_path_mapping in pre_flood_paths.items():
            site_path_mapping[site] = {}
            for tile_id in tile_path_mapping.keys():
                post_flood_path = list(self.post_flood_dir.glob(f"{tile_id}_*.tif"))
                site_path_mapping[site][tile_id] = post_flood_path

        return site_path_mapping

    def _extract_tile_id(self, path: Path) -> str:
        stem = path.stem

        if "_S1Hand" in stem or "_S1Weak" in stem:
            return "_".join(stem.split("_")[:2])
        elif "pre_flood_" in stem:
            return path.parent.name
        else:
            parts = stem.split("_")
            if len(parts) >= 2:
                return "_".join(parts[:2])
            else:
                return stem

    def _create_tile_pairs(self, pre_flood_paths: FloodPathsType, post_flood_paths: FloodPathsType) -> list[dict]:
        tile_pairs = []

        for site, tile_path_mapping in pre_flood_paths.items():
            for tile_id, tile_paths in tile_path_mapping.items():
                post_flood_path = post_flood_paths[site][tile_id]
                tile_pairs.append(
                    {
                        "tile_id": tile_id,
                        "pre_flood_paths": [str(path.absolute()) for path in tile_paths],
                        "post_flood_path": str(post_flood_path[0].absolute()),
                        "site": site,
                    }
                )

        return tile_pairs

    def _create_patch_metadata(self) -> list[dict]:
        total_patches = 0
        tile_patch_counts = []

        for tile_pair in self.tile_pairs:
            try:
                all_image_paths = tile_pair["pre_flood_paths"] + [tile_pair["post_flood_path"]]

                min_height, min_width = float("inf"), float("inf")

                for img_path in all_image_paths:
                    img_height, img_width = get_image_size(img_path)
                    min_height = min(min_height, img_height)
                    min_width = min(min_width, img_width)

                if min_height < self.patch_size or min_width < self.patch_size:
                    continue

                max_i = min_height - self.patch_size + 1
                max_j = min_width - self.patch_size + 1
                patch_count = len(range(0, max_i, self.patch_stride)) * len(range(0, max_j, self.patch_stride))
                tile_patch_counts.append(
                    (
                        tile_pair,
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

            for i in range(0, max_i, self.patch_stride):
                for j in range(0, max_j, self.patch_stride):
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

    def _extract_patches_at_coords(
        self, image_paths: list[Path], patch_coords: tuple[int, int], format: str
    ) -> list[np.ndarray]:
        patches = []
        i, j = patch_coords

        for img_path in image_paths:
            if format == "geotiff":
                with rasterio.open(img_path) as src:
                    data = src.read()
                    data = np.transpose(data, (1, 2, 0))
            elif format == "numpy":
                data = np.load(img_path)
            else:
                raise ValueError(f"Invalid format: {format}")

            patch = data[i : i + self.patch_size, j : j + self.patch_size, :]
            patches.append(patch)

        return patches

    def __len__(self) -> int:
        return len(self.patch_metadata)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        patch_meta = self.patch_metadata[idx]
        tile_pair = patch_meta["tile_pair"]
        patch_coords = patch_meta["patch_coords"]

        pre_flood_patches = self._extract_patches_at_coords(
            tile_pair["pre_flood_paths"], patch_coords, self.pre_flood_format
        )

        post_flood_patches = self._extract_patches_at_coords(
            [tile_pair["post_flood_path"]], patch_coords, self.post_flood_format
        )

        pre_flood_patches_numpy = np.stack(pre_flood_patches)  # (T, H, W, C)
        post_flood_patches_numpy = np.stack(post_flood_patches)  # (1, H, W, C)

        if self.transform:
            pre_flood_patches_numpy = self.transform(pre_flood_patches_numpy)
            post_flood_patches_numpy = self.transform(post_flood_patches_numpy)

        pre_flood_patches_numpy = pre_flood_patches_numpy.transpose(0, 3, 1, 2)  # (T,H,W,C) -> (T,C,H,W)
        post_flood_patches_numpy = post_flood_patches_numpy.transpose(0, 3, 1, 2)  # (1,H,W,C) -> (1,C,H,W)

        pre_flood_tensor = torch.from_numpy(pre_flood_patches_numpy).float()
        post_flood_tensor = torch.from_numpy(post_flood_patches_numpy).float()

        return pre_flood_tensor, post_flood_tensor
