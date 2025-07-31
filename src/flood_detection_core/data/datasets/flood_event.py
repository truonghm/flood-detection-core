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
from flood_detection_core.data.processing.split import get_flood_event_tile_pairs


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
        dataset_type: Literal["train", "val", "test", "pretrain"],
        pre_flood_split_csv_path: Path,
        post_flood_split_csv_path: Path,
        sites: list[str] = HandLabeledSen1Flood11Sites,
        num_temporal_length: int = 4,
        patch_size: int = 16,
        patch_stride: int = 1,
        transform: Callable | None = None,
        vv_clipped_range: tuple[float, float] | None = None,
        vh_clipped_range: tuple[float, float] | None = None,
    ) -> None:
        self.pre_flood_split_csv_path = pre_flood_split_csv_path
        self.post_flood_split_csv_path = post_flood_split_csv_path
        self.num_temporal_length = num_temporal_length
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.transform = transform
        self.sites = sites
        self.vv_clipped_range = vv_clipped_range
        self.vh_clipped_range = vh_clipped_range
        self.dataset_type = dataset_type

        self.tile_pairs = get_flood_event_tile_pairs(
            dataset_type=dataset_type,
            pre_flood_split_csv_path=self.pre_flood_split_csv_path,
            post_flood_split_csv_path=self.post_flood_split_csv_path,
        )
        self.patch_metadata = self._create_patch_metadata()

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
        self,
        image_paths: list[Path | str],
        patch_coords: tuple[int, int],
        vv_clipped_range: tuple[float, float] | None = None,
        vh_clipped_range: tuple[float, float] | None = None,
    ) -> list[np.ndarray]:
        patches = []
        i, j = patch_coords

        for img_path in image_paths:
            if img_path.endswith(".tif"):
                with rasterio.open(img_path) as src:
                    data = src.read()
                    data = np.transpose(data, (1, 2, 0))
            elif img_path.endswith(".npy"):
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

            patch = data[i : i + self.patch_size, j : j + self.patch_size, :]
            patches.append(patch)

        return patches

    def __len__(self) -> int:
        return len(self.patch_metadata)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        patch_meta = self.patch_metadata[idx]
        tile_pair = patch_meta["tile_pair"]
        patch_coords = patch_meta["patch_coords"]

        pre_flood_patches = self._extract_patches_at_coords(tile_pair["pre_flood_paths"], patch_coords)

        post_flood_patches = self._extract_patches_at_coords(
            [tile_pair["post_flood_path"]],
            patch_coords,
            vv_clipped_range=self.vv_clipped_range,
            vh_clipped_range=self.vh_clipped_range,
        )

        pre_flood_patches_numpy = np.stack(pre_flood_patches, axis=0)  # (T, H, W, C)
        post_flood_patches_numpy = np.stack(post_flood_patches, axis=0)  # (1, H, W, C)

        if self.transform:
            pre_flood_patches_numpy = self.transform(pre_flood_patches_numpy)
            post_flood_patches_numpy = self.transform(post_flood_patches_numpy)

        pre_flood_tensor = torch.from_numpy(pre_flood_patches_numpy).float()
        post_flood_tensor = torch.from_numpy(post_flood_patches_numpy).float()

        return pre_flood_tensor, post_flood_tensor


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    from flood_detection_core.config import CLVAEConfig, DataConfig

    data_config = DataConfig.from_yaml("./yamls/data.yaml")
    model_config = CLVAEConfig.from_yaml("./yamls/model_clvae.yaml")

    train_dataset = FloodEventDataset(
        dataset_type="train",
        pre_flood_split_csv_path=data_config.splits.pre_flood_split,
        post_flood_split_csv_path=data_config.splits.post_flood_split,
        sites=["mekong"],
        num_temporal_length=model_config.site_specific.num_temporal_length,
        patch_size=model_config.site_specific.patch_size,
        patch_stride=model_config.site_specific.patch_stride,
        vv_clipped_range=model_config.site_specific.vv_clipped_range,
        vh_clipped_range=model_config.site_specific.vh_clipped_range,
    )

    dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    train_data = next(iter(dataloader))

    print(train_data[0].shape, train_data[1].shape)
