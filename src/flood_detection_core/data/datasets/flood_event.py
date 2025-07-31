import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

from flood_detection_core.data.constants import HandLabeledSen1Flood11Sites
from flood_detection_core.data.processing.patches import create_patch_metadata, extract_patches_at_coords
from flood_detection_core.data.processing.split import get_flood_event_tile_pairs


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

        tile_pairs = get_flood_event_tile_pairs(
            dataset_type=dataset_type,
            pre_flood_split_csv_path=self.pre_flood_split_csv_path,
            post_flood_split_csv_path=self.post_flood_split_csv_path,
        )
        self.patch_metadata = create_patch_metadata(
            tile_pairs=tile_pairs,
            num_temporal_length=self.num_temporal_length,
            patch_size=self.patch_size,
            patch_stride=self.patch_stride,
        )

        del tile_pairs

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

            patch = data[i : i + self.patch_size, j : j + self.patch_size, :]
            patches.append(patch)

        return patches

    def __len__(self) -> int:
        return len(self.patch_metadata)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        patch_meta = self.patch_metadata[idx]
        tile_pair = patch_meta["tile_pair"]
        patch_coords = patch_meta["patch_coords"]

        # start_time = time.time()
        pre_flood_patches = extract_patches_at_coords(
            image_paths=tile_pair["pre_flood_paths"],
            patch_coords=patch_coords,
            patch_size=self.patch_size,
            vv_clipped_range=self.vv_clipped_range,
            vh_clipped_range=self.vh_clipped_range,
        )

        post_flood_patches = extract_patches_at_coords(
            image_paths=[tile_pair["post_flood_path"]],
            patch_coords=patch_coords,
            patch_size=self.patch_size,
            vv_clipped_range=self.vv_clipped_range,
            vh_clipped_range=self.vh_clipped_range,
        )
        pre_flood_patches = [np.ascontiguousarray(p) for p in pre_flood_patches]
        post_flood_patches = [np.ascontiguousarray(p) for p in post_flood_patches]
        # print(f"Time taken to extract patches: {time.time() - start_time} seconds")

        # start_time = time.time()

        # pre_flood_patches_numpy = np.stack(pre_flood_patches, axis=0)
        # post_flood_patches_numpy = np.stack(post_flood_patches, axis=0)
        # pre_flood_patches_numpy = np.array(pre_flood_patches)
        # post_flood_patches_numpy = np.array(post_flood_patches)

        pre_flood_patches_numpy = np.empty(
            (len(pre_flood_patches), self.patch_size, self.patch_size, 2), dtype=np.float32
        )
        post_flood_patches_numpy = np.empty((1, self.patch_size, self.patch_size, 2), dtype=np.float32)

        for i, patch in enumerate(pre_flood_patches):
            pre_flood_patches_numpy[i] = patch

        for i, patch in enumerate(post_flood_patches):
            post_flood_patches_numpy[i] = patch

        # print(f"Time taken to stack patches: {time.time() - start_time} seconds")

        # start_time = time.time()
        if self.transform:
            pre_flood_patches_numpy = self.transform(pre_flood_patches_numpy)
            post_flood_patches_numpy = self.transform(post_flood_patches_numpy)

        pre_flood_tensor = torch.from_numpy(pre_flood_patches_numpy).float()
        post_flood_tensor = torch.from_numpy(post_flood_patches_numpy).float()

        # print(f"Time taken to convert to tensor: {time.time() - start_time} seconds")

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

    print("Length of train dataset:", len(train_dataset))

    dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    train_data = next(iter(dataloader))

    print(train_data[0].shape, train_data[1].shape)
