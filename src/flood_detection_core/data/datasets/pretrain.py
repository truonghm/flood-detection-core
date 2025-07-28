from collections.abc import Callable
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from flood_detection_core.data.processing.patches import PreTrainPatchesExtractor


class PretrainDataset(Dataset):
    """
    Dataset for CLVAE pre-training phase.
    Loads 100 random patches with 4 pre-flood images each for learning basic SAR patterns.

    Expected output format:
    - Pre-flood image sequences: (time_steps, channels, height, width) = (T, C, H, W)
    - After batching: (batch_size, time_steps, channels, height, width) = (B, T, C, H, W)
    - No labels needed - unsupervised learning
    """

    def __init__(
        self,
        pre_flood_dir: Path,
        pre_flood_format: Literal["numpy", "geotiff"],
        pretrain_dir: Path,
        num_patches: int = 100,
        num_temporal_length: int = 4,
        patch_size: int = 16,
        replacement: bool = False,
        transform: Callable | None = None,
    ) -> None:
        self.pretrain_dir = pretrain_dir
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.num_temporal_length = num_temporal_length
        self.transform = transform
        self.patches_extractor = PreTrainPatchesExtractor(
            input_dir=pre_flood_dir,
            input_format=pre_flood_format,
            output_dir=self.pretrain_dir,
            num_patches=num_patches,
            num_temporal_length=num_temporal_length,
            patch_size=patch_size,
            patch_stride=16,
            replacement=replacement,
        )

    def __len__(self) -> int:
        return self.num_patches

    def __getitem__(self, idx: int) -> torch.Tensor:
        patch_sequence = self.patches_extractor(idx)
        if len(patch_sequence) != self.num_temporal_length:
            raise ValueError(f"Patch sequence has {len(patch_sequence)} images, expected {self.num_temporal_length}")

        temporal_images = np.stack(patch_sequence, axis=0)  # (T, H, W, C)
        if self.transform:
            temporal_images = self.transform(temporal_images)

        # temporal_images = temporal_images.transpose(0, 3, 1, 2)  # (T,H,W,C) -> (T,C,H,W)
        return torch.from_numpy(temporal_images).float()


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    from flood_detection_core.config import CLVAEConfig, DataConfig

    data_config = DataConfig.from_yaml("./yamls/data.yaml")
    model_config = CLVAEConfig.from_yaml("./yamls/model_clvae.yaml")

    pretrain_dataset = PretrainDataset(
        pre_flood_dir=data_config.gee.pre_flood_dir,
        pre_flood_format="geotiff",
        pretrain_dir=data_config.gee.pretrain_dir,
        num_patches=model_config.pretrain.num_patches,
        num_temporal_length=model_config.pretrain.num_temporal_length,
        patch_size=model_config.pretrain.patch_size,
        replacement=model_config.pretrain.replacement,
    )

    pretrain_dataloader = DataLoader(pretrain_dataset, batch_size=32, shuffle=True)

    pretrain_data = next(iter(pretrain_dataloader))

    print(pretrain_data.shape)
