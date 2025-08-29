from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from flood_detection_core.data.processing.patches import PreTrainPatchesExtractor


class PretrainDataset(Dataset):
    """Dataset for CLVAE pre-training phase. Loads 100 random patches with 4 pre-flood images each for learning
    basic SAR patterns.

    Expected output format:
    - Pre-flood image sequences: (time_steps, channels, height, width) = (T, C, H, W)
    - After batching: (batch_size, time_steps, channels, height, width) = (B, T, C, H, W)
    - No labels needed - unsupervised learning
    """

    def __init__(
        self,
        split_csv_path: Path,
        pretrain_dir: Path,
        num_patches: int = 100,
        num_temporal_length: int = 4,
        patch_size: int = 16,
        vv_clipped_range: tuple[float, float] | None = None,
        vh_clipped_range: tuple[float, float] | None = None,
        transform: Callable | None = None,
    ) -> None:
        self.pretrain_dir = pretrain_dir
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.num_temporal_length = num_temporal_length
        self.transform = transform
        self.patches_extractor = PreTrainPatchesExtractor(
            split_csv_path=split_csv_path,
            output_dir=self.pretrain_dir,
            num_patches=num_patches,
            num_temporal_length=num_temporal_length,
            patch_size=patch_size,
            vv_clipped_range=vv_clipped_range,
            vh_clipped_range=vh_clipped_range,
        )

    def __len__(self) -> int:
        return self.num_patches

    def __getitem__(self, idx: int) -> torch.Tensor:
        patch_sequence = self.patches_extractor(idx)
        if len(patch_sequence) != self.num_temporal_length:
            raise ValueError(f"Patch sequence has {len(patch_sequence)} images, expected {self.num_temporal_length}")

        temporal_images = np.stack(patch_sequence, axis=0)
        if self.transform:
            temporal_images = self.transform(temporal_images)

        return torch.from_numpy(temporal_images).float()


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    from flood_detection_core.config import CLVAEConfig, DataConfig

    data_config = DataConfig.from_yaml("./flood-detection-core/yamls/data_sen1flood11.yaml")
    model_config = CLVAEConfig.from_yaml("./flood-detection-core/yamls/model_clvae.yaml")

    pretrain_dataset = PretrainDataset(
        split_csv_path=data_config.csv_files.pre_flood_split,
        pretrain_dir=data_config.data_dirs.pretrain_cache,
        num_patches=model_config.pretrain.num_patches,
        num_temporal_length=model_config.pretrain.num_temporal_length,
        patch_size=model_config.pretrain.patch_size,
    )

    pretrain_dataloader = DataLoader(pretrain_dataset, batch_size=32, shuffle=True)

    pretrain_data = next(iter(pretrain_dataloader))

    print(pretrain_data.shape)
