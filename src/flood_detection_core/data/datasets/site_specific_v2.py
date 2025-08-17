import csv
import random
from collections import OrderedDict
from pathlib import Path
from typing import Literal

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

from flood_detection_core.config import AugmentationConfig, CLVAEConfig, DataConfig
from flood_detection_core.data.processing.augmentation import augment_data
from flood_detection_core.data.processing.utils import choose_pre_flood_paths
from flood_detection_core.data.processing.utils import get_image_size_v2 as get_image_size


class SiteSpecificTrainingDataset(Dataset):
    """Dataset for site-specific training using only pre-flood images with MoCo-style pairing.

    Implements proper contrastive learning where:
    - Positive pairs: Two different augmentations of the SAME patch
    - Negative pairs: Augmentations of DIFFERENT patches

    Output:
        - If use_contrastive_pairing_rules is False:
            returns (seq1, seq2)
        - If use_contrastive_pairing_rules is True:
            returns (seq1, seq2, label) where label in {0, 1}

    Shapes:
        seq: (T, H, W, C) where T=num_temporal_length, H=W=patch_size, C=2
        After batching: (B, T, H, W, C)
    """

    def __init__(
        self,
        pre_flood_split_csv_path: Path,
        sites: list[str],
        # Core parameters
        num_temporal_length: int = 4,
        patch_size: int = 16,
        # Augmentation
        augmentation_config: AugmentationConfig | None = None,
        use_contrastive_pairing_rules: bool = False,
        positive_pair_ratio: float = 0.5,
        # Data limits
        max_patches_per_pair: int | Literal["all"] | None = "all",
        # Normalization
        vv_clipped_range: tuple[float, float] | None = None,
        vh_clipped_range: tuple[float, float] | None = None,
        # Deterministic sampling
        base_seed: int | None = None,
    ) -> None:
        super().__init__()

        self.pre_flood_split_csv_path = Path(pre_flood_split_csv_path)
        self.sites = sites
        self.num_temporal_length = num_temporal_length
        self.patch_size = patch_size
        self.max_patches_per_pair = max_patches_per_pair
        self.vv_clipped_range = vv_clipped_range
        self.vh_clipped_range = vh_clipped_range
        self.augmentation_config = augmentation_config
        self.use_contrastive_pairing_rules = use_contrastive_pairing_rules
        self.positive_pair_ratio = positive_pair_ratio

        # Seed for deterministic sampling if provided
        self._base_seed = base_seed if base_seed is not None else random.randint(0, 2**31 - 1)

        # Small in-memory LRU cache to minimize IO; stores normalized arrays per image path
        self._image_cache: OrderedDict[Path, np.ndarray] = OrderedDict()
        self._image_cache_capacity: int = 128

        # Build mapping: site -> tile -> list[pre_flood_paths]
        self.site_tile_to_paths = self._load_pre_flood_split_csv(self.pre_flood_split_csv_path, set(self.sites))

        # Pre-compute ALL individual patch locations (not pairs)
        self.patch_locations: list[tuple[str, str, int, int]] = []  # (site, tile, i, j)
        self._precompute_patch_coordinates()

    def _precompute_patch_coordinates(self) -> None:
        """Pre-compute all individual patch locations (not pairs)."""
        for site, tiles_dict in self.site_tile_to_paths.items():
            for tile, paths in tiles_dict.items():
                if len(paths) < self.num_temporal_length:
                    continue

                try:
                    # Pick a sample sequence to get dimensions
                    sample_paths = self._pick_sequence_paths(site, tile)
                    min_h = min(get_image_size(Path(p))[0] for p in sample_paths)
                    min_w = min(get_image_size(Path(p))[1] for p in sample_paths)

                    if min_h < self.patch_size or min_w < self.patch_size:
                        continue

                    # Generate grid positions
                    stride = self.patch_size
                    for i in range(0, min_h - self.patch_size + 1, stride):
                        for j in range(0, min_w - self.patch_size + 1, stride):
                            self.patch_locations.append((site, tile, i, j))

                except Exception:
                    # Skip problematic tiles
                    continue

        # Apply max_patches limit if specified
        if self.max_patches_per_pair != "all" and self.max_patches_per_pair is not None:
            rng = random.Random(self._base_seed)
            rng.shuffle(self.patch_locations)
            self.patch_locations = self.patch_locations[: self.max_patches_per_pair]

    @staticmethod
    def _load_pre_flood_split_csv(
        split_csv_path: Path,
        sites_filter: set[str],
    ) -> dict[str, dict[str, list[str]]]:
        mapping: dict[str, dict[str, list[str]]] = {}
        with open(split_csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("dataset_type") != "train_val":
                    continue
                site = row.get("site", "")
                tile = row.get("tile", "")
                if site not in sites_filter:
                    continue
                mapping.setdefault(site, {})
                mapping[site].setdefault(tile, [])
                mapping[site][tile].append(row["path"])
        return mapping

    def __len__(self) -> int:
        return len(self.patch_locations)

    def _pick_sequence_paths(self, site: str, tile: str) -> list[Path | str]:
        paths = self.site_tile_to_paths[site][tile]
        # choose_pre_flood_paths supports list[str] and list[Path]
        return choose_pre_flood_paths(paths, self.num_temporal_length)

    def set_epoch(self, epoch: int) -> None:
        """Clear image cache when epoch changes."""
        self._image_cache.clear()

    def _get_positive_pair(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate a positive pair: two different augmentations of the SAME patch.

        Parameters
        ----------
        idx : int
            Index for the patch location

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Two augmented versions of the same temporal patch
        """
        # Get patch location
        site, tile, i, j = self.patch_locations[idx]

        # Get temporal sequence paths
        seq_paths = self._pick_sequence_paths(site, tile)

        # Extract the temporal patch (T, H, W, C)
        patches = [
            self._get_image_array(Path(p))[i : i + self.patch_size, j : j + self.patch_size, :] for p in seq_paths
        ]
        temporal_patch = np.stack([np.ascontiguousarray(p) for p in patches], axis=0)

        # Apply two different random augmentations to the SAME patch
        if self.augmentation_config is not None:
            seq_a = augment_data(temporal_patch.copy(), self.augmentation_config, normalize=False)
            seq_b = augment_data(temporal_patch.copy(), self.augmentation_config, normalize=False)
        else:
            seq_a = temporal_patch.copy()
            seq_b = temporal_patch.copy()

        return seq_a, seq_b

    def _get_negative_pair(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate a negative pair: augmentations of DIFFERENT patches.

        Parameters
        ----------
        idx : int
            Index for the first patch location

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Two augmented versions of different temporal patches
        """
        # Get first patch
        site1, tile1, i1, j1 = self.patch_locations[idx]
        seq_paths1 = self._pick_sequence_paths(site1, tile1)

        patches1 = [
            self._get_image_array(Path(p))[i1 : i1 + self.patch_size, j1 : j1 + self.patch_size, :] for p in seq_paths1
        ]
        temporal_patch1 = np.stack([np.ascontiguousarray(p) for p in patches1], axis=0)

        # Get a different patch (ensure it's not the same location)
        other_idx = random.randint(0, len(self.patch_locations) - 1)
        while other_idx == idx:
            other_idx = random.randint(0, len(self.patch_locations) - 1)

        site2, tile2, i2, j2 = self.patch_locations[other_idx]
        seq_paths2 = self._pick_sequence_paths(site2, tile2)

        patches2 = [
            self._get_image_array(Path(p))[i2 : i2 + self.patch_size, j2 : j2 + self.patch_size, :] for p in seq_paths2
        ]
        temporal_patch2 = np.stack([np.ascontiguousarray(p) for p in patches2], axis=0)

        # Apply augmentations independently
        if self.augmentation_config is not None:
            seq_a = augment_data(temporal_patch1, self.augmentation_config, normalize=False)
            seq_b = augment_data(temporal_patch2, self.augmentation_config, normalize=False)
        else:
            seq_a = temporal_patch1
            seq_b = temporal_patch2

        return seq_a, seq_b

    def _get_image_array(self, img_path: Path) -> np.ndarray:
        if not isinstance(img_path, Path):
            img_path = Path(img_path)
        # LRU hit
        if img_path in self._image_cache:
            arr = self._image_cache.pop(img_path)
            self._image_cache[img_path] = arr
            return arr
        if img_path.suffix == ".tif":
            with rasterio.open(img_path) as src:
                data = src.read()
                if data.shape[0] == 2:
                    data = np.transpose(data, (1, 2, 0))
                else:
                    data = np.asarray(data)
        elif img_path.suffix == ".npy":
            data = np.load(img_path)
        else:
            raise ValueError(f"Invalid format: {img_path}")

        # Apply per-channel clipping/normalization if requested
        if self.vv_clipped_range is not None:
            vv_band = data[:, :, 0].copy()
            vv_band = np.where(np.isnan(vv_band), self.vv_clipped_range[0], vv_band)
            data[:, :, 0] = np.clip(
                (vv_band - self.vv_clipped_range[0]) / (self.vv_clipped_range[1] - self.vv_clipped_range[0]),
                0,
                1,
            )
        if self.vh_clipped_range is not None:
            vh_band = data[:, :, 1].copy()
            vh_band = np.where(np.isnan(vh_band), self.vh_clipped_range[0], vh_band)
            data[:, :, 1] = np.clip(
                (vh_band - self.vh_clipped_range[0]) / (self.vh_clipped_range[1] - self.vh_clipped_range[0]),
                0,
                1,
            )

        # Insert into LRU
        self._image_cache[img_path] = data
        if len(self._image_cache) > self._image_cache_capacity:
            # Evict LRU
            self._image_cache.popitem(last=False)
        return data

    def __getitem__(self, idx: int):
        """Get a training sample (positive or negative pair based on configuration)."""
        try:
            if self.use_contrastive_pairing_rules:
                # Decide if this is a positive or negative pair
                is_positive = random.random() < self.positive_pair_ratio

                if is_positive:
                    seq_a, seq_b = self._get_positive_pair(idx)
                    label = 1
                else:
                    seq_a, seq_b = self._get_negative_pair(idx)
                    label = 0

                # Convert to tensors
                tensor_a = torch.from_numpy(seq_a).float()
                tensor_b = torch.from_numpy(seq_b).float()

                return tensor_a, tensor_b, torch.tensor(label, dtype=torch.long)
            else:
                # Non-contrastive mode: just return two different patches
                seq_a, seq_b = self._get_negative_pair(idx)

                tensor_a = torch.from_numpy(seq_a).float()
                tensor_b = torch.from_numpy(seq_b).float()

                return tensor_a, tensor_b

        except Exception as e:
            # Fallback to a different sample if error occurs
            fallback_idx = random.randint(0, len(self.patch_locations) - 1)
            if fallback_idx != idx:
                return self.__getitem__(fallback_idx)
            else:
                raise RuntimeError(f"Failed to load sample {idx}: {e}")


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    data_config = DataConfig.from_yaml(Path("./flood-detection-core/yamls/data.yaml"))
    clvae_config = CLVAEConfig.from_yaml(Path("./flood-detection-core/yamls/model_clvae.yaml"))

    dataset = SiteSpecificTrainingDataset(
        pre_flood_split_csv_path=data_config.splits.pre_flood_split,
        sites=["bolivia"],
        num_temporal_length=clvae_config.site_specific.num_temporal_length,
        patch_size=clvae_config.site_specific.patch_size,
        max_patches_per_pair="all",  # Use all available patches
        augmentation_config=clvae_config.augmentation,
        use_contrastive_pairing_rules=True,
        positive_pair_ratio=0.5,
    )
    print(f"Dataset length: {len(dataset)}")
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
    batch = next(iter(loader))
    if isinstance(batch, tuple | list) and len(batch) == 3:
        a, b, y = batch
        print("Batch shapes:", a.shape, b.shape, y.shape)
        print(f"Labels in batch: {y.tolist()}")
        print(f"Positive pairs: {(y == 1).sum().item()}, Negative pairs: {(y == 0).sum().item()}")
    else:
        a, b = batch
        print("Batch shapes:", a.shape, b.shape)
