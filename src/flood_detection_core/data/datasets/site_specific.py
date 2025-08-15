import csv
import itertools
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
    """
    Dataset for site-specific training using only pre-flood images.

    Provides pairs of pre-flood sequences (P1, P2) from different tiles within the same site.
    Uses grid-based sampling to extract all possible patches from tile pairs.

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
        self.site_tile_to_paths = self._load_pre_flood_split_csv(
            self.pre_flood_split_csv_path, set(self.sites)
        )

        # Build list of tile pairs per site
        site_tile_pairs: list[tuple[str, str, str]] = []  # (site, tileA, tileB)
        for site, tiles_dict in self.site_tile_to_paths.items():
            tiles = [t for t, paths in tiles_dict.items() if len(paths) >= self.num_temporal_length]
            if len(tiles) < 2:
                continue
            for a, b in itertools.combinations(sorted(tiles), 2):
                site_tile_pairs.append((site, a, b))

        # Prepare augmentation configs for pairing rules (only if augmentation is enabled)
        self._non_geometric_only_config = (
            self._build_non_geometric_only_config(self.augmentation_config)
            if self.augmentation_config is not None
            else None
        )

        # Pre-compute ALL patch coordinates for all tile pairs
        self.patch_samples: list[tuple[int, int, int]] = []  # (pair_idx, i, j)
        self._tile_pairs = site_tile_pairs
        self._precompute_patch_coordinates()

    def _precompute_patch_coordinates(self) -> None:
        """Pre-compute all patch coordinates for all tile pairs."""
        for pair_idx, (site, tile_a, tile_b) in enumerate(self._tile_pairs):
            try:
                # Pick temporal sequences for this pair
                seq_paths_a = self._pick_sequence_paths(site, tile_a)
                seq_paths_b = self._pick_sequence_paths(site, tile_b)

                # Compute common dimensions
                min_h_a = min(get_image_size(Path(p))[0] for p in seq_paths_a)
                min_w_a = min(get_image_size(Path(p))[1] for p in seq_paths_a)
                min_h_b = min(get_image_size(Path(p))[0] for p in seq_paths_b)
                min_w_b = min(get_image_size(Path(p))[1] for p in seq_paths_b)

                min_h = min(min_h_a, min_h_b)
                min_w = min(min_w_a, min_w_b)

                if min_h < self.patch_size or min_w < self.patch_size:
                    continue  # Skip pairs that are too small

                # Generate non-overlapping grid with stride = patch_size
                stride = self.patch_size
                i_positions = list(range(0, min_h - self.patch_size + 1, stride))
                j_positions = list(range(0, min_w - self.patch_size + 1, stride))
                coords = [(i, j) for i in i_positions for j in j_positions]

                # Apply max_patches_per_pair limit
                if self.max_patches_per_pair != "all" and self.max_patches_per_pair is not None:
                    # Use deterministic sampling based on pair_idx for reproducibility
                    rng = random.Random(self._base_seed + pair_idx)
                    rng.shuffle(coords)
                    coords = coords[:self.max_patches_per_pair]

                # Add all coordinates for this pair
                for i, j in coords:
                    self.patch_samples.append((pair_idx, i, j))

            except Exception:
                # Skip problematic pairs
                continue

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

    @staticmethod
    def _build_non_geometric_only_config(base_config: AugmentationConfig) -> AugmentationConfig:
        # Disable geometric by setting zero probs and zero rotation range
        from copy import deepcopy

        cfg = deepcopy(base_config)
        cfg.geometric.left_right = 0.0
        cfg.geometric.up_down = 0.0
        cfg.geometric.rotate = (0, 0)
        # Keep non-geometric as-is
        return cfg

    def __len__(self) -> int:
        return len(self.patch_samples)

    def _pick_sequence_paths(self, site: str, tile: str) -> list[Path | str]:
        paths = self.site_tile_to_paths[site][tile]
        # choose_pre_flood_paths supports list[str] and list[Path]
        return choose_pre_flood_paths(paths, self.num_temporal_length)

    def set_epoch(self, epoch: int) -> None:
        """Clear image cache when epoch changes."""
        self._image_cache.clear()

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
        # Get pre-computed patch coordinates
        pair_idx, i, j = self.patch_samples[idx]
        site, tile_a, tile_b = self._tile_pairs[pair_idx]

        try:
            # Get temporal sequences for this pair
            seq_paths_a = self._pick_sequence_paths(site, tile_a)
            seq_paths_b = self._pick_sequence_paths(site, tile_b)

            # Extract patches at the pre-computed coordinates
            patches_a = [
                self._get_image_array(Path(p))[i : i + self.patch_size, j : j + self.patch_size, :]
                for p in seq_paths_a
            ]
            patches_b = [
                self._get_image_array(Path(p))[i : i + self.patch_size, j : j + self.patch_size, :]
                for p in seq_paths_b
            ]

            seq_a = np.stack([np.ascontiguousarray(p) for p in patches_a], axis=0)
            seq_b = np.stack([np.ascontiguousarray(p) for p in patches_b], axis=0)

            label = None
            if self.use_contrastive_pairing_rules:
                # y=1 positive (non-geometric only), y=0 negative (geometric + non-geometric)
                is_positive = random.random() < self.positive_pair_ratio
                if self.augmentation_config is not None:
                    aug_cfg = self._non_geometric_only_config if is_positive else self.augmentation_config
                    seq_a = augment_data(seq_a, aug_cfg, normalize=False)
                    seq_b = augment_data(seq_b, aug_cfg, normalize=False)
                label = 1 if is_positive else 0
            else:
                if self.augmentation_config is not None:
                    seq_a = augment_data(seq_a, self.augmentation_config, normalize=False)
                    seq_b = augment_data(seq_b, self.augmentation_config, normalize=False)

            tensor_a = torch.from_numpy(seq_a).float()
            tensor_b = torch.from_numpy(seq_b).float()

            if label is None:
                return tensor_a, tensor_b
            else:
                return tensor_a, tensor_b, torch.tensor(label, dtype=torch.long)

        except Exception as e:
            # If there's an error, return a random valid sample as fallback
            fallback_idx = random.randint(0, len(self.patch_samples) - 1)
            if fallback_idx != idx:
                return self.__getitem__(fallback_idx)
            else:
                raise RuntimeError(f"Failed to load sample {idx}: {e}")


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    data_config = DataConfig.from_yaml(Path("./yamls/data.yaml"))
    clvae_config = CLVAEConfig.from_yaml(Path("./yamls/model_clvae.yaml"))

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
    else:
        a, b = batch
        print("Batch shapes:", a.shape, b.shape)
