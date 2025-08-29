from pathlib import Path
from typing import Literal

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

from flood_detection_core.config import CLVAEConfig, DataConfig
from flood_detection_core.data.processing.imaging import resize_to_target
from flood_detection_core.data.processing.split import get_flood_event_tile_pairs


def _read_tif_hw2(path: Path) -> np.ndarray:
    """Read a GeoTIFF into H×W×2 float32 array without any normalization."""
    with rasterio.open(path) as src:
        data = src.read()  # (C, H, W)
    # Expecting 2 channels (VV, VH)
    data = np.transpose(data, (1, 2, 0)).astype(np.float32)
    data = resize_to_target(data)
    return data


def _normalize(img_hw2: np.ndarray, vv_range: tuple[float, float], vh_range: tuple[float, float]) -> np.ndarray:
    """Normalize only, assuming img has channels [VV, VH].

    Returns [0,1] scaled copy.
    """
    vv_low, vv_high = vv_range
    vh_low, vh_high = vh_range
    out = img_hw2.copy()

    # Replace NaNs with lower bounds before clipping
    vv_band = out[:, :, 0]
    vh_band = out[:, :, 1]
    vv_band = np.where(np.isnan(vv_band), vv_low, vv_band)
    vh_band = np.where(np.isnan(vh_band), vh_low, vh_band)

    out[:, :, 0] = np.clip((vv_band - vv_low) / (vv_high - vv_low), 0.0, 1.0)
    out[:, :, 1] = np.clip((vh_band - vh_low) / (vh_high - vh_low), 0.0, 1.0)
    return out.astype(np.float32)


def _reflect_pad(arr: np.ndarray, pad: int) -> np.ndarray:
    if pad <= 0:
        return arr
    if arr.ndim == 3:  # (H, W, C)
        return np.pad(arr, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    if arr.ndim == 4:  # (T, H, W, C)
        return np.pad(arr, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="reflect")
    raise ValueError(f"Unsupported array ndim for reflect pad: {arr.ndim}")


class FloodDetectionDataset(Dataset):
    """High-performance inference dataset for CLVAE change detection.

    - Uses tile-level caching: pre-loads 4 pre-flood images and 1 post-flood image per tile once, then
    serves stride-1 patches by slicing (no per-patch file IO).
    - Pre-flood images are NOT normalized; post-flood image is normalized to [0,1].
    - Reflect padding (pad_size) is applied before patch slicing.

    Returns per item:
        pre_seq:  (T, patch, patch, 2)
        post_seq: (T, patch, patch, 2)  # post replicated along time dimension
        meta:     dict(tile_id, site, i, j)  # optional, enabled via return_metadata
    """

    def __init__(
        self,
        dataset_type: Literal["train", "val", "test"],
        pre_flood_split_csv_path: Path,
        post_flood_split_csv_path: Path,
        sites: list[str] | None = None,
        num_temporal_length: int = 4,
        patch_size: int = 16,
        stride: int = 1,
        pad_size: int = 8,
        pre_flood_vv_clipped_range: tuple[float, float] | None = None,
        pre_flood_vh_clipped_range: tuple[float, float] | None = None,
        post_flood_vv_clipped_range: tuple[float, float] | None = None,
        post_flood_vh_clipped_range: tuple[float, float] | None = None,
        deterministic_preflood: bool = True,
        return_metadata: bool = False,
        normalize_site_name: bool = True,
    ) -> None:
        super().__init__()

        self.dataset_type = dataset_type
        self.pre_flood_split_csv_path = Path(pre_flood_split_csv_path)
        self.post_flood_split_csv_path = Path(post_flood_split_csv_path)
        self.num_temporal_length = num_temporal_length
        self.patch_size = patch_size
        self.stride = stride
        self.pad_size = pad_size
        self.return_metadata = return_metadata
        self.deterministic_preflood = deterministic_preflood
        self.pre_flood_vv_clipped_range = pre_flood_vv_clipped_range
        self.pre_flood_vh_clipped_range = pre_flood_vh_clipped_range
        self.post_flood_vv_clipped_range = post_flood_vv_clipped_range
        self.post_flood_vh_clipped_range = post_flood_vh_clipped_range

        # Build tile pairs (paths are strings relative to root); convert to absolute Paths
        raw_tile_pairs = get_flood_event_tile_pairs(
            dataset_type=self.dataset_type,
            pre_flood_split_csv_path=self.pre_flood_split_csv_path,
            post_flood_split_csv_path=self.post_flood_split_csv_path,
        )

        if sites is not None:
            sites = [s.lower() for s in sites] if normalize_site_name else sites
            raw_tile_pairs = [tp for tp in raw_tile_pairs if tp["site"] in sites]

        self.tile_pairs: list[dict] = raw_tile_pairs
        # Precompute per-tile window counts (after cropping to min common H/W and padding)
        self._tile_window_counts: list[int] = []
        self._tile_hw: list[tuple[int, int]] = []  # min common H/W per tile before padding
        for tp in self.tile_pairs:
            # Compute min common H/W across all pre/post images for this tile
            heights = []
            widths = []
            # Read sizes without loading full arrays (rasterio.open metadata)
            for p in tp["pre_flood_paths"] + [tp["post_flood_path"]]:
                with rasterio.open(p) as src:
                    heights.append(src.height)
                    widths.append(src.width)
            min_h, min_w = int(min(heights)), int(min(widths))
            self._tile_hw.append((min_h, min_w))

            Hp = min_h + 2 * self.pad_size
            Wp = min_w + 2 * self.pad_size
            n_i = 1 + (Hp - self.patch_size) // self.stride
            n_j = 1 + (Wp - self.patch_size) // self.stride
            self._tile_window_counts.append(n_i * n_j)

        # Prefix sums for global index mapping
        self._prefix_counts: list[int] = [0]
        for c in self._tile_window_counts:
            self._prefix_counts.append(self._prefix_counts[-1] + c)

        self._cache_tile_idx: int | None = None
        self._cache_pre: np.ndarray | None = None  # (T, Hp, Wp, 2)
        self._cache_post: np.ndarray | None = None  # (1, Hp, Wp, 2)

    def __len__(self) -> int:
        return self._prefix_counts[-1]

    def _global_index_to_tile_and_coords(self, idx: int) -> tuple[int, int, int]:
        # Binary search over prefix counts
        lo, hi = 0, len(self._prefix_counts) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self._prefix_counts[mid + 1] <= idx:
                lo = mid + 1
            else:
                hi = mid
        tile_idx = lo
        base = self._prefix_counts[tile_idx]
        offset = idx - base

        min_h, min_w = self._tile_hw[tile_idx]
        # Hp = min_h + 2 * self.pad_size
        Wp = min_w + 2 * self.pad_size
        n_j = 1 + (Wp - self.patch_size) // self.stride
        i_idx = offset // n_j
        j_idx = offset % n_j
        i = i_idx * self.stride
        j = j_idx * self.stride
        return tile_idx, i, j

    def _select_preflood_paths(self, pre_paths: list[Path]) -> list[Path]:
        if self.num_temporal_length <= 0:
            raise ValueError("num_temporal_length must be > 0")
        # Deterministic: order by increasing chronological date (oldest first)
        # Note: Smaller numeric indices correspond to more recent dates in filenames,
        # so to sort chronologically increasing we sort by index in descending order.
        if self.deterministic_preflood:
            import re

            indices = []
            for p in pre_paths:
                m = re.search(r"pre_flood_(\d+)_", p.name)
                if not m:
                    continue
                indices.append((int(m.group(1)), p))
            if len(indices) < self.num_temporal_length:
                raise ValueError("Not enough pre-flood images for deterministic selection")
            indices.sort(key=lambda x: x[0], reverse=True)
            selected = [pair[1] for pair in indices[: self.num_temporal_length]]
            return selected
        else:
            from flood_detection_core.data.processing.utils import choose_pre_flood_paths

            return choose_pre_flood_paths(pre_paths, self.num_temporal_length)

    def _load_tile_into_cache(self, tile_idx: int) -> None:
        if self._cache_tile_idx == tile_idx:
            return

        tp = self.tile_pairs[tile_idx]
        min_h, min_w = self._tile_hw[tile_idx]

        # Select T pre-flood paths (deterministic or not)
        selected_pre_paths = self._select_preflood_paths([Path(p) for p in tp["pre_flood_paths"]])

        # Read and crop pre-flood images WITHOUT normalization
        pre_imgs = []
        for p in selected_pre_paths:
            arr = _read_tif_hw2(Path(p))
            if self.pre_flood_vv_clipped_range and self.pre_flood_vh_clipped_range:
                arr = _normalize(arr, self.pre_flood_vv_clipped_range, self.pre_flood_vh_clipped_range)
            pre_imgs.append(arr[:min_h, :min_w, :])
        pre_stack = np.stack(pre_imgs, axis=0)  # (T, H, W, 2)

        # Read and crop post-flood image WITH normalization
        post_arr = _read_tif_hw2(Path(tp["post_flood_path"]))
        post_arr = post_arr[:min_h, :min_w, :]

        # only normalize post-flood image because pre-flood images are already normalized in download
        if self.post_flood_vv_clipped_range and self.post_flood_vh_clipped_range:
            post_arr = _normalize(post_arr, self.post_flood_vv_clipped_range, self.post_flood_vh_clipped_range)
        # post_arr = match_histograms(pre_imgs[0], post_arr)
        post_stack = post_arr[None, ...]  # (1, H, W, 2)

        # Reflect pad both sequences
        pre_stack = _reflect_pad(pre_stack, self.pad_size)
        post_stack = _reflect_pad(post_stack, self.pad_size)

        self._cache_tile_idx = tile_idx
        self._cache_pre = pre_stack.astype(np.float32)
        self._cache_post = post_stack.astype(np.float32)

    def __getitem__(self, idx: int):
        tile_idx, i, j = self._global_index_to_tile_and_coords(idx)
        self._load_tile_into_cache(tile_idx)
        assert self._cache_pre is not None and self._cache_post is not None

        T = self.num_temporal_length
        ps = self.patch_size

        pre_patch = self._cache_pre[:, i : i + ps, j : j + ps, :]
        post_single = self._cache_post[:, i : i + ps, j : j + ps, :]
        # Replicate post across time to match T
        post_patch = np.repeat(post_single, repeats=T, axis=0)

        pre_tensor = torch.from_numpy(pre_patch).float()
        post_tensor = torch.from_numpy(post_patch).float()

        if self.return_metadata:
            meta = {
                "tile_id": self.tile_pairs[tile_idx]["tile_id"],
                "site": self.tile_pairs[tile_idx]["site"],
                "i": i,
                "j": j,
            }
            return pre_tensor, post_tensor, meta
        return pre_tensor, post_tensor


if __name__ == "__main__":
    # Quick smoke test

    data_cfg = DataConfig.from_yaml(Path("./flood-detection-core/yamls/data_urban_sar.yaml"))
    clvae_cfg = CLVAEConfig.from_yaml(Path("./flood-detection-core/yamls/model_clvae_urban_sar.yaml"))

    ds = FloodDetectionDataset(
        dataset_type="test",
        pre_flood_split_csv_path=data_cfg.csv_files.pre_flood_split,
        post_flood_split_csv_path=data_cfg.csv_files.post_flood_split,
        num_temporal_length=clvae_cfg.site_specific.num_temporal_length,
        patch_size=clvae_cfg.site_specific.patch_size,
        stride=1,
        pad_size=8,
        post_flood_vh_clipped_range=clvae_cfg.site_specific.vh_clipped_range,
        post_flood_vv_clipped_range=clvae_cfg.site_specific.vv_clipped_range,
        return_metadata=True,
    )
    print("Dataset length:", len(ds))
    sample = ds[0]
    print(type(sample), isinstance(sample, tuple))
    print(sample[0].shape, sample[1].shape, sample[2])
