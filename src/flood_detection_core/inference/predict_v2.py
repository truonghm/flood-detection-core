"""
Inference pipeline v2 for CLVAE change detection using FloodDetectionInferenceDataset.

This module implements Algorithm 1 from the paper with the current dataset API:
- Mirror padding (pad size typically 8)
- Dense stride-1 patch extraction (patch size typically 16)
- Encoder-only latent mean comparison via cosine distance
- Thresholding (default 0.5) to obtain binary change maps

Key differences vs the old pipeline:
- Works directly with FloodDetectionInferenceDataset via DataLoader
- Uses dataset-provided metadata to reassemble a full-resolution map per tile
"""

from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from flood_detection_core.models.clvae import CLVAE

__all__ = ["predict_change_maps_from_dataset"]


def _load_model(model_path: Path | str, device: str = "cpu") -> CLVAE:
    """Load CLVAE from checkpoint, supporting multiple state dict keys."""
    model = CLVAE()
    checkpoint = torch.load(Path(model_path), map_location=device)

    state = (
        checkpoint.get("model_state")
        or checkpoint.get("model_state_dict")
        or checkpoint.get("state_dict")
        or checkpoint
    )
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def _compute_cosine_distances(
    model: CLVAE,
    pre_seq: torch.Tensor,
    post_seq: torch.Tensor,
    num_temporal_length: int,
) -> torch.Tensor:
    """Compute cosine distances between latent means for pre/post sequences.

    pre_seq, post_seq: (B, T or 1, H, W, C)
    Returns distances: (B,)
    """
    # Align time dimension
    if post_seq.shape[1] == 1:
        post_seq = post_seq.repeat(1, num_temporal_length, 1, 1, 1)
    else:
        post_seq = post_seq[:, -num_temporal_length:]
    pre_seq = pre_seq[:, -num_temporal_length:]

    pre_mu, _ = model.encode(pre_seq)
    post_mu, _ = model.encode(post_seq)

    cos_sim = F.cosine_similarity(pre_mu, post_mu, dim=1)
    distances = 1 - cos_sim
    return distances


def predict_change_maps_from_dataset(
    model_path: Path | str,
    dataset: Dataset,
    *,
    device: str = "cpu",
    batch_size: int = 512,
    num_temporal_length: int = 4,
    threshold: float = 0.5,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Run inference and reconstruct per-tile change maps.

    Parameters
    ----------
    model_path: Path | str
        Checkpoint path containing model weights.
    dataset: Dataset
        FloodDetectionInferenceDataset instantiated with return_metadata=True.
    device: str
        Torch device string (e.g., "cuda" or "cpu").
    batch_size: int
        Inference batch size for DataLoader (paper uses 512).
    num_temporal_length: int
        Number of temporal steps to consider during inference (e.g., 4).
    threshold: float
        Cosine-distance threshold to binarize change (default 0.5).

    Returns
    -------
    tuple(dict, dict)
        - change_maps: tile_id -> binary ndarray of shape (H, W)
        - distance_maps: tile_id -> float ndarray of shape (H, W)
    """
    model = _load_model(model_path, device=device)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Aggregators
    per_tile_distances: dict[str, dict[tuple[int, int], float]] = defaultdict(dict)
    # Track maxima of raw padded indices per tile to recover original H/W
    per_tile_i_max: dict[str, int] = {}
    per_tile_j_max: dict[str, int] = {}

    # Best-effort retrieval of patch/pad sizes from dataset, with safe defaults
    patch_size = int(getattr(dataset, "patch_size", 16))
    pad_size = int(getattr(dataset, "pad_size", 8))

    with torch.no_grad():
        for batch in loader:
            if not isinstance(batch, tuple | list) or len(batch) < 3:
                raise ValueError("Dataset must be constructed with return_metadata=True to reconstruct full maps.")

            pre, post, meta = batch

            pre = pre.to(device)
            post = post.to(device)

            distances = (
                _compute_cosine_distances(
                    model=model,
                    pre_seq=pre,
                    post_seq=post,
                    num_temporal_length=num_temporal_length,
                )
                .cpu()
                .numpy()
            )

            # Collated meta has structure: dict[str, list|tensor]
            tile_ids: Iterable[str]
            if isinstance(meta["tile_id"], list | tuple):
                tile_ids = meta["tile_id"]
            else:
                tile_ids = [str(x) for x in meta["tile_id"]]

            # Indices are typically tensors after collation
            is_tensor = torch.is_tensor(meta["i"]) and torch.is_tensor(meta["j"])
            i_vals = meta["i"].tolist() if is_tensor else list(meta["i"])  # type: ignore[arg-type]
            j_vals = meta["j"].tolist() if is_tensor else list(meta["j"])  # type: ignore[arg-type]

            for b, tile_id in enumerate(tile_ids):
                i = int(i_vals[b])
                j = int(j_vals[b])

                # Map patch center to original (unpadded) coordinates
                orig_i = i - pad_size + patch_size // 2
                orig_j = j - pad_size + patch_size // 2

                if orig_i < 0 or orig_j < 0:
                    continue

                per_tile_distances[tile_id][(orig_i, orig_j)] = float(distances[b])

                # Track padded index maxima to deduce original dimensions
                per_tile_i_max[tile_id] = max(per_tile_i_max.get(tile_id, 0), i)
                per_tile_j_max[tile_id] = max(per_tile_j_max.get(tile_id, 0), j)

    # Build dense arrays per tile
    change_maps: dict[str, np.ndarray] = {}
    distance_maps: dict[str, np.ndarray] = {}

    for tile_id, coords_dict in per_tile_distances.items():
        # Recover original unpadded H/W from maxima of padded top-left indices
        # i_max = min_h + 2*pad - patch_size  =>  min_h = i_max - 2*pad + patch_size
        i_max = per_tile_i_max.get(tile_id, 0)
        j_max = per_tile_j_max.get(tile_id, 0)
        H = max(0, i_max - 2 * pad_size + patch_size)
        W = max(0, j_max - 2 * pad_size + patch_size)
        dist_arr = np.zeros((H, W), dtype=np.float32)
        for (r, c), v in coords_dict.items():
            if 0 <= r < H and 0 <= c < W:
                dist_arr[r, c] = v
        bin_arr = (dist_arr > threshold).astype(np.uint8)
        distance_maps[tile_id] = dist_arr
        change_maps[tile_id] = bin_arr

    return change_maps, distance_maps
