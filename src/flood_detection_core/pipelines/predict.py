"""Inference pipeline v2 for CLVAE change detection using FloodDetectionInferenceDataset.

This module implements Algorithm 1 from the paper with the current dataset API:
- Mirror padding (pad size typically 8)
- Dense stride-1 patch extraction (patch size typically 16)
- Encoder-only latent mean comparison via cosine distance
- Thresholding (default 0.5) to obtain binary change maps

Key differences vs the old pipeline:
- Works directly with FloodDetectionInferenceDataset via DataLoader
- Uses dataset-provided metadata to reassemble a full-resolution map per tile
"""

import datetime
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from rich import print
from torch.utils.data import DataLoader

import wandb
from wandb.sdk.wandb_run import Run
from flood_detection_core.config import CLVAEConfig, DataConfig
from flood_detection_core.data.datasets import FloodDetectionDataset
from flood_detection_core.models.clvae import CLVAE
from flood_detection_core.utils import get_best_model_info, get_site_specific_latest_run

__all__ = ["generate_distance_maps", "load_distance_maps"]


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
    return_latents: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict]:
    """Compute cosine distances between latent means for pre/post sequences.

    pre_seq, post_seq: (B, T or 1, H, W, C)
    Returns distances: (B,) or (distances, latent_info) if return_latents=True
    """
    # Align time dimension
    if post_seq.shape[1] == 1:
        post_seq = post_seq.repeat(1, num_temporal_length, 1, 1, 1)
    else:
        post_seq = post_seq[:, -num_temporal_length:]
    pre_seq = pre_seq[:, -num_temporal_length:]

    pre_mu, pre_logvar = model.encode(pre_seq)
    post_mu, post_logvar = model.encode(post_seq)

    cos_sim = F.cosine_similarity(pre_mu, post_mu, dim=1)
    distances = 1 - cos_sim

    if return_latents:
        latent_info = {
            "pre_mu": pre_mu.cpu().numpy(),
            "pre_sigma": torch.exp(0.5 * pre_logvar).cpu().numpy(),
            "post_mu": post_mu.cpu().numpy(),
            "post_sigma": torch.exp(0.5 * post_logvar).cpu().numpy(),
            "cos_sim": cos_sim.cpu().numpy(),
        }
        return distances, latent_info

    return distances


def _log_latent_info(
    latent_info: dict,
    distance: torch.Tensor,
    save_path: Path,
    wandb_run: Run,
) -> None:
    pre_mu_mean = latent_info["pre_mu"].mean()
    pre_mu_std = latent_info["pre_mu"].std()
    pre_mu_min = latent_info["pre_mu"].min()
    pre_mu_max = latent_info["pre_mu"].max()
    post_mu_mean = latent_info["post_mu"].mean()
    post_mu_std = latent_info["post_mu"].std()
    post_mu_min = latent_info["post_mu"].min()
    post_mu_max = latent_info["post_mu"].max()
    pre_sigma_mean = latent_info["pre_sigma"].mean()
    pre_sigma_std = latent_info["pre_sigma"].std()
    pre_sigma_min = latent_info["pre_sigma"].min()
    pre_sigma_max = latent_info["pre_sigma"].max()
    post_sigma_mean = latent_info["post_sigma"].mean()
    post_sigma_std = latent_info["post_sigma"].std()
    post_sigma_min = latent_info["post_sigma"].min()
    post_sigma_max = latent_info["post_sigma"].max()
    cos_sim_mean = latent_info["cos_sim"].mean()
    cos_sim_std = latent_info["cos_sim"].std()
    cos_sim_min = latent_info["cos_sim"].min()
    cos_sim_max = latent_info["cos_sim"].max()
    distance_mean = distance.mean()
    distance_std = distance.std()
    distance_min = distance.min()
    distance_max = distance.max()

    with open(save_path, "a") as f:
        # also need header
        if f.tell() == 0:
            f.write(
                "pre_mu_mean,pre_mu_std,pre_mu_min,pre_mu_max,post_mu_mean,post_mu_std,post_mu_min,post_mu_max,pre_sigma_mean,pre_sigma_std,pre_sigma_min,pre_sigma_max,post_sigma_mean,post_sigma_std,post_sigma_min,post_sigma_max,cos_sim_mean,cos_sim_std,cos_sim_min,cos_sim_max,distance_mean,distance_std,distance_min,distance_max\n"
            )
        f.write(
            f"{pre_mu_mean},{pre_mu_std},{pre_mu_min},{pre_mu_max},{post_mu_mean},{post_mu_std},{post_mu_min},{post_mu_max},{pre_sigma_mean},{pre_sigma_std},{pre_sigma_min},{pre_sigma_max},{post_sigma_mean},{post_sigma_std},{post_sigma_min},{post_sigma_max},{cos_sim_mean},{cos_sim_std},{cos_sim_min},{cos_sim_max},{distance_mean},{distance_std},{distance_min},{distance_max}\n"
        )

    if wandb_run:
        wandb_run.log(
            {
                "pre_mu_mean": pre_mu_mean,
                "pre_mu_std": pre_mu_std,
                "pre_mu_min": pre_mu_min,
                "pre_mu_max": pre_mu_max,
                "post_mu_mean": post_mu_mean,
                "post_mu_std": post_mu_std,
                "post_mu_min": post_mu_min,
                "post_mu_max": post_mu_max,
                "pre_sigma_mean": pre_sigma_mean,
                "pre_sigma_std": pre_sigma_std,
                "pre_sigma_min": pre_sigma_min,
                "pre_sigma_max": pre_sigma_max,
                "post_sigma_mean": post_sigma_mean,
                "post_sigma_std": post_sigma_std,
                "post_sigma_min": post_sigma_min,
                "post_sigma_max": post_sigma_max,
                "cos_sim_mean": cos_sim_mean,
                "cos_sim_std": cos_sim_std,
                "cos_sim_min": cos_sim_min,
                "cos_sim_max": cos_sim_max,
                "distance_mean": distance_mean,
                "distance_std": distance_std,
                "distance_min": distance_min,
                "distance_max": distance_max,
            }
        )


def generate_distance_maps(
    site: str,
    data_config: DataConfig,
    model_config: CLVAEConfig,
    model_path: Path | str | None = None,
    save_results: bool = True,
    wandb_run: Run | None = None,
    log_latents: bool = False,
) -> dict[str, np.ndarray]:
    """Run inference and reconstruct per-tile distance maps.

    Parameters
    ----------
    site: str
        Site name.
    data_config: DataConfig
        Data configuration.
    model_config: CLVAEConfig
        Model configuration.
    model_path: Path | str | None
        Checkpoint path containing model weights. If None, the latest checkpoint is used.
    save_results: bool
        Whether to save the results to the data_config.artifacts_dirs.site_specific.
    Returns
    -------
    dict[str, np.ndarray]
        distance_maps: tile_id -> float ndarray of shape (H, W)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not model_path:
        latest_run = get_site_specific_latest_run(site, data_config)
        model_info = get_best_model_info(data_config.artifacts_dirs.site_specific / latest_run)
        model_path = Path(model_info["checkpoint_path"])
        print(f"No model path provided, using latest checkpoint from {model_path}")
    model = _load_model(model_path, device=device)

    dataset = FloodDetectionDataset(
        dataset_type="test",
        pre_flood_split_csv_path=data_config.csv_files.pre_flood_split,
        post_flood_split_csv_path=data_config.csv_files.post_flood_split,
        sites=[site],
        num_temporal_length=model_config.inference.num_temporal_length,
        patch_size=model_config.inference.patch_size,
        stride=model_config.inference.patch_stride,
        pad_size=model_config.inference.pad_size,
        pre_flood_vh_clipped_range=model_config.inference.pre_flood_vh_clipped_range,
        pre_flood_vv_clipped_range=model_config.inference.pre_flood_vv_clipped_range,
        post_flood_vh_clipped_range=model_config.inference.post_flood_vh_clipped_range,
        post_flood_vv_clipped_range=model_config.inference.post_flood_vv_clipped_range,
        return_metadata=True,
        normalize_site_name=model_config.inference.normalize_site_name,
    )

    if wandb_run:
        wandb_run.config.update(model_config.inference.model_dump())

    loader = DataLoader(dataset, batch_size=model_config.inference.batch_size, shuffle=False)

    # Aggregators
    per_tile_distances: dict[str, dict[tuple[int, int], float]] = defaultdict(dict)
    # Track maxima of raw padded indices per tile to recover original H/W
    per_tile_i_max: dict[str, int] = {}
    per_tile_j_max: dict[str, int] = {}

    # Best-effort retrieval of patch/pad sizes from dataset, with safe defaults
    patch_size = int(getattr(dataset, "patch_size", 16))
    pad_size = int(getattr(dataset, "pad_size", 8))

    log_save_path = model_path.parent / "latent_info.csv"

    with torch.no_grad():
        for batch in loader:
            if not isinstance(batch, tuple | list) or len(batch) < 3:
                raise ValueError("Dataset must be constructed with return_metadata=True to reconstruct full maps.")

            pre, post, meta = batch

            pre = pre.to(device)
            post = post.to(device)

            result = _compute_cosine_distances(
                model=model,
                pre_seq=pre,
                post_seq=post,
                num_temporal_length=model_config.inference.num_temporal_length,
                return_latents=True,
            )
            distances, latent_info = result
            distances = distances.cpu().numpy()

            if log_latents and len(per_tile_distances) < 3:
                _log_latent_info(latent_info, distances, log_save_path, wandb_run)

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
    distance_maps: dict[str, np.ndarray] = {}

    for tile_id, coords_dict in per_tile_distances.items():
        # Recover original unpadded H/W from maxima of padded top-left indices
        # i_max = min_h + 2*pad - patch_size  =>  min_h = i_max - 2*pad + patch_size
        # i_max = per_tile_i_max.get(tile_id, 0)
        # j_max = per_tile_j_max.get(tile_id, 0)
        # H = max(0, i_max - 2 * pad_size + patch_size)
        # W = max(0, j_max - 2 * pad_size + patch_size)
        H, W = 512, 512
        dist_arr = np.zeros((H, W), dtype=np.float32)
        for (r, c), v in coords_dict.items():
            if 0 <= r < H and 0 <= c < W:
                dist_arr[r, c] = v
        distance_maps[tile_id] = dist_arr

    if save_results:
        distance_maps_dir = model_path.parent / "distance_maps"
        distance_maps_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving distance maps to [green]{distance_maps_dir}[/green]")
        for tile_id, distance_map in distance_maps.items():
            np.save(distance_maps_dir / f"{tile_id}_distance_map.npy", distance_map)
            if wandb_run:
                metadata = {
                    "site": site,
                    "datetime": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "local_path": (distance_maps_dir / f"{tile_id}_distance_map.npy").absolute(),
                    "model_path": model_path,
                }
                artifact = wandb.Artifact("distance_maps", type="dataset", metadata=metadata)
                artifact.add_file(
                    local_path=distance_maps_dir / f"{tile_id}_distance_map.npy",
                    name=f"distance_maps/{tile_id}_distance_map.npy",
                )
                wandb_run.log_artifact(artifact)

    return distance_maps


def load_distance_maps(
    run_dir: Path | None = None, site: str | None = None, data_config: DataConfig | None = None
) -> dict[str, np.ndarray]:
    if not run_dir:
        if not site or not data_config:
            raise ValueError("site and data_config must be provided if run_dir is not provided")
        latest_run = get_site_specific_latest_run(site, data_config)
        run_dir = data_config.artifacts_dirs.site_specific / latest_run
    distance_maps_dir = run_dir / "distance_maps"
    npy_files = list(distance_maps_dir.glob("*.npy"))
    # print(npy_files)
    # return npy_files
    return {pth.stem.replace("_distance_map", ""): np.load(pth) for pth in npy_files}
