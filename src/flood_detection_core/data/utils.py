from pathlib import Path

import numpy as np
import rasterio


def create_patches_from_array(image_array: np.ndarray, patch_size: int = 16, stride: int = 16) -> np.ndarray:
    """Create patches from an image array"""
    if len(image_array.shape) != 3:
        raise ValueError(f"Expected 3D array, got {len(image_array.shape)}D")

    height, width, channels = image_array.shape

    # Calculate number of patches
    num_patches_h = (height - patch_size) // stride + 1
    num_patches_w = (width - patch_size) // stride + 1

    patches = []

    for i in range(num_patches_h):
        for j in range(num_patches_w):
            start_h = i * stride
            start_w = j * stride
            end_h = start_h + patch_size
            end_w = start_w + patch_size

            patch = image_array[start_h:end_h, start_w:end_w, :]
            patches.append(patch)

    return np.array(patches)


def create_patches_from_tif(
    tif_path: Path,
    patch_size: int = 16,
    stride: int = 16,
) -> np.ndarray:
    with rasterio.open(tif_path) as src:
        image_array = src.read()
        return create_patches_from_array(image_array, patch_size, stride)
