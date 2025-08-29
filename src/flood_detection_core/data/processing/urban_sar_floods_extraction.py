"""Transformation functions for Urban SAR Floods dataset.

This module provides functions to transform Urban SAR Floods data (8-band, 20m resolution)
into a format compatible with Sen1Floods11 (2-band, 10m resolution).

Urban SAR Floods band structure:
- Band 1: pre-event coherence VH
- Band 2: pre-event coherence VV
- Band 3: co-event coherence VH
- Band 4: co-event coherence VV
- Band 5: pre-event intensity VH
- Band 6: pre-event intensity VV
- Band 7: post-event intensity VH
- Band 8: post-event intensity VV
"""

from typing import Literal

import numpy as np
from scipy import ndimage


def extract_pre_event_bands(data: np.ndarray) -> np.ndarray:
    """Extract pre-event intensity bands from Urban SAR Floods data.

    Parameters
    ----------
    data : np.ndarray
        Input array of shape (H, W, 8) containing all bands.

    Returns
    -------
    np.ndarray
        Array of shape (H, W, 2) containing pre-event VH and VV bands.

    Raises
    ------
    ValueError
        If input data does not have 8 bands.
    """
    if data.shape[-1] != 8:
        raise ValueError(f"Expected 8 bands, got {data.shape[-1]}")

    # Bands 5 and 6: pre-event intensity VH and VV (0-indexed: 4 and 5)
    return data[..., 4:6]


def extract_post_event_bands(data: np.ndarray) -> np.ndarray:
    """Extract post-event intensity bands from Urban SAR Floods data.

    Parameters
    ----------
    data : np.ndarray
        Input array of shape (H, W, 8) containing all bands.

    Returns
    -------
    np.ndarray
        Array of shape (H, W, 2) containing post-event VH and VV bands.

    Raises
    ------
    ValueError
        If input data does not have 8 bands.
    """
    if data.shape[-1] != 8:
        raise ValueError(f"Expected 8 bands, got {data.shape[-1]}")

    # Bands 7 and 8: post-event intensity VH and VV (0-indexed: 6 and 7)
    return data[..., 6:8]


def split_temporal_bands(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split Urban SAR Floods data into temporal components.

    Parameters
    ----------
    data : np.ndarray
        Input array of shape (H, W, 8) containing all bands.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple of (pre_event, post_event) arrays, each of shape (H, W, 2).
    """
    pre_event = extract_pre_event_bands(data)
    post_event = extract_post_event_bands(data)

    return pre_event, post_event


def resample_to_10m(data: np.ndarray, method: Literal["nearest", "bilinear", "cubic"] = "bilinear") -> np.ndarray:
    """Resample data from 20m to 10m resolution (2x upsampling).

    Parameters
    ----------
    data : np.ndarray
        Input array of shape (512, 512, C) at 20m resolution.
    method : {"nearest", "bilinear", "cubic"}, optional
        Interpolation method, by default "bilinear".

    Returns
    -------
    np.ndarray
        Array of shape (1024, 1024, C) at 10m resolution.

    Raises
    ------
    ValueError
        If interpolation method is unknown or array has unexpected dimensions.
    """
    if method == "nearest":
        order = 0
    elif method == "bilinear":
        order = 1
    elif method == "cubic":
        order = 3
    else:
        raise ValueError(f"Unknown interpolation method: {method}")

    # Calculate zoom factors
    zoom_factor = 2.0

    if data.ndim == 2:
        zoom_factors = (zoom_factor, zoom_factor)
    elif data.ndim == 3:
        # Don't zoom the channel dimension
        zoom_factors = (zoom_factor, zoom_factor, 1.0)
    else:
        raise ValueError(f"Expected 2D or 3D array, got shape {data.shape}")

    # Perform resampling
    resampled = ndimage.zoom(data, zoom_factors, order=order, mode="reflect")

    return resampled


def crop_center(data: np.ndarray, target_size: int = 512) -> np.ndarray:
    """Crop the center portion of an image to target size.

    Parameters
    ----------
    data : np.ndarray
        Input array of shape (H, W, C) where H, W >= target_size.
    target_size : int, optional
        Target size for the output square image, by default 512.

    Returns
    -------
    np.ndarray
        Center-cropped array of shape (target_size, target_size, C).

    Raises
    ------
    ValueError
        If input image is smaller than target_size in any dimension.
    """
    h, w = data.shape[:2]

    if h < target_size or w < target_size:
        raise ValueError(f"Image size {h}x{w} is smaller than target size {target_size}x{target_size}")

    # Calculate center crop coordinates
    start_h = (h - target_size) // 2
    start_w = (w - target_size) // 2
    end_h = start_h + target_size
    end_w = start_w + target_size

    return data[start_h:end_h, start_w:end_w]


def split_into_four_tiles(data: np.ndarray) -> list[np.ndarray]:
    """Split a 1024x1024 image into 4 non-overlapping 512x512 tiles.

    Parameters
    ----------
    data : np.ndarray
        Input array of shape (1024, 1024, C).

    Returns
    -------
    list[np.ndarray]
        List of 4 arrays, each of shape (512, 512, C):
        [top_left, top_right, bottom_left, bottom_right].

    Raises
    ------
    ValueError
        If input image is not 1024x1024.
    """
    h, w = data.shape[:2]

    if h != 1024 or w != 1024:
        raise ValueError(f"Expected 1024x1024 image, got {h}x{w}")

    mid_h = h // 2
    mid_w = w // 2

    top_left = data[:mid_h, :mid_w]
    top_right = data[:mid_h, mid_w:]
    bottom_left = data[mid_h:, :mid_w]
    bottom_right = data[mid_h:, mid_w:]

    return [top_left, top_right, bottom_left, bottom_right]


def transform_urban_sar(
    data: np.ndarray, resample: bool = True, split_tiles: bool = True
) -> tuple[np.ndarray | list[np.ndarray], np.ndarray | list[np.ndarray]]:
    """Transform Urban SAR Floods data to Sen1Floods11-compatible format.

    This function:
    1. Splits the 8-band data into pre-event and post-event components
    2. Optionally resamples from 20m to 10m resolution
    3. Optionally splits each component into 4 tiles

    Parameters
    ----------
    data : np.ndarray
        Input array of shape (512, 512, 8) at 20m resolution.
    resample : bool, optional
        If True, resample from 20m to 10m resolution, by default True.
    split_tiles : bool, optional
        If True, split resampled data into 4 tiles of 512x512, by default True.

    Returns
    -------
    tuple[np.ndarray | list[np.ndarray], np.ndarray | list[np.ndarray]]
        Tuple of (pre_event, post_event) where each element is:
        - If split_tiles=False: array of shape (H, W, 2)
        - If split_tiles=True: list of 4 arrays, each of shape (512, 512, 2)

    Raises
    ------
    ValueError
        If split_tiles is True but resample is False (cannot split 512x512 into 4x512x512).
    """
    # Split temporal bands
    pre_event, post_event = split_temporal_bands(data)

    if resample:
        # Resample each temporal component
        pre_event = resample_to_10m(pre_event, method="bilinear")
        post_event = resample_to_10m(post_event, method="bilinear")

    if split_tiles and resample:
        # Split each component into 4 tiles
        pre_event = split_into_four_tiles(pre_event)
        post_event = split_into_four_tiles(post_event)
    elif split_tiles and not resample:
        raise ValueError(
            "Cannot split into tiles without resampling. "
            "Original data is 512x512, need 1024x1024 to split into 4x512x512 tiles."
        )

    return pre_event, post_event
