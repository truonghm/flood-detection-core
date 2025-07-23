import numpy as np
from rich import print
from scipy import ndimage
from skimage import measure


def detect_change(
    before_data: np.ndarray,
    after_data: np.ndarray,
    permanent_water_mask: np.ndarray | None = None,
    threshold: float = 1.25,
    min_connected_pixels: int = 8,
    slope_threshold: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
        before_data: Pre-flood SAR data
        after_data: Post-flood SAR data
        permanent_water_mask: Permanent water mask (0=no water, 1=permanent water)
        threshold: Change detection threshold
        min_connected_pixels: Minimum connected pixels to keep

    Returns
    -------
        Tuple of (difference_image, raw_change_mask, refined_change_mask)
    """
    if before_data.shape != after_data.shape:
        print(f"[bold red]Warning:[/bold red] Shape mismatch - before: {before_data.shape}, after: {after_data.shape}")
        # Resize to match the smaller dimensions
        min_h = min(before_data.shape[0], after_data.shape[0])
        min_w = min(before_data.shape[1], after_data.shape[1])
        before_data = before_data[:min_h, :min_w, :]
        after_data = after_data[:min_h, :min_w, :]

    # Use VH band for change detection (index 1)
    before_vh = before_data[:, :, 1]
    after_vh = after_data[:, :, 1]

    # Avoid division by zero
    before_vh = np.where(before_vh == 0, 1e-10, before_vh)

    # Calculate ratio (similar to GEE script)
    difference = after_vh / before_vh

    # Apply threshold to create binary change mask
    raw_change_mask = difference < threshold

    # Start with raw change mask for refinement
    refined_change_mask = raw_change_mask.copy()

    # Apply permanent water masking (remove permanent water areas)
    if permanent_water_mask is not None:
        # Resize permanent water mask if necessary
        if permanent_water_mask.shape != refined_change_mask.shape:
            from skimage.transform import resize

            permanent_water_mask = resize(
                permanent_water_mask.astype(float),
                refined_change_mask.shape,
                order=0,
                preserve_range=True,
            ).astype(bool)

        # Remove areas identified as permanent water
        refined_change_mask = refined_change_mask & (~permanent_water_mask.astype(bool))
        print(f"    Permanent water masking removed {np.sum(permanent_water_mask)} pixels")

    # Connected component filtering (remove small isolated areas)
    if min_connected_pixels > 0:
        labeled_mask = measure.label(refined_change_mask, connectivity=2)
        component_sizes = np.bincount(labeled_mask.ravel())[1:]  # Exclude background
        small_components = np.where(component_sizes < min_connected_pixels)[0] + 1

        for component_id in small_components:
            refined_change_mask[labeled_mask == component_id] = False

        removed_pixels = np.sum(raw_change_mask) - np.sum(refined_change_mask)
        print(f"    Connected component filtering removed {removed_pixels} pixels")

    # remove steep slopes
    if slope_threshold > 0:
        # Calculate slope
        slope = np.gradient(difference, axis=0)
        slope_mask = np.abs(slope) > slope_threshold
        refined_change_mask = refined_change_mask & ~slope_mask
        print(f"    Slope filtering removed {np.sum(slope_mask)} pixels")

    return difference, raw_change_mask, refined_change_mask


def filter_speckle(data: np.ndarray, radius_meters: float = 50.0, pixel_size: float = 10.0) -> np.ndarray:
    """
    Parameters
    ----------
        data: SAR data array with shape (height, width, 2) for VV, VH
        radius_meters: Radius in meters for the smoothing kernel
        pixel_size: Pixel size in meters (default 10m for Sentinel-1)

    Returns
    -------
        Filtered SAR data
    """
    if len(data.shape) != 3 or data.shape[2] != 2:
        raise ValueError(f"Expected data shape (height, width, 2), got {data.shape}")

    processed_data = data.copy()

    # Convert radius from meters to pixels
    radius_pixels = radius_meters / pixel_size

    # Create circular kernel similar to GEE's focal_mean
    kernel_size = int(2 * radius_pixels + 1)
    y, x = np.ogrid[:kernel_size, :kernel_size]
    center = kernel_size // 2
    mask = (x - center) ** 2 + (y - center) ** 2 <= radius_pixels**2
    kernel = mask.astype(float)
    kernel = kernel / kernel.sum()  # Normalize

    # Apply filtering to each band
    for band in range(processed_data.shape[2]):
        processed_data[:, :, band] = ndimage.convolve(processed_data[:, :, band], kernel, mode="reflect")

    return processed_data
