import numpy as np
import scipy.ndimage as ndimage
from skimage import filters
from skimage.transform import resize


def simulate_multilooking(
    image: np.ndarray,
    range_looks: int = 4,
    azimuth_looks: int = 1,
    preserve_range: bool = True,
) -> np.ndarray:
    """Simulate multilooking effect applied during ASF SLC processing.

    Multilooking reduces speckle by averaging neighboring pixels but also
    reduces spatial resolution and creates smoother appearance.

    Args:
        image: Input SAR image array (H, W) or (H, W, C)
        range_looks: Number of looks in range direction (default: 4)
        azimuth_looks: Number of looks in azimuth direction (default: 1)
        preserve_range: Whether to preserve the original value range

    Returns:
        np.ndarray: Multilooked image with same shape as input
    """
    if image.ndim == 2:
        # Single band image
        kernel = np.ones((range_looks, azimuth_looks)) / (range_looks * azimuth_looks)
        result = ndimage.convolve(image, kernel, mode="reflect")
    elif image.ndim == 3:
        # Multi-band image
        result = np.zeros_like(image)
        kernel = np.ones((range_looks, azimuth_looks)) / (range_looks * azimuth_looks)
        for i in range(image.shape[2]):
            result[:, :, i] = ndimage.convolve(image[:, :, i], kernel, mode="reflect")
    else:
        raise ValueError(f"Unsupported image dimensions: {image.ndim}")

    if preserve_range:
        # Ensure output range matches input range
        result = np.clip(result, np.min(image), np.max(image))

    return result


def simulate_goldstein_filter(
    image: np.ndarray,
    window_size: int = 9,
    method: str = "adaptive_gaussian",
    sigma: float = 1.5,
    preserve_range: bool = True,
) -> np.ndarray:
    """Simulate Goldstein filtering applied during ASF interferogram processing.

    The Goldstein filter is complex and specific to interferometric processing,
    but we can approximate its smoothing effect using adaptive filtering.

    Args:
        image: Input SAR image array (H, W) or (H, W, C)
        window_size: Size of the filter window (default: 9, matching ASF)
        method: Filtering method ('adaptive_gaussian', 'median', 'mean')
        sigma: Gaussian sigma for adaptive_gaussian method
        preserve_range: Whether to preserve the original value range

    Returns:
        np.ndarray: Filtered image with same shape as input
    """
    if image.ndim == 2:
        # Single band image
        if method == "adaptive_gaussian":
            result = filters.gaussian(image, sigma=sigma, preserve_range=preserve_range)
        elif method == "median":
            result = filters.median(image, np.ones((window_size, window_size)))
        elif method == "mean":
            kernel = np.ones((window_size, window_size)) / (window_size * window_size)
            result = ndimage.convolve(image, kernel, mode="reflect")
        else:
            raise ValueError(f"Unknown filtering method: {method}")

    elif image.ndim == 3:
        # Multi-band image
        result = np.zeros_like(image)
        for i in range(image.shape[2]):
            if method == "adaptive_gaussian":
                result[:, :, i] = filters.gaussian(image[:, :, i], sigma=sigma, preserve_range=preserve_range)
            elif method == "median":
                result[:, :, i] = filters.median(image[:, :, i], np.ones((window_size, window_size)))
            elif method == "mean":
                kernel = np.ones((window_size, window_size)) / (window_size * window_size)
                result[:, :, i] = ndimage.convolve(image[:, :, i], kernel, mode="reflect")
    else:
        raise ValueError(f"Unsupported image dimensions: {image.ndim}")

    return result


def harmonize_preprocessing(
    gee_image: np.ndarray,
    apply_multilooking: bool = True,
    apply_goldstein: bool = True,
    range_looks: int = 4,
    azimuth_looks: int = 1,
    goldstein_window_size: int = 9,
    goldstein_method: str = "adaptive_gaussian",
    goldstein_sigma: float = 1.5,
) -> np.ndarray:
    """Apply ASF-like processing to GEE image to harmonize with post-flood data.

    This function applies processing steps to make GEE pre-flood images visually
    similar to ASF post-flood images by simulating multilooking and Goldstein filtering.

    Args:
        gee_image: Input GEE SAR image array (H, W) or (H, W, C)
        apply_multilooking: Whether to apply multilooking simulation
        apply_goldstein: Whether to apply Goldstein filtering simulation
        range_looks: Number of looks in range direction for multilooking
        azimuth_looks: Number of looks in azimuth direction for multilooking
        goldstein_window_size: Window size for Goldstein filter simulation
        goldstein_method: Method for Goldstein filter ('adaptive_gaussian', 'median', 'mean')
        goldstein_sigma: Gaussian sigma for adaptive_gaussian method

    Returns:
        np.ndarray: Harmonized image with ASF-like characteristics

    Example:
        >>> import numpy as np
        >>> # Simulate a GEE pre-flood image (VV, VH channels)
        >>> gee_image = np.random.randn(512, 512, 2) * 4 - 12  # Typical SAR values in dB
        >>> # Apply harmonization to match ASF post-flood processing
        >>> harmonized = harmonize_preprocessing(gee_image)
        >>> print(f"Original shape: {gee_image.shape}, Harmonized shape: {harmonized.shape}")
    """
    result = gee_image.copy()

    if apply_multilooking:
        result = simulate_multilooking(result, range_looks=range_looks, azimuth_looks=azimuth_looks)

    if apply_goldstein:
        result = simulate_goldstein_filter(
            result, window_size=goldstein_window_size, method=goldstein_method, sigma=goldstein_sigma
        )

    return result


def batch_harmonize_preprocessing(
    image_batch: np.ndarray,
    apply_multilooking: bool = True,
    apply_goldstein: bool = True,
    range_looks: int = 4,
    azimuth_looks: int = 1,
    goldstein_window_size: int = 9,
    goldstein_method: str = "adaptive_gaussian",
    goldstein_sigma: float = 1.5,
) -> np.ndarray:
    """Apply harmonization preprocessing to a batch of images.

    Args:
        image_batch: Batch of images (N, H, W) or (N, H, W, C)
        apply_multilooking: Whether to apply multilooking simulation
        apply_goldstein: Whether to apply Goldstein filtering simulation
        range_looks: Number of looks in range direction for multilooking
        azimuth_looks: Number of looks in azimuth direction for multilooking
        goldstein_window_size: Window size for Goldstein filter simulation
        goldstein_method: Method for Goldstein filter ('adaptive_gaussian', 'median', 'mean')
        goldstein_sigma: Gaussian sigma for adaptive_gaussian method

    Returns:
        np.ndarray: Batch of harmonized images with same shape as input
    """
    if image_batch.ndim < 3:
        raise ValueError(f"Expected batch with at least 3 dimensions, got {image_batch.ndim}")

    batch_size = image_batch.shape[0]
    result = np.zeros_like(image_batch)

    for i in range(batch_size):
        result[i] = harmonize_preprocessing(
            image_batch[i],
            apply_multilooking=apply_multilooking,
            apply_goldstein=apply_goldstein,
            range_looks=range_looks,
            azimuth_looks=azimuth_looks,
            goldstein_window_size=goldstein_window_size,
            goldstein_method=goldstein_method,
            goldstein_sigma=goldstein_sigma,
        )

    return result


def compare_image_characteristics(
    pre_flood: np.ndarray, post_flood: np.ndarray, channel_names: tuple[str, ...] | None = None
) -> dict:
    """Compare statistical characteristics between pre-flood and post-flood images.

    Useful for evaluating how well the harmonization process matches the
    visual characteristics of the two image types.

    Args:
        pre_flood: Pre-flood image array (H, W) or (H, W, C)
        post_flood: Post-flood image array (H, W) or (H, W, C)
        channel_names: Names for channels (e.g., ('VV', 'VH'))

    Returns:
        dict: Statistical comparison including means, stds, and texture measures
    """
    if pre_flood.shape != post_flood.shape:
        raise ValueError("Pre-flood and post-flood images must have the same shape")

    def calculate_stats(image, name_prefix):
        stats = {}
        if image.ndim == 2:
            # Single channel
            stats[f"{name_prefix}_mean"] = np.mean(image)
            stats[f"{name_prefix}_std"] = np.std(image)
            stats[f"{name_prefix}_texture"] = np.std(filters.sobel(image))
        else:
            # Multi-channel
            n_channels = image.shape[2]
            names = channel_names if channel_names else [f"ch{i}" for i in range(n_channels)]
            for i, ch_name in enumerate(names):
                channel = image[:, :, i]
                stats[f"{name_prefix}_{ch_name}_mean"] = np.mean(channel)
                stats[f"{name_prefix}_{ch_name}_std"] = np.std(channel)
                stats[f"{name_prefix}_{ch_name}_texture"] = np.std(filters.sobel(channel))
        return stats

    comparison = {}
    comparison.update(calculate_stats(pre_flood, "pre"))
    comparison.update(calculate_stats(post_flood, "post"))

    # Calculate differences
    for key in comparison:
        if key.startswith("pre_"):
            post_key = key.replace("pre_", "post_")
            if post_key in comparison:
                diff_key = key.replace("pre_", "diff_")
                comparison[diff_key] = comparison[post_key] - comparison[key]

    return comparison


def resize_to_target(data: np.ndarray, target_size: tuple = (512, 512)) -> np.ndarray:
    """Resize downloaded data to target dimensions."""
    if len(data.shape) == 3:  # Multi-band image
        # skimage resize expects (height, width, channels) and preserves channels
        resized = resize(
            data,
            (*target_size, data.shape[2]),
            order=1,  # Linear interpolation (equivalent to cv2.INTER_LINEAR)
            preserve_range=True,  # Keep original data range
            anti_aliasing=False,
        )  # Disable anti-aliasing for consistency
        return resized.astype(data.dtype)
    else:  # Single band
        resized = resize(data, target_size, order=1, preserve_range=True, anti_aliasing=False)
        return resized.astype(data.dtype)
