import matplotlib.pyplot as plt
import numpy as np

from flood_detection_core.data.loaders.tif import TifResult


def create_rgb_composite(data: np.ndarray, enhance: bool = True) -> np.ndarray:
    """
    Create RGB composite from VV/VH SAR data.

    Parameters
    ----------
        data: SAR data with shape (height, width, 2)
        enhance: Whether to enhance contrast

    Returns
    -------
        RGB composite array
    """
    if data.shape[2] != 2:
        raise ValueError("Expected 2 bands (VV, VH)")

    vv = data[:, :, 0]
    vh = data[:, :, 1]

    # Create RGB composite: R=VH, G=VV, B=VH/VV ratio
    # Normalize to 0-1 range
    vv_norm = (vv - np.nanmin(vv)) / (np.nanmax(vv) - np.nanmin(vv))
    vh_norm = (vh - np.nanmin(vh)) / (np.nanmax(vh) - np.nanmin(vh))

    # Calculate VH/VV ratio (avoid division by zero)
    ratio = np.divide(vh, vv, out=np.zeros_like(vh), where=vv != 0)
    ratio_norm = (ratio - np.nanmin(ratio)) / (np.nanmax(ratio) - np.nanmin(ratio))

    # Stack as RGB
    rgb = np.stack([vh_norm, vv_norm, ratio_norm], axis=2)

    # Enhance contrast if requested
    if enhance:
        rgb = np.clip(rgb * 1.5, 0, 1)

    return rgb


def plot_ts_images(
    pre_flood_data: list[TifResult],
    post_flood_data: TifResult,
    tile_name: str,
) -> tuple[plt.Figure, np.ndarray]:
    n_cols = len(pre_flood_data) + 1
    n_rows = 3

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 12))
    fig.suptitle(f"SAR Time Series Comparison - {tile_name}", fontsize=16)

    for i, img_data in enumerate(pre_flood_data):
        axes[0, i].imshow(img_data.data[:, :, 0], cmap="gray")
        axes[0, i].set_title(f"VV - {img_data.date.strftime('%Y-%m-%d')}", fontsize=10)
        axes[0, i].axis("off")

        axes[1, i].imshow(img_data.data[:, :, 1], cmap="gray")
        axes[1, i].set_title(f"VH - {img_data.date.strftime('%Y-%m-%d')}", fontsize=10)
        axes[1, i].axis("off")

        rgb = create_rgb_composite(img_data.data)
        axes[2, i].imshow(rgb)
        axes[2, i].set_title(f"RGB - {img_data.date.strftime('%Y-%m-%d')}", fontsize=10)
        axes[2, i].axis("off")

    # post flood
    axes[0, -1].imshow(post_flood_data.data[:, :, 0], cmap="gray")
    axes[0, -1].set_title(
        f"VV - {post_flood_data.date.strftime('%Y-%m-%d')} (POST)",
        fontsize=10,
        color="red",
        weight="bold",
    )
    axes[0, -1].axis("off")

    axes[1, -1].imshow(post_flood_data.data[:, :, 1], cmap="gray")
    axes[1, -1].set_title(
        f"VH - {post_flood_data.date.strftime('%Y-%m-%d')} (POST)",
        fontsize=10,
        color="red",
        weight="bold",
    )
    axes[1, -1].axis("off")

    post_flood_rgb = create_rgb_composite(post_flood_data.data)
    axes[2, -1].imshow(post_flood_rgb)
    axes[2, -1].set_title(
        f"RGB - {post_flood_data.date.strftime('%Y-%m-%d')} (POST)",
        fontsize=10,
        color="red",
        weight="bold",
    )
    axes[2, -1].axis("off")

    axes[0, 0].set_ylabel("VV Band", fontsize=12, weight="bold")
    axes[1, 0].set_ylabel("VH Band", fontsize=12, weight="bold")
    axes[2, 0].set_ylabel("RGB Composite", fontsize=12, weight="bold")

    plt.tight_layout()
    return fig, axes


def plot_change_detection(
    tile_name: str,
    pre_data: np.ndarray,
    post_data: np.ndarray,
    difference: np.ndarray,
    raw_change_mask: np.ndarray,
    refined_change_mask: np.ndarray,
    permanent_water_data: TifResult,
    ground_truth_data: TifResult,
) -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    fig.suptitle(f"Change Detection - {tile_name}", fontsize=16)

    # Row 1: SAR Images
    # Before flood (VH)
    im1 = axes[0, 0].imshow(pre_data[:, :, 1], cmap="gray")
    axes[0, 0].set_title("Before Flood (VH)")
    axes[0, 0].axis("off")
    plt.colorbar(im1, ax=axes[0, 0])

    # After flood (VH)
    im2 = axes[0, 1].imshow(post_data[:, :, 1], cmap="gray")
    axes[0, 1].set_title("After Flood (VH)")
    axes[0, 1].axis("off")
    plt.colorbar(im2, ax=axes[0, 1])

    # Difference image
    im3 = axes[0, 2].imshow(difference, cmap="RdYlBu_r", vmin=0, vmax=2)
    axes[0, 2].set_title("Difference (After/Before)")
    axes[0, 2].axis("off")
    plt.colorbar(im3, ax=axes[0, 2])

    # Row 2: Change Detection Results
    # Raw change mask
    im4 = axes[1, 0].imshow(raw_change_mask, cmap="Blues")
    axes[1, 0].set_title("Raw Change Detection")
    axes[1, 0].axis("off")
    plt.colorbar(im4, ax=axes[1, 0])

    # Refined change mask
    im5 = axes[1, 1].imshow(refined_change_mask, cmap="Blues")
    axes[1, 1].set_title("Refined Flood Areas")
    axes[1, 1].axis("off")
    plt.colorbar(im5, ax=axes[1, 1])

    # Permanent water mask
    im6 = axes[1, 2].imshow(permanent_water_data.data, cmap="Greens", vmin=0, vmax=1)
    axes[1, 2].set_title("Permanent Water Mask")
    axes[1, 2].axis("off")
    plt.colorbar(im6, ax=axes[1, 2])

    # Row 3
    # Ground truth
    im7 = axes[2, 0].imshow(ground_truth_data.data, cmap="Blues", vmin=-1, vmax=1)
    axes[2, 0].set_title("Ground Truth of surface water")
    axes[2, 0].axis("off")
    plt.colorbar(im7, ax=axes[2, 0])

    # Ground truth minus permanent water mask
    im8 = axes[2, 1].imshow(ground_truth_data.data - permanent_water_data.data, cmap="Blues", vmin=-1, vmax=1)
    axes[2, 1].set_title("GT surface water without Permanent Water Mask")
    axes[2, 1].axis("off")
    plt.colorbar(im8, ax=axes[2, 1])

    fig.delaxes(axes[2, 2])

    plt.tight_layout()
    return fig, axes
