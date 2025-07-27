import numpy as np
from scipy import ndimage

from flood_detection_core.config import AugmentationConfig


def augment_data(
    data: np.ndarray,
    augmentation_config: AugmentationConfig | None = None,
    normalize: bool = False,
) -> np.ndarray:
    """
    Data augmentation for SAR time series patches.
    Input data is assumed to be normalized to [0, 1] already.

    Augmentation specs:
    - Geometric: Flips (left-right p=0.5, up-down p=0.2), rotation (-90° to 90°)
    - Non-geometric: Gaussian blur (3×3 kernel), gamma contrast (0.25-2.0)

    Parameters
    ----------
        data: SAR time series data of shape (time, height, width, channels)
            Expected shape: (4, 16, 16, 2) for CLVAE pre-training
        normalize: If True, normalize data to [0,1] before gamma correction.
            Set to False (default) if data is already normalized.

    Returns
    -------
        Augmented data with same shape as input


    """
    augmented_data = data.copy()
    if not augmentation_config:
        augmentation_config = AugmentationConfig(
            **{
                "geometric": {
                    "left_right": 0.5,
                    "up_down": 0.2,
                    "rotate": [-90, 90],
                },
                "non_geometric": {
                    "gaussian_blur": 0.3,
                    "gamma_contrast_prob": 0.5,
                    "gamma_contrast": (0.25, 2.0),
                },
            }
        )

    lr_prob = augmentation_config.geometric.left_right
    ud_prob = augmentation_config.geometric.up_down
    rotate_range = augmentation_config.geometric.rotate
    gaussian_blur_prob = augmentation_config.non_geometric.gaussian_blur
    gamma_contrast_prob = augmentation_config.non_geometric.gamma_contrast_prob
    gamma_contrast_range = augmentation_config.non_geometric.gamma_contrast

    # geometric
    # left-right flip
    if np.random.random() < lr_prob:
        augmented_data = np.flip(augmented_data, axis=2).copy()

    # up-down flip
    if np.random.random() < ud_prob:
        augmented_data = np.flip(augmented_data, axis=1).copy()

    # random rotate between -90° and 90°
    angle = np.random.uniform(rotate_range[0], rotate_range[1])
    for t in range(augmented_data.shape[0]):
        for c in range(augmented_data.shape[3]):
            augmented_data[t, :, :, c] = ndimage.rotate(
                augmented_data[t, :, :, c], angle, reshape=False, mode="reflect"
            )

    # non-geometric
    # gaussian blur
    if np.random.random() < gaussian_blur_prob:
        sigma = np.random.uniform(0.5, 1.0)
        for t in range(augmented_data.shape[0]):
            for c in range(augmented_data.shape[3]):
                augmented_data[t, :, :, c] = ndimage.gaussian_filter(augmented_data[t, :, :, c], sigma=sigma)

    # gamma contrast
    if np.random.random() < gamma_contrast_prob:
        gamma = np.random.uniform(gamma_contrast_range[0], gamma_contrast_range[1])

        if normalize:
            data_min = augmented_data.min()
            data_max = augmented_data.max()
            normalized_data = (augmented_data - data_min) / (data_max - data_min + 1e-8)
            gamma_corrected = np.power(normalized_data, gamma)
            augmented_data = gamma_corrected * (data_max - data_min) + data_min
        else:
            augmented_data = np.power(np.clip(augmented_data, 0, 1), gamma)

    return augmented_data
