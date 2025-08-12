import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from flood_detection_core.config import AugmentationConfig

try:
    from torchvision.transforms.functional import rotate as torch_rotate
except ImportError:
    torch_rotate = None


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


def gaussian_kernel_2d(kernel_size: int, sigma: float, device: torch.device) -> torch.Tensor:
    """Create a 2D Gaussian kernel for convolution."""
    x = torch.arange(kernel_size, device=device, dtype=torch.float32) - kernel_size // 2
    y = torch.arange(kernel_size, device=device, dtype=torch.float32) - kernel_size // 2
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def augment_data_torch(
    data: torch.Tensor,
    augmentation_config: AugmentationConfig | None = None,
    normalize: bool = False,
) -> torch.Tensor:
    """
    Data augmentation for SAR time series patches using PyTorch tensors.
    Input data is assumed to be normalized to [0, 1] already.

    Augmentation specs:
    - Geometric: Flips (left-right p=0.5, up-down p=0.2), rotation (-90° to 90°)
    - Non-geometric: Gaussian blur (3×3 kernel), gamma contrast (0.25-2.0)

    Parameters
    ----------
        data: SAR time series data of shape (time, height, width, channels)
            Expected shape: (4, 16, 16, 2) for CLVAE pre-training
        augmentation_config: Configuration for augmentation parameters
        normalize: If True, normalize data to [0,1] before gamma correction.
            Set to False (default) if data is already normalized.

    Returns
    -------
        Augmented data with same shape as input (torch.Tensor)

    """
    augmented_data = data.clone()
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

    # geometric augmentations
    # left-right flip
    if torch.rand(1).item() < lr_prob:
        augmented_data = torch.flip(augmented_data, dims=[2])

    # up-down flip
    if torch.rand(1).item() < ud_prob:
        augmented_data = torch.flip(augmented_data, dims=[1])

    # random rotate between -90° and 90°
    if torch_rotate is not None:
        angle = torch.rand(1).item() * (rotate_range[1] - rotate_range[0]) + rotate_range[0]

        # Reshape for torchvision rotation: (time*channels, height, width)
        t, h, w, c = augmented_data.shape
        reshaped_data = augmented_data.permute(0, 3, 1, 2).reshape(t * c, h, w)

        # Apply rotation to all time-channel combinations
        rotated_data = torch_rotate(reshaped_data, angle, fill=0.0)

        # Reshape back to original format
        augmented_data = rotated_data.reshape(t, c, h, w).permute(0, 2, 3, 1)

    # non-geometric augmentations
    # gaussian blur
    if torch.rand(1).item() < gaussian_blur_prob:
        sigma = torch.rand(1).item() * 0.5 + 0.5  # uniform between 0.5 and 1.0
        kernel_size = 3

        # Create Gaussian kernel
        kernel = gaussian_kernel_2d(kernel_size, sigma, augmented_data.device)
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims

        # Apply blur to each time step and channel
        for t in range(augmented_data.shape[0]):
            for c in range(augmented_data.shape[3]):
                # Add batch and channel dimensions for conv2d
                channel_data = augmented_data[t, :, :, c].unsqueeze(0).unsqueeze(0)
                blurred = F.conv2d(channel_data, kernel, padding=kernel_size // 2)
                augmented_data[t, :, :, c] = blurred.squeeze()

    # gamma contrast
    if torch.rand(1).item() < gamma_contrast_prob:
        gamma = torch.rand(1).item() * (gamma_contrast_range[1] - gamma_contrast_range[0]) + gamma_contrast_range[0]

        if normalize:
            data_min = augmented_data.min()
            data_max = augmented_data.max()
            normalized_data = (augmented_data - data_min) / (data_max - data_min + 1e-8)
            gamma_corrected = torch.pow(normalized_data, gamma)
            augmented_data = gamma_corrected * (data_max - data_min) + data_min
        else:
            augmented_data = torch.pow(torch.clamp(augmented_data, 0, 1), gamma)

    return augmented_data
