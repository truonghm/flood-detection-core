import datetime
import random
import re
from pathlib import Path

import magic
import rasterio

from flood_detection_core.exceptions import NotEnoughImagesError


def get_image_size(file_path: Path) -> tuple[int, int]:
    metadata = magic.from_file(file_path)
    height = re.search(r"height=(\d+)", metadata).group(1)
    width = re.search(r"width=(\d+)", metadata).group(1)
    return int(height), int(width)


def get_image_size_v2(file_path: Path) -> tuple[int, int]:
    with rasterio.open(file_path) as src:
        return src.height, src.width


def choose_pre_flood_paths(paths: list[Path | str], num_temporal_length: int) -> list[Path]:
    if isinstance(paths[0], str):
        path_objs = [Path(path) for path in paths]
    else:
        path_objs = paths

    ts_length = len(path_objs)
    n_ts_length = num_temporal_length
    path_indices = [int(re.search(r"pre_flood_(\d+)_", path.name).group(1)) for path in path_objs]

    # sort paths based on indices
    sorted_paths = [path for _, path in sorted(zip(path_indices, path_objs))]

    if ts_length < n_ts_length:
        raise NotEnoughImagesError(
            f"Tile has fewer than {n_ts_length} images ({ts_length} < {n_ts_length})" + f"\nCheck: {path_objs[0].name}"
        )

    # randomly pick n_ts_length CONSECUTIVE images
    ts_start_idx = random.randint(0, ts_length - n_ts_length)
    ts_end_idx = ts_start_idx + n_ts_length

    chosen_tile_paths = sorted_paths[ts_start_idx:ts_end_idx]

    return chosen_tile_paths
