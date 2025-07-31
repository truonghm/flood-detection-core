import datetime
from pathlib import Path

from flood_detection_core.types import FloodPathsType


def get_pre_flood_paths_mapping(pre_flood_dir: Path, sites: list[str], num_temporal_length: int) -> FloodPathsType:
    site_path_mapping = {}
    for site in sites:
        tile_path_mapping = {}
        for tile_dir in pre_flood_dir.glob(f"{site}/*/"):
            tile_paths = []
            tile_dates = []
            for tile_path in tile_dir.glob("*.tif"):
                img_date = datetime.datetime.strptime(tile_path.stem.split("_")[-1].replace(".tif", ""), "%Y-%m-%d")
                tile_paths.append(tile_path)
                tile_dates.append(img_date)

            tile_paths, tile_dates = zip(*sorted(zip(tile_paths, tile_dates), key=lambda x: x[1]))
            tile_paths = list(tile_paths)[:num_temporal_length]
            tile_path_mapping[tile_dir.name] = tile_paths

        site_path_mapping[site] = tile_path_mapping

    return site_path_mapping


def get_sen1flood11_paths_mapping(pre_flood_paths: FloodPathsType, dir: Path) -> FloodPathsType:
    site_path_mapping = {}
    for site, tile_path_mapping in pre_flood_paths.items():
        site_path_mapping[site] = {}
        for tile_id in tile_path_mapping.keys():
            post_flood_path = list(dir.glob(f"{tile_id}_*.tif"))
            site_path_mapping[site][tile_id] = post_flood_path

    return site_path_mapping


def create_tile_pairs(pre_flood_paths: FloodPathsType, post_flood_paths: FloodPathsType) -> list[dict]:
    tile_pairs = []

    for site, tile_path_mapping in pre_flood_paths.items():
        for tile_id, tile_paths in tile_path_mapping.items():
            post_flood_path = post_flood_paths[site][tile_id]
            tile_pairs.append(
                {
                    "tile_id": tile_id,
                    "pre_flood_paths": [str(path.absolute()) for path in tile_paths],
                    "post_flood_path": str(post_flood_path[0].absolute()),
                    "site": site,
                }
            )

    return tile_pairs
