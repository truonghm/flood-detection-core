import csv
import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from rich import print
from sklearn.model_selection import train_test_split

from flood_detection_core.config import DataConfig
from flood_detection_core.data.constants import BOLIVIA_ALLOWED_TILES, HandLabeledSen1Flood11Sites


@dataclass
class SplitRatio:
    pretrain: float
    train: float
    validation: float
    test: float


def split_data(
    data_config: DataConfig,
) -> None:
    site_tiles_mapping = {}
    for site in HandLabeledSen1Flood11Sites:
        tile_dir_paths = list(data_config.data_dirs.pre_flood.glob(f"{site}/*/"))
        tiles = [tile_dir_path.name for tile_dir_path in tile_dir_paths]
        if site == "bolivia":
            tiles = [tile for tile in tiles if tile in BOLIVIA_ALLOWED_TILES]
        n_tiles = len(tiles)

        train_val_tiles = test_tiles = pretrain_tiles = tiles
        train_tiles, val_tiles = train_test_split(train_val_tiles, test_size=0.2, random_state=42)

        print(f"Site: {site}")
        print(
            "train:",
            len(train_tiles),
            "val:",
            len(val_tiles),
            "test:",
            len(test_tiles),
            "pretrain:",
            len(pretrain_tiles),
        )
        print(
            "train:",
            f"{len(train_tiles) / n_tiles:.2f}",
            "val:",
            f"{len(val_tiles) / n_tiles:.2f}",
            "test:",
            f"{len(test_tiles) / n_tiles:.2f}",
            "pretrain:",
            f"{len(pretrain_tiles) / n_tiles:.2f}",
        )

        site_tiles_mapping[site] = {
            "train": train_tiles,
            "val": val_tiles,
            "test": test_tiles,
            "pretrain": pretrain_tiles,
        }
        print("-" * 30)

    pre_flood_split = []
    post_flood_split = []

    for site, data_tiles_mapping in site_tiles_mapping.items():
        for dataset_type, tiles in data_tiles_mapping.items():
            for tile in tiles:
                pre_flood_paths = list(data_config.data_dirs.pre_flood.glob(f"{site}/{tile}/*.tif"))
                post_flood_paths = list(data_config.data_dirs.post_flood.glob(f"{tile.capitalize()}_*.tif"))
                ground_truth_paths = list(data_config.data_dirs.ground_truth.glob(f"{tile.capitalize()}_*.tif"))
                for path in pre_flood_paths:
                    pre_flood_split.append(
                        {
                            "dataset_type": dataset_type,
                            "site": site,
                            "tile": tile,
                            "path": path.as_posix(),
                        }
                    )
                for pf_path, gt_path in zip(post_flood_paths, ground_truth_paths):
                    post_flood_split.append(
                        {
                            "dataset_type": dataset_type,
                            "site": site,
                            "tile": tile,
                            "post_flood": pf_path.as_posix(),
                            "ground_truth": gt_path.as_posix(),
                        }
                    )

    print("Example of pre-flood split:")
    print(pre_flood_split[:2])
    print()
    print("Example of post-flood split:")
    print(post_flood_split[:2])

    with open(data_config.csv_files.pre_flood_split, "w") as f:
        writer = csv.DictWriter(f, fieldnames=pre_flood_split[0].keys())
        writer.writeheader()
        writer.writerows(pre_flood_split)
    print(f"Saved pre-flood split to `{data_config.csv_files.pre_flood_split}`")
    with open(data_config.csv_files.post_flood_split, "w") as f:
        writer = csv.DictWriter(f, fieldnames=post_flood_split[0].keys())
        writer.writeheader()
        writer.writerows(post_flood_split)
    print(f"Saved post-flood split to `{data_config.csv_files.post_flood_split}`")


def get_flood_event_tile_pairs(
    dataset_type: Literal["train", "val", "test", "pretrain"],
    pre_flood_split_csv_path: Path,
    post_flood_split_csv_path: Path,
) -> list[dict]:
    site_tile_path_mapping = {}

    with open(pre_flood_split_csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["dataset_type"] == dataset_type:
                if row["site"] not in site_tile_path_mapping:
                    site_tile_path_mapping[row["site"]] = {}
                if row["tile"] not in site_tile_path_mapping[row["site"]]:
                    site_tile_path_mapping[row["site"]][row["tile"]] = {}
                if "pre_flood" not in site_tile_path_mapping[row["site"]][row["tile"]]:
                    site_tile_path_mapping[row["site"]][row["tile"]]["pre_flood"] = []
                site_tile_path_mapping[row["site"]][row["tile"]]["pre_flood"].append(row["path"])

    with open(post_flood_split_csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["dataset_type"] == dataset_type:
                if row["site"] not in site_tile_path_mapping:
                    site_tile_path_mapping[row["site"]] = {}
                if row["tile"] not in site_tile_path_mapping[row["site"]]:
                    site_tile_path_mapping[row["site"]][row["tile"]] = {}
                if "post_flood" not in site_tile_path_mapping[row["site"]][row["tile"]]:
                    site_tile_path_mapping[row["site"]][row["tile"]]["post_flood"] = []
                site_tile_path_mapping[row["site"]][row["tile"]]["post_flood"].append(row["post_flood"])
                if "ground_truth" not in site_tile_path_mapping[row["site"]][row["tile"]]:
                    site_tile_path_mapping[row["site"]][row["tile"]]["ground_truth"] = []
                site_tile_path_mapping[row["site"]][row["tile"]]["ground_truth"].append(row["ground_truth"])

    tile_pairs = []
    for site, tile_path_mapping in site_tile_path_mapping.items():
        for tile, tile_paths in tile_path_mapping.items():
            pre_flood_paths = tile_paths["pre_flood"]
            dates = [datetime.datetime.strptime(Path(pth).stem.split("_")[-1], "%Y-%m-%d") for pth in pre_flood_paths]
            pre_flood_paths = [pth for _, pth in sorted(zip(dates, pre_flood_paths))]
            tile_pairs.append(
                {
                    "tile_id": tile,
                    "site": site,
                    "pre_flood_paths": pre_flood_paths,
                    "post_flood_path": tile_paths["post_flood"][0],
                    "ground_truth_path": tile_paths["ground_truth"][0],
                }
            )

    return tile_pairs
