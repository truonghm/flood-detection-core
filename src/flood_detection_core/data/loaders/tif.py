import datetime
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine
from rich import print
import magic
import re

from flood_detection_core.config import DataConfig
from flood_detection_core.data.metadata.sen1flood11 import PrefloodSiteMetadata, TileMetadata


@dataclass
class TifResult:
    date: datetime.datetime | None
    data: np.ndarray
    filepath: Path
    transform: Affine
    crs: CRS


def load_dual_band_tif(
    file_path: str,
    *,
    date: datetime.datetime | None = None,
    vv_norm_range: tuple[float, float] | None = None,
    vh_norm_range: tuple[float, float] | None = None,
) -> TifResult:
    with rasterio.open(file_path, "r") as src:
        data = src.read()
        if data.shape[0] == 2:
            data = np.transpose(data, (1, 2, 0))

        if vv_norm_range is not None:
            data[:, :, 0] = np.clip(
                (data[:, :, 0] - vv_norm_range[0]) / (vv_norm_range[1] - vv_norm_range[0]),
                0,
                1,
            )

        if vh_norm_range is not None:
            data[:, :, 1] = np.clip(
                (data[:, :, 1] - vh_norm_range[0]) / (vh_norm_range[1] - vh_norm_range[0]),
                0,
                1,
            )

        data_dict = {
            "date": date,
            "data": data,
            "filepath": file_path,
            "transform": src.transform,
            "crs": src.crs,
        }
        return TifResult(**data_dict)


def load_single_band_tif(
    file_path: str,
    norm_range: tuple[float, float] | None = None,
) -> TifResult:
    with rasterio.open(file_path, "r") as src:
        data = src.read(1)

        if norm_range is not None:
            data = np.clip((data - norm_range[0]) / (norm_range[1] - norm_range[0]), 0, 1)

        data_dict = {
            "date": None,
            "data": data,
            "filepath": file_path,
            "transform": src.transform,
            "crs": src.crs,
        }
        return TifResult(**data_dict)


class TifRawLoader:
    def __init__(self, data_config: DataConfig):
        self.data_config = data_config

    def get_tiles_and_dates_from_site(self, site_name: str) -> dict[str, list[datetime.datetime]]:
        result = {}
        pre_flood_metadata_path = self.data_config.gee.get_pre_flood_site_metadata_path(site_name)
        pre_flood_metadata = PrefloodSiteMetadata.from_json(pre_flood_metadata_path)
        for tile_name, tile_metadata in pre_flood_metadata.tiles.items():
            dates = tile_metadata.downloaded_dates
            result[tile_name] = dates
        return result

    def load_pre_flood_tile_data(
        self,
        tile_name: str,
        is_sorted: bool = True,
        dates: list[datetime.datetime] | None = None,
    ) -> list[TifResult]:
        """
        Loads the pre-flood data for a given tile.
        If is_sorted is True, the data is loaded in the order of the dates.
        If is_sorted is False, the data is loaded in the order of the file paths.
        If dates is provided, and is_sorted is True, the data is loaded in the order of the dates.

        Parameters
        ----------
        tile_name : str
            The name of the tile to load data for.
        is_sorted : bool, optional
            Whether to sort the data by date, by default True.
        dates : list[datetime.datetime], optional
            The dates to load the data for. If not provided, the data is loaded in the order of the file paths.

        Returns
        -------
        list[TifResult]
            The pre-flood data for the given tile.
        """
        site_name = tile_name.split("_")[0].lower()
        pre_flood_data = []
        if not is_sorted:
            file_paths = self.data_config.gee.get_pre_flood_tile_paths(site_name, tile_name)
            for file_path in file_paths:
                inferred_date = datetime.datetime.strptime(
                    file_path.stem.split("_")[-1].replace(".tif", ""), "%Y-%m-%d"
                )
                load_result = load_dual_band_tif(file_path, date=inferred_date)
                pre_flood_data.append(load_result)
            return pre_flood_data

        if not dates:
            pre_flood_metadata_path = self.data_config.gee.get_pre_flood_site_metadata_path(site_name)
            pre_flood_metadata = PrefloodSiteMetadata.from_json(pre_flood_metadata_path)
            dates = pre_flood_metadata.tiles[tile_name].downloaded_dates
        for idx, date in enumerate(dates):
            date_str = date.strftime("%Y-%m-%d")
            print(f"Loading pre-flood data for [bold green]{date_str}[/bold green]")
            reverse_idx = len(dates) - idx - 1
            expected_file_name = f"pre_flood_{reverse_idx + 1}_{date_str}.tif"
            expected_file_path = self.data_config.gee.pre_flood_dir / site_name / tile_name / expected_file_name
            if expected_file_path.exists():
                load_result = load_dual_band_tif(expected_file_path, date=date)
                pre_flood_data.append(load_result)
        return pre_flood_data

    def load_hl_post_flood_s1_tile_data(
        self,
        tile_name: str,
        vv_norm_range: tuple[float, float] | None = None,
        vh_norm_range: tuple[float, float] | None = None,
    ) -> TifResult:
        file_paths = self.data_config.hand_labeled_sen1flood11.get_post_flood_s1_file_paths(tile_name)
        post_flood_tile_metadata = TileMetadata.from_json(
            json_path=self.data_config.hand_labeled_sen1flood11.get_catalog_source_tile_path(tile_name),
            tile_id=tile_name,
        )
        post_flood_date = datetime.datetime.strptime(post_flood_tile_metadata.post_flood_date, "%Y-%m-%d")

        if len(file_paths) > 1:
            raise ValueError(f"Multiple post flood S1 files found for tile {tile_name}")

        post_flood_data = load_dual_band_tif(
            file_paths[0],
            date=post_flood_date,
            vv_norm_range=vv_norm_range,
            vh_norm_range=vh_norm_range,
        )
        return post_flood_data

    def load_hl_permanent_water_tile_data(self, tile_name: str) -> TifResult:
        file_paths = self.data_config.hand_labeled_sen1flood11.get_permanent_water_file_paths(tile_name)
        if len(file_paths) > 1:
            raise ValueError(f"Multiple permanent water files found for tile {tile_name}")

        permanent_water_data = load_single_band_tif(file_paths[0])
        return permanent_water_data

    def load_hl_ground_truth_tile_data(self, tile_name: str) -> TifResult:
        file_paths = self.data_config.hand_labeled_sen1flood11.get_ground_truth_file_paths(tile_name)
        if len(file_paths) > 1:
            raise ValueError(f"Multiple ground truth files found for tile {tile_name}")

        ground_truth_data = load_single_band_tif(file_paths[0])
        return ground_truth_data

    def load_pre_flood_site_data(self, site_name: str) -> dict[str, list[TifResult]]:
        tile_dates_mapping = self.get_tiles_and_dates_from_site(site_name)
        result = {}
        for tile_name, dates in tile_dates_mapping.items():
            result[tile_name] = self.load_pre_flood_tile_data(
                tile_name,
                is_sorted=True,
                dates=dates,
            )
        return result

    def load_hl_post_flood_s1_site_data(
        self,
        site_name: str,
        vv_norm_range: tuple[float, float] | None = None,
        vh_norm_range: tuple[float, float] | None = None,
    ) -> dict[str, TifResult]:
        file_paths = self.data_config.hand_labeled_sen1flood11.get_post_flood_s1_site_paths(site_name)
        result = {}
        for file_path in file_paths:
            tile_name = "_".join(file_path.stem.split("_")[:2])
            result[tile_name] = self.load_hl_post_flood_s1_tile_data(
                tile_name,
                vv_norm_range=vv_norm_range,
                vh_norm_range=vh_norm_range,
            )
        return result

    def load_hl_permanent_water_site_data(self, site_name: str) -> dict[str, TifResult]:
        file_paths = self.data_config.hand_labeled_sen1flood11.get_permanent_water_site_paths(site_name)
        result = {}
        for file_path in file_paths:
            tile_name = "_".join(file_path.stem.split("_")[:2])
            result[tile_name] = self.load_hl_permanent_water_tile_data(tile_name)
        return result

    def load_hl_ground_truth_site_data(self, site_name: str) -> dict[str, TifResult]:
        file_paths = self.data_config.hand_labeled_sen1flood11.get_ground_truth_site_paths(site_name)
        result = {}
        for file_path in file_paths:
            tile_name = "_".join(file_path.stem.split("_")[:2])
            result[tile_name] = self.load_hl_ground_truth_tile_data(tile_name)
        return result

    def load_hl_post_flood_s2_site_data(self, site_name: str) -> dict[str, TifResult]:
        file_paths = self.data_config.hand_labeled_sen1flood11.get_post_flood_s2_site_paths(site_name)
        result = {}
        for file_path in file_paths:
            tile_name = "_".join(file_path.stem.split("_")[:2])
            result[tile_name] = self.load_hl_post_flood_s2_tile_data(tile_name)
        return result


def get_image_size(file_path: Path) -> tuple[int, int]:
    metadata = magic.from_file(file_path)
    height = re.search(r"height=(\d+)", metadata).group(1)
    width = re.search(r"width=(\d+)", metadata).group(1)
    return int(height), int(width)
