import datetime
import json
import os
from pathlib import Path
from typing import Any, Literal

import ee
import geemap
import numpy as np
import requests
from rich import print

from flood_detection_core.config import (
    DataConfig,
    GEEDownloadConfig,
    Sen1Flood11GeeDownloadConfig,
)
from flood_detection_core.data.metadata.sen1flood11 import (
    PerSiteTilesMetadata,
    load_sen1flood11_metadata,
)


class GEEDownloader:
    def __init__(self, download_config: GEEDownloadConfig, data_config: DataConfig):
        self.download_config = download_config
        self.data_config = data_config
        self.sen1flood11_metadata = load_sen1flood11_metadata(data_config.hand_labeled_sen1flood11.metadata)

    @staticmethod
    def get_s1_collection(
        aoi: ee.Geometry,
        start_date: str,
        end_date: str,
        orbit_pass: str,
        relative_orbit: int,
    ) -> ee.ImageCollection:
        """
        Get Sentinel-1 collection filtered by area, date, and exact orbital parameters

        Args:
            aoi: Earth Engine geometry defining the area of interest
            start_date: Start date for collection (YYYY-MM-DD)
            end_date: End date for collection (YYYY-MM-DD)
            orbit_pass: 'ASCENDING' or 'DESCENDING'
            relative_orbit: Specific relative orbit number

        Returns:
            ee.ImageCollection: Filtered Sentinel-1 collection
        """
        # Validate date format
        try:
            datetime.datetime.strptime(start_date, "%Y-%m-%d")
            datetime.datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Invalid date format {start_date} or {end_date}. Expected YYYY-MM-DD")

        collection = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(aoi)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.eq("orbitProperties_pass", orbit_pass))
            .filter(ee.Filter.eq("relativeOrbitNumber_start", relative_orbit))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
            .select(["VV", "VH"])
        )

        return collection

    @staticmethod
    def preprocess_s1_image(
        image: ee.Image,
        vv_clipped_range: tuple[float, float],
        vh_clipped_range: tuple[float, float],
    ) -> ee.Image:
        """
        Preprocess Sentinel-1 image according to CLVAE paper specifications

        Args:
            image: Earth Engine image

        Returns:
            ee.Image: Preprocessed image with values in [0, 1]
        """
        vv = image.select("VV")
        vh = image.select("VH")

        vv_clipped = vv.clamp(vv_clipped_range[0], vv_clipped_range[1])
        vh_clipped = vh.clamp(vh_clipped_range[0], vh_clipped_range[1])

        # see documentation of ee.Image.unitScale: https://developers.google.com/earth-engine/apidocs/ee-image-unitscale
        vv_norm = vv_clipped.unitScale(vv_clipped_range[0], vv_clipped_range[1])
        vh_norm = vh_clipped.unitScale(vh_clipped_range[0], vh_clipped_range[1])

        return (
            ee.Image.cat([vv_norm, vh_norm])
            .rename(["VV", "VH"])
            .copyProperties(image, ["system:time_start", "relativeOrbitNumber_start"])
        )

    @staticmethod
    def to_numpy(
        image: ee.Image,
        aoi: ee.Geometry,
        filename: str,
        scale: int = 10,
        dimensions: int = 512,
        enable_debug: bool = False,
    ) -> bool:
        try:
            data = geemap.ee_to_numpy(
                image,
                region=aoi,
                scale=scale,
            )

            np.save(filename, data)
            return True
        except Exception as e:
            print(f"Error saving {filename}: {e}")
            if enable_debug:
                raise e
            return False

    @staticmethod
    def to_geotiff(
        image: ee.Image,
        aoi: ee.Geometry,
        filename: str,
        scale: int = 10,
        dimensions: int = 512,
        enable_debug: bool = False,
    ) -> bool:
        try:
            url = image.getDownloadURL(
                {
                    "region": aoi,
                    "format": "GEO_TIFF",
                    "dimensions": dimensions,
                }
            )

            response = requests.get(url)
            with open(filename, "wb") as f:
                f.write(response.content)

            return True
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
            if enable_debug:
                raise e
            return False


class SitePrefloodDataDownloader(GEEDownloader):
    def download_site(
        self,
        site_name: str,
        output_dir: Path,
        vv_clipped_range: tuple[float, float],
        vh_clipped_range: tuple[float, float],
        num_pre_images: int = 8,
        days_before_flood_max: int = 120,
        days_before_flood_min: int = 5,
        download_format: Literal["numpy", "geotiff"] = "numpy",
        max_dimension: int = 512,
        enable_debug: bool = False,
    ) -> None:
        site_dir = output_dir / site_name
        site_dir.mkdir(parents=True, exist_ok=True)

        site_metadata = self.sen1flood11_metadata[site_name]

        tiles_metadata = PerSiteTilesMetadata.from_json(
            self.data_config.hand_labeled_sen1flood11.catalog_source,
            site_name,
        )

        post_flood_date = site_metadata.post_flood_date
        orbit_pass = site_metadata.orbit_pass
        relative_orbit = site_metadata.relative_orbit

        print(f"Downloading pre-flood data for {site_name}:")
        print(f"  Post-flood date: {post_flood_date}")
        print(f"  Orbit pass: {orbit_pass}")
        print(f"  Relative orbit: {relative_orbit}")
        print(f"  Found {len(tiles_metadata.tiles)} individual tiles to process")

        post_flood_dt = datetime.datetime.strptime(post_flood_date, "%Y-%m-%d")
        pre_flood_end = post_flood_dt - datetime.timedelta(days=days_before_flood_min)  # Day before flood
        pre_flood_start = post_flood_dt - datetime.timedelta(days=days_before_flood_max)  # 4 months before

        pre_flood_start_str = pre_flood_start.strftime("%Y-%m-%d")
        pre_flood_end_str = pre_flood_end.strftime("%Y-%m-%d")

        print(f"  Pre-flood period: {pre_flood_start_str} to {pre_flood_end_str}")

        all_tile_metadata = {}
        successful_tiles = 0

        for tile_id, tile_data in tiles_metadata.tiles.items():
            print(f"\n  Processing tile {tile_id}...")

            try:
                bbox = tile_data.bbox
                tile_post_flood_date = tile_data.post_flood_date
                print(f"    Tile bbox: {bbox}")
                print("    Tile dimensions should be 512x512")

                aoi = ee.Geometry.Rectangle(bbox)

                pre_flood_collection = self.get_s1_collection(
                    aoi,
                    pre_flood_start_str,
                    pre_flood_end_str,
                    orbit_pass,
                    relative_orbit,
                )

                collection_size = pre_flood_collection.size().getInfo()
                if collection_size == 0:
                    print(f"    Warning: No pre-flood images found for tile {tile_id}")
                    continue

                print(f"    Found {collection_size} pre-flood images with matching orbital parameters")

                pre_flood_images = pre_flood_collection.sort("system:time_start", False).limit(num_pre_images)
                actual_count = min(num_pre_images, collection_size)

                preprocessed_images = pre_flood_images.map(
                    lambda x: self.preprocess_s1_image(
                        x,
                        vv_clipped_range=vv_clipped_range,
                        vh_clipped_range=vh_clipped_range,
                    )
                )

                tile_dir = site_dir / tile_id
                tile_dir.mkdir(parents=True, exist_ok=True)

                print(f"    Downloading {actual_count} pre-flood images for tile {tile_id}...")
                pre_flood_list = preprocessed_images.toList(actual_count)

                downloaded_dates = []
                for i in range(actual_count):
                    pre_image = ee.Image(pre_flood_list.get(i))
                    pre_date = ee.Date(pre_image.get("system:time_start")).format("YYYY-MM-dd").getInfo()

                    if download_format == "numpy":
                        filename = os.path.join(tile_dir, f"pre_flood_{i + 1}_{pre_date}.npy")
                        success = self.to_numpy(
                            pre_image,
                            aoi,
                            filename,
                            dimensions=max_dimension,
                            enable_debug=enable_debug,
                        )
                    else:
                        filename = os.path.join(tile_dir, f"pre_flood_{i + 1}_{pre_date}.tif")
                        success = self.to_geotiff(
                            pre_image,
                            aoi,
                            filename,
                            dimensions=max_dimension,
                            enable_debug=enable_debug,
                        )

                    if success:
                        downloaded_dates.append(pre_date)
                        print(f"      Downloaded pre-flood image {i + 1}/{actual_count}: {pre_date}")
                    else:
                        print(f"      Failed to download pre-flood image {i + 1}")

                tile_metadata = {
                    "tile_id": tile_id,
                    "site_name": site_name,
                    "bbox": bbox,
                    "post_flood_date": tile_post_flood_date,
                    "orbit_pass": orbit_pass,
                    "relative_orbit": relative_orbit,
                    "pre_flood_period": {
                        "start": pre_flood_start_str,
                        "end": pre_flood_end_str,
                    },
                    "downloaded_dates": downloaded_dates,
                    "preprocessing": {
                        "vv_clip_range": vv_clipped_range,
                        "vh_clip_range": vh_clipped_range,
                        "normalization": "[0, 1]",
                    },
                }

                all_tile_metadata[tile_id] = tile_metadata

                metadata_file = os.path.join(tile_dir, "metadata.json")
                with open(metadata_file, "w") as f:
                    json.dump(tile_metadata, f, indent=2)

                print(f"    Successfully downloaded {len(downloaded_dates)} images for tile {tile_id}")
                successful_tiles += 1

            except Exception as e:
                print(f"    Error downloading data for tile {tile_id}: {e}")
                if enable_debug:
                    raise e

        overall_metadata = {
            "site_name": site_name,
            "post_flood_date": post_flood_date,
            "orbit_pass": orbit_pass,
            "relative_orbit": relative_orbit,
            "total_tiles": len(tiles_metadata),
            "successful_tiles": successful_tiles,
            "pre_flood_period": {
                "start": pre_flood_start_str,
                "end": pre_flood_end_str,
            },
            "tiles": all_tile_metadata,
            "preprocessing": {
                "vv_clip_range": vv_clipped_range,
                "vh_clip_range": vh_clipped_range,
                "normalization": "[0, 1]",
            },
        }

        overall_metadata_file = os.path.join(site_dir, "overall_metadata.json")
        with open(overall_metadata_file, "w") as f:
            json.dump(overall_metadata, f, indent=2)

        print(f"\n  Site summary: {successful_tiles}/{len(tiles_metadata)} tiles processed successfully")
        print(f"  Overall metadata saved to: {overall_metadata_file}")

    def download_all(
        self,
        output_dir: Path,
        vv_clipped_range: tuple[float, float],
        vh_clipped_range: tuple[float, float],
        num_pre_images: int = 8,
        days_before_flood_max: int = 120,
        days_before_flood_min: int = 5,
        download_format: Literal["numpy", "geotiff"] = "numpy",
        max_dimension: int = 4000,
        enable_debug: bool = False,
    ) -> None:
        for site_name in self.sen1flood11_metadata.keys():
            self.download_site(
                site_name=site_name,
                output_dir=output_dir,
                vv_clipped_range=vv_clipped_range,
                vh_clipped_range=vh_clipped_range,
                days_before_flood_max=days_before_flood_max,
                days_before_flood_min=days_before_flood_min,
                num_pre_images=num_pre_images,
                download_format=download_format,
                max_dimension=max_dimension,
                enable_debug=enable_debug,
            )

    def __call__(self) -> Any:
        vv_clipped_range = (
            self.download_config.preprocessing.vv_clip_lower_bound,
            self.download_config.preprocessing.vv_clip_upper_bound,
        )
        vh_clipped_range = (
            self.download_config.preprocessing.vh_clip_lower_bound,
            self.download_config.preprocessing.vh_clip_upper_bound,
        )
        if self.download_config.target == "all":
            return self.download_all(
                output_dir=self.data_config.gee.pre_flood_dir,
                vv_clipped_range=vv_clipped_range,
                vh_clipped_range=vh_clipped_range,
                num_pre_images=self.download_config.flood_sites.num_pre_images,
                days_before_flood_max=self.download_config.flood_sites.days_before_flood_max,
                days_before_flood_min=self.download_config.flood_sites.days_before_flood_min,
                download_format=self.download_config.output.format,
                max_dimension=self.download_config.output.max_dimension,
                enable_debug=self.download_config.debug,
            )

        return self.download_site(
            site_name=self.download_config.target,
            output_dir=self.data_config.gee.pre_flood_dir,
            vv_clipped_range=vv_clipped_range,
            vh_clipped_range=vh_clipped_range,
            num_pre_images=self.download_config.flood_sites.num_pre_images,
            days_before_flood_max=self.download_config.flood_sites.days_before_flood_max,
            days_before_flood_min=self.download_config.flood_sites.days_before_flood_min,
            download_format=self.download_config.output.format,
            max_dimension=self.download_config.output.max_dimension,
            enable_debug=self.download_config.debug,
        )


if __name__ == "__main__":
    from flood_detection_core.utils import authenticate_gee

    authenticate_gee(os.getenv("GCP_PROJECT_ID"))

    download_config = Sen1Flood11GeeDownloadConfig(
        _yaml_file="./flood-detection-core/src/flood_detection_core/yamls/gee_download_config.yaml"
    )
    data_config = DataConfig(_yaml_file="./flood-detection-core/src/flood_detection_core/yamls/data_config.yaml")
    downloader = SitePrefloodDataDownloader(download_config, data_config)
    downloader()
