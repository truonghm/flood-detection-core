import datetime
import json
import os
import random
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
    Sen1Flood11SiteMetadata,
    load_sen1flood11_metadata,
)
from flood_detection_core.data.utils import create_patches_from_array


class GEEDownloader:
    def __init__(self, download_config: GEEDownloadConfig, data_config: DataConfig):
        self.download_config = download_config
        self.data_config = data_config
        self.sen1flood11_metadata = load_sen1flood11_metadata(data_config.sen1flood11.metadata)

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
            raise ValueError(
                f"Invalid date format {start_date} or {end_date}. Expected YYYY-MM-DD"
            )

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
        create_patches: bool = False,
        patch_size: int = 16,
        patch_stride: int = 16,
        enable_debug: bool = False,
    ) -> None:
        site_dir = output_dir / site_name
        site_dir.mkdir(parents=True, exist_ok=True)

        site_metadata = self.sen1flood11_metadata[site_name]

        tiles_metadata = PerSiteTilesMetadata.from_json(
            self.data_config.sen1flood11.hand_labeled_catalog_source,
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
        pre_flood_end = post_flood_dt - datetime.timedelta(
            days=days_before_flood_min
        )  # Day before flood
        pre_flood_start = post_flood_dt - datetime.timedelta(
            days=days_before_flood_max
        )  # 4 months before

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

                print(
                    f"    Found {collection_size} pre-flood images with matching orbital parameters"
                )

                pre_flood_images = pre_flood_collection.sort(
                    "system:time_start", False
                ).limit(num_pre_images)
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

                print(
                    f"    Downloading {actual_count} pre-flood images for tile {tile_id}..."
                )
                pre_flood_list = preprocessed_images.toList(actual_count)

                downloaded_dates = []
                for i in range(actual_count):
                    pre_image = ee.Image(pre_flood_list.get(i))
                    pre_date = (
                        ee.Date(pre_image.get("system:time_start"))
                        .format("YYYY-MM-dd")
                        .getInfo()
                    )

                    if download_format == "numpy":
                        filename = os.path.join(
                            tile_dir, f"pre_flood_{i + 1}_{pre_date}.npy"
                        )
                        success = self.to_numpy(
                            pre_image,
                            aoi,
                            filename,
                            dimensions=max_dimension,
                            enable_debug=enable_debug,
                        )
                    else:
                        filename = os.path.join(
                            tile_dir, f"pre_flood_{i + 1}_{pre_date}.tif"
                        )
                        success = self.to_geotiff(
                            pre_image,
                            aoi,
                            filename,
                            dimensions=max_dimension,
                            enable_debug=enable_debug,
                        )

                    if success:
                        downloaded_dates.append(pre_date)
                        print(
                            f"      Downloaded pre-flood image {i + 1}/{actual_count}: {pre_date}"
                        )

                        if create_patches and download_format == "numpy":
                            try:
                                data = np.load(filename)
                                print(
                                    f"      Image shape: {data.shape} (should be close to 512x512x2)"
                                )
                                patches_array = create_patches_from_array(
                                    data, patch_size, patch_stride
                                )
                                patch_filename = filename.replace(
                                    ".npy", "_patches.npy"
                                )
                                np.save(patch_filename, patches_array)
                                print(f"      Created {patches_array.shape[0]} patches")
                            except Exception as e:
                                print(f"      Error creating patches: {e}")
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

                print(
                    f"    Successfully downloaded {len(downloaded_dates)} images for tile {tile_id}"
                )
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

        print(
            f"\n  Site summary: {successful_tiles}/{len(tiles_metadata)} tiles processed successfully"
        )
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
        create_patches: bool = False,
        patch_size: int = 16,
        patch_stride: int = 16,
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
                create_patches=create_patches,
                patch_size=patch_size,
                patch_stride=patch_stride,
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
                output_dir=self.data_config.gee.data_dir,
                vv_clipped_range=vv_clipped_range,
                vh_clipped_range=vh_clipped_range,
                num_pre_images=self.download_config.flood_sites.num_pre_images,
                days_before_flood_max=self.download_config.flood_sites.days_before_flood_max,
                days_before_flood_min=self.download_config.flood_sites.days_before_flood_min,
                download_format=self.download_config.output.format,
                max_dimension=self.download_config.output.max_dimension,
                create_patches=self.download_config.patches.create_patches,
                patch_size=self.download_config.patches.patch_size,
                patch_stride=self.download_config.patches.patch_stride,
                enable_debug=self.download_config.debug,
            )

        return self.download_site(
            site_name=self.download_config.target,
            output_dir=self.data_config.gee.data_dir,
            vv_clipped_range=vv_clipped_range,
            vh_clipped_range=vh_clipped_range,
            num_pre_images=self.download_config.flood_sites.num_pre_images,
            days_before_flood_max=self.download_config.flood_sites.days_before_flood_max,
            days_before_flood_min=self.download_config.flood_sites.days_before_flood_min,
            download_format=self.download_config.output.format,
            max_dimension=self.download_config.output.max_dimension,
            create_patches=self.download_config.patches.create_patches,
            patch_size=self.download_config.patches.patch_size,
            patch_stride=self.download_config.patches.patch_stride,
            enable_debug=self.download_config.debug,
        )


class PreTrainDataDownloader(GEEDownloader):
    @staticmethod
    def get_random_patches_for_pretraining(
        sites_metadata: dict[str, Sen1Flood11SiteMetadata], num_patches: int
    ) -> list[tuple[ee.Geometry, str, str, int]]:
        """
        Generate random patches from flood sites for pre-training

        Args:
            sites_metadata: Dictionary of site metadata
            num_patches: Number of random patches to generate

        Returns:
            List of (aoi, site_name, orbit_pass, relative_orbit) tuples
        """
        patches = []

        for i in range(num_patches):
            # Randomly select a site
            site_name = random.choice(list(sites_metadata.keys()))
            site_info = sites_metadata[site_name]

            # Get site bounds
            min_lon, min_lat, max_lon, max_lat = site_info.bbox

            # Generate random patch within site bounds (use smaller patch for variation)
            patch_size_deg = 0.01  # Approximately 1km at equator

            if (max_lon - min_lon) < patch_size_deg or (
                max_lat - min_lat
            ) < patch_size_deg:
                # If site is too small, use the entire site
                patch_aoi = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])
            else:
                # Generate random patch
                patch_min_lon = random.uniform(min_lon, max_lon - patch_size_deg)
                patch_min_lat = random.uniform(min_lat, max_lat - patch_size_deg)
                patch_max_lon = patch_min_lon + patch_size_deg
                patch_max_lat = patch_min_lat + patch_size_deg

                patch_aoi = ee.Geometry.Rectangle(
                    [patch_min_lon, patch_min_lat, patch_max_lon, patch_max_lat]
                )

            patches.append(
                (patch_aoi, site_name, site_info.orbit_pass, site_info.relative_orbit)
            )

        return patches

    def download(
        self,
        output_dir: Path,
        vv_clipped_range: tuple[float, float],
        vh_clipped_range: tuple[float, float],
        num_patches: int = 100,
        num_images: int = 4,
        download_format: Literal["numpy", "geotiff"] = "numpy",
        max_dimension: int = 512,
        create_patches: bool = False,
        patch_size: int = 16,
        patch_stride: int = 16,
        enable_debug: bool = False,
    ) -> None:
        pretrain_dir = output_dir / "pretrain"
        pretrain_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"Downloading pre-training data: {num_patches} patches with {num_images} images each"
        )

        # Generate random patches
        patches = self.get_random_patches_for_pretraining(
            self.sen1flood11_metadata,
            num_patches,
        )

        successful_downloads = 0

        for i, (patch_aoi, site_name, orbit_pass, relative_orbit) in enumerate(patches):
            print(f"Processing patch {i + 1}/{num_patches} from {site_name}")

            # Use time period well before any floods (2018-2019 was flood period for most sites)
            # Use 2017 data for pre-training to avoid contamination
            start_date = "2017-01-01"
            end_date = "2017-12-31"

            # Get pre-flood collection
            try:
                pre_flood_collection = self.get_s1_collection(
                    patch_aoi, start_date, end_date, orbit_pass, relative_orbit
                )

                collection_size = pre_flood_collection.size().getInfo()
                if collection_size < num_images:
                    print(
                        f"  Skipping patch {i + 1}: only {collection_size} images available, need {num_images}"
                    )
                    continue

                # Get the most recent images
                pre_flood_images = pre_flood_collection.sort(
                    "system:time_start", False
                ).limit(num_images)
                preprocessed_images = pre_flood_images.map(
                    lambda x: self.preprocess_s1_image(
                        x,
                        vv_clipped_range=vv_clipped_range,
                        vh_clipped_range=vh_clipped_range,
                    )
                )

                # Download images for this patch
                patch_dir = os.path.join(pretrain_dir, f"patch_{i + 1:03d}")
                os.makedirs(patch_dir, exist_ok=True)

                pre_flood_list = preprocessed_images.toList(num_images)
                patch_success = True

                for j in range(num_images):
                    pre_image = ee.Image(pre_flood_list.get(j))
                    pre_date = (
                        ee.Date(pre_image.get("system:time_start"))
                        .format("YYYY-MM-dd")
                        .getInfo()
                    )

                    if download_format == "numpy":
                        filename = os.path.join(
                            patch_dir, f"pre_flood_{j + 1}_{pre_date}.npy"
                        )
                        success = self.to_numpy(
                            image=pre_image,
                            aoi=patch_aoi,
                            filename=filename,
                            dimensions=max_dimension,
                            enable_debug=enable_debug,
                        )
                    else:
                        filename = os.path.join(
                            patch_dir, f"pre_flood_{j + 1}_{pre_date}.tif"
                        )
                        success = self.to_geotiff(
                            image=pre_image,
                            aoi=patch_aoi,
                            filename=filename,
                            dimensions=max_dimension,
                            enable_debug=enable_debug,
                        )

                    if not success:
                        patch_success = False
                        break

                    # Create patches if requested
                    if create_patches and download_format == "numpy":
                        try:
                            data = np.load(filename)
                            patches_array = create_patches_from_array(
                                data, patch_size, patch_stride
                            )
                            patch_filename = filename.replace(".npy", "_patches.npy")
                            np.save(patch_filename, patches_array)
                            print(
                                f"    Created {patches_array.shape[0]} patches from {filename}"
                            )
                        except Exception as e:
                            print(f"    Error creating patches from {filename}: {e}")

                if patch_success:
                    successful_downloads += 1
                    print(f"  Successfully downloaded patch {i + 1}")
                else:
                    print(f"  Failed to download patch {i + 1}")

            except Exception as e:
                print(f"  Error processing patch {i + 1}: {e}")
                if enable_debug:
                    raise e

        print(
            f"Pre-training data download completed: {successful_downloads}/{num_patches} patches successful"
        )


if __name__ == "__main__":
    from flood_detection_core.utils import authenticate_gee

    authenticate_gee(os.getenv("GCP_PROJECT_ID"))

    download_config = Sen1Flood11GeeDownloadConfig(
        _yaml_file="./flood-detection-core/src/flood_detection_core/yamls/gee_download_config.yaml"
    )
    data_config = DataConfig(
        _yaml_file="./flood-detection-core/src/flood_detection_core/yamls/data_config.yaml"
    )
    downloader = SitePrefloodDataDownloader(download_config, data_config)
    downloader()
