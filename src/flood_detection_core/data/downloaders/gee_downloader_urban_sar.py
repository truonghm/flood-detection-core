import datetime
import os
from pathlib import Path
from typing import Any, Literal

import ee
import geemap
import numpy as np
import requests
from rich import print
from rich.progress import Progress

from flood_detection_core.config import (
    DataConfig,
    GEEDownloadConfig,
)
from flood_detection_core.data.constants import UrbanSARSites
from flood_detection_core.data.metadata.base import BasePerSiteTilesMetadata, SiteMetadata


class GEEDownloader:
    def __init__(
        self,
        download_config: GEEDownloadConfig,
        data_config: DataConfig,
        sites_metadata: dict[str, SiteMetadata],
        tiles_metadata_loader: BasePerSiteTilesMetadata,
    ):
        self.download_config = download_config
        self.data_config = data_config
        self.sites_metadata = sites_metadata
        self.tiles_metadata_loader = tiles_metadata_loader

    @staticmethod
    def get_s1_collection(
        aoi: ee.Geometry,
        start_date: str,
        end_date: str,
        orbit_pass: str,
        relative_orbit: int,
    ) -> ee.ImageCollection:
        """Get Sentinel-1 collection filtered by area, date, and exact orbital parameters.

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
        vv_clipped_range: tuple[float, float] | None = None,
        vh_clipped_range: tuple[float, float] | None = None,
    ) -> ee.Image:
        """Preprocess Sentinel-1 image according to CLVAE paper specifications.

        Args:
            image: Earth Engine image

        Returns:
            ee.Image: Preprocessed image with values in [0, 1]
        """
        if vv_clipped_range is None or vh_clipped_range is None:
            return image

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
    def mosaic_by_unique_date(collection: ee.ImageCollection) -> ee.ImageCollection:
        """Mosaic images that share the same calendar date so that each date yields a single image covering the AOI
        as completely as possible.

        Returns an image collection where each image corresponds to one unique
        date and carries a normalized `system:time_start` (00:00 UTC of that day)
        and a `date` property in YYYY-MM-dd format.
        """

        def add_date(img: ee.Image) -> ee.Image:
            date_str = ee.Date(img.get("system:time_start")).format("YYYY-MM-dd")
            return img.set("date", date_str)

        with_date = collection.map(add_date)
        unique_dates = ee.List(with_date.aggregate_array("date").distinct())

        def mosaic_for_date(d):
            d = ee.String(d)
            day_coll = with_date.filter(ee.Filter.eq("date", d))
            first = ee.Image(day_coll.first())
            time_start = ee.Date.parse("YYYY-MM-dd", d).millis()
            mosaicked = day_coll.mosaic()
            return (
                mosaicked.set("date", d)
                .set("system:time_start", time_start)
                .copyProperties(first, ["relativeOrbitNumber_start"])
            )

        return ee.ImageCollection(unique_dates.map(mosaic_for_date))

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
                    # "dimensions": dimensions,
                    "scale": scale,
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
        vv_clipped_range: tuple[float, float] | None = None,
        vh_clipped_range: tuple[float, float] | None = None,
        num_pre_images: int = 8,
        days_before_flood_max: int = 120,
        days_before_flood_min: int = 5,
        download_format: Literal["numpy", "geotiff"] = "numpy",
        scale: int = 10,
        max_dimension: int = 512,
        enable_debug: bool = False,
    ) -> None:
        site_dir = output_dir / site_name
        site_dir.mkdir(parents=True, exist_ok=True)

        site_metadata = self.sites_metadata[site_name]

        tiles_metadata = self.tiles_metadata_loader.from_json(
            self.data_config.csv_files.path_mapping,
            site_name,
        )

        # post_flood_date = site_metadata.post_flood_date
        orbit_pass = site_metadata.orbit_pass
        relative_orbit = site_metadata.relative_orbit

        print(f"Downloading pre-flood data for {site_name}:")
        # print(f"  Post-flood date: {post_flood_date}")
        print(f"  Orbit pass: {orbit_pass}")
        print(f"  Relative orbit: {relative_orbit}")
        print(f"  Found {len(tiles_metadata.tiles)} individual tiles to process")

        # post_flood_dt = datetime.datetime.strptime(post_flood_date, "%Y-%m-%d")
        # pre_flood_end = post_flood_dt - datetime.timedelta(days=days_before_flood_min)  # Day before flood
        # pre_flood_start = post_flood_dt - datetime.timedelta(days=days_before_flood_max)  # 4 months before

        # pre_flood_start_str = pre_flood_start.strftime("%Y-%m-%d")
        # pre_flood_end_str = pre_flood_end.strftime("%Y-%m-%d")

        # print(f"  Pre-flood period: {pre_flood_start_str} to {pre_flood_end_str}")

        successful_tiles = 0
        with Progress() as pb:
            tiles_task = pb.add_task(f"[cyan]{site_name}", total=len(tiles_metadata))
            for tile_id, tile_data in tiles_metadata.tiles.items():
                # pb.update(tiles_task, description=f"[cyan]Processing tile {tile_id}")
                # print(f"\n  Processing tile {tile_id}...")
                ext = "npy" if download_format == "numpy" else "tif"
                tile_dir = site_dir / tile_id
                if tile_dir.exists() and len(list(tile_dir.glob(f"*{ext}"))) >= num_pre_images:
                    # print(f"    Skipping tile {tile_id} (already downloaded)")
                    pb.advance(tiles_task)
                    continue
                tile_dir.mkdir(parents=True, exist_ok=True)

                try:
                    post_flood_date = tile_data.post_flood_date
                    post_flood_dt = datetime.datetime.strptime(post_flood_date, "%Y-%m-%d")
                    pre_flood_end = post_flood_dt - datetime.timedelta(days=days_before_flood_min)  # Day before flood
                    pre_flood_start = post_flood_dt - datetime.timedelta(days=days_before_flood_max)  # 4 months before

                    # pre_flood_start_str = pre_flood_start.strftime("%Y-%m-%d")
                    pre_flood_end_str = pre_flood_end.strftime("%Y-%m-%d")

                    # print(f"  Post-flood date: {post_flood_date}")
                    # print(f"  Pre-flood period: {pre_flood_start_str} to {pre_flood_end_str}")

                    bbox = tile_data.bbox
                    # print(f"    Tile bbox: {bbox}")
                    # print("    Tile dimensions should be 512x512")

                    aoi = ee.Geometry.Rectangle(bbox)

                    # Build collection and progressively extend window backward until
                    # we have at least `num_pre_images` unique dates after mosaicking.
                    search_start_dt = pre_flood_start
                    earliest_s1_dt = datetime.datetime(2014, 10, 1)
                    extension_step_days = days_before_flood_max

                    def build_collection(start_dt: datetime.datetime) -> ee.ImageCollection:
                        return self.get_s1_collection(
                            aoi,
                            start_dt.strftime("%Y-%m-%d"),
                            pre_flood_end_str,
                            orbit_pass,
                            relative_orbit,
                        )

                    while True:
                        pre_flood_collection = build_collection(search_start_dt)
                        mosaicked_by_date = self.mosaic_by_unique_date(pre_flood_collection)
                        unique_dates_count = mosaicked_by_date.size().getInfo()

                        if unique_dates_count >= num_pre_images:
                            break

                        if search_start_dt <= earliest_s1_dt:
                            break

                        prev_start = search_start_dt
                        search_start_dt = max(
                            earliest_s1_dt,
                            prev_start - datetime.timedelta(days=extension_step_days),
                        )
                        print(
                            "    Not enough unique dates "
                            f"({unique_dates_count}/{num_pre_images}). Extending start date from "
                            f"{prev_start.strftime('%Y-%m-%d')} to "
                            f"{search_start_dt.strftime('%Y-%m-%d')}"
                        )

                    mosaicked_by_date = self.mosaic_by_unique_date(build_collection(search_start_dt))
                    collection_size = mosaicked_by_date.size().getInfo()
                    if collection_size == 0:
                        print(f"    Warning: No pre-flood images found for tile {tile_id}")
                        continue

                    print(
                        "    Found "
                        f"{collection_size} unique pre-flood dates (after mosaicking by date) "
                        "with matching orbital parameters"
                    )

                    pre_flood_images = mosaicked_by_date.sort("system:time_start", False).limit(num_pre_images)
                    actual_count = min(num_pre_images, collection_size)

                    preprocessed_images = pre_flood_images.map(
                        lambda x: self.preprocess_s1_image(
                            x,
                            vv_clipped_range=vv_clipped_range,
                            vh_clipped_range=vh_clipped_range,
                        )
                    )

                    # print(f"    Downloading {actual_count} pre-flood images (unique dates) for tile {tile_id}...")
                    pre_flood_list = preprocessed_images.toList(actual_count)

                    downloaded_dates = []
                    images_task = pb.add_task(f"[green]  {tile_id}", total=actual_count)
                    for i in range(actual_count):
                        pre_image = ee.Image(pre_flood_list.get(i))
                        pre_date = ee.Date(pre_image.get("system:time_start")).format("YYYY-MM-dd").getInfo()

                        filename = tile_dir / f"pre_flood_{i + 1}_{pre_date}.{ext}"
                        # pb.update(images_task, description=f"[green]  Image {i + 1}/{actual_count}: {pre_date}")
                        if filename.exists():
                            # print(f"      Skipping pre-flood image {i + 1}: {pre_date} (already downloaded)")
                            pb.advance(images_task)
                            continue

                        if download_format == "numpy":
                            success = self.to_numpy(
                                pre_image,
                                aoi,
                                filename,
                                scale=scale,
                                dimensions=max_dimension,
                                enable_debug=enable_debug,
                            )
                        else:
                            success = self.to_geotiff(
                                pre_image,
                                aoi,
                                filename,
                                scale=scale,
                                dimensions=max_dimension,
                                enable_debug=enable_debug,
                            )

                        if success:
                            downloaded_dates.append(pre_date)
                            # print(f"      Downloaded pre-flood image {i + 1}/{actual_count}: {pre_date}")
                        else:
                            print(f"      Failed to download pre-flood image {i + 1}")
                        pb.advance(images_task)
                    pb.remove_task(images_task)  # Clear the inner progress bar
                    # print(f"    Successfully downloaded {len(downloaded_dates)} images for tile {tile_id}")
                    successful_tiles += 1

                except Exception as e:
                    print(f"    Error downloading data for tile {tile_id}: {e}")
                    if enable_debug:
                        raise e
                finally:
                    pb.advance(tiles_task)  # Advance the outer progress bar

        print(f"\n  Site summary: {successful_tiles}/{len(tiles_metadata)} tiles processed successfully")

    def download_all(
        self,
        output_dir: Path,
        vv_clipped_range: tuple[float, float] | None = None,
        vh_clipped_range: tuple[float, float] | None = None,
        scale: int = 10,
        num_pre_images: int = 8,
        days_before_flood_max: int = 120,
        days_before_flood_min: int = 5,
        download_format: Literal["numpy", "geotiff"] = "numpy",
        max_dimension: int = 4000,
        enable_debug: bool = False,
    ) -> None:
        for site_name in UrbanSARSites:
            self.download_site(
                site_name=site_name,
                output_dir=output_dir,
                vv_clipped_range=vv_clipped_range,
                vh_clipped_range=vh_clipped_range,
                scale=scale,
                days_before_flood_max=days_before_flood_max,
                days_before_flood_min=days_before_flood_min,
                num_pre_images=num_pre_images,
                download_format=download_format,
                max_dimension=max_dimension,
                enable_debug=enable_debug,
            )

    def __call__(self) -> Any:
        if self.download_config.preprocessing is None:
            vv_clipped_range = None
            vh_clipped_range = None
        else:
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
                output_dir=self.data_config.data_dirs.pre_flood,
                vv_clipped_range=vv_clipped_range,
                vh_clipped_range=vh_clipped_range,
                scale=self.download_config.output.scale,
                num_pre_images=self.download_config.flood_sites.num_pre_images,
                days_before_flood_max=self.download_config.flood_sites.days_before_flood_max,
                days_before_flood_min=self.download_config.flood_sites.days_before_flood_min,
                download_format=self.download_config.output.format,
                max_dimension=self.download_config.output.max_dimension,
                enable_debug=self.download_config.debug,
            )

        return self.download_site(
            site_name=self.download_config.target,
            output_dir=self.data_config.data_dirs.pre_flood,
            vv_clipped_range=vv_clipped_range,
            vh_clipped_range=vh_clipped_range,
            scale=self.download_config.output.scale,
            num_pre_images=self.download_config.flood_sites.num_pre_images,
            days_before_flood_max=self.download_config.flood_sites.days_before_flood_max,
            days_before_flood_min=self.download_config.flood_sites.days_before_flood_min,
            download_format=self.download_config.output.format,
            max_dimension=self.download_config.output.max_dimension,
            enable_debug=self.download_config.debug,
        )


if __name__ == "__main__":
    from flood_detection_core.config import UrbanSARGeeDownloadConfig
    from flood_detection_core.data.metadata.urban_sar import PerSiteTilesMetadata, load_urban_sar_metadata
    from flood_detection_core.utils import authenticate_gee

    authenticate_gee(os.getenv("GCP_PROJECT_ID"))

    download_config = UrbanSARGeeDownloadConfig.from_yaml("./flood-detection-core/yamls/gee_urban_sar.yaml")
    data_config = DataConfig.from_yaml("./flood-detection-core/yamls/data_urban_sar.yaml")

    sites_metadata = load_urban_sar_metadata(data_config.site_metadata)
    downloader = SitePrefloodDataDownloader(
        download_config, data_config, sites_metadata=sites_metadata, tiles_metadata_loader=PerSiteTilesMetadata
    )
    downloader()
