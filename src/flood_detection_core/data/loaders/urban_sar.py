import datetime
import json
import warnings
from pathlib import Path
from typing import Literal

import duckdb as ddb
import numpy as np
import pandas as pd
import rasterio
from rich import print
from rich.progress import Progress

from .base import RawDataset


def extract_from_filename(filename: str, suffix: str) -> tuple[str, str, str, datetime.datetime]:
    date_location_id = filename[: -(len(suffix) + 1)]
    parts = date_location_id.rsplit("_ID_", 1)
    tile_date_str = parts[0][:8]
    location = parts[0][9:]
    idx = parts[1]
    tile_date = datetime.datetime.strptime(tile_date_str, "%Y%m%d")

    return date_location_id, location, idx, tile_date


class UrbanSARDataset(RawDataset):
    """
    Process
    ----------

    1. Prepare `slc_file_names.txt` or `slc_site_product_mapping`
    2. Fetch sites metadata (if not exist)
    3. Prepare the dataset (if not done)
    4. Create path mapping
    5. Ready
    """

    __default_yaml_path = Path("./flood-detection-core/yamls/data_urban_sar.yaml")
    __catalogue_odata_url = "https://catalogue.dataspace.copernicus.eu/odata/v1"
    __site_info_csv_name = "site_info.csv"
    __slc_file_names_txt_name = "slc_file_names.txt"
    __pre_dir_name = "PRE"
    __post_dir_name = "POST"
    __gt_dir_name = "GT"
    __sar_dir_name = "SAR"
    __nf_gt = f"01_NF/{__gt_dir_name}"
    __nf_sar = f"01_NF/{__sar_dir_name}"
    __fo_gt = f"02_FO/{__gt_dir_name}"
    __fo_sar = f"02_FO/{__sar_dir_name}"
    __fu_gt = f"03_FU/{__gt_dir_name}"
    __fu_sar = f"03_FU/{__sar_dir_name}"
    __nf_pre = f"01_NF/{__pre_dir_name}"
    __fo_pre = f"02_FO/{__pre_dir_name}"
    __fu_pre = f"03_FU/{__pre_dir_name}"
    __nf_post = f"01_NF/{__post_dir_name}"
    __fo_post = f"02_FO/{__post_dir_name}"
    __fu_post = f"03_FU/{__post_dir_name}"
    __metadata_dir_name = "metadata"
    __path_mapping_csv_name = "original_path_mapping.csv"

    def __init__(
        self,
        yaml_path: str | Path | None = None,
        force_create_path_mapping: bool = False,
        force_extract_bands: bool = False,
        force_fetch_metadata: bool = False,
    ):
        self.force_create_path_mapping = force_create_path_mapping
        self.force_extract_bands = force_extract_bands
        self.force_fetch_metadata = force_fetch_metadata
        self.root_dir = self._load_root_dir()
        self.data_config = self.load_data_config(yaml_path if yaml_path else self.__default_yaml_path)
        self.nf_sar_dir = self.root_dir / self.__nf_sar
        self.fo_sar_dir = self.root_dir / self.__fo_sar
        self.fu_sar_dir = self.root_dir / self.__fu_sar
        self.nf_gt_dir = self.root_dir / self.__nf_gt
        self.fo_gt_dir = self.root_dir / self.__fo_gt
        self.fu_gt_dir = self.root_dir / self.__fu_gt

        self.nf_pre_dir = self.root_dir / self.__nf_pre
        self.fo_pre_dir = self.root_dir / self.__fo_pre
        self.fu_pre_dir = self.root_dir / self.__fu_pre
        self.nf_post_dir = self.root_dir / self.__nf_post
        self.fo_post_dir = self.root_dir / self.__fo_post
        self.fu_post_dir = self.root_dir / self.__fu_post
        self.nf_metadata_dir = self.root_dir / "01_NF" / self.__metadata_dir_name
        self.fo_metadata_dir = self.root_dir / "02_FO" / self.__metadata_dir_name
        self.fu_metadata_dir = self.root_dir / "03_FU" / self.__metadata_dir_name

        self.metadata_dir = self.root_dir / self.__metadata_dir_name
        self.__path_mapping_csv_file = self.root_dir / self.__path_mapping_csv_name

        if not self.__path_mapping_csv_file.exists() or force_create_path_mapping:
            print("No path mapping file found, generating path mapping...")
            # all dir must exist
            print("No pre/post flood directory found, preparing...")
            site_info_path = self.root_dir / self.__site_info_csv_name
            if not site_info_path.exists() or force_fetch_metadata:
                print("No site info file found, fetching sites metadata...")
                self.fetch_sites_metadata()
            self.prepare()
            self.map_all_paths()

    def _load_root_dir(self) -> Path:
        import os

        root_dir = os.getenv("URBAN_SAR_ROOT_DIR")
        if root_dir is None:
            raise ValueError("URBAN_SAR_ROOT_DIR environment variable is not set")
        return Path(root_dir)

    def prepare(self) -> None:
        site_info_path = self.root_dir / self.__site_info_csv_name
        site_info_df = pd.read_csv(site_info_path, parse_dates=["SiteBeginningDateTime", "SiteEndingDateTime"])

        print("Found site info: ", site_info_df.shape)

        site_metadata = {}

        start: pd.Timestamp
        end: pd.Timestamp
        for site_name, orbit_pass, rel_orbit, start, end in site_info_df.itertuples(index=False):
            start_dt = start.to_pydatetime().replace(tzinfo=None)
            end_dt = end.to_pydatetime().replace(tzinfo=None)

            site_metadata[site_name] = {
                "site_name": site_name,
                "orbit_pass": orbit_pass.upper(),
                "relative_orbit": rel_orbit,
                "start_dt": start_dt.strftime("%Y-%m-%d"),
                "end_dt": end_dt.strftime("%Y-%m-%d"),
            }

            path_type_mapping = [
                (self.nf_sar_dir, "NF"),
                (self.fu_sar_dir, "FU"),
                (self.fo_sar_dir, "FO"),
            ]

            with Progress() as pb:
                outer = pb.add_task("Preparing site", total=len(path_type_mapping))
                inner = pb.add_task("Preparing tile", total=0)

                for sar_dir, flood_type in path_type_mapping:
                    print(f"Processing: {site_name} - {sar_dir} - {flood_type}")
                    tile_paths = list(sar_dir.glob(f"*_{site_name}_*.tif"))
                    pb.update(task_id=inner, total=len(tile_paths), completed=0)
                    # print(f"Found {len(tile_paths)} tiles")
                    if len(tile_paths) == 0:
                        print(f"[red]No tile paths found for {site_name} - {sar_dir} - {flood_type}[/red]")
                        continue
                    for tile_path in tile_paths:
                        self.process_tile(
                            tile_path=tile_path,
                            is_sar=True,
                            flood_type=flood_type,
                            site_start_dt=start_dt,
                            site_end_dt=end_dt,
                        )
                        pb.update(task_id=inner, advance=1)
                    pb.update(task_id=outer, advance=1)
        with open(self.root_dir / "overall_metadata.json", "w") as f:
            json.dump(site_metadata, f)

    def process_tile(
        self,
        tile_path: Path,
        is_sar: bool,
        flood_type: Literal["NF", "FU", "FO"],
        site_start_dt: datetime.datetime,
        site_end_dt: datetime.datetime,
    ) -> None:
        suffix = self.__sar_dir_name if is_sar else self.__gt_dir_name
        date_location_id, location, idx, tile_date = extract_from_filename(tile_path.stem, suffix)
        sar_file_name = f"{date_location_id}_{self.__sar_dir_name}.tif"
        sar_path = tile_path.parent / sar_file_name
        catalog_dir = sar_path.parent.parent / self.__metadata_dir_name
        if not catalog_dir.exists():
            print(f"Creating catalog directory: {catalog_dir}")
            catalog_dir.mkdir(parents=True, exist_ok=True)
        pre_dir = sar_path.parent.parent / self.__pre_dir_name
        if not pre_dir.exists():
            print(f"Creating pre directory: {pre_dir}")
            pre_dir.mkdir(parents=True, exist_ok=True)
        post_dir = sar_path.parent.parent / self.__post_dir_name
        if not post_dir.exists():
            print(f"Creating post directory: {post_dir}")
            post_dir.mkdir(parents=True, exist_ok=True)

        post_flood_date = min(tile_date, site_start_dt, site_end_dt)

        pre_event_path = pre_dir / f"{date_location_id}_{self.__pre_dir_name}.tif"
        post_event_path = post_dir / f"{date_location_id}_{self.__post_dir_name}.tif"
        metadata_path = catalog_dir / f"{date_location_id}_{self.__metadata_dir_name}.json"
        to_extract_pre = False
        to_extract_post = False
        to_extract_metadata = False
        if self.force_extract_bands:
            to_extract_pre = True
            to_extract_post = True
            to_extract_metadata = True
        else:
            if not pre_event_path.exists():
                to_extract_pre = True
            if not post_event_path.exists():
                to_extract_post = True
            if not metadata_path.exists():
                to_extract_metadata = True

        if any([to_extract_post, to_extract_pre, to_extract_metadata]):
            with rasterio.open(sar_path, "r") as src:
                sar_data = src.read()
            # Reorder bands: VV first (index 0), then VH (index 1)
            pre_event = sar_data[[5, 4], ...]  # [VV, VH] from bands [6, 5]
            post_event = sar_data[[7, 6], ...]  # [VV, VH] from bands [8, 7]
            bbox = src.bounds

            # calculate post flood % of nan data
            post_flood_nan_ratio = np.sum(np.isnan(post_event)) / post_event.size

            tile_metadata = {
                "tile_id": f"{location}_ID_{idx}",
                "bbox": bbox,
                "post_flood_date": post_flood_date.strftime("%Y-%m-%d"),
                "flood_type": flood_type,
                "post_flood_nan_ratio": post_flood_nan_ratio.round(2),
            }

            if to_extract_pre:
                with rasterio.open(
                    pre_event_path,
                    "w",
                    driver="GTiff",
                    height=pre_event.shape[1],
                    width=pre_event.shape[2],
                    count=pre_event.shape[0],
                    dtype=pre_event.dtype,
                    crs=src.crs,
                    transform=src.transform,
                ) as dst:
                    dst.write(pre_event)

            # Write post-event as TIFF file
            if to_extract_post:
                with rasterio.open(
                    post_event_path,
                    "w",
                    driver="GTiff",
                    height=post_event.shape[1],
                    width=post_event.shape[2],
                    count=post_event.shape[0],
                    dtype=post_event.dtype,
                    crs=src.crs,
                    transform=src.transform,
                ) as dst:
                    dst.write(post_event)

            if to_extract_metadata:
                with open(metadata_path, "w") as f:
                    json.dump(tile_metadata, f)

    def fetch_sites_metadata(
        self,
        slc_site_product_mapping: dict[str : list[str]] | None = None,
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        slc_site_product_mapping: dict[str: list[str]] | None
            A dictionary of site name and list of original SLC product names.
            If not available, will load from a csv file named "slc_file_names.txt" in the root directory.

        Returns
        -------
        site_info_df: pd.DataFrame
            A dataframe of site name, orbit pass, relative orbit, start date, and end date

        Raises
        ------
        ValueError: If the result may contain duplication per ParentProductName

        Examples
        --------

        >>> raw_site_product_mapping = {
        ...     "Sentinel-1_SLC_data/20170830_Houston": [
        ...         "S1A_IW_SLC__1SDV_20170824T122248_20170824T122318_018065_01E54E_5C27",
        ...         "S1B_IW_SLC__1SDV_20170818T122205_20170818T122235_006994_00C525_0F49",
        ...         "S1B_IW_SLC__1SDV_20170830T122203_20170830T122233_007169_00CA2C_C92C",
        ...     ],
        ... }
        """
        import requests

        if not slc_site_product_mapping:
            print("No slc_site_product_mapping found, loading from slc_file_names.txt")
            with open(self.root_dir / self.__slc_file_names_txt_name) as f:
                data = f.read()
            slc_site_product_mapping = {}
            current_key = None
            for row in data.split("\n"):
                if not row.strip():
                    continue
                elif row.strip().startswith("Sentinel-1_SLC_data/"):
                    current_key = row.strip()
                    if current_key not in slc_site_product_mapping:
                        slc_site_product_mapping[current_key] = []
                else:
                    slc_site_product_mapping[current_key].append(row.strip())

        results_mapping = {}

        errors = []
        with Progress() as pb:
            outer = pb.add_task("Fetching sites metadata", total=len(slc_site_product_mapping))
            inner = pb.add_task("Fetching product metadata", total=0)

            for k, v in slc_site_product_mapping.items():
                pb.update(task_id=inner, total=len(v), completed=0)
                for filename in v:
                    site_name = k.split("/")[-1][9:]
                    # print(site_name)
                    search_query = f"{self.__catalogue_odata_url}/Bursts?$filter=ParentProductName eq '{filename}.SAFE'"
                    response = requests.get(search_query).json()
                    response_val = response.get("value")
                    if not response_val:
                        errors.append({"site": site_name, "fn": filename, "error": "No value found"})
                    result = pd.DataFrame.from_dict(response.get("value"))
                    result["SiteName"] = site_name
                    results_mapping[filename] = result
                    pb.update(task_id=inner, advance=1)
                pb.update(task_id=outer, advance=1)

        df = pd.concat(list(results_mapping.values()))
        site_info_df = df[
            [
                "SiteName",
                "ParentProductName",
                "OrbitDirection",
                "RelativeOrbitNumber",
                "BeginningDateTime",
                "EndingDateTime",
            ]
        ].copy()
        site_info_df["BeginningDateTime"] = pd.to_datetime(site_info_df["BeginningDateTime"])
        site_info_df["EndingDateTime"] = pd.to_datetime(site_info_df["EndingDateTime"])

        site_info_df["SiteBeginningDateTime"] = site_info_df.groupby("SiteName")["BeginningDateTime"].transform("min")
        site_info_df["SiteEndingDateTime"] = site_info_df.groupby("SiteName")["EndingDateTime"].transform("max")

        site_info_df = site_info_df[
            [
                "SiteName",
                # "ParentProductName",
                "OrbitDirection",
                "RelativeOrbitNumber",
                "SiteBeginningDateTime",
                "SiteEndingDateTime",
            ]
        ].drop_duplicates()

        if site_info_df["SiteName"].nunique() != site_info_df.shape[0]:
            # raise ValueError("Result may contain duplication per SiteName")
            warnings.warn("Result may contain duplication per SiteName")

        site_info_df.to_csv(self.root_dir / self.__site_info_csv_name, index=False)
        return site_info_df

    def map_all_paths(self) -> None:
        nf_pre_paths = self.nf_pre_dir.glob("*.tif")
        fo_pre_paths = self.fo_pre_dir.glob("*.tif")
        fu_pre_paths = self.fu_pre_dir.glob("*.tif")
        nf_gt_paths = self.nf_gt_dir.glob("*.tif")
        fo_gt_paths = self.fo_gt_dir.glob("*.tif")
        fu_gt_paths = self.fu_gt_dir.glob("*.tif")
        nf_post_paths = self.nf_post_dir.glob("*.tif")
        fo_post_paths = self.fo_post_dir.glob("*.tif")
        fu_post_paths = self.fu_post_dir.glob("*.tif")
        nf_md_paths = self.nf_metadata_dir.glob("*.json")
        fo_md_paths = self.fo_metadata_dir.glob("*.json")
        fu_md_paths = self.fu_metadata_dir.glob("*.json")

        flood_type_mapping = [
            ("NF", nf_pre_paths, "PRE"),
            ("NF", nf_post_paths, "POST"),
            ("NF", nf_gt_paths, "GT"),
            ("FO", fo_pre_paths, "PRE"),
            ("FO", fo_post_paths, "POST"),
            ("FO", fo_gt_paths, "GT"),
            ("FU", fu_pre_paths, "PRE"),
            ("FU", fu_post_paths, "POST"),
            ("FU", fu_gt_paths, "GT"),
            ("NF", nf_md_paths, "metadata"),
            ("FO", fo_md_paths, "metadata"),
            ("FU", fu_md_paths, "metadata"),
        ]
        path_mapping = []

        for flood_type, paths, img_type in flood_type_mapping:
            print(f"Processing: {flood_type} - {img_type}")
            for pth in paths:
                post_flood_nan_ratio = None
                if img_type == "metadata":
                    with open(pth) as f:
                        metadata = json.load(f)
                    post_flood_nan_ratio = metadata.get("post_flood_nan_ratio", None)
                _, location, idx, tile_date = extract_from_filename(pth.stem, img_type)
                path_mapping.append(
                    {
                        "flood_type": flood_type,
                        "path": pth.as_posix(),
                        "img_type": img_type,
                        "date": tile_date.strftime("%Y-%m-%d"),
                        "site": location,
                        "tile_id": idx,
                        "post_flood_nan_ratio": post_flood_nan_ratio,
                    }
                )

        path_mapping_df = pd.DataFrame(path_mapping)
        post_df = path_mapping_df[path_mapping_df["img_type"] == "POST"].rename(columns={"path": "post_flood_path"})
        pre_df = path_mapping_df[path_mapping_df["img_type"] == "PRE"].rename(columns={"path": "pre_flood_path"})
        gt_df = path_mapping_df[path_mapping_df["img_type"] == "GT"].rename(columns={"path": "gt_path"})
        md_df = path_mapping_df[path_mapping_df["img_type"] == "metadata"].rename(columns={"path": "metadata_path"})
        pre_df = pre_df.drop(columns=["img_type", "post_flood_nan_ratio"])
        post_df = post_df.drop(columns=["img_type", "post_flood_nan_ratio"])
        gt_df = gt_df.drop(columns=["img_type", "post_flood_nan_ratio"])
        md_df = md_df.drop(columns=["img_type"])
        # post_df.to_csv(self.root_dir / "post_df.csv", index=False)
        # pre_df.to_csv(self.root_dir / "pre_df.csv", index=False)
        # gt_df.to_csv(self.root_dir / "gt_df.csv", index=False)
        df = (
            post_df.merge(pre_df, on=["flood_type", "site", "tile_id", "date"], how="inner")
            .merge(gt_df, on=["flood_type", "site", "tile_id", "date"], how="inner")
            .merge(md_df, on=["flood_type", "site", "tile_id", "date"], how="inner")
        )
        df = df[
            [
                "flood_type",
                "post_flood_path",
                "pre_flood_path",
                "gt_path",
                "metadata_path",
                "date",
                "site",
                "tile_id",
                "post_flood_nan_ratio",
            ]
        ]
        df.to_csv(self.root_dir / self.__path_mapping_csv_name, index=False)

    def sample(self, sample_pct: float = 0.1) -> None:
        df = pd.read_csv(self.root_dir / self.__path_mapping_csv_name)
        df = df.sort_values(by="date")
        df = df.groupby(["site", "tile_id"]).first().reset_index()
        df["tile_id"] = df.apply(
            lambda row: f"{row['site']}_ID_{row['tile_id']}_type_{row['flood_type']}",
            axis=1,
        )
        df_sampled = (
            df.groupby("flood_type")
            .apply(lambda x: x.sample(frac=sample_pct), include_groups=False)
            .reset_index()
            .drop(columns=["level_1"])
        )
        df_sampled.to_csv(self.root_dir / f"{self.__path_mapping_csv_name}_sampled_{sample_pct}.csv", index=False)

    def arrange_data(self) -> None:
        pass

    def query_paths(self, query: str) -> pd.DataFrame:
        if "path_mapping" not in query:
            raise ValueError("Query must contain `path_mapping`")
        paths_csv = (self.root_dir / self.__path_mapping_csv_name).as_posix()
        query = query.replace("path_mapping", f"'{paths_csv}'")
        return ddb.query(query).to_df()


if __name__ == "__main__":
    ds = UrbanSARDataset(
        force_create_path_mapping=True,
        force_extract_bands=True,
        force_fetch_metadata=False,
    )
