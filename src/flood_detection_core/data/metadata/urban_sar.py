import json
from pathlib import Path
from typing import Literal

import duckdb as ddb

from .base import BasePerSiteTilesMetadata, SiteMetadata, TileMetadata


class UrbanSARSiteMetadata(SiteMetadata):
    site_name: str
    orbit_pass: Literal["ASCENDING", "DESCENDING"]
    relative_orbit: int
    start_dt: str
    end_dt: str


def load_urban_sar_metadata(path: Path) -> dict[str, UrbanSARSiteMetadata]:
    if path.is_dir():
        json_path = path / "overall_metadata.json"
    elif path.is_file() and path.suffix == ".json":
        json_path = path
    else:
        raise ValueError(f"Invalid path: {path}")

    with open(json_path) as f:
        metadata = json.load(f)

    sites = {k: UrbanSARSiteMetadata(**v) for k, v in metadata.items()}
    return sites


class UrbanSARTileMetadata(TileMetadata):
    tile_id: str
    bbox: list[float]
    post_flood_date: str
    flood_type: Literal["NF", "FU", "FO"]

    @classmethod
    def from_json(cls, json_path: Path) -> "UrbanSARTileMetadata":
        with open(json_path) as f:
            tile_data = json.load(f)

        return UrbanSARTileMetadata(**tile_data)


class PerSiteTilesMetadata(BasePerSiteTilesMetadata):
    site: str | Literal["all"]
    tiles: dict[str, UrbanSARTileMetadata]

    @classmethod
    def from_json(
        cls,
        path_mapping_csv: Path,
        site_name: str | Literal["all"] | None = None,
        flood_type: Literal["NF", "FU", "FO"] | None = None,
    ) -> "PerSiteTilesMetadata":
        query = f"""
        select site, tile_id, metadata_path, flood_type, date
        from read_csv_auto('{path_mapping_csv.as_posix()}')
        where 1=1
        """
        if site_name is not None:
            query += f" and site = '{site_name}'"
        else:
            site_name = "all"
        if flood_type is not None:
            query += f" and flood_type = '{flood_type}'"
        df = ddb.sql(query).to_df()
        tiles_metadata: dict[str, TileMetadata] = {}
        for _, row in df.iterrows():
            tiles_metadata[row["tile_id"]] = UrbanSARTileMetadata.from_json(Path(row["metadata_path"]))
        return PerSiteTilesMetadata(site=site_name, tiles=tiles_metadata)


if __name__ == "__main__":
    metadata = PerSiteTilesMetadata.from_json(
        Path("/media/truonghm/EXT4_SSD_500GB/urban_sar_floods/sampled_path_mapping.csv")
    )
