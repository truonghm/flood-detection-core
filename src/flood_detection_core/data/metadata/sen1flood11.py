import datetime
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, field_validator
from rich import print

from .base import BasePerSiteTilesMetadata, SiteMetadata, TileMetadata
from flood_detection_core.data.constants import (
    CatalogSubdirs,
    EquivalentNameMapping,
    HandLabeledSen1Flood11Sites,
)


def calculate_bbox_from_coordinates(coordinates: list[list[float]]) -> list[float]:
    """Calculate bounding box [min_lon, min_lat, max_lon, max_lat] from polygon coordinates.

    Args:
        coordinates: List of [longitude, latitude] pairs from polygon geometry

    Returns:
        list[float]: [min_longitude, min_latitude, max_longitude, max_latitude]
    """
    if not coordinates:
        return []

    lons = [coord[0] for coord in coordinates]
    lats = [coord[1] for coord in coordinates]

    return [min(lons), min(lats), max(lons), max(lats)]


class PreFloodPeriod(BaseModel):
    start: str
    end: str


class PrefloodPreprocessing(BaseModel):
    vv_clip_range: list[float]
    vh_clip_range: list[float]
    normalization: str


class PrefloodTileMetadata(BaseModel):
    tile_id: str
    site_name: str
    bbox: list[float]
    post_flood_date: str
    orbit_pass: Literal["ASCENDING", "DESCENDING"]
    relative_orbit: int
    pre_flood_period: PreFloodPeriod
    downloaded_dates: list[str] | list[datetime.datetime]
    preprocessing: PrefloodPreprocessing

    @field_validator("downloaded_dates", mode="before")
    def validate_downloaded_dates(cls, v: list[str] | list[datetime.datetime]) -> list[datetime.datetime]:
        if isinstance(v, list):
            return sorted([datetime.datetime.strptime(date, "%Y-%m-%d") for date in v])
        return v


class PrefloodSiteMetadata(BaseModel):
    site_name: str
    post_flood_date: str
    orbit_pass: Literal["ASCENDING", "DESCENDING"]
    relative_orbit: int
    total_tiles: int
    successful_tiles: int
    pre_flood_period: PreFloodPeriod
    tiles: dict[str, PrefloodTileMetadata]
    preprocessing: PrefloodPreprocessing

    @classmethod
    def from_json(cls, json_path: Path) -> "PrefloodSiteMetadata":
        with open(json_path) as f:
            metadata = json.load(f)
        return cls(**metadata)


class Sen1Flood11SiteMetadata(SiteMetadata):
    site_name: str
    iso_cc: str = ""
    post_flood_date: str
    orbit_pass: Literal["ASCENDING", "DESCENDING"]
    relative_orbit: int
    train_tiles: int = 0
    val_tiles: int = 0
    vh_threshold: float = -20.0
    bbox: list[float] = []


def load_sen1flood11_metadata(path: Path, is_hand_labeled: bool = True) -> dict[str, Sen1Flood11SiteMetadata]:
    if path.is_dir():
        json_path = path / "Sen1Floods11_Metadata.geojson"
    elif path.is_file() and path.suffix == ".geojson":
        json_path = path
    else:
        raise ValueError(f"Invalid path: {path}")

    with open(json_path) as f:
        metadata = json.load(f)

    sites = {}

    for feature in metadata["features"]:
        props = feature["properties"]
        site_name = props["location"].lower().replace("-", "_")
        if is_hand_labeled and site_name not in HandLabeledSen1Flood11Sites and site_name not in EquivalentNameMapping:
            continue
        if site_name in EquivalentNameMapping:
            site_name = EquivalentNameMapping[site_name]
        s1_date = props["s1_date"].replace("/", "-")
        coordinates = feature.get("geometry", {}).get("coordinates", [])
        if len(coordinates) != 1:
            print(f"[red]Warning:[/red] Invalid coordinates for site {site_name}")
            continue

        polygon_coords = coordinates[0]
        bbox = calculate_bbox_from_coordinates(polygon_coords)

        site_metadata = {
            "site_name": site_name,
            "iso_cc": props.get("ISO_CC", ""),
            "post_flood_date": s1_date,
            "orbit_pass": props["orbit"],
            "relative_orbit": int(props["rel_orbit_num"]),
            "train_tiles": props.get("train_tile", 0),
            "val_tiles": props.get("val_chip", 0),
            "vh_threshold": props.get("VH_thresh", -20.0),
            "bbox": bbox,
        }
        sites[site_name] = Sen1Flood11SiteMetadata(**site_metadata)

    print(sites.keys())
    return sites


class Sen1Flood11TileMetadata(TileMetadata):
    tile_id: str
    bbox: list[float]
    country: str
    # tile_id: str
    datetime: str
    geometry: dict[str, Any]
    post_flood_date: str

    @classmethod
    def from_json(cls, json_path: Path, tile_id: str | None = None) -> "TileMetadata":
        if tile_id is None:
            tile_id = "_".join(json_path.stem.split("_")[:2])

        with open(json_path) as f:
            tile_data = json.load(f)

        properties = tile_data.get("properties", {})
        bbox = tile_data.get("bbox", [])

        if len(bbox) != 4:
            print(f"Warning: Invalid bbox for tile {tile_id}")
            return None

        metadata = {
            "tile_id": tile_id,
            "bbox": bbox,  # [min_lon, min_lat, max_lon, max_lat]
            "country": properties.get("country", ""),
            # "tile_id": properties.get("tile_id", ""),
            "datetime": properties.get("datetime", ""),
            "geometry": tile_data.get("geometry", {}),
            "post_flood_date": properties.get("datetime", "").split("T")[0] if properties.get("datetime") else "",
        }
        return TileMetadata(**metadata)


class PerSiteTilesMetadata(BasePerSiteTilesMetadata):
    @staticmethod
    def load_site_tiles_metadata(catalog_path: Path, site_name: str) -> dict[str, TileMetadata]:
        tiles_metadata: dict[str, TileMetadata] = {}

        pattern = f"{site_name.title()}_*"
        tile_dirs = list(catalog_path.glob(pattern))

        print(f"Found {len(tile_dirs)} tile directories for {site_name}")

        for tile_dir in tile_dirs:
            if not tile_dir.is_dir():
                continue

            tile_id = "_".join(tile_dir.name.split("_")[:2])
            json_file = tile_dir / f"{tile_id}.json"

            if not json_file.exists():
                print(f"Warning: JSON file not found for tile {tile_id}")
                continue

            try:
                tile_metadata = TileMetadata.from_json(json_file, tile_id)
                tiles_metadata[tile_id] = tile_metadata
                print(f"  Loaded metadata for tile {tile_id}: bbox={tile_metadata.bbox}")

            except Exception as e:
                print(f"Error loading metadata for tile {tile_id}: {e}")
                continue

        return tiles_metadata

    @classmethod
    def from_json(cls, catalog_path: str, site_name: str | None = None) -> "PerSiteTilesMetadata":
        if not Path(catalog_path).is_dir():
            raise ValueError(f"Catalog path {catalog_path} is not a directory")

        if Path(catalog_path).name not in CatalogSubdirs:
            raise ValueError(
                f"Catalog path {catalog_path} is not a valid catalog subdirectory, must be one of {CatalogSubdirs}"
            )

        if site_name is None or site_name == "all":
            # get all sites
            sites = [subdir.name for subdir in Path(catalog_path).iterdir() if subdir.is_dir()]
            all_metadata: dict[str, TileMetadata] = {}
            for site in sites:
                all_metadata.update(
                    PerSiteTilesMetadata.load_site_tiles_metadata(Path(catalog_path), site.split("_")[0].lower())
                )
            return PerSiteTilesMetadata(site="all", tiles=all_metadata)
        else:
            all_metadata = PerSiteTilesMetadata.load_site_tiles_metadata(Path(catalog_path), site_name)
            return PerSiteTilesMetadata(site=site_name, tiles=all_metadata)


if __name__ == "__main__":
    test_coordinates = [
        [-65.317559857533908, -15.959244925391689],
        [-65.158254847813453, -15.959244879070081],
        [-64.998949860147647, -15.959244896460911],
        [-64.361638085153402, -11.391041883289134],
        [-64.521034883785887, -11.391041903057278],
        [-65.317559857533908, -15.959244925391689],
    ]

    bbox = calculate_bbox_from_coordinates(test_coordinates)
    print(f"Test polygon coordinates converted to bbox: {bbox}")
    print("Expected format: [min_lon, min_lat, max_lon, max_lat]")

    try:
        metadata = load_sen1flood11_metadata(Path("./data/sen1flood11/v1.1/Sen1Floods11_Metadata.geojson"))
        print(f"\nLoaded metadata for {len(metadata)} sites")

        for site_name, site_data in list(metadata.items())[:1]:
            print(f"\nExample site '{site_name}':")
            print(f"  Bbox: {site_data.bbox}")
            print(f"  Orbit: {site_data.orbit_pass}")
            print(f"  Relative orbit: {site_data.relative_orbit}")
            break

    except FileNotFoundError:
        print("Metadata file not found - this is expected if running from different directory")

    site = "all"
    try:
        tiles_metadata = PerSiteTilesMetadata.from_json(
            "./data/sen1flood11/v1.1/catalog/sen1floods11_hand_labeled_source", site
        )
        print(f"\nLoaded tiles metadata for {len(tiles_metadata)} tiles")

        for tile_id, tile_data in list(tiles_metadata.tiles.items())[:1]:
            print(f"\nExample tile '{tile_id}':")
            print(f"  Bbox: {tile_data.bbox}")
            print(f"  Country: {tile_data.country}")
            break

    except (FileNotFoundError, ValueError) as e:
        print(f"Tiles metadata loading failed - this is expected if running from different directory: {e}")

    try:
        site_name = "bolivia"
        tile_name = "Bolivia_103757"
        site_metadata = PrefloodSiteMetadata.from_json(f"./data/ee/{site_name}/overall_metadata.json")
        print(f"\nExample pre-flood metadata for {site_name}:")
        if tile_name in site_metadata.tiles:
            print(f"Tile {tile_name} downloaded dates: {site_metadata.tiles[tile_name].downloaded_dates}")
        else:
            print(f"Available tiles: {list(site_metadata.tiles.keys())[:3]}...")
    except FileNotFoundError:
        print("Pre-flood metadata file not found - this is expected if data hasn't been downloaded yet")
