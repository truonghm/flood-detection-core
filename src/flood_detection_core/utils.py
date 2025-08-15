import json
from pathlib import Path

import ee
from rich import print

from flood_detection_core.config import DataConfig


def authenticate_gee(project: str):
    """Authenticate to Google Earth Engine"""
    try:
        ee.Initialize(project=project)
        print("GEE authentication successful!")
    except ee.ee_exception.EEException:
        ee.Authenticate(quiet=False, auth_mode="localhost")
        ee.Initialize(project=project)
        print("GEE authentication completed!")


def get_site_specific_latest_run(site_name: str, data_config: DataConfig | None = None) -> str:
    if not data_config:
        data_config = DataConfig()
    with open(data_config.artifact.site_specific_dir / f"{site_name}_latest_run.txt") as f:
        latest_run = f.read()
    return latest_run


def get_best_model_info(model_dir: Path | str) -> dict:
    with open(model_dir / "best_model_info.json") as f:
        return json.load(f)
