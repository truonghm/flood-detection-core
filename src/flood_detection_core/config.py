from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict, YamlConfigSettingsSource
from rich import print

gee_download_config_path = Path(__file__).parent / "yamls/gee.yaml"
data_config_path = Path(__file__).parent / "yamls/data.yaml"
model_config_path = Path(__file__).parent / "yamls/model_clvae.yaml"


class OutputConfig(BaseModel):
    format: Literal["numpy", "geotiff"]
    max_dimension: int = 4000


class FloodSitesConfig(BaseModel):
    num_pre_images: int = 8
    days_before_flood_max: int = 120
    days_before_flood_min: int = 5


class PreprocessingConfig(BaseModel):
    vv_clip_lower_bound: float = -23
    vv_clip_upper_bound: float = 0
    vh_clip_lower_bound: float = -28
    vh_clip_upper_bound: float = -5


class GEEDownloadConfig(BaseSettings): ...


class Sen1Flood11GeeDownloadConfig(GEEDownloadConfig):
    debug: bool = False
    target: str | Literal["all", "pretrain"]
    output: OutputConfig
    flood_sites: FloodSitesConfig
    preprocessing: PreprocessingConfig

    model_config = SettingsConfigDict(yaml_file=gee_download_config_path)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (YamlConfigSettingsSource(settings_cls),)


class DataDirConfig(BaseModel):
    data_dir: str | Path


class GEEDataConfig(DataDirConfig):
    pre_flood_dir: str | Path
    pretrain_dir: str | Path

    def get_pretrain_metadata_path(self) -> Path:
        return self.pretrain_dir / "metadata.json"

    def get_pretrain_data_paths(self, patch_id: str | None = None) -> list[Path]:
        if patch_id is None:
            return list((self.pretrain_dir).glob("patch_*/*.tif"))
        return list((self.pretrain_dir / patch_id).glob("*.tif"))

    def get_pre_flood_tile_paths(self, site_name: str, tile_name: str) -> list[Path]:
        return list(self.pre_flood_dir.glob(f"{site_name}/{tile_name}/*.tif"))

    def get_pre_flood_site_paths(self, site_name: str) -> list[Path]:
        return list(self.pre_flood_dir.glob(f"{site_name}/{site_name.capitalize()}_/*.tif"))

    def get_all_pre_flood_tiles_paths(self) -> list[Path]:
        return list(self.pre_flood_dir.glob("*/*/*.tif"))

    def get_pre_flood_site_metadata_path(self, site_name: str) -> Path:
        return self.pre_flood_dir / site_name / "overall_metadata.json"


class Sen1Flood11HandLabeledDataConfig(DataDirConfig):
    metadata: str | Path
    catalog_source: str | Path
    catalog_label: str | Path
    permanent_water: str | Path
    ground_truth: str | Path
    post_flood_s1: str | Path
    post_flood_s2: str | Path

    def get_catalog_source_tile_path(self, tile_name: str) -> Path:
        return self.catalog_source / tile_name / f"{tile_name}.json"

    def get_catalog_source_site_paths(self, site_name: str) -> list[Path]:
        site_name = site_name.lower()
        tile_pattern = f"{site_name.capitalize()}_*"
        return list(self.catalog_source.glob(f"{tile_pattern}/{tile_pattern}.json"))

    def get_catalog_label_tile_path(self, tile_name: str) -> Path:
        return self.catalog_label / tile_name / f"{tile_name}.json"

    def get_catalog_label_site_paths(self, site_name: str) -> list[Path]:
        site_name = site_name.lower()
        tile_pattern = f"{site_name.capitalize()}_*"
        return list(self.catalog_label.glob(f"{tile_pattern}/{tile_pattern}.json"))

    def get_permanent_water_file_paths(self, tile_name: str) -> list[Path]:
        return list(self.permanent_water.glob(f"{tile_name}_*.tif"))

    def get_permanent_water_site_paths(self, site_name: str) -> list[Path]:
        site_name = site_name.lower()
        tile_pattern = f"{site_name.capitalize()}_*"
        return list(self.permanent_water.glob(f"{tile_pattern}.tif"))

    def get_ground_truth_file_paths(self, tile_name: str) -> list[Path]:
        return list(self.ground_truth.glob(f"{tile_name}_*.tif"))

    def get_ground_truth_site_paths(self, site_name: str) -> list[Path]:
        site_name = site_name.lower()
        tile_pattern = f"{site_name.capitalize()}_*"
        return list(self.ground_truth.glob(f"{tile_pattern}.tif"))

    def get_post_flood_s1_file_paths(self, tile_name: str) -> list[Path]:
        return list(self.post_flood_s1.glob(f"{tile_name}_*.tif"))

    def get_post_flood_s1_site_paths(self, site_name: str) -> list[Path]:
        site_name = site_name.lower()
        tile_pattern = f"{site_name.capitalize()}_*"
        return list(self.post_flood_s1.glob(f"{tile_pattern}.tif"))

    def get_post_flood_s2_file_paths(self, tile_name: str) -> list[Path]:
        return list(self.post_flood_s2.glob(f"{tile_name}_*.tif"))

    def get_post_flood_s2_site_paths(self, site_name: str) -> list[Path]:
        site_name = site_name.lower()
        tile_pattern = f"{site_name.capitalize()}_*"
        return list(self.post_flood_s2.glob(f"{tile_pattern}.tif"))


class DataConfig(BaseSettings):
    relative_to: str | Path = Field(default=".", description="The relative path to the root of the project")
    gee: GEEDataConfig
    hand_labeled_sen1flood11: Sen1Flood11HandLabeledDataConfig

    @model_validator(mode="before")
    def validate_paths(cls, values):
        if isinstance(values, dict):
            # Get the relative_to path (root path)
            relative_to = values.get("relative_to", ".")
            root_path = Path(relative_to).absolute()
            # validate that it exists
            if not root_path.exists():
                raise ValueError(f"Root path {root_path} does not exist")

            # Process both gee and sen1flood11 configs
            for config_name in ["gee", "hand_labeled_sen1flood11"]:
                config_values = values.get(config_name)
                if config_values and isinstance(config_values, dict):
                    data_dir = config_values.get("data_dir")
                    if data_dir is not None:
                        # Convert data_dir to root_path / data_dir
                        data_dir_path = root_path / data_dir
                        config_values["data_dir"] = data_dir_path

                        # For all other fields in the config, make them root_path / data_dir / field_value
                        for field_name, field_value in config_values.items():
                            if field_name != "data_dir" and field_value is not None:
                                config_values[field_name] = data_dir_path / field_value

        return values

    model_config = SettingsConfigDict(yaml_file=data_config_path)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (YamlConfigSettingsSource(settings_cls),)


class CLVAEPretrainConfig(BaseModel):
    num_patches: int = 100
    num_temporal_length: int = 4
    patch_size: int = 16
    patch_stride: int = 16
    replacement: bool = False


class CLVAEConfig(BaseSettings):
    pretrain: CLVAEPretrainConfig

    model_config = SettingsConfigDict(yaml_file=model_config_path)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (YamlConfigSettingsSource(settings_cls),)


if __name__ == "__main__":
    download_config = Sen1Flood11GeeDownloadConfig(_yaml_file=gee_download_config_path)
    print(download_config)

    data_config = DataConfig(_yaml_file=data_config_path)
    print(data_config)
