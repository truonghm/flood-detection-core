from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from rich import print

gee_download_config_path = Path(__file__).parent / "yamls/gee.yaml"
data_config_path = Path(__file__).parent / "yamls/data.yaml"
model_config_path = Path(__file__).parent / "yamls/model_clvae.yaml"


class BaseSettingsWithYaml(BaseSettings):
    @classmethod
    def from_yaml(cls, yaml_file: Path):
        return cls.model_validate(cls._load_yaml(yaml_file))

    @classmethod
    def _load_yaml(cls, yaml_file: Path):
        with open(yaml_file) as f:
            return yaml.safe_load(f)


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


class GEEDownloadConfig(BaseSettingsWithYaml): ...


class Sen1Flood11GeeDownloadConfig(GEEDownloadConfig):
    debug: bool = False
    target: str | Literal["all", "pretrain"]
    output: OutputConfig
    flood_sites: FloodSitesConfig
    preprocessing: PreprocessingConfig

    model_config = SettingsConfigDict(yaml_file=gee_download_config_path)


class DataDirConfig(BaseModel):
    data_dir: str | Path


class ArtifactDirConfig(DataDirConfig):
    pretrain_dir: str | Path
    site_specific_dir: str | Path


class SplitDirConfig(DataDirConfig):
    pre_flood_split: str | Path
    post_flood_split: str | Path


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


class DataConfig(BaseSettingsWithYaml):
    relative_to: str | Path = Field(default=".", description="The relative path to the root of the project")
    gee: GEEDataConfig
    hand_labeled_sen1flood11: Sen1Flood11HandLabeledDataConfig
    artifact: ArtifactDirConfig
    splits: SplitDirConfig

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
            for config_name in ["gee", "hand_labeled_sen1flood11", "artifact", "splits"]:
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


class CLVAEPretrainConfig(BaseModel):
    num_patches: int = 100
    num_temporal_length: int = 4
    patch_size: int = 16
    input_channels: int = 2
    hidden_channels: int = 64
    latent_dim: int = 128
    learning_rate: float = 0.001
    max_epochs: int = 50
    scheduler_patience: int = 5
    scheduler_factor: float = 0.1
    scheduler_min_lr: float = 0.00001
    early_stopping_patience: int = 10
    batch_size: int = 32


class CLVAESiteSpecificConfig(BaseModel):
    num_temporal_length: int = 4
    patch_size: int = 16
    patch_stride: int = 16
    input_channels: int = 2
    hidden_channels: int = 64
    latent_dim: int = 128
    learning_rate: float = 0.001
    max_epochs: int = 25
    scheduler_patience: int = 5
    scheduler_factor: float = 0.1
    scheduler_min_lr: float = 0.00001
    batch_size: int = 32
    early_stopping_patience: int = 5
    vv_clipped_range: tuple[float, float] | None = None
    vh_clipped_range: tuple[float, float] | None = None
    # Loss weights from paper
    alpha: float = 0.1
    beta: float = 0.7
    max_patches_per_pair: int = 128


class CLVAEInferenceConfig(BaseModel):
    num_temporal_length: int = 4
    patch_size: int = 16
    patch_stride: int = 1
    pad_size: int = 8
    batch_size: int = 512
    threshold: float = 0.5
    post_flood_vh_clipped_range: tuple[float, float] | None = None
    post_flood_vv_clipped_range: tuple[float, float] | None = None


class GeometricAugmentationConfig(BaseModel):
    left_right: float = 0.5
    up_down: float = 0.2
    rotate: tuple[float, float] = (-90, 90)


class NonGeometricAugmentationConfig(BaseModel):
    gaussian_blur: float = 0.3
    gamma_contrast_prob: float = 0.5
    gamma_contrast: tuple[float, float] = (0.25, 2.0)


class AugmentationConfig(BaseModel):
    geometric: GeometricAugmentationConfig
    non_geometric: NonGeometricAugmentationConfig


class CLVAEConfig(BaseSettingsWithYaml):
    pretrain: CLVAEPretrainConfig
    site_specific: CLVAESiteSpecificConfig
    augmentation: AugmentationConfig
    inference: CLVAEInferenceConfig
    model_config = SettingsConfigDict(yaml_file=model_config_path)


if __name__ == "__main__":
    download_config = Sen1Flood11GeeDownloadConfig.from_yaml(gee_download_config_path)
    print(download_config)

    data_config = DataConfig.from_yaml(data_config_path)
    print(data_config)
