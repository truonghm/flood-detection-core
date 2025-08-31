from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, field_validator, model_validator
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
    max_dimension: int = 512
    scale: float = 10.0


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
    preprocessing: PreprocessingConfig | None = None

    model_config = SettingsConfigDict(yaml_file=gee_download_config_path)


class UrbanSARGeeDownloadConfig(GEEDownloadConfig):
    debug: bool = False
    target: str | Literal["all", "pretrain"]
    output: OutputConfig
    flood_sites: FloodSitesConfig
    preprocessing: PreprocessingConfig | None = None

    model_config = SettingsConfigDict(yaml_file=gee_download_config_path)


class PathConfig(BaseModel): ...


class DataSourcesConfig(PathConfig):
    pretrain_cache: str | Path
    pre_flood: str | Path
    catalog_source: str | Path
    ground_truth: str | Path
    post_flood: str | Path
    pre_flood_base: str | Path | None = None


class ArtifactsDirConfig(PathConfig):
    pretrain: str | Path
    site_specific: str | Path


class SplitDirConfig(PathConfig):
    pre_flood_split: str | Path
    post_flood_split: str | Path


class CSVsConfig(PathConfig):
    path_mapping: str | Path
    pre_flood_split: str | Path
    post_flood_split: str | Path


class DataConfig(BaseSettingsWithYaml):
    site_metadata: str | Path
    data_dirs: DataSourcesConfig
    artifacts_dirs: ArtifactsDirConfig
    csv_files: CSVsConfig

    @model_validator(mode="before")
    def validate_paths(cls, values):
        if isinstance(values, dict):
            for config_name in ["data_dirs", "artifacts_dirs", "csv_files"]:
                config_values = values.get(config_name)
                if config_values and isinstance(config_values, dict):
                    for field_name, field_value in config_values.items():
                        if field_value is not None:
                            config_values[field_name] = Path(field_value)

        return values

    @field_validator("site_metadata")
    def validate_site_metadata(cls, v: str | Path) -> Path:
        if isinstance(v, str):
            v = Path(v)
        return v

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
    vv_clipped_range: tuple[float, float] | None = None
    vh_clipped_range: tuple[float, float] | None = None


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
    max_patches_per_pair: int | Literal["all"] | None = "all"
    positive_pair_ratio: float = 0.5  # Ratio of positive pairs in contrastive learning
    contrastive_margin: float = 0.3


class CLVAEInferenceConfig(BaseModel):
    num_temporal_length: int = 4
    patch_size: int = 16
    patch_stride: int = 1
    pad_size: int = 8
    batch_size: int = 512
    threshold: float = 0.5
    pre_flood_vh_clipped_range: tuple[float, float] | None = None
    pre_flood_vv_clipped_range: tuple[float, float] | None = None
    post_flood_vh_clipped_range: tuple[float, float] | None = None
    post_flood_vv_clipped_range: tuple[float, float] | None = None
    normalize_site_name: bool = True


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
