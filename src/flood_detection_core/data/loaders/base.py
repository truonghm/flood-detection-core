from abc import ABC, abstractmethod
from pathlib import Path

from flood_detection_core.config import DataConfig


class RawDataset(ABC):
    def load_data_config(self, yaml_path: str | Path) -> DataConfig:
        return DataConfig.from_yaml(yaml_path)

    @abstractmethod
    def map_all_paths(self) -> dict[str, Path]:
        pass
