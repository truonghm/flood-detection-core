from typing import Literal

from pydantic import BaseModel


class SiteMetadata(BaseModel): ...


class TileMetadata(BaseModel): ...


class BasePerSiteTilesMetadata(BaseModel):
    site: str | Literal["all"]
    tiles: dict[str, TileMetadata]

    def __len__(self) -> int:
        return len(self.tiles)
