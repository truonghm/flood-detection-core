import csv
import json
import random
import re
from pathlib import Path

import numpy as np
import rasterio

from flood_detection_core.exceptions import NotEnoughImagesError, TileTooSmallError


def get_patches_cache_key(num_patches: int, num_temporal_length: int, patch_size: int) -> str:
    return f"cache_{num_patches}_{num_temporal_length}_{patch_size}"


class PreTrainPatchesExtractor:
    def __init__(
        self,
        split_csv_path: Path,
        output_dir: Path,
        num_patches: int = 100,
        num_temporal_length: int = 4,
        patch_size: int = 16,
        vv_clipped_range: tuple[float, float] | None = None,
        vh_clipped_range: tuple[float, float] | None = None,
    ) -> None:
        self.split_csv_path = split_csv_path
        self.num_patches = num_patches
        self.num_temporal_length = num_temporal_length
        self.patch_size = patch_size
        self.vv_clipped_range = vv_clipped_range
        self.vh_clipped_range = vh_clipped_range
        self.cache_key = get_patches_cache_key(self.num_patches, self.num_temporal_length, self.patch_size)
        self.tile_to_paths = self.load_pre_flood_split_csv(self.split_csv_path)
        self.cache_dir = output_dir / self.cache_key
        self.assignment_path = self.cache_dir / "assignment.json"

        # assignment: idx -> {"tile": str, "y": int, "x": int}
        self.assignment: dict[int, dict] = {}
        # per-tile state: tile -> {
        #     "paths": list[str],
        #     "height": int,
        #     "width": int,
        #     "available_coords": list[(y,x)],
        #     "used_coords": set[str],
        # }
        self._tile_state: dict[str, dict] = {}

        if self.assignment_path.exists():
            try:
                with open(self.assignment_path) as f:
                    raw = json.load(f)
                # keys stored as strings in JSON
                self.assignment = {int(k): v for k, v in raw.get("assignment", {}).items()}
                # rebuild used coords per tile
                for idx, item in self.assignment.items():
                    tile = item["tile"]
                    y = int(item["y"])
                    x = int(item["x"])
                    state = self._tile_state.setdefault(
                        tile,
                        {
                            "paths": None,
                            "height": None,
                            "width": None,
                            "available_coords": [],
                            "used_coords": set(),
                        },
                    )
                    state["used_coords"].add(f"{y},{x}")
            except Exception:
                # If corrupted, start fresh
                self.assignment = {}
                self._tile_state = {}

    def load_pre_flood_split_csv(self, split_csv_path: Path) -> dict[str, list[Path]]:
        data = {}
        with open(split_csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["dataset_type"] == "pretrain":
                    # data.append(row)
                    if row["tile"] not in data:
                        data[row["tile"]] = []
                    data[row["tile"]].append(Path(row["path"]))
        return data

    def _save_assignment(self) -> None:
        self.assignment_path.parent.mkdir(parents=True, exist_ok=True)
        serializable = {"assignment": {str(k): v for k, v in self.assignment.items()}}
        with open(self.assignment_path, "w") as f:
            json.dump(serializable, f)

    def _choose_preflood_paths_for_tile(self, tile: str) -> list[Path]:
        paths = self.tile_to_paths[tile]
        # sort by pre_flood_(\d+)_ index if present; otherwise keep original order
        try:
            indexed = []
            for p in paths:
                m = re.search(r"pre_flood_(\d+)_", p.name)
                if m:
                    indexed.append((int(m.group(1)), p))
                else:
                    indexed.append((10**9, p))
            indexed.sort(key=lambda x: x[0])
            sorted_paths = [p for _, p in indexed]
        except Exception:
            sorted_paths = list(paths)

        if len(sorted_paths) < self.num_temporal_length:
            msg = (
                "Tile has fewer than "
                f"{self.num_temporal_length} images "
                f"({len(sorted_paths)} < {self.num_temporal_length})\n"
                f"Check: {sorted_paths[0].name if sorted_paths else tile}"
            )
            raise NotEnoughImagesError(msg)

        start = random.randint(0, len(sorted_paths) - self.num_temporal_length)
        end = start + self.num_temporal_length
        return sorted_paths[start:end]

    def _ensure_tile_state(self, tile: str) -> None:
        if tile in self._tile_state and self._tile_state[tile].get("available_coords"):
            return
        existing = self._tile_state.get(tile)
        used_coords = existing.get("used_coords") if isinstance(existing, dict) and "used_coords" in existing else set()
        state = self._tile_state.setdefault(
            tile,
            {
                "paths": None,
                "height": None,
                "width": None,
                "available_coords": [],
                "used_coords": used_coords,
            },
        )
        if state["paths"] is None:
            chosen_paths = self._choose_preflood_paths_for_tile(tile)
            # compute min height/width across chosen paths using metadata only
            min_h = float("inf")
            min_w = float("inf")
            for p in chosen_paths:
                with rasterio.open(p) as src:
                    h, w = src.height, src.width
                min_h = min(min_h, h)
                min_w = min(min_w, w)
            state["paths"] = [str(p) for p in chosen_paths]
            state["height"] = int(min_h)
            state["width"] = int(min_w)

        # build available non-overlapping coords grid aligned to patch_size
        patch = self.patch_size
        grid_h = state["height"] // patch
        grid_w = state["width"] // patch
        if grid_h == 0 or grid_w == 0:
            raise TileTooSmallError(f"Tile {tile} is too small ({state['height']}x{state['width']})")

        coords: list[tuple[int, int]] = []
        for gy in range(grid_h):
            for gx in range(grid_w):
                y = gy * patch
                x = gx * patch
                if f"{y},{x}" not in state["used_coords"]:
                    coords.append((y, x))
        random.shuffle(coords)
        state["available_coords"] = coords
        self._tile_state[tile] = state

    def _extract_one_sequence_at_coords(
        self,
        tile_name: str,
        selected_paths: list[str],
        start_y: int,
        start_x: int,
    ) -> list[np.ndarray]:
        images_data = []
        for img_path_str in selected_paths:
            img_path = Path(img_path_str)
            if img_path.suffix == ".tif":
                with rasterio.open(img_path) as src:
                    data = src.read()
                    data = np.transpose(data, (1, 2, 0))
            elif img_path.suffix == ".npy":
                data = np.load(img_path)
            else:
                raise ValueError(f"Invalid extension: {img_path.suffix}")

            # Apply normalization if clipped ranges are provided
            if self.vv_clipped_range is not None:
                # handle nan values
                vv_band = data[:, :, 0].copy()
                vv_band = np.where(np.isnan(vv_band), self.vv_clipped_range[0], vv_band)
                data[:, :, 0] = np.clip(
                    (vv_band - self.vv_clipped_range[0]) / (self.vv_clipped_range[1] - self.vv_clipped_range[0]),
                    0,
                    1,
                )
            if self.vh_clipped_range is not None:
                vh_band = data[:, :, 1].copy()
                vh_band = np.where(np.isnan(vh_band), self.vh_clipped_range[0], vh_band)
                data[:, :, 1] = np.clip(
                    (vh_band - self.vh_clipped_range[0]) / (self.vh_clipped_range[1] - self.vh_clipped_range[0]),
                    0,
                    1,
                )

            images_data.append(data)

        patch_size = self.patch_size
        # Basic bounds check to avoid accidental OOB in unexpected tiles
        min_height = min(arr.shape[0] for arr in images_data)
        min_width = min(arr.shape[1] for arr in images_data)
        if start_y + patch_size > min_height or start_x + patch_size > min_width:
            msg = (
                f"Assigned coords out of bounds for tile {tile_name}: "
                f"({start_y},{start_x}) with size {patch_size} on "
                f"{min_height}x{min_width}"
            )
            raise TileTooSmallError(msg)

        patch_sequence: list[np.ndarray] = []
        for data in images_data:
            patch = data[start_y : start_y + patch_size, start_x : start_x + patch_size, :]
            patch_sequence.append(patch)
        return patch_sequence

    def extract(self, idx: int) -> list[np.ndarray]:
        if idx < 0 or idx >= self.num_patches:
            raise ValueError(f"idx {idx} is out of range [0, {self.num_patches})")

        # check if cache exists and is complete
        patch_cache_dir = self.cache_dir / f"{idx}"
        patch_sequence = []
        if patch_cache_dir.exists():
            cached_files = list(patch_cache_dir.glob("*.npy"))
            if len(cached_files) == self.num_temporal_length:
                for patch_path in sorted(cached_files):  # sort to ensure consistent order
                    patch = np.load(patch_path)
                    patch_sequence.append(patch)
            else:
                # incomplete cache, remove and regenerate
                import shutil

                shutil.rmtree(patch_cache_dir)

        if not patch_sequence:  # if cache didn't exist or was incomplete
            # ensure assignment exists for idx
            if idx not in self.assignment:
                # find a tile with available coords
                tiles = list(self.tile_to_paths.keys())
                random.shuffle(tiles)
                assigned = False
                last_err: Exception | None = None
                for tile in tiles:
                    try:
                        self._ensure_tile_state(tile)
                        state = self._tile_state[tile]
                        if not state["available_coords"]:
                            continue
                        y, x = state["available_coords"].pop()
                        state["used_coords"].add(f"{y},{x}")
                        self.assignment[idx] = {"tile": tile, "y": int(y), "x": int(x)}
                        self._save_assignment()
                        assigned = True
                        break
                    except (NotEnoughImagesError, TileTooSmallError) as e:
                        last_err = e
                        continue
                if not assigned:
                    if last_err is not None:
                        raise last_err
                    msg = "Not enough non-overlapping patches across available tiles. Reduce num_patches or patch_size."
                    raise ValueError(msg)

            assign = self.assignment[idx]
            tile_name = assign["tile"]
            state = self._tile_state.get(tile_name)
            if state is None or state.get("paths") is None:
                # ensure tile state is loaded (paths/dims) if assignment loaded from disk
                self._ensure_tile_state(tile_name)
                state = self._tile_state[tile_name]
            try:
                patch_sequence = self._extract_one_sequence_at_coords(
                    tile_name=tile_name,
                    selected_paths=state["paths"],
                    start_y=int(assign["y"]),
                    start_x=int(assign["x"]),
                )
            except (NotEnoughImagesError, TileTooSmallError):
                # This should be rare; rethrow to indicate configuration issue
                raise

        # only cache if we generated new data (not loaded from cache)
        if not (self.cache_dir / f"{idx}").exists():
            self.cache(patch_sequence, idx)
        return patch_sequence

    def cache(self, patch_sequence: list[np.ndarray], idx: int) -> None:
        for i, patch in enumerate(patch_sequence):
            patch_dir = self.cache_dir / f"{idx}"
            patch_path = patch_dir / f"{i}.npy"
            patch_dir.mkdir(parents=True, exist_ok=True)
            np.save(patch_path, patch)

    def __call__(self, idx: int) -> list[np.ndarray]:
        return self.extract(idx)


def extract_patches_at_coords(
    image_paths: list[Path | str],
    patch_coords: tuple[int, int],
    patch_size: int,
    vv_clipped_range: tuple[float, float] | None = None,
    vh_clipped_range: tuple[float, float] | None = None,
) -> list[np.ndarray]:
    patches = []
    i, j = patch_coords

    for img_path in image_paths:
        if isinstance(img_path, str):
            img_path = Path(img_path)

        if img_path.suffix == ".tif":
            with rasterio.open(img_path) as src:
                data = src.read()
                data = np.transpose(data, (1, 2, 0))
        elif img_path.suffix == ".npy":
            data = np.load(img_path)
        else:
            raise ValueError(f"Invalid format: {img_path}")

        if vv_clipped_range is not None:
            # handle nan values
            vv_band = data[:, :, 0].copy()
            vv_band = np.where(np.isnan(vv_band), vv_clipped_range[0], vv_band)
            data[:, :, 0] = np.clip(
                (vv_band - vv_clipped_range[0]) / (vv_clipped_range[1] - vv_clipped_range[0]),
                0,
                1,
            )
        if vh_clipped_range is not None:
            vh_band = data[:, :, 1].copy()
            vh_band = np.where(np.isnan(vh_band), vh_clipped_range[0], vh_band)
            data[:, :, 1] = np.clip(
                (vh_band - vh_clipped_range[0]) / (vh_clipped_range[1] - vh_clipped_range[0]),
                0,
                1,
            )

        patch = data[i : i + patch_size, j : j + patch_size, :]
        patches.append(patch)

    return patches
