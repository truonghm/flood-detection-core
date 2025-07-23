import datetime
import json
from pathlib import Path

import matplotlib.pyplot as plt
from rich import print
from rich.progress import track

from flood_detection_core.analysis.basic_cd import detect_change, filter_speckle
from flood_detection_core.analysis.plotting import plot_change_detection, plot_ts_images
from flood_detection_core.config import DataConfig
from flood_detection_core.data.constants import HandLabeledSen1Flood11Sites
from flood_detection_core.data.loaders.tif import TifRawLoader


def gen_eda_plots_for_tile(
    loader: TifRawLoader,
    tile_name: str,
    output_dir: str | Path,
    pre_flood_index_for_cd: int = -1,
    vv_norm_range: tuple[float, float] | None = None,
    vh_norm_range: tuple[float, float] | None = None,
    detection_threshold: float = 0.75,
    min_connected_pixels: int = 8,
    slope_threshold: float = 0.05,
    speckle_filtering: bool = True,
    speckle_filtering_radius_meters: float = 50.0,
    speckle_filtering_pixel_size: float = 10.0,
) -> None:
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pre_flood_data = loader.load_pre_flood_tile_data(tile_name)
    post_flood_data = loader.load_hl_post_flood_s1_tile_data(
        tile_name=tile_name,
        vv_norm_range=vv_norm_range,
        vh_norm_range=vh_norm_range,
    )
    permanent_water_data = loader.load_hl_permanent_water_tile_data(tile_name)
    ground_truth_data = loader.load_hl_ground_truth_tile_data(tile_name)

    fig, axes = plot_ts_images(
        pre_flood_data=pre_flood_data,
        post_flood_data=post_flood_data,
        tile_name=tile_name,
    )
    fig.savefig(output_dir / f"{tile_name}_ts.png")
    plt.close(fig)

    if speckle_filtering:
        pre_data = filter_speckle(
            data=pre_flood_data[pre_flood_index_for_cd].data,
            radius_meters=speckle_filtering_radius_meters,
            pixel_size=speckle_filtering_pixel_size,
        )
        post_data = filter_speckle(
            data=post_flood_data.data,
            radius_meters=speckle_filtering_radius_meters,
            pixel_size=speckle_filtering_pixel_size,
        )
    else:
        pre_data = pre_flood_data[pre_flood_index_for_cd].data
        post_data = post_flood_data.data

    difference, raw_change_mask, refined_change_mask = detect_change(
        before_data=pre_data,
        after_data=post_data,
        permanent_water_mask=permanent_water_data.data,
        threshold=detection_threshold,
        min_connected_pixels=min_connected_pixels,
        slope_threshold=slope_threshold,
    )

    fig, axes = plot_change_detection(
        tile_name=tile_name,
        pre_data=pre_data,
        post_data=post_data,
        difference=difference,
        raw_change_mask=raw_change_mask,
        refined_change_mask=refined_change_mask,
        permanent_water_data=permanent_water_data,
        ground_truth_data=ground_truth_data,
    )
    fig.savefig(output_dir / f"{tile_name}_change_detection.png")
    plt.close(fig)

def gen_eda_plots_for_site(
    loader: TifRawLoader,
    site_name: str,
    output_dir: str | Path,
    pre_flood_index_for_cd: int = -1,
    vv_norm_range: tuple[float, float] | None = None,
    vh_norm_range: tuple[float, float] | None = None,
    detection_threshold: float = 0.75,
    min_connected_pixels: int = 8,
    slope_threshold: float = 0.05,
    speckle_filtering: bool = True,
    speckle_filtering_radius_meters: float = 50.0,
    speckle_filtering_pixel_size: float = 10.0,
) -> None:
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    output_dir = output_dir / site_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # get tile names
    tile_paths = loader.data_config.hand_labeled_sen1flood11.get_post_flood_s1_site_paths(site_name)
    tile_names = ["_".join(tile_path.stem.split("_")[:2]) for tile_path in tile_paths]

    for tile_name in track(tile_names, description="Generating plots for site..."):
        gen_eda_plots_for_tile(
            loader=loader,
            tile_name=tile_name,
            output_dir=output_dir,
            pre_flood_index_for_cd=pre_flood_index_for_cd,
            vv_norm_range=vv_norm_range,
            vh_norm_range=vh_norm_range,
            detection_threshold=detection_threshold,
            min_connected_pixels=min_connected_pixels,
            slope_threshold=slope_threshold,
            speckle_filtering=speckle_filtering,
            speckle_filtering_radius_meters=speckle_filtering_radius_meters,
            speckle_filtering_pixel_size=speckle_filtering_pixel_size,
        )


def gen_eda_plots(
    data_config_path: str | Path,
    site_name: str | None = None,
    tile_name: str | None = None,
    output_dir: str | Path | None = None,
    pre_flood_index_for_cd: int = -1,
    # vv_norm_range: tuple[float, float] | None = None,
    # vh_norm_range: tuple[float, float] | None = None,
    vv_norm_lb: float = -23,
    vv_norm_ub: float = 0,
    vh_norm_lb: float = -28,
    vh_norm_ub: float = -5,
    detection_threshold: float = 0.75,
    min_connected_pixels: int = 8,
    slope_threshold: float = 0.05,
    speckle_filtering: bool = True,
    speckle_filtering_radius_meters: float = 50.0,
    speckle_filtering_pixel_size: float = 10.0,
) -> None:
    if not output_dir:
        output_dir = Path("visualizations")
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shared_kwargs = {
        "pre_flood_index_for_cd": pre_flood_index_for_cd,
        "vv_norm_range": (vv_norm_lb, vv_norm_ub),
        "vh_norm_range": (vh_norm_lb, vh_norm_ub),
        "detection_threshold": detection_threshold,
        "min_connected_pixels": min_connected_pixels,
        "slope_threshold": slope_threshold,
        "speckle_filtering": speckle_filtering,
        "speckle_filtering_radius_meters": speckle_filtering_radius_meters,
        "speckle_filtering_pixel_size": speckle_filtering_pixel_size,
    }
    with open(output_dir / "visualization_metadata.json", "w") as f:
        viz_metadata = {
            "datetime": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        viz_metadata.update(shared_kwargs)
        json.dump(viz_metadata, f, indent=4)

    data_config = DataConfig(_yaml_file=data_config_path)
    loader = TifRawLoader(data_config)

    if not site_name and not tile_name:
        print("Will generate plots for all sites and their tiles")
        # raise NotImplementedError("Not implemented")
        print(f"Available sites: {HandLabeledSen1Flood11Sites}")
        for site_name in HandLabeledSen1Flood11Sites:
            print(f"Generating plots for site: {site_name}")
            gen_eda_plots_for_site(
                loader=loader,
                site_name=site_name,
                output_dir=output_dir,
                **shared_kwargs,
            )
            print("--------------------------------")

    if not site_name and tile_name:
        site_name = tile_name.split("_")[0].lower()

    if tile_name:
        print(f"Generating plots for tile: {tile_name}")
        return gen_eda_plots_for_tile(
            loader=loader,
            tile_name=tile_name,
            output_dir=output_dir,
            **shared_kwargs,
        )
    elif site_name:
        print(f"Generating plots for site: {site_name}")
        return gen_eda_plots_for_site(
            loader=loader,
            site_name=site_name,
            output_dir=output_dir,
            **shared_kwargs,
        )


if __name__ == "__main__":
    gen_eda_plots(
        data_config_path="flood-detection-core/src/yamls/data_config.yaml",
        output_dir="./visualizations",
    )
