from typing import Annotated

import typer

app = typer.Typer()


@app.command()
def eda(
    data_config_path: Annotated[str, typer.Option("--data-config-path")],
    site_name: Annotated[str | None, typer.Option("--site-name", "-sn")] = None,
    tile_name: Annotated[str | None, typer.Option("--tile-name", "-tn")] = None,
    output_dir: Annotated[str | None, typer.Option("--output-dir", "-o")] = None,
    pre_flood_index_for_cd: Annotated[int, typer.Option("--pf-index")] = -1,
    vv_norm_lb: Annotated[float, typer.Option("--vv-norm-lb")] = -23,
    vv_norm_ub: Annotated[float, typer.Option("--vv-norm-ub")] = 0,
    vh_norm_lb: Annotated[float, typer.Option("--vh-norm-lb")] = -28,
    vh_norm_ub: Annotated[float, typer.Option("--vh-norm-ub")] = -5,
    detection_threshold: Annotated[float, typer.Option("--detection-threshold")] = 0.75,
    min_connected_pixels: Annotated[int, typer.Option("--min-connected-pixels")] = 8,
    slope_threshold: Annotated[float, typer.Option("--slope-threshold")] = 0.05,
    speckle_filtering: Annotated[bool, typer.Option("--speckle-filtering")] = True,
    speckle_filtering_radius_meters: Annotated[float, typer.Option("--sf-radius-meters")] = 50.0,
    speckle_filtering_pixel_size: Annotated[float, typer.Option("--sf-pixel-size")] = 10.0,
):
    from flood_detection_core.analysis.gen_eda_plots import gen_eda_plots

    gen_eda_plots(
        data_config_path=data_config_path,
        site_name=site_name,
        tile_name=tile_name,
        output_dir=output_dir,
        pre_flood_index_for_cd=pre_flood_index_for_cd,
        vv_norm_lb=vv_norm_lb,
        vv_norm_ub=vv_norm_ub,
        vh_norm_lb=vh_norm_lb,
        vh_norm_ub=vh_norm_ub,
        detection_threshold=detection_threshold,
        min_connected_pixels=min_connected_pixels,
        slope_threshold=slope_threshold,
        speckle_filtering=speckle_filtering,
        speckle_filtering_radius_meters=speckle_filtering_radius_meters,
        speckle_filtering_pixel_size=speckle_filtering_pixel_size,
    )

@app.command()
def download_gee():
    print("Not yet implemented")
    typer.Exit()


if __name__ == "__main__":
    app()
