from typing import Annotated

import typer

app = typer.Typer()

default_data_config_path = "./flood-detection-core/yamls/data.yaml"
default_model_config_path = "./flood-detection-core/yamls/model_clvae.yaml"
default_download_config_path = "./flood-detection-core/yamls/gee.yaml"


@app.command()
def eda(
    data_config_path: Annotated[str, typer.Option("--data-config-path")] = default_data_config_path,
    site_name: Annotated[str | None, typer.Option("--site-name", "-sn")] = None,
    tile_name: Annotated[str | None, typer.Option("--tile-name", "-tn")] = None,
    output_dir: Annotated[str | None, typer.Option("--output-dir", "-o")] = "./visualizations",
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
) -> None:
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
def dl_gee_sen1flood11(
    data_config_path: Annotated[str, typer.Option("--data-config-path")] = default_data_config_path,
    download_config_path: Annotated[str, typer.Option("--download-config-path")] = default_download_config_path,
    gcp_project_id: Annotated[str | None, typer.Option("--gcp-project-id")] = None,
) -> None:
    import os

    from flood_detection_core.config import DataConfig, Sen1Flood11GeeDownloadConfig
    from flood_detection_core.data.downloaders.gee_downloader import SitePrefloodDataDownloader
    from flood_detection_core.utils import authenticate_gee

    if not gcp_project_id:
        gcp_project_id = os.getenv("GCP_PROJECT_ID")
    authenticate_gee(gcp_project_id)
    download_config = Sen1Flood11GeeDownloadConfig.from_yaml(download_config_path)
    data_config = DataConfig.from_yaml(data_config_path)
    downloader = SitePrefloodDataDownloader(download_config, data_config)
    downloader()


@app.command()
def pretrain(
    data_config_path: Annotated[str, typer.Option("--data-config-path")] = default_data_config_path,
    model_config_path: Annotated[str, typer.Option("--model-config-path")] = default_model_config_path,
    use_wandb: Annotated[bool, typer.Option("--wandb/--no-wandb")] = True,
) -> None:
    from flood_detection_core.config import CLVAEConfig, DataConfig
    from flood_detection_core.pipelines import pretrain

    data_config = DataConfig.from_yaml(data_config_path)
    model_config = CLVAEConfig.from_yaml(model_config_path)

    if use_wandb:
        import wandb

        with wandb.init(project="flood-detection-dl", name="clvae-pretrain", tags=["clvae"]) as run:
            pretrain(data_config, model_config, wandb_run=run)
    else:
        pretrain(data_config, model_config)


@app.command()
def pretrain_test(
    data_config_path: Annotated[str, typer.Option("--data-config-path")] = default_data_config_path,
    model_config_path: Annotated[str, typer.Option("--model-config-path")] = default_model_config_path,
    use_wandb: Annotated[bool, typer.Option("--wandb/--no-wandb")] = True,
) -> None:
    from flood_detection_core.config import CLVAEConfig, DataConfig
    from flood_detection_core.pipelines import pretrain

    data_config = DataConfig.from_yaml(data_config_path)
    model_config = CLVAEConfig.from_yaml(model_config_path)

    if use_wandb:
        import wandb

        with wandb.init(project="flood-detection-dl", name="clvae-pretrain", tags=["clvae"]) as run:
            pretrain(data_config, model_config, wandb_run=run, num_patches=10)
    else:
        pretrain(data_config, model_config, num_patches=10)


@app.command()
def site_specific_train(
    site_name: Annotated[str, typer.Option("--site-name")],
    pretrained_model_path: Annotated[str, typer.Option("--pretrained-model-path")],
    data_config_path: Annotated[str, typer.Option("--data-config-path")] = default_data_config_path,
    model_config_path: Annotated[str, typer.Option("--model-config-path")] = default_model_config_path,
    use_wandb: Annotated[bool, typer.Option("--wandb/--no-wandb")] = True,
) -> None:
    from flood_detection_core.config import CLVAEConfig, DataConfig
    from flood_detection_core.pipelines import site_specific_train

    data_config = DataConfig.from_yaml(data_config_path)
    model_config = CLVAEConfig.from_yaml(model_config_path)

    if use_wandb:
        import wandb

        with wandb.init(
            project="flood-detection-dl", name=f"clvae-site-specific-{site_name}", tags=["clvae", "site-specific"]
        ) as run:
            site_specific_train(
                data_config=data_config,
                model_config=model_config,
                pretrained_model_path=pretrained_model_path,
                site_name=site_name,
                wandb_run=run,
            )
    else:
        site_specific_train(
            data_config=data_config,
            model_config=model_config,
            pretrained_model_path=pretrained_model_path,
            site_name=site_name,
        )


@app.command()
def split_data(
    data_config_path: Annotated[str, typer.Option("--data-config-path")] = default_data_config_path,
) -> None:
    from flood_detection_core.config import DataConfig
    from flood_detection_core.data.processing.split import split_data

    data_config = DataConfig.from_yaml(data_config_path)
    split_data(data_config)


@app.command()
def train(
    data_config_path: Annotated[str, typer.Option("--data-config-path")] = default_data_config_path,
    model_config_path: Annotated[str, typer.Option("--model-config-path")] = default_model_config_path,
    use_wandb: Annotated[bool, typer.Option("--wandb/--no-wandb")] = True,
    pretrain_extra_tags: Annotated[list[str], typer.Option("--pretrain-extra-tags")] = [],
    site_specific_extra_tags: Annotated[list[str], typer.Option("--site-specific-extra-tags")] = [],
) -> None:
    from flood_detection_core.config import CLVAEConfig, DataConfig
    from flood_detection_core.pipelines import TrainingInput, TrainingManager

    data_config = DataConfig.from_yaml(data_config_path)
    model_config = CLVAEConfig.from_yaml(model_config_path)

    training_input_raw = {
        "pretrain": {"mode": "fresh", "path": None},
        "site_specific": [
            {"site": "bolivia", "mode": "fresh", "path": None},
            # {"site": "mekong", "mode": "fresh", "path": None},
            # {"site": "somalia", "mode": "fresh", "path": None},
            # {"site": "spain", "mode": "fresh", "path": None},
        ],
    }
    training_input = TrainingInput(**training_input_raw)
    print(training_input)

    training_manager = TrainingManager(data_config, model_config)
    training_manager.run(
        training_input,
        use_wandb=use_wandb,
        pretrain_extra_tags=pretrain_extra_tags,
        site_specific_extra_tags=site_specific_extra_tags,
    )


if __name__ == "__main__":
    app()
