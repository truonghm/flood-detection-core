import datetime
import json
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel
from rich import print

import wandb
from wandb.sdk.wandb_run import Run
from .eval import load_ground_truths, test_thresholds
from .predict import generate_distance_maps
from .pretrain import pretrain
from .site_specific import site_specific_train
from flood_detection_core.config import CLVAEConfig, DataConfig
from flood_detection_core.models.clvae import CLVAE
from flood_detection_core.utils import get_best_model_info, get_site_specific_latest_run


class TrainingMode(str, Enum):
    FRESH: str = "fresh"
    RESUME: str = "resume"


class PretrainInput(BaseModel):
    mode: TrainingMode
    path: Path | None
    kwargs: dict[str, Any] = {}


class SiteSpecificInput(BaseModel):
    site: str
    mode: TrainingMode
    path: Path | None
    kwargs: dict[str, Any] = {}


class TrainingInput(BaseModel):
    site_specific: list[SiteSpecificInput]
    pretrain: PretrainInput | None = None


class TrainingManager:
    """
    Different situation for resuming training:

    For pretrain:
    1. fresh -> {"mode": "fresh", "path": None}
    2. resume pretrain -> {"mode": "resume", "path": path_to_pretrain}

    For site-specific:
    1. fresh (start with pretrain weights) -> {"site": None, "mode": "fresh", "path": path_to_pretrain}
    2. resume site-specific / Mix -> [
                {"site": "bolivia", "mode": "resume", "path": path_to_bolivia},
        {"site": "mekong", "mode": "fresh", "path": None},
    ]

    Basically, we have:
    - pretrain run once, then run for each sites
    - for site-specific: can only either resume site-specific or start with pretrain weights (fresh)

    Pipeline trigger type:

    1. fresh pretrain -> fully fresh site-specific: need to pass pretrain path from pretrain step to site-specific step
    2. resume pretrain -> fully fresh site-specific: need to pass pretrain path from pretrain step to site-specific step
    3. skip pretrain entirely -> resume site-specific / fresh site-specific / both
    """

    def __init__(self, data_config: DataConfig, model_config: CLVAEConfig):
        self.data_config = data_config
        self.model_config = model_config
        self.wandb_project = "flood-detection-dl"
        self.wandb_pretrain_name = "clvae-pretrain"
        self.wandb_site_specific_name = "clvae-site-specific"
        self.wandb_tags = ["clvae"]
        self.wandb_group = "experiment-" + wandb.util.generate_id()

    def _get_current_datetime(self) -> str:
        return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=7))).strftime("%Y%m%d_%H%M%S")

    def pretrain(
        self,
        run_name: str | None = None,
        wandb_run: Run | None = None,
        resume_checkpoint: Path | str | None = None,
        **kwargs: Any,
    ) -> tuple[CLVAE, Path]:
        return pretrain(
            data_config=self.data_config,
            model_config=self.model_config,
            run_name=run_name,
            wandb_run=wandb_run,
            resume_checkpoint=resume_checkpoint,
            **kwargs,
        )

    def site_specific(
        self,
        site: str,
        run_name: str | None = None,
        wandb_run: Run | None = None,
        pretrained_model_path: Path | str | None = None,
        resume_checkpoint: Path | str | None = None,
        **kwargs: Any,
    ) -> tuple[CLVAE, Path]:
        return site_specific_train(
            data_config=self.data_config,
            model_config=self.model_config,
            run_name=run_name,
            site_name=site,
            wandb_run=wandb_run,
            pretrained_model_path=pretrained_model_path,
            resume_checkpoint=resume_checkpoint,
            **kwargs,
        )

    def predict(
        self,
        site: str,
        wandb_run: Run | None = None,
        model_path: Path | str | None = None,
    ) -> dict[str, np.ndarray]:
        return generate_distance_maps(
            site=site,
            data_config=self.data_config,
            model_config=self.model_config,
            model_path=model_path,
            wandb_run=wandb_run,
            log_latents=True,
        )

    def run(
        self,
        training_input: TrainingInput,
        use_wandb: bool = False,
        extra_tags: list[str] | None = None,
        notes: str | None = None,
    ):
        """
        How to set up TrainingInput:
        1. Pretrain:
            - mode: "fresh" or "resume"
            - path: path to pretrain checkpoint
            - kwargs: kwargs for pretrain to override model config
        2. Site-specific:
            - site: site name
            - mode: "fresh" or "resume"
            - path: path to site-specific checkpoint (if resume) or pretrain weights (if fresh and pretrain is skipped)
            - kwargs: kwargs for site-specific to override model config

        Example
        ---------

        ```python
        training_input_raw = {
            "pretrain": {
                "mode": "fresh",
                "path": None,
                "kwargs": {
                    "num_patches": 10,
                },
            },
            "site_specific": [
                {"site": "bolivia", "mode": "fresh", "path": None, "kwargs": {"max_epochs": 2}},
                {"site": "mekong", "mode": "fresh", "path": None, "kwargs": {"max_epochs": 2}},
                {"site": "somalia", "mode": "fresh", "path": None, "kwargs": {"max_epochs": 2}},
                {"site": "spain", "mode": "fresh", "path": None, "kwargs": {"max_epochs": 2}},
            ],
        }
        training_input = TrainingInput(**training_input_raw)
        ```
        """

        # step 1: pretrain

        pretrain_run_name = f"pretrain_{self._get_current_datetime()}"
        if training_input.pretrain:
            if training_input.pretrain.mode == TrainingMode.RESUME:
                print(f"Resuming pretrain from {training_input.pretrain.path}")
            else:
                print("Starting fresh pretrain")

            if any(
                site_specific_input.mode == TrainingMode.RESUME for site_specific_input in training_input.site_specific
            ):
                raise ValueError("Not allowed to resume pretrain if site-specific is also resuming")

            if use_wandb:
                with wandb.init(
                    project=self.wandb_project,
                    name=pretrain_run_name,
                    group=self.wandb_group,
                    job_type="pre-training",
                    tags=self.wandb_tags + (extra_tags or []),
                    notes=notes,
                ) as run:
                    _, model_info_path = self.pretrain(
                        run_name=pretrain_run_name,
                        wandb_run=run,
                        resume_checkpoint=training_input.pretrain.path,
                        **training_input.pretrain.kwargs,
                    )
            else:
                _, model_info_path = self.pretrain(
                    run_name=pretrain_run_name,
                    wandb_run=None,
                    resume_checkpoint=training_input.pretrain.path,
                    **training_input.pretrain.kwargs,
                )

            with open(model_info_path) as f:
                model_info = json.load(f)
            training_input.pretrain.path = model_info["checkpoint_path"]

        # step 2: site-specific training
        print("Starting site-specific")
        for site_specific_input in training_input.site_specific:
            site_name = site_specific_input.site
            site_specific_run_name = f"site_specific_{site_name}_{self._get_current_datetime()}"
            print(f"Starting site-specific for {site_name}")
            if site_specific_input.mode == TrainingMode.FRESH:
                print(f"Starting fresh site-specific for {site_name}")
                if training_input.pretrain:
                    if training_input.pretrain.path:
                        kwargs = {"pretrained_model_path": training_input.pretrain.path}
                    else:
                        raise ValueError(f"{site_name}: No pretrain path provided, cannot start fresh site-specific")
                else:
                    kwargs = {"resume_checkpoint": site_specific_input.path}

            elif site_specific_input.mode == TrainingMode.RESUME:
                print(f"Resuming site-specific for {site_name}")
                if not site_specific_input.path:
                    raise ValueError(f"{site_name}: No site-specific path provided, cannot resume site-specific")

                kwargs = {"resume_checkpoint": site_specific_input.path}

            kwargs.update(site_specific_input.kwargs)
            if use_wandb:
                with wandb.init(
                    project=self.wandb_project,
                    name=site_specific_run_name,
                    group=self.wandb_group,
                    job_type="training",
                    tags=self.wandb_tags + (extra_tags or []) + [site_name],
                    notes=notes,
                ) as run:
                    self.site_specific(site=site_name, run_name=site_specific_run_name, wandb_run=run, **kwargs)
            else:
                self.site_specific(site=site_name, run_name=site_specific_run_name, wandb_run=None, **kwargs)

        # step 3: predict
        print("Starting predict")
        for site_specific_input in training_input.site_specific:
            site_name = site_specific_input.site
            print(f"Starting prediction and eval for {site_name}")
            latest_run = get_site_specific_latest_run(site_name, self.data_config)
            model_info = get_best_model_info(self.data_config.artifact.site_specific_dir / latest_run)
            model_path = Path(model_info["checkpoint_path"])
            if use_wandb:
                with wandb.init(
                    project=self.wandb_project,
                    name=f"predict_{site_name}_{self._get_current_datetime()}",
                    group=self.wandb_group,
                    job_type="predict",
                    tags=self.wandb_tags + (extra_tags or []) + [site_name],
                    notes=notes,
                ) as run:
                    distance_maps = self.predict(site=site_name, wandb_run=run, model_path=model_path)
            else:
                distance_maps = self.predict(site=site_name, wandb_run=None, model_path=model_path)

            ground_truths = load_ground_truths(site=site_name, data_config=self.data_config)

            th_test_df = test_thresholds(
                thresholds=[
                    0.001,
                    0.005,
                    0.009,
                    0.01,
                    0.03,
                    0.05,
                    0.07,
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                    0.5,
                    0.7,
                    0.9,
                    0.95,
                    0.99,
                    1.0,
                    1.2,
                ],
                distance_maps=distance_maps,
                ground_truths=ground_truths,
            )

            if use_wandb:
                with wandb.init(
                    project=self.wandb_project,
                    name=f"eval_{site_name}_{self._get_current_datetime()}",
                    group=self.wandb_group,
                    job_type="eval",
                    tags=self.wandb_tags + (extra_tags or []) + [site_name],
                    notes=notes,
                ) as run:
                    th_test_df_wandb = wandb.Table(dataframe=th_test_df)
                    run.log(
                        {
                            f"{site_name}'s test metrics": th_test_df_wandb,
                        }
                    )
            th_test_df.to_csv(model_path.parent / "th_test.csv", index=False)


if __name__ == "__main__":
    from flood_detection_core.config import CLVAEConfig, DataConfig

    data_config = DataConfig.from_yaml("./flood-detection-core/yamls/data.yaml")
    model_config = CLVAEConfig.from_yaml("./flood-detection-core/yamls/model_clvae.yaml")

    training_input_raw = {
        "pretrain": {
            "mode": "fresh",
            "path": None,
            "kwargs": {
                "num_patches": 100,
            },
        },
        "site_specific": [
            {"site": "bolivia", "mode": "fresh", "path": None, "kwargs": {}},
            {"site": "mekong", "mode": "fresh", "path": None, "kwargs": {}},
            {"site": "somalia", "mode": "fresh", "path": None, "kwargs": {}},
            {"site": "spain", "mode": "fresh", "path": None, "kwargs": {}},
        ],
    }
    training_input = TrainingInput(**training_input_raw)

    print(training_input)

    # available wandb kwargs: https://docs.wandb.ai/ref/python/sdk/functions/init/
    training_manager = TrainingManager(data_config, model_config)
    training_manager.run(
        training_input=training_input,
    )
