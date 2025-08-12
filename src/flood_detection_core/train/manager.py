import json
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from rich import print

import wandb
from wandb.sdk.wandb_run import Run
from .pretrain import pretrain
from .site_specific import site_specific_train
from flood_detection_core.config import CLVAEConfig, DataConfig
from flood_detection_core.models.clvae import CLVAE


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

    def pretrain(
        self,
        wandb_run: Run | None = None,
        resume_checkpoint: Path | str | None = None,
        **kwargs: Any,
    ) -> tuple[CLVAE, Path]:
        return pretrain(
            data_config=self.data_config,
            model_config=self.model_config,
            wandb_run=wandb_run,
            resume_checkpoint=resume_checkpoint,
            **kwargs,
        )

    def site_specific(
        self,
        site: str,
        wandb_run: Run | None = None,
        pretrained_model_path: Path | str | None = None,
        resume_checkpoint: Path | str | None = None,
        **kwargs: Any,
    ) -> tuple[CLVAE, Path]:
        return site_specific_train(
            data_config=self.data_config,
            model_config=self.model_config,
            site_name=site,
            wandb_run=wandb_run,
            pretrained_model_path=pretrained_model_path,
            resume_checkpoint=resume_checkpoint,
            **kwargs,
        )

    def run(
        self,
        training_input: TrainingInput,
        use_wandb: bool = False,
        pretrain_extra_tags: list[str] | None = None,
        site_specific_extra_tags: list[str] | None = None,
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
                    name=self.wandb_pretrain_name,
                    tags=self.wandb_tags + ["pretrain"] + (pretrain_extra_tags or []),
                ) as run:
                    _, model_info_path = self.pretrain(
                        wandb_run=run, resume_checkpoint=training_input.pretrain.path, **training_input.pretrain.kwargs
                    )
            else:
                _, model_info_path = self.pretrain(
                    resume_checkpoint=training_input.pretrain.path, **training_input.pretrain.kwargs
                )

            with open(model_info_path) as f:
                model_info = json.load(f)
            training_input.pretrain.path = model_info["checkpoint_path"]

        print("Starting site-specific")
        for site_specific_input in training_input.site_specific:
            site_name = site_specific_input.site
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
                    name=self.wandb_site_specific_name,
                    tags=self.wandb_tags + ["site-specific", site_name] + (site_specific_extra_tags or []),
                ) as run:
                    self.site_specific(site=site_name, wandb_run=run, **kwargs)
            else:
                self.site_specific(site=site_name, **kwargs)


if __name__ == "__main__":
    from flood_detection_core.config import CLVAEConfig, DataConfig

    data_config = DataConfig.from_yaml("./yamls/data.yaml")
    model_config = CLVAEConfig.from_yaml("./yamls/model_clvae.yaml")

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
