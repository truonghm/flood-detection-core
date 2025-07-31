# TODO: scale with multi-gpu setup

import datetime
import json
from pathlib import Path
from typing import Any

import torch
from rich import print
from rich.progress import Progress
from torch.utils.data import DataLoader

import wandb
from wandb.sdk.wandb_run import Run
from flood_detection_core.config import CLVAEConfig, DataConfig
from flood_detection_core.data.datasets import FloodEventDataset
from flood_detection_core.data.processing.augmentation import augment_data
from flood_detection_core.models.clvae import CLVAE


def site_specific_train(
    data_config: DataConfig,
    model_config: CLVAEConfig,
    site_name: str,
    wandb_run: Run | None = None,
    debug: bool = False,
    pretrained_model_path: Path | str | None = None,
    resume_checkpoint: Path | str | None = None,
    **kwargs: Any,
) -> tuple[CLVAE, Path]:
    """
    Site-specific fine-tuning for each flood event using contrastive learning.

    **Purpose**: Site-specific fine-tuning for each flood event
    **Logic**:
    - Loads pre-trained weights from PreTrainingPipeline
    - Uses FloodEventDataset for specific flood site
    - 25 epochs per flood site with early stopping (patience 5)
    - Learning rate: 0.0001 → 0.000001 (patience 3)
    - Implements contrastive learning between pre/post flood representations using basic pairs from dataset
    - Saves site-specific model checkpoints

    Parameters
    ----------
    data_config : DataConfig
        Data configuration containing paths and settings
    model_config : CLVAEConfig
        Model configuration with hyperparameters
    site_name : str
        Name of the flood site for training
    wandb_run : Run | None
        Optional W&B run for logging
    debug : bool
        Enable debug mode for testing
    pretrained_model_path : str | Path | None
        Path to pretrained model checkpoint (mutually exclusive with resume_checkpoint)
    resume_checkpoint : str | Path | None
        Path to resume checkpoint (mutually exclusive with pretrained_model_path)
    **kwargs : Any
        Additional keyword arguments to override config parameters

    Returns
    -------
    CLVAE
        Fine-tuned model for the specific site
    """
    # Validation: exactly one of pretrained_model_path or resume_checkpoint must be provided
    if pretrained_model_path and resume_checkpoint:
        raise ValueError("Cannot provide both pretrained_model_path and resume_checkpoint. Use only one.")
    if not pretrained_model_path and not resume_checkpoint:
        raise ValueError("Must provide either pretrained_model_path or resume_checkpoint.")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not data_config.artifact.site_specific_dir.exists():
        data_config.artifact.site_specific_dir.mkdir(parents=True, exist_ok=True)

    config = dict(
        input_channels=kwargs.get("input_channels", model_config.site_specific.input_channels),
        hidden_channels=kwargs.get("hidden_channels", model_config.site_specific.hidden_channels),
        latent_dim=kwargs.get("latent_dim", model_config.site_specific.latent_dim),
        learning_rate=kwargs.get("learning_rate", model_config.site_specific.learning_rate),
        max_epochs=kwargs.get("max_epochs", model_config.site_specific.max_epochs),
        scheduler_patience=kwargs.get("scheduler_patience", model_config.site_specific.scheduler_patience),
        scheduler_factor=kwargs.get("scheduler_factor", model_config.site_specific.scheduler_factor),
        scheduler_min_lr=kwargs.get("scheduler_min_lr", model_config.site_specific.scheduler_min_lr),
        early_stopping_patience=kwargs.get(
            "early_stopping_patience", model_config.site_specific.early_stopping_patience
        ),
        batch_size=kwargs.get("batch_size", model_config.site_specific.batch_size),
        num_temporal_length=kwargs.get("num_temporal_length", model_config.site_specific.num_temporal_length),
        patch_size=kwargs.get("patch_size", model_config.site_specific.patch_size),
        patch_stride=kwargs.get("patch_stride", model_config.site_specific.patch_stride),
        vv_clipped_range=kwargs.get("vv_clipped_range", model_config.site_specific.vv_clipped_range),
        vh_clipped_range=kwargs.get("vh_clipped_range", model_config.site_specific.vh_clipped_range),
        site_name=site_name,
    )

    if wandb_run:
        wandb_run.config.update(config)

    print("Config:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    current_date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = data_config.artifact.site_specific_dir / f"site_specific_{site_name}_{current_date}"
    model_dir.mkdir(parents=True, exist_ok=True)

    with open(model_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    model = CLVAE(
        input_channels=config["input_channels"],
        hidden_channels=config["hidden_channels"],
        latent_dim=config["latent_dim"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=config["scheduler_patience"],
        factor=config["scheduler_factor"],
        min_lr=config["scheduler_min_lr"],
    )

    if pretrained_model_path:
        pretrained_path = Path(pretrained_model_path)
        if not pretrained_path.exists():
            raise FileNotFoundError(f"Pretrained model not found at {pretrained_path}")

        print(f"Loading pretrained weights from {pretrained_path}")
        state_dict = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(state_dict["model_state"])

    train_dataset = FloodEventDataset(
        dataset_type="train",
        pre_flood_split_csv_path=data_config.splits.pre_flood_split,
        post_flood_split_csv_path=data_config.splits.post_flood_split,
        sites=[site_name],
        num_temporal_length=config["num_temporal_length"],
        patch_size=config["patch_size"],
        patch_stride=config["patch_stride"],
        transform=lambda x: augment_data(x, model_config.augmentation, False),
        vv_clipped_range=config.get("vv_clipped_range", model_config.site_specific.vv_clipped_range),
        vh_clipped_range=config.get("vh_clipped_range", model_config.site_specific.vh_clipped_range),
    )
    val_dataset = FloodEventDataset(
        dataset_type="val",
        pre_flood_split_csv_path=data_config.splits.pre_flood_split,
        post_flood_split_csv_path=data_config.splits.post_flood_split,
        sites=[site_name],
        num_temporal_length=config["num_temporal_length"],
        patch_size=config["patch_size"],
        patch_stride=config["patch_stride"],
        transform=lambda x: augment_data(x, model_config.augmentation, False),
        vv_clipped_range=config.get("vv_clipped_range", model_config.site_specific.vv_clipped_range),
        vh_clipped_range=config.get("vh_clipped_range", model_config.site_specific.vh_clipped_range),
    )
    train_size = len(train_dataset)
    val_size = len(val_dataset)
    print(f"Site {site_name} - Train size: {train_size}, Val size: {val_size}")

    if train_size == 0 or val_size == 0:
        raise ValueError(f"Insufficient data for site {site_name}. Total samples: {len(train_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0
    patience = config["early_stopping_patience"]

    if resume_checkpoint:
        resume_checkpoint = Path(resume_checkpoint)
        if resume_checkpoint.exists():
            checkpoint = torch.load(resume_checkpoint, map_location=device)
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            start_epoch = checkpoint["epoch"] + 1
            best_val_loss = checkpoint["best_val_loss"]
            patience_counter = checkpoint["patience_counter"]
            print(f"Resuming training for {site_name} from epoch {start_epoch} with best val loss {best_val_loss}")

    print(f"Starting site-specific training for {site_name}")

    with Progress() as progress:
        outer = progress.add_task(f"Site-specific training for {site_name}", total=config["max_epochs"])
        train_task = progress.add_task("Training", total=len(train_loader))
        val_task = progress.add_task("Validation", total=len(val_loader))

        for epoch in range(start_epoch, config["max_epochs"]):
            model.train()
            train_loss = 0.0
            train_recon_loss = 0.0
            train_kl_loss = 0.0
            train_contrastive_loss = 0.0

            # reset progress bar
            progress.update(train_task, completed=0)
            progress.update(val_task, completed=0)
            for batch_idx, (pre_flood_seq, post_flood_img) in enumerate(train_loader):
                pre_flood_seq = pre_flood_seq.to(device)  # (B, T, H, W, C)
                post_flood_img = post_flood_img.to(device)  # (B, 1, H, W, C)

                # TODO: remove this once we have a better way to handle the temporal dimension
                # For site-specific training, we focus on the last 4 frames of pre-flood sequence
                # to match the pretrained model input format
                if pre_flood_seq.shape[1] > 4:
                    pre_flood_input = pre_flood_seq[:, -4:, :, :, :]  # Take last 4 frames
                else:
                    pre_flood_input = pre_flood_seq

                # Squeeze post-flood to match pre-flood temporal dimension for processing
                post_flood_input = post_flood_img.squeeze(1)  # (B, H, W, C)
                # Replicate post-flood image to create 4-frame sequence for model input
                post_flood_input = post_flood_input.unsqueeze(1).repeat(1, 4, 1, 1, 1)  # (B, 4, H, W, C)

                pre_mu, pre_logvar = model.encode(pre_flood_input)
                post_mu, post_logvar = model.encode(post_flood_input)

                x_recon, mu, logvar, z = model(pre_flood_input)

                loss_dict = model.compute_loss(
                    pre_flood_input,
                    x_recon,
                    mu,
                    logvar,
                    contrastive_pairs=(pre_mu, post_mu),
                )

                loss = loss_dict["total_loss"]

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()
                train_recon_loss += loss_dict["reconstruction_loss"].item()
                train_kl_loss += loss_dict["kl_loss"].item()
                train_contrastive_loss += loss_dict["contrastive_loss"].item()

                if debug:
                    print("DEBUG - Loss components:")
                    for k, v in loss_dict.items():
                        print(f"  {k}: {v.item():.6f}, NaN: {torch.isnan(v).any()}")
                    return model

                progress.update(train_task, advance=1)

            model.eval()
            val_loss = 0.0
            val_recon_loss = 0.0
            val_kl_loss = 0.0
            val_contrastive_loss = 0.0

            with torch.no_grad():
                for batch_idx, (pre_flood_seq, post_flood_img) in enumerate(val_loader):
                    pre_flood_seq = pre_flood_seq.to(device)
                    post_flood_img = post_flood_img.to(device)

                    if pre_flood_seq.shape[1] > 4:
                        pre_flood_input = pre_flood_seq[:, -4:, :, :, :]
                    else:
                        pre_flood_input = pre_flood_seq

                    post_flood_input = post_flood_img.squeeze(1)
                    post_flood_input = post_flood_input.unsqueeze(1).repeat(1, 4, 1, 1, 1)

                    pre_mu, pre_logvar = model.encode(pre_flood_input)
                    post_mu, post_logvar = model.encode(post_flood_input)
                    x_recon, mu, logvar, z = model(pre_flood_input)

                    loss_dict = model.compute_loss(
                        pre_flood_input,
                        x_recon,
                        mu,
                        logvar,
                        contrastive_pairs=(pre_mu, post_mu),
                    )

                    val_loss += loss_dict["total_loss"].item()
                    val_recon_loss += loss_dict["reconstruction_loss"].item()
                    val_kl_loss += loss_dict["kl_loss"].item()
                    val_contrastive_loss += loss_dict["contrastive_loss"].item()

                    progress.update(val_task, advance=1)

            train_loss /= len(train_loader)
            train_recon_loss /= len(train_loader)
            train_kl_loss /= len(train_loader)
            train_contrastive_loss /= len(train_loader)

            val_loss /= len(val_loader)
            val_recon_loss /= len(val_loader)
            val_kl_loss /= len(val_loader)
            val_contrastive_loss /= len(val_loader)

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                checkpoint_path = model_dir / f"site_specific_model_{site_name}_epoch_{epoch}.pth"
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "best_val_loss": val_loss,
                        "epoch": epoch,
                        "patience_counter": patience_counter,
                    },
                    checkpoint_path,
                )

                best_model_info = {
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "checkpoint_path": str(checkpoint_path),
                    "site_name": site_name,
                }
                with open(model_dir / "best_model_info.json", "w") as f:
                    json.dump(best_model_info, f, indent=2)
            else:
                patience_counter += 1

            # Logging
            if epoch % 5 == 0 or epoch == config["max_epochs"] - 1:
                print(f"Epoch {epoch}/{config['max_epochs']}")

                print(f"  Train Loss: {train_loss:.6f}")
                print(f"  Train Recon Loss: {train_recon_loss:.6f}")
                print(f"  Train KL Loss: {train_kl_loss:.6f}")
                print(f"  Train Contrastive Loss: {train_contrastive_loss:.6f}")
                print(f"  Val Loss: {val_loss:.6f}")
                print(f"  Val Recon Loss: {val_recon_loss:.6f}")
                print(f"  Val KL Loss: {val_kl_loss:.6f}")
                print(f"  Val Contrastive Loss: {val_contrastive_loss:.6f}")
                print(f"  LR: {optimizer.param_groups[0]['lr']:.8f}")

            if wandb_run:
                wandb_run.log(
                    {
                        f"{site_name}/train_loss": train_loss,
                        f"{site_name}/train_recon_loss": train_recon_loss,
                        f"{site_name}/train_kl_loss": train_kl_loss,
                        f"{site_name}/train_contrastive_loss": train_contrastive_loss,
                        f"{site_name}/val_loss": val_loss,
                        f"{site_name}/val_recon_loss": val_recon_loss,
                        f"{site_name}/val_kl_loss": val_kl_loss,
                        f"{site_name}/val_contrastive_loss": val_contrastive_loss,
                        f"{site_name}/learning_rate": optimizer.param_groups[0]["lr"],
                        "epoch": epoch,
                    }
                )

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch} for site {site_name}")
                break
            progress.update(outer, advance=1)

    print(f"Site-specific training completed for {site_name}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Model saved in: {model_dir}")

    return model, model_dir / "best_model_info.json"


if __name__ == "__main__":
    from flood_detection_core.config import CLVAEConfig, DataConfig

    use_wandb = False
    data_config = DataConfig.from_yaml("./yamls/data.yaml")
    model_config = CLVAEConfig.from_yaml("./yamls/model_clvae.yaml")

    pretrained_model_path = data_config.artifact.pretrain_dir / "pretrain_20250729_180639" / "pretrained_model_49.pth"
    site_name = "bolivia"

    test_kwargs = {
        "max_epochs": 2,
        "batch_size": 64,
        "early_stopping_patience": 2,
        "scheduler_patience": 1,
    }
    if use_wandb:
        with wandb.init(
            project="flood-detection-dl", name=f"clvae-site-specific-{site_name}", tags=["clvae", "site-specific"]
        ) as run:
            model = site_specific_train(
                data_config,
                model_config,
                site_name,
                wandb_run=run,
                debug=False,
                pretrained_model_path=pretrained_model_path,
                **test_kwargs,
            )
    else:
        model = site_specific_train(
            data_config,
            model_config,
            site_name,
            debug=False,
            pretrained_model_path=pretrained_model_path,
            **test_kwargs,
        )
