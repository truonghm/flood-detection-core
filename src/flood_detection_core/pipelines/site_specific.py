# TODO: scale with multi-gpu setup

import json
from pathlib import Path
from typing import Any

import torch
from rich import print
from rich.progress import Progress
from torch.utils.data import DataLoader

import wandb
from wandb.sdk.wandb_run import Run
from .utils import get_site_specific_run_name
from flood_detection_core.config import CLVAEConfig, DataConfig
from flood_detection_core.data.datasets import SiteSpecificTrainingDataset
from flood_detection_core.models.clvae import CLVAE


def site_specific_train(
    data_config: DataConfig,
    model_config: CLVAEConfig,
    site_name: str,
    run_name: str | None = None,
    wandb_run: Run | None = None,
    debug: bool = False,
    pretrained_model_path: Path | str | None = None,
    resume_checkpoint: Path | str | None = None,
    **kwargs: Any,
) -> tuple[CLVAE, Path]:
    """Site-specific fine-tuning for each flood event using contrastive learning.

    Purpose
    -------
        Site-specific fine-tuning for each flood event

    Logic
    -----
    - Loads pre-trained weights from PreTrainingPipeline
    - Uses SiteSpecificTrainingDataset (pre-flood only) for the specified site
    - 25 epochs per flood site with early stopping (patience 5)
    - Learning rate: 0.0001 → 0.000001 (patience 3)
    - Implements contrastive learning between two pre-flood sequences from different tiles
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
        vv_clipped_range=kwargs.get("vv_clipped_range", model_config.site_specific.vv_clipped_range),
        vh_clipped_range=kwargs.get("vh_clipped_range", model_config.site_specific.vh_clipped_range),
        site_name=site_name,
        # Loss weights from paper
        alpha=kwargs.get("alpha", model_config.site_specific.alpha),
        beta=kwargs.get("beta", model_config.site_specific.beta),
    )

    if wandb_run:
        wandb_run.config.update(config)

    print("Config:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    run_name = run_name or get_site_specific_run_name(site_name)
    model_dir = data_config.artifact.site_specific_dir / run_name
    model_dir.mkdir(parents=True, exist_ok=True)

    with open(model_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    model = CLVAE(
        input_channels=config["input_channels"],
        hidden_channels=config["hidden_channels"],
        latent_dim=config["latent_dim"],
        alpha=config["alpha"],
        beta=config["beta"],
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

    dataset = SiteSpecificTrainingDataset(
        pre_flood_split_csv_path=data_config.splits.pre_flood_split,
        sites=[site_name],
        num_temporal_length=config["num_temporal_length"],
        patch_size=config["patch_size"],
        augmentation_config=model_config.augmentation,
        use_contrastive_pairing_rules=True,
        positive_pair_ratio=0.5,
        max_patches_per_pair="all",
        vv_clipped_range=config["vv_clipped_range"],
        vh_clipped_range=config["vh_clipped_range"],
    )
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.85, 0.15])
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
            for batch_idx, batch in enumerate(train_loader):
                # Batch may include labels when using pairing rules
                if isinstance(batch, list | tuple) and len(batch) == 3:
                    seq_a, seq_b, _ = batch
                else:
                    seq_a, seq_b = batch

                seq_a = seq_a.to(device)  # (B, T, H, W, C)
                seq_b = seq_b.to(device)

                # Model now supports variable T; dataset provides sequences with
                # config["num_temporal_length"], so no trimming is needed.

                # Forward both streams
                x1_recon, mu1, logvar1, z1 = model(seq_a)
                x2_recon, mu2, logvar2, z2 = model(seq_b)

                # Compute combined loss with contrastive on latent representations
                combined_loss = model.compute_loss(
                    x=seq_a, reconstruction=x1_recon, mu=mu1, logvar=logvar1, z=z1, contrastive_z=z2
                )

                # Add the second stream's reconstruction and KL losses (symmetric treatment)
                comp2 = model.compute_loss(seq_b, x2_recon, mu2, logvar2, z=None, contrastive_z=None)

                # Combine both streams symmetrically as in paper Eq. (1)
                recon_loss = 0.5 * (combined_loss["reconstruction_loss"] + comp2["reconstruction_loss"])
                kl_loss = 0.5 * (combined_loss["kl_loss"] + comp2["kl_loss"])
                contrastive_loss = combined_loss["contrastive_loss"]  # Computed on latent representations

                contrastive_weight = 1 - model.alpha - model.beta
                loss = model.alpha * kl_loss + model.beta * recon_loss + contrastive_weight * contrastive_loss

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()
                train_recon_loss += recon_loss.item()
                train_kl_loss += kl_loss.item()
                train_contrastive_loss += contrastive_loss.item()

                if debug:
                    print("DEBUG - Loss components:")
                    debug_losses = {
                        "total_loss": loss,
                        "reconstruction_loss": recon_loss,
                        "kl_loss": kl_loss,
                        "contrastive_loss": contrastive_loss,
                    }
                    for k, v in debug_losses.items():
                        print(f"  {k}: {v.item():.6f}, NaN: {torch.isnan(v).any()}")
                    return model

                progress.update(train_task, advance=1)
                # break

            # break
            model.eval()
            val_loss = 0.0
            val_recon_loss = 0.0
            val_kl_loss = 0.0
            val_contrastive_loss = 0.0

            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    if isinstance(batch, list | tuple) and len(batch) == 3:
                        seq_a, seq_b, _ = batch
                    else:
                        seq_a, seq_b = batch

                    seq_a = seq_a.to(device)
                    seq_b = seq_b.to(device)

                    # Model supports variable T; no trimming.

                    x1_recon, mu1, logvar1, z1 = model(seq_a)
                    x2_recon, mu2, logvar2, z2 = model(seq_b)

                    # Compute combined loss with contrastive on latent representations
                    combined_loss = model.compute_loss(
                        x=seq_a, reconstruction=x1_recon, mu=mu1, logvar=logvar1, z=z1, contrastive_z=z2
                    )

                    # Add the second stream's reconstruction and KL losses
                    comp2 = model.compute_loss(seq_b, x2_recon, mu2, logvar2, z=None, contrastive_z=None)

                    recon_loss = 0.5 * (combined_loss["reconstruction_loss"] + comp2["reconstruction_loss"])
                    kl_loss = 0.5 * (combined_loss["kl_loss"] + comp2["kl_loss"])
                    contrastive_loss = combined_loss["contrastive_loss"]

                    contrastive_weight = 1 - model.alpha - model.beta
                    total = model.alpha * kl_loss + model.beta * recon_loss + contrastive_weight * contrastive_loss

                    val_loss += total.item()
                    val_recon_loss += recon_loss.item()
                    val_kl_loss += kl_loss.item()
                    val_contrastive_loss += contrastive_loss.item()

                    progress.update(val_task, advance=1)
                    # break

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
                print(f"  KL Weight (α): {model.alpha:.6f}")

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
                        "train_loss": train_loss,
                        "train_recon_loss": train_recon_loss,
                        "train_kl_loss": train_kl_loss,
                        "train_contrastive_loss": train_contrastive_loss,
                        "val_loss": val_loss,
                        "val_recon_loss": val_recon_loss,
                        "val_kl_loss": val_kl_loss,
                        "val_contrastive_loss": val_contrastive_loss,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "kl_weight_alpha": model.alpha,
                        "epoch": epoch,
                    }
                )
            with open(model_dir / "loss_log.csv", "a") as f:
                # write header if file is empty
                if f.tell() == 0:
                    f.write(
                        "epoch,kl_weight_alpha,train_loss,train_recon_loss,train_kl_loss,train_contrastive_loss,val_loss,val_recon_loss,val_kl_loss,val_contrastive_loss\n"
                    )
                f.write(
                    f"{epoch},{model.alpha},{train_loss},{train_recon_loss},{train_kl_loss},{train_contrastive_loss},{val_loss},{val_recon_loss},{val_kl_loss},{val_contrastive_loss}\n"
                )

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch} for site {site_name}")
                break
            progress.update(outer, advance=1)

    print(f"Site-specific training completed for {site_name}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Model saved in: {model_dir}")

    with open(data_config.artifact.site_specific_dir / f"{site_name}_latest_run.txt", "w") as f:
        f.write(model_dir.name)

    if wandb_run:
        with open(model_dir / "best_model_info.json") as f:
            best_model_info = json.load(f)
        checkpoint_path = Path(best_model_info["checkpoint_path"])
        artifact = wandb.Artifact("site_specific", type="model", metadata=best_model_info)
        artifact.add_file(
            local_path=checkpoint_path,
            name=f"site_specific/{checkpoint_path.name}",
        )
        wandb_run.log_artifact(artifact)

    return model, model_dir / "best_model_info.json"


if __name__ == "__main__":
    from flood_detection_core.config import CLVAEConfig, DataConfig

    use_wandb = False
    data_config = DataConfig.from_yaml("./yamls/data.yaml")
    model_config = CLVAEConfig.from_yaml("./yamls/model_clvae.yaml")

    pretrained_model_path = "artifacts/pretrain/pretrain_20250814_091623/pretrained_model_48.pth"
    site_name = "bolivia"

    test_kwargs = {
        "max_epochs": 2,
        "batch_size": 128,
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
