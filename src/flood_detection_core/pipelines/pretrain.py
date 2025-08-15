# TODO: scale with multi-gpu setup

import json
from pathlib import Path
from typing import Any

import torch
from rich import print
from rich.progress import track
from torch.utils.data import DataLoader

import wandb
from wandb.sdk.wandb_run import Run
from .utils import get_pretrain_run_name
from flood_detection_core.config import CLVAEConfig, DataConfig
from flood_detection_core.data.datasets import PretrainDataset
from flood_detection_core.data.processing.augmentation import augment_data
from flood_detection_core.models.clvae import CLVAE


def pretrain(
    data_config: DataConfig,
    model_config: CLVAEConfig,
    run_name: str | None = None,
    wandb_run: Run | None = None,
    resume_checkpoint: Path | str | None = None,
    **kwargs: Any,
) -> tuple[CLVAE, Path]:
    """
    **Purpose**: Handles pre-training phase on pre-flood SAR images
    **Logic**:
    - Loads PreTrainDataset with 100 random patches from various sites
    - 50 epochs training with early stopping (patience 10)
    - Learning rate schedule: 0.001 → 0.00001 (reduce on plateau, patience 5)
    - Saves best model checkpoint based on validation loss
    - Focuses on learning normal SAR patterns without flood conditions
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not data_config.artifact.pretrain_dir.exists():
        data_config.artifact.pretrain_dir.mkdir(parents=True, exist_ok=True)

    config = dict(
        input_channels=kwargs.get("input_channels", model_config.pretrain.input_channels),
        hidden_channels=kwargs.get("hidden_channels", model_config.pretrain.hidden_channels),
        latent_dim=kwargs.get("latent_dim", model_config.pretrain.latent_dim),
        learning_rate=kwargs.get("learning_rate", model_config.pretrain.learning_rate),
        max_epochs=kwargs.get("max_epochs", model_config.pretrain.max_epochs),
        scheduler_patience=kwargs.get("scheduler_patience", model_config.pretrain.scheduler_patience),
        scheduler_factor=kwargs.get("scheduler_factor", model_config.pretrain.scheduler_factor),
        scheduler_min_lr=kwargs.get("scheduler_min_lr", model_config.pretrain.scheduler_min_lr),
        batch_size=kwargs.get("batch_size", model_config.pretrain.batch_size),
        num_patches=kwargs.get("num_patches", model_config.pretrain.num_patches),
        num_temporal_length=kwargs.get("num_temporal_length", model_config.pretrain.num_temporal_length),
        patch_size=kwargs.get("patch_size", model_config.pretrain.patch_size),
        early_stopping_patience=kwargs.get("early_stopping_patience", model_config.pretrain.early_stopping_patience),
    )
    if wandb_run:
        wandb_run.config.update(config)

    run_name = run_name or get_pretrain_run_name()
    model_dir = data_config.artifact.pretrain_dir / run_name
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / "config.json", "w") as f:
        json.dump(config, f)

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

    dataset = PretrainDataset(
        split_csv_path=data_config.splits.pre_flood_split,
        pretrain_dir=data_config.gee.pretrain_dir,
        num_patches=config["num_patches"],
        num_temporal_length=config["num_temporal_length"],
        patch_size=config["patch_size"],
        transform=lambda x: augment_data(x, model_config.augmentation, False),
    )
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    print(f"Train size: {train_size}, Val size: {val_size}")
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=True)

    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0
    patience = config["early_stopping_patience"]

    if resume_checkpoint:
        resume_checkpoint = Path(resume_checkpoint)
        if resume_checkpoint.exists():
            checkpoint = torch.load(resume_checkpoint)
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            start_epoch = checkpoint["epoch"] + 1
            best_val_loss = checkpoint["best_val_loss"]
            patience_counter = checkpoint["patience_counter"]
            print(f"Resuming pre-training from epoch {start_epoch} with best val loss {best_val_loss}")

    for epoch in track(range(start_epoch, config["max_epochs"]), description="Pre-training:"):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            x = batch.to(device)
            x_recon, mu, logvar, z = model(x)

            loss_dict = model.compute_loss(x, x_recon, mu, logvar)
            loss = loss_dict["total_loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch.to(device)
                x_recon, mu, logvar, z = model(x)

                loss_dict = model.compute_loss(x, x_recon, mu, logvar)
                val_loss += loss_dict["total_loss"].item()

        val_loss /= len(val_loader)
        train_loss /= len(train_loader)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            checkpoint_path = model_dir / f"pretrained_model_{epoch}.pth"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "best_val_loss": val_loss,
                    "epoch": epoch,
                    "patience_counter": patience_counter,
                },
                checkpoint_path
            )

            best_model_info = {
                "epoch": epoch,
                "val_loss": val_loss,
                "checkpoint_path": str(checkpoint_path),
                "train_loss": train_loss,
            }
            with open(model_dir / "best_model_info.json", "w") as f:
                json.dump(best_model_info, f, indent=2)
        else:
            patience_counter += 1

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            if wandb_run:
                wandb_run.log({"train_loss": train_loss, "val_loss": val_loss})
            with open(model_dir / "loss_log.csv", "a") as f:
                if f.tell() == 0:
                    f.write(
                        "epoch,train_loss,val_loss\n"
                    )
                f.write(
                    f"{epoch},{train_loss},{val_loss}\n"
                )

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    print("Finish pre-training, best model info is saved at:")
    print(model_dir / "best_model_info.json")

    with open(data_config.artifact.pretrain_dir / "latest_run.txt", "w") as f:
        f.write(model_dir.name)

    if wandb_run:
        with open(model_dir / "best_model_info.json") as f:
            best_model_info = json.load(f)
        checkpoint_path = Path(best_model_info["checkpoint_path"])
        artifact = wandb.Artifact("pretrain", type="model", metadata=best_model_info)
        artifact.add_file(
            local_path=checkpoint_path,
            name=f"pretrain/{checkpoint_path.name}",
        )
        wandb_run.log_artifact(artifact)

    return model, model_dir / "best_model_info.json"


if __name__ == "__main__":
    data_config = DataConfig.from_yaml("./yamls/data.yaml")
    model_config = CLVAEConfig.from_yaml("./yamls/model_clvae.yaml")

    use_wandb = False

    # # set num_patches to 10 for testing
    if use_wandb:
        with wandb.init(project="flood-detection-dl", name="clvae-pretrain", tags=["clvae", "test"]) as run:
            pretrain(data_config, model_config, wandb_run=run, num_patches=10)
    else:
        pretrain(data_config, model_config, num_patches=10)
