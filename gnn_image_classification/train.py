from typing import Callable

import click
import torch
import wandb
from torch.utils.data import DataLoader

from gnn_image_classification.datasets import build_train_val_dataloaders
from gnn_image_classification.model import GNNImageClassificator


def train_one_epoch(
    model: GNNImageClassificator,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    criterion: Callable,
    batches_passed: int,
) -> int:
    model.train()
    for batch in train_loader:
        batch_node_features = batch["batch_node_features"]
        batch_edge_indices = batch["batch_edge_indices"]
        classes = batch["classes"]

        logits = model(batch_node_features=batch_node_features, batch_edge_indices=batch_edge_indices)
        predicted_classes = torch.amax(logits, dim=1)

        loss = criterion(logits, classes).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accuracy = (predicted_classes == classes).float().mean()

        wandb.log({
            "train_accuracy": float(accuracy.detach().cpu().numpy()),
            "train_loss": float(loss.detach().cpu().numpy()),
            "batch": batches_passed,
        })

        batches_passed += 1

    return batches_passed


@click.command()
@click.option("--batch-size", type=int, default=64)
@click.option("--epochs", type=int, default=100)
@click.option("--device", type=str, default="cpu")
@click.option("--hidden-dim", type=int, default=152)
def train(
    batch_size: int,
    epochs: int,
    device: str,
    hidden_dim: int,
) -> None:
    wandb.init(project="cifar-10-gnn-classification")
    wandb.config.batch_size = batch_size
    wandb.config.epochs = epochs
    wandb.config.device = device
    wandb.config.hidden_dim = hidden_dim

    wandb.define_metric("batch")
    wandb.define_metric("epoch")
    wandb.define_metric("train_accuracy", step_metric="batch")
    wandb.define_metric("train_loss", step_metric="batch")
    wandb.define_metric("val_accuracy", step_metric="epoch")
    wandb.define_metric("val_loss", step_metric="epoch")

    model = GNNImageClassificator(in_channels=3, hidden_dim=hidden_dim).to(device)
    train_loader, val_loader = build_train_val_dataloaders(batch_size=batch_size, device=device)
    optimizer = torch.optim.Adam(lr=3e-4, params=model.parameters())

    batches_passed = 0

    for epochs_passed in range(epochs):
        batches_passed = train_one_epoch(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            criterion=torch.nn.CrossEntropyLoss(),
            batches_passed=batches_passed,
        )


if __name__ == "__main__":
    train()
