from typing import Callable, cast

import click
import torch
import wandb
from torch.utils.data import DataLoader
from torch_geometric.datasets import MNISTSuperpixels

from gnn_image_classification.datasets import build_train_val_dataloaders
from gnn_image_classification.model import GNNImageClassificator
from gnn_image_classification.visualize_graphs import visualize


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
        predicted_classes = torch.argmax(logits, dim=1)

        loss = criterion(logits, classes).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accuracy = (predicted_classes == classes).to(torch.float32).mean()

        wandb.log({
            "train_accuracy": float(accuracy.detach().cpu().numpy()),
            "train_loss": float(loss.detach().cpu().numpy()),
            "batch": batches_passed,
        })

        batches_passed += 1

    return batches_passed


@torch.no_grad()
def evaluate(
    model: GNNImageClassificator,
    val_loader: DataLoader,
    epochs_passed: int,
) -> None:
    model.eval()

    accuracy_sum: float = 0.0
    num_samples: int = 0

    for batch in val_loader:
        batch_node_features = batch["batch_node_features"]
        batch_edge_indices = batch["batch_edge_indices"]
        classes = batch["classes"]

        logits = model(batch_node_features=batch_node_features, batch_edge_indices=batch_edge_indices)
        predicted_classes = torch.argmax(logits, dim=1)

        accuracy_sum += float((predicted_classes == classes).to(torch.float32).mean().cpu().numpy()) * len(classes)
        num_samples += len(classes)

    accuracy = accuracy_sum / num_samples

    wandb.log({
        "val_accuracy": accuracy,
        "epoch": epochs_passed,
    })


@click.command()
@click.option("--batch-size", type=int, default=64)
@click.option("--epochs", type=int, default=100)
@click.option("--device", type=str, default="cpu")
@click.option("--hidden-dim", type=int, default=152)
@click.option("--lr", type=float, default=1e-3)
def train(
    batch_size: int,
    epochs: int,
    device: str,
    hidden_dim: int,
    lr: float,
) -> None:
    wandb.init(project="cifar-10-gnn-classification")
    wandb.config.batch_size = batch_size
    wandb.config.epochs = epochs
    wandb.config.device = device
    wandb.config.hidden_dim = hidden_dim
    wandb.config.lr = lr

    wandb.define_metric("batch")
    wandb.define_metric("epoch")
    wandb.define_metric("train_accuracy", step_metric="batch")
    wandb.define_metric("train_loss", step_metric="batch")
    wandb.define_metric("val_accuracy", step_metric="epoch")

    model = GNNImageClassificator(in_channels=3, hidden_dim=hidden_dim).to(device)
    train_loader, val_loader = build_train_val_dataloaders(batch_size=batch_size, device=device)
    optimizer = torch.optim.Adam(lr=lr, params=model.parameters())

    # SAVE VISUALIZATION
    visualize(
        cast(MNISTSuperpixels, train_loader.dataset),
        image_name="all_classes.jpg",
    )

    visualize(
        cast(MNISTSuperpixels, train_loader.dataset),
        image_name="one_class.jpg",
        classes=(4,),
        examples_per_class=1,
    )

    wandb.log({
        "sample_images": [
            wandb.Image("all_classes.jpg"),
            wandb.Image("one_class.jpg"),
        ]
    })
    # SAVE VISUALIZATION END

    batches_passed = 0

    for epoch_ix in range(epochs):
        batches_passed = train_one_epoch(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            criterion=torch.nn.CrossEntropyLoss(),
            batches_passed=batches_passed,
        )

        evaluate(
            model=model,
            val_loader=val_loader,
            epochs_passed=epoch_ix + 1,
        )


if __name__ == "__main__":
    train()
