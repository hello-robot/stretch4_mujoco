"""
Fit `StretchBCNet` to the demonstrations collected by `collect.py`.

Plain supervised behaviour cloning: minimise the smooth-L1 error between the
network's predicted action chunk and the expert's, on normalised actions. There
is no DAgger, no reward and no value function -- the goal is a learned policy
that plugs into the same benchmark harness as the expert so the two can be
compared on the eight evaluations.

The checkpoint is written self-describing (camera names, chunk size,
normalisation statistics) because `StretchBCPolicy` cannot detect a mismatch in
any of them at load time; it would just quietly behave badly.

Usage:
    python -m examples.machine_learning.molmospaces.training.train_bc \\
        --dataset-dir data/stretch_pick --output checkpoints/stretch_pick.pt
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import click
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from examples.machine_learning.molmospaces.policies.networks import StretchBCNet
from examples.machine_learning.molmospaces.training.dataset import StretchBCDataset

log = logging.getLogger(__name__)


def train(
    dataset_dir: Path,
    output_path: Path,
    chunk_size: int = 8,
    epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    validation_fraction: float = 0.1,
    num_workers: int = 4,
    device: str | None = None,
    seed: int = 0,
) -> Path:
    """Train a chunked behaviour-cloning policy and write a checkpoint.

    Args:
        dataset_dir: a directory built by `training/dataset.py`.
        output_path: where to write the checkpoint (a `.pt` file).
        chunk_size: how many future actions each prediction covers.
        epochs: passes over the training split.
        batch_size: samples per optimiser step.
        learning_rate: AdamW learning rate.
        weight_decay: AdamW weight decay.
        validation_fraction: fraction of *trajectories* held out. Splitting by
            trajectory rather than by transition matters: consecutive transitions
            within an episode are near-duplicates, so a transition-level split
            reports a validation loss that is really a training loss.
        num_workers: DataLoader workers.
        device: torch device, defaults to CUDA when available.
        seed: seed for the split and for initialisation.

    Returns:
        The checkpoint path.
    """
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = StretchBCDataset(dataset_dir, chunk_size=chunk_size)
    statistics = dataset.statistics()
    train_indices, validation_indices = _split_by_trajectory(dataset, validation_fraction, seed)
    log.info(
        f"[train] {len(dataset)} transitions from {dataset.metadata.num_trajectories} "
        f"trajectories -> {len(train_indices)} train / {len(validation_indices)} val"
    )

    normaliser = _Normaliser(statistics, device)
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    validation_loader = (
        DataLoader(
            Subset(dataset, validation_indices),
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        )
        if validation_indices
        else None
    )

    model = StretchBCNet(num_cameras=len(dataset.metadata.camera_names), chunk_size=chunk_size).to(
        device
    )
    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=max(epochs, 1))
    criterion = nn.SmoothL1Loss()

    best_validation = float("inf")
    history = []
    for epoch in range(epochs):
        train_loss = _run_epoch(model, train_loader, normaliser, criterion, device, optimiser)
        validation_loss = (
            _run_epoch(model, validation_loader, normaliser, criterion, device, None)
            if validation_loader is not None
            else train_loss
        )
        schedule.step()
        history.append(
            {
                "epoch": epoch,
                "train": train_loss,
                "validation": validation_loss,
                "learning_rate": schedule.get_last_lr()[0],
            }
        )
        log.info(
            f"[train] epoch {epoch + 1}/{epochs} train={train_loss:.5f} val={validation_loss:.5f}"
        )
        # Written every epoch, not at the end, so a run that is still going (or
        # that dies) can still be inspected.
        _write_training_artifacts(output_path, history)

        if validation_loss < best_validation:
            best_validation = validation_loss
            _save_checkpoint(
                output_path, model, dataset, chunk_size, statistics, epoch, validation_loss
            )

    log.info(f"[train] best validation loss {best_validation:.5f}; checkpoint at {output_path}")
    log.info(f"[train] curves at {output_path.parent / (output_path.stem + '_curves.png')}")
    return output_path


def _write_training_artifacts(output_path: Path, history: list[dict]) -> None:
    """Dump the loss history as JSON, CSV and a plot, beside the checkpoint.

    Three formats because they are read by three different things: the JSON by
    code, the CSV by a spreadsheet, and the PNG by a person who wants to see at
    a glance whether the run converged or diverged.
    """
    stem = output_path.stem
    directory = output_path.parent
    (directory / f"{stem}_history.json").write_text(json.dumps(history, indent=2))

    with (directory / f"{stem}_history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    try:
        import matplotlib

        # No display during training, and on a headless box the default backend
        # fails at import rather than at draw time.
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    epochs = [entry["epoch"] + 1 for entry in history]
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(epochs, [entry["train"] for entry in history], label="train")
    axis.plot(epochs, [entry["validation"] for entry in history], label="validation")
    axis.set_xlabel("epoch")
    axis.set_ylabel("smooth L1 loss (normalised actions)")
    axis.set_title(stem)
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(directory / f"{stem}_curves.png", dpi=120)
    plt.close(figure)


class _Normaliser:
    """Applies the dataset's state/action normalisation on-device."""

    def __init__(self, statistics: dict[str, np.ndarray], device: str) -> None:
        def as_tensor(key: str) -> torch.Tensor:
            return torch.from_numpy(statistics[key].astype(np.float32)).to(device)

        self.state_mean = as_tensor("state_mean")
        self.state_std = as_tensor("state_std")
        self.action_mean = as_tensor("action_mean")
        self.action_std = as_tensor("action_std")

    def state(self, states: torch.Tensor) -> torch.Tensor:
        return (states - self.state_mean) / self.state_std

    def action(self, actions: torch.Tensor) -> torch.Tensor:
        return (actions - self.action_mean) / self.action_std


def _run_epoch(model, loader, normaliser, criterion, device, optimiser) -> float:
    training = optimiser is not None
    model.train(training)
    total_loss, total_batches = 0.0, 0

    with torch.set_grad_enabled(training):
        for images, states, actions in loader:
            images = images.to(device, non_blocking=True)
            states = normaliser.state(states.to(device, non_blocking=True))
            targets = normaliser.action(actions.to(device, non_blocking=True))

            loss = criterion(model(images, states), targets)
            if training:
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()

            total_loss += float(loss.item())
            total_batches += 1

    return total_loss / max(total_batches, 1)


def _split_by_trajectory(
    dataset: StretchBCDataset, validation_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    """Hold out whole trajectories, not individual transitions."""
    shard_ids = np.array([shard_id for shard_id, _ in dataset._index])
    unique_shards = np.unique(shard_ids)

    generator = np.random.default_rng(seed)
    shuffled = generator.permutation(unique_shards)
    num_validation = int(round(len(shuffled) * validation_fraction))
    # Never hold out so much that training has nothing left, and never hold out
    # the only trajectory in a tiny smoke-test dataset.
    num_validation = min(num_validation, max(len(shuffled) - 1, 0))
    validation_shards = set(shuffled[:num_validation].tolist())

    train_indices, validation_indices = [], []
    for index, shard_id in enumerate(shard_ids):
        (validation_indices if shard_id in validation_shards else train_indices).append(index)
    return train_indices, validation_indices


def _save_checkpoint(
    path: Path,
    model: StretchBCNet,
    dataset: StretchBCDataset,
    chunk_size: int,
    statistics: dict[str, np.ndarray],
    epoch: int,
    validation_loss: float,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "camera_names": list(dataset.metadata.camera_names),
            "chunk_size": chunk_size,
            "normalisation": {key: value.tolist() for key, value in statistics.items()},
            "epoch": epoch,
            "validation_loss": validation_loss,
        },
        path,
    )


@click.command()
@click.option(
    "--dataset-dir",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Dataset directory built by training/collect.py.",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=Path("checkpoints/stretch_bc.pt"),
    help="Checkpoint path to write.",
)
@click.option("--chunk-size", type=int, default=8, help="Actions predicted per step.")
@click.option("--epochs", type=int, default=30)
@click.option("--batch-size", type=int, default=64)
@click.option("--learning-rate", type=float, default=3e-4)
@click.option("--validation-fraction", type=float, default=0.1)
@click.option("--num-workers", type=int, default=4)
@click.option("--device", type=str, default=None)
def main(
    dataset_dir: Path,
    output: Path,
    chunk_size: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    validation_fraction: float,
    num_workers: int,
    device: str | None,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    checkpoint = train(
        dataset_dir=dataset_dir,
        output_path=output,
        chunk_size=chunk_size,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        validation_fraction=validation_fraction,
        num_workers=num_workers,
        device=device,
    )
    click.secho(f"Checkpoint written to {checkpoint}", fg="green")
    click.echo(
        "Evaluate it with:\n"
        "  python -m examples.machine_learning.molmospaces.run_benchmarks \\\n"
        "      --policy bc --checkpoint "
        f"{checkpoint} --episodes 50"
    )


if __name__ == "__main__":
    main()
