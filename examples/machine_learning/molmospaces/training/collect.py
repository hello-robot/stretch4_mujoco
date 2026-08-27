"""
Turn generated rollouts into behaviour-cloning shards.

The demonstrations themselves come from `finetuning/generate_dataset.py`, which
drives MolmoSpaces' data generation pipeline. That is the only generator here on
purpose: it samples tasks procedurally from the training splits, whereas the
benchmark's own episode list is the *test set*, and a policy cloned off the
episodes it is then scored on measures memorisation rather than skill. Both
roads -- behaviour cloning here, VLA fine-tuning in `finetuning/` -- therefore
learn from the same data.

So this module is only the second half of the old pipeline: rollouts on disk ->
`.npz` shards `train_bc.py` can read.

Usage:
    # 1. generate (once; the expensive part)
    python -m examples.machine_learning.molmospaces.finetuning.generate_dataset \\
        --task pick --task pnp --episodes 2000 --num-workers 8 \\
        --output-dir data/stretch_manip --no-export

    # 2. build the BC dataset from what it wrote
    python -m examples.machine_learning.molmospaces.training.collect \\
        --rollouts data/stretch_manip/rollouts --output-dir data/stretch_manip/bc

`--rollouts` is repeatable, which is how per-task runs are pooled into one
dataset. Each path is searched recursively for `house_*/trajectories*.h5`, so it
can be either a single task's directory or the parent holding several.
"""

from __future__ import annotations

import logging
from pathlib import Path

import click

from examples.machine_learning.molmospaces.stretch.config import HEAD_CAMERA, WRIST_CAMERA_LEFT
from examples.machine_learning.molmospaces.training.dataset import build_dataset

log = logging.getLogger(__name__)


def collect_demonstrations(
    rollout_dirs: list[Path],
    output_dir: Path,
    successful_only: bool = True,
) -> Path:
    """Build a behaviour-cloning dataset from recorded rollouts.

    Args:
        rollout_dirs: directories holding `house_*/trajectories*.h5` plus their
            side-car MP4s. Several are merged into one dataset.
        output_dir: where the shards and `dataset_meta.json` go.
        successful_only: keep only episodes the task judged successful. This is
            the whole point of cloning a partial expert -- the episodes it
            actually completes are the demonstrations; the rest are
            counter-examples.

    Returns:
        The dataset directory.
    """
    output_dir = Path(output_dir)
    build_dataset(
        run_dirs=[Path(directory) for directory in rollout_dirs],
        output_dir=output_dir,
        camera_names=[HEAD_CAMERA, WRIST_CAMERA_LEFT],
        successful_only=successful_only,
    )
    return output_dir


@click.command()
@click.option(
    "--rollouts",
    "rollout_dirs",
    multiple=True,
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Rollout directory from generate_dataset.py. Repeatable; several are "
    "pooled into one dataset.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    required=True,
    help="Where to write the dataset shards.",
)
@click.option(
    "--keep-failures/--successful-only",
    default=False,
    help="Include episodes the task judged unsuccessful. Off by default: a partial "
    "expert's failures are counter-examples, not demonstrations.",
)
def main(rollout_dirs: tuple[Path, ...], output_dir: Path, keep_failures: bool) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    dataset_dir = collect_demonstrations(
        rollout_dirs=list(rollout_dirs),
        output_dir=output_dir,
        successful_only=not keep_failures,
    )
    click.secho(f"Dataset written to {dataset_dir}", fg="green")


if __name__ == "__main__":
    main()
