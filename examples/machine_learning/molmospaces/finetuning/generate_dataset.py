"""
Generate Stretch demonstrations, then export them for fine-tuning.

One command for the whole data half of the pipeline:

    # smallest thing that proves the setup works: 2 episodes, 1 house
    python -m examples.machine_learning.molmospaces.finetuning.generate_dataset \\
        --task debug

    # a real run: 2000 pick episodes across procthor-objaverse, 8 workers
    python -m examples.machine_learning.molmospaces.finetuning.generate_dataset \\
        --task pick --episodes 2000 --num-workers 8 --output-dir data/stretch_pick

    # several families pooled into one training set
    python -m examples.machine_learning.molmospaces.finetuning.generate_dataset \\
        --task pick --task pnp --task open --episodes 1000 \\
        --output-dir data/stretch_manipulation

    # rollouts already on disk; just re-export them
    python -m examples.machine_learning.molmospaces.finetuning.generate_dataset \\
        --rollouts data/stretch_pick/rollouts --output-dir data/stretch_pick

Two stages, either of which can be run alone (`--no-export`, `--rollouts`):

1. **Generate.** MolmoSpaces' `ParallelRolloutRunner` over one of the Stretch
   datagen configs in `datagen_configs.py`, which writes HDF5 trajectories plus
   side-car MP4s under `<output-dir>/rollouts/<task>/`.
2. **Export.** `lerobot_export.py` turns those into a LeRobot dataset under
   `<output-dir>/lerobot/`, which is what a fine-tuning run consumes.

The generation stage is the expensive one -- it is a full physics rollout with
rendering per episode -- so it keeps its raw output rather than streaming
straight into the export. Re-exporting into a different action space is then
seconds rather than hours, which matters because `--action-space` is the choice
you are most likely to want to change your mind about.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import click

from examples.machine_learning.molmospaces.finetuning.datagen_configs import (
    DATAGEN_CONFIGS,
    qualified_config_name,
)
from examples.machine_learning.molmospaces.finetuning.lerobot_export import (
    ACTION_SPACES,
    export_lerobot_dataset,
)

log = logging.getLogger(__name__)


def generate_rollouts(
    task: str,
    output_dir: Path,
    episodes: int | None = None,
    num_workers: int = 1,
    scene_dataset: str | None = None,
    data_split: str | None = None,
    houses: int | None = None,
    seed: int | None = None,
) -> Path:
    """Run the data generation pipeline for one task family.

    Args:
        task: a key of `DATAGEN_CONFIGS`.
        output_dir: where the rollouts go. The pipeline appends its own
            `<ConfigName>/<timestamp>/` beneath this.
        episodes: total episodes to attempt. Spread over `houses` houses.
        num_workers: parallel rollout worker processes.
        scene_dataset: override the config's scene dataset, e.g. `procthor-10k`.
        data_split: `train`, `val` or `test`. Left at the config's default
            (`train`) unless given -- generating fine-tuning data out of `val` is
            how a benchmark score stops meaning anything.
        houses: how many houses to draw from. Defaults to enough that each house
            contributes a handful of episodes rather than hundreds, which is
            what keeps the scene distribution wide.
        seed: task-sampling seed.

    Returns:
        The directory the pipeline actually wrote to.
    """
    import importlib

    from molmo_spaces.data_generation.config_registry import get_config_class
    from molmo_spaces.data_generation.pipeline import ParallelRolloutRunner

    module_name, class_name = qualified_config_name(task).split(":")
    importlib.import_module(module_name)
    config = get_config_class(class_name)()

    if scene_dataset is not None:
        config.scene_dataset = scene_dataset
    if data_split is not None:
        config.data_split = data_split
    if seed is not None:
        config.seed = seed
    config.num_workers = num_workers
    config.use_wandb = False

    if episodes is not None:
        _spread_episodes(config, episodes, houses)

    config.output_dir = Path(output_dir) / task
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.save_config()

    log.info(
        f"[datagen] {class_name} | {config.scene_dataset}/{config.data_split} | "
        f"{len(config.task_sampler_config.house_inds)} houses x "
        f"{config.task_sampler_config.samples_per_house} episodes | "
        f"{num_workers} workers -> {config.output_dir}"
    )
    successes, total = ParallelRolloutRunner(config).run()
    log.info(f"[datagen] {task}: {successes}/{total} episodes succeeded")
    return config.output_dir


def _spread_episodes(config, episodes: int, houses: int | None) -> None:
    """Turn a total episode count into houses x samples-per-house.

    The samplers count in those two numbers rather than in episodes, and how the
    total is split matters: all 2000 episodes in one house is 2000 rollouts of
    one room. Defaults to roughly 4 episodes per house, capped at the 10 the
    sampler will retry a single house for before it gives up on it.
    """
    sampler_config = config.task_sampler_config
    per_house = 4 if houses is None else max(1, episodes // max(houses, 1))
    per_house = min(per_house, 10)
    house_count = houses if houses is not None else max(1, -(-episodes // per_house))

    available = list(sampler_config.house_inds or [])
    if len(available) < house_count:
        # The datagen configs ship a short house list (the first few) for
        # debugging; a real run needs more of the dataset than that.
        available = list(range(house_count))
    sampler_config.house_inds = available[:house_count]
    sampler_config.samples_per_house = per_house
    sampler_config.max_tasks = episodes


@click.command()
@click.option(
    "--task",
    "tasks",
    multiple=True,
    type=click.Choice(sorted(DATAGEN_CONFIGS)),
    default=("pick",),
    help="Task family to generate. Repeatable; several are pooled into one dataset.",
)
@click.option(
    "--episodes",
    type=int,
    default=None,
    help="Episodes to attempt per task. Defaults to the config's own house list.",
)
@click.option("--num-workers", type=int, default=1, help="Parallel rollout workers.")
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    required=True,
    help="Root for `rollouts/` and `lerobot/`.",
)
@click.option(
    "--rollouts",
    "rollout_dirs",
    multiple=True,
    type=click.Path(path_type=Path, exists=True),
    help="Skip generation and export these existing rollout directories instead.",
)
@click.option(
    "--action-space",
    type=click.Choice(ACTION_SPACES),
    default="stretch",
    help="Action/state space to export in. Only 'stretch', the native 10-dim "
    "move-group vector. See lerobot_export.py.",
)
@click.option(
    "--scene-dataset",
    type=str,
    default=None,
    help="Override the scene dataset, e.g. procthor-10k for a fast local run.",
)
@click.option(
    "--data-split",
    type=click.Choice(["train", "val", "test"]),
    default=None,
    help="Scene split. Leave unset to use the config's default (train).",
)
@click.option("--houses", type=int, default=None, help="How many houses to draw episodes from.")
@click.option("--seed", type=int, default=None, help="Task-sampling seed.")
@click.option(
    "--keep-failures/--successful-only",
    default=False,
    help="Include episodes the task judged unsuccessful. Off by default: a partial "
    "expert's failures are counter-examples, not demonstrations.",
)
@click.option("--export/--no-export", "want_export", default=True, help="Run the export stage.")
@click.option(
    "--fps", type=float, default=15.0, help="Frame rate to record in the dataset metadata."
)
def main(
    tasks: tuple[str, ...],
    episodes: int | None,
    num_workers: int,
    output_dir: Path,
    rollout_dirs: tuple[Path, ...],
    action_space: str,
    scene_dataset: str | None,
    data_split: str | None,
    houses: int | None,
    seed: int | None,
    keep_failures: bool,
    want_export: bool,
    fps: float,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # MolmoSpaces renders off-screen through EGL; without this a headless run
    # fails inside the camera manager rather than at startup.
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if rollout_dirs:
        directories = [Path(directory) for directory in rollout_dirs]
        click.secho(f"Exporting {len(directories)} existing rollout directories.", fg="cyan")
    else:
        directories = [
            generate_rollouts(
                task=task,
                output_dir=output_dir / "rollouts",
                episodes=episodes,
                num_workers=num_workers,
                scene_dataset=scene_dataset,
                data_split=data_split,
                houses=houses,
                seed=seed,
            )
            for task in tasks
        ]

    if not want_export:
        click.secho(f"Rollouts under {output_dir / 'rollouts'}", fg="green")
        return

    dataset_dir = output_dir / "lerobot"
    metadata = export_lerobot_dataset(
        rollout_dirs=directories,
        output_dir=dataset_dir,
        action_space=action_space,
        successful_only=not keep_failures,
        fps=fps,
    )
    click.echo("")
    click.secho(
        f"{metadata.num_episodes} episodes / {metadata.num_frames} frames "
        f"in {action_space} action space -> {dataset_dir}",
        fg="green",
    )
    click.echo(
        "Fine-tune with:\n"
        f"  python -m examples.machine_learning.molmospaces.finetuning.finetune "
        f"--dataset {dataset_dir} --dry-run"
    )


if __name__ == "__main__":
    main()
