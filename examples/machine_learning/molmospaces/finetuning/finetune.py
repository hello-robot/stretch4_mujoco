"""
Prepare a Stretch dataset for fine-tuning, and launch the trainer.

Three trainers, and the choice decides what the data has to look like:

`molmobot` (the default)
    MolmoBot (https://github.com/allenai/MolmoBot) trains straight off
    MolmoSpaces trajectories -- `MolmoBot/olmo/data/synthmanip_dataset.py` opens
    `{data_path}/{split}/house_*/*.h5` and reads `obs/agent/qpos`,
    `actions/joint_pos_rel`, `obs_scene["task_description"]` and
    `obs/sensor_data/{camera}`. So there is **no conversion**: point it at the
    rollouts `generate_dataset.py` produced.

    Better than that, MolmoBot's action space is configurable *by move group*
    (`--action_move_groups`, `--camera_names`), so it learns Stretch's own
    ten-dimensional move-group action directly, and `SynthVLAPolicy` hands
    MolmoSpaces back an action dict keyed by move group, which is exactly what
    Stretch's controllers take.

`openpi` / `lerobot`
    These want a LeRobot dataset, so they take the output of
    `lerobot_export.py`. Its actions are Stretch's own ten numbers, so a
    pretrained checkpoint contributes its vision and language weights but its
    action head is re-learned; see that module.

What this script actually does, in either case: check the data, do the two
mechanical preparation steps MolmoBot needs (fill in the video paths MolmoSpaces'
saver leaves out, lay the houses out as `train/` and `val/`), compute the
normalisation statistics, write the trainer config, and print or run the command.
The training itself happens in the trainer's own repository, because that is
where the model, its JAX or PyTorch stack and its checkpoint format live, and
none of them is a dependency here.

    # MolmoBot, from generated rollouts
    python -m examples.machine_learning.molmospaces.finetuning.finetune \\
        --rollouts data/stretch_pick/rollouts/pick --trainer molmobot

    # ... and actually launch it
    python -m examples.machine_learning.molmospaces.finetuning.finetune \\
        --rollouts data/stretch_pick/rollouts/pick --trainer molmobot \\
        --trainer-repo ~/src/MolmoBot --base-checkpoint 8b --run

    # pi0.5, from an exported LeRobot dataset
    python -m examples.machine_learning.molmospaces.finetuning.finetune \\
        --dataset data/stretch_pick/lerobot --trainer openpi --base-checkpoint pi05_droid
"""

from __future__ import annotations

import json
import logging
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import click
import numpy as np

log = logging.getLogger(__name__)

TRAINERS = ("molmobot", "openpi", "lerobot")

STRETCH_ACTION_SPEC: dict[str, int] = {
    "base": 3,
    "lift": 1,
    "arm": 1,
    "wrist": 3,
    "gripper": 2,
}
"""
Stretch's move groups and their widths, as MolmoBot's `action_spec` wants them.

`Stretch4RobotView.MOVE_GROUP_ORDER` and the widths in `robot_view.py`, summing
to ten. Compare MolmoBot's own presets: `franka_joint` is `arm(7), gripper(1)`
and `RBY1_full` is seven groups totalling 29, so ten across five groups is an
ordinary shape for it -- there is no preset for Stretch, which is why
`--action_move_groups` and `--action_dim` are passed explicitly.

The gripper is 2 wide because the MJCF models the one `stretch_gripper` actuator
as a mirrored finger pair and the recorded `actions/joint_pos` carries both. It
is one commanded degree of freedom; see `lerobot_export.GRIPPER_CHANNEL_NAMES`.
"""

STRETCH_CAMERA_NAMES = ["head_camera", "wrist_camera"]
"""
Cameras to train on, in the order MolmoBot should take them.

These are the names `Stretch4CameraSystem` records under, so they match the
generated rollouts. `head_camera` is the
*centre* camera of Stretch 4's fixed three-camera head (MJCF
`camera_center_link`, 1.62m up, pitched 35 degrees down); the left/right stereo
pair look 47 degrees down and are not used.

Conveniently `train_molmobot.py --point_prompt_camera` already defaults to
`head_camera`, so point prompts work without further argument.
"""

MOLMOBOT_ACTION_TYPES = ("joint_pos_rel", "joint_pos")
"""
MolmoBot's action types, as `serve_molmo.py --action-type` accepts them.

`joint_pos_rel` is the per-step difference and is what
`synthmanip_dataset.py` prefers, falling back to `joint_pos` when the relative
key is absent. Both are written by the generated rollouts and by
`live_recorder.py`, so either works -- but the serving side has to be told the
same one, or the arm treats absolute targets as deltas.
"""

# =============================================================================
# Reading what is on disk
# =============================================================================


@dataclass
class DatasetSummary:
    """What a prepared dataset says about itself, whichever kind it is."""

    root: Path
    kind: str
    """`lerobot` or `molmospaces`."""

    action_space: str
    state_dim: int
    action_dim: int
    num_episodes: int
    num_frames: int
    fps: float
    video_keys: list[str] = field(default_factory=list)
    tasks: list[str] = field(default_factory=list)
    splits: dict[str, int] = field(default_factory=dict)
    """Houses per split. MolmoSpaces datasets only."""


def read_lerobot_dataset(root: Path) -> DatasetSummary:
    """Read the metadata `lerobot_export.py` wrote."""
    root = Path(root)
    info_path = root / "meta" / "info.json"
    export_path = root / "meta" / "stretch_export.json"
    if not info_path.exists():
        raise click.ClickException(
            f"{info_path} not found, so this is not an exported LeRobot dataset. Build one with\n"
            "  python -m examples.machine_learning.molmospaces.finetuning.generate_dataset\n"
            "or pass --rollouts for a raw MolmoSpaces run (which is what --trainer molmobot wants)."
        )
    info = json.loads(info_path.read_text())
    export = json.loads(export_path.read_text()) if export_path.exists() else {}

    return DatasetSummary(
        root=root,
        kind="lerobot",
        action_space=export.get("action_space", "unknown"),
        state_dim=int(info["features"]["observation.state"]["shape"][0]),
        action_dim=int(info["features"]["action"]["shape"][0]),
        num_episodes=int(info["total_episodes"]),
        num_frames=int(info["total_frames"]),
        fps=float(info["fps"]),
        video_keys=[
            key for key, feature in info["features"].items() if feature.get("dtype") == "video"
        ],
        tasks=export.get("tasks", []),
    )


def prepare_molmospaces_dataset(
    rollout_dir: Path,
    task_dir: Path | None = None,
    val_fraction: float = 0.1,
    link: bool = True,
    fps: float = 15.0,
) -> DatasetSummary:
    """Make a raw rollout run trainable by MolmoBot, and summarise it.

    Two preparation steps, both mechanical and both easy to forget:

    1. `ensure_sensor_data_paths()` -- MolmoSpaces' saver strips camera
       observations before batching, so it writes an empty `obs/sensor_data`
       group even though the MP4s are right there. MolmoBot reads the video
       filename out of that group, so without this every trajectory looks
       image-less.
    2. `arrange_train_val_split()` -- MolmoSpaces writes houses flat, MolmoBot
       wants `train/` and `val/` subdirectories.

    Args:
        rollout_dir: a run containing `house_*/trajectories*.h5`.
        task_dir: where to build the `train/`+`val/` layout. Defaults to
            `<rollout_dir>/../molmobot/<rollout_dir.name>`, i.e. beside the
            rollouts rather than inside them, so re-running is idempotent and
            the raw run stays untouched.
        val_fraction: share of houses held out for validation.
        link: symlink houses into the split rather than copying them.
        fps: frame rate the rollouts were recorded at, for the report.
    """
    from examples.machine_learning.molmospaces.hdf5_layout import (
        arrange_train_val_split,
        count_trajectories,
        ensure_sensor_data_paths,
    )

    rollout_dir = Path(rollout_dir)
    if not any(rollout_dir.rglob("trajectories*.h5")):
        raise click.ClickException(
            f"No house_*/trajectories*.h5 under {rollout_dir}. Generate some with\n"
            "  python -m examples.machine_learning.molmospaces.finetuning.generate_dataset "
            "--task pick --output-dir data/stretch_pick"
        )

    ensure_sensor_data_paths(rollout_dir, camera_names=STRETCH_CAMERA_NAMES)
    task_dir = Path(task_dir) if task_dir else rollout_dir.parent / "molmobot" / rollout_dir.name
    placed = arrange_train_val_split(rollout_dir, task_dir, val_fraction=val_fraction, link=link)

    return DatasetSummary(
        root=task_dir,
        kind="molmospaces",
        action_space="stretch_move_groups",
        state_dim=sum(STRETCH_ACTION_SPEC.values()),
        action_dim=sum(STRETCH_ACTION_SPEC.values()),
        num_episodes=count_trajectories(rollout_dir),
        num_frames=0,  # counting frames means opening every trajectory; not worth it here
        fps=fps,
        video_keys=list(STRETCH_CAMERA_NAMES),
        splits={split: len(houses) for split, houses in placed.items()},
    )


# =============================================================================
# Normalisation statistics
# =============================================================================


def dataset_statistics(summary: DatasetSummary) -> dict[str, list[float]]:
    """Mean and standard deviation of state and action over the whole dataset.

    For a LeRobot dataset, pooled from the per-episode statistics in
    `meta/episodes_stats.jsonl` -- cheaper than re-reading every parquet file and
    exact for the mean. The pooled standard deviation combines the within-episode
    variance with the spread of the episode means, which is what makes it the
    dataset's deviation rather than the average of the episodes' deviations.

    For a MolmoSpaces dataset this returns nothing: MolmoBot computes its own
    statistics with `calculate_stats.py` into the `synthmanip_norm_stats.yaml`
    that `train_molmobot.py --stats_path` reads, and a second set computed here
    would be a second source of truth for the same numbers.

    Standard deviations are floored, for the reason `training/dataset.py` gives:
    several action dimensions are constant in a scripted demonstration, and
    normalising by a true zero produces NaNs that only surface much later as a
    policy emitting garbage.
    """
    if summary.kind != "lerobot":
        return {}

    records = [
        json.loads(line)
        for line in (summary.root / "meta" / "episodes_stats.jsonl").read_text().splitlines()
        if line.strip()
    ]
    statistics: dict[str, list[float]] = {}
    for key in ("observation.state", "action"):
        counts = np.array([record["stats"][key]["count"][0] for record in records], dtype=float)
        means = np.array([record["stats"][key]["mean"] for record in records], dtype=float)
        stds = np.array([record["stats"][key]["std"] for record in records], dtype=float)
        weights = (counts / counts.sum())[:, None]
        pooled_mean = (weights * means).sum(axis=0)
        pooled_variance = (weights * (stds**2 + (means - pooled_mean) ** 2)).sum(axis=0)
        statistics[f"{key}.mean"] = pooled_mean.tolist()
        statistics[f"{key}.std"] = np.maximum(np.sqrt(pooled_variance), 1e-3).tolist()
    return statistics


# =============================================================================
# Trainer configs and commands
# =============================================================================


def write_trainer_config(
    summary: DatasetSummary,
    trainer: str,
    base_checkpoint: str,
    output_dir: Path,
    batch_size: int,
    steps: int,
    learning_rate: float,
    action_type: str,
) -> Path:
    """Write the trainer's config next to the dataset and return its path.

    Emitted as JSON rather than as the trainer's native Python or YAML: all three
    trainers resolve configs from dataclasses whose fields move between versions,
    so a generated module is a file that stops importing. JSON of the same field
    names is stable, diffable, and reviewable before it is handed to something
    that will spend a day on it.
    """
    config = {
        "trainer": trainer,
        "base_checkpoint": base_checkpoint,
        "dataset": {
            "root": str(summary.root),
            "kind": summary.kind,
            "action_space": summary.action_space,
            "fps": summary.fps,
            "num_episodes": summary.num_episodes,
            "splits": summary.splits,
        },
        "optimizer": {
            "batch_size": batch_size,
            "num_train_steps": steps,
            "learning_rate": learning_rate,
        },
        "output_dir": str(output_dir),
        "evaluation": {"command": _evaluation_command(summary, trainer)},
    }
    if summary.kind == "molmospaces":
        config["action"] = {
            "action_type": action_type,
            "action_move_groups": list(STRETCH_ACTION_SPEC),
            "action_spec": dict(STRETCH_ACTION_SPEC),
            "action_dim": summary.action_dim,
            "camera_names": list(STRETCH_CAMERA_NAMES),
        }
    else:
        config["features"] = {
            "observation.state": {"shape": [summary.state_dim]},
            "action": {"shape": [summary.action_dim]},
            "images": summary.video_keys,
        }
        config["normalization"] = dataset_statistics(summary)

    path = summary.root / f"finetune_{trainer}.json"
    path.write_text(json.dumps(config, indent=2))
    return path


def preparation_commands(summary: DatasetSummary, action_type: str) -> list[list[str]]:
    """MolmoBot's own preprocessing steps, which have to run before training.

    Both come from MolmoBot's README and both live in its repository, so they
    are printed rather than run: `validate_trajectories.py` writes the
    `valid_trajectory_index.json` manifest `synthmanip_dataset.py` reads, and
    `calculate_stats.py` writes the normalisation YAML `train_molmobot.py`
    expects at `--stats_path`.

    The `--check-visibility` argument is deliberately omitted. MolmoBot's example
    passes it the camera and object its Franka datasets were generated with; the
    equivalent for Stretch is `head_camera` and whatever the task's target is,
    which varies per episode. Skipping the check keeps every generated trajectory
    rather than silently dropping the ones whose target was occluded from a
    camera name that does not exist in this data.
    """
    if summary.kind != "molmospaces":
        return []
    return [
        ["python", "validate_trajectories.py", f"{summary.root}/train"],
        [
            "python",
            "calculate_stats.py",
            f"{summary.root}/train",
            "--keys",
            f"actions/{action_type}",
            "obs/agent/qpos",
        ],
    ]


def trainer_command(
    summary: DatasetSummary,
    trainer: str,
    config_path: Path,
    base_checkpoint: str,
    output_dir: Path,
    batch_size: int,
    steps: int,
    action_type: str,
    seq_len: int,
) -> list[str]:
    """The command line that runs the fine-tune in the trainer's own repository."""
    if trainer == "molmobot":
        return [
            "python",
            "launch_scripts/train_molmobot.py",
            base_checkpoint,
            "--data_paths",
            str(summary.root),
            "--seq_len",
            str(seq_len),
            "--action_dim",
            str(summary.action_dim),
            "--action_move_groups",
            *STRETCH_ACTION_SPEC,
            "--camera_names",
            *STRETCH_CAMERA_NAMES,
            "--action_type",
            action_type,
            "--global_batch_size",
            str(batch_size),
            f"--exp_name=stretch4_{summary.root.name}",
        ]
    if trainer == "openpi":
        return [
            "uv",
            "run",
            "scripts/train.py",
            base_checkpoint,
            f"--exp-name=stretch4_{summary.action_space}",
            f"--data.repo-id={summary.root}",
            f"--checkpoint-dir={output_dir}",
            f"--overrides={config_path}",
        ]
    return [
        "python",
        "-m",
        "lerobot.scripts.train",
        f"--dataset.root={summary.root}",
        f"--dataset.repo_id=stretch4/{summary.root.name}",
        f"--policy.path={base_checkpoint}",
        f"--output_dir={output_dir}",
        f"--config_path={config_path}",
    ]


def _evaluation_command(summary: DatasetSummary, trainer: str) -> str:
    """How to score the resulting checkpoint on the benchmarks."""
    if trainer == "molmobot" and summary.kind == "molmospaces":
        return (
            "python -m examples.machine_learning.molmospaces.run_benchmarks "
            "--policy molmobot --checkpoint <checkpoint> --benchmark pick"
        )
    # There is no `--policy` for an openpi or LeRobot checkpoint: those trainers
    # serve their own inference protocols, and the policy adapter that used to
    # bridge one (`--policy vla`) only existed to remap Franka-space actions and
    # went with the rest of the remapping. A checkpoint trained on Stretch's move
    # groups needs a MolmoSpaces `InferencePolicy` that speaks its trainer's
    # protocol -- `policies/molmobot_policy.py` is the pattern to copy.
    return (
        f"(no scorer in this repo for a {trainer} checkpoint -- add an "
        "InferencePolicy for its serving protocol, as policies/molmobot_policy.py "
        "does for MolmoBot, or fine-tune with --trainer molmobot instead)"
    )


# =============================================================================
# CLI
# =============================================================================


@click.command()
@click.option(
    "--rollouts",
    type=click.Path(path_type=Path, exists=True),
    default=None,
    help="A raw MolmoSpaces rollout run (`house_*/trajectories*.h5`). What --trainer "
    "molmobot wants, and what generate_dataset.py writes under `rollouts/`.",
)
@click.option(
    "--dataset",
    type=click.Path(path_type=Path, exists=True),
    default=None,
    help="An exported LeRobot dataset (the `lerobot/` directory). What --trainer "
    "openpi and lerobot want.",
)
@click.option(
    "--trainer",
    type=click.Choice(TRAINERS),
    default="molmobot",
    help="Which trainer to prepare for. 'molmobot' trains on MolmoSpaces "
    "trajectories directly, in Stretch's own move groups, with no remapping.",
)
@click.option(
    "--trainer-repo",
    type=click.Path(path_type=Path),
    default=None,
    help="Where the trainer repository is checked out. Required for --run.",
)
@click.option(
    "--base-checkpoint",
    type=str,
    default=None,
    help="Checkpoint to fine-tune from. Defaults to '8b' for molmobot (its base "
    "model) and 'pi05_droid' otherwise. Pass a MolmoBot checkpoint path, or "
    "'allenai/MolmoBot-DROID' to start from the released Franka-space model -- its "
    "vision and language weights carry over, its action head is re-learned on "
    "Stretch's move groups.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Where checkpoints go. Defaults to <data>/../checkpoints.",
)
@click.option(
    "--action-type",
    type=click.Choice(MOLMOBOT_ACTION_TYPES),
    default="joint_pos_rel",
    help="MolmoBot action type. Must match what the checkpoint is later served "
    "with (`serve_molmo.py --action-type`), or absolute targets get applied as deltas.",
)
@click.option(
    "--val-fraction",
    type=float,
    default=0.1,
    help="Share of *houses* held out for validation. Houses are never split, so a "
    "policy is not scored on a room it trained in.",
)
@click.option(
    "--link/--copy",
    default=True,
    help="Symlink houses into the train/val layout rather than copying them. Copy "
    "if the trainer runs somewhere the symlink target will not resolve.",
)
@click.option("--seq-len", type=int, default=2048, help="MolmoBot --seq_len.")
@click.option("--batch-size", type=int, default=32)
@click.option("--steps", type=int, default=30000)
@click.option("--learning-rate", type=float, default=1e-5)
@click.option(
    "--run/--dry-run",
    default=False,
    help="Execute the training command instead of printing it. Off by default.",
)
def main(
    rollouts: Path | None,
    dataset: Path | None,
    trainer: str,
    trainer_repo: Path | None,
    base_checkpoint: str | None,
    output_dir: Path | None,
    action_type: str,
    val_fraction: float,
    link: bool,
    seq_len: int,
    batch_size: int,
    steps: int,
    learning_rate: float,
    run: bool,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if (rollouts is None) == (dataset is None):
        raise click.UsageError(
            "Pass exactly one of --rollouts (a MolmoSpaces run, for --trainer molmobot) "
            "or --dataset (an exported LeRobot dataset, for --trainer openpi/lerobot)."
        )
    if trainer == "molmobot" and rollouts is None:
        raise click.UsageError(
            "--trainer molmobot trains on MolmoSpaces trajectories, so it needs --rollouts. "
            "There is no conversion step: point it at the `rollouts/<task>` directory "
            "generate_dataset.py wrote."
        )
    if trainer != "molmobot" and dataset is None:
        raise click.UsageError(
            f"--trainer {trainer} needs a LeRobot dataset, so pass --dataset. Build one with "
            "`generate_dataset.py`."
        )

    base_checkpoint = base_checkpoint or ("8b" if trainer == "molmobot" else "pi05_droid")
    summary = (
        prepare_molmospaces_dataset(rollouts, val_fraction=val_fraction, link=link)
        if rollouts is not None
        else read_lerobot_dataset(dataset)
    )
    output_dir = Path(output_dir) if output_dir else summary.root.parent / "checkpoints"

    _print_summary(summary)

    config_path = write_trainer_config(
        summary=summary,
        trainer=trainer,
        base_checkpoint=base_checkpoint,
        output_dir=output_dir,
        batch_size=batch_size,
        steps=steps,
        learning_rate=learning_rate,
        action_type=action_type,
    )
    preparation = preparation_commands(summary, action_type)
    command = trainer_command(
        summary=summary,
        trainer=trainer,
        config_path=config_path,
        base_checkpoint=base_checkpoint,
        output_dir=output_dir,
        batch_size=batch_size,
        steps=steps,
        action_type=action_type,
        seq_len=seq_len,
    )

    click.echo("")
    click.secho(f"Wrote {config_path}", fg="green")
    where = trainer_repo or f"<your {trainer} checkout>"
    if preparation:
        click.secho(f"First, in {where} (MolmoBot's own preprocessing):", bold=True)
        for step in preparation:
            click.echo("  " + " ".join(shlex.quote(part) for part in step))
        click.echo("")
    click.secho(f"Then train, in {where}:", bold=True)
    click.echo("  " + " ".join(shlex.quote(part) for part in command))
    click.echo("")
    click.secho("Then score the checkpoint:", bold=True)
    click.echo("  " + _evaluation_command(summary, trainer))
    if summary.kind == "molmospaces":
        click.echo(
            f"  ... serving it with `--action-type {action_type}`, matching what it was "
            "trained on."
        )

    if not run:
        return
    if trainer_repo is None:
        raise click.UsageError("--run needs --trainer-repo, the trainer's checkout.")
    if not Path(trainer_repo).exists():
        raise click.UsageError(f"--trainer-repo {trainer_repo} does not exist.")
    if preparation:
        raise click.UsageError(
            "--run will not skip MolmoBot's preprocessing. Run the two commands above "
            "first (they write valid_trajectory_index.json and the norm stats the "
            "trainer reads), then re-run with --run."
        )

    click.echo("")
    click.secho(f"Launching in {trainer_repo} ...", fg="cyan")
    completed = subprocess.run(command, cwd=trainer_repo, check=False)
    raise SystemExit(completed.returncode)


def _print_summary(summary: DatasetSummary) -> None:
    click.echo("")
    click.secho(f"Dataset  {summary.root}  ({summary.kind})", bold=True)
    if summary.kind == "molmospaces":
        click.echo(
            f"  {summary.num_episodes} trajectories at {summary.fps}Hz\n"
            f"  splits: {', '.join(f'{k}={v} houses' for k, v in summary.splits.items())}\n"
            f"  action: {summary.action_dim}-dim over Stretch's own move groups "
            f"{tuple(STRETCH_ACTION_SPEC)}\n"
            f"  cameras: {', '.join(summary.video_keys)}"
        )
        return
    click.echo(
        f"  {summary.num_episodes} episodes / {summary.num_frames} frames at {summary.fps}Hz\n"
        f"  action space: {summary.action_space} "
        f"(state {summary.state_dim}, action {summary.action_dim})\n"
        f"  images: {', '.join(summary.video_keys)}\n"
        f"  {len(summary.tasks)} distinct instructions"
    )


if __name__ == "__main__":
    main()
