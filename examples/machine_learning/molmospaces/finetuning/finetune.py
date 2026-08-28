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
normalisation statistics, write the trainer config, put the trainer's own
repository on disk, and write a shell script that runs the whole remaining
sequence. The training itself happens in that repository, because that is where
the model, its PyTorch stack and its checkpoint format live, and none of them is
a dependency here.

Nothing heavyweight runs from here. `--trainer molmobot` clones MolmoBot and
downloads its two data-postprocessing scripts -- which are not in its git
repository, see `molmobot_repo.py` -- and then stops. Creating MolmoBot's
virtualenv pulls torch, so `uv sync` is the first line of the generated script
rather than something this command does on your behalf.

    # MolmoBot, from generated rollouts: prepare, clone, write run_molmobot.sh
    python -m examples.machine_learning.molmospaces.finetuning.finetune \\
        --rollouts data/stretch_pick/rollouts/pick --trainer molmobot

    # ... then run it yourself
    bash data/stretch_pick/rollouts/molmobot/pick/run_molmobot.sh

    # pi0.5, from an exported LeRobot dataset
    python -m examples.machine_learning.molmospaces.finetuning.finetune \\
        --dataset data/stretch_pick/lerobot --trainer openpi --base-checkpoint pi05_droid
"""

from __future__ import annotations

import json
import logging
import shlex
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

from examples.machine_learning.molmospaces.finetuning.lerobot_export import (
    CAMERA_FEATURE_NAMES,
)
from examples.machine_learning.molmospaces.finetuning.molmobot_repo import (
    DEFAULT_CHECKOUT,
    MolmoBotCheckout,
    MolmoBotSetupError,
    ensure_checkout,
)

CAMERA_NAME_ALIASES: dict[str, str] = {
    "head": "head_camera",
    "head_camera": "head_camera",
    "wrist": "wrist_camera_left",
    "wrist_camera": "wrist_camera_left",
    "wrist_left": "wrist_camera_left",
    "wrist_camera_left": "wrist_camera_left",
    "wrist_right": "wrist_camera_right",
    "wrist_camera_right": "wrist_camera_right",
    "stereo": "wrist_camera_stereo",
    "wrist_stereo": "wrist_camera_stereo",
    "wrist_camera_stereo": "wrist_camera_stereo",
    "wrist_depth": "wrist_camera_stereo",
    "wrist_camera_depth": "wrist_camera_stereo",
    "gripper_camera_stereo_depth": "wrist_camera_stereo",
    "left": "head_camera_left",
    "head_left": "head_camera_left",
    "head_camera_left": "head_camera_left",
    "right": "head_camera_right",
    "head_right": "head_camera_right",
    "head_camera_right": "head_camera_right",
}

DEFAULT_CAMERA_NAMES: list[str] = [
    "head_camera",
    "wrist_camera_left",
    "wrist_camera_right",
    "head_camera_left",
    "head_camera_right",
]
STRETCH_CAMERA_NAMES = DEFAULT_CAMERA_NAMES
"""
Default cameras available for fine-tuning.

These are the names `Stretch4CameraSystem` records under (`head_camera`,
`wrist_camera_left`, `wrist_camera_right`, `head_camera_left`, and `head_camera_right`). When fine-tuning,
the user can choose which subset of camera streams to train on via `--cameras`.
"""


def parse_camera_names(
    cameras_str: str | None, available_cameras: list[str] | None = None
) -> list[str]:
    """Parse a camera selection string (e.g. 'head,wrist' or 'head_camera,head_camera_left') into canonical camera names."""
    if not cameras_str:
        return list(available_cameras) if available_cameras else list(DEFAULT_CAMERA_NAMES)
    tokens = [t.strip() for t in cameras_str.split(",") if t.strip()]
    resolved: list[str] = []
    for token in tokens:
        canonical = CAMERA_NAME_ALIASES.get(token.lower(), token)
        if canonical not in resolved:
            resolved.append(canonical)
    return resolved


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
    camera_names: list[str] | None = None,
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
        camera_names: cameras to include in the dataset manifest.
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

    # If camera_names is None, ensure_sensor_data_paths will inspect available MP4s
    ensure_sensor_data_paths(rollout_dir, camera_names=camera_names)
    task_dir = Path(task_dir) if task_dir else rollout_dir.parent / "molmobot" / rollout_dir.name
    placed = arrange_train_val_split(rollout_dir, task_dir, val_fraction=val_fraction, link=link)

    # Determine video keys (cameras) from the first house's available MP4s
    detected_cameras: list[str] = []
    first_h5 = next(rollout_dir.rglob("trajectories*.h5"), None)
    if first_h5 is not None:
        for mp4 in first_h5.parent.glob("episode_00000000_*.mp4"):
            cam_name = mp4.stem.replace("episode_00000000_", "").split("_batch_")[0]
            if cam_name in DEFAULT_CAMERA_NAMES and cam_name not in detected_cameras:
                detected_cameras.append(cam_name)

    active_cameras = camera_names or detected_cameras or list(DEFAULT_CAMERA_NAMES)

    return DatasetSummary(
        root=task_dir,
        kind="molmospaces",
        action_space="stretch_move_groups",
        state_dim=sum(STRETCH_ACTION_SPEC.values()),
        action_dim=sum(STRETCH_ACTION_SPEC.values()),
        num_episodes=count_trajectories(rollout_dir),
        num_frames=0,  # counting frames means opening every trajectory; not worth it here
        fps=fps,
        video_keys=active_cameras,
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

    For a MolmoSpaces dataset this returns nothing: `train_molmobot.py` computes
    its own statistics from the trajectories on the first run -- quantiles over
    the actions, min/max over qpos -- and caches them in the
    `synthmanip_norm_stats.yaml` at its `--stats_path`. A second set computed
    here would be a second source of truth for the same numbers.

    Standard deviations are floored, for the reason `training/dataset.py` gives:
    several action dimensions are constant in a simple_ik demonstration, and
    normalising by a true zero produces NaNs that only surface much later as a
    policy emitting garbage.
    """
    stats_path = summary.root / "meta" / "episodes_stats.jsonl"
    if not stats_path.exists():
        return {}

    records = [
        json.loads(line)
        for line in stats_path.read_text().splitlines()
        if line.strip()
    ]
    if not records:
        return {}
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
    camera_names: list[str] | None = None,
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
        selected = camera_names or summary.video_keys or list(DEFAULT_CAMERA_NAMES)
        config["action"] = {
            "action_type": action_type,
            "action_move_groups": list(STRETCH_ACTION_SPEC),
            "action_spec": dict(STRETCH_ACTION_SPEC),
            "action_dim": summary.action_dim,
            "camera_names": list(selected),
        }
    else:
        if camera_names:
            selected_features = {
                CAMERA_FEATURE_NAMES.get(c, c) for c in camera_names
            } | set(camera_names)
            image_keys = [
                k
                for k in summary.video_keys
                if k in selected_features or k.split(".")[-1] in selected_features
            ]
        else:
            image_keys = summary.video_keys

        config["features"] = {
            "observation.state": {"shape": [summary.state_dim]},
            "action": {"shape": [summary.action_dim]},
            "images": image_keys,
        }
        config["normalization"] = dataset_statistics(summary)

    path = summary.root / f"finetune_{trainer}.json"
    path.write_text(json.dumps(config, indent=2))
    return path


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
    camera_names: list[str] | None = None,
    dataset_ref: str | None = None,
    python: str = "python",
) -> list[str]:
    """The command line that runs the fine-tune in the trainer's own repository.

    Args:
        dataset_ref: how to spell the dataset root in the command. Defaults to
            its absolute path; the generated shell script passes `"$DATASET"`
            so the emitted command stays readable.
        python: the interpreter to invoke. The generated script passes the
            trainer virtualenv's own, since it is never on `$PATH` here.

    `--data_paths` takes the *task* directory, not a split: MolmoBot appends
    `train` or `val` itself (`SynthmanipDataset._resolve_data_path`). MolmoBot's
    README shows a Franka example with `/train` already on the end, which would
    make it look for `train/train`.
    """
    root = dataset_ref if dataset_ref is not None else str(summary.root)
    if trainer == "molmobot":
        selected = camera_names or summary.video_keys or list(DEFAULT_CAMERA_NAMES)
        return [
            python,
            "launch_scripts/train_molmobot.py",
            base_checkpoint,
            "--data_paths",
            root,
            "--seq_len",
            str(seq_len),
            "--action_dim",
            str(summary.action_dim),
            "--action_move_groups",
            *STRETCH_ACTION_SPEC,
            "--camera_names",
            *selected,
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
            f"--data.repo-id={root}",
            f"--checkpoint-dir={output_dir}",
            f"--overrides={config_path}",
        ]
    return [
        python,
        "-m",
        "lerobot.scripts.train",
        f"--dataset.root={root}",
        f"--dataset.repo_id=stretch4/{summary.root.name}",
        f"--policy.path={base_checkpoint}",
        f"--output_dir={output_dir}",
        f"--config_path={config_path}",
    ]


# =============================================================================
# The generated shell script
# =============================================================================


def _shell(parts: list[str]) -> str:
    """Join a command for the generated script, leaving `"$VAR"` references expandable.

    A token that opens with `"$` was written by the code below with its own
    quoting already in place -- `"$DATASET"/train` and the like -- and must pass
    through untouched, or the variable is emitted literally and the script looks
    for a directory called `$DATASET`. Everything else is a value and gets
    quoted.
    """
    return " ".join(part if part.startswith('"$') else shlex.quote(part) for part in parts)


def write_launch_script(
    summary: DatasetSummary,
    trainer: str,
    config_path: Path,
    base_checkpoint: str,
    output_dir: Path,
    batch_size: int,
    steps: int,
    action_type: str,
    seq_len: int,
    camera_names: list[str] | None = None,
    checkout: MolmoBotCheckout | None = None,
) -> Path:
    """Write the rest of the workflow as a shell script, and return its path.

    A script rather than a printed list of commands, because the sequence is
    four steps long, every path in it is absolute, and two of the steps are easy
    to skip by accident -- and because a file can be read, edited and re-run,
    which a terminal scrollback cannot.

    It is emitted, never executed. `uv sync` pulls torch into MolmoBot's
    virtualenv and the training itself runs for a day; both are decisions to
    take deliberately, not side effects of a command that mostly inspects data.
    """
    body = (
        _molmobot_script(
            summary=summary,
            checkout=checkout,
            base_checkpoint=base_checkpoint,
            config_path=config_path,
            output_dir=output_dir,
            batch_size=batch_size,
            steps=steps,
            action_type=action_type,
            seq_len=seq_len,
            camera_names=camera_names,
        )
        if trainer == "molmobot" and checkout is not None
        else _generic_trainer_script(
            summary=summary,
            trainer=trainer,
            config_path=config_path,
            base_checkpoint=base_checkpoint,
            output_dir=output_dir,
            batch_size=batch_size,
            steps=steps,
            action_type=action_type,
            seq_len=seq_len,
            camera_names=camera_names,
        )
    )

    header = [
        "#!/usr/bin/env bash",
        "#",
        f"# Fine-tune {trainer} on {summary.root.name}.",
        "#",
        "# Generated by examples/machine_learning/molmospaces/finetuning/finetune.py.",
        "# Re-running that command overwrites this file; nothing reads it back, so",
        "# edit freely once you have it.",
        "",
        "set -euo pipefail",
        "",
    ]
    path = summary.root / f"run_{trainer}.sh"
    path.write_text("\n".join(header + body) + "\n")
    path.chmod(0o755)
    return path


def _molmobot_script(
    summary: DatasetSummary,
    checkout: MolmoBotCheckout,
    base_checkpoint: str,
    config_path: Path,
    output_dir: Path,
    batch_size: int,
    steps: int,
    action_type: str,
    seq_len: int,
    camera_names: list[str] | None,
) -> list[str]:
    """The four steps between a prepared rollout run and a MolmoBot checkpoint."""
    dataset = summary.root.resolve()
    package = checkout.package_dir.resolve()
    scripts = checkout.data_scripts_dir.resolve()
    splits = [split for split, count in summary.splits.items() if count]

    validate = [
        _shell(
            [
                '"$PYTHON"',
                f'"$SCRIPTS"/validate_trajectories.py',
                f'"$DATASET"/{split}',
                "--num-workers",
                "8",
            ]
        )
        for split in splits
    ]

    return [
        f"PACKAGE={shlex.quote(str(package))}",
        f"SCRIPTS={shlex.quote(str(scripts))}",
        f"DATASET={shlex.quote(str(dataset))}",
        'PYTHON="$PACKAGE"/.venv/bin/python',
        "",
        'cd "$PACKAGE"',
        "",
        "# --------------------------------------------------------------------------",
        "# 1. MolmoBot's virtualenv.",
        "#",
        "#    `--extra train` is the extra that carries h5py, which both postprocessing",
        "#    scripts import; decord and tqdm are already core dependencies. This is the",
        "#    step that downloads torch, so it is the slow one.",
        "# --------------------------------------------------------------------------",
        "uv sync --extra train",
        "",
        "# --------------------------------------------------------------------------",
        "# 2. valid_trajectory_index.json, once per split.",
        "#",
        "#    The only mandatory step: SynthmanipDataset opens this file and raises if",
        "#    it is missing from a split directory. It also writes a `valid_traj_mask`",
        "#    into each HDF5, marking trajectories whose actions, qpos/qvel or videos do",
        "#    not decode, or whose video frame count disagrees with the trajectory",
        "#    length -- those are then skipped rather than crashing the dataloader.",
        "#",
        "#    Add `--check-visibility head_camera pickup_obj` to additionally drop",
        "#    episodes where the target object is not in frame at the first step. Left",
        "#    off here because it discards data, and how much depends on the task.",
        "# --------------------------------------------------------------------------",
        *validate,
        "",
        "# --------------------------------------------------------------------------",
        "# 3. Per-trajectory statistics, into a `stats` group in each HDF5 plus an",
        "#    aggregated_stats.json.",
        "#",
        "#    Optional for the run below, despite MolmoBot's README presenting it as a",
        "#    prerequisite: train_molmobot.py normalises actions by quantiles over the",
        "#    raw `actions` datasets and state by min/max over raw `obs/agent/qpos`, and",
        "#    reads neither the `stats` group nor the JSON. It is kept because",
        "#    MolmoBot's min_max and mean_std normalisation modes do read that group.",
        "#    Delete it if the dataset is large and you are training as configured.",
        "#",
        f"#    `actions/joint_pos` is in the keys alongside `actions/{action_type}`",
        "#    because the min_max path looks the gripper up under joint_pos regardless",
        "#    of the action type, and silently yields nothing when it is absent.",
        "# --------------------------------------------------------------------------",
        _shell(
            [
                '"$PYTHON"',
                '"$SCRIPTS"/calculate_stats.py',
                '"$DATASET"/train',
                "--keys",
                f"actions/{action_type}",
                "actions/joint_pos",
                "obs/agent/qpos",
            ]
        ),
        "",
        "# --------------------------------------------------------------------------",
        "# 4. Train.",
        "#",
        "#    --stats_path defaults to synthmanip_norm_stats.yaml, which the trainer",
        "#    computes on the first run and reuses afterwards. Delete it to recompute",
        "#    after the dataset changes, or the policy is normalised against the old one.",
        "#",
        "#    Scale --global_batch_size to the hardware; MolmoBot micro-batches with",
        "#    --device_batch_size, and its own runs used torchrun across 8-64 H100s.",
        "# --------------------------------------------------------------------------",
        "export PYTHONPATH=\"$PACKAGE${PYTHONPATH:+:$PYTHONPATH}\"",
        _shell(
            trainer_command(
                summary=summary,
                trainer="molmobot",
                config_path=config_path,
                base_checkpoint=base_checkpoint,
                output_dir=output_dir,
                batch_size=batch_size,
                steps=steps,
                action_type=action_type,
                seq_len=seq_len,
                camera_names=camera_names,
                dataset_ref='"$DATASET"',
                python='"$PYTHON"',
            )
        ),
        "",
        "# Then score the checkpoint, back in the stretch4_mujoco repo -- natively, with",
        "# no action remapping:",
        f"#   {_evaluation_command(summary, 'molmobot')}",
        f"# serving it with `--action-type {action_type}`, matching what it trained on.",
    ]


def _generic_trainer_script(
    summary: DatasetSummary,
    trainer: str,
    config_path: Path,
    base_checkpoint: str,
    output_dir: Path,
    batch_size: int,
    steps: int,
    action_type: str,
    seq_len: int,
    camera_names: list[str] | None,
) -> list[str]:
    """openpi and LeRobot: one command, in a checkout this repo does not manage.

    Neither is cloned for you the way MolmoBot is. They take a LeRobot dataset
    that is already complete, so there is no postprocessing to sequence, and the
    only thing worth generating is the command with the right config path in it.
    """
    return [
        f"# Run this from your {trainer} checkout.",
        f"TRAINER_REPO=${{TRAINER_REPO:?set TRAINER_REPO to your {trainer} checkout}}",
        'cd "$TRAINER_REPO"',
        "",
        _shell(
            trainer_command(
                summary=summary,
                trainer=trainer,
                config_path=config_path,
                base_checkpoint=base_checkpoint,
                output_dir=output_dir,
                batch_size=batch_size,
                steps=steps,
                action_type=action_type,
                seq_len=seq_len,
                camera_names=camera_names,
                dataset_ref=str(summary.root.resolve()),
            )
        ),
        "",
        f"# {_evaluation_command(summary, trainer)}",
    ]


def _evaluation_command(summary: DatasetSummary, trainer: str) -> str:
    """How to score the resulting checkpoint on the benchmarks."""
    if trainer == "molmobot" and summary.kind == "molmospaces":
        return (
            "python -m examples.machine_learning.molmospaces.run_benchmarks "
            "--policy molmobot --checkpoint <checkpoint> --benchmark pick"
        )
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
    "--cameras",
    type=str,
    default=None,
    help="Comma-separated list of camera names to train on (e.g. 'head_camera,wrist_camera_right', "
    "'head,wrist,left,right'). Defaults to all cameras available in the dataset.",
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
    help="Where MolmoBot is, or should be, checked out. Defaults to "
    f"{DEFAULT_CHECKOUT}; cloned if absent. Ignored by --trainer openpi/lerobot, "
    "which this repo does not check out for you.",
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
    "--clone/--no-clone",
    default=True,
    help="Clone MolmoBot into --trainer-repo when it is not there, and download the "
    "two postprocessing scripts that ship with its HuggingFace dataset rather than "
    "its git repository. --no-clone makes a missing checkout an error instead.",
)
def main(
    rollouts: Path | None,
    dataset: Path | None,
    cameras: str | None,
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
    clone: bool,
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
    selected_cameras = parse_camera_names(cameras) if cameras else None
    summary = (
        prepare_molmospaces_dataset(
            rollout_dir=rollouts,
            val_fraction=val_fraction,
            link=link,
            camera_names=selected_cameras,
        )
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
        camera_names=selected_cameras,
    )
    checkout = None
    if trainer == "molmobot":
        try:
            checkout = ensure_checkout(trainer_repo or DEFAULT_CHECKOUT, clone=clone)
        except MolmoBotSetupError as e:
            raise click.ClickException(str(e)) from e
        _print_checkout(checkout)

    script_path = write_launch_script(
        summary=summary,
        trainer=trainer,
        config_path=config_path,
        base_checkpoint=base_checkpoint,
        output_dir=output_dir,
        batch_size=batch_size,
        steps=steps,
        action_type=action_type,
        seq_len=seq_len,
        camera_names=selected_cameras,
        checkout=checkout,
    )

    click.echo("")
    click.secho(f"Wrote {config_path}", fg="green")
    click.secho(f"Wrote {script_path}", fg="green")
    click.echo("")
    click.secho("Read it, then run it:", bold=True)
    click.echo(f"  bash {script_path}")
    click.echo("")
    click.echo(
        "It installs MolmoBot's training dependencies, writes the trajectory index the\n"
        "dataloader requires, and starts the fine-tune -- the first of those downloads\n"
        "torch and the last runs for a long time, which is why it is yours to launch."
        if checkout is not None
        else f"Set TRAINER_REPO to your {trainer} checkout first."
    )


def _print_checkout(checkout: MolmoBotCheckout) -> None:
    click.echo("")
    click.secho(
        f"MolmoBot  {checkout.root}  ({'cloned just now' if checkout.cloned else 'already there'})",
        bold=True,
    )
    fetched = checkout.fetched_scripts
    click.echo(
        f"  scripts: {', '.join(fetched)} (downloaded)"
        if fetched
        else "  scripts: validate_trajectories.py, calculate_stats.py (already there)"
    )
    click.echo(
        f"  venv:    {checkout.venv_python}"
        if checkout.has_venv
        else "  venv:    not created yet -- the generated script's first step"
    )


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
