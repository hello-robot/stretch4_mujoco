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

    # several tasks, one language-conditioned policy across all of them
    python -m examples.machine_learning.molmospaces.finetuning.finetune \\
        --rollouts data/stretch_pick/rollouts/pick \\
        --rollouts data/stretch_pick/rollouts/pnp --sample-rates "0.6,0.4"

    # pi0.5, from an exported LeRobot dataset
    python -m examples.machine_learning.molmospaces.finetuning.finetune \\
        --dataset data/stretch_pick/lerobot --trainer openpi --base-checkpoint pi05_droid
"""

from __future__ import annotations

import json
import logging
import re
import shlex
import sys
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
    ADAMW8BIT,
    DEFAULT_CHECKOUT,
    GRIPPER_STATE_WIDTH,
    STRETCH_PRESET_NAMES,
    MolmoBotCheckout,
    MolmoBotSetupError,
    METRICS_ENV_VAR,
    ensure_adamw8bit,
    ensure_checkout,
    ensure_metrics_log,
    ensure_stretch_presets,
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


DEFAULT_MOLMOBOT_CHECKPOINT = "allenai/MolmoBot-DROID"
"""
What `--trainer molmobot` fine-tunes from unless told otherwise.

Not MolmoBot's own `8b`: that string appears in `train_molmobot.py`'s help as
"Path to checkpoint or '8b' for base model", but nothing in the trainer maps it
to a model. `select_checkpoint` calls `os.listdir` on whatever it is handed, so
`8b` dies with `FileNotFoundError: '8b'` after the dataset has already loaded.

This repository holds `model.pt` and `config.yaml`, which is exactly the shape
`select_checkpoint` and `get_model` accept, so the generated script downloads it
and passes the directory. Starting from a trained MolmoBot policy also wants far
less data than starting from the Molmo2 VLM, which is the other documented route
-- see `_checkpoint_step`.
"""

VRAM_TIERS: list[tuple[int, int, int, str]] = [
    (70000, 2048, 8, "80GB, H100/A100"),
    (44000, 1024, 4, "48GB, A6000/L40S"),
    (30000, 528, 2, "32GB, RTX 5090"),
    (22000, 528, 1, "24GB, RTX 4090/3090"),
    (0, 528, 1, "under 24GB, or no nvidia-smi"),
]
"""
`(min MiB, seq_len, device_batch_size, label)`, most memory first.

The generated script picks a row from what `nvidia-smi` reports for the smallest
GPU it can see, so the same script runs on a workstation and a cluster node
without editing. Both values are plain shell variables afterwards, so a wrong
guess costs one edit and a re-run rather than a regeneration.

These are **starting points, not measurements**. Nothing here has been profiled
against MolmoBot; they are extrapolated from its README's own recipe (an 80GB
card at `seq_len=528` with `device_batch_size=32` and the LLM *unfrozen*, which
this configuration is far cheaper than) with a wide margin for the resident 4B
backbone. Expect to turn them down for more cameras, or up when the run fits with
room to spare.
"""

DEFAULT_SEQ_LEN = 528
"""
`--seq_len`, and the first thing to lower when the GPU runs out of memory.

Not a ceiling: the data loader is built with `pad="to_max"`, so the preprocessor
derives fixed output shapes from this and **every sample is padded to it**.
Doubling it doubles the activation memory of every step whether or not any
trajectory needs the room.

528 is what MolmoBot's own README uses for the configuration this generates --
two images, `crop_mode=resize`, `max_crops=1`, 3x3 pooling. Raise it if the
trainer complains that a sequence does not fit; the shapes it logs on startup
("Building ... dataset with output shapes") say how much is actually used.

This is the floor the generated script falls back to when it cannot see a GPU;
passing `--seq-len` pins it instead of letting `VRAM_TIERS` choose.
"""

DEFAULT_DEVICE_BATCH_SIZE = 1
"""
`--device_batch_size`: samples per forward/backward, and the second memory knob.

MolmoBot micro-batches already -- `Trainer.split_batch` chops the per-device
batch into `ceil(batch / microbatch)` pieces, with no divisibility requirement --
but its default is 2, which is what a 4B-parameter model on a 24GB card chokes
on. `--global_batch_size` is unaffected: the gradient accumulation just runs
more, smaller steps for the same effective batch.
"""

FISHEYE_CAMERA_NAMES = frozenset({"head_camera_left", "head_camera_right"})
"""
The Stretch cameras whose frames are already barrel-distorted when recorded.

`stretch/config.py` gives these two a ~123 degree field of view and installs
`install_fisheye_distortion_hook()`, which runs
`StretchCameras.cam_nav_rgb_se4_{left,right}.post_processing_callback` -- the
`_distort` function -- on every rendered frame before the MP4 is written. So the
wide-angle geometry is baked into the dataset, not something a trainer has to
reproduce.

That matters because MolmoBot has a `--cameras_to_warp` flag whose help reads
"apply GoPro fisheye warping (resize to 640x480 4:3 + barrel distortion)", which
is exactly what someone with wide-angle cameras reaches for. Pointed at these
two it distorts already-distorted frames, and the policy trains on a lens that
does not exist. The flag is for the opposite case: a camera rendered rectilinear
in simulation that is wide-angle on the robot.

`head_camera`, `wrist_camera_left` and `wrist_camera_right` are the rectilinear
ones -- they take the MJCF's default FOV and their post-processing callback is
None, so they get rotation only.
"""

PROGRESS_FILTER = Path(__file__).resolve().parent / "train_progress.py"
"""
The progress bar the generated script pipes training through.

A file in this repository rather than shell generated into the script: it is a
hundred lines of Python with a regex in it, and that belongs somewhere it can be
linted, tested and read on its own. Stdlib-only, so running it under MolmoBot's
virtualenv interpreter costs nothing.
"""

TRAINING_REPORT = Path(__file__).resolve().parent / "training_report.py"
"""
The report over `metrics.jsonl`, run from the generated script when training ends.

Called by path with MolmoBot's own interpreter, like `train_progress.py` and for
the same reason: the script runs inside the checkout's virtualenv, and neither
file imports anything this repository provides.
"""

REPORT_PYTHON = sys.executable
"""
The interpreter the *plots* are drawn with: this repository's, not the checkout's.

The text report is stdlib-only and runs under either. Drawing needs matplotlib,
which MolmoBot's virtualenv does not have and has no reason to -- so the live
plot is launched with whatever interpreter generated the script, recorded here
as an absolute path. If that interpreter is gone by the time the script runs,
the script says so and trains anyway.
"""

MOLMOBOT_OPTIONAL_FLAGS: list[tuple[str, list[str]]] = [
    (
        "--img_aug",
        [
            "Colour and crop jitter on the input images. Cheap, and the usual",
            "first reach for a policy that overfits a handful of houses.",
        ],
    ),
    (
        "--weighted_sampling",
        [
            "Grasp-aware timestep sampling: upweights the steps around a",
            "successful grasp and downweights failed ones. MolmoBot's own",
            "Franka and RBY1 runs both use it.",
        ],
    ),
    (
        "--randomize_prompts",
        [
            "Samples a phrasing per example from the scene's referral",
            "expressions instead of always using `task_description`. Needs the",
            "scene's `task_type` to be in MolmoBot's DEFAULT_PROMPT_TEMPLATES;",
            "if it is not, it logs a warning and falls back, so this is safe to",
            "try and easy to verify from the log.",
        ],
    ),
    (
        "--use_point_prompts",
        [
            "Appends the target's image points to the goal string. The rollouts",
            "do carry `obs/extra/object_image_points` per camera, so this works",
            "-- add `--point_prompt_camera <name>` if head_camera is not among",
            "the cameras being trained on.",
        ],
    ),
    (
        "--no_val",
        [
            "Skip validation. Worth it when a task has one held-out house: the",
            "number is noise, and the val pass still costs memory and time.",
        ],
    ),
    (
        "--ft_llm=True --ft_vit=True --ft_connector=True",
        [
            "Unfreeze the language model, vision tower and connector. Off by",
            "default -- only the action expert trains, which is what makes this",
            "fit on one card at all. Turning any of these on adds gradients and",
            "Adam state for billions of parameters; MolmoBot used 8-64 H100s.",
        ],
    ),
]
"""
Flags the generated script lists commented-out, with the reason to reach for each.

Chosen from `train_molmobot.py`'s argparse for being useful *and* non-obvious:
each either changes what the policy learns or is the thing to try when a run
underfits, and none of them can be guessed from the flag name alone.
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
        DuplicateSplitError,
        arrange_train_val_split,
        count_trajectories,
        ensure_sensor_data_paths,
        trajectory_files,
    )

    rollout_dir = Path(rollout_dir)
    if not trajectory_files(rollout_dir):
        raise click.ClickException(
            f"No house_*/trajectories*.h5 under {rollout_dir}. Generate some with\n"
            "  python -m examples.machine_learning.molmospaces.finetuning.generate_dataset "
            "--task pick --output-dir data/stretch_pick"
        )

    # If camera_names is None, ensure_sensor_data_paths will inspect available MP4s
    ensure_sensor_data_paths(rollout_dir, camera_names=camera_names)
    task_dir = Path(task_dir) if task_dir else rollout_dir.parent / "molmobot" / rollout_dir.name
    try:
        placed = arrange_train_val_split(
            rollout_dir, task_dir, val_fraction=val_fraction, link=link
        )
    except DuplicateSplitError as e:
        # A traceback here would bury the one thing worth reading: which houses,
        # and that val loss from a previous run against this layout was not real.
        raise click.ClickException(str(e)) from e

    # Determine video keys (cameras) from the first house's available MP4s
    detected_cameras: list[str] = []
    first_h5 = next(iter(trajectory_files(rollout_dir)), None)
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
        # The state is narrower than the action by the gripper; see GRIPPER_STATE_WIDTH.
        state_dim=sum(
            GRIPPER_STATE_WIDTH if "gripper" in group else width
            for group, width in STRETCH_ACTION_SPEC.items()
        ),
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


def mixture_root(datasets: list[DatasetSummary]) -> Path:
    """Where a run's config and launch script go.

    One task: inside its own directory, where it was before multi-task training
    existed. Several: their shared parent, which `prepare_molmospaces_dataset`
    guarantees for tasks generated into the same output directory -- the run
    spans all of them, so it does not belong inside any one.
    """
    return datasets[0].root if len(datasets) == 1 else datasets[0].root.parent


def write_trainer_config(
    datasets: list[DatasetSummary],
    trainer: str,
    base_checkpoint: str,
    output_dir: Path,
    batch_size: int,
    steps: int,
    learning_rate: float,
    action_type: str,
    camera_names: list[str] | None = None,
    sample_rates: list[float] | None = None,
) -> Path:
    """Write the trainer's config beside the dataset(s) and return its path.

    Emitted as JSON rather than as the trainer's native Python or YAML: all three
    trainers resolve configs from dataclasses whose fields move between versions,
    so a generated module is a file that stops importing. JSON of the same field
    names is stable, diffable, and reviewable before it is handed to something
    that will spend a day on it.

    `datasets` is a list because MolmoBot trains one language-conditioned policy
    across several task directories at once; the action spec and cameras are
    global to the run, so they are read off the first.
    """
    summary = datasets[0]
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
    if len(datasets) > 1:
        config["mixture"] = {
            "tasks": [
                {
                    "name": d.root.name,
                    "root": str(d.root),
                    "num_episodes": d.num_episodes,
                    "splits": d.splits,
                }
                for d in datasets
            ],
            "sample_rates": list(sample_rates) if sample_rates else None,
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

    path = mixture_root(datasets) / f"finetune_{trainer}.json"
    path.write_text(json.dumps(config, indent=2))
    return path


def experiment_name(datasets: list[DatasetSummary]) -> str:
    """`--exp_name`, naming every task in the mixture so runs are told apart."""
    return "stretch4_" + "_".join(d.root.name for d in datasets)


def trainer_command(
    datasets: list[DatasetSummary],
    trainer: str,
    config_path: Path,
    base_checkpoint: str,
    output_dir: Path,
    batch_size: int | str,
    steps: int | str,
    action_type: str,
    seq_len: int | str,
    camera_names: list[str] | None = None,
    dataset_refs: list[str] | None = None,
    sample_rates: list[float] | None = None,
    launcher: list[str] | None = None,
    save_folder_ref: str | None = None,
    device_batch_size: int | str = DEFAULT_DEVICE_BATCH_SIZE,
    num_workers: int | str = 4,
    log_interval: int | str = 20,
    extra_args: list[str] | None = None,
) -> list[str]:
    """The command line that runs the fine-tune in the trainer's own repository.

    Args:
        datasets: the task directories to train across, most-important first.
            MolmoBot aggregates several into one mixture and one policy,
            conditioned on each trajectory's `task_description`; the action spec,
            cameras and experiment name are global to the run.
        dataset_refs: how to spell those roots in the command. Defaults to their
            absolute paths; the generated shell script passes `"$DATASET_PICK"`
            and friends so the emitted command stays readable.
        sample_rates: one weight per dataset. Equal weighting when omitted.
        launcher: what to invoke the trainer with, e.g. `["torchrun",
            "--nproc-per-node=2"]`. Defaults to a bare `python`.

    `--data_paths` takes the *task* directory, not a split: MolmoBot appends
    `train` or `val` itself (`SynthmanipDataset._resolve_data_path`). MolmoBot's
    README shows a Franka example with `/train` already on the end, which would
    make it look for `train/train`.

    `--val_data_paths` is passed explicitly whenever there is more than one task,
    because MolmoBot otherwise validates on the `val/` of the *first* data path
    alone -- so a two-task run would report pick's loss while training on both.

    Three of MolmoBot's arguments are not argparse options at all. It builds a
    `TrainConfig` and then merges the leftover `--name=value` arguments into it
    as an OmegaConf dotlist, so `save_folder`, `max_duration` and `wandb` are
    fields of that config rather than flags with help text:

    - `save_folder` is `omegaconf.MISSING` and is the only mandatory field, so
      leaving it off fails at `OmegaConf.to_object` -- after the checkpoint has
      loaded and the normalisation statistics have been computed.
    - `wandb` interpolates `${oc.env:WANDB_PROJECT}` and `${oc.env:WANDB_ENTITY}`,
      which the same `to_object` call resolves; with those unset it raises
      immediately after `save_folder` is satisfied. `wandb=null` turns the
      logging off instead, and the generated script says how to turn it back on.
    - `max_duration` is hardcoded to 200000 in the constructed config, so this is
      where `--steps` actually lands.
    """
    summary = datasets[0]
    refs = dataset_refs if dataset_refs is not None else [str(d.root) for d in datasets]
    run = launcher or ["python"]
    if trainer == "molmobot":
        selected = camera_names or summary.video_keys or list(DEFAULT_CAMERA_NAMES)
        save_folder = save_folder_ref or str(output_dir / experiment_name(datasets))
        # In the generated script the path is its own variable, so a sweep over
        # one dataset can compute the normalisation statistics once and point
        # every run at them; the printed command (no `$SAVE_FOLDER`) spells the
        # default out instead, because it has no variables to refer to.
        stats_path = (
            '"$STATS_PATH"'
            if save_folder.startswith('"$')
            else str(Path(save_folder) / "synthmanip_norm_stats.yaml")
        )
        command = [
            *run,
            "launch_scripts/train_molmobot.py",
            base_checkpoint,
            "--data_paths",
            *refs,
        ]
        if len(refs) > 1:
            command += ["--val_data_paths", *refs]
            if sample_rates:
                command += ["--dataset_sample_rates", *(str(rate) for rate in sample_rates)]
        return command + [
            "--seq_len",
            str(seq_len),
            "--action_dim",
            str(summary.action_dim),
            # The preset, not --action_move_groups: MolmoBot takes the per-group
            # widths only from ACTION_SPECS[action_preset], and sets the move
            # groups from the same entry. `molmobot_repo.ensure_stretch_presets`
            # writes it from STRETCH_ACTION_SPEC, so the order the policy unpacks
            # in comes from one place.
            "--action_preset",
            STRETCH_PRESET_NAMES[action_type],
            "--camera_names",
            *selected,
            "--action_type",
            action_type,
            "--global_batch_size",
            str(batch_size),
            "--device_batch_size",
            str(device_batch_size),
            "--num_workers",
            str(num_workers),
            "--log_interval",
            str(log_interval),
            "--stats_path",
            stats_path,
            f"--exp_name={experiment_name(datasets)}",
            f"--save_folder={save_folder}",
            f"--max_duration={steps}",
            *(extra_args or ["--wandb=null"]),
        ]
    if trainer == "openpi":
        return [
            "uv",
            "run",
            "scripts/train.py",
            base_checkpoint,
            f"--exp-name=stretch4_{summary.action_space}",
            f"--data.repo-id={refs[0]}",
            f"--checkpoint-dir={output_dir}",
            f"--overrides={config_path}",
        ]
    return [
        *run,
        "-m",
        "lerobot.scripts.train",
        f"--dataset.root={refs[0]}",
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

    A token containing `"$` was written by the code below with its own quoting
    already in place -- `"$DATASET"/train`, `--nproc-per-node="$NPROC"` -- and
    must pass through untouched, or the variable is emitted literally and the
    script looks for a directory called `$DATASET`. Everything else is a value
    from the dataset or the CLI and gets quoted.
    """
    return " ".join(part if '"$' in part else shlex.quote(part) for part in parts)


def dataset_variables(datasets: list[DatasetSummary]) -> list[tuple[str, DatasetSummary]]:
    """Shell variable names for the task directories, one per dataset.

    A single task keeps the plain `DATASET`; several are named after themselves
    -- `DATASET_PICK`, `DATASET_PNP` -- so the trainer's `--data_paths` line says
    which task each path is without anyone matching up absolute paths by eye.
    """
    if len(datasets) == 1:
        return [("DATASET", datasets[0])]
    names: list[tuple[str, DatasetSummary]] = []
    taken: set[str] = set()
    for index, dataset in enumerate(datasets):
        stem = re.sub(r"[^A-Z0-9]+", "_", dataset.root.name.upper()).strip("_") or "TASK"
        name = f"DATASET_{stem}"
        if name in taken:  # two tasks whose directory names differ only in punctuation
            name = f"{name}_{index}"
        taken.add(name)
        names.append((name, dataset))
    return names


def is_hf_repo_id(reference: str) -> bool:
    """True for `org/name`, the form a base checkpoint takes when it needs downloading.

    MolmoBot's released checkpoints are HuggingFace repositories, but its trainer
    only takes a local directory -- `select_checkpoint` calls `os.listdir` on
    whatever it is given. So the generated script downloads first, and this is
    how it tells a repository from a path already on disk.
    """
    return (
        reference.count("/") == 1
        and not reference.startswith((".", "/", "~"))
        and not Path(reference).exists()
    )


def write_launch_script(
    datasets: list[DatasetSummary],
    trainer: str,
    config_path: Path,
    base_checkpoint: str,
    output_dir: Path,
    batch_size: int,
    steps: int,
    action_type: str,
    seq_len: int | None,
    camera_names: list[str] | None = None,
    sample_rates: list[float] | None = None,
    device_batch_size: int | None = None,
    checkout: MolmoBotCheckout | None = None,
) -> Path:
    """Write the rest of the workflow as a shell script, and return its path.

    A script rather than a printed list of commands, because the sequence is
    five steps long, every path in it is absolute, and most of the steps are easy
    to skip by accident -- and because a file can be read, edited and re-run,
    which a terminal scrollback cannot.

    It is emitted, never executed. `uv sync` pulls torch into MolmoBot's
    virtualenv, the base checkpoint is several gigabytes, and the training itself
    runs for a day; those are decisions to take deliberately, not side effects of
    a command that mostly inspects data.
    """
    body = (
        _molmobot_script(
            datasets=datasets,
            checkout=checkout,
            base_checkpoint=base_checkpoint,
            config_path=config_path,
            output_dir=output_dir,
            batch_size=batch_size,
            steps=steps,
            action_type=action_type,
            seq_len=seq_len,
            camera_names=camera_names,
            sample_rates=sample_rates,
            device_batch_size=device_batch_size,
        )
        if trainer == "molmobot" and checkout is not None
        else _generic_trainer_script(
            datasets=datasets,
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

    tasks = ", ".join(d.root.name for d in datasets)
    header = [
        "#!/usr/bin/env bash",
        "#",
        f"# Fine-tune {trainer} on {tasks}."
        + (" One policy across all of them, conditioned on each" if len(datasets) > 1 else ""),
        *(["# trajectory's task description."] if len(datasets) > 1 else []),
        "#",
        "# Generated by examples/machine_learning/molmospaces/finetuning/finetune.py.",
        "# Re-running that command overwrites this file; nothing reads it back, so",
        "# edit freely once you have it.",
        "",
        "set -euo pipefail",
        "",
        "step() {",
        '    echo',
        '    echo "------------------------------------------------------------"',
        '    echo "$*"',
        '    echo "------------------------------------------------------------"',
        "}",
        "",
        "# Ask before doing something the caller may not have meant. Anything without a",
        "# terminal on stdin -- CI, nohup, a pipeline -- cannot answer, so it proceeds",
        "# rather than hanging forever waiting on a read; ASSUME_YES=1 does the same",
        "# deliberately.",
        "confirm() {",
        '    if [ "${ASSUME_YES:-0}" = 1 ]; then',
        '        echo "  -> continuing (ASSUME_YES=1)"',
        "        return 0",
        "    fi",
        '    if [ ! -t 0 ]; then',
        '        echo "  -> continuing (stdin is not a terminal, cannot prompt)"',
        "        return 0",
        "    fi",
        "    printf '  %s [y/N] ' \"$1\"",
        "    read -r _reply",
        '    case "$_reply" in',
        "        [yY] | [yY][eE][sS]) return 0 ;;",
        '        *) echo "  Aborted. Nothing has been written."; exit 1 ;;',
        "    esac",
        "}",
        "",
    ]
    path = mixture_root(datasets) / f"run_{trainer}.sh"
    path.write_text("\n".join(header + body) + "\n")
    path.chmod(0o755)
    return path


def _molmobot_script(
    datasets: list[DatasetSummary],
    checkout: MolmoBotCheckout,
    base_checkpoint: str,
    config_path: Path,
    output_dir: Path,
    batch_size: int,
    steps: int,
    action_type: str,
    seq_len: int | None,
    camera_names: list[str] | None,
    sample_rates: list[float] | None = None,
    device_batch_size: int | None = None,
) -> list[str]:
    """The five steps between a prepared rollout run and a MolmoBot checkpoint."""
    package = checkout.package_dir.resolve()
    scripts = checkout.data_scripts_dir.resolve()
    variables = dataset_variables(datasets)

    validate = [
        _shell(
            [
                '"$PYTHON"',
                '"$SCRIPTS"/validate_trajectories.py',
                f'"${name}"/{split}',
                "--num-workers",
                "8",
            ]
        )
        for name, dataset in variables
        for split, count in dataset.splits.items()
        if count
    ]
    statistics = [
        _shell(
            [
                '"$PYTHON"',
                '"$SCRIPTS"/calculate_stats.py',
                f'"${name}"/train',
                "--keys",
                f"actions/{action_type}",
                "actions/joint_pos",
                "obs/agent/qpos",
            ]
        )
        for name, _ in variables
    ]
    training_command = _shell(
        trainer_command(
            datasets=datasets,
            trainer="molmobot",
            config_path=config_path,
            base_checkpoint='"$CHECKPOINT"',
            output_dir=output_dir,
            batch_size='"$GLOBAL_BATCH"',
            steps='"$MAX_STEPS"',
            action_type=action_type,
            seq_len='"$SEQ_LEN"',
            camera_names=camera_names,
            dataset_refs=[f'"${name}"' for name, _ in variables],
            sample_rates=sample_rates,
            save_folder_ref='"$SAVE_FOLDER"',
            device_batch_size='"$DEVICE_BATCH"',
            num_workers='"$NUM_WORKERS"',
            log_interval='"$LOG_INTERVAL"',
            extra_args=['"${EXTRA_ARGS[@]}"'],
            launcher=[
                '"$TORCHRUN"',
                "--nnodes=1",
                '--nproc-per-node="$NPROC"',
                "--master_port=29401",
            ],
        )
    )

    return [
        f"PACKAGE={shlex.quote(str(package))}",
        f"SCRIPTS={shlex.quote(str(scripts))}",
        *(f"{name}={shlex.quote(str(d.root.resolve()))}" for name, d in variables),
        # Overridable so a sweep can point several runs of this one script at
        # save folders of their own -- see hparam_probe.py.
        'SAVE_FOLDER="${SAVE_FOLDER:-'
        + shlex.quote(str((output_dir / experiment_name(datasets)).resolve()))
        + '}"',
        f"PROGRESS_FILTER={shlex.quote(str(PROGRESS_FILTER))}",
        f"TRAINING_REPORT={shlex.quote(str(TRAINING_REPORT))}",
        f"REPORT_PYTHON={shlex.quote(str(REPORT_PYTHON))}",
        'PYTHON="$PACKAGE"/.venv/bin/python',
        'TORCHRUN="$PACKAGE"/.venv/bin/torchrun',
        "",
        *_tuning_block(
            seq_len,
            device_batch_size,
            batch_size,
            steps,
            cameras=camera_names or datasets[0].video_keys or list(DEFAULT_CAMERA_NAMES),
        ),
        'cd "$PACKAGE"',
        "",
        "# --------------------------------------------------------------------------",
        "# 1. MolmoBot's virtualenv.",
        "#",
        "#    `--extra train` is the extra that carries h5py, which both postprocessing",
        "#    scripts import; decord and tqdm are already core dependencies. This is the",
        "#    step that downloads torch, so it is the slow one.",
        "#",
        "#    SYNC=auto runs it only when there is no virtualenv yet. That default is",
        "#    there to protect a hand-built one: on a 5090 the wheels MolmoBot resolves",
        "#    have no Blackwell (sm_120) kernels, so the venv gets torch reinstalled",
        "#    from the cu128 index by hand -- and `uv sync` would put the resolved",
        "#    version back, silently, costing the GPU. SYNC=on to sync anyway.",
        "# --------------------------------------------------------------------------",
        'if [ "$SYNC" = on ] || { [ "$SYNC" = auto ] && [ ! -x "$PYTHON" ]; }; then',
        '    step "Syncing MolmoBot dependencies"',
        "    uv sync --extra train",
        '    if [ "$OPTIMIZER" = adamw8bit ]; then',
        "        # --no-deps deliberately, and not in MolmoBot's pyproject.toml at all:",
        "        # resolved as a project dependency, torchao pins torch back to a release",
        "        # without Blackwell (sm_120) wheels, which silently costs you the GPU.",
        "        # The 8-bit optimizers are pure PyTorch plus torch.compile, so nothing",
        "        # torchao's own resolution would add is needed.",
        "        uv pip install --no-deps torchao",
        "    fi",
        'elif [ "$SYNC" = auto ]; then',
        '    echo "MolmoBot venv already there; not syncing (SYNC=on to sync anyway)."',
        "fi",
        "",
        *_checkpoint_step(base_checkpoint, output_dir),
        "# --------------------------------------------------------------------------",
        "# 3. valid_trajectory_index.json, once per split of every task.",
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
        'if [ "$PREPARE" = on ]; then',
        'step "Validating trajectories"',
        *validate,
        "fi",
        "",
        "# --------------------------------------------------------------------------",
        "# 4. Per-trajectory statistics, into a `stats` group in each HDF5 plus an",
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
        'if [ "$PREPARE" = on ]; then',
        'step "Calculating statistics"',
        *statistics,
        "fi",
        "",
        "# --------------------------------------------------------------------------",
        "# 5. Train.",
        "#",
        "#    Everything tunable comes from the block at the top of this file.",
        "#",
        "#    --save_folder and --max_duration are not argparse options:",
        "#    train_molmobot.py builds a TrainConfig and merges leftover --name=value",
        "#    arguments into it as an OmegaConf dotlist. save_folder is that config's",
        "#    only mandatory field, so leaving it off fails at OmegaConf.to_object --",
        "#    after the checkpoint has loaded and the statistics have been computed.",
        "#",
        "#    --stats_path is kept inside SAVE_FOLDER rather than at its relative",
        "#    default, which would write it into the shared MolmoBot checkout and",
        "#    silently reuse one mixture's normalisation for the next. Delete it to",
        "#    recompute after the data changes.",
        *(
            [
                "#",
                "#    --val_data_paths is passed explicitly because MolmoBot otherwise",
                "#    validates on the first --data_paths entry alone, which would report",
                f"#    {datasets[0].root.name}'s loss for a run training on all of them.",
            ]
            if len(datasets) > 1
            else []
        ),
        "# --------------------------------------------------------------------------",
        "# 5a. Preflight: say where things land, and ask about the two ways this run",
        "#     can quietly not be what was intended -- too many passes over the data,",
        "#     or resuming someone else's run.",
        "# --------------------------------------------------------------------------",
        'step "Preflight"',
        "",
        'TRAIN_FRAMES=$("$PYTHON" - "$DATASET"/train/valid_trajectory_index.json <<\'PY\'',
        "import json, sys",
        "index = json.load(open(sys.argv[1]))",
        "print(sum(n for house in index.values() for f in house.values() for n in f.values()))",
        "PY",
        ")",
        "SAMPLES=$((MAX_STEPS * GLOBAL_BATCH))",
        'EPOCHS=$("$PYTHON" -c "print(f\'{$SAMPLES / $TRAIN_FRAMES:.2f}\')")',
        "",
        'echo "checkpoints:  $SAVE_FOLDER"',
        'echo "              step<N>/ every save_interval, plus step<N>_bestfit/ at the"',
        'echo "              best validation loss (kept, never rotated away)"',
        'echo "data:         $TRAIN_FRAMES train frames"',
        'echo "run:          $MAX_STEPS steps x $GLOBAL_BATCH = $SAMPLES samples'
        ' = $EPOCHS passes over the data"',
        'echo "optimizer:    $OPTIMIZER      trainable: $TRAINABLE"',
        "",
        "# `bc` is not guaranteed to be installed; python is, and is already resolved.",
        'if [ "$("$PYTHON" -c "print(int($EPOCHS > $MAX_EPOCHS_WARN))")" = 1 ]; then',
        "    echo",
        '    echo "  WARNING: $EPOCHS passes over $TRAIN_FRAMES frames exceeds'
        ' MAX_EPOCHS_WARN=$MAX_EPOCHS_WARN."',
        '    echo "  For a single task this is well into the range where the policy memorises"',
        '    echo "  trajectories instead of learning the skill. Validation loss is the check:"',
        '    echo "  if it flattens while training loss keeps falling, that is what happened."',
        "    echo",
        '    echo "  To train fewer passes, set MAX_STEPS (which also shortens the LR schedule"',
        '    echo "  to match, so the run still finishes its decay):"',
        '    echo "    MAX_STEPS=$("$PYTHON" -c'
        ' "print(int($MAX_EPOCHS_WARN * $TRAIN_FRAMES / $GLOBAL_BATCH))") bash $0"',
        '    confirm "Run $MAX_STEPS steps anyway?"',
        "fi",
        "",
        "# Best-fit checkpointing lives in the MolmoBot checkout, which is gitignored and",
        "# recreated from scratch if it is ever deleted. The exports above would then be",
        "# read by nobody and the only surviving checkpoint would be the newest one --",
        "# a silent downgrade, and exactly the thing this is meant to prevent. Check.",
        'if [ "$MOLMOBOT_BESTFIT" = 1 ] && ! "$PYTHON" -c \\',
        '    "from olmo.train.trainer import Trainer',
        "raise SystemExit(0 if hasattr(Trainer, 'maybe_save_bestfit_checkpoint') else 1)\" 2>/dev/null; then",
        "    echo",
        '    echo "  WARNING: this MolmoBot checkout has no best-fit checkpointing, so no"',
        '    echo "  step<N>_bestfit/ will be written and save_num_checkpoints_to_keep will"',
        '    echo "  leave you only the newest step<N>/ -- not the best one."',
        '    echo "  The checkout was probably recreated. Re-run finetune.py to reapply it,"',
        '    echo "  or set BESTFIT=0 to stop advertising it here."',
        '    confirm "Train without best-fit checkpointing?"',
        "fi",
        "",
        "# run_trainer.py resumes from the newest stepN/ in the save folder whenever",
        "# allow_resume is on, which it is by default -- it does not start over. So a",
        "# Ctrl-C'd run continues from its last *saved* step, losing whatever came after",
        "# it, and a save folder from a different experiment gets silently continued.",
        'LATEST_CHECKPOINT=""',
        'if [ -d "$SAVE_FOLDER" ]; then',
        "    LATEST_CHECKPOINT=$(find \"$SAVE_FOLDER\" -maxdepth 1 -type d -name 'step*' 2>/dev/null |",
        "        sed 's/.*\\/step//' | sed 's/_bestfit//' | sort -n | tail -1)",
        "fi",
        'if [ -n "$LATEST_CHECKPOINT" ]; then',
        "    echo",
        '    echo "  This save folder already holds a checkpoint at step $LATEST_CHECKPOINT."',
        '    echo "  Training will RESUME from it, not restart, and will overwrite the"',
        '    echo "  checkpoints after it. Anything between step $LATEST_CHECKPOINT and where the"',
        '    echo "  previous run stopped is already gone."',
        "    echo",
        '    echo "  To start clean instead, move or delete the folder first:"',
        '    echo "    mv $SAVE_FOLDER $SAVE_FOLDER.$(date +%Y%m%d-%H%M%S)"',
        '    confirm "Resume from step $LATEST_CHECKPOINT?"',
        "else",
        "    echo",
        '    echo "  No checkpoint in the save folder; starting from $CHECKPOINT."',
        "fi",
        "",
        "# --------------------------------------------------------------------------",
        'step "Starting training, this may take a while"',
        'echo "GPUs: $NPROC x ${VRAM_MIB}MiB   seq_len=$SEQ_LEN'
        ' device_batch=$DEVICE_BATCH global_batch=$GLOBAL_BATCH"',
        'mkdir -p "$SAVE_FOLDER"',
        'export PYTHONPATH="$PACKAGE${PYTHONPATH:+:$PYTHONPATH}"',
        "",
        'if [ "$OPTIMIZER" = adamw8bit ]; then',
        "    # Do not drop this: with adamw8bit on a 32GiB card the run OOMs without it.",
        "    # 8-bit moments get the run to fit, but only just -- about 28GiB deep -- and",
        "    # the default allocator strands ~2GiB in reserved-but-unallocated blocks,",
        "    # enough that the first backward cannot find a contiguous 2GiB even though",
        "    # the memory is free. expandable_segments grows existing segments instead.",
        "    # Measured both ways on a 5090: OOM without it, trains with it, nothing else",
        "    # changed. Only set here because this is the configuration that needs it --",
        "    # adamw has margin to spare on a card big enough to run it at all.",
        '    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"',
        "fi",
        "",
        "# The trainer already knows how far along it is -- Trainer.fit prints",
        "# `[step=N/max, eta=...]` every LOG_INTERVAL steps -- but as the first line",
        "# of a multi-line metrics dump, so it scrolls past. train_progress.py passes",
        "# every line through untouched and appends a bar when it sees one.",
        "#",
        "# `set -o pipefail` is on, so a training failure still fails this script even",
        "# though the trainer is no longer the last command in the pipeline. 2>&1",
        "# because the headers go to the logging handler's stream, not stdout.",
        "# METRICS is read by the recorder molmobot_repo.ensure_metrics_log patched",
        "# into the trainer: one JSON line per metrics dump, with the numbers the",
        "# console filters out -- the per-group learning rates and gradient norms, and",
        "# the validation loss, which otherwise only exists in W&B. Unset it and the",
        "# recorder returns immediately.",
        'if [ "$METRICS" != off ]; then',
        f'    export {METRICS_ENV_VAR}="$METRICS"',
        '    echo "metrics:      $METRICS"',
        "fi",
        "",
        "# The live plot. Backgrounded, so it redraws while the trainer has the GPU,",
        "# and killed on the way out however this script ends -- a stray watcher",
        "# polling a finished run is a confusing thing to find days later.",
        'if [ "$METRICS" != off ] && [ "$PLOT" != off ]; then',
        '    if [ -x "$REPORT_PYTHON" ]; then',
        '        "$REPORT_PYTHON" "$TRAINING_REPORT" "$METRICS" --plot "$PLOT" --watch &',
        "        PLOT_PID=$!",
        "        # shellcheck disable=SC2064  -- PLOT_PID is wanted now, not at signal time",
        "        trap \"kill $PLOT_PID 2>/dev/null || true\" EXIT",
        '        echo "plot:         $PLOT (redrawn as the run goes)"',
        "    else",
        '        echo "plot:         skipped -- no interpreter at $REPORT_PYTHON."',
        '        echo "              Plots need matplotlib, which MolmoBot\'s venv has no"',
        '        echo "              reason to carry; re-run finetune.py to regenerate this"',
        '        echo "              script against the interpreter you have now."',
        "    fi",
        "fi",
        "",
        'if [ "$PROGRESS" = on ]; then',
        f"    {training_command} 2>&1 | \"$PYTHON\" -u \"$PROGRESS_FILTER\"",
        "else",
        f"    {training_command}",
        "fi",
        "",
        "# --------------------------------------------------------------------------",
        "# 6. What the run did.",
        "#",
        "#    Reads the metrics file back and says whether it converged, whether it",
        "#    overfitted, and whether the learning rates look right -- the questions",
        "#    that are unanswerable from a scrollback of loss values. Re-runnable at",
        "#    any time, including while a run is still going.",
        "# --------------------------------------------------------------------------",
        "# The watcher stops first: the last evaluation of the run lands after its",
        "# final redraw, and the report below is what puts it in the picture.",
        'if [ -n "${PLOT_PID:-}" ]; then',
        '    kill "$PLOT_PID" 2>/dev/null || true',
        '    wait "$PLOT_PID" 2>/dev/null || true',
        "    PLOT_PID=",
        "fi",
        'if [ "$METRICS" != off ] && [ -s "$METRICS" ]; then',
        '    if [ "$PLOT" != off ] && [ -x "$REPORT_PYTHON" ]; then',
        '        "$REPORT_PYTHON" "$TRAINING_REPORT" "$METRICS" --plot "$PLOT" || true',
        "    else",
        '        "$PYTHON" "$TRAINING_REPORT" "$METRICS" || true',
        "    fi",
        "fi",
        "",
        "# Then score the checkpoint, back in the stretch4_mujoco repo -- natively, with",
        "# no action remapping:",
        *(f"#   {_evaluation_command(d, 'molmobot')}" for d in datasets),
        f"# serving it with `--action-type {action_type}`, matching what it trained on.",
    ]


def _tuning_block(
    seq_len: int | None,
    device_batch_size: int | None,
    batch_size: int,
    steps: int,
    cameras: list[str] | None = None,
) -> list[str]:
    """The knobs, at the top of the script, sized from whatever GPU is present.

    Everything here is `${NAME:-default}`, so a value can be overridden from the
    environment for one run (`SEQ_LEN=1024 bash run_molmobot.sh`) or edited in
    place to keep it. `seq_len` and `device_batch_size` of None mean "size it
    from the detected VRAM"; a number pins it and skips the table.
    """
    tiers: list[str] = []
    for index, (floor, tier_seq, tier_batch, label) in enumerate(VRAM_TIERS):
        assignment = f"SEQ_LEN_AUTO={tier_seq}; DEVICE_BATCH_AUTO={tier_batch}"
        # The last row is an unconditional `else`, so the variables are always
        # set -- a `-ge 0` test that somehow failed would leave them unbound and
        # `set -u` would kill the run at the training line rather than here.
        if floor == 0 or index == len(VRAM_TIERS) - 1:
            tiers.append(f"else {assignment}  # {label}")
            break
        keyword = "if" if index == 0 else "elif"
        tiers.append(f'{keyword} [ "$VRAM_MIB" -ge {floor} ]; then {assignment}  # {label}')
    tiers.append("fi")

    return [
        "# ==========================================================================",
        "# Tuning. Everything worth changing is in this block.",
        "# ==========================================================================",
        "# Any of these can be set for one run without editing the file:",
        "#",
        "#     SEQ_LEN=1024 DEVICE_BATCH=2 bash <this script>",
        "#",
        "# Steps 1-4 below are all idempotent, so re-running after a change repeats",
        "# no work: the venv is synced, the checkpoint is on disk, and the indices",
        "# and statistics are rebuilt in seconds.",
        "",
        "# --- What hardware is actually here ---------------------------------------",
        "# The smallest GPU's memory, because every rank runs the same microbatch.",
        'NPROC="${NPROC:-$(nvidia-smi --list-gpus 2>/dev/null | wc -l)}"',
        '[ "${NPROC:-0}" -ge 1 ] 2>/dev/null || NPROC=1',
        'VRAM_MIB="${VRAM_MIB:-$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits'
        ' 2>/dev/null | sort -n | head -1)}"',
        '[ "${VRAM_MIB:-0}" -ge 1 ] 2>/dev/null || VRAM_MIB=0',
        "",
        "# --- The two VRAM knobs, sized from that ----------------------------------",
        "# SEQ_LEN is first because the loader is built with pad=to_max: the",
        "# preprocessor derives fixed output shapes from it, so *every sample is",
        "# padded to it* whether or not any trajectory needs the room. Halving it",
        "# roughly halves activation memory. 528 is what MolmoBot's README uses for",
        "# this exact shape -- two images, crop_mode=resize, max_crops=1, 3x3 pooling.",
        "#",
        "# DEVICE_BATCH is samples per forward/backward. split_batch() chops the",
        "# per-device batch into ceil(batch/DEVICE_BATCH) microbatches with no",
        "# divisibility requirement, so 1 is always valid. MolmoBot's default is 2.",
        "#",
        "# These rows are starting points, not measurements -- extrapolated from",
        "# MolmoBot's own recipe with margin for the resident 4B backbone. Turn them",
        "# down for more cameras; up if the run fits with room to spare.",
        *tiers,
        "",
        f'SEQ_LEN="${{SEQ_LEN:-{seq_len if seq_len is not None else "$SEQ_LEN_AUTO"}}}"',
        'DEVICE_BATCH="${DEVICE_BATCH:-'
        f'{device_batch_size if device_batch_size is not None else "$DEVICE_BATCH_AUTO"}}}"',
        "",
        "# The effective batch, reached by accumulating gradients over microbatches.",
        "# This changes the optimisation, not the peak memory -- lower it only if you",
        "# mean to train differently.",
        f'GLOBAL_BATCH="${{GLOBAL_BATCH:-{batch_size}}}"',
        "",
        "# MAX_STEPS also sets the learning-rate horizon, not just where training stops:",
        "# scheduler.t_max is unset, so the cosine decay to alpha_f spans exactly this",
        "# many steps. Halving it gives a complete shorter schedule; interrupting a",
        "# longer one leaves the run stranded mid-decay at a high LR. Change the number,",
        "# do not Ctrl-C early.",
        "#",
        "# The preflight below works the real number of passes out from the trajectory",
        "# index and asks before running anything wildly over MAX_EPOCHS_WARN.",
        f'MAX_STEPS="${{MAX_STEPS:-{steps}}}"',
        "",
        "# Passes over the training frames that trigger the preflight prompt. Above",
        "# roughly ten, a 5B VLA fine-tune on one task is likelier to be memorising",
        "# than learning -- but it depends on the data, so this asks rather than caps.",
        'MAX_EPOCHS_WARN="${MAX_EPOCHS_WARN:-10}"',
        "",
        "# Dataloader worker processes. Host RAM and CPU, not VRAM -- lower it if the",
        "# machine starts swapping while the GPU sits idle.",
        'NUM_WORKERS="${NUM_WORKERS:-4}"',
        "",
        "# --- Progress reporting ----------------------------------------------------",
        "# How often the trainer prints its step header, which is also how often the",
        "# progress bar can redraw. Every header costs a host-device sync for the",
        "# metrics behind it, so this is a real (small) throughput trade.",
        'LOG_INTERVAL="${LOG_INTERVAL:-20}"',
        "",
        "# PREPARE=off skips the two steps that only depend on the *data* -- the",
        "# trajectory index and the statistics pass. They are idempotent, so skipping",
        "# them changes nothing except the minutes they take; it is what makes running",
        "# this script several times over one dataset cheap, which is what a",
        "# hyperparameter sweep does.",
        'PREPARE="${PREPARE:-on}"',
        "",
        "# SYNC=auto installs MolmoBot's dependencies only when its virtualenv is",
        "# missing, so a torch rebuilt by hand for this GPU is not silently replaced by",
        "# the one MolmoBot resolves. on syncs every run; off never does.",
        'SYNC="${SYNC:-auto}"',
        "",
        "# The normalisation statistics the trainer computes at startup. Inside",
        "# SAVE_FOLDER by default so two mixtures cannot share one file by accident;",
        "# point several runs over the *same* data at one path to compute it once.",
        'STATS_PATH="${STATS_PATH:-$SAVE_FOLDER/synthmanip_norm_stats.yaml}"',
        "",
        "# Where the metrics recorder writes, one JSON line per dump; METRICS=off turns",
        "# it off. training_report.py reads this file -- and so does hparam_probe.py,",
        "# which is how two runs at different learning rates are compared.",
        'METRICS="${METRICS:-$SAVE_FOLDER/metrics.jsonl}"',
        "",
        "# PLOT is a picture of the run, redrawn every time the metrics file grows --",
        "# loss, per-move-group action loss, learning rates, gradient norms, throughput.",
        "# Written to the same path throughout, so an image viewer left open on it",
        "# follows the run. PLOT=off skips it; it costs a second of CPU per redraw and",
        "# nothing of the GPU.",
        'PLOT="${PLOT:-$SAVE_FOLDER/training.png}"',
        "",
        "# PROGRESS=off pipes the trainer straight to the terminal instead of through",
        "# train_progress.py. The filter only passes lines through and appends a bar,",
        "# but a pipe is a pipe -- turn it off if you are debugging output ordering.",
        'PROGRESS="${PROGRESS:-on}"',
        "",
        "# --- Weights & Biases ------------------------------------------------------",
        "# Off by default, and not merely for quiet: the trainer's wandb config",
        "# interpolates ${oc.env:WANDB_PROJECT} and ${oc.env:WANDB_ENTITY}, and both",
        "# are resolved in the same call that validates the rest of the config -- so",
        "# with either unset the run dies *after* loading the checkpoint and computing",
        "# the normalisation statistics. Set WANDB=on with both exported to log.",
        'WANDB="${WANDB:-off}"',
        "",
        *_lens_block(cameras or []),
        "# --- What actually trains --------------------------------------------------",
        "#   action_expert  (default) only the action expert. Cheapest.",
        "#   vision         + the vision tower, at VIT_LR.",
        "#   full           + the LLM and connector. MolmoBot's own recipe, on 8-64 H100s.",
        'TRAINABLE="${TRAINABLE:-vision}"',
        "",
        "# Per-component learning rates, all of them OmegaConf dotlist fields on",
        "# TrainConfig.optimizer.",
        'ACTION_EXPERT_LR="${ACTION_EXPERT_LR:-1e-4}"',
        'VIT_LR="${VIT_LR:-5e-6}"',
        'LLM_LR="${LLM_LR:-1e-5}"',
        "",
        "# --- Where the optimizer state lives --------------------------------------",
        "#   adamw8bit  (default) torchao's block-wise 8-bit AdamW. Same update rule",
        "#              as adamw, moments quantized: ~2 bytes/param of trainable",
        "#              weight instead of 8. At TRAINABLE=vision that is 918M",
        "#              trainable params costing 1.7GiB rather than 6.8GiB, which is",
        "#              what makes the vision tower fit on a single 32GiB card.",
        "#              Needs torchao, installed by step 1 when this is selected.",
        "#   adamw      fp32 moments. The stock choice when the memory is there.",
        "#",
        "# torchao and not bitsandbytes because this trainer runs FSDP2, so",
        "# parameters are DTensors: torchao's optimizers shard their quantized state",
        "# with the parameter, bitsandbytes' cannot.",
        'OPTIMIZER="${OPTIMIZER:-adamw8bit}"',
        "",
        "# --- Validation, and keeping the checkpoint that is actually best -----------",
        "# How often to run the held-out split. Each pass is the whole val set, and it",
        "# is the only honest read on whether the policy is still learning or has",
        "# started memorising.",
        'EVAL_INTERVAL="${EVAL_INTERVAL:-500}"',
        "",
        "# save_num_checkpoints_to_keep keeps the *newest* step<N>/, which on a fine-tune",
        "# is reliably not the best one: validation loss bottoms out partway through and",
        "# drifts up after. So the trainer also writes step<N>_bestfit/ whenever the",
        "# monitored validation metric improves, under its own retention -- the rotation",
        "# cannot delete it.",
        "#",
        "# BESTFIT_METRIC is action_flow_loss, the flow-matching loss on the actions,",
        "# because that is what is being trained. CrossEntropyLoss and Accuracy in the",
        "# same dump belong to the language head, which is frozen here, and watching",
        "# them would tell you nothing.",
        "#",
        "# After BESTFIT_PATIENCE evaluations with no improvement worth",
        "# BESTFIT_MIN_DELTA (relative, so 0.005 = 0.5%), the run says so in the log and",
        "# names the best checkpoint. It keeps training to MAX_STEPS -- the message is",
        "# the signal to stop, not an automatic stop.",
        'export MOLMOBOT_BESTFIT="${BESTFIT:-1}"',
        'export MOLMOBOT_BESTFIT_METRIC="${BESTFIT_METRIC:-action_flow_loss}"',
        'export MOLMOBOT_BESTFIT_PATIENCE="${BESTFIT_PATIENCE:-4}"',
        'export MOLMOBOT_BESTFIT_MIN_DELTA="${BESTFIT_MIN_DELTA:-0.005}"',
        "",
        "# --- Optional training flags ----------------------------------------------",
        "# Uncomment a line to turn one on. They are appended verbatim to the trainer.",
        "EXTRA_ARGS=()",
        "",
        *_optional_flag_lines(),
        "if [ \"$WANDB\" = on ]; then",
        '    : "${WANDB_PROJECT:?WANDB=on needs WANDB_PROJECT exported}"',
        '    : "${WANDB_ENTITY:?WANDB=on needs WANDB_ENTITY exported}"',
        "    export WANDB_PROJECT WANDB_ENTITY",
        "else",
        "    EXTRA_ARGS+=(--wandb=null)",
        "fi",
        "",
        "# run_trainer.py freezes a component when its ft_* flag is false, so the",
        "# defaults below are the frozen case and each tier only adds.",
        'case "$TRAINABLE" in',
        "    action_expert) ;;",
        "    vision)  EXTRA_ARGS+=(--ft_vit=True) ;;",
        "    full)    EXTRA_ARGS+=(--ft_vit=True --ft_llm=True --ft_connector=True) ;;",
        "    *)",
        '        echo "TRAINABLE must be action_expert, vision or full (got: $TRAINABLE)" >&2',
        "        exit 1",
        "        ;;",
        "esac",
        'case "$OPTIMIZER" in',
        "    adamw|adamw8bit) ;;",
        "    *)",
        '        echo "OPTIMIZER must be adamw or adamw8bit (got: $OPTIMIZER)" >&2',
        "        exit 1",
        "        ;;",
        "esac",
        "EXTRA_ARGS+=(",
        '    --optimizer.name="$OPTIMIZER"',
        '    --val_interval "$EVAL_INTERVAL"',
        '    --optimizer.action_expert_learning_rate="$ACTION_EXPERT_LR"',
        '    --optimizer.vit_learning_rate="$VIT_LR"',
        '    --optimizer.llm_learning_rate="$LLM_LR"',
        ")",
        "",
        'if [ -n "$WARP_CAMERAS" ]; then',
        "    # Unquoted on purpose: WARP_CAMERAS is a space-separated list and",
        "    # --cameras_to_warp takes nargs=*.",
        "    # shellcheck disable=SC2206",
        "    EXTRA_ARGS+=(--cameras_to_warp $WARP_CAMERAS)",
        "fi",
        "",
    ]


def _lens_block(cameras: list[str]) -> list[str]:
    """The wide-angle section, named after the cameras this run actually uses.

    Written per-run rather than as a general note because the correct answer
    depends on which cameras were selected, and the wrong answer is invisible:
    warping an already-fisheye frame produces an image that still looks like a
    plausible wide-angle photo.
    """
    fisheye = [camera for camera in cameras if camera in FISHEYE_CAMERA_NAMES]
    rectilinear = [camera for camera in cameras if camera not in FISHEYE_CAMERA_NAMES]

    lines = [
        "# --- Wide-angle lenses ------------------------------------------------------",
    ]
    if fisheye:
        carries = "already carries" if len(fisheye) == 1 else "already carry"
        lines += [
            f"# {', '.join(fisheye)} {carries} the wide lens: stretch/config.py gives",
            "# it a ~123 degree FOV and runs a barrel-distortion callback on every frame at",
            "# render time, so the distortion is baked into the recorded MP4s.",
            "#",
            "# This is why WARP_CAMERAS is empty. MolmoBot's --cameras_to_warp applies",
            "# *another* GoPro-style barrel distortion, and pointing it at that camera",
            "# would train the policy on a lens nothing has -- while still producing images",
            "# that look like plausible wide-angle photos, so nothing would flag it.",
        ]
        if rectilinear:
            is_are = "is rectilinear" if len(rectilinear) == 1 else "are rectilinear"
            lines += [
                "#",
                f"# {', '.join(rectilinear)} {is_are} (MJCF default FOV, no distortion",
                "# callback). Warping that is the flag's real use: it is for a camera that",
                "# renders rectilinear in simulation but is wide-angle on the robot.",
            ]
    else:
        renders = "renders" if len(cameras) == 1 else "render"
        lines += [
            f"# {', '.join(cameras) or 'The selected cameras'} {renders} rectilinear -- MJCF",
            "# default FOV, no distortion callback. If the corresponding camera on the",
            "# robot is wide-angle, --cameras_to_warp closes that sim2real gap by applying",
            "# GoPro-style barrel distortion during training. Stretch's head_camera_left",
            "# and head_camera_right are already distorted at render time and must never",
            "# be listed here.",
        ]
    lines += [
        "#",
        "# Space-separated camera names to warp, e.g. WARP_CAMERAS=\"head_camera\".",
        'WARP_CAMERAS="${WARP_CAMERAS:-}"',
        "",
    ]
    return lines


def _optional_flag_lines() -> list[str]:
    """`MOLMOBOT_OPTIONAL_FLAGS` as commented-out `EXTRA_ARGS+=(...)` lines."""
    lines: list[str] = []
    for flag, explanation in MOLMOBOT_OPTIONAL_FLAGS:
        lines.extend(f"# {sentence}" for sentence in explanation)
        lines.append(f"# EXTRA_ARGS+=({flag})")
        lines.append("")
    return lines


def _checkpoint_step(base_checkpoint: str, output_dir: Path) -> list[str]:
    """Put the base checkpoint on disk, because the trainer will not fetch it.

    `train_molmobot.py`'s help says its positional argument is a "Path to
    checkpoint or '8b' for base model", but nothing maps `8b` to anything --
    `select_checkpoint` calls `os.listdir` straight on the string, so `8b` fails
    with a bare `FileNotFoundError: '8b'`. A local directory is the only thing
    that works, and MolmoBot's released checkpoints are HuggingFace repositories
    of exactly the `model.pt` + `config.yaml` that `select_checkpoint` and
    `get_model` want, so downloading one is the whole step.
    """
    header = [
        "# --------------------------------------------------------------------------",
        "# 2. The base checkpoint.",
        "#",
        "#    train_molmobot.py takes a local directory holding model.pt and",
        "#    config.yaml; it resolves no names and downloads nothing itself. Its own",
        "#    `8b` placeholder is not handled anywhere and fails as a missing path.",
    ]
    if not is_hf_repo_id(base_checkpoint):
        return [
            *header,
            "# --------------------------------------------------------------------------",
            f"CHECKPOINT={shlex.quote(base_checkpoint)}",
            'if [ ! -e "$CHECKPOINT"/model.pt ] && [ ! -d "$CHECKPOINT"/model_and_optim ]; then',
            '    echo "No checkpoint at $CHECKPOINT (expected model.pt or model_and_optim/)." >&2',
            "    exit 1",
            "fi",
            "",
        ]
    local = (output_dir.parent / "base_checkpoints" / base_checkpoint.split("/")[-1]).resolve()
    return [
        *header,
        "#",
        "#    To start from the Molmo2 VLM instead of a trained MolmoBot policy, follow",
        "#    MolmoBot's README: untar Molmo2-4B and run launch_scripts/convert_to_",
        "#    unsharded.py, then point CHECKPOINT at the unsharded directory. That is",
        "#    the from-scratch recipe and wants far more data than a fine-tune does.",
        "#",
        "#    CHECKPOINT is also how a run continues from weights of your own:",
        "#",
        "#      CHECKPOINT=<save folder>/step9500_bestfit \\",
        "#      SAVE_FOLDER=<save folder>_v2 MAX_STEPS=30000 bash run_molmobot.sh",
        "#",
        "#    select_checkpoint takes either shape -- a model.pt or a model_and_optim/ --",
        "#    and a checkpoint passed this way arrives as initial_model_checkpoint, which",
        "#    resets the optimizer and the step counter. So the weights carry over and",
        "#    the run gets one clean schedule over the new MAX_STEPS, instead of the",
        "#    learning-rate jump a resume into a longer horizon produces. Give it a save",
        "#    folder of its own, or allow_resume will find the old run's step<N>/ and",
        "#    resume from that instead.",
        "# --------------------------------------------------------------------------",
        f"CHECKPOINT_DEFAULT={shlex.quote(str(local))}",
        'CHECKPOINT="${CHECKPOINT:-$CHECKPOINT_DEFAULT}"',
        'if [ ! -e "$CHECKPOINT"/model.pt ] && [ ! -d "$CHECKPOINT"/model_and_optim ]; then',
        '    if [ "$CHECKPOINT" != "$CHECKPOINT_DEFAULT" ]; then',
        "        # Only the default is downloadable; anything else was named by hand,",
        "        # and filling a mistyped path with a 19GB download helps nobody.",
        '        echo "No checkpoint at $CHECKPOINT (expected model.pt or model_and_optim/)." >&2',
        "        exit 1",
        "    fi",
        f'    step "Downloading {base_checkpoint}"',
        f'    "$PACKAGE"/.venv/bin/hf download {shlex.quote(base_checkpoint)} '
        '--local-dir "$CHECKPOINT"',
        "fi",
        "",
    ]


def _generic_trainer_script(
    datasets: list[DatasetSummary],
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
                datasets=datasets,
                trainer=trainer,
                config_path=config_path,
                base_checkpoint=base_checkpoint,
                output_dir=output_dir,
                batch_size=batch_size,
                steps=steps,
                action_type=action_type,
                seq_len=seq_len,
                camera_names=camera_names,
                dataset_refs=[str(datasets[0].root.resolve())],
            )
        ),
        "",
        f"# {_evaluation_command(datasets[0], trainer)}",
    ]


def _evaluation_command(summary: DatasetSummary, trainer: str) -> str:
    """How to score the resulting checkpoint on the benchmarks."""
    if trainer == "molmobot" and summary.kind == "molmospaces":
        # The rollout directory is named after the task it was generated for, and
        # the benchmark keys are the same names, so a mixture scores one task per
        # line rather than always claiming to be `pick`.
        return (
            "python -m examples.machine_learning.molmospaces.run_benchmarks "
            f"--policy molmobot --checkpoint <checkpoint> --benchmark {summary.root.name}"
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
    multiple=True,
    help="A raw MolmoSpaces rollout run (`house_*/trajectories*.h5`). What --trainer "
    "molmobot wants, and what generate_dataset.py writes under `rollouts/`. Repeat it "
    "to train one policy across several tasks -- MolmoBot conditions on each "
    "trajectory's task description, so pick and pnp share a checkpoint rather than "
    "needing one each.",
)
@click.option(
    "--sample-rates",
    type=str,
    default=None,
    help="Comma-separated mixture weight per --rollouts, in the same order (e.g. "
    "'0.6,0.4'). Equal weighting when omitted. Only meaningful with several tasks.",
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
    help=f"Checkpoint to fine-tune from. Defaults to '{DEFAULT_MOLMOBOT_CHECKPOINT}' for "
    "molmobot and 'pi05_droid' otherwise. An `org/name` is downloaded from HuggingFace "
    "by the generated script; anything else is used as a local checkpoint directory. "
    "The released Franka-space model's vision and language weights carry over, and its "
    "action head is re-learned on Stretch's move groups.",
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
@click.option(
    "--seq-len",
    type=int,
    default=None,
    help="MolmoBot --seq_len. By default the generated script sizes this from the GPU it "
    "finds at run time; pass a number to pin it. Every sample is padded to it, so it is "
    "the first thing to lower when CUDA runs out of memory -- not a ceiling that costs "
    "nothing when unused.",
)
@click.option(
    "--device-batch-size",
    type=int,
    default=None,
    help="MolmoBot --device_batch_size: samples per forward/backward. Sized from the "
    "detected GPU by default; pass a number to pin it. Gradient accumulation makes "
    "--batch-size up regardless, so this trades speed for VRAM without changing the "
    "optimisation.",
)
@click.option("--batch-size", type=int, default=32, help="MolmoBot --global_batch_size.")
@click.option(
    "--steps",
    type=int,
    default=10000,
    help="Optimizer steps, which is also the learning-rate horizon. 10000 x a global "
    "batch of 32 is roughly two passes over a 150k-frame task; the generated script's "
    "preflight computes the real number and asks before running far more.",
)
@click.option("--learning-rate", type=float, default=1e-5)
@click.option(
    "--clone/--no-clone",
    default=True,
    help="Clone MolmoBot into --trainer-repo when it is not there, and download the "
    "two postprocessing scripts that ship with its HuggingFace dataset rather than "
    "its git repository. --no-clone makes a missing checkout an error instead.",
)
def main(
    rollouts: tuple[Path, ...],
    sample_rates: str | None,
    dataset: Path | None,
    cameras: str | None,
    trainer: str,
    trainer_repo: Path | None,
    base_checkpoint: str | None,
    output_dir: Path | None,
    action_type: str,
    val_fraction: float,
    link: bool,
    seq_len: int | None,
    device_batch_size: int | None,
    batch_size: int,
    steps: int,
    learning_rate: float,
    clone: bool,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if bool(rollouts) == (dataset is not None):
        raise click.UsageError(
            "Pass --rollouts (one or more MolmoSpaces runs, for --trainer molmobot) or "
            "--dataset (an exported LeRobot dataset, for --trainer openpi/lerobot), not both."
        )
    if trainer == "molmobot" and not rollouts:
        raise click.UsageError(
            "--trainer molmobot trains on MolmoSpaces trajectories, so it needs --rollouts. "
            "There is no conversion step: point it at the `rollouts/<task>` directory "
            "generate_dataset.py wrote, once per task."
        )
    if trainer != "molmobot" and dataset is None:
        raise click.UsageError(
            f"--trainer {trainer} needs a LeRobot dataset, so pass --dataset. Build one with "
            "`generate_dataset.py`."
        )
    if trainer != "molmobot" and len(rollouts) > 1:
        raise click.UsageError(
            f"--trainer {trainer} trains on one LeRobot dataset. Mixing several tasks in one "
            "run is a MolmoBot feature; export them into a single dataset instead."
        )

    rates = _parse_sample_rates(sample_rates, len(rollouts))
    base_checkpoint = base_checkpoint or (
        DEFAULT_MOLMOBOT_CHECKPOINT if trainer == "molmobot" else "pi05_droid"
    )
    selected_cameras = parse_camera_names(cameras) if cameras else None
    datasets = (
        [
            prepare_molmospaces_dataset(
                rollout_dir=rollout_dir,
                val_fraction=val_fraction,
                link=link,
                camera_names=selected_cameras,
            )
            for rollout_dir in rollouts
        ]
        if rollouts
        else [read_lerobot_dataset(dataset)]  # type: ignore[arg-type]
    )
    output_dir = Path(output_dir) if output_dir else mixture_root(datasets).parent / "checkpoints"

    for summary in datasets:
        _print_summary(summary)
    _warn_about_empty_val_splits(datasets)

    config_path = write_trainer_config(
        datasets=datasets,
        trainer=trainer,
        base_checkpoint=base_checkpoint,
        output_dir=output_dir,
        batch_size=batch_size,
        steps=steps,
        learning_rate=learning_rate,
        action_type=action_type,
        camera_names=selected_cameras,
        sample_rates=rates,
    )
    checkout = None
    if trainer == "molmobot":
        try:
            checkout = ensure_checkout(trainer_repo or DEFAULT_CHECKOUT, clone=clone)
            registered = ensure_stretch_presets(checkout, STRETCH_ACTION_SPEC)
            patched_optimizer = ensure_adamw8bit(checkout)
            patched_metrics = ensure_metrics_log(checkout)
        except MolmoBotSetupError as e:
            raise click.ClickException(str(e)) from e
        _print_checkout(checkout, registered, patched_optimizer, patched_metrics)

    script_path = write_launch_script(
        datasets=datasets,
        trainer=trainer,
        config_path=config_path,
        base_checkpoint=base_checkpoint,
        output_dir=output_dir,
        batch_size=batch_size,
        steps=steps,
        action_type=action_type,
        seq_len=seq_len,
        camera_names=selected_cameras,
        sample_rates=rates,
        device_batch_size=device_batch_size,
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


def _parse_sample_rates(rates: str | None, expected: int) -> list[float] | None:
    """`--sample-rates '0.6,0.4'` -> `[0.6, 0.4]`, checked against the task count."""
    if not rates:
        return None
    try:
        parsed = [float(token) for token in rates.split(",") if token.strip()]
    except ValueError as e:
        raise click.UsageError(f"--sample-rates must be comma-separated numbers: {e}") from e
    if len(parsed) != expected:
        raise click.UsageError(
            f"--sample-rates has {len(parsed)} weights for {expected} --rollouts. MolmoBot "
            "takes one per task, in the order they are passed."
        )
    if any(rate <= 0 for rate in parsed):
        raise click.UsageError("--sample-rates must all be positive; drop the task instead.")
    return parsed


def _warn_about_empty_val_splits(datasets: list[DatasetSummary]) -> None:
    """Say so when a task held out no validation houses, because the trainer will not.

    MolmoBot requires a `val/` directory beside every `train/`, and
    `arrange_train_val_split` does create one -- but a task with a single house
    leaves it empty, and an empty split has no `valid_trajectory_index.json` for
    `SynthmanipDataset` to open. That surfaces as a raise deep in the dataloader
    long after `uv sync` and the checkpoint download have run.
    """
    starved = [d.root.name for d in datasets if d.kind == "molmospaces" and not d.splits.get("val")]
    if not starved:
        return
    click.echo("")
    click.secho(
        f"Warning: {', '.join(starved)} held out no validation houses, so MolmoBot will "
        "raise\nwhen it opens that split. Generate more houses for it, or drop the task "
        "from the run.",
        fg="yellow",
    )


def _print_checkout(
    checkout: MolmoBotCheckout,
    registered: list[str],
    patched_optimizer: bool = False,
    patched_metrics: bool = False,
) -> None:
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
        f"  presets: {', '.join(registered)} (registered in synthmanip_presets.py)"
        if registered
        else f"  presets: {', '.join(sorted(set(STRETCH_PRESET_NAMES.values())))} (already there)"
    )
    click.echo(
        f"  optim:   {ADAMW8BIT} "
        f"({'registered in optim.py' if patched_optimizer else 'already there'})"
    )
    click.echo(
        "  metrics: metrics.jsonl "
        f"({'logging patched into trainer.py' if patched_metrics else 'already there'})"
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
