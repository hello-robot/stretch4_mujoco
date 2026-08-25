"""
Recorded Stretch rollouts -> a LeRobot dataset a VLA trainer can read.

MolmoSpaces writes a rollout as one HDF5 file per house plus side-car MP4s:

    <run>/house_101/trajectories_batch_1_of_1.h5
    <run>/house_101/episode_00000000_head_camera_batch_1_of_1.mp4
    <run>/house_101/episode_00000000_wrist_camera_batch_1_of_1.mp4

with per-step state and commanded actions inside the HDF5 as JSON blobs
(`obs/agent/qpos`, `actions/joint_pos`), the language instruction in
`obs_scene["task_description"]`, and the images only in the videos. Nothing
consumes that layout but MolmoSpaces. VLA trainers -- openpi, LeRobot's own,
anything built on `LeRobotDataset` -- consume the LeRobot format, so this
converts.

The interesting decision is what the *action space* should be, and there are two
defensible answers. Both are implemented, `--action-space` picks:

`franka` (default)
    The same 8-dimensional Franka joint space the pretrained model already
    speaks: seven arm joints plus a gripper scalar. Stretch's recorded tool
    poses are run backwards through `franka_remapping/` -- tool pose to virtual
    Franka joints -- so the numbers are Franka numbers describing Stretch
    motions. Fine-tuning on this needs no surgery on the model's action head,
    the pretrained weights start from something meaningful, and at evaluation
    the *same* remapper turns the outputs back into Stretch commands. The
    forward and reverse maps being the same code is the point: whatever the
    retarget cannot express, the fine-tuning data does not contain either, so
    the model is never trained to ask for something the robot cannot do.

`stretch`
    Stretch's own 10-dimensional move-group vector, the encoding in
    `policies/networks.py`. Nothing is lost to a retarget and the policy drives
    the robot directly, including its base. But the action head has to be
    reshaped and re-learned, so it wants far more data -- this is the option for
    training a Stretch policy, not for fine-tuning a Franka one.

The output targets **LeRobot dataset format v2.1**: parquet per episode under
`data/`, MP4 per camera per episode under `videos/`, and JSON metadata under
`meta/`. It is written directly with pyarrow rather than through
`lerobot.LeRobotDataset`, because `lerobot` is not a dependency of this repo --
which means the layout is *targeted*, not validated by the library that defines
it. `--validate` checks it against an installed `lerobot` if you have one, and
`meta/stretch_export.json` records the same shape information in a form that
does not depend on anyone's format version.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from examples.machine_learning.molmospaces.franka_remapping.episode_frame import (
    default_frame_for,
)
from examples.machine_learning.molmospaces.franka_remapping.franka_arm import FrankaArm
from examples.machine_learning.molmospaces.franka_remapping.pose_solver import StretchPoseSolver
from examples.machine_learning.molmospaces.policies.networks import (
    ACTION_DIM,
    STATE_DIM,
    encode_action,
    encode_state,
)
from examples.machine_learning.molmospaces.stretch.config import (
    HEAD_CAMERA,
    WRIST_CAMERA,
    Stretch4RobotConfig,
)
from examples.machine_learning.molmospaces.stretch.robot_view import StretchGripperGroup

log = logging.getLogger(__name__)

ACTION_SPACES = ("franka", "stretch")

CODEBASE_VERSION = "v2.1"
"""The LeRobot dataset format version the layout below targets."""

CHUNK_SIZE = 1000
"""Episodes per `chunk-*` directory. LeRobot's own default."""

DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"

CAMERA_FEATURE_NAMES = {
    HEAD_CAMERA: "observation.images.head",
    WRIST_CAMERA: "observation.images.wrist",
}
"""MolmoSpaces camera name -> LeRobot feature key."""

FRANKA_STATE_NAMES = [f"franka_joint_{i + 1}" for i in range(7)] + ["stretch_gripper"]

GRIPPER_CHANNEL_NAMES = ["stretch_gripper", "stretch_gripper_mirror"]
"""
Names for the two gripper channels of the `stretch` encoding.

Stretch 4 has **one** commanded gripper degree of freedom. On the robot stack it
is `EndOfArmSubsystem.stretch_gripper` -- a single `JointSubsystem` over
`Actuators.gripper`, which `parallel_gripper` is only an alias for -- driven with
one `move_to` in aperture radians. The MJCF splits it into two mirrored finger
joints so the fingers can be simulated independently under contact, and
`StretchGripperGroup` and `policies/networks.py` both carry the pair, which is
why the encoding is two numbers wide rather than one.

So the second channel is named for what it is: the mirror of the first, not a
second actuator. In an *action* the two are always identical -- every policy here
commands `[target, target]` -- and in a *state* they can differ by a few
hundredths of a radian when one finger is loaded harder than the other. A model
trained on this should be read as controlling one gripper.
"""

STRETCH_STATE_NAMES = [
    "lift",
    "arm",
    "wrist_yaw",
    "wrist_pitch",
    "wrist_roll",
] + GRIPPER_CHANNEL_NAMES
STRETCH_ACTION_NAMES = [
    "base_forward",
    "base_left",
    "base_yaw",
    "lift",
    "arm",
    "wrist_yaw",
    "wrist_pitch",
    "wrist_roll",
] + GRIPPER_CHANNEL_NAMES

_TRAJECTORY_FILE_PATTERN = re.compile(r"^trajectories(?P<suffix>.*)\.h5$")
_TRAJECTORY_KEY_PATTERN = re.compile(r"^traj_(?P<index>\d+)$")

METADATA_FILENAME = "stretch_export.json"

IMPLAUSIBLE_BASE_COMMAND_M = 1.0
"""
How far a recorded base command may sit from the observed base before it is
treated as unusable and replaced with "hold position".

There is exactly one such command per episode and it is always step 0, where the
recorded action is a zeroed no-op rather than anything the policy chose --
`HoloJointsRobotBaseGroup`'s no-op control is a zero vector, which as an
*absolute* planar position means the world origin rather than "stay put". Left
alone it encodes a 25-metre tool displacement into the first frame of every
episode, which is enough to move the mean of any statistic computed over the
dataset. 1.0m is far beyond anything a 15Hz step can command (the scripted
expert's own per-solve cap is 0.35m), so the test cannot fire on a real command.
"""


@dataclass
class ExportMetadata:
    """What an export produced. Written to `meta/stretch_export.json`."""

    action_space: str
    state_dim: int
    action_dim: int
    num_episodes: int = 0
    num_frames: int = 0
    fps: float = 15.0
    camera_names: list[str] = field(default_factory=list)
    video_keys: list[str] = field(default_factory=list)
    tasks: list[str] = field(default_factory=list)
    skipped_unsuccessful: int = 0
    skipped_missing_video: int = 0
    replaced_base_commands: int = 0
    """Frames whose recorded base command was unusable; see `IMPLAUSIBLE_BASE_COMMAND_M`."""
    mean_shadow_ik_error_m: float = 0.0
    """
    Average tool-position error of the virtual Franka's IK, over every exported frame.

    Only meaningful for `--action-space franka`, and worth reading before
    training on the result: it is how much of Stretch's actual motion the
    Franka-space encoding failed to represent. Millimetres are fine.
    Centimetres mean the demonstrations spent significant time in poses the
    virtual arm cannot reach, and the exported actions are not the motions that
    were recorded.
    """

    def write(self, directory: Path) -> None:
        (directory / METADATA_FILENAME).write_text(json.dumps(asdict(self), indent=2))


class _FrankaSpaceEncoder:
    """Encodes Stretch state and commanded actions as virtual Franka joints.

    One instance per episode, because the virtual arm's frame is pinned to where
    the episode *started*. That is deliberate and it has to match the evaluation
    side: `action_remap.FrankaActionRemapper` also pins its frame once per
    episode, so a base motion appears as a tool motion in the Franka frame in
    both directions. Re-deriving the frame per step instead would bolt the
    virtual arm to the moving base, and the model would be trained on a
    coordinate system it is not evaluated in.
    """

    state_dim = 8
    action_dim = 8
    state_names = FRANKA_STATE_NAMES
    action_names = FRANKA_STATE_NAMES

    def __init__(self, franka: FrankaArm, solver: StretchPoseSolver, base_pose: np.ndarray) -> None:
        self._franka = franka
        self._solver = solver
        self._frame = default_frame_for(base_pose)
        self._state_seed = franka.clip(self._frame.init_qpos)
        self._action_seed = self._state_seed.copy()
        self.ik_errors: list[float] = []

    def state(self, qpos: dict, tool_pose_world: np.ndarray) -> np.ndarray:
        solution = self._franka.inverse(
            self._frame.tool_pose_from_world(tool_pose_world), seed=self._state_seed
        )
        self._state_seed = solution.qpos
        self.ik_errors.append(solution.position_error)
        return np.append(solution.qpos, _stretch_gripper_closedness(qpos["gripper"])).astype(
            np.float32
        )

    def action(self, commanded: dict, base_xytheta: np.ndarray) -> np.ndarray:
        del base_xytheta  # the commanded base pose is already absolute and world-framed
        tool_pose_world = self._solver.forward(
            {
                "base": np.asarray(commanded["base"], dtype=float).reshape(-1)[:3],
                "lift": np.asarray(commanded["lift"], dtype=float).reshape(-1)[:1],
                "arm": np.asarray(commanded["arm"], dtype=float).reshape(-1)[:1],
                "wrist": np.asarray(commanded["wrist"], dtype=float).reshape(-1)[:3],
            }
        )
        solution = self._franka.inverse(
            self._frame.tool_pose_from_world(tool_pose_world), seed=self._action_seed
        )
        self._action_seed = solution.qpos
        self.ik_errors.append(solution.position_error)
        return np.append(solution.qpos, _stretch_gripper_closedness(commanded["gripper"])).astype(
            np.float32
        )


class _StretchSpaceEncoder:
    """Encodes state and actions in Stretch's own move-group vector.

    A thin adapter over `policies/networks.py` so both spaces present the same
    interface here; the encoding itself is shared with the behaviour-cloning
    trainer rather than duplicated, which is the only way the two stay in step.
    """

    state_dim = STATE_DIM
    action_dim = ACTION_DIM
    state_names = STRETCH_STATE_NAMES
    action_names = STRETCH_ACTION_NAMES

    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs
        self.ik_errors: list[float] = []

    def state(self, qpos: dict, tool_pose_world: np.ndarray) -> np.ndarray:
        del tool_pose_world
        return encode_state(qpos)

    def action(self, commanded: dict, base_xytheta: np.ndarray) -> np.ndarray:
        return encode_action(commanded, base_xytheta)


def export_lerobot_dataset(
    rollout_dirs: list[Path],
    output_dir: Path,
    action_space: str = "franka",
    successful_only: bool = True,
    fps: float = 15.0,
    camera_names: tuple[str, ...] = (HEAD_CAMERA, WRIST_CAMERA),
    robot_config=None,
    validate: bool = False,
) -> ExportMetadata:
    """Convert recorded rollouts into a LeRobot-format dataset.

    Args:
        rollout_dirs: directories containing `house_*/trajectories*.h5`. Several
            are pooled, which is how task families are mixed into one dataset.
        output_dir: dataset root. Created; existing contents are left alone
            except for files this writes.
        action_space: `franka` or `stretch`. See the module docstring.
        successful_only: keep only trajectories the task judged successful.
        fps: frame rate to record in the metadata. Must match the rate the
            rollouts were recorded at (`policy_dt_ms`), or every timestamp in
            the dataset is wrong.
        camera_names: MolmoSpaces cameras to include, in `CAMERA_FEATURE_NAMES`.
        robot_config: robot config for the Stretch forward kinematics. Defaults
            to `Stretch4RobotConfig()`; only used by the `franka` space.
        validate: after writing, try to open the result with an installed
            `lerobot` and report what it says.

    Returns:
        `ExportMetadata`, also written to `meta/stretch_export.json`.
    """
    import h5py

    if action_space not in ACTION_SPACES:
        raise ValueError(f"action_space must be one of {ACTION_SPACES}, got {action_space!r}")

    output_dir = Path(output_dir)
    (output_dir / "meta").mkdir(parents=True, exist_ok=True)

    franka = FrankaArm() if action_space == "franka" else None
    solver = (
        StretchPoseSolver(robot_config or Stretch4RobotConfig())
        if action_space == "franka"
        else None
    )
    encoder_cls = _FrankaSpaceEncoder if action_space == "franka" else _StretchSpaceEncoder

    metadata = ExportMetadata(
        action_space=action_space,
        state_dim=encoder_cls.state_dim,
        action_dim=encoder_cls.action_dim,
        fps=fps,
        camera_names=list(camera_names),
        video_keys=[CAMERA_FEATURE_NAMES[name] for name in camera_names],
    )
    task_indices: dict[str, int] = {}
    episode_records: list[dict] = []
    episode_stats: list[dict] = []
    all_ik_errors: list[float] = []

    for rollout_dir in rollout_dirs:
        for h5_path in sorted(Path(rollout_dir).rglob("trajectories*.h5")):
            match = _TRAJECTORY_FILE_PATTERN.match(h5_path.name)
            if match is None:
                continue
            batch_suffix = match.group("suffix")

            with h5py.File(h5_path, "r") as h5_file:
                for traj_key in sorted(h5_file.keys()):
                    key_match = _TRAJECTORY_KEY_PATTERN.match(traj_key)
                    if key_match is None:
                        continue
                    trajectory = h5_file[traj_key]

                    if successful_only and not bool(np.any(trajectory["success"][:])):
                        metadata.skipped_unsuccessful += 1
                        continue

                    episode_index = metadata.num_episodes
                    episode = _build_episode(
                        trajectory=trajectory,
                        encoder=encoder_cls(
                            franka,
                            solver,
                            _base_pose_matrix(trajectory["obs/extra/robot_base_pose"][0]),
                        ),
                        fps=fps,
                    )
                    if episode is None:
                        continue

                    videos = _copy_videos(
                        episode_dir=h5_path.parent,
                        source_episode_index=int(key_match.group("index")),
                        batch_suffix=batch_suffix,
                        camera_names=camera_names,
                        output_dir=output_dir,
                        episode_index=episode_index,
                    )
                    if videos is None:
                        metadata.skipped_missing_video += 1
                        continue

                    task = episode["task"]
                    task_index = task_indices.setdefault(task, len(task_indices))
                    _write_episode_parquet(
                        output_dir=output_dir,
                        episode_index=episode_index,
                        episode=episode,
                        task_index=task_index,
                        global_index_start=metadata.num_frames,
                    )

                    episode_records.append(
                        {
                            "episode_index": episode_index,
                            "tasks": [task],
                            "length": episode["length"],
                        }
                    )
                    episode_stats.append(
                        {
                            "episode_index": episode_index,
                            "stats": _episode_stats(episode),
                        }
                    )
                    metadata.num_episodes += 1
                    metadata.num_frames += episode["length"]
                    metadata.replaced_base_commands += episode["replaced_base_commands"]
                    all_ik_errors.extend(episode["ik_errors"])
                    log.info(
                        f"[export] {h5_path.parent.name}/{traj_key} -> "
                        f"episode_{episode_index:06d} ({episode['length']} frames)"
                    )

    if metadata.num_episodes == 0:
        raise RuntimeError(
            f"No episodes exported from {[str(d) for d in rollout_dirs]}. "
            f"Skipped {metadata.skipped_unsuccessful} unsuccessful and "
            f"{metadata.skipped_missing_video} with missing videos; pass "
            "successful_only=False to keep failures."
        )

    metadata.tasks = sorted(task_indices, key=task_indices.get)
    metadata.mean_shadow_ik_error_m = float(np.mean(all_ik_errors)) if all_ik_errors else 0.0
    _write_metadata(output_dir, metadata, task_indices, episode_records, episode_stats)
    metadata.write(output_dir / "meta")

    log.info(
        f"[export] {metadata.num_episodes} episodes / {metadata.num_frames} frames "
        f"in {action_space} space -> {output_dir}"
    )
    if metadata.replaced_base_commands:
        log.info(
            f"[export] replaced {metadata.replaced_base_commands} unusable base commands with "
            f"hold-position; see IMPLAUSIBLE_BASE_COMMAND_M."
        )
    if action_space == "franka":
        log.info(
            f"[export] virtual Franka IK error {metadata.mean_shadow_ik_error_m * 1000:.1f}mm "
            "on average; see ExportMetadata.mean_shadow_ik_error_m."
        )
    if validate:
        _validate_with_lerobot(output_dir)
    return metadata


# =============================================================================
# Per-episode conversion
# =============================================================================


def _build_episode(trajectory, encoder, fps: float) -> dict | None:
    """Pack one trajectory's states, actions and flags into arrays.

    Returns None for a trajectory with no usable step. The returned dict carries
    a `replaced_base_commands` count; see `IMPLAUSIBLE_BASE_COMMAND_M`.
    """
    qpos_rows = trajectory["obs/agent/qpos"][:]
    action_rows = trajectory["actions/joint_pos"][:]
    base_poses = trajectory["obs/extra/robot_base_pose"][:]
    tcp_poses = trajectory["obs/extra/tcp_pose"][:]
    successes = trajectory["success"][:]
    num_steps = len(qpos_rows)

    states, actions, valid = [], [], []
    replaced_base_commands = 0
    for step in range(num_steps):
        qpos = _decode_json_blob(qpos_rows[step])
        commanded = _decode_json_blob(action_rows[step])
        if not commanded:
            # A no-op step: the policy returned {} because its plan was
            # exhausted. There is no action to imitate here.
            continue

        # Fill any move group the policy left unspecified with "hold position",
        # which is exactly what the controllers do with an absent group.
        merged = {group: np.asarray(value, dtype=float) for group, value in qpos.items()}
        merged.update({group: np.asarray(value, dtype=float) for group, value in commanded.items()})

        observed_base = np.asarray(qpos["base"], dtype=float).reshape(-1)[:3]
        if np.linalg.norm(merged["base"][:2] - observed_base[:2]) > IMPLAUSIBLE_BASE_COMMAND_M:
            merged["base"] = observed_base.copy()
            replaced_base_commands += 1

        base_pose = _base_pose_matrix(base_poses[step])
        # `tcp_pose` is recorded in the robot's *base* frame (verified against
        # the compiled model: at lift 0.6 / arm 0.1 / wrist 0 it reads
        # (0.567, -0.087, 0.835), which is the tool's offset from the base
        # origin, not a world position).
        tool_pose_world = base_pose @ _pose_matrix(tcp_poses[step])
        base_xytheta = np.array(
            [base_pose[0, 3], base_pose[1, 3], np.arctan2(base_pose[1, 0], base_pose[0, 0])]
        )

        states.append(encoder.state(qpos, tool_pose_world))
        actions.append(encoder.action(merged, base_xytheta))
        valid.append(step)

    if not valid:
        return None

    length = len(valid)
    scene = _decode_scene(trajectory)
    return {
        "length": length,
        "task": scene.get("task_description") or "complete the task",
        "state": np.stack(states).astype(np.float32),
        "action": np.stack(actions).astype(np.float32),
        "timestamp": (np.arange(length, dtype=np.float32) / float(fps)),
        "success": np.asarray([bool(successes[step]) for step in valid], dtype=bool),
        "source_steps": np.asarray(valid, dtype=np.int64),
        "ik_errors": list(encoder.ik_errors),
        "replaced_base_commands": replaced_base_commands,
    }


def _copy_videos(
    episode_dir: Path,
    source_episode_index: int,
    batch_suffix: str,
    camera_names: tuple[str, ...],
    output_dir: Path,
    episode_index: int,
) -> dict[str, Path] | None:
    """Copy this episode's MP4s into the dataset, or None if any is missing.

    Copied rather than re-encoded. MolmoSpaces already wrote them at the
    rollout's own frame rate, and a re-encode would cost an hour of CPU per
    thousand episodes to make the pixels slightly worse.

    Dropping an episode whose video is missing is deliberate: a state/action pair
    with no image is not a training sample for a vision-language-action model,
    and silently exporting one would show up much later as a model that ignores
    its cameras.
    """
    copied: dict[str, Path] = {}
    for camera in camera_names:
        source = episode_dir / f"episode_{source_episode_index:08d}_{camera}{batch_suffix}.mp4"
        if not source.exists():
            log.warning(f"[export] missing video {source}; skipping episode")
            return None
        destination = output_dir / VIDEO_PATH.format(
            episode_chunk=episode_index // CHUNK_SIZE,
            video_key=CAMERA_FEATURE_NAMES[camera],
            episode_index=episode_index,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        copied[camera] = destination
    return copied


def _write_episode_parquet(
    output_dir: Path,
    episode_index: int,
    episode: dict,
    task_index: int,
    global_index_start: int,
) -> None:
    """Write one episode's per-frame table."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    length = episode["length"]
    table = pa.table(
        {
            "observation.state": pa.array(
                [row.tolist() for row in episode["state"]],
                type=pa.list_(pa.float32(), episode["state"].shape[1]),
            ),
            "action": pa.array(
                [row.tolist() for row in episode["action"]],
                type=pa.list_(pa.float32(), episode["action"].shape[1]),
            ),
            "timestamp": pa.array(episode["timestamp"], type=pa.float32()),
            "frame_index": pa.array(np.arange(length, dtype=np.int64)),
            "episode_index": pa.array(np.full(length, episode_index, dtype=np.int64)),
            "index": pa.array(
                np.arange(global_index_start, global_index_start + length, dtype=np.int64)
            ),
            "task_index": pa.array(np.full(length, task_index, dtype=np.int64)),
            "next.success": pa.array(episode["success"], type=pa.bool_()),
            "next.done": pa.array(np.arange(length) == length - 1, type=pa.bool_()),
        }
    )
    path = output_dir / DATA_PATH.format(
        episode_chunk=episode_index // CHUNK_SIZE, episode_index=episode_index
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def _episode_stats(episode: dict) -> dict:
    """Per-feature min/max/mean/std for one episode, in LeRobot's shape."""
    stats = {}
    for key in ("observation.state", "action"):
        values = episode["state" if key == "observation.state" else "action"]
        stats[key] = {
            "min": values.min(axis=0).tolist(),
            "max": values.max(axis=0).tolist(),
            "mean": values.mean(axis=0).tolist(),
            "std": values.std(axis=0).tolist(),
            "count": [int(len(values))],
        }
    return stats


# =============================================================================
# Dataset-level metadata
# =============================================================================


def _write_metadata(
    output_dir: Path,
    metadata: ExportMetadata,
    task_indices: dict[str, int],
    episode_records: list[dict],
    episode_stats: list[dict],
) -> None:
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    features: dict[str, dict] = {
        "observation.state": {
            "dtype": "float32",
            "shape": [metadata.state_dim],
            "names": _encoder_names(metadata.action_space)[0],
        },
        "action": {
            "dtype": "float32",
            "shape": [metadata.action_dim],
            "names": _encoder_names(metadata.action_space)[1],
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
        "next.success": {"dtype": "bool", "shape": [1], "names": None},
        "next.done": {"dtype": "bool", "shape": [1], "names": None},
    }
    for video_key in metadata.video_keys:
        features[video_key] = {
            "dtype": "video",
            "shape": [None, None, 3],
            "names": ["height", "width", "channel"],
            "video_info": {
                "video.fps": metadata.fps,
                "video.codec": "h264",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }

    chunks = max(1, -(-metadata.num_episodes // CHUNK_SIZE))
    info = {
        "codebase_version": CODEBASE_VERSION,
        "robot_type": "stretch4",
        "total_episodes": metadata.num_episodes,
        "total_frames": metadata.num_frames,
        "total_tasks": len(task_indices),
        "total_videos": metadata.num_episodes * len(metadata.video_keys),
        "total_chunks": chunks,
        "chunks_size": CHUNK_SIZE,
        "fps": metadata.fps,
        "splits": {"train": f"0:{metadata.num_episodes}"},
        "data_path": DATA_PATH,
        "video_path": VIDEO_PATH,
        "features": features,
    }
    (meta_dir / "info.json").write_text(json.dumps(info, indent=2))
    _write_jsonl(
        meta_dir / "tasks.jsonl",
        [
            {"task_index": index, "task": task}
            for task, index in sorted(task_indices.items(), key=lambda item: item[1])
        ],
    )
    _write_jsonl(meta_dir / "episodes.jsonl", episode_records)
    _write_jsonl(meta_dir / "episodes_stats.jsonl", episode_stats)


def _encoder_names(action_space: str) -> tuple[list[str], list[str]]:
    if action_space == "franka":
        return FRANKA_STATE_NAMES, FRANKA_STATE_NAMES
    return STRETCH_STATE_NAMES, STRETCH_ACTION_NAMES


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _validate_with_lerobot(output_dir: Path) -> None:
    """Open the written dataset with an installed `lerobot`, if there is one."""
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        log.warning(
            "[export] --validate asked for, but `lerobot` is not installed, so the "
            "layout could not be checked against the library that defines it."
        )
        return
    dataset = LeRobotDataset(repo_id="stretch4/local", root=output_dir)
    log.info(
        f"[export] lerobot opened the dataset: {dataset.num_episodes} episodes, "
        f"{dataset.num_frames} frames, features {sorted(dataset.features)}"
    )


# =============================================================================
# HDF5 helpers
# =============================================================================


def _decode_json_blob(row: np.ndarray) -> dict:
    """MolmoSpaces stores dict observations as NUL-padded UTF-8 in a uint8 row."""
    return json.loads(bytes(row).rstrip(b"\x00").decode("utf-8"))


def _decode_scene(trajectory) -> dict:
    """`obs_scene`, which carries the episode's language instruction.

    Stored as a single JSON string per trajectory rather than per step. Returns
    an empty dict rather than raising if it is absent or unparseable: a missing
    instruction costs the episode its prompt, not its trajectory.
    """
    if "obs_scene" not in trajectory:
        return {}
    raw = trajectory["obs_scene"][()]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        log.warning("[export] could not parse obs_scene; episode will have no instruction")
        return {}


def _pose_matrix(pose7: np.ndarray) -> np.ndarray:
    """`[x, y, z, qw, qx, qy, qz]` -> a 4x4 pose."""
    from scipy.spatial.transform import Rotation as R

    pose7 = np.asarray(pose7, dtype=float).reshape(-1)
    pose = np.eye(4)
    pose[:3, 3] = pose7[:3]
    pose[:3, :3] = R.from_quat(pose7[3:7], scalar_first=True).as_matrix()
    return pose


def _base_pose_matrix(pose7: np.ndarray) -> np.ndarray:
    """The base's recorded pose, flattened onto the floor.

    The recorded z is dropped because Stretch's base *is* on the floor: the
    holonomic base group only ever reads x, y and yaw, so a nonzero recorded z
    is noise from the sampler rather than a height the robot was at, and
    carrying it into the virtual Franka's mount would shift every tool pose in
    the episode by it.
    """
    pose = _pose_matrix(pose7)
    pose[2, 3] = 0.0
    return pose


def _stretch_gripper_closedness(gripper_qpos) -> float:
    """Stretch's gripper opening -> the [0, 1] closedness a VLA speaks.

    Averages the mirrored MJCF finger pair; Stretch has one gripper DOF. See
    `GRIPPER_CHANNEL_NAMES`.
    """
    opening = float(np.mean(np.asarray(gripper_qpos, dtype=float).reshape(-1)[:2]))
    return float(np.clip(1.0 - opening / StretchGripperGroup.OPEN_JOINT_POS, 0.0, 1.0))
