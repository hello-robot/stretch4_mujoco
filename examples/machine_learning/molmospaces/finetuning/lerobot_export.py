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

Everything is exported in `stretch` space: Stretch's own 10-dimensional
move-group vector, the encoding in `policies/networks.py`. The policy drives the
robot's own joints directly, including its base, so nothing about the recorded
motion is lost in translation and nothing has to be translated back at
evaluation time.

This used to offer a second `franka` space, which re-encoded each recorded tool
pose as the joints of a virtual Franka bolted to Stretch's mast, so that a
DROID-pretrained model could keep its 8-dimensional action head. That is gone.
It bought a warm-started head at the cost of a coordinate frame nobody could
see: the encoding silently dropped every pose the virtual arm could not reach,
it had to be paired with a matching `frame_source` at evaluation or the arm
reached consistently short with nothing in the logs to say so, and the two
robots' workspaces only overlap by about two thirds in the first place. Training
the head from scratch on numbers that mean what they say is the cheaper mistake.

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
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from examples.machine_learning.molmospaces.hdf5_layout import (
    TRAJECTORY_FILE_PATTERN,
    TRAJECTORY_KEY_PATTERN,
    decode_json_blob,
    video_filename,
)
from examples.machine_learning.molmospaces.policies.networks import (
    ACTION_DIM,
    STATE_DIM,
    encode_action,
    encode_state,
)
from examples.machine_learning.molmospaces.stretch.config import HEAD_CAMERA, WRIST_CAMERA

log = logging.getLogger(__name__)

ACTION_SPACES = ("stretch",)

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
dataset. 1.0m is far beyond anything a 15Hz step can command (the simple_ik
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
    def write(self, directory: Path) -> None:
        (directory / METADATA_FILENAME).write_text(json.dumps(asdict(self), indent=2))


class _StretchSpaceEncoder:
    """Encodes state and actions in Stretch's own move-group vector.

    A thin named wrapper over `policies/networks.py`: the encoding itself is
    shared with the behaviour-cloning trainer rather than duplicated, which is
    the only way the two stay in step.
    """

    state_dim = STATE_DIM
    action_dim = ACTION_DIM
    state_names = STRETCH_STATE_NAMES
    action_names = STRETCH_ACTION_NAMES

    def state(self, qpos: dict) -> np.ndarray:
        return encode_state(qpos)

    def action(self, commanded: dict, base_xytheta: np.ndarray) -> np.ndarray:
        return encode_action(commanded, base_xytheta)


def export_lerobot_dataset(
    rollout_dirs: list[Path],
    output_dir: Path,
    action_space: str = "stretch",
    successful_only: bool = True,
    fps: float = 15.0,
    camera_names: tuple[str, ...] = (HEAD_CAMERA, WRIST_CAMERA),
    validate: bool = False,
) -> ExportMetadata:
    """Convert recorded rollouts into a LeRobot-format dataset.

    Args:
        rollout_dirs: directories containing `house_*/trajectories*.h5`. Several
            are pooled, which is how task families are mixed into one dataset.
        output_dir: dataset root. Created; existing contents are left alone
            except for files this writes.
        action_space: only `stretch`. See the module docstring.
        successful_only: keep only trajectories the task judged successful.
        fps: frame rate to record in the metadata. Must match the rate the
            rollouts were recorded at (`policy_dt_ms`), or every timestamp in
            the dataset is wrong.
        camera_names: MolmoSpaces cameras to include, in `CAMERA_FEATURE_NAMES`.
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

    encoder_cls = _StretchSpaceEncoder

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

    for rollout_dir in rollout_dirs:
        for h5_path in sorted(Path(rollout_dir).rglob("trajectories*.h5")):
            match = TRAJECTORY_FILE_PATTERN.match(h5_path.name)
            if match is None:
                continue
            batch_suffix = match.group("suffix")

            with h5py.File(h5_path, "r") as h5_file:
                for traj_key in sorted(h5_file.keys()):
                    key_match = TRAJECTORY_KEY_PATTERN.match(traj_key)
                    if key_match is None:
                        continue
                    trajectory = h5_file[traj_key]

                    if successful_only and not bool(np.any(trajectory["success"][:])):
                        metadata.skipped_unsuccessful += 1
                        continue

                    episode_index = metadata.num_episodes
                    episode = _build_episode(
                        trajectory=trajectory,
                        encoder=encoder_cls(),
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
    successes = trajectory["success"][:]
    num_steps = len(qpos_rows)

    states, actions, valid = [], [], []
    replaced_base_commands = 0
    for step in range(num_steps):
        qpos = decode_json_blob(qpos_rows[step])
        commanded = decode_json_blob(action_rows[step])
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
        base_xytheta = np.array(
            [base_pose[0, 3], base_pose[1, 3], np.arctan2(base_pose[1, 0], base_pose[0, 0])]
        )

        states.append(encoder.state(qpos))
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
        source = episode_dir / video_filename(source_episode_index, camera, batch_suffix)
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
    del action_space  # only `stretch` remains; kept so the metadata writer reads in order
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
    is noise from the sampler rather than a height the robot was at.
    """
    pose = _pose_matrix(pose7)
    pose[2, 3] = 0.0
    return pose
