"""
Turn recorded MolmoSpaces trajectories into behaviour-cloning training data.

The input is a MolmoSpaces rollout directory -- whatever
`finetuning/generate_dataset.py` produced -- whose format is described by
`../hdf5_layout.py`, which this module borrows its patterns and decoders from.
Per-step robot state and commanded actions live in the HDF5 as JSON blobs
(`obs/agent/qpos`, `actions/joint_pos`); the images live only in the side-car
MP4s.

Training directly off that layout would mean re-decoding an MP4 for every
minibatch, so `build_dataset()` does the decoding once: it keeps only the
successful trajectories, resizes frames to `IMAGE_SIZE`, encodes state and
actions with the shared scheme in `policies/networks.py`, and writes one
compressed `.npz` shard per trajectory. `StretchBCDataset` then reads those.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
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
    IMAGE_SIZE,
    STATE_DIM,
    encode_action,
    encode_state,
)

log = logging.getLogger(__name__)

METADATA_FILENAME = "dataset_meta.json"


@dataclass
class DatasetMetadata:
    """What a built dataset directory contains."""

    camera_names: list[str]
    num_trajectories: int
    num_transitions: int
    image_size: int = IMAGE_SIZE
    state_dim: int = STATE_DIM
    action_dim: int = ACTION_DIM

    def write(self, directory: Path) -> None:
        (directory / METADATA_FILENAME).write_text(json.dumps(self.__dict__, indent=2))

    @classmethod
    def read(cls, directory: Path) -> "DatasetMetadata":
        return cls(**json.loads((directory / METADATA_FILENAME).read_text()))


def _read_video(path: Path, num_frames: int) -> np.ndarray:
    """Decode an episode video to `(num_frames, IMAGE_SIZE, IMAGE_SIZE, 3)` uint8.

    Videos are re-encoded by the saver and can come back a frame or two short of
    the HDF5's step count; the last frame is repeated rather than dropping the
    tail of an otherwise good trajectory.
    """
    import cv2

    capture = cv2.VideoCapture(str(path))
    frames = []
    while len(frames) < num_frames:
        ok, frame = capture.read()
        if not ok:
            break
        frame = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()

    if not frames:
        raise ValueError(f"No frames decoded from {path}")
    while len(frames) < num_frames:
        frames.append(frames[-1])
    return np.stack(frames[:num_frames]).astype(np.uint8)


def build_dataset(
    run_dirs: list[Path],
    output_dir: Path,
    camera_names: list[str],
    successful_only: bool = True,
) -> DatasetMetadata:
    """Convert one or more evaluation-output directories into training shards.

    Args:
        run_dirs: directories produced by `run_evaluation` (the ones containing
            `house_*/`). Several can be merged into one dataset, which is how
            per-benchmark expert runs are pooled.
        output_dir: where to write the shards and `dataset_meta.json`.
        camera_names: cameras to include, in the order the policy will feed them
            to the network. Order is part of the trained model's interface.
        successful_only: keep only trajectories the task judged successful. This
            is the whole point of behaviour cloning off a partial expert -- the
            25%-ish of episodes the scripted policy actually completes are the
            demonstrations; the rest are counter-examples.

    Returns:
        Metadata describing what was written.
    """
    import h5py

    output_dir.mkdir(parents=True, exist_ok=True)
    shard_index = 0
    total_transitions = 0

    for run_dir in run_dirs:
        for h5_path in sorted(Path(run_dir).rglob("trajectories*.h5")):
            match = TRAJECTORY_FILE_PATTERN.match(h5_path.name)
            if match is None:
                continue
            batch_suffix = match.group("suffix")

            with h5py.File(h5_path, "r") as h5_file:
                for traj_key in sorted(h5_file.keys()):
                    key_match = TRAJECTORY_KEY_PATTERN.match(traj_key)
                    if key_match is None:
                        continue
                    episode_index = int(key_match.group("index"))
                    trajectory = h5_file[traj_key]

                    if successful_only and not bool(np.any(trajectory["success"][:])):
                        continue

                    shard = _build_shard(
                        trajectory, h5_path.parent, episode_index, batch_suffix, camera_names
                    )
                    if shard is None:
                        continue

                    np.savez_compressed(output_dir / f"shard_{shard_index:05d}.npz", **shard)
                    shard_index += 1
                    total_transitions += len(shard["states"])
                    log.info(
                        f"[dataset] {h5_path.parent.name}/{traj_key}: "
                        f"{len(shard['states'])} transitions"
                    )

    metadata = DatasetMetadata(
        camera_names=list(camera_names),
        num_trajectories=shard_index,
        num_transitions=total_transitions,
    )
    metadata.write(output_dir)
    log.info(
        f"[dataset] wrote {metadata.num_trajectories} trajectories "
        f"({metadata.num_transitions} transitions) to {output_dir}"
    )
    return metadata


def _build_shard(
    trajectory,
    episode_dir: Path,
    episode_index: int,
    batch_suffix: str,
    camera_names: list[str],
) -> dict[str, np.ndarray] | None:
    """Pack one trajectory into arrays, or None if its videos are missing."""
    qpos_rows = trajectory["obs/agent/qpos"][:]
    action_rows = trajectory["actions/joint_pos"][:]
    base_poses = trajectory["obs/extra/robot_base_pose"][:]
    num_steps = len(qpos_rows)

    videos = []
    for camera in camera_names:
        video_path = episode_dir / video_filename(episode_index, camera, batch_suffix)
        if not video_path.exists():
            log.warning(f"[dataset] missing video {video_path}; skipping trajectory")
            return None
        videos.append(_read_video(video_path, num_steps))

    states = np.zeros((num_steps, STATE_DIM), dtype=np.float32)
    actions = np.zeros((num_steps, ACTION_DIM), dtype=np.float32)
    valid = np.zeros(num_steps, dtype=bool)

    for step in range(num_steps):
        qpos = decode_json_blob(qpos_rows[step])
        commanded = decode_json_blob(action_rows[step])
        if not commanded:
            # A no-op step: the policy returned {} (its plan was exhausted, or it
            # had nothing to say). There is no action to imitate here.
            continue
        # Fill any move group the policy left unspecified with "hold position",
        # which is exactly what the controllers do with an absent group.
        merged = {group: np.asarray(qpos[group], dtype=np.float32) for group in qpos}
        merged.update(
            {group: np.asarray(value, dtype=np.float32) for group, value in commanded.items()}
        )

        base_pose = base_poses[step]
        base_xytheta = np.array(
            [base_pose[0], base_pose[1], _yaw_from_quaternion(base_pose[3:7])], dtype=np.float32
        )
        states[step] = encode_state(qpos)
        actions[step] = encode_action(merged, base_xytheta)
        valid[step] = True

    if not valid.any():
        return None

    return {
        "images": np.stack(videos, axis=1)[valid],  # (T, num_cameras, H, W, 3)
        "states": states[valid],
        "actions": actions[valid],
    }


def _yaw_from_quaternion(quaternion: np.ndarray) -> float:
    qw, qx, qy, qz = (float(component) for component in quaternion)
    return float(np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))


class StretchBCDataset:
    """Chunked (images, state) -> (action chunk) samples from a built dataset.

    Implements the `torch.utils.data.Dataset` protocol without importing torch,
    so the dataset can be inspected without a GPU stack present.
    """

    def __init__(self, directory: Path, chunk_size: int = 8) -> None:
        self.directory = Path(directory)
        self.metadata = DatasetMetadata.read(self.directory)
        self.chunk_size = chunk_size

        self._shard_paths = sorted(self.directory.glob("shard_*.npz"))
        if not self._shard_paths:
            raise FileNotFoundError(f"No shards under {self.directory}")

        # (shard, start step) for every window. Windows are allowed to run off
        # the end of a trajectory and are padded by repeating the final action,
        # so the last few steps -- which for a successful episode are the ones
        # that actually complete the task -- are not dropped from training.
        self._index: list[tuple[int, int]] = []
        self._lengths: list[int] = []
        for shard_id, path in enumerate(self._shard_paths):
            with np.load(path) as shard:
                length = len(shard["states"])
            self._lengths.append(length)
            self._index.extend((shard_id, step) for step in range(length))

        self._cache_id: int | None = None
        self._cache: dict[str, np.ndarray] | None = None

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int):
        shard_id, step = self._index[index]
        shard = self._load(shard_id)

        images = shard["images"][step].astype(np.float32) / 255.0
        images = np.transpose(images, (0, 3, 1, 2))  # (cameras, 3, H, W)

        actions = shard["actions"]
        end = min(step + self.chunk_size, len(actions))
        chunk = actions[step:end]
        if len(chunk) < self.chunk_size:
            chunk = np.concatenate(
                [chunk, np.repeat(chunk[-1:], self.chunk_size - len(chunk), axis=0)]
            )

        return images, shard["states"][step], chunk.astype(np.float32)

    def statistics(self) -> dict[str, np.ndarray]:
        """Per-dimension mean and standard deviation of states and actions.

        Standard deviations are floored: several action dimensions are constant
        in a scripted demonstration (the wrist never rolls, for instance), and
        dividing by their true zero deviation would produce NaNs that only show
        up much later as a policy that outputs garbage.
        """
        states, actions = [], []
        for path in self._shard_paths:
            with np.load(path) as shard:
                states.append(shard["states"])
                actions.append(shard["actions"])
        states = np.concatenate(states)
        actions = np.concatenate(actions)
        return {
            "state_mean": states.mean(axis=0),
            "state_std": np.maximum(states.std(axis=0), 1e-3),
            "action_mean": actions.mean(axis=0),
            "action_std": np.maximum(actions.std(axis=0), 1e-3),
        }

    def _load(self, shard_id: int) -> dict[str, np.ndarray]:
        """Keep one decompressed shard resident.

        Samples are drawn in shuffled order, so this is not a hit-rate
        optimisation -- it exists so that a single `__getitem__` does not hold a
        whole trajectory's images in a temporary. With a DataLoader the workers
        each keep their own, which is why shards are per-trajectory and small.
        """
        if self._cache_id != shard_id:
            with np.load(self._shard_paths[shard_id]) as shard:
                self._cache = {key: shard[key] for key in shard.files}
            self._cache_id = shard_id
        return self._cache
