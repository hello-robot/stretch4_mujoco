"""
The MolmoSpaces trajectory format, as a writer and a repair kit.

MolmoSpaces' rollout format is the pivot of everything downstream of a rollout,
which is why this module sits above both consumers rather than inside either.
Its data generation pipeline writes it, `finetuning/lerobot_export.py` and
`training/dataset.py` read it, and -- the reason this module exists beyond its
patterns and codecs -- MolmoBot's trainer reads it *directly*:
`MolmoBot/olmo/data/synthmanip_dataset.py` opens `{data_path}/{split}/house_*/*.h5`
and pulls `obs/agent/qpos`, `actions/joint_pos_rel` (falling back to
`actions/joint_pos`), `obs_scene["task_description"]` and
`obs/sensor_data/{camera}`. So for MolmoBot there is no dataset conversion at
all: generate rollouts and point the trainer at them.

Two things stand between "MolmoSpaces wrote a rollout" and "MolmoBot can train
on it", and both are here:

`ensure_sensor_data_paths()`
    MolmoSpaces' saver strips camera observations *before* batching, to keep the
    HDF5 small (`prepare_episode_for_saving(remove_sensors_if_save_dir=True)`).
    `_save_sensor_data_from_batched()` then finds no camera sensors and leaves
    `obs/sensor_data` an empty group -- while the MP4s it wrote sit in the same
    directory. MolmoBot looks for the filename in that group, so this fills it
    in from the videos that are already there. Nothing is invented: the name is
    the one the saver would have written, `episode_{idx:08d}_{camera}{suffix}.mp4`.

`arrange_train_val_split()`
    MolmoSpaces writes `<run>/house_*/`; MolmoBot expects
    `<task>/train/house_*/` and `<task>/val/house_*/`. Houses are moved (or
    symlinked) whole, never split, so a house's episodes never straddle the
    boundary -- a policy evaluated on a room it trained in is measuring
    memorisation.

The encoding helpers are here too, because the format's two quirks are easy to
get subtly wrong: dict-valued observations are NUL-padded UTF-8 in a uint8 row,
and a camera's video filename is a *fixed 100-byte* uint8 array.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

TRAJECTORY_FILE_PATTERN = re.compile(r"^trajectories(?P<suffix>.*)\.h5$")
TRAJECTORY_KEY_PATTERN = re.compile(r"^traj_(?P<index>\d+)$")

VIDEO_PATH_FIELD_BYTES = 100
"""
Width of the `obs/sensor_data/<camera>` byte array, from MolmoSpaces' own writer.

`_save_sensor_data_from_batched()` allocates exactly `np.zeros(100, np.uint8)`
and writes the filename into the front of it. The width is part of the format,
not a suggestion: a reader that takes the whole row and strips NULs gets the
right answer only if the padding is NULs to exactly this length.
"""

JSON_BLOB_BYTES = 2000
"""
Width MolmoSpaces uses for a dict-valued observation row (`obs/agent/qpos`).

Also fixed-width and NUL-padded. 2000 bytes is what the saver's `fill_arrays`
produces for a qpos dict and is far more than five move groups of floats need;
matching it keeps a recorded episode byte-comparable with a generated one.
"""


# =============================================================================
# Encoding
# =============================================================================


def encode_json_blob(value: dict, width: int = JSON_BLOB_BYTES) -> np.ndarray:
    """A dict -> the NUL-padded uint8 row MolmoSpaces stores it as."""
    payload = json.dumps(value, separators=(",", ":")).encode("utf-8")
    if len(payload) > width:
        raise ValueError(
            f"Encoded observation is {len(payload)} bytes, over the {width}-byte row width. "
            "Widen the row rather than truncating: a clipped JSON blob fails to parse at "
            "training time, thousands of episodes later."
        )
    row = np.zeros(width, dtype=np.uint8)
    row[: len(payload)] = np.frombuffer(payload, dtype=np.uint8)
    return row


def decode_json_blob(row: np.ndarray) -> dict:
    """The inverse of `encode_json_blob`."""
    return json.loads(bytes(row).rstrip(b"\x00").decode("utf-8"))


def encode_video_path(filename: str) -> np.ndarray:
    """A video filename -> the fixed-width byte array `obs/sensor_data/<camera>` holds."""
    payload = filename.encode("utf-8")
    if len(payload) > VIDEO_PATH_FIELD_BYTES:
        raise ValueError(
            f"Video filename {filename!r} is {len(payload)} bytes, over the format's "
            f"{VIDEO_PATH_FIELD_BYTES}-byte field."
        )
    row = np.zeros(VIDEO_PATH_FIELD_BYTES, dtype=np.uint8)
    row[: len(payload)] = np.frombuffer(payload, dtype=np.uint8)
    return row


def decode_video_path(row: np.ndarray) -> str:
    """The inverse of `encode_video_path`."""
    return bytes(row).rstrip(b"\x00").decode("utf-8")


def video_filename(episode_index: int, camera: str, batch_suffix: str = "") -> str:
    """The name MolmoSpaces gives an episode's camera video."""
    return f"episode_{episode_index:08d}_{camera}{batch_suffix}.mp4"


# =============================================================================
# Repair
# =============================================================================


def ensure_sensor_data_paths(
    rollout_dir: Path,
    camera_names: list[str] | None = None,
    dry_run: bool = False,
) -> dict[str, int]:
    """Fill in `obs/sensor_data/<camera>` from the MP4s beside the HDF5.

    MolmoSpaces deletes camera observations before batching, so its saver writes
    an empty `obs/sensor_data` group even though it wrote the videos. MolmoBot's
    dataset reads the video filename out of that group, so without this it sees
    a trajectory with no images.

    Args:
        rollout_dir: a directory containing `house_*/trajectories*.h5` and the
            episode MP4s. Searched recursively.
        camera_names: cameras to write entries for. Defaults to whatever the
            MP4s next to each HDF5 reveal, which is the honest source -- it
            cannot invent a camera that was never rendered.
        dry_run: report what would be written without touching the files.

    Returns:
        `{"trajectories": n, "entries": n, "already_present": n, "missing_video": n}`.
    """
    import h5py

    counts = {"trajectories": 0, "entries": 0, "already_present": 0, "missing_video": 0}
    for h5_path in sorted(Path(rollout_dir).rglob("trajectories*.h5")):
        match = TRAJECTORY_FILE_PATTERN.match(h5_path.name)
        if match is None:
            continue
        batch_suffix = match.group("suffix")

        with h5py.File(h5_path, "r" if dry_run else "a") as h5_file:
            for traj_key in sorted(h5_file.keys()):
                key_match = TRAJECTORY_KEY_PATTERN.match(traj_key)
                if key_match is None:
                    continue
                counts["trajectories"] += 1
                episode_index = int(key_match.group("index"))
                trajectory = h5_file[traj_key]

                cameras = camera_names or _cameras_beside(
                    h5_path.parent, episode_index, batch_suffix
                )
                group = trajectory.require_group("obs/sensor_data") if not dry_run else None
                for camera in cameras:
                    if f"obs/sensor_data/{camera}" in trajectory:
                        counts["already_present"] += 1
                        continue
                    filename = video_filename(episode_index, camera, batch_suffix)
                    if not (h5_path.parent / filename).exists():
                        counts["missing_video"] += 1
                        log.warning(f"[hdf5] no video {h5_path.parent / filename}")
                        continue
                    counts["entries"] += 1
                    if not dry_run:
                        group.create_dataset(
                            camera, data=encode_video_path(filename), dtype=np.uint8
                        )

    log.info(
        f"[hdf5] {'would write' if dry_run else 'wrote'} {counts['entries']} sensor_data "
        f"entries over {counts['trajectories']} trajectories in {rollout_dir} "
        f"({counts['already_present']} already present, {counts['missing_video']} missing videos)"
    )
    return counts


DEBUG_CAMERAS: set[str] = {"chase_camera", "tracker_camera", "review"}


def _cameras_beside(episode_dir: Path, episode_index: int, batch_suffix: str) -> list[str]:
    """Camera names inferred from the MP4s sitting next to a trajectory file."""
    prefix = f"episode_{episode_index:08d}_"
    suffix = f"{batch_suffix}.mp4"
    cameras = []
    for path in sorted(episode_dir.glob(f"{prefix}*{suffix}")):
        camera = path.name[len(prefix) : -len(suffix)] if suffix else path.stem
        if camera and camera not in DEBUG_CAMERAS:
            cameras.append(camera)
    return cameras


# =============================================================================
# Layout
# =============================================================================


def arrange_train_val_split(
    rollout_dir: Path,
    output_dir: Path,
    val_fraction: float = 0.1,
    link: bool = True,
) -> dict[str, list[Path]]:
    """Lay a flat run of `house_*/` directories out as `train/` and `val/`.

    MolmoBot's `--data_paths` wants a task directory holding `train/` and `val/`
    subdirectories of `house_*/*.h5`; MolmoSpaces writes the houses flat. This
    rearranges them.

    Houses move *whole*. Splitting one house's episodes across the boundary
    would put the same room in both sets, and a policy scored on a room it
    trained in is measuring memorisation -- which is the entire failure mode
    generating fresh procedural data was meant to avoid.

    Args:
        rollout_dir: the flat run, containing `house_*/`.
        output_dir: task directory to create `train/` and `val/` under.
        val_fraction: share of *houses* held out. Always at least one house if
            there are two or more.
        link: symlink the house directories rather than copying them. A rollout
            directory is mostly MP4s, so copying doubles a dataset on disk for
            no benefit; pass False if the trainer runs somewhere the symlink
            target will not resolve.

    Returns:
        `{"train": [...], "val": [...]}`, the house directories in each split.
    """
    rollout_dir = Path(rollout_dir)
    output_dir = Path(output_dir)
    houses = sorted(
        path
        for path in rollout_dir.rglob("house_*")
        if path.is_dir() and any(path.glob("trajectories*.h5"))
    )
    if not houses:
        raise FileNotFoundError(
            f"No house_*/trajectories*.h5 under {rollout_dir}. Generate rollouts first with "
            "`python -m examples.machine_learning.molmospaces.finetuning.generate_dataset`."
        )

    # Deterministic and interleaved rather than a random draw or a tail slice:
    # houses come out of the generator in index order, and taking the tail would
    # correlate the split with whatever the last-generated houses have in common.
    stride = max(2, int(round(1.0 / max(val_fraction, 1e-6)))) if val_fraction > 0 else 0
    val_houses = houses[::stride] if stride and len(houses) > 1 else []
    train_houses = [house for house in houses if house not in set(val_houses)]
    if not train_houses:  # one house: train on it, and say so.
        train_houses, val_houses = houses, []
        log.warning(f"[hdf5] only one house in {rollout_dir}; no validation split held out")

    splits = {"train": train_houses, "val": val_houses}
    placed: dict[str, list[Path]] = {}
    for split, split_houses in splits.items():
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        placed[split] = []
        for split_house in split_houses:
            destination = split_dir / split_house.name
            # A symlink that no longer resolves is repaired rather than kept.
            # The targets written here are absolute, so moving or copying a data
            # directory to another machine -- or another path on this one --
            # leaves every split entry dangling. Treating "a symlink exists" as
            # "this house is placed" would then hand the trainer a layout whose
            # houses all point at nothing, and the failure surfaces as an empty
            # dataset rather than as a broken path.
            if destination.is_symlink() and not destination.resolve().exists():
                log.warning(
                    f"[hdf5] {destination} pointed at a missing "
                    f"{os.readlink(destination)}; relinking"
                )
                destination.unlink()
            if destination.exists() or destination.is_symlink():
                placed[split].append(destination)
                continue
            if link:
                destination.symlink_to(split_house.resolve(), target_is_directory=True)
            else:
                shutil.copytree(split_house, destination)
            placed[split].append(destination)

    log.info(
        f"[hdf5] {output_dir}: {len(placed['train'])} train houses, "
        f"{len(placed['val'])} val houses ({'symlinked' if link else 'copied'})"
    )
    return placed


def count_trajectories(rollout_dir: Path, successful_only: bool = False) -> int:
    """How many trajectories a rollout directory holds. For reporting."""
    import h5py

    total = 0
    for h5_path in sorted(Path(rollout_dir).rglob("trajectories*.h5")):
        with h5py.File(h5_path, "r") as h5_file:
            for traj_key in h5_file:
                if TRAJECTORY_KEY_PATTERN.match(traj_key) is None:
                    continue
                if successful_only and not bool(np.any(h5_file[traj_key]["success"][:])):
                    continue
                total += 1
    return total
