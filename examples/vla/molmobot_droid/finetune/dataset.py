"""
Stretch 4 trajectories in the format MolmoBot trains from, and the dataset directory it reads.

`run_benchmark_follow_franka.py` writes one h5 per successful episode (`write_episode()`), next
to that episode's videos, laid out as molmospaces' own trajectories are, for MolmoBot's
`SynthmanipDataset`:

    traj_0/obs/agent/qpos          T = steps + 1 rows, one per observation: JSON {move group:
                                   values} (`stretch4_joints.MOVE_GROUPS`), as uint8 bytes
    traj_0/actions/joint_pos       steps + 2 rows: {} (frame 0 is padding), the action taken
                                   after each observation, then the last one again (the "done"
                                   frame, which training leaves out)
    traj_0/obs/sensor_data/<name>  the file name of that camera's mp4, beside the h5; its frame
                                   i is observation i
    traj_0/obs_scene               JSON: task_description, task_type
    traj_0/stretch4/               the same states and actions as plain arrays, the Franka's
                                   too, Stretch's base pose and how each action was reached
    attrs                          FORMAT, the tool, the instruction, the scene, the run's flags

`build_dataset()` puts the chosen episodes in a directory MolmoBot reads,
`{train,val}/<run>/<episode>.h5` with each split's `valid_trajectory_index.json`, by linking to
the run directories instead of copying them. For a policy whose state and actions stay the
Franka's (ACTION_SPACES), each `<run>/` is a directory of its own instead: copies of the h5 files
with `obs/agent/qpos` and `actions/joint_pos` the Franka's (`stretch4/franka_*8`), beside links to
their videos.
"""

from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from examples.vla.molmobot_droid.finetune.stretch4_joints import JOINTS, MOVE_GROUPS, to_move_groups

FORMAT = "stretch4_follow_franka/1"
"""The h5 files' `format` attribute: how `find_episodes()` tells them from other h5 files."""

FRANKA_FORMAT = "stretch4_follow_franka_as_franka/1"
"""The `format` of `build_dataset()`'s Franka copies, which `find_episodes()` leaves out."""

FRANKA_MOVE_GROUPS: dict[str, int] = {"arm": 7, "gripper": 1}
"""The Franka's move groups, as molmospaces stores them (MolmoBot's `franka_joint` preset)."""

ACTION_SPACES: dict[str, dict[str, int]] = {"stretch4": MOVE_GROUPS, "franka": FRANKA_MOVE_GROUPS}
"""What a fine-tuned policy's state and actions are: Stretch's joints (`stretch4_joints`), or
the Franka's 7 joints and gripper the episode's policy drove, as MolmoBot-DROID's are."""

INDEX_FILE = "valid_trajectory_index.json"
SPLITS = ("train", "val")


@dataclass
class StretchTrajectory:
    """
    One episode of Stretch 4 following the Franka, as `FollowFrankaEnv` records it. Observations
    are keyed by the number of steps before them, so observing twice between steps (as a
    benchmark episode starts) keeps the later one.
    """

    states: dict[int, np.ndarray] = field(default_factory=dict)
    """Stretch's state8 (`stretch4_joints`)."""
    franka_states: dict[int, np.ndarray] = field(default_factory=dict)
    """What the policy was given: the Franka's 7 joints and the driver angle from Stretch's gripper."""
    base_poses: dict[int, np.ndarray] = field(default_factory=dict)
    """Stretch's footprint in the world: x, y (m), yaw (rad)."""
    sim_times: dict[int, float] = field(default_factory=dict)
    actions: list[np.ndarray] = field(default_factory=list)
    """Stretch's action8 for each step: the joint targets retargeting sent it."""
    franka_actions: list[np.ndarray] = field(default_factory=list)
    """The policy's action for each step, after joint-delta limiting."""
    reach: list[str] = field(default_factory=list)
    """For each step: "reached", "clamped", "twisted" or "unreachable" (Stretch held still)."""

    @property
    def steps(self) -> int:
        return len(self.actions)

    def observe(self, state8, franka_state8, base_pose, sim_time: float) -> None:
        step = self.steps
        self.states[step] = np.asarray(state8, dtype=float)
        self.franka_states[step] = np.asarray(franka_state8, dtype=float)
        self.base_poses[step] = np.asarray(base_pose, dtype=float)
        self.sim_times[step] = float(sim_time)

    def act(self, action8, franka_action8, reach: str) -> None:
        self.actions.append(np.asarray(action8, dtype=float))
        self.franka_actions.append(np.asarray(franka_action8, dtype=float))
        self.reach.append(reach)


def _json_rows(rows: list[dict]) -> np.ndarray:
    """One JSON document per row, as zero-padded uint8 bytes (molmospaces' encoding)."""
    encoded = [json.dumps(row).encode() for row in rows]
    table = np.zeros((len(encoded), max(len(e) for e in encoded)), dtype=np.uint8)
    for i, e in enumerate(encoded):
        table[i, : len(e)] = np.frombuffer(e, dtype=np.uint8)
    return table


def write_episode(
    path: Path,
    trajectory: StretchTrajectory,
    videos: dict[str, str],
    instruction: str,
    task_type: str = "pick",
    attrs: dict | None = None,
) -> Path:
    """
    Write `trajectory` to `path` (see the module docstring). `videos` names each camera's mp4,
    in the h5's directory, with one frame per observation. `attrs` (JSON-able) are added to the
    file's. Written to a temporary name first, so a file there is always whole.
    """
    import h5py

    steps = trajectory.steps
    missing = [i for i in range(steps + 1) if i not in trajectory.states]
    if steps == 0 or missing:
        raise ValueError(f"Need an observation before and after each of {steps} steps; missing {missing[:5]}")
    observations = range(steps + 1)
    states = [trajectory.states[i] for i in observations]

    partial = path.with_name(path.name + ".partial")
    with h5py.File(partial, "w") as f:
        traj = f.create_group("traj_0")
        traj["obs/agent/qpos"] = _json_rows([to_move_groups(s) for s in states])
        actions = [to_move_groups(a) for a in trajectory.actions]
        traj["actions/joint_pos"] = _json_rows([{}] + actions + [actions[-1]])
        for camera, video in videos.items():
            traj[f"obs/sensor_data/{camera}"] = np.frombuffer(video.encode(), dtype=np.uint8)
        traj["obs_scene"] = json.dumps({"task_description": instruction, "task_type": task_type})
        traj["success"] = np.array([True])

        native = traj.create_group("stretch4")
        native["state8"] = np.array(states)
        native["action8"] = np.array(trajectory.actions)
        native["franka_state8"] = np.array([trajectory.franka_states[i] for i in observations])
        native["franka_action8"] = np.array(trajectory.franka_actions)
        native["base_pose_world"] = np.array([trajectory.base_poses[i] for i in observations])
        native["sim_time"] = np.array([trajectory.sim_times[i] for i in observations])
        native["reach"] = np.array(trajectory.reach, dtype="S11")
        native.attrs["joints"] = json.dumps(JOINTS)
        f["valid_traj_mask"] = np.array([True])

        f.attrs["format"] = FORMAT
        f.attrs["instruction"] = instruction
        f.attrs["steps"] = steps
        f.attrs["cameras"] = json.dumps(sorted(videos))
        f.attrs["move_groups"] = json.dumps(MOVE_GROUPS)
        for key, value in (attrs or {}).items():
            f.attrs[key] = value if isinstance(value, (str, int, float)) else json.dumps(value)
    partial.replace(path)
    return path


# ---------------------------------------------------------------------------
# Reading them back, for fine-tuning
# ---------------------------------------------------------------------------


@dataclass
class SavedEpisode:
    path: Path
    steps: int
    tool_name: str
    cameras: list[str]
    instruction: str


def find_episodes(paths) -> list[SavedEpisode]:
    """Every h5 `write_episode()` wrote under `paths` (directories, searched recursively, or files)."""
    import h5py

    found: dict[Path, SavedEpisode] = {}
    for root in map(Path, paths):
        for path in [root] if root.is_file() else sorted(root.rglob("*.h5")):
            resolved = path.resolve()
            if resolved in found:
                continue
            with h5py.File(path, "r") as f:
                if f.attrs.get("format") != FORMAT:  # also leaves out `build_dataset()`'s Franka copies
                    continue
                found[resolved] = SavedEpisode(
                    resolved, int(f.attrs["steps"]), str(f.attrs.get("tool_name", "")),
                    json.loads(f.attrs["cameras"]), str(f.attrs["instruction"]),
                )
    return sorted(found.values(), key=lambda e: e.path)


def write_franka_episode(source: Path, house_dir: Path) -> None:
    """
    `source`'s episode in `house_dir` with the Franka's state and actions (FRANKA_MOVE_GROUPS)
    in place of Stretch's: a copy of the h5, and links to its videos (MolmoBot looks for them
    beside it). Its `stretch4/` group is left as it was.
    """
    import h5py

    path = house_dir / source.name
    partial = path.with_name(path.name + ".partial")
    shutil.copyfile(source, partial)
    with h5py.File(partial, "r+") as f:
        traj = f["traj_0"]
        states = [to_move_groups(s, FRANKA_MOVE_GROUPS) for s in traj["stretch4/franka_state8"][:]]
        actions = [to_move_groups(a, FRANKA_MOVE_GROUPS) for a in traj["stretch4/franka_action8"][:]]
        del traj["obs/agent/qpos"], traj["actions/joint_pos"]
        traj["obs/agent/qpos"] = _json_rows(states)
        traj["actions/joint_pos"] = _json_rows([{}] + actions + [actions[-1]])
        f.attrs["format"] = FRANKA_FORMAT
        f.attrs["move_groups"] = json.dumps(FRANKA_MOVE_GROUPS)
        videos = [bytes(traj["obs/sensor_data"][camera][:]).decode() for camera in traj["obs/sensor_data"]]
    partial.replace(path)
    for video in videos:
        link = house_dir / video
        if not link.is_symlink():
            link.symlink_to(source.parent / video)


def build_dataset(episodes: list[SavedEpisode], out_dir: Path, val_fraction: float, seed: int = 0,
                  action_space: str = "stretch4") -> dict[str, int]:
    """
    Lay `episodes` out under `out_dir` as MolmoBot's `SynthmanipDataset` reads them: each split
    holds a link per run directory (its videos are found beside the h5) and an index of the
    split's episodes in it. With `action_space` "franka", a directory per run instead, of the
    episodes' Franka copies (`write_franka_episode()`). `val_fraction` of them (at least one, if
    there are two or more and it is not 0), picked at random with `seed`, are the val split.
    Replaces whatever splits `out_dir` had. Returns the episodes per split.
    """
    if action_space not in ACTION_SPACES:
        raise ValueError(f"Unknown action space {action_space!r}: one of {list(ACTION_SPACES)}")
    order = list(episodes)
    random.Random(seed).shuffle(order)
    n_val = round(len(order) * val_fraction)
    if val_fraction > 0 and len(order) >= 2:
        n_val = min(max(n_val, 1), len(order) - 1)
    chosen = {"val": order[:n_val], "train": order[n_val:]}

    # A run directory is linked as its own name; two runs of the same name get a suffix.
    houses: dict[Path, str] = {}
    for episode in sorted(episodes, key=lambda e: e.path):
        run_dir = episode.path.parent
        if run_dir not in houses:
            name, n = run_dir.name, 1
            while name in houses.values():
                n += 1
                name = f"{run_dir.name}_{n}"
            houses[run_dir] = name

    counts = {}
    for split in SPLITS:
        split_dir = out_dir / split
        if split_dir.exists():
            # Links, the index and Franka copies only: rmtree removes links, never follows them.
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True)
        index: dict[str, dict] = {}
        for episode in chosen[split]:
            house = houses[episode.path.parent]
            link = split_dir / house
            if action_space == "franka":
                link.mkdir(exist_ok=True)
                write_franka_episode(episode.path, link)
            elif not link.exists():
                link.symlink_to(episode.path.parent, target_is_directory=True)
            # Rows of actions/joint_pos: the padding frame, one per step, the done frame.
            index.setdefault(house, {})[f"{house}/{episode.path.name}"] = {"traj_0": episode.steps + 2}
        (split_dir / INDEX_FILE).write_text(json.dumps(index, indent=1))
        counts[split] = len(chosen[split])
    return counts
