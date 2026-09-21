"""
Replay a recorded Franka trajectory on Stretch, through the retargeting, with no policy.

The debugging loop this exists to shorten: a Franka rollout picks the object up,
the matching Stretch rollout does not, and the question is whether the
retargeting could have followed the actions that worked. Answering it by running
the policy again costs a VLA in the loop and gives a *different* action stream
every time, because Stretch's observations differ. Answering it from the
recording costs a few seconds of kinematics and asks the question directly:

    the Franka's own recorded joint commands, step by step
      -> FrankaOnStretchView.retarget_franka_joint_pos
      -> where Stretch's gripper actually ends up
      -> against `tcp_pose`, where the Franka's actually was

Nothing here runs a checkpoint, so a replay is deterministic and repeatable, and
changing a retargeting parameter and re-running it is seconds rather than
minutes. That is the whole point: it turns "does the retargeting lose this grasp"
into an experiment you can run in a loop.

Where the trajectories come from
--------------------------------
Not from anything this package writes -- MolmoSpaces' evaluation pipeline already
saves them. Every run leaves one HDF5 per house under

    <run>/<EvalConfigClass>/<timestamp>/house_<n>/trajectories_batch_*.h5

with a group per episode, and in each of those `actions/joint_pos` is the action
dict the policy emitted, JSON-encoded per step: `{"arm": [7 joint targets],
"gripper": [command]}`. `obs/extra/tcp_pose` is where the robot's tool actually
was, and `obs/extra/robot_base_pose` where it was standing. That is everything a
replay needs, so this reads the benchmark's own record rather than a parallel one
written alongside it -- there is only one set of numbers, and it is the set the
report was computed from.

What a replay is and is not
---------------------------
It is kinematic. Joint targets are written and `mj_forward` run, exactly as in
`tests/test_retargeting.py`, so what it measures is the retargeting rather than
the controllers that track it. It replays *the Franka's* actions, so it only
makes sense on a Franka trajectory; a Stretch recording is already retargeted and
is rejected rather than replayed into nonsense.

It is also open-loop, which is the important caveat. The real Stretch rollout
would have seen its own camera frames and diverged from step one; this shows
where the retargeting takes Stretch if it is fed the actions that worked. A
replay that tracks perfectly says the retargeting is not what lost the grasp. A
replay that cannot follow says it is, and says where.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from mujoco import MjData, MjSpec
from scipy.spatial.transform import Rotation as R

from examples.machine_learning.molmospaces.policies import franka_retarget as fr
from examples.machine_learning.molmospaces.retargetting import mini_benchmark
from examples.machine_learning.molmospaces.stretch.config import Stretch4RobotConfig
from examples.machine_learning.molmospaces.stretch.robot import Stretch4Robot
from examples.machine_learning.molmospaces.stretch.robot_view import Stretch4RobotView

log = logging.getLogger(__name__)


TRAJECTORY_GLOB = "trajectories_batch_*.h5"
"""What the evaluation pipeline names its per-house trajectory files."""

FRANKA_ARM_JOINTS = fr.VirtualFranka.N_JOINTS
"""Seven, and the length an action's `arm` entry has to be to be a Franka's."""


# =============================================================================
# Reading what a run recorded
# =============================================================================


@dataclass
class RecordedEpisode:
    """One episode's Franka action stream, and where its robot was."""

    source: Path
    group: str
    house: int

    arm_commands: np.ndarray = field(default_factory=lambda: np.zeros((0, FRANKA_ARM_JOINTS)))
    """`(steps, 7)` Franka joint targets, in the order the policy emitted them."""

    gripper_commands: np.ndarray = field(default_factory=lambda: np.zeros(0))
    """`(steps,)` Robotiq 0-255 commands."""

    tcp_world: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    """`(steps, 3)` where the recorded robot's tool actually was, in *world* coordinates.

    Converted on load, because the pipeline records `tcp_pose` in the robot's own
    base frame: the Franka's home tool comes out as (0.307, 0, 1.015), which is
    relative to the mocap base at the foot of its pedestal. Comparing that
    against a world-frame Stretch pose would be wrong by wherever the robot
    happens to be standing -- several metres, in a kitchen.
    """

    base_pose: np.ndarray = field(default_factory=lambda: np.zeros(7))
    """The robot's base pose at the start, as position + quaternion."""

    instruction: str = ""
    """
    What the episode asked for, when a caller can supply it.

    Left empty by `load_episodes`, because it is not in the trajectory file --
    `obs/extra/task_info` carries `episode_step`, `position_error`,
    `rotation_error` and `success`, and no description. The run's own probe
    records have it; `replay_run` takes a mapping when a caller wants the
    filenames to say which object.
    """

    success: bool = False

    @property
    def steps(self) -> int:
        return len(self.arm_commands)

    @property
    def base_xytheta(self) -> np.ndarray:
        """The base as Stretch's own `(x, y, yaw)`, dropping the recorded height.

        The height is dropped rather than carried because it is the *Franka's*:
        a benchmark Franka stands on a 0.58m pedestal and Stretch stands on the
        floor, so reusing the recorded z would put Stretch's wheels half a metre
        up. The xy and the yaw are what the episode chose and are shared between
        the two robots (`setups._point_base_at` places both from the same rule).
        """
        position = np.asarray(self.base_pose[:3], dtype=float)
        quaternion = np.asarray(self.base_pose[3:7], dtype=float)
        yaw = float(R.from_quat(quaternion, scalar_first=True).as_euler("xyz")[2])
        return np.array([position[0], position[1], yaw])


def _decode(dataset: Any, index: int) -> Any:
    """One JSON-encoded, zero-padded row of an HDF5 dataset.

    The pipeline stores each step's action and observation dicts as UTF-8 JSON
    padded into a fixed-width `uint8` row, which is why these datasets come back
    as `(steps, 2000)` byte arrays rather than anything numeric.
    """
    raw = bytes(np.asarray(dataset[index]).tobytes()).rstrip(b"\x00")
    if not raw:
        return None
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


def _pose_matrix(pose7: np.ndarray) -> np.ndarray:
    """A `(7,)` position + (w, x, y, z) quaternion as a 4x4."""
    pose7 = np.asarray(pose7, dtype=float).reshape(-1)
    matrix = np.eye(4)
    matrix[:3, :3] = R.from_quat(pose7[3:7], scalar_first=True).as_matrix()
    matrix[:3, 3] = pose7[:3]
    return matrix


def _tcp_to_world(trajectory: Any) -> np.ndarray:
    """`obs/extra/tcp_pose`, lifted out of the robot's base frame into the world."""
    if "obs/extra/tcp_pose" not in trajectory or "obs/extra/robot_base_pose" not in trajectory:
        return np.zeros((0, 3))
    tcp = np.asarray(trajectory["obs/extra/tcp_pose"], dtype=float)
    base = np.asarray(trajectory["obs/extra/robot_base_pose"], dtype=float)
    steps = min(len(tcp), len(base))
    return np.array(
        [(_pose_matrix(base[i]) @ _pose_matrix(tcp[i]))[:3, 3] for i in range(steps)]
    )


def find_trajectory_files(root: Path) -> list[Path]:
    """Every trajectory HDF5 under `root`, newest run first.

    Searched rather than constructed, because the pipeline puts a timestamp in
    the path: a run directory holds `<EvalConfigClass>/<timestamp>/house_<n>/`,
    and asking for the newest is how you get the run you just did rather than one
    from last week.
    """
    files = sorted(
        Path(root).rglob(TRAJECTORY_GLOB), key=lambda p: p.stat().st_mtime, reverse=True
    )
    if not files:
        log.warning(
            f"[replay] no {TRAJECTORY_GLOB} under {root}. The evaluation writes one per "
            f"house; if the run crashed before saving there will be none."
        )
    return files


def _house_index(path: Path) -> int:
    """The house number from a `house_<n>` directory in the path, or -1."""
    for part in reversed(path.parts):
        if part.startswith("house_"):
            try:
                return int(part.split("_", 1)[1])
            except ValueError:
                break
    return -1


def load_episodes(path: Path) -> list[RecordedEpisode]:
    """Every Franka episode in one trajectory file.

    A file whose actions are not seven-jointed is skipped with a message rather
    than read: that is a Stretch recording, whose actions are already retargeted
    lift/arm/wrist targets, and replaying those *through* the retargeting would
    apply it twice and produce a confident answer to a question nobody asked.
    """
    import h5py

    episodes: list[RecordedEpisode] = []
    house = _house_index(path)
    with h5py.File(path, "r") as handle:
        for group in sorted(handle.keys()):
            trajectory = handle[group]
            if "actions/joint_pos" not in trajectory:
                log.warning(f"[replay] {path.name}:{group} has no actions/joint_pos, skipping")
                continue

            actions = trajectory["actions/joint_pos"]
            arm, gripper = [], []
            for step in range(actions.shape[0]):
                decoded = _decode(actions, step) or {}
                joints = np.asarray(decoded.get("arm", []), dtype=float).reshape(-1)
                if joints.size != FRANKA_ARM_JOINTS:
                    arm = []
                    break
                arm.append(joints)
                grip = np.asarray(decoded.get("gripper", [0.0]), dtype=float).reshape(-1)
                gripper.append(float(grip[0]) if grip.size else 0.0)

            if not arm:
                log.info(
                    f"[replay] {path.name}:{group} is not a Franka trajectory -- its actions "
                    f"are not seven-jointed, so it is already retargeted. Skipping; replay "
                    f"the Franka half of the pair instead."
                )
                continue

            base_pose = (
                np.asarray(trajectory["obs/extra/robot_base_pose"][0], dtype=float)
                if "obs/extra/robot_base_pose" in trajectory
                else np.zeros(7)
            )
            episodes.append(
                RecordedEpisode(
                    source=path,
                    group=group,
                    house=house,
                    arm_commands=np.asarray(arm, dtype=float),
                    gripper_commands=np.asarray(gripper, dtype=float),
                    tcp_world=_tcp_to_world(trajectory),
                    base_pose=base_pose,
                    success=bool(np.any(np.asarray(trajectory["success"])))
                    if "success" in trajectory
                    else False,
                )
            )
    return episodes


# =============================================================================
# Standing Stretch where the recording stood
# =============================================================================


def scene_for_house(house_index: int, scene_count: int):
    """The `mini_benchmark` scene whose house is `house_index`.

    Matched on `Scene.house_index`, not on position in the list, and the
    difference is not academic: the `house_<n>` directory a run writes is the
    *released dataset's* house index, so a five-scene run over borrowed kitchens
    leaves directories named `house_0`, `house_1011`, `house_1013`, `house_1014`,
    `house_1033`. Treating those as list indices -- which the first version of
    this did -- asks `build_scenes` for 1034 scenes and hangs.

    `scene_count` is how wide to search, and wants to be the `--scenes` the run
    used: `build_scenes` is deterministic, so the first `n` scenes of a replay are
    the first `n` of the run.
    """
    scenes = mini_benchmark.build_scenes(max(1, scene_count))
    for scene in scenes:
        if scene.house_index == house_index:
            return scene
    log.warning(
        f"[replay] no scene with house_index {house_index} among the first {scene_count}; "
        f"falling back to {scenes[0].key}. If the run used more scenes than this, pass a "
        f"larger --scenes so the right house is in the list."
    )
    return scenes[0]


def build_stretch_in_scene(
    house_index: int,
    base_xytheta: np.ndarray,
    scene_count: int = mini_benchmark.DEFAULT_SCENE_COUNT,
):
    """Stretch, spawned in the recorded episode's house at the recorded base pose.

    The house is looked up by index rather than by position; see
    `scene_for_house`.

    The objects are *not* staged. A replay measures where the retargeting puts
    the gripper, which is a question about the robot and the room; the object
    would only be there to be knocked over by an open-loop playback. What the
    recording's own `tcp_pose` gives is a better reference than the object
    anyway: it is where the Franka's gripper actually was at that step.
    """
    from molmo_spaces.utils.lazy_loading_utils import (
        install_scene_with_objects_and_grasps_from_path,
    )

    scene = scene_for_house(house_index, scene_count)
    scene_path = mini_benchmark.house_scene_path(scene)
    install_scene_with_objects_and_grasps_from_path(str(scene_path))
    spec = MjSpec.from_file(str(scene_path))

    config = Stretch4RobotConfig()
    namespace = config.robot_namespace
    x, y, yaw = (float(v) for v in np.asarray(base_xytheta, dtype=float).reshape(-1)[:3])
    Stretch4Robot.add_robot_to_scene(
        config,
        spec,
        prefix=namespace,
        pos=[x, y],
        quat=R.from_euler("z", yaw).as_quat(scalar_first=True),
    )
    Stretch4Robot.apply_control_overrides(spec, config)

    model = spec.compile()
    data = MjData(model)
    view = Stretch4RobotView(data, namespace)
    qpos = dict(config.init_qpos)
    qpos["base"] = [x, y, yaw]
    view.set_qpos_dict(qpos)
    mujoco.mj_forward(model, data)
    return model, data, view, namespace


@dataclass
class ReplayResult:
    """What a replay measured, per step and in summary."""

    episode: RecordedEpisode
    position_error_m: np.ndarray = field(default_factory=lambda: np.zeros(0))
    """Per step: how far Stretch's tool ended up from where the retargeting asked."""

    franka_gap_m: np.ndarray = field(default_factory=lambda: np.zeros(0))
    """Per step: how far Stretch's tool ended up from where the *Franka's* actually was.

    Not the same question as `position_error_m`, and the difference is the point.
    That one is "could Stretch reach the target"; this one is "did Stretch end up
    where the Franka was", which also carries the mount offset
    (`FRANKA_MOUNT_OFFSET_XY`, about 9cm) and `target_z_offset`. A replay whose
    residual is zero and whose gap is 9cm is a retargeting working exactly as
    designed, in a frame deliberately offset from the Franka's.
    """

    unreachable_steps: int = 0

    def summary(self) -> str:
        if not len(self.position_error_m):
            return f"{self.episode.group}: nothing replayed"

        return (
            f"{self.episode.group:8s} {self.episode.instruction[:28]:30s} "
            f"{self.episode.steps:4d} steps  "
            f"residual mean {self.position_error_m.mean() * 1000:6.1f}mm "
            f"max {self.position_error_m.max() * 1000:6.1f}mm  "
            f"unreachable {self.unreachable_steps:4d}"
        )


def replay_episode(
    episode: RecordedEpisode,
    target_z_offset: float = 0.0,
    match_robotiq_aperture: bool = True,
    include_base: bool = True,
    scene_count: int = mini_benchmark.DEFAULT_SCENE_COUNT,
    frame_sink: Any = None,
) -> ReplayResult:
    """Drive Stretch through one recorded Franka trajectory and measure the result.

    `frame_sink`, if given, is called with `(step, model, data, view, proxy)` after
    every step -- which is how a caller renders a video without this function
    knowing anything about rendering.
    """
    model, data, view, namespace = build_stretch_in_scene(
        episode.house, episode.base_xytheta, scene_count=scene_count
    )
    proxy = fr.FrankaOnStretchView(
        view,
        namespace,
        fr.franka_mount_pose_from_base(view.get_move_group("base").joint_pos),
        include_base=include_base,
        target_z_offset=target_z_offset,
        match_robotiq_aperture=match_robotiq_aperture,
    )
    # The rollout's own opening move, for the reason `RetargetRig.restore`
    # gives: `StretchArmIK` converges on a target near the configuration it is
    # seeded from, and Stretch's stowed pose is nowhere near the first command.
    proxy.reset()
    proxy.snap_to_franka_joint_pos()

    residuals, gaps = [], []
    for step, command in enumerate(episode.arm_commands):
        targets = proxy.retarget_franka_joint_pos(command)
        if step < len(episode.gripper_commands):
            targets["gripper"] = proxy.retarget_robotiq_ctrl(episode.gripper_commands[step])
        for group, value in targets.items():
            move_group = view.get_move_group(group)
            move_group.joint_pos = value
            move_group.ctrl = value
        mujoco.mj_forward(model, data)

        residuals.append(proxy.last_position_error)
        reached = np.asarray(view.get_move_group("wrist").leaf_frame_to_world, dtype=float)
        if step < len(episode.tcp_world):
            gaps.append(float(np.linalg.norm(reached[:3, 3] - episode.tcp_world[step])))
        if frame_sink is not None:
            frame_sink(step, model, data, view, proxy)

    return ReplayResult(
        episode=episode,
        position_error_m=np.asarray(residuals, dtype=float),
        franka_gap_m=np.asarray(gaps, dtype=float),
        unreachable_steps=int(proxy.unreachable_steps),
    )


# =============================================================================
# Rendering one
# =============================================================================

REPLAY_PANEL_SIZE = (960, 540)
REPLAY_AZIMUTH_OFFSET_DEG = 90.0
REPLAY_ELEVATION = -20.0
REPLAY_DISTANCE = 2.4
REPLAY_LOOKAT_HEIGHT_M = 1.05
"""The replay camera, matching `params_search_side_by_side`'s so the two read alike."""


def replay_to_video(
    episode: RecordedEpisode,
    output_path: Path,
    fps: float = 15.0,
    **replay_kwargs: Any,
) -> ReplayResult:
    """`replay_episode`, written to an MP4 with the commanded and reached frames drawn.

    Green is where the retargeting asked Stretch's tool to be and orange is where
    it got to, the same colours `tests/test_retargeting.py` uses, so the gap
    between two balls means the same thing in both. A replay where they stay
    coincident is a retargeting that could have followed the Franka's actions.
    """
    import cv2

    output_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = REPLAY_PANEL_SIZE
    state: dict[str, Any] = {"writer": None, "renderer": None, "camera": None}

    def sink(step: int, model, data, view, proxy) -> None:
        if state["renderer"] is None:
            state["renderer"] = mujoco.Renderer(model, height, width)
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            base = np.asarray(view.get_move_group("base").joint_pos, dtype=float)
            camera.lookat[:] = [float(base[0]), float(base[1]), REPLAY_LOOKAT_HEIGHT_M]
            camera.azimuth = math.degrees(float(base[2])) + REPLAY_AZIMUTH_OFFSET_DEG
            camera.elevation = REPLAY_ELEVATION
            camera.distance = REPLAY_DISTANCE
            state["camera"] = camera

        renderer, camera = state["renderer"], state["camera"]
        option = mujoco.MjvOption()
        option.sitegroup = 0
        renderer.update_scene(data, camera=camera, scene_option=option)

        commanded = proxy.franka_tool_pose_to_world(proxy.franka.fk(proxy.last_arm_ctrl))
        reached = np.asarray(view.get_move_group("wrist").leaf_frame_to_world, dtype=float)
        fr.add_frame_marker(
            renderer.scene, commanded, color=fr.FRANKA_TOOL_COLOR, label="commanded"
        )
        fr.add_frame_marker(
            renderer.scene,
            reached,
            color=fr.STRETCH_TOOL_COLOR,
            label=f"stretch {proxy.last_position_error * 1000:.0f}mm",
        )
        frame = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
        if state["writer"] is None:
            state["writer"] = cv2.VideoWriter(
                str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
            )
        state["writer"].write(frame)

    try:
        result = replay_episode(episode, frame_sink=sink, **replay_kwargs)
    finally:
        if state["writer"] is not None:
            state["writer"].release()
        if state["renderer"] is not None:
            state["renderer"].close()
    return result


def replay_run(
    root: Path,
    output_dir: Path,
    render: bool = True,
    limit: int | None = None,
    **replay_kwargs: Any,
) -> list[ReplayResult]:
    """Replay every Franka episode recorded under `root`. Returns what it measured."""
    results: list[ReplayResult] = []
    for path in find_trajectory_files(root):
        for episode in load_episodes(path):
            if limit is not None and len(results) >= limit:
                return results
            name = f"house_{episode.house}_{episode.group}_{_slug(episode.instruction)}"
            log.info(f"[replay] {name}: {episode.steps} recorded steps")
            if render:
                results.append(
                    replay_to_video(episode, output_dir / f"{name}.mp4", **replay_kwargs)
                )
            else:
                results.append(replay_episode(episode, **replay_kwargs))
            log.info(f"[replay] {results[-1].summary()}")
    return results


def _slug(text: str) -> str:
    """A filesystem-safe fragment of an instruction."""
    cleaned = "".join(c if c.isalnum() else "_" for c in (text or "").lower())
    return cleaned.strip("_")[:40] or "episode"


# =============================================================================
# The shared command-line surface
# =============================================================================


def report(results: list[ReplayResult], output_dir: Path, rendered: bool) -> None:
    """Print what a replay measured, loudest first. Shared by both entry points."""
    import click

    if not results:
        click.secho(
            "Nothing to replay. A replay reads the evaluation's own trajectory files "
            f"({TRAJECTORY_GLOB}); run the evaluation first, or point --output-dir at a "
            "directory that already holds one.",
            fg="red",
        )
        return

    click.secho(f"\nReplayed {len(results)} Franka episode(s) as Stretch 4:", bold=True)
    for result in sorted(results, key=lambda r: -r.unreachable_steps):
        click.echo(f"  {result.summary()}")

    worst = max(results, key=lambda r: r.unreachable_steps)
    if worst.unreachable_steps:
        click.secho(
            f"\n{worst.unreachable_steps} of {worst.episode.steps} steps in "
            f"{worst.episode.group} asked for somewhere Stretch could not go with the lift "
            f"against a limit. That is the retargeting being unable to follow the Franka's "
            f"own actions, not the policy aiming badly -- the actions replayed here are the "
            f"ones the Franka rollout used.",
            fg="yellow",
        )
    else:
        click.secho(
            "\nEvery step was reachable: fed the Franka's own actions, the retargeting "
            "follows them. Whatever lost the Stretch rollout, it was not this.",
            fg="green",
        )
    if rendered:
        click.echo(f"\nVideos: {output_dir}")


def replay_output_dir(output_dir: Path) -> Path:
    """Where a replay writes, under a run's own output directory."""
    return Path(output_dir) / "replay_as_stretch4"
