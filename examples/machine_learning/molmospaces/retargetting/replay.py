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

Watching one
------------
`replay_to_video` writes the single third-person panel this module started with.
`replay_to_panels` writes the two panels
`params_search_side_by_side.compose_pair` tiles -- the scene, and the camera row
beneath it -- so a replay can be put beside the Franka footage it was recorded
from and read as a side-by-side rather than on its own. That is what
`params_search_side_by_side --replay-as-stretch4` produces.
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
from examples.machine_learning.molmospaces.retargetting.cameras import (
    ExoCameraParams,
    postprocess_exo_frame,
)
from examples.machine_learning.molmospaces.stretch.config import Stretch4RobotConfig
from examples.machine_learning.molmospaces.stretch.robot import Stretch4Robot
from examples.machine_learning.molmospaces.stretch.robot_view import Stretch4RobotView

log = logging.getLogger(__name__)


TRAJECTORY_GLOB = "trajectories_batch_*.h5"
"""What the evaluation pipeline names its per-house trajectory files."""

REPLAY_EXO_CAMERA = "replay_exo_camera"
"""Name of the exo camera this module bolts into a replay scene.

Its own name rather than the evaluation's `exo_camera_1`, because this one is an
MJCF camera compiled into the model while that one is a MolmoSpaces
`RobotMountedCameraConfig` placed at runtime. Same pose, same optics, different
mechanism -- and a shared name would make two different things look like one in
a traceback. The *panel* is labelled with the evaluation's name, which is what a
viewer is comparing against.
"""

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
    def panel_key(self) -> str:
        """The key `SplitPanelRecorder` filed this same episode's panels under.

        The two namings come from different halves of one run and have to be
        matched for a replay to be tiled against the Franka footage it was
        recorded from: the trajectory file calls an episode `traj_<i>` within
        `house_<n>/`, the recorder calls it `house_<n>_ep<i:04d>`. Both count
        from zero in the order the house's episodes ran, so the index carries
        across -- and `replay_panels` checks the step counts agree rather than
        trusting that.
        """
        index = self.group.rsplit("_", 1)[-1]
        return f"house_{self.house}_ep{int(index):04d}" if index.isdigit() else self.group

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


def latest_run_trajectories(root: Path) -> list[Path]:
    """The trajectory files of the *newest* run under `root`, and only those.

    `find_trajectory_files` returns everything it can find, which is the right
    answer for "replay whatever is here" and the wrong one for pairing against a
    run's recorded panels: a directory that has been evaluated twice holds two
    `<EvalConfigClass>/<timestamp>/` trees, every house appears in both, and
    every episode key then arrives twice -- the second overwriting the first's
    panels with a replay of a different rollout.

    So the newest file decides the run, and only its siblings are read. The
    layout is `<run>/<EvalConfigClass>/<timestamp>/house_<n>/<file>`, so that
    run directory is the file's grandparent.
    """
    files = find_trajectory_files(root)
    if not files:
        return []
    run_dir = files[0].parent.parent
    chosen = [path for path in files if path.parent.parent == run_dir]
    if len(chosen) != len(files):
        log.info(
            f"[replay] {root} holds more than one run; replaying the newest "
            f"({run_dir.name}) and ignoring {len(files) - len(chosen)} older file(s)."
        )
    return sorted(chosen)


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


BENCHMARK_FILE = Path("benchmark") / "benchmark.json"
"""Where `mini_benchmark.build` leaves the episode specs, under a run's output directory."""


def episode_staging(output_dir: Path, house_index: int, index: int, instruction: str = "") -> dict:
    """The `scene_modifications` of one episode of the run's own benchmark.

    Which object was on the counter and which of the house's own clutter was
    taken off it is not in the trajectory file -- the benchmark holds it, and it
    is the benchmark the rollout was run from, so it is read rather than
    recomputed. `mini_benchmark.build_episodes` writes scene-major in
    `TARGETS` order, which is the order the houses' episodes run in, so the
    index within a house selects the episode.

    `instruction`, when the caller has one, is checked against the spec's own
    task description: a benchmark rebuilt at a different `--scenes` between the
    run and the replay would line the indices up against different objects, and
    staging the wrong object is worse than staging none.

    Returns `{}` when there is no benchmark to read, which stages nothing.
    """
    path = Path(output_dir) / BENCHMARK_FILE
    if not path.is_file():
        log.warning(
            f"[replay] no {path}; the replay scene will hold the house but not the object "
            f"the episode was about."
        )
        return {}
    try:
        episodes = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as error:
        log.warning(f"[replay] could not read {path}: {error}")
        return {}

    in_house = [e for e in episodes if int(e.get("house_index", -1)) == house_index]
    if not 0 <= index < len(in_house):
        log.warning(
            f"[replay] house {house_index} has {len(in_house)} episodes in {path.name} but "
            f"episode {index} was asked for; staging nothing."
        )
        return {}
    spec = in_house[index]
    described = (spec.get("language") or {}).get("task_description", "")
    if instruction and described and described != instruction:
        log.warning(
            f"[replay] house {house_index} episode {index} is {described!r} in the benchmark "
            f"and {instruction!r} in the recording; staging nothing rather than the wrong "
            f"object. Rebuild the benchmark at the --scenes the run used."
        )
        return {}
    return spec.get("scene_modifications") or {}


def _stage_objects(spec: MjSpec, modifications: dict[str, Any]) -> None:
    """Put the episode's pickup object on the counter, and take its clutter off.

    The same two edits `JsonEvalTaskSampler.add_auxiliary_objects` makes to the
    same scene: delete `removed_objects`, attach each of `added_objects` at its
    recorded pose. A replay used to skip both, on the grounds that it measures
    the gripper rather than the grasp -- which was true of a replay watched on
    its own and is not true of one tiled beside the Franka footage, where the
    left panel shows a hand closing on a bowl and the right showed the same hand
    closing on an empty counter.

    The object is placed and then left alone. Nothing here steps the simulator,
    so it neither falls nor is picked up: it marks where the grasp was supposed
    to happen, and the gap between it and Stretch's gripper is the thing to
    read. It is not a grasp that failed.
    """
    from molmo_spaces.molmo_spaces_constants import ASSETS_DIR
    from molmo_spaces.utils.lazy_loading_utils import install_uid

    from examples.machine_learning.molmospaces.added_pickup_repair import (
        repair_added_pickup_masses,
    )

    mini_benchmark._delete_bodies(spec, list(modifications.get("removed_objects") or []))

    poses = modifications.get("object_poses") or {}
    for object_name, relative in (modifications.get("added_objects") or {}).items():
        object_xml = Path(ASSETS_DIR) / relative
        if not object_xml.is_file():
            object_xml = Path(install_uid(Path(relative).stem))
        try:
            object_spec = MjSpec.from_file(str(object_xml))
            body = object_spec.worldbody.bodies[0]
            if not body.first_joint():
                body.add_joint(name="XYZ_jntfree", type=mujoco.mjtJoint.mjJNT_FREE)
            pose = list(poses.get(object_name, [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]))
            prefix = object_name.rpartition("/")[0] + "/"
            frame = spec.worldbody.add_frame(pos=pose[:3], quat=pose[3:7])
            frame.attach_body(body, prefix, "")
            # Without it a THOR prefab attached straight from the asset weighs
            # tens of kilograms; see `settled_pose`. It changes nothing
            # kinematically, and keeps this scene the one the benchmark settled
            # the object in rather than a heavier variant of it.
            repair_added_pickup_masses(spec, [object_name])
        except Exception as error:  # noqa: BLE001 - a lost object is not a lost replay
            log.warning(f"[replay] could not stage {object_name}: {error}")


def _add_exo_camera(spec: MjSpec, namespace: str, exo: ExoCameraParams) -> None:
    """Bolt the setup's exo camera into the spec, where the evaluation mounts it.

    The same pose and the same optics `exo_camera_config` hands MolmoSpaces,
    expressed as an MJCF camera because a replay has no MolmoSpaces environment
    to place a `RobotMountedCameraConfig` for it. Rendering through a real
    camera also puts the headlight in the right place for free, which the free-
    camera path does not; see `StretchCameraRig`.

    A mount body the replay scene does not have is a lost camera panel, not a
    lost replay: the Franka setups pin the exo camera to an arm link, and there
    is no arm link here to pin it to.
    """
    body_name = exo.mount_body
    if not body_name.startswith(namespace):
        body_name = namespace + body_name
    try:
        body = spec.body(body_name)
    except (KeyError, ValueError):
        body = None
    if body is None:
        log.warning(
            f"[replay] no body {body_name!r} to mount the exo camera on; the replay's "
            f"camera row will show the wrist camera only."
        )
        return
    body.add_camera(
        name=namespace + REPLAY_EXO_CAMERA,
        pos=list(exo.pos),
        quat=exo.quat_wxyz(),
        fovy=float(exo.fovy),
        resolution=list(exo.render_size),
    )


@dataclass
class ReplayScene:
    """One compiled replay scene, and how to put it back the way it was compiled.

    A scene is nearly all of what a replay costs: installing the house's assets,
    assembling the MJCF and compiling it take seconds, against tens of
    milliseconds for the kinematics or the couple of seconds of stepping that
    follow. A parameter search runs the same episode at a dozen points and
    nothing about the scene differs between them, so it builds one of these once
    and hands it back to `replay_episode` each time. `reset` is what makes that
    sound: `mj_resetData` restores `qpos0`, which for the staged pickup object is
    the pose the benchmark settled it in, so every trial starts from the same
    counter rather than from wherever the last one left the bowl.
    """

    model: Any
    data: MjData
    view: Stretch4RobotView
    namespace: str
    init_qpos: dict[str, Any]

    object_name: str = ""
    """The staged pickup object's body name, or empty if the scene holds none.

    Carried on the scene rather than re-derived per replay because it is decided
    by what was compiled in: a caller reusing a scene passes no `stage`, and
    asking it for the object name again would get nothing.
    """

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.view.set_qpos_dict(self.init_qpos)
        mujoco.mj_forward(self.model, self.data)


def build_stretch_in_scene(
    house_index: int,
    base_xytheta: np.ndarray,
    scene_count: int = mini_benchmark.DEFAULT_SCENE_COUNT,
    exo: ExoCameraParams | None = None,
    stage: dict[str, Any] | None = None,
):
    """Stretch, spawned in the recorded episode's house at the recorded base pose.

    The house is looked up by index rather than by position; see
    `scene_for_house`.

    `exo`, when given, mounts that setup's third-person camera on the robot so a
    replay can show what the policy would have been looking at. Left out, the
    scene carries only Stretch's own cameras, which is all the numbers need.

    `stage` is the episode's `scene_modifications` -- see `episode_staging` and
    `_stage_objects`. Left out, the scene is the bare house: no pickup object and
    the clutter the episode removes still standing. That is enough to measure
    where the retargeting puts the gripper, which is all a replay used to be
    asked, and not enough to watch one beside the rollout it came from.

    Returns a `ReplayScene`, which a caller replaying the same episode more than
    once should keep and hand back rather than rebuild.
    """
    from molmo_spaces.utils.lazy_loading_utils import (
        install_scene_with_objects_and_grasps_from_path,
    )

    scene = scene_for_house(house_index, scene_count)
    scene_path = mini_benchmark.house_scene_path(scene)
    install_scene_with_objects_and_grasps_from_path(str(scene_path))
    spec = MjSpec.from_file(str(scene_path))

    if stage:
        _stage_objects(spec, stage)

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
    if exo is not None:
        _add_exo_camera(spec, namespace, exo)

    model = spec.compile()
    data = MjData(model)
    view = Stretch4RobotView(data, namespace)
    qpos = dict(config.init_qpos)
    qpos["base"] = [x, y, yaw]
    scene = ReplayScene(
        model=model,
        data=data,
        view=view,
        namespace=namespace,
        init_qpos=qpos,
        object_name=staged_object_name(stage),
    )
    scene.reset()
    return scene


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

    grasp: "GraspOutcome | None" = None
    """What the object did, when the replay was stepped with the physics on.

    `None` for a kinematic replay, which cannot have an opinion: it writes joint
    positions and runs `mj_forward`, so the object never moves and "it did not
    pick it up" is a property of the replay rather than of the retargeting. See
    `replay_episode`'s `physics`.
    """

    def summary(self) -> str:
        if not len(self.position_error_m):
            return f"{self.episode.group}: nothing replayed"

        line = (
            f"{self.episode.group:8s} {self.episode.instruction[:28]:30s} "
            f"{self.episode.steps:4d} steps  "
            f"residual mean {self.position_error_m.mean() * 1000:6.1f}mm "
            f"max {self.position_error_m.max() * 1000:6.1f}mm  "
            f"unreachable {self.unreachable_steps:4d}"
        )
        if self.grasp is not None:
            line += (
                f"  {'PICKED  ' if self.grasp.success else 'no grasp'}"
                f" approach {self.grasp.approach:4.2f}"
                f" gap {self.grasp.min_distance_m * 1000:5.1f}mm"
                f" lift {self.grasp.best_lift_m * 1000:5.1f}mm"
            )
        return line


# =============================================================================
# Grasping, when the replay is allowed to touch anything
# =============================================================================

GRASP_SUCCESS_LIFT_M = 0.01
"""How far the object has to come off its start height to count, in metres.

`PickTaskConfig.succ_pos_threshold`, which is what the benchmark scores on and
therefore the only threshold worth measuring against here.
"""

SETTLE_SECONDS = 0.4
"""How long the scene is stepped, with the robot holding still, before a replay begins.

The staged object is attached at the pose the benchmark settled it in, but it is
attached into a *different* compilation -- Stretch's, not the Franka's -- so it
arrives with whatever penetration the two models' contact parameters disagree
about and drops a fraction of a millimetre. Stepping that out first is what makes
`best_lift_m` a measurement of the grasp rather than of the settle.
"""


@dataclass
class GraspOutcome:
    """Whether a physics replay actually picked the object up, and how close it came.

    The same four numbers `scoring.EpisodeScore` records for a real rollout, and
    deliberately so: a replay is only useful as a stand-in for a rollout if the
    two are scored the same way, and `scoring.EpisodeScore` is what the report
    ranks setups on. `success` is `PickTask.get_info`'s own test -- the object
    off its start height by `GRASP_SUCCESS_LIFT_M` while nothing but the robot is
    touching it -- rather than a lift threshold on its own, which would pass an
    object still resting on the counter that the gripper had shoved uphill.
    """

    object_name: str = ""
    success: bool = False
    touched: bool = False
    start_distance_m: float = float("nan")
    min_distance_m: float = float("nan")
    best_lift_m: float = 0.0
    """The furthest off its start height the object got *while only the robot held it*.

    Gated on the same contact test as `success` for the same reason: an object
    the gripper has driven into the counter and wedged upwards has not been
    lifted, and an ungated maximum would report it as a near miss worth chasing.
    """

    final_gap_m: float = float("nan")
    """Finger separation at the end, in metres. A hand that shut to 0 gripped nothing."""

    min_finger_gap_m: float = float("inf")
    """The closest the *fingers* ever came to the object, surface to surface, in metres.

    `min_distance_m` above is the one `scoring.EpisodeScore` uses -- tool frame to
    object origin -- and it is the wrong number to read a Stretch grasp off for
    two reasons that both make a good grasp look like a miss. Stretch's tool
    frame sits 1.5cm *past* its fingertips where the Robotiq's sits between its
    pads, and `grasp_offset_m` deliberately drives that frame past the object by
    another few centimetres. So a hand closed perfectly around a potato reports
    five or six centimetres of "distance". This measures between the geoms that
    actually touch it, so zero means the fingers are on the object and nothing
    else does.
    """

    @property
    def approach(self) -> float:
        """Fraction of the opening gripper-to-object gap that was closed, in [0, 1]."""
        if not np.isfinite(self.start_distance_m) or self.start_distance_m <= 1e-6:
            return 0.0
        closed = (self.start_distance_m - self.min_distance_m) / self.start_distance_m
        return float(np.clip(closed, 0.0, 1.0))


class _GraspWatch:
    """Watches one staged object through a physics replay.

    Reads the same three things `PickTask` and `scoring.GraspProbe` read -- the
    object's height against where it started, which root bodies are touching it,
    and how far the grasp frame is from it -- so a replay's verdict and a
    rollout's are the same verdict.
    """

    FINGER_BODIES = ("gripper_finger_right_link", "gripper_finger_left_link")

    def __init__(
        self, model, data, view: Stretch4RobotView, object_name: str, namespace: str = ""
    ) -> None:
        self.model, self.data, self.view = model, data, view
        self.outcome = GraspOutcome(object_name=object_name)
        self.body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_name)
        if self.body < 0:
            log.warning(
                f"[replay] no body {object_name!r} in the compiled scene; the replay will run "
                f"but cannot say whether it grasped anything."
            )
            self.root = -1
        else:
            self.root = int(model.body_rootid[self.body])
        self.robot_root = int(view.base.root_body_id)
        self.start_z = float("nan")
        self.object_geoms = self._collidable(self.body) if self.body >= 0 else []
        self.finger_geoms = [
            geom
            for name in self.FINGER_BODIES
            for geom in self._collidable(
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, namespace + name)
            )
        ]

    def _collidable(self, body_id: int) -> list[int]:
        """The geoms under `body_id` that can touch something.

        Visual-only geoms are excluded -- a decorative shell that overhangs the
        fingers would report a gap of zero while the pads were still centimetres
        off -- and so is a body that is not in the model, which is how a gripper
        whose links are named differently degrades to "no finger measurement"
        rather than to a traceback.
        """
        if body_id is None or body_id < 0:
            return []
        from molmo_spaces.utils.mj_model_and_data_utils import descendant_geoms

        return [
            geom
            for geom in descendant_geoms(self.model, body_id, visible_only=False)
            if self.model.geom_contype[geom] or self.model.geom_conaffinity[geom]
        ]

    @property
    def live(self) -> bool:
        return self.body >= 0

    def start(self) -> None:
        """Freeze the object's rest height, after the settle. The denominator of a lift."""
        if not self.live:
            return
        self.start_z = float(self.data.xpos[self.body][2])
        self.outcome.start_distance_m = self._distance()
        self.outcome.min_distance_m = self.outcome.start_distance_m

    def _distance(self) -> float:
        tool = np.asarray(self.view.get_move_group("gripper").leaf_frame_to_world, dtype=float)
        return float(np.linalg.norm(tool[:3, 3] - np.asarray(self.data.xpos[self.body])))

    def _contacts(self) -> tuple[bool, bool]:
        """`(the robot is touching it, something else is)`. See `PickTask.get_info`."""
        robot = other = False
        model = self.model
        for contact in self.data.contact:
            root1 = int(model.body_rootid[model.geom_bodyid[contact.geom1]])
            root2 = int(model.body_rootid[model.geom_bodyid[contact.geom2]])
            if (root1 == self.root) ^ (root2 == self.root):
                who = root1 if root1 != self.root else root2
                if who == self.robot_root:
                    robot = True
                else:
                    other = True
                    break
        return robot, other

    def _finger_gap(self) -> float:
        """Surface-to-surface metres between the nearest finger geom and the object."""
        if not (self.object_geoms and self.finger_geoms):
            return float("inf")
        closest = float("inf")
        for finger in self.finger_geoms:
            for geom in self.object_geoms:
                gap = mujoco.mj_geomDistance(self.model, self.data, finger, geom, 1.0, None)
                closest = min(closest, float(gap))
        return closest

    def step(self) -> None:
        if not self.live:
            return
        outcome = self.outcome
        outcome.min_distance_m = min(outcome.min_distance_m, self._distance())
        outcome.min_finger_gap_m = min(outcome.min_finger_gap_m, self._finger_gap())
        robot, other = self._contacts()
        outcome.touched = outcome.touched or robot
        if robot and not other:
            lift = float(self.data.xpos[self.body][2]) - self.start_z
            outcome.best_lift_m = max(outcome.best_lift_m, lift)
            outcome.success = outcome.success or lift >= GRASP_SUCCESS_LIFT_M

    def finish(self) -> None:
        if not self.live:
            return
        self.outcome.final_gap_m = float(self.view.get_move_group("gripper").inter_finger_dist)


def staged_object_name(stage: dict[str, Any] | None) -> str:
    """The body name `_stage_objects` will give the episode's pickup object.

    `added_objects` is keyed by the name MolmoSpaces attaches the prefab under,
    and `_stage_objects` attaches it with that same name as its prefix -- so the
    key is the compiled body name, and the one `PickTask` would have scored.
    """
    added = list((stage or {}).get("added_objects") or {})
    return added[0] if added else ""


class _Actuation:
    """The control stack MolmoSpaces puts between a policy's targets and `mj_data.ctrl`.

    Rebuilt here rather than borrowed from `Stretch4Robot`, which needs a whole
    `MlSpacesExpConfig` and an environment to exist; what a replay needs is the
    two things that stand between a retargeted target and the actuator, and both
    are constructible from a move group alone. Same controllers, same wrapper,
    same `ctrl_dt` -- see `stretch.robot.Stretch4Robot.__init__`, which this
    mirrors, and `stretch.motion_limits`, which is why an unshaped replay would
    move the lift at eight times the hardware's top speed and bounce the object
    off the counter.
    """

    GROUPS = ("base", "lift", "arm", "wrist", "gripper")

    def __init__(self, view: Stretch4RobotView, ctrl_dt: float) -> None:
        from molmo_spaces.controllers.joint_pos import JointPosController

        from examples.machine_learning.molmospaces.stretch.motion_limits import rate_limited

        self.controllers = {
            group: rate_limited(JointPosController(view.get_move_group(group)), group, ctrl_dt)
            for group in self.GROUPS
        }

    def update(self, targets: dict[str, Any]) -> None:
        """Set this policy step's targets. Groups left out are told to hold still.

        The same rule `Robot.update_control` follows, and it matters for the base:
        with `include_base` off there is no base entry in a retargeted action, and
        a controller left holding a stale target would keep driving towards it.
        """
        for group, controller in self.controllers.items():
            value = targets.get(group)
            if value is not None:
                controller.set_target(np.asarray(value, dtype=float).reshape(-1))
            elif not controller.stationary:
                controller.set_to_stationary()

    def hold(self) -> None:
        """Target wherever the robot is now -- what a settle needs."""
        for controller in self.controllers.values():
            controller.set_to_stationary()

    def apply(self) -> None:
        """One control interval. `Robot.compute_control`."""
        for controller in self.controllers.values():
            controller.robot_move_group.ctrl = controller.compute_ctrl_inputs()


def replay_episode(
    episode: RecordedEpisode,
    target_z_offset: float = 0.0,
    match_robotiq_aperture: bool = True,
    include_base: bool = True,
    scene_count: int = mini_benchmark.DEFAULT_SCENE_COUNT,
    exo: ExoCameraParams | None = None,
    stage: dict[str, Any] | None = None,
    tool_correction: tuple[float, float] | None = None,
    frame_sink: Any = None,
    physics: bool = False,
    policy_dt_ms: float = 66.0,
    ctrl_dt_ms: float = 2.0,
    sim_dt_ms: float = 2.0,
    aperture_m: float | None = None,
    scene: "ReplayScene | None" = None,
) -> ReplayResult:
    """Drive Stretch through one recorded Franka trajectory and measure the result.

    `frame_sink`, if given, is called with `(step, model, data, view, proxy)` after
    every step -- which is how a caller renders a video without this function
    knowing anything about rendering.

    `tool_correction` is `(wrist_tilt_deg, grasp_offset_m)`, the two terms a
    setup adds to the fixed Franka-to-Stretch tool transform. Passed by a caller
    replaying *against a setup* -- without it the replay measures the bare
    retargeting, which is a different question from the one a side-by-side
    against `stretch_baseline` asks. See `setups.apply_tool_correction`.

    `physics` decides what kind of replay this is, and it is the difference
    between two quite different questions:

    * Off (the default): joint targets are *written* and `mj_forward` run, so the
      robot is teleported through the retargeted configurations and nothing it
      touches moves. That measures the retargeting's reach -- `position_error_m`,
      `franka_gap_m`, `unreachable_steps` -- and it is all those numbers need.
      It cannot pick anything up, so a video of one showing an object sitting
      still is not a failed grasp; it is a replay with the physics off.
    * On: the targets go through the same controllers an evaluation uses
      (`_Actuation`) and the scene is stepped at `sim_dt_ms` for each
      `policy_dt_ms` tick, so the gripper closes on the object, the object has
      mass and friction, and `ReplayResult.grasp` says whether it came off the
      counter. That is the question "would the retargeting have held this grasp
      if the policy had aimed perfectly", and it is answerable in seconds with no
      VLA in the loop -- which is what makes tuning `grasp_offset_m` a search
      rather than a series of twenty-minute benchmark runs.

    The open-loop caveat is the same in both and is worth restating for the
    physics one: these are the *Franka's* actions, replayed blind. A real Stretch
    rollout would see its own camera and diverge. A physics replay that grasps
    says the retargeting can hold the grasp the Franka found; it does not say the
    policy will find it through Stretch's camera.

    `aperture_m` overrides how wide "open" is on Stretch, in metres between the
    pads -- `fr.ROBOTIQ_MAX_APERTURE_M` when left out. Only meaningful with
    `match_robotiq_aperture`, which is what narrows the hand at all.

    A recorded episode stops at the step the *Franka* succeeded on, which raises
    the obvious worry that Stretch -- whose controllers are shaped to the
    hardware's joint speeds, so every command arrives as a ramp -- is still
    tracking the last few when the replay runs out, and that a grasp is being
    scored mid-reach. Measured, by repeating the final command for 20 and 40
    further ticks across all 20 episodes: not one outcome changes, and neither
    does any lift to a tenth of a millimetre. The arm has converged well before
    the recording ends, so there is no hold here and nothing to tune.

    `scene`, when given, is reset and reused instead of a fresh one being built;
    the house, the base pose, the exo camera and the staging are then already
    decided by whoever built it, and `scene_count`, `exo` and `stage` are
    ignored. See `ReplayScene`, and note that a scene built without `stage` holds
    no object to grasp however the physics is stepped.
    """
    if scene is None:
        scene = build_stretch_in_scene(
            episode.house, episode.base_xytheta, scene_count=scene_count, exo=exo, stage=stage
        )
    else:
        scene.reset()
    model, data, view, namespace = scene.model, scene.data, scene.view, scene.namespace
    if physics:
        model.opt.timestep = sim_dt_ms / 1000.0
    proxy = fr.FrankaOnStretchView(
        view,
        namespace,
        fr.franka_mount_pose_from_base(view.get_move_group("base").joint_pos),
        include_base=include_base,
        target_z_offset=target_z_offset,
        match_robotiq_aperture=match_robotiq_aperture,
        robotiq_aperture_m=aperture_m,
    )
    if tool_correction is not None:
        from examples.machine_learning.molmospaces.retargetting.setups import (
            apply_tool_correction,
        )

        apply_tool_correction(
            proxy, wrist_tilt_deg=tool_correction[0], grasp_offset_m=tool_correction[1]
        )
    # The rollout's own opening move, for the reason `RetargetRig.restore`
    # gives: `StretchArmIK` converges on a target near the configuration it is
    # seeded from, and Stretch's stowed pose is nowhere near the first command.
    proxy.reset()
    proxy.snap_to_franka_joint_pos()

    actuation = watch = None
    if physics:
        actuation = _Actuation(view, ctrl_dt_ms / 1000.0)
        actuation.hold()
        # The snap wrote a configuration straight into `qpos`; settling lets the
        # controllers take hold of it and the staged object come to rest before
        # its height is read, so `best_lift_m` measures the grasp and not the drop.
        for _ in range(max(0, int(round(SETTLE_SECONDS / (sim_dt_ms / 1000.0))))):
            actuation.apply()
            mujoco.mj_step(model, data)
        watch = _GraspWatch(
            model, data, view, scene.object_name or staged_object_name(stage), namespace
        )
        watch.start()

    substeps = max(1, int(round(policy_dt_ms / sim_dt_ms)))
    commands = list(episode.arm_commands)
    residuals, gaps = [], []
    for step, command in enumerate(commands):
        targets = proxy.retarget_franka_joint_pos(command)
        gripper_step = min(step, len(episode.gripper_commands) - 1)
        if gripper_step >= 0:
            targets["gripper"] = proxy.retarget_robotiq_ctrl(episode.gripper_commands[gripper_step])

        if actuation is None:
            for group, value in targets.items():
                move_group = view.get_move_group(group)
                move_group.joint_pos = value
                move_group.ctrl = value
            mujoco.mj_forward(model, data)
        else:
            actuation.update(targets)
            for _ in range(substeps):
                actuation.apply()
                mujoco.mj_step(model, data)
            watch.step()

        residuals.append(proxy.last_position_error)
        reached = np.asarray(view.get_move_group("wrist").leaf_frame_to_world, dtype=float)
        if step < len(episode.tcp_world):
            gaps.append(float(np.linalg.norm(reached[:3, 3] - episode.tcp_world[step])))
        if frame_sink is not None:
            frame_sink(step, model, data, view, proxy)

    if watch is not None:
        watch.finish()
    return ReplayResult(
        episode=episode,
        position_error_m=np.asarray(residuals, dtype=float),
        franka_gap_m=np.asarray(gaps, dtype=float),
        unreachable_steps=int(proxy.unreachable_steps),
        grasp=watch.outcome if watch is not None else None,
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


class _ScenePanel:
    """The third-person replay panel: one fixed camera, the two tool markers.

    `base_xytheta` is where the *episode* stood the robot, and it has to be
    passed in rather than read off the robot when the first frame arrives. With
    `include_base` set -- the default -- the base is part of what the IK solves,
    and the opening `snap_to_franka_joint_pos` can drive it metres to reach the
    Franka's home pose before a single frame is rendered. A camera framed on
    where the base ended up after that is a camera outside the house looking at
    the exterior wall, which is exactly what replay videos used to show.

    The renderer still waits for the first frame, because the model arrives with
    it. The camera is then held, never re-aimed as the base moves, for the
    reason `params_search_side_by_side.SplitPanelRecorder._scene_frame` gives: a
    panel that drifts cannot be tiled against one that does not.
    """

    def __init__(
        self, base_xytheta: np.ndarray, size: tuple[int, int] = REPLAY_PANEL_SIZE
    ) -> None:
        self.size = size
        self._base = np.asarray(base_xytheta, dtype=float).reshape(-1)[:3]
        self._renderer: Any = None
        self._camera: Any = None

    def frame(self, model, data, view, proxy) -> np.ndarray:
        """One BGR panel: the scene, the commanded frame, and the reached one.

        Green is where the retargeting asked Stretch's tool to be and orange is
        where it got to, the same colours `tests/test_retargeting.py` uses, so
        the gap between two balls means the same thing in both. A replay where
        they stay coincident is a retargeting that could have followed the
        Franka's actions.
        """
        import cv2

        width, height = self.size
        if self._renderer is None:
            self._renderer = mujoco.Renderer(model, height, width)
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            base = self._base
            camera.lookat[:] = [float(base[0]), float(base[1]), REPLAY_LOOKAT_HEIGHT_M]
            camera.azimuth = math.degrees(float(base[2])) + REPLAY_AZIMUTH_OFFSET_DEG
            camera.elevation = REPLAY_ELEVATION
            camera.distance = REPLAY_DISTANCE
            self._camera = camera

        option = mujoco.MjvOption()
        option.sitegroup = 0
        self._renderer.update_scene(data, camera=self._camera, scene_option=option)

        commanded = proxy.franka_tool_pose_to_world(proxy.franka.fk(proxy.last_arm_ctrl))
        reached = np.asarray(view.get_move_group("wrist").leaf_frame_to_world, dtype=float)
        fr.add_frame_marker(
            self._renderer.scene, commanded, color=fr.FRANKA_TOOL_COLOR, label="commanded"
        )
        fr.add_frame_marker(
            self._renderer.scene,
            reached,
            color=fr.STRETCH_TOOL_COLOR,
            label=f"stretch {proxy.last_position_error * 1000:.0f}mm",
        )
        return cv2.cvtColor(self._renderer.render(), cv2.COLOR_RGB2BGR)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
        self._renderer = None


class _CameraPanels:
    """What a replayed Stretch's policy would have been looking at.

    The exo camera under test plus Stretch's own right wrist camera -- the pair
    `setups.stretch_camera_system` gives a Stretch setup, rendered here the two
    ways those two cameras are rendered there: the exo through the MJCF camera
    `_add_exo_camera` compiled in and then `postprocess_exo_frame` (so a fisheye
    setup's replay is warped, cropped and turned exactly as the rollout's was),
    the wrist through `StretchCameraRig`, which is the hardware-accurate path
    `install_stretch_camera_hooks` patches into the evaluation.

    What it is *not* is what the policy saw, because in a replay no policy ran
    and the Franka's actions are being followed open-loop. It is what the same
    camera would have shown of the pose the retargeting actually reached.
    """

    def __init__(self, model, namespace: str, exo: ExoCameraParams | None) -> None:
        from examples.machine_learning.molmospaces.demo_droid_on_stretch import (
            StretchCameraRig,
        )
        from examples.machine_learning.molmospaces.retargetting.setups import EXO_CAMERA
        from examples.machine_learning.molmospaces.stretch.config import (
            STRETCH_CAMERA_FOR_CAMERA,
            WRIST_CAMERA_RIGHT,
        )

        self._exo_name = EXO_CAMERA
        self._exo_params = exo
        self._exo_renderer = None
        camera_name = namespace + REPLAY_EXO_CAMERA
        mounted = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name) >= 0
        if exo is not None and mounted:
            width, height = exo.render_size
            self._exo_renderer = mujoco.Renderer(model, height, width)
            self._exo_camera = camera_name
        self._option = mujoco.MjvOption()
        self._option.sitegroup = 0
        self._wrist_name = WRIST_CAMERA_RIGHT
        self._wrist = StretchCameraRig(
            model, namespace, {WRIST_CAMERA_RIGHT: STRETCH_CAMERA_FOR_CAMERA[WRIST_CAMERA_RIGHT]}
        )

    def frames(self, data) -> list[tuple[str, np.ndarray]]:
        """This step's panels, BGR and in the order the evaluation records them."""
        import cv2

        panels: list[tuple[str, np.ndarray]] = []
        if self._exo_renderer is not None:
            self._exo_renderer.update_scene(
                data, camera=self._exo_camera, scene_option=self._option
            )
            frame = postprocess_exo_frame(self._exo_renderer.render(), self._exo_params)
            panels.append((self._exo_name, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)))
        wrist = self._wrist.render(data)[self._wrist_name]
        wrist = cv2.cvtColor(np.ascontiguousarray(wrist), cv2.COLOR_RGB2BGR)
        panels.append((self._wrist_name, wrist))
        return panels

    def close(self) -> None:
        if self._exo_renderer is not None:
            self._exo_renderer.close()
        self._exo_renderer = None


def camera_row(frames: list[tuple[str, np.ndarray | None]], width: int) -> np.ndarray:
    """The camera panels as one row `width` wide, laid out as a recorded run's are.

    Shared with `params_search_side_by_side.SplitPanelRecorder`, which is the
    point: the row under a replayed Stretch and the row under the Franka it is
    tiled against have to be the same shape and the same proportions, or the
    split screen stops being one picture.
    """
    from examples.machine_learning.molmospaces.visualize import EpisodeVideoRecorder

    height = max(1, int(round(width / max(1, len(frames)) * 0.6)))
    return EpisodeVideoRecorder._camera_grid(frames, width, height)


def _ensure_offscreen_buffer(model, *sizes: tuple[int, int]) -> None:
    """Grow the model's offscreen framebuffer to hold every size about to be rendered.

    A scene declares its own `<visual><global offwidth=...>` and MuJoCo refuses
    to build a larger renderer, as a hard error. Same guard
    `stretch.config.install_stretch_camera_hooks` puts on the evaluation path,
    for the same reason: a fisheye render size is bigger than the 640x480 a
    scene is likely to declare.
    """
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), *(s[0] for s in sizes))
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), *(s[1] for s in sizes))


def replay_to_video(
    episode: RecordedEpisode,
    output_path: Path,
    fps: float = 15.0,
    **replay_kwargs: Any,
) -> ReplayResult:
    """`replay_episode`, written to a single-panel MP4. See `_ScenePanel.frame`."""
    import cv2

    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel = _ScenePanel(episode.base_xytheta)
    state: dict[str, Any] = {"writer": None}

    def sink(step: int, model, data, view, proxy) -> None:
        if state["writer"] is None:
            _ensure_offscreen_buffer(model, REPLAY_PANEL_SIZE)
        frame = panel.frame(model, data, view, proxy)
        if state["writer"] is None:
            state["writer"] = cv2.VideoWriter(
                str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, REPLAY_PANEL_SIZE
            )
        state["writer"].write(frame)

    try:
        result = replay_episode(episode, frame_sink=sink, **replay_kwargs)
    finally:
        if state["writer"] is not None:
            state["writer"].release()
        panel.close()
    return result


def replay_to_panels(
    episode: RecordedEpisode,
    panel_dir: Path,
    key: str,
    exo: ExoCameraParams | None,
    stage: dict[str, Any] | None = None,
    fps: float = 15.0,
    **replay_kwargs: Any,
) -> ReplayResult:
    """A replay written as the *two* panels `compose_pair` tiles: scene, and cameras.

    Deliberately the same two files under the same names a rollout's
    `SplitPanelRecorder` writes -- `<key>_scene.mp4` and `<key>_cams.mp4` in a
    panel directory -- so the tiling code needs to know nothing about replays.
    What ends up on the right of the split screen is then produced by the same
    composer as what is on the left, and the only difference between the halves
    is the one being investigated.
    """
    import cv2

    panel_dir.mkdir(parents=True, exist_ok=True)
    scene = _ScenePanel(episode.base_xytheta)
    state: dict[str, Any] = {"scene": None, "cams": None, "cameras": None}

    def open_writer(path: Path, frame: np.ndarray):
        height, width = frame.shape[:2]
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            log.warning(f"[replay] could not open {path}")
            return None
        return writer

    def sink(step: int, model, data, view, proxy) -> None:
        if state["cameras"] is None:
            sizes = [REPLAY_PANEL_SIZE] + ([exo.render_size] if exo is not None else [])
            _ensure_offscreen_buffer(model, *sizes)
            state["cameras"] = _CameraPanels(model, proxy.namespace, exo)

        frame = scene.frame(model, data, view, proxy)
        if state["scene"] is None:
            state["scene"] = open_writer(panel_dir / f"{key}_scene.mp4", frame)
        if state["scene"] is not None:
            state["scene"].write(frame)

        row = camera_row(state["cameras"].frames(data), REPLAY_PANEL_SIZE[0])
        if state["cams"] is None:
            state["cams"] = open_writer(panel_dir / f"{key}_cams.mp4", row)
        if state["cams"] is not None:
            state["cams"].write(row)

    try:
        result = replay_episode(episode, exo=exo, stage=stage, frame_sink=sink, **replay_kwargs)
    finally:
        for writer in (state["scene"], state["cams"]):
            if writer is not None:
                writer.release()
        scene.close()
        if state["cameras"] is not None:
            state["cameras"].close()
    return result


def replay_run(
    root: Path,
    output_dir: Path,
    render: bool = True,
    limit: int | None = None,
    **replay_kwargs: Any,
) -> list[ReplayResult]:
    """Replay every Franka episode recorded under `root`. Returns what it measured.

    Stages each episode's own pickup object when the replay is a physics one,
    from the benchmark `root` holds. A physics replay of a scene with nothing in
    it is a slower way to get the kinematic numbers, so this is not an
    embellishment -- it is what makes the flag mean anything.
    """
    results: list[ReplayResult] = []
    staging = bool(replay_kwargs.get("physics")) and "stage" not in replay_kwargs
    for path in find_trajectory_files(root):
        for episode in load_episodes(path):
            if limit is not None and len(results) >= limit:
                return results
            name = f"house_{episode.house}_{episode.group}_{_slug(episode.instruction)}"
            log.info(f"[replay] {name}: {episode.steps} recorded steps")
            kwargs = dict(replay_kwargs)
            if staging:
                # `traj_<i>` counts within the house in the order the episodes
                # ran, which is the order the benchmark lists them in. No
                # instruction to check it against -- `load_episodes` has none --
                # so `episode_staging` is trusted on the index alone here.
                index = episode.group.rsplit("_", 1)[-1]
                kwargs["stage"] = episode_staging(
                    root, episode.house, int(index) if index.isdigit() else 0
                )
            if render:
                results.append(replay_to_video(episode, output_dir / f"{name}.mp4", **kwargs))
            else:
                results.append(replay_episode(episode, **kwargs))
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

    grasped = [r for r in results if r.grasp is not None]
    if grasped:
        picked = sum(1 for r in grasped if r.grasp.success)
        click.secho(
            f"\n{picked}/{len(grasped)} picked up. These are the recorded Franka's own "
            f"actions, so the Franka picked up all {len(grasped)}: the difference is what "
            f"the retargeting costs a grasp that is known to work.",
            fg="green" if picked == len(grasped) else "yellow",
        )

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
