"""
Retarget a MolmoSpaces benchmark episode from its authoring robot onto Stretch 4.

Every episode in `molmospaces-bench-v1`/`-v2` was generated with a Franka Droid
(65k episodes) or an RBY1 (40k), and freezes robot-specific facts into the JSON:
a `robot.init_qpos` keyed by *that* robot's move groups, a `cameras` list of
extrinsics mounted on that robot's bodies, and a `task.robot_base_pose` chosen so
its workspace covered the target.

MolmoSpaces has a hook for exactly this -- `EvalRuntimeParams.robot_override_fn`,
invoked by `JsonEvalTaskSampler.__init__` after it has built the recorded camera
config and before the task sampler is constructed, with the episode spec and the
experiment config both mutable. `stretch_episode_override()` is that hook. The
scene, the object poses, the language instruction and the success criteria are
left completely untouched, so the task being scored is still the benchmark's.

Three things get rewritten:

    cameras          -> Stretch's own head and wrist MJCF cameras
    robot.init_qpos  -> Stretch's five move groups
    robot_base_pose  -> a standoff Stretch can actually reach from

The base pose is the only interesting one; see `retarget_base_pose()`.

This is scene and spawn adaptation, not a translation of the authoring robot's
*actions*: nothing here converts one robot's joint vector into another's. Every
policy path -- the simple_ik expert, a BC checkpoint, a MolmoBot fine-tune --
drives Stretch's own move groups directly and goes through this same override.
"""

import logging
from typing import TYPE_CHECKING

import numpy as np

from examples.machine_learning.molmospaces.stretch.config import (
    RENDER_BUFFER_RESOLUTION,
    Stretch4CameraSystem,
    Stretch4RobotConfig,
)
from molmo_spaces.evaluation.benchmark_schema import EpisodeSpec

if TYPE_CHECKING:
    from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig

log = logging.getLogger(__name__)

# Stretch's tool centre point sits on the base's +x axis and moves along it as
# the telescoping arm extends. Measured by forward kinematics on the compiled
# MJCF at wrist yaw 0: 0.467m fully retracted, 0.987m fully extended.
TCP_MIN_REACH_M = 0.467
TCP_MAX_REACH_M = 0.987

# The annulus the retargeted base aims to leave the target in. Inset from the
# true limits so the arm has travel on both sides for the approach, the grasp and
# the retreat, rather than being pinned against a joint limit.
REACH_BAND_M = (0.55, 0.90)

# Where in that band to place a target that the original base pose left out of
# reach entirely.
PREFERRED_STANDOFF_M = 0.72

# Task classes where the robot is not meant to start within arm's reach of
# anything -- moving the spawn point would change what the episode is testing.
_NAVIGATION_TASK_CLASSES = frozenset({"molmo_spaces.tasks.nav_task.NavToObjTask"})

# Task fields that name a 7-element [x, y, z, qw, qx, qy, qz] pose the arm has to
# get to. An episode with several (pick-and-place has two) is retargeted against
# their centroid, which is what the original base pose was chosen to cover.
_MANIPULATION_TARGET_POSE_FIELDS = (
    "pickup_obj_start_pose",
    "place_receptacle_start_pose",
)


def stretch_home_init_qpos() -> dict[str, list[float]]:
    """Per-move-group start pose to write over the episode's Franka/RBY1 one.

    Stretch starts *stowed*. That is load-bearing, not cosmetic: see the comment
    on `Stretch4RobotConfig.init_qpos`. An unstowed Stretch has its tool 0.57m in
    front of the base, which is inside the reach band the base pose is placed
    at, so it spawns embedded in whatever the target object is sitting on.

    Taken from `Stretch4RobotConfig.init_qpos` so there is a single definition of
    where Stretch starts an episode. The base entry is dropped: the base is
    placed from `task.robot_base_pose`, and leaving a zeroed base pose in
    `init_qpos` would teleport the robot back to the world origin, because
    `JsonEvalTaskSampler.randomize_scene()` applies `init_qpos` group by group.
    """
    init_qpos = {
        k: list(v) for k, v in Stretch4RobotConfig.model_fields["init_qpos"].default.items()
    }
    init_qpos.pop("base", None)
    return init_qpos


def manipulation_target_xy(task: dict) -> np.ndarray | None:
    """Centroid of the xy positions the arm has to reach in this episode.

    Returns None for tasks that declare no reachable target pose -- door opening
    names a body rather than a pose, and navigation has no manipulation target at
    all -- in which case `retarget_base_pose()` leaves the position alone.
    """
    points = [
        np.asarray(task[field][:2], dtype=float)
        for field in _MANIPULATION_TARGET_POSE_FIELDS
        if isinstance(task.get(field), (list, tuple)) and len(task[field]) == 7
    ]
    if not points:
        return None
    return np.mean(points, axis=0)


def retarget_base_pose(task: dict) -> None:
    """Rewrite `task["robot_base_pose"]` for Stretch, in place.

    The rule is to keep as much of the original placement as possible, because
    the episode's author validated it: the base is reachable from the room's
    free space, it is on the side of the target the scene actually allows, and
    nothing occludes it.

    So the *direction* from target to base is always preserved, and the distance
    along that direction is only changed when it has to be:

    - target already inside `REACH_BAND_M`: position untouched.
    - target too close or too far: the base slides along the same ray to
      `PREFERRED_STANDOFF_M`.

    Yaw is always recomputed to point the base's +x axis at the target, since
    that is the axis Stretch's telescoping arm extends along. For a Franka
    episode this is close to a no-op -- its arm also reaches along +x -- but it
    stays correct after the base has been moved, and it fixes up RBY1 episodes
    whose base yaw was chosen for a two-armed torso.

    Navigation episodes and anything with no declared target pose keep their
    position and yaw; only z is normalised, since Stretch's base is on the floor
    while a Franka's is on a plinth. (The z is cosmetic in any case:
    `HoloJointsRobotBaseGroup.pose` only reads x, y and yaw out of the matrix.)
    """
    base_pose = list(task["robot_base_pose"])
    base_xy = np.asarray(base_pose[:2], dtype=float)

    target_xy = None
    if task.get("task_cls") not in _NAVIGATION_TASK_CLASSES:
        target_xy = manipulation_target_xy(task)

    if target_xy is not None:
        offset = base_xy - target_xy
        distance = float(np.linalg.norm(offset))
        if distance < 1e-6:
            # Degenerate: the authored base sits on top of the target. Back off
            # along the base's own +x axis instead of an undefined direction.
            yaw = _yaw_of(base_pose)
            direction = np.array([np.cos(yaw), np.sin(yaw)])
        else:
            direction = offset / distance

        if not REACH_BAND_M[0] <= distance <= REACH_BAND_M[1]:
            log.info(
                f"Retargeting base standoff {distance:.2f}m -> {PREFERRED_STANDOFF_M:.2f}m "
                f"to bring the target into Stretch's reach band {REACH_BAND_M}."
            )
            base_xy = target_xy + direction * PREFERRED_STANDOFF_M

        # +x of the base points from the base back at the target.
        yaw = float(np.arctan2(-direction[1], -direction[0]))
    else:
        yaw = _yaw_of(base_pose)

    task["robot_base_pose"] = [
        float(base_xy[0]),
        float(base_xy[1]),
        0.0,
        float(np.cos(yaw / 2.0)),
        0.0,
        0.0,
        float(np.sin(yaw / 2.0)),
    ]


def _yaw_of(pose7: list[float]) -> float:
    """Yaw about +z of a [x, y, z, qw, qx, qy, qz] pose."""
    qw, qx, qy, qz = pose7[3:7]
    return float(np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))


def stretch_episode_override(
    episode_spec: EpisodeSpec,
    exp_config: "MlSpacesExpConfig",
) -> None:
    """The registered `robot_override_fn`: cameras, start configuration, base pose.

    See the module docstring.
    """
    episode_spec.robot.robot_name = "stretch4"
    episode_spec.robot.init_qpos = stretch_home_init_qpos()

    # The recorded authoring-robot cameras are mounted on bodies
    # ("robot_0/fr3_link0", "robot_0/gripper/base") that do not exist in a
    # Stretch scene, so they are replaced wholesale rather than adjusted.
    #
    # This has to mutate the camera system in place rather than assign a fresh
    # one over `exp_config.camera_config`. `JsonEvalTaskSampler` keeps its own
    # reference to the recorded system in `self._recorded_camera_config` and
    # passes *that* to `camera_manager.setup_cameras()`, while the sensor suite
    # is built from `exp_config.camera_config`. The two are the same object right
    # up until something rebinds the attribute -- at which point the cameras
    # actually created and the camera sensors created to read them stop agreeing.
    # (`cap_robot_eval_override` mutates in place for the same reason.)
    #
    # `img_resolution` is the shared offscreen buffer rather than an image size:
    # what each camera renders at is fixed by `CAMERA_RENDER_SIZE`, so an episode
    # recorded before per-camera sizing carries a buffer too small to hold them.
    # Taking the enclosing size of the two means a replay renders at exactly the
    # sizes datagen did, which is the only way an eval score is comparable with
    # what the policy trained on.
    recorded_resolution = tuple(episode_spec.img_resolution)
    stretch_cameras = Stretch4CameraSystem(
        img_resolution=(
            max(recorded_resolution[0], RENDER_BUFFER_RESOLUTION[0]),
            max(recorded_resolution[1], RENDER_BUFFER_RESOLUTION[1]),
        )
    )
    exp_config.camera_config.cameras = list(stretch_cameras.cameras)
    exp_config.camera_config.img_resolution = stretch_cameras.img_resolution

    retarget_base_pose(episode_spec.task)
