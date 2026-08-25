"""
The authoring arm's frame, carried from episode load to policy reset.

Retargeting an episode destroys the two facts a Franka-space policy needs:
`stretch_episode_override()` overwrites `robot.init_qpos` with Stretch's stowed
pose and `retarget_base_pose()` moves `task.robot_base_pose`. Both of those are
correct -- Stretch cannot spawn in a Franka's configuration or at a Franka's
standoff -- but they throw away the frame the VLA's numbers are expressed in.

This module keeps them. `record()` is called by the override, on the way past,
and `current()` is read by the policy in `reset()`. Nothing in between has to
know about it, which is why it is a module-level record rather than another
field threaded through the exp config.

Why a module global is the right shape here, rather than a smell:

- The evaluation runner builds a fresh `JsonEvalTaskSampler` per episode
  (`JsonEvalRunner.should_close_episode_task_sampler()` returns True), and the
  override runs inside that constructor. So exactly one episode's frame is live
  at a time in a worker process.
- `task.reset()` calls the registered policy's `reset()`
  (`BaseMujocoTask.reset`), and the task is sampled *after* the sampler is
  built, so the record is always written before it is read.
- Rollout workers are separate processes, so there is no cross-episode sharing
  to race on. `MlSpacesExpConfig` offers no per-episode channel from the
  override to the policy, and the alternative -- stuffing extra keys into
  `episode_spec.task` -- feeds a dict that MolmoSpaces uses to build a task
  config, where an unrecognised key is a validation error waiting to happen.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation as R

log = logging.getLogger(__name__)


@dataclass
class FrankaEpisodeFrame:
    """Where the authoring arm stood, and how it was holding itself."""

    base_pose: np.ndarray
    """
    4x4 world pose of the authoring robot's `fr3_link0`, before retargeting.

    MolmoSpaces attaches a Franka to the scene by its `fr3_link0` body at the
    episode's `robot_base_pose`, so this is that pose read straight off the
    episode JSON -- see `franka_arm.ROOT_BODY`. Composing a VLA-space tool pose
    with it gives a *world* pose, and world is the one frame both robots agree
    on: the object the episode is about is in the same place for either of them.
    """

    init_qpos: np.ndarray
    """
    The 7-joint configuration the episode starts the authoring arm in.

    `episode_spec.robot.init_qpos["arm"]`. Stretch cannot adopt it directly, but
    its *tool pose* is where the VLA expects to be looking from on step one, so
    the remapper drives Stretch there before handing over.
    """

    gripper_closedness: float = 0.0
    """Where the authoring arm's gripper starts, on the VLA's [0, 1] scale."""

    stretch_base_pose: np.ndarray | None = None
    """
    4x4 world pose Stretch's base was retargeted *to*, for reference and logging.

    Not used by the remapping itself -- the live base pose is read from the robot
    view, which is authoritative once the episode is running -- but it is what
    makes a retarget's standoff change inspectable after the fact.
    """

    robot_name: str = "franka_droid"
    """The authoring robot, as the episode spec named it."""

    metadata: dict = field(default_factory=dict)
    """Anything worth logging alongside a retarget: task class, standoffs, etc."""

    def tool_pose_to_world(self, tool_pose: np.ndarray) -> np.ndarray:
        """A tool pose in the authoring arm's base frame -> a world pose."""
        return self.base_pose @ np.asarray(tool_pose, dtype=float)

    def tool_pose_from_world(self, world_pose: np.ndarray) -> np.ndarray:
        """A world tool pose -> the authoring arm's base frame. Inverse of above."""
        return _invert_pose(self.base_pose) @ np.asarray(world_pose, dtype=float)


_current: FrankaEpisodeFrame | None = None


def record(frame: FrankaEpisodeFrame) -> None:
    """Publish the frame for the episode being loaded."""
    global _current
    _current = frame


def current() -> FrankaEpisodeFrame | None:
    """The frame recorded for the episode being run, or None if there is none.

    None is a real, reachable answer, not a bug: the remapping policy also runs
    against non-retargeted setups -- `live_policy.py`, procedurally sampled
    scenes from `finetuning/datagen_configs.py` -- where there is no authoring
    Franka to have a frame. Callers fall back to
    `default_frame_for(robot_view)`.
    """
    return _current


def clear() -> None:
    """Forget the recorded frame. Called by tests; unnecessary in a worker."""
    global _current
    _current = None


MAST_MOUNT_FORWARD_M = 0.20
MAST_MOUNT_HEIGHT_M = 0.20
"""
Where on Stretch to hang the virtual Franka's shoulder: 0.20m forward of the
base axis and 0.20m up.

Chosen by *workspace overlap*, not by eye. Sampling 360 tool poses across the
lift, the telescoping arm, and the wrist yaw and pitch a manipulation policy
uses, and asking the Franka's IK to reach each one, this pair leaves 69% of
Stretch's tool workspace reachable by the virtual arm at a mean position error
of 38mm -- the best of a grid over forward offsets 0.00..0.24m and heights
0.15..0.45m. The sweep is a dozen lines over `StretchPoseSolver.forward` and
`FrankaArm.inverse`; `lerobot_export.ExportMetadata.mean_shadow_ik_error_m`
reports the same residual per dataset, which is the number to watch if the
robot or the wrist ever changes.

The 31% that stays out of reach is not a tuning failure, it is the shape of the
two robots: Stretch's tool sweeps a radius from 0.12m to 1.00m of its base axis
and 1.3m of height, and no fixed-shoulder 7-DOF arm covers that. Which is
exactly why this constant only matters where there is no real Franka frame to
use -- and why `lerobot_export.py` reports the residual per dataset rather than
assuming it away.
"""


def default_frame_for(base_pose: np.ndarray) -> FrankaEpisodeFrame:
    """The frame to use when no episode recorded one: a virtual Franka on Stretch itself.

    Mounts the virtual arm's shoulder on Stretch's own base, offset by
    `MAST_MOUNT_FORWARD_M` along the base's +x and `MAST_MOUNT_HEIGHT_M` up. It
    exists so a Franka-space policy is runnable, and Stretch trajectories are
    encodable, where there is no authoring Franka to borrow a frame from --
    `live_policy.py`, a procedurally sampled scene from
    `finetuning/datagen_configs.py`, `finetuning/lerobot_export.py`. A
    retargeted benchmark episode always has the real thing and never comes here.
    """
    pose = np.asarray(base_pose, dtype=float).copy()
    pose[:3, 3] = pose[:3, 3] + pose[:3, 0] * MAST_MOUNT_FORWARD_M
    pose[2, 3] += MAST_MOUNT_HEIGHT_M
    return FrankaEpisodeFrame(
        base_pose=pose,
        init_qpos=np.array([0.0, -0.7853, 0.0, -2.35619, 0.0, 1.57079, 0.0]),
        metadata={"source": "default_mast_mount"},
    )


def _invert_pose(pose: np.ndarray) -> np.ndarray:
    inverse = np.eye(4)
    inverse[:3, :3] = pose[:3, :3].T
    inverse[:3, 3] = -pose[:3, :3].T @ pose[:3, 3]
    return inverse


def pose_from_position_quaternion(pose7: list[float] | np.ndarray) -> np.ndarray:
    """`[x, y, z, qw, qx, qy, qz]` -> a 4x4 pose. MolmoSpaces' pose convention."""
    pose7 = np.asarray(pose7, dtype=float).reshape(-1)
    pose = np.eye(4)
    pose[:3, 3] = pose7[:3]
    pose[:3, :3] = R.from_quat(pose7[3:7], scalar_first=True).as_matrix()
    return pose
