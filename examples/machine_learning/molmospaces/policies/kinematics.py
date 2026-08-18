"""
Reach solving for Stretch 4, on a scratch copy of the robot.

Stretch's manipulator is almost Cartesian, which makes reaching a much smaller
problem than it is for a 7-DOF arm:

    lift        moves the tool straight up
    arm         extends the tool along the base's +x axis
    wrist yaw   swings the tool laterally about +z
    wrist pitch tips the approach axis from horizontal (+x) to straight down (-z)
    wrist roll  spins the fingers about the approach axis

Position and approach direction therefore decouple almost completely: pitch and
roll choose *how* to grasp, and the rest choose *where*. So this module fixes
pitch and roll from the caller's chosen grasp style and solves the remaining
degrees of freedom against a position-only Jacobian.

Which degrees of freedom those are depends on the grasp, and the reason is worth
stating because it constrains everything the scripted policy can do. In a
*horizontal* grasp the tool sticks out ahead of the wrist yaw axis, so yaw gives
real lateral authority and `lift + arm + wrist yaw` is a square, well-conditioned
3x3 system. In a *top-down* grasp the tool hangs directly beneath that axis, the
lever arm collapses, and yaw stops moving the tool sideways at all -- the
Jacobian goes singular in exactly the direction the solve needs. Stretch simply
has no wrist-driven lateral freedom when reaching straight down; a real Stretch
drives its base instead. So top-down solves add the two holonomic base slides,
heavily weighted against so the base only creeps when nothing else will do.

Solving happens on a private, mesh-free copy of the robot (the same trick
`MlSpacesKinematics` uses) so a candidate reach can be evaluated without
disturbing the live simulation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import mujoco
import numpy as np
from mujoco import MjSpec
from scipy.spatial.transform import Rotation as R

from examples.machine_learning.molmospaces.stretch.robot_view import Stretch4RobotView

if TYPE_CHECKING:
    from molmo_spaces.configs.robot_configs import BaseRobotConfig

log = logging.getLogger(__name__)

# Wrist pitch that points the gripper's approach axis straight down, and the one
# that points it horizontally along the base's +x. Verified by forward
# kinematics on the compiled MJCF: pitch 1.57 gives an approach of (0, 0, -1),
# pitch 0 gives (1, 0, 0).
PITCH_TOP_DOWN = np.pi / 2
PITCH_HORIZONTAL = 0.0

# Column of `get_jacobian("gripper", ["base", "lift", "arm", "wrist"])` that each
# named degree of freedom occupies. The base contributes (x, y, theta), lift and
# the telescoping arm one each, and the wrist (yaw, pitch, roll).
_DOF_COLUMN = {
    "base_x": 0,
    "base_y": 1,
    "base_theta": 2,
    "lift": 3,
    "arm": 4,
    "wrist_yaw": 5,
}

# Relative willingness to move each degree of freedom, as diagonal weights in a
# weighted damped-least-squares step. Driving the base to fine-tune a grasp is a
# last resort -- it is slow, and it invalidates the collision-free standoff the
# benchmark episode was authored with -- so the slides are worth two orders of
# magnitude less than the arm.
_DOF_WEIGHT = {
    "base_x": 0.01,
    "base_y": 0.01,
    "base_theta": 0.01,
    "lift": 1.0,
    "arm": 1.0,
    "wrist_yaw": 1.0,
}

HORIZONTAL_REACH_DOFS = ("lift", "arm", "wrist_yaw")
TOP_DOWN_REACH_DOFS = ("lift", "arm", "base_x", "base_y")


class StretchReachSolver:
    """Solves a subset of Stretch's joints to put the tool centre at a world point.

    Base *yaw* is always held: `stretch/episode_overrides.py` already aimed the
    base at the target, and re-aiming mid-reach would swing the whole arm through
    the scene.
    """

    def __init__(self, robot_config: "BaseRobotConfig") -> None:
        spec = MjSpec()
        robot_config.robot_cls.add_robot_to_scene(
            robot_config,
            spec,
            prefix=robot_config.robot_namespace,
            pos=[0.0, 0.0, 0.0],
            quat=[1.0, 0.0, 0.0, 0.0],
            strip_meshes=True,
        )
        self._model = spec.compile()
        self._data = mujoco.MjData(self._model)
        self._view = Stretch4RobotView(self._data, robot_config.robot_namespace)
        self._limits = {
            group: self._view.get_move_group(group).joint_pos_limits
            for group in ("lift", "arm", "wrist")
        }

    @property
    def joint_limits(self) -> dict[str, np.ndarray]:
        """Per-group `(n, 2)` joint limit arrays for lift, arm and wrist."""
        return self._limits

    def forward(self, configuration: dict[str, np.ndarray]) -> np.ndarray:
        """World 4x4 pose of the tool centre for a `{group: joint_pos}` mapping."""
        self._apply(configuration)
        return self._view.get_move_group("gripper").leaf_frame_to_world

    def solve(
        self,
        base_pose: np.ndarray,
        target_position: np.ndarray,
        wrist_pitch: float = PITCH_HORIZONTAL,
        wrist_roll: float = 0.0,
        seed: dict[str, np.ndarray] | None = None,
        dofs: tuple[str, ...] | None = None,
        max_base_travel: float = 0.35,
        tolerance: float = 5e-3,
        max_iterations: int = 80,
        step_damping: float = 1e-5,
    ) -> dict[str, np.ndarray] | None:
        """Joint targets that put the tool centre at `target_position`.

        Args:
            base_pose: 4x4 world pose the base starts at. Yaw is preserved; the
                position may move if `dofs` includes the base slides.
            target_position: world xyz the tool centre should reach.
            wrist_pitch: approach tilt, `PITCH_HORIZONTAL` or `PITCH_TOP_DOWN`.
            wrist_roll: spin of the fingers about the approach axis.
            seed: starting configuration keyed by move group. Seeding from the
                robot's current pose keeps successive solves continuous, which
                matters because a phase-by-phase policy re-solves every step.
            dofs: which degrees of freedom to solve over. Defaults to
                `TOP_DOWN_REACH_DOFS` for a top-down pitch and
                `HORIZONTAL_REACH_DOFS` otherwise -- see the module docstring.
            max_base_travel: cap, in metres, on how far the solve may slide the
                base from `base_pose`.
            tolerance: position error, in metres, that counts as solved.
            max_iterations: give up after this many damped least-squares steps.
            step_damping: Levenberg-Marquardt damping.

        Returns:
            `{"base": (3,), "lift": (1,), "arm": (1,), "wrist": (3,)}`, or None if
            the target could not be reached within `max_iterations`.
        """
        if dofs is None:
            dofs = (
                TOP_DOWN_REACH_DOFS
                if abs(wrist_pitch - PITCH_TOP_DOWN) < 0.3
                else HORIZONTAL_REACH_DOFS
            )
        columns = [_DOF_COLUMN[dof] for dof in dofs]
        weights = np.array([_DOF_WEIGHT[dof] for dof in dofs])

        start_xy = np.asarray(base_pose[:2, 3], dtype=float)
        configuration = self._seed(base_pose, seed, wrist_pitch, wrist_roll)
        target_position = np.asarray(target_position, dtype=float)

        for _ in range(max_iterations):
            self._apply(configuration)
            error = (
                target_position - self._view.get_move_group("gripper").leaf_frame_to_world[:3, 3]
            )
            if np.linalg.norm(error) < tolerance:
                return {key: value.copy() for key, value in configuration.items()}

            jacobian = self._view.get_jacobian("gripper", ["base", "lift", "arm", "wrist"])
            jacobian = jacobian[:3, columns]
            weighted = jacobian * weights
            step = weights * (
                jacobian.T
                @ np.linalg.solve(weighted @ jacobian.T + step_damping * np.eye(3), error)
            )
            configuration = self._integrate(configuration, dofs, step, start_xy, max_base_travel)

        return None

    def _seed(
        self,
        base_pose: np.ndarray,
        seed: dict[str, np.ndarray] | None,
        wrist_pitch: float,
        wrist_roll: float,
    ) -> dict[str, np.ndarray]:
        yaw = float(np.arctan2(base_pose[1, 0], base_pose[0, 0]))
        base = np.array([base_pose[0, 3], base_pose[1, 3], yaw])
        if seed is None:
            lift, arm, wrist_yaw = 0.6, 0.1, 0.0
        else:
            lift = float(np.asarray(seed["lift"]).reshape(-1)[0])
            arm = float(np.asarray(seed["arm"]).reshape(-1)[0])
            wrist_yaw = float(np.asarray(seed["wrist"]).reshape(-1)[0])
        return {
            "base": base,
            "lift": np.array([lift]),
            "arm": np.array([arm]),
            "wrist": np.array([wrist_yaw, wrist_pitch, wrist_roll]),
        }

    def _integrate(
        self,
        configuration: dict[str, np.ndarray],
        dofs: tuple[str, ...],
        step: np.ndarray,
        start_xy: np.ndarray,
        max_base_travel: float,
    ) -> dict[str, np.ndarray]:
        updated = {key: value.copy() for key, value in configuration.items()}
        for dof, delta in zip(dofs, step):
            if dof == "base_x":
                updated["base"][0] += delta
            elif dof == "base_y":
                updated["base"][1] += delta
            elif dof == "base_theta":
                updated["base"][2] += delta
            elif dof == "lift":
                updated["lift"][0] = np.clip(updated["lift"][0] + delta, *self._limits["lift"][0])
            elif dof == "arm":
                updated["arm"][0] = np.clip(updated["arm"][0] + delta, *self._limits["arm"][0])
            elif dof == "wrist_yaw":
                updated["wrist"][0] = np.clip(
                    updated["wrist"][0] + delta, *self._limits["wrist"][0]
                )

        travel = updated["base"][:2] - start_xy
        distance = float(np.linalg.norm(travel))
        if distance > max_base_travel:
            updated["base"][:2] = start_xy + travel * (max_base_travel / distance)
        return updated

    def _apply(self, configuration: dict[str, np.ndarray]) -> None:
        base = configuration["base"]
        pose = np.eye(4)
        pose[:2, 3] = base[:2]
        pose[:3, :3] = R.from_euler("z", base[2]).as_matrix()
        self._view.base.pose = pose
        self._view.get_move_group("lift").joint_pos = configuration["lift"]
        self._view.get_move_group("arm").joint_pos = configuration["arm"]
        self._view.get_move_group("wrist").joint_pos = configuration["wrist"]
        mujoco.mj_kinematics(self._model, self._data)
        mujoco.mj_comPos(self._model, self._data)


def yaw_of_pose(pose: np.ndarray) -> float:
    """Yaw about +z of a 4x4 pose matrix."""
    return float(np.arctan2(pose[1, 0], pose[0, 0]))


def planar_pose(x: float, y: float, yaw: float) -> np.ndarray:
    """A 4x4 pose on the floor plane."""
    pose = np.eye(4)
    pose[0, 3] = x
    pose[1, 3] = y
    pose[:3, :3] = R.from_euler("z", yaw).as_matrix()
    return pose
