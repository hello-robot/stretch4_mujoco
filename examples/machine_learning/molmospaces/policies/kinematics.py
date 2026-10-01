"""
Reach solving and forward kinematics for Stretch 4 using Pinocchio (stretch4_kinematics).
"""

from __future__ import annotations

import contextlib
import io
import logging
import warnings
from functools import lru_cache
from typing import TYPE_CHECKING

import mujoco
import numpy as np
import pinocchio as pin
from mujoco import MjSpec
from scipy.spatial.transform import Rotation as R
from stretch4_kinematics.kinematic_models.base_kinematic_models import StretchKinematics
from stretch4_kinematics.state.joint_positions import StretchJointPositions

from examples.machine_learning.molmospaces.stretch.robot_view import (
    Stretch4RobotView,
    commandable_limits,
)

if TYPE_CHECKING:
    from molmo_spaces.configs.robot_configs import BaseRobotConfig

log = logging.getLogger(__name__)

PITCH_TOP_DOWN = np.pi / 2
PITCH_HORIZONTAL = 0.0

TCP_FRAME = "grasp_center_link"
APPROACH_YAW_SPREAD_RAD = np.pi / 3
APPROACH_YAW_SAMPLES = 9

# Transformation from MolmoSpaces grasp frame (+z approach, +-y fingers)
# to Stretch TCP tool frame (+x approach, +-y fingers).
GRASP_LIBRARY_TO_TCP = R.from_euler("y", -np.pi / 2).as_matrix()


def grasp_orientation(approach_yaw: float, wrist_pitch: float, wrist_roll: float) -> np.ndarray:
    """3x3 world rotation matrix from ZYX Euler angles: Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    return (
        R.from_euler("z", approach_yaw).as_matrix()
        @ R.from_euler("y", wrist_pitch).as_matrix()
        @ R.from_euler("x", wrist_roll).as_matrix()
    )


def tcp_orientation_from_grasp(grasp_pose: np.ndarray) -> tuple[float, float, float]:
    """Convert a 4x4 grasp library pose into (approach_yaw, wrist_pitch, wrist_roll)."""
    rotation = np.asarray(grasp_pose, dtype=float)[:3, :3] @ GRASP_LIBRARY_TO_TCP
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Gimbal lock detected")
        yaw, pitch, roll = R.from_matrix(rotation).as_euler("ZYX")
    return float(yaw), float(pitch), float(roll)


def yaw_of_pose(pose: np.ndarray) -> float:
    """Extract yaw angle about +z from a 4x4 pose matrix."""
    return float(np.arctan2(pose[1, 0], pose[0, 0]))


def planar_pose(x: float, y: float, yaw: float) -> np.ndarray:
    """Construct a 4x4 SE(2) ground plane pose matrix."""
    pose = np.eye(4)
    pose[0, 3] = x
    pose[1, 3] = y
    pose[:3, :3] = R.from_euler("z", yaw).as_matrix()
    return pose


_MODEL_CALIBRATION_CACHE: dict[tuple[str, str], tuple[np.ndarray, dict[str, np.ndarray]]] = {}


def _model_calibration(
    robot_config: "BaseRobotConfig",
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """The two things this module needs out of the compiled MJCF.

    Returns:
        The translation offset from the MuJoCo holonomic base origin to the URDF
        base origin, in base coordinates, and the commandable joint limits per
        move group (see `commandable_limits`).

    Compiling the robot on its own is what makes both of these readable without
    a scene: they are properties of the robot description, so a throwaway spec
    with the meshes stripped is enough, and the result is cached per description.
    """
    key = (str(robot_config.robot_dir), robot_config.robot_namespace)
    if key in _MODEL_CALIBRATION_CACHE:
        return _MODEL_CALIBRATION_CACHE[key]

    namespace = robot_config.robot_namespace
    spec = MjSpec()
    robot_config.robot_cls.add_robot_to_scene(
        robot_config,
        spec,
        prefix=namespace,
        pos=[0.0, 0.0, 0.0],
        quat=[1.0, 0.0, 0.0, 0.0],
        strip_meshes=True,
    )
    model = spec.compile()
    data = mujoco.MjData(model)
    view = Stretch4RobotView(data, namespace)

    ref = {"lift": np.array([0.6]), "arm": np.array([0.26]), "wrist": np.zeros(3)}
    for group, val in ref.items():
        view.get_move_group(group).joint_pos = val
    mujoco.mj_kinematics(model, data)
    mujoco_tcp = np.asarray(view.get_move_group("gripper").leaf_frame_to_world[:3, 3], dtype=float)

    pin_tcp = _kinematics().forward(
        StretchJointPositions(
            lift=float(ref["lift"][0]),
            arm=float(ref["arm"][0]),
            wrist_yaw=0.0,
            wrist_pitch=0.0,
            wrist_roll=0.0,
        ),
        TCP_FRAME,
    ).translation

    offset = np.asarray(pin_tcp, dtype=float) - mujoco_tcp
    limits = {
        group: commandable_limits(view.get_move_group(group))
        for group in Stretch4RobotView.MOVE_GROUP_ORDER
    }
    _MODEL_CALIBRATION_CACHE[key] = (offset, limits)
    return offset, limits


@lru_cache(maxsize=1)
def _kinematics() -> StretchKinematics:
    return StretchKinematics()


def _approach_yaw_candidates(nominal: float, spread: float, samples: int) -> list[float]:
    """Generate approach yaw candidates around nominal heading."""
    if spread <= 0.0 or samples <= 1:
        return [nominal]
    offsets = np.linspace(0.0, spread, samples // 2 + 1)[1:]
    candidates = [nominal]
    for offset in offsets:
        candidates += [nominal + offset, nominal - offset]
    return candidates


class StretchReachSolver:
    """Solves reach kinematics for Stretch 4 using Pinocchio."""

    # Which pinocchio configuration index each MuJoCo move group's joints land
    # on, for `model_ik`'s joint order: mobile_base_rotation, lift_joint,
    # arm_l4_joint, wrist_yaw, wrist_pitch, wrist_roll.
    _PIN_CONFIG_INDICES = {"lift": (1,), "arm": (2,), "wrist": (3, 4, 5)}

    def __init__(self, robot_config: "BaseRobotConfig") -> None:
        self._kinematics = _kinematics()
        self._namespace = robot_config.robot_namespace
        self._offset, model_limits = _model_calibration(robot_config)

        # The IK's limits are the MJCF's, so a solution is by construction
        # something `JointPosController` can command without clipping -- and a
        # solution the controller *would* clip is one the robot parks short of
        # while the policy waits for an arrival that cannot come.
        #
        # Two things were wrong with writing them here instead. The mast was
        # widened to 1.23m, which is a number no description contains: both the
        # URDF and the MJCF say 1.2, so every solve that saturated the lift
        # returned 30mm of travel that does not exist. The wrist pitch was
        # widened to -1.571, which is right -- the URDF really does stop 25
        # degrees short of the MJCF's full 90 up -- but only until one of the two
        # descriptions changes, and then it is another 1.23.
        model = self._kinematics.model_ik
        self._limits = {}
        for group, indices in self._PIN_CONFIG_INDICES.items():
            limits = np.asarray(model_limits[group], dtype=float)
            assert len(limits) == len(indices), (
                f"move group {group!r} reports {len(limits)} commandable joints, "
                f"but the IK model has {len(indices)}"
            )
            for row, index in enumerate(indices):
                model.lowerPositionLimit[index] = limits[row, 0]
                model.upperPositionLimit[index] = limits[row, 1]
            self._limits[group] = limits

    @property
    def joint_limits(self) -> dict[str, np.ndarray]:
        """Commandable `(n, 2)` limits per move group, as compiled into the MJCF."""
        return self._limits

    def forward(self, configuration: dict[str, np.ndarray]) -> np.ndarray:
        """Compute 4x4 tool center pose in world coordinates from a joint dictionary."""
        base = np.asarray(configuration["base"], dtype=float).reshape(-1)
        wrist = np.asarray(configuration["wrist"], dtype=float).reshape(-1)
        pose_se3 = self._kinematics.forward(
            StretchJointPositions(
                base_x=float(base[0]),
                base_y=float(base[1]),
                base_theta=float(base[2]),
                lift=float(np.asarray(configuration["lift"]).reshape(-1)[0]),
                arm=float(np.asarray(configuration["arm"]).reshape(-1)[0]),
                wrist_yaw=float(wrist[0]),
                wrist_pitch=float(wrist[1]),
                wrist_roll=float(wrist[2]),
            ),
            TCP_FRAME,
        )
        pose = np.eye(4)
        pose[:3, :3] = pose_se3.rotation
        pose[:3, 3] = pose_se3.translation - R.from_euler("z", base[2]).as_matrix() @ self._offset
        return pose

    def solve(
        self,
        base_pose: np.ndarray,
        target_position: np.ndarray,
        wrist_pitch: float = PITCH_HORIZONTAL,
        wrist_roll: float = 0.0,
        seed: dict[str, np.ndarray] | None = None,
        approach_yaw: float | None = None,
        yaw_spread: float = APPROACH_YAW_SPREAD_RAD,
        yaw_samples: int = APPROACH_YAW_SAMPLES,
        tolerance: float = 5e-3,
        max_iterations: int = 200,
        step_damping: float = 1e-6,
    ) -> dict[str, np.ndarray] | None:
        """Solve IK to place the tool center at target_position with given orientation constraints.

        Args:
            base_pose: Current 4x4 world pose of the robot base.
            target_position: Desired tool center position [x, y, z] in world frame.
            wrist_pitch: Tool pitch angle.
            wrist_roll: Tool roll angle.
            seed: Initial joint configuration for continuity.
            approach_yaw: World heading for approach, or None to use nominal bearing.
            yaw_spread: Angular search spread around approach heading.
            yaw_samples: Number of heading samples to try.
            tolerance: Position convergence threshold in meters.
            max_iterations: Maximum CLIK solver iterations.
            step_damping: Damping factor for pseudo-inverse.

        Returns:
            Dictionary {"base": [x, y, theta], "lift": [lift], "arm": [arm], "wrist": [yaw, pitch, roll]},
            or None if no valid solution was found.
        """
        base_pose = np.asarray(base_pose, dtype=float)
        base_xy = base_pose[:2, 3]
        base_yaw = yaw_of_pose(base_pose)
        target_position = np.asarray(target_position, dtype=float).reshape(3)

        to_target = target_position - np.array([base_xy[0], base_xy[1], 0.0])
        local_target_nominal = (
            R.from_euler("z", -base_yaw).as_matrix() @ to_target + self._offset
        )

        nominal_yaw = (
            float(np.arctan2(local_target_nominal[1], local_target_nominal[0]))
            if approach_yaw is None
            else float(approach_yaw) - base_yaw
        )

        for candidate_yaw in _approach_yaw_candidates(nominal_yaw, yaw_spread, yaw_samples):
            solution = self._solve_at_yaw(
                to_target,
                base_xy,
                base_yaw,
                candidate_yaw,
                wrist_pitch,
                wrist_roll,
                seed,
                max_iterations,
                step_damping,
                tolerance,
            )
            if solution is not None:
                achieved = self.forward(solution)[:3, 3]
                if float(np.linalg.norm(achieved - target_position)) <= tolerance:
                    return solution

        # If unreachable from current base position, solve with base translation towards target
        offset_2d = target_position[:2] - base_xy
        dist_2d = float(np.linalg.norm(offset_2d))
        if dist_2d > 1e-4:
            dir_2d = offset_2d / dist_2d
            preferred_standoff = 0.70
            new_base_xy = target_position[:2] - dir_2d * preferred_standoff
            new_base_yaw = float(np.arctan2(dir_2d[1], dir_2d[0]))
            new_to_target = target_position - np.array([new_base_xy[0], new_base_xy[1], 0.0])
            new_nominal_yaw = 0.0 if approach_yaw is None else float(approach_yaw) - new_base_yaw

            for candidate_yaw in _approach_yaw_candidates(new_nominal_yaw, yaw_spread, yaw_samples):
                solution = self._solve_at_yaw(
                    new_to_target,
                    new_base_xy,
                    new_base_yaw,
                    candidate_yaw,
                    wrist_pitch,
                    wrist_roll,
                    seed,
                    max_iterations,
                    step_damping,
                    tolerance,
                )
                if solution is not None:
                    achieved = self.forward(solution)[:3, 3]
                    if float(np.linalg.norm(achieved - target_position)) <= tolerance:
                        return solution

        return None

    def _solve_at_yaw(
        self,
        to_target: np.ndarray,
        base_xy: np.ndarray,
        base_yaw: float,
        candidate_yaw: float,
        wrist_pitch: float,
        wrist_roll: float,
        seed: dict[str, np.ndarray] | None,
        max_iterations: int,
        step_damping: float,
        tolerance: float,
    ) -> dict[str, np.ndarray] | None:
        rotate_into_world = R.from_euler("z", -base_yaw).as_matrix() @ to_target
        base_rotation = 0.0

        for _ in range(3):
            local_target = (
                rotate_into_world + R.from_euler("z", base_rotation).as_matrix() @ self._offset
            )
            solution = self._solve_local(
                local_target,
                grasp_orientation(candidate_yaw, wrist_pitch, wrist_roll),
                seed,
                max_iterations,
                step_damping,
                tolerance,
            )
            if solution is None:
                return None
            if abs(solution.base_theta - base_rotation) < 1e-6:
                break
            base_rotation = solution.base_theta

        return {
            "base": np.array([base_xy[0], base_xy[1], base_yaw + solution.base_theta]),
            "lift": np.array([solution.lift]),
            "arm": np.array([solution.arm]),
            "wrist": np.array([solution.wrist_yaw, solution.wrist_pitch, solution.wrist_roll]),
        }

    def _solve_local(
        self,
        local_target: np.ndarray,
        local_rotation: np.ndarray,
        seed: dict[str, np.ndarray] | None,
        max_iterations: int,
        step_damping: float,
        tolerance: float,
    ) -> StretchJointPositions | None:
        guess = StretchJointPositions(
            base_x=0.0,
            base_y=0.0,
            base_theta=0.0,
            lift=0.6 if seed is None else float(np.asarray(seed["lift"]).reshape(-1)[0]),
            arm=0.1 if seed is None else float(np.asarray(seed["arm"]).reshape(-1)[0]),
            wrist_yaw=0.0 if seed is None else float(np.asarray(seed["wrist"]).reshape(-1)[0]),
            wrist_pitch=0.0 if seed is None else float(np.asarray(seed["wrist"]).reshape(-1)[1]),
            wrist_roll=0.0 if seed is None else float(np.asarray(seed["wrist"]).reshape(-1)[2]),
        )
        target_pose = pin.SE3(np.asarray(local_rotation, dtype=float), np.asarray(local_target))

        try:
            with contextlib.redirect_stdout(io.StringIO()):
                return self._kinematics.inverse_6dof_local(
                    target_frame=TCP_FRAME,
                    target_pose=target_pose,
                    q_guess=guess,
                    max_iter=max_iterations,
                    eps=min(tolerance, 1e-3),
                    damp=step_damping,
                )
        except (ValueError, RuntimeError) as failure:
            log.debug(f"[stretch-kinematics] IK did not converge: {failure}")
            return None
