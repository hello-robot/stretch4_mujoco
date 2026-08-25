"""
The Franka Droid arm, as a pure kinematics object.

A Franka-space VLA does not talk about tool poses. It reads seven joint angles
plus a gripper scalar and writes seven joint angles plus a gripper scalar (see
`molmo_spaces/policy/learned_policy/pi_policy.py` and `dreamzero_policy.py`,
which are the reference clients for that interface). Stretch has no seven-joint
arm to put those numbers in, so before anything can be retargeted the joint
vector has to be turned back into the thing it was a parameterisation *of*: the
pose of the gripper.

That is all this module does. It compiles the same `franka_droid/model.xml`
MolmoSpaces authored the episodes with -- so the link lengths, the Robotiq
mounting and the `gripper/grasp_site` tool point are the benchmark's own, not a
re-derived DH table -- and exposes forward and inverse kinematics on it:

    qpos (7,)          -> 4x4 tool pose in the `fr3_link0` frame     `forward()`
    4x4 tool pose      -> qpos (7,)                                  `inverse()`

`forward()` is what turns a VLA action into a pose target. `inverse()` is what
turns Stretch's *actual* tool pose back into the seven-number proprioception the
VLA expects to be fed, which is what keeps the model's input distribution close
to what it was trained on even though the arm underneath is a different robot.

The model is kinematics-only: nothing here steps physics or touches the live
simulation, and the compiled model is cached for the process.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np

log = logging.getLogger(__name__)

TOOL_SITE = "gripper/grasp_site"
"""The Franka Droid's tool centre point: the site between the Robotiq fingers."""

ROOT_BODY = "fr3_link0"
"""
The body MolmoSpaces attaches at an episode's `robot_base_pose`.

`Robot.add_robot_to_scene()` attaches `robot_model_root_name()` -- which
`FrankaRobot` defines as `fr3_link0` -- directly at the requested pose, so an
episode's `robot_base_pose` *is* the pose of this body, with no plinth or
pedestal transform in between. That is what makes `episode_frame.py` able to
reconstruct the authoring arm's frame from the episode JSON alone.
"""

HOME_QPOS = np.array([0.0, -0.7853, 0.0, -2.35619, 0.0, 1.57079, 0.0])
"""
`FrankaRobotConfig.init_qpos["arm"]`, duplicated as a plain array.

Used as the null-space attractor and the seed of last resort for `inverse()`.
Hard-coded rather than imported so this module stays a kinematics utility that
does not depend on MolmoSpaces' config tree.
"""

GRIPPER_QPOS_CLOSED = 0.824033
"""
Driver-joint angle the Franka-space VLA clients treat as fully closed.

Taken from the normalisation both `pi_policy.py` and `dreamzero_policy.py` apply
(`qpos["gripper"][0] / 0.824033`), not from the joint's MJCF range of
`[0, 0.9]` -- the models were trained against the clients' scale, so that is the
scale their gripper channel means.
"""

GRIPPER_APERTURE_RANGE_M = (0.0, 0.087)
"""
Fingertip separation at fully closed and fully open, in metres.

`RobotIQGripperGroup.inter_finger_dist_range` in
`molmo_spaces/robots/robot_views/franka_droid_view.py`.
"""


def franka_model_path() -> Path:
    """Where MolmoSpaces put the Franka Droid MJCF for this installation.

    MolmoSpaces hashes its install path into the asset directory, so this is not
    a fixed location. `MLSPACES_ASSETS_DIR` overrides it.
    """
    from molmo_spaces.molmo_spaces_constants import ASSETS_DIR

    return Path(ASSETS_DIR) / "robots" / "franka_droid" / "model.xml"


@lru_cache(maxsize=1)
def _compiled_model() -> mujoco.MjModel:
    """The Franka Droid model, compiled once per process.

    Compiling with meshes takes ~0.15s and gives the exact collision-free
    geometry of the authored robot, so there is nothing to gain from stripping
    them the way `policies/kinematics.py` does for Stretch (which pays that cost
    per solver instance because it attaches the robot to a scene).
    """
    path = franka_model_path()
    if not path.exists():
        raise FileNotFoundError(
            f"No Franka Droid MJCF at {path}. The franka_droid robot assets are part of "
            "the MolmoSpaces resource bundle; they are downloaded on first use of a "
            "Franka config, or with `python -m molmo_spaces.utils.download_resources`."
        )
    return mujoco.MjModel.from_xml_path(str(path))


@dataclass
class FrankaIkSolution:
    """What `FrankaArm.inverse()` managed, and how close it got."""

    qpos: np.ndarray
    """The 7-joint configuration. Always populated; see `converged`."""

    position_error: float
    """Metres between the solved tool position and the requested one."""

    rotation_error: float
    """Radians between the solved tool orientation and the requested one."""

    converged: bool
    """Whether both residuals met the tolerances the solve was asked for."""


class FrankaArm:
    """Forward and inverse kinematics for the Franka Droid arm.

    One instance owns one `MjData`, so it is single-threaded but cheap to make:
    the compiled model behind it is shared.
    """

    N_JOINTS = 7

    def __init__(self) -> None:
        self._model = _compiled_model()
        self._data = mujoco.MjData(self._model)
        self._qpos_adr = np.array(
            [self._model.joint(f"fr3_joint{i + 1}").qposadr[0] for i in range(self.N_JOINTS)]
        )
        self._dof_adr = np.array(
            [self._model.joint(f"fr3_joint{i + 1}").dofadr[0] for i in range(self.N_JOINTS)]
        )
        self._site_id = self._model.site(TOOL_SITE).id
        self._root_body_id = self._model.body(ROOT_BODY).id
        self.joint_limits = np.array(
            [self._model.joint(f"fr3_joint{i + 1}").range for i in range(self.N_JOINTS)]
        )

    # =========================================================================
    # Kinematics
    # =========================================================================

    def forward(self, qpos: np.ndarray) -> np.ndarray:
        """4x4 tool pose in the `fr3_link0` frame for a 7-joint configuration.

        Joint values outside the model's limits are clipped rather than
        rejected: a VLA is free to emit an unreachable target, and clipping is
        what the `JointPosController` driving the real Franka would do with it.
        """
        self._apply(self.clip(qpos))
        return self._tool_pose()

    def inverse(
        self,
        target_pose: np.ndarray,
        seed: np.ndarray | None = None,
        position_tolerance: float = 2e-3,
        rotation_tolerance: float = 1e-2,
        max_iterations: int = 150,
        step_damping: float = 1e-4,
        null_space_gain: float = 0.05,
        rotation_weight: float = 0.3,
    ) -> "FrankaIkSolution":
        """The 7-joint configuration whose tool sits closest to `target_pose`.

        Damped least squares on the full 6-D site Jacobian. The arm is
        redundant, so the seven-dimensional step is biased in the null space
        toward `HOME_QPOS`; without that the solve drifts along the arm's
        self-motion manifold and returns elbow-up/elbow-down configurations
        arbitrarily between calls, which reads to the VLA as proprioception
        teleporting.

        Always returns a configuration -- the best iterate seen, not None. The
        caller is feeding a language model a proprioception vector, so the
        useful answer to "this pose is not quite reachable" is the nearest arm
        pose plus its residual, not a hole in the observation stream.
        `FrankaIkSolution.converged` says whether the residual met the
        tolerances.

        Args:
            target_pose: 4x4 tool pose in the `fr3_link0` frame.
            seed: configuration to start from. Continuity matters more than
                accuracy here -- see the null-space note above -- so callers
                should pass their previous solution.
            position_tolerance: metres of tool position error that counts as
                solved.
            rotation_tolerance: radians of tool orientation error that counts as
                solved.
            max_iterations: give up after this many steps.
            step_damping: Levenberg-Marquardt damping.
            null_space_gain: how hard to pull toward `HOME_QPOS` in the null
                space, per iteration.
            rotation_weight: how much of the least-squares step goes to
                orientation. Below 1 because position is the part that decides
                whether the arm is near the object: an orientation the Franka
                cannot hold at that point should bend the wrist as far as it
                goes and stop, not walk the tool away from the target to get
                there.
        """
        best = self._descend(
            target_pose,
            self.clip(HOME_QPOS if seed is None else seed),
            position_tolerance,
            rotation_tolerance,
            max_iterations,
            step_damping,
            null_space_gain,
            rotation_weight,
        )
        if best.converged or seed is None:
            return best

        # A warm start is the right first guess -- it keeps the elbow where it
        # was -- but damped least squares only walks downhill, so a seed on the
        # wrong side of the arm's self-motion manifold sits in a local minimum
        # and stays there. Retrying from home costs one more descent and, on
        # recorded Stretch trajectories, takes the mean residual from 91mm to
        # single millimetres: without it a handful of stuck steps dominate the
        # average and look exactly like a mis-calibrated mount.
        from_home = self._descend(
            target_pose,
            HOME_QPOS.copy(),
            position_tolerance,
            rotation_tolerance,
            max_iterations,
            step_damping,
            null_space_gain,
            rotation_weight,
        )
        return from_home if from_home.position_error < best.position_error else best

    def _descend(
        self,
        target_pose: np.ndarray,
        qpos: np.ndarray,
        position_tolerance: float,
        rotation_tolerance: float,
        max_iterations: int,
        step_damping: float,
        null_space_gain: float,
        rotation_weight: float,
    ) -> "FrankaIkSolution":
        """One damped least-squares descent from a single starting configuration."""
        target_position = np.asarray(target_pose)[:3, 3]
        target_rotation = np.asarray(target_pose)[:3, :3]

        best = FrankaIkSolution(qpos, np.inf, np.inf, False)
        for _ in range(max_iterations):
            self._apply(qpos)
            current = self._tool_pose()
            position_error = target_position - current[:3, 3]
            rotation_error = _rotation_error(current[:3, :3], target_rotation)
            position_residual = float(np.linalg.norm(position_error))
            rotation_residual = float(np.linalg.norm(rotation_error))

            if position_residual + rotation_weight * rotation_residual < (
                best.position_error + rotation_weight * best.rotation_error
            ):
                best = FrankaIkSolution(
                    qpos.copy(),
                    position_residual,
                    rotation_residual,
                    position_residual < position_tolerance
                    and rotation_residual < rotation_tolerance,
                )
            if best.converged:
                return best

            jacobian = self._tool_jacobian()
            error = np.concatenate([position_error, rotation_weight * rotation_error])
            gram = jacobian @ jacobian.T + step_damping * np.eye(6)
            step = jacobian.T @ np.linalg.solve(gram, error)

            # Null-space pull toward home, projected so it cannot fight the task.
            bias = null_space_gain * (HOME_QPOS - qpos)
            projector = np.eye(self.N_JOINTS) - jacobian.T @ np.linalg.solve(gram, jacobian)
            qpos = self.clip(qpos + step + projector @ bias)

        return best

    def clip(self, qpos: np.ndarray) -> np.ndarray:
        """`qpos` brought inside the model's joint limits."""
        return np.clip(
            np.asarray(qpos, dtype=float).reshape(-1)[: self.N_JOINTS],
            self.joint_limits[:, 0],
            self.joint_limits[:, 1],
        )

    # =========================================================================
    # Gripper
    # =========================================================================

    @staticmethod
    def gripper_closedness_from_qpos(driver_joint_qpos: float) -> float:
        """Driver-joint angle -> the [0, 1] "closedness" a VLA is fed.

        0 is fully open, 1 fully closed, matching the sign convention of
        `observation/gripper_position` in the reference clients.
        """
        return float(np.clip(driver_joint_qpos / GRIPPER_QPOS_CLOSED, 0.0, 1.0))

    @staticmethod
    def gripper_qpos_from_closedness(closedness: float) -> float:
        """Inverse of `gripper_closedness_from_qpos`."""
        return float(np.clip(closedness, 0.0, 1.0)) * GRIPPER_QPOS_CLOSED

    @staticmethod
    def gripper_aperture_m(closedness: float) -> float:
        """Fingertip separation, in metres, for a [0, 1] closedness."""
        low, high = GRIPPER_APERTURE_RANGE_M
        return float(high + (low - high) * np.clip(closedness, 0.0, 1.0))

    # =========================================================================
    # Internals
    # =========================================================================

    def _apply(self, qpos: np.ndarray) -> None:
        self._data.qpos[self._qpos_adr] = qpos
        mujoco.mj_kinematics(self._model, self._data)
        mujoco.mj_comPos(self._model, self._data)

    def _tool_pose(self) -> np.ndarray:
        """Tool pose relative to `fr3_link0`, which is where the model's root is.

        The standalone model puts `fr3_link0` at the world origin with identity
        orientation, so this is a straight read rather than a frame change --
        but it is written as a frame change anyway, because the same class is
        useful against a model whose root has been moved.
        """
        root_position = self._data.xpos[self._root_body_id]
        root_rotation = self._data.xmat[self._root_body_id].reshape(3, 3)
        pose = np.eye(4)
        pose[:3, :3] = root_rotation.T @ self._data.site_xmat[self._site_id].reshape(3, 3)
        pose[:3, 3] = root_rotation.T @ (self._data.site_xpos[self._site_id] - root_position)
        return pose

    def _tool_jacobian(self) -> np.ndarray:
        """6x7 Jacobian of the tool site with respect to the seven arm joints."""
        position_jacobian = np.zeros((3, self._model.nv))
        rotation_jacobian = np.zeros((3, self._model.nv))
        mujoco.mj_jacSite(
            self._model, self._data, position_jacobian, rotation_jacobian, self._site_id
        )
        return np.vstack([position_jacobian[:, self._dof_adr], rotation_jacobian[:, self._dof_adr]])


def _rotation_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rotation vector that takes `current` onto `target`, in world axes.

    The world-frame convention matters: `mj_jacSite`'s rotational rows are the
    site's angular velocity in world axes, so the error fed alongside them has
    to be expressed the same way for the least-squares step to be consistent.
    """
    relative = target @ current.T
    angle = np.arccos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    if angle < 1e-9:
        return np.zeros(3)
    axis = np.array(
        [
            relative[2, 1] - relative[1, 2],
            relative[0, 2] - relative[2, 0],
            relative[1, 0] - relative[0, 1],
        ]
    ) / (2.0 * np.sin(angle))
    return axis * angle
