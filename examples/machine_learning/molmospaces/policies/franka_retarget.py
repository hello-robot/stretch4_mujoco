"""
Speak Franka to a Stretch: run a Franka-trained policy on Stretch 4's actuators.

The released `allenai/MolmoBot-DROID` was trained on DROID, so it emits seven
Franka arm joint targets and a Robotiq 2F-85 gripper command, and reads seven
Franka joint angles back as proprioception. Stretch 4 has no seven-jointed arm to
send those to. This module is the interface layer that lets the checkpoint drive
Stretch anyway:

    policy action (7 Franka joint targets)
      -> VirtualFranka.fk           -> tool pose in the Franka's base frame
      -> franka_mount_pose          -> tool pose in the world
      -> FRANKA_TO_STRETCH_TOOL     -> Stretch's tool convention
      -> StretchArmIK.solve         -> base / lift / arm extension / wrist targets
      -> Stretch's own move groups

and back the other way for proprioception, so what the policy reads is where
Stretch's gripper actually is rather than an echo of its own last command.

`FrankaOnStretchView` is the whole interface, and it is deliberately usable two
ways. It answers `get_move_group("arm" | "gripper").joint_pos` and `.ctrl` like a
Franka `RobotView` would, which is what a hand-written rollout loop drives (see
`demo_droid_on_stretch.py`); and it exposes the same retargeting as pure
functions returning per-move-group dicts (`retarget_franka_joint_pos`,
`retarget_robotiq_ctrl`), which is the shape MolmoSpaces' evaluation pipeline
wants, since there the pipeline owns the write to the actuators rather than the
policy. See `policies/molmobot_droid_policy.py` for that path.

What this fixes and what it does not: retargeting fixes the *action interface*,
not the visual domain gap. The policy is still looking at a Stretch arm through a
Stretch camera, which is not what it was trained on.

This is not the repository's general-purpose Stretch IK. `policies/kinematics.py`
holds that -- a Pinocchio solver for a tool *position* plus a wrist pitch and
roll, which is what the scripted experts ask for. What is needed here is
different in three ways, so `StretchArmIK` below is its own solver: the target is
a full 6-DOF pose (the policy picks the orientation, not a grasp heuristic), the
holonomic base has to be able to join the solve, and position has to outrank
orientation rather than trade against it. See `_task_priority_step`.
"""

from __future__ import annotations

import math
from typing import Any

import mujoco
import numpy as np
from mujoco import MjData, MjModel, MjSpec
from scipy.spatial.transform import Rotation as R

from examples.machine_learning.molmospaces.stretch.robot_view import (
    Stretch4RobotView,
    commandable_limits,
)

# The rotation from the Franka's `gripper/grasp_site` frame to Stretch's
# `grasp_center_link` frame. Both models put their tool frame between the
# fingers, and both separate their fingers along that frame's +-y -- but the
# direction the gripper reaches along is +z for the Robotiq and +x for Stretch's.
# Measured, not assumed: at the Franka's `init_qpos` the grasp site's pads sit at
# [0, +-0.045, -0.005] and its z axis points down the approach; Stretch's
# fingertips sit at [-0.019, +-0.094, 0] with the approach along +x.
#
# Mapping x_stretch = z_franka, y_stretch = y_franka (and therefore
# z_stretch = -x_franka) is a -90 degree rotation about y. Without it every
# target would arrive rotated a quarter turn and the gripper would try to grasp
# edge-on.
FRANKA_TO_STRETCH_TOOL = R.from_euler("y", -90, degrees=True).as_matrix()

# Per-iteration caps on the pose error an IK step is allowed to chase, and on the
# joint motion it is allowed to ask for. See `_clamped_pose_error`.
MAX_IK_LINEAR_STEP = 0.05  # metres
MAX_IK_ANGULAR_STEP = 0.20  # radians
MAX_IK_JOINT_STEP = 0.20  # radians (or metres, for the prismatic lift and arm)

# Robotiq 2F-85 driver-joint angle at each end of the actuator's 0-255 range,
# measured by stepping the Franka model to rest at each end. The policy is fed
# `obs["qpos"]["gripper"][0]`, so these are the units its gripper state is in and
# Stretch's finger angle has to be expressed in them.
ROBOTIQ_DRIVER_OPEN = 0.003
ROBOTIQ_DRIVER_CLOSED = 0.824
ROBOTIQ_CTRL_RANGE = (0.0, 255.0)

# Stretch finger joint angle, per `gripper_finger_{right,left}_joint`: 0 closed,
# 0.5 rad fully open.
STRETCH_FINGER_OPEN = 0.5
STRETCH_FINGER_CLOSED = 0.0

# The pedestal the DROID Franka is bolted to in MolmoSpaces' own scenes, and in
# the notebook this module was ported from: `FrankaRobotConfig(base_size=[0.5,
# 0.5, 0.75])`. It is the height of `fr3_link0` above the floor, which is what
# makes a Franka tool pose land on a countertop rather than under one.
FRANKA_PEDESTAL_HEIGHT = 0.75

# Where the virtual Franka stands, in Stretch's own base frame.
#
# The notebook this comes from bolted the Franka to one spot in one kitchen
# (world [6.8, 9.75], yaw 90) and stood Stretch just behind it (world [6.73,
# 9.7], same yaw), so that a given policy output picked out the same point of the
# same room on both robots -- which is what made the two runs comparable. A
# benchmark spawns the robot in a different house every episode, so the same
# relationship has to be expressed relative to the robot instead of to the world:
# those two world poses differ by 0.05m along the base's own +x and 0.07m along
# its -y, and that is what is recorded here.
#
# Reading it the other way round is the useful way: the policy's actions are
# interpreted as those of a Franka standing on a 0.75m pedestal at Stretch's own
# feet, facing the way Stretch faces. That is the frame the whole retargeting is
# expressed in, and `franka_mount_pose_from_base()` puts it wherever the robot
# happens to be standing.
FRANKA_MOUNT_OFFSET_XY = (0.05, -0.07)


def pose_matrix(pos, quat_wxyz) -> np.ndarray:
    """A 4x4 homogeneous transform from a position and a (w, x, y, z) quaternion."""
    pose = np.eye(4)
    pose[:3, :3] = R.from_quat(np.asarray(quat_wxyz, dtype=float), scalar_first=True).as_matrix()
    pose[:3, 3] = np.asarray(pos, dtype=float)
    return pose


def franka_mount_pose_from_base(base_xytheta, pedestal_height: float = FRANKA_PEDESTAL_HEIGHT):
    """Where the virtual Franka stands, given where Stretch is standing.

    `base_xytheta` is what `StretchBaseGroup.joint_pos` reports: the base's
    (x, y, yaw) in world coordinates. The mount is built from those three numbers
    rather than from the base's 4x4 pose so that a base frame that is pitched or
    rolled -- a robot on a ramp, or mid-transient after a reset -- cannot tip the
    virtual Franka over with it. See `FRANKA_MOUNT_OFFSET_XY`.
    """
    x, y, theta = np.asarray(base_xytheta, dtype=float).reshape(-1)[:3]
    base = pose_matrix([x, y, 0.0], R.from_euler("z", theta).as_quat(scalar_first=True))
    offset = pose_matrix(
        [FRANKA_MOUNT_OFFSET_XY[0], FRANKA_MOUNT_OFFSET_XY[1], pedestal_height], [1, 0, 0, 0]
    )
    return base @ offset


def _pose_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    """The 6-vector (linear, angular) taking `current` to `target`, in world axes."""
    error = np.empty(6)
    error[:3] = target[:3, 3] - current[:3, 3]
    error[3:] = R.from_matrix(target[:3, :3] @ current[:3, :3].T).as_rotvec()
    return error


def _clamp_norm(vector: np.ndarray, limit: float) -> np.ndarray:
    """`vector`, shortened to `limit` if it is longer. Direction preserved."""
    norm = float(np.linalg.norm(vector))
    return vector if norm <= limit or norm == 0.0 else vector * (limit / norm)


def _clamped_pose_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    """`_pose_error`, with the linear and angular halves capped separately.

    A Gauss-Newton step is only valid near the current configuration, and both
    solvers here are routinely handed a target far from it -- the first call of
    an episode, or a policy chunk that jumps. Feeding the raw error in makes the
    first step enormous, which lands the arm on its joint limits, where the
    Jacobian keeps pointing outwards and the solve never comes back. Capping the
    error turns the same solve into a short line search along the same
    direction.
    """
    error = _pose_error(current, target)
    return np.concatenate(
        [_clamp_norm(error[:3], MAX_IK_LINEAR_STEP), _clamp_norm(error[3:], MAX_IK_ANGULAR_STEP)]
    )


def _damped_least_squares(jacobian: np.ndarray, error: np.ndarray, damping: float) -> np.ndarray:
    """`J^T (J J^T + lambda^2 I)^-1 error` -- the step that stays finite at singularities.

    Plain `J^+` is what an unconstrained arm would use, but neither arm here is
    unconstrained: the Franka is at a singularity whenever the policy asks for
    one, and Stretch's five manipulator DOFs cannot span a 6-DOF pose error at
    all, so `J J^T` is genuinely near-singular a lot of the time. Damping trades
    a little tracking accuracy for a bounded step instead of a joint-space
    explosion.
    """
    n = jacobian.shape[0]
    return jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + (damping**2) * np.eye(n), error)


def _damped_pseudo_inverse(jacobian: np.ndarray, damping: float) -> np.ndarray:
    """`J^T (J J^T + lambda^2 I)^-1` -- the matrix `_damped_least_squares` applies."""
    n = jacobian.shape[0]
    return jacobian.T @ np.linalg.inv(jacobian @ jacobian.T + (damping**2) * np.eye(n))


def _task_priority_step(
    linear_jacobian: np.ndarray,
    linear_error: np.ndarray,
    angular_jacobian: np.ndarray,
    angular_error: np.ndarray,
    damping: float,
    joint_scale: np.ndarray | None = None,
) -> np.ndarray:
    """A joint step that serves position first and orientation with what is left.

    Stretch's manipulator has five DOFs: enough to put the gripper at a point
    (three) with two to spare, and not enough to also pick its orientation
    freely. Asking one least-squares solve for all six at once means choosing an
    exchange rate between metres and radians, and whatever rate is chosen, poses
    the arm cannot orient bleed into position error -- measured, a target 18cm
    inside the reachable envelope came back 18cm short because the solver was
    buying unreachable orientation with it.

    Task priority removes the exchange rate. The position step is solved first;
    the orientation step is then solved only within its null space, so it can
    never move the tool off the point it was placed on. That is also the right
    priority for these tasks: reaching the object matters, and the angle the
    gripper arrives at is worth having only once it gets there.

    `joint_scale` weights the joints against each other: a joint scaled to half
    contributes half as much to the same step, so the solve reaches for it only
    when the others cannot do the job. It is applied as a change of variables
    (solve in `q / scale`, scale the answer back), which leaves the priority
    structure above untouched.
    """
    if joint_scale is not None:
        linear_jacobian = linear_jacobian * joint_scale
        angular_jacobian = angular_jacobian * joint_scale

    linear_inverse = _damped_pseudo_inverse(linear_jacobian, damping)
    step = linear_inverse @ linear_error

    null_space = np.eye(linear_jacobian.shape[1]) - linear_inverse @ linear_jacobian
    residual = angular_error - angular_jacobian @ step
    projected = angular_jacobian @ null_space
    step = step + null_space @ (_damped_pseudo_inverse(projected, damping) @ residual)
    return step if joint_scale is None else step * joint_scale


class VirtualFranka:
    """A Franka DROID arm that exists only to translate between joint angles and tool poses.

    The policy was trained on a Franka: its actions are seven joint targets and
    its proprioception is seven joint angles. Neither means anything to Stretch
    directly, so this model stands in the middle -- forward kinematics turn an
    action into a tool pose that Stretch can be asked to reach, and inverse
    kinematics turn the pose Stretch actually reached back into the seven
    numbers the policy expects to read.

    It is the same `franka_droid/model.xml` MolmoSpaces would put in the scene,
    compiled standalone and never stepped: only `mj_kinematics` runs on it, so it
    costs a few hundred microseconds per call and has no dynamics to diverge.
    """

    N_JOINTS = 7

    def __init__(self) -> None:
        from molmo_spaces.configs.robot_configs import FrankaRobotConfig
        from molmo_spaces.molmo_spaces_constants import get_robot_path

        config = FrankaRobotConfig()
        self.model: MjModel = MjSpec.from_file(
            str(get_robot_path(config.name) / config.robot_xml_path)
        ).compile()
        self.data = MjData(self.model)

        self._joint_qposadr = np.array(
            [
                self.model.jnt_qposadr[self.model.joint(f"fr3_joint{i + 1}").id]
                for i in range(self.N_JOINTS)
            ]
        )
        self._joint_dofadr = np.array(
            [
                self.model.jnt_dofadr[self.model.joint(f"fr3_joint{i + 1}").id]
                for i in range(self.N_JOINTS)
            ]
        )
        self._limits = np.array(
            [
                self.model.jnt_range[self.model.joint(f"fr3_joint{i + 1}").id]
                for i in range(self.N_JOINTS)
            ]
        )
        self._grasp_site_id = self.model.site("gripper/grasp_site").id
        self._base_body_id = self.model.body("fr3_link0").id
        self.init_qpos = np.asarray(config.init_qpos["arm"], dtype=float)

    @property
    def joint_limits(self) -> np.ndarray:
        return self._limits

    def fk(self, joint_pos: np.ndarray) -> np.ndarray:
        """Tool pose for a joint vector, as a 4x4 in the `fr3_link0` frame."""
        self.data.qpos[self._joint_qposadr] = np.clip(
            np.asarray(joint_pos, dtype=float), self._limits[:, 0], self._limits[:, 1]
        )
        mujoco.mj_kinematics(self.model, self.data)
        pose = np.eye(4)
        pose[:3, :3] = self.data.site_xmat[self._grasp_site_id].reshape(3, 3)
        pose[:3, 3] = self.data.site_xpos[self._grasp_site_id]
        # The standalone model puts fr3_link0 at the origin with no rotation, so
        # world and base frame coincide; asserted rather than assumed because a
        # future model.xml could wrap the arm in a mount body.
        assert np.allclose(self.data.xpos[self._base_body_id], 0.0, atol=1e-9)
        return pose

    def ik(
        self,
        target_pose: np.ndarray,
        seed: np.ndarray,
        iterations: int = 60,
        damping: float = 0.05,
        tolerance: float = 1e-4,
    ) -> np.ndarray:
        """Joint angles whose tool pose is `target_pose` (in the `fr3_link0` frame).

        Warm-started from `seed`, which in use is the previous step's answer, so
        successive calls stay on the same IK branch. Without that the reported
        arm state could jump between elbow-up and elbow-down between two
        physically adjacent tool poses, which the policy would read as the arm
        having teleported.
        """
        joint_pos = np.clip(
            np.asarray(seed, dtype=float).copy(), self._limits[:, 0], self._limits[:, 1]
        )
        jacobian = np.zeros((6, self.model.nv))
        for _ in range(iterations):
            self.data.qpos[self._joint_qposadr] = joint_pos
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)

            current = np.eye(4)
            current[:3, :3] = self.data.site_xmat[self._grasp_site_id].reshape(3, 3)
            current[:3, 3] = self.data.site_xpos[self._grasp_site_id]
            error = _clamped_pose_error(current, target_pose)
            if np.linalg.norm(error) < tolerance:
                break

            jacobian[:] = 0.0
            mujoco.mj_jacSite(
                self.model, self.data, jacobian[:3], jacobian[3:], self._grasp_site_id
            )
            step = _damped_least_squares(jacobian[:, self._joint_dofadr], error, damping)
            joint_pos = np.clip(
                joint_pos + np.clip(step, -MAX_IK_JOINT_STEP, MAX_IK_JOINT_STEP),
                self._limits[:, 0],
                self._limits[:, 1],
            )
        return joint_pos


class StretchArmIK:
    """Solves for Stretch's base / lift / telescoping arm / wrist given a tool pose.

    Runs on its own `MjData` over the scene's model rather than on the live one.
    An IK iteration has to move the robot to evaluate the next Jacobian, and
    doing that in the data the simulation is stepping would drag the robot
    through every intermediate guess. The scratch copy is re-synced from the live
    `qpos` at the start of each solve, so the base pose (and anything else the
    chain hangs off) is current.

    Five DOFs against a six-dimensional pose error, so most targets are not
    exactly reachable. `_task_priority_step` decides what gives: position is
    solved for outright and orientation only in what is left over, so an
    unreachable *angle* costs nothing in *position*.

    Those five are lift, extension and wrist -- and on their own they are not
    enough for these tasks. The arm telescopes along one fixed direction and the
    wrist can swing the tool about 0.2m off that line, so a standing Stretch
    reaches a narrow corridor: measured in the kitchen this was developed in, a
    base that can touch the salt shaker was 0.41m short of the bowl 0.73m away.
    The real robot solves this by driving, which is what `include_base` lets the
    solver do -- the holonomic base joins the IK as three more DOFs.

    It joins on a leash and at a price. `base_leash` bounds how far the base may
    end up from where it was placed, so a solve for an unreachable target cannot
    walk the robot out of the room; `base_cost` makes a metre of driving as
    expensive as `base_cost` metres of arm motion, so the base stays put while
    the arm can still do the job and contributes only when it cannot. Turn
    `include_base` off to see what the arm alone can do.
    """

    ARM_GROUPS = ("lift", "arm", "wrist")

    def __init__(
        self,
        stretch_view: Stretch4RobotView,
        namespace: str,
        include_base: bool = True,
        base_leash: tuple[float, float, float] = (0.7, 0.15, math.pi / 3),
        base_cost: float = 5.0,
        iterations: int = 80,
        damping: float = 0.08,
        tolerance: float = 1e-3,
    ) -> None:
        self._live_view = stretch_view
        self._live_data: MjData = stretch_view.mj_data
        self._scratch_data = MjData(self._live_data.model)
        self._scratch_view = Stretch4RobotView(self._scratch_data, namespace)

        self.GROUPS = (("base",) if include_base else ()) + self.ARM_GROUPS
        self._iterations = iterations
        self._damping = damping
        self._tolerance = tolerance
        self._widths = [self._scratch_view.get_move_group(g).pos_dim for g in self.GROUPS]
        self._limits = np.concatenate(
            [commandable_limits(self._scratch_view.get_move_group(g)) for g in self.GROUPS]
        )
        self._joint_scale = np.ones(sum(self._widths))
        self._base_leash = np.asarray(base_leash, dtype=float)
        self._include_base = include_base
        self._base_cost = float(base_cost)
        if include_base:
            self.releash()

    def releash(self) -> None:
        """Re-centre the base's leash on wherever the robot is standing now.

        The base's own limits are the +-25m travel of the virtual slide joints,
        which is no constraint at all, so they are replaced with a box around the
        robot's current position. Called at construction and again on every
        `FrankaOnStretchView.reset()`, because an episode that starts in a new
        house starts with the box centred on the last one otherwise.

        The box is in *world* axes, because that is what
        `HoloJointsRobotBaseGroup` reports. The default leash is therefore
        deliberately close to isotropic in the plane rather than tight across the
        robot's facing: a per-episode spawn yaw is not known here, and a box that
        assumed one would be a leash that let the robot drive into the counter in
        half the houses. It is the IK's only collision awareness -- it is solving
        kinematics, not contacts.
        """
        if not self._include_base:
            return
        home = np.asarray(self._live_view.get_move_group("base").joint_pos, dtype=float)
        self._limits[:3, 0] = home - self._base_leash
        self._limits[:3, 1] = home + self._base_leash
        self._joint_scale[:3] = 1.0 / self._base_cost

    def _read(self, view: Any) -> np.ndarray:
        return np.concatenate(
            [np.asarray(view.get_move_group(g).joint_pos, dtype=float) for g in self.GROUPS]
        )

    def _write(self, view: Any, joint_pos: np.ndarray) -> None:
        offset = 0
        for group, width in zip(self.GROUPS, self._widths):
            view.get_move_group(group).joint_pos = joint_pos[offset : offset + width]
            offset += width

    def split(self, joint_pos: np.ndarray) -> dict[str, np.ndarray]:
        """A flat joint vector split into the per-move-group dict callers want."""
        offset = 0
        out = {}
        for group, width in zip(self.GROUPS, self._widths):
            out[group] = joint_pos[offset : offset + width]
            offset += width
        return out

    def tool_pose(self) -> np.ndarray:
        """The live robot's current tool pose in the world, as a 4x4."""
        return self._live_view.get_move_group("wrist").leaf_frame_to_world

    def solve(self, target_pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Joint targets for `target_pose` (world frame), and the residual 6-vector.

        Seeded from the robot's current configuration, so the answer is the
        nearest compromise to where the arm already is rather than an unrelated
        branch of the same least-squares problem.
        """
        self._scratch_data.qpos[:] = self._live_data.qpos
        joint_pos = np.clip(self._read(self._live_view), self._limits[:, 0], self._limits[:, 1])

        error = np.zeros(6)
        for _ in range(self._iterations):
            self._write(self._scratch_view, joint_pos)
            mujoco.mj_kinematics(self._live_data.model, self._scratch_data)
            mujoco.mj_comPos(self._live_data.model, self._scratch_data)

            current = self._scratch_view.get_move_group("wrist").leaf_frame_to_world
            error = _pose_error(current, target_pose)
            step_error = _clamped_pose_error(current, target_pose)
            if np.linalg.norm(step_error) < self._tolerance:
                break

            jacobian = self._scratch_view.get_jacobian("wrist", list(self.GROUPS))
            step = _task_priority_step(
                jacobian[:3],
                step_error[:3],
                jacobian[3:],
                step_error[3:],
                self._damping,
                self._joint_scale,
            )
            joint_pos = np.clip(
                joint_pos + np.clip(step, -MAX_IK_JOINT_STEP, MAX_IK_JOINT_STEP),
                self._limits[:, 0],
                self._limits[:, 1],
            )
        return joint_pos, error


class _ProxyArmGroup:
    """Stretch's lift + arm + wrist, wearing the Franka arm's seven-joint interface."""

    def __init__(self, view: "FrankaOnStretchView") -> None:
        self._view = view

    @property
    def joint_pos(self) -> np.ndarray:
        return self._view.franka_joint_pos()

    @property
    def ctrl(self) -> np.ndarray:
        return self._view.last_arm_ctrl.copy()

    @ctrl.setter
    def ctrl(self, value) -> None:
        self._view.command_franka_joint_pos(value)

    @property
    def noop_ctrl(self) -> np.ndarray:
        return self.joint_pos.copy()

    @property
    def joint_pos_limits(self) -> np.ndarray:
        return self._view.franka.joint_limits


class _ProxyGripperGroup:
    """Stretch's two fingers, wearing the Robotiq 2F-85's interface."""

    def __init__(self, view: "FrankaOnStretchView") -> None:
        self._view = view

    @property
    def joint_pos(self) -> np.ndarray:
        """Stretch's finger angle expressed as Robotiq driver-joint angles.

        Two values, because the Franka view reports two and the caller hands the
        whole vector to the policy -- which reads only the first
        (`gripper_representation_count = 1`), but reads it in these units.
        """
        finger = float(np.mean(self._view.stretch_view.get_move_group("gripper").joint_pos))
        fraction = (finger - STRETCH_FINGER_CLOSED) / (STRETCH_FINGER_OPEN - STRETCH_FINGER_CLOSED)
        driver = ROBOTIQ_DRIVER_CLOSED + fraction * (ROBOTIQ_DRIVER_OPEN - ROBOTIQ_DRIVER_CLOSED)
        return np.array([driver, driver])

    @property
    def ctrl(self) -> np.ndarray:
        return self._view.last_gripper_ctrl.copy()

    @ctrl.setter
    def ctrl(self, value) -> None:
        self._view.command_robotiq_ctrl(value)

    @property
    def noop_ctrl(self) -> np.ndarray:
        return self.ctrl


class FrankaOnStretchView:
    """Drives Stretch 4 through the move-group interface MolmoBot-DROID expects.

    A policy loop only ever does four things to a robot view: read
    `get_move_group("arm").joint_pos`, read `get_move_group("gripper").joint_pos`,
    and write `.ctrl` on each. This class answers all four in Franka DROID
    coordinates while Stretch does the moving. The retargeting is also available
    as `retarget_franka_joint_pos` / `retarget_robotiq_ctrl`, which return the
    per-move-group targets instead of writing them -- for MolmoSpaces'
    evaluation pipeline, which applies the action itself.

    `franka_mount_pose` is where the virtual Franka stands. Anchoring it to
    Stretch's own base (see `franka_mount_pose_from_base`) is what makes a
    policy trained in the Franka's frame mean anything here: the actions are
    interpreted as those of a Franka on a pedestal at Stretch's feet, facing the
    way Stretch faces, so they pick out points in front of the robot in whatever
    house the episode is in.

    Two things this cannot paper over, both worth keeping in mind when reading a
    rollout:

    * Stretch's manipulator has five DOFs (lift, extension, and a 3-DOF wrist)
      against the Franka's seven. Six-DOF tool poses are therefore approximated,
      not reached -- `last_residual` reports by how much, and
      `last_position_error` is the part of it you can see. The holonomic base
      joins the solve when `include_base` is set, which buys most of the reach
      back; see `StretchArmIK`.
    * The policy is looking through Stretch's camera at a Stretch arm, which is
      not what it was trained on. Retargeting fixes the action interface, not
      the visual domain gap.

    `target_z_offset` raises every retargeted target by a fixed height, and
    lowers Stretch's reported tool pose by the same amount on the way back -- so
    it is exactly "stand the virtual Franka this much higher", and the two
    directions stay each other's inverse. It is there because Stretch's tool
    centre ends up *lower* than the Franka's wherever the lift runs out of
    travel: over a counter, with the gripper pointing down, the lift caps the
    grasp centre at 1.082m against the Franka's home 1.185m, and a gripper 10cm
    deeper than the one the policy was trained with is a gripper that hits the
    countertop and knocks over what the Franka would have cleared.
    `measure_tool_height_offset()` returns the shortfall to set it from.

    Two things it does not do. Where the lift is *already* saturated the offset
    changes nothing -- the target moves up, the robot cannot follow, and the
    residual simply grows; it buys clearance in the part of the workspace where
    the arm still has somewhere to go. And it is a bias, not a correction: the
    policy is closing its loop through Stretch's camera, so it will spend some
    of the offset driving back down towards whatever it is looking at. Raise it
    for clearance, lower it towards zero to grasp.
    """

    def __init__(
        self,
        stretch_view: Stretch4RobotView,
        namespace: str,
        franka_mount_pose: np.ndarray,
        include_base: bool = True,
        target_z_offset: float = 0.0,
    ) -> None:
        self.stretch_view = stretch_view
        self.namespace = namespace
        self.target_z_offset = float(target_z_offset)

        self.franka = VirtualFranka()
        self.arm_ik = StretchArmIK(stretch_view, namespace, include_base=include_base)

        self._tool_correction = np.eye(4)
        self._tool_correction[:3, :3] = FRANKA_TO_STRETCH_TOOL
        self._tool_correction_inverse = np.linalg.inv(self._tool_correction)

        self._move_groups = {"arm": _ProxyArmGroup(self), "gripper": _ProxyGripperGroup(self)}
        self._franka_seed = self.franka.init_qpos.copy()
        self.last_arm_ctrl = self.franka.init_qpos.copy()
        self.last_gripper_ctrl = np.array([ROBOTIQ_CTRL_RANGE[0]])
        self.last_residual = np.zeros(6)
        self.set_franka_mount_pose(franka_mount_pose)

    # -- the bits of the RobotView interface a policy loop uses ---------------

    def move_group_ids(self) -> list[str]:
        return list(self._move_groups)

    def get_move_group(self, move_group_id: str):
        return self._move_groups[move_group_id]

    def set_franka_mount_pose(self, franka_mount_pose: np.ndarray) -> None:
        """Move the virtual Franka. See `franka_mount_pose_from_base`."""
        self.franka_mount_pose = np.asarray(franka_mount_pose, dtype=float)
        self._mount_inverse = np.linalg.inv(self.franka_mount_pose)

    def reset(self) -> None:
        """Re-seed the Franka-side state from wherever Stretch currently is.

        Call this after resetting the simulation. The IK seed, the reported
        `ctrl` and the base's leash are the only state this class carries across
        steps, and all of them describe a robot configuration -- left over from
        the previous episode they would make the first action of the new one a
        step away from a pose the robot is no longer in, and would leash the base
        to a house it has left.
        """
        self.arm_ik.releash()
        self._franka_seed = self.franka.init_qpos.copy()
        self.last_arm_ctrl = self.franka_joint_pos()
        self.last_gripper_ctrl = np.array([ROBOTIQ_CTRL_RANGE[0]])
        self.last_residual = np.zeros(6)

    def snap_to_franka_joint_pos(self, joint_pos=None) -> np.ndarray:
        """Put Stretch in the configuration that best matches a Franka arm pose.

        The two robots have unrelated home configurations, so a freshly reset
        Stretch stands somewhere the policy's first observation reads as an arm
        two-and-a-bit radians from where it expects to be -- a large apparent
        jump before it has acted at all, which `relative_max_joint_delta` then
        spends its first chunk correcting. Starting Stretch at the Franka's home
        *tool pose* removes that.

        Writes `joint_pos` rather than commanding it, so the robot is there
        immediately rather than a settling transient later, and leaves the
        controllers targeting the same configuration. Returns the residual
        6-vector -- expect a non-zero one, since the Franka's home pose is near
        the top of Stretch's lift travel.
        """
        joint_pos = self.franka.init_qpos if joint_pos is None else joint_pos
        joint_pos = np.asarray(joint_pos, dtype=float)

        target = self.franka_tool_pose_to_world(self.franka.fk(joint_pos))
        solution, residual = self.arm_ik.solve(target)
        for group, value in self.arm_ik.split(solution).items():
            move_group = self.stretch_view.get_move_group(group)
            move_group.joint_pos = value
            move_group.ctrl = value
        mujoco.mj_forward(self.stretch_view.mj_data.model, self.stretch_view.mj_data)

        self.last_arm_ctrl = joint_pos.copy()
        self._franka_seed = joint_pos.copy()
        self.last_residual = residual
        return residual

    # -- the retargeting itself ----------------------------------------------

    def stretch_tool_pose_to_franka(self, tool_pose_world: np.ndarray) -> np.ndarray:
        """A Stretch tool pose in the world -> the Franka grasp site's pose in its base frame."""
        pose = np.array(tool_pose_world, dtype=float, copy=True)
        pose[2, 3] -= self.target_z_offset
        return self._mount_inverse @ pose @ self._tool_correction_inverse

    def franka_tool_pose_to_world(self, tool_pose_franka: np.ndarray) -> np.ndarray:
        """A Franka grasp-site pose in its base frame -> a Stretch tool pose in the world."""
        pose = self.franka_mount_pose @ tool_pose_franka @ self._tool_correction
        pose[2, 3] += self.target_z_offset
        return pose

    def measure_tool_height_offset(self, joint_pos=None) -> float:
        """How far below a Franka tool pose Stretch's own tool centre ends up, in metres.

        Solves for the Franka's home pose (or `joint_pos`) *ignoring* whatever
        offset is currently set and returns the z component of what is left over
        -- positive when Stretch comes up short, which is the value
        `target_z_offset` wants. Runs on the IK's scratch data, so it measures
        without moving the robot; seeded from where the robot is standing now, so
        call it after the reset that puts it there.
        """
        joint_pos = self.franka.init_qpos if joint_pos is None else joint_pos
        target = (
            self.franka_mount_pose
            @ self.franka.fk(np.asarray(joint_pos, dtype=float))
            @ self._tool_correction
        )
        _, residual = self.arm_ik.solve(target)
        return float(residual[2])

    def franka_joint_pos(self) -> np.ndarray:
        """Where Stretch's gripper is, reported as seven Franka joint angles.

        Seeded from the last command rather than from the last answer, which is
        what makes this degrade gracefully. Stretch cannot reach every pose the
        Franka can, so some of these solves have no exact answer; seeding from
        the command means the reported state is "the configuration closest to
        what I was asked for that matches where the gripper actually is", and
        collapses to an echo of the command exactly when Stretch tracked it.
        Seeding from the previous answer instead lets a run of unreachable
        targets walk the reported arm somewhere the policy never sent it.
        """
        target = self.stretch_tool_pose_to_franka(self.arm_ik.tool_pose())
        self._franka_seed = self.franka.ik(target, self.last_arm_ctrl)
        return self._franka_seed.copy()

    def retarget_franka_joint_pos(self, joint_pos) -> dict[str, np.ndarray]:
        """Seven Franka joint targets -> per-move-group targets for Stretch.

        Pure: it solves and records the residual but writes nothing to the robot,
        so the caller decides whether these become `.ctrl` (a hand-written
        rollout loop) or an action dict (MolmoSpaces' evaluation pipeline, which
        applies it and clips it against the model's limits).
        """
        joint_pos = np.asarray(joint_pos, dtype=float).reshape(-1)[: VirtualFranka.N_JOINTS]
        self.last_arm_ctrl = joint_pos.copy()

        target = self.franka_tool_pose_to_world(self.franka.fk(joint_pos))
        solution, self.last_residual = self.arm_ik.solve(target)
        return self.arm_ik.split(solution)

    def retarget_robotiq_ctrl(self, value) -> np.ndarray:
        """A Robotiq 0-255 command -> Stretch's two finger targets, in radians."""
        command = float(np.asarray(value, dtype=float).reshape(-1)[0])
        self.last_gripper_ctrl = np.array([command])
        fraction = np.clip(
            (command - ROBOTIQ_CTRL_RANGE[0]) / (ROBOTIQ_CTRL_RANGE[1] - ROBOTIQ_CTRL_RANGE[0]),
            0.0,
            1.0,
        )
        # 0 is open on the Robotiq and closed on Stretch, hence the flip.
        finger = STRETCH_FINGER_OPEN + fraction * (STRETCH_FINGER_CLOSED - STRETCH_FINGER_OPEN)
        return np.array([finger, finger])

    def command_franka_joint_pos(self, joint_pos) -> None:
        """Send seven Franka joint targets, retargeted onto Stretch's actuators."""
        for group, value in self.retarget_franka_joint_pos(joint_pos).items():
            self.stretch_view.get_move_group(group).ctrl = value

    def command_robotiq_ctrl(self, value) -> None:
        """Send a Robotiq 0-255 command, retargeted onto Stretch's two fingers."""
        self.stretch_view.get_move_group("gripper").ctrl = self.retarget_robotiq_ctrl(value)

    # -- diagnostics ----------------------------------------------------------

    @property
    def last_position_error(self) -> float:
        """How far the last commanded tool position was from what Stretch can reach, in metres."""
        return float(np.linalg.norm(self.last_residual[:3]))

    @property
    def last_orientation_error(self) -> float:
        """The same for orientation, in radians."""
        return float(np.linalg.norm(self.last_residual[3:]))


# =============================================================================
# Looking at where the tool frame actually is
# =============================================================================

# Colours the overlay below draws each robot's tool frame in, and the reference
# height with. Kept together so the same marker means the same thing in every
# image -- a retargeting target and the pose Stretch reached for it are most
# usefully looked at side by side.
FRANKA_TOOL_COLOR = (0.10, 0.85, 0.25, 1.0)
STRETCH_TOOL_COLOR = (1.00, 0.35, 0.10, 1.0)
REFERENCE_PLANE_COLOR = (0.10, 0.85, 0.25, 0.20)

# Maps the geom-local +z that `mjGEOM_ARROW` points along onto each axis of the
# frame being drawn, so one arrow primitive can draw all three.
_AXIS_TO_ARROW = (
    R.from_euler("y", 90, degrees=True).as_matrix(),
    R.from_euler("x", -90, degrees=True).as_matrix(),
    np.eye(3),
)


def _add_decor_geom(scene, geom_type, size, pos, mat, rgba, label: str = "") -> None:
    """Append one decorative geom to an already-updated `MjvScene`."""
    if scene.ngeom >= scene.maxgeom:
        raise RuntimeError(f"MjvScene is full at {scene.maxgeom} geoms")
    scene.ngeom += 1
    geom = scene.geoms[scene.ngeom - 1]
    mujoco.mjv_initGeom(
        geom,
        geom_type,
        np.asarray(size, dtype=float),
        np.asarray(pos, dtype=float),
        np.asarray(mat, dtype=float).reshape(9),
        np.asarray(rgba, dtype=np.float32),
    )
    geom.category = mujoco.mjtCatBit.mjCAT_DECOR
    geom.label = label


def add_frame_marker(
    scene,
    pose: np.ndarray,
    color=FRANKA_TOOL_COLOR,
    label: str = "",
    ball_radius: float = 0.012,
    axis_length: float = 0.07,
    axis_radius: float = 0.004,
) -> None:
    """Draw a 4x4 pose into a rendered scene: a ball at the origin, arrows on the axes.

    The ball is the tool centre, and what to compare between two images; the
    arrows are what tell you two frames are also oriented differently -- the
    Robotiq reaches along its tool +z and Stretch's gripper along its tool +x,
    which is the whole job of `FRANKA_TO_STRETCH_TOOL`, and is much easier to
    believe on sight than from a rotation matrix.

    Axes are coloured x/y/z as red/green/blue as usual; `color` is the ball, and
    identifies which frame it is.
    """
    origin = pose[:3, 3]
    _add_decor_geom(
        scene, mujoco.mjtGeom.mjGEOM_SPHERE, [ball_radius] * 3, origin, np.eye(3), color, label
    )
    for axis, axis_color in enumerate(((1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1))):
        _add_decor_geom(
            scene,
            mujoco.mjtGeom.mjGEOM_ARROW,
            [axis_radius, axis_radius, axis_length],
            origin,
            pose[:3, :3] @ _AXIS_TO_ARROW[axis],
            axis_color,
        )


def add_height_plane(
    scene,
    height: float,
    center,
    color=REFERENCE_PLANE_COLOR,
    radius: float = 0.30,
    label: str = "",
) -> None:
    """A translucent horizontal disk at `height`, to carry one z across two images.

    Two separately rendered scenes have no shared ruler. Drawing the *same*
    world height into both gives one: whichever tool ball sits under its own disk
    is the lower of the two, by however much it hangs below.
    """
    _add_decor_geom(
        scene,
        mujoco.mjtGeom.mjGEOM_CYLINDER,
        # `mjGEOM_CYLINDER` reads size as (radius, half-length): a disk, not a
        # drum, or it swallows the very markers it is a ruler for.
        [radius, 0.0015, 0.0],
        [center[0], center[1], height],
        np.eye(3),
        color,
        label,
    )


def render_tool_frame_view(
    model: MjModel,
    data: MjData,
    lookat,
    markers: list[tuple[np.ndarray, tuple, str]],
    reference_height: float | None = None,
    width: int = 640,
    height: int = 360,
    distance: float = 1.6,
    azimuth: float = 25.0,
    elevation: float = -12.0,
) -> np.ndarray:
    """One free-camera frame of `data`, with tool-frame markers drawn over it.

    A free camera rather than the robot's own, because the point is to see the
    gripper from outside at a stated height; passing the same `lookat`,
    `distance`, `azimuth` and `elevation` to two calls makes them the same view
    of two different robots. Sites stay hidden (`sitegroup = 0`) as everywhere
    else here, so the only frame drawn is the one asked for.
    """
    renderer = mujoco.Renderer(model, height, width)
    try:
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = np.asarray(lookat, dtype=float)
        camera.distance = distance
        camera.azimuth = azimuth
        camera.elevation = elevation

        scene_option = mujoco.MjvOption()
        scene_option.sitegroup = 0
        renderer.update_scene(data, camera=camera, scene_option=scene_option)

        if reference_height is not None:
            add_height_plane(renderer.scene, reference_height, lookat)
        for pose, color, label in markers:
            add_frame_marker(renderer.scene, pose, color=color, label=label)
        return renderer.render()
    finally:
        renderer.close()


def side_by_side(*frames: np.ndarray, background: int = 0) -> np.ndarray:
    """Frames laid out in a row, top-aligned and padded to the tallest.

    Plain `np.hstack` is enough while both cameras are 640x360. Stretch's right
    head camera comes out of its quarter turn as a portrait frame, so they no
    longer share a height.
    """
    height = max(frame.shape[0] for frame in frames)
    padded = [
        np.pad(frame, ((0, height - frame.shape[0]), (0, 0), (0, 0)), constant_values=background)
        for frame in frames
    ]
    return np.hstack(padded)
