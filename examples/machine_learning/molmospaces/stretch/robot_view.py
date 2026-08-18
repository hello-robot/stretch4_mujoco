"""
A MolmoSpaces `RobotView` for the Stretch 4 MJCF.

MolmoSpaces drives every robot through the `RobotView` / `MoveGroup` abstraction
(`molmo_spaces/robots/robot_views/abstract.py`): a view is a dict of named move
groups, each of which maps a set of MuJoCo joints onto a set of actuators and
exposes a leaf frame. Tasks, sensors and controllers only ever talk to that
abstraction, which is why a correct view is most of what it takes to run Stretch
on their benchmarks.

Stretch 4 is split into five groups:

    base     3  virtual holonomic joints (x, y, theta), added at attach time by
                `Stretch4Robot.add_robot_to_scene()`. See `robot.py` for why the
                real omniwheels are not used.
    lift     1  prismatic mast joint
    arm      1  telescoping extension -- four prismatic segments driven by a
                single tendon actuator, see `StretchTelescopingArmGroup`
    wrist    3  yaw, pitch, roll
    gripper  2  the two finger joints

which is 10 commanded degrees of freedom, matching the 10 actuators the
generated MJCF ends up with once the wheel actuators are replaced.
"""

from functools import cached_property

import numpy as np
from mujoco import MjData

from molmo_spaces.robots.robot_views.abstract import (
    GripperGroup,
    HoloJointsRobotBaseGroup,
    MJCFFrameMixin,
    MoveGroup,
    RobotView,
    SimplyActuatedMoveGroup,
)
from molmo_spaces.utils.linalg_utils import normalize_ang_error
from molmo_spaces.utils.mj_model_and_data_utils import body_pose

# The body the Stretch MJCF hangs its whole kinematic chain off, and the body we
# treat as the tool centre point. `grasp_center_link` sits between the two
# fingers, so it is the natural analogue of the Franka model's `grasp_site`.
TCP_BODY = "grasp_center_link"

# Parent of both `grasp_center_link` and the finger chain. Used as the gripper
# group's root body, which is what `BaseMujocoTask.get_task_objects()` reports as
# the "gripper" body and what contact checks descend from -- so it has to be an
# ancestor of the finger geoms, not the TCP body itself.
GRIPPER_ROOT_BODY = "quick_connect_interface_link"

# The four telescoping segments are equality-constrained to move together, and
# the "arm" actuator drives a tendon whose length is their sum.
N_TELESCOPING_SEGMENTS = 4


class _TCPLeafMixin(MJCFFrameMixin):
    """Leaf frame shared by the lift/arm/wrist groups: the gripper's TCP body.

    All three groups are serial links in the same chain, so reporting the TCP as
    the leaf of each of them is what makes `RobotView.get_jacobian("wrist",
    ["lift", "arm", "wrist"])` return a Jacobian of the tool frame with respect
    to the whole manipulator -- which is what `MlSpacesKinematics.ik()` needs.
    """

    @property
    def leaf_frame_id(self) -> int:
        return self._tcp_body_id

    @property
    def leaf_frame_type(self):
        return "body"

    @property
    def root_frame_to_world(self) -> np.ndarray:
        return body_pose(self.mj_data, self._root_body_id)


class StretchBaseGroup(HoloJointsRobotBaseGroup):
    """The virtual holonomic (x, y, theta) base added by `add_robot_to_scene()`."""

    def __init__(self, mj_data: MjData, namespace: str = "") -> None:
        model = mj_data.model
        super().__init__(
            mj_data,
            world_site_id=model.site(f"{namespace}world").id,
            holo_base_site_id=model.site(f"{namespace}base_site").id,
            joint_ids=[model.joint(f"{namespace}base_{a}").id for a in ("x", "y", "theta")],
            actuator_ids=[
                model.actuator(f"{namespace}base_{a}_act").id for a in ("x", "y", "theta")
            ],
            root_body_id=model.body(f"{namespace}base").id,
        )


class StretchLiftGroup(_TCPLeafMixin, SimplyActuatedMoveGroup):
    """The single prismatic lift joint that runs up the mast."""

    def __init__(self, mj_data: MjData, base_group: StretchBaseGroup, namespace: str = "") -> None:
        model = mj_data.model
        self._tcp_body_id = model.body(f"{namespace}{TCP_BODY}").id
        super().__init__(
            mj_data,
            [model.joint(f"{namespace}lift_joint").id],
            [model.actuator(f"{namespace}lift").id],
            model.body(f"{namespace}lift_link").id,
            base_group,
        )


class StretchTelescopingArmGroup(_TCPLeafMixin, MoveGroup):
    """The telescoping arm, presented to MolmoSpaces as a single extension DOF.

    The MJCF models the arm as four prismatic segments (`arm_l1_joint` ..
    `arm_l4_joint`) held equal by three `<equality joint>` constraints and driven
    by one tendon actuator whose control range is the *total* extension. Every
    MolmoSpaces controller (see `controllers/joint_pos.py`) assumes a move group's
    `joint_pos` is directly comparable to its actuator control range, so this
    group reports the sum of the four segments as a length-1 joint position and
    splits a commanded extension evenly back across them.

    All four joints are still passed to `MoveGroup.__init__`, so the group owns
    the right MuJoCo DOFs for Jacobian purposes; `Stretch4RobotView.get_jacobian()`
    collapses those four columns into the one extension column that matches
    `vel_dim`.
    """

    def __init__(self, mj_data: MjData, base_group: StretchBaseGroup, namespace: str = "") -> None:
        model = mj_data.model
        self._tcp_body_id = model.body(f"{namespace}{TCP_BODY}").id
        joint_ids = [
            model.joint(f"{namespace}arm_l{i}_joint").id
            for i in range(1, N_TELESCOPING_SEGMENTS + 1)
        ]
        super().__init__(
            mj_data,
            joint_ids,
            [model.actuator(f"{namespace}arm").id],
            model.body(f"{namespace}arm_l0_link").id,
            base_group,
        )

    @cached_property
    def pos_dim(self) -> int:
        return 1

    @cached_property
    def vel_dim(self) -> int:
        return 1

    @property
    def joint_pos(self) -> np.ndarray:
        return np.array([self.mj_data.qpos[self._joint_posadr].sum()])

    @joint_pos.setter
    def joint_pos(self, joint_pos: np.ndarray) -> None:
        extension = float(np.asarray(joint_pos).reshape(-1)[0])
        self.mj_data.qpos[self._joint_posadr] = extension / N_TELESCOPING_SEGMENTS

    @property
    def joint_vel(self) -> np.ndarray:
        return np.array([self.mj_data.qvel[self._joint_veladr].sum()])

    @joint_vel.setter
    def joint_vel(self, joint_vel: np.ndarray) -> None:
        rate = float(np.asarray(joint_vel).reshape(-1)[0])
        self.mj_data.qvel[self._joint_veladr] = rate / N_TELESCOPING_SEGMENTS

    @property
    def joint_pos_limits(self) -> np.ndarray:
        # The per-segment limits summed, which is also the tendon actuator's range.
        limits = np.array(
            [self.mj_model.jnt_range[jnt_id] for jnt_id in self._joint_ids], dtype=float
        )
        return limits.sum(axis=0).reshape(1, 2)

    def integrate_joint_vel(self, joint_pos: np.ndarray, joint_vel: np.ndarray) -> np.ndarray:
        return np.asarray(joint_pos) + np.asarray(joint_vel)

    @property
    def noop_ctrl(self) -> np.ndarray:
        return self.joint_pos.copy()


class StretchWristGroup(_TCPLeafMixin, SimplyActuatedMoveGroup):
    """Wrist yaw / pitch / roll, the three revolute joints ahead of the gripper."""

    JOINTS = ("wrist_yaw", "wrist_pitch", "wrist_roll")

    def __init__(self, mj_data: MjData, base_group: StretchBaseGroup, namespace: str = "") -> None:
        model = mj_data.model
        self._tcp_body_id = model.body(f"{namespace}{TCP_BODY}").id
        super().__init__(
            mj_data,
            [model.joint(f"{namespace}{j}_joint").id for j in self.JOINTS],
            [model.actuator(f"{namespace}{j}").id for j in self.JOINTS],
            model.body(f"{namespace}wrist_link").id,
            base_group,
        )


class StretchGripperGroup(MJCFFrameMixin, GripperGroup):
    """The two-finger gripper.

    Each finger has its own revolute joint and actuator, both with range
    [0, 0.5] rad where 0 is fully closed. The fingertip separation that range
    spans was measured off the compiled model, see `INTER_FINGER_DIST_RANGE`.

    The compliant fingertip joints (`gripper_fingertip_*_compliant_{x,y}`) are
    deliberately left out: they are passive, so including them would make
    `joint_pos` wider than the actuator vector and break `JointPosController`.
    """

    # Fingertip separation at joint_pos 0.0 and 0.5, from forward kinematics on
    # the compiled MJCF.
    INTER_FINGER_DIST_RANGE = (0.0, 0.1885)
    OPEN_JOINT_POS = 0.5
    CLOSED_JOINT_POS = 0.0

    def __init__(self, mj_data: MjData, base_group: StretchBaseGroup, namespace: str = "") -> None:
        model = mj_data.model
        joint_ids = [
            model.joint(f"{namespace}gripper_finger_right_joint").id,
            model.joint(f"{namespace}gripper_finger_left_joint").id,
        ]
        actuator_ids = [
            model.actuator(f"{namespace}gripper_right_finger").id,
            model.actuator(f"{namespace}gripper_left_finger").id,
        ]
        super().__init__(
            mj_data,
            joint_ids,
            actuator_ids,
            model.body(f"{namespace}{GRIPPER_ROOT_BODY}").id,
            base_group,
        )
        self._tcp_body_id = model.body(f"{namespace}{TCP_BODY}").id
        self._right_tip_body_id = model.body(f"{namespace}gripper_fingertip_right_link").id
        self._left_tip_body_id = model.body(f"{namespace}gripper_fingertip_left_link").id

    @property
    def leaf_frame_id(self) -> int:
        return self._tcp_body_id

    @property
    def leaf_frame_type(self):
        return "body"

    @property
    def root_frame_to_world(self) -> np.ndarray:
        return self.leaf_frame_to_world

    def set_gripper_ctrl_open(self, open: bool) -> None:
        target = self.OPEN_JOINT_POS if open else self.CLOSED_JOINT_POS
        self.ctrl = np.array([target, target])

    @property
    def inter_finger_dist_range(self) -> tuple[float, float]:
        return self.INTER_FINGER_DIST_RANGE

    @property
    def inter_finger_dist(self) -> float:
        right = self.mj_data.xpos[self._right_tip_body_id]
        left = self.mj_data.xpos[self._left_tip_body_id]
        return float(np.linalg.norm(right - left))


class Stretch4RobotView(RobotView):
    """The five Stretch 4 move groups, in the order a flat action vector uses."""

    MOVE_GROUP_ORDER = ("base", "lift", "arm", "wrist", "gripper")

    def __init__(self, mj_data: MjData, namespace: str = "") -> None:
        self._namespace = namespace
        base = StretchBaseGroup(mj_data, namespace=namespace)
        super().__init__(
            mj_data,
            {
                "base": base,
                "lift": StretchLiftGroup(mj_data, base, namespace=namespace),
                "arm": StretchTelescopingArmGroup(mj_data, base, namespace=namespace),
                "wrist": StretchWristGroup(mj_data, base, namespace=namespace),
                "gripper": StretchGripperGroup(mj_data, base, namespace=namespace),
            },
        )

    @property
    def name(self) -> str:
        return f"{self._namespace}stretch4"

    @property
    def base(self) -> StretchBaseGroup:
        return self._move_groups["base"]

    def get_joint_position(self, move_group_ids: list[str]) -> np.ndarray:
        """Joint positions of several move groups, concatenated in order."""
        return np.concatenate(
            [self.get_move_group(mg_id).joint_pos.copy() for mg_id in move_group_ids]
        )

    def distance_to(self, move_group_ids: list[str], target_pose: list) -> float:
        """Combined position/heading error against an (x, y, theta) target.

        Not part of the `RobotView` base class -- MolmoSpaces defines it only on
        `RBY1RobotView` -- but `AStarPlannerPolicy` calls it on whatever view it
        is given to decide when a navigation waypoint has been reached. Stretch's
        holonomic base takes the planner's commands unmodified, so matching
        RBY1's definition here is what makes that policy usable as the
        navigation baseline. The yaw term is wrapped so a target just across the
        +-pi boundary does not read as a two-revolution error.
        """
        assert len(target_pose) == 3, f"Expected [x, y, theta] pose, got {target_pose}"
        current = self.get_joint_position(move_group_ids)
        return float(
            np.linalg.norm(
                np.array(
                    [
                        current[0] - target_pose[0],
                        current[1] - target_pose[1],
                        normalize_ang_error(current[2] - target_pose[2]),
                    ]
                )
            )
        )

    def is_close_to(
        self, move_group_ids: list[str], target_pose: list, threshold: float = 0.05
    ) -> bool:
        """Whether `distance_to` is under `threshold`. See `distance_to`."""
        return self.distance_to(move_group_ids, target_pose) < threshold

    def get_jacobian(self, move_group_id: str, input_move_group_ids: list[str]) -> np.ndarray:
        """Jacobian of one group's leaf frame w.r.t. the joints of several groups.

        Reimplemented rather than inherited because the telescoping arm owns four
        MuJoCo DOFs but exposes one, so its four segment columns have to be
        folded into the single column that corresponds to commanding total
        extension. `StretchTelescopingArmGroup` defines that coordinate as the
        *sum* of the segments and distributes a commanded extension `a` as
        `a / 4` per segment, so the chain rule gives the **mean** of the four
        columns, not their sum. (Summing would overstate the arm's sensitivity
        fourfold, which a damped least-squares step reads as "the arm is four
        times more effective than it is" and under-corrects accordingly.)

        `MlSpacesKinematics.ik()` walks the returned columns using each group's
        `vel_dim`, which is why the widths have to line up like this.
        """
        full_jacobian = self._move_groups[move_group_id].get_jacobian()
        columns = []
        for mg_id in input_move_group_ids:
            move_group = self._move_groups[mg_id]
            group_columns = full_jacobian[:, move_group._joint_veladr]
            if isinstance(move_group, StretchTelescopingArmGroup):
                group_columns = group_columns.mean(axis=1, keepdims=True)
            columns.append(group_columns)
        return np.concatenate(columns, axis=1)
