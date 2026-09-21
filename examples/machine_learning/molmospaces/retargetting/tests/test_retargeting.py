"""
Does the Franka-to-Stretch retargeting actually put Stretch's gripper where the Franka's is?

`policies/franka_retarget.py` claims a chain of six transforms -- a virtual
Franka's FK, a mount pose at Stretch's feet, a tool-convention rotation, and a
five-DOF-plus-base IK -- turns seven Franka joint targets into Stretch joint
targets that land the gripper in the same place. Every piece of that is argued
for in comments and none of it was measured end to end. The failure mode it
leaves open is the quiet one: a transform composed in the wrong order, or a tool
rotation off by a quarter turn, still produces plausible Stretch motion, and the
only symptom is a policy that scores badly for reasons that get blamed on the
camera.

So this compares against a real Franka rather than against the maths. Two
MuJoCo instances of the *same* kitchen are built -- one with Stretch standing
where the mini benchmark stands it, one with a Franka Droid bolted to the
pedestal the retargeting imagines at Stretch's feet -- and every check drives
both from one command:

    a Franka arm command q
      -> Franka instance:  set q, read `gripper/grasp_site` in the world
      -> Stretch instance: retarget q, read `grasp_center_link` in the world
      -> the two must be the same point, and the same frame once
         `FRANKA_TO_STRETCH_TOOL` is applied

The Franka instance is the ground truth, which is the whole reason it is here:
it is a real arm in a real scene, so it cannot agree with a transposed mount
pose or a mirrored tool frame out of politeness.

Two things are deliberately *not* tested. There is no physics -- joint positions
are written and `mj_forward` run, because the retargeting is kinematics and a
settling transient would only add noise to what is being measured (the
controllers that track these targets are tested by `tests/test_movement.py`).
And there is no policy: the commands are generated here, so a failure is the
retargeting rather than the checkpoint.

What is tested is the envelope too. Stretch's lift runs out of travel about 10cm
below the Franka's home tool height, and above that ceiling the retargeting
cannot win -- `test_above_the_lift_ceiling...` pins that shortfall to the number
`measure_tool_height_offset()` reports, so the limitation stays documented and a
*change* in it still fails.

    # the checks
    pytest examples/machine_learning/molmospaces/retargetting/tests/test_retargeting.py

    # the same commands, rendered: both robots, both tool frames, side by side
    python -m examples.machine_learning.molmospaces.retargetting.tests.test_retargeting \
        --visualize
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import click

if sys.platform == "linux":
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from mujoco import MjData, MjModel, MjSpec  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402

from examples.machine_learning.molmospaces.policies import franka_retarget as fr  # noqa: E402
from examples.machine_learning.molmospaces.retargetting import mini_benchmark  # noqa: E402
from examples.machine_learning.molmospaces.retargetting.diagnose import (  # noqa: E402
    build_standing_robot,
)
from examples.machine_learning.molmospaces.retargetting.setups import (  # noqa: E402
    FRANKA_LINK0_HEIGHT,
)

# =============================================================================
# What counts as "the same pose"
# =============================================================================

POSITION_TOLERANCE_M = 0.005
"""
How far Stretch's tool centre may sit from the Franka's, for a command inside
Stretch's workspace.

Both IKs are iterative and stop on their own residual, so exact agreement is not
on offer: the observed spread over the waypoints below is 0.02mm to 1.9mm, and
5mm is that with room for solver jitter while still being far under the 17cm a
mis-stated pedestal height costs or the 6cm a dropped tool rotation does (both
measured, by breaking them). It is *not* a claim about the real
robot, which has to track these targets through its controllers.
"""

ORIENTATION_TOLERANCE_RAD = 0.02
"""
The same for the tool frame's orientation -- about 1.1 degrees.

Stretch's five manipulator DOFs cannot span a 6-DOF pose error, and
`StretchArmIK` resolves that by serving position first, so orientation is the
half that gives when a target is awkward. The waypoints below are all poses the
wrist can actually hold (observed error under 0.001 rad) precisely so that this
tolerance stays tight enough to catch a rotation composed on the wrong side.
"""

FRANKA_IK_TOLERANCE_M = 0.001
"""
How exactly the *virtual* Franka must reach a waypoint for it to be used as a
command.

A waypoint is only a fair test if the Franka can hold it: a pose the Franka's own
IK misses is not a command any policy would emit, and comparing Stretch against
a Franka that is somewhere else is a test of nothing. Asserted rather than
assumed, so that a waypoint drifting out of the Franka's reach fails loudly here
instead of silently weakening `test_stretch_reaches_the_commanded_pose`.
"""

TRAJECTORY_ORIENTATION_TOLERANCE_RAD = 0.12
"""
How far the grasp may lag along a *continuously commanded* path, as opposed to at
a pose solved from the snap.

Looser than `ORIENTATION_TOLERANCE_RAD`, and the gap is a real property of the
solver rather than a fudge. Seeded from the previous step instead of from the
snap, the IK can walk into a corner the same target does not have when solved
fresh: one waypoint here (`yaw_in`, a 45-degree twist of the tool) used to settle
0.43 rad short and stay there, with both `wrist_yaw` and `wrist_roll` pinned at
their limits -- unrecoverable by iteration, measured at 80, 240, 800 and 3000.

`franka_retarget.JAW_FLIP` is what closed most of that: the same grasp held half a
turn about the approach axis puts the wrist back in open range, and the worst
lag on this path is now 0.059 rad. The remaining tolerance is roughly twice that,
which leaves room for solver jitter while still being far too tight for the old
stall to come back unnoticed.

What made even the 0.43 rad tolerable is that it never cost *position*, which
this test asserts separately and at the full tolerance. A regression that started
trading position away would fail there rather than here.
"""


GRASP_HEIGHT_DROP_M = 0.155
"""
How far below the Franka's home tool height the waypoints are centred.

The Franka's home puts its grasp site at 1.185m, which is above the ceiling
Stretch's lift can reach (see `test_above_the_lift_ceiling_the_error_is_the_lift_shortfall`),
so waypoints centred there would measure the lift's travel rather than the
retargeting. 0.155m down is 1.03m -- roughly 9cm over this kitchen's counter,
which is where a grasp in this benchmark actually happens, and comfortably
inside the lift's range so the arm has somewhere to go in both directions.
"""


@dataclass(frozen=True)
class Waypoint:
    """One commanded tool pose, as an offset from the centred grasp pose.

    Translations are in the *robot's* frame rather than the world's, so a
    waypoint means the same thing ("15cm further from the robot") regardless of
    which way the episode happens to have the base facing: `forward` is the axis
    Stretch's arm telescopes along and the Franka reaches along, `left` is across
    it, `up` is the world's.

    `tool_rotation` is applied on the right -- in the tool's own frame -- because
    that is how a policy's orientation command reads: tilt the gripper, not the
    room.
    """

    label: str
    forward: float = 0.0
    left: float = 0.0
    up: float = 0.0
    tool_rotation: tuple[str, float] | None = None
    robotiq: float = 0.0
    """
    The gripper command that goes with this pose, in the Robotiq's own 0-255.

    0 is *open* on the Robotiq and 255 shut, which is the opposite of Stretch's
    finger angle -- the flip lives in `retarget_robotiq_ctrl`. Defaulting to 0
    means the waypoints approach with an open hand, and the two that close are
    the ones that say so.
    """


WAYPOINTS: tuple[Waypoint, ...] = (
    Waypoint("centred"),
    # Reach: in and out along the telescoping axis, and across it. The base
    # joins the IK for the far ones, which is the point -- Stretch's arm alone
    # reaches a narrow corridor (see `StretchArmIK`), so a retargeting that
    # tracks these is one the holonomic base is contributing to correctly.
    Waypoint("reach_out", forward=0.15),
    Waypoint("reach_in", forward=-0.15),
    Waypoint("across_left", left=0.15),
    Waypoint("across_right", left=-0.15),
    Waypoint("out_and_left", forward=0.25, left=0.10),
    Waypoint("out_and_right", forward=0.30, left=-0.20),
    # Height, and the gripper: descend and close, which is the shape of a grasp.
    # `lowest` closing is what puts the gripper mapping in the rendered path --
    # without a gripper command anywhere in this list both robots would simply
    # sit at their own model defaults, which are not the same (Stretch's
    # `init_qpos` is shut, the Franka's is open) and would read as the
    # retargeting disagreeing when nothing had been asked of it.
    Waypoint("lower", up=-0.10),
    Waypoint("lowest", up=-0.20, robotiq=255.0),
    Waypoint("lifted_closed", up=-0.05, robotiq=255.0),
    # Orientation: the wrist. Each of these is a quarter of the way to a
    # different axis, which is what a tool-frame rotation composed on the wrong
    # side or about the wrong axis shows up in.
    Waypoint("tilt_down", tool_rotation=("y", 20.0)),
    Waypoint("tilt_up", tool_rotation=("y", -20.0)),
    Waypoint("roll_left", tool_rotation=("x", 20.0)),
    Waypoint("roll_right", tool_rotation=("x", -20.0)),
    Waypoint("yaw_in", tool_rotation=("z", 45.0)),
    Waypoint("yaw_out", tool_rotation=("z", -30.0)),
)
"""
The commanded poses. Chosen to span what a grasp in this kitchen asks for --
reach, height and wrist angle, one at a time -- rather than to be a dense grid:
each one is a separate `pytest` case, and each costs two IK solves.
"""


def pose_difference(reached: np.ndarray, expected: np.ndarray) -> tuple[float, float]:
    """How far apart two 4x4 poses are: (metres, radians). Exact, no symmetry allowed."""
    reached = np.asarray(reached, dtype=float)
    expected = np.asarray(expected, dtype=float)
    position = float(np.linalg.norm(reached[:3, 3] - expected[:3, 3]))
    rotation = float(
        np.linalg.norm(R.from_matrix(reached[:3, :3] @ expected[:3, :3].T).as_rotvec())
    )
    return position, rotation


def grasp_difference(reached: np.ndarray, expected: np.ndarray) -> tuple[float, float]:
    """`pose_difference`, but counting a half-turned jaw as having arrived.

    The retargeting is allowed to hold the gripper half a turn about its approach
    axis when that is the branch Stretch's wrist can reach -- see
    `franka_retarget.JAW_FLIP`. Both orientations close the same jaw on the same
    object along the same line, so measuring against the commanded orientation
    alone would score a successful grasp as being a whole pi out.

    So this is the question the assertions actually want answered: is Stretch's
    gripper in a position and attitude to make the commanded grasp? Position is
    compared exactly, as always -- the flip moves no points. `pose_difference`
    stays available, and the checks that compare two *Frankas* keep using it,
    since nothing there is allowed any latitude.
    """
    position, upright = pose_difference(reached, expected)
    _, flipped = pose_difference(reached, expected @ fr.JAW_FLIP)
    return position, min(upright, flipped)


# =============================================================================
# The two simulations
# =============================================================================


def build_franka_on_the_pedestal(
    mount_pose: np.ndarray,
) -> tuple[MjModel, MjData, object, str]:
    """A Franka Droid in the mini benchmark's kitchen, standing at `mount_pose`.

    `mount_pose` is where `franka_retarget` *imagines* the Franka: the 4x4 that
    `franka_mount_pose_from_base` puts at `fr3_link0`. Spawning a real one there
    is the only way to check that imagination against a robot, so the height
    handed to `add_robot_to_scene` is the mount's z less the pedestal it adds
    underneath -- exactly what `setups.franka_episode_override` does with
    `FRANKA_LINK0_HEIGHT`, and for the same reason: get it wrong and `fr3_link0`
    sits 0.58m off, which is a discrepancy that would be blamed on the
    retargeting.

    Returns the model, its data, a `FrankaDroidRobotView` over it, and the
    namespace. The Franka's base is a mocap body and nothing here moves it, so
    the arm is the only thing that changes between commands.
    """
    from molmo_spaces.configs.robot_configs import FrankaRobotConfig
    from molmo_spaces.robots.franka import FrankaRobot
    from molmo_spaces.robots.robot_views.franka_droid_view import FrankaDroidRobotView
    from molmo_spaces.utils.lazy_loading_utils import (
        install_scene_with_objects_and_grasps_from_path,
    )

    scene = mini_benchmark.house_scene_path()
    install_scene_with_objects_and_grasps_from_path(str(scene))
    spec = MjSpec.from_file(str(scene))

    config = FrankaRobotConfig()
    pedestal = float(config.base_size[2]) if config.base_size else 0.0
    mount_pose = np.asarray(mount_pose, dtype=float)
    FrankaRobot.add_robot_to_scene(
        config,
        spec,
        prefix=config.robot_namespace,
        pos=[float(mount_pose[0, 3]), float(mount_pose[1, 3]), FRANKA_LINK0_HEIGHT - pedestal],
        quat=list(R.from_matrix(mount_pose[:3, :3]).as_quat(scalar_first=True)),
    )

    model = spec.compile()
    data = MjData(model)
    view = FrankaDroidRobotView(data, config.robot_namespace)
    for group, values in config.init_qpos.items():
        if group in view.move_group_ids():
            view.get_move_group(group).joint_pos = values
    mujoco.mj_forward(model, data)
    return model, data, view, config.robot_namespace


FRANKA_GRIPPER_SETTLE_STEPS = 300
"""
How many physics steps the Robotiq needs to reach a commanded opening.

The Robotiq 2F-85 is a *linkage*, not a pair of jaws: six hinges per hand held
together by three equality constraints and a tendon, of which the move group
exposes only the two driver joints. Equality constraints are enforced by the
constraint solver during `mj_step`, and `mj_forward` does not run it -- so
writing the drivers and calling `mj_forward`, which is how everything else here
poses a robot, leaves the spring links and followers where they were and the pads
never move. The hand reports itself fully open at every driver angle.

That is the one place in this file where kinematics is not enough, so this is the
one place physics runs. Measured on the compiled model: 300 steps takes the hand
from open to shut (0.999 -> 0.000 of its travel) and back, and costs about half a
second, which is why `RetargetRig` settles each command once and caches the
result rather than doing it per frame.
"""


def opening_fraction(move_group) -> float:
    """How far open a gripper is, as a fraction of its own travel.

    The two grippers are different hardware with different spans -- the Robotiq
    2F-85 and Stretch's fingers do not open to the same number of millimetres --
    so "the same command produced the same grasp" can only be asked of them in
    these terms, not in metres.
    """
    closed, wide = move_group.inter_finger_dist_range
    return float((move_group.inter_finger_dist - closed) / (wide - closed))


@dataclass
class Reached:
    """Where one command put each robot, and by how much they disagree."""

    command: np.ndarray
    """The seven Franka joint angles both robots were driven from."""

    franka_pose: np.ndarray
    """The Franka's `gripper/grasp_site`, in the world. The ground truth."""

    stretch_pose: np.ndarray
    """Stretch's `grasp_center_link`, in the world, after retargeting."""

    expected_pose: np.ndarray
    """Where the retargeting *asked* Stretch to be: `franka_pose` in Stretch's tool convention."""

    position_error: float
    orientation_error: float

    residual: np.ndarray
    """`StretchArmIK`'s own 6-vector -- what it knew it could not reach."""

    jaw_flipped: bool = False
    """Whether the arm took the half-turned jaw to get here. See `franka_retarget.JAW_FLIP`."""

    franka_opening: float = 0.0
    """How far open the Franka's gripper is, as a fraction of its own travel."""

    stretch_opening: float = 0.0
    """The same for Stretch's, so the two are comparable across different hardware."""

    @property
    def vertical_error(self) -> float:
        """The signed z part of the miss. Positive when Stretch came up short."""
        return float(self.expected_pose[2, 3] - self.stretch_pose[2, 3])


class RetargetRig:
    """Both robots, in the same kitchen, driven from one Franka command.

    Holds a Stretch instance, a Franka instance on the virtual pedestal, and the
    `FrankaOnStretchView` under test. Reused across checks because building it
    compiles two copies of a furnished house, which is nearly all the runtime of
    this file -- so `command()` restores Stretch to its standing pose first, and
    every check is independent of the order they run in. That restore matters:
    `StretchArmIK` seeds from wherever the robot currently is, so without it a
    waypoint would be solved from the previous waypoint's answer and the numbers
    would depend on collection order.

    The mount pose is set once, from where Stretch is standing, and never
    re-anchored -- which is what `MolmoBotDroidPolicy._build_proxy` does per
    episode, deliberately, so that the base driving during a solve does not drag
    the frame the policy's actions are interpreted in along with it.
    """

    def __init__(self, include_base: bool = True, target_z_offset: float = 0.0) -> None:
        self.model, self.data, self.view, self.namespace = build_standing_robot()
        self._home_qpos = self.data.qpos.copy()

        base_xytheta = np.asarray(self.view.get_move_group("base").joint_pos, dtype=float)
        self.mount_pose = fr.franka_mount_pose_from_base(base_xytheta)
        self.proxy = fr.FrankaOnStretchView(
            self.view,
            self.namespace,
            self.mount_pose,
            include_base=include_base,
            target_z_offset=target_z_offset,
        )

        (
            self.franka_model,
            self.franka_data,
            self.franka_view,
            self.franka_namespace,
        ) = build_franka_on_the_pedestal(self.mount_pose)

        # The Robotiq's own joints, and a scratch `MjData` to settle them in. The
        # hand's linkage is internal to the hand, so a settled configuration
        # depends only on the command and can be cached and reused at any arm
        # pose; settling on scratch data keeps the stepping away from the arm
        # whose pose is being measured. See `FRANKA_GRIPPER_SETTLE_STEPS`.
        self._franka_gripper_qposadr = np.array(
            [
                self.franka_model.jnt_qposadr[joint]
                for joint in range(self.franka_model.njnt)
                if "gripper" in self.franka_model.joint(joint).name
            ]
        )
        self._franka_gripper_scratch = MjData(self.franka_model)
        self._settled_gripper: dict[float, np.ndarray] = {}

        # The robot's own axes, for `Waypoint`. Taken from the base's reported
        # yaw rather than from its 4x4, for the reason `franka_mount_pose_from_base`
        # gives: a base frame mid-transient can be pitched, and a "forward" that
        # pointed into the floor would silently move every waypoint.
        yaw = float(base_xytheta[2])
        self.forward = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        self.left = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
        self.up = np.array([0.0, 0.0, 1.0])

    # -- the poses -----------------------------------------------------------

    @property
    def franka_home_command(self) -> np.ndarray:
        """The Franka's own `init_qpos` arm configuration."""
        return self.proxy.franka.init_qpos.copy()

    def centred_pose(self) -> np.ndarray:
        """The pose the waypoints are offsets from: the Franka's home, lowered to counter height."""
        home = self.proxy.franka_tool_pose_to_world(
            self.proxy.franka.fk(self.franka_home_command)
        )
        pose = home.copy()
        pose[:3, 3] = home[:3, 3] - GRASP_HEIGHT_DROP_M * self.up
        return pose

    def waypoint_pose(self, waypoint: Waypoint) -> np.ndarray:
        """The world-frame Stretch tool pose a waypoint asks for."""
        centred = self.centred_pose()
        pose = centred.copy()
        pose[:3, 3] = (
            centred[:3, 3]
            + waypoint.forward * self.forward
            + waypoint.left * self.left
            + waypoint.up * self.up
        )
        if waypoint.tool_rotation is not None:
            axis, degrees = waypoint.tool_rotation
            pose[:3, :3] = centred[:3, :3] @ R.from_euler(axis, degrees, degrees=True).as_matrix()
        return pose

    def franka_command_for(self, pose: np.ndarray) -> np.ndarray:
        """The seven Franka joint targets a policy aiming at `pose` would emit.

        The inverse of the observation half of the retargeting: `pose` is taken
        back through the tool correction and the mount into the Franka's own base
        frame, and the virtual Franka's IK reads off the joint angles. Seeded
        from the Franka's home for every waypoint rather than from the previous
        answer, so a command depends only on its own waypoint.
        """
        target = self.proxy.stretch_tool_pose_to_franka(pose)
        return self.proxy.franka.ik(target, self.franka_home_command, iterations=300)

    def settled_franka_gripper(self, robotiq: float) -> np.ndarray:
        """The Robotiq's joint angles once it has closed on a 0-255 command.

        Stepped rather than written, because writing does not work on this hand
        (`FRANKA_GRIPPER_SETTLE_STEPS` says why), and cached per command because
        stepping costs half a second. The arm is held at the Franka's home pose
        and commanded there while the hand settles, so nothing sags into the
        counter and adds contacts to what should be a free-space motion.
        """
        key = round(float(robotiq), 3)
        if key not in self._settled_gripper:
            scratch = self._franka_gripper_scratch
            scratch.qpos[:] = self.franka_data.qpos
            scratch.qvel[:] = 0.0
            scratch.ctrl[:] = 0.0
            view = type(self.franka_view)(scratch, self.franka_namespace)
            view.get_move_group("arm").joint_pos = self.franka_home_command
            view.get_move_group("arm").ctrl = self.franka_home_command
            view.get_move_group("gripper").ctrl = [key]
            for _ in range(FRANKA_GRIPPER_SETTLE_STEPS):
                mujoco.mj_step(self.franka_model, scratch)
            self._settled_gripper[key] = scratch.qpos[self._franka_gripper_qposadr].copy()
        return self._settled_gripper[key]

    # -- driving them --------------------------------------------------------

    def restore(self) -> None:
        """Put Stretch back at the configuration a rollout's first action is solved from.

        Three steps, in the order a rollout does them: back to where the robot
        spawned, re-seed the retargeting and re-centre the base's leash on it,
        then `snap_to_franka_joint_pos()` -- which puts Stretch at the Franka's
        *home tool pose* rather than its own stowed one.

        The snap is not tidiness. `StretchArmIK` takes 80 damped steps capped at
        5cm and 0.2 rad each, seeded from wherever the robot currently is, so it
        converges on a target near that configuration and merely gets closer to
        one far from it. Stowed, Stretch has its gripper tucked at its side
        pointing along the base, and every waypoint here is half a metre away and
        a quarter turn around -- solved from there the IK runs out of iterations
        with position roughly right and orientation still 0.5 rad out, which
        would show up in these assertions as retargeting error when it is
        nothing of the sort. A rollout never asks that of it: the policy snaps
        once before its first action and every command after that is a step from
        the pose the previous one reached. Starting each check from the same snap
        is that condition, without making one check depend on the last.
        """
        self.data.qpos[:] = self._home_qpos
        mujoco.mj_forward(self.model, self.data)
        self.proxy.reset()
        self.proxy.snap_to_franka_joint_pos()
        # No gripper command here on purpose. `snap_to_franka_joint_pos` now opens
        # Stretch's hand to match the Franka's home, and every check should run
        # against that rather than against a state the harness arranged -- an
        # earlier version of this method opened the hand itself, which would hide
        # a regression in the snap behind the test's own setup.

    def command_gripper(self, robotiq: float) -> None:
        """Put both grippers at one Robotiq 0-255 command, without touching the arms."""
        self.franka_data.qpos[self._franka_gripper_qposadr] = self.settled_franka_gripper(robotiq)
        mujoco.mj_forward(self.franka_model, self.franka_data)
        stretch_gripper = self.view.get_move_group("gripper")
        stretch_gripper.joint_pos = self.proxy.retarget_robotiq_ctrl(robotiq)
        stretch_gripper.ctrl = stretch_gripper.joint_pos
        mujoco.mj_forward(self.model, self.data)

    def command(
        self, joint_pos: np.ndarray, robotiq: float = 0.0, restore: bool = True
    ) -> Reached:
        """Drive both robots from one Franka command -- arm and gripper -- and compare them.

        Joint positions are *written*, not commanded: `mj_forward` then puts each
        robot exactly at the configuration its IK asked for, which is what makes
        this a measurement of the retargeting rather than of two controllers'
        settling behaviour.

        `robotiq` is the gripper half of the command, in the Robotiq's 0-255. Both
        robots are driven from it -- Stretch through `retarget_robotiq_ctrl`, which
        is the code under test, and the Franka through `settled_franka_gripper`.
        Driving both is what makes the pair comparable at all: left to their own
        `init_qpos` the Franka sits open and Stretch sits shut, which looks like a
        retargeting failure and is really just two models' defaults disagreeing
        about a command neither was given.
        """
        if restore:
            self.restore()
        joint_pos = np.asarray(joint_pos, dtype=float).reshape(-1)[: fr.VirtualFranka.N_JOINTS]

        franka_arm = self.franka_view.get_move_group("arm")
        franka_arm.joint_pos = joint_pos
        self.franka_data.qpos[self._franka_gripper_qposadr] = self.settled_franka_gripper(robotiq)
        mujoco.mj_forward(self.franka_model, self.franka_data)
        franka_pose = np.asarray(franka_arm.leaf_frame_to_world, dtype=float)

        targets = self.proxy.retarget_franka_joint_pos(joint_pos)
        targets["gripper"] = self.proxy.retarget_robotiq_ctrl(robotiq)
        for group, values in targets.items():
            move_group = self.view.get_move_group(group)
            move_group.joint_pos = values
            move_group.ctrl = values
        mujoco.mj_forward(self.model, self.data)
        stretch_pose = np.asarray(
            self.view.get_move_group("wrist").leaf_frame_to_world, dtype=float
        )

        # Where the retargeting asked Stretch to go, expressed from the Franka
        # that is actually standing there: the same point, rotated into Stretch's
        # tool convention, and raised by whatever `target_z_offset` is set to.
        expected_pose = franka_pose.copy()
        expected_pose[:3, :3] = franka_pose[:3, :3] @ fr.FRANKA_TO_STRETCH_TOOL
        expected_pose[2, 3] += self.proxy.target_z_offset

        position_error, orientation_error = grasp_difference(stretch_pose, expected_pose)
        return Reached(
            command=joint_pos,
            franka_pose=franka_pose,
            stretch_pose=stretch_pose,
            expected_pose=expected_pose,
            position_error=position_error,
            orientation_error=orientation_error,
            residual=self.proxy.last_residual.copy(),
            jaw_flipped=bool(self.proxy.jaw_flipped),
            franka_opening=opening_fraction(self.franka_view.get_move_group("gripper")),
            stretch_opening=opening_fraction(self.view.get_move_group("gripper")),
        )

    def command_waypoint(self, waypoint: Waypoint) -> Reached:
        """`command()` for the Franka command that reaches a waypoint."""
        return self.command(
            self.franka_command_for(self.waypoint_pose(waypoint)), robotiq=waypoint.robotiq
        )


# =============================================================================
# The checks
# =============================================================================


@pytest.fixture(scope="module")
def rig() -> RetargetRig:
    """One rig for the whole module. See `RetargetRig` on why sharing it is safe."""
    return RetargetRig()


def test_the_virtual_franka_agrees_with_a_real_one(rig: RetargetRig) -> None:
    """`VirtualFranka.fk` plus the mount pose must be where a real Franka's tool is.

    The first link in the chain, and the one with nothing else to catch it: the
    virtual Franka is compiled from the same `model.xml` but never stepped and
    never placed in the scene, so the only thing asserting that `mount_pose @
    fk(q)` is a pose *in the room* is a Franka standing in the room. Checked at
    several commands rather than at home alone, because a mount pose composed in
    the wrong order agrees at some configurations and not others.
    """
    for waypoint in (WAYPOINTS[0], WAYPOINTS[1], WAYPOINTS[5], WAYPOINTS[-1]):
        command = rig.franka_command_for(rig.waypoint_pose(waypoint))

        rig.franka_view.get_move_group("arm").joint_pos = command
        mujoco.mj_forward(rig.franka_model, rig.franka_data)
        real = np.asarray(
            rig.franka_view.get_move_group("arm").leaf_frame_to_world, dtype=float
        )
        virtual = rig.mount_pose @ rig.proxy.franka.fk(command)

        position, orientation = pose_difference(real, virtual)
        assert position < 1e-6, (
            f"the virtual Franka's tool is {position * 1000:.3f}mm from the real one's at "
            f"waypoint {waypoint.label!r}. Both are the same model at the same joint angles, "
            f"so this is `franka_mount_pose_from_base` or `FRANKA_PEDESTAL_HEIGHT` against "
            f"`FRANKA_LINK0_HEIGHT` and the pedestal `add_robot_to_scene` adds."
        )
        assert orientation < 1e-6, (
            f"the virtual Franka's tool frame is rotated {orientation:.6f} rad from the real "
            f"one's at waypoint {waypoint.label!r}."
        )


@pytest.mark.parametrize("waypoint", WAYPOINTS, ids=lambda waypoint: waypoint.label)
def test_stretch_reaches_the_commanded_pose(rig: RetargetRig, waypoint: Waypoint) -> None:
    """The whole point: one Franka command, and both grippers end up in the same place.

    The comparison is against the Franka instance's own grasp site, not against
    the pose the waypoint asked for, so this stays a test of the retargeting even
    if a waypoint is one the Franka holds imperfectly -- and the assertion above
    it keeps the waypoints honest about that.
    """
    requested = rig.waypoint_pose(waypoint)
    command = rig.franka_command_for(requested)

    reached = rig.command(command)

    franka_miss, _ = pose_difference(reached.franka_pose, requested)
    assert franka_miss < FRANKA_IK_TOLERANCE_M, (
        f"waypoint {waypoint.label!r} is not a pose the Franka itself holds -- its own IK "
        f"missed by {franka_miss * 1000:.1f}mm, so it is not a command a policy would emit "
        f"and comparing Stretch against it tests nothing. Move the waypoint back into the "
        f"Franka's reach."
    )
    assert reached.position_error < POSITION_TOLERANCE_M, (
        f"Stretch's tool centre is {reached.position_error * 1000:.1f}mm from the Franka's at "
        f"waypoint {waypoint.label!r} (vertical component {reached.vertical_error * 1000:+.1f}mm; "
        f"the IK's own residual was {np.linalg.norm(reached.residual[:3]) * 1000:.1f}mm). "
        f"A residual of about this size means Stretch cannot reach the pose and the waypoint "
        f"has left its workspace; a much smaller residual means the IK thinks it succeeded, "
        f"which points at the transform chain instead."
    )
    assert reached.orientation_error < ORIENTATION_TOLERANCE_RAD, (
        f"Stretch's tool frame is {reached.orientation_error:.4f} rad from the Franka's at "
        f"waypoint {waypoint.label!r}, rotated into Stretch's convention. About 1.57 rad here "
        f"is `FRANKA_TO_STRETCH_TOOL` applied on the wrong side or about the wrong axis."
    )


def _unit(vector: np.ndarray) -> np.ndarray:
    """`vector`, normalised."""
    vector = np.asarray(vector, dtype=float)
    return vector / np.linalg.norm(vector)


def _gripper_axes(model, data, namespace: str, parts: tuple[str, str, str]):
    """A gripper's (finger-separation, approach) axes in the world, from its bodies.

    `parts` names the two fingers and the wrist they hang off. Both axes are read
    off body positions rather than out of a tool frame, which is the point: these
    are where the metal is, so they are true whatever convention either model
    expresses its tool frame in.

    Separation is finger to finger. Approach is the wrist to the point between
    the fingers -- the direction the hand reaches -- defined identically for both
    robots so that comparing the two compares the grippers rather than two
    different definitions.
    """
    left, right, wrist = (f"{namespace}{part}" for part in parts)
    position = lambda name: np.asarray(data.xpos[model.body(name).id], dtype=float)  # noqa: E731
    fingers = (position(left) + position(right)) / 2.0
    return _unit(position(left) - position(right)), _unit(fingers - position(wrist))


FRANKA_GRIPPER_BODIES = ("gripper/left_pad", "gripper/right_pad", "gripper/base")
"""The Robotiq's two pads, and the hand they reach away from."""

STRETCH_GRIPPER_BODIES = (
    "gripper_fingertip_left_link",
    "gripper_fingertip_right_link",
    "wrist_roll_link",
)
"""Stretch's two fingertips, and the wrist they reach away from."""


@pytest.mark.parametrize(
    "waypoint", WAYPOINTS[::3] + (WAYPOINTS[10], WAYPOINTS[14]), ids=lambda w: w.label
)
def test_both_grippers_end_up_pointing_the_same_way(rig: RetargetRig, waypoint: Waypoint) -> None:
    """The two grippers have to be oriented alike in the room, measured off their fingers.

    This exists because `test_stretch_reaches_the_commanded_pose` cannot see a
    wrong `FRANKA_TO_STRETCH_TOOL`: it builds the pose it expects by applying
    that same constant, so a tool rotation that is inverted or about the wrong
    axis moves the expectation and the result together and the assertion passes.
    Confirmed by breaking it -- `+90` degrees instead of `-90` leaves that test
    green.

    So nothing here goes through the tool frame at all. Both axes come from body
    positions in the two models: where each gripper's fingers are, and which way
    they reach away from its wrist. A gripper that agrees with another on both is
    oriented the same way in the room, whatever either one calls its tool frame.

    The approach axis is signed -- the fingers must point the *same* way, not
    merely along the same line, which is exactly what an inverted rotation gets
    wrong (it scores -1.0 here). The finger-separation axis is compared up to
    sign, because a parallel gripper turned half a turn about its own approach
    grasps identically and neither model's "left" has to be the other's.
    """
    rig.command(rig.franka_command_for(rig.waypoint_pose(waypoint)))

    franka_separation, franka_approach = _gripper_axes(
        rig.franka_model, rig.franka_data, rig.franka_namespace, FRANKA_GRIPPER_BODIES
    )
    stretch_separation, stretch_approach = _gripper_axes(
        rig.model, rig.data, rig.namespace, STRETCH_GRIPPER_BODIES
    )

    approach = float(np.dot(franka_approach, stretch_approach))
    assert approach > math.cos(ORIENTATION_TOLERANCE_RAD), (
        f"at waypoint {waypoint.label!r} the two grippers reach in directions "
        f"{math.degrees(math.acos(np.clip(approach, -1, 1))):.1f} degrees apart. "
        f"Near -1.0 means `FRANKA_TO_STRETCH_TOOL` is inverted -- Stretch is reaching away "
        f"from what the policy is reaching towards; near 0.0 means it is about the wrong "
        f"axis, or missing."
    )
    separation = abs(float(np.dot(franka_separation, stretch_separation)))
    assert separation > math.cos(ORIENTATION_TOLERANCE_RAD), (
        f"at waypoint {waypoint.label!r} the grippers' fingers separate along lines "
        f"{math.degrees(math.acos(np.clip(separation, -1, 1))):.1f} degrees apart, so one "
        f"would be grasping across what the other grasps along -- a rotation about the "
        f"approach axis that `FRANKA_TO_STRETCH_TOOL` is not applying."
    )


def test_proprioception_reports_where_stretch_actually_is(rig: RetargetRig) -> None:
    """The observation half: seven Franka angles that describe Stretch's real gripper pose.

    `franka_joint_pos()` is what the policy reads, and it has to close the loop
    -- FK of what it reports must land back on the tool pose Stretch is actually
    holding. If it does not, the policy is reading an echo of its own command
    (or, worse, a drifting IK branch) and cannot tell that a target was missed.

    Checked at a pose Stretch reaches *and* at the Franka's home pose, which it
    cannot: there the reported state must follow the gripper to where it really
    is rather than report the command back.
    """
    for label, command in (
        ("a reachable pose", rig.franka_command_for(rig.waypoint_pose(WAYPOINTS[0]))),
        ("the Franka's home pose", rig.franka_home_command),
    ):
        reached = rig.command(command)
        reported = rig.proxy.franka_joint_pos()
        round_tripped = rig.proxy.franka_tool_pose_to_world(rig.proxy.franka.fk(reported))

        position, orientation = pose_difference(round_tripped, reached.stretch_pose)
        assert position < POSITION_TOLERANCE_M, (
            f"at {label}, the seven joint angles `franka_joint_pos()` reports FK back to a "
            f"tool pose {position * 1000:.1f}mm from where Stretch's gripper actually is."
        )
        assert orientation < ORIENTATION_TOLERANCE_RAD, (
            f"at {label}, the reported arm state's tool frame is {orientation:.4f} rad from "
            f"Stretch's real one."
        )


def test_above_the_lift_ceiling_the_error_is_the_lift_shortfall(rig: RetargetRig) -> None:
    """The known limit, pinned: the Franka's home pose is above where Stretch's lift reaches.

    This is not a bug being tolerated, it is the fact `target_z_offset` exists
    for -- over a counter with the gripper pointing down, Stretch's grasp centre
    caps out about 10cm below the Franka's home. Asserted three ways so the
    limitation stays a documented number rather than folklore: the miss is real,
    it is *vertical* (the arm is short, not lost), and it is the same quantity
    `measure_tool_height_offset()` reports, which is what the policy config
    scales by `z_offset_fraction`.
    """
    reached = rig.command(rig.franka_home_command)
    shortfall = rig.proxy.measure_tool_height_offset()

    assert reached.position_error > POSITION_TOLERANCE_M, (
        "Stretch reached the Franka's home tool pose. That would be good news, but it "
        "contradicts `FrankaOnStretchView.target_z_offset`'s reason for existing -- if the "
        "lift's travel has changed, that docstring and `z_offset_fraction`'s default need "
        "revisiting together with this test."
    )
    horizontal = float(np.linalg.norm(reached.expected_pose[:2, 3] - reached.stretch_pose[:2, 3]))
    assert horizontal < POSITION_TOLERANCE_M, (
        f"the miss at the Franka's home pose is {horizontal * 1000:.1f}mm horizontal. It is "
        f"supposed to be the lift running out of travel, which is purely vertical; a "
        f"horizontal component means something other than the ceiling is wrong."
    )
    assert reached.vertical_error == pytest.approx(shortfall, abs=POSITION_TOLERANCE_M), (
        f"Stretch came up {reached.vertical_error * 1000:.1f}mm short vertically, but "
        f"`measure_tool_height_offset()` reports {shortfall * 1000:.1f}mm. These are the same "
        f"quantity measured two ways -- the offset the policy applies is computed from the "
        f"second, so it is the first that the gripper actually experiences."
    )


def test_the_target_height_offset_raises_the_pose_by_exactly_itself(rig: RetargetRig) -> None:
    """`target_z_offset` must be "stand the virtual Franka this much higher", and nothing else.

    Two claims from its docstring, both cheap to break by changing one of the two
    conversions without the other. First that the pair are inverses, so a
    commanded pose survives a round trip through them. Second that setting it
    lifts where Stretch goes by that much and leaves the rest of the pose alone
    -- it is the reason a retargeted grasp clears the countertop, so an offset
    that also nudged the pose sideways would be quietly aiming off the object.
    """
    offset = 0.06
    pose = rig.waypoint_pose(Waypoint("test", up=-0.10))

    rig.proxy.target_z_offset = 0.0
    try:
        command = rig.franka_command_for(pose)
        without = rig.command(command)

        round_tripped = rig.proxy.franka_tool_pose_to_world(
            rig.proxy.stretch_tool_pose_to_franka(pose)
        )
        position, orientation = pose_difference(round_tripped, pose)
        assert position < 1e-9 and orientation < 1e-9, (
            f"`stretch_tool_pose_to_franka` and `franka_tool_pose_to_world` are not inverses: "
            f"a round trip moved the pose {position * 1000:.3f}mm and {orientation:.6f} rad."
        )

        rig.proxy.target_z_offset = offset
        with_offset = rig.command(command)
    finally:
        rig.proxy.target_z_offset = 0.0

    rise = with_offset.stretch_pose[2, 3] - without.stretch_pose[2, 3]
    assert rise == pytest.approx(offset, abs=POSITION_TOLERANCE_M), (
        f"a target_z_offset of {offset * 100:.0f}cm moved Stretch's tool centre "
        f"{rise * 100:.1f}cm, not {offset * 100:.0f}cm. The lift has travel left at this "
        f"pose, so the offset should arrive in full."
    )
    sideways = float(
        np.linalg.norm(with_offset.stretch_pose[:2, 3] - without.stretch_pose[:2, 3])
    )
    assert sideways < POSITION_TOLERANCE_M, (
        f"the same offset also moved the tool {sideways * 1000:.1f}mm horizontally. It is a "
        f"height offset; a horizontal component means it is being applied in the Franka's "
        f"frame rather than the world's."
    )


GRIPPER_OPENING_TOLERANCE = 0.04
"""
How far apart the two grippers' openings may be, as a fraction of their own
travel -- 4% of the way from shut to wide.

Not in metres, because the two hands are different sizes; see
`opening_fraction`. It is not tighter because the two are not the same mechanism:
`retarget_robotiq_ctrl` is linear in the command, and Stretch's fingers are close
to linear in their own angle, but the Robotiq reaches its opening through a
linkage that is not. Measured across the range, the two agree to 0.004 at the
ends and bulge to 0.023 around the middle, which is that nonlinearity and nothing
else. 4% covers it with room to spare and is still nowhere near loose enough to
let an inverted mapping through -- that one misses by a full 1.0 at both ends.
"""


def test_the_hand_starts_where_a_droid_episode_starts_it(rig: RetargetRig) -> None:
    """At episode start both policies must read the same gripper state: open.

    The checkpoint reads its gripper through one number, `qpos["gripper"][0]`, in
    Robotiq driver units. Every DROID episode it was trained on begins with that
    hand open, and `franka_droid_policy` -- the Franka control condition -- starts
    at 0.003, the open end. Stretch's `Stretch4RobotConfig.init_qpos` starts its
    fingers *shut*, so before `snap_to_franka_joint_pos` was taught to place the
    hand, the retargeted condition began every episode reporting 0.824: the far
    end of the same channel, on a policy that decides when to close from it.

    That is a confound rather than a crash, which is why it wanted a test. The
    two conditions of the study differed in their starting gripper state, and
    nothing would have failed or logged to say so.

    Driven through the episode-start sequence a rollout actually performs --
    spawn configuration, `reset()`, then the snap -- rather than through
    `RetargetRig.restore`, so what is asserted is the production path.
    """
    rig.data.qpos[:] = rig._home_qpos
    mujoco.mj_forward(rig.model, rig.data)
    rig.proxy.reset()
    rig.proxy.snap_to_franka_joint_pos()

    franka_home = float(rig.proxy.franka.init_gripper_qpos[0])
    reported = float(np.asarray(rig.proxy.get_move_group("gripper").joint_pos)[0])
    assert reported == pytest.approx(franka_home, abs=1e-3), (
        f"at episode start the retargeted policy reads qpos['gripper'][0] = {reported:.4f} "
        f"where the Franka condition reads {franka_home:.4f}. "
        f"{fr.ROBOTIQ_DRIVER_CLOSED:.3f} means Stretch's hand is shut and the checkpoint is "
        f"being told so on its first observation of an episode it expects to start open."
    )

    opening = opening_fraction(rig.view.get_move_group("gripper"))
    assert opening > 0.95, (
        f"the reported state says open but Stretch's fingers are {opening * 100:.0f}% open, "
        f"so the snap is writing `ctrl` without placing the hand."
    )

    # ...and the proxy must not claim a command it has not established.
    claimed = float(rig.proxy.last_gripper_ctrl[0])
    assert claimed == pytest.approx(fr.ROBOTIQ_CTRL_RANGE[0], abs=1.0), (
        f"`last_gripper_ctrl` claims {claimed:.1f} on a hand that is "
        f"{opening * 100:.0f}% open. `ctrl` and `joint_pos` are then describing different "
        f"grippers, which is what `robotiq_ctrl_from_stretch_fingers` exists to prevent."
    )


@pytest.mark.parametrize("robotiq", [0.0, 96.0, 255.0])
def test_reset_reports_the_hand_it_can_see(rig: RetargetRig, robotiq: float) -> None:
    """`reset()` must read the gripper command off the fingers, not assume one.

    Separate from `test_the_hand_starts_where_a_droid_episode_starts_it` because
    it covers the case that test cannot: with `snap_to_franka_home` left on, the
    snap opens the hand and an assumed "open" is accidentally right, so a
    regression to hardcoding it passes everything else. Turning the flag off is a
    supported configuration -- the snap is a visible teleport of the arm, which
    is a fair thing to not want -- and in it, `reset()`'s assumption is the only
    thing deciding what the proxy believes about its own gripper.

    So the fingers are put somewhere known and `reset()` is asked what it thinks
    the hand has been told. Anything other than the command that produced those
    fingers means `ctrl` and `joint_pos` are describing different grippers.
    """
    gripper = rig.view.get_move_group("gripper")
    gripper.joint_pos = rig.proxy.retarget_robotiq_ctrl(robotiq)
    gripper.ctrl = gripper.joint_pos
    mujoco.mj_forward(rig.model, rig.data)

    rig.proxy.reset()

    claimed = float(rig.proxy.last_gripper_ctrl[0])
    assert claimed == pytest.approx(robotiq, abs=2.0), (
        f"the fingers were placed by a Robotiq {robotiq:.0f} command, but after `reset()` the "
        f"proxy reports `last_gripper_ctrl` = {claimed:.1f}. A constant "
        f"{fr.ROBOTIQ_CTRL_RANGE[0]:.0f} here is the old assumption that the hand starts open, "
        f"which is wrong on a robot whose `init_qpos` shuts it."
    )
    rig.restore()


@pytest.mark.parametrize("robotiq", [0.0, 64.0, 128.0, 192.0, 255.0])
def test_both_grippers_open_the_same_amount(rig: RetargetRig, robotiq: float) -> None:
    """One Robotiq command, and both hands end up equally open.

    This is the check that the rendered comparison was missing, and it was worth
    adding for exactly that reason: the first version of `--visualize` commanded
    only the arms, so each robot's gripper stayed at its own `init_qpos` -- the
    Franka's open, Stretch's shut -- and the video showed the two disagreeing
    about a command neither had been given. The retargeting was right and the
    picture was wrong, which is the more dangerous way round.

    `test_the_gripper_command_maps_across_the_full_finger_range` already pins
    Stretch's end of the mapping against its own limits. What this adds is the
    Franka, so the claim becomes relative rather than absolute: the same 0-255
    that opens the hand the policy was trained on opens Stretch's hand by the
    same fraction of its travel. An inverted flip passes the first check -- the
    range is still fully covered, just backwards -- and fails this one at every
    command except the midpoint.

    The Franka's hand is settled with physics rather than posed, because it
    cannot be posed; `FRANKA_GRIPPER_SETTLE_STEPS` explains why, and it is the
    reason this check is worth its runtime.
    """
    reached = rig.command(rig.franka_command_for(rig.centred_pose()), robotiq=robotiq)

    assert reached.stretch_opening == pytest.approx(
        reached.franka_opening, abs=GRIPPER_OPENING_TOLERANCE
    ), (
        f"at Robotiq {robotiq:.0f} the Franka's hand is {reached.franka_opening * 100:.0f}% "
        f"open and Stretch's is {reached.stretch_opening * 100:.0f}%. Near-opposite values "
        f"mean the flip in `retarget_robotiq_ctrl` has been applied twice or not at all "
        f"(0 is open on the Robotiq and shut on Stretch)."
    )


def test_the_gripper_command_maps_across_the_full_finger_range(rig: RetargetRig) -> None:
    """A Robotiq 0-255 command has to arrive as an open or closed Stretch gripper.

    0 is open on the Robotiq and closed on Stretch, so this mapping contains a
    flip -- and a flip is exactly the kind of thing that survives a rollout
    looking merely unlucky, because the policy sees a gripper that closes when it
    reaches and opens when it grasps. Checked by *measuring the fingers* rather
    than by re-deriving the arithmetic: the assertion is on inter-finger distance
    in metres, which is what an object between them experiences.
    """
    gripper = rig.view.get_move_group("gripper")
    closed_m, open_m = gripper.inter_finger_dist_range

    def finger_distance(robotiq_command: float) -> float:
        gripper.joint_pos = rig.proxy.retarget_robotiq_ctrl(robotiq_command)
        mujoco.mj_forward(rig.model, rig.data)
        return float(gripper.inter_finger_dist)

    low, high = fr.ROBOTIQ_CTRL_RANGE
    assert finger_distance(low) == pytest.approx(open_m, abs=1e-3), (
        f"Robotiq {low:.0f} is the Robotiq's *open* command, so Stretch's fingers should be at "
        f"their widest ({open_m:.4f}m), not {finger_distance(low):.4f}m."
    )
    assert finger_distance(high) == pytest.approx(closed_m, abs=1e-3), (
        f"Robotiq {high:.0f} is the Robotiq's *closed* command, so Stretch's fingers should be "
        f"shut ({closed_m:.4f}m), not {finger_distance(high):.4f}m."
    )

    # Monotone in between, and clipped outside: a policy mid-grasp commands the
    # middle of the range, and one that overshoots must not wrap around to open.
    distances = [finger_distance(command) for command in np.linspace(low, high, 9)]
    assert all(later <= earlier + 1e-6 for earlier, later in zip(distances, distances[1:])), (
        f"the finger opening is not monotone in the Robotiq command: {np.round(distances, 4)}"
    )
    assert finger_distance(low - 50.0) == pytest.approx(open_m, abs=1e-3)
    assert finger_distance(high + 50.0) == pytest.approx(closed_m, abs=1e-3)


def test_the_grippers_stay_together_along_a_continuous_path(rig: RetargetRig) -> None:
    """Not just at the waypoints: the two grippers have to agree all the way between them.

    Every other check here solves one pose from the snap, which is the first
    action of a rollout and none of the rest. A policy emits a chunk of absolute
    joint targets and the arm is walked through them, each solve seeded from
    where the last one left the robot -- so the retargeting has to hold up under
    its own history, and a version that only agreed at poses reached from a known
    configuration would pass everything above and still drift through a rollout.

    Interpolated in joint space, which is the faithful analogue: the commands are
    Franka joint targets with a bounded delta between them, not tool poses.
    Position is asserted at the full tolerance at *every* intermediate command,
    orientation at the looser `TRAJECTORY_ORIENTATION_TOLERANCE_RAD` for the
    reason written there.

    This is what `--visualize` renders, and it is the one check whose failure is
    worth looking at rather than reading -- it is a path, so where it goes wrong
    matters.
    """
    steps = 12
    commands = [rig.franka_command_for(rig.waypoint_pose(waypoint)) for waypoint in WAYPOINTS]
    rig.restore()

    worst_position = (0.0, "")
    worst_orientation = (0.0, "")
    for index, command in enumerate(commands):
        previous = commands[index - 1] if index else command
        for step in range(1, steps + 1):
            reached = rig.command(
                previous + (step / steps) * (command - previous), restore=False
            )
            label = f"{WAYPOINTS[index].label} step {step}/{steps}"
            worst_position = max(worst_position, (reached.position_error, label))
            worst_orientation = max(worst_orientation, (reached.orientation_error, label))

    error, label = worst_position
    assert error < POSITION_TOLERANCE_M, (
        f"walking the waypoints continuously, the two tool centres drifted "
        f"{error * 1000:.1f}mm apart at {label}. Position is the half of the pose "
        f"`StretchArmIK` is supposed to serve outright, so a drift here is not the wrist "
        f"running out of DOFs -- check whether the base has wandered to the end of its leash "
        f"(`StretchArmIK.releash`), which is the one piece of state a path accumulates."
    )
    error, label = worst_orientation
    assert error < TRAJECTORY_ORIENTATION_TOLERANCE_RAD, (
        f"walking the waypoints continuously, the tool frames drifted {error:.3f} rad apart "
        f"at {label}, past the {TRAJECTORY_ORIENTATION_TOLERANCE_RAD} rad that the wrist's "
        f"five DOFs are known to cost on this path. Run --visualize to see where it goes."
    )


def test_the_arm_alone_is_short_and_the_base_makes_up_the_difference() -> None:
    """`include_base` has to be doing something, and this says how much.

    `StretchArmIK`'s docstring rests a design decision on a measurement -- the
    arm telescopes along one line, so a standing Stretch reaches a narrow
    corridor and the holonomic base is what buys the rest back. This checks the
    claim still holds by solving the same far waypoint both ways. It builds its
    own rig because `include_base` is fixed at construction.

    Deliberately an inequality rather than a threshold: the point is that the
    base contributes, not that it contributes some particular number of
    centimetres, which depends on the kitchen.
    """
    without_base = RetargetRig(include_base=False)
    waypoint = Waypoint("far", forward=0.30, left=-0.20)
    command = without_base.franka_command_for(without_base.waypoint_pose(waypoint))

    arm_only = without_base.command(command)
    with_base = RetargetRig().command(command)

    assert with_base.position_error < POSITION_TOLERANCE_M, (
        f"with the base in the IK, {waypoint.label!r} should be reachable, but Stretch was "
        f"{with_base.position_error * 1000:.1f}mm off."
    )
    assert arm_only.position_error > with_base.position_error, (
        f"the arm alone reached {waypoint.label!r} as well as the arm plus the base "
        f"({arm_only.position_error * 1000:.1f}mm vs {with_base.position_error * 1000:.1f}mm). "
        f"Either the waypoint is now inside the arm's own corridor -- in which case it is no "
        f"longer testing what `include_base` is for -- or the base is not joining the solve."
    )


# =============================================================================
# Looking at it
# =============================================================================

VISUALIZE_FRAME_SIZE = (960, 540)
"""Per-robot frame size for `--visualize`, as (width, height). The pair is twice as wide."""


def visualize(
    output_dir: Path,
    frames_per_waypoint: int = 12,
    fps: int = 20,
    azimuth: float = 25.0,
    elevation: float = -18.0,
    distance: float = 1.15,
) -> Path:
    """Render both robots walking through `WAYPOINTS`, side by side, to an MP4.

    The left half is the Franka on its pedestal, the right half is Stretch, and
    both are rendered from the *same* free camera -- same lookat, azimuth,
    elevation and distance -- which is what makes the two halves comparable: the
    virtual Franka stands 5cm in front of Stretch and faces the same way, so one
    camera pose frames both robots' work the same.

    Three things are drawn over the pair. Each robot's tool frame as a ball and
    three axis arrows, in that frame's own colour, so "the same pose" can be seen
    rather than read off a table -- and the quarter turn between the two tool
    conventions is visible as the arrows pointing differently while the balls
    coincide. Then the commanded pose in the Franka's colour on *both* halves, so
    Stretch's ball can be compared with where it was asked to be. And a
    translucent disk at the commanded height in both halves, which is the only
    shared ruler two separately rendered scenes have.

    Joint space is interpolated between consecutive commands, and every
    intermediate command goes through the retargeting exactly as a waypoint does.
    That is the part a still cannot show: the two grippers have to stay together
    *along the way*, and a retargeting that only agreed at the waypoints would be
    visible here as Stretch taking a different route between them.
    """
    import cv2

    rig = RetargetRig()
    commands = [rig.franka_command_for(rig.waypoint_pose(waypoint)) for waypoint in WAYPOINTS]
    labels = [waypoint.label for waypoint in WAYPOINTS]

    # Snap once, then never again -- the rollout's own opening move. Without it
    # the motion starts from Stretch's stowed pose, half a metre and a quarter
    # turn from the first command, and `StretchArmIK` spends the whole first
    # waypoint catching up: it renders as the two grippers 20mm apart at a
    # waypoint the checks above measure at 0.02mm, which would be the
    # visualisation slandering the retargeting.
    rig.restore()

    # A fixed lookat, at the middle of the commanded poses, so the camera holds
    # still and the grippers are what moves.
    poses = [rig.waypoint_pose(waypoint) for waypoint in WAYPOINTS]
    lookat = np.mean([pose[:3, 3] for pose in poses], axis=0)

    width, height = VISUALIZE_FRAME_SIZE
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "retargeting.mp4"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width * 2, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open {video_path} for writing")

    def render(reached: Reached, label: str) -> np.ndarray:
        hand = "closed" if reached.franka_opening < 0.5 else "open"
        franka_frame = fr.render_tool_frame_view(
            rig.franka_model,
            rig.franka_data,
            lookat,
            markers=[
                (reached.franka_pose, fr.FRANKA_TOOL_COLOR, f"franka {label} / {hand}")
            ],
            reference_height=float(reached.expected_pose[2, 3]),
            width=width,
            height=height,
            distance=distance,
            azimuth=azimuth,
            elevation=elevation,
        )
        stretch_frame = fr.render_tool_frame_view(
            rig.model,
            rig.data,
            lookat,
            markers=[
                (reached.expected_pose, fr.FRANKA_TOOL_COLOR, "commanded"),
                (
                    reached.stretch_pose,
                    fr.STRETCH_TOOL_COLOR,
                    f"stretch {reached.position_error * 1000:.0f}mm"
                    f" / {'closed' if reached.stretch_opening < 0.5 else 'open'}"
                    # Worth saying on the frame: when the jaw is flipped the two
                    # sets of axis arrows are half a turn apart on purpose, and
                    # the label is what stops that reading as a bug.
                    + (" (jaw flipped)" if reached.jaw_flipped else ""),
                ),
            ],
            reference_height=float(reached.expected_pose[2, 3]),
            width=width,
            height=height,
            distance=distance,
            azimuth=azimuth,
            elevation=elevation,
        )
        return fr.side_by_side(franka_frame, stretch_frame)

    click.echo(f"rendering {len(commands)} waypoints at {frames_per_waypoint} frames each")
    worst = (0.0, "")
    for index, (command, label) in enumerate(zip(commands, labels)):
        previous = commands[index - 1] if index else command
        # The first waypoint is held rather than interpolated into from nowhere;
        # every other one is walked into from the one before.
        for step in range(1, frames_per_waypoint + 1):
            blend = step / frames_per_waypoint
            reached = rig.command(
                previous + blend * (command - previous),
                # The gripper steps rather than ramps: a policy emits a gripper
                # command per chunk, not a trajectory, and only the two extremes
                # appear in `WAYPOINTS` -- which also keeps the Robotiq to two
                # settles for the whole render. See `settled_franka_gripper`.
                robotiq=WAYPOINTS[index].robotiq,
                # Not restored between frames: this is one continuous motion, and
                # each solve should seed from the configuration the last one left
                # the robot in, exactly as it does during a rollout.
                restore=False,
            )
            frame = render(reached, label)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            worst = max(worst, (reached.position_error, f"{label} step {step}"))

        # The still is the last frame of the walk, which is the robot *at* the
        # waypoint -- reused rather than re-rendered, since it is the same image.
        cv2.imwrite(
            str(output_dir / f"{index:02d}_{label}.png"),
            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
        )
        click.echo(
            f"  {label:16s} tool centres {reached.position_error * 1000:6.2f}mm apart, "
            f"tool frames {reached.orientation_error:.4f} rad apart"
        )

    writer.release()
    click.secho(
        f"\nworst over every rendered frame, not just the waypoints: "
        f"{worst[0] * 1000:.2f}mm, at {worst[1]}",
        bold=True,
    )
    click.echo(f"video:  {video_path}")
    click.echo(f"stills: {output_dir}/NN_<waypoint>.png")
    return video_path


@click.command(name="test_retargeting")
@click.option(
    "--visualize",
    "do_visualize",
    is_flag=True,
    help="Render both robots through the waypoints to an MP4 and per-waypoint stills.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("retargeting_check"),
    show_default=True,
    help="Where --visualize writes its MP4 and stills.",
)
@click.option(
    "--frames-per-waypoint",
    type=click.IntRange(min=1),
    default=12,
    show_default=True,
    help="Interpolation steps between consecutive commands. More is smoother and slower.",
)
@click.option("--fps", type=int, default=20, show_default=True, help="Frame rate of the MP4.")
@click.option(
    "--azimuth", type=float, default=25.0, show_default=True, help="Free-camera azimuth."
)
@click.option(
    "--elevation", type=float, default=-18.0, show_default=True, help="Free-camera elevation."
)
@click.option(
    "--distance",
    type=float,
    default=1.15,
    show_default=True,
    help="Free-camera distance. The default frames the grippers rather than the robots -- "
    "the tool frames are what there is to compare.",
)
def cli(
    do_visualize: bool,
    output_dir: Path,
    frames_per_waypoint: int,
    fps: int,
    azimuth: float,
    elevation: float,
    distance: float,
) -> None:
    """The retargeting checks, and a rendering of them.

    With no flags this runs the same assertions `pytest` does, printing a table
    of how far apart the two grippers end up at each waypoint. `--visualize`
    renders both robots walking through those same waypoints instead.
    """
    import logging

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    if do_visualize:
        visualize(
            output_dir,
            frames_per_waypoint=frames_per_waypoint,
            fps=fps,
            azimuth=azimuth,
            elevation=elevation,
            distance=distance,
        )
        return

    raise SystemExit(pytest.main(["-v", __file__]))


if __name__ == "__main__":
    cli()
