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

    # either mode at the target height offset a rollout actually applies
    python -m ...test_retargeting --target-z-offset 0.05
    python -m ...test_retargeting --target-z-offset 0.05 --visualize

A note on that last one, because it is the interesting case rather than a
curiosity. `target_z_offset` raises every retargeted target to keep Stretch's
gripper off the countertop. It now defaults to 0 in the policy as well as here,
so the checks below describe what a rollout actually does -- but the parameter is
still worth sweeping, and this is how to see what a non-zero value costs.

The offset moves *Stretch* and nothing else. The Franka is the reference the
whole comparison is measured against, so the waypoints and the commands that
reach them are built with the offset held at zero (`RetargetRig.franka_ground_truth`)
and are bit-for-bit identical at every offset; what changes is how far above the
Franka's tool Stretch is asked to hold its own. Verified: at 0.00, 0.02 and 0.05
the Franka's command is unchanged and its tool stays at z = 0.8303, while
Stretch's rises to exactly +20.2mm and +50.0mm above it.

At 0.05 the waypoints still arrive, because they sit far enough below the lift's
ceiling to absorb it. What fails is
`test_the_grippers_stay_together_along_a_continuous_path`, by around 45mm, and
all of that miss is vertical: joint-space interpolation between two waypoints
swings the Franka's tool higher than either end, and with every target raised
5cm those intermediate poses land above what Stretch's lift can reach. Not a
regression and not a loose tolerance -- the path between the waypoints leaves
the workspace even though the waypoints do not, which is worth knowing about the
offset the benchmark actually runs at.
"""

from __future__ import annotations

import math
import os
import sys
from contextlib import contextmanager
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


TARGET_Z_OFFSET_ENV_VAR = "RETARGET_TEST_TARGET_Z_OFFSET"
"""
Environment variable naming the `target_z_offset` the rig is built with.

An environment variable rather than a module constant because the checks are run
two ways and both have to reach it: `--target-z-offset` on this module's own
command line, which then runs `pytest` in-process, and a bare
`pytest tests/test_retargeting.py` with the variable exported. A pytest option
would only serve the second.

It defaults to 0.0, which is both the offset at which "Stretch's tool goes
exactly where the Franka's is" is a meaningful claim *and*, now, what a benchmark
rollout applies: `StretchMolmoBotDroidPolicyConfig.target_z_offset` is a
hand-set 0 rather than a fraction of a measured shortfall. It stays exposed
because the offset is still a parameter worth sweeping, and running these checks
at the value a sweep picks is how you see what it costs.
"""


def configured_target_z_offset() -> float:
    """The offset from the environment, or 0.0. See `TARGET_Z_OFFSET_ENV_VAR`."""
    raw = os.environ.get(TARGET_Z_OFFSET_ENV_VAR, "").strip()
    if not raw:
        return 0.0
    try:
        return float(raw)
    except ValueError as error:
        raise ValueError(
            f"{TARGET_Z_OFFSET_ENV_VAR}={raw!r} is not a number of metres."
        ) from error


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


def commanded_aperture_fraction(aperture_m: float) -> float:
    """An aperture as a fraction of the *commanded* range, which both hands share.

    Not a fraction of either hand's own travel. Stretch's jaw still physically
    opens to 0.1885m, but `match_robotiq_aperture` puts the top of the commanded
    range at the Robotiq's 0.0870m, so "fully open" is 1.0 here on both robots
    while being 0.46 of Stretch's mechanical travel. Used only to label a frame
    open or shut; the assertions compare metres.
    """
    return float(np.clip(aperture_m / fr.ROBOTIQ_MAX_APERTURE_M, 0.0, 1.0))


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

    target_z_offset: float = 0.0
    """The offset in force, so a `Reached` can say what its own poses mean."""

    franka_aperture: float = 0.0
    """How wide the Franka's jaw is, in metres."""

    stretch_aperture: float = 0.0
    """The same for Stretch's. Directly comparable: see `ROBOTIQ_MAX_APERTURE_M`."""

    @property
    def franka_reference_pose(self) -> np.ndarray:
        """The Franka's grasp centre, in Stretch's tool convention, *without* the offset.

        `expected_pose` is this pose raised by `target_z_offset`, so this is the
        un-offset reference the offset is measured from -- the same physical point
        as `franka_pose`, rotated so its axes are comparable with Stretch's.
        Drawn on the Stretch panel so a render at a non-zero offset shows all
        three of the things that matter: where the Franka's tool is, where Stretch
        was asked to put its own, and where it got to.
        """
        pose = np.array(self.expected_pose, dtype=float, copy=True)
        pose[2, 3] -= self.target_z_offset
        return pose

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

    def __init__(
        self,
        include_base: bool = True,
        target_z_offset: float = 0.0,
        jaw_mode: str = "auto",
    ) -> None:
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
            jaw_mode=jaw_mode,
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

    @contextmanager
    def franka_ground_truth(self):
        """Hold `target_z_offset` at zero for the duration.

        Every pose that describes *the Franka* has to be built inside this. The
        Franka is the reference the whole comparison is measured against, so the
        offset is not allowed anywhere near it: `target_z_offset` means "put
        Stretch's tool this much above where the Franka's is", and it earns that
        meaning only if the Franka does not move when it changes.

        `franka_tool_pose_to_world` and `stretch_tool_pose_to_franka` both fold
        the offset in, which is right for the retargeting -- they are how a
        commanded Stretch target gets raised -- and wrong for generating a
        waypoint or the command that reaches it. An earlier version of this file
        used them for both, and the result was exactly backwards: raising the
        offset moved the Franka *down* and left Stretch's target where it was.
        """
        offset = self.proxy.target_z_offset
        self.proxy.target_z_offset = 0.0
        try:
            yield
        finally:
            self.proxy.target_z_offset = offset

    @property
    def franka_home_command(self) -> np.ndarray:
        """The Franka's own `init_qpos` arm configuration."""
        return self.proxy.franka.init_qpos.copy()

    def centred_pose(self) -> np.ndarray:
        """The pose the waypoints are offsets from: the Franka's home, lowered to counter height.

        A pose *the Franka's tool* reaches, and therefore built without the
        offset -- the waypoints are the fixed ground truth that raising
        `target_z_offset` is measured against, not something it moves.
        """
        with self.franka_ground_truth():
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

        `pose` is taken back through the tool correction and the mount into the
        Franka's own base frame, and the virtual Franka's IK reads off the joint
        angles. Seeded from the Franka's home for every waypoint rather than from
        the previous answer, so a command depends only on its own waypoint.

        Offset-free, because this is the Franka's command: see
        `franka_ground_truth`. What `target_z_offset` then does is raise the
        Stretch target this command retargets to, which is its entire job.
        """
        with self.franka_ground_truth():
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
            target_z_offset=float(self.proxy.target_z_offset),
            franka_aperture=float(self.franka_view.get_move_group("gripper").inter_finger_dist),
            stretch_aperture=float(self.view.get_move_group("gripper").inter_finger_dist),
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
    offset = configured_target_z_offset()
    if offset:
        log_line = (
            f"target_z_offset = {offset:+.4f}m: every commanded grasp is raised by this "
            f"much, so the checks below are measuring the retargeting as a rollout "
            f"actually applies it."
        )
        print(f"\n[retargeting] {log_line}")
    return RetargetRig(target_z_offset=offset)


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

    # The Franka's own pose against the waypoint, with the offset taken back out
    # of `expected_pose`. Both sides are then Franka ground truth, so this asks
    # the offset-independent question -- did the Franka hold the pose it was
    # asked for -- and stays true at any `target_z_offset`, which does not move
    # the Franka at all. (`franka_pose` is in the Franka's own tool convention;
    # `expected_pose` is the same point rotated into Stretch's, which is the
    # convention the waypoints are written in.)
    franka_reached = reached.expected_pose.copy()
    franka_reached[2, 3] -= rig.proxy.target_z_offset
    franka_miss, _ = pose_difference(franka_reached, requested)
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
        f"waypoint {waypoint.label!r}, rotated into Stretch's convention (jaw_mode "
        f"{rig.proxy.jaw_mode!r}). About 1.57 rad here is `FRANKA_TO_STRETCH_TOOL` applied on "
        f"the wrong side or about the wrong axis; about 0.43 rad at a large tool yaw is "
        f"`jaw_mode=\"flipped\"` running the wrist into a roll limit, which is that mode's "
        f"known cost -- see `franka_retarget.JAW_MODES`."
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
    sets as `target_z_offset`.
    """
    reached = rig.command(rig.franka_home_command)
    # `measure_tool_height_offset` deliberately ignores whatever offset is set and
    # reports the raw shortfall, while the miss measured here is against a target
    # the offset has already raised -- so with `--target-z-offset` in force the
    # two differ by exactly that offset, and the lift being saturated is what
    # makes it exact: the target moves up and the robot cannot follow.
    shortfall = rig.proxy.measure_tool_height_offset() + rig.proxy.target_z_offset

    assert reached.position_error > POSITION_TOLERANCE_M, (
        "Stretch reached the Franka's home tool pose. That would be good news, but it "
        "contradicts `FrankaOnStretchView.target_z_offset`'s reason for existing -- if the "
        "lift's travel has changed, that docstring and `target_z_offset`'s default need "
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


@pytest.mark.parametrize("offset", [0.0, 0.02, 0.05])
def test_the_offset_moves_stretch_and_never_the_franka(rig: RetargetRig, offset: float) -> None:
    """`target_z_offset` is allowed to move Stretch. It is not allowed to move the Franka.

    The Franka is the reference the whole comparison is measured against, so the
    offset has to be a one-way adjustment: given a fixed Franka command -- which
    is what a policy emits, and is never re-derived -- the Franka's tool goes
    where it goes, and only Stretch's target is raised.

    Three claims, because a transform that folded the offset into the wrong side
    would satisfy some of them:

    1. `mount @ fk(command)` does not depend on the offset at all. This is the
       Franka's pose in the room, and nothing about a Stretch-side correction may
       reach it.
    2. The target the retargeting asks Stretch for is the Franka's pose plus
       exactly the offset.
    3. `franka_joint_pos()` hands the policy back the command it gave, at any
       offset -- so the observation half removes exactly what the action half
       added. Without this the policy would read its own command as having moved
       and spend the next chunk correcting a correction.

    This exists because the *test harness* had it backwards for a while: it built
    its Franka commands by inverting through `stretch_tool_pose_to_franka`, which
    subtracts the offset, so raising the offset moved the Franka down and left
    Stretch where it was. The production path was right all along, and this is
    what keeps it that way.
    """
    command = rig.franka_command_for(rig.waypoint_pose(WAYPOINTS[8]))
    franka_world = rig.mount_pose @ rig.proxy.franka.fk(command)

    previous = rig.proxy.target_z_offset
    rig.proxy.target_z_offset = offset
    try:
        reached = rig.command(command, restore=False)
        target = rig.proxy.franka_tool_pose_to_world(rig.proxy.franka.fk(command))
        reported = rig.proxy.franka_joint_pos()
    finally:
        rig.proxy.target_z_offset = previous

    unmoved = rig.mount_pose @ rig.proxy.franka.fk(command)
    assert np.allclose(unmoved, franka_world, atol=1e-12), (
        f"the Franka's own tool pose changed when target_z_offset became {offset}. It is the "
        f"reference everything else is measured against; nothing on the Stretch side may "
        f"move it."
    )
    assert reached.franka_pose[2, 3] == pytest.approx(franka_world[2, 3], abs=1e-9), (
        f"the real Franka's grasp site moved with the offset, to "
        f"{reached.franka_pose[2, 3]:.4f} from {franka_world[2, 3]:.4f}."
    )
    rise = target[2, 3] - franka_world[2, 3]
    assert rise == pytest.approx(offset, abs=1e-9), (
        f"an offset of {offset} raised Stretch's commanded target by {rise:.4f}m. These are "
        f"the same number by definition -- `franka_tool_pose_to_world` adds the offset and "
        f"nothing else should."
    )
    assert np.max(np.abs(reported - command)) < 1e-2, (
        f"at offset {offset} the arm state reported back to the policy differs from the "
        f"command by {np.max(np.abs(reported - command)):.4f} rad. The observation half has "
        f"to subtract exactly what the action half added, or the policy reads its own "
        f"command as having moved."
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

    # Restored to whatever the rig was *built* with, not to zero: with
    # `--target-z-offset` set, zeroing it here would leave the shared rig in a
    # configuration the rest of the module is not expecting. See
    # `TARGET_Z_OFFSET_ENV_VAR`.
    configured = rig.proxy.target_z_offset
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
        rig.proxy.target_z_offset = configured

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


GRIPPER_APERTURE_TOLERANCE_M = 0.005
"""
How far apart the two grippers' apertures may be, in metres.

In metres, and that is the point: before `match_robotiq_aperture` the two hands
were different sizes and only the *fraction* of each one's travel could be
compared, which said nothing about whether an object of a given width fitted.
Now a command means the same gap on both, so the comparison is the physical one.

Measured across the range: 0.4mm apart at open, 0.1mm at shut, bulging to 2.7mm
around the middle. That bulge is mechanism, not error -- `retarget_robotiq_ctrl`
is linear in the command and Stretch's fingers are nearly linear in their angle,
while the Robotiq reaches its aperture through a linkage that is not. 5mm covers
it and is still far tighter than the 100mm an unmatched hand was out by at the
open end.
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

    aperture = float(rig.view.get_move_group("gripper").inter_finger_dist)
    opening = commanded_aperture_fraction(aperture)
    assert opening > 0.95, (
        f"the reported state says open but Stretch's jaw is {aperture * 1000:.1f}mm wide, "
        f"{opening * 100:.0f}% of the commanded range, so the snap is writing `ctrl` without "
        f"placing the hand."
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

    assert reached.stretch_aperture == pytest.approx(
        reached.franka_aperture, abs=GRIPPER_APERTURE_TOLERANCE_M
    ), (
        f"at Robotiq {robotiq:.0f} the Franka's jaw is {reached.franka_aperture * 1000:.1f}mm "
        f"wide and Stretch's is {reached.stretch_aperture * 1000:.1f}mm. Roughly double means "
        f"`match_robotiq_aperture` is off, so an open command spreads Stretch's fingers to "
        f"their own 188mm; near-opposite values mean the flip in `retarget_robotiq_ctrl` has "
        f"been applied twice or not at all (0 is open on the Robotiq, shut on Stretch)."
    )


def test_the_gripper_command_maps_across_the_full_finger_range(rig: RetargetRig) -> None:
    """A Robotiq 0-255 command has to arrive as an open or closed Stretch gripper.

    0 is open on the Robotiq and closed on Stretch, so this mapping contains a
    flip -- and a flip is exactly the kind of thing that survives a rollout
    looking merely unlucky, because the policy sees a gripper that closes when it
    reaches and opens when it grasps. Checked by *measuring the fingers* rather
    than by re-deriving the arithmetic: the assertion is on inter-finger distance
    in metres, which is what an object between them experiences.

    "Open" is the Robotiq's aperture, not Stretch's own widest. Stretch's jaw
    reaches 0.1885m and the Robotiq's 0.0870m, and `match_robotiq_aperture`
    narrows the commanded range to the latter so that a command means the same
    gap on both robots -- so the open end asserted here is
    `ROBOTIQ_MAX_APERTURE_M`, and Stretch's remaining travel above it is
    deliberately unreachable. See `ROBOTIQ_MAX_APERTURE_M`.
    """
    gripper = rig.view.get_move_group("gripper")
    closed_m = gripper.inter_finger_dist_range[0]
    open_m = fr.ROBOTIQ_MAX_APERTURE_M

    def finger_distance(robotiq_command: float) -> float:
        gripper.joint_pos = rig.proxy.retarget_robotiq_ctrl(robotiq_command)
        mujoco.mj_forward(rig.model, rig.data)
        return float(gripper.inter_finger_dist)

    low, high = fr.ROBOTIQ_CTRL_RANGE
    assert finger_distance(low) == pytest.approx(open_m, abs=1e-3), (
        f"Robotiq {low:.0f} is the Robotiq's *open* command, so Stretch's fingers should be "
        f"{open_m:.4f}m apart -- the Robotiq's own aperture -- not {finger_distance(low):.4f}m. "
        f"{gripper.inter_finger_dist_range[1]:.4f}m means `match_robotiq_aperture` is off and "
        f"the hand is opening to Stretch's full width again."
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

    worst_position = (0.0, "", 0.0)
    worst_orientation = (0.0, "")
    for index, command in enumerate(commands):
        previous = commands[index - 1] if index else command
        for step in range(1, steps + 1):
            reached = rig.command(
                previous + (step / steps) * (command - previous), restore=False
            )
            label = f"{WAYPOINTS[index].label} step {step}/{steps}"
            worst_position = max(
                worst_position, (reached.position_error, label, reached.vertical_error)
            )
            worst_orientation = max(worst_orientation, (reached.orientation_error, label))

    error, label, vertical = worst_position
    offset = rig.proxy.target_z_offset
    # Which of the two causes it was, rather than a guess: a miss that is almost
    # entirely upwards is the lift out of travel, and `target_z_offset` makes that
    # more likely by raising every target. Anything else is the base.
    if abs(vertical) > 0.8 * error and vertical > 0:
        cause = (
            f"The miss is {vertical * 1000:+.1f}mm of it vertical, so this is the lift out of "
            f"travel rather than anything lateral"
            + (
                f" -- and `--target-z-offset {offset:g}` is raising every target by that much, "
                f"which is what pushed this intermediate pose past the ceiling. The waypoints "
                f"themselves still arrive; it is the path between them that leaves the "
                f"workspace. Expect this at a realistic offset."
                if offset
                else ", which at offset 0 means a waypoint has drifted out of reach."
            )
        )
    else:
        cause = (
            "The miss is mostly lateral, so it is not the lift: check whether the base has "
            "wandered to the end of its leash (`StretchArmIK.releash`), which is the one "
            "piece of state a path accumulates."
        )
    assert error < POSITION_TOLERANCE_M, (
        f"walking the waypoints continuously, the two tool centres drifted "
        f"{error * 1000:.1f}mm apart at {label}. {cause}"
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

GRASP_FRAME_BALL_RADIUS = 0.013
GRASP_FRAME_AXIS_LENGTH = 0.14
GRASP_FRAME_AXIS_RADIUS = 0.007

GRASP_FRAME_BALL_ALPHA = 0.55
GRASP_FRAME_BALL_NESTING = 0.30
"""
How the balls are made mutually visible when they land on the same point.

Which is the case that matters most: at `target_z_offset` 0 and a retargeting
that works, every marker on the Stretch panel is at the *same* position -- and
two opaque spheres of the same radius at the same place show one sphere, so the
picture that should read "these agree" instead reads as a single unexplained
ball.

Two changes together, because neither is enough alone. Translucency lets the
nearer ball show what is behind it, but two coincident translucent spheres of
equal size just blend into a third colour. So each marker is also drawn a little
smaller than the one before it (`GRASP_FRAME_BALL_NESTING` of the base radius per
step), which makes a coincident group read as nested shells: outermost is the
first marker listed, innermost the last. Separated markers are unaffected beyond
looking slightly different in size.

The axis arrows stay opaque -- `add_frame_marker` colours those itself -- and so
do the labels, which need to be legible rather than subtle.
"""
"""
How big a grasp-centre coordinate frame is drawn, in metres.

`add_frame_marker`'s own defaults (0.012 ball, 0.07 axes, 0.004 shafts) are sized
for a close-up. At the 1.15m the video is framed at they come out a few pixels
across: the balls are visible and the three axis arrows are not, which loses the
half of the marker that carries the *orientation* -- the thing the tool-frame
comparison is mostly about, since the two grippers reach along different axes and
`FRANKA_TO_STRETCH_TOOL` is what reconciles them.

Scaled up until the arrows read at that distance without the frame swamping the
gripper it belongs to. The ball is kept small on purpose: this view draws *two*
frames at once -- commanded and reached -- and when the retargeting is working
they are on top of each other, so a large ball would hide the very coincidence
it is there to show. A closer camera wants smaller numbers; they are constants
here rather than a function of `--distance` because a marker that resized itself
with the camera would stop being comparable between two renders.
"""


LABEL_LIFT_PX = 30
LABEL_STACK_PX = 24
LABEL_FONT_SCALE = 0.52
"""
How a marker's label is placed, in pixels above the ball it belongs to.

Drawn here rather than by MuJoCo, which is what `add_frame_marker`'s own `label`
would do. Two reasons, both visible in the frames this produced before: MuJoCo
draws every label at its geom's position in one fixed colour, so the commanded
and reached labels land on top of each other -- they are *supposed* to be at the
same point -- and neither says which ball it belongs to. Lifting them clear and
colouring each to match its ball fixes both.

`LABEL_STACK_PX` is the gap between successive labels, so the markers passed to
one `frame()` call stack upwards in order rather than collide. The first marker
listed ends up highest.
"""


@dataclass(frozen=True)
class Marker:
    """One thing to draw over a rendered frame: a ball, optionally a frame, a label.

    A named type rather than a tuple because markers stopped being uniform: some
    are in the picture as a *frame* and some only as a *position*, and a bare
    `(pose, colour, label)` had nowhere to say which.
    """

    pose: np.ndarray
    color: tuple
    label: str = ""

    axes: bool = True
    """
    Whether to draw the red/green/blue axis arrows as well as the ball.

    Off for a reference point borrowed from the other robot. On the Stretch panel
    the Franka's grasp centre is there to say *where* the Franka's tool is -- the
    point the offset is measured from and the position Stretch is being compared
    against. Its orientation is in the Franka's own tool convention, which differs
    from Stretch's by `FRANKA_TO_STRETCH_TOOL`, so drawing its axes next to
    Stretch's invites exactly the comparison that convention exists to prevent.
    """


class ToolFrameRenderer:
    """One reusable `mujoco.Renderer` for one model, held open across many frames.

    `franka_retarget.render_tool_frame_view` builds a `Renderer` per call and
    closes it again, which is right for its own caller -- the demo takes one
    snapshot per episode -- but it means an EGL context is created and torn down
    for every image. Measured here, that is 226ms of the 250ms a frame costs, so
    a second of 30fps video would spend fourteen seconds in context setup.

    Holding the renderer open instead makes a smooth video practical. The camera
    and scene options are fixed at construction because the whole point of this
    comparison is that both robots are seen from the same place: pass the same
    ones to both renderers and the two halves stay comparable frame by frame.

    The decoration is `franka_retarget`'s own (`add_height_plane`,
    `add_frame_marker`), so the markers here mean exactly what they mean in a
    `render_tool_frame_view` image.
    """

    def __init__(
        self,
        model: MjModel,
        lookat,
        width: int,
        height: int,
        distance: float,
        azimuth: float,
        elevation: float,
    ) -> None:
        self._renderer = mujoco.Renderer(model, height, width)
        self._camera = mujoco.MjvCamera()
        self._camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self._camera.lookat[:] = np.asarray(lookat, dtype=float)
        self._camera.distance = distance
        self._camera.azimuth = azimuth
        self._camera.elevation = elevation
        # Sites hidden, as everywhere else here, so the only frames drawn are the
        # ones asked for.
        self._option = mujoco.MjvOption()
        self._option.sitegroup = 0

    def frame(
        self,
        data: MjData,
        markers: list[Marker],
        reference_height: float | None = None,
        note: str = "",
    ) -> np.ndarray:
        """One frame, with the markers drawn over it and `note` written in the corner."""
        self._renderer.update_scene(data, camera=self._camera, scene_option=self._option)
        if reference_height is not None:
            fr.add_height_plane(self._renderer.scene, reference_height, self._camera.lookat)
        for index, marker in enumerate(markers):
            # Sized for this camera rather than left at the close-up defaults;
            # see `GRASP_FRAME_AXIS_LENGTH`. Each marker is a ball at the grasp
            # centre plus red/green/blue arrows on its x/y/z, so a frame that is
            # in the right *place* but the wrong way round is visible as such.
            #
            # No label: MuJoCo would draw it at the ball, in one colour, on top
            # of the other marker's. `_label` does it instead.
            fr.add_frame_marker(
                self._renderer.scene,
                marker.pose,
                color=tuple(marker.color[:3]) + (GRASP_FRAME_BALL_ALPHA,),
                label="",
                ball_radius=self._ball_radius(index),
                axis_length=GRASP_FRAME_AXIS_LENGTH,
                axis_radius=GRASP_FRAME_AXIS_RADIUS,
                axes=marker.axes,
            )
        image = self._label(np.ascontiguousarray(self._renderer.render()), markers)
        return self._note(image, note) if note else image

    @staticmethod
    def _ball_radius(index: int) -> float:
        """The `index`-th marker's ball radius, shrinking so coincident ones nest."""
        return GRASP_FRAME_BALL_RADIUS * max(0.25, 1.0 - index * GRASP_FRAME_BALL_NESTING)

    def project(self, point, width: int, height: int) -> tuple[int, int] | None:
        """A world point as a pixel in the rendered image, or None if it is behind.

        The renderer's own GL camera, which `update_scene` fills in, so this is
        the frustum the image was actually drawn with rather than a reconstruction
        from the camera parameters. Checked against the pixels: a point on the
        view axis lands on the image centre to within half a pixel at 16:9, 16:9
        at a different size, and 1:1, and the scale matches a measured fit to
        0.05%.
        """
        camera = self._renderer.scene.camera[0]
        position = np.array(camera.pos, dtype=float)
        forward = np.array(camera.forward, dtype=float)
        up = np.array(camera.up, dtype=float)
        right = np.cross(forward, up)
        norm = np.linalg.norm(right)
        if norm == 0.0:
            return None
        right /= norm

        offset = np.asarray(point, dtype=float) - position
        depth = float(offset @ forward)
        if depth <= camera.frustum_near:
            return None  # behind the camera, or inside the near plane

        half_height = (camera.frustum_top - camera.frustum_bottom) / 2.0
        if half_height <= 0.0:
            return None
        half_width = half_height * (width / height)
        horizontal = float(offset @ right) * camera.frustum_near / depth
        vertical = float(offset @ up) * camera.frustum_near / depth
        ndc_x = (horizontal - camera.frustum_center) / half_width
        ndc_y = (vertical - (camera.frustum_top + camera.frustum_bottom) / 2.0) / half_height
        return int(round((ndc_x * 0.5 + 0.5) * width)), int(round((0.5 - ndc_y * 0.5) * height))

    @staticmethod
    def _note(image: np.ndarray, note: str) -> np.ndarray:
        """Write a standing note along the bottom of the panel.

        Bottom-left, because the top of these frames is where the marker labels
        stack and the middle is where the robot is. Neutral white rather than a
        marker colour: it is a statement about the drawing, not about any one
        thing in it.
        """
        import cv2

        origin = (12, image.shape[0] - 14)
        for thickness, shade in ((3, (0, 0, 0)), (1, (235, 235, 235))):
            cv2.putText(
                image, note, origin, cv2.FONT_HERSHEY_SIMPLEX,
                LABEL_FONT_SCALE, shade, thickness, cv2.LINE_AA,
            )
        return image

    def _label(self, image: np.ndarray, markers) -> np.ndarray:
        """Draw each marker's label above its ball, in the ball's own colour."""
        import cv2

        height, width = image.shape[:2]
        for index, marker in enumerate(markers):
            if not marker.label:
                continue
            placed = self.project(np.asarray(marker.pose, dtype=float)[:3, 3], width, height)
            if placed is None:
                continue
            x, y = placed
            y -= LABEL_LIFT_PX + index * LABEL_STACK_PX
            if not (0 <= y < height):
                continue
            # The renderer hands back RGB, so the colour goes in unswapped; the
            # caller is what converts the finished frame to BGR.
            rgb = tuple(int(round(255 * channel)) for channel in marker.color[:3])
            origin = (max(4, min(x + 10, width - 8)), y)
            for thickness, shade in ((3, (0, 0, 0)), (1, rgb)):
                cv2.putText(
                    image, marker.label, origin, cv2.FONT_HERSHEY_SIMPLEX,
                    LABEL_FONT_SCALE, shade, thickness, cv2.LINE_AA,
                )
        return image

    def close(self) -> None:
        self._renderer.close()

    def __enter__(self) -> "ToolFrameRenderer":
        return self

    def __exit__(self, *exception) -> None:
        self.close()


FRANKA_AXES_NOTE = "franka axes drawn in Stretch's tool convention: x/red = approach"
STRETCH_AXES_NOTE = "stretch axes are its own: x/red = approach"
"""
What each panel says about the frame it is drawing.

The Franka's panel earns its note: the axes on it are not the Franka's own
lettering but the same point rotated by `FRANKA_TO_STRETCH_TOOL`, so that one
colour means one direction across both halves. Stretch's note is there so the
pair reads as a deliberate choice rather than an unexplained asymmetry -- its
axes needed no rotating, and saying so is cheaper than leaving the reader to
wonder which panel was adjusted.
"""

VISUALIZE_FRAME_SIZE = (960, 540)
"""Per-robot frame size for `--visualize`, as (width, height). The pair is twice as wide."""


def visualize(
    output_dir: Path,
    target_z_offset: float = 0.0,
    fps: int = 30,
    seconds_per_move: float = 1.0,
    hold_seconds: float = 0.4,
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

    Timed in seconds rather than in frames, so the motion plays at the same speed
    whatever `fps` is: `seconds_per_move` is how long the arm takes to travel
    between two waypoints and `hold_seconds` how long it pauses on arrival. The
    pause matters more than it sounds -- without it every frame is mid-motion and
    there is nothing to actually look at. Held frames are the arrival frame
    written again rather than re-rendered, since a robot standing still produces
    the same image; they cost nothing.
    """
    import cv2

    rig = RetargetRig(target_z_offset=target_z_offset)
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

    move_frames = max(1, round(fps * seconds_per_move))
    hold_frames = max(0, round(fps * hold_seconds))
    camera = dict(
        lookat=lookat,
        width=width,
        height=height,
        distance=distance,
        azimuth=azimuth,
        elevation=elevation,
    )

    worst = (0.0, "")
    written = 0
    # Both renderers held open for the whole video, and given identical camera
    # settings -- see `ToolFrameRenderer`.
    with (
        ToolFrameRenderer(rig.franka_model, **camera) as franka_camera,
        ToolFrameRenderer(rig.model, **camera) as stretch_camera,
    ):

        def render(reached: Reached, label: str) -> np.ndarray:
            hand = f"{reached.franka_aperture * 1000:.0f}mm"
            # Each half gets its *own* reference height, not one shared number.
            # The disk means "where this robot's tool is supposed to be", and
            # with `target_z_offset` in force those are deliberately different
            # heights: the Franka's stays at its own grasp centre, and only
            # Stretch's rises. At offset 0 the two coincide, which is the shared
            # ruler `add_height_plane` was written for.
            # The Franka's grasp centre, but with its axes drawn in *Stretch's*
            # tool convention -- `franka_reference_pose` is that same point with
            # the tool rotation already applied. Its own convention is correct
            # and unhelpful here: the Robotiq reaches along +z and Stretch along
            # +x, so drawing both robots' true frames puts a blue arrow into the
            # counter on one panel and a red one on the other, which reads as a
            # 180 degree error and is not one. Expressed in one convention, the
            # same colour means the same direction on both halves and a real
            # difference is the only thing that shows.
            #
            # The panel says so, because silently redrawing one robot's frame in
            # another's convention is its own trap.
            franka_frame = franka_camera.frame(
                rig.franka_data,
                markers=[
                    Marker(
                        reached.franka_reference_pose,
                        fr.FRANKA_TOOL_COLOR,
                        f"franka {label} / {hand}",
                    )
                ],
                reference_height=float(reached.franka_pose[2, 3]),
                note=FRANKA_AXES_NOTE,
            )
            # Three things on the Stretch panel, which is what a non-zero offset
            # needs: the Franka's own grasp centre (the reference), where Stretch
            # was actually asked to go, and where it got to. With no offset the
            # first two coincide and the middle marker is dropped rather than
            # drawn on top of the first.
            stretch_markers = [
                # A position, not a frame -- see `Marker.axes`.
                Marker(
                    reached.franka_reference_pose,
                    fr.FRANKA_TOOL_COLOR,
                    "franka grasp centre",
                    axes=False,
                )
            ]
            if reached.target_z_offset:
                stretch_markers.append(
                    Marker(
                        reached.expected_pose,
                        fr.REFERENCE_TARGET_COLOR,
                        f"commanded +{reached.target_z_offset * 100:.0f}cm",
                        axes=False,
                    )
                )
            stretch_frame = stretch_camera.frame(
                rig.data,
                markers=stretch_markers
                + [
                    # The one frame on this panel, so its axes are unambiguous.
                    Marker(
                        reached.stretch_pose,
                        fr.STRETCH_TOOL_COLOR,
                        label=f"stretch {reached.position_error * 1000:.0f}mm"
                        f" / jaw {reached.stretch_aperture * 1000:.0f}mm"
                        # Worth saying on the frame: when the jaw is flipped the
                        # two sets of axis arrows are half a turn apart on
                        # purpose, and the label is what stops that reading as a
                        # bug.
                        + (" (jaw flipped)" if reached.jaw_flipped else ""),
                    ),
                ],
                reference_height=float(reached.expected_pose[2, 3]),
                note=STRETCH_AXES_NOTE,
            )
            return fr.side_by_side(franka_frame, stretch_frame)

        total = len(commands) * (move_frames + hold_frames) - move_frames
        click.echo(
            f"rendering {len(commands)} waypoints: {seconds_per_move:.2f}s per move plus "
            f"{hold_seconds:.2f}s held, at {fps}fps -- about {total} frames, "
            f"{total / fps:.1f}s of video"
        )

        for index, (command, label) in enumerate(zip(commands, labels)):
            previous = commands[index - 1] if index else command
            robotiq = WAYPOINTS[index].robotiq

            # No motion into the first pose -- the snap already put the robot
            # there -- so it is established in one command and only held.
            steps = range(1, move_frames + 1) if index else range(1, 2)
            for step in steps:
                blend = step / move_frames if index else 1.0
                reached = rig.command(
                    previous + blend * (command - previous),
                    # The gripper steps rather than ramps: a policy emits a
                    # gripper command per chunk, not a trajectory, and only the
                    # two extremes appear in `WAYPOINTS` -- which also keeps the
                    # Robotiq to two settles for the whole render. See
                    # `settled_franka_gripper`.
                    robotiq=robotiq,
                    # Not restored between frames: this is one continuous motion,
                    # and each solve should seed from the configuration the last
                    # one left the robot in, exactly as it does during a rollout.
                    restore=False,
                )
                frame = render(reached, label)
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                written += 1
                worst = max(worst, (reached.position_error, f"{label} step {step}"))

            # Hold on arrival: the same image again, so the eye has time to
            # compare the two halves before the arm moves off.
            arrival = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            for _ in range(hold_frames):
                writer.write(arrival)
                written += 1

            cv2.imwrite(str(output_dir / f"{index:02d}_{label}.png"), arrival)
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
    click.echo(f"video:  {video_path}  ({written} frames, {written / fps:.1f}s at {fps}fps)")
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
    "--target-z-offset",
    type=float,
    default=0.0,
    show_default=True,
    help="Metres to raise every retargeted target by, as "
    "`FrankaOnStretchView.target_z_offset`. Applies to the checks and to "
    "--visualize. The policy's own default is 0; raise this to see what an offset "
    "costs before sweeping it.",
)
@click.option(
    "--fps",
    type=click.IntRange(min=1),
    default=30,
    show_default=True,
    help="Frame rate of the MP4. The motion is timed in seconds, so this changes smoothness "
    "and render time rather than speed.",
)
@click.option(
    "--seconds-per-move",
    type=click.FloatRange(min=0.0, min_open=True),
    default=1.0,
    show_default=True,
    help="How long the arm takes to travel between two waypoints.",
)
@click.option(
    "--hold-seconds",
    type=click.FloatRange(min=0.0),
    default=0.4,
    show_default=True,
    help="How long to pause on arrival at each waypoint. Held frames are free.",
)
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
    target_z_offset: float,
    fps: int,
    seconds_per_move: float,
    hold_seconds: float,
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

    # Exported so the in-process `pytest.main` below reaches it too, which is
    # what makes one flag serve both modes. See `TARGET_Z_OFFSET_ENV_VAR`.
    os.environ[TARGET_Z_OFFSET_ENV_VAR] = repr(float(target_z_offset))

    if do_visualize:
        visualize(
            output_dir,
            target_z_offset=target_z_offset,
            fps=fps,
            seconds_per_move=seconds_per_move,
            hold_seconds=hold_seconds,
            azimuth=azimuth,
            elevation=elevation,
            distance=distance,
        )
        return

    raise SystemExit(pytest.main(["-v", __file__]))


if __name__ == "__main__":
    cli()
