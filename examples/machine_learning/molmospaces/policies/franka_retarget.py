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
      -> Stretch4KinematicsArmIK.solve -> base / lift / arm extension / wrist targets
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
different in two ways, so `Stretch4KinematicsArmIK` below solves with the
stretch4_kinematics library instead: the target is a full 6-DOF pose (the policy
picks the orientation, not a grasp heuristic), and the holonomic base has to be
able to join the solve.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import os
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np
from mujoco import MjData, MjModel, MjSpec
from scipy.spatial.transform import Rotation as R

from examples.machine_learning.molmospaces.stretch.config import (
    PG4_TOOL_NAME,
    SG4_TOOL_NAME,
    stretch4_tool_name,
)
from examples.machine_learning.molmospaces.stretch.robot_view import (
    SG4_GRIPPER,
    Stretch4RobotView,
    commandable_limits,
)

log = logging.getLogger(__name__)

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

STRETCH_MAX_GRASP_HEIGHT_M = 1.0824
"""
The highest `grasp_center_link` gets, in world metres, at the Franka's home tool pose.

Measured on the mini benchmark's standing robot: the Franka's home puts its grasp
site at 1.1853m and Stretch's lift runs out 0.1028m below that, which is exactly
what `FrankaOnStretchView.measure_tool_height_offset()` returns. Re-measure with

    python -m examples.machine_learning.molmospaces.retargetting.diagnose

which prints the same shortfall under "the target height offset".

A constant rather than a live measurement because the callers that need it are
episode overrides, which run before there is a Stretch to measure -- and because
it is a property of the arm, not of the scene. It is the ceiling for *this* tool
orientation; a pose reaching further out tops out lower, so treat it as the best
case rather than as a bound.
"""

# Per-iteration caps on the pose error an IK step is allowed to chase, and on the
# joint motion it is allowed to ask for. See `_clamped_pose_error`.
MAX_IK_LINEAR_STEP = 0.05  # metres
MAX_IK_ANGULAR_STEP = 0.20  # radians
MAX_IK_JOINT_STEP = 0.20  # radians (or metres, for the prismatic lift and arm)

LIFT_SATURATION_MARGIN_M = 0.002
"""
How close the lift has to be to a travel limit to count as saturated, in metres.

Small, because the IK clamps its solution to the commandable interval, so a
saturated lift sits *exactly* on the limit rather than near it. The margin is
only there to absorb the last step's floating point.
"""

UNREACHABLE_WARNING_M = 0.01
"""
How far short the tool has to fall before a saturated lift is worth warning about.

A centimetre, which is the scale at which a grasp starts missing. Below that the
residual is the IK's normal compromise between five DOFs and a six-dimensional
pose error and says nothing about the lift.
"""

# Robotiq 2F-85 driver-joint angle at each end of the actuator's 0-255 range,
# measured by stepping the Franka model to rest at each end. The policy is fed
# `obs["qpos"]["gripper"][0]`, so these are the units its gripper state is in and
# Stretch's finger angle has to be expressed in them.
ROBOTIQ_DRIVER_OPEN = 0.003
ROBOTIQ_DRIVER_CLOSED = 0.824
ROBOTIQ_CTRL_RANGE = (0.0, 255.0)

# The SG4's finger joint angle, per `gripper_finger_{right,left}_joint`: 0 closed,
# 0.5 rad fully open. **SG4 only.** The PG4's fingers are slides running 0 to
# -0.04 m, so nothing that drives a hand should read these: every open/closed
# position below comes off the gripper move group being driven
# (`StretchGripperGroup.OPEN_JOINT_POS` / `CLOSED_JOINT_POS`), and these remain
# for the SG4-only analysis scripts that predate the PG4.
STRETCH_FINGER_OPEN = SG4_GRIPPER.open_joint_pos
STRETCH_FINGER_CLOSED = SG4_GRIPPER.closed_joint_pos

ROBOTIQ_MAX_APERTURE_M = 0.1885
"""
How wide Stretch's hand is allowed to open, in metres between the fingertips.

The Robotiq 2F-85 opens to 0.087 between its pads and Stretch's hand to 0.1885
between its fingertips, 2.17 times as wide. Left alone that is a domain gap on
the channel a grasping policy cares most about: told to open, Stretch spreads
its fingers more than twice as far as any hand in the checkpoint's training
data. `match_robotiq_aperture` narrows it, and is on by default -- see
`FrankaOnStretchView.finger_open`.

**Matching the tip separation is the wrong invariant, and this is what it should
be matched on instead.** The Robotiq's pads are held parallel by its linkage, so
its jaw is the same width all the way in -- 86.6mm at the grasp site, 86.6mm
three centimetres deeper. Stretch's fingers curve inwards behind their tips and
meet about 7cm back, so its width falls away with depth, and how deep an object
sits is exactly what `grasp_offset_m` decides. `retargetting/aperture.py` puts
both hands in one scene and solves for the tip separation at which Stretch's jaw
is as wide, *where the object will be*, as the Robotiq's is when told to open:

    grasp_offset_m   0.000   0.015   0.030   0.045   0.055
    tips to match     99mm   131mm   132mm   167mm   impossible

At 55mm the hand cannot do it at all: wide open it manages 80mm against the
Robotiq's 87mm. The setting this study shipped for months -- 55mm of offset with
the tips capped at 120mm -- was therefore wrong twice over, once in asking for a
depth the hand cannot open around and once in capping it to less than half what
that depth needs.

**0.1885, not the calibrated 132mm, and the difference is measured.** Replaying
all 20 recorded `franka_baseline` episodes through the retargeting with the
physics on, everything else at `stretch_baseline`:

    offset / tips   0.055/120   0.030/132   0.015/131   0.045/167   0.030/188
    picked             5/20        3/20        3/20        3/20       6/20

Matching the Robotiq's width exactly is the right *fidelity* target -- it makes a
gripper command mean the same gap on both hands -- and it is not the performance
optimum. Opening wider than the Robotiq costs nothing in fit and buys clearance
for the aiming error the retargeting still has, and 0.030/188 is the only
setting in that table that ever picks up the knife. Set `aperture_m` on a trial
to get the calibrated value back; `aperture.py --grasp-offset` recomputes it for
any depth.

At this value `match_robotiq_aperture` is a no-op, because the calibrated
aperture has reached the end of Stretch's own travel. That is not an accident to
be tidied away -- it is the finding.

**Everything above is the SG4.** The PG4's pads are parallel, like the Robotiq's,
and open to 80mm against its 87mm, so it has no wedge to calibrate and no width to
narrow: any aperture from 80mm up is simply "wide open", and this value -- or the
Robotiq's own 87mm -- asks for exactly that.
"""


def stretch_finger_for_aperture(move_group, model, data, aperture_m: float) -> float:
    """The finger angle at which Stretch's jaw is `aperture_m` wide.

    Measured on the model rather than assumed linear: the angle-to-aperture
    relation is close to a straight 0.377 m/rad but not exactly, and this is
    solved once per `FrankaOnStretchView` so there is no reason to approximate
    it. A bisection, because the relation is monotone and the model is the only
    thing that knows it -- and because a closed form would go stale the next time
    the gripper's geometry changes.

    Runs on whatever `data` it is handed, which the caller is expected to make
    scratch data: it moves the fingers to measure them.

    Bisects between the group's own closed and open positions, whichever tool it
    is: the width grows monotonically from one to the other on both, and the
    PG4's open end being the *smaller* number does not change which half a
    bisection keeps.
    """
    low, high = float(move_group.CLOSED_JOINT_POS), float(move_group.OPEN_JOINT_POS)

    def width(angle: float) -> float:
        move_group.joint_pos = [angle, angle]
        mujoco.mj_kinematics(model, data)
        return float(move_group.inter_finger_dist)

    if aperture_m >= width(high):
        # Wider than the hand opens: give it everything it has, rather than
        # silently returning a bisection's midpoint.
        return float(high)
    for _ in range(40):
        middle = 0.5 * (low + high)
        if width(middle) < aperture_m:
            low = middle
        else:
            high = middle
    return float(0.5 * (low + high))

# The pedestal the DROID Franka is bolted to in MolmoSpaces' own scenes, and in
# the notebook this module was ported from: `FrankaRobotConfig(base_size=[0.5,
# 0.5, 0.75])`. It is the height of `fr3_link0` above the floor, which is what
# makes a Franka tool pose land on a countertop rather than under one.
FRANKA_PEDESTAL_HEIGHT = 0.75

FRANKA_MOUNT_OFFSET_XY = (0.0, 0.0)
"""
Where the virtual Franka stands, in Stretch's own base frame.

The policy's actions are interpreted as those of a Franka standing on a 0.75m
pedestal at Stretch's feet, facing the way Stretch faces; that is the frame the
whole retargeting is expressed in, and `franka_mount_pose_from_base()` puts it
wherever the robot happens to be standing. This offset is the "at Stretch's
feet" part, and it is zero because a benchmark stands the two robots at the same
place: `setups._point_base_at` writes one `robot_base_pose` and both the Franka
and the Stretch condition are spawned from it.

**It used to be `(0.05, -0.07)`, and that was 8.6cm of pure error.** The numbers
came from the notebook this module was ported from, which bolted a Franka to one
spot in one kitchen (world [6.8, 9.75], yaw 90) and stood Stretch just behind it
(world [6.73, 9.7], same yaw) -- two robots at two different places, so a policy
action picked out the same point of the room on both only if the mount carried
the difference between them. A benchmark does not stand them apart, so carrying
that difference displaced every commanded grasp by it.

Measured, on the recorded `franka_baseline` episodes replayed through the
retargeting (`retargetting/replay.py`, which compares Stretch's tool against the
Franka's recorded `tcp_pose` step by step):

    offset            closest the tools ever came     at the last step
    (0.05, -0.07)     82-86 mm                        86-87 mm
    (0.0, 0.0)        0.2-0.5 mm                      1-7 mm

Which is the whole of it: with the offset gone the retargeting reproduces the
Franka's own tool trajectory to a fraction of a millimetre, and with it in place
Stretch reached for a point 8.6cm to one side of the object for the entire
episode. Nothing else in the retargeting was ever going to recover that -- the
IK was solving exactly, for the wrong target.

Anything comparing against a *real* Franka standing somewhere other than
Stretch's own base has to say so, by passing `offset_xy` to
`franka_mount_pose_from_base`. Nothing in this repository does: the benchmark
stands them together, and `demo_droid_on_stretch.py` has no Franka to be
consistent with -- it stands Stretch at the notebook's spot and its own comment
already says the virtual Franka is "at Stretch's own feet", which this makes
true. Its grasps move 8.6cm with everything else's.
"""


STRETCH_SPAWN_GRASP_FORWARD_AT_ARM_0_M = 0.4667
"""How far forward of its own base Stretch's grasp centre sits with the arm stowed at 0.

Measured at the mini benchmark's spawn. The telescoping arm extends along this
same axis, so the grasp centre at any extension is this plus the extension --
which is what `stretch_spawn_base_offset_xy` relies on.
"""

STRETCH_SPAWN_GRASP_ACROSS_M = 0.0874
"""How far *across* its own base Stretch's grasp centre sits. Independent of the extension."""

FRANKA_SPAWN_GRASP_FORWARD_M = 0.3069
"""How far forward of the base it is mounted on the Franka's grasp site sits at its home pose."""


@dataclass(frozen=True)
class StretchToolGeometry:
    """The numbers in the retargeting that move with where the tool's grasp centre is.

    The SG4's grasp centre sits 0.2350m out from `quick_connect_interface_link`
    and the PG4's 0.1669m, so every measurement taken *at* `grasp_center_link`
    shifts by that 68.1mm. The module constants above (`STRETCH_MAX_GRASP_HEIGHT_M`,
    `STRETCH_SPAWN_GRASP_*`) are the SG4's values and keep their names;
    `stretch_tool_geometry()` is what the retargeting reads, so the tool published
    for the process decides which it gets.

    The PG4 row was measured on the compiled PG4 model with the procedure that
    reproduces the SG4 row to the tenth of a millimetre (the SG4 height came out
    at 1.0827 against the recorded 1.0824). It has not been re-derived from
    rollouts the way the SG4's `grasp_offset_m` was -- see `grasp_offset_m`.
    """

    max_grasp_height_m: float
    """`STRETCH_MAX_GRASP_HEIGHT_M` for this tool: the lift at its top, tool pointing down."""
    spawn_grasp_forward_at_arm_0_m: float
    """`STRETCH_SPAWN_GRASP_FORWARD_AT_ARM_0_M` for this tool."""
    spawn_grasp_across_m: float
    """`STRETCH_SPAWN_GRASP_ACROSS_M` for this tool. The tool's length does not move it."""
    wrist_camera_reach_m: float
    """`setups.STRETCH_WRIST_CAMERA_REACH_M` for this tool: `gripper_camera_right_rgb`
    to `grasp_center_link`. The camera is on `wrist_roll_link`, not on the tool, so
    only the grasp centre's end of this moves."""
    grasp_offset_m: float
    """The Stretch setups' `grasp_offset_m` default, `setups.STRETCH_GRASP_OFFSET_M`
    for the SG4.

    The PG4's is a geometric first guess, not a searched value. Its grasp centre
    sits exactly on its pads' front edge, and the Robotiq's grasp site 4.0mm behind
    its own; +0.004 puts the two front edges flush, the alignment
    `setups.STRETCH_GRASP_OFFSET_M` explains as the one that survives two
    differently shaped hands. The PG4's pads are parallel and 120mm long, so
    unlike the SG4 a deeper grasp costs it no width. For scale: the SG4's search
    settled 39mm deeper than its own geometric answer, because a replay closes on
    the Franka's step. Both tools now default to their geometric alignment, so
    the two are comparable; a search may move either.
    """


STRETCH_TOOL_GEOMETRY: dict[str, StretchToolGeometry] = {
    SG4_TOOL_NAME: StretchToolGeometry(
        max_grasp_height_m=STRETCH_MAX_GRASP_HEIGHT_M,
        spawn_grasp_forward_at_arm_0_m=STRETCH_SPAWN_GRASP_FORWARD_AT_ARM_0_M,
        spawn_grasp_across_m=STRETCH_SPAWN_GRASP_ACROSS_M,
        wrist_camera_reach_m=0.2414,
        grasp_offset_m=-0.009,
    ),
    PG4_TOOL_NAME: StretchToolGeometry(
        max_grasp_height_m=1.1504,
        spawn_grasp_forward_at_arm_0_m=0.3986,
        spawn_grasp_across_m=0.0874,
        wrist_camera_reach_m=0.1760,
        grasp_offset_m=0.004,
    ),
}


def stretch_tool_geometry(tool_name: str | None = None) -> StretchToolGeometry:
    """`STRETCH_TOOL_GEOMETRY` for `tool_name`, or for the tool published for this process."""
    return STRETCH_TOOL_GEOMETRY[tool_name or stretch4_tool_name()]


def stretch_spawn_base_offset_xy() -> tuple[float, float]:
    """How far to stand Stretch back so its spawn gripper pose is the Franka's, in its own axes.

    Derived rather than hard-coded, because it depends on
    `setups.STRETCH_SPAWN_ARM_M`: the retreat has to cancel where Stretch's grasp
    centre actually is at spawn, and the arm telescopes along the very axis being
    retreated. Pinned to a number it came out at 0.3598 for a 0.2m spawn
    extension and stayed there when the extension changed, which retreated the
    robot 0.1m too far and left `match_stretch_spawn_pose_to_franka` overshooting
    by exactly the difference -- the grippers landing 0.1m apart in the
    *opposite* direction to the one the convention exists to fix.

    At the 0.2m extension this reproduces the measured (-0.3598, 0.0874); with
    the arm stowed at 0 it is the 0.16m the cheap version of this idea costs.

    **It is expensive, which is why `match_stretch_spawn_pose_to_franka` is off
    by default.** Two costs, both scaling with the retreat:

    * **Reach.** Stretch's grasp centre reaches 0.9867m from the base with the
      arm at its 0.52m stop, and the benchmark's objects sit 0.6220m away. After
      the retreat they sit that much further off -- at a 0.2m spawn extension,
      0.9818m, which is 5mm inside the hard limit. Every grasp then depends on
      the base driving back in, and the base accelerates at 0.25 m/s^2. A shorter
      spawn extension buys this back a millimetre for a millimetre.
    * **The camera.** `stretch_baseline` hangs the exo camera off `base_link`, so
      the camera retreats with the robot -- and the whole premise of that setup is
      a camera at the same place in the room as the Franka's. 0.36m is not a
      refinement of that comparison, it is the end of it.

    Heights are not addressed and cannot be: they come out within 7mm on their
    own (1.1782m against 1.1853m), which the base could not have fixed anyway.

    The retreat is cancelled in the virtual Franka's mount (see
    `franka_mount_pose_from_base`'s `offset_xy`), so the frame a policy action
    means is unchanged: what moves is the robot, not the retargeting.
    """
    from examples.machine_learning.molmospaces.retargetting.setups import (
        STRETCH_SPAWN_ARM_M,
    )

    geometry = stretch_tool_geometry()
    forward = (
        geometry.spawn_grasp_forward_at_arm_0_m
        + float(STRETCH_SPAWN_ARM_M)
        - FRANKA_SPAWN_GRASP_FORWARD_M
    )
    return (-forward, geometry.spawn_grasp_across_m)


def pose_matrix(pos, quat_wxyz) -> np.ndarray:
    """A 4x4 homogeneous transform from a position and a (w, x, y, z) quaternion."""
    pose = np.eye(4)
    pose[:3, :3] = R.from_quat(np.asarray(quat_wxyz, dtype=float), scalar_first=True).as_matrix()
    pose[:3, 3] = np.asarray(pos, dtype=float)
    return pose


POSE_CONVENTION_ENV_VARS = {
    "change_franka_start_pose_limit_height": "STRETCH4_CHANGE_FRANKA_START_POSE_LIMIT_HEIGHT",
    "change_stretch_start_pose_pitch_deg": "STRETCH4_CHANGE_STRETCH_START_POSE_PITCH_DEG",
    "match_stretch_spawn_pose_to_franka": "STRETCH4_MATCH_STRETCH_SPAWN_POSE_TO_FRANKA",
}
"""
One environment variable per field of `PoseConventions`, named after its flag.

Environment variables rather than arguments because the places that need to know
are episode overrides and policy constructors, which `run_evaluation` builds
itself from a "module:Class" string in worker processes it forks -- there is no
seam to pass a flag along. Same route, and the same reason, as
`setups.publish_params` and `configs.MOLMOBOT_ACTION_TYPE_ENV_VAR`.

One variable per field rather than one for all of them, so that a run which sets
some and not others cannot be confused with a run of an older build that knew
about fewer.
"""


@dataclass(frozen=True)
class PoseConventions:
    """How the two robots' start poses and wrist frames are made to relate.

    Every field defaults to False, and each is asked for by its own flag. Off,
    the Franka starts at the DROID home the checkpoint was trained from and the
    retargeting is the bare `FRANKA_TO_STRETCH_TOOL` -- which is the condition
    every measurement in this package was taken under unless it says otherwise.

    These are conventions rather than parameters: none of them changes what a
    grasp *is*, only where an episode begins and how the wrist is posed at the start.
    That is exactly why they need naming and publishing rather than hard-coding --
    a comparison whose two halves disagree about a convention is not a comparison.
    """

    change_franka_start_pose_limit_height: bool = False
    """Cap the Franka's start tool height at `STRETCH_MAX_GRASP_HEIGHT_M`.

    Without it the Stretch condition begins every episode already saturated: the
    Franka's home puts its grasp site 10.3cm above the highest Stretch's lift can
    put its own with the tool pointing down, so the arm sits at its stop reaching
    for a pose it cannot hold and reports proprioception from a configuration it
    never reached.
    """

    change_stretch_start_pose_pitch_deg: float = 0.0
    """Pitch Stretch's wrist by this many degrees at the snap, and only at the snap.

    Degrees about Stretch's tool y -- the jaw line -- applied to the pose
    `snap_to_franka_joint_pos` writes the robot into, so positive tips the
    gripper up towards horizontal and negative tips it down. The Franka's home
    points its hand straight down, and from there Stretch's gripper camera looks
    down its own approach axis at whatever is directly beneath the hand rather
    than out across the counter. This is the knob for "let the first frames see
    the workspace".

    **Start only, by construction.** It is applied to the snap target and to
    nothing else: every step after it goes through `retarget_franka_joint_pos`
    untouched, and `franka_joint_pos` reports the pitched arm honestly rather
    than hiding it. So the policy's first observation shows a wrist aimed across
    the workspace, its proprioception agrees that the wrist is aimed there, and
    from the first action onwards the pose is whatever the policy asks for. The
    pitch decays because the policy commands it away, not because anything here
    cancels it.

    That honesty is the whole difference from `RetargetParams.wrist_tilt_deg`,
    which is the same rotation folded into the *tool transform* instead. That one
    is subtracted again by `_tool_correction_inverse` before the policy sees it,
    so the hand sits 19.5 or 45 degrees from where the checkpoint reads it as
    being, with nothing in the observation saying so -- a constant, unobservable
    offset in a loop closed through the wrist camera, which the policy drives out
    within a few steps while believing it is holding still. Read that field for
    the measurement. This one claims less and therefore survives being true: it
    moves the start and says so.

    Stretch only, and it does not need `snap_to_franka_home` -- it *is* the snap.

    **Tried, and it costs grasps where the policy was working.** Eight episodes
    per cell, everything else held:

        setup              pitch 0      +19        +30
        stretch_baseline    5/8        1/8        2/8
        stretch_stretchcam  2/8         --        2/8

    and the continuous measure moves the same way -- mean closest approach to the
    object on `baseline` goes from 58mm to 100mm at +19 and 109mm at +30, worse
    in almost every episode rather than in one or two, which is what makes the
    shift readable at n=8 when the counts alone would not be.

    Not the retargeting: position error stays at 0.6-2.3mm and orientation at
    0.0003-0.0022 rad across all three. The arm went where it was told; the
    policy told it somewhere worse.

    The split between the two setups is the whole explanation. `baseline` is the
    condition whose exo camera *is* the DROID shoulder camera the checkpoint
    trained on, and it is the best Stretch result in the study -- so the prior
    driving it is both confident and correct, which is exactly what an
    out-of-distribution opening derails: the first observation carries a Franka
    joint vector and an arm posture no DROID episode begins in, and the policy
    commits a whole action chunk to it. `stretchcam` is already at 2/8 through an
    exo camera it does not recognise, so the same pitch changes nothing. You can
    only break what was working.

    Kept off by default and kept at all because the measurement is worth having:
    it says the opening observation matters a great deal on the one Stretch
    configuration that works, which is a fact about where to spend effort.
    """

    match_stretch_spawn_pose_to_franka: bool = False
    """Stand Stretch back far enough that its spawn gripper pose is the Franka's.

    Stretch's arm reaches *further* at its shortest than the Franka's does at its
    home -- 0.467m against 0.307m from the base, before the spawn extension in
    `setups.STRETCH_SPAWN_ARM_M` adds its own -- so the only way to make the two
    grippers start in the same place is to stand Stretch further back. See
    `fr.stretch_spawn_base_offset_xy`, which measures how far, and what it
    costs; it is off by default because what it costs is most of the arm's
    remaining reach and the exo camera's agreement with the Franka's.

    The virtual Franka does *not* move with it: the retreat is cancelled in the
    mount, so the frame a policy action is interpreted in is unchanged and only
    the physical robot has moved. That is what keeps this a change to where
    Stretch stands rather than a change to what the retargeting means.
    """

    def __bool__(self) -> bool:
        """True when any convention is being changed from the default."""
        return any(getattr(self, field) for field in POSE_CONVENTION_ENV_VARS)

    @property
    def changes_franka_start_pose(self) -> bool:
        return self.change_franka_start_pose_limit_height

    def describe(self) -> str:
        asked = [
            name if isinstance(getattr(self, name), bool) else f"{name}={getattr(self, name):g}"
            for name in POSE_CONVENTION_ENV_VARS
            if getattr(self, name)
        ]
        return ", ".join(asked) if asked else "none (DROID home)"


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() not in ("", "0", "false", "no")


def _pose_convention_types() -> dict[str, type]:
    """Each convention's declared type, so the environment round-trip cannot guess wrong.

    Read off the dataclass rather than listed separately: a convention added as a
    number and plumbed as a flag would publish "1" and come back as 1.0 degree,
    which is a silently different run rather than an error.
    """
    return {f.name: f.type for f in dataclasses.fields(PoseConventions)}


def _env_number(name: str) -> float:
    """One numeric convention off the environment. Unset, unparseable or blank is 0."""
    raw = os.environ.get(name, "").strip()
    try:
        return float(raw)
    except ValueError:
        if raw:
            log.warning(f"[retarget] {name}={raw!r} is not a number; reading it as 0.")
        return 0.0


def pose_conventions_requested() -> PoseConventions:
    """Which pose conventions this process was asked for."""
    types = _pose_convention_types()
    return PoseConventions(
        **{
            field: (_env_flag(variable) if types[field] is bool else _env_number(variable))
            for field, variable in POSE_CONVENTION_ENV_VARS.items()
        }
    )


def publish_pose_conventions(conventions: PoseConventions) -> None:
    """Put the pose conventions in the environment, for this process and its workers.

    Every variable is written in both directions: a run that does *not* ask for a
    convention must clear a variable an earlier run in the same shell exported,
    or the two halves of a comparison silently start at different poses -- which
    is precisely the failure these flags were added to remove.
    """
    for field, variable in POSE_CONVENTION_ENV_VARS.items():
        value = getattr(conventions, field)
        if not value:
            os.environ.pop(variable, None)
        elif isinstance(value, bool):
            os.environ[variable] = "1"
        else:
            os.environ[variable] = repr(float(value))


def stretch_startable_arm_qpos(
    franka: "VirtualFranka",
    mount_height_m: float = FRANKA_PEDESTAL_HEIGHT,
    *,
    max_height_m: float | None = STRETCH_MAX_GRASP_HEIGHT_M,
    iterations: int = 300,
) -> np.ndarray:
    """The Franka's home arm configuration, lowered so Stretch can start there too.

    The Franka's home puts its grasp site at 1.185m and Stretch's lift tops out
    10.3cm below that (`STRETCH_MAX_GRASP_HEIGHT_M`), so unretargeted the Stretch
    condition begins every episode already saturated, reaching for a pose it
    cannot hold and reporting proprioception from a configuration it never
    actually reached. Capping the Franka's start at `max_height_m` is what makes
    "both robots start in the same place" true rather than aspirational.
    `max_height_m=None` returns `franka.init_qpos` unchanged, which is what makes
    "the flag was not passed" cost nothing rather than round-trip the home pose
    through an IK solve.

    The clamp is applied to the tool's z *in the arm's own base frame*, offset by
    `mount_height_m`, which is the same number as world z only because every mount
    this module builds is level -- `franka_mount_pose_from_base` yaws and nothing
    else. Asserted there rather than re-derived here.

    Returns seven joint angles, IK'd from the home configuration so the result is
    the nearest way to hold that pose rather than an unrelated branch. The height
    is a *cap*, not a move: a home pose already below the ceiling is left at its
    own height.
    """
    if max_height_m is None:
        return np.asarray(franka.init_qpos, dtype=float).copy()
    pose = franka.fk(franka.init_qpos)
    pose[2, 3] = min(float(pose[2, 3]), float(max_height_m) - float(mount_height_m))
    return franka.ik(pose, franka.init_qpos, iterations=iterations)


def franka_start_arm_qpos(
    franka: "VirtualFranka",
    conventions: PoseConventions,
    mount_height_m: float = FRANKA_PEDESTAL_HEIGHT,
) -> np.ndarray:
    """`stretch_startable_arm_qpos` driven by a `PoseConventions`.

    The one place the Franka start-pose flag is turned into an argument, so the
    Franka half of the change (`setups.franka_episode_override`) and the Stretch
    half (`FrankaOnStretchView`) cannot disagree about what the flag means.
    """
    return stretch_startable_arm_qpos(
        franka,
        mount_height_m,
        max_height_m=(
            stretch_tool_geometry().max_grasp_height_m
            if conventions.change_franka_start_pose_limit_height
            else None
        ),
    )


def franka_mount_pose_from_base(
    base_xytheta,
    pedestal_height: float = FRANKA_PEDESTAL_HEIGHT,
    offset_xy: tuple[float, float] | None = None,
):
    """Where the virtual Franka stands, given where Stretch is standing.

    `base_xytheta` is what `StretchBaseGroup.joint_pos` reports: the base's
    (x, y, yaw) in world coordinates. The mount is built from those three numbers
    rather than from the base's 4x4 pose so that a base frame that is pitched or
    rolled -- a robot on a ramp, or mid-transient after a reset -- cannot tip the
    virtual Franka over with it.

    `offset_xy` moves the virtual Franka off Stretch's base, in the base's own
    axes, and defaults to `FRANKA_MOUNT_OFFSET_XY` -- which is zero, because a
    benchmark stands both robots at the same place. Pass it only to model a real
    Franka that stood somewhere else; whatever you pass displaces every grasp by
    exactly that much, so it wants a measurement behind it. See
    `FRANKA_MOUNT_OFFSET_XY` for the one that used to be here.

    Under `match_stretch_spawn_pose_to_franka` the default gains the *opposite*
    of `stretch_spawn_base_offset_xy()`, which is how that convention moves the
    robot without moving the frame: Stretch stands back, the virtual Franka
    stays where the real one is, and a policy action still means the same point
    of the same room. Read from the environment rather than threaded through
    every caller for the reason `POSE_CONVENTION_ENV_VARS` gives -- and because
    a caller that forgot would silently undo the cancellation.
    """
    x, y, theta = np.asarray(base_xytheta, dtype=float).reshape(-1)[:3]
    if offset_xy is None:
        offset_xy = FRANKA_MOUNT_OFFSET_XY
        if pose_conventions_requested().match_stretch_spawn_pose_to_franka:
            retreat = stretch_spawn_base_offset_xy()
            offset_xy = (offset_xy[0] - retreat[0], offset_xy[1] - retreat[1])
    base = pose_matrix([x, y, 0.0], R.from_euler("z", theta).as_quat(scalar_first=True))
    offset = pose_matrix([offset_xy[0], offset_xy[1], pedestal_height], [1, 0, 0, 0])
    return base @ offset


def robotiq_ctrl_from_driver(driver_angle) -> float:
    """A Robotiq driver-joint angle expressed as a 0-255 command.

    The inverse of the arithmetic in `FrankaOnStretchView.retarget_robotiq_ctrl`,
    and the way to ask "what command would leave the Franka's hand like this?" --
    used to start Stretch's hand where a DROID episode starts the Franka's.
    """
    fraction = np.clip(
        (float(np.asarray(driver_angle, dtype=float).reshape(-1)[0]) - ROBOTIQ_DRIVER_OPEN)
        / (ROBOTIQ_DRIVER_CLOSED - ROBOTIQ_DRIVER_OPEN),
        0.0,
        1.0,
    )
    return float(ROBOTIQ_CTRL_RANGE[0] + fraction * (ROBOTIQ_CTRL_RANGE[1] - ROBOTIQ_CTRL_RANGE[0]))


def robotiq_ctrl_from_stretch_fingers(
    finger_angles,
    finger_open: float = STRETCH_FINGER_OPEN,
    finger_closed: float = STRETCH_FINGER_CLOSED,
) -> float:
    """Stretch's finger angles expressed as the Robotiq 0-255 that would produce them.

    The other inverse of `retarget_robotiq_ctrl`: it answers "what has this hand
    been told?" from where the fingers actually are, so the proxy can report a
    gripper command it has established rather than one it assumed.

    `finger_open` is the angle that counts as fully open, and must be the same one
    the forward mapping used -- `FrankaOnStretchView.finger_open`, which is
    narrowed to the Robotiq's aperture by default. Passing the module constant
    while the forward direction used a narrowed angle would make the two stop
    being inverses, which is exactly the kind of drift that shows up as a policy
    reading its own gripper command back wrong.

    `finger_closed` is the tool's shut position. The defaults are the SG4's;
    `FrankaOnStretchView` passes its own, which is what makes this right on the
    PG4, whose "open" is the negative end of its travel.
    """
    finger = float(np.mean(np.asarray(finger_angles, dtype=float)))
    open_fraction = np.clip((finger - finger_closed) / (finger_open - finger_closed), 0.0, 1.0)
    # 0 is open on the Robotiq and closed on Stretch, so an open hand is ctrl 0.
    return float(
        ROBOTIQ_CTRL_RANGE[1] + open_fraction * (ROBOTIQ_CTRL_RANGE[0] - ROBOTIQ_CTRL_RANGE[1])
    )


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


def _prioritised_step(
    tasks: list[tuple[np.ndarray, np.ndarray]],
    damping: float,
    joint_scale: np.ndarray | None = None,
) -> np.ndarray:
    """A joint step that serves `tasks` in order, each only with what the ones before leave.

    `tasks` is `(jacobian, error)` pairs, highest priority first. Each is solved
    within the null space of every task above it, so a lower one can never move
    a higher one off what it achieved. That removes the exchange rate a single
    least-squares solve has to choose: Stretch's five arm DOFs cannot both place
    the gripper and orient it freely, and solved together, poses the wrist cannot
    orient bleed into position error -- measured, a target 18cm inside the
    reachable envelope came back 18cm short because the solver was buying
    unreachable orientation with it.

    `joint_scale` weights the joints against each other: a joint scaled to half
    contributes half as much to the same step, so the solve reaches for it only
    when the others cannot do the job. It is applied as a change of variables
    (solve in `q / scale`, scale the answer back), which leaves the priority
    structure untouched.
    """
    width = tasks[0][0].shape[1]
    scale = np.ones(width) if joint_scale is None else joint_scale
    step = np.zeros(width)
    null_space = np.eye(width)
    for jacobian, error in tasks:
        scaled = jacobian * scale
        projected = scaled @ null_space
        inverse = _damped_pseudo_inverse(projected, damping)
        step = step + null_space @ (inverse @ (error - scaled @ step))
        null_space = null_space - null_space @ (inverse @ projected)
    return step * scale


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
        self.default_init_qpos = self.init_qpos.copy()
        """
        The arm configuration `FrankaRobotConfig` ships, kept whatever `init_qpos` becomes.

        `init_qpos` is rewritten in place when either start-pose flag is on
        (see `FrankaOnStretchView.__init__`), because everything that asks "where
        does the Franka start" has to get the same answer. This is the other
        question -- "where does the Franka *normally* start" -- which anything
        holding a fixed reference against a moving start pose needs, and which a
        rewritten `init_qpos` would otherwise have destroyed.
        """
        self.init_gripper_qpos = np.asarray(config.init_qpos["gripper"], dtype=float)
        """The Robotiq driver angles a DROID episode starts from. See `robotiq_ctrl_from_driver`."""

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


IK_BASE_TRANSLATION_ENV_VAR = "STRETCH4_IK_BASE_TRANSLATION"
"""
Whether the base may translate in the IK, as well as turn.

An environment variable for the reason `POSE_CONVENTION_ENV_VARS` are: the policy
that builds the view is constructed by `run_evaluation` in worker processes,
from a class name, with no seam to pass an argument through.
"""


@dataclass(frozen=True)
class IKChoice:
    """How `Stretch4KinematicsArmIK` -- the one IK there is -- may use the base.

    Whether the base is in the IK at all is `include_base`.
    """

    base_translation: bool = False
    """Let the base translate as well as rotate.

    Off, a base in the IK only turns in place (`Stretch4IKModes.BASE_ROTATE`);
    on, it is the full planar base (`BASE_PLANAR`).
    """

    def describe(self) -> str:
        return "stretch4_kinematics" + (", base translation on" if self.base_translation else "")


def ik_choice_requested() -> IKChoice:
    """Which IK options this process was asked for."""
    return IKChoice(base_translation=_env_flag(IK_BASE_TRANSLATION_ENV_VAR))


def publish_ik_choice(choice: IKChoice) -> None:
    """Put the IK choice in the environment, for this process and its workers.

    Written in both directions, as `publish_pose_conventions` does and for the
    same reason.
    """
    if choice.base_translation:
        os.environ[IK_BASE_TRANSLATION_ENV_VAR] = "1"
    else:
        os.environ.pop(IK_BASE_TRANSLATION_ENV_VAR, None)


def _planar_pose(x: float, y: float, theta: float) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, :3] = R.from_euler("z", theta).as_matrix()
    pose[:2, 3] = (x, y)
    return pose


def _translation(offset: np.ndarray) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, 3] = offset
    return pose


class Stretch4KinematicsArmIK:
    """Solves for Stretch's base / lift / telescoping arm / wrist given a tool pose,
    with the stretch4_kinematics library. The only IK the retargeting uses.

    The solve runs on the library's Pinocchio model of the URDF -- its forward
    kinematics and its Jacobians -- with a task-priority loop of this module's
    in place of the library's own CLIK; see `_clik` for why and what it
    measured. The rest of what this class adds is what it takes to put that
    model and the MuJoCo one in the same place:

    * **Frames.** The two descriptions agree on every joint and every rotation,
      but not on two translations: the URDF's base origin sits 41mm behind the
      MJCF's, and the library's URDF always carries the SG4, so on a PG4 the
      grasp centre is somewhere else on the hand. Both are rigid, so they are
      measured rather than assumed -- `_calibrate` fits one offset in the base
      frame and one in the tool frame off a handful of poses -- and every target
      is moved into the URDF's frames before it is solved.
    * **Limits.** The library's lift / arm / wrist limits are replaced with the
      MJCF's commandable ones, as `kinematics.StretchReachSolver` does, so a
      solution is something the controllers can reach. Its planar model has no
      in-loop clipping at all (its configuration is not a plain vector), so
      there the solution is clipped afterwards.
    * **The base.** `include_base` off fixes it. On, it turns in place
      (`BASE_ROTATE`), and `base_translation` lets it drive as well
      (`BASE_PLANAR`). Either way it is held to `base_leash` around where
      `releash()` last found it.

    Position outranks the approach direction, which outranks the roll about it
    (`_prioritised_step`), so on a pose the five arm DOFs cannot orient, the
    miss is the roll. The residual this
    returns is measured on the MuJoCo model.
    """

    ARM_GROUPS = ("lift", "arm", "wrist")
    TOOL_FRAME = "grasp_center_link"
    CALIBRATION_POSES = 8
    CALIBRATION_WARNING_M = 1e-3
    DRIVE_GAIN_M = 1e-3
    """How much closer driving has to get the tool than turning in place, to be worth it."""
    SETTLED_POSITION_M = 1e-4
    SETTLED_ORIENTATION_RAD = 1e-3
    SETTLE_WINDOW = 20
    """When a solve is done for practical purposes, well inside what anything here measures
    (`tests/test_retargeting.py` holds 5mm and 0.02 rad): the tool within 0.1mm, and
    oriented within a milliradian or no longer improving by a tenth of one over
    `SETTLE_WINDOW` iterations."""
    STALL_STEP = 1e-6
    """A solve whose configuration moved less than this in one iteration has settled, in
    radians or metres -- for a pose it cannot match exactly, on the best compromise."""
    WRIST_LIMIT_MARGIN_RAD = 0.5
    WRIST_LIMIT_FLOOR = 0.1
    """How the wrist is kept off its stops: inside this margin of a limit a wrist joint's
    column is scaled down linearly, to `WRIST_LIMIT_FLOOR` at the stop itself.

    A weighted least-norm, so the solve prefers the configurations that leave the
    wrist room to orient. Without it, walking `tests/test_retargeting.py`'s path
    drove `wrist_roll_joint` onto -1.135 rad on the way to `out_and_left` and the
    tool could not be turned the rest of the way: 0.187 rad short, with position
    exact. With it, 0.030 rad, and the largest step the wrist takes between two
    commands (1.21 rad) is no larger than the retargeting's earlier MuJoCo IK's
    (1.34). The wrist only: a lift or arm at its stop is where a reach
    legitimately ends.
    """

    def __init__(
        self,
        stretch_view: Stretch4RobotView,
        namespace: str,
        include_base: bool = True,
        base_translation: bool = False,
        base_leash: tuple[float, float, float] = (0.7, 0.15, math.pi / 3),
        base_cost: float = 5.0,
        iterations: int = 100,
        damping: float = 0.08,
        tolerance: float = 1e-5,
    ) -> None:
        if base_translation and not include_base:
            raise ValueError("base_translation needs include_base: there is no base in the IK.")

        from stretch4_kinematics import StretchKinematics

        self._live_view = stretch_view
        self._live_data: MjData = stretch_view.mj_data
        self._scratch_data = MjData(self._live_data.model)
        self._scratch_view = Stretch4RobotView(self._scratch_data, namespace)

        self.GROUPS = (("base",) if include_base else ()) + self.ARM_GROUPS
        self._include_base = include_base
        self._base_translation = base_translation
        self._iterations = iterations
        self._damping = damping
        self._tolerance = tolerance
        self._widths = [self._scratch_view.get_move_group(g).pos_dim for g in self.GROUPS]
        self._arm_limits = np.concatenate(
            [commandable_limits(self._scratch_view.get_move_group(g)) for g in self.ARM_GROUPS]
        )
        self._base_leash = np.asarray(base_leash, dtype=float)
        self._base_cost = float(base_cost)
        self._home = np.zeros(3)

        # Two models: the planar one drives the base in x, y and theta, and the
        # rotate one only in theta -- which is also how the base is fixed, its
        # rotation pinned at 0 (see `_rotation_limits`).
        self._kinematics = StretchKinematics()
        for model in (self._kinematics.model, self._kinematics.model_ik):
            arm_start = model.joints[model.getJointId("lift_joint")].idx_q
            model.lowerPositionLimit[arm_start:] = self._arm_limits[:, 0]
            model.upperPositionLimit[arm_start:] = self._arm_limits[:, 1]

        self._base_offset = np.zeros(3)
        self._tool_offset = np.zeros(3)
        self.releash()

    def releash(self) -> None:
        """Re-centre the base's leash on where the robot is standing, and re-measure the frames.

        The frame offsets are measured
        here too because this is called on every `FrankaOnStretchView.reset()`,
        and an episode in a new house may stand the robot at a new height.
        """
        self._home = np.asarray(self._live_view.get_move_group("base").joint_pos, dtype=float)
        self._calibrate()

    def _calibrate(self) -> None:
        """Fit the base-frame and tool-frame offsets between the MJCF and the library's URDF.

        For each pose, the MuJoCo tool pose in the base frame is
        `T(base_offset) @ urdf_tool_pose @ T(tool_offset)`. The rotations are the
        same on both sides, so the translations give `base_offset + R @
        tool_offset = mujoco - urdf`, which is linear in the two offsets.
        """
        from stretch4_kinematics import StretchJointPositions

        self._scratch_data.qpos[:] = self._live_data.qpos
        base_inverse = np.linalg.inv(self._base_pose(self._home))
        rng = np.random.default_rng(0)
        rows, gaps, rotation_gap = [], [], 0.0
        for _ in range(self.CALIBRATION_POSES):
            joint_pos = rng.uniform(self._arm_limits[:, 0], self._arm_limits[:, 1])
            self._write_arm(self._scratch_view, joint_pos)
            mujoco.mj_kinematics(self._live_data.model, self._scratch_data)
            mujoco_tool = (
                base_inverse @ self._scratch_view.get_move_group("wrist").leaf_frame_to_world
            )
            urdf_tool = self._kinematics.forward(
                StretchJointPositions(
                    lift=joint_pos[0],
                    arm=joint_pos[1],
                    wrist_yaw=joint_pos[2],
                    wrist_pitch=joint_pos[3],
                    wrist_roll=joint_pos[4],
                ),
                self.TOOL_FRAME,
            )
            rotation = R.from_matrix(urdf_tool.rotation.T @ mujoco_tool[:3, :3])
            rotation_gap = max(rotation_gap, float(np.linalg.norm(rotation.as_rotvec())))
            rows.append(np.hstack([np.eye(3), urdf_tool.rotation]))
            gaps.append(mujoco_tool[:3, 3] - urdf_tool.translation)
        rows, gaps = np.vstack(rows), np.concatenate(gaps)
        offsets = np.linalg.lstsq(rows, gaps, rcond=None)[0]
        self._base_offset, self._tool_offset = offsets[:3], offsets[3:]
        fit = float(np.max(np.abs(rows @ offsets - gaps)))
        if fit > self.CALIBRATION_WARNING_M or rotation_gap > self.CALIBRATION_WARNING_M:
            log.warning(
                f"[stretch4_kinematics] the URDF and the MJCF disagree by more than a rigid "
                f"offset: {fit * 1000:.1f}mm, {rotation_gap:.4f} rad after fitting one. The "
                "IK will aim off by about that much."
            )

    @staticmethod
    def _base_pose(base: np.ndarray) -> np.ndarray:
        return _planar_pose(*base)

    def _write_arm(self, view: Any, joint_pos: np.ndarray) -> None:
        offset = 0
        for group in self.ARM_GROUPS:
            width = view.get_move_group(group).pos_dim
            view.get_move_group(group).joint_pos = joint_pos[offset : offset + width]
            offset += width

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

    def _rotation_limits(self, theta: float) -> tuple[float, float]:
        """The rotation joint's interval this solve, relative to where the base faces now."""
        if not self._include_base:
            return 0.0, 0.0
        low = self._home[2] - self._base_leash[2] - theta
        high = self._home[2] + self._base_leash[2] - theta
        return max(low, -math.pi), min(high, math.pi)

    def _clik(self, planar: bool, target: np.ndarray, seed: np.ndarray) -> np.ndarray:
        """Solve the library's model for `target`, from `seed`: position first, then orientation.

        The model, its forward kinematics and its Jacobians are stretch4_kinematics';
        the loop is not the library's `_closed_loop_inverse_kinematics`, and the
        difference is measured rather than stylistic. That one takes a full,
        nearly undamped Gauss-Newton step and clips to the limits afterwards, and
        trades position against orientation in one least-squares solve. Walking a
        policy's joint-space path with it, `tests/test_retargeting.py` measured
        the tool 90mm off, 41 of 180 steps past 5mm: a step from a seed near a
        limit would throw the wrist across its range into the opposite limit and
        stay there, and a pose the wrist could not orient cost position too.

        So, as the retargeting's own solver used to: `_prioritised_step` solves
        position outright, then the approach direction, then the roll about it,
        each only in what the ones before leave, the pose error a
        step chases is capped (`MAX_IK_LINEAR_STEP`, `MAX_IK_ANGULAR_STEP`) and so
        is the joint motion it may ask for (`MAX_IK_JOINT_STEP`), the base's
        columns are scaled by `1 / base_cost` so it moves only when the arm cannot
        do the job, the wrist's are scaled down near its stops
        (`WRIST_LIMIT_MARGIN_RAD`), and the limits are applied every iteration --
        on the planar model too, whose arm the library leaves unclipped. Measured
        on the same path: under 1mm worst, none past 5mm.
        """
        import pinocchio as pin

        model, data = (
            (self._kinematics.model, self._kinematics.data)
            if planar
            else (self._kinematics.model_ik, self._kinematics.data_ik)
        )
        frame = model.getFrameId(self.TOOL_FRAME)
        base_dofs = 3 if planar else 1
        base_scale = np.ones(model.nv)
        base_scale[:base_dofs] = 1.0 / self._base_cost
        wrist_low, wrist_high = self._arm_limits[2:, 0], self._arm_limits[2:, 1]

        q = np.array(seed, dtype=float)
        q = self._clip_to_limits(model, q, planar)
        orientation_history: list[float] = []
        for _ in range(self._iterations):
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)
            current = data.oMf[frame]
            linear = target[:3, 3] - current.translation
            angular = pin.log3(target[:3, :3] @ current.rotation.T)
            if np.linalg.norm(np.concatenate([linear, angular])) < self._tolerance:
                break
            # Done for practical purposes: placed, and either oriented or no longer
            # getting any more so. Without this, a pose with the arm against a stop
            # crawls the last fraction of a millimetre for a thousand iterations.
            orientation_history.append(float(np.linalg.norm(angular)))
            if np.linalg.norm(linear) < self.SETTLED_POSITION_M and (
                orientation_history[-1] < self.SETTLED_ORIENTATION_RAD
                or (
                    len(orientation_history) > self.SETTLE_WINDOW
                    and orientation_history[-1 - self.SETTLE_WINDOW] - orientation_history[-1]
                    < self.SETTLED_ORIENTATION_RAD / 10
                )
            ):
                break
            jacobian = pin.computeFrameJacobian(
                model, data, q, frame, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )
            # The wrist is the last three joints in either model's configuration.
            wrist = q[-3:]
            room = np.minimum(wrist - wrist_low, wrist_high - wrist)
            joint_scale = base_scale.copy()
            joint_scale[-3:] *= np.clip(
                room / self.WRIST_LIMIT_MARGIN_RAD, self.WRIST_LIMIT_FLOOR, 1.0
            )
            # Position, then the approach direction, then the roll about it: on a
            # pose the wrist cannot fully orient, the gripper still points where
            # it was asked to and only its roll falls short. +x is the approach;
            # see `FRANKA_TO_STRETCH_TOOL`.
            approach = current.rotation[:, 0]
            across = np.eye(3) - np.outer(approach, approach)
            tasks = [
                (jacobian[:3], _clamp_norm(linear, MAX_IK_LINEAR_STEP)),
                (
                    across @ jacobian[3:],
                    _clamp_norm(np.cross(approach, target[:3, 0]), MAX_IK_ANGULAR_STEP),
                ),
                (
                    approach[None, :] @ jacobian[3:],
                    np.clip([float(angular @ approach)], -MAX_IK_ANGULAR_STEP, MAX_IK_ANGULAR_STEP),
                ),
            ]
            step = _prioritised_step(tasks, self._damping, joint_scale)
            step = np.clip(step, -MAX_IK_JOINT_STEP, MAX_IK_JOINT_STEP)
            moved = q
            q = self._clip_to_limits(model, pin.integrate(model, q, step), planar)
            # Settled on a compromise: a pose the wrist cannot fully orient never
            # reaches `tolerance`, and the iterations after this point move nothing.
            if np.max(np.abs(q - moved)) < self.STALL_STEP:
                break
        return q

    def _clip_to_limits(self, model: Any, q: np.ndarray, planar: bool) -> np.ndarray:
        """`q` inside the joint limits, and a planar base inside the leash's extent.

        The rotate model's configuration is a plain vector and carries its own
        limits (the arm's are the MJCF's, see `__init__`; the turn's are set per
        solve by `_solve_turning`). The planar model's base is `[x, y, cos, sin]`,
        a move from where the robot stands, so it is held to the leash's size
        here and to the leash itself -- which is centred on `releash()`'s pose,
        not this one -- by `_solve_driving` afterwards.
        """
        if not planar:
            return np.clip(q, model.lowerPositionLimit, model.upperPositionLimit)
        q = q.copy()
        q[4:] = np.clip(q[4:], self._arm_limits[:, 0], self._arm_limits[:, 1])
        q[:2] = np.clip(q[:2], -self._base_leash[:2], self._base_leash[:2])
        theta = float(np.clip(math.atan2(q[3], q[2]), -self._base_leash[2], self._base_leash[2]))
        q[2], q[3] = math.cos(theta), math.sin(theta)
        return q

    def _to_urdf(self, local_target: np.ndarray, lever: np.ndarray) -> np.ndarray:
        """A tool target in the MuJoCo base frame -> the library's. `lever` is the base offset."""
        return _translation(-lever) @ local_target @ _translation(-self._tool_offset)

    def _solve_turning(
        self,
        base: np.ndarray,
        target_pose: np.ndarray,
        arm: np.ndarray,
        limits: tuple[float, float],
    ) -> tuple[float, np.ndarray]:
        """The library's rotate-in-place solve from `base`: the turn, and the arm joints.

        MuJoCo turns the base about its own origin and the library about the
        URDF's, `base_offset` away -- so the offset to take out depends on the
        turn being solved for. A fixed point, settled in a pass or two because
        the turn barely moves the lever arm. `limits` (0, 0) is the fixed base.
        """
        model = self._kinematics.model_ik
        model.lowerPositionLimit[0], model.upperPositionLimit[0] = limits
        local = np.linalg.inv(self._base_pose(base)) @ target_pose
        turn = 0.0
        q = np.concatenate([[0.0], arm])
        for _ in range(1 if limits == (0.0, 0.0) else 3):
            lever = R.from_euler("z", turn).as_matrix() @ self._base_offset
            q = self._clik(False, self._to_urdf(local, lever), q)
            settled = abs(float(q[0]) - turn) < 1e-6
            turn = float(q[0])
            if settled:
                break
        return turn, np.asarray(q[1:], dtype=float)

    def _solve_driving(
        self, base: np.ndarray, target_pose: np.ndarray, arm: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """The library's planar solve from `base`: the new base pose, and the arm joints.

        target = T(d) Rz(turn) T(base_offset) F(q) T(tool_offset) on the MuJoCo
        side, and the library solves T(d') Rz(turn) F(q) -- so handing it the
        target with both offsets taken out gives d' = d + Rz(turn) base_offset -
        base_offset, which is undone here in closed form.

        The planar model is not clipped inside the solve -- not the base, and not
        the lift, arm or wrist either -- so a solution past any limit is clipped
        to it afterwards and the arm solved again, with limits, from that base,
        rather than left reaching for a pose it is not going to get.
        """
        local = np.linalg.inv(self._base_pose(base)) @ target_pose
        seed = np.concatenate([[0.0, 0.0, 1.0, 0.0], arm])
        q = self._clik(True, self._to_urdf(local, self._base_offset), seed)
        turn = math.atan2(q[3], q[2])
        shift = (
            q[:2]
            + self._base_offset[:2]
            - R.from_euler("z", turn).as_matrix()[:2, :2] @ self._base_offset[:2]
        )
        xy = base[:2] + R.from_euler("z", base[2]).as_matrix()[:2, :2] @ shift
        theta = base[2] + turn
        clipped = np.clip(
            [xy[0], xy[1], theta], self._home - self._base_leash, self._home + self._base_leash
        )
        arm = np.clip(q[4:], self._arm_limits[:, 0], self._arm_limits[:, 1])
        if np.allclose(clipped, [xy[0], xy[1], theta]) and np.allclose(arm, q[4:]):
            return clipped, arm
        _, arm = self._solve_turning(clipped, target_pose, arm, (0.0, 0.0))
        return clipped, arm

    def solve(self, target_pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Joint targets for `target_pose` (world frame), and the residual 6-vector.

        Seeded from the robot's current configuration, and solved in the frame of the base where it stands now -- the library's
        base joint starts at zero, so its answer for the base is a move relative
        to here.
        """
        target_pose = np.asarray(target_pose, dtype=float)
        self._scratch_data.qpos[:] = self._live_data.qpos
        base = np.asarray(self._live_view.get_move_group("base").joint_pos, dtype=float)
        arm = np.clip(
            np.concatenate(
                [
                    np.asarray(self._live_view.get_move_group(g).joint_pos, dtype=float)
                    for g in self.ARM_GROUPS
                ]
            ),
            self._arm_limits[:, 0],
            self._arm_limits[:, 1],
        )

        turn, turned_arm = self._solve_turning(
            base, target_pose, arm, self._rotation_limits(base[2])
        )
        joint_pos = np.concatenate([[base[0], base[1], base[2] + turn], turned_arm])
        if not self._include_base:
            joint_pos = joint_pos[3:]
        error = self._residual(joint_pos, target_pose)
        if not self._base_translation:
            return joint_pos, error

        # Driving only when it gets the tool closer than turning in place does.
        # The library's planar solve does not weigh base motion against arm
        # motion, and it solves without the arm's limits, so on a target the arm
        # cannot reach it happily walks the base somewhere that only helps a lift
        # that does not exist. The comparison is what keeps the base still while
        # the arm can do the job.
        driven = np.concatenate(self._solve_driving(base, target_pose, arm))
        driven_error = self._residual(driven, target_pose)
        if np.linalg.norm(driven_error[:3]) + self.DRIVE_GAIN_M < np.linalg.norm(error[:3]):
            return driven, driven_error
        return joint_pos, error

    def _residual(self, joint_pos: np.ndarray, target_pose: np.ndarray) -> np.ndarray:
        """`joint_pos`'s miss on `target_pose`, measured on the MuJoCo model itself."""
        self._write(self._scratch_view, joint_pos)
        mujoco.mj_kinematics(self._live_data.model, self._scratch_data)
        reached = self._scratch_view.get_move_group("wrist").leaf_frame_to_world
        return _pose_error(reached, target_pose)


def make_stretch_arm_ik(
    stretch_view: Stretch4RobotView,
    namespace: str,
    include_base: bool,
    ik_choice: IKChoice,
) -> Stretch4KinematicsArmIK:
    """The IK, with the base in it or not as `include_base` says."""
    return Stretch4KinematicsArmIK(
        stretch_view,
        namespace,
        include_base=include_base,
        base_translation=ik_choice.base_translation,
    )


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
        fraction = (finger - self._view.finger_closed) / (
            self._view.finger_open - self._view.finger_closed
        )
        fraction = float(np.clip(fraction, 0.0, 1.0))
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
      back; see `Stretch4KinematicsArmIK`.
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
        match_robotiq_aperture: bool = True,
        pose_conventions: PoseConventions | None = None,
        robotiq_aperture_m: float | None = None,
        ik_choice: IKChoice | None = None,
    ) -> None:
        self.stretch_view = stretch_view
        self.namespace = namespace
        self.target_z_offset = float(target_z_offset)

        self.franka = VirtualFranka()
        # Rewriting `init_qpos` rather than carrying the adjusted pose alongside
        # it, because "where the Franka starts" is read from there by everything
        # that needs it -- `snap_to_franka_joint_pos`, the IK seeds, `reset` --
        # and a second source of truth would have them disagree about which pose
        # an episode began at. See `stretch_startable_arm_qpos`.
        if pose_conventions is None:
            pose_conventions = pose_conventions_requested()
        self.pose_conventions = pose_conventions
        if self.pose_conventions.changes_franka_start_pose:
            self.franka.init_qpos = franka_start_arm_qpos(
                self.franka, self.pose_conventions, float(franka_mount_pose[2, 3])
            )
        if self.pose_conventions:
            log.info(f"[retarget] pose conventions: {self.pose_conventions.describe()}")
        # Read off the environment when not passed, like the pose conventions:
        # the evaluation's workers build this view with no way to be told.
        self.ik_choice = ik_choice_requested() if ik_choice is None else ik_choice
        log.info(f"[retarget] IK: {self.ik_choice.describe()}")
        self.arm_ik = make_stretch_arm_ik(stretch_view, namespace, include_base, self.ik_choice)

        # What "fully open" and "shut" mean on Stretch, in finger-joint units --
        # radians on the SG4, metres on the PG4 -- read off the gripper group so
        # that they are the tool this model carries. Open is narrowed to the
        # position at which its jaw is as wide as the Robotiq's so that a gripper
        # command means the same aperture on both robots; see
        # `ROBOTIQ_MAX_APERTURE_M` for why, and for what it costs. Solved on the
        # IK's scratch data, which is what keeps the measurement from moving the
        # robot that is about to be commanded.
        gripper = stretch_view.get_move_group("gripper")
        self.gripper_kind = gripper.kind
        self.finger_closed = float(gripper.CLOSED_JOINT_POS)
        self.finger_open = float(gripper.OPEN_JOINT_POS)
        self.robotiq_aperture_m = float(
            ROBOTIQ_MAX_APERTURE_M if robotiq_aperture_m is None else robotiq_aperture_m
        )
        """What "open" means on this hand, in metres between the pads. See `ROBOTIQ_MAX_APERTURE_M`.

        A parameter rather than the constant outright because it is the one
        number in the retargeting whose right value is a property of the *task*
        as much as of the two grippers: matched to the Robotiq exactly, Stretch
        cannot get round anything the Robotiq could only just swallow, and it
        approaches every object with less clearance for the aiming error the
        retargeting still has. Left `None` it is the constant, so nothing that
        does not ask for it sees a change.
        """
        if match_robotiq_aperture:
            self.finger_open = stretch_finger_for_aperture(
                self.arm_ik._scratch_view.get_move_group("gripper"),
                self.arm_ik._scratch_data.model,
                self.arm_ik._scratch_data,
                self.robotiq_aperture_m,
            )

        # The bare axis correction, under every convention.
        self._tool_correction = np.eye(4)
        self._tool_correction[:3, :3] = FRANKA_TO_STRETCH_TOOL
        self._tool_correction_inverse = np.linalg.inv(self._tool_correction)

        self._move_groups = {"arm": _ProxyArmGroup(self), "gripper": _ProxyGripperGroup(self)}
        self._franka_seed = self.franka.init_qpos.copy()
        self.last_arm_ctrl = self.franka.init_qpos.copy()
        self.last_gripper_ctrl = np.array([ROBOTIQ_CTRL_RANGE[0]])
        self.last_residual = np.zeros(6)
        self.unreachable_steps = 0
        """Steps this episode whose target was out of reach with the lift saturated."""
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
        # Read off the fingers rather than assumed to be open. The arm half of
        # this method has always reported where Stretch actually is; the gripper
        # half used to hardcode `ROBOTIQ_CTRL_RANGE[0]`, which is a claim that the
        # hand is open and is wrong on a robot whose `init_qpos` shuts it. That
        # made `ctrl` and `joint_pos` disagree about the same gripper -- and with
        # `snap_to_franka_home` off, nothing else would have corrected it.
        fingers = self.stretch_view.get_move_group("gripper").joint_pos
        self.last_gripper_ctrl = np.array(
            [robotiq_ctrl_from_stretch_fingers(fingers, self.finger_open, self.finger_closed)]
        )
        self.last_residual = np.zeros(6)
        self.unreachable_steps = 0

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
        pitch = float(self.pose_conventions.change_stretch_start_pose_pitch_deg)
        if pitch:
            # About Stretch's tool y -- the jaw line -- so the approach tips up
            # or down and the hand does not roll. `target` is already in
            # Stretch's tool convention here (`franka_tool_pose_to_world` has
            # applied the correction), so this composes on the right.
            #
            # Here and nowhere else: `retarget_franka_joint_pos` is left alone,
            # so this moves where the episode *begins* and not the frame the
            # policy's actions are interpreted in. A rotation folded into
            # `_tool_correction` instead would be undone again by
            # `_tool_correction_inverse` on the way back out and the policy would
            # never see it -- see
            # `PoseConventions.change_stretch_start_pose_pitch_deg`.
            spin = np.eye(4)
            spin[:3, :3] = R.from_euler("y", pitch, degrees=True).as_matrix()
            target = target @ spin
            log.info(
                f"[retarget] snapping with the wrist pitched {pitch:+.1f} deg about the jaw line"
            )
        solution, residual = self.arm_ik.solve(target)
        for group, value in self.arm_ik.split(solution).items():
            move_group = self.stretch_view.get_move_group(group)
            move_group.joint_pos = value
            move_group.ctrl = value

        # The hand as well as the arm. Stretch's `init_qpos` starts its fingers
        # shut and the DROID Franka's starts open, so without this the policy's
        # first observation reports a closed gripper -- 0.824 against the 0.003 a
        # Franka episode reports, the opposite end of the range it reads that
        # channel in -- on a checkpoint whose training episodes all begin with an
        # open hand. It is also the state `reset()` already claims in
        # `last_gripper_ctrl`, so leaving the fingers shut made the proxy
        # contradict itself about its own gripper.
        gripper_ctrl = robotiq_ctrl_from_driver(self.franka.init_gripper_qpos)
        gripper = self.stretch_view.get_move_group("gripper")
        gripper.joint_pos = self.retarget_robotiq_ctrl(gripper_ctrl)
        gripper.ctrl = gripper.joint_pos

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
        targets = self.arm_ik.split(solution)
        self._warn_if_lift_saturated(targets, self.last_residual)
        return targets

    def _warn_if_lift_saturated(self, targets: dict, residual: np.ndarray) -> None:
        """Say so, out loud, when the lift has run out of travel and the tool is short.

        This is the failure that otherwise looks like nothing. The IK reports a
        residual, the action is applied, the episode continues, and the only
        symptom is a gripper that closes a few centimetres above the object --
        which gets read as the policy aiming badly. `target_z_offset` makes it
        more likely by design, since it raises every target, and
        `FrankaOnStretchView`'s own docstring notes that where the lift is already
        saturated the offset buys nothing. That note is worth a log line when it
        actually happens.

        Only when both are true: the lift is against a travel limit, *and* the
        tool is at least `UNREACHABLE_WARNING_M` short. A saturated lift on a
        target the arm reaches anyway is not a problem, and a large residual with
        travel left is a different problem -- the message would be wrong about
        the cause.

        Warned once per episode and counted thereafter. A rollout is ~300 steps
        at 15Hz and this condition persists for as long as the policy keeps
        asking, so warning every step would bury the run's own output; the count
        goes into `get_info` via `unreachable_steps`.
        """
        vertical = float(residual[2])
        if vertical < UNREACHABLE_WARNING_M:
            return
        lift = np.asarray(targets.get("lift", []), dtype=float).reshape(-1)
        if not lift.size:
            return
        lift_target = float(lift[0])
        low, high = commandable_limits(self.stretch_view.get_move_group("lift"))[0]
        at_limit = (high - lift_target) < LIFT_SATURATION_MARGIN_M or (
            lift_target - low
        ) < LIFT_SATURATION_MARGIN_M
        if not at_limit:
            return

        self.unreachable_steps += 1
        message = (
            f"[retarget] lift is at its {'top' if lift_target > low else 'bottom'} "
            f"({lift_target:.4f}m of {low:.3f}..{high:.3f}) and the commanded tool pose is "
            f"still {vertical * 1000:.0f}mm away vertically "
            f"({self.last_position_error * 1000:.0f}mm in total). Stretch cannot reach where "
            f"the policy is pointing"
        )
        if self.target_z_offset:
            message += (
                f", and target_z_offset is adding {self.target_z_offset * 100:.1f}cm of that "
                f"-- at a saturated lift the offset buys no clearance and only grows the miss"
            )
        if self.unreachable_steps == 1:
            log.warning(f"{message}. Further occurrences this episode are counted, not logged.")
        else:
            log.debug(message)

    def retarget_robotiq_ctrl(self, value) -> np.ndarray:
        """A Robotiq 0-255 command -> Stretch's two finger targets, in the tool's joint units."""
        command = float(np.asarray(value, dtype=float).reshape(-1)[0])
        self.last_gripper_ctrl = np.array([command])
        fraction = np.clip(
            (command - ROBOTIQ_CTRL_RANGE[0]) / (ROBOTIQ_CTRL_RANGE[1] - ROBOTIQ_CTRL_RANGE[0]),
            0.0,
            1.0,
        )
        # 0 is open on the Robotiq and closed on Stretch, hence the flip. `finger_open`
        # rather than `STRETCH_FINGER_OPEN` so that "open" means the Robotiq's
        # aperture; see `ROBOTIQ_MAX_APERTURE_M`.
        finger = self.finger_open + fraction * (self.finger_closed - self.finger_open)
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

REFERENCE_TARGET_COLOR = (0.25, 0.65, 1.00, 1.0)
"""
The pose Stretch was *asked* for, when that is not the Franka's own pose.

A third colour because with `target_z_offset` in force there are three points
worth telling apart and two colours cannot do it: where the Franka's tool is
(green), where Stretch was commanded to put its own -- that pose raised by the
offset (this blue) -- and where Stretch got to (orange). At zero offset the first
two are the same point and only two colours appear.
"""

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
    axes: bool = True,
) -> None:
    """Draw a 4x4 pose into a rendered scene: a ball at the origin, arrows on the axes.

    The ball is the tool centre, and what to compare between two images; the
    arrows are what tell you two frames are also oriented differently -- the
    Robotiq reaches along its tool +z and Stretch's gripper along its tool +x,
    which is the whole job of `FRANKA_TO_STRETCH_TOOL`, and is much easier to
    believe on sight than from a rotation matrix.

    Axes are coloured x/y/z as red/green/blue as usual; `color` is the ball, and
    identifies which frame it is.

    `axes=False` draws the ball alone, for a marker that is in the picture as a
    *position* rather than as a frame. A reference point from another robot is the
    case that wants it: its orientation is in a different tool convention, so
    three more arrows beside the ones that matter are clutter that invites the
    wrong comparison.
    """
    origin = pose[:3, 3]
    _add_decor_geom(
        scene, mujoco.mjtGeom.mjGEOM_SPHERE, [ball_radius] * 3, origin, np.eye(3), color, label
    )
    if not axes:
        return
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
