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
      -> JAW_FLIP (maybe)           -> the grasp-equivalent branch the wrist can hold
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

import logging
import math
import os
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np
from mujoco import MjData, MjModel, MjSpec
from scipy.spatial.transform import Rotation as R

from examples.machine_learning.molmospaces.stretch.robot_view import (
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

# A half turn about Stretch's tool +x -- its approach axis. A parallel jaw grasps
# identically either way round: the rotation maps one finger onto where the other
# was and leaves the jaw line and the approach direction unchanged, so both poses
# close on the same object the same way. Only the labels "left finger" and "right
# finger" swap, and nothing downstream depends on which is which.
#
# It is worth a second solve because Stretch's wrist roll has an asymmetric range
# (`wrist_roll_joint`, about [-4.28, +1.14] rad): a commanded twist can pin both
# `wrist_yaw` and `wrist_roll` against their upper limits with the orientation
# still a quarter turn out, while the same grasp expressed half a turn round sits
# in open range. Measured, on the waypoints in
# `retargetting/tests/test_retargeting.py`: one of fifteen stalls at 0.43 rad of
# orientation error that no number of iterations recovers (80, 240, 800 and 3000
# all settle at 0.433), and the flipped branch reaches it to 0.001 rad with no
# change in position error. See `FrankaOnStretchView._solve_either_jaw`.
JAW_FLIP = np.eye(4)
JAW_FLIP[:3, :3] = R.from_euler("x", 180, degrees=True).as_matrix()

ROBOTIQ_ROLL_180 = np.eye(4)
ROBOTIQ_ROLL_180[:3, :3] = R.from_euler("z", 180, degrees=True).as_matrix()
"""
A half turn about the *Robotiq's* approach axis -- the Franka-convention twin of `JAW_FLIP`.

`JAW_FLIP` is the same physical rotation written in Stretch's tool frame, where
the approach is +x; the Robotiq reaches along +z, so here it is a turn about z.
Both leave a parallel jaw grasping the identical object the identical way (see
`JAW_FLIP`) -- what changes is which way round the hand, and therefore the wrist
camera bolted to it, is facing.

That is the whole point of `--change_franka_start_pose_flip_wrist`: the grasp is unaffected
and the picture is not.
"""

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

JAW_MODES = ("auto", "flipped", "upright")
"""
How `FrankaOnStretchView` chooses which way round to hold the jaw.

* `"auto"` -- the default. Try the branch already in use and switch only on a
  clear orientation win. Two IK solves per step, and the only mode that gives up
  nothing: measured across `tests/test_retargeting.py`'s waypoints it holds every
  one to 0.0006 rad.
* `"flipped"` -- always the half-turned branch. One solve per step, no branch
  that can change mid-reach, and *better position everywhere* -- worst 0.39mm
  against auto's 1.87mm across those waypoints, with 0.0000 rad of orientation at
  fourteen of sixteen. The catch is the other two: at large tool yaws (`yaw_in`
  at +45 degrees, `yaw_out` at -30) the flipped branch runs the wrist into a roll
  limit and settles **0.43 rad** short.
* `"upright"` -- never flip. The behaviour before jaw symmetry existed, kept so a
  result can be compared against runs that predate it. Measures the same as
  "auto" on those waypoints, because solved fresh from the snap the upright
  branch suffices; the two diverge along a continuous path.

The 0.43 rad is why "flipped" is not the default despite winning on position. It
is not a symmetry to be forgiven: a jaw rotated that far about its approach axis
closes along a different line, so it would take a knife across rather than along
-- and `test_both_grippers_end_up_pointing_the_same_way`, which reads the finger
bodies rather than any tool frame, fails at those two waypoints under "flipped".
The tests are left tight on purpose, so selecting a worse mode makes the suite
name the poses it costs.

All three grasp the same object the same way where they agree; see `JAW_FLIP`.
What differs is which poses Stretch's wrist can hold, and that is not a strict
ordering -- a fixed branch that is right for most of the workspace is wrong at
its edges.
"""

JAW_FLIP_GAIN_RAD = 0.10
"""
How much orientation the other jaw branch has to win before the wrist rolls over
to it.

This is hysteresis, not a tolerance. Both branches are legitimate solutions, so
without a deadband the choice would follow whichever residual happened to be
smaller and the wrist would spin half a turn between steps that command almost
the same pose -- real motion, at the moment a grasp is closing. Requiring a clear
win against the branch already in use means the arm changes its mind only when
there is something to gain: over the 180 commands of the test's continuous path,
twice, both times mid-reorientation rather than mid-grasp.
"""

JAW_FLIP_POSITION_SLACK_M = 0.001
"""
How much position the other jaw branch may cost while still being preferred.

Position outranks orientation everywhere else in this module
(`_task_priority_step` solves for it outright and gives orientation the leftovers),
so it would be incoherent for a jaw flip to buy an angle at the price of a
millimetre of reach. The slack is there only so that two solves which reach the
same point to within solver noise are judged on their orientation rather than on
which one happened to round down.
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

# Stretch finger joint angle, per `gripper_finger_{right,left}_joint`: 0 closed,
# 0.5 rad fully open.
STRETCH_FINGER_OPEN = 0.5
STRETCH_FINGER_CLOSED = 0.0

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
    """
    low, high = STRETCH_FINGER_CLOSED, STRETCH_FINGER_OPEN

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


STRETCH_SPAWN_BASE_OFFSET_XY = (-0.3598, 0.0874)
"""
How far to stand Stretch back so its spawn gripper pose is the Franka's, in its own axes.

Measured, with both robots at the mini benchmark's spawn: the Franka's grasp
site sits 0.3069m forward and 0.0m across of the base it is mounted on, and
Stretch's grasp centre at `STRETCH_SPAWN_ARM_M` sits 0.6667m forward and 0.0874m
across of its own. The difference is this. Heights come out within 7mm on their
own (1.1782m against 1.1853m), which the base could not have fixed anyway.

**It is expensive, which is why `match_stretch_spawn_pose_to_franka` is off by
default.** Two costs, both measured:

* **Reach.** Stretch's grasp centre reaches 0.9867m from the base with the arm
  at its 0.52m stop. The benchmark's objects sit 0.6220m away; after this
  retreat they sit 0.9818m away, which is 5mm inside the hard limit. Every grasp
  then depends on the base driving back in, and the base accelerates at
  0.25 m/s^2.
* **The camera.** `stretch_baseline` hangs the exo camera off `base_link`, so the
  camera retreats with the robot -- and the whole premise of that setup is a
  camera at the same place in the room as the Franka's. 0.36m is not a
  refinement of that comparison, it is the end of it.

The retreat is cancelled in the virtual Franka's mount (see
`franka_mount_pose_from_base`'s `offset_xy`), so the frame a policy action means
is unchanged: what moves is the robot, not the retargeting.

The cheaper version of the same idea, if the point is only that the two grippers
start together: spawn with the arm stowed at 0 instead, where the retreat is
0.16m and the objects land 0.782m away with 205mm of reach to spare.
"""


def pose_matrix(pos, quat_wxyz) -> np.ndarray:
    """A 4x4 homogeneous transform from a position and a (w, x, y, z) quaternion."""
    pose = np.eye(4)
    pose[:3, :3] = R.from_quat(np.asarray(quat_wxyz, dtype=float), scalar_first=True).as_matrix()
    pose[:3, 3] = np.asarray(pos, dtype=float)
    return pose


POSE_CONVENTION_ENV_VARS = {
    "change_franka_start_pose_flip_wrist": "STRETCH4_CHANGE_FRANKA_START_POSE_FLIP_WRIST",
    "change_franka_start_pose_limit_height": "STRETCH4_CHANGE_FRANKA_START_POSE_LIMIT_HEIGHT",
    "change_stretch_start_pose_flip_wrist": "STRETCH4_CHANGE_STRETCH_START_POSE_FLIP_WRIST",
    "map_franka_wrist_to_flipped_stretch4_wrist": "STRETCH4_MAP_FRANKA_WRIST_TO_FLIPPED_STRETCH4_WRIST",
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
    grasp *is*, only which way round a wrist is held and where an episode begins.
    That is exactly why they need naming and publishing rather than hard-coding --
    a comparison whose two halves disagree about a convention is not a comparison.
    """

    change_franka_start_pose_flip_wrist: bool = False
    """Roll the Franka's start pose half a turn about the Robotiq's approach axis.

    The grasp is unaffected -- see `JAW_FLIP` -- and what swings round is the
    hand, and the wrist camera bolted off to one side of it. Applied to the real
    Franka's episode `init_qpos` and to the virtual one the retargeting snaps
    Stretch to, from the same flag, so the two cannot disagree.
    """

    change_franka_start_pose_limit_height: bool = False
    """Cap the Franka's start tool height at `STRETCH_MAX_GRASP_HEIGHT_M`.

    Without it the Stretch condition begins every episode already saturated: the
    Franka's home puts its grasp site 10.3cm above the highest Stretch's lift can
    put its own with the tool pointing down, so the arm sits at its stop reaching
    for a pose it cannot hold and reports proprioception from a configuration it
    never reached.
    """

    change_stretch_start_pose_flip_wrist: bool = False
    """Spawn Stretch with its own wrist rolled half a turn.

    The Stretch-side counterpart of `change_franka_start_pose_flip_wrist`, and it
    moves Stretch's wrist cameras to the other side of the hand in the same way.
    Note what it does *not* survive: with `snap_to_franka_home` on -- the default
    -- the first `get_action` writes the arm and wrist to whatever matches the
    Franka's start tool pose, so this decides the spawn and the first observation
    and is then overwritten. Turn the snap off to hold it for the episode.
    """

    map_franka_wrist_to_flipped_stretch4_wrist: bool = False
    """Retarget every pose onto the half-turned branch of Stretch's wrist.

    A half turn about the approach axis folded into the tool transform itself
    (`JAW_FLIP`), rather than chosen per step the way `jaw_mode` chooses it. The
    two are not the same thing. `jaw_mode="flipped"` holds the flipped branch and
    then *reports the pose back unflipped*, so the policy never sees it; this
    changes the frame the retargeting is defined in, so both directions carry the
    turn and it stays self-consistent -- and Stretch's wrist, with the cameras on
    it, ends up the other way round for good.

    The grasp is identical either way (see `JAW_FLIP`). The reason to want it is
    the wrist camera: pair it with `change_franka_start_pose_flip_wrist` to put
    both robots' wrist cameras on the same side of their respective hands.
    """

    match_stretch_spawn_pose_to_franka: bool = False
    """Stand Stretch back far enough that its spawn gripper pose is the Franka's.

    Stretch's arm reaches *further* at its shortest than the Franka's does at its
    home -- 0.467m against 0.307m from the base, before the spawn extension in
    `setups.STRETCH_SPAWN_ARM_M` adds its own -- so the only way to make the two
    grippers start in the same place is to stand Stretch further back. See
    `setups.STRETCH_SPAWN_BASE_OFFSET_XY`, which measures how far, and what it
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
        return self.change_franka_start_pose_flip_wrist or self.change_franka_start_pose_limit_height

    def describe(self) -> str:
        asked = [name for name in POSE_CONVENTION_ENV_VARS if getattr(self, name)]
        return ", ".join(asked) if asked else "none (DROID home, unflipped)"


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() not in ("", "0", "false", "no")


def pose_conventions_requested() -> PoseConventions:
    """Which pose conventions this process was asked for."""
    return PoseConventions(
        **{field: _env_flag(variable) for field, variable in POSE_CONVENTION_ENV_VARS.items()}
    )


def publish_pose_conventions(conventions: PoseConventions) -> None:
    """Put the pose conventions in the environment, for this process and its workers.

    Every variable is written in both directions: a run that does *not* ask for a
    convention must clear a variable an earlier run in the same shell exported,
    or the two halves of a comparison silently start at different poses -- which
    is precisely the failure these flags were added to remove.
    """
    for field, variable in POSE_CONVENTION_ENV_VARS.items():
        if getattr(conventions, field):
            os.environ[variable] = "1"
        else:
            os.environ.pop(variable, None)


def stretch_startable_arm_qpos(
    franka: "VirtualFranka",
    mount_height_m: float = FRANKA_PEDESTAL_HEIGHT,
    *,
    roll_180: bool = True,
    max_height_m: float | None = STRETCH_MAX_GRASP_HEIGHT_M,
    iterations: int = 300,
) -> np.ndarray:
    """The Franka's home arm configuration, adjusted so Stretch can start there too.

    Two changes to the home tool pose, independent of each other and each asked
    for by its own flag (`StartPoseChanges`), both of which exist because the two
    robots do not start an episode in the same place and the study needs them to.
    `roll_180=False` skips the first, `max_height_m=None` skips the second, and
    with both off this returns `franka.init_qpos` unchanged -- which is what
    makes "neither flag was passed" cost nothing rather than round-trip the home
    pose through an IK solve.

    * **The wrist rolled** as near half a turn as `fr3_joint7` goes -- see
      `roll_franka_wrist`, which turns that one joint and leaves the grasp site
      exactly where it was. A parallel jaw grasps identically either way round, so
      this costs the grasp nothing; what swings round is the hand, and the wrist
      camera bolted off to one side of it.

      Worth knowing what it does to that camera, since it is the reason to want
      it. `gripper/wrist_camera` -- the one the DROID checkpoint reads -- sits
      7.4cm behind the grasp site at the Franka's home, with a view direction of
      x=+0.333 in the arm's base frame. After the roll it is 7.0cm the other side,
      at x=-0.332. That mirroring is not a choice this function makes: with the
      approach pointing straight down, a half turn about it takes anything offset
      to one side over to the other, and the camera is offset. Render the camera
      and look before concluding it is the view you wanted.
    * **Lowered to `max_height_m`**, Stretch's own ceiling. The Franka's home puts
      its grasp site at 1.185m and Stretch's lift tops out 10.3cm below that
      (`STRETCH_MAX_GRASP_HEIGHT_M`), so unretargeted the Stretch condition begins
      every episode already saturated, reaching for a pose it cannot hold and
      reporting proprioception from a configuration it never actually reached.
      Capping the Franka's start is what makes "both robots start in the same
      place" true rather than aspirational.

    The clamp is applied to the tool's z *in the arm's own base frame*, offset by
    `mount_height_m`, which is the same number as world z only because every mount
    this module builds is level -- `franka_mount_pose_from_base` yaws and nothing
    else. Asserted there rather than re-derived here.

    Returns seven joint angles, IK'd from the home configuration so the result is
    the nearest way to hold that pose rather than an unrelated branch. The height
    is a *cap*, not a move: a home pose already below the ceiling is left at its
    own height and only rolled.
    """
    if not roll_180 and max_height_m is None:
        return np.asarray(franka.init_qpos, dtype=float).copy()
    pose = franka.fk(franka.init_qpos)
    if max_height_m is not None:
        pose[2, 3] = min(float(pose[2, 3]), float(max_height_m) - float(mount_height_m))
    qpos = franka.ik(pose, franka.init_qpos, iterations=iterations)
    if roll_180:
        # The wrist joint first, then a solve for the exact half turn seeded from
        # it. `fr3_joint7` gets 172.8 of the 180 degrees on its own and the solve
        # finds the remaining 7.2 in the joints behind it -- seeded from the
        # rolled wrist, so it converges on the configuration *next to* this one
        # rather than on some other arm shape that also happens to hold the pose.
        # See `roll_franka_wrist` for what the seed is worth.
        qpos = franka.ik(
            pose @ ROBOTIQ_ROLL_180,
            roll_franka_wrist(franka, qpos),
            iterations=iterations,
        )
    return qpos


def franka_start_arm_qpos(
    franka: "VirtualFranka",
    conventions: PoseConventions,
    mount_height_m: float = FRANKA_PEDESTAL_HEIGHT,
) -> np.ndarray:
    """`stretch_startable_arm_qpos` driven by a `PoseConventions`.

    The one place the two Franka start-pose flags are turned into the two
    arguments, so the Franka half of the change (`setups.franka_episode_override`)
    and the Stretch half (`FrankaOnStretchView`) cannot disagree about what a
    flag means.
    """
    return stretch_startable_arm_qpos(
        franka,
        mount_height_m,
        roll_180=conventions.change_franka_start_pose_flip_wrist,
        max_height_m=(
            STRETCH_MAX_GRASP_HEIGHT_M
            if conventions.change_franka_start_pose_limit_height
            else None
        ),
    )


def roll_franka_wrist(franka: "VirtualFranka", joint_pos: np.ndarray) -> np.ndarray:
    """`joint_pos` with the last wrist joint turned as close to half a turn as it goes.

    `fr3_joint7` *is* the roll about the Robotiq's approach axis, so turning it is
    the whole operation: the grasp site sits on that axis and does not move, the
    other six joints keep their angles, and what swings round is the hand -- the
    jaw line, and the wrist camera bolted off to one side of it.

    Done by turning the joint rather than by IK'ing `pose @ ROBOTIQ_ROLL_180`,
    which is what this used to do and is not the same operation at all: a fresh
    solve is free to reach the rolled pose from any configuration, and from the
    Franka's home it picked one with *every* joint moved --
    `[-0.81, -1.02, 0.60, -2.60, 0.50, 1.66, 2.63]` against a home of
    `[0, -0.79, 0, -2.36, 0, 1.57, 0]`. That is re-posing the arm, not rolling the
    wrist, and it moved the elbow and shoulder into a configuration whose
    consequences (a different jaw branch at large tool yaws) had nothing to do
    with the roll that was asked for.

    **The joint cannot quite manage a half turn on its own.** `fr3_joint7` runs to
    +-3.0159 rad and home is 0, so it stops 0.126 rad -- 7.2 degrees -- short
    either way. That is a property of the arm, not a choice here. Which is why
    this is a *seed* rather than the answer: `stretch_startable_arm_qpos` turns
    this joint as far as it goes and then solves for the exact half turn starting
    from here, so the last 7.2 degrees come out of the joints behind the wrist and
    the solution stays the one next to this configuration.

    The direction is whichever has more headroom, which from a home of 0 is a
    tie broken towards positive.
    """
    qpos = np.asarray(joint_pos, dtype=float).copy()
    low, high = franka.joint_limits[6]
    forward, backward = qpos[6] + math.pi, qpos[6] - math.pi
    # Whichever half turn the joint can follow furthest before its limit stops it.
    if min(high, forward) - qpos[6] >= qpos[6] - max(low, backward):
        qpos[6] = min(high, forward)
    else:
        qpos[6] = max(low, backward)
    return qpos


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
    of `STRETCH_SPAWN_BASE_OFFSET_XY`, which is how that convention moves the
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
            offset_xy = (
                offset_xy[0] - STRETCH_SPAWN_BASE_OFFSET_XY[0],
                offset_xy[1] - STRETCH_SPAWN_BASE_OFFSET_XY[1],
            )
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
    finger_angles, finger_open: float = STRETCH_FINGER_OPEN
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
    """
    finger = float(np.mean(np.asarray(finger_angles, dtype=float)))
    open_fraction = np.clip(
        (finger - STRETCH_FINGER_CLOSED) / (finger_open - STRETCH_FINGER_CLOSED), 0.0, 1.0
    )
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
        fraction = (finger - STRETCH_FINGER_CLOSED) / (
            self._view.finger_open - STRETCH_FINGER_CLOSED
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
      back; see `StretchArmIK`. What is left of the orientation shortfall is
      partly bought back by holding the jaw half a turn over where that is the
      branch the wrist can reach, which grasps the same object the same way;
      see `JAW_FLIP` and `_solve_either_jaw`. `jaw_flipped` says which branch the
      arm is currently in, and `franka_joint_pos` reports the arm state back in
      the orientation the policy commanded either way.
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
        jaw_mode: str = "auto",
        pose_conventions: PoseConventions | None = None,
        robotiq_aperture_m: float | None = None,
    ) -> None:
        if jaw_mode not in JAW_MODES:
            raise ValueError(f"jaw_mode must be one of {JAW_MODES}, not {jaw_mode!r}")
        self.jaw_mode = jaw_mode
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
        self.arm_ik = StretchArmIK(stretch_view, namespace, include_base=include_base)

        # What "fully open" means on Stretch, in finger-joint radians. Narrowed to
        # the angle at which its jaw is as wide as the Robotiq's so that a gripper
        # command means the same aperture on both robots; see
        # `ROBOTIQ_MAX_APERTURE_M` for why, and for what it costs. Solved on the
        # IK's scratch data, which is what keeps the measurement from moving the
        # robot that is about to be commanded.
        self.finger_open = STRETCH_FINGER_OPEN
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

        self._tool_correction = np.eye(4)
        self._tool_correction[:3, :3] = FRANKA_TO_STRETCH_TOOL
        if self.pose_conventions.map_franka_wrist_to_flipped_stretch4_wrist:
            # Folded into the transform rather than chosen per step: `JAW_FLIP` is
            # in Stretch's tool convention, so it composes on the right, after the
            # axis correction has decided which way the approach points. Both
            # directions then carry it, because `_tool_correction_inverse` is
            # taken from this matrix -- which is what keeps the pose the policy
            # reads back the pose it asked for. See
            # `PoseConventions.map_franka_wrist_to_flipped_stretch4_wrist`.
            self._tool_correction = self._tool_correction @ JAW_FLIP
        self._tool_correction_inverse = np.linalg.inv(self._tool_correction)

        self._move_groups = {"arm": _ProxyArmGroup(self), "gripper": _ProxyGripperGroup(self)}
        self._franka_seed = self.franka.init_qpos.copy()
        self.last_arm_ctrl = self.franka.init_qpos.copy()
        self.last_gripper_ctrl = np.array([ROBOTIQ_CTRL_RANGE[0]])
        self.last_residual = np.zeros(6)
        self.jaw_flipped = jaw_mode == "flipped"
        """Whether the arm is currently holding the half-turned jaw. See `JAW_FLIP`."""
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
            [robotiq_ctrl_from_stretch_fingers(fingers, self.finger_open)]
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
        # Through the same branch selection as a commanded step, so that the
        # configuration the robot is written into and `jaw_flipped` cannot
        # disagree -- the reported arm state is read back through that flag, and
        # a snap that picked one branch while the flag said the other would
        # report an arm half a turn from the one Stretch is holding.
        self.jaw_flipped = self.jaw_mode == "flipped"
        solution, residual, self.jaw_flipped = self._solve_either_jaw(target)
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

    def _solve_either_jaw(self, target_pose: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
        """Solve for `target_pose`, allowing the gripper to be held half a turn over.

        Returns the joint targets, the residual, and which branch they are. Both
        branches grasp the commanded object identically (see `JAW_FLIP`), so this
        is free reach rather than a compromise: the arm is allowed to pick the
        one its wrist can actually hold.

        Under `jaw_mode` "flipped" or "upright" the branch is fixed and this is a
        single solve; only "auto" pays for two and chooses. See `JAW_MODES`.

        In "auto", the branch in use is tried first and kept unless the other one
        clearly wins, which is what stops the wrist rolling over on solver noise;
        see `JAW_FLIP_GAIN_RAD`.
        """
        if self.jaw_mode != "auto":
            fixed = self.jaw_mode == "flipped"
            solution, residual = self.arm_ik.solve(
                target_pose @ JAW_FLIP if fixed else target_pose
            )
            return solution, residual, fixed

        current = target_pose @ JAW_FLIP if self.jaw_flipped else target_pose
        current_solution, current_residual = self.arm_ik.solve(current)

        other = target_pose if self.jaw_flipped else target_pose @ JAW_FLIP
        other_solution, other_residual = self.arm_ik.solve(other)

        def split(residual):
            return float(np.linalg.norm(residual[:3])), float(np.linalg.norm(residual[3:]))

        current_position, current_angle = split(current_residual)
        other_position, other_angle = split(other_residual)
        switch = (
            other_position <= current_position + JAW_FLIP_POSITION_SLACK_M
            and other_angle < current_angle - JAW_FLIP_GAIN_RAD
        )
        if switch:
            return other_solution, other_residual, not self.jaw_flipped
        return current_solution, current_residual, self.jaw_flipped

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
        tool_pose = self.arm_ik.tool_pose()
        if self.jaw_flipped:
            # Reported in the orientation the policy asked for, not the
            # grasp-equivalent one the wrist adopted. The two describe the same
            # grasp (see `JAW_FLIP`), but only one of them is in the frame the
            # policy is closing its loop in: report the flipped pose and the
            # checkpoint reads its own command as having been rolled half a turn
            # and spends the next chunk undoing it.
            tool_pose = tool_pose @ JAW_FLIP
        target = self.stretch_tool_pose_to_franka(tool_pose)
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
        solution, self.last_residual, self.jaw_flipped = self._solve_either_jaw(target)
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
        """A Robotiq 0-255 command -> Stretch's two finger targets, in radians."""
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
        finger = self.finger_open + fraction * (STRETCH_FINGER_CLOSED - self.finger_open)
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
