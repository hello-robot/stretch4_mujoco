"""
The seven robot/camera setups the search runs, and the eval configs that run them.

The study walks the exo camera from the one MolmoBot-DROID was trained with to
the one Stretch actually has, one change at a time, on both robots:

    1  franka_baseline      Franka, the DROID shoulder camera that ships with MolmoBot
    2  franka_stretchcam    Franka, a pinhole at Stretch's head-camera pose
    3  stretch_stretchcam   Stretch, the same pinhole -- so 2 -> 3 is the robot alone
    4  franka_fisheye       Franka, Stretch's real 123-degree fisheye
    5  stretch_fisheye      Stretch, the same -- 4 -> 5 is again the robot alone
    6  franka_rectified     Franka, that fisheye rectified
    7  stretch_rectified    Stretch, that fisheye rectified

Read in pairs the table separates two things that are otherwise confounded.
1 -> 2 is the camera's *pose* on a robot the policy knows; 2 -> 3, 4 -> 5 and
6 -> 7 are the *robot* under a camera held fixed; 2 -> 4 -> 6 is the *lens*, with
the robot held fixed. A drop that appears at 2 -> 3 but not at 1 -> 2 is the
retargeting; one that appears at 2 -> 4 on the Franka is the fisheye, and no
amount of work on Stretch's kinematics will fix it.

How a trial's parameters reach a worker
---------------------------------------
`run_evaluation()` builds the experiment config itself, from a class named by a
"module:Class" string, and offers no hook for setting a field on it. So the
trial is passed in the environment, as JSON, and read back in `model_post_init`
-- the same route `configs.py` uses for the viewer, the action type and the MP4
export, and for the same reason. It also survives the trip into a rollout
worker, which re-imports this module rather than inheriting anything from the
parent process.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from examples.machine_learning.molmospaces.added_pickup_repair import install_eval_repair
from examples.machine_learning.molmospaces.policies import franka_retarget as fr
from examples.machine_learning.molmospaces.policies.franka_retarget import (
    VirtualFranka,
    franka_start_arm_qpos,
    pose_conventions_requested,
)
from examples.machine_learning.molmospaces.policies.molmobot_droid_policy import (
    StretchMolmoBotDroidPolicy,
    StretchMolmoBotDroidPolicyConfig,
)
from examples.machine_learning.molmospaces.retargetting.cameras import (
    FISHEYE_DISTORTED,
    FISHEYE_NONE,
    FISHEYE_RECTIFIED,
    ExoCameraParams,
    RetargetParams,
    clear_exo_cameras,
    exo_camera_config,
    register_exo_camera,
)
from examples.machine_learning.molmospaces.retargetting.franka_droid_policy import (
    FRANKA_EXO_CAMERA,
    FRANKA_WRIST_CAMERA,
    FrankaMolmoBotDroidPolicyConfig,
)
from examples.machine_learning.molmospaces.stretch import config as stretch_config
from examples.machine_learning.molmospaces.stretch.config import (
    WRIST_CAMERA_RIGHT,
    Stretch4CameraSystem,
    Stretch4RobotConfig,
)
from examples.machine_learning.molmospaces.stretch.episode_overrides import (
    stretch_home_init_qpos,
)
from molmo_spaces.configs.camera_configs import CameraSystemConfig, MjcfCameraConfig
from molmo_spaces.configs.robot_configs import FrankaRobotConfig
from molmo_spaces.evaluation.benchmark_schema import EpisodeSpec
from molmo_spaces.evaluation.configs.evaluation_configs import JsonBenchmarkEvalConfig
from molmo_spaces.evaluation.robot_eval_overrides import (
    ROBOT_OVERRIDE_REGISTRY,
    register_robot_override,
)

log = logging.getLogger(__name__)

CONFIG_MODULE = "examples.machine_learning.molmospaces.retargetting.setups"

PARAMS_ENV_VAR = "STRETCH_RETARGET_PARAMS"
"""JSON blob naming the setup and its parameters. See the module docstring."""

PROBE_SINK_ENV_VAR = "STRETCH_RETARGET_PROBE_SINK"
"""
Directory each rollout process appends its per-episode probe records to.

The same injection route as `PARAMS_ENV_VAR`, and the reason a search can run
`--num-workers` above 1 at all: with several workers the process that watches a
rollout is not the process collecting results, so the probe has to leave its
records somewhere shared. A worker re-imports *this* module (it is the config
module named in the "module:Class" string) and installs the hook below at import
time, exactly as `configs.py` installs the MP4 recorder from
`VIDEO_EXPORT_ENV_VAR`.
"""


def install_worker_hooks() -> None:
    """Install the probe and the MP4 recorder in this process, if asked for.

    Called at import, so it runs in the parent *and* in every rollout worker --
    the parent's own call is harmless because both installers are idempotent.
    Wrapped because a worker that cannot record telemetry should still run the
    rollout; losing a trial to a missing directory would be worse than losing its
    probe data.
    """
    sink = os.environ.get(PROBE_SINK_ENV_VAR)
    if not sink:
        return
    try:
        from examples.machine_learning.molmospaces.retargetting.scoring import install_probe
        from examples.machine_learning.molmospaces.visualize import install_eval_video_hook

        install_probe(Path(sink))
        install_eval_video_hook()
    except Exception as error:  # noqa: BLE001 - telemetry must not sink a worker
        log.warning(f"[retarget] could not install the rollout hooks: {error}")

EXO_CAMERA = FRANKA_EXO_CAMERA
"""The observation key the third-person view arrives under, on either robot."""

FRANKA_WRIST_MJCF = "gripper/wrist_camera"
FRANKA_WRIST_FOV = 56.74
FRANKA_WRIST_RENDER_SIZE = (640, 368)
"""`FrankaDroidCameraSystem`'s wrist camera, reproduced so setup 1 is that system."""

FRANKA_WRIST_CAMERA_REACH_M = 0.1553
STRETCH_WRIST_CAMERA_REACH_M = 0.2414
"""
How far each robot's wrist camera sits from the point its gripper closes on.

Measured on the compiled models at the Franka's home tool pose -- the Franka's
`gripper/wrist_camera` against `gripper/grasp_site`, Stretch's
`gripper_camera_right_rgb` against `grasp_center_link`. Stretch's hand is longer
and its camera is further back along it, which is the whole of the difference.
"""

MATCHED_WRIST_FOV_DEG = 38.3
"""
The wrist FOV at which Stretch frames a grasp the way the Robotiq does, in degrees.

The two cameras have almost the same lens (56.7 degrees against Stretch's 58.0)
and are not in the same place, so an object at the grasp point subtends
1.55x less of Stretch's frame. Matching the *framing* rather than the lens
means solving

    tan(fov_stretch / 2) = tan(fov_franka / 2) * reach_franka / reach_stretch

which is this. Narrowing a render's FOV is what cropping the centre out of the
real camera's frame does, so this is available on the robot as built -- at the
cost of the peripheral view during the approach. `RetargetParams.wrist_fov_deg`
is the knob; nothing sets it by default, because whether the trade is worth it
is a measurement rather than an argument.
"""

FRANKA_LINK0_HEIGHT = 0.75
"""
Height of `fr3_link0` above the floor in a benchmark episode, in metres.

Not a free choice: it is what the released benchmarks put the Franka at (their
`robot_base_pose` z of 0.174 plus `FrankaRobotConfig.base_size[2]` of 0.58), and
it is what `franka_retarget.FRANKA_PEDESTAL_HEIGHT` stands the *virtual* Franka
at on Stretch. The two agreeing is what makes a policy output mean the same
point of the same room on both robots.
"""

FISHEYE_FOVY = 123.19774353065998
"""
`StretchCameras.cam_nav_rgb_se4_right`'s FOV, as MuJoCo wants it.

Hard-coded rather than read at import so this module stays importable without
the camera enum, and because it is a number a search may want to move away from
-- which is the whole point of setups 4 to 7.
"""

FISHEYE_RENDER_SIZE = (640, 400)
"""
1.6 aspect, the sensor's own.

`StretchCameras.fisheye_params_for_frame` projects a 1920x1200 calibration onto
whatever size it is handed, and falls back to a centred approximation when the
aspect disagrees -- which quietly warps the frame differently from the hardware.
"""

DROID_FRAME_SIZE = (640, 360)
"""What the checkpoint was trained on. `--exo-crop` brings a fisheye frame to it."""

STRETCH_GRASP_OFFSET_M = 0.030
STRETCH_TARGET_Z_OFFSET_M = 0.0
"""
The tool correction the Stretch setups retarget with, measured by search.

**The grasp centre.** The retargeting drives `grasp_center_link` to the pose the
policy asked for its Robotiq's `grasp_site`, but that point is in a quite
different place on the two hands: on Stretch it sits 1.5cm *past* the fingertips
and 10.6cm past the finger pads, so an object put there is outside the gripper
and the fingers close behind it. `grasp_offset_m` pushes the commanded grasp
centre that much further along the approach, which pulls the object that much
deeper into the jaw -- 0.015 brings it to the fingertips, 0.106 to the pads.
`diagnose.py` prints both numbers off the compiled model.

Deep is not simply better, for two reasons that bound it from opposite ends.
The offset drives the fingertips `offset - 0.015` metres *past* the object along
the approach, and for a top-down grasp that is straight down: at 0.09 the tips
reach 7.5cm below the object's centre, which for every object in this benchmark
is through the worktop. And Stretch's fingers converge behind their tips, so the
deeper the object sits the narrower the jaw it has to fit in -- past about 0.045
the hand cannot open around what the Robotiq would have swallowed at all. See
`franka_retarget.ROBOTIQ_MAX_APERTURE_M`, which carries that calibration; the
two parameters pick a point on one wedge and cannot be chosen apart.

0.030 is where that comes out, measured by replaying the recorded
`franka_baseline` episodes through the retargeting with the physics on
(`retargetting/replay.py --replay-as-stretch4`), over all 20 of them:

    offset (at the aperture calibrated for it)   0.000  0.015  0.030  0.045  0.055
    picked                                        2/20   3/20   6/20   3/20    5/20

A shallow read of that is that it barely matters, and the reason it looks that
way is worth knowing: a replay is open loop and the gripper closes on the step
the *Franka's* did, which is the right step for a hand 9cm shorter. What the
table can be trusted on is the shape -- nothing at 0, falling away past 0.045 --
and not on 6 against 5.

**The height.** `target_z_offset_m` raises every target for clearance over the
worktop; it is 0 here and the measurement it used to be a fraction of is taken
at the Franka's home pose, above Stretch's lift ceiling, so it corrected targets
that needed no correcting. Anything larger than the object is tall closes the
gripper above it: over these counters the four objects stand 2.6cm (knife) to
5.5cm (salt shaker) proud.

Both are tuned on five kitchens from one robot pose each. Confirm on a real
benchmark before treating them as the retargeting's defaults everywhere:

    run_benchmarks.py --policy molmobot_droid --benchmark pick
"""

HEAD_CAMERA_PITCH_DEG = 43.0
"""
The pitch Stretch's head camera is actually built at, in degrees.

Measured off the MJCF: `camera_right_link`'s optical frame decomposes to yaw -90,
pitch 43, roll -90, and 43 here is 47 degrees below horizontal. It is a fixed
property of the shell -- there is no head tilt joint on an SE4 -- which is
exactly why `virtual_pitch_deg` exists.
"""

RECTIFIED_CROP_NOTE = """
Why the rectified setups sweep a *virtual* pitch.

`pitch_deg` moves the camera, and on a real Stretch nothing moves it: the head
cameras are bolted to the shell at 43 degrees. So a sweep that finds a better
physical pitch has found something you would have to re-manufacture the robot to
use.

The fisheye's 123-degree field is much wider than the 640x360 window the
checkpoint wants, though, so a view that looks like it was taken at another
pitch can be cut out of the frame the hardware already produces -- rectify, then
crop off-centre. These setups pin the camera at `HEAD_CAMERA_PITCH_DEG` and
sweep `virtual_pitch_deg` over the same values the fisheye setups sweep
`pitch_deg` over, so each rectified trial has a physically-tilted twin at the
same angle. If the pair scores alike, the tilt is available in software on the
robot as built; if the rectified one is worse, the difference is what
rectification and cropping cost.
"""

HEAD_CAMERA_ROLL_DEG = -90.0
"""
How far Stretch's head cameras are rolled about their view axis: a quarter turn.

They are bolted on sideways. Measured, not assumed: decomposed intrinsic-ZXZ,
the MJCF's own `camera_right_link` optical frame is yaw -90, pitch 43, roll -90,
and `demo_droid_on_stretch.add_reconstructed_exo_camera` is the same thing with
the roll dropped -- which is deliberate there, because that reconstruction is
for an upright *pinhole* view.

For the fisheye setups the roll has to be back, for two independent reasons:

* **The warp.** `apply_fisheye_distortion` is handed a calibration expressed in
  the sensor's own frame (`fx, fy, cx, cy` of a 1920x1200 sensor, with the
  optical centre ~20px off-centre). Applied to an upright render, that warps the
  frame along axes the lens does not have.
* **The quarter turn.** `quarter_turns=-1` exists to *undo* this roll, the way
  `StatusStretchCamera.get_camera_data` undoes it on the real robot. With the
  roll missing, that turn takes an already-upright frame and lays it on its
  side -- which is what it was doing.

So a fisheye setup renders sideways, warps in the sensor frame, and turns the
result upright: exactly the order `Stretch4MujocoSimulator` does it in.
"""


# =============================================================================
# The setups
# =============================================================================


@dataclass(frozen=True)
class Setup:
    """One row of the table in the module docstring."""

    key: str
    robot: str
    """"franka" or "stretch". Decides the eval config, and nothing else here."""

    description: str
    params: RetargetParams

    camera_dims: tuple[str, ...] = ("pitch_deg", "fovy")
    """
    What `--search sweep`'s camera stage varies for this setup.

    Empty means the setup is not swept at all and gets a single run at its own
    parameters. `franka_baseline` is empty for that reason: it is the control --
    the camera MolmoBot ships with, on the robot the checkpoint was trained on --
    and a "better" pitch for it would answer a question nobody asked while
    costing 15 of the 16 trials.

    The rectified setups swap `pitch_deg` for `virtual_pitch_deg`, because the
    thing worth measuring about a rectified fisheye is not where you could bolt
    it but what you can cut out of it; see `RECTIFIED_CROP_NOTE`.
    """

    @property
    def eval_config(self) -> str:
        return {
            "franka": "RetargetFrankaDroidEvalConfig",
            "stretch": "RetargetStretchDroidEvalConfig",
        }[self.robot]


def _stretch_head_camera_params(mount_body: str, height: float, **overrides: Any) -> ExoCameraParams:
    """Stretch's right head camera, as a mounted pose on `mount_body`.

    The position is `camera_right_link`'s true offset from `base_link`, and the
    orientation is the one `demo_droid_on_stretch.add_reconstructed_exo_camera`
    derives: the link's own axes are not a camera basis on the real robot (the
    URDF remaps them through `camera_right_optical_link`, itself rolled 90
    degrees because the camera is bolted on sideways), so the view *direction*
    is carried across and an upright basis rebuilt around it.

    `height` is the z of that mount in the reference body's own frame, and it is
    the one number that has to differ between the two robots. See
    `FRANKA_STRETCHCAM_HEIGHT` for why.
    """
    return ExoCameraParams(
        mount_body=mount_body,
        pos=(0.0788406, -0.075, height),
        yaw_deg=-90.0,
        pitch_deg=43.0,
        **overrides,
    )


STRETCH_HEAD_CAMERA_OPTICAL_POSE = ((0.0933, -0.075, 1.5277), (-90.0, 43.0, -90.0))
"""
`camera_right_link`'s optical frame relative to `base_link`: `(pos, (yaw, pitch, roll))`.

Measured by walking the generated MJCF from `base_link` to
`camera_right_optical_link` and composing the poses. Recorded rather than used,
because the setups below mount at `STRETCH_STRETCHCAM_HEIGHT` instead -- the
reconstruction `demo_droid_on_stretch.py` uses, which every setup shares so that
the comparison between them is about the lens and not the mount.

The two differ by 1.45cm forward and 6.9cm up. That is the whole of the residual
between `stretch_fisheye` and what `Stretch4MujocoSimulator` renders through the
real camera: with the mount roll fixed the warp, the crop and the quarter turn
all agree, and what is left is parallax from those 7cm. Substitute this pose
here to make the two pixel-identical, at the cost of the fisheye setups sitting
7cm from the pinhole ones.
"""

STRETCH_STRETCHCAM_HEIGHT = 1.4587
"""
Stretch's head camera above `base_link`, in metres.

`camera_right_link`'s true offset (1.5432) less that body's own 0.0845, exactly
as `demo_droid_on_stretch.RECONSTRUCTED_EXO_CAMERA` has it.

Above `base_link`, which is what `_stretch_head_camera_params` wants and so is
right as it stands. It is *not* the height above the floor -- add
`STRETCH_BASE_LINK_HEIGHT_M` for that, which is what `FRANKA_STRETCHCAM_HEIGHT`
does when it puts the same camera on a robot with a different base.
"""

FRANKA_EXO_MOUNT_BODY = "robot_0/fr3_link1"
"""
The body the Franka setups pin the exo camera to, as specified for this study.

`fr3_link1` turns with joint 1, so this camera *yaws with the arm* -- it is a
shoulder-mounted view that follows where the robot is working, not a fixed
exocentric one. That is deliberate: it is the closest a Franka gets to Stretch's
head camera, which likewise turns with the base the arm reaches from.

It does mean setups 2, 4 and 6 are not held still the way 1 is, so a difference
between them and their Stretch counterparts carries this as well as the lens.
`robot_0/fr3_link0` is the stationary parent if that needs ruling out; at the
home pose the two give pixel-identical frames, and they diverge only once the
arm turns.
"""

DROID_SHOULDER_CAMERA_POS = (0.1, 0.57, 0.66)
DROID_SHOULDER_CAMERA_ANGLES = (-139.8504, 52.7169, 7.6879)
"""
The DROID shoulder camera MolmoBot ships with: its offset from `fr3_link0`, and
its intrinsic ZXZ angles.

Pulled out of `franka_baseline` so `stretch_baseline` can mount the *same*
camera without restating the numbers. The angles are the euler form of the
quaternion `FrankaDroidCameraSystem` uses; see `franka_baseline`.
"""

STRETCH_BASE_LINK_HEIGHT_M = 0.028
"""
How far Stretch's `base_link` origin sits above the floor, in metres.

**This is the robot, not a bug, and the robot is not floating.** `base_footprint`
is the root body and it is exactly on the floor; all six wheel geoms bottom out
at z = 0.0000; `base_link` is the chassis origin 28mm above that, and
`base_link_collision` -- the shell -- clears the floor by the same 28mm, which is
the ground clearance the wheels are there to provide. See
`stretch.robot.Stretch4Robot.add_robot_to_scene`, which hangs `base_footprint`
off the holonomic joints, and its module docstring for why the wheels are made
inert.

It is worth a constant because several heights in this file are reasoned about
as "height in the room" and then written as an offset from `base_link`, and
those are the same number only if this is zero. It is not, so every Stretch
camera here sits 2.8cm higher in the room than its docstring claims, and the
Franka setups that transplant a Stretch camera (`FRANKA_STRETCHCAM_HEIGHT`) are
2.8cm lower than the Stretch ones they are paired with.

Nothing subtracts it yet. Correcting it means taking it off
`STRETCH_BASELINE_HEIGHT` and adding it to `FRANKA_STRETCHCAM_HEIGHT`, which
moves every camera in the study at once -- and 2.8cm at 1.4m, looking at a
counter a metre away, is about a degree of framing, well under the run-to-run
noise in `scoring.REPEAT_NOISE_NOTE`. So it is recorded rather than applied in
the middle of a comparison.
"""

STRETCH_BASELINE_ROOM_HEIGHT = FRANKA_LINK0_HEIGHT + DROID_SHOULDER_CAMERA_POS[2]
"""
How high the DROID shoulder camera is **above the floor**: 1.41 m.

On the Franka it sits 0.66 m above `fr3_link0`, which is itself 0.75 m up on a
pedestal. That is the number the pair is built around: the camera has to end up
at the same height in the room on both robots, or the pair stops being a
comparison of the robot and becomes a comparison of two viewpoints.
"""

STRETCH_BASELINE_HEIGHT = STRETCH_BASELINE_ROOM_HEIGHT - STRETCH_BASE_LINK_HEIGHT_M
"""
Where to mount it on Stretch: 1.382 m above `base_link`, which is 1.41 m above the floor.

The two are not the same number and that is the whole of this constant.
`ExoCameraParams.pos` is a `camera_offset` from `mount_body` -- see
`cameras.exo_camera_config` -- so a height reasoned about as "above the floor"
and written straight into it lands 28 mm high, because `base_link` is the
chassis origin and sits that far up (`STRETCH_BASE_LINK_HEIGHT_M`).

It used to be written straight in. Measured, with both robots stood at the mini
benchmark's spawn and the same yaw, the two cameras came out at the same xy to
the millimetre and 28 mm apart in z; with this subtraction they agree in z too.

The xy offset and the orientation are carried across unchanged -- it is the same
camera, looking the same way, at the same height. What differs is the robot under
it, which is the whole point of the pair.
"""

FRANKA_LINK1_HEIGHT = 0.333
"""`fr3_link1`'s origin above `fr3_link0`, measured off the compiled model."""

FRANKA_STRETCHCAM_HEIGHT = (
    STRETCH_STRETCHCAM_HEIGHT
    + STRETCH_BASE_LINK_HEIGHT_M
    + 0.0845
    - FRANKA_LINK0_HEIGHT
    - FRANKA_LINK1_HEIGHT
)
"""
Where Stretch's head camera goes on a Franka: 0.460 above `fr3_link1`.

Derived, not written down, and the derivation is the point: the camera must end
up at **the same height in the room** on both robots. That is what makes setups
2 and 3 (or 4 and 5) a comparison of the *robot* rather than of two different
viewpoints -- the whole premise of transplanting Stretch's camera onto a Franka
is that the camera does not move, only the arm under it does.

Stretch's head camera is 1.5432 m above `base_link`, which is
`STRETCH_BASE_LINK_HEIGHT_M` off the floor, so 1.5712 m above the floor -- the
distinction this constant used to miss, and it was 28 mm low for it. A benchmark
Franka's `fr3_link0` is 0.75 m up on a pedestal (`FRANKA_LINK0_HEIGHT`) and
`fr3_link1` another 0.333 m above that, so matching the height in the room means
1.5712 - 0.75 - 0.333 = 0.488 m above `fr3_link1`.

The number that looks like Stretch's -- 1.21024, which is 1.5432 less
`fr3_link1`'s own 0.333 -- is the offset that would be right if `fr3_link0`
stood on the floor. It does not, so that value puts the camera 2.29 m up,
0.75 m higher above the room than the same mount is on Stretch, and it is a
different viewpoint rather than the same one. Rendered on this kitchen at the
pinhole defaults the target object is not in a single frame of the episode.

Fixed, either way: no setup and no search moves it, which is why
`params_search.DIMENSIONS` contains no translation.
"""

SETUPS: dict[str, Setup] = {
    setup.key: setup
    for setup in (
        Setup(
            key="franka_baseline",
            robot="franka",
            camera_dims=(),
            description="Franka + the DROID shoulder camera MolmoBot ships with (control, not swept)",
            params=RetargetParams(
                exo=ExoCameraParams(
                    mount_body="robot_0/fr3_link0",
                    pos=DROID_SHOULDER_CAMERA_POS,
                    # The quaternion [-0.3633, -0.1241, 0.4263, 0.8191] that
                    # `FrankaDroidCameraSystem` mounts this camera with, as the
                    # euler triple every other setup here is written in.
                    # `FrankaDroidCameraSystem` mounts this one with the
                    # quaternion [-0.3633, -0.1241, 0.4263, 0.8191]; these are
                    # its intrinsic ZXZ angles, which reproduce it exactly.
                    yaw_deg=DROID_SHOULDER_CAMERA_ANGLES[0],
                    pitch_deg=DROID_SHOULDER_CAMERA_ANGLES[1],
                    roll_deg=DROID_SHOULDER_CAMERA_ANGLES[2],
                    fovy=71.0,
                    render_size=DROID_FRAME_SIZE,
                    fisheye=FISHEYE_NONE,
                    quarter_turns=0,
                )
            ),
        ),
        Setup(
            key="stretch_baseline",
            robot="stretch",
            camera_dims=(),
            description="Stretch + the DROID shoulder camera, at the same height in the room",
            params=RetargetParams(
                grasp_offset_m=STRETCH_GRASP_OFFSET_M,
                target_z_offset_m=STRETCH_TARGET_Z_OFFSET_M,
                exo=ExoCameraParams(
                    # The same camera as `franka_baseline`, at the same height
                    # above the floor -- see `STRETCH_BASELINE_HEIGHT`. It hangs
                    # off `base_link` rather than an arm link, so unlike the
                    # Franka's it does not turn with the arm; that is the one
                    # thing the transplant cannot preserve, because Stretch has
                    # no link that moves the way `fr3_link0` does.
                    mount_body="robot_0/base_link",
                    pos=(
                        DROID_SHOULDER_CAMERA_POS[0],
                        DROID_SHOULDER_CAMERA_POS[1],
                        STRETCH_BASELINE_HEIGHT,
                    ),
                    yaw_deg=DROID_SHOULDER_CAMERA_ANGLES[0],
                    pitch_deg=DROID_SHOULDER_CAMERA_ANGLES[1],
                    roll_deg=DROID_SHOULDER_CAMERA_ANGLES[2],
                    fovy=71.0,
                    render_size=DROID_FRAME_SIZE,
                    fisheye=FISHEYE_NONE,
                    quarter_turns=0,
                ),
            ),
        ),
        Setup(
            key="franka_stretchcam",
            robot="franka",
            description="Franka + an upright pinhole at Stretch's head-camera pose",
            params=RetargetParams(
                exo=_stretch_head_camera_params(
                    FRANKA_EXO_MOUNT_BODY,
                    FRANKA_STRETCHCAM_HEIGHT,
                    fovy=71.0,
                    render_size=DROID_FRAME_SIZE,
                )
            ),
        ),
        Setup(
            key="stretch_stretchcam",
            robot="stretch",
            description="Stretch + the same upright pinhole",
            params=RetargetParams(
                grasp_offset_m=STRETCH_GRASP_OFFSET_M,
                target_z_offset_m=STRETCH_TARGET_Z_OFFSET_M,
                exo=_stretch_head_camera_params(
                    "robot_0/base_link",
                    STRETCH_STRETCHCAM_HEIGHT,
                    fovy=71.0,
                    render_size=DROID_FRAME_SIZE,
                )
            ),
        ),
        Setup(
            key="franka_fisheye",
            robot="franka",
            description="Franka + Stretch's real 123-degree fisheye",
            params=RetargetParams(
                exo=_stretch_head_camera_params(
                    FRANKA_EXO_MOUNT_BODY,
                    FRANKA_STRETCHCAM_HEIGHT,
                    fovy=FISHEYE_FOVY,
                    render_size=FISHEYE_RENDER_SIZE,
                    roll_deg=HEAD_CAMERA_ROLL_DEG,
                    fisheye=FISHEYE_DISTORTED,
                    quarter_turns=-1,
                )
            ),
        ),
        Setup(
            key="stretch_fisheye",
            robot="stretch",
            description="Stretch + its real 123-degree fisheye",
            params=RetargetParams(
                grasp_offset_m=STRETCH_GRASP_OFFSET_M,
                target_z_offset_m=STRETCH_TARGET_Z_OFFSET_M,
                exo=_stretch_head_camera_params(
                    "robot_0/base_link",
                    STRETCH_STRETCHCAM_HEIGHT,
                    fovy=FISHEYE_FOVY,
                    render_size=FISHEYE_RENDER_SIZE,
                    roll_deg=HEAD_CAMERA_ROLL_DEG,
                    fisheye=FISHEYE_DISTORTED,
                    quarter_turns=-1,
                )
            ),
        ),
        Setup(
            key="franka_rectified",
            robot="franka",
            camera_dims=("pitch_deg", "fovy"),
            description="Franka + that fisheye, rectified, whole frame",
            params=RetargetParams(
                # Rectified and nothing else: no `crop_to`, so the whole
                # rectified frame is delivered, and no `virtual_pitch_deg`, so
                # the only tilt is the camera's own physical one. Both were here
                # to trade field of view for the landscape shape the checkpoint
                # was trained on and to synthesise a pitch by cropping
                # off-centre; neither is part of "rectify the 123-degree fisheye",
                # and leaving them in made this pair carry two changes rather
                # than one.
                exo=_stretch_head_camera_params(
                    FRANKA_EXO_MOUNT_BODY,
                    FRANKA_STRETCHCAM_HEIGHT,
                    fovy=FISHEYE_FOVY,
                    render_size=FISHEYE_RENDER_SIZE,
                    roll_deg=HEAD_CAMERA_ROLL_DEG,
                    fisheye=FISHEYE_RECTIFIED,
                    quarter_turns=-1,
                )
            ),
        ),
        Setup(
            key="stretch_rectified",
            robot="stretch",
            camera_dims=("pitch_deg", "fovy"),
            description="Stretch + that fisheye, rectified, whole frame",
            params=RetargetParams(
                grasp_offset_m=STRETCH_GRASP_OFFSET_M,
                target_z_offset_m=STRETCH_TARGET_Z_OFFSET_M,
                # Rectified and nothing else; see `franka_rectified`.
                exo=_stretch_head_camera_params(
                    "robot_0/base_link",
                    STRETCH_STRETCHCAM_HEIGHT,
                    fovy=FISHEYE_FOVY,
                    render_size=FISHEYE_RENDER_SIZE,
                    roll_deg=HEAD_CAMERA_ROLL_DEG,
                    fisheye=FISHEYE_RECTIFIED,
                    quarter_turns=-1,
                )
            ),
        ),
    )
}

SETUP_KEYS = tuple(SETUPS)


# =============================================================================
# Carrying a trial into the evaluation, and into its workers
# =============================================================================


def params_to_json(setup_key: str, params: RetargetParams) -> str:
    payload = {"setup": setup_key, "params": dataclasses.asdict(params)}
    return json.dumps(payload)


def params_from_json(blob: str) -> tuple[str, RetargetParams]:
    payload = json.loads(blob)
    exo = dict(payload["params"].pop("exo"))
    # JSON has no tuples, and these are compared, hashed and unpacked as pairs.
    exo["pos"] = tuple(exo["pos"])
    exo["render_size"] = tuple(exo["render_size"])
    exo["crop_to"] = tuple(exo["crop_to"]) if exo["crop_to"] is not None else None
    return payload["setup"], RetargetParams(exo=ExoCameraParams(**exo), **payload["params"])


def publish_params(setup_key: str, params: RetargetParams) -> None:
    """Put a trial in the environment, where the eval config and its workers can read it."""
    os.environ[PARAMS_ENV_VAR] = params_to_json(setup_key, params)
    # This process imported the module before the trial was chosen, so the
    # registration it would have done at import time has to happen now.
    clear_exo_cameras()
    _register_cameras(params)


def active_trial() -> tuple[str, RetargetParams]:
    """The trial this process is running. Falls back to the first setup's defaults."""
    blob = os.environ.get(PARAMS_ENV_VAR)
    if not blob:
        default = SETUPS[SETUP_KEYS[0]]
        return default.key, default.params
    return params_from_json(blob)


def _register_cameras(params: RetargetParams) -> None:
    """Tell the render hook what to do with each camera's frames, in this process."""
    register_exo_camera(EXO_CAMERA, params.exo)
    # Not post-processed, but its render rectangle still has to be declared or
    # it inherits the shared buffer's aspect and sees a wider scene than the
    # hardware does. See `cameras.register_exo_camera`.
    stretch_config.CAMERA_RENDER_SIZE.setdefault(FRANKA_WRIST_CAMERA, FRANKA_WRIST_RENDER_SIZE)


# =============================================================================
# Camera systems
# =============================================================================


def _buffer_resolution(*sizes: tuple[int, int]) -> tuple[int, int]:
    """The offscreen buffer every camera renders into: big enough for all of them."""
    return (max(size[0] for size in sizes), max(size[1] for size in sizes))


def franka_camera_system(params: RetargetParams) -> CameraSystemConfig:
    """The exo camera under test, plus the Franka's own wrist camera."""
    return CameraSystemConfig(
        img_resolution=_buffer_resolution(params.exo.render_size, FRANKA_WRIST_RENDER_SIZE),
        cameras=[
            exo_camera_config(EXO_CAMERA, params.exo),
            MjcfCameraConfig(
                name=FRANKA_WRIST_CAMERA,
                mjcf_name=FRANKA_WRIST_MJCF,
                robot_namespace="robot_0/",
                fov=FRANKA_WRIST_FOV,
            ),
        ],
    )


def stretch_camera_system(params: RetargetParams) -> CameraSystemConfig:
    """The exo camera under test, plus Stretch's own right wrist camera.

    The wrist camera is `Stretch4CameraSystem`'s -- an MJCF camera rendered
    through `install_stretch_camera_hooks` like any other -- at the hardware's
    own FOV unless `params.wrist_fov_deg` says otherwise. Narrowing it is a
    centre crop of the real camera's frame expressed as a render setting; see
    `RetargetParams.wrist_fov_deg` for the measurement that motivates one and
    for what it costs.
    """
    wrist = next(
        camera for camera in Stretch4CameraSystem().cameras if camera.name == WRIST_CAMERA_RIGHT
    )
    if params.wrist_fov_deg:
        # A copy, because `Stretch4CameraSystem()` hands back configs that other
        # callers in this process share; mutating this one would narrow the
        # wrist camera of every Stretch evaluation that follows.
        wrist = wrist.model_copy(update={"fov": float(params.wrist_fov_deg)})
    wrist_size = stretch_config.CAMERA_RENDER_SIZE[WRIST_CAMERA_RIGHT]
    return CameraSystemConfig(
        img_resolution=_buffer_resolution(params.exo.render_size, wrist_size),
        cameras=[exo_camera_config(EXO_CAMERA, params.exo), wrist],
    )


# =============================================================================
# Robot configs, and the episode overrides that retarget an episode onto them
# =============================================================================


class RetargetFrankaRobotConfig(FrankaRobotConfig):
    """`FrankaRobotConfig` under its own override key.

    A subclass purely so `franka_episode_override` can be registered against it:
    `get_robot_override` walks the MRO and takes the first class it finds, so a
    subclass wins over anything registered for the parent, and registering
    against `FrankaRobotConfig` itself would change every other Franka
    evaluation in the process.
    """


class RetargetStretch4RobotConfig(Stretch4RobotConfig):
    """`Stretch4RobotConfig` under its own override key. Same reason as above --
    and here it matters more, because `configs.py` has already registered
    `stretch_episode_override` for the parent, and that one installs the full
    `Stretch4CameraSystem` over the exo camera this study is measuring."""


def _point_base_at(task: dict, base_z: float, retreat: bool = False) -> None:
    """Face the robot's +x at the pickup object, at the height its base belongs at.

    Position is left exactly where the episode authored it: the study needs every
    setup to see the same scene from the same place, and the mini benchmark
    already stands the robot at a distance both robots can work at. Yaw is
    recomputed rather than trusted because +x is the axis Stretch's arm extends
    along and the Franka's reaches along, so it is what "facing the object"
    means for either.

    `retreat` asks for `stretch_spawn_base_pose`, which moves Stretch back so its
    spawn gripper pose is the Franka's -- Stretch only, and only when
    `match_stretch_spawn_pose_to_franka` is on. The yaw is computed from the
    authored position either way, so the robot still faces the object it was
    placed to face rather than the point it retreated from.
    """
    base_pose = list(task["robot_base_pose"])
    base_xy = np.asarray(base_pose[:2], dtype=float)
    target_xy = np.asarray(task["pickup_obj_start_pose"][:2], dtype=float)

    offset = target_xy - base_xy
    yaw = float(np.arctan2(offset[1], offset[0])) if np.linalg.norm(offset) > 1e-6 else 0.0
    if retreat:
        base_xy = np.asarray(stretch_spawn_base_pose(base_xy, yaw), dtype=float)
    task["robot_base_pose"] = [
        float(base_xy[0]),
        float(base_xy[1]),
        float(base_z),
        float(np.cos(yaw / 2.0)),
        0.0,
        0.0,
        float(np.sin(yaw / 2.0)),
    ]


def franka_episode_override(episode_spec: EpisodeSpec, exp_config: Any) -> None:
    """Put the episode on a Franka Droid with the trial's exo camera.

    Mutates `exp_config.camera_config` in place rather than assigning a new one:
    `JsonEvalTaskSampler` keeps its own reference to the recorded system and
    hands *that* to `setup_cameras()`, while the sensor suite is built from
    `exp_config.camera_config`. They are the same object until something rebinds
    the attribute, at which point the cameras created and the sensors created to
    read them stop agreeing. (`stretch_episode_override` and
    `cap_robot_eval_override` mutate in place for the same reason.)
    """
    _, params = active_trial()
    _register_cameras(params)

    episode_spec.robot.robot_name = "franka_droid"
    episode_spec.robot.init_qpos = {
        key: list(value)
        for key, value in FrankaRobotConfig.model_fields["init_qpos"].default.items()
    }
    conventions = pose_conventions_requested()
    if conventions.changes_franka_start_pose:
        # The Franka half of the start-pose flags. The Stretch half is in
        # `FrankaOnStretchView.__init__`, which adjusts the *virtual* Franka the
        # retargeting snaps Stretch to; both read the same two variables through
        # `start_pose_changes_requested`, so the two conditions cannot end up
        # starting at different poses -- which is the one thing these flags exist
        # to prevent.
        episode_spec.robot.init_qpos["arm"] = [
            float(angle)
            for angle in franka_start_arm_qpos(VirtualFranka(), conventions, FRANKA_LINK0_HEIGHT)
        ]

    system = franka_camera_system(params)
    exp_config.camera_config.cameras = list(system.cameras)
    exp_config.camera_config.img_resolution = system.img_resolution

    base_size = exp_config.robot_config.base_size
    pedestal = float(base_size[2]) if base_size else 0.0
    _point_base_at(episode_spec.task, FRANKA_LINK0_HEIGHT - pedestal)


STRETCH_SPAWN_ARM_M = 0.2
"""
How far Stretch's arm is telescoped out when an episode spawns it, in metres.

`Stretch4RobotConfig.init_qpos` stows the arm at 0, and stowing is load-bearing
for the *other* benchmarks -- `stretch.episode_overrides.stretch_home_init_qpos`
explains that an unstowed Stretch spawns with its tool inside whatever the target
is sitting on. It is not load-bearing here, because this study stands the robot
where the Franka stands rather than in Stretch's own reach band, and a spawn at
the very bottom of the arm's 0..0.52 travel is a spawn with the IK's seed pinned
against a limit in the direction it most often needs to move.

Overridden here rather than in `Stretch4RobotConfig` so that only this study's
episodes see it.
"""


def stretch_spawn_base_pose(base_xy, yaw: float) -> tuple[float, float]:
    """`base_xy`, moved back so Stretch's spawn gripper pose is the Franka's.

    A no-op unless `match_stretch_spawn_pose_to_franka` is on. The offset is in
    the base's own axes, so it is rotated by the spawn yaw before it is applied.
    """
    base_xy = np.asarray(base_xy, dtype=float)[:2]
    if not pose_conventions_requested().match_stretch_spawn_pose_to_franka:
        return float(base_xy[0]), float(base_xy[1])
    forward = np.array([np.cos(yaw), np.sin(yaw)])
    across = np.array([-np.sin(yaw), np.cos(yaw)])
    moved = (
        base_xy
        + forward * fr.STRETCH_SPAWN_BASE_OFFSET_XY[0]
        + across * fr.STRETCH_SPAWN_BASE_OFFSET_XY[1]
    )
    return float(moved[0]), float(moved[1])


def stretch_spawn_init_qpos() -> dict[str, list[float]]:
    """Stretch's per-move-group spawn pose for this study.

    `stretch_home_init_qpos()` with the arm out at `STRETCH_SPAWN_ARM_M`, and the
    wrist rolled half a turn when `--change_stretch_start_pose_flip_wrist` asks
    for it -- a roll about the approach axis, so the grasp centre does not move
    and what swings round is the hand and the two cameras on it.

    The roll goes *negative*: `wrist_roll_joint` runs about [-4.276, +1.135] rad,
    so +pi is outside its travel and -pi is not. Measured, not assumed, and the
    grasp centre comes out in the same place to the micrometre either way.
    """
    init_qpos = stretch_home_init_qpos()
    init_qpos["arm"] = [float(STRETCH_SPAWN_ARM_M)]
    if pose_conventions_requested().change_stretch_start_pose_flip_wrist:
        yaw, pitch, _ = init_qpos.get("wrist", [0.0, 0.0, 0.0])
        init_qpos["wrist"] = [float(yaw), float(pitch), -float(np.pi)]
    return init_qpos


def stretch_episode_override(episode_spec: EpisodeSpec, exp_config: Any) -> None:
    """Put the episode on Stretch 4 with the trial's exo camera.

    The camera half of `stretch.episode_overrides.stretch_episode_override`
    replaced -- that one installs all six Stretch cameras, which would render
    the exo view this study varies as a seventh nobody reads. The base pose is
    the same; the start configuration is `stretch_spawn_init_qpos`.
    """
    _, params = active_trial()
    _register_cameras(params)

    episode_spec.robot.robot_name = "stretch4"
    episode_spec.robot.init_qpos = stretch_spawn_init_qpos()

    system = stretch_camera_system(params)
    exp_config.camera_config.cameras = list(system.cameras)
    exp_config.camera_config.img_resolution = system.img_resolution

    _point_base_at(episode_spec.task, 0.0, retreat=True)


def register_overrides() -> None:
    """Register both overrides, once per process.

    `register_robot_override` raises on a duplicate, and this module is imported
    once per rollout worker *and* again when an eval config is resolved from its
    "module:Class" string, so the guard is load-bearing.
    """
    if RetargetFrankaRobotConfig not in ROBOT_OVERRIDE_REGISTRY:
        register_robot_override(RetargetFrankaRobotConfig, franka_episode_override)
    if RetargetStretch4RobotConfig not in ROBOT_OVERRIDE_REGISTRY:
        register_robot_override(RetargetStretch4RobotConfig, stretch_episode_override)


register_overrides()
install_worker_hooks()

# Unconditionally, and not inside `install_worker_hooks`, which is gated on the
# probe's environment variable: this one is not telemetry and must apply in every
# worker whether or not a trial is being recorded.
#
# The benchmark's four target objects are *added* pickupables, and an added THOR
# prefab carries no visual-mesh mass, so MuJoCo derives one from surface area at
# 1000 kg/m^3. `added_pickup_repair` measures the 29 graspable potato assets at
# 16-40 kg, median 23. `configs.py` installs this for `run_benchmarks.py`, and
# this study never imported `configs.py` -- its eval configs derive from
# MolmoSpaces' `JsonBenchmarkEvalConfig` directly -- so every trial here has been
# scoring a ~23 kg potato.
#
# The signature is unmistakable in `eval_output_sept_19`: over 245 Franka
# episodes the potato was *touched* 239 times and approached closer than any
# other object (21mm against 28-79mm), yet lifted a median 5.99mm against the
# 10mm success threshold, and picked up exactly 0 times -- on the un-retargeted
# Franka, with the checkpoint's own camera. See `added_pickup_repair`, whose own
# measurement of this was 0/8 solved before and 3/8 after.
install_eval_repair()


# =============================================================================
# The Stretch policy, with the gripper parameters this study adds
# =============================================================================


class RetargetStretchMolmoBotDroidPolicyConfig(StretchMolmoBotDroidPolicyConfig):
    """`StretchMolmoBotDroidPolicyConfig` plus a searchable tool correction."""

    grasp_offset_m: float = 0.0
    """See `RetargetParams.grasp_offset_m`."""

    wrist_tilt_deg: float = 0.0
    """See `RetargetParams.wrist_tilt_deg`."""

    aperture_m: float = 0.0
    """See `RetargetParams.aperture_m`. 0 keeps `ROBOTIQ_MAX_APERTURE_M`."""

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        # The parent sets these to the un-parameterised classes.
        self.policy_cls = RetargetStretchMolmoBotDroidPolicy
        from molmo_spaces.utils.function_utils import make_lenient

        self.policy_factory = make_lenient(RetargetStretchMolmoBotDroidPolicy)


class RetargetStretchMolmoBotDroidPolicy(StretchMolmoBotDroidPolicy):
    """The retargeting policy with the Franka-to-Stretch tool transform opened up.

    `FrankaOnStretchView` carries that transform as `_tool_correction`, a fixed
    -90 degree rotation about y that lines the Robotiq's approach axis (+z) up
    with Stretch's (+x). It is the right transform for two grippers of the same
    length, and Stretch's is not: its fingers reach some centimetres further past
    the wrist than the Robotiq's do, so a tool pose that would close the Robotiq
    around an object closes Stretch's palm around the air behind it.

    So the correction gains two terms -- a pitch and a translation along the
    approach -- and the search decides them. Replacing the matrix after
    construction rather than threading parameters through
    `FrankaOnStretchView.__init__` keeps this study out of the shared
    retargeting module; the transform is one attribute and its inverse, and both
    are recomputed here together.
    """

    def _build_proxy(self, robot_view):
        proxy = super()._build_proxy(robot_view)
        policy_config = self.config.policy_config
        apply_tool_correction(
            proxy,
            wrist_tilt_deg=float(getattr(policy_config, "wrist_tilt_deg", 0.0)),
            grasp_offset_m=float(getattr(policy_config, "grasp_offset_m", 0.0)),
        )
        apply_aperture(proxy, float(getattr(policy_config, "aperture_m", 0.0)))
        return proxy


def apply_tool_correction(proxy: Any, wrist_tilt_deg: float, grasp_offset_m: float) -> None:
    """Rewrite a `FrankaOnStretchView`'s tool transform in place.

    The composition is rotate-then-translate in the *Stretch* tool frame:

        correction = FRANKA_TO_STRETCH_TOOL @ R_y(tilt) @ T(offset along +x)

    so `grasp_offset_m` moves the commanded grasp centre along the approach
    direction after the tilt has decided which way that is, which is the way
    round that makes the two parameters independent.
    """
    from scipy.spatial.transform import Rotation as R

    from examples.machine_learning.molmospaces.policies.franka_retarget import (
        FRANKA_TO_STRETCH_TOOL,
    )

    from examples.machine_learning.molmospaces.policies.franka_retarget import JAW_FLIP

    correction = np.eye(4)
    correction[:3, :3] = FRANKA_TO_STRETCH_TOOL @ R.from_euler(
        "y", wrist_tilt_deg, degrees=True
    ).as_matrix()
    # +x is Stretch's approach axis; see `franka_retarget.FRANKA_TO_STRETCH_TOOL`.
    correction[:3, 3] = correction[:3, :3] @ np.array([grasp_offset_m, 0.0, 0.0])
    if getattr(proxy, "pose_conventions", None) and (
        proxy.pose_conventions.map_franka_wrist_to_flipped_stretch4_wrist
    ):
        # This rebuilds the transform from scratch, so it has to re-apply the
        # half turn `FrankaOnStretchView.__init__` folded in -- otherwise asking
        # for a tool correction would silently undo the convention. On the right,
        # after the offset, for the same reason it is on the right there: the
        # turn is about the approach axis, which the rotation above defines.
        correction = correction @ JAW_FLIP

    proxy._tool_correction = correction
    proxy._tool_correction_inverse = np.linalg.inv(correction)


def apply_aperture(proxy: Any, aperture_m: float) -> None:
    """Re-solve a `FrankaOnStretchView`'s open finger angle for a different aperture.

    `FrankaOnStretchView` takes this at construction; the policy builds its proxy
    through a base class that does not pass it, so this redoes the one
    measurement that depends on it. Zero leaves the proxy alone, which is what
    every setup that does not name an aperture wants.

    Does nothing to a proxy built with `match_robotiq_aperture` off -- that hand
    is already open as wide as it goes, and there is nothing to narrow.
    """
    from examples.machine_learning.molmospaces.policies.franka_retarget import (
        STRETCH_FINGER_OPEN,
        stretch_finger_for_aperture,
    )

    if not aperture_m or proxy.finger_open >= STRETCH_FINGER_OPEN:
        return
    proxy.robotiq_aperture_m = float(aperture_m)
    proxy.finger_open = stretch_finger_for_aperture(
        proxy.arm_ik._scratch_view.get_move_group("gripper"),
        proxy.arm_ik._scratch_data.model,
        proxy.arm_ik._scratch_data,
        float(aperture_m),
    )


# =============================================================================
# Eval configs
# =============================================================================


class _RetargetEvalConfig(JsonBenchmarkEvalConfig):
    """Settings both robots' configs share.

    The timing is `Stretch4BenchmarkEvalConfig`'s, so a rollout from this study
    steps the same way a `run_benchmarks.py` one does and the scores are
    comparable with the benchmark numbers.
    """

    policy_dt_ms: float = 66.0
    ctrl_dt_ms: float = 2.0
    sim_dt_ms: float = 2.0
    end_on_success: bool = True

    def _apply_trial(self) -> RetargetParams:
        """Read this process's trial and register its cameras. Called from `model_post_init`."""
        _, params = active_trial()
        _register_cameras(params)
        return params


class RetargetFrankaDroidEvalConfig(_RetargetEvalConfig):
    """MolmoBot-DROID on the Franka it was trained on. Setups 1, 2, 4 and 6."""

    robot_config: RetargetFrankaRobotConfig = RetargetFrankaRobotConfig()
    policy_config: FrankaMolmoBotDroidPolicyConfig = FrankaMolmoBotDroidPolicyConfig()
    camera_config: CameraSystemConfig | None = CameraSystemConfig()

    @property
    def tag(self) -> str:
        return "retarget_franka_droid"

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        params = self._apply_trial()
        system = franka_camera_system(params)
        self.camera_config.cameras = list(system.cameras)
        self.camera_config.img_resolution = system.img_resolution
        self.robot_config.action_noise_config.enabled = False


class RetargetStretchDroidEvalConfig(_RetargetEvalConfig):
    """MolmoBot-DROID retargeted onto Stretch 4. Setups 3, 5 and 7."""

    robot_config: RetargetStretch4RobotConfig = RetargetStretch4RobotConfig()
    policy_config: RetargetStretchMolmoBotDroidPolicyConfig = (
        RetargetStretchMolmoBotDroidPolicyConfig(
            exo_camera=EXO_CAMERA, wrist_camera=WRIST_CAMERA_RIGHT
        )
    )
    camera_config: CameraSystemConfig | None = CameraSystemConfig()

    @property
    def tag(self) -> str:
        return "retarget_stretch_droid"

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        params = self._apply_trial()
        system = stretch_camera_system(params)
        self.camera_config.cameras = list(system.cameras)
        self.camera_config.img_resolution = system.img_resolution
        # Guarded so that a subclass swapping the policy out -- for a dummy
        # policy, to check that a setup's scene and cameras load without waiting
        # on a VLA -- gets the cameras without being rejected for lacking the
        # gripper fields.
        if isinstance(self.policy_config, RetargetStretchMolmoBotDroidPolicyConfig):
            self.policy_config.grasp_offset_m = params.grasp_offset_m
            self.policy_config.wrist_tilt_deg = params.wrist_tilt_deg
            self.policy_config.target_z_offset = params.target_z_offset_m
            self.policy_config.aperture_m = params.aperture_m


def qualified_config_name(class_name: str) -> str:
    """"module:Class", the form `run_evaluation` resolves a config name in."""
    return f"{CONFIG_MODULE}:{class_name}"
