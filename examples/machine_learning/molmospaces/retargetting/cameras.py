"""
The exo camera under test: where it is mounted, what it sees, what it does to the pixels.

Every one of the seven setups in `setups.py` differs only in this camera, so it
is the one thing this package parameterises end to end. `ExoCameraParams`
carries the numbers a trial varies; `exo_camera_config()` turns them into the
`RobotMountedCameraConfig` MolmoSpaces places, and `install_exo_camera_hook()`
applies the parts MolmoSpaces has no concept of -- the fisheye warp, the
rectification, the quarter turn and the crop.

Why a *mounted* camera rather than an MJCF one
----------------------------------------------
Stretch's real head camera is `camera_right_link` in the generated MJCF, and
`Stretch4CameraSystem` reads it through an `MjcfCameraConfig`. That is the right
thing when the question is "what does the robot see"; it is the wrong thing
here, because a search over pitch and field of view cannot touch an
`MjcfCameraConfig` -- the extrinsics live in the compiled model. A
`RobotMountedCameraConfig` is a free camera pinned to a body, so the same
parameters describe the view on a Franka's `fr3_link1` and on Stretch's
`base_link`, which is what makes setups 2 and 3 (or 4 and 5) comparable at all.

The mount *position* is not among those parameters. It is the transplant under
test and is held fixed everywhere; only the optics around it move.

At the default parameters the mounted camera reproduces the MJCF one: the
defaults in `setups.py` are `camera_right_link`'s own pose.

The four image-space stages
---------------------------
A MuJoCo render is a pinhole render, and the real head camera is not a pinhole,
so what the policy is shown goes through up to four more steps. Each is a
parameter, because each is a candidate explanation for a bad rollout:

1. **distortion** -- `apply_fisheye_distortion` warps the wide pinhole with the
   camera's real coefficients and crops to the part the pinhole could fill. This
   is what the hardware sees. `FISHEYE_NONE` skips it, which is the pinhole
   ablation.
2. **rectification** -- undoing 1 with the same model, which is what a
   rectified stream off the real robot is. Not the same as skipping 1: the
   round trip keeps the fisheye's resampling loss and its blind corners, so the
   difference between `FISHEYE_RECTIFIED` and `FISHEYE_NONE` is exactly what
   rectification cannot give back.
3. **the quarter turn** -- the head cameras are bolted on sideways and
   `StatusStretchCamera.get_camera_data` rotates every frame before anyone sees
   it, which swaps width and height. A policy trained on 640x360 landscape gets
   400x640 portrait if this is left in.
4. **the crop** -- taking a 640x360 landscape window back out of the portrait
   frame, which is the shape the checkpoint was trained on.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

from molmo_spaces.configs.camera_configs import RobotMountedCameraConfig
from stretch4_mujoco.enums.stretch_cameras import StretchCameras

log = logging.getLogger(__name__)

FISHEYE_NONE = "none"
"""Leave the pinhole render alone. Straight lines stay straight."""

FISHEYE_DISTORTED = "distorted"
"""Warp with the real coefficients. What the hardware actually delivers."""

FISHEYE_RECTIFIED = "rectified"
"""Warp, then undo the warp. What a rectified stream off the robot looks like."""

FISHEYE_MODES = (FISHEYE_NONE, FISHEYE_DISTORTED, FISHEYE_RECTIFIED)

FISHEYE_CAMERA = StretchCameras.cam_nav_rgb_se4_right
"""
The calibration the warp is taken from: Stretch 4's right head camera.

The right one rather than the left because it is the one
`demo_droid_on_stretch.py --fisheye` uses and the one whose mounting pose the
default parameters reproduce. Its `rotate_number_of_times` is -1, a quarter turn
clockwise.
"""


@dataclass(frozen=True)
class ExoCameraParams:
    """Everything a trial may vary about the third-person view.

    Frozen because a trial's parameters are a record of what was run: a search
    produces new ones with `replace()` rather than editing one in place.
    """

    mount_body: str = "robot_0/base_link"
    """Body the camera is pinned to, namespace included.

    On Stretch this is `base_link`, which is stationary. On the Franka it is
    `fr3_link1`, which turns with joint 1, so that view yaws with the arm the way
    Stretch's head camera yaws with the base its arm reaches from. See
    `setups.FRANKA_EXO_MOUNT_BODY`.
    """

    pos: tuple[float, float, float] = (0.0788406, -0.075, 1.4587)
    """Camera position in the mount body's frame, in metres.

    Fixed by the setup and never searched: it is where Stretch's right head
    camera mounts, and the whole premise of the transplant is that this point is
    held while the optics around it change. `params_search.DIMENSIONS` has no
    translation in it for that reason -- what a search moves is the pitch and the
    field of view, which is how the workspace is brought into frame from a mount
    that cannot move. See `setups.FRANKA_STRETCHCAM_HEIGHT`.
    """

    yaw_deg: float = -90.0
    """Which way the camera faces in the mount body's frame, in degrees.

    -90 aims it along the body's +x -- forward, for both Stretch's base and the
    Franka's shoulder -- which is what every setup reproducing Stretch's head
    camera uses. Setup 1's DROID shoulder camera looks diagonally across the
    workspace instead, and needs its own value. Not a search dimension.
    """

    pitch_deg: float = 43.0
    """How far the camera is tilted up from straight down, in degrees.

    0 looks at the floor, 90 looks at the horizon, so **smaller points further
    down** -- the opposite way round from how "pitch" usually reads. The default
    43 comes out 47 degrees below horizontal, which is the tilt of Stretch's
    head camera.
    """

    roll_deg: float = 0.0
    """Rotation about the view axis, in degrees: how the camera is *mounted*.

    0 is an upright camera -- the horizon runs across the frame. `-90` is how
    Stretch's head cameras are actually bolted on, sideways, which is why every
    frame off the real robot is turned a quarter turn before anyone sees it (see
    `quarter_turns`).

    This is not a search dimension. It is a fact about the hardware, and it has
    to be right for two separate reasons: the fisheye calibration is expressed
    in the *sensor's* frame, so a warp applied to an upright render distorts
    along the wrong axes; and `quarter_turns` exists to undo this roll, so with
    the roll missing that quarter turn puts an already-upright frame on its
    side. Measured against the MJCF: `camera_right_link`'s optical frame is the
    upright basis rolled exactly -90 degrees.
    """

    fovy: float = 71.0
    """Vertical field of view in degrees, as MuJoCo means it.

    71 is the DROID exo camera's. The fisheye setups default to Stretch's own
    123.4, which is that camera's *horizontal* FOV stored under a vertical name
    -- a quirk of the calibration folded into one MuJoCo number, reproduced here
    because `Stretch4CameraSystem` reproduces it and the two agreeing is what
    makes a rendered frame match the simulator's.
    """

    render_size: tuple[int, int] = (640, 360)
    """`(width, height)` the pinhole render is taken at, before any warp.

    Sets the horizontal FOV as much as `fovy` does, since MuJoCo derives it from
    the viewport aspect. For a fisheye setup this has to keep the sensor's 1.6
    aspect or the calibration cannot be projected onto the frame -- see
    `StretchCameras.fisheye_params_for_frame`, which falls back to a centred
    approximation otherwise.
    """

    fisheye: str = FISHEYE_NONE
    """One of `FISHEYE_MODES`."""

    quarter_turns: int = 0
    """`np.rot90` turns applied after the warp. -1 for the real right head camera."""

    virtual_pitch_deg: float | None = None
    """Pitch to *synthesise* by moving the crop window, with the camera left where it is.

    A real head camera cannot be tilted -- it is bolted to the shell at
    `HEAD_CAMERA_ROLL_DEG` and `pitch_deg` 43, and no amount of software changes
    that. But a 123-degree fisheye sees far more than a 640x360 window needs, so
    a view that *looks* like it was taken at another pitch can be cut out of the
    frame the hardware already produces: shifting the crop window up the image
    is, to first order, the same as pointing the camera up.

    That is what this is for, and it is the only knob here that is deployable on
    the robot as built. `pitch_deg` moves the camera and answers "would a
    differently-mounted camera do better"; `virtual_pitch_deg` answers "can we
    get that from the camera we have". Requires `crop_to`, and only means
    anything once the frame has been rectified -- cropping a barrel-distorted
    frame off-centre shifts the distortion with it.

    None leaves the crop centred. See `_pitch_shifted_crop` for the geometry and
    for where the small-angle approximation gives out.
    """

    crop_to: tuple[int, int] | None = None
    """`(width, height)` to take back out of the frame, or None to leave it.

    A centre crop at the requested aspect followed by a resize, so it changes
    the field of view rather than the resolution alone. This is how a portrait
    fisheye frame is brought back to the 640x360 landscape the checkpoint was
    trained on.
    """

    def rotation(self) -> R:
        """The camera's orientation in the mount body's frame.

        An intrinsic ZXZ decomposition -- aim, then tilt, then roll about the
        resulting view axis -- chosen because the three stay independent:
        changing the tilt does not change which way is up in the frame, and
        changing the mount roll does not change where the camera points. It also
        covers all of SO(3), so setup 1's DROID shoulder camera is expressible in
        the same three numbers as the rest.

        The three cameras this study needs, decomposed:

            upright pinhole reconstruction   yaw -90.00  pitch 43.00  roll   0.00
            camera_right_link (hardware)     yaw -90.00  pitch 43.00  roll -90.00
            DROID shoulder camera            yaw -139.85 pitch 52.72  roll   7.69

        The first two differ in one number, and it is the mount roll -- which is
        exactly what `demo_droid_on_stretch.add_reconstructed_exo_camera` drops
        when it rebuilds an upright basis. All three were checked against their
        source: the first two against the compiled MJCF's own
        `camera_right_link` optical frame, the third against
        `FrankaDroidCameraSystem`'s quaternion.
        """
        return R.from_euler(
            "ZXZ", [self.yaw_deg, self.pitch_deg, self.roll_deg], degrees=True
        )

    def quat_wxyz(self) -> list[float]:
        """`rotation()` as the [w, x, y, z] quaternion MolmoSpaces wants."""
        return list(self.rotation().as_quat(scalar_first=True))

    def output_size(self) -> tuple[int, int]:
        """`(width, height)` of the frame that reaches the policy."""
        if self.crop_to is not None:
            return self.crop_to
        width, height = self.render_size
        return (height, width) if self.quarter_turns % 2 else (width, height)

    def describe(self) -> str:
        """One line for a log or a report row."""
        width, height = self.output_size()
        return (
            f"{self.mount_body} pos={np.round(self.pos, 4).tolist()} "
            f"pitch={self.pitch_deg:.1f} roll={self.roll_deg:+.0f} fovy={self.fovy:.1f} "
            f"render={self.render_size[0]}x{self.render_size[1]} "
            f"fisheye={self.fisheye} turns={self.quarter_turns} "
            + (
                f"vpitch={self.virtual_pitch_deg:.0f} "
                if self.virtual_pitch_deg is not None
                else ""
            )
            + 
            f"out={width}x{height}"
        )


@dataclass(frozen=True)
class RetargetParams:
    """One point in the search space: the exo camera, plus how the tool is retargeted.

    The second half only bites on the Stretch setups -- there is nothing to
    retarget when the policy is driving the Franka it was trained on -- and is
    ignored, not rejected, on the Franka ones, so that a single parameter vector
    describes a trial on either robot.
    """

    exo: ExoCameraParams = field(default_factory=ExoCameraParams)

    grasp_offset_m: float = 0.0
    """
    Metres to push the commanded grasp centre along Stretch's own approach axis.

    Stretch's gripper is much longer than the Robotiq the policy was trained
    with, so a tool pose that puts the Robotiq's pads around an object puts
    Stretch's palm there and its fingertips somewhere past it. Positive moves
    the commanded grasp centre *forward* along the approach (deeper); negative
    pulls it back towards the wrist.
    """

    wrist_tilt_deg: float = 0.0
    """
    Extra pitch, in degrees, between the Franka's tool frame and Stretch's.

    Applied about the tool frame's y axis on top of the fixed -90 degree
    correction the retargeting already carries, so 0 is today's behaviour and 45
    tips the gripper down by a further 45 degrees relative to what the policy
    asked for.
    """

    target_z_offset_m: float = 0.0
    """
    Metres to raise every commanded target by.

    `StretchMolmoBotDroidPolicyConfig.target_z_offset`, exposed because it trades
    grasp depth against clearance in the same way `grasp_offset_m` does and the
    two interact.

    Absolute metres rather than the fraction-of-a-measurement this used to be.
    The measurement it scaled -- `measure_tool_height_offset()` -- is taken at the
    Franka's home pose, which is above Stretch's lift ceiling, so it describes a
    pose the robot cannot reach and applied a correction to every target that
    mostly did not need one. See
    `StretchMolmoBotDroidPolicyConfig.target_z_offset`.
    """

    def describe(self) -> str:
        return (
            f"{self.exo.describe()} | grasp_offset={self.grasp_offset_m:+.3f}m "
            f"wrist_tilt={self.wrist_tilt_deg:+.1f}deg z_offset={self.target_z_offset_m:+.3f}m"
        )


# =============================================================================
# From parameters to a camera MolmoSpaces will place
# =============================================================================


def exo_camera_config(name: str, params: ExoCameraParams) -> RobotMountedCameraConfig:
    """The camera spec for `params`, ready to go in a `CameraSystemConfig`.

    `camera_quaternion` is what makes this a pose rather than a look-at: with it
    set, MolmoSpaces ignores `lookat_offset` entirely, which is what lets a
    pitch be searched over without also having to decide what the camera should
    be pointed at.
    """
    return RobotMountedCameraConfig(
        name=name,
        reference_body_names=[params.mount_body],
        camera_offset=list(params.pos),
        camera_quaternion=params.quat_wxyz(),
        fov=float(params.fovy),
    )


# =============================================================================
# From a render to what the policy is shown
# =============================================================================


def _fisheye_calibration(width: int, height: int) -> tuple[float, float, float, float, tuple]:
    """`(fx, fy, cx, cy, distortion)` for a frame of this size, from the real camera."""
    fx, fy, cx, cy, distortion, _ = FISHEYE_CAMERA.fisheye_params_for_frame(width, height)
    return fx, fy, cx, cy, tuple(distortion)


def _rectify(frame: np.ndarray) -> np.ndarray:
    """Undo the fisheye warp with the model that applied it.

    `cv2.fisheye.initUndistortRectifyMap` with `P = K` keeps the focal length,
    so the rectified frame covers the same angular window the distorted one did
    and the two are directly comparable. What it cannot do is invent the corners
    the warp threw away, which is the point of having this mode at all.
    """
    import cv2

    height, width = frame.shape[:2]
    fx, fy, cx, cy, distortion = _fisheye_calibration(width, height)
    camera_matrix = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    coefficients = np.asarray(distortion, dtype=np.float64).reshape(-1, 1)[:4]
    map_x, map_y = cv2.fisheye.initUndistortRectifyMap(
        camera_matrix, coefficients, np.eye(3), camera_matrix, (width, height), cv2.CV_32FC1
    )
    return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def _pitch_shift_pixels(frame_height: int, delta_deg: float) -> int:
    """How far to move the crop window to synthesise `delta_deg` of extra pitch.

    A rectified frame is a pinhole projection, so a point `delta` degrees off the
    optical axis lands `f * tan(delta)` pixels from the centre. Moving the window
    by that much therefore re-centres the view on a direction `delta` degrees
    away -- which is what tilting the camera would have done.

    `f` is the calibration's focal length carried onto this frame. The head
    camera is calibrated on a 1920x1200 sensor whose *long* axis becomes the
    frame's vertical after the quarter turn, so the focal length that governs
    vertical shifts is `fx`, scaled by how tall the rotated frame is.

    Positive `delta_deg` means "look further up", and the window moves up (toward
    row 0), because image rows increase downward.

    First-order only: `tan` is exact for the re-centring, but the rectification
    it rides on was fitted to the lens near its own axis, so a synthesised pitch
    tens of degrees out lands on the part of the fisheye the calibration
    describes worst. Twenty degrees is comfortable; sixty is not.
    """
    settings = FISHEYE_CAMERA.initial_camera_settings
    focal_x, _ = settings.focal
    focal = focal_x * (frame_height / float(settings.width))
    return int(round(-focal * np.tan(np.radians(delta_deg))))


def _centre_crop_resize(frame: np.ndarray, size: tuple[int, int], shift_y: int = 0) -> np.ndarray:
    """The largest centred window of `frame` with `size`'s aspect, resized to `size`.

    Cropping first and resizing second, rather than resizing straight to
    `size`, because the two are different views: a plain resize squashes a
    portrait frame into a landscape one and shows the policy a scene with the
    wrong proportions, which is not a thing any camera produces.

    `shift_y` moves the window off centre, clamped so it stays inside the frame
    -- a synthesised pitch that would run off the top of the image gives the
    most extreme view the lens actually captured rather than a band of black.
    """
    import cv2

    target_width, target_height = size
    height, width = frame.shape[:2]
    scale = min(width / target_width, height / target_height)
    crop_width = max(1, int(round(target_width * scale)))
    crop_height = max(1, int(round(target_height * scale)))
    left = (width - crop_width) // 2
    top = int(np.clip((height - crop_height) // 2 + shift_y, 0, height - crop_height))
    window = frame[top : top + crop_height, left : left + crop_width]
    if (crop_width, crop_height) == (target_width, target_height):
        return np.ascontiguousarray(window)
    interpolation = cv2.INTER_AREA if crop_width > target_width else cv2.INTER_LINEAR
    return np.ascontiguousarray(cv2.resize(window, size, interpolation=interpolation))


def postprocess_exo_frame(frame: np.ndarray, params: ExoCameraParams) -> np.ndarray:
    """Apply the four image-space stages, in the order the hardware applies them.

    Contiguous on the way out: MolmoSpaces hands rendered frames over as flipped
    views with negative strides, and MolmoBot's preprocessor puts them straight
    into `torch.from_numpy`, which refuses those.
    """
    if frame is None:
        return frame

    if params.fisheye != FISHEYE_NONE:
        from stretch4_mujoco import utils as stretch_utils

        height, width = frame.shape[:2]
        fx, fy, cx, cy, distortion = _fisheye_calibration(width, height)
        fov_deg = float(FISHEYE_CAMERA.initial_camera_settings.field_of_view_vertical_in_degrees)
        frame = stretch_utils.apply_fisheye_distortion(
            frame, fx, fy, cx, cy, distortion, fov_deg=fov_deg
        )
        if params.fisheye == FISHEYE_RECTIFIED:
            frame = _rectify(frame)

    if params.quarter_turns:
        frame = np.rot90(frame, params.quarter_turns)
    if params.crop_to is not None:
        # The shift is computed on the upright frame, because that is the one
        # whose vertical axis pitch moves.
        shift = 0
        if params.virtual_pitch_deg is not None:
            shift = _pitch_shift_pixels(
                frame.shape[0], float(params.virtual_pitch_deg) - float(params.pitch_deg)
            )
        frame = _centre_crop_resize(np.ascontiguousarray(frame), params.crop_to, shift)
    return np.ascontiguousarray(frame)


# =============================================================================
# Making MolmoSpaces run the above
# =============================================================================

_ACTIVE: dict[str, ExoCameraParams] = {}
"""Camera name -> what to do to its frames, for whatever this process is rendering.

Module state rather than something carried on the config because the render
happens in `CPUMujocoEnv.render_rgb_frame`, which is reached from
`CameraSensor.get_observation` -- MolmoSpaces builds those itself in
`get_core_sensors`, and there is no seam in the config to pass anything through.
The eval config sets this in `model_post_init`, which runs once per process,
including in a rollout worker that re-imports the module rather than inheriting
this dict.
"""


def register_exo_camera(name: str, params: ExoCameraParams) -> None:
    """Have this process post-process `name`'s frames according to `params`.

    Also publishes the render size to `stretch.config.CAMERA_RENDER_SIZE`, which
    is how `install_stretch_camera_hooks` decides what rectangle of the shared
    offscreen buffer a camera renders into -- and therefore the camera's
    horizontal field of view. A camera missing from that map is rendered into
    the whole buffer at the buffer's aspect, which for a 640x400 fisheye in a
    656x400 buffer is a subtly wider view than the hardware has.
    """
    from examples.machine_learning.molmospaces.stretch import config as stretch_config

    _ACTIVE[name] = params
    stretch_config.CAMERA_RENDER_SIZE[name] = tuple(params.render_size)
    install_exo_camera_hook()


def clear_exo_cameras() -> None:
    """Forget every registration. Between trials in one process."""
    _ACTIVE.clear()


def install_exo_camera_hook() -> None:
    """Post-process registered cameras on their way out of the renderer.

    Layered *outside* `install_stretch_camera_hooks`, deliberately and not
    incidentally: that hook is what sizes the render rectangle to the camera's
    aspect and puts the headlight back on the camera, and both have to have
    happened before there is a frame worth warping. It leaves frames from
    cameras it does not recognise -- which is all of these -- untouched, so the
    two compose rather than fight.

    Idempotent: this module is imported once per worker process and again when
    an eval config is resolved from its "module:Class" string.
    """
    from molmo_spaces.env.env import CPUMujocoEnv

    if getattr(CPUMujocoEnv, "_retarget_exo_hooked", False):
        return

    original_render_rgb_frame = CPUMujocoEnv.render_rgb_frame

    def render_rgb_frame(self: Any, camera_name: str) -> Any:
        frame = original_render_rgb_frame(self, camera_name)
        params = _ACTIVE.get(camera_name)
        if params is None:
            return frame
        try:
            return postprocess_exo_frame(frame, params)
        except Exception as error:  # noqa: BLE001 - a broken warp must not sink the rollout
            log.warning(f"[exo] post-processing {camera_name} failed: {error}")
            return frame

    CPUMujocoEnv.render_rgb_frame = render_rgb_frame
    CPUMujocoEnv._retarget_exo_hooked = True


__all__ = [
    "ExoCameraParams",
    "RetargetParams",
    "FISHEYE_NONE",
    "FISHEYE_DISTORTED",
    "FISHEYE_RECTIFIED",
    "FISHEYE_MODES",
    "exo_camera_config",
    "postprocess_exo_frame",
    "register_exo_camera",
    "clear_exo_cameras",
    "install_exo_camera_hook",
]
