"""
MolmoSpaces configuration objects for Stretch 4: robot config and camera system.

`Stretch4RobotConfig` plays the same role for Stretch that
`molmo_spaces.configs.robot_configs.MobileFrankaRobotConfig` plays for the mobile
Franka: it tells MolmoSpaces which `Robot` class to instantiate, which
`RobotView` to build, where the MJCF lives, and what pose to reset to.

The MJCF itself is not a prepackaged MolmoSpaces asset -- it is generated on
demand from the Stretch URDF by `models/stretch_4/mjcf_generator.py` -- so the
config points `robot_dir`/`robot_xml_path` at whatever
`Stretch4MujocoSimulator.get_robot_xml_path()` produces instead of going through
`molmo_spaces_constants.get_robot_path()`.
"""

import atexit
import fcntl
import functools
import logging
import math
import shutil
import tempfile
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from mujoco import MjData

from examples.machine_learning.molmospaces.hdf5_layout import (
    MACROBLOCK,
    camera_output_size,
    camera_render_size,
)
from examples.machine_learning.molmospaces.stretch.robot import Stretch4Robot
from examples.machine_learning.molmospaces.stretch.robot_view import Stretch4RobotView
from molmo_spaces.configs.camera_configs import CameraSystemConfig, MjcfCameraConfig
from molmo_spaces.configs.robot_configs import BaseRobotConfig
from molmo_spaces.robots.abstract import Robot
from molmo_spaces.robots.robot_views.abstract import RobotViewFactory
from stretch4_mujoco.enums.stretch_cameras import StretchCameras

# Names of the Stretch MJCF cameras this integration exposes to policies.
HEAD_CAMERA = "head_camera"
WRIST_CAMERA_LEFT = "wrist_camera_left"
WRIST_CAMERA_RIGHT = "wrist_camera_right"
WRIST_CAMERA_STEREO = "wrist_camera_stereo"
HEAD_CAMERA_LEFT = "head_camera_left"
HEAD_CAMERA_RIGHT = "head_camera_right"
CHASE_CAMERA = "chase_camera"
TRACKER_CAMERA = "tracker_camera"

# The corresponding camera elements in the generated MJCF. Stretch 4's head is a
# fixed stereo + centre assembly (there is no pan/tilt joint in the SE4 URDF), so
# the head camera is a forward-and-down view rigidly tied to the base yaw; the
# wrist cameras are mounted on the gripper itself and follow the arm/wrist.
HEAD_CAMERA_MJCF_NAME = "camera_center_link"
WRIST_LEFT_CAMERA_MJCF_NAME = "gripper_camera_left_rgb"
WRIST_RIGHT_CAMERA_MJCF_NAME = "gripper_camera_right_rgb"
WRIST_STEREO_CAMERA_MJCF_NAME = "gripper_camera_stereo_depth"
HEAD_CAMERA_LEFT_MJCF_NAME = "camera_left_link"
HEAD_CAMERA_RIGHT_MJCF_NAME = "camera_right_link"
CHASE_CAMERA_MJCF_NAME = "chase_camera"

MJCF_NAME_FOR_CAMERA: dict[str, str] = {
    HEAD_CAMERA: HEAD_CAMERA_MJCF_NAME,
    HEAD_CAMERA_LEFT: HEAD_CAMERA_LEFT_MJCF_NAME,
    HEAD_CAMERA_RIGHT: HEAD_CAMERA_RIGHT_MJCF_NAME,
    WRIST_CAMERA_LEFT: WRIST_LEFT_CAMERA_MJCF_NAME,
    WRIST_CAMERA_RIGHT: WRIST_RIGHT_CAMERA_MJCF_NAME,
    WRIST_CAMERA_STEREO: WRIST_STEREO_CAMERA_MJCF_NAME,
}


def stretch_camera_for_mjcf_name(mjcf_name: str) -> StretchCameras:
    """The `StretchCameras` member that renders a given MJCF camera.

    Derived rather than hand-written, because the two stacks name the same
    physical camera differently and getting the pairing wrong is invisible:
    feeding a policy trained on `gripper_camera_left_rgb` the *right* gripper
    camera produces plausible-looking images from 2cm away and a policy that
    quietly does worse.

    Several members can share an MJCF camera -- the head centre camera has a
    full-resolution and a low-resolution member -- so prefer the one Stretch 4
    exposes outwardly. `Stretch4MujocoSimulator` swaps in the low-resolution
    variant internally and reports frames back under the outward name.
    """
    matches = [
        camera
        for camera in StretchCameras.all_stretch4()
        if camera.camera_name_in_mjcf == mjcf_name
    ]
    if not matches:
        raise ValueError(
            f"No StretchCameras member renders MJCF camera {mjcf_name!r}; "
            f"available: {sorted(c.camera_name_in_mjcf for c in StretchCameras.all_stretch4())}"
        )
    return matches[0]


STRETCH_CAMERA_FOR_CAMERA: dict[str, StretchCameras] = {
    name: stretch_camera_for_mjcf_name(mjcf_name)
    for name, mjcf_name in MJCF_NAME_FOR_CAMERA.items()
}
"""MolmoSpaces camera name -> the `StretchCameras` member that describes it.

`StretchCameras.initial_camera_settings` is the single source of truth for what
each Stretch camera sees, and both stacks have to read it or they render
different views of the same scene. `Stretch4MujocoSimulator` reads it directly
(`MujocoServerCameraManagerSync.set_camera_params` writes `cam_fovy` and sizes
the offscreen renderer from it); MolmoSpaces cannot, because it renders a *free*
camera whose vertical FOV comes from `MjcfCameraConfig.fov` and whose horizontal
FOV falls out of the render viewport's aspect ratio. So both numbers are carried
across below.
"""


def _camera_fov(camera_name: str) -> float:
    """The vertical FOV MuJoCo needs for a camera, in degrees.

    Note that for the two head fisheyes this is `2*atan(width / (2*fx))`
    (`stretch_cameras.py`), which is the *horizontal* FOV of the 1920-wide
    sensor under a name that says vertical. That is a quirk of the calibration
    being folded into a single MuJoCo number, and it is deliberately reproduced
    rather than corrected here: the simulator feeds the same value to `cam_fovy`,
    and the two stacks agreeing is the point.
    """
    settings = STRETCH_CAMERA_FOR_CAMERA[camera_name].initial_camera_settings
    return float(settings.field_of_view_vertical_in_degrees)


CAMERA_RENDER_SIZE: dict[str, tuple[int, int]] = {
    name: camera_render_size(camera) for name, camera in STRETCH_CAMERA_FOR_CAMERA.items()
}
"""Camera name -> (width, height) to render at, before rotation. See `hdf5_layout`."""

CAMERA_OUTPUT_SIZE: dict[str, tuple[int, int]] = {
    name: camera_output_size(camera) for name, camera in STRETCH_CAMERA_FOR_CAMERA.items()
}
"""Camera name -> (width, height) of the frame that leaves the pipeline."""

RENDER_BUFFER_RESOLUTION: tuple[int, int] = (
    max(size[0] for size in CAMERA_RENDER_SIZE.values()),
    max(size[1] for size in CAMERA_RENDER_SIZE.values()),
)
"""The offscreen buffer every camera renders into, as (width, height).

MolmoSpaces builds one renderer per environment from
`camera_config.img_resolution` and renders every camera through it, so this is
not an image size but an upper bound: it has to enclose every entry of
`CAMERA_RENDER_SIZE`, and each frame is then rendered into the sub-rectangle of
that buffer that matches its own camera's aspect ratio.
"""


def install_stretch_camera_hooks() -> None:
    """Make MolmoSpaces render Stretch's cameras the way the simulator does.

    MolmoSpaces renders every camera through one renderer at one resolution,
    with a free camera whose vertical FOV is `MjcfCameraConfig.fov`. Left alone
    that produces a different view from `Stretch4MujocoSimulator`, which renders
    through the MJCF camera at `initial_camera_settings.width x height` with
    `cam_fovy` set from the same settings. The FOV half of the gap is closed by
    `Stretch4CameraSystem` passing real FOVs; this closes the rest:

    * each frame is rendered into the sub-rectangle of the shared buffer that
      has its camera's aspect ratio, so the horizontal FOV matches too;
    * the headlight follows the camera, which MolmoSpaces' aim-the-camera-after-
      updating-the-scene order leaves behind -- see
      `_render_with_camera_headlight`, and note that this one is not a Stretch
      concern at all, it is every MolmoSpaces frame of every robot;
    * the head cameras get the fisheye warp and the quarter turn the simulator
      applies in `_render_camera` and `StatusStretchCameras.get_camera_data`;
    * the wrist depth camera gets the same `depth_limits["gripper"]` clip;
    * the recorded intrinsics describe the frame that comes out, rather than a
      45-degree pinhole at the shared buffer's resolution.

    With all of them in place a MolmoSpaces frame is pixel-identical to the
    simulator's, which is what `test_datagen_render_matches_the_simulators_camera`
    asserts.

    Patching rather than subclassing because the render path is reached from
    `CameraSensor.get_observation`, which MolmoSpaces' `get_core_sensors` builds
    itself -- there is no seam in the config to pass a different env class
    through.
    """
    try:
        from molmo_spaces.env.env import CPUMujocoEnv
        from molmo_spaces.env.sensors_cameras import CameraParameterSensor
    except ImportError:
        return

    if getattr(CPUMujocoEnv, "_stretch_camera_hooked", False):
        return

    _orig_render_rgb_frame = CPUMujocoEnv.render_rgb_frame
    _orig_render_depth_frame = CPUMujocoEnv.render_depth_frame
    _orig_camera_parameters = CameraParameterSensor.get_observation
    _orig_initialize_with_model = CPUMujocoEnv._initialize_with_model

    def _stretch_initialize_with_model(self: Any, mj_model: Any, *args: Any, **kwargs: Any) -> Any:
        """Guarantee the offscreen framebuffer can hold `RENDER_BUFFER_RESOLUTION`.

        A scene's `<visual><global offwidth=.../>` is whatever its author wrote,
        and MuJoCo refuses to build a renderer larger than it -- as a hard error
        at environment construction, before any of the per-camera sizing below
        gets a say. The default 640x480 happens to fit, but a scene that declares
        something smaller would otherwise take the whole run down.
        """
        width, height = RENDER_BUFFER_RESOLUTION
        mj_model.vis.global_.offwidth = max(int(mj_model.vis.global_.offwidth), width)
        mj_model.vis.global_.offheight = max(int(mj_model.vis.global_.offheight), height)
        return _orig_initialize_with_model(self, mj_model, *args, **kwargs)

    def _fit_to_buffer(size: tuple[int, int], renderer: Any) -> tuple[int, int]:
        """`size`, shrunk to fit the renderer's buffer without changing aspect.

        Only bites when an environment was built with a smaller
        `img_resolution` than `RENDER_BUFFER_RESOLUTION` -- a benchmark episode
        recorded before per-camera sizing, say. Dropping resolution keeps the
        view; letting the rectangle overrun the offscreen buffer would not.
        """
        width, height = size
        max_width = getattr(renderer, "width", None)
        max_height = getattr(renderer, "height", None)
        if not max_width or not max_height:
            return size
        if width <= max_width and height <= max_height:
            return size
        scale = min(max_width / width, max_height / height)
        # Floored to macroblocks rather than rounded: rounding up is what would
        # overrun the buffer this is here to stay inside.
        return (
            max(MACROBLOCK, int(width * scale) // MACROBLOCK * MACROBLOCK),
            max(MACROBLOCK, int(height * scale) // MACROBLOCK * MACROBLOCK),
        )

    def _render_with_camera_headlight(renderer: Any, **kwargs: Any) -> Any:
        """Put the headlight back on the camera, then render.

        MolmoSpaces aims its free camera by calling `mjv_updateScene` with a
        default `MjvCamera` and *then* overwriting `scene.camera[i].pos/forward/up`.
        The geometry comes out right, but `mjv_updateScene` has already placed
        the headlight -- `scene.lights[0]`, which MuJoCo pins to the camera for a
        fixed camera -- at the default viewpoint it was handed, several metres
        away from where the frame is actually taken. Every frame is then lit from
        somewhere the Stretch camera is not, which is most of the difference
        between a MolmoSpaces frame and a `Stretch4MujocoSimulator` one: 57% of
        the pixels of a wrist view, on a scene whose geometry aligns to
        sub-pixel.

        The camera is set by the time `render()` is called, so this is where the
        light can follow it.
        """
        scene = renderer.scene
        if scene.nlight > 0 and scene.lights[0].headlight:
            scene.lights[0].pos = scene.camera[0].pos
            scene.lights[0].dir = scene.camera[0].forward
        return type(renderer).render(renderer, **kwargs)

    def _render_at_camera_aspect(env: Any, camera_name: str, render_fn: Callable) -> Any:
        """Run `render_fn` with the shared renderer clamped to this camera's rect."""
        renderer = getattr(env, "_renderer", None)
        if renderer is None:
            return render_fn(env, camera_name)

        size = CAMERA_RENDER_SIZE.get(camera_name)
        if size is None:
            rect: dict[str, int] = {}  # an unknown camera keeps the whole buffer
        else:
            width, height = _fit_to_buffer(size, renderer)
            rect = {"width": width, "height": height}
        # An instance attribute shadows the bound method for the duration of the
        # call, which is the only seam MolmoSpaces' `_render_frame` leaves: it
        # calls `self._renderer.render()` with no arguments.
        renderer.render = functools.partial(_render_with_camera_headlight, renderer, **rect)
        try:
            return render_fn(env, camera_name)
        finally:
            renderer.__dict__.pop("render", None)

    def _postprocess_camera_frame(
        camera_name: str, frame: np.ndarray, is_depth: bool = False
    ) -> np.ndarray:
        """Apply what the simulator applies after `renderer.render()`.

        The simulator splits this across two places -- distortion and depth
        clipping in `MujocoServerCameraManagerSync._render_camera`, the rotation
        in `StatusStretchCameras.get_camera_data(auto_rotate=True)` -- but every
        consumer sees both, so both belong here.
        """
        if frame is None:
            return frame
        camera = STRETCH_CAMERA_FOR_CAMERA.get(camera_name)
        if camera is None:
            return frame
        try:
            settings = camera.initial_camera_settings
            post_processing = camera.post_processing_callback
            if post_processing is not None and camera.is_depth == is_depth:
                frame = post_processing(frame)
            if settings.rotate_number_of_times != 0:
                frame = np.rot90(frame, settings.rotate_number_of_times)
        except Exception as e:
            logging.getLogger(__name__).warning(
                f"Error applying camera post-processing to {camera_name}: {e}"
            )
        return frame

    def _stretch_render_rgb_frame(self: Any, camera_name: str) -> Any:
        frame = _render_at_camera_aspect(self, camera_name, _orig_render_rgb_frame)
        return _postprocess_camera_frame(camera_name, frame, is_depth=False)

    def _stretch_render_depth_frame(self: Any, camera_name: str) -> Any:
        frame = _render_at_camera_aspect(self, camera_name, _orig_render_depth_frame)
        return _postprocess_camera_frame(camera_name, frame, is_depth=True)

    def _stretch_camera_parameters(
        self: Any, env: Any, task: Any, batch_index: int = 0, *args: Any, **kwargs: Any
    ) -> dict:
        data = _orig_camera_parameters(self, env, task, batch_index, *args, **kwargs)
        camera = STRETCH_CAMERA_FOR_CAMERA.get(self.camera_name)
        size = CAMERA_RENDER_SIZE.get(self.camera_name)
        if camera is None or size is None or not isinstance(data, dict):
            return data

        # MolmoSpaces builds K from `self.img_resolution`, which is the shared
        # buffer rather than this camera's rectangle, and does not know the
        # frame is about to be rotated. Both are known here.
        width, height = size
        fov = env.camera_manager.registry[self.camera_name].fov
        focal = (height / 2.0) / math.tan(math.radians(fov / 2.0))
        cx, cy = width / 2.0, height / 2.0

        if camera.applies_fisheye_distortion:
            # The fisheye distortion crops away the surround the pinhole render
            # cannot fill and rescales what is left back to `size`, so the frame
            # this K describes is zoomed in on a window of the raw render.
            crop_x, crop_y, crop_width, crop_height = camera.fisheye_crop_rect(width, height)
            zoom = width / float(crop_width)
            focal *= zoom
            cx = (cx - crop_x) * width / float(crop_width)
            cy = (cy - crop_y) * height / float(crop_height)

        quarter_turns = camera.initial_camera_settings.rotate_number_of_times % 4
        if quarter_turns == 1:  # np.rot90(frame, 1): (r, c) -> (W-1-c, r)
            cx, cy = cy, (width - 1) - cx
        elif quarter_turns == 3:  # np.rot90(frame, -1): (r, c) -> (c, H-1-r)
            cx, cy = (height - 1) - cy, cx

        data["intrinsic_cv"] = [
            [focal, 0.0, cx],
            [0.0, focal, cy],
            [0.0, 0.0, 1.0],
        ]
        return data

    CPUMujocoEnv._initialize_with_model = _stretch_initialize_with_model
    CPUMujocoEnv.render_rgb_frame = _stretch_render_rgb_frame
    CPUMujocoEnv.render_depth_frame = _stretch_render_depth_frame
    CameraParameterSensor.get_observation = _stretch_camera_parameters
    CPUMujocoEnv._stretch_camera_hooked = True


install_stretch_camera_hooks()


@lru_cache(maxsize=1)
def default_stretch_robot_xml_path() -> Path:
    """Absolute path to a private copy of the generated Stretch 4 MJCF.

    `Stretch4MujocoSimulator.get_robot_xml_path()` regenerates the MJCF from the
    URDF and always writes it to the same file,
    `stretch4_mujoco/models/stretch_temp_abs.xml`. That is fine for a single
    process and a race for `run_benchmarks.py --num-workers 8`, where every
    worker imports this module and regenerates that one file while its siblings
    may be reading it.

    So generation happens under an exclusive file lock, and the result is
    immediately copied somewhere only this process will touch. The copy is safe
    to relocate because the generated XML refers to its `<include>`s by absolute
    path.
    """
    from stretch4_mujoco.stretch4_mujoco_simulator import Stretch4MujocoSimulator

    private_dir = Path(tempfile.mkdtemp(prefix="stretch4_molmospaces_"))
    atexit.register(shutil.rmtree, private_dir, True)
    private_path = private_dir / "stretch4.xml"

    lock_path = Path(tempfile.gettempdir()) / "stretch4_mjcf_generation.lock"
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            shutil.copy(Stretch4MujocoSimulator.get_robot_xml_path(), private_path)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
    return private_path


class Stretch4CameraSystem(CameraSystemConfig):
    """Head + wrist + left/right fisheye views from the Stretch MJCF.

    Using `MjcfCameraConfig` rather than `RobotMountedCameraConfig` means the
    camera extrinsics come from the robot model itself, so a simulated view lines
    up with what the corresponding camera sees on hardware.

    The intrinsics have to be carried across by hand, because MolmoSpaces reads
    only `fov` off these specs and MuJoCo's own default (45 degrees, which is what
    a `<camera>` with no `fovy` attribute gets, and `mjcf_generator.py` writes
    none) is not any Stretch camera. Every FOV below is
    `StretchCameras.<member>.initial_camera_settings.field_of_view_vertical_in_degrees`
    -- the same number `MujocoServerCameraManagerSync.set_camera_params` writes
    into `cam_fovy` -- so that the datagen render, the Rerun feed and the
    simulator all frame the scene identically. `install_stretch_camera_hooks`
    does the other half, matching each camera's aspect ratio.
    """

    img_resolution: tuple[int, int] = RENDER_BUFFER_RESOLUTION
    cameras: list[Any] = [
        MjcfCameraConfig(
            name=HEAD_CAMERA,
            mjcf_name=HEAD_CAMERA_MJCF_NAME,
            robot_namespace="robot_0/",
            fov=_camera_fov(HEAD_CAMERA),
        ),
        MjcfCameraConfig(
            name=WRIST_CAMERA_LEFT,
            mjcf_name=WRIST_LEFT_CAMERA_MJCF_NAME,
            robot_namespace="robot_0/",
            fov=_camera_fov(WRIST_CAMERA_LEFT),
        ),
        MjcfCameraConfig(
            name=WRIST_CAMERA_RIGHT,
            mjcf_name=WRIST_RIGHT_CAMERA_MJCF_NAME,
            robot_namespace="robot_0/",
            fov=_camera_fov(WRIST_CAMERA_RIGHT),
        ),
        MjcfCameraConfig(
            name=WRIST_CAMERA_STEREO,
            mjcf_name=WRIST_STEREO_CAMERA_MJCF_NAME,
            robot_namespace="robot_0/",
            fov=_camera_fov(WRIST_CAMERA_STEREO),
            record_rgb=False,
            record_depth=True,
        ),
        MjcfCameraConfig(
            name=HEAD_CAMERA_LEFT,
            mjcf_name=HEAD_CAMERA_LEFT_MJCF_NAME,
            robot_namespace="robot_0/",
            fov=_camera_fov(HEAD_CAMERA_LEFT),
        ),
        MjcfCameraConfig(
            name=HEAD_CAMERA_RIGHT,
            mjcf_name=HEAD_CAMERA_RIGHT_MJCF_NAME,
            robot_namespace="robot_0/",
            fov=_camera_fov(HEAD_CAMERA_RIGHT),
        ),
    ]


class Stretch4RobotConfig(BaseRobotConfig):
    """Robot config for Stretch 4 on a virtual holonomic base."""

    robot_cls: type[Robot] | None = Stretch4Robot
    robot_factory: Callable[[MjData, Any], Robot] | None = Stretch4Robot
    robot_view_factory: RobotViewFactory | None = Stretch4RobotView
    robot_namespace: str = "robot_0/"

    name: str = "stretch4"
    # Filled in by `model_post_init` from `default_stretch_robot_xml_path()`; the
    # generated MJCF's location is not known until the URDF has been converted.
    robot_dir: Path | None = None
    robot_xml_path: Path = Path("stretch_temp_abs.xml")

    # base: (x, y, theta); lift: mast height; arm: total telescoping extension;
    # wrist: (yaw, pitch, roll); gripper: (right finger, left finger).
    init_qpos: dict[str, list[float]] = {
        # This is the "stow" pose in `keyframes.xml`
        # with base x moved 1m away so the robot spawns a little far away from the workspace.
        "base": [0.0, 0.0, 0.0],
        "lift": [1.0],
        "arm": [0.0],
        "wrist": [0, 0, 0.0],
        "gripper": [0.0, 0.0],
    }
    init_qpos_noise_range: dict[str, list[float]] | None = None

    command_mode: dict[str, str] = {
        "base": "holo_joint_planar_position",
        "lift": "joint_position",
        "arm": "joint_position",
        "wrist": "joint_position",
        "gripper": "joint_position",
    }

    # Gravity compensation is what makes the lift and the telescoping arm
    # position-controllable at these gains: without it the mast has to hold ~40kg
    # of arm and payload against gravity out of position error alone.
    gravcomp: bool = True

    # Gains for the three virtual base actuators. `kd` is set for critical
    # damping of the robot's ~106kg compiled mass (kd = 2*sqrt(kp*m)); the yaw
    # actuator is tuned against the base's z-axis inertia instead.
    base_control_params: dict[str, dict[str, float]] = {
        "base_x_act": {"kp": 25000.0, "kd": 3250.0, "ctrlrange": 25.0},
        "base_y_act": {"kp": 25000.0, "kd": 3250.0, "ctrlrange": 25.0},
        # The yaw control range is deliberately enormous rather than +-pi.
        # `HoloJointsRobotBaseGroup.ctrl` unwraps a commanded yaw to the
        # revolution the base is currently on before writing it, and the hinge
        # joint itself is unlimited, so a commanded target can legitimately sit
        # many radians from zero -- while `JointPosController` clips targets to
        # exactly this range. A tight range would quietly cap turns.
        "base_theta_act": {"kp": 5000.0, "kd": 350.0, "ctrlrange": 1000.0},
    }

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        if self.robot_dir is None:
            xml_path = default_stretch_robot_xml_path()
            self.robot_dir = xml_path.parent
            self.robot_xml_path = Path(xml_path.name)
        assert (
            self.command_mode["gripper"] == "joint_position"
        ), "Relative finger commands cannot hold a grasp; keep the gripper on absolute position."
