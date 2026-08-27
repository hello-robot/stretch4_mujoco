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
import shutil
import tempfile
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from mujoco import MjData

from examples.machine_learning.molmospaces.stretch.robot import Stretch4Robot
from examples.machine_learning.molmospaces.stretch.robot_view import Stretch4RobotView
from molmo_spaces.configs.camera_configs import CameraSystemConfig, MjcfCameraConfig
from molmo_spaces.configs.robot_configs import BaseRobotConfig
from molmo_spaces.robots.abstract import Robot
from molmo_spaces.robots.robot_views.abstract import RobotViewFactory

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


def install_fisheye_distortion_hook() -> None:
    """Hook CPUMujocoEnv.render_rgb_frame and render_depth_frame to apply distortion and rotation to nav cameras."""
    try:
        from molmo_spaces.env.env import CPUMujocoEnv
    except ImportError:
        return

    if getattr(CPUMujocoEnv, "_stretch_camera_hooked", False):
        return

    _orig_render_rgb_frame = CPUMujocoEnv.render_rgb_frame
    _orig_render_depth_frame = CPUMujocoEnv.render_depth_frame

    def _postprocess_camera_frame(camera_name: str, frame: np.ndarray, is_depth: bool = False) -> np.ndarray:
        if frame is None:
            return frame
        try:
            from stretch4_mujoco.enums.stretch_cameras import StretchCameras

            if camera_name in (HEAD_CAMERA_LEFT, HEAD_CAMERA_LEFT_MJCF_NAME, "cam_nav_rgb_se4_left"):
                if not is_depth:
                    cb = StretchCameras.cam_nav_rgb_se4_left.post_processing_callback
                    if cb is not None:
                        frame = cb(frame)
                rot = StretchCameras.cam_nav_rgb_se4_left.initial_camera_settings.rotate_number_of_times
                if rot != 0:
                    frame = np.rot90(frame, rot)
                return frame
            elif camera_name in (HEAD_CAMERA_RIGHT, HEAD_CAMERA_RIGHT_MJCF_NAME, "cam_nav_rgb_se4_right"):
                if not is_depth:
                    cb = StretchCameras.cam_nav_rgb_se4_right.post_processing_callback
                    if cb is not None:
                        frame = cb(frame)
                rot = StretchCameras.cam_nav_rgb_se4_right.initial_camera_settings.rotate_number_of_times
                if rot != 0:
                    frame = np.rot90(frame, rot)
                return frame
            elif camera_name in (HEAD_CAMERA, HEAD_CAMERA_MJCF_NAME, "cam_nav_rgb_se4_center", "cam_nav_rgb_se4_center_low_rez"):
                rot = StretchCameras.cam_nav_rgb_se4_center.initial_camera_settings.rotate_number_of_times
                if rot != 0:
                    frame = np.rot90(frame, rot)
                return frame
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Error applying camera post-processing to {camera_name}: {e}")
        return frame

    def _distorted_render_rgb_frame(self: Any, camera_name: str) -> Any:
        frame = _orig_render_rgb_frame(self, camera_name)
        return _postprocess_camera_frame(camera_name, frame, is_depth=False)

    def _distorted_render_depth_frame(self: Any, camera_name: str) -> Any:
        frame = _orig_render_depth_frame(self, camera_name)
        return _postprocess_camera_frame(camera_name, frame, is_depth=True)

    CPUMujocoEnv.render_rgb_frame = _distorted_render_rgb_frame
    CPUMujocoEnv.render_depth_frame = _distorted_render_depth_frame
    CPUMujocoEnv._stretch_camera_hooked = True
    CPUMujocoEnv._stretch_fisheye_hooked = True


install_fisheye_distortion_hook()


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
    """

    img_resolution: tuple[int, int] = (640, 368)
    cameras: list[Any] = [
        MjcfCameraConfig(
            name=HEAD_CAMERA,
            mjcf_name=HEAD_CAMERA_MJCF_NAME,
            robot_namespace="robot_0/",
            fov=None,
        ),
        MjcfCameraConfig(
            name=WRIST_CAMERA_LEFT,
            mjcf_name=WRIST_LEFT_CAMERA_MJCF_NAME,
            robot_namespace="robot_0/",
            fov=None,
        ),
        MjcfCameraConfig(
            name=WRIST_CAMERA_RIGHT,
            mjcf_name=WRIST_RIGHT_CAMERA_MJCF_NAME,
            robot_namespace="robot_0/",
            fov=None,
        ),
        MjcfCameraConfig(
            name=WRIST_CAMERA_STEREO,
            mjcf_name=WRIST_STEREO_CAMERA_MJCF_NAME,
            robot_namespace="robot_0/",
            fov=None,
            record_rgb=False,
            record_depth=True,
        ),
        MjcfCameraConfig(
            name=HEAD_CAMERA_LEFT,
            mjcf_name=HEAD_CAMERA_LEFT_MJCF_NAME,
            robot_namespace="robot_0/",
            fov=123.39,
        ),
        MjcfCameraConfig(
            name=HEAD_CAMERA_RIGHT,
            mjcf_name=HEAD_CAMERA_RIGHT_MJCF_NAME,
            robot_namespace="robot_0/",
            fov=123.20,
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
        "lift": [0.35],
        "arm": [0.0],
        "wrist": [3.14, -0.4, 0.0],
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
