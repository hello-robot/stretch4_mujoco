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

from mujoco import MjData

from examples.machine_learning.molmospaces.stretch.robot import Stretch4Robot
from examples.machine_learning.molmospaces.stretch.robot_view import Stretch4RobotView
from molmo_spaces.configs.camera_configs import CameraSystemConfig, MjcfCameraConfig
from molmo_spaces.configs.robot_configs import BaseRobotConfig
from molmo_spaces.robots.abstract import Robot
from molmo_spaces.robots.robot_views.abstract import RobotViewFactory

# Names of the two Stretch MJCF cameras this integration exposes to policies.
# The benchmark episodes name Franka cameras ("wrist_camera", "exo_camera_1",
# ...), which is one of the things `episode_overrides.py` rewrites.
HEAD_CAMERA = "head_camera"
WRIST_CAMERA = "wrist_camera"

# The corresponding camera elements in the generated MJCF. Stretch 4's head is a
# fixed stereo + centre assembly (there is no pan/tilt joint in the SE4 URDF), so
# the head camera is a forward-and-down view rigidly tied to the base yaw; the
# gripper camera rides the wrist and provides the close-up view.
HEAD_CAMERA_MJCF_NAME = "camera_center_link"
WRIST_CAMERA_MJCF_NAME = "gripper_camera_left_rgb"


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
    """Head + wrist views, taken straight from the cameras in the Stretch MJCF.

    Using `MjcfCameraConfig` rather than `RobotMountedCameraConfig` means the
    camera extrinsics come from the robot model itself, so a simulated view lines
    up with what the corresponding camera sees on hardware. `fov=None` makes the
    camera manager fall back to the MJCF's own `fovy`.
    """

    img_resolution: tuple[int, int] = (224, 224)
    cameras: list[Any] = [
        MjcfCameraConfig(
            name=HEAD_CAMERA,
            mjcf_name=HEAD_CAMERA_MJCF_NAME,
            robot_namespace="robot_0/",
            fov=None,
        ),
        MjcfCameraConfig(
            name=WRIST_CAMERA,
            mjcf_name=WRIST_CAMERA_MJCF_NAME,
            robot_namespace="robot_0/",
            fov=None,
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
