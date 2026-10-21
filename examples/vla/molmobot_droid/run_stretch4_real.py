"""
MolmoBot-DROID driving a real Stretch 4, through Franka retargeting, continuously: type an
instruction and press Enter, and the robot works on it until told otherwise (see KEYS_HELP).
It asks before it first moves the robot to the start pose.

Rerun shows the robot's cameras and pose live from the moment they arrive, and, while an
instruction runs, the policy's views and the ghost Franka it is retargeted from.

On the robot, run the image + joint-state publisher from stretch4_compliant_gripper first:

    python send_gripper_and_head_images_with_joint_states.py -r --head_camera left,right --wrist_camera left --wrist_resolution 400

with both head fisheyes, and its wrist camera side matching --gripper_camera. The policy's exo
view is the head fisheye on the side the tool is toward (see `head_camera_for_tool()`), so
--exo_camera is not used. Commands go
through stretch4_body's `RobotClient`; with --wait-for-arrival (the default) each action waits
for the robot to stop moving.

There is no scene to read the target's height from, so give it with --object-height: the
virtual Franka stands where molmospaces would put it for an object at that height. Or, with
--custom_franka_start_pose, at the height that starts Stretch's tool at the top of its reach.

Between instructions, `jog` moves the virtual Franka's joints and pedestal from the keys, and
the robot follows (see `StretchStartPoseEditor` and KEYS_HELP); `home` or `reset` goes back to the
start pose; `offset x y z [roll yaw pitch]` changes the grasp offset (mm, deg).

Where Stretch's wrist cannot turn the tool as far as the Franka's grasp needs, it goes there
turned less (see `FrankaStretchRetargeter.franka_to_stretch`); an action it cannot follow at all
leaves it where it is, and --recover-after of those in a row send it back to the start pose. Both
are printed, and logged to Rerun under RETARGET_LOG.

With --rectify, the exo view is the head fisheye redrawn as the DROID exo camera would see it
(`FisheyeRectifier`), through the lens calibration the publisher sends with the images when it
has one (see `head_rectifiers()`), instead of only cropped to DROID's aspect.

With --overlay_franka_gripper, the policy's views show the Franka where it is told its arm and
hand are, and Stretch painted out: the Robotiq's fingers for Stretch's gripper in the wrist view,
the whole Franka for Stretch's arm in the exo view (see `RealFrankaOverlay`).

Usage:
    python -m examples.vla.molmobot_droid.run_stretch4_real --robot_ip 10.0.0.12 --object-height 0.80 \\
        --exo_camera left --gripper_camera left --slow
"""

from __future__ import annotations

import functools
import math
import os
import sys
import threading
import time
from typing import Callable

import click
import cv2
import mujoco
import numpy as np

from examples.vla.molmobot_droid import rerun_scene
from examples.vla.molmobot_droid.checkpoint import (
    FRANKA_HOME_QPOS,
    GRIPPER_CLOSED,
    POLICY_DT,
    load_policy,
    unload_policy,
)
from examples.vla.molmobot_droid.droid import (
    HEAD_CAMERAS,
    FrankaOverlayView,
    FrankaSpawn,
    Observation,
    OverlayKind,
    RobotPose,
    add_franka_ghost,
    franka_link0_height_for_object,
    mat_to_quat,
    resize_franka_pedestal,
)
from examples.vla.molmobot_droid.franka_retarget.stretch4_retarget import (
    CUSTOM_START_QPOS,
    HEAD_SIDES,
    FisheyeRectifier,
    MIN_BASE_ROTATION,
    PARALLEL_GRIPPER_TOOL,
    SLOW_FACTOR,
    SPAWN_LIFT_FRACTION,
    STEP_ARRIVAL_TIMEOUT,
    FrankaStretchRetargeter,
    ReachMiss,
    RetargetParams,
    StretchJoints,
    StretchTargets,
    apply_overlay_grasp_offset,
    custom_start_link0_height,
    custom_start_option,
    franka_yaw_for_start,
    head_camera_for_tool,
    head_rectifier,
    overlay_gripper_option,
    params_from_kwargs,
    planar_transform,
    prepare_exo,
    retarget_options,
    stretch_cameras_to_use,
    wrist_view,
)
from examples.vla.molmobot_droid.franka_retarget.console import KEYS_HELP, Console, continuous_session
from examples.vla.molmobot_droid.franka_retarget.gripper_image_fix import (
    HARDCODED_CALIBRATIONS,
    HARDCODED_CAMERA_CORRECTIONS,
    GripperImageFix,
    calibration_from_message,
    reject_with_fix,
)
from examples.vla.molmobot_droid.franka_retarget.start_pose_editor import StretchStartPoseEditor
from examples.vla.molmobot_droid.rollout import StepInfo
from stretch4_mujoco.enums.stretch_cameras import StretchCameras

IMAGE_PORT = 4409
"""`gripper_networking.gripper_and_joints_port` in stretch4_compliant_gripper."""

GRIPPER_PCT_OPEN = 300.0
"""stretch4_body StretchGripper `pct_max_open` for SE4 (range_deg [-100, 300]); 0 is fingertips touching."""

PARALLEL_GRIPPER_OPEN_MM = 77.0
"""The parallel gripper's widest opening: two fingers of 40 mm travel (the PG4 URDF); 0 is closed."""

# stretch4_body SE4 default motion profiles; --slow runs at SLOW_FACTOR of them.
DEFAULT_SPEEDS = {"lift": 0.3, "arm": 0.4, "wrist": 7.0, "base_rotate": 2.0}


MOTION_SUBSYSTEMS = ["arm", "lift", "omnibase", "end_of_arm"]
LIFT_RANGE = (0.0, 1.2)

SETTLED_TOLERANCE_M = 0.002
SETTLED_TOLERANCE_RAD = math.radians(1.0)
"""How close a message's joint state has to be to the robot's status for it to count as taken
after the robot stopped (see `RealStretch4Env.receive_settled()`)."""

SETTLED_TIMEOUT = 1.0
"""s. How long to wait for such a message before using the newest one anyway."""

DEFAULT_RECOVER_AFTER = 10
"""Unreachable steps in a row before going back to the start pose (--recover-after)."""

RETARGET_LOG = f"{rerun_scene.LOG_PATH}/retarget"
"""Rerun entity the actions Stretch could not follow are logged under, one per step."""

def _start_client(robot_ip: str):
    from stretch4_body.robot.robot_client import RobotClient

    robot = RobotClient(ip_address=robot_ip)
    # Over the network this check has to be skipped: it stats the robot's local server socket,
    # which a workstation does not have (see examples/digital_twin.py `_connect`).
    if not robot.startup(allow_different_user_connection=True):
        raise click.ClickException(f"Could not connect to Stretch 4 at {robot_ip}")
    return robot


def _configured_tool() -> str:
    from stretch4_body.core.robot_params import RobotParams

    return RobotParams.get_params()[1]["robot"]["tool"]


def connect(robot_ip: str):
    """
    A `RobotClient` whose end of arm is the robot's real one, and that gripper's joint name.

    stretch4_body builds the end of arm from the fleet directory's tool, and reads that once,
    at import. On a workstation the fleet directory is `ensure_fleet_directory()`'s stand-in,
    which names the simulator's default tool (the Stretch gripper); when the robot reports the
    parallel gripper, or vice versa, the stand-in is rewritten, stretch4_body re-imported, and
    the client started again.
    """
    from examples.digital_twin import (
        TOOL_FOR_GRIPPER_JOINT,
        ensure_fleet_directory,
        is_nominal_fleet_directory,
        robot_gripper_joint,
    )

    robot = _start_client(robot_ip)
    robot.pull_status()
    gripper = robot_gripper_joint(robot)
    if gripper is None:
        robot.stop()
        raise click.ClickException("The robot reports neither a stretch_gripper nor a parallel_gripper")
    tool = TOOL_FOR_GRIPPER_JOINT[gripper]
    configured = _configured_tool()
    if configured == tool:
        return robot, gripper

    robot.stop()
    if not is_nominal_fleet_directory():
        raise click.ClickException(
            f"The fleet directory ({os.environ.get('HELLO_FLEET_PATH')}/{os.environ.get('HELLO_FLEET_ID')}) "
            f"says the tool is {configured}, but the robot has a {gripper} ({tool}). Fix its "
            "stretch_configuration_params.yaml, or unset HELLO_FLEET_PATH to use a stand-in."
        )
    click.secho(f"The robot has a {gripper}; rebuilding the client for {tool}.", fg="yellow")
    ensure_fleet_directory(tool, rewrite=True)
    for name in [n for n in sys.modules if n == "stretch4_body" or n.startswith("stretch4_body.")]:
        del sys.modules[name]
    robot = _start_client(robot_ip)
    robot.pull_status()
    if robot_gripper_joint(robot) != gripper or _configured_tool() != tool:
        robot.stop()
        raise click.ClickException(f"Rebuilt the client for {tool}, but it still does not match the robot")
    return robot, gripper


def rotate_head_image_to_upright(image: np.ndarray, head_camera_side: str) -> np.ndarray:
    """
    Stretch 4's head fisheyes are mounted with landscape sensors: left turns 90° CCW, right 90° CW
    (as in recv_gripper_and_head_images_with_joint_states.py).
    """
    if head_camera_side == "left":
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if head_camera_side == "right":
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return image


def _decode(message: dict, *keys: str) -> np.ndarray | None:
    """The first image present under `keys` (JPEG bytes or raw BGR), as RGB."""
    for key in keys:
        if key not in message:
            continue
        value = message[key]
        image = cv2.imdecode(np.frombuffer(value, np.uint8), cv2.IMREAD_COLOR) if key.endswith("compressed") else value
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return None


def head_rectifiers(message: dict, say: Callable[[str, str | None], None]) -> dict[str, FisheyeRectifier]:
    """
    --rectify: a `FisheyeRectifier` per head fisheye in `message`, through the calibration the
    publisher sends with it (`head_<side>_camera_matrix`, scaled to the frames it sends) when that
    is a fisheye one: four coefficients, or the rest zero, as Luxonis reports its fisheye model.
    Otherwise stretch4_mujoco's calibration, the simulator's (another robot's lens).
    """
    rectifiers = {}
    for side, image in message["head_rgbs"].items():
        camera = HEAD_CAMERAS[side]
        matrix, distortion = message.get(f"head_{side}_camera_matrix"), message.get(f"head_{side}_distortion_coefficients")
        distortion = None if distortion is None else np.asarray(distortion, dtype=float).ravel()
        if matrix is None or distortion is None or len(distortion) < 4 or np.any(distortion[4:]):
            say(f"Rectifying the {side} head fisheye with stretch4_mujoco's calibration, not this robot's: "
                "the publisher sends " + ("none" if matrix is None else "one that is not a fisheye model"), "yellow")
            rectifiers[side] = head_rectifier(side)
            continue
        # The publisher's calibration is for its landscape frames, before they were turned upright.
        height, width = image.shape[:2]
        size = (height, width) if camera.initial_camera_settings.rotate_number_of_times % 2 else (width, height)
        rectifiers[side] = FisheyeRectifier(camera, matrix, distortion, size)
    return rectifiers


class ImageAndJointReceiver:
    """
    Subscribes to the robot's synced wrist + head frames and joint states on a thread, as
    recv_gripper_and_head_images_with_joint_states.py does, keeping only the newest message,
    decoded: `head_rgbs` has each head fisheye the publisher sends, upright, by side (a side
    missing from a message keeps its last frame). `receive()` waits for the next one; `latest()`
    does not wait.
    """

    def __init__(self, robot_ip: str, port: int = IMAGE_PORT):
        import zmq

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.socket.setsockopt(zmq.SNDHWM, 1)
        self.socket.setsockopt(zmq.RCVHWM, 1)
        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.connect(f"tcp://{robot_ip}:{port}")
        self._message: dict | None = None
        self._heads: dict[str, np.ndarray] = {}
        self._count = 0
        self._new = threading.Condition()
        self._running = True
        self._thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._thread.start()

    def _receive_loop(self) -> None:
        while self._running:
            if not self.socket.poll(100):
                continue
            message = self.socket.recv_pyobj()
            message["wrist_rgb"] = _decode(
                message, "wrist_color_image_compressed", "wrist_color_image", "color_image_compressed", "color_image"
            )
            primary = message.get("head_camera_side", "")
            heads = dict(self._heads)
            for side in message.get("head_camera_sides") or [primary]:
                # The first side is under the unprefixed keys too: an older publisher's only ones.
                keys = [f"head_{side}_color_image_compressed", f"head_{side}_color_image"]
                if side == primary:
                    keys += ["head_color_image_compressed", "head_color_image"]
                head = _decode(message, *keys)
                if head is not None:
                    heads[side] = rotate_head_image_to_upright(head, side)
            message["head_rgbs"] = self._heads = heads
            with self._new:
                self._message, self._count = message, self._count + 1
                self._new.notify_all()

    def latest(self) -> tuple[int, dict | None]:
        """(how many messages so far, the newest), without waiting."""
        with self._new:
            return self._count, self._message

    def receive(self, timeout_s: float = 5.0) -> dict:
        """The next message to arrive."""
        with self._new:
            seen = self._count
            if not self._new.wait_for(lambda: self._count > seen, timeout_s):
                raise TimeoutError("No images from the robot; is the publisher running with -r?")
            return self._message

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)
        self.socket.close(linger=0)
        self.context.term()


class RobotModel:
    """
    Stretch 4 (the stretch4_mujoco model) and the ghost Franka on a floor, posed from the real
    robot's joint states, for Rerun. Also where the virtual Franka stands for retargeting.
    """

    def __init__(self, link0_height: float, tool_name: str, franka_yaw: float = 0.0):
        """The Franka stands over the footprint where the run starts, turned `franka_yaw` (rad) to
        its left (see `franka_yaw_for_start()`)."""
        from stretch4_mujoco.stretch4_mujoco_simulator import Stretch4MujocoSimulator

        self.tool_name = tool_name
        spec = mujoco.MjSpec.from_file(Stretch4MujocoSimulator.get_robot_xml_path(tool_name))
        # The wheels' contact pairs name a geom "floor".
        spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.1])
        self.franka = add_franka_ghost(spec, RobotPose(0.0, 0.0, franka_yaw), link0_height)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

    def pose(self, world_from_footprint: np.ndarray, joints: StretchJoints, franka_q7, data=None) -> None:
        """Pose `data` (default: `self.data`) from the robot's joints and the Franka's."""
        data = self.data if data is None else data
        data.joint("floating_base").qpos = np.concatenate(
            [world_from_footprint[:3, 3], mat_to_quat(world_from_footprint[:3, :3])]
        )
        data.joint("lift_joint").qpos = joints.lift
        for i in range(1, 5):
            data.joint(f"arm_l{i}_joint").qpos = joints.arm / 4
        for joint in ("wrist_yaw", "wrist_pitch", "wrist_roll"):
            data.joint(f"{joint}_joint").qpos = getattr(joints, joint)
        closed, opened = self._finger_travel
        for joint in self._finger_joints:
            data.joint(joint).qpos = closed + joints.gripper_open_fraction * (opened - closed)
        for i, value in enumerate(franka_q7):
            data.joint(f"{self.franka.prefix}fr3_joint{i + 1}").qpos = value
        mujoco.mj_kinematics(self.model, data)

    @functools.cached_property
    def _finger_joints(self) -> list[str]:
        """The tool's finger joints: the Stretch gripper's, or the parallel gripper's."""
        names = [f"{prefix}finger_{side}_joint" for prefix in ("gripper_", "") for side in ("left", "right")]
        return [name for name in names if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) != -1]

    @functools.cached_property
    def _finger_travel(self) -> tuple[float, float]:
        """The finger joints' (closed, open) positions, as stretch4_mujoco maps the tool's gripper:
        the Stretch gripper's revolute fingers open upwards to 0.5 rad, the parallel gripper's
        prismatic ones downwards to -0.04 m."""
        from stretch4_mujoco import config

        settings = config.robot_settings_se4_pg4 if self.tool_name == PARALLEL_GRIPPER_TOOL else config.robot_settings_se4
        conversion = settings["gripper_conversion"]
        return conversion["urdf_closed_rad"], conversion["urdf_open_rad"]

    def set_franka_pedestal_height(self, height: float) -> None:
        """Raise or lower the virtual Franka to `fr3_link0` `height` above the floor: where it is
        retargeted from, and its ghost."""
        resize_franka_pedestal(self.model, self.franka, height)


class RealFrankaOverlay:
    """
    --overlay_franka_gripper on the real robot, for one camera: its image with the Franka drawn
    where the policy is told its arm and hand are, and the part of Stretch it stands in for
    painted out from the pixels around it (`kind`, see `FrankaOverlayView`: the Robotiq's
    fingers for the tool in the gripper camera, the whole Franka for the arm in the head
    camera). There is no scene to render what is behind Stretch, as the simulator's mirror does,
    and the camera is stretch4_mujoco's model of it, so this is only as good as the robot's
    calibration matches that model.

    Renders from `RobotModel`'s Stretch and ghost on a `MjData` of its own: `LiveView`'s thread
    poses the other.
    """

    TOOL_MARGIN_PX = 9
    """Stretch's mask is grown by this much before painting it out, for calibration error (at the
    working size)."""

    MAX_WORKING_SIZE_PX = 640
    """Images are composited no bigger than this on their longer side: the policy sees 320x180,
    and painting out the arm in a full 1920x1200 head image takes ~0.1 s."""

    OVERLAY_AMBIENT, OVERLAY_DIFFUSE = 0.4, 0.6
    """The headlight the Robotiq is rendered in."""

    def __init__(self, robot_model: RobotModel, camera: StretchCameras, kind: OverlayKind):
        self.robot_model = robot_model
        self.data = mujoco.MjData(robot_model.model)
        self.view = FrankaOverlayView(robot_model.model, robot_model.franka, camera, kind)
        # The model has no lights of its own, and nothing else renders it.
        robot_model.model.vis.headlight.ambient = [self.OVERLAY_AMBIENT] * 3
        robot_model.model.vis.headlight.diffuse = [self.OVERLAY_DIFFUSE] * 3
        self._warned_aspect = False

    def apply(self, image: np.ndarray, world_from_footprint: np.ndarray, joints: StretchJoints, state8) -> np.ndarray:
        """`image` with the Franka in place of Stretch's part, at most MAX_WORKING_SIZE_PX big."""
        self.robot_model.pose(world_from_footprint, joints, state8[:7], data=self.data)
        self.view.pose_fingers(self.data, state8[7])
        rgb = self.view.render(self.data)
        if not self._warned_aspect and abs(rgb.shape[1] / rgb.shape[0] - image.shape[1] / image.shape[0]) > 0.01:
            click.secho(f"The {self.view.kind} overlay's camera sends {image.shape[1]}x{image.shape[0]} images, not "
                        f"the model's {rgb.shape[1]}x{rgb.shape[0]} aspect; the overlay is stretched to fit", fg="yellow")
            self._warned_aspect = True
        scale = min(1.0, self.MAX_WORKING_SIZE_PX / max(image.shape[:2]))
        height, width = round(image.shape[0] * scale), round(image.shape[1] * scale)
        if image.shape[:2] != (height, width):
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        if rgb.shape[:2] != (height, width):
            rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
        fingers, tool = self.view.masks(self.data, (width, height))
        # All of Stretch's part, under the Franka too: what is left of it would be painted outwards.
        margin = np.ones((self.TOOL_MARGIN_PX, self.TOOL_MARGIN_PX), np.uint8)
        hole = cv2.dilate(tool.astype(np.uint8), margin) * 255
        out = cv2.inpaint(np.ascontiguousarray(image), hole, 5, cv2.INPAINT_TELEA)
        out[fingers] = rgb[fingers]
        return out

    def close(self) -> None:
        self.view.close()


class RealStretch4Env:
    """A real Stretch 4 driven by Franka actions; the real-robot counterpart of `Stretch4SimEnv`."""

    def __init__(self, robot, receiver: ImageAndJointReceiver, params: RetargetParams, franka: FrankaSpawn):
        self.robot = robot
        self.receiver = receiver
        self.params = params
        self.retargeter = FrankaStretchRetargeter(franka, params)
        self._gripper_closed: bool | None = None
        self._odometry_origin: np.ndarray | None = None
        self.last_message: dict = {}
        self.last_state8 = np.append(FRANKA_HOME_QPOS, 0.0)
        """The Franka state the policy last saw; Rerun's ghost follows it."""
        self.start_q7 = np.array(FRANKA_HOME_QPOS, dtype=float)
        """The Franka pose `go_home()` goes to."""
        self.gripper_overlay: RealFrankaOverlay | None = None
        self.exo_overlays: dict[str, RealFrankaOverlay] = {}
        """With --overlay_franka_gripper: what puts the Franka in the policy's wrist view, and in
        its exo view from each head fisheye."""
        self.exo_side = "right"
        """The head fisheye the policy's exo view comes from (`update_exo_side()`)."""
        self.interrupted = lambda: False
        """Polled while waiting on the robot; True cuts the wait short (a key was pressed)."""
        self.stale_observations = 0
        """Observations no message after the robot stopped arrived in time for (`receive_settled()`)."""
        self.say: Callable[[str, str | None], None] = lambda message, color=None: click.secho(message, fg=color)
        """Where `recover()` reports (the console's events while it runs)."""
        self.recover_after = DEFAULT_RECOVER_AFTER
        """Unreachable steps in a row before `recover()`; 0 never."""
        self.unreachable_streak = 0
        self.recoveries = 0
        self.last_miss: ReachMiss | None = None
        """Why the last step's action was twisted or unreachable (`FrankaStretchRetargeter.last_miss`)."""
        self.head_rectifiers: dict[str, FisheyeRectifier] = {}
        """With --rectify, what redraws each head fisheye as the DROID exo camera (`head_rectifiers()`)."""
        self.check_camera_sides()
        self.gripper_image_fix: GripperImageFix | None = None
        if params.fix_gripper_image_mismatch:
            # Through the lens calibration: hardcoded for now (see HARDCODED_CALIBRATIONS), else
            # the one the publisher sends with the images.
            camera = stretch_cameras_to_use(params)[0]
            calibration = HARDCODED_CALIBRATIONS.get(camera)
            if calibration is not None:
                click.secho(f"Gripper image fix: using the HARDCODED {camera.name} calibration (stretch-se4-4024's), "
                            "not this robot's", fg="yellow")
            else:
                calibration = calibration_from_message(self.receiver.latest()[1])
            if calibration is None:
                click.secho("The robot's publisher sends no gripper camera calibration: the gripper image fix "
                            "takes the camera to be a pinhole, and the lens's distortion is left in", fg="yellow")
            # And where the camera is: the model's, turned as HARDCODED_CAMERA_CORRECTIONS has it for now.
            correction = HARDCODED_CAMERA_CORRECTIONS.get(camera)
            if correction is not None:
                click.secho(f"Gripper image fix: turning the {camera.name} by a HARDCODED {correction} degrees "
                            "(fitted on stretch-se4-4024), not this robot's", fg="yellow")
            self.gripper_image_fix = GripperImageFix(
                params.tool_name, camera, real=True, calibration=calibration, camera_correction=correction
            )

    def check_camera_sides(self) -> None:
        message = self.receiver.receive()
        heads, wrist = sorted(message["head_rgbs"]), message.get("wrist_camera_side")
        if heads != sorted(HEAD_SIDES) or self.params.gripper_camera != wrist:
            raise click.ClickException(
                f"The robot publishes head={','.join(heads) or 'none'}, wrist={wrist}; restart its publisher "
                f"with --head_camera {','.join(HEAD_SIDES)} --wrist_camera {self.params.gripper_camera}"
            )
        self._state(message)  # fixes the odometry origin: the world frame is where we start
        if self.params.rectify:
            self.head_rectifiers = head_rectifiers(message, self.say)
        self.robot.pull_status()
        self.update_exo_side(quiet=True)

    def update_exo_side(self, quiet: bool = False) -> str:
        """Point `exo_side` at the head fisheye the tool is toward now, from the status last
        pulled, saying so when it changes (unless `quiet`)."""
        eoa = self.robot.status["end_of_arm"]
        facing, side = eoa.get("tool_facing"), eoa.get("tool_side")
        exo_side = head_camera_for_tool(facing, side)
        if exo_side != self.exo_side and not quiet:
            self.say(f"◐ exo view from the {exo_side} head camera (tool facing {facing}, on the {side})")
        self.exo_side = exo_side
        return exo_side

    # -- state ------------------------------------------------------------

    def _state(self, message: dict) -> tuple[np.ndarray, StretchJoints]:
        state = message["closest_joint_state"]
        odometry = state["base_odometry"]
        pose = np.array([odometry["x"], odometry["y"], odometry["theta"]])
        if self._odometry_origin is None:
            self._odometry_origin = pose
        # The world frame is the footprint where the run started, which is where the Franka stands.
        x0, y0, theta0 = self._odometry_origin
        dx, dy = pose[0] - x0, pose[1] - y0
        c, s = math.cos(-theta0), math.sin(-theta0)
        world_from_footprint = planar_transform(c * dx - s * dy, s * dx + c * dy, pose[2] - theta0)
        gripper = state["gripper"]
        if "pos_mm" in gripper:
            open_fraction = gripper["pos_mm"] / PARALLEL_GRIPPER_OPEN_MM
        else:
            open_fraction = gripper.get("pos_pct", 0.0) / GRIPPER_PCT_OPEN
        joints = StretchJoints(
            lift=state["lift"]["height"],
            arm=state["arm"]["extension"],
            wrist_yaw=state["wrist_yaw"]["angle"],
            wrist_pitch=state["wrist_pitch"]["angle"],
            wrist_roll=state["wrist_roll"]["angle"],
            gripper_open_fraction=float(np.clip(open_fraction, 0, 1)),
        )
        return world_from_footprint, joints

    def receive_settled(self) -> dict:
        """
        The next message whose joint state is where the robot is now, not where it was.

        A message's images and joint state were captured some way back (exposure, the
        publisher, encoding, the network), so the first one after a step can show the robot
        partway through it. At --slow that is a few mm; at full speed it can be most of the
        step, and the policy, told the arm has not moved, pushes it further. Waits up to
        SETTLED_TIMEOUT, counting a timeout in `stale_observations`.
        """
        deadline = time.monotonic() + SETTLED_TIMEOUT
        while True:
            message = self.receiver.receive()
            if self._matches(message["closest_joint_state"], self.live_joint_state()):
                return message
            if time.monotonic() > deadline:
                self.stale_observations += 1
                return message

    @staticmethod
    def _matches(state: dict, live: dict) -> bool:
        off_m = max(
            abs(state["lift"]["height"] - live["lift"]["height"]),
            abs(state["arm"]["extension"] - live["arm"]["extension"]),
        )
        off_rad = max(
            abs(state[joint]["angle"] - live[joint]["angle"]) for joint in ("wrist_yaw", "wrist_pitch", "wrist_roll")
        )
        return off_m <= SETTLED_TOLERANCE_M and off_rad <= SETTLED_TOLERANCE_RAD

    def live_joint_state(self) -> dict:
        """
        The robot's joints now, read from its status, in the form of a message's
        `closest_joint_state` (which the publisher builds from the same status).
        """
        self.robot.pull_status()
        status = self.robot.status
        eoa = status["end_of_arm"]
        if "parallel_gripper" in eoa:
            gripper = {"pos_mm": eoa["parallel_gripper"].get("pos_mm", 0.0)}
        else:
            gripper = {"pos_pct": eoa.get("stretch_gripper", {}).get("pos_pct", 0.0)}
        return {
            "lift": {"height": status["lift"]["pos"]},
            "arm": {"extension": status["arm"]["pos"]},
            **{joint: {"angle": eoa[joint]["pos"]} for joint in ("wrist_yaw", "wrist_pitch", "wrist_roll")},
            "gripper": gripper,
            "base_odometry": {key: status["base"][key] for key in ("x", "y", "theta")},
        }

    def observe(self) -> Observation:
        # Without --wait-for-arrival the robot is still moving, and no message would match.
        message = self.receive_settled() if self.params.wait_for_arrival else self.receiver.receive()
        # The images are as old as the pipeline makes them, but the joints need not be: moving,
        # the message's are an image's latency behind, and the policy would start each chunk,
        # and limit_joint_delta() each action, from a little way back along the motion.
        message = {**message, "closest_joint_state": self.live_joint_state()}
        self.last_message = message
        footprint, joints = self._state(message)
        state8, _ = self.retargeter.stretch_to_franka(footprint, joints)
        self.last_state8 = state8
        side = self.update_exo_side()  # from the status live_joint_state() pulled
        wrist, head = message["wrist_rgb"], message["head_rgbs"][side]
        if self.gripper_overlay is not None:
            wrist = self.gripper_overlay.apply(wrist, footprint, joints, state8)
        if side in self.exo_overlays:
            head = self.exo_overlays[side].apply(head, footprint, joints, state8)
        if self.gripper_image_fix is not None:
            wrist = self.gripper_image_fix(wrist, state8[7], self.params.tcp_offset)
        else:
            wrist = wrist_view(wrist)
        return Observation(
            exo_rgb=prepare_exo(head, self.params, side, self.head_rectifiers.get(side)),
            wrist_rgb=wrist,
            state8=state8,
            extra_cameras={"head_raw": message["head_rgbs"][side], "gripper_raw": message["wrist_rgb"]},
        )

    # -- acting -------------------------------------------------------------

    def step(self, action8: np.ndarray) -> StretchTargets | None:
        # From where the robot is now: the base turns by a relative amount.
        footprint, joints = self._state({"closest_joint_state": self.live_joint_state()})
        targets = self.retargeter.franka_to_stretch(action8, footprint, joints)
        self.last_miss = self.retargeter.last_miss
        self.unreachable_streak = 0 if targets is not None else self.unreachable_streak + 1
        if self.recover_after and self.unreachable_streak >= self.recover_after:
            self.recover()
            return None
        start = time.monotonic()
        if targets is not None:
            self.send(targets)
        else:
            self.send_gripper(bool(action8[7] >= GRIPPER_CLOSED / 2))
            self.robot.push_command()
        if self.params.wait_for_arrival:
            self.wait_for_arrival(STEP_ARRIVAL_TIMEOUT)
        remaining = POLICY_DT - (time.monotonic() - start)
        if remaining > 0 and not self.interrupted():
            time.sleep(remaining)
        return targets

    def wait_for_arrival(self, timeout: float) -> bool:
        """
        `robot.wait_command()`, except that `interrupted()` ends it early. Returns whether the
        robot stopped moving.
        """
        self.robot.wait_on_motion_start(MOTION_SUBSYSTEMS, timeout=0.2)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.interrupted():
                return False
            self.robot.pull_status()
            if not self.robot.is_moving():
                return True
            time.sleep(0.02)
        return False

    def _speed(self, joint: str) -> float | None:
        return SLOW_FACTOR * DEFAULT_SPEEDS[joint] if self.params.slow else None

    def send(self, targets: StretchTargets) -> None:
        robot = self.robot
        if abs(targets.base_rotate_by) > MIN_BASE_ROTATION:
            robot.base.rotate_by(targets.base_rotate_by, v_r=self._speed("base_rotate"))
        robot.lift.move_to(targets.lift, v_m=self._speed("lift"))
        robot.arm.move_to(targets.arm, v_m=self._speed("arm"))
        for joint in ("wrist_yaw", "wrist_pitch", "wrist_roll"):
            robot.end_of_arm.move_to(joint, getattr(targets, joint), v_r=self._speed("wrist"))
        self.send_gripper(targets.gripper_closed)
        robot.push_command()

    def send_gripper(self, closed: bool) -> None:
        if closed != self._gripper_closed:
            self._gripper_closed = closed
            joint = "parallel_gripper" if self.params.use_parallel_gripper else "stretch_gripper"
            self.robot.end_of_arm.pose(joint, "close" if closed else "open")

    def tool_error_m(self) -> float | None:
        """How far Stretch's tool is from where the policy wants it."""
        target = self.retargeter.last_target_tool_world
        if target is None or not self.last_message:
            return None
        footprint, joints = self._state(self.last_message)
        return float(np.linalg.norm(self.retargeter.stretch_tool_world(footprint, joints)[:3, 3] - target[:3, 3]))

    def move_to_franka_pose(self, franka_q7=FRANKA_HOME_QPOS, gripper_closed: bool = False) -> None:
        self.retargeter.reset()
        self.retargeter.franka_seed = np.asarray(franka_q7, dtype=float)
        footprint, joints = self._state(self.receiver.receive())
        gripper = GRIPPER_CLOSED if gripper_closed else 0.0
        targets = self.retargeter.franka_to_stretch(np.concatenate([franka_q7, [gripper]]), footprint, joints)
        if targets is None:
            raise RuntimeError("Stretch 4 cannot reach the Franka's start pose; check --object-height")
        self._gripper_closed = None
        self.send(targets)
        self.wait_for_arrival(30.0)

    def go_home(self, franka_q7=None, keep_gripper: bool = False) -> None:
        """
        Back to the start pose: the lift up to SPAWN_LIFT_FRACTION of its travel first, so the
        gripper clears whatever it is over, then where the Franka at `franka_q7` (default
        `start_q7`) has its tool. The gripper opens, unless `keep_gripper`.
        """
        franka_q7 = self.start_q7 if franka_q7 is None else franka_q7
        self.stop_motion()
        _, joints = self._state(self.receiver.receive())
        high = LIFT_RANGE[0] + SPAWN_LIFT_FRACTION * (LIFT_RANGE[1] - LIFT_RANGE[0])
        if joints.lift < high:
            self.robot.lift.move_to(high, v_m=self._speed("lift"))
            self.robot.push_command()
            if not self.wait_for_arrival(15.0):
                return
        self.move_to_franka_pose(franka_q7, gripper_closed=keep_gripper and bool(self._gripper_closed))

    def recover(self) -> None:
        """
        Stretch has not been able to follow `recover_after` actions in a row (its wrist against
        its limits, most likely, with the policy pushing on): back to the start pose, which
        leaves the wrist the most room, with the gripper as it is, for the policy to carry on
        from. The retargeting counts carry on too.
        """
        self.say(f"↺ {self.unreachable_streak} unreachable steps in a row; back to the start pose, gripper kept", "yellow")
        self.unreachable_streak = 0
        self.recoveries += 1
        retargeter = self.retargeter
        counts = retargeter.ik_failures, retargeter.ik_clamped, retargeter.ik_twisted
        try:
            self.go_home(keep_gripper=True)
        except RuntimeError as error:
            self.say(f"↺ {error}", "red")
        retargeter.ik_failures, retargeter.ik_clamped, retargeter.ik_twisted = counts

    def stop_motion(self) -> None:
        """Hold every joint where it is now."""
        self.robot.pull_status()
        status = self.robot.status
        self.robot.base.set_velocity(0.0, 0.0, 0.0)
        self.robot.lift.move_to(status["lift"]["pos"])
        self.robot.arm.move_to(status["arm"]["pos"])
        for joint in ("wrist_yaw", "wrist_pitch", "wrist_roll"):
            self.robot.end_of_arm.move_to(joint, status["end_of_arm"][joint]["pos"])
        self.robot.push_command()


class LiveView:
    """
    Logs the robot's raw cameras and pose (and the ghost Franka at the state the policy last
    saw) to Rerun at `hz`, on its own thread, from the first message on -- whether or not an
    instruction is running.

    On the `time` timeline, and on `step` at `step()`, the step the policy's views were last
    logged at: Rerun keeps the time per thread, so without it this thread's logs would be on no
    `step` at all, and gone when replaying by step.
    """

    def __init__(self, env: RealStretch4Env, robot_model: RobotModel, step: Callable[[], int], hz: float = 10.0):
        self.env, self.robot_model, self.step, self.period = env, robot_model, step, 1.0 / hz
        _, message = env.receiver.latest()
        robot_model.pose(*self._robot_state(message or env.receiver.receive()))
        self.scene = rerun_scene.RerunScene(robot_model.model, robot_model.data, ["stretch4", robot_model.franka.base_name])
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _robot_state(self, message: dict):
        footprint, joints = self.env._state(message)
        return footprint, joints, self.env.last_state8[:7]

    def _loop(self) -> None:
        last = 0
        while self._running:
            count, message = self.env.receiver.latest()
            if message is not None and count != last:
                last = count
                rerun_scene.rr.set_time("time", timestamp=time.time())
                rerun_scene.set_step(self.step())
                head = message["head_rgbs"].get(self.env.exo_side)
                rerun_scene.log_cameras({"head_raw": head, "gripper_raw": message["wrist_rgb"]})
                self.robot_model.pose(*self._robot_state(message))
                self.scene.log(self.robot_model.data)
            time.sleep(self.period)

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)


@click.command()
@click.option("--robot_ip", "--robot-ip", "robot_ip", required=True, help="Stretch 4's IP address.")
@click.option("--object-height", type=float, default=0.80, show_default=True,
              help="Height of the target object above the floor (m); sets the virtual Franka's height "
              "(not used with --custom_franka_start_pose).")
@click.option("--max-steps", type=int, default=0, show_default=True,
              help="Stop an instruction after this many steps (15/s). 0: run until told otherwise.")
@click.option("--recover-after", type=int, default=DEFAULT_RECOVER_AFTER, show_default=True,
              help="After this many actions in a row Stretch cannot follow, go back to the start pose "
              "(gripper kept) and carry on. 0: never.")
@click.option("--checkpoint", default=None)
@click.option("--rerun/--no-rerun", default=True, show_default=True)
@retarget_options(gripper_option=False)
@custom_start_option
@overlay_gripper_option
def main(robot_ip, object_height, max_steps, recover_after, checkpoint, rerun, custom_franka_start_pose,
         overlay_franka_gripper, **kwargs):
    params = params_from_kwargs(kwargs)
    reject_with_fix(params.fix_gripper_image_mismatch, overlay_franka_gripper=overlay_franka_gripper)
    if params.exo_camera == "droid":
        raise click.BadParameter("the real robot has no DROID exo camera; its exo view is the head camera on the "
                                 "tool's side", param_hint="--exo_camera")

    click.secho(f"Connecting to Stretch 4 at {robot_ip}...", dim=True)
    robot, gripper = connect(robot_ip)
    if not robot.is_homed():
        robot.stop()
        raise click.ClickException("Home the robot first")

    # The gripper on the robot decides the tool, its kinematics and its default grasp offset.
    params.use_parallel_gripper = gripper == "parallel_gripper"
    if overlay_franka_gripper:
        apply_overlay_grasp_offset(params)  # before the start pose, which depends on it
    if custom_franka_start_pose:
        link0_height = custom_start_link0_height(params, floor_z=0.0)
        start_q7 = np.array(CUSTOM_START_QPOS)
        click.secho(f"Virtual Franka's fr3_link0 at {link0_height:.3f} m, starting tilted", dim=True)
    else:
        link0_height = franka_link0_height_for_object(object_height)
        start_q7 = np.array(FRANKA_HOME_QPOS)
    # Turned the way the start pose would turn Stretch's base, so the base does not have to.
    robot_model = RobotModel(link0_height, params.tool_name, franka_yaw_for_start(params, link0_height, start_q7))
    receiver = ImageAndJointReceiver(robot_ip)
    env = live = console = None
    try:
        env = RealStretch4Env(robot, receiver, params, robot_model.franka)
        env.start_q7 = start_q7
        env.recover_after = recover_after
        if overlay_franka_gripper:
            gripper_camera = stretch_cameras_to_use(params)[0]
            env.gripper_overlay = RealFrankaOverlay(robot_model, gripper_camera, "fingers")
            env.exo_overlays = {side: RealFrankaOverlay(robot_model, HEAD_CAMERAS[side], "franka") for side in HEAD_SIDES}
        offset = ",".join(f"{v:g}" for v in params.effective_grasp_offset_mm)
        click.echo(
            click.style("Stretch 4 ", bold=True) + f"{robot_ip} · {gripper.replace('_', ' ')} · grasp offset {offset} mm\n"
            + click.style("Cameras  ", bold=True) + f"head on the tool's side (now {env.exo_side}), gripper {params.gripper_camera}"
            + (" (head rectified to DROID's exo camera)" if params.rectify
               else " (head cropped to DROID)" if params.head_crop == "droid" else "")
            + (", gripper image matched to DROID's wrist camera" if params.fix_gripper_image_mismatch else "") + "\n"
            + click.style("Motion   ", bold=True) + ("slow" if params.slow else "full speed")
            + (", waiting for each action to arrive" if params.wait_for_arrival else "")
            + f", {params.execute_first_n} of every {params.execute_horizon} predicted actions"
            + ("\n" + click.style("Overlay  ", bold=True) + "the Franka in the wrist and exo views" if overlay_franka_gripper else "")
        )
        step_count = 0
        if rerun:
            rerun_scene.init_rerun(
                "MolmoBot-DROID Stretch 4 (real)", ["exo", "wrist", "head_raw", "gripper_raw"], text_log=True,
                main_cameras=["exo", "wrist"],
            )
            live = LiveView(env, robot_model, step=lambda: step_count)
            click.secho("Rerun is showing the cameras live.", dim=True)

        click.secho("Loading MolmoBot-DROID...", dim=True)
        policy = load_policy(checkpoint)

        def log_frame(observation: Observation) -> None:
            """One Rerun step: rollout steps and start pose moves share the `step` timeline."""
            nonlocal step_count
            step_count += 1
            if not rerun:
                return
            rerun_scene.set_step(step_count)
            rerun_scene.rr.set_time("time", timestamp=time.time())
            rerun_scene.log_cameras({"exo": observation.exo_rgb, "wrist": observation.wrist_rgb})
            footprint, joints = env._state(env.last_message)
            rerun_scene.log_tool_poses(
                env.retargeter.last_target_tool_world, env.retargeter.stretch_tool_world(footprint, joints)
            )

        def log_step(info: StepInfo) -> None:
            log_frame(info.observation)
            miss = env.last_miss
            if rerun and miss is not None:
                rerun_scene.log_text(
                    f"{RETARGET_LOG}/{miss.kind}", miss.describe(), "ERROR" if miss.kind == "unreachable" else "WARN"
                )

        if click.confirm(click.style("Raise the lift and move Stretch 4 to the start pose?", bold=True), default=True):
            click.secho("Moving to the start pose...", dim=True)
            env.go_home()
        else:
            click.secho("Staying put; type home when ready.", dim=True)
        log_frame(env.observe())  # the ghost and Rerun to where the robot is

        click.echo("\n" + KEYS_HELP + "\n")
        console = Console()
        editor = StretchStartPoseEditor(
            env,
            go_to=env.go_home,
            set_pedestal_height=robot_model.set_franka_pedestal_height,
            keys=console.keys,
            say=console.event,
            on_move=lambda: log_frame(env.observe()),
        )
        continuous_session(env, policy, params, max_steps, log_step, console, editor, reset=env.go_home)
    except KeyboardInterrupt:
        pass
    finally:
        if console is not None:
            console.close()
        click.secho("Stopping the robot and exiting.", dim=True)
        if env is not None:
            env.stop_motion()
        if live is not None:
            live.close()
        for overlay in (env.gripper_overlay, *env.exo_overlays.values()) if env is not None else ():
            if overlay is not None:
                overlay.close()
        unload_policy()
        receiver.close()
        robot.stop()


if __name__ == "__main__":
    main()
