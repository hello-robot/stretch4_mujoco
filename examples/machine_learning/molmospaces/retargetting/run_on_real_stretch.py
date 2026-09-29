"""
Drive a real Stretch 4 with the retargeted MolmoBot-DROID checkpoint, from a workstation.

Everything else in this package runs the retargeting in MuJoCo. This runs the
same retargeting against the robot on the bench: the policy and the kinematics
live on this machine (a GPU it can hold the checkpoint on), the robot streams its
two cameras and its joint states over ZMQ, and the joint targets go back over
`stretch4_body`'s `RobotClient`. You type the instruction at the prompt.

    # on the robot
    python send_gripper_and_head_images_with_joint_states.py \\
        --wrist-camera right --head-camera right

    # on the workstation, with the flags the study's best side-by-side run used
    python -m examples.machine_learning.molmospaces.retargetting.run_on_real_stretch \\
        --robot-ip 100.71.110.52 \\
        --map_franka_wrist_to_flipped_stretch4_wrist \\
        --change_franka_start_pose_limit_height \\
        --grasp-offset-m -0.009 --target-z-offset-m 0.035

    > Pick up the cup          # the instruction, handed to the policy verbatim
    >                          # space or enter: stop the arm where it is, now
    > Pick up the bowl         # and the next instruction starts from there
    > go                       # carry on with the one before
    > home                     # go back to the Franka's home pose
    > quit

Space and Enter stop the robot the moment they are pressed, without a line to
finish -- see `Console`, which reads keys rather than lines wherever the terminal
allows it, and `RobotCommander.hold`, which stops the arm where it is rather than
letting it finish the reach it was last sent.

The chain, once per control step
--------------------------------
    the robot's stream       wrist + head frames and the joint state sampled
                             nearest to the wrist frame          (`RobotStream`)
      -> `RobotMirror`       those joints written into a headless MuJoCo Stretch,
                             which is what the IK and the FK are solved on
      -> `FrankaOnStretchView`  the arm, as seven Franka joint angles
      -> `RealRobotVLAPolicy`   seven Franka joint targets and a Robotiq command
      -> `FrankaOnStretchView`  base / lift / arm / wrist / gripper targets
      -> `RobotCommander`       `move_to` per joint, one `push_command` per step

The MuJoCo model is a *mirror*, not a simulation: nothing is ever stepped in it.
It is there because the retargeting is kinematics -- `StretchArmIK` solves on an
`MjData` and `franka_joint_pos()` reads the tool pose out of one -- so the real
robot's joint angles are written in and the solve happens against the pose the
robot is actually in. That also makes the robot the only integrator in the loop:
what it reports is what the next action is computed from.

What is different from a sim rollout, and worth knowing before the first run
---------------------------------------------------------------------------
* **The cameras are the real ones, and only the real ones.** A sim setup can
  mount whatever exo camera it likes at Stretch's head; the robot has a
  123-degree fisheye bolted on sideways. So the head channel here is
  `stretch_fisheye`'s: rotate upright, optionally rectify, optionally crop --
  see `head_frame_for_policy`. `--head-mode rectified` is `stretch_rectified`.
  There is no equivalent of the pinhole setups on hardware.
* **The frames arrive as BGR.** The stream carries OpenCV-decoded JPEG; the
  checkpoint was trained on RGB. The conversion is in `RobotStream`, once, on the
  way in, because a channel swap is invisible in a viewer that draws either.
* **Nothing resets.** `reset` in a benchmark re-spawns a robot; here it re-seeds
  the retargeting from wherever the arm happens to be (`new_episode`). The one
  thing that does move the robot on its own is the opening snap to the Franka's
  home pose, which is a real motion and the first thing to watch.
* **The base is out of the IK by default.** `--include-base` puts it back and the
  policy can then drive the robot; on hardware that is a different kind of risk
  from a lift command, so it is opt-in and speed-capped. See `RobotCommander`.
* **A stale stream stops the robot.** If frames stop arriving for
  `--max-obs-age` seconds the loop stops sending and says so, rather than driving
  on the last thing it saw -- and it stays stopped once they come back, because a
  policy resuming a reach across a gap it could not see is worse than a pause.
  Type `go` to carry on.
* **Which physical camera feeds each channel is the sender's flag, not one
  here.** `--wrist-camera left` on the robot is what the sim study calls
  `--use_left_gripper_camera`, and `--head-camera left` its
  `--use_left_fisheye_camera`; the side the robot chose is printed at startup and
  used to stand the head frames up the right way.

Requires `stretch4_body` (`uv pip install -e ".[digital-twin]"`), `pyzmq`,
`rerun-sdk`, and MolmoBot on the path -- see `policies/molmobot_droid_policy.py`
for the last one. `stretch4_body` also wants a *fleet directory*, which a
workstation does not have; `digital_twin.ensure_fleet_directory` writes a nominal
stand-in so the import succeeds, and `--fleet-path` / `--fleet-id` point at the
robot's real one, which is what you want for anything read back out of
`robot_params`. `--dry-run` needs none of `stretch4_body`: it runs the whole
chain against the stream and logs it, without connecting to the robot or moving
anything, which is the way to check the cameras and the retargeting before
anything can move.
"""

from __future__ import annotations

import logging
import math
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# MuJoCo binds the backend named by MUJOCO_GL when `mujoco` is first imported,
# which the imports below trigger. Nothing here renders -- the mirror is never
# drawn -- but the import still has to find a backend on a headless workstation.
if sys.platform == "linux":
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import click  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from mujoco import MjData, MjSpec  # noqa: E402

from examples.digital_twin import ensure_fleet_directory  # noqa: E402
from examples.machine_learning.molmospaces.policies import franka_retarget as fr  # noqa: E402
from examples.machine_learning.molmospaces.policies.molmobot_droid_policy import (  # noqa: E402
    DROID_EXO_CAMERA_KEY,
    DROID_WRIST_CAMERA_KEY,
)
from examples.machine_learning.molmospaces.retargetting.cameras import (  # noqa: E402
    _centre_crop_resize,
)
from examples.machine_learning.molmospaces.retargetting.setups import (  # noqa: E402
    DROID_FRAME_SIZE,
    HEAD_CAMERA_PITCH_DEG,
    apply_aperture,
    apply_tool_correction,
)
from examples.machine_learning.molmospaces.stretch.config import Stretch4RobotConfig  # noqa: E402
from examples.machine_learning.molmospaces.stretch.robot import Stretch4Robot  # noqa: E402
from examples.machine_learning.molmospaces.stretch.robot_view import (  # noqa: E402
    Stretch4RobotView,
    commandable_limits,
)
from stretch4_mujoco.config import robot_settings_se4  # noqa: E402
from stretch4_mujoco.enums.stretch_cameras import StretchCameras  # noqa: E402
from stretch4_mujoco.gamepad_joints import WRIST_ROLL_SIM_SIGN  # noqa: E402

log = logging.getLogger(__name__)

# `connect_robot` imports `stretch4_body`, which reads a fleet directory while it
# is being imported and exits if there is none -- so the stand-in has to be in
# place before then, and here is the last point that is true of regardless of
# which way the run reaches that import. A no-op where the environment already
# names a fleet directory; see `digital_twin.ensure_fleet_directory` for what
# using a stand-in costs.
ensure_fleet_directory()


# =============================================================================
# What the two ends of the link agree on
# =============================================================================

DEFAULT_JOINTS_PORT = 4409
"""`gripper_networking.gripper_and_joints_port`, restated so this module imports
without the gripper repository on the path. `--port` overrides it, and the real
value is read from `gripper_networking` when that package *is* importable."""

CONTROL_HZ = 25.0
"""The rate every other MolmoBot-DROID rollout in this repository runs at.

The action *scale* is tied to it -- the checkpoint was cloned from an expert
stepping at 66ms -- so running the robot faster does not make it quicker, it
makes each commanded step a larger fraction of a motion the model metered out.
The loop below holds the rate when it can and falls behind on the steps where
the model is queried, which is safe here because every target is absolute.
"""

WRIST_CAMERA_FOV_DEG = float(
    StretchCameras.cam_gripper_se4_right_rgb.initial_camera_settings
    .field_of_view_vertical_in_degrees
)
"""The gripper camera's own vertical field of view. `--wrist-fov-deg` crops into it."""

WRIST_OUTPUT_SIZE = (656, 368)
"""What a Stretch wrist frame is when a sim rollout hands one to the policy.

`stretch.config.CAMERA_OUTPUT_SIZE["wrist_camera_right"]`. The hardware streams
640x400, so the real frame is cropped and resized to this: the checkpoint reads
the wrist channel more closely than any other (see `cameras.RetargetParams.
wrist_fov_deg`), and a frame with a different aspect is a differently framed
grasp rather than the same one at another size.
"""

GRIPPER_JOINT = "stretch_gripper"
"""The tool joint this runs against.

The robot's sender polls `status['end_of_arm']['stretch_gripper']` and publishes
`pos_pct`, so that is the hand this can read. A PG4 would stream nothing under
that key and the mirror's fingers would sit at whatever they were last written
to, which is why the absence is an error rather than a default.
"""

STRETCH_GRIPPER_CLOSED_PCT = -100.0
"""`pos_pct` with the fingers shut. `GripperUnits` maps the range onto the model's."""

MAX_TARGET_STEP = {
    "lift": 0.10,
    "arm": 0.10,
    "wrist": 0.60,
    "gripper": 0.50,
}
"""How far one step's target may sit from where the joint currently is: metres
for the prismatic joints, radians for the wrist and the fingers.

A guard against a garbage prediction, not a speed limit, and deliberately sized
well above what the robot can do in a control period -- the robot lags a streamed
target by construction (see `digital_twin.py`), so a clamp near its actual speed
would throttle every step rather than only the bad ones. `--step-limit-scale`
tightens all four together.
"""

SLOW_SPEED_SCALE = 0.4
"""What `--slow` multiplies every commanded velocity and acceleration by.

60% slower than the profile the joint would otherwise run, which is its `max`.
Both numbers, not just the velocity: scaling the speed alone leaves every motion
starting and stopping as sharply as it did, and the abruptness is most of what
makes a robot driven by a policy feel fast in a room with people in it.

It changes how the robot *follows*, not what it is asked for. The targets are the
same targets -- so a slower robot lags further behind a stream that is still
running at 15Hz, and the policy sees a gripper that has not arrived yet, which is
a real difference in what it is closing its loop on. Read a slow rollout as a
rehearsal, not as a measurement.
"""

MAX_BASE_SPEED_MPS = 0.25
MAX_BASE_TURN_RADPS = 0.5
"""Caps on what one step may ask the base for, under `--include-base`.

Half of what `live_policy.py` allows itself in simulation. A base command here
moves a 25kg robot across a room it shares with whoever is running it, and the
policy's base output is an IK by-product -- the solver drives the base only
because the arm could not reach -- so it is worth being slower than the arm
rather than faster.
"""


# =============================================================================
# The robot's stream
# =============================================================================


@dataclass
class RobotObservation:
    """One synchronized message from `send_gripper_and_head_images_with_joint_states.py`."""

    wrist_rgb: np.ndarray | None
    head_rgb: np.ndarray | None
    wrist_side: str
    head_side: str
    joints: dict[str, Any]
    """`closest_joint_state`: the joint sample nearest in time to the wrist frame."""

    head_camera_matrix: np.ndarray | None
    head_distortion: np.ndarray | None
    received_at: float
    """`time.monotonic()` when this message was decoded here, which is what
    `age` is measured against -- the two machines' wall clocks are not the same
    clock and the difference between them is not the latency."""

    sequence: int
    head_sync_offset_ms: float

    @property
    def age(self) -> float:
        return time.monotonic() - self.received_at

    def joint(self, *path: str, default: float = 0.0) -> float:
        """One number out of the joint state, or `default` when the robot did not send it."""
        node: Any = self.joints
        for key in path:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        try:
            return float(node)
        except (TypeError, ValueError):
            return default


class RobotStream:
    """Subscribes to the robot's camera-and-joint-state publisher, on its own thread.

    `zmq.CONFLATE` with a queue depth of one, the way every receiver in the
    gripper repository sets it up: a control loop wants the newest frame and a
    backlog is latency, not data. The thread therefore holds exactly one message
    and the loop reads whatever is there when it looks.

    Images are decoded and converted to RGB here rather than at the policy,
    because that is the one place it can be done once per message and because a
    BGR frame reaching a checkpoint trained on RGB is invisible in any viewer.
    """

    def __init__(self, address: str) -> None:
        self.address = address
        self._latest: RobotObservation | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.messages = 0
        self.dropped = 0
        self._last_sequence: int | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="robot-stream", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def latest(self) -> RobotObservation | None:
        with self._lock:
            return self._latest

    def wait_for_first(self, timeout: float) -> RobotObservation | None:
        """Block until the robot has sent something, or `timeout` runs out."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not self._stop.is_set():
            observation = self.latest()
            if observation is not None:
                return observation
            time.sleep(0.05)
        return None

    def _loop(self) -> None:
        import zmq

        context = zmq.Context.instance()
        socket = context.socket(zmq.SUB)
        socket.setsockopt(zmq.SUBSCRIBE, b"")
        socket.setsockopt(zmq.SNDHWM, 1)
        socket.setsockopt(zmq.RCVHWM, 1)
        socket.setsockopt(zmq.CONFLATE, 1)
        socket.connect(self.address)
        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)
        try:
            while not self._stop.is_set():
                # Polled rather than blocking on recv, so Ctrl-C in the main
                # thread does not leave this one waiting on a robot that has
                # stopped publishing.
                if not poller.poll(timeout=200):
                    continue
                try:
                    message = socket.recv_pyobj()
                except Exception as error:  # noqa: BLE001 - one bad message is not the end
                    log.warning(f"[stream] could not read a message: {error!r}")
                    continue
                try:
                    observation = self._decode(message)
                except Exception as error:  # noqa: BLE001
                    log.warning(f"[stream] could not decode a message: {error!r}")
                    continue
                with self._lock:
                    self._latest = observation
        finally:
            poller.unregister(socket)
            socket.close(linger=0)

    def _decode(self, message: dict) -> RobotObservation:
        sequence = int(message.get("image_number", -1))
        if self._last_sequence is not None and sequence > 0:
            self.dropped += max(0, sequence - self._last_sequence - 1)
        self._last_sequence = sequence if sequence > 0 else self._last_sequence
        self.messages += 1

        joints = message.get("closest_joint_state")
        if joints is None:
            history = message.get("joint_state_history") or []
            joints = history[-1] if history else {}

        head_matrix = message.get("head_camera_matrix")
        head_distortion = message.get("head_distortion_coefficients")
        return RobotObservation(
            wrist_rgb=_decode_frame(message, "wrist_color_image", "color_image"),
            head_rgb=_decode_frame(message, "head_color_image"),
            wrist_side=str(message.get("wrist_camera_side", "right")),
            head_side=str(message.get("head_camera_side", "right")),
            joints=joints,
            head_camera_matrix=None if head_matrix is None else np.asarray(head_matrix, float),
            head_distortion=(
                None if head_distortion is None else np.asarray(head_distortion, float)
            ),
            received_at=time.monotonic(),
            sequence=sequence,
            head_sync_offset_ms=float(message.get("head_sync_offset_ms", 0.0)),
        )


def _decode_frame(message: dict, *prefixes: str) -> np.ndarray | None:
    """One image out of a message, as RGB.

    The sender publishes each camera either as raw pixels or as a JPEG buffer
    under a `_compressed` key, and carries the wrist camera under two names for
    receivers written before the head camera existed -- so the lookup is by
    prefix rather than by a single key.
    """
    import cv2

    for prefix in prefixes:
        compressed = message.get(f"{prefix}_compressed")
        if compressed is not None:
            frame = cv2.imdecode(np.frombuffer(compressed, np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                return np.ascontiguousarray(frame[..., ::-1])
        raw = message.get(prefix)
        if raw is not None:
            frame = np.asarray(raw)
            if frame.ndim == 3 and frame.shape[2] == 3:
                return np.ascontiguousarray(frame[..., ::-1])
            return np.ascontiguousarray(frame)
    return None


# =============================================================================
# Real frames -> what the checkpoint was trained to look at
# =============================================================================

_RECTIFY_MAPS: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}


def _rectify_fisheye(
    frame: np.ndarray, camera_matrix: np.ndarray, distortion: np.ndarray
) -> np.ndarray:
    """Undo the head camera's fisheye warp, with the calibration the robot sent.

    `P = K`, as in `cameras._rectify`: keeping the focal length means the
    rectified frame covers the same angular window the distorted one did, so a
    rectified run and a distorted run are looking at the same scene rather than
    two different crops of it.

    The calibration comes off the stream instead of `StretchCameras`, because
    this is *this* robot's camera and the fleet calibration the sim warps with is
    a different lens of the same model. The remap tables are cached: they cost
    more to build than the remap itself and neither the size nor the calibration
    changes during a run.
    """
    import cv2

    height, width = frame.shape[:2]
    key = (width, height, camera_matrix.tobytes(), distortion.tobytes())
    maps = _RECTIFY_MAPS.get(key)
    if maps is None:
        coefficients = np.asarray(distortion, dtype=np.float64).reshape(-1, 1)[:4]
        maps = cv2.fisheye.initUndistortRectifyMap(
            np.asarray(camera_matrix, dtype=np.float64),
            coefficients,
            np.eye(3),
            np.asarray(camera_matrix, dtype=np.float64),
            (width, height),
            cv2.CV_32FC1,
        )
        _RECTIFY_MAPS[key] = maps
    return cv2.remap(frame, maps[0], maps[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def _upright_quarter_turns(head_side: str) -> int:
    """`np.rot90` turns that stand a head frame up, for the camera on that side.

    The head cameras are bolted on sideways and the two are turned opposite ways:
    `StretchCameras.cam_nav_rgb_se4_right.initial_camera_settings.
    rotate_number_of_times` is -1 and the left camera's is +1. The robot's own
    viewer does the same thing with `cv2.ROTATE_90_CLOCKWISE` for the right
    camera, and `stretch.config` applies the same turn to a rendered frame -- so
    a frame that is upright here is the frame a sim rollout fed the policy.
    """
    camera = (
        StretchCameras.cam_nav_rgb_se4_left
        if head_side.lower() == "left"
        else StretchCameras.cam_nav_rgb_se4_right
    )
    return int(camera.initial_camera_settings.rotate_number_of_times)


def _pitch_shift_pixels(frame_height: int, delta_deg: float, focal: float) -> int:
    """How far to move a crop window to synthesise `delta_deg` of extra pitch.

    `cameras._pitch_shift_pixels`, with the focal length taken from the robot's
    own calibration rather than the sim camera's: a rectified frame is a pinhole
    projection, so a direction `delta` degrees off the optical axis lands
    `f * tan(delta)` pixels from the centre, and moving the window there re-centres
    the view on it. Positive is "look further up", and rows increase downward.

    The head camera is bolted to the shell at a fixed 43 degrees
    (`setups.HEAD_CAMERA_PITCH_DEG`) -- there is no head tilt joint on an SE4 --
    which is the whole reason this exists: the fisheye sees far more than the
    window the checkpoint wants, so a view from another pitch can be cut out of
    the frame the hardware already produces.
    """
    return int(round(-focal * math.tan(math.radians(delta_deg))))


@dataclass(frozen=True)
class HeadCameraOptions:
    """What to do to the head frames between the lens and the policy."""

    rectify: bool = False
    """`stretch_rectified` rather than `stretch_fisheye`. See `setups.SETUPS`."""

    crop_to: tuple[int, int] | None = None
    """Cut a window of this size out of the upright frame, or None for all of it.

    None is `stretch_fisheye`'s own default -- `ExoCameraParams.crop_to` is unset
    there -- which hands the policy the whole 400x640 portrait frame. Crop to
    `DROID_FRAME_SIZE` to give it the landscape shape it was trained on, at the
    cost of the field of view that made the fisheye worth having.
    """

    virtual_pitch_deg: float | None = None
    """Re-centre the crop as though the camera were pitched here instead.

    Only means anything with `crop_to` set; there is nothing to move the window
    within otherwise. Measured from `HEAD_CAMERA_PITCH_DEG`, so 43 is no change.
    """


def head_frame_for_policy(
    frame: np.ndarray,
    head_side: str,
    options: HeadCameraOptions,
    camera_matrix: np.ndarray | None,
    distortion: np.ndarray | None,
) -> np.ndarray:
    """A raw head frame, put through the stages a sim rollout's exo camera goes through.

    `cameras.postprocess_exo_frame` in the order the hardware imposes it, minus
    the first stage: a rendered frame has the fisheye *applied* to it to make it
    look like this camera, and this one came out of that camera already. What is
    left is the same -- rectify in the sensor frame, stand the frame up, crop --
    and the crop is `cameras._centre_crop_resize` itself rather than a copy of
    it, so a real frame and a rendered one are framed identically.
    """
    if options.rectify:
        if camera_matrix is None or distortion is None:
            raise ValueError(
                "--head-mode rectified needs the head camera's calibration, and this "
                "stream carries none. Run the robot's sender from a checkout whose "
                "`head_camera_matrix` is published, or use --head-mode fisheye."
            )
        frame = _rectify_fisheye(frame, camera_matrix, distortion)

    # After the rectification, which is expressed in the sensor's own frame, and
    # before the crop, which is a window on the upright view.
    turns = _upright_quarter_turns(head_side)
    if turns:
        frame = np.rot90(frame, turns)

    if options.crop_to is None:
        return np.ascontiguousarray(frame)

    shift = 0
    if options.virtual_pitch_deg is not None:
        # The sensor is sideways, so its *horizontal* focal length -- `fx` -- is
        # the one that governs the upright frame's vertical axis.
        if camera_matrix is not None:
            # The stream's calibration describes the frames the stream carries,
            # so it is already in this frame's pixels.
            focal = float(camera_matrix[0, 0])
        else:
            # The fleet calibration instead, which is for the full 1920x1200
            # sensor: scale it by how tall this frame came out, since the upright
            # frame's height is the sensor's width.
            settings = StretchCameras.cam_nav_rgb_se4_right.initial_camera_settings
            focal = settings.focal[0] * (frame.shape[0] / float(settings.width))
        shift = _pitch_shift_pixels(
            frame.shape[0], float(options.virtual_pitch_deg) - HEAD_CAMERA_PITCH_DEG, focal
        )
    return _centre_crop_resize(np.ascontiguousarray(frame), options.crop_to, shift)


def wrist_frame_for_policy(
    frame: np.ndarray, jaw_flipped: bool, keep_flipped_frame: bool, fov_deg: float = 0.0
) -> np.ndarray:
    """A raw wrist frame, framed and oriented the way the checkpoint reads it.

    Two corrections, both of which `StretchMolmoBotDroidPolicy` applies to a
    rendered frame:

    * **The half turn.** Holding the jaw half over rolls the wrist camera 180
      degrees about its own axis, and the checkpoint was trained on a Franka
      whose hand is not turned over -- hand it an upside-down frame and every
      lateral correction comes back inverted. See
      `StretchMolmoBotDroidPolicy._wrist_camera` for the measurement, and for
      what turning it back still leaves wrong.
    * **The framing.** The hardware streams 640x400 and a sim rollout hands the
      policy 656x368 (`WRIST_OUTPUT_SIZE`), so the frame is cropped to that
      aspect rather than squashed into it.

    `fov_deg` narrows the view first, which is `RetargetParams.wrist_fov_deg` as
    it exists on hardware: Stretch's camera sits further back along a longer hand
    than the Robotiq's, so an object at the grasp point subtends 1.55x less of
    the frame, and cropping into the centre is how a real camera is narrowed.
    `setups.MATCHED_WRIST_FOV_DEG` is the angle at which the two match.
    """
    if fov_deg and fov_deg < WRIST_CAMERA_FOV_DEG:
        scale = math.tan(math.radians(fov_deg / 2.0)) / math.tan(
            math.radians(WRIST_CAMERA_FOV_DEG / 2.0)
        )
        height, width = frame.shape[:2]
        crop_height = max(1, int(round(height * scale)))
        crop_width = max(1, int(round(width * scale)))
        top = (height - crop_height) // 2
        left = (width - crop_width) // 2
        frame = frame[top : top + crop_height, left : left + crop_width]

    if jaw_flipped and not keep_flipped_frame:
        frame = np.rot90(frame, 2)
    return _centre_crop_resize(np.ascontiguousarray(frame), WRIST_OUTPUT_SIZE)


# =============================================================================
# The mirror: the real robot's pose, in a MuJoCo model
# =============================================================================


def build_mirror() -> tuple[Any, MjData, Stretch4RobotView, str]:
    """A Stretch 4 standing on an empty floor, for the kinematics to be solved on.

    No house and no objects: nothing in this model is ever stepped, rendered or
    collided. It exists because `StretchArmIK` differentiates the forward
    kinematics on an `MjData` and `FrankaOnStretchView.franka_joint_pos` reads
    the tool pose out of one, and both want the configuration the real robot is
    in -- which `RobotMirror.sync` writes in each step.

    The robot is spawned at the origin. The base's *reported* pose is written in
    from the robot's odometry with everything else, so where it is spawned only
    decides which pose the odometry's zero corresponds to.
    """
    spec = MjSpec()
    spec.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_PLANE, size=[10.0, 10.0, 0.1], name="floor"
    )
    config = Stretch4RobotConfig()
    namespace = config.robot_namespace
    Stretch4Robot.add_robot_to_scene(
        config, spec, prefix=namespace, pos=[0.0, 0.0, 0.0], quat=[1.0, 0.0, 0.0, 0.0]
    )
    Stretch4Robot.apply_control_overrides(spec, config)

    model = spec.compile()
    data = MjData(model)
    view = Stretch4RobotView(data, namespace)
    view.set_qpos_dict(dict(config.init_qpos))
    mujoco.mj_forward(model, data)
    return model, data, view, namespace


class GripperUnits:
    """Converts between the robot's gripper percentage and the model's finger angle.

    The two ends measure the same hand differently: `stretch_gripper` is
    commanded and reported in percent over its servo sweep, and the MJCF carries
    two finger joints running 0 (shut) to 0.5 rad (wide open, 188mm between the
    tips).

    Mapped as a *fraction of the hand's travel* rather than metre for metre,
    which is the same choice `digital_twin.GripperMirror` makes for the same
    reason: what the policy reads on this channel is where the hand is between
    shut and open (`_ProxyGripperGroup.joint_pos` normalises it again on the way
    into Robotiq units), and the two stacks' aperture constants disagree by a
    centimetre at the open end -- 0.1885 m measured on the compiled model against
    `robot_settings_se4`'s hand-measured 0.177. A fraction is exact at both ends
    of the travel and within that centimetre in between; a metre-for-metre map is
    exact nowhere and pretends otherwise.

    `open_pct` comes off the robot's own parameters, because it is the one number
    that differs between tools and calibrations.
    """

    def __init__(self, open_pct: float, closed_pct: float = STRETCH_GRIPPER_CLOSED_PCT) -> None:
        self.open_pct = float(open_pct)
        self.closed_pct = float(closed_pct)

    @classmethod
    def nominal(cls) -> "GripperUnits":
        """The percentages the model's own settings imply, for a run with no robot.

        `--dry-run` has no `robot_params` to read, and the answer is not 100:
        `stretch_gripper`'s percentage is scaled so that the *closed* end is -100,
        which puts fully open at `100 * |open / closed|` of the servo sweep --
        +300 on an SE4, whose sweep is -100 to +300 degrees
        (`robot_settings_se4`, and `stretch_body`'s `StretchGripper.pct_max_open`,
        which is the same arithmetic). Assuming 100 would have the mirror read a
        wide open hand as a third open.
        """
        closed_deg, open_deg = robot_settings_se4["stretch_gripper"]["gripper_servo_range_deg"]
        return cls(open_pct=100.0 * abs(open_deg / closed_deg) if closed_deg else 100.0)

    @classmethod
    def from_robot(cls, robot: Any) -> "GripperUnits":
        """The percentages this robot reports at each end of its travel.

        `range_deg` is the servo sweep either side of the fingers touching, and
        percent is that sweep scaled so -100 is fully shut -- so the open end is
        100 * |open / closed|, exactly as `digital_twin.GripperMirror` derives it.
        """
        params = getattr(robot, "robot_params", {}) or {}
        range_deg = params.get(GRIPPER_JOINT, {}).get("range_deg")
        if range_deg and range_deg[0]:
            return cls(open_pct=100.0 * abs(range_deg[1] / range_deg[0]))
        return cls(open_pct=100.0)

    def fraction(self, pct: float) -> float:
        span = self.open_pct - self.closed_pct
        if not span:
            return 0.0
        return float(np.clip((pct - self.closed_pct) / span, 0.0, 1.0))

    def finger_rad(self, pct: float) -> float:
        """Where the model's fingers are, for a gripper the robot reports at `pct`."""
        return fr.STRETCH_FINGER_CLOSED + self.fraction(pct) * (
            fr.STRETCH_FINGER_OPEN - fr.STRETCH_FINGER_CLOSED
        )

    def pct(self, finger_rad: float) -> float:
        """What to tell the robot, for a retargeted finger angle."""
        span = fr.STRETCH_FINGER_OPEN - fr.STRETCH_FINGER_CLOSED
        fraction = float(np.clip((finger_rad - fr.STRETCH_FINGER_CLOSED) / span, 0.0, 1.0))
        return self.closed_pct + fraction * (self.open_pct - self.closed_pct)


class RobotMirror:
    """Keeps the MuJoCo model standing where the real robot is standing.

    One direction only. Nothing in this class reads the model to command the
    robot -- that is `RobotCommander` -- so the mirror can never drive anything;
    it is the robot's measured state expressed as an `MjData`, which is the form
    the retargeting needs it in.

    The wrist roll is the one axis whose sign differs between the two: the URDF
    turns it the opposite way from the robot's servo convention, which
    `WRIST_ROLL_SIM_SIGN` records and `digital_twin.py` applies in the same
    place. Getting it wrong is not subtle -- the hand rolls the wrong way -- but
    it is invisible until something moves.
    """

    def __init__(self, view: Stretch4RobotView, gripper: GripperUnits) -> None:
        self.view = view
        self.gripper = gripper
        self.data = view.mj_data
        self.model = view.mj_data.model
        self._limits = {
            group: commandable_limits(view.get_move_group(group))
            for group in ("lift", "arm", "wrist", "gripper")
        }

    def sync(self, observation: RobotObservation) -> dict[str, float]:
        """Write one joint sample into the model. Returns what was written, flat."""
        lift = observation.joint("lift", "height")
        arm = observation.joint("arm", "extension")
        yaw = observation.joint("wrist_yaw", "angle")
        pitch = observation.joint("wrist_pitch", "angle")
        roll = observation.joint("wrist_roll", "angle")
        finger = self.gripper.finger_rad(observation.joint("gripper", "pos_pct"))

        qpos = {
            "base": [
                observation.joint("base_odometry", "x"),
                observation.joint("base_odometry", "y"),
                observation.joint("base_odometry", "theta"),
            ],
            # Clipped to the model's own travel, because a joint written outside
            # it puts the IK's seed outside the box it solves in and the first
            # solve then spends its step walking back to the edge.
            "lift": [self._clip("lift", 0, lift)],
            "arm": [self._clip("arm", 0, arm)],
            "wrist": [
                self._clip("wrist", 0, yaw),
                self._clip("wrist", 1, pitch),
                self._clip("wrist", 2, WRIST_ROLL_SIM_SIGN * roll),
            ],
            "gripper": [self._clip("gripper", 0, finger), self._clip("gripper", 1, finger)],
        }
        self.view.set_qpos_dict(qpos)
        # The tool pose and the Jacobians are read straight off `data` by the
        # retargeting, and `set_qpos_dict` only writes `qpos`.
        mujoco.mj_forward(self.model, self.data)
        return {
            "lift": lift,
            "arm": arm,
            "wrist_yaw": yaw,
            "wrist_pitch": pitch,
            "wrist_roll": roll,
            "gripper_pct": observation.joint("gripper", "pos_pct"),
            "gripper_effort": observation.joint("gripper", "effort"),
            "base_x": qpos["base"][0],
            "base_y": qpos["base"][1],
            "base_theta": qpos["base"][2],
        }

    def _clip(self, group: str, index: int, value: float) -> float:
        low, high = self._limits[group][index]
        return float(np.clip(value, low, high))


# =============================================================================
# Sending the retargeted targets to the robot
# =============================================================================


class RobotCommander:
    """Streams per-move-group targets to a `RobotClient`.

    Absolute positions at the robot's `max` motion profile, one `push_command()`
    per control step -- which is `stretch_puppet_teleop.py`'s recipe and the one
    `examples/digital_twin.py` documents at length. The reason it is positions
    and not deltas is worth restating here, because it is what makes a 15Hz
    stream move the robot at all: a relative command re-plans the trapezoidal
    profile from a standstill every cycle, so the joint spends each step in the
    opening of a ramp and crawls. An absolute target stays out ahead of the robot
    and it tracks.

    Every target is clamped to `MAX_TARGET_STEP` of where the joint currently is.
    That is a guard against one bad prediction, not a speed limit; see the
    constant.
    """

    def __init__(
        self,
        robot: Any,
        gripper: GripperUnits,
        step_limit_scale: float = 1.0,
        include_base: bool = False,
        control_period_s: float = 1.0 / CONTROL_HZ,
        speed_scale: float = 1.0,
    ) -> None:
        from examples.digital_twin import EOA_ACCELERATION_R, EOA_VELOCITY_R

        self.robot = robot
        self.gripper = gripper
        self.include_base = include_base
        self.control_period_s = control_period_s
        self.speed_scale = float(speed_scale)
        self.step_limits = {
            group: value * step_limit_scale for group, value in MAX_TARGET_STEP.items()
        }
        self._refused: set[str] = set()
        self.last_sent: dict[str, float] = {}

        # Read once. They come out of the robot's own parameters, which do not
        # change while it is up, and the reads walk a nested dict per joint. Kept
        # at full rate as well as scaled, because a stop is not a motion to make
        # gently -- see `hold`.
        self._lift_max = self._motion_limits("lift")
        self._arm_max = self._motion_limits("arm")
        self._eoa_max = (EOA_VELOCITY_R, EOA_ACCELERATION_R)
        self._lift_limits = self._scaled(self._lift_max)
        self._arm_limits = self._scaled(self._arm_max)
        self._eoa_limits = self._scaled(self._eoa_max)

    def _scaled(
        self, limits: tuple[float | None, float | None]
    ) -> tuple[float | None, float | None]:
        """`limits` at `speed_scale`. A `None` stays `None` -- see `_motion_limits`."""
        return tuple(None if value is None else value * self.speed_scale for value in limits)

    def _motion_limits(self, subsystem: str) -> tuple[float | None, float | None]:
        """`(velocity, acceleration)` from the joint's `max` profile, unscaled.

        `digital_twin.DigitalTwin._joint_motion_limits`, including the lift's
        backed-off acceleration -- the puppet teleop does not run the lift at its
        full number and neither should this.
        """
        from examples.digital_twin import LIFT_ACCELERATION_SCALE, MOTION_PROFILE

        params = getattr(getattr(self.robot, subsystem, None), "params", None)
        if not isinstance(params, dict):
            # `None` is not zero: `move_to` reads it as "no limit given" and uses
            # the joint's own default profile, which is what this had before it
            # could find the parameters. `--slow` cannot scale a number that was
            # never read, and says so rather than multiplying `None`.
            log.warning(
                f"[robot] no motion parameters for {subsystem}; it will move at its "
                "default profile, and --slow will not apply to it."
            )
            return None, None
        motion = params.get("motion", {}).get(MOTION_PROFILE, {})
        velocity, acceleration = motion.get("vel_m"), motion.get("accel_m")
        if subsystem == "lift" and acceleration is not None:
            acceleration *= LIFT_ACCELERATION_SCALE
        return velocity, acceleration

    def send(self, targets: dict[str, np.ndarray], measured: dict[str, float]) -> dict[str, float]:
        """One step's targets. Returns what was actually commanded, in robot units."""
        eoa_velocity, eoa_acceleration = self._eoa_limits
        commanded: dict[str, float] = {}

        if "lift" in targets:
            value = self._clamp("lift", float(np.ravel(targets["lift"])[0]), measured["lift"])
            velocity, acceleration = self._lift_limits
            self._check("lift", self.robot.lift.move_to(value, v_m=velocity, a_m=acceleration))
            commanded["lift"] = value

        if "arm" in targets:
            value = self._clamp("arm", float(np.ravel(targets["arm"])[0]), measured["arm"])
            velocity, acceleration = self._arm_limits
            self._check("arm", self.robot.arm.move_to(value, v_m=velocity, a_m=acceleration))
            commanded["arm"] = value

        if "wrist" in targets:
            wrist = np.ravel(np.asarray(targets["wrist"], dtype=float))
            # Back through the sign the mirror applied on the way in, so the
            # robot is told to roll the way the model is holding the hand.
            for index, (name, sign) in enumerate(
                (("wrist_yaw", 1.0), ("wrist_pitch", 1.0), ("wrist_roll", WRIST_ROLL_SIM_SIGN))
            ):
                value = self._clamp("wrist", sign * float(wrist[index]), measured[name])
                self._check(
                    name,
                    self.robot.end_of_arm.move_to(name, value, eoa_velocity, eoa_acceleration),
                )
                commanded[name] = value

        if "gripper" in targets:
            finger = float(np.mean(np.asarray(targets["gripper"], dtype=float)))
            measured_finger = self.gripper.finger_rad(measured["gripper_pct"])
            finger = self._clamp("gripper", finger, measured_finger)
            value = self.gripper.pct(finger)
            self._check(
                GRIPPER_JOINT,
                self.robot.end_of_arm.move_to(
                    GRIPPER_JOINT, value, eoa_velocity, eoa_acceleration
                ),
            )
            commanded["gripper_pct"] = value

        if self.include_base and "base" in targets:
            goal = np.ravel(np.asarray(targets["base"], dtype=float))
            commanded.update(self._send_base(goal, measured))

        self.robot.push_command()
        self.last_sent = commanded
        return commanded

    def _send_base(self, goal: np.ndarray, measured: dict[str, float]) -> dict[str, float]:
        """A base pose target, as the velocity that would close the gap this step.

        The IK hands back an absolute pose and this base steers three omniwheels,
        so there is nothing absolute to command: the same conversion
        `live_policy.apply_action` makes, rotated into the base's own frame,
        capped at `MAX_BASE_SPEED_MPS` / `MAX_BASE_TURN_RADPS`.
        """
        yaw = measured["base_theta"]
        delta = goal[:2] - np.array([measured["base_x"], measured["base_y"]])
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        forward = (cos_yaw * delta[0] + sin_yaw * delta[1]) / self.control_period_s
        left = (-sin_yaw * delta[0] + cos_yaw * delta[1]) / self.control_period_s
        turn = math.atan2(math.sin(goal[2] - yaw), math.cos(goal[2] - yaw)) / self.control_period_s

        forward = float(np.clip(forward, -MAX_BASE_SPEED_MPS, MAX_BASE_SPEED_MPS))
        left = float(np.clip(left, -MAX_BASE_SPEED_MPS, MAX_BASE_SPEED_MPS))
        turn = float(np.clip(turn, -MAX_BASE_TURN_RADPS, MAX_BASE_TURN_RADPS))
        self.robot.omnibase.set_velocity(forward, left, turn)
        return {"base_forward": forward, "base_left": left, "base_turn": turn}

    def hold(self, measured: dict[str, float] | None = None) -> None:
        """Stop the robot where it is now, rather than where it was last sent.

        Not sending is not stopping. Every target streamed here is absolute and
        deliberately out ahead of the robot -- that is what makes it track -- so a
        loop that simply stops sending leaves the arm finishing the last reach it
        was given, which is exactly the motion you pressed a key to end.
        Commanding each joint to its *measured* position instead turns the
        remaining travel into a decelerate-and-hold at the robot's own profile.

        **The hand is left alone.** Its measured position while it is squeezing
        is not the target holding the object -- re-commanding it would ease the
        grip, and a stop that drops what the robot is carrying is worse than the
        motion it stopped. The fingers keep the last target they were given.

        `measured` comes from the mirror's last sync; without one (a stop before
        any frame arrived) there is nothing to hold at and only the base is told.

        **At full rate, whatever `--slow` says.** `--slow` is about how briskly
        the robot goes about the task; a stop is not part of the task, and a
        deceleration scaled down by the same factor is a stop that takes two and a
        half times as long to arrive. The targets here are where the joints
        already are, so the full profile is a deceleration and not a lurch.
        """
        try:
            eoa_velocity, eoa_acceleration = self._eoa_max

            if measured is not None:
                velocity, acceleration = self._lift_max
                self.robot.lift.move_to(measured["lift"], v_m=velocity, a_m=acceleration)
                velocity, acceleration = self._arm_max
                self.robot.arm.move_to(measured["arm"], v_m=velocity, a_m=acceleration)
                for name in ("wrist_yaw", "wrist_pitch", "wrist_roll"):
                    self.robot.end_of_arm.move_to(
                        name, measured[name], eoa_velocity, eoa_acceleration
                    )
            if self.include_base:
                self.robot.omnibase.set_velocity(0.0, 0.0, 0.0)
            self.robot.push_command()
        except Exception as error:  # noqa: BLE001 - stopping must not raise on the way out
            log.warning(f"[robot] could not stop the robot: {error!r}")

    def _clamp(self, group: str, target: float, measured: float) -> float:
        limit = self.step_limits[group]
        return float(np.clip(target, measured - limit, measured + limit))

    def _check(self, name: str, accepted: Any) -> None:
        """Complain once when the robot refuses a command.

        `RobotClient`'s movement calls *return* False rather than raising, and
        log to the robot's own logger -- so a dropped return value is how a run
        ends up looking connected while nothing moves. Once, because this is
        called five times a step.
        """
        if accepted is False and name not in self._refused:
            self._refused.add(name)
            click.secho(
                f"The robot refused a move_to on {name}. It is usually not homed, the "
                "target is outside the joint's range, or the tool does not have that joint.",
                fg="red",
            )


FLEET_HELP = """stretch4_body could not read the robot's parameters.

`RobotClient` builds them locally even when it is talking to a robot over the
network -- `RobotParams` reads the *fleet directory* at import time -- so a
workstation needs a copy of that robot's fleet directory and the two environment
variables that name it. On the robot they are already in the shell; here they are
not, which is what this failure is.

    # on the robot, to find out what they are
    echo $HELLO_FLEET_PATH $HELLO_FLEET_ID
    # e.g. /home/hello-robot/stretch_user stretch-se4-3001

    # on this workstation, once
    scp -r 'hello-robot@{robot}:$HELLO_FLEET_PATH/$HELLO_FLEET_ID' ~/stretch_user/
    export HELLO_FLEET_PATH=~/stretch_user
    export HELLO_FLEET_ID=<the fleet id>

(the quotes matter: those two variables are the robot's, and an unquoted remote
path is expanded by this shell, which does not have them.)

`--fleet-path ~/stretch_user --fleet-id <the fleet id>` sets the same two for one
run. `--dry-run` needs none of it: it runs the cameras, the mirror and the
retargeting, and moves nothing."""


def prepare_fleet_environment(fleet_path: str | None, fleet_id: str | None) -> None:
    """Name the robot's fleet directory, before anything imports `stretch4_body`.

    The variables have to be in the environment by the time
    `stretch4_body.core.robot_params` is imported, because it reads them at class
    definition time -- so this is a `os.environ` write rather than an argument,
    and it has to happen before `connect_robot`.

    With a path and no id, the id is inferred when exactly one directory under it
    looks like a fleet directory. One candidate is an answer; several is a choice
    this cannot make for you.
    """
    if not fleet_path and not fleet_id:
        return
    if fleet_path:
        os.environ["HELLO_FLEET_PATH"] = str(Path(fleet_path).expanduser())
    if fleet_id:
        os.environ["HELLO_FLEET_ID"] = fleet_id
        click.echo(f"  fleet      : {os.environ['HELLO_FLEET_PATH']}/{fleet_id}")
        return

    # A path and no id. The id is *always* inferred from that path rather than
    # left at whatever is in the environment, because by the time this runs
    # `ensure_fleet_directory` has put its stand-in there -- and a run that names
    # a real fleet directory and silently keeps the stand-in's id would read the
    # stand-in.
    root = Path(os.environ["HELLO_FLEET_PATH"])
    candidates = [
        child.name
        for child in sorted(root.glob("*"))
        if child.is_dir() and (child / "stretch_configuration_params.yaml").exists()
    ]
    if len(candidates) == 1:
        os.environ["HELLO_FLEET_ID"] = candidates[0]
        click.echo(f"  fleet      : {root}/{candidates[0]}")
    elif candidates:
        raise SystemExit(
            f"{root} holds several fleet directories ({', '.join(candidates)}). "
            "Name the one this robot is with --fleet-id."
        )
    else:
        raise SystemExit(
            f"{root} holds no fleet directory -- nothing under it has a "
            "stretch_configuration_params.yaml. Check --fleet-path."
        )


def connect_robot(robot_ip: str | None) -> Any:
    """Start a `RobotClient`, and refuse to go on with a robot that is not homed.

    `examples/digital_twin.py._connect` with the homing check folded in, because
    an un-homed robot here does not fail loudly: every `move_to` returns False,
    the loop carries on computing actions, and the robot stands still while the
    terminal fills with a policy running normally.
    """
    try:
        from stretch4_body.robot.robot_client import RobotClient
    except ImportError as error:
        raise SystemExit(
            f"stretch4_body is required to command the robot ({error}).\n"
            'Install it with: uv pip install -e ".[digital-twin]", or pass --dry-run '
            "to run everything except the motion."
        )
    except KeyError as error:
        # Not an ImportError: `stretch4_body` is installed and reads the fleet
        # environment while it is being imported, so a workstation without it
        # fails here with a bare `KeyError: 'HELLO_FLEET_PATH'` from four frames
        # inside somebody else's module.
        if str(error).strip("\"'") not in ("HELLO_FLEET_PATH", "HELLO_FLEET_ID"):
            raise
        raise SystemExit(FLEET_HELP.format(robot=robot_ip or "<robot>"))

    where = f"tcp://{robot_ip}" if robot_ip else "the local robot"
    click.echo(f"Connecting to {where}...")
    robot = RobotClient(ip_address=robot_ip)
    # Over the network, `allow_different_user_connection` is not optional. The
    # check it skips asks whether the *local* server socket belongs to this user
    # -- `is_server_owned_by_current_user` stats `/tmp/stretch_zmq/port_admin`,
    # an ipc path that only exists on the robot -- so on a workstation it does
    # not return False, it raises `FileNotFoundError` from four frames inside
    # `pathlib`. The connection itself has already been made and verified by the
    # time it runs. Left on for a local robot, where it is a real check against
    # taking a session another user on that machine is holding.
    started = (
        robot.startup(allow_different_user_connection=True) if robot_ip else robot.startup()
    )
    if not started:
        raise SystemExit(
            f"Failed to start the RobotClient. Is the Stretch Body Server running on {where}?"
        )
    robot.pull_status(blocking=True)
    if not robot.is_homed():
        robot.stop()
        raise SystemExit("The robot is not fully homed. Home it first, then rerun.")
    return robot


# =============================================================================
# Telemetry
# =============================================================================


class RerunTelemetry:
    """Streams the run to a Rerun viewer, and optionally to an `.rrd` file.

    Three things, laid out so a failure is readable without scrolling: what the
    policy is looking at (the processed frames, which is not what the cameras
    produced), what the robot is doing (measured against commanded, per joint),
    and how well the retargeting is coping (the residual, and how far behind the
    stream the loop is running).

    The raw camera frames are logged too, beside the processed ones. Most of what
    goes wrong on a first real run is in the gap between those two rows -- a
    frame turned the wrong way, a channel swap, a crop that cut the workspace out
    -- and side by side it is one glance.
    """

    def __init__(
        self, save_path: Path | None = None, spawn: bool = True, jpeg_quality: int = 75
    ) -> None:
        import rerun as rr
        import rerun.blueprint as rrb

        self._rr = rr
        self.jpeg_quality = int(jpeg_quality)
        rr.init("Stretch4 retargeted DROID policy", spawn=spawn and save_path is None)
        if save_path is not None:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            rr.save(str(save_path))
        rr.send_blueprint(
            rrb.Horizontal(
                rrb.Vertical(
                    rrb.Horizontal(
                        rrb.Spatial2DView(origin="policy/head", name="head (to the policy)"),
                        rrb.Spatial2DView(origin="policy/wrist", name="wrist (to the policy)"),
                    ),
                    rrb.Horizontal(
                        rrb.Spatial2DView(origin="camera/head", name="head (raw)"),
                        rrb.Spatial2DView(origin="camera/wrist", name="wrist (raw)"),
                    ),
                    rrb.TextDocumentView(origin="run/task", name="instruction"),
                    row_shares=[2, 2, 1],
                ),
                rrb.Vertical(
                    rrb.TimeSeriesView(origin="robot/measured", name="where the robot is"),
                    rrb.TimeSeriesView(origin="robot/commanded", name="what it was told"),
                    rrb.TimeSeriesView(origin="retarget", name="retargeting and timing"),
                ),
                column_shares=[3, 2],
            )
        )
        self._task: str | None = None

    def _image(self, frame: np.ndarray) -> Any:
        """One frame, JPEG-encoded unless asked for raw.

        Four images a step at 15Hz is about 45 MB/s of raw pixels, which fills a
        recording faster than a run lasts and makes the viewer stutter on a
        machine that is also running the policy. At quality 75 it is nearer
        3 MB/s and nothing about what these panels are for is lost.
        `--rerun-jpeg-quality 0` keeps the pixels, for when the question is about
        the pixels.
        """
        image = self._rr.Image(frame)
        return image if self.jpeg_quality <= 0 else image.compress(jpeg_quality=self.jpeg_quality)

    def log_step(
        self,
        step: int,
        images: dict[str, np.ndarray],
        raw: dict[str, np.ndarray | None],
        measured: dict[str, float],
        commanded: dict[str, float],
        diagnostics: dict[str, float],
        task: str,
    ) -> None:
        rr = self._rr
        rr.set_time("policy_step", sequence=step)
        rr.set_time("wall_time", timestamp=time.time())

        rr.log("policy/head", self._image(images["head"]))
        rr.log("policy/wrist", self._image(images["wrist"]))
        for name, frame in raw.items():
            if frame is not None:
                rr.log(f"camera/{name}", self._image(frame))

        for key, value in measured.items():
            rr.log(f"robot/measured/{key}", rr.Scalars(float(value)))
        for key, value in commanded.items():
            rr.log(f"robot/commanded/{key}", rr.Scalars(float(value)))
        for key, value in diagnostics.items():
            rr.log(f"retarget/{key}", rr.Scalars(float(value)))

        if task != self._task:
            self._task = task
            rr.log("run/task", rr.TextDocument(task))


# =============================================================================
# The console
# =============================================================================


STOP_KEYS = (" ", "\r", "\n")
"""Keys that stop the robot the moment they are pressed, with nothing typed yet.

Space and Enter, because a stop you have to finish typing and confirm is not a
stop -- the arm is moving while you spell it. Neither is lost as an input
character: Enter still ends an instruction you have started typing, and a space
inside one is a space (see `Console._key_loop`), so "Pick up the cup" types
normally and only a space at the *start* of an empty prompt means stop.
"""


class Console:
    """Reads what you type, a key at a time where the terminal allows it.

    On its own thread rather than polled from the control loop, because reading
    stdin blocks and the loop must not: the robot is being held by a stream of
    position targets, and a loop that stops to wait for a line stops feeding it.

    Two modes, and the difference is how quickly a stop lands:

    * **A terminal.** stdin goes into cbreak mode, so a keystroke arrives here
      when it is pressed rather than when the line is finished. `STOP_KEYS` stop
      the robot immediately; any other printable key starts an instruction, which
      is sent on Enter. Backspace edits it, Ctrl-D ends the run.
    * **Anything else** -- a pipe, a redirect, a terminal that will not give up
      canonical mode: plain `readline`, where an empty line is the stop. The
      commands are identical; only the keystroke shortcut is missing.

    Characters are echoed by hand because cbreak turns the terminal's own echo
    off. `tty.setcbreak` leaves `ISIG` alone, so Ctrl-C still ends the run the way
    it does everywhere else.
    """

    HELP = (
        "Type an instruction to run it. While one is running:\n"
        "  space / enter    stop the arm where it is, ready for the next instruction\n"
        "  stop / s         the same, spelled out\n"
        "  go / g           carry on with the current instruction\n"
        "  home             move back to the Franka's home pose and hold\n"
        "  ?                this message\n"
        "  quit / q         stop the robot and exit"
    )

    def __init__(self) -> None:
        self.lines: queue.Queue[str] = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._terminal: tuple[int, Any] | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="console", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._restore_terminal()
        if self._thread is not None:
            self._thread.join(timeout=0.5)

    # -- the two input modes -----------------------------------------------

    def _loop(self) -> None:
        if self._enter_cbreak():
            try:
                self._key_loop()
            finally:
                self._restore_terminal()
        else:
            self._line_loop()

    def _enter_cbreak(self) -> bool:
        """Put the terminal in cbreak mode, or say that this stdin cannot be.

        False for a pipe, a redirect, a `nohup`, or a platform without termios --
        each of which is a perfectly good way to run this, just without the
        single-key stop.
        """
        try:
            import termios
            import tty

            if not sys.stdin.isatty():
                return False
            descriptor = sys.stdin.fileno()
            self._terminal = (descriptor, termios.tcgetattr(descriptor))
            tty.setcbreak(descriptor)
            return True
        except Exception as error:  # noqa: BLE001 - falling back is not a failure
            log.debug(f"[console] reading lines rather than keys: {error!r}")
            self._terminal = None
            return False

    def _restore_terminal(self) -> None:
        """Hand the terminal back, exactly once.

        Not tidiness: a terminal left in cbreak echoes nothing and buffers
        nothing after this process exits, which looks like a broken shell.
        """
        if self._terminal is None:
            return
        descriptor, settings = self._terminal
        self._terminal = None
        try:
            import termios

            termios.tcsetattr(descriptor, termios.TCSADRAIN, settings)
        except Exception as error:  # noqa: BLE001
            log.debug(f"[console] could not restore the terminal: {error!r}")

    def _key_loop(self) -> None:
        """One keystroke at a time, assembling an instruction as it goes."""
        import select

        typed: list[str] = []
        while not self._stop.is_set():
            # A timeout rather than a blocking read, so `stop()` ends this thread
            # instead of leaving it holding a terminal nobody is typing at.
            if not select.select([sys.stdin], [], [], 0.2)[0]:
                continue
            try:
                key = sys.stdin.read(1)
            except Exception:  # noqa: BLE001 - a closed stdin ends the console
                key = ""
            if not key or key == "\x04":  # EOF, or Ctrl-D
                self.lines.put("quit")
                return

            if key in ("\r", "\n"):
                if typed:
                    self._echo("\n")
                    self.lines.put("".join(typed).strip())
                    typed.clear()
                else:
                    self._echo("\n")
                    self.lines.put("stop")
            elif key == " " and not typed:
                # Only on an empty prompt. Inside an instruction a space is a
                # space, which is what makes this usable for typing at all.
                self._echo("\n")
                self.lines.put("stop")
            elif key in ("\x7f", "\b"):
                if typed:
                    typed.pop()
                    # Back over the character, blank it, and back over it again:
                    # a bare backspace only moves the cursor.
                    self._echo("\b \b")
            elif key.isprintable():
                typed.append(key)
                self._echo(key)

    def _line_loop(self) -> None:
        """Whole lines, for a stdin that is not a terminal. An empty one stops."""
        while not self._stop.is_set():
            try:
                line = sys.stdin.readline()
            except Exception:  # noqa: BLE001 - a closed stdin ends the console, not the run
                line = ""
            if not line:
                self.lines.put("quit")
                return
            self.lines.put(line.strip() or "stop")

    @staticmethod
    def _echo(text: str) -> None:
        """Put a typed character on the screen, since cbreak mode will not.

        Straight to `sys.stdout` rather than through `click.echo`, because this
        runs per keystroke and has to leave the cursor where it found it.
        """
        try:
            sys.stdout.write(text)
            sys.stdout.flush()
        except Exception:  # noqa: BLE001 - a closed stdout is not worth ending a run over
            pass


# =============================================================================
# The run
# =============================================================================


@dataclass
class RunSettings:
    """Everything the loop is parameterised by, so `main` stays a wiring function."""

    task: str = ""
    control_hz: float = CONTROL_HZ
    max_obs_age: float = 0.5
    head: HeadCameraOptions = field(default_factory=HeadCameraOptions)
    wrist_fov_deg: float = 0.0


class RealStretchRunner:
    """One control loop: stream in, retargeted targets out.

    Holds the three pieces of state a step needs and nothing else -- the mirror
    (where the robot is), the proxy (the retargeting) and the policy (the
    checkpoint and its action chunk) -- so that `step` reads as the chain the
    module docstring describes.
    """

    def __init__(
        self,
        stream: RobotStream,
        mirror: RobotMirror,
        proxy: fr.FrankaOnStretchView,
        policy: Any,
        commander: RobotCommander | None,
        telemetry: RerunTelemetry | None,
        settings: RunSettings,
    ) -> None:
        self.stream = stream
        self.mirror = mirror
        self.proxy = proxy
        self.policy = policy
        self.commander = commander
        self.telemetry = telemetry
        self.settings = settings

        self.task = settings.task
        self.running = bool(settings.task)
        self.step_count = 0
        self.residuals: list[tuple[float, float]] = []
        self.last_measured: dict[str, float] | None = None
        """The most recent joint sample written into the mirror, which is what a
        stop holds the arm at. See `RobotCommander.hold`."""

        self._warned: set[str] = set()

    # -- episodes ----------------------------------------------------------

    def new_episode(self, task: str) -> None:
        """Start driving towards `task`, from wherever the robot is now.

        The nearest thing to a reset there is on hardware. Both halves matter:
        the policy drops the action chunk and observation history it was
        executing for the previous instruction, and the proxy re-seeds the
        retargeting from the robot's current pose and re-centres the base's leash
        on where it is standing -- left over from the last instruction, both
        describe a robot that has since moved.
        """
        observation = self._current_observation()
        if observation is not None:
            self.last_measured = self.mirror.sync(observation)
        self.proxy.set_franka_mount_pose(
            fr.franka_mount_pose_from_base(self.mirror.view.get_move_group("base").joint_pos)
        )
        self.proxy.reset()
        self.policy.reset()
        self.residuals = []
        self.task = task
        self.running = True
        click.secho(f"[policy] running: {task}", fg="green")

    def pause(self, reason: str = "") -> None:
        """Stop sending, and say so once.

        Only on the transition: a stream that has gone quiet reaches this every
        step until it comes back, and both the message and the base's stop
        command are things to do once rather than fifteen times a second.
        """
        if not self.running:
            return
        click.secho(f"[policy] holding{f' ({reason})' if reason else ''}", fg="yellow")
        self.running = False
        if self.commander is not None:
            self.commander.hold(self.last_measured)

    def resume(self) -> None:
        if not self.task:
            click.secho("No instruction yet. Type one.", fg="yellow")
            return
        self.new_episode(self.task)

    def go_home(self) -> None:
        """Move the robot to the pose the Franka's home tool pose retargets to.

        The policy's first observation is read against a Franka at its home
        configuration; from an arbitrary Stretch pose that reads as an arm two
        radians from where the checkpoint expects it, and the first chunk is
        spent correcting rather than reaching. `snap_to_franka_joint_pos` solves
        for the configuration that matches it and *writes* it into the mirror --
        which on hardware is a pose to drive to, so it is written, read back out
        and commanded as one slow motion.
        """
        if self.commander is None:
            click.secho("--dry-run: not moving to the home pose.", fg="yellow")
            return
        observation = self._current_observation()
        if observation is None:
            click.secho("No frames from the robot yet; not moving.", fg="red")
            return
        measured = self.last_measured = self.mirror.sync(observation)
        residual = self.proxy.snap_to_franka_joint_pos()
        targets = {
            group: np.asarray(self.mirror.view.get_move_group(group).joint_pos, dtype=float)
            for group in ("lift", "arm", "wrist", "gripper")
        }
        # Which branch the wrist settled in, because the difference is a hand
        # turned most of the way over and it is the first thing about the home
        # pose anyone notices. Under `jaw_mode="auto"` it is the solver's choice
        # rather than anything asked for at the command line -- see `--jaw-mode`.
        branch = "flipped" if self.proxy.jaw_flipped else "upright"
        click.echo(
            f"Moving to the Franka home pose. Jaw: {branch} "
            f"({self.proxy.jaw_mode}). Residual (dx dy dz | drx dry drz): "
            f"{np.round(residual, 4).tolist()}"
        )
        self._report_start_pose()
        click.echo(f"  targets: {({k: np.round(v, 3).tolist() for k, v in targets.items()})}")

        # Streamed at the control rate rather than sent once, for the reason
        # every other command here is: an absolute target has to be re-sent to
        # stay ahead of the robot, and `RobotCommander`'s step clamp is measured
        # against where the joint currently is -- so a move longer than the clamp
        # is made of several, each one measured afresh.
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            observation = self._current_observation()
            if observation is not None:
                measured = self.last_measured = self.mirror.sync(observation)
            self.commander.send(targets, measured)
            if self._at_targets(targets, measured):
                break
            time.sleep(1.0 / self.settings.control_hz)
        else:
            click.secho("The robot did not reach the home pose within 15s.", fg="yellow")

        # The mirror is now back on the robot rather than on the pose that was
        # asked for, and the proxy is re-seeded from it.
        observation = self._current_observation()
        if observation is not None:
            self.last_measured = self.mirror.sync(observation)
        self.proxy.reset()
        self.pause("at the home pose")

    def _report_start_pose(self) -> None:
        """Say how far the start pose is from the Franka's home, in the policy's own units.

        The residual above is a *tool pose* error, and it is not what the
        checkpoint reads. What it reads is seven Franka joint angles out of
        `franka_joint_pos()`, and the question that decides whether an episode
        begins where the policy expects is how far those are from the home
        configuration it was trained to start from -- measured against
        `relative_max_joint_delta`, the per-joint bound its own action scaling
        applies, because a joint further out than that is one the first chunk
        spends steps walking back rather than reaching with.

        Expect one joint to be out and the rest to be close. Stretch's wrist roll
        runs into its stop holding the Franka's home orientation, and the shortfall
        lands almost entirely in the Franka's last joint -- a roll about the
        approach axis, which is the axis a parallel jaw cares least about.
        """
        reported = np.asarray(self.proxy.get_move_group("arm").joint_pos, dtype=float)
        home = np.asarray(self.proxy.franka.init_qpos, dtype=float)
        delta = np.abs(reported - home)
        clamp = np.asarray(
            getattr(self.policy, "relative_max_joint_delta", 0.2), dtype=float
        ).reshape(-1)
        clamp = float(clamp[0]) if clamp.size else 0.2
        worst = int(np.argmax(delta))
        over = int(np.sum(delta > clamp))
        click.secho(
            f"  the policy reads the arm {delta.max():.3f} rad from the Franka's home at "
            f"its worst joint (fr3_joint{worst + 1}); {over} of 7 are beyond its "
            f"{clamp:.2f} rad per-step clamp.",
            fg="yellow" if over > 1 else "green",
        )
        click.echo(f"  per joint: {np.round(delta, 3).tolist()}")

    def _at_targets(self, targets: dict[str, np.ndarray], measured: dict[str, float]) -> bool:
        reached = {
            "lift": abs(float(np.ravel(targets["lift"])[0]) - measured["lift"]) < 0.01,
            "arm": abs(float(np.ravel(targets["arm"])[0]) - measured["arm"]) < 0.01,
        }
        wrist = np.ravel(targets["wrist"])
        for index, (name, sign) in enumerate(
            (("wrist_yaw", 1.0), ("wrist_pitch", 1.0), ("wrist_roll", WRIST_ROLL_SIM_SIGN))
        ):
            reached[name] = abs(sign * float(wrist[index]) - measured[name]) < 0.05
        return all(reached.values())

    # -- one step ----------------------------------------------------------

    def _current_observation(self) -> RobotObservation | None:
        observation = self.stream.latest()
        if observation is None:
            return None
        if observation.wrist_rgb is None or observation.head_rgb is None:
            self._warn_once(
                "cameras",
                "The stream is missing a camera. Start the robot's sender with both "
                "a wrist and a head camera.",
            )
            return None
        return observation

    def step(self) -> None:
        """One control cycle: observe, act, retarget, send, log."""
        observation = self._current_observation()
        if observation is None:
            self.pause("no frames from the robot")
            return
        if observation.age > self.settings.max_obs_age:
            self.pause(f"the newest frame is {observation.age:.1f}s old")
            return

        measured = self.last_measured = self.mirror.sync(observation)

        head = head_frame_for_policy(
            observation.head_rgb,
            observation.head_side,
            self.settings.head,
            observation.head_camera_matrix,
            observation.head_distortion,
        )
        wrist = wrist_frame_for_policy(
            observation.wrist_rgb,
            self.proxy.jaw_flipped,
            self.proxy.pose_conventions.keep_flipped_wrist_camera_frame,
            self.settings.wrist_fov_deg,
        )

        if not self.running:
            # Held: the cameras and the mirror keep running, the policy does not.
            #
            # It used to, on the theory that its observation history should stay
            # warm across a pause. It should not, and it cannot: every way out of
            # a hold goes through `new_episode`, which calls `policy.reset()` and
            # drops that history anyway. What running it did instead was query the
            # model on the GPU every `execute_horizon` steps for actions nobody
            # would send, and -- because `retarget_franka_joint_pos` is what warns
            # about an unreachable target -- report that "Stretch cannot reach
            # where the policy is pointing" about a policy that was not driving
            # anything. A held robot should be quiet.
            self._log(observation, head, wrist, measured, commanded={}, action=None)
            return

        policy_observation = {
            "task": self.task,
            "qpos": {
                "arm": self.proxy.get_move_group("arm").joint_pos,
                "gripper": self.proxy.get_move_group("gripper").joint_pos,
            },
            DROID_EXO_CAMERA_KEY: head,
            DROID_WRIST_CAMERA_KEY: wrist,
        }

        started = time.perf_counter()
        action = self.policy.get_action(policy_observation)
        inference_s = time.perf_counter() - started

        targets = self.proxy.retarget_franka_joint_pos(action["arm"])
        targets["gripper"] = self.proxy.retarget_robotiq_ctrl(action["gripper"])
        self.residuals.append((self.proxy.last_position_error, self.proxy.last_orientation_error))

        commanded: dict[str, float] = {}
        if self.commander is not None and self.running:
            commanded = self.commander.send(targets, measured)

        self._log(observation, head, wrist, measured, commanded, action, inference_s)

    def _log(
        self,
        observation: RobotObservation,
        head: np.ndarray,
        wrist: np.ndarray,
        measured: dict[str, float],
        commanded: dict[str, float],
        action: dict[str, Any] | None,
        inference_s: float = 0.0,
    ) -> None:
        """One row of telemetry, whether or not the policy was asked for an action.

        A held robot still logs: the frames, where it is, and how far behind the
        stream the loop is. What it does not log is a retargeting that did not
        happen -- the residual and the gripper command are the previous step's
        once the policy stops being queried, and a flat line in the viewer is
        easier to read than a stale one.
        """
        self.step_count += 1
        if self.telemetry is None:
            return
        diagnostics = {
            "observation_age_s": observation.age,
            "head_sync_offset_ms": observation.head_sync_offset_ms,
            "running": float(self.running),
        }
        if action is not None:
            diagnostics.update(
                {
                    "position_error_m": self.proxy.last_position_error,
                    "orientation_error_rad": self.proxy.last_orientation_error,
                    "jaw_flipped": float(self.proxy.jaw_flipped),
                    "unreachable_steps": float(self.proxy.unreachable_steps),
                    "inference_s": inference_s,
                    "robotiq_ctrl": float(np.ravel(action["gripper"])[0]),
                }
            )
        self.telemetry.log_step(
            step=self.step_count,
            images={"head": head, "wrist": wrist},
            raw={"head": observation.head_rgb, "wrist": observation.wrist_rgb},
            measured=measured,
            commanded=commanded,
            diagnostics=diagnostics,
            task=self.task,
        )

    def _warn_once(self, key: str, message: str) -> None:
        """Say something loud the first time, then stay quiet about it.

        The loop runs at 15Hz, so anything that reports every step buries the
        prompt the run is driven from.
        """
        if key not in self._warned:
            self._warned.add(key)
            click.secho(message, fg="red")

    def report(self) -> None:
        if not self.residuals:
            return
        position, orientation = np.array(self.residuals).T
        click.echo(
            f"tool position residual: mean {position.mean():.3f}m, max {position.max():.3f}m"
        )
        click.echo(
            f"tool orientation residual: mean {orientation.mean():.3f}rad, "
            f"max {orientation.max():.3f}rad"
        )
        if self.proxy.unreachable_steps:
            click.secho(
                f"{self.proxy.unreachable_steps} steps asked for a pose the lift could not "
                "reach. The policy was pointing outside this robot's workspace.",
                fg="yellow",
            )


# =============================================================================
# CLI
# =============================================================================


def _publish_conventions(**flags: Any) -> fr.PoseConventions:
    """Put the pose conventions in the environment, and return them.

    The environment because that is how every other part of the retargeting reads
    them -- `FrankaOnStretchView` constructs its own `PoseConventions` from there
    when it is not handed one, and `franka_mount_pose_from_base` reads them
    directly. Publishing rather than passing means this run and anything it
    imports cannot disagree about which conventions are in force. See
    `franka_retarget.POSE_CONVENTION_ENV_VARS`.
    """
    conventions = fr.PoseConventions(**flags)
    fr.publish_pose_conventions(conventions)
    if conventions:
        click.echo(f"  conventions: {conventions.describe()}")
    return conventions


def _parse_size(value: str | None) -> tuple[int, int] | None:
    if not value:
        return None
    if value.lower() == "droid":
        return DROID_FRAME_SIZE
    try:
        width, height = (int(part) for part in value.lower().split("x"))
    except ValueError:
        raise click.BadParameter(f"{value!r} is not a size; use WxH, e.g. 640x360, or 'droid'.")
    return width, height


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--robot-ip",
    type=str,
    default=None,
    help="The robot's address, for both the camera stream and the RobotClient. Omit to "
    "use gripper_networking's configured IP for the stream and a local RobotClient.",
)
@click.option("--port", type=int, default=None, help="The stream's port. Defaults to 4409.")
@click.option(
    "--fleet-path",
    type=str,
    default=None,
    help="Directory holding the robot's fleet directory, for this run. A workstation "
    "needs a copy of it even to drive a robot over the network; see FLEET_HELP.",
)
@click.option(
    "--fleet-id",
    type=str,
    default=None,
    help="Which fleet directory under --fleet-path, e.g. stretch-se4-3001. Inferred when "
    "there is only one.",
)
@click.option(
    "--task",
    type=str,
    default="",
    help="Start with this instruction rather than waiting for one at the prompt.",
)
@click.option(
    "--checkpoint",
    type=str,
    default=None,
    help="A local DROID checkpoint. Defaults to fetching the released one from the Hub.",
)
@click.option("--control-hz", type=float, default=CONTROL_HZ, show_default=True)
@click.option(
    "--include-base/--no-include-base",
    default=False,
    show_default=True,
    help="Let the holonomic base join the IK, so the policy can drive the robot. Off by "
    "default: on hardware this moves 25kg around the room you are standing in.",
)
@click.option(
    "--head-mode",
    type=click.Choice(["fisheye", "rectified"]),
    default="fisheye",
    show_default=True,
    help="What to do with the head frames. 'fisheye' is the setup `stretch_fisheye` "
    "measures; 'rectified' undoes the lens with the robot's own calibration, which is "
    "`stretch_rectified`.",
)
@click.option(
    "--head-crop",
    type=str,
    default=None,
    help="Crop the upright head frame to WxH (or 'droid' for 640x360). Unset hands the "
    "policy the whole 400x640 portrait frame, which is what the sim setups do.",
)
@click.option(
    "--head-virtual-pitch-deg",
    type=float,
    default=None,
    help="Re-centre the crop as though the camera were pitched here. Needs --head-crop; "
    "the head is bolted on at 43 degrees and does not move.",
)
@click.option(
    "--wrist-fov-deg",
    type=float,
    default=0.0,
    help="Narrow the wrist view by cropping into it. 38.3 frames a grasp the way the "
    "Robotiq does; see setups.MATCHED_WRIST_FOV_DEG. 0 leaves the camera alone.",
)
@click.option(
    "--grasp-offset-m",
    type=float,
    default=0.0,
    show_default=True,
    help="Move the commanded grasp centre along the approach. See "
    "setups.STRETCH_GRASP_OFFSET_M for what the search settled on and why.",
)
@click.option(
    "--wrist-tilt-deg",
    type=float,
    default=0.0,
    show_default=True,
    help="Pitch the retargeted tool frame about the jaw line.",
)
@click.option(
    "--target-z-offset-m",
    type=float,
    default=0.0,
    show_default=True,
    help="Raise every retargeted target by this much, for clearance over a worktop.",
)
@click.option(
    "--aperture-m",
    type=float,
    default=0.0,
    help="What 'open' means on this hand, in metres between the fingertips. 0 keeps "
    "franka_retarget.ROBOTIQ_MAX_APERTURE_M.",
)
@click.option(
    "--jaw-mode",
    type=click.Choice(list(fr.JAW_MODES)),
    default="auto",
    show_default=True,
    help="Which way round the jaw is held. 'auto' lets the IK pick the branch that "
    "reaches each target best, which is usually the half-turned one -- so the hand "
    "coming out rolled over is this, not a pose convention -- and it may change branch "
    "mid-run, which on hardware is a sudden large roll of the wrist and its cameras. "
    "'flipped' and 'upright' pin it. See franka_retarget.JAW_MODES.",
)
@click.option(
    "--snap-to-franka-home/--no-snap-to-franka-home",
    default=True,
    show_default=True,
    help="Move the arm to the Franka's home pose before the first instruction. This is a "
    "real motion of the real robot, and it is the first thing to watch.",
)
@click.option(
    "--change_franka_start_pose_flip_wrist",
    is_flag=True,
    help="Roll the Franka's start pose half a turn about its approach axis, which moves "
    "where `home` puts the arm.",
)
@click.option(
    "--change_franka_start_pose_limit_height",
    is_flag=True,
    help="Cap the Franka's start tool height at Stretch's own reach ceiling "
    "(1.0824 m). `home` then leaves the lift at its stop by construction, so any action "
    "asking for more height is unreachable and says so.",
)
@click.option(
    "--change_stretch_start_pose_flip_wrist",
    is_flag=True,
    help="No effect here. It rolls the wrist of a Stretch *spawned in simulation*, and "
    "on hardware there is no spawn -- the arm starts wherever it is, and `home` decides "
    "the start pose. Accepted so a sim command line can be pasted unchanged.",
)
@click.option(
    "--change_stretch_start_pose_pitch_deg",
    type=float,
    default=0.0,
    help="Pitch the wrist by this many degrees about the jaw line at the start pose, "
    "i.e. at `home`. Does not change the frame actions are interpreted in.",
)
@click.option(
    "--keep_flipped_wrist_camera_frame",
    is_flag=True,
    help="Feed the policy the wrist frame as the camera produced it while the jaw is "
    "half-turned, instead of turning it back upright.",
)
@click.option(
    "--map_franka_wrist_to_flipped_stretch4_wrist",
    is_flag=True,
    help="Retarget onto the half-turned branch of Stretch's wrist and hold it there. "
    "The conventions above are franka_retarget.PoseConventions, spelled as "
    "params_search_side_by_side.py spells them, so a run here can be given the same "
    "ones a sim comparison was run with.",
)
@click.option(
    "--slow",
    is_flag=True,
    help="Move at 40% of the robot's max profile -- 60% slower, in both velocity and "
    "acceleration. The targets are unchanged, so the robot lags a 15Hz stream further; "
    "read a slow rollout as a rehearsal rather than a measurement.",
)
@click.option(
    "--step-limit-scale",
    type=float,
    default=1.0,
    show_default=True,
    help="Scale the per-step target clamps in MAX_TARGET_STEP. Below 1 is more cautious.",
)
@click.option(
    "--max-obs-age",
    type=float,
    default=0.5,
    show_default=True,
    help="Stop sending when the newest frame is older than this, in seconds.",
)
@click.option("--rerun/--no-rerun", default=True, show_default=True, help="Stream to Rerun.")
@click.option(
    "--rerun-jpeg-quality",
    type=int,
    default=75,
    show_default=True,
    help="Encode the logged frames as JPEG at this quality. 0 logs raw pixels, which is "
    "about 45 MB/s of camera at 15Hz.",
)
@click.option(
    "--rrd",
    type=click.Path(path_type=Path),
    default=None,
    help="Write the Rerun stream to this .rrd file instead of spawning a viewer.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Run the whole chain against the stream without connecting to the robot or "
    "moving anything. The way to check the cameras and the retargeting first.",
)
@click.option("--no-prompt", is_flag=True, help="Skip the confirmation before the robot may move.")
def main(
    robot_ip: str | None,
    port: int | None,
    fleet_path: str | None,
    fleet_id: str | None,
    task: str,
    checkpoint: str | None,
    control_hz: float,
    include_base: bool,
    head_mode: str,
    head_crop: str | None,
    head_virtual_pitch_deg: float | None,
    wrist_fov_deg: float,
    grasp_offset_m: float,
    wrist_tilt_deg: float,
    target_z_offset_m: float,
    aperture_m: float,
    jaw_mode: str,
    snap_to_franka_home: bool,
    change_franka_start_pose_flip_wrist: bool,
    change_franka_start_pose_limit_height: bool,
    change_stretch_start_pose_flip_wrist: bool,
    change_stretch_start_pose_pitch_deg: float,
    keep_flipped_wrist_camera_frame: bool,
    map_franka_wrist_to_flipped_stretch4_wrist: bool,
    slow: bool,
    step_limit_scale: float,
    max_obs_age: float,
    rerun: bool,
    rerun_jpeg_quality: int,
    rrd: Path | None,
    dry_run: bool,
    no_prompt: bool,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    crop_to = _parse_size(head_crop)
    if head_virtual_pitch_deg is not None and crop_to is None:
        raise click.BadParameter(
            "--head-virtual-pitch-deg only does something with --head-crop: the pitch is "
            "synthesised by moving the crop window, and there is no window without one."
        )

    address = _stream_address(robot_ip, port)
    click.secho("Retargeted MolmoBot-DROID on a real Stretch 4", bold=True)
    click.echo(f"  stream     : {address}")
    where = "--dry-run, not connecting" if dry_run else (robot_ip or "local")
    click.echo(f"  robot      : {where}")
    click.echo(f"  head       : {head_mode}, crop {crop_to or 'none'}")
    click.echo(f"  rate       : {control_hz} Hz")
    click.echo(
        f"  speed      : {SLOW_SPEED_SCALE * 100:.0f}% of the max profile (--slow)"
        if slow
        else "  speed      : the robot's max profile"
    )
    click.echo(f"  python     : {sys.executable}")
    conventions = _publish_conventions(
        change_franka_start_pose_flip_wrist=change_franka_start_pose_flip_wrist,
        change_franka_start_pose_limit_height=change_franka_start_pose_limit_height,
        change_stretch_start_pose_flip_wrist=change_stretch_start_pose_flip_wrist,
        change_stretch_start_pose_pitch_deg=change_stretch_start_pose_pitch_deg,
        keep_flipped_wrist_camera_frame=keep_flipped_wrist_camera_frame,
        map_franka_wrist_to_flipped_stretch4_wrist=map_franka_wrist_to_flipped_stretch4_wrist,
        match_stretch_spawn_pose_to_franka=False,
    )

    stream = RobotStream(address)
    stream.start()
    click.echo("Waiting for the robot's camera and joint stream...")
    first = stream.wait_for_first(timeout=30.0)
    if first is None:
        stream.stop()
        raise SystemExit(
            f"Nothing arrived on {address} in 30s. Is "
            "send_gripper_and_head_images_with_joint_states.py running on the robot, and "
            "is the port open (see gripper_networking.print_network_info)?"
        )
    click.echo(
        f"  cameras    : wrist {first.wrist_side} {_shape(first.wrist_rgb)}, "
        f"head {first.head_side} {_shape(first.head_rgb)}"
    )
    # Which lens each channel reads is decided on the robot, by the sender's own
    # flags, and it is the one setting of this run that cannot be set from here.
    # Said out loud because it is also a condition the sim study has a name for,
    # and a run compared against the wrong one is compared against nothing.
    for side, channel, flag in (
        (first.wrist_side, "wrist", "--use_left_gripper_camera"),
        (first.head_side, "head", "--use_left_fisheye_camera"),
    ):
        if side.lower() == "left":
            click.echo(
                f"               the {channel} channel is Stretch's LEFT camera, which is "
                f"the study's {flag} condition"
            )
    if not first.joints:
        stream.stop()
        raise SystemExit(
            "The stream carries no joint state. This runs against the sender's "
            f"`closest_joint_state`, whose {GRIPPER_JOINT} entry is also the only gripper "
            "state it can read."
        )

    if not dry_run:
        prepare_fleet_environment(fleet_path, fleet_id)
    robot = None if dry_run else connect_robot(robot_ip)
    gripper_units = GripperUnits.nominal() if robot is None else GripperUnits.from_robot(robot)
    click.echo(
        f"  gripper    : {gripper_units.closed_pct:.0f}% shut to {gripper_units.open_pct:.0f}% open"
    )

    click.echo("Building the kinematic mirror...")
    _, _, view, namespace = build_mirror()
    mirror = RobotMirror(view, gripper_units)
    measured = mirror.sync(first)
    click.echo(f"  robot is at: {({k: round(v, 3) for k, v in measured.items()})}")

    proxy = fr.FrankaOnStretchView(
        view,
        namespace,
        fr.franka_mount_pose_from_base(view.get_move_group("base").joint_pos),
        include_base=include_base,
        target_z_offset=target_z_offset_m,
        jaw_mode=jaw_mode,
        pose_conventions=conventions,
    )
    # The study's two tool parameters, applied the way `setups.py` applies them
    # to a trial, so a run here is the same retargeting a sim trial measured.
    apply_tool_correction(proxy, wrist_tilt_deg=wrist_tilt_deg, grasp_offset_m=grasp_offset_m)
    apply_aperture(proxy, aperture_m)
    proxy.reset()
    click.echo(
        f"  retarget   : grasp offset {grasp_offset_m:+.4f}m, wrist tilt {wrist_tilt_deg:+.1f}deg, "
        f"z offset {target_z_offset_m:+.4f}m, jaw {proxy.jaw_mode}, opens to "
        f"{proxy.finger_open:.4f} rad, base {'in' if include_base else 'out of'} the IK"
    )

    click.echo("Loading the checkpoint...")
    from examples.machine_learning.molmospaces.demo_droid_on_stretch import load_droid_policy

    policy = load_droid_policy(checkpoint)

    telemetry = (
        RerunTelemetry(save_path=rrd, jpeg_quality=rerun_jpeg_quality)
        if (rerun or rrd is not None)
        else None
    )
    commander = (
        None
        if robot is None
        else RobotCommander(
            robot,
            gripper_units,
            step_limit_scale=step_limit_scale,
            include_base=include_base,
            control_period_s=1.0 / control_hz,
            speed_scale=SLOW_SPEED_SCALE if slow else 1.0,
        )
    )

    if commander is not None and not no_prompt:
        click.secho(
            "\nThe real robot will move: lift, arm, wrist and gripper"
            + (", and the base." if include_base else ".")
            + "\nClear the area, keep the runstop within reach.",
            fg="yellow",
        )
        click.prompt("Hit enter to begin", default="", show_default=False)

    runner = RealStretchRunner(
        stream=stream,
        mirror=mirror,
        proxy=proxy,
        policy=policy,
        commander=commander,
        telemetry=telemetry,
        settings=RunSettings(
            task=task,
            control_hz=control_hz,
            max_obs_age=max_obs_age,
            head=HeadCameraOptions(
                rectify=head_mode == "rectified",
                crop_to=crop_to,
                virtual_pitch_deg=head_virtual_pitch_deg,
            ),
            wrist_fov_deg=wrist_fov_deg,
        ),
    )

    if snap_to_franka_home:
        runner.go_home()
    if task:
        runner.new_episode(task)

    console = Console()
    console.start()
    click.secho("\n" + Console.HELP, fg="cyan")
    click.echo("> ", nl=False)

    period = 1.0 / control_hz
    try:
        while True:
            started = time.perf_counter()
            if not _handle_console(console, runner):
                break
            # Stepped whether or not the policy is running, so the cameras, the
            # mirror and the viewer stay live while the robot is held. `step`
            # decides what that means: held, it logs and returns without querying
            # the model or sending anything.
            runner.step()
            elapsed = time.perf_counter() - started
            if elapsed < period:
                time.sleep(period - elapsed)
    except KeyboardInterrupt:
        pass
    finally:
        console.stop()
        if commander is not None:
            commander.hold()
        runner.report()
        stream.stop()
        if robot is not None:
            robot.stop()
        click.secho(f"\nRan for {runner.step_count} policy steps.", fg="green")


def _handle_console(console: Console, runner: RealStretchRunner) -> bool:
    """Apply whatever was typed. False means the run should end."""
    while True:
        try:
            line = console.lines.get_nowait()
        except queue.Empty:
            return True

        command = line.strip()
        lowered = command.lower()
        if lowered in ("quit", "q", "exit"):
            return False
        if lowered in ("stop", "s", "halt"):
            runner.pause("asked to")
        elif lowered in ("go", "g", "run", "resume"):
            runner.resume()
        elif lowered == "home":
            runner.go_home()
        elif lowered in ("?", "help", "h"):
            click.echo(Console.HELP)
        elif command:
            runner.new_episode(command)
        click.echo("> ", nl=False)


def _stream_address(robot_ip: str | None, port: int | None) -> str:
    """Where to subscribe: the flags, or whatever `gripper_networking` is configured with.

    Read from that module when it is importable, so a workstation that already
    has the gripper repository set up needs no flags and cannot end up pointed at
    a different robot than its other receivers.
    """
    host, default_port = robot_ip, DEFAULT_JOINTS_PORT
    try:
        from stretch4_gripper_modeling_and_control import gripper_networking as networking

        default_port = int(networking.gripper_and_joints_port)
        host = host or str(networking.robot_ip)
    except Exception:  # noqa: BLE001 - the flags are enough without it
        host = host or "127.0.0.1"
    return f"tcp://{host}:{int(port or default_port)}"


def _shape(frame: np.ndarray | None) -> str:
    return "missing" if frame is None else f"{frame.shape[1]}x{frame.shape[0]}"


if __name__ == "__main__":
    main()
