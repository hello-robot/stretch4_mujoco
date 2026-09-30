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
    python -m examples.machine_learning.molmospaces.retargetting.run_on_real_stretch \
        --robot-ip 100.71.110.52 \
        --change_franka_start_pose_limit_height \
        --grasp-offset-m -0.009

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

What the viewer shows
---------------------
Two tabs (`RerunTelemetry`). **run**, the one that opens, is for watching a
rollout: the two frames as the policy receives them, a text log of how long each
model query took, and one 3D view holding *three* robots at once --

* Stretch, in its own colours, drawn from the MuJoCo mirror, which is the robot
  as the retargeting believes it to be;
* the virtual Franka it is imitating, a green ghost standing on its imaginary
  pedestal at the commanded joint angles (`RerunRobotScene`), with the tail of
  the policy's action chunk drawn ahead of its hand;
* this robot's own URDF, violet, posed from `RobotClient`'s status rather than
  from the camera stream (`RerunUrdfOverlay`).

The two tool centres are balled and labelled with the gap between them, and the
plot beside it separates that gap into what the retargeting costs, what the robot
has not tracked yet, and what the policy is actually closing its loop on -- see
`tool_errors`. The third robot is there to answer a question the other two
cannot: it is the same machine as the MuJoCo Stretch from a different joint
source, so the distance between *their* grippers is stream latency and
calibration rather than anything the policy did.

**diagnostics** is the layout this module had before: processed frames beside the
raw ones, measured against commanded per joint, and the retargeting residual.
`--no-rerun-3d` drops the 3D view and `--no-rerun-urdf` just the third robot.

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
from collections.abc import Callable
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

from examples.digital_twin import (  # noqa: E402
    REAL_ROBOT_COLOR,
    TOOL_FOR_GRIPPER_JOINT,
    RerunMujocoRobot,
    RerunUrdfRobot,
    ensure_fleet_directory,
    load_stretch_urdf,
    rgba8,
)
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
from examples.machine_learning.molmospaces.stretch.config import (  # noqa: E402
    Stretch4RobotConfig,
    publish_stretch4_tool,
)
from examples.machine_learning.molmospaces.stretch.robot import Stretch4Robot  # noqa: E402
from examples.machine_learning.molmospaces.stretch.robot_view import (  # noqa: E402
    GripperKind,
    Stretch4RobotView,
    commandable_limits,
)
from stretch4_mujoco.config import robot_settings_se4  # noqa: E402
from stretch4_mujoco.enums.stretch_cameras import StretchCameras  # noqa: E402

log = logging.getLogger(__name__)

# `connect_robot` imports `stretch4_body`, which reads a fleet directory while it
# is being imported and exits if there is none. The stand-in is *not* written
# here, at import, any more: it has to declare the tool that is on the robot, and
# that is only known once the stream has said which gripper it is reporting --
# see `main` and `stream_gripper_joint`. `stretch4_body` is imported in
# `connect_robot` and nowhere earlier, so `main` is still in time.


# =============================================================================
# What the two ends of the link agree on
# =============================================================================

DEFAULT_JOINTS_PORT = 4409
"""`gripper_networking.gripper_and_joints_port`, restated so this module imports
without the gripper repository on the path. `--port` overrides it, and the real
value is read from `gripper_networking` when that package *is* importable."""

CONTROL_HZ = 15.0
"""The rate every other MolmoBot-DROID rollout in this repository runs at."""

WRIST_CAMERA_FOV_DEG = float(
    StretchCameras.cam_gripper_se4_right_rgb.initial_camera_settings
    .field_of_view_vertical_in_degrees
)
"""The gripper camera's own vertical field of view. `--wrist-fov-deg` crops into it."""


GRIPPER_JOINT = "stretch_gripper"
PARALLEL_GRIPPER_JOINT = "parallel_gripper"
"""The end-of-arm joints the two tools register: the SG4's and the PG4's.

The robot's sender reads whichever of the two `status['end_of_arm']` carries and
publishes it as `closest_joint_state['gripper']`, naming it under `joint` --
`pos_pct` for the SG4, `pos_mm` for the PG4. See `stream_gripper_joint`.
"""

STRETCH_GRIPPER_CLOSED_PCT = -50.0
"""`pos_pct` with the SG4's fingers shut. `GripperUnits` maps the range onto the model's."""

PARALLEL_GRIPPER_NOMINAL_RANGE_M = 0.080
"""The PG4's travel, 0 (tips touching) to this, in metres: `SE4_parallel_gripper_DW4`'s
`range_mm` in stretch4_body's nominal parameters. `GripperUnits.for_joint` reads the
robot's own when there is a robot to read it from."""

MAX_TARGET_STEP = {
    "lift": 0.10,
    "arm": 0.10,
    "wrist": 0.60,
    "gripper": 1.00,
    "base": 0.10,
    "base_yaw": 0.30,
}
"""How far one step's target may sit from where the joint currently is: metres
for the prismatic joints and the base's translation, radians for the wrist and
the base's heading, and a *fraction of the hand's travel* for the gripper -- the
SG4's fingers are radians and the PG4's metres, so no one number means the same
on both. 1.0 is the whole travel, which is what the SG4's old 0.5 rad was.

A guard against a garbage prediction, not a speed limit, and deliberately sized
well above what the robot can do in a control period -- the robot lags a streamed
target by construction (see `digital_twin.py`), so a clamp near its actual speed
would throttle every step rather than only the bad ones. `--step-limit-scale`
tightens all of them together.
"""

SLOW_SPEED_SCALE = 0.2
"""What `--slow` multiplies every commanded velocity and acceleration by.

A fifth of the profile the joint would otherwise run, which is its `max`. Both
numbers, not just the velocity: scaling the speed alone leaves every motion
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
"""The speed the base's relative moves are profiled at, under `--include-base`, at
full speed.

Half of what `live_policy.py` allows itself in simulation. A base command here
moves a 25kg robot across a room it shares with whoever is running it, and the
policy's base output is an IK by-product -- the solver drives the base only
because the arm could not reach -- so it is worth being slower than the arm
rather than faster.

`--slow` scales these too, in `RobotCommander._send_base` rather than here: the
arm's limits are scaled per-commander by `speed_scale`, and a base whose cap was
scaled at module level would be slow on a full-speed run as well, which is the
one thing `--slow` is supposed to be the switch for. Clipping here also costs
nothing in accuracy -- the IK hands back an absolute base *pose* and the loop
re-solves from the measured one every step, so a clipped base closes the same gap
over more steps rather than aiming somewhere else.
"""

BASE_DEADBAND_M = 0.005
BASE_DEADBAND_RAD = 0.01
"""A base gap smaller than this is not sent. See `RobotCommander._send_base`."""

ARRIVAL_TIMEOUT_S = 2.0
"""How long `--wait-for-arrival` waits for one step's motion to finish.

Generous for a single step: every target is clamped to `MAX_TARGET_STEP` of the
measured position, so a step that is still moving after this is not a slow
step, it is a joint that is not going to get there. The loop carries on rather
than stalling on it, and says so once."""

MOTION_START_WINDOW_S = 0.2
"""How long `--wait-for-arrival` waits to see a step's motion *start*.

`is_moving()` reads the last status pulled, and right after `push_command` that
status can still describe a robot at rest -- the same race
`RobotClient.wait_on_motion_finish` covers with `wait_on_motion_start`. Shorter
than its 0.5s because it is paid in full by every step whose targets are already
where the robot is (a held gripper, a clamped joint), and at 15Hz that adds up.
"""

ARRIVAL_POLL_S = 0.01
"""Sleep between `is_moving()` polls under `--wait-for-arrival`."""

DESYNC_TOOL_M = 0.01
DESYNC_JOINT_RAD = 0.05
"""Past either, the step is flagged as desynced in Rerun's text log. See `Desync`.

A centimetre at the tool, the scale at which a grasp starts missing and the one
`fr.UNREACHABLE_WARNING_M` uses. 0.05 rad at the worst Franka joint, a quarter
of the 0.2 rad per-step cap `relative_max_joint_delta` puts on a whole action --
a gap that size is a large fraction of what the next delta is going to move."""


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


def wrist_frame_for_policy(frame: np.ndarray, fov_deg: float = 0.0) -> np.ndarray:
    """A raw wrist frame, as the checkpoint reads it.

    The frame goes to the policy at the hardware's own 640x400, uncropped: it is
    *not* reframed to the 656x368 a sim rollout renders. It is never rotated,
    whichever branch the wrist is held on: the policy sees the frame the way the
    camera produced it.

    `fov_deg` is the one crop, and it is opt-in: it is `RetargetParams.wrist_fov_deg` as
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
    return np.ascontiguousarray(frame)


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
    """Converts between the robot's gripper units and the model's finger joints.

    The two ends measure the same hand differently, and differently again per
    tool:

        tool  robot joint        robot units          model fingers
        SG4   stretch_gripper    percent of servo     revolute, 0 shut .. 0.5 rad open
        PG4   parallel_gripper   metres of jaw        prismatic, 0 shut .. -0.04 m open

    Mapped as a *fraction of the hand's travel* rather than unit for unit, which
    is the same choice `digital_twin.GripperMirror` makes for the same reason:
    what the policy reads on this channel is where the hand is between shut and
    open (`_ProxyGripperGroup.joint_pos` normalises it again on the way into
    Robotiq units). On the SG4 the two stacks' aperture constants disagree by a
    centimetre at the open end -- 0.1885 m measured on the compiled model against
    `robot_settings_se4`'s hand-measured 0.177 -- and a fraction is exact at both
    ends and within that centimetre in between. On the PG4 the fraction is also
    what makes the model's negative-is-open slides and the robot's
    positive-is-open millimetres agree.

    The open end comes off the robot's own parameters, because it is the number
    that differs between tools and calibrations; the model's ends come off the
    gripper move group (`GripperKind`), which is the tool the mirror was built
    with.
    """

    def __init__(
        self,
        joint: str,
        kind: GripperKind,
        robot_open: float,
        robot_closed: float,
        stream_field: str,
        stream_scale: float = 1.0,
        unit: str = "%",
    ) -> None:
        self.joint = joint
        self.kind = kind
        self.robot_open = float(robot_open)
        self.robot_closed = float(robot_closed)
        self.stream_field = stream_field
        """What the sender publishes the position under, in `closest_joint_state['gripper']`."""
        self.stream_scale = float(stream_scale)
        """Stream units -> robot units: 0.001 for the PG4's `pos_mm`, commanded in metres."""
        self.unit = unit

    @classmethod
    def for_joint(cls, joint: str, kind: GripperKind, robot: Any = None) -> "GripperUnits":
        """The units for `joint` on a mirror built with `kind`, from `robot`'s parameters if given.

        **SG4.** `range_deg` is the servo sweep either side of the fingers
        touching, and percent is that sweep scaled so -100 is fully shut -- so the
        open end is `100 * |open / closed|`, exactly as `digital_twin.GripperMirror`
        derives it. With no robot, `--dry-run`, the model's own settings give the
        same arithmetic -- +300 on an SE4, whose sweep is -100 to +300 degrees, and
        not 100: assuming 100 would have the mirror read a wide open hand as a
        third open.

        **PG4.** `move_to` takes metres of jaw, 0 (tips touching) to `range_mm`,
        and the sender streams `pos_mm`.
        """
        params = getattr(robot, "robot_params", {}) or {}
        if joint == PARALLEL_GRIPPER_JOINT:
            range_mm = params.get(joint, {}).get("range_mm")
            robot_open = float(range_mm) / 1000.0 if range_mm else PARALLEL_GRIPPER_NOMINAL_RANGE_M
            return cls(joint, kind, robot_open, 0.0, "pos_mm", stream_scale=0.001, unit="m")
        range_deg = params.get(joint, {}).get("range_deg") if robot is not None else None
        if not range_deg:
            range_deg = robot_settings_se4["stretch_gripper"]["gripper_servo_range_deg"]
        closed_deg, open_deg = range_deg
        robot_open = 100.0 * abs(open_deg / closed_deg) if closed_deg else 100.0
        return cls(joint, kind, robot_open, STRETCH_GRIPPER_CLOSED_PCT, "pos_pct")

    def describe(self) -> str:
        return (
            f"{self.joint} ({self.kind.name}), {self.robot_closed:g}{self.unit} shut to "
            f"{self.robot_open:g}{self.unit} open"
        )

    def read(self, observation: "RobotObservation") -> float:
        """The gripper position in one stream sample, in robot units."""
        return observation.joint("gripper", self.stream_field) * self.stream_scale

    def fraction(self, robot_value: float) -> float:
        """How open a robot-units position is, 0 shut to 1 open."""
        span = self.robot_open - self.robot_closed
        if not span:
            return 0.0
        return float(np.clip((robot_value - self.robot_closed) / span, 0.0, 1.0))

    def finger(self, robot_value: float) -> float:
        """Where the model's fingers are, for a gripper the robot reports at `robot_value`."""
        return float(self.kind.joint_pos_for_fraction(self.fraction(robot_value)))

    def finger_fraction(self, finger: float) -> float:
        """How open a model finger position is, 0 shut to 1 open."""
        return float(np.clip(self.kind.open_fraction(finger), 0.0, 1.0))

    def robot_value(self, fraction: float) -> float:
        """What to tell the robot, for a hand `fraction` of the way open."""
        fraction = float(np.clip(fraction, 0.0, 1.0))
        return self.robot_closed + fraction * (self.robot_open - self.robot_closed)


def stream_gripper_joint(observation: "RobotObservation") -> str | None:
    """Which gripper the robot's sender says it is reporting, or None if it does not say.

    A sender from before the PG4 publishes the SG4's `pos_pct` and no `joint`,
    and on a PG4 robot it publishes nothing that moves at all -- it takes its
    new-sample pulse off the SG4's status, which is not there. So None is not
    evidence of an SG4; `main` assumes one only because that is all an old sender
    could be serving, and `connect_robot` checks the robot's own answer.
    """
    gripper = observation.joints.get("gripper") if isinstance(observation.joints, dict) else None
    joint = gripper.get("joint") if isinstance(gripper, dict) else None
    return joint if joint in (GRIPPER_JOINT, PARALLEL_GRIPPER_JOINT) else None


class RobotMirror:
    """Keeps the MuJoCo model standing where the real robot is standing.

    One direction only. Nothing in this class reads the model to command the
    robot -- that is `RobotCommander` -- so the mirror can never drive anything;
    it is the robot's measured state expressed as an `MjData`, which is the form
    the retargeting needs it in.

    Every joint, the wrist roll included, is written as the robot reports it:
    `RobotClient` reports roll in the URDF's own sign, which is what
    stretch4_body's self-collision and IK feed the URDF. The URDF's roll *limits*
    are the servo's mirrored, which `mjcf_generator.FLIP_WRIST_ROLL_RANGE`
    corrects in the model this clips to.
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
        gripper = self.gripper.read(observation)
        finger = self.gripper.finger(gripper)

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
                self._clip("wrist", 2, roll),
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
            "gripper_pos": gripper,
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
            for index, name in enumerate(("wrist_yaw", "wrist_pitch", "wrist_roll")):
                value = self._clamp("wrist", float(wrist[index]), measured[name])
                self._check(
                    name,
                    self.robot.end_of_arm.move_to(name, value, eoa_velocity, eoa_acceleration),
                )
                commanded[name] = value

        if "gripper" in targets:
            # In fractions of the hand's travel, the one unit the two tools share;
            # see `GripperUnits` and `MAX_TARGET_STEP`.
            finger = float(np.mean(np.asarray(targets["gripper"], dtype=float)))
            fraction = self._clamp(
                "gripper",
                self.gripper.finger_fraction(finger),
                self.gripper.fraction(measured["gripper_pos"]),
            )
            value = self.gripper.robot_value(fraction)
            self._check(
                self.gripper.joint,
                end_of_arm_move_to(
                    self.robot, self.gripper.joint, value, eoa_velocity, eoa_acceleration
                ),
            )
            commanded["gripper_pos"] = value

        if self.include_base and "base" in targets:
            goal = np.ravel(np.asarray(targets["base"], dtype=float))
            commanded.update(self._send_base(goal, measured))

        self.robot.push_command()
        self.last_sent = commanded
        return commanded

    def wait_until_settled(self, timeout: float, keep_waiting: Callable[[], bool]) -> bool:
        """Block until the motion `send` started has finished, by `RobotClient.is_moving()`.

        What `--wait-for-arrival` does after every step, so that the next
        `joint_pos_rel` delta is added to where the last one actually got to
        rather than to a joint still on its way there.

        Not `robot.wait_command()`: that returns None whatever happened, so a
        timeout looks like an arrival, and it cannot be interrupted -- a stop key
        pressed during it lands only once the motion is over. `keep_waiting` is
        polled every cycle for exactly that; it returning False ends the wait.

        Motion has to be *seen* starting before a still robot counts as arrived,
        within `MOTION_START_WINDOW_S`; see there. Returns True on arrival, False
        on a timeout or when `keep_waiting` ended it.
        """
        started_at = time.monotonic()
        seen_moving = False
        while time.monotonic() - started_at < timeout:
            if not keep_waiting():
                return False
            self.robot.pull_status(blocking=True)
            if self.robot.is_moving():
                seen_moving = True
            elif seen_moving or time.monotonic() - started_at >= MOTION_START_WINDOW_S:
                return True
            time.sleep(ARRIVAL_POLL_S)
        return False

    def _send_base(self, goal: np.ndarray, measured: dict[str, float]) -> dict[str, float]:
        """A base pose target, as one relative move toward it.

        The IK hands back an absolute pose and this base steers three omniwheels,
        so there is nothing absolute to command: the gap from the measured pose,
        rotated into the base's own frame, goes out as a `translate_by` or a
        `rotate_by` -- position moves the wheels close themselves, rather than a
        velocity that keeps driving until the next step replaces it. Profiled at
        `MAX_BASE_SPEED_MPS` / `MAX_BASE_TURN_RADPS` times `speed_scale`, so that
        `--slow` slows the base as well as the arm; see `MAX_BASE_SPEED_MPS` for
        why the scaling is here and not in the constant.

        **One or the other, never both.** `omnibase.move_by` refuses to blend a
        translation with a rotation, and two incremental commands in one
        `push_command` overwrite each other's wheel targets, so the second would
        silently erase the first. Each step sends whichever axis is further
        behind, in time at its own speed cap; the other is still there on the
        next step, re-measured, because the gap is taken from the measured pose
        every time.

        The gap is clamped to `MAX_TARGET_STEP` like every other joint, and one
        inside `BASE_DEADBAND_M` / `BASE_DEADBAND_RAD` is not sent at all: an
        incremental command re-triggers the wheels' trajectory, and a base that
        has arrived should be left to hold rather than nudged by IK noise.
        """
        yaw = measured["base_theta"]
        delta = goal[:2] - np.array([measured["base_x"], measured["base_y"]])
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        forward = float(cos_yaw * delta[0] + sin_yaw * delta[1])
        left = float(-sin_yaw * delta[0] + cos_yaw * delta[1])
        turn = math.atan2(math.sin(goal[2] - yaw), math.cos(goal[2] - yaw))

        distance = math.hypot(forward, left)
        step = self.step_limits["base"]
        if distance > step:
            forward, left = forward * step / distance, left * step / distance
            distance = step
        turn = self._clamp("base_yaw", turn, 0.0)

        speed = MAX_BASE_SPEED_MPS * self.speed_scale
        rate = MAX_BASE_TURN_RADPS * self.speed_scale
        translate = distance >= BASE_DEADBAND_M
        rotate = abs(turn) >= BASE_DEADBAND_RAD
        if rotate and (not translate or abs(turn) / rate > distance / speed):
            self.robot.omnibase.rotate_by(turn, v_r=rate)
            return {"base_dx": 0.0, "base_dy": 0.0, "base_dyaw": turn}
        if translate:
            self.robot.omnibase.translate_by(forward, left, v_m=speed)
            return {"base_dx": forward, "base_dy": left, "base_dyaw": 0.0}
        return {"base_dx": 0.0, "base_dy": 0.0, "base_dyaw": 0.0}

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


def robot_gripper_joint(robot: Any) -> str | None:
    """Which gripper the robot has on, from its own status: `stretch_gripper` or
    `parallel_gripper`, or None if it reports neither.

    `robot.end_of_arm.status` is the server's whole `end_of_arm` status, copied
    in by `pull_status` whatever the local fleet directory says the tool is -- so
    this is the robot's answer, not the fleet directory's.
    """
    status = getattr(getattr(robot, "end_of_arm", None), "status", None) or {}
    for joint in (GRIPPER_JOINT, PARALLEL_GRIPPER_JOINT):
        if joint in status:
            return joint
    return None


def end_of_arm_joint_homed(robot: Any, joint: str) -> bool:
    """Whether the robot reports `joint` calibrated, from its own status."""
    return bool(robot.end_of_arm.status.get(joint, {}).get("pos_calibrated", False))


def end_of_arm_move_to(robot: Any, joint: str, value: float, v_r=None, a_r=None) -> bool:
    """`robot.end_of_arm.move_to`, for a joint the local fleet directory may not know.

    `EndOfArmClient` builds its joint list from the *local* fleet directory and
    refuses any joint not in it, so a directory naming the SG4 refuses every
    command to a PG4's `parallel_gripper` -- although the robot's server, which
    has the robot's own parameters, would execute it. A joint the client knows
    goes through the client as always. One it does not is queued exactly as the
    client's own `move_to` queues it, after the same homing check made against
    the robot's status rather than the local parameters: the server resolves the
    joint, applies its calibration and clamps the target to its range.
    """
    end_of_arm = robot.end_of_arm
    if joint in end_of_arm.joints:
        return end_of_arm.move_to(joint, value, v_r, a_r)
    if not end_of_arm_joint_homed(robot, joint):
        return False
    end_of_arm._queue_command(f"{joint}.end_of_arm", "move_to", joint, value, v_r, a_r)
    return True


def robot_is_homed(robot: Any, gripper_joint: str | None) -> bool:
    """`robot.is_homed()`, with the end of arm checked against what the robot has on.

    `RobotClient.is_homed` checks the end-of-arm joints the local fleet directory
    lists, so on a PG4 driven through an SG4 directory it waits for a
    `stretch_gripper` that will never report calibrated. The other subsystems
    are checked as `is_homed` checks them; the end of arm is checked joint by
    joint in the robot's own status -- the wrist, and the gripper it reports.
    """
    for subsystem in robot.subsystems.values():
        if subsystem is robot.end_of_arm or not hasattr(subsystem, "is_homed"):
            continue
        if not subsystem.is_homed():
            return False
    joints = ["wrist_yaw", "wrist_pitch", "wrist_roll"] + ([gripper_joint] if gripper_joint else [])
    return all(end_of_arm_joint_homed(robot, joint) for joint in joints)


def connect_robot(robot_ip: str | None) -> tuple[Any, str]:
    """Start a `RobotClient`, and refuse to go on with a robot that is not homed.
    Returns the client and the gripper the robot reports having on.

    `examples/digital_twin.py._connect` with the homing check folded in, because
    an un-homed robot here does not fail loudly: every `move_to` returns False,
    the loop carries on computing actions, and the robot stands still while the
    terminal fills with a policy running normally.

    The gripper is read off the robot's status (`robot_gripper_joint`) rather
    than the fleet directory, so a directory naming the other tool costs only
    the nominal `range_mm` -- see `GripperUnits.for_joint` -- and is said, not
    fatal: commands go through `end_of_arm_move_to` and homing through
    `robot_is_homed`, neither of which trusts the local end of arm.
    """
    from stretch4_body.robot.robot_client import RobotClient

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
    gripper_joint = robot_gripper_joint(robot)
    if gripper_joint is None:
        robot.stop()
        raise SystemExit(
            f"The robot reports neither a {GRIPPER_JOINT} nor a {PARALLEL_GRIPPER_JOINT} "
            f"(its end of arm reports {sorted(robot.end_of_arm.status)}). This runs a gripper."
        )
    if gripper_joint not in robot.end_of_arm.joints:
        click.secho(
            f"  the fleet directory's end of arm is {sorted(robot.end_of_arm.joints)}, but the "
            f"robot has a {gripper_joint} on. Commanding it by the robot's status; only its "
            "nominal range is known here.",
            fg="yellow",
        )
    if not robot_is_homed(robot, gripper_joint):
        robot.stop()
        raise SystemExit("The robot is not fully homed. Home it first, then rerun.")
    return robot, gripper_joint


# =============================================================================
# Telemetry
# =============================================================================


PLANNED_TOOL_COLOR = (0.25, 0.65, 1.00, 1.0)
"""What the policy's unexecuted action chunk is drawn in. See `RerunRobotScene.log_plan`."""

STRETCH_ROOT_BODY = "base"
"""The body Stretch hangs off in the mirror, under the robot's namespace.

`build_mirror` adds the robot with `Stretch4RobotConfig.robot_namespace`, so the
body is `robot_0/base` -- the holonomic base wrapper, above `base_link`. Named
because `digital_twin.visual_geoms` takes a subtree rather than a name prefix:
the mirror's model is a robot on a floor today, and a scene with furniture in it
would otherwise put the furniture in the view.
"""

FRANKA_ROOT_BODY = "fr3_link0"
"""The virtual Franka's base link, and the root of everything drawn of it.

The standalone `franka_droid/model.xml` hangs the whole arm off this, welded to
an unnamed wrapper body at the origin -- which is why the root named here is the
link and not the wrapper. `VirtualFranka.fk` asserts that same body sits at the
origin, so the two agree about where this robot begins.
"""


class RerunRobotScene:
    """Both robots in one Rerun 3D view: Stretch solid, the Franka a ghost inside it.

    The same picture `test_retargeting.OverlayScene` renders for the sim study,
    built the way Rerun wants it instead of the way MuJoCo's renderer does. There
    is no third scene here: the two models this run already drives are drawn into
    one entity tree by `digital_twin.RerunMujocoRobot`, Stretch's geoms in world
    coordinates and the Franka's under a `world/franka` transform that stands it
    on its virtual pedestal. So the overlay cannot flatter the comparison -- a
    mistake in this class puts a robot in the wrong place, it cannot change a
    number.

    What the picture is for is the pair of tool centres. Stretch's is where its
    gripper actually is; the Franka's is the pose the retargeting asked it to
    reach, which is the ghost's own hand (see `log_robots` on the height offset).
    The line drawn between them, labelled in millimetres, is the whole answer.
    """

    ROOT = "world"
    STRETCH = "world/stretch"
    FRANKA = "world/franka"

    def __init__(
        self,
        rr: Any,
        stretch_model: Any,
        stretch_namespace: str,
        franka_model: Any,
    ) -> None:
        self._rr = rr
        self.stretch = RerunMujocoRobot(
            rr, stretch_model, f"{stretch_namespace}{STRETCH_ROOT_BODY}", self.STRETCH
        )
        # Painted its own marker colour rather than left grey and translucent.
        # Both robots are grey plastic and white metal and they stand inside one
        # another, so the first version of this view was one in which it was
        # genuinely hard to say which limb belonged to which -- a failure of the
        # only thing an overlay is for. The tint matches the ball below, which
        # says whose hand that is in the language the view already speaks.
        self.franka = RerunMujocoRobot(
            rr,
            franka_model,
            FRANKA_ROOT_BODY,
            self.FRANKA,
            tint=fr.FRANKA_TOOL_COLOR[:3],
        )

        rr.log(self.ROOT, rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        for path, color in (
            ("world/tool/stretch", fr.STRETCH_TOOL_COLOR),
            ("world/tool/franka", fr.FRANKA_TOOL_COLOR),
        ):
            rr.log(f"{path}/frame", rr.TransformAxes3D(axis_length=0.08), static=True)
            rr.log(
                path,
                rr.Points3D(positions=[[0.0, 0.0, 0.0]], radii=[0.012], colors=[rgba8(color)]),
                static=True,
            )

    # -- where everything is, logged per step -------------------------------

    def log_robots(self, stretch_data: Any, franka_data: Any | None, ghost_pose: Any) -> None:
        """Stretch where it is, and the Franka where the retargeting imagines it.

        `ghost_pose` is the virtual Franka's mount pose *with `target_z_offset`
        added to its height*, not the bare `franka_mount_pose`. The offset is
        defined as "stand the virtual Franka this much higher" -- that is exactly
        what `franka_tool_pose_to_world` does to every target -- so raising the
        ghost by it is what keeps the ghost's hand and the target tool centre the
        same point. Drawn at the bare mount pose the ghost's hand would sit
        `target_z_offset` below the pose Stretch is actually reaching for, and
        the gap the view exists to show would read that much too large.

        `franka_data` is `None` on a step the policy was not queried on -- the
        ghost then keeps the pose it was last commanded to, which is where the
        robot was last told to go and is the honest thing for it to be showing.
        """
        self.stretch.log(stretch_data)
        if franka_data is None:
            return
        ghost_pose = np.asarray(ghost_pose, dtype=float)
        self._rr.log(
            self.FRANKA,
            self._rr.Transform3D(translation=ghost_pose[:3, 3], mat3x3=ghost_pose[:3, :3]),
        )
        self.franka.log(franka_data)

    def log_tool_frames(self, stretch_pose: Any, franka_pose: Any | None) -> None:
        """The two tool centres, and the gap between them labelled in millimetres.

        The one number this whole view is for. Both poses are in the world, both
        are 4x4, and the label is the euclidean distance between their origins --
        which is the retargeting residual plus whatever the robot has not yet
        tracked, i.e. how far the real gripper is from the Franka gripper the
        policy thinks it is driving.
        """
        rr = self._rr
        self._log_tool("world/tool/stretch", stretch_pose)
        self._log_tool("world/tool/franka", franka_pose)
        if franka_pose is None:
            return
        stretch_point = np.asarray(stretch_pose, dtype=float)[:3, 3]
        franka_point = np.asarray(franka_pose, dtype=float)[:3, 3]
        gap = float(np.linalg.norm(franka_point - stretch_point))
        rr.log(
            "world/tool/gap",
            rr.LineStrips3D(
                [[stretch_point, franka_point]],
                radii=[0.003],
                colors=[[255, 255, 255, 255]],
                labels=[f"{gap * 1000:.0f} mm"],
            ),
        )

    def _log_tool(self, path: str, pose: Any | None) -> None:
        if pose is None:
            return
        pose = np.asarray(pose, dtype=float)
        self._rr.log(path, self._rr.Transform3D(translation=pose[:3, 3], mat3x3=pose[:3, :3]))

    def log_plan(self, points: Any) -> None:
        """Where the policy is intending to take the tool, as a polyline.

        The unexecuted tail of the action chunk, put through the virtual Franka's
        forward kinematics. See `RealStretchRunner._planned_tool_path` for what
        the approximation in it is -- this only draws what it is handed, and
        draws nothing when it is handed nothing.
        """
        rr = self._rr
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        if len(points) < 1:
            rr.log("world/plan", rr.Clear(recursive=True))
            return
        color = rgba8(PLANNED_TOOL_COLOR)
        rr.log("world/plan/points", rr.Points3D(positions=points, radii=0.006, colors=[color]))
        if len(points) >= 2:
            rr.log("world/plan/path", rr.LineStrips3D([points], radii=[0.002], colors=[color]))


class RerunUrdfOverlay:
    """The robot's *own* URDF, posed from `RobotClient`, as a third robot in the view.

    `digital_twin.RerunUrdfRobot` with this run's wiring around it: the same
    drawing the digital twin example puts beside its simulated robot, here put
    beside the MuJoCo mirror and the virtual Franka. What it adds is a robot
    neither of the others is -- the mirror is a model of this robot driven by the
    *camera stream's* `closest_joint_state`, and the virtual Franka is not this
    robot at all. This one is `stretch4_urdf`'s description of the machine on the
    bench, with that machine's batch and tool, posed from the status
    `RobotClient` reads over its own connection.

    So two disagreements become visible that nothing else here can show:

    * **Between this and the MJCF Stretch**, a disagreement about *time*. Both
      are the same robot; they are fed from two different links with two
      different latencies, so the gap between their grippers is how far behind
      the stream the policy's view of the arm is running.
    * **Between this and the geometry the MJCF was built from**, a disagreement
      about *the robot*. A steady offset is a calibration or batch difference the
      retargeting is carrying silently.

    It is drawn under the mirror's `base_link`, not at its own `base_footprint`,
    so the two Stretches are pinned together at the base by construction and
    every millimetre between their hands is joint angles rather than odometry.
    """

    ROOT = "world/urdf"
    TOOL = "world/tool/urdf"
    BASE_LINK = "base_link"
    """The frame every link pose is reported in. See the class docstring."""

    def __init__(self, rr: Any, robot: Any) -> None:
        self._rr = rr
        self._robot = robot
        urdf, self.description = load_stretch_urdf(robot)
        self.drawing = RerunUrdfRobot(rr, urdf, self.ROOT, base_link=self.BASE_LINK)
        rr.log(f"{self.TOOL}/frame", rr.TransformAxes3D(axis_length=0.08), static=True)
        rr.log(
            self.TOOL,
            rr.Points3D(
                positions=[[0.0, 0.0, 0.0]], radii=[0.012], colors=[rgba8(REAL_ROBOT_COLOR)]
            ),
            static=True,
        )
        click.echo(
            f"  urdf       : {'/'.join(self.description)}, {len(self.drawing.links)} links, "
            f"{self.drawing.vertices} vertices"
        )

    def log(self, base_pose: np.ndarray) -> np.ndarray | None:
        """Pose the overlay under `base_pose`, and return its tool pose in the world.

        `base_pose` is the *mirror's* `base_link` in the world, which is what pins
        this robot to the other two -- see the class docstring.

        `pull_status` is non-blocking. The control loop is holding the arm with a
        stream of targets and must not stop to wait for a status message; a step
        where none had arrived keeps the joints it had, which is a viewer one
        frame stale rather than a robot one frame late. The transforms are
        re-logged either way, because the base may have moved even if the arm has
        not.
        """
        if self._robot is not None and self._robot.pull_status(blocking=False):
            self.drawing.pose_from_status(self._robot.status)
        tool = self.drawing.log(base_pose)
        if tool is not None:
            self._rr.log(
                self.TOOL,
                self._rr.Transform3D(translation=tool[:3, 3], mat3x3=tool[:3, :3]),
            )
        return tool


def urdf_overlay_builder(robot: Any) -> "Callable[[Any], RerunUrdfOverlay] | None":
    """A factory for `RerunUrdfOverlay`, or `None` and a reason why not.

    Deferred rather than built outright because the overlay logs its meshes the
    moment it exists and nothing may be logged before `rr.init`, which happens
    inside `RerunTelemetry`. What *is* settled here is the part worth answering
    before a viewer is up: whether there is a robot to read a status from, and
    whether the two packages that describe it are installed. Both are ordinary
    absences rather than faults -- `--dry-run` has no robot by definition -- so
    each is a line on the way past and the run carries on with two robots in the
    view instead of three.
    """
    if robot is None:
        click.echo("  urdf       : skipped -- --dry-run has no robot to read a status from.")
        return None
    try:
        import stretch4_urdf  # noqa: F401
        import yourdfpy  # noqa: F401
    except ImportError as error:
        click.secho(
            f"  urdf       : skipped ({error}). Install it with "
            "`uv pip install yourdfpy stretch4-urdf`, or pass --no-rerun-urdf.",
            fg="yellow",
        )
        return None
    return lambda rr: RerunUrdfOverlay(rr, robot)


@dataclass
class SceneSnapshot:
    """One step's worth of 3D: where each robot is, and where the tool centres are.

    Assembled by `RealStretchRunner._scene`, which is the only place that has
    both models and the retargeting, and handed to the telemetry as a unit so
    that `log_step` stays one call per step. Every field may be absent except
    `stretch_data` and `stretch_tool`: a held robot still has a pose, and nothing
    else on this list means anything when the policy has not been asked.
    """

    stretch_data: Any
    stretch_tool: np.ndarray
    base_pose: np.ndarray | None = None
    """The mirror's `base_link` in the world, which the URDF overlay hangs off."""
    franka_data: Any | None = None
    ghost_pose: np.ndarray | None = None
    franka_tool: np.ndarray | None = None
    plan: np.ndarray | None = None


@dataclass
class TimingNote:
    """How long the policy took this step, and whether the model was actually run.

    The distinction is the whole point of logging this. `RealRobotVLAPolicy`
    predicts `action_horizon` actions and hands back `execute_horizon` of them
    before querying again, so most steps are a list index and one in eight is a
    forward pass through a VLM -- and an average over all of them describes
    neither. `queried` separates the two, and `pending` says how many actions are
    left before the next one.
    """

    inference_s: float
    queried: bool
    pending: int


class RerunTelemetry:
    """Streams the run to a Rerun viewer, and optionally to an `.rrd` file.

    Two tabs, because there are two questions and they want different pictures.

    **run**, the one that opens, is for watching a rollout: what the policy is
    being shown, how long it is taking to answer, where the two tool centres are
    relative to each other in 3D, and the errors between them. Everything on it
    is on the `policy_step` timeline, so scrubbing moves the frames, the robot
    and the plots together.

    **diagnostics** is the layout this module had before, kept whole: the
    processed frames beside the *raw* ones, and measured against commanded per
    joint. Most of what goes wrong on a first real run is in the gap between
    those two image rows -- a frame turned the wrong way, a channel swap, a crop
    that cut the workspace out -- and that is a different glance from watching a
    reach, so it is a different tab rather than a busier one.
    """

    def __init__(
        self,
        save_path: Path | None = None,
        spawn: bool = True,
        jpeg_quality: int = 75,
        scene: "Callable[[Any], RerunRobotScene] | None" = None,
        urdf: "Callable[[Any], RerunUrdfOverlay] | None" = None,
    ) -> None:
        import rerun as rr
        import rerun.blueprint as rrb

        self._rr = rr
        self.jpeg_quality = int(jpeg_quality)
        rr.init("Stretch4 retargeted DROID policy", spawn=spawn and save_path is None)
        if save_path is not None:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            rr.save(str(save_path))

        # Built here, from factories, rather than handed in ready-made: both log
        # their geometry the moment they exist -- a few megabytes of mesh, once --
        # and nothing may be logged before `rr.init` has decided where it is
        # going. The blueprint below then reads `self.scene` to know whether
        # there is a 3D view to lay out.
        self.scene = None if scene is None else scene(rr)
        self.urdf = None
        if urdf is not None:
            try:
                self.urdf = urdf(rr)
            except Exception as error:  # noqa: BLE001 - a third robot is not the run
                # Loudly, and not fatally. Everything this can fail on is about
                # *describing* the robot -- a fleet directory that resolves to
                # nominal geometry has no meshes to load -- and none of it stops
                # the robot being driven or the other two robots being drawn.
                click.secho(f"  urdf       : skipped ({error}).", fg="yellow")

        rr.send_blueprint(
            rrb.Blueprint(
                rrb.Tabs(
                    self._run_tab(rrb),
                    self._diagnostics_tab(rrb),
                    active_tab=0,
                )
            )
        )
        self._task: str | None = None
        self._queries: list[float] = []
        """Every model query's wall time this run, for the rolling mean in the log."""
        self._desynced = False
        """Whether the last logged step was flagged. See `_log_desync`."""

    def _run_tab(self, rrb: Any) -> Any:
        """The default tab: the plan, the frames behind it, and the errors."""
        left: list[Any] = []
        if self.scene is not None:
            left.append(
                rrb.Spatial3DView(
                    origin=RerunRobotScene.ROOT,
                    name="Stretch, with the Franka it is imitating",
                )
            )
        left.append(rrb.TimeSeriesView(origin="error", name="tool-centre errors"))
        return rrb.Horizontal(
            rrb.Vertical(*left, row_shares=[3, 2][: len(left)]),
            rrb.Vertical(
                rrb.Spatial2DView(origin="policy/head", name="head -> the policy"),
                rrb.Spatial2DView(origin="policy/wrist", name="wrist -> the policy"),
                rrb.TextLogView(origin="log", name="inference"),
                row_shares=[2, 2, 1],
            ),
            column_shares=[3, 2],
            name="run",
        )

    @staticmethod
    def _diagnostics_tab(rrb: Any) -> Any:
        """The layout this module shipped with, unchanged. See the class docstring."""
        return rrb.Horizontal(
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
            name="diagnostics",
        )

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
        errors: dict[str, float] | None = None,
        scene: SceneSnapshot | None = None,
        timing: TimingNote | None = None,
        desync: Desync | None = None,
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
        for key, value in (errors or {}).items():
            rr.log(f"error/{key}", rr.Scalars(float(value)))

        if scene is not None and self.scene is not None:
            self.scene.log_robots(scene.stretch_data, scene.franka_data, scene.ghost_pose)
            self.scene.log_tool_frames(scene.stretch_tool, scene.franka_tool)
            self.scene.log_plan(scene.plan if scene.plan is not None else ())
        if scene is not None and self.urdf is not None and scene.base_pose is not None:
            # The one error that is about the *robot* rather than the policy: the
            # same machine drawn from two joint sources, so the gap between the
            # two grippers is stream latency and calibration, not retargeting.
            tool = self.urdf.log(scene.base_pose)
            if tool is not None:
                rr.log(
                    "error/urdf_vs_mirror_m",
                    rr.Scalars(float(np.linalg.norm(tool[:3, 3] - scene.stretch_tool[:3, 3]))),
                )

        if timing is not None:
            self._log_timing(step, timing)
        if desync is not None:
            self._log_desync(step, desync)

        if task != self._task:
            self._task = task
            rr.log("run/task", rr.TextDocument(task))

    def _log_desync(self, step: int, desync: Desync) -> None:
        """The `Desync` scalars every step; a WARN line for every step it is flagged.

        Every flagged step rather than once, so that scrubbing the timeline to a
        bad reach finds the line at that step. Coming back into sync gets one
        INFO line, so the end of a desynced stretch is marked too.
        """
        rr = self._rr
        rr.log("error/desync_tool_m", rr.Scalars(desync.tool_m))
        rr.log("error/desync_joint_rad", rr.Scalars(desync.joint_rad))
        rr.log("error/desync_gripper_fraction", rr.Scalars(desync.gripper))
        if desync.flagged:
            rr.log(
                "log/desync",
                rr.TextLog(
                    f"step {step}: DESYNC -- the Franka the policy is told about is "
                    f"{desync.tool_m * 100:.1f}cm from where it last sent it at the tool, "
                    f"{desync.joint_rad:.3f} rad at fr3_joint{desync.worst_joint + 1} "
                    f"(flagged past {DESYNC_TOOL_M * 100:.0f}cm / {DESYNC_JOINT_RAD:.2f} rad); "
                    f"gripper {desync.gripper:.0%} of its travel apart.",
                    level="WARN",
                ),
            )
        elif self._desynced:
            rr.log(
                "log/desync",
                rr.TextLog(
                    f"step {step}: back in sync -- {desync.tool_m * 100:.1f}cm, "
                    f"{desync.joint_rad:.3f} rad.",
                    level="INFO",
                ),
            )
        self._desynced = desync.flagged

    def _log_timing(self, step: int, timing: TimingNote) -> None:
        """One line per step about how long the policy took, plus the same as scalars.

        The text is what you read while the robot is moving and the scalars are
        what you scrub afterwards, so both are logged. Only the model queries go
        into the mean -- see `TimingNote`.
        """
        rr = self._rr
        rr.log("timing/policy_s", rr.Scalars(timing.inference_s))
        rr.log("timing/model_query", rr.Scalars(float(timing.queried)))
        rr.log("timing/actions_pending", rr.Scalars(float(timing.pending)))
        if timing.queried:
            self._queries.append(timing.inference_s)
            rr.log("timing/inference_s", rr.Scalars(timing.inference_s))

        if timing.queried:
            mean = sum(self._queries) / len(self._queries)
            text = (
                f"step {step}: model query took {timing.inference_s * 1000:.0f}ms "
                f"({1.0 / timing.inference_s:.1f}Hz) -- mean {mean * 1000:.0f}ms over "
                f"{len(self._queries)} queries. {timing.pending} actions buffered."
            )
        else:
            text = (
                f"step {step}: from the buffer in {timing.inference_s * 1000:.1f}ms, "
                f"{timing.pending} actions left before the next query."
            )
        rr.log("log/inference", rr.TextLog(text, level="INFO" if timing.queried else "DEBUG"))


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


def set_chunking(
    policy: Any, action_horizon: int | None, execute_horizon: int | None
) -> tuple[int, int, int]:
    """Apply `--action-horizon` / `--execute-horizon` to a loaded `RealRobotVLAPolicy`.

    `execute_horizon` is how many actions of each chunk are sent before the model
    is queried again, and `get_action` reads it off the policy every call, so
    setting the attribute is the whole change. Lower re-plans more often on
    fresher frames, at a forward pass per re-plan.

    `action_horizon` is not what it looks like. MolmoBot's policy stores its
    config's value and never reads it: the chunk's length is the *checkpoint's*
    `action_horizon`, the length of the noise the action expert is sampled from.
    So this cannot make the model predict more or fewer actions -- it keeps the
    first `action_horizon` of each chunk and drops the rest, which is what caps
    `execute_horizon` and what the viewer's plan is drawn from. It can only
    shorten, never lengthen, the chunk.

    Returns the `(action_horizon, execute_horizon)` in effect, and the chunk
    length the checkpoint predicts.
    """
    predicted = int(getattr(getattr(policy, "agent", None), "action_horizon", policy.action_horizon))
    kept = predicted if action_horizon is None else action_horizon
    executed = policy.execute_horizon if execute_horizon is None else execute_horizon
    if not 1 <= kept <= predicted:
        raise click.BadParameter(
            f"must be between 1 and {predicted}, the chunk length this checkpoint predicts.",
            param_hint="--action-horizon",
        )
    if not 1 <= executed <= kept:
        raise click.BadParameter(
            f"must be between 1 and the action horizon ({kept}): the policy cannot "
            "execute actions it does not have.",
            param_hint="--execute-horizon",
        )

    policy.action_horizon = kept
    policy.execute_horizon = executed
    if kept < predicted:
        populate = policy._populate_action_buffer

        def populate_truncated(observation: Any) -> None:
            populate(observation)
            del policy.action_buffer[kept:]

        policy._populate_action_buffer = populate_truncated
    return kept, executed, predicted


@dataclass
class RunSettings:
    """Everything the loop is parameterised by, so `main` stays a wiring function."""

    task: str = ""
    control_hz: float = CONTROL_HZ
    max_obs_age: float = 0.5
    head: HeadCameraOptions = field(default_factory=HeadCameraOptions)
    wrist_fov_deg: float = 0.0


ROBOTIQ_HAND_JOINTS = tuple(
    f"gripper/{name}"
    for name in (
        "left_driver_joint",
        "left_spring_link_joint",
        "left_follower",
        "right_driver_joint",
        "right_spring_link_joint",
        "right_follower_joint",
    )
)
"""The virtual Franka's six Robotiq joints, which `pose_virtual_franka` sets together.

The 2F-85 is a closed linkage held shut by equality constraints, and
`mj_kinematics` does not enforce those -- so setting the driver alone draws a
hand that has come apart. Stepping the model to rest at five commands from 0 to
255 puts all six within 0.0002 rad of the driver angle, so the driver angle
written into all six is the linkage's own pose, with no physics to run."""


def pose_virtual_franka(
    franka: fr.VirtualFranka, data: MjData, joint_pos, gripper_ctrl: float | None = None
) -> None:
    """Put a *second* `MjData` for the virtual Franka at `joint_pos`, in place.

    `VirtualFranka.fk` already does these kinematics, but it does them in the
    arm's own `MjData` -- the one the retargeting solves in and overwrites on the
    next call -- and what the viewer needs is a pose that survives long enough to
    be drawn, and that drawing a planned trajectory afterwards cannot disturb.

    The joints are addressed by name, the way `VirtualFranka` addresses its own,
    so this cannot silently drift onto the wrong `qpos` slots if the model gains
    a body.

    `gripper_ctrl` is the policy's Robotiq command, 0-255, and closes the hand
    to match; see `ROBOTIQ_HAND_JOINTS`. Without it the hand stays where it was
    last put. It used to be left at the model's default aperture always, which
    drew a Franka whose hand never closed next to a Stretch whose hand closed
    fully on the same command.
    """
    joint_pos = np.asarray(joint_pos, dtype=float).reshape(-1)[: fr.VirtualFranka.N_JOINTS]
    model = franka.model
    for index, value in enumerate(np.clip(joint_pos, *franka.joint_limits.T)):
        data.qpos[model.jnt_qposadr[model.joint(f"fr3_joint{index + 1}").id]] = value
    if gripper_ctrl is not None:
        low, high = fr.ROBOTIQ_CTRL_RANGE
        fraction = float(np.clip((gripper_ctrl - low) / (high - low), 0.0, 1.0))
        driver = fr.ROBOTIQ_DRIVER_OPEN + fraction * (
            fr.ROBOTIQ_DRIVER_CLOSED - fr.ROBOTIQ_DRIVER_OPEN
        )
        for name in ROBOTIQ_HAND_JOINTS:
            data.qpos[model.jnt_qposadr[model.joint(name).id]] = driver
    mujoco.mj_kinematics(model, data)


class CommandedPose:
    """Where the joint targets that were actually sent would put the tool.

    The third pose in a picture that otherwise has two. The retargeting reports
    how far its solution fell short of the Franka's tool pose
    (`last_position_error`), and the mirror says where the gripper is -- but the
    gap between those two is a sum of two unrelated things: a five-DOF arm
    approximating a six-DOF pose, and a robot that has not finished moving. This
    separates them, by running forward kinematics on the targets
    `RobotCommander.send` actually put on the wire.

    Which is not the same vector the IK produced: every target is clamped to
    `MAX_TARGET_STEP` of where the joint is, so on a large reach the robot is
    chasing a nearer pose than the one the policy asked for, and a tracking error
    measured against the unclamped target would read that clamp as lag.

    On scratch data, for the same reason `StretchArmIK` is: writing joint
    positions into the live mirror would move the robot the retargeting is about
    to be solved against. The base is copied from the live model rather than
    commanded -- under `--include-base` the commander sends relative base moves,
    and there is no base *pose* on the wire to do kinematics with.
    """

    WRIST = ("wrist_yaw", "wrist_pitch", "wrist_roll")
    """The commanded wrist, in the model's order -- the same order as the loop in
    `RobotCommander.send` that produced it. A command read back off the wire is
    already in the model's units and sign."""

    def __init__(self, mirror: RobotMirror, namespace: str) -> None:
        self._live = mirror.data
        self._data = MjData(mirror.model)
        self._view = Stretch4RobotView(self._data, namespace)

    def tool_pose(self, commanded: dict[str, float]) -> np.ndarray | None:
        """The tool pose for one step's commands, or None if the arm was not commanded.

        `None` covers both a held robot and `--dry-run`, where nothing was sent
        and there is no commanded pose to be short of -- which is a gap in the
        plot rather than a zero, zero being a claim that the robot arrived.
        """
        if not all(name in commanded for name in ("lift", "arm", *self.WRIST)):
            return None
        self._data.qpos[:] = self._live.qpos
        self._view.set_qpos_dict(
            {
                "lift": [commanded["lift"]],
                "arm": [commanded["arm"]],
                "wrist": [commanded[name] for name in self.WRIST],
            }
        )
        mujoco.mj_kinematics(self._data.model, self._data)
        return self._view.get_move_group("wrist").leaf_frame_to_world


def tool_errors(
    measured: np.ndarray,
    commanded: np.ndarray | None,
    target: np.ndarray | None,
    reported: np.ndarray | None = None,
) -> dict[str, float]:
    """The distances between the poses in `CommandedPose`'s docstring.

    `target` is where the policy pointed, `commanded` is what the robot was told
    after the IK approximated it and the step clamp trimmed it, and `measured` is
    where the gripper is. So `retarget` is what the retargeting costs, `tracking`
    is what the robot has not done yet, and `total` is what the policy is
    actually closing its loop on -- which is the one that decides whether a grasp
    lands, and is not the sum of the other two, the three poses not being
    collinear.

    `reported` is the fourth: the Franka tool pose the policy was *told* the arm
    is at, i.e. FK of `franka_joint_pos()`. That state comes out of a second IK,
    Franka-side, and every `joint_pos_rel` action is a delta on top of it -- so
    `reported_state` is an offset that lands in every target and reads as the
    retargeting missing, when it is the observation that is wrong. It should be
    near zero whenever the Franka IK converged, moving or not.
    """
    errors: dict[str, float] = {}
    if reported is not None:
        errors["reported_state_position_m"] = float(
            np.linalg.norm(reported[:3, 3] - measured[:3, 3])
        )
    if commanded is not None:
        errors["tracking_position_m"] = float(np.linalg.norm(commanded[:3, 3] - measured[:3, 3]))
    if target is not None:
        errors["total_position_m"] = float(np.linalg.norm(target[:3, 3] - measured[:3, 3]))
        if commanded is not None:
            errors["commanded_vs_target_m"] = float(
                np.linalg.norm(target[:3, 3] - commanded[:3, 3])
            )
    return errors


@dataclass
class Desync:
    """How far the Franka the policy is *told* about is from the one it last commanded.

    The policy never sees Stretch. It sees seven Franka joint angles, solved
    back out of wherever Stretch's gripper actually is, and adds its next
    `joint_pos_rel` delta to them. As long as Stretch reaches each command those
    angles are the command echoed back, and the policy's picture of its own arm
    is the one it was trained with. When Stretch falls short -- still moving, a
    `MAX_TARGET_STEP` clamp, a pose outside its reach -- the angles it is told
    are somewhere it did not send the arm, and the next delta is added to that.
    `--wait-for-arrival` removes the first cause, not the other two.

    `gripper` is in fractions of the Robotiq's travel and is reported, not
    flagged: fingers stopped on an object never reach a close command, on the
    Franka as much as here, so a gap there while holding something is a grasp.
    """

    tool_m: float
    joint_rad: float
    worst_joint: int
    gripper: float

    @property
    def flagged(self) -> bool:
        return self.tool_m > DESYNC_TOOL_M or self.joint_rad > DESYNC_JOINT_RAD

    @classmethod
    def measure(
        cls,
        franka: fr.VirtualFranka,
        commanded_arm: np.ndarray,
        reported_arm: np.ndarray,
        commanded_gripper: float,
        reported_gripper: float,
    ) -> "Desync":
        """Both arms through the Franka's FK, so the tool gap is in the policy's own terms."""
        gap = np.abs(np.asarray(reported_arm, dtype=float) - np.asarray(commanded_arm, dtype=float))
        tool = float(
            np.linalg.norm(franka.fk(reported_arm)[:3, 3] - franka.fk(commanded_arm)[:3, 3])
        )
        low, high = fr.ROBOTIQ_CTRL_RANGE
        return cls(
            tool_m=tool,
            joint_rad=float(gap.max()),
            worst_joint=int(gap.argmax()),
            gripper=float(abs(reported_gripper - commanded_gripper) / (high - low)),
        )


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

        # Both exist only to be looked at, so neither is built when nothing is
        # looking: an `MjData` per model is a few megabytes and the kinematics
        # they run are per step.
        self._commanded = None if telemetry is None else CommandedPose(mirror, proxy.namespace)
        self._ghost = None if telemetry is None else MjData(proxy.franka.model)
        """The virtual Franka's pose as *drawn*, separate from the one it solves in.
        See `pose_virtual_franka`."""
        self._base_body = mirror.model.body(f"{proxy.namespace}{RerunUrdfOverlay.BASE_LINK}").id
        """What the URDF overlay is pinned to. See `RerunUrdfOverlay`."""

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
        click.echo(
            "Moving to the Franka home pose. Residual (dx dy dz | drx dry drz): "
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
        for index, name in enumerate(("wrist_yaw", "wrist_pitch", "wrist_roll")):
            reached[name] = abs(float(wrist[index]) - measured[name]) < 0.05
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

    def step(self) -> bool:
        """One control cycle: observe, act, retarget, send, log.

        Returns whether anything was sent to the robot, which is what
        `--wait-for-arrival` has to wait on.
        """
        observation = self._current_observation()
        if observation is None:
            self.pause("no frames from the robot")
            return False
        if observation.age > self.settings.max_obs_age:
            self.pause(f"the newest frame is {observation.age:.1f}s old")
            return False

        measured = self.last_measured = self.mirror.sync(observation)

        head = head_frame_for_policy(
            observation.head_rgb,
            observation.head_side,
            self.settings.head,
            observation.head_camera_matrix,
            observation.head_distortion,
        )
        wrist = wrist_frame_for_policy(observation.wrist_rgb, self.settings.wrist_fov_deg)

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
            return False

        policy_observation = {
            "task": self.task,
            "qpos": {
                "arm": self.proxy.get_move_group("arm").joint_pos,
                "gripper": self.proxy.get_move_group("gripper").joint_pos,
            },
            DROID_EXO_CAMERA_KEY: head,
            DROID_WRIST_CAMERA_KEY: wrist,
        }

        # Copied before the policy sees it: this is the state its `joint_pos_rel`
        # deltas are added to, and `_log` checks it against the measured gripper.
        reported_arm = np.array(policy_observation["qpos"]["arm"], dtype=float)
        desync = self._desync(reported_arm, policy_observation["qpos"]["gripper"])

        started = time.perf_counter()
        action = self.policy.get_action(policy_observation)
        timing = self._timing(time.perf_counter() - started)

        targets = self.proxy.retarget_franka_joint_pos(action["arm"])
        targets["gripper"] = self.proxy.retarget_robotiq_ctrl(action["gripper"])
        self.residuals.append((self.proxy.last_position_error, self.proxy.last_orientation_error))

        commanded: dict[str, float] = {}
        if self.commander is not None and self.running:
            commanded = self.commander.send(targets, measured)

        self._log(
            observation, head, wrist, measured, commanded, action, timing, reported_arm, desync
        )
        return bool(commanded)

    def _desync(self, reported_arm: np.ndarray, reported_gripper: Any) -> Desync | None:
        """This step's `Desync`, against the command the last step sent.

        Read before `get_action`, which overwrites `last_arm_ctrl`. None when
        nothing is driving the robot (`--dry-run`), where an arm that never moves
        is desynced by construction and flagging it every step says nothing.
        After `new_episode` the proxy's reset has made the last command the arm's
        own pose, so an episode's first step compares the arm with itself.
        """
        if self.commander is None:
            return None
        return Desync.measure(
            self.proxy.franka,
            commanded_arm=np.asarray(self.proxy.last_arm_ctrl, dtype=float),
            reported_arm=reported_arm,
            commanded_gripper=float(np.ravel(self.proxy.last_gripper_ctrl)[0]),
            reported_gripper=fr.robotiq_ctrl_from_driver(reported_gripper),
        )

    def _timing(self, inference_s: float) -> TimingNote:
        """How long `get_action` took, and whether that was the model or a list index.

        `RealRobotVLAPolicy` predicts a chunk and hands it out one action at a
        time, re-querying once `execute_horizon` of them have gone -- which it
        signals by resetting `buffer_index`, so an index of 1 on the way out is
        exactly a step the model ran on. Read rather than timed, because the
        threshold between "a forward pass through a VLM" and "a list index" is
        three orders of magnitude on a good day and not a number worth guessing
        on a bad one.

        Everything here is read defensively: this module drives MolmoBot's class
        directly (see `demo_droid_on_stretch.load_droid_policy`) and its buffer
        is an implementation detail of somebody else's policy. Without it the
        timing still logs, as a query every step, which is what a policy with no
        chunk would actually be doing.
        """
        index = int(getattr(self.policy, "buffer_index", 1))
        horizon = int(getattr(self.policy, "execute_horizon", 1))
        return TimingNote(
            inference_s=inference_s,
            queried=index <= 1,
            pending=max(0, horizon - index),
        )

    def _planned_tool_path(self, action: dict[str, Any]) -> np.ndarray | None:
        """Where the rest of this chunk would put the tool, as world points.

        The steps between now and the next forward pass are already decided, so
        they can be drawn. Only those: the chunk is `action_horizon` long and
        only `execute_horizon` of it is ever executed, and drawing the tail would
        be drawing a plan that is about to be thrown away.

        **It is a heading, not a trajectory.** The actions are `joint_pos_rel` --
        each is a delta against the arm state at the step it is executed at, and
        those states do not exist yet -- so the current one stands in for all of
        them. What the line shows is where the chunk points from here, which is
        the question worth asking of it while the robot is moving: whether the
        policy is reaching for the object or for somewhere else.
        """
        buffer = getattr(self.policy, "action_buffer", None)
        if not buffer:
            return None
        index = int(getattr(self.policy, "buffer_index", 0))
        horizon = int(getattr(self.policy, "execute_horizon", len(buffer)))
        pending = list(buffer[index : min(horizon, len(buffer))])

        here = np.asarray(self.proxy.get_move_group("arm").joint_pos, dtype=float)
        relative = getattr(self.policy, "action_type", "") == "joint_pos_rel"
        joints = [np.asarray(action["arm"], dtype=float).reshape(-1)]
        for future in pending:
            step = np.asarray(future["arm"], dtype=float).reshape(-1)
            joints.append(step[:7] + here if relative else step[:7])
        return np.array(
            [
                self.proxy.franka_tool_pose_to_world(self.proxy.franka.fk(each))[:3, 3]
                for each in joints
            ]
        )

    def _scene(self, action: dict[str, Any] | None) -> SceneSnapshot:
        """This step's 3D: Stretch where it is, and the Franka it is imitating.

        The ghost stands at the mount pose raised by `target_z_offset`, which is
        what that offset *is* -- `franka_tool_pose_to_world` adds it to every
        target, so the Franka the retargeting is chasing is one standing that
        much higher. Drawn at the bare mount pose the ghost's hand would sit the
        offset below the pose Stretch is actually reaching for, and the gap the
        view exists to show would read that much too large.
        """
        base_pose = np.eye(4)
        base_pose[:3, :3] = self.mirror.data.xmat[self._base_body].reshape(3, 3)
        base_pose[:3, 3] = self.mirror.data.xpos[self._base_body]
        scene = SceneSnapshot(
            stretch_data=self.mirror.data,
            stretch_tool=self.proxy.arm_ik.tool_pose(),
            base_pose=base_pose,
        )
        if action is None or self._ghost is None:
            return scene
        pose_virtual_franka(
            self.proxy.franka,
            self._ghost,
            action["arm"],
            gripper_ctrl=float(np.ravel(action["gripper"])[0]),
        )
        ghost = np.array(self.proxy.franka_mount_pose, dtype=float)
        ghost[2, 3] += self.proxy.target_z_offset
        scene.franka_data = self._ghost
        scene.ghost_pose = ghost
        scene.franka_tool = self.proxy.franka_tool_pose_to_world(
            self.proxy.franka.fk(np.asarray(action["arm"], dtype=float).reshape(-1))
        )
        scene.plan = self._planned_tool_path(action)
        return scene

    def _log(
        self,
        observation: RobotObservation,
        head: np.ndarray,
        wrist: np.ndarray,
        measured: dict[str, float],
        commanded: dict[str, float],
        action: dict[str, Any] | None,
        timing: TimingNote | None = None,
        reported_arm: np.ndarray | None = None,
        desync: Desync | None = None,
    ) -> None:
        """One row of telemetry, whether or not the policy was asked for an action.

        A held robot still logs: the frames, where it is, how far behind the
        stream the loop is, and where its own gripper is in 3D. What it does not
        log is a retargeting that did not happen -- the residual, the gripper
        command, the ghost Franka and the plan are the previous step's once the
        policy stops being queried, and a gap in the viewer is easier to read
        than a stale line.
        """
        self.step_count += 1
        if self.telemetry is None:
            return
        diagnostics = {
            "observation_age_s": observation.age,
            "head_sync_offset_ms": observation.head_sync_offset_ms,
            "running": float(self.running),
        }
        errors: dict[str, float] = {}
        scene = self._scene(action)
        if action is not None:
            diagnostics.update(
                {
                    "position_error_m": self.proxy.last_position_error,
                    "orientation_error_rad": self.proxy.last_orientation_error,
                    "unreachable_steps": float(self.proxy.unreachable_steps),
                    "inference_s": timing.inference_s if timing else 0.0,
                    "robotiq_ctrl": float(np.ravel(action["gripper"])[0]),
                }
            )
            errors = {
                "retarget_position_m": self.proxy.last_position_error,
                "retarget_orientation_rad": self.proxy.last_orientation_error,
            }
            commanded_pose = (
                None if self._commanded is None else self._commanded.tool_pose(commanded)
            )
            reported_pose = (
                None
                if reported_arm is None
                else self.proxy.franka_tool_pose_to_world(self.proxy.franka.fk(reported_arm))
            )
            errors.update(
                tool_errors(scene.stretch_tool, commanded_pose, scene.franka_tool, reported_pose)
            )
        self.telemetry.log_step(
            step=self.step_count,
            images={"head": head, "wrist": wrist},
            raw={"head": observation.head_rgb, "wrist": observation.wrist_rgb},
            measured=measured,
            commanded=commanded,
            diagnostics=diagnostics,
            task=self.task,
            errors=errors,
            scene=scene,
            timing=timing,
            desync=desync,
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
    "--execute-horizon",
    type=int,
    default=None,
    help="Actions of each chunk to send before querying the model again. Defaults to "
    "the policy config's (8). Lower re-plans on fresher frames, at a forward pass each "
    "time. See set_chunking.",
)
@click.option(
    "--action-horizon",
    type=int,
    default=None,
    help="Keep only the first N actions of each predicted chunk. Defaults to the "
    "checkpoint's own chunk length (16). Cannot make the model predict more -- the "
    "length is the checkpoint's -- and must be at least --execute-horizon.",
)
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
    "setups.STRETCH_GRASP_OFFSET_M for the sim setups' default (-0.009 on the SG4) and why.",
)
@click.option(
    "--tool-offset-x-m",
    type=float,
    default=0.0,
    show_default=True,
    help="Move the commanded grasp centre forward along the approach, on top of "
    "--grasp-offset-m: the knob for the Franka's wrist camera sitting nearer its fingers "
    "than Stretch's does. +0.101 (SG4) or +0.033 (PG4) puts Stretch's gripper camera where "
    "the Franka's is. See cameras.RetargetParams.tool_offset_x_m.",
)
@click.option(
    "--tool-offset-y-m",
    type=float,
    default=0.0,
    show_default=True,
    help="Move the commanded grasp centre along the jaw line. +0.041 (right gripper "
    "camera) or +0.021 (left) puts Stretch's camera level with the Franka's across the "
    "hand. See cameras.RetargetParams.tool_offset_x_m.",
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
    "--snap-to-franka-home/--no-snap-to-franka-home",
    default=True,
    show_default=True,
    help="Move the arm to the Franka's home pose before the first instruction. This is a "
    "real motion of the real robot, and it is the first thing to watch.",
)
@click.option(
    "--change_franka_start_pose_limit_height",
    is_flag=True,
    help="Cap the Franka's start tool height at Stretch's own reach ceiling "
    "(1.0824 m). `home` then leaves the lift at its stop by construction, so any action "
    "asking for more height is unreachable and says so.",
)
@click.option(
    "--change_stretch_start_pose_pitch_deg",
    type=float,
    default=0.0,
    help="Pitch the wrist by this many degrees about the jaw line at the start pose, "
    "i.e. at `home`. Does not change the frame actions are interpreted in.",
)
@click.option(
    "--slow",
    is_flag=True,
    help=f"Move at {SLOW_SPEED_SCALE:.0%} of the robot's max profile, in velocity and "
    "acceleration alike, and cap the base at the same fraction. The targets are "
    "unchanged, so the robot lags the stream further; read a slow rollout as a "
    "rehearsal rather than a measurement.",
)
@click.option(
    "--step-limit-scale",
    type=float,
    default=1.0,
    show_default=True,
    help="Scale the per-step target clamps in MAX_TARGET_STEP. Below 1 is more cautious.",
)
@click.option(
    "--wait-for-arrival/--no-wait-for-arrival",
    default=False,
    show_default=True,
    help="After each step, wait for the robot to stop moving (RobotClient.is_moving) and "
    "for a frame taken after that, before the next step. Every joint_pos_rel delta then "
    "starts from where the last one arrived, at the cost of a stop-and-go robot.",
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
    "--rerun-3d/--no-rerun-3d",
    default=True,
    show_default=True,
    help="Draw Stretch and the virtual Franka it is imitating in one 3D view. Costs a "
    "transform per visual geom per step; turn it off on a machine the policy is "
    "already saturating.",
)
@click.option(
    "--rerun-urdf/--no-rerun-urdf",
    default=True,
    show_default=True,
    help="Add a third robot to that view: this robot's own URDF, posed from "
    "RobotClient's status rather than from the camera stream. Needs stretch4_urdf and "
    "yourdfpy, and a robot -- it is skipped under --dry-run.",
)
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
    task: str,
    checkpoint: str | None,
    control_hz: float,
    execute_horizon: int | None,
    action_horizon: int | None,
    include_base: bool,
    head_mode: str,
    head_crop: str | None,
    head_virtual_pitch_deg: float | None,
    wrist_fov_deg: float,
    grasp_offset_m: float,
    tool_offset_x_m: float,
    tool_offset_y_m: float,
    wrist_tilt_deg: float,
    target_z_offset_m: float,
    aperture_m: float,
    snap_to_franka_home: bool,
    change_franka_start_pose_limit_height: bool,
    change_stretch_start_pose_pitch_deg: float,
    slow: bool,
    step_limit_scale: float,
    wait_for_arrival: bool,
    max_obs_age: float,
    rerun: bool,
    rerun_3d: bool,
    rerun_urdf: bool,
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
    click.echo(
        f"  rate       : {control_hz} Hz"
        + (", each step waits for the robot to settle (--wait-for-arrival)" if wait_for_arrival else "")
    )
    click.echo(
        f"  speed      : {SLOW_SPEED_SCALE * 100:.0f}% of the max profile (--slow)"
        if slow
        else "  speed      : the robot's max profile"
    )
    click.echo(f"  python     : {sys.executable}")
    conventions = _publish_conventions(
        change_franka_start_pose_limit_height=change_franka_start_pose_limit_height,
        change_stretch_start_pose_pitch_deg=change_stretch_start_pose_pitch_deg,
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
            "`closest_joint_state`, whose gripper entry is also the only gripper state it "
            "can read."
        )

    # The tool, before anything that depends on it: the model the retargeting
    # solves on and the units the gripper is read and commanded in. The robot's
    # own status decides (`connect_robot`); the stream's word is what a dry run
    # has instead, and what the stand-in fleet directory is written for, when
    # there has to be one, so that it matches the robot too.
    streamed_joint = stream_gripper_joint(first)
    if streamed_joint is None:
        click.secho(
            "  the stream does not say which gripper it is reporting, which is a sender "
            "from before the parallel gripper. On a PG4 that sender streams a joint state "
            "that never updates -- update it on the robot.",
            fg="yellow",
        )
    gripper_joint = streamed_joint or GRIPPER_JOINT
    robot = None
    if not dry_run:
        ensure_fleet_directory(tool_name=TOOL_FOR_GRIPPER_JOINT[gripper_joint])
        robot, gripper_joint = connect_robot(robot_ip)
        if streamed_joint not in (None, gripper_joint):
            click.secho(
                f"  the stream says {streamed_joint} but the robot reports {gripper_joint}; "
                "going by the robot. Restart the sender on the robot.",
                fg="yellow",
            )
    publish_stretch4_tool(TOOL_FOR_GRIPPER_JOINT[gripper_joint])

    click.echo("Building the kinematic mirror...")
    _, _, view, namespace = build_mirror()
    gripper_units = GripperUnits.for_joint(
        gripper_joint, view.get_move_group("gripper").kind, robot
    )
    click.echo(f"  gripper    : {gripper_units.describe()}")
    mirror = RobotMirror(view, gripper_units)
    measured = mirror.sync(first)
    click.echo(f"  robot is at: {({k: round(v, 3) for k, v in measured.items()})}")

    proxy = fr.FrankaOnStretchView(
        view,
        namespace,
        fr.franka_mount_pose_from_base(view.get_move_group("base").joint_pos),
        include_base=include_base,
        target_z_offset=target_z_offset_m,
        pose_conventions=conventions,
    )
    # The study's tool parameters, applied the way `setups.py` applies them to a
    # trial, so a run here is the same retargeting a sim trial measured.
    apply_tool_correction(
        proxy,
        wrist_tilt_deg=wrist_tilt_deg,
        grasp_offset_m=grasp_offset_m,
        tool_offset_x_m=tool_offset_x_m,
        tool_offset_y_m=tool_offset_y_m,
    )
    apply_aperture(proxy, aperture_m)
    proxy.reset()
    click.echo(
        f"  retarget   : grasp offset {grasp_offset_m:+.4f}m, "
        f"tool offset ({tool_offset_x_m:+.4f}, {tool_offset_y_m:+.4f})m, "
        f"wrist tilt {wrist_tilt_deg:+.1f}deg, "
        f"z offset {target_z_offset_m:+.4f}m, {proxy.gripper_kind.name} jaw, "
        f"opens to {proxy.finger_open:.4f} {proxy.gripper_kind.unit}, "
        f"base {'in' if include_base else 'out of'} the IK"
    )

    click.echo("Loading the checkpoint...")
    from examples.machine_learning.molmospaces.demo_droid_on_stretch import load_droid_policy

    policy = load_droid_policy(checkpoint)
    kept, executed, predicted = set_chunking(policy, action_horizon, execute_horizon)
    click.echo(
        f"  chunking   : execute {executed} of {kept} actions per model query"
        + (f" (of the {predicted} the checkpoint predicts)" if kept < predicted else "")
    )

    telemetry = None
    if rerun or rrd is not None:
        telemetry = RerunTelemetry(
            save_path=rrd,
            jpeg_quality=rerun_jpeg_quality,
            scene=(
                (lambda rr: RerunRobotScene(rr, mirror.model, namespace, proxy.franka.model))
                if rerun_3d
                else None
            ),
            urdf=urdf_overlay_builder(robot) if (rerun_3d and rerun_urdf) else None,
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
            sent = runner.step()
            if sent and wait_for_arrival and commander is not None:
                # Inside the period, so `--control-hz` stays a ceiling: a step that
                # settles quickly still waits out the rest of it.
                if not _wait_for_arrival(console, runner, commander, stream):
                    break
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


def _wait_for_arrival(
    console: Console, runner: RealStretchRunner, commander: RobotCommander, stream: RobotStream
) -> bool:
    """`--wait-for-arrival`: hold the loop until the robot has settled and been seen settled.

    Two waits. `RobotCommander.wait_until_settled` for the motion, and then one
    for a stream message received after it -- the policy's arm state comes from
    the stream, not from `RobotClient`'s status, and the newest message when the
    motion ends can still be one taken mid-reach. "Received after" is the most
    this side can check: the sender's clock is not this one (see
    `RobotObservation.received_at`), so the frame may have been taken up to one
    network latency before the robot stopped. The second wait gives up after
    `--max-obs-age`, where `step` pauses on a stale stream anyway.

    The console is served throughout, so a stop key lands mid-wait, and any wait
    ends as soon as the runner is no longer running. False means the run should
    end.
    """
    quit_requested = False

    def keep_waiting() -> bool:
        nonlocal quit_requested
        if not _handle_console(console, runner):
            quit_requested = True
        return not quit_requested and runner.running

    arrived = commander.wait_until_settled(ARRIVAL_TIMEOUT_S, keep_waiting)
    if not arrived and keep_waiting():
        runner._warn_once(
            "arrival",
            f"The robot was still moving {ARRIVAL_TIMEOUT_S:.1f}s after a step was sent; "
            "carrying on without it. Said once.",
        )

    settled_at = time.monotonic()
    while keep_waiting() and time.monotonic() - settled_at < runner.settings.max_obs_age:
        latest = stream.latest()
        if latest is not None and latest.received_at > settled_at:
            break
        time.sleep(ARRIVAL_POLL_S)
    return not quit_requested


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
