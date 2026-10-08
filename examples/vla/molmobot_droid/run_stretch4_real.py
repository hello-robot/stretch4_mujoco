"""
MolmoBot-DROID driving a real Stretch 4, through Franka retargeting, continuously: type an
instruction and press Enter, and the robot works on it until told otherwise (see KEYS_HELP).
It asks before it first moves the robot to the start pose.

Rerun shows the robot's cameras and pose live from the moment they arrive, and, while an
instruction runs, the policy's views and the ghost Franka it is retargeted from.

On the robot, run the image + joint-state publisher from stretch4_compliant_gripper first:

    python send_gripper_and_head_images_with_joint_states.py -r --head_camera left --wrist_camera left --wrist_resolution 400

with its head and wrist camera sides matching --exo_camera and --gripper_camera. Commands go
through stretch4_body's `RobotClient`; with --wait-for-arrival (the default) each action waits
for the robot to stop moving.

There is no scene to read the target's height from, so give it with --object-height: the
virtual Franka stands where molmospaces would put it for an object at that height.

Usage:
    python -m examples.vla.molmobot_droid.run_stretch4_real --robot_ip 10.0.0.12 --object-height 0.80 \\
        --exo_camera left --gripper_camera left --slow
"""

from __future__ import annotations

import math
import os
import queue
import sys
import threading
import time

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
    FrankaSpawn,
    Observation,
    RobotPose,
    add_franka_ghost,
    franka_link0_height_for_object,
    mat_to_quat,
)
from examples.vla.molmobot_droid.franka_retarget.stretch4_retarget import (
    MIN_BASE_ROTATION,
    SLOW_FACTOR,
    SPAWN_LIFT_FRACTION,
    STEP_ARRIVAL_TIMEOUT,
    FrankaStretchRetargeter,
    RetargetParams,
    StretchJoints,
    StretchTargets,
    params_from_kwargs,
    planar_transform,
    prepare_exo,
    retarget_options,
    wrist_view,
)
from examples.vla.molmobot_droid.rollout import StepInfo, run_rollout

IMAGE_PORT = 4409
"""`gripper_networking.gripper_and_joints_port` in stretch4_compliant_gripper."""

GRIPPER_PCT_OPEN = 300.0
"""stretch4_body StretchGripper `pct_max_open` for SE4 (range_deg [-100, 300]); 0 is fingertips touching."""

PARALLEL_GRIPPER_OPEN_MM = 77.0
"""The parallel gripper's widest opening: two fingers of 40 mm travel (the PG4 URDF); 0 is closed."""

# stretch4_body SE4 default motion profiles; --slow runs at SLOW_FACTOR of them.
DEFAULT_SPEEDS = {"lift": 0.3, "arm": 0.4, "wrist": 7.0, "base_rotate": 2.0}


KEYS_HELP = (
    "  Type an instruction + Enter   run it (typing another while one runs switches to it)\n"
    "  Enter or Space                stop the robot where it is\n"
    "  'home' + Enter                  raise the lift, go back to the start pose\n"
    "  'quit' + Enter, or Ctrl+C       stop and exit"
)

MOTION_SUBSYSTEMS = ["arm", "lift", "omnibase", "end_of_arm"]
LIFT_RANGE = (0.0, 1.2)


class Console:
    """
    The terminal while the robot runs: a status line at the bottom with what is being typed
    after it, events printed above it, and commands queued for the control loop as they are
    typed -- ("stop", None) the moment Enter or Space is pressed on an empty line, ("text",
    line) when a line is entered. Falls back to plain lines when stdin is not a terminal.
    """

    PROMPT = click.style("› ", fg="cyan", bold=True)

    def __init__(self):
        self.queue: queue.Queue[tuple[str, str | None]] = queue.Queue()
        self._lock = threading.Lock()
        self._status = ""
        self._buffer = ""
        self._terminal = None
        if sys.stdin.isatty():
            import termios
            import tty

            self._terminal = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin)  # keys arrive one at a time; Ctrl+C still interrupts
        threading.Thread(target=self._read, daemon=True).start()
        self._render()

    # -- output -------------------------------------------------------------

    def _render(self) -> None:
        if self._terminal is None:
            return
        status = f"{self._status}  " if self._status else ""
        sys.stdout.write(f"\r\x1b[2K{status}{self.PROMPT}{self._buffer}")
        sys.stdout.flush()

    def event(self, message: str, color: str | None = None) -> None:
        """A line above the status line."""
        with self._lock:
            if self._terminal is not None:
                sys.stdout.write("\r\x1b[2K")
            click.secho(message, fg=color)
            self._render()

    def status(self, text: str) -> None:
        with self._lock:
            self._status = text
            self._render()

    # -- input --------------------------------------------------------------

    def _read(self) -> None:
        if self._terminal is None:
            for line in sys.stdin:
                text = line.strip()
                self.queue.put(("text", text) if text else ("stop", None))
            self.queue.put(("text", "quit"))
            return
        while True:
            char = os.read(sys.stdin.fileno(), 1).decode(errors="ignore")
            with self._lock:
                if not char:
                    self.queue.put(("text", "quit"))
                    return
                if char in ("\n", "\r") or (char == " " and not self._buffer):
                    text, self._buffer = self._buffer.strip(), ""
                    self.queue.put(("text", text) if text else ("stop", None))
                elif char in ("\x7f", "\b"):
                    self._buffer = self._buffer[:-1]
                elif char.isprintable():
                    self._buffer += char
                self._render()

    def pending(self) -> bool:
        return not self.queue.empty()

    def get(self) -> tuple[str, str | None]:
        return self.queue.get()

    def close(self) -> None:
        if self._terminal is not None:
            import termios

            with self._lock:
                sys.stdout.write("\r\x1b[2K")
                sys.stdout.flush()
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._terminal)
                self._terminal = None


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


class ImageAndJointReceiver:
    """
    Subscribes to the robot's synced wrist + head frames and joint states on a thread, as
    recv_gripper_and_head_images_with_joint_states.py does, keeping only the newest message,
    decoded. `receive()` waits for the next one; `latest()` does not wait.
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
            head = _decode(message, "head_color_image_compressed", "head_color_image")
            message["head_rgb"] = (
                rotate_head_image_to_upright(head, message.get("head_camera_side", "")) if head is not None else None
            )
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

    def __init__(self, object_height: float, tool_name: str):
        from stretch4_mujoco.stretch4_mujoco_simulator import Stretch4MujocoSimulator

        spec = mujoco.MjSpec.from_file(Stretch4MujocoSimulator.get_robot_xml_path(tool_name))
        # The wheels' contact pairs name a geom "floor".
        spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.1])
        self.franka = add_franka_ghost(spec, RobotPose(0.0, 0.0, 0.0), franka_link0_height_for_object(object_height))
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

    def pose(self, world_from_footprint: np.ndarray, joints: StretchJoints, franka_q7) -> None:
        data = self.data
        data.joint("floating_base").qpos = np.concatenate(
            [world_from_footprint[:3, 3], mat_to_quat(world_from_footprint[:3, :3])]
        )
        data.joint("lift_joint").qpos = joints.lift
        for i in range(1, 5):
            data.joint(f"arm_l{i}_joint").qpos = joints.arm / 4
        for joint in ("wrist_yaw", "wrist_pitch", "wrist_roll"):
            data.joint(f"{joint}_joint").qpos = getattr(joints, joint)
        for i, value in enumerate(franka_q7):
            data.joint(f"{self.franka.prefix}fr3_joint{i + 1}").qpos = value
        mujoco.mj_kinematics(self.model, data)


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
        self.interrupted = lambda: False
        """Polled while waiting on the robot; True cuts the wait short (a key was pressed)."""
        self.check_camera_sides()

    def check_camera_sides(self) -> None:
        message = self.receiver.receive()
        sides = {"head": message.get("head_camera_side"), "wrist": message.get("wrist_camera_side")}
        if self.params.exo_camera != sides["head"] or self.params.gripper_camera != sides["wrist"]:
            raise click.ClickException(
                f"The robot publishes head={sides['head']}, wrist={sides['wrist']}; run with "
                f"--exo_camera {sides['head']} --gripper_camera {sides['wrist']} or restart its publisher"
            )
        if message["head_rgb"] is None:
            raise click.ClickException("The robot is not publishing a head image")
        self._state(message)  # fixes the odometry origin: the world frame is where we start

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

    def observe(self) -> Observation:
        message = self.receiver.receive()
        self.last_message = message
        footprint, joints = self._state(message)
        state8, _ = self.retargeter.stretch_to_franka(footprint, joints)
        self.last_state8 = state8
        return Observation(
            exo_rgb=prepare_exo(message["head_rgb"], self.params),
            wrist_rgb=wrist_view(message["wrist_rgb"], self.retargeter.tcp_flipped),
            state8=state8,
            extra_cameras={"head_raw": message["head_rgb"], "gripper_raw": message["wrist_rgb"]},
        )

    # -- acting -------------------------------------------------------------

    def step(self, action8: np.ndarray) -> StretchTargets | None:
        footprint, joints = self._state(self.receiver.receive())
        targets = self.retargeter.franka_to_stretch(action8, footprint, joints)
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

    def move_to_franka_pose(self, franka_q7=FRANKA_HOME_QPOS) -> None:
        self.retargeter.reset()
        self.retargeter.franka_seed = np.asarray(franka_q7, dtype=float)
        footprint, joints = self._state(self.receiver.receive())
        targets = self.retargeter.franka_to_stretch(np.concatenate([franka_q7, [0.0]]), footprint, joints)
        if targets is None:
            raise RuntimeError("Stretch 4 cannot reach the Franka's home pose; check --object-height")
        self._gripper_closed = None
        self.send(targets)
        self.wait_for_arrival(30.0)

    def go_home(self, franka_q7=FRANKA_HOME_QPOS) -> None:
        """
        Back to the start pose: the lift up to SPAWN_LIFT_FRACTION of its travel first, so the
        gripper clears whatever it is over, then where the Franka at `franka_q7` has its tool.
        """
        self.stop_motion()
        _, joints = self._state(self.receiver.receive())
        high = LIFT_RANGE[0] + SPAWN_LIFT_FRACTION * (LIFT_RANGE[1] - LIFT_RANGE[0])
        if joints.lift < high:
            self.robot.lift.move_to(high, v_m=self._speed("lift"))
            self.robot.push_command()
            if not self.wait_for_arrival(15.0):
                return
        self.move_to_franka_pose(franka_q7)

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
    """

    def __init__(self, env: RealStretch4Env, robot_model: RobotModel, hz: float = 10.0):
        self.env, self.robot_model, self.period = env, robot_model, 1.0 / hz
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
                rerun_scene.log_cameras({"head_raw": message["head_rgb"], "gripper_raw": message["wrist_rgb"]})
                self.robot_model.pose(*self._robot_state(message))
                self.scene.log(self.robot_model.data)
            time.sleep(self.period)

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)


def continuous_session(
    env: RealStretch4Env, policy, params: RetargetParams, max_steps: int, on_step, console: Console
) -> None:
    """
    The control loop: runs the current instruction until a key says otherwise (KEYS_HELP),
    indefinitely unless `max_steps` > 0.
    """
    env.interrupted = console.pending
    command = console.get()
    while True:
        kind, text = command
        if kind == "stop":
            env.stop_motion()
            console.status("")
            console.event("■ stopped", "yellow")
        elif text in ("quit", "exit", "q"):
            return
        elif text == "home":
            console.status(click.style("◆ going home", fg="yellow"))
            try:
                env.go_home()
                console.event("◆ home" if not console.pending() else "◆ home (interrupted)", "yellow")
            except RuntimeError as error:
                console.event(f"◆ {error}", "red")
            console.status("")
        else:
            console.event(f"▶ {text}", "green")

            def step(info: StepInfo) -> None:
                on_step(info)
                error = env.tool_error_m()
                console.status(
                    click.style(f"▶ {text}", fg="green")
                    + click.style(
                        f"  step {info.step}"
                        + (f" · tool {error * 1000:.0f} mm off" if error is not None else "")
                        + (f" · {env.retargeter.ik_failures} unreachable" if env.retargeter.ik_failures else "")
                        + "  · Enter/Space stops",
                        dim=True,
                    )
                )

            result = run_rollout(
                env, policy, text, params.execute_horizon, params.execute_first_n,
                max_steps if max_steps > 0 else 10**12, step, should_stop=console.pending,
            )
            if not console.pending():
                env.stop_motion()
                console.status("")
                console.event(f"■ done after {result.steps} steps", "yellow")
        command = console.get()


@click.command()
@click.option("--robot_ip", "--robot-ip", "robot_ip", required=True, help="Stretch 4's IP address.")
@click.option("--object-height", type=float, default=0.80, show_default=True,
              help="Height of the target object above the floor (m); sets the virtual Franka's height.")
@click.option("--max-steps", type=int, default=0, show_default=True,
              help="Stop an instruction after this many steps (15/s). 0: run until told otherwise.")
@click.option("--checkpoint", default=None)
@click.option("--rerun/--no-rerun", default=True, show_default=True)
@retarget_options
def main(robot_ip, object_height, max_steps, checkpoint, rerun, **kwargs):
    params = params_from_kwargs(kwargs)
    if params.exo_camera == "droid":
        raise click.BadParameter("the real robot has no DROID exo camera; use left, right or center", param_hint="--exo_camera")

    click.secho(f"Connecting to Stretch 4 at {robot_ip}...", dim=True)
    robot, gripper = connect(robot_ip)
    if not robot.is_homed():
        robot.stop()
        raise click.ClickException("Home the robot first")

    # The gripper on the robot decides the tool, its kinematics and its default grasp offset.
    detected_parallel = gripper == "parallel_gripper"
    if params.use_parallel_gripper != detected_parallel:
        if params.use_parallel_gripper:
            click.secho(f"--use_parallel_gripper was given, but the robot has a {gripper}; using it.", fg="yellow")
        params.use_parallel_gripper = detected_parallel
    robot_model = RobotModel(object_height, params.tool_name)
    receiver = ImageAndJointReceiver(robot_ip)
    env = live = console = None
    try:
        env = RealStretch4Env(robot, receiver, params, robot_model.franka)
        offset = ",".join(f"{v:g}" for v in params.effective_grasp_offset_mm)
        click.echo(
            click.style("Stretch 4 ", bold=True) + f"{robot_ip} · {gripper.replace('_', ' ')} · grasp offset {offset} mm\n"
            + click.style("Cameras  ", bold=True) + f"head {params.exo_camera}, gripper {params.gripper_camera}"
            + (" (head cropped to DROID)" if params.head_crop == "droid" else "") + "\n"
            + click.style("Motion   ", bold=True) + ("slow" if params.slow else "full speed")
            + (", waiting for each action to arrive" if params.wait_for_arrival else "")
            + f", {params.execute_first_n} of every {params.execute_horizon} predicted actions"
        )
        if rerun:
            rerun_scene.init_rerun("MolmoBot-DROID Stretch 4 (real)", ["exo", "wrist", "head_raw", "gripper_raw"])
            live = LiveView(env, robot_model)
            click.secho("Rerun is showing the cameras live.", dim=True)

        click.secho("Loading MolmoBot-DROID...", dim=True)
        policy = load_policy(checkpoint)

        if click.confirm(click.style("Raise the lift and move Stretch 4 to the start pose?", bold=True), default=True):
            click.secho("Moving to the start pose...", dim=True)
            env.go_home()
        else:
            click.secho("Staying put; type home when ready.", dim=True)

        click.echo("\n" + KEYS_HELP + "\n")
        console = Console()
        step_count = 0

        def on_step(info: StepInfo) -> None:
            nonlocal step_count
            step_count += 1
            if not rerun:
                return
            rerun_scene.set_step(step_count)
            rerun_scene.rr.set_time("time", timestamp=time.time())
            rerun_scene.log_cameras({"exo": info.observation.exo_rgb, "wrist": info.observation.wrist_rgb})
            footprint, joints = env._state(env.last_message)
            rerun_scene.log_tool_poses(
                env.retargeter.last_target_tool_world, env.retargeter.stretch_tool_world(footprint, joints)
            )

        continuous_session(env, policy, params, max_steps, on_step, console)
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
        unload_policy()
        receiver.close()
        robot.stop()


if __name__ == "__main__":
    main()
