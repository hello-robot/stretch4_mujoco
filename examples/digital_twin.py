"""
Digital twin: keep a real Stretch 4 and the MuJoCo sim in sync.

Requires the `digital-twin` extra: `uv pip install -e ".[digital-twin]"`.

The bridge runs in two independent directions, selected with `--controller`:

sim -> robot (`--controller sim`)
    Commanding a joint in the sim hands that joint to the sim, and the sim's
    resulting position is streamed to the robot as `RobotClient.move_to()` at
    the robot's `max` motion profile, until the sim stops commanding it. The
    trigger is installed on the simulator's command boundary (`_move_to` /
    `_move_by` / `_set_joint_velocity` / `_set_base_velocity`), so anything that
    drives the sim -- the teleop in this example, `sim.arm.move_to(...)`, your
    own script -- drives the robot too.

    Positions rather than the sim's own relative commands, because a `move_by`
    stream re-plans the robot's trapezoidal profile from a standstill every
    cycle: the joint spends each tick in the opening of a ramp and crawls.
    Streaming an absolute target keeps it out ahead of the robot, which is what
    makes the follower track in `stretch_puppet_teleop.py`. The base is the
    exception and is still forwarded command-for-command -- its motions are
    one-shot `translate_by` / `rotate_by` or a velocity, neither of which
    restarts a profile. Everything a tick produces goes out under a single
    `push_command()`, which is how `stretch4_body` expects a loop to batch.

robot -> sim (`--controller robot`)
    The robot's *joint status* is followed, not its commands: whatever moves the
    real robot -- a backdriven arm, another process, the gamepad on the robot --
    shows up in the sim. Joints are position-matched; the base is velocity-
    matched from the robot's body-frame odometry velocity, since the sim base
    has no pose reset.

bidirectional (default)
    Both of the above. The two directions would otherwise fight: the real robot
    lags the sim, so its status would drag the sim back toward where the sim
    used to be. A per-joint latch resolves it -- while the sim leads a joint the
    robot's status for it is ignored, and the joint is released once the robot
    catches up, or once it has failed to for `settle_s` (at which point reality
    wins, which is the point of a twin). Releasing it is also what lets the
    robot be backdriven: a joint the sim is no longer commanding stops being
    held at the sim's last setpoint.

The mechanism follows `stretch4_body/tools/stretch_puppet_teleop.py`: connect a
`RobotClient` to `--robot_ip`, check it is homed, stream the leader's measured
positions with generous velocity and acceleration limits, and `push_command()`
once per control cycle.

"""

from __future__ import annotations

import math
import os
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import click
import mujoco
import numpy as np

from stretch4_mujoco.config import robot_settings_se4
from stretch4_mujoco.enums.actuators import Actuators
from stretch4_mujoco.stretch4_mujoco_simulator import Stretch4MujocoSimulator

# =============================================================================
# The fleet directory stretch4_body reads before it will import
# =============================================================================

NOMINAL_FLEET_PATH = Path(tempfile.gettempdir()) / "stretch4_mujoco_fleet"
NOMINAL_FLEET_ID = "stretch-se4-nominal"
"""Where `ensure_fleet_directory` puts the stand-in, and what it calls it.

Named rather than random so a run reuses the previous run's directory instead of
leaving one behind per launch, and "nominal" rather than a plausible serial
number because anything that prints the fleet id should say what this is.
"""


def ensure_fleet_directory(tool_name: str | None = None) -> Path:
    """Give `stretch4_body` a fleet directory to read, inventing one if there is none.

    `RobotParams` reads `$HELLO_FLEET_PATH/$HELLO_FLEET_ID/` **while it is being
    imported** -- the two YAMLs are read in the class body -- and calls
    `sys.exit(1)` when they are not there. That is fine on a robot, where the
    variables are in the shell, and it is why importing `RobotClient` on a
    workstation dies with `KeyError: 'HELLO_FLEET_PATH'` before any of this
    repository's code gets to run. This writes the smallest directory that
    satisfies it, under the system temporary directory, and exports the two
    variables.

    What it costs, and it is worth knowing before trusting a number: the
    parameters `stretch4_body` then builds are the **nominal** ones for this model
    and tool, not the ones calibrated for the robot being driven. For commanding
    a robot over the network that is mostly harmless -- the joint targets are in
    SI units and the robot's own server applies its calibration to them -- but
    anything read back from `robot.robot_params` here is a catalogue value.
    `GripperMirror`'s `range_deg` is the one this repository actually reads.
    Copy the robot's own fleet directory over and set the variables yourself when
    that matters; this never overwrites an environment that already names one.

    `tool_name` is the end of arm to declare, and it has to be the tool that is
    physically on the robot or the client builds the wrong end of arm -- a
    `stretch_gripper` where there is a `parallel_gripper`. It defaults to the
    same tool the simulator builds its own model with, so the two halves of a
    twin describe one robot.

    Returns the fleet directory in use, spoofed or not.
    """
    fleet_path, fleet_id = os.environ.get("HELLO_FLEET_PATH"), os.environ.get("HELLO_FLEET_ID")
    if fleet_path and fleet_id:
        return Path(fleet_path) / fleet_id

    model_name, _, default_tool = Stretch4MujocoSimulator.get_default_model_batch_tool_names()
    tool_name = tool_name or default_tool
    directory = NOMINAL_FLEET_PATH / NOMINAL_FLEET_ID
    directory.mkdir(parents=True, exist_ok=True)
    # `nominal_system_params` builds a rotating log file handler at
    # `$HELLO_FLEET_PATH/log/stretch_body_logger/` -- beside the fleet directory,
    # not inside it -- and a handler whose directory does not exist raises the
    # moment anything configures logging.
    (NOMINAL_FLEET_PATH / "log" / "stretch_body_logger").mkdir(parents=True, exist_ok=True)

    # Only `robot.model_name` and `robot.tool` are load-bearing: the first picks
    # the `robot_params_<model>` module the nominal parameters come from, the
    # second the end of arm expanded out of it. Everything else in a real
    # configuration file is this robot's calibration, which is exactly what a
    # stand-in cannot invent.
    (directory / "stretch_configuration_params.yaml").write_text(
        "# Written by examples/digital_twin.py: ensure_fleet_directory().\n"
        "# A stand-in for a real robot's fleet directory: nominal parameters for\n"
        "# this model and tool, and no calibration. Delete it to have it rebuilt.\n"
        "robot:\n"
        f"  model_name: {model_name}\n"
        f"  tool: {tool_name}\n"
        "  batch_name: nominal\n"
        "  serial_no: nominal\n"
    )
    # Read second and overlaid on the above, so the user half of the parameters
    # is deliberately empty: there is no user to have tuned anything here.
    (directory / "stretch_user_params.yaml").write_text(
        "# Written by examples/digital_twin.py: ensure_fleet_directory().\n"
        "{}\n"
    )

    os.environ["HELLO_FLEET_PATH"] = str(NOMINAL_FLEET_PATH)
    os.environ["HELLO_FLEET_ID"] = NOMINAL_FLEET_ID
    click.secho(
        f"HELLO_FLEET_PATH was not set, so stretch4_body is reading {directory} instead: "
        f"nominal {model_name} parameters with {tool_name}, and none of this robot's "
        "calibration. Set HELLO_FLEET_PATH and HELLO_FLEET_ID to a copy of the robot's "
        "own fleet directory to use its.",
        fg="yellow",
    )
    return directory


# Before anything imports `stretch4_body`, which this module does lazily in
# `_connect` -- and which is far enough down that the import would otherwise be
# the first thing on this machine to discover the variables are missing.
ensure_fleet_directory()

# Base channels are mirrored as relative motions / velocities rather than
# positions, so they get their own names in the pending-command table.
BASE_TRANSLATE = "base_translate"
BASE_ROTATE = "base_rotate"
BASE_VELOCITY = "base_velocity"
GRIPPER = "gripper"


@dataclass(frozen=True)
class MirroredJoint:
    """A joint the sim and the robot command in the same units and sign.

    The wrist roll included: `RobotClient` reports it in the URDF's own
    convention, which is what stretch4_body's self-collision and IK feed the
    URDF, so a sim roll *is* the robot's roll.
    """

    name: str
    actuator: Actuators
    group: str
    subsystem: str
    """`"lift"` / `"arm"` for `robot.<name>.move_to()`, `"end_of_arm"` for
    `robot.end_of_arm.move_to(name, ...)`."""
    deadband: float
    """How far the robot may differ from the sim before the sim is corrected.
    Keeps sensor noise from turning into a stream of sim commands."""


MIRRORED_JOINTS = (
    MirroredJoint("lift", Actuators.lift, "lift", "lift", deadband=0.003),
    MirroredJoint("arm", Actuators.arm, "arm", "arm", deadband=0.003),
    MirroredJoint("wrist_yaw", Actuators.wrist_yaw, "wrist", "end_of_arm", deadband=0.01),
    MirroredJoint("wrist_pitch", Actuators.wrist_pitch, "wrist", "end_of_arm", deadband=0.01),
    MirroredJoint(
        "wrist_roll",
        Actuators.wrist_roll,
        "wrist",
        "end_of_arm",
        deadband=0.01,
    ),
)

JOINT_GROUPS = ("base", "lift", "arm", "wrist", "gripper")

# What to follow the sim with, all taken from `stretch_puppet_teleop.py`, which
# streams its puppet the same way. The robot's own `max` profile for lift and
# arm; the end of arm's servos are given a flat rate rather than a profile.
MOTION_PROFILE = "max"
EOA_VELOCITY_R = 12.0
EOA_ACCELERATION_R = 10.0
LIFT_ACCELERATION_SCALE = 0.7

# Actuators that are driven by the mirrored joints above rather than commanded
# directly, or that Stretch 4 does not have. Commanding them in sim is not an
# error, there is just nothing to send to the robot.
IGNORED_ACTUATORS = (
    Actuators.head_pan,
    Actuators.head_tilt,
    Actuators.gripper_left_finger,
    Actuators.gripper_right_finger,
    Actuators.left_wheel_vel,
    Actuators.right_wheel_vel,
    Actuators.back_wheel_vel,
)


def _aperture_angle_rad(aperture_m: float, finger_length_m: float) -> float:
    """An aperture, measured fingertip to fingertip, as the angle between fingers.

    The chord-over-radius that `MujocoServer` and stretch_body's
    `GripperConversion` both use.
    """
    return 2 * math.asin(aperture_m / (2 * finger_length_m))


def _sim_aperture_range(sim: Stretch4MujocoSimulator) -> tuple[float, float]:
    """The closed and fully-open gripper aperture, in the radians the sim commands.

    Stretch 4 drives the gripper as two finger joints, so `pull_joint_limits()`
    reports limits for the fingers and not for the aperture that
    `Actuators.gripper` is commanded in. Fall back to deriving the aperture range
    from the same `gripper_conversion` settings the server converts through.
    """
    limits = sim.pull_joint_limits()
    if Actuators.gripper in limits:
        return limits[Actuators.gripper]

    conversion = robot_settings_se4["gripper_conversion"]
    finger_length_m = conversion["finger_length_m"]
    return (
        _aperture_angle_rad(conversion["aperture_closed_m"], finger_length_m),
        _aperture_angle_rad(conversion["aperture_open_m"], finger_length_m),
    )


class GripperMirror:
    """Converts between the sim's gripper aperture and the robot's tool units.

    The sim commands the gripper as an aperture angle in radians; the robot
    commands `stretch_gripper` in percent and `parallel_gripper` in meters. Both
    are mapped through a normalized 0.0 (closed) - 1.0 (open) fraction, so the
    conversion works for either tool.
    """

    def __init__(self, sim: Stretch4MujocoSimulator, robot):
        self.sim_min, self.sim_max = _sim_aperture_range(sim)

        eoa_joints = getattr(robot.end_of_arm, "joints", [])
        if "parallel_gripper" in eoa_joints:
            self.joint = "parallel_gripper"
            range_mm = robot.robot_params.get("parallel_gripper", {}).get("range_mm", 80.0)
            self.robot_min, self.robot_max = 0.0, range_mm / 1000.0
        elif "stretch_gripper" in eoa_joints:
            self.joint = "stretch_gripper"
            range_deg = robot.robot_params.get("stretch_gripper", {}).get("range_deg")
            pct_max_open = (
                100 * abs(range_deg[1] / range_deg[0]) if range_deg and range_deg[0] else 100.0
            )
            self.robot_min, self.robot_max = -100.0, pct_max_open
        else:
            raise ValueError(f"No gripper found on the robot's tool. Joints: {eoa_joints}")

        self._robot = robot

    def to_robot(self, sim_value: float) -> float:
        fraction = self._fraction(sim_value, self.sim_min, self.sim_max)
        return self.robot_min + fraction * (self.robot_max - self.robot_min)

    def to_sim(self, robot_value: float) -> float:
        fraction = self._fraction(robot_value, self.robot_min, self.robot_max)
        return self.sim_min + fraction * (self.sim_max - self.sim_min)

    def robot_position(self) -> float | None:
        """The gripper's position from the robot's status, in robot units."""
        try:
            status = self._robot.status["end_of_arm"][self.joint]
        except KeyError:
            return None
        if self.joint == "parallel_gripper":
            return status.get("pos_mm", 0.0) / 1000.0
        return status.get("pos_pct", status.get("pos", 0.0))

    @staticmethod
    def _fraction(value: float, low: float, high: float) -> float:
        span = high - low
        if not span:
            return 0.0
        return min(max((value - low) / span, 0.0), 1.0)


class DigitalTwin:
    """Mirrors motion between a `Stretch4MujocoSimulator` and a `RobotClient`."""

    def __init__(
        self,
        sim: Stretch4MujocoSimulator,
        robot,
        controller: str = "bidirectional",
        groups: tuple[str, ...] = JOINT_GROUPS,
        hold_s: float = 0.5,
        settle_s: float = 3.0,
        follow_s: float = 2.0,
        base_linear_deadband: float = 0.01,
        base_angular_deadband: float = 0.02,
        debug: bool = False,
    ):
        self.debug = debug
        self.sim = sim
        self.robot = robot
        self.mirror_sim_to_robot = controller in ("sim", "bidirectional")
        self.mirror_robot_to_sim = controller in ("robot", "bidirectional")
        self.groups = groups
        self.hold_s = hold_s
        self.settle_s = settle_s
        self.follow_s = follow_s
        self.base_linear_deadband = base_linear_deadband
        self.base_angular_deadband = base_angular_deadband

        self.joints = tuple(j for j in MIRRORED_JOINTS if j.group in groups)
        self._joint_by_actuator = {joint.actuator: joint for joint in self.joints}
        self._joint_by_name = {joint.name: joint for joint in self.joints}
        self.gripper = GripperMirror(sim, robot) if "gripper" in groups else None
        self.mirror_base = "base" in groups
        self._motion_limits = {
            joint.name: self._joint_motion_limits(joint) for joint in self.joints
        }

        # Base commands the sim has issued and we have not yet pushed to the
        # robot, keyed by channel: {channel: (kind, value)}. Commanding the sim
        # happens on whatever thread drives it (a teleop thread, say), flushing
        # happens on the thread running step(), so the table is locked.
        self._pending: dict[str, tuple[str, object]] = {}
        self._lock = threading.Lock()

        # channel -> time of the most recent sim command. Decides which side
        # leads a joint, in both directions.
        self._sim_commanded_at: dict[str, float] = {}

        self._base_was_moving = False
        self._installed = False
        self._refused: set[str] = set()
        self._targets: dict[str, float] = {}
        self._last_debug = 0.0

        # The unwrapped sim commands. Following the robot's status calls these,
        # so the correction does not echo back out to the robot as a command.
        self._sim_move_to = sim._move_to
        self._sim_move_by = sim._move_by
        self._sim_set_joint_velocity = sim._set_joint_velocity
        self._sim_set_base_velocity = sim._set_base_velocity

    # -- sim -> robot ------------------------------------------------------

    def install(self) -> None:
        """Wrap the simulator's command methods so they also reach the robot."""
        if self._installed or not self.mirror_sim_to_robot:
            return

        def move_to(actuator, pos):
            self._record("move_to", actuator, pos)
            return self._sim_move_to(actuator, pos)

        def move_by(actuator, pos):
            self._record("move_by", actuator, pos)
            return self._sim_move_by(actuator, pos)

        def set_joint_velocity(actuator, v_m):
            self._record("set_velocity", actuator, v_m)
            return self._sim_set_joint_velocity(actuator, v_m)

        def set_base_velocity(v_x, v_y, omega):
            if self.mirror_base:
                self._queue(BASE_VELOCITY, "set_velocity", (v_x, v_y, omega))
            return self._sim_set_base_velocity(v_x, v_y, omega)

        self.sim._move_to = move_to
        self.sim._move_by = move_by
        self.sim._set_joint_velocity = set_joint_velocity
        self.sim._set_base_velocity = set_base_velocity
        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            return
        self.sim._move_to = self._sim_move_to
        self.sim._move_by = self._sim_move_by
        self.sim._set_joint_velocity = self._sim_set_joint_velocity
        self.sim._set_base_velocity = self._sim_set_base_velocity
        self._installed = False

    def _record(self, kind: str, actuator: str | Actuators, value: float) -> None:
        if isinstance(actuator, str):
            actuator = Actuators[actuator]

        if actuator in (Actuators.base_translate, Actuators.base_translate_y):
            if self.mirror_base and kind == "move_by":
                dx = value if actuator == Actuators.base_translate else 0.0
                dy = value if actuator == Actuators.base_translate_y else 0.0
                self._queue(BASE_TRANSLATE, "move_by", (dx, dy))
            return
        if actuator == Actuators.base_rotate:
            if self.mirror_base and kind == "move_by":
                self._queue(BASE_ROTATE, "move_by", value)
            return
        if actuator == Actuators.gripper:
            if self.gripper is not None:
                self._hand_to_sim(GRIPPER)
            return
        if actuator in IGNORED_ACTUATORS:
            return

        joint = self._joint_by_actuator.get(actuator)
        if joint is not None:
            self._hand_to_sim(joint.name)

    def _hand_to_sim(self, channel: str) -> None:
        """Note that the sim just commanded `channel`, so the sim leads it.

        The command itself is not forwarded. What reaches the robot is the sim's
        resulting position, streamed as an absolute `move_to` by
        `_stream_joints_to_robot()`.
        """
        with self._lock:
            self._sim_commanded_at[channel] = time.time()

    def _queue(self, channel: str, kind: str, value) -> None:
        """Hold a base command until the next flush.

        Relative motions accumulate -- a teleop emitting `move_by` at 30Hz into a
        twin flushing at 30Hz must not drop the deltas it lands between flushes.
        Everything else is a setpoint, where the newest command is the only one
        that matters.
        """
        with self._lock:
            previous = self._pending.get(channel)
            if previous is not None and previous[0] == kind == "move_by":
                value = _add(previous[1], value)
            self._pending[channel] = (kind, value)
            self._sim_commanded_at[channel] = time.time()

    def _joint_motion_limits(self, joint: MirroredJoint) -> tuple[float | None, float | None]:
        """The velocity and acceleration to send `joint`'s `move_to` with.

        Left at the robot's defaults, a joint streamed at `rate_hz` barely moves:
        each command re-plans the trapezoidal profile from where the joint is
        now, so it spends every cycle in the opening of a slow ramp and never
        reaches speed. The puppet teleop passes the joint's `max` profile for
        exactly this reason, and these are its numbers.
        """
        if joint.subsystem == "end_of_arm":
            return EOA_VELOCITY_R, EOA_ACCELERATION_R

        params = getattr(getattr(self.robot, joint.subsystem, None), "params", None)
        if not isinstance(params, dict):
            return None, None
        motion = params.get("motion", {}).get(MOTION_PROFILE, {})
        velocity, acceleration = motion.get("vel_m"), motion.get("accel_m")
        if joint.name == "lift" and acceleration is not None:
            # The puppet teleop backs the lift off its full acceleration.
            acceleration *= LIFT_ACCELERATION_SCALE
        return velocity, acceleration

    def _stream_joints_to_robot(self, sim_status) -> bool:
        """Send the sim's pose for every joint the sim is currently leading.

        Absolute positions, not the sim's own relative commands: a `move_by`
        stream at `rate_hz` restarts the robot's motion profile every cycle and
        the joint crawls. Streaming the position the sim has *reached* keeps a
        target out ahead of the robot, which is how `stretch_puppet_teleop.py`
        makes a follower track.
        """
        now = time.time()
        sent = False

        for joint in self.joints:
            if not self._sim_is_leading(joint.name, now):
                continue
            target = joint.actuator.get_position(sim_status)
            velocity, acceleration = self._motion_limits[joint.name]
            if joint.subsystem == "end_of_arm":
                accepted = self.robot.end_of_arm.move_to(
                    joint.name, target, velocity, acceleration
                )
            else:
                accepted = getattr(self.robot, joint.subsystem).move_to(
                    target, v_m=velocity, a_m=acceleration
                )
            self._targets[joint.name] = target
            sent = self._check_accepted(joint.name, accepted) or sent

        if self.gripper is not None and self._sim_is_leading(GRIPPER, now):
            target = self.gripper.to_robot(Actuators.gripper.get_position(sim_status))
            accepted = self.robot.end_of_arm.move_to(
                self.gripper.joint, target, EOA_VELOCITY_R, EOA_ACCELERATION_R
            )
            self._targets[GRIPPER] = target
            sent = self._check_accepted(self.gripper.joint, accepted) or sent
            sent = True

        return sent

    def _warn_once(self, key: str, message: str) -> None:
        """Say something loud the first time, then stay quiet about it.

        The loop runs at `rate_hz`, so anything that reports every tick buries
        the terminal and the run becomes unreadable.
        """
        if key not in self._refused:
            self._refused.add(key)
            click.secho(message, fg="red")

    def _check_accepted(self, name: str, accepted) -> bool:
        """Whether the robot queued the command, complaining once if it did not.

        `RobotClient`'s movement calls *return* False when they refuse one --
        most often because the joint is not homed -- and log to the robot's
        logger rather than to this terminal. Dropping the return value is how a
        twin ends up looking connected while nothing moves.
        """
        if accepted is False:
            self._warn_once(
                name,
                f"The robot refused a move_to on {name}. It is usually not homed, "
                "or the tool does not have that joint.",
            )
            return False
        return True

    def _sim_is_leading(self, channel: str, now: float) -> bool:
        """Whether the sim has commanded `channel` recently enough to own it.

        Streaming continues for `follow_s` past the last sim command so the robot
        can finish converging on the pose the sim stopped at. After that the
        channel is released, which is what lets the robot be backdriven in
        bidirectional mode instead of being held at the sim's last setpoint.
        """
        commanded_at = self._sim_commanded_at.get(channel)
        return commanded_at is not None and now - commanded_at < self.follow_s

    def _flush_base_to_robot(self) -> bool:
        """Send the base commands the sim issued since the last tick.

        The base is the one channel still forwarded command-for-command: its
        motions are one-shot `translate_by` / `rotate_by` or a velocity that is
        already a continuous setpoint, so neither suffers the profile restart
        that makes joint deltas stall.
        """
        with self._lock:
            pending, self._pending = self._pending, {}
        if not pending:
            return False

        # `translate_by` and `rotate_by` are queued under the same key on the
        # robot, so only the last one of a tick would survive. Send the
        # translation now and keep the rotation for the next tick.
        deferred = {}
        if BASE_TRANSLATE in pending and BASE_ROTATE in pending:
            deferred[BASE_ROTATE] = pending.pop(BASE_ROTATE)

        for channel, (_, value) in sorted(pending.items()):
            if channel == BASE_TRANSLATE:
                self.robot.omnibase.translate_by(value[0], value[1])
            elif channel == BASE_ROTATE:
                self.robot.omnibase.rotate_by(value)
            elif channel == BASE_VELOCITY:
                self.robot.omnibase.set_velocity(*value)

        if deferred:
            with self._lock:
                for channel, (kind, value) in deferred.items():
                    previous = self._pending.get(channel)
                    if previous is not None and previous[0] == kind == "move_by":
                        value = _add(previous[1], value)
                    self._pending[channel] = (kind, value)

        return True

    # -- robot -> sim ------------------------------------------------------

    def _apply_robot_status(self, sim_status) -> None:
        now = time.time()

        for joint in self.joints:
            robot_value = self._robot_position(joint)
            if robot_value is None:
                continue
            target = robot_value
            current = joint.actuator.get_position(sim_status)
            if self._sim_leads(joint.name, current, target, joint.deadband, now):
                continue
            if abs(target - current) <= joint.deadband:
                continue
            self._sim_move_to(joint.actuator, target)

        if self.gripper is not None:
            robot_value = self.gripper.robot_position()
            if robot_value is not None:
                target = self.gripper.to_sim(robot_value)
                current = Actuators.gripper.get_position(sim_status)
                deadband = 0.02
                if not self._sim_leads(GRIPPER, current, target, deadband, now) and (
                    abs(target - current) > deadband
                ):
                    self._sim_move_to(Actuators.gripper, target)

        if self.mirror_base:
            self._apply_robot_base(now)

    def _apply_robot_base(self, now: float) -> None:
        """Drive the sim base at the robot's measured velocity.

        Both sides report body-frame velocities (forward, left, counter-clockwise),
        so the robot's odometry velocity is a sim base command as-is. The sim base
        has no pose reset, so this tracks motion rather than pose: the two odometry
        frames stay close while driving, but do not re-converge after a slip.
        """
        for channel in (BASE_TRANSLATE, BASE_ROTATE, BASE_VELOCITY):
            commanded_at = self._sim_commanded_at.get(channel)
            if commanded_at is not None and now - commanded_at < self.hold_s:
                return

        status = self.robot.omnibase.status
        v_x = status.get("x_vel", 0.0)
        v_y = status.get("y_vel", 0.0)
        omega = status.get("theta_vel", 0.0)

        is_moving = (
            max(abs(v_x), abs(v_y)) > self.base_linear_deadband
            or abs(omega) > self.base_angular_deadband
        )
        if is_moving:
            self._sim_set_base_velocity(v_x, v_y, omega)
        elif self._base_was_moving:
            # One last zero, so the sim base decelerates instead of coasting on
            # the last setpoint.
            self._sim_set_base_velocity(0.0, 0.0, 0.0)
        self._base_was_moving = is_moving

    def _sim_leads(
        self, channel: str, sim_position: float, robot_position: float, deadband: float, now: float
    ) -> bool:
        """Whether a sim command for `channel` is still in flight on the robot.

        While it is, the robot's status is stale by construction -- it is where
        the robot is on its way from -- and following it would drag the sim back.
        The latch releases as soon as the robot arrives, and gives up after
        `settle_s` if it never does, so a blocked or run-stopped robot ends up
        reflected in the sim rather than silently ignored.
        """
        if self._sim_is_leading(channel, now):
            return True
        commanded_at = self._sim_commanded_at.get(channel)
        if commanded_at is None:
            return False
        # Streaming has stopped, but give the robot a tail to arrive in.
        age = now - commanded_at
        return age < self.settle_s and abs(robot_position - sim_position) > deadband * 5

    def _robot_position(self, joint: MirroredJoint) -> float | None:
        """`joint`'s position from the robot's status, in robot units."""
        try:
            if joint.subsystem == "end_of_arm":
                return self.robot.status["end_of_arm"][joint.name]["pos"]
            return getattr(self.robot, joint.subsystem).status["pos"]
        except (KeyError, AttributeError, TypeError):
            return None

    # -- lifecycle ---------------------------------------------------------

    def sync_sim_to_robot(self) -> None:
        """Snap the sim onto the robot's current pose, before mirroring starts.

        Always runs in the robot's direction: moving the sim is free, and it
        means the first mirrored command is not a jump across the difference
        between the two poses.
        """
        self.robot.pull_status(blocking=True)
        for joint in self.joints:
            robot_value = self._robot_position(joint)
            if robot_value is not None:
                self._sim_move_to(joint.actuator, robot_value)
        if self.gripper is not None:
            robot_value = self.gripper.robot_position()
            if robot_value is not None:
                self._sim_move_to(Actuators.gripper, self.gripper.to_sim(robot_value))
        self.sim.wait_command(timeout=10.0)

    def step(self) -> None:
        """Run one control cycle of the bridge."""
        # Non-blocking, as in the puppet teleop: waiting on status here would
        # push the robot's commands out of the cycle they were queued in.
        self.robot.pull_status(blocking=False)
        sim_status = self.sim.pull_status()

        if self.mirror_sim_to_robot:
            sent = self._flush_base_to_robot()
            # Guarded so that a joint the robot chokes on cannot take the base
            # down with it: both ride out on the one `push_command()` below, so
            # without this an exception here strands an already-queued base
            # command and the base goes dead for a reason that is not its own.
            try:
                sent = self._stream_joints_to_robot(sim_status) or sent
            except Exception as exception:
                self._warn_once(
                    "stream", f"Could not stream joints to the robot: {exception!r}"
                )
            if sent:
                self.robot.push_command()

        if self.mirror_robot_to_sim:
            self._apply_robot_status(sim_status)

        if self.debug:
            self._report(sim_status)

    def _report(self, sim_status) -> None:
        """Print where each joint stands, at 2Hz, to show which link is broken.

        Reads left to right the way the mirror runs: what the sim is doing, then
        whether the sim owns the joint, then what the robot was told, then where
        the robot actually is.
        """
        now = time.time()
        if now - self._last_debug < 0.5:
            return
        self._last_debug = now

        rows = []
        for joint in self.joints:
            robot_value = self._robot_position(joint)
            rows.append(
                (
                    joint.name,
                    f"{joint.actuator.get_position(sim_status):+.3f}",
                    "sim" if self._sim_is_leading(joint.name, now) else "-",
                    f"{self._targets[joint.name]:+.3f}" if joint.name in self._targets else "-",
                    "-" if robot_value is None else f"{robot_value:+.3f}",
                )
            )
        if self.gripper is not None:
            robot_value = self.gripper.robot_position()
            rows.append(
                (
                    GRIPPER,
                    f"{Actuators.gripper.get_position(sim_status):+.3f}",
                    "sim" if self._sim_is_leading(GRIPPER, now) else "-",
                    f"{self._targets[GRIPPER]:+.3f}" if GRIPPER in self._targets else "-",
                    "-" if robot_value is None else f"{robot_value:+.3f}",
                )
            )

        click.echo(f"{'joint':<12}{'sim':>8}{'leads':>7}{'sent':>9}{'robot':>9}")
        for row in rows:
            click.echo(f"{row[0]:<12}{row[1]:>8}{row[2]:>7}{row[3]:>9}{row[4]:>9}")

    def stop_robot_motion(self) -> None:
        """Leave the robot stationary, whatever the sim was last doing."""
        if not self.mirror_sim_to_robot:
            return
        with self._lock:
            self._pending.clear()
            # Ends the position stream, so nothing is sent after this.
            self._sim_commanded_at.clear()
        try:
            if self.mirror_base:
                self.robot.omnibase.set_velocity(0.0, 0.0, 0.0)
                self.robot.push_command()
        except Exception as exception:
            click.secho(f"Could not stop the robot's base: {exception}", fg="red")


# =============================================================================
# Drawing robots in Rerun
# =============================================================================

GHOST_ALPHA = 0.35
"""How opaque an overlaid robot is drawn, where 1.0 is the model's own colours.

Two robots standing inside one another are only readable if the one in front can
be seen through. At full opacity the overlay covers exactly the part of the other
robot the picture exists to compare it with, and the render is of one robot with
another known to be somewhere behind it.
"""

OVERLAY_ALPHA = 0.6
"""The same, for a robot that is *coincident* with the one underneath rather than near it.

A digital twin that is working draws its two robots in the same place to within a
millimetre, which is the one case the ghost alpha above is wrong for: two
surfaces at the same depth, and the translucent one reads as a faint sheen on
the solid one rather than as a robot. Solid enough to be the thing you are
looking at, sheer enough to see the other through -- and the moment the two do
separate, which is what the view is for, it is unmistakable.
"""

REAL_ROBOT_COLOR = (0.62, 0.42, 0.98)
"""What the real robot's URDF is drawn in, against the simulated robot's own colours.

Translucency alone is not enough to tell two Stretches apart: they are the same
grey plastic and white metal in the same pose, and at a camera distance that
frames the gripper there is not much of either in shot. So the real robot is
painted one colour outright, and its tool ball is painted the same -- which says
which robot that ball belongs to without a legend. The simulated one keeps its
real colours, being the one whose model is under question.
"""

SIM_TOOL_COLOR = (1.00, 0.35, 0.10)
"""The simulated robot's tool centre."""


def visual_geoms(model, root_body: str) -> list[int]:
    root = model.body(root_body).id
    inside = [False] * model.nbody
    # MuJoCo orders bodies parents-first, so one pass settles every descendant.
    for body in range(model.nbody):
        inside[body] = body == root or (body > 0 and inside[model.body_parentid[body]])

    solid = [
        geom
        for geom in range(model.ngeom)
        if inside[model.geom_bodyid[geom]]
        and int(model.geom_type[geom])
        not in (mujoco.mjtGeom.mjGEOM_PLANE, mujoco.mjtGeom.mjGEOM_HFIELD)
    ]
    visual = [
        geom for geom in solid if not (model.geom_contype[geom] or model.geom_conaffinity[geom])
    ]
    return visual or solid


def geom_color(model, geom: int) -> np.ndarray:
    """A geom's RGBA, taking the material's where it has one.

    A geom with a material carries `rgba` values the renderer ignores, so reading
    `geom_rgba` alone paints most of an arm the MJCF's default off-white.
    """
    material = int(model.geom_matid[geom])
    rgba = model.mat_rgba[material] if material >= 0 else model.geom_rgba[geom]
    return np.asarray(rgba, dtype=float)


def rgba8(color) -> list[int]:
    """A float RGB or RGBA sequence as the 0-255 integers Rerun wants."""
    return [int(round(float(channel) * 255)) for channel in color]


class RerunMujocoRobot:

    def __init__(
        self,
        rr,
        model,
        root_body: str,
        path: str,
        tint=None,
        alpha: float = GHOST_ALPHA,
    ) -> None:
        """`tint` repaints every geom one colour, at `alpha`. None keeps the model's own."""
        self._rr = rr
        self.model = model
        self.path = path
        self.geoms = visual_geoms(model, root_body)
        self._paths: dict[int, str] = {}
        for geom in self.geoms:
            name = model.geom(geom).name or f"geom_{geom}"
            self._paths[geom] = f"{path}/{name.replace('/', '_')}"
            color = geom_color(model, geom) if tint is None else np.array([*tint[:3], alpha], float)
            self._log_shape(self._paths[geom], geom, rgba8(color))

    def _log_shape(self, path: str, geom: int, color: list[int]) -> None:
        """One geom's shape, in its own frame, logged `static`.

        MuJoCo's primitives are all centred on the geom frame and aligned with
        its z -- except Rerun's capsule, which grows from the origin along +z,
        hence the half-length shifted back. Anything else is skipped rather than
        approximated: a shape drawn as the wrong shape is worse in a comparison
        view than a shape that is not drawn.
        """
        rr, model = self._rr, self.model
        kind = mujoco.mjtGeom(int(model.geom_type[geom]))
        size = np.asarray(model.geom_size[geom], dtype=float)
        if kind == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = int(model.geom_dataid[geom])
            vertex = int(model.mesh_vertadr[mesh])
            vertices = int(model.mesh_vertnum[mesh])
            face = int(model.mesh_faceadr[mesh])
            faces = int(model.mesh_facenum[mesh])
            rr.log(
                path,
                rr.Mesh3D(
                    vertex_positions=model.mesh_vert[vertex : vertex + vertices],
                    triangle_indices=model.mesh_face[face : face + faces],
                    albedo_factor=color,
                ),
                static=True,
            )
        elif kind == mujoco.mjtGeom.mjGEOM_BOX:
            rr.log(
                path, rr.Boxes3D(half_sizes=[size], colors=[color], fill_mode="solid"), static=True
            )
        elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            rr.log(
                path,
                rr.Ellipsoids3D(half_sizes=[[size[0]] * 3], colors=[color], fill_mode="solid"),
                static=True,
            )
        elif kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            rr.log(
                path,
                rr.Ellipsoids3D(half_sizes=[size], colors=[color], fill_mode="solid"),
                static=True,
            )
        elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
            rr.log(
                path,
                rr.Cylinders3D(
                    lengths=[2.0 * size[1]], radii=[size[0]], colors=[color], fill_mode="solid"
                ),
                static=True,
            )
        elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            rr.log(
                path,
                rr.Capsules3D(
                    lengths=[2.0 * size[1]],
                    radii=[size[0]],
                    translations=[[0.0, 0.0, -size[1]]],
                    colors=[color],
                    fill_mode="solid",
                ),
                static=True,
            )

    def log(self, data) -> None:
        """Where every drawn geom is now, from an `MjData` already forwarded."""
        rr = self._rr
        for geom, path in self._paths.items():
            rr.log(
                path,
                rr.Transform3D(
                    translation=data.geom_xpos[geom],
                    mat3x3=data.geom_xmat[geom].reshape(3, 3),
                ),
            )

    def body_pose(self, data, body: str) -> np.ndarray:
        """One body's 4x4 pose in the model's world frame."""
        index = self.model.body(body).id
        pose = np.eye(4)
        pose[:3, :3] = data.xmat[index].reshape(3, 3)
        pose[:3, 3] = data.xpos[index]
        return pose


class RerunUrdfRobot:
    """One `yourdfpy` URDF, drawn into a Rerun entity tree.

    `stretch4_body/tools/stretch_joint_viz.py` in reusable form, and the same
    static-geometry / per-tick-transform split as `RerunMujocoRobot`. What it
    adds over drawing the MJCF twice is that this is the robot's *own*
    description -- its batch, its tool, its meshes -- so a steady offset between
    this and a MuJoCo model of the same robot is a real difference between the
    model and the machine rather than a difference in joint angles.

    Link poses are reported relative to `base_link` rather than the URDF's own
    root, so that a caller can hang this off whatever it is comparing against and
    have the two pinned together at the base by construction. Every millimetre
    between the hands is then joint angles, and not an argument about odometry.
    """

    def __init__(
        self,
        rr,
        urdf,
        path: str,
        tool_link: str = "grasp_center_link",
        base_link: str = "base_link",
        color=REAL_ROBOT_COLOR,
        alpha: float = OVERLAY_ALPHA,
    ) -> None:
        self._rr = rr
        self.urdf = urdf
        self.path = path
        self.tool_link = tool_link
        self.base_link = base_link
        self.links: list[str] = []
        self.vertices = 0
        """How much geometry actually reached the viewer. See `_log_meshes`."""
        self._log_meshes(rgba8((*color[:3], alpha)))

    def _log_meshes(self, color: list[int]) -> None:
        """Every visual mesh, once, in its link's frame. Fills `links` and `vertices`.

        Logged as `Mesh3D` from vertices `yourdfpy` has already loaded, rather
        than as `Asset3D` pointing at the STL on disk. Three reasons, and the
        first is the one that matters: this is the same archetype the MuJoCo half
        of the view uses, so the two robots cannot end up rendering differently
        for reasons to do with file formats. The mesh bytes are also already in
        memory -- `URDF.load` parses them to build its scene -- so re-reading
        them through a second loader buys nothing. And a vertex count is
        something this class can report, where "the viewer was handed a path"
        is not: a silently empty robot is exactly the failure that is hard to
        see, because what it looks like is a view with one robot in it.

        Links with no mesh are dropped rather than posed as empty entities: a
        Stretch URDF carries a dozen millimetre placeholder boxes for camera
        optical frames, and a view with those in it is a view with a dozen
        unlabelled specks floating in it.
        """
        import trimesh

        rr = self._rr
        for name, link in self.urdf.link_map.items():
            meshes = [
                visual
                for visual in link.visuals
                if visual.geometry is not None
                and visual.geometry.mesh is not None
                and visual.geometry.mesh.filename
            ]
            if not meshes or name not in self.urdf.scene.graph.nodes:
                continue
            self.links.append(name)
            for index, visual in enumerate(meshes):
                path = f"{self.path}/{name}/mesh_{index}"
                if visual.origin is not None:
                    origin = np.asarray(visual.origin, dtype=float)
                    rr.log(
                        path,
                        rr.Transform3D(translation=origin[:3, 3], mat3x3=origin[:3, :3]),
                        static=True,
                    )
                try:
                    # `force="mesh"` because an STL with several solids in it
                    # loads as a `Scene`, which has no `.vertices` -- and a link
                    # that came back as one would otherwise raise here and be
                    # dropped from a robot that is only missing a part.
                    mesh = trimesh.load(visual.geometry.mesh.filename, force="mesh")
                    vertices = np.asarray(mesh.vertices, dtype=np.float32)
                    if visual.geometry.mesh.scale is not None:
                        vertices = vertices * np.asarray(
                            visual.geometry.mesh.scale, dtype=np.float32
                        ).reshape(-1)
                    rr.log(
                        path,
                        rr.Mesh3D(
                            vertex_positions=vertices,
                            triangle_indices=np.asarray(mesh.faces, dtype=np.uint32),
                            vertex_normals=np.asarray(mesh.vertex_normals, dtype=np.float32),
                            albedo_factor=color,
                        ),
                        static=True,
                    )
                    self.vertices += len(vertices)
                except Exception as error:  # noqa: BLE001 - one missing mesh is not a run
                    click.secho(
                        f"Could not load {visual.geometry.mesh.filename}: {error}", fg="yellow"
                    )

    def pose_from_status(self, status: dict) -> None:
        """Move onto the pose `RobotClient.status` describes.

        The status-to-URDF conversion needs one thing only this object has: the
        travel of the PG4's slide joint, which is a property of the description
        being drawn rather than a constant. See `_parallel_gripper_fingers`.
        """
        self.pose(stretch_urdf_configuration(status, finger_limits=self.finger_limits))

    @property
    def finger_limits(self) -> tuple[float, float] | None:
        """The PG4 slide joint's travel, or None on a URDF whose hand is not one."""
        joint = self.urdf.joint_map.get("finger_left_joint")
        if joint is None or joint.limit is None:
            return None
        return float(joint.limit.lower), float(joint.limit.upper)

    def pose(self, configuration: dict[str, float]) -> None:
        """Move the URDF's kinematics onto `configuration`, tolerating a bad name."""
        try:
            self.urdf.update_cfg(configuration)
        except Exception as error:  # noqa: BLE001 - a bad joint name is not a run
            click.secho(f"urdf.update_cfg refused {configuration}: {error}", fg="yellow")

    def log(self, base_pose) -> np.ndarray | None:
        """Draw the robot under `base_pose`, and return its tool pose in the world.

        Re-logged every tick even when the joints have not moved, because the
        parent may have: `base_pose` is where this robot's `base_link` is, and it
        moves whenever the robot it is being compared against drives.
        """
        rr = self._rr
        base_pose = np.asarray(base_pose, dtype=float)
        rr.log(self.path, rr.Transform3D(translation=base_pose[:3, 3], mat3x3=base_pose[:3, :3]))
        for link in self.links:
            pose = self.relative(link)
            if pose is not None:
                rr.log(
                    f"{self.path}/{link}",
                    rr.Transform3D(translation=pose[:3, 3], mat3x3=pose[:3, :3]),
                )
        tool = self.relative(self.tool_link)
        return None if tool is None else base_pose @ tool

    def relative(self, link: str) -> np.ndarray | None:
        """One link's pose in `base_link`, or None if it is not in the scene graph."""
        try:
            matrix, _ = self.urdf.scene.graph.get(link, self.base_link)
        except Exception:  # noqa: BLE001 - a link off the graph is one not drawn
            return None
        return np.asarray(matrix, dtype=float)


TOOL_FOR_GRIPPER_JOINT = {
    "stretch_gripper": "eoa_wrist_dw4_tool_sg4",
    "parallel_gripper": "eoa_wrist_dw4_tool_pg4",
}


def robot_gripper_joint(robot) -> str | None:
    """Which gripper the robot's *server* says it has, or None if it has not said."""
    status = getattr(robot, "status", None) or {}
    end_of_arm = status.get("end_of_arm") or {}
    for joint in TOOL_FOR_GRIPPER_JOINT:
        if joint in end_of_arm:
            return joint
    return None


def load_stretch_urdf(robot=None) -> tuple[object, tuple[str, str, str]]:
    """This robot's own URDF, by the model, batch and tool its fleet directory names."""
    import io

    import stretch4_urdf
    import yourdfpy
    from stretch4_body.core.robot_params import RobotParams

    _, params = RobotParams.get_params()
    model, batch, tool = (
        params["robot"]["model_name"],
        params["robot"]["batch_name"],
        params["robot"]["tool"],
    )
    tool = _tool_matching_robot(tool, robot)
    default_batch = Stretch4MujocoSimulator.get_default_model_batch_tool_names()[1]
    try:
        contents = stretch4_urdf.get_urdf(
            model, batch, tool, do_add_file_prefix_to_absolute_paths=False
        )
    except FileNotFoundError:
        if batch == default_batch:
            raise
        click.secho(
            f"               {model}/{batch} has no URDF description shipped, so the "
            f"overlay is drawn from {model}/{default_batch} instead. Point "
            "HELLO_FLEET_PATH at the robot's own fleet directory to draw its batch.",
            fg="yellow",
        )
        batch = default_batch
        contents = stretch4_urdf.get_urdf(
            model, batch, tool, do_add_file_prefix_to_absolute_paths=False
        )
    return yourdfpy.URDF.load(io.StringIO(contents)), (model, batch, tool)


def _tool_matching_robot(tool: str, robot) -> str:
    reported = robot_gripper_joint(robot)
    if reported is None:
        return tool
    wanted = TOOL_FOR_GRIPPER_JOINT[reported]
    if tool.endswith(wanted.rsplit("_", 1)[-1]):
        return tool
    click.secho(
        f"               the fleet directory says the tool is {tool}, but the robot is "
        f"reporting a {reported}. Drawing {wanted} to match the robot -- check "
        "HELLO_FLEET_PATH, or whether this robot's tool has been configured.",
        fg="yellow",
    )
    return wanted


def stretch_urdf_configuration(
    status: dict,
    gripper_joint: str | None = None,
    finger_limits: tuple[float, float] | None = None,
) -> dict[str, float]:
    configuration = {
        f"arm_l{segment}_joint": float(status["arm"]["pos"]) / 4.0 for segment in (1, 2, 3, 4)
    }
    configuration["lift_joint"] = float(status["lift"]["pos"])
    end_of_arm = status.get("end_of_arm", {})
    for joint in ("wrist_yaw", "wrist_pitch", "wrist_roll"):
        if joint in end_of_arm:
            configuration[f"{joint}_joint"] = float(end_of_arm[joint].get("pos", 0.0))

    if gripper_joint is None:
        gripper_joint = next((j for j in TOOL_FOR_GRIPPER_JOINT if j in end_of_arm), None)
    hand = end_of_arm.get(gripper_joint) or {}
    if gripper_joint == "parallel_gripper":
        configuration.update(_parallel_gripper_fingers(hand, finger_limits))
    else:
        # `gripper_conversion` is published beside the SG4's raw servo angle
        # precisely so that nothing downstream has to redo the chord-over-radius
        # that turns one into the other. `finger_rad` is the URDF's own units.
        conversion = hand.get("gripper_conversion") or {}
        if conversion.get("finger_rad") is not None:
            finger = float(conversion["finger_rad"])
            configuration["gripper_finger_left_joint"] = finger
            configuration["gripper_finger_right_joint"] = finger
    return configuration


def _parallel_gripper_fingers(hand: dict, limits: tuple[float, float] | None) -> dict[str, float]:
    pos_mm = hand.get("pos_mm")
    if pos_mm is None or limits is None:
        return {}
    from stretch4_body.core.robot_params import RobotParams

    _, params = RobotParams.get_params()
    range_mm = float(params.get("parallel_gripper", {}).get("range_mm", 80.0)) or 80.0
    lower, upper = limits
    value = upper + (float(pos_mm) / range_mm) * (lower - upper)
    return {"finger_left_joint": value, "finger_right_joint": value}


SIM_ROOT_BODY = "stretch4"
"""The body the simulated robot hangs off in `scene_stretch4.xml`."""

SIM_ARM_SEGMENTS = 4
"""How many telescoping segments carry the arm's extension in the MJCF.
"""


class DigitalTwinView:
    """The twin in 3D: the simulated robot, and the real one drawn inside it."""

    SIM = "world/sim"
    REAL = "world/real"
    BASE_LINK = "base_link"

    def __init__(
        self,
        sim: Stretch4MujocoSimulator,
        robot,
        joints: tuple[MirroredJoint, ...] = MIRRORED_JOINTS,
        gripper: "GripperMirror | None" = None,
        scene_xml_path: str | None = None,
        root_body: str = SIM_ROOT_BODY,
        spawn: bool = True,
        save_path: Path | None = None,
    ) -> None:
        """`joints` and `gripper` are the mirror's own, not this view's idea of them."""
        import rerun as rr
        import rerun.blueprint as rrb

        self._rr = rr
        self.sim = sim
        self.robot = robot
        self.joints = tuple(joints)
        self.gripper = gripper
        self.tick = 0

        # A second copy of the scene, compiled here. The simulator runs MuJoCo in
        # its own process and hands back a status dataclass, not an `MjData`, so
        # there is no live model on this side to read geom poses out of -- this
        # one is posed from that status and never stepped, exactly as
        # `run_on_real_stretch.build_mirror` does for the real robot.
        self.model = mujoco.MjModel.from_xml_path(
            scene_xml_path or sim.scene_xml_path or Stretch4MujocoSimulator.get_scene_xml_path()
        )
        self.data = mujoco.MjData(self.model)
        self.root_body = root_body
        self._joints = self._joint_addresses()

        rr.init("Stretch 4 digital twin", spawn=spawn and save_path is None)
        if save_path is not None:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            rr.save(str(save_path))
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

        self.sim_robot = RerunMujocoRobot(rr, self.model, root_body, self.SIM)
        self.real_robot = self._load_real_robot(rr)
        for path, color in (
            ("world/tool/sim", SIM_TOOL_COLOR),
            ("world/tool/real", REAL_ROBOT_COLOR),
        ):
            rr.log(f"{path}/frame", rr.TransformAxes3D(axis_length=0.08), static=True)
            rr.log(
                path,
                rr.Points3D(positions=[[0.0, 0.0, 0.0]], radii=[0.012], colors=[rgba8(color)]),
                static=True,
            )
        rr.send_blueprint(
            rrb.Blueprint(
                rrb.Horizontal(
                    rrb.Spatial3DView(
                        origin="world", name="sim (solid) and the real robot (violet)"
                    ),
                    rrb.Vertical(
                        rrb.TimeSeriesView(
                            origin="error", name="how far the robot is from the sim"
                        ),
                        self._joint_grid(rrb),
                        row_shares=[1, 2],
                    ),
                    column_shares=[3, 2],
                )
            )
        )

    def _joint_grid(self, rrb):
        """One plot per mirrored channel, tiled.

        A single plot over `joint/` puts every channel on one pair of axes, which
        is unreadable twice over: eleven lines in a space that has room for three,
        and metres and radians sharing a y axis, so the lift's half-metre of
        travel flattens a wrist's whole range into the baseline. Per channel,
        each plot holds exactly the two lines that belong together -- where the
        sim is and where the robot is -- on an axis scaled to that joint, which
        is the comparison the view is named for.
        """
        plots = [
            rrb.TimeSeriesView(origin=f"joint/{joint.name}", name=joint.name)
            for joint in self.joints
        ]
        if self.gripper is not None:
            plots.append(rrb.TimeSeriesView(origin=f"joint/{GRIPPER}", name=GRIPPER))
        return rrb.Grid(
            *plots,
            # Two, so a tile stays wide enough to read a time axis on in the
            # right-hand column of the window. Six channels is the usual case and
            # lands as three rows of two.
            grid_columns=min(2, len(plots)) or 1,
            name="sim against robot, per joint",
        )

    def _load_real_robot(self, rr):
        """The real robot's URDF drawing, or `None` and a reason why not.

        An absent overlay is an ordinary outcome rather than a fault: a
        workstation without `yourdfpy`, or a stand-in fleet directory naming
        batch `nominal`, which ships no meshes. Either way the sim robot is still
        worth watching, so this reports and returns.
        """
        try:
            urdf, description = load_stretch_urdf(self.robot)
        except Exception as error:  # noqa: BLE001 - one robot in the view is still a view
            click.secho(f"  rerun      : no URDF overlay ({error}).", fg="yellow")
            return None
        drawing = RerunUrdfRobot(rr, urdf, self.REAL, base_link=self.BASE_LINK)
        click.echo(
            f"  rerun      : real robot {'/'.join(description)}, "
            f"{len(drawing.links)} links, {drawing.vertices} vertices"
        )
        if not drawing.vertices:
            # A URDF that parsed and drew nothing is the failure that looks like
            # success: the view comes up, one robot is in it, and nothing says
            # the other one is missing rather than hidden behind it.
            click.secho(
                "               none of its meshes loaded, so the overlay will be empty.",
                fg="yellow",
            )
        return drawing

    def _joint_addresses(self) -> dict[str, int]:
        """`qpos` address for every model joint this view writes, by name.

        Looked up once. A joint the model does not have is simply absent, and
        `_pose_sim` skips it -- a scene built with the parallel gripper has no
        `gripper_finger_*_joint`, and that is a hand this cannot draw open rather
        than a run it should end.
        """
        wanted = [
            "lift_joint",
            *(f"arm_l{segment}_joint" for segment in range(1, SIM_ARM_SEGMENTS + 1)),
            "wrist_yaw_joint",
            "wrist_pitch_joint",
            "wrist_roll_joint",
            "gripper_finger_left_joint",
            "gripper_finger_right_joint",
        ]
        addresses = {}
        for name in wanted:
            try:
                addresses[name] = int(self.model.joint(name).qposadr[0])
            except KeyError:
                continue
        return addresses

    # -- one tick -----------------------------------------------------------

    def _pose_sim(self, status) -> None:
        """Write the simulator's reported joints into the local model and forward it.

        `mj_forward` rather than `mj_step`: this model exists to be photographed
        at the configuration the simulator is *in*, not to settle at one of its
        own. The base goes in through the free joint rather than through an
        entity transform so that the model's own `base_link` -- which is what the
        real robot is hung off -- is in the same world as everything else here.
        """
        values = {
            "lift_joint": status.lift.pos,
            "wrist_yaw_joint": status.wrist_yaw.pos,
            "wrist_pitch_joint": status.wrist_pitch.pos,
            "wrist_roll_joint": status.wrist_roll.pos,
            "gripper_finger_left_joint": status.gripper_left_finger.pos,
            "gripper_finger_right_joint": status.gripper_right_finger.pos,
        }
        for segment in range(1, SIM_ARM_SEGMENTS + 1):
            values[f"arm_l{segment}_joint"] = status.arm.pos / SIM_ARM_SEGMENTS
        for name, value in values.items():
            if name in self._joints:
                self.data.qpos[self._joints[name]] = float(value)

        free = self.model.body(self.root_body).jntadr[0]
        if free >= 0 and self.model.jnt_type[free] == mujoco.mjtJoint.mjJNT_FREE:
            address = int(self.model.jnt_qposadr[free])
            half = status.base.theta / 2.0
            self.data.qpos[address : address + 2] = (status.base.x, status.base.y)
            self.data.qpos[address + 2] = self.model.qpos0[address + 2]
            self.data.qpos[address + 3 : address + 7] = (math.cos(half), 0.0, 0.0, math.sin(half))
        mujoco.mj_forward(self.model, self.data)

    def log(self, sim_status) -> None:
        """One tick of the view: both robots, both tool centres, and the errors.

        The channels are the mirror's own `MirroredJoint` entries, taken at
        construction, so the plots show exactly the joints the mirror keeps in
        step.
        """
        rr = self._rr
        self.tick += 1
        rr.set_time("tick", sequence=self.tick)
        rr.set_time("wall_time", timestamp=time.time())

        self._pose_sim(sim_status)
        self.sim_robot.log(self.data)
        sim_tool = self.sim_robot.body_pose(self.data, "grasp_center_link")
        self._log_tool("world/tool/sim", sim_tool)

        status = getattr(self.robot, "status", None)
        real_tool = None
        if self.real_robot is not None and status:
            self.real_robot.pose_from_status(status)
            real_tool = self.real_robot.log(self.sim_robot.body_pose(self.data, self.BASE_LINK))
            self._log_tool("world/tool/real", real_tool)

        if real_tool is not None:
            gap = float(np.linalg.norm(real_tool[:3, 3] - sim_tool[:3, 3]))
            rr.log("error/tool_gap_m", rr.Scalars(gap))
            rr.log(
                "world/tool/gap",
                rr.LineStrips3D(
                    [[sim_tool[:3, 3], real_tool[:3, 3]]],
                    radii=[0.003],
                    colors=[[255, 255, 255, 255]],
                    labels=[f"{gap * 1000:.0f} mm"],
                ),
            )

        self._log_joints(sim_status, status)

    def _log_joints(self, sim_status, status) -> None:
        """Both sides of every mirrored joint, and the difference, as scalars.

        The `_report` table this example already prints at 2Hz, plotted instead:
        the same three numbers per joint, on a timeline, where a channel that
        stops tracking is a line that separates rather than a row that has to be
        noticed going past.
        """
        rr = self._rr
        for joint in self.joints:
            simulated = joint.actuator.get_position(sim_status)
            rr.log(f"joint/{joint.name}/sim", rr.Scalars(float(simulated)))
            measured = self._robot_position(joint, status)
            if measured is None:
                continue
            rr.log(f"joint/{joint.name}/robot", rr.Scalars(float(measured)))
            rr.log(f"error/{joint.name}", rr.Scalars(abs(float(measured - simulated))))

        if self.gripper is not None:
            simulated = Actuators.gripper.get_position(sim_status)
            rr.log(f"joint/{GRIPPER}/sim", rr.Scalars(float(simulated)))
            measured = self.gripper.robot_position()
            if measured is not None:
                measured = self.gripper.to_sim(measured)
                rr.log(f"joint/{GRIPPER}/robot", rr.Scalars(float(measured)))
                rr.log(f"error/{GRIPPER}", rr.Scalars(abs(float(measured - simulated))))

    @staticmethod
    def _robot_position(joint: MirroredJoint, status) -> float | None:
        """`joint`'s position out of `RobotClient.status`, in robot units.

        The same lookup `DigitalTwin._robot_position` makes, off the status dict
        rather than off the client, so this view can be given a status it has
        already pulled instead of pulling one of its own inside a control loop.
        """
        if not status:
            return None
        try:
            if joint.subsystem == "end_of_arm":
                return status["end_of_arm"][joint.name]["pos"]
            return status[joint.subsystem]["pos"]
        except (KeyError, TypeError):
            return None

    def _log_tool(self, path: str, pose) -> None:
        if pose is None:
            return
        pose = np.asarray(pose, dtype=float)
        self._rr.log(path, self._rr.Transform3D(translation=pose[:3, 3], mat3x3=pose[:3, :3]))


def _add(a, b):
    """Accumulate two relative motions, scalar or (dx, dy)."""
    if isinstance(a, tuple):
        return tuple(x + y for x, y in zip(a, b))
    return a + b


def _connect(robot_ip: str | None):
    """Start a `RobotClient`, local or over IP."""
    try:
        from stretch4_body.robot.robot_client import RobotClient
    except ImportError as exception:
        raise SystemExit(
            f"stretch4_body is required for the digital twin ({exception}).\n"
            'Install it with: uv pip install -e ".[digital-twin]"'
        )

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
    return robot


@click.command()
@click.option(
    "--robot_ip",
    type=str,
    default=None,
    help="IP address of the robot running Stretch Body Server (e.g. 192.168.1.10). "
    "Omit to connect to a robot running on this machine.",
)
@click.option(
    "--controller",
    type=click.Choice(["robot", "sim", "bidirectional"]),
    default="bidirectional",
    help="Which side leads. 'sim' forwards the sim's commands to the robot, 'robot' follows "
    "the robot's joint status in the sim, 'bidirectional' does both.",
)
@click.option(
    "--joints",
    type=str,
    default=",".join(JOINT_GROUPS),
    help=f"Comma-separated joint groups to mirror, from: {', '.join(JOINT_GROUPS)}.",
)
@click.option(
    "--teleop",
    type=click.Choice(["keyboard", "gamepad", "none"]),
    default="keyboard",
    help="How to drive the sim. Use 'none' to drive it from your own script instead.",
)
@click.option("--scene-xml-path", type=str, default=None, help="Path to the scene xml file")
@click.option(
    "--rerun/--no-rerun",
    default=True,
    show_default=True,
    help="Open a Rerun window showing the simulated robot with the real robot's own URDF "
    "drawn inside it, and the per-joint gap between them. Needs rerun-sdk, and yourdfpy "
    "for the overlay.",
)
@click.option(
    "--rrd",
    type=click.Path(path_type=Path),
    default=None,
    help="Write the Rerun stream to this .rrd file instead of opening a window. Implies "
    "--rerun.",
)
@click.option(
    "--headless/--viewer",
    default=True,
    show_default=True,
    help="Run the sim without MuJoCo's own viewer. On by default because --rerun is: the "
    "Rerun window shows the simulated robot and the real one together, which is what the "
    "MuJoCo viewer cannot do. --viewer brings it back alongside.",
)
@click.option("--rate_hz", type=float, default=30.0, help="Control rate of the bridge")
@click.option("--no_prompt", is_flag=True, help="Skip the confirmation before the robot may move")
@click.option(
    "--debug",
    is_flag=True,
    help="Print, at 2Hz, each joint's sim position, which side leads it, what was sent to the "
    "robot and where the robot is.",
)
def main(
    robot_ip: str | None,
    controller: str,
    joints: str,
    teleop: str,
    scene_xml_path: str | None,
    rerun: bool,
    rrd: Path | None,
    headless: bool,
    rate_hz: float,
    no_prompt: bool,
    debug: bool,
):
    groups = tuple(group.strip() for group in joints.split(",") if group.strip())
    unknown = [group for group in groups if group not in JOINT_GROUPS]
    if unknown:
        raise SystemExit(f"Unknown joint group(s) {unknown}. Choose from: {list(JOINT_GROUPS)}")

    # Printed up front because the two things that most often differ between one
    # terminal and another -- the flags recalled from that shell's history, and
    # which interpreter is on its PATH -- are both invisible otherwise, and both
    # change what this run will do.
    click.echo("Digital twin")
    click.echo(f"  robot      : {f'tcp://{robot_ip}' if robot_ip else 'local'}")
    click.echo(f"  controller : {controller}")
    click.echo(f"  joints     : {', '.join(groups)}")
    click.echo(f"  teleop     : {teleop}")
    click.echo(f"  rate       : {rate_hz} Hz")
    click.echo(f"  python     : {sys.executable}")

    robot = _connect(robot_ip)

    robot.pull_status(blocking=True)
    if not robot.is_homed():
        robot.stop()
        raise SystemExit("The robot is not fully homed. Home it first, then rerun.")

    sim = Stretch4MujocoSimulator(scene_xml_path=scene_xml_path)
    twin = None
    view = None
    teleop_controller = None
    try:
        sim.start(headless=headless)
        if not sim.is_running():
            # Otherwise the run carries on against a dead simulator, reporting a
            # twin that is synchronizing and mirroring nothing.
            raise SystemExit("The MuJoCo simulator did not start. Nothing to mirror.")

        twin = DigitalTwin(sim, robot, controller=controller, groups=groups, debug=debug)

        if rerun or rrd is not None:
            # After the twin, because the view is laid out around the channels
            # the twin is actually mirroring and reads the real hand through the
            # `GripperMirror` it built; and before the sync, so the view's first
            # tick is the two robots as they were when the run started rather
            # than after one of them has jumped.
            view = DigitalTwinView(
                sim,
                robot,
                joints=twin.joints,
                gripper=twin.gripper,
                scene_xml_path=scene_xml_path,
                save_path=rrd,
            )

        click.echo("Synchronizing the sim to the robot...")
        twin.sync_sim_to_robot()

        if twin.mirror_sim_to_robot and not no_prompt:
            click.secho(
                f"\nThe real robot will follow the sim ({', '.join(groups)})."
                + (" This includes driving the base." if twin.mirror_base else "")
                + "\nClear the area around the robot.",
                fg="yellow",
            )
            click.prompt("Hit enter to begin", default="", show_default=False)

        twin.install()

        if teleop == "keyboard":
            from stretch4_mujoco.sim_teleop import KeyboardTeleop, print_keyboard_help

            print_keyboard_help()
            teleop_controller = KeyboardTeleop(sim)
            teleop_controller.start()
        elif teleop == "gamepad":
            from stretch4_mujoco.sim_teleop import GamepadTeleop

            teleop_controller = GamepadTeleop(sim)
            teleop_controller.start()

        click.secho(f"Digital twin running ({controller}). Press Ctrl-C to exit.", fg="green")

        period = 1.0 / rate_hz
        while sim.is_running():
            start = time.perf_counter()
            twin.step()
            if view is not None:
                # Off the status both sides have just pulled in `twin.step()`,
                # rather than pulling again: a second `pull_status` here would
                # cost a round trip per tick and would draw a robot at a
                # different instant from the one the twin just acted on.
                view.log(sim.pull_status())
            elapsed = time.perf_counter() - start
            if elapsed < period:
                time.sleep(period - elapsed)
    except KeyboardInterrupt:
        pass
    finally:
        if teleop_controller is not None:
            teleop_controller.stop()
        if twin is not None:
            twin.uninstall()
            twin.stop_robot_motion()
        robot.stop()
        sim.stop()


if __name__ == "__main__":
    main()
