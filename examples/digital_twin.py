"""
Digital twin: keep a real Stretch 4 and the MuJoCo sim in sync.

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

Requires the `digital-twin` extra: `uv pip install -e ".[digital-twin]"`.
"""

from __future__ import annotations

import math
import sys
import threading
import time
from dataclasses import dataclass

import click

from stretch4_mujoco.config import robot_settings_se4
from stretch4_mujoco.enums.actuators import Actuators
from stretch4_mujoco.gamepad_joints import WRIST_ROLL_SIM_SIGN
from stretch4_mujoco.stretch4_mujoco_simulator import Stretch4MujocoSimulator

# Base channels are mirrored as relative motions / velocities rather than
# positions, so they get their own names in the pending-command table.
BASE_TRANSLATE = "base_translate"
BASE_ROTATE = "base_rotate"
BASE_VELOCITY = "base_velocity"
GRIPPER = "gripper"


@dataclass(frozen=True)
class MirroredJoint:
    """A joint the sim and the robot command in the same units, up to `sign`.

    `sign` accounts for axes the URDF mirrors relative to the robot's servo
    convention (see `WRIST_ROLL_SIM_SIGN`): `sim_value = sign * robot_value`.
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
    sign: float = 1.0


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
        sign=WRIST_ROLL_SIM_SIGN,
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
            target = joint.sign * joint.actuator.get_position(sim_status)
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
            target = joint.sign * robot_value
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
                self._sim_move_to(joint.actuator, joint.sign * robot_value)
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
    if not robot.startup():
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
@click.option("--headless", is_flag=True, help="Run the sim without a viewer")
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
    teleop_controller = None
    try:
        sim.start(headless=headless)
        if not sim.is_running():
            # Otherwise the run carries on against a dead simulator, reporting a
            # twin that is synchronizing and mirroring nothing.
            raise SystemExit("The MuJoCo simulator did not start. Nothing to mirror.")

        twin = DigitalTwin(sim, robot, controller=controller, groups=groups, debug=debug)

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
