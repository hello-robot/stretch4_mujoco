"""
Reusable keyboard/gamepad teleoperation for driving a Stretch in sim.

`KeyboardTeleop` and `GamepadTeleop` each own a background thread: once
started, they read the input device and drive `sim` on their own, so a
script only needs to start one, run its own loop (camera feeds, lidar
viz, ...), and stop it on the way out:

    teleop = KeyboardTeleop(sim)
    teleop.start()
    try:
        while sim.is_running():
            ...
    finally:
        teleop.stop()

See `examples/keyboard_teleop.py`, `examples/gamepad_teleop.py`, and
`examples/molmo_environment.py` (--keyboard/--gamepad) for scripts built on
top of these.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from enum import Enum

import click
from pynput import keyboard

from stretch4_mujoco.enums.actuators import Actuators
from stretch4_mujoco.gamepad_controller import (
    ButtonPressCounter,
    GamePadController,
    JointEffortTracker,
)
from stretch4_mujoco.stretch4_mujoco_simulator import Stretch4MujocoSimulator
from stretch4_mujoco.stretch_mujoco_simulator import StretchMujocoSimulator

KEYBOARD_HELP = """
       Keyboard Controls:
=====================================
W / A / S / D: Move BASE
Q / E: Rotate BASE
U / H / J / K: Move LIFT & ARM
O / P: Move WRIST YAW
C / V: Move WRIST PITCH
T / Y: Move WRIST ROLL
N / M: Open & Close GRIPPER
ctrl + (shift) + ): Enable keyboard input
ctrl + (shift) + (: Disable keyboard input
ctrl + (shift) + @: Increase base velocity
ctrl + (shift) + !: Decrease base velocity
L : Print status
. : Stop
====================================="""


def print_keyboard_help() -> None:
    click.secho(KEYBOARD_HELP, fg="yellow")


class _KeyboardBaseVelocity:
    """Tracks which of the W/A/S/D/Q/E direction keys are currently held."""

    is_forward: bool | None = None
    is_right: bool | None = None  # diff drive base ignores this direction
    is_clockwise: bool | None = None

    def __init__(self):
        self.forward_velocity = 0.2
        self.right_velocity = 0.2
        self.clockwise_velocity = 0.8

    def get_forward_velocity(self):
        if self.is_forward is None:
            return 0.0
        return self.forward_velocity if self.is_forward else -self.forward_velocity

    def get_right_velocity(self):
        if self.is_right is None:
            return 0.0
        return self.right_velocity if self.is_right else -self.right_velocity

    def get_clockwise_velocity(self):
        if self.is_clockwise is None:
            return 0.0
        return self.clockwise_velocity if self.is_clockwise else -self.clockwise_velocity

    def increase_base_velocity(self):
        self.forward_velocity += 0.1
        self.right_velocity += 0.1
        self.clockwise_velocity += 0.1
        print(f"Forward velocity: {self.forward_velocity}")
        print(f"Right velocity: {self.right_velocity}")
        print(f"Clockwise velocity: {self.clockwise_velocity}")

    def decrease_base_velocity(self):
        self.forward_velocity -= 0.1
        self.right_velocity -= 0.1
        self.clockwise_velocity -= 0.1
        print(f"Forward velocity: {self.forward_velocity}")
        print(f"Right velocity: {self.right_velocity}")
        print(f"Clockwise velocity: {self.clockwise_velocity}")

    def apply(self, sim: StretchMujocoSimulator):
        if isinstance(sim, Stretch4MujocoSimulator):
            sim.base.set_velocity(
                self.get_forward_velocity(),
                self.get_right_velocity(),
                self.get_clockwise_velocity(),
            )
        else:
            sim.base.set_velocity(self.get_forward_velocity(), 0.0, self.get_clockwise_velocity())


class KeyboardTeleop:
    """
    Drives `sim` from the keyboard, using pynput to listen globally (the
    MuJoCo viewer window does not need focus).

    Held movement keys are re-applied on a background thread at `rate_hz`
    until released, mirroring the pattern in `examples/keyboard_teleop.py`.
    """

    def __init__(self, sim: StretchMujocoSimulator, rate_hz: float = 30.0):
        self.sim = sim
        self._rate_hz = rate_hz
        self.base_velocity = _KeyboardBaseVelocity()

        # Allow multiple key-presses, references https://stackoverflow.com/a/74910695
        self.key_buffer: list = []

        self.is_keyboard_control_active = True  # when false, disable kbd commands
        self.ctrl_pressed = False

        self._listener: keyboard.Listener | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self):
        self._stop_event.clear()
        self._listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self._listener.start()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _loop(self):
        period = 1.0 / self._rate_hz
        while not self._stop_event.is_set():
            start = time.perf_counter()
            if self.is_keyboard_control_active:
                for key in list(self.key_buffer):
                    if isinstance(key, keyboard.KeyCode):
                        self.keyboard_control(key.char)
            elapsed = time.perf_counter() - start
            if elapsed < period:
                time.sleep(period - elapsed)

    def enable_keyboard(self):
        self.is_keyboard_control_active = True
        print("Keyboard control enabled")

    def disable_keyboard(self):
        self.is_keyboard_control_active = False
        print("Keyboard control disabled")

    def on_press(self, key):
        if key == keyboard.Key.ctrl:
            self.ctrl_pressed = True

        if key not in self.key_buffer and len(self.key_buffer) <= 3:
            self.key_buffer.append(key)

    def on_release(self, key):
        if key == keyboard.Key.ctrl:
            self.ctrl_pressed = False
        if key in self.key_buffer:
            self.key_buffer.remove(key)
        if isinstance(key, keyboard.KeyCode):
            self.keyboard_control_release(key.char)

    def keyboard_control(self, key: str | None):
        from pprint import pprint

        if self.ctrl_pressed:
            if key == ")":
                self.enable_keyboard()
            elif key == "(":
                self.disable_keyboard()
            elif key == "@":
                self.base_velocity.increase_base_velocity()
            elif key == "!":
                self.base_velocity.decrease_base_velocity()

        if key == "w":
            self.base_velocity.is_forward = True
        elif key == "s":
            self.base_velocity.is_forward = False
        elif key == "a":
            self.base_velocity.is_right = True
        elif key == "d":
            self.base_velocity.is_right = False
        elif key == "e":
            self.base_velocity.is_clockwise = False
        elif key == "q":
            self.base_velocity.is_clockwise = True

        self.base_velocity.apply(self.sim)

        if key == "u":
            self.sim.lift.set_velocity(0.10)
        elif key == "j":
            self.sim.lift.set_velocity(-0.10)
        elif key == "h":
            self.sim.arm.set_velocity(-0.08)
        elif key == "k":
            self.sim.arm.set_velocity(0.08)
        elif key == "o":
            self.sim.end_of_arm.wrist_yaw.set_velocity(-0.3)
        elif key == "p":
            self.sim.end_of_arm.wrist_yaw.set_velocity(0.3)
        elif key == "c":
            self.sim.end_of_arm.wrist_pitch.set_velocity(0.3)
        elif key == "v":
            self.sim.end_of_arm.wrist_pitch.set_velocity(-0.3)
        elif key == "t":
            self.sim.end_of_arm.wrist_roll.set_velocity(0.3)
        elif key == "y":
            self.sim.end_of_arm.wrist_roll.set_velocity(-0.3)
        elif key == "n":
            self.sim.end_of_arm.stretch_gripper.set_velocity(0.2)
        elif key == "m":
            self.sim.end_of_arm.stretch_gripper.set_velocity(-0.2)
        elif key == "l":
            pprint(self.sim.pull_status())
        elif key == ".":
            self.sim.stop()

    def keyboard_control_release(self, key: str | None):
        if key == "w":
            self.base_velocity.is_forward = None
        elif key == "s":
            self.base_velocity.is_forward = None
        elif key == "a":
            self.base_velocity.is_right = None
        elif key == "d":
            self.base_velocity.is_right = None
        elif key == "e":
            self.base_velocity.is_clockwise = None
        elif key == "q":
            self.base_velocity.is_clockwise = None

        self.base_velocity.apply(self.sim)

        if key in ("u", "j"):
            self.sim.lift.set_velocity(0.0)
        elif key in ("h", "k"):
            self.sim.arm.set_velocity(0.0)
        elif key in ("o", "p"):
            self.sim.end_of_arm.wrist_yaw.set_velocity(0.0)
        elif key in ("c", "v"):
            self.sim.end_of_arm.wrist_pitch.set_velocity(0.0)
        elif key in ("t", "y"):
            self.sim.end_of_arm.wrist_roll.set_velocity(0.0)
        elif key in ("n", "m"):
            self.sim.end_of_arm.stretch_gripper.set_velocity(0.0)


class GripperHandedness(Enum):
    LEFT = 0
    RIGHT = 1

    def move_to(self, sim: StretchMujocoSimulator):
        """Moves the wrist to achieve this handedness."""
        print(f"Moving wrist to {self.name}")

        if self is GripperHandedness.RIGHT:
            yaw_to = 0.0
            pitch_to = 0.0
            roll_to = 0.0
            sim.end_of_arm.wrist_roll.move_to(roll_to)
            sim.wait_until_at_setpoint(Actuators["wrist_roll"])
            sim.end_of_arm.wrist_pitch.move_to(pitch_to)
            sim.wait_until_at_setpoint(Actuators["wrist_pitch"])
            sim.end_of_arm.wrist_yaw.move_to(yaw_to)
            sim.wait_until_at_setpoint(Actuators["wrist_yaw"])
        elif self is GripperHandedness.LEFT:
            yaw_to = -math.pi
            pitch_to = -math.pi
            roll_to = math.pi
            sim.end_of_arm.wrist_yaw.move_to(yaw_to)
            sim.wait_until_at_setpoint(Actuators["wrist_yaw"])
            sim.end_of_arm.wrist_pitch.move_to(pitch_to)
            sim.wait_until_at_setpoint(Actuators["wrist_pitch"])
            sim.end_of_arm.wrist_roll.move_to(roll_to)
            sim.wait_until_at_setpoint(Actuators["wrist_roll"])
        else:
            raise NotImplementedError(f"No move_to defined for {self}")


class MotionProfile(Enum):
    SLOW = 1
    DEFAULT = 2
    FAST = 3
    MAX = 4

    def cycle(self, is_forward: bool):
        index_offset = 1 if is_forward else -1
        members = list(type(self))
        index = members.index(self)
        return members[(index + index_offset) % len(members)]

    @property
    def multiplier(self):
        if self == MotionProfile.SLOW:
            return 0.5
        elif self == MotionProfile.DEFAULT:
            return 1.5
        elif self == MotionProfile.FAST:
            return 2.5
        elif self == MotionProfile.MAX:
            return 3.0


class ControlMapping(Enum):
    OMNIBASE = 1
    MANIPULATION = 2

    def cycle(self, is_forward: bool):
        index_offset = 1 if is_forward else -1
        members = list(type(self))
        index = members.index(self)
        return members[(index + index_offset) % len(members)]

    def do_motion(self, robot: "_GamepadRobotState", adapter: "GamepadTeleopAdapter"):
        if self == ControlMapping.OMNIBASE:
            return self._map_omnibase(robot, adapter)
        elif self == ControlMapping.MANIPULATION:
            return self._map_manipulation(robot, adapter)
        else:
            raise NotImplementedError(f"No controls callback for {self}")

    def _map_omnibase(self, robot, adapter):
        dxl_zero_vel_set_division_factor = 3
        actuated_joints = {}
        if adapter.use_devices.get("eoa"):
            if adapter.controller_state.get("right_shoulder_button_pressed"):
                adapter.wrist_yaw_command.command_button_to_motion(-1, robot)
                actuated_joints["joint_wrist_yaw"] = 1
            elif adapter.controller_state.get("left_shoulder_button_pressed"):
                adapter.wrist_yaw_command.command_button_to_motion(1, robot)
                actuated_joints["joint_wrist_yaw"] = -1
            else:
                if adapter._i % dxl_zero_vel_set_division_factor == 0:
                    adapter.wrist_yaw_command.stop_motion(robot)
            if adapter.controller_state.get("top_pad_pressed"):
                cmd = 1 if adapter.gripper_handedness is GripperHandedness.RIGHT else -1
                adapter.wrist_pitch_command.command_button_to_motion(cmd, robot)
                actuated_joints["joint_wrist_pitch"] = cmd
            elif adapter.controller_state.get("bottom_pad_pressed"):
                cmd = -1 if adapter.gripper_handedness is GripperHandedness.RIGHT else 1
                adapter.wrist_pitch_command.command_button_to_motion(cmd, robot)
                actuated_joints["joint_wrist_pitch"] = cmd
            else:
                if adapter._i % dxl_zero_vel_set_division_factor == 0:
                    adapter.wrist_pitch_command.stop_motion(robot)
            if adapter.controller_state.get("left_pad_pressed"):
                adapter.wrist_roll_command.command_button_to_motion(1, robot)
                actuated_joints["joint_wrist_roll"] = -1
            elif adapter.controller_state.get("right_pad_pressed"):
                adapter.wrist_roll_command.command_button_to_motion(-1, robot)
                actuated_joints["joint_wrist_roll"] = 1
            else:
                if adapter._i % dxl_zero_vel_set_division_factor == 0:
                    adapter.wrist_roll_command.stop_motion(robot)

        if adapter.use_devices.get("arm"):
            cmd = adapter.controller_state.get("right_stick_x", 0) if adapter.use_arm_lift_mode else 0
            adapter.arm_command.command_stick_to_motion(cmd, robot)
            if abs(cmd) > 0.1:
                actuated_joints["arm"] = cmd
        if adapter.use_devices.get("lift"):
            cmd = adapter.controller_state.get("right_stick_y", 0) if adapter.use_arm_lift_mode else 0
            adapter.lift_command.command_stick_to_motion(cmd, robot)
            if abs(cmd) > 0.1:
                actuated_joints["lift"] = cmd
        if adapter.use_devices.get("base"):
            cmd_y = adapter.controller_state.get("left_stick_y", 0)
            cmd_x = -adapter.controller_state.get("left_stick_x", 0)
            cmd_t = (
                -adapter.controller_state.get("right_stick_x", 0)
                if not adapter.use_arm_lift_mode
                else 0
            )
            adapter.base_command.command_stick_to_motion(cmd_y, cmd_x, cmd_t, robot)
            if abs(cmd_y) > 0.1 or abs(cmd_x) > 0.1 or abs(cmd_t) > 0.1:
                actuated_joints["base"] = cmd_x + cmd_y + cmd_t

        if adapter.use_devices.get("gripper"):
            if adapter.controller_state.get("right_button_pressed"):
                adapter.gripper.open_gripper(robot)
                actuated_joints[adapter.gripper.name] = 1
            elif adapter.controller_state.get("bottom_button_pressed"):
                adapter.gripper.close_gripper(robot)
                actuated_joints[adapter.gripper.name] = -1
            else:
                adapter.gripper.stop_gripper(robot)

        return actuated_joints

    def _map_manipulation(self, robot, adapter):
        adapter.precision_mode = adapter.controller_state.get("left_trigger_pulled", 0) > 0.9
        adapter.use_arm_lift_mode = adapter.controller_state.get("right_trigger_pulled", 0) > 0.9

        dxl_zero_vel_set_division_factor = 3

        right_stick_x = adapter.controller_state.get("right_stick_x", 0)
        right_stick_y = adapter.controller_state.get("right_stick_y", 0)

        actuated_joints = {}
        if adapter.use_devices.get("lift"):
            if adapter.controller_state.get("top_pad_pressed"):
                adapter.lift_command.command_button_to_motion(0.4, robot)
                actuated_joints["lift"] = 0.4
            elif adapter.controller_state.get("bottom_pad_pressed"):
                adapter.lift_command.command_button_to_motion(-0.4, robot)
                actuated_joints["lift"] = -0.4
            else:
                if adapter._i % dxl_zero_vel_set_division_factor == 0:
                    adapter.lift_command.stop_motion(robot)

        if adapter.use_devices.get("eoa") and adapter.use_arm_lift_mode:
            adapter.base_command.stop_motion(robot)

            if abs(right_stick_x) > 0.1:
                adapter.wrist_yaw_command.command_stick_to_motion(right_stick_x, robot)
                actuated_joints["joint_wrist_yaw"] = right_stick_x

            if abs(right_stick_y) > 0.1:
                handedness_inversion = (
                    -1 if adapter.gripper_handedness is GripperHandedness.LEFT else 1
                )
                cmd = handedness_inversion * right_stick_y
                adapter.wrist_pitch_command.command_stick_to_motion(cmd, robot)
                actuated_joints["joint_wrist_pitch"] = right_stick_y

            if adapter.controller_state.get("left_pad_pressed"):
                adapter.wrist_roll_command.command_button_to_motion(1, robot)
                actuated_joints["joint_wrist_roll"] = -1
            elif adapter.controller_state.get("right_pad_pressed"):
                adapter.wrist_roll_command.command_button_to_motion(-1, robot)
                actuated_joints["joint_wrist_roll"] = 1
            else:
                if adapter._i % dxl_zero_vel_set_division_factor == 0:
                    adapter.wrist_roll_command.stop_motion(robot)

            if adapter.use_devices.get("arm"):
                cmd = (
                    adapter.controller_state.get("left_stick_y", 0)
                    if adapter.use_arm_lift_mode
                    else 0
                )
                adapter.arm_command.command_stick_to_motion(cmd, robot)
                if abs(cmd) > 0.1:
                    actuated_joints["arm"] = cmd

        else:
            if adapter.use_devices.get("arm"):
                adapter.arm_command.stop_motion(robot)
            if adapter.use_devices.get("eoa"):
                adapter.wrist_yaw_command.stop_motion(robot)
                adapter.wrist_pitch_command.stop_motion(robot)
                adapter.wrist_roll_command.stop_motion(robot)

            if adapter.use_devices.get("base"):
                cmd_y = (
                    adapter.controller_state.get("left_stick_y", 0)
                    if not adapter.use_arm_lift_mode
                    else 0
                )
                cmd_x = (
                    -adapter.controller_state.get("left_stick_x", 0)
                    if not adapter.use_arm_lift_mode
                    else 0
                )
                cmd_t = (
                    -adapter.controller_state.get("right_stick_x", 0)
                    if not adapter.use_arm_lift_mode
                    else 0
                )
                adapter.base_command.command_stick_to_motion(cmd_y, cmd_x, cmd_t, robot)
                if abs(cmd_y) > 0.1 or abs(cmd_x) > 0.1 or abs(cmd_t) > 0.1:
                    actuated_joints["base"] = cmd_x + cmd_y + cmd_t

        if adapter.use_devices.get("gripper"):
            if adapter.controller_state.get("right_button_pressed"):
                adapter.gripper.open_gripper(robot)
                actuated_joints[adapter.gripper.name] = 1
            elif adapter.controller_state.get("bottom_button_pressed"):
                adapter.gripper.close_gripper(robot)
                actuated_joints[adapter.gripper.name] = -1
            else:
                adapter.gripper.stop_gripper(robot)

        return actuated_joints


def _get_subsystem_by_name(sim, name):
    if name in ["lift", "arm"]:
        return getattr(sim, name)
    elif name in ["wrist_yaw", "wrist_pitch", "wrist_roll"]:
        return getattr(sim.end_of_arm, name)
    elif name == "gripper":
        return sim.end_of_arm.stretch_gripper
    elif name in ["head_pan", "head_tilt"]:
        return getattr(sim.head, name) if hasattr(sim, "head") else None
    return None


class _MockCommand:
    def __init__(self, sim, actuator_name, scale=1.0):
        self.sim = sim
        self.actuator_name = actuator_name
        self.scale = scale
        self.name = actuator_name

    def command_button_to_motion(self, val, robot):
        sub = _get_subsystem_by_name(self.sim, self.actuator_name)
        if sub:
            sub.move_by(val * self.scale * robot.precision_multiplier * robot.profile_multiplier)

    def command_stick_to_motion(self, val, robot):
        sub = _get_subsystem_by_name(self.sim, self.actuator_name)
        if sub:
            sub.move_by(val * self.scale * robot.precision_multiplier * robot.profile_multiplier)

    def stop_motion(self, robot):
        pass


class _MockGripperCommand:
    def __init__(self, sim):
        self.sim = sim
        self.name = "gripper"

    def open_gripper(self, robot):
        val = 0.07
        self.sim.end_of_arm.stretch_gripper.move_by(val * robot.precision_multiplier)

    def close_gripper(self, robot):
        val = -0.07
        self.sim.end_of_arm.stretch_gripper.move_by(val * robot.precision_multiplier)

    def stop_gripper(self, robot):
        pass


class _MockBaseCommand:
    def __init__(self, sim):
        self.sim = sim
        self.name = "base"

    def command_stick_to_motion(self, cmd_y, cmd_x, cmd_t, robot):
        if abs(cmd_x) < 0.001:
            cmd_x = 0
        if abs(cmd_y) < 0.001:
            cmd_y = 0
        if abs(cmd_t) < 0.001:
            cmd_t = 0

        velocity = 0.3  # m/s
        angular_velocity = 1.0  # rad/s

        v_x_linear = cmd_y * velocity * robot.precision_multiplier * robot.profile_multiplier
        v_y_linear = cmd_x * velocity * robot.precision_multiplier * robot.profile_multiplier
        omega = cmd_t * angular_velocity * robot.precision_multiplier * robot.profile_multiplier

        if isinstance(self.sim, Stretch4MujocoSimulator):
            self.sim.base.set_velocity(v_x_linear, v_y_linear, omega)
        else:
            self.sim.base.set_velocity(v_y_linear, 0.0, omega)

    def stop_motion(self, robot):
        self.sim.base.set_velocity(0.0, 0.0, 0.0)


class GamepadTeleopAdapter:
    def __init__(self, sim):
        self._i = 0
        self.use_devices = {"eoa": True, "arm": True, "lift": True, "base": True, "gripper": True}
        self.gripper_handedness = GripperHandedness.RIGHT
        self.use_arm_lift_mode = False
        self.precision_mode = False
        self.controller_state = {}

        self.wrist_yaw_command = _MockCommand(sim, "wrist_yaw", 0.2)
        self.wrist_pitch_command = _MockCommand(sim, "wrist_pitch", 0.2)
        self.wrist_roll_command = _MockCommand(sim, "wrist_roll", 0.2)
        self.arm_command = _MockCommand(sim, "arm", 0.05)
        self.lift_command = _MockCommand(sim, "lift", 0.1)
        self.base_command = _MockBaseCommand(sim)
        self.gripper = _MockGripperCommand(sim)


class _GamepadRobotState:
    def __init__(self):
        self.precision_multiplier = 1.0
        self.profile_multiplier = 1.0


class GamepadTeleop:
    """
    Drives `sim` from an Xbox-style gamepad over `GamePadController`.

    Once started, a background thread polls the gamepad state at `rate_hz`
    and applies the same button/stick mapping as
    `examples/gamepad_teleop.py`: MANIPULATION mapping by default (right
    stick = wrist yaw/pitch, D-pad = lift, right trigger = arm/lift stick
    mode), cycled to OMNIBASE with the select/back button. See
    `examples/gamepad_teleop.py`'s module docstring-equivalent controls list
    for the full mapping.
    """

    def __init__(
        self,
        sim: StretchMujocoSimulator,
        use_head_joints: bool | None = None,
        rate_hz: float = 15.0,
    ):
        self.sim = sim
        self.use_head_joints = (
            use_head_joints if use_head_joints is not None else not isinstance(sim, Stretch4MujocoSimulator)
        )
        self._rate_hz = rate_hz

        self.gamepad = GamePadController()
        self.mapping = ControlMapping.MANIPULATION
        self.motion_profile = MotionProfile.DEFAULT
        self.adapter = GamepadTeleopAdapter(sim)
        self._robot = _GamepadRobotState()
        self._dex_switch = False

        self._select_button_counter = ButtonPressCounter("select_button_pressed")
        self._start_button_counter = ButtonPressCounter("start_button_pressed")
        self._top_button_counter = ButtonPressCounter("top_button_pressed")
        self._left_button_counter = ButtonPressCounter("left_button_pressed")

        self._effort_trackers = {
            "joint_lift": JointEffortTracker("lift", pos_thresholds=[100.0, 200.0]),
            "joint_arm": JointEffortTracker("arm", pos_thresholds=[50.0, 100.0]),
            "joint_wrist_yaw": JointEffortTracker(
                "eoa", pos_thresholds=[2.0, 5.0], joint_name="wrist_yaw"
            ),
            "joint_wrist_pitch": JointEffortTracker(
                "eoa", pos_thresholds=[2.0, 5.0], joint_name="wrist_pitch"
            ),
            "joint_wrist_roll": JointEffortTracker(
                "eoa", pos_thresholds=[2.0, 5.0], joint_name="wrist_roll"
            ),
            "gripper": JointEffortTracker("eoa", pos_thresholds=[5.0, 15.0], joint_name="gripper"),
        }

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self):
        self._stop_event.clear()
        self.gamepad.start()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self.gamepad.stop()

    def _change_mapping(self):
        self.mapping = self.mapping.cycle(is_forward=True)
        print(f"Switched mapping to {self.mapping.name}")
        self.gamepad.vibrate(duration_ms=150, strong_magnitude=1.0, weak_magnitude=1.0)

    def _change_handedness(self):
        self.adapter.gripper_handedness = (
            GripperHandedness.LEFT
            if self.adapter.gripper_handedness == GripperHandedness.RIGHT
            else GripperHandedness.RIGHT
        )
        self.adapter.gripper_handedness.move_to(self.sim)
        duration = 150 * (self.adapter.gripper_handedness.value + 1)
        self.gamepad.vibrate(duration_ms=duration, strong_magnitude=1.0, weak_magnitude=1.0)

    def _change_motion_profile(self):
        self.motion_profile = self.motion_profile.cycle(is_forward=True)
        print(f"Switched motion profile to {self.motion_profile.name}")
        duration = 150 * self.motion_profile.value
        self.gamepad.vibrate(duration_ms=duration, strong_magnitude=1.0, weak_magnitude=1.0)

    def _toggle_dex_switch(self):
        self._dex_switch = not self._dex_switch
        print(f"Setting dex_switch to {self._dex_switch}")

    def _loop(self):
        period = 1.0 / self._rate_hz
        while not self._stop_event.is_set() and self.sim.is_running():
            start = time.perf_counter()
            gamepad_state = self.gamepad.get_state()

            self._robot.precision_multiplier = 1.0 - 0.75 * gamepad_state.get(
                "left_trigger_pulled", 0.0
            )
            self.adapter.controller_state = gamepad_state.copy()
            self.adapter._i += 1

            # Use back_button_pressed if it's pressed as fallback for select
            if gamepad_state.get("back_button_pressed", False):
                gamepad_state["select_button_pressed"] = True

            self._select_button_counter.step(gamepad_state)
            self._start_button_counter.step(gamepad_state)
            self._top_button_counter.step(gamepad_state)
            self._left_button_counter.step(gamepad_state)

            self._select_button_counter.trigger_on_tap(self._change_mapping)
            self._start_button_counter.trigger_on_tap(self._change_handedness)
            self._top_button_counter.trigger_on_tap(self._change_motion_profile)

            if self.use_head_joints:
                self._left_button_counter.trigger_on_tap(self._toggle_dex_switch)

                if self._dex_switch and hasattr(self.sim, "head"):
                    if gamepad_state.get("bottom_pad_pressed"):
                        self.sim.head.head_tilt.move_by(1 * 0.2 * self._robot.precision_multiplier)
                    elif gamepad_state.get("top_pad_pressed"):
                        self.sim.head.head_tilt.move_by(-1 * 0.2 * self._robot.precision_multiplier)

                    if gamepad_state.get("left_pad_pressed"):
                        self.sim.head.head_pan.move_by(1 * 0.2 * self._robot.precision_multiplier)
                    elif gamepad_state.get("right_pad_pressed"):
                        self.sim.head.head_pan.move_by(-1 * 0.2 * self._robot.precision_multiplier)

                    self.adapter.controller_state["bottom_pad_pressed"] = False
                    self.adapter.controller_state["top_pad_pressed"] = False
                    self.adapter.controller_state["left_pad_pressed"] = False
                    self.adapter.controller_state["right_pad_pressed"] = False

            self._robot.profile_multiplier = self.motion_profile.multiplier

            actuated_joints = self.mapping.do_motion(self._robot, self.adapter)

            status = self.sim.pull_status()
            if status.is_self_colliding:
                self.gamepad.vibrate_sequence(
                    sequence_ms=[150, 100, 200],
                    strong_magnitude=0.5,
                    weak_magnitude=1.0,
                    tag="collision",
                    cooldown=2.0,
                )

            if actuated_joints and self.adapter.precision_mode:
                for joint_id, tracker in self._effort_trackers.items():
                    direction = actuated_joints.get(joint_id, 0)
                    is_actuated = direction != 0

                    tracker.step(status, is_actuated, direction)
                    tracker.trigger_on_hold(0.1, self._make_vibrate_callback(joint_id, tracker))

            elapsed = time.perf_counter() - start
            if elapsed < period:
                time.sleep(period - elapsed)

    def _make_vibrate_callback(self, joint_id, tracker):
        def trigger_vibrate(effort):
            try:
                abs_effort = abs(effort)
                max_e = tracker.pos_thresholds[1] if tracker.last_direction >= 0 else tracker.neg_thresholds[1]
                min_e = tracker.pos_thresholds[0] if tracker.last_direction >= 0 else tracker.neg_thresholds[0]
                if max_e > min_e:
                    fraction = min(1.0, max(0.0, (abs_effort - min_e) / (max_e - min_e)))
                else:
                    fraction = 1.0 if abs_effort >= min_e else 0.0

                self.gamepad.vibrate(
                    duration_ms=100,
                    strong_magnitude=fraction,
                    weak_magnitude=fraction,
                    tag=f"effort_{joint_id}",
                    cooldown=0.25,
                )
            except Exception as e:
                logging.error(f"Got error {e}", exc_info=True)

        return trigger_vibrate
