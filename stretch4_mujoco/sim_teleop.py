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

`GamepadTeleop` mirrors the real robot's `stretch4_body.core.gamepad_teleop`:
the same two control mappings (JOINT_SPACE and FLYING_GRIPPER_IK, cycled with
Y), the same joint command classes, and the same modifier buttons. The
supporting modules are ports of their robot-side counterparts, kept close to the
originals so they can be diffed:

    gamepad_controller.py       <- stretch4_body/core/gamepad_controller.py
    gamepad_enums.py            <- stretch4_body/core/gamepad_enums.py
    gamepad_joints.py           <- stretch4_body/core/gamepad_joints.py
    gamepad_control_mappings.py <- stretch4_body/core/gamepad_control_mappings.py

See `examples/keyboard_teleop.py`, `examples/gamepad_teleop.py`, and
`examples/molmo_environment.py` (--keyboard/--gamepad) for scripts built on
top of these.
"""

from __future__ import annotations

import threading
import time
import traceback
from typing import Callable

import click
from pynput import keyboard

from stretch4_mujoco import gamepad_joints
from stretch4_mujoco.gamepad_control_mappings import TRIGGER_THRESHOLD, ControlMapping
from stretch4_mujoco.gamepad_controller import (
    ButtonPressCounter,
    GamePadController,
    JointEffortTracker,
)
from stretch4_mujoco.gamepad_enums import (
    GripperHandedness,
    GuardedContactSensitivity,
    MotionProfile,
)
from stretch4_mujoco.stretch4_mujoco_simulator import Stretch4MujocoSimulator
from stretch4_mujoco.stretch_mujoco_simulator import StretchMujocoSimulator

# Header constants, mirroring stretch4_body/core/gamepad_teleop.py
STEP_SLEEP = 1 / 15

# Button Hold Durations
START_BUTTON_HOLD_TIME_S = 3
FN_BUTTON_DETECT_SPAN_S = 0.5

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


class GamepadTeleop:
    """
    Drives `sim` from an Xbox-style gamepad, mirroring the real robot's
    `stretch4_body.core.gamepad_teleop.GamePadTeleop`.

    Both robot control mappings are available and are cycled with the Y button:

    * `ControlMapping.JOINT_SPACE` (default) - direct joint control.
    * `ControlMapping.FLYING_GRIPPER_IK` - IK-based Cartesian control of the gripper.

    Print `teleop.control_mapping.description()` for the full button table of the
    active mapping. Modifiers are the robot's:

    * LT              - precision mode (scales every joint down)
    * RT              - mapping-specific modifier (straight-line base, wrist roll, ...)
    * Y               - cycle control mapping
    * RT + A          - cycle motion profile (SLOW / MEDIUM / FAST)
    * RT + B          - cycle contact-sensitivity profile (announced only; no-op in sim)
    * RT + Back       - announce the current settings
    * Hold Back 3s    - stow
    * Hold Start 3s   - change gripper handedness (moves the wrist)
    * Hold X 0.5s     - run `fn_button_command` (disabled unless `enable_fn_button` is set)
    * Hold L3 / R3    - run `left_stick_button_fn` / `right_stick_button_fn` if assigned

    Once started, a background thread polls the gamepad and steps the mapping at
    `1 / STEP_SLEEP` Hz.
    """

    def __init__(
        self,
        sim: StretchMujocoSimulator,
        rate_hz: float | None = None,
        print_mapping_on_start: bool = True,
    ):
        self.sim = sim
        self.robot = sim  # the mappings take the simulator where the robot takes a Robot

        self.motion_profile = MotionProfile.MEDIUM
        self.gripper_handedness = GripperHandedness.RIGHT
        self.control_mapping = ControlMapping.JOINT_SPACE
        self.contact_sensitivity_profile = GuardedContactSensitivity.MEDIUM

        self.gamepad_controller = GamePadController()
        self.precision_mode = 0.0
        self.use_arm_lift_mode = False
        self.controller_state = self.gamepad_controller.get_state()
        self.status = sim.pull_status()

        self.sleep = STEP_SLEEP if rate_hz is None else 1.0 / rate_hz
        self.print_mode = False
        self.print_mapping_on_start = print_mapping_on_start
        self._i = 0

        # The robot reads these off its Device params; there is no robot config in sim.
        self.enable_fn_button = False
        self.fn_button_command: str | None = None
        self.fn_button_detect_span = FN_BUTTON_DETECT_SPAN_S

        self._last_fn_btn_press = None
        self._last_left_stick_fn_btn_press = None
        self._last_right_stick_fn_btn_press = None
        self.start_button_counter = ButtonPressCounter("start_button_pressed")
        self.top_button_counter = ButtonPressCounter("top_button_pressed")
        self.bottom_button_counter = ButtonPressCounter("bottom_button_pressed")
        self.right_button_counter = ButtonPressCounter("right_button_pressed")
        self.select_button_counter = ButtonPressCounter("select_button_pressed")
        self.gripper = None

        self.gripper_name = "stretch_gripper"
        self.use_devices = {
            "arm": True,
            "eoa": True,
            "lift": True,
            "base": True,
            "gripper": True,
        }

        self.effort_trackers = {
            "lift": JointEffortTracker(
                "lift", pos_thresholds=[34.0, 45.0], neg_thresholds=[25.0, 35.0]
            ),
            "arm": JointEffortTracker(
                "arm", pos_thresholds=[10.0, 20.0], neg_thresholds=[10.0, 20.0]
            ),
            "wrist_yaw_joint": JointEffortTracker(
                "eoa",
                pos_thresholds=[3.0, 10.0],
                neg_thresholds=[3.0, 10.0],
                joint_name="wrist_yaw",
            ),
            "wrist_pitch_joint": JointEffortTracker(
                "eoa",
                pos_thresholds=[3.0, 10.0],
                neg_thresholds=[3.0, 10.0],
                joint_name="wrist_pitch",
            ),
            "wrist_roll_joint": JointEffortTracker(
                "eoa",
                pos_thresholds=[3.0, 10.0],
                neg_thresholds=[3.0, 10.0],
                joint_name="wrist_roll",
            ),
            self.gripper_name: JointEffortTracker(
                "eoa",
                pos_thresholds=[5.0, 20.0],
                neg_thresholds=[5.0, 20.0],
                joint_name=self.gripper_name,
            ),
        }

        self.lock = threading.Lock()

        self.left_stick_button_fn: Callable | None = None
        self.right_stick_button_fn: Callable | None = None
        self.currently_stowing = False

        self._flying_gripper_controller = None

        self.contact_sensitivity_profile.apply(self.sim)

        self.set_joint_command()

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    @property
    def flying_gripper_controller(self):
        """The IK controller is built lazily: generating the planar-IK URDF and loading
        pinocchio takes a few seconds, and JOINT_SPACE never needs it."""
        if self._flying_gripper_controller is None:
            self._flying_gripper_controller = _build_flying_gripper_controller(self.sim)
        return self._flying_gripper_controller

    def set_joint_command(self):
        self.base_command = gamepad_joints.CommandBase(
            motion_profile=self.motion_profile.get_name(),
            motion_profile_angular=self.motion_profile.get_one_lower_speed().get_name(),
        )
        self.lift_command = gamepad_joints.CommandLift(
            motion_profile=self.motion_profile.get_name()
        )
        if self.use_devices["arm"]:
            self.arm_command = gamepad_joints.CommandArm(
                motion_profile=self.motion_profile.get_name()
            )
        if self.use_devices["eoa"]:
            self.wrist_yaw_command = gamepad_joints.CommandWristYaw(
                motion_profile=self.motion_profile.get_name(), dt=self.sleep
            )
            self.wrist_pitch_command = gamepad_joints.CommandWristPitch(
                motion_profile=self.motion_profile.get_name(), dt=self.sleep
            )
            self.wrist_roll_command = gamepad_joints.CommandWristRoll(
                motion_profile=self.motion_profile.get_name(), dt=self.sleep
            )
        if self.use_devices["gripper"]:
            self.gripper = gamepad_joints.CommandStretchGripperPosition(
                motion_profile=self.motion_profile.get_name()
            )

    # --- Settings cycling -------------------------------------------------

    def cycle_motion_profile(self):
        self.motion_profile = self.motion_profile.cycle(is_forward=True)

        print(f"Switched to {self.motion_profile.name} motion_profile.")

        self.motion_profile.play_sound_file()
        duration = 150 * self.motion_profile.value
        self.gamepad_controller.vibrate(
            duration_ms=duration, strong_magnitude=1.0, weak_magnitude=1.0
        )
        self.set_joint_command()

    def cycle_mapping(self):
        next_mapping = self.control_mapping.cycle(is_forward=True)

        if next_mapping == ControlMapping.FLYING_GRIPPER_IK:
            # Warm the IK controller up so the first frame is not a multi-second stall,
            # and stay on the current mapping if this robot model has no IK URDF.
            try:
                self.flying_gripper_controller
            except Exception as e:
                print(f"Cannot switch to {next_mapping.name}: {e}")
                self.gamepad_controller.vibrate(
                    duration_ms=400, strong_magnitude=1.0, weak_magnitude=1.0
                )
                return

        self.control_mapping = next_mapping

        print(f"Switched to {self.control_mapping.name} gamepad mapping.")

        self.control_mapping.play_sound_file()
        if self.control_mapping == ControlMapping.FLYING_GRIPPER_IK:
            self.gamepad_controller.vibrate_sequence(
                sequence_ms=[150, 100, 150],
                strong_magnitude=1.0,
                weak_magnitude=1.0,
                tag="mapping_fg",
                cooldown=0.0,
            )
        else:
            self.gamepad_controller.vibrate(
                duration_ms=300, strong_magnitude=1.0, weak_magnitude=1.0
            )

    def cycle_contact_sensitivity_profile(self):
        self.contact_sensitivity_profile = self.contact_sensitivity_profile.cycle(is_forward=True)

        print(f"Switched to {self.contact_sensitivity_profile.name} contact_sensitivity_profile.")

        self.contact_sensitivity_profile.play_sound_file()
        duration = 150 * self.contact_sensitivity_profile.value
        self.gamepad_controller.vibrate(
            duration_ms=duration, strong_magnitude=1.0, weak_magnitude=1.0
        )
        self.contact_sensitivity_profile.apply(self.sim)

    def _handle_vibration(self, actuated_joints):
        """
        Handle effort-based vibration feedback for the gamepad controller.

        Not called by `do_motion()`, matching the robot, where it is commented out.
        """
        for joint_id, tracker in self.effort_trackers.items():
            is_actuated = joint_id in actuated_joints
            tracker.step(self.status, is_actuated, actuated_joints.get(joint_id, 0))

            if not is_actuated:
                continue

            def trigger_vibrate(effort, j_id=joint_id, t=tracker):
                strong_mag = 1.0
                weak_mag = 1.0
                try:
                    thresholds = t.pos_thresholds if t.last_direction >= 0 else t.neg_thresholds
                    min_e, max_e = thresholds
                    abs_effort = abs(effort)
                    if max_e > min_e:
                        fraction = min(1.0, max(0.0, (abs_effort - min_e) / (max_e - min_e)))
                        strong_mag = 0.2 + 0.8 * fraction
                        weak_mag = strong_mag
                except Exception:
                    pass

                self.gamepad_controller.vibrate_sequence(
                    sequence_ms=[100, 50, 100],
                    strong_magnitude=strong_mag,
                    weak_magnitude=weak_mag,
                    tag=f"effort_{j_id}",
                    cooldown=0.1,
                )

            tracker.trigger_on_hold(0.25, trigger_vibrate)

    # --- Control loop -----------------------------------------------------

    def do_motion(self, state=None, robot=None):
        """
        This method should be called in the control loop (mainloop())

        Parameters
        ----------
        state : Dict
            Override the gamepad controller state providing custom state,
            check out GamePadController.get_state()
        robot : StretchMujocoSimulator
            Valid simulator instance

        Returns
        -------
        Whether the robot was commanded to do some motion
        """
        if not robot:
            robot = self.sim
        self._i = self._i + 1
        self._update_state(state)
        self._update_modes()
        with self.lock:
            if self.currently_stowing:  # No control during stowing
                return False

            if self.controller_state is None:  # No control if gamepad not being controlled
                return False

            self.status = robot.pull_status()

            self.manage_start_button(robot)

            # Regular control
            if self.gamepad_controller.is_gamepad_active or state:
                self.manage_fn_button(robot, self.controller_state["left_button_pressed"])

                self.precision_mode = self.controller_state["left_trigger_pulled"]
                self.use_arm_lift_mode = (
                    self.controller_state["right_trigger_pulled"] > TRIGGER_THRESHOLD
                )

                actuated_joints = self.control_mapping.do_motion(robot, self)

                if actuated_joints:
                    if self.status.is_self_colliding:
                        self.gamepad_controller.vibrate_sequence(
                            sequence_ms=[150, 100, 200],
                            strong_magnitude=1.0,
                            weak_magnitude=1.0,
                            tag="collision",
                            cooldown=1.0,
                        )

                    # if self.precision_mode:
                    #     self._handle_vibration(actuated_joints)

                self.manage_settings_buttons(robot)

                self.manage_left_stick_fn_button(self.controller_state["left_stick_button_pressed"])
                self.manage_right_stick_fn_button(
                    self.controller_state["right_stick_button_pressed"]
                )
            else:
                self._safety_stop(robot)
        return True

    def _update_state(self, state=None):
        with self.lock:
            self.controller_state = state if state else self.gamepad_controller.get_state()

    def _update_modes(self):
        if self.use_devices["arm"]:
            self.arm_command.precision_mode = self.precision_mode
        self.lift_command.precision_mode = self.precision_mode
        self.base_command.precision_mode = self.precision_mode
        if self.use_devices["gripper"]:
            self.gripper.precision_mode = self.precision_mode
        if self.use_devices["eoa"]:
            self.wrist_pitch_command.precision_mode = self.precision_mode
            self.wrist_roll_command.precision_mode = self.precision_mode
            self.wrist_yaw_command.precision_mode = self.precision_mode

    # --- Buttons ----------------------------------------------------------

    def manage_settings_buttons(self, robot):
        """
        Manage settings and mode switching.
        """
        rt_pulled = self.controller_state.get("right_trigger_pulled", 0.0) > TRIGGER_THRESHOLD

        self.top_button_counter.step(self.controller_state)
        self.bottom_button_counter.step(self.controller_state)
        self.right_button_counter.step(self.controller_state)
        self.select_button_counter.step(self.controller_state)

        def on_top_tap():
            self.cycle_mapping()

        self.top_button_counter.trigger_on_tap(on_top_tap)

        def on_bottom_tap():
            if rt_pulled:
                self.cycle_motion_profile()

        self.bottom_button_counter.trigger_on_tap(on_bottom_tap)

        def on_right_tap():
            if rt_pulled:
                self.cycle_contact_sensitivity_profile()

        self.right_button_counter.trigger_on_tap(on_right_tap)

        def on_select_tap():
            if rt_pulled:
                self.gripper_handedness.play_sound_file()
                self.motion_profile.play_sound_file()
                self.contact_sensitivity_profile.play_sound_file()
                self.control_mapping.play_sound_file()

        self.select_button_counter.trigger_on_tap(on_select_tap)

        def on_select_hold():
            self.stow_robot()

        self.select_button_counter.trigger_on_hold(START_BUTTON_HOLD_TIME_S, on_select_hold)

    def manage_start_button(self, robot):
        """
        Manage the state of the Start button.

        The robot homes on a Start tap when uncalibrated; a simulated robot is always
        homed, so only the hold behaviour applies: holding Start for
        START_BUTTON_HOLD_TIME_S (3s) changes gripper handedness with motion.
        """
        self.start_button_counter.step(self.controller_state)

        if not self.currently_stowing:
            """If the user holds the start button, it will do the automatic handedness
            change motion"""
            self.start_button_counter.trigger_on_hold(
                START_BUTTON_HOLD_TIME_S,
                lambda: self.change_gripper_handedness(robot, do_motion=True),
            )

    def change_gripper_handedness(self, robot, *, do_motion: bool):
        """
        Change the gripper handedness (Left/Right).

        Args:
            robot (StretchMujocoSimulator): Valid simulator instance.
            do_motion (bool): If True, the wrist will physically move to the new orientation.
        """
        if not self.use_devices["eoa"]:
            print("No eoa device")
            return

        if self.use_devices["lift"] and self.status.lift.pos <= 0.35:
            print("Lift too low for handedness change")
            self.gamepad_controller.vibrate(
                duration_ms=400, strong_magnitude=1.0, weak_magnitude=1.0
            )
            return

        print("Switching gripper handedness")

        if self.gripper_handedness == GripperHandedness.RIGHT:
            self.gripper_handedness = GripperHandedness.LEFT
        else:
            self.gripper_handedness = GripperHandedness.RIGHT

        self.gripper_handedness.play_sound_file()
        duration = 150 * (self.gripper_handedness.value + 1)
        self.gamepad_controller.vibrate(
            duration_ms=duration, strong_magnitude=1.0, weak_magnitude=1.0
        )

        if do_motion:
            self.gripper_handedness.move_to(robot)

    def manage_left_stick_fn_button(self, button_state):
        """
        Trigger custom user function for left stick button press.

        The function is executed if the button is held for `fn_button_detect_span`.
        """
        if self.left_stick_button_fn is None:
            return

        if button_state:
            if not self._last_left_stick_fn_btn_press:
                self._last_left_stick_fn_btn_press = time.time()

            if time.time() - self._last_left_stick_fn_btn_press >= self.fn_button_detect_span:
                click.secho("Executing Left Stick Custom Function", fg="green", bold=True)
                self.left_stick_button_fn()
                self._last_left_stick_fn_btn_press = None
        else:
            self._last_left_stick_fn_btn_press = None

    def manage_right_stick_fn_button(self, button_state):
        """
        Trigger custom user function for right stick button press.

        The function is executed if the button is held for `fn_button_detect_span`.
        """
        if self.right_stick_button_fn is None:
            return

        if button_state:
            if not self._last_right_stick_fn_btn_press:
                self._last_right_stick_fn_btn_press = time.time()

            if time.time() - self._last_right_stick_fn_btn_press >= self.fn_button_detect_span:
                click.secho("Executing Right Stick Custom Function", fg="green", bold=True)
                self.right_stick_button_fn()
                self._last_right_stick_fn_btn_press = None
        else:
            self._last_right_stick_fn_btn_press = None

    def manage_fn_button(self, robot, button_state):
        """
        Detect function button press (X / left button).

        Executes `fn_button_command` in a detached shell if the button is held for
        FN_BUTTON_DETECT_SPAN_S. Disabled by default: set `enable_fn_button` and
        `fn_button_command` to use it.
        """
        if self.enable_fn_button:
            if button_state:
                if not self._last_fn_btn_press:
                    self._last_fn_btn_press = time.time()

                if time.time() - self._last_fn_btn_press >= FN_BUTTON_DETECT_SPAN_S:
                    self._last_fn_btn_press = None
                    click.secho(
                        f"Executing Function command: {self.fn_button_command}",
                        fg="green",
                        bold=True,
                    )
                    self._execute_fn_cmd()
            else:
                self._last_fn_btn_press = None

    def _execute_fn_cmd(self):
        if self.fn_button_command:
            execute_command_non_blocking(self.fn_button_command)

    def _safety_stop(self, robot):
        """
        Stop all robot motions.

        This is called when the gamepad is inactive or no input is detected to ensure
        the robot doesn't drift or continue moving.
        """
        if self.use_devices["eoa"]:
            self.wrist_yaw_command.command_button_to_motion(0, robot)
            self.wrist_pitch_command.command_button_to_motion(0, robot)
            self.wrist_roll_command.command_button_to_motion(0, robot)
        if self.use_devices["arm"]:
            self.arm_command.command_stick_to_motion(0, robot)
        if self.use_devices["lift"]:
            self.lift_command.command_stick_to_motion(0, robot)
        if self.use_devices["base"]:
            self.base_command.command_stick_to_motion(0, 0, 0, robot)

    def stow_robot(self):
        """
        Stow the robot to a safe position.
        """
        self.currently_stowing = True
        try:
            self._safety_stop(self.sim)
            self.sim.stow()
        finally:
            self.currently_stowing = False

    # --- Threading --------------------------------------------------------

    def step_mainloop(self, robot=None):
        """
        Execute a single step of the main control loop.
        """
        if not robot:
            robot = self.sim
        self.do_motion(robot=robot)
        time.sleep(self.sleep)

    def mainloop(self):
        """
        Run the main control loop until the simulator stops.
        """
        try:
            while self.sim.is_running() and not self._stop_event.is_set():
                self.step_mainloop()
        except (KeyboardInterrupt, SystemExit):
            self.stop()
        except Exception:
            traceback.print_exc()
            self.stop()

    def start(self):
        """Start the gamepad controller and run the control loop on a background thread."""
        self._stop_event.clear()
        self.gamepad_controller.startup()
        if self.print_mapping_on_start:
            print(self.control_mapping.description())
        self._thread = threading.Thread(target=self.mainloop, daemon=True)
        self._thread.start()

    def stop(self):
        """
        Stop the control loop and the gamepad controller.
        """
        self._stop_event.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=1.0)
            self._thread = None
        self.gamepad_controller.stop()

    # Kept for symmetry with the robot's attribute name.
    @property
    def gamepad(self):
        return self.gamepad_controller


def _build_flying_gripper_controller(sim: StretchMujocoSimulator):
    """Builds the same `KinematicController` the robot uses, from a planar-IK URDF
    generated for the simulated robot model."""
    import tempfile

    if not isinstance(sim, Stretch4MujocoSimulator):
        raise NotImplementedError(
            "FLYING_GRIPPER_IK needs the Stretch 4 planar-IK URDF; it is not available "
            f"for {type(sim).__name__}."
        )

    from stretch4_flying_gripper.kinematic_controller import KinematicController
    from stretch4_urdf import make_planar_ik_urdf
    from yourdfpy import urdf as ud

    out_dir = tempfile.mkdtemp(prefix="sim_gamepad_teleop_")
    robot = ud.URDF.load(type(sim).get_urdf_path())
    urdf_path = make_planar_ik_urdf(
        robot, "sim_gamepad_teleop", out_dir, is_merge_arm=True, is_fixed_wrist=False
    )
    print(f"Flying gripper IK URDF: {urdf_path}")
    return KinematicController(urdf_path)


def execute_command_non_blocking(command):
    import os
    import subprocess

    try:
        # Use subprocess.Popen to start the command in a separate process that won't get
        # killed when the main process is killed
        subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setpgrp,  # Detach the child process from the parent
        )
    except Exception as e:
        print(f"An error occurred: {e}")
