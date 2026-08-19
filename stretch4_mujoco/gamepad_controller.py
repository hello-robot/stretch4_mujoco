#!/usr/bin/env python3
"""
Port of the real robot's `stretch4_body/core/gamepad_controller.py`.

The `GamePadController` is a threading class that polls for gamepad events by
listening to the gamepad's USB dongle, processes them, and makes an easy to
consume gamepad state available as a dictionary - `GamePadController.get_state()`.

Kept deliberately close to the robot-side file so the two can be diffed. The
only removals are the `stretch4_body` dependencies (`Device`, `LoopStats` and
the `/tmp` teleop singleton lock), which have no meaning in simulation.
"""

import glob
import logging
import threading
import time
from dataclasses import dataclass
from select import select
from typing import Callable

from evdev import InputDevice, ecodes, list_devices

logger = logging.getLogger(__name__)

# --- Utilities to discover a joystick device --------------------------------
WANTED_KEY_CODES = {
    ecodes.BTN_SOUTH,
    ecodes.BTN_EAST,
    ecodes.BTN_NORTH,
    ecodes.BTN_WEST,
    ecodes.BTN_TL,
    ecodes.BTN_TR,
    ecodes.BTN_THUMBL,
    ecodes.BTN_THUMBR,
    ecodes.BTN_SELECT,
    ecodes.BTN_START,
    getattr(ecodes, "BTN_MODE", 0),
}
WANTED_ABS_CODES = {
    ecodes.ABS_X,
    ecodes.ABS_Y,
    ecodes.ABS_RX,
    ecodes.ABS_RY,
    ecodes.ABS_Z,
    ecodes.ABS_RZ,
    ecodes.ABS_HAT0X,
    ecodes.ABS_HAT0Y,
}


def is_probable_gamepad(dev: InputDevice) -> bool:
    caps = dev.capabilities()
    if ecodes.EV_ABS not in caps or ecodes.EV_KEY not in caps:
        return False
    keyset = set(caps.get(ecodes.EV_KEY, []))
    absset = set(a if isinstance(a, int) else a[0] for a in caps.get(ecodes.EV_ABS, []))
    # Must have at least one face button and a stick axis
    return (len(keyset & WANTED_KEY_CODES) >= 1) and (len(absset & WANTED_ABS_CODES) >= 2)


def find_first_gamepad() -> str | None:
    # Prefer stable by-id symlinks when present
    for path in sorted(glob.glob("/dev/input/by-id/*-event-joystick")):
        try:
            dev = InputDevice(path)
            ok = is_probable_gamepad(dev)
            dev.close()
            if ok:
                return path
        except Exception:
            pass

    # Fallback: scan all event devices
    for path in list_devices():
        try:
            dev = InputDevice(path)
            ok = is_probable_gamepad(dev)
            dev.close()
            if ok:
                return path
        except Exception:
            pass
    return None


# --- Controller -----------------------------------
class UnpluggedError(Exception):
    pass


@dataclass
class GPEvent:
    code: str
    state: int
    ev_type: str = ""  # not required, but used by the optional print


class Stick:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.norm = float(pow(2, 15))

    def update_x(self, abs_x):
        self.x = int(abs_x) / self.norm

    def update_y(self, abs_y):
        self.y = -int(abs_y) / self.norm

    def print_string(self):
        return "x: {0:4.2f}, y:{1:4.2f}".format(self.x, self.y)


class Button:
    def __init__(self):
        self.pressed = False

    def update(self, state):
        self.pressed = state == 1

    def print_string(self):
        return str(self.pressed)


class Trigger:
    def __init__(self, xbox_one=False):
        num_bits = 10 if xbox_one else 8
        self.norm = float(pow(2, num_bits) - 1)
        self.pulled = 0.0

    def update(self, state):
        self.pulled = int(state) / self.norm
        if self.pulled > 1.0:
            self.pulled = 1.0

    def print_string(self):
        return "{0:4.2f}".format(self.pulled)


class GamePadController:
    """Interface to gamepad controllers.

    Successfully tested with the following controllers:
        + Xbox One Controller connected using a USB cable (change xbox_one parameter to True
          for full 10 bit trigger information)
        + EasySMX wireless controller set to appropriate mode (Xbox 360 mode with upper half
          of ring LED illuminated - top two LED quarter circle arcs)
        + JAMSWALL Xbox 360 Wireless Controller
    """

    def __init__(self, print_events=False, is_xbox_one=False):
        self.name = "gamepad_controller"
        self.print_events = print_events

        self.rumble_effect_id = -1

        self.left_stick = Stick()
        self.right_stick = Stick()

        self.left_stick_button = Button()
        self.right_stick_button = Button()

        self.middle_led_ring_button = Button()

        self.bottom_button = Button()
        self.top_button = Button()
        self.left_button = Button()
        self.right_button = Button()

        self.right_shoulder_button = Button()
        self.left_shoulder_button = Button()

        self.select_button = Button()
        self.start_button = Button()

        self.left_trigger = Trigger(xbox_one=is_xbox_one)
        self.right_trigger = Trigger(xbox_one=is_xbox_one)

        self.left_pad = Button()
        self.right_pad = Button()
        self.top_pad = Button()
        self.bottom_pad = Button()

        self.lock = threading.Lock()

        self.device_path = None
        self.dev: InputDevice | None = None
        self.is_gamepad_active = False

        self._last_vibrated_tags = {}

        # Filtering parameters
        self.last_event_ts = 0.0
        self.EVENT_ACTIVITY_TIMEOUT = 0.5
        self.zero_state_sent_counter = 6
        self.STOP_FRAME_COUNT = 5

        # Threading attributes natively managed
        self.thread_rate_hz = 25.0
        self.thread = None
        self.thread_shutdown_flag = threading.Event()

    def startup(self):
        if self.thread is not None:
            self.thread_shutdown_flag.set()
            self.thread.join(1)
        self.thread = threading.Thread(target=self._thread_target, daemon=True)
        self.thread_shutdown_flag.clear()
        self.thread.start()

        logger.warning("Waiting for Gamepad Dongle...")

        return True

    # `start()` is kept as an alias so existing sim scripts keep working.
    start = startup

    def _thread_target(self):
        period = 1.0 / self.thread_rate_hz
        while not self.thread_shutdown_flag.is_set():
            start = time.perf_counter()
            self.update()
            elapsed = time.perf_counter() - start
            if not self.thread_shutdown_flag.is_set() and elapsed < period:
                time.sleep(period - elapsed)

    def stop(self):
        self.thread_shutdown_flag.set()
        if self.thread is not None:
            self.thread.join(1)
            self.thread = None
        try:
            self.dev.close()
        except Exception:
            pass

    def poll_till_gamepad_dongle_present(self):
        with self.lock:
            self.is_gamepad_active = False
        try:
            self.device_path = find_first_gamepad()
            if self.device_path:
                # Open non-blocking; do NOT grab to allow other readers if needed
                self.dev = InputDevice(self.device_path)
                self.rumble_effect_id = -1
                logger.info(f"Gamepad Dongle FOUND! ({self.dev.name})")
                with self.lock:
                    self.is_gamepad_active = True
        except Exception:
            # keep trying silently
            pass

    # --- Event retrieval (non-blocking) ---
    def get_gamepad_events(self) -> list[GPEvent]:
        if not self.dev:
            raise UnpluggedError("No gamepad found.")
        # Wait briefly for readiness; 20ms timeout keeps thread responsive
        r, _, _ = select([self.dev.fd], [], [], 0.02)
        if not r:
            return []
        events = []
        try:
            for ev in self.dev.read():
                if ev.type not in (ecodes.EV_KEY, ecodes.EV_ABS):
                    continue
                code_name = ecodes.bytype[ev.type][ev.code]  # e.g., 'ABS_X', 'BTN_SOUTH'
                events.append(
                    GPEvent(
                        code=code_name,
                        state=ev.value,
                        ev_type="EV_KEY" if ev.type == ecodes.EV_KEY else "EV_ABS",
                    )
                )
        except BlockingIOError:
            pass
        except OSError:
            # device likely went away
            raise UnpluggedError("Gamepad disconnected.")
        return events

    def update(self):
        if not self.is_gamepad_active:
            self.poll_till_gamepad_dongle_present()
            return

        try:
            events = self.get_gamepad_events()
            if events:
                self.last_event_ts = time.monotonic()
            self.update_button_encodings(events)
        except (UnpluggedError, OSError):
            logger.error("Gamepad Dongle DISCONNECTED...")
            try:
                self.dev.close()
            except Exception:
                pass
            self.dev = None
            with self.lock:
                self.is_gamepad_active = False
            self.set_zero_state()

    def vibrate(
        self,
        duration_ms: int = 150,
        strong_magnitude: float = 1.0,
        weak_magnitude: float = 1.0,
        tag: str | None = None,
        cooldown: float = 0.0,
    ):
        """
        Vibrates the gamepad.
        duration_ms: Duration of the vibration in milliseconds.
        strong_magnitude: Magnitude of the strong motor (0.0 to 1.0).
        weak_magnitude: Magnitude of the weak motor (0.0 to 1.0).
        tag: Optional string tag to identify this vibration event.
        cooldown: Optional cooldown in seconds. If another vibration with the same tag
                  is requested within this cooldown, it will be ignored.
        """
        if tag is not None and cooldown > 0.0:
            now = time.time()
            if now - self._last_vibrated_tags.get(tag, 0.0) < cooldown:
                return
            self._last_vibrated_tags[tag] = now

        with self.lock:
            if not self.dev:
                return
            try:
                from evdev import ff

                strong = max(0, min(int(strong_magnitude * 65535), 65535))
                weak = max(0, min(int(weak_magnitude * 65535), 65535))

                rumble = ff.Rumble(strong_magnitude=strong, weak_magnitude=weak)
                effect_type = ff.EffectType(ff_rumble_effect=rumble)

                effect = ff.Effect(
                    ecodes.FF_RUMBLE,
                    self.rumble_effect_id if self.rumble_effect_id != -1 else -1,
                    0,
                    ff.Trigger(0, 0),
                    ff.Replay(int(duration_ms), 0),
                    effect_type,
                )

                self.rumble_effect_id = self.dev.upload_effect(effect)
                self.dev.write(ecodes.EV_FF, self.rumble_effect_id, 1)
            except Exception:
                pass

    def vibrate_sequence(
        self,
        sequence_ms: list[int],
        strong_magnitude: float = 1.0,
        weak_magnitude: float = 1.0,
        tag: str | None = None,
        cooldown: float = 0.0,
    ):
        """
        Vibrates the gamepad in a sequence of alternating on/off durations.
        sequence_ms: List of durations in milliseconds. Example: [200, 100, 200]
                     will vibrate 200ms, pause 100ms, vibrate 200ms.
        """
        if tag is not None and cooldown > 0.0:
            now = time.time()
            if now - self._last_vibrated_tags.get(tag, 0.0) < cooldown:
                return
            self._last_vibrated_tags[tag] = now

        with self.lock:
            if not self.dev:
                return
        threading.Thread(
            target=self._vibrate_sequence_thread,
            args=(sequence_ms, strong_magnitude, weak_magnitude),
            daemon=True,
        ).start()

    def _vibrate_sequence_thread(
        self, sequence_ms: list[int], strong_magnitude: float, weak_magnitude: float
    ):
        for i, duration in enumerate(sequence_ms):
            if i % 2 == 0:
                self.vibrate(
                    duration_ms=duration,
                    strong_magnitude=strong_magnitude,
                    weak_magnitude=weak_magnitude,
                )
            time.sleep(duration / 1000.0)

    def update_button_encodings(self, events):
        with self.lock:
            for event in events:
                if event.code == "ABS_X":
                    self.left_stick.update_x(event.state)
                if event.code == "ABS_Y":
                    self.left_stick.update_y(event.state)
                if event.code == "ABS_RX":
                    self.right_stick.update_x(event.state)
                if event.code == "ABS_RY":
                    self.right_stick.update_y(event.state)

                # This is the glowing X button on an authentic Xbox controller
                if event.code == "BTN_MODE":
                    self.middle_led_ring_button.update(event.state)

                if "BTN_SOUTH" in list(event.code):  # green A, bottom button
                    self.bottom_button.update(event.state)
                if "BTN_WEST" in list(event.code):  # yellow Y, top button
                    self.top_button.update(event.state)
                if "BTN_NORTH" in list(event.code):  # blue X, left button
                    self.left_button.update(event.state)
                if "BTN_EAST" in list(event.code):  # red B, right button
                    self.right_button.update(event.state)

                if event.code == "BTN_TL":
                    self.left_shoulder_button.update(event.state)
                if event.code == "BTN_TR":
                    self.right_shoulder_button.update(event.state)

                if event.code == "ABS_Z":
                    self.left_trigger.update(event.state)
                if event.code == "ABS_RZ":
                    self.right_trigger.update(event.state)

                if event.code == "BTN_SELECT":
                    self.select_button.update(event.state)
                if event.code == "BTN_START":
                    self.start_button.update(event.state)

                if event.code == "BTN_THUMBL":
                    self.left_stick_button.update(event.state)
                if event.code == "BTN_THUMBR":
                    self.right_stick_button.update(event.state)

                # 4-way pad
                if event.code == "ABS_HAT0Y":
                    if event.state == 0:
                        self.top_pad.update(0)
                        self.bottom_pad.update(0)
                    elif event.state == 1:
                        self.top_pad.update(0)
                        self.bottom_pad.update(1)
                    elif event.state == -1:
                        self.bottom_pad.update(0)
                        self.top_pad.update(1)

                if event.code == "ABS_HAT0X":
                    if event.state == 0:
                        self.left_pad.update(0)
                        self.right_pad.update(0)
                    elif event.state == 1:
                        self.left_pad.update(0)
                        self.right_pad.update(1)
                    elif event.state == -1:
                        self.right_pad.update(0)
                        self.left_pad.update(1)

                if self.print_events:
                    print(event.ev_type, event.code, event.state)

    def set_zero_state(self):
        with self.lock:
            self.middle_led_ring_button.pressed = False
            self.left_stick.x = 0
            self.left_stick.y = 0
            self.right_stick.x = 0
            self.right_stick.y = 0

            self.left_stick_button.pressed = False
            self.right_stick_button.pressed = False
            self.bottom_button.pressed = False
            self.top_button.pressed = False
            self.left_button.pressed = False
            self.right_button.pressed = False
            self.left_shoulder_button.pressed = False
            self.right_shoulder_button.pressed = False
            self.select_button.pressed = False
            self.start_button.pressed = False
            self.bottom_pad.pressed = False
            self.top_pad.pressed = False
            self.left_pad.pressed = False
            self.right_pad.pressed = False

            self.left_trigger.pulled = 0
            self.right_trigger.pulled = 0
        self.zero_state_sent_counter = 0

    def get_state(self):
        """Returns the gamepad state dict, or None once the gamepad has been idle
        for a few frames (so a control loop knows to stop commanding motion)."""
        with self.lock:
            state = {
                "middle_led_ring_button_pressed": self.middle_led_ring_button.pressed,
                "left_stick_x": self.left_stick.x,
                "left_stick_y": self.left_stick.y,
                "right_stick_x": self.right_stick.x,
                "right_stick_y": self.right_stick.y,
                "left_stick_button_pressed": self.left_stick_button.pressed,
                "right_stick_button_pressed": self.right_stick_button.pressed,
                "bottom_button_pressed": self.bottom_button.pressed,
                "top_button_pressed": self.top_button.pressed,
                "left_button_pressed": self.left_button.pressed,
                "right_button_pressed": self.right_button.pressed,
                "left_shoulder_button_pressed": self.left_shoulder_button.pressed,
                "right_shoulder_button_pressed": self.right_shoulder_button.pressed,
                "select_button_pressed": self.select_button.pressed,
                "start_button_pressed": self.start_button.pressed,
                "left_trigger_pulled": self.left_trigger.pulled,
                "right_trigger_pulled": self.right_trigger.pulled,
                "bottom_pad_pressed": self.bottom_pad.pressed,
                "top_pad_pressed": self.top_pad.pressed,
                "left_pad_pressed": self.left_pad.pressed,
                "right_pad_pressed": self.right_pad.pressed,
            }

            # Check for activity
            is_active = False
            # 1. Recent events
            if time.monotonic() - self.last_event_ts < self.EVENT_ACTIVITY_TIMEOUT:
                is_active = True

            # 2. Holding state (buttons pressed, sticks moved, triggers pulled)
            if not is_active:
                if (
                    state["middle_led_ring_button_pressed"]
                    or state["left_stick_button_pressed"]
                    or state["right_stick_button_pressed"]
                    or state["bottom_button_pressed"]
                    or state["top_button_pressed"]
                    or state["left_button_pressed"]
                    or state["right_button_pressed"]
                    or state["left_shoulder_button_pressed"]
                    or state["right_shoulder_button_pressed"]
                    or state["select_button_pressed"]
                    or state["start_button_pressed"]
                    or state["bottom_pad_pressed"]
                    or state["top_pad_pressed"]
                    or state["left_pad_pressed"]
                    or state["right_pad_pressed"]
                ):
                    is_active = True
                elif (
                    abs(state["left_stick_x"]) > 1e-3
                    or abs(state["left_stick_y"]) > 1e-3
                    or abs(state["right_stick_x"]) > 1e-3
                    or abs(state["right_stick_y"]) > 1e-3
                ):
                    is_active = True
                elif state["left_trigger_pulled"] > 1e-3 or state["right_trigger_pulled"] > 1e-3:
                    is_active = True

            if is_active:
                self.zero_state_sent_counter = 0
                return state

            if self.zero_state_sent_counter < self.STOP_FRAME_COUNT:
                self.zero_state_sent_counter += 1
                return state

            return None


class ButtonPressCounter:
    """
    Provides an easy way to track button presses and holds.
    You can assign callback using `trigger_on_tap` and `trigger_on_hold` to perform an action
    when a button is tapped or held.
    Call `step()` in the main loop to make sure this works properly.
    """

    def __init__(self, button_name: str) -> None:
        self.button_name = button_name
        self.first_press_after_hold = 0.0
        self.last_hold_time = 0.0
        self.last_hold_duration = 0.0
        self.is_released = True
        self.was_released_last_step = False

        self.hold_triggered_cooldown_start_time = 0.0

    def _is_pressed(self, controller_state):
        return controller_state[self.button_name]

    @property
    def hold_duration(self):
        return self.last_hold_time - self.first_press_after_hold

    @property
    def _hold_triggered_elapsed(self):
        return self.last_hold_time - self.hold_triggered_cooldown_start_time

    def trigger_on_tap(self, callback: Callable, max_tap_duration: float = 1.0):
        """Calls the callback when the button is tapped."""
        if self.was_released_last_step and self.last_hold_duration < max_tap_duration:
            callback()

    def trigger_on_hold(self, hold_duration: float, callback: Callable):
        """Triggers when the user keeps the button held. If the button is continuously held,
        it will trigger again after the `hold_duration`"""
        if self.hold_duration >= hold_duration and (self._hold_triggered_elapsed > hold_duration):
            callback()
            self.hold_triggered_cooldown_start_time = time.time()

    def step(self, controller_state):
        """Call step in the main loop to keep track of user button presses."""

        self.was_released_last_step = False

        is_pressed = self._is_pressed(controller_state)

        if self.is_released and is_pressed:  # pressed for the first time
            self.is_released = False
            self.first_press_after_hold = time.time()
            self.last_hold_time = time.time()
        elif not self.is_released and is_pressed:  # holding down
            self.last_hold_time = time.time()
        elif not self.is_released and not is_pressed:  # let go
            self.is_released = True
            self.last_hold_duration = self.hold_duration
            self.first_press_after_hold = 0.0
            self.last_hold_time = 0.0
            self.was_released_last_step = True
            self.hold_triggered_cooldown_start_time = 0.0


class JointEffortTracker:
    """
    Provides an easy way to track joint efforts over time.
    Call step() in the main loop to track actuated joint efforts.
    """

    def __init__(
        self,
        joint_type: str,
        pos_thresholds: list[float],
        neg_thresholds: list[float] | None = None,
        joint_name: str | None = None,
    ) -> None:
        self.joint_type = joint_type
        self.joint_name = joint_name
        self.pos_thresholds = pos_thresholds
        self.neg_thresholds = neg_thresholds if neg_thresholds is not None else pos_thresholds

        self.first_exceed_time = 0.0
        self.last_exceed_time = 0.0
        self.is_below = True

        self.hold_triggered_cooldown_start_time = 0.0
        self.current_effort = 0.0
        self.last_direction = 0

    @property
    def hold_duration(self):
        return self.last_exceed_time - self.first_exceed_time

    @property
    def _hold_triggered_elapsed(self):
        return self.last_exceed_time - self.hold_triggered_cooldown_start_time

    def trigger_on_hold(self, hold_duration: float, callback: Callable):
        """Triggers when the effort exceeds the threshold continuously."""
        if self.hold_duration >= hold_duration and (self._hold_triggered_elapsed > hold_duration):
            callback(self.current_effort)
            self.hold_triggered_cooldown_start_time = time.time()

    def _reset(self):
        self.is_below = True
        self.first_exceed_time = 0.0
        self.last_exceed_time = 0.0
        self.hold_triggered_cooldown_start_time = 0.0

    def step(self, status, is_actuated, direction=0):
        self.current_effort = 0.0

        if not is_actuated or direction != self.last_direction:
            self._reset()
            self.last_direction = direction
            return

        self.current_effort = get_joint_effort(status, self.joint_type, self.joint_name)
        self.last_direction = direction
        thresholds = self.pos_thresholds if direction >= 0 else self.neg_thresholds

        is_exceeding = abs(self.current_effort) >= thresholds[0]

        if is_exceeding:
            self.last_exceed_time = time.time()
            if self.is_below:
                self.is_below = False
                self.first_exceed_time = time.time()
            return

        if self.hold_duration > 0.1:
            self._reset()


def get_joint_effort(status, joint_type, joint_name=None):
    """Sim equivalent of the robot-side `get_joint_effort()`: reads efforts off a
    `StatusStretchJoints` snapshot instead of off a `robot.Robot` instance."""
    try:
        if joint_type == "lift":
            return status.lift.effort
        elif joint_type == "arm":
            return status.arm.effort
        elif joint_type == "eoa":
            if joint_name == "wrist_yaw":
                return status.wrist_yaw.effort
            elif joint_name == "wrist_pitch":
                return status.wrist_pitch.effort
            elif joint_name == "wrist_roll":
                return status.wrist_roll.effort
            elif joint_name and "gripper" in joint_name:
                if hasattr(status, "gripper") and hasattr(status.gripper, "effort"):
                    return status.gripper.effort
                return status.gripper_left_finger.effort
        return 0.0
    except Exception:
        return 0.0


def main():
    import curses

    gamepad_controller = GamePadController(print_events=False)
    gamepad_controller.startup()

    def live_view(stdscr, controller):
        curses.curs_set(0)  # hide cursor
        prev_state = None
        controller.set_zero_state()  # allows for some zero state messages to return from get_state()
        while True:
            curr_state = controller.get_state()
            state = curr_state or prev_state
            stdscr.erase()
            stdscr.addstr(
                0,
                0,
                f"GAMEPAD CONTROLLER STATE {'- ACTIVE' if curr_state else '- INACTIVE'}",
                curses.A_BOLD,
            )
            for i, (k, v) in enumerate(state.items(), start=2):
                stdscr.addstr(i, 2, f"{k:30}: {v}")
            stdscr.refresh()
            time.sleep(0.05)
            prev_state = state

    try:
        while not gamepad_controller.is_gamepad_active:
            time.sleep(0.05)
        curses.wrapper(live_view, gamepad_controller)
    except (KeyboardInterrupt, SystemExit):
        print("Closing gamepad controller...")
        gamepad_controller.stop()


if __name__ == "__main__":
    main()
