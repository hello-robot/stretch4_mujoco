#!/usr/bin/env python3
"""
Port of the real robot's `stretch4_body/core/gamepad_controller.py`.

The `GamePadController` is a threading class that polls for gamepad events by
listening to the gamepad's USB dongle, processes them, and makes an easy to
consume gamepad state available as a dictionary - `GamePadController.get_state()`.

Kept deliberately close to the robot-side file so the two can be diffed. The
removals are the `stretch4_body` dependencies (`Device`, `LoopStats` and the
`/tmp` teleop singleton lock), which have no meaning in simulation.

The other departure is the input backend. The robot reads the pad straight off
`/dev/input` with evdev, which only exists on Linux, while the sim is expected
to run on Linux, macOS and Windows too. This file therefore reads the pad
through pygame/SDL2 and re-synthesises the evdev-style events it would have
seen (`ABS_X`, `BTN_SOUTH`, ...), so `update_button_encodings()` and everything
downstream of it stays the robot's code.
"""

import logging
import math
import os
import threading
import time
from dataclasses import dataclass
from typing import Callable

# SDL only runs its event pump - which is also what refreshes joystick state -
# once a video subsystem exists, but the sim's window belongs to MuJoCo/GLFW, so
# ask SDL for the headless "dummy" driver rather than have it open a window of
# its own. That also keeps the pump off macOS' Cocoa backend, which insists on
# being pumped from the main thread; `_poll_thread` below is not it.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
# Without an SDL window there is nothing to hold keyboard/controller focus, so
# ask SDL to keep delivering pad events regardless of which window has it.
os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame

try:  # pragma: no cover - depends on how pygame was built
    from pygame._sdl2 import controller as sdl2_controller
except ImportError:  # pragma: no cover
    sdl2_controller = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# --- SDL2 constants ---------------------------------------------------------
# SDL_GameControllerButton / SDL_GameControllerAxis values. Spelled out because
# pygame only re-exports them as `pygame.CONTROLLER_*` on some builds; they are
# part of SDL2's stable ABI, so hardcoding them is safe.
SDL_BUTTON_A = 0
SDL_BUTTON_B = 1
SDL_BUTTON_X = 2
SDL_BUTTON_Y = 3
SDL_BUTTON_BACK = 4
SDL_BUTTON_GUIDE = 5
SDL_BUTTON_START = 6
SDL_BUTTON_LEFTSTICK = 7
SDL_BUTTON_RIGHTSTICK = 8
SDL_BUTTON_LEFTSHOULDER = 9
SDL_BUTTON_RIGHTSHOULDER = 10
SDL_BUTTON_DPAD_UP = 11
SDL_BUTTON_DPAD_DOWN = 12
SDL_BUTTON_DPAD_LEFT = 13
SDL_BUTTON_DPAD_RIGHT = 14

SDL_AXIS_LEFTX = 0
SDL_AXIS_LEFTY = 1
SDL_AXIS_RIGHTX = 2
SDL_AXIS_RIGHTY = 3
SDL_AXIS_TRIGGERLEFT = 4
SDL_AXIS_TRIGGERRIGHT = 5

# --- Event synthesis tuning -------------------------------------------------
# Scales the normalized [-1, 1] SDL axis onto the 16-bit range `Stick` divides
# by (`Stick.norm == 2**15`), i.e. the range evdev reported.
STICK_EVENT_SCALE = 32767
# Likewise for `Trigger`, whose 8-bit `norm` is what an evdev pad reported.
TRIGGER_EVENT_SCALE = 255
# evdev pads come out of a kernel driver that has already been told the stick's
# resting slop; SDL hands over the raw HID value, which on a used pad can sit a
# couple of percent off centre forever. The joint commands downstream use a
# dead zone of 1e-4, so without one here the base would creep on its own.
STICK_DEADZONE = 0.06
# Quantize axes before diffing them, so resting jitter doesn't synthesise a
# stream of events (which `get_state()` would read as "the pad is being used").
AXIS_QUANTUM = 0.01


def ensure_pygame_ready() -> None:
    """Bring up the pygame subsystems the gamepad needs, once per process."""
    if not pygame.display.get_init():
        # Needed for `pygame.event`; see the SDL_VIDEODRIVER note above.
        pygame.display.init()
    if not pygame.joystick.get_init():
        pygame.joystick.init()
    if sdl2_controller is not None and not sdl2_controller.get_init():
        sdl2_controller.init()


def _normalize_axis(raw: float) -> float:
    """Put an axis sample on [-1, 1].

    `pygame.joystick.Joystick.get_axis()` already returns a float there, while
    the SDL game-controller API returns the raw signed 16-bit int, so which one
    this is has to be told from the sample's *type*: the two ranges overlap at
    -1, 0 and 1, and a stick resting one count off centre reports exactly that.
    Reading a raw 1 as a normalized 1.0 would peg the axis at full deflection
    for as long as it rested there.
    """
    value = float(raw)
    if isinstance(raw, int):
        value /= 32767.0
    return max(-1.0, min(1.0, value))


def is_probable_gamepad(joystick: "pygame.joystick.Joystick") -> bool:
    """Heuristic for "this is a gamepad, not a wheel/throttle/3D mouse"."""
    # Two sticks' worth of axes and at least the four face buttons plus shoulders.
    return joystick.get_numaxes() >= 4 and joystick.get_numbuttons() >= 6


def find_first_gamepad() -> int | None:
    """Return the pygame joystick index of the first attached gamepad, or None.

    Pads that SDL recognises - i.e. that have an Xbox-style entry in its game
    controller database - are preferred, since for those SDL normalises the
    axis/button numbering for us instead of us guessing at the driver's layout.
    """
    ensure_pygame_ready()
    indices = list(range(pygame.joystick.get_count()))

    if sdl2_controller is not None:
        for index in indices:
            try:
                if sdl2_controller.is_controller(index):
                    return index
            except Exception:
                pass

    for index in indices:
        try:
            joystick = pygame.joystick.Joystick(index)
            ok = is_probable_gamepad(joystick)
            joystick.quit()
            if ok:
                return index
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


class GamepadSource:
    """Reads one pad through pygame and reports it as evdev-style `GPEvent`s.

    Subclasses supply `read()`, a snapshot in gamepad terms; this class turns
    the snapshot into the codes and integer ranges `update_button_encodings()`
    expects, emitting an event only where a value actually changed - which is
    what evdev did, and what `GamePadController`'s activity timeout assumes.
    """

    BUTTON_CODES = {
        # The Linux names are rotated relative to the Xbox labels: input-event-codes.h
        # aliases BTN_NORTH to BTN_X and BTN_WEST to BTN_Y, and the mapping below
        # keeps `update_button_encodings()`'s comments true.
        "a": "BTN_SOUTH",
        "b": "BTN_EAST",
        "x": "BTN_NORTH",
        "y": "BTN_WEST",
        "lb": "BTN_TL",
        "rb": "BTN_TR",
        "back": "BTN_SELECT",
        "start": "BTN_START",
        "guide": "BTN_MODE",
        "left_stick": "BTN_THUMBL",
        "right_stick": "BTN_THUMBR",
    }
    STICK_CODES = {
        "left_x": "ABS_X",
        "left_y": "ABS_Y",
        "right_x": "ABS_RX",
        "right_y": "ABS_RY",
    }
    TRIGGER_CODES = {"left_trigger": "ABS_Z", "right_trigger": "ABS_RZ"}
    # `dpad_y` is reported the evdev way round, +1 meaning down.
    DPAD_CODES = {"dpad_x": "ABS_HAT0X", "dpad_y": "ABS_HAT0Y"}

    def __init__(self, name: str, instance_id: int, stick_deadzone: float = STICK_DEADZONE):
        self.name = name
        self.instance_id = instance_id
        self.stick_deadzone = stick_deadzone
        self._last: dict[str, int] = {}

    # -- to be provided by the backend --
    def read(self) -> dict[str, float]:
        raise NotImplementedError

    def rumble(self, low_frequency: float, high_frequency: float, duration_ms: int) -> bool:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    # -- snapshot -> events --
    def _apply_deadzone(self, value: float) -> float:
        if abs(value) <= self.stick_deadzone:
            return 0.0
        # Rescale the live part of the travel back onto [-1, 1] so the dead zone
        # doesn't introduce a step at its edge.
        scaled = (abs(value) - self.stick_deadzone) / (1.0 - self.stick_deadzone)
        return math.copysign(min(scaled, 1.0), value)

    def _diff(self, events: list[GPEvent], code: str, state: int, ev_type: str) -> None:
        if self._last.get(code) == state:
            return
        self._last[code] = state
        events.append(GPEvent(code=code, state=state, ev_type=ev_type))

    def poll(self) -> list[GPEvent]:
        snapshot = self.read()
        events: list[GPEvent] = []

        for key, code in self.STICK_CODES.items():
            value = self._apply_deadzone(snapshot[key])
            state = int(round(round(value / AXIS_QUANTUM) * AXIS_QUANTUM * STICK_EVENT_SCALE))
            self._diff(events, code, state, "EV_ABS")

        for key, code in self.TRIGGER_CODES.items():
            value = max(0.0, min(1.0, snapshot[key]))
            state = int(round(round(value / AXIS_QUANTUM) * AXIS_QUANTUM * TRIGGER_EVENT_SCALE))
            self._diff(events, code, state, "EV_ABS")

        for key, code in self.DPAD_CODES.items():
            self._diff(events, code, int(snapshot[key]), "EV_ABS")

        for key, code in self.BUTTON_CODES.items():
            self._diff(events, code, int(bool(snapshot[key])), "EV_KEY")

        return events


class ControllerSource(GamepadSource):
    """Backend for a pad SDL has a game-controller mapping for.

    This is the good case: SDL has already normalised the pad onto the Xbox
    layout, so the axis and button numbers below mean the same thing on every
    platform and for every pad in its database.
    """

    def __init__(self, index: int, stick_deadzone: float = STICK_DEADZONE):
        assert sdl2_controller is not None
        self._controller = sdl2_controller.Controller(index)
        joystick = pygame.joystick.Joystick(index)
        super().__init__(
            name=joystick.get_name(),
            instance_id=joystick.get_instance_id(),
            stick_deadzone=stick_deadzone,
        )
        self._joystick = joystick

    def read(self) -> dict[str, float]:
        controller = self._controller

        def axis(which: int) -> float:
            return _normalize_axis(controller.get_axis(which))

        def trigger(which: int) -> float:
            # SDL reports triggers over the positive half of the axis only.
            return max(0.0, axis(which))

        def button(which: int) -> bool:
            return bool(controller.get_button(which))

        return {
            "left_x": axis(SDL_AXIS_LEFTX),
            "left_y": axis(SDL_AXIS_LEFTY),
            "right_x": axis(SDL_AXIS_RIGHTX),
            "right_y": axis(SDL_AXIS_RIGHTY),
            "left_trigger": trigger(SDL_AXIS_TRIGGERLEFT),
            "right_trigger": trigger(SDL_AXIS_TRIGGERRIGHT),
            "a": button(SDL_BUTTON_A),
            "b": button(SDL_BUTTON_B),
            "x": button(SDL_BUTTON_X),
            "y": button(SDL_BUTTON_Y),
            "lb": button(SDL_BUTTON_LEFTSHOULDER),
            "rb": button(SDL_BUTTON_RIGHTSHOULDER),
            "back": button(SDL_BUTTON_BACK),
            "start": button(SDL_BUTTON_START),
            "guide": button(SDL_BUTTON_GUIDE),
            "left_stick": button(SDL_BUTTON_LEFTSTICK),
            "right_stick": button(SDL_BUTTON_RIGHTSTICK),
            # The controller API exposes the d-pad as four buttons rather than a hat.
            "dpad_x": int(button(SDL_BUTTON_DPAD_RIGHT)) - int(button(SDL_BUTTON_DPAD_LEFT)),
            "dpad_y": int(button(SDL_BUTTON_DPAD_DOWN)) - int(button(SDL_BUTTON_DPAD_UP)),
        }

    def rumble(self, low_frequency: float, high_frequency: float, duration_ms: int) -> bool:
        for device in (self._controller, self._joystick):
            rumble = getattr(device, "rumble", None)
            if rumble is None:
                continue
            try:
                if rumble(low_frequency, high_frequency, duration_ms):
                    return True
            except Exception:
                pass
        return False

    def close(self) -> None:
        for device in (self._controller, self._joystick):
            try:
                device.quit()
            except Exception:
                pass


class JoystickSource(GamepadSource):
    """Fallback backend for a pad SDL has no game-controller mapping for.

    The axis/button numbers are then whatever the platform's driver chose, so
    all this can do is assume the near-universal XInput ordering. Preferring
    `ControllerSource` keeps this to genuinely unknown hardware.
    """

    AXES = {
        "left_x": 0,
        "left_y": 1,
        "left_trigger": 2,
        "right_x": 3,
        "right_y": 4,
        "right_trigger": 5,
    }
    BUTTONS = {
        "a": 0,
        "b": 1,
        "x": 2,
        "y": 3,
        "lb": 4,
        "rb": 5,
        "back": 6,
        "start": 7,
        "guide": 8,
        "left_stick": 9,
        "right_stick": 10,
    }

    def __init__(self, index: int, stick_deadzone: float = STICK_DEADZONE):
        joystick = pygame.joystick.Joystick(index)
        joystick.init()
        super().__init__(
            name=joystick.get_name(),
            instance_id=joystick.get_instance_id(),
            stick_deadzone=stick_deadzone,
        )
        self._joystick = joystick
        # Some drivers rest a trigger at -1 and others at 0; see `_trigger()`.
        self._bipolar_triggers: dict[int, bool] = {}

    def _axis(self, which: int) -> float:
        if which >= self._joystick.get_numaxes():
            return 0.0
        return _normalize_axis(self._joystick.get_axis(which))

    def _trigger(self, which: int) -> float:
        value = self._axis(which)
        # A trigger that has ever reported a strongly negative value must span
        # [-1, 1] rather than [0, 1]. Latching it this way means a pad SDL has
        # not yet sampled (all axes read 0) corrects itself on the first pull
        # instead of sitting at half-pulled.
        if value <= -0.5:
            self._bipolar_triggers[which] = True
        if self._bipolar_triggers.get(which):
            return (value + 1.0) / 2.0
        return max(0.0, value)

    def _button(self, which: int) -> bool:
        if which >= self._joystick.get_numbuttons():
            return False
        return bool(self._joystick.get_button(which))

    def read(self) -> dict[str, float]:
        hat_x, hat_y = self._joystick.get_hat(0) if self._joystick.get_numhats() else (0, 0)

        snapshot: dict[str, float] = {
            "left_x": self._axis(self.AXES["left_x"]),
            "left_y": self._axis(self.AXES["left_y"]),
            "right_x": self._axis(self.AXES["right_x"]),
            "right_y": self._axis(self.AXES["right_y"]),
            "left_trigger": self._trigger(self.AXES["left_trigger"]),
            "right_trigger": self._trigger(self.AXES["right_trigger"]),
            "dpad_x": hat_x,
            # SDL's hat has +1 up; evdev's ABS_HAT0Y has +1 down.
            "dpad_y": -hat_y,
        }
        for key, index in self.BUTTONS.items():
            snapshot[key] = self._button(index)
        return snapshot

    def rumble(self, low_frequency: float, high_frequency: float, duration_ms: int) -> bool:
        rumble = getattr(self._joystick, "rumble", None)
        if rumble is None:
            return False
        try:
            return bool(rumble(low_frequency, high_frequency, duration_ms))
        except Exception:
            return False

    def close(self) -> None:
        try:
            self._joystick.quit()
        except Exception:
            pass


def open_gamepad(index: int, stick_deadzone: float = STICK_DEADZONE) -> GamepadSource:
    """Open a pad by pygame joystick index, preferring SDL's controller mapping."""
    if sdl2_controller is not None:
        try:
            if sdl2_controller.is_controller(index):
                return ControllerSource(index, stick_deadzone=stick_deadzone)
        except Exception:
            pass
    return JoystickSource(index, stick_deadzone=stick_deadzone)


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

    Reads the pad through pygame/SDL2, so it works on Linux, macOS and Windows.
    Any pad in SDL's game controller database (every Xbox/PlayStation-style pad,
    wired or wireless) is picked up with the correct button layout; anything
    else falls back to the XInput layout, see `JoystickSource`.
    """

    def __init__(self, print_events=False, is_xbox_one=False):
        self.name = "gamepad_controller"
        self.print_events = print_events

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

        # `is_xbox_one` is accepted for parity with the robot-side signature but no
        # longer selects a trigger resolution: SDL hands over a normalized trigger,
        # which `GamepadSource` re-scales onto the 8-bit range regardless of pad.
        self.left_trigger = Trigger(xbox_one=False)
        self.right_trigger = Trigger(xbox_one=False)

        self.left_pad = Button()
        self.right_pad = Button()
        self.top_pad = Button()
        self.bottom_pad = Button()

        self.lock = threading.Lock()

        self.device_path = None  # the pad's SDL name, once one is found
        self.dev: GamepadSource | None = None
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
        # SDL wants its video subsystem brought up on the process' main thread on
        # macOS, so do it here rather than from `_thread_target`.
        ensure_pygame_ready()

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
            try:
                self.update()
            except Exception:
                # Letting this thread die would freeze the pad state at whatever
                # it last read, and a state frozen mid-deflection reads as a
                # stick held down forever - i.e. the robot would keep moving.
                logger.exception("Gamepad polling failed; zeroing the gamepad state.")
                self.set_zero_state()
            elapsed = time.perf_counter() - start
            if not self.thread_shutdown_flag.is_set() and elapsed < period:
                time.sleep(period - elapsed)

    def stop(self):
        self.thread_shutdown_flag.set()
        if self.thread is not None:
            self.thread.join(1)
            self.thread = None
        if self.dev is not None:
            self.dev.close()
            self.dev = None

    def _pump_sdl_events(self) -> None:
        """Run SDL's event pump, which is what refreshes the pad's state.

        The queue is drained rather than just pumped (`pygame.event.pump()`)
        because nothing else in the sim consumes SDL events, and an undrained
        queue eventually fills up and starts dropping the hotplug events below.
        """
        for event in pygame.event.get():
            if event.type == pygame.JOYDEVICEREMOVED and self.dev is not None:
                if getattr(event, "instance_id", None) == self.dev.instance_id:
                    raise UnpluggedError("Gamepad disconnected.")

    def poll_till_gamepad_dongle_present(self):
        with self.lock:
            self.is_gamepad_active = False
        try:
            self._pump_sdl_events()  # picks up hotplugged pads
            index = find_first_gamepad()
            if index is not None:
                self.dev = open_gamepad(index)
                self.device_path = self.dev.name
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
        self._pump_sdl_events()
        try:
            return self.dev.poll()
        except pygame.error as e:
            # device likely went away
            raise UnpluggedError("Gamepad disconnected.") from e

    def update(self):
        if not self.is_gamepad_active:
            self.poll_till_gamepad_dongle_present()
            return

        try:
            events = self.get_gamepad_events()
            if events:
                self.last_event_ts = time.monotonic()
            self.update_button_encodings(events)
        except (UnpluggedError, OSError, pygame.error):
            logger.error("Gamepad Dongle DISCONNECTED...")
            if self.dev is not None:
                self.dev.close()
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
            # SDL's low/high frequency motors are the strong/weak ones respectively.
            self.dev.rumble(
                max(0.0, min(strong_magnitude, 1.0)),
                max(0.0, min(weak_magnitude, 1.0)),
                int(duration_ms),
            )

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

                if event.code == "BTN_SOUTH":  # green A, bottom button
                    self.bottom_button.update(event.state)
                if event.code == "BTN_WEST":  # yellow Y, top button
                    self.top_button.update(event.state)
                if event.code == "BTN_NORTH":  # blue X, left button
                    self.left_button.update(event.state)
                if event.code == "BTN_EAST":  # red B, right button
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
    gamepad_controller = GamePadController(print_events=False)
    gamepad_controller.startup()

    # A plain redraw rather than curses, which does not exist on Windows.
    prev_state = None
    try:
        while not gamepad_controller.is_gamepad_active:
            time.sleep(0.05)
        gamepad_controller.set_zero_state()  # so get_state() returns some zero states
        while True:
            curr_state = gamepad_controller.get_state()
            state = curr_state or prev_state or {}
            lines = [f"GAMEPAD CONTROLLER STATE {'- ACTIVE' if curr_state else '- INACTIVE'}", ""]
            lines += [f"  {k:30}: {v}" for k, v in state.items()]
            print("\033[H\033[J" + "\n".join(lines), end="\r\n", flush=True)
            time.sleep(0.05)
            prev_state = state
    except (KeyboardInterrupt, SystemExit):
        print("Closing gamepad controller...")
        gamepad_controller.stop()


if __name__ == "__main__":
    main()
