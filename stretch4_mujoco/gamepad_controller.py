#!/usr/bin/env python3
from __future__ import print_function

import threading
import time

import click
from inputs import DeviceManager, GamepadLED, UnpluggedError

"""
This script is copy of the GamePadController class from the Stretch Body package.
https://github.com/hello-robot/stretch_body/blob/master/body/stretch_body/gamepad_controller.py

The GamePadController is a threading class that polls for the gamepad inputs (gamepad_state) by
listening to the gamepad's USB dongle plugged into the robot.
"""


class Stick:
    def __init__(self):
        # joystick pushed
        #   all the way down: y = -1.0
        #   all the way up: y ~= 1.0
        #   all the way left: x = -1.0
        #   all the way right: x ~= 1.0
        self.x = 0.0
        self.y = 0.0
        # normalized signed 16 bit integers to be in the range [-1.0, 1.0]
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
        if state == 0:
            self.pressed = False
        elif state == 1:
            self.pressed = True

    def print_string(self):
        return str(self.pressed)


class Trigger:
    def __init__(self, xbox_one=False):
        # Xbox One trigger
        #   not pulled = 0
        #   max pulled = 1023
        # normalize unsigned 10 bit integer to be in the range [0.0, 1.0]

        # Xbox 360 trigger
        #   not pulled = 0
        #   max pulled = 255
        # normalize unsigned 8 bit integer to be in the range [0.0, 1.0]
        if xbox_one:
            # xbox one
            num_bits = 10
        else:
            # xbox 360
            num_bits = 8
        self.norm = float(pow(2, num_bits) - 1)
        self.pulled = 0.0

    def update(self, state):
        self.pulled = int(state) / self.norm
        # Ensure that the pulled value is not greater than 1.0, which
        # will can happen with the use of an Xbox One controller, if
        # the option was not properly set.
        if self.pulled > 1.0:
            self.pulled = 1.0

    def print_string(self):
        return "{0:4.2f}".format(self.pulled)


class GamePadDevice(DeviceManager):
    def _parse_led_path(self, path):
        name = path.rsplit("/", 1)[1]
        if name.startswith("xpad"):
            self.leds.append(GamepadLED(self, path, name))


class GamePadController(threading.Thread):
    """Successfully tested with the following controllers:
         + Xbox One Controller connected using a USB cable (change xbox_one parameter to True for full 10 bit
         trigger information)
         + EasySMX wireless controller set to appropriate mode (Xbox 360 mode with upper half of ring LED illuminated -
         top two LED quarter circle arcs)
         + JAMSWALL Xbox 360 Wireless Controller (Sometimes issues would occur after inactivity that would seem to
         require unplugging and replugging the USB dongle.)

    Unsuccessful tests:
         - Xbox One Controller connected via Bluetooth
         - Xbox 360 Controller connected with an Insten Wireless Controller USB Charging Cable
         +/- VOYEE Wired Xbox 360 Controller mostly worked, but it had various issues including false
        middle LED button presses, phantom shoulder button presses, and low joystick sensitivity that made
        small motions more difficult to execute.
    """

    def __init__(self, print_events=False, print_dongle_status=True):
        threading.Thread.__init__(self, name=self.__class__.__name__)
        self.print_events = print_events
        self.devices = GamePadDevice()
        self.is_gamepad_dongle = False
        self._i = 0
        self.print_dongle_status = print_dongle_status

        self.left_stick = Stick()
        self.right_stick = Stick()

        self.ros_logger = None

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

        self.left_trigger = Trigger(xbox_one=False)
        self.right_trigger = Trigger(xbox_one=False)

        self.left_pad = Button()
        self.right_pad = Button()
        self.top_pad = Button()
        self.bottom_pad = Button()

        self.lock = threading.Lock()
        # self.thread = threading.Thread(target=self.update,name="GamepadEvents_thread")
        self.daemon = True
        self.stop_thread = False
        self.shutdown_flag = threading.Event()

        self.set_zero_state()
        self.gamepad_state = self.get_state()

    def run(self):
        while not self.shutdown_flag.is_set():
            if not self.shutdown_flag.is_set():
                self.update()

    def get_gamepad(self):
        """Get a single action from a gamepad."""
        try:
            gamepad = self.devices.gamepads[0]
            return gamepad.read()
        except Exception:
            raise UnpluggedError("No gamepad found.")

    # def start(self):
    #     self.stop_thread = False
    # self.thread.start()

    def stop(self):
        if not self.stop_thread:
            with self.lock:
                self.stop_thread = True
            # self.thread.join() # Thread._wait_for_tstate_lock() never returns if trying to join this thread

    def poll_till_gamepad_dongle_present(self):
        # self.is_gamepad_dongle = False
        # while not self.is_gamepad_dongle:
        with self.lock:
            self.is_gamepad_dongle = False
        if self._i % 50 == 0 and self.print_dongle_status:
            click.secho("Waiting for Gamepad Dongle.................", fg="yellow")
        try:
            self.devices.__init__()
            if len(self.devices.gamepads) > 0:
                click.secho("Gamepad Dongle FOUND!", fg="green", bold=True)
                with self.lock:
                    self.is_gamepad_dongle = True
        except Exception:
            pass

    def update(self):
        while not self.stop_thread:
            self._i = self._i + 1
            if len(self.devices.gamepads) > 0:
                self.is_gamepad_dongle = True
                try:
                    events = self.get_gamepad()
                    self.update_button_encodings(events)
                except (OSError, UnpluggedError, Exception):
                    click.secho("Gamepad Dongle DISCONNECTED........", fg="red", bold=True)
                    self.poll_till_gamepad_dongle_present()
            else:
                self.is_gamepad_dongle = False
                self.poll_till_gamepad_dongle_present()
            if not self.is_gamepad_dongle:
                if self._i % 2 == 0:
                    self.set_zero_state()
            self.gamepad_state = self.get_state()

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
                if event.code == "BTN_WEST":  # yellow Y, ***top button*** WEIRD!
                    self.top_button.update(event.state)
                if event.code == "BTN_NORTH":  # blue X, ***left button*** WEIRD!
                    self.left_button.update(event.state)
                if event.code == "BTN_EAST":  # red B, right button
                    self.right_button.update(event.state)

                if event.code == "BTN_TL":  # left shoulder button
                    self.left_shoulder_button.update(event.state)
                if event.code == "BTN_TR":  # right shoulder button
                    self.right_shoulder_button.update(event.state)

                if event.code == "ABS_Z":  # left trigger 0-1023
                    self.left_trigger.update(event.state)
                if event.code == "ABS_RZ":  # right trigger 0-1023
                    self.right_trigger.update(event.state)

                if event.code == "BTN_SELECT":  # 1/0
                    self.select_button.update(event.state)
                if event.code == "BTN_START":  # 1/0
                    self.start_button.update(event.state)

                if event.code == "BTN_THUMBL":  # 1/0
                    self.left_stick_button.update(event.state)
                if event.code == "BTN_THUMBR":  # 1/0
                    self.right_stick_button.update(event.state)

                # 4-way pad
                if event.code == "ABS_HAT0Y":  # -1 up / 1 down
                    if event.state == 0:
                        self.top_pad.update(0)
                        self.bottom_pad.update(0)
                    elif event.state == 1:
                        self.top_pad.update(0)
                        self.bottom_pad.update(1)
                    elif event.state == -1:
                        self.bottom_pad.update(0)
                        self.top_pad.update(1)

                if event.code == "ABS_HAT0X":  # -1 left / 1 right
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

    def get_state(self):
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
        return state

    def vibrate(self, duration_ms: int = 150, strong_magnitude: float = 1.0, weak_magnitude: float = 1.0, tag: str = None, cooldown: float = 0.0):
        if not hasattr(self, '_last_vibrated_tags'):
            self._last_vibrated_tags = {}
        if tag is not None and cooldown > 0.0:
            now = time.time()
            if now - self._last_vibrated_tags.get(tag, 0.0) < cooldown:
                return
            self._last_vibrated_tags[tag] = now

        def _vib():
            try:
                import evdev
                from evdev import ecodes, ff
                if not hasattr(self, '_evdev_dev') or self._evdev_dev is None:
                    import glob
                    dev_path = None
                    for path in glob.glob("/dev/input/event*"):
                        try:
                            d = evdev.InputDevice(path)
                            caps = d.capabilities()
                            if ecodes.EV_FF in caps and ecodes.EV_ABS in caps:
                                dev_path = path
                                break
                            d.close()
                        except:
                            pass
                    if dev_path:
                        self._evdev_dev = evdev.InputDevice(dev_path)
                    else:
                        return
                
                strong = max(0, min(int(strong_magnitude * 65535), 65535))
                weak = max(0, min(int(weak_magnitude * 65535), 65535))
                rumble = ff.Rumble(strong_magnitude=strong, weak_magnitude=weak)
                effect_type = ff.EffectType(ff_rumble_effect=rumble)
                effect = ff.Effect(
                    ecodes.FF_RUMBLE, -1, 0,
                    ff.Trigger(0, 0),
                    ff.Replay(int(duration_ms), 0),
                    effect_type
                )
                effect_id = self._evdev_dev.upload_effect(effect)
                self._evdev_dev.write(ecodes.EV_FF, effect_id, 1)
            except Exception:
                self._evdev_dev = None
        threading.Thread(target=_vib, daemon=True).start()

    def vibrate_sequence(self, sequence_ms: list, strong_magnitude: float = 1.0, weak_magnitude: float = 1.0, tag: str = None, cooldown: float = 0.0):
        if not hasattr(self, '_last_vibrated_tags'):
            self._last_vibrated_tags = {}
        if tag is not None and cooldown > 0.0:
            now = time.time()
            if now - self._last_vibrated_tags.get(tag, 0.0) < cooldown:
                return
            self._last_vibrated_tags[tag] = now
            
        def _vibrate_sequence_thread():
            for i, duration in enumerate(sequence_ms):
                if i % 2 == 0:
                    self.vibrate(duration_ms=duration, strong_magnitude=strong_magnitude, weak_magnitude=weak_magnitude)
                time.sleep(duration / 1000.0)
                
        threading.Thread(target=_vibrate_sequence_thread, daemon=True).start()

class ButtonPressCounter:
    """
    Provides an easy way to track button presses and holds.
    You can assign callback using `trigger_on_tap` and `trigger_on_hold` to perform an action when a button is tapped or held.
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
        return controller_state.get(self.button_name, False)
    
    @property
    def hold_duration(self):
        return self.last_hold_time - self.first_press_after_hold
    
    @property
    def _hold_triggered_elapsed(self):
        return self.last_hold_time - self.hold_triggered_cooldown_start_time
    
    def trigger_on_tap(self, callback, max_tap_duration:float = 1.0):
        """Calls the callback when the button is tapped."""
        if self.was_released_last_step and self.last_hold_duration < max_tap_duration:
            callback()
    
    def trigger_on_hold(self, hold_duration:float, callback):
        """Triggers when the user keeps the button held. If the button is continously held, it will trigger again after the `hold_duration`"""
        if self.hold_duration >= hold_duration and (self._hold_triggered_elapsed > hold_duration):
            callback()
            self.hold_triggered_cooldown_start_time = time.time()

    def step(self, controller_state):
        """Call step in the main loop to keep track of user button presses."""

        self.was_released_last_step = False

        is_pressed = self._is_pressed(controller_state)

        if self.is_released and is_pressed: # pressed for the first time
            self.is_released = False
            self.first_press_after_hold = time.time()
            self.last_hold_time = time.time()
        elif not self.is_released and is_pressed: # holding down
            self.last_hold_time = time.time()
        elif not self.is_released and not is_pressed: # let go
            self.is_released = True
            self.last_hold_duration = self.hold_duration
            self.first_press_after_hold = 0.0
            self.last_hold_time = 0.0
            self.was_released_last_step = True
            self.hold_triggered_cooldown_start_time = 0.0

def get_joint_effort(status, joint_type, joint_name=None):
    try:
        if joint_type == 'lift':
            return status.lift.effort
        elif joint_type == 'arm':
            return status.arm.effort
        elif joint_type == 'eoa':
            if joint_name == 'wrist_yaw':
                return status.wrist_yaw.effort
            elif joint_name == 'wrist_pitch':
                return status.wrist_pitch.effort
            elif joint_name == 'wrist_roll':
                return status.wrist_roll.effort
            elif 'gripper' in joint_name:
                if hasattr(status, 'gripper') and hasattr(status.gripper, 'effort'):
                    return status.gripper.effort
                else:
                    return status.gripper_left_finger.effort
        return 0.0
    except Exception:
        return 0.0

class JointEffortTracker:
    def __init__(self, joint_type: str, pos_thresholds: list[float], neg_thresholds: list[float] = None, joint_name: str = None) -> None:
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
        
    def trigger_on_hold(self, hold_duration: float, callback):
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
        is_exceeding = False
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


def main():
    gamepad_controller = GamePadController(print_events=False)
    gamepad_controller.start()
    try:
        while True:
            state = gamepad_controller.get_state()
            print("------------------------------")
            print("GAMEPAD CONTROLLER STATE")
            for k in state.keys():
                print(k, " : ", state[k])
            print("------------------------------")
            time.sleep(0.1)
    except (KeyboardInterrupt, SystemExit):
        pass


if __name__ == "__main__":
    main()
