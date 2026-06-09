from time import sleep
from pynput import keyboard
from pprint import pprint

import click

from examples.rerun_utils import init_pointcloud_viz, update_pointcloud_viz
from examples.camera_feeds import show_camera_feeds_sync
from examples.laser_scan import show_laser_scan
from stretch4_mujoco import StretchMujocoSimulator
from stretch4_mujoco.enums.actuators import Actuators
from stretch4_mujoco.enums.stretch_cameras import StretchCameras
from stretch4_mujoco.enums.stretch_sensors import StretchSensors
from stretch4_mujoco.stretch4_mujoco_simulator import Stretch4MujocoSimulator


def print_keyboard_options():
    click.secho("\n       Keyboard Controls:", fg="yellow")
    click.secho("=====================================", fg="yellow")
    print("W / A / S / D: Move BASE")
    print("Q / E: Rotate BASE")
    print("U / H / J / K: Move LIFT & ARM")
    print("O / P: Move WRIST YAW")
    print("C / V: Move WRIST PITCH")
    print("T / Y: Move WRIST ROLL")
    print("N / M: Open & Close GRIPPER")
    print("ctrl + (shift) + ): Enable keyboard input")
    print("ctrl + (shift) + (: Disable keyboard input")
    print("ctrl + (shift) + @: Increase base velocity")
    print("ctrl + (shift) + !: Decrease base velocity")
    print("L : Print status")
    print(". : Stop")
    click.secho("=====================================", fg="yellow")


class BaseController:
    is_forward: bool | None = None
    is_right: bool | None = None  # diff drive base ignores this direction
    is_clockwise: bool | None = None

    def __init__(self):
        self.forward_velocity = 0.7
        self.right_velocity = 0.7
        self.clockwise_velocity = 1.0

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

    def handle_base_velocity(self, sim: StretchMujocoSimulator):

        if isinstance(sim, Stretch4MujocoSimulator):
            sim.set_base_velocity(
                self.get_forward_velocity(),
                self.get_right_velocity(),
                self.get_clockwise_velocity(),
            )
        else:
            sim.set_base_velocity(self.get_forward_velocity(), self.get_clockwise_velocity())


class KeyboardController:
    """A container for keyboard state, such as if `ctrl_pressed` or `is_keyboard_control_active`."""

    def __init__(self, sim: StretchMujocoSimulator, use_stretch_3: bool):
        self.sim = sim
        self.base_controller = BaseController()
        self.use_stretch_3 = use_stretch_3

        # Allow multiple key-presses, references https://stackoverflow.com/a/74910695
        self.key_buffer = []

        self.is_keyboard_control_active = True  # when false, disable kbd commands

        self.ctrl_pressed = False

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

        if self.ctrl_pressed:
            if key == ")":
                self.enable_keyboard()
            elif key == "(":
                self.disable_keyboard()
            elif key == "@":
                self.base_controller.increase_base_velocity()
            elif key == "!":
                self.base_controller.decrease_base_velocity()

        if key == "w":
            self.base_controller.is_forward = True
        elif key == "s":
            self.base_controller.is_forward = False
        elif key == "a":
            self.base_controller.is_right = True
        elif key == "d":
            self.base_controller.is_right = False
        elif key == "e":
            self.base_controller.is_clockwise = False
        elif key == "q":
            self.base_controller.is_clockwise = True

        self.base_controller.handle_base_velocity(self.sim)

        if key == "u":
            self.sim.move_by(Actuators.lift, 0.1)
        elif key == "j":
            self.sim.move_by(Actuators.lift, -0.1)
        elif key == "h":
            self.sim.move_by(Actuators.arm, -0.05)
        elif key == "k":
            self.sim.move_by(Actuators.arm, 0.05)
        elif key == "o":
            self.sim.move_by(Actuators.wrist_yaw, -0.2)
        elif key == "p":
            self.sim.move_by(Actuators.wrist_yaw, 0.2)
        elif key == "c":
            self.sim.move_by(Actuators.wrist_pitch, 0.2)
        elif key == "v":
            self.sim.move_by(Actuators.wrist_pitch, -0.2)
        elif key == "t":
            self.sim.move_by(Actuators.wrist_roll, 0.2)
        elif key == "y":
            self.sim.move_by(Actuators.wrist_roll, -0.2)
        elif key == "n":
            if self.use_stretch_3:
                self.sim.move_by(Actuators.gripper, 0.07)
            else:
                self.sim.move_by(Actuators.gripper, 5.0)  # degrees (aperture angle)
        elif key == "m":
            if self.use_stretch_3:
                self.sim.move_by(Actuators.gripper, -0.07)
            else:
                self.sim.move_by(Actuators.gripper, -5.0)  # degrees (aperture angle)
        elif key == "l":
            pprint(self.sim.pull_status())
        elif key == ".":
            self.sim.stop()

    def keyboard_control_release(self, key: str | None):
        if key == "w":
            self.base_controller.is_forward = None
        elif key == "s":
            self.base_controller.is_forward = None
        elif key == "a":
            self.base_controller.is_right = None
        elif key == "d":
            self.base_controller.is_right = None
        elif key == "e":
            self.base_controller.is_clockwise = None
        elif key == "q":
            self.base_controller.is_clockwise = None

        self.base_controller.handle_base_velocity(self.sim)


@click.command()
@click.option("--scene-xml-path", type=str, default=None, help="Path to the scene xml file")
@click.option("--select_env", is_flag=True, help="Use robocasa environment")
@click.option("--imagery", is_flag=True, help="Show all the cameras' imagery")
@click.option("--lidar2d", is_flag=True, help="Show the lidar scan in Matplotlib")
@click.option("--lidar3d", is_flag=True, help="Show the point cloud in Rerun")
@click.option("--print-ratio", is_flag=True, help="Print the sim-to-real time ratio to the cli.")
@click.option("--use_stretch_3", type=bool, is_flag=True, help="Use Stretch 3")
def main(
    scene_xml_path: str | None,
    select_env: bool,
    imagery: bool,
    lidar2d: bool,
    lidar3d: bool,
    print_ratio: bool,
    use_stretch_3: bool,
):

    simulator_class = StretchMujocoSimulator if use_stretch_3 else Stretch4MujocoSimulator

    cameras_to_use = simulator_class.get_rgb_cameras() if imagery else []

    if lidar3d and not simulator_class is Stretch4MujocoSimulator:
        raise NotImplementedError("3D Lidar is only supported in Stretch4MujocoSimulator.")

    if lidar3d:
        cameras_to_use += StretchCameras.hemispherical_lidars()
        init_pointcloud_viz()

    use_imagery = len(cameras_to_use) > 0

    model = None

    if select_env:
        from stretch4_mujoco.robocasa_gen import model_generation_wizard

        model, xml, objects_info = model_generation_wizard(
            stretch_xml_absolute=simulator_class.get_robot_xml_path()
        )

    rate = 10.00 if lidar3d else 30.0
    # Lower camera hz for lidar3d to avoid performance issues

    sim = simulator_class(
        model=model,
        scene_xml_path=scene_xml_path,
        cameras_to_use=cameras_to_use,
        camera_hz=rate,
    )

    try:
        sim.start()

        print_keyboard_options()

        keyboard_controller = KeyboardController(sim=sim, use_stretch_3=use_stretch_3)

        listener = keyboard.Listener(
            on_press=keyboard_controller.on_press, on_release=keyboard_controller.on_release
        )
        listener.start()

        while sim.is_running():

            # rate limit loop
            elapsed = perf_counter()-last_loop
            if elapsed < loop_time:
                sleep(loop_time-elapsed)
            last_loop = perf_counter()
            
            if keyboard_controller.is_keyboard_control_active:
                for key in keyboard_controller.key_buffer:
                    if isinstance(key, keyboard.KeyCode):
                        keyboard_controller.keyboard_control(key.char)

            if print_ratio:
                print(f"{sim.pull_status().sim_to_real_time_ratio_msg}")

            if use_imagery:
                show_camera_feeds_sync(sim, False)

            if lidar3d:
                update_pointcloud_viz(
                    sim.pull_hemi_lidar_points(in_world_frame=True), "world/lidar_points"
                )

            if lidar2d:
                sensor_data = sim.pull_sensor_data()

                try:
                    show_laser_scan(
                        scan_data=sensor_data.get_data(StretchSensors.base_lidar),
                        is_se4=simulator_class is Stretch4MujocoSimulator,
                    )
                except:
                    ...

        listener.stop()

    except KeyboardInterrupt:
        sim.stop()


if __name__ == "__main__":
    main()
